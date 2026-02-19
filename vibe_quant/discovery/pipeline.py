"""Genetic discovery pipeline for trading strategy evolution.

Orchestrates population initialization, fitness evaluation, selection,
crossover, mutation, and convergence detection to discover profitable
strategy candidates expressed as DSL YAML dicts.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vibe_quant.discovery.fitness import FitnessResult, evaluate_population
from vibe_quant.discovery.genome import chromosome_to_dsl
from vibe_quant.discovery.operators import (
    StrategyChromosome,
    _random_chromosome,
    apply_elitism,
    crossover,
    initialize_population,
    is_valid_chromosome,
    mutate,
    tournament_select,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Max retries when generating valid offspring via crossover+mutation
_MAX_OFFSPRING_RETRIES: int = 10


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DiscoveryConfig:
    """Configuration for the genetic discovery pipeline.

    Attributes:
        population_size: Number of individuals per generation.
        max_generations: Maximum evolutionary generations.
        mutation_rate: Per-gene mutation probability.
        crossover_rate: Probability of crossover per pair (0-1). If < 1, parents
            are copied directly with probability (1 - crossover_rate).
        elite_count: Number of top individuals preserved unchanged.
        tournament_size: Tournament selection pool size.
        convergence_generations: Stop after N generations with no improvement.
        top_k: Number of top strategies to output.
        min_trades: Minimum trades for a strategy to be considered valid.
        symbols: Trading symbols to evaluate on.
        timeframe: Bar timeframe (e.g. "1h").
        start_date: Backtest start date (ISO format).
        end_date: Backtest end date (ISO format).
    """

    population_size: int = 50
    max_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_count: int = 2
    tournament_size: int = 3
    convergence_generations: int = 10
    top_k: int = 5
    min_trades: int = 50
    symbols: list[str] = field(default_factory=list)
    timeframe: str = "1h"
    start_date: str = ""
    end_date: str = ""

    def __post_init__(self) -> None:
        errors: list[str] = []
        if self.population_size < 2:
            errors.append("population_size must be >= 2")
        if self.max_generations < 1:
            errors.append("max_generations must be >= 1")
        if not (0.0 <= self.mutation_rate <= 1.0):
            errors.append("mutation_rate must be in [0, 1]")
        if not (0.0 <= self.crossover_rate <= 1.0):
            errors.append("crossover_rate must be in [0, 1]")
        if self.elite_count < 0:
            errors.append("elite_count must be >= 0")
        if self.elite_count >= self.population_size:
            errors.append("elite_count must be < population_size")
        if self.tournament_size < 1:
            errors.append("tournament_size must be >= 1")
        if self.convergence_generations < 1:
            errors.append("convergence_generations must be >= 1")
        if self.top_k < 1:
            errors.append("top_k must be >= 1")
        if errors:
            raise ValueError("; ".join(errors))


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Metrics for a single generation.

    Attributes:
        generation: Generation index (0-based).
        best_fitness: Highest adjusted_score in this generation.
        mean_fitness: Mean adjusted_score across the population.
        worst_fitness: Lowest adjusted_score in this generation.
        best_chromosome: Chromosome with the highest fitness.
        population_size: Number of individuals evaluated.
        num_passed_filters: Count of individuals that passed overfitting filters.
    """

    generation: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    best_chromosome: StrategyChromosome
    population_size: int
    num_passed_filters: int


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Final output of the discovery pipeline.

    Attributes:
        generations: Per-generation metrics.
        top_strategies: Top K (chromosome, fitness) pairs sorted descending.
        total_candidates_evaluated: Cumulative evaluations across all generations.
        converged: Whether the pipeline terminated due to convergence.
        convergence_generation: Generation index where convergence detected (None if not).
    """

    generations: list[GenerationResult]
    top_strategies: list[tuple[StrategyChromosome, FitnessResult]]
    total_candidates_evaluated: int
    converged: bool
    convergence_generation: int | None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class DiscoveryPipeline:
    """Genetic algorithm pipeline for strategy discovery.

    Args:
        config: Pipeline configuration.
        backtest_fn: Callable that runs a backtest for a chromosome and returns
            a dict with keys: sharpe_ratio, max_drawdown, profit_factor, total_trades.
        filter_fn: Optional callable for overfitting filter evaluation.
    """

    def __init__(
        self,
        config: DiscoveryConfig,
        backtest_fn: Callable[[StrategyChromosome], dict[str, float | int]],
        filter_fn: Callable[[StrategyChromosome, dict[str, float | int]], dict[str, bool]] | None = None,
    ) -> None:
        self.config = config
        self._backtest_fn = backtest_fn
        self._filter_fn = filter_fn

    # -- public API ---------------------------------------------------------

    def run(self) -> DiscoveryResult:
        """Execute the full evolutionary discovery loop.

        Returns:
            DiscoveryResult containing generation history and top strategies.
        """
        cfg = self.config
        population = initialize_population(cfg.population_size)
        generation_results: list[GenerationResult] = []
        total_evaluated = 0
        last_fitness_results: list[FitnessResult] = []

        # Track global best for top-K across all generations
        all_scored: list[tuple[StrategyChromosome, FitnessResult]] = []

        converged = False
        convergence_gen: int | None = None

        for gen in range(cfg.max_generations):
            # Evaluate
            fitness_results = evaluate_population(
                population,
                self._backtest_fn,
                self._filter_fn,
            )
            last_fitness_results = fitness_results
            total_evaluated += len(population)

            # Record per-individual scores
            for chrom, fr in zip(population, fitness_results, strict=True):
                all_scored.append((chrom.clone(), fr))

            # Build generation metrics
            scores = [fr.adjusted_score for fr in fitness_results]
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            gen_result = GenerationResult(
                generation=gen,
                best_fitness=scores[best_idx],
                mean_fitness=sum(scores) / len(scores),
                worst_fitness=min(scores),
                best_chromosome=population[best_idx].clone(),
                population_size=len(population),
                num_passed_filters=sum(1 for fr in fitness_results if fr.passed_filters),
            )
            generation_results.append(gen_result)

            logger.info(
                "gen=%d best=%.4f mean=%.4f pop=%d passed=%d",
                gen,
                gen_result.best_fitness,
                gen_result.mean_fitness,
                gen_result.population_size,
                gen_result.num_passed_filters,
            )

            # Convergence check
            if self._check_convergence(generation_results):
                converged = True
                convergence_gen = gen
                logger.info("Converged at generation %d", gen)
                break

            # Evolve next generation (skip on last iteration)
            population = self._evolve_generation(population, fitness_results)

        # Select top-K
        all_scored.sort(key=lambda t: t[1].adjusted_score, reverse=True)
        top_strategies = all_scored[: cfg.top_k]
        if last_fitness_results:
            exported = self._export_top_strategies(population, last_fitness_results)
            logger.debug("Exported %d top strategy DSL dicts", len(exported))

        return DiscoveryResult(
            generations=generation_results,
            top_strategies=top_strategies,
            total_candidates_evaluated=total_evaluated,
            converged=converged,
            convergence_generation=convergence_gen,
        )

    # -- internal -----------------------------------------------------------

    def _evolve_generation(
        self,
        population: list[StrategyChromosome],
        fitness_results: list[FitnessResult],
    ) -> list[StrategyChromosome]:
        """Produce next generation via elitism + tournament/crossover/mutation.

        Args:
            population: Current generation chromosomes.
            fitness_results: Parallel fitness results.

        Returns:
            New population of the same size.
        """
        cfg = self.config
        scores = [fr.adjusted_score for fr in fitness_results]

        # Elites carried forward unchanged
        new_pop = apply_elitism(population, scores, cfg.elite_count)

        # Fill remaining slots
        remaining = cfg.population_size - len(new_pop)
        while remaining > 0:
            parent_a = tournament_select(population, scores, cfg.tournament_size)
            parent_b = tournament_select(population, scores, cfg.tournament_size)
            if random.random() < cfg.crossover_rate:
                child_a, child_b = crossover(parent_a, parent_b)
            else:
                child_a, child_b = parent_a, parent_b
            child_a = mutate(child_a, cfg.mutation_rate)
            child_b = mutate(child_b, cfg.mutation_rate)

            for child in (child_a, child_b):
                if remaining <= 0:
                    break
                # Retry if invalid; fall back to random if all retries fail
                valid_child = child
                for _ in range(_MAX_OFFSPRING_RETRIES):
                    if is_valid_chromosome(valid_child):
                        break
                    valid_child = mutate(child, cfg.mutation_rate)
                else:
                    if not is_valid_chromosome(valid_child):
                        valid_child = _random_chromosome()
                new_pop.append(valid_child)
                remaining -= 1

        return new_pop

    def _check_convergence(self, generation_results: list[GenerationResult]) -> bool:
        """Check if best fitness has stagnated for convergence_generations.

        Requires at least 2*n generations to avoid false convergence from
        unusually good random initialization.

        Args:
            generation_results: All generation results so far.

        Returns:
            True if no improvement for convergence_generations consecutive gens.
        """
        n = self.config.convergence_generations
        if len(generation_results) < 2 * n:
            return False

        recent = generation_results[-n:]
        best_before = max(
            gr.best_fitness for gr in generation_results[: -n]
        )
        best_recent = max(gr.best_fitness for gr in recent)
        return best_recent <= best_before

    def _export_top_strategies(
        self,
        population: list[StrategyChromosome],
        fitness_results: list[FitnessResult],
    ) -> list[dict[str, object]]:
        """Export top K strategies as DSL-compatible YAML dicts.

        Uses genome.chromosome_to_dsl for conversion (single source of truth).

        Args:
            population: Current population.
            fitness_results: Parallel fitness results.

        Returns:
            List of DSL YAML dicts for the top-K strategies.
        """
        scores = [fr.adjusted_score for fr in fitness_results]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = ranked[: self.config.top_k]

        dsl_dicts: list[dict[str, object]] = []
        for idx in top_indices:
            chrom = population[idx]
            dsl = chromosome_to_dsl(chrom)
            # Override timeframe from pipeline config
            dsl["timeframe"] = self.config.timeframe
            dsl_dicts.append(dsl)
        return dsl_dicts
