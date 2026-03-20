"""Genetic discovery pipeline for trading strategy evolution.

Orchestrates population initialization, fitness evaluation, selection,
crossover, mutation, and convergence detection to discover profitable
strategy candidates expressed as DSL YAML dicts.
"""

from __future__ import annotations

import json
import logging
import random
import statistics
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from vibe_quant.discovery.fitness import FitnessResult, evaluate_population
from vibe_quant.discovery.genome import chromosome_to_dsl
from vibe_quant.discovery.guardrails import GuardrailConfig, GuardrailResult, apply_guardrails
from vibe_quant.discovery.operators import (
    StrategyChromosome,
    _random_chromosome,
    apply_elitism,
    crossover,
    crowding_replace,
    initialize_population,
    is_valid_chromosome,
    mutate,
    tournament_select,
)
from vibe_quant.utils import compute_bar_count

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from vibe_quant.discovery.operators import Direction

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

    population_size: int = 20
    max_generations: int = 15
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_count: int = 2
    tournament_size: int = 3
    convergence_generations: int = 3
    top_k: int = 5
    min_trades: int = 50
    max_workers: int | None = 0  # 0 = auto (cpu_count), None = sequential
    symbols: list[str] = field(default_factory=list)
    timeframe: str = "4h"
    start_date: str = ""
    end_date: str = ""
    indicator_pool: list[str] | None = None  # None = use all available
    direction: str | None = None  # "long", "short", "both", or None (random)
    use_crowding: bool = True  # Use deterministic crowding (True) or classic tournament (False)
    immigrant_fraction: float = 0.15  # Fraction of population replaced when entropy is low
    entropy_threshold: float = 0.4  # Entropy below this triggers immigrant injection
    min_diversity_distance: float = 0.15  # Min Gower distance for top-K dedup
    train_test_split: float = 0.0  # 0 = disabled; >0 = fraction for train (e.g. 0.5)
    cross_window_months: list[int] = field(default_factory=list)  # shifted windows, e.g. [1, 2]
    cross_window_min_pass: int = 2  # min windows (of total) that must pass
    cross_window_min_sharpe: float = 0.5  # min Sharpe on each window to count as pass

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
        # Cap convergence_gens so early stop can trigger before max_gen
        # _check_convergence requires 2*n gens, so cap at max_gen // 2
        max_conv = max(1, self.max_generations // 2)
        if self.convergence_generations > max_conv:
            object.__setattr__(self, "convergence_generations", max_conv)
        if self.top_k < 1:
            errors.append("top_k must be >= 1")
        if self.train_test_split < 0.0 or self.train_test_split >= 1.0:
            errors.append("train_test_split must be in [0, 1)")
        if errors:
            raise ValueError("; ".join(errors))

        # Auto-set min_trades for sub-5m timeframes if left at default (50).
        # 1m strategies need 100+ trades for statistical significance (bd-yu02).
        if self.min_trades == 50 and self.timeframe in ("1m", "2m", "3m"):
            from vibe_quant.discovery.fitness import MIN_TRADES_1M

            object.__setattr__(self, "min_trades", MIN_TRADES_1M)


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
class HoldoutResult:
    """Holdout (out-of-sample) evaluation for a single strategy.

    Attributes:
        sharpe_ratio: Holdout Sharpe ratio.
        max_drawdown: Holdout max drawdown.
        profit_factor: Holdout profit factor.
        total_trades: Holdout trade count.
        total_return: Holdout total return.
    """

    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    total_trades: int
    total_return: float


@dataclass(frozen=True, slots=True)
class CrossWindowResult:
    """Cross-window validation result for a single strategy.

    Attributes:
        window_results: Per-window HoldoutResult (index 0 = original, 1+ = shifted).
        windows_passed: Number of windows where strategy passed thresholds.
        total_windows: Total number of windows evaluated.
        passed: Whether strategy met cross_window_min_pass.
    """

    window_results: list[HoldoutResult]
    windows_passed: int
    total_windows: int
    passed: bool


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Final output of the discovery pipeline.

    Attributes:
        generations: Per-generation metrics.
        top_strategies: Top K (chromosome, fitness) pairs sorted descending.
        total_candidates_evaluated: Cumulative evaluations across all generations.
        converged: Whether the pipeline terminated due to convergence.
        convergence_generation: Generation index where convergence detected (None if not).
        holdout_results: Per-strategy holdout metrics (parallel to top_strategies). Empty if no split.
        train_dates: (start, end) for train period. None if no split.
        holdout_dates: (start, end) for holdout period. None if no split.
    """

    generations: list[GenerationResult]
    top_strategies: list[tuple[StrategyChromosome, FitnessResult]]
    total_candidates_evaluated: int
    converged: bool
    convergence_generation: int | None
    holdout_results: list[HoldoutResult] = field(default_factory=list)
    train_dates: tuple[str, str] | None = None
    holdout_dates: tuple[str, str] | None = None
    cross_window_results: list[CrossWindowResult] = field(default_factory=list)


def _select_diverse_top_k(
    scored: Sequence[tuple[StrategyChromosome, FitnessResult | float]],
    top_k: int = 5,
    min_distance: float = 0.15,
) -> list[tuple[StrategyChromosome, FitnessResult | float]]:
    """Select top-K strategies with diversity enforcement.

    Iterates through candidates sorted by fitness (descending). A candidate
    is added only if its distance to ALL already-selected strategies exceeds
    min_distance.

    Args:
        scored: List of (chromosome, fitness) tuples, sorted by fitness desc.
        top_k: Maximum number of strategies to select.
        min_distance: Minimum Gower distance to all selected strategies.

    Returns:
        List of up to top_k diverse (chromosome, fitness) tuples.
    """
    from vibe_quant.discovery.distance import chromosome_distance

    selected: list[tuple[StrategyChromosome, FitnessResult | float]] = []

    for chrom, fitness in scored:
        if len(selected) >= top_k:
            break

        is_diverse = all(
            chromosome_distance(chrom, sel_chrom) >= min_distance
            for sel_chrom, _ in selected
        )

        if is_diverse:
            selected.append((chrom, fitness))

    return selected


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
        filter_fn: Callable[[StrategyChromosome, dict[str, float | int]], dict[str, bool]]
        | None = None,
        progress_file: str | Path | None = None,
        holdout_backtest_fn: Callable[[StrategyChromosome], dict[str, float | int]] | None = None,
        backtest_fn_factory: Callable[[str, str], Callable[[StrategyChromosome], dict[str, float | int]]] | None = None,
        seed_chromosomes: list[StrategyChromosome] | None = None,
    ) -> None:
        self.config = config
        self._backtest_fn = backtest_fn
        self._filter_fn = filter_fn
        self._progress_file = Path(progress_file) if progress_file else None
        self._holdout_backtest_fn = holdout_backtest_fn
        self._backtest_fn_factory = backtest_fn_factory
        self._seed_chromosomes = seed_chromosomes
        self._direction_constraint: Direction | None = None

    # -- public API ---------------------------------------------------------

    @staticmethod
    def _create_executor(
        max_workers: int | None, population_size: int
    ) -> ProcessPoolExecutor | None:
        """Create a reusable worker pool for the entire discovery run.

        Returns None if parallelism is disabled (max_workers=None or 1)
        or if pool creation fails (macOS sandbox, Rust runtime conflicts).
        Falls back to per-generation pool creation in evaluate_population.
        """
        if max_workers is None or max_workers == 1:
            return None
        import os

        workers = max_workers if max_workers > 0 else (os.cpu_count() or 4)
        workers = min(workers, population_size)
        try:
            return ProcessPoolExecutor(max_workers=workers)
        except (OSError, RuntimeError):
            logger.warning("Failed to create shared worker pool, falling back to per-generation pools")
            return None

    def _apply_indicator_pool_filter(self) -> None:
        """Filter operators.INDICATOR_POOL to only include configured indicators.

        Safe to mutate module globals since discovery runs as a subprocess.
        """
        if self.config.indicator_pool is None:
            return
        from vibe_quant.discovery.operators import (
            _INDICATOR_NAMES,
            INDICATOR_POOL,
            _ensure_pool,
        )

        _ensure_pool()
        allowed = set(self.config.indicator_pool)
        to_remove = [k for k in INDICATOR_POOL if k not in allowed]
        for k in to_remove:
            del INDICATOR_POOL[k]
        _INDICATOR_NAMES.clear()
        _INDICATOR_NAMES.extend(INDICATOR_POOL.keys())
        logger.info("Indicator pool filtered to: %s", list(INDICATOR_POOL.keys()))

    def run(self) -> DiscoveryResult:
        """Execute the full evolutionary discovery loop.

        Returns:
            DiscoveryResult containing generation history and top strategies.
        """
        cfg = self.config
        self._apply_indicator_pool_filter()

        # Parse direction constraint
        from vibe_quant.discovery.operators import Direction
        direction_constraint: Direction | None = None
        if cfg.direction:
            direction_constraint = Direction(cfg.direction)
        self._direction_constraint = direction_constraint

        population = initialize_population(
            cfg.population_size,
            direction_constraint=direction_constraint,
            seed_chromosomes=self._seed_chromosomes,
        )
        generation_results: list[GenerationResult] = []
        total_evaluated = 0
        last_fitness_results: list[FitnessResult] = []

        # Track global best for top-K across all generations
        all_scored: list[tuple[StrategyChromosome, FitnessResult]] = []

        converged = False
        convergence_gen: int | None = None
        pipeline_start = time.monotonic()

        logger.info(
            "=== DISCOVERY START: pop=%d max_gen=%d symbols=%s tf=%s ===",
            cfg.population_size,
            cfg.max_generations,
            cfg.symbols,
            cfg.timeframe,
        )
        logger.info(
            "Config: mutation=%.2f crossover=%.2f elite=%d tournament=%d "
            "convergence_gens=%d direction=%s max_workers=%s indicator_pool=%s",
            cfg.mutation_rate,
            cfg.crossover_rate,
            cfg.elite_count,
            cfg.tournament_size,
            cfg.convergence_generations,
            cfg.direction or "random",
            cfg.max_workers,
            cfg.indicator_pool or "all",
        )

        # Create a long-lived worker pool to avoid per-generation pool startup
        # overhead (fixes idle workers when pool creation is slower than work)
        executor = self._create_executor(cfg.max_workers, cfg.population_size)

        for gen in range(cfg.max_generations):
            gen_start = time.monotonic()

            # Evaluate (parallel if max_workers configured)
            fitness_results = evaluate_population(
                population,
                self._backtest_fn,
                self._filter_fn,
                max_workers=cfg.max_workers,
                executor=executor,
                min_trades=cfg.min_trades,
            )
            last_fitness_results = fitness_results
            total_evaluated += len(population)

            gen_elapsed = time.monotonic() - gen_start
            total_elapsed = time.monotonic() - pipeline_start

            # Record per-individual scores (skip zero-fitness to reduce memory)
            for chrom, fr in zip(population, fitness_results, strict=True):
                if fr.adjusted_score > 0:
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

            # ETA calculation
            avg_gen_time = total_elapsed / (gen + 1)
            remaining_gens = cfg.max_generations - gen - 1
            eta_seconds = avg_gen_time * remaining_gens

            # Best metrics from top scorer
            best_fr = fitness_results[best_idx]
            best_trades = best_fr.total_trades
            best_return = best_fr.total_return

            # === Population analytics ===

            # Score distribution
            score_std = (sum((s - gen_result.mean_fitness) ** 2 for s in scores) / len(scores)) ** 0.5
            nonzero_scores = [s for s in scores if s > 0]
            zero_count = len(scores) - len(nonzero_scores)
            median_score = statistics.median(scores) if scores else 0.0

            # Metric distributions across population (for correctness checks)
            all_sharpes = [fr.sharpe_ratio for fr in fitness_results if fr.total_trades > 0]
            all_trades = [fr.total_trades for fr in fitness_results if fr.total_trades > 0]
            all_returns = [fr.total_return for fr in fitness_results if fr.total_trades > 0]
            all_dds = [fr.max_drawdown for fr in fitness_results if fr.total_trades > 0]

            # Indicator frequency across population
            ind_counter: Counter[str] = Counter()
            for c in population:
                for g in c.entry_genes + c.exit_genes:
                    ind_counter[g.indicator_type] += 1
            total_genes = sum(ind_counter.values()) or 1
            ind_pcts = {k: f"{v/total_genes*100:.0f}%" for k, v in ind_counter.most_common()}

            # Direction distribution
            dir_counts: dict[str, int] = {}
            for c in population:
                d = c.direction.value if hasattr(c.direction, "value") else str(c.direction)
                dir_counts[d] = dir_counts.get(d, 0) + 1

            # Gen-over-gen improvement tracking
            if gen > 0:
                prev_best = generation_results[-2].best_fitness
                improvement = gen_result.best_fitness - prev_best
                improvement_str = f" Δbest={improvement:+.4f}" if improvement != 0 else " (no change)"
            else:
                improvement_str = " (initial)"

            logger.info(
                "=== GEN %d/%d === best=%.4f mean=%.4f median=%.4f std=%.4f | "
                "zero_score=%d/%d | gen_time=%.1fs total=%.0fs ETA=%.0fs%s",
                gen + 1,
                cfg.max_generations,
                gen_result.best_fitness,
                gen_result.mean_fitness,
                median_score,
                score_std,
                zero_count,
                len(scores),
                gen_elapsed,
                total_elapsed,
                eta_seconds,
                improvement_str,
            )

            # Best chromosome details with full metrics
            best_chrom = population[best_idx]
            entry_indicators = [g.indicator_type for g in best_chrom.entry_genes]
            exit_indicators = [g.indicator_type for g in best_chrom.exit_genes]
            logger.info(
                "  Best: uid=%s dir=%s entry=%s exit=%s sl=%.1f%% tp=%.1f%% | "
                "sharpe=%.2f pf=%.2f dd=%.1f%% return=%.1f%% trades=%d",
                best_chrom.uid,
                best_chrom.direction.value if hasattr(best_chrom.direction, "value") else best_chrom.direction,
                entry_indicators,
                exit_indicators,
                best_chrom.stop_loss_pct,
                best_chrom.take_profit_pct,
                best_fr.sharpe_ratio,
                best_fr.profit_factor,
                best_fr.max_drawdown * 100,
                best_fr.total_return * 100,
                best_fr.total_trades,
            )

            # Score decomposition for best (helps verify fitness calc)
            logger.info(
                "  Score breakdown: raw=%.4f - complexity=%.4f - overtrade=%.4f = %.4f",
                best_fr.raw_score,
                best_fr.complexity_penalty,
                best_fr.overtrade_penalty,
                best_fr.adjusted_score,
            )

            # Population distributions
            if all_sharpes:
                logger.info(
                    "  Population (n=%d active): sharpe=[%.2f, %.2f, %.2f] "
                    "trades=[%d, %d, %d] dd=[%.1f%%, %.1f%%, %.1f%%] return=[%.1f%%, %.1f%%, %.1f%%]",
                    len(all_sharpes),
                    min(all_sharpes), statistics.median(all_sharpes), max(all_sharpes),
                    min(all_trades), int(statistics.median(all_trades)), max(all_trades),
                    min(all_dds) * 100, statistics.median(all_dds) * 100, max(all_dds) * 100,
                    min(all_returns) * 100, statistics.median(all_returns) * 100, max(all_returns) * 100,
                )

            # Diversity metrics
            from vibe_quant.discovery.diversity import population_entropy
            logger.info(
                "  Diversity: entropy=%.3f indicators=%s directions=%s",
                population_entropy(population),
                ind_pcts,
                dir_counts,
            )

            # Backtest timing stats (from gen_elapsed and pop size)
            avg_bt_time = gen_elapsed / len(population) if population else 0
            logger.info(
                "  Timing: %.1fs/chromosome avg (%.1fs total for %d chromosomes)",
                avg_bt_time,
                gen_elapsed,
                len(population),
            )

            self._write_progress(
                generation=gen + 1,
                max_generations=cfg.max_generations,
                best_fitness=gen_result.best_fitness,
                mean_fitness=gen_result.mean_fitness,
                worst_fitness=gen_result.worst_fitness,
                best_trades=best_trades,
                best_return=best_return,
                gen_time=gen_elapsed,
                total_elapsed=total_elapsed,
                eta_seconds=eta_seconds,
                total_evaluated=total_evaluated,
            )

            # Convergence check with progress tracking
            stagnant_gens = self._stagnant_generations(generation_results)
            if stagnant_gens > 0:
                logger.info(
                    "  Convergence: %d/%d stagnant gens (best unchanged)",
                    stagnant_gens,
                    cfg.convergence_generations,
                )
            if self._check_convergence(generation_results):
                converged = True
                convergence_gen = gen
                logger.info(
                    "=== CONVERGED at gen %d/%d after %.0fs ===",
                    gen + 1,
                    cfg.max_generations,
                    total_elapsed,
                )
                break

            # Evolve next generation (skip on last iteration)
            population = self._evolve_generation(population, fitness_results)

            # Diversity monitoring + immigrant injection
            from vibe_quant.discovery.diversity import (
                inject_random_immigrants,
                population_entropy,
                should_inject_immigrants,
            )
            entropy = population_entropy(population)
            if should_inject_immigrants(entropy, threshold=cfg.entropy_threshold):
                scores_for_inject = [fr.adjusted_score for fr in fitness_results]
                population = inject_random_immigrants(
                    population, scores_for_inject, fraction=cfg.immigrant_fraction,
                    direction_constraint=self._direction_constraint,
                )
                n_immigrants = max(1, int(len(population) * cfg.immigrant_fraction))
                logger.info(
                    "  Diversity intervention: entropy=%.3f < %.1f, injected %d random immigrants",
                    entropy, cfg.entropy_threshold, n_immigrants,
                )

        # Shut down worker pool after all generations complete
        if executor is not None:
            executor.shutdown(wait=True)

        # Select top-K with structural diversity enforcement
        all_scored.sort(key=lambda t: t[1].adjusted_score, reverse=True)
        top_strategies_raw = _select_diverse_top_k(
            all_scored,
            top_k=cfg.top_k,
            min_distance=cfg.min_diversity_distance,
        )
        # Cast back to the expected type (FitnessResult, not float)
        top_strategies: list[tuple[StrategyChromosome, FitnessResult]] = [
            (chrom, fr) for chrom, fr in top_strategies_raw  # type: ignore[misc]
        ]
        if last_fitness_results:
            exported = self._export_top_strategies(population, last_fitness_results)
            logger.debug("Exported %d top strategy DSL dicts", len(exported))

        # Log empirical Sharpe distribution across GA evaluations (diagnostic only).
        # NOTE: NOT passed to DSR — cross-strategy Sharpe dispersion is a category
        # error for the paper's V[{SR_n}] which measures within-strategy estimation
        # noise. See vibe-quant-fici.
        if len(all_scored) >= 2:
            sharpes = [fr.sharpe_ratio for _, fr in all_scored]
            mean_sr = sum(sharpes) / len(sharpes)
            var_sr = sum((s - mean_sr) ** 2 for s in sharpes) / (len(sharpes) - 1)
            logger.info(
                "GA Sharpe distribution: n=%d mean=%.3f std=%.3f min=%.3f max=%.3f",
                len(sharpes), mean_sr, var_sr ** 0.5, min(sharpes), max(sharpes),
            )

        # Validate top strategies with guardrails (DSR + min trades + complexity + bootstrap CI)
        validated = self._validate_top_strategies(top_strategies, total_evaluated)
        if validated is None:
            pass  # Soft failures only — keep unfiltered
        else:
            top_strategies = validated  # May be empty if hard guardrails rejected all

        # Final summary
        total_time = time.monotonic() - pipeline_start
        avg_gen_time = total_time / len(generation_results) if generation_results else 0

        logger.info(
            "=== DISCOVERY COMPLETE: %.0fs total (%.1fs/gen avg) | %d gens | %d evaluated | converged=%s ===",
            total_time,
            avg_gen_time,
            len(generation_results),
            total_evaluated,
            converged,
        )

        # Evolution timeline: how best score progressed across generations
        if generation_results:
            timeline = " → ".join(f"{gr.best_fitness:.3f}" for gr in generation_results)
            logger.info("  Evolution: %s", timeline)

            # Find which generation found the overall best
            best_gen_idx = max(range(len(generation_results)), key=lambda i: generation_results[i].best_fitness)
            logger.info(
                "  Peak gen: %d/%d (score=%.4f, %s of total time)",
                best_gen_idx + 1,
                len(generation_results),
                generation_results[best_gen_idx].best_fitness,
                f"{(best_gen_idx + 1) / len(generation_results) * 100:.0f}%",
            )

        # Top-K strategy details (not just winner)
        if top_strategies:
            logger.info("  --- Top %d strategies ---", len(top_strategies))
            for rank, (chrom, fit) in enumerate(top_strategies, 1):
                entry = [g.indicator_type for g in chrom.entry_genes]
                exit_ = [g.indicator_type for g in chrom.exit_genes]
                _dir = chrom.direction.value if hasattr(chrom.direction, "value") else chrom.direction
                logger.info(
                    "  #%d uid=%s score=%.4f sharpe=%.2f pf=%.2f dd=%.1f%% return=%.1f%% trades=%d | "
                    "dir=%s entry=%s exit=%s sl=%.1f%% tp=%.1f%%",
                    rank,
                    chrom.uid,
                    fit.adjusted_score,
                    fit.sharpe_ratio,
                    fit.profit_factor,
                    fit.max_drawdown * 100,
                    fit.total_return * 100,
                    fit.total_trades,
                    _dir,
                    entry,
                    exit_,
                    chrom.stop_loss_pct,
                    chrom.take_profit_pct,
                )

                # Log entry/exit gene details (thresholds, params) for reproduceability
                for i, gene in enumerate(chrom.entry_genes):
                    logger.info(
                        "    entry[%d]: %s(%s) %s %.4f%s",
                        i,
                        gene.indicator_type,
                        ", ".join(f"{k}={v}" for k, v in gene.parameters.items()),
                        gene.condition.value if hasattr(gene.condition, "value") else gene.condition,
                        gene.threshold,
                        f" sub={gene.sub_value}" if gene.sub_value else "",
                    )
                for i, gene in enumerate(chrom.exit_genes):
                    logger.info(
                        "    exit[%d]: %s(%s) %s %.4f%s",
                        i,
                        gene.indicator_type,
                        ", ".join(f"{k}={v}" for k, v in gene.parameters.items()),
                        gene.condition.value if hasattr(gene.condition, "value") else gene.condition,
                        gene.threshold,
                        f" sub={gene.sub_value}" if gene.sub_value else "",
                    )

        # Holdout evaluation: run top strategies on unseen data
        holdout_results: list[HoldoutResult] = []
        train_dates: tuple[str, str] | None = None
        holdout_dates: tuple[str, str] | None = None
        if self._holdout_backtest_fn is not None and top_strategies and cfg.train_test_split > 0:
            holdout_results = self._evaluate_holdout(top_strategies)
            # Compute split dates for metadata
            from vibe_quant.utils import split_date_range
            ts, te, hs, he = split_date_range(cfg.start_date, cfg.end_date, cfg.train_test_split)
            train_dates = (ts, te)
            holdout_dates = (hs, he)

        # Cross-window validation: run top strategies on shifted date windows
        cross_window_results: list[CrossWindowResult] = []
        if (
            cfg.cross_window_months
            and self._backtest_fn_factory is not None
            and top_strategies
        ):
            cross_window_results, top_strategies = self._evaluate_cross_windows(
                top_strategies,
            )

        return DiscoveryResult(
            generations=generation_results,
            top_strategies=top_strategies,
            total_candidates_evaluated=total_evaluated,
            converged=converged,
            convergence_generation=convergence_gen,
            holdout_results=holdout_results,
            train_dates=train_dates,
            holdout_dates=holdout_dates,
            cross_window_results=cross_window_results,
        )

    def _evaluate_holdout(
        self,
        top_strategies: list[tuple[StrategyChromosome, FitnessResult]],
    ) -> list[HoldoutResult]:
        """Evaluate top strategies on holdout (out-of-sample) period.

        Returns HoldoutResult for each strategy, parallel to top_strategies.
        """
        assert self._holdout_backtest_fn is not None
        holdout_fn = self._holdout_backtest_fn
        results: list[HoldoutResult] = []

        logger.info("=== HOLDOUT EVALUATION: %d strategies ===", len(top_strategies))

        for rank, (chrom, train_fit) in enumerate(top_strategies, 1):
            try:
                bt = holdout_fn(chrom)
                import math as _math
                sharpe = float(bt.get("sharpe_ratio", 0.0))
                max_dd = float(bt.get("max_drawdown", 1.0))
                pf = float(bt.get("profit_factor", 0.0))
                trades = int(bt.get("total_trades", 0))
                ret = float(bt.get("total_return", 0.0))
                if _math.isnan(sharpe):
                    sharpe = 0.0
                if _math.isnan(max_dd):
                    max_dd = 1.0
                if _math.isnan(pf):
                    pf = 0.0
                if _math.isnan(ret):
                    ret = 0.0
                hr = HoldoutResult(
                    sharpe_ratio=sharpe,
                    max_drawdown=max_dd,
                    profit_factor=pf,
                    total_trades=trades,
                    total_return=ret,
                )
            except Exception:
                logger.warning("Holdout eval failed for %s", chrom.uid, exc_info=True)
                hr = HoldoutResult(
                    sharpe_ratio=0.0, max_drawdown=1.0,
                    profit_factor=0.0, total_trades=0, total_return=0.0,
                )
            results.append(hr)

            # Log train vs holdout comparison
            logger.info(
                "  #%d %s: TRAIN sharpe=%.2f dd=%.1f%% ret=%.1f%% trades=%d → "
                "HOLDOUT sharpe=%.2f dd=%.1f%% ret=%.1f%% trades=%d",
                rank, chrom.uid,
                train_fit.sharpe_ratio, train_fit.max_drawdown * 100,
                train_fit.total_return * 100, train_fit.total_trades,
                hr.sharpe_ratio, hr.max_drawdown * 100,
                hr.total_return * 100, hr.total_trades,
            )

        # Summary: how much did strategies degrade on holdout?
        if results:
            train_sharpes = [f.sharpe_ratio for _, f in top_strategies]
            holdout_sharpes = [h.sharpe_ratio for h in results]
            avg_train = sum(train_sharpes) / len(train_sharpes)
            avg_holdout = sum(holdout_sharpes) / len(holdout_sharpes)
            degradation = (avg_train - avg_holdout) / avg_train * 100 if avg_train > 0 else 0
            logger.info(
                "  Holdout summary: avg_train_sharpe=%.2f → avg_holdout_sharpe=%.2f "
                "(%.1f%% degradation)",
                avg_train, avg_holdout, degradation,
            )

        return results

    def _evaluate_cross_windows(
        self,
        top_strategies: list[tuple[StrategyChromosome, FitnessResult]],
    ) -> tuple[list[CrossWindowResult], list[tuple[StrategyChromosome, FitnessResult]]]:
        """Evaluate top strategies across shifted time windows.

        Creates shifted windows by offsetting start/end dates by N months.
        Filters strategies that don't pass on enough windows.

        Returns:
            (cross_window_results, filtered_top_strategies)
        """
        assert self._backtest_fn_factory is not None
        cfg = self.config
        offsets = cfg.cross_window_months
        min_sharpe = cfg.cross_window_min_sharpe
        min_pass = cfg.cross_window_min_pass

        from datetime import datetime as _dt

        from dateutil.relativedelta import relativedelta

        base_start = _dt.strptime(cfg.start_date, "%Y-%m-%d")
        base_end = _dt.strptime(cfg.end_date, "%Y-%m-%d")

        # Build list of windows: original + shifted
        windows: list[tuple[str, str]] = [(cfg.start_date, cfg.end_date)]
        for months in offsets:
            ws = (base_start + relativedelta(months=months)).strftime("%Y-%m-%d")
            we = (base_end + relativedelta(months=months)).strftime("%Y-%m-%d")
            windows.append((ws, we))

        logger.info(
            "=== CROSS-WINDOW VALIDATION: %d strategies × %d windows ===",
            len(top_strategies), len(windows),
        )
        for i, (ws, we) in enumerate(windows):
            label = "original" if i == 0 else f"+{offsets[i-1]}mo"
            logger.info("  Window %d (%s): %s → %s", i, label, ws, we)

        cross_results: list[CrossWindowResult] = []
        filtered: list[tuple[StrategyChromosome, FitnessResult]] = []

        for rank, (chrom, train_fit) in enumerate(top_strategies, 1):
            window_hrs: list[HoldoutResult] = []
            passes = 0

            for w_idx, (ws, we) in enumerate(windows):
                try:
                    if w_idx == 0:
                        # Original window — use train fitness directly
                        hr = HoldoutResult(
                            sharpe_ratio=train_fit.sharpe_ratio,
                            max_drawdown=train_fit.max_drawdown,
                            profit_factor=train_fit.profit_factor,
                            total_trades=train_fit.total_trades,
                            total_return=train_fit.total_return,
                        )
                    else:
                        bt_fn = self._backtest_fn_factory(ws, we)
                        bt = bt_fn(chrom)
                        import math as _math
                        sharpe = float(bt.get("sharpe_ratio", 0.0))
                        max_dd = float(bt.get("max_drawdown", 1.0))
                        pf = float(bt.get("profit_factor", 0.0))
                        trades = int(bt.get("total_trades", 0))
                        ret = float(bt.get("total_return", 0.0))
                        if _math.isnan(sharpe):
                            sharpe = 0.0
                        if _math.isnan(max_dd):
                            max_dd = 1.0
                        if _math.isnan(pf):
                            pf = 0.0
                        if _math.isnan(ret):
                            ret = 0.0
                        hr = HoldoutResult(
                            sharpe_ratio=sharpe,
                            max_drawdown=max_dd,
                            profit_factor=pf,
                            total_trades=trades,
                            total_return=ret,
                        )
                except Exception:
                    logger.warning(
                        "Cross-window eval failed: %s window %d", chrom.uid, w_idx,
                        exc_info=True,
                    )
                    hr = HoldoutResult(
                        sharpe_ratio=0.0, max_drawdown=1.0,
                        profit_factor=0.0, total_trades=0, total_return=0.0,
                    )

                window_hrs.append(hr)
                if hr.total_return > 0 and hr.sharpe_ratio >= min_sharpe:
                    passes += 1

            passed = passes >= min_pass
            cwr = CrossWindowResult(
                window_results=window_hrs,
                windows_passed=passes,
                total_windows=len(windows),
                passed=passed,
            )
            cross_results.append(cwr)

            # Log per-strategy cross-window summary
            window_strs = []
            for i, hr in enumerate(window_hrs):
                label = "W0" if i == 0 else f"W+{offsets[i-1]}mo"
                status = "PASS" if (hr.total_return > 0 and hr.sharpe_ratio >= min_sharpe) else "FAIL"
                window_strs.append(
                    f"{label}: sharpe={hr.sharpe_ratio:.2f} ret={hr.total_return*100:.1f}% [{status}]"
                )
            logger.info(
                "  #%d %s: %s → %d/%d windows %s",
                rank, chrom.uid,
                " | ".join(window_strs),
                passes, len(windows),
                "PROMOTED" if passed else "REJECTED",
            )

            if passed:
                filtered.append((chrom, train_fit))

        logger.info(
            "  Cross-window summary: %d/%d strategies promoted",
            len(filtered), len(top_strategies),
        )

        # If all rejected, keep original strategies with a warning
        if not filtered:
            logger.warning(
                "All strategies failed cross-window validation — keeping originals"
            )
            filtered = top_strategies

        return cross_results, filtered

    def _write_progress(self, **kwargs: object) -> None:
        """Write progress JSON file for API polling."""
        if not self._progress_file:
            return
        try:
            self._progress_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._progress_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(kwargs, default=str))
            tmp.rename(self._progress_file)
        except Exception:
            logger.debug("Failed to write progress file", exc_info=True)

    # -- internal -----------------------------------------------------------

    def _evolve_generation(
        self,
        population: list[StrategyChromosome],
        fitness_results: list[FitnessResult],
    ) -> list[StrategyChromosome]:
        """Produce next generation via crowding or classic tournament.

        Args:
            population: Current generation chromosomes.
            fitness_results: Parallel fitness results.

        Returns:
            New population of the same size.
        """
        cfg = self.config
        scores = [fr.adjusted_score for fr in fitness_results]

        if cfg.use_crowding:
            return self._evolve_crowding(population, scores)
        return self._evolve_tournament(population, scores)

    def _evolve_tournament(
        self,
        population: list[StrategyChromosome],
        scores: list[float],
    ) -> list[StrategyChromosome]:
        """Classic evolution: elitism + tournament selection + crossover + mutation."""
        cfg = self.config
        new_pop = apply_elitism(population, scores, cfg.elite_count)

        remaining = cfg.population_size - len(new_pop)
        retries = 0
        random_fallbacks = 0
        while remaining > 0:
            parent_a = tournament_select(population, scores, cfg.tournament_size)
            parent_b = tournament_select(population, scores, cfg.tournament_size)
            if random.random() < cfg.crossover_rate:
                child_a, child_b = crossover(parent_a, parent_b)
            else:
                child_a, child_b = parent_a, parent_b
            child_a = mutate(child_a, cfg.mutation_rate)
            child_b = mutate(child_b, cfg.mutation_rate)

            if self._direction_constraint is not None:
                child_a.direction = self._direction_constraint
                child_b.direction = self._direction_constraint

            for child in (child_a, child_b):
                if remaining <= 0:
                    break
                valid_child = child
                for attempt in range(_MAX_OFFSPRING_RETRIES):
                    if is_valid_chromosome(valid_child):
                        if attempt > 0:
                            retries += attempt
                        break
                    valid_child = mutate(child, cfg.mutation_rate)
                else:
                    if not is_valid_chromosome(valid_child):
                        valid_child = _random_chromosome(direction_constraint=self._direction_constraint)
                        random_fallbacks += 1
                new_pop.append(valid_child)
                remaining -= 1

        if retries > 0 or random_fallbacks > 0:
            logger.info(
                "  Evolution: %d mutation retries, %d random fallbacks",
                retries,
                random_fallbacks,
            )

        return new_pop

    def _evolve_crowding(
        self,
        population: list[StrategyChromosome],
        scores: list[float],
    ) -> list[StrategyChromosome]:
        """Deterministic crowding evolution.

        1. Keep 1 elite as safety net
        2. Randomly pair remaining individuals
        3. Each pair produces 2 offspring (crossover + mutation)
        4. Offspring replaces most-similar parent only if fitter

        Offspring aren't evaluated until next gen, so they get a "free pass"
        into the population. The real crowding competition happens at
        subsequent generations when fitness values are available.
        """
        cfg = self.config

        # Keep 1 elite as safety net (down from default 2)
        elite = apply_elitism(population, scores, min(1, cfg.elite_count))

        # Build replacement pool (all non-elite)
        elite_indices = set()
        if elite:
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            elite_indices.add(best_idx)

        pool_indices = [i for i in range(len(population)) if i not in elite_indices]
        random.shuffle(pool_indices)

        # Pair up for crowding
        new_pop = list(elite)
        retries = 0
        random_fallbacks = 0

        for pair_start in range(0, len(pool_indices) - 1, 2):
            i, j = pool_indices[pair_start], pool_indices[pair_start + 1]
            parent_a, parent_b = population[i], population[j]

            # Crossover + mutation
            if random.random() < cfg.crossover_rate:
                child_a, child_b = crossover(parent_a, parent_b)
            else:
                child_a, child_b = parent_a.clone(), parent_b.clone()
            child_a = mutate(child_a, cfg.mutation_rate)
            child_b = mutate(child_b, cfg.mutation_rate)

            if self._direction_constraint is not None:
                child_a.direction = self._direction_constraint
                child_b.direction = self._direction_constraint

            # Validate offspring
            children = []
            for child in (child_a, child_b):
                valid = child
                for attempt in range(_MAX_OFFSPRING_RETRIES):
                    if is_valid_chromosome(valid):
                        if attempt > 0:
                            retries += attempt
                        break
                    valid = mutate(child, cfg.mutation_rate)
                else:
                    if not is_valid_chromosome(valid):
                        valid = _random_chromosome(direction_constraint=self._direction_constraint)
                        random_fallbacks += 1
                children.append(valid)

            # Crowding replacement: offspring compete against similar parent
            result = crowding_replace(
                parents=[parent_a, parent_b],
                parent_fitness=[scores[i], scores[j]],
                offspring=children,
                offspring_fitness=[scores[i], scores[j]],  # No real fitness yet; tie = offspring wins
            )
            new_pop.extend(result)

        # Handle odd pool (last unpaired individual)
        if len(pool_indices) % 2 == 1:
            new_pop.append(population[pool_indices[-1]].clone())

        # Trim to population size
        if len(new_pop) > cfg.population_size:
            new_pop = new_pop[: cfg.population_size]
        # Pad if somehow short
        while len(new_pop) < cfg.population_size:
            new_pop.append(_random_chromosome(direction_constraint=self._direction_constraint))

        if retries > 0 or random_fallbacks > 0:
            logger.info(
                "  Evolution: %d mutation retries, %d random fallbacks",
                retries,
                random_fallbacks,
            )

        return new_pop

    def _stagnant_generations(self, generation_results: list[GenerationResult]) -> int:
        """Count consecutive generations without improvement from the end."""
        if len(generation_results) < 2:
            return 0
        current_best = generation_results[-1].best_fitness
        count = 0
        for gr in reversed(generation_results[:-1]):
            if gr.best_fitness >= current_best:
                count += 1
            else:
                break
        return count

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
        best_before = max(gr.best_fitness for gr in generation_results[:-n])
        best_recent = max(gr.best_fitness for gr in recent)
        return best_recent <= best_before

    def _validate_top_strategies(
        self,
        top_strategies: list[tuple[StrategyChromosome, FitnessResult]],
        total_evaluated: int,
    ) -> list[tuple[StrategyChromosome, FitnessResult]] | None:
        """Apply guardrails (DSR, complexity, min trades) to top strategies.

        Logs validation results and filters out strategies that fail.

        When all candidates fail:
        - If any "hard" guardrail failed (bootstrap CI, min trades), returns
          empty list — the run found nothing statistically significant.
        - If only "soft" guardrails failed (DSR, complexity), returns None
          so the caller keeps best-available candidates.

        DSR uses theoretical variance (1/(T-1)) rather than empirical
        cross-strategy variance, which inflates SR₀ beyond what any
        strategy can achieve (see vibe-quant-fici).
        """
        guardrail_cfg = GuardrailConfig(
            min_trades=self.config.min_trades,
            max_complexity=8,
            require_dsr=True,
            require_wfa=False,  # WFA requires separate out-of-sample data
            require_purged_kfold=False,
            require_bootstrap_ci=True,
        )

        # Use actual bar count for DSR (not total_trades * 5 proxy)
        bar_count = compute_bar_count(
            self.config.start_date, self.config.end_date, self.config.timeframe,
        )

        import numpy as np

        validated: list[tuple[StrategyChromosome, FitnessResult]] = []
        any_hard_failure = False
        for chrom, fitness in top_strategies:
            num_genes = len(chrom.entry_genes) + len(chrom.exit_genes)
            num_obs = bar_count if bar_count else max(100, fitness.total_trades * 5)

            trade_ret = np.array(fitness.trade_returns) if fitness.trade_returns else None
            result: GuardrailResult = apply_guardrails(
                fitness=fitness,
                num_genes=num_genes,
                config=guardrail_cfg,
                num_trials=total_evaluated,
                num_observations=num_obs,
                skewness=fitness.skewness,
                kurtosis=fitness.kurtosis,
                trade_returns=trade_ret,
                # trials_sharpe_variance intentionally omitted — use theoretical
                # 1/(T-1). Cross-strategy Sharpe dispersion from GA is NOT what
                # the paper's V[{SR_n}] measures (see vibe-quant-fici).
            )
            if result.passed:
                validated.append((chrom, fitness))
                logger.info(
                    "Guardrail PASS: %s score=%.4f sharpe=%.2f trades=%d",
                    chrom.uid,
                    fitness.adjusted_score,
                    fitness.sharpe_ratio,
                    fitness.total_trades,
                )
            else:
                # Track hard guardrail failures (bootstrap CI, min trades)
                if not result.min_trades_passed or result.bootstrap_passed is False:
                    any_hard_failure = True
                logger.info(
                    "Guardrail FAIL: %s score=%.4f reasons=%s",
                    chrom.uid,
                    fitness.adjusted_score,
                    result.reasons,
                )

        if not validated:
            if any_hard_failure:
                logger.warning(
                    "All top strategies failed hard guardrails (bootstrap CI / min trades) "
                    "— run found nothing statistically significant"
                )
                return []
            logger.warning("All top strategies failed soft guardrails, keeping unfiltered")
            return None

        logger.info("%d/%d top strategies passed guardrails", len(validated), len(top_strategies))
        return validated

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
