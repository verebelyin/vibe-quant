"""Tests for vibe_quant.discovery.pipeline."""

from __future__ import annotations

from typing import Any

import pytest

from vibe_quant.discovery.genome import chromosome_to_dsl
from vibe_quant.discovery.operators import (
    StrategyChromosome,
    initialize_population,
    is_valid_chromosome,
)
from vibe_quant.discovery.pipeline import (
    DiscoveryConfig,
    DiscoveryPipeline,
    DiscoveryResult,
    GenerationResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_backtest(chrom: StrategyChromosome) -> dict[str, Any]:
    """Deterministic mock backtest: more entry genes -> slightly better sharpe."""
    n_genes = len(chrom.entry_genes) + len(chrom.exit_genes)
    return {
        "sharpe_ratio": 1.0 + n_genes * 0.1,
        "max_drawdown": 0.1,
        "profit_factor": 1.5,
        "total_trades": 100,
    }


def _mock_backtest_few_trades(chrom: StrategyChromosome) -> dict[str, Any]:
    """Mock backtest that returns too few trades."""
    return {
        "sharpe_ratio": 2.0,
        "max_drawdown": 0.05,
        "profit_factor": 2.0,
        "total_trades": 10,
    }


def _mock_filter(chrom: StrategyChromosome, bt: dict[str, Any]) -> dict[str, bool]:
    """Mock filter: pass if sharpe > 1."""
    return {"sharpe_check": bt["sharpe_ratio"] > 1.0}


def _make_config(**overrides: Any) -> DiscoveryConfig:
    """Create a small config suitable for testing."""
    defaults: dict[str, Any] = {
        "population_size": 10,
        "max_generations": 5,
        "mutation_rate": 0.1,
        "elite_count": 2,
        "tournament_size": 3,
        "convergence_generations": 3,
        "top_k": 3,
        "min_trades": 50,
        "symbols": ["BTC/USDT"],
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-06-01",
    }
    defaults.update(overrides)
    return DiscoveryConfig(**defaults)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestDiscoveryConfig:
    def test_defaults(self) -> None:
        cfg = DiscoveryConfig(symbols=["BTC/USDT"], start_date="2024-01-01", end_date="2024-06-01")
        assert cfg.population_size == 20
        assert cfg.max_generations == 15
        assert cfg.mutation_rate == 0.1
        assert cfg.elite_count == 2
        assert cfg.tournament_size == 3
        assert cfg.convergence_generations == 10
        assert cfg.top_k == 5
        assert cfg.min_trades == 50
        assert cfg.max_workers == 0
        assert cfg.timeframe == "4h"

    def test_invalid_population_size(self) -> None:
        with pytest.raises(ValueError, match="population_size"):
            DiscoveryConfig(population_size=1)

    def test_invalid_mutation_rate_low(self) -> None:
        with pytest.raises(ValueError, match="mutation_rate"):
            DiscoveryConfig(mutation_rate=-0.1)

    def test_invalid_mutation_rate_high(self) -> None:
        with pytest.raises(ValueError, match="mutation_rate"):
            DiscoveryConfig(mutation_rate=1.5)

    def test_elite_count_ge_population(self) -> None:
        with pytest.raises(ValueError, match="elite_count"):
            DiscoveryConfig(population_size=5, elite_count=5)

    def test_invalid_max_generations(self) -> None:
        with pytest.raises(ValueError, match="max_generations"):
            DiscoveryConfig(max_generations=0)

    def test_invalid_tournament_size(self) -> None:
        with pytest.raises(ValueError, match="tournament_size"):
            DiscoveryConfig(tournament_size=0)

    def test_invalid_convergence_generations(self) -> None:
        with pytest.raises(ValueError, match="convergence_generations"):
            DiscoveryConfig(convergence_generations=0)

    def test_invalid_top_k(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            DiscoveryConfig(top_k=0)

    def test_multiple_errors(self) -> None:
        with pytest.raises(ValueError, match=";"):
            DiscoveryConfig(population_size=0, max_generations=0)


# ---------------------------------------------------------------------------
# Pipeline init
# ---------------------------------------------------------------------------


class TestPipelineInit:
    def test_init_stores_config(self) -> None:
        cfg = _make_config()
        pipe = DiscoveryPipeline(cfg, _mock_backtest)
        assert pipe.config is cfg

    def test_init_with_filter(self) -> None:
        cfg = _make_config()
        pipe = DiscoveryPipeline(cfg, _mock_backtest, filter_fn=_mock_filter)
        assert pipe._filter_fn is _mock_filter


# ---------------------------------------------------------------------------
# Single generation evolution
# ---------------------------------------------------------------------------


class TestEvolveGeneration:
    def test_evolve_preserves_population_size(self) -> None:
        cfg = _make_config(population_size=10, elite_count=2)
        pipe = DiscoveryPipeline(cfg, _mock_backtest)
        pop = initialize_population(10)
        from vibe_quant.discovery.fitness import evaluate_population

        fitness = evaluate_population(pop, _mock_backtest)  # type: ignore[arg-type]
        new_pop = pipe._evolve_generation(pop, fitness)
        assert len(new_pop) == 10

    def test_evolve_returns_valid_chromosomes(self) -> None:
        cfg = _make_config(population_size=8, elite_count=1)
        pipe = DiscoveryPipeline(cfg, _mock_backtest)
        pop = initialize_population(8)
        from vibe_quant.discovery.fitness import evaluate_population

        fitness = evaluate_population(pop, _mock_backtest)  # type: ignore[arg-type]
        new_pop = pipe._evolve_generation(pop, fitness)
        for chrom in new_pop:
            assert is_valid_chromosome(chrom)


# ---------------------------------------------------------------------------
# Convergence detection
# ---------------------------------------------------------------------------


class TestConvergence:
    def _make_gen_result(self, gen: int, best: float) -> GenerationResult:
        pop = initialize_population(2)
        return GenerationResult(
            generation=gen,
            best_fitness=best,
            mean_fitness=best * 0.5,
            worst_fitness=0.0,
            best_chromosome=pop[0],
            population_size=2,
            num_passed_filters=2,
        )

    def test_not_converged_too_few_generations(self) -> None:
        cfg = _make_config(convergence_generations=3)
        pipe = DiscoveryPipeline(cfg, _mock_backtest)
        # Need at least 2*n=6 generations
        results = [self._make_gen_result(i, 1.0) for i in range(5)]
        assert not pipe._check_convergence(results)

    def test_converged_stagnation(self) -> None:
        cfg = _make_config(convergence_generations=3)
        pipe = DiscoveryPipeline(cfg, _mock_backtest)
        # Gens 0-2 improve, then gens 3-5 stagnate
        results = [
            self._make_gen_result(0, 1.0),
            self._make_gen_result(1, 1.5),
            self._make_gen_result(2, 2.0),
            self._make_gen_result(3, 1.0),
            self._make_gen_result(4, 1.0),
            self._make_gen_result(5, 1.0),
        ]
        assert pipe._check_convergence(results)

    def test_not_converged_improving(self) -> None:
        cfg = _make_config(convergence_generations=3)
        pipe = DiscoveryPipeline(cfg, _mock_backtest)
        results = [
            self._make_gen_result(0, 1.0),
            self._make_gen_result(1, 1.1),
            self._make_gen_result(2, 1.2),
            self._make_gen_result(3, 1.3),
            self._make_gen_result(4, 1.4),
            self._make_gen_result(5, 1.5),
        ]
        assert not pipe._check_convergence(results)


# ---------------------------------------------------------------------------
# Full pipeline run
# ---------------------------------------------------------------------------


class TestPipelineRun:
    def test_run_returns_discovery_result(self) -> None:
        cfg = _make_config(population_size=6, max_generations=3, top_k=2, elite_count=1)
        pipe = DiscoveryPipeline(cfg, _mock_backtest)
        result = pipe.run()
        assert isinstance(result, DiscoveryResult)
        assert len(result.generations) <= 3
        assert len(result.generations) >= 1
        assert len(result.top_strategies) <= 2
        assert result.total_candidates_evaluated >= 6

    def test_run_generation_results_increasing_gen_index(self) -> None:
        cfg = _make_config(population_size=6, max_generations=5, elite_count=1)
        pipe = DiscoveryPipeline(cfg, _mock_backtest)
        result = pipe.run()
        gens = [gr.generation for gr in result.generations]
        assert gens == list(range(len(gens)))

    def test_run_with_filter(self) -> None:
        cfg = _make_config(population_size=6, max_generations=3, elite_count=1)
        pipe = DiscoveryPipeline(cfg, _mock_backtest, filter_fn=_mock_filter)
        result = pipe.run()
        # All should pass since mock_backtest gives sharpe > 1
        for gr in result.generations:
            assert gr.num_passed_filters > 0

    def test_run_top_strategies_sorted_descending(self) -> None:
        cfg = _make_config(population_size=8, max_generations=3, top_k=3, elite_count=1)
        pipe = DiscoveryPipeline(cfg, _mock_backtest)
        result = pipe.run()
        scores = [fr.adjusted_score for _, fr in result.top_strategies]
        assert scores == sorted(scores, reverse=True)

    def test_run_convergence_triggers(self) -> None:
        """Constant backtest forces convergence."""
        call_count = 0

        def constant_backtest(chrom: StrategyChromosome) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.1,
                "profit_factor": 1.5,
                "total_trades": 100,
            }

        cfg = _make_config(
            population_size=6,
            max_generations=50,
            convergence_generations=3,
            elite_count=1,
        )
        pipe = DiscoveryPipeline(cfg, constant_backtest)
        result = pipe.run()
        # Should converge well before 50 generations
        assert result.converged
        assert result.convergence_generation is not None
        assert len(result.generations) < 50


# ---------------------------------------------------------------------------
# Top-K DSL export
# ---------------------------------------------------------------------------


class TestExportTopStrategies:
    def test_export_produces_valid_dsl_dicts(self) -> None:
        cfg = _make_config(population_size=6, max_generations=2, top_k=2, elite_count=1)
        pipe = DiscoveryPipeline(cfg, _mock_backtest)
        pop = initialize_population(6)
        from vibe_quant.discovery.fitness import evaluate_population

        fitness = evaluate_population(pop, _mock_backtest)  # type: ignore[arg-type]
        dsl_dicts = pipe._export_top_strategies(pop, fitness)
        assert len(dsl_dicts) == 2
        for d in dsl_dicts:
            assert "name" in d
            assert "timeframe" in d
            assert "indicators" in d
            assert "entry_conditions" in d
            assert "exit_conditions" in d
            assert "stop_loss" in d
            assert "take_profit" in d
            assert d["timeframe"] == "1h"
            assert d["stop_loss"]["type"] == "fixed_pct"
            assert d["take_profit"]["type"] == "fixed_pct"

    def test_export_dsl_has_indicators(self) -> None:
        pop = initialize_population(4)
        dsl = chromosome_to_dsl(pop[0])
        # Should have at least entry + exit indicators
        assert len(dsl["indicators"]) >= 2


# ---------------------------------------------------------------------------
# Elite preservation
# ---------------------------------------------------------------------------


class TestElitePreservation:
    def test_elites_present_in_next_generation(self) -> None:
        cfg = _make_config(population_size=8, elite_count=2)
        pipe = DiscoveryPipeline(cfg, _mock_backtest)
        pop = initialize_population(8)
        from vibe_quant.discovery.fitness import evaluate_population

        fitness = evaluate_population(pop, _mock_backtest)  # type: ignore[arg-type]
        scores = [fr.adjusted_score for fr in fitness]
        # Identify elite indices
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        elite_slp = [pop[ranked[0]].stop_loss_pct, pop[ranked[1]].stop_loss_pct]

        new_pop = pipe._evolve_generation(pop, fitness)
        new_slp = [c.stop_loss_pct for c in new_pop[:2]]
        # Elites are cloned into first positions
        assert new_slp == elite_slp
