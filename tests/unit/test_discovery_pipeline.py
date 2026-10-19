"""Tests for vibe_quant.discovery.pipeline."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
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
    DiscoveryEvaluationError,
    DiscoveryPipeline,
    DiscoveryResult,
    GenerationResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_backtest(chrom: StrategyChromosome) -> dict[str, Any]:
    """Deterministic mock backtest: more entry genes -> slightly better sharpe."""
    import random as _rng

    n_genes = len(chrom.entry_genes) + len(chrom.exit_genes)
    total_return = 0.5 + n_genes * 0.05
    total_trades = 150
    # Generate synthetic trade returns with strong positive mean for bootstrap CI
    r = _rng.Random(n_genes)
    mean_ret = total_return / total_trades  # ~0.003+
    trade_returns = tuple(r.gauss(mean_ret, mean_ret * 0.3) for _ in range(total_trades))
    return {
        "sharpe_ratio": 2.0 + n_genes * 0.1,
        "max_drawdown": 0.08,
        "profit_factor": 2.0,
        "total_trades": total_trades,
        "total_return": total_return,
        "trade_returns": trade_returns,
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


def _fit(
    sharpe: float = 2.0, trades: int = 150, ret: float = 0.4, score: float = 0.6,
    dd: float = 0.1, pf: float = 1.8,
) -> Any:
    """Minimal passing FitnessResult for gate tests."""
    from vibe_quant.discovery.fitness import FitnessResult

    return FitnessResult(
        sharpe_ratio=sharpe, max_drawdown=dd, profit_factor=pf, total_trades=trades,
        total_return=ret, complexity_penalty=0.0, overtrade_penalty=0.0, sl_tp_penalty=0.0,
        raw_score=score, adjusted_score=score, passed_filters=True, filter_results={},
    )


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
        # Sequential: these tests use tiny populations where a spawn pool's
        # startup dominates the work. Parallel evaluation is covered by
        # test_discovery_symbol_split.py.
        "max_workers": 1,
        # The mock Sharpes (~2.5 over 5 months) are not DSR-significant; gates
        # fail closed now, so tests of OTHER gates disable DSR explicitly.
        "require_dsr": False,
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
        assert cfg.convergence_generations == 3
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


class TestCreateExecutor:
    """``DiscoveryPipeline._create_executor`` gates the shared worker pool.

    A pool is created only for ``max_workers > 1`` and is capped by the
    population; 1/None stay sequential (the pipeline tests rely on that), and
    0 or negative mean "auto" (``os.cpu_count()``).
    """

    def test_none_and_one_disable_pool(self) -> None:
        assert DiscoveryPipeline._create_executor(None, 10) is None
        assert DiscoveryPipeline._create_executor(1, 10) is None

    def test_two_workers_create_pool(self) -> None:
        executor = DiscoveryPipeline._create_executor(2, 10)
        try:
            assert isinstance(executor, ProcessPoolExecutor)
            assert executor._max_workers == 2
        finally:
            executor.shutdown()

    def test_workers_capped_by_population(self) -> None:
        executor = DiscoveryPipeline._create_executor(8, 3)
        try:
            assert isinstance(executor, ProcessPoolExecutor)
            assert executor._max_workers == 3
        finally:
            executor.shutdown()

    def test_zero_or_negative_means_auto(self) -> None:
        expected = min(os.cpu_count() or 4, 10)
        for max_workers in (0, -2):
            executor = DiscoveryPipeline._create_executor(max_workers, 10)
            try:
                assert isinstance(executor, ProcessPoolExecutor)
                assert executor._max_workers == expected
            finally:
                executor.shutdown()


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
        new_pop, known = pipe._evolve_generation(pop, fitness)
        assert len(new_pop) == 10
        assert len(known) == 10

    def test_evolve_returns_valid_chromosomes(self) -> None:
        cfg = _make_config(population_size=8, elite_count=1)
        pipe = DiscoveryPipeline(cfg, _mock_backtest)
        pop = initialize_population(8)
        from vibe_quant.discovery.fitness import evaluate_population

        fitness = evaluate_population(pop, _mock_backtest)  # type: ignore[arg-type]
        new_pop, _known = pipe._evolve_generation(pop, fitness)
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
        cfg = _make_config(convergence_generations=3, max_generations=10)
        pipe = DiscoveryPipeline(cfg, _mock_backtest)
        # Need at least 2*n=6 generations
        results = [self._make_gen_result(i, 1.0) for i in range(5)]
        assert not pipe._check_convergence(results)

    def test_converged_stagnation(self) -> None:
        cfg = _make_config(convergence_generations=3, max_generations=10)
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
        cfg = _make_config(convergence_generations=3, max_generations=10)
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
            max_workers=1,  # local closure is unpicklable; pool path used to fail silently
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


class TestStructuralDedup:
    """Tests for distance-based top-K deduplication."""

    def test_clones_rejected(self) -> None:
        """Structurally identical chromosomes are deduped even with different UIDs."""
        from vibe_quant.discovery.operators import ConditionType, Direction, StrategyGene
        from vibe_quant.discovery.pipeline import _select_diverse_top_k

        gene = StrategyGene(
            indicator_type="RSI", parameters={"period": 14.0},
            condition=ConditionType.GT, threshold=50.0,
        )
        candidates = []
        for i in range(5):
            c = StrategyChromosome(
                entry_genes=[gene.clone()], exit_genes=[gene.clone()],
                stop_loss_pct=2.0, take_profit_pct=5.0, direction=Direction.LONG,
            )
            candidates.append((c, 0.8 - i * 0.01))
        result = _select_diverse_top_k(candidates, top_k=5, min_distance=0.15)
        assert len(result) == 1  # Only 1 unique structure

    def test_diverse_strategies_all_kept(self) -> None:
        """Structurally different strategies are all kept."""
        from vibe_quant.discovery.operators import ConditionType, Direction, StrategyGene
        from vibe_quant.discovery.pipeline import _select_diverse_top_k

        indicators = ["RSI", "CCI", "ATR", "STOCH", "ADX"]
        candidates = []
        for i, ind in enumerate(indicators):
            params: dict[str, float] = {"period": 14.0}
            if ind == "STOCH":
                params = {"k_period": 14.0, "d_period": 3.0}
            gene = StrategyGene(
                indicator_type=ind, parameters=params,
                condition=ConditionType.GT, threshold=50.0 if ind in ("RSI", "CCI") else 0.01,
            )
            c = StrategyChromosome(
                entry_genes=[gene], exit_genes=[gene.clone()],
                stop_loss_pct=2.0, take_profit_pct=5.0, direction=Direction.LONG,
            )
            candidates.append((c, 0.9 - i * 0.01))
        result = _select_diverse_top_k(candidates, top_k=5, min_distance=0.15)
        assert len(result) == 5  # All different indicators


class TestBehaviouralDedup:
    """Top-K slots must not go to behaviourally identical genomes.

    Two structurally distant chromosomes can still produce the exact same
    trade list, hence bit-identical backtest metrics. Such a duplicate must be
    skipped even though the structural distance rule would have kept it.
    """

    @staticmethod
    def _chrom(indicator: str) -> StrategyChromosome:
        from vibe_quant.discovery.operators import ConditionType, Direction, StrategyGene

        params: dict[str, float] = {"period": 14.0}
        if indicator == "STOCH":
            params = {"k_period": 14.0, "d_period": 3.0}
        gene = StrategyGene(
            indicator_type=indicator, parameters=params,
            condition=ConditionType.GT,
            threshold=50.0 if indicator in ("RSI", "CCI") else 0.01,
        )
        return StrategyChromosome(
            entry_genes=[gene], exit_genes=[gene.clone()],
            stop_loss_pct=2.0, take_profit_pct=5.0, direction=Direction.LONG,
        )

    def test_identical_metrics_deduped_next_distinct_fills_slot(self) -> None:
        """P1: identical metrics + distant genotype -> second skipped, next fills."""
        from vibe_quant.discovery.pipeline import _select_diverse_top_k

        rsi = self._chrom("RSI")
        cci = self._chrom("CCI")
        atr = self._chrom("ATR")
        candidates = [
            (rsi, _fit(ret=0.4)),
            (cci, _fit(ret=0.4)),  # identical fingerprint, distant genotype
            (atr, _fit(ret=0.3)),
        ]
        result = _select_diverse_top_k(candidates, top_k=2, min_distance=0.15)
        assert [c.uid for c, _ in result] == [rsi.uid, atr.uid]

    def test_one_field_differs_by_epsilon_both_kept(self) -> None:
        """P2: exact float equality — a 1e-12 change is a different behaviour."""
        from vibe_quant.discovery.pipeline import _select_diverse_top_k

        rsi = self._chrom("RSI")
        cci = self._chrom("CCI")
        atr = self._chrom("ATR")
        candidates = [
            (rsi, _fit(ret=0.4)),
            (cci, _fit(ret=0.4 + 1e-12)),
            (atr, _fit(ret=0.3)),
        ]
        result = _select_diverse_top_k(candidates, top_k=2, min_distance=0.15)
        assert [c.uid for c, _ in result] == [rsi.uid, cci.uid]

    def test_zero_trade_duplicates_deduped(self) -> None:
        """P4: two identical zero-trade results are behaviourally identical."""
        from vibe_quant.discovery.pipeline import _select_diverse_top_k

        rsi = self._chrom("RSI")
        cci = self._chrom("CCI")
        candidates = [
            (rsi, _fit(sharpe=0.0, trades=0, ret=0.0)),
            (cci, _fit(sharpe=0.0, trades=0, ret=0.0)),
        ]
        result = _select_diverse_top_k(candidates, top_k=2, min_distance=0.15)
        assert [c.uid for c, _ in result] == [rsi.uid]

    def test_nan_metrics_never_deduped(self) -> None:
        """P5: NaN != NaN, so NaN fingerprints are never treated as duplicates."""
        from vibe_quant.discovery.pipeline import _select_diverse_top_k

        rsi = self._chrom("RSI")
        cci = self._chrom("CCI")
        candidates = [
            (rsi, _fit(sharpe=float("nan"))),
            (cci, _fit(sharpe=float("nan"))),
        ]
        result = _select_diverse_top_k(candidates, top_k=2, min_distance=0.15)
        assert [c.uid for c, _ in result] == [rsi.uid, cci.uid]

    def test_float_fitness_entries_unchanged(self) -> None:
        """P3: legacy float fitness entries get no fingerprint dedupe."""
        from vibe_quant.discovery.pipeline import _select_diverse_top_k

        rsi = self._chrom("RSI")
        cci = self._chrom("CCI")
        candidates = [(rsi, 0.9), (cci, 0.9)]
        result = _select_diverse_top_k(candidates, top_k=2, min_distance=0.15)
        assert [c.uid for c, _ in result] == [rsi.uid, cci.uid]

    def test_total_trades_differs_by_one_both_kept(self) -> None:
        """P2 per-field: the fingerprint includes total_trades (differ by 1)."""
        from vibe_quant.discovery.pipeline import _select_diverse_top_k

        rsi = self._chrom("RSI")
        cci = self._chrom("CCI")
        candidates = [(rsi, _fit(trades=150)), (cci, _fit(trades=151))]
        result = _select_diverse_top_k(candidates, top_k=2, min_distance=0.15)
        assert [c.uid for c, _ in result] == [rsi.uid, cci.uid]

    def test_max_drawdown_differs_by_epsilon_both_kept(self) -> None:
        """P2 per-field: the fingerprint includes max_drawdown (differ by 1e-12)."""
        from vibe_quant.discovery.pipeline import _select_diverse_top_k

        rsi = self._chrom("RSI")
        cci = self._chrom("CCI")
        candidates = [(rsi, _fit(dd=0.1)), (cci, _fit(dd=0.1 + 1e-12))]
        result = _select_diverse_top_k(candidates, top_k=2, min_distance=0.15)
        assert [c.uid for c, _ in result] == [rsi.uid, cci.uid]

    def test_profit_factor_differs_by_epsilon_both_kept(self) -> None:
        """P2 per-field: the fingerprint includes profit_factor (differ by 1e-12)."""
        from vibe_quant.discovery.pipeline import _select_diverse_top_k

        rsi = self._chrom("RSI")
        cci = self._chrom("CCI")
        candidates = [(rsi, _fit(pf=1.8)), (cci, _fit(pf=1.8 + 1e-12))]
        result = _select_diverse_top_k(candidates, top_k=2, min_distance=0.15)
        assert [c.uid for c, _ in result] == [rsi.uid, cci.uid]

    def test_distance_rejected_candidate_does_not_record_fingerprint(self) -> None:
        """P6: a distance-rejected candidate must not poison the fingerprint set.

        Scan order A, B, C: B is structurally near A (distance-rejected, its
        fingerprint F is never selected) and C is distant from A but shares F.
        Recording B's fingerprint anyway would wrongly drop C, so the result
        must be [A, C].
        """
        from vibe_quant.discovery.pipeline import _select_diverse_top_k

        a = self._chrom("RSI")  # kept (first selected)
        b = self._chrom("RSI")  # structurally identical to A -> distance 0 < 0.15
        c = self._chrom("ATR")  # distant from A
        candidates = [
            (a, _fit(ret=0.4)),
            (b, _fit(ret=0.55)),  # distance-rejected; fingerprint F (not recorded)
            (c, _fit(ret=0.55)),  # shares F with B, differs from A -> fills slot 2
        ]
        result = _select_diverse_top_k(candidates, top_k=2, min_distance=0.15)
        assert [c.uid for c, _ in result] == [a.uid, c.uid]


# ---------------------------------------------------------------------------
# Elite preservation
# ---------------------------------------------------------------------------


class TestSplitDateRange:
    """Tests for the date range splitting utility."""

    def test_50_50_split(self) -> None:
        from vibe_quant.utils import split_date_range
        ts, te, hs, he = split_date_range("2024-01-01", "2024-05-01", 0.5)
        assert ts == "2024-01-01"
        assert te == hs  # train end == holdout start
        assert he == "2024-05-01"

    def test_invalid_ratio(self) -> None:
        from vibe_quant.utils import split_date_range
        with pytest.raises(ValueError, match="split_ratio"):
            split_date_range("2024-01-01", "2024-05-01", 0.0)
        with pytest.raises(ValueError, match="split_ratio"):
            split_date_range("2024-01-01", "2024-05-01", 1.0)


class TestHoldoutEvaluation:
    """Tests for train/test split holdout evaluation."""

    def test_holdout_results_populated_when_split_enabled(self) -> None:
        """When holdout_backtest_fn is provided, holdout_results should be populated."""
        # Simulate CLI pre-split: train=Jan-Mar, holdout=Mar-Jun
        cfg = _make_config(
            population_size=6, max_generations=2, top_k=2, elite_count=1,
            train_test_split=0.5,
            start_date="2024-01-01", end_date="2024-03-16",
            holdout_start_date="2024-03-16", holdout_end_date="2024-06-01",
        )

        def holdout_backtest(chrom: StrategyChromosome) -> dict[str, Any]:
            """Holdout returns slightly worse metrics."""
            n = len(chrom.entry_genes) + len(chrom.exit_genes)
            return {
                "sharpe_ratio": 0.8 + n * 0.05,
                "max_drawdown": 0.15,
                "profit_factor": 1.2,
                "total_trades": 80,
                "total_return": 0.1,
            }

        pipe = DiscoveryPipeline(cfg, _mock_backtest, holdout_backtest_fn=holdout_backtest)
        result = pipe.run()
        assert len(result.holdout_results) == len(result.top_strategies)
        assert result.train_dates is not None
        assert result.holdout_dates is not None
        # Train end == holdout start (pre-computed by CLI)
        assert result.train_dates[1] == result.holdout_dates[0]
        assert result.holdout_dates == ("2024-03-16", "2024-06-01")
        for hr in result.holdout_results:
            assert hr.sharpe_ratio > 0
            assert hr.total_trades > 0

    def test_no_holdout_when_split_disabled(self) -> None:
        """With train_test_split=0, no holdout evaluation occurs."""
        cfg = _make_config(population_size=6, max_generations=2, top_k=2, elite_count=1)
        pipe = DiscoveryPipeline(cfg, _mock_backtest)
        result = pipe.run()
        assert result.holdout_results == []
        assert result.train_dates is None
        assert result.holdout_dates is None

    def test_invalid_train_test_split(self) -> None:
        """Split ratio must be in [0, 1)."""
        with pytest.raises(ValueError, match="train_test_split"):
            _make_config(train_test_split=1.0)
        with pytest.raises(ValueError, match="train_test_split"):
            _make_config(train_test_split=-0.1)


class TestCrossWindowValidation:
    """Tests for cross-window shifted date validation."""

    def test_cross_window_filters_strategies(self) -> None:
        """Strategies must pass on shifted windows to be promoted."""
        call_log: list[str] = []

        def _window_backtest(start: str, end: str) -> Any:
            """Factory that returns backtest fns for different windows."""
            def bt(chrom: StrategyChromosome) -> dict[str, Any]:
                call_log.append(f"{start}-{end}")
                n = len(chrom.entry_genes) + len(chrom.exit_genes)
                return {
                    "sharpe_ratio": 1.5 + n * 0.1,
                    "max_drawdown": 0.1,
                    "profit_factor": 1.8,
                    "total_trades": 120,
                    "total_return": 0.3,
                }
            return bt

        cfg = _make_config(
            population_size=6, max_generations=2, top_k=2, elite_count=1,
            cross_window_months=[1, 2],
        )
        pipe = DiscoveryPipeline(
            cfg, _mock_backtest, backtest_fn_factory=_window_backtest,
        )
        result = pipe.run()

        # Should have cross-window results for each top strategy
        assert len(result.cross_window_results) > 0
        for cwr in result.cross_window_results:
            # Only the 2 SHIFTED windows count (in-sample window excluded)
            assert cwr.total_windows == 2
            assert len(cwr.window_results) == 2
            # Shifted windows stay inside the train range (end clipped)
            assert cwr.window_dates == [
                ("2024-02-01", "2024-06-01"),
                ("2024-03-01", "2024-06-01"),
            ]

        # Factory should have been called for shifted windows
        assert len(call_log) > 0

    def test_cross_window_rejects_bad_strategies(self) -> None:
        """Strategies that fail on shifted windows should be rejected."""
        call_count = [0]

        def _failing_factory(start: str, end: str) -> Any:
            """Factory returning backtest that fails on shifted windows."""
            def bt(chrom: StrategyChromosome) -> dict[str, Any]:
                call_count[0] += 1
                return {
                    "sharpe_ratio": 0.1,  # Below min_sharpe
                    "max_drawdown": 0.5,
                    "profit_factor": 0.8,
                    "total_trades": 30,
                    "total_return": -0.1,  # Negative return
                }
            return bt

        cfg = _make_config(
            population_size=6, max_generations=2, top_k=2, elite_count=1,
            cross_window_months=[1],  # One shifted window
            cross_window_min_pass=2,  # Need 2/2 to pass
        )
        pipe = DiscoveryPipeline(
            cfg, _mock_backtest, backtest_fn_factory=_failing_factory,
        )
        result = pipe.run()

        # Every strategy fails the shifted window -> 0 champions (fail closed,
        # never "keeping originals"), each with a cross-window rejection
        assert result.top_strategies == []
        assert result.cross_window_results == []
        stages = {r["stage"] for r in result.guardrail_rejections}
        assert "cross_window" in stages

    def test_no_cross_window_when_disabled(self) -> None:
        """No cross-window when cross_window_months is empty."""
        cfg = _make_config(population_size=6, max_generations=2, top_k=2, elite_count=1)
        pipe = DiscoveryPipeline(cfg, _mock_backtest)
        result = pipe.run()
        assert result.cross_window_results == []


class TestWarmStart:
    """Tests for warm-starting GA with seed chromosomes."""

    def test_seed_chromosomes_in_initial_population(self) -> None:
        """Seed chromosomes should appear in the initial population."""
        from vibe_quant.discovery.operators import ConditionType, Direction, StrategyGene

        seed_gene = StrategyGene(
            indicator_type="CCI", parameters={"period": 20.0},
            condition=ConditionType.GT, threshold=100.0,
        )
        seed = StrategyChromosome(
            entry_genes=[seed_gene], exit_genes=[seed_gene.clone()],
            stop_loss_pct=3.0, take_profit_pct=6.0, direction=Direction.SHORT,
        )

        pop = initialize_population(10, seed_chromosomes=[seed])
        # First chromosome should be the seed (cloned)
        assert pop[0].entry_genes[0].indicator_type == "CCI"
        assert pop[0].stop_loss_pct == 3.0
        assert len(pop) == 10

    def test_seed_capped_at_half_population(self) -> None:
        """Seeds can't exceed half the population."""
        seeds = [initialize_population(1)[0] for _ in range(10)]
        pop = initialize_population(6, seed_chromosomes=seeds)
        assert len(pop) == 6
        # At most 3 seeds (6 // 2)

    def test_chromosome_round_trip_serialization(self) -> None:
        """chromosome_to_serializable → serializable_to_chromosome should round-trip."""
        from vibe_quant.discovery.genome import (
            chromosome_to_serializable,
            serializable_to_chromosome,
        )
        from vibe_quant.discovery.operators import ConditionType, Direction, StrategyGene

        gene = StrategyGene(
            indicator_type="STOCH", parameters={"k_period": 14.0, "d_period": 3.0},
            condition=ConditionType.LT, threshold=20.0, sub_value="k",
        )
        original = StrategyChromosome(
            entry_genes=[gene], exit_genes=[gene.clone()],
            stop_loss_pct=2.5, take_profit_pct=4.0, direction=Direction.SHORT,
        )
        d = chromosome_to_serializable(original)
        restored = serializable_to_chromosome(d)

        assert restored.entry_genes[0].indicator_type == "STOCH"
        assert restored.entry_genes[0].parameters == {"k_period": 14.0, "d_period": 3.0}
        assert restored.entry_genes[0].condition == ConditionType.LT
        assert restored.entry_genes[0].threshold == 20.0
        assert restored.entry_genes[0].sub_value == "k"
        assert restored.stop_loss_pct == 2.5
        assert restored.take_profit_pct == 4.0
        assert restored.direction == Direction.SHORT


class TestWFARollingValidation:
    """Tests for WFA rolling OOS validation."""

    def test_wfa_rolling_produces_results(self) -> None:
        """WFA rolling should evaluate on multiple OOS windows."""
        def _factory(start: str, end: str) -> Any:
            def bt(chrom: StrategyChromosome) -> dict[str, Any]:
                return {
                    "sharpe_ratio": 1.5,
                    "max_drawdown": 0.1,
                    "profit_factor": 1.8,
                    "total_trades": 80,
                    "total_return": 0.15,
                }
            return bt

        cfg = _make_config(
            population_size=6, max_generations=2, top_k=2, elite_count=1,
            train_test_split=0.5,
            start_date="2024-01-01", end_date="2024-03-16",
            holdout_start_date="2024-03-16", holdout_end_date="2024-06-01",
            wfa_oos_step_days=30,  # ~1 month windows
            wfa_min_consistency=0.5,
        )
        pipe = DiscoveryPipeline(
            cfg, _mock_backtest,
            holdout_backtest_fn=_mock_backtest,
            backtest_fn_factory=_factory,
        )
        result = pipe.run()

        assert len(result.wfa_results) > 0
        for wfa in result.wfa_results:
            assert wfa.total_windows > 0
            assert len(wfa.oos_windows) == wfa.total_windows

    def test_wfa_rolling_exposes_sharpe_consistency(self) -> None:
        """Sharpe-based consistency counted separately from return-based.

        Window 0: return +0.15 AND sharpe +1.5 → both count.
        Window 1: return -0.01 BUT sharpe +1.1 → only sharpe+ counts.
        Window 2: return -0.02 AND sharpe -0.5 → neither.
        """
        call_count = [0]
        def _factory(start: str, end: str) -> Any:
            def bt(chrom: StrategyChromosome) -> dict[str, Any]:
                i = call_count[0] % 3
                call_count[0] += 1
                profiles = [
                    (1.5, 0.15),
                    (1.1, -0.01),
                    (-0.5, -0.02),
                ]
                sr, ret = profiles[i]
                return {"sharpe_ratio": sr, "max_drawdown": 0.1,
                        "profit_factor": 1.2, "total_trades": 80, "total_return": ret}
            return bt

        # 91-day TRAIN range -> 3 rolling 30d windows (WFA runs on the train
        # range; the holdout is reserved for the final gate)
        cfg = _make_config(
            population_size=6, max_generations=2, top_k=1, elite_count=1,
            start_date="2024-01-01", end_date="2024-04-01",
            wfa_oos_step_days=30, wfa_min_consistency=0.5,
        )
        pipe = DiscoveryPipeline(cfg, _mock_backtest, backtest_fn_factory=_factory)
        top = [(initialize_population(1)[0], _fit())]
        results, error = pipe._evaluate_wfa_rolling(top, cfg.start_date, cfg.end_date)

        assert error is None
        assert len(results) == 1
        wfa = results[0]
        assert wfa.total_windows == 3
        assert wfa.windows_profitable == 1
        assert wfa.windows_sharpe_positive == 2
        assert wfa.consistency == pytest.approx(1 / 3)
        assert wfa.sharpe_consistency == pytest.approx(2 / 3)
        # Gate still return-based: 1/3 < 0.5 → FAIL
        assert wfa.passed is False

    def test_wfa_runs_on_train_range_without_split(self) -> None:
        """WFA windows tile the TRAIN range, so no holdout split is needed."""
        seen: list[tuple[str, str]] = []

        def _factory(start: str, end: str) -> Any:
            def bt(chrom: StrategyChromosome) -> dict[str, Any]:
                seen.append((start, end))
                return {"sharpe_ratio": 1.5, "max_drawdown": 0.1,
                        "profit_factor": 1.8, "total_trades": 80, "total_return": 0.15}
            return bt

        cfg = _make_config(
            population_size=6, max_generations=2, top_k=2, elite_count=1,
            wfa_oos_step_days=30,  # train_test_split defaults to 0
        )
        result = DiscoveryPipeline(cfg, _mock_backtest, backtest_fn_factory=_factory).run()
        assert result.wfa_results
        assert seen
        assert all(cfg.start_date <= s and e <= cfg.end_date for s, e in seen)

    def test_wfa_skip_warns_when_no_backtest_factory(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """WFA requested with split but no backtest_fn_factory → warning."""
        cfg = _make_config(
            population_size=6, max_generations=2, top_k=2, elite_count=1,
            train_test_split=0.5,
            start_date="2024-01-01", end_date="2024-03-16",
            holdout_start_date="2024-03-16", holdout_end_date="2024-06-01",
            wfa_oos_step_days=30,
        )
        # No backtest_fn_factory passed.
        pipe = DiscoveryPipeline(cfg, _mock_backtest, holdout_backtest_fn=_mock_backtest)
        with caplog.at_level("WARNING", logger="vibe_quant.discovery.pipeline"):
            pipe.run()
        assert any(
            "WFA requested" in r.message and "backtest_fn_factory" in r.message
            for r in caplog.records
        ), f"expected WFA skip warning for missing factory, got: {[r.message for r in caplog.records]}"

    def test_wfa_skip_warns_when_no_top_strategies(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """WFA requested with everything wired but no top strategies → warning."""
        def _factory(start: str, end: str) -> Any:
            def bt(chrom: StrategyChromosome) -> dict[str, Any]:
                return {"sharpe_ratio": 1.5, "max_drawdown": 0.1,
                        "profit_factor": 1.8, "total_trades": 80, "total_return": 0.15}
            return bt

        cfg = _make_config(
            population_size=6, max_generations=2, top_k=2, elite_count=1,
            min_trades=9999,  # guarantees 0 strategies pass
            train_test_split=0.5,
            start_date="2024-01-01", end_date="2024-03-16",
            holdout_start_date="2024-03-16", holdout_end_date="2024-06-01",
            wfa_oos_step_days=30,
        )
        pipe = DiscoveryPipeline(
            cfg, _mock_backtest_few_trades,
            holdout_backtest_fn=_mock_backtest_few_trades,
            backtest_fn_factory=_factory,
        )
        with caplog.at_level("WARNING", logger="vibe_quant.discovery.pipeline"):
            result = pipe.run()
        assert len(result.top_strategies) == 0
        assert any(
            "WFA requested" in r.message and "no top strategies" in r.message
            for r in caplog.records
        ), f"expected WFA skip warning for empty top strategies, got: {[r.message for r in caplog.records]}"


class TestIndicatorPoolFilter:
    @pytest.fixture(autouse=True)
    def _restore_pool(self) -> Any:
        from vibe_quant.discovery.operators import _INDICATOR_NAMES, INDICATOR_POOL

        saved = dict(INDICATOR_POOL), list(_INDICATOR_NAMES)
        yield
        INDICATOR_POOL.clear()
        INDICATOR_POOL.update(saved[0])
        _INDICATOR_NAMES.clear()
        _INDICATOR_NAMES.extend(saved[1])

    def test_unknown_indicator_raises(self) -> None:
        cfg = _make_config(indicator_pool=["RSI", "NOPE"])
        pipe = DiscoveryPipeline(cfg, _mock_backtest)
        with pytest.raises(ValueError, match="not available for GA discovery"):
            pipe._apply_indicator_pool_filter()

    def test_known_subset_filters_cleanly(self) -> None:
        cfg = _make_config(indicator_pool=["RSI", "NATR"])
        pipe = DiscoveryPipeline(cfg, _mock_backtest)
        pipe._apply_indicator_pool_filter()
        from vibe_quant.discovery.operators import INDICATOR_POOL

        assert set(INDICATOR_POOL.keys()) == {"RSI", "NATR"}

    def test_empty_indicator_pool_raises(self) -> None:
        """P3: an empty pool is refused by the shared validator (never a silent clear)."""
        cfg = _make_config(indicator_pool=[])
        pipe = DiscoveryPipeline(cfg, _mock_backtest)
        with pytest.raises(
            ValueError, match="indicator_pool must not be empty; omit it \\(null\\)"
        ):
            pipe._apply_indicator_pool_filter()

    def test_pool_filter_uses_shared_validator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P5: ``_apply_indicator_pool_filter`` validates via the shared validator."""
        from vibe_quant.discovery import operators

        seen: list[list[str]] = []
        real = operators.validate_indicator_pool

        def spy(names: object) -> None:
            seen.append(list(names))  # type: ignore[call-overload]
            real(names)  # type: ignore[arg-type]

        monkeypatch.setattr(operators, "validate_indicator_pool", spy)
        cfg = _make_config(indicator_pool=["RSI"])
        DiscoveryPipeline(cfg, _mock_backtest)._apply_indicator_pool_filter()
        assert seen == [["RSI"]]


class TestElitePreservation:
    def test_elites_present_in_next_generation(self) -> None:
        cfg = _make_config(population_size=8, elite_count=2, use_crowding=False)
        pipe = DiscoveryPipeline(cfg, _mock_backtest)
        pop = initialize_population(8)
        from vibe_quant.discovery.fitness import evaluate_population

        fitness = evaluate_population(pop, _mock_backtest)  # type: ignore[arg-type]
        scores = [fr.adjusted_score for fr in fitness]
        # Identify elite indices
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        elite_slp = [pop[ranked[0]].stop_loss_pct, pop[ranked[1]].stop_loss_pct]

        new_pop, known = pipe._evolve_generation(pop, fitness)
        new_slp = [c.stop_loss_pct for c in new_pop[:2]]
        # Elites keep their (known) fitness -- no re-evaluation next gen
        assert known[0] is fitness[ranked[0]]
        assert known[1] is fitness[ranked[1]]
        # Elites are cloned into first positions
        assert new_slp == elite_slp


class TestEvaluationFailure:
    def test_all_evaluations_raise_aborts_run(self) -> None:
        def bt(_: StrategyChromosome) -> dict[str, Any]:
            raise RuntimeError("boom")

        pipe = DiscoveryPipeline(_make_config(population_size=6, max_generations=3, max_workers=1), bt)
        with pytest.raises(DiscoveryEvaluationError, match="boom"):
            pipe.run()

    def test_single_crashing_genome_scores_zero_run_continues(self) -> None:
        calls = {"n": 0}

        def bt(chrom: StrategyChromosome) -> dict[str, Any]:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("one bad genome")
            return _mock_backtest(chrom)

        pipe = DiscoveryPipeline(
            _make_config(population_size=6, max_generations=2, elite_count=1, max_workers=1), bt
        )
        result = pipe.run()
        assert isinstance(result, DiscoveryResult)
        assert result.total_candidates_evaluated >= 6
        errored = [fr for fr in pipe._fitness_cache.values() if fr.error]
        assert len(errored) == 1
        assert errored[0].adjusted_score == 0.0
        assert "one bad genome" in (errored[0].error or "")


class TestAbortThreshold:
    """Batch-abort rule: any all-failed batch before a success; only a large one after."""

    @staticmethod
    def _pipe(fail: dict[str, bool]) -> DiscoveryPipeline:
        def bt(chrom: StrategyChromosome) -> dict[str, Any]:
            if fail["on"]:
                raise RuntimeError("genome-specific")
            return _mock_backtest(chrom)

        return DiscoveryPipeline(_make_config(population_size=20, max_workers=1), bt)

    @staticmethod
    def _fresh(n: int, offset: int) -> list[StrategyChromosome]:
        return initialize_population(offset + n)[offset:]

    def test_small_failed_batch_after_success_does_not_abort(self) -> None:
        fail = {"on": False}
        pipe = self._pipe(fail)
        pipe._evaluate_new(self._fresh(5, 0))
        assert pipe._ok_evals == 5
        fail["on"] = True
        # threshold = max(2, 20 // 4) = 5; a batch of 4 stays below it
        out = pipe._evaluate_new(self._fresh(4, 5))
        assert all(fr.error for fr in out)

    def test_large_failed_batch_after_success_aborts(self) -> None:
        fail = {"on": False}
        pipe = self._pipe(fail)
        pipe._evaluate_new(self._fresh(5, 0))
        fail["on"] = True
        with pytest.raises(DiscoveryEvaluationError, match=r"ok_evals=5"):
            pipe._evaluate_new(self._fresh(5, 5))

    def test_single_failed_batch_before_any_success_aborts(self) -> None:
        fail = {"on": True}
        pipe = self._pipe(fail)
        with pytest.raises(DiscoveryEvaluationError, match=r"ok_evals=0"):
            pipe._evaluate_new(self._fresh(1, 0))


class TestLogGenerationBestLine:
    """The per-generation "Best" line must not print sentinel metrics as real.

    Run 927 printed ``Best: uid=... sharpe=-1.00 pf=0.00 dd=100.0% ...`` for a
    generation in which every genome scored 0: the tie-broken first genome's
    untouched/worst-window metrics, presented as if it were the winner.
    """

    @staticmethod
    def _log(caplog: Any, pipe: DiscoveryPipeline, scores: list[float]) -> list[str]:
        pop = initialize_population(len(scores))
        frs = [_fit(score=s) for s in scores]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        gen_res = GenerationResult(
            generation=0,
            best_fitness=max(scores),
            mean_fitness=sum(scores) / len(scores),
            worst_fitness=min(scores),
            best_chromosome=pop[best_idx],
            population_size=len(scores),
            num_passed_filters=0,
        )
        with caplog.at_level(logging.INFO, logger="vibe_quant.discovery.pipeline"):
            pipe._log_generation(
                gen=0,
                population=pop,
                fitness_results=frs,
                generation_results=[gen_res],
                best_idx=best_idx,
                gen_elapsed=1.0,
                total_elapsed=1.0,
                evals_this_gen=len(scores),
            )
        return [r.getMessage() for r in caplog.records]

    def test_all_zero_scores_log_none_instead_of_a_fake_best(self, caplog: Any) -> None:
        pipe = DiscoveryPipeline(_make_config(population_size=4), _mock_backtest)
        msgs = self._log(caplog, pipe, [0.0, 0.0, 0.0, 0.0])
        assert "  Best: none (all 4 genomes scored 0)" in msgs
        assert not any(m.startswith("  Best: uid=") for m in msgs)
        assert not any(m.startswith("  Score breakdown:") for m in msgs)

    def test_one_positive_score_keeps_the_real_best_and_breakdown(self, caplog: Any) -> None:
        pipe = DiscoveryPipeline(_make_config(population_size=4), _mock_backtest)
        msgs = self._log(caplog, pipe, [0.0, 0.6, 0.0, 0.0])
        assert "  Best: none (all 4 genomes scored 0)" not in msgs
        assert any(m.startswith("  Best: uid=") for m in msgs)
        assert any(m.startswith("  Score breakdown:") for m in msgs)
