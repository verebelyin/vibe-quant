"""Tests for fitness evaluation module."""

from __future__ import annotations

from typing import Any

import pytest

from vibe_quant.discovery.fitness import (
    COMPLEXITY_PENALTY_CAP,
    MIN_TRADES,
    FitnessResult,
    compute_complexity_penalty,
    compute_fitness_score,
    evaluate_population,
    pareto_dominates,
    pareto_rank,
)
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gene(indicator: str = "RSI", period: int = 14) -> StrategyGene:
    return StrategyGene(
        indicator_type=indicator,
        parameters={"period": float(period)},
        condition=ConditionType.CROSSES_ABOVE,
        threshold=30.0,
    )


def _make_chromosome(n_entry: int = 1, n_exit: int = 1) -> StrategyChromosome:
    return StrategyChromosome(
        entry_genes=[_make_gene() for _ in range(n_entry)],
        exit_genes=[_make_gene() for _ in range(n_exit)],
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        direction=Direction.LONG,
    )


def _make_result(
    sharpe: float = 1.5,
    max_dd: float = 0.2,
    pf: float = 1.8,
    trades: int = 100,
    penalty: float = 0.0,
    raw: float = 0.5,
    adjusted: float = 0.5,
    passed: bool = True,
    filters: dict[str, bool] | None = None,
) -> FitnessResult:
    return FitnessResult(
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        profit_factor=pf,
        total_trades=trades,
        complexity_penalty=penalty,
        raw_score=raw,
        adjusted_score=adjusted,
        passed_filters=passed,
        filter_results=filters or {},
    )


# =============================================================================
# compute_fitness_score
# =============================================================================


class TestComputeFitnessScore:
    def test_known_values(self) -> None:
        # Sharpe=2.0 -> clamp [-1,4], norm = (2-(-1))/(4-(-1)) = 3/5 = 0.6
        # MaxDD=0.2 -> dd_norm = 1 - 0.2 = 0.8
        # PF=2.5 -> norm = 2.5/5 = 0.5
        # Score = 0.4*0.6 + 0.3*0.8 + 0.3*0.5 = 0.24 + 0.24 + 0.15 = 0.63
        score = compute_fitness_score(sharpe_ratio=2.0, max_drawdown=0.2, profit_factor=2.5)
        assert score == pytest.approx(0.63, abs=1e-6)

    def test_perfect_scores(self) -> None:
        # Sharpe=4, MaxDD=0, PF=5 => all components = 1.0 => score = 1.0
        score = compute_fitness_score(sharpe_ratio=4.0, max_drawdown=0.0, profit_factor=5.0)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_worst_scores(self) -> None:
        # Sharpe=-1, MaxDD=1.0, PF=0 => all components = 0.0 => score = 0.0
        score = compute_fitness_score(sharpe_ratio=-1.0, max_drawdown=1.0, profit_factor=0.0)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_negative_sharpe_clamped(self) -> None:
        # Sharpe=-5 clamped to -1
        score_low = compute_fitness_score(sharpe_ratio=-5.0, max_drawdown=0.5, profit_factor=1.0)
        score_min = compute_fitness_score(sharpe_ratio=-1.0, max_drawdown=0.5, profit_factor=1.0)
        assert score_low == pytest.approx(score_min, abs=1e-6)

    def test_high_sharpe_clamped(self) -> None:
        # Sharpe=10 clamped to 4
        score_high = compute_fitness_score(sharpe_ratio=10.0, max_drawdown=0.5, profit_factor=1.0)
        score_max = compute_fitness_score(sharpe_ratio=4.0, max_drawdown=0.5, profit_factor=1.0)
        assert score_high == pytest.approx(score_max, abs=1e-6)

    def test_high_pf_clamped(self) -> None:
        score_high = compute_fitness_score(sharpe_ratio=1.0, max_drawdown=0.5, profit_factor=20.0)
        score_max = compute_fitness_score(sharpe_ratio=1.0, max_drawdown=0.5, profit_factor=5.0)
        assert score_high == pytest.approx(score_max, abs=1e-6)

    def test_score_is_between_0_and_1(self) -> None:
        score = compute_fitness_score(sharpe_ratio=1.0, max_drawdown=0.3, profit_factor=1.5)
        assert 0.0 <= score <= 1.0


# =============================================================================
# compute_complexity_penalty
# =============================================================================


class TestComplexityPenalty:
    def test_no_penalty_at_two_genes(self) -> None:
        assert compute_complexity_penalty(2) == 0.0

    def test_no_penalty_below_threshold(self) -> None:
        assert compute_complexity_penalty(1) == 0.0
        assert compute_complexity_penalty(0) == 0.0

    def test_penalty_three_genes(self) -> None:
        # (3-2) * 0.02 = 0.02
        assert compute_complexity_penalty(3) == pytest.approx(0.02)

    def test_penalty_five_genes(self) -> None:
        # (5-2) * 0.02 = 0.06
        assert compute_complexity_penalty(5) == pytest.approx(0.06)

    def test_penalty_capped(self) -> None:
        # (10-2) * 0.02 = 0.16 -> capped at 0.1
        assert compute_complexity_penalty(10) == pytest.approx(COMPLEXITY_PENALTY_CAP)

    def test_cap_boundary(self) -> None:
        # Exact cap: (7-2)*0.02 = 0.10 = cap
        assert compute_complexity_penalty(7) == pytest.approx(COMPLEXITY_PENALTY_CAP)
        # One below cap: (6-2)*0.02 = 0.08
        assert compute_complexity_penalty(6) == pytest.approx(0.08)


# =============================================================================
# Minimum trade filter (tested via evaluate_population)
# =============================================================================


class TestMinTradeFilter:
    def test_below_minimum_gets_zero_score(self) -> None:
        chrom = _make_chromosome()

        def bt_fn(_: StrategyChromosome) -> dict[str, Any]:
            return {
                "sharpe_ratio": 3.0,
                "max_drawdown": 0.1,
                "profit_factor": 3.0,
                "total_trades": MIN_TRADES - 1,
            }

        results = evaluate_population([chrom], bt_fn)
        assert results[0].adjusted_score == 0.0
        # raw_score should still be computed
        assert results[0].raw_score > 0.0

    def test_at_minimum_gets_nonzero(self) -> None:
        chrom = _make_chromosome()

        def bt_fn(_: StrategyChromosome) -> dict[str, Any]:
            return {
                "sharpe_ratio": 2.0,
                "max_drawdown": 0.2,
                "profit_factor": 2.0,
                "total_trades": MIN_TRADES,
            }

        results = evaluate_population([chrom], bt_fn)
        assert results[0].adjusted_score > 0.0

    def test_zero_trades(self) -> None:
        chrom = _make_chromosome()

        def bt_fn(_: StrategyChromosome) -> dict[str, Any]:
            return {
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0,
            }

        results = evaluate_population([chrom], bt_fn)
        assert results[0].adjusted_score == 0.0


# =============================================================================
# Pareto dominance
# =============================================================================


class TestParetoDominates:
    def test_a_dominates_b(self) -> None:
        a = _make_result(sharpe=2.0, max_dd=0.1, pf=3.0)
        b = _make_result(sharpe=1.0, max_dd=0.2, pf=2.0)
        assert pareto_dominates(a, b) is True

    def test_b_not_dominated_when_better_in_one(self) -> None:
        a = _make_result(sharpe=2.0, max_dd=0.3, pf=3.0)
        b = _make_result(sharpe=1.0, max_dd=0.1, pf=2.0)
        # a has worse drawdown than b, so a does not dominate b
        assert pareto_dominates(a, b) is False

    def test_equal_does_not_dominate(self) -> None:
        a = _make_result(sharpe=2.0, max_dd=0.2, pf=3.0)
        b = _make_result(sharpe=2.0, max_dd=0.2, pf=3.0)
        assert pareto_dominates(a, b) is False

    def test_strictly_worse(self) -> None:
        a = _make_result(sharpe=0.5, max_dd=0.5, pf=0.5)
        b = _make_result(sharpe=2.0, max_dd=0.1, pf=3.0)
        assert pareto_dominates(a, b) is False
        assert pareto_dominates(b, a) is True

    def test_one_better_rest_equal(self) -> None:
        a = _make_result(sharpe=2.0, max_dd=0.2, pf=3.1)
        b = _make_result(sharpe=2.0, max_dd=0.2, pf=3.0)
        assert pareto_dominates(a, b) is True
        assert pareto_dominates(b, a) is False


# =============================================================================
# Pareto ranking
# =============================================================================


class TestParetoRank:
    def test_empty_population(self) -> None:
        assert pareto_rank([]) == []

    def test_single_individual(self) -> None:
        r = _make_result(sharpe=1.0, max_dd=0.2, pf=1.5)
        assert pareto_rank([r]) == [0]

    def test_two_nondominated(self) -> None:
        # a better sharpe, b better drawdown -- neither dominates
        a = _make_result(sharpe=3.0, max_dd=0.5, pf=2.0)
        b = _make_result(sharpe=1.0, max_dd=0.1, pf=2.0)
        ranks = pareto_rank([a, b])
        assert ranks == [0, 0]

    def test_clear_fronts(self) -> None:
        # best dominates mid, mid dominates worst
        best = _make_result(sharpe=3.0, max_dd=0.1, pf=3.0)
        mid = _make_result(sharpe=2.0, max_dd=0.2, pf=2.0)
        worst = _make_result(sharpe=1.0, max_dd=0.3, pf=1.0)
        ranks = pareto_rank([worst, best, mid])
        # best=idx1 -> rank 0, mid=idx2 -> rank 1, worst=idx0 -> rank 2
        assert ranks[0] == 2  # worst
        assert ranks[1] == 0  # best
        assert ranks[2] == 1  # mid

    def test_three_strategies_mixed(self) -> None:
        # a: best sharpe, worst dd
        # b: mid everything
        # c: worst sharpe, best dd
        a = _make_result(sharpe=3.0, max_dd=0.5, pf=2.0)
        b = _make_result(sharpe=2.0, max_dd=0.3, pf=2.5)
        c = _make_result(sharpe=1.0, max_dd=0.1, pf=1.0)
        ranks = pareto_rank([a, b, c])
        # b dominates neither a (a has better sharpe) nor c (c has better dd)
        # a doesn't dominate b (b has better dd)
        # c doesn't dominate a or b (worse sharpe, worse/equal pf)
        # Actually: b has sharpe=2 > c=1, dd=0.3<0.5 vs a, pf=2.5 > a=2.0
        # b dominates a? sharpe 2<3 nope. So all on front 0? Let's check:
        # a vs b: sharpe 3>2 but dd 0.5>0.3 (a worse) -> no domination
        # a vs c: sharpe 3>1, dd 0.5>0.1 (a worse) -> no domination
        # b vs c: sharpe 2>1, dd 0.3>0.1 (b worse) -> no domination
        # All non-dominated = front 0
        assert ranks == [0, 0, 0]


# =============================================================================
# evaluate_population
# =============================================================================


class TestEvaluatePopulation:
    def test_basic_evaluation(self) -> None:
        chrom = _make_chromosome(n_entry=2, n_exit=1)  # 3 genes

        def bt_fn(_: StrategyChromosome) -> dict[str, Any]:
            return {
                "sharpe_ratio": 2.0,
                "max_drawdown": 0.2,
                "profit_factor": 2.5,
                "total_trades": 100,
            }

        results = evaluate_population([chrom], bt_fn)
        assert len(results) == 1
        r = results[0]
        assert r.sharpe_ratio == 2.0
        assert r.max_drawdown == 0.2
        assert r.profit_factor == 2.5
        assert r.total_trades == 100
        assert r.complexity_penalty == pytest.approx(0.02)
        assert r.raw_score == pytest.approx(0.63, abs=1e-6)
        assert r.adjusted_score == pytest.approx(0.63 - 0.02, abs=1e-6)
        assert r.passed_filters is True

    def test_multiple_chromosomes(self) -> None:
        chroms = [_make_chromosome() for _ in range(5)]
        call_count = 0

        def bt_fn(_: StrategyChromosome) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {
                "sharpe_ratio": 1.0,
                "max_drawdown": 0.1,
                "profit_factor": 1.5,
                "total_trades": 60,
            }

        results = evaluate_population(chroms, bt_fn)
        assert len(results) == 5
        assert call_count == 5

    def test_complexity_penalty_applied(self) -> None:
        # 4 entry + 1 exit = 5 genes -> penalty = (5-2)*0.02 = 0.06
        chrom = _make_chromosome(n_entry=4, n_exit=1)

        def bt_fn(_: StrategyChromosome) -> dict[str, Any]:
            return {
                "sharpe_ratio": 1.0,
                "max_drawdown": 0.2,
                "profit_factor": 1.5,
                "total_trades": 100,
            }

        results = evaluate_population([chrom], bt_fn)
        assert results[0].complexity_penalty == pytest.approx(0.06)
        assert results[0].adjusted_score == pytest.approx(
            results[0].raw_score - 0.06, abs=1e-6
        )

    def test_filter_fn_integration(self) -> None:
        chrom = _make_chromosome()

        def bt_fn(_: StrategyChromosome) -> dict[str, Any]:
            return {
                "sharpe_ratio": 2.0,
                "max_drawdown": 0.1,
                "profit_factor": 3.0,
                "total_trades": 200,
            }

        def filter_fn(_c: StrategyChromosome, _bt: dict[str, Any]) -> dict[str, bool]:
            return {"DSR": True, "WFA": False, "K-Fold": True}

        results = evaluate_population([chrom], bt_fn, filter_fn=filter_fn)
        assert results[0].passed_filters is False
        assert results[0].filter_results == {"DSR": True, "WFA": False, "K-Fold": True}

    def test_filter_all_pass(self) -> None:
        chrom = _make_chromosome()

        def bt_fn(_: StrategyChromosome) -> dict[str, Any]:
            return {
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.2,
                "profit_factor": 2.0,
                "total_trades": 100,
            }

        def filter_fn(_c: StrategyChromosome, _bt: dict[str, Any]) -> dict[str, bool]:
            return {"DSR": True, "WFA": True}

        results = evaluate_population([chrom], bt_fn, filter_fn=filter_fn)
        assert results[0].passed_filters is True

    def test_adjusted_score_nonnegative(self) -> None:
        # Even with high penalty, score should not go negative
        chrom = _make_chromosome(n_entry=5, n_exit=5)  # 10 genes -> penalty=0.1

        def bt_fn(_: StrategyChromosome) -> dict[str, Any]:
            return {
                "sharpe_ratio": -0.5,
                "max_drawdown": 0.8,
                "profit_factor": 0.3,
                "total_trades": 100,
            }

        results = evaluate_population([chrom], bt_fn)
        assert results[0].adjusted_score >= 0.0


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    def test_negative_sharpe(self) -> None:
        score = compute_fitness_score(sharpe_ratio=-0.5, max_drawdown=0.5, profit_factor=0.8)
        assert 0.0 <= score <= 1.0

    def test_zero_everything(self) -> None:
        score = compute_fitness_score(sharpe_ratio=0.0, max_drawdown=0.0, profit_factor=0.0)
        # Sharpe=0 norm=(0-(-1))/5=0.2, DD=1.0, PF=0.0
        # 0.4*0.2 + 0.3*1.0 + 0.3*0.0 = 0.08 + 0.3 = 0.38
        assert score == pytest.approx(0.38, abs=1e-6)

    def test_fitness_result_is_frozen(self) -> None:
        r = _make_result()
        with pytest.raises(AttributeError):
            r.sharpe_ratio = 999.0  # type: ignore[misc]
