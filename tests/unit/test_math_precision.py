"""Numerical precision tests: verify optimized math matches reference implementations.

Each test implements the ORIGINAL (pre-optimization) formula and compares
it against the current optimized code to ensure algebraic equivalence.
"""

from __future__ import annotations

import math
import statistics

import pytest

from vibe_quant.discovery.fitness import (
    _PF_INV_RANGE,
    _SHARPE_INV_RANGE,
    DRAWDOWN_WEIGHT,
    PF_MAX,
    PF_MIN,
    PROFIT_FACTOR_WEIGHT,
    SHARPE_MAX,
    SHARPE_MIN,
    SHARPE_WEIGHT,
    FitnessResult,
    compute_fitness_score,
    pareto_dominates,
    pareto_rank,
)
from vibe_quant.overfitting.dsr import (
    DeflatedSharpeRatio,
    _norm_sf,
)
from vibe_quant.overfitting.purged_kfold import CVConfig, FoldResult, PurgedKFoldCV
from vibe_quant.overfitting.wfa import WalkForwardAnalysis, WFAConfig, WFAWindow
from vibe_quant.screening.pipeline import BacktestMetrics, compute_pareto_front

# ============================================================================
# 1. DSR: _norm_sf vs reference
# ============================================================================


def _reference_norm_sf(x: float) -> float:
    """Reference implementation: standard normal survival function.

    Uses the definition: P(Z > x) = 0.5 * erfc(x / sqrt(2))
    This is the textbook formula from probability theory.
    """
    return 0.5 * math.erfc(x / math.sqrt(2.0))


class TestNormSfPrecision:
    """Verify _norm_sf matches the textbook formula exactly."""

    @pytest.mark.parametrize(
        "x",
        [
            0.0,
            1.0,
            -1.0,
            2.0,
            -2.0,
            3.0,
            -3.0,
            0.5,
            -0.5,
            1.96,  # 95% CI
            2.576,  # 99% CI
            4.0,
            -4.0,
            6.0,  # extreme tail
            -6.0,
            10.0,  # very extreme
            -10.0,
            0.001,  # near zero
            -0.001,
        ],
    )
    def test_norm_sf_matches_reference(self, x: float) -> None:
        optimized = _norm_sf(x)
        reference = _reference_norm_sf(x)
        # Should match to machine epsilon since it's the same math
        assert optimized == pytest.approx(reference, abs=1e-15), (
            f"_norm_sf({x}) = {optimized}, reference = {reference}"
        )

    def test_norm_sf_symmetry(self) -> None:
        """P(Z > x) + P(Z > -x) = 1 for all x."""
        for x in [0.5, 1.0, 2.0, 3.0]:
            assert _norm_sf(x) + _norm_sf(-x) == pytest.approx(1.0, abs=1e-15)

    def test_norm_sf_at_zero(self) -> None:
        """P(Z > 0) = 0.5 exactly."""
        assert _norm_sf(0.0) == 0.5

    def test_norm_sf_known_values(self) -> None:
        """Check against well-known statistical values."""
        # P(Z > 1.96) ≈ 0.025
        assert _norm_sf(1.96) == pytest.approx(0.025, abs=0.0001)
        # P(Z > 2.576) ≈ 0.005
        assert _norm_sf(2.576) == pytest.approx(0.005, abs=0.0001)


# ============================================================================
# 2. DSR: _sharpe_variance algebraic simplification
# ============================================================================


def _reference_sharpe_variance(
    sharpe: float, num_observations: int, skewness: float, kurtosis: float
) -> float:
    """Original formula from Bailey & Lopez de Prado (2014).

    Var(SR) = (1 + 0.5*SR^2 - skew*SR + (kurt-3)/4 * SR^2) / (T-1)
    """
    t_minus_1 = num_observations - 1
    if t_minus_1 <= 0:
        return float("inf")

    sr_sq = sharpe**2
    numerator = 1.0 + 0.5 * sr_sq - skewness * sharpe + (kurtosis - 3) / 4.0 * sr_sq
    return numerator / t_minus_1


class TestSharpeVariancePrecision:
    """Verify the algebraic simplification of _sharpe_variance."""

    @pytest.mark.parametrize(
        "sharpe,num_obs,skew,kurt",
        [
            # Normal distribution (skew=0, kurt=3)
            (1.0, 252, 0.0, 3.0),
            (0.5, 252, 0.0, 3.0),
            (2.0, 252, 0.0, 3.0),
            (-1.0, 252, 0.0, 3.0),
            (0.0, 252, 0.0, 3.0),
            # Skewed distributions
            (1.5, 500, -0.5, 3.0),
            (1.5, 500, 0.5, 3.0),
            (1.5, 500, -2.0, 3.0),
            # Leptokurtic (fat tails)
            (1.0, 252, 0.0, 5.0),
            (1.0, 252, 0.0, 10.0),
            (1.0, 252, 0.0, 1.0),  # minimum kurtosis
            # Combined non-normality
            (2.0, 100, -1.0, 6.0),
            (0.3, 1000, 0.5, 4.0),
            # Edge cases
            (0.0, 2, 0.0, 3.0),  # minimum observations
            (4.0, 252, -3.0, 12.0),  # extreme values
            (0.001, 10000, 0.0, 3.0),  # near-zero Sharpe, many obs
        ],
    )
    def test_variance_matches_reference(
        self, sharpe: float, num_obs: int, skew: float, kurt: float
    ) -> None:
        optimized = DeflatedSharpeRatio._sharpe_variance(sharpe, num_obs, skew, kurt)
        reference = _reference_sharpe_variance(sharpe, num_obs, skew, kurt)
        assert optimized == pytest.approx(reference, rel=1e-12), (
            f"sharpe={sharpe}, T={num_obs}, skew={skew}, kurt={kurt}: "
            f"optimized={optimized}, reference={reference}"
        )

    def test_normal_distribution_simplification(self) -> None:
        """For skew=0, kurt=3: Var = (1 + 0.5*SR^2) / (T-1)."""
        for sr in [0.0, 0.5, 1.0, 2.0, -1.0]:
            result = DeflatedSharpeRatio._sharpe_variance(sr, 252, 0.0, 3.0)
            expected = (1.0 + 0.5 * sr * sr) / 251.0
            assert result == pytest.approx(expected, rel=1e-14)


# ============================================================================
# 3. DSR: _expected_max_sharpe
# ============================================================================


def _reference_expected_max_sharpe(num_trials: int) -> float:
    """Reference implementation without pre-computed constants."""
    if num_trials <= 1:
        return 0.0
    gamma = 0.5772156649015329  # Euler-Mascheroni
    log_n = math.log(num_trials)
    base = math.sqrt(2.0 * log_n)
    correction = 1.0 - gamma / log_n + (gamma**2) / (2.0 * log_n**2)
    return base * correction


class TestExpectedMaxSharpePrecision:
    """Verify _expected_max_sharpe with pre-computed constants."""

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 50, 100, 200, 500, 1000, 10000])
    def test_matches_reference(self, n: int) -> None:
        optimized = DeflatedSharpeRatio._expected_max_sharpe(n)
        reference = _reference_expected_max_sharpe(n)
        assert optimized == pytest.approx(reference, rel=1e-14), (
            f"n={n}: optimized={optimized}, reference={reference}"
        )

    def test_monotonically_increasing(self) -> None:
        """More trials → higher expected max Sharpe."""
        prev = 0.0
        for n in [2, 5, 10, 50, 100, 500, 1000]:
            val = DeflatedSharpeRatio._expected_max_sharpe(n)
            assert val > prev, f"n={n}: {val} should be > {prev}"
            prev = val

    def test_single_trial_is_zero(self) -> None:
        assert DeflatedSharpeRatio._expected_max_sharpe(1) == 0.0


# ============================================================================
# 4. DSR: Full end-to-end calculation
# ============================================================================


class TestDSREndToEnd:
    """Verify full DSR calculation produces consistent results."""

    def test_high_sharpe_many_trials(self) -> None:
        """A very high Sharpe with few trials should still be significant."""
        dsr = DeflatedSharpeRatio(significance_level=0.05)
        result = dsr.calculate(
            observed_sharpe=3.0,
            num_trials=10,
            num_observations=252,
        )
        assert result.is_significant is True
        assert result.p_value < 0.05

    def test_low_sharpe_many_trials(self) -> None:
        """A low Sharpe with many trials should NOT be significant."""
        dsr = DeflatedSharpeRatio(significance_level=0.05)
        result = dsr.calculate(
            observed_sharpe=0.5,
            num_trials=1000,
            num_observations=252,
        )
        assert result.is_significant is False
        assert result.p_value > 0.05

    def test_single_trial_no_deflation(self) -> None:
        """With 1 trial, expected max sharpe is 0, so DSR = SR / sqrt(var)."""
        dsr = DeflatedSharpeRatio()
        result = dsr.calculate(
            observed_sharpe=2.0,
            num_trials=1,
            num_observations=252,
        )
        assert result.expected_max_sharpe == 0.0
        # deflated_sharpe = (2.0 - 0) / sqrt(var)
        expected_var = (1.0 + 0.5 * 4.0) / 251.0  # normal dist
        expected_dsr = 2.0 / math.sqrt(expected_var)
        assert result.deflated_sharpe == pytest.approx(expected_dsr, rel=1e-10)

    def test_dsr_with_non_normal_returns(self) -> None:
        """Non-normal returns should change the variance.

        With few trials (SR > SR_0, positive DSR), higher variance → smaller
        deflated Sharpe → higher p-value.
        """
        dsr = DeflatedSharpeRatio()
        # Use few trials so SR=3.0 > expected_max_sharpe (positive DSR)
        normal = dsr.calculate(
            observed_sharpe=3.0, num_trials=5, num_observations=500,
            skewness=0.0, kurtosis=3.0,
        )
        fat_tails = dsr.calculate(
            observed_sharpe=3.0, num_trials=5, num_observations=500,
            skewness=-1.0, kurtosis=6.0,
        )
        # Fat tails + negative skew → higher variance
        assert fat_tails.sharpe_variance > normal.sharpe_variance
        # With positive DSR: higher variance (larger denominator) → lower DSR
        assert fat_tails.deflated_sharpe < normal.deflated_sharpe
        # Lower DSR → higher p-value (less significant)
        assert fat_tails.p_value > normal.p_value


# ============================================================================
# 5. Fitness score: inlined vs original
# ============================================================================


def _reference_clamp(value: float, lo: float, hi: float) -> float:
    """Original _clamp function."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _reference_normalize(value: float, lo: float, hi: float) -> float:
    """Original _normalize function."""
    if hi <= lo:
        return 0.0
    return (value - lo) / (hi - lo)


def _reference_fitness_score(
    sharpe_ratio: float, max_drawdown: float, profit_factor: float
) -> float:
    """Original compute_fitness_score with function calls."""
    sharpe_norm = _reference_normalize(
        _reference_clamp(sharpe_ratio, SHARPE_MIN, SHARPE_MAX), SHARPE_MIN, SHARPE_MAX
    )
    dd_norm = 1.0 - _reference_clamp(max_drawdown, 0.0, 1.0)
    pf_norm = _reference_normalize(
        _reference_clamp(profit_factor, PF_MIN, PF_MAX), PF_MIN, PF_MAX
    )
    return SHARPE_WEIGHT * sharpe_norm + DRAWDOWN_WEIGHT * dd_norm + PROFIT_FACTOR_WEIGHT * pf_norm


class TestFitnessScorePrecision:
    """Verify inlined fitness score matches original."""

    @pytest.mark.parametrize(
        "sharpe,dd,pf",
        [
            # Normal ranges
            (1.0, 0.1, 1.5),
            (2.0, 0.05, 2.0),
            (0.0, 0.5, 1.0),
            (-0.5, 0.3, 0.8),
            # At bounds
            (SHARPE_MIN, 0.0, PF_MIN),
            (SHARPE_MAX, 1.0, PF_MAX),
            # Beyond bounds (should clamp)
            (-5.0, -0.1, -1.0),
            (10.0, 1.5, 10.0),
            # Typical crypto strategy values
            (1.2, 0.15, 1.8),
            (0.8, 0.25, 1.3),
            (3.5, 0.02, 4.5),
        ],
    )
    def test_matches_reference(self, sharpe: float, dd: float, pf: float) -> None:
        optimized = compute_fitness_score(sharpe, dd, pf)
        reference = _reference_fitness_score(sharpe, dd, pf)
        assert optimized == pytest.approx(reference, abs=1e-15), (
            f"sharpe={sharpe}, dd={dd}, pf={pf}: "
            f"optimized={optimized}, reference={reference}"
        )

    def test_pre_computed_inverse_ranges(self) -> None:
        """Verify pre-computed constants match actual ranges."""
        assert pytest.approx(
            1.0 / (SHARPE_MAX - SHARPE_MIN), rel=1e-15
        ) == _SHARPE_INV_RANGE
        assert pytest.approx(
            1.0 / (PF_MAX - PF_MIN), rel=1e-15
        ) == _PF_INV_RANGE


# ============================================================================
# 6. Pareto dominance: inlined vs original
# ============================================================================


def _reference_pareto_dominates(a: FitnessResult, b: FitnessResult) -> bool:
    """Original pareto_dominates using tuple iteration."""
    a_obj = (a.sharpe_ratio, 1.0 - a.max_drawdown, a.profit_factor)
    b_obj = (b.sharpe_ratio, 1.0 - b.max_drawdown, b.profit_factor)
    return all(ai >= bi for ai, bi in zip(a_obj, b_obj, strict=True)) and any(
        ai > bi for ai, bi in zip(a_obj, b_obj, strict=True)
    )


def _make_fitness(sharpe: float, dd: float, pf: float) -> FitnessResult:
    """Helper to create FitnessResult with given objectives."""
    return FitnessResult(
        sharpe_ratio=sharpe,
        max_drawdown=dd,
        profit_factor=pf,
        total_trades=100,
        complexity_penalty=0.0,
        raw_score=0.5,
        adjusted_score=0.5,
        passed_filters=True,
        filter_results={},
    )


class TestParetoDominancePrecision:
    """Verify inlined Pareto dominance matches tuple-based original."""

    @pytest.mark.parametrize(
        "a_args,b_args,expected",
        [
            # Clear dominance
            ((2.0, 0.1, 2.0), (1.0, 0.2, 1.5), True),
            # Not dominated (b is better in one)
            ((2.0, 0.1, 1.5), (1.0, 0.2, 2.0), False),
            # Equal in all (not dominated)
            ((1.0, 0.1, 1.5), (1.0, 0.1, 1.5), False),
            # Equal in some, better in one
            ((1.0, 0.1, 2.0), (1.0, 0.1, 1.5), True),
            # Reverse
            ((1.0, 0.2, 1.5), (2.0, 0.1, 2.0), False),
            # Edge: zero values
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), False),
            ((0.1, 0.0, 0.0), (0.0, 0.0, 0.0), True),
            # Negative Sharpe
            ((-0.5, 0.1, 1.0), (-1.0, 0.2, 0.8), True),
            # Very close values (float precision test)
            ((1.0000001, 0.1, 1.5), (1.0, 0.1, 1.5), True),
            ((1.0, 0.1, 1.5), (1.0000001, 0.1, 1.5), False),
        ],
    )
    def test_matches_reference(
        self,
        a_args: tuple[float, float, float],
        b_args: tuple[float, float, float],
        expected: bool,
    ) -> None:
        a = _make_fitness(*a_args)
        b = _make_fitness(*b_args)
        optimized = pareto_dominates(a, b)
        reference = _reference_pareto_dominates(a, b)
        assert optimized == reference == expected, (
            f"a={a_args}, b={b_args}: optimized={optimized}, reference={reference}"
        )


# ============================================================================
# 7. Pareto front (screening): inlined vs original
# ============================================================================


def _reference_pareto_front(results: list[BacktestMetrics]) -> list[int]:
    """Original Pareto front using function calls and tuple comparisons."""
    if not results:
        return []
    n = len(results)
    if n == 1:
        return [0]

    def objectives(r: BacktestMetrics) -> tuple[float, float, float]:
        return (r.sharpe_ratio, 1.0 - r.max_drawdown, r.profit_factor)

    def dominates(obj_a: tuple[float, ...], obj_b: tuple[float, ...]) -> bool:
        return all(a >= b for a, b in zip(obj_a, obj_b, strict=True)) and any(
            a > b for a, b in zip(obj_a, obj_b, strict=True)
        )

    is_pareto = [True] * n
    objs = [objectives(r) for r in results]
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if dominates(objs[j], objs[i]):
                is_pareto[i] = False
                break
    return [i for i in range(n) if is_pareto[i]]


class TestParetoFrontPrecision:
    """Verify optimized Pareto front matches reference."""

    def _make_results(
        self, data: list[tuple[float, float, float]]
    ) -> list[BacktestMetrics]:
        return [
            BacktestMetrics(parameters={}, sharpe_ratio=s, max_drawdown=d, profit_factor=p)
            for s, d, p in data
        ]

    def test_simple_front(self) -> None:
        results = self._make_results([
            (2.0, 0.1, 2.0),  # Pareto
            (1.0, 0.2, 1.5),  # Dominated
            (1.5, 0.05, 1.0),  # Pareto (lower PF but better DD)
        ])
        assert compute_pareto_front(results) == _reference_pareto_front(results)

    def test_all_pareto(self) -> None:
        """No dominance → all are Pareto optimal."""
        results = self._make_results([
            (2.0, 0.3, 1.0),
            (1.0, 0.1, 2.0),
            (1.5, 0.2, 1.5),
        ])
        opt = compute_pareto_front(results)
        ref = _reference_pareto_front(results)
        assert sorted(opt) == sorted(ref)

    def test_single_result(self) -> None:
        results = self._make_results([(1.0, 0.1, 1.5)])
        assert compute_pareto_front(results) == [0]
        assert _reference_pareto_front(results) == [0]

    def test_empty(self) -> None:
        assert compute_pareto_front([]) == _reference_pareto_front([])

    def test_identical_results(self) -> None:
        """All identical → all Pareto (none dominates another)."""
        results = self._make_results([(1.0, 0.1, 1.5)] * 5)
        opt = compute_pareto_front(results)
        ref = _reference_pareto_front(results)
        assert sorted(opt) == sorted(ref)

    def test_chain_dominance(self) -> None:
        """A > B > C → only A is Pareto."""
        results = self._make_results([
            (3.0, 0.05, 3.0),  # Best in all
            (2.0, 0.10, 2.0),  # Dominated by [0]
            (1.0, 0.20, 1.0),  # Dominated by [0] and [1]
        ])
        assert compute_pareto_front(results) == _reference_pareto_front(results) == [0]

    def test_large_population(self) -> None:
        """Stress test with 200 points."""
        import random as rng

        rng.seed(42)
        data = [(rng.uniform(-1, 4), rng.uniform(0, 0.5), rng.uniform(0, 5)) for _ in range(200)]
        results = self._make_results(data)
        opt = sorted(compute_pareto_front(results))
        ref = sorted(_reference_pareto_front(results))
        assert opt == ref, f"Mismatch: opt={opt}, ref={ref}"


# ============================================================================
# 8. Pareto rank: inlined vs reference
# ============================================================================


def _reference_pareto_rank(population_fitness: list[FitnessResult]) -> list[int]:
    """Original pareto_rank using pareto_dominates function calls."""
    n = len(population_fitness)
    if n == 0:
        return []

    def ref_dominates(a: FitnessResult, b: FitnessResult) -> bool:
        a_obj = (a.sharpe_ratio, 1.0 - a.max_drawdown, a.profit_factor)
        b_obj = (b.sharpe_ratio, 1.0 - b.max_drawdown, b.profit_factor)
        return all(ai >= bi for ai, bi in zip(a_obj, b_obj, strict=True)) and any(
            ai > bi for ai, bi in zip(a_obj, b_obj, strict=True)
        )

    ranks = [-1] * n
    remaining = set(range(n))
    current_rank = 0

    while remaining:
        front = []
        remaining_list = list(remaining)
        for i in remaining_list:
            dominated = False
            for j in remaining_list:
                if i == j:
                    continue
                if ref_dominates(population_fitness[j], population_fitness[i]):
                    dominated = True
                    break
            if not dominated:
                front.append(i)

        for i in front:
            ranks[i] = current_rank
            remaining.discard(i)
        current_rank += 1

    return ranks


class TestParetoRankPrecision:
    """Verify inlined pareto_rank matches reference."""

    def test_basic_ranking(self) -> None:
        pop = [
            _make_fitness(3.0, 0.05, 3.0),  # Front 0
            _make_fitness(2.0, 0.10, 2.0),  # Front 1
            _make_fitness(1.0, 0.20, 1.0),  # Front 2
        ]
        assert pareto_rank(pop) == _reference_pareto_rank(pop)

    def test_tied_front(self) -> None:
        pop = [
            _make_fitness(2.0, 0.3, 1.0),  # Front 0
            _make_fitness(1.0, 0.1, 2.0),  # Front 0
            _make_fitness(0.5, 0.4, 0.5),  # Front 1 (dominated by both)
        ]
        opt = pareto_rank(pop)
        ref = _reference_pareto_rank(pop)
        assert opt == ref

    def test_all_equal(self) -> None:
        pop = [_make_fitness(1.0, 0.1, 1.5)] * 4
        opt = pareto_rank(pop)
        ref = _reference_pareto_rank(pop)
        assert opt == ref

    def test_empty(self) -> None:
        assert pareto_rank([]) == _reference_pareto_rank([])

    def test_stress(self) -> None:
        """50-element population (typical GA size)."""
        import random as rng

        rng.seed(123)
        pop = [
            _make_fitness(rng.uniform(-1, 4), rng.uniform(0, 0.5), rng.uniform(0, 5))
            for _ in range(50)
        ]
        opt = pareto_rank(pop)
        ref = _reference_pareto_rank(pop)
        assert opt == ref


# ============================================================================
# 9. PurgedKFold variance: single-pass vs two-pass
# ============================================================================


class TestPurgedKFoldVariancePrecision:
    """Verify single-pass variance matches statistics.stdev."""

    @pytest.mark.parametrize(
        "sharpes",
        [
            [1.0, 1.5, 0.8, 1.2, 0.9],
            [2.0, 2.0, 2.0, 2.0, 2.0],  # zero variance
            [-1.0, 0.0, 1.0, 2.0, 3.0],
            [0.001, 0.002, 0.003, 0.004, 0.005],  # very small
            [100.0, 101.0, 102.0, 103.0, 104.0],  # large similar values
            [0.0, 0.0, 0.0, 0.0, 0.0],  # all zeros
        ],
    )
    def test_variance_matches_statistics_module(self, sharpes: list[float]) -> None:
        """Compare our single-pass std against statistics.stdev."""
        fold_results = [
            FoldResult(
                fold_index=i,
                train_size=800,
                test_size=200,
                train_sharpe=2.0,
                test_sharpe=s,
                train_return=10.0,
                test_return=5.0,
            )
            for i, s in enumerate(sharpes)
        ]

        cv = PurgedKFoldCV(
            config=CVConfig(n_splits=len(sharpes), purge_pct=0.0, embargo_pct=0.0),
        )
        result = cv._aggregate_results(fold_results)

        expected_mean = statistics.mean(sharpes)
        assert result.mean_oos_sharpe == pytest.approx(expected_mean, rel=1e-12)

        if len(set(sharpes)) > 1:
            expected_std = statistics.stdev(sharpes)
            assert result.std_oos_sharpe == pytest.approx(expected_std, rel=1e-10), (
                f"sharpes={sharpes}: got std={result.std_oos_sharpe}, "
                f"expected={expected_std}"
            )
        else:
            # Constant values → std should be 0
            assert result.std_oos_sharpe == pytest.approx(0.0, abs=1e-12)

    def test_single_fold(self) -> None:
        """Single fold should have std=0."""
        fold_results = [
            FoldResult(
                fold_index=0,
                train_size=800,
                test_size=200,
                train_sharpe=2.0,
                test_sharpe=1.5,
                train_return=10.0,
                test_return=5.0,
            )
        ]
        cv = PurgedKFoldCV(
            config=CVConfig(n_splits=2, purge_pct=0.0, embargo_pct=0.0),
        )
        result = cv._aggregate_results(fold_results)
        assert result.std_oos_sharpe == 0.0


# ============================================================================
# 10. WFA: single-pass aggregation vs multi-pass reference
# ============================================================================


def _make_wfa_window(
    idx: int,
    is_sharpe: float,
    oos_sharpe: float,
    is_return: float,
    oos_return: float,
) -> WFAWindow:
    return WFAWindow(
        window_index=idx,
        is_start_date="2024-01-01",
        is_end_date="2024-09-30",
        oos_start_date="2024-10-01",
        oos_end_date="2024-12-31",
        is_sharpe=is_sharpe,
        oos_sharpe=oos_sharpe,
        is_return=is_return,
        oos_return=oos_return,
        best_params={},
    )


def _reference_aggregate_wfa(
    windows: list[WFAWindow], config: WFAConfig
) -> dict[str, float]:
    """Reference multi-pass aggregation."""
    n = len(windows)
    oos_sharpes = [w.oos_sharpe for w in windows]
    oos_returns = [w.oos_return for w in windows]
    is_returns = [w.is_return for w in windows]
    degradations = [w.sharpe_degradation for w in windows]

    agg_oos_sharpe = sum(oos_sharpes) / n
    agg_oos_return = sum(oos_returns) / n
    mean_is_return = sum(is_returns) / n
    efficiency = (
        0.0
        if mean_is_return == 0
        else (agg_oos_return / mean_is_return if agg_oos_return != 0 else 0.0)
    )
    avg_degradation = sum(degradations) / n
    consistency = sum(1 for w in windows if w.oos_return > 0) / n

    return {
        "oos_sharpe": agg_oos_sharpe,
        "oos_return": agg_oos_return,
        "efficiency": efficiency,
        "degradation": avg_degradation,
        "consistency": consistency,
    }


class TestWFAAggregationPrecision:
    """Verify single-pass WFA aggregation matches multi-pass reference."""

    def test_basic_aggregation(self) -> None:
        windows = [
            _make_wfa_window(0, 1.5, 0.8, 10.0, 5.0),
            _make_wfa_window(1, 1.2, 0.6, 8.0, 3.0),
            _make_wfa_window(2, 1.8, 1.0, 12.0, 7.0),
            _make_wfa_window(3, 1.0, 0.3, 6.0, -1.0),
        ]

        config = WFAConfig.default()
        wfa = WalkForwardAnalysis(config=config)
        result = wfa._aggregate_results(windows)
        ref = _reference_aggregate_wfa(windows, config)

        assert result.aggregated_oos_sharpe == pytest.approx(ref["oos_sharpe"], rel=1e-12)
        assert result.aggregated_oos_return == pytest.approx(ref["oos_return"], rel=1e-12)
        assert result.efficiency == pytest.approx(ref["efficiency"], rel=1e-12)
        assert result.is_vs_oos_degradation == pytest.approx(ref["degradation"], rel=1e-12)
        assert result.consistency_ratio == pytest.approx(ref["consistency"], rel=1e-12)

    def test_all_profitable(self) -> None:
        windows = [
            _make_wfa_window(i, 1.5, 0.8, 10.0, 5.0)
            for i in range(10)
        ]
        config = WFAConfig.default()
        wfa = WalkForwardAnalysis(config=config)
        result = wfa._aggregate_results(windows)
        ref = _reference_aggregate_wfa(windows, config)

        assert result.consistency_ratio == pytest.approx(1.0)
        assert result.aggregated_oos_sharpe == pytest.approx(ref["oos_sharpe"], rel=1e-12)

    def test_all_losing(self) -> None:
        windows = [
            _make_wfa_window(i, 1.5, -0.5, 10.0, -3.0)
            for i in range(8)
        ]
        config = WFAConfig.default()
        wfa = WalkForwardAnalysis(config=config)
        result = wfa._aggregate_results(windows)
        ref = _reference_aggregate_wfa(windows, config)

        assert result.consistency_ratio == pytest.approx(0.0)
        assert result.aggregated_oos_sharpe == pytest.approx(ref["oos_sharpe"], rel=1e-12)

    def test_zero_is_return(self) -> None:
        """When mean IS return is zero, efficiency should handle gracefully."""
        windows = [
            _make_wfa_window(0, 1.0, 0.5, 5.0, 3.0),
            _make_wfa_window(1, 1.0, 0.5, -5.0, 3.0),  # IS returns cancel out
        ]
        config = WFAConfig.default()
        wfa = WalkForwardAnalysis(config=config)
        result = wfa._aggregate_results(windows)
        # mean_is_return = 0, agg_oos_return = 3 → 0.0 (undefined ratio)
        assert result.efficiency == 0.0


# ============================================================================
# 11. Catalog bar aggregation divisor
# ============================================================================


class TestCatalogDivisorPrecision:
    """Verify combined ns divisor matches chained integer division."""

    @pytest.mark.parametrize("target_minutes", [5, 15, 60, 240])
    def test_divisor_equivalence(self, target_minutes: int) -> None:
        """Combined divisor must produce same grouping as chained division."""
        ns_per_target = 1_000_000 * 60_000 * target_minutes

        # Test with realistic nanosecond timestamps
        # 2024-01-01 00:00:00 UTC in nanoseconds
        base_ns = 1_704_067_200_000_000_000

        for offset_minutes in range(0, 500):
            ts = base_ns + offset_minutes * 60_000_000_000  # Add minutes in ns

            # Chained division (original method)
            chained = ts // 1_000_000 // 60_000 // target_minutes

            # Combined division (optimized method)
            combined = ts // ns_per_target

            assert chained == combined, (
                f"target={target_minutes}m, offset={offset_minutes}m: "
                f"chained={chained}, combined={combined}"
            )

    def test_divisor_value(self) -> None:
        """Verify the actual divisor value for known timeframes."""
        # 5-minute: 5 * 60 * 10^9 nanoseconds
        assert 1_000_000 * 60_000 * 5 == 300_000_000_000
        # 1 minute = 60 * 10^9 = 60_000_000_000 ns
        assert 1_000_000 * 60_000 * 1 == 60_000_000_000
        # 1 hour = 3600 * 10^9 ns
        assert 1_000_000 * 60_000 * 60 == 3_600_000_000_000


# ============================================================================
# 12. Fill model slippage formula
# ============================================================================


class TestSlippageFormulaPrecision:
    """Verify the slippage formula matches the SPEC."""

    @pytest.fixture(autouse=True)
    def _skip_without_nautilus(self) -> None:
        pytest.importorskip("nautilus_trader")

    def test_spec_formula(self) -> None:
        """slippage = spread/2 + k * volatility * sqrt(order_size / avg_volume)"""
        from vibe_quant.validation.fill_model import SlippageEstimator

        estimator = SlippageEstimator(impact_coefficient=0.1)

        k = 0.1
        vol = 0.02
        spread = 0.001
        order_size = 10.0
        avg_vol = 1000.0

        expected = spread / 2.0 + k * vol * math.sqrt(order_size / avg_vol)
        actual = estimator.calculate(order_size, avg_vol, vol, spread)

        assert actual == pytest.approx(expected, rel=1e-14)

    def test_zero_volatility_gives_half_spread(self) -> None:
        from vibe_quant.validation.fill_model import SlippageEstimator

        estimator = SlippageEstimator(impact_coefficient=0.1)
        result = estimator.calculate(10.0, 1000.0, 0.0, 0.002)
        assert result == pytest.approx(0.001, rel=1e-14)  # spread/2

    def test_zero_volume_gives_half_spread(self) -> None:
        from vibe_quant.validation.fill_model import SlippageEstimator

        estimator = SlippageEstimator(impact_coefficient=0.1)
        result = estimator.calculate(10.0, 0.0, 0.02, 0.002)
        assert result == pytest.approx(0.001, rel=1e-14)  # spread/2

    def test_slippage_scales_with_sqrt_of_size(self) -> None:
        """Doubling order size should increase impact by sqrt(2)."""
        from vibe_quant.validation.fill_model import SlippageEstimator

        estimator = SlippageEstimator(impact_coefficient=0.1)
        small = estimator.calculate(100.0, 10000.0, 0.02, 0.0)
        large = estimator.calculate(200.0, 10000.0, 0.02, 0.0)
        # With spread=0: slippage = k*vol*sqrt(size/vol)
        assert large / small == pytest.approx(math.sqrt(2.0), rel=1e-12)
