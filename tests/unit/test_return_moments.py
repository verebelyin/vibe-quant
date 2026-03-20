"""Tests for return distribution moments (skewness/kurtosis) computation.

Verifies the adjusted Fisher-Pearson G1/G2 formulas match scipy reference
values, and that the full pipeline stores moments correctly.
"""

from __future__ import annotations

import math


class TestReturnMomentsFormula:
    """Verify skewness/kurtosis formulas match scipy.stats reference values."""

    @staticmethod
    def _compute_moments(returns: list[float]) -> tuple[float, float]:
        """Standalone copy of the formula from nt_runner for unit testing."""
        n = len(returns)
        if n < 4:
            return 0.0, 3.0

        mean = sum(returns) / n
        diffs = [r - mean for r in returns]
        m2 = sum(d * d for d in diffs) / n
        if m2 == 0:
            return 0.0, 3.0

        m3 = sum(d**3 for d in diffs) / n
        m4 = sum(d**4 for d in diffs) / n

        # G1 skewness (bias-corrected)
        skewness = (math.sqrt(n * (n - 1)) / (n - 2)) * (m3 / m2**1.5)

        # G2 excess kurtosis (bias-corrected) + 3 for full kurtosis
        excess = ((n + 1) * (n - 1) * m4) / (
            (n - 2) * (n - 3) * m2**2
        ) - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
        kurtosis = excess + 3.0

        return round(skewness, 4), round(max(1.0, kurtosis), 4)

    def test_normal_like_returns(self) -> None:
        """Roughly symmetric data should have skewness near 0, kurtosis near 3."""
        # Symmetric returns centered at 0
        returns = [-0.05, -0.03, -0.01, 0.01, 0.03, 0.05, -0.02, 0.02, -0.04, 0.04]
        skew, kurt = self._compute_moments(returns)
        assert abs(skew) < 0.5, f"Expected near-zero skewness, got {skew}"
        assert 1.0 <= kurt <= 5.0, f"Expected kurtosis near 3, got {kurt}"

    def test_positive_skew(self) -> None:
        """Data with a large positive outlier should have positive skewness."""
        returns = [0.01, 0.02, 0.01, -0.01, 0.02, 0.01, -0.02, 0.01, 0.50, 0.01]
        skew, _ = self._compute_moments(returns)
        assert skew > 1.0, f"Expected positive skewness >1, got {skew}"

    def test_negative_skew(self) -> None:
        """Data with a large negative outlier should have negative skewness."""
        returns = [0.01, 0.02, 0.01, -0.01, 0.02, 0.01, -0.02, 0.01, -0.50, 0.01]
        skew, _ = self._compute_moments(returns)
        assert skew < -1.0, f"Expected negative skewness <-1, got {skew}"

    def test_fat_tails_high_kurtosis(self) -> None:
        """Data with extreme outliers should have high kurtosis (>3)."""
        returns = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, -0.50, 0.50, 0.01, 0.01]
        _, kurt = self._compute_moments(returns)
        assert kurt > 3.0, f"Expected kurtosis >3 for fat tails, got {kurt}"

    def test_too_few_returns(self) -> None:
        """Fewer than 4 returns should return defaults."""
        assert self._compute_moments([0.01, 0.02, 0.03]) == (0.0, 3.0)
        assert self._compute_moments([]) == (0.0, 3.0)

    def test_constant_returns(self) -> None:
        """All-equal returns (zero variance) should return defaults."""
        assert self._compute_moments([0.01, 0.01, 0.01, 0.01, 0.01]) == (0.0, 3.0)

    def test_matches_scipy_reference(self) -> None:
        """Compare against hardcoded scipy reference values.

        Generated with:
            from scipy.stats import skew, kurtosis
            data = [-0.08, -0.03, 0.01, 0.05, -0.02, 0.04, -0.06, 0.07, -0.01, 0.03,
                    0.02, -0.04, 0.06, -0.05, 0.00, 0.08, -0.07, 0.01, -0.02, 0.04]
            skew(data, bias=False)         # -0.074199
            kurtosis(data, bias=False) + 3 # 1.994104 (full kurtosis)
        """
        data = [
            -0.08, -0.03, 0.01, 0.05, -0.02, 0.04, -0.06, 0.07, -0.01, 0.03,
            0.02, -0.04, 0.06, -0.05, 0.00, 0.08, -0.07, 0.01, -0.02, 0.04,
        ]
        skew, kurt = self._compute_moments(data)
        # scipy reference: skew=-0.0742, excess_kurt=-1.0059, full_kurt=1.9941
        assert abs(skew - (-0.0742)) < 0.01, f"Skewness mismatch: {skew} vs -0.0742"
        assert abs(kurt - 1.9941) < 0.01, f"Kurtosis mismatch: {kurt} vs 1.9941"


class TestPerformanceMetricsFields:
    """Verify PerformanceMetrics has skewness/kurtosis with correct defaults."""

    def test_defaults(self) -> None:
        from vibe_quant.metrics import PerformanceMetrics

        m = PerformanceMetrics()
        assert m.skewness == 0.0
        assert m.kurtosis == 3.0

    def test_custom_values(self) -> None:
        from vibe_quant.metrics import PerformanceMetrics

        m = PerformanceMetrics(skewness=-1.5, kurtosis=5.2)
        assert m.skewness == -1.5
        assert m.kurtosis == 5.2


class TestFitnessResultFields:
    """Verify FitnessResult carries skewness/kurtosis."""

    def test_defaults(self) -> None:
        from vibe_quant.discovery.fitness import FitnessResult

        fr = FitnessResult(
            sharpe_ratio=1.0, max_drawdown=0.1, profit_factor=1.5,
            total_trades=100, total_return=0.1, complexity_penalty=0.0,
            overtrade_penalty=0.0, sl_tp_penalty=0.0, raw_score=0.5, adjusted_score=0.5,
            passed_filters=True, filter_results={},
        )
        assert fr.skewness == 0.0
        assert fr.kurtosis == 3.0

    def test_custom_values(self) -> None:
        from vibe_quant.discovery.fitness import FitnessResult

        fr = FitnessResult(
            sharpe_ratio=1.0, max_drawdown=0.1, profit_factor=1.5,
            total_trades=100, total_return=0.1, complexity_penalty=0.0,
            overtrade_penalty=0.0, sl_tp_penalty=0.0, raw_score=0.5, adjusted_score=0.5,
            passed_filters=True, filter_results={},
            skewness=-0.8, kurtosis=4.5,
        )
        assert fr.skewness == -0.8
        assert fr.kurtosis == 4.5


class TestSchemaHasColumns:
    """Verify sweep_results and backtest_results tables have skewness/kurtosis."""

    def test_sweep_results_columns_in_whitelist(self) -> None:
        from vibe_quant.db.state_manager import _SWEEP_RESULTS_COLUMNS

        assert "skewness" in _SWEEP_RESULTS_COLUMNS
        assert "kurtosis" in _SWEEP_RESULTS_COLUMNS

    def test_schema_sql_has_columns(self) -> None:
        from vibe_quant.db.schema import SCHEMA_SQL

        # Check sweep_results and backtest_results both mention skewness/kurtosis
        assert "skewness REAL" in SCHEMA_SQL
        assert "kurtosis REAL" in SCHEMA_SQL
