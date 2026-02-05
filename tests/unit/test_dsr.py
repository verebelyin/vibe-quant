"""Unit tests for Deflated Sharpe Ratio (DSR) module."""

import math

import pytest

from vibe_quant.overfitting.dsr import (
    EULER_MASCHERONI,
    DSRResult,
    DeflatedSharpeRatio,
    calculate_dsr,
)


class TestDeflatedSharpeRatioInit:
    """Tests for DeflatedSharpeRatio initialization."""

    def test_default_significance(self) -> None:
        """Default significance level is 0.05."""
        dsr = DeflatedSharpeRatio()
        assert dsr.significance_level == 0.05

    def test_custom_significance(self) -> None:
        """Custom significance level is accepted."""
        dsr = DeflatedSharpeRatio(significance_level=0.01)
        assert dsr.significance_level == 0.01

    def test_zero_significance_raises(self) -> None:
        """Zero significance level raises ValueError."""
        with pytest.raises(ValueError, match="significance_level must be in"):
            DeflatedSharpeRatio(significance_level=0.0)

    def test_one_significance_raises(self) -> None:
        """Significance level of 1 raises ValueError."""
        with pytest.raises(ValueError, match="significance_level must be in"):
            DeflatedSharpeRatio(significance_level=1.0)

    def test_negative_significance_raises(self) -> None:
        """Negative significance level raises ValueError."""
        with pytest.raises(ValueError, match="significance_level must be in"):
            DeflatedSharpeRatio(significance_level=-0.05)


class TestInputValidation:
    """Tests for input validation."""

    @pytest.fixture
    def dsr(self) -> DeflatedSharpeRatio:
        """Create DSR instance for tests."""
        return DeflatedSharpeRatio()

    def test_zero_trials_raises(self, dsr: DeflatedSharpeRatio) -> None:
        """Zero trials raises ValueError."""
        with pytest.raises(ValueError, match="num_trials must be >= 1"):
            dsr.calculate(observed_sharpe=1.0, num_trials=0, num_observations=100)

    def test_negative_trials_raises(self, dsr: DeflatedSharpeRatio) -> None:
        """Negative trials raises ValueError."""
        with pytest.raises(ValueError, match="num_trials must be >= 1"):
            dsr.calculate(observed_sharpe=1.0, num_trials=-5, num_observations=100)

    def test_one_observation_raises(self, dsr: DeflatedSharpeRatio) -> None:
        """One observation raises ValueError (need >= 2 for variance)."""
        with pytest.raises(ValueError, match="num_observations must be >= 2"):
            dsr.calculate(observed_sharpe=1.0, num_trials=10, num_observations=1)

    def test_zero_observations_raises(self, dsr: DeflatedSharpeRatio) -> None:
        """Zero observations raises ValueError."""
        with pytest.raises(ValueError, match="num_observations must be >= 2"):
            dsr.calculate(observed_sharpe=1.0, num_trials=10, num_observations=0)

    def test_invalid_kurtosis_raises(self, dsr: DeflatedSharpeRatio) -> None:
        """Kurtosis < 1 raises ValueError."""
        with pytest.raises(ValueError, match="kurtosis must be >= 1"):
            dsr.calculate(
                observed_sharpe=1.0,
                num_trials=10,
                num_observations=100,
                kurtosis=0.5,
            )


class TestExpectedMaxSharpe:
    """Tests for expected maximum Sharpe calculation."""

    @pytest.fixture
    def dsr(self) -> DeflatedSharpeRatio:
        """Create DSR instance."""
        return DeflatedSharpeRatio()

    def test_single_trial_no_bias(self, dsr: DeflatedSharpeRatio) -> None:
        """Single trial has no multiple testing bias."""
        result = dsr.calculate(observed_sharpe=1.0, num_trials=1, num_observations=252)
        assert result.expected_max_sharpe == 0.0

    def test_more_trials_higher_expected_max(self, dsr: DeflatedSharpeRatio) -> None:
        """More trials means higher expected max Sharpe."""
        result_10 = dsr.calculate(
            observed_sharpe=1.0, num_trials=10, num_observations=252
        )
        result_100 = dsr.calculate(
            observed_sharpe=1.0, num_trials=100, num_observations=252
        )
        result_1000 = dsr.calculate(
            observed_sharpe=1.0, num_trials=1000, num_observations=252
        )

        assert result_100.expected_max_sharpe > result_10.expected_max_sharpe
        assert result_1000.expected_max_sharpe > result_100.expected_max_sharpe

    def test_expected_max_sharpe_reasonable_magnitude(
        self, dsr: DeflatedSharpeRatio
    ) -> None:
        """Expected max Sharpe should be in reasonable range for typical N."""
        # For N=100, expected max ~= sqrt(2*ln(100)) ~= 3.03
        result = dsr.calculate(observed_sharpe=1.0, num_trials=100, num_observations=252)
        assert 2.5 < result.expected_max_sharpe < 3.5

        # For N=1000, expected max ~= sqrt(2*ln(1000)) ~= 3.72
        result = dsr.calculate(
            observed_sharpe=1.0, num_trials=1000, num_observations=252
        )
        assert 3.0 < result.expected_max_sharpe < 4.0


class TestSharpeVariance:
    """Tests for Sharpe variance calculation."""

    @pytest.fixture
    def dsr(self) -> DeflatedSharpeRatio:
        """Create DSR instance."""
        return DeflatedSharpeRatio()

    def test_more_observations_lower_variance(self, dsr: DeflatedSharpeRatio) -> None:
        """More observations = lower variance."""
        result_100 = dsr.calculate(
            observed_sharpe=1.0, num_trials=1, num_observations=100
        )
        result_1000 = dsr.calculate(
            observed_sharpe=1.0, num_trials=1, num_observations=1000
        )

        assert result_1000.sharpe_variance < result_100.sharpe_variance

    def test_variance_with_normal_returns(self, dsr: DeflatedSharpeRatio) -> None:
        """Variance formula with normal returns (skew=0, kurt=3)."""
        # Var(SR) = (1 + 0.5*SR^2) / (T-1) for normal returns
        result = dsr.calculate(
            observed_sharpe=1.0,
            num_trials=1,
            num_observations=101,
            skewness=0.0,
            kurtosis=3.0,
        )
        expected_var = (1 + 0.5 * 1.0**2) / 100
        assert abs(result.sharpe_variance - expected_var) < 1e-10

    def test_skewness_affects_variance(self, dsr: DeflatedSharpeRatio) -> None:
        """Positive skewness with positive Sharpe reduces variance."""
        result_no_skew = dsr.calculate(
            observed_sharpe=2.0,
            num_trials=1,
            num_observations=252,
            skewness=0.0,
            kurtosis=3.0,
        )
        result_pos_skew = dsr.calculate(
            observed_sharpe=2.0,
            num_trials=1,
            num_observations=252,
            skewness=1.0,  # Positive skew
            kurtosis=3.0,
        )
        # Formula: -skew*SR term, so positive skew reduces numerator
        assert result_pos_skew.sharpe_variance < result_no_skew.sharpe_variance

    def test_fat_tails_increase_variance(self, dsr: DeflatedSharpeRatio) -> None:
        """Fat tails (high kurtosis) increase variance for non-zero Sharpe."""
        result_normal = dsr.calculate(
            observed_sharpe=1.5,
            num_trials=1,
            num_observations=252,
            skewness=0.0,
            kurtosis=3.0,
        )
        result_fat = dsr.calculate(
            observed_sharpe=1.5,
            num_trials=1,
            num_observations=252,
            skewness=0.0,
            kurtosis=6.0,  # Fat tails
        )
        # Formula: (excess_kurt/4)*SR^2 term
        assert result_fat.sharpe_variance > result_normal.sharpe_variance


class TestDSRCalculation:
    """Tests for full DSR calculation."""

    @pytest.fixture
    def dsr(self) -> DeflatedSharpeRatio:
        """Create DSR instance."""
        return DeflatedSharpeRatio()

    def test_high_sharpe_single_trial_significant(
        self, dsr: DeflatedSharpeRatio
    ) -> None:
        """High Sharpe with single trial should be significant."""
        result = dsr.calculate(
            observed_sharpe=3.0,  # Very high
            num_trials=1,  # No multiple testing
            num_observations=252,
        )
        # Expected max = 0 for N=1, so DSR = SR / sqrt(var)
        # Should be highly significant
        assert result.is_significant
        assert result.p_value < 0.01

    def test_high_sharpe_many_trials_not_significant(
        self, dsr: DeflatedSharpeRatio
    ) -> None:
        """High Sharpe from many trials may not be significant."""
        result = dsr.calculate(
            observed_sharpe=2.5,  # High but not extreme
            num_trials=10000,  # Many trials
            num_observations=252,
        )
        # Expected max Sharpe for 10000 trials is ~4.3
        # 2.5 < 4.3, so not significant
        assert not result.is_significant
        assert result.p_value > 0.05

    def test_negative_sharpe_not_significant(self, dsr: DeflatedSharpeRatio) -> None:
        """Negative Sharpe is never significant."""
        result = dsr.calculate(
            observed_sharpe=-0.5,
            num_trials=10,
            num_observations=252,
        )
        assert not result.is_significant
        assert result.p_value > 0.5

    def test_deflated_sharpe_below_zero_when_sr_below_expected(
        self, dsr: DeflatedSharpeRatio
    ) -> None:
        """Deflated Sharpe < 0 when observed < expected max."""
        result = dsr.calculate(
            observed_sharpe=1.0,
            num_trials=100,  # Expected max ~3
            num_observations=252,
        )
        assert result.deflated_sharpe < 0
        assert result.observed_sharpe < result.expected_max_sharpe


class TestDSRResult:
    """Tests for DSRResult dataclass."""

    def test_result_immutable(self) -> None:
        """DSRResult is frozen (immutable)."""
        result = DSRResult(
            deflated_sharpe=1.5,
            p_value=0.03,
            is_significant=True,
            expected_max_sharpe=2.0,
            sharpe_variance=0.01,
            observed_sharpe=2.5,
            num_trials=100,
            num_observations=252,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            result.p_value = 0.5  # type: ignore[misc]

    def test_result_contains_inputs(self) -> None:
        """Result stores original inputs."""
        dsr = DeflatedSharpeRatio()
        result = dsr.calculate(
            observed_sharpe=1.5,
            num_trials=50,
            num_observations=365,
        )
        assert result.observed_sharpe == 1.5
        assert result.num_trials == 50
        assert result.num_observations == 365


class TestConvenienceMethods:
    """Tests for confidence_level and passes_threshold methods."""

    @pytest.fixture
    def dsr(self) -> DeflatedSharpeRatio:
        """Create DSR instance."""
        return DeflatedSharpeRatio()

    def test_confidence_level_inverse_of_pvalue(
        self, dsr: DeflatedSharpeRatio
    ) -> None:
        """Confidence level = 1 - p_value."""
        result = dsr.calculate(
            observed_sharpe=2.0,
            num_trials=10,
            num_observations=252,
        )
        assert abs(dsr.confidence_level(result) - (1 - result.p_value)) < 1e-10

    def test_passes_threshold_95(self, dsr: DeflatedSharpeRatio) -> None:
        """passes_threshold with 0.95 threshold."""
        # Very high Sharpe, few trials = high confidence
        result = dsr.calculate(
            observed_sharpe=5.0,
            num_trials=10,
            num_observations=500,
        )
        assert dsr.passes_threshold(result, threshold=0.95)

        # Low Sharpe, many trials = low confidence
        result_low = dsr.calculate(
            observed_sharpe=1.0,
            num_trials=1000,
            num_observations=100,
        )
        assert not dsr.passes_threshold(result_low, threshold=0.95)


class TestConvenienceFunction:
    """Tests for calculate_dsr convenience function."""

    def test_calculate_dsr_matches_class(self) -> None:
        """Convenience function returns same result as class method."""
        dsr = DeflatedSharpeRatio(significance_level=0.05)
        result_class = dsr.calculate(
            observed_sharpe=1.5,
            num_trials=50,
            num_observations=252,
            skewness=0.5,
            kurtosis=4.0,
        )
        result_func = calculate_dsr(
            observed_sharpe=1.5,
            num_trials=50,
            num_observations=252,
            skewness=0.5,
            kurtosis=4.0,
            significance_level=0.05,
        )
        assert result_class.deflated_sharpe == result_func.deflated_sharpe
        assert result_class.p_value == result_func.p_value
        assert result_class.is_significant == result_func.is_significant


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def dsr(self) -> DeflatedSharpeRatio:
        """Create DSR instance."""
        return DeflatedSharpeRatio()

    def test_zero_sharpe(self, dsr: DeflatedSharpeRatio) -> None:
        """Zero Sharpe ratio is handled correctly."""
        result = dsr.calculate(
            observed_sharpe=0.0,
            num_trials=10,
            num_observations=252,
        )
        assert not result.is_significant
        assert result.deflated_sharpe < 0  # 0 < expected_max

    def test_very_high_num_trials(self, dsr: DeflatedSharpeRatio) -> None:
        """Very high number of trials is handled."""
        result = dsr.calculate(
            observed_sharpe=5.0,
            num_trials=1_000_000,
            num_observations=252,
        )
        # Expected max for 1M trials is ~5.26
        # So even Sharpe=5 is borderline
        assert result.expected_max_sharpe > 5.0
        assert not math.isnan(result.deflated_sharpe)

    def test_very_few_observations(self, dsr: DeflatedSharpeRatio) -> None:
        """Minimum observations (2) works but has high variance."""
        result = dsr.calculate(
            observed_sharpe=2.0,
            num_trials=10,
            num_observations=2,
        )
        # With T=2, variance = numerator / 1 which is large
        assert result.sharpe_variance > 1.0
        assert not math.isnan(result.p_value)

    def test_extreme_skewness(self, dsr: DeflatedSharpeRatio) -> None:
        """Extreme skewness is handled."""
        result = dsr.calculate(
            observed_sharpe=1.0,
            num_trials=10,
            num_observations=252,
            skewness=5.0,  # Very skewed
            kurtosis=3.0,
        )
        assert not math.isnan(result.deflated_sharpe)

    def test_very_high_kurtosis(self, dsr: DeflatedSharpeRatio) -> None:
        """Very high kurtosis (heavy tails) is handled."""
        result = dsr.calculate(
            observed_sharpe=1.0,
            num_trials=10,
            num_observations=252,
            skewness=0.0,
            kurtosis=50.0,  # Extreme fat tails
        )
        assert not math.isnan(result.deflated_sharpe)
        # High kurtosis increases variance
        result_normal = dsr.calculate(
            observed_sharpe=1.0,
            num_trials=10,
            num_observations=252,
            skewness=0.0,
            kurtosis=3.0,
        )
        assert result.sharpe_variance > result_normal.sharpe_variance


class TestSpecCompliance:
    """Tests verifying compliance with SPEC.md requirements."""

    def test_filter_threshold_95_confidence(self) -> None:
        """SPEC.md: DSR > 0.95 means 95% confidence Sharpe is not a fluke."""
        dsr = DeflatedSharpeRatio()

        # Strategy with genuinely high Sharpe
        result = dsr.calculate(
            observed_sharpe=4.0,
            num_trials=100,  # 100 param combos
            num_observations=500,  # ~2 years daily
        )

        # Should pass 95% confidence threshold
        assert dsr.passes_threshold(result, threshold=0.95)

    def test_rejects_lucky_sharpe_with_many_trials(self) -> None:
        """DSR should reject strategies that are likely lucky finds."""
        dsr = DeflatedSharpeRatio()

        # Test 10000 param combos, get Sharpe of 3
        # Expected max for 10000 trials is ~4.3
        result = dsr.calculate(
            observed_sharpe=3.0,
            num_trials=10000,
            num_observations=252,
        )

        # Should NOT pass - this Sharpe is likely from multiple testing
        assert not dsr.passes_threshold(result, threshold=0.95)
        assert result.p_value > 0.5


class TestEulerMascheroniConstant:
    """Test Euler-Mascheroni constant is correct."""

    def test_constant_value(self) -> None:
        """Euler-Mascheroni constant has correct value."""
        # Known value to high precision
        expected = 0.5772156649015328606065120900824024310421
        assert abs(EULER_MASCHERONI - expected) < 1e-15
