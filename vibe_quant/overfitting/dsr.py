"""Deflated Sharpe Ratio (DSR) for multiple hypothesis testing correction.

Implements Bailey & Lopez de Prado's DSR formula to determine if an observed
Sharpe ratio is statistically significant after accounting for the number of
trials (parameter combinations) tested.

When testing N parameter combinations, the expected maximum Sharpe by chance
increases with N. DSR adjusts for this selection bias.

Reference:
    Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting and Non-Normality."
    Journal of Portfolio Management.

Formula:
    DSR = Phi[(SR_hat - SR_0) * sqrt(T-1) / sqrt(1 - gamma3*SR_hat + ((gamma4-1)/4)*SR_hat^2)]

Where:
    - SR_hat: observed Sharpe ratio
    - SR_0: expected maximum Sharpe under null (depends on N trials)
    - gamma3: skewness of returns
    - gamma4: kurtosis of returns
    - T: number of observations
    - Phi: standard normal CDF
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

# Euler-Mascheroni constant
EULER_MASCHERONI: float = 0.5772156649015329
# Pre-compute squared constant to avoid repeated multiplication
_EULER_MASCHERONI_SQ: float = EULER_MASCHERONI * EULER_MASCHERONI
# Pre-compute 1/sqrt(2) for fast normal CDF
_INV_SQRT2: float = 1.0 / math.sqrt(2.0)


def _norm_sf(x: float) -> float:
    """Fast survival function (1 - CDF) of the standard normal distribution.

    Uses math.erfc which is implemented in C and avoids the overhead
    of scipy.stats.norm.sf (which imports and dispatches through
    multiple layers). Numerically equivalent to scipy.stats.norm.sf(x).

    Args:
        x: Z-score value.

    Returns:
        P(Z > x) for standard normal Z.
    """
    return 0.5 * math.erfc(x * _INV_SQRT2)


@dataclass(frozen=True)
class DSRResult:
    """Result from Deflated Sharpe Ratio calculation.

    Attributes:
        deflated_sharpe: Adjusted Sharpe ratio (z-score).
        p_value: Probability of observing this Sharpe by chance.
        is_significant: True if p_value < significance threshold.
        expected_max_sharpe: Expected max Sharpe under null hypothesis.
        sharpe_variance: Variance of Sharpe estimator.
        observed_sharpe: Original observed Sharpe ratio.
        num_trials: Number of parameter combinations tested.
        num_observations: Number of time periods in backtest.
    """

    deflated_sharpe: float
    p_value: float
    is_significant: bool
    expected_max_sharpe: float
    sharpe_variance: float
    observed_sharpe: float
    num_trials: int
    num_observations: int


class DeflatedSharpeRatio:
    """Calculate Deflated Sharpe Ratio for multiple testing correction.

    DSR determines if an observed Sharpe ratio is genuinely positive or
    simply the result of testing many parameter combinations.

    Args:
        significance_level: Threshold for is_significant (default 0.05).
            A p_value below this indicates the Sharpe is likely not a fluke.

    Example:
        >>> dsr = DeflatedSharpeRatio()
        >>> result = dsr.calculate(
        ...     observed_sharpe=1.5,
        ...     num_trials=100,
        ...     num_observations=252,
        ... )
        >>> print(f"p-value: {result.p_value:.4f}, significant: {result.is_significant}")
    """

    def __init__(self, significance_level: float = 0.05) -> None:
        """Initialize DSR calculator.

        Args:
            significance_level: p-value threshold for significance.

        Raises:
            ValueError: If significance_level not in (0, 1).
        """
        if not 0 < significance_level < 1:
            msg = f"significance_level must be in (0, 1), got {significance_level}"
            raise ValueError(msg)
        self.significance_level = significance_level

    def calculate(
        self,
        observed_sharpe: float,
        num_trials: int,
        num_observations: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> DSRResult:
        """Calculate Deflated Sharpe Ratio.

        Args:
            observed_sharpe: The Sharpe ratio being tested.
            num_trials: Number of parameter combinations tested.
            num_observations: Number of time periods in backtest (bars/trades).
            skewness: Return distribution skewness (default 0 = symmetric).
            kurtosis: Return distribution kurtosis (default 3 = normal).
                Note: this is full kurtosis, not excess kurtosis.
                TODO: The screening pipeline does not yet store per-candidate
                skewness/kurtosis, so DSR always uses the normal distribution
                assumption. For heavy-tailed crypto returns this understates
                the Sharpe variance, making DSR slightly optimistic. Store
                skewness/kurtosis in sweep_results once available.

        Returns:
            DSRResult with deflated Sharpe, p-value, and significance flag.

        Raises:
            ValueError: If inputs are invalid.
        """
        self._validate_inputs(observed_sharpe, num_trials, num_observations, kurtosis)

        # Compute expected max Sharpe under null
        expected_max_sharpe = self._expected_max_sharpe(num_trials)

        # Compute variance of Sharpe estimator (with non-normality correction)
        sharpe_variance = self._sharpe_variance(
            observed_sharpe, num_observations, skewness, kurtosis
        )

        # Compute deflated Sharpe (z-score)
        if sharpe_variance <= 0:
            # Edge case: return extreme value
            deflated_sharpe = (
                float("inf") if observed_sharpe > expected_max_sharpe else float("-inf")
            )
        else:
            deflated_sharpe = (observed_sharpe - expected_max_sharpe) / math.sqrt(sharpe_variance)

        # Compute p-value (probability of observing this DSR or higher under null)
        # Using survival function (1 - CDF) for upper tail
        # Uses math.erfc instead of scipy.stats.norm.sf for ~100x speedup
        if math.isinf(deflated_sharpe):
            p_value = 0.0 if deflated_sharpe > 0 else 1.0
        else:
            p_value = _norm_sf(deflated_sharpe)

        is_significant = p_value < self.significance_level

        return DSRResult(
            deflated_sharpe=deflated_sharpe,
            p_value=p_value,
            is_significant=is_significant,
            expected_max_sharpe=expected_max_sharpe,
            sharpe_variance=sharpe_variance,
            observed_sharpe=observed_sharpe,
            num_trials=num_trials,
            num_observations=num_observations,
        )

    def _validate_inputs(
        self,
        observed_sharpe: float,
        num_trials: int,
        num_observations: int,
        kurtosis: float,
    ) -> None:
        """Validate calculation inputs.

        Args:
            observed_sharpe: Sharpe ratio (any real number allowed).
            num_trials: Must be >= 1.
            num_observations: Must be >= 2.
            kurtosis: Must be >= 1 (theoretical minimum).

        Raises:
            ValueError: If any input is invalid.
        """
        if num_trials < 1:
            msg = f"num_trials must be >= 1, got {num_trials}"
            raise ValueError(msg)
        if num_observations < 2:
            msg = f"num_observations must be >= 2, got {num_observations}"
            raise ValueError(msg)
        if kurtosis < 1:
            msg = f"kurtosis must be >= 1, got {kurtosis}"
            raise ValueError(msg)

    @staticmethod
    @lru_cache(maxsize=1024)
    def _expected_max_sharpe(num_trials: int) -> float:
        """Compute expected maximum Sharpe ratio under null hypothesis.

        Uses the asymptotic approximation for the expected maximum of N
        independent standard normal random variables:

            E[max(Z_1, ..., Z_N)] ≈ sqrt(2*ln(N)) * (1 - gamma/(ln(N)) + gamma^2/(2*ln(N)^2))

        For N=1, returns 0 (single trial has no multiple testing bias).

        Cached because num_trials is typically a small integer that repeats
        across many candidates in the same screening pipeline.

        Args:
            num_trials: Number of independent trials (N).

        Returns:
            Expected maximum Sharpe under null hypothesis.
        """
        if num_trials <= 1:
            return 0.0

        log_n = math.log(num_trials)

        # Asymptotic approximation
        # E[max] ≈ sqrt(2*ln(N)) * (1 - gamma/ln(N) + gamma^2/(2*ln(N)^2))
        base = math.sqrt(2.0 * log_n)

        # Higher-order correction terms using pre-computed squared constant
        inv_log_n = 1.0 / log_n
        correction = (
            1.0 - EULER_MASCHERONI * inv_log_n + _EULER_MASCHERONI_SQ * 0.5 * inv_log_n * inv_log_n
        )

        return base * correction

    @staticmethod
    def _sharpe_variance(
        sharpe: float,
        num_observations: int,
        skewness: float,
        kurtosis: float,
    ) -> float:
        """Compute variance of Sharpe ratio estimator.

        Accounts for non-normality using Lo's (2002) correction:

            Var(SR) = (1 + 0.5*SR^2 - skew*SR + (kurt-3)/4 * SR^2) / (T-1)

        For normally distributed returns (skew=0, kurt=3), simplifies to:

            Var(SR) = (1 + 0.5*SR^2) / (T-1)

        Args:
            sharpe: Observed Sharpe ratio.
            num_observations: Number of observations (T).
            skewness: Return distribution skewness (gamma_3).
            kurtosis: Return distribution kurtosis (gamma_4).

        Returns:
            Variance of the Sharpe ratio estimator.
        """
        t_minus_1 = num_observations - 1
        if t_minus_1 <= 0:
            return float("inf")

        sr_sq = sharpe * sharpe  # Faster than sharpe**2

        # Combine SR^2 coefficients: 0.5 + (kurtosis - 3)/4
        # = 0.5 + kurtosis/4 - 0.75
        # = kurtosis/4 - 0.25
        sr_sq_coeff = kurtosis * 0.25 - 0.25
        numerator = 1.0 + sr_sq_coeff * sr_sq - skewness * sharpe

        return numerator / t_minus_1

    def confidence_level(self, result: DSRResult) -> float:
        """Return confidence level (1 - p_value) as percentage.

        Convenience method to express DSR result as confidence %.

        Args:
            result: DSRResult from calculate().

        Returns:
            Confidence level as decimal (e.g., 0.95 for 95%).
        """
        return 1.0 - result.p_value

    def passes_threshold(self, result: DSRResult, threshold: float = 0.95) -> bool:
        """Check if result passes confidence threshold.

        Per SPEC.md: DSR > 0.95 means 95% confidence Sharpe is not a fluke.

        Args:
            result: DSRResult from calculate().
            threshold: Confidence threshold (default 0.95).

        Returns:
            True if confidence level >= threshold.
        """
        return self.confidence_level(result) >= threshold


def calculate_dsr(
    observed_sharpe: float,
    num_trials: int,
    num_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    significance_level: float = 0.05,
) -> DSRResult:
    """Convenience function to calculate DSR without instantiating class.

    Args:
        observed_sharpe: The Sharpe ratio being tested.
        num_trials: Number of parameter combinations tested.
        num_observations: Number of time periods in backtest.
        skewness: Return distribution skewness (default 0).
        kurtosis: Return distribution kurtosis (default 3).
        significance_level: p-value threshold (default 0.05).

    Returns:
        DSRResult with deflated Sharpe, p-value, and significance.
    """
    dsr = DeflatedSharpeRatio(significance_level=significance_level)
    return dsr.calculate(
        observed_sharpe=observed_sharpe,
        num_trials=num_trials,
        num_observations=num_observations,
        skewness=skewness,
        kurtosis=kurtosis,
    )
