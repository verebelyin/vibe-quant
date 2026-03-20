"""Bootstrap confidence interval for Sharpe ratio.

Resamples trade-level PnL returns with replacement to compute a
confidence interval on the Sharpe ratio. Strategies where the lower
bound falls below a threshold are rejected as statistically
insignificant — they may have achieved high Sharpe by luck from
a small number of trades.

Usage:
    from vibe_quant.overfitting.bootstrap_sharpe import (
        bootstrap_sharpe_ci, BootstrapResult
    )
    result = bootstrap_sharpe_ci(trade_returns, n_bootstrap=10_000)
    print(f"Sharpe 95% CI: [{result.ci_lower:.2f}, {result.ci_upper:.2f}]")
    if result.ci_lower < 1.0:
        print("REJECT: insufficient statistical significance")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Result of bootstrap Sharpe ratio confidence interval.

    Attributes:
        observed_sharpe: Sharpe computed from the original trade returns.
        ci_lower: Lower bound of the confidence interval.
        ci_upper: Upper bound of the confidence interval.
        ci_level: Confidence level (e.g. 0.95 for 95% CI).
        n_trades: Number of trades in the original sample.
        n_bootstrap: Number of bootstrap iterations performed.
        passed: Whether ci_lower >= min_sharpe threshold.
        min_sharpe: The threshold used for the pass/fail decision.
        bootstrap_sharpes: Full array of bootstrap Sharpe ratios (for plotting).
    """

    observed_sharpe: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_trades: int
    n_bootstrap: int
    passed: bool
    min_sharpe: float
    bootstrap_sharpes: np.ndarray


def _sharpe_from_returns(returns: np.ndarray) -> float:
    """Compute Sharpe ratio from an array of trade returns.

    Uses trade-level Sharpe: mean / std * sqrt(n), consistent with
    the random_baseline and screening metric computation.
    """
    n = len(returns)
    if n < 2:
        return 0.0
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    if std < 1e-10:
        return 0.0
    return mean / std * np.sqrt(n)


def bootstrap_sharpe_ci(
    trade_returns: np.ndarray | list[float],
    *,
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
    min_sharpe: float = 1.0,
    seed: int | None = 42,
) -> BootstrapResult:
    """Compute bootstrap confidence interval for trade-level Sharpe.

    Resamples (with replacement) the trade returns n_bootstrap times,
    computes Sharpe for each resample, then takes percentile CI.

    Args:
        trade_returns: Array of per-trade PnL returns (as fractions or
            percentages — just be consistent). Can be list or ndarray.
        n_bootstrap: Number of bootstrap iterations.
        ci_level: Confidence level (default 0.95 = 95% CI).
        min_sharpe: Minimum Sharpe for the lower CI bound to pass.
        seed: Random seed for reproducibility.

    Returns:
        BootstrapResult with CI bounds, pass/fail, and full distribution.
    """
    returns = np.asarray(trade_returns, dtype=np.float64)
    n = len(returns)

    if n < 5:
        logger.warning("Only %d trades — bootstrap CI unreliable", n)
        return BootstrapResult(
            observed_sharpe=_sharpe_from_returns(returns),
            ci_lower=float("-inf"),
            ci_upper=float("inf"),
            ci_level=ci_level,
            n_trades=n,
            n_bootstrap=0,
            passed=False,
            min_sharpe=min_sharpe,
            bootstrap_sharpes=np.array([]),
        )

    rng = np.random.default_rng(seed)
    observed = _sharpe_from_returns(returns)

    # Vectorized bootstrap: generate all resamples at once
    # Shape: (n_bootstrap, n) — each row is one bootstrap sample
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    samples = returns[indices]  # (n_bootstrap, n)

    # Compute Sharpe for each bootstrap sample
    means = np.mean(samples, axis=1)
    stds = np.std(samples, axis=1, ddof=1)
    # Avoid division by zero
    safe_stds = np.where(stds < 1e-10, 1.0, stds)
    boot_sharpes = means / safe_stds * np.sqrt(n)
    # Zero out where std was effectively zero
    boot_sharpes = np.where(stds < 1e-10, 0.0, boot_sharpes)

    alpha = 1.0 - ci_level
    ci_lower = float(np.percentile(boot_sharpes, alpha / 2 * 100))
    ci_upper = float(np.percentile(boot_sharpes, (1 - alpha / 2) * 100))

    passed = ci_lower >= min_sharpe

    logger.debug(
        "Bootstrap Sharpe CI: observed=%.2f [%.2f, %.2f] (n=%d, %d bootstrap) → %s",
        observed,
        ci_lower,
        ci_upper,
        n,
        n_bootstrap,
        "PASS" if passed else "FAIL",
    )

    return BootstrapResult(
        observed_sharpe=observed,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        n_trades=n,
        n_bootstrap=n_bootstrap,
        passed=passed,
        min_sharpe=min_sharpe,
        bootstrap_sharpes=boot_sharpes,
    )
