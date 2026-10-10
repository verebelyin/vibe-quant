"""Guard rails for genetic strategy discovery.

Applies minimum trade filters, complexity guards, DSR correction for multiple
testing across the entire discovery run, and Walk-Forward validation before
promoting candidates.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vibe_quant.overfitting.dsr import (
    TRADING_DAYS_PER_YEAR,
    DeflatedSharpeRatio,
    daily_sharpe_inputs,
    deannualize_sharpe,
    lag1_autocorrelation,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import date

    import numpy as np

    from vibe_quant.discovery.fitness import FitnessResult
    from vibe_quant.overfitting.bootstrap_sharpe import BootstrapResult
    from vibe_quant.overfitting.dsr import DSRResult
    from vibe_quant.overfitting.purged_kfold import (
        BacktestRunner as KFoldRunner,
    )
    from vibe_quant.overfitting.purged_kfold import (
        CVResult,
        PurgedKFoldCV,
    )
    from vibe_quant.overfitting.wfa import WalkForwardAnalysis, WFAResult

logger = logging.getLogger(__name__)

#: Worst-mode per-symbol significance floor (bd vibe-quant-yul7u.24, review
#: SF1): besides the pooled DSR, every symbol's own UNDEFLATED probabilistic
#: Sharpe PSR(SR > 0) must reach this, so one strong symbol cannot carry
#: null ones through the pooled test (~annualized SR >= 0.38 at T=1264).
SYMBOL_PSR_FLOOR: float = 0.8

#: How the dense daily series handles each (sub-)window's first day; recorded
#: in every DSR record so the observation count T is reproducible.
DAY0_CONVENTION = (
    "returns for the days after the train start (NT Sharpe convention); "
    "eval sub-windows 2..N contribute their first-day return vs a fresh balance, "
    "so T = train days - 1 for any eval_windows"
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GuardrailConfig:
    """Configuration for discovery guard rails.

    Attributes:
        min_trades: Minimum trades required per candidate.
        max_complexity: Maximum total genes (entry + exit) allowed.
        require_dsr: Whether to apply DSR multiple-testing correction.
        dsr_significance_level: p-value threshold for DSR significance.
        require_wfa: Whether to require Walk-Forward validation.
        wfa_min_efficiency: Minimum WFA efficiency to pass.
        require_purged_kfold: Whether to require Purged K-Fold CV (expensive).
    """

    min_trades: int = 50
    min_return: float = 0.0  # minimum total return (fraction, e.g. 0.0 = breakeven)
    max_complexity: int = 8
    require_dsr: bool = True
    dsr_significance_level: float = 0.05
    require_wfa: bool = True
    wfa_min_efficiency: float = 0.5
    require_purged_kfold: bool = False
    require_bootstrap_ci: bool = False  # Bootstrap Sharpe CI filter
    bootstrap_min_sharpe: float = 1.0  # Reject if CI lower bound < this
    bootstrap_ci_level: float = 0.95  # Confidence level (95%)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GuardrailResult:
    """Result of applying all guard rails to a candidate.

    Attributes:
        passed: Overall verdict (True only if all enabled checks pass).
        min_trades_passed: Whether min trade filter passed.
        complexity_passed: Whether complexity guard passed.
        dsr_passed: Whether DSR check passed (None if disabled).
        wfa_passed: Whether WFA check passed (None if disabled).
        kfold_passed: Whether Purged K-Fold check passed (None if disabled).
        reasons: List of failure reasons (empty if all passed).
        dsr_result: Full DSR result if DSR was run.
        wfa_result: Full WFA result if WFA was run.
        kfold_result: Full CV result if K-Fold was run.
    """

    passed: bool
    min_trades_passed: bool
    complexity_passed: bool
    dsr_passed: bool | None = None
    wfa_passed: bool | None = None
    kfold_passed: bool | None = None
    bootstrap_passed: bool | None = None
    reasons: list[str] = field(default_factory=list)
    dsr_result: DSRResult | None = None
    wfa_result: WFAResult | None = None
    kfold_result: CVResult | None = None
    bootstrap_result: BootstrapResult | None = None
    # JSON-safe record of the DSR inputs/outputs (persisted with the run).
    dsr_record: dict[str, object] | None = None


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_min_trades(total_trades: int, min_trades: int) -> tuple[bool, str | None]:
    """Check if candidate meets minimum trade threshold.

    Args:
        total_trades: Number of trades the candidate produced.
        min_trades: Minimum required trades.

    Returns:
        Tuple of (passed, reason_if_failed).
    """
    if total_trades < min_trades:
        reason = f"Too few trades: {total_trades} < {min_trades}"
        logger.info("Guardrail reject: %s", reason)
        return False, reason
    return True, None


def check_complexity(num_genes: int, max_complexity: int) -> tuple[bool, str | None]:
    """Check if candidate complexity is within bounds.

    Args:
        num_genes: Total genes (entry + exit).
        max_complexity: Maximum allowed genes.

    Returns:
        Tuple of (passed, reason_if_failed).
    """
    if num_genes > max_complexity:
        reason = f"Too complex: {num_genes} genes > {max_complexity} max"
        logger.info("Guardrail reject: %s", reason)
        return False, reason
    return True, None


def apply_discovery_dsr(
    observed_sharpe: float,
    num_trials: int,
    num_observations: int,
    significance_level: float = 0.05,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    trials_sharpe_variance: float | None = None,
) -> DSRResult:
    """Apply DSR correction for the entire discovery run.

    num_trials should be the total number of candidates evaluated across
    all generations, accounting for the full multiple-testing burden.

    The observed Sharpe is expected in NT's annualized units and is
    de-annualized to daily units here (sqrt-252 scaling) so it matches
    ``num_observations`` — pass the number of DAYS in the backtest window,
    not the bar count. Mixing annualized SR with bar counts made the
    deflation nearly toothless (vibe-quant-zmzzh).

    Args:
        observed_sharpe: Best candidate's Sharpe ratio (annualized, as
            reported by NT).
        num_trials: Total candidates evaluated in the discovery run.
        num_observations: Number of daily observations in the backtest.
        significance_level: p-value threshold for significance.
        skewness: Return distribution skewness.
        kurtosis: Return distribution kurtosis.
        trials_sharpe_variance: Empirical variance of ANNUALIZED Sharpe
            ratios across all evaluated candidates (converted to daily
            units internally). Improves DSR accuracy for fat-tailed
            return distributions.

    Returns:
        DSRResult with significance test.
    """
    dsr = DeflatedSharpeRatio(significance_level=significance_level)
    result = dsr.calculate(
        observed_sharpe=deannualize_sharpe(observed_sharpe),
        num_trials=num_trials,
        num_observations=num_observations,
        skewness=skewness,
        kurtosis=kurtosis,
        trials_sharpe_variance=(
            trials_sharpe_variance / TRADING_DAYS_PER_YEAR
            if trials_sharpe_variance is not None
            else None
        ),
    )
    logger.info(
        "DSR: sharpe=%.3f trials=%d p=%.4f significant=%s",
        observed_sharpe,
        num_trials,
        result.p_value,
        result.is_significant,
    )
    return result


def _clamped_dsr(
    dsr: DeflatedSharpeRatio, sharpe: float, num_trials: int, t: int, skew: float, kurt: float
) -> DSRResult:
    """DSR with conservative moments (review SF3): skew <= 0, kurtosis >= 3.

    A positively skewed / thin-tailed sample LOWERS the Sharpe estimator's
    variance and so the bar to pass; those moments are not trusted.
    """
    return dsr.calculate(sharpe, num_trials, t, min(skew, 0.0), max(kurt, 3.0))


def apply_discovery_dsr_returns(
    daily_returns: Sequence[float] | None,
    num_trials: int,
    significance_level: float = 0.05,
    symbol_daily: Mapping[str, Mapping[str, float] | None] | None = None,
    symbol_psr_floor: float = SYMBOL_PSR_FLOOR,
) -> tuple[DSRResult | None, dict[str, object], list[str]]:
    """DSR on a dense DAILY return series (bd vibe-quant-yul7u.24).

    The tested series is the champion candidate's daily balance returns over
    the whole train range (portfolio mode: the shared account; worst mode:
    the equal-weight pool of the symbols' own runs). Sharpe, skewness,
    kurtosis and T (= series length) all come from that one series, at the
    raw trial count N. Moments are clamped (skew <= 0, kurt >= 3).

    Worst mode (``symbol_daily`` given, 2+ symbols): each symbol must also
    reach ``symbol_psr_floor`` undeflated PSR(SR > 0). With one symbol the
    pooled series IS that symbol and no extra floor applies.

    Fails closed: a missing/short/non-finite series, or a symbol without one,
    is rejected with a ``DSR input missing`` reason.

    Returns:
        ``(result_or_None, record, reasons)`` -- empty ``reasons`` = pass.
    """
    record: dict[str, object] = {
        "input": "pooled_daily_returns",
        "trials": num_trials,
        "significance_level": significance_level,
        "day0_convention": DAY0_CONVENTION,
    }
    values = list(daily_returns) if daily_returns is not None else []
    if len(values) < 2 or not all(math.isfinite(v) for v in values):
        record["missing"] = True
        reason = (
            "DSR input missing: no usable daily return series "
            f"(length {len(values)}{', non-finite values' if values else ''})"
        )
        logger.info("Guardrail reject: %s", reason)
        return None, record, [reason]
    missing_syms = sorted(s for s, v in (symbol_daily or {}).items() if v is None)
    if missing_syms:
        record["missing"] = True
        reason = f"DSR input missing: no daily return series for {', '.join(missing_syms)}"
        logger.info("Guardrail reject: %s", reason)
        return None, record, [reason]

    dsr = DeflatedSharpeRatio(significance_level=significance_level)
    sharpe, skew, kurt = daily_sharpe_inputs(values)
    t = len(values)
    result = _clamped_dsr(dsr, sharpe, num_trials, t, skew, kurt)
    unclamped = dsr.calculate(sharpe, num_trials, t, skew, kurt)
    ann = math.sqrt(TRADING_DAYS_PER_YEAR)
    record.update(
        observations=t,
        sharpe_daily=sharpe,
        sharpe_annualized=sharpe * ann,
        skewness=skew,
        kurtosis=kurt,
        skewness_used=min(skew, 0.0),
        kurtosis_used=max(kurt, 3.0),
        expected_max_sharpe_daily=result.expected_max_sharpe,
        p_value=result.p_value,
        p_value_unclamped=unclamped.p_value,
        significant=result.is_significant,
        lag1_autocorrelation=lag1_autocorrelation(values),
    )
    reasons: list[str] = []
    if not result.is_significant:
        reasons.append(
            f"DSR not significant: p={result.p_value:.4f} >= {significance_level} "
            f"(pooled daily SR={sharpe * ann:.3f} ann, T={t}, N={num_trials})"
        )

    if symbol_daily is not None and len(symbol_daily) >= 2:
        stats = {s: v for s, v in symbol_daily.items() if v is not None}
        mean_total = sum(float(v["mean"]) for v in stats.values())
        per_symbol: dict[str, dict[str, float | int | None]] = {}
        for sym, st in stats.items():
            psr = 1.0 - _clamped_dsr(
                dsr,
                float(st["sharpe"]),
                1,
                max(2, int(st["observations"])),
                float(st["skewness"]),
                float(st["kurtosis"]),
            ).p_value
            per_symbol[sym] = {
                "sharpe_annualized": float(st["sharpe"]) * ann,
                "psr": psr,
                # Share of the pooled mean daily return this symbol supplies.
                "contribution_share": float(st["mean"]) / mean_total if mean_total else None,
                "observations": int(st["observations"]),
            }
            if not psr >= symbol_psr_floor:
                reasons.append(
                    f"DSR per-symbol floor: {sym} PSR(SR>0)={psr:.3f} < {symbol_psr_floor:.2f} "
                    f"(SR={float(st['sharpe']) * ann:.3f} ann)"
                )
        record["symbol_psr_floor"] = symbol_psr_floor
        record["symbols"] = per_symbol

    logger.info(
        "DSR(pooled daily): sharpe=%.3f ann T=%d trials=%d p=%.4f (unclamped %.4f) reasons=%s",
        sharpe * ann, t, num_trials, result.p_value, unclamped.p_value, reasons,
    )
    for reason in reasons:
        logger.info("Guardrail reject: %s", reason)
    return result, record, reasons


def check_walk_forward(
    wfa: WalkForwardAnalysis,
    strategy_id: str,
    data_start: date,
    data_end: date,
    param_grid: dict[str, list[object]],
    min_efficiency: float = 0.5,
) -> tuple[WFAResult, bool, str | None]:
    """Validate candidate with Walk-Forward Analysis.

    Args:
        wfa: Configured WalkForwardAnalysis instance (with runner set).
        strategy_id: Strategy identifier.
        data_start: Data start date.
        data_end: Data end date.
        param_grid: Parameter grid for WFA optimization.
        min_efficiency: Minimum WFA efficiency to pass.

    Returns:
        Tuple of (WFAResult, passed, reason_if_failed).
    """
    result = wfa.run(strategy_id, data_start, data_end, param_grid)
    passed = result.is_robust and result.efficiency >= min_efficiency
    reason: str | None = None
    if not passed:
        parts: list[str] = []
        if not result.is_robust:
            parts.append("WFA not robust")
        if result.efficiency < min_efficiency:
            parts.append(f"WFA efficiency {result.efficiency:.3f} < {min_efficiency}")
        reason = "; ".join(parts)
        logger.info("Guardrail reject: %s", reason)
    else:
        logger.info("WFA passed: efficiency=%.3f", result.efficiency)
    return result, passed, reason


def check_purged_kfold(
    cv: PurgedKFoldCV,
    n_samples: int,
    runner: KFoldRunner,
) -> tuple[CVResult, bool, str | None]:
    """Validate candidate with Purged K-Fold CV.

    Args:
        cv: Configured PurgedKFoldCV instance.
        n_samples: Total samples in dataset.
        runner: Backtest runner for CV folds.

    Returns:
        Tuple of (CVResult, passed, reason_if_failed).
    """
    result = cv.run(n_samples, runner)
    passed = result.is_robust
    reason: str | None = None
    if not passed:
        reason = (
            f"K-Fold CV not robust: mean_oos_sharpe={result.mean_oos_sharpe:.3f} "
            f"std={result.std_oos_sharpe:.3f}"
        )
        logger.info("Guardrail reject: %s", reason)
    else:
        logger.info("K-Fold CV passed: mean_oos_sharpe=%.3f", result.mean_oos_sharpe)
    return result, passed, reason


# ---------------------------------------------------------------------------
# Combined guardrail check
# ---------------------------------------------------------------------------


def apply_guardrails(
    fitness: FitnessResult,
    num_genes: int,
    config: GuardrailConfig,
    *,
    num_trials: int = 1,
    num_observations: int = 252,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    trials_sharpe_variance: float | None = None,
    wfa: WalkForwardAnalysis | None = None,
    wfa_strategy_id: str = "",
    wfa_data_start: date | None = None,
    wfa_data_end: date | None = None,
    wfa_param_grid: dict[str, list[object]] | None = None,
    kfold_cv: PurgedKFoldCV | None = None,
    kfold_n_samples: int = 0,
    kfold_runner: KFoldRunner | None = None,
    trade_returns: np.ndarray | None = None,
    daily_returns: Sequence[float] | None = None,
    symbol_daily: Mapping[str, Mapping[str, float] | None] | None = None,
    dsr_requires_returns: bool = False,
) -> GuardrailResult:
    """Run all enabled guard rails on a candidate.

    Args:
        fitness: Fitness result from backtest evaluation.
        num_genes: Total genes (entry + exit) in the chromosome.
        config: Guardrail configuration.
        num_trials: Total candidates evaluated in discovery run (for DSR).
        num_observations: Number of bars/periods in backtest (for DSR).
        skewness: Return distribution skewness (for DSR).
        kurtosis: Return distribution kurtosis (for DSR).
        trials_sharpe_variance: Empirical variance of Sharpe ratios across
            all evaluated candidates (for DSR).
        wfa: WalkForwardAnalysis instance (required if require_wfa).
        wfa_strategy_id: Strategy ID for WFA.
        wfa_data_start: Data start date for WFA.
        wfa_data_end: Data end date for WFA.
        wfa_param_grid: Parameter grid for WFA.
        kfold_cv: PurgedKFoldCV instance (required if require_purged_kfold).
        kfold_n_samples: Number of samples for K-Fold.
        kfold_runner: Backtest runner for K-Fold.
        daily_returns: Dense daily return series of the candidate. When given
            (or when ``dsr_requires_returns``), DSR tests THIS series -- see
            :func:`apply_discovery_dsr_returns` -- instead of
            ``fitness.sharpe_ratio`` with ``num_observations``/moments.
        symbol_daily: Worst mode per-symbol daily Sharpe inputs (floor).
        dsr_requires_returns: Real backtests always produce the series; a
            missing one then fails DSR closed instead of falling back.

    Returns:
        GuardrailResult with per-check verdicts and overall pass/fail.
    """
    reasons: list[str] = []

    # 1. Minimum trades
    min_trades_passed, min_trades_reason = check_min_trades(fitness.total_trades, config.min_trades)
    if min_trades_reason:
        reasons.append(min_trades_reason)

    # 1b. Minimum return (reject positive-Sharpe but negative-return strategies)
    min_return_passed = fitness.total_return >= config.min_return
    if not min_return_passed:
        reasons.append(
            f"Return {fitness.total_return:.2%} below minimum {config.min_return:.2%}"
        )

    # 2. Complexity
    complexity_passed, complexity_reason = check_complexity(num_genes, config.max_complexity)
    if complexity_reason:
        reasons.append(complexity_reason)

    # 3. DSR
    dsr_passed: bool | None = None
    dsr_result: DSRResult | None = None
    dsr_record: dict[str, object] | None = None
    if config.require_dsr and (daily_returns is not None or dsr_requires_returns):
        dsr_result, dsr_record, dsr_reasons = apply_discovery_dsr_returns(
            daily_returns,
            num_trials=num_trials,
            significance_level=config.dsr_significance_level,
            symbol_daily=symbol_daily,
        )
        dsr_passed = not dsr_reasons
        reasons.extend(dsr_reasons)
    elif config.require_dsr:
        # Synthetic/mock backtests carry no daily series: legacy input.
        dsr_result = apply_discovery_dsr(
            observed_sharpe=fitness.sharpe_ratio,
            num_trials=num_trials,
            num_observations=num_observations,
            significance_level=config.dsr_significance_level,
            skewness=skewness,
            kurtosis=kurtosis,
            trials_sharpe_variance=trials_sharpe_variance,
        )
        dsr_passed = dsr_result.is_significant
        dsr_record = {
            "input": "fitness_sharpe",
            "trials": num_trials,
            "observations": num_observations,
            "p_value": dsr_result.p_value,
            "significant": dsr_passed,
        }
        if not dsr_passed:
            reasons.append(
                f"DSR not significant: p={dsr_result.p_value:.4f} "
                f">= {config.dsr_significance_level}"
            )

    # 4. Walk-Forward
    wfa_passed: bool | None = None
    wfa_result: WFAResult | None = None
    if config.require_wfa:
        if wfa is None or wfa_data_start is None or wfa_data_end is None:
            wfa_passed = False
            reasons.append("WFA required but WFA instance/dates not provided")
        else:
            try:
                wfa_result, wfa_passed, wfa_reason = check_walk_forward(
                    wfa=wfa,
                    strategy_id=wfa_strategy_id,
                    data_start=wfa_data_start,
                    data_end=wfa_data_end,
                    param_grid=wfa_param_grid or {},
                    min_efficiency=config.wfa_min_efficiency,
                )
                if wfa_reason:
                    reasons.append(wfa_reason)
            except (ValueError, RuntimeError) as exc:
                wfa_passed = False
                reasons.append(f"WFA failed with error: {exc}")
                logger.warning("WFA guardrail error: %s", exc)

    # 5. Purged K-Fold
    kfold_passed: bool | None = None
    kfold_result: CVResult | None = None
    if config.require_purged_kfold:
        if kfold_cv is None or kfold_runner is None or kfold_n_samples < 1:
            kfold_passed = False
            reasons.append("K-Fold CV required but CV instance/runner not provided")
        else:
            kfold_result, kfold_passed, kfold_reason = check_purged_kfold(
                cv=kfold_cv,
                n_samples=kfold_n_samples,
                runner=kfold_runner,
            )
            if kfold_reason:
                reasons.append(kfold_reason)

    # 6. Bootstrap Sharpe CI
    bootstrap_passed: bool | None = None
    bootstrap_result: BootstrapResult | None = None
    if config.require_bootstrap_ci:
        if trade_returns is not None and len(trade_returns) >= 5:
            from vibe_quant.overfitting.bootstrap_sharpe import bootstrap_sharpe_ci

            bootstrap_result = bootstrap_sharpe_ci(
                trade_returns,
                ci_level=config.bootstrap_ci_level,
                min_sharpe=config.bootstrap_min_sharpe,
            )
            bootstrap_passed = bootstrap_result.passed
            if not bootstrap_passed:
                reasons.append(
                    f"Bootstrap CI lower bound {bootstrap_result.ci_lower:.2f} "
                    f"< {config.bootstrap_min_sharpe:.1f} "
                    f"(observed Sharpe={bootstrap_result.observed_sharpe:.2f}, "
                    f"n={bootstrap_result.n_trades})"
                )
            else:
                logger.info(
                    "Bootstrap CI passed: [%.2f, %.2f] (n=%d)",
                    bootstrap_result.ci_lower,
                    bootstrap_result.ci_upper,
                    bootstrap_result.n_trades,
                )
        else:
            bootstrap_passed = False
            n_ret = len(trade_returns) if trade_returns is not None else 0
            reasons.append(
                f"Bootstrap CI required but insufficient trade returns "
                f"(got {n_ret}, need ≥5)"
            )

    # Overall verdict: all enabled checks must pass
    overall = min_trades_passed and min_return_passed and complexity_passed
    if dsr_passed is not None:
        overall = overall and dsr_passed
    if wfa_passed is not None:
        overall = overall and wfa_passed
    if kfold_passed is not None:
        overall = overall and kfold_passed
    if bootstrap_passed is not None:
        overall = overall and bootstrap_passed

    return GuardrailResult(
        passed=overall,
        min_trades_passed=min_trades_passed,
        complexity_passed=complexity_passed,
        dsr_passed=dsr_passed,
        wfa_passed=wfa_passed,
        kfold_passed=kfold_passed,
        bootstrap_passed=bootstrap_passed,
        reasons=reasons,
        dsr_result=dsr_result,
        dsr_record=dsr_record,
        wfa_result=wfa_result,
        kfold_result=kfold_result,
        bootstrap_result=bootstrap_result,
    )
