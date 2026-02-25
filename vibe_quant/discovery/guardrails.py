"""Guard rails for genetic strategy discovery.

Applies minimum trade filters, complexity guards, DSR correction for multiple
testing across the entire discovery run, and Walk-Forward validation before
promoting candidates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vibe_quant.overfitting.dsr import DeflatedSharpeRatio

if TYPE_CHECKING:
    from datetime import date

    from vibe_quant.discovery.fitness import FitnessResult
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
    max_complexity: int = 8
    require_dsr: bool = True
    dsr_significance_level: float = 0.05
    require_wfa: bool = True
    wfa_min_efficiency: float = 0.5
    require_purged_kfold: bool = False


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
    reasons: list[str] = field(default_factory=list)
    dsr_result: DSRResult | None = None
    wfa_result: WFAResult | None = None
    kfold_result: CVResult | None = None


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
) -> DSRResult:
    """Apply DSR correction for the entire discovery run.

    num_trials should be the total number of candidates evaluated across
    all generations, accounting for the full multiple-testing burden.

    Args:
        observed_sharpe: Best candidate's Sharpe ratio.
        num_trials: Total candidates evaluated in the discovery run.
        num_observations: Number of bars/periods in the backtest.
        significance_level: p-value threshold for significance.
        skewness: Return distribution skewness.
        kurtosis: Return distribution kurtosis.

    Returns:
        DSRResult with significance test.
    """
    dsr = DeflatedSharpeRatio(significance_level=significance_level)
    result = dsr.calculate(
        observed_sharpe=observed_sharpe,
        num_trials=num_trials,
        num_observations=num_observations,
        skewness=skewness,
        kurtosis=kurtosis,
    )
    logger.info(
        "DSR: sharpe=%.3f trials=%d p=%.4f significant=%s",
        observed_sharpe,
        num_trials,
        result.p_value,
        result.is_significant,
    )
    return result


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
    wfa: WalkForwardAnalysis | None = None,
    wfa_strategy_id: str = "",
    wfa_data_start: date | None = None,
    wfa_data_end: date | None = None,
    wfa_param_grid: dict[str, list[object]] | None = None,
    kfold_cv: PurgedKFoldCV | None = None,
    kfold_n_samples: int = 0,
    kfold_runner: KFoldRunner | None = None,
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
        wfa: WalkForwardAnalysis instance (required if require_wfa).
        wfa_strategy_id: Strategy ID for WFA.
        wfa_data_start: Data start date for WFA.
        wfa_data_end: Data end date for WFA.
        wfa_param_grid: Parameter grid for WFA.
        kfold_cv: PurgedKFoldCV instance (required if require_purged_kfold).
        kfold_n_samples: Number of samples for K-Fold.
        kfold_runner: Backtest runner for K-Fold.

    Returns:
        GuardrailResult with per-check verdicts and overall pass/fail.
    """
    reasons: list[str] = []

    # 1. Minimum trades
    min_trades_passed, min_trades_reason = check_min_trades(
        fitness.total_trades, config.min_trades
    )
    if min_trades_reason:
        reasons.append(min_trades_reason)

    # 2. Complexity
    complexity_passed, complexity_reason = check_complexity(
        num_genes, config.max_complexity
    )
    if complexity_reason:
        reasons.append(complexity_reason)

    # 3. DSR
    dsr_passed: bool | None = None
    dsr_result: DSRResult | None = None
    if config.require_dsr:
        dsr_result = apply_discovery_dsr(
            observed_sharpe=fitness.sharpe_ratio,
            num_trials=num_trials,
            num_observations=num_observations,
            significance_level=config.dsr_significance_level,
            skewness=skewness,
            kurtosis=kurtosis,
        )
        dsr_passed = dsr_result.is_significant
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

    # Overall verdict: all enabled checks must pass
    overall = min_trades_passed and complexity_passed
    if dsr_passed is not None:
        overall = overall and dsr_passed
    if wfa_passed is not None:
        overall = overall and wfa_passed
    if kfold_passed is not None:
        overall = overall and kfold_passed

    return GuardrailResult(
        passed=overall,
        min_trades_passed=min_trades_passed,
        complexity_passed=complexity_passed,
        dsr_passed=dsr_passed,
        wfa_passed=wfa_passed,
        kfold_passed=kfold_passed,
        reasons=reasons,
        dsr_result=dsr_result,
        wfa_result=wfa_result,
        kfold_result=kfold_result,
    )
