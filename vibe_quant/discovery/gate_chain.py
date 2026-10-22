"""Champion gate vocabulary shared by the pipeline, power harness and routes.

Behaviour-preserving home (rabmo.6 T6) for the pieces that carry no gate state:
the window result types, the holdout / cross-window / WFA window builders, the
DSR input checks and the rejection record. ``vibe_quant.discovery.pipeline``
imports this module and re-exports the three result classes.

Runtime imports stay limited to config_check, genome and utils so that
``pipeline.py`` can import this module without a cycle; the config / fitness /
operator types are TYPE_CHECKING-only (the PEP 695 ``Candidate`` alias is lazy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vibe_quant.discovery.config_check import (
    cross_window_ranges_for,
    wfa_window_ranges_for,
)
from vibe_quant.discovery.genome import chromosome_to_dsl
from vibe_quant.utils import compute_day_count

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vibe_quant.discovery.fitness import FitnessResult
    from vibe_quant.discovery.operators import StrategyChromosome
    from vibe_quant.discovery.pipeline import DiscoveryConfig
    from vibe_quant.metrics import DailyReturns

type Candidate = tuple[StrategyChromosome, FitnessResult]

# ---------------------------------------------------------------------------
# Window results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HoldoutResult:
    """Metrics of one strategy on one (holdout / shifted / rolling) window.

    Attributes:
        sharpe_ratio: Sharpe ratio.
        max_drawdown: Max drawdown.
        profit_factor: Profit factor.
        total_trades: Trade count.
        total_return: Total return.
    """

    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    total_trades: int
    total_return: float
    # symbol_agg="worst" only: portfolio-comparable trade sum and per-symbol
    # {sharpe, trades, return} (total_trades above is the MIN symbol).
    trades_sum: int | None = None
    symbol_metrics: dict[str, dict[str, float | int]] | None = None
    # Report-only liquidation count of this window (bead vibe-quant-2gyk1.1.9,
    # R5); None when the run's mode is off (the bt dict has no key).
    liquidation_count: int | None = None


@dataclass(frozen=True, slots=True)
class CrossWindowResult:
    """Cross-window validation result for a single strategy.

    Attributes:
        window_results: Per SHIFTED window metrics (the in-sample training
            window is excluded -- it never counts as a pass).
        windows_passed: Number of shifted windows that passed.
        total_windows: Number of shifted windows evaluated.
        passed: Whether ``windows_passed >= required`` shifted windows.
        window_dates: (start, end) per shifted window, parallel to results.
        offsets_months: Month offset per shifted window, parallel to results.
        required: Shifted windows that had to pass.
    """

    window_results: list[HoldoutResult]
    windows_passed: int
    total_windows: int
    passed: bool
    window_dates: list[tuple[str, str]] = field(default_factory=list)
    offsets_months: list[int] = field(default_factory=list)
    required: int = 0


@dataclass(frozen=True, slots=True)
class WFARollingResult:
    """Rolling-window stability check (over the TRAIN range) for one strategy.

    Attributes:
        oos_windows: Per-window HoldoutResult for each rolling window.
        window_dates: (start, end) for each rolling window.
        windows_profitable: Number of windows with total_return > 0.
        windows_sharpe_positive: Number of windows with sharpe > 0.
        total_windows: Total rolling windows.
        consistency: Fraction of profitable (return-based) windows — the gate.
        sharpe_consistency: Fraction of sharpe-positive windows — exposure only.
        passed: Whether consistency >= wfa_min_consistency (return-based).
    """

    oos_windows: list[HoldoutResult]
    window_dates: list[tuple[str, str]]
    windows_profitable: int
    windows_sharpe_positive: int
    total_windows: int
    consistency: float
    sharpe_consistency: float
    passed: bool


def metrics_to_holdout_result(bt: Mapping[str, float | int]) -> HoldoutResult:
    """Window metrics dict -> HoldoutResult, NaN coerced to failing values."""
    sharpe = float(bt.get("sharpe_ratio", 0.0))
    max_dd = float(bt.get("max_drawdown", 1.0))
    pf = float(bt.get("profit_factor", 0.0))
    trades = int(bt.get("total_trades", 0))
    ret = float(bt.get("total_return", 0.0))
    return HoldoutResult(
        sharpe_ratio=0.0 if math.isnan(sharpe) else sharpe,
        max_drawdown=1.0 if math.isnan(max_dd) else max_dd,
        profit_factor=0.0 if math.isnan(pf) else pf,
        total_trades=trades,
        total_return=0.0 if math.isnan(ret) else ret,
        trades_sum=int(bt["trades_sum"]) if "trades_sum" in bt else None,
        symbol_metrics=bt.get("symbol_metrics"),  # type: ignore[arg-type]
        liquidation_count=int(bt["liquidation_count"]) if "liquidation_count" in bt else None,
    )


FAILED_WINDOW = HoldoutResult(
    sharpe_ratio=0.0, max_drawdown=1.0, profit_factor=0.0, total_trades=0, total_return=0.0,
)

# ---------------------------------------------------------------------------
# Holdout / cross-window / WFA window planning
# ---------------------------------------------------------------------------


def symbol_mtm_stats(
    symbol_mtm_daily: dict[str, DailyReturns | None] | None,
) -> dict[str, dict[str, float] | None] | None:
    """Per-symbol DSR Sharpe inputs from each symbol's MARK-TO-MARKET series.

    One dict per symbol (sharpe / skewness / kurtosis / observations / mean
    via :func:`daily_sharpe_inputs`), built from ``symbol_mtm_daily`` so the worst-mode per-symbol PSR floor judges the
    SAME series the pooled test does (vibe-quant-y9ywf). A missing or empty
    symbol series maps to ``None`` so the floor fails closed; a ``None`` input
    (portfolio mode) yields ``None`` -- no per-symbol floor applies.
    """
    if symbol_mtm_daily is None:
        return None
    from vibe_quant.overfitting.dsr import daily_sharpe_inputs

    stats: dict[str, dict[str, float] | None] = {}
    for sym, series in symbol_mtm_daily.items():
        if series is None or len(series) == 0:
            stats[sym] = None
            continue
        sharpe, skewness, kurtosis = daily_sharpe_inputs(series.values)
        stats[sym] = {
            "sharpe": sharpe,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "observations": len(series),
            "mean": sum(series.values) / len(series),
        }
    return stats


def worst_symbol_gate_reasons(fitness: FitnessResult, min_trades: int) -> list[str]:
    """Hard train gate for ``symbol_agg="worst"`` champions (vibe-quant-ox73t).

    The soft GA score lets a genome with a losing symbol rank; a champion must
    still have EVERY symbol profitable (train, worst-of-windows) with at least
    ``min_trades`` trades. Empty list = pass (also for portfolio mode).
    """
    if fitness.symbol_stats is None:
        return []
    reasons: list[str] = []
    for sym, (ret, trades) in fitness.symbol_stats.items():
        if not ret > 0:
            reasons.append(f"worst-symbol train return <= 0 ({sym})")
        if trades < min_trades:
            reasons.append(f"worst-symbol train trades < {min_trades} ({sym})")
    return reasons


def holdout_reasons(hr: HoldoutResult, min_trades: int, min_sharpe: float) -> list[str]:
    """Failure reasons of the holdout gate (empty = pass)."""
    reasons: list[str] = []
    if hr.total_trades < min_trades:
        reasons.append(f"Holdout: {hr.total_trades} trades < {min_trades}")
    if not hr.total_return > 0:
        reasons.append(f"Holdout: return {hr.total_return:.2%} <= 0")
    if not hr.sharpe_ratio > min_sharpe:
        reasons.append(f"Holdout: sharpe {hr.sharpe_ratio:.2f} <= {min_sharpe:.2f}")
    return reasons


def holdout_min_trades(config: DiscoveryConfig) -> int:
    """Holdout trade floor (explicit, or half the train-gate trade rate)."""
    if config.holdout_min_trades is not None:
        return config.holdout_min_trades
    train_days = compute_day_count(config.start_date, config.end_date)
    holdout_days = compute_day_count(config.holdout_start_date, config.holdout_end_date)
    if not train_days or not holdout_days:
        return 1
    return max(1, int(config.min_trades * holdout_days / (2 * train_days)))


def cross_window_ranges(config: DiscoveryConfig) -> list[tuple[int, str, str]]:
    """Shifted cross-windows ``(offset_months, start, end)`` inside the TRAIN range.

    Window k starts ``offset_k`` months after the train start and ends at
    the train end -- shifting the END forward would run into the holdout
    (or past the data), so windows are clipped to the train range.

    Raises:
        ValueError: A window is shorter than ``_MIN_CROSS_WINDOW_DAYS`` or
            (defensively) overlaps the holdout.
    """
    return cross_window_ranges_for(
        config.cross_window_months, config.start_date, config.end_date,
        config.holdout_start_date if config.has_holdout else "",
    )


def wfa_window_ranges(
    config: DiscoveryConfig, range_start: str, range_end: str
) -> list[tuple[str, str]]:
    """Rolling ``wfa_oos_step_days`` windows tiling [range_start, range_end]."""
    return wfa_window_ranges_for(
        config.wfa_oos_step_days, range_start, range_end,
        config.holdout_start_date if config.has_holdout else "",
    )


def dsr_requires_returns(config: DiscoveryConfig, backtest_fn: object) -> bool:
    """Whether DSR tests the MARK-TO-MARKET series for this backtest fn."""
    from vibe_quant.discovery.backtest_fn import NTBacktestFn

    return isinstance(backtest_fn, NTBacktestFn) and config.require_dsr


# ---------------------------------------------------------------------------
# Rejection record
# ---------------------------------------------------------------------------


def rejection_record(
    chrom: StrategyChromosome,
    fitness: FitnessResult,
    stage: str,
    reasons: list[str],
    *,
    timeframe: str,
    dsr: dict[str, object] | None = None,
    buy_hold: dict[str, object] | None = None,
) -> dict[str, object]:
    """Record a rejected candidate (persisted with the run, shown in the UI)."""
    entry: dict[str, object] = {
        "uid": chrom.uid,
        "stage": stage,
        "score": round(fitness.adjusted_score, 4),
        "sharpe": round(fitness.sharpe_ratio, 3),
        "trades": fitness.total_trades,
        "reasons": list(reasons),
        # DSL kept so a near-miss can still be backtested/validated by hand.
        "dsl": {**chromosome_to_dsl(chrom), "timeframe": timeframe},
    }
    if dsr is not None:
        entry["dsr"] = dsr
    if buy_hold is not None:
        entry["buy_hold"] = buy_hold
    # Fee drag on the rejection row (bd vibe-quant-2gyk1.4): most runs end
    # with zero champions, so this is often the only place a user sees that
    # fees, not the signal, killed a high-turnover genome. Always written
    # (JSON null when unavailable).
    entry["fee_drag"] = fitness.fee_drag
    entry["fees_pct_equity_yr"] = fitness.fees_pct_equity_yr
    # Report-only liquidation counts (bead vibe-quant-2gyk1.1.9, R1): written
    # only when the mode is on, so rejection rows stay unchanged off (R4).
    if fitness.liquidation_count is not None:
        entry["liquidation_count"] = fitness.liquidation_count
    if fitness.liquidation_forfeited is not None:
        entry["liquidation_forfeited"] = fitness.liquidation_forfeited
    return entry
