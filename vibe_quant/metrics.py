"""Shared performance metrics used across screening and validation pipelines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

#: Finite stand-in for an infinite profit factor (profits but no losing trade).
#: Every computed PF is clamped to this value so ``inf``/``NaN`` never reach the
#: DB, Pareto ranking or the UI. Discovery fitness clamps PF to 5 anyway, so any
#: value >= 5 scores identically there; 100 is chosen to read as "no losses" in
#: the UI rather than as a plausible measured PF.
PROFIT_FACTOR_CAP: float = 100.0


def profit_factor(trade_pnls: Iterable[float]) -> float:
    """Trade profit factor: gross profit / gross loss over closed trades.

    ``trade_pnls`` are per-trade NET PnLs (after fees and, where modeled,
    funding/slippage). This is the single PF definition shared by screening
    and validation (bd vibe-quant-e70tl.7 — previously NT's daily-return PF
    was reported, which is a different statistic).

    Returns:
        - ``0.0`` when there are no trades or no winning trades,
        - :data:`PROFIT_FACTOR_CAP` when there are winners but no losers,
        - otherwise ``gross_profit / gross_loss`` clamped to the cap.
        Non-finite PnLs are ignored.
    """
    gross_profit = 0.0
    gross_loss = 0.0
    for pnl in trade_pnls:
        if not math.isfinite(pnl):
            continue
        if pnl > 0.0:
            gross_profit += pnl
        elif pnl < 0.0:
            gross_loss -= pnl
    if gross_profit <= 0.0:
        return 0.0
    if gross_loss <= 0.0:
        return PROFIT_FACTOR_CAP
    return min(gross_profit / gross_loss, PROFIT_FACTOR_CAP)


_DAY_NS = 86_400 * 1_000_000_000


def daily_balance_returns(
    starting_balance: float,
    cash_events: Iterable[tuple[int, float]],
    start_ns: int,
    end_ns: int,
) -> dict[int, float]:
    """Daily returns of the realized account balance over the whole window.

    Mirrors NT's ``PortfolioAnalyzer`` portfolio returns (UTC daily last
    balance, forward-filled, ``pct_change``) but (a) lets callers add cash
    flows NT never sees (funding, modeled slippage) and (b) spans the full
    backtest window instead of stopping at the last fill, so idle days after
    the last trade count as zero-return days.

    Args:
        starting_balance: Account balance at ``start_ns``.
        cash_events: ``(ts_ns, delta)`` realized balance changes (trade net
            PnL at close, funding payments as negative deltas, ...). Events
            outside the window are clamped to its first/last day.
        start_ns / end_ns: Backtest window (end exclusive).

    Returns:
        ``{day_start_ns: return}`` for every day after the first, the format
        NT's pyo3 ``SharpeRatio.calculate_from_returns`` expects.
    """
    first_day = start_ns // _DAY_NS
    last_day = max(first_day, (end_ns - 1) // _DAY_NS)
    n_days = last_day - first_day + 1
    per_day = [0.0] * n_days
    for ts, delta in cash_events:
        idx = min(max(ts // _DAY_NS - first_day, 0), n_days - 1)
        per_day[idx] += delta
    returns: dict[int, float] = {}
    balance = starting_balance + per_day[0]
    for i in range(1, n_days):
        prev = balance
        balance += per_day[i]
        if prev != 0.0:
            ret = balance / prev - 1.0
            if math.isfinite(ret):
                returns[(first_day + i) * _DAY_NS] = ret
    return returns


@dataclass
class PerformanceMetrics:
    """Core performance metrics common to all backtest result types.

    Both :class:`screening.pipeline.BacktestMetrics` and
    :class:`validation.runner.ValidationResult` extend this base so that
    dashboard, overfitting filters, and other consumers can work with a
    single set of canonical field names.

    Field units:
        total_return: Stored as a decimal fraction (0.15 = 15%, -0.05 = -5%).
        profit_factor: Trade PF from :func:`profit_factor` (capped at
            :data:`PROFIT_FACTOR_CAP`, 0.0 with no trades).
    """

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0  # Decimal fraction: 0.15 means 15%
    profit_factor: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    total_fees: float = 0.0
    total_funding: float = 0.0
    execution_time_seconds: float = 0.0
    skewness: float = 0.0  # Return distribution skewness (0 = symmetric)
    kurtosis: float = 3.0  # Return distribution kurtosis (3 = normal)
    trade_returns: tuple[float, ...] = ()  # Per-trade PnL returns (fraction)
    # Funding settlements charged at the fallback rate because the archive
    # had no rate for them (0 = funding fully from archived rates).
    funding_fallback_settlements: int = 0
