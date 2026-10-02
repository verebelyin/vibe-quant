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
