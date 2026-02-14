"""Shared performance metrics used across screening and validation pipelines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Core performance metrics common to all backtest result types.

    Both :class:`screening.pipeline.BacktestMetrics` and
    :class:`validation.runner.ValidationResult` extend this base so that
    dashboard, overfitting filters, and other consumers can work with a
    single set of canonical field names.

    Field units:
        total_return: Stored as a decimal fraction (0.15 = 15%, -0.05 = -5%).
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
