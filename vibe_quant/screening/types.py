"""Type definitions for screening pipeline.

Dataclasses and type aliases used across the screening module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from vibe_quant.metrics import PerformanceMetrics


@dataclass(frozen=True, slots=True)
class MetricFilters:
    """Hard filters for screening results.

    These are applied before ranking to eliminate clearly unsuitable
    parameter combinations.

    Attributes:
        min_sharpe: Minimum Sharpe ratio (default 0.0)
        min_profit_factor: Minimum profit factor (default 1.0)
        max_drawdown: Maximum drawdown as decimal (default 0.3 = 30%)
        min_trades: Minimum number of trades (default 50)
    """

    min_sharpe: float = 0.0
    min_profit_factor: float = 1.0
    max_drawdown: float = 0.3
    min_trades: int = 50


@dataclass
class BacktestMetrics(PerformanceMetrics):
    """Performance metrics from a single screening backtest run.

    Extends :class:`~vibe_quant.metrics.PerformanceMetrics` with the
    specific parameter combination that produced these results.
    """

    parameters: dict[str, float | int] = field(default_factory=dict)


@dataclass
class ScreeningResult:
    """Result of a screening pipeline run.

    Attributes:
        strategy_name: Name of the strategy
        total_combinations: Total parameter combinations tested
        passed_filters: Number passing hard filters
        execution_time_seconds: Total pipeline execution time
        results: List of BacktestMetrics sorted by ranking
        pareto_optimal_indices: Indices of Pareto-optimal results
    """

    strategy_name: str
    total_combinations: int
    passed_filters: int
    execution_time_seconds: float
    results: list[BacktestMetrics] = field(default_factory=list)
    pareto_optimal_indices: list[int] = field(default_factory=list)


# Type alias for backtest runner function
BacktestRunner = Callable[[dict[str, float | int]], BacktestMetrics]
