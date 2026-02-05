"""Screening pipeline for parallel parameter sweep backtesting.

This module implements the screening pipeline for fast parameter sweeps
using NautilusTrader in simplified execution mode. The pipeline:

1. Takes a StrategyDSL with sweep parameters
2. Generates parameter combinations (Cartesian product)
3. Runs backtests in parallel via multiprocessing
4. Computes performance metrics
5. Applies hard filters and ranks results
6. Stores results in SQLite sweep_results table
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.dsl.schema import StrategyDSL

logger = logging.getLogger(__name__)


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
class BacktestMetrics:
    """Performance metrics from a single backtest run.

    Attributes:
        parameters: Parameter combination used
        sharpe_ratio: Sharpe ratio
        sortino_ratio: Sortino ratio
        max_drawdown: Maximum drawdown as decimal
        total_return: Total return as decimal
        profit_factor: Gross profit / gross loss
        win_rate: Winning trades / total trades
        num_trades: Total number of trades
        total_fees: Total fees paid
        total_funding: Total funding payments
        execution_time_seconds: Time to run backtest
    """

    parameters: dict[str, float | int]
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    num_trades: int = 0
    total_fees: float = 0.0
    total_funding: float = 0.0
    execution_time_seconds: float = 0.0


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


def build_parameter_grid(
    sweep: dict[str, list[int] | list[float]],
) -> list[dict[str, float | int]]:
    """Build Cartesian product of sweep parameters.

    Args:
        sweep: Dictionary mapping parameter names to lists of values.
            Example: {"rsi.period": [7, 14, 21], "stop_loss.percent": [1.0, 2.0]}

    Returns:
        List of parameter dictionaries, one for each combination.
        Example: [{"rsi.period": 7, "stop_loss.percent": 1.0}, ...]
    """
    if not sweep:
        return [{}]

    # Get param names and values in consistent order
    param_names = list(sweep.keys())
    param_values = [sweep[name] for name in param_names]

    # Generate Cartesian product
    combinations: list[dict[str, float | int]] = []
    for values in itertools.product(*param_values):
        combo = dict(zip(param_names, values, strict=True))
        combinations.append(combo)

    return combinations


def filter_by_metrics(
    results: list[BacktestMetrics],
    filters: MetricFilters,
) -> list[BacktestMetrics]:
    """Apply hard metric filters to screening results.

    Args:
        results: List of backtest metrics
        filters: Filter thresholds

    Returns:
        Filtered list of results that pass all thresholds
    """
    filtered: list[BacktestMetrics] = []

    for r in results:
        # Check all filter conditions
        if r.sharpe_ratio < filters.min_sharpe:
            continue
        if r.profit_factor < filters.min_profit_factor:
            continue
        if r.max_drawdown > filters.max_drawdown:
            continue
        if r.num_trades < filters.min_trades:
            continue

        filtered.append(r)

    return filtered


def rank_by_sharpe(results: list[BacktestMetrics]) -> list[BacktestMetrics]:
    """Sort results by Sharpe ratio descending.

    Args:
        results: List of backtest metrics

    Returns:
        Sorted list with highest Sharpe first
    """
    return sorted(results, key=lambda r: r.sharpe_ratio, reverse=True)


def compute_pareto_front(
    results: list[BacktestMetrics],
) -> list[int]:
    """Compute Pareto-optimal indices using 3 objectives.

    Objectives (all maximized):
    - Sharpe ratio
    - 1 - max_drawdown (inverted so higher is better)
    - Profit factor

    A result is Pareto-optimal if no other result dominates it
    (i.e., is better in ALL objectives simultaneously).

    Args:
        results: List of backtest metrics

    Returns:
        List of indices into results that are Pareto-optimal
    """
    if not results:
        return []

    n = len(results)
    is_pareto = [True] * n

    for i in range(n):
        if not is_pareto[i]:
            continue

        r_i = results[i]
        obj_i = (
            r_i.sharpe_ratio,
            1.0 - r_i.max_drawdown,
            r_i.profit_factor,
        )

        for j in range(n):
            if i == j or not is_pareto[j]:
                continue

            r_j = results[j]
            obj_j = (
                r_j.sharpe_ratio,
                1.0 - r_j.max_drawdown,
                r_j.profit_factor,
            )

            # Check if j dominates i (j >= i in all, j > i in at least one)
            j_ge_i = all(obj_j[k] >= obj_i[k] for k in range(3))
            j_gt_i_any = any(obj_j[k] > obj_i[k] for k in range(3))

            if j_ge_i and j_gt_i_any:
                # j dominates i
                is_pareto[i] = False
                break

    return [i for i in range(n) if is_pareto[i]]


# Type alias for backtest runner function
BacktestRunner = Callable[[dict[str, float | int]], BacktestMetrics]


def _run_mock_backtest(params: dict[str, float | int]) -> BacktestMetrics:
    """Mock backtest runner for testing.

    This generates deterministic fake metrics based on parameter values.
    Used for testing the pipeline structure without NautilusTrader.

    Args:
        params: Parameter combination

    Returns:
        Mock backtest metrics
    """
    import hashlib

    # Create deterministic seed from params
    param_str = json.dumps(params, sort_keys=True)
    seed = int(hashlib.md5(param_str.encode()).hexdigest()[:8], 16)

    # Generate pseudo-random but deterministic metrics (decimal fractions)
    sharpe = (seed % 300) / 100.0 - 0.5  # Range: -0.5 to 2.5
    sortino = sharpe * 1.2
    max_dd = (seed % 30) / 100.0 + 0.02  # Range: 0.02 to 0.32 (2-32%)
    total_return = (seed % 40) / 100.0 - 0.10  # Range: -0.10 to 0.30 (-10% to 30%)
    pf = (seed % 200) / 100.0 + 0.5  # Range: 0.5 to 2.5
    win_rate = (seed % 40) / 100.0 + 0.30  # Range: 0.30 to 0.70 (30-70%)
    num_trades = (seed % 200) + 20  # Range: 20 to 220

    return BacktestMetrics(
        parameters=params,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        total_return=total_return,
        profit_factor=pf,
        win_rate=win_rate,
        num_trades=num_trades,
        total_fees=num_trades * 0.001,
        total_funding=num_trades * 0.0005,
        execution_time_seconds=0.01,
    )


class ScreeningPipeline:
    """Pipeline for parallel parameter sweep screening.

    The pipeline runs NautilusTrader backtests in simplified mode
    for fast parameter exploration. It:

    1. Compiles the DSL strategy
    2. Builds parameter grid from sweep section
    3. Runs backtests in parallel
    4. Filters and ranks results
    5. Stores results in database

    Example:
        pipeline = ScreeningPipeline(
            dsl=strategy_dsl,
            backtest_runner=run_backtest,  # Or None for mock
        )
        result = pipeline.run(filters=MetricFilters(min_sharpe=0.5))
    """

    def __init__(
        self,
        dsl: StrategyDSL,
        backtest_runner: BacktestRunner | None = None,
        max_workers: int | None = None,
    ) -> None:
        """Initialize screening pipeline.

        Args:
            dsl: Parsed StrategyDSL with sweep parameters
            backtest_runner: Function to run single backtest. Uses mock if None.
            max_workers: Max parallel workers. Defaults to cpu_count - 1.
        """
        self.dsl = dsl
        self._runner = backtest_runner or _run_mock_backtest
        self._max_workers = max_workers or max(1, cpu_count() - 1)

        # Build parameter grid
        self._param_grid = build_parameter_grid(dsl.sweep)

    @property
    def strategy_name(self) -> str:
        """Get strategy name from DSL."""
        return self.dsl.name

    @property
    def num_combinations(self) -> int:
        """Get total number of parameter combinations."""
        return len(self._param_grid)

    def run(
        self,
        filters: MetricFilters | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ScreeningResult:
        """Run the screening pipeline.

        Args:
            filters: Hard metric filters. Uses defaults if None.
            progress_callback: Optional callback(completed, total) for progress.

        Returns:
            ScreeningResult with filtered and ranked results
        """
        start_time = time.time()
        filters = filters or MetricFilters()

        logger.info(
            "Starting screening for %s with %d combinations, %d workers",
            self.strategy_name,
            self.num_combinations,
            self._max_workers,
        )

        # Run backtests in parallel
        all_results = self._run_parallel(progress_callback)

        # Apply filters
        filtered = filter_by_metrics(all_results, filters)
        logger.info(
            "Filtered %d/%d results pass hard filters",
            len(filtered),
            len(all_results),
        )

        # Rank by Sharpe
        ranked = rank_by_sharpe(filtered)

        # Compute Pareto front
        pareto_indices = compute_pareto_front(ranked)
        logger.info("Found %d Pareto-optimal results", len(pareto_indices))

        execution_time = time.time() - start_time

        return ScreeningResult(
            strategy_name=self.strategy_name,
            total_combinations=self.num_combinations,
            passed_filters=len(filtered),
            execution_time_seconds=execution_time,
            results=ranked,
            pareto_optimal_indices=pareto_indices,
        )

    def _run_parallel(
        self,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[BacktestMetrics]:
        """Run backtests in parallel using ProcessPoolExecutor.

        Args:
            progress_callback: Optional progress callback

        Returns:
            List of all backtest results
        """
        results: list[BacktestMetrics] = []
        total = self.num_combinations
        completed = 0

        # Use ProcessPoolExecutor for CPU-bound backtests
        with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
            # Submit all tasks
            future_to_params = {
                executor.submit(self._runner, params): params
                for params in self._param_grid
            }

            # Collect results as they complete
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Log error but continue with other backtests
                    logger.warning(
                        "Backtest failed for params %s: %s",
                        params,
                        e,
                    )
                    # Add failed result with zero metrics
                    results.append(
                        BacktestMetrics(
                            parameters=params,
                            sharpe_ratio=float("-inf"),
                        )
                    )

                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

        return results

    def save_results(
        self,
        result: ScreeningResult,
        state_manager: StateManager,
        run_id: int,
    ) -> None:
        """Save screening results to database.

        Args:
            result: ScreeningResult to save
            state_manager: StateManager instance
            run_id: Backtest run ID to associate results with
        """
        # Convert results to dicts for batch insert
        result_dicts: list[dict[str, Any]] = []

        for i, metrics in enumerate(result.results):
            is_pareto = i in result.pareto_optimal_indices
            result_dicts.append({
                "parameters": metrics.parameters,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "max_drawdown": metrics.max_drawdown,
                "total_return": metrics.total_return,
                "profit_factor": metrics.profit_factor,
                "win_rate": metrics.win_rate,
                "total_trades": metrics.num_trades,
                "total_fees": metrics.total_fees,
                "total_funding": metrics.total_funding,
                "is_pareto_optimal": is_pareto,
            })

        # Batch save
        state_manager.save_sweep_results_batch(run_id, result_dicts)

        logger.info(
            "Saved %d screening results for run %d",
            len(result_dicts),
            run_id,
        )


def create_screening_pipeline(
    dsl: StrategyDSL,
    use_mock: bool = True,
    max_workers: int | None = None,
) -> ScreeningPipeline:
    """Factory function to create a screening pipeline.

    Args:
        dsl: Parsed StrategyDSL
        use_mock: If True, use mock backtest runner (for testing)
        max_workers: Max parallel workers

    Returns:
        Configured ScreeningPipeline
    """
    runner = _run_mock_backtest if use_mock else None
    return ScreeningPipeline(dsl=dsl, backtest_runner=runner, max_workers=max_workers)
