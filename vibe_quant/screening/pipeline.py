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

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import TYPE_CHECKING, Any

from vibe_quant.screening.grid import (
    build_parameter_grid,
    compute_pareto_front,
    filter_by_metrics,
    rank_by_sharpe,
)
from vibe_quant.screening.types import (
    BacktestMetrics,
    BacktestRunner,
    MetricFilters,
    ScreeningResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.dsl.schema import StrategyDSL

logger = logging.getLogger(__name__)


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
    n_trades = (seed % 200) + 20  # Range: 20 to 220

    return BacktestMetrics(
        parameters=params,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        total_return=total_return,
        profit_factor=pf,
        win_rate=win_rate,
        total_trades=n_trades,
        total_fees=n_trades * 0.001,
        total_funding=n_trades * 0.0005,
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

        # Rank ALL results by Sharpe (save everything so user can inspect)
        ranked_all = rank_by_sharpe(all_results)

        # Compute Pareto front on filtered results only
        ranked_filtered = rank_by_sharpe(filtered)
        pareto_indices_filtered = compute_pareto_front(ranked_filtered)

        # Map pareto indices back to the all-results list
        pareto_params = {
            json.dumps(ranked_filtered[i].parameters, sort_keys=True)
            for i in pareto_indices_filtered
        }
        pareto_indices = [
            i for i, r in enumerate(ranked_all)
            if json.dumps(r.parameters, sort_keys=True) in pareto_params
        ]
        logger.info("Found %d Pareto-optimal results", len(pareto_indices))

        execution_time = time.time() - start_time

        return ScreeningResult(
            strategy_name=self.strategy_name,
            total_combinations=self.num_combinations,
            passed_filters=len(filtered),
            execution_time_seconds=execution_time,
            results=ranked_all,
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
        # Use set for O(1) membership test instead of O(k) list scan
        pareto_set = frozenset(result.pareto_optimal_indices)

        for i, metrics in enumerate(result.results):
            is_pareto = i in pareto_set
            result_dicts.append({
                "parameters": metrics.parameters,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "max_drawdown": metrics.max_drawdown,
                "total_return": metrics.total_return,
                "profit_factor": metrics.profit_factor,
                "win_rate": metrics.win_rate,
                "total_trades": metrics.total_trades,
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
    use_mock: bool = False,
    max_workers: int | None = None,
    symbols: list[str] | None = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    catalog_path: str | None = None,
) -> ScreeningPipeline:
    """Factory function to create a screening pipeline.

    When use_mock=False, creates a real NautilusTrader BacktestNode-based
    runner for actual screening. Requires symbols and date range.

    Args:
        dsl: Parsed StrategyDSL
        use_mock: If True, use mock backtest runner (for testing)
        max_workers: Max parallel workers
        symbols: List of symbols (required when use_mock=False)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        catalog_path: Path to ParquetDataCatalog (uses default if None)

    Returns:
        Configured ScreeningPipeline
    """
    if use_mock:
        runner: BacktestRunner = _run_mock_backtest
    else:
        from vibe_quant.screening.nt_runner import NTScreeningRunner

        effective_symbols = symbols or ["BTCUSDT"]
        # Convert DSL to dict for pickling across process boundaries
        dsl_dict = _dsl_to_dict(dsl)
        runner = NTScreeningRunner(
            dsl_dict=dsl_dict,
            symbols=effective_symbols,
            start_date=start_date,
            end_date=end_date,
            catalog_path=catalog_path,
        )
    return ScreeningPipeline(dsl=dsl, backtest_runner=runner, max_workers=max_workers)


def _dsl_to_dict(dsl: StrategyDSL) -> dict[str, Any]:
    """Convert a StrategyDSL to a serializable dict for pickling.

    Handles Pydantic BaseModel (model_dump), dataclasses (asdict),
    and plain dicts.

    Args:
        dsl: Parsed StrategyDSL object.

    Returns:
        Dictionary representation suitable for validate_strategy_dict().
    """
    # Pydantic BaseModel — StrategyDSL is a Pydantic v2 model
    if hasattr(dsl, "model_dump"):
        return dsl.model_dump()

    import dataclasses

    def _to_dict(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        elif isinstance(obj, list):
            return [_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items()}
        return obj

    return dict(_to_dict(dsl))  # type: ignore[arg-type]
