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
from pathlib import Path
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

    Optimized with inlined comparisons and pre-extracted attributes
    to eliminate function-call overhead and tuple allocation in the
    inner loop. ~3-4x faster than the original implementation.

    Args:
        results: List of backtest metrics

    Returns:
        List of indices into results that are Pareto-optimal
    """
    if not results:
        return []

    n = len(results)
    if n == 1:
        return [0]

    # Pre-extract objectives to avoid repeated attribute access in O(n²) loop
    sharpes = [r.sharpe_ratio for r in results]
    inv_dds = [1.0 - r.max_drawdown for r in results]
    pfs = [r.profit_factor for r in results]

    is_pareto = [True] * n

    for i in range(n):
        if not is_pareto[i]:
            continue

        s_i = sharpes[i]
        d_i = inv_dds[i]
        p_i = pfs[i]

        for j in range(n):
            if i == j or not is_pareto[j]:
                continue

            s_j = sharpes[j]
            d_j = inv_dds[j]
            p_j = pfs[j]

            # Inlined dominance check: j >= i in all AND j > i in at least one
            if (s_j >= s_i and d_j >= d_i and p_j >= p_i
                    and (s_j > s_i or d_j > d_i or p_j > p_i)):
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


class NTScreeningRunner:
    """Real NautilusTrader backtest runner for screening mode.

    Runs a single-parameter-combination backtest using BacktestNode with
    screening venue config (no latency, simple fill model). Designed to be
    picklable for ProcessPoolExecutor.

    This is the real screening runner that replaces the mock. It compiles
    the strategy DSL, creates a BacktestNode, runs it, and extracts metrics.
    """

    def __init__(
        self,
        dsl_dict: dict[str, Any],
        symbols: list[str],
        start_date: str,
        end_date: str,
        catalog_path: str | None = None,
    ) -> None:
        """Initialize NTScreeningRunner.

        Args:
            dsl_dict: Strategy DSL as dict (picklable, unlike StrategyDSL).
            symbols: List of symbols to screen.
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD).
            catalog_path: Path to ParquetDataCatalog. Uses default if None.
        """
        self._dsl_dict = dsl_dict
        self._symbols = symbols
        self._start_date = start_date
        self._end_date = end_date
        self._catalog_path = catalog_path

        # Cached per-process compilation results (populated on first __call__)
        self._compiled = False
        self._module_path: str = ""
        self._strategy_cls_name: str = ""
        self._config_cls_name: str = ""

    def __call__(self, params: dict[str, float | int]) -> BacktestMetrics:
        """Run a single screening backtest with the given parameters.

        Args:
            params: Parameter combination to test.

        Returns:
            BacktestMetrics from the backtest run.
        """
        start_time = time.time()
        try:
            return self._run_backtest(params, start_time)
        except Exception as e:
            logger.warning("NT screening backtest failed for %s: %s", params, e)
            return BacktestMetrics(
                parameters=params,
                sharpe_ratio=float("-inf"),
                execution_time_seconds=time.time() - start_time,
            )

    def _ensure_compiled(self) -> None:
        """Parse and compile DSL once per worker process.

        Results are cached in instance attributes so subsequent calls
        to _run_backtest skip recompilation.
        """
        if self._compiled:
            return

        from vibe_quant.data.catalog import (
            DEFAULT_CATALOG_PATH,
            INSTRUMENT_CONFIGS,
            CatalogManager,
            create_instrument,
        )
        from vibe_quant.dsl.compiler import StrategyCompiler
        from vibe_quant.dsl.parser import validate_strategy_dict

        dsl = validate_strategy_dict(self._dsl_dict)
        compiler = StrategyCompiler()
        compiler.compile_to_module(dsl)  # registers in sys.modules

        class_name = "".join(word.capitalize() for word in dsl.name.split("_"))
        self._module_path = f"vibe_quant.dsl.generated.{dsl.name}"
        self._strategy_cls_name = f"{class_name}Strategy"
        self._config_cls_name = f"{class_name}Config"

        # Cache parsed DSL fields needed for data config
        self._all_timeframes: set[str] = {dsl.timeframe}
        self._all_timeframes.update(dsl.additional_timeframes)
        for ind_config in dsl.indicators.values():
            if ind_config.timeframe:
                self._all_timeframes.add(ind_config.timeframe)

        # Catalog path and instruments (once per process)
        self._resolved_catalog_path = (
            Path(self._catalog_path) if self._catalog_path else DEFAULT_CATALOG_PATH
        )
        catalog_mgr = CatalogManager(self._resolved_catalog_path)
        for symbol in self._symbols:
            if symbol in INSTRUMENT_CONFIGS:
                instrument = create_instrument(symbol)
                catalog_mgr.write_instrument(instrument)

        self._compiled = True

    def _run_backtest(
        self, params: dict[str, float | int], start_time: float
    ) -> BacktestMetrics:
        """Execute the NautilusTrader backtest."""
        from nautilus_trader.backtest.node import BacktestNode
        from nautilus_trader.config import (
            BacktestDataConfig,
            BacktestEngineConfig,
            BacktestRunConfig,
            ImportableStrategyConfig,
        )
        from nautilus_trader.core.nautilus_pyo3 import (
            MaxDrawdown,
            ProfitFactor,
            SharpeRatio,
            SortinoRatio,
            WinRate,
        )

        from vibe_quant.data.catalog import (
            INTERVAL_TO_AGGREGATION,
        )
        from vibe_quant.validation.venue import (
            create_backtest_venue_config,
            create_venue_config_for_screening,
        )

        # Compile DSL once per worker process
        self._ensure_compiled()

        module_path = self._module_path
        strategy_cls_name = self._strategy_cls_name
        config_cls_name = self._config_cls_name
        catalog_path = self._resolved_catalog_path

        # Strategy configs (with parameter overrides)
        strategy_configs: list[ImportableStrategyConfig] = []
        for symbol in self._symbols:
            instrument_id = f"{symbol}-PERP.BINANCE"
            config_dict: dict[str, Any] = {"instrument_id": instrument_id}
            config_dict.update(params)
            strategy_configs.append(
                ImportableStrategyConfig(
                    strategy_path=f"{module_path}:{strategy_cls_name}",
                    config_path=f"{module_path}:{config_cls_name}",
                    config=config_dict,
                )
            )

        # Data configs
        data_configs: list[BacktestDataConfig] = []
        for symbol in self._symbols:
            instrument_id = f"{symbol}-PERP.BINANCE"
            for tf in sorted(self._all_timeframes):
                if tf not in INTERVAL_TO_AGGREGATION:
                    continue
                step, agg = INTERVAL_TO_AGGREGATION[tf]
                data_configs.append(
                    BacktestDataConfig(
                        catalog_path=str(catalog_path.resolve()),
                        data_cls="nautilus_trader.model.data:Bar",
                        instrument_id=instrument_id,
                        bar_spec=f"{step}-{agg.name}-LAST",
                        start_time=self._start_date,
                        end_time=self._end_date,
                    )
                )

        if not data_configs:
            return BacktestMetrics(
                parameters=params,
                sharpe_ratio=float("-inf"),
                execution_time_seconds=time.time() - start_time,
            )

        # Screening venue config: no latency, simple fills
        venue_config = create_venue_config_for_screening()
        bt_venue_config = create_backtest_venue_config(venue_config)

        engine_config = BacktestEngineConfig(
            strategies=strategy_configs,
            run_analysis=True,
        )

        bt_run_config = BacktestRunConfig(
            engine=engine_config,
            venues=[bt_venue_config],
            data=data_configs,
            start=self._start_date,
            end=self._end_date,
            dispose_on_completion=False,
        )

        # Run the backtest
        node = BacktestNode(configs=[bt_run_config])
        try:
            node.build()

            # Register statistics
            stats = [SharpeRatio(), SortinoRatio(), MaxDrawdown(), WinRate(), ProfitFactor()]
            for engine in node.get_engines():
                analyzer = engine.kernel.portfolio.analyzer
                for stat in stats:
                    analyzer.register_statistic(stat)

            node.run()

            engine = node.get_engine(bt_run_config.id)
            bt_result = engine.get_result()

            return self._extract_metrics(params, bt_result, engine, start_time)
        finally:
            node.dispose()

    def _extract_metrics(
        self,
        params: dict[str, float | int],
        bt_result: Any,
        engine: Any,
        start_time: float,
    ) -> BacktestMetrics:
        """Extract BacktestMetrics from NT BacktestResult."""
        metrics = BacktestMetrics(
            parameters=params,
            execution_time_seconds=time.time() - start_time,
        )

        if bt_result is None:
            return metrics

        metrics.num_trades = bt_result.total_positions

        # Extract from PnL stats first (more comprehensive, includes total return)
        _known_pnl_keys = {"pnl (total)", "pnl% (total)", "sharpe", "sortino",
                           "max drawdown", "win rate", "profit factor"}
        stats_pnls = bt_result.stats_pnls or {}
        for _currency, pnl_stats in stats_pnls.items():
            for key, value in pnl_stats.items():
                if value is None:
                    continue
                key_lower = key.lower()
                fval = float(value)
                if key_lower == "pnl% (total)":
                    metrics.total_return = fval
                elif "sharpe" in key_lower:
                    metrics.sharpe_ratio = fval
                elif "sortino" in key_lower:
                    metrics.sortino_ratio = fval
                elif key_lower == "max drawdown":
                    metrics.max_drawdown = abs(fval)
                elif key_lower == "win rate":
                    metrics.win_rate = fval
                elif key_lower == "profit factor":
                    metrics.profit_factor = fval
                elif not any(k in key_lower for k in _known_pnl_keys):
                    logger.debug("Unmatched PnL stats key: %s = %s", key, value)

        # Fill from returns stats only if not already set by PnL stats
        _known_returns_keys = {"sharpe", "sortino", "max drawdown", "win rate",
                               "profit factor"}
        stats_returns = bt_result.stats_returns or {}
        for key, value in stats_returns.items():
            if value is None:
                continue
            key_lower = key.lower()
            fval = float(value)
            if "sharpe" in key_lower and metrics.sharpe_ratio == 0.0:
                metrics.sharpe_ratio = fval
            elif "sortino" in key_lower and metrics.sortino_ratio == 0.0:
                metrics.sortino_ratio = fval
            elif "max drawdown" in key_lower and metrics.max_drawdown == 0.0:
                metrics.max_drawdown = abs(fval)
            elif key_lower == "win rate" and metrics.win_rate == 0.0:
                metrics.win_rate = fval
            elif key_lower == "profit factor" and metrics.profit_factor == 0.0:
                metrics.profit_factor = fval
            elif not any(k in key_lower for k in _known_returns_keys):
                logger.debug("Unmatched returns stats key: %s = %s", key, value)

        # Extract fees from closed positions
        try:
            positions = engine.kernel.cache.positions()
            total_fees = 0.0
            for pos in positions:
                if pos.is_closed:
                    total_fees += sum(abs(float(c)) for c in pos.commissions)
            metrics.total_fees = total_fees
        except Exception:
            logger.warning("Could not extract fees from engine cache", exc_info=True)

        return metrics


def create_screening_pipeline(
    dsl: StrategyDSL,
    use_mock: bool = True,
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
        effective_symbols = symbols or ["BTCUSDT"]
        # Convert DSL to dict for pickling across process boundaries
        dsl_dict = dsl.to_dict() if hasattr(dsl, "to_dict") else _dsl_to_dict(dsl)
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

    Args:
        dsl: Parsed StrategyDSL object.

    Returns:
        Dictionary representation suitable for validate_strategy_dict().
    """
    import dataclasses

    def _to_dict(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        elif isinstance(obj, list):
            return [_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items()}
        return obj

    return _to_dict(dsl)
