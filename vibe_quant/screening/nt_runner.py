"""NautilusTrader backtest runner for screening mode.

Runs single-parameter-combination backtests using BacktestNode with
screening venue config (no latency, simple fill model). Designed to be
picklable for :class:`concurrent.futures.ProcessPoolExecutor`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vibe_quant.screening.types import BacktestMetrics

logger = logging.getLogger(__name__)


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
        from vibe_quant.screening.types import BacktestMetrics

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

        # Catalog path (instruments already written during data ingest/rebuild;
        # writing here from parallel workers causes parquet corruption)
        self._resolved_catalog_path = (
            Path(self._catalog_path) if self._catalog_path else DEFAULT_CATALOG_PATH
        )

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
            ProfitFactor,
            SharpeRatio,
            SortinoRatio,
            WinRate,
        )

        from vibe_quant.data.catalog import (
            INTERVAL_TO_AGGREGATION,
        )
        from vibe_quant.screening.types import BacktestMetrics
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
            # MaxDrawdown excluded: lacks calculate_from_realized_pnls in NT 1.222
            stats = [SharpeRatio(), SortinoRatio(), WinRate(), ProfitFactor()]
            for engine in node.get_engines():
                analyzer = engine.kernel.portfolio.analyzer
                for stat in stats:
                    analyzer.register_statistic(stat)

            node.run()

            engine = node.get_engine(bt_run_config.id)
            if engine is None:
                return BacktestMetrics(
                    parameters=params,
                    sharpe_ratio=float("-inf"),
                    execution_time_seconds=time.time() - start_time,
                )
            bt_result = engine.get_result()

            return self._extract_metrics(params, bt_result, engine, start_time)
        finally:
            node.dispose()  # type: ignore[no-untyped-call]

    def _extract_metrics(
        self,
        params: dict[str, float | int],
        bt_result: Any,
        engine: Any,
        start_time: float,
    ) -> BacktestMetrics:
        """Extract BacktestMetrics from NT BacktestResult."""
        from vibe_quant.screening.types import BacktestMetrics

        metrics = BacktestMetrics(
            parameters=params,
            execution_time_seconds=time.time() - start_time,
        )

        if bt_result is None:
            return metrics

        metrics.total_trades = bt_result.total_positions

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
                    total_fees += sum(abs(float(c)) for c in pos.commissions())
            metrics.total_fees = total_fees
        except Exception:
            logger.warning("Could not extract fees from engine cache", exc_info=True)

        return metrics
