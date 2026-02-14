"""Validation runner for full-fidelity backtesting.

Loads strategy from SQLite, compiles to NautilusTrader Strategy,
runs backtest with realistic execution simulation, and stores results.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.logging.events import EventType, create_event
from vibe_quant.logging.writer import EventWriter
from vibe_quant.validation.latency import LatencyPreset
from vibe_quant.validation.results import TradeRecord, ValidationResult
from vibe_quant.validation.venue import (
    VenueConfig,
    create_backtest_venue_config,
    create_venue_config_for_validation,
)

if TYPE_CHECKING:
    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.backtest.results import BacktestResult

    from vibe_quant.dsl.schema import StrategyDSL

logger = logging.getLogger(__name__)


class ValidationRunnerError(Exception):
    """Error during validation run."""

    pass


@dataclass(frozen=True)
class WalkForwardWindow:
    """Single walk-forward train/test window."""

    train_start: str
    train_end: str
    test_start: str
    test_end: str


class ValidationRunner:
    """Runner for validation backtests with full-fidelity execution.

    Loads strategy from database, compiles to NautilusTrader Strategy,
    configures venue with latency/slippage, runs backtest, stores results.

    Example:
        runner = ValidationRunner(db_path=Path("data/state/vibe_quant.db"))
        result = runner.run(run_id=42)
        print(f"Sharpe: {result.sharpe_ratio}")
    """

    def __init__(
        self,
        db_path: Path | None = None,
        logs_path: Path | str = "logs/events",
    ) -> None:
        """Initialize ValidationRunner.

        Args:
            db_path: Path to state database. Uses default if None.
            logs_path: Path for event log files.
        """
        self._state = StateManager(db_path)
        self._logs_path = Path(logs_path)
        self._compiler = StrategyCompiler()

    def close(self) -> None:
        """Close database connection."""
        self._state.close()

    def run(
        self,
        run_id: int,
        latency_preset: LatencyPreset | str | None = None,
    ) -> ValidationResult:
        """Run validation backtest for a given run_id.

        Args:
            run_id: Backtest run ID from database.
            latency_preset: Override latency preset from database.

        Returns:
            ValidationResult with metrics and trades.

        Raises:
            ValidationRunnerError: If run fails.
        """
        start_time = time.monotonic()

        # Load run config from database
        run_config = self._load_run_config(run_id)
        strategy_id_raw = run_config["strategy_id"]
        if not isinstance(strategy_id_raw, int):
            strategy_id_raw = int(str(strategy_id_raw))
        strategy_id: int = strategy_id_raw

        # Load strategy DSL
        strategy_data = self._state.get_strategy(strategy_id)
        if strategy_data is None:
            msg = f"Strategy {strategy_id} not found"
            raise ValidationRunnerError(msg)

        strategy_name = str(strategy_data["name"])
        dsl_config = strategy_data["dsl_config"]

        # Validate strategy DSL (compilation happens in _run_backtest)
        dsl = self._validate_dsl(dsl_config)

        # Determine latency preset
        effective_latency = self._resolve_latency(run_config, latency_preset)

        # Configure venue
        venue_config = self._create_venue_config(run_config, effective_latency)

        # Update run status to running
        self._state.update_backtest_run_status(run_id, "running")

        try:
            # Create event writer
            with EventWriter(run_id=str(run_id), base_path=self._logs_path) as writer:
                self._write_start_event(writer, run_id, strategy_name, venue_config)

                result = self._run_backtest(
                    run_id=run_id,
                    strategy_name=strategy_name,
                    dsl=dsl,
                    venue_config=venue_config,
                    run_config=run_config,
                    writer=writer,
                )

                self._write_completion_event(writer, run_id, strategy_name, result)

            result.execution_time_seconds = time.monotonic() - start_time
            self._store_results(run_id, result)
            self._state.update_backtest_run_status(run_id, "completed")
            return result
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            try:
                self._state.update_backtest_run_status(
                    run_id, "failed", error_message=error_msg
                )
            except Exception:
                logger.exception("Failed to update run %d status to failed", run_id)
            raise ValidationRunnerError(error_msg) from exc

    def run_walk_forward(
        self,
        run_id: int,
        *,
        train_days: int = 90,
        test_days: int = 30,
        step_days: int | None = None,
        latency_preset: LatencyPreset | str | None = None,
    ) -> list[ValidationResult]:
        """Run walk-forward validation over multiple rolling windows.

        Windows are constructed over the run's configured [start_date, end_date]
        range using a rolling train window followed by an out-of-sample test
        window. Each test window is backtested independently.

        Args:
            run_id: Backtest run ID from database.
            train_days: Training window size in days.
            test_days: Out-of-sample test window size in days.
            step_days: Step size between windows. Defaults to test_days.
            latency_preset: Optional latency override.

        Returns:
            List of ValidationResult objects, one per test window.

        Raises:
            ValidationRunnerError: If window generation or any window run fails.
        """
        if train_days <= 0 or test_days <= 0:
            msg = "train_days and test_days must be positive"
            raise ValidationRunnerError(msg)

        if step_days is None:
            step_days = test_days
        if step_days <= 0:
            msg = "step_days must be positive"
            raise ValidationRunnerError(msg)

        start_time = time.monotonic()
        run_config = self._load_run_config(run_id)
        strategy_id_raw = run_config["strategy_id"]
        if not isinstance(strategy_id_raw, int):
            strategy_id_raw = int(str(strategy_id_raw))
        strategy_id: int = strategy_id_raw

        strategy_data = self._state.get_strategy(strategy_id)
        if strategy_data is None:
            msg = f"Strategy {strategy_id} not found"
            raise ValidationRunnerError(msg)

        strategy_name = str(strategy_data["name"])
        dsl_config = strategy_data["dsl_config"]
        dsl = self._validate_dsl(dsl_config)

        effective_latency = self._resolve_latency(run_config, latency_preset)
        venue_config = self._create_venue_config(run_config, effective_latency)

        range_start = self._parse_run_date(run_config.get("start_date"), "start_date")
        range_end = self._parse_run_date(run_config.get("end_date"), "end_date")
        windows = self._build_walk_forward_windows(
            range_start=range_start,
            range_end=range_end,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
        )
        if not windows:
            msg = (
                "No walk-forward windows fit the configured date range. "
                f"start={range_start.isoformat()}, end={range_end.isoformat()}, "
                f"train_days={train_days}, test_days={test_days}, step_days={step_days}"
            )
            raise ValidationRunnerError(msg)

        self._state.update_backtest_run_status(run_id, "running")

        try:
            window_results: list[ValidationResult] = []
            with EventWriter(run_id=str(run_id), base_path=self._logs_path) as writer:
                self._write_start_event(writer, run_id, strategy_name, venue_config)
                writer.write(
                    create_event(
                        event_type=EventType.LIFECYCLE,
                        run_id=str(run_id),
                        strategy_name=strategy_name,
                        data={
                            "event": "WALK_FORWARD_START",
                            "window_count": len(windows),
                            "train_days": train_days,
                            "test_days": test_days,
                            "step_days": step_days,
                        },
                    )
                )

                for index, window in enumerate(windows, start=1):
                    window_run_config = dict(run_config)
                    window_run_config["start_date"] = window.test_start
                    window_run_config["end_date"] = window.test_end

                    window_result = self._run_backtest(
                        run_id=run_id,
                        strategy_name=strategy_name,
                        dsl=dsl,
                        venue_config=venue_config,
                        run_config=window_run_config,
                        writer=writer,
                    )
                    window_results.append(window_result)
                    writer.write(
                        create_event(
                            event_type=EventType.LIFECYCLE,
                            run_id=str(run_id),
                            strategy_name=strategy_name,
                            data={
                                "event": "WALK_FORWARD_WINDOW_COMPLETE",
                                "window_index": index,
                                "window_count": len(windows),
                                "train_start": window.train_start,
                                "train_end": window.train_end,
                                "test_start": window.test_start,
                                "test_end": window.test_end,
                                "total_return": window_result.total_return,
                                "sharpe_ratio": window_result.sharpe_ratio,
                                "max_drawdown": window_result.max_drawdown,
                                "total_trades": window_result.total_trades,
                            },
                        )
                    )

                aggregate = self._aggregate_walk_forward_results(
                    run_id=run_id,
                    strategy_name=strategy_name,
                    window_results=window_results,
                )
                self._write_completion_event(writer, run_id, strategy_name, aggregate)

            aggregate.execution_time_seconds = time.monotonic() - start_time
            self._store_results(run_id, aggregate)
            self._state.update_backtest_run_status(run_id, "completed")
            return window_results
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            try:
                self._state.update_backtest_run_status(
                    run_id, "failed", error_message=error_msg
                )
            except Exception:
                logger.exception("Failed to update run %d status to failed", run_id)
            raise ValidationRunnerError(error_msg) from exc

    @staticmethod
    def _parse_run_date(value: object, field_name: str) -> date:
        """Parse run date fields from DB config."""
        if not isinstance(value, str):
            msg = f"Run config missing valid {field_name}"
            raise ValidationRunnerError(msg)
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            msg = f"Invalid {field_name}: {value}"
            raise ValidationRunnerError(msg) from exc

    @staticmethod
    def _build_walk_forward_windows(
        *,
        range_start: date,
        range_end: date,
        train_days: int,
        test_days: int,
        step_days: int,
    ) -> list[WalkForwardWindow]:
        """Build rolling walk-forward windows over a date range."""
        if range_end <= range_start:
            return []

        step = timedelta(days=step_days)
        train_delta = timedelta(days=train_days)
        test_delta = timedelta(days=test_days)

        cursor = range_start
        windows: list[WalkForwardWindow] = []
        while True:
            train_start = cursor
            train_end = train_start + train_delta
            test_start = train_end
            test_end = test_start + test_delta

            if test_end > range_end:
                break

            windows.append(
                WalkForwardWindow(
                    train_start=train_start.isoformat(),
                    train_end=train_end.isoformat(),
                    test_start=test_start.isoformat(),
                    test_end=test_end.isoformat(),
                )
            )
            cursor += step

        return windows

    @staticmethod
    def _aggregate_walk_forward_results(
        *,
        run_id: int,
        strategy_name: str,
        window_results: list[ValidationResult],
    ) -> ValidationResult:
        """Aggregate per-window validation results into one persisted result.

        Uses compounded returns (not averaged), trade-weighted averages for
        ratios, and additive sums for counts/costs.
        """
        if not window_results:
            msg = "Cannot aggregate empty walk-forward result set"
            raise ValidationRunnerError(msg)

        total_trades = sum(r.total_trades for r in window_results)
        winning_trades = sum(r.winning_trades for r in window_results)
        losing_trades = sum(r.losing_trades for r in window_results)

        # Compound returns: prod(1 + r_i) - 1
        compounded = 1.0
        for r in window_results:
            compounded *= 1.0 + r.total_return
        compounded_return = compounded - 1.0

        # Trade-weighted average for ratios (avoids bias from low-trade windows)
        def _trade_weighted_avg(attr: str) -> float:
            if total_trades == 0:
                return 0.0
            return sum(
                getattr(r, attr) * r.total_trades for r in window_results
            ) / total_trades

        # Win rate from actual counts, not averaged percentages
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        aggregate = ValidationResult(
            run_id=run_id,
            strategy_name=strategy_name,
            total_return=compounded_return,
            sharpe_ratio=_trade_weighted_avg("sharpe_ratio"),
            sortino_ratio=_trade_weighted_avg("sortino_ratio"),
            max_drawdown=max(r.max_drawdown for r in window_results),
            profit_factor=_trade_weighted_avg("profit_factor"),
            win_rate=win_rate,
            total_trades=total_trades,
            total_fees=sum(r.total_fees for r in window_results),
            total_funding=sum(r.total_funding for r in window_results),
            total_slippage=sum(r.total_slippage for r in window_results),
            trades=[trade for result in window_results for trade in result.trades],
            cagr=_trade_weighted_avg("cagr"),
            calmar_ratio=_trade_weighted_avg("calmar_ratio"),
            volatility_annual=_trade_weighted_avg("volatility_annual"),
            max_drawdown_duration_days=max(
                r.max_drawdown_duration_days for r in window_results
            ),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_trade_duration_hours=_trade_weighted_avg("avg_trade_duration_hours"),
            max_consecutive_wins=max(r.max_consecutive_wins for r in window_results),
            max_consecutive_losses=max(r.max_consecutive_losses for r in window_results),
            largest_win=max(r.largest_win for r in window_results),
            largest_loss=min(r.largest_loss for r in window_results),
            avg_win=_trade_weighted_avg("avg_win"),
            avg_loss=_trade_weighted_avg("avg_loss"),
            starting_balance=window_results[0].starting_balance,
        )
        return aggregate

    def _load_run_config(self, run_id: int) -> dict[str, object]:
        """Load backtest run configuration from database.

        Args:
            run_id: Run ID to load.

        Returns:
            Run configuration dict.

        Raises:
            ValidationRunnerError: If run not found.
        """
        run_config = self._state.get_backtest_run(run_id)
        if run_config is None:
            msg = f"Backtest run {run_id} not found"
            raise ValidationRunnerError(msg)

        # Validate run mode
        if run_config.get("run_mode") != "validation":
            msg = f"Run {run_id} is not a validation run (mode: {run_config.get('run_mode')})"
            raise ValidationRunnerError(msg)

        return run_config

    def _validate_dsl(self, dsl_config: dict[str, object]) -> StrategyDSL:
        """Validate DSL configuration.

        Args:
            dsl_config: DSL config dict from database.

        Returns:
            Validated StrategyDSL.

        Raises:
            ValidationRunnerError: If validation fails.
        """
        try:
            return validate_strategy_dict(dsl_config)
        except Exception as e:
            msg = f"Strategy DSL validation failed: {e}"
            raise ValidationRunnerError(msg) from e

    def _resolve_latency(
        self,
        run_config: dict[str, object],
        override: LatencyPreset | str | None,
    ) -> LatencyPreset | str | None:
        """Resolve effective latency preset.

        Args:
            run_config: Run configuration from database.
            override: Override latency preset.

        Returns:
            Effective latency preset.
        """
        if override is not None:
            return override

        db_latency = run_config.get("latency_preset")
        if db_latency:
            return str(db_latency)

        return LatencyPreset.RETAIL  # Default

    def _create_venue_config(
        self,
        run_config: dict[str, object],
        latency_preset: LatencyPreset | str | None,
    ) -> VenueConfig:
        """Create venue configuration for validation.

        Args:
            run_config: Run configuration.
            latency_preset: Latency preset to use.

        Returns:
            Configured VenueConfig.
        """
        balance = run_config.get("starting_balance", 100_000)
        if isinstance(balance, bool) or not isinstance(balance, (int, float)) or balance <= 0:
            balance = 100_000
        return create_venue_config_for_validation(
            starting_balance_usdt=int(balance),
            latency_preset=latency_preset or LatencyPreset.RETAIL,
        )

    def _run_backtest(
        self,
        run_id: int,
        strategy_name: str,
        dsl: StrategyDSL,
        venue_config: VenueConfig,
        run_config: dict[str, object],
        writer: EventWriter,
    ) -> ValidationResult:
        """Run NautilusTrader backtest with full-fidelity execution.

        Compiles the strategy DSL to a NautilusTrader Strategy, loads market
        data from the ParquetDataCatalog, configures the venue with latency
        and slippage models, and runs a BacktestNode.

        Args:
            run_id: Run ID.
            strategy_name: Strategy name.
            dsl: Validated strategy DSL.
            venue_config: Venue configuration.
            run_config: Run configuration from database.
            writer: Event writer for logging.

        Returns:
            ValidationResult with real metrics and trades.

        Raises:
            ValidationRunnerError: If backtest setup or execution fails.
        """
        from nautilus_trader.backtest.node import BacktestNode
        from nautilus_trader.config import (
            BacktestDataConfig,
            BacktestEngineConfig,
            BacktestRunConfig,
            ImportableStrategyConfig,
        )

        from vibe_quant.data.catalog import (
            DEFAULT_CATALOG_PATH,
            INSTRUMENT_CONFIGS,
            INTERVAL_TO_AGGREGATION,
            CatalogManager,
            create_instrument,
        )

        # Parse symbols from run config
        symbols = self._parse_symbols(run_config)

        # Parse date range
        start_date = str(run_config.get("start_date", "2024-01-01"))
        end_date = str(run_config.get("end_date", "2024-12-31"))

        # Collect all timeframes needed by the strategy
        all_timeframes = {dsl.timeframe}
        all_timeframes.update(dsl.additional_timeframes)
        for ind_config in dsl.indicators.values():
            if ind_config.timeframe:
                all_timeframes.add(ind_config.timeframe)

        # Ensure instruments exist in catalog
        catalog_path = DEFAULT_CATALOG_PATH
        catalog_mgr = CatalogManager(catalog_path)
        for symbol in symbols:
            if symbol in INSTRUMENT_CONFIGS:
                instrument = create_instrument(symbol)
                catalog_mgr.write_instrument(instrument)

        # Compile strategy to an importable module (registers in sys.modules)
        module = self._compiler.compile_to_module(dsl)
        class_name = "".join(word.capitalize() for word in dsl.name.split("_"))
        module_path = f"vibe_quant.dsl.generated.{dsl.name}"

        # Verify generated classes exist in the module
        strategy_cls_name = f"{class_name}Strategy"
        config_cls_name = f"{class_name}Config"
        if not hasattr(module, strategy_cls_name) or not hasattr(module, config_cls_name):
            msg = (
                f"Compiled module missing expected classes: "
                f"{strategy_cls_name}, {config_cls_name}"
            )
            raise ValidationRunnerError(msg)

        # Build strategy config dict: instrument_id + any override parameters
        strategy_params = self._build_strategy_params(run_config)

        # Build strategy configs (one per symbol)
        strategy_configs: list[ImportableStrategyConfig] = []
        for symbol in symbols:
            instrument_id = f"{symbol}-PERP.BINANCE"
            config_dict = {"instrument_id": instrument_id, **strategy_params}
            strategy_configs.append(
                ImportableStrategyConfig(
                    strategy_path=f"{module_path}:{strategy_cls_name}",
                    config_path=f"{module_path}:{config_cls_name}",
                    config=config_dict,
                )
            )

        # Build data configs (one per symbol per timeframe)
        data_configs: list[BacktestDataConfig] = []
        for symbol in symbols:
            instrument_id = f"{symbol}-PERP.BINANCE"
            for tf in sorted(all_timeframes):
                if tf not in INTERVAL_TO_AGGREGATION:
                    logger.warning("Unknown timeframe %s, skipping", tf)
                    continue
                step, agg = INTERVAL_TO_AGGREGATION[tf]
                data_configs.append(
                    BacktestDataConfig(
                        catalog_path=str(catalog_path.resolve()),
                        data_cls="nautilus_trader.model.data:Bar",
                        instrument_id=instrument_id,
                        bar_spec=f"{step}-{agg.name}-LAST",
                        start_time=start_date,
                        end_time=end_date,
                    )
                )

        if not data_configs:
            msg = "No valid data configurations could be built"
            raise ValidationRunnerError(msg)

        # Convert our VenueConfig to NautilusTrader BacktestVenueConfig
        bt_venue_config = create_backtest_venue_config(venue_config)

        # Create engine config
        engine_config = BacktestEngineConfig(
            strategies=strategy_configs,
            run_analysis=True,
        )

        # Create run config -- dispose_on_completion=False so we can
        # access engine.trader for positions report after run completes.
        bt_run_config = BacktestRunConfig(
            engine=engine_config,
            venues=[bt_venue_config],
            data=data_configs,
            start=start_date,
            end=end_date,
            dispose_on_completion=False,
        )

        logger.info(
            "Starting NautilusTrader backtest for run %d: "
            "%d symbols, %d timeframes, %s to %s",
            run_id,
            len(symbols),
            len(all_timeframes),
            start_date,
            end_date,
        )

        # Execute the backtest
        node = BacktestNode(configs=[bt_run_config])
        try:
            # Build engines, then register portfolio statistics before running
            node.build()
            self._register_statistics(node)
            node.run()

            engine = node.get_engine(bt_run_config.id)
            if engine is None:
                raise ValidationRunnerError(
                    f"Backtest engine not found for run config {bt_run_config.id}"
                )
            bt_result = engine.get_result()

            # Extract metrics and trades from the engine
            result = self._extract_results(
                run_id=run_id,
                strategy_name=strategy_name,
                bt_result=bt_result,
                engine=engine,
                venue_config=venue_config,
            )

            # Log trade events
            for trade in result.trades:
                self._write_trade_events(writer, run_id, strategy_name, trade)

            return result
        finally:
            # Reset engines before dispose to avoid
            # InvalidStateTrigger('RUNNING -> DISPOSE')
            import contextlib

            for eng in node.get_engines():
                with contextlib.suppress(Exception):
                    eng.reset()
            node.dispose()  # type: ignore[no-untyped-call]

    def _register_statistics(self, node: object) -> None:
        """Register portfolio statistics on the engine's analyzer.

        NautilusTrader's PortfolioAnalyzer starts with no registered
        statistics.  We register the standard set so that BacktestResult
        stats_pnls / stats_returns are populated.

        Args:
            node: BacktestNode (after build, before run).
        """
        from nautilus_trader.core.nautilus_pyo3 import (
            AvgLoser,
            AvgWinner,
            Expectancy,
            LongRatio,
            ProfitFactor,
            SharpeRatio,
            SortinoRatio,
            WinRate,
        )

        # MaxDrawdown excluded: lacks calculate_from_realized_pnls in NT 1.222
        stats = [
            SharpeRatio(),
            SortinoRatio(),
            WinRate(),
            ProfitFactor(),
            Expectancy(),
            LongRatio(),
            AvgWinner(),
            AvgLoser(),
        ]

        for engine in node.get_engines():  # type: ignore[attr-defined]
            analyzer = engine.kernel.portfolio.analyzer
            for stat in stats:
                analyzer.register_statistic(stat)

    def _build_strategy_params(self, run_config: dict[str, object]) -> dict[str, object]:
        """Extract strategy parameter overrides from run config.

        If the run was created to validate specific sweep parameters,
        those are stored in run_config['parameters'] and should be
        forwarded to the compiled strategy's config.

        Args:
            run_config: Run configuration dict from database.

        Returns:
            Dict of parameter overrides to merge into ImportableStrategyConfig.
        """
        params: dict[str, object] = {}
        raw_params = run_config.get("parameters")
        if not isinstance(raw_params, dict):
            return params

        # Direct parameter overrides (e.g., from validated screening results)
        for key, value in raw_params.items():
            # Skip meta-keys that aren't strategy parameters
            if key in ("sweep", "overfitting_filters"):
                continue
            params[key] = value

        return params

    def _parse_symbols(self, run_config: dict[str, object]) -> list[str]:
        """Parse symbol list from run configuration.

        Args:
            run_config: Run configuration dict from database.

        Returns:
            List of symbol strings (e.g., ['BTCUSDT', 'ETHUSDT']).
        """
        symbols_raw = run_config.get("symbols", ["BTCUSDT"])
        if isinstance(symbols_raw, str):
            return list(json.loads(symbols_raw))
        elif isinstance(symbols_raw, list):
            return [str(s) for s in symbols_raw]
        return ["BTCUSDT"]

    def _extract_results(
        self,
        run_id: int,
        strategy_name: str,
        bt_result: BacktestResult,
        engine: BacktestEngine,
        venue_config: VenueConfig,
    ) -> ValidationResult:
        """Extract ValidationResult from NautilusTrader backtest output.

        Delegates to :func:`vibe_quant.validation.extraction.extract_results`
        for the actual extraction logic.

        Args:
            run_id: Run ID.
            strategy_name: Strategy name.
            bt_result: NautilusTrader BacktestResult.
            engine: BacktestEngine after run for report generation.
            venue_config: Venue config for leverage info.

        Returns:
            Populated ValidationResult.
        """
        from vibe_quant.validation.extraction import extract_results

        return extract_results(run_id, strategy_name, bt_result, engine, venue_config)

    def _write_start_event(
        self,
        writer: EventWriter,
        run_id: int,
        strategy_name: str,
        venue_config: VenueConfig,
    ) -> None:
        """Write backtest start event."""
        event = create_event(
            event_type=EventType.LIFECYCLE,
            run_id=str(run_id),
            strategy_name=strategy_name,
            data={
                "event": "BACKTEST_START",
                "venue": venue_config.name,
                "latency_preset": str(venue_config.latency_preset)
                if venue_config.latency_preset
                else None,
                "starting_balance": venue_config.starting_balance_usdt,
            },
        )
        writer.write(event)

    def _write_completion_event(
        self,
        writer: EventWriter,
        run_id: int,
        strategy_name: str,
        result: ValidationResult,
    ) -> None:
        """Write backtest completion event."""
        event = create_event(
            event_type=EventType.LIFECYCLE,
            run_id=str(run_id),
            strategy_name=strategy_name,
            data={
                "event": "BACKTEST_COMPLETE",
                "total_return": result.total_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "total_trades": result.total_trades,
            },
        )
        writer.write(event)

    def _write_trade_events(
        self,
        writer: EventWriter,
        run_id: int,
        strategy_name: str,
        trade: TradeRecord,
    ) -> None:
        """Write events for a trade (entry and exit)."""
        # Position open event
        open_event = create_event(
            event_type=EventType.POSITION_OPEN,
            run_id=str(run_id),
            strategy_name=strategy_name,
            data={
                "symbol": trade.symbol,
                "side": trade.direction,
                "entry_price": trade.entry_price,
                "quantity": trade.quantity,
                "leverage": trade.leverage,
            },
        )
        writer.write(open_event)

        # Position close event
        if trade.exit_time:
            close_event = create_event(
                event_type=EventType.POSITION_CLOSE,
                run_id=str(run_id),
                strategy_name=strategy_name,
                data={
                    "symbol": trade.symbol,
                    "exit_price": trade.exit_price,
                    "gross_pnl": trade.gross_pnl,
                    "net_pnl": trade.net_pnl,
                    "exit_reason": trade.exit_reason,
                },
            )
            writer.write(close_event)

    def _store_results(self, run_id: int, result: ValidationResult) -> None:
        """Store validation results in database.

        Args:
            run_id: Run ID.
            result: Validation result to store.
        """
        # Save backtest results
        self._state.save_backtest_result(run_id, result.to_metrics_dict())

        # Save individual trades
        trade_dicts = [t.to_dict() for t in result.trades]
        self._state.save_trades_batch(run_id, trade_dicts)


def list_validation_runs(
    db_path: Path | None = None,
    limit: int = 20,
) -> list[dict[str, object]]:
    """List recent validation runs.

    Args:
        db_path: Path to state database.
        limit: Maximum runs to return.

    Returns:
        List of validation run dicts with status and key metrics.
    """
    state = StateManager(db_path)
    try:
        # Get validation runs
        runs = state.list_backtest_runs(status=None)

        # Filter to validation mode and limit
        validation_runs = [r for r in runs if r.get("run_mode") == "validation"][:limit]

        # Enrich with results if available
        enriched = []
        for run in validation_runs:
            run_id = run["id"]
            result = state.get_backtest_result(int(run_id))

            enriched_run = {
                "run_id": run_id,
                "strategy_id": run.get("strategy_id"),
                "status": run.get("status"),
                "latency_preset": run.get("latency_preset"),
                "created_at": run.get("created_at"),
                "completed_at": run.get("completed_at"),
            }

            if result:
                enriched_run["sharpe_ratio"] = result.get("sharpe_ratio")
                enriched_run["total_return"] = result.get("total_return")
                enriched_run["max_drawdown"] = result.get("max_drawdown")
                enriched_run["total_trades"] = result.get("total_trades")

            enriched.append(enriched_run)

        return enriched
    finally:
        state.close()


__all__ = [
    "ValidationRunner",
    "ValidationRunnerError",
    "ValidationResult",
    "TradeRecord",
    "list_validation_runs",
]
