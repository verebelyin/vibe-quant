"""Validation runner for full-fidelity backtesting.

Loads strategy from SQLite, compiles to NautilusTrader Strategy,
runs backtest with realistic execution simulation, and stores results.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime as dt, timedelta
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
        detail_timeframe: str | None = None,
    ) -> ValidationResult:
        """Run validation backtest for a given run_id.

        Args:
            run_id: Backtest run ID from database.
            latency_preset: Override latency preset from database.
            detail_timeframe: Sub-bar timeframe for realistic fill simulation
                (e.g., '5s'). When provided and data exists in catalog,
                loads detail bars alongside strategy bars so the matching
                engine can fill orders at sub-bar resolution. This enables
                LatencyModel for 1m strategies (normally skipped because
                bar data has no sub-bar timestamps).

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
        dsl = self._validate_dsl(dsl_config, strategy_name=strategy_name)

        # Resolve detail timeframe: auto-detect 5s for 1m strategies
        effective_detail = self._resolve_detail_timeframe(
            run_config, dsl.timeframe, detail_timeframe
        )

        # Determine latency preset — re-enable when detail data provides
        # sub-bar resolution for the matching engine
        effective_latency = self._resolve_latency(run_config, latency_preset)

        # Configure venue (timeframe-aware: skips latency for sub-5m bars
        # unless detail data provides sub-bar resolution)
        venue_config = self._create_venue_config(
            run_config,
            effective_latency,
            timeframe=dsl.timeframe,
            has_detail_data=effective_detail is not None,
        )

        # Update run status to running
        self._state.update_backtest_run_status(run_id, "running")

        logger.info(
            "Run %d setup: strategy=%s (id=%d) timeframe=%s latency=%s detail=%s "
            "leverage=%sx balance=%s USDT",
            run_id,
            strategy_name,
            strategy_id,
            dsl.timeframe,
            effective_latency or "none",
            effective_detail or "none",
            venue_config.default_leverage,
            venue_config.starting_balance_usdt,
        )

        try:
            # Create event writer
            with EventWriter(run_id=str(run_id), base_path=self._logs_path) as writer:
                self._write_start_event(
                    writer,
                    run_id,
                    strategy_name,
                    venue_config,
                    extra={
                        "strategy_id": strategy_id,
                        "timeframe": dsl.timeframe,
                        "symbols": self._parse_symbols(run_config),
                        "start_date": run_config.get("start_date"),
                        "end_date": run_config.get("end_date"),
                        "detail_timeframe": effective_detail,
                        "leverage": float(venue_config.default_leverage),
                    },
                )

                result = self._run_backtest(
                    run_id=run_id,
                    strategy_name=strategy_name,
                    dsl=dsl,
                    venue_config=venue_config,
                    run_config=run_config,
                    writer=writer,
                    detail_timeframe=effective_detail,
                )

                self._write_completion_event(writer, run_id, strategy_name, result)

            result.execution_time_seconds = time.monotonic() - start_time
            self._store_results(run_id, result)

            logger.info(
                "Run %d result: trades=%d (%dW/%dL) return=%.2f%% sharpe=%.2f "
                "maxDD=%.2f%% pf=%.2f win=%.1f%% slippage=%.2f in %.1fs",
                run_id,
                result.total_trades,
                result.winning_trades,
                result.losing_trades,
                result.total_return * 100,
                result.sharpe_ratio,
                result.max_drawdown * 100,
                result.profit_factor,
                result.win_rate * 100,
                result.total_slippage,
                result.execution_time_seconds,
            )

            if result.total_trades == 0:
                error_msg = "Validation produced 0 trades — likely missing/empty data"
                logger.error("Run %d: %s", run_id, error_msg)
                self._state.update_backtest_run_status(run_id, "failed", error_message=error_msg)
            else:
                self._state.update_backtest_run_status(run_id, "completed")
            return result
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            try:
                self._state.update_backtest_run_status(run_id, "failed", error_message=error_msg)
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
        detail_timeframe: str | None = None,
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
            detail_timeframe: Sub-bar timeframe for fill resolution (e.g., '5s').

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
        dsl = self._validate_dsl(dsl_config, strategy_name=strategy_name)

        effective_detail = self._resolve_detail_timeframe(
            run_config, dsl.timeframe, detail_timeframe
        )
        effective_latency = self._resolve_latency(run_config, latency_preset)
        venue_config = self._create_venue_config(
            run_config,
            effective_latency,
            timeframe=dsl.timeframe,
            has_detail_data=effective_detail is not None,
        )

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
                        detail_timeframe=effective_detail,
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

            if aggregate.total_trades == 0:
                error_msg = "Walk-forward produced 0 trades — likely missing/empty data"
                logger.error("Run %d: %s", run_id, error_msg)
                self._state.update_backtest_run_status(run_id, "failed", error_message=error_msg)
            else:
                self._state.update_backtest_run_status(run_id, "completed")
            return window_results
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            try:
                self._state.update_backtest_run_status(run_id, "failed", error_message=error_msg)
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
            weighted_sum: float = sum(
                float(getattr(r, attr)) * r.total_trades for r in window_results
            )
            return weighted_sum / total_trades

        # avg_win/avg_loss must weight by winning/losing trade counts, not
        # total trades — a window with many losses but a big avg_win would
        # otherwise pull the aggregate avg_win toward itself.
        def _count_weighted_avg(attr: str, count_attr: str) -> float:
            counts = [getattr(r, count_attr) for r in window_results]
            total = sum(counts)
            if total == 0:
                return 0.0
            weighted_sum: float = sum(
                float(getattr(r, attr)) * c for r, c in zip(window_results, counts, strict=True)
            )
            return weighted_sum / float(total)

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
            max_drawdown_duration_days=max(r.max_drawdown_duration_days for r in window_results),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_trade_duration_hours=_trade_weighted_avg("avg_trade_duration_hours"),
            max_consecutive_wins=max(r.max_consecutive_wins for r in window_results),
            max_consecutive_losses=max(r.max_consecutive_losses for r in window_results),
            largest_win=max(r.largest_win for r in window_results),
            largest_loss=min(r.largest_loss for r in window_results),
            avg_win=_count_weighted_avg("avg_win", "winning_trades"),
            avg_loss=_count_weighted_avg("avg_loss", "losing_trades"),
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

    def _validate_dsl(
        self, dsl_config: dict[str, object], strategy_name: str = "strategy"
    ) -> StrategyDSL:
        """Validate DSL configuration, translating frontend format if needed.

        Args:
            dsl_config: DSL config dict from database.
            strategy_name: Strategy name used when translating frontend format.

        Returns:
            Validated StrategyDSL.

        Raises:
            ValidationRunnerError: If validation fails.
        """
        from vibe_quant.dsl.translator import translate_dsl_config

        try:
            translated = translate_dsl_config(dsl_config, strategy_name=strategy_name)
            return validate_strategy_dict(translated)
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

        return LatencyPreset.CLOUD  # Default

    # Timeframes where LatencyModel causes artificial next-bar delay
    # (bar data has no sub-bar timestamps, so any latency = full bar delay)
    _SUB_BAR_TIMEFRAMES = frozenset({"1s", "1m", "3m", "5m"})

    def _create_venue_config(
        self,
        run_config: dict[str, object],
        latency_preset: LatencyPreset | str | None,
        timeframe: str = "4h",
        *,
        has_detail_data: bool = False,
    ) -> VenueConfig:
        """Create venue configuration for validation.

        For sub-5m timeframes WITHOUT detail data, latency is skipped because
        NT's LatencyModel defers orders to the next bar (60s on 1m data)
        regardless of actual latency value. Slippage probability compensates.

        When detail data IS available (e.g., 5s bars alongside 1m bars),
        LatencyModel is re-enabled because the matching engine can process
        orders at sub-bar resolution (next 5s bar instead of next 1m bar).

        Args:
            run_config: Run configuration.
            latency_preset: Latency preset to use.
            timeframe: Strategy primary timeframe.
            has_detail_data: Whether sub-bar detail data is loaded.

        Returns:
            Configured VenueConfig.
        """
        from decimal import Decimal

        from vibe_quant.validation.venue import DEFAULT_STARTING_BALANCE_USDT

        # UI/API launch values live in the parameters JSON, not as top-level
        # run columns (initial_balance was silently ignored before this).
        raw_params = run_config.get("parameters")
        launch_params: dict[str, object] = (
            raw_params if isinstance(raw_params, dict) else {}
        )

        balance = launch_params.get(
            "initial_balance",
            run_config.get("starting_balance", DEFAULT_STARTING_BALANCE_USDT),
        )
        if isinstance(balance, bool) or not isinstance(balance, (int, float)) or balance <= 0:
            balance = DEFAULT_STARTING_BALANCE_USDT

        leverage_raw = launch_params.get("leverage")
        default_leverage = Decimal("10")
        if (
            not isinstance(leverage_raw, bool)
            and isinstance(leverage_raw, (int, float))
            and leverage_raw > 0
        ):
            default_leverage = Decimal(str(leverage_raw))

        # Skip latency for sub-5m timeframes ONLY when no detail data
        # provides sub-bar resolution for the matching engine
        if timeframe in self._SUB_BAR_TIMEFRAMES and not has_detail_data:
            if latency_preset is not None:
                logger.warning(
                    "latency preset %r dropped for %s: no sub-bar detail data "
                    "(pass detail_timeframe e.g. '5s' to enable)",
                    latency_preset,
                    timeframe,
                )
            return create_venue_config_for_validation(
                starting_balance_usdt=int(balance),
                default_leverage=default_leverage,
                latency_preset=None,
            )

        return create_venue_config_for_validation(
            starting_balance_usdt=int(balance),
            default_leverage=default_leverage,
            latency_preset=latency_preset or LatencyPreset.CLOUD,
        )

    def _run_backtest(
        self,
        run_id: int,
        strategy_name: str,
        dsl: StrategyDSL,
        venue_config: VenueConfig,
        run_config: dict[str, object],
        writer: EventWriter,
        detail_timeframe: str | None = None,
    ) -> ValidationResult:
        """Run NautilusTrader backtest with full-fidelity execution.

        Compiles the strategy DSL to a NautilusTrader Strategy, loads market
        data from the ParquetDataCatalog, configures the venue with latency
        and slippage models, and runs a BacktestNode.

        When detail_timeframe is provided (e.g., '5s'), loads sub-bar data
        alongside strategy bars. NT processes all data chronologically, so
        the matching engine can fill orders at the next detail bar instead
        of the next strategy bar. This enables realistic latency simulation
        on 1m strategies (e.g., 60ms latency fills at next 5s bar = ~5s
        delay instead of 60s).

        Args:
            run_id: Run ID.
            strategy_name: Strategy name.
            dsl: Validated strategy DSL.
            venue_config: Venue configuration.
            run_config: Run configuration from database.
            writer: Event writer for logging.
            detail_timeframe: Sub-bar timeframe for fill resolution (e.g., '5s').

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
        from nautilus_trader.model.data import Bar

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
                f"Compiled module missing expected classes: {strategy_cls_name}, {config_cls_name}"
            )
            raise ValidationRunnerError(msg)

        # Build strategy config dict: instrument_id + any override parameters
        strategy_params = self._augment_strategy_params_for_validation(
            self._build_strategy_params(run_config),
            timeframe=dsl.timeframe,
            has_detail_data=detail_timeframe is not None,
        )

        # NT 1.226+ rejects unknown config fields (fast-fail decoding), so
        # forward only params the generated StrategyConfig actually declares.
        # Run-level knobs like initial_balance/leverage live on the venue, not
        # the strategy config.
        config_fields: tuple[str, ...] = getattr(
            getattr(module, config_cls_name), "__struct_fields__", ()
        )
        if config_fields:
            dropped = sorted(k for k in strategy_params if k not in config_fields)
            if dropped:
                logger.info(
                    "Run %d: dropping non-strategy-config params: %s", run_id, dropped
                )
            strategy_params = {
                k: v for k, v in strategy_params.items() if k in config_fields
            }
        logger.info(
            "Run %d strategy params: %s",
            run_id,
            strategy_params if strategy_params else "(compiled defaults)",
        )

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
                # NT 1.226+: pass data_cls as the CLASS, not the import
                # string — BacktestDataConfig.query compares `data_cls is Bar`
                # so a string silently disables bar-type narrowing and loads
                # every bar timeframe in the catalog (~300x the needed data).
                data_configs.append(
                    BacktestDataConfig(
                        catalog_path=str(catalog_path.resolve()),
                        data_cls=Bar,
                        bar_types=[f"{instrument_id}-{step}-{agg.name}-LAST-EXTERNAL"],
                        start_time=start_date,
                        end_time=end_date,
                    )
                )

            # Add detail (sub-bar) data for fill resolution if requested
            if detail_timeframe and detail_timeframe not in all_timeframes:
                if detail_timeframe in INTERVAL_TO_AGGREGATION:
                    detail_step, detail_agg = INTERVAL_TO_AGGREGATION[detail_timeframe]
                    data_configs.append(
                        BacktestDataConfig(
                            catalog_path=str(catalog_path.resolve()),
                            data_cls=Bar,
                            bar_types=[
                                f"{instrument_id}-{detail_step}-{detail_agg.name}-LAST-EXTERNAL"
                            ],
                            start_time=start_date,
                            end_time=end_date,
                        )
                    )
                    logger.info(
                        "Loading %s detail bars for sub-bar fill resolution (run %d)",
                        detail_timeframe,
                        run_id,
                    )
                else:
                    logger.warning(
                        "Unknown detail timeframe %s, skipping sub-bar data",
                        detail_timeframe,
                    )

        if not data_configs:
            msg = "No valid data configurations could be built"
            raise ValidationRunnerError(msg)

        # Convert our VenueConfig to NautilusTrader BacktestVenueConfig
        bt_venue_config = create_backtest_venue_config(venue_config)

        # Create engine config. NT engine verbosity is tunable per run via
        # VIBE_QUANT_NT_LOG_LEVEL (TRACE/DEBUG/INFO/WARNING/ERROR); default
        # INFO matches NT's own default.
        import os

        from nautilus_trader.config import LoggingConfig

        engine_config = BacktestEngineConfig(
            strategies=strategy_configs,
            run_analysis=True,
            logging=LoggingConfig(
                log_level=os.environ.get("VIBE_QUANT_NT_LOG_LEVEL", "INFO")
            ),
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
            # Surface engine-build/run errors directly. Without this,
            # node.build() logs the real exception (e.g. a strategy-config
            # decode failure) and get_engine() returns None, which we could
            # only report as an opaque "engine not found" (vibe-quant-wrlea).
            raise_exception=True,
        )

        detail_info = f", detail={detail_timeframe}" if detail_timeframe else ""
        logger.info(
            "Starting NautilusTrader backtest for run %d: %d symbols, %d timeframes%s, %s to %s",
            run_id,
            len(symbols),
            len(all_timeframes),
            detail_info,
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

            logger.info(
                "Run %d engine stats: %d iterations, %d events, %d orders, "
                "%d positions over %.0f-day window",
                run_id,
                bt_result.iterations,
                bt_result.total_events,
                bt_result.total_orders,
                bt_result.total_positions,
                bt_result.elapsed_time / 86_400,
            )

            # Extract metrics and trades from the engine
            result = self._extract_results(
                run_id=run_id,
                strategy_name=strategy_name,
                bt_result=bt_result,
                engine=engine,
                venue_config=venue_config,
                primary_timeframe=dsl.timeframe,
                run_start_date=start_date,
                run_end_date=end_date,
            )

            # Log trade events
            for trade in result.trades:
                self._write_trade_events(writer, run_id, strategy_name, trade)

            return result
        finally:
            # Reset engines before dispose to avoid
            # InvalidStateTrigger('RUNNING -> DISPOSE')
            import contextlib

            from vibe_quant.nt_compat import retain_log_guard

            for eng in node.get_engines():
                retain_log_guard(eng)
                with contextlib.suppress(Exception):
                    eng.reset()
            node.dispose()  # type: ignore[no-untyped-call]

            # NT writes corrupt epoch-timestamp instrument parquet on dispose()
            from vibe_quant.data.catalog import cleanup_epoch_parquet

            cleanup_epoch_parquet(catalog_path)

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
            MaxDrawdown,
            ProfitFactor,
            SharpeRatio,
            SortinoRatio,
            WinRate,
        )

        # MaxDrawdown gained calculate_from_realized_pnls in NT 1.223+
        # (was excluded on 1.222).
        stats = [
            SharpeRatio(),
            SortinoRatio(),
            WinRate(),
            ProfitFactor(),
            Expectancy(),
            LongRatio(),
            AvgWinner(),
            AvgLoser(),
            MaxDrawdown(),
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

    def _augment_strategy_params_for_validation(
        self,
        params: dict[str, object],
        *,
        timeframe: str,
        has_detail_data: bool = False,
    ) -> dict[str, object]:
        """Inject validation-only runtime degradation knobs when appropriate.

        When detail data provides sub-bar resolution, LatencyModel handles
        degradation so execution_delay_probability is not needed.
        """
        augmented = dict(params)
        if timeframe in self._SUB_BAR_TIMEFRAMES and not has_detail_data:
            augmented.setdefault("execution_delay_probability", 0.3)
        return augmented

    # Default detail timeframe for sub-5m strategies when detail data exists
    _DEFAULT_DETAIL_TIMEFRAME = "5s"

    # Default fill-resolution detail for supra-5m strategies (intrabar SL/TP)
    _DEFAULT_COARSE_DETAIL_TIMEFRAME = "1m"

    def _resolve_detail_timeframe(
        self,
        run_config: dict[str, object],
        strategy_timeframe: str,
        override: str | None,
    ) -> str | None:
        """Resolve the effective detail timeframe for sub-bar fill resolution.

        Auto-detects available detail data for sub-5m strategies. Returns
        None if no detail data is needed or available.

        Args:
            run_config: Run configuration from database.
            strategy_timeframe: Primary strategy timeframe.
            override: Explicit detail timeframe override.

        Returns:
            Detail timeframe string (e.g., '5s') or None.
        """
        # Explicit override always wins
        if override is not None:
            return override

        # Check run config for detail_timeframe parameter
        params = run_config.get("parameters")
        if isinstance(params, dict) and params.get("detail_timeframe"):
            return str(params["detail_timeframe"])

        # Auto-detect: check if default detail data covers the run window
        # for ALL symbols. Venue config (latency on/off) and strategy params
        # (execution_delay_probability) are run-wide, so partial coverage
        # would remove degradation for symbols without sub-bar data.
        from vibe_quant.data.catalog import (
            DEFAULT_CATALOG_PATH,
            INTERVAL_TO_AGGREGATION,
            CatalogManager,
        )

        symbols = self._parse_symbols(run_config)
        if not symbols:
            return None

        # Sub-5m strategies get 5s detail; coarser strategies default to 1m
        # detail so the matching engine triggers stops/TPs intrabar. The
        # latter preserves historical validation semantics: before the NT
        # 1.230 data-targeting fix, untargeted catalog loading fed the venue
        # the full 1m stream on every validation run.
        detail_tf = (
            self._DEFAULT_DETAIL_TIMEFRAME
            if strategy_timeframe in self._SUB_BAR_TIMEFRAMES
            else self._DEFAULT_COARSE_DETAIL_TIMEFRAME
        )
        if detail_tf == strategy_timeframe or detail_tf not in INTERVAL_TO_AGGREGATION:
            return None

        # Parse run date window for coverage check
        run_start = run_config.get("start_date")
        run_end = run_config.get("end_date")
        if not isinstance(run_start, str) or not isinstance(run_end, str):
            return None

        try:
            window_start = dt.fromisoformat(run_start)
            window_end = dt.fromisoformat(run_end)
        except ValueError:
            return None

        catalog_mgr = CatalogManager(DEFAULT_CATALOG_PATH)

        # Require ALL symbols to have detail data covering the run window
        for symbol in symbols:
            date_range = catalog_mgr.get_bar_date_range(symbol, detail_tf)
            if date_range is None:
                logger.info(
                    "No %s detail data for %s — skipping sub-bar resolution",
                    detail_tf,
                    symbol,
                )
                return None

            data_start, data_end = date_range
            # Compare timezone-naive to handle mixed tz/naive dates
            data_s = data_start.replace(tzinfo=None)
            data_e = data_end.replace(tzinfo=None)
            win_s = window_start.replace(tzinfo=None)
            win_e = window_end.replace(tzinfo=None)
            if data_s > win_s:
                logger.info(
                    "Detail %s data for %s starts %s, after run start %s — skipping",
                    detail_tf,
                    symbol,
                    data_start.isoformat(),
                    run_start,
                )
                return None
            if data_e < win_e:
                logger.info(
                    "Detail %s data for %s ends %s, before run end %s — skipping",
                    detail_tf,
                    symbol,
                    data_end.isoformat(),
                    run_end,
                )
                return None

        logger.info(
            "Auto-detected %s detail data covering run window for all %d symbols",
            detail_tf,
            len(symbols),
        )
        return detail_tf

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
        primary_timeframe: str | None = None,
        run_start_date: str | None = None,
        run_end_date: str | None = None,
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
            primary_timeframe: Strategy primary timeframe for market-stat
                bar-group selection.
            run_start_date / run_end_date: Backtest window for CAGR.

        Returns:
            Populated ValidationResult.
        """
        from vibe_quant.validation.extraction import extract_results
        from vibe_quant.validation.funding import FundingCalculator

        return extract_results(
            run_id,
            strategy_name,
            bt_result,
            engine,
            venue_config,
            primary_timeframe=primary_timeframe,
            funding_calculator=FundingCalculator(),
            run_start_date=run_start_date,
            run_end_date=run_end_date,
        )

    def _write_start_event(
        self,
        writer: EventWriter,
        run_id: int,
        strategy_name: str,
        venue_config: VenueConfig,
        extra: dict[str, object] | None = None,
    ) -> None:
        """Write backtest start event.

        Args:
            extra: Additional run-setup context (symbols, dates, timeframe,
                leverage, …) merged into the event payload for analysis.
        """
        data: dict[str, object] = {
            "event": "BACKTEST_START",
            "venue": venue_config.name,
            "latency_preset": str(venue_config.latency_preset)
            if venue_config.latency_preset
            else None,
            "starting_balance": venue_config.starting_balance_usdt,
        }
        if extra:
            data.update(extra)
        event = create_event(
            event_type=EventType.LIFECYCLE,
            run_id=str(run_id),
            strategy_name=strategy_name,
            data=data,
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
