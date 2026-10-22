"""Validation runner for full-fidelity backtesting.

Loads strategy from SQLite, compiles to NautilusTrader Strategy,
runs backtest with realistic execution simulation, and stores results.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from vibe_quant.balance_settings import BalanceConfigError
from vibe_quant.data import catalog as catalog_mod
from vibe_quant.data.catalog import (
    INTERVAL_TO_AGGREGATION,
    CatalogManager,
)
from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.launch_settings import PARSE_ORDER, Field, LaunchSettings, iter_fields
from vibe_quant.liquidation_settings import LiquidationConfigError
from vibe_quant.logging.events import EventType, create_event
from vibe_quant.logging.writer import EventWriter
from vibe_quant.metrics import profit_factor
from vibe_quant.sizing_settings import SizingConfigError, SizingSettings, recorded_sizing
from vibe_quant.utils import log_dir
from vibe_quant.validation.funding import FundingCalculator
from vibe_quant.validation.latency import LatencyPreset
from vibe_quant.validation.results import TradeRecord, ValidationResult
from vibe_quant.validation.venue import (
    VenueConfig,
    create_backtest_venue_config,
    create_venue_config_for_validation,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.backtest.results import BacktestResult

    from vibe_quant.dsl.schema import StrategyDSL

logger = logging.getLogger(__name__)


class ValidationRunnerError(Exception):
    """Error during validation run."""

    pass


# Detail data that gives a strategy timeframe intrabar fill resolution:
# 1m strategies look for 5s bars (optional, usually absent); every
# coarser strategy -- including 3m/5m, which used to ask for 5s and so
# never got detail -- uses 1m bars (bd vibe-quant-e70tl.13).
_SUB_MINUTE_DETAIL_TIMEFRAME = "5s"
_DEFAULT_COARSE_DETAIL_TIMEFRAME = "1m"
_TIMEFRAME_SECONDS: dict[str, int] = {  # noqa: RUF012 — read-only table
    "1s": 1,
    "5s": 5,
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "1h": 3_600,
    "4h": 14_400,
    "1d": 86_400,
}


def _launch_params(run_config: dict[str, object]) -> dict[str, object]:
    """The run's stored ``parameters`` as a plain dict (``{}`` when not a mapping)."""
    raw_params = run_config.get("parameters")
    return raw_params if isinstance(raw_params, dict) else {}


def _parse_launch_fields(
    params: dict[str, object],
    order: tuple[Field, ...],
    parsed: Mapping[Field, object] | None = None,
) -> dict[Field, object]:
    """Parse one phase of a validation run's launch parameters (R26/R27).

    ``iter_fields`` raises each field's own component error at its step; this helper maps
    those errors onto the ONE user-visible text each failure class has always had (the
    three wrappers are byte-identical to the pre-rabmo.3 code). Dispatch is by exception
    class, because the lazily-raised error does not say which field produced it.

    Args:
        params: The run's stored parameters.
        order: The phase's parse order from :data:`~vibe_quant.launch_settings.PARSE_ORDER`.
        parsed: Fields parsed by an earlier phase, if any (``run`` parses around the
            detail/clamp/latency block).

    Returns:
        The parsed fields, including ``parsed``.

    Raises:
        ValidationRunnerError: For an invalid balance, liquidation, sizing, cap or fee.
    """
    fields: dict[Field, object] = dict(parsed or {})
    try:
        fields.update(iter_fields(params, order, parsed=fields))
    except BalanceConfigError as exc:
        raise ValidationRunnerError(f"invalid initial_balance: {exc}") from exc
    except (LiquidationConfigError, SizingConfigError) as exc:
        # The liquidation mode/leverage and the sizing (incl. the leverage cap) keep their
        # component's own text.
        raise ValidationRunnerError(str(exc)) from exc
    except ValueError as exc:
        # The fee step raises a plain ValueError (FeeSchedule.from_params).
        raise ValidationRunnerError(f"invalid fee schedule parameters: {exc}") from exc
    return fields


def _parse_symbols(run_config: dict[str, object]) -> list[str]:
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


def resolve_detail_timeframe(
    run_config: dict[str, object],
    strategy_timeframe: str,
    override: str | None,
    *,
    catalog_path: Path | None = None,
) -> str | None:
    """Resolve the detail (fill-resolution) timeframe for a run.

    Explicit overrides (argument or ``parameters.detail_timeframe``) win;
    they must be a loadable timeframe no coarser than the strategy's
    (``ValidationRunnerError`` otherwise).
    Otherwise strategies coarser than 1m use 1m detail -- whose presence
    and coverage :func:`clamp_run_window` then enforces (missing 1m data
    fails the run; it no longer silently falls back to strategy-TF
    fills). 1m strategies use 5s detail only when every symbol has it;
    without it they run with full one-bar deferral (see
    :meth:`ValidationRunner._augment_strategy_params_for_validation`).

    Args:
        run_config: Run configuration from database.
        strategy_timeframe: Primary strategy timeframe.
        override: Explicit detail timeframe override.
        catalog_path: Catalog to look up sub-minute detail data in. Defaults
            to ``vibe_quant.data.catalog.DEFAULT_CATALOG_PATH``, read at call
            time.

    Returns:
        Detail timeframe string (e.g., '1m') or None.
    """
    # Explicit override always wins (argument, then run parameters)
    params = run_config.get("parameters")
    if override is None and isinstance(params, dict) and params.get("detail_timeframe"):
        override = str(params["detail_timeframe"])
    if override is not None:
        # Orders release on the strategy's own next detail bar, so the
        # detail bar must be loadable and not coarser than the strategy
        # bar (1h detail under 15m bars = orders up to 1h late).
        if override not in INTERVAL_TO_AGGREGATION:
            msg = f"Unknown detail timeframe {override!r}"
            raise ValidationRunnerError(msg)
        strategy_secs = _TIMEFRAME_SECONDS.get(strategy_timeframe)
        if strategy_secs is not None and _TIMEFRAME_SECONDS[override] > strategy_secs:
            msg = (
                f"Detail timeframe {override!r} is coarser than the "
                f"{strategy_timeframe} strategy timeframe"
            )
            raise ValidationRunnerError(msg)
        return override

    symbols = _parse_symbols(run_config)
    if not symbols:
        return None
    if not isinstance(run_config.get("start_date"), str) or not isinstance(
        run_config.get("end_date"), str
    ):
        return None

    strategy_seconds = _TIMEFRAME_SECONDS.get(strategy_timeframe)
    if strategy_seconds is None:
        return None
    detail_tf = (
        _DEFAULT_COARSE_DETAIL_TIMEFRAME
        if strategy_seconds > _TIMEFRAME_SECONDS["1m"]
        else _SUB_MINUTE_DETAIL_TIMEFRAME
    )
    if (
        detail_tf not in INTERVAL_TO_AGGREGATION
        or _TIMEFRAME_SECONDS[detail_tf] >= strategy_seconds
    ):
        return None

    if detail_tf == _SUB_MINUTE_DETAIL_TIMEFRAME:
        # Sub-minute data is optional: use it only if every symbol has it
        if catalog_path is None:
            catalog_path = catalog_mod.DEFAULT_CATALOG_PATH
        catalog_mgr = CatalogManager(catalog_path)
        for symbol in symbols:
            if catalog_mgr.get_bar_date_range(symbol, detail_tf) is None:
                logger.info(
                    "No %s detail data for %s — %s strategy runs with one-bar "
                    "deferral instead of latency",
                    detail_tf,
                    symbol,
                    strategy_timeframe,
                )
                return None

    logger.info("Using %s detail data for %s strategy", detail_tf, strategy_timeframe)
    return detail_tf


def clamp_run_window(
    run_config: dict[str, object],
    fill_timeframe: str,
    *,
    catalog_path: Path | None = None,
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Clamp the run window to the span the fill-resolution data covers.

    The window end used to be allowed past the last 1m bar, which made
    the detail data silently drop out (fills at the next 4h close with
    latency on). Now every symbol must have ``fill_timeframe`` bars;
    the window is clamped to whole UTC days fully covered by all of
    them, with a WARNING and a note persisted in the result notes.

    Args:
        run_config: Run configuration from database.
        fill_timeframe: Timeframe of the fill-resolution bars.
        catalog_path: Catalog to look up the bars' coverage in. Defaults to
            ``vibe_quant.data.catalog.DEFAULT_CATALOG_PATH``, read at call time.

    Returns:
        ``(run_config, note)`` -- a copy with effective dates when
        clamped (note describes requested vs effective window), else the
        original config and ``None``.

    Raises:
        ValidationRunnerError: a symbol has no ``fill_timeframe`` data,
            or no full day of the window is covered.
    """
    run_start = run_config.get("start_date")
    run_end = run_config.get("end_date")
    if not isinstance(run_start, str) or not isinstance(run_end, str):
        return run_config, None
    if fill_timeframe not in INTERVAL_TO_AGGREGATION:
        return run_config, None
    start = _parse_run_date(run_start, "start_date")
    end = _parse_run_date(run_end, "end_date")

    if catalog_path is None:
        catalog_path = catalog_mod.DEFAULT_CATALOG_PATH
    catalog_mgr = CatalogManager(catalog_path)
    bar_seconds = _TIMEFRAME_SECONDS.get(fill_timeframe, 0)
    covered_from: date | None = None
    covered_to: date | None = None
    for symbol in _parse_symbols(run_config):
        date_range = catalog_mgr.get_bar_date_range(symbol, fill_timeframe)
        if date_range is None:
            msg = (
                f"No {fill_timeframe} bars for {symbol} in the catalog. Validation "
                f"needs {fill_timeframe} data for intrabar fills/latency — download "
                f"it (Data Management) before validating."
            )
            raise ValidationRunnerError(msg)
        first_open, last_open = date_range
        # Whole UTC days covered: from the first midnight at/after the first
        # bar's open to the midnight at/before the last bar's close.
        first_day = first_open.date()
        if (first_open.hour, first_open.minute, first_open.second) != (0, 0, 0):
            first_day += timedelta(days=1)
        last_day = (last_open + timedelta(seconds=bar_seconds)).date()
        covered_from = first_day if covered_from is None else max(covered_from, first_day)
        covered_to = last_day if covered_to is None else min(covered_to, last_day)

    if covered_from is None or covered_to is None:
        return run_config, None
    eff_start = max(start, covered_from)
    eff_end = min(end, covered_to)
    if eff_end <= eff_start:
        msg = (
            f"{fill_timeframe} data covers {covered_from}..{covered_to}, which does not "
            f"overlap the run window {run_start}..{run_end}"
        )
        raise ValidationRunnerError(msg)
    if eff_start == start and eff_end == end:
        return run_config, None

    note: dict[str, object] = {
        "requested_start": run_start,
        "requested_end": run_end,
        "effective_start": eff_start.isoformat(),
        "effective_end": eff_end.isoformat(),
        "coverage_timeframe": fill_timeframe,
        "reason": f"{fill_timeframe} data covers {covered_from}..{covered_to}",
    }
    logger.warning(
        "Run window %s..%s clamped to %s..%s: %s data only covers %s..%s",
        run_start,
        run_end,
        eff_start,
        eff_end,
        fill_timeframe,
        covered_from,
        covered_to,
    )
    clamped = dict(run_config)
    clamped["start_date"] = eff_start.isoformat()
    clamped["end_date"] = eff_end.isoformat()
    return clamped, note


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
        logs_path: Path | str | None = None,
    ) -> None:
        """Initialize ValidationRunner.

        Args:
            db_path: Path to state database. Uses default if None.
            logs_path: Path for event log files. Defaults to ``log_dir()/events``
                (``VIBE_QUANT_LOG_DIR`` when set, else ``logs/events``).
        """
        self._state = StateManager(db_path)
        self._logs_path = (
            Path(logs_path) if logs_path is not None else log_dir(create=False) / "events"
        )
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

        # Post-only maker entries are validation-only and opt-in. Validate the
        # run-level maker params now so a bad value fails the run with a clear
        # message instead of surfacing from the generated on_start mid-engine.
        try:
            entry_order = self._resolve_maker_params(run_config)
        except ValidationRunnerError as exc:
            self._state.update_backtest_run_status(run_id, "failed", error_message=str(exc))
            raise

        # Liquidation overlay is opt-in; a bad mode fails the run before the engine starts.
        # Parsed ONCE per run: the venue config and every extraction reuse these settings.
        # Sizing and its leverage cap are validated in the same step, right after the
        # liquidation settings: a stored cap/risk the strategy could not honour, or a cap
        # the margin leverage cannot cover, fails the run before it reaches "running".
        # Both phases run BEFORE the detail check (R27a), while the balance and fee are
        # parsed after it inside _create_venue_config (R27b).
        try:
            pre = _parse_launch_fields(
                _launch_params(run_config), PARSE_ORDER["validation_pre_detail"]
            )
        except ValidationRunnerError as exc:
            self._state.update_backtest_run_status(run_id, "failed", error_message=str(exc))
            raise

        # Resolve fill-resolution detail data and clamp the run window to the
        # data actually available (bd vibe-quant-e70tl.13). Missing detail data
        # fails the run instead of silently degrading to strategy-TF fills.
        try:
            effective_detail = self._resolve_detail_timeframe(
                run_config, dsl.timeframe, detail_timeframe
            )
            run_config, window_note = self._clamp_run_window(
                run_config, effective_detail or dsl.timeframe
            )
        except ValidationRunnerError as exc:
            self._state.update_backtest_run_status(run_id, "failed", error_message=str(exc))
            raise

        # Determine latency preset -- with detail data it only selects the fill
        # model; orders are released on each strategy's own next detail bar
        effective_latency = self._resolve_latency(run_config, latency_preset)

        # Configure venue (timeframe-aware: no latency model for sub-5m bars
        # without detail data, nor with detail data -- own-bar release instead).
        # A bad fee parameter fails the run rather than silently defaulting.
        try:
            venue_config, launch = self._create_venue_config(
                run_config,
                effective_latency,
                timeframe=dsl.timeframe,
                has_detail_data=effective_detail is not None,
                pre=pre,
            )
        except ValidationRunnerError as exc:
            self._state.update_backtest_run_status(run_id, "failed", error_message=str(exc))
            raise

        # Update run status to running
        self._state.update_backtest_run_status(run_id, "running")

        logger.info(
            "Run %d setup: strategy=%s (id=%d) timeframe=%s latency=%s detail=%s "
            "leverage=%sx balance=%s USDT",
            run_id,
            strategy_name,
            strategy_id,
            dsl.timeframe,
            venue_config.latency_preset or (
                self._OWN_DETAIL_RELEASE if effective_detail else "none"
            ),
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
                        "data_window": window_note,
                        **self._execution_event_fields(
                            venue_config, effective_latency, effective_detail
                        ),
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
                    launch=launch,
                )

                # Market-mode TWIN of the identical backtest (same window,
                # symbols, fee schedule, latency) so the maker entries can be
                # compared against the market baseline they replace. It never
                # writes trades or a result row -- the run keeps exactly one
                # result row, the post_only one (D3-6). A twin that fails is
                # reported in the notes and never fails the primary run.
                if entry_order == "post_only":
                    maker_notes = result.notes.get("maker_entries")
                    if isinstance(maker_notes, dict):
                        maker_notes["market_twin"] = self._run_market_twin(
                            run_id=run_id,
                            strategy_name=strategy_name,
                            dsl=dsl,
                            venue_config=venue_config,
                            run_config=run_config,
                            writer=writer,
                            detail_timeframe=effective_detail,
                            launch=launch,
                        )

                self._write_completion_event(writer, run_id, strategy_name, result)

            result.execution_time_seconds = time.monotonic() - start_time
            if window_note is not None:
                result.notes["data_window"] = window_note
            result.notes.update(self._execution_notes(run_config, effective_detail))
            result.notes["fee_schedule"] = venue_config.fee_schedule.to_notes()
            if result.funding_fallback_settlements:
                result.notes["funding"] = {
                    "fallback_settlements": result.funding_fallback_settlements,
                    "fallback_rate_per_8h": FundingCalculator().default_rate,
                }
            self._store_results(run_id, result)
            consistency = self._check_screening_consistency(
                run_id,
                strategy_id,
                strategy_name,
                result,
                run_config=run_config,
                sizing=launch.sizing,
            )
            if consistency is not None:
                result.notes["consistency"] = consistency
            if result.notes:
                self._state.update_result_notes(run_id, json.dumps(result.notes))

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

        # A bad liquidation mode fails the run before any window's engine starts.
        # Parsed ONCE per run: the venue config and every window's extraction reuse it.
        # The same sizing gate as ``run``: it only has to pass once for the whole run,
        # before any window's engine starts (no per-window reference check needs it).
        # Both phases run BEFORE the detail check (R27a); the balance and fee are parsed
        # after it inside _create_venue_config (R27b).
        try:
            pre = _parse_launch_fields(
                _launch_params(run_config), PARSE_ORDER["validation_pre_detail"]
            )
        except ValidationRunnerError as exc:
            self._state.update_backtest_run_status(run_id, "failed", error_message=str(exc))
            raise

        try:
            effective_detail = self._resolve_detail_timeframe(
                run_config, dsl.timeframe, detail_timeframe
            )
            run_config, _window_note = self._clamp_run_window(
                run_config, effective_detail or dsl.timeframe
            )
        except ValidationRunnerError as exc:
            self._state.update_backtest_run_status(run_id, "failed", error_message=str(exc))
            raise
        effective_latency = self._resolve_latency(run_config, latency_preset)
        try:
            venue_config, launch = self._create_venue_config(
                run_config,
                effective_latency,
                timeframe=dsl.timeframe,
                has_detail_data=effective_detail is not None,
                pre=pre,
            )
        except ValidationRunnerError as exc:
            self._state.update_backtest_run_status(run_id, "failed", error_message=str(exc))
            raise

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
                self._write_start_event(
                    writer,
                    run_id,
                    strategy_name,
                    venue_config,
                    extra=self._execution_event_fields(
                        venue_config, effective_latency, effective_detail
                    ),
                )
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
                        launch=launch,
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
            aggregate.notes.update(self._execution_notes(run_config, effective_detail))
            aggregate.notes["fee_schedule"] = venue_config.fee_schedule.to_notes()
            if aggregate.notes:
                self._state.update_result_notes(run_id, json.dumps(aggregate.notes))

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
        """Parse run date fields from DB config (delegates to the module function)."""
        return _parse_run_date(value, field_name)

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
            # Pooled trade PF over every window's trades (same definition
            # as a single run), not a trade-weighted mean of window PFs.
            profit_factor=profit_factor(
                t.net_pnl for r in window_results for t in r.trades
            ),
            win_rate=win_rate,
            total_trades=total_trades,
            total_fees=sum(r.total_fees for r in window_results),
            total_funding=sum(r.total_funding for r in window_results),
            total_slippage=sum(r.total_slippage for r in window_results),
            # cost-free PnL adds across windows like the costs it is measured against
            gross_pnl=sum(r.gross_pnl for r in window_results),
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
        aggregate.liquidation_events = [e for r in window_results for e in r.liquidation_events]
        liquidation_note = ValidationRunner._aggregate_liquidation_notes(
            window_results, total_trades
        )
        if liquidation_note is not None:
            aggregate.notes["liquidation"] = liquidation_note
        margin_notes = [
            cast("dict[str, Any]", r.notes["margin_limit"])
            for r in window_results
            if "margin_limit" in r.notes
        ]
        if margin_notes:
            aggregate.notes["margin_limit"] = {
                "max_margin_pct": margin_notes[0]["max_margin_pct"],
                "refused_entries": sum(int(n["refused_entries"]) for n in margin_notes),
            }
        return aggregate

    @staticmethod
    def _aggregate_liquidation_notes(
        window_results: list[ValidationResult], total_trades: int
    ) -> dict[str, object] | None:
        """Sum the windows' ``notes["liquidation"]`` (``None`` when no window used the overlay).

        Counts and loss sums add up, ``pct_trades`` is recomputed on the pooled trades,
        ``bust_ts`` is the first window's bust and ``path_divergence`` any window's; the
        event list concatenates (capped like a single run's).
        """
        notes = [r.notes["liquidation"] for r in window_results if "liquidation" in r.notes]
        if not notes:
            return None
        from vibe_quant.liquidation.scan import _MAX_NOTE_EVENTS  # only a liquidation run gets here

        first = cast("dict[str, Any]", notes[0])
        windows = cast("list[dict[str, Any]]", notes)
        count = sum(int(n["count"]) for n in windows)
        merged: dict[str, Any] = dict(first)
        merged["count"] = count
        for name in ("total_forfeited", "total_realized_pnl", "total_insurance_deficit"):
            merged[name] = float(sum(float(n[name]) for n in windows))
        merged["dropped"] = sum(int(n["dropped"]) for n in windows)
        merged["pct_trades"] = (100.0 * count / total_trades) if total_trades > 0 else 0.0
        merged["bust_ts"] = next((n["bust_ts"] for n in windows if n["bust_ts"] is not None), None)
        merged["path_divergence"] = any(bool(n["path_divergence"]) for n in windows)
        merged["events"] = [e for n in windows for e in n["events"]][:_MAX_NOTE_EVENTS]
        return merged

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

    # notes["execution_release"] when detail data is loaded: every strategy
    # queues its commands and releases them on its OWN next detail bar
    # (generated command outbox, command_release="bar"). NT's LatencyModel
    # released venue-wide on the next datum of ANY instrument, so in a
    # multi-symbol run the 2nd symbol filled at its signal-bar close
    # (bd vibe-quant-yul7u.11).
    _OWN_DETAIL_RELEASE = "own_detail_bar"

    def _execution_event_fields(
        self,
        venue_config: VenueConfig,
        latency_preset: LatencyPreset | str | None,
        detail_timeframe: str | None,
    ) -> dict[str, object]:
        """BACKTEST_START fields: the preset that chose the fill model (the
        venue itself runs no LatencyModel with detail data) + release mode."""
        if detail_timeframe is not None:
            preset: object = latency_preset or LatencyPreset.CLOUD
        else:
            preset = venue_config.latency_preset
        return {
            "latency_preset": str(preset) if preset else None,
            "execution_release": self._OWN_DETAIL_RELEASE if detail_timeframe else None,
        }

    def _execution_notes(
        self, run_config: dict[str, object], detail_timeframe: str | None
    ) -> dict[str, object]:
        """Result notes for own-bar release, incl. an overridden per-run delay
        (see :meth:`_augment_strategy_params_for_validation`)."""
        if detail_timeframe is None:
            return {}
        notes: dict[str, object] = {"execution_release": self._OWN_DETAIL_RELEASE}
        requested = self._build_strategy_params(run_config).get("execution_delay_probability")
        if requested:
            notes["execution_delay_override"] = {"requested": requested, "applied": 0.0}
        return notes

    def _create_venue_config(
        self,
        run_config: dict[str, object],
        latency_preset: LatencyPreset | str | None,
        timeframe: str = "4h",
        *,
        has_detail_data: bool = False,
        pre: Mapping[Field, object] | None = None,
    ) -> tuple[VenueConfig, LaunchSettings]:
        """Create venue configuration for validation.

        For sub-5m timeframes WITHOUT detail data, latency is skipped because
        NT's LatencyModel defers orders to the next bar (60s on 1m data)
        regardless of actual latency value. Slippage probability compensates.

        When detail data IS available (e.g., 1m bars under a 4h strategy),
        the fill model is still chosen from the latency preset (identical to
        the former latency path), but the LatencyModel itself is dropped:
        each strategy releases its orders on its own next detail bar instead
        (see ``_OWN_DETAIL_RELEASE``). The preset only matters for tick data.

        Args:
            run_config: Run configuration.
            latency_preset: Latency preset to use.
            timeframe: Strategy primary timeframe.
            has_detail_data: Whether sub-bar detail data is loaded.
            pre: The launch fields ``run`` / ``run_walk_forward`` parsed before the
                detail check (liquidation, sizing, cap). ``None`` for a direct caller,
                which parses its own balance/liquidation/fee row instead.

        Returns:
            The configured :class:`VenueConfig` and the run's :class:`LaunchSettings`.
        """
        from decimal import Decimal

        params = _launch_params(run_config)

        # Balance and fee are the LAST launch fields parsed: ``run`` parses liquidation,
        # sizing and the leverage cap before the detail check (R27a), so a bad balance
        # beats a bad fee here and the detail-coverage check beats both (R27b). A direct
        # caller (``pre`` is None) has nothing parsed yet and walks its own row
        # (balance, liquidation, fee); it never parses the sizing (R27e).
        if pre is None:
            fields = _parse_launch_fields(params, PARSE_ORDER["validation_direct_venue"])
        else:
            fields = _parse_launch_fields(
                params, PARSE_ORDER["validation_post_detail"], parsed=pre
            )
        launch = LaunchSettings.from_fields(fields)

        starting_balance = launch.starting_balance
        # One rule with screening (vibe-quant-6wq6w): the leverage is validated in EVERY
        # mode and never silently falls back to 10x.
        default_leverage = Decimal(str(launch.liquidation.leverage))
        fee_schedule = launch.fee_schedule

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
            return (
                replace(
                    create_venue_config_for_validation(
                        starting_balance_usdt=starting_balance,
                        default_leverage=default_leverage,
                        latency_preset=None,
                    ),
                    fee_schedule=fee_schedule,
                ),
                launch,
            )

        venue_config = replace(
            create_venue_config_for_validation(
                starting_balance_usdt=starting_balance,
                default_leverage=default_leverage,
                latency_preset=latency_preset or LatencyPreset.CLOUD,
            ),
            fee_schedule=fee_schedule,
        )
        if has_detail_data:
            # Fill model chosen above from the preset; release is own-bar.
            venue_config = replace(
                venue_config, latency_preset=None, latency_config=None
            )
        return venue_config, launch

    def _run_backtest(
        self,
        run_id: int,
        strategy_name: str,
        dsl: StrategyDSL,
        venue_config: VenueConfig,
        run_config: dict[str, object],
        writer: EventWriter,
        detail_timeframe: str | None = None,
        entry_order_override: str | None = None,
        write_trade_events: bool = True,
        *,
        launch: LaunchSettings,
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
            entry_order_override: Force entry_order for this run (the market-mode
                twin of a post_only run); None keeps the run's own parameter.
            write_trade_events: False for the twin, which must not write trade
                events (it also never reaches result storage).
            launch: The run's already-parsed launch settings (``run`` /
                ``run_walk_forward`` parse them once); the venue and the extraction
                reuse the same settings.

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
        from vibe_quant.validation.extraction import finest_timeframe, maker_entry_stats

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
        from vibe_quant.dsl import aux_data

        aux_data.configure(None, catalog_path)  # needs_context indicators (default archive)
        # run_config window is already clamped to data coverage here
        aux_data.preflight(
            [c.type for c in dsl.indicators.values()], symbols, start_date, end_date
        )
        for symbol in symbols:
            if symbol in INSTRUMENT_CONFIGS:
                instrument = create_instrument(symbol)
                catalog_mgr.write_instrument(instrument)

        # Compile strategy to an importable module (registers in sys.modules)
        module = self._compiler.compile_to_module(dsl)
        class_name = "".join(word.capitalize() for word in dsl.name.split("_"))
        module_path = module.__name__  # content-addressed (vibe-quant-e70tl.1)

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
        # The backtest margin limit divides notionals by the RUN leverage (the venue's own),
        # and only when the limit is on, so an off run's config stays byte-identical. Both
        # values come from the ONE launch the run already parsed (R40: no second parse).
        strategy_params.update(
            {
                key: value
                for key, value in launch.sizing.strategy_params(
                    launch.liquidation.leverage
                ).items()
                if key == "margin_leverage"
            }
        )
        # Market-mode twin: same run, market entries (D3-6).
        if entry_order_override is not None:
            strategy_params["entry_order"] = entry_order_override

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
        # The mode the generated config actually got: post_only drives the
        # maker-entry statistics recorded in the notes.
        effective_entry_order = str(strategy_params.get("entry_order", "market"))
        logger.info(
            "Run %d strategy params: %s",
            run_id,
            strategy_params if strategy_params else "(compiled defaults)",
        )

        if detail_timeframe and detail_timeframe not in INTERVAL_TO_AGGREGATION:
            # Venue latency is off with detail data; without a release bar
            # orders would fill at the signal bar's own close.
            msg = f"Unknown detail timeframe {detail_timeframe!r}"
            raise ValidationRunnerError(msg)

        # Build strategy configs (one per symbol). With detail data each
        # strategy releases its queued commands on its OWN detail bar.
        strategy_configs: list[ImportableStrategyConfig] = []
        for symbol in symbols:
            instrument_id = f"{symbol}-PERP.BINANCE"
            config_dict = {"instrument_id": instrument_id, **strategy_params}
            if detail_timeframe:
                release_step, release_agg = INTERVAL_TO_AGGREGATION[detail_timeframe]
                config_dict["command_release"] = "bar"
                config_dict["command_release_bar_type"] = (
                    f"{instrument_id}-{release_step}-{release_agg.name}-LAST-EXTERNAL"
                )
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
                execution_timeframe=finest_timeframe(
                    all_timeframes | ({detail_timeframe} if detail_timeframe else set())
                ),
                catalog_path=catalog_path,
                liquidation_mode=launch.liquidation.mode,
            )

            # Log trade events (the market-mode twin writes none)
            if write_trade_events:
                for trade in result.trades:
                    self._write_trade_events(writer, run_id, strategy_name, trade)
                self._write_liquidation_events(writer, run_id, strategy_name, result)

            if effective_entry_order == "post_only":
                timeout_raw = strategy_params.get("maker_entry_timeout_bars", 3)
                timeout_bars = (
                    timeout_raw
                    if isinstance(timeout_raw, int) and not isinstance(timeout_raw, bool)
                    else 3
                )
                result.notes["maker_entries"] = maker_entry_stats(
                    engine,
                    primary_bar_ns=_TIMEFRAME_SECONDS.get(dsl.timeframe, 3600)
                    * 1_000_000_000,
                    maker_entry_timeout_bars=timeout_bars,
                )

            if strategy_params.get("max_margin_pct") is not None:
                result.notes["margin_limit"] = {
                    "max_margin_pct": strategy_params["max_margin_pct"],
                    "refused_entries": sum(
                        int(getattr(strategy, "_margin_refusals", 0))
                        for strategy in engine.trader.strategies()
                    ),
                }

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

    def _run_market_twin(
        self,
        *,
        run_id: int,
        strategy_name: str,
        dsl: StrategyDSL,
        venue_config: VenueConfig,
        run_config: dict[str, object],
        writer: EventWriter,
        detail_timeframe: str | None,
        launch: LaunchSettings,
    ) -> dict[str, object]:
        """Market-mode twin of a post_only run, as the notes entry (D3-6).

        The twin is a comparison aid, so a failure is logged and returned as
        ``{"error": ...}`` instead of failing the already-finished primary run.
        """
        try:
            twin = self._run_backtest(
                run_id=run_id,
                strategy_name=strategy_name,
                dsl=dsl,
                venue_config=venue_config,
                run_config=run_config,
                writer=writer,
                detail_timeframe=detail_timeframe,
                entry_order_override="market",
                write_trade_events=False,
                launch=launch,
            )
        except Exception as exc:
            logger.exception("Run %d: market twin failed; the post_only result is kept", run_id)
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {
            "sharpe_ratio": twin.sharpe_ratio,
            "total_return": twin.total_return,
            "total_trades": twin.total_trades,
            "total_fees": twin.total_fees,
        }

    def _resolve_maker_params(self, run_config: dict[str, object]) -> str:
        """Validate and return the run's entry_order (vibe-quant-2gyk1.4 D3-6).

        Post-only maker entries are validation-only and opt-in. A bad mode or
        maker knob fails the run before the engine starts, rather than being
        rejected by the generated on_start mid-run.

        Raises:
            ValidationRunnerError: For an unknown entry_order or a
                maker_offset_ticks / maker_entry_timeout_bars < 1.
        """
        params = self._build_strategy_params(run_config)
        entry_order = params.get("entry_order", "market")
        if entry_order not in ("market", "post_only"):
            msg = f"entry_order must be 'market' or 'post_only', got {entry_order!r}"
            raise ValidationRunnerError(msg)
        if entry_order == "post_only":
            for key, default in (
                ("maker_offset_ticks", 1),
                ("maker_entry_timeout_bars", 3),
            ):
                value = params.get(key, default)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    msg = f"{key} must be an integer >= 1, got {value!r}"
                    raise ValidationRunnerError(msg)
        return str(entry_order)

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
            # Skip meta-keys that aren't strategy parameters. The fee schedule
            # is venue-level; forwarding it would break the generated config.
            if key in (
                "sweep",
                "overfitting_filters",
                "fee_tier",
                "bnb_discount",
                "fee_schedule_source",
                "liquidation_mode",
            ):
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

        When detail data provides sub-bar resolution, orders are released on
        the strategy's own next detail bar (command outbox), which requires
        execution_delay_probability=0 -- the generated on_start raises
        otherwise, so an explicit per-run delay is overridden.

        Without finer data (1m strategies have no 5s catalog data) latency
        is skipped, so EVERY entry/exit is deferred one bar, exactly like
        screening. The previous 30% deferral left 70% of fills at the signal
        bar's own close -- same-bar look-ahead (bd vibe-quant-e70tl.13).
        """
        augmented = dict(params)
        if has_detail_data:
            if augmented.get("execution_delay_probability"):
                logger.warning(
                    "execution_delay_probability=%s ignored: detail data releases "
                    "orders on each strategy's own next detail bar",
                    augmented["execution_delay_probability"],
                )
            augmented["execution_delay_probability"] = 0.0
        elif timeframe in self._SUB_BAR_TIMEFRAMES:
            augmented.setdefault("execution_delay_probability", 1.0)
        return augmented

    def _resolve_detail_timeframe(
        self,
        run_config: dict[str, object],
        strategy_timeframe: str,
        override: str | None,
    ) -> str | None:
        """Resolve the detail (fill-resolution) timeframe for a run.

        Delegate to :func:`resolve_detail_timeframe` (kept as a method so
        existing call sites and test monkeypatches keep working).
        """
        return resolve_detail_timeframe(run_config, strategy_timeframe, override)

    def _clamp_run_window(
        self,
        run_config: dict[str, object],
        fill_timeframe: str,
    ) -> tuple[dict[str, object], dict[str, object] | None]:
        """Clamp the run window to the span the fill-resolution data covers.

        Delegate to :func:`clamp_run_window` (kept as a method so existing
        call sites and test monkeypatches keep working).
        """
        return clamp_run_window(run_config, fill_timeframe)

    def _parse_symbols(self, run_config: dict[str, object]) -> list[str]:
        """Parse symbol list from run configuration.

        Delegate to :func:`_parse_symbols`.
        """
        return _parse_symbols(run_config)

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
        execution_timeframe: str | None = None,
        catalog_path: Path | None = None,
        liquidation_mode: str = "off",
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
            execution_timeframe / catalog_path: Execution bars for the
                mark-to-market drawdown.
            liquidation_mode: "off" (default), "isolated" or "cross".

        Returns:
            Populated ValidationResult.
        """
        from vibe_quant.validation.extraction import extract_results

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
            execution_timeframe=execution_timeframe,
            catalog_path=catalog_path,
            liquidation_mode=liquidation_mode,
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

    def _write_liquidation_events(
        self,
        writer: EventWriter,
        run_id: int,
        strategy_name: str,
        result: ValidationResult,
    ) -> None:
        """Write one LIQUIDATION event per liquidation hit (``liquidation_mode`` on)."""
        for hit in result.liquidation_events:
            writer.write(
                create_event(
                    event_type=EventType.LIQUIDATION,
                    run_id=str(run_id),
                    strategy_name=strategy_name,
                    data={
                        "position_id": hit["key"],
                        "symbol": hit["symbol"],
                        "liquidation_price": hit["price"],
                        "quantity": hit["qty"],
                        "loss": hit["loss"],
                        "ts_ns": hit["ts"],
                        "mode": hit["mode"],
                        "mark_price": hit["mark"],
                        "clearance_fee": hit["clearance_fee"],
                        "margin_balance": hit["margin_balance"],
                        "maintenance_margin": hit["maint_margin"],
                    },
                )
            )

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
        # Result + trades replace any prior attempt of this run atomically.
        self._state.save_backtest_result(
            run_id, result.to_metrics_dict(), trades=[t.to_dict() for t in result.trades]
        )

    def _check_screening_consistency(
        self,
        run_id: int,
        strategy_id: int,
        strategy_name: str,
        result: ValidationResult,
        run_config: dict[str, object] | None = None,
        sizing: SizingSettings | None = None,
    ) -> dict[str, object] | None:
        """Flag overfit promotions: validation collapsing vs screening.

        Compares this validation result against the strategy's screening
        reference (the screening row for the validated params, else the best
        finite-Sharpe row with trades, else the discovery champion's
        full-range metrics) and returns the report for the result notes when
        flagged (vibe-quant-o11tp). Best-effort — never fails the run.

        Only a reference at the same sizing is comparable (S29): the same strategy run at a
        different exposure cap is a different policy, not an overfit/underfit signal.
        """
        from vibe_quant.validation.consistency import (
            assess_consistency,
            find_screening_reference,
        )

        try:
            cfg = run_config or {}
            start, end = cfg.get("start_date"), cfg.get("end_date")
            val_window = (str(start), str(end)) if start and end else None
            params = cfg.get("parameters")
            val_sizing = (
                sizing.to_params()
                if sizing is not None
                else recorded_sizing(params if isinstance(params, dict) else {}).to_params()
            )
            reference = find_screening_reference(
                self._state,
                strategy_id,
                strategy_name,
                validated_params=self._build_strategy_params(cfg),
                val_window=val_window,
                val_symbols=[str(x) for x in syms] if isinstance(syms := cfg.get("symbols"), list) else None,
                val_sizing=val_sizing,
            )
            if reference is None:
                return None
            report = assess_consistency(
                reference,
                val_sharpe=result.sharpe_ratio,
                val_trades=result.total_trades,
                val_window=val_window,
            )
            if not report.is_flagged:
                logger.info(
                    "Run %d consistent with screening (%s): sharpe %.2f → %.2f",
                    run_id,
                    reference.source,
                    reference.sharpe,
                    result.sharpe_ratio,
                )
                return None
            for flag in report.flags:
                logger.warning("Run %d %s", run_id, flag)
            return report.to_dict()
        except Exception:  # noqa: BLE001 — advisory check must not fail the run
            logger.exception("Run %d: consistency check failed", run_id)
            return None


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
