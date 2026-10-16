"""Paper trading node implementation.

Provides PaperTradingNode class that wraps NautilusTrader TradingNode
for paper trading with Binance testnet (live only with explicit opt-in).

Safety model (see :mod:`vibe_quant.paper.guard`):

* a :class:`~vibe_quant.paper.guard.TradingGuard` actor gates every order and
  enforces max drawdown / daily loss / consecutive losses / position count;
* halt and kill flatten positions with reduce-only market orders, cancel orders
  and stop strategies; pause only blocks new entries (SL/TP stay live);
* operator commands arrive through the ``paper_commands`` table
  (:class:`~vibe_quant.paper.persistence.PaperCommandQueue`), not POSIX signals,
  and the system kill switch in the state DB is polled continuously;
* the node trades the exact strategy parameters + leverage of the validation run
  it was started from and logs any difference;
* the venue can close a position on its own (liquidation, ADL) without NT ever seeing
  the fill, so a REPORT-ONLY watcher task polls forceOrders beside the control loop,
  writes a LIQUIDATION event and alerts — it never submits, cancels, closes or halts.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import signal
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.fees import FeeSchedule
from vibe_quant.liquidation_settings import LiquidationConfigError, LiquidationSettings
from vibe_quant.logging.events import Event, EventType, create_event
from vibe_quant.logging.writer import EventWriter
from vibe_quant.paper.config import (
    DEFAULT_MARGIN_TYPE,
    DEFAULT_VALIDATION_LEVERAGE,
    ConfigurationError,
    PaperTradingConfig,
)
from vibe_quant.paper.errors import ErrorContext, ErrorHandler
from vibe_quant.paper.force_orders import build_binance_fetch
from vibe_quant.paper.guard import (
    OPERATOR_RESUMABLE_HALTS,
    CloseAllResult,
    GuardListener,
    GuardState,
    HaltReason,
    RiskState,
    TradingGuard,
    TradingGuardConfig,
)
from vibe_quant.paper.liquidation_watch import LiquidationWatcher
from vibe_quant.paper.persistence import (
    PaperCommand,
    PaperCommandQueue,
    StateCheckpoint,
    StatePersistence,
)

if TYPE_CHECKING:
    from types import ModuleType

    from vibe_quant.alerts.telegram import TelegramBot
    from vibe_quant.dsl.schema import StrategyDSL
    from vibe_quant.paper.force_orders import ForceOrder

logger = logging.getLogger(__name__)

# ValidationRunner._build_strategy_params skips these meta keys.
_RUN_META_KEYS = frozenset({"sweep", "overfitting_filters"})
# Backtest-only simulation knobs that must never reach a live strategy.
_BACKTEST_ONLY_PARAMS = frozenset(
    {
        "execution_delay_probability",
        "execution_delay_seed",
        "command_release",
        "command_release_bar_type",
    }
)

#: R42b: how long the venue may stay unconnected before the node reports the startup
#: refusal (an exchange that refuses the margin change leaves the node not trading).
STARTUP_CONNECT_TIMEOUT_S = 120

#: ``PositionOpened.side`` (a ``PositionSide``) -> the sign the solvency formula wants.
_POSITION_SIDE_SIGNS = {"LONG": 1, "SHORT": -1}

#: R44b: the estimate's ``basis`` names the method; it is not a free-form note.
_ISOLATED_LP_BASIS = (
    "isolated formula at entry: entry_price, leverage, Binance tier MMR (table {version}); "
    "excludes fees, funding and later margin changes"
)
#: R45: a crossed LP depends on the account wallet and every other open position.
_CROSS_LP_REASON = "cross LP depends on the account wallet and other positions; not estimated"


class _TradingNodeLifecycle(Protocol):
    """Minimal lifecycle contract for live trading node integration."""

    def run(self, raise_exception: bool = False) -> None:
        """Start and run the node (blocking — used only by legacy tests)."""

    async def run_async(self) -> None:
        """Start and run the node on the current event loop."""

    def stop(self) -> None:
        """Stop the node gracefully."""

    async def stop_async(self) -> None:
        """Stop the node gracefully on the current event loop."""

    def dispose(self) -> None:
        """Dispose resources."""


class NodeState(StrEnum):
    """State of the paper trading node."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    HALTED = "halted"
    STOPPED = "stopped"
    ERROR = "error"


_GUARD_TO_NODE_STATE = {
    GuardState.RUNNING: NodeState.RUNNING,
    GuardState.PAUSED: NodeState.PAUSED,
    GuardState.HALTED: NodeState.HALTED,
}


@dataclass
class NodeStatus:
    """Current status of the paper trading node.

    Attributes:
        state: Current node state.
        started_at: When node was started.
        updated_at: Last status update.
        halt_reason: Reason if halted.
        error_message: Error message if in error state.
        positions: Number of open positions.
        daily_pnl: Daily PnL.
        total_pnl: Total PnL since start.
        trades_today: Number of trades today.
        consecutive_losses: Current consecutive loss count.
    """

    state: NodeState = NodeState.INITIALIZING
    started_at: datetime | None = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    halt_reason: HaltReason | None = None
    error_message: str | None = None
    positions: int = 0
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0

    def to_dict(self) -> dict[str, str | int | float | None]:
        """Convert to dictionary for storage/serialization."""
        return {
            "state": self.state.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "updated_at": self.updated_at.isoformat(),
            "halt_reason": self.halt_reason.value if self.halt_reason else None,
            "error_message": self.error_message,
            "positions": self.positions,
            "daily_pnl": round(self.daily_pnl, 2),
            "total_pnl": round(self.total_pnl, 2),
            "trades_today": self.trades_today,
            "consecutive_losses": self.consecutive_losses,
        }


@dataclass
class ResolvedRunConfig:
    """What the node actually trades, plus the validated reference it came from."""

    symbols: list[str]
    strategy_params: dict[str, object]
    leverage: int
    margin_type: str
    validated: dict[str, object] | None = None
    diff: list[dict[str, object]] = field(default_factory=list)


class OrderRejectedError(Exception):
    """Venue rejected an order (routed through ErrorHandler for halt + alert)."""


class OrderDeniedError(Exception):
    """NT RiskEngine denied an order."""


def reject_context_indicators(dsl: StrategyDSL) -> None:
    """Refuse strategies using ``needs_context`` indicators (FUNDING, ...).

    Their aux data comes from the historical archive; there is no live feed.
    """
    from vibe_quant.dsl.indicators import indicator_registry

    bad = sorted(
        {
            cfg.type
            for cfg in dsl.indicators.values()
            if (spec := indicator_registry.get(cfg.type)) is not None and spec.needs_context
        }
    )
    if bad:
        raise ConfigurationError(
            f"Paper trading does not support context indicators ({', '.join(bad)}): "
            "they read archived data with no live feed"
        )


def _ns_to_dt(ns: int) -> datetime:
    return datetime.fromtimestamp(ns / 1_000_000_000, tz=UTC)


def _int_map(value: object) -> dict[str, int]:
    """The int-valued entries of a stored map (a checkpoint round-trips through JSON)."""
    result: dict[str, int] = {}
    if isinstance(value, dict):
        for key, entry in value.items():
            if isinstance(entry, int) and not isinstance(entry, bool):
                result[str(key)] = entry
    return result


def validated_run_strategy_params(run: dict[str, Any]) -> dict[str, object]:
    """Strategy overrides exactly as ``ValidationRunner`` builds them for this run."""
    from vibe_quant.validation.runner import ValidationRunner

    # _build_strategy_params only reads run_config; reuse it so paper can never
    # drift from what validation applied. (No ValidationRunner state needed.)
    runner = ValidationRunner.__new__(ValidationRunner)
    return runner._build_strategy_params(run)


def validated_run_liquidation(run: dict[str, Any]) -> LiquidationSettings:
    """The stored ``liquidation_mode`` + ``leverage`` of a validation run.

    One shared rule for reading a run's settings
    (:meth:`LiquidationSettings.from_params`): a missing key is the default, and
    an explicit ``None`` / garbage value is rejected instead of silently
    defaulted. ``ValidationRunner._create_venue_config`` uses ``leverage``.

    Raises:
        ConfigurationError: If the run's stored liquidation settings are invalid.
    """
    params = run.get("parameters")
    try:
        return LiquidationSettings.from_params(params if isinstance(params, dict) else {})
    except LiquidationConfigError as exc:
        raise ConfigurationError(
            f"Validation run {run.get('id')} has invalid liquidation settings: {exc}"
        ) from exc


def validated_run_leverage(run: dict[str, Any]) -> float:
    """Venue leverage used by ``ValidationRunner._create_venue_config`` for this run.

    A thin wrapper over :func:`validated_run_liquidation` (the single parse rule).
    """
    return validated_run_liquidation(run).leverage


def validated_leverage_override(value: object) -> int:
    """An explicit ``leverage`` override (config-file / :class:`PaperTradingConfig`).

    The same shared whole-number-in-``[1, 125]`` rule as the stored settings, so
    ``int()`` truncation (``2.5`` -> ``2``) and ``bool`` (``True`` -> ``1``) can
    never reach the venue.

    Raises:
        ConfigurationError: If ``value`` is not a whole number in ``[1, 125]``.
    """
    try:
        settings = LiquidationSettings.from_params({"leverage": value})
    except LiquidationConfigError as exc:
        raise ConfigurationError(str(exc)) from exc
    return int(settings.leverage)


class _GuardBridge(GuardListener):
    """Forwards guard callbacks to the node."""

    def __init__(self, node: PaperTradingNode) -> None:
        self._node = node

    def on_state_change(
        self,
        state: GuardState,
        reason: HaltReason | None,
        message: str,
        metrics: dict[str, object],
    ) -> None:
        self._node._on_guard_state_change(state, reason, message, metrics)

    def on_order_rejected(self, event: Any) -> None:
        self._node._on_order_rejected(event)

    def on_order_denied(self, event: Any, *, by_gate: bool) -> None:
        self._node._on_order_denied(event, by_gate=by_gate)

    def on_position_opened(self, event: Any) -> None:
        self._node._on_position_opened(event)

    def on_position_closed(self, event: Any) -> None:
        self._node._on_position_closed(event)

    def on_warning(self, message: str) -> None:
        self._node._on_guard_warning(message)


def _is_tier_error(exc: Exception) -> bool:
    """True for the liquidation rule's own domain errors (expected, reported as a reason)."""
    from vibe_quant.liquidation.tiers import InsolventPosition, TierTableError, UnknownSymbol

    return isinstance(exc, TierTableError | InsolventPosition | UnknownSymbol)


class PaperTradingNode:
    """Paper trading node for live simulated execution.

    Wraps NautilusTrader TradingNode for paper trading with Binance testnet.
    Provides:
    - Strategy deployment from DSL with the validated run's exact parameters
    - Risk enforcement + halt/pause/close-all via TradingGuard
    - State persistence for crash recovery (risk state restored on restart)
    - Command-queue control, kill-switch polling, Telegram alerts

    Example:
        config = PaperTradingConfig(trader_id="PAPER-001", binance=..., validation_run_id=870,
                                    strategy_id=240)
        node = PaperTradingNode(config)
        await node.start()  # Runs until SIGINT/SIGTERM
    """

    def __init__(self, config: PaperTradingConfig) -> None:
        """Initialize paper trading node.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        errors = config.validate()
        if errors:
            raise ConfigurationError(f"Invalid configuration: {'; '.join(errors)}")

        self._config = config
        self._state_manager = StateManager(config.db_path)
        self._compiler = StrategyCompiler()
        self._status = NodeStatus()
        self._event_writer: EventWriter | None = None
        self._trading_node: _TradingNodeLifecycle | None = None
        self._persistence: StatePersistence | None = None
        self._commands: PaperCommandQueue | None = None
        self._guard: TradingGuard | None = None
        self._restored_risk: RiskState | None = None
        self._resolved: ResolvedRunConfig | None = None
        self._shutdown_event = asyncio.Event()
        self._strategy: StrategyDSL | None = None
        self._compiled_module: ModuleType | None = None
        self._compiled_strategies: list[object] = []
        self._error_handler = ErrorHandler(
            on_halt=self._on_error_halt,
            on_alert=self._on_error_alert,
        )
        self._alert_tasks: set[asyncio.Task[bool]] = set()
        self._control_task: asyncio.Task[None] | None = None
        self._liquidation_task: asyncio.Task[None] | None = None
        self._liquidation_stop = asyncio.Event()
        self._liquidation_watcher: LiquidationWatcher | None = None
        # R50b: the restored watch state plus the checkpoint timestamp it is anchored to.
        self._restored_watch: dict[str, dict[str, int]] | None = None
        self._restored_watch_ms = 0
        self._connected: bool | None = None
        self._startup_timeout_reported = False
        self._kill_reason_seen: str | None = None
        self._previous_state: NodeState | None = None  # pre-start halt fallback
        self._trades_day: date | None = None

        # Optional Telegram alerts (if env vars configured)
        self._telegram: TelegramBot | None = None
        try:
            from vibe_quant.alerts.telegram import TelegramBot as _TBot

            self._telegram = _TBot.from_env()
        except Exception:
            logger.info("Telegram alerts disabled (TELEGRAM_BOT_TOKEN/CHAT_ID not set)")

    @property
    def config(self) -> PaperTradingConfig:
        """Get node configuration."""
        return self._config

    @property
    def status(self) -> NodeStatus:
        """Get current node status."""
        return self._status

    @property
    def error_handler(self) -> ErrorHandler:
        """Get error handler."""
        return self._error_handler

    @property
    def guard(self) -> TradingGuard | None:
        """The trading guard (None before the trading node is built)."""
        return self._guard

    @property
    def resolved_config(self) -> ResolvedRunConfig | None:
        """Effective symbols/params/leverage (after ``_initialize``)."""
        return self._resolved

    # ------------------------------------------------------------- alerts

    def _on_error_halt(self, reason: str, message: str) -> None:
        """ErrorHandler decided an error is fatal for trading."""
        self.halt(HaltReason.ERROR, f"{reason}: {message}")

    def _on_error_alert(self, alert_type: str, context: ErrorContext) -> None:
        """ErrorHandler wants the operator to know about an error."""
        # oledr/k8oei: route through _write_then_alert so the alert goes out even when
        # the write raises; the write's own error then propagates unchanged.
        self._write_then_alert(
            EventType.RISK_CHECK,
            {
                "action": "alert",
                "alert_type": alert_type,
                "error_category": context.category.value,
                "error_type": type(context.error).__name__,
                "error_message": str(context.error),
                "operation": context.operation,
                "symbol": context.symbol,
                "retry_count": context.retry_count,
                "passed": False,
            },
            "error",
            f"{alert_type}: {context.error}\nOp: {context.operation}"
            + (f"\nSymbol: {context.symbol}" if context.symbol else ""),
        )

    def _send_telegram_alert(self, kind: str, message: str, *, critical: bool = False) -> None:
        """Send alert via Telegram if configured. Fire-and-forget, never raises.

        ``critical`` alerts (halts, kill switch, risk breaches) bypass the
        per-type rate limit so they are never silently dropped.
        """
        if self._telegram is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("telegram alert not sent (no running event loop): %s", kind)
            return
        tg = self._telegram
        if kind == "circuit_breaker":
            coro = tg.send_circuit_breaker(message, bypass_rate_limit=critical)
        else:
            coro = tg.send_error(message, bypass_rate_limit=critical)
        task = loop.create_task(coro)
        self._alert_tasks.add(task)
        task.add_done_callback(self._alert_done)

    def _alert_done(self, task: asyncio.Task[bool]) -> None:
        self._alert_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            # TelegramBot already redacts its own failures; never feed alert
            # failures back into ErrorHandler (that would alert about alerts).
            logger.warning("telegram alert task failed: %s", type(exc).__name__)

    def _write_then_alert(
        self,
        event_type: EventType,
        data: dict[str, object],
        kind: str,
        message: str,
        *,
        critical: bool = False,
        timestamp: datetime | None = None,
    ) -> None:
        """Write the event, then alert, even when the write itself raises (R1/irdig).

        Order stays write -> alert, but a persistent writer failure (disk full,
        permissions) must not also swallow the operator's warning, so the alert is
        sent with the same text and ``critical`` flag and only then does the write's
        own exception propagate unchanged: the caller keeps its retry semantics
        (R47b - the watcher fails the symbol's cycle and re-delivers the row).
        """
        try:
            self._write_event(event_type, data, timestamp=timestamp)
        except Exception as exc:
            # R1b: class only. ``logger.exception`` would print the active exception's
            # ``__context__`` chain too (a fetch error with signature/apiKey text).
            logger.error(
                "paper event write failed (%s); sending the alert anyway", type(exc).__name__
            )
            self._send_telegram_alert(kind, message, critical=critical)
            raise
        self._send_telegram_alert(kind, message, critical=critical)

    # ------------------------------------------------------ strategy setup

    def _load_strategy(self) -> StrategyDSL:
        """Load and validate strategy from database.

        Raises:
            ConfigurationError: If strategy not found or invalid.
        """
        strategy_id = self._config.strategy_id
        if strategy_id is None:
            raise ConfigurationError("strategy_id is required")

        strategy_data = self._state_manager.get_strategy(strategy_id)
        if strategy_data is None:
            raise ConfigurationError(f"Strategy {strategy_id} not found")

        dsl_config = strategy_data["dsl_config"]

        try:
            dsl = validate_strategy_dict(dsl_config)
        except Exception as e:
            raise ConfigurationError(f"Invalid strategy DSL: {e}") from e

        reject_context_indicators(dsl)
        return dsl

    def _compile_strategy(self, dsl: StrategyDSL) -> ModuleType:
        """Compile DSL to a live-loadable Python module."""
        try:
            return self._compiler.compile_to_module(dsl)
        except Exception as e:
            raise ConfigurationError(f"Strategy compilation failed: {e}") from e

    @staticmethod
    def _class_names(dsl: StrategyDSL) -> tuple[str, str]:
        class_name = "".join(word.capitalize() for word in dsl.name.split("_"))
        return f"{class_name}Strategy", f"{class_name}Config"

    def _resolve_run_config(self, module: ModuleType, dsl: StrategyDSL) -> ResolvedRunConfig:
        """Derive symbols/params/leverage; compare against the validated run.

        Raises:
            ConfigurationError: If the validation run is missing/incompatible.
        """
        from nautilus_trader.trading.config import StrategyConfig

        cfg = self._config
        _, config_cls_name = self._class_names(dsl)
        config_cls = getattr(module, config_cls_name, None)
        # Only fields the generated config declares itself; NT base fields
        # (strategy_id, order_id_tag, manage_stop, ...) are node-controlled.
        base_fields = set(StrategyConfig.__struct_fields__)
        fields: tuple[str, ...] = tuple(
            f for f in getattr(config_cls, "__struct_fields__", ()) if f not in base_fields
        )

        validated: dict[str, object] | None = None
        base_params: dict[str, object] = {}
        symbols = list(cfg.symbols)
        leverage_f = float(DEFAULT_VALIDATION_LEVERAGE)
        # The margin type is derived from the validated run's liquidation mode
        # (R39) and only falls back to CROSSED when there is nothing to derive it
        # from (no validation run at all).
        derived_margin_type = DEFAULT_MARGIN_TYPE

        if cfg.validation_run_id is not None:
            run = self._state_manager.get_backtest_run(cfg.validation_run_id)
            if run is None:
                raise ConfigurationError(f"Validation run {cfg.validation_run_id} not found")
            if run.get("run_mode") != "validation":
                raise ConfigurationError(
                    f"Run {cfg.validation_run_id} is a {run.get('run_mode')!r} run, "
                    "not a validation run"
                )
            if run.get("status") != "completed":
                raise ConfigurationError(
                    f"Validation run {cfg.validation_run_id} is {run.get('status')!r}, "
                    "not completed"
                )
            if run.get("strategy_id") != cfg.strategy_id:
                raise ConfigurationError(
                    f"Validation run {cfg.validation_run_id} is for strategy "
                    f"{run.get('strategy_id')}, not {cfg.strategy_id}"
                )
            run_symbols = [str(s) for s in run.get("symbols") or []]
            if not symbols:
                symbols = run_symbols
            raw = validated_run_strategy_params(run)
            # Post-only maker entries are validation-only for now: paper has had
            # no risk review for them, so never drop the param silently below.
            entry_order = raw.get("entry_order", "market")
            if entry_order != "market":
                raise ConfigurationError(
                    f"Validation run {cfg.validation_run_id} used "
                    f"entry_order={entry_order!r}; paper trading supports only "
                    "'market' entries (post-only maker entries need a risk review)"
                )
            # The fee schedule is venue-level: paper trades a real Binance
            # account whose commission comes from the exchange, so the
            # validated schedule is reported (never applied to the node or
            # forwarded to the strategy). An invalid schedule fails closed,
            # exactly like validation does.
            raw_params = run.get("parameters")
            run_params: dict[str, object] = raw_params if isinstance(raw_params, dict) else {}
            try:
                fee_schedule = FeeSchedule.from_params(run_params)
            except ValueError as exc:
                raise ConfigurationError(
                    f"Validation run {cfg.validation_run_id} has invalid fee schedule "
                    f"parameters: {exc}"
                ) from exc
            base_params = {
                k: v for k, v in raw.items() if k in fields and k not in _BACKTEST_ONLY_PARAMS
            }
            # Fee keys never reach the strategy config; name them explicitly so a
            # non-default schedule is not silently dropped from the log either.
            venue_fields = [k for k in ("fee_tier", "bnb_discount") if k in run_params]
            dropped = sorted((set(raw) - set(base_params)) | set(venue_fields))
            if dropped:
                logger.info(
                    "validation run %d params not applied to the live strategy "
                    "(venue-level or backtest-only): %s",
                    cfg.validation_run_id,
                    dropped,
                )
            liquidation = validated_run_liquidation(run)
            leverage_f = liquidation.leverage
            # R39: an isolated validation is the one mode that maps to ISOLATED;
            # cross and off both trade NT's backtest MARGIN account (cross).
            derived_margin_type = (
                "ISOLATED" if liquidation.mode == "isolated" else DEFAULT_MARGIN_TYPE
            )
            validated = {
                "validation_run_id": cfg.validation_run_id,
                "symbols": run_symbols,
                "strategy_params": dict(base_params),
                "leverage": leverage_f,
                "margin_type": derived_margin_type,
                # R43: whether validation simulated liquidation at all decides
                # whether paper's exchange liquidation is comparable to it.
                "liquidation": liquidation.mode,
                "fee_schedule": fee_schedule.to_notes(),
            }

        if not symbols:
            raise ConfigurationError("No symbols to trade")

        params = dict(base_params)
        if cfg.sizing.risk_per_trade is not None:
            params["risk_per_trade"] = float(cfg.sizing.risk_per_trade)
        if cfg.sizing.max_position_pct is not None:
            params["max_position_pct"] = float(cfg.sizing.max_position_pct)
        unknown = sorted(k for k in params if fields and k not in fields)
        if unknown:
            raise ConfigurationError(f"Strategy config has no fields {unknown}")

        if cfg.leverage is not None:
            leverage_f = float(validated_leverage_override(cfg.leverage))
        leverage = int(leverage_f)
        if leverage > cfg.sizing.max_leverage:
            raise ConfigurationError(
                f"leverage {leverage} exceeds max_leverage {cfg.sizing.max_leverage}"
            )

        resolved = ResolvedRunConfig(
            symbols=symbols,
            strategy_params=params,
            leverage=leverage,
            # An explicit config-file margin type overrides the derived one (R40);
            # None means "use the validated run's, or CROSSED without one" (R39).
            margin_type=cfg.margin_type if cfg.margin_type is not None else derived_margin_type,
            validated=validated,
        )
        if validated is not None:
            resolved.diff = self._config_diff(validated, resolved)
        return resolved

    @staticmethod
    def _config_diff(
        validated: dict[str, object], resolved: ResolvedRunConfig
    ) -> list[dict[str, object]]:
        diff: list[dict[str, object]] = []
        v_params = validated.get("strategy_params")
        v_params = v_params if isinstance(v_params, dict) else {}
        for key in sorted(set(v_params) | set(resolved.strategy_params)):
            v, p = v_params.get(key), resolved.strategy_params.get(key)
            if v != p:
                diff.append({"field": f"strategy_params.{key}", "validated": v, "paper": p})
        v_lev = validated.get("leverage")
        if v_lev != float(resolved.leverage):
            diff.append({"field": "leverage", "validated": v_lev, "paper": resolved.leverage})
        if validated.get("margin_type") != resolved.margin_type:
            diff.append(
                {
                    "field": "margin_type",
                    "validated": validated.get("margin_type"),
                    "paper": resolved.margin_type,
                }
            )
        if validated.get("symbols") != resolved.symbols:
            diff.append(
                {"field": "symbols", "validated": validated.get("symbols"), "paper": resolved.symbols}
            )
        # R43: validation that ran with liquidation off never exercised the
        # liquidation the venue will apply, so paper cannot claim parity - name it
        # (after symbols, before fee_schedule) instead of a silent exact match.
        if validated.get("liquidation") == "off":
            diff.append(
                {
                    "field": "liquidation",
                    "validated": "off",
                    "paper": f"exchange liquidation ({resolved.margin_type})",
                    "note": (
                        "validation did not simulate liquidation; paper liquidation "
                        "parity is unverified"
                    ),
                }
            )
        # A non-default validated fee schedule cannot be reproduced by paper: the
        # live Binance account charges the exchange's own commission. Surface it
        # so a completed BNB run never claims an exact match without fee evidence.
        fee = validated.get("fee_schedule")
        if isinstance(fee, dict) and not FeeSchedule.from_params(fee).is_default:
            diff.append(
                {
                    "field": "fee_schedule",
                    "validated": fee,
                    "paper": "exchange commission (Binance live fee model)",
                    "note": (
                        "live fees come from the exchange; the BNB/fee-tier "
                        "assumption is unverified"
                    ),
                }
            )
        return diff

    def _instantiate_strategies(self, module: ModuleType, dsl: StrategyDSL) -> list[object]:
        """Build one Strategy instance per symbol with the resolved parameters.

        Configs are decoded with ``StrategyConfig.parse`` (msgspec) exactly like
        ``ImportableStrategyConfig`` does in validation, so parameter types match.
        Each strategy claims its instrument's external orders/positions so a
        restart adopts the venue position instead of opening a second one.
        """
        strategy_cls_name, config_cls_name = self._class_names(dsl)
        strategy_cls = getattr(module, strategy_cls_name, None)
        config_cls = getattr(module, config_cls_name, None)
        if strategy_cls is None or config_cls is None:
            raise ConfigurationError(
                f"Compiled module missing {strategy_cls_name}/{config_cls_name}"
            )
        resolved = self._resolved
        if resolved is None:
            resolved = self._resolve_run_config(module, dsl)
            self._resolved = resolved

        strategies: list[object] = []
        for idx, symbol in enumerate(resolved.symbols):
            instrument_id = f"{symbol}-PERP.BINANCE"
            raw_cfg = {
                **resolved.strategy_params,
                "instrument_id": instrument_id,
                "order_id_tag": f"{idx:03d}",
                "external_order_claims": [instrument_id],
            }
            try:
                cfg = config_cls.parse(json.dumps(raw_cfg))
            except Exception as e:
                raise ConfigurationError(f"Invalid strategy config {raw_cfg}: {e}") from e
            # Second gate, on the EFFECTIVE config: a strategy launched by
            # strategy_id alone skips the validation branch above and would
            # otherwise trade the compiled default. Post-only maker entries
            # have had no risk review for paper, so refuse them here too.
            effective_entry_order = getattr(cfg, "entry_order", "market")
            if effective_entry_order != "market":
                raise ConfigurationError(
                    f"Effective strategy config used entry_order={effective_entry_order!r}; "
                    "paper trading supports only 'market' entries (post-only maker "
                    "entries need a risk review)"
                )
            strategies.append(strategy_cls(config=cfg))
        return strategies

    def _create_live_trading_node(self) -> _TradingNodeLifecycle:
        """Create and build a NautilusTrader TradingNode instance (+ guard actor).

        Raises:
            ConfigurationError: If NautilusTrader or Binance adapter
                dependencies are not installed or incompatible.
        """
        try:
            from nautilus_trader.adapters.binance import config as binance_config
            from nautilus_trader.adapters.binance.common.enums import BinanceEnvironment
            from nautilus_trader.adapters.binance.common.symbol import BinanceSymbol
            from nautilus_trader.adapters.binance.config import (
                BinanceDataClientConfig,
                BinanceExecClientConfig,
            )
            from nautilus_trader.adapters.binance.factories import (
                BinanceLiveDataClientFactory,
                BinanceLiveExecClientFactory,
            )
            from nautilus_trader.adapters.binance.futures.enums import (
                BinanceFuturesMarginType,
            )
            from nautilus_trader.config import InstrumentProviderConfig
            from nautilus_trader.live.config import TradingNodeConfig
            from nautilus_trader.live.node import TradingNode
            from nautilus_trader.model.identifiers import InstrumentId
        except ImportError as e:
            raise ConfigurationError(
                f"NautilusTrader live trading dependencies not installed: {e}. "
                "Install nautilus_trader with Binance adapter support."
            ) from e

        resolved = self._resolved
        if resolved is None:
            raise ConfigurationError("run config not resolved before building the node")

        account_type_cls = getattr(binance_config, "BinanceAccountType", None)
        if account_type_cls is None:
            raise ConfigurationError(
                "BinanceAccountType is unavailable in NautilusTrader config module"
            )

        try:
            account_type = account_type_cls(self._config.binance.account_type)
        except ValueError as e:
            raise ConfigurationError(
                f"Unsupported Binance account_type '{self._config.binance.account_type}'",
            ) from e

        # Without `instrument_provider` the Binance adapter logs
        # "No loading configured" and the ExecEngine never reaches a
        # connected state. Scope loading to the configured symbols.
        load_ids = frozenset(
            InstrumentId.from_str(f"{symbol}-PERP.BINANCE") for symbol in resolved.symbols
        )
        instrument_provider = InstrumentProviderConfig(load_ids=load_ids)

        environment = (
            BinanceEnvironment.TESTNET if self._config.binance.testnet else BinanceEnvironment.LIVE
        )
        margin_type = (
            BinanceFuturesMarginType.CROSS
            if resolved.margin_type == "CROSSED"
            else BinanceFuturesMarginType.ISOLATED
        )

        node_config = TradingNodeConfig(
            trader_id=self._config.trader_id,
            data_clients={
                "BINANCE": BinanceDataClientConfig(
                    api_key=self._config.binance.api_key,
                    api_secret=self._config.binance.api_secret,
                    account_type=account_type,
                    environment=environment,
                    instrument_provider=instrument_provider,
                ),
            },
            exec_clients={
                "BINANCE": BinanceExecClientConfig(
                    api_key=self._config.binance.api_key,
                    api_secret=self._config.binance.api_secret,
                    account_type=account_type,
                    environment=environment,
                    instrument_provider=instrument_provider,
                    futures_leverages={
                        BinanceSymbol(s): resolved.leverage for s in resolved.symbols
                    },
                    futures_margin_types={BinanceSymbol(s): margin_type for s in resolved.symbols},
                ),
            },
        )
        node = TradingNode(node_config)
        self._guard = self._build_guard(node.kernel.risk_engine)
        # Actors start before strategies, so the gate is installed before any
        # strategy can submit an order.
        node.trader.add_actor(self._guard)
        # Strategies MUST be registered on the trader before ``node.build()``
        # or the live engine starts with zero strategies and exits early.
        for strategy in self._compiled_strategies:
            node.trader.add_strategy(strategy)
        node.add_data_client_factory("BINANCE", BinanceLiveDataClientFactory)
        node.add_exec_client_factory("BINANCE", BinanceLiveExecClientFactory)
        node.build()
        return node

    def _build_guard(self, risk_engine: Any) -> TradingGuard:
        risk = self._config.risk
        return TradingGuard(
            TradingGuardConfig(
                max_drawdown_pct=risk.max_drawdown_pct,
                max_daily_loss_pct=risk.max_daily_loss_pct,
                max_consecutive_losses=risk.max_consecutive_losses,
                max_position_count=risk.max_position_count,
            ),
            strategies=self._compiled_strategies,
            risk_engine=risk_engine,
            listener=_GuardBridge(self),
            risk_state=self._restored_risk,
        )

    # ------------------------------------------------------------ control

    def _setup_signal_handlers(self) -> None:
        """POSIX fallback controls for CLI use (the API uses the command queue).

        SIGINT/SIGTERM are deliberately NOT registered here — Nautilus
        installs its own graceful-shutdown handlers on the running loop.

        SIGUSR1 → halt (flatten, cancel, stop strategies) — one signal, full halt
        SIGUSR2 → resume (refused while the kill switch is engaged)
        """
        loop = asyncio.get_running_loop()

        def halt_handler(_sig: int) -> None:
            self.halt(HaltReason.SIGNAL, "halt via SIGUSR1")

        def resume_handler(_sig: int) -> None:
            ok, msg = self.resume()
            if not ok:
                logger.warning("resume via SIGUSR2 refused: %s", msg)

        loop.add_signal_handler(signal.SIGUSR1, halt_handler, signal.SIGUSR1)
        loop.add_signal_handler(signal.SIGUSR2, resume_handler, signal.SIGUSR2)

    def _kill_switch(self) -> tuple[bool, str | None]:
        try:
            sys_state = self._state_manager.get_system_state()
        except Exception:
            logger.exception("could not read system kill switch")
            return False, None
        return bool(sys_state.get("kill_switch")), sys_state.get("reason")

    async def _control_loop(self) -> None:
        """Poll operator commands, the kill switch and connectivity."""
        interval = self._config.control_poll_interval
        while not self._shutdown_event.is_set():
            try:
                await self.process_commands()
                self._check_kill_switch()
                self._check_connectivity()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("paper control loop iteration failed")
            await asyncio.sleep(interval)

    def _check_kill_switch(self) -> None:
        engaged, reason = self._kill_switch()
        if not engaged:
            self._kill_reason_seen = None
            return
        guard = self._guard
        already = (
            guard is not None
            and guard.guard_state == GuardState.HALTED
            and guard.halt_reason == HaltReason.KILL_SWITCH
        )
        if not already:
            self.halt(HaltReason.KILL_SWITCH, f"system kill switch: {reason or 'no reason'}")
        self._kill_reason_seen = reason

    def _check_connectivity(self) -> None:
        kernel = getattr(self._trading_node, "kernel", None)
        connected = False
        if kernel is not None:
            try:
                connected = bool(
                    kernel.data_engine.check_connected() and kernel.exec_engine.check_connected()
                )
            except Exception as exc:
                # s3pqx: class only - the probe can run while a fetch error (signed URL,
                # api key) is the active exception, and ``logger.exception`` would print
                # its ``__context__`` chain. Same rule as _write_then_alert (R1b).
                logger.error("connectivity check failed (%s)", type(exc).__name__)
                if self._connected is None:
                    # R42e: a stuck connect can break the probe itself; that is still
                    # "not connected" for the startup timeout.
                    self._check_startup_connect_timeout()
                return
        if self._connected is None:
            if not connected:
                # R42b/R42e: the probe ran first, so only a venue that is still not
                # connected is reported (a stalled loop that finds it up stays silent).
                self._check_startup_connect_timeout()
                return
            self._connected = True  # first time fully connected
            if self._startup_timeout_reported:
                # R42c: the venue set up late; say so once, without a critical alert.
                if self._status.error_message == self._startup_timeout_message():
                    # R42d: clear only our own message, never a later halt message.
                    self._status.error_message = None
                    self._status.updated_at = self._now()
                # oledr/k8oei: the same helper, so a failing write cannot drop the
                # only signal that the venue came up late.
                self._write_then_alert(
                    EventType.RISK_CHECK,
                    {"action": "venue_connected_after_startup_timeout", "passed": True},
                    "error",
                    f"paper {self._config.trader_id} venue connected after startup timeout",
                )
            return
        if kernel is None:
            return
        if self._connected and not connected:
            self._connected = False
            self.handle_error(
                ConnectionError("venue connection lost (data or execution client disconnected)"),
                operation="connectivity",
            )
        elif not self._connected and connected:
            self._connected = True
            self._error_handler.reset_retry_count("connectivity", "")
            self._write_then_alert(
                EventType.SIGNAL, {"action": "reconnected"}, "error", "venue connection restored"
            )

    def _startup_timeout_message(self) -> str:
        return (
            f"paper {self._config.trader_id} not connected after {STARTUP_CONNECT_TIMEOUT_S} s; "
            "venue setup refused (e.g. margin type change with open orders or position); "
            "strategy NOT trading"
        )

    def _check_startup_connect_timeout(self) -> None:
        """R42b: report ONCE when the venue has not connected after the startup timeout.

        An exchange that refuses the margin change (open orders or a position) leaves
        NT connecting, so the node reports the refusal and stays silent until it can
        say something new. Report only: no strategy is started and no order is placed.
        """
        if self._startup_timeout_reported:
            return
        started_at = self._status.started_at
        if started_at is None:
            return
        if (self._now() - started_at).total_seconds() <= STARTUP_CONNECT_TIMEOUT_S:
            return
        self._startup_timeout_reported = True
        message = self._startup_timeout_message()
        if not self._status.error_message:
            # R42d: never overwrite an existing (halt) message.
            self._status.error_message = message
            self._status.updated_at = self._now()
        # R1c: the alert goes out even if the write raises (the latch is already set, so
        # there is no retry); the write failure is logged by class and never escapes,
        # which keeps it from chaining onto a connectivity-probe exception.
        # Critical: bypass the per-type rate limit, this venue never finished setup.
        with contextlib.suppress(Exception):
            self._write_then_alert(
                EventType.RISK_CHECK,
                {
                    "action": "startup_connect_timeout",
                    "timeout_s": STARTUP_CONNECT_TIMEOUT_S,
                    "margin_type": (
                        self._resolved.margin_type if self._resolved else DEFAULT_MARGIN_TYPE
                    ),
                    "passed": False,
                },
                "error",
                message,
                critical=True,
            )

    # ----------------------------------------------- liquidation watch (T6)

    def _now_ms(self) -> int:
        """Wall clock in epoch milliseconds (R50: one seam for the watch cursors)."""
        return int(self._now().timestamp() * 1_000)

    @staticmethod
    def _checkpoint_ms(timestamp: datetime) -> int:
        """The epoch milliseconds of a checkpoint row's timestamp (R50b).

        ``StatePersistence`` stores ``datetime('now')``, i.e. NAIVE UTC text
        (persistence.py:39), so the timestamp is taken as UTC and the distance to the
        epoch is computed in whole seconds: reading a naive timestamp through
        ``.timestamp()`` would shift every restored cursor by the host's UTC offset.
        """
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        delta = timestamp.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
        return int(delta.total_seconds()) * 1_000

    def _start_liquidation_task(self) -> None:
        """Start the report-only watch task beside the control loop (R47, R53)."""
        self._liquidation_stop = asyncio.Event()
        self._liquidation_task = asyncio.create_task(self._liquidation_loop())

    async def _stop_liquidation_loop(self) -> None:
        """R53: set the stop event, cancel the task, never hang and never raise."""
        task = self._liquidation_task
        self._liquidation_task = None
        self._liquidation_stop.set()
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    async def _liquidation_loop(self) -> None:
        """Poll forceOrders until shutdown (R47-R53, REPORT ONLY).

        A failure here can lose an alert but never move a position, so it is logged and
        ends the task instead of stopping the node: the watcher holds no reference to the
        strategy, the guard or the exec engine (R49).
        """
        try:
            watcher = self._build_liquidation_watcher()
            await watcher.run(self._liquidation_stop)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # R4 (k8oei): class only. ``logger.exception`` would print the active
            # exception's ``__context__`` chain (a fetch error with signature/apiKey text).
            logger.error(
                "liquidation watch task failed (%s); paper trading continues", type(exc).__name__
            )
            # R52g: a dead detector must not be silent. Only the exception class is
            # reported: its text can carry request URLs or credentials.
            message = f"liquidation watch failed: {type(exc).__name__}"
            with contextlib.suppress(Exception):
                # R1/irdig: the alert goes out even if the write raises; the write's
                # own error is already logged, and this terminal handler never raises.
                self._write_then_alert(
                    EventType.RISK_CHECK,
                    {
                        "action": "liquidation_watch_failed",
                        "message": message,
                        "passed": False,
                    },
                    "error",
                    f"{message}; exchange liquidations are NOT being reported",
                    critical=True,
                )

    def _build_liquidation_watcher(self) -> LiquidationWatcher:
        """Build the watcher from the node's own Binance config and checkpoint (R50).

        Raises:
            ConfigurationError: If the run config was not resolved.
        """
        resolved = self._resolved
        if resolved is None:
            raise ConfigurationError("run config not resolved before the liquidation watcher")
        binance = self._config.binance
        fetch = build_binance_fetch(
            binance.api_key, binance.api_secret, binance.account_type, binance.testnet
        )
        cursors: dict[str, int] | None = None
        seen: dict[str, int] | None = None
        restored = self._restored_watch
        if restored is not None:
            # R50b/R50c: the checkpoint's own cursor per traded symbol; one it does not
            # name starts at the checkpoint itself, which covers the downtime gap.
            cursors = {
                symbol: restored["cursors"].get(symbol, self._restored_watch_ms)
                for symbol in resolved.symbols
            }
            seen = restored["seen"]
        watcher = LiquidationWatcher(
            fetch,
            resolved.symbols,
            on_liquidation=self._on_exchange_liquidation,
            on_warning=self._on_liquidation_warning,
            on_recovered=self._on_liquidation_recovered,
            now_ms=self._now_ms,
            cursors=cursors,
            seen=seen,
        )
        self._liquidation_watcher = watcher
        return watcher

    def _unstarted_watch_state(self) -> dict[str, dict[str, int]]:
        """The restored watch state as it stands before a watcher exists (R50d).

        Traded symbols the stored state omits get the checkpoint timestamp, the same
        cursor ``_build_liquidation_watcher`` would give them (R50c).
        """
        restored = self._restored_watch
        assert restored is not None
        cursors = dict(restored["cursors"])
        if self._resolved is not None:
            for symbol in self._resolved.symbols:
                cursors.setdefault(symbol, self._restored_watch_ms)
        return {"cursors": cursors, "seen": dict(restored["seen"])}

    async def _on_exchange_liquidation(self, order: ForceOrder) -> None:
        """Report one exchange-initiated close: one event, one CRITICAL alert (R47).

        R47b: a missing or failing event writer RAISES, so the watcher fails that
        symbol's cycle (its cursor stays put) and delivers the row on the next one.
        R1/irdig: a failing write does not cost the alert - it is sent, with this
        same text, before the writer's own exception is re-raised (a duplicate alert
        on the retry is accepted). Nothing is swallowed here and nothing is ordered:
        paper cannot reproduce the venue's close, it can only say that it happened.
        """
        if self._event_writer is None:
            raise RuntimeError("cannot report an exchange liquidation without an event writer")
        margin_type = (
            self._resolved.margin_type if self._resolved is not None else DEFAULT_MARGIN_TYPE
        )
        # R47e: the event's own vocabulary ('isolated'/'cross'), not the venue's
        # 'ISOLATED'/'CROSSED' margin type.
        mode = "isolated" if margin_type == "ISOLATED" else "cross"
        self._write_then_alert(
            EventType.LIQUIDATION,
            {
                "source": "binance_force_orders",
                "position_id": f"{order.symbol}-PERP.BINANCE",
                "symbol": order.symbol,
                "side": order.side,
                "auto_close_type": order.auto_close_type,
                "liquidation_price": order.avg_price,
                "quantity": order.executed_qty,
                # R47: never a fake 0 - the venue's close carries no PnL we can trust.
                "loss": None,
                "ts_ns": int(order.ts_ns),
                "mode": mode,
                "venue_order_id": str(order.order_id),
                "client_order_id": order.client_order_id,
                "status": order.status,
            },
            "error",
            f"EXCHANGE {order.auto_close_type} {order.symbol} {order.side} "
            f"qty {order.executed_qty} @ {order.avg_price} (order {order.order_id}); "
            "paper position state may be stale; NO paper action taken",
            critical=True,
            timestamp=_ns_to_dt(order.ts_ns),
        )

    async def _on_liquidation_warning(self, message: str) -> None:
        """R52: a degraded or clamped detector warns without a critical alert."""
        self._write_then_alert(
            EventType.RISK_CHECK,
            {"action": "liquidation_watch_degraded", "message": message, "passed": False},
            "error",
            f"WARNING: {message}",
        )

    async def _on_liquidation_recovered(self, symbol: str, failures: int) -> None:
        """R52: the first success after a degraded run is reported once."""
        self._write_then_alert(
            EventType.RISK_CHECK,
            {
                "action": "liquidation_watch_recovered",
                "symbol": symbol,
                "failures": failures,
                "passed": True,
            },
            "error",
            f"liquidation detector recovered: {symbol} after {failures} failures",
        )

    @staticmethod
    def _restored_watch_state(value: object) -> dict[str, dict[str, int]]:
        """Read ``node_status["liquidation_watch"]`` from a prior checkpoint (R50).

        A missing or malformed payload is an empty state rather than an error: every
        traded symbol then starts at the checkpoint timestamp (R50b), which only ever
        costs a duplicate alert.
        """
        payload = value if isinstance(value, dict) else {}
        return {
            "cursors": _int_map(payload.get("cursors")),
            "seen": _int_map(payload.get("seen")),
        }

    async def process_commands(self) -> None:
        """Execute queued operator commands for this trader."""
        if self._commands is None:
            return
        for cmd in self._commands.claim_pending(self._config.trader_id):
            try:
                ok, result, error = await self._execute_command(cmd)
            except Exception as exc:
                logger.exception("paper command %s failed", cmd.command)
                self._commands.fail(cmd.id, f"{type(exc).__name__}: {exc}")
                continue
            if ok:
                self._commands.complete(cmd.id, result)
            else:
                self._commands.fail(cmd.id, error or "failed", result)

    async def _execute_command(self, cmd: PaperCommand) -> tuple[bool, dict[str, Any], str | None]:
        payload = cmd.payload or {}
        message = str(payload.get("message") or f"{cmd.command} via API")
        if cmd.command == "halt":
            self.halt(HaltReason.MANUAL, message)
            return True, self._state_payload(), None
        if cmd.command == "kill":
            self.halt(HaltReason.KILL_SWITCH, message)
            return True, self._state_payload(), None
        if cmd.command == "pause":
            self.pause()
            return True, self._state_payload(), None
        if cmd.command == "resume":
            ok, msg = self.resume()
            return ok, {**self._state_payload(), "message": msg}, None if ok else msg
        if cmd.command == "close_all":
            wait = float(payload.get("wait_secs", 10.0))
            result = self.close_all_positions()
            still_open = await self._wait_flat(wait)
            out: dict[str, Any] = {**result.to_dict(), "still_open": still_open}
            errors = list(result.errors)
            if still_open:
                errors.append(f"positions still open after {wait:.0f}s: {still_open}")
            if errors:
                return False, out, "; ".join(errors)
            return True, out, None
        return False, {}, f"unknown command {cmd.command!r}"

    def _state_payload(self) -> dict[str, Any]:
        return {
            "state": self._status.state.value,
            "halt_reason": self._status.halt_reason.value if self._status.halt_reason else None,
            "message": self._status.error_message,
        }

    def _open_position_ids(self) -> list[str]:
        cache = getattr(self._trading_node, "cache", None)
        if cache is None:
            return []
        return [str(p.id) for p in cache.positions_open()]

    async def _wait_flat(self, timeout: float) -> list[str]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            still = self._open_position_ids()
            if not still or loop.time() >= deadline:
                return still
            await asyncio.sleep(0.25)

    # ------------------------------------------------- guard callbacks

    def _on_guard_state_change(
        self,
        state: GuardState,
        reason: HaltReason | None,
        message: str,
        metrics: dict[str, object],
    ) -> None:
        prev = self._status.state
        self._status.state = _GUARD_TO_NODE_STATE[state]
        self._status.halt_reason = reason if state == GuardState.HALTED else None
        self._status.error_message = message
        self._status.updated_at = datetime.now(UTC)
        try:
            if state == GuardState.HALTED and reason is not None:
                self._write_then_alert(
                    EventType.RISK_CHECK,
                    {
                        "action": "halt",
                        "check_type": reason.value,
                        "reason": reason.value,
                        "message": message,
                        "passed": False,
                        **metrics,
                    },
                    "circuit_breaker",
                    f"HALT ({reason.value}): {message}",
                    critical=True,
                )
            else:
                payload: dict[str, object] = {
                    "action": "state_change",
                    "from_state": prev.value,
                    "to_state": self._status.state.value,
                    "message": message,
                }
                # oledr/k8oei: only the RESUMED branch alerts today, so only it goes
                # through the helper; a plain state change still writes without alerting.
                if prev == NodeState.HALTED and state == GuardState.RUNNING:
                    self._write_then_alert(
                        EventType.SIGNAL,
                        payload,
                        "circuit_breaker",
                        f"RESUMED: {message}",
                        critical=True,
                    )
                else:
                    self._write_event(EventType.SIGNAL, payload)
        finally:
            # bykze: the transition is checkpointed even when the event write raises. The
            # status was updated above, so the checkpoint carries the new state/the halt
            # reason and a disk-full writer cannot cost the restart its halt. The helper
            # swallows its own failure, so it never masks the write's error.
            self._save_checkpoint_now()

    def _on_order_rejected(self, event: Any) -> None:
        self._write_event(
            EventType.ORDER,
            {
                "action": "rejected",
                "order_id": str(event.client_order_id),
                "symbol": str(event.instrument_id),
                "strategy_id": str(event.strategy_id),
                "reason": str(event.reason),
            },
            timestamp=_ns_to_dt(event.ts_event),
        )
        self.handle_error(
            OrderRejectedError(f"order rejected: {event.reason}"),
            operation=f"submit_order {event.client_order_id}",
            symbol=str(event.instrument_id),
        )

    def _on_order_denied(self, event: Any, *, by_gate: bool) -> None:
        payload: dict[str, object] = {
            "action": "denied",
            "by_guard": by_gate,
            "order_id": str(event.client_order_id),
            "symbol": str(event.instrument_id),
            "strategy_id": str(event.strategy_id),
            "reason": str(event.reason),
        }
        timestamp = _ns_to_dt(event.ts_init)
        # oledr/k8oei: only a risk-engine denial alerts today, so only it goes through
        # the helper; a gate denial still writes without alerting.
        if not by_gate:
            self._write_then_alert(
                EventType.ORDER,
                payload,
                "error",
                f"Order denied by risk engine: {event.reason}\n{event.instrument_id}",
                timestamp=timestamp,
            )
        else:
            self._write_event(EventType.ORDER, payload, timestamp=timestamp)

    def _on_position_opened(self, event: Any) -> None:
        self._write_event(
            EventType.POSITION_OPEN,
            {
                "position_id": str(event.position_id),
                "symbol": str(event.instrument_id),
                "side": event.side.name,
                "quantity": float(event.quantity),
                "entry_price": float(event.avg_px_open),
                "leverage": self._resolved.leverage if self._resolved else 1,
            },
            timestamp=_ns_to_dt(event.ts_event),
        )
        self._emit_liquidation_estimate(event)

    def _emit_liquidation_estimate(self, event: Any) -> None:
        """R44/R45/R45b: report the estimated liquidation price of a newly opened position.

        Report only: never trades, never halts and never raises (it runs inside the NT
        position callback, so a raise would also skip the guard's risk evaluation). An
        ISOLATED position is priced with the shared ``vibe_quant.liquidation`` rule at
        its entry margin (notional / leverage, so the number is a pure estimate); a
        CROSSED one depends on the whole account wallet. Any failure, expected or not,
        is reported as a null price plus a reason.
        """
        resolved = self._resolved
        payload: dict[str, object] = {
            "action": "liquidation_estimate",
            "position_id": None,
            "symbol": None,
            "side": None,
            "quantity": None,
            "entry_price": None,
            "leverage": resolved.leverage if resolved else 1,
            "margin_type": resolved.margin_type if resolved else DEFAULT_MARGIN_TYPE,
            "isolated_margin": None,
            "est_liquidation_price": None,
            "tier_table_version": None,
            "basis": None,
            "passed": True,
        }
        timestamp: datetime | None = None
        try:
            timestamp = _ns_to_dt(event.ts_event)
            self._fill_liquidation_estimate(payload, event)
        except Exception as exc:
            # Domain errors (tier table, insolvent) are expected; anything else is a bug.
            if not _is_tier_error(exc):
                logger.exception("liquidation estimate failed")
            payload["est_liquidation_price"] = None
            payload["reason"] = str(exc) or type(exc).__name__
        try:
            self._write_event(EventType.RISK_CHECK, payload, timestamp=timestamp)
        except Exception:
            logger.exception("liquidation estimate could not be written")

    def _fill_liquidation_estimate(self, payload: dict[str, object], event: Any) -> None:
        leverage = payload["leverage"]
        assert isinstance(leverage, int)
        symbol = str(event.instrument_id).split(".")[0].removesuffix("-PERP")
        side = event.side.name  # PositionSide: LONG / SHORT (never the entry OrderSide)
        quantity = float(event.quantity)
        entry_price = float(event.avg_px_open)
        payload.update(
            position_id=str(event.position_id),
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
        )
        if payload["margin_type"] != "ISOLATED":
            payload["reason"] = _CROSS_LP_REASON
            return
        isolated_margin = quantity * entry_price / leverage
        payload["isolated_margin"] = isolated_margin
        # Imported here on purpose: a run that trades no isolated position never
        # loads the liquidation package.
        from vibe_quant.liquidation.tiers import isolated_liq_price, load_tier_tables

        tables = load_tier_tables()
        payload["tier_table_version"] = tables.version
        payload["est_liquidation_price"] = isolated_liq_price(
            _POSITION_SIDE_SIGNS[side],
            quantity,
            entry_price,
            isolated_margin,
            tables[symbol],
        )
        payload["basis"] = _ISOLATED_LP_BASIS.format(version=tables.version)

    def _now(self) -> datetime:
        """Current UTC time (one seam for the time-based checks)."""
        return datetime.now(UTC)

    def _on_position_closed(self, event: Any) -> None:
        pnl = event.realized_pnl
        net = float(pnl) if pnl is not None else 0.0
        commissions = 0.0
        exit_reason = "unknown"
        self._count_closed_trade()
        cache = self._guard.cache if self._guard is not None else None
        if cache is not None:
            pos = cache.position(event.position_id)
            if pos is not None and pnl is not None:
                # NT realized_pnl is net of commissions; gross adds them back.
                commissions = sum(float(c) for c in pos.commissions() if c.currency == pnl.currency)
            closing = cache.order(event.closing_order_id) if event.closing_order_id else None
            exit_reason = self._exit_reason(closing)
        self._write_event(
            EventType.POSITION_CLOSE,
            {
                "position_id": str(event.position_id),
                "symbol": str(event.instrument_id),
                "exit_price": float(event.avg_px_close),
                "gross_pnl": net + commissions,
                "net_pnl": net,
                "exit_reason": exit_reason,
            },
            timestamp=_ns_to_dt(event.ts_event),
        )

    def _count_closed_trade(self) -> None:
        today = datetime.now(UTC).date()
        if self._trades_day != today:
            self._trades_day = today
            self._status.trades_today = 0
        self._status.trades_today += 1

    @staticmethod
    def _exit_reason(order: Any) -> str:
        if order is None:
            return "unknown"
        tags = order.tags or []
        if "MARKET_EXIT" in tags:
            return "halt_or_close_all"
        type_name = order.order_type.name
        if order.is_reduce_only and type_name in ("STOP_MARKET", "STOP_LIMIT"):
            return "stop_loss"
        if order.is_reduce_only and type_name == "LIMIT":
            return "take_profit"
        return "signal"

    def _on_guard_warning(self, message: str) -> None:
        # R2 (oledr/k8oei): the warning survives a failing write, and the write's own
        # error still propagates, like every other write-then-alert pair.
        self._write_then_alert(
            EventType.RISK_CHECK,
            {"action": "warning", "message": message},
            "error",
            f"WARNING: {message}",
        )

    # ------------------------------------------------------ checkpoints

    def _capture_checkpoint(self) -> StateCheckpoint:
        """Capture current node state for persistence.

        Reads open positions/orders and per-account balances off the
        TradingNode cache so a crash-restart can reconcile against the
        live venue. Before ``node.build()`` and after teardown the cache
        is unavailable — in those cases we emit an empty snapshot rather
        than propagate an exception that would kill the periodic
        checkpoint loop.
        """
        positions: dict[str, object] = {}
        orders: dict[str, object] = {}
        balance: dict[str, object] = {}

        cache = getattr(self._trading_node, "cache", None)
        portfolio = getattr(self._trading_node, "portfolio", None)
        if cache is not None:
            with contextlib.suppress(Exception):
                for pos in cache.positions_open():
                    data = pos.to_dict()
                    data["unrealized_pnl"] = self._unrealized_pnl(portfolio, pos)
                    data["leverage"] = self._resolved.leverage if self._resolved else None
                    positions[pos.id.value] = data
            with contextlib.suppress(Exception):
                for order in cache.orders_open():
                    orders[order.client_order_id.value] = order.to_dict()
            with contextlib.suppress(Exception):
                for account in cache.accounts():
                    balance[account.id.value] = self._serialize_account_balance(account)
            self._status.positions = len(positions)

        node_status: dict[str, object] = dict(self._status.to_dict())
        node_status["trader_id"] = self._config.trader_id
        node_status["testnet"] = self._config.binance.testnet
        if self._guard is not None:
            guard_status = self._guard.status()
            node_status["risk"] = guard_status["risk"]
            node_status["guard"] = guard_status
        if self._resolved is not None:
            node_status["config_diff"] = self._resolved.diff
            node_status["validation_run_id"] = self._config.validation_run_id
            node_status["leverage"] = self._resolved.leverage
        if self._liquidation_watcher is not None:
            # R50/R52b: per-symbol cursors and the keys delivered so far. This periodic
            # checkpoint is what bounds re-alerting after a crash.
            node_status["liquidation_watch"] = self._liquidation_watcher.state()
        elif self._restored_watch is not None:
            # R50d: a checkpoint written while no watcher runs (the one `_initialize`
            # saves, an early-failure shutdown, a failed watcher) must not drop the
            # restored cursors and `seen`, or the downtime gap is never rescanned.
            node_status["liquidation_watch"] = self._unstarted_watch_state()

        return StateCheckpoint(
            trader_id=self._config.trader_id,
            positions=positions,
            orders=orders,
            balance=balance,
            node_status=node_status,
        )

    @staticmethod
    def _unrealized_pnl(portfolio: Any, pos: Any) -> float | None:
        if portfolio is None:
            return None
        try:
            pnl = portfolio.unrealized_pnl(pos.instrument_id)
        except Exception:
            return None
        return float(pnl) if pnl is not None else None

    @staticmethod
    def _serialize_account_balance(account: object) -> dict[str, dict[str, str]]:
        """Serialize AccountBalance data to a JSON-friendly shape.

        Returns ``{"total": {CCY: amount_str}, "free": {...}, "locked": {...}}``.
        """

        def _as_currency_amounts(balances: object) -> dict[str, str]:
            out: dict[str, str] = {}
            if balances is None:
                return out
            items = balances.items() if hasattr(balances, "items") else []
            for currency, money in items:
                code = getattr(currency, "code", str(currency))
                amount = money.as_decimal() if hasattr(money, "as_decimal") else money
                out[code] = str(amount)
            return out

        return {
            "total": _as_currency_amounts(getattr(account, "balances_total", lambda: None)()),
            "free": _as_currency_amounts(getattr(account, "balances_free", lambda: None)()),
            "locked": _as_currency_amounts(getattr(account, "balances_locked", lambda: None)()),
        }

    def _save_checkpoint_now(self) -> None:
        if self._persistence is None:
            return
        try:
            self._persistence.save_checkpoint(self._capture_checkpoint())
        except Exception as exc:
            # s3pqx: class only - this now always runs in a finally after a failed
            # guard-transition write (bykze), so an active exception is present and
            # ``logger.exception`` would print that error and its ``__context__`` chain
            # next to this one. Same rule as _check_connectivity / _write_then_alert.
            logger.error("immediate checkpoint failed (%s)", type(exc).__name__)

    def _restore_from_checkpoint(self) -> None:
        """Load the latest checkpoint for this trader_id (restart / /restore).

        Restores the risk bookkeeping (high water mark, UTC-day baseline,
        consecutive losses) so a restart cannot reset the drawdown/daily-loss
        limits. Positions and orders come from NautilusTrader's venue
        reconciliation (claimed via ``external_order_claims``); the checkpoint
        copy is only logged for comparison.
        """
        if self._persistence is None:
            return
        prior = self._persistence.load_latest_checkpoint(self._config.trader_id)
        if prior is None:
            return
        # R50b/R50c: any checkpoint restores the watch state; the ones written before
        # the watcher existed still cover the downtime gap through their timestamp.
        self._restored_watch_ms = self._checkpoint_ms(prior.timestamp)
        self._restored_watch = self._restored_watch_state(
            prior.node_status.get("liquidation_watch")
        )
        risk_raw = prior.node_status.get("risk") if isinstance(prior.node_status, dict) else None
        self._restored_risk = RiskState.from_dict(risk_raw if isinstance(risk_raw, dict) else None)
        self._write_event(
            EventType.SIGNAL,
            {
                "action": "restore",
                "checkpoint_time": prior.timestamp.isoformat(),
                "prior_state": prior.node_status.get("state"),
                "prior_halt_reason": prior.node_status.get("halt_reason"),
                "prior_positions": sorted(prior.positions),
                "prior_orders": sorted(prior.orders),
                "restored_risk": self._restored_risk.to_dict(),
            },
        )
        logger.info(
            "restored risk state for %s from checkpoint %s: %s",
            self._config.trader_id,
            prior.timestamp.isoformat(),
            self._restored_risk.to_dict(),
        )

    # -------------------------------------------------------------- events

    def _write_event(
        self,
        event_type: EventType,
        data: dict[str, object],
        timestamp: datetime | None = None,
    ) -> None:
        """Write an event to the paper event log, preserving the full payload.

        Position events use the typed classes (their fields are the payload);
        everything else is written as a base ``Event`` because the typed
        SIGNAL/RISK_CHECK classes drop unknown keys (and default ``passed``
        to True, which would record a halt as a passed risk check).
        """
        if self._event_writer is None:
            return
        strategy_name = self._strategy.name if self._strategy else "unknown"
        ts = timestamp or datetime.now(UTC)
        event: Event
        if event_type in (EventType.POSITION_OPEN, EventType.POSITION_CLOSE):
            event = create_event(
                event_type=event_type,
                run_id=self._config.trader_id,
                strategy_name=strategy_name,
                data=data,
                timestamp=ts,
            )
        else:
            event = Event(
                timestamp=ts,
                event_type=event_type,
                run_id=self._config.trader_id,
                strategy_name=strategy_name,
                data=data,
            )
        self._event_writer.write(event)

    # ------------------------------------------------------------ lifecycle

    def _close_all_positions(self) -> CloseAllResult:
        """Backward-compatible alias for :meth:`close_all_positions`."""
        return self.close_all_positions()

    def close_all_positions(self) -> CloseAllResult:
        """Flatten every position (reduce-only) and cancel orders; keep running."""
        if self._guard is None:
            result = CloseAllResult(errors=["trading node not running"])
        else:
            result = self._guard.close_all()
        payload: dict[str, object] = {"action": "close_all_positions", **result.to_dict()}
        # oledr/k8oei: only a close-all with errors alerts today, so only it goes
        # through the helper; a clean close-all still writes without alerting.
        if result.errors:
            self._write_then_alert(
                EventType.SIGNAL,
                payload,
                "error",
                "Close-all problems: " + "; ".join(result.errors),
                critical=True,
            )
        else:
            self._write_event(EventType.SIGNAL, payload)
        return result

    async def _initialize(self) -> None:
        """Initialize node components.

        Raises:
            ConfigurationError: If initialization fails.
        """
        self._status.state = NodeState.INITIALIZING
        self._status.updated_at = datetime.now(UTC)

        engaged, reason = self._kill_switch()
        if engaged:
            raise ConfigurationError(
                f"System kill switch engaged ({reason or 'no reason'}); refusing to start"
            )

        # Create event writer
        self._config.logs_path.mkdir(parents=True, exist_ok=True)
        self._event_writer = EventWriter(
            run_id=self._config.trader_id,
            base_path=self._config.logs_path,
        )

        # Load, compile, resolve params, instantiate strategy (one per symbol)
        self._strategy = self._load_strategy()
        self._compiled_module = self._compile_strategy(self._strategy)
        self._resolved = self._resolve_run_config(self._compiled_module, self._strategy)
        self._compiled_strategies = self._instantiate_strategies(
            self._compiled_module, self._strategy
        )

        resolved = self._resolved
        self._write_event(
            EventType.SIGNAL,
            {
                "action": "start",
                "trader_id": self._config.trader_id,
                "strategy_id": self._config.strategy_id,
                "validation_run_id": self._config.validation_run_id,
                "symbols": resolved.symbols,
                "testnet": self._config.binance.testnet,
                "leverage": resolved.leverage,
                "margin_type": resolved.margin_type,
                "strategy_params": resolved.strategy_params,
                "config_diff": resolved.diff,
            },
        )
        if resolved.validated is None:
            logger.warning(
                "paper %s not started from a validation run: trading compiled DSL "
                "defaults, nothing to compare against",
                self._config.trader_id,
            )
        elif resolved.diff:
            logger.warning(
                "paper %s differs from validation run %s: %s",
                self._config.trader_id,
                self._config.validation_run_id,
                resolved.diff,
            )
        else:
            logger.info(
                "paper %s config matches validation run %s exactly (diff empty)",
                self._config.trader_id,
                self._config.validation_run_id,
            )

        # Persistence first: restore risk state before the guard is built.
        self._persistence = StatePersistence(
            db_path=self._config.db_path,
            trader_id=self._config.trader_id,
            checkpoint_interval=self._config.state_persistence_interval,
        )
        self._restore_from_checkpoint()
        self._commands = PaperCommandQueue(self._config.db_path)

        self._trading_node = self._create_live_trading_node()
        self._persistence.save_checkpoint(self._capture_checkpoint())

    async def _run_loop(self) -> None:
        """Main event loop for paper trading.

        Delegates execution lifecycle to NautilusTrader TradingNode.
        """
        if self._trading_node is None:
            raise RuntimeError("Trading node is not initialized")

        self._status.state = NodeState.RUNNING
        self._status.started_at = datetime.now(UTC)
        self._status.updated_at = datetime.now(UTC)

        self._write_event(
            EventType.SIGNAL,
            {
                "action": "state_change",
                "from_state": NodeState.INITIALIZING.value,
                "to_state": NodeState.RUNNING.value,
            },
        )

        if self._persistence is not None:
            await self._persistence.start_periodic_checkpointing(self._capture_checkpoint)

        self._control_task = asyncio.create_task(self._control_loop())
        self._start_liquidation_task()

        # Run Nautilus on our event loop (run_async) instead of in a worker
        # thread — wrapping the blocking ``run()`` in ``asyncio.to_thread``
        # produces "Event loop stopped before Future completed" on shutdown
        # because Nautilus spawns its own loop inside the thread.
        run_task = asyncio.create_task(self._trading_node.run_async())
        shutdown_task = asyncio.create_task(self._shutdown_event.wait())

        done, pending = await asyncio.wait(
            {run_task, shutdown_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        run_exc: BaseException | None = None
        try:
            if shutdown_task in done and not run_task.done():
                await self._trading_node.stop_async()
                await run_task
            elif run_task in done:
                self._shutdown_event.set()
                run_exc = run_task.exception()
        finally:
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await self._stop_control_loop()
            await self._stop_liquidation_loop()

        if run_exc is not None:
            raise run_exc

    async def _stop_control_loop(self) -> None:
        task = self._control_task
        self._control_task = None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    async def _shutdown(self) -> None:
        """Graceful shutdown of node components."""
        if self._status.state != NodeState.ERROR:
            self._status.state = NodeState.STOPPED
        self._status.updated_at = datetime.now(UTC)

        # R5 (k8oei): a failing stop-event write must not abort the whole shutdown;
        # class only, so no traceback (or chained fetch error) reaches the log.
        try:
            self._write_event(
                EventType.SIGNAL,
                {
                    "action": "stop",
                    "trader_id": self._config.trader_id,
                    "reason": (
                        self._status.halt_reason.value if self._status.halt_reason else "normal"
                    ),
                    "total_pnl": self._status.total_pnl,
                },
            )
        except Exception as exc:
            logger.error(
                "paper stop event write failed (%s); continuing shutdown", type(exc).__name__
            )

        await self._stop_control_loop()
        # R53: before the final checkpoint, so the watch state it captures is still there.
        await self._stop_liquidation_loop()

        if self._persistence is not None:
            with contextlib.suppress(Exception):
                self._persistence.save_checkpoint(self._capture_checkpoint())
            with contextlib.suppress(Exception):
                await self._persistence.stop_periodic_checkpointing()
            with contextlib.suppress(Exception):
                self._persistence.close()
            self._persistence = None

        if self._commands is not None:
            self._commands.close()
            self._commands = None

        # Stop trading node. ``TradingNode.dispose()`` stops the running
        # event loop (it was designed for ``run()`` which owns its own loop),
        # so we intentionally skip it.
        if self._trading_node is not None:
            with contextlib.suppress(Exception):
                await self._trading_node.stop_async()
            self._trading_node = None

        if self._event_writer is not None:
            with contextlib.suppress(Exception):
                self._event_writer.flush()
            # R6 (k8oei): a close() failure is logged by class and the writer dropped;
            # the remaining teardown (telegram, state manager) still runs.
            try:
                self._event_writer.close()
            except Exception as exc:
                logger.error(
                    "paper event writer close failed (%s); continuing shutdown",
                    type(exc).__name__,
                )
            self._event_writer = None

        # Let in-flight alerts (e.g. the halt that caused shutdown) go out.
        if self._alert_tasks:
            with contextlib.suppress(Exception):
                await asyncio.wait(set(self._alert_tasks), timeout=10)
        if self._telegram is not None:
            with contextlib.suppress(Exception):
                await self._telegram.close()

        self._state_manager.close()

    async def start(self) -> None:
        """Start the paper trading node.

        Runs until interrupted by signal (SIGINT/SIGTERM).
        """
        try:
            self._setup_signal_handlers()
            await self._initialize()
            await self._run_loop()
        except Exception as e:
            self._status.state = NodeState.ERROR
            self._status.error_message = str(e)
            # R3 (k8oei): the CRITICAL "failed" alert goes out even if the error-event
            # write raises, and the ORIGINAL error `e` is what start() re-raises:
            # suppressing the write's own failure keeps the two from being confused.
            with contextlib.suppress(Exception):
                self._write_then_alert(
                    EventType.RISK_CHECK,
                    {"action": "error", "error": str(e), "passed": False},
                    "error",
                    f"Paper node failed: {e}",
                    critical=True,
                )
            raise
        finally:
            await self._shutdown()

    def halt(self, reason: HaltReason, message: str | None = None) -> None:
        """Halt trading: block entries, flatten (reduce-only), cancel orders, stop strategies.

        A daily-loss halt keeps strategies running (flat, entries blocked) and
        lifts at the next UTC day. Halts are idempotent; a halt only replaces a
        less severe one.
        """
        msg = message or reason.value
        if self._guard is not None:
            self._guard.halt(reason, msg)
            return
        # Before the trading node exists there is nothing to flatten; record it.
        if self._status.state in (NodeState.RUNNING, NodeState.PAUSED):
            self._previous_state = self._status.state
        self._status.state = NodeState.HALTED
        self._status.halt_reason = reason
        self._status.error_message = msg
        self._status.updated_at = datetime.now(UTC)
        # R1 (k8oei): the CRITICAL halt alert cannot be swallowed by a failing event
        # write; the write's own error then propagates to the caller.
        self._write_then_alert(
            EventType.RISK_CHECK,
            {"action": "halt", "reason": reason.value, "message": msg, "passed": False},
            "circuit_breaker",
            f"HALT ({reason.value}): {msg}",
            critical=True,
        )

    def pause(self) -> None:
        """Block new entries; keep protective SL/TP orders and exit logic running."""
        if self._guard is not None:
            self._guard.pause()
            return
        if self._status.state == NodeState.RUNNING:
            self._status.state = NodeState.PAUSED
            self._status.updated_at = datetime.now(UTC)
            self._write_event(
                EventType.SIGNAL,
                {
                    "action": "state_change",
                    "from_state": NodeState.RUNNING.value,
                    "to_state": NodeState.PAUSED.value,
                },
            )

    def resume(self) -> tuple[bool, str]:
        """Resume from pause / operator-resumable halt. Never while killed."""
        engaged, _ = self._kill_switch()
        if self._guard is not None:
            return self._guard.resume(kill_switch_engaged=engaged)
        if engaged:
            return False, "system kill switch is engaged; unlock it before resuming"
        if self._status.state == NodeState.PAUSED:
            self._status.state = NodeState.RUNNING
            self._status.updated_at = datetime.now(UTC)
            self._write_event(
                EventType.SIGNAL,
                {
                    "action": "state_change",
                    "from_state": NodeState.PAUSED.value,
                    "to_state": NodeState.RUNNING.value,
                },
            )
            return True, "resumed from pause"
        if self._status.state == NodeState.RUNNING:
            return True, "already running"
        reason = self._status.halt_reason
        if self._status.state == NodeState.HALTED and reason in OPERATOR_RESUMABLE_HALTS:
            target = self._previous_state or NodeState.RUNNING
            self._status.state = target
            self._status.halt_reason = None
            self._status.error_message = None
            self._status.updated_at = datetime.now(UTC)
            self._previous_state = None
            self._write_event(
                EventType.SIGNAL,
                {
                    "action": "state_change",
                    "from_state": NodeState.HALTED.value,
                    "to_state": target.value,
                    "resumed": True,
                },
            )
            return True, "resumed from halt"
        if self._status.state == NodeState.HALTED:
            why = reason.value if reason else "unknown"
            return False, f"{why} halt cannot be resumed by the operator"
        return False, "trading node not running"

    def resume_from_halt(self) -> bool:
        """Resume trading from halted state (operator-resumable halts only)."""
        if self._status.state != NodeState.HALTED:
            return False
        ok, _ = self.resume()
        return ok

    def handle_error(
        self,
        error: BaseException,
        operation: str = "",
        symbol: str = "",
    ) -> ErrorContext:
        """Handle error through the error handler (halt and/or alert)."""
        return self._error_handler.handle_error(
            error=error,
            operation=operation,
            symbol=symbol,
        )


async def run_paper_trading(config: PaperTradingConfig) -> None:
    """Run paper trading with the given configuration."""
    node = PaperTradingNode(config)
    await node.start()


__all__ = [
    "HaltReason",
    "NodeState",
    "NodeStatus",
    "PaperTradingNode",
    "ResolvedRunConfig",
    "run_paper_trading",
    "validated_leverage_override",
    "validated_run_leverage",
    "validated_run_liquidation",
    "validated_run_strategy_params",
]
