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
import copy
import json
import logging
import math
import re
import signal
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from vibe_quant.alerts.telegram import (
    AlertType,
    TelegramBot,
    TelegramUnavailableError,
    require_working_telegram,
)
from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.fees import FeeSchedule
from vibe_quant.liquidation_settings import LiquidationConfigError, LiquidationSettings
from vibe_quant.logging.events import Event, EventType, create_event
from vibe_quant.logging.writer import EventWriter
from vibe_quant.margin_limit import DEFAULT_PAPER_MAX_MARGIN_PCT
from vibe_quant.paper.config import (
    DEFAULT_MARGIN_TYPE,
    DEFAULT_VALIDATION_LEVERAGE,
    ConfigurationError,
    PaperTradingConfig,
    account_env,
    check_paper_exposure,
    effective_max_margin_pct,
    effective_max_position_pct,
    is_key_rotation,
)
from vibe_quant.paper.config import (
    account_id as binance_account_id,
)
from vibe_quant.paper.errors import ErrorContext, ErrorHandler
from vibe_quant.paper.event_log import EventLogWorker, EventWriteTimeoutError
from vibe_quant.paper.force_orders import build_binance_fetch
from vibe_quant.paper.guard import (
    ENTRY_CHECK_FAILED_PREFIX,
    MARGIN_DENIAL_PREFIX,
    RESTORED_HALT_PREFIX,
    RESTORED_HALTS,
    CloseAllResult,
    GuardListener,
    GuardState,
    HaltReason,
    RestoredHalt,
    RiskState,
    TradingGuard,
    TradingGuardConfig,
)
from vibe_quant.paper.halt_state import PrestartHalt
from vibe_quant.paper.kill_switch_monitor import (
    KILL_SWITCH_RESUME_FRESH_S,
    KILL_SWITCH_STALE_LIMIT_S,
    KILL_SWITCH_STARTUP_TIMEOUT_S,
    KILL_SWITCH_WATCH_TICK_S,
    KillSwitchMonitor,
)
from vibe_quant.paper.liquidation_watch import (
    FORCE_ORDERS_WEIGHT,
    IP_WEIGHT_LIMIT_PER_MIN,
    WATCH_WEIGHT_BUDGET_PER_MIN,
    LiquidationWatcher,
)
from vibe_quant.paper.persistence import (
    PaperCommand,
    PaperCommandQueue,
    StateCheckpoint,
    StatePersistence,
    StoreWorker,
    wait_future,
)
from vibe_quant.sizing_settings import DEFAULT_MAX_POSITION_PCT

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future
    from decimal import Decimal
    from types import ModuleType

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
        # Paper enforces the margin limit in the guard, on the venue-VERIFIED leverage; a
        # strategy config must not carry the backtest's rule (the template would default
        # margin_leverage to 10).
        "max_margin_pct",
        "margin_leverage",
    }
)

#: R42b: how long the venue may stay unconnected before the node reports the startup
#: refusal (an exchange that refuses the margin change leaves the node not trading).
STARTUP_CONNECT_TIMEOUT_S = 120
# R19c: the final shutdown checkpoint is waited at most this long; R19d: a store worker's
# close is waited at most this long (a hung statement closes after it returns).
SHUTDOWN_CHECKPOINT_WAIT_S = 10.0
STORE_CLOSE_WAIT_S = 2.0
# ogd37: event-log appends run on their own worker. A site that must know the write landed
# (the liquidation reports) waits at most EVENT_WRITE_WAIT_S; start waits at most
# EVENT_STARTUP_WAIT_S for the first lines (an unwritable log refuses to start).
EVENT_WRITE_WAIT_S = 5.0
EVENT_STARTUP_WAIT_S = 10.0

#: ci7q9: seconds a trade-alert task waits before reading the cache, so the strategy's
#: SL/TP orders (submitted only after the entry fill) are visible. Never on the NT callback.
TRADE_ALERT_DELAY_S = 2.0

#: 3ljxh: direction -> (open emoji, arrow emoji) for the human-friendly trade texts.
_DIRECTION_EMOJI = {"LONG": ("🟢", "📈"), "SHORT": ("🔴", "📉")}

#: 3ljxh: exit reason -> the text shown in the close alert (fallback: ``❔ <reason>``).
_EXIT_REASON_TEXT = {
    "stop_loss": "🛑 stop loss hit",
    "take_profit": "🎯 take profit hit",
    "signal": "📊 strategy signal",
    "halt_or_close_all": "⏹️ closed by halt",
}


def _signed(value: float, decimals: int = 2) -> str:
    """A number with an explicit sign; negatives use U+2212 MINUS SIGN (Telegram-friendly)."""
    sign = "+" if value >= 0 else "−"
    return f"{sign}{abs(value):.{decimals}f}"


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

#: The pre-guard halt record speaks ``GuardState``; INITIALIZING/STOPPED/ERROR have
#: no pre-halt state to return to, so they map to ``None``.
_NODE_TO_GUARD: dict[NodeState, GuardState] = {
    node: guard for guard, node in _GUARD_TO_NODE_STATE.items()
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
    #: The guard's margin cushion (fraction of equity) and the per-symbol cap it is judged by.
    max_margin_pct: Decimal = DEFAULT_PAPER_MAX_MARGIN_PCT
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT


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


def restored_halt_from_status(node_status: object, checkpoint_time: str) -> RestoredHalt | None:
    """Decide what a prior checkpoint's ``node_status`` restores (D-H1/H22).

    MANUAL/ERROR/SIGNAL/KILL_SWITCH restore HALTED with a
    ``RESTORED_HALT_PREFIX`` message (idempotent across restarts); risk halts are
    left to the restored bookkeeping (they re-trip in ``evaluate_risk``);
    anything unrecognised fails closed as ERROR. ``None`` means start RUNNING.
    """
    if not isinstance(node_status, dict):
        return None
    raw = node_status.get("halt_reason")
    if raw is None:
        return None
    reason: HaltReason | None = None
    if isinstance(raw, str):
        with contextlib.suppress(ValueError):
            reason = HaltReason(raw)
    if reason is not None:
        if reason not in RESTORED_HALTS:
            return None
        text_raw = node_status.get("error_message")
        text = text_raw if isinstance(text_raw, str) and text_raw.strip() else reason.value
        if text.startswith(RESTORED_HALT_PREFIX):
            text = text[len(RESTORED_HALT_PREFIX) :]
        return RestoredHalt(
            reason=reason,
            message=RESTORED_HALT_PREFIX + text,
            checkpoint_time=checkpoint_time,
        )
    return RestoredHalt(
        reason=HaltReason.ERROR,
        message=f"{RESTORED_HALT_PREFIX}unrecognised prior halt_reason {raw!r}",
        checkpoint_time=checkpoint_time,
    )


#: One leading prefix of a stored halt message: this module's own restore prefix or an
#: inherited one. Stripped in a loop so messages do not grow along a chain of sessions.
_HALT_PREFIX_RE = re.compile(rf"(?:{re.escape(RESTORED_HALT_PREFIX)}|inherited from [A-Za-z0-9_-]+: )")


def inherited_halt_message(trader_id: str, message: str) -> str:
    """Message of an ERROR halt inherited from ``trader_id``'s last checkpoint.

    Strips every leading restored/inherited prefix, then prefixes exactly one
    ``inherited from <trader_id>: `` (R14e: no growth along a chain of sessions).
    """
    rest = message
    while match := _HALT_PREFIX_RE.match(rest):
        rest = rest[match.end() :]
    return f"inherited from {trader_id}: {rest}"


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

    def on_position_close_settled(self, event: Any) -> None:
        self._node._on_position_close_settled(event)

    def on_warning(self, message: str) -> None:
        self._node._on_guard_warning(message)

    def on_entry_allowed(self, symbol: str) -> None:
        self._node._margin_alerted.discard(symbol)
        self._node._entry_check_alerted.discard(symbol)


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
        # T2b/B2: ALL checkpoint writes go through one FIFO worker (capture order =
        # write order) and the command queue has its own worker; each worker closes
        # its own store after its last job (R19d), so the loop never blocks on the DB.
        self._db_worker = StoreWorker("vq-paper-db")
        self._cmd_worker = StoreWorker("vq-paper-cmd")
        # ogd37: event-file appends have their own FIFO worker (not _db_worker, which
        # stops before the trading engine does), so a stalled disk never blocks the loop.
        self._event_log = EventLogWorker("vq-paper-events")
        # R12/Q2: the counters as of the previous control tick, and the (failed, dropped)
        # baseline of the outage streak now in progress (None while the log is healthy).
        self._event_log_seen: tuple[int, int, int] | None = None
        self._event_log_streak: tuple[int, int] | None = None
        self._guard: TradingGuard | None = None
        self._restored_risk: RiskState | None = None
        #: trader_id whose checkpoint supplied ``_restored_risk`` when it is not our own.
        self._risk_inherited_from: str | None = None
        #: Symbols whose margin-limit refusal was already alerted (re-armed by an allowed entry).
        self._margin_alerted: set[str] = set()
        #: Symbols whose raising-entry-check refusal was already alerted (re-armed on entry).
        self._entry_check_alerted: set[str] = set()
        self._restored_halt: RestoredHalt | None = None
        #: The pre-guard halt record: a halt that arrives before the guard exists
        #: (SIGUSR1 during start-up) is recorded here and the guard applies it right
        #: after it is built (``_apply_startup_halt``).
        self._prestart = PrestartHalt()
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
        self._connected_seen_in_stop = False
        self._kill_reason_seen: str | None = None
        # The kill switch is read on a reader thread that owns its own state-DB
        # handle (T2a/B2); the loop only ever looks at its snapshot. The latch is
        # the one ERROR halt per stale streak.
        self._kill_monitor = KillSwitchMonitor(lambda: StateManager(config.db_path))
        self._kill_read_halted: bool = False
        #: J5: True once the executing close_all has issued its flatten orders; read by
        #: ``process_commands`` when that command is cancelled while the flatten confirms.
        self._flatten_issued: bool = False
        self._kill_watch_task: asyncio.Task[None] | None = None
        self._trades_day: date | None = None
        # R26: entries already logged, so a repeated snapshot failure logs once.
        self._snapshot_failures_logged: set[str] = set()

        # Telegram alerts are built in ``_initialize`` (u2r9r): a configured bot
        # must pass the start probe before the session is allowed to run.
        self._telegram: TelegramBot | None = None

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

    # --------------------------------------------------------- trade alerts

    def _send_trade_alert(self, build: Callable[[], str], *, delay: float = 0.0) -> None:
        """Fire-and-forget entry/exit alert; the text is built inside the alert task.

        ci7q9, report-only: the text is built lazily so a raising cache lookup lands in
        the task's own handler, never on the NT fill callback (a raise there would also
        skip the guard's risk evaluation). An entry alert waits ``delay`` before
        building, because the compiled strategy submits its SL/TP orders only AFTER the
        entry fill. No Telegram / setting off -> nothing is scheduled.
        """
        telegram = self._telegram
        if not self._config.trade_alerts or telegram is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            logger.warning("trade alert not sent (%s)", type(exc).__name__)
            return
        task = loop.create_task(self._trade_alert_task(telegram, build, delay))
        self._alert_tasks.add(task)
        task.add_done_callback(self._alert_done)

    async def _trade_alert_task(
        self, telegram: TelegramBot, build: Callable[[], str], delay: float
    ) -> bool:
        """Send one TRADE alert; every failure is swallowed with a class-only log.

        R31: the TRADE per-type rate limit is bypassed. OPEN and CLOSE share
        ``AlertType.TRADE`` with a 60 s limit, so a fast round trip would silently drop
        the CLOSE; the trade count is the natural bound and trade alerts are opt-in.
        """
        try:
            if delay:
                await asyncio.sleep(delay)
            return await telegram.send_alert(AlertType.TRADE, build(), bypass_rate_limit=True)
        except Exception as exc:
            # Class only, never the text: a chained cache/httpx error can carry secrets.
            logger.warning("trade alert not sent (%s)", type(exc).__name__)
            return False

    def _trade_open_message(self, event: Any) -> str:
        """The exact entry text: size, leverage, entry and the strategy's SL/TP."""
        cache = self._guard.cache if self._guard is not None else None
        symbol = self._symbol(event.instrument_id)
        direction = "LONG" if event.side.name == "LONG" else "SHORT"
        head, arrow = _DIRECTION_EMOJI[direction]
        quantity = float(event.quantity)
        entry = float(event.avg_px_open)
        leverage = self._resolved.leverage if self._resolved else 1
        stop, take = self._protective_levels(cache, event.instrument_id)
        return "\n".join(
            [
                f"{head} OPENED {direction} {arrow} {symbol}",
                f"💰 Size: {quantity} {self._base_asset(cache, event.instrument_id, symbol)}"
                f" (≈ ${quantity * entry:,.2f})",
                f"⚡ Leverage: {leverage}x",
                f"🎯 Entry: {self._alert_price(cache, event.instrument_id, entry)}",
                f"🛑 Stop: {self._level_text(cache, event.instrument_id, stop, entry)}",
                f"✅ Target: {self._level_text(cache, event.instrument_id, take, entry)}",
                self._session_label(),
            ]
        )

    def _trade_close_message(
        self, event: Any, net: float, commissions: float, exit_reason: str
    ) -> str:
        """The exact exit text: outcome, net P&L, exit price/reason and held time.

        A NT ``PositionClosed.side`` is always FLAT, so the closed direction comes from
        ``event.entry`` (the entry order side): BUY -> LONG, SELL -> SHORT.
        """
        cache = self._guard.cache if self._guard is not None else None
        symbol = self._symbol(event.instrument_id)
        direction = "LONG" if event.entry.name == "BUY" else "SHORT"
        arrow = _DIRECTION_EMOJI[direction][1]
        pos = cache.position(event.position_id) if cache is not None else None
        entry = float(pos.avg_px_open) if pos is not None else None
        entry_text = (
            self._alert_price(cache, event.instrument_id, entry)
            if entry is not None
            else "unknown"
        )
        exit_price = self._alert_price(cache, event.instrument_id, float(event.avg_px_close))
        # PositionClosed.quantity is 0 (the position is flat); peak_qty is the size held.
        pnl_line = self._pnl_text(net, entry, float(event.peak_qty))
        reason_text = _EXIT_REASON_TEXT.get(exit_reason, f"❔ {exit_reason}")
        return "\n".join(
            [
                f"{'✅' if net >= 0 else '❌'} CLOSED {direction} {arrow} {symbol} ·"
                f" {'WIN' if net >= 0 else 'LOSS'}",
                pnl_line,
                f"🚪 Exit: {exit_price} · {reason_text}",
                f"🎯 Entry: {entry_text}",
                f"⏱️ Held: {self._held_text(event.duration_ns)}",
                f"💸 Fees: {commissions:.2f} USDT",
                self._session_label(),
            ]
        )

    @staticmethod
    def _pnl_text(net: float, entry: float | None, quantity: float) -> str:
        """The P&L line: signed net USDT, plus the signed % of entry notional when known."""
        arm = "📈" if net >= 0 else "📉"
        amount = _signed(net)
        # vgjcg: a 0 notional (peak_qty 0, or a zero entry) has no percentage to show.
        # Dividing raised inside the alert task and dropped the whole CLOSE message.
        if entry is None or quantity * entry == 0:
            return f"{arm} P&L: {amount} USDT"
        percent = net / (quantity * entry) * 100
        return f"{arm} P&L: {amount} USDT ({_signed(percent)}%)"

    @staticmethod
    def _held_text(duration_ns: int) -> str:
        """Format a position's lifetime: ``<h>h <mm>m`` >= 1 h, ``<m>m <ss>s`` >= 1 min, else ``<s>s``."""
        total_s = duration_ns // 1_000_000_000
        hours, remainder = divmod(total_s, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours >= 1:
            return f"{hours}h {minutes:02d}m"
        if minutes >= 1:
            return f"{minutes}m {seconds:02d}s"
        return f"{seconds}s"

    @staticmethod
    def _symbol(instrument_id: Any) -> str:
        """The instrument symbol without its venue suffix (``BTCUSDT-PERP.BINANCE`` -> ``BTCUSDT``)."""
        return str(instrument_id).split(".")[0].removesuffix("-PERP")

    @staticmethod
    def _base_asset(cache: Any, instrument_id: Any, symbol: str) -> str:
        """The instrument's base currency, else the symbol minus a USDT/USDC quote suffix."""
        if cache is not None:
            instrument = cache.instrument(instrument_id)
            base = getattr(instrument, "base_currency", None)
            if base is not None:
                return str(base)
        for quote in ("USDT", "USDC"):
            if symbol.endswith(quote):
                return symbol[: -len(quote)]
        return symbol

    def _level_text(self, cache: Any, instrument_id: Any, level: float | None, entry: float) -> str:
        """``<price> (±x.xx%)`` for a protective level, or ``none placed``."""
        if level is None:
            return "none placed"
        price = self._alert_price(cache, instrument_id, level)
        return f"{price} ({_signed((level / entry - 1) * 100)}%)"

    def _session_label(self) -> str:
        """The trailing line: the trader id and whether the venue is testnet or live."""
        env = "🧪 testnet" if self._config.binance.testnet else "💵 LIVE"
        return f"🤖 {self._config.trader_id} · {env}"

    @staticmethod
    def _protective_levels(cache: Any, instrument_id: Any) -> tuple[float | None, float | None]:
        """(stop, take) prices for the instrument's open reduce-only orders.

        Stop = STOP_MARKET/STOP_LIMIT trigger price, take-profit = LIMIT price. Missing
        on either side is ``None``.
        """
        stop: float | None = None
        take: float | None = None
        if cache is not None:
            orders = list(cache.orders_open(instrument_id=instrument_id)) + list(
                cache.orders_inflight(instrument_id=instrument_id)
            )
            for order in orders:
                if not getattr(order, "is_reduce_only", False):
                    continue
                name = order.order_type.name
                if stop is None and name in ("STOP_MARKET", "STOP_LIMIT"):
                    stop = float(order.trigger_price)
                elif take is None and name == "LIMIT":
                    take = float(order.price)
        return (stop, take)

    @staticmethod
    def _alert_price(cache: Any, instrument_id: Any, value: float) -> str:
        """Format ``value`` at the instrument's price precision with thousands separators."""
        precision = None
        if cache is not None:
            instrument = cache.instrument(instrument_id)
            if instrument is not None:
                precision = getattr(instrument, "price_precision", None)
        if isinstance(precision, int):
            return f"{value:,.{precision}f}"
        return str(value)

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
        """Submit the event, then alert, even when the submit itself raises (R1/irdig).

        The append runs on the event-log worker (ogd37), so the alert never waits for the
        disk: the line may land after the alert. A submit-time error (an unbuildable
        event, a stopped worker) still sends the alert with the same text and
        ``critical`` flag and only then propagates unchanged. Sites that must know the
        line landed use :meth:`_awrite_then_alert`.
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

    async def _awrite_then_alert(
        self,
        event_type: EventType,
        data: dict[str, object],
        kind: str,
        message: str,
        *,
        critical: bool = False,
        timestamp: datetime | None = None,
    ) -> None:
        """Write the event, wait for it (bounded), then alert, even when the write fails.

        For the liquidation reports (R47b: a failed write fails the watcher's cycle so the
        row is re-delivered). A timeout raises ``EventWriteTimeoutError`` and a worker
        error is re-raised as is, each after the alert; a cancellation (shutdown) sends the
        alert once and propagates. The loop keeps running while it waits.
        """
        try:
            future = self._write_event(event_type, data, timestamp=timestamp)
            if future is not None:
                if not await wait_future(future, EVENT_WRITE_WAIT_S):
                    raise EventWriteTimeoutError(
                        f"paper event write not confirmed within {EVENT_WRITE_WAIT_S:g} s"
                    )
                future.result()
        except asyncio.CancelledError:
            self._send_telegram_alert(kind, message, critical=critical)
            raise
        except Exception as exc:
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
        run_parameters: object = None
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
            run_parameters = raw_params
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

        # D-S5: the same shared rule the API runs before it writes any run row.
        # Right after the max_leverage check so that error keeps precedence, and
        # before ResolvedRunConfig, so no strategy / checkpoint / TradingNode is
        # built for a run whose cap its leverage cannot cover.
        check_paper_exposure(
            validation_run_id=cfg.validation_run_id,
            run_parameters=run_parameters,
            max_position_pct_override=cfg.sizing.max_position_pct,
            leverage_override=cfg.leverage,
        )

        max_margin_pct = effective_max_margin_pct(
            run_parameters, cfg.sizing.max_margin_pct, validation_run_id=cfg.validation_run_id
        )
        resolved = ResolvedRunConfig(
            max_margin_pct=max_margin_pct,
            max_position_pct=effective_max_position_pct(
                run_parameters, cfg.sizing.max_position_pct
            ),
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
                {
                    "field": "symbols",
                    "validated": validated.get("symbols"),
                    "paper": resolved.symbols,
                }
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

    def _warn_margin_cushion(self, resolved: ResolvedRunConfig) -> None:
        """Q3: warn (never refuse) when every symbol at its cap would exceed the cushion."""
        possible = len(resolved.symbols) * resolved.max_position_pct / resolved.leverage
        cushion = float(resolved.max_margin_pct)
        if possible <= cushion:
            return
        message = (
            f"max possible margin {possible:.0%} exceeds cushion {cushion:.0%}; "
            "entries beyond it will be refused"
        )
        logger.warning("paper %s: %s", self._config.trader_id, message)
        self._write_event(
            EventType.SIGNAL,
            {
                "action": "margin_cushion_warning",
                "message": message,
                "max_possible_margin_pct": possible,
                "max_margin_pct": cushion,
            },
        )

    def _build_guard(self, risk_engine: Any) -> TradingGuard:
        risk = self._config.risk
        return TradingGuard(
            TradingGuardConfig(
                max_drawdown_pct=risk.max_drawdown_pct,
                max_daily_loss_pct=risk.max_daily_loss_pct,
                max_consecutive_losses=risk.max_consecutive_losses,
                max_position_count=risk.max_position_count,
                max_margin_pct=float(
                    self._resolved.max_margin_pct
                    if self._resolved is not None
                    else DEFAULT_PAPER_MAX_MARGIN_PCT
                ),
            ),
            strategies=self._compiled_strategies,
            risk_engine=risk_engine,
            listener=_GuardBridge(self),
            risk_state=self._restored_risk,
            restored_halt=self._restored_halt,
            risk_source=self._risk_inherited_from,
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

    async def _control_loop(self) -> None:
        """Poll operator commands and connectivity.

        Each step runs in its own try. The kill switch has its own watcher task
        (``_kill_watch_loop``), so a stalled command claim cannot delay it.
        """
        interval = self._config.control_poll_interval
        while not self._shutdown_event.is_set():
            try:
                await self.process_commands()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # ott8r: class only (R1b). ``logger.exception`` prints the active
                # exception's ``__context__`` chain too - a fetch error carrying a
                # signed URL / api key - plus its traceback.
                logger.error("paper control loop iteration failed (%s)", type(exc).__name__)
            try:
                self._check_connectivity()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("paper control loop iteration failed (%s)", type(exc).__name__)
            try:
                self._check_event_log()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("paper control loop iteration failed (%s)", type(exc).__name__)
            await asyncio.sleep(interval)

    def _mono(self) -> float:
        """Monotonic seconds (one seam for the kill-switch snapshot age)."""
        return time.monotonic()

    async def _kill_watch_loop(self) -> None:
        """Look at the kill-switch snapshot every ``KILL_SWITCH_WATCH_TICK_S`` until shutdown.

        Own task, own try per tick: nothing else on the loop can skip the check.
        """
        while not self._shutdown_event.is_set():
            try:
                self._check_kill_switch()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("paper kill-switch watch iteration failed (%s)", type(exc).__name__)
            await asyncio.sleep(KILL_SWITCH_WATCH_TICK_S)

    def _check_kill_switch(self) -> None:
        """Fail closed on a stale snapshot, escalate on an engaged one. No DB access."""
        snap = self._kill_monitor.snapshot()
        age = snap.age(self._mono())
        if age >= KILL_SWITCH_STALE_LIMIT_S:
            # The latch is set first, so later stale ticks are silent. ERROR, never
            # KILL_SWITCH (a kill reason would make a MAX_DRAWDOWN halt resumable).
            if not self._kill_read_halted:
                self._kill_read_halted = True
                detail = snap.error_class or "no clean read"
                self.halt(
                    HaltReason.ERROR,
                    f"system kill switch unreadable for {age:.1f} s ({detail}); failing closed",
                )
            return
        self._kill_read_halted = False
        engaged, reason = snap.engaged, snap.reason
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

    def _shutdown_in_progress(self) -> bool:
        """True while the venue is being torn down on an operator stop request.

        NT 1.231 has no public ``is_stopping``: ``kernel._is_stopping`` is set at the
        top of ``stop_async`` -- BEFORE the data/exec clients disconnect -- and cleared
        only when it returns (``system/kernel.py``, verified at the source). A tick in
        that window is the SIGTERM we asked for, not a venue failure. The node's
        ``_shutdown_event`` covers the ticks between ``run_async`` returning and the
        control loop being cancelled.
        """
        kernel = getattr(self._trading_node, "kernel", None)
        if getattr(kernel, "_is_stopping", False):
            return True
        return self._shutdown_event.is_set()

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
        # yejfx/R2: a disconnect during a stop is the stop itself ONLY if this node saw
        # the venue connected after the stop began; otherwise it was lost before.
        if not self._shutdown_in_progress():
            self._connected_seen_in_stop = False
        elif connected:
            self._connected_seen_in_stop = True
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
            # yejfx: the clients disconnect BECAUSE the operator asked the node to stop;
            # paging "venue connection lost" there is a false alarm during every SIGTERM.
            # Fail open: a loss seen before any connected observation in the stop
            # is a genuine disconnect and still alerts.
            if self._connected_seen_in_stop and self._shutdown_in_progress():
                return
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

    def _check_event_log(self) -> None:
        """Alert once per event-log outage, report once when it writes again (R12/Q2).

        ``health()`` is a lock-protected snapshot, so this never blocks the loop. An
        increase in ``failed + dropped`` since the previous tick opens a streak: one
        CRITICAL alert, with the counts accumulated since the streak began, is sent and
        no later rising tick repeats it (writing an event line here would be pointless -
        the log is exactly what is broken). The first tick that sees a new successful
        write and no new failure closes the streak: one ``RISK_CHECK``
        ``event_log_recovered`` line through the normal write path plus one
        non-critical alert. Unchanged counters are silent in both directions.
        """
        health = self._event_log.health()
        failed, dropped, written = health.failed, health.dropped, health.written
        previous = self._event_log_seen
        # Recorded before anything can raise: a failed alert must not re-fire next tick.
        self._event_log_seen = (failed, dropped, written)
        if previous is None:
            return
        prev_failed, prev_dropped, prev_written = previous
        if failed + dropped > prev_failed + prev_dropped:
            if self._event_log_streak is not None:
                return
            self._event_log_streak = (prev_failed, prev_dropped)
            error_class = health.last_error_class if failed > prev_failed else "queue full"
            self._send_telegram_alert(
                "error",
                f"paper {self._config.trader_id} event log not writing: "
                f"{failed - prev_failed} failed, {dropped - prev_dropped} dropped "
                f"({error_class}); trading continues",
                critical=True,
            )
            return
        if written > prev_written and self._event_log_streak is not None:
            base_failed, base_dropped = self._event_log_streak
            # Cleared before the write: if the recovery line itself fails to land, the
            # next failing tick opens a fresh streak rather than repeating this one.
            self._event_log_streak = None
            recovered_failed = failed - base_failed
            recovered_dropped = dropped - base_dropped
            self._write_then_alert(
                EventType.RISK_CHECK,
                {
                    "action": "event_log_recovered",
                    "failed": recovered_failed,
                    "dropped": recovered_dropped,
                    "passed": True,
                },
                "error",
                f"paper {self._config.trader_id} event log writing again "
                f"({recovered_dropped} dropped, {recovered_failed} failed)",
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
                await self._awrite_then_alert(
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
            # 0m3tf/1cvw6: restored cursors of symbols this session does NOT trade are
            # handed over as well; the watcher polls them after the traded symbols.
            cursors = dict(restored["cursors"])
            for symbol in resolved.symbols:
                cursors.setdefault(symbol, self._restored_watch_ms)
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
        weight = watcher.weight_per_minute()
        if weight > WATCH_WEIGHT_BUDGET_PER_MIN:
            logger.warning(
                "paper %s: liquidation watch request weight %d/min over the %d/min budget "
                "(%d symbols x %d per 60 s; Binance IP cap %d/min)",
                self._config.trader_id,
                weight,
                WATCH_WEIGHT_BUDGET_PER_MIN,
                len(watcher.state()["cursors"]),
                FORCE_ORDERS_WEIGHT,
                IP_WEIGHT_LIMIT_PER_MIN,
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
        await self._awrite_then_alert(
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
            f"{self._liquidation_scope(order.symbol)}; NO paper action taken",
            critical=True,
            timestamp=_ns_to_dt(order.ts_ns),
        )

    def _liquidation_scope(self, symbol: str) -> str:
        """toqng: a carried symbol's alert says the position predates this session."""
        resolved = self._resolved
        if resolved is not None and symbol not in resolved.symbols:
            return "not traded this session (position left by an earlier session)"
        return "paper position state may be stale"

    async def _on_liquidation_warning(self, message: str) -> None:
        """R52: a degraded or clamped detector warns without a critical alert."""
        await self._awrite_then_alert(
            EventType.RISK_CHECK,
            {"action": "liquidation_watch_degraded", "message": message, "passed": False},
            "error",
            f"WARNING: {message}",
        )

    async def _on_liquidation_recovered(self, symbol: str, failures: int) -> None:
        """R52: the first success after a degraded run is reported once."""
        await self._awrite_then_alert(
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

    async def process_commands(self, *, startup_halts_only: bool = False) -> None:
        """Execute queued operator commands for this trader.

        ``startup_halts_only`` is the pre-NT drain: only the leading halts/kills that
        no pending resume precedes are claimed (see ``claim_startup_halts``).
        """
        commands = self._commands
        if commands is None:
            return
        claim_fn = commands.claim_startup_halts if startup_halts_only else commands.claim_pending
        claim = self._cmd_worker.submit(claim_fn, self._config.trader_id)
        try:
            claimed: list[PaperCommand] = await asyncio.wrap_future(claim)
        except asyncio.CancelledError:
            # The worker may still finish the claim: fail those rows (never leave 'running').
            self._cmd_worker.submit(self._fail_claimed, commands, claim, [])
            raise
        for index, cmd in enumerate(claimed):
            finishing = False  # True once the command ran and its outcome is being written
            # J5: set by a close_all once its flatten orders are out; read by the
            # CancelledError handler below so such a row does not claim it never ran.
            self._flatten_issued = False
            try:
                try:
                    ok, result, error = await self._execute_command(cmd)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # ott8r: class only (R1b) - the command name identifies the command,
                    # while ``logger.exception`` would print its text and the active
                    # exception's ``__context__`` chain (a fetch error with signature/apiKey).
                    logger.error("paper command %s failed (%s)", cmd.command, type(exc).__name__)
                    finishing = True
                    await self._cmd_worker.run(
                        commands.fail,
                        cmd.id,
                        f"{type(exc).__name__}: command failed; see the paper node log",
                    )
                    continue
                finishing = True
                if ok:
                    await self._cmd_worker.run(commands.complete, cmd.id, result)
                else:
                    await self._cmd_worker.run(commands.fail, cmd.id, error or "failed", result)
            except asyncio.CancelledError:
                # A command whose outcome write was in flight keeps that write.
                rest = claimed[index + 1 :] if finishing else claimed[index:]
                # J5: the head of ``rest`` is the command that was executing, and only a
                # command still executing (``not finishing``) can have just issued a flatten.
                self._cmd_worker.submit(
                    self._fail_claimed,
                    commands,
                    None,
                    rest,
                    self._flatten_issued and not finishing,
                )
                raise

    _NOT_EXECUTED = "node stopping; command not executed"
    #: J2: what a row left ``running`` by a previous process gets - its outcome is unknown.
    _STALE_RUNNING = (
        "node restarted; command outcome unknown (was running when the node stopped)"
    )
    #: J5: a close_all cancelled after its flatten orders went out, flat still unconfirmed.
    _FLAT_NOT_CONFIRMED = "node stopping; close_all issued, flat not confirmed"

    @classmethod
    def _fail_claimed(
        cls,
        commands: PaperCommandQueue,
        claim: Future[list[PaperCommand]] | None,
        rest: list[PaperCommand],
        flatten_issued: bool = False,
    ) -> None:
        """Worker job: fail claimed-but-unexecuted commands (runs after the claim returned).

        ``flatten_issued``: the first unexecuted command (``rest[0]``) is a close_all whose
        flatten orders are already out, so its row says so instead of "not executed".
        """
        pending = list(rest)
        if claim is not None and not claim.cancelled() and claim.exception() is None:
            pending.extend(claim.result())
        for index, cmd in enumerate(pending):
            error = cls._FLAT_NOT_CONFIRMED if index == 0 and flatten_issued else cls._NOT_EXECUTED
            commands.fail(cmd.id, error)

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
            # J5: from here on the flatten orders are out; a cancellation landing in
            # ``_wait_flat`` must not report this command as never executed.
            self._flatten_issued = True
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
        reason = str(event.reason)
        symbol = str(event.instrument_id)
        if (
            by_gate
            and reason.startswith(MARGIN_DENIAL_PREFIX)
            and symbol not in self._margin_alerted
        ):
            # One alert per symbol until an entry for it passes again (latch set before the
            # write, like every per-tick report).
            self._margin_alerted.add(symbol)
            self._write_then_alert(
                EventType.ORDER,
                payload,
                "warning",
                f"Entry refused (margin limit): {reason}\n{symbol}",
                timestamp=timestamp,
            )
        elif (
            by_gate
            and reason.startswith(ENTRY_CHECK_FAILED_PREFIX)
            and symbol not in self._entry_check_alerted
        ):
            # One alert per symbol until an entry for it passes again (latch set before the
            # write): a strategy retrying every bar must not flood the operator, but a
            # symbol whose entry check recovers and then fails again alerts anew.
            self._entry_check_alerted.add(symbol)
            self._write_then_alert(
                EventType.ORDER,
                payload,
                "error",
                f"Entry check failed (order denied): {reason}\n{symbol}",
                timestamp=timestamp,
            )
        elif not by_gate:
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
        # ci7q9: one entry message, built after the strategy's SL/TP land in the cache.
        self._send_trade_alert(
            lambda: self._trade_open_message(event), delay=TRADE_ALERT_DELAY_S
        )

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
        # ci7q9: one exit message from the event's own P&L numbers (built in the task).
        self._send_trade_alert(
            lambda: self._trade_close_message(event, net, commissions, exit_reason)
        )

    def _on_position_close_settled(self, event: Any) -> None:
        """8vc7h: checkpoint right after a close so a SIGKILL cannot lose the losing streak."""
        self._save_checkpoint_now()

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

        R21-R26: one unreadable item (or a listing that itself raises) must
        not hide the others. Each failure lands in
        ``node_status["snapshot_errors"]`` as ``"<kind> <id>: <ExceptionClass>"``
        and is logged once per process; a clean capture adds no key.
        """
        positions: dict[str, object] = {}
        orders: dict[str, object] = {}
        balance: dict[str, object] = {}
        snapshot_errors: list[str] = []

        cache = getattr(self._trading_node, "cache", None)
        portfolio = getattr(self._trading_node, "portfolio", None)
        if cache is not None:
            try:
                open_positions = list(cache.positions_open())
            except Exception as exc:
                self._log_snapshot_failure("positions", exc, snapshot_errors)
            else:
                for index, pos in enumerate(open_positions, start=1):
                    try:
                        position_id = str(pos.id.value)
                    except Exception as exc:
                        self._log_snapshot_failure(f"position #{index}", exc, snapshot_errors)
                        continue
                    try:
                        data = pos.to_dict()
                    except Exception as exc:
                        self._log_snapshot_failure(f"position {position_id}", exc, snapshot_errors)
                        continue
                    try:
                        data["unrealized_pnl"] = self._unrealized_pnl(portfolio, pos)
                    except Exception as exc:
                        data["unrealized_pnl"] = None
                        self._log_snapshot_failure(
                            f"unrealized_pnl {position_id}", exc, snapshot_errors
                        )
                    data["leverage"] = self._resolved.leverage if self._resolved else None
                    positions[position_id] = data
                # R21/R22: the live count, so one unserializable position keeps the
                # others and the API still reports how many the venue holds. Only a
                # raising listing leaves the previous value untouched.
                self._status.positions = len(open_positions)

            try:
                open_orders = list(cache.orders_open())
            except Exception as exc:
                self._log_snapshot_failure("orders", exc, snapshot_errors)
            else:
                for index, order in enumerate(open_orders, start=1):
                    try:
                        order_id = str(order.client_order_id.value)
                    except Exception as exc:
                        self._log_snapshot_failure(f"order #{index}", exc, snapshot_errors)
                        continue
                    try:
                        orders[order_id] = order.to_dict()
                    except Exception as exc:
                        self._log_snapshot_failure(f"order {order_id}", exc, snapshot_errors)

            try:
                open_accounts = list(cache.accounts())
            except Exception as exc:
                self._log_snapshot_failure("accounts", exc, snapshot_errors)
            else:
                for index, account in enumerate(open_accounts, start=1):
                    try:
                        account_id = str(account.id.value)
                    except Exception as exc:
                        self._log_snapshot_failure(f"account #{index}", exc, snapshot_errors)
                        continue
                    try:
                        balance[account_id] = self._serialize_account_balance(account)
                    except Exception as exc:
                        self._log_snapshot_failure(f"account {account_id}", exc, snapshot_errors)

        node_status: dict[str, object] = dict(self._status.to_dict())
        node_status["trader_id"] = self._config.trader_id
        node_status["testnet"] = self._config.binance.testnet
        # 3fqw9: the account identity (secret-free fingerprint) lets the next session
        # on this account find the newest checkpoint whatever its trader_id.
        node_status["account"] = binance_account_id(self._config.binance)
        if self._guard is not None:
            guard_status = self._guard.status()
            node_status["risk"] = guard_status["risk"]
            node_status["guard"] = guard_status
        elif self._restored_risk is not None:
            # H19: no guard yet (failed restart) - persist the restored risk
            # bookkeeping instead of wiping it from the checkpoint.
            node_status["risk"] = self._restored_risk.to_dict()
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
        if snapshot_errors:
            node_status["snapshot_errors"] = snapshot_errors

        return StateCheckpoint(
            trader_id=self._config.trader_id,
            positions=positions,
            orders=orders,
            balance=balance,
            node_status=node_status,
        )

    def _log_snapshot_failure(
        self, item: str, exc: BaseException, errors: list[str]
    ) -> None:
        """Record one unserializable checkpoint item; log it once per process.

        Class only: a capture can run under an active exception (a failing
        guard-transition write leaves its error chained), so the exception
        text - and its ``__context__`` - must never reach the log.
        """
        entry = f"{item}: {type(exc).__name__}"
        errors.append(entry)
        if entry in self._snapshot_failures_logged:
            return
        self._snapshot_failures_logged.add(entry)
        logger.error("checkpoint snapshot item failed: %s", entry)

    @staticmethod
    def _unrealized_pnl(portfolio: Any, pos: Any) -> float | None:
        if portfolio is None:
            return None
        pnl = portfolio.unrealized_pnl(pos.instrument_id)
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
            # Capture on the loop, write on the FIFO worker: no DB call on the loop.
            future = self._db_worker.submit(
                self._persistence.save_checkpoint, self._capture_checkpoint()
            )
            future.add_done_callback(self._log_checkpoint_failure)
        except Exception as exc:
            # s3pqx: class only - this now always runs in a finally after a failed
            # guard-transition write (bykze), so an active exception is present and
            # ``logger.exception`` would print that error and its ``__context__`` chain
            # next to this one. Same rule as _check_connectivity / _write_then_alert.
            logger.error("immediate checkpoint failed (%s)", type(exc).__name__)

    @staticmethod
    def _log_checkpoint_failure(future: Future[Any]) -> None:
        """Done-callback of a worker save: class only (see ``_save_checkpoint_now``)."""
        if not future.cancelled() and (exc := future.exception()) is not None:
            logger.error("immediate checkpoint failed (%s)", type(exc).__name__)

    def _mirror_restored_halt(self) -> None:
        """Carry the restored halt into the node status (both restore paths agree).

        ``_restore_from_checkpoint`` writes it so every checkpoint taken before
        ``_run_loop`` keeps the halt (H19); ``_run_loop`` re-asserts it when no guard
        was built to own the state.
        """
        restored = self._restored_halt
        if restored is None:
            return
        self._status.state = NodeState.HALTED
        self._status.halt_reason = restored.reason
        self._status.error_message = restored.message
        self._status.updated_at = datetime.now(UTC)

    def _restore_from_checkpoint(self) -> None:
        """Restore the risk state (and halt) of the previous session.

        The trader's own latest checkpoint (restart / /restore) is read as before.
        The newest checkpoint of ANOTHER trader on the same account that is newer than
        the own one is ALWAYS looked up. Unless ``start_fresh``, it is the source of
        the risk bookkeeping (high water mark, UTC-day baseline, consecutive losses),
        so a new session id cannot reset the drawdown/daily-loss limits. The
        liquidation watch (forceOrders cursors + dedupe keys) is inherited from that
        newer row regardless of ``start_fresh`` -- it is report-only, never a risk
        limit -- merged with the own row's cursors and bounded by the older of the two
        timestamps, so no venue liquidation between the two sessions is skipped. The
        own watch is captured BEFORE the newer-row lookup, so an unreadable newer row
        (which halts ERROR) never discards it.
        Halts: the own halt always wins; from another session only an ERROR halt is
        inherited (an unrecognised reason counts as ERROR). ``start_fresh`` never
        discards the own checkpoint; it resets only the risk/halt inheritance
        (vibe-quant-1ghod).
        Positions and orders come from NautilusTrader's venue reconciliation
        (claimed via ``external_order_claims``); the checkpoint copy is only logged.
        """
        if self._persistence is None:
            return
        account = binance_account_id(self._config.binance)
        try:
            # P-R7/P-R7b: only a row of THIS account (or a legacy row without the key)
            # is ours; the same trader_id on another account (testnet vs live, another
            # key) carries nothing, but its older matching history is still restored.
            own = self._persistence.load_latest_account_checkpoint(
                self._config.trader_id, account
            )
            # W5/1ghod: the OWN watch is captured BEFORE the newer-row lookup, so an
            # unreadable newer row still halts ERROR (W3) without discarding the own
            # cursors and dedupe keys -- a duplicate alert is acceptable, a miss is not.
            if own is not None:
                self._restored_watch_ms = self._checkpoint_ms(own.timestamp)
                self._restored_watch = self._restored_watch_state(
                    own.node_status.get("liquidation_watch")
                )
            # qa42q/ae6ah: the trader's own NEWEST row in the same venue + env (any
            # key) is a key rotation; its risk and halt carry (below), so a rotation
            # can neither re-arm a halted trader nor reset its drawdown/streak
            # bookkeeping, and rotating back A -> B -> A restores B's newer row.
            latest = self._persistence.load_latest_env_checkpoint(
                self._config.trader_id, account
            )
            rotated = None
            if latest is not None and is_key_rotation(latest.node_status.get("account"), account):
                rotated = latest
                logger.warning(
                    "paper %s: key rotation from account %s to %s; "
                    "own risk and halt carried",
                    self._config.trader_id,
                    latest.node_status.get("account"),
                    account,
                )
            newest = self._persistence.load_latest_checkpoint(self._config.trader_id)
            if newest is not None:
                newest_account = newest.node_status.get("account")
                if isinstance(newest_account, str) and account_env(
                    newest_account
                ) != account_env(account):
                    logger.warning(
                        "paper %s: own checkpoint is for account %s, not %s; ignored",
                        self._config.trader_id,
                        newest_account,
                        account,
                    )
            newer = self._persistence.load_newer_account_checkpoint(
                account, self._config.trader_id
            )
            # 1ghod: the lookup ALWAYS runs. ``start_fresh`` only drops the newer row
            # as the RISK/halt source -- the liquidation watch is report-only and is
            # still inherited, so a venue liquidation in the gap is never silent.
            inherited = None if self._config.start_fresh else newer
            source = inherited if inherited is not None else own
            # 1zh0m + 1ghod: the watch source is the newer row whatever start_fresh
            # says, merged over the own row captured above. The newer row wins for a
            # symbol both name; a symbol only the own (older) row names keeps ITS
            # cursor, and every symbol neither names starts at the older of the two
            # timestamps, so no venue liquidation in the gap is skipped. With no own row
            # the watch source bounds the gap by itself.
            if newer is not None:
                if own is None:
                    self._restored_watch_ms = self._checkpoint_ms(newer.timestamp)
                merged_watch = self._restored_watch
                if merged_watch is None:
                    merged_watch = {"cursors": {}, "seen": {}}
                source_watch = self._restored_watch_state(
                    newer.node_status.get("liquidation_watch")
                )
                merged_watch["cursors"].update(source_watch["cursors"])
                merged_watch["seen"].update(source_watch["seen"])
                self._restored_watch = merged_watch
            if rotated is not None:
                # The watch above stays account-scoped; only risk and halt rotate.
                own = rotated
                if inherited is None:
                    source = rotated
            if source is None:
                if not self._config.start_fresh:
                    logger.warning(
                        "paper %s: no prior session for account %s; "
                        "limits start at current equity",
                        self._config.trader_id,
                        account,
                    )
                return
            self._restored_risk, problems = RiskState.from_checkpoint(
                source.node_status.get("risk")
            )
            self._risk_inherited_from = source.trader_id if inherited is not None else None
            self._restored_halt = (
                restored_halt_from_status(own.node_status, own.timestamp.isoformat())
                if own is not None
                else None
            )
            if self._restored_halt is None and inherited is not None:
                other = restored_halt_from_status(
                    inherited.node_status, inherited.timestamp.isoformat()
                )
                if other is not None and other.reason == HaltReason.ERROR:
                    self._restored_halt = RestoredHalt(
                        reason=other.reason,
                        message=inherited_halt_message(inherited.trader_id, other.message),
                        checkpoint_time=other.checkpoint_time,
                    )
        except Exception as exc:
            # R18/R19: an unreadable checkpoint row (failed load / parse, non-dict
            # node_status, or any raise below) fails closed. A non-None risk state and
            # a restored ERROR halt keep BOTH in the failure checkpoint the shutdown
            # writes, so the next restart is HALTED, never a blank RUNNING state (H19).
            # The risk state really is blank here, and the guard re-baselines at the
            # first equity read.
            self._restored_risk = RiskState()
            self._restored_halt = RestoredHalt(
                reason=HaltReason.ERROR,
                message=(
                    f"{RESTORED_HALT_PREFIX}checkpoint unreadable ({type(exc).__name__}); "
                    "limits re-baselined at current equity"
                ),
                checkpoint_time=datetime.now(UTC).isoformat(),
            )
            self._status.state = NodeState.HALTED
            self._status.halt_reason = HaltReason.ERROR
            self._status.error_message = self._restored_halt.message
            self._status.updated_at = datetime.now(UTC)
            logger.error(
                "paper %s: checkpoint unreadable (%s); halting",
                self._config.trader_id,
                type(exc).__name__,
            )
            self._write_event(
                EventType.SIGNAL,
                {
                    "action": "restore",
                    "checkpoint_time": None,
                    "prior_state": None,
                    "prior_halt_reason": None,
                    "prior_positions": [],
                    "prior_orders": [],
                    "restored_risk": self._restored_risk.to_dict(),
                    "restored_halt": HaltReason.ERROR.value,
                    "risk_problems": [],
                    "risk_source_trader_id": None,
                    "risk_source_checkpoint_time": None,
                },
            )
            return
        if self._restored_halt is None and problems:
            # R12/R12m: an unreadable or partly missing risk state restarts HALTED
            # with an ERROR halt through the existing restored-halt path (guard
            # starts HALTED, on_start arms the deferred flatten, the first tick
            # flattens owned+claimed positions and stops strategies). The risk
            # limits are re-baselined at the current equity (the restored values
            # could not be trusted).
            self._restored_halt = RestoredHalt(
                reason=HaltReason.ERROR,
                message=(
                    f"{RESTORED_HALT_PREFIX}unreadable risk state in checkpoint "
                    f"({', '.join(problems)}); limits re-baselined at current equity; "
                    "owned and claimed positions flattened"
                ),
                checkpoint_time=source.timestamp.isoformat(),
            )
        if problems:
            # R12/R13: one logger.error, values only (never exc_info - this can run
            # under an active fetch error; see _check_connectivity / _write_then_alert).
            logger.error(
                "paper %s: unreadable risk state in checkpoint (%s); halting",
                self._config.trader_id,
                ", ".join(problems),
            )
        if self._restored_halt is not None:
            # H19: every checkpoint written before _run_loop keeps the halt.
            self._mirror_restored_halt()
        self._write_event(
            EventType.SIGNAL,
            {
                "action": "restore",
                "checkpoint_time": own.timestamp.isoformat() if own is not None else None,
                "prior_state": own.node_status.get("state") if own is not None else None,
                "prior_halt_reason": (
                    own.node_status.get("halt_reason") if own is not None else None
                ),
                "prior_positions": sorted(own.positions) if own is not None else [],
                "prior_orders": sorted(own.orders) if own is not None else [],
                "restored_risk": self._restored_risk.to_dict(),
                "restored_halt": (
                    self._restored_halt.reason.value if self._restored_halt else None
                ),
                "risk_problems": list(problems),
                "risk_source_trader_id": source.trader_id,
                "risk_source_checkpoint_time": source.timestamp.isoformat(),
            },
        )
        if self._risk_inherited_from is not None:
            # P-R8: a new trader id silently arming the old limits would surprise.
            logger.warning(
                "paper %s: inherited risk state from %s (high water mark %s, "
                "%d consecutive losses); a limit breach halts at the first equity read; "
                "start fresh skips this other trader id's risk state; "
                "this trader id's own saved limits and halt still carry",
                self._config.trader_id,
                self._risk_inherited_from,
                self._restored_risk.high_water_mark,
                self._restored_risk.consecutive_losses,
            )
        logger.info(
            "restored risk state for %s from checkpoint %s of %s: %s",
            self._config.trader_id,
            source.timestamp.isoformat(),
            source.trader_id,
            self._restored_risk.to_dict(),
        )

    # -------------------------------------------------------------- events

    def _write_event(
        self,
        event_type: EventType,
        data: dict[str, object],
        timestamp: datetime | None = None,
    ) -> Future[None] | None:
        """Queue an event for the paper event log, preserving the full payload.

        The line is appended on the event-log worker (ogd37): the returned future
        resolves when it is on disk (or carries the write error / ``EventLogFullError``),
        ``None`` when there is no writer. The timestamp is taken here, at submit.

        The payload is deep-copied here, at submit: the worker serializes the event
        later, so a caller that mutates its dict after this returns (a new key, a new
        value, a nested list/dict) must not change the written line.

        Position events use the typed classes (their fields are the payload);
        everything else is written as a base ``Event`` because the typed
        SIGNAL/RISK_CHECK classes drop unknown keys (and default ``passed``
        to True, which would record a halt as a passed risk check).
        """
        if self._event_writer is None:
            return None
        strategy_name = self._strategy.name if self._strategy else "unknown"
        ts = timestamp or datetime.now(UTC)
        payload = copy.deepcopy(data)
        event: Event
        if event_type in (EventType.POSITION_OPEN, EventType.POSITION_CLOSE):
            event = create_event(
                event_type=event_type,
                run_id=self._config.trader_id,
                strategy_name=strategy_name,
                data=payload,
                timestamp=ts,
            )
        else:
            event = Event(
                timestamp=ts,
                event_type=event_type,
                run_id=self._config.trader_id,
                strategy_name=strategy_name,
                data=payload,
            )
        return self._event_log.submit(self._event_writer, event)

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

        self._kill_monitor.start()
        snap = await self._kill_monitor.wait_first_read(KILL_SWITCH_STARTUP_TIMEOUT_S)
        if snap.last_ok_mono is None:
            raise ConfigurationError(
                f"System kill switch state unreadable ({snap.error_class or 'no read within 10 s'});"
                " refusing to start"
            )
        if snap.engaged:
            raise ConfigurationError(
                f"System kill switch engaged ({snap.reason or 'no reason'}); refusing to start"
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
                "max_margin_pct": float(resolved.max_margin_pct),
                "start_fresh": self._config.start_fresh,
                "account": binance_account_id(self._config.binance),
            },
        )
        self._warn_margin_cushion(resolved)
        # ogd37: the first lines must land, or the log is unwritable and start is refused
        # (what a synchronous write used to do by raising).
        if not await wait_future(self._event_log.barrier(), EVENT_STARTUP_WAIT_S):
            raise ConfigurationError(
                f"paper event log not writable within {EVENT_STARTUP_WAIT_S:g} s; "
                "refusing to start"
            )
        log_health = self._event_log.health()
        if log_health.failed > 0:
            raise ConfigurationError(
                f"paper event log not writable ({log_health.last_error_class}); "
                "refusing to start"
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

        # u2r9r: a configured-but-broken Telegram silently loses every CRITICAL
        # alert (halt, kill switch, liquidation), so the start is refused. The
        # check runs before the persistence/restore step and before the NT trading
        # node exists, so no strategy exists and 0 orders are structurally possible.
        try:
            telegram_config = await require_working_telegram()
        except TelegramUnavailableError as exc:
            # str(exc) is token-free by construction; `from None` keeps the httpx
            # error (whose str() holds the token URL) out of the chain.
            raise ConfigurationError(str(exc)) from None
        if telegram_config is None:
            logger.info("Telegram alerts disabled (TELEGRAM_BOT_TOKEN/CHAT_ID not set)")
        else:
            self._telegram = TelegramBot(telegram_config)

        # Persistence first: restore risk state before the guard is built.
        self._persistence = StatePersistence(
            db_path=self._config.db_path,
            trader_id=self._config.trader_id,
            checkpoint_interval=self._config.state_persistence_interval,
        )
        self._restore_from_checkpoint()
        self._commands = PaperCommandQueue(self._config.db_path)
        # jf7c3 J2: rows a previous process left 'running' can never reach an outcome, so
        # they are failed before this process claims anything - on the queue's own worker
        # thread, like every other queue call - and never at the cost of the start.
        try:
            swept = await self._cmd_worker.run(
                self._commands.fail_stale_running,
                self._config.trader_id,
                self._STALE_RUNNING,
            )
        except Exception as exc:
            logger.error("stale command sweep failed (%s)", type(exc).__name__)
        else:
            if swept >= 1:
                logger.warning("failed %d stale running paper command(s) at startup", swept)

        self._trading_node = self._create_live_trading_node()
        # No await between the guard's creation and the replay, so no signal can land
        # in between; the init checkpoint below then carries the halt.
        self._apply_startup_halt()
        await self._db_worker.run(self._persistence.save_checkpoint, self._capture_checkpoint())
        # k8ygz: an API halt/kill (or the system kill cascade) queued during start-up
        # reaches the guard before NT starts any strategy; this is the last await
        # before run_async. A pending resume stops the drain there: its halt belongs
        # to the control loop (a later operator action must win, in queue order).
        try:
            await self.process_commands(startup_halts_only=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("start-up command drain failed (%s)", type(exc).__name__)

    async def _run_loop(self) -> None:
        """Main event loop for paper trading.

        Delegates execution lifecycle to NautilusTrader TradingNode.
        """
        if self._trading_node is None:
            raise RuntimeError("Trading node is not initialized")

        guard = self._guard
        if guard is not None:
            # The guard is the authority: a halt it took at start-up (restored, or a
            # replayed SIGUSR1) must not be reset to "running" here.
            self._status.state = _GUARD_TO_NODE_STATE[guard.guard_state]
            if guard.guard_state == GuardState.HALTED:
                self._status.halt_reason = guard.halt_reason
                self._status.error_message = guard.message
            else:
                self._status.halt_reason = None
        elif self._restored_halt is not None:
            self._mirror_restored_halt()
        else:
            self._status.state = NodeState.RUNNING
        self._status.started_at = datetime.now(UTC)
        self._status.updated_at = datetime.now(UTC)

        self._write_event(
            EventType.SIGNAL,
            {
                "action": "state_change",
                "from_state": NodeState.INITIALIZING.value,
                "to_state": self._status.state.value,
            },
        )

        if self._persistence is not None:
            await self._persistence.start_periodic_checkpointing(
                self._capture_checkpoint, self._db_worker.executor
            )

        self._control_task = asyncio.create_task(self._control_loop())
        self._kill_watch_task = asyncio.create_task(self._kill_watch_loop())
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
            await self._stop_kill_watch()
            await self._stop_liquidation_loop()
            self._stop_kill_monitor()

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

    async def _stop_kill_watch(self) -> None:
        task = self._kill_watch_task
        self._kill_watch_task = None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    def _stop_kill_monitor(self) -> None:
        """Signal the reader thread and wait a bounded time (idempotent, never raises)."""
        try:
            self._kill_monitor.stop()
        except Exception as exc:
            logger.error("kill-switch monitor stop failed (%s)", type(exc).__name__)

    async def _final_checkpoint(self, persistence: StatePersistence) -> None:
        """R19c: submit the final save and wait at most ``SHUTDOWN_CHECKPOINT_WAIT_S`` for it.

        A hung store costs this wait, not the shutdown: the trading engine stops either way.
        """
        try:
            future = self._db_worker.submit(persistence.save_checkpoint, self._capture_checkpoint())
        except Exception as exc:
            logger.error("final checkpoint not submitted (%s)", type(exc).__name__)
            return
        if not await wait_future(future, SHUTDOWN_CHECKPOINT_WAIT_S):
            logger.error(
                "final checkpoint not saved within %.0f s; continuing shutdown",
                SHUTDOWN_CHECKPOINT_WAIT_S,
            )
        elif (final_exc := future.exception()) is not None:
            logger.error("final checkpoint failed (%s)", type(final_exc).__name__)

    async def _close_store(
        self, worker: StoreWorker, close: Callable[[], None], label: str
    ) -> None:
        """R19d: the store's own worker closes it after its last job; wait a bounded time.

        Nothing closes the connection from the loop. On a hung statement the wait ends
        after ``STORE_CLOSE_WAIT_S`` and the worker closes once the statement returns.
        A raising close is logged class only.
        """

        def guarded_close() -> None:
            try:
                close()
            except Exception as exc:
                logger.error("%s close failed (%s); continuing shutdown", label, type(exc).__name__)

        if not await wait_future(worker.begin_stop(guarded_close), STORE_CLOSE_WAIT_S):
            logger.error("%s still busy; its worker closes it when the job returns", label)

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
        await self._stop_kill_watch()
        self._stop_kill_monitor()
        # R53: before the final checkpoint, so the watch state it captures is still there.
        await self._stop_liquidation_loop()

        if self._persistence is not None:
            persistence = self._persistence
            with contextlib.suppress(Exception):
                await persistence.stop_periodic_checkpointing()
            # 18def66b680c20f8-4926: nothing before stop_async may raise past here.
            try:
                await self._final_checkpoint(persistence)
                await self._close_store(
                    self._db_worker, persistence.close, "paper checkpoint store"
                )
            except Exception as exc:
                logger.error(
                    "paper checkpoint store shutdown failed (%s); continuing shutdown",
                    type(exc).__name__,
                )
            self._persistence = None

        # otrm3: like the R6 writer close below, a raising ``close()`` is caught and
        # logged class only, so the trading engine below is still stopped; class only
        # (R1b) keeps the active exception's ``__context__`` chain (signed URL / api
        # key text) and its traceback out of the log.
        if self._commands is not None:
            try:
                await self._close_store(
                    self._cmd_worker, self._commands.close, "paper command queue"
                )
            except Exception as exc:
                logger.error(
                    "paper command queue shutdown failed (%s); continuing shutdown",
                    type(exc).__name__,
                )
            self._commands = None

        # Stop trading node. ``TradingNode.dispose()`` stops the running
        # event loop (it was designed for ``run()`` which owns its own loop),
        # so we intentionally skip it.
        if self._trading_node is not None:
            try:
                await self._trading_node.stop_async()
            except Exception as exc:
                # R27: class only (the stop can run under an active exception), and
                # the rest of teardown - event writer, telegram, state manager - runs.
                logger.error(
                    "trading node stop failed (%s); continuing shutdown", type(exc).__name__
                )
            self._trading_node = None

        # ogd37: after stop_async (events NT emits while stopping are kept). Later
        # ``_write_event`` calls are no-ops; the event worker flushes and closes the file
        # after its last queued line. R6 (k8oei): a close() failure is logged by class and
        # the remaining teardown (telegram, state manager) still runs.
        writer = self._event_writer
        self._event_writer = None

        def close_writer() -> None:
            if writer is None:
                return
            with contextlib.suppress(Exception):
                writer.flush()
            writer.close()

        try:
            await self._close_store(self._event_log.worker, close_writer, "paper event writer")
        except Exception as exc:
            logger.error(
                "paper event writer shutdown failed (%s); continuing shutdown",
                type(exc).__name__,
            )

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
            if self._status.halt_reason is not None:
                # H19b (oe9yc): a restored/live halt owns the status message. Store
                # the failure class only (never the exception text) so the halted
                # message survives into the failure checkpoint and the next restart.
                logger.error("paper node start failed while halted (%s)", type(e).__name__)
            else:
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
        # Before the guard exists there is nothing to flatten: record the halt. The guard
        # applies it when it is built (_apply_startup_halt), through its own severity rule,
        # flatten, event, single alert and checkpoint. No event write, alert or DB call
        # here: the alert would be a no-op before Telegram is built (W1-W3).
        record = self._prestart.record(
            _NODE_TO_GUARD.get(self._status.state),
            self._status.halt_reason,
            reason,
            msg,
        )
        if not record:
            return
        self._status.state = NodeState.HALTED
        self._status.halt_reason = reason
        self._status.error_message = msg
        self._status.updated_at = datetime.now(UTC)
        logger.warning(
            "paper %s: %s halt during start-up recorded; the trading guard applies it when built",
            self._config.trader_id,
            reason.value,
        )

    def _apply_startup_halt(self) -> None:
        """Replay a halt recorded before the guard existed through the new guard."""
        pending = self._prestart.take()
        if pending is not None and self._guard is not None:
            self._guard.halt(*pending)

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
        snap = self._kill_monitor.snapshot()
        age = snap.age(self._mono())
        if snap.error_class is not None or age > KILL_SWITCH_RESUME_FRESH_S:
            detail = snap.error_class or (
                "no clean read" if math.isinf(age) else f"{age:.1f} s old"
            )
            return False, (
                f"system kill switch state not fresh ({detail}); "
                "cannot resume until it reads cleanly"
            )
        engaged = snap.engaged
        if self._guard is not None:
            return self._guard.resume(kill_switch_engaged=engaged)
        if engaged:
            return False, "system kill switch is engaged; unlock it before resuming"
        node_state = _NODE_TO_GUARD.get(self._status.state)
        ok, text, target = self._prestart.resume(
            node_state,
            self._status.halt_reason,
            kill_switch_engaged=engaged,
        )
        if not ok or target is None:
            return ok, text
        previous = self._status.state
        self._status.state = _GUARD_TO_NODE_STATE[target]
        self._status.updated_at = datetime.now(UTC)
        payload: dict[str, object] = {
            "action": "state_change",
            "from_state": previous.value,
            "to_state": self._status.state.value,
        }
        if previous == NodeState.HALTED:
            self._status.halt_reason = None
            self._status.error_message = None
            payload["resumed"] = True
        self._write_event(EventType.SIGNAL, payload)
        return True, text

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
