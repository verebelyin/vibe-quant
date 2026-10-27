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
import math
import signal
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, NoReturn, Protocol

from vibe_quant.alerts.telegram import (
    TelegramBot,
    TelegramUnavailableError,
    require_working_telegram,
)
from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.fees import FeeSchedule
from vibe_quant.liquidation_settings import LiquidationConfigError, LiquidationSettings
from vibe_quant.logging.events import EventType
from vibe_quant.logging.writer import EventWriter
from vibe_quant.margin_limit import DEFAULT_PAPER_MAX_MARGIN_PCT
from vibe_quant.paper import lineage
from vibe_quant.paper.admission import (
    Admitted,
    Refusal,
    RefusalCode,
    StartFacts,
    StartIntent,
    check,
)
from vibe_quant.paper.command_channel import (
    CHECKPOINT_BLOCK_MESSAGE,
    COMMAND_CLAIM_FAILURE_LIMIT,
    COMMAND_CLAIM_STALL_S,
    RESUME_OWED_MESSAGE,
    STALL_LABEL,
    STARTUP_CHECKPOINT_STEP,
    STARTUP_SWEEP_STEP,
    CommandChannelHealth,
    block_message,
    failed_alert_text,
    guard_state_text,
    restored_alert_text,
    startup_checkpoint_failed_alert_text,
    startup_stall_alert_text,
)
from vibe_quant.paper.config import (
    DEFAULT_MARGIN_TYPE,
    DEFAULT_VALIDATION_LEVERAGE,
    ConfigurationError,
    PaperTradingConfig,
    check_paper_exposure,
    effective_max_margin_pct,
    effective_max_position_pct,
)
from vibe_quant.paper.config import (
    account_id as binance_account_id,
)
from vibe_quant.paper.errors import ErrorContext, ErrorHandler
from vibe_quant.paper.event_log import EventLogWorker
from vibe_quant.paper.force_orders import build_binance_fetch
from vibe_quant.paper.guard import (
    ENTRY_CHECK_FAILED_PREFIX,
    MARGIN_DENIAL_PREFIX,
    UNOWNED_PREFIX,
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
from vibe_quant.paper.journal import Alert, SessionJournal
from vibe_quant.paper.kill_switch_monitor import (
    KILL_SWITCH_RESUME_FRESH_S,
    KILL_SWITCH_STALE_LIMIT_S,
    KILL_SWITCH_STARTUP_TIMEOUT_S,
    KILL_SWITCH_WATCH_TICK_S,
    KillSwitchMonitor,
)

# The halt helpers live in lineage beside resolve(); these two names stay importable
# from here (tests and operators do ``from vibe_quant.paper.node import ...``).
from vibe_quant.paper.lineage import (
    inherited_halt_message as inherited_halt_message,
)
from vibe_quant.paper.lineage import (
    restored_halt_from_status as restored_halt_from_status,
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
    from collections.abc import Awaitable, Callable
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
# ogd37: event-log appends run on their own worker. Start waits at most
# EVENT_STARTUP_WAIT_S for the first lines (an unwritable log refuses to start).
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

#: The node's one refusal -> ConfigurationError mapping (rabmo.10b R3): today's
#: startup texts, byte for byte. ``{reason}`` falls back to ``no reason`` when the
#: engaged text is empty; ``{detail}`` is the fact admission reported. Codes the
#: node cannot produce are absent on purpose: they assert instead of defaulting.
_NODE_REFUSAL_TEXT: dict[RefusalCode, str] = {
    RefusalCode.KILL_UNREADABLE: (
        "System kill switch state unreadable ({detail}); refusing to start"
    ),
    RefusalCode.KILL_ENGAGED: "System kill switch engaged ({reason}); refusing to start",
    RefusalCode.TELEGRAM_UNAVAILABLE: "{detail}",
}

#: The startup stages take no consent: the node has no live/start_fresh flags.
_NO_CONSENT_START = StartIntent(
    live=False, confirm_live=False, start_fresh=False, confirm_fresh_live=False
)


def _raise_start_refusal(refusal: Refusal) -> NoReturn:
    """Map admission's Refusal onto the node's ConfigurationError (R3)."""
    template = _NODE_REFUSAL_TEXT.get(refusal.code)
    if template is None:
        raise AssertionError(f"node never produces {refusal.code}")
    error = ConfigurationError(
        template.format(detail=refusal.detail, reason=refusal.detail or "no reason")
    )
    if refusal.code is RefusalCode.TELEGRAM_UNAVAILABLE:
        # u2r9r: explicit suppression, so a handler running inside another handled
        # exception never chains it (or the httpx token URL) into this refusal.
        raise error from None
    raise error


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


def validated_run_strategy_params(
    run: dict[str, Any], config_fields: tuple[str, ...] | None = None
) -> dict[str, object]:
    """Strategy overrides exactly as ``ValidationRunner`` builds them for this run.

    With ``config_fields`` (the compiled config's ``__struct_fields__``) the stored keys are
    resolved onto those fields by validation's own resolver, so ``stop_loss.percent`` reaches
    ``stop_loss_percent`` here exactly as it did in the validation run, and a key validation
    would refuse raises ``ValidationRunnerError`` instead of being dropped. The run's sizing
    keys are kept under their stored names (paper inherits them, validation merges them from
    the parsed launch) and so do the backtest-only keys (paper's own filter drops them); the
    other run-level keys are validation's own and are left out.
    Without ``config_fields`` the raw stored keys are returned.
    """
    from vibe_quant.sizing_settings import SIZING_FIELDS
    from vibe_quant.validation.runner import ValidationRunner, _resolve_config_params

    # _build_strategy_params only reads run_config; reuse it so paper can never
    # drift from what validation applied. (No ValidationRunner state needed.)
    runner = ValidationRunner.__new__(ValidationRunner)
    raw = runner._build_strategy_params(run)
    if config_fields is None:
        return raw
    # The runner-owned and backtest-only keys are not strategy overrides: they never reach
    # the resolver (it refuses the runner-owned fields) and paper filters them out below,
    # but they stay in the result so the "not applied" log names them.
    kept = {k: v for k, v in raw.items() if k in SIZING_FIELDS or k in _BACKTEST_ONLY_PARAMS}
    resolved = _resolve_config_params(
        {k: v for k, v in raw.items() if k not in _BACKTEST_ONLY_PARAMS}, config_fields
    )
    resolved.update(kept)
    return resolved


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
        # T13a: every event write and Telegram alert this node makes goes through the
        # journal.
        self._journal = SessionJournal(config.trader_id, EventLogWorker("vq-paper-events"))
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
        #: 3fg9d: consecutive failed command claims and who owns the entry block.
        self._cmd_channel = CommandChannelHealth()
        #: The hung start-up checkpoint failed late: a later successful save lifts its block.
        self._startup_checkpoint_failed = False
        #: 1A: after that late failure entries stay blocked until an OPERATOR resume.
        self._startup_resume_owed = False
        self._checkpoint_loop: asyncio.AbstractEventLoop | None = None
        #: 3fg9d: a control-loop claim that gave no answer in time; polled again next tick
        #: (never a second claim queued behind it).
        self._claim_inflight: Future[list[PaperCommand]] | None = None
        #: 3fg9d: claimed rows after one whose outcome write raised; run first on the next
        #: tick, so no later recovery lifts the entry block past a queued halt/kill.
        self._unrun_claimed: list[PaperCommand] = []
        self._kill_watch_task: asyncio.Task[None] | None = None
        self._trades_day: date | None = None
        # R26: entries already logged, so a repeated snapshot failure logs once.
        self._snapshot_failures_logged: set[str] = set()

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
        is_connectivity = context.operation == "connectivity"
        # oledr/k8oei: route through journal.record(alert=...) so the alert goes out even
        # when the write raises; the write's own error then propagates unchanged.
        self._journal.record(
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
            alert=Alert(
                "error",
                f"{alert_type}: {context.error}\nOp: {context.operation}"
                + (f"\nSymbol: {context.symbol}" if context.symbol else ""),
                # wi1m0: "venue connection lost" pairs with "restored"; it must reach the
                # chat after an unrelated recent ERROR (bounded once per 60 s by the journal).
                critical=is_connectivity,
                monitor="venue:lost" if is_connectivity else None,
            ),
        )

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
            self._alert_price(cache, event.instrument_id, entry) if entry is not None else "unknown"
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
            from vibe_quant.validation.runner import ValidationRunnerError

            try:
                raw = validated_run_strategy_params(
                    run, tuple(getattr(config_cls, "__struct_fields__", ()))
                )
            except ValidationRunnerError as exc:
                raise ConfigurationError(
                    f"Validation run {cfg.validation_run_id} has parameters the strategy "
                    f"config cannot take: {exc}"
                ) from exc
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
        self._journal.record(
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
                logger.warning("resume via SIGUSR2 not applied: %s", msg)

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
                self._journal.check_health()
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
        # wi1m0/C5: a venue state held back by its 60 s window goes out on this tick.
        self._journal.flush_monitors()
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
                # its ``__context__`` chain. Same rule as journal.record(alert=...) (R1b).
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
                # R42c: the venue set up late; say so once. Critical (bounded per 60 s by
                # the journal): the startup-timeout ERROR it answers may be recent (wi1m0).
                if self._status.error_message == self._startup_timeout_message():
                    # R42d: clear only our own message, never a later halt message.
                    self._status.error_message = None
                    self._status.updated_at = self._now()
                # oledr/k8oei: the same helper, so a failing write cannot drop the
                # only signal that the venue came up late.
                self._journal.record(
                    EventType.RISK_CHECK,
                    {"action": "venue_connected_after_startup_timeout", "passed": True},
                    alert=Alert(
                        "error",
                        f"paper {self._config.trader_id} venue connected after startup timeout",
                        critical=True,
                        monitor="venue:restored",
                    ),
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
            self._journal.record(
                EventType.SIGNAL,
                {"action": "reconnected"},
                # Critical: the 60 s per-type limit would drop it right after the
                # disconnect's own ERROR alert, leaving the operator on "lost" (wi1m0).
                alert=Alert(
                    "error", "venue connection restored", critical=True, monitor="venue:restored"
                ),
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
            self._journal.record(
                EventType.RISK_CHECK,
                {
                    "action": "startup_connect_timeout",
                    "timeout_s": STARTUP_CONNECT_TIMEOUT_S,
                    "margin_type": (
                        self._resolved.margin_type if self._resolved else DEFAULT_MARGIN_TYPE
                    ),
                    "passed": False,
                },
                alert=Alert("error", message, critical=True),
            )

    # ----------------------------------------------- liquidation watch (T6)

    def _now_ms(self) -> int:
        """Wall clock in epoch milliseconds (R50: one seam for the watch cursors)."""
        return int(self._now().timestamp() * 1_000)

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
                await self._journal.record_durable(
                    EventType.RISK_CHECK,
                    {
                        "action": "liquidation_watch_failed",
                        "message": message,
                        "passed": False,
                    },
                    Alert(
                        "error", f"{message}; exchange liquidations are NOT being reported", True
                    ),
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
            cursors = lineage.watch_cursors(restored, self._restored_watch_ms, resolved.symbols)
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
        symbols = self._resolved.symbols if self._resolved is not None else []
        return {
            "cursors": lineage.watch_cursors(restored, self._restored_watch_ms, symbols),
            "seen": dict(restored["seen"]),
        }

    async def _on_exchange_liquidation(self, order: ForceOrder) -> None:
        """Report one exchange-initiated close: one event, one CRITICAL alert (R47).

        R47b: a missing or failing event writer RAISES, so the watcher fails that
        symbol's cycle (its cursor stays put) and delivers the row on the next one.
        R1/irdig: a failing write does not cost the alert - it is sent, with this
        same text, before the writer's own exception is re-raised (a duplicate alert
        on the retry is accepted). Nothing is swallowed here and nothing is ordered:
        paper cannot reproduce the venue's close, it can only say that it happened.
        """
        if not self._journal.has_writer:
            raise RuntimeError("cannot report an exchange liquidation without an event writer")
        margin_type = (
            self._resolved.margin_type if self._resolved is not None else DEFAULT_MARGIN_TYPE
        )
        # R47e: the event's own vocabulary ('isolated'/'cross'), not the venue's
        # 'ISOLATED'/'CROSSED' margin type.
        mode = "isolated" if margin_type == "ISOLATED" else "cross"
        await self._journal.record_durable(
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
            Alert(
                "error",
                f"EXCHANGE {order.auto_close_type} {order.symbol} {order.side} "
                f"qty {order.executed_qty} @ {order.avg_price} (order {order.order_id}); "
                f"{self._liquidation_scope(order.symbol)}; NO paper action taken",
                True,
            ),
            timestamp=_ns_to_dt(order.ts_ns),
        )

    def _liquidation_scope(self, symbol: str) -> str:
        """toqng: a carried symbol's alert says the position predates this session."""
        resolved = self._resolved
        if resolved is not None and symbol not in resolved.symbols:
            return "not traded this session (position left by an earlier session)"
        return "paper position state may be stale"

    async def _on_liquidation_warning(self, message: str) -> None:
        """R52: a degraded detector warns critically (bounded per 60 s); a clamp does not.

        The degraded notice is sent once per episode, at the failure that opens it, so
        a limiter drop after an unrelated recent ERROR would lose it for good (wi1m0).
        """
        degraded = message.startswith("liquidation detector degraded:")
        await self._journal.record_durable(
            EventType.RISK_CHECK,
            {"action": "liquidation_watch_degraded", "message": message, "passed": False},
            Alert(
                "error",
                f"WARNING: {message}",
                critical=degraded,
                monitor="liquidation:degraded" if degraded else None,
            ),
        )

    async def _on_liquidation_recovered(self, symbol: str, failures: int) -> None:
        """R52: the first success after a degraded run is reported once."""
        await self._journal.record_durable(
            EventType.RISK_CHECK,
            {
                "action": "liquidation_watch_recovered",
                "symbol": symbol,
                "failures": failures,
                "passed": True,
            },
            # Critical: sent within 60 s of the degraded alert sharing this ERROR window,
            # so the limit would drop it and leave the operator on "degraded" (wi1m0).
            Alert(
                "error",
                f"liquidation detector recovered: {symbol} after {failures} failures",
                critical=True,
                monitor="liquidation:recovered",
            ),
        )

    async def process_commands(self, *, startup_halts_only: bool = False) -> None:
        """Execute queued operator commands for this trader.

        ``startup_halts_only`` is the pre-NT drain: only the leading halts/kills that
        no pending resume precedes are claimed (see ``claim_startup_halts``).
        """
        commands = self._commands
        if commands is None:
            return
        if self._unrun_claimed:
            # 3fg9d: rows claimed earlier whose predecessor's outcome write raised.
            leftover, self._unrun_claimed = self._unrun_claimed, []
            await self._run_claimed(commands, leftover)
        claim_fn = commands.claim_startup_halts if startup_halts_only else commands.claim_pending
        claim = self._claim_inflight
        if claim is None:
            claim = self._cmd_worker.submit(claim_fn, self._config.trader_id)
        try:
            # 3fg9d: a claim that gives no answer (busy DB lock, NFS) counts as a failed
            # claim after COMMAND_CLAIM_STALL_S; its future is kept and polled again
            # next tick, so its rows are executed (never left running) once it returns.
            # wi1m0(b): the start-up drain is bounded the same way - a hung drain blocks
            # entries at its first failure (limit 1) and alerts, instead of holding the
            # start with no word to the operator.
            self._claim_inflight = claim
            if not await wait_future(claim, COMMAND_CLAIM_STALL_S):
                self._on_claim_failure(STALL_LABEL, startup=startup_halts_only)
                return
            self._claim_inflight = None
            claimed: list[PaperCommand] = claim.result()
        except asyncio.CancelledError:
            # The worker may still finish the claim: fail those rows (never leave 'running').
            self._claim_inflight = None
            self._cmd_worker.submit(self._fail_claimed, commands, claim, [])
            raise
        except Exception as exc:
            # 3fg9d: a queued halt/kill is unreadable while claims fail. Count it (class
            # only), block entries once the streak is long enough, then let the caller log.
            self._claim_inflight = None
            self._on_claim_failure(type(exc).__name__, startup=startup_halts_only)
            raise
        self._cmd_channel.record_success()
        await self._run_claimed(commands, claimed)
        self._finish_channel_recovery()

    async def _run_claimed(self, commands: PaperCommandQueue, claimed: list[PaperCommand]) -> None:
        """Execute claimed commands in order and write each outcome.

        An outcome write that raises leaves the rows after it ``running`` with nobody to
        run them: they are kept in ``_unrun_claimed`` and run first on the next tick
        (3fg9d), so a later recovery never lifts the entry block past a queued halt/kill.
        """
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
            except Exception:
                self._unrun_claimed = claimed[index + 1 :]
                raise
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

    def _on_claim_failure(self, error: str, *, startup: bool) -> None:
        """Count one failed claim; at the limit block entries and alert once per streak.

        The start-up drain blocks on its first failure (a kill queued for this start is
        unreadable and NT is about to trade). Never raises: the claim error must reach
        the caller unchanged.
        """
        channel = self._cmd_channel
        limit = 1 if startup else COMMAND_CLAIM_FAILURE_LIMIT
        first = channel.record_failure(error, limit=limit)
        if not channel.blocking:
            return
        blocked_now = self._block_entries_for_channel()
        if not first:
            return
        guard = self._guard
        entries, entries_text = self._entries_after_block(blocked_now)
        state_text = (
            guard_state_text(guard.guard_state, guard.halt_reason) if guard is not None else "none"
        )
        try:
            self._journal.record(
                EventType.RISK_CHECK,
                {
                    "action": "command_channel_failed",
                    "failures": channel.failures,
                    "error_type": error,
                    "entries": entries,
                    "state": state_text,
                    "passed": False,
                },
                alert=Alert(
                    "error",
                    failed_alert_text(
                        self._config.trader_id, channel.failures, error, entries_text
                    ),
                    critical=True,
                ),
            )
        except Exception as exc:
            logger.error("command channel alert failed (%s)", type(exc).__name__)

    def _entries_after_block(self, blocked_now: bool) -> tuple[str, str]:
        """What became of new entries after a block attempt: (event value, alert words)."""
        guard = self._guard
        guard_state: GuardState | None = guard.guard_state if guard is not None else None
        if blocked_now:
            return "blocked", "new entries blocked"
        if guard is None or guard_state == GuardState.RUNNING:
            # No guard, or the pause failed on a RUNNING one: say so, never claim a block.
            why = "no trading guard" if guard is None else "pause failed"
            return "not_blocked", f"new entries NOT blocked ({why})"
        state = guard_state_text(guard.guard_state, guard.halt_reason)
        return "already_blocked", f"entries already blocked ({state})"

    def _lift_channel_pause(self) -> None:
        """Lift the pause this channel set once neither the channel nor the unsaved
        initial checkpoint still needs it (see ``_finish_channel_recovery``)."""
        channel = self._cmd_channel
        guard = self._guard
        if not channel.owns_pause or guard is None:
            return
        if channel.blocking or channel.checkpoint_blocking or self._startup_resume_owed:
            return
        state: GuardState = guard.guard_state
        if state == GuardState.PAUSED:
            ok, _text = self.resume()
            if ok:
                channel.owns_pause = False
        elif state == GuardState.RUNNING:
            channel.owns_pause = False

    def _finish_channel_recovery(self) -> None:
        """After a claim succeeded AND its batch ran: lift the channel's own pause, report.

        Only the pause this channel set (``owns_pause``) is ever lifted, and only from
        PAUSED: a halt stays (the operator resumes it), an operator pause stays (the
        ``pause`` command cleared the ownership). ``resume`` refuses without a fresh
        kill-switch read; the pause then stays owned and the next good claim retries.
        """
        channel = self._cmd_channel
        guard = self._guard
        self._lift_channel_pause()
        failures = channel.unreported_recovery
        if failures is None:
            return
        # Cleared before the write: a raising write must not repeat the alert every tick.
        channel.unreported_recovery = None
        blocked = guard is not None and guard.guard_state != GuardState.RUNNING
        state_text = (
            guard_state_text(guard.guard_state, guard.halt_reason) if guard is not None else "none"
        )
        entries = f"entries stay blocked ({state_text})" if blocked else "entries resumed"
        try:
            self._journal.record(
                EventType.RISK_CHECK,
                {
                    "action": "command_channel_restored",
                    "failures": failures,
                    "entries": "blocked" if blocked else "resumed",
                    "state": state_text,
                    "passed": True,
                },
                # Critical: the 60 s per-type limit would drop it right after the streak's own
                # critical failure alert, leaving the operator on "entries blocked".
                alert=Alert(
                    "error",
                    restored_alert_text(self._config.trader_id, failures, entries),
                    critical=True,
                ),
            )
        except Exception as exc:
            logger.error("command channel alert failed (%s)", type(exc).__name__)

    def _block_entries_for_channel(self, message: str | None = None) -> bool:
        """Pause the guard for a failing channel; True when this call paused it.

        Only a RUNNING guard is paused, so a halt, an operator pause or a restored halt is
        never replaced. The channel owns the pause it set (``owns_pause``) and only that
        one may be lifted on recovery. ``message``: the pause text (default: the failing
        channel's). Idempotent; never raises.
        """
        guard = self._guard
        channel = self._cmd_channel
        if guard is None:
            return False
        state: GuardState = guard.guard_state
        if state != GuardState.RUNNING:
            return False
        if message is None:
            # A RUNNING guard: whoever lifted the unsaved-checkpoint pause, it is gone.
            channel.checkpoint_blocking = False
            message = block_message(channel.failures, channel.last_error)
        try:
            paused = guard.pause(message)
        except Exception as exc:
            # The guard is PAUSED before its listener runs, so a raising state-change
            # write must not leave the pause unowned (never lifted).
            logger.error("command channel pause failed (%s)", type(exc).__name__)
            paused = guard.guard_state == GuardState.PAUSED
        if paused:
            channel.owns_pause = True
        return paused

    _NOT_EXECUTED = "node stopping; command not executed"
    #: J2: what a row left ``running`` by a previous process gets - its outcome is unknown.
    _STALE_RUNNING = "node restarted; command outcome unknown (was running when the node stopped)"
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
            # An operator pause is never the channel's to lift on recovery.
            self._cmd_channel.owns_pause = False
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

    def _entry_block_message(self) -> str | None:
        """The pause text a RUNNING transition is re-paused with, or None when none applies."""
        channel = self._cmd_channel
        if channel.checkpoint_blocking:
            return CHECKPOINT_BLOCK_MESSAGE
        if self._startup_resume_owed:
            return RESUME_OWED_MESSAGE
        if channel.blocking:
            return block_message(channel.failures, channel.last_error)
        return None

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
                self._journal.record(
                    EventType.RISK_CHECK,
                    {
                        "action": "halt",
                        "check_type": reason.value,
                        "reason": reason.value,
                        "message": message,
                        "passed": False,
                        **metrics,
                    },
                    alert=Alert("circuit_breaker", f"HALT ({reason.value}): {message}", True),
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
                held = self._entry_block_message()
                if state == GuardState.RUNNING and (prev == NodeState.HALTED or held is not None):
                    # The checkpoint/channel block re-pauses this transition below: the
                    # operator is told it is still paused, never that it resumed.
                    text = f"RESUMED: {message}"
                    if held is not None:
                        lead = (
                            "day rolled over"
                            if message.startswith("UTC day rollover")
                            else "resume re-paused"
                        )
                        text = f"{lead}, still paused: {held}"
                    self._journal.record(
                        EventType.SIGNAL,
                        payload,
                        alert=Alert("circuit_breaker", text, True),
                    )
                else:
                    self._journal.record(EventType.SIGNAL, payload)
        finally:
            # bykze: the transition is checkpointed even when the event write raises. The
            # status was updated above, so the checkpoint carries the new state/the halt
            # reason and a disk-full writer cannot cost the restart its halt. The helper
            # swallows its own failure, so it never masks the write's error.
            self._save_checkpoint_now()
            # 3fg9d: still no readable command channel -> the guard never stays RUNNING
            # (a daily-loss lift or SIGUSR2 resume is re-paused inside the same transition,
            # so no NT callback can submit an entry in between).
            # xfjji: the same for the unsaved start-up checkpoint - a daily-loss lift at UTC
            # midnight must not reopen entries before the checkpoint is saved.
            # 1A: and for the operator resume a late start-up checkpoint failure still owes.
            held = self._entry_block_message()
            if state == GuardState.RUNNING and held is not None:
                try:
                    self._block_entries_for_channel(held)
                except Exception as exc:
                    logger.error("command channel re-block failed (%s)", type(exc).__name__)

    def _on_order_rejected(self, event: Any) -> None:
        self._journal.record(
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
            self._journal.record(
                EventType.ORDER,
                payload,
                alert=Alert("warning", f"Entry refused (margin limit): {reason}\n{symbol}"),
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
            self._journal.record(
                EventType.ORDER,
                payload,
                alert=Alert("error", f"Entry check failed (order denied): {reason}\n{symbol}"),
                timestamp=timestamp,
            )
        elif not by_gate:
            self._journal.record(
                EventType.ORDER,
                payload,
                alert=Alert(
                    "error", f"Order denied by risk engine: {event.reason}\n{event.instrument_id}"
                ),
                timestamp=timestamp,
            )
        else:
            self._journal.record(EventType.ORDER, payload, timestamp=timestamp)

    def _on_position_opened(self, event: Any) -> None:
        self._journal.record(
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
        self._journal.trade_alert(
            lambda: self._trade_open_message(event),
            delay=TRADE_ALERT_DELAY_S,
            enabled=self._config.trade_alerts,
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
            self._journal.record(EventType.RISK_CHECK, payload, timestamp=timestamp)
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
        self._journal.record(
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
        self._journal.trade_alert(
            lambda: self._trade_close_message(event, net, commissions, exit_reason),
            delay=0.0,
            enabled=self._config.trade_alerts,
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
        self._journal.record(
            EventType.RISK_CHECK,
            {"action": "warning", "message": message},
            # 7fylz: the unowned-position text is the operator's only report of a
            # position the close-all left open, so it bypasses the 60 s ERROR limit
            # (the old composed close-all alert was critical too).
            alert=Alert(
                "error", f"WARNING: {message}", critical=message.startswith(UNOWNED_PREFIX)
            ),
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

    def _log_snapshot_failure(self, item: str, exc: BaseException, errors: list[str]) -> None:
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
            after_failure = self._startup_checkpoint_failed
            future = self._db_worker.submit(
                self._persistence.save_checkpoint, self._capture_checkpoint()
            )
            future.add_done_callback(self._log_checkpoint_failure)
            if after_failure:  # only a save submitted after the failure counts
                future.add_done_callback(self._failed_startup_checkpoint_saved)
        except Exception as exc:
            # s3pqx: class only - this now always runs in a finally after a failed
            # guard-transition write (bykze), so an active exception is present and
            # ``logger.exception`` would print that error and its ``__context__`` chain
            # next to this one. Same rule as _check_connectivity / journal.record(alert=...).
            logger.error("immediate checkpoint failed (%s)", type(exc).__name__)

    @staticmethod
    def _log_checkpoint_failure(future: Future[Any]) -> None:
        """Done-callback of a worker save: class only (see ``_save_checkpoint_now``)."""
        if not future.cancelled() and (exc := future.exception()) is not None:
            logger.error("immediate checkpoint failed (%s)", type(exc).__name__)

    def _failed_startup_checkpoint_saved(self, future: Future[Any]) -> None:
        """Done-callback (worker thread): a save landed after the start-up one failed."""
        loop = self._checkpoint_loop
        if future.cancelled() or future.exception() is not None or loop is None:
            return
        try:
            loop.call_soon_threadsafe(self._clear_failed_startup_checkpoint)
        except RuntimeError:  # loop closed: shutting down
            logger.warning("failed start-up checkpoint block not cleared: loop closed")

    def _clear_failed_startup_checkpoint(self) -> None:
        """Lift the block of the failed start-up checkpoint; the guard stays PAUSED."""
        if not self._startup_checkpoint_failed:
            return
        self._startup_checkpoint_failed = False
        self._cmd_channel.checkpoint_blocking = False
        self._cmd_channel.owns_pause = False  # only an operator resume lifts it now
        guard = self._guard
        # The pause text must stay true: the checkpoint is saved, the resume is owed.
        owed = self._startup_resume_owed
        if owed and guard is not None and guard.set_pause_message(RESUME_OWED_MESSAGE):
            self._status.error_message = RESUME_OWED_MESSAGE
            self._status.updated_at = datetime.now(UTC)

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

        The decision is :func:`vibe_quant.paper.lineage.resolve`: this trader's own
        latest checkpoint, the newest row of another trader on the same account
        (the risk source unless ``start_fresh``), the key rotation of the same venue
        + env, the liquidation watch merge and the halt precedence, all from the
        rows alone. Here the result is only applied: it fills the ``_restored_*``
        wiring, mirrors a restored halt into the node status (so a checkpoint taken
        before ``_run_loop`` keeps it, H19) and writes the one restore event. An
        unreadable row fails closed (blank risk, ERROR halt, status HALTED) and
        never raises.

        Positions and orders come from NautilusTrader's venue reconciliation
        (claimed via ``external_order_claims``); the checkpoint copy is only logged.
        """
        if self._persistence is None:
            return
        account = binance_account_id(self._config.binance)
        inheritance = lineage.resolve(
            self._persistence,
            self._config.trader_id,
            account,
            start_fresh=self._config.start_fresh,
            now=lambda: datetime.now(UTC),
        )
        self._restored_risk = inheritance.risk
        self._restored_halt = inheritance.halt
        self._risk_inherited_from = inheritance.risk_source
        self._restored_watch = inheritance.watch
        self._restored_watch_ms = inheritance.watch_ms
        if inheritance.unreadable is not None:
            # Resolve failed closed (R18/R19): the same status the except branch of
            # the old inline restore set, and the failure event it returned below.
            assert inheritance.halt is not None
            self._status.state = NodeState.HALTED
            self._status.halt_reason = HaltReason.ERROR
            self._status.error_message = inheritance.halt.message
            self._status.updated_at = datetime.now(UTC)
        elif inheritance.halt is not None:
            self._mirror_restored_halt()
        # Base order: the event (its sorted(positions/orders) can raise on a corrupt
        # column) is built only AFTER risk, halt, watch and status are installed.
        if inheritance.restore_event is not None:
            self._journal.record(EventType.SIGNAL, inheritance.restore_event())
        if inheritance.log_restored is not None:
            inheritance.log_restored()

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
        # 7fylz: the guard's listener already warned the unowned-position text, so
        # the composed alert carries only the remaining problems - one event, one alert.
        problems = [e for e in result.errors if not e.startswith(UNOWNED_PREFIX)]
        if problems:
            self._journal.record(
                EventType.SIGNAL,
                payload,
                alert=Alert("error", "Close-all problems: " + "; ".join(problems), True),
            )
        else:
            self._journal.record(EventType.SIGNAL, payload)
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
        # r3 F10: a failed LATEST read keeps the earlier clean timestamp, so the
        # predicate is "no clean read yet", not "the latest read errored"; the
        # engaged text is read only once a clean read exists, as today.
        readable = snap.last_ok_mono is not None
        facts = StartFacts(
            kill_unreadable=(None if readable else (snap.error_class or "no read within 10 s")),
            kill_engaged=(snap.reason or "") if readable and snap.engaged else None,
        )
        decision = check(_NO_CONSENT_START, facts)
        if isinstance(decision, Refusal):
            _raise_start_refusal(decision)

        # Create event writer
        self._config.logs_path.mkdir(parents=True, exist_ok=True)
        self._journal.writer = EventWriter(
            run_id=self._config.trader_id,
            base_path=self._config.logs_path,
        )

        # Load, compile, resolve params, instantiate strategy (one per symbol)
        self._strategy = self._load_strategy()
        self._journal.strategy_name = self._strategy.name
        self._compiled_module = self._compile_strategy(self._strategy)
        self._resolved = self._resolve_run_config(self._compiled_module, self._strategy)
        self._compiled_strategies = self._instantiate_strategies(
            self._compiled_module, self._strategy
        )

        resolved = self._resolved
        self._journal.record(
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
        if not await wait_future(self._journal.event_log.barrier(), EVENT_STARTUP_WAIT_S):
            raise ConfigurationError(
                f"paper event log not writable within {EVENT_STARTUP_WAIT_S:g} s; refusing to start"
            )
        log_health = self._journal.event_log.health()
        if log_health.failed > 0:
            raise ConfigurationError(
                f"paper event log not writable ({log_health.last_error_class}); refusing to start"
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
            # str(exc) is token-free by construction; the mapped refusal keeps the
            # probe out of the chain (``from None`` in _raise_start_refusal), so the
            # httpx error (whose str() holds the token URL) never surfaces.
            decision = check(_NO_CONSENT_START, StartFacts(telegram_error=str(exc)))
            if isinstance(decision, Admitted):
                raise AssertionError(f"a telegram error must refuse, got {decision}") from None
            if decision.code is not RefusalCode.TELEGRAM_UNAVAILABLE:
                raise AssertionError(f"node never produces {decision.code}") from None
            _raise_start_refusal(decision)
        if telegram_config is None:
            logger.info("Telegram alerts disabled (TELEGRAM_BOT_TOKEN/CHAT_ID not set)")
        else:
            self._journal.telegram = TelegramBot(telegram_config)

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
            # wi1m0(b): bounded like the drain below - a hung sweep alerts, the start goes on.
            swept = await self._startup_step(
                self._cmd_worker.run(
                    self._commands.fail_stale_running,
                    self._config.trader_id,
                    self._STALE_RUNNING,
                ),
                STARTUP_SWEEP_STEP,
                0,
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
        # wi1m0(b) r3: a hung write keeps entries paused until it is saved (no trading
        # without crash persistence); its late save lifts only that block.
        await self._startup_step(
            self._db_worker.run(self._persistence.save_checkpoint, self._capture_checkpoint()),
            STARTUP_CHECKPOINT_STEP,
            None,
            blocks_entries=True,
        )
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

    async def _startup_step(
        self, job: Awaitable[Any], step: str, unanswered: Any, *, blocks_entries: bool = False
    ) -> Any:
        """Await a start-up worker job at most ``COMMAND_CLAIM_STALL_S``; its result.

        wi1m0(b): a job that gives no answer (a busy DB lock, NFS) raises ONE critical
        alert and returns ``unanswered``: the start continues instead of waiting on it for
        good with no word to the operator. The job stays queued on its worker; a late
        failure of it is dropped, unless ``blocks_entries``: then entries are paused until
        the job answers (``_startup_job_late``). The job's own exception propagates.
        """
        task = asyncio.ensure_future(job)
        try:
            done, _ = await asyncio.wait({task}, timeout=COMMAND_CLAIM_STALL_S)
        except asyncio.CancelledError:
            task.cancel()
            raise
        if done:
            return task.result()
        entries = ""
        if blocks_entries:
            self._cmd_channel.checkpoint_blocking = True
            blocked_now = self._block_entries_for_channel(CHECKPOINT_BLOCK_MESSAGE)
            kind, entries = self._entries_after_block(blocked_now)
            if kind != "not_blocked":
                entries += " until it is saved"
            task.add_done_callback(lambda t: self._startup_job_late(t, step))
        else:
            task.add_done_callback(lambda t: t.cancelled() or t.exception())
        logger.error("start-up %s gave no answer within %g s", step, COMMAND_CLAIM_STALL_S)
        self._journal.record(
            EventType.RISK_CHECK,
            {"action": "startup_stalled", "step": step, "passed": False},
            alert=Alert(
                "error", startup_stall_alert_text(self._config.trader_id, step, entries), True
            ),
        )
        return unanswered

    def _startup_job_late(self, task: asyncio.Future[Any], step: str) -> None:
        """A stalled entry-blocking start-up job answered: a save lifts only its block.

        A late failure keeps entries blocked (nothing saved; an operator resume lifts it)
        and raises one critical alert, class only. Never raises.
        """
        if task.cancelled() or self._shutdown_in_progress():
            if not task.cancelled():
                task.exception()  # retrieved: a failure during shutdown is not news
            return
        exc = task.exception()
        if exc is None:
            self._cmd_channel.checkpoint_blocking = False
            self._lift_channel_pause()
            return
        logger.error("start-up %s failed late (%s)", step, type(exc).__name__)
        # A later successful save clears the block (not the pause): the operator resumes.
        self._startup_checkpoint_failed = True
        self._checkpoint_loop = asyncio.get_running_loop()
        self._cmd_channel.owns_pause = False
        guard = self._guard
        blocked = guard is not None and guard.guard_state != GuardState.RUNNING
        self._startup_resume_owed = blocked
        entries = (
            "new entries stay blocked until a checkpoint saves and an operator resumes"
            if blocked
            else "new entries NOT blocked"
        )
        try:
            self._journal.record(
                EventType.RISK_CHECK,
                {
                    "action": "startup_failed_late",
                    "step": step,
                    "error_type": type(exc).__name__,
                    "passed": False,
                },
                alert=Alert(
                    "error",
                    startup_checkpoint_failed_alert_text(
                        self._config.trader_id, type(exc).__name__, entries
                    ),
                    True,
                ),
            )
        except Exception as record_exc:
            logger.error("start-up failure alert failed (%s)", type(record_exc).__name__)

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

        self._journal.record(
            EventType.SIGNAL,
            {
                "action": "state_change",
                "from_state": NodeState.INITIALIZING.value,
                "to_state": self._status.state.value,
            },
        )

        if self._persistence is not None:
            # 1A: a routine save that succeeds also lifts a failed start-up checkpoint block.
            await self._persistence.start_periodic_checkpointing(
                self._capture_checkpoint,
                self._db_worker.executor,
                on_saved=self._clear_failed_startup_checkpoint,
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
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        # 3fg9d: a stalled claim kept between ticks has no poller left; whatever it claims
        # ends failed (never left running) once the worker returns from it.
        inflight = self._claim_inflight
        self._claim_inflight = None
        if inflight is not None and self._commands is not None:
            self._cmd_worker.submit(self._fail_claimed, self._commands, inflight, [])
        unrun, self._unrun_claimed = self._unrun_claimed, []
        if unrun and self._commands is not None:
            self._cmd_worker.submit(self._fail_claimed, self._commands, None, unrun)

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
            self._journal.record(
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
        # ``journal.record`` calls are no-ops; the event worker flushes and closes the
        # file after its last queued line. R6 (k8oei): a close() failure is logged by
        # class and the remaining teardown (telegram, state manager) still runs.
        await self._journal.close(self._close_store)

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
                self._journal.record(
                    EventType.RISK_CHECK,
                    {"action": "error", "error": str(e), "passed": False},
                    alert=Alert("error", f"Paper node failed: {e}", True),
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
            self._journal.record(
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
            return self._resume_guard(self._guard, engaged)
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
        self._journal.record(EventType.SIGNAL, payload)
        return True, text

    def _resume_guard(self, guard: TradingGuard, engaged: bool) -> tuple[bool, str]:
        """An operator resume of the guard; a resume a block re-pauses is reported as such.

        It pays the resume a late start-up checkpoint failure owes (1A) once that block
        has cleared; a refused resume leaves it owed.
        """
        owed = self._startup_resume_owed and not self._cmd_channel.checkpoint_blocking
        if owed:
            self._startup_resume_owed = False
        held = self._entry_block_message()
        ok, text = guard.resume(kill_switch_engaged=engaged)
        if not ok:
            self._startup_resume_owed = self._startup_resume_owed or owed
            return ok, text
        if held is not None and guard.guard_state == GuardState.PAUSED:
            return False, f"resume re-paused, still paused: {held}"
        return ok, text

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
