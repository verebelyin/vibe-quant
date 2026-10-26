"""Trading guard for paper/live nodes: order gate, risk limits, halt/pause/close-all.

One NautilusTrader ``Actor`` owns the trading state of a paper/live node:

* **Order gate.** The guard re-registers the message-bus ``RiskEngine.execute``
  endpoint so every ``SubmitOrder``/``SubmitOrderList`` from every strategy passes
  through :meth:`TradingGuard._gate_execute` before NT's own RiskEngine. In
  ``RUNNING`` state the gate enforces a single-position invariant (no order may
  open a second position, add to or flip an existing one, or open while another
  entry order is still pending) and ``max_position_count``. In ``PAUSED`` /
  ``HALTED`` state only position-reducing orders pass. Denials are real NT
  ``OrderDenied`` events, so strategies see them through ``on_order_denied``.
  NT's own ``TradingState.REDUCING`` is not used because it lets a BUY through
  when the portfolio is flat (it only checks ``is_net_long``).
* **Risk limits.** Equity = account wallet balance + unrealized PnL
  (``Portfolio.equity``) in the settlement currency. Max drawdown from the high
  water mark, max daily loss vs. the UTC-day starting equity, and max consecutive
  losing positions are evaluated on a timer and on every position/account event.
* **Halt.** Gate → reduce-only, then every strategy runs NT's ``market_exit()``
  (cancel open orders, close positions with reduce-only market orders, retry until
  flat), then the strategy is stopped. A daily-loss halt flattens and blocks
  entries but keeps strategies running (indicator buffers stay warm) and lifts
  automatically at the next UTC midnight.
* **Pause.** Gate → reduce-only only. Protective SL/TP orders and strategy exit
  logic keep working.
* **Close-all.** ``market_exit()`` on every strategy; result lists the targeted
  positions, positions owned by no guarded strategy, and per-strategy errors.

The guard never talks to the database or Telegram itself; it reports through a
:class:`GuardListener` so the node can persist, log and alert.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.core.rust.common import ComponentState
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.execution.messages import SubmitOrder, SubmitOrderList
from nautilus_trader.model.enums import OrderSide, PriceType
from nautilus_trader.model.events import (
    OrderDenied,
    OrderRejected,
    PositionClosed,
    PositionOpened,
)
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Currency

from vibe_quant.margin_limit import (
    DEFAULT_PAPER_MAX_MARGIN_PCT,
    MarginItem,
    check_entry_margin,
    unique_pending_entries,
)
from vibe_quant.paper.halt_state import (
    _KEEP_STRATEGIES_RUNNING,
    OPERATOR_RESUMABLE_HALTS,
    RESTORED_HALT_PREFIX,
    RESTORED_HALTS,
    GuardState,
    HaltReason,
    HaltState,
    RestoredHalt,
    halt_supersedes,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

#: Prefix on every OrderDenied reason produced by the guard's gate.
GATE_DENIAL_PREFIX = "VQ_GUARD"

#: Prefix on a gate denial produced by the margin limit (node alerts once per symbol).
MARGIN_DENIAL_PREFIX = f"{GATE_DENIAL_PREFIX}: margin limit: "

#: Prefix on a gate denial produced by a raising entry check (node alerts once per symbol).
ENTRY_CHECK_FAILED_PREFIX = f"{GATE_DENIAL_PREFIX}: entry check failed ("

_RISK_ENDPOINT = "RiskEngine.execute"
_TICK_TIMER = "TradingGuard.tick"

#: Prefix on the warning listing open positions no guarded strategy owns.
_UNOWNED_PREFIX = "positions not owned by any guarded strategy were NOT closed: "


# NT component states in which a guarded strategy has not started yet. A stopping
# halt applied now (e.g. in on_start, before NT starts the strategies) cannot
# flatten them, but must record them so they are stopped once they run.
_STRATEGY_NOT_STARTED: frozenset[ComponentState] = frozenset(
    {ComponentState.PRE_INITIALIZED, ComponentState.READY, ComponentState.STARTING}
)


@dataclass(frozen=True)
class RiskLimits:
    """Risk limits enforced by the guard (fractions, e.g. 0.05 = 5%)."""

    max_drawdown_pct: Decimal
    max_daily_loss_pct: Decimal
    max_consecutive_losses: int
    max_position_count: int

    def __post_init__(self) -> None:
        if not Decimal("0") < self.max_drawdown_pct <= Decimal("1"):
            raise ValueError(f"max_drawdown_pct must be in (0, 1], got {self.max_drawdown_pct}")
        if not Decimal("0") < self.max_daily_loss_pct <= Decimal("1"):
            raise ValueError(f"max_daily_loss_pct must be in (0, 1], got {self.max_daily_loss_pct}")
        if self.max_consecutive_losses < 1:
            raise ValueError("max_consecutive_losses must be >= 1")
        if self.max_position_count < 1:
            raise ValueError("max_position_count must be >= 1")


#: RiskState keys a checkpoint round-trips (see :meth:`RiskState.to_dict`); the
#: derived ``drawdown_pct`` / ``daily_loss_pct`` are ignored on restore (R10).
_RISK_STATE_KEYS: tuple[str, ...] = (
    "high_water_mark",
    "day",
    "day_start_equity",
    "consecutive_losses",
    "last_equity",
)

#: Decimal fields whose restored value must be strictly positive to be a usable
#: baseline: a zero/negative high water mark or UTC-day start would re-baseline
#: the drawdown / daily-loss limits (R20).
_RISK_POSITIVE_DECIMAL_KEYS: frozenset[str] = frozenset({"high_water_mark", "day_start_equity"})


@dataclass
class RiskState:
    """Mutable risk bookkeeping, persisted in checkpoints so restarts keep limits."""

    high_water_mark: Decimal | None = None
    day: date | None = None
    day_start_equity: Decimal | None = None
    consecutive_losses: int = 0
    last_equity: Decimal | None = None

    @property
    def drawdown_pct(self) -> Decimal | None:
        """Current drawdown from the high water mark (fraction)."""
        if self.high_water_mark is None or self.last_equity is None:
            return None
        if self.high_water_mark <= 0:
            return None
        return (self.high_water_mark - self.last_equity) / self.high_water_mark

    @property
    def daily_loss_pct(self) -> Decimal | None:
        """Loss since the UTC-day start (fraction, positive = loss)."""
        if self.day_start_equity is None or self.last_equity is None:
            return None
        if self.day_start_equity <= 0:
            return None
        return (self.day_start_equity - self.last_equity) / self.day_start_equity

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly representation (Decimals as strings)."""
        return {
            "high_water_mark": _dec_str(self.high_water_mark),
            "day": self.day.isoformat() if self.day else None,
            "day_start_equity": _dec_str(self.day_start_equity),
            "consecutive_losses": self.consecutive_losses,
            "last_equity": _dec_str(self.last_equity),
            "drawdown_pct": _dec_str(self.drawdown_pct),
            "daily_loss_pct": _dec_str(self.daily_loss_pct),
        }

    @classmethod
    def from_checkpoint(cls, data: object) -> tuple[RiskState, tuple[str, ...]]:
        """Restore from :meth:`to_dict` output and never raise.

        Returns ``(state, problems)``. ``None``, ``{}`` and an all-null dict are a
        clean fresh state; any other non-dict value is a ``risk=<type>`` problem.
        A non-empty dict with at least one field set and at least one null/absent
        field is partial (R21): each null/absent field becomes a ``"<key> missing"``
        problem and each unreadable value a ``"<key>=<repr>"`` one. Only an all-null
        or all-set dict restores cleanly.
        """
        if data is None:
            return cls(), ()
        if not isinstance(data, dict):
            return cls(), (f"risk={type(data).__name__}",)
        if not data:
            return cls(), ()

        def _unset(key: str) -> bool:
            if key not in data or data[key] is None:
                return True
            # consecutive_losses=0 is the field's own default: not "set".
            return key == "consecutive_losses" and data[key] == 0

        if all(_unset(key) for key in _RISK_STATE_KEYS):
            return cls(), ()

        problems: list[str] = []
        high_water_mark: Decimal | None = None
        day: date | None = None
        day_start_equity: Decimal | None = None
        consecutive_losses = 0
        last_equity: Decimal | None = None
        for key in _RISK_STATE_KEYS:
            if key not in data or data[key] is None:
                problems.append(f"{key} missing")
                continue
            raw = data[key]
            problem: str | None
            if key == "day":
                day, problem = _restore_day(raw)
            elif key == "consecutive_losses":
                consecutive_losses, problem = _restore_losses(raw)
            elif key == "high_water_mark":
                high_water_mark, problem = _restore_decimal(key, raw)
            elif key == "day_start_equity":
                day_start_equity, problem = _restore_decimal(key, raw)
            else:
                last_equity, problem = _restore_decimal(key, raw)
            if problem is not None:
                problems.append(problem)
        return (
            cls(
                high_water_mark=high_water_mark,
                day=day,
                day_start_equity=day_start_equity,
                consecutive_losses=consecutive_losses,
                last_equity=last_equity,
            ),
            tuple(problems),
        )


@dataclass
class CloseAllResult:
    """Outcome of a close-all request (fills complete asynchronously)."""

    targeted_positions: list[str] = field(default_factory=list)
    unowned_positions: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "targeted_positions": list(self.targeted_positions),
            "unowned_positions": list(self.unowned_positions),
            "errors": list(self.errors),
        }


class GuardListener:
    """Callbacks from the guard to its owner (node). Default: no-ops."""

    def on_state_change(
        self,
        state: GuardState,
        reason: HaltReason | None,
        message: str,
        metrics: dict[str, object],
    ) -> None:
        """Guard state changed (halt, pause, resume, auto-resume)."""

    def on_order_rejected(self, event: Any) -> None:
        """The venue rejected an order."""

    def on_order_denied(self, event: Any, *, by_gate: bool) -> None:
        """An order was denied (by the guard's gate or by NT's RiskEngine)."""

    def on_position_opened(self, event: Any) -> None:
        """A guarded strategy opened a position."""

    def on_position_closed(self, event: Any) -> None:
        """A guarded strategy closed a position."""

    def on_position_close_settled(self, event: Any) -> None:
        """A guarded strategy's close is fully booked (losing streak and risk evaluated)."""

    def on_warning(self, message: str) -> None:
        """Non-fatal condition the operator should know about."""

    def on_entry_allowed(self, symbol: str) -> None:
        """An entry for ``symbol`` passed the margin limit (re-arms its refusal alert)."""


class TradingGuardConfig(ActorConfig, frozen=True):  # type: ignore[misc]
    """Configuration for :class:`TradingGuard`."""

    venue: str = "BINANCE"
    settlement_currency: str = "USDT"
    max_drawdown_pct: Decimal = Decimal("0.15")
    max_daily_loss_pct: Decimal = Decimal("0.02")
    max_consecutive_losses: int = 10
    max_position_count: int = 5
    check_interval_secs: int = 5
    #: Margin cushion (fraction of equity) for new entries; paper default 50%.
    max_margin_pct: float = float(DEFAULT_PAPER_MAX_MARGIN_PCT)


def _dec_str(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _restore_decimal(key: str, raw: object) -> tuple[Decimal | None, str | None]:
    """Parse one persisted Decimal field; returns ``(value, problem)``, never raises.

    A bool, a non-scalar, a non-finite ``Decimal`` ("NaN"/"Infinity"/"sNaN") or (for
    the baseline fields) a non-positive value is a problem (R4-R6, R20); the value
    is not restored.
    """
    problem = f"{key}={repr(raw)[:40]}"
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str, Decimal)):
        return None, problem
    try:
        value = Decimal(str(raw))
    except (ArithmeticError, ValueError):
        return None, problem
    if not value.is_finite():
        return None, problem
    if key in _RISK_POSITIVE_DECIMAL_KEYS and value <= 0:
        return None, problem
    return value, None


def _restore_day(raw: object) -> tuple[date | None, str | None]:
    """Parse the persisted UTC ``day``; returns ``(value, problem)``, never raises."""
    problem = f"day={repr(raw)[:40]}"
    if isinstance(raw, str):
        try:
            return date.fromisoformat(raw), None
        except ValueError:
            return None, problem
    if isinstance(raw, date):
        return raw, None
    return None, problem


def _restore_losses(raw: object) -> tuple[int, str | None]:
    """Parse ``consecutive_losses``; returns ``(value, problem)``, never raises.

    Only a non-negative ``int`` or a bare (no whitespace, no sign, no dot) string
    of Unicode decimal digits is accepted; a float, bool, negative or otherwise
    unreadable value is a problem and restores the default 0 (R9/R9b). ``isdecimal``
    (not ``isdigit``) is the guard: ``'³'.isdigit()`` is True but ``int('³')``
    raises, and a decimal string past ``int()``'s digit limit raises too - so the
    conversion stays inside a ``try``.
    """
    problem = f"consecutive_losses={repr(raw)[:40]}"
    if isinstance(raw, bool):
        return 0, problem
    if isinstance(raw, int):
        return (raw, None) if raw >= 0 else (0, problem)
    if isinstance(raw, str) and raw == raw.strip() and raw.isdecimal():
        try:
            return int(raw), None
        except ValueError:
            return 0, problem
    return 0, problem


def _order_reduces(side: OrderSide, quantity: Decimal, net_position: Decimal) -> bool:
    """True when an order only reduces (never opens/flips) the net position."""
    if side == OrderSide.BUY:
        return net_position < 0 and quantity <= -net_position
    if side == OrderSide.SELL:
        return net_position > 0 and quantity <= net_position
    return False


class TradingGuard(Actor):  # type: ignore[misc]
    """Owns trading state (running/paused/halted) and enforces it on every order."""

    def __init__(
        self,
        config: TradingGuardConfig,
        strategies: Sequence[Any],
        risk_engine: Any,
        listener: GuardListener | None = None,
        risk_state: RiskState | None = None,
        restored_halt: RestoredHalt | None = None,
        risk_source: str | None = None,
    ) -> None:
        super().__init__(config)
        self._cfg = config
        self._limits = RiskLimits(
            max_drawdown_pct=Decimal(str(config.max_drawdown_pct)),
            max_daily_loss_pct=Decimal(str(config.max_daily_loss_pct)),
            max_consecutive_losses=config.max_consecutive_losses,
            max_position_count=config.max_position_count,
        )
        self._strategies = list(strategies)
        self._risk_engine = risk_engine
        self._listener = listener or GuardListener()
        self._risk = risk_state or RiskState()
        # Trader whose checkpoint supplied the risk state (None: this trader's own).
        self._risk_hint = (
            f" (limits inherited from {risk_source}; start fresh skips that trader id's "
            "risk state; this trader id's own saved limits and halt still carry)"
            if risk_source is not None
            else ""
        )
        self._venue = Venue(config.venue)
        self._currency = Currency.from_str(config.settlement_currency)
        self._halt = HaltState.from_restored(restored_halt)
        self._pending_stop: set[str] = set()
        # Strategies recorded for stopping while they were not running yet (a
        # halt in on_start, before NT starts them): they still owe a market exit.
        self._pending_exit: set[str] = set()
        self._gate_installed = False
        self._warned: set[str] = set()
        self._gate_denials = 0
        self._restored_halt = restored_halt
        self._restored_flatten_pending = False
        # True once on_start ran: after that a halt's flatten is the only report.
        self._started = False
        # Unowned positions a pre-start halt's flatten saw (NT reconciles the
        # account's open positions after that halt, before on_start).
        self._prestart_unowned: list[str] | None = None

    # ------------------------------------------------------------------ props

    @property
    def guard_state(self) -> GuardState:
        return self._halt.state

    @property
    def halt_reason(self) -> HaltReason | None:
        return self._halt.reason

    @property
    def message(self) -> str:
        return self._halt.message

    @property
    def risk_state(self) -> RiskState:
        return self._risk

    @property
    def limits(self) -> RiskLimits:
        return self._limits

    @property
    def gate_denials(self) -> int:
        return self._gate_denials

    def status(self) -> dict[str, object]:
        """Snapshot for checkpoints / API."""
        return {
            "state": self._halt.state.value,
            "halt_reason": self._halt.reason.value if self._halt.reason else None,
            "message": self._halt.message,
            "gate": "active" if self._halt.state == GuardState.RUNNING else "reduce_only",
            "pending_stop": sorted(self._pending_stop),
            "gate_denials": self._gate_denials,
            "risk": self._risk.to_dict(),
            "limits": {
                "max_drawdown_pct": str(self._limits.max_drawdown_pct),
                "max_daily_loss_pct": str(self._limits.max_daily_loss_pct),
                "max_consecutive_losses": self._limits.max_consecutive_losses,
                "max_position_count": self._limits.max_position_count,
            },
        }

    # -------------------------------------------------------------- lifecycle

    def on_start(self) -> None:
        self._started = True
        self.install_gate()
        self.msgbus.subscribe(topic="events.position.*", handler=self._on_position_event)
        self.msgbus.subscribe(topic="events.order.*", handler=self._on_order_event)
        self.msgbus.subscribe(topic="events.account.*", handler=self._on_account_event)
        self.clock.set_timer(
            name=_TICK_TIMER,
            interval=timedelta(seconds=self._cfg.check_interval_secs),
            callback=self._on_tick,
        )
        if (
            self._restored_halt is not None
            and self._halt.state == GuardState.HALTED
            and self._halt.reason == self._restored_halt.reason
        ):
            # The state check is defensive (kept, no behaviour change): a resume
            # before on_start (H20) already cleared the flag in _lift_halt, and
            # _on_tick re-checks HALTED before using it. Armed only while the restored
            # halt is still the active one: a halt applied before on_start (the node's
            # replayed start-up SIGUSR1) that superseded it already flattened or queued
            # the exit, and a second flatten on the first tick would repeat it.
            self._restored_flatten_pending = True
        self.evaluate_risk()
        if (
            self._restored_halt is not None
            and self._halt.state == GuardState.HALTED
            and self._halt.reason == self._restored_halt.reason
        ):
            # msocy: an escalation above (evaluate_risk re-halted with a more
            # severe reason) already notified through halt(); only an un-escalated
            # restored halt notifies here, so the operator sees one event/alert.
            try:
                self._listener.on_state_change(
                    self._halt.state,
                    self._halt.reason,
                    self._halt.message,
                    {
                        "restored": True,
                        "checkpoint_time": self._restored_halt.checkpoint_time,
                    },
                )
            except Exception as exc:
                logger.error("restored halt notify failed (%s)", type(exc).__name__)

        # A halt applied before this actor started flattened the book as it was
        # then; NT reconciles the account's open positions after that halt and
        # before this point, so report what it found once. An escalation above
        # (evaluate_risk -> halt, _started already True) cleared the list: its
        # own flatten is the only warning.
        reported, self._prestart_unowned = self._prestart_unowned, None
        if reported is not None and self._halt.state == GuardState.HALTED:
            unowned = self._split_positions()[1]
            if unowned and unowned != reported:
                try:
                    self._listener.on_warning(_UNOWNED_PREFIX + ", ".join(unowned))
                except Exception as exc:
                    logger.error("unowned-position report failed (%s)", type(exc).__name__)

    def on_stop(self) -> None:
        # The gate stays installed: strategies may still be running and must
        # keep obeying the trading state until the node is disposed.
        for topic, handler in (
            ("events.position.*", self._on_position_event),
            ("events.order.*", self._on_order_event),
            ("events.account.*", self._on_account_event),
        ):
            if self.msgbus.is_subscribed(topic, handler):
                self.msgbus.unsubscribe(topic=topic, handler=handler)

    def on_dispose(self) -> None:
        self.uninstall_gate()

    def install_gate(self) -> None:
        """Route ``RiskEngine.execute`` through the gate (idempotent)."""
        if self._gate_installed:
            return
        self.msgbus.deregister(_RISK_ENDPOINT, self._risk_engine.execute)
        self.msgbus.register(_RISK_ENDPOINT, self._gate_execute)
        self._gate_installed = True

    def uninstall_gate(self) -> None:
        """Restore NT's direct ``RiskEngine.execute`` endpoint."""
        if not self._gate_installed:
            return
        self.msgbus.deregister(_RISK_ENDPOINT, self._gate_execute)
        self.msgbus.register(_RISK_ENDPOINT, self._risk_engine.execute)
        self._gate_installed = False

    # ------------------------------------------------------------------- gate

    def _gate_execute(self, command: Any) -> None:
        if isinstance(command, SubmitOrder):
            reason = self._checked_denial_reason(command.order)
            if reason is not None:
                self._deny(command.order, reason)
                return
        elif isinstance(command, SubmitOrderList):
            for order in command.order_list.orders:
                reason = self._checked_denial_reason(order)
                if reason is not None:
                    for o in command.order_list.orders:
                        self._deny(o, reason)
                    return
        self._risk_engine.execute(command)

    def _checked_denial_reason(self, order: Any) -> str | None:
        """``_denial_reason`` with a fail-safe wrapper: a raising entry check DENIES the
        order instead of letting the exception escape. An escaped exception leaves the order
        INITIALIZED, so ``_unclosed_orders`` keeps counting it (its notional in the margin
        sum, its instrument pending-refused) until a restart. Logged by class only: the
        check's own message may carry venue/account data."""
        try:
            return self._denial_reason(order)
        except Exception as exc:
            logger.error("entry check failed (%s); order denied", type(exc).__name__)
            return f"{ENTRY_CHECK_FAILED_PREFIX}{type(exc).__name__}); order denied"

    def _denial_reason(self, order: Any) -> str | None:
        if order.is_reduce_only:
            return None
        instrument_id = order.instrument_id
        net = Decimal(str(self.portfolio.net_position(instrument_id)))
        qty = order.quantity.as_decimal()
        if _order_reduces(order.side, qty, net):
            return None

        if self._halt.state != GuardState.RUNNING:
            why = self._halt.reason.value if self._halt.reason else self._halt.state.value
            return (
                f"{GATE_DENIAL_PREFIX}: trading {self._halt.state.value} ({why}); "
                "only position-reducing orders allowed"
            )
        if net != 0:
            return (
                f"{GATE_DENIAL_PREFIX}: {instrument_id} already has net position {net}; "
                "order would add to or flip it"
            )
        pending = unique_pending_entries(
            self._unclosed_orders(instrument_id=instrument_id),
            (),
            exclude_client_order_id=order.client_order_id,
        )
        if pending:
            return (
                f"{GATE_DENIAL_PREFIX}: entry order {pending[0].client_order_id} for "
                f"{instrument_id} still pending"
            )
        open_count = len(self.cache.positions_open())
        if open_count >= self._limits.max_position_count:
            return (
                f"{GATE_DENIAL_PREFIX}: {open_count} open positions >= "
                f"max_position_count {self._limits.max_position_count}"
            )
        return self._margin_denial(order)

    # ----------------------------------------------------------- margin limit

    def _margin_denial(self, order: Any) -> str | None:
        """Refuse an entry whose margin (with every open position and pending entry) would
        exceed ``max_margin_pct`` of equity. Venue-verified leverage only (DR1); an entry
        is never shrunk. Runs LAST so state/net/pending/count denials keep precedence."""
        reason = self._margin_reason(order)
        if reason is not None:
            return f"{MARGIN_DENIAL_PREFIX}{reason}"
        try:
            self._listener.on_entry_allowed(str(order.instrument_id))
        except Exception as exc:
            logger.error("entry-allowed notify failed (%s)", type(exc).__name__)
        return None

    def _margin_reason(self, order: Any) -> str | None:
        account = self.portfolio.account(self._venue)
        pct = self._cfg.max_margin_pct
        if account is None:
            # No account means neither equity nor leverage is known; the equity reason
            # is the one the operator can act on.
            return check_entry_margin(
                open_items=[],
                pending_items=[],
                new_item=MarginItem(Decimal(0), 1),
                equity=None,
                max_margin_pct=pct,
            ).reason
        open_items: list[MarginItem] = []
        for pos in self.cache.positions_open(venue=self._venue):
            iid = pos.instrument_id
            notional = (
                abs(Decimal(str(pos.quantity)))
                * Decimal(str(pos.avg_px_open))
                * self._multiplier(iid)
            )
            open_items.append(MarginItem(notional, self._verified_leverage(account, iid), str(iid)))
        pending_items: list[MarginItem] = []
        pending = unique_pending_entries(
            self._unclosed_orders(venue=self._venue),
            (),
            exclude_client_order_id=order.client_order_id,
        )
        for pending_order in pending:
            item = self._order_item(account, pending_order, Decimal(str(pending_order.leaves_qty)))
            if isinstance(item, str):
                return item
            pending_items.append(item)
        new_item = self._order_item(account, order, Decimal(str(order.quantity)))
        if isinstance(new_item, str):
            return new_item
        return check_entry_margin(
            open_items=open_items,
            pending_items=pending_items,
            new_item=new_item,
            equity=self.current_equity(),
            max_margin_pct=pct,
        ).reason

    def _unclosed_orders(self, **filters: Any) -> list[Any]:
        """Every order that is not closed: INITIALIZED + open + inflight + emulated.

        An entry this gate admitted stays INITIALIZED until the (queued, on a live node)
        risk engine and exec client pick it up, and NT indexes it in neither
        ``orders_open`` nor ``orders_inflight``: a second entry in the same synchronous step
        would see no pending margin or pending entry. The cache scan needs no bookkeeping
        of its own (nothing to prune when an order is denied, rejected, cancelled or filled)
        and is the ONE pending set for both the pending-entry check and the margin sum.
        """
        return [o for o in self.cache.orders(**filters) if not o.is_closed]

    def _order_item(self, account: Any, order: Any, qty: Decimal) -> MarginItem | str:
        """The margin item of an order, or the refusal text when it cannot be priced."""
        iid = order.instrument_id
        px = self._reference_price(order)
        if px is None:
            return f"no reference price for {iid}"
        notional = qty * px * self._multiplier(iid)
        return MarginItem(notional, self._verified_leverage(account, iid), str(iid))

    def _reference_price(self, order: Any) -> Decimal | None:
        """Order price, then trigger price, then the cache's LAST, then MID."""
        for attr in ("price", "trigger_price"):
            value = getattr(order, attr, None)
            if value is not None:
                return Decimal(str(value))
        for price_type in (PriceType.LAST, PriceType.MID):
            value = self.cache.price(order.instrument_id, price_type)
            if value is not None:
                return Decimal(str(value))
        return None

    def _multiplier(self, instrument_id: Any) -> Decimal:
        instrument = self.cache.instrument(instrument_id)
        return Decimal(str(instrument.multiplier)) if instrument is not None else Decimal(1)

    @staticmethod
    def _verified_leverage(account: Any, instrument_id: Any) -> Decimal | None:
        """``account.leverage(id)``: what the venue's symbolConfig recorded, else None."""
        getter = getattr(account, "leverage", None)
        raw = getter(instrument_id) if callable(getter) else None
        if raw is None:
            return None
        value = Decimal(str(raw))
        # Out of the rule's domain is as unverified as missing: never raise in the gate.
        return value if Decimal(1) <= value <= Decimal(125) else None

    def _deny(self, order: Any, reason: str) -> None:
        if order.is_closed:
            return
        if not self.cache.order_exists(order.client_order_id):
            self.cache.add_order(order)
        self._gate_denials += 1
        denied = OrderDenied(
            trader_id=order.trader_id,
            strategy_id=order.strategy_id,
            instrument_id=order.instrument_id,
            client_order_id=order.client_order_id,
            reason=reason,
            event_id=UUID4(),
            ts_init=self.clock.timestamp_ns(),
        )
        self.msgbus.send(endpoint="ExecEngine.process", msg=denied)

    # ----------------------------------------------------------------- events

    def _on_tick(self, _event: Any) -> None:
        if self._halt.state == GuardState.HALTED and self._restored_flatten_pending:
            self._restored_flatten_pending = False
            reason = self._halt.reason
            errors = self._flatten(stop_strategies=reason not in _KEEP_STRATEGIES_RUNNING).errors
            for error in errors:
                if error.startswith(_UNOWNED_PREFIX):
                    continue  # _flatten already warned it (vibe-quant-v4fi4)
                self._listener.on_warning(f"restored halt flatten: {error}")
        self._stop_exited_strategies()
        self.evaluate_risk()

    def _on_account_event(self, _event: Any) -> None:
        self.evaluate_risk()

    def _on_order_event(self, event: Any) -> None:
        if isinstance(event, OrderRejected):
            self._listener.on_order_rejected(event)
        elif isinstance(event, OrderDenied):
            by_gate = str(event.reason).startswith(GATE_DENIAL_PREFIX)
            self._listener.on_order_denied(event, by_gate=by_gate)

    def _on_position_event(self, event: Any) -> None:
        guarded = self._is_guarded_strategy(event.strategy_id)
        closed_guarded = False
        if isinstance(event, PositionOpened):
            if guarded:
                self._listener.on_position_opened(event)
        elif isinstance(event, PositionClosed):
            if guarded:
                self._listener.on_position_closed(event)
                pnl = event.realized_pnl
                if pnl is not None and pnl.as_decimal() < 0:
                    self._risk.consecutive_losses += 1
                else:
                    self._risk.consecutive_losses = 0
                closed_guarded = True
            self._stop_exited_strategies()
        self.evaluate_risk()
        if closed_guarded:
            self._listener.on_position_close_settled(event)

    # ------------------------------------------------------------------- risk

    def current_equity(self) -> Decimal | None:
        """Wallet balance + unrealized PnL in the settlement currency, or None."""
        account = self.portfolio.account(self._venue)
        if account is None:
            self._warn_once("no_account", f"no {self._venue} account yet; risk checks paused")
            return None
        missing = self.portfolio.missing_price_instruments(self._venue)
        if missing:
            self._warn_once(
                f"unpriced:{sorted(str(i) for i in missing)}",
                f"no price for open position(s) {sorted(str(i) for i in missing)}; "
                "equity unknown, risk checks paused",
            )
            return None
        equity = self.portfolio.equity(self._venue).get(self._currency)
        if equity is None:
            self._warn_once(
                "no_ccy", f"account has no {self._currency} balance; risk checks paused"
            )
            return None
        value: Decimal = equity.as_decimal()
        return value

    def evaluate_risk(self) -> None:
        """Update HWM / daily baseline and halt on any breached limit."""
        equity = self.current_equity()
        if equity is None:
            return
        risk = self._risk
        risk.last_equity = equity
        today = self._utc_now().date()
        if risk.day != today:
            risk.day = today
            risk.day_start_equity = equity
            if (
                self._halt.state == GuardState.HALTED
                and self._halt.reason == HaltReason.MAX_DAILY_LOSS
            ):
                self._lift_halt(f"UTC day rollover to {today.isoformat()}")
        if risk.high_water_mark is None or equity > risk.high_water_mark:
            risk.high_water_mark = equity

        metrics: dict[str, object] = {
            "equity": str(equity),
            "high_water_mark": _dec_str(risk.high_water_mark),
            "day_start_equity": _dec_str(risk.day_start_equity),
            "consecutive_losses": risk.consecutive_losses,
        }
        dd = risk.drawdown_pct
        if dd is not None and dd >= self._limits.max_drawdown_pct:
            self.halt(
                HaltReason.MAX_DRAWDOWN,
                f"drawdown {dd:.2%} >= limit {self._limits.max_drawdown_pct:.2%} "
                f"(equity {equity}, high water mark {risk.high_water_mark})"
                f"{self._risk_hint}",
                metrics={**metrics, "drawdown_pct": str(dd)},
            )
            return
        if risk.consecutive_losses >= self._limits.max_consecutive_losses:
            self.halt(
                HaltReason.MAX_CONSECUTIVE_LOSSES,
                f"{risk.consecutive_losses} consecutive losing positions >= limit "
                f"{self._limits.max_consecutive_losses}{self._risk_hint}",
                metrics=metrics,
            )
            return
        daily = risk.daily_loss_pct
        if daily is not None and daily >= self._limits.max_daily_loss_pct:
            self.halt(
                HaltReason.MAX_DAILY_LOSS,
                f"daily loss {daily:.2%} >= limit {self._limits.max_daily_loss_pct:.2%} "
                f"(equity {equity}, UTC-day start {risk.day_start_equity}); "
                "halted until next UTC day",
                metrics={**metrics, "daily_loss_pct": str(daily)},
            )

    # ---------------------------------------------------------------- control

    def halt(
        self,
        reason: HaltReason,
        message: str,
        metrics: dict[str, object] | None = None,
    ) -> bool:
        """Halt trading: block entries, flatten, cancel orders, stop strategies.

        Returns True if the halt was applied (False when an equal-or-more-severe
        halt is already active).
        """
        if not self._halt.apply(reason, message):
            return False
        # This halt flattens right here: a deferred restored flatten must not
        # run again on the first tick (H8: exactly once).
        self._restored_flatten_pending = False
        stop = reason not in _KEEP_STRATEGIES_RUNNING
        result = self._flatten(stop_strategies=stop)
        errors = result.errors
        self._prestart_unowned = list(result.unowned_positions) if not self._started else None
        payload: dict[str, object] = dict(metrics or {})
        if errors:
            payload["flatten_errors"] = errors
        logger.warning("trading halted (%s): %s", reason.value, message)
        self._listener.on_state_change(self._halt.state, reason, message, payload)
        return True

    def pause(self, message: str = "paused by operator") -> bool:
        """Block new entries; keep SL/TP and exit logic. No-op unless RUNNING."""
        if not self._halt.pause(message):
            return False
        self._listener.on_state_change(self._halt.state, None, message, {})
        return True

    def resume(self, *, kill_switch_engaged: bool = False) -> tuple[bool, str]:
        """Resume from pause or an operator-resumable halt (never while killed)."""
        restarting = self._halt.state == GuardState.PAUSED
        ok, text, lift = self._halt.resume(kill_switch_engaged=kill_switch_engaged)
        if lift:
            self._lift_halt("resumed by operator")
        elif ok and restarting:
            self._listener.on_state_change(self._halt.state, None, self._halt.message, {})
        return ok, text

    def close_all(self) -> CloseAllResult:
        """Flatten every position and cancel every order; strategies keep running."""
        return self._flatten(stop_strategies=False)

    # -------------------------------------------------------------- internals

    def _lift_halt(self, message: str) -> None:
        self._restored_flatten_pending = False
        for strategy in self._strategies:
            sid = str(strategy.id)
            self._pending_stop.discard(sid)
            self._pending_exit.discard(sid)
            if strategy.is_stopped:
                strategy.resume()
                sync = getattr(strategy, "_sync_position_state", None)
                if callable(sync):
                    sync()
        self._halt.lift(message)
        self._listener.on_state_change(self._halt.state, None, message, {})

    def _split_positions(self) -> tuple[list[str], list[str]]:
        """Open positions split into (guarded strategy ids, unowned descriptions)."""
        targeted: list[str] = []
        unowned: list[str] = []
        guarded_ids = {str(s.id) for s in self._strategies}
        for position in self.cache.positions_open():
            if str(position.strategy_id) in guarded_ids:
                targeted.append(str(position.id))
            else:
                unowned.append(
                    f"{position.id} (strategy {position.strategy_id}, {position.instrument_id})"
                )
        return targeted, unowned

    def _flatten(self, *, stop_strategies: bool) -> CloseAllResult:
        result = CloseAllResult()
        result.targeted_positions, result.unowned_positions = self._split_positions()

        for strategy in self._strategies:
            sid = str(strategy.id)
            try:
                if strategy.is_running:
                    if not strategy.is_exiting():
                        strategy.market_exit()
                    if stop_strategies:
                        self._pending_stop.add(sid)
                elif stop_strategies and strategy.state in _STRATEGY_NOT_STARTED:
                    # Not started yet (READY/STARTING/PRE_INITIALIZED): a stopping
                    # halt tripped before NT started this strategy, so there is
                    # nothing to flatten yet, but it must still be stopped.
                    # _stop_exited_strategies exits and stops it once it runs.
                    self._pending_stop.add(sid)
                    self._pending_exit.add(sid)
                else:
                    self._flatten_stopped_strategy(strategy)
            except Exception as exc:
                logger.exception("flatten failed for strategy %s", sid)
                result.errors.append(f"{sid}: {type(exc).__name__}: {exc}")

        if result.unowned_positions:
            msg = _UNOWNED_PREFIX + ", ".join(result.unowned_positions)
            result.errors.append(msg)
            self._listener.on_warning(msg)
        return result

    def _flatten_stopped_strategy(self, strategy: Any) -> None:
        """A stopped strategy gets no events; cancel + close via its order API."""
        instruments = {o.instrument_id for o in self.cache.orders_open(strategy_id=strategy.id)}
        instruments |= {p.instrument_id for p in self.cache.positions_open(strategy_id=strategy.id)}
        for instrument_id in instruments:
            strategy.cancel_all_orders(instrument_id)
            strategy.close_all_positions(instrument_id, reduce_only=True)

    def _stop_exited_strategies(self) -> None:
        if not self._pending_stop:
            return
        for strategy in self._strategies:
            sid = str(strategy.id)
            if sid not in self._pending_stop:
                continue
            if not strategy.is_running:
                if strategy.is_stopped:
                    # A stopped strategy is not coming back; its orders were
                    # handled by _flatten_stopped_strategy, so drop the entry.
                    self._pending_stop.discard(sid)
                    self._pending_exit.discard(sid)
                # A recorded strategy that has not started yet keeps waiting.
                continue
            if sid in self._pending_exit:
                # Recorded before it started: give it its market exit now that it
                # runs (a no-op when flat) and wait for that exit to finish.
                self._pending_exit.discard(sid)
                if not strategy.is_exiting():
                    strategy.market_exit()
                continue
            if strategy.is_exiting():
                continue
            if self.cache.positions_open(strategy_id=strategy.id):
                # market_exit gave up with positions still open: try again.
                self._listener.on_warning(
                    f"{sid}: market exit finished with open positions; retrying flatten"
                )
                strategy.market_exit()
                continue
            self._pending_stop.discard(sid)
            strategy.stop()

    def _is_guarded_strategy(self, strategy_id: Any) -> bool:
        sid = str(strategy_id)
        return any(str(s.id) == sid for s in self._strategies)

    def _utc_now(self) -> datetime:
        now: datetime = self.clock.utc_now()
        return now.astimezone(UTC)

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning("trading guard: %s", message)
        self._listener.on_warning(message)


__all__ = [
    "ENTRY_CHECK_FAILED_PREFIX",
    "GATE_DENIAL_PREFIX",
    "MARGIN_DENIAL_PREFIX",
    "RESTORED_HALTS",
    "RESTORED_HALT_PREFIX",
    "CloseAllResult",
    "GuardListener",
    "GuardState",
    "OPERATOR_RESUMABLE_HALTS",
    "HaltReason",
    "RestoredHalt",
    "RiskLimits",
    "RiskState",
    "TradingGuard",
    "TradingGuardConfig",
    "halt_supersedes",
]
