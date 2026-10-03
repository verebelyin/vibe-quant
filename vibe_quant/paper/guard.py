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
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.execution.messages import SubmitOrder, SubmitOrderList
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.events import (
    OrderDenied,
    OrderRejected,
    PositionClosed,
    PositionOpened,
)
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Currency

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

#: Prefix on every OrderDenied reason produced by the guard's gate.
GATE_DENIAL_PREFIX = "VQ_GUARD"

_RISK_ENDPOINT = "RiskEngine.execute"
_TICK_TIMER = "TradingGuard.tick"


class GuardState(StrEnum):
    """Trading state owned by the guard."""

    RUNNING = "running"
    PAUSED = "paused"
    HALTED = "halted"


class HaltReason(StrEnum):
    """Reason for halting the trading node."""

    MAX_DRAWDOWN = "max_drawdown"
    MAX_DAILY_LOSS = "max_daily_loss"
    MAX_CONSECUTIVE_LOSSES = "max_consecutive_losses"
    MANUAL = "manual"
    ERROR = "error"
    SIGNAL = "signal"
    KILL_SWITCH = "kill_switch"


# A new halt only replaces the current one when it is at least as severe.
_HALT_PRIORITY: dict[HaltReason, int] = {
    HaltReason.MAX_DAILY_LOSS: 1,
    HaltReason.ERROR: 2,
    HaltReason.MANUAL: 3,
    HaltReason.SIGNAL: 3,
    HaltReason.MAX_CONSECUTIVE_LOSSES: 4,
    HaltReason.MAX_DRAWDOWN: 5,
    HaltReason.KILL_SWITCH: 6,
}

# Halts that flatten and block entries but keep strategies running so they can
# resume automatically (daily loss lifts at the next UTC midnight).
_KEEP_STRATEGIES_RUNNING: frozenset[HaltReason] = frozenset({HaltReason.MAX_DAILY_LOSS})

# Halts an operator may resume from without restarting the session.
_OPERATOR_RESUMABLE: frozenset[HaltReason] = frozenset(
    {HaltReason.MANUAL, HaltReason.SIGNAL, HaltReason.ERROR, HaltReason.KILL_SWITCH}
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
    def from_dict(cls, data: dict[str, Any] | None) -> RiskState:
        """Restore from :meth:`to_dict` output; unknown/missing keys -> defaults."""
        if not data:
            return cls()
        day_raw = data.get("day")
        losses_raw = data.get("consecutive_losses", 0)
        return cls(
            high_water_mark=_dec_or_none(data.get("high_water_mark")),
            day=date.fromisoformat(day_raw) if isinstance(day_raw, str) else None,
            day_start_equity=_dec_or_none(data.get("day_start_equity")),
            consecutive_losses=int(losses_raw) if isinstance(losses_raw, (int, str)) else 0,
            last_equity=_dec_or_none(data.get("last_equity")),
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

    def on_warning(self, message: str) -> None:
        """Non-fatal condition the operator should know about."""


class TradingGuardConfig(ActorConfig, frozen=True):  # type: ignore[misc]
    """Configuration for :class:`TradingGuard`."""

    venue: str = "BINANCE"
    settlement_currency: str = "USDT"
    max_drawdown_pct: Decimal = Decimal("0.15")
    max_daily_loss_pct: Decimal = Decimal("0.02")
    max_consecutive_losses: int = 10
    max_position_count: int = 5
    check_interval_secs: int = 5


def _dec_str(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _dec_or_none(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, str, Decimal)):
        try:
            return Decimal(str(value))
        except ArithmeticError:
            return None
    return None


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
        self._venue = Venue(config.venue)
        self._currency = Currency.from_str(config.settlement_currency)
        self._state = GuardState.RUNNING
        self._halt_reason: HaltReason | None = None
        self._message = ""
        self._pending_stop: set[str] = set()
        self._gate_installed = False
        self._warned: set[str] = set()
        self._gate_denials = 0

    # ------------------------------------------------------------------ props

    @property
    def guard_state(self) -> GuardState:
        return self._state

    @property
    def halt_reason(self) -> HaltReason | None:
        return self._halt_reason

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
            "state": self._state.value,
            "halt_reason": self._halt_reason.value if self._halt_reason else None,
            "message": self._message,
            "gate": "active" if self._state == GuardState.RUNNING else "reduce_only",
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
        self.install_gate()
        self.msgbus.subscribe(topic="events.position.*", handler=self._on_position_event)
        self.msgbus.subscribe(topic="events.order.*", handler=self._on_order_event)
        self.msgbus.subscribe(topic="events.account.*", handler=self._on_account_event)
        self.clock.set_timer(
            name=_TICK_TIMER,
            interval=timedelta(seconds=self._cfg.check_interval_secs),
            callback=self._on_tick,
        )
        self.evaluate_risk()

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
            reason = self._denial_reason(command.order)
            if reason is not None:
                self._deny(command.order, reason)
                return
        elif isinstance(command, SubmitOrderList):
            for order in command.order_list.orders:
                reason = self._denial_reason(order)
                if reason is not None:
                    for o in command.order_list.orders:
                        self._deny(o, reason)
                    return
        self._risk_engine.execute(command)

    def _denial_reason(self, order: Any) -> str | None:
        if order.is_reduce_only:
            return None
        instrument_id = order.instrument_id
        net = Decimal(str(self.portfolio.net_position(instrument_id)))
        qty = order.quantity.as_decimal()
        if _order_reduces(order.side, qty, net):
            return None

        if self._state != GuardState.RUNNING:
            why = self._halt_reason.value if self._halt_reason else self._state.value
            return (
                f"{GATE_DENIAL_PREFIX}: trading {self._state.value} ({why}); "
                "only position-reducing orders allowed"
            )
        if net != 0:
            return (
                f"{GATE_DENIAL_PREFIX}: {instrument_id} already has net position {net}; "
                "order would add to or flip it"
            )
        pending = [
            o
            for o in (
                *self.cache.orders_open(instrument_id=instrument_id),
                *self.cache.orders_inflight(instrument_id=instrument_id),
            )
            if not o.is_reduce_only and o.client_order_id != order.client_order_id
        ]
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
        return None

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
            self._stop_exited_strategies()
        self.evaluate_risk()

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
            if self._state == GuardState.HALTED and self._halt_reason == HaltReason.MAX_DAILY_LOSS:
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
                f"(equity {equity}, high water mark {risk.high_water_mark})",
                metrics={**metrics, "drawdown_pct": str(dd)},
            )
            return
        if risk.consecutive_losses >= self._limits.max_consecutive_losses:
            self.halt(
                HaltReason.MAX_CONSECUTIVE_LOSSES,
                f"{risk.consecutive_losses} consecutive losing positions >= limit "
                f"{self._limits.max_consecutive_losses}",
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
        current = self._halt_reason
        if (
            self._state == GuardState.HALTED
            and current is not None
            and _HALT_PRIORITY[reason] <= _HALT_PRIORITY[current]
        ):
            return False
        self._state = GuardState.HALTED
        self._halt_reason = reason
        self._message = message
        stop = reason not in _KEEP_STRATEGIES_RUNNING
        errors = self._flatten(stop_strategies=stop).errors
        payload: dict[str, object] = dict(metrics or {})
        if errors:
            payload["flatten_errors"] = errors
        logger.warning("trading halted (%s): %s", reason.value, message)
        self._listener.on_state_change(self._state, reason, message, payload)
        return True

    def pause(self, message: str = "paused by operator") -> bool:
        """Block new entries; keep SL/TP and exit logic. No-op unless RUNNING."""
        if self._state != GuardState.RUNNING:
            return False
        self._state = GuardState.PAUSED
        self._message = message
        self._listener.on_state_change(self._state, None, message, {})
        return True

    def resume(self, *, kill_switch_engaged: bool = False) -> tuple[bool, str]:
        """Resume from pause or an operator-resumable halt."""
        if self._state == GuardState.RUNNING:
            return True, "already running"
        if self._state == GuardState.PAUSED:
            self._state = GuardState.RUNNING
            self._message = "resumed"
            self._listener.on_state_change(self._state, None, self._message, {})
            return True, "resumed from pause"
        reason = self._halt_reason
        if kill_switch_engaged:
            return False, "system kill switch is engaged; unlock it before resuming"
        if reason == HaltReason.MAX_DAILY_LOSS:
            return False, "daily-loss halt lifts automatically at the next UTC day"
        if reason not in _OPERATOR_RESUMABLE:
            why = reason.value if reason else "unknown"
            return False, f"{why} halt requires manual review; restart the session to trade again"
        self._lift_halt("resumed by operator")
        return True, "resumed from halt"

    def close_all(self) -> CloseAllResult:
        """Flatten every position and cancel every order; strategies keep running."""
        return self._flatten(stop_strategies=False)

    # -------------------------------------------------------------- internals

    def _lift_halt(self, message: str) -> None:
        for strategy in self._strategies:
            sid = str(strategy.id)
            self._pending_stop.discard(sid)
            if strategy.is_stopped:
                strategy.resume()
                sync = getattr(strategy, "_sync_position_state", None)
                if callable(sync):
                    sync()
        self._state = GuardState.RUNNING
        self._halt_reason = None
        self._message = message
        self._listener.on_state_change(self._state, None, message, {})

    def _flatten(self, *, stop_strategies: bool) -> CloseAllResult:
        result = CloseAllResult()
        guarded_ids = {str(s.id) for s in self._strategies}
        for position in self.cache.positions_open():
            if str(position.strategy_id) in guarded_ids:
                result.targeted_positions.append(str(position.id))
            else:
                result.unowned_positions.append(
                    f"{position.id} (strategy {position.strategy_id}, {position.instrument_id})"
                )

        for strategy in self._strategies:
            sid = str(strategy.id)
            try:
                if strategy.is_running:
                    if not strategy.is_exiting():
                        strategy.market_exit()
                    if stop_strategies:
                        self._pending_stop.add(sid)
                else:
                    self._flatten_stopped_strategy(strategy)
            except Exception as exc:
                logger.exception("flatten failed for strategy %s", sid)
                result.errors.append(f"{sid}: {type(exc).__name__}: {exc}")

        if result.unowned_positions:
            msg = "positions not owned by any guarded strategy were NOT closed: " + ", ".join(
                result.unowned_positions
            )
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
                self._pending_stop.discard(sid)
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
    "GATE_DENIAL_PREFIX",
    "CloseAllResult",
    "GuardListener",
    "GuardState",
    "HaltReason",
    "RiskLimits",
    "RiskState",
    "TradingGuard",
    "TradingGuardConfig",
]
