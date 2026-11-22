"""Paper-side SL/TP repair of a bot-owned open position (opspt P3, design F3).

The generated strategy's ``_ensure_protection`` re-arms a position only when it has NO
leg and otherwise resizes what it finds, so a position left with one leg (TP resting,
SL refused) is never completed; its ATR-not-ready retry also runs inside ``on_bar``
BEFORE that bar's exit (legs placed, then cancelled by the exit). The paper node runs
:func:`repair` instead, always outside ``on_bar`` (generated templates unchanged: no
backtest number moves):

* only an owned position (the bot's own record equals NT's net), never a hand one;
* the strategy's non-terminal reduce-only STOP_MARKET (stop-loss) and LIMIT
  (take-profit) legs; an order some other path is cancelling, or a leg Binance has not
  answered yet, makes it ``unresolved``: nothing is sent;
* a missing leg gets its level from the strategy's own ``_calculate_sl_price`` /
  ``_calculate_tp_price`` at the position's entry price; an ATR level not ready yet is
  owed (``retry``, asked again; the other leg is still placed); otherwise that leg alone
  is submitted. No local price decides a stop (paper subscribes bars only): Binance
  refuses one the market already crossed (-2021), named as such for the user;
* a leg Binance refused is not sent again at the same level for that position (memo);
* an undersized leg (a partial fill grew the position) is cancelled and replaced at its
  own level only after Binance confirmed the cancel;
* ``protected`` only once every leg is accepted; a repair leg Binance refused, or a
  resize whose cancel it refused once, is ``unprotected`` (with the legs still owed);
* flat on Binance, NT and the bot's record (RO-R2): the bot's own leftover reduce-only
  legs of that strategy are cancelled by their ids.

The caller keeps one :class:`RepairMemo` per strategy for the session (it resets when
the position changes) and calls again on order events until the report is final.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nautilus_trader.model.enums import OrderStatus, OrderType, PositionSide, TimeInForce
from nautilus_trader.model.events import OrderRejected

if TYPE_CHECKING:
    from collections.abc import Callable
    from decimal import Decimal

PROTECTED = "protected"
#: Legs sent or cancelled by this repair are not answered yet.
PENDING = "pending"
#: An order another path sent or is cancelling is unanswered: nothing is sent.
UNRESOLVED = "unresolved"
#: An ATR level is not ready yet: nothing sent for that leg, ask again later.
RETRY = "retry"
#: A leg cannot be placed (stop already crossed, rejected, no level): set it by hand.
UNPROTECTED = "unprotected"
FLAT = "flat"
#: The open position is not the bot's own record: nothing is sent.
NOT_OWNED = "not_owned"

STOP_LOSS = "stop-loss"
TAKE_PROFIT = "take-profit"
_LEG_TYPES = {STOP_LOSS: OrderType.STOP_MARKET, TAKE_PROFIT: OrderType.LIMIT}
_SHORT_NAMES = {STOP_LOSS: "stop", TAKE_PROFIT: "take-profit"}
#: Order states in which a leg is unanswered (sent, not yet accepted or refused).
_UNANSWERED = (
    OrderStatus.INITIALIZED,
    OrderStatus.SUBMITTED,
    OrderStatus.PENDING_UPDATE,
    OrderStatus.PENDING_CANCEL,
)
_WORKING = (OrderStatus.ACCEPTED, OrderStatus.TRIGGERED, OrderStatus.PARTIALLY_FILLED)
_ATR_STOPS = ("atr_fixed", "atr_trailing")
_REASON_MAX_CHARS = 200
#: Binance's answer to a stop the market already crossed (code -2021).
_WOULD_TRIGGER = ("-2021", "would immediately trigger")
#: A refusal by the bot's own wire gate while held (opspt P1/P4), not by Binance: the
#: leg is sent again after the hold clears.
_WIRE_REFUSAL = "VQ venue "


@dataclass(frozen=True)
class ProtectionReport:
    """One repair pass: ``state`` (module constants), the user text of an
    ``unprotected`` report (every missing leg), what was sent (``"stop-loss"``,
    ``"cancel take-profit"``), the legs still owed (an ATR level not ready yet) and
    whether a leg cannot be placed that was not reported for this position before."""

    state: str
    detail: str | None = None
    sent: tuple[str, ...] = ()
    owed: tuple[str, ...] = ()
    new_problem: bool = False


@dataclass
class RepairMemo:
    """What one strategy's repair sent, cancelled and reported for its open position
    (reset when the position changes)."""

    #: (position id, open time) the memo is about; None = flat.
    position: tuple[str, int] | None = None
    #: client order id -> leg, for every leg this repair submitted.
    sent: dict[str, str] = field(default_factory=dict)
    #: client order id -> (leg, its level), for every undersized leg it cancelled.
    cancelled: dict[str, tuple[str, Any]] = field(default_factory=dict)
    #: leg -> the level of its cancelled undersized order, to replace it at.
    replace: dict[str, Any] = field(default_factory=dict)
    #: legs whose resize Binance refused once already.
    resize_refused: set[str] = field(default_factory=set)
    #: leg -> (level, Binance's reason) of a repair leg Binance refused: never re-sent
    #: at that level.
    refused: dict[str, tuple[Any, str]] = field(default_factory=dict)
    #: leg -> every "cannot be placed" text already reported for it on this position
    #: (forgotten once the leg rests, so the same problem recurring alerts again).
    reported: dict[str, set[str]] = field(default_factory=dict)

    def follow(self, position: tuple[str, int] | None) -> None:
        """Start over when the open position is another one (or none)."""
        if position != self.position:
            fresh = RepairMemo(position=position)
            self.__dict__.update(fresh.__dict__)


def open_position(strategy: Any) -> Any:
    """The strategy's open position on its instrument (as ``_ensure_protection`` finds
    it), else None."""
    iid = strategy.instrument_id
    for position in strategy.cache.positions_open(venue=iid.venue):
        if position.instrument_id == iid and position.is_open:
            return position
    return None


def _symbol(instrument_id: Any) -> str:
    return str(instrument_id).split(".")[0].removesuffix("-PERP")


def _required_legs(strategy: Any, is_long: bool) -> list[str]:
    """The legs the strategy's config asks for (a leg with no type is not placed)."""
    legs = []
    if strategy._sl_config(is_long)[0]:
        legs.append(STOP_LOSS)
    if strategy._tp_config(is_long)[0]:
        legs.append(TAKE_PROFIT)
    return legs


def _uses_atr(strategy: Any, leg: str, is_long: bool) -> bool:
    """Does ``leg``'s level read an ATR indicator (the template's own rule, per leg)?"""
    sl_atr = strategy._sl_config(is_long)[0] in _ATR_STOPS
    if leg == STOP_LOSS:
        return bool(sl_atr)
    tp_type = strategy._tp_config(is_long)[0]
    return tp_type == "atr_fixed" or (tp_type == "risk_reward" and sl_atr)


def _level(strategy: Any, leg: str, entry: float, is_long: bool) -> float | None:
    """The leg's level at ``entry`` from the strategy's own math (a trailing stop never
    looser than the trail already reached)."""
    if leg == TAKE_PROFIT:
        level: float | None = strategy._calculate_tp_price(entry, is_long)
        return level
    level = strategy._calculate_sl_price(entry, is_long)
    best = getattr(strategy, "_trailing_best_sl", None)
    if strategy._sl_config(is_long)[0] == "atr_trailing" and best is not None:
        level = best if level is None else (max(level, best) if is_long else min(level, best))
    return level


def _reject_reason(order: Any) -> str:
    reasons = [str(e.reason) for e in order.events if isinstance(e, OrderRejected)]
    return (reasons[-1] if reasons else "rejected")[:_REASON_MAX_CHARS]


def _unprotected(
    symbol: str, position: Any, leg: str, why: str, action: str = "set one by hand, then restart"
) -> str:
    side = "long" if position.side == PositionSide.LONG else "short"
    return (
        f"{symbol}: bot {side} {position.quantity} has no {_SHORT_NAMES[leg]} on Binance "
        f"({why}); {action}"
    )


def _not_ready(symbol: str, position: Any, leg: str) -> str:
    """The note on a leg still owed (its ATR level not ready): the alert names it too."""
    side = "long" if position.side == PositionSide.LONG else "short"
    return (
        f"{symbol}: bot {side} {position.quantity} has no {_SHORT_NAMES[leg]} on Binance yet "
        "(its ATR level is not ready; the bot sends it at the first ready bar)"
    )


def _refused(symbol: str, position: Any, leg: str, level: Any, reason: str) -> str:
    """A repair leg Binance refused: a stop it says would trigger at once (R53), else
    its reason."""
    if leg == STOP_LOSS and any(mark in reason.lower() for mark in _WOULD_TRIGGER):
        why = f"stop {level} would trigger at once"
        return _unprotected(symbol, position, leg, why, "set one by hand or close it, then restart")
    return _unprotected(symbol, position, leg, f"Binance rejected it: {reason}")


def _level_of(order: Any) -> Any:
    return order.trigger_price if order.has_trigger_price else order.price


def _cancel_leftovers(
    strategy: Any, orders: list[Any], is_own: Callable[[str], bool]
) -> tuple[str, ...]:
    """RO-R2: cancel, by id, the strategy's own working reduce-only legs left on a coin
    that is flat everywhere (an SL/TP fill whose partner's cancel was refused)."""
    sent = []
    for order in orders:
        leg = next((k for k, t in _LEG_TYPES.items() if t == order.order_type), None)
        if (
            leg is None
            or not order.is_reduce_only
            or order.status not in _WORKING
            or not is_own(str(order.client_order_id))
        ):
            continue
        strategy.cancel_order(order)
        sent.append(f"cancel {leg}")
    return tuple(sent)


def _unresolved(orders: list[Any], memo: RepairMemo) -> bool:
    """An order of the strategy another path is cancelling, or a leg it sent that Binance
    has not answered: counting legs now could duplicate or miss one."""
    for order in orders:
        cid = str(order.client_order_id)
        if cid in memo.sent or cid in memo.cancelled:
            continue
        if order.status == OrderStatus.PENDING_CANCEL:
            return True
        if order.is_reduce_only and order.status in _UNANSWERED:
            return True
    return False


def _settle_memo(cache: Any, memo: RepairMemo) -> bool:
    """Fold Binance's answers into ``memo`` (a refused leg moves to ``refused``; one the
    bot's own wire refused while held is forgotten, so it is sent again). Returns True
    while a leg or cancel of this repair is unanswered."""
    from nautilus_trader.model.identifiers import ClientOrderId

    waiting = False
    for cid, leg in list(memo.sent.items()):
        order = cache.order(ClientOrderId(cid))
        if order is None:
            continue
        if order.status in _UNANSWERED:
            waiting = True
        elif order.status in (OrderStatus.REJECTED, OrderStatus.DENIED):
            del memo.sent[cid]
            reason = _reject_reason(order)
            if not reason.startswith(_WIRE_REFUSAL):
                memo.refused[leg] = (_level_of(order), reason)
    for cid, (leg, level) in list(memo.cancelled.items()):
        order = cache.order(ClientOrderId(cid))
        if order is not None and order.status == OrderStatus.PENDING_CANCEL:
            waiting = True
            continue
        del memo.cancelled[cid]
        if order is not None and order.status == OrderStatus.CANCELED:
            memo.replace[leg] = level  # confirmed: replace it at its own level
        elif order is not None and order.is_open:
            memo.resize_refused.add(leg)  # the cancel was refused: the old leg rests
    return waiting


def _position_key(position: Any) -> tuple[str, int] | None:
    return None if position is None else (str(position.id), int(position.ts_opened))


def repair(
    strategy: Any,
    *,
    owned: Decimal,
    memo: RepairMemo,
    is_own: Callable[[str], bool] | None = None,
    venue_flat: bool = False,
) -> ProtectionReport:
    """One repair pass over ``strategy``'s open position (module doc). ``owned`` is the
    bot's own signed record for the strategy's symbol; ``venue_flat``: Binance showed the
    coin flat at the pass that started this run; ``is_own``: is a client order id the
    bot's own (None = none is). Sends at most one order per missing or undersized leg;
    never cancels or replaces a leg it cannot confirm."""
    position = open_position(strategy)
    memo.follow(_position_key(position))
    iid = strategy.instrument_id
    cache = strategy.cache
    orders = [
        o for o in cache.orders(instrument_id=iid, strategy_id=strategy.id) if not o.is_closed
    ]
    if position is None:
        if owned != 0 or not venue_flat or is_own is None:
            return ProtectionReport(FLAT)
        return ProtectionReport(FLAT, sent=_cancel_leftovers(strategy, orders, is_own))
    quantity = position.quantity
    is_long = position.side == PositionSide.LONG
    if (quantity.as_decimal() if is_long else -quantity.as_decimal()) != owned:
        return ProtectionReport(NOT_OWNED)
    symbol = _symbol(iid)
    if _unresolved(orders, memo):
        return ProtectionReport(UNRESOLVED)
    if _settle_memo(cache, memo):
        return ProtectionReport(PENDING)
    problems: list[str] = []
    lines: list[str] = []  # every missing leg, in leg order: problems and owed notes
    leg_problems: dict[str, str] = {}
    sent: list[str] = []
    owed: list[str] = []
    entry = float(position.avg_px_open)
    exit_side = position.closing_order_side()
    instrument = cache.instrument(iid)
    for leg in _required_legs(strategy, is_long):
        legs = [
            o
            for o in orders
            if o.is_reduce_only and o.order_type == _LEG_TYPES[leg] and o.status in _WORKING
        ]
        if any(o.leaves_qty >= quantity for o in legs):
            memo.reported.pop(leg, None)
            continue
        problem = None
        if legs:
            if leg in memo.resize_refused:
                why = f"its {_SHORT_NAMES[leg]} covers less and Binance refused the resize"
                problem = _unprotected(symbol, position, leg, why)
            else:
                for order in legs:  # undersized: cancel first, replace once confirmed
                    memo.cancelled[str(order.client_order_id)] = (leg, _level_of(order))
                    strategy.cancel_order(order)
                sent.append(f"cancel {leg}")
                continue
        level = memo.replace.pop(leg, None)
        if problem is None and level is None:
            if _uses_atr(strategy, leg, is_long) and not strategy._indicators_ready():
                owed.append(leg)
                lines.append(_not_ready(symbol, position, leg))
                continue
            value = _level(strategy, leg, entry, is_long)
            if value is None or value <= 0:
                problem = _unprotected(symbol, position, leg, "no level computable")
            else:
                level = instrument.make_price(value)
        refused = memo.refused.get(leg)
        if problem is None and refused is not None and refused[0] == level:
            problem = _refused(symbol, position, leg, level, refused[1])
        if problem is not None:
            problems.append(problem)
            lines.append(problem)
            leg_problems[leg] = problem
            continue
        memo.refused.pop(leg, None)
        if leg == STOP_LOSS:
            order = strategy.order_factory.stop_market(
                instrument_id=iid,
                order_side=exit_side,
                quantity=quantity,
                trigger_price=level,
                time_in_force=TimeInForce.GTC,
                reduce_only=True,
            )
            if strategy._sl_config(is_long)[0] == "atr_trailing":
                strategy._trailing_best_sl = float(level)  # the trail starts from it
        else:
            order = strategy.order_factory.limit(
                instrument_id=iid,
                order_side=exit_side,
                quantity=quantity,
                price=level,
                time_in_force=TimeInForce.GTC,
                reduce_only=True,
            )
        memo.sent[str(order.client_order_id)] = leg
        strategy.submit_order(order)
        sent.append(leg)
    if problems:
        new = any(text not in memo.reported.get(leg, ()) for leg, text in leg_problems.items())
        for leg, text in leg_problems.items():
            memo.reported.setdefault(leg, set()).add(text)
        return ProtectionReport(
            UNPROTECTED, "; ".join(lines), tuple(sent), tuple(owed), new_problem=new
        )
    if sent:
        return ProtectionReport(PENDING, sent=tuple(sent), owed=tuple(owed))
    return ProtectionReport(RETRY if owed else PROTECTED, owed=tuple(owed))


__all__ = [
    "FLAT",
    "NOT_OWNED",
    "PENDING",
    "PROTECTED",
    "RETRY",
    "STOP_LOSS",
    "TAKE_PROFIT",
    "UNPROTECTED",
    "UNRESOLVED",
    "ProtectionReport",
    "RepairMemo",
    "open_position",
    "repair",
]
