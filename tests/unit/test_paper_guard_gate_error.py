"""Gate fail-safe: an entry check that RAISES denies the order instead of escaping.

Bead vibe-quant-muk2e. Today ``TradingGuard._gate_execute`` calls ``_denial_reason`` bare:
an exception (margin math, ``portfolio.net_position``) escapes, the order stays INITIALIZED,
and the cache scan in ``_unclosed_orders`` keeps counting it -- its notional stays in the
margin sum and the per-instrument pending check refuses every later entry on that instrument
until a restart. Fails safe but blocks trading.

These tests pin the fix: the order is DENIED explicitly (so it closes and stops counting),
the error is logged by CLASS ONLY, and trading on the instrument continues.

The fakes drive the REAL ``TradingGuard._gate_execute``/``_deny`` with real NT ``MarketOrder``
objects (``SubmitOrder`` type-checks the order) and real ``OrderDenied`` events; the fake
msgbus applies each ``OrderDenied`` to its order exactly as NT's ``ExecEngine`` does, so a
denied order becomes ``is_closed`` and leaves the pending set.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.execution.messages import SubmitOrder, SubmitOrderList
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.events import OrderDenied
from nautilus_trader.model.identifiers import (
    ClientOrderId,
    OrderListId,
    StrategyId,
    TraderId,
)
from nautilus_trader.model.orders import MarketOrder, OrderList

from vibe_quant.data.catalog import create_instrument
from vibe_quant.paper.guard import (
    GATE_DENIAL_PREFIX,
    GuardListener,
    TradingGuard,
    TradingGuardConfig,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

SYMBOL = "BTCUSDT"
_TS = 1_700_000_000_000_000_000


class _MarginError(Exception):
    """A broken entry check (the class name is what the contract pins)."""


def _order(
    tag: str, *, symbol: str = SYMBOL, reduce_only: bool = False
) -> MarketOrder:
    inst = create_instrument(symbol)
    return MarketOrder(
        trader_id=TraderId("TESTER-001"),
        strategy_id=StrategyId("S-001"),
        instrument_id=inst.id,
        client_order_id=ClientOrderId(tag),
        order_side=OrderSide.BUY,
        quantity=inst.make_qty("1"),
        init_id=UUID4(),
        ts_init=0,
        time_in_force=TimeInForce.IOC,
        reduce_only=reduce_only,
    )


def _submit(order: MarketOrder) -> SubmitOrder:
    return SubmitOrder(order.trader_id, order.strategy_id, order, UUID4(), _TS)


class _Cache:
    """Minimal order cache; ``apply_denied`` models the ExecEngine closing a denied order."""

    def __init__(self) -> None:
        self._orders: list[Any] = []

    def add_order(self, order: Any) -> None:
        if not self.order_exists(order.client_order_id):
            self._orders.append(order)

    def order_exists(self, client_order_id: Any) -> bool:
        return any(o.client_order_id == client_order_id for o in self._orders)

    def orders(self, **filters: Any) -> list[Any]:
        iid = filters.get("instrument_id")
        return [
            o for o in self._orders if not o.is_closed and (iid is None or o.instrument_id == iid)
        ]

    def orders_open(self, **_k: Any) -> list[Any]:
        return []

    def orders_inflight(self, **_k: Any) -> list[Any]:
        return []

    def positions_open(self, **_k: Any) -> list[Any]:
        return []

    def instrument(self, _iid: Any) -> Any:
        return SimpleNamespace(multiplier=Decimal(1))

    def price(self, _iid: Any, _price_type: Any) -> None:
        return None

    def apply_denied(self, event: OrderDenied) -> None:
        for order in self._orders:
            if order.client_order_id == event.client_order_id and not order.is_closed:
                order.apply(event)


class _Msgbus:
    def __init__(self, cache: _Cache) -> None:
        self._cache = cache
        self.sent: list[tuple[str, Any]] = []

    def send(self, *, endpoint: str, msg: Any) -> None:
        self.sent.append((endpoint, msg))
        if isinstance(msg, OrderDenied):
            self._cache.apply_denied(msg)


class _Portfolio:
    def net_position(self, _iid: Any) -> Decimal:
        return Decimal(0)


class _Clock:
    def timestamp_ns(self) -> int:
        return 0


class _RiskEngine:
    def __init__(self) -> None:
        self.commands: list[Any] = []

    def execute(self, command: Any) -> None:
        self.commands.append(command)


class _UnitGuard(TradingGuard):
    """TradingGuard with fake cache/msgbus/portfolio/clock and a recording risk engine."""

    def __init__(self, cache: _Cache, msgbus: _Msgbus, risk_engine: Any) -> None:
        self._c = cache
        self._m = msgbus
        super().__init__(
            TradingGuardConfig(),
            strategies=[],
            risk_engine=risk_engine,
            listener=GuardListener(),
        )

    @property
    def cache(self) -> Any:  # type: ignore[override]
        return self._c

    @property
    def msgbus(self) -> Any:  # type: ignore[override]
        return self._m

    @property
    def portfolio(self) -> Any:  # type: ignore[override]
        return _Portfolio()

    @property
    def clock(self) -> Any:  # type: ignore[override]
        return _Clock()


def _build(
    initial: Sequence[MarketOrder] = (),
) -> tuple[_UnitGuard, _RiskEngine, _Msgbus, _Cache]:
    cache = _Cache()
    for order in initial:
        cache.add_order(order)
    risk = _RiskEngine()
    msgbus = _Msgbus(cache)
    return _UnitGuard(cache, msgbus, risk), risk, msgbus, cache


def _raise_on(tag: str, exc: Exception) -> Callable[[Any], str | None]:
    """Patched ``_margin_reason``: raise for ``tag``, pass every other order."""

    def _check(order: Any) -> str | None:
        if str(order.client_order_id) == tag:
            raise exc
        return None

    return _check


#: The class name in the reason/log must come from the RAISED exception, not a literal.
#: SF1 on rev-gr-guard-1: hard-coding ``_MarginError`` in both the f-string and the log call
#: survived every round-1 test, so the unrelated classes below are what kill that mutant.
_GATE_ERROR_CASES = [
    pytest.param(_MarginError, "_MarginError", id="margin-error"),
    pytest.param(RuntimeError, "RuntimeError", id="runtime-error"),
    pytest.param(ZeroDivisionError, "ZeroDivisionError", id="zero-division"),
]


def _gate_error_reason(class_name: str) -> str:
    return f"{GATE_DENIAL_PREFIX}: entry check failed ({class_name}); order denied"


@pytest.mark.parametrize(("exc_cls", "class_name"), _GATE_ERROR_CASES)
def test_raising_entry_check_denies_the_order(
    monkeypatch: pytest.MonkeyPatch, exc_cls: type[Exception], class_name: str
) -> None:
    """G1+G5: the raising check denies the order with the exact G1 reason -- naming the
    RAISED class, whatever it is -- and never reaches the risk engine; ``_gate_denials``
    counts the denial."""
    order = _order("O-FAIL")
    guard, risk, msgbus, _cache = _build([order])
    monkeypatch.setattr(guard, "_margin_reason", _raise_on("O-FAIL", exc_cls("broken check")))

    guard._gate_execute(_submit(order))

    assert [msg.reason for _endpoint, msg in msgbus.sent] == [_gate_error_reason(class_name)]
    assert risk.commands == []
    assert guard.gate_denials == 1


@pytest.mark.parametrize(("exc_cls", "class_name"), _GATE_ERROR_CASES)
def test_raising_entry_check_logs_class_only(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    exc_cls: type[Exception],
    class_name: str,
) -> None:
    """G2: the log names the RAISED exception CLASS only; the message text never leaks."""
    order = _order("O-FAIL")
    guard, _risk, _msgbus, _cache = _build([order])
    monkeypatch.setattr(
        guard, "_margin_reason", _raise_on("O-FAIL", exc_cls("ZZLEAK9 secret detail"))
    )

    with caplog.at_level(logging.ERROR, logger="vibe_quant.paper.guard"):
        guard._gate_execute(_submit(order))

    assert f"entry check failed ({class_name}); order denied" in caplog.text
    assert "ZZLEAK9" not in caplog.text


def test_raising_entry_check_denies_every_order_in_the_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """G3+G5: a SubmitOrderList whose SECOND order raises denies ALL three with the G1
    reason (risk engine not called) and counts three denials."""
    first, second, third = _order("A-1"), _order("A-2"), _order("A-3")
    order_list = OrderList(OrderListId("OL-1"), [first, second, third])
    guard, risk, msgbus, _cache = _build()
    monkeypatch.setattr(guard, "_margin_reason", _raise_on("A-2", _MarginError()))

    guard._gate_execute(
        SubmitOrderList(first.trader_id, first.strategy_id, order_list, UUID4(), _TS)
    )

    assert [msg.reason for _endpoint, msg in msgbus.sent] == [_gate_error_reason("_MarginError")] * 3
    assert risk.commands == []
    assert guard.gate_denials == 3


def test_denied_order_is_closed_and_the_next_entry_is_admitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """G4 (bead outcome): the denied order closes and stops counting, so a later entry on
    the same instrument with a working check reaches the risk engine."""
    failing = _order("O-FAIL")
    guard, risk, _msgbus, cache = _build([failing])
    monkeypatch.setattr(guard, "_margin_reason", _raise_on("O-FAIL", _MarginError()))

    guard._gate_execute(_submit(failing))

    assert failing.is_closed
    assert cache.orders(instrument_id=failing.instrument_id) == []

    following = _order("O-OK")
    command = _submit(following)
    guard._gate_execute(command)

    assert risk.commands == [command]


def test_reduce_only_order_is_never_denied_by_a_raising_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """G6: ``_denial_reason`` returns None for a reduce-only order before the margin check,
    so the raise is never reached and the exit is admitted."""
    order = _order("O-EXIT", reduce_only=True)
    guard, risk, msgbus, _cache = _build([order])
    monkeypatch.setattr(guard, "_margin_reason", _raise_on("O-EXIT", _MarginError()))

    command = _submit(order)
    guard._gate_execute(command)

    assert msgbus.sent == []
    assert risk.commands == [command]
    assert guard.gate_denials == 0


def test_base_exception_is_not_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """G7: only ``Exception`` is caught; ``KeyboardInterrupt``/``SystemExit`` propagate."""
    order = _order("O-FAIL")
    guard, risk, msgbus, _cache = _build([order])

    def _interrupt(_order: Any) -> str | None:
        raise KeyboardInterrupt

    monkeypatch.setattr(guard, "_margin_reason", _interrupt)

    with pytest.raises(KeyboardInterrupt):
        guard._gate_execute(_submit(order))

    assert msgbus.sent == []
    assert risk.commands == []
    assert guard.gate_denials == 0
