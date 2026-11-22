"""Paper SL/TP repair of a bot-owned position (opspt P3, rows R50-R57).

Harness: ``tests/unit/test_paper_wire_gate.py``'s offline ``TradingNode`` with the REAL
``BinanceFuturesExecutionClient`` and a recording transport; the strategy is a compiled
DSL strategy, so its own level math (``_calculate_sl_price``) prices every leg. Unit rows
call :func:`repair` directly; R57 rows go through the node's restart adoption path (the
first agreeing venue check runs the repair).
"""

from __future__ import annotations

import contextlib
import json
from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

import pytest
from nautilus_trader.model.enums import OrderSide, OrderStatus
from nautilus_trader.model.identifiers import VenueOrderId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.test_kit.stubs.data import TestDataStubs
from nautilus_trader.test_kit.stubs.events import TestEventStubs

from tests.unit.test_audit_paper_node import seeded as seeded  # noqa: F401  # fixture
from tests.unit.test_paper_wire_gate import (
    _ACCOUNT,
    OPEN_ALGO,
    OPEN_ORDERS,
    _agree,
    _healthy_venue,
    _Live,
    _live,
    _recoverable_hold,
    _restarted,
    _settle,
    algo_row,
    first_check,
    owned_restart,
)
from tests.unit.test_paper_wire_gate import (  # autouse: requests reach the fake transport
    _real_send_to_fake_transport as _real_send_to_fake_transport,  # noqa: F401
)
from vibe_quant.paper.ownership import OwnershipLedger
from vibe_quant.paper.protection import (
    NOT_OWNED,
    PENDING,
    PROTECTED,
    RETRY,
    UNPROTECTED,
    UNRESOLVED,
    RepairMemo,
    repair,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from nautilus_trader.trading.strategy import Strategy

_SL_POST = ("POST", "/fapi/v1/algoOrder")
_TP_POST = ("POST", "/fapi/v1/order")
_TP_CANCEL = ("DELETE", "/fapi/v1/order")
_FIXED_5 = {"type": "fixed_pct", "percent": 5.0}
_ATR = {"type": "atr_fixed", "atr_multiplier": 2.0, "indicator": "atr"}


def _compiled(
    sl: dict[str, object] = _FIXED_5,
    tp: dict[str, object] = _FIXED_5,
    exit_above: str | None = None,
    claims: bool = False,
) -> Strategy:
    """A compiled DSL strategy on BTCUSDT 1m with the given SL/TP (an ATR(3) is always
    built); ``exit_above`` adds the exit ``close > <it>``; ``claims``: it claims BTC's
    reconciled orders, as paper's strategies do."""
    from vibe_quant.dsl.compiler import StrategyCompiler
    from vibe_quant.dsl.parser import validate_strategy_dict

    spec: dict[str, object] = {
        "name": "repair_probe",
        "timeframe": "1m",
        "indicators": {"sma_2": {"type": "SMA", "period": 2}, "atr": {"type": "ATR", "period": 3}},
        "entry_conditions": {"long": ["sma_2 > 0"]},
        "stop_loss": sl,
        "take_profit": tp,
    }
    if exit_above is not None:
        spec["exit_conditions"] = {"long": [f"close > {exit_above}"]}
    module = StrategyCompiler().compile_to_module(validate_strategy_dict(spec))
    raw: dict[str, object] = {"instrument_id": "BTCUSDT-PERP.BINANCE", "order_id_tag": "000"}
    if claims:
        raw["external_order_claims"] = ["BTCUSDT-PERP.BINANCE"]
    cfg = module.RepairProbeConfig.parse(json.dumps(raw))
    strategy: Strategy = module.RepairProbeStrategy(config=cfg)
    return strategy


async def _fill(live: _Live, side: OrderSide, qty: str, px: str) -> None:
    """A market order of the strategy filled at ``px`` (NT's own position update)."""
    entry = live.accepted(
        live.strategy.order_factory.market(live.iid, side, Quantity.from_str(qty))
    )
    live.kernel.exec_engine.process(
        TestEventStubs.order_filled(
            entry,
            instrument=live.kernel.cache.instrument(live.iid),
            account_id=_ACCOUNT,
            last_px=Price.from_str(px),
        )
    )
    await live.drain()


@contextlib.asynccontextmanager
async def _owned(
    strategy: Strategy,
    side: OrderSide = OrderSide.BUY,
    qty: str = "0.400",
    px: str = "10000.0",
    setup: Callable[[_Live], object] | None = None,
) -> AsyncIterator[_Live]:
    """The strategy starts on an adopted position (NT reconciled it, ``setup`` adds its
    resting legs); its own start-up re-arm is refused before the venue check, then the
    check agrees (writes pass from here)."""

    async def adopt(live: _Live) -> None:
        await _fill(live, side, qty, px)
        if setup is not None:
            setup(live)

    async with _live(strategy=strategy, verified=False, before_start=adopt) as live:
        await live.drain()
        live.verified = True
        yield live


def _leg(live: _Live, kind: str, qty: str, level: str, state: str = "accepted") -> Any:
    """A reduce-only SELL leg of the strategy: ``kind`` stop|tp, NT state per ``state``."""
    factory = live.strategy.order_factory
    if kind == "stop":
        order = factory.stop_market(
            live.iid,
            OrderSide.SELL,
            Quantity.from_str(qty),
            trigger_price=Price.from_str(level),
            reduce_only=True,
        )
    else:
        order = factory.limit(
            live.iid,
            OrderSide.SELL,
            Quantity.from_str(qty),
            Price.from_str(level),
            reduce_only=True,
        )
    if state == "accepted":
        return live.accepted(order)
    live.kernel.cache.add_order(order)
    order.apply(TestEventStubs.order_submitted(order, account_id=_ACCOUNT))
    if state == "pending_cancel":
        order.apply(
            TestEventStubs.order_accepted(
                order, account_id=_ACCOUNT, venue_order_id=VenueOrderId("77")
            )
        )
        order.apply(TestEventStubs.order_pending_cancel(order))
    live.kernel.cache.update_order(order)
    return order


def _quote(live: _Live, bid: str, ask: str) -> None:
    instrument = live.kernel.cache.instrument(live.iid)
    live.kernel.cache.add_quote_tick(
        TestDataStubs.quote_tick(instrument, bid_price=float(bid), ask_price=float(ask))
    )


def _legs(live: _Live) -> list[tuple[str, str, str, str]]:
    """``(type, qty, level, status)`` of every reduce-only SL/TP, sorted."""
    return sorted(
        (
            o.order_type.name,
            str(o.quantity),
            str(o.trigger_price if o.has_trigger_price else o.price),
            o.status.name,
        )
        for o in live.kernel.cache.orders()
        if o.is_reduce_only
        and o.order_type.name in ("STOP_MARKET", "LIMIT")
        and o.status != OrderStatus.REJECTED  # the start-up re-arm refused before the check
    )


async def _repair(live: _Live, memo: RepairMemo, owned: str) -> Any:
    report = repair(live.strategy, owned=Decimal(owned), memo=memo)
    await live.drain()
    return report


# ------------------------------------------------------------------ R50-R56 (unit)


@pytest.mark.parametrize(
    ("side", "qty", "entry", "percent", "legs"),
    [
        pytest.param(
            OrderSide.BUY,
            "0.400",
            "10000.0",
            5.0,
            [
                ("LIMIT", "0.400", "10500.0", "ACCEPTED"),
                ("STOP_MARKET", "0.400", "9500.0", "ACCEPTED"),
            ],
            id="long_0.400",
        ),
        pytest.param(
            OrderSide.SELL,
            "1.000",
            "20000.0",
            10.0,
            [
                ("LIMIT", "1.000", "18000.0", "ACCEPTED"),
                ("STOP_MARKET", "1.000", "22000.0", "ACCEPTED"),
            ],
            id="short_1.000",
        ),
    ],
)
async def test_unprotected_position_gets_both_legs_once(
    side: OrderSide, qty: str, entry: str, percent: float, legs: list[tuple[str, ...]]
) -> None:
    """R50: an adopted position with no leg gets SL + TP once, sized to it, at the
    strategy's own levels; the report is pending until Binance accepted both, then
    protected, and a further repair sends nothing."""
    pct = {"type": "fixed_pct", "percent": percent}
    async with _owned(_compiled(pct, pct), side, qty, entry) as live:
        owned = qty if side == OrderSide.BUY else f"-{qty}"
        _healthy_venue(live)
        memo = RepairMemo()
        first = await _repair(live, memo, owned)
        second = await _repair(live, memo, owned)
        third = await _repair(live, memo, owned)
        writes = live.transport.writes()
        held = _legs(live)

    assert (first.state, first.sent) == (PENDING, ("stop-loss", "take-profit"))
    assert (second.state, third.state) == (PROTECTED, PROTECTED)
    assert sorted(writes) == sorted([_SL_POST, _TP_POST])
    assert held == legs


@pytest.mark.parametrize(
    ("resting", "sent", "write", "legs"),
    [
        pytest.param(
            "tp",
            ("stop-loss",),
            _SL_POST,
            [
                ("LIMIT", "0.400", "10500.0", "ACCEPTED"),
                ("STOP_MARKET", "0.400", "9500.0", "ACCEPTED"),
            ],
            id="tp_rests_sl_missing",
        ),
        pytest.param(
            "stop",
            ("take-profit",),
            _TP_POST,
            [
                ("LIMIT", "0.400", "10500.0", "ACCEPTED"),
                ("STOP_MARKET", "0.400", "9500.0", "ACCEPTED"),
            ],
            id="sl_rests_tp_missing",
        ),
    ],
)
async def test_missing_leg_is_rearmed_alone(
    resting: str, sent: tuple[str, ...], write: tuple[str, str], legs: list[tuple[str, ...]]
) -> None:
    """R51: one leg rests, the other is missing: only the missing leg is submitted; the
    resting one is untouched (no cancel, no resize)."""
    async with _owned(_compiled(), OrderSide.BUY, "0.400", "10000.0") as live:
        kept = _leg(live, resting, "0.400", "10500.0" if resting == "tp" else "9500.0")
        _healthy_venue(live)
        memo = RepairMemo()
        report = await _repair(live, memo, "0.400")
        after = await _repair(live, memo, "0.400")
        writes = live.transport.writes()
        held = _legs(live)
        kept_status = kept.status

    assert (report.state, report.sent) == (PENDING, sent)
    assert after.state == PROTECTED
    assert writes == [write]
    assert held == legs
    assert kept_status == OrderStatus.ACCEPTED


@pytest.mark.parametrize("state", ["pending_cancel", "submitted"])
async def test_unanswered_leg_sends_nothing(state: str) -> None:
    """R52: a leg another path is cancelling (PENDING_CANCEL) or that Binance has not
    answered (SUBMITTED): nothing is sent, the report is unresolved."""
    async with _owned(_compiled(), OrderSide.BUY, "0.400", "10000.0") as live:
        _leg(live, "stop", "0.400", "9500.0", state=state)
        report = await _repair(live, RepairMemo(), "0.400")
        writes = live.transport.writes()

    assert report.state == UNRESOLVED
    assert writes == []


def _would_trigger(live: _Live) -> None:
    """Binance refusing every stop (algo order) POST with -2021 (the market already
    crossed its trigger); every other request goes to the transport below."""
    fake = live.transport.request

    async def request(method: Any, url: str, **kwargs: Any) -> Any:
        path = urlparse(url).path
        if str(method).endswith("POST") and path == _SL_POST[1]:
            live.transport.calls.append(_SL_POST)
            body = json.dumps({"code": -2021, "msg": "Order would immediately trigger."})
            return SimpleNamespace(status=400, body=body.encode(), headers={})
        return await fake(method, url, **kwargs)

    live.transport.request = request  # type: ignore[method-assign,assignment]


@pytest.mark.parametrize("price", ["bar_close_below", "quote_below", "no_price"])
async def test_stop_goes_to_binance_without_a_local_price_check(price: str) -> None:
    """R53 (rev B1): paper subscribes bars only, so no local price is fresh: a last bar
    close (9450) or a cached quote (bid 9400) below a long's 9500 stop does not keep the
    stop back; Binance decides (-2021, next test)."""
    async with _owned(_compiled(), OrderSide.BUY, "0.400", "10000.0") as live:
        _leg(live, "tp", "0.400", "10500.0")
        if price == "bar_close_below":
            live.strategy._last_close = 9450.0  # type: ignore[attr-defined]
        elif price == "quote_below":
            _quote(live, "9400.0", "9400.1")
        _healthy_venue(live)
        report = await _repair(live, RepairMemo(), "0.400")
        writes = live.transport.writes()

    assert (report.state, report.sent, writes) == (PENDING, ("stop-loss",), [_SL_POST])


@pytest.mark.parametrize(
    ("side", "stop", "moved"),
    [
        pytest.param(OrderSide.BUY, "9500.0", 9400.0, id="long"),
        pytest.param(OrderSide.SELL, "10500.0", 10600.0, id="short"),
    ],
)
async def test_stop_binance_says_would_trigger_is_named_and_resent_only_at_a_new_level(
    monkeypatch: pytest.MonkeyPatch, side: OrderSide, stop: str, moved: float
) -> None:
    """R53 (chief REV-B1): Binance refuses the stop with -2021 (it would trigger at
    once): the report names it with the hand action; repairs at the same level send
    nothing more; once the strategy's level moves the stop is sent once again."""
    async with _owned(_compiled(), side, "0.400", "10000.0") as live:
        long = side == OrderSide.BUY
        live.accepted(
            live.strategy.order_factory.limit(
                live.iid,
                OrderSide.SELL if long else OrderSide.BUY,
                Quantity.from_str("0.400"),
                Price.from_str("10500.0" if long else "9500.0"),
                reduce_only=True,
            )
        )
        _healthy_venue(live)
        _would_trigger(live)
        memo = RepairMemo()
        owned = "0.400" if long else "-0.400"
        sent = await _repair(live, memo, owned)
        named = await _repair(live, memo, owned)
        again = await _repair(live, memo, owned)
        same_level = list(live.transport.writes())
        monkeypatch.setattr(live.strategy, "_calculate_sl_price", lambda *_a: moved)
        resent = await _repair(live, memo, owned)
        writes = live.transport.writes()

    way = "long" if long else "short"
    assert (sent.state, sent.sent) == (PENDING, ("stop-loss",))
    assert (named.state, named.detail) == (
        UNPROTECTED,
        f"BTCUSDT: bot {way} 0.400 has no stop on Binance (stop {stop} would trigger at "
        "once); set one by hand or close it, then restart",
    )
    assert (again.state, again.sent) == (UNPROTECTED, ())
    assert same_level == [_SL_POST]
    assert (resent.state, resent.sent) == (PENDING, ("stop-loss",))
    assert writes == [_SL_POST, _SL_POST]


async def test_leg_problem_is_alerted_again_after_the_leg_was_placed_and_lost() -> None:
    """R4: a stop Binance refuses (-2021) is alerted once. The stop is later placed (it
    rests) and then lost: the same problem at the same level is a NEW one and alerts again
    (the memo forgets a leg's texts once the leg rests)."""
    async with _owned(_compiled(), OrderSide.BUY, "0.400", "10000.0") as live:
        _leg(live, "tp", "0.400", "10500.0")
        _healthy_venue(live)
        _would_trigger(live)
        memo = RepairMemo()
        await _repair(live, memo, "0.400")
        first = await _repair(live, memo, "0.400")
        again = await _repair(live, memo, "0.400")
        stop = _leg(live, "stop", "0.400", "9500.0")  # placed by hand / accepted by Binance
        protected = await _repair(live, memo, "0.400")
        stop.apply(TestEventStubs.order_canceled(stop))  # lost
        live.kernel.cache.update_order(stop)
        recurred = await _repair(live, memo, "0.400")

    assert (first.state, first.new_problem) == (UNPROTECTED, True)
    assert (again.state, again.new_problem) == (UNPROTECTED, False)
    assert protected.state == PROTECTED
    assert (recurred.state, recurred.new_problem) == (UNPROTECTED, True)
    assert recurred.detail == first.detail


def _bars(live: _Live, closes: list[str], start: int = 1) -> None:
    from nautilus_trader.model.data import Bar

    strategy: Any = live.strategy
    for i, close in enumerate(closes, start=start):
        px = Price.from_str(close)
        ts = i * 60_000_000_000
        hi, lo = (
            Price.from_str(f"{float(close) + 50:.1f}"),
            Price.from_str(f"{float(close) - 50:.1f}"),
        )
        strategy.handle_bar(
            Bar(strategy.primary_bar_type, px, hi, lo, px, Quantity.from_str("1"), ts, ts)
        )


async def test_atr_not_ready_waits_then_repairs_on_a_ready_bar() -> None:
    """R54: the ATR stop's level is not ready: nothing is sent, the report says retry; once
    the strategy's indicators are ready the next repair sends the stop at entry - 2 ATR
    (high-low 100 each bar: ATR 100 -> 10000 - 200)."""
    async with _owned(_compiled(sl=_ATR), OrderSide.BUY, "0.400", "10000.0") as live:
        _healthy_venue(live)
        memo = RepairMemo()
        before = await _repair(live, memo, "0.400")
        writes_before = list(live.transport.writes())
        _bars(live, ["10000.0", "10000.0", "10000.0", "10000.0"])
        ready = await _repair(live, memo, "0.400")
        held = _legs(live)

    assert (before.state, before.sent) == (PENDING, ("take-profit",)), "fixed TP goes out alone"
    assert writes_before == [_TP_POST]
    assert (ready.state, ready.sent) == (PENDING, ("stop-loss",))
    assert ("STOP_MARKET", "0.400", "9800.0", "ACCEPTED") in held


async def test_atr_not_ready_with_nothing_else_missing_is_retry() -> None:
    """R54: the only missing leg is the ATR stop: retry, 0 writes."""
    async with _owned(_compiled(sl=_ATR), OrderSide.BUY, "0.400", "10000.0") as live:
        _leg(live, "tp", "0.400", "10500.0")
        report = await _repair(live, RepairMemo(), "0.400")
        writes = live.transport.writes()

    assert (report.state, writes) == (RETRY, [])


def _cancel_acked_venue(live: _Live) -> list[Any]:
    """Binance confirming every regular-order cancel (the user stream's CANCELED)."""
    from nautilus_trader.model.identifiers import ClientOrderId

    fake = live.transport.request
    cancelled: list[Any] = []

    async def request(method: Any, url: str, **kwargs: Any) -> Any:
        path = urlparse(url).path
        if str(method).endswith("DELETE") and path == "/fapi/v1/order":
            live.transport.calls.append(("DELETE", path))
            cid = parse_qs(urlparse(url).query)["origClientOrderId"][0]
            order = live.kernel.cache.order(ClientOrderId(cid))
            cancelled.append(order)
            body = {
                "orderId": int(order.venue_order_id.value),
                "clientOrderId": cid,
                "symbol": "BTCUSDT",
                "side": "SELL",
                "positionSide": "BOTH",
                "timeInForce": "GTC",
                "reduceOnly": True,
                "price": str(order.price),
                "origQty": str(order.quantity),
                "executedQty": "0",
                "status": "CANCELED",
                "type": "LIMIT",
            }
            return SimpleNamespace(status=200, body=json.dumps(body).encode(), headers={})
        return await fake(method, url, **kwargs)

    live.transport.request = request  # type: ignore[method-assign,assignment]
    return cancelled


@pytest.mark.parametrize("ack", [True, False], ids=["cancel_acked", "cancel_refused"])
async def test_undersized_leg_is_replaced_only_after_its_cancel_is_acked(ack: bool) -> None:
    """R55: a TP of 0.200 for a 0.400 position (a partial fill): the repair cancels it and
    sends nothing more until Binance confirms the cancel; then the TP is replaced at its
    own level (10600.0, not the strategy's 10500.0) for 0.400. A refused cancel (the TP
    rests) is never replaced: the report names it for the user."""
    async with _owned(_compiled(), OrderSide.BUY, "0.400", "10000.0") as live:
        _leg(live, "stop", "0.400", "9500.0")
        small = _leg(live, "tp", "0.200", "10600.0")
        _healthy_venue(live)
        if ack:
            _cancel_acked_venue(live)
        memo = RepairMemo()
        first = await _repair(live, memo, "0.400")
        after_cancel = list(live.transport.writes())
        await live.drain()  # Binance's answer to the cancel (refused: NT restores the TP)
        waiting = await _repair(live, memo, "0.400")
        if ack:
            live.kernel.exec_engine.process(
                TestEventStubs.order_canceled(small, account_id=_ACCOUNT)
            )
            await live.drain()
        replaced = await _repair(live, memo, "0.400")
        writes = live.transport.writes()
        held = _legs(live)
        small_status = small.status

    assert (first.state, first.sent) == (PENDING, ("cancel take-profit",))
    assert after_cancel == [_TP_CANCEL]
    if ack:
        assert waiting.state == PENDING, "no replacement before the cancel is confirmed"
        assert (replaced.state, replaced.sent) == (PENDING, ("take-profit",))
        assert writes == [_TP_CANCEL, _TP_POST]
        assert ("LIMIT", "0.400", "10600.0", "ACCEPTED") in held
        assert ("LIMIT", "0.200", "10600.0", "CANCELED") in held
        return
    assert small_status == OrderStatus.ACCEPTED
    assert waiting.state == UNPROTECTED
    assert waiting.detail == (
        "BTCUSDT: bot long 0.400 has no take-profit on Binance (its take-profit covers less "
        "and Binance refused the resize); set one by hand, then restart"
    )
    assert writes == [_TP_CANCEL]


async def test_rejected_repair_leg_is_reported_and_not_resent() -> None:
    """R56: Binance rejects the repair's stop: the report is unprotected and names the
    venue's reason; the next repair sends nothing more for it."""
    async with _owned(_compiled(), OrderSide.BUY, "0.400", "10000.0") as live:
        _leg(live, "tp", "0.400", "10500.0")
        memo = RepairMemo()
        sent = await _repair(live, memo, "0.400")  # the fake transport rejects the POST
        report = await _repair(live, memo, "0.400")
        again = await _repair(live, memo, "0.400")
        writes = live.transport.writes()

    assert sent.state == PENDING
    assert report.state == UNPROTECTED
    assert report.detail is not None
    assert report.detail.startswith(
        "BTCUSDT: bot long 0.400 has no stop on Binance (Binance rejected it: "
    )
    assert "test venue reject" in report.detail
    assert report.detail.endswith("); set one by hand, then restart")
    assert again.sent == ()
    assert writes == [_SL_POST]


@pytest.mark.parametrize("owned", ["0", "0.200", "-0.400"])
async def test_position_not_in_the_bots_record_gets_nothing(owned: str) -> None:
    """Q8/P2: NT's long 0.400 is not the bot's own record (a hand position, or partly
    hand): nothing is sent."""
    async with _owned(_compiled(), OrderSide.BUY, "0.400", "10000.0") as live:
        report = await _repair(live, RepairMemo(), owned)
        writes = live.transport.writes()

    assert (report.state, writes) == (NOT_OWNED, [])


# -------------------------------------------------- R57: through the restart adoption path


def _owning(paper: Any, amount: str) -> None:
    paper._ownership = OwnershipLedger({"BTCUSDT": Decimal(amount)}, note=None)


async def _adopted(live: _Live) -> None:
    live.venue = _healthy_venue(live)  # type: ignore[attr-defined]
    await _fill(live, OrderSide.BUY, "0.400", "10000.0")


async def _adopted_with_tp(live: _Live) -> None:
    await _adopted(live)
    _leg(live, "tp", "0.400", "10500.0")


async def _adopted_with_tp_stop_would_trigger(live: _Live) -> None:
    await _adopted_with_tp(live)
    _would_trigger(live)


async def _adopted_with_queued_stop(live: _Live) -> None:
    await _adopted(live)
    _leg(live, "stop", "0.400", "9500.0", state="pending_cancel")


@pytest.mark.parametrize(
    ("adopted", "writes", "alert"),
    [
        pytest.param(_adopted, [_SL_POST, _TP_POST], None, id="R50_no_leg"),
        pytest.param(_adopted_with_tp, [_SL_POST], None, id="R51_tp_rests"),
        pytest.param(
            _adopted_with_queued_stop,
            [],
            "SL/TP check after the Binance check was skipped: a cancel refused before it is "
            "unresolved for {sid}; check its protection on Binance",
            id="R52_cancel_unresolved",
        ),
        pytest.param(
            _adopted_with_tp_stop_would_trigger,
            [_SL_POST],
            "BTCUSDT: bot long 0.400 has no stop on Binance (stop 9500.0 would trigger at "
            "once); set one by hand or close it, then restart",
            id="R53_would_trigger",
        ),
    ],
)
async def test_restart_adoption_repairs_through_the_node(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    adopted: Any,
    writes: list[tuple[str, str]],
    alert: str | None,
) -> None:
    """R57: a compiled strategy restarts on an owned long 0.400; its start-up re-arm is
    refused before the first agreeing check, which then runs the repair: both legs
    (R50), the missing stop only (R51), nothing while a cancel is unresolved (R52), and
    one critical alert for a stop Binance refuses as already crossed (R53, -2021)."""
    alerts: list[Any] = []
    async with _restarted(seeded, adopted, strategy=_compiled()) as (paper, live):
        _owning(paper, "0.400")
        monkeypatch.setattr(paper._journal, "alert", alerts.append)
        assert live.transport.writes() == [], "nothing reaches Binance before the check"
        assert _agree(paper) is True
        await _settle(live, paper)
        sent = live.transport.writes()
        sid = str(live.strategy.id)

    assert sorted(sent) == sorted(writes)
    if alert is None:
        assert alerts == []
        return
    [one] = alerts
    assert one.critical
    assert one.message == f"paper PAPER-007: {alert.format(sid=sid)}"


async def test_restart_adoption_rejected_leg_alerts_once(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R56 via the node: Binance rejects both repair legs: one critical alert names both,
    nothing is sent again."""
    alerts: list[Any] = []

    async def adopted(live: _Live) -> None:
        await _fill(live, OrderSide.BUY, "0.400", "10000.0")

    async with _restarted(seeded, adopted, strategy=_compiled()) as (paper, live):
        _owning(paper, "0.400")
        monkeypatch.setattr(paper._journal, "alert", alerts.append)
        assert _agree(paper) is True
        await _settle(live, paper)
        await _settle(live, paper)
        sent = live.transport.writes()

    assert sorted(sent) == sorted([_SL_POST, _TP_POST])
    [one] = alerts
    assert one.critical
    assert one.message.startswith(
        "paper PAPER-007: BTCUSDT: bot long 0.400 has no stop on Binance (Binance rejected it: "
    )
    assert "; BTCUSDT: bot long 0.400 has no take-profit on Binance (Binance rejected it: " in (
        one.message
    )


_RR = {"type": "risk_reward", "risk_reward_ratio": 2.0}


@pytest.mark.parametrize("tp_rests", [True, False], ids=["tp_rests", "no_leg"])
@pytest.mark.parametrize(
    ("closes", "legs"),
    [
        pytest.param(["10000.0", "10000.0", "10200.0"], [], id="ready_bar_exits"),
        pytest.param(
            ["10000.0", "10000.0", "10000.0"],
            [("STOP_MARKET", "0.400", "9800.0", "ACCEPTED")],
            id="ready_bar_holds",
        ),
    ],
)
async def test_atr_rearm_places_no_leg_on_the_bar_whose_exit_fires(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    closes: list[str],
    legs: list[tuple[str, ...]],
    tp_rests: bool,
) -> None:
    """R54 + ro-opspt-p1-6 residual: the adopted long's ATR levels are not ready at the
    first agreement (retry, nothing sent). On the first ready bar the strategy's exit
    fires (close > 10100): the repair waits for the close, the position is flat, no leg
    is ever placed (before: legs placed on that bar, then cancelled by its exit). Without
    the exit the stop goes out after that bar (entry - 2 ATR = 9800.0); with no TP
    resting the risk-reward TP joins it."""
    import vibe_quant.paper.node as node_module

    monkeypatch.setattr(node_module, "REARM_POLL_S", 0.01)

    async def adopted(live: _Live) -> None:
        live.venue = _healthy_venue(live)  # type: ignore[attr-defined]
        await _fill(live, OrderSide.BUY, "0.400", "10000.0")
        if tp_rests:
            _leg(live, "tp", "0.400", "10500.0")

    strategy = _compiled(sl=_ATR, tp=_RR, exit_above="10100")
    async with _restarted(seeded, adopted, strategy=strategy) as (paper, live):
        _owning(paper, "0.400")
        assert _agree(paper) is True
        await _spin(live)
        before = live.transport.writes()
        _bars(live, closes)
        await _spin(live)
        placed = [leg for leg in _legs(live) if leg[3] == "ACCEPTED" and leg[2] != "10500.0"]
        flat = live.kernel.cache.positions_open() == []

    assert before == []
    if legs and not tp_rests:
        legs = [("LIMIT", "0.400", "10400.0", "ACCEPTED"), *legs]
    assert placed == legs
    assert flat == (not legs)


async def _spin(live: _Live, rounds: int = 6) -> None:
    """Drain NT's queues a few times (the node's repair retries every REARM_POLL_S)."""
    for _ in range(rounds):
        await live.drain()


# ------------------------------------- per-leg retry, refusals across holds (round 2)


def _repair_alerts(alerts: list[Any]) -> list[str]:
    """The repair's alert texts (the recoverable-hold alerts have their own key)."""
    from vibe_quant.paper.journal import VENUE_HOLD_KEY

    return [a.message for a in alerts if a.key != VENUE_HOLD_KEY]


async def test_atr_stop_is_still_placed_when_the_take_profit_was_rejected(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ro B1 (chief RO-B1): restart on the bot's long 0.400, ATR stop not ready, fixed TP
    rejected by Binance. The one alert names BOTH missing legs (the TP to set by hand,
    the stop the bot still owes); the stop's retry goes on: after the ready bars it is
    sent (Binance rejects it here too: a second alert names it)."""
    import vibe_quant.paper.node as node_module

    monkeypatch.setattr(node_module, "REARM_POLL_S", 0.01)
    alerts: list[Any] = []

    async def adopted(live: _Live) -> None:
        await _fill(live, OrderSide.BUY, "0.400", "10000.0")  # the fake rejects every write

    async with _restarted(seeded, adopted, strategy=_compiled(sl=_ATR)) as (paper, live):
        _owning(paper, "0.400")
        monkeypatch.setattr(paper._journal, "alert", alerts.append)
        assert _agree(paper) is True
        await _spin(live)
        mid = (list(live.transport.writes()), _repair_alerts(alerts))
        _bars(live, ["10000.0", "10000.0", "10000.0", "10000.0"])
        await _spin(live, 20)
        end = (live.transport.writes(), _repair_alerts(alerts), paper._rearm_waiting)

    tp_gone = "BTCUSDT: bot long 0.400 has no take-profit on Binance (Binance rejected it: "
    assert mid[0] == [_TP_POST]
    [first] = mid[1]
    assert first.startswith(
        "paper PAPER-007: BTCUSDT: bot long 0.400 has no stop on Binance yet (its ATR level "
        "is not ready; the bot sends it at the first ready bar); " + tp_gone
    )
    assert "test venue reject" in first
    assert first.endswith("); set one by hand, then restart")
    assert end[0] == [_TP_POST, _SL_POST]
    assert end[1][0] == first
    [_, second] = end[1]
    assert second.startswith(
        "paper PAPER-007: BTCUSDT: bot long 0.400 has no stop on Binance (Binance rejected it: "
    )
    assert tp_gone in second
    assert end[2] == [], "every leg is final: the run ends"


async def test_refused_stop_is_not_resent_or_realerted_after_a_hold_clears(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RO-R3 + N2: Binance refuses the repair's stop (-2021) at the first agreement: one
    alert. An expired hold comes and clears: the repair runs again, but the stop at the
    same level is neither re-sent nor re-alerted (one per level for the session)."""
    alerts: list[Any] = []
    async with _restarted(seeded, _adopted_with_tp_stop_would_trigger, strategy=_compiled()) as (
        paper,
        live,
    ):
        _owning(paper, "0.400")
        now = [0.0]
        monkeypatch.setattr(paper, "_venue_now", lambda: now[0])
        monkeypatch.setattr(paper._journal, "alert", alerts.append)
        assert _agree(paper) is True
        await _settle(live, paper)
        first = (list(live.transport.writes()), _repair_alerts(alerts))
        _recoverable_hold(paper, "expired", now)
        now[0] = 100.0
        assert _agree(paper) is True
        await _settle(live, paper)
        after = (live.transport.writes(), _repair_alerts(alerts))

    assert first[0] == [_SL_POST]
    assert len(first[1]) == 1
    assert after == first


async def _protected_adopted(live: _Live) -> None:
    await _adopted(live)
    _leg(live, "tp", "0.400", "10500.0")
    _leg(live, "stop", "0.400", "9500.0")


@pytest.mark.parametrize("case", ["owned", "hand", "halt_owed"])
async def test_repair_runs_once_after_a_recoverable_hold_clears(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """S1 (chief #8047): the long 0.400 rests with both legs at the first agreement. An
    expired hold begins and Binance cancels the stop: nothing is sent while held. The next
    pass runs the repair: the bot's own position gets its stop back, once; a hand
    position gets nothing; a halt owed during the hold closes first and no leg follows."""
    import vibe_quant.paper.node as node_module
    from vibe_quant.paper.guard import HaltReason

    monkeypatch.setattr(node_module, "REARM_POLL_S", 0.01)
    async with _restarted(
        seeded, _protected_adopted, strategy=_compiled(), owned=case != "hand"
    ) as (paper, live):
        now = [0.0]
        monkeypatch.setattr(paper, "_venue_now", lambda: now[0])
        assert _agree(paper) is True
        await _spin(live)
        _recoverable_hold(paper, "expired", now)
        [stop] = [o for o in live.kernel.cache.orders() if o.order_type.name == "STOP_MARKET"]
        live.kernel.exec_engine.process(TestEventStubs.order_canceled(stop, account_id=_ACCOUNT))
        guard = live.guard
        assert guard is not None
        if case == "halt_owed":
            assert guard.halt(HaltReason.MANUAL, "operator halt")
        await _spin(live)
        during = list(live.transport.writes())
        now[0] = 100.0
        assert _agree(paper) is True
        await _spin(live)
        if case == "halt_owed":
            live.verified = True  # the guard's view of the pass: its owed close runs
            guard._on_tick(None)
        await _spin(live, 12)
        took = [o.order_type.name for o in live.venue]  # type: ignore[attr-defined]

    assert during == []
    assert took == {"owned": ["STOP_MARKET"], "hand": [], "halt_owed": ["MARKET"]}[case]


async def test_recoverable_hold_ends_a_waiting_atr_retry_until_the_next_pass(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S1: the ATR stop waits for its level (retry) when an expired hold begins; the
    indicators get ready during the hold: no stop is even submitted while held. The next
    pass starts the repair again and the stop goes out once (entry - 2 ATR = 9800.0)."""
    import vibe_quant.paper.node as node_module

    monkeypatch.setattr(node_module, "REARM_POLL_S", 0.01)
    async with _restarted(seeded, _adopted_with_tp, strategy=_compiled(sl=_ATR)) as (
        paper,
        live,
    ):
        _owning(paper, "0.400")
        now = [0.0]
        monkeypatch.setattr(paper, "_venue_now", lambda: now[0])
        assert _agree(paper) is True
        await _spin(live)
        _recoverable_hold(paper, "expired", now)
        _bars(live, ["10000.0", "10000.0", "10000.0", "10000.0"])
        await _spin(live, 12)
        during = [o for o in live.kernel.cache.orders() if o.order_type.name == "STOP_MARKET"]
        now[0] = 100.0
        assert _agree(paper) is True
        await _spin(live, 12)
        writes = live.transport.writes()
        held = _legs(live)

    assert during == []
    assert writes == [_SL_POST]
    assert ("STOP_MARKET", "0.400", "9800.0", "ACCEPTED") in held


# ----------------------------------------- RO-R2: own leftover legs on a flat coin

_OWN_ORPHAN = "O-20261121-000000-007-000-91"
_HAND_TP = "web_hand_tp_1"


def _deleted_ids(live: _Live) -> list[str]:
    """Record the client id of every cancel that reaches the transport."""
    fake = live.transport.request
    ids: list[str] = []

    async def request(method: Any, url: str, **kwargs: Any) -> Any:
        if str(method).endswith("DELETE"):
            query = parse_qs(urlparse(url).query)
            ids.extend(query.get("origClientOrderId") or query.get("clientAlgoId") or ["?"])
        return await fake(method, url, **kwargs)

    live.transport.request = request  # type: ignore[method-assign,assignment]
    return ids


def _resting_on_flat_coin(cid: str, kind: str) -> Callable[[_Live], Any]:
    """A restart on a flat BTC coin where NT reconciled a reduce-only SELL 0.400
    ``kind`` (stop|tp) ``cid`` that Binance's open-order reads also show."""

    async def reconcile(live: _Live) -> None:
        from nautilus_trader.model.identifiers import ClientOrderId

        factory = live.strategy.order_factory
        qty = Quantity.from_str("0.400")
        if kind == "stop":
            order = factory.stop_market(
                live.iid,
                OrderSide.SELL,
                qty,
                trigger_price=Price.from_str("9500.0"),
                reduce_only=True,
                client_order_id=ClientOrderId(cid),
            )
            live.transport.rows[OPEN_ALGO].append(algo_row(cid, "0.400"))
        else:
            order = factory.limit(
                live.iid,
                OrderSide.SELL,
                qty,
                Price.from_str("10500.0"),
                reduce_only=True,
                client_order_id=ClientOrderId(cid),
            )
            live.transport.rows[OPEN_ORDERS].append(
                {
                    "symbol": "BTCUSDT",
                    "clientOrderId": cid,
                    "side": "SELL",
                    "origQty": "0.400",
                    "type": "LIMIT",
                }
            )
        live.accepted(order)

    return reconcile


@pytest.mark.parametrize(
    ("cid", "kind", "own", "deletes"),
    [
        pytest.param(_OWN_ORPHAN, "stop", True, [("DELETE", "/fapi/v1/algoOrder")], id="own_sl"),
        pytest.param(_HAND_TP, "tp", False, [], id="hand_reduce_only_limit"),
    ],
)
async def test_leftover_reduce_only_leg_on_a_flat_coin(
    seeded: dict[str, Any],  # noqa: F811
    cid: str,
    kind: str,
    own: bool,
    deletes: list[tuple[str, str]],
) -> None:
    """RO-R2 (chief #8076): a restart where the bot's record, NT's view AND Binance are
    flat on BTC, and the bot's own stop from before still rests: the first agreement's
    repair cancels it by its own id (never symbol-wide). A hand reduce-only LIMIT on the
    flat coin is never touched (the check holds on it)."""
    prior = {
        "state": "running",
        "owned_net": {"BTCUSDT": "0"},
        "own_orders": {cid: "0"} if own else {},
    }
    async with owned_restart(
        seeded, prior, _resting_on_flat_coin(cid, kind), strategy=_compiled(claims=True)
    ) as (paper, live):
        ids = _deleted_ids(live)
        agreed = await first_check(paper, live, "0")
        await _settle(live, paper)
        writes = live.transport.writes()

    assert agreed is own
    assert writes == deletes
    assert ids == ([cid] if own else [])
