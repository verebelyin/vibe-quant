"""The paper bot's own orders and the net their fills add up to (opspt P2, R44-R47).

Through the real restore path, the real Binance exec client under the write gate and the
real node bridge (harness: tests/unit/test_paper_wire_gate.py); no network.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast

import pytest
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import ClientOrderId, VenueOrderId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.test_kit.stubs.events import TestEventStubs

from tests.unit.test_audit_paper_node import seeded as seeded  # noqa: F401  # fixture
from tests.unit.test_paper_wire_gate import (
    _ACCOUNT,
    _OWN_SL,
    OPEN_ALGO,
    OPEN_ORDERS,
    _Live,
    algo_row,
    btc_reading,
    first_check,
    owned_restart,
    resting_sl,
    venue_long,
    venue_reading,
)
from tests.unit.test_paper_wire_gate import (  # autouse: requests reach the fake transport
    _real_send_to_fake_transport as _real_send_to_fake_transport,  # noqa: F401
)
from vibe_quant.paper.ownership import OWNERSHIP_UNKNOWN, OwnershipLedger

_RUNNING = {"state": "running"}


def test_ledger_nets_exactly() -> None:
    """R44: three own BUY fills of 0.003 then an own SELL 0.009 net 0.009 then exactly 0;
    a repeated sync of the same total counts nothing; a foreign id never moves it."""
    ledger = OwnershipLedger(note=None)
    for n in range(3):
        ledger.admit(f"buy-{n}")
    ledger.admit("sell")
    moved = [ledger.sync(f"buy-{n}", "BTCUSDT", 1, Decimal("0.003")) for n in range(3)]
    after_buys = ledger.owned("BTCUSDT")
    again = ledger.sync("buy-0", "BTCUSDT", 1, Decimal("0.003"))
    foreign = ledger.sync("hand", "BTCUSDT", 1, Decimal("5"))
    ledger.sync("sell", "BTCUSDT", -1, Decimal("0.004"))
    ledger.sync("sell", "BTCUSDT", -1, Decimal("0.009"))  # the order's running total

    assert moved == [True, True, True]
    assert (again, foreign) == (False, False)
    assert str(after_buys) == "0.009"
    assert ledger.owned("BTCUSDT") == 0
    assert ledger.status(["BTCUSDT", "ETHUSDT"], ["sell", "hand"]) == {
        "owned_net": {"BTCUSDT": "0.000", "ETHUSDT": "0"},
        "own_orders": {"sell": "0.009"},
    }


def test_unreadable_record_logs_the_class_only(caplog: pytest.LogCaptureFixture) -> None:
    """A non-decimal saved amount or order id is unknown ownership; the log never echoes
    the value."""
    ledger = OwnershipLedger.restore(
        {"owned_net": {"BTCUSDT": "SECRET-MARKER"}, "own_orders": {"SECRET-ID ": "0"}}
    )

    assert (ledger.note, ledger.owned("BTCUSDT")) == (OWNERSHIP_UNKNOWN, 0)
    assert "ownership record missing or unreadable" in caplog.text
    assert "SECRET-" not in caplog.text


def test_restore_failure_is_unknown_and_logs_the_class_only(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """B1: anything raising while the record is read (an id NT's own check rejects) is
    unknown ownership, never a crash; the log names the class only."""
    from nautilus_trader.model import identifiers

    def reject(_value: str) -> None:
        raise ValueError("SECRET-MARKER")

    monkeypatch.setattr(identifiers, "ClientOrderId", reject)
    ledger = OwnershipLedger.restore({"owned_net": {}, "own_orders": {_OWN_SL: "0"}})

    assert (ledger.note, ledger.own_ids()) == (OWNERSHIP_UNKNOWN, [])
    assert "ownership record restore failed (ValueError)" in caplog.text
    assert "SECRET-MARKER" not in caplog.text


_JUNK = [
    pytest.param({}, id="absent"),
    pytest.param({"owned_net": None, "own_orders": {}}, id="null"),
    pytest.param({"owned_net": "x", "own_orders": {}}, id="not_a_dict"),
    pytest.param({"owned_net": {"BTCUSDT": "abc"}, "own_orders": {}}, id="not_decimal"),
    pytest.param({"owned_net": {"BTCUSDT": 1.0}, "own_orders": {}}, id="float_not_string"),
    pytest.param({"owned_net": {"BTCUSDT": "NaN"}, "own_orders": {}}, id="nan"),
    pytest.param({"owned_net": {"BTCUSDT": "0"}, "own_orders": {"": "0"}}, id="empty_order_id"),
]


_VALID_NET = {"BTCUSDT": "1.000"}
_VALID_ORDERS = {_OWN_SL: "0.500"}


@pytest.mark.parametrize(
    ("owned_net", "own_orders"),
    [
        pytest.param(_VALID_NET, {"": "0"}, id="empty_id"),
        pytest.param(_VALID_NET, {" ": "0"}, id="blank_id"),
        pytest.param(_VALID_NET, {" O-1": "0"}, id="padded_id"),
        pytest.param(_VALID_NET, {"\u00fc-hand": "0"}, id="non_ascii_id"),
        pytest.param(_VALID_NET, {"O-\x01": "0"}, id="control_char_id"),
        pytest.param(_VALID_NET, {_OWN_SL: "-0"}, id="negative_zero_fill"),
        pytest.param(_VALID_NET, {_OWN_SL: "1e3"}, id="exponent_fill"),
        pytest.param({"BTCUSDT": "1E+999999999"}, _VALID_ORDERS, id="huge_exponent"),
        pytest.param({"BTCUSDT": "-0"}, _VALID_ORDERS, id="negative_zero"),
        pytest.param({"BTCUSDT": "-0.000"}, _VALID_ORDERS, id="negative_zero_scaled"),
        pytest.param({"BTCUSDT": "inf"}, _VALID_ORDERS, id="infinity"),
        pytest.param({"BTCUSDT": "1.00000000000000000000000000001"}, _VALID_ORDERS, id="too_long"),
        pytest.param({"BTCUSDT": "\u0661"}, _VALID_ORDERS, id="non_ascii_digit"),
        pytest.param({"": "1.000"}, _VALID_ORDERS, id="empty_symbol"),
        pytest.param({"BTCUSDT": "junk"}, _VALID_ORDERS, id="junk_net_valid_orders"),
        pytest.param(_VALID_NET, {_OWN_SL: "junk"}, id="valid_net_junk_orders"),
    ],
)
def test_any_junk_makes_the_whole_record_unknown(owned_net: object, own_orders: object) -> None:
    """B1/S1/N1 (R45): one junk key or quantity anywhere in the saved record (an invalid
    order id, an exponent, a negative zero, a value Decimal would round) seeds the WHOLE
    record unknown: no own order, 0 for every coin. Never raises."""
    ledger = OwnershipLedger.restore({"owned_net": owned_net, "own_orders": own_orders})

    assert (ledger.note, ledger.own_ids(), ledger.owned("BTCUSDT")) == (OWNERSHIP_UNKNOWN, [], 0)


def test_valid_record_restores_exactly() -> None:
    """The control: a valid record restores every id and amount digit for digit, and
    saves back the same text."""
    raw = {"owned_net": {"BTCUSDT": "-0.400", "ETHUSDT": "0"}, "own_orders": {_OWN_SL: "0.400"}}
    ledger = OwnershipLedger.restore(raw)

    assert (ledger.note, ledger.own_ids()) == (None, [_OWN_SL])
    assert ledger.status(["BTCUSDT", "ETHUSDT"], [_OWN_SL]) == raw


async def test_junk_net_beside_own_orders_cannot_rebuild_a_record(
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """S1: junk ``owned_net`` beside a valid ``own_orders``; the bot's own stop filled while
    it was down and Binance is short 1.000. The stop is not counted as the bot's (the
    whole record is unknown), so the first check holds."""
    from nautilus_trader.core.uuid import UUID4
    from nautilus_trader.execution.reports import PositionStatusReport
    from nautilus_trader.model.enums import PositionSide

    async def reconcile(live: _Live) -> None:
        await _sl_filled_while_down(live)
        report = PositionStatusReport(
            account_id=_ACCOUNT,
            instrument_id=live.iid,
            position_side=PositionSide.SHORT,
            quantity=Quantity.from_str("1.000"),
            report_id=UUID4(),
            ts_last=1,
            ts_init=1,
            avg_px_open=Decimal("10000"),
        )
        assert live.kernel.exec_engine._reconcile_position_report_netting(report)

    prior = {**_RUNNING, "owned_net": {"BTCUSDT": "junk"}, "own_orders": {_OWN_SL: "0"}}
    async with owned_restart(seeded, prior, reconcile) as (paper, live):
        agreed = await first_check(paper, live, "-1.000")
        owned = paper._ownership.owned("BTCUSDT")
        held = paper._venue_hold()

    assert (agreed, owned) == (False, 0)
    assert held is not None and held.startswith("venue position BTCUSDT -1.000 != session 0")


@pytest.mark.parametrize("record", _JUNK)
@pytest.mark.parametrize(("venue", "passes"), [("0", True), ("1.000", False)])
async def test_unknown_record_seeds_zero(
    seeded: dict[str, Any],  # noqa: F811
    record: dict[str, object],
    venue: str,
    passes: bool,
) -> None:
    """R45 (user Q8=A): no usable ownership record seeds 0 with ``ownership_unknown``:
    Binance flat passes; Binance long 1.000 holds at the first check, and the hold says
    plainly the bot has no record of it and what to do (no one-time trust)."""
    async with owned_restart(seeded, {**_RUNNING, **record}, venue_long(venue)) as (paper, live):
        note = paper._ownership.note
        seeded_net = paper._ownership.owned("BTCUSDT")
        agreed = await first_check(paper, live, venue)
        held = paper._venue_hold()
        writes = live.transport.writes()
        saved = paper._capture_checkpoint().node_status  # never raises on ownership data

    assert (note, seeded_net) == (OWNERSHIP_UNKNOWN, 0)
    assert ("own_orders" in saved) is passes, "keys omitted while the record is unknown"
    assert agreed is passes
    assert writes == []
    if passes:
        assert held is None
    else:
        assert held == (
            "venue position BTCUSDT 1.000 != session 0 (the bot has no saved record of "
            "which positions it opened: BTCUSDT 1.000 long on Binance, bot record none; "
            "close it by hand on Binance, then restart)"
        )


@pytest.mark.parametrize(
    "own_orders",
    [
        pytest.param(None, id="absent"),
        pytest.param(["x"], id="list"),
        pytest.param({_OWN_SL: "-1"}, id="negative"),
    ],
)
async def test_unknown_own_orders_hold_any_resting_order(
    seeded: dict[str, Any],  # noqa: F811
    own_orders: object,
) -> None:
    """R45: absent or junk ``own_orders`` seeds no own order (``ownership_unknown``): the
    bot's restored stop on its coin then counts as foreign and the first check holds."""
    record: dict[str, object] = {**_RUNNING, "owned_net": {"BTCUSDT": "1.000"}}
    if own_orders is not None:
        record["own_orders"] = own_orders
    async with owned_restart(seeded, record, resting_sl(_OWN_SL)) as (paper, live):
        note = paper._ownership.note
        agreed = await first_check(paper, live, "1.000")
        held = paper._venue_hold()
        writes = live.transport.writes()

    assert note == OWNERSHIP_UNKNOWN
    assert agreed is False
    assert held is not None and held.startswith(f"foreign order {_OWN_SL} SELL 1.000 on BTCUSDT")
    assert writes == []


async def _sl_filled_while_down(live: _Live) -> None:
    """The bot's own stop (in the saved ``own_orders``) filled while the bot was down: NT
    reconciles it FILLED before any actor listens; Binance is flat."""
    sl = live.strategy.order_factory.stop_market(
        live.iid,
        OrderSide.SELL,
        Quantity.from_str("1.000"),
        trigger_price=Price.from_str("9000.0"),
        reduce_only=True,
        client_order_id=ClientOrderId(_OWN_SL),
    )
    live.accepted(sl)
    sl.apply(
        TestEventStubs.order_filled(
            sl,
            instrument=live.kernel.cache.instrument(live.iid),
            account_id=_ACCOUNT,
            venue_order_id=VenueOrderId("1"),
            last_px=Price.from_str("9000.0"),
        )
    )
    live.kernel.cache.update_order(sl)


async def _own_stop_resting_flat(live: _Live) -> None:
    """Binance flat, the bot's own stop still resting (in the saved ``own_orders``)."""
    sl = live.strategy.order_factory.stop_market(
        live.iid,
        OrderSide.SELL,
        Quantity.from_str("1.000"),
        trigger_price=Price.from_str("9000.0"),
        reduce_only=True,
        client_order_id=ClientOrderId(_OWN_SL),
    )
    live.accepted(sl)


@pytest.mark.parametrize(
    ("record", "before_start", "venue", "passes", "hold", "cleared"),
    [
        pytest.param(
            {"BTCUSDT": "1.000"},
            _sl_filled_while_down,
            "0",
            True,
            None,
            False,
            id="own_stop_filled_while_down",
        ),
        pytest.param(
            {"BTCUSDT": "1.000"},
            venue_long("0"),
            "0",
            True,
            None,
            True,
            id="closed_by_hand_while_down",
        ),
        pytest.param(
            {"BTCUSDT": "1.000"},
            venue_long("1.000"),
            "0",
            False,
            "venue position BTCUSDT 0 != session 1.000",
            False,
            id="binance_flat_but_bot_live_view_long",
        ),
        pytest.param(
            {"BTCUSDT": "1.000"},
            _own_stop_resting_flat,
            "0",
            False,
            "venue position BTCUSDT 0 != session 1.000",
            False,
            id="flat_but_own_order_resting",
        ),
        pytest.param(
            {"BTCUSDT": "0"},
            venue_long("1.000"),
            "0",
            False,
            "venue position BTCUSDT 0 != bot live view 1.000",
            False,
            id="record_agrees_bot_live_view_differs",
        ),
        pytest.param(
            {"BTCUSDT": "1.000"},
            venue_long("2.000"),
            "2.000",
            False,
            "venue position BTCUSDT 2.000 != session 1.000",
            False,
            id="binance_more_than_record",
        ),
    ],
)
async def test_restart_reading_against_the_record(
    seeded: dict[str, Any],  # noqa: F811
    record: dict[str, str],
    before_start: Any,
    venue: str,
    passes: bool,
    hold: str | None,
    cleared: bool,
) -> None:
    """Chief rulings (r8001, r8003): the bot's own stop that filled while it was down is
    counted (record 0, no hold). Binance flat while the record says long 1.000 (closed by
    hand) clears the record at the restart's first check, with one info line, but only
    when the bot's live view is flat too. Binance must equal both the record and the live
    view; Binance 2.000 vs the record 1.000 holds. Nothing reaches Binance."""
    from vibe_quant.api.routers.paper_trading import _shown_state

    prior = {**_RUNNING, "owned_net": record, "own_orders": {_OWN_SL: "0"}}
    async with owned_restart(seeded, prior, before_start) as (paper, live):
        alerts: list[tuple[str, str]] = []
        real = paper._journal.alert
        paper._journal.alert = lambda alert: (  # type: ignore[method-assign]
            alerts.append((alert.kind, alert.message)),
            real(alert),
        )[1]
        agreed = await first_check(paper, live, venue)
        held = paper._venue_hold()
        writes = live.transport.writes()
        owned = paper._ownership.owned("BTCUSDT")
        _state, status = _shown_state(paper._capture_checkpoint().node_status)

    assert agreed is passes
    assert writes == []
    if hold is None:
        assert held is None
        assert owned == 0
    else:
        assert held == hold
    line = "bot record BTCUSDT 1.000 long, Binance flat: record cleared"
    info = [a for a in alerts if "record cleared" in a[1]]
    # R6/N5: an info alert (not an error), and the same line in /paper/status.
    assert info == ([("info", f"paper PAPER-007: {line}")] if cleared else [])
    assert (status is not None and line in status) is cleared


def _own_stop_on_binance_only(cid: str, amount: str | None = "1.000") -> Any:
    """Binance flat; ``cid`` rests on Binance as an algo stop, but NT's start-up algo read
    failed (NT 1.231 drops every algo report on a BinanceError): NT's cache has no order."""

    async def reconcile(live: _Live) -> None:
        live.transport.rows[OPEN_ALGO].append(algo_row(cid, amount))

    return reconcile


@pytest.mark.parametrize(
    ("record", "rows", "hold"),
    [
        pytest.param(
            {"BTCUSDT": "1.000"},
            {OPEN_ALGO: [algo_row(_OWN_SL)]},
            "venue position BTCUSDT 0 != session 1.000 (BTCUSDT: Binance flat, bot record "
            f"1.000 long; the bot's stop-loss order {_OWN_SL} still rests on Binance: cancel "
            "it by hand on Binance, then restart)",
            id="own_stop_resting_unseen_by_nt",
        ),
        pytest.param(
            {"BTCUSDT": "0"},
            {OPEN_ALGO: [algo_row("web_hand_stop", None)]},
            "foreign order web_hand_stop SELL whole position on BTCUSDT: the bot did not place it",
            id="hand_close_position_stop",
        ),
        pytest.param(
            {"BTCUSDT": "0"},
            {
                OPEN_ORDERS: [
                    {
                        "symbol": "BTCUSDT",
                        "clientOrderId": "web_hand_limit",
                        "side": "BUY",
                        "type": "LIMIT",
                        "origQty": "0.010",
                    }
                ]
            },
            "foreign order web_hand_limit BUY 0.010 on BTCUSDT: the bot did not place it",
            id="hand_limit_unseen_by_nt",
        ),
    ],
)
async def test_binance_open_orders_are_read_by_the_check(
    seeded: dict[str, Any],  # noqa: F811
    record: dict[str, str],
    rows: dict[str, list[object]],
    hold: str,
) -> None:
    """B2/R1/R2 (ruling #8022): Binance flat and NT's cache holds no order (its algo read
    failed, or a user-stream gap lost a hand order), yet Binance's own open-order reads
    show one on BTC. The bot's own stop: no record clear, and the hold names the stop and
    the hand action. A hand closePosition stop or a hand limit: the foreign-order hold.
    Nothing reaches Binance, and a later bot cancel-all (symbol-wide on Binance: it would
    cancel the user's order) sends nothing."""
    prior = {**_RUNNING, "owned_net": record, "own_orders": {_OWN_SL: "0"}}

    async def reconcile(live: _Live) -> None:
        for path, answered in rows.items():
            live.transport.rows[path].extend(answered)

    async with owned_restart(seeded, prior, reconcile) as (paper, live):
        agreed = await first_check(paper, live, "0")
        owned = paper._ownership.owned("BTCUSDT")
        held = paper._venue_hold()
        stop = live.strategy.order_factory.stop_market(
            live.iid,
            OrderSide.SELL,
            Quantity.from_str("1.000"),
            trigger_price=Price.from_str("9000.0"),
            reduce_only=True,
        )
        live.accepted(stop)  # NT would cancel-all symbol-wide: it owns every cached order
        live.strategy.cancel_all_orders(live.iid)
        await live.drain()
        writes = live.transport.writes()

    assert agreed is False
    assert owned == Decimal(record["BTCUSDT"]), "the record is never cleared"
    assert held is not None and held.startswith(hold), held
    assert writes == []


@pytest.mark.parametrize(
    ("path", "answer"),
    [
        pytest.param(OPEN_ALGO, 500, id="algo_read_500"),
        pytest.param(OPEN_ORDERS, 500, id="regular_read_500"),
        pytest.param(OPEN_ALGO, [{"symbol": "BTCUSDT", "side": "SELL"}], id="row_without_id"),
        pytest.param(OPEN_ORDERS, {"code": 0}, id="not_a_list"),
        pytest.param(OPEN_ALGO, ["junk"], id="row_without_symbol"),
    ],
)
async def test_failed_open_order_read_is_inconclusive(
    seeded: dict[str, Any],  # noqa: F811
    path: str,
    answer: object,
) -> None:
    """#8022 + R5: a failed or unreadable open-order read (an HTTP 500, a row with no id,
    a body that is not a list) proves nothing about hand orders: no pass, no record clear,
    no hold, no write. The restart clear stays available: once the read recovers, the
    same Binance-flat reading clears the record and passes."""
    prior = {**_RUNNING, "owned_net": {"BTCUSDT": "1.000"}, "own_orders": {}}
    async with owned_restart(seeded, prior, venue_long("0")) as (paper, live):
        if isinstance(answer, int):
            live.transport.errors[path] = answer
        else:
            live.transport.rows[path] = cast("list[object]", answer)
        agreed = await first_check(paper, live, "0")
        first = (agreed, paper._ownership.owned("BTCUSDT"), paper._venue_hold())
        failure = paper._venue_last_failure
        live.transport.errors.clear()
        live.transport.rows[path] = []
        recovered = await first_check(paper, live, "0")
        owned = paper._ownership.owned("BTCUSDT")
        writes = live.transport.writes()

    assert first == (False, Decimal("1.000"), None)
    assert failure is not None and "open orders unverified" in failure, failure
    assert (recovered, owned) == (True, 0)
    assert writes == []


async def test_unverified_symbol_keeps_its_restart_clear(
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """R5: BTC's positionRisk row is unreadable on the restart's first reading; its
    one-time record clear waits for BTC's first verified reading (Binance flat: cleared,
    pass), instead of latching the stale record."""
    prior = {**_RUNNING, "owned_net": {"BTCUSDT": "1.000"}, "own_orders": {}}
    async with owned_restart(seeded, prior, venue_long("0")) as (paper, live):
        junk = [{"symbol": "BTCUSDT", "positionSide": "BOTH", "positionAmt": "1e3"}]
        first = await first_check(paper, live, "0", junk)
        held_first = paper._venue_hold()
        agreed = await first_check(paper, live, "0")
        owned = paper._ownership.owned("BTCUSDT")

    assert (first, held_first) == (False, None)
    assert (agreed, owned) == (True, 0)


async def test_unknown_record_is_not_trusted_on_an_unreadable_row(
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """N2: no saved record (unknown); BTC's row is unreadable on the first reading. The
    record stays unknown, so the next reading's drift still says plainly the bot has no
    saved record."""
    async with owned_restart(seeded, {**_RUNNING}, venue_long("1.000")) as (paper, live):
        junk = [{"symbol": "BTCUSDT", "positionSide": "BOTH", "positionAmt": "x"}]
        await venue_reading(paper, "1.000", junk)
        note = paper._ownership.note
        await venue_reading(paper, "1.000")
        held = paper._venue_hold()

    assert note == OWNERSHIP_UNKNOWN
    assert held is not None and "(the bot has no saved record" in held, held


@pytest.mark.parametrize("event", ["updated", "triggered"])
async def test_foreign_update_or_trigger_latches(
    seeded: dict[str, Any],  # noqa: F811
    event: str,
) -> None:
    """N3: a running, checked session sees an OrderUpdated / OrderTriggered of an order the
    bot never placed on its coin (published on NT's bus as the exec engine does): the
    foreign-order hold latches and names it."""
    prior = {**_RUNNING, "owned_net": {"BTCUSDT": "0"}, "own_orders": {}}
    async with owned_restart(seeded, prior, venue_long("0")) as (paper, live):
        assert await first_check(paper, live, "0") is True
        hand = live.strategy.order_factory.stop_market(
            live.iid,
            OrderSide.BUY,
            Quantity.from_str("0.500"),
            trigger_price=Price.from_str("12000.0"),
            client_order_id=ClientOrderId("web_hand_stop_2"),
        )
        live.accepted(hand)
        stub = (
            TestEventStubs.order_updated(hand, quantity=Quantity.from_str("0.500"))
            if event == "updated"
            else TestEventStubs.order_triggered(hand)
        )
        live.kernel.msgbus.publish(topic=f"events.order.{hand.strategy_id}", msg=stub)
        held = paper._venue_hold()

    assert held is not None
    assert held.startswith("foreign order web_hand_stop_2 BUY 0.500 on BTCUSDT"), held


async def test_ownership_save_failure_saves_no_record(
    seeded: dict[str, Any],  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    """B1: building the ownership keys raising never breaks the checkpoint save: the keys
    are left out (the next start is unknown: fail closed); the log names the class only."""
    prior = {**_RUNNING, "owned_net": {"BTCUSDT": "0"}, "own_orders": {}}
    async with owned_restart(seeded, prior, venue_long("0")) as (paper, live):
        assert await first_check(paper, live, "0") is True

        def boom() -> list[str]:
            raise RuntimeError("SECRET-MARKER")

        paper._ownership.own_ids = boom  # type: ignore[method-assign]
        saved = paper._capture_checkpoint().node_status

    assert "owned_net" not in saved and "own_orders" not in saved
    assert "ownership record save failed (RuntimeError)" in caplog.text
    assert "SECRET-MARKER" not in caplog.text


async def test_runtime_flat_venue_still_holds(seeded: dict[str, Any]) -> None:  # noqa: F811
    """Chief ruling (r8001 c): the record clear is for a restart's FIRST check only; a later
    reading of a flat Binance while the bot owns 1.000 holds (P1 latch)."""
    prior = {**_RUNNING, "owned_net": {"BTCUSDT": "1.000"}, "own_orders": {}}
    async with owned_restart(seeded, prior, venue_long("1.000")) as (paper, live):
        assert await first_check(paper, live, "1.000") is True
        paper._venue_apply_positions(btc_reading("0"), None)
        held = paper._venue_hold()

    assert held == "venue position BTCUSDT 0 != session 1.000"


async def test_own_fill_moves_the_record_and_saves(
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """R46/R47: the gate admits a two-order list (NT's own risk engine then denies one):
    both ids are the bot's own. An own fill of 0.400 moves the record and saves one
    checkpoint holding ``owned_net`` and the still-open own orders."""
    from types import SimpleNamespace

    from nautilus_trader.model.orders.list import OrderList

    async with owned_restart(
        seeded, {**_RUNNING, "owned_net": {}, "own_orders": {}}, venue_long("0")
    ) as (paper, live):
        assert await first_check(paper, live, "0") is True
        guard = live.guard
        assert guard is not None
        sent: list[Any] = []
        guard._risk_engine = SimpleNamespace(execute=sent.append)  # NT's risk engine, unreached
        factory = live.strategy.order_factory
        a = factory.limit(
            live.iid,
            OrderSide.SELL,
            Quantity.from_str("1.000"),
            Price.from_str("11000.0"),
            reduce_only=True,
        )
        b = factory.limit(
            live.iid,
            OrderSide.SELL,
            Quantity.from_str("1.000"),
            Price.from_str("12000.0"),
            reduce_only=True,
        )
        live.strategy.submit_order_list(OrderList(factory.generate_order_list_id(), [a, b]))
        await live.drain()  # the admission save (R4) lands before the fill is counted
        saves: list[int] = []
        paper._save_checkpoint_now = lambda: saves.append(1)  # type: ignore[method-assign]
        live.kernel.exec_engine.process(TestEventStubs.order_submitted(a, account_id=_ACCOUNT))
        live.kernel.exec_engine.process(
            TestEventStubs.order_accepted(a, account_id=_ACCOUNT, venue_order_id=VenueOrderId("9"))
        )
        live.kernel.exec_engine.process(
            TestEventStubs.order_filled(
                a,
                instrument=live.kernel.cache.instrument(live.iid),
                account_id=_ACCOUNT,
                venue_order_id=VenueOrderId("9"),
                last_qty=Quantity.from_str("0.400"),
                last_px=Price.from_str("11000.0"),
            )
        )
        await live.drain()
        own = {str(a.client_order_id), str(b.client_order_id)}
        status = paper._capture_checkpoint().node_status
        held = paper._venue_hold()

    assert len(sent) == 1, "admitted once, as a list"
    assert own <= set(paper._ownership.own_ids())
    assert held is None, "own fills are not foreign"
    assert saves == [1]
    assert status["owned_net"] == {"BTCUSDT": "-0.400"}
    assert status["own_orders"] == {str(a.client_order_id): "0.400", str(b.client_order_id): "0"}


async def test_failing_checks_hold_and_log_the_class_only(
    seeded: dict[str, Any],  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A raising admission report still forwards the order (its later events are then
    foreign: fail closed); a raising ownership check on an order event holds. Logs name
    the exception class only."""
    from types import SimpleNamespace

    prior = {**_RUNNING, "owned_net": {}, "own_orders": {}}
    async with owned_restart(seeded, prior, venue_long("0")) as (paper, live):
        assert await first_check(paper, live, "0") is True
        guard = live.guard
        assert guard is not None
        sent: list[Any] = []
        guard._risk_engine = SimpleNamespace(execute=sent.append)

        def boom(*_args: Any) -> None:
            raise RuntimeError("SECRET-MARKER")

        listener = guard._listener
        listener.on_order_admitted = boom  # type: ignore[method-assign]
        order = live.strategy.order_factory.limit(
            live.iid,
            OrderSide.SELL,
            Quantity.from_str("1.000"),
            Price.from_str("11000.0"),
            reduce_only=True,
        )
        live.strategy.submit_order(order)
        admitted = paper._ownership.is_own(str(order.client_order_id))
        paper._ownership.is_own = boom  # type: ignore[method-assign]
        paper._on_order_event(TestEventStubs.order_submitted(order, account_id=_ACCOUNT))
        held = paper._venue_hold()

    assert (len(sent), admitted) == (1, False)
    assert held == "ownership check failed (RuntimeError)"
    assert "order admission report failed (RuntimeError)" in caplog.text
    assert "SECRET-MARKER" not in caplog.text


async def test_record_clear_event_failure_still_clears(
    seeded: dict[str, Any],  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The record clear's event write raising does not stop the clear (the check passes);
    the log names the class only."""
    prior = {**_RUNNING, "owned_net": {"BTCUSDT": "1.000"}, "own_orders": {}}
    async with owned_restart(seeded, prior, venue_long("0")) as (paper, live):

        def boom(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("SECRET-MARKER")

        paper._journal.record = boom  # type: ignore[method-assign]
        agreed = await first_check(paper, live, "0")
        owned = paper._ownership.owned("BTCUSDT")

    assert (agreed, owned) == (True, 0)
    assert "record clear event failed (RuntimeError)" in caplog.text
    assert "SECRET-MARKER" not in caplog.text


def _latest_saved(paper: Any) -> dict[str, Any]:
    """The newest saved checkpoint's ``node_status``, after every queued save landed."""
    from vibe_quant.paper.persistence import StatePersistence

    paper._db_worker.submit(lambda: None).result(5)
    store = StatePersistence(db_path=paper._persistence._db_path, trader_id="PAPER-007")
    try:
        saved = store.load_latest_checkpoint("PAPER-007")
    finally:
        store.close()
    assert saved is not None
    return dict(saved.node_status)


async def test_graceful_stop_close_fill_is_saved_and_the_next_start_is_quiet(
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """R3: a stop with BTC 0.010 long. The strategies' on_stop close is admitted (own) and
    fills while ``stop_async`` runs, after the guard actor stopped listening. The saved
    record then matches Binance (owned 0): the next start finds Binance flat, passes with
    no "record cleared" alert and no notice."""
    from vibe_quant.paper.persistence import StatePersistence

    prior = {**_RUNNING, "owned_net": {"BTCUSDT": "0.010"}, "own_orders": {}}
    async with owned_restart(seeded, prior, venue_long("0.010")) as (paper, live):
        assert await first_check(paper, live, "0.010") is True
        paper._persistence = StatePersistence(db_path=seeded["db"], trader_id="PAPER-007")
        guard = live.guard
        assert guard is not None
        guard._risk_engine = SimpleNamespace(execute=lambda command: None)
        close = live.strategy.order_factory.market(
            live.iid, OrderSide.SELL, Quantity.from_str("0.010"), reduce_only=True
        )

        class _StoppingNode:
            async def stop_async(_self) -> None:
                # The guard (an actor) is stopped first: it no longer hears order events.
                guard._listener.on_order_event = lambda event: None  # type: ignore[method-assign]
                live.strategy.submit_order(close)
                engine = live.kernel.exec_engine
                engine.process(TestEventStubs.order_submitted(close, account_id=_ACCOUNT))
                engine.process(
                    TestEventStubs.order_accepted(
                        close, account_id=_ACCOUNT, venue_order_id=VenueOrderId("77")
                    )
                )
                engine.process(
                    TestEventStubs.order_filled(
                        close,
                        instrument=live.kernel.cache.instrument(live.iid),
                        account_id=_ACCOUNT,
                        venue_order_id=VenueOrderId("77"),
                        last_qty=Quantity.from_str("0.010"),
                        last_px=Price.from_str("11000.0"),
                    )
                )
                await live.drain()

        paper._trading_node = _StoppingNode()  # type: ignore[assignment]
        await paper._shutdown()
        saved = _latest_saved_after_close(seeded)

    assert saved["owned_net"] == {"BTCUSDT": "0.000"}
    async with owned_restart(seeded, None, venue_long("0")) as (paper, live):
        alerts: list[str] = []
        real = paper._journal.alert
        paper._journal.alert = lambda alert: (  # type: ignore[method-assign]
            alerts.append(alert.message),
            real(alert),
        )[1]
        agreed = await first_check(paper, live, "0")
        notice = paper._status.notice

    assert agreed is True
    assert [a for a in alerts if "record cleared" in a] == []
    assert notice is None


def _latest_saved_after_close(seeded: dict[str, Any]) -> dict[str, Any]:
    from vibe_quant.paper.persistence import StatePersistence

    store = StatePersistence(db_path=seeded["db"], trader_id="PAPER-007")
    try:
        saved = store.load_latest_checkpoint("PAPER-007")
    finally:
        store.close()
    assert saved is not None
    return dict(saved.node_status)


async def test_partial_stop_close_checkpoint_still_lists_position_orders_and_balance(
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """B1: a stop with BTC 0.010 long; the on_stop close fills only 0.004 (Binance holds
    0.006 unprotected). The newest checkpoint, saved after stop_async, is captured from
    the real cache: it lists the 0.006 long, its orders and the balance, never an empty
    snapshot (the stopped session's Positions page reads it)."""
    from vibe_quant.paper.persistence import StatePersistence

    prior = {**_RUNNING, "owned_net": {"BTCUSDT": "0.010"}, "own_orders": {}}
    async with owned_restart(seeded, prior, venue_long("0.010")) as (paper, live):
        assert await first_check(paper, live, "0.010") is True
        paper._persistence = StatePersistence(db_path=seeded["db"], trader_id="PAPER-007")
        guard = live.guard
        assert guard is not None
        guard._risk_engine = SimpleNamespace(execute=lambda command: None)
        close = live.strategy.order_factory.market(
            live.iid, OrderSide.SELL, Quantity.from_str("0.010"), reduce_only=True
        )

        class _StoppingNode:
            # The real kernel's cache and portfolio: what NT keeps while it stops.
            cache = live.kernel.cache
            portfolio = live.kernel.portfolio

            async def stop_async(_self) -> None:
                guard._listener.on_order_event = lambda event: None  # type: ignore[method-assign]
                live.strategy.submit_order(close)
                engine = live.kernel.exec_engine
                engine.process(TestEventStubs.order_submitted(close, account_id=_ACCOUNT))
                engine.process(
                    TestEventStubs.order_accepted(
                        close, account_id=_ACCOUNT, venue_order_id=VenueOrderId("77")
                    )
                )
                engine.process(
                    TestEventStubs.order_filled(
                        close,
                        instrument=live.kernel.cache.instrument(live.iid),
                        account_id=_ACCOUNT,
                        venue_order_id=VenueOrderId("77"),
                        last_qty=Quantity.from_str("0.004"),
                        last_px=Price.from_str("11000.0"),
                    )
                )
                await live.drain()

        paper._trading_node = _StoppingNode()  # type: ignore[assignment]
        await paper._shutdown()
        # What NT's cache holds after the stop (this harness's cache is hedging: the
        # partial close is a second position beside the 0.010 one).
        in_cache = {str(p.id.value) for p in live.kernel.cache.positions_open()}
        store = StatePersistence(db_path=seeded["db"], trader_id="PAPER-007")
        try:
            saved = store.load_latest_checkpoint("PAPER-007")
        finally:
            store.close()

    assert saved is not None
    assert saved.node_status["owned_net"] == {"BTCUSDT": "0.006"}
    assert in_cache and set(saved.positions) == in_cache, "positions are not wiped"
    assert saved.balance, "the balance snapshot is not wiped"


async def test_stop_close_that_never_fills_keeps_the_position_in_the_record(
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """R3: when the close does not fill before stop_async returns, the saved record keeps
    the position (the next start decides from Binance as before) and no second save runs."""
    from vibe_quant.paper.persistence import StatePersistence

    prior = {**_RUNNING, "owned_net": {"BTCUSDT": "0.010"}, "own_orders": {}}
    async with owned_restart(seeded, prior, venue_long("0.010")) as (paper, live):
        assert await first_check(paper, live, "0.010") is True
        paper._persistence = StatePersistence(db_path=seeded["db"], trader_id="PAPER-007")

        class _StoppingNode:
            async def stop_async(_self) -> None:
                return None

        paper._trading_node = _StoppingNode()  # type: ignore[assignment]
        await paper._shutdown()
        saved = _latest_saved_after_close(seeded)

    assert saved["owned_net"] == {"BTCUSDT": "0.010"}


async def test_own_orders_are_saved_when_admitted(
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """R4: the SL and TP legs the guard admits after an entry fill are in the saved
    checkpoint at once (one save for the pair), so a kill before the periodic save does
    not leave them unknown: the restart finds the stop resting and passes with no
    foreign-order hold."""
    from vibe_quant.paper.persistence import StatePersistence

    prior = {**_RUNNING, "owned_net": {"BTCUSDT": "1.000"}, "own_orders": {}}
    async with owned_restart(seeded, prior, venue_long("1.000")) as (paper, live):
        assert await first_check(paper, live, "1.000") is True
        paper._persistence = StatePersistence(db_path=seeded["db"], trader_id="PAPER-007")
        guard = live.guard
        assert guard is not None
        guard._risk_engine = SimpleNamespace(execute=lambda command: None)
        saves: list[int] = []
        real_save = paper._save_checkpoint_now

        def counted() -> None:
            saves.append(1)
            real_save()

        paper._save_checkpoint_now = counted  # type: ignore[method-assign]
        factory = live.strategy.order_factory
        stop = factory.stop_market(
            live.iid,
            OrderSide.SELL,
            Quantity.from_str("1.000"),
            trigger_price=Price.from_str("9000.0"),
            reduce_only=True,
        )
        take = factory.limit(
            live.iid,
            OrderSide.SELL,
            Quantity.from_str("1.000"),
            Price.from_str("12000.0"),
            reduce_only=True,
        )
        live.strategy.submit_order(stop)
        live.strategy.submit_order(take)
        await live.drain()
        saved = _latest_saved(paper)  # nothing saves after this: the process is killed
        legs = {str(stop.client_order_id), str(take.client_order_id)}

    assert saves == [1], "one save for the pair"
    assert set(saved["own_orders"]) == legs
    assert saved["owned_net"] == {"BTCUSDT": "1.000"}
    prior = {**_RUNNING, "owned_net": saved["owned_net"], "own_orders": saved["own_orders"]}
    async with owned_restart(seeded, prior, resting_sl(str(stop.client_order_id))) as (
        paper,
        live,
    ):
        agreed = await first_check(paper, live, "1.000")
        held = paper._venue_hold()

    assert (agreed, held) == (True, None)
