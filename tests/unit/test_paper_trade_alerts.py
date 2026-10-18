"""Optional per-session Telegram trade alerts (bead vibe-quant-ci7q9).

``PaperTradingConfig.trade_alerts`` (default False) makes the paper node message the
operator on every position entry (size, leverage, entry, SL, TP) and exit (price,
reason, net P&L). Alerts are REPORT-ONLY: they never touch the event log, the guard's
risk evaluation or the trade path, and every failure is swallowed with a class-only log.

The entry text is built 2 s after the fill (inside the alert task, never on the NT
callback) because the compiled strategy submits its SL/TP orders only then
(``dsl/templates.py::_submit_sl_tp_orders``), so the operator sees the protective
levels the strategy actually placed.
"""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from nautilus_trader.common.component import TestClock
from nautilus_trader.common.factories import OrderFactory
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import PositionId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.position import Position
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.test_kit.stubs.events import TestEventStubs
from nautilus_trader.test_kit.stubs.identifiers import TestIdStubs

from tests.unit.test_paper_halt_restore import _trace_node
from tests.unit.test_paper_halt_restore import seeded as _restore_seeded
from vibe_quant.alerts import telegram as telegram_module
from vibe_quant.alerts.telegram import AlertType, TelegramBot, TelegramConfig
from vibe_quant.logging.events import EventType
from vibe_quant.paper import node as node_module
from vibe_quant.paper.guard import TradingGuard
from vibe_quant.paper.node import TRADE_ALERT_DELAY_S, PaperTradingNode, _GuardBridge

if TYPE_CHECKING:
    from pathlib import Path

BTC = "BTCUSDT-PERP.BINANCE"
TS = 1_700_000_000_000_000_000


@pytest.fixture()
def seeded(tmp_path: Path) -> dict[str, Any]:
    return _restore_seeded.__wrapped__(tmp_path)  # type: ignore[attr-defined,no-any-return]


# ----------------------------------------------------------------- fakes


class _Order:
    def __init__(
        self,
        order_type: str,
        *,
        reduce_only: bool = True,
        trigger_price: str | None = None,
        price: str | None = None,
        tags: tuple[str, ...] = (),
        instrument_id: str = BTC,
    ) -> None:
        self.order_type = SimpleNamespace(name=order_type)
        self.is_reduce_only = reduce_only
        self.tags = list(tags)
        self.instrument_id = instrument_id
        if trigger_price is not None:
            self.trigger_price = Decimal(trigger_price)
        if price is not None:
            self.price = Decimal(price)


class _Cache:
    def __init__(
        self,
        *,
        open_orders: tuple[Any, ...] = (),
        positions: dict[str, Any] | None = None,
        orders: dict[str, Any] | None = None,
        price_precision: int = 2,
        raising: bool = False,
        instrument_raises: bool = False,
    ) -> None:
        self._open_orders = list(open_orders)
        self._positions = positions or {}
        self._orders = orders or {}
        self._price_precision = price_precision
        self._raising = raising
        self._instrument_raises = instrument_raises
        self.reads = 0

    def orders_open(self, *a: Any, **k: Any) -> list[Any]:
        self.reads += 1
        if self._raising:
            raise RuntimeError("ZZCACHEDOWN")
        return list(self._open_orders)

    def orders_inflight(self, *a: Any, **k: Any) -> list[Any]:
        if self._raising:
            raise RuntimeError("ZZCACHEDOWN")
        return []

    def instrument(self, instrument_id: Any) -> Any:
        self.reads += 1
        if self._instrument_raises:
            raise OSError("ZZINSTRUMENTDOWN")
        return SimpleNamespace(price_precision=self._price_precision)

    def position(self, position_id: Any) -> Any:
        return self._positions.get(str(position_id))

    def order(self, order_id: Any) -> Any:
        return self._orders.get(str(order_id))


class _FakeTelegram:
    def __init__(self, *, result: bool = True, raises: Exception | None = None) -> None:
        self.calls: list[tuple[AlertType, str, bool]] = []
        self.result = result
        self.raises = raises

    async def send_alert(
        self, alert_type: AlertType, message: str, *, bypass_rate_limit: bool = False
    ) -> bool:
        self.calls.append((alert_type, message, bypass_rate_limit))
        if self.raises is not None:
            raise self.raises
        return self.result


# ----------------------------------------------------------------- events


class _Trip:
    """A REAL NautilusTrader round trip: Position + PositionOpened + PositionClosed.

    Real events matter: a NT ``PositionClosed.side`` is always FLAT (the direction lives
    in ``entry``), which a hand-built fake with ``side=LONG`` hides.
    """

    def __init__(
        self,
        entry: str = "BUY",
        quantity: str = "0.002",
        entry_px: str = "67600.10",
        exit_px: str = "67498.70",
    ) -> None:
        instrument = TestInstrumentProvider.btcusdt_perp_binance()
        factory = OrderFactory(
            trader_id=TestIdStubs.trader_id(),
            strategy_id=TestIdStubs.strategy_id(),
            clock=TestClock(),
        )
        side = OrderSide.BUY if entry == "BUY" else OrderSide.SELL
        flip = OrderSide.SELL if entry == "BUY" else OrderSide.BUY
        qty = Quantity.from_str(quantity)
        pid = PositionId("P-1")
        fill_in = TestEventStubs.order_filled(
            factory.market(instrument.id, side, qty),
            instrument,
            position_id=pid,
            last_px=Price.from_str(entry_px),
            ts_event=TS,
        )
        self.position = Position(instrument, fill_in)
        self.opened = TestEventStubs.position_opened(self.position)
        self.position.apply(
            TestEventStubs.order_filled(
                factory.market(instrument.id, flip, qty),
                instrument,
                position_id=pid,
                last_px=Price.from_str(exit_px),
                ts_event=TS + 1,
            )
        )
        self.closed = TestEventStubs.position_closed(self.position)

    def cache(self, closing: Any = None, **kw: Any) -> _Cache:
        orders = {str(self.closed.closing_order_id): closing} if closing is not None else {}
        return _Cache(positions={"P-1": self.position}, orders=orders, **kw)


def _node(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],
    *,
    telegram: _FakeTelegram | None,
    cache: _Cache | None = None,
    trade_alerts: bool = True,
    leverage: int = 10,
) -> tuple[PaperTradingNode, _FakeTelegram]:
    node, _trace = _trace_node(monkeypatch, seeded, trade_alerts=trade_alerts)
    fake = telegram if telegram is not None else _FakeTelegram()
    node._telegram = fake
    node._guard = SimpleNamespace(cache=cache) if cache is not None else None
    node._resolved = SimpleNamespace(leverage=leverage, margin_type="CROSSED")
    return node, fake


async def _drain(node: PaperTradingNode) -> None:
    """Wait for every fire-and-forget alert task to reach its final state."""
    while node._alert_tasks:
        await asyncio.gather(*list(node._alert_tasks), return_exceptions=True)


def _alert_calls(fake: _FakeTelegram) -> list[str]:
    return [message for _type, message, _bypass in fake.calls]


# ----------------------------------------------------------------- T1: gating


def test_delay_constant_is_two_seconds() -> None:
    """T2: the entry text is built two seconds after the fill (SL/TP in the cache)."""
    assert TRADE_ALERT_DELAY_S == 2.0


async def test_disabled_sends_nothing_and_keeps_the_event_log(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """T1: trade_alerts False (default) -> no messages, event log unchanged."""
    node, _trace = _trace_node(monkeypatch, seeded)  # trade_alerts defaults to False
    fake = _FakeTelegram()
    node._telegram = fake
    node._guard = SimpleNamespace(
        cache=_Cache(open_orders=(_Order("STOP_MARKET", trigger_price="67498.70"),))
    )
    node._resolved = SimpleNamespace(leverage=10, margin_type="CROSSED")

    trip = _Trip()
    node._on_position_opened(trip.opened)
    await _drain(node)
    node._on_position_closed(trip.closed)
    await _drain(node)

    assert fake.calls == []
    assert node._alert_tasks == set()
    opens = [c for c in _trace.calls if c[0] == "write" and c[1] == EventType.POSITION_OPEN]
    assert len(opens) == 1, "the POSITION_OPEN write is unchanged"
    assert opens[0][2]["entry_price"] == pytest.approx(67600.10)


async def test_enabled_without_telegram_sends_nothing(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """T1: trade_alerts True but Telegram not configured -> no task, no message."""
    node, _trace = _trace_node(monkeypatch, seeded, trade_alerts=True)
    node._telegram = None
    node._guard = SimpleNamespace(cache=_Cache())

    trip = _Trip()
    node._on_position_opened(trip.opened)
    node._on_position_closed(trip.closed)

    assert node._alert_tasks == set()


# ----------------------------------------------------------------- T2: entry


async def test_entry_alert_long_with_stop_and_take(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """T2: exact entry text for a LONG whose cache carries an SL and a TP."""
    monkeypatch.setattr(node_module, "TRADE_ALERT_DELAY_S", 0.0)
    cache = _Cache(
        open_orders=(
            _Order("STOP_MARKET", trigger_price="67498.70"),
            _Order("LIMIT", price="67701.50"),
        )
    )
    node, fake = _node(monkeypatch, seeded, telegram=None, cache=cache)

    node._on_position_opened(_Trip().opened)
    # Fire-and-forget: the NT callback returned before anything was sent.
    assert fake.calls == [], "the callback must not block on the alert"
    await _drain(node)

    assert _alert_calls(fake) == [
        "OPEN LONG BTCUSDT-PERP.BINANCE\n"
        "Size: 0.002 (≈ 135.20 USDT)\n"
        "Leverage: 10x\n"
        "Entry: 67600.10\n"
        "Stop loss: 67498.70\n"
        "Take profit: 67701.50\n"
        "Trader: PAPER-007"
    ]
    assert fake.calls[0][0] is AlertType.TRADE


async def test_entry_alert_short_without_protective_orders(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """T2: a SHORT with no reduce-only SL/TP -> both levels read 'none placed'.

    A non-reduce-only LIMIT in the cache must NOT be mistaken for a take-profit.
    """
    monkeypatch.setattr(node_module, "TRADE_ALERT_DELAY_S", 0.0)
    cache = _Cache(open_orders=(_Order("LIMIT", reduce_only=False, price="68000.00"),))
    node, fake = _node(monkeypatch, seeded, telegram=None, cache=cache)

    node._on_position_opened(_Trip("SELL", "1.000", "68000.00").opened)
    await _drain(node)

    assert _alert_calls(fake) == [
        "OPEN SHORT BTCUSDT-PERP.BINANCE\n"
        "Size: 1.0 (≈ 68000.00 USDT)\n"
        "Leverage: 10x\n"
        "Entry: 68000.00\n"
        "Stop loss: none placed\n"
        "Take profit: none placed\n"
        "Trader: PAPER-007"
    ]


# ----------------------------------------------------------------- T3: exit


@pytest.mark.parametrize(
    ("entry", "exit_px", "order_type", "reason", "direction", "pnl"),
    [
        ("BUY", "67498.70", "STOP_MARKET", "stop_loss", "LONG", "-0.25"),
        ("BUY", "68400.00", "LIMIT", "take_profit", "LONG", "1.55"),
        ("SELL", "67498.70", "LIMIT", "take_profit", "SHORT", "0.15"),
        ("SELL", "68400.00", "STOP_MARKET", "stop_loss", "SHORT", "-1.65"),
    ],
)
async def test_exit_alert_text(
    seeded: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
    exit_px: str,
    order_type: str,
    reason: str,
    direction: str,
    pnl: str,
) -> None:
    """T3 / F1: exact exit text for a REAL NT close of a LONG and of a SHORT.

    A real ``PositionClosed.side`` is FLAT; the closed direction comes from ``entry``.
    """
    trip = _Trip(entry=entry, exit_px=exit_px)
    assert trip.closed.side.name == "FLAT", "the NT invariant this test guards"
    node, fake = _node(
        monkeypatch, seeded, telegram=None, cache=trip.cache(closing=_Order(order_type))
    )

    node._on_position_closed(trip.closed)
    await _drain(node)

    assert _alert_calls(fake) == [
        f"CLOSE {direction} BTCUSDT-PERP.BINANCE\n"
        f"Exit: {exit_px} ({reason})\n"
        "Entry: 67600.10\n"
        f"P&L: {pnl} USDT (net, after 0.05 fees)\n"
        "Trader: PAPER-007"
    ]
    assert fake.calls[0][0] is AlertType.TRADE


# ----------------------------------------------------------------- T4: failures


async def test_send_raising_never_reaches_the_callback(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """T4: a raising send is swallowed with a class-only log and no exception text."""
    caplog.set_level(logging.WARNING)

    cache = _Cache(open_orders=(_Order("STOP_MARKET", trigger_price="67498.70"),))
    node, _fake = _node(monkeypatch, seeded, telegram=None, cache=cache)
    node._telegram = _FakeTelegram(raises=RuntimeError("QQMARK"))
    monkeypatch.setattr(node_module, "TRADE_ALERT_DELAY_S", 0.0)

    node._on_position_opened(_Trip().opened)  # must not raise into the callback
    await _drain(node)

    assert node._alert_tasks == set()
    assert "trade alert not sent (RuntimeError)" in caplog.text
    assert "ZZMARK" not in caplog.text and "QQMARK" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


async def test_returning_false_never_reaches_the_callback(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """T4: a rate-limited/failed send (returns False) is swallowed."""
    monkeypatch.setattr(node_module, "TRADE_ALERT_DELAY_S", 0.0)
    node, _fake = _node(monkeypatch, seeded, telegram=None, cache=_Cache())
    fake = _FakeTelegram(result=False)
    node._telegram = fake

    node._on_position_opened(_Trip().opened)
    await _drain(node)

    assert len(fake.calls) == 1
    assert node._alert_tasks == set()


async def test_cache_lookup_raising_is_swallowed(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """T4: a raising cache lookup is caught and logged class-only."""
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(node_module, "TRADE_ALERT_DELAY_S", 0.0)
    node, _fake = _node(monkeypatch, seeded, telegram=None, cache=_Cache(raising=True))

    node._on_position_opened(_Trip().opened)
    await _drain(node)

    assert node._alert_tasks == set()
    assert "trade alert not sent (RuntimeError)" in caplog.text
    assert "ZZCACHEDOWN" not in caplog.text


def test_no_running_loop_is_swallowed(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """T4: without a running loop the alert is dropped with a class-only log."""
    caplog.set_level(logging.WARNING)
    node, _fake = _node(monkeypatch, seeded, telegram=None, cache=_Cache())

    node._on_position_opened(_Trip().opened)  # sync test: no running loop

    assert node._alert_tasks == set()
    assert "trade alert not sent (RuntimeError)" in caplog.text


# ----------------------------------------------- R31: TRADE rate-limit bypass


async def test_fast_open_then_close_sends_both_alerts(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R31: a CLOSE one second after an OPEN is not dropped by the TRADE rate limit.

    OPEN and CLOSE both use ``AlertType.TRADE``, whose per-type limit is
    ``RATE_LIMIT_SECONDS == 60``: without the bypass the CLOSE of a fast round trip
    is silently dropped. This drives the REAL ``TelegramBot`` over faked HTTP (not the
    stub), so the rate-limit branch is actually exercised -- on the pre-fix code only
    the OPEN request is made. The clock advances exactly 1 s between the two fills.
    """
    monkeypatch.setattr(node_module, "TRADE_ALERT_DELAY_S", 0.0)
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"ok": True})

    bot = TelegramBot(TelegramConfig(bot_token="123:ABC", chat_id="456"))
    bot._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    clock = {"t": 1_000.0}
    monkeypatch.setattr(telegram_module, "time", SimpleNamespace(time=lambda: clock["t"]))

    trip = _Trip()
    cache = trip.cache(
        closing=_Order("STOP_MARKET"),
        open_orders=(_Order("STOP_MARKET", trigger_price="67498.70"),),
    )
    node, _trace = _trace_node(monkeypatch, seeded, trade_alerts=True)
    node._telegram = bot
    node._guard = SimpleNamespace(cache=cache)
    node._resolved = SimpleNamespace(leverage=10, margin_type="CROSSED")

    try:
        node._on_position_opened(trip.opened)
        await _drain(node)
        clock["t"] += 1.0
        node._on_position_closed(trip.closed)
        await _drain(node)

        assert len(sent) == 2, "a fast OPEN/CLOSE round trip must deliver BOTH alerts"
        first, second = (request.content.decode() for request in sent)
        assert "[TRADE]" in first and "OPEN LONG BTCUSDT-PERP.BINANCE" in first
        assert "[TRADE]" in second and "CLOSE LONG BTCUSDT-PERP.BINANCE" in second
        assert bot._last_alert_times[AlertType.TRADE] == 1_001.0
    finally:
        await bot.close()


# ----------------------------------------------- F3: the entry delay is real


async def test_entry_alert_waits_two_seconds_before_reading_the_cache(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """T2 / F3: the task sleeps 2.0 s first; the SL/TP that land DURING the wait show up.

    The sleep is held open (an event, no wall-clock wait): nothing may read the cache or
    send before it is released, and an order added while it waits is in the message.
    """
    real_sleep = asyncio.sleep
    slept: list[float] = []
    release = asyncio.Event()

    async def held_sleep(delay: float) -> None:
        slept.append(delay)
        await release.wait()

    cache = _Cache()
    node, fake = _node(monkeypatch, seeded, telegram=None, cache=cache)
    monkeypatch.setattr(asyncio, "sleep", held_sleep)

    node._on_position_opened(_Trip().opened)
    await real_sleep(0.05)  # let the task start and park on the sleep

    assert slept == [2.0]
    assert cache.reads == 0, "no cache read before the delay has elapsed"
    assert fake.calls == []

    cache._open_orders.append(_Order("STOP_MARKET", trigger_price="67498.70"))
    cache._open_orders.append(_Order("LIMIT", price="67701.50"))
    release.set()
    await _drain(node)

    assert cache.reads > 0
    message = _alert_calls(fake)[0]
    assert "Stop loss: 67498.70" in message and "Take profit: 67701.50" in message


# ------------------------------------- F2: alerts never disturb the NT callback


@pytest.mark.parametrize("alerts_on", [False, True])
async def test_instrument_lookup_failure_never_skips_the_close_bookkeeping(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch, alerts_on: bool
) -> None:
    """F2: ``cache.instrument`` raising must not skip the close write, the losing-streak
    update or ``evaluate_risk`` (the exact trace main produces), alerts off OR on.

    Drives the REAL ``TradingGuard._on_position_event`` -> ``_GuardBridge`` -> node path
    with a real NT ``PositionClosed``.
    """
    trip = _Trip()  # a losing LONG
    node, trace = _trace_node(monkeypatch, seeded, trade_alerts=alerts_on)
    cache = trip.cache(closing=_Order("STOP_MARKET"), instrument_raises=True)
    if alerts_on:
        node._telegram = _FakeTelegram()
    node._guard = SimpleNamespace(cache=cache)
    node._resolved = SimpleNamespace(leverage=10, margin_type="CROSSED")
    guard = SimpleNamespace(
        cache=cache,
        _risk=SimpleNamespace(consecutive_losses=0),
        _listener=_GuardBridge(node),
        _is_guarded_strategy=lambda _sid: True,
        _stop_exited_strategies=lambda: trace.calls.append(("stop_exited",)),
        evaluate_risk=lambda: trace.calls.append(("evaluate_risk",)),
    )

    TradingGuard._on_position_event(guard, trip.closed)  # type: ignore[arg-type]
    await _drain(node)

    assert [call[0] for call in trace.calls] == ["write", "stop_exited", "evaluate_risk"]
    assert trace.calls[0][1] == EventType.POSITION_CLOSE
    assert guard._risk.consecutive_losses == 1
