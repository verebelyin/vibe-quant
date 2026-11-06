"""Paper ATR sizing on the REAL NautilusTrader ``TradingNode`` (vibe-quant-hrhok).

A restart leaves the sizing ATR cold: the node requests no history, so the ATR sees only live
closed bars. The risk-officer's condition: with a position adopted from the venue and a cold
ATR, (1) the adopted position gets its SL/TP re-armed at once, and (2) no new entry is sent
until the ATR has ``sizing_atr_period`` closed bars. The approval switch is monkeypatched on:
paper still refuses ATR runs by default (``test_paper_atr_sizing.py``).

The exec client is a stub (no network); the strategy, risk engine, exec engine, data engine
and the paper ``TradingGuard`` are the real live ones. Bars are fed through the live data
engine.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from nautilus_trader.live.factories import LiveDataClientFactory, LiveExecClientFactory
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import LiquiditySide, OmsType, OrderSide, OrderType, PositionSide
from nautilus_trader.model.identifiers import (
    AccountId,
    InstrumentId,
    PositionId,
    TradeId,
    VenueOrderId,
)
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.model.position import Position
from nautilus_trader.test_kit.stubs.events import TestEventStubs
from nautilus_trader.test_kit.stubs.execution import TestExecStubs

from tests.unit.test_paper_atr_sizing import _node
from tests.unit.test_paper_halt_restore import node_timing  # noqa: F401  # fixture
from tests.unit.test_paper_margin_refusal_real_node import (
    _FastNodeConfig,
    _StubDataClient,
    _VerifiedExecClient,
)
from vibe_quant.data.catalog import get_bar_type
from vibe_quant.nt_compat import retain_log_guard
from vibe_quant.paper import node as node_mod

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

_PERIOD = 5
_INSTRUMENT = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
_ADOPTED_QTY = "0.010"
_ADOPTED_ID = PositionId("P-ADOPTED")
#: Always-long; EMA(2) is ready after two bars, so the SIZING ATR(5) is the only entry gate
#: left after bar 2. Fixed-percent SL/TP: re-armable with no indicator warm-up.
_SOURCE: dict[str, object] = {
    "name": "atr_restart_probe",
    "timeframe": "1h",
    "indicators": {"ema": {"type": "EMA", "period": 2}},
    "entry_conditions": {"long": ["ema > 0"]},
    "stop_loss": {"type": "fixed_pct", "percent": 5.0},
    "take_profit": {"type": "fixed_pct", "percent": 10.0},
}
_PARAMS: dict[str, object] = {
    "sizing_mode": "atr",
    "sizing_atr_period": _PERIOD,
    "sizing_atr_multiplier": 2.0,
}


class _Venue:
    """What reached the stub exec client, in order."""

    submitted: list[Any] = []  # noqa: RUF012 - reset per test
    adopt_for: Any = None


class _AdoptingExecClient(_VerifiedExecClient):
    """Healthy venue that already holds a long BTC position for the strategy at connect."""

    async def _connect(self) -> None:
        await super()._connect()
        strategy_id = _Venue.adopt_for
        instrument = self._cache.instrument(_INSTRUMENT)
        order = TestExecStubs.market_order(
            instrument=instrument,
            order_side=OrderSide.BUY,
            quantity=Quantity.from_str(_ADOPTED_QTY),
        )
        fill = TestEventStubs.order_filled(
            order,
            instrument=instrument,
            strategy_id=strategy_id,
            account_id=AccountId("BINANCE-001"),
            position_id=_ADOPTED_ID,
            last_px=Price.from_str("40000.0"),
        )
        self._cache.add_position(Position(instrument=instrument, fill=fill), OmsType.HEDGING)

    async def _submit_order(self, command: Any) -> None:
        order = command.order
        _Venue.submitted.append(order)
        ts = self._clock.timestamp_ns()
        self.generate_order_submitted(
            order.strategy_id, order.instrument_id, order.client_order_id, ts
        )
        self.generate_order_accepted(
            order.strategy_id,
            order.instrument_id,
            order.client_order_id,
            VenueOrderId(f"V{len(_Venue.submitted)}"),
            ts,
        )

    async def _cancel_order(self, command: Any) -> None:
        return

    async def _cancel_all_orders(self, command: Any) -> None:
        return

    def fill_at(self, order: Any, px: str) -> None:
        """The venue fills ``order`` fully (the SL closes the adopted long)."""
        self.generate_order_filled(
            strategy_id=order.strategy_id,
            instrument_id=order.instrument_id,
            client_order_id=order.client_order_id,
            venue_order_id=order.venue_order_id,
            venue_position_id=_ADOPTED_ID,
            trade_id=TradeId("T-1"),
            order_side=order.side,
            order_type=order.order_type,
            last_qty=order.quantity,
            last_px=Price.from_str(px),
            quote_currency=USDT,
            commission=Money(0, USDT),
            liquidity_side=LiquiditySide.TAKER,
            ts_event=self._clock.timestamp_ns(),
        )


class _BarDataClient(_StubDataClient):
    """The stub data client, accepting the strategy's bar subscription."""

    async def _subscribe_bars(self, command: Any) -> None:
        return


class _BarDataFactory(LiveDataClientFactory):
    @staticmethod
    def create(  # type: ignore[override]
        loop: Any, name: str, config: Any, msgbus: Any, cache: Any, clock: Any
    ) -> _BarDataClient:
        return _BarDataClient(loop, msgbus, cache, clock)


class _AdoptingExecFactory(LiveExecClientFactory):
    @staticmethod
    def create(  # type: ignore[override]
        loop: Any, name: str, config: Any, msgbus: Any, cache: Any, clock: Any
    ) -> Any:
        client = _AdoptingExecClient(loop, msgbus, cache, clock)
        _AdoptingExecFactory.created = client
        return client

    created: _AdoptingExecClient


def _bar(index: int) -> Bar:
    """Hourly bar ``index`` with a 2000-point range (ATR > 0 once it has two bars)."""
    bar_type = get_bar_type("BTCUSDT", "1h")
    ts = 1_704_067_200_000_000_000 + index * 3_600_000_000_000
    return Bar(
        bar_type,
        Price.from_str("40000.0"),
        Price.from_str("41000.0"),
        Price.from_str("39000.0"),
        Price.from_str("40000.0"),
        Quantity.from_str("10.000"),
        ts,
        ts,
    )


def _entries() -> list[Any]:
    return [o for o in _Venue.submitted if not o.is_reduce_only]


def _protection() -> list[Any]:
    return [o for o in _Venue.submitted if o.is_reduce_only]


async def _until(predicate: Any, what: str, timeout: float = 15.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, f"timed out waiting for {what}"
        await asyncio.sleep(0.01)


async def test_restart_with_adopted_position_and_cold_atr_arms_protection_and_defers_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nautilus_trader.adapters.binance.factories as binance_factories

    monkeypatch.setattr(node_mod, "ATR_SIZING_APPROVED", True)
    _Venue.submitted = []
    monkeypatch.setattr(binance_factories, "BinanceLiveExecClientFactory", _AdoptingExecFactory)
    monkeypatch.setattr(binance_factories, "BinanceLiveDataClientFactory", _BarDataFactory)
    monkeypatch.setattr("nautilus_trader.live.config.TradingNodeConfig", _FastNodeConfig)

    async def _idle_watch() -> None:
        await asyncio.Event().wait()

    node, _module, _dsl = _node(tmp_path, dict(_PARAMS), "1h", source=_SOURCE)
    monkeypatch.setattr(node, "_liquidation_loop", _idle_watch)
    await node._initialize()
    trading_node: Any = node._trading_node
    assert trading_node is not None
    strategy = node._compiled_strategies[0]
    _Venue.adopt_for = strategy.id
    assert (strategy.config.sizing_mode, strategy.config.sizing_atr_period) == ("atr", _PERIOD)

    run_loop = asyncio.create_task(node._run_loop())
    try:
        # (1) the adopted long is protected at start, before any bar: a reduce-only stop and
        # a reduce-only take-profit, each for the adopted quantity.
        await _until(lambda: len(_protection()) >= 2, "SL and TP for the adopted position")
        stop = next(o for o in _protection() if o.order_type == OrderType.STOP_MARKET)
        take = next(o for o in _protection() if o.order_type == OrderType.LIMIT)
        assert (stop.side, take.side) == (OrderSide.SELL, OrderSide.SELL)
        assert (stop.quantity, take.quantity) == (Quantity.from_str(_ADOPTED_QTY),) * 2
        assert stop.trigger_price == Price.from_str("38000.0")  # 40000 less 5 %
        assert take.price == Price.from_str("44000.0")  # 40000 plus 10 %
        assert _entries() == []

        # the position is the strategy's, still open
        cache = trading_node.kernel.cache
        assert [p.side for p in cache.positions_open()] == [PositionSide.LONG]

        # (2) bars 0-1 (position open), then the SL fills: flat, EMA ready, sizing ATR has
        # 2 of 5 bars. Bars 2-3 would each enter if the ATR did not gate them.
        engine = trading_node.kernel.data_engine
        for i in (0, 1):
            engine.process(_bar(i))
        await _until(lambda: cache.bar(get_bar_type("BTCUSDT", "1h")) is not None, "bar 1")
        # flat at the entry price (zero PnL): the balance stays 1000 whatever the account's
        # update timing, so the ATR size below is exact
        _AdoptingExecFactory.created.fill_at(stop, "40000.0")
        await _until(lambda: not cache.positions_open(), "the SL to close the position")
        for i in (2, 3):
            engine.process(_bar(i))
            await asyncio.sleep(0.3)
        assert _entries() == [], "an entry was sent while the sizing ATR was cold"

        # bar 4 is the 5th closed bar: ATR(5) initialized, sized by ATR (a real entry)
        engine.process(_bar(4))
        await _until(lambda: len(_entries()) == 1, "the first entry once the ATR is warm")
        entry = _entries()[0]
        assert entry.side == OrderSide.BUY
        # risk 2 % of 1000 = 20; ATR = 2000 (every bar's high-low), x 2.0 -> 4000; 20 / 4000
        # = 0.005. Not the 5 % stop's 0.010 (risk mode) nor the position cap's 0.012.
        assert entry.quantity == Quantity.from_str("0.005")
        node._shutdown_event.set()
        await asyncio.wait_for(run_loop, timeout=10.0)
    finally:
        node._shutdown_event.set()
        if not run_loop.done():
            run_loop.cancel()
            await asyncio.gather(run_loop, return_exceptions=True)
        await asyncio.wait_for(node._shutdown(), timeout=5.0)
        retain_log_guard(trading_node)
