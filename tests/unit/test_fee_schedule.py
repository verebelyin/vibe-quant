"""Fee schedule + schedule-driven FeeModel (bd vibe-quant-2gyk1.4, D3-1).

Covers the Binance USDT-M VIP0 schedule (optional BNB discount) and the
``ScheduleFeeModel`` that charges it, plus the venue plumbing that keeps the
default (VIP0, no BNB) on NT's ``MakerTakerFeeModel`` path bit-identical.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from nautilus_trader.backtest.config import FeeModelFactory
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import FillModel, MakerTakerFeeModel
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import (
    AccountType,
    LiquiditySide,
    OmsType,
    OrderSide,
    OrderType,
    TimeInForce,
)
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.test_kit.stubs.events import TestEventStubs
from nautilus_trader.test_kit.stubs.execution import TestExecStubs
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

from vibe_quant.data.catalog import create_instrument, get_bar_type
from vibe_quant.fees import FeeSchedule
from vibe_quant.nt_compat import retain_log_guard
from vibe_quant.validation.extraction import all_positions, mark_open_positions
from vibe_quant.validation.fee_model import ScheduleFeeModel, ScheduleFeeModelConfig
from vibe_quant.validation.venue import (
    BINANCE_MAKER_FEE,
    BINANCE_TAKER_FEE,
    VenueConfig,
    create_backtest_venue_config,
)

IID = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
BT = get_bar_type("BTCUSDT", "1m")
_MIN_NS = 60_000_000_000
_QTY = Decimal("0.050")

BNB = FeeSchedule(tier="VIP0", bnb_discount=True)
VIP0 = FeeSchedule()


class _Cfg(StrategyConfig, frozen=True):
    bar_type: BarType = BT


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _bar(p: float, minute: int) -> Bar:
    return Bar(
        bar_type=BT,
        open=Price.from_str(f"{p:.1f}"),
        high=Price.from_str(f"{p + 5:.1f}"),
        low=Price.from_str(f"{p - 5:.1f}"),
        close=Price.from_str(f"{p:.1f}"),
        volume=Quantity.from_str("100.000"),
        ts_event=minute * _MIN_NS,
        ts_init=(minute + 1) * _MIN_NS - 1_000_000,
    )


def _bar_ohlc(o: float, h: float, low: float, c: float, minute: int) -> Bar:
    return Bar(
        bar_type=BT,
        open=Price.from_str(f"{o:.1f}"),
        high=Price.from_str(f"{h:.1f}"),
        low=Price.from_str(f"{low:.1f}"),
        close=Price.from_str(f"{c:.1f}"),
        volume=Quantity.from_str("100.000"),
        ts_event=minute * _MIN_NS,
        ts_init=(minute + 1) * _MIN_NS - 1_000_000,
    )


def _engine(fee_model: object) -> BacktestEngine:
    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    retain_log_guard(engine)
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(1_000, USDT)],
        default_leverage=Decimal("10"),
        fill_model=FillModel(prob_fill_on_limit=1.0, prob_slippage=0.0, random_seed=42),
        fee_model=fee_model,  # type: ignore[arg-type]
        bar_execution=True,
        use_position_ids=True,
        use_reduce_only=True,
    )
    engine.add_instrument(create_instrument("BTCUSDT"))
    return engine


def _dispose(engine: BacktestEngine) -> None:
    engine.reset()
    engine.dispose()


def _fills(engine: BacktestEngine) -> list[object]:
    fills: list[object] = []
    for pos in sorted(all_positions(engine), key=lambda p: int(p.ts_opened)):
        fills.extend(pos.events)
    return fills


def _expected(inst: object, fill_qty: object, fill_px: object, rate: Decimal) -> Money:
    notional = inst.notional_value(  # type: ignore[attr-defined]
        fill_qty, fill_px, use_quote_for_inverse=False
    )
    return Money(notional.as_decimal() * rate, USDT)


class _AlternatingMarket(Strategy):
    """One market order per bar, side alternating -> 10 round trips over 20 bars."""

    def __init__(self, config: _Cfg) -> None:
        super().__init__(config)
        self._long = False

    def on_start(self) -> None:
        self.subscribe_bars(BT)

    def on_bar(self, bar: Bar) -> None:
        inst = self.cache.instrument(IID)
        side = OrderSide.SELL if self._long else OrderSide.BUY
        self.submit_order(
            self.order_factory.market(
                IID, side, inst.make_qty(_QTY), time_in_force=TimeInForce.IOC
            )
        )
        self._long = not self._long


class _MakerThenTaker(Strategy):
    """Resting BUY limit at 9900 (fills as MAKER), then a market SELL (TAKER)."""

    def __init__(self, config: _Cfg) -> None:
        super().__init__(config)
        self._i = 0

    def on_start(self) -> None:
        self.subscribe_bars(BT)

    def on_bar(self, bar: Bar) -> None:
        inst = self.cache.instrument(IID)
        if self._i == 0:
            self.submit_order(
                self.order_factory.limit(
                    IID,
                    OrderSide.BUY,
                    inst.make_qty(_QTY),
                    Price.from_str("9900.0"),
                    time_in_force=TimeInForce.GTC,
                )
            )
        elif self._i == 2:
            self.submit_order(
                self.order_factory.market(
                    IID, OrderSide.SELL, inst.make_qty(_QTY), time_in_force=TimeInForce.IOC
                )
            )
        self._i += 1


class _TakeProfit(Strategy):
    """Market BUY at bar 0, resting SELL limit (TP) at 10200 from bar 1."""

    def __init__(self, config: _Cfg) -> None:
        super().__init__(config)
        self._i = 0

    def on_start(self) -> None:
        self.subscribe_bars(BT)

    def on_bar(self, bar: Bar) -> None:
        inst = self.cache.instrument(IID)
        if self._i == 0:
            self.submit_order(
                self.order_factory.market(
                    IID, OrderSide.BUY, inst.make_qty(_QTY), time_in_force=TimeInForce.IOC
                )
            )
        elif self._i == 1:
            self.submit_order(
                self.order_factory.limit(
                    IID,
                    OrderSide.SELL,
                    inst.make_qty(_QTY),
                    Price.from_str("10200.0"),
                    time_in_force=TimeInForce.GTC,
                )
            )
        self._i += 1


class _StopLoss(Strategy):
    """Market BUY at bar 0, SELL stop-market (SL) at 9800 from bar 1."""

    def __init__(self, config: _Cfg) -> None:
        super().__init__(config)
        self._i = 0

    def on_start(self) -> None:
        self.subscribe_bars(BT)

    def on_bar(self, bar: Bar) -> None:
        inst = self.cache.instrument(IID)
        if self._i == 0:
            self.submit_order(
                self.order_factory.market(
                    IID, OrderSide.BUY, inst.make_qty(_QTY), time_in_force=TimeInForce.IOC
                )
            )
        elif self._i == 1:
            self.submit_order(
                self.order_factory.stop_market(
                    IID,
                    OrderSide.SELL,
                    inst.make_qty(_QTY),
                    Price.from_str("9800.0"),
                    time_in_force=TimeInForce.GTC,
                )
            )
        self._i += 1


class _OpenAtEnd(Strategy):
    """Buys on the last bar only, leaving the position open at the end of the run."""

    def __init__(self, config: _Cfg, last_bar: int) -> None:
        super().__init__(config)
        self._i = 0
        self._last = last_bar

    def on_start(self) -> None:
        self.subscribe_bars(BT)

    def on_bar(self, bar: Bar) -> None:
        if self._i == self._last:
            inst = self.cache.instrument(IID)
            self.submit_order(
                self.order_factory.market(
                    IID, OrderSide.BUY, inst.make_qty(_QTY), time_in_force=TimeInForce.IOC
                )
            )
        self._i += 1


_10_RT_PRICES = [
    10000.0, 10100.0, 10200.0, 10300.0, 10400.0, 10500.0, 10600.0, 10700.0, 10800.0, 10900.0,
    11000.0, 11100.0, 11200.0, 11300.0, 11400.0, 11500.0, 11600.0, 11700.0, 11800.0, 11900.0,
]


def _run_ten_round_trips(fee_model: object) -> BacktestEngine:
    engine = _engine(fee_model)
    engine.add_data([_bar(p, i) for i, p in enumerate(_10_RT_PRICES)])
    engine.add_strategy(_AlternatingMarket(_Cfg()))
    engine.run()
    return engine


# ---------------------------------------------------------------------------
# RED test: 10 market round trips charged the BNB taker rate (0.00045)
# ---------------------------------------------------------------------------


def test_ten_trade_fixture_taker_bnb() -> None:
    fee_model = FeeModelFactory.create(
        create_backtest_venue_config(VenueConfig(fee_schedule=BNB)).fee_model
    )
    engine = _run_ten_round_trips(fee_model)
    try:
        inst = engine.cache.instrument(IID)
        positions = all_positions(engine)
        assert len(positions) == 10
        assert all(p.is_closed for p in positions)

        fills = _fills(engine)
        assert len(fills) == 20  # 10 round trips = 20 fills
        for fill in fills:
            assert fill.liquidity_side == LiquiditySide.TAKER
            assert fill.commission == _expected(inst, fill.last_qty, fill.last_px, Decimal("0.00045"))
        # Concrete Decimal arithmetic: 0.05 * 10000 * 0.00045 == 0.225
        assert Money(Decimal("0.225"), USDT) in [f.commission for f in fills]
    finally:
        _dispose(engine)


def test_ten_trade_fixture_taker_vip0() -> None:
    engine = _run_ten_round_trips(
        ScheduleFeeModel(
            config=ScheduleFeeModelConfig(
                maker_rate=str(VIP0.maker_rate), taker_rate=str(VIP0.taker_rate)
            )
        )
    )
    try:
        inst = engine.cache.instrument(IID)
        fills = _fills(engine)
        assert len(fills) == 20
        for fill in fills:
            assert fill.commission == _expected(inst, fill.last_qty, fill.last_px, Decimal("0.0005"))
        assert Money(Decimal("0.25"), USDT) in [f.commission for f in fills]
    finally:
        _dispose(engine)


# ---------------------------------------------------------------------------
# Model equivalence / liquidity side handling
# ---------------------------------------------------------------------------


def _run_maker_then_taker(fee_model: object) -> BacktestEngine:
    engine = _engine(fee_model)
    engine.add_data([_bar(p, i) for i, p in enumerate([10000.0, 9850.0, 10000.0, 10000.0])])
    engine.add_strategy(_MakerThenTaker(_Cfg()))
    engine.run()
    return engine


def test_schedule_model_matches_maker_taker_bit_identical() -> None:
    reference = _run_maker_then_taker(MakerTakerFeeModel())
    scheduled = _run_maker_then_taker(
        ScheduleFeeModel(
            config=ScheduleFeeModelConfig(
                maker_rate=str(VIP0.maker_rate), taker_rate=str(VIP0.taker_rate)
            )
        )
    )
    try:
        ref_fills = _fills(reference)
        got_fills = _fills(scheduled)
        assert len(ref_fills) == len(got_fills) == 2
        sides = {f.liquidity_side for f in got_fills}
        assert sides == {LiquiditySide.MAKER, LiquiditySide.TAKER}
        for ref, got in zip(ref_fills, got_fills, strict=True):
            assert ref.commission == got.commission
            assert ref.liquidity_side == got.liquidity_side
    finally:
        _dispose(reference)
        _dispose(scheduled)


_ORDER_QTY = Decimal("0.050")
_FILL_QTY = Decimal("0.010")
_FILL_PX = Decimal("10000.0")


def _partial_fill_order(instrument: object, liquidity_side: LiquiditySide) -> object:
    """An order for 0.050 that only fills 0.010, with the given liquidity side."""
    order = TestExecStubs.market_order(
        instrument=instrument,
        order_side=OrderSide.BUY,
        quantity=instrument.make_qty(_ORDER_QTY),  # type: ignore[attr-defined]
    )
    order.apply(TestEventStubs.order_submitted(order=order))
    order.apply(TestEventStubs.order_accepted(order=order))
    order.apply(
        TestEventStubs.order_filled(
            order=order,
            instrument=instrument,
            last_qty=instrument.make_qty(_FILL_QTY),  # type: ignore[attr-defined]
            last_px=instrument.make_price(_FILL_PX),  # type: ignore[attr-defined]
            liquidity_side=liquidity_side,
            commission=Money(0, USDT),
        )
    )
    return order


def test_schedule_model_charges_fill_qty_not_order_qty() -> None:
    """A 0.050 order partially filling 0.010 @ 10000 is charged on the FILL.

    ScheduleFeeModel must equal MakerTakerFeeModel's commission exactly (Money
    equality) for both liquidity sides; because the order quantity (0.050) differs
    from the fill quantity (0.010), this fails if the model ever reads
    ``order.quantity`` instead of ``fill_qty``.
    """
    inst = create_instrument("BTCUSDT")
    fill_qty = inst.make_qty(_FILL_QTY)
    fill_px = inst.make_price(_FILL_PX)

    vip0 = ScheduleFeeModel(
        config=ScheduleFeeModelConfig(
            maker_rate=str(VIP0.maker_rate), taker_rate=str(VIP0.taker_rate)
        )
    )
    bnb = ScheduleFeeModel(
        config=ScheduleFeeModelConfig(
            maker_rate=str(BNB.maker_rate), taker_rate=str(BNB.taker_rate)
        )
    )
    reference = MakerTakerFeeModel()

    # 0.010 * 10000 = 100 notional; x rate -> exact USDT.
    expected_vip0 = {
        LiquiditySide.MAKER: Money(Decimal("0.02"), USDT),  # 100 * 0.0002
        LiquiditySide.TAKER: Money(Decimal("0.05"), USDT),  # 100 * 0.0005
    }
    expected_bnb = {
        LiquiditySide.MAKER: Money(Decimal("0.018"), USDT),  # 100 * 0.00018
        LiquiditySide.TAKER: Money(Decimal("0.045"), USDT),  # 100 * 0.00045
    }

    for liquidity_side in (LiquiditySide.MAKER, LiquiditySide.TAKER):
        order = _partial_fill_order(inst, liquidity_side)
        # Partial fill: order quantity is strictly larger than the fill.
        assert float(order.quantity) == 0.05  # type: ignore[attr-defined]
        assert float(order.filled_qty) == 0.01  # type: ignore[attr-defined]
        assert order.liquidity_side == liquidity_side  # type: ignore[attr-defined]

        got_vip0 = vip0.get_commission(order, fill_qty, fill_px, inst)  # type: ignore[arg-type]
        got_bnb = bnb.get_commission(order, fill_qty, fill_px, inst)  # type: ignore[arg-type]
        got_ref = reference.get_commission(order, fill_qty, fill_px, inst)  # type: ignore[arg-type]

        # VIP0 schedule is bit-identical to NT's MakerTakerFeeModel (both read the fill).
        assert got_vip0 == got_ref
        assert got_vip0 == expected_vip0[liquidity_side]
        assert got_bnb == expected_bnb[liquidity_side]


def _run_exit(fee_model: object, strategy: Strategy, bars: list[Bar]) -> BacktestEngine:
    engine = _engine(fee_model)
    engine.add_data(bars)
    engine.add_strategy(strategy)
    engine.run()
    return engine


def test_tp_limit_maker_sl_stop_taker() -> None:
    fee_model = ScheduleFeeModel(
        config=ScheduleFeeModelConfig(maker_rate="0.00018", taker_rate="0.00045")
    )
    tp_bars = [
        _bar_ohlc(10000, 10005, 9995, 10000, 0),
        _bar_ohlc(10000, 10005, 9995, 10000, 1),
        _bar_ohlc(10000, 10300, 9995, 10300, 2),
    ]
    sl_bars = [
        _bar_ohlc(10000, 10005, 9995, 10000, 0),
        _bar_ohlc(10000, 10005, 9995, 10000, 1),
        _bar_ohlc(10000, 10005, 9700, 9750, 2),
    ]
    tp_engine = _run_exit(fee_model, _TakeProfit(_Cfg()), tp_bars)
    sl_engine = _run_exit(fee_model, _StopLoss(_Cfg()), sl_bars)
    try:
        tp_inst = tp_engine.cache.instrument(IID)
        sl_inst = sl_engine.cache.instrument(IID)

        tp_exit = [f for f in _fills(tp_engine) if f.order_type == OrderType.LIMIT]
        sl_exit = [f for f in _fills(sl_engine) if f.order_type == OrderType.STOP_MARKET]
        assert len(tp_exit) == 1 and len(sl_exit) == 1

        assert float(tp_exit[0].last_px) == 10200.0
        assert tp_exit[0].liquidity_side == LiquiditySide.MAKER
        assert tp_exit[0].commission == _expected(
            tp_inst, tp_exit[0].last_qty, tp_exit[0].last_px, Decimal("0.00018")
        )

        assert float(sl_exit[0].last_px) == 9800.0
        assert sl_exit[0].liquidity_side == LiquiditySide.TAKER
        assert sl_exit[0].commission == _expected(
            sl_inst, sl_exit[0].last_qty, sl_exit[0].last_px, Decimal("0.00045")
        )
    finally:
        _dispose(tp_engine)
        _dispose(sl_engine)


# ---------------------------------------------------------------------------
# Venue plumbing
# ---------------------------------------------------------------------------


def test_default_schedule_keeps_maker_taker_path() -> None:
    default_venue = create_backtest_venue_config(VenueConfig())
    assert default_venue.fee_model.fee_model_path == "nautilus_trader.backtest.models:MakerTakerFeeModel"
    assert default_venue.fee_model.config_path == "nautilus_trader.config:MakerTakerFeeModelConfig"
    assert default_venue.fee_model.config == {}

    bnb_venue = create_backtest_venue_config(VenueConfig(fee_schedule=BNB))
    assert bnb_venue.fee_model.fee_model_path == "vibe_quant.validation.fee_model:ScheduleFeeModel"
    assert (
        bnb_venue.fee_model.config_path
        == "vibe_quant.validation.fee_model:ScheduleFeeModelConfig"
    )
    assert bnb_venue.fee_model.config == {"maker_rate": "0.00018", "taker_rate": "0.00045"}
    # The config must actually resolve and build through NT's own factory.
    built = FeeModelFactory.create(bnb_venue.fee_model)
    assert isinstance(built, ScheduleFeeModel)
    assert built.taker_rate == Decimal("0.00045")


def test_venue_fee_props_derive_from_schedule() -> None:
    default_config = VenueConfig()
    assert default_config.maker_fee == BINANCE_MAKER_FEE
    assert default_config.taker_fee == BINANCE_TAKER_FEE
    assert default_config.fee_schedule.is_default

    bnb_config = VenueConfig(fee_schedule=BNB)
    assert bnb_config.maker_fee == Decimal("0.00018")
    assert bnb_config.taker_fee == Decimal("0.00045")
    assert not bnb_config.fee_schedule.is_default


def test_non_binance_venue_with_schedule_raises() -> None:
    with pytest.raises(ValueError, match="BINANCE"):
        create_backtest_venue_config(
            VenueConfig(name="ETHEREAL", fee_schedule=FeeSchedule("VIP0", True))
        )
    # A non-default schedule is the only thing guarded: the default still builds.
    assert create_backtest_venue_config(VenueConfig(name="ETHEREAL")) is not None


# ---------------------------------------------------------------------------
# Open-position exit-fee estimate
# ---------------------------------------------------------------------------


def test_open_mark_uses_schedule_taker() -> None:
    engine = _engine(MakerTakerFeeModel())
    engine.add_data([_bar(p, i) for i, p in enumerate([10000.0, 10100.0, 10200.0, 10300.0])])
    engine.add_strategy(_OpenAtEnd(_Cfg(), last_bar=3))
    engine.run()
    try:
        open_positions = [p for p in all_positions(engine) if not p.is_closed]
        assert len(open_positions) == 1
        last_close = 10300.0

        scheduled = mark_open_positions(
            engine, open_positions, execution_timeframe="1m", taker_fee=BNB.taker_rate
        )
        assert len(scheduled) == 1
        assert scheduled[0].exit_fee == pytest.approx(float(_QTY) * last_close * 0.00045)

        # Default (taker_fee omitted) still reads the instrument's taker fee.
        default = mark_open_positions(engine, open_positions, execution_timeframe="1m")
        assert len(default) == 1
        assert default[0].exit_fee == pytest.approx(float(_QTY) * last_close * 0.0005)
    finally:
        _dispose(engine)


def test_open_mark_fee_fallback_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A failing instrument lookup must WARN and still charge the 0.0005 fallback.

    The estimated exit fee is a reported number: silently swapping in 0.0005 hid
    the fact that the instrument's taker rate was never read. The value is
    unchanged (exactness); only the log is new.
    """
    import logging
    from types import SimpleNamespace

    class _BoomCache:
        def bars(self, bar_type: object) -> list[object]:
            return [
                SimpleNamespace(
                    ts_init=_MIN_NS, open=10000.0, high=10005.0, low=9995.0, close=10300.0
                )
            ]

        def instrument(self, instrument_id: object) -> object:
            raise KeyError(str(instrument_id))

    engine = SimpleNamespace(kernel=SimpleNamespace(cache=_BoomCache()))
    pos = SimpleNamespace(instrument_id=IID, signed_qty=_QTY, avg_px_open=10000.0)

    with caplog.at_level(logging.WARNING, logger="vibe_quant.validation.extraction"):
        marks = mark_open_positions(engine, [pos], execution_timeframe="1m", log=False)

    assert len(marks) == 1
    assert marks[0].exit_fee == pytest.approx(abs(float(_QTY)) * 10300.0 * 0.0005)
    assert marks[0].exit_fee == 0.2575
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "no taker rate for" in m and str(IID) in m and "KeyError" in m for m in messages
    ), messages


# ---------------------------------------------------------------------------
# FeeSchedule
# ---------------------------------------------------------------------------


def test_unknown_tier_raises() -> None:
    with pytest.raises(ValueError, match="tier"):
        FeeSchedule.from_params({"fee_tier": "VIP9"})
    with pytest.raises(ValueError, match="bnb_discount"):
        FeeSchedule.from_params({"bnb_discount": "yes"})


def test_default_schedule_round_trips_and_rates() -> None:
    assert FeeSchedule.from_params({}) == FeeSchedule()
    assert FeeSchedule.from_params(FeeSchedule.from_params({"fee_tier": "VIP0"}).to_notes()) == VIP0
    assert VIP0.maker_rate == Decimal("0.0002")
    assert VIP0.taker_rate == Decimal("0.0005")
    assert BNB.maker_rate == Decimal("0.00018")
    assert BNB.taker_rate == Decimal("0.00045")
