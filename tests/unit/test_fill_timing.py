"""Look-ahead guard: screening/discovery must not fill entries on the signal bar.

The classic vibe-coded backtest bug is acting on bar ``t``'s own close (an unlagged
signal): you "earn" a move you could only have captured by trading before the bar closed.
vibe-quant runs on NautilusTrader, an event-driven engine, but that alone does NOT prevent
the bug -- it depends on *when* the order is submitted relative to the bar.

Two layers are proven here:

1. **Engine reality (why deferral is needed).** With ``bar_execution=True`` NT fills a
   market order at the *bar's close*, never its open. With no latency, an order submitted in
   ``on_bar(t)`` therefore fills at ``close[t]`` -- the very bar that produced the signal.
   That is same-bar look-ahead. Adding latency defers the fill to ``close[t+1]``.

2. **The screening fix (the actual guard).** Screening/discovery share one code path
   (``NTScreeningRunner`` -> ``StrategyCompiler`` -> NautilusTrader) and run with **no
   latency**, so without intervention every champion's entries would fill at ``close[t]``.
   The runner sets ``execution_delay_probability=1.0`` so the compiled strategy defers every
   entry/exit by one bar; the fill then lands at ``close[t+1]``, matching the validation tier.
   We compile a trivial strategy and prove that with the knob OFF (0.0) it fills at the signal
   bar's close (the look-ahead), and with it ON (1.0, screening's setting) it fills one bar
   later. A regression that drops the knob would flip this test red.

A controlled bar series with a large gap between ``close[t]`` and ``close[t+1]`` makes the
source bar of each fill unambiguous. ``prob_slippage=0`` keeps the engine-level fills exact.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import FillModel, LatencyModel
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.objects import Currency, Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

from vibe_quant.data.catalog import create_instrument, get_bar_type
from vibe_quant.dsl.compiler import StrategyCompiler, _to_class_name
from vibe_quant.dsl.parser import validate_strategy_dict

# Signal fires on bar 0. The look-ahead fill is bar 0's close; the honest (deferred) fill is
# bar 1's close. The ~1000 gap (>> 1 tick of 0.1) makes the source bar unambiguous.
SIGNAL_BAR_CLOSE = 10_000.0
NEXT_BAR_OPEN = 11_000.0
NEXT_BAR_CLOSE = 11_020.0

_INSTRUMENT_ID = "BTCUSDT-PERP.BINANCE"


def _make_bar(bar_type: BarType, o: float, h: float, lo: float, c: float, minute: int) -> Bar:
    ts_event = minute * 60_000 * 1_000_000  # ms -> ns
    ts_init = (minute * 60_000 + 59_999) * 1_000_000
    return Bar(
        bar_type=bar_type,
        open=Price.from_str(f"{o:.1f}"),
        high=Price.from_str(f"{h:.1f}"),
        low=Price.from_str(f"{lo:.1f}"),
        close=Price.from_str(f"{c:.1f}"),
        volume=Quantity.from_str("100.000"),
        ts_event=ts_event,
        ts_init=ts_init,
    )


def _build_bars(bar_type: BarType) -> list[Bar]:
    # bar 0 is the signal bar; bar 1 is the earliest honest fill.
    return [
        _make_bar(bar_type, 10_000.0, 10_010.0, 9_990.0, SIGNAL_BAR_CLOSE, minute=0),
        _make_bar(bar_type, NEXT_BAR_OPEN, 11_050.0, 10_990.0, NEXT_BAR_CLOSE, minute=1),
        _make_bar(bar_type, 12_000.0, 12_050.0, 11_990.0, 12_000.0, minute=2),
        _make_bar(bar_type, 13_000.0, 13_050.0, 12_990.0, 13_000.0, minute=3),
    ]


def _new_engine() -> BacktestEngine:
    engine = BacktestEngine(
        config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"))
    )
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(1_000, Currency.from_str("USDT"))],
        default_leverage=Decimal("10"),
        # Screening fill model, but slippage off so fills land exactly on the touched price.
        fill_model=FillModel(prob_fill_on_limit=0.8, prob_slippage=0.0),
        latency_model=None,
        bar_execution=True,  # matches create_backtest_venue_config()
    )
    engine.add_instrument(create_instrument("BTCUSDT"))
    return engine


# ---------------------------------------------------------------------------
# Layer 1 -- engine reality: NT fills market orders at the bar CLOSE.
# ---------------------------------------------------------------------------


class _RawConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    order_side: OrderSide
    quantity: str


class _RawMarketOrderStrategy(Strategy):
    """Submits one raw market order on the first bar; records the entry fill price."""

    def __init__(self, config: _RawConfig) -> None:
        super().__init__(config)
        self.submitted = False
        self.signal_bar_close: float | None = None
        self.fill_px: float | None = None
        self.fill_ts: int | None = None

    def on_start(self) -> None:
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        if self.submitted:
            return
        self.signal_bar_close = float(bar.close)
        instrument = self.cache.instrument(self.config.instrument_id)
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=self.config.order_side,
            quantity=instrument.make_qty(Decimal(self.config.quantity)),
        )
        self.submit_order(order)
        self.submitted = True

    def on_order_filled(self, event: object) -> None:
        if self.fill_px is None:
            self.fill_px = float(event.last_px)  # type: ignore[attr-defined]
            self.fill_ts = int(event.ts_event)  # type: ignore[attr-defined]


def _run_raw(
    order_side: OrderSide, latency_model: LatencyModel | None
) -> _RawMarketOrderStrategy:
    bar_type = get_bar_type("BTCUSDT", "1m")
    engine = _new_engine()
    if latency_model is not None:
        # Rebuild the venue with latency (add_venue already ran in _new_engine without it).
        engine.reset()
        engine.dispose()
        engine = BacktestEngine(
            config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"))
        )
        engine.add_venue(
            venue=Venue("BINANCE"),
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            starting_balances=[Money(1_000, Currency.from_str("USDT"))],
            default_leverage=Decimal("10"),
            fill_model=FillModel(prob_fill_on_limit=0.8, prob_slippage=0.0),
            latency_model=latency_model,
            bar_execution=True,
        )
        engine.add_instrument(create_instrument("BTCUSDT"))

    engine.add_data(_build_bars(bar_type))
    config = _RawConfig(
        instrument_id=InstrumentId(Symbol("BTCUSDT-PERP"), Venue("BINANCE")),
        bar_type=bar_type,
        order_side=order_side,
        quantity="0.010",
    )
    strategy = _RawMarketOrderStrategy(config=config)
    engine.add_strategy(strategy)
    try:
        engine.run()
    finally:
        engine.reset()
        engine.dispose()
    return strategy


def test_engine_fills_market_order_at_signal_bar_close_without_latency() -> None:
    """No latency: a market order from on_bar(t) fills at bar t's CLOSE (same-bar).

    This documents the engine behavior the screening deferral exists to neutralize. If NT ever
    starts filling at the next bar on its own, the strategy-level guard below would still hold,
    but this expectation should be revisited.
    """
    strategy = _run_raw(OrderSide.BUY, latency_model=None)
    assert strategy.fill_px is not None, "order never filled"
    assert strategy.signal_bar_close == pytest.approx(SIGNAL_BAR_CLOSE)
    # Fills on the SIGNAL bar's close -- the same-bar look-ahead.
    assert strategy.fill_px == pytest.approx(SIGNAL_BAR_CLOSE, abs=1.0)


def test_engine_latency_defers_fill_to_next_bar_close() -> None:
    """CLOUD latency defers the fill to the NEXT bar -- and NT fills at that bar's CLOSE.

    Note this is close[t+1] (11020), not open[t+1] (11000): NT cannot fill a market order at a
    bar's open. SPEC.md describing "next bar open" is aspirational; the realized price is the
    next bar's close, which is the conservative (honest) direction.
    """
    latency = LatencyModel(base_latency_nanos=60_000_000)  # CLOUD preset (60ms)
    strategy = _run_raw(OrderSide.BUY, latency_model=latency)
    assert strategy.fill_px is not None, "order never filled"
    assert abs(strategy.fill_px - SIGNAL_BAR_CLOSE) > 500.0, "latency must not fill same-bar"
    assert strategy.fill_px == pytest.approx(NEXT_BAR_CLOSE, abs=1.0)


# ---------------------------------------------------------------------------
# Layer 2 -- the screening fix: compiled strategy defers entries one bar.
# ---------------------------------------------------------------------------


def _entry_fill_price(*, long: bool, execution_delay_probability: float) -> float | None:
    """Compile a trivial always-enter strategy, run it, return the entry fill price.

    ``execution_delay_probability`` mirrors what NTScreeningRunner injects: screening sets 1.0
    (always defer one bar). The entry condition is trivially true and there are no indicators,
    so the signal fires on bar 0; the only thing that moves the fill is the delay knob.
    """
    direction = "long" if long else "short"
    cond = "close > 1" if long else "close < 99999999"
    dsl_dict = {
        "name": f"fill_probe_{direction}",
        "timeframe": "1m",
        "indicators": {},
        "entry_conditions": {direction: [cond]},
        "exit_conditions": {},
        # Far-away SL/TP (max allowed) so neither triggers across the rising series.
        "stop_loss": {"type": "fixed_pct", "percent": 50.0},
        "take_profit": {"type": "fixed_pct", "percent": 50.0},
    }
    dsl = validate_strategy_dict(dsl_dict)
    module = StrategyCompiler().compile_to_module(dsl)
    camel = _to_class_name(dsl.name)
    strategy_cls = getattr(module, f"{camel}Strategy")
    config_cls = getattr(module, f"{camel}Config")

    config = config_cls(
        instrument_id=_INSTRUMENT_ID,
        execution_delay_probability=execution_delay_probability,
    )
    strategy = strategy_cls(config=config)

    bar_type = get_bar_type("BTCUSDT", "1m")
    engine = _new_engine()
    engine.add_data(_build_bars(bar_type))
    engine.add_strategy(strategy)
    try:
        engine.run()
        positions = list(engine.cache.positions()) + list(engine.cache.position_snapshots())
        if not positions:
            return None
        return float(positions[0].avg_px_open)
    finally:
        engine.reset()
        engine.dispose()


@pytest.mark.parametrize("long", [True, False], ids=["long", "short"])
def test_screening_deferral_moves_entry_off_the_signal_bar(long: bool) -> None:
    """Screening's execution_delay_probability=1.0 fills entries at close[t+1], not close[t].

    With the knob OFF the compiled strategy reproduces the same-bar look-ahead (fill at the
    signal bar's close); with it ON (screening's setting) the fill defers one bar. This is the
    decisive regression guard for the discovery pipeline -- every champion's metrics depend on
    entries NOT being filled on the bar that generated the signal.
    """
    # Knob OFF -> same-bar look-ahead (the bug the fix removes).
    leaky = _entry_fill_price(long=long, execution_delay_probability=0.0)
    assert leaky is not None, "strategy never entered (knob off)"
    assert leaky == pytest.approx(SIGNAL_BAR_CLOSE, abs=1.0), (
        f"expected same-bar fill {SIGNAL_BAR_CLOSE} with delay off, got {leaky}"
    )

    # Knob ON (screening default) -> deferred one bar, honest fill.
    honest = _entry_fill_price(long=long, execution_delay_probability=1.0)
    assert honest is not None, "strategy never entered (knob on)"
    assert honest == pytest.approx(NEXT_BAR_CLOSE, abs=1.0), (
        f"expected deferred fill {NEXT_BAR_CLOSE} with delay on, got {honest}"
    )
    assert abs(honest - SIGNAL_BAR_CLOSE) > 500.0, (
        f"deferred fill {honest} still on the signal bar close {SIGNAL_BAR_CLOSE} "
        "-> screening look-ahead not fixed"
    )
