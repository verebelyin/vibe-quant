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

3. **Own-datum release (yul7u.9-.11).** Coarser-than-1m screening and validation do not use a
   venue-wide LatencyModel: the generated strategy queues every submit/cancel in a
   per-instrument outbox and releases it on its OWN next fill tick (screening, 1m closes at
   boundary T and T+1m) or its OWN 1m detail bar (validation). A venue-wide release let symbol
   A's datum send symbol B's order at B's stale book, i.e. at B's signal-bar close. The
   look-ahead / multi-symbol guards at the bottom use DISTINCT prices per minute (BTC
   10000 + 100*minute, ETH 20000 + 100*minute) so every wrong fill is unambiguous.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import FillModel, LatencyModel
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, OrderType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.objects import Currency, Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

from vibe_quant.data.catalog import create_instrument, get_bar_type
from vibe_quant.data.fill_ticks import build_fill_ticks
from vibe_quant.dsl.compiler import StrategyCompiler, _to_class_name
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.nt_compat import retain_log_guard
from vibe_quant.screening.nt_runner import screening_fill_mode

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


def _new_engine(symbols: tuple[str, ...] = ("BTCUSDT",)) -> BacktestEngine:
    engine = BacktestEngine(
        config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"))
    )
    # NT 1.223+: dispose() tears down logging but the Rust logger can only be
    # set once per process — retain the guard or the next engine hard-aborts.
    retain_log_guard(engine)
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(10_000, Currency.from_str("USDT"))],
        default_leverage=Decimal("10"),
        # Screening fill model, but slippage off so fills land exactly on the touched price.
        fill_model=FillModel(prob_fill_on_limit=0.8, prob_slippage=0.0),
        latency_model=None,
        bar_execution=True,  # matches create_backtest_venue_config()
    )
    for sym in symbols:
        engine.add_instrument(create_instrument(sym))
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
        retain_log_guard(engine)
        engine.add_venue(
            venue=Venue("BINANCE"),
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            starting_balances=[Money(10_000, Currency.from_str("USDT"))],
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


# ---------------------------------------------------------------------------
# Layer 3 -- own-datum release: the outbox fills on the strategy's OWN fill tick / 1m bar.
# ---------------------------------------------------------------------------

_S = 1_000_000_000
_M = 60 * _S
_M5 = 5 * _M
_MS = 1_000_000
_TICK1_A, _TICK1_B = 10_500.0, 20_500.0  # close of the first 1m bar after the signal
_SIGNAL_CLOSE = {"BTCUSDT": 10_000.0, "ETHUSDT": 20_000.0}
_BASE = {"BTCUSDT": 10_000.0, "ETHUSDT": 20_000.0}
_NEXT_BAR_CLOSE = {"BTCUSDT": 11_020.0, "ETHUSDT": 21_020.0}
_PREC = {"BTCUSDT": 1, "ETHUSDT": 2}
_N_BARS = 5


def _iid(sym: str) -> str:
    return f"{sym[:-4]}USDT-PERP.BINANCE"


def _px(sym: str, v: float) -> Price:
    return Price.from_str(f"{v:.{_PREC[sym]}f}")


def _minute_close(sym: str, minute: int) -> float:
    return _BASE[sym] + 100.0 * minute  # minute 5 -> tick1, minute 6 -> tick2


def _bars_5m(
    sym: str, *, bar1: tuple[float, float, float, float] | None = None
) -> list[Bar]:
    """5m strategy bars. Bar 0 (signal) closes at the signal close; bar 1 at NEXT_BAR_CLOSE.

    ts_init = close - 1ms, like the catalog (the minute-4 1m bar shares the signal's ts_init).
    """
    bt = BarType.from_str(f"{_iid(sym)}-5-MINUTE-LAST-EXTERNAL")
    base = _BASE[sym]
    sig = _SIGNAL_CLOSE[sym]
    nxt = _NEXT_BAR_CLOSE[sym]
    ohlc = [
        (sig, sig + 10, sig - 10, sig),
        bar1 or (nxt - 20, nxt + 30, nxt - 30, nxt),
        (base + 2000, base + 2050, base + 1990, base + 2000),
        (base + 3000, base + 3050, base + 2990, base + 3000),
        (base + 4000, base + 4050, base + 3990, base + 4000),
    ]
    return [
        Bar(
            bar_type=bt,
            open=_px(sym, o),
            high=_px(sym, h),
            low=_px(sym, lo),
            close=_px(sym, c),
            volume=Quantity.from_str("100.000"),
            ts_event=i * _M5,
            ts_init=(i + 1) * _M5 - _MS,
        )
        for i, (o, h, lo, c) in enumerate(ohlc)
    ]


def _bars_1m(sym: str, minutes: list[int]) -> list[Bar]:
    bt = BarType.from_str(f"{_iid(sym)}-1-MINUTE-LAST-EXTERNAL")
    out = []
    for m in minutes:
        c = _px(sym, _minute_close(sym, m))
        out.append(
            Bar(
                bar_type=bt,
                open=c,
                high=c,
                low=c,
                close=c,
                volume=Quantity.from_str("100.000"),
                ts_event=m * _M,
                ts_init=(m + 1) * _M - _MS,
            )
        )
    return out


def _fill_ticks(sym: str, minutes: list[int]) -> list[Any]:
    """Production build_fill_ticks on synthetic 1m closes (boundary + 1m ticks only)."""
    series = SimpleNamespace(
        ts=np.array([(m + 1) * _M - _MS for m in minutes], dtype=np.int64),
        close=np.array([_minute_close(sym, m) for m in minutes]),
    )
    ticks, _gaps = build_fill_ticks(series, "5m", create_instrument(sym))
    return ticks


def _compile(*, long: bool, timeframe: str = "5m", sl_pct: float = 50.0) -> tuple[type, type]:
    direction = "long" if long else "short"
    d: dict[str, object] = {
        "name": f"own_release_{direction}_{timeframe}_{int(sl_pct)}",
        "timeframe": timeframe,
        "indicators": {},
        "entry_conditions": {direction: ["close > 1" if long else "close < 99999999"]},
        "exit_conditions": {},
        "stop_loss": {"type": "fixed_pct", "percent": sl_pct},
        "take_profit": {"type": "fixed_pct", "percent": 50.0},
    }
    dsl = validate_strategy_dict(d)
    module = StrategyCompiler().compile_to_module(dsl)
    camel = _to_class_name(dsl.name)
    return getattr(module, f"{camel}Strategy"), getattr(module, f"{camel}Config")


def _cross_subscribing(strategy_cls: type, others: list[str], release: str) -> type:
    """Strategy that ALSO receives the other symbols' ticks / 1m bars (the old shared-venue view).

    NT only delivers subscribed data, so without this a handler that forgot its own-instrument
    check could never be seen misbehaving. With it, a foreign datum reaches the handler and
    must not release this strategy's queue.
    """

    class Cross(strategy_cls):  # type: ignore[misc, valid-type]
        def on_start(self) -> None:
            super().on_start()
            for sym in others:
                if release == "trade_tick":
                    self.subscribe_trade_ticks(InstrumentId.from_str(_iid(sym)))
                elif release == "bar":
                    self.subscribe_bars(BarType.from_str(f"{_iid(sym)}-1-MINUTE-LAST-EXTERNAL"))

    return Cross


def _run_release(
    symbols: tuple[str, ...],
    data: list[Any],
    *,
    long: bool = True,
    sl_pct: float = 50.0,
    timeframe: str = "5m",
    mode: tuple[str, float] | None = None,
    detail_bar_type: str = "1-MINUTE",
    cross: bool = True,
) -> BacktestEngine:
    """Run one compiled strategy per symbol with the screening-derived release knobs.

    ``mode`` defaults to the runner's own policy (``screening_fill_mode``) for ``timeframe``
    (``trade_tick`` unless 1m bars are loaded); pass ``("bar", 0.0)`` for validation.
    """
    strategy_cls, config_cls = _compile(long=long, timeframe=timeframe, sl_pct=sl_pct)
    if mode is None:
        fm = screening_fill_mode(timeframe, [])
        mode = (fm.command_release, fm.execution_delay_probability)
    release, delay = mode
    engine = _new_engine(symbols)
    engine.add_data(data)
    for sym in symbols:
        cls = (
            _cross_subscribing(strategy_cls, [o for o in symbols if o != sym], release)
            if cross
            else strategy_cls
        )
        kw: dict[str, Any] = {
            "instrument_id": _iid(sym),
            "execution_delay_probability": delay,
        }
        if release:
            kw["command_release"] = release
        if release == "bar":
            kw["command_release_bar_type"] = f"{_iid(sym)}-{detail_bar_type}-LAST-EXTERNAL"
        engine.add_strategy(cls(config=config_cls(**kw)))
    engine.run()
    return engine


def _entries(engine: BacktestEngine, sym: str) -> list[float]:
    return [
        float(o.avg_px)
        for o in engine.cache.orders()
        if str(o.instrument_id) == _iid(sym) and not o.is_reduce_only and o.filled_qty
    ]


def _dispose(engine: BacktestEngine) -> None:
    engine.reset()
    engine.dispose()


@pytest.mark.parametrize("long", [True, False], ids=["long", "short"])
def test_own_tick_fills_at_first_minute_close(long: bool) -> None:
    """Coarse screening fills at the close of the first 1m bar after the signal, per side."""
    minutes = list(range(0, _N_BARS * 5))
    data = _bars_5m("BTCUSDT") + _fill_ticks("BTCUSDT", minutes)
    engine = _run_release(("BTCUSDT",), data, long=long)
    try:
        fills = _entries(engine, "BTCUSDT")
        assert fills == [pytest.approx(_TICK1_A)], fills
        assert fills[0] != pytest.approx(SIGNAL_BAR_CLOSE)
        assert fills[0] != pytest.approx(NEXT_BAR_CLOSE)
        # on_stop sends nothing in outbox mode: the position stays open, its SL/TP stay live
        # (an end-of-run close/cancel would fill/cancel them).
        assert len(engine.cache.positions_open()) == 1
        protective = [o for o in engine.cache.orders() if o.is_reduce_only]
        assert len(protective) == 2 and all(o.is_open for o in protective), protective
    finally:
        _dispose(engine)


@pytest.mark.parametrize("cross", [True, False], ids=["cross_subscribed", "plain"])
@pytest.mark.parametrize("a_first", [True, False], ids=["btc_first", "eth_first"])
@pytest.mark.parametrize("long", [True, False], ids=["long", "short"])
def test_two_symbols_each_fill_on_own_tick1(a_first: bool, long: bool, cross: bool) -> None:
    """Each symbol fills at ITS OWN tick1, whichever symbol's data/strategy comes first.

    A venue-wide release lets one symbol's datum send the other's order at the other's
    stale book (= its signal-bar close). Zero fills may land on a signal-bar close.
    """
    minutes = list(range(0, _N_BARS * 5))
    a, b = ("BTCUSDT", "ETHUSDT") if a_first else ("ETHUSDT", "BTCUSDT")
    data = (
        _bars_5m(a) + _bars_5m(b) + _fill_ticks(a, minutes) + _fill_ticks(b, minutes)
    )
    engine = _run_release((a, b), data, long=long, cross=cross)
    try:
        assert _entries(engine, "BTCUSDT") == [pytest.approx(_TICK1_A)]
        assert _entries(engine, "ETHUSDT") == [pytest.approx(_TICK1_B)]
        every = _entries(engine, "BTCUSDT") + _entries(engine, "ETHUSDT")
        assert not {round(f) for f in every} & {10_000, 20_000}, every
    finally:
        _dispose(engine)


@pytest.mark.parametrize(
    ("missing", "expected"),
    [
        ([5], 10_600.0),  # minute-0 tick gone -> tick2 (minute 6 close)
        ([5, 6], 11_000.0),  # both gone -> next boundary's tick1 (minute 10 close)
    ],
    ids=["tick1_missing", "both_missing"],
)
def test_missing_own_tick_waits_never_signal_close(missing: list[int], expected: float) -> None:
    """A gap in the boundary ticks makes the queue wait for the next OWN tick."""
    minutes = [m for m in range(0, _N_BARS * 5) if m not in missing]
    data = _bars_5m("BTCUSDT") + _fill_ticks("BTCUSDT", minutes)
    engine = _run_release(("BTCUSDT",), data)
    try:
        fills = _entries(engine, "BTCUSDT")
        assert fills == [pytest.approx(expected)], fills
        assert fills[0] != pytest.approx(SIGNAL_BAR_CLOSE)
    finally:
        _dispose(engine)


def test_missing_own_ticks_everywhere_never_fills() -> None:
    """With no own ticks at all the order stays queued: no fill, certainly not at 10000."""
    engine = _run_release(("BTCUSDT",), _bars_5m("BTCUSDT") + _fill_ticks("BTCUSDT", []))
    try:
        assert _entries(engine, "BTCUSDT") == []
    finally:
        _dispose(engine)


def test_brackets_live_after_own_tick2() -> None:
    """Entry fills on tick1; SL/TP are queued on the fill and live from tick2.

    Bar t+1's low pierces the stop: the exit must fill AT the stop price (not at the bar close
    and not never -- which is what a bracket released only at the next boundary would give).
    """
    stop_bar = (10_600.0, 10_700.0, 9_900.0, 10_050.0)  # low 9900 < stop 9975
    minutes = list(range(0, _N_BARS * 5))
    data = _bars_5m("BTCUSDT", bar1=stop_bar) + _fill_ticks("BTCUSDT", minutes)
    engine = _run_release(("BTCUSDT",), data, sl_pct=5.0)
    try:
        # the always-true entry re-enters after the stop-out; only the first cycle matters
        assert _entries(engine, "BTCUSDT")[0] == pytest.approx(_TICK1_A)
        stops = [
            o
            for o in engine.cache.orders()
            if o.is_reduce_only and o.order_type == OrderType.STOP_MARKET
        ]
        stops.sort(key=lambda o: o.ts_init)
        assert float(stops[0].trigger_price) == pytest.approx(_TICK1_A * 0.95, abs=0.2)
        assert stops[0].is_closed and stops[0].filled_qty > 0, "stop never filled"
        # sent on tick2 (minute 6 close), NOT at the tick1 fill: events[0]=Initialized (creation
        # = the tick1 fill), events[1]=Submitted (when the command actually left the outbox)
        assert stops[0].events[1].ts_event == 7 * _M - _MS, stops[0].events
        assert float(stops[0].avg_px) == pytest.approx(float(stops[0].trigger_price), abs=0.2)
    finally:
        _dispose(engine)


@pytest.mark.parametrize("a_first", [True, False], ids=["btc_first", "eth_first"])
def test_validation_mode_releases_on_own_1m_bar(a_first: bool) -> None:
    """Validation (``bar`` release): each symbol fills at ITS OWN first 1m bar after the signal.

    The minute-4 bar shares the signal's ts_init (must not release: strict >), so its distinct
    close (10400/20400) is a trap for ``>=``.
    """
    minutes = list(range(0, _N_BARS * 5))
    a, b = ("BTCUSDT", "ETHUSDT") if a_first else ("ETHUSDT", "BTCUSDT")
    data = (
        _bars_5m(a) + _bars_5m(b) + _bars_1m(a, minutes) + _bars_1m(b, minutes)
    )
    engine = _run_release((a, b), data, mode=("bar", 0.0))
    try:
        assert _entries(engine, "BTCUSDT") == [pytest.approx(_TICK1_A)]
        assert _entries(engine, "ETHUSDT") == [pytest.approx(_TICK1_B)]
    finally:
        _dispose(engine)


@pytest.mark.parametrize("long", [True, False], ids=["long", "short"])
def test_one_minute_strategy_keeps_deferral(long: bool) -> None:
    """A 1m strategy takes no outbox: screening defers one bar (delay 1.0) -> next 1m close."""
    fm = screening_fill_mode("1m", [])
    assert (fm.command_release, fm.execution_delay_probability, fm.tick_timeframe) == ("", 1.0, None)
    bt = get_bar_type("BTCUSDT", "1m")
    engine = _new_engine()
    engine.add_data(_build_bars(bt))
    strategy_cls, config_cls = _compile(long=long, timeframe="1m")
    engine.add_strategy(
        strategy_cls(
            config=config_cls(
                instrument_id=_INSTRUMENT_ID,
                execution_delay_probability=fm.execution_delay_probability,
            )
        )
    )
    try:
        engine.run()
        fills = _entries(engine, "BTCUSDT")
        assert fills == [pytest.approx(NEXT_BAR_CLOSE, abs=1.0)], fills
    finally:
        _dispose(engine)
