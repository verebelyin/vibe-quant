"""``signal_level`` stops and targets (design-v2 section 3.2, semantics rows R7, R9-R16).

The level source is DONCHIAN (``dc.lower`` / ``dc.upper``: the lowest low / highest high of the last
two bars, the current bar included), so every level is a literal read off the bars below. Real
NautilusTrader events on a raw ``BacktestEngine`` with 1m bars and no outbox (``immediate`` mode)
or the screening outbox on production fill ticks (``tick`` mode).

Account 10 000 USDT, risk 2 % per trade (200 USDT), BTCUSDT-PERP tick 0.1, size step 0.001.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any, NamedTuple

import pytest
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import FillModel, LatencyModel
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, TimeInForce
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Currency, Money, Price, Quantity
from pydantic import ValidationError

from tests.unit.test_position_mode_reverse import _fill_ticks
from vibe_quant.data.catalog import create_instrument
from vibe_quant.dsl.compiler import StrategyCompiler, _to_class_name
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.dsl.schema import StrategyDSL
from vibe_quant.nt_compat import retain_log_guard

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

IID = "BTCUSDT-PERP.BINANCE"
_M = 60_000_000_000
_MS = 1_000_000
BIG = "100000.000"


class Row(NamedTuple):
    side: str
    reduce_only: bool
    type: str
    status: str
    qty: float
    px: float
    trigger: float


class Out(NamedTuple):
    orders: list[Row]
    open_orders: list[Row]
    open_positions: list[tuple[str, float]]
    strategy: Any


_engines: list[BacktestEngine] = []


@pytest.fixture(autouse=True)
def _dispose_engines() -> Iterator[None]:
    yield
    while _engines:
        engine = _engines.pop()
        engine.reset()
        engine.dispose()


class B(NamedTuple):
    """One 1m bar: open, high, low, close, volume."""

    o: float
    h: float
    lo: float
    c: float
    v: str = BIG


def _bars(rows: list[B], tf: str = "1m") -> list[Bar]:
    step, spec = (5 * _M, "5-MINUTE") if tf == "5m" else (_M, "1-MINUTE")
    bt = BarType.from_str(f"{IID}-{spec}-LAST-EXTERNAL")
    return [
        Bar(
            bar_type=bt,
            open=Price.from_str(f"{r.o:.1f}"),
            high=Price.from_str(f"{r.h:.1f}"),
            low=Price.from_str(f"{r.lo:.1f}"),
            close=Price.from_str(f"{r.c:.1f}"),
            volume=Quantity.from_str(r.v),
            ts_event=i * step,
            ts_init=(i + 1) * step - _MS,
        )
        for i, r in enumerate(rows)
    ]


def dsl(name: str = "sl_unit", **over: Any) -> dict[str, Any]:
    """Long on ``close > 10000`` with dc.lower as stop and dc.upper as target (short: no signal)."""
    d: dict[str, Any] = {
        "name": name,
        "timeframe": "1m",
        "indicators": {"dc": {"type": "DONCHIAN", "period": 2}},
        "entry_conditions": {"long": ["close > 10000"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
        "stop_loss_long": {"type": "signal_level", "indicator": "dc.lower"},
        "take_profit_long": {"type": "signal_level", "indicator": "dc.upper"},
    }
    d.update(over)
    return d


def compile_dsl(d: dict[str, Any]) -> tuple[type, type]:
    module = StrategyCompiler().compile_to_module(validate_strategy_dict(d))
    cn = _to_class_name(d["name"])
    return getattr(module, f"{cn}Strategy"), getattr(module, f"{cn}Config")


def _row(o: Any) -> Row:
    filled = float(o.filled_qty) > 0
    return Row(
        side=o.side.name,
        reduce_only=bool(o.is_reduce_only),
        type=o.order_type.name,
        status=o.status.name,
        qty=float(o.filled_qty) or float(o.quantity),
        px=float(o.avg_px) if filled else (float(o.price) if getattr(o, "price", None) is not None else 0.0),
        trigger=float(o.trigger_price) if getattr(o, "trigger_price", None) is not None else 0.0,
    )


def run(
    rows: list[B],
    *,
    d: dict[str, Any] | None = None,
    wrap: Callable[[type], type] | None = None,
    balance: int = 10_000,
    tf: str = "1m",
    mode: str = "immediate",
    latency: LatencyModel | None = None,
    delay: float = 0.0,
    **cfg: Any,
) -> Out:
    """``immediate``: bars only, commands sent at once (venue latency, if any, is the only delay);
    ``tick``: 5m bars plus the production 1m fill ticks, commands released by the outbox."""
    d = d or dsl()
    strategy_cls, config_cls = compile_dsl(d)
    if wrap is not None:
        strategy_cls = wrap(strategy_cls)
    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    retain_log_guard(engine)
    _engines.append(engine)
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(balance, Currency.from_str("USDT"))],
        default_leverage=Decimal("10"),
        fill_model=FillModel(prob_fill_on_limit=1.0, prob_slippage=0.0),
        latency_model=latency,
        bar_execution=True,
        use_reduce_only=True,
    )
    engine.add_instrument(create_instrument("BTCUSDT"))
    if mode == "tick":
        engine.add_data(_bars(rows, "5m") + _fill_ticks([r.c for r in rows]))
        cfg = {"command_release": "trade_tick", **cfg}
    else:
        engine.add_data(_bars(rows, tf))
    strategy = strategy_cls(config=config_cls(instrument_id=IID, execution_delay_probability=delay, **cfg))
    engine.add_strategy(strategy)
    engine.run()
    cache = engine.cache
    return Out(
        orders=[_row(o) for o in cache.orders()],
        open_orders=[_row(o) for o in cache.orders_open()],
        open_positions=[(p.side.name, float(p.quantity)) for p in cache.positions_open()],
        strategy=strategy,
    )


# ---------------------------------------------------------------------------
# R12: a partial fill keeps the snapshot prices, also when the pair is re-armed from scratch
# ---------------------------------------------------------------------------

# b0/b1 give dc.lower 9 500 and dc.upper 11 000 at the signal bar b1 (close 10 250). b2's volume
# is 0.1 BTC. From b3 on the Donchian window has moved to
# 10 200 / 10 300: a re-arm that recomputed the levels would not land on 9 500 / 11 000.
# (Without an outbox a market order sent on the signal bar fills against that bar's own book, and the
# venue fills a quarter of the bar volume per order: 0.400 BTC -> 0.1 filled of the 0.266 sized.)
_R12_BARS = [
    B(10200, 10300, 9500, 10250),
    B(10200, 11000, 10100, 10250, v="0.400"),  # signal bar: the 0.266 IOC entry fills 0.1 of it
    B(10250, 10260, 10240, 10250),
    B(10250, 10300, 10200, 10250),
    B(10250, 10300, 10200, 10250),
    B(10250, 10300, 10200, 10250),
]


def _snapshot(strategy: Any) -> tuple[list[tuple[str, float, float, float]], list[tuple[str, float]]]:
    """Live protective orders (type, qty, trigger, limit price) and open positions, mid-run."""
    orders = sorted(
        (
            o.order_type.name,
            float(o.leaves_qty),
            float(o.trigger_price) if getattr(o, "trigger_price", None) is not None else 0.0,
            float(o.price) if getattr(o, "price", None) is not None else 0.0,
        )
        for o in strategy.cache.orders_open(instrument_id=strategy.instrument_id)
        if o.is_reduce_only
    )
    positions = [(p.side.name, float(p.quantity)) for p in strategy.cache.positions_open() if p.is_open]
    return orders, positions


def probe(steps: dict[int, Callable[[Any], None]]) -> Callable[[type], type]:
    """Subclass hook: ``steps[n](strategy)`` runs right after the strategy handled its n-th primary bar."""

    def wrap(base: type) -> type:
        class Probe(base):  # type: ignore[misc, valid-type]
            def __init__(self, config: Any) -> None:
                super().__init__(config)
                self.n = 0

            def on_bar(self, bar: Bar) -> None:
                super().on_bar(bar)
                if bar.bar_type == self.primary_bar_type:
                    self.n += 1
                    if self.n in steps:
                        steps[self.n](self)

        return Probe

    return wrap


def _record(into: dict[str, Any], key: str) -> Callable[[Any], None]:
    def step(strategy: Any) -> None:
        into[key] = _snapshot(strategy)

    return step


def _cancel_protection(strategy: Any) -> None:
    for order in strategy._protective_orders():
        strategy.cancel_order(order)


def _rearm(strategy: Any) -> None:
    (pos,) = [p for p in strategy.cache.positions_open() if p.is_open]
    strategy._sync_protective_orders(pos)


def test_signal_level_snapshot_survives_partial_fill() -> None:
    seen: dict[str, Any] = {}
    steps = {
        3: _record(seen, "after_fill"),
        4: _cancel_protection,  # protection vanishes (e.g. a queued pair was dropped) ...
        5: _rearm,  # ... and the whole position is re-armed from scratch
        6: _record(seen, "after_rearm"),
    }
    run(_R12_BARS, wrap=probe(steps))

    pair = [("LIMIT", 0.1, 0.0, 11000.0), ("STOP_MARKET", 0.1, 9500.0, 0.0)]
    assert seen["after_fill"] == (pair, [("LONG", 0.1)])  # the 0.266 entry filled 0.1: pair sized to it
    assert seen["after_rearm"] == (pair, [("LONG", 0.1)])  # same prices after the from-scratch re-arm


# ---------------------------------------------------------------------------
# R9: the long snapshot -- stop, target and the sizing distance
# ---------------------------------------------------------------------------

_PAIR_9500_11000 = [("LIMIT", 0.266, 0.0, 11000.0), ("STOP_MARKET", 0.266, 9500.0, 0.0)]

_R9_BARS = [
    B(10200, 10300, 9500, 10150),
    B(10200, 11000, 10100, 10250),  # signal bar: close 10 250, dc.lower 9 500, dc.upper 11 000
    B(10250, 10260, 10240, 10250),
    B(10250, 10260, 10240, 10250),
]


def test_long_snapshot_sets_stop_target_and_the_sizing_distance() -> None:
    """R9: close 10 250, stop 9 500, target 11 000 -> distance 750, so 200 / 750 = 0.266 BTC (the
    generic 2 % stop would size 0.487 and sit at 10 045); the pair is placed at the snapshot prices."""
    seen: dict[str, Any] = {}
    d = dsl(entry_conditions={"long": ["close > 10200"]})
    out = run(_R9_BARS, d=d, wrap=probe({3: _record(seen, "live")}))

    (entry,) = [o for o in out.orders if not o.reduce_only]
    assert (entry.side, entry.qty, entry.px) == ("BUY", 0.266, 10250.0)
    assert seen["live"] == (_PAIR_9500_11000, [("LONG", 0.266)])
    assert out.strategy._sizing_skips == {}


def test_screening_outbox_places_the_pair_at_the_signal_bar_snapshot() -> None:
    """The screening tick release (entry fills a minute later) keeps the signal bar's levels."""
    rows = [
        B(10150, 10200, 9500, 10150),
        B(10150, 11000, 10100, 10250),  # signal bar (5m): dc.lower 9 500, dc.upper 11 000
        B(10250, 10300, 10200, 10250),  # the window has moved on: 10 100 / 11 000
        B(10250, 10300, 10200, 10250),
    ]
    d = dsl(timeframe="5m", entry_conditions={"long": ["close > 10200"]})
    out = run(rows, d=d, mode="tick")

    stops = [o for o in out.open_orders if o.type == "STOP_MARKET"]
    limits = [o for o in out.open_orders if o.type == "LIMIT"]
    assert [(o.trigger, o.qty) for o in stops] == [(9500.0, 0.266)]
    assert [(o.px, o.qty) for o in limits] == [(11000.0, 0.266)]


# ---------------------------------------------------------------------------
# R10: refresh_on_signal
# ---------------------------------------------------------------------------

# entry b1 (levels 9 500 / 11 000); a quiet b2; b3 is a second signal with valid levels 10 100 / 10 400;
# b4 quiet; b5 is a signal whose target (10 300) EQUALS the close -> invalid; b6 quiet.
# (A command sent on a bar is processed by the venue with the next bar, so the probes read one bar later.)
_R10_BARS = [
    B(10150, 10300, 9500, 10150),
    B(10200, 11000, 10100, 10250),
    B(10200, 10250, 10100, 10150),
    B(10200, 10400, 10120, 10300),
    B(10200, 10250, 10120, 10150),
    B(10200, 10300, 10130, 10300),
    B(10200, 10250, 10130, 10150),
]


def _refresh_dsl(**over: Any) -> dict[str, Any]:
    return dsl(entry_conditions={"long": ["close > 10200"]}, refresh_on_signal=True, **over)


def test_refresh_with_valid_levels_replaces_the_protection_and_invalid_keeps_it() -> None:
    """R10: the second signal (b3) moves the pair to 10 100 / 10 400; the third (b5) has a target equal to
    its close, so the pair stays and the refusal is counted."""
    seen: dict[str, Any] = {}
    steps = {3: _record(seen, "before"), 5: _record(seen, "refreshed"), 7: _record(seen, "after_invalid")}
    out = run(_R10_BARS, d=_refresh_dsl(), wrap=probe(steps))

    refreshed = [("LIMIT", 0.266, 0.0, 10400.0), ("STOP_MARKET", 0.266, 10100.0, 0.0)]
    assert seen["before"] == (_PAIR_9500_11000, [("LONG", 0.266)])
    assert seen["refreshed"] == (refreshed, [("LONG", 0.266)])
    assert seen["after_invalid"] == (refreshed, [("LONG", 0.266)])
    assert out.strategy._refresh_invalid == 1


def test_without_refresh_a_second_signal_keeps_the_entry_snapshot() -> None:
    """Legacy behaviour: refresh_on_signal absent -> the b3 signal changes nothing."""
    seen: dict[str, Any] = {}
    d = dsl(entry_conditions={"long": ["close > 10200"]})
    run(_R10_BARS, d=d, wrap=probe({4: _record(seen, "after_signal")}))

    assert seen["after_signal"] == (_PAIR_9500_11000, [("LONG", 0.266)])


def test_refresh_on_the_entry_bar_itself_does_nothing() -> None:
    """The signal that opens the position is not also a refresh of it (b1 only: no extra cancel/submit)."""
    out = run(_R10_BARS[:3], d=_refresh_dsl())

    assert len([o for o in out.orders if o.reduce_only and o.type == "STOP_MARKET"]) == 1


def test_a_later_refresh_supersedes_an_earlier_queued_one_in_the_outbox() -> None:
    """Two refreshes before the next release datum (screening outbox): only the second pair is ever sent
    and each old order is cancelled exactly once."""
    rows = [
        B(10150, 10200, 9500, 10150),
        B(10150, 11000, 10100, 10250),  # 5m signal bar b1: entry
        B(10250, 10300, 10200, 10250),
        B(10250, 10300, 10200, 10250),
        B(10250, 10300, 10200, 10250),
    ]

    def twice(strategy: Any) -> None:
        bar = strategy.cache.bar(strategy.primary_bar_type)
        for stop, target in ((10100.0, 10400.0), (10150.0, 10500.0)):
            strategy._get_indicator_value = lambda name, s=stop, t=target: s if name.endswith("lower") else t
            strategy._refresh_levels(bar, True)

    seen: dict[str, Any] = {}
    run(rows, d=_refresh_dsl(timeframe="5m"), mode="tick", wrap=probe({3: twice, 4: _record(seen, "live")}))

    assert seen["live"][0] == [("LIMIT", 0.266, 0.0, 10500.0), ("STOP_MARKET", 0.266, 10150.0, 0.0)]


# ---------------------------------------------------------------------------
# R11: an invalid level skips the entry -- never the 2 % fallback
# ---------------------------------------------------------------------------

_NAN, _INF = float("nan"), float("inf")


def _with_level(overrides: dict[str, float]) -> Callable[[type], type]:
    """Serve ``overrides`` ({"dc_lower": ..., "dc_upper": ...}) instead of the Donchian outputs."""

    def wrap(base: type) -> type:
        class Probe(base):  # type: ignore[misc, valid-type]
            def _get_indicator_value(self, name: str) -> float:
                return overrides[name] if name in overrides else super()._get_indicator_value(name)

        return Probe

    return wrap


# the signal bar closes at 10 250; the tick is 0.1; Donchian's own 9 500 / 11 000 serve the other side
_R11_BARS = [
    B(10150, 10300, 9500, 10150),
    B(10200, 11000, 10100, 10250),
    B(10250, 10260, 10140, 10150),
    B(10150, 10260, 10140, 10150),
]
_R11_DSL = dsl(entry_conditions={"long": ["close > 10200"]})

INVALID_LEVELS = [
    pytest.param({"dc_lower": _NAN}, id="stop-nan"),
    pytest.param({"dc_lower": _INF}, id="stop-inf"),
    pytest.param({"dc_lower": -_INF}, id="stop--inf"),
    pytest.param({"dc_lower": 0.0}, id="stop-zero"),
    pytest.param({"dc_lower": -5.0}, id="stop-negative"),
    pytest.param({"dc_lower": 0.04}, id="stop-rounds-to-zero"),
    pytest.param({"dc_lower": 10250.0}, id="stop-equals-close"),
    pytest.param({"dc_lower": 10250.04}, id="stop-rounds-up-to-the-close"),
    pytest.param({"dc_lower": 10250.1}, id="stop-above-close"),
    pytest.param({"dc_upper": _NAN}, id="target-nan"),
    pytest.param({"dc_upper": _INF}, id="target-inf"),
    pytest.param({"dc_upper": 0.0}, id="target-zero"),
    pytest.param({"dc_upper": -7.0}, id="target-negative"),
    pytest.param({"dc_upper": 10250.0}, id="target-equals-close"),
    pytest.param({"dc_upper": 10249.96}, id="target-rounds-down-to-the-close"),
    pytest.param({"dc_upper": 10249.9}, id="target-below-close"),
    pytest.param({"dc_lower": 9500.0, "dc_upper": 1e30}, id="target-beyond-the-price-range"),
]


@pytest.mark.parametrize("levels", INVALID_LEVELS)
def test_an_invalid_level_skips_the_entry_and_never_falls_back_to_two_percent(levels: dict[str, float]) -> None:
    out = run(_R11_BARS, d=_R11_DSL, wrap=_with_level(levels))

    assert out.orders == []  # no entry, hence no 2 % stop either
    assert out.open_positions == []
    assert out.strategy._sizing_skips == {"level_invalid": 1}


@pytest.mark.parametrize(
    ("levels", "stop", "target"),
    [
        pytest.param({"dc_lower": 10249.9}, 10249.9, 11000.0, id="stop-one-tick-below-the-close"),
        pytest.param({"dc_upper": 10250.1}, 9500.0, 10250.1, id="target-one-tick-above-the-close"),
        pytest.param({"dc_lower": 10249.94, "dc_upper": 10250.06}, 10249.9, 10250.1, id="rounded-to-the-tick"),
    ],
)
def test_levels_one_tick_from_the_close_are_valid_and_placed_rounded(
    levels: dict[str, float], stop: float, target: float
) -> None:
    seen: dict[str, Any] = {}
    flat_after = [*_R11_BARS[:2], B(10250, 10250, 10250, 10250), B(10250, 10250, 10250, 10250)]  # touches neither
    out = run(flat_after, d=_R11_DSL, wrap=_compose(_with_level(levels), probe({3: _record(seen, "live")})))

    assert out.strategy._sizing_skips == {}
    live = {(o[0], o[2] or o[3]) for o in seen["live"][0]}
    assert live == {("STOP_MARKET", stop), ("LIMIT", target)}


def _compose(*wraps: Callable[[type], type]) -> Callable[[type], type]:
    def wrap(base: type) -> type:
        for w in wraps:
            base = w(base)
        return base

    return wrap


def test_a_single_signal_level_side_validates_only_what_it_configures() -> None:
    """Only the target is signal_level: an invalid Donchian LOWER (unused) must not block the entry, the
    stop stays the generic 2 % of the fill (10 250 -> 10 045)."""
    d = _R11_DSL | {"stop_loss_long": None}
    seen: dict[str, Any] = {}
    out = run(_R11_BARS, d=d, wrap=_compose(_with_level({"dc_lower": _NAN}), probe({3: _record(seen, "live")})))

    # sized on the generic 2 % stop (205 away -> 0.975 BTC) and capped at 50 % of the account: 0.487
    assert seen["live"][0] == [("LIMIT", 0.487, 0.0, 11000.0), ("STOP_MARKET", 0.487, 10045.0, 0.0)]
    assert out.strategy._sizing_skips == {}


# ---------------------------------------------------------------------------
# R13: reversal into a signal_level side -- the snapshot is taken on the reversal's submit bar
# ---------------------------------------------------------------------------

# A long is open when b4 (close 10 100) flips to a short. On the SUBMIT bar b4 the Donchian window gives
# stop (dc.upper) 10 130 and target (dc.lower) 9 990. With a 30 s venue latency the exit and the short
# entry fill bars later, when the window has moved on (target 9 995, then 10 050; stop 10 130, then
# 10 110): a snapshot read at fill time would land on those.
_R13_BARS = [
    B(10000, 10020, 9990, 10000),
    B(10000, 10020, 9990, 10000),
    B(10000, 10020, 9990, 10000),
    B(10000, 10020, 9990, 10000),
    B(10000, 10130, 9995, 10100),
    B(10100, 10110, 10050, 10100),
    B(10100, 10105, 10060, 10100),
    B(10100, 10105, 10060, 10100),
    B(10100, 10105, 10060, 10100),
    B(10100, 10105, 10060, 10100),
]
_R13_DSL = dsl(
    timeframe="5m",
    entry_conditions={"long": ["close < 10050"], "short": ["close > 10050"]},
    stop_loss_long=None,
    take_profit_long=None,
    stop_loss_short={"type": "signal_level", "indicator": "dc.upper"},
    take_profit_short={"type": "signal_level", "indicator": "dc.lower"},
    position_mode="reverse",
)


def test_reversal_into_a_signal_level_side_uses_the_reversal_submit_bar_snapshot() -> None:
    seen: dict[str, Any] = {}
    out = run(
        _R13_BARS,
        d=_R13_DSL,
        tf="5m",
        latency=LatencyModel(base_latency_nanos=30 * 1_000_000_000),
        wrap=probe({10: _record(seen, "live")}),
    )

    orders, positions = seen["live"]
    assert [side for side, _ in positions] == ["SHORT"]
    assert [(o[0], o[2] or o[3]) for o in orders] == [("LIMIT", 9990.0), ("STOP_MARKET", 10130.0)]
    assert out.strategy._sizing_skips == {}


def _after_flip(seen: dict[str, Any], strategy: Any) -> None:
    seen["live"] = _snapshot(strategy)
    seen["intent"] = strategy._intent


def test_reversal_into_an_invalid_signal_level_side_keeps_the_position() -> None:
    """An invalid level on the new side refuses the whole reversal on its submit bar: the long stays
    open with its protection, no exit is sent."""
    seen: dict[str, Any] = {}
    out = run(
        _R13_BARS[:5],
        d=_R13_DSL,
        tf="5m",
        wrap=_compose(_with_level({"dc_upper": 10100.0}), probe({5: lambda st: _after_flip(seen, st)})),
    )

    orders, positions = seen["live"]
    assert [side for side, _ in positions] == ["LONG"]
    assert [o[0] for o in orders] == ["LIMIT", "STOP_MARKET"]  # the long's own pair
    assert out.strategy._sizing_skips == {"level_invalid": 1}
    assert seen["intent"] is None  # nothing was begun: no exit was queued either


# ---------------------------------------------------------------------------
# R14: an adopted position has no snapshot
# ---------------------------------------------------------------------------


def _forget_snapshot_and_protection(strategy: Any) -> None:
    """What a restart leaves: an open position, no snapshot, no protection on the venue."""
    strategy._levels = None
    _cancel_protection(strategy)


def _try_ensure_protection(seen: dict[str, Any]) -> Callable[[Any], None]:
    def step(strategy: Any) -> None:
        try:
            strategy._ensure_protection()
        except RuntimeError as exc:
            seen["error"] = str(exc)

    return step


def test_adopted_position_without_a_snapshot_cannot_be_rearmed() -> None:
    seen: dict[str, Any] = {}
    run(_R9_BARS, d=dsl(entry_conditions={"long": ["close > 10200"]}), wrap=probe(
        {3: _forget_snapshot_and_protection, 4: _try_ensure_protection(seen)}
    ))

    assert seen["error"] == "signal_level protection cannot re-arm an adopted position"


def test_adopted_position_whose_orders_are_still_on_the_venue_is_left_alone() -> None:
    """Restart with the stop/target still resting: nothing to re-arm, so nothing raises."""
    seen: dict[str, Any] = {}
    steps = {
        3: lambda strategy: setattr(strategy, "_levels", None),
        4: _try_ensure_protection(seen),
        5: _record(seen, "live"),
    }
    run(_R9_BARS + [B(10250, 10260, 10240, 10250)], d=dsl(entry_conditions={"long": ["close > 10200"]}),
        wrap=probe(steps))

    assert "error" not in seen
    assert seen["live"][0] == _PAIR_9500_11000


# ---------------------------------------------------------------------------
# R15: schema
# ---------------------------------------------------------------------------


def _validate(**over: Any) -> StrategyDSL:
    return StrategyDSL.model_validate(dsl("sl_schema", **over))


def test_signal_level_with_a_real_output_is_accepted() -> None:
    parsed = _validate()
    assert parsed.stop_loss_long is not None and parsed.stop_loss_long.indicator == "dc.lower"
    assert parsed.take_profit_long is not None and parsed.take_profit_long.type == "signal_level"
    assert parsed.refresh_on_signal is None  # absent -> legacy


@pytest.mark.parametrize("side", ["stop_loss_long", "stop_loss_short", "take_profit_long", "take_profit_short"])
def test_every_directional_field_takes_signal_level(side: str) -> None:
    assert getattr(_validate(**{side: {"type": "signal_level", "indicator": "dc.middle"}}), side).type == (
        "signal_level"
    )


_TWO_INDICATORS = {"dc": {"type": "DONCHIAN", "period": 2}, "sma": {"type": "SMA", "period": 2}}


@pytest.mark.parametrize(
    ("over", "message"),
    [
        pytest.param({"stop_loss_long": {"type": "signal_level"}}, "indicator", id="no-indicator-stop"),
        pytest.param({"take_profit_long": {"type": "signal_level"}}, "indicator", id="no-indicator-target"),
        pytest.param(
            {"stop_loss_long": {"type": "signal_level", "indicator": "dc.nope"}}, "not an output", id="missing-output"
        ),
        pytest.param(
            {"stop_loss_long": {"type": "signal_level", "indicator": "dc"}}, "not an output", id="multi-output-no-output"
        ),
        pytest.param(
            {"stop_loss_long": {"type": "signal_level", "indicator": "zz.lower"}}, "not defined", id="unknown-indicator"
        ),
        pytest.param({"stop_loss": {"type": "signal_level", "indicator": "dc.lower"}}, "only allowed in", id="generic-stop"),
        pytest.param(
            {"take_profit": {"type": "signal_level", "indicator": "dc.upper"}}, "only allowed in", id="generic-target"
        ),
        pytest.param(
            {"stop_loss_long": {"type": "signal_level", "indicator": "dc.lower", "percent": 2.0}},
            "takes only",
            id="stray-percent",
        ),
        pytest.param(
            {"take_profit_long": {"type": "signal_level", "indicator": "dc.upper", "risk_reward_ratio": 2.0}},
            "takes only",
            id="stray-ratio",
        ),
        pytest.param(
            {"stop_loss_long": None, "take_profit_long": None, "refresh_on_signal": True},
            "refresh_on_signal",
            id="refresh-without-a-signal-level-side",
        ),
        pytest.param({"refresh_on_signal": "yes"}, "refresh_on_signal", id="refresh-not-a-bool"),
        pytest.param({"refresh_on_signal": 1}, "refresh_on_signal", id="refresh-int"),
    ],
)
def test_signal_level_misuse_is_a_schema_error(over: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _validate(**over)


def test_single_output_indicator_is_referenced_by_its_bare_name() -> None:
    parsed = _validate(indicators=_TWO_INDICATORS, stop_loss_long={"type": "signal_level", "indicator": "sma"})
    assert parsed.stop_loss_long is not None and parsed.stop_loss_long.indicator == "sma"


@pytest.mark.parametrize("value", [None, False])
def test_refresh_absent_null_or_false_compile_the_same_source(value: bool | None) -> None:
    base = dsl("sl_refresh")
    off = StrategyCompiler().compile(validate_strategy_dict({**base, "refresh_on_signal": value}))
    absent = StrategyCompiler().compile(validate_strategy_dict(base))
    assert _norm(off) == _norm(absent)
    assert "_maybe_delay_validation_action('long_refresh')" not in absent  # no signal-bar refresh emitted


def _norm(src: str) -> str:
    import re

    return re.sub(r"Generated: [0-9T:.\-]+", "Generated: X", src)


def test_legacy_strategy_carries_no_signal_level_state_or_methods() -> None:
    """Only the three dead branches reach a legacy strategy; the snapshot machinery is emitted on demand."""
    legacy = {k: v for k, v in dsl("sl_legacy").items() if k not in ("stop_loss_long", "take_profit_long")}
    src = _norm(StrategyCompiler().compile(validate_strategy_dict(legacy)))

    for marker in ("def _take_levels", "def _read_levels", "def _refresh_levels", "def _signal_level_px",
                   "_pending_levels", "self._levels", "_refresh_invalid"):
        assert marker not in src, marker
    assert "self._signal_level_px('sl', is_long)" in src  # the dead branch the goldens pin


# ---------------------------------------------------------------------------
# R16: directional fields beat the generic ones
# ---------------------------------------------------------------------------

_R16_BARS = [
    B(9800, 9900, 9790, 9800),
    B(9800, 9810, 9700, 9750),  # short signal (close < 9800): fills at 9 750
    B(9750, 9760, 9740, 9750),
    B(9750, 9760, 9740, 9750),
]


def test_the_side_without_an_override_keeps_the_generic_percent_stop() -> None:
    """R16: the long has signal_level stop/target, the short has none: it takes the generic 2 % / 4 %
    of its fill (9 750 -> stop 9 945, target 9 360) and never touches the level snapshot."""
    d = dsl(
        entry_conditions={"long": ["close > 10200"], "short": ["close < 9800"]},
    )
    seen: dict[str, Any] = {}
    out = run(_R16_BARS, d=d, wrap=probe({3: _record(seen, "live")}))

    orders, positions = seen["live"]
    assert [side for side, _ in positions] == ["SHORT"]
    assert [(o[0], o[2] or o[3]) for o in orders] == [("LIMIT", 9360.0), ("STOP_MARKET", 9945.0)]
    assert out.strategy._sizing_skips == {}


def test_a_directional_override_beats_the_generic_stop() -> None:
    """The short's own 1 % stop wins over the generic 2 % (existing precedence, unchanged)."""
    d = dsl(
        entry_conditions={"long": ["close > 10200"], "short": ["close < 9800"]},
        stop_loss_short={"type": "fixed_pct", "percent": 1.0},
    )
    seen: dict[str, Any] = {}
    run(_R16_BARS, d=d, wrap=probe({3: _record(seen, "live")}))

    assert [(o[0], o[2] or o[3]) for o in seen["live"][0]] == [("LIMIT", 9360.0), ("STOP_MARKET", 9847.5)]


# ---------------------------------------------------------------------------
# R7: post_only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("refresh", [False, True])
def test_signal_level_with_post_only_is_refused_at_start(refresh: bool) -> None:
    d = dsl("sl_post_only", **({"refresh_on_signal": True} if refresh else {}))
    with pytest.raises(ValueError, match="post_only"):
        run(_R9_BARS, d=d, entry_order="post_only")


# ---------------------------------------------------------------------------
# B1: a refused reversal must not leave its sizing snapshot behind
# ---------------------------------------------------------------------------

# The long owns stop dc.lower; the opposite signal sizes a short whose levels are 10 130 / 9 990 and is
# refused. The long's protection, read afterwards, must still be the long's own.
_LONG_OWNS_SL = _R13_DSL | {
    "stop_loss_long": {"type": "signal_level", "indicator": "dc.lower"},
    "take_profit": {"type": "fixed_pct", "percent": 90},
}


def _long_state_then_lose_protection(seen: dict[str, Any]) -> Callable[[Any], None]:
    def step(strategy: Any) -> None:
        seen["state"] = (
            strategy._levels,
            strategy._pending_levels,
            dict(strategy._sizing_skips),
            strategy._calculate_sl_price(10000, True),
        )
        _cancel_protection(strategy)

    return step


def test_reversal_refused_for_min_notional_keeps_the_long_stop() -> None:
    """The short's 0.001 BTC minimum is below the 50 USDT min notional: refused. The re-arm that follows
    submits the LONG's 9 990 stop, not the refused short's 10 130."""
    seen: dict[str, Any] = {}
    steps = {5: _long_state_then_lose_protection(seen), 6: _rearm, 7: _record(seen, "rearmed")}
    run(_R13_BARS, d=_LONG_OWNS_SL, tf="5m", risk_per_trade=0.00001, wrap=probe(steps))

    assert seen["state"] == ((9990.0, None), None, {"min_notional": 1}, 9990.0)
    assert seen["rearmed"] == (
        [("LIMIT", 0.009, 0.0, 19000.0), ("STOP_MARKET", 0.009, 9990.0, 0.0)],
        [("LONG", 0.009)],
    )


def test_protection_reads_never_see_a_pending_snapshot() -> None:
    """Defence in depth: even with a sizing snapshot lying around (here a forged one), arming and re-arming
    read only the open position's own levels; the pending one is readable inside sizing alone."""
    seen: dict[str, Any] = {}

    def forged_pending_then_lose_protection(strategy: Any) -> None:
        strategy._pending_levels = (1.0, 2.0)
        _cancel_protection(strategy)
        _rearm(strategy)  # same step: the next bar's sizing would overwrite the forged snapshot

    steps = {5: forged_pending_then_lose_protection, 6: _record(seen, "rearmed")}
    run(_R13_BARS, d=_LONG_OWNS_SL, tf="5m", risk_per_trade=0.00001, wrap=probe(steps))

    assert seen["rearmed"] == (
        [("LIMIT", 0.009, 0.0, 19000.0), ("STOP_MARKET", 0.009, 9990.0, 0.0)],
        [("LONG", 0.009)],
    )


# A 5 000 low gives the long a far stop (and a 60 000 target the jump never reaches); at 55 000 the 0.005 BTC
# the 50 USDT min notional demands exceeds the 0.55 % size cap (0.0009 BTC): refused as min_qty_cap.
_CAP_ROWS = [
    B(10000, 10020, 5000, 10000),
    *[B(10000, 10020, 9995, 10000)] * 4,
    B(55000, 55100, 54900, 55000),
    *[B(55000, 55050, 54950, 55000)] * 3,
]


def test_reversal_refused_by_the_size_cap_keeps_the_long_stop() -> None:
    seen: dict[str, Any] = {}
    steps = {6: _long_state_then_lose_protection(seen), 7: _rearm, 8: _record(seen, "rearmed")}
    d = _LONG_OWNS_SL | {"take_profit": {"type": "risk_reward", "risk_reward_ratio": 10}}
    run(_CAP_ROWS, d=d, tf="5m", max_position_pct=0.0055, wrap=probe(steps))

    assert seen["state"] == ((5000.0, None), None, {"min_qty_cap": 1}, 5000.0)
    assert seen["rearmed"] == (
        [("LIMIT", 0.005, 0.0, 60000.0), ("STOP_MARKET", 0.005, 5000.0, 0.0)],
        [("LONG", 0.005)],
    )


def test_reversal_refused_while_the_sizing_atr_warms_up_keeps_the_long_stop() -> None:
    seen: dict[str, Any] = {}

    def atr_cold(strategy: Any) -> None:
        strategy._sizing_atr_value = lambda: None

    steps = {4: atr_cold, 5: _long_state_then_lose_protection(seen), 6: _rearm, 7: _record(seen, "rearmed")}
    run(_R13_BARS, d=_LONG_OWNS_SL, tf="5m", sizing_mode="atr", sizing_atr_period=2, wrap=probe(steps))

    levels, pending, skips, stop = seen["state"]
    assert (levels, pending, skips, stop) == ((9990.0, None), None, {"atr_warmup": 1}, 9990.0)
    assert [(o[0], o[2]) for o in seen["rearmed"][0] if o[0] == "STOP_MARKET"] == [("STOP_MARKET", 9990.0)]


# ---------------------------------------------------------------------------
# B2: a partial exit while a refresh waits in the outbox
# ---------------------------------------------------------------------------


def _refresh_then_partial_reduce(strategy: Any) -> None:
    bar = strategy.cache.bar(strategy.primary_bar_type)
    strategy._get_indicator_value = lambda name: 10100.0 if name.endswith("lower") else 10400.0
    strategy._refresh_levels(bar, True)
    # a real venue fill (reduce-only 0.100 of the 0.266) while the refreshed commands wait for their release
    strategy.submit_order(
        strategy.order_factory.market(
            instrument_id=strategy.instrument_id,
            order_side=OrderSide.SELL,
            quantity=Quantity.from_str("0.100"),
            time_in_force=TimeInForce.IOC,
            reduce_only=True,
        )
    )


_R10_TWO_BARS = [B(10150, 10200, 9500, 10150), B(10150, 11000, 10100, 10250), *[B(10250, 10300, 10200, 10250)] * 5]


def test_partial_exit_while_a_refresh_waits_keeps_the_refreshed_levels_at_the_new_size() -> None:
    seen: dict[str, Any] = {}
    out = run(
        _R10_TWO_BARS,
        d=_refresh_dsl(timeframe="5m"),
        mode="tick",
        wrap=probe({3: _refresh_then_partial_reduce, 4: _record(seen, "after")}),
    )

    assert seen["after"] == (
        [("LIMIT", 0.166, 0.0, 10400.0), ("STOP_MARKET", 0.166, 10100.0, 0.0)],
        [("LONG", 0.166)],
    )
    assert out.strategy._levels == (10100.0, 10400.0)  # the refreshed snapshot, never the old 9 500 / 11 000
    old = [o for o in out.orders if o.reduce_only and o.type in ("LIMIT", "STOP_MARKET") and o.px in (11000.0,)]
    assert [o.status for o in old] == ["CANCELED"]


def test_two_partial_exits_while_a_refresh_waits_requeue_the_pair_once_each_time() -> None:
    seen: dict[str, Any] = {}

    def second_reduce(strategy: Any) -> None:
            strategy.submit_order(
            strategy.order_factory.market(
                instrument_id=strategy.instrument_id,
                order_side=OrderSide.SELL,
                quantity=Quantity.from_str("0.050"),
                time_in_force=TimeInForce.IOC,
                reduce_only=True,
            )
        )

    rows = _R10_TWO_BARS + [B(10250, 10300, 10200, 10250)]
    run(
        rows,
        d=_refresh_dsl(timeframe="5m"),
        mode="tick",
        wrap=probe({3: _compose_steps(_refresh_then_partial_reduce, second_reduce), 4: _record(seen, "after")}),
    )

    assert seen["after"] == (
        [("LIMIT", 0.116, 0.0, 10400.0), ("STOP_MARKET", 0.116, 10100.0, 0.0)],
        [("LONG", 0.116)],
    )


def _compose_steps(*steps: Callable[[Any], None]) -> Callable[[Any], None]:
    def step(strategy: Any) -> None:
        for each in steps:
            each(strategy)

    return step


def test_sync_without_a_queued_refresh_still_resizes_both_legs_at_their_prices() -> None:
    """The R12 resize is unchanged: a partial fill with no refresh pending keeps the entry's prices."""
    seen: dict[str, Any] = {}
    run(_R12_BARS, d=dsl(), wrap=probe({3: _record(seen, "after_fill"), 4: _rearm, 5: _record(seen, "resized")}))

    pair = [("LIMIT", 0.1, 0.0, 11000.0), ("STOP_MARKET", 0.1, 9500.0, 0.0)]
    assert seen["after_fill"] == (pair, [("LONG", 0.1)])
    assert seen["resized"] == (pair, [("LONG", 0.1)])


# ---------------------------------------------------------------------------
# B3: a refresh obeys the validation defer
# ---------------------------------------------------------------------------


def _levels_and_pending(into: dict[str, Any], key: str) -> Callable[[Any], None]:
    def step(strategy: Any) -> None:
        into[key] = (strategy._levels, strategy._pending_validation_action, strategy._refresh_invalid)

    return step


def test_refresh_is_deferred_one_bar_and_snapshots_the_dispatch_bar() -> None:
    """execution_delay_probability=1.0: the entry signal b1 is dispatched on b2 (levels of b1+b2, 0.492 BTC on
    the 50 USDT-risk distance); the refresh signal b3 only queues the refresh, b4 dispatches it and reads the
    levels there (b3+b4: stop 10 120, target 10 400); the venue replaces the pair with the next bar."""
    seen: dict[str, Any] = {}
    steps = {
        4: _compose_steps(_levels_and_pending(seen, "signal_bar"), _record(seen, "signal_live")),
        5: _levels_and_pending(seen, "dispatch_bar"),
        6: _record(seen, "replaced"),
    }
    run(_R10_BARS[:6], d=_refresh_dsl(), delay=1.0, wrap=probe(steps))

    assert seen["signal_bar"] == ((10100.0, 11000.0), "long_refresh", 0)  # signal bar: nothing replaced yet
    assert seen["signal_live"][0] == _PAIR_AT_10100_11000
    assert seen["dispatch_bar"] == ((10120.0, 10400.0), None, 0)  # dispatched: levels of the dispatch bar
    assert seen["replaced"][0] == [("LIMIT", 0.492, 0.0, 10400.0), ("STOP_MARKET", 0.492, 10120.0, 0.0)]


_PAIR_AT_10100_11000 = [("LIMIT", 0.492, 0.0, 11000.0), ("STOP_MARKET", 0.492, 10100.0, 0.0)]


def test_deferred_refresh_with_invalid_dispatch_levels_keeps_the_old_protection() -> None:
    seen: dict[str, Any] = {}

    def invalid_target(strategy: Any) -> None:  # the dispatch bar closes at 10 150: a target there is invalid
        strategy._get_indicator_value = lambda name: 10150.0 if name.endswith("upper") else 10120.0

    steps = {
        4: invalid_target,
        5: _compose_steps(_levels_and_pending(seen, "dispatched"), _record(seen, "live")),
    }
    run(_R10_BARS[:6], d=_refresh_dsl(), delay=1.0, wrap=probe(steps))

    assert seen["dispatched"] == ((10100.0, 11000.0), None, 1)
    assert seen["live"][0] == _PAIR_AT_10100_11000


def test_deferred_refresh_is_a_no_op_when_the_position_closed_before_dispatch() -> None:
    """The b3 bar signals a refresh AND an exit: the exit goes out at once, so at dispatch (b4) the position
    is gone and the refresh is neither applied nor counted as invalid."""
    seen: dict[str, Any] = {}
    d = _refresh_dsl(exit_conditions={"long": ["close > 10290"]})
    run(
        _R10_BARS[:6],
        d=d,
        delay=1.0,
        wrap=probe({5: _compose_steps(_levels_and_pending(seen, "after"), _record(seen, "live"))}),
    )

    assert seen["after"] == (None, None, 0)
    assert seen["live"] == ([], [])


def test_time_filters_still_gate_a_refresh() -> None:
    """R10b: entry time filters apply to the refresh. 1970-01-01 is a Thursday: with it blocked the entry
    never happens, so no refresh is ever queued."""
    seen: dict[str, Any] = {}
    d = _refresh_dsl(time_filters={"blocked_days": ["Thursday"]})
    run(_R10_BARS, d=d, delay=1.0, wrap=probe({6: _levels_and_pending(seen, "after")}))

    assert seen["after"] == (None, None, 0)


# ---------------------------------------------------------------------------
# B4: an adopted position missing a configured signal_level leg
# ---------------------------------------------------------------------------


def _lose_leg(order_type: str) -> Callable[[Any], None]:
    def step(strategy: Any) -> None:
        strategy._levels = None
        for order in strategy._protective_orders():
            if order.order_type.name == order_type:
                strategy.cancel_order(order)

    return step


_R9_PLUS = _R9_BARS + [B(10250, 10260, 10240, 10250)]
_R9_LONG_DSL = dsl(entry_conditions={"long": ["close > 10200"]})


@pytest.mark.parametrize(
    ("lost", "kept"),
    [
        pytest.param("STOP_MARKET", [("LIMIT", 0.266, 0.0, 11000.0)], id="stop-missing"),
        pytest.param("LIMIT", [("STOP_MARKET", 0.266, 9500.0, 0.0)], id="target-missing"),
    ],
)
def test_adopted_position_missing_a_configured_leg_raises_and_adds_nothing(
    lost: str, kept: list[tuple[str, float, float, float]]
) -> None:
    seen: dict[str, Any] = {}
    steps = {3: _lose_leg(lost), 4: _try_ensure_protection(seen), 5: _record(seen, "live")}
    run(_R9_PLUS, d=_R9_LONG_DSL, wrap=probe(steps))

    assert seen["error"] == "signal_level protection cannot re-arm an adopted position"
    assert seen["live"] == (kept, [("LONG", 0.266)])


def test_adopted_position_with_the_complete_pair_is_resized_at_its_existing_prices() -> None:
    """The pair still rests (at a size that no longer fits): kept at its own prices, no error."""
    seen: dict[str, Any] = {}

    def forget_and_resize_pair(strategy: Any) -> None:
        strategy._levels = None
        for order in strategy._protective_orders():
            strategy.cancel_order(order)
        # the venue still holds a pair at 9 400 / 11 500 sized for an older, smaller position
        strategy.submit_order(
            strategy.order_factory.stop_market(
                instrument_id=strategy.instrument_id,
                order_side=OrderSide.SELL,
                quantity=Quantity.from_str("0.100"),
                trigger_price=Price.from_str("9400.0"),
                time_in_force=TimeInForce.GTC,
                reduce_only=True,
            )
        )
        strategy.submit_order(
            strategy.order_factory.limit(
                instrument_id=strategy.instrument_id,
                order_side=OrderSide.SELL,
                quantity=Quantity.from_str("0.100"),
                price=Price.from_str("11500.0"),
                time_in_force=TimeInForce.GTC,
                reduce_only=True,
            )
        )

    steps = {3: forget_and_resize_pair, 4: _try_ensure_protection(seen), 5: _record(seen, "live")}
    run(_R9_PLUS + [B(10250, 10260, 10240, 10250)], d=_R9_LONG_DSL, wrap=probe(steps))

    assert "error" not in seen
    assert seen["live"][0] == [("LIMIT", 0.266, 0.0, 11500.0), ("STOP_MARKET", 0.266, 9400.0, 0.0)]


# ---------------------------------------------------------------------------
# S1: a single-output indicator is referenced as 'sma' or 'sma.value'
# ---------------------------------------------------------------------------


def test_single_output_indicator_accepts_the_dotted_value_spelling() -> None:
    parsed = _validate(
        indicators=_TWO_INDICATORS, stop_loss_long={"type": "signal_level", "indicator": "sma.value"}
    )
    assert parsed.stop_loss_long is not None and parsed.stop_loss_long.indicator == "sma.value"


def test_single_output_indicator_rejects_any_other_dotted_output() -> None:
    with pytest.raises(ValidationError, match="single output"):
        _validate(indicators=_TWO_INDICATORS, stop_loss_long={"type": "signal_level", "indicator": "sma.upper"})


@pytest.mark.parametrize("ref", ["sma", "sma.value"])
def test_both_spellings_of_a_single_output_level_compile_and_place_the_same_stop(ref: str) -> None:
    """SMA(2) of the closes 10 150 / 10 250 is 10 200: the stop rests there under either spelling, and
    the generated config names the runtime key 'sma' (never 'sma_value')."""
    d = dsl(
        "sl_sma",
        indicators=_TWO_INDICATORS,
        entry_conditions={"long": ["close > 10200"]},
        stop_loss_long={"type": "signal_level", "indicator": ref},
    )
    src = _norm(StrategyCompiler().compile(validate_strategy_dict(d)))
    seen: dict[str, Any] = {}
    run(_R9_BARS, d=d, wrap=probe({3: _record(seen, "live")}))

    assert 'stop_loss_long_indicator: str = "sma"' in src
    assert [(o[0], o[2]) for o in seen["live"][0] if o[0] == "STOP_MARKET"] == [("STOP_MARKET", 10200.0)]
