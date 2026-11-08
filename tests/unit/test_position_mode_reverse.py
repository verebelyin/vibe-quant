"""``position_mode: reverse`` -- an opposite entry signal closes the position and opens the other
side on the exit's own fill tick (design-v2 section 3.1, semantics rows R1, R2c, R4a-R4g, R5-R8).

Real NautilusTrader events on a raw ``BacktestEngine``: piecewise-constant prices (every 5m bar
and its five minutes sit at one level, so a fill at the NEXT bar's level is unambiguous), the
production fill-tick builder for the screening outbox, flat bars. Runner-level rows (R2, R2a, R2b,
R3, R4, R4h, R4i, R8a, R8b) live in ``tests/unit/test_reverse_runners.py``.

Default strategy: long while ``close < 10050``, short while ``close > 10050``, SL 2 % / TP 4 %.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pytest
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import FillModel, LatencyModel
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType, OrderType, TimeInForce
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Currency, Money, Price, Quantity
from pydantic import ValidationError

from vibe_quant.data.catalog import create_instrument
from vibe_quant.data.fill_ticks import build_fill_ticks
from vibe_quant.dsl import templates
from vibe_quant.dsl.compiler import StrategyCompiler, _to_class_name
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.dsl.schema import StrategyDSL
from vibe_quant.nt_compat import retain_log_guard

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

IID = "BTCUSDT-PERP.BINANCE"
_S = 1_000_000_000
_M = 60 * _S
_M5 = 5 * _M
_MS = 1_000_000


def tick_ts(bar_index: int) -> int:
    """The release tick of the signal on 5m bar ``bar_index``: the next 1m close."""
    return (bar_index + 1) * _M5 + _M - _MS


def bar_ts(bar_index: int) -> int:
    return (bar_index + 1) * _M5 - _MS


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class Row(NamedTuple):
    side: str
    reduce_only: bool
    type: str
    status: str
    qty: float
    px: float
    ts: int
    trigger: float
    cancel_requested: int  # ts of the first OrderPendingCancel event (0 = none)


class Out(NamedTuple):
    orders: list[Row]
    open_orders: list[Row]
    open_positions: list[tuple[str, float]]
    strategy: Any
    engine: BacktestEngine

    def entries(self) -> list[Row]:
        return [o for o in self.orders if not o.reduce_only and o.qty > 0]

    def exits(self) -> list[Row]:
        return [o for o in self.orders if o.reduce_only and o.type == "MARKET" and o.qty > 0]


_engines: list[BacktestEngine] = []


@pytest.fixture(autouse=True)
def _dispose_engines() -> Iterator[None]:
    yield
    while _engines:
        engine = _engines.pop()
        engine.reset()
        engine.dispose()


def _price(v: float) -> Price:
    return Price.from_str(f"{v:.1f}")


def _bars(closes: list[float], tf: str) -> list[Bar]:
    step, spec = (_M5, "5-MINUTE") if tf == "5m" else (_M, "1-MINUTE")
    bt = BarType.from_str(f"{IID}-{spec}-LAST-EXTERNAL")
    out = []
    for i, c in enumerate(closes):
        out.append(
            Bar(
                bar_type=bt,
                open=_price(c),
                high=_price(c),
                low=_price(c),
                close=_price(c),
                volume=Quantity.from_str("1000.000"),
                ts_event=i * step,
                ts_init=(i + 1) * step - _MS,
            )
        )
    return out


def _per_minute(closes: list[float]) -> list[float]:
    return [c for c in closes for _ in range(5)]


def _fill_ticks(closes: list[float]) -> list[Any]:
    minutes = _per_minute(closes)
    series = SimpleNamespace(
        ts=np.array([(m + 1) * _M - _MS for m in range(len(minutes))], dtype=np.int64),
        close=np.array(minutes),
    )
    ticks, _ = build_fill_ticks(series, "5m", create_instrument("BTCUSDT"))
    return ticks


def dsl(name: str, **over: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "name": name,
        "timeframe": "5m",
        "indicators": {},
        "entry_conditions": {"long": ["close < 10050"], "short": ["close > 10050"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
        "position_mode": "reverse",
    }
    d.update(over)
    return d


def compile_dsl(d: dict[str, Any]) -> tuple[type, type]:
    module = StrategyCompiler().compile_to_module(validate_strategy_dict(d))
    cn = _to_class_name(d["name"])
    return getattr(module, f"{cn}Strategy"), getattr(module, f"{cn}Config")


def _row(o: Any) -> Row:
    filled = float(o.filled_qty) > 0
    cancels = [e.ts_event for e in o.events if type(e).__name__ == "OrderPendingCancel"]
    trigger = float(o.trigger_price) if getattr(o, "trigger_price", None) is not None else 0.0
    limit_px = float(o.price) if getattr(o, "price", None) is not None else 0.0
    return Row(
        side=o.side.name,
        reduce_only=bool(o.is_reduce_only),
        type=o.order_type.name,
        status=o.status.name,
        qty=float(o.filled_qty),
        px=float(o.avg_px) if filled else limit_px,
        ts=int(o.ts_last),
        trigger=trigger,
        cancel_requested=int(cancels[0]) if cancels else 0,
    )


def run(
    closes: list[float],
    *,
    d: dict[str, Any] | None = None,
    mode: str = "tick",
    latency: LatencyModel | None = None,
    wrap: Callable[[type], type] | None = None,
    balance: int = 10_000,
    **cfg: Any,
) -> Out:
    """One compiled strategy on ``closes`` (one 5m bar each). ``mode``: ``tick`` = screening outbox on
    production fill ticks, ``bar`` = validation outbox on 1m detail bars, ``immediate`` = no outbox
    (venue latency, if any, is the only delay), ``defer`` = 1m strategy with the one-bar defer."""
    d = d or dsl("rev_unit")
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
    kw: dict[str, Any] = {"instrument_id": IID, **cfg}
    if mode == "tick":
        engine.add_data(_bars(closes, "5m") + _fill_ticks(closes))
        kw.update(command_release="trade_tick", execution_delay_probability=0.0)
    elif mode == "bar":
        engine.add_data(_bars(closes, "5m") + _bars(_per_minute(closes), "1m"))
        kw.update(
            command_release="bar",
            command_release_bar_type=f"{IID}-1-MINUTE-LAST-EXTERNAL",
            execution_delay_probability=0.0,
        )
    elif mode == "immediate":
        engine.add_data(_bars(closes, "5m"))
        kw.update(execution_delay_probability=0.0)
    else:
        assert mode == "defer"
        engine.add_data(_bars(closes, "1m"))
        kw.update(execution_delay_probability=1.0)
    strategy = strategy_cls(config=config_cls(**kw))
    engine.add_strategy(strategy)
    engine.run()
    cache = engine.cache
    return Out(
        orders=sorted((_row(o) for o in cache.orders()), key=lambda r: (r.ts, r.type)),
        open_orders=[_row(o) for o in cache.orders_open()],
        open_positions=[(p.side.name, float(p.quantity)) for p in cache.positions_open()],
        strategy=strategy,
        engine=engine,
    )


# a long enters on bar 0 (fills on bar 1's tick); the short signal is bar `SIG`, one bar only
def one_bar_short(signal_bar: int, n: int, *, up: float = 10100.0) -> list[float]:
    return [up if i == signal_bar else 10000.0 for i in range(n)]


# ---------------------------------------------------------------------------
# R8 / R1: the schema field and the byte-identical legacy compile
# ---------------------------------------------------------------------------


class TestSchema:
    @staticmethod
    def _base() -> dict[str, Any]:
        return dsl("rev_schema")

    def test_absent_and_null_are_legacy_and_reverse_is_accepted(self) -> None:
        base = self._base()
        del base["position_mode"]
        assert StrategyDSL.model_validate(base).position_mode is None
        assert StrategyDSL.model_validate({**base, "position_mode": None}).position_mode is None
        assert StrategyDSL.model_validate({**base, "position_mode": "reverse"}).position_mode == "reverse"

    @pytest.mark.parametrize("bad", ["Reverse", "flat", "", 1, True, "x"])
    def test_anything_else_names_the_valid_value(self, bad: object) -> None:
        with pytest.raises(ValidationError, match="reverse"):
            StrategyDSL.model_validate({**self._base(), "position_mode": bad})


def _src(d: dict[str, Any]) -> str:
    import re

    return re.sub(r"Generated: [0-9T:.\-]+", "Generated: X", StrategyCompiler().compile(validate_strategy_dict(d)))


class TestLegacyCompileUnchanged:
    def test_absent_null_compile_identically_and_carry_no_reversal_code(self) -> None:
        absent = dsl("rev_legacy")
        del absent["position_mode"]
        assert _src(absent) == _src({**absent, "position_mode": None})
        for marker in ("ReversalIntent", "_begin_reverse", "_reverse_on_closed", "self._intent", "_expire_reversal"):
            assert marker not in _src(absent), marker

    def test_reverse_emits_the_hooks_exactly_where_the_design_puts_them(self) -> None:
        src = _src(dsl("rev_emit"))
        assert "class ReversalIntent(NamedTuple)" in src
        assert "if not self._reverse_on_closed(event):" in src
        assert src.count("self._begin_reverse(bar, OrderSide.") == 2
        assert "self._expire_reversal(bar)" in src

    def test_legacy_templates_are_not_the_reverse_ones(self) -> None:
        assert templates.ORDER_METHODS_LINES != templates.ORDER_METHODS_REVERSE_LINES
        assert "_begin_reverse" not in "\n".join(templates.ORDER_METHODS_LINES + templates.ON_EVENT_LINES)

    def test_a_reverse_strategy_with_post_only_is_refused_at_start(self) -> None:
        with pytest.raises(ValueError, match="post_only"):
            run([10000.0] * 3, entry_order="post_only")  # R7


# ---------------------------------------------------------------------------
# R2c: active venue latency -- the entry is in flight, so it fills later than the exit
# ---------------------------------------------------------------------------


def test_active_latency_puts_the_entry_in_flight_after_the_exit() -> None:
    """R2c: venue latency 30 s on 5m bars: an order sent at bar t fills on bar t+1. The exit is sent
    on the signal bar 2 and fills on bar 3; PositionClosed submits the entry, which fills on bar 4."""
    closes = [10000.0, 10000.0, 10100.0, 10200.0, 10300.0, 10300.0]
    out = run(closes, mode="immediate", latency=LatencyModel(base_latency_nanos=30 * _S))

    long_entry, short_entry = out.entries()
    (exit_fill,) = out.exits()
    assert (long_entry.side, short_entry.side, exit_fill.side) == ("BUY", "SELL", "SELL")
    assert exit_fill.ts == bar_ts(3) and exit_fill.px == 10200.0
    assert short_entry.ts == bar_ts(4) and short_entry.px == 10300.0  # reported, not equal to the exit
    assert out.open_positions == [("SHORT", short_entry.qty)]


# ---------------------------------------------------------------------------
# R4f: the reversal path sends NO broad cancel -- the new protection must survive a slow cancel
# ---------------------------------------------------------------------------


def test_slow_venue_cancel_cannot_remove_the_new_sides_protection() -> None:
    """R4f: cancel latency 12 min vs insert 30 s. The legacy PositionClosed broad cancel_all
    (sent when the long closes, bar 3) would land AFTER the short's stop/target are live and cancel
    them; on the reversal path it is not sent, so the short's two orders are never cancel-requested
    before on_stop at the end of the data, and the long's are cancelled by id."""
    closes = [10000.0, 10000.0, 10100.0] + [10100.0] * 9
    lat = LatencyModel(base_latency_nanos=30 * _S, cancel_latency_nanos=12 * _M)
    out = run(closes, mode="immediate", latency=lat)

    (_, short_entry) = out.entries()
    assert short_entry.side == "SELL" and short_entry.px == 10100.0
    new_pair = sorted((o.type, o.trigger or o.px) for o in out.orders if o.side == "BUY" and o.reduce_only
                      and o.type in ("STOP_MARKET", "LIMIT"))
    assert new_pair == [("LIMIT", pytest.approx(10100.0 * 0.96)), ("STOP_MARKET", pytest.approx(10100.0 * 1.02))]
    for o in out.orders:
        if o.side == "BUY" and o.reduce_only and o.type in ("STOP_MARKET", "LIMIT"):
            assert o.status != "CANCELED" and o.cancel_requested >= bar_ts(len(closes) - 1), o
    old_pair = [o for o in out.orders if o.side == "SELL" and o.reduce_only and o.type in ("STOP_MARKET", "LIMIT")]
    assert sorted(o.status for o in old_pair) == ["CANCELED", "CANCELED"]
    assert out.open_positions == [("SHORT", short_entry.qty)]


# ---------------------------------------------------------------------------
# R4a / R4b / R4c / R4d: what happens to a pending reversal
# ---------------------------------------------------------------------------


def test_a_stop_that_closes_the_position_first_drops_the_reversal() -> None:
    """R4a: the release tick (9000) is below the long's stop (9800): the stop fills before the
    queued exit is sent, PositionClosed names the STOP as the closer, so no short is opened."""
    closes = [10000.0, 10000.0, 10000.0, 10100.0, 9000.0, 9000.0, 9000.0]
    out = run(closes)

    assert [o.side for o in out.entries()] == ["BUY", "BUY"]  # the long, then a plain re-entry once flat
    stops = [o for o in out.orders if o.type == "STOP_MARKET" and o.qty > 0]
    assert len(stops) == 1 and stops[0].side == "SELL"
    assert out.strategy._reverse_dropped == 1
    assert out.exits() == []  # the queued reduce-only exit hit a flat account: it filled nothing


def test_an_exit_that_never_fills_expires_at_the_next_bar_and_rearms_the_position() -> None:
    """R4b: the reversal exit is rejected/never filled (withheld here). The intent expires at the
    next primary bar, nothing is sold, and the long gets its cancelled protection back."""

    def withhold_exit(cls: type) -> type:
        class Withheld(cls):  # type: ignore[valid-type, misc]
            def _send_command(self, fn, *args):  # type: ignore[no-untyped-def]
                if fn == self.submit_order and args[0].is_reduce_only and args[0].order_type == OrderType.MARKET:
                    return
                super()._send_command(fn, *args)

        return Withheld

    out = run(one_bar_short(3, 8), wrap=withhold_exit)

    assert [o.side for o in out.entries()] == ["BUY"]
    assert out.strategy._reverse_dropped == 1 and out.strategy._intent is None
    assert out.open_positions == [("LONG", out.entries()[0].qty)]
    live = sorted((o.type, o.side) for o in out.open_orders)
    assert live == [("LIMIT", "SELL"), ("STOP_MARKET", "SELL")], live  # re-armed, not naked
    assert sum(o.status == "CANCELED" for o in out.orders) == 2  # the first pair was cancelled by id


def _after_bars(pack: list[tuple[bool, int, list[tuple[str, float]]]]) -> Any:
    """A wrapper recording, after each ``on_bar``, (intent gone, dropped, protection types/qty)."""

    def wrap(cls: type) -> type:
        class After(cls):  # type: ignore[valid-type, misc]
            def on_bar(self, bar):  # type: ignore[no-untyped-def]
                super().on_bar(bar)
                pack.append(
                    (
                        self._intent is None,
                        self._reverse_dropped,
                        [(o.order_type.name, float(o.quantity)) for o in self._protective_orders()],
                    )
                )

        return After

    return wrap


def test_a_partial_ioc_exit_is_terminal_and_the_leftover_is_rearmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R4b' terminal path on a real event: the exit REACHES the venue and fills 0.001 of the 0.5
    on the low-volume signal bar; the IOC cancels the rest, so the cached exit is closed. At the
    next primary bar the intent expires (the order is in the cache and terminal), no short is
    opened, and the 0.499 leftover gets its own SL/TP back."""
    real_bars = _bars

    def low_volume_bars(closes: list[float], tf: str) -> list[Bar]:
        return [
            b if i < 3 else Bar(
                bar_type=b.bar_type, open=b.open, high=b.high, low=b.low, close=b.close,
                volume=Quantity.from_str("0.004"), ts_event=b.ts_event, ts_init=b.ts_init,
            )
            for i, b in enumerate(real_bars(closes, tf))
        ]

    monkeypatch.setattr(sys.modules[__name__], "_bars", low_volume_bars)
    after: list[tuple[bool, int, list[tuple[str, float]]]] = []
    out = run([10000.0] * 3 + [10100.0] + [10000.0] * 4, mode="immediate",
              wrap=_after_bars(after))

    (partial,) = [o for o in out.orders
                  if o.reduce_only and o.type == "MARKET" and o.ts == bar_ts(3)]
    assert (partial.status, partial.qty) == ("CANCELED", 0.001)  # a real, terminal partial
    assert [o.side for o in out.entries()] == ["BUY"]  # no opposite entry
    assert out.strategy._intent is None and out.strategy._reverse_dropped == 1
    intent_gone, dropped, protection = after[4]  # the next primary bar
    assert intent_gone and dropped == 1
    assert sorted(protection) == [("LIMIT", 0.499), ("STOP_MARKET", 0.499)]  # the remainder


def test_a_rejected_ioc_exit_is_terminal_and_the_full_position_is_rearmed() -> None:
    """R4b' terminal path on a real event: the exit is submitted but the risk engine DENIES it
    (a quantity precision the instrument cannot hold), so the cached exit is terminal with no
    fill. The next primary bar expires the intent, no short is opened, and the full 0.5 is
    re-armed."""
    after: list[tuple[bool, int, list[tuple[str, float]]]] = []

    def reject_exit(cls: type) -> type:
        class Reject(cls):  # type: ignore[valid-type, misc]
            def _send_command(self, fn, *args):  # type: ignore[no-untyped-def]
                if fn == self.submit_order and args[0].is_reduce_only \
                        and args[0].order_type == OrderType.MARKET:
                    self.submit_order(
                        self.order_factory.market(
                            instrument_id=self.instrument_id,
                            order_side=args[0].side,
                            quantity=Quantity(0.5, 4),  # precision 4 > size_precision 3: denied
                            time_in_force=TimeInForce.IOC,
                            reduce_only=True,
                            client_order_id=args[0].client_order_id,
                        )
                    )
                    return
                super()._send_command(fn, *args)  # type: ignore[misc]

            def on_bar(self, bar):  # type: ignore[no-untyped-def]
                super().on_bar(bar)
                after.append(
                    (
                        self._intent is None,
                        self._reverse_dropped,
                        [(o.order_type.name, float(o.quantity)) for o in self._protective_orders()],
                    )
                )

        return Reject

    out = run([10000.0] * 3 + [10100.0] + [10000.0] * 4, mode="immediate", wrap=reject_exit)

    (denied,) = [o for o in out.orders
                 if o.reduce_only and o.type == "MARKET" and o.ts == bar_ts(3)]
    assert (denied.status, denied.qty) == ("DENIED", 0.0)  # a real, terminal rejection
    assert [o.side for o in out.entries()] == ["BUY"]  # no opposite entry
    assert out.strategy._intent is None and out.strategy._reverse_dropped == 1
    intent_gone, dropped, protection = after[4]
    assert intent_gone and dropped == 1
    assert sorted(protection) == [("LIMIT", 0.5), ("STOP_MARKET", 0.5)]  # the full position


def test_a_repeated_opposite_signal_while_a_reversal_is_pending_is_ignored() -> None:
    """R4c: a second `_submit_short_entry` on the signal bar creates no second intent or exit."""

    def twice(cls: type) -> type:
        class Twice(cls):  # type: ignore[valid-type, misc]
            def _submit_short_entry(self, bar):  # type: ignore[no-untyped-def]
                super()._submit_short_entry(bar)
                super()._submit_short_entry(bar)

        return Twice

    out = run([10000.0] * 3 + [10100.0] * 5, wrap=twice)

    assert [o.side for o in out.entries()] == ["BUY", "SELL"]
    reduce_only_markets = [o for o in out.orders if o.reduce_only and o.type == "MARKET"]
    assert len(reduce_only_markets) == 1, reduce_only_markets  # a second exit would be a second order
    assert out.strategy._reverse_dropped == 0


def test_a_close_arriving_after_the_intent_expired_opens_nothing() -> None:
    """R4d: the exit is released only after the intent expired (bar 4): the close is ordinary."""

    def late_exit(cls: type) -> type:
        class Late(cls):  # type: ignore[valid-type, misc]
            held = None

            def _send_command(self, fn, *args):  # type: ignore[no-untyped-def]
                if fn == self.submit_order and args[0].is_reduce_only and args[0].order_type == OrderType.MARKET:
                    self.held = args[0]
                    return
                super()._send_command(fn, *args)

            def on_bar(self, bar):  # type: ignore[no-untyped-def]
                super().on_bar(bar)
                if self.held is not None and self._intent is None and self._reverse_dropped:
                    order, self.held = self.held, None
                    self.submit_order(order)  # now the exit reaches the venue, after the expiry

        return Late

    out = run(one_bar_short(3, 8), wrap=late_exit)

    assert [o.side for o in out.entries() if o.side == "SELL"] == []  # no short after the late close
    assert out.exits() and out.exits()[0].side == "SELL"  # the exit did close the long
    assert out.strategy._reverse_dropped == 1  # counted once, at the expiry


# ---------------------------------------------------------------------------
# R4e: the time-exit count restarts with the new entry
# ---------------------------------------------------------------------------


def test_time_exit_counts_from_the_reversal_entry() -> None:
    """R4e: max_bars_in_trade=6. The long is 4 bars old at the reversal (bar 4). With the count
    carried the short would time-exit at bar 6; restarted at 0 it exits at bar 10."""
    closes = [10000.0] * 4 + [10100.0] * 8
    out = run(closes, d=dsl("rev_age", max_bars_in_trade=6))

    long_entry, short_entry = out.entries()[:2]
    assert (long_entry.side, short_entry.side) == ("BUY", "SELL")
    assert short_entry.ts == tick_ts(4)
    time_exit = [x for x in out.exits() if x.side == "BUY"][0]
    assert time_exit.ts == tick_ts(10), (time_exit.ts, tick_ts(6), tick_ts(10))


# ---------------------------------------------------------------------------
# R5: same-bar interactions with the time exit and the time filter
# ---------------------------------------------------------------------------


def test_time_exit_on_the_signal_bar_wins_and_no_reversal_follows() -> None:
    """R5: max_bars_in_trade=4: the 4th bar with the long open is bar 4, the short signal bar.
    The time exit fires alone; the short is a plain entry one bar later, not on the exit's tick."""
    closes = [10000.0] * 4 + [10100.0] * 6
    out = run(closes, d=dsl("rev_r5a", max_bars_in_trade=4))

    exit_fill = out.exits()[0]
    assert (exit_fill.side, exit_fill.ts) == ("SELL", tick_ts(4))
    short_entry = [o for o in out.entries() if o.side == "SELL"][0]
    assert short_entry.ts == tick_ts(5) > exit_fill.ts
    assert out.strategy._reverse_dropped == 0


def test_entries_blocked_by_a_time_filter_keep_the_position() -> None:
    """R5: sessions allow 00:00-00:30 only; the short signal arrives on bar 7 (close 00:40)."""
    d = dsl("rev_r5b", time_filters={"allowed_sessions": [{"start": "00:00", "end": "00:30"}]})
    out = run(one_bar_short(7, 12), d=d)

    assert [o.side for o in out.entries()] == ["BUY"]
    assert out.exits() == []
    assert out.open_positions == [("LONG", out.entries()[0].qty)]
    assert out.strategy._intent is None


# ---------------------------------------------------------------------------
# R6: 1m defer -- the state at dispatch decides
# ---------------------------------------------------------------------------


class TestDeferredDispatch:
    @staticmethod
    def _closes(stop_bar: int | None) -> list[float]:
        # 1m bars: long from bar 0, short signal at bar 20, bar 21 is where the stop may fire
        c = [10000.0] * 40
        c[20] = 10100.0
        if stop_bar is not None:
            c[stop_bar] = 9000.0
        return c

    @staticmethod
    def _run(closes: list[float], wrap: Callable[[type], type] | None = None) -> Out:
        d = dsl("rev_r6", timeframe="1m", entry_conditions={"long": ["close < 10050"], "short": ["close > 10050"]})
        return run(closes, d=d, mode="defer", wrap=wrap)

    def test_flat_at_dispatch_is_a_plain_entry(self) -> None:
        """The long's stop fills on bar 21, before the deferred short is dispatched: flat -> a plain
        short entry (no reduce-only market exit, no intent, nothing dropped)."""
        out = self._run(self._closes(stop_bar=21))

        assert [o.side for o in out.entries()][:2] == ["BUY", "SELL"]
        assert [o for o in out.orders if o.reduce_only and o.type == "MARKET" and o.ts < 30 * _M] == []
        assert out.strategy._reverse_dropped == 0

    def test_same_side_at_dispatch_does_nothing(self) -> None:
        """A deferred LONG entry dispatched while a long is open sends nothing."""

        def pending_long(cls: type) -> type:
            class Pending(cls):  # type: ignore[valid-type, misc]
                def on_bar(self, bar):  # type: ignore[no-untyped-def]
                    if bar.ts_init == 11 * _M - _MS and self._position_open:
                        self._pending_validation_action = "long_entry"
                    super().on_bar(bar)

            return Pending

        baseline = self._run([10000.0] * 40)
        out = self._run([10000.0] * 40, wrap=pending_long)

        assert [o.side for o in out.entries()] == ["BUY"]
        assert len(out.orders) == len(baseline.orders)
        # (the only exit is on_stop closing the open long when the data ends)
        assert [x for x in out.exits() if x.ts < 39 * _M] == [] and out.strategy._reverse_dropped == 0


# ---------------------------------------------------------------------------
# R4g: on_stop / on_reset clear the intent
# ---------------------------------------------------------------------------


def test_stop_and_reset_clear_a_pending_intent() -> None:
    """R4g: the signal is the LAST bar, so the intent is still pending when the run ends."""
    out = run(one_bar_short(3, 4))

    assert out.strategy._intent is None  # on_stop ran at the end of engine.run()
    out.strategy._intent = object()
    out.engine.reset()
    assert out.strategy._intent is None
