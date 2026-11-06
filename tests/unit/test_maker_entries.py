"""Post-only maker entries in the generated strategy (validation-only).

With ``entry_order="post_only"`` an entry is sized at signal time, queued in the
command outbox as an INTENT and priced when released: the limit sits
``maker_offset_ticks`` away from the RELEASING datum's price, not the signal
close. NT's matching engine processes a bar before the strategy sees it, so the
book already shows the release close and a limit one tick inside it is not
marketable (a post-only order priced at the signal close was rejected on ~half
of the boundaries). Unfilled entries are cancelled after
``maker_entry_timeout_bars`` primary bars, exits and opposite signals withdraw
them, and market mode (the default) is untouched.

Direct BacktestEngine runs on synthetic 5m strategy bars + 1m release bars.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.data import Bar, BarType, TradeTick
from nautilus_trader.model.enums import (
    AccountType,
    AggressorSide,
    LiquiditySide,
    OmsType,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from nautilus_trader.model.identifiers import TradeId, Venue
from nautilus_trader.model.objects import Currency, Money, Price, Quantity

from vibe_quant.data.catalog import create_instrument
from vibe_quant.dsl.compiler import StrategyCompiler, _to_class_name
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.nt_compat import retain_log_guard

if TYPE_CHECKING:
    from collections.abc import Iterator

_BTC = "BTCUSDT-PERP.BINANCE"
_S = 1_000_000_000
_M = 60 * _S
_M5 = 5 * _M
_VOL = "100000.000"
_TICK = 0.1


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@pytest.fixture
def engines() -> Iterator[list[BacktestEngine]]:
    created: list[BacktestEngine] = []
    yield created
    for engine in created:
        engine.dispose()


def _bar(
    spec: str, step: int, i: int, o: float, h: float, low: float, c: float, *, iid: str = _BTC, prec: int = 1
) -> Bar:
    """Bar ``i`` of length ``step``; ts_init = bar close (like the catalog)."""
    return Bar(
        bar_type=BarType.from_str(f"{iid}-{spec}-LAST-EXTERNAL"),
        open=Price.from_str(f"{o:.{prec}f}"),
        high=Price.from_str(f"{h:.{prec}f}"),
        low=Price.from_str(f"{low:.{prec}f}"),
        close=Price.from_str(f"{c:.{prec}f}"),
        volume=Quantity.from_str(_VOL),
        ts_event=i * step,
        ts_init=(i + 1) * step,
    )


def _strategy_bars(closes: list[float]) -> list[Bar]:
    return [_bar("5-MINUTE", _M5, i, c, c, c, c) for i, c in enumerate(closes)]


def _one_minute(
    n: int, *, base: float = 100.0, ohlc: dict[int, tuple[float, float, float, float]] | None = None
) -> list[Bar]:
    """``n`` flat 1m bars at ``base``; ``ohlc[i]`` overrides bar ``i`` (ts_init = (i+1)*_M)."""
    ohlc = ohlc or {}
    return [_bar("1-MINUTE", _M, i, *ohlc.get(i, (base, base, base, base))) for i in range(n)]


def _first_release(signal_bar: int) -> int:
    """Index of the first 1m bar whose ts_init is strictly after 5m bar ``signal_bar`` closes."""
    return 5 * (signal_bar + 1)


def _engine(symbol: str = "BTCUSDT") -> BacktestEngine:
    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    retain_log_guard(engine)
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(1_000_000, Currency.from_str("USDT"))],
        default_leverage=Decimal("10"),
        fill_model=FillModel(prob_fill_on_limit=1.0, prob_slippage=0.0, random_seed=42),
        bar_execution=True,
        use_reduce_only=True,
    )
    engine.add_instrument(create_instrument(symbol))
    return engine


def _dsl(name: str, **overrides: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "name": name,
        "timeframe": "5m",
        "indicators": {"sma": {"type": "SMA", "period": 2}},
        "entry_conditions": {"long": ["close > 100.5"]},
        "exit_conditions": {"long": ["close < 50"]},
        "stop_loss": {"type": "fixed_pct", "percent": 20.0},
        "take_profit": {"type": "fixed_pct", "percent": 40.0},
    }
    d.update(overrides)
    return d


def _compiled(d: dict[str, Any]) -> tuple[type, type]:
    module = StrategyCompiler().compile_to_module(validate_strategy_dict(d))
    cn = _to_class_name(d["name"])
    return getattr(module, f"{cn}Strategy"), getattr(module, f"{cn}Config")


def _run(
    engines: list[BacktestEngine],
    *,
    data: list[Any],
    dsl: dict[str, Any],
    release: str = "bar",
    symbol: str = "BTCUSDT",
    strategy_wrap: Any = None,
    **config: Any,
) -> tuple[BacktestEngine, Any]:
    strategy_cls, config_cls = _compiled(dsl)
    if strategy_wrap is not None:
        strategy_cls = strategy_wrap(strategy_cls)
    iid = f"{symbol}-PERP.BINANCE"
    engine = _engine(symbol)
    engines.append(engine)
    engine.add_data(data)
    kw: dict[str, Any] = {"instrument_id": iid, "command_release": release, "entry_order": "post_only"}
    if release == "bar":
        kw["command_release_bar_type"] = f"{iid}-1-MINUTE-LAST-EXTERNAL"
    kw.update(config)
    strat = strategy_cls(config_cls(**kw))
    engine.add_strategy(strat)
    engine.run()
    return engine, strat


def _entries(engine: BacktestEngine) -> list[Any]:
    return [o for o in engine.cache.orders() if not o.is_reduce_only]


def _protective(engine: BacktestEngine) -> list[Any]:
    return [o for o in engine.cache.orders() if o.is_reduce_only]


# Long signal on 5m bar 2 (close 101 > 100.5; SMA(2) ready); the rest sit at 100.
_SIGNAL_CLOSES = [100.0, 100.0, 101.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
_REL = _first_release(2)  # first 1m bar after the signal bar closes


# ---------------------------------------------------------------------------
# 1. Priced at release, not at the signal close
# ---------------------------------------------------------------------------


def test_post_only_priced_at_release_not_rejected_on_first_minute_dip(
    engines: list[BacktestEngine],
) -> None:
    # Signal bar closes at P=101.0; the release 1m bar (first minute after it) closes
    # 5 ticks lower at 100.5 -> the limit is release - 1 tick = P - 6 ticks = 100.4.
    # A limit priced at the signal close - 1 tick (100.9) is marketable against the
    # book (bid = ask = 100.5) and NT rejects post-only orders that would take.
    one_m = _one_minute(_REL + 10, ohlc={i: (100.5,) * 4 for i in range(_REL, _REL + 10)})
    engine, _ = _run(engines, data=_strategy_bars(_SIGNAL_CLOSES[:4]) + one_m, dsl=_dsl("mk_first_dip"))
    (order,) = _entries(engine)
    assert order.status != OrderStatus.REJECTED
    assert order.is_open and order.status == OrderStatus.ACCEPTED
    assert order.order_type == OrderType.LIMIT
    assert order.is_post_only and order.time_in_force == TimeInForce.GTC
    assert order.side == OrderSide.BUY
    assert float(order.price) == pytest.approx(100.5 - _TICK)
    assert order.ts_init == (_REL + 1) * _M  # placed on the release bar, not the signal bar


def test_offset_ticks_and_short_side_price_away_from_release(engines: list[BacktestEngine]) -> None:
    dsl = _dsl("mk_short", entry_conditions={"short": ["close > 100.5"]}, exit_conditions={"short": ["close < 50"]})
    one_m = _one_minute(_REL + 10, ohlc={i: (100.5,) * 4 for i in range(_REL, _REL + 10)})
    engine, _ = _run(
        engines, data=_strategy_bars(_SIGNAL_CLOSES[:4]) + one_m, dsl=dsl, maker_offset_ticks=3
    )
    (order,) = _entries(engine)
    assert order.side == OrderSide.SELL and order.is_open
    assert float(order.price) == pytest.approx(100.5 + 3 * _TICK)


def test_immediate_mode_prices_from_current_bar_close(engines: list[BacktestEngine]) -> None:
    # command_release "": sent on the signal bar itself, priced from that bar's close.
    engine, _ = _run(
        engines,
        data=_strategy_bars(_SIGNAL_CLOSES[:4]) + _one_minute(30),
        dsl=_dsl("mk_immediate"),
        release="",
    )
    (order,) = _entries(engine)
    assert order.ts_init == 3 * _M5  # the signal bar's close
    assert float(order.price) == pytest.approx(101.0 - _TICK)


# ---------------------------------------------------------------------------
# 2. Fills as maker at the limit
# ---------------------------------------------------------------------------


def test_post_only_entry_fills_as_maker_at_limit(engines: list[BacktestEngine]) -> None:
    ohlc = {i: (100.5,) * 4 for i in range(_REL, _REL + 10)}
    ohlc[_REL + 1] = (100.5, 100.5, 100.3, 100.4)  # trades through the 100.4 limit
    one_m = _one_minute(_REL + 10, ohlc=ohlc)
    engine, _ = _run(engines, data=_strategy_bars(_SIGNAL_CLOSES[:4]) + one_m, dsl=_dsl("mk_fill"))
    (order,) = _entries(engine)
    assert order.status == OrderStatus.FILLED
    (fill,) = [e for e in order.events if type(e).__name__ == "OrderFilled"]
    assert fill.liquidity_side == LiquiditySide.MAKER
    assert float(fill.last_px) == pytest.approx(100.4)
    inst = engine.cache.instrument(order.instrument_id)
    notional = float(fill.last_qty) * 100.4
    assert float(fill.commission) == pytest.approx(notional * float(inst.maker_fee), rel=1e-6)
    assert float(fill.commission) < notional * float(inst.taker_fee)


# ---------------------------------------------------------------------------
# 3. Timeout cancels, no fee
# ---------------------------------------------------------------------------


def test_post_only_unfilled_cancelled_after_n_bars_no_fee(engines: list[BacktestEngine]) -> None:
    # Placed on the release bar of signal bar 2; ages on primary bars 3, 4, 5.
    one_m = _one_minute(60, ohlc={i: (100.5,) * 4 for i in range(_REL, 60)})
    engine, strat = _run(
        engines,
        data=_strategy_bars(_SIGNAL_CLOSES) + one_m,
        dsl=_dsl("mk_timeout"),
        maker_entry_timeout_bars=3,
    )
    (order,) = _entries(engine)  # no re-entry: only bar 2 signals
    assert order.status == OrderStatus.CANCELED
    assert order.filled_qty == 0
    assert not [e for e in order.events if type(e).__name__ == "OrderFilled"]
    assert not engine.cache.positions()
    assert not _protective(engine)
    # cancel queued at the 3rd primary bar after placement (bar 5, ts 6*M5), sent on the next 1m bar
    assert order.ts_last == 6 * _M5 + _M
    assert strat._outbox == []
    account = engine.cache.account_for_venue(Venue("BINANCE"))
    assert float(account.balance_total(Currency.from_str("USDT"))) == 1_000_000.0  # no fee


def test_order_survives_until_timeout(engines: list[BacktestEngine]) -> None:
    # Same run with timeout 7: still resting when the data ends (bars 3..8 = 6 ages < 7).
    one_m = _one_minute(60, ohlc={i: (100.5,) * 4 for i in range(_REL, 60)})
    engine, _ = _run(
        engines,
        data=_strategy_bars(_SIGNAL_CLOSES) + one_m,
        dsl=_dsl("mk_timeout_long"),
        maker_entry_timeout_bars=7,
    )
    (order,) = _entries(engine)
    assert order.status == OrderStatus.ACCEPTED


# ---------------------------------------------------------------------------
# 4. Exit and opposite signals withdraw a resting / queued entry
# ---------------------------------------------------------------------------


def test_exit_signal_cancels_resting_entry(engines: list[BacktestEngine]) -> None:
    # Bar 3 closes 100.0 < 100.2: the long exit fires while the entry rests, no position.
    one_m = _one_minute(60, ohlc={i: (100.5,) * 4 for i in range(_REL, 60)})
    engine, _ = _run(
        engines,
        data=_strategy_bars(_SIGNAL_CLOSES) + one_m,
        dsl=_dsl("mk_exit_rest", exit_conditions={"long": ["close < 100.2"]}),
        maker_entry_timeout_bars=9,
    )
    (order,) = _entries(engine)
    assert order.status == OrderStatus.CANCELED
    # withdrawn by bar 3 (ts 4*M5), long before the 9-bar timeout
    assert order.ts_last == 4 * _M5 + _M
    assert not engine.cache.positions()


def test_exit_signal_drops_queued_entry_intent(engines: list[BacktestEngine]) -> None:
    # trade_tick release with no tick between the signal and the exit signal: the
    # intent is still queued at bar 3, the exit removes it, a late tick places nothing.
    late_tick = TradeTick(
        instrument_id=create_instrument("BTCUSDT").id,
        price=Price.from_str("100.5"),
        size=Quantity.from_str("1000000.000"),
        aggressor_side=AggressorSide.BUYER,
        trade_id=TradeId("T-late"),
        ts_event=5 * _M5,
        ts_init=5 * _M5,
    )
    engine, strat = _run(
        engines,
        data=_strategy_bars(_SIGNAL_CLOSES[:5]) + [late_tick],
        dsl=_dsl("mk_exit_queued", exit_conditions={"long": ["close < 100.2"]}),
        release="trade_tick",
    )
    assert _entries(engine) == []
    assert strat._outbox == []
    assert not strat._has_pending_entry()


def test_opposite_signal_cancels_resting_entry(engines: list[BacktestEngine]) -> None:
    dsl = _dsl(
        "mk_opposite",
        entry_conditions={"long": ["close > 100.5"], "short": ["close < 99.5"]},
        exit_conditions={"long": ["close < 50"], "short": ["close > 150"]},
    )
    closes = [100.0, 100.0, 101.0, 99.0, 100.0, 100.0, 100.0, 100.0]  # long, then short signal
    one_m = _one_minute(50, ohlc={i: (100.5,) * 4 for i in range(_REL, 50)})
    engine, _ = _run(
        engines, data=_strategy_bars(closes) + one_m, dsl=dsl, maker_entry_timeout_bars=9
    )
    (order,) = _entries(engine)  # the short signal skipped its bar: no second entry
    assert order.side == OrderSide.BUY
    assert order.status == OrderStatus.CANCELED
    assert order.ts_last == 4 * _M5 + _M  # withdrawn on bar 3's release minute


# ---------------------------------------------------------------------------
# 5. Race: a fill on the cancel's release tick still gets SL/TP
# ---------------------------------------------------------------------------


def test_entry_fill_on_cancel_release_tick_gets_sl_tp(engines: list[BacktestEngine]) -> None:
    cancel_release = 6 * 5  # bar 5 closes at 6*_M5 -> first 1m bar after is index 30
    ohlc = {i: (100.5,) * 4 for i in range(_REL, 60)}
    # The 1m bar that releases the timeout cancel trades through the 100.4 limit: the
    # engine fills the entry while processing that bar, BEFORE the strategy flushes
    # the queued cancel (which then finds a closed order).
    ohlc[cancel_release] = (100.5, 100.5, 100.3, 100.4)
    engine, _ = _run(
        engines,
        data=_strategy_bars(_SIGNAL_CLOSES) + _one_minute(60, ohlc=ohlc),
        dsl=_dsl("mk_race"),
        maker_entry_timeout_bars=3,
    )
    (order,) = _entries(engine)
    assert order.status == OrderStatus.FILLED
    assert order.ts_last == (cancel_release + 1) * _M
    (position,) = engine.cache.positions()
    assert position.is_open
    protective = _protective(engine)
    assert {o.order_type for o in protective} == {OrderType.STOP_MARKET, OrderType.LIMIT}
    assert all(o.is_open and o.is_reduce_only for o in protective)
    assert all(o.quantity == position.quantity for o in protective)


# ---------------------------------------------------------------------------
# 6. Market mode is unchanged
# ---------------------------------------------------------------------------

_MARKET_LONG_ENTRY = '''\
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=qty,
            time_in_force=TimeInForce.IOC,
        )
        self._send_command(self.submit_order, order)
        # SL/TP orders are submitted from on_event(PositionOpened) after fill
'''
_MARKET_SHORT_ENTRY = _MARKET_LONG_ENTRY.replace("OrderSide.BUY", "OrderSide.SELL")
_MARKET_EXIT = '''\
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=exit_side,
            quantity=quantity,
            time_in_force=TimeInForce.IOC,
            reduce_only=True,
        )
        self._send_command(self.submit_order, order)
'''


def test_market_mode_generated_source_unchanged(engines: list[BacktestEngine]) -> None:
    src = StrategyCompiler().compile(validate_strategy_dict(_dsl("mk_market_src")))
    # market order construction is textually the pre-change code (maker code only ADDS
    # a post_only branch that returns before it)
    for block in (_MARKET_LONG_ENTRY, _MARKET_SHORT_ENTRY, _MARKET_EXIT):
        assert block in src, block
    assert 'entry_order: str = "market"' in src
    assert "maker_offset_ticks: int = 1" in src
    assert "maker_entry_timeout_bars: int = 3" in src

    # behaviour: the default config places a market IOC entry, never a limit
    strategy_cls, config_cls = _compiled(_dsl("mk_market_run"))
    engine = _engine()
    engines.append(engine)
    engine.add_data(_strategy_bars(_SIGNAL_CLOSES[:4]) + _one_minute(30))
    cfg = config_cls(
        instrument_id=_BTC, command_release="bar", command_release_bar_type=f"{_BTC}-1-MINUTE-LAST-EXTERNAL"
    )
    assert cfg.entry_order == "market"
    engine.add_strategy(strategy_cls(cfg))
    engine.run()
    (order,) = _entries(engine)
    assert order.order_type == OrderType.MARKET and order.time_in_force == TimeInForce.IOC
    assert order.status == OrderStatus.FILLED
    assert order.ts_last == (_REL + 1) * _M  # released on the first 1m bar, at its close


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"entry_order": "postonly"}, "entry_order"),
        ({"maker_offset_ticks": 0}, "maker_offset_ticks"),
        ({"maker_entry_timeout_bars": 0}, "maker_entry_timeout_bars"),
    ],
    ids=["typo_mode", "zero_offset", "zero_timeout"],
)
def test_invalid_maker_config_raises_at_start(
    engines: list[BacktestEngine], overrides: dict[str, Any], match: str
) -> None:
    strategy_cls, config_cls = _compiled(_dsl("mk_guard"))
    engine = _engine()
    engines.append(engine)
    cfg = config_cls(instrument_id=_BTC, **overrides)
    if "entry_order" not in overrides:
        cfg = config_cls(instrument_id=_BTC, entry_order="post_only", **overrides)
    strat = strategy_cls(cfg)
    engine.add_strategy(strat)
    with pytest.raises(ValueError, match=match):
        strat.on_start()


# ---------------------------------------------------------------------------
# 7. Release price is the releasing datum's, in the instrument's own tick
# ---------------------------------------------------------------------------


def _tick(minute: int, price: float, tid: str) -> TradeTick:
    return TradeTick(
        instrument_id=create_instrument("BTCUSDT").id,
        price=Price.from_str(f"{price:.1f}"),
        size=Quantity.from_str("1000000.000"),
        aggressor_side=AggressorSide.BUYER,
        trade_id=TradeId(tid),
        ts_event=minute * _M,
        ts_init=minute * _M,
    )


def test_trade_tick_release_prices_from_the_tick(engines: list[BacktestEngine]) -> None:
    # signal close 101.0, own trade tick at 100.5 releases -> limit 100.4 (not 100.9)
    engine, _ = _run(
        engines,
        data=_strategy_bars(_SIGNAL_CLOSES[:3]) + [_tick(16, 100.5, "T-px")],
        dsl=_dsl("mk_tick_px"),
        release="trade_tick",
    )
    (order,) = _entries(engine)
    assert order.status == OrderStatus.ACCEPTED
    assert float(order.price) == pytest.approx(100.4)


def test_price_uses_the_instruments_own_tick_size(engines: list[BacktestEngine]) -> None:
    # ETHUSDT tick is 0.01 (BTC 0.1): release close 100.55, 2 ticks -> 100.53
    eth = "ETHUSDT-PERP.BINANCE"
    five = [_bar("5-MINUTE", _M5, i, c, c, c, c, iid=eth, prec=2) for i, c in enumerate(_SIGNAL_CLOSES[:4])]
    one_m = [
        _bar("1-MINUTE", _M, i, *((100.55 if i >= _REL else 100.0,) * 4), iid=eth, prec=2)
        for i in range(_REL + 10)
    ]
    engine, _ = _run(
        engines, data=five + one_m, dsl=_dsl("mk_eth_tick"), symbol="ETHUSDT", maker_offset_ticks=2
    )
    (order,) = _entries(engine)
    assert order.status == OrderStatus.ACCEPTED
    assert float(order.price) == pytest.approx(100.53)


# ---------------------------------------------------------------------------
# 8. Rejected post-only entry: no residue, the next signal enters
# ---------------------------------------------------------------------------


def test_rejected_post_only_entry_clears_pending_and_next_signal_enters(
    engines: list[BacktestEngine],
) -> None:
    def marketable_first(cls: type) -> type:
        class Marketable(cls):  # type: ignore[misc, valid-type]
            first = True

            def _place_entry_intent(self, side: Any, qty: Any) -> None:
                if self.first:  # price the first limit THROUGH the book: NT rejects post-only
                    self.first = False
                    self._release_px = float(self._release_px) + 5 * _TICK
                super()._place_entry_intent(side, qty)

        return Marketable

    closes = [100.0, 100.0, 101.0, 100.0, 100.0, 101.0, 100.0, 100.0, 100.0]
    one_m = _one_minute(60, ohlc={i: (100.5,) * 4 for i in range(_REL, 60)})
    engine, strat = _run(
        engines,
        data=_strategy_bars(closes) + one_m,
        dsl=_dsl("mk_reject"),
        strategy_wrap=marketable_first,
        maker_entry_timeout_bars=9,
    )
    first, second = sorted(_entries(engine), key=lambda o: o.ts_init)
    assert first.status == OrderStatus.REJECTED
    assert not [e for e in first.events if type(e).__name__ == "OrderFilled"]
    assert second.status == OrderStatus.ACCEPTED  # the next signal (bar 5) could enter
    assert strat._has_pending_entry()  # ... and it is the second order that is pending
    account = engine.cache.account_for_venue(Venue("BINANCE"))
    assert float(account.balance_total(Currency.from_str("USDT"))) == 1_000_000.0  # no fee


# ---------------------------------------------------------------------------
# 9. Exit / opposite withdrawal gates the SAME bar's entry evaluation
# ---------------------------------------------------------------------------


def test_exit_signal_withdrawing_queued_entry_skips_same_bar_entry(engines: list[BacktestEngine]) -> None:
    # bar 2 queues a long intent; bar 3 has BOTH an entry signal (100.5 < 101 < 101.5)
    # and an exit signal (101 < 101.5): the exit withdraws the intent and the bar is
    # skipped, the entry must not be re-queued in the same pass. Bar 4 (102) has neither.
    engine, strat = _run(
        engines,
        data=_strategy_bars([100.0, 100.0, 101.0, 101.0, 102.0]) + [_tick(26, 100.5, "T-skip")],
        dsl=_dsl(
            "mk_exit_skip",
            entry_conditions={"long": ["close > 100.5", "close < 101.5"]},
            exit_conditions={"long": ["close < 101.5"]},
        ),
        release="trade_tick",
        maker_entry_timeout_bars=9,
    )
    assert _entries(engine) == []
    assert strat._outbox == []


def test_short_exit_signal_withdrawing_queued_entry_skips_same_bar_entry(
    engines: list[BacktestEngine],
) -> None:
    # Short-side mirror of the long case above: bar 2 queues a SELL intent; bar 3 has
    # BOTH a short entry signal (98.5 < 99 < 99.5) and a short exit signal (99 > 98.5):
    # the exit withdraws the intent and the bar is skipped, so the same pass must not
    # re-queue the entry it just withdrew. Bar 4 (98) has neither.
    engine, strat = _run(
        engines,
        data=_strategy_bars([100.0, 100.0, 99.0, 99.0, 98.0]) + [_tick(26, 100.5, "T-short-skip")],
        dsl=_dsl(
            "mk_short_exit_skip",
            entry_conditions={"short": ["close < 99.5", "close > 98.5"]},
            exit_conditions={"short": ["close > 98.5"]},
        ),
        release="trade_tick",
        maker_entry_timeout_bars=9,
    )
    assert _entries(engine) == []
    assert strat._outbox == []


def test_opposite_entry_and_exit_same_bar_skips_the_bar(engines: list[BacktestEngine]) -> None:
    dsl = _dsl(
        "mk_opp_exit",
        entry_conditions={"long": ["close > 100.5"], "short": ["close < 99.5"]},
        exit_conditions={"long": ["close < 99.5"], "short": ["close > 150"]},
    )
    engine, _ = _run(
        engines,
        data=_strategy_bars([100.0, 100.0, 101.0, 99.0, 100.0]) + [_tick(26, 100.5, "T-opp")],
        dsl=dsl,
        release="trade_tick",
        maker_entry_timeout_bars=9,
    )
    assert _entries(engine) == []  # the short signal on the exit bar was skipped, not queued


# ---------------------------------------------------------------------------
# 10. One cancel per order; partial maker fills keep ONE protective pair
# ---------------------------------------------------------------------------


def test_timeout_queues_one_cancel_per_order(engines: list[BacktestEngine]) -> None:
    # own ticks are sparse: the cancel stays queued across several primary bars
    engine, strat = _run(
        engines,
        data=_strategy_bars([100.0, 100.0, 101.0] + [100.5] * 6) + [_tick(16, 100.5, "T-sparse")],
        dsl=_dsl("mk_sparse"),
        release="trade_tick",
        maker_entry_timeout_bars=1,
    )
    (order,) = _entries(engine)
    cancels = [cmd for cmd in strat._outbox if cmd[1] == strat.cancel_order]
    assert order.status == OrderStatus.ACCEPTED  # the cancel never reached the venue (no tick)
    assert len(cancels) == 1 and cancels[0][2][0].client_order_id == order.client_order_id


def _partial_fill_run(engines: list[BacktestEngine], *, fill_minutes: set[int], timeout: int) -> BacktestEngine:
    """A 0.002 maker entry that fills 0.001 per through-bar (thin bar volume)."""

    def two_lots(cls: type) -> type:
        class Lots(cls):  # type: ignore[misc, valid-type]
            def _calculate_position_size(self, bar: Any, is_long: bool = True) -> Any:
                return self.instrument.make_qty(0.002)

        return Lots

    data: list[Any] = _strategy_bars([100.0, 100.0, 101.0] + [100.5] * 3)
    for i in range(30):
        if i in fill_minutes:
            bar = _bar("1-MINUTE", _M, i, 100.3, 100.3, 100.3, 100.3)
            bar = Bar(
                bar_type=bar.bar_type, open=bar.open, high=bar.high, low=bar.low, close=bar.close,
                volume=Quantity.from_str("0.004"), ts_event=bar.ts_event, ts_init=bar.ts_init,
            )
        else:
            bar = _bar("1-MINUTE", _M, i, 100.5, 100.5, 100.5, 100.5)
        data.append(bar)
    engine, _ = _run(
        engines, data=data, dsl=_dsl("mk_partial"), strategy_wrap=two_lots, maker_entry_timeout_bars=timeout
    )
    return engine


@pytest.mark.parametrize(
    ("fill_minutes", "timeout"), [({16, 17}, 9), ({20}, 1)], ids=["two_fills", "fill_on_cancel_release"]
)
def test_partial_maker_fills_keep_one_protective_pair(
    engines: list[BacktestEngine], fill_minutes: set[int], timeout: int
) -> None:
    engine = _partial_fill_run(engines, fill_minutes=fill_minutes, timeout=timeout)
    (position,) = engine.cache.positions()
    assert position.quantity == Quantity.from_str("0.002")
    live = [o for o in _protective(engine) if o.is_open]
    assert sorted(o.order_type.name for o in live) == ["LIMIT", "STOP_MARKET"]
    assert all(o.quantity == position.quantity for o in live)
