"""vibe-quant-e70tl.4 (DSL runtime part): order management in compiled strategies.

The generated strategy runtime is shared by backtests AND paper/live, so:

(a) sizing never falls back to ``make_qty(1.0)``; equity is read in the
    settlement currency; min-qty clamp may not exceed max_position_pct;
    quantities round DOWN;
(b) no second entry while an entry order is in flight;
(c) exits are reduce-only (an exit racing a stop fill cannot open a reverse);
(d) SL/TP follow the full position on partial fills (PositionChanged);
(e) on_start adopts an open position and re-arms missing SL/TP;
(f) a trailing stop never loosens (first update included).
"""

from __future__ import annotations

import types
from decimal import Decimal
from typing import Any

import pytest
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import LatencyModel
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.currencies import BNB, BTC, USDT
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import (
    AccountType,
    OmsType,
    OrderSide,
    OrderType,
    PositionSide,
    TimeInForce,
)
from nautilus_trader.model.events import PositionOpened
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Currency, Money, Price, Quantity
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from vibe_quant.data.catalog import create_instrument
from vibe_quant.dsl.compiler import StrategyCompiler, _to_class_name
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.dsl.templates import ORDER_METHODS_LINES
from vibe_quant.nt_compat import retain_log_guard

_IID = "BTCUSDT-PERP.BINANCE"
_M = 60_000_000_000


# ---------------------------------------------------------------------------
# Engine harness
# ---------------------------------------------------------------------------


def _bar(bt: BarType, i: int, o: float, h: float, lo: float, c: float, vol: str) -> Bar:
    return Bar(
        bar_type=bt,
        open=Price.from_str(f"{o:.1f}"),
        high=Price.from_str(f"{h:.1f}"),
        low=Price.from_str(f"{lo:.1f}"),
        close=Price.from_str(f"{c:.1f}"),
        volume=Quantity.from_str(vol),
        ts_event=i * _M,
        ts_init=(i + 1) * _M - 1,
    )


def _engine(latency_ms: float | None = None, balance: int = 100_000) -> BacktestEngine:
    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    retain_log_guard(engine)
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(balance, Currency.from_str("USDT"))],
        default_leverage=Decimal("10"),
        latency_model=(
            LatencyModel(base_latency_nanos=int(latency_ms * 1_000_000)) if latency_ms else None
        ),
        bar_execution=True,
        use_reduce_only=True,
    )
    engine.add_instrument(create_instrument("BTCUSDT"))
    return engine


def _dsl(name: str, **overrides: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "name": name,
        "timeframe": "1m",
        "indicators": {"sma": {"type": "SMA", "period": 2}},
        "entry_conditions": {"long": ["close > 100"]},
        "exit_conditions": {"long": ["close < 100"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
    }
    d.update(overrides)
    return d


def _compiled(d: dict[str, Any]) -> tuple[type, type]:
    module = StrategyCompiler().compile_to_module(validate_strategy_dict(d))
    cn = _to_class_name(d["name"])
    return getattr(module, f"{cn}Strategy"), getattr(module, f"{cn}Config")


class TestEntryGuardAndReduceOnlyExit:
    """(b)+(c): 90s latency on 1m bars -- an order lands ~1.5 bars after submit."""

    @staticmethod
    def _run() -> tuple[Any, BacktestEngine]:
        strategy_cls, config_cls = _compiled(_dsl("c4_race"))

        class Probe(strategy_cls):  # type: ignore[misc, valid-type]
            def __init__(self, config: Any) -> None:
                super().__init__(config)
                self.opened_sides: list[PositionSide] = []

            def on_event(self, event: Any) -> None:
                if isinstance(event, PositionOpened):
                    self.opened_sides.append(event.side)
                super().on_event(event)

        bt = BarType.from_str(f"{_IID}-1-MINUTE-LAST-EXTERNAL")
        v = "100000.000"
        bars = [
            _bar(bt, 0, 101.0, 101.2, 100.8, 101.0, v),  # warmup (SMA 2)
            _bar(bt, 1, 101.0, 101.2, 100.8, 101.0, v),  # entry signal -> order A
            _bar(bt, 2, 101.0, 101.2, 100.8, 101.0, v),  # A in flight: no 2nd entry
            _bar(bt, 3, 100.5, 100.6, 100.4, 100.5, v),  # A fills -> SL/TP submitted
            _bar(bt, 4, 100.5, 100.6, 100.4, 100.5, v),
            _bar(bt, 5, 100.2, 100.3, 99.4, 99.5, v),  # exit signal -> reduce-only sell
            _bar(bt, 6, 99.0, 99.1, 97.0, 97.5, v),  # stop fills before the exit lands
            _bar(bt, 7, 97.0, 97.2, 96.8, 97.0, v),  # exit lands with no position
            _bar(bt, 8, 97.0, 97.2, 96.8, 97.0, v),
            _bar(bt, 9, 97.0, 97.2, 96.8, 97.0, v),
        ]
        engine = _engine(latency_ms=90_000)
        engine.add_data(bars)
        strat = Probe(config_cls(instrument_id=_IID))
        engine.add_strategy(strat)
        engine.run()
        return strat, engine

    def test_single_entry_and_no_reverse_position(self) -> None:
        strat, engine = self._run()
        orders = engine.cache.orders()
        entries = [o for o in orders if not o.is_reduce_only]
        assert len(entries) == 1, [str(o) for o in entries]  # no double entry
        # never a short position on a long-only strategy
        assert strat.opened_sides == [PositionSide.LONG]
        # the racing exit was reduce-only and did not fill
        exits = [o for o in orders if o.is_reduce_only and o.order_type == OrderType.MARKET]
        assert exits and all(float(o.filled_qty) == 0.0 for o in exits)
        engine.dispose()


class TestPartialFillResize:
    """(d): a 2-fill entry must leave SL and TP sized to the whole position."""

    def test_sl_tp_follow_full_position(self) -> None:
        strategy_cls, config_cls = _compiled(_dsl("c4_partial"))

        class Probe(strategy_cls):  # type: ignore[misc, valid-type]
            def __init__(self, config: Any) -> None:
                super().__init__(config)
                self.sent = False
                self.snapshots: list[tuple[float, list[tuple[OrderType, float]]]] = []

            def _submit_long_entry(self, bar: Bar) -> None:
                # Large GTC market vs thin bars -> NT fills it 0.001 per bar.
                if self.sent:
                    return
                self.sent = True
                self.submit_order(
                    self.order_factory.market(
                        instrument_id=self.instrument_id,
                        order_side=OrderSide.BUY,
                        quantity=self.instrument.make_qty(0.002),
                        time_in_force=TimeInForce.GTC,
                    )
                )

            def on_bar(self, bar: Bar) -> None:
                super().on_bar(bar)
                pos = [p for p in self.cache.positions_open() if p.is_open]
                if pos:
                    self.snapshots.append(
                        (
                            float(pos[0].quantity),
                            sorted(
                                (o.order_type, float(o.leaves_qty))
                                for o in self.cache.orders_open(instrument_id=self.instrument_id)
                                if o.is_reduce_only
                            ),
                        )
                    )

        bt = BarType.from_str(f"{_IID}-1-MINUTE-LAST-EXTERNAL")
        bars = [_bar(bt, i, 101.0, 101.1, 100.9, 101.0, "0.004") for i in range(8)]
        engine = _engine()
        engine.add_data(bars)
        strat = Probe(config_cls(instrument_id=_IID))
        engine.add_strategy(strat)
        engine.run()

        # SL/TP were first armed for the 0.001 partial fill, then replaced
        first_armed = sorted(
            float(o.quantity) for o in engine.cache.orders() if o.is_reduce_only and o.is_canceled
        )
        assert first_armed.count(0.001) == 2  # (on_stop later cancels the 0.002 pair)
        final_qty, protective = strat.snapshots[-1]
        assert final_qty == pytest.approx(0.002)
        assert protective == [(OrderType.LIMIT, 0.002), (OrderType.STOP_MARKET, 0.002)]
        engine.dispose()


# ---------------------------------------------------------------------------
# Fake-self harness for the template methods
# ---------------------------------------------------------------------------

_NS: dict[str, Any] = {
    "Bar": Bar,
    "Quantity": Quantity,
    "OrderSide": OrderSide,
    "OrderType": OrderType,
    "PositionSide": PositionSide,
    "TimeInForce": TimeInForce,
}
exec("\n".join(ORDER_METHODS_LINES), _NS)  # noqa: S102

_BTC = TestInstrumentProvider.btcusdt_perp_binance()  # size inc 0.001, min qty 0.001, min notional 10


class _Log:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)

    info = warning


class _Acct:
    def __init__(self, bals: dict[Currency, float]) -> None:
        self.bals = bals

    def currencies(self) -> list[Currency]:
        return list(self.bals)

    def balance_total(self, ccy: Currency) -> Money | None:
        return Money(self.bals[ccy], ccy) if ccy in self.bals else None


def _fake(**attrs: Any) -> types.SimpleNamespace:
    """SimpleNamespace with every template method bound as a method."""
    fake = types.SimpleNamespace(log=_Log(), **attrs)
    for name, fn in _NS.items():
        if callable(fn) and getattr(fn, "__module__", None) is None and name.startswith("_"):
            setattr(fake, name, types.MethodType(fn, fake))
    return fake


def _sizer(account: _Acct | None, *, sl_pct: float = 1.0, max_pos: float = 0.5) -> types.SimpleNamespace:
    cfg = types.SimpleNamespace(
        risk_per_trade=0.02,
        max_position_pct=max_pos,
        stop_loss_type="fixed_pct",
        stop_loss_percent=sl_pct,
    )
    return _fake(
        config=cfg,
        instrument=_BTC,
        instrument_id=_BTC.id,
        cache=types.SimpleNamespace(account_for_venue=lambda venue: account),
    )


_BAR = types.SimpleNamespace(close=Price.from_str("60000.0"))


class TestSizing:
    def test_no_account_skips_entry(self) -> None:
        s = _sizer(None)
        assert s._calculate_position_size(_BAR, True) is None  # was make_qty(1.0) = 1 BTC
        assert "no account" in s.log.warnings[0]

    @pytest.mark.parametrize("bals", [{USDT: 0.0}, {BNB: 5.0, USDT: 0.0}, {BTC: 1.0}])
    def test_no_settlement_equity_skips_entry(self, bals: dict[Currency, float]) -> None:
        s = _sizer(_Acct(bals))
        assert s._calculate_position_size(_BAR, True) is None

    def test_bnb_first_wallet_sizes_from_usdt(self) -> None:
        # 10k USDT, risk 2% = 200, stop 1% of 60k = 600 -> 0.333 BTC; cap 5k/60k = 0.08333
        s = _sizer(_Acct({BNB: 0.05, USDT: 10_000.0}))
        assert s._calculate_position_size(_BAR, True) == Quantity.from_str("0.083")

    def test_rounds_down(self) -> None:
        # cap 0.5 * 1050 / 60000 = 0.00875 -> 0.008 (round-half-even would give 0.009)
        s = _sizer(_Acct({USDT: 1050.0}))
        assert s._calculate_position_size(_BAR, True) == Quantity.from_str("0.008")

    def test_min_qty_clamp_never_exceeds_max_position_pct(self) -> None:
        # equity 50: cap 25/60000 = 0.0004 BTC < min qty 0.001 -> skip (was 0.001 = 1.2x equity)
        s = _sizer(_Acct({USDT: 50.0}))
        assert s._calculate_position_size(_BAR, True) is None
        assert "max_position_pct" in s.log.warnings[0]

    def test_min_qty_clamp_within_cap_allowed(self) -> None:
        # equity 200, max_pos 1.0: cap 0.00333; risk sizing (tiny stop budget) below
        # min qty -> clamped up to 0.001 (within cap)
        s = _sizer(_Acct({USDT: 200.0}), sl_pct=50.0, max_pos=1.0)
        assert s._calculate_position_size(_BAR, True) == Quantity.from_str("0.001")

    def test_below_min_notional_skips(self) -> None:
        cheap = types.SimpleNamespace(close=Price.from_str("5000.0"))
        # equity 30 -> cap 15/5000 = 0.003 BTC, notional 15 >= 10 ok; equity 15 -> 0.0015 -> 0.001 -> 5 < 10
        assert _sizer(_Acct({USDT: 30.0}))._calculate_position_size(cheap, True) == Quantity.from_str("0.003")
        s = _sizer(_Acct({USDT: 15.0}))
        assert s._calculate_position_size(cheap, True) is None


class _Price:
    def __init__(self, v: float) -> None:
        self.v = v

    def as_double(self) -> float:
        return self.v


def _trail_fake() -> tuple[types.SimpleNamespace, list[dict[str, Any]]]:
    submitted: list[dict[str, Any]] = []
    iid = types.SimpleNamespace(venue="BINANCE")
    pos = types.SimpleNamespace(instrument_id=iid, is_open=True, quantity=1)
    cfg = types.SimpleNamespace(
        stop_loss_type="atr_trailing",
        stop_loss_indicator="atr",
        stop_loss_atr_multiplier=2.0,
        take_profit_type="fixed_pct",
        take_profit_percent=50.0,
    )
    fake = _fake(
        config=cfg,
        _position_open=True,
        _position_side=OrderSide.BUY,
        _trailing_best_sl=None,
        _get_indicator_value=lambda n: 2.0,
        instrument_id=iid,
        instrument=types.SimpleNamespace(make_price=lambda p: p),
        cache=types.SimpleNamespace(orders_open=lambda venue: [], positions_open=lambda venue: [pos]),
        order_factory=types.SimpleNamespace(stop_market=lambda **kw: kw, limit=lambda **kw: kw),
        submit_order=submitted.append,
        cancel_order=lambda o: None,
    )
    return fake, submitted


class TestTrailingStop:
    def test_first_update_never_loosens(self) -> None:
        fake, submitted = _trail_fake()
        # Entry 100, ATR 2, mult 2 -> initial SL 96 submitted on PositionOpened
        fake._submit_sl_tp_orders(100.0, OrderSide.BUY, 1)
        assert submitted[0]["trigger_price"] == 96.0
        assert fake._trailing_best_sl == 96.0
        # Next close 98 -> candidate 94 is looser: no resubmit
        fake._update_trailing_stop(types.SimpleNamespace(close=_Price(98.0)))
        assert [o["trigger_price"] for o in submitted if "trigger_price" in o] == [96.0]
        # Close 103 -> 99 tightens
        fake._update_trailing_stop(types.SimpleNamespace(close=_Price(103.0)))
        assert submitted[-1]["trigger_price"] == 99.0
        assert fake._trailing_best_sl == 99.0

    def test_short_trail_never_loosens(self) -> None:
        fake, submitted = _trail_fake()
        fake._position_side = OrderSide.SELL
        fake._submit_sl_tp_orders(100.0, OrderSide.SELL, 1)  # SL 104
        fake._update_trailing_stop(types.SimpleNamespace(close=_Price(102.0)))  # 106: looser
        assert [o["trigger_price"] for o in submitted if "trigger_price" in o] == [104.0]
        fake._update_trailing_stop(types.SimpleNamespace(close=_Price(97.0)))  # 101: tighter
        assert submitted[-1]["trigger_price"] == 101.0

    def test_rearm_keeps_tighter_trail(self) -> None:
        fake, submitted = _trail_fake()
        fake._trailing_best_sl = 99.0  # trail already moved up
        fake._submit_sl_tp_orders(100.0, OrderSide.BUY, 1)  # fresh calc would be 96
        assert submitted[0]["trigger_price"] == 99.0


class TestRestartRecovery:
    def _fake_with_position(self, protective: list[Any], ready: bool, sl_type: str = "fixed_pct") -> Any:
        submitted: list[dict[str, Any]] = []
        iid = _BTC.id
        pos = types.SimpleNamespace(
            instrument_id=iid,
            is_open=True,
            quantity=Quantity.from_str("0.010"),
            side=PositionSide.LONG,
            avg_px_open=Price.from_str("60000.0"),
        )
        cfg = types.SimpleNamespace(
            stop_loss_type=sl_type,
            stop_loss_percent=1.0,
            stop_loss_indicator="atr",
            stop_loss_atr_multiplier=2.0,
            take_profit_type="fixed_pct",
            take_profit_percent=2.0,
        )
        fake = _fake(
            config=cfg,
            id="S-000",
            instrument_id=iid,
            instrument=_BTC,
            _position_open=False,
            _position_side=None,
            _trailing_best_sl=None,
            _rearm_protection=False,
            _indicators_ready=lambda: ready,
            _get_indicator_value=lambda n: 100.0,
            cache=types.SimpleNamespace(
                positions_open=lambda venue=None, instrument_id=None: [pos],
                orders_open=lambda instrument_id=None, strategy_id=None, venue=None: list(protective),
                orders_inflight=lambda instrument_id=None, strategy_id=None: [],
            ),
            order_factory=types.SimpleNamespace(stop_market=lambda **kw: kw, limit=lambda **kw: kw),
            submit_order=submitted.append,
            cancel_order=lambda o: None,
        )
        fake.submitted = submitted
        return fake

    def _on_start_recovery(self, fake: Any) -> None:
        from vibe_quant.dsl.templates import ON_START_RECOVERY_LINES

        body = "\n".join(f"    {line}" for line in ON_START_RECOVERY_LINES)
        ns: dict[str, Any] = {}
        exec(f"def _recover(self):\n{body}\n", ns)  # noqa: S102
        ns["_recover"](fake)

    def test_unprotected_position_rearmed_on_start(self) -> None:
        fake = self._fake_with_position([], ready=False)
        self._on_start_recovery(fake)
        assert fake._position_open and fake._position_side == OrderSide.BUY
        prices = sorted(float(o.get("trigger_price", o.get("price"))) for o in fake.submitted)
        assert prices == [59400.0, 61200.0]  # SL -1%, TP +2% on avg_px_open
        assert all(o["quantity"] == Quantity.from_str("0.010") and o["reduce_only"] for o in fake.submitted)
        assert not fake._rearm_protection

    def test_atr_levels_wait_for_indicators(self) -> None:
        fake = self._fake_with_position([], ready=False, sl_type="atr_fixed")
        self._on_start_recovery(fake)
        assert fake._position_open
        assert fake.submitted == []  # ATR not warm: an entry-price stop would be wrong
        assert fake._rearm_protection
        fake._indicators_ready = lambda: True
        assert fake._ensure_protection()
        assert len(fake.submitted) == 2 and not fake._rearm_protection

    def test_existing_protection_not_duplicated(self) -> None:
        sl = types.SimpleNamespace(
            client_order_id="O-1",
            is_pending_cancel=False,
            is_reduce_only=True,
            order_type=OrderType.STOP_MARKET,
            leaves_qty=Quantity.from_str("0.010"),
            trigger_price=Price.from_str("59000.0"),
        )
        fake = self._fake_with_position([sl], ready=True)
        self._on_start_recovery(fake)
        assert fake._position_open
        assert fake.submitted == []
