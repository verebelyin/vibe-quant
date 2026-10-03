"""Mark-to-market max drawdown in screening + validation (bd vibe-quant-e70tl.11).

Validation used NT's daily realized-balance DD and screening a closed-trade
curve, so a 40% adverse excursion that recovered to entry reported DD 0.00025
instead of ~0.20.
"""

from __future__ import annotations

import time
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pytest
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import FillModel, MakerTakerFeeModel
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, TimeInForce
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Currency, Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

from tests.unit.test_audit_metrics_fills import IID
from vibe_quant.data.catalog import create_instrument, get_bar_type
from vibe_quant.metrics import (
    BarSeries,
    LedgerEvent,
    mark_to_market_max_drawdown,
    pnl_path,
)
from vibe_quant.nt_compat import retain_log_guard
from vibe_quant.screening.nt_runner import NTScreeningRunner
from vibe_quant.validation.extraction import extract_results, mark_to_market_drawdown
from vibe_quant.validation.venue import create_venue_config_for_validation

_MIN_NS = 60_000_000_000


def _series(rows: list[tuple[float, float, float, float]]) -> BarSeries:
    arr = np.array(rows, dtype=float)
    ts = np.array([(i + 1) * _MIN_NS - 1_000_000 for i in range(len(rows))], dtype=np.int64)
    return BarSeries(ts, arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])


def _flat(path: list[float], spread: float = 5.0) -> BarSeries:
    return _series([(p, p + spread, p - spread, p) for p in path])


def _dd(bars: BarSeries, events: list[LedgerEvent], balance: float = 1000.0) -> float:
    pc, pm = pnl_path(bars, events)
    return mark_to_market_max_drawdown(balance, [(bars.ts, pc, pm)])


def _mkt(bars: BarSeries, i: int, qty: float) -> LedgerEvent:
    """Market fill at bar i's close (NT bar execution, no latency)."""
    return LedgerEvent(ts=int(bars.ts[i]), qty=qty, price=float(bars.close[i]))


class TestPureMarkToMarket:
    def test_half_notional_long_40pct_excursion_recovered(self) -> None:
        """0.05 BTC (50% of 1000) long, 10000 -> 6000 -> 10000: DD ~0.20, not ~0."""
        bars = _flat([10000, 9000, 6000, 7000, 9000, 10000, 10000])
        events = [_mkt(bars, 0, 0.05), _mkt(bars, 5, -0.05)]
        # worst: low of the 6000 bar = 5995 -> 1000 + 0.05*(5995-10000) = 799.75
        assert _dd(bars, events) == pytest.approx(200.25 / 1000.0)

    def test_no_trades_zero_drawdown(self) -> None:
        bars = _flat([10000, 6000, 10000])
        assert _dd(bars, []) == 0.0
        assert mark_to_market_max_drawdown(1000.0, []) == 0.0

    def test_short_adverse_excursion_uses_highs(self) -> None:
        bars = _flat([10000, 12000, 14000, 12000, 10000, 10000])
        events = [_mkt(bars, 0, -0.05), _mkt(bars, 4, 0.05)]
        # worst: high of the 14000 bar = 14005 -> 1000 - 0.05*4005 = 799.75
        assert _dd(bars, events) == pytest.approx(200.25 / 1000.0)

    def test_long_ignores_highs_short_ignores_lows(self) -> None:
        bars = _series([(100, 100, 100, 100), (100, 150, 99, 100), (100, 100, 100, 100)])
        long_dd = _dd(bars, [_mkt(bars, 0, 1.0), _mkt(bars, 2, -1.0)])
        short_dd = _dd(bars, [_mkt(bars, 0, -1.0), _mkt(bars, 2, 1.0)])
        assert long_dd == pytest.approx(1.0 / 1000.0)
        assert short_dd == pytest.approx(50.0 / 1000.0)

    def test_stop_exit_marks_at_stop_not_bar_low(self) -> None:
        """SL resting fill at 9900 inside a bar whose low is 9000."""
        bars = _series([(10000, 10000, 10000, 10000), (10000, 10010, 9000, 9100)])
        events = [
            _mkt(bars, 0, 1.0),
            LedgerEvent(ts=int(bars.ts[1]), qty=-1.0, price=9900.0, resting=True),
        ]
        assert _dd(bars, events, balance=10_000.0) == pytest.approx(100.0 / 10_000.0)

    def test_market_exit_at_close_marks_whole_bar(self) -> None:
        """A signal exit filled at the bar close was held through the bar's low."""
        bars = _series([(10000, 10000, 10000, 10000), (10000, 10010, 9000, 9900)])
        events = [_mkt(bars, 0, 1.0), _mkt(bars, 1, -1.0)]
        assert _dd(bars, events, balance=10_000.0) == pytest.approx(1000.0 / 10_000.0)

    def test_tp_exit_marks_low_only_if_low_came_first(self) -> None:
        """Adaptive ordering: nearer extreme first. TP 10100 resting fill."""
        low_first = _series([(10000,) * 4, (10000, 10150, 9950, 10000)])  # |L-O| < |H-O|
        high_first = _series([(10000,) * 4, (10000, 10150, 9800, 10000)])  # |H-O| < |L-O|
        for bars, expected in ((low_first, 50.0), (high_first, 0.0)):
            events = [
                _mkt(bars, 0, 1.0),
                LedgerEvent(ts=int(bars.ts[1]), qty=-1.0, price=10100.0, resting=True),
            ]
            assert _dd(bars, events, balance=10_000.0) == pytest.approx(expected / 10_000.0)

    def test_fees_and_funding_reduce_equity(self) -> None:
        bars = _flat([100, 100, 100], spread=0.0)
        events = [
            LedgerEvent(ts=int(bars.ts[0]), qty=1.0, price=100.0, cash=-1.0),
            LedgerEvent(ts=int(bars.ts[1]) - 5, cash=-2.0),  # funding before bar 1
            LedgerEvent(ts=int(bars.ts[2]), qty=-1.0, price=100.0, cash=-1.0),
        ]
        assert _dd(bars, events, balance=100.0) == pytest.approx(0.04)

    def test_realized_pnl_matches_netting_accounting(self) -> None:
        bars = _flat([100, 110, 90, 95], spread=0.0)
        events = [_mkt(bars, 0, 2.0), _mkt(bars, 1, -1.0), _mkt(bars, 2, -1.0)]
        pc, _ = pnl_path(bars, events)
        assert pc[-1] == pytest.approx(10.0 - 10.0)  # +10 then -10

    def test_drawdown_capped_at_one(self) -> None:
        bars = _flat([100, 1, 1], spread=0.0)
        assert _dd(bars, [_mkt(bars, 0, 20.0)], balance=100.0) == 1.0

    def test_multi_instrument_combined(self) -> None:
        a = _flat([100, 90, 100], spread=0.0)
        b = _flat([100, 95, 100], spread=0.0)
        pa = pnl_path(a, [_mkt(a, 0, 1.0)])
        pb = pnl_path(b, [_mkt(b, 0, 1.0)])
        dd = mark_to_market_max_drawdown(1000.0, [(a.ts, *pa), (b.ts, *pb)])
        assert dd == pytest.approx(15.0 / 1000.0)


# ---------------------------------------------------------------------------
# Real NT engine: screening and validation extraction agree
# ---------------------------------------------------------------------------

BT = get_bar_type("BTCUSDT", "1m")


class _Cfg(StrategyConfig, frozen=True):
    bar_type: BarType
    script: str  # "bar:BUY|SELL,..." market orders


class _Script(Strategy):
    def __init__(self, config: _Cfg) -> None:
        super().__init__(config)
        self._actions = {
            int(k): v for k, v in (p.split(":") for p in config.script.split(",") if p)
        }
        self._i = -1

    def on_start(self) -> None:
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        self._i += 1
        action = self._actions.get(self._i)
        if action is None:
            return
        side = OrderSide.BUY if action == "BUY" else OrderSide.SELL
        inst = self.cache.instrument(IID)
        self.submit_order(
            self.order_factory.market(
                IID, side, inst.make_qty(Decimal("0.050")), time_in_force=TimeInForce.IOC
            )
        )


def _nt_bar(p: float, minute: int) -> Bar:
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


@pytest.fixture
def excursion_engine() -> BacktestEngine:
    """Long 0.05 BTC at 10000, crash to 6000, back to 10000, exit (audit repro)."""
    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    retain_log_guard(engine)
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(1_000, Currency.from_str("USDT"))],
        default_leverage=Decimal("10"),
        fill_model=FillModel(prob_fill_on_limit=1.0, prob_slippage=0.0, random_seed=42),
        fee_model=MakerTakerFeeModel(),
        bar_execution=True,
        bar_adaptive_high_low_ordering=True,
        use_position_ids=True,
        use_reduce_only=True,
    )
    engine.add_instrument(create_instrument("BTCUSDT"))
    path = [10000, 9000, 6000, 7000, 9000, 10000, 10000]
    engine.add_data([_nt_bar(p, i) for i, p in enumerate(path)])
    engine.add_strategy(_Script(_Cfg(bar_type=BT, script="0:BUY,5:SELL")))
    engine.run()
    yield engine
    engine.reset()
    engine.dispose()


def test_nt_engine_mtm_drawdown_matches_hand_computed(excursion_engine: BacktestEngine) -> None:
    dd = mark_to_market_drawdown(excursion_engine, 1000.0, execution_timeframe="1m")
    # entry fee 0.25 (taker 5bp of 500) then worst low 5995
    assert dd == pytest.approx((0.25 + 0.05 * (10000 - 5995)) / 1000.0, rel=1e-9)


def test_screening_and_validation_drawdown_agree(excursion_engine: BacktestEngine) -> None:
    bt_result = excursion_engine.get_result()
    runner = NTScreeningRunner(
        dsl_dict={}, symbols=["BTCUSDT"], start_date="", end_date="",
        funding_archive_path="/nonexistent/vq_test_funding_archive.db",
    )
    runner._all_timeframes = {"1m"}
    screening = runner._extract_metrics({}, bt_result, excursion_engine, time.time(), 1000.0)
    venue = create_venue_config_for_validation(latency_preset=None)
    # engine slippage "on" disables post-fill SPEC slippage: same costs as screening
    venue.fill_config = SimpleNamespace(impact_coefficient=0.1, prob_slippage=1.0)  # type: ignore[assignment]
    validation = extract_results(
        1, "x", bt_result, excursion_engine, venue, primary_timeframe="1m",
        execution_timeframe="1m",
    )
    assert screening.max_drawdown == pytest.approx(0.200_5, abs=1e-3)
    assert validation.max_drawdown == pytest.approx(screening.max_drawdown)
    # old behaviour: daily realized-balance / closed-trade DD ~0.0005
    assert validation.max_drawdown > 0.19


def test_catalog_bar_decoding_bit_identical_to_nt(tmp_path: object) -> None:
    """numpy parquet decode == NT Bar floats (incl. off-grid aggregated raws)."""
    from pathlib import Path

    from vibe_quant.data.catalog import CatalogManager
    from vibe_quant.validation.extraction import load_catalog_bars

    catalog = Path(str(tmp_path)) / "catalog"
    # minute >= 1: the catalog writer drops epoch-zero timestamps
    bars = [_nt_bar(10_000.0 + 0.1 * i + (0.07 if i % 3 else 0.0), i + 1) for i in range(50)]
    CatalogManager(catalog).write_bars(bars)
    series = load_catalog_bars(catalog, str(BT))
    assert series is not None
    assert series.ts.tolist() == [b.ts_init for b in bars]
    for name in ("open", "high", "low", "close"):
        assert getattr(series, name).tolist() == [float(getattr(b, name)) for b in bars]
    assert load_catalog_bars(catalog, str(BT)) is series  # cached per process
