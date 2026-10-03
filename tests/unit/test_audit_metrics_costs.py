"""Validation cost-accounting audit items (bd vibe-quant-e70tl.23 medium triage).

- Open position at end dropped from trades + return when latency is on: the
  on_stop flatten order is still in flight when the engine stops.
- Post-fill SPEC slippage was charged on the entry leg only.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import FillModel, LatencyModel, MakerTakerFeeModel
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, TimeInForce
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Currency, Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

from tests.unit.test_audit_metrics_fills import IID
from vibe_quant.data.catalog import create_instrument, get_bar_type
from vibe_quant.nt_compat import retain_log_guard
from vibe_quant.validation.extraction import (
    DEFAULT_AVG_BAR_VOLUME,
    DEFAULT_BAR_VOLATILITY,
    extract_results,
    extract_trades,
)
from vibe_quant.validation.fill_model import SlippageEstimator
from vibe_quant.validation.results import ValidationResult
from vibe_quant.validation.venue import create_venue_config_for_validation

BT = get_bar_type("BTCUSDT", "1m")
_MIN_NS = 60_000_000_000


class _Cfg(StrategyConfig, frozen=True):
    bar_type: BarType


class _EnterAndFlattenOnStop(Strategy):
    """Buys on the first bar; flattens in on_stop like the compiled DSL strategies."""

    def __init__(self, config: _Cfg) -> None:
        super().__init__(config)
        self._done = False

    def on_start(self) -> None:
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        if self._done:
            return
        self._done = True
        inst = self.cache.instrument(IID)
        self.submit_order(
            self.order_factory.market(
                IID, OrderSide.BUY, inst.make_qty(Decimal("0.050")), time_in_force=TimeInForce.IOC
            )
        )

    def on_stop(self) -> None:
        self.cancel_all_orders(IID)
        self.close_all_positions(IID)


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


@pytest.fixture
def latency_engine() -> BacktestEngine:
    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    retain_log_guard(engine)
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(1_000, Currency.from_str("USDT"))],
        default_leverage=Decimal("10"),
        fill_model=FillModel(prob_fill_on_limit=1.0, prob_slippage=0.0, random_seed=42),
        latency_model=LatencyModel(base_latency_nanos=60_000_000),
        fee_model=MakerTakerFeeModel(),
        bar_execution=True,
        use_position_ids=True,
        use_reduce_only=True,
    )
    engine.add_instrument(create_instrument("BTCUSDT"))
    engine.add_data([_bar(p, i) for i, p in enumerate([10000, 10100, 10200, 10300, 10400])])
    engine.add_strategy(_EnterAndFlattenOnStop(_Cfg(bar_type=BT)))
    engine.run()
    yield engine
    engine.reset()
    engine.dispose()


def test_open_position_at_end_is_marked_not_dropped(latency_engine: BacktestEngine) -> None:
    positions = list(latency_engine.cache.positions())
    assert len(positions) == 1 and positions[0].is_open  # the flatten never filled
    entry = float(positions[0].avg_px_open)  # latency: next bar close, 10100

    venue = create_venue_config_for_validation(latency_preset=None)
    # engine slippage "on" disables post-fill SPEC slippage: same costs as screening
    venue.fill_config = SimpleNamespace(impact_coefficient=0.1, prob_slippage=1.0)  # type: ignore[assignment]
    result = extract_results(
        1, "x", latency_engine.get_result(), latency_engine, venue,
        primary_timeframe="1m", execution_timeframe="1m",
    )
    assert result.total_trades == 1
    trade = result.trades[0]
    assert trade.exit_reason == "end_of_data"
    assert trade.exit_price == 10400.0
    unrealized = 0.05 * (10400.0 - entry)
    entry_fee = 0.05 * entry * 0.0005
    exit_fee = 0.05 * 10400.0 * 0.0005
    assert trade.net_pnl == pytest.approx(unrealized - entry_fee - exit_fee)
    # headline return now includes the open trade (NT's PnL% only had -entry_fee)
    assert result.total_return == pytest.approx((unrealized - entry_fee - exit_fee) / 1000.0)
    assert result.total_fees == pytest.approx(entry_fee + exit_fee)


class _Position:
    is_closed = True
    is_open = False
    realized_pnl = 100.0
    avg_px_open = 40_000.0
    avg_px_close = 40_100.0
    peak_qty = 0.1
    ts_opened = 1_700_000_000_000_000_000
    ts_closed = 1_700_000_360_000_000_000
    entry = "BUY"
    instrument_id = "BTCUSDT-PERP.BINANCE"
    events: list[object] = []

    def commissions(self) -> list[float]:
        return [2.0]


def test_post_fill_slippage_charged_on_both_legs() -> None:
    cache = SimpleNamespace(positions=lambda: [_Position()], position_snapshots=list, bars=list)
    engine = SimpleNamespace(kernel=SimpleNamespace(cache=cache))
    venue = SimpleNamespace(
        default_leverage=Decimal("10"),
        fill_config=SimpleNamespace(impact_coefficient=0.1, prob_slippage=0.0),
    )
    result = ValidationResult(starting_balance=100_000.0)
    extract_trades(result, engine, venue)  # type: ignore[arg-type]

    est = SlippageEstimator(impact_coefficient=0.1)
    factor = est.calculate(0.1, DEFAULT_AVG_BAR_VOLUME, DEFAULT_BAR_VOLATILITY, 0.0001)
    expected = factor * 0.1 * (40_000.0 + 40_100.0)
    assert result.trades[0].slippage_cost == pytest.approx(expected)
    entry_only = factor * 0.1 * 40_000.0
    assert result.trades[0].slippage_cost > 1.9 * entry_only
