"""Validation fill model: resting limits / triggered stops fill at their price (e70tl.2).

``VolumeSlippageFillModel`` used to hand NT a synthetic book for EVERY order, so
resting TP limits and triggered SL stops filled at the bar extreme (long TP
limit 10100 with bar high 10300 -> 10300; SL stop 9900 with bar low 9700 ->
9700). Limits must never fill better than their limit price; triggered stops
fill at the trigger (+ the configured adverse ticks only), gaps at the open;
market orders keep their adverse degradation.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import FillModel, MakerTakerFeeModel
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, TimeInForce
from nautilus_trader.model.events import OrderFilled, PositionOpened
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.objects import Currency, Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

from vibe_quant.data.catalog import create_instrument, get_bar_type
from vibe_quant.nt_compat import retain_log_guard
from vibe_quant.validation.fill_model import (
    VolumeSlippageFillModel,
    VolumeSlippageFillModelConfig,
)

IID = InstrumentId(Symbol("BTCUSDT-PERP"), Venue("BINANCE"))
BT = get_bar_type("BTCUSDT", "1m")
TICK = 0.1


def _bar(o: float, h: float, lo: float, c: float, minute: int) -> Bar:
    return Bar(
        bar_type=BT,
        open=Price.from_str(f"{o:.1f}"),
        high=Price.from_str(f"{h:.1f}"),
        low=Price.from_str(f"{lo:.1f}"),
        close=Price.from_str(f"{c:.1f}"),
        volume=Quantity.from_str("100.000"),
        ts_event=minute * 60_000_000_000,
        ts_init=(minute * 60_000 + 59_999) * 1_000_000,
    )


class _Cfg(StrategyConfig, frozen=True):
    bar_type: BarType
    side: str = "BUY"
    tp_pct: float = 0.0
    sl_pct: float = 0.0
    entry_bar: int = 0


class _Bracket(Strategy):
    """Market entry on ``entry_bar``; reduce-only TP limit / SL stop after the fill."""

    def __init__(self, config: _Cfg) -> None:
        super().__init__(config)
        self.fills: list[tuple[str, str, float]] = []
        self._bar_idx = -1

    def on_start(self) -> None:
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        self._bar_idx += 1
        if self._bar_idx != self.config.entry_bar:
            return
        side = OrderSide.BUY if self.config.side == "BUY" else OrderSide.SELL
        inst = self.cache.instrument(IID)
        order = self.order_factory.market(
            IID, side, inst.make_qty(Decimal("0.010")), time_in_force=TimeInForce.IOC
        )
        self.submit_order(order)

    def on_event(self, event: object) -> None:
        if isinstance(event, PositionOpened):
            pos = self.cache.position(event.position_id)
            inst = self.cache.instrument(IID)
            entry = float(pos.avg_px_open)
            long = pos.is_long
            xside = OrderSide.SELL if long else OrderSide.BUY
            if self.config.tp_pct:
                tp = entry * (1 + self.config.tp_pct / 100 * (1 if long else -1))
                self.submit_order(
                    self.order_factory.limit(
                        IID, xside, pos.quantity, inst.make_price(tp),
                        time_in_force=TimeInForce.GTC, reduce_only=True,
                    )
                )
            if self.config.sl_pct:
                sl = entry * (1 - self.config.sl_pct / 100 * (1 if long else -1))
                self.submit_order(
                    self.order_factory.stop_market(
                        IID, xside, pos.quantity, inst.make_price(sl),
                        time_in_force=TimeInForce.GTC, reduce_only=True,
                    )
                )

    def on_order_filled(self, event: OrderFilled) -> None:
        self.fills.append(
            (event.order_type.name, event.order_side.name, float(event.last_px))
        )


def _run(
    fill_model: FillModel, bars: list[Bar], *, side: str = "BUY", tp: float = 0.0,
    sl: float = 0.0, entry_bar: int = 0, adaptive: bool = False,
) -> list[tuple[str, str, float]]:
    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    retain_log_guard(engine)
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(1_000, Currency.from_str("USDT"))],
        default_leverage=Decimal("10"),
        fill_model=fill_model,
        fee_model=MakerTakerFeeModel(),
        bar_execution=True,
        reject_stop_orders=False,
        use_position_ids=True,
        use_reduce_only=True,
        bar_adaptive_high_low_ordering=adaptive,
    )
    engine.add_instrument(create_instrument("BTCUSDT"))
    engine.add_data(bars)
    strategy = _Bracket(_Cfg(bar_type=BT, side=side, tp_pct=tp, sl_pct=sl, entry_bar=entry_bar))
    engine.add_strategy(strategy)
    try:
        engine.run()
        return list(strategy.fills)
    finally:
        engine.reset()
        engine.dispose()


def _validation_model(**overrides: object) -> VolumeSlippageFillModel:
    """Validation latency-path config (best-price fills) with limit fills certain."""
    cfg: dict[str, object] = {
        "prob_fill_on_limit": 1.0,
        "prob_best_price_fill": 1.0,
        "max_adverse_ticks": 1,
        "random_seed": 42,
    }
    cfg.update(overrides)
    return VolumeSlippageFillModel(config=VolumeSlippageFillModelConfig(**cfg))  # type: ignore[arg-type]


def _exit(fills: list[tuple[str, str, float]]) -> tuple[str, str, float]:
    assert len(fills) == 2, fills
    return fills[1]


# Entry fills at bar-0 close 10000 (no latency); bar 1 spikes through TP/SL.
TP_LONG_BARS = [_bar(10000, 10010, 9990, 10000, 0), _bar(10000, 10300, 9995, 10050, 1),
                _bar(10050, 10060, 10040, 10050, 2)]
SL_LONG_BARS = [_bar(10000, 10010, 9990, 10000, 0), _bar(10000, 10005, 9700, 9800, 1),
                _bar(9800, 9810, 9790, 9800, 2)]
TP_SHORT_BARS = [_bar(10000, 10010, 9990, 10000, 0), _bar(10000, 10005, 9700, 9950, 1),
                 _bar(9950, 9960, 9940, 9950, 2)]
GAP_LONG_BARS = [_bar(10000, 10010, 9990, 10000, 0), _bar(9800, 9810, 9700, 9750, 1),
                 _bar(9750, 9760, 9740, 9750, 2)]


def test_long_tp_limit_fills_at_limit_not_bar_high() -> None:
    order_type, side, px = _exit(_run(_validation_model(), TP_LONG_BARS, tp=1.0))
    assert (order_type, side) == ("LIMIT", "SELL")
    assert px == 10100.0


def test_short_tp_buy_limit_fills_at_limit_not_bar_low() -> None:
    order_type, side, px = _exit(_run(_validation_model(), TP_SHORT_BARS, side="SELL", tp=1.0))
    assert (order_type, side) == ("LIMIT", "BUY")
    assert px == 9900.0


def test_long_sl_fills_at_trigger_plus_configured_tick() -> None:
    """Default config: SPEC 'stop price + 1-tick slippage (pessimistic)'."""
    order_type, side, px = _exit(_run(_validation_model(), SL_LONG_BARS, sl=1.0))
    assert (order_type, side) == ("STOP_MARKET", "SELL")
    assert px == pytest.approx(9900.0 - TICK)


def test_long_sl_fills_exactly_at_trigger_with_zero_stop_ticks() -> None:
    _, _, px = _exit(_run(_validation_model(stop_slippage_ticks=0), SL_LONG_BARS, sl=1.0))
    assert px == 9900.0


def test_short_sl_fills_at_trigger_plus_tick() -> None:
    bars = [_bar(10000, 10010, 9990, 10000, 0), _bar(10000, 10300, 9995, 10200, 1),
            _bar(10200, 10210, 10190, 10200, 2)]
    order_type, side, px = _exit(_run(_validation_model(), bars, side="SELL", sl=1.0))
    assert (order_type, side) == ("STOP_MARKET", "BUY")
    assert px == pytest.approx(10100.0 + TICK)


def test_gap_through_stop_fills_at_open() -> None:
    """Bar opens at 9800, below the 9900 stop: the fill is the open (+ tick), not 9700."""
    _, _, px = _exit(_run(_validation_model(stop_slippage_ticks=0), GAP_LONG_BARS, sl=1.0))
    assert px == 9800.0
    _, _, px_tick = _exit(_run(_validation_model(), GAP_LONG_BARS, sl=1.0))
    assert px_tick == pytest.approx(9800.0 - TICK)


def test_degraded_config_never_fills_limits_better_than_limit() -> None:
    """Sub-5m config (30% degraded market fills) must not leak into resting limits."""
    # The degraded MARKET entry lands 1-2 ticks off 10000, so the TP limit is
    # entry * 1.01 rounded to the tick; the exit must be exactly that limit.
    model = _validation_model(prob_best_price_fill=0.0, max_adverse_ticks=2)
    fills = _run(model, TP_LONG_BARS, tp=1.0)
    assert fills[0][2] > 10000.0  # entry degraded (buy higher)
    assert _exit(fills)[2] == round(fills[0][2] * 1.01, 1)
    assert _exit(fills)[2] < 10300.0
    model = _validation_model(prob_best_price_fill=0.0, max_adverse_ticks=2)
    fills = _run(model, TP_SHORT_BARS, side="SELL", tp=1.0)
    assert fills[0][2] < 10000.0  # entry degraded (sell lower)
    assert _exit(fills)[2] == round(fills[0][2] * 0.99, 1)
    assert _exit(fills)[2] > 9700.0


@pytest.mark.parametrize(("side", "sign"), [("BUY", 1.0), ("SELL", -1.0)])
def test_market_order_slippage_still_adverse(side: str, sign: float) -> None:
    """Always-degraded market fills move against the order (buy higher, sell lower)."""
    bars = [_bar(10000, 10010, 9990, 10000, 0), _bar(10000, 10010, 9990, 10000, 1)]
    model = _validation_model(prob_best_price_fill=0.0, max_adverse_ticks=2)
    fills = _run(model, bars, side=side)
    assert fills[0][0] == "MARKET"
    slip = (fills[0][2] - 10000.0) * sign
    assert slip in (pytest.approx(TICK), pytest.approx(2 * TICK))


def test_market_order_best_price_without_degradation() -> None:
    bars = [_bar(10000, 10010, 9990, 10000, 0), _bar(10000, 10010, 9990, 10000, 1)]
    assert _run(_validation_model(), bars)[0][2] == 10000.0


def test_repeat_runs_bit_identical() -> None:
    """Same seed, same data -> identical fills (validation repeatability)."""
    def scenario() -> list[tuple[str, str, float]]:
        model = _validation_model(prob_best_price_fill=0.5, max_adverse_ticks=2)
        bars = [_bar(10000 + i, 10010 + i, 9990 + i, 10000 + i, i) for i in range(30)]
        return _run(model, bars, tp=0.05, sl=0.05, entry_bar=0)

    assert scenario() == scenario()


def test_stop_slippage_ticks_validated() -> None:
    with pytest.raises(ValueError, match="stop_slippage_ticks"):
        VolumeSlippageFillModel(stop_slippage_ticks=2)
