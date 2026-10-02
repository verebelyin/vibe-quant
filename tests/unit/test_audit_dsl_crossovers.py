"""vibe-quant-e70tl.3: price-vs-indicator crossovers.

Before the fix price operands had no previous value, so
``close crosses_above ema`` compiled to ``close > ema and close <= prev_ema`` --
impossible for EMA/KAMA (the MA moves toward price) and wrong for SMA.

The fire-count tests drive real compiled strategies through a NautilusTrader
``BacktestEngine`` over a 20k-bar random walk and require that the strategy
fires on exactly the bars where a true cross happened on the values the
strategy itself saw (``prev_left <= prev_right and left > right``), and that
those values are the real moving average.
"""

from __future__ import annotations

import types
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import pytest
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Currency, Money, Price, Quantity

from vibe_quant.data.catalog import create_instrument
from vibe_quant.dsl.compiler import StrategyCompiler, _to_class_name
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.dsl.plugins.kama import compute_kama
from vibe_quant.nt_compat import retain_log_guard

_IID = "BTCUSDT-PERP.BINANCE"
_H = 3_600_000_000_000
_N_BARS = 20_000


def _dsl(name: str, ma_type: str, period: int, long_cond: str, short_cond: str) -> dict[str, Any]:
    return {
        "name": name,
        "timeframe": "1h",
        "indicators": {"ma": {"type": ma_type, "period": period}},
        "entry_conditions": {"long": [long_cond], "short": [short_cond]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
    }


def _random_walk_bars(n: int, seed: int = 1) -> tuple[list[Bar], np.ndarray]:
    rng = np.random.default_rng(seed)
    close = np.round(30_000.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), 1)
    bar_type = BarType.from_str(f"{_IID}-1-HOUR-LAST-EXTERNAL")
    bars = []
    prev = close[0]
    for i, c in enumerate(close):
        o = prev
        hi, lo = max(o, c) + 5.0, min(o, c) - 5.0
        bars.append(
            Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{o:.1f}"),
                high=Price.from_str(f"{hi:.1f}"),
                low=Price.from_str(f"{lo:.1f}"),
                close=Price.from_str(f"{c:.1f}"),
                volume=Quantity.from_str("1.000"),
                ts_event=i * _H,
                ts_init=(i + 1) * _H - 1,
            )
        )
        prev = c
    return bars, close


def _probe_class(module: Any, class_name: str) -> type:
    base = getattr(module, f"{class_name}Strategy")

    class Probe(base):  # type: ignore[misc, valid-type]
        def __init__(self, config: Any) -> None:
            super().__init__(config)
            self.bar_idx = -1
            self.long_fires: list[int] = []
            self.short_fires: list[int] = []
            self.seen: dict[int, tuple[float, float]] = {}

        def on_bar(self, bar: Bar) -> None:
            self.bar_idx += 1
            super().on_bar(bar)
            if self._indicators_ready():
                self.seen[self.bar_idx] = (
                    float(bar.close.as_double()),
                    self._get_indicator_value("ma"),
                )

        def _submit_long_entry(self, bar: Bar) -> None:  # record, never trade
            self.long_fires.append(self.bar_idx)

        def _submit_short_entry(self, bar: Bar) -> None:
            self.short_fires.append(self.bar_idx)

    return Probe


def _run(dsls: list[dict[str, Any]], bars: list[Bar]) -> list[Any]:
    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    retain_log_guard(engine)
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(1_000, Currency.from_str("USDT"))],
        default_leverage=Decimal("10"),
        bar_execution=True,
    )
    engine.add_instrument(create_instrument("BTCUSDT"))
    engine.add_data(bars)
    strategies = []
    for d in dsls:
        module = StrategyCompiler().compile_to_module(validate_strategy_dict(d))
        cn = _to_class_name(d["name"])
        probe_cls = _probe_class(module, cn)
        cfg = getattr(module, f"{cn}Config")(instrument_id=_IID)
        strat = probe_cls(cfg)
        engine.add_strategy(strat)
        strategies.append(strat)
    engine.run()
    engine.dispose()
    return strategies


def _expected(seen: dict[int, tuple[float, float]], price_left: bool) -> tuple[list[int], list[int]]:
    """True crosses on the values the strategy saw (first ready bar never fires)."""
    up: list[int] = []
    down: list[int] = []
    for i in sorted(seen):
        if i - 1 not in seen:
            continue
        c0, m0 = seen[i - 1]
        c1, m1 = seen[i]
        left0, right0, left1, right1 = (c0, m0, c1, m1) if price_left else (m0, c0, m1, c1)
        if left0 <= right0 and left1 > right1:
            up.append(i)
        if left0 >= right0 and left1 < right1:
            down.append(i)
    return up, down


@pytest.fixture(scope="module")
def walk() -> tuple[list[Bar], np.ndarray]:
    return _random_walk_bars(_N_BARS)


@pytest.mark.parametrize(
    ("ma_type", "period"),
    [("EMA", 20), ("SMA", 20), ("KAMA", 10)],
)
def test_price_vs_ma_crossover_fire_count(
    walk: tuple[list[Bar], np.ndarray], ma_type: str, period: int
) -> None:
    bars, close = walk
    tag = ma_type.lower()
    price_left, price_right = _run(
        [
            _dsl(f"xp_left_{tag}", ma_type, period, "close crosses_above ma", "close crosses_below ma"),
            _dsl(f"xp_right_{tag}", ma_type, period, "ma crosses_above close", "ma crosses_below close"),
        ],
        bars,
    )

    # The values the strategy saw are the real moving average.
    idx = np.array(sorted(price_left.seen))
    seen_ma = np.array([price_left.seen[i][1] for i in idx])
    s = pd.Series(close)
    if ma_type == "EMA":
        ref = s.ewm(span=period, adjust=False).mean().to_numpy()
    elif ma_type == "SMA":
        ref = s.rolling(period).mean().to_numpy()
    else:
        ref = compute_kama(
            pd.DataFrame({"open": s, "high": s, "low": s, "close": s, "volume": 1.0}),
            {"period": period},
        ).to_numpy()
    # KAMA runs on a capped rolling buffer; it converges to the full-history value.
    np.testing.assert_allclose(seen_ma[-5000:], ref[idx][-5000:], rtol=1e-6)

    for strat, is_price_left in ((price_left, True), (price_right, False)):
        exp_up, exp_down = _expected(strat.seen, price_left=is_price_left)
        assert len(exp_up) > 500 and len(exp_down) > 500  # a meaningful sample
        assert strat.long_fires == exp_up
        assert strat.short_fires == exp_down


# ---------------------------------------------------------------------------
# Unit-level edge cases on the generated condition methods.
# ---------------------------------------------------------------------------


class _P:
    def __init__(self, v: float) -> None:
        self.v = v

    def as_double(self) -> float:
        return self.v


def _check(prev: dict[str, float], close: float, ema: float, cond: str = "close crosses_above ema") -> bool:
    dsl = validate_strategy_dict(
        {
            "name": "xprice_unit",
            "timeframe": "1h",
            "indicators": {"ema": {"type": "EMA", "period": 20}},
            "entry_conditions": {"long": [cond]},
            "stop_loss": {"type": "fixed_pct", "percent": 2},
            "take_profit": {"type": "fixed_pct", "percent": 4},
        }
    )
    mod = StrategyCompiler().compile_to_module(dsl)
    fake = types.SimpleNamespace(_prev_values=dict(prev), _get_indicator_value=lambda n: ema)
    bar = types.SimpleNamespace(close=_P(close))
    return bool(mod.XpriceUnitStrategy._check_long_entry(fake, bar))


class TestCrossoverEdges:
    def test_true_cross_fires(self) -> None:
        assert _check({"ema": 100.0, "@close": 99.0}, close=101.0, ema=100.2)

    def test_close_already_above_does_not_fire(self) -> None:
        # Old code fired here (close > ema and close <= prev_ema) although close
        # was above the EMA on both bars.
        assert not _check({"ema": 102.0, "@close": 103.0}, close=101.5, ema=100.0)

    def test_equality_edge_fires_once(self) -> None:
        assert _check({"ema": 100.0, "@close": 100.0}, close=100.5, ema=100.2)

    def test_first_bar_after_warmup_never_fires(self) -> None:
        # No prev values yet (first ready bar): never a cross, even if close > ema.
        assert not _check({}, close=101.0, ema=100.0)
        # Indicator prev present but price prev missing -> still guarded.
        assert not _check({"ema": 100.0}, close=101.0, ema=100.2)

    def test_price_on_right(self) -> None:
        # ema crosses_above close: ema 99 -> 101, close 100 -> 100.5
        assert _check({"ema": 99.0, "@close": 100.0}, close=100.5, ema=101.0, cond="ema crosses_above close")
        assert not _check({"ema": 101.0, "@close": 100.0}, close=100.5, ema=101.0, cond="ema crosses_above close")

    def test_price_vs_literal_cross(self) -> None:
        # close crosses_above 100 (literal lifted to config threshold) needs prev close
        dsl = validate_strategy_dict(
            {
                "name": "xprice_lit",
                "timeframe": "1h",
                "indicators": {"ema": {"type": "EMA", "period": 20}},
                "entry_conditions": {"long": ["close crosses_above 100"]},
                "stop_loss": {"type": "fixed_pct", "percent": 2},
                "take_profit": {"type": "fixed_pct", "percent": 4},
            }
        )
        mod = StrategyCompiler().compile_to_module(dsl)
        cfg = types.SimpleNamespace(close_100_0_threshold=100.0)

        def run(prev_close: float | None, close: float) -> bool:
            prev = {} if prev_close is None else {"@close": prev_close}
            fake = types.SimpleNamespace(_prev_values=prev, config=cfg)
            return bool(mod.XpriceLitStrategy._check_long_entry(fake, types.SimpleNamespace(close=_P(close))))

        assert run(99.0, 101.0)
        assert not run(101.0, 102.0)
        assert not run(None, 101.0)
