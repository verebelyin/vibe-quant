"""vibe-quant-e70tl.14: ATR threshold genes were in the wrong units.

ATR is an absolute price distance (BTC 1m median ~40 USD, SOL ~0.1 USD) but its
GA threshold range was (0.001, 0.15), so every ATR gene was constant true/false.
ATR leaves the GA threshold pool; NATR (100 * ATR / close, rescaled to a 1h
bar) replaces it with a range validated on real BTC/ETH/SOL data.
"""

from __future__ import annotations

import math
import random
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.indicators import AverageTrueRange
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Currency, Money, Price, Quantity

from vibe_quant.data.catalog import create_instrument
from vibe_quant.dsl.compiler import StrategyCompiler, _to_class_name
from vibe_quant.dsl.derived import compute_natr_hourly
from vibe_quant.dsl.indicators import indicator_registry
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.nt_compat import retain_log_guard

_IID = "BTCUSDT-PERP.BINANCE"
_CATALOG = Path(__file__).resolve().parents[2] / "data" / "catalog"


class TestRegistry:
    def test_atr_out_of_ga_threshold_pool(self) -> None:
        from vibe_quant.discovery.genome import build_indicator_pool

        atr = indicator_registry.get("ATR")
        assert atr is not None and atr.threshold_range is None
        pool = build_indicator_pool()
        assert "ATR" not in pool
        assert "NATR" in pool
        assert pool["NATR"].default_threshold_range == (0.9, 1.2)

    def test_non_atr_genes_unaffected(self) -> None:
        from vibe_quant.discovery.genome import build_indicator_pool

        pool = build_indicator_pool()
        expected = {
            "RSI": (25.0, 75.0),
            "MACD": (-0.05, 0.05),
            "STOCH": (20.0, 80.0),
            "CCI": (-200.0, 200.0),
            "WILLR": (-80.0, -20.0),
            "ROC": (-5.0, 5.0),
            "ADX": (15.0, 60.0),
            "BBANDS": (0.0, 1.0),
            "DONCHIAN": (0.0, 1.0),
            "MFI": (20.0, 80.0),
        }
        for name, rng in expected.items():
            assert pool[name].default_threshold_range == rng

    def test_helper_formula(self) -> None:
        class _Atr:
            value = 400.0

        # 100 * 400 / 40000 = 1.0% on 1h; x2 on 15m (sqrt(60/15)); /2 on 4h
        assert compute_natr_hourly(_Atr(), 40_000.0, 60) == pytest.approx(1.0)
        assert compute_natr_hourly(_Atr(), 40_000.0, 15) == pytest.approx(2.0)
        assert compute_natr_hourly(_Atr(), 40_000.0, 240) == pytest.approx(0.5)
        assert compute_natr_hourly(_Atr(), 0.0, 60) == 0.0


def _bars(n: int, spec: str, step_ns: int, seed: int) -> list[Bar]:
    rng = np.random.default_rng(seed)
    close = np.round(30_000.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n))), 1)
    bt = BarType.from_str(f"{_IID}-{spec}-LAST-EXTERNAL")
    out = []
    prev = close[0]
    for i, c in enumerate(close):
        spread = float(np.round(abs(rng.normal(0, 40)), 1)) + 1.0
        out.append(
            Bar(
                bar_type=bt,
                open=Price.from_str(f"{prev:.1f}"),
                high=Price.from_str(f"{max(prev, c) + spread:.1f}"),
                low=Price.from_str(f"{min(prev, c) - spread:.1f}"),
                close=Price.from_str(f"{c:.1f}"),
                volume=Quantity.from_str("1.000"),
                ts_event=i * step_ns,
                ts_init=(i + 1) * step_ns - 1,
            )
        )
        prev = c
    return out


@pytest.mark.parametrize(("timeframe", "spec", "minutes"), [("1h", "1-HOUR", 60), ("15m", "15-MINUTE", 15)])
def test_compiled_natr_matches_reference(timeframe: str, spec: str, minutes: int) -> None:
    bars = _bars(300, spec, minutes * 60_000_000_000, seed=minutes)
    dsl = {
        "name": f"natr_probe_{timeframe}",
        "timeframe": timeframe,
        "indicators": {"natr": {"type": "NATR", "period": 14}},
        "entry_conditions": {"long": ["natr > 1000"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
    }
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
    module = StrategyCompiler().compile_to_module(validate_strategy_dict(dsl))
    cn = _to_class_name(dsl["name"])
    strat = getattr(module, f"{cn}Strategy")(getattr(module, f"{cn}Config")(instrument_id=_IID))
    engine.add_strategy(strat)
    engine.run()

    ref = AverageTrueRange(14)
    for b in bars:
        ref.handle_bar(b)
    expected = ref.value / float(bars[-1].close) * 100 * math.sqrt(60 / minutes)
    assert strat._get_indicator_value("natr") == pytest.approx(expected, rel=1e-12)
    assert strat._prev_values["natr"] == pytest.approx(expected, rel=1e-12)
    engine.dispose()


@pytest.mark.skipif(not _CATALOG.exists(), reason="needs data/catalog with BTC/ETH/SOL bars")
@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
@pytest.mark.parametrize(("tf", "minutes"), [("1-HOUR", 60), ("4-HOUR", 240)])
def test_random_natr_gene_fires_5_to_95_pct_on_real_data(symbol: str, tf: str, minutes: int) -> None:
    from nautilus_trader.persistence.catalog import ParquetDataCatalog

    bars = ParquetDataCatalog(str(_CATALOG)).bars(
        bar_types=[f"{symbol}-PERP.BINANCE-{tf}-LAST-EXTERNAL"], start="2024-01-01", end="2026-03-01"
    )
    if len(bars) < 1000:
        pytest.skip(f"not enough {symbol} {tf} bars in catalog")
    natr = indicator_registry.get("NATR")
    assert natr is not None and natr.threshold_range is not None
    (p_lo, p_hi), (t_lo, t_hi) = natr.param_ranges["period"], natr.threshold_range
    rng = random.Random(14)
    series: dict[int, np.ndarray] = {}
    for _ in range(30):  # random genes, as the GA samples them
        period = rng.randint(int(p_lo), int(p_hi))
        thr = rng.uniform(t_lo, t_hi)
        if period not in series:
            atr = AverageTrueRange(period)
            vals = []
            for b in bars:
                atr.handle_bar(b)
                if atr.initialized:
                    vals.append(compute_natr_hourly(atr, float(b.close), minutes))
            series[period] = np.array(vals)
        above = float((series[period] > thr).mean())
        for frac in (above, 1.0 - above):  # natr > thr  and  natr < thr
            assert 0.05 <= frac <= 0.95, (symbol, tf, period, thr, frac)
