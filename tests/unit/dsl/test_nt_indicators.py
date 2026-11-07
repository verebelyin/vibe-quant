"""NT indicators with the TradingView / pandas-ta-classic definition (nt_indicators.py).

Synthetic bars through the public classes and the registry, compared with pandas-ta-classic
after each indicator's warm-up (NT seeds averages from the first value, TA from an SMA of the
first n: the gap fades out, so parity is checked from the row where it is below 1e-9).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
import pytest
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.objects import Price, Quantity

from tests.unit.test_audit_dsl_medium import _hour_bars, _run
from vibe_quant.dsl import nt_indicators as nti
from vibe_quant.dsl.compiler import compiler_version_hash
from vibe_quant.dsl.derived import compute_natr_hourly
from vibe_quant.dsl.indicators import indicator_registry

_BT = BarType.from_str("BTCUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL")
_N = 1200


def _ohlc(n: int = _N, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = np.round(100.0 + np.cumsum(rng.normal(0.0, 1.0, n)) + 50.0, 2)
    spread = rng.uniform(0.05, 1.5, n)
    high = np.round(close + spread * rng.uniform(0.0, 1.0, n), 2)
    low = np.round(close - spread * rng.uniform(0.0, 1.0, n), 2)
    return pd.DataFrame({"high": high, "low": low, "close": close})


def _bar(h: float, low: float, c: float, i: int) -> Bar:
    return Bar(
        bar_type=_BT, open=Price(c, 2), high=Price(h, 2), low=Price(low, 2), close=Price(c, 2),
        volume=Quantity.from_str("1"), ts_event=i, ts_init=i,
    )


def _feed(ind: Any, df: pd.DataFrame, attrs: dict[str, str]) -> pd.DataFrame:
    rows = []
    for i, (h, low, c) in enumerate(zip(df.high, df.low, df.close, strict=True)):
        ind.handle_bar(_bar(h, low, c, i))
        rows.append({k: getattr(ind, a) for k, a in attrs.items()} | {"init": ind.initialized})
    return pd.DataFrame(rows)


def _max_diff(got: pd.Series, ref: pd.Series, start: int) -> float:
    return float((got.iloc[start:] - ref.iloc[start:]).abs().max())


def test_rsi_is_wilder() -> None:
    df = _ohlc()
    out = _feed(nti.WilderRelativeStrengthIndex(14), df, {"v": "value"})
    assert _max_diff(out.v * 100.0, ta.rsi(df.close, 14), 420) <= 1e-9


@pytest.mark.parametrize(("period", "warm"), [(14, 420), (30, 900)])
def test_atr_is_wilder(period: int, warm: int) -> None:
    df = _ohlc()
    out = _feed(nti.WilderAverageTrueRange(period), df, {"v": "value"})
    assert _max_diff(out.v, ta.atr(df.high, df.low, df.close, period), warm) <= 1e-9


def test_roc_is_n_bar_and_ready_at_row_n() -> None:
    df = _ohlc()
    out = _feed(nti.NBarRateOfChange(10), df, {"v": "value"})
    assert _max_diff(out.v * 100.0, ta.roc(df.close, 10), 10) <= 1e-9
    assert out.init.idxmax() == 10


def test_roc_period_one() -> None:
    df = _ohlc(5)
    out = _feed(nti.NBarRateOfChange(1), df, {"v": "value"})
    assert out.init.idxmax() == 1
    assert out.v.iloc[1] * 100.0 == pytest.approx(100.0 * (df.close.iloc[1] / df.close.iloc[0] - 1))


def _stoch_ref(df: pd.DataFrame, k: int, d: int) -> pd.DataFrame:
    return ta.stoch(df.high, df.low, df.close, k=k, d=d, smooth_k=1)


def test_stoch_k_d_match_ta_and_ready_at_row_k_plus_d_minus_2() -> None:
    df = _ohlc()
    out = _feed(nti.SmaDStochastics(14, 3), df, {"k": "value_k", "d": "value_d"})
    ref = _stoch_ref(df, 14, 3)
    assert _max_diff(out.k, ref.iloc[:, 0], 14) <= 1e-12
    assert _max_diff(out.d, ref.iloc[:, 1], 15) <= 1e-12
    assert out.init.idxmax() == 15
    assert ref.iloc[:, 1].first_valid_index() == 15


def test_stoch_flat_window_is_zero() -> None:
    flat = pd.DataFrame({"high": [50.0] * 16, "low": [50.0] * 16, "close": [50.0] * 16})
    out = _feed(nti.SmaDStochastics(14, 3), flat, {"k": "value_k", "d": "value_d"})
    assert out.k.iloc[-1] == 0.0
    assert out.d.iloc[-1] == 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = _stoch_ref(flat, 14, 3)
    assert ref.iloc[-1, 0] == 0.0


def test_stoch_d_one() -> None:
    df = _ohlc(60)
    out = _feed(nti.SmaDStochastics(14, 1), df, {"k": "value_k", "d": "value_d"})
    assert (out.k.iloc[13:] == out.d.iloc[13:]).all()
    assert out.init.idxmax() == 13


def test_stoch_reset_restarts_warmup() -> None:
    df = _ohlc(30)
    ind = nti.SmaDStochastics(14, 3)
    first = _feed(ind, df, {"k": "value_k", "d": "value_d"})
    ind.reset()
    assert not ind.initialized
    again = _feed(ind, df, {"k": "value_k", "d": "value_d"})
    assert again.equals(first)


def test_bbands_use_close_not_hlc3() -> None:
    df = _ohlc()
    out = _feed(nti.CloseBollingerBands(20, 2.0), df, {"u": "upper", "m": "middle", "l": "lower"})
    ref = ta.bbands(df.close, length=20, std=2.0)
    assert _max_diff(out.l, ref.iloc[:, 0], 19) <= 1e-9
    assert _max_diff(out.m, ref.iloc[:, 1], 19) <= 1e-9
    assert _max_diff(out.u, ref.iloc[:, 2], 19) <= 1e-9


def test_bbands_flat_bars_collapse_to_close() -> None:
    flat = pd.DataFrame({"high": [60.0] * 20, "low": [58.0] * 20, "close": [59.0] * 20})
    out = _feed(nti.CloseBollingerBands(20, 2.0), flat, {"u": "upper", "m": "middle", "l": "lower"})
    assert (out.u.iloc[-1], out.m.iloc[-1], out.l.iloc[-1]) == (59.0, 59.0, 59.0)


def test_kc_is_close_ema_with_ema_true_range_band() -> None:
    df = _ohlc()
    out = _feed(nti.CloseKeltnerChannel(20, 2.0), df, {"u": "upper", "m": "middle", "l": "lower"})
    ref = ta.kc(df.high, df.low, df.close, length=20, scalar=2.0)
    assert _max_diff(out.l, ref.iloc[:, 0], 300) <= 1e-9
    assert _max_diff(out.m, ref.iloc[:, 1], 300) <= 1e-9
    assert _max_diff(out.u, ref.iloc[:, 2], 300) <= 1e-9
    assert out.init.idxmax() == 19


def test_kc_period_one_is_close_plus_minus_k_true_range() -> None:
    df = _ohlc(6)
    out = _feed(nti.CloseKeltnerChannel(1, 2.0), df, {"u": "upper", "l": "lower"})
    prev_c = df.close.shift(1)
    tr = np.maximum(df.high - df.low, np.maximum((df.high - prev_c).abs(), (df.low - prev_c).abs()))
    assert out.u.iloc[1:].to_numpy() == pytest.approx((df.close + 2.0 * tr).iloc[1:].to_numpy())
    assert out.l.iloc[1:].to_numpy() == pytest.approx((df.close - 2.0 * tr).iloc[1:].to_numpy())


def test_kc_reset_restarts_warmup() -> None:
    df = _ohlc(30)
    ind = nti.CloseKeltnerChannel(20, 2.0)
    first = _feed(ind, df, {"u": "upper"})
    ind.reset()
    assert not ind.initialized
    assert _feed(ind, df, {"u": "upper"}).equals(first)


# R15: the registry builds and compiles the standard-definition classes
_EXPECTED = {
    "RSI": nti.WilderRelativeStrengthIndex,
    "ATR": nti.WilderAverageTrueRange,
    "NATR": nti.WilderAverageTrueRange,
    "ROC": nti.NBarRateOfChange,
    "STOCH": nti.SmaDStochastics,
    "BBANDS": nti.CloseBollingerBands,
    "KC": nti.CloseKeltnerChannel,
}


@pytest.mark.parametrize("name", list(_EXPECTED))
def test_registry_builds_standard_classes(name: str) -> None:
    assert type(indicator_registry.create_nt_indicator(name, {})) is _EXPECTED[name]


def test_registry_rsi_values_match_ta() -> None:
    df = _ohlc()
    ind = indicator_registry.create_nt_indicator("RSI", {"period": 14})
    out = _feed(ind, df, {"v": "value"})
    assert _max_diff(out.v * 100.0, ta.rsi(df.close, 14), 420) <= 1e-9


def test_registry_natr_matches_ta_on_1h() -> None:
    """R3: the DSL NATR (registry class + derived.compute_natr_hourly) is ta.natr on 1h bars."""
    df = _ohlc()
    ind = indicator_registry.create_nt_indicator("NATR", {"period": 14})
    got = []
    for i, (h, low, c) in enumerate(zip(df.high, df.low, df.close, strict=True)):
        ind.handle_bar(_bar(h, low, c, i))
        got.append(compute_natr_hourly(ind, c, 60))
    ref = ta.natr(df.high, df.low, df.close, length=14)
    assert _max_diff(pd.Series(got), ref, 420) <= 1e-9


def test_registry_roc_period_is_the_dsl_period() -> None:
    ind = indicator_registry.create_nt_indicator("ROC", {"period": 3})
    out = _feed(ind, _ohlc(10), {"v": "value"})
    assert out.init.idxmax() == 3


def test_compiled_source_imports_the_standard_classes() -> None:
    from vibe_quant.dsl.compiler import StrategyCompiler
    from vibe_quant.dsl.parser import validate_strategy_dict

    dsl: dict[str, Any] = {
        "name": "std_defs", "timeframe": "1h",
        "indicators": {"rsi": {"type": "RSI", "period": 14}, "bb": {"type": "BBANDS", "period": 20}},
        "entry_conditions": {"long": ["rsi < 30"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
    }
    src = StrategyCompiler().compile(validate_strategy_dict(dsl))
    assert "from vibe_quant.dsl.nt_indicators import WilderRelativeStrengthIndex" in src
    assert "from vibe_quant.dsl.nt_indicators import CloseBollingerBands" in src


# R14: the compiler hash covers the indicator module
def test_compiler_hash_changes_with_nt_indicators_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    before = compiler_version_hash()
    orig = Path.read_bytes

    def patched(self: Path) -> bytes:
        data = orig(self)
        return data + b"# edit" if self.name == "nt_indicators.py" else data

    monkeypatch.setattr(Path, "read_bytes", patched)
    assert compiler_version_hash() != before


# R8: the sizing ATR is the DSL ATR (one class, same value every bar)
class _AtrRecorder:
    def on_start(self) -> None:
        super().on_start()  # type: ignore[misc]
        self.pairs: list[tuple[float, float]] = []

    def on_bar(self, bar: Bar) -> None:
        self.pairs.append((self.ind_atr.value, self._sizing_atr.value))  # type: ignore[attr-defined]
        super().on_bar(bar)  # type: ignore[misc]


def test_sizing_atr_is_dsl_atr() -> None:
    dsl: dict[str, Any] = {
        "name": "sizing_is_dsl_atr", "timeframe": "1h",
        "indicators": {"atr": {"type": "ATR", "period": 14}},
        "entry_conditions": {"long": ["atr > 1000000"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
    }
    closes = [float(100.0 + 3.0 * np.sin(i / 3.0) + 0.05 * i) for i in range(80)]
    strat = _run(dsl, _hour_bars(closes), _AtrRecorder, sizing_mode="atr", sizing_atr_period=14)
    assert type(strat._sizing_atr) is nti.WilderAverageTrueRange
    assert len(strat.pairs) == 80
    assert max(abs(a - b) for a, b in strat.pairs) == 0.0
    assert strat.pairs[-1][0] > 0.0
