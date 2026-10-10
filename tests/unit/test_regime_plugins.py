"""BTC_TREND / BTC_ROC / OWN_TREND cross-asset regime indicators (vibe-quant-kgkpb)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from vibe_quant.dsl import aux_data
from vibe_quant.dsl.aux_data import AuxDataUnavailableError
from vibe_quant.dsl.indicators import indicator_registry
from vibe_quant.dsl.pta_buffer import PtaBuffer

if TYPE_CHECKING:
    from collections.abc import Iterator

H = 3_600 * 1_000_000_000
DAY = 24 * H
DAY0 = 1_704_067_200 * 1_000_000_000  # 2024-01-01T00:00:00Z
N = 300


def _closes(seed: int = 0, n: int = N) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))


def _ref(close: np.ndarray, start: int = DAY0) -> tuple[np.ndarray, np.ndarray]:
    """Daily bar i covers day i; RAW ts_init is 23:59:59.999 (close = next midnight)."""
    ts = start + (np.arange(close.size, dtype=np.int64) + 1) * DAY - 1_000_000
    return ts, close


def _close_ts(n: int, start: int = DAY0) -> np.ndarray:
    return start + (np.arange(n, dtype=np.int64) + 1) * DAY


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    aux_data._REF.clear()
    aux_data._REF_DERIVED.clear()
    yield
    aux_data._REF.clear()
    aux_data._REF_DERIVED.clear()
    aux_data.configure(None, None)


def _patch_series(data: dict[str, tuple[np.ndarray, np.ndarray]]):  # type: ignore[no-untyped-def]
    """Replace the raw bar loader: symbol -> (close_ts, close)."""

    class _Bars:
        def __init__(self, ts: np.ndarray, close: np.ndarray) -> None:
            self.ts, self.close = ts, close

    cache = {s: _Bars(*v) for s, v in data.items()}

    def fake(_catalog: object, bar_type: str) -> object:
        return cache.get(bar_type.split("-")[0])

    return patch("vibe_quant.validation.extraction.load_catalog_bars", fake)


def _df(symbol: str, close_ns: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"close": np.ones(close_ns.size)})
    df.attrs["bar_close_ns"] = close_ns
    df.attrs["symbol"] = f"{symbol}-PERP.BINANCE"
    return df


def _run(name: str, df: pd.DataFrame, period: int) -> np.ndarray:
    import vibe_quant.dsl.plugins.regime  # noqa: F401

    spec = indicator_registry.get(name)
    assert spec is not None and spec.compute_fn is not None
    return np.asarray(spec.compute_fn(df, {"period": period}), dtype=float)


def test_no_lookahead_daily_ref() -> None:
    close = _closes()
    d = 50  # strategy day D
    with _patch_series({"BTCUSDT": _ref(close)}):
        t = np.array(
            [DAY0 + d * DAY + h * H for h in (4, 8, 12, 16, 20)] + [DAY0 + (d + 1) * DAY],
            dtype=np.int64,
        )
        got = aux_data.ref_close_asof("BTCUSDT", t)
    # day D's bar closes at D+1 00:00: only the final bar may see it
    assert np.all(got[:5] == close[d - 1])
    assert got[5] == close[d]


def test_btc_trend_matches_manual_ema() -> None:
    close = _closes(1)
    t = DAY0 + np.arange(30, 250, dtype=np.int64) * 4 * H + 4 * H
    with _patch_series({"BTCUSDT": _ref(close)}):
        got = _run("BTC_TREND", _df("ETHUSDT", t), 50)
    ema = pd.Series(close).ewm(span=50, adjust=False).mean().to_numpy()
    trend = close / ema - 1
    trend[:49] = np.nan  # warmup
    idx = np.searchsorted(_close_ts(close.size), t, side="right") - 1
    assert np.allclose(got, trend[idx], rtol=1e-12, equal_nan=True)
    assert np.isnan(trend[:49]).all()


def test_btc_roc_matches_manual() -> None:
    close = _closes(2)
    t = DAY0 + np.arange(0, 250, dtype=np.int64) * 4 * H + 4 * H
    with _patch_series({"BTCUSDT": _ref(close)}):
        got = _run("BTC_ROC", _df("ETHUSDT", t), 20)
    roc = pd.Series(close).pct_change(20).to_numpy()
    idx = np.searchsorted(_close_ts(close.size), t, side="right") - 1
    exp = np.where(idx >= 0, roc[np.clip(idx, 0, None)], np.nan)
    assert np.array_equal(np.isnan(got), np.isnan(exp))
    assert np.allclose(got, exp, rtol=1e-12, equal_nan=True)
    assert np.isnan(got).any() and (~np.isnan(got)).any()


def test_own_trend_uses_strategy_symbol() -> None:
    btc, eth = _closes(3), _closes(4)
    t = DAY0 + np.arange(60, 200, dtype=np.int64) * 4 * H
    with _patch_series({"BTCUSDT": _ref(btc), "ETHUSDT": _ref(eth)}):
        own = _run("OWN_TREND", _df("ETHUSDT", t), 30)
        via_btc = _run("BTC_TREND", _df("ETHUSDT", t), 30)
    ema = pd.Series(eth).ewm(span=30, adjust=False).mean().to_numpy()
    idx = np.searchsorted(_close_ts(eth.size), t, side="right") - 1
    exp = eth / ema - 1
    exp[:29] = np.nan
    assert np.allclose(own, exp[idx], rtol=1e-12, equal_nan=True)
    assert not np.allclose(own, via_btc, equal_nan=True)


def test_stale_ref_returns_nan() -> None:
    close = _closes(5, 100)
    ts, c = _ref(close)
    ts = ts + 1_000_000
    keep = np.r_[0:40, 45:100]  # days 40..44 missing
    with _patch_series({"BTCUSDT": (ts[keep] - 1_000_000, c[keep])}):
        # last ref close before hole = ts[39] = DAY0 + 40 DAY
        ok = aux_data.ref_trend_asof("BTCUSDT", np.array([ts[39] + 2 * DAY], dtype=np.int64), 10)
        stale = aux_data.ref_trend_asof(
            "BTCUSDT", np.array([ts[39] + 2 * DAY + H], dtype=np.int64), 10
        )
        before = aux_data.ref_trend_asof("BTCUSDT", np.array([DAY0], dtype=np.int64), 10)
    assert np.isfinite(ok[0])
    assert np.isnan(stale[0])
    assert np.isnan(before[0])


def test_preflight_raises_when_ref_missing_or_short() -> None:
    close = _closes(6, 100)  # closes DAY0+1d .. DAY0+100d
    with _patch_series({"BTCUSDT": _ref(close)}):
        with pytest.raises(AuxDataUnavailableError, match="start after"):
            aux_data.preflight(["BTC_TREND"], ["ETHUSDT"], "2023-06-01", "2024-02-01")
        with pytest.raises(AuxDataUnavailableError, match="end before"):
            aux_data.preflight(["BTC_ROC"], ["ETHUSDT"], "2024-01-05", "2024-12-01")
        with pytest.raises(AuxDataUnavailableError, match="ETHUSDT"):
            aux_data.preflight(["OWN_TREND"], ["ETHUSDT"], "2024-01-05", "2024-03-01")
        # window inside coverage (end - 2d <= last close): fine, no funding archive needed
        aux_data.preflight(["BTC_TREND"], ["BTCUSDT"], "2024-01-05", "2024-04-05")
    with _patch_series({}), pytest.raises(AuxDataUnavailableError, match="BTCUSDT"):
        aux_data.preflight(["BTC_TREND"], ["ETHUSDT"], "2024-01-05", "2024-03-01")


def test_registered_and_excluded_from_default_pool() -> None:
    import vibe_quant.dsl.plugins.regime  # noqa: F401
    from vibe_quant.discovery.genome import build_indicator_pool

    default = build_indicator_pool()
    full = build_indicator_pool(include_context=True)
    for name in ("BTC_TREND", "BTC_ROC", "OWN_TREND"):
        spec = indicator_registry.get(name)
        assert spec is not None and spec.needs_context
        assert name not in default
        assert name in full
    trend = indicator_registry.get("BTC_TREND")
    assert trend is not None and trend.param_ranges == {"period": (20.0, 250.0)}
    assert trend.threshold_range == (-0.5, 0.5)


def test_partial_current_day_bar_is_not_a_close() -> None:
    """In-progress daily bars (ts_init mid-day) must never be seen, before or after noon."""
    close = _closes(7, 60)
    ts, c = _ref(close)
    d = 60  # day D: partial bar appended
    for partial_h, tag in ((11, "before-noon"), (14, "after-noon")):
        part_ts = DAY0 + d * DAY + partial_h * H - 1_000_000  # hh:59:59.999-style, off-boundary
        aux_data._REF.clear()
        aux_data._REF_DERIVED.clear()
        data = (np.r_[ts, part_ts], np.r_[c, 999.0])
        with _patch_series({"BTCUSDT": data}):
            t = np.array([DAY0 + d * DAY, DAY0 + d * DAY + 4 * H, DAY0 + d * DAY + 16 * H], dtype=np.int64)
            got = aux_data.ref_close_asof("BTCUSDT", t)
        assert np.all(got == close[-1]), tag
        assert 999.0 not in got


def test_trend_warmup_is_nan() -> None:
    close = _closes(8, 100)
    with _patch_series({"BTCUSDT": _ref(close)}):
        ts, out = aux_data._ref_derived("BTCUSDT", "1d", "trend", 20)
    assert np.isnan(out[:19]).all() and np.isfinite(out[19:]).all()


def test_ref_bars_per_call_path_does_not_reload() -> None:
    close = _closes(9, 60)
    calls: list[object] = []
    with _patch_series({"BTCUSDT": _ref(close)}):
        aux_data.ref_close_asof("BTCUSDT", np.array([DAY0 + 30 * DAY], dtype=np.int64))
        with patch("vibe_quant.validation.extraction.load_catalog_bars", lambda *a: calls.append(a)):
            aux_data.ref_trend_asof("BTCUSDT", np.array([DAY0 + 30 * DAY], dtype=np.int64), 10)
            aux_data.ref_trend_asof("BTCUSDT", np.array([DAY0 + 31 * DAY], dtype=np.int64), 10)
    assert calls == []


def test_context_nan_overrides_last_valid_value_in_generated_strategy() -> None:
    from vibe_quant.dsl.compiler import StrategyCompiler
    from vibe_quant.dsl.parser import validate_strategy_dict

    def src(ind_type: str) -> str:
        dsl = validate_strategy_dict(
            {
                "name": "nan_ctx",
                "timeframe": "4h",
                "indicators": {"x": {"type": ind_type, "period": 20}},
                "entry_conditions": {"long": ["x > 0"]},
                "exit_conditions": {"long": ["x < 0"]},
                "stop_loss": {"type": "fixed_pct", "percent": 2.0},
                "take_profit": {"type": "fixed_pct", "percent": 4.0},
            }
        )
        return StrategyCompiler().compile(dsl)

    import vibe_quant.dsl.plugins.regime  # noqa: F401

    ctx = src("BTC_TREND")
    assert 'self._pta_values["x"] = float(_v)' in ctx
    assert "if not pd.isna(_v)" not in ctx.split("def _update_pta_indicators")[1].split("\n    def ")[0]
    # NaN stored -> comparison False -> no entry on stale context
    assert not (float("nan") > 0) and not (float("nan") < 0)


# -- real data (catalog under data/) -----------------------------------------------------


def _regime_runner(start: str, end: str):  # type: ignore[no-untyped-def]
    from vibe_quant.dsl.parser import validate_strategy_dict
    from vibe_quant.screening.nt_runner import NTScreeningRunner
    from vibe_quant.screening.pipeline import _dsl_to_dict

    dsl = validate_strategy_dict(
        {
            "name": "btc_regime_smoke",
            "timeframe": "4h",
            "indicators": {"bt": {"type": "BTC_TREND", "period": 200}},
            "entry_conditions": {"long": ["bt > 0"]},
            "exit_conditions": {"long": ["bt < 0"]},
            "stop_loss": {"type": "fixed_pct", "percent": 2.0},
            "take_profit": {"type": "fixed_pct", "percent": 4.0},
        }
    )
    return NTScreeningRunner(_dsl_to_dict(dsl), ["ETHUSDT"], start, end)


def test_real_data_screening_trades() -> None:
    from pathlib import Path

    if not Path("data/catalog/data/bar/BTCUSDT-PERP.BINANCE-1-DAY-LAST-EXTERNAL").is_dir():
        pytest.skip("BTC 1d catalog data not available")
    result = _regime_runner("2023-01-01", "2024-12-31")({})
    assert result.total_trades > 0


def test_real_data_window_before_ref_start_raises() -> None:
    from pathlib import Path

    if not Path("data/catalog/data/bar/BTCUSDT-PERP.BINANCE-1-DAY-LAST-EXTERNAL").is_dir():
        pytest.skip("BTC 1d catalog data not available")
    with pytest.raises(AuxDataUnavailableError):
        _regime_runner("2021-06-01", "2024-12-31")({})


def test_context_nan_after_valid_value_blocks_entry_at_bar_level() -> None:
    """Bar handler: valid -> valid -> NaN leaves NaN in _pta_values and the entry False."""
    from types import SimpleNamespace

    import vibe_quant.dsl.plugins.regime  # noqa: F401
    from vibe_quant.dsl.compiler import StrategyCompiler
    from vibe_quant.dsl.parser import validate_strategy_dict

    dsl = validate_strategy_dict(
        {
            "name": "nan_bar_ctx",
            "timeframe": "4h",
            "indicators": {"x": {"type": "BTC_TREND", "period": 20}},
            "entry_conditions": {"long": ["x > 0"]},
            "exit_conditions": {"long": ["x < 0"]},
            "stop_loss": {"type": "fixed_pct", "percent": 2.0},
            "take_profit": {"type": "fixed_pct", "percent": 4.0},
        }
    )
    module = StrategyCompiler().compile_to_module(dsl)
    cls = next(v for k, v in vars(module).items() if k.endswith("Strategy") and k != "Strategy")
    script = iter([0.5, 0.5, float("nan")])
    module.compute_btc_trend = lambda df, params: pd.Series(  # type: ignore[attr-defined]
        [next(script)] * len(df.index), index=df.index
    )
    cfg = SimpleNamespace(instrument_id="ETHUSDT-PERP.BINANCE", x_0_0_threshold=0.0)
    harness = type("Harness", (cls,), {"config": cfg})  # Actor.config is read-only
    inst = harness.__new__(harness)
    inst._pta_values = {}
    inst._pta_params = {"x": {"period": 20}}
    inst._pta_lookback = {"x": 1}
    inst._pta_bufs = {"4h": PtaBuffer(0, with_close_ns=True)}
    for i in range(3):
        inst._pta_bufs["4h"].append(1.0, 1.0, 1.0, 1.0, 1.0, i)
        inst._update_pta_indicators("4h")
        if i == 1:
            assert inst._pta_values["x"] == 0.5
            assert inst._check_long_entry(None) is True
    assert np.isnan(inst._pta_values["x"])
    assert inst._check_long_entry(None) is False
