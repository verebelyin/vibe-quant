"""Every registered ``compute_fn`` is safe at exactly its ``pta_lookback`` rows.

Generated strategies call ``compute_fn`` as soon as the buffer holds
``pta_lookback`` bars, so the call must neither raise nor emit a pandas-ta
"Series has N rows but indicator requires at least M" warning (the library
logs it; tests treat both ``warnings`` and log records as failures). Checked on
the numpy-port path and, for indicators with a ``*_port_ok`` self-check, on the
forced pandas-ta fallback path.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import vibe_quant.dsl.plugin_loader  # noqa: F401  (registers plugin indicators)
from tests.unit.pta_stream_harness import random_walk_bars
from vibe_quant.dsl import aux_data
from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry, pta_lookback

_SPECS = [s for s in indicator_registry.all_specs() if s.compute_fn is not None]

_DAY_NS = 86_400 * 10**9
_FUNDING_NS = 8 * 3_600 * 10**9
# Strategy bars close on day 300 of synthetic history: funding covers days 0-600,
# daily reference bars days 1-400 (enough for BTC_TREND/OWN_TREND period 250).
_BAR_T0_NS = 300 * _DAY_NS
_BAR_STEP_NS = 3_600 * 10**9


def _ohlcv(n: int) -> pd.DataFrame:
    bars = random_walk_bars(n)
    return pd.DataFrame(
        {
            "open": [b[1] for b in bars],
            "high": [b[2] for b in bars],
            "low": [b[3] for b in bars],
            "close": [b[4] for b in bars],
            "volume": [b[5] for b in bars],
        }
    )


@pytest.fixture
def synthetic_aux(monkeypatch: pytest.MonkeyPatch) -> None:
    """In-memory funding + daily reference bars so ``needs_context`` fns run for real."""
    rng = np.random.RandomState(3)
    settle = [i * _FUNDING_NS for i in range(1800)]
    funding_source = (settle, [float(r) for r in rng.randn(1800) * 1e-4])
    ref_ts = (np.arange(1, 401, dtype=np.int64)) * _DAY_NS
    ref_entry = (object(), ref_ts, 100.0 + np.cumsum(rng.randn(400)))
    # TAKER_IMBALANCE reads 1m klines around the synthetic bar closes (day 300):
    # consecutive minutes, 2.0 volume and 1.0 taker-buy each.
    minute = 60 * 10**9
    n_klines = 4000
    kline_close = _BAR_T0_NS - 3000 * minute + (np.arange(n_klines, dtype=np.int64) + 1) * minute
    steps = np.arange(n_klines + 1, dtype=np.int64)
    kline_entry = (kline_close, steps * 2, steps, steps)
    monkeypatch.setattr(aux_data, "_source", lambda symbol: (Path("synthetic.db"), funding_source))
    monkeypatch.setattr(aux_data, "_ref_bars", lambda symbol, tf, refresh=False: ref_entry)
    monkeypatch.setattr(aux_data, "_taker_klines", lambda symbol: kline_entry)
    for cache in ("_SNAPPED", "_DERIVED", "_REF", "_REF_DERIVED"):
        monkeypatch.setattr(aux_data, cache, {})


def _frame(n: int, *, context: bool = False) -> pd.DataFrame:
    df = _ohlcv(n)
    if context:
        df.attrs["bar_close_ns"] = _BAR_T0_NS + np.arange(n, dtype=np.int64) * _BAR_STEP_NS
        df.attrs["symbol"] = "BTCUSDT-PERP.BINANCE"
    return df


def _merged(spec: IndicatorSpec, overrides: dict[str, object]) -> dict[str, object]:
    """Params as the compiler passes them: a DSL field replaces its spec alias."""
    merged: dict[str, object] = dict(spec.default_params)
    for key, value in overrides.items():
        target = StrategyCompiler._PARAM_ALIASES.get(key, key)
        for alias, dsl_field in StrategyCompiler._PARAM_ALIASES.items():
            if dsl_field == target:
                merged.pop(alias, None)
        merged[target] = value
    return merged


def _param_combos(spec: IndicatorSpec) -> list[dict[str, object]]:
    """Defaults plus every int param at its GA range low and high (others default)."""
    combos: list[dict[str, object]] = [_merged(spec, {})]
    for key, (lo, hi) in spec.param_ranges.items():
        if spec.param_schema.get(key) is not int:
            continue
        combos.extend(_merged(spec, {key: int(bound)}) for bound in (lo, hi))
    return combos


def _force_pandas_ta_path(monkeypatch: pytest.MonkeyPatch) -> int:
    patched = 0
    for name, mod in list(sys.modules.items()):
        if not name.startswith("vibe_quant.dsl."):
            continue
        for attr in list(vars(mod)):
            if attr.endswith("_port_ok") and callable(getattr(mod, attr)):
                monkeypatch.setattr(mod, attr, lambda: False)
                patched += 1
    return patched


def _check_all(caplog: pytest.LogCaptureFixture) -> list[str]:
    failures: list[str] = []
    for spec in _SPECS:
        assert spec.compute_fn is not None
        for params in _param_combos(spec):
            n = max(pta_lookback(spec, params), 1)
            df = _frame(n, context=spec.needs_context)
            caplog.clear()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    out = spec.compute_fn(df, params)
            except Exception as exc:
                failures.append(f"{spec.name} {params} n={n}: {type(exc).__name__}: {exc}")
                continue
            series = list(out.values()) if isinstance(out, dict) else [out]
            for s in series:
                if s is None:
                    failures.append(f"{spec.name} {params} n={n}: returned None")
                elif len(s) != n:
                    failures.append(f"{spec.name} {params} n={n}: output length {len(s)}")
                elif not (s.isna().all() or np.isfinite(s.iloc[-1]) or np.isnan(s.iloc[-1])):
                    failures.append(f"{spec.name} {params} n={n}: non-finite last value")
            noisy = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
            if noisy:
                failures.append(f"{spec.name} {params} n={n}: logged {noisy[0][:90]}")
    return failures


@pytest.mark.usefixtures("synthetic_aux")
def test_compute_fn_at_lookback_port_path(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    assert _check_all(caplog) == []


@pytest.mark.usefixtures("synthetic_aux")
def test_compute_fn_at_lookback_forced_pandas_ta_path(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _force_pandas_ta_path(monkeypatch) >= 3  # adx, macd, kama at least
    caplog.set_level(logging.WARNING)
    assert _check_all(caplog) == []


def _raw_or_none(call: object) -> pd.DataFrame | pd.Series | None:
    """Raw pandas-ta result; None when it returns None or raises (short input)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return call()  # type: ignore[operator, no-any-return]
    except IndexError:
        return None


@pytest.mark.usefixtures("synthetic_aux")
def test_context_fns_produce_values_on_synthetic_data() -> None:
    """The synthetic aux fixture is live: every context fn yields a finite last value
    at its lookback (so the registry sweep above exercises real lookups, not NaN stubs)."""
    context_specs = [s for s in _SPECS if s.needs_context]
    assert {s.name for s in context_specs} >= {
        "BTC_ROC",
        "BTC_TREND",
        "OWN_TREND",
        "FUNDING",
        "FUNDING_Z",
    }
    for spec in context_specs:
        assert spec.compute_fn is not None
        for params in _param_combos(spec):
            n = max(pta_lookback(spec, params), 1)
            out = spec.compute_fn(_frame(n, context=True), params)
            assert np.isfinite(out.iloc[-1]), f"{spec.name} {params}"


@pytest.mark.parametrize("period", [1, 2, 3, 5, 14, 20])
def test_dema_kc_guards_do_not_change_values_at_any_length(
    period: int, caplog: pytest.LogCaptureFixture
) -> None:
    """DEMA/KC guards only replace pandas-ta's None/raise by NaN: for every n (both
    sides of each first-valid-bar threshold) the guarded output equals the raw
    pandas-ta output, bit for bit."""
    from vibe_quant.dsl import compute_builtins as cb

    ta = cb._ta()
    caplog.set_level(logging.WARNING)
    for n in range(1, 4 * period + 8):
        df = _ohlcv(n)

        raw_dema = _raw_or_none(lambda df=df: ta.dema(df["close"], length=period))
        caplog.clear()
        got_dema = cb.compute_dema(df, {"period": period})
        assert not caplog.records, f"DEMA p={period} n={n}: {caplog.records[0].getMessage()}"
        if raw_dema is None:
            assert got_dema.isna().all(), f"DEMA p={period} n={n}"
        else:
            pd.testing.assert_series_equal(
                got_dema, raw_dema, check_names=False, check_exact=True
            )

        raw_kc = _raw_or_none(
            lambda df=df: ta.kc(df["high"], df["low"], df["close"], length=period, scalar=2.0)
        )
        caplog.clear()
        got_kc = cb.compute_kc(df, {"period": period, "atr_multiplier": 2.0})
        assert not caplog.records, f"KC p={period} n={n}: {caplog.records[0].getMessage()}"
        for i, name in enumerate(("lower", "middle", "upper")):
            if raw_kc is None:
                assert got_kc[name].isna().all(), f"KC p={period} n={n}"
            else:
                np.testing.assert_array_equal(
                    got_kc[name].to_numpy(),
                    raw_kc.iloc[:, i].to_numpy(),
                    err_msg=f"KC p={period} n={n}",
                )


@pytest.mark.parametrize(
    ("period", "d_period"),
    [(1, 1), (3, 1), (3, 2), (5, 3), (14, 1), (14, 3), (14, 9), (2, 7)],
)
def test_stoch_guard_does_not_change_values_at_any_length(
    period: int, d_period: int, caplog: pytest.LogCaptureFixture
) -> None:
    """The STOCH guard threshold (k + 3 + d - 2) follows d_period: below it the output
    is all-NaN with no pandas-ta log; at and above it, bit-equal to raw pandas-ta."""
    from vibe_quant.dsl import compute_builtins as cb

    ta = cb._ta()
    caplog.set_level(logging.WARNING)
    for n in range(1, period + d_period + 12):
        df = _ohlcv(n)
        raw_st = _raw_or_none(
            lambda df=df: ta.stoch(df["high"], df["low"], df["close"], k=period, d=d_period)
        )
        caplog.clear()
        got_st = cb.compute_stoch(df, {"period": period, "d_period": d_period})
        label = f"STOCH k={period} d={d_period} n={n}"
        assert not caplog.records, f"{label}: {caplog.records[0].getMessage()}"
        for i, name in enumerate(("k", "d")):
            if raw_st is None:
                assert got_st[name].isna().all(), label
            else:
                np.testing.assert_array_equal(
                    got_st[name].to_numpy(), raw_st.iloc[:, i].to_numpy(), err_msg=label
                )
