"""Chart-overlay indicator computation (``vibe_quant.data.indicator_compute``)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pandas_ta_classic  # noqa: F401 — registers .ta accessor
import pytest

from tests.unit.pta_stream_harness import compile_indicators, random_walk_bars, run_stream
from vibe_quant.data.indicator_compute import compute_indicators
from vibe_quant.dsl.compute_builtins import compute_bbands
from vibe_quant.dsl.indicators import (
    IndicatorSpec,
    indicator_registry,
    invoke_compute_fn,
    pta_lookback,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

BarEvent = tuple[str, float, float, float, float, float]


def _ohlcv(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    high = close * (1.0 + rng.uniform(0.0005, 0.01, n))
    low = close * (1.0 - rng.uniform(0.0005, 0.01, n))
    return pd.DataFrame(
        {
            "open_time": np.arange(n, dtype=np.int64) * 3_600_000,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(1.0, 10.0, n),
        }
    )


def _values(series: object) -> np.ndarray:
    return np.array([np.nan if p.value is None else p.value for p in series.data])  # type: ignore[attr-defined]


def test_keltner_channel_overlay_returns_lower_middle_upper_bands() -> None:
    df = _ohlcv()
    out = compute_indicators(df, [{"type": "KC", "period": 20, "atr_multiplier": 1.5}])

    by_name = {s.output_name: s for s in out}
    assert set(by_name) == {"lower", "middle", "upper"}

    ref = df.ta.kc(high=df["high"], low=df["low"], close=df["close"], length=20, scalar=1.5)
    cols = ref.columns.tolist()
    expected = {"lower": cols[0], "middle": cols[1], "upper": cols[2]}
    for name, col in expected.items():
        np.testing.assert_array_equal(_values(by_name[name]), ref[col].to_numpy(), err_msg=name)
    assert by_name["upper"].display_label == "KC Upper(20)"


def test_donchian_channel_overlay_returns_lower_middle_upper_bands() -> None:
    df = _ohlcv()
    out = compute_indicators(df, [{"type": "DONCHIAN", "period": 20}])

    by_name = {s.output_name: s for s in out}
    assert set(by_name) == {"lower", "middle", "upper"}

    ref = df.ta.donchian(high=df["high"], low=df["low"], lower_length=20, upper_length=20)
    for name, col in {"lower": "DCL_20_20", "middle": "DCM_20_20", "upper": "DCU_20_20"}.items():
        np.testing.assert_array_equal(_values(by_name[name]), ref[col].to_numpy(), err_msg=name)


def test_bbands_overlay_bands_match_engine_on_typical_price() -> None:
    df = _ohlcv()
    out = compute_indicators(df, [{"type": "BBANDS", "period": 20, "std_dev": 2.0}])

    by_name = {s.output_name: s for s in out}
    assert set(by_name) == {"lower", "middle", "upper"}

    ref = compute_bbands(df, {"period": 20, "std_dev": 2.0})
    for name in ("lower", "middle", "upper"):
        np.testing.assert_array_equal(_values(by_name[name]), ref[name].to_numpy(), err_msg=name)


# ---------------------------------------------------------------------------
# Registry fallback: every registered type computable from OHLCV alone is
# computed through its spec's own compute path; the rest get an explicit
# per-indicator unsupported marker instead of vanishing.
# ---------------------------------------------------------------------------

# Params deliberately differ from each spec's default_params: the fallback must
# forward the request's params, never silently fall back to the defaults.
_COMPUTABLE_FROM_OHLCV = [
    ("KAMA", {"period": 20}),  # default 10
    ("VIDYA", {"period": 21}),  # default 14
    ("PRICE_POSITION", {"period": 20}),  # default 52
    ("RAMS", {"period": 30}),  # default 50
    ("ADAPTIVE_RSI", {"period": 21, "alpha": 0.3}),  # default 14 / 0.5
    ("FRAMA", {"period": 10}),  # default 16
    ("TREND_ENSEMBLE", {"max_lookback": 60}),  # default 250
]


@pytest.mark.parametrize(("indicator_type", "params"), _COMPUTABLE_FROM_OHLCV)
def test_registry_fallback_matches_the_strategy_compute_path(
    indicator_type: str, params: dict[str, object]
) -> None:
    """Short histories match ``invoke_compute_fn`` full history bit for bit.

    ``n=400`` is below the first trim (``cap + cap // 4 >= 500``) and the
    overlay also reproduces the runtime warmup gate, so the comparison starts
    at ``pta_lookback``; the post-trim stream is pinned in test_pta_replay.py.
    """
    df = _ohlcv(n=400)
    out = compute_indicators(df, [{"type": indicator_type, **params}])

    assert len(out) == 1, f"{indicator_type}: expected one series, got {len(out)}"
    series = out[0]
    assert series.unsupported is False
    assert series.indicator_type == indicator_type
    assert len(series.data) == len(df)

    spec = indicator_registry.get(indicator_type)
    ref = invoke_compute_fn(spec, df, params)
    assert not isinstance(ref, dict)
    expected = ref.to_numpy(dtype=np.float64)
    assert np.isfinite(expected).any(), f"{indicator_type}: reference is all-NaN"

    lookback = pta_lookback(spec, params)
    chart = _values(series)
    assert np.isnan(chart[: lookback - 1]).all(), f"{indicator_type}: warmup gate"
    np.testing.assert_array_equal(
        chart[lookback - 1 :], expected[lookback - 1 :], err_msg=indicator_type
    )


_UNSUPPORTED_FROM_OHLCV = [
    ("NATR", {"period": 14}, "computed by NautilusTrader at runtime; no chart formula"),
    ("BTC_TREND", {"period": 200}, "needs external context (not available in chart overlays)"),
]


@pytest.mark.parametrize(("indicator_type", "params", "reason"), _UNSUPPORTED_FROM_OHLCV)
def test_unsupported_types_return_an_explicit_marker(
    indicator_type: str, params: dict[str, object], reason: str
) -> None:
    out = compute_indicators(_ohlcv(), [{"type": indicator_type, **params}])

    assert len(out) == 1, f"{indicator_type} vanished instead of being marked"
    series = out[0]
    assert series.unsupported is True
    assert series.unsupported_reason == reason
    assert series.indicator_type == indicator_type
    assert series.display_label == f"{indicator_type}({params['period']})"
    assert series.data == []


@pytest.mark.parametrize("indicator_type", ["NOT_A_REAL_TYPE", "ALSO_NOT_A_TYPE"])
def test_unknown_type_returns_a_marker_instead_of_vanishing(indicator_type: str) -> None:
    out = compute_indicators(_ohlcv(), [{"type": indicator_type, "period": 5}])

    assert len(out) == 1
    assert out[0].unsupported is True
    assert out[0].unsupported_reason == "unknown indicator type"
    assert out[0].data == []


def test_markers_are_per_indicator_and_keep_request_order() -> None:
    out = compute_indicators(
        _ohlcv(n=400),
        [{"type": "NATR", "period": 14}, {"type": "KAMA", "period": 10}],
    )

    assert [s.indicator_type for s in out] == ["NATR", "KAMA"]
    assert out[0].unsupported is True
    assert out[1].unsupported is False
    assert len(out[1].data) == 400


# Params are intentionally OMITTED here: the fallback must run the spec's
# compute_fn with the spec's default_params (the base the compiler's
# _merge_effective_params starts from), never the raw `{}` request params.
@pytest.mark.parametrize("indicator_type", [t for t, _ in _COMPUTABLE_FROM_OHLCV])
def test_registry_fallback_with_omitted_params_uses_the_spec_defaults(
    indicator_type: str,
) -> None:
    """A request that omits params must equal ``invoke_compute_fn(spec, df,
    spec.default_params)`` after the runtime warmup gate. PRICE_POSITION's
    compute_fn falls back to period 14 while its spec default is 52, so the
    raw-params call diverged. ``n=400`` is below the first buffer trim."""
    df = _ohlcv(n=400)
    out = compute_indicators(df, [{"type": indicator_type}])

    assert len(out) == 1, f"{indicator_type}: expected one series, got {len(out)}"
    series = out[0]
    assert series.unsupported is False
    assert series.params == {}  # echoed request params are unchanged

    spec = indicator_registry.get(indicator_type)
    ref = invoke_compute_fn(spec, df, spec.default_params)
    assert not isinstance(ref, dict)
    expected = ref.to_numpy(dtype=np.float64)
    assert np.isfinite(expected).any(), f"{indicator_type}: reference is all-NaN"

    lookback = pta_lookback(spec, spec.default_params)
    chart = _values(series)
    assert np.isnan(chart[: lookback - 1]).all(), f"{indicator_type}: warmup gate"
    np.testing.assert_array_equal(
        chart[lookback - 1 :], expected[lookback - 1 :], err_msg=indicator_type
    )


def test_registry_fallback_multi_output_series_follow_spec_output_order() -> None:
    """The multi-output fallback orders the returned series by
    ``spec.output_names``, not by the compute_fn dict's insertion order."""
    df = _ohlcv(n=60)

    def _compute(_df: pd.DataFrame, _params: dict[str, object]) -> dict[str, pd.Series]:
        # Dict insertion order deliberately differs from output_names below.
        return {
            "signal": _df["close"] * 3.0,
            "upper": _df["close"] * 2.0,
            "lower": _df["close"],
        }

    name = "CHART_MULTI_PROBE"
    indicator_registry.register_spec(
        IndicatorSpec(
            name=name,
            nt_class=None,
            pandas_ta_func=None,
            default_params={},
            param_schema={},
            output_names=("lower", "upper", "signal"),
            compute_fn=_compute,
            # Gate from the first bar: this test pins output order, not warmup.
            pta_lookback_fn=lambda _params: 1,
        )
    )
    try:
        out = compute_indicators(df, [{"type": name}])
        assert [s.output_name for s in out] == ["lower", "upper", "signal"]
        np.testing.assert_array_equal(_values(out[2]), (df["close"] * 3.0).to_numpy())
    finally:
        indicator_registry.unregister(name)


# ---------------------------------------------------------------------------
# Chart overlay == strategy runtime path. The generated strategy feeds each
# compute_fn a capped rolling PtaBuffer (trim once len > cap + cap // 4) and
# calls it per bar; the overlay must reproduce that stream bar for bar, not
# the full-history value.
# ---------------------------------------------------------------------------

# The registry-fallback overlay types whose runtime window truncation changes
# the value (ADAPTIVE_RSI/VIDYA) or exercises repeated trims anyway.
_CHART_PARITY_CASES = [
    ("ADAPTIVE_RSI", {"period": 14, "alpha": 0.5}),
    ("VIDYA", {"period": 14}),
    ("KAMA", {"period": 10}),
    ("FRAMA", {"period": 16}),
    ("RAMS", {"period": 50}),
    ("TREND_ENSEMBLE", {"max_lookback": 250}),
]


def _frame_from_bars(
    bars: list[tuple[str, float, float, float, float, float]],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open_time": np.arange(len(bars), dtype=np.int64) * 3_600_000,
            "open": [b[1] for b in bars],
            "high": [b[2] for b in bars],
            "low": [b[3] for b in bars],
            "close": [b[4] for b in bars],
            "volume": [b[5] for b in bars],
        }
    )


@pytest.mark.parametrize(("indicator_type", "params"), _CHART_PARITY_CASES)
def test_chart_overlay_equals_runtime_stream_bar_for_bar(
    indicator_type: str, params: dict[str, object]
) -> None:
    """The overlay equals the generated strategy's capped-buffer stream bitwise."""
    bars = random_walk_bars(3000, seed=7)
    df = _frame_from_bars(bars)

    _, cls = compile_indicators({"ind": {"type": indicator_type, **params}})
    snapshots = run_stream(
        cls, bars, memo=False, specs={"ind": (indicator_type, params, "1h")}
    )
    runtime = np.array([snap.get("ind", np.nan) for snap in snapshots], dtype=np.float64)

    out = compute_indicators(df, [{"type": indicator_type, **params}])
    assert len(out) == 1, f"{indicator_type}: expected one series, got {len(out)}"
    chart = _values(out[0])
    assert len(chart) == len(runtime)

    differing = int((~((chart == runtime) | (np.isnan(chart) & np.isnan(runtime)))).sum())
    assert np.array_equal(chart, runtime, equal_nan=True), (
        f"{indicator_type}: chart overlay differs from the runtime stream on "
        f"{differing}/{len(chart)} bars"
    )

    if indicator_type in {"ADAPTIVE_RSI", "VIDYA"}:
        # Guard against a vacuous test: for these path-dependent plugins the
        # runtime (and now the chart) last value differs from full history.
        spec = indicator_registry.get(indicator_type)
        full = invoke_compute_fn(spec, df, params)
        assert not isinstance(full, dict)
        assert np.isfinite(chart[-1])
        assert chart[-1] != float(full.iloc[-1]), (
            f"{indicator_type}: chart[-1] equals the full-history value; the "
            "capped runtime differs, so this assertion is vacuous"
        )


# Built-in types whose registry spec has ``nt_class is None`` (compute_fn path
# at runtime). Every one of them must replay the runtime's capped buffer, not a
# full-history pandas-ta formula. Random volume matters: MFI and VOLSMA read it,
# and random_walk_bars uses a constant 1.0 that hides their drift.
_MOVED_PARITY_CASES = [
    ("TEMA", {"period": 14}),
    ("MACD", {"fast_period": 12, "slow_period": 26, "signal_period": 9}),
    ("ADX", {"period": 14}),
    ("WILLR", {"period": 14}),
    ("MFI", {"period": 14}),
    ("ICHIMOKU", {"tenkan": 9, "kijun": 26, "senkou": 52}),
    ("VOLSMA", {"period": 20}),
]


def _bars_with_random_volume(
    n: int = 3000, seed: int = 7
) -> list[tuple[str, float, float, float, float, float]]:
    volumes = np.random.RandomState(seed).uniform(1.0, 100.0, n)
    return [
        (tf, o, h, lo, c, float(v))
        for (tf, o, h, lo, c, _), v in zip(random_walk_bars(n, seed=seed), volumes, strict=True)
    ]


@pytest.mark.parametrize(("indicator_type", "params"), _MOVED_PARITY_CASES)
def test_hardcoded_pta_overlay_equals_runtime_stream_bar_for_bar(
    indicator_type: str, params: dict[str, object]
) -> None:
    """Every compute_fn-path built-in overlay equals the runtime stream bitwise."""
    bars = _bars_with_random_volume()
    df = _frame_from_bars(bars)

    spec = indicator_registry.get(indicator_type)
    assert spec is not None
    outputs = spec.output_names

    _, cls = compile_indicators({"ind": {"type": indicator_type, **params}})
    snapshots = run_stream(
        cls, bars, memo=False, specs={"ind": (indicator_type, params, "1h")}
    )

    out = compute_indicators(df, [{"type": indicator_type, **params}])
    assert [s.output_name for s in out] == list(outputs), (
        f"{indicator_type}: expected one series per spec output in spec order "
        f"{outputs}, got {[s.output_name for s in out]}"
    )
    by_name = {s.output_name: s for s in out}
    assert len(by_name) == len(outputs)

    for output in outputs:
        key = "ind" if len(outputs) == 1 else f"ind_{output}"
        runtime = np.array([snap.get(key, np.nan) for snap in snapshots], dtype=np.float64)
        chart = _values(by_name[output])
        assert len(chart) == len(runtime)

        differing = int((~((chart == runtime) | (np.isnan(chart) & np.isnan(runtime)))).sum())
        assert np.array_equal(chart, runtime, equal_nan=True), (
            f"{indicator_type}/{output}: chart overlay differs from the runtime "
            f"stream on {differing}/{len(chart)} bars"
        )


# ---------------------------------------------------------------------------
# R22: every compute_fn is a pure function of df's values and params.
# ---------------------------------------------------------------------------


def _pta_candidates() -> list[str]:
    """Registry types that could reach the generated compute_fn buffer path."""
    names: list[str] = []
    for name in sorted(indicator_registry.list_indicators()):
        spec = indicator_registry.get(name)
        if spec is not None and spec.compute_fn is not None and not spec.needs_context:
            names.append(name)
    return names


_PTA_CANDIDATES = _pta_candidates()

# compute_fns whose pandas_ta call raises on a PRODUCTION context frame
# (``df.attrs["bar_close_ns"]`` is an int64 array and pandas' concat compares
# attrs dicts with ``==``). Both are NT-class specs on the compiled path.
_PANDAS_CONCAT_ATTRS_BREAK = frozenset({"ATR", "KC"})


def test_purity_candidates_are_not_vacuous() -> None:
    """A candidate list that silently empties would run both purity checks zero times."""
    assert {"WILLR", "MFI", "SMA", "ATR"} <= set(_PTA_CANDIDATES)


def _ohlcv_from_bars(bars: Sequence[BarEvent]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [b[1] for b in bars],
            "high": [b[2] for b in bars],
            "low": [b[3] for b in bars],
            "close": [b[4] for b in bars],
            "volume": [b[5] for b in bars],
        }
    )


def _tail(out: object) -> object:
    if out is None:
        return None
    if isinstance(out, dict):
        return {key: _tail(value) for key, value in out.items()}
    assert isinstance(out, pd.Series)
    assert len(out) > 0
    return float(out.iloc[-1])


def _same(left: float, right: float) -> bool:
    both_nan = (
        isinstance(left, float)
        and isinstance(right, float)
        and math.isnan(left)
        and math.isnan(right)
    )
    return both_nan or left == right


def _assert_tails_equal(left: object, right: object, label: str) -> None:
    if isinstance(left, dict):
        assert isinstance(right, dict), label
        assert list(left) == list(right), label
        for key in left:
            _assert_tails_equal(left[key], right[key], f"{label}.{key}")
        return
    if left is None or right is None:
        assert left is None and right is None, label
        return
    assert isinstance(left, float) and isinstance(right, float), label
    assert _same(left, right), f"{label}: tail {left!r} != {right!r}"


@pytest.mark.parametrize("name", _PTA_CANDIDATES)
def test_compute_fns_do_not_mutate_inputs(name: str) -> None:
    spec = indicator_registry.get(name)
    assert spec is not None and spec.compute_fn is not None
    fn = spec.compute_fn
    bars = random_walk_bars(600, seed=23)

    df = _ohlcv_from_bars(bars)
    columns = list(df.columns)
    values = {column: df[column].to_numpy(dtype=np.float64, copy=True) for column in columns}
    params = dict(spec.default_params)
    params_before = dict(params)

    _tail(fn(df, params))

    assert list(df.columns) == columns
    for column in columns:
        np.testing.assert_array_equal(
            df[column].to_numpy(dtype=np.float64), values[column], err_msg=f"{name}.{column}"
        )
    assert dict(df.attrs) == {}, f"{name} read/added df.attrs"
    assert params == params_before and list(params) == list(params_before)


def _attrs_params() -> list[Any]:
    return [
        pytest.param(
            name,
            marks=pytest.mark.xfail(
                strict=True,
                reason="pandas concat compares attrs dicts with `==` and the production "
                "bar_close_ns attr is an int64 array (ValueError inside pandas_ta "
                "true_range); ATR/KC are NT-class specs, so their compute_fn is not on "
                "the ctx+pta memo path",
            ),
        )
        if name in _PANDAS_CONCAT_ATTRS_BREAK
        else pytest.param(name)
        for name in _PTA_CANDIDATES
    ]


@pytest.mark.parametrize("name", _attrs_params())
def test_compute_fns_ignore_df_attrs(name: str) -> None:
    """The tail must not depend on df.attrs: only needs_context specs may read them."""
    spec = indicator_registry.get(name)
    assert spec is not None and spec.compute_fn is not None
    fn = spec.compute_fn
    bars = random_walk_bars(600, seed=23)
    params = dict(spec.default_params)

    plain = _tail(fn(_ohlcv_from_bars(bars), params))

    with_attrs = _ohlcv_from_bars(bars)
    with_attrs.attrs["symbol"] = "X"
    with_attrs.attrs["bar_close_ns"] = np.arange(len(bars), dtype=np.int64)
    _assert_tails_equal(plain, _tail(fn(with_attrs, params)), name)
