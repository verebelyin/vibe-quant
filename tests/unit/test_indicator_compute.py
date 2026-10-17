"""Chart-overlay indicator computation (``vibe_quant.data.indicator_compute``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta_classic  # noqa: F401 — registers .ta accessor
import pytest

from vibe_quant.data.indicator_compute import compute_indicators
from vibe_quant.dsl.compute_builtins import compute_bbands
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry, invoke_compute_fn


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
    """Values must equal ``invoke_compute_fn(spec, df, params)`` bit for bit —
    the same function and params the strategy compiler uses."""
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

    np.testing.assert_array_equal(_values(series), expected, err_msg=indicator_type)


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
    spec.default_params)``. PRICE_POSITION's compute_fn falls back to period
    14 while its spec default is 52, so the raw-params call diverged."""
    df = _ohlcv(n=600)
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

    np.testing.assert_array_equal(_values(series), expected, err_msg=indicator_type)


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
        )
    )
    try:
        out = compute_indicators(df, [{"type": name}])
        assert [s.output_name for s in out] == ["lower", "upper", "signal"]
        np.testing.assert_array_equal(_values(out[2]), (df["close"] * 3.0).to_numpy())
    finally:
        indicator_registry.unregister(name)
