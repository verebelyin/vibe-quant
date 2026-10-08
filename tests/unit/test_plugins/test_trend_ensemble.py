"""Contract + behavior tests for the TREND_ENSEMBLE plugin.

Definition (Zarattini, Pagani, Barbon 2025, "Catching Crypto Trends" —
ensemble Donchian trend signals): per lookback L, the state flips to +1 on
a close above the shifted rolling-high channel, to -1 on a close below the
shifted rolling-low channel, and carries otherwise; the indicator averages
the states over the included lookbacks. The shift makes every value causal.
The published strategy is long-only with trailing stops; the symmetric -1
(short) state is this project's extension.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vibe_quant.dsl import invoke_compute_fn
from vibe_quant.dsl.indicators import indicator_registry, pta_buffer_cap, pta_lookback
from vibe_quant.dsl.plugins.trend_ensemble import (
    _shifted_donchian_channels,
    compute_trend_ensemble,
)

_LOOKBACKS = (5, 10, 20, 30, 60, 90, 150, 250)


def _df(close: np.ndarray) -> pd.DataFrame:
    close_series = pd.Series(np.asarray(close, dtype=np.float64))
    return pd.DataFrame(
        {
            "open": close_series,
            "high": close_series + 1.0,
            "low": close_series - 1.0,
            "close": close_series,
            "volume": pd.Series(np.full(len(close_series), 100.0)),
        }
    )


def test_monotonic_rise_saturates_long() -> None:
    close = 100.0 + 2.0 * np.arange(300, dtype=np.float64)
    value = compute_trend_ensemble(_df(close), {"max_lookback": 250})
    assert value.iloc[-1] == 1.0


def test_monotonic_fall_saturates_short() -> None:
    close = 1000.0 - 2.0 * np.arange(300, dtype=np.float64)
    value = compute_trend_ensemble(_df(close), {"max_lookback": 250})
    assert value.iloc[-1] == -1.0


def test_sideways_inside_channel_keeps_carried_state() -> None:
    """After an up-breakout, flat bars inside every channel carry +1.

    The carry is bounded at 8*L bars: with max_lookback=60 (lookbacks
    5/10/20/30/60) the shortest memory (L=5, 40 bars) holds +1 through bar
    118 and resets on bar 119, then L=10 resets on bar 159.
    """
    rise = 100.0 + 2.0 * np.arange(80, dtype=np.float64)
    flat = np.full(80, rise[-1])
    close = np.concatenate([rise, flat])
    value = compute_trend_ensemble(_df(close), {"max_lookback": 60})

    assert value.iloc[79] == 1.0
    assert (value.iloc[80:119] == 1.0).all()
    assert value.iloc[119] == 0.8  # L=5 forgot its breakout at 8 * 5 bars
    assert value.iloc[-1] == 0.6  # L=10 forgot too (8 * 10 bars)


def test_no_lookahead() -> None:
    """Value at bar t must be identical when computed on bars[: t + 1]."""
    rng = np.random.RandomState(7)
    close = 100.0 + np.cumsum(rng.randn(400))
    params = {"max_lookback": 250}
    full = compute_trend_ensemble(_df(close), params)

    sampler = np.random.RandomState(11)
    for t in sampler.randint(250, len(close), size=20):
        prefix = compute_trend_ensemble(_df(close[: t + 1]), params)
        assert full.iloc[t] == prefix.iloc[-1]


def test_trim_invariance() -> None:
    """Tail values must not depend on the engine's rolling-buffer trim.

    A 500-bar uptrend followed by a 3000-bar tight range: after the trend
    ends, no lookback breaks out again, so the bounded state memory (8*L
    bars) must flatten every state to 0 before the tail — identical to the
    value computed when the buffer starts mid-range (already trimmed).
    """
    trend = 100.0 + 2.0 * np.arange(500, dtype=np.float64)
    sine = trend[-1] + 0.05 * np.sin(
        2.0 * np.pi * np.arange(3000, dtype=np.float64) / 480.0
    )
    close = np.concatenate([trend, sine])
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(len(close), 100.0),
        }
    )
    params = {"max_lookback": 250}
    cap = pta_buffer_cap([pta_lookback("TREND_ENSEMBLE", params)], full_history=False)

    full = compute_trend_ensemble(df, params)
    trimmed = compute_trend_ensemble(df.iloc[-cap:], params)
    tail_full = full.iloc[-200:].to_numpy(dtype=np.float64)
    tail_trimmed = trimmed.iloc[-200:].to_numpy(dtype=np.float64)
    assert np.isfinite(tail_full).all()
    assert np.array_equal(tail_full, tail_trimmed, equal_nan=True), (
        f"trimmed buffer changes tail values: "
        f"full={tail_full[:5]}, trimmed={tail_trimmed[:5]}"
    )


def test_max_lookback_below_five_raises() -> None:
    with pytest.raises(ValueError, match="max_lookback must be >= 5"):
        compute_trend_ensemble(
            _df(100.0 + np.arange(50, dtype=np.float64)), {"max_lookback": 4}
        )


def test_matches_literal_bar_loop() -> None:
    """The vectorized state carry must equal a literal per-bar loop."""
    rng = np.random.RandomState(5)
    close = 100.0 + np.cumsum(rng.randn(260))
    high = close + 1.0
    low = close - 1.0
    max_lookback = 250
    n = len(close)

    reference = np.full((len(_LOOKBACKS), n), np.nan)
    for row, lookback in enumerate(_LOOKBACKS):
        if lookback > max_lookback:
            break
        prev = 0.0
        last_breakout = -1
        for t in range(lookback, n):
            upper = high[t - lookback : t].max()
            lower = low[t - lookback : t].min()
            if close[t] > upper:
                prev = 1.0
                last_breakout = t
            elif close[t] < lower:
                prev = -1.0
                last_breakout = t
            elif t - last_breakout >= 8 * lookback:
                prev = 0.0
            reference[row, t] = prev
    count = np.count_nonzero(~np.isnan(reference), axis=0)
    total = np.nansum(reference, axis=0)
    expected = np.where(count > 0, total / np.maximum(count, 1), np.nan)

    got = compute_trend_ensemble(_df(close), {"max_lookback": max_lookback})
    np.testing.assert_allclose(got.to_numpy(), expected, rtol=0, atol=0)


def test_channels_match_pandas_reference() -> None:
    """scipy.ndimage channels == pandas rolling(...).max().shift(1) reference."""
    rng = np.random.RandomState(13)
    close = 100.0 + np.cumsum(rng.randn(2510))
    df = _df(close)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    for lookback in _LOOKBACKS:
        upper, lower = _shifted_donchian_channels(high, low, lookback)
        expected_upper = (
            df["high"].rolling(lookback).max().shift(1).to_numpy(dtype=np.float64)
        )
        expected_lower = (
            df["low"].rolling(lookback).min().shift(1).to_numpy(dtype=np.float64)
        )
        assert np.array_equal(upper, expected_upper, equal_nan=True), lookback
        assert np.array_equal(lower, expected_lower, equal_nan=True), lookback


def test_trim_invariance_random_walk() -> None:
    """Tail values must not depend on the engine's rolling-buffer trim.

    A 6000-bar random walk: for several tail windows longer than the buffer
    cap (``df.iloc[-(cap + k):]``), the last value must equal the value from
    the full series exactly — the 8L-bounded state memory plus L bars of
    channel history fit inside the cap, so trimming never changes the tail.
    """
    rng = np.random.default_rng(7)
    n = 6000
    normal = rng.normal(size=n)
    close = 100.0 * np.exp(np.cumsum(0.01 * normal))
    open_ = np.empty(n, dtype=np.float64)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    df = pd.DataFrame(
        {
            "open": open_,
            "high": close * (1.0 + np.abs(0.003 * normal)),
            "low": close * (1.0 - np.abs(0.003 * normal)),
            "close": close,
            "volume": np.ones(n, dtype=np.float64),
        }
    )
    params = {"max_lookback": 250}
    cap = pta_buffer_cap([pta_lookback("TREND_ENSEMBLE", params)], full_history=False)

    full = compute_trend_ensemble(df, params)
    if full.iloc[-1] != 0.0:
        t = n - 1
    else:
        nonzero = np.flatnonzero(full.to_numpy(dtype=np.float64) != 0.0)
        t = int(nonzero[-1])

    tail_values = []
    for k in (0, 50, 200, cap // 4):
        window = df.iloc[t - cap - k + 1 : t + 1]
        value = compute_trend_ensemble(window, params).iloc[-1]
        assert value == full.iloc[t], f"cap+k={cap + k} changes the tail value"
        tail_values.append(value)
    assert any(value != 0.0 for value in tail_values)


def test_registered_and_contract() -> None:
    spec = indicator_registry.get("TREND_ENSEMBLE")
    assert spec is not None
    assert spec.compute_fn is compute_trend_ensemble
    assert spec.output_names == ("value",)
    assert spec.param_ranges == {"max_lookback": (60.0, 250.0)}
    assert spec.threshold_range == (-1.0, 1.0)
    assert pta_lookback(spec, {"max_lookback": 250}) == 251
    assert pta_lookback(spec, {"max_lookback": 90}) == 91

    df = _df(100.0 + np.arange(70, dtype=np.float64))
    out = invoke_compute_fn(spec, df, {"max_lookback": 60})
    assert isinstance(out, pd.Series)
    assert len(out) == len(df)
    assert np.isfinite(out.iloc[-1])
    assert -1.0 <= out.iloc[-1] <= 1.0
