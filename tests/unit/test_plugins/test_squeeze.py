"""Contract + behavior tests for the TTM Squeeze plugins.

SQUEEZE_RATIO = Bollinger width / Keltner width; SQUEEZE_MOM = percent-
normalised linreg endpoint of the squeeze momentum source. The hand
computation below is written in pure pandas/numpy straight from the plugin
formula — a zero-tolerance reference for the ported math.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vibe_quant.dsl.indicators import indicator_registry
from vibe_quant.dsl.plugins.squeeze import compute_squeeze_mom, compute_squeeze_ratio

_LENGTH = 20


def _frame(
    close: np.ndarray,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
) -> pd.DataFrame:
    close_s = pd.Series(np.asarray(close, dtype=np.float64))
    high_s = (
        pd.Series(np.asarray(high, dtype=np.float64))
        if high is not None
        else close_s + 0.5
    )
    low_s = (
        pd.Series(np.asarray(low, dtype=np.float64))
        if low is not None
        else close_s - 0.5
    )
    return pd.DataFrame(
        {
            "open": close_s,
            "high": high_s,
            "low": low_s,
            "close": close_s,
            "volume": pd.Series(np.full(len(close_s), 100.0)),
        }
    )


def _hand_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """Wilder ATR exactly as pandas_ta_classic: rma(true_range, length)."""
    prev_close = close.shift(1)
    ranges = pd.concat([high - low, high - prev_close, prev_close - low], axis=1)
    tr = ranges.abs().max(axis=1)
    tr.iloc[0] = np.nan  # true_range zeroes the first `drift` bars
    atr = tr.copy()
    atr.iloc[: length - 1] = np.nan
    atr.iloc[length - 1] = tr.iloc[0:length].mean()  # rma's SMA seed (skips NaN)
    return atr.ewm(alpha=1.0 / length, adjust=False).mean()


def _hand_squeeze_ratio(
    df: pd.DataFrame, length: int, bb_mult: float, kc_mult: float
) -> pd.Series:
    sd = df["close"].rolling(length).std(ddof=0)  # population std (bbands default)
    bb_width = 2.0 * bb_mult * sd
    kc_width = 2.0 * kc_mult * _hand_atr(df["high"], df["low"], df["close"], length)
    return bb_width / kc_width.replace(0.0, np.nan)


def test_squeeze_ratio_matches_hand_computation() -> None:
    rng = np.random.RandomState(42)
    n = 200
    close = pd.Series(100.0 + np.cumsum(rng.randn(n) * 0.5))
    high = close + 0.1 + rng.rand(n) * 0.4
    low = close - 0.1 - rng.rand(n) * 0.4
    df = _frame(close.to_numpy(), high.to_numpy(), low.to_numpy())

    params = {"length": _LENGTH, "bb_mult": 2.0, "kc_mult": 1.5}
    ratio = compute_squeeze_ratio(df, params)
    expected = _hand_squeeze_ratio(df, _LENGTH, 2.0, 1.5)

    np.testing.assert_allclose(
        ratio.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=0.0, equal_nan=True
    )
    # Valid (non-NaN) well within the 200-bar window once warm.
    assert ratio.iloc[_LENGTH:].notna().all()


def test_squeeze_mom_matches_hand_computation() -> None:
    rng = np.random.RandomState(42)
    n = 200
    close = pd.Series(100.0 + np.cumsum(rng.randn(n) * 0.5))
    high = close + 0.1 + rng.rand(n) * 0.4
    low = close - 0.1 - rng.rand(n) * 0.4
    df = _frame(close.to_numpy(), high.to_numpy(), low.to_numpy())

    mom = compute_squeeze_mom(df, {"length": _LENGTH})

    c = close.to_numpy()
    h = high.to_numpy()
    lo = low.to_numpy()
    src = np.full(n, np.nan)
    for t in range(_LENGTH - 1, n):
        donchian_mid = (
            h[t - _LENGTH + 1 : t + 1].max() + lo[t - _LENGTH + 1 : t + 1].min()
        ) / 2.0
        sma = c[t - _LENGTH + 1 : t + 1].mean()
        src[t] = c[t] - (donchian_mid + sma) / 2.0

    ref = np.full(n, np.nan)
    for t in range(2 * _LENGTH - 2, n):
        k, b = np.polyfit(np.arange(_LENGTH), src[t - _LENGTH + 1 : t + 1], 1)
        ref[t] = (k * (_LENGTH - 1) + b) / c[t] * 100.0

    np.testing.assert_allclose(
        mom.to_numpy(), ref, rtol=1e-9, atol=1e-12, equal_nan=True
    )
    # The source's leading length-1 NaNs push the first valid bar to 2L-2.
    assert np.isnan(mom.iloc[2 * _LENGTH - 3])
    assert np.isfinite(mom.iloc[2 * _LENGTH - 2])


def test_flat_market_ratio_is_nan_not_inf() -> None:
    n = 60
    flat = _frame(np.full(n, 100.0), np.full(n, 100.0), np.full(n, 100.0))
    ratio = compute_squeeze_ratio(flat, {"length": _LENGTH})
    assert not np.isinf(ratio.to_numpy()).any()
    # pandas_ta_classic.true_range adds float epsilon to zero high-low ranges
    # (non_zero_range), so a flat market's KC width is eps-scaled, not exactly
    # 0: BB width 0 lands on the finite ratio 0.0 — never inf, never NaN warm.
    assert np.isfinite(ratio.iloc[_LENGTH:]).all()

    # An exactly-zero KC width must be NaN (guard), not inf.
    rng = np.random.RandomState(42)
    varying = _frame(100.0 + np.cumsum(rng.randn(n) * 0.5))
    zero_kc = compute_squeeze_ratio(varying, {"length": _LENGTH, "kc_mult": 0.0})
    assert zero_kc.iloc[_LENGTH:].isna().all()


def test_contraction_drives_ratio_below_one_and_expansion_raises_it() -> None:
    n = 220
    idx = np.arange(n)
    close = 100.0 + 0.5 * np.sin(idx * 0.7)
    high = close + 1.0
    low = close - 1.0
    # Last 60 bars: volatility collapses to near zero.
    tail = np.arange(60)
    close[160:] = 100.0 + 0.01 * np.sin(tail * 0.7)
    high[160:] = close[160:] + 0.02
    low[160:] = close[160:] - 0.02
    # Final bar: sudden expansion.
    close[219] = 103.0
    high[219] = 108.0
    low[219] = 98.0

    ratio = compute_squeeze_ratio(_frame(close, high, low), {"length": _LENGTH})

    squeezed = ratio.iloc[218]
    expanded = ratio.iloc[219]
    assert squeezed < 1.0
    assert expanded > squeezed


def test_squeeze_mom_sign_follows_trend() -> None:
    n = 200
    up = _frame(100.0 + 0.5 * np.arange(n))
    down = _frame(200.0 - 0.5 * np.arange(n))

    up_mom = compute_squeeze_mom(up, {"length": _LENGTH}).iloc[-1]
    down_mom = compute_squeeze_mom(down, {"length": _LENGTH}).iloc[-1]

    assert np.isfinite(up_mom) and np.isfinite(down_mom)
    assert up_mom > 0.0
    assert down_mom < 0.0


def test_no_lookahead_both_indicators() -> None:
    rng = np.random.RandomState(7)
    n = 300
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    high = close + 0.1 + rng.rand(n) * 0.4
    low = close - 0.1 - rng.rand(n) * 0.4
    df = _frame(close, high, low)
    params = {"length": _LENGTH}

    full_ratio = compute_squeeze_ratio(df, params)
    full_mom = compute_squeeze_mom(df, params)

    ts = np.random.RandomState(11).randint(2 * _LENGTH + 5, n, size=20)
    for t in ts:
        trunc = df.iloc[: t + 1]
        np.testing.assert_allclose(
            compute_squeeze_ratio(trunc, params).iloc[t],
            full_ratio.iloc[t],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            compute_squeeze_mom(trunc, params).iloc[t],
            full_mom.iloc[t],
            rtol=0.0,
            atol=0.0,
        )


def test_registered_and_ga_eligible() -> None:
    ratio = indicator_registry.get("SQUEEZE_RATIO")
    assert ratio is not None
    assert ratio.default_params == {"length": 20, "bb_mult": 2.0, "kc_mult": 1.5}
    assert ratio.threshold_range == (0.3, 2.0)
    assert ratio.param_ranges == {
        "length": (10.0, 40.0),
        "bb_mult": (1.5, 2.5),
        "kc_mult": (1.0, 2.0),
    }
    assert ratio.compute_fn is compute_squeeze_ratio

    mom = indicator_registry.get("SQUEEZE_MOM")
    assert mom is not None
    assert mom.default_params == {"length": 20}
    assert mom.threshold_range == (-5.0, 5.0)
    assert mom.param_ranges == {"length": (10.0, 40.0)}
    assert mom.compute_fn is compute_squeeze_mom
