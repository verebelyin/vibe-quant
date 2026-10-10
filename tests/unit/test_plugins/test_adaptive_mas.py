"""Golden-value + contract tests for KAMA, VIDYA, FRAMA plugins (bd-fvbo).

Values pinned against a deterministic 100-bar fixture (seed=42). Golden
values drift will catch algorithm regressions (e.g. off-by-one in the
window split, wrong smoothing-constant formula); tolerances are loose
enough to absorb float platform differences.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pta_stream_harness import (  # noqa: E402
    compile_indicators,
    memo_env,
    random_walk_bars,
    run_stream,
)

from vibe_quant.dsl.indicators import indicator_registry  # noqa: E402
from vibe_quant.dsl.plugins.frama import compute_frama  # noqa: E402
from vibe_quant.dsl.plugins.kama import compute_kama  # noqa: E402
from vibe_quant.dsl.plugins.vidya import compute_vidya  # noqa: E402
from vibe_quant.dsl.prefix_memo import MEMO  # noqa: E402


@pytest.fixture
def fixture_df() -> pd.DataFrame:
    """Deterministic OHLCV — same seed as test_plugin_end_to_end.py so
    values can be cross-referenced."""
    rng = np.random.RandomState(42)
    close = 100.0 + np.cumsum(rng.randn(100) * 0.5)
    return pd.DataFrame(
        {
            "open": close - rng.rand(100) * 0.2,
            "high": close + rng.rand(100) * 0.5,
            "low": close - rng.rand(100) * 0.5,
            "close": close,
            "volume": rng.randint(100, 1000, 100).astype(float),
        }
    )


# ---------------------------------------------------------------------------
# Registration / spec metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["KAMA", "VIDYA", "FRAMA"])
def test_plugin_registered(name: str) -> None:
    spec = indicator_registry.get(name)
    assert spec is not None, f"{name} missing from registry"
    assert spec.category == "Trend"
    assert spec.chart_placement == "overlay"
    # MAs are price-level — excluded from GA (threshold_range=None).
    assert spec.threshold_range is None


# ---------------------------------------------------------------------------
# KAMA
# ---------------------------------------------------------------------------


def test_kama_golden_values(fixture_df: pd.DataFrame) -> None:
    """KAMA thin-wraps pandas_ta_classic.kama. Pin a few bars to catch
    upstream regressions."""
    result = compute_kama(fixture_df, {"period": 10})
    assert isinstance(result, pd.Series)
    assert len(result) == 100

    # Early bars (warmup) should be NaN or the seed zero.
    assert pd.isna(result.iloc[0])

    valid = result.dropna()
    assert len(valid) > 50

    # Pin values (hand-verified from upstream pandas_ta_classic.kama).
    # pandas-ta-classic 0.6 seeds KAMA from price instead of ramping from 0,
    # so warm-up bars now track close (old 0.3.x pins were ramp artifacts).
    assert abs(result.iloc[20] - 99.559) < 0.5, f"bar 20: {result.iloc[20]:.4f}"
    assert abs(result.iloc[50] - 95.520) < 0.5, f"bar 50: {result.iloc[50]:.4f}"
    assert abs(result.iloc[80] - 95.799) < 0.5, f"bar 80: {result.iloc[80]:.4f}"

    # KAMA tracks price — max deviation should be modest.
    close_final = fixture_df["close"].iloc[-1]
    assert abs(result.iloc[-1] - close_final) < 10.0


# ---------------------------------------------------------------------------
# VIDYA
# ---------------------------------------------------------------------------


def test_vidya_golden_values(fixture_df: pd.DataFrame) -> None:
    """VIDYA wraps pandas_ta_classic.vidya (CMO-adaptive EMA)."""
    result = compute_vidya(fixture_df, {"period": 14})
    assert isinstance(result, pd.Series)
    assert len(result) == 100

    valid = result.dropna()
    assert len(valid) > 50

    # Pin values.
    # pandas-ta-classic 0.6 seeds VIDYA from price instead of ramping from 0,
    # so warm-up bars now track close (old 0.3.x pins were ramp artifacts).
    assert abs(result.iloc[20] - 100.816) < 1.0, f"bar 20: {result.iloc[20]:.4f}"
    assert abs(result.iloc[50] - 96.786) < 1.0, f"bar 50: {result.iloc[50]:.4f}"
    assert abs(result.iloc[80] - 96.101) < 1.0, f"bar 80: {result.iloc[80]:.4f}"


def test_vidya_deterministic_repeated_call(fixture_df: pd.DataFrame) -> None:
    """bd-r8i7: rule out pandas-ta VIDYA as a non-determinism source.

    If called twice on identical input, VIDYA must return bit-identical
    output. A failure here would explain discovery→screening Sharpe
    drift on VIDYA champions.
    """
    r1 = compute_vidya(fixture_df, {"period": 14})
    r2 = compute_vidya(fixture_df, {"period": 14})
    pd.testing.assert_series_equal(r1, r2, check_exact=True)


def test_vidya_incremental_buffer_stability(fixture_df: pd.DataFrame) -> None:
    """bd-r8i7: VIDYA[t] must not depend on buffer length past t.

    The compiler's per-bar dispatcher re-invokes compute_vidya on the
    entire growing close buffer each bar, reading only ``.iloc[-1]``.
    So VIDYA(close[:t+1]).iloc[-1] must equal VIDYA(close[:t+k]).iloc[t]
    for any k>=1. If not, the recursive ``vidya.iloc[i-1]`` accumulator
    produces different values for the same bar depending on how many
    future bars are in the series — which would make every VIDYA-based
    champion non-reproducible.
    """
    for t in (30, 60, 90):
        for k in (1, 5, 20):
            if t + k >= len(fixture_df):
                continue
            short = compute_vidya(fixture_df.iloc[: t + 1], {"period": 14})
            long = compute_vidya(fixture_df.iloc[: t + 1 + k], {"period": 14})
            v_short = short.iloc[-1]
            v_long = long.iloc[t]
            if pd.isna(v_short) and pd.isna(v_long):
                continue
            assert v_short == v_long, (
                f"buffer-length non-determinism: t={t} k={k} "
                f"short={v_short!r} long={v_long!r}"
            )


# ---------------------------------------------------------------------------
# FRAMA
# ---------------------------------------------------------------------------


def test_frama_golden_values(fixture_df: pd.DataFrame) -> None:
    """FRAMA — custom Ehlers implementation. Golden values from initial
    hand-verified run; regressions here signal a change to the fractal
    dimension or smoothing formulas."""
    result = compute_frama(fixture_df, {"period": 16})
    assert isinstance(result, pd.Series)
    assert len(result) == 100

    # First `period-1` values should be NaN.
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[14])

    valid = result.dropna()
    assert len(valid) > 50

    # Output tracks price — within a reasonable band.
    min_close = float(fixture_df["close"].min())
    max_close = float(fixture_df["close"].max())
    assert valid.min() >= min_close - 2.0, f"below price floor: {valid.min():.4f}"
    assert valid.max() <= max_close + 2.0, f"above price ceiling: {valid.max():.4f}"

    # Pin values (hand-verified against the 100-bar seed=42 fixture).
    assert abs(result.iloc[20] - 99.082) < 0.5, f"bar 20: {result.iloc[20]:.4f}"
    assert abs(result.iloc[50] - 94.799) < 0.5, f"bar 50: {result.iloc[50]:.4f}"
    assert abs(result.iloc[80] - 95.847) < 0.5, f"bar 80: {result.iloc[80]:.4f}"


def test_frama_odd_period_forced_even(fixture_df: pd.DataFrame) -> None:
    """Ehlers' FRAMA requires period even (half-window split); odd
    period should be silently coerced to period-1."""
    r_even = compute_frama(fixture_df, {"period": 16})
    r_odd = compute_frama(fixture_df, {"period": 17})
    # With period=17 → forced to 16, results identical.
    pd.testing.assert_series_equal(r_even, r_odd)


def test_frama_flat_market_converges() -> None:
    """In a perfectly flat market, FRAMA should not NaN after warmup —
    the fractal dimension falls back to 1.0 (perfect trend), alpha=1,
    output = close."""
    n = 50
    df = pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [100.0] * n,
            "low": [100.0] * n,
            "close": [100.0] * n,
            "volume": [1000.0] * n,
        }
    )
    result = compute_frama(df, {"period": 16})
    # After warmup, output should be flat at 100.
    tail = result.iloc[20:]
    assert (tail == 100.0).all(), f"non-flat output: {tail.unique()}"


# ---------------------------------------------------------------------------
# FRAMA memo (vibe-quant-yul7u.3): bitwise vs the original unmemoized loop
# ---------------------------------------------------------------------------


def _frama_reference(df: pd.DataFrame, period: int) -> np.ndarray:
    """Verbatim copy of the pre-memo compute_frama loop (independent reference)."""
    if period % 2 == 1:
        period -= 1
    period = max(period, 2)
    half = period // 2
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    n = len(close)
    frama = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return frama
    frama[period - 1] = close[period - 1]
    for i in range(period, n):
        h1 = high[i - period + 1 : i - half + 1].max()
        l1 = low[i - period + 1 : i - half + 1].min()
        h2 = high[i - half + 1 : i + 1].max()
        l2 = low[i - half + 1 : i + 1].min()
        h3 = high[i - period + 1 : i + 1].max()
        l3 = low[i - period + 1 : i + 1].min()
        n1 = (h1 - l1) / half if half > 0 else 0.0
        n2 = (h2 - l2) / half if half > 0 else 0.0
        n3 = (h3 - l3) / period
        if n1 > 0 and n2 > 0 and n3 > 0:
            d = (np.log(n1 + n2) - np.log(n3)) / np.log(2.0)
            d = max(1.0, min(2.0, d))
        else:
            d = 1.0
        alpha = np.exp(-4.6 * (d - 1.0))
        if alpha < 0.01:
            alpha = 0.01
        elif alpha > 1.0:
            alpha = 1.0
        frama[i] = alpha * close[i] + (1.0 - alpha) * frama[i - 1]
    return frama


def _bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64).view(np.uint64)


def _ohlc(n: int, seed: int = 5) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    high = close + np.abs(rng.randn(n)) * 0.4
    low = close - np.abs(rng.randn(n)) * 0.4
    # flat windows (zero range, D falls back to 1) and a clamped-alpha stretch
    high[60:90] = low[60:90] = close[60:90] = close[59]
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close})


def _stream_frames(df: pd.DataFrame, cap: int, lookback: int) -> list[pd.DataFrame]:
    """Rolling frames on the generated buffer schedule (grow to cap+cap//4, cut to cap)."""
    frames = []
    start = 0
    for end in range(1, len(df) + 1):
        if end - start > cap + cap // 4:
            start = end - cap
        if end - start >= lookback:
            frames.append(df.iloc[start:end].reset_index(drop=True))
    return frames


@pytest.mark.parametrize(("period", "cap"), [(16, 120), (17, 120), (6, 40), (50, 300)])
def test_frama_memo_full_array_bitwise_over_buffer_schedule(period: int, cap: int) -> None:

    df = _ohlc(cap * 3)
    df.loc[150, "high"] = np.nan  # NaN gap poisons max() -> NaN path
    with memo_env(True):
        for frame in _stream_frames(df, cap, 2 * period):
            got = compute_frama(frame, {"period": period}).to_numpy()
            assert np.array_equal(_bits(got), _bits(_frama_reference(frame, period)))
        assert MEMO.hits > len(df) // 2  # memo really engaged (also across buffer trims)


def test_frama_memo_key_is_even_rounded_period() -> None:
    """period 17 and 16 compute the same series, so they share one memo key."""

    df = _ohlc(200)
    with memo_env(True):
        compute_frama(df.iloc[:100], {"period": 16})
        compute_frama(df.iloc[:101], {"period": 17})
        assert MEMO.hits == 1
        assert MEMO.n_slots(("frama", 16)) == 1
        compute_frama(df.iloc[:102], {"period": 18})  # different series: miss, own key
        assert MEMO.hits == 1
        assert MEMO.n_slots(("frama", 18)) == 1


def test_frama_memo_stream_bitwise() -> None:
    """Compiled strategy stream (real _feed_pta_buffer, trims): memo on == memo off, bitwise."""

    specs = {
        "frama_a": ("FRAMA", {"period": 17}, "1h"),
        "frama_b": ("FRAMA", {"period": 10}, "1h"),
    }
    _, cls = compile_indicators(
        {
            "frama_a": {"type": "FRAMA", "period": 17},
            "frama_b": {"type": "FRAMA", "period": 10},
        }
    )
    bars = list(random_walk_bars(1500, seed=13))
    flat = bars[700][1:5]
    for i in range(700, 740):  # flat windows
        bars[i] = ("1h", *flat, 1.0)
    bars[900] = ("1h", bars[900][1], float("nan"), bars[900][3], bars[900][4], 1.0)  # NaN high
    on = run_stream(cls, bars, True, specs=specs)
    hits = MEMO.hits
    off = run_stream(cls, bars, False, specs=specs)
    assert hits > len(bars)  # both indicators hit nearly every bar
    assert len(on) == len(off) == len(bars)
    assert any("frama_a" in d for d in on)
    for a, b in zip(on, off, strict=True):
        assert a.keys() == b.keys()
        for k in a:
            assert np.float64(a[k]).view(np.uint64) == np.float64(b[k]).view(np.uint64)


def test_frama_memo_changed_prefix_misses() -> None:
    """Same length, one early element differs (also -0.0 vs +0.0): must miss, not extend."""
    df = _ohlc(120)
    changed = df.copy()
    changed.loc[5, "low"] -= 0.3  # early element, inside the recurrence history
    with memo_env(True):
        compute_frama(df.iloc[:100], {"period": 16})
        got = compute_frama(changed.iloc[:101], {"period": 16}).to_numpy()
        assert MEMO.hits == 0
        assert np.array_equal(_bits(got), _bits(_frama_reference(changed.iloc[:101], 16)))
        compute_frama(df.iloc[:101], {"period": 16})  # true extension of a stored prefix
        assert MEMO.hits == 1
