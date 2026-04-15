"""Golden-value + contract tests for KAMA, VIDYA, FRAMA plugins (bd-fvbo).

Values pinned against a deterministic 100-bar fixture (seed=42). Golden
values drift will catch algorithm regressions (e.g. off-by-one in the
window split, wrong smoothing-constant formula); tolerances are loose
enough to absorb float platform differences.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vibe_quant.dsl.indicators import indicator_registry
from vibe_quant.dsl.plugins.frama import compute_frama
from vibe_quant.dsl.plugins.kama import compute_kama
from vibe_quant.dsl.plugins.vidya import compute_vidya


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
    assert abs(result.iloc[20] - 84.754) < 0.5, f"bar 20: {result.iloc[20]:.4f}"
    assert abs(result.iloc[50] - 95.115) < 0.5, f"bar 50: {result.iloc[50]:.4f}"
    assert abs(result.iloc[80] - 95.715) < 0.5, f"bar 80: {result.iloc[80]:.4f}"

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
    assert abs(result.iloc[20] - 23.016) < 1.0, f"bar 20: {result.iloc[20]:.4f}"
    assert abs(result.iloc[50] - 84.242) < 1.0, f"bar 50: {result.iloc[50]:.4f}"
    assert abs(result.iloc[80] - 90.038) < 1.0, f"bar 80: {result.iloc[80]:.4f}"


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
