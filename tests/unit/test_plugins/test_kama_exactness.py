"""compute_kama must be bit-identical to pandas_ta_classic.kama.

The plugin ports the library's recurrence to numpy for speed (the library
loops with ``.iloc`` per element — ~60x slower per call). Any arithmetic
divergence would silently change screening/validation results, so equality
is asserted exactly (not approximately).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
import pytest

from vibe_quant.dsl.plugins.kama import compute_kama


def _df(seed: int, n: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    close = pd.Series(100.0 + np.cumsum(rng.randn(n) * 0.5))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": pd.Series(np.full(n, 100.0)),
        }
    )


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("n", [40, 120, 500])
@pytest.mark.parametrize("period", [5, 10, 21, 50])
def test_bit_identical_to_library(seed: int, n: int, period: int) -> None:
    df = _df(seed, n)
    expected = ta.kama(df["close"], length=period)
    actual = compute_kama(df, {"period": period})
    if expected is None:
        # Series shorter than warmup: plugin returns a zeroed series
        assert (actual == 0).all()
        return
    # Exact — no tolerance. NaN positions must also match.
    np.testing.assert_array_equal(actual.to_numpy(), expected.to_numpy())
    assert actual.name == expected.name


def test_too_short_series_returns_zeroes() -> None:
    df = _df(7, 8)  # shorter than slow=30 warmup
    result = compute_kama(df, {"period": 10})
    assert len(result) == 8
    assert (result == 0).all()


def test_flat_series_exact() -> None:
    """Constant price exercises non_zero_range's zero-guard epsilon."""
    n = 100
    close = pd.Series(np.full(n, 100.0))
    df = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "volume": pd.Series(np.full(n, 1.0))}
    )
    expected = ta.kama(df["close"], length=10)
    actual = compute_kama(df, {"period": 10})
    assert expected is not None
    np.testing.assert_array_equal(actual.to_numpy(), expected.to_numpy())
