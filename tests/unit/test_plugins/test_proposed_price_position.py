"""Contract + bit-identical equivalence tests for proposed indicator PRICE_POSITION.

Generated as part of the scaffold pipeline (bd-3p1k.1.3), extended with a
zero-tolerance sweep proving the scipy.ndimage fast path is bit-identical to
the pandas rolling reference (``_price_position_pandas``) across every period
in the param range, short/long buffers, and zero-range bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vibe_quant.dsl.indicators import indicator_registry, invoke_compute_fn
from vibe_quant.dsl.plugins.proposed_price_position import (
    _price_position_pandas,
    compute_price_position,
)

# Every period in param_ranges['period'] = (20.0, 252.0), plus the clamp path.
_PERIODS = (1, 2, *range(20, 253))
_SIZE_KINDS = ("n15", "n40", "n400", "period-1", "period", "period+1")
_WALKS = ("random_walk", "flat", "mixed")


def _sample_ohlcv(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(42)
    close = 100.0 + rng.standard_normal(n).cumsum()
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.uniform(1000.0, 10000.0, n),
        },
        index=idx,
    )


def _walk_ohlcv(kind: str, n: int, seed: int) -> pd.DataFrame:
    """flat: constant price with high == low (span == 0 -> div-by-zero path).
    mixed: random walk with zero-range bars and a constant high==low plateau
    tail, so some windows have span == 0 and others don't. random_walk: plain
    walk with a fixed high/low spread."""
    idx = pd.date_range("2024-01-01", periods=max(n, 0), freq="1h")
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.standard_normal(n).cumsum()
    if kind == "flat":
        close = np.full(n, 100.0)
        high = close.copy()
        low = close.copy()
    elif kind == "mixed":
        high = close + 0.5
        low = close - 0.5
        zero_range_at = np.arange(0, n, 7)
        high[zero_range_at] = close[zero_range_at]
        low[zero_range_at] = close[zero_range_at]
        if n >= 4:
            plateau = n // 2
            close[plateau:] = close[plateau - 1]
            high[plateau:] = close[plateau - 1]
            low[plateau:] = close[plateau - 1]
    else:
        high = close + 0.5
        low = close - 0.5
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 1000.0),
        },
        index=idx,
    )


def _buffer_size(size_kind: str, period: int) -> int:
    return {
        "n15": 15,
        "n40": 40,
        "n400": 400,
        "period-1": period - 1,
        "period": period,
        "period+1": period + 1,
    }[size_kind]


def test_price_position_registered() -> None:
    spec = indicator_registry.get("PRICE_POSITION")
    assert spec is not None, "PRICE_POSITION plugin did not register"
    assert spec.compute_fn is not None


def test_price_position_contract_length_and_index() -> None:
    spec = indicator_registry.get("PRICE_POSITION")
    assert spec is not None
    df = _sample_ohlcv()
    out = invoke_compute_fn(spec, df, spec.default_params)
    assert isinstance(out, pd.Series)
    assert len(out) == len(df)
    assert out.index.equals(df.index)


def test_price_position_not_all_nan_past_warmup() -> None:
    spec = indicator_registry.get("PRICE_POSITION")
    assert spec is not None
    df = _sample_ohlcv()
    out = invoke_compute_fn(spec, df, spec.default_params)
    assert isinstance(out, pd.Series)
    # Past the second half of the series we expect at least one finite
    # value; a fully-NaN tail means the body computes nothing.
    tail = out.iloc[len(out) // 2 :]
    assert tail.notna().any(), "PRICE_POSITION produced all-NaN past warmup"


@pytest.mark.parametrize("walk", _WALKS)
@pytest.mark.parametrize("size_kind", _SIZE_KINDS)
@pytest.mark.parametrize("period", _PERIODS)
def test_price_position_bit_identical_to_pandas(
    period: int, size_kind: str, walk: str
) -> None:
    n = _buffer_size(size_kind, period)
    df = _walk_ohlcv(walk, n, seed=period * 1009 + n)
    params: dict[str, object] = {"period": period}
    old = _price_position_pandas(df, params).to_numpy(dtype=np.float64)
    new = compute_price_position(df, params).to_numpy(dtype=np.float64)
    assert np.array_equal(new, old, equal_nan=True), (
        f"drift: period={period} size_kind={size_kind} n={n} walk={walk}"
    )


def test_price_position_flat_bars_div_zero_matches_pandas() -> None:
    """high == low in every window -> span == 0; the old code masks 0-span to
    NaN (avoiding 0/0), so the fast path must produce NaN there too."""
    df = _walk_ohlcv("flat", 60, seed=3)
    params: dict[str, object] = {"period": 14}
    old = _price_position_pandas(df, params)
    new = compute_price_position(df, params)
    assert np.array_equal(
        new.to_numpy(dtype=np.float64), old.to_numpy(dtype=np.float64), equal_nan=True
    )
    assert old.iloc[13:].isna().all(), "reference: flat bars must be NaN past warmup"
    assert new.iloc[13:].isna().all(), "fast path: flat bars must be NaN past warmup"
    assert old.iloc[:13].isna().all() and new.iloc[:13].isna().all()


def test_price_position_warmup_is_nan() -> None:
    """First period-1 values are NaN (min_periods=period) on both paths."""
    df = _walk_ohlcv("random_walk", 50, seed=11)
    params: dict[str, object] = {"period": 20}
    old = _price_position_pandas(df, params)
    new = compute_price_position(df, params)
    assert old.iloc[:19].isna().all() and new.iloc[:19].isna().all()
    assert np.isfinite(new.iloc[19:]).all()
    assert np.array_equal(
        new.to_numpy(dtype=np.float64), old.to_numpy(dtype=np.float64), equal_nan=True
    )


def test_price_position_zero_span_offset_close_matches_pandas() -> None:
    """Bad-data bar: high == low over every window while close sits away from
    the pegged range -> span == 0 with a nonzero numerator. Both paths must mask
    to NaN (not +-inf) and agree with the pandas reference."""
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = 100.0 + np.arange(n, dtype=np.float64)
    flat = np.full(n, 123.45)
    df = pd.DataFrame(
        {
            "open": close,
            "high": flat,
            "low": flat,
            "close": close,
            "volume": np.full(n, 1000.0),
        },
        index=idx,
    )
    params: dict[str, object] = {"period": 14}
    old = _price_position_pandas(df, params).to_numpy(dtype=np.float64)
    new = compute_price_position(df, params).to_numpy(dtype=np.float64)
    assert np.array_equal(new, old, equal_nan=True)
    assert np.isnan(old[13:]).all(), "reference: 0-span windows must be NaN"
    assert np.isnan(new[13:]).all(), "fast path: 0-span windows must be NaN"
    assert not np.isinf(old).any() and not np.isinf(new).any()
    assert np.isnan(old[:13]).all() and np.isnan(new[:13]).all()


def test_price_position_nan_inputs_match_pandas() -> None:
    """NaN anywhere in high/low/close must match the pandas reference exactly.

    pandas rolling with min_periods=period makes any NaN-containing window NaN;
    the scipy min/max filters propagate NaN inconsistently, so the fast path must
    fall back to pandas whenever high/low contain NaN. NaN close only poisons
    its own row's numerator on both paths, so close alone needs no fallback —
    the close-only group below proves that stays true."""
    rng = np.random.default_rng(20261009)
    frame = 0
    for cols in (("high",), ("low",), ("close",), ("high", "low", "close")):
        for _ in range(75):
            n = int(rng.integers(5, 80))
            period = int(rng.integers(1, 30))
            df = _walk_ohlcv("random_walk", n, seed=frame)
            for _ in range(int(rng.integers(0, 4))):
                col = cols[int(rng.integers(0, len(cols)))]
                df.loc[df.index[int(rng.integers(0, n))], col] = np.nan
            params: dict[str, object] = {"period": period}
            old = _price_position_pandas(df, params).to_numpy(dtype=np.float64)
            new = compute_price_position(df, params).to_numpy(dtype=np.float64)
            assert np.array_equal(new, old, equal_nan=True), (
                f"NaN drift: frame={frame} cols={cols} n={n} period={period}"
            )
            frame += 1
