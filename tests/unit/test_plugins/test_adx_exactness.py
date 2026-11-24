"""compute_adx must be bit-identical to ``pandas_ta_classic.adx`` (ADX column).

The compute_fn is an operation-for-operation numpy/float port of the 0.8 library
path (true_range -> Wilder-smoothed TR/+DM/-DM (TA-Lib seeding) -> DI -> dx ->
rma(dx) seeded at the first valid dx, incl. pandas' ``ewm(adjust=False)`` loop
with NaN handling). Equality is exact (no tolerance); NaN positions must match.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
import pytest

from vibe_quant.dsl.compute_builtins import compute_adx
from vibe_quant.dsl.indicators import indicator_registry

_spec = indicator_registry.get("ADX")
assert _spec is not None and _spec.param_ranges is not None
LO, HI = _spec.param_ranges["period"]
PERIODS = list(range(int(LO), int(HI) + 1))


def _walk(seed: int, n: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    high = close + np.abs(rng.randn(n)) * 0.4
    low = close - np.abs(rng.randn(n)) * 0.4
    return pd.DataFrame({"high": high, "low": low, "close": close})


def _flat(n: int) -> pd.DataFrame:
    c = np.full(n, 50.0)
    return pd.DataFrame({"high": c, "low": c, "close": c})


def _flat_stretch(seed: int, n: int) -> pd.DataFrame:
    df = _walk(seed, n)
    a, b = n // 4, n // 2
    df.loc[a:b, ["high", "low", "close"]] = df.loc[a, "close"]  # zero-range stretch
    return df


def _nan_gaps(seed: int, n: int) -> pd.DataFrame:
    df = _walk(seed, n)
    df.loc[n // 3, "high"] = np.nan
    df.loc[n // 2 : n // 2 + 2, "close"] = np.nan
    df.loc[(2 * n) // 3, ["high", "low", "close"]] = np.nan
    return df


def _lead_nan(seed: int, n: int) -> pd.DataFrame:
    """Chained input: all three columns NaN for the first rows."""
    df = _walk(seed, n)
    df.loc[: min(4, n - 1), ["high", "low", "close"]] = np.nan
    return df


def _lead_nan_close_only(seed: int, n: int) -> pd.DataFrame:
    """Only close is NaN at first: true range blanks up to the first all-finite bar."""
    df = _walk(seed, n)
    df.loc[: min(2, n - 1), "close"] = np.nan
    return df


def _int_prices(seed: int, n: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    close = np.round(100.0 + np.cumsum(rng.randn(n)))
    return pd.DataFrame({"high": close + 1.0, "low": close - 1.0, "close": close})


def _reference(df: pd.DataFrame, period: int) -> np.ndarray:
    res = ta.adx(df["high"], df["low"], df["close"], length=period)
    if res is None:
        return np.full(len(df), np.nan)
    return np.asarray(res.iloc[:, 0].to_numpy())


def _frames(n: int) -> list[pd.DataFrame]:
    return [
        _walk(1, n),
        _walk(2, n),
        _flat(n),
        _flat_stretch(3, n),
        _nan_gaps(4, n),
        _int_prices(5, n),
        _lead_nan(6, n),
        _lead_nan_close_only(7, n),
    ]


@pytest.mark.parametrize("period", PERIODS)
def test_adx_port_matches_pandas_ta_bitwise(period: int) -> None:
    sizes = {
        15,
        40,
        400,
        period - 1,
        period,
        period + 1,
        2 * period - 1,
        2 * period,
        2 * period + 3,
    }
    for n in sorted(sizes):
        for k, df in enumerate(_frames(n)):
            expected = _reference(df, period)
            actual = compute_adx(df, {"period": period})
            assert len(actual) == n
            assert actual.index.equals(df.index)
            assert np.array_equal(actual.to_numpy(), expected, equal_nan=True), (
                f"period={period} n={n} frame={k}"
            )


@pytest.mark.parametrize("period", [1, 2])
def test_adx_degenerate_periods(period: int) -> None:
    df = _walk(9, 60)
    expected = ta.adx(df["high"], df["low"], df["close"], length=period)
    assert expected is not None
    actual = compute_adx(df, {"period": period})
    assert np.array_equal(actual.to_numpy(), expected.iloc[:, 0].to_numpy(), equal_nan=True)


@pytest.mark.parametrize("period", [0, -3])
def test_adx_non_positive_period_raises_like_library(period: int) -> None:
    """pandas-ta-classic 0.9 refuses a non-positive length instead of defaulting to 14."""
    df = _walk(9, 60)
    msg = f"adx() length must be an integer > 0, got {period}"
    with pytest.raises(ValueError, match=re.escape(msg)):
        ta.adx(df["high"], df["low"], df["close"], length=period)
    with pytest.raises(ValueError, match=re.escape(msg)):
        compute_adx(df, {"period": period})


def test_adx_name_and_index_preserved() -> None:
    df = _walk(0, 50)
    df.index = pd.RangeIndex(1000, 1050)
    out = compute_adx(df, {"period": 14})
    res = ta.adx(df["high"], df["low"], df["close"], length=14)
    assert res is not None
    assert out.name == res.iloc[:, 0].name
    assert out.index.equals(df.index)


def test_self_check_passes_here() -> None:
    from vibe_quant.dsl import compute_builtins as cb

    cb._adx_port_ok.cache_clear()
    assert cb._adx_port_ok() is True


def test_mismatch_falls_back_to_pandas(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from vibe_quant.dsl import compute_builtins as cb

    real = cb._adx_port

    def perturbed(df: pd.DataFrame, length: int) -> pd.Series:
        out = real(df, length)
        out.iloc[-1] = out.iloc[-1] + 1e-9
        return out

    monkeypatch.setattr(cb, "_adx_port", perturbed)
    cb._adx_port_ok.cache_clear()
    try:
        df = _walk(11, 80)
        with caplog.at_level("WARNING"):
            out = compute_adx(df, {"period": 14})
        assert any("falling back" in r.message for r in caplog.records)
        assert np.array_equal(out.to_numpy(), _reference(df, 14), equal_nan=True)
    finally:
        monkeypatch.undo()
        cb._adx_port_ok.cache_clear()


def test_non_float64_uses_pandas_path() -> None:
    df = _int_prices(5, 60).astype("float32")
    out = compute_adx(df, {"period": 14})
    res = ta.adx(df["high"], df["low"], df["close"], length=14)
    assert res is not None
    assert np.array_equal(out.to_numpy(), res.iloc[:, 0].to_numpy(), equal_nan=True)


def test_adx_duplicate_seed_label_falls_back_to_library() -> None:
    """The library blanks the DI seed bar BY LABEL; with a repeated label the
    positional port would blank the wrong bar, so a non-unique index must take
    the library path (exact, NaN-aware equality)."""
    rng = np.random.RandomState(11)
    close = 100.0 + np.cumsum(rng.randn(80) * 0.5)
    df = pd.DataFrame({"high": close + 0.4, "low": close - 0.4, "close": close})
    labels = np.arange(80)
    labels[50] = 13  # repeats the seed bar's label (period 14 -> seed position 13)
    df.index = pd.Index(labels)
    assert not df.index.is_unique
    expected = _reference(df, 14)
    actual = compute_adx(df, {"period": 14})
    assert np.array_equal(actual.to_numpy(), expected, equal_nan=True)
