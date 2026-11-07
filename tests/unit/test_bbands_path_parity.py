"""BBANDS reads the same numbers on the NT path and the compute_fn (pandas-ta) path.

Both paths are **close-based**: the NT path runs ``CloseBollingerBands`` (NT's
stock ``BollingerBands`` averages the typical price HLC3) and the compute_fn path
hands ``compute_bbands`` the close. NT's path derives ``percent_b`` / ``bandwidth``
with ``derived.py`` over its live bands; ``compute_bbands`` returns them straight
from the returned bands. A DSL threshold such as ``bb.bandwidth > 0.05`` (the GA
samples [0, 0.2]) must mean the same on both. pandas-ta's raw BBB is percent
(x100) and adds an epsilon on a flat window, so ``compute_bbands`` uses the
``derived.py`` formulas and guards instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.objects import Price, Quantity

from vibe_quant.dsl import derived
from vibe_quant.dsl.compute_builtins import compute_bbands
from vibe_quant.dsl.nt_indicators import CloseBollingerBands

_BT = BarType.from_str("BTCUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL")


def _bar(h: float, low: float, c: float, i: int) -> Bar:
    return Bar(
        bar_type=_BT,
        open=Price(c, 2),
        high=Price(h, 2),
        low=Price(low, 2),
        close=Price(c, 2),
        volume=Quantity.from_str("1"),
        ts_event=i,
        ts_init=i,
    )


def _feed_close_bollinger(
    df: pd.DataFrame, period: int, std_dev: float
) -> tuple[CloseBollingerBands, pd.DataFrame]:
    """Feed real OHLC bars through ``CloseBollingerBands`` (the NT path)."""
    nt = CloseBollingerBands(period, std_dev)
    rows = []
    for i, (h, low, c) in enumerate(zip(df["high"], df["low"], df["close"], strict=True)):
        nt.handle_bar(_bar(h, low, c, i))
        rows.append(
            {
                "upper": nt.upper,
                "middle": nt.middle,
                "lower": nt.lower,
                "percent_b": derived.compute_percent_b(nt, float(c)),
                "bandwidth": derived.compute_bandwidth(nt, float(c)),
            }
        )
    return nt, pd.DataFrame(rows)


def _flat_bars(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = np.round(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n))), 2)
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0})


def _ohlc_bars(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    close = np.round(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n))), 2)
    high = np.round(close * (1.0 + rng.uniform(0.0005, 0.01, n)), 2)
    low = np.round(close * (1.0 - rng.uniform(0.0005, 0.01, n)), 2)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": 1.0})


@pytest.mark.parametrize(("period", "std_dev"), [(20, 2.0), (10, 1.5), (30, 3.0)])
def test_bbands_match_close_bollinger_on_real_ohlc_bars(period: int, std_dev: float) -> None:
    """Close-based bands on bars with a real high-low range, upper/middle/lower.

    NT's stock ``BollingerBands`` would read hlc3 and sit ~0.1% off the close
    bands; ``CloseBollingerBands`` feeds the close, so both paths agree.
    """
    df = _ohlc_bars()
    _nt, nt_df = _feed_close_bollinger(df, period, std_dev)

    got = compute_bbands(df, {"period": period, "std_dev": std_dev})

    warm = slice(period + 5, None)
    for name in ("upper", "middle", "lower"):
        np.testing.assert_allclose(
            got[name].iloc[warm].to_numpy(),
            nt_df[name].iloc[warm].to_numpy(),
            rtol=0.0,
            atol=1e-9,
            err_msg=name,
        )
    # The old typical-price bands are far enough away to prove the bands changed.
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    old_upper = (
        typical.rolling(period).mean()
        + std_dev * typical.rolling(period).std(ddof=0)
    )
    assert float((got["upper"] - old_upper).iloc[period + 5 :].abs().max()) > 1e-4


@pytest.mark.parametrize(("period", "std_dev"), [(20, 2.0), (10, 1.5), (30, 3.0)])
def test_bbands_derived_outputs_match_nt_path_on_real_ohlc(period: int, std_dev: float) -> None:
    """percent_b / bandwidth over the close bands agree with ``derived.py`` (R6b)."""
    df = _ohlc_bars()
    _nt, nt_df = _feed_close_bollinger(df, period, std_dev)

    got = compute_bbands(df, {"period": period, "std_dev": std_dev})

    warm = slice(period + 5, None)
    for name in ("percent_b", "bandwidth"):
        np.testing.assert_allclose(
            got[name].iloc[warm].to_numpy(),
            nt_df[name].iloc[warm].to_numpy(),
            rtol=0.0,
            atol=1e-9,
            err_msg=name,
        )


@pytest.mark.parametrize("level", [100.0, 0.5, 63_000.0])
def test_flat_window_matches_nt_guards_exactly(level: float) -> None:
    """Zero-width bands: NT/derived give bandwidth 0.0 and percent_b 0.5 exactly.

    pandas-ta adds an epsilon (bandwidth 2.2e-18, percent_b 1.0) which would flip
    ``bandwidth <= 0`` / ``percent_b`` thresholds between the two paths.
    """
    n, period = 30, 20
    df = pd.DataFrame({k: np.full(n, level) for k in ("open", "high", "low", "close", "volume")})
    nt, nt_df = _feed_close_bollinger(df, period, 2.0)

    got = compute_bbands(df, {"period": period, "std_dev": 2.0})

    assert float(got["bandwidth"].iloc[-1]) == nt_df["bandwidth"].iloc[-1] == 0.0
    assert float(got["percent_b"].iloc[-1]) == nt_df["percent_b"].iloc[-1] == 0.5
    assert bool((got["bandwidth"].iloc[period:] <= 0).all())
    # Warm-up rows (bands undefined) stay NaN rather than turning into 0.0 / 0.5.
    assert got["bandwidth"].iloc[: period - 1].isna().all()
    assert got["percent_b"].iloc[: period - 1].isna().all()
    assert nt.upper == nt.middle == nt.lower == level


def test_flat_close_with_real_high_low_is_also_zero_width() -> None:
    """The bands are close-based: a flat close collapses them even when H/L move.

    The old typical-price bands would be wider than zero here (H and L differ),
    so this pins the close input on both paths, not just the flat-bar case.
    """
    n, period, level = 30, 20, 250.0
    up = np.where(np.arange(n) % 2 == 0, 2.0, 1.0)
    down = np.where(np.arange(n) % 2 == 0, 1.0, 2.0)
    df = pd.DataFrame(
        {"open": level, "high": level + up, "low": level - down, "close": level, "volume": 1.0}
    )
    _nt, nt_df = _feed_close_bollinger(df, period, 2.0)

    got = compute_bbands(df, {"period": period, "std_dev": 2.0})

    assert float(got["bandwidth"].iloc[-1]) == nt_df["bandwidth"].iloc[-1] == 0.0
    assert float(got["percent_b"].iloc[-1]) == nt_df["percent_b"].iloc[-1] == 0.5
    assert float(got["upper"].iloc[-1]) == float(got["lower"].iloc[-1]) == level


@pytest.mark.parametrize(
    ("low", "high"), [(-1.0, 1.0), (-101.0, -99.0)], ids=["zero_middle", "negative_middle"]
)
def test_non_positive_middle_matches_nt_guard_exactly(low: float, high: float) -> None:
    """Non-flat windows with middle == 0 / middle < 0: bandwidth is exactly 0.0.

    (upper - lower) / middle would be inf (middle 0) or negative (middle < 0) and
    flip ``bandwidth >= 0`` thresholds; NT/derived return the 0.0 fallback. The bands
    are wide (percent_b is a real 0.25 / 0.75, not the flat-window 0.5).
    """
    n, period = 40, 20
    close = np.where(np.arange(n) % 2 == 0, low, high)
    df = pd.DataFrame({k: close for k in ("open", "high", "low", "close", "volume")})
    _nt, nt_df = _feed_close_bollinger(df, period, 2.0)

    got = compute_bbands(df, {"period": period, "std_dev": 2.0})

    warm = slice(period + 5, None)
    assert (got["middle"].iloc[warm] <= 0).all()
    assert (got["upper"].iloc[warm] > got["lower"].iloc[warm]).all()
    assert got["bandwidth"].iloc[warm].tolist() == nt_df["bandwidth"].iloc[warm].tolist() == [
        0.0
    ] * len(nt_df["bandwidth"].iloc[warm])
    assert got["percent_b"].iloc[warm].tolist() == nt_df["percent_b"].iloc[warm].tolist()
    assert {0.25, 0.75} == set(nt_df["percent_b"].iloc[warm])
    # Warm-up rows (bands undefined) stay NaN.
    assert got["bandwidth"].iloc[: period - 1].isna().all()
    assert got["percent_b"].iloc[: period - 1].isna().all()
