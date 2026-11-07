"""STOCH reads the same numbers on the NT path and the compute_fn (pandas-ta) path.

Both paths are the TradingView **fast** stochastic: %K over ``k`` bars, ``%D =
SMA(d)`` of %K started at the first defined %K (``ta.stoch(..., smooth_k=1)``).
The NT path runs ``SmaDStochastics``; the compute_fn path runs ``compute_stoch``.
pandas-ta's default ``smooth_k=3`` would make %K a 3-bar SMA too, so the mirror
must pass ``smooth_k=1`` and let the first %D row through at ``k + d - 2``
(0-based, i.e. the call needs ``k + d - 1`` rows).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.objects import Price, Quantity

from vibe_quant.dsl.compute_builtins import compute_stoch
from vibe_quant.dsl.nt_indicators import SmaDStochastics

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


def _ohlc(n: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = np.round(100.0 + np.cumsum(rng.normal(0.0, 1.0, n)) + 50.0, 2)
    spread = rng.uniform(0.05, 1.5, n)
    high = np.round(close + spread * rng.uniform(0.0, 1.0, n), 2)
    low = np.round(close - spread * rng.uniform(0.0, 1.0, n), 2)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": 1.0})


def _feed_nt(df: pd.DataFrame, k: int, d: int) -> pd.DataFrame:
    ind = SmaDStochastics(k, d)
    rows = []
    for i, (h, low, c) in enumerate(zip(df["high"], df["low"], df["close"], strict=True)):
        ind.handle_bar(_bar(h, low, c, i))
        rows.append({"k": ind.value_k, "d": ind.value_d, "init": ind.initialized})
    return pd.DataFrame(rows)


@pytest.mark.parametrize(("k", "d"), [(14, 3), (5, 2), (14, 1)])
def test_stoch_k_and_d_match_sma_d_stochastics(k: int, d: int) -> None:
    """%K from row k-1, %D from row k+d-2, matching the NT class bit for bit."""
    df = _ohlc()
    nt_df = _feed_nt(df, k, d)

    got = compute_stoch(df, {"period": k, "d_period": d})

    k_first = k - 1
    d_first = k + d - 2
    assert k_first == got["k"].first_valid_index()
    assert d_first == got["d"].first_valid_index()
    assert nt_df["init"].idxmax() == d_first
    np.testing.assert_allclose(
        got["k"].iloc[k_first:].to_numpy(),
        nt_df["k"].iloc[k_first:].to_numpy(),
        rtol=0.0,
        atol=1e-12,
        err_msg="k",
    )
    np.testing.assert_allclose(
        got["d"].iloc[d_first:].to_numpy(),
        nt_df["d"].iloc[d_first:].to_numpy(),
        rtol=0.0,
        atol=1e-12,
        err_msg="d",
    )


def test_stoch_matches_ta_smooth_k_one_not_the_default_slow() -> None:
    """The mirror must pass smooth_k=1: ta's default (3) gives different %K/%D."""
    df = _ohlc()
    got = compute_stoch(df, {"period": 14, "d_period": 3})

    fast = df.ta.stoch(high=df["high"], low=df["low"], close=df["close"], k=14, d=3, smooth_k=1)
    slow = df.ta.stoch(high=df["high"], low=df["low"], close=df["close"], k=14, d=3)
    np.testing.assert_array_equal(got["k"].to_numpy(), fast.iloc[:, 0].to_numpy())
    np.testing.assert_array_equal(got["d"].to_numpy(), fast.iloc[:, 1].to_numpy())
    assert float((fast.iloc[:, 0] - slow.iloc[:, 0]).iloc[20:].abs().max()) > 1.0


@pytest.mark.parametrize(
    ("k", "d", "n"), [(14, 3, 15), (14, 3, 16), (14, 3, 17), (5, 2, 4), (5, 2, 5), (5, 2, 6)]
)
def test_stoch_short_input_guard_is_exact_at_the_first_defined_row(
    k: int, d: int, n: int
) -> None:
    """Below ``k + d - 1`` rows every output is NaN with no pandas-ta log; at the
    boundary the last %D row is defined (first defined %D is row ``k + d - 2``)."""
    df = _ohlc(n=n)
    got = compute_stoch(df, {"period": k, "d_period": d})

    if n < k + d - 1:
        assert got["k"].isna().all()
        assert got["d"].isna().all()
    else:
        assert got["d"].first_valid_index() == k + d - 2
        np.testing.assert_allclose(
            got["d"].iloc[[k + d - 2]].to_numpy(),
            df.ta.stoch(
                high=df["high"], low=df["low"], close=df["close"], k=k, d=d, smooth_k=1
            )
            .iloc[:, 1]
            .iloc[[k + d - 2]]
            .to_numpy(),
            rtol=0.0,
            atol=1e-12,
        )
