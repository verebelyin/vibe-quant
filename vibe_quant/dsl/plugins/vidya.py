"""VIDYA — Variable Index Dynamic Average (Chande 1992).

Uses the Chande Momentum Oscillator (CMO) as the adaptive smoothing
factor. Weights the EMA smoothing by abs(CMO/100) so the moving average
reacts faster when price momentum is strong.

Reference: Tushar Chande, "Adapting Moving Averages to Market
Volatility", *Technical Analysis of Stocks & Commodities*, 1992.
Thin wrapper over ``pandas_ta_classic.vidya``.

Usage::

    indicators:
      vidya_med:
        type: VIDYA
        period: 14
    entry_conditions:
      long:
        - close > vidya_med
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from vibe_quant.dsl.compute_builtins import int_param, nan_like
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry

if TYPE_CHECKING:
    import pandas as pd


def _vidya_library(df: pd.DataFrame, period: int) -> pd.Series:
    """Reference path: ``pandas_ta_classic.vidya`` (the pre-port code)."""
    import pandas_ta_classic as ta

    result = ta.vidya(df["close"], length=period)
    if result is None:
        return nan_like(df)  # insufficient data: not ready (NaN), never 0
    return cast("pd.Series", result)


def _vidya_port(df: pd.DataFrame, period: int) -> pd.Series | None:
    """Numpy port of ``pandas_ta_classic.vidya`` (default drift 1, no offset).

    Operation-for-operation with the library: ``mom = close.diff(1)``, the
    positive/negative parts, pandas' own ``rolling(period).sum()`` (the
    library's Cython accumulator -- a hand-rolled loop is NOT bit identical),
    ``|CMO| = |(ps - ns) / (ps + ns)|``, an SMA seed at ``period - 1`` and the
    ``alpha * |cmo| * close + prev * (1 - alpha * |cmo|)`` recurrence with
    ``alpha = 2 / (period + 1)`` on plain floats.

    Assumes a float64 close column with a finite seed window (the public
    ``compute_vidya`` routes everything else to the library). ``None`` below
    ``period`` rows. ``tests/unit/test_indicator_fast_paths.py`` is the
    zero-tolerance reference.
    """
    import numpy as np
    import pandas as pd

    period = period if period > 0 else 14
    n = len(df)
    if n < period:
        return None
    close = df["close"].to_numpy(dtype=np.float64)
    mom = np.empty(n, dtype=np.float64)
    mom[0] = np.nan
    mom[1:] = close[1:] - close[:-1]
    positive = np.clip(mom, 0.0, None)
    negative = np.abs(np.clip(mom, None, 0.0))
    sums = (
        pd.DataFrame(np.column_stack((positive, negative)), copy=False)
        .rolling(period)
        .sum()
        .to_numpy()
    )
    psum = sums[:, 0]
    nsum = sums[:, 1]
    with np.errstate(all="ignore"):
        abs_cmo = np.abs((psum - nsum) / (psum + nsum))
    out = np.full(n, np.nan)
    out[period - 1] = np.mean(close[:period])  # SMA seed
    alpha = 2.0 / (period + 1)
    cmo_list = abs_cmo.tolist()
    close_list = close.tolist()
    out_list = out.tolist()
    prev = out_list[period - 1]
    for i in range(period, n):
        a = cmo_list[i]
        prev = alpha * a * close_list[i] + prev * (1 - alpha * a)
        out_list[i] = prev
    return pd.Series(
        np.asarray(out_list, dtype=np.float64),
        index=df.index,
        name=f"VIDYA_{period}",
    )


def compute_vidya(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """VIDYA over ``close`` at the given period (numpy fast path).

    Falls back to ``pandas_ta_classic.vidya`` for a non-float64 close column or
    a NaN inside the SMA seed window (the library's leading-NaN skip and
    ``skipna`` seed are not reproduced here).
    """
    import numpy as np

    period = int_param(params, "period", 14)
    close = df["close"]
    if close.dtype != "float64":
        return _vidya_library(df, period)
    normalized = period if period > 0 else 14
    if len(close) < normalized:
        return nan_like(df)  # insufficient data: not ready (NaN), never 0
    if np.isnan(close.to_numpy()[:normalized]).any():
        return _vidya_library(df, period)
    return cast("pd.Series", _vidya_port(df, period))


indicator_registry.register_spec(
    IndicatorSpec(
        name="VIDYA",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"period": 14},
        param_schema={"period": int},
        compute_fn=compute_vidya,
        pta_lookback_fn=lambda p: int_param(p, "period", 14) * 3,
        display_name="Variable Index Dynamic Avg",
        description=(
            "CMO-weighted adaptive moving average: faster when price "
            "momentum is strong, slower when flat."
        ),
        category="Trend",
        chart_placement="overlay",
        param_ranges={"period": (5.0, 50.0)},
        threshold_range=None,
        ma_kind=True,
    )
)
