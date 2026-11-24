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
from vibe_quant.dsl.warmup_class import SEED_PARITY_REASON, refuse

if TYPE_CHECKING:
    import pandas as pd


def _vidya_library(df: pd.DataFrame, period: int) -> pd.Series:
    """Reference path: ``pandas_ta_classic.vidya`` (the pre-port code).

    Only non-float64 or non-finite closes and non-positive periods (the library
    raises ``ValueError``) reach it.
    """
    import pandas_ta_classic as ta

    result = ta.vidya(df["close"], length=period)
    if result is None:
        return nan_like(df)  # insufficient data: not ready (NaN), never 0
    return cast("pd.Series", result)


def _vidya_port(df: pd.DataFrame, period: int) -> pd.Series | None:
    """Numpy port of ``pandas_ta_classic.vidya`` 0.9 (default drift 1, no offset).

    Operation-for-operation with the library: ``mom = close.diff(1)``, the
    positive/negative parts, pandas' own ``rolling(period).sum()`` (the
    library's Cython accumulator -- a hand-rolled loop is NOT bit identical),
    ``|CMO| = |(ps - ns) / (ps + ns)|``, an SMA seed at ``period - 1`` and the
    ``alpha * |cmo| * close + prev * (1 - alpha * |cmo|)`` recurrence with
    ``alpha = 2 / (period + 1)`` on plain floats. A zero denominator (``period``
    equal closes) gives ``|CMO| = 0`` (TA-Lib's CMO; the library masks it the
    same way since 0.9), so the value holds and resumes on the next move.

    Assumes ``period > 0`` and a finite float64 close column (the public
    ``compute_vidya`` routes everything else to the library). ``None`` below
    ``period`` rows. ``tests/unit/test_indicator_fast_paths.py`` is the
    zero-tolerance reference.
    """
    import math

    import numpy as np
    import pandas as pd

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
    # A window of ``period`` equal closes is CMO 0/0: weight 0 holds the previous value
    # (vibe-quant-d42m5; the 0.8 library's NaN stuck for the rest of the series).
    abs_cmo[psum + nsum == 0.0] = 0.0
    out = [math.nan] * (period - 1)
    prev = float(np.mean(close[:period]))  # SMA seed
    out.append(prev)
    alpha = 2.0 / (period + 1)
    # ``alpha * a * close + prev * (1 - alpha * a)`` with its per-bar coefficients taken
    # elementwise up front (the same IEEE products, no FMA); only ``k + prev * keep``
    # stays sequential.
    weight = alpha * abs_cmo[period:]
    gain = (weight * close[period:]).tolist()
    keep = (1 - weight).tolist()
    out.extend([prev := k + prev * c for k, c in zip(gain, keep, strict=True)])
    return pd.Series(np.array(out, dtype=np.float64), index=df.index, name=f"VIDYA_{period}")


def compute_vidya(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """VIDYA over ``close`` at the given period (numpy fast path).

    Falls back to ``pandas_ta_classic.vidya`` for a non-positive period (the
    library raises ``ValueError``), a non-float64 close column or a non-finite
    close (the library's missing-bar skip is not reproduced here).
    """
    import numpy as np

    period = int_param(params, "period", 14)
    close = df["close"]
    if period < 1 or close.dtype != "float64":
        return _vidya_library(df, period)
    if len(close) < period:
        return nan_like(df)  # insufficient data: not ready (NaN), never 0
    if not np.isfinite(close.to_numpy()).all():
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
        paper_warmup=refuse(SEED_PARITY_REASON),
        catalog_origin=True,
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
