"""FRAMA — Fractal Adaptive Moving Average (Ehlers 2005).

Adjusts smoothing by the fractal dimension D of the price series over
the lookback window. D is estimated from the ratio of half-window
high-low ranges to the full-window range (Ehlers' approximation of the
Hurst exponent). When the market is trending D → 1 and smoothing tightens;
in chop D → 2 and smoothing widens.

alpha_t = exp(-4.6 * (D_t - 1))
FRAMA_t = alpha_t * close_t + (1 - alpha_t) * FRAMA_{t-1}

Reference: John Ehlers, "FRAMA — Fractal Adaptive Moving Average",
*Technical Analysis of Stocks & Commodities*, October 2005. Period must
be even (standard Ehlers formulation splits window in half).

Usage::

    indicators:
      frama_med:
        type: FRAMA
        period: 16
    entry_conditions:
      long:
        - close > frama_med
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vibe_quant.dsl.compute_builtins import int_param
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry

if TYPE_CHECKING:
    import pandas as pd


def compute_frama(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """Compute FRAMA per Ehlers 2005.

    Period is forced to the next lower even number because the algorithm
    splits the window in half; an odd period would bias one half.
    """
    import pandas as pd

    period = int_param(params, "period", 16)
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
        return pd.Series(frama, index=df.index)

    # Seed FRAMA with the first completed-window close.
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

        # Fractal dimension: log2 approximation of Hurst exponent.
        if n1 > 0 and n2 > 0 and n3 > 0:
            d = (np.log(n1 + n2) - np.log(n3)) / np.log(2.0)
            # Clamp D to its theoretical [1, 2] range — prevents alpha
            # blowing up on flat-window edge cases.
            d = max(1.0, min(2.0, d))
        else:
            d = 1.0

        alpha = np.exp(-4.6 * (d - 1.0))
        # Ehlers bounds: alpha in [0.01, 1].
        if alpha < 0.01:
            alpha = 0.01
        elif alpha > 1.0:
            alpha = 1.0

        frama[i] = alpha * close[i] + (1.0 - alpha) * frama[i - 1]

    return pd.Series(frama, index=df.index)


indicator_registry.register_spec(
    IndicatorSpec(
        name="FRAMA",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"period": 16},
        param_schema={"period": int},
        compute_fn=compute_frama,
        pta_lookback_fn=lambda p: int(p.get("period", 16)) * 2,
        requires_high_low=True,
        display_name="Fractal Adaptive MA",
        description=(
            "Ehlers' FRAMA: smoothing adapts to fractal dimension — "
            "fast in trends, slow in chop."
        ),
        category="Trend",
        chart_placement="overlay",
        param_ranges={"period": (6.0, 50.0)},
        threshold_range=None,
    )
)
