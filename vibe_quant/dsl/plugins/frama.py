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

from typing import TYPE_CHECKING, Any

import numpy as np

from vibe_quant.dsl.compute_builtins import int_param
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry

if TYPE_CHECKING:
    import pandas as pd


def _frama_value(
    h1: Any,
    l1: Any,
    h2: Any,
    l2: Any,
    h3: Any,
    l3: Any,
    close_i: Any,
    prev: Any,
    period: int,
    half: int,
) -> Any:
    """FRAMA value from the three window extrema (shared by every path)."""
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

    return alpha * close_i + (1.0 - alpha) * prev


def _frama_step(state: Any, inputs: Any, i: int) -> tuple[float, Any]:
    """One FRAMA window iteration (memo-extend path).

    ``state`` is ``(previous FRAMA value, period, half)``; ``inputs`` is
    ``(high, low, close)``.
    """
    prev, period, half = state
    high, low, close = inputs
    value = _frama_value(
        high[i - period + 1 : i - half + 1].max(),
        low[i - period + 1 : i - half + 1].min(),
        high[i - half + 1 : i + 1].max(),
        low[i - half + 1 : i + 1].min(),
        high[i - period + 1 : i + 1].max(),
        low[i - period + 1 : i + 1].min(),
        close[i],
        prev,
        period,
        half,
    )
    return value, (value, period, half)


def _frama_full(inputs: Any, period: int, half: int) -> tuple[Any, Any]:
    """Full recompute (n > period). Window extrema are vectorized: max/min are
    exact and NaN-propagating, so they equal the per-window slice reductions
    of :func:`_frama_step`; the per-bar arithmetic is the shared helper."""
    from numpy.lib.stride_tricks import sliding_window_view as swv

    high, low, close = inputs
    n = len(close)
    frama = np.full(n, np.nan, dtype=np.float64)
    # Seed FRAMA with the first completed-window close.
    frama[period - 1] = close[period - 1]
    hh = swv(high, half).max(axis=1).tolist()  # hh[j] = max(high[j : j + half])
    ll = swv(low, half).min(axis=1).tolist()
    hp = swv(high, period).max(axis=1).tolist()
    lp = swv(low, period).min(axis=1).tolist()
    cl = close.tolist()
    prev = float(frama[period - 1])
    for i in range(period, n):
        j = i - period + 1
        k = i - half + 1
        prev = _frama_value(hh[j], ll[j], hh[k], ll[k], hp[j], lp[j], cl[i], prev, period, half)
        frama[i] = prev
    return frama, (prev, period, half)


def compute_frama(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """Compute FRAMA per Ehlers 2005.

    Period is forced to the next lower even number because the algorithm
    splits the window in half; an odd period would bias one half.

    The per-bar rebuild is memoized on the actual (high, low, close) arrays,
    keyed by the even-rounded period: a previously seen input plus one appended
    bar runs a single :func:`_frama_step` (``prefix_memo``; kill switch
    ``VIBE_QUANT_PTA_MEMO=0``). A miss recomputes through the same step helper.
    """
    import pandas as pd

    from vibe_quant.dsl.prefix_memo import memo_run

    period = int_param(params, "period", 16)
    if period % 2 == 1:
        period -= 1
    period = max(period, 2)
    half = period // 2

    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    n = len(close)

    if n < period:
        return pd.Series(np.full(n, np.nan, dtype=np.float64), index=df.index)

    def full(inputs: Any) -> tuple[Any, Any]:
        return _frama_full(inputs, period, half)

    if n <= period:  # seed bar is the appended element: nothing to extend from
        return pd.Series(full((high, low, close))[0], index=df.index)
    out = memo_run(("frama", period), (high, low, close), full, _frama_step)
    return pd.Series(out, index=df.index)


indicator_registry.register_spec(
    IndicatorSpec(
        name="FRAMA",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"period": 16},
        param_schema={"period": int},
        compute_fn=compute_frama,
        pta_lookback_fn=lambda p: int_param(p, "period", 16) * 2,
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
        ma_kind=True,
    )
)
