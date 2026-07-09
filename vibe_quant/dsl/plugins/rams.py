"""RAMS — Risk-Adjusted Master Score.

Composite entry-quality score sourced from a Reddit r/algotrading post
(research extraction 447, item scraped 2026-05-05): "combining macro trend
(50 EMA / 200 SMA ratio), score trajectory slope over 50 days, distance
from recent 50-day local maximum, and RSI condition."

The post gave no closed formula (the scaffolder rightly refused), so this
implementation defines each component explicitly:

- ``macro``      = EMA(close, macro_fast) / SMA(close, macro_slow)
- ``trajectory`` = least-squares slope of close over ``trajectory_lookback``
  bars, divided by the window's mean close (unitless per-bar drift × 100)
- ``proximity``  = close / rolling_max(close, trajectory_lookback)  (0..1]
- ``rsi``        = RSI(trajectory_lookback // 3) / 100               (0..1)

Score = macro + trajectory + proximity + rsi. All components are causal
(rolling windows over past bars only). Author-observed range ~1.8-3.4;
with these definitions a flat market scores ~2.4 (1 + 0 + 1 + 0.5).

Usage::

    indicators:
      rams:
        type: RAMS
        period: 50
    entry_conditions:
      long:
        - rams > 2.6
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from vibe_quant.dsl.compute_builtins import int_param
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry

if TYPE_CHECKING:
    import pandas as pd

_MACRO_FAST = 50
_MACRO_SLOW = 200


def compute_rams(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """Risk-Adjusted Master Score over ``close``.

    ``period`` maps to the post's ``trajectory_lookback`` (default 50);
    macro components use the canonical 50/200 pair.
    """
    import pandas as pd

    lookback = int_param(params, "period", 50)
    close = df["close"]

    macro = close.ewm(span=_MACRO_FAST, adjust=False).mean() / close.rolling(
        _MACRO_SLOW
    ).mean()

    # Least-squares slope over the lookback window, normalized by the
    # window mean. With x = 0..n-1, slope_t = dot(window_t, weights) where
    # weights = (x - mean(x)) / var(x) — a fixed kernel, so the rolling dot
    # product is a single convolution.
    n = lookback
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    x_var = float(((x - x_mean) ** 2).sum())
    weights = (x - x_mean) / x_var  # dot(close_window, weights) = slope
    c = close.to_numpy(dtype=np.float64)
    slope_arr = np.full(c.size, np.nan, dtype=np.float64)
    if c.size >= n:
        slope_arr[n - 1 :] = np.convolve(c, weights[::-1], mode="valid")
    slope = pd.Series(slope_arr, index=close.index)
    trajectory = 100.0 * slope / close.rolling(n).mean()

    proximity = close / close.rolling(n).max()

    rsi_period = max(2, lookback // 3)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / rsi_period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50.0) / 100.0

    score = macro + trajectory + proximity + rsi
    return cast("pd.Series", pd.Series(score, index=df.index, name=f"RAMS_{lookback}"))


indicator_registry.register_spec(
    IndicatorSpec(
        name="RAMS",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"period": 50},
        param_schema={"period": int},
        compute_fn=compute_rams,
        pta_lookback_fn=lambda p: _MACRO_SLOW + int_param(p, "period", 50),
        display_name="Risk-Adjusted Master Score",
        description=(
            "Composite entry-quality score: macro trend (EMA50/SMA200) + "
            "normalized price slope + proximity to recent high + RSI. "
            "From r/algotrading research (extraction 447)."
        ),
        category="Trend",
        chart_placement="separate",
        param_ranges={"period": (20.0, 100.0)},
        threshold_range=(1.8, 3.2),
    )
)
