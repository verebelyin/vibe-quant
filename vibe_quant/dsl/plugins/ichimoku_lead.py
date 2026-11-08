"""ICHIMOKU_LEAD — Ichimoku leading spans as 14_ichimoku_hull_macd.pine compares them.

Transcribes lines 19-27 (Pine v4)::

    donchian(len) => avg(lowest(len), highest(len))     # lowest(len)=lowest(low,len), highest(len)=highest(high,len)
    conversionLine = donchian(conversionPeriod)          # 9
    baseLine       = donchian(basePeriod)                # 26
    leadLine1      = avg(conversionLine, baseLine)
    leadLine2      = donchian(laggingSpanPeriod)         # 52

``avg(a, b)`` is ``(a + b) / 2``. The script's entry test is ``leadLine1 > leadLine2`` on the
UN-displaced lines (l.42, 45); ``displacement`` only offsets the plots (l.28, 77-78), so the
default here is ``offset`` 0. ``offset`` lags both outputs like Pine ``x[offset]`` (a
displaced cloud read at the current bar is ``leadLine[displacement]``).

Outputs: ``lead1`` (primary) and ``lead2``, NaN until their window exists (``lead1`` needs
``max(conversion, base)`` bars, ``lead2`` ``span`` bars). Highest/lowest are pure comparisons,
so the indicator depends on its last ``max(conversion, base, span) + offset`` bars only (finite
lookback). The runtime gate and buffer use that one scalar, so both outputs are stored from the
bar where the LATEST window exists.

Usage::

    indicators:
      cloud:
        type: ICHIMOKU_LEAD
        conversion_period: 9
        base_period: 26
        span_period: 52
    entry_conditions:
      long:
        - cloud.lead1 > cloud.lead2
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vibe_quant.dsl.compute_builtins import int_param
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry
from vibe_quant.dsl.plugins._tv_ma import rolling_max, rolling_min, shift_back

if TYPE_CHECKING:
    import pandas as pd


def _donchian(high: np.ndarray, low: np.ndarray, n: int) -> np.ndarray:
    """``avg(lowest(low, n), highest(high, n))``."""
    return np.asarray((rolling_min(low, n) + rolling_max(high, n)) / 2, dtype=np.float64)


def compute_ichimoku_lead(df: pd.DataFrame, params: dict[str, object]) -> dict[str, pd.Series]:
    """``lead1`` and ``lead2`` lagged by ``offset`` bars."""
    import pandas as pd

    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    conversion = _donchian(high, low, int_param(params, "conversion_period", 9))
    base = _donchian(high, low, int_param(params, "base_period", 26))
    lead1 = (conversion + base) / 2
    lead2 = _donchian(high, low, int_param(params, "span_period", 52))
    offset = int_param(params, "offset", 0)
    return {
        "lead1": pd.Series(shift_back(lead1, offset), index=df.index),
        "lead2": pd.Series(shift_back(lead2, offset), index=df.index),
    }


def _lookback(params: dict[str, object]) -> int:
    periods = [
        int_param(params, "conversion_period", 9),
        int_param(params, "base_period", 26),
        int_param(params, "span_period", 52),
    ]
    if min(periods) < 1:
        msg = f"ICHIMOKU_LEAD periods must be >= 1, got {periods}"
        raise ValueError(msg)
    offset = int_param(params, "offset", 0)
    if offset < 0:
        msg = f"ICHIMOKU_LEAD offset must be >= 0, got {offset}"
        raise ValueError(msg)
    # One scalar lookback gates both outputs and sizes the finite buffer (cap derives from it), so it
    # must cover the LONGEST window plus offset: the earliest output's window would trim the other's.
    return max(periods) + offset


indicator_registry.register_spec(
    IndicatorSpec(
        name="ICHIMOKU_LEAD",
        nt_class=None,
        pandas_ta_func=None,
        default_params={
            "conversion_period": 9,
            "base_period": 26,
            "span_period": 52,
            "offset": 0,
        },
        param_schema={
            "conversion_period": int,
            "base_period": int,
            "span_period": int,
            "offset": int,
        },
        compute_fn=compute_ichimoku_lead,
        pta_lookback_fn=_lookback,
        output_names=("lead1", "lead2"),
        primary_output="lead1",
        requires_high_low=True,
        display_name="Ichimoku leading spans",
        description="Ichimoku leadLine1 = avg(conversion, base) and leadLine2 = donchian(span), un-displaced.",
        category="Trend",
        chart_placement="overlay",
    )
)
