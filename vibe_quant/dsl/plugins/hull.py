"""HULL — Hull Suite (07_hull_suite.pine) moving average.

``mode``: ``hma`` = ``wma(2*wma(src, n/2) - wma(src, n), round(sqrt(n)))``;
``hma_tv`` = Pine's BUILT-IN ``hma``, which Likely FLOORS the root
(``wma(2*wma(src, n/2) - wma(src, n), floor(sqrt(n)))``; UNVERIFIED vs TradingView
output, no export available; 14_ichimoku_hull_macd.pine calls it, period 14: floor 3 vs
round 4); ``ehma`` = the same as ``hma`` with EMAs; ``thma`` = ``THMA(src, n/2)`` as the script's
mode switch calls it (``wma(wma(src,m/3)*3 - wma(src,m/2) - wma(src,m), m)``,
``m = n/2``). ``n/2`` is a truncated series length, as in Pine.
``src`` is ``open|high|low|close|hl2|hlc3|ohlc4``; ``offset`` lags the result
like Pine ``HULL[offset]`` (the script compares ``HULL[0]`` with ``HULL[2]``).
NaN until ready.

``ehma`` is recursive, so the spec sets ``requires_full_history`` (the strategy
keeps an unbounded buffer for the timeframe, for every ``mode``).

Usage::

    indicators:
      hull:
        type: HULL
        period: 55
        mode: hma
      hull_2_ago:
        type: HULL
        period: 55
        mode: hma
        offset: 2
    entry_conditions:
      long:
        - hull > hull_2_ago
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vibe_quant.dsl.compute_builtins import int_param
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry
from vibe_quant.dsl.plugins._tv_ma import (
    SOURCES,
    shift_back,
    source_array,
    str_param,
    tv_hull,
    tv_hull_ready,
)

if TYPE_CHECKING:
    import pandas as pd


def compute_hull(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """``tv_hull(mode, src, period)`` lagged by ``offset`` bars."""
    import pandas as pd

    values = tv_hull(
        str_param(params, "mode", "hma"),
        source_array(df, str_param(params, "src", "close")),
        int_param(params, "period", 55),
    )
    return pd.Series(shift_back(values, int_param(params, "offset", 0)), index=df.index)


def _lookback(params: dict[str, object]) -> int:
    src = str_param(params, "src", "close")
    if src not in SOURCES:
        msg = f"HULL src {src!r} unknown; valid: {', '.join(SOURCES)}"
        raise ValueError(msg)
    offset = int_param(params, "offset", 0)
    if offset < 0:
        msg = f"HULL offset must be >= 0, got {offset}"
        raise ValueError(msg)
    return tv_hull_ready(str_param(params, "mode", "hma"), int_param(params, "period", 55)) + offset


indicator_registry.register_spec(
    IndicatorSpec(
        name="HULL",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"period": 55, "mode": "hma", "src": "close", "offset": 0},
        param_schema={"period": int, "mode": str, "src": str, "offset": int},
        compute_fn=compute_hull,
        pta_lookback_fn=_lookback,
        requires_high_low=True,
        requires_full_history=True,
        display_name="Hull Suite MA",
        description=(
            "Hull moving average, Hull Suite variants (hma, ehma, thma) of any price source, "
            "optionally lagged."
        ),
        category="Trend",
        chart_placement="overlay",
    )
)
