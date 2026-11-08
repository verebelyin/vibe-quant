"""PRICE — a price source, optionally lagged: Pine ``src[offset]``.

``src`` is ``open|high|low|close|hl2|hlc3|ohlc4`` (the ``IndicatorConfig.source``
field only allows ``close`` for compiled indicators, so the plugin takes its own
``src`` param). ``offset`` is the number of bars back (``0`` = this bar). NaN
until ``offset + 1`` bars exist.

Usage (TradingView script 02: ``close[slowLength]``; script 14: ``price[1]``)::

    indicators:
      close_26_ago:
        type: PRICE
        offset: 26
      open_prev:
        type: PRICE
        src: open
        offset: 1
    entry_conditions:
      long:
        - close_26_ago > open_prev
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vibe_quant.dsl.compute_builtins import int_param
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry
from vibe_quant.dsl.plugins._tv_ma import SOURCES, shift_back, source_array, str_param

if TYPE_CHECKING:
    import pandas as pd


def compute_price(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """``src[offset]`` (Pine history reference); NaN for the first ``offset`` bars."""
    import pandas as pd

    offset = int_param(params, "offset", 0)
    values = shift_back(source_array(df, str_param(params, "src", "close")), offset)
    return pd.Series(values, index=df.index)


def _lookback(params: dict[str, object]) -> int:
    offset = int_param(params, "offset", 0)
    if offset < 0:
        msg = f"PRICE offset must be >= 0, got {offset}"
        raise ValueError(msg)
    src = str_param(params, "src", "close")
    if src not in SOURCES:
        msg = f"PRICE src {src!r} unknown; valid: {', '.join(SOURCES)}"
        raise ValueError(msg)
    return offset + 1


indicator_registry.register_spec(
    IndicatorSpec(
        name="PRICE",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"src": "close", "offset": 0},
        param_schema={"src": str, "offset": int},
        compute_fn=compute_price,
        pta_lookback_fn=_lookback,
        requires_high_low=True,
        display_name="Price (lagged)",
        description=(
            "A price source (open/high/low/close/hl2/hlc3/ohlc4), optionally lagged "
            "by `offset` bars: TradingView `src[offset]`."
        ),
        category="Trend",
        chart_placement="overlay",
    )
)
