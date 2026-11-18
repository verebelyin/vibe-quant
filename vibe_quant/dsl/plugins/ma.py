"""MA — TradingView-exact moving average of a chosen kind and source.

``kind``: ``sma ema wma rma hma ehma thma dema tema t3 tma wwma zlema var tsf``
(definitions in ``_tv_ma.py``, each transcribed from the Pine reference or the
script that defines it). ``src`` is ``open|high|low|close|hl2|hlc3|ohlc4``,
``offset`` lags the result like Pine ``ma[offset]``. NaN until ready.

The recursive kinds (ema, rma, ehma, dema, tema, t3, wwma, zlema, var) depend on
every earlier bar, so the spec sets ``requires_full_history``: the strategy keeps
an unbounded buffer for the timeframe, for every ``kind`` (the flag is per spec).

Usage::

    indicators:
      fast:
        type: MA
        kind: ema
        period: 10
      slow:
        type: MA
        kind: hma
        period: 21
        src: hl2
    entry_conditions:
      long:
        - fast crosses_above slow
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
    tv_ma,
    tv_ma_ready,
)
from vibe_quant.dsl.warmup_class import adaptive, by_kind, ema, finite, rma

if TYPE_CHECKING:
    import pandas as pd


def compute_ma(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """``tv_ma(kind, src, period)`` lagged by ``offset`` bars."""
    import pandas as pd

    values = tv_ma(
        str_param(params, "kind", "sma"),
        source_array(df, str_param(params, "src", "close")),
        int_param(params, "period", 14),
    )
    return pd.Series(shift_back(values, int_param(params, "offset", 0)), index=df.index)


def _lookback(params: dict[str, object]) -> int:
    src = str_param(params, "src", "close")
    if src not in SOURCES:
        msg = f"MA src {src!r} unknown; valid: {', '.join(SOURCES)}"
        raise ValueError(msg)
    offset = int_param(params, "offset", 0)
    if offset < 0:
        msg = f"MA offset must be >= 0, got {offset}"
        raise ValueError(msg)
    return tv_ma_ready(str_param(params, "kind", "sma"), int_param(params, "period", 14)) + offset


indicator_registry.register_spec(
    IndicatorSpec(
        name="MA",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"period": 14, "kind": "sma", "src": "close", "offset": 0},
        param_schema={"period": int, "kind": str, "src": str, "offset": int},
        compute_fn=compute_ma,
        pta_lookback_fn=_lookback,
        # ``kind`` swaps the kernel, so the bound follows the effective kind:
        # finite windows (sma/wma/hma/thma/tma/tsf), EMA recursions
        # (ema/zlema/ehma/dema/tema/t3), Wilder rma/wwma and the adaptive var.
        paper_warmup=by_kind(
            {
                "sma": finite(),
                "wma": finite(),
                "hma": finite(),
                "hma_tv": finite(),
                "thma": finite(),
                "tma": finite(),
                "tsf": finite(),
                "ema": ema(),
                "zlema": ema(),
                "ehma": ema(),
                "dema": ema(),
                "tema": ema(),
                "t3": ema(),
                "rma": rma(),
                "wwma": rma(),
                "var": adaptive(),
            },
            param="kind",
            default="sma",
        ),
        requires_high_low=True,
        requires_full_history=True,
        display_name="Moving Average (TradingView)",
        description=(
            "TradingView-exact moving average: sma, ema, wma, rma, hma, ehma, thma, dema, "
            "tema, t3, tma, wwma, zlema, var or tsf of any price source, optionally lagged."
        ),
        category="Trend",
        chart_placement="overlay",
    )
)
