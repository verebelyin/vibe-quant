"""TAKER_IMBALANCE -- taker-buy share of volume from archived 1m klines.

``sum(taker_buy_volume) / sum(volume) - 0.5`` over the last ``period`` MINUTES
of archived 1m klines, looked up as-of each bar close through
``vibe_quant.dsl.aux_data`` (``needs_context=True``): a kline closes at
``open_time + 60s``, so the value at a bar closing at ``t`` uses klines closed
in ``(t - period*60s, t]`` and never a kline that opens at ``t``.

NaN unless exactly ``period`` consecutive klines end at ``t`` and their volume
sum is positive. The series is never carried forward.

Screening/validation only; paper/live refuse strategies that use it.

Usage::

    indicators:
      ti:
        type: TAKER_IMBALANCE
        period: 60
    entry_conditions:
      long:
        - ti > 0.05
"""

from __future__ import annotations

import pandas as pd

from vibe_quant.dsl import aux_data
from vibe_quant.dsl.compute_builtins import int_param
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry


def compute_taker_imbalance(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """Taker-buy share of volume minus 0.5 over the last ``period`` minutes."""
    symbol, close_ns = aux_data.context_of(df)
    period = int_param(params, "period", 60)
    return pd.Series(aux_data.taker_imbalance_asof(symbol, close_ns, period), index=df.index)


indicator_registry.register_spec(
    IndicatorSpec(
        name="TAKER_IMBALANCE",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"period": 60},
        param_schema={"period": int},
        compute_fn=compute_taker_imbalance,
        pta_lookback_fn=lambda p: 1,
        needs_context=True,
        asof_context=True,
        display_name="Taker Buy Imbalance",
        description="Taker-buy share of volume minus 0.5 over the last N minutes, as of the bar close.",
        category="Volume",
        param_ranges={"period": (5.0, 240.0)},
        threshold_range=(-0.15, 0.15),
    )
)
