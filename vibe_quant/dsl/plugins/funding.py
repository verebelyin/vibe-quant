"""FUNDING / FUNDING_Z -- perpetual funding-rate context indicators.

Both read archived Binance funding settlements through
``vibe_quant.dsl.aux_data`` (``needs_context=True``): no look-ahead, a bar
only sees settlements at or before its close.

* ``FUNDING``: last settled rate in basis points (0.01% == 1 bps); optional
  ``period`` > 1 averages that many settlements (default 1 = last settled).
* ``FUNDING_Z``: z-score of the rate over the last ``period`` settlements
  (8h each), computed on the full settlement series, not the bar buffer.

Screening/validation only; paper/live refuse strategies that use them.

Usage::

    indicators:
      fz:
        type: FUNDING_Z
        period: 30
    entry_conditions:
      long:
        - fz < -2
"""

from __future__ import annotations

import pandas as pd

from vibe_quant.dsl import aux_data
from vibe_quant.dsl.compute_builtins import int_param
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry


def compute_funding(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """Last settled funding rate (bps; mean of last ``period`` settlements) as of each bar close."""
    symbol, close_ns = aux_data.context_of(df)
    period = int_param(params, "period", 1)
    return pd.Series(aux_data.funding_asof(symbol, close_ns, period) * 1e4, index=df.index)


def compute_funding_z(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """Funding z-score over ``period`` settlements as of each bar close."""
    symbol, close_ns = aux_data.context_of(df)
    period = int_param(params, "period", 30)
    return pd.Series(aux_data.funding_z_asof(symbol, close_ns, period), index=df.index)


indicator_registry.register_spec(
    IndicatorSpec(
        name="FUNDING",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"period": 1},
        param_schema={"period": int},
        compute_fn=compute_funding,
        pta_lookback_fn=lambda p: 1,
        needs_context=True,
        display_name="Funding Rate (bps)",
        description="Last settled perpetual funding rate in basis points.",
        category="Custom",
        param_ranges={"period": (1.0, 6.0)},
        threshold_range=(-5.0, 5.0),
    )
)

indicator_registry.register_spec(
    IndicatorSpec(
        name="FUNDING_Z",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"period": 30},
        param_schema={"period": int},
        compute_fn=compute_funding_z,
        pta_lookback_fn=lambda p: 1,
        needs_context=True,
        display_name="Funding Z-Score",
        description="Z-score of the funding rate over the last N 8h settlements.",
        category="Custom",
        param_ranges={"period": (10.0, 90.0)},
        threshold_range=(-3.0, 3.0),
    )
)
