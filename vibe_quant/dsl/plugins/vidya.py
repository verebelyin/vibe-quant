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

from vibe_quant.dsl.compute_builtins import int_param
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry

if TYPE_CHECKING:
    import pandas as pd


def compute_vidya(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """VIDYA over ``close`` at the given period."""
    import pandas_ta_classic as ta

    period = int_param(params, "period", 14)
    result = ta.vidya(df["close"], length=period)
    if result is None:
        return cast("pd.Series", df["close"] * 0)
    return cast("pd.Series", result)


indicator_registry.register_spec(
    IndicatorSpec(
        name="VIDYA",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"period": 14},
        param_schema={"period": int},
        compute_fn=compute_vidya,
        pta_lookback_fn=lambda p: int(p.get("period", 14)) * 3,
        display_name="Variable Index Dynamic Avg",
        description=(
            "CMO-weighted adaptive moving average: faster when price "
            "momentum is strong, slower when flat."
        ),
        category="Trend",
        chart_placement="overlay",
        param_ranges={"period": (5.0, 50.0)},
        threshold_range=None,
    )
)
