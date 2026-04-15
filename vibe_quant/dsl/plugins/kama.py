"""KAMA — Kaufman Adaptive Moving Average.

Perry Kaufman's adaptive smoothing applied to price. Uses the efficiency
ratio (ER) — abs(direction) / sum(abs(deltas)) — to blend between a
fast and slow EMA smoothing constant. In a trend, ER approaches 1 and
smoothing tightens; in chop, ER approaches 0 and smoothing widens.

Reference: Perry Kaufman, *Trading Systems and Methods*, 5th ed. (2013).
Thin wrapper over ``pandas_ta_classic.kama``.

Usage::

    indicators:
      kama_fast:
        type: KAMA
        period: 10
    entry_conditions:
      long:
        - close > kama_fast
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from vibe_quant.dsl.compute_builtins import int_param
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry

if TYPE_CHECKING:
    import pandas as pd


def compute_kama(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """KAMA over ``close`` at the given period. Fast/slow periods follow
    Kaufman's canonical 2/30 defaults inside pandas-ta-classic."""
    import pandas_ta_classic as ta

    period = int_param(params, "period", 10)
    result = ta.kama(df["close"], length=period)
    if result is None:
        return cast("pd.Series", df["close"] * 0)
    return cast("pd.Series", result)


indicator_registry.register_spec(
    IndicatorSpec(
        name="KAMA",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"period": 10},
        param_schema={"period": int},
        compute_fn=compute_kama,
        pta_lookback_fn=lambda p: int(p.get("period", 10)) * 3,
        display_name="Kaufman Adaptive MA",
        description=(
            "Adaptive moving average that tightens in trends and widens "
            "in chop via the Kaufman efficiency ratio."
        ),
        category="Trend",
        chart_placement="overlay",
        param_ranges={"period": (5.0, 50.0)},
        threshold_range=None,
    )
)
