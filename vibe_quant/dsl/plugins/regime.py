"""BTC_TREND / BTC_ROC / OWN_TREND -- cross-asset daily regime context indicators.

Read daily closes through ``vibe_quant.dsl.aux_data`` (``needs_context=True``).
Series are computed once on the FULL daily history and looked up as-of the
strategy bar close: a bar sees only daily bars that closed at or before it
(a 4h bar closing 20:00 on day D sees day D-1, the 00:00 bar of D+1 sees D).
NaN when the as-of daily bar is more than 2 days old (data hole).

* ``BTC_TREND``: BTCUSDT daily ``close / EMA(close, span=period) - 1`` (default 200).
* ``BTC_ROC``: BTCUSDT daily ``close / close[period bars ago] - 1`` (default 20).
* ``OWN_TREND``: same as BTC_TREND on the strategy's own symbol.

Screening/validation only; paper/live refuse strategies that use them.

Usage::

    indicators:
      regime:
        type: BTC_TREND
        period: 200
    entry_conditions:
      long:
        - regime > 0
"""

from __future__ import annotations

import pandas as pd

from vibe_quant.dsl import aux_data
from vibe_quant.dsl.compute_builtins import int_param
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry


def compute_btc_trend(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """BTC daily close / EMA(period) - 1, as of each bar close."""
    _, close_ns = aux_data.context_of(df)
    period = int_param(params, "period", 200)
    return pd.Series(aux_data.ref_trend_asof(aux_data.REF_SYMBOL, close_ns, period), index=df.index)


def compute_btc_roc(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """BTC daily rate of change over ``period`` daily bars, as of each bar close."""
    _, close_ns = aux_data.context_of(df)
    period = int_param(params, "period", 20)
    return pd.Series(aux_data.ref_roc_asof(aux_data.REF_SYMBOL, close_ns, period), index=df.index)


def compute_own_trend(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """Own-symbol daily close / EMA(period) - 1, as of each bar close."""
    symbol, close_ns = aux_data.context_of(df)
    period = int_param(params, "period", 200)
    return pd.Series(aux_data.ref_trend_asof(symbol, close_ns, period), index=df.index)


def _register(name: str, fn: object, period: int, rng: tuple[float, float], title: str, desc: str) -> None:
    indicator_registry.register_spec(
        IndicatorSpec(
            name=name,
            nt_class=None,
            pandas_ta_func=None,
            default_params={"period": period},
            param_schema={"period": int},
            compute_fn=fn,  # type: ignore[arg-type]
            pta_lookback_fn=lambda p: 1,
            needs_context=True,
            display_name=title,
            description=desc,
            category="Custom",
            param_ranges={"period": rng},
            threshold_range=(-0.5, 0.5),
        )
    )


_register(
    "BTC_TREND", compute_btc_trend, 200, (20.0, 250.0), "BTC Daily Trend",
    "BTCUSDT daily close / EMA(period) - 1, as of the last closed daily bar.",
)
_register(
    "BTC_ROC", compute_btc_roc, 20, (5.0, 60.0), "BTC Daily ROC",
    "BTCUSDT daily rate of change over N daily bars, as of the last closed daily bar.",
)
_register(
    "OWN_TREND", compute_own_trend, 200, (20.0, 250.0), "Own-Symbol Daily Trend",
    "Strategy symbol's daily close / EMA(period) - 1, as of the last closed daily bar.",
)
