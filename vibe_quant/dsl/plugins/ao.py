"""AO — Awesome Oscillator: ``sma(hl2, fast) - sma(hl2, slow)``.

08_ao_stoch_rsi_atr.pine computes ``(sma(hl2,fast)-sma(hl2,slow))*1000`` and
compares it with ``awesome[1]``; the x1000 is a display scale (a ratio-free
comparison is unaffected), so this is the standard AO. ``offset`` lags it like
Pine ``awesome[offset]``. NaN until ``max(fast, slow) + offset`` bars exist.

Usage::

    indicators:
      ao:
        type: AO
      ao_prev:
        type: AO
        offset: 1
    entry_conditions:
      long:
        - ao > ao_prev
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vibe_quant.dsl.compute_builtins import int_param
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry
from vibe_quant.dsl.plugins._tv_ma import shift_back, source_array, tv_ma, tv_ma_ready

if TYPE_CHECKING:
    import pandas as pd


def compute_ao(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """``sma(hl2, fast_period) - sma(hl2, slow_period)`` lagged by ``offset`` bars."""
    import pandas as pd

    hl2 = source_array(df, "hl2")
    values = tv_ma("sma", hl2, int_param(params, "fast_period", 5)) - tv_ma(
        "sma", hl2, int_param(params, "slow_period", 34)
    )
    return pd.Series(shift_back(values, int_param(params, "offset", 0)), index=df.index)


def _lookback(params: dict[str, object]) -> int:
    offset = int_param(params, "offset", 0)
    if offset < 0:
        msg = f"AO offset must be >= 0, got {offset}"
        raise ValueError(msg)
    return (
        max(
            tv_ma_ready("sma", int_param(params, "fast_period", 5)),
            tv_ma_ready("sma", int_param(params, "slow_period", 34)),
        )
        + offset
    )


indicator_registry.register_spec(
    IndicatorSpec(
        name="AO",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"fast_period": 5, "slow_period": 34, "offset": 0},
        param_schema={"fast_period": int, "slow_period": int, "offset": int},
        compute_fn=compute_ao,
        pta_lookback_fn=_lookback,
        requires_high_low=True,
        display_name="Awesome Oscillator",
        description="Bill Williams' Awesome Oscillator: SMA(hl2, fast) - SMA(hl2, slow), optionally lagged.",
        category="Momentum",
        chart_placement="oscillator",
    )
)
