"""OI_CHANGE / LS_RATIO_Z -- Binance Vision positioning metrics as context indicators.

Both read archived 5-minute ``metrics`` rows through ``vibe_quant.dsl.aux_data``
(``needs_context=True``). A row stamped ``s`` is usable from bar close ``s + 600 s``
(publication lag) and is dropped once the close is 600 s past that, so one missing
row is bridged and two are not. Periods count 5-minute ROWS, whatever the bar
timeframe.

* ``OI_CHANGE``: ``sum_open_interest`` (contracts) of the as-of row over the row
  stamped ``period`` rows earlier, minus 1. NaN if that row is missing.
* ``LS_RATIO_Z``: z-score of ``sum_toptrader_long_short_ratio`` over ``period``
  contiguous rows ending at the as-of row (NaN over any hole).

Screening/validation only; paper/live refuse strategies that use them.

Usage::

    indicators:
      oi:
        type: OI_CHANGE
        period: 12
    entry_conditions:
      long:
        - oi < -0.002
"""

from __future__ import annotations

import pandas as pd

from vibe_quant.dsl import aux_data
from vibe_quant.dsl.compute_builtins import int_param
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry
from vibe_quant.dsl.warmup_class import CONTEXT_REASON, refuse

_SYMBOLS: dict[str, str] = {}


def _symbol(instrument_id: str) -> str:
    symbol = _SYMBOLS.get(instrument_id)
    if symbol is None:
        from vibe_quant.validation.funding import FundingCalculator

        symbol = _SYMBOLS[instrument_id] = FundingCalculator.symbol_from_instrument_id(
            instrument_id
        )
    return symbol


def compute_oi_change(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """Open-interest change over the last ``period`` 5-minute rows, as of each bar close."""
    symbol, close_ns = aux_data.context_of(df)
    period = int_param(params, "period", 12)
    return pd.Series(aux_data.oi_change_asof(symbol, close_ns, period), index=df.index)


def oi_change_scalar(instrument_id: str, close_ns: int, params: dict[str, object]) -> float:
    """``compute_oi_change`` for ONE bar close (bit-equal to its last row)."""
    return aux_data.oi_change_at(_symbol(instrument_id), close_ns, int_param(params, "period", 12))


def compute_ls_ratio_z(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """Top-trader long/short ratio z-score over ``period`` 5-minute rows, as of each close."""
    symbol, close_ns = aux_data.context_of(df)
    period = int_param(params, "period", 288)
    return pd.Series(aux_data.ls_ratio_z_asof(symbol, close_ns, period), index=df.index)


def ls_ratio_z_scalar(instrument_id: str, close_ns: int, params: dict[str, object]) -> float:
    """``compute_ls_ratio_z`` for ONE bar close (bit-equal to its last row)."""
    return aux_data.ls_ratio_z_at(
        _symbol(instrument_id), close_ns, int_param(params, "period", 288)
    )


# The compiler reads these for ``needs_context`` + ``asof_context`` specs (see compiler.py).
compute_oi_change.asof_scalar = oi_change_scalar  # type: ignore[attr-defined]
compute_ls_ratio_z.asof_scalar = ls_ratio_z_scalar  # type: ignore[attr-defined]

indicator_registry.register_spec(
    IndicatorSpec(
        name="OI_CHANGE",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"period": 12},
        param_schema={"period": int},
        compute_fn=compute_oi_change,
        pta_lookback_fn=lambda p: 1,
        needs_context=True,
        asof_context=True,
        display_name="Open Interest Change",
        description="Open-interest change (contracts) over the last N 5-minute rows, as of the bar close.",
        paper_warmup=refuse(CONTEXT_REASON),
        category="Custom",
        param_ranges={"period": (3.0, 288.0)},
        threshold_range=(-0.05, 0.05),
    )
)

indicator_registry.register_spec(
    IndicatorSpec(
        name="LS_RATIO_Z",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"period": 288},
        param_schema={"period": int},
        compute_fn=compute_ls_ratio_z,
        pta_lookback_fn=lambda p: 1,
        needs_context=True,
        asof_context=True,
        display_name="Top-Trader L/S Ratio Z-Score",
        description="Z-score of the top-trader long/short position ratio over the last N 5-minute rows.",
        paper_warmup=refuse(CONTEXT_REASON),
        category="Custom",
        param_ranges={"period": (12.0, 288.0)},
        threshold_range=(-3.0, 3.0),
    )
)
