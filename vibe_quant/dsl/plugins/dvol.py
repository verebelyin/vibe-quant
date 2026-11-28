"""DVOL_Z -- z-score of the Deribit DVOL implied-volatility index as a context indicator.

Reads archived hourly DVOL candles through ``vibe_quant.dsl.aux_data``
(``needs_context=True``): ETH DVOL for ETHUSDT, BTC DVOL for every other symbol. A
candle opened at ``o`` is usable from bar close ``o + 1 h`` (when it is complete) and
is dropped once the close is 2 h past that, so one missing candle is bridged and two
are not. ``period`` counts hourly candles, whatever the bar timeframe; a missing
candle inside the window is NaN (strict).

Screening/validation only; paper/live refuse strategies that use it.

Usage::

    indicators:
      ivz:
        type: DVOL_Z
        period: 168
    entry_conditions:
      long:
        - ivz < -2
"""

from __future__ import annotations

import pandas as pd

from vibe_quant.dsl import aux_data
from vibe_quant.dsl.compute_builtins import int_param
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry
from vibe_quant.dsl.plugins.positioning import _symbol
from vibe_quant.dsl.warmup_class import CONTEXT_REASON, refuse


def compute_dvol_z(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """DVOL close z-score over ``period`` hourly candles, as of each bar close."""
    symbol, close_ns = aux_data.context_of(df)
    period = int_param(params, "period", 168)
    return pd.Series(aux_data.dvol_z_asof(symbol, close_ns, period), index=df.index)


def dvol_z_scalar(instrument_id: str, close_ns: int, params: dict[str, object]) -> float:
    """``compute_dvol_z`` for ONE bar close (bit-equal to its last row)."""
    return aux_data.dvol_z_at(_symbol(instrument_id), close_ns, int_param(params, "period", 168))


# The compiler reads this for ``needs_context`` + ``asof_context`` specs (see compiler.py).
compute_dvol_z.asof_scalar = dvol_z_scalar  # type: ignore[attr-defined]

indicator_registry.register_spec(
    IndicatorSpec(
        name="DVOL_Z",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"period": 168},
        param_schema={"period": int},
        compute_fn=compute_dvol_z,
        pta_lookback_fn=lambda p: 1,
        needs_context=True,
        asof_context=True,
        display_name="DVOL Z-Score",
        description="Z-score of the Deribit DVOL implied-volatility index (BTC; ETH for ETHUSDT) over the last N hourly candles.",
        paper_warmup=refuse(CONTEXT_REASON),
        category="Custom",
        param_ranges={"period": (24.0, 720.0)},
        threshold_range=(-3.0, 3.0),
    )
)
