"""SUPERTREND — KivancOzbilgic SuperTrend (03_supertrend_strategy.pine, Pine v4).

Transcribes lines 14-24 operation by operation::

    atr   = changeATR ? atr(Periods) : sma(tr, Periods)          (l.14-15)
    up    = src - Multiplier*atr;  up1 = nz(up[1], up)           (l.16-17)
    up   := close[1] > up1 ? max(up, up1) : up                   (l.18)
    dn    = src + Multiplier*atr;  dn1 = nz(dn[1], dn)           (l.19-20)
    dn   := close[1] < dn1 ? min(dn, dn1) : dn                   (l.21)
    trend := nz(trend[1], 1)                                     (l.22-23)
    trend := trend == -1 and close > dn1 ? 1
           : trend == 1 and close < up1 ? -1 : trend             (l.24)

``up[1]``/``dn[1]`` are the previous bar's FINAL (ratcheted) values. Outputs:
``value`` = the active line (``trend == 1 ? up : dn``, l.25/29), ``dir`` = ``trend``
(+1 / -1; the script's ``buySignal`` is ``dir`` turning +1, ``sellSignal`` turning -1 =
``dir crosses_above 0`` / ``crosses_below 0``), ``up`` and ``dn`` the final bands.

``src``: ``open|high|low|close|hl2|hlc3|ohlc4`` (script default ``hl2``).
``atr_method``: ``rma`` (``changeATR`` true, the default), ``sma`` (``changeATR`` false) or
``rma_tr`` (see ``_tv_ma.tv_atr``). ``offset`` lags every output like Pine ``x[offset]``.
All outputs are NaN until the ATR exists (the trend itself is 1 from bar 0 in Pine, but
nothing is plotted there).

The ratchet is a recursion over every earlier bar, so the spec sets
``requires_full_history`` (unbounded buffer; a trimmed buffer would change the bands).

Usage::

    indicators:
      st:
        type: SUPERTREND
        period: 10
        multiplier: 3.0
    entry_conditions:
      long:
        - st.dir crosses_above 0
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from vibe_quant.dsl.compute_builtins import float_param, int_param
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry
from vibe_quant.dsl.plugins._tv_ma import (
    SOURCES,
    shift_back,
    source_array,
    str_param,
    tv_atr,
    tv_atr_ready,
)

if TYPE_CHECKING:
    import pandas as pd

NAN = math.nan


def supertrend_arrays(
    src: np.ndarray, close: np.ndarray, atr: np.ndarray, multiplier: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(value, dir, up, dn)`` of the Pine ratchet; NaN wherever ``atr`` is NaN."""
    size = src.size
    s = src.tolist()
    c = close.tolist()
    a = atr.tolist()
    up_out = [NAN] * size
    dn_out = [NAN] * size
    dir_out = [NAN] * size
    val_out = [NAN] * size
    prev_up = NAN
    prev_dn = NAN
    trend = 1
    for t in range(size):
        up = s[t] - multiplier * a[t]
        up1 = up if prev_up != prev_up else prev_up  # nz(up[1], up)
        if t > 0 and c[t - 1] > up1:  # NaN compares false, like Pine na
            up = max(up, up1)
        dn = s[t] + multiplier * a[t]
        dn1 = dn if prev_dn != prev_dn else prev_dn
        if t > 0 and c[t - 1] < dn1:
            dn = min(dn, dn1)
        if trend == -1 and c[t] > dn1:
            trend = 1
        elif trend == 1 and c[t] < up1:
            trend = -1
        prev_up = up
        prev_dn = dn
        if a[t] == a[t]:
            up_out[t] = up
            dn_out[t] = dn
            dir_out[t] = float(trend)
            val_out[t] = up if trend == 1 else dn
    return (
        np.array(val_out, dtype=np.float64),
        np.array(dir_out, dtype=np.float64),
        np.array(up_out, dtype=np.float64),
        np.array(dn_out, dtype=np.float64),
    )


def compute_supertrend(df: pd.DataFrame, params: dict[str, object]) -> dict[str, pd.Series]:
    """``value``, ``dir``, ``up``, ``dn`` of the SuperTrend, lagged by ``offset``."""
    import pandas as pd

    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    atr = tv_atr(
        high,
        low,
        close,
        int_param(params, "period", 10),
        str_param(params, "atr_method", "rma"),
    )
    value, direction, up, dn = supertrend_arrays(
        source_array(df, str_param(params, "src", "hl2")),
        close,
        atr,
        float_param(params, "multiplier", 3.0),
    )
    offset = int_param(params, "offset", 0)
    return {
        "value": pd.Series(shift_back(value, offset), index=df.index),
        "dir": pd.Series(shift_back(direction, offset), index=df.index),
        "up": pd.Series(shift_back(up, offset), index=df.index),
        "dn": pd.Series(shift_back(dn, offset), index=df.index),
    }


def _lookback(params: dict[str, object]) -> int:
    src = str_param(params, "src", "hl2")
    if src not in SOURCES:
        msg = f"SUPERTREND src {src!r} unknown; valid: {', '.join(SOURCES)}"
        raise ValueError(msg)
    offset = int_param(params, "offset", 0)
    if offset < 0:
        msg = f"SUPERTREND offset must be >= 0, got {offset}"
        raise ValueError(msg)
    return (
        tv_atr_ready(str_param(params, "atr_method", "rma"), int_param(params, "period", 10))
        + offset
    )


indicator_registry.register_spec(
    IndicatorSpec(
        name="SUPERTREND",
        nt_class=None,
        pandas_ta_func=None,
        default_params={
            "period": 10,
            "multiplier": 3.0,
            "src": "hl2",
            "atr_method": "rma",
            "offset": 0,
        },
        param_schema={
            "period": int,
            "multiplier": float,
            "src": str,
            "atr_method": str,
            "offset": int,
        },
        compute_fn=compute_supertrend,
        pta_lookback_fn=_lookback,
        output_names=("value", "dir", "up", "dn"),
        primary_output="value",
        requires_high_low=True,
        requires_full_history=True,
        display_name="SuperTrend",
        description=(
            "SuperTrend (KivancOzbilgic): ATR bands ratcheted against the previous close; "
            "dir is +1 in an up trend, -1 in a down trend."
        ),
        category="Trend",
        chart_placement="overlay",
    )
)
