"""PMAX — KivancOzbilgic PMax Explorer (05_pmax_explorer.pine, Pine v4).

Transcribes ``Pmax_Func`` (l.81-92) with ``MAvg = getMA(src, length)`` (l.45-80)::

    longStop  = MAvg - Multiplier*atr;  longStopPrev = nz(longStop[1], longStop)
    longStop := MAvg > longStopPrev ? max(longStop, longStopPrev) : longStop   (l.84)
    shortStop = MAvg + Multiplier*atr;  shortStopPrev = nz(shortStop[1], shortStop)
    shortStop := MAvg < shortStopPrev ? min(shortStop, shortStopPrev) : shortStop  (l.87)
    dir := nz(dir[1], 1)
    dir := dir == -1 and MAvg > shortStopPrev ? 1
         : dir == 1 and MAvg < longStopPrev ? -1 : dir                          (l.90)
    PMax = dir == 1 ? longStop : shortStop                                       (l.91)

(The ``Pmax`` function at l.166-175 only feeds the multi-symbol screener label.)
``atr`` is ``atr(Periods)`` or ``sma(tr, Periods)`` (l.17-18, ``atr_method`` ``rma`` /
``sma``; ``rma_tr`` as in ``_tv_ma.tv_atr``). ``kind`` selects the script's moving
average (``sma ema wma tma var wwma zlema tsf``, the kernels of ``_tv_ma.tv_ma``) over
``ma_period`` bars of ``src`` (script default ``hl2``). The script's signals are
``crossover(MAvg, PMax)`` (``mavg crosses_above value``) and ``crossover(src, PMax)``
(a PRICE indicator with the same ``src``).

Outputs: ``value`` = PMax, ``mavg`` = MAvg, ``dir`` = +1 / -1. NaN until both the ATR and
the moving average are ready. The stops ratchet on the RAW script series (``tv_ma_raw``: EMA,
WWMA, VAR and ZLEMA exist from their first Pine-defined bar and the ATR from its seed), so the
history Pine builds before the public warm-up is kept; the readiness mask hides the exposed
outputs only. ``offset`` lags every output like Pine ``x[offset]``.

Recursive (the stops ratchet, VAR/WWMA/ZLEMA/EMA recurse), so ``requires_full_history``.

Usage::

    indicators:
      pm:
        type: PMAX
        period: 10
        ma_period: 10
        kind: ema
        multiplier: 3.0
    entry_conditions:
      long:
        - pm.mavg crosses_above pm
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
    tv_ma,
    tv_ma_raw,
    tv_ma_ready,
)

if TYPE_CHECKING:
    import pandas as pd

KINDS: tuple[str, ...] = ("sma", "ema", "wma", "tma", "var", "wwma", "zlema", "tsf")
NAN = math.nan


def pmax_arrays(
    mavg: np.ndarray, atr: np.ndarray, multiplier: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(PMax, dir)`` of the Pine ratchet; NaN wherever ``mavg`` or ``atr`` is NaN."""
    size = mavg.size
    m = mavg.tolist()
    a = atr.tolist()
    pmax = [NAN] * size
    direction = [NAN] * size
    prev_long = NAN
    prev_short = NAN
    dir_ = 1
    for t in range(size):
        long_stop = m[t] - multiplier * a[t]
        long_prev = long_stop if prev_long != prev_long else prev_long
        if m[t] > long_prev:  # NaN compares false, like Pine na
            long_stop = max(long_stop, long_prev)
        short_stop = m[t] + multiplier * a[t]
        short_prev = short_stop if prev_short != prev_short else prev_short
        if m[t] < short_prev:
            short_stop = min(short_stop, short_prev)
        if dir_ == -1 and m[t] > short_prev:
            dir_ = 1
        elif dir_ == 1 and m[t] < long_prev:
            dir_ = -1
        prev_long = long_stop
        prev_short = short_stop
        if long_stop == long_stop:
            direction[t] = float(dir_)
            pmax[t] = long_stop if dir_ == 1 else short_stop
    return np.array(pmax, dtype=np.float64), np.array(direction, dtype=np.float64)


def compute_pmax(df: pd.DataFrame, params: dict[str, object]) -> dict[str, pd.Series]:
    """``value`` (PMax), ``mavg`` and ``dir``, lagged by ``offset`` bars."""
    import pandas as pd

    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    atr = tv_atr(
        high, low, close, int_param(params, "period", 10), str_param(params, "atr_method", "rma")
    )
    kind = str_param(params, "kind", "ema")
    src = source_array(df, str_param(params, "src", "hl2"))
    ma_period = int_param(params, "ma_period", 10)
    pmax, direction = pmax_arrays(
        tv_ma_raw(kind, src, ma_period), atr, float_param(params, "multiplier", 3.0)
    )
    mavg = tv_ma(kind, src, ma_period)  # exposed: NaN until ready
    hidden = np.isnan(mavg) | np.isnan(atr)  # readiness mask on the outputs, not on the recursion
    pmax[hidden] = np.nan
    direction[hidden] = np.nan
    offset = int_param(params, "offset", 0)
    return {
        "value": pd.Series(shift_back(pmax, offset), index=df.index),
        "mavg": pd.Series(shift_back(mavg, offset), index=df.index),
        "dir": pd.Series(shift_back(direction, offset), index=df.index),
    }


def _lookback(params: dict[str, object]) -> int:
    src = str_param(params, "src", "hl2")
    if src not in SOURCES:
        msg = f"PMAX src {src!r} unknown; valid: {', '.join(SOURCES)}"
        raise ValueError(msg)
    kind = str_param(params, "kind", "ema")
    if kind not in KINDS:
        msg = f"PMAX kind {kind!r} unknown; valid: {', '.join(KINDS)}"
        raise ValueError(msg)
    offset = int_param(params, "offset", 0)
    if offset < 0:
        msg = f"PMAX offset must be >= 0, got {offset}"
        raise ValueError(msg)
    return (
        max(
            tv_atr_ready(str_param(params, "atr_method", "rma"), int_param(params, "period", 10)),
            tv_ma_ready(kind, int_param(params, "ma_period", 10)),
        )
        + offset
    )


indicator_registry.register_spec(
    IndicatorSpec(
        name="PMAX",
        nt_class=None,
        pandas_ta_func=None,
        default_params={
            "period": 10,
            "ma_period": 10,
            "multiplier": 3.0,
            "kind": "ema",
            "src": "hl2",
            "atr_method": "rma",
            "offset": 0,
        },
        param_schema={
            "period": int,
            "ma_period": int,
            "multiplier": float,
            "kind": str,
            "src": str,
            "atr_method": str,
            "offset": int,
        },
        compute_fn=compute_pmax,
        pta_lookback_fn=_lookback,
        output_names=("value", "mavg", "dir"),
        primary_output="value",
        requires_high_low=True,
        requires_full_history=True,
        display_name="PMax",
        description=(
            "PMax (KivancOzbilgic): ratcheting ATR stops around a moving average "
            "(sma, ema, wma, tma, var, wwma, zlema, tsf); dir is +1 / -1."
        ),
        category="Trend",
        chart_placement="overlay",
    )
)
