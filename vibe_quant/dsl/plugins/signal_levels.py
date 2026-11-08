"""SIGNAL_LEVELS — stop and target price levels computed on the signal bar.

Both scripts size their exits from levels read on the bar where the entry signal fires:

* 06_3commas_bot.pine l.166 (atr), 170-171 (extrema), 226-233 (stop/risk/target) (v5)::

      lowestLow  = ta.lowest(low, swinglookback);  highestHigh = ta.highest(high, swinglookback)
      longStop   = lowestLow  - atr*RiskM;          shortStop   = highestHigh + atr*RiskM
      longRisk   = close - longStop;                shortRisk   = shortStop - close
      longlimit  = close + RnR*longRisk;            shortlimit  = close - RnR*shortRisk

  (``atr = ta.atr(atrLen)``; defaults RnR 1, RiskM 1, swinglookback 5, atrLen 14).
* 08_ao_stoch_rsi_atr.pine l.49, 55-64 (v4): ``stoploss = low - atrvalue``,
  ``takeprofit = close + atrvalue`` (short: ``high + atrvalue``, ``close - atrvalue``) with
  ``atrvalue = rma(tr, atrlen)`` = ``lookback 1``, ``risk_mult 1``, ``target_mode atr``,
  ``target_atr 1``, ``atr_method rma_tr``.

Params: ``lookback`` (swing window), ``atr_period``, ``atr_method`` (``rma`` = ``ta.atr``,
``rma_tr`` = ``rma(tr, n)``, ``sma``), ``risk_mult``, ``target_mode`` (``rr``: reward = ``rr`` x
risk; ``atr``: reward = ``target_atr`` x ATR), ``rr``, ``target_atr``, ``offset`` (lag).

Outputs (price levels, NaN until ready): ``long_stop``, ``long_target``, ``short_stop``,
``short_target``. The ``signal_level`` stop/target types of the engine read them as
``name.output`` on the signal bar; this plugin does not decide validity (a level on the wrong
side of the close is the engine's ``level_invalid``).

The ATR is a Wilder recursion, so the spec sets ``requires_full_history`` even though the
swing window itself is finite (``sma`` ATR would be finite; the flag is per spec).

Usage::

    indicators:
      lv:
        type: SIGNAL_LEVELS
        lookback: 5
        atr_period: 14
        rr: 1.5
    stop_loss_long: {type: signal_level, indicator: lv.long_stop}
    take_profit_long: {type: signal_level, indicator: lv.long_target}
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vibe_quant.dsl.compute_builtins import float_param, int_param
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry
from vibe_quant.dsl.plugins._tv_ma import (
    rolling_max,
    rolling_min,
    shift_back,
    str_param,
    tv_atr,
    tv_atr_ready,
)

if TYPE_CHECKING:
    import pandas as pd

TARGET_MODES: tuple[str, ...] = ("rr", "atr")


def _target_mode(params: dict[str, object]) -> str:
    mode = str_param(params, "target_mode", "rr")
    if mode not in TARGET_MODES:
        msg = f"SIGNAL_LEVELS target_mode {mode!r} unknown; valid: {', '.join(TARGET_MODES)}"
        raise ValueError(msg)
    return mode


def compute_signal_levels(df: pd.DataFrame, params: dict[str, object]) -> dict[str, pd.Series]:
    """``long_stop``, ``long_target``, ``short_stop``, ``short_target`` lagged by ``offset``."""
    import pandas as pd

    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    atr = tv_atr(
        high,
        low,
        close,
        int_param(params, "atr_period", 14),
        str_param(params, "atr_method", "rma"),
    )
    lookback = int_param(params, "lookback", 5)
    risk_mult = float_param(params, "risk_mult", 1.0)
    long_stop = rolling_min(low, lookback) - atr * risk_mult
    short_stop = rolling_max(high, lookback) + atr * risk_mult
    if _target_mode(params) == "rr":
        rr = float_param(params, "rr", 1.0)
        long_target = close + rr * (close - long_stop)
        short_target = close - rr * (short_stop - close)
    else:
        target_atr = float_param(params, "target_atr", 1.0)
        long_target = close + target_atr * atr
        short_target = close - target_atr * atr
    offset = int_param(params, "offset", 0)
    return {
        "long_stop": pd.Series(shift_back(long_stop, offset), index=df.index),
        "long_target": pd.Series(shift_back(long_target, offset), index=df.index),
        "short_stop": pd.Series(shift_back(short_stop, offset), index=df.index),
        "short_target": pd.Series(shift_back(short_target, offset), index=df.index),
    }


def _lookback(params: dict[str, object]) -> int:
    lookback = int_param(params, "lookback", 5)
    if lookback < 1:
        msg = f"SIGNAL_LEVELS lookback must be >= 1, got {lookback}"
        raise ValueError(msg)
    _target_mode(params)
    offset = int_param(params, "offset", 0)
    if offset < 0:
        msg = f"SIGNAL_LEVELS offset must be >= 0, got {offset}"
        raise ValueError(msg)
    atr_ready = tv_atr_ready(
        str_param(params, "atr_method", "rma"), int_param(params, "atr_period", 14)
    )
    return max(lookback, atr_ready) + offset


indicator_registry.register_spec(
    IndicatorSpec(
        name="SIGNAL_LEVELS",
        nt_class=None,
        pandas_ta_func=None,
        default_params={
            "lookback": 5,
            "atr_period": 14,
            "atr_method": "rma",
            "risk_mult": 1.0,
            "target_mode": "rr",
            "rr": 1.0,
            "target_atr": 1.0,
            "offset": 0,
        },
        param_schema={
            "lookback": int,
            "atr_period": int,
            "atr_method": str,
            "risk_mult": float,
            "target_mode": str,
            "rr": float,
            "target_atr": float,
            "offset": int,
        },
        compute_fn=compute_signal_levels,
        pta_lookback_fn=_lookback,
        output_names=("long_stop", "long_target", "short_stop", "short_target"),
        primary_output="long_stop",
        requires_high_low=True,
        requires_full_history=True,
        display_name="Signal-bar stop/target levels",
        description=(
            "Swing-low/high minus/plus ATR stop and a reward-to-risk or ATR target, as price "
            "levels for signal_level stops."
        ),
        category="Volatility",
        chart_placement="overlay",
    )
)
