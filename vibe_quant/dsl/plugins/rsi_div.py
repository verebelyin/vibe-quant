"""RSI_DIV — RSI divergence entry/exit pulses (13_rsi_divergence.pine, Pine v4).

``osc = rsi(src, len)`` (l.38) is Pine's RSI: ``up = rma(max(change, 0), len)``,
``down = rma(-min(change, 0), len)``, ``down == 0 ? 100 : up == 0 ? 0 : 100 - 100/(1 + up/down)``
(NOT the NautilusTrader RSI, which differs). A pivot is confirmed ``lb_right`` bars late::

    plFound = not na(pivotlow(osc, lbL, lbR))          (l.46; phFound with pivothigh, l.47)
    _inRange(cond) => rangeLower <= barssince(cond) <= rangeUpper         (l.49-51)
    oscHL  = osc[lbR] > valuewhen(plFound, osc[lbR], 1) and _inRange(plFound[1])      (l.57)
    priceLL = low[lbR] < valuewhen(plFound, low[lbR], 1)                              (l.60)
    bullCond = plotBull and priceLL and oscHL and plFound                             (l.62)
    hiddenBullCond = plotHiddenBull and priceHL and oscLL and plFound                 (l.90-95)
    bearCond = plotBear and priceHH and oscLH and phFound                             (l.145-150)
    hiddenBearCond = plotHiddenBear and priceLH and oscHH and phFound                 (l.177-182)
    longCondition = bullCond or hiddenBullCond                                        (l.118)
    longCloseCondition = crossover(osc, takeProfitRSILevel) or bearCond               (l.204)

``valuewhen(plFound, x, 1)`` is the value at the previous pivot-found bar;
``barssince(plFound[1])`` at a found bar is ``t - 1 - u`` for the previous found bar ``u``.

Outputs: ``rsi`` (NaN until ``period + 1`` bars) and the PULSES ``bull``, ``hidden_bull``,
``bear``, ``hidden_bear`` and ``long`` (= ``bull`` or ``hidden_bull``): 1.0 on the bar a
divergence is confirmed and 0.0 on every other bar, NEVER NaN. The runtime keeps the last
non-NaN value of an output, so a pulse that emitted NaN between events would latch at 1.0;
the explicit 0.0 resets it. Entry: ``rsi_div.long > 0``; exit: ``rsi_div.bear > 0`` or
``rsi crosses_above 80``. Flags ``plot_bull`` / ``plot_hidden_bull`` / ``plot_bear`` /
``plot_hidden_bear`` (0 or 1; the script's defaults 1/1/1/0) switch the pulses on.

UNVERIFIED vs TradingView output (no export available): Pine's ``pivotlow``/``pivothigh``
tie rule. Here the centre must be strictly lower (higher) than every ``lb_left`` bar and
lower-or-equal (higher-or-equal) than every ``lb_right`` bar, so the left-most of equal
extremes is the pivot; a window holding a NaN (RSI warm-up) is never a pivot.

The RSI recursion and the pivot history (``valuewhen``/``barssince`` reach back to the first
pivot) make this depend on every earlier bar: ``requires_full_history``. ``offset`` lags every
output (pulses are 0.0, not NaN, for the first ``offset`` rows).

Usage::

    indicators:
      div:
        type: RSI_DIV
        period: 9
        lb_right: 3
        lb_left: 1
    entry_conditions:
      long:
        - div.long > 0
    exit_conditions:
      long:
        - div.bear > 0
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from vibe_quant.dsl.compute_builtins import int_param
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry
from vibe_quant.dsl.plugins._tv_ma import SOURCES, shift_back, source_array, str_param, tv_ma

if TYPE_CHECKING:
    import pandas as pd

NAN = math.nan
_PULSES = ("bull", "hidden_bull", "bear", "hidden_bear", "long")


def tv_rsi(src: np.ndarray, length: int) -> np.ndarray:
    """Pine ``rsi(src, length)``, NaN until ``length + 1`` bars exist."""
    change = src - shift_back(src, 1)
    up = tv_ma("rma", np.maximum(change, 0.0), length)
    down = tv_ma("rma", -np.minimum(change, 0.0), length)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100 - 100 / (1 + up / down)
    rsi = np.where(up == 0, 0.0, rsi)
    rsi = np.where(down == 0, 100.0, rsi)
    return np.asarray(rsi, dtype=np.float64)


def _is_pivot(osc: list[float], c: int, left: int, right: int, *, low: bool) -> bool:
    """Pine ``pivotlow`` (``low``) / ``pivothigh`` centred on bar ``c``."""
    centre = osc[c]
    if centre != centre:
        return False
    for i in range(c - left, c):
        v = osc[i]
        if v != v or (v <= centre if low else v >= centre):
            return False
    for i in range(c + 1, c + right + 1):
        v = osc[i]
        if v != v or (v < centre if low else v > centre):
            return False
    return True


def rsi_div_pulses(
    osc: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    flags: tuple[bool, bool, bool, bool],
    left: int,
    right: int,
    range_lower: int,
    range_upper: int,
) -> dict[str, np.ndarray]:
    """Per-bar divergence pulses (1.0 / 0.0) of the Pine logic above."""
    size = osc.size
    o = osc.tolist()
    lo = low.tolist()
    hi = high.tolist()
    plot_bull, plot_hidden_bull, plot_bear, plot_hidden_bear = flags
    out = {name: [0.0] * size for name in _PULSES}
    last_pl = -1  # previous pivot-low-found bar u (-1: none)
    last_ph = -1
    for t in range(size):
        c = t - right
        if c - left < 0:
            continue
        if _is_pivot(o, c, left, right, low=True):
            if last_pl >= 0:
                bars = t - 1 - last_pl  # barssince(plFound[1])
                in_range = range_lower <= bars <= range_upper
                prev_osc = o[last_pl - right]
                prev_low = lo[last_pl - right]
                if plot_bull and lo[c] < prev_low and o[c] > prev_osc and in_range:
                    out["bull"][t] = 1.0
                if plot_hidden_bull and lo[c] > prev_low and o[c] < prev_osc and in_range:
                    out["hidden_bull"][t] = 1.0
            last_pl = t
        if _is_pivot(o, c, left, right, low=False):
            if last_ph >= 0:
                bars = t - 1 - last_ph
                in_range = range_lower <= bars <= range_upper
                prev_osc = o[last_ph - right]
                prev_high = hi[last_ph - right]
                if plot_bear and hi[c] > prev_high and o[c] < prev_osc and in_range:
                    out["bear"][t] = 1.0
                if plot_hidden_bear and hi[c] < prev_high and o[c] > prev_osc and in_range:
                    out["hidden_bear"][t] = 1.0
            last_ph = t
    longs = [1.0 if a or b else 0.0 for a, b in zip(out["bull"], out["hidden_bull"], strict=True)]
    out["long"] = longs
    return {name: np.array(vals, dtype=np.float64) for name, vals in out.items()}


def _lag_pulse(a: np.ndarray, offset: int) -> np.ndarray:
    """``a[offset]`` for a pulse: the first ``offset`` rows are 0.0 (no event), never NaN."""
    if offset == 0:
        return a
    out = np.zeros(a.shape, dtype=np.float64)
    if offset < a.size:
        out[offset:] = a[:-offset]
    return out


def _flags(params: dict[str, object]) -> tuple[bool, bool, bool, bool]:
    return (
        int_param(params, "plot_bull", 1) != 0,
        int_param(params, "plot_hidden_bull", 1) != 0,
        int_param(params, "plot_bear", 1) != 0,
        int_param(params, "plot_hidden_bear", 0) != 0,
    )


def compute_rsi_div(df: pd.DataFrame, params: dict[str, object]) -> dict[str, pd.Series]:
    """``rsi`` plus the divergence pulses, lagged by ``offset`` bars."""
    import pandas as pd

    osc = tv_rsi(
        source_array(df, str_param(params, "src", "close")), int_param(params, "period", 9)
    )
    pulses = rsi_div_pulses(
        osc,
        df["low"].to_numpy(dtype=np.float64),
        df["high"].to_numpy(dtype=np.float64),
        _flags(params),
        int_param(params, "lb_left", 1),
        int_param(params, "lb_right", 3),
        int_param(params, "range_lower", 5),
        int_param(params, "range_upper", 60),
    )
    offset = int_param(params, "offset", 0)
    result = {"rsi": pd.Series(shift_back(osc, offset), index=df.index)}
    for name in _PULSES:
        result[name] = pd.Series(_lag_pulse(pulses[name], offset), index=df.index)
    return result


def _lookback(params: dict[str, object]) -> int:
    src = str_param(params, "src", "close")
    if src not in SOURCES:
        msg = f"RSI_DIV src {src!r} unknown; valid: {', '.join(SOURCES)}"
        raise ValueError(msg)
    period = int_param(params, "period", 9)
    if period < 1:
        msg = f"RSI_DIV period must be >= 1, got {period}"
        raise ValueError(msg)
    for key in ("lb_left", "lb_right"):
        if int_param(params, key, 1) < 1:
            msg = f"RSI_DIV {key} must be >= 1"
            raise ValueError(msg)
    offset = int_param(params, "offset", 0)
    if offset < 0:
        msg = f"RSI_DIV offset must be >= 0, got {offset}"
        raise ValueError(msg)
    return period + 1 + offset  # first finite rsi is bar index `period`


indicator_registry.register_spec(
    IndicatorSpec(
        name="RSI_DIV",
        nt_class=None,
        pandas_ta_func=None,
        default_params={
            "period": 9,
            "src": "close",
            "lb_left": 1,
            "lb_right": 3,
            "range_lower": 5,
            "range_upper": 60,
            "plot_bull": 1,
            "plot_hidden_bull": 1,
            "plot_bear": 1,
            "plot_hidden_bear": 0,
            "offset": 0,
        },
        param_schema={
            "period": int,
            "src": str,
            "lb_left": int,
            "lb_right": int,
            "range_lower": int,
            "range_upper": int,
            "plot_bull": int,
            "plot_hidden_bull": int,
            "plot_bear": int,
            "plot_hidden_bear": int,
            "offset": int,
        },
        compute_fn=compute_rsi_div,
        pta_lookback_fn=_lookback,
        output_names=("rsi", "bull", "hidden_bull", "bear", "hidden_bear", "long"),
        primary_output="rsi",
        requires_high_low=True,
        requires_full_history=True,
        display_name="RSI divergence",
        description=(
            "TradingView RSI plus divergence pulses (1.0 on the confirming bar, else 0.0): "
            "regular/hidden bullish and bearish."
        ),
        category="Momentum",
        chart_placement="oscillator",
    )
)
