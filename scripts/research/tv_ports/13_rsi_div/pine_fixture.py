"""Independent numpy transcription of ``13_rsi_divergence.pine`` signals (R62).

The bar set is a deterministic synthetic 1h walk (fixed seed, 960 bars = 40 days, ~0.8 %
hourly volatility). The seed was chosen so the long entry fires at least three times, so the
comparison exercises the pivot/divergence rules rather than a run of ``False``.

Pine rules transcribed (script defaults: RSI 9, pivot lookback left 1 / right 3, take-profit
RSI level 80, range 5..60, plotBull / plotHiddenBull / plotBear true, plotHiddenBear false)::

    osc    = rsi(src, 9)
    plFound = not na(pivotlow(osc, 1, 3));  phFound = not na(pivothigh(osc, 1, 3))
    _inRange(cond) => rangeLower <= barssince(cond) <= rangeUpper
    oscHL   = osc[3] > valuewhen(plFound, osc[3], 1) and _inRange(plFound[1])
    priceLL = low[3] < valuewhen(plFound, low[3], 1)
    bullCond = plotBull and priceLL and oscHL and plFound
    hiddenBullCond = plotHiddenBull and priceHL and oscLL and plFound
    bearCond = plotBear and priceHH and oscLH and phFound
    longCondition      = bullCond or hiddenBullCond
    longCloseCondition = crossover(osc, 80) or bearCond

``pivotlow``/``pivothigh`` are confirmed ``lbR`` bars late and use the project's tie rule (the
centre is strictly beyond every left bar, and within-or-beyond on the right), transcribed here
rather than imported. ``WARMUP`` excludes NT's Wilder-RSI seed difference: the entry pulses come
from raw pivot comparisons, and a fraction of a point of RSI moves a pivot; the engine's NT RSI
and Pine's SMA-seeded ``rsi()`` agree to about 1e-9 only after ~310 bars of this walk.
"""

from __future__ import annotations

import math

import numpy as np

BARS = 960
SEED = 13

RSI_LENGTH = 9
LB_LEFT = 1
LB_RIGHT = 3
TAKE_PROFIT_LEVEL = 80.0
RANGE_LOWER = 5
RANGE_UPPER = 60
PLOT_BULL = True
PLOT_HIDDEN_BULL = True
PLOT_BEAR = True

#: First compared bar: past the engine gate and NT's Wilder-RSI seed difference.
WARMUP = 320


def _sma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(x.size, np.nan)
    for t in range(n - 1, x.size):
        window = x[t - n + 1 : t + 1]
        if np.isfinite(window).all():
            out[t] = window.sum() / n
    return out


def _rma(x: np.ndarray, n: int) -> np.ndarray:
    seed = _sma(x, n)
    out = np.full(x.size, np.nan)
    prev = math.nan
    alpha = 1.0 / n
    for t in range(x.size):
        prev = seed[t] if prev != prev else alpha * x[t] + (1 - alpha) * prev
        out[t] = prev
    return out


def _rsi(x: np.ndarray, n: int) -> np.ndarray:
    delta = np.concatenate([[math.nan], np.diff(x)])
    gain = np.where(delta > 0.0, delta, 0.0)
    loss = np.where(delta < 0.0, -delta, 0.0)
    gain[0] = math.nan
    loss[0] = math.nan
    up = _rma(gain, n)
    down = _rma(loss, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 - 100.0 / (1.0 + up / down)
    out = np.where((down == 0.0) & (up > 0.0), 100.0, out)
    out = np.where((down == 0.0) & (up == 0.0), math.nan, out)
    return out


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


def _pulses(
    osc: np.ndarray, low: np.ndarray, high: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(long, long_exit, bull, bear)`` pulses of the script (1.0 / 0.0 per bar)."""
    size = osc.size
    o = osc.tolist()
    lo = low.tolist()
    hi = high.tolist()
    bull = np.zeros(size)
    hidden_bull = np.zeros(size)
    bear = np.zeros(size)
    last_pl = -1  # previous pivot-low-FOUND bar (-1: none)
    last_ph = -1
    for t in range(size):
        c = t - LB_RIGHT
        if c - LB_LEFT < 0:
            continue
        if _is_pivot(o, c, LB_LEFT, LB_RIGHT, low=True):
            if last_pl >= 0:
                bars = t - 1 - last_pl  # barssince(plFound[1])
                in_range = RANGE_LOWER <= bars <= RANGE_UPPER
                prev_osc = o[last_pl - LB_RIGHT]
                prev_low = lo[last_pl - LB_RIGHT]
                if PLOT_BULL and lo[c] < prev_low and o[c] > prev_osc and in_range:
                    bull[t] = 1.0
                if PLOT_HIDDEN_BULL and lo[c] > prev_low and o[c] < prev_osc and in_range:
                    hidden_bull[t] = 1.0
            last_pl = t
        if _is_pivot(o, c, LB_LEFT, LB_RIGHT, low=False):
            if last_ph >= 0:
                bars = t - 1 - last_ph
                in_range = RANGE_LOWER <= bars <= RANGE_UPPER
                prev_osc = o[last_ph - LB_RIGHT]
                prev_high = hi[last_ph - LB_RIGHT]
                if PLOT_BEAR and hi[c] > prev_high and o[c] < prev_osc and in_range:
                    bear[t] = 1.0
            last_ph = t
    long_ = np.where((bull > 0.0) | (hidden_bull > 0.0), 1.0, 0.0)
    prev = np.concatenate([[math.nan], osc[:-1]])
    crossed = (osc > TAKE_PROFIT_LEVEL) & (prev <= TAKE_PROFIT_LEVEL)
    long_exit = np.where(crossed | (bear > 0.0), 1.0, 0.0)
    return long_, long_exit, bull, bear


def bars() -> dict[str, np.ndarray]:
    """Deterministic 1h OHLCV bars (open, high, low, close, volume)."""
    rng = np.random.default_rng(SEED)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.008, BARS)))
    open_ = np.concatenate([[100.0], close[:-1]])
    spread = np.abs(rng.normal(0.0, 0.003, BARS))
    return {
        "open": open_,
        "high": np.maximum(open_, close) * (1.0 + spread),
        "low": np.minimum(open_, close) * (1.0 - spread),
        "close": close,
        "volume": np.full(BARS, 100.0),
    }


def events(bar_data: dict[str, np.ndarray]) -> list[tuple[int, str, str]]:
    """``(bar, side, action)`` of every Pine signal (script inputs at their defaults)."""
    close = bar_data["close"]
    osc = _rsi(close, RSI_LENGTH)
    long_, long_exit, _bull, _bear = _pulses(osc, bar_data["low"], bar_data["high"])

    out: list[tuple[int, str, str]] = []
    for t in range(close.size):
        if long_[t] == 1.0:
            out.append((t, "BUY", "entry"))
        if long_exit[t] == 1.0:
            out.append((t, "SELL", "exit"))
    return out
