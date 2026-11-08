"""Independent numpy transcription of ``05_pmax_explorer.pine`` signals (R62).

The bar set is a deterministic synthetic 1h walk (fixed seed, 480 bars = 20 days, ~0.9 %
hourly volatility). The seed was chosen so both entries fire at least three times.

Pine rules transcribed (script defaults: hl2, ATR 10 with changeATR, EMA 10, multiplier 3.0)::

    atr    = atr(Periods)
    MAvg   = ema(src, length)                       (the "EMA" default of the mav switch)
    longStop  = MAvg - Multiplier*atr;  longStopPrev = nz(longStop[1], longStop)
    longStop := MAvg > longStopPrev ? max(longStop, longStopPrev) : longStop
    shortStop = MAvg + Multiplier*atr; shortStopPrev = nz(shortStop[1], shortStop)
    shortStop := MAvg < shortStopPrev ? min(shortStop, shortStopPrev) : shortStop
    dir := nz(dir[1], 1); dir := dir == -1 and MAvg > shortStopPrev ? 1
                                : dir == 1 and MAvg < longStopPrev ? -1 : dir
    PMax   = dir == 1 ? longStop : shortStop
    buySignalk  = crossover(MAvg, PMax)
    sellSignalk = crossunder(MAvg, PMax)

The plugin's exposed ``mavg`` is NaN until the MA's warm-up and its ``value`` until both the
ATR and MA exist, so ``WARMUP`` starts after the ATR's first value (index 9). The EMA seeds
with its first source value (the project's Pine reference kernel), which matters only inside
the masked warm-up.
"""

from __future__ import annotations

import math

import numpy as np

BARS = 480
SEED = 5

PERIOD = 10
MA_PERIOD = 10
MULTIPLIER = 3.0

#: First compared bar: the ATR/MA warm-up ends at index 9.
WARMUP = 10


def _sma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(x.size, np.nan)
    for t in range(n - 1, x.size):
        out[t] = x[t - n + 1 : t + 1].sum() / n
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


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    alpha = 2.0 / (n + 1)
    out = np.full(x.size, np.nan)
    prev = math.nan
    for t in range(x.size):
        prev = x[t] if prev != prev else alpha * x[t] + (1 - alpha) * prev
        out[t] = prev
    return out


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.concatenate([[math.nan], close[:-1]])
    return np.maximum(
        np.maximum(high - low, np.abs(high - prev_close)), np.abs(low - prev_close)
    )


def _pmax(mavg: np.ndarray, atr: np.ndarray, multiplier: float) -> np.ndarray:
    """Pine ``Pmax_Func`` (``PMax``), masked where the exposed MAvg or the ATR is NaN."""
    size = mavg.size
    pmax = np.full(size, math.nan)
    prev_long = math.nan
    prev_short = math.nan
    current = 1
    for t in range(size):
        long_stop = mavg[t] - multiplier * atr[t]
        long_prev = long_stop if prev_long != prev_long else prev_long
        if mavg[t] > long_prev:
            long_stop = max(long_stop, long_prev)
        short_stop = mavg[t] + multiplier * atr[t]
        short_prev = short_stop if prev_short != prev_short else prev_short
        if mavg[t] < short_prev:
            short_stop = min(short_stop, short_prev)
        if current == -1 and mavg[t] > short_prev:
            current = 1
        elif current == 1 and mavg[t] < long_prev:
            current = -1
        prev_long, prev_short = long_stop, short_stop
        if np.isfinite(long_stop):
            pmax[t] = long_stop if current == 1 else short_stop
    return pmax


def bars() -> dict[str, np.ndarray]:
    """Deterministic 1h OHLCV bars (open, high, low, close, volume)."""
    rng = np.random.default_rng(SEED)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.009, BARS)))
    open_ = np.concatenate([[100.0], close[:-1]])
    spread = np.abs(rng.normal(0.0, 0.004, BARS))
    return {
        "open": open_,
        "high": np.maximum(open_, close) * (1.0 + spread),
        "low": np.minimum(open_, close) * (1.0 - spread),
        "close": close,
        "volume": np.full(BARS, 100.0),
    }


def _cross_up(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    prev_left = np.concatenate([[math.nan], left[:-1]])
    prev_right = np.concatenate([[math.nan], right[:-1]])
    return (left > right) & (prev_left <= prev_right)


def _cross_dn(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    prev_left = np.concatenate([[math.nan], left[:-1]])
    prev_right = np.concatenate([[math.nan], right[:-1]])
    return (left < right) & (prev_left >= prev_right)


def events(bar_data: dict[str, np.ndarray]) -> list[tuple[int, str, str]]:
    """``(bar, side, action)`` of every Pine signal (script inputs at their defaults)."""
    high = bar_data["high"]
    low = bar_data["low"]
    close = bar_data["close"]
    src = (high + low) / 2.0  # input(hl2)
    atr = _rma(_true_range(high, low, close), PERIOD)
    mavg = _ema(src, MA_PERIOD)
    pmax = _pmax(mavg, atr, MULTIPLIER)
    hidden = ~np.isfinite(mavg) | ~np.isfinite(atr)
    pmax[hidden] = math.nan
    mavg = mavg.copy()
    mavg[hidden] = math.nan

    long_entry = _cross_up(mavg, pmax)
    short_entry = _cross_dn(mavg, pmax)
    out: list[tuple[int, str, str]] = []
    for t in range(close.size):
        if long_entry[t]:
            out.append((t, "BUY", "entry"))
        if short_entry[t]:
            out.append((t, "SELL", "entry"))
    return out
