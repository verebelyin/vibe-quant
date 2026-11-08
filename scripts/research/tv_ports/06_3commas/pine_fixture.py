"""Independent numpy transcription of ``06_3commas_bot.pine`` signals (R62).

The bar set is a deterministic synthetic 1h walk (fixed seed, 480 bars = 20 days, ~0.8 %
hourly volatility). The seed was chosen so both MA crosses fire at least three times, so the
comparison exercises the rules.

Pine rules transcribed (script defaults: EMA 21 / EMA 50, ATR 14, swing lookback 5,
RiskM 1.0, RnR 1.0, FLIP false, useLimit true, trailStop false)::

    ma1 = getMA(maType1, maLength1); ma2 = getMA(maType2, maLength2)
    atr = ta.atr(atrLen)
    lowestLow  = ta.lowest(low, swinglookback); highestHigh = ta.highest(high, swinglookback)
    longStop  = lowestLow  - atr*RiskM;  shortStop = highestHigh + atr*RiskM
    longlimit = close + RnR*(close - longStop); shortlimit = close - RnR*(shortStop - close)
    validLongEntry  = ta.crossover(ma1, ma2)  and not na(atr)
    validShortEntry = ta.crossunder(ma1, ma2) and not na(atr)

The entries only fire while FLAT (FLIP defaults to false), so the port has no
``position_mode``. The stops/targets are the levels computed on the signal bar
(``SIGNAL_LEVELS``), the script's ``strategy.exit`` mirror. ``WARMUP`` starts after the
slow EMA's warm-up (the engine's readiness gate); Pine's EMA exists from its first bar.
"""

from __future__ import annotations

import math

import numpy as np

BARS = 480
SEED = 6

MA_LENGTH_1 = 21
MA_LENGTH_2 = 50
ATR_LENGTH = 14

#: First compared bar: the 50-bar EMA's readiness gate ends at index 49.
WARMUP = 50


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


def _cross_up(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    prev_left = np.concatenate([[math.nan], left[:-1]])
    prev_right = np.concatenate([[math.nan], right[:-1]])
    return (left > right) & (prev_left <= prev_right)


def _cross_dn(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    prev_left = np.concatenate([[math.nan], left[:-1]])
    prev_right = np.concatenate([[math.nan], right[:-1]])
    return (left < right) & (prev_left >= prev_right)


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


def levels(bar_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The signal-bar stop/target levels of ``compute_signal_levels`` (rr mode, defaults)."""
    high = bar_data["high"]
    low = bar_data["low"]
    close = bar_data["close"]
    atr = _rma(_true_range(high, low, close), ATR_LENGTH)
    lowest = np.full(close.size, math.nan)
    highest = np.full(close.size, math.nan)
    for t in range(4, close.size):
        lowest[t] = low[t - 4 : t + 1].min()
        highest[t] = high[t - 4 : t + 1].max()
    long_stop = lowest - atr
    short_stop = highest + atr
    return {
        "long_stop": long_stop,
        "long_target": close + (close - long_stop),
        "short_stop": short_stop,
        "short_target": close - (short_stop - close),
    }


def events(bar_data: dict[str, np.ndarray]) -> list[tuple[int, str, str]]:
    """``(bar, side, action)`` of every Pine signal (script inputs at their defaults)."""
    close = bar_data["close"]
    ma1 = _ema(close, MA_LENGTH_1)
    ma2 = _ema(close, MA_LENGTH_2)
    atr = _rma(_true_range(bar_data["high"], bar_data["low"], close), ATR_LENGTH)
    ready = np.isfinite(atr)

    long_entry = _cross_up(ma1, ma2) & ready
    short_entry = _cross_dn(ma1, ma2) & ready
    out: list[tuple[int, str, str]] = []
    for t in range(close.size):
        if long_entry[t]:
            out.append((t, "BUY", "entry"))
        if short_entry[t]:
            out.append((t, "SELL", "entry"))
    return out
