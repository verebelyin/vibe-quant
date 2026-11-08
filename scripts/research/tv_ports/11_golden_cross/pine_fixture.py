"""Independent numpy transcription of ``11_golden_cross_sma200.pine`` signals (R62).

The bar set is a deterministic synthetic 1h walk (fixed seed, 480 bars = 20 days, ~0.8 %
hourly volatility). The seed was chosen so the bullish cross fires at least three times and
the bearish cross at least twice, so the comparison exercises both the entry and the exit.

Pine rules transcribed (script defaults: SMA 50 / SMA 200, long only)::

    movingaverage_fast = sma(close, 50)
    movingaverage_slow = sma(close, 200)
    bullish_cross = crossover(movingaverage_fast, movingaverage_slow)
    bearish_cross = crossunder(movingaverage_fast, movingaverage_slow)
    if bullish_cross
        strategy.entry("long", strategy.long)
    strategy.close("long", when = bearish_cross)

There is no short side: the script only enters long and closes that position on the
crossunder. ``WARMUP`` starts after the 200-bar SMA's readiness gate.
"""

from __future__ import annotations

import math

import numpy as np

BARS = 480
SEED = 418

FAST_LENGTH = 50
SLOW_LENGTH = 200

#: First compared bar: the 200-bar SMA's first value is index 199.
WARMUP = SLOW_LENGTH


def _sma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(x.size, math.nan)
    for t in range(n - 1, x.size):
        out[t] = x[t - n + 1 : t + 1].sum() / n
    return out


def _cross_up(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pine ``crossover``: ``left > right and left[1] <= right[1]`` (NaN compares false)."""
    prev_left = np.concatenate([[math.nan], left[:-1]])
    prev_right = np.concatenate([[math.nan], right[:-1]])
    return (left > right) & (prev_left <= prev_right)


def _cross_dn(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pine ``crossunder``: ``left < right and left[1] >= right[1]`` (NaN compares false)."""
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


def events(bar_data: dict[str, np.ndarray]) -> list[tuple[int, str, str]]:
    """``(bar, side, action)`` of every Pine signal (script inputs at their defaults)."""
    close = bar_data["close"]
    fast = _sma(close, FAST_LENGTH)
    slow = _sma(close, SLOW_LENGTH)

    long_entry = _cross_up(fast, slow)
    long_exit = _cross_dn(fast, slow)
    out: list[tuple[int, str, str]] = []
    for t in range(close.size):
        if long_entry[t]:
            out.append((t, "BUY", "entry"))
        if long_exit[t]:
            out.append((t, "SELL", "exit"))
    return out
