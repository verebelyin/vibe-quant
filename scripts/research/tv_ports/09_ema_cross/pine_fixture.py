"""Independent numpy transcription of ``09_single_ema_cross.pine`` signals (R62).

The bar set is a deterministic synthetic 1h walk (fixed seed, 480 bars = 20 days, ~0.8 %
hourly volatility). The seed was chosen so both crosses fire at least three times, so the
comparison exercises the rules rather than a run of ``False``.

Pine rules transcribed (script defaults: EMA 1 = 10, EMA 2 = 20, test window always true)::

    expo = ema(close, ema1);  ma = ema(close, ema2)
    longCondition  = crossover(expo, ma)
    shortCondition = crossunder(expo, ma)
    if testPeriod()
        strategy.entry("Long",  strategy.long,  when=longCondition)
        strategy.entry("Short", strategy.short, when=shortCondition)

``strategy.entry`` with the opposite id reverses the open position, so the port sets
``position_mode: reverse``. ``WARMUP`` starts after the slow EMA's readiness gate (the engine's
indicator gate); Pine's ``ema`` exists from its first bar, the port's from ``length`` bars.
"""

from __future__ import annotations

import math

import numpy as np

BARS = 480
SEED = 4

EMA1_LENGTH = 10
EMA2_LENGTH = 20

#: First compared bar: the slow EMA's readiness gate ends at index 19.
WARMUP = EMA2_LENGTH


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    """Pine ``ema``: ``alpha = 2/(n+1)``, seeded with the first source value."""
    alpha = 2.0 / (n + 1)
    out = np.full(x.size, math.nan)
    prev = math.nan
    for t in range(x.size):
        prev = x[t] if prev != prev else alpha * x[t] + (1 - alpha) * prev
        out[t] = prev
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
    expo = _ema(close, EMA1_LENGTH)
    ma = _ema(close, EMA2_LENGTH)

    long_entry = _cross_up(expo, ma)
    short_entry = _cross_dn(expo, ma)
    out: list[tuple[int, str, str]] = []
    for t in range(close.size):
        if long_entry[t]:
            out.append((t, "BUY", "entry"))
        if short_entry[t]:
            out.append((t, "SELL", "entry"))
    return out
