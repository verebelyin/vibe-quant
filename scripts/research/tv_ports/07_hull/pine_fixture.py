"""Independent numpy transcription of ``07_hull_suite.pine`` signals (R62).

The bar set is a deterministic synthetic 1h walk (fixed seed, 480 bars = 20 days, ~0.8 %
hourly volatility). The seed was chosen so the HULL/HULL[2] crossings give at least three
entries and three exits, so the comparison exercises the rules.

Pine rules transcribed (script defaults: Hma mode, length 55, source close)::

    HMA(src, n)  = wma(2*wma(src, n/2) - wma(src, n), round(sqrt(n)))
    EHMA(src, n) = ema(2*ema(src, n/2) - ema(src, n), round(sqrt(n)))
    THMA(src, n) = wma(wma(src, n/3)*3 - wma(src, n/2) - wma(src, n), n)
    HULL  = Mode(modeSwitch, src, length)      (Thma calls THMA with length/2)
    long  = HULL[0] > HULL[2]
    short = HULL[0] < HULL[2]

The script is long-only by default (``strategy.risk.allow_entry_in(long)``), so the port
expresses the ignored short entry as the long EXIT (design E7) and stays flat between trades:
this fixture's ``events`` return the long entry and the long exit. ``WARMUP`` starts after the
offset-2 HULL companion's first value (HMA(55)'s first value is index 60, so index 62).
"""

from __future__ import annotations

import math

import numpy as np

BARS = 480
SEED = 7

LENGTH = 55

#: First compared bar: HMA(55) is first finite at index 60; ``HULL[2]`` at index 62.
WARMUP = 62


def _wma(x: np.ndarray, n: int) -> np.ndarray:
    """Pine ``wma``: weights 1..n from the oldest bar, over ``n`` bars."""
    weights = np.arange(1.0, n + 1.0)
    out = np.full(x.size, math.nan)
    for t in range(n - 1, x.size):
        window = x[t - n + 1 : t + 1]
        if np.isfinite(window).all():
            out[t] = float(window @ weights) / float(weights.sum())
    return out


def _hma(x: np.ndarray, n: int) -> np.ndarray:
    inner = 2.0 * _wma(x, n // 2) - _wma(x, n)
    return _wma(inner, int(math.floor(math.sqrt(n) + 0.5)))


def _shift(values: np.ndarray, offset: int) -> np.ndarray:
    out = np.full(values.size, math.nan)
    out[offset:] = values[:-offset]
    return out


def bars() -> dict[str, np.ndarray]:
    """Deterministic 1h OHLCV bars (open, high, low, close, volume)."""
    rng = np.random.default_rng(SEED)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.008, BARS)))
    open_ = np.concatenate([[100.0], close[:-1]])
    spread = np.abs(rng.normal(0.0, 0.004, BARS))
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
    hull = _hma(close, LENGTH)
    hull_2 = _shift(hull, 2)
    out: list[tuple[int, str, str]] = []
    for t in range(close.size):
        if hull[t] > hull_2[t]:
            out.append((t, "BUY", "entry"))
        elif hull[t] < hull_2[t]:
            out.append((t, "SELL", "exit"))  # the short entry, expressed as the long exit
    return out
