"""Independent numpy transcription of ``02_macd_sma200.pine`` signals (R62).

The bar set is a deterministic synthetic 1h walk (fixed seed, 960 bars = 40 days, ~0.8 %
hourly volatility). The seed was chosen so both entries fire at least three times, so the
comparison exercises the rules.

Pine rules transcribed (script defaults: MACD 12/26/9, SMA 200)::

    fastMA     = sma(source, fastLength)
    slowMA     = sma(source, slowLength)
    veryslowMA = sma(source, veryslowLength)
    macd   = fastMA - slowMA;  signal = sma(macd, signalLength);  hist = macd - signal
    long   = crossover(hist, 0)  and macd > 0 and fastMA > slowMA
             and close[slowLength] > veryslowMA
    short  = crossunder(hist, 0) and macd < 0 and fastMA < slowMA
             and close[slowLength] < veryslowMA

``close[slowLength]`` is the literal close series (not the ``source`` input), so the
transcription reads ``close`` there. The script's ``strategy.cancel`` calls only manage its
pending STOP orders; the port enters at market, so they disappear with the stop-entry gap.
``WARMUP`` skips the bars before the 200-bar SMA exists (the engine's readiness gate).
"""

from __future__ import annotations

import math

import numpy as np

BARS = 960
SEED = 7

FAST = 12
SLOW = 26
SIGNAL = 9
VERYSLOW = 200

#: First compared bar: the 200-bar SMA's first value is index 199.
WARMUP = 200


def _sma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(x.size, np.nan)
    for t in range(n - 1, x.size):
        out[t] = x[t - n + 1 : t + 1].sum() / n
    return out


def _cross_up(left: np.ndarray, right: float) -> np.ndarray:
    prev = np.concatenate([[math.nan], left[:-1]])
    return (left > right) & (prev <= right)


def _cross_dn(left: np.ndarray, right: float) -> np.ndarray:
    prev = np.concatenate([[math.nan], left[:-1]])
    return (left < right) & (prev >= right)


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
    fast = _sma(close, FAST)
    slow = _sma(close, SLOW)
    veryslow = _sma(close, VERYSLOW)
    macd = fast - slow
    signal = _sma(macd, SIGNAL)
    hist = macd - signal
    close_back = np.concatenate([[math.nan] * SLOW, close[:-SLOW]])

    long_entry = (
        _cross_up(hist, 0.0) & (macd > 0.0) & (fast > slow) & (close_back > veryslow)
    )
    short_entry = (
        _cross_dn(hist, 0.0) & (macd < 0.0) & (fast < slow) & (close_back < veryslow)
    )
    out: list[tuple[int, str, str]] = []
    for t in range(close.size):
        if long_entry[t]:
            out.append((t, "BUY", "entry"))
        if short_entry[t]:
            out.append((t, "SELL", "entry"))
    return out
