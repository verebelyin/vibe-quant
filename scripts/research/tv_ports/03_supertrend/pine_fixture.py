"""Independent numpy transcription of ``03_supertrend_strategy.pine`` signals (R62).

The bar set is a deterministic synthetic 1h walk (fixed seed, 600 bars, ~0.9 % hourly
volatility). The seed was chosen so the script's entry signal fires at least three times in
each direction, so the comparison exercises the rules rather than a run of ``False``.

Pine rules transcribed (lines 14-24 of the script, verbatim semantics)::

    atr   = changeATR ? atr(Periods) : sma(tr, Periods)
    up    = src - Multiplier*atr;  up1 = nz(up[1], up);  up := close[1] > up1 ? max(up, up1) : up
    dn    = src + Multiplier*atr;  dn1 = nz(dn[1], dn);  dn := close[1] < dn1 ? min(dn, dn1) : dn
    trend := nz(trend[1], 1);     trend := trend == -1 and close > dn1 ? 1
                                          : trend == 1  and close < up1 ? -1 : trend
    buySignal  = trend == 1  and trend[1] == -1
    sellSignal = trend == -1 and trend[1] == 1

``ema``/``rma`` follow the project's Pine reference kernels: RMA seeds with the SMA of its
window, EMA with its first source value; ``tr`` is the usual true range. ``WARMUP`` skips the
bars where the engine's ATR does not exist yet, so the compiled strategy has no previous
``dir`` to compare against (Pine's ``trend`` is defined from bar 0).
"""

from __future__ import annotations

import math

import numpy as np

BARS = 600
SEED = 3

#: First compared bar: 10 = the ATR's warm-up (index 9 is its first value) plus one bar.
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


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    tr = high - low
    prev_close = np.concatenate([[math.nan], close[:-1]])
    tr = np.maximum(tr, np.abs(high - prev_close))
    return np.maximum(tr, np.abs(low - prev_close))


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


def _supertrend(
    src: np.ndarray, close: np.ndarray, atr: np.ndarray, multiplier: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(trend, atr)`` of the Pine ratchet; ``trend`` is +/-1 (NaN only where ``atr`` is)."""
    size = src.size
    trend = np.ones(size)
    prev_up = math.nan
    prev_dn = math.nan
    current = 1
    for t in range(size):
        up = src[t] - multiplier * atr[t]
        up1 = up if prev_up != prev_up else prev_up
        if t > 0 and close[t - 1] > up1:
            up = max(up, up1)
        dn = src[t] + multiplier * atr[t]
        dn1 = dn if prev_dn != prev_dn else prev_dn
        if t > 0 and close[t - 1] < dn1:
            dn = min(dn, dn1)
        if current == -1 and close[t] > dn1:
            current = 1
        elif current == 1 and close[t] < up1:
            current = -1
        prev_up, prev_dn = up, dn
        trend[t] = current
    return trend, atr


def events(bar_data: dict[str, np.ndarray]) -> list[tuple[int, str, str]]:
    """``(bar, side, action)`` of every Pine signal (script inputs at their defaults)."""
    open_ = bar_data["open"]
    high = bar_data["high"]
    low = bar_data["low"]
    close = bar_data["close"]
    src = (high + low) / 2.0  # input(hl2)
    atr = _rma(_true_range(high, low, close), 10)  # Periods 10, changeATR true
    trend, atr = _supertrend(src, close, atr, 3.0)  # Multiplier 3.0
    _ = open_
    prev = np.concatenate([[math.nan], trend[:-1]])
    ready = np.isfinite(atr)
    out: list[tuple[int, str, str]] = []
    for t in range(close.size):
        if not ready[t]:
            continue
        if trend[t] == 1 and prev[t] == -1:
            out.append((t, "BUY", "entry"))
        elif trend[t] == -1 and prev[t] == 1:
            out.append((t, "SELL", "entry"))
    return out
