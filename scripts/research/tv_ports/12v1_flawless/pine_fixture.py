"""Independent numpy transcription of ``12_flawless_victory.pine`` v1 signals (R62).

The bar set is a deterministic synthetic 1h walk (fixed seed, 960 bars = 40 days, ~0.8 %
hourly volatility). The seed was chosen so the v1 long entry and its exit both fire at least
three times, so the comparison exercises the rules rather than a run of ``False``.

Pine rules transcribed (v1 block only: the script's defaults are v1=true, v2=false, v3=false;
literals BB 20 / 1.0, RSI 14, guards 42 / 70)::

    rsi    = rsi(close, 14)
    basis  = sma(close, 20);  dev = 1.0 * stdev(close, 20)
    upper  = basis + dev;     lower = basis - dev
    Buy_1  = close < lower and rsi > 42
    Sell_1 = close > upper and rsi > 70
    if v1 == true
        strategy.entry("Long", strategy.long, when = Buy_1)
        strategy.close("Long", when = Sell_1)

``stdev`` is Pine's biased (population) standard deviation, matching the BBANDS indicator.
Long only. ``WARMUP`` is far past the 20-bar Bollinger window on purpose: these conditions are
DIRECT threshold comparisons (``rsi > 42``, not a crossover), so a fraction-of-a-point
difference in RSI flips a bar. The engine's RSI is NT's Wilder RSI, which seeds from its first
bar, while Pine's ``rsi()`` seeds with the SMA of the first ``len`` deltas; on this walk the
two agree to under 1e-9 from bar ~313, so every bar in that seed difference is excluded.
"""

from __future__ import annotations

import math

import numpy as np

BARS = 960
SEED = 12

BB_LENGTH = 20
BB_MULT = 1.0
RSI_LENGTH = 14
RSI_LOWER = 42.0
RSI_UPPER = 70.0

#: First compared bar: past the engine gate and NT's Wilder-RSI seed convergence to Pine.
WARMUP = 320


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


def _stdev(x: np.ndarray, n: int) -> np.ndarray:
    """Pine ``stdev``: the biased (population) standard deviation."""
    out = np.full(x.size, math.nan)
    for t in range(n - 1, x.size):
        window = x[t - n + 1 : t + 1]
        out[t] = math.sqrt(float(np.mean((window - window.mean()) ** 2)))
    return out


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
    """``(bar, side, action)`` of every Pine signal (v1 defaults)."""
    close = bar_data["close"]
    rsi = _rsi(close, RSI_LENGTH)
    basis = _sma(close, BB_LENGTH)
    dev = BB_MULT * _stdev(close, BB_LENGTH)
    upper = basis + dev
    lower = basis - dev

    long_entry = (close < lower) & (rsi > RSI_LOWER)
    long_exit = (close > upper) & (rsi > RSI_UPPER)
    out: list[tuple[int, str, str]] = []
    for t in range(close.size):
        if long_entry[t]:
            out.append((t, "BUY", "entry"))
        if long_exit[t]:
            out.append((t, "SELL", "exit"))
    return out
