"""Independent numpy transcription of ``12_flawless_victory.pine`` v2 signals (R62).

The bar set is a deterministic synthetic 1h walk (fixed seed, 960 bars = 40 days, ~0.8 %
hourly volatility). The seed was chosen so the v2 long entry fires at least three times.

Pine rules transcribed (v2 block only: the script's defaults are v1=true, v2=false, v3=false;
this folder fixes v2=true, v3=false; literals BB 17 / 1.0, RSI 14, guards 42 / 76)::

    rsi    = rsi(close, 14)
    basis  = sma(close, 17);  dev = 1.0 * stdev(close, 17)
    upper  = basis + dev;     lower = basis - dev
    Buy_2  = close < lower and rsi > 42
    Sell_2 = close > upper and rsi > 76
    if v2 == true
        strategy.entry("Long", strategy.long, when = Buy_2)
        strategy.close("Long", when = Sell_2)
        strategy.exit("Stoploss/TP", "Long", stop = ..., limit = ...)

The ``strategy.exit`` levels are ``position_avg_price * (1 -/+ pct/100)``; the port's
``fixed_pct`` SL/TP is the same level anchored on the entry price, so the fixture only needs
the signal bars (the broker emulator checks the levels in R63). ``WARMUP`` excludes NT's
Wilder-RSI seed difference (these are direct threshold comparisons, so a fraction of a point
flips a bar; see the v1 fixture).
"""

from __future__ import annotations

import math

import numpy as np

BARS = 960
SEED = 12

BB_LENGTH = 17
BB_MULT = 1.0
RSI_LENGTH = 14
RSI_LOWER = 42.0
RSI_UPPER = 76.0

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
    """``(bar, side, action)`` of every Pine signal (v2 defaults)."""
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
