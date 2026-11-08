"""Independent numpy transcription of ``12_flawless_victory.pine`` v3 signals (R62).

The bar set is a deterministic synthetic 1h walk (fixed seed, 960 bars = 40 days, ~0.8 %
hourly volatility). The seed was chosen so the v3 long entry fires at least three times.

Pine rules transcribed (v3 block only: this folder fixes v1=false, v2=false, v3=true; literals
BB 20 / 1.0, RSI 14, MFI 14, guards 60 / 64 / 65)::

    rsi    = rsi(close, 14)
    MFIlength = 14;  MFIsrc = hlc3
    MFIupper = sum(volume * (change(MFIsrc) <= 0 ? 0 : MFIsrc), 14)
    MFIlower = sum(volume * (change(MFIsrc) >= 0 ? 0 : MFIsrc), 14)
    mfi    = MFIlower == 0 ? 100 : MFIupper == 0 ? 0 : 100 - 100/(1 + MFIupper/MFIlower)
    basis  = sma(close, 20);  dev = 1.0 * stdev(close, 20)
    upper  = basis + dev;     lower = basis - dev
    Buy_3  = close < lower and mfi < 60
    Sell_3 = close > upper and rsi > 65 and mfi > 64
    if v3 == true
        strategy.entry("Long", strategy.long, when = Buy_3)
        strategy.close("Long", when = Sell_3)
        strategy.exit("Stoploss/TP", "Long", stop = ..., limit = ...)

``mfi`` is transcribed literally from the script (its own ``_rsi(MFIupper, MFIlower)`` helper on
a ``hlc3`` money flow), not delegated to the MFI built-in, so this file shares no project code.
The ``strategy.exit`` levels are ``position_avg_price * (1 -/+ pct/100)``, which the port's
``fixed_pct`` SL/TP reproduces from the entry price. ``WARMUP`` excludes NT's Wilder-RSI seed
difference (direct threshold comparisons; see the v1 fixture).
"""

from __future__ import annotations

import math

import numpy as np

BARS = 960
SEED = 12

BB_LENGTH = 20
BB_MULT = 1.0
RSI_LENGTH = 14
MFI_LENGTH = 14
MFI_LOWER = 60.0
MFI_UPPER = 64.0
RSI_UPPER = 65.0

#: First compared bar: past the engine gate and NT's Wilder-RSI seed difference.
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


def _mfi(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, n: int
) -> np.ndarray:
    """The script's MFI: two rolling sums of signed ``hlc3`` money flow, then its own ratio."""
    src = (high + low + close) / 3.0
    change = np.concatenate([[math.nan], np.diff(src)])
    upper = np.where(change <= 0.0, 0.0, src)  # change <= 0 counts 0 (NaN compares false)
    lower = np.where(change >= 0.0, 0.0, src)
    upper = np.concatenate([[math.nan], upper[1:]])
    lower = np.concatenate([[math.nan], lower[1:]])
    up = _rolling_sum(upper * volume, n)
    down = _rolling_sum(lower * volume, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 - 100.0 / (1.0 + up / down)
    out = np.where(down == 0.0, 100.0, out)
    out = np.where((down != 0.0) & (up == 0.0), 0.0, out)
    return out


def _rolling_sum(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(x.size, math.nan)
    for t in range(n - 1, x.size):
        window = x[t - n + 1 : t + 1]
        if np.isfinite(window).all():
            out[t] = float(window.sum())
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
    """``(bar, side, action)`` of every Pine signal (v3 defaults)."""
    close = bar_data["close"]
    rsi = _rsi(close, RSI_LENGTH)
    mfi = _mfi(bar_data["high"], bar_data["low"], close, bar_data["volume"], MFI_LENGTH)
    basis = _sma(close, BB_LENGTH)
    dev = BB_MULT * _stdev(close, BB_LENGTH)
    upper = basis + dev
    lower = basis - dev

    long_entry = (close < lower) & (mfi < MFI_LOWER)
    long_exit = (close > upper) & (rsi > RSI_UPPER) & (mfi > MFI_UPPER)
    out: list[tuple[int, str, str]] = []
    for t in range(close.size):
        if long_entry[t]:
            out.append((t, "BUY", "entry"))
        if long_exit[t]:
            out.append((t, "SELL", "exit"))
    return out
