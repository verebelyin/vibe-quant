"""Independent numpy transcription of ``01b_bollinger_rsi_double_v1_0.pine`` signals (R62).

The bar set is a deterministic synthetic 1h walk (fixed seed, 480 bars = 20 days, ~0.8 %
hourly volatility). The seed was chosen so the joint RSI + Bollinger condition fires at least
three times in each direction, so the comparison exercises the rules.

Pine rules transcribed (script defaults: RSI length 16, RSI value 45, Bollinger 20, 2.0)::

    RSIoverSold = RSIvalue        (45)
    RSIoverBought = 100 - RSIvalue (55)
    vrsi     = rsi(close, RSIlength)
    BBbasis  = sma(close, BBlength)
    BBupper  = BBbasis + BBmult*stdev(close, BBlength)
    BBlower  = BBbasis - BBmult*stdev(close, BBlength)
    buyEntry = crossover(vrsi, RSIoverSold)  and crossover(close, BBlower)
    sellEntry= crossunder(vrsi, RSIoverBought) and crossunder(close, BBupper)

``stdev`` is Pine's biased (population) standard deviation, matching the BBANDS indicator.
The strategy has no exits and no stop/target; ``strategy.entry`` reverses an open position and
the port replaces the stop-entry at the band with a market entry (the known gap). ``WARMUP``
skips the bars before the Bollinger window exists (the engine's readiness gate).
"""

from __future__ import annotations

import math

import numpy as np

BARS = 480
SEED = 391

RSI_LENGTH = 16
RSI_VALUE = 45
BB_LENGTH = 20
BB_MULT = 2.0

#: First compared bar: the 20-bar Bollinger window's first value is index 19.
WARMUP = 20


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
        rs = up / down
        out = 100.0 - 100.0 / (1.0 + rs)
    out = np.where((down == 0.0) & (up > 0.0), 100.0, out)
    out = np.where((down == 0.0) & (up == 0.0), math.nan, out)
    return out


def _stdev(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(x.size, np.nan)
    for t in range(n - 1, x.size):
        window = x[t - n + 1 : t + 1]
        out[t] = math.sqrt(float(np.mean((window - window.mean()) ** 2)))
    return out


def _cross_up(left: np.ndarray, right: np.ndarray | float) -> np.ndarray:
    """Pine ``crossover``: ``left > right and left[1] <= right[1]`` (NaN compares false)."""
    right = np.broadcast_to(np.asarray(right, dtype=np.float64), left.shape)
    prev_left = np.concatenate([[math.nan], left[:-1]])
    prev_right = np.concatenate([[math.nan], right[:-1]])
    return (left > right) & (prev_left <= prev_right)


def _cross_dn(left: np.ndarray, right: np.ndarray | float) -> np.ndarray:
    """Pine ``crossunder``: ``left < right and left[1] >= right[1]`` (NaN compares false)."""
    right = np.broadcast_to(np.asarray(right, dtype=np.float64), left.shape)
    prev_left = np.concatenate([[math.nan], left[:-1]])
    prev_right = np.concatenate([[math.nan], right[:-1]])
    return (left < right) & (prev_left >= prev_right)


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
    rsi = _rsi(close, RSI_LENGTH)
    basis = _sma(close, BB_LENGTH)
    dev = BB_MULT * _stdev(close, BB_LENGTH)
    upper = basis + dev
    lower = basis - dev

    long_entry = _cross_up(rsi, RSI_VALUE) & _cross_up(close, lower)
    short_entry = _cross_dn(rsi, 100 - RSI_VALUE) & _cross_dn(close, upper)
    out: list[tuple[int, str, str]] = []
    for t in range(close.size):
        if long_entry[t]:
            out.append((t, "BUY", "entry"))
        if short_entry[t]:
            out.append((t, "SELL", "entry"))
    return out
