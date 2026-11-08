"""Independent numpy transcription of ``08_ao_stoch_rsi_atr.pine`` signals (R62).

The bar set is a deterministic synthetic 1h walk (fixed seed, 960 bars = 40 days, ~0.9 %
hourly volatility). The seed was chosen so both entry conditions fire at least three times,
so the comparison exercises the rules rather than a run of ``False``.

Pine rules transcribed (script defaults: AO 5/34, Stoch K 14 / D 3, RSI 10, ATR 14)::

    awesome  = (sma(hl2, 5) - sma(hl2, 34)) * 1000
    k        = sma(stoch(close, high, low, 14), 3)
    rsi      = rsi(close, 10)
    atrvalue = rma(tr, 14)
    LongCondition  = k < 20  and rsi < 30 and awesome > awesome[1]
    ShortCondition = k > 80  and rsi > 70 and awesome < awesome[1]

The x1000 display scale is irrelevant to the two comparisons, so this file uses the plain AO.
``strategy.entry`` reverses an open position (the port sets ``position_mode: reverse``) and the
signal-bar ATR stops/targets are ``SIGNAL_LEVELS`` in the port, so they do not appear here.
``WARMUP`` excludes NT's Wilder-RSI seed difference: the RSI conditions are direct threshold
comparisons, so a fraction of a point flips a bar; the engine's NT RSI and Pine's SMA-seeded
``rsi()`` agree to about 1e-9 only after ~310 bars of this walk.
"""

from __future__ import annotations

import math

import numpy as np

BARS = 960
SEED = 1

AO_FAST = 5
AO_SLOW = 34
STOCH_K = 14
STOCH_D = 3
RSI_LENGTH = 10

#: First compared bar: past the engine gate and NT's Wilder-RSI seed difference.
WARMUP = 320


def _sma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(x.size, np.nan)
    for t in range(n - 1, x.size):
        window = x[t - n + 1 : t + 1]
        if np.isfinite(window).all():
            out[t] = window.sum() / n
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


def _stoch_raw(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int
) -> np.ndarray:
    """Pine ``stoch(close, high, low, n)``: ``100 * (close - lowest(low)) / (highest(high) - lowest(low))``."""
    out = np.full(close.size, math.nan)
    for t in range(n - 1, close.size):
        hh = high[t - n + 1 : t + 1].max()
        ll = low[t - n + 1 : t + 1].min()
        out[t] = 0.0 if hh == ll else 100.0 * (close[t] - ll) / (hh - ll)
    return out


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full(a.size, math.nan)
    out[k:] = a[:-k]
    return out


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


def events(bar_data: dict[str, np.ndarray]) -> list[tuple[int, str, str]]:
    """``(bar, side, action)`` of every Pine signal (script inputs at their defaults)."""
    high = bar_data["high"]
    low = bar_data["low"]
    close = bar_data["close"]
    hl2 = (high + low) / 2.0
    awesome = _sma(hl2, AO_FAST) - _sma(hl2, AO_SLOW)
    awesome_prev = _shift(awesome, 1)
    k = _sma(_stoch_raw(high, low, close, STOCH_K), STOCH_D)
    rsi = _rsi(close, RSI_LENGTH)

    long_entry = (k < 20.0) & (rsi < 30.0) & (awesome > awesome_prev)
    short_entry = (k > 80.0) & (rsi > 70.0) & (awesome < awesome_prev)
    out: list[tuple[int, str, str]] = []
    for t in range(close.size):
        if long_entry[t]:
            out.append((t, "BUY", "entry"))
        if short_entry[t]:
            out.append((t, "SELL", "entry"))
    return out
