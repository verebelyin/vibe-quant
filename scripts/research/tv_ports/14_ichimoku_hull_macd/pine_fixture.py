"""Independent numpy transcription of ``14_ichimoku_hull_macd.pine`` signals (R62).

The bar set is a deterministic synthetic 1h walk (fixed seed, 1200 bars = 50 whole UTC days,
~0.8 % hourly volatility), so the 1d bars aggregate from whole days. The seed was chosen so the
long entry fires at least three times.

Pine rules transcribed (script defaults: Period 14, Ichimoku 9/26/52, MACD 12/26/9, `price` =
open, resolution "D")::

    hma1 = hma(open, 14);  hma2 = hma(open[1], 14)
    D1   = security(tickerid, "D", open);  D2 = security(tickerid, "D", open[1])
    leadLine1 = avg(donchian(9), donchian(26));  leadLine2 = donchian(52)
    MACD = hma(open, 12) - hma(open, 26);  aMACD = hma(MACD, 9)
    long  = hma1 > hma2 and D1 > D2 and open > hma2 and leadLine1 > leadLine2 and MACD > aMACD
    short = hma1 < hma2 and D1 < D2 and open < hma2 and leadLine1 < leadLine2 and MACD < aMACD

``hma`` is Pine's BUILT-IN (floored root, ruling R-H1), transcribed here rather than imported.
``donchian(n) = avg(lowest(low, n), highest(high, n))``.

The daily values: ``D1``/``D2`` are the CURRENT day's open and the PREVIOUS day's open, and a
1h primary sees the new day's value on its FINAL bar of the old day (the engine merges "1d"
before "1h" at the midnight tie). ``events`` therefore evaluates the day's daily pair from the
bar's own day index.

``WARMUP`` starts after the slowest gate: ``hma(open, 26)``, then ``hma(·, 9)`` of the MACD, then
its 1-bar lag (``hma2``), i.e. ``26 + 9 - 1 + 1 = 35``, and after the first whole day's daily
pair exists (bar 23).
"""

from __future__ import annotations

import math

import numpy as np

BARS = 1200
SEED = 14
HOURS_PER_DAY = 24

PERIOD = 14
CONVERSION = 9
BASE = 26
SPAN = 52
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

#: First compared bar: past ``hma(26)`` -> ``hma(9)`` -> offset 1 and the first daily pair.
WARMUP = 36


def _wma(x: np.ndarray, n: int) -> np.ndarray:
    """Pine ``wma``: weights ``1..n`` from the oldest bar."""
    weights = np.arange(1.0, n + 1.0)
    out = np.full(x.size, math.nan)
    for t in range(n - 1, x.size):
        window = x[t - n + 1 : t + 1]
        if np.isfinite(window).all():
            out[t] = float(window @ weights) / float(weights.sum())
    return out


def _hma(x: np.ndarray, n: int) -> np.ndarray:
    """Pine BUILT-IN ``hma``: ``wma(2*wma(src, n//2) - wma(src, n), isqrt(n))`` (R-H1)."""
    return _wma(2.0 * _wma(x, n // 2) - _wma(x, n), math.isqrt(n))


def _shift(values: np.ndarray, offset: int) -> np.ndarray:
    out = np.full(values.size, math.nan)
    out[offset:] = values[:-offset]
    return out


def _donchian(high: np.ndarray, low: np.ndarray, n: int) -> np.ndarray:
    """``avg(lowest(low, n), highest(high, n))``."""
    out = np.full(high.size, math.nan)
    for t in range(n - 1, high.size):
        out[t] = (float(low[t - n + 1 : t + 1].min()) + float(high[t - n + 1 : t + 1].max())) / 2.0
    return out


def bars() -> dict[str, np.ndarray]:
    """Deterministic 1h OHLCV bars (open, high, low, close, volume), a whole number of days."""
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


def daily_opens(bar_data: dict[str, np.ndarray]) -> np.ndarray:
    """The 1d ``open`` per day (each day's first hourly open)."""
    open_ = bar_data["open"]
    days = open_.size // HOURS_PER_DAY
    return np.array([open_[d * HOURS_PER_DAY] for d in range(days)])


def _daily_pair(bar_data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """``(D1, D2)`` per HOURLY bar: the day's own open, and the previous day's open.

    A 1h primary reads the new day's values on its final bar (the engine's "1d" < "1h" merge
    order at the midnight tie), and a 15m primary would read them one bar late — which is why
    the port refuses 15m (ruling D5).
    """
    opens = daily_opens(bar_data)
    d1 = np.full(bar_data["open"].size, math.nan)
    d2 = np.full(bar_data["open"].size, math.nan)

    def prev_day_open(day: int, lag: int) -> float:
        index = day - lag
        return float(opens[index]) if index >= 0 else math.nan

    for t in range(d1.size):
        day = t // HOURS_PER_DAY
        # The day's own 1d bar has ts_init = the day's LAST 1m close, i.e. the same instant as
        # the final hourly bar; "1d" sorts before "1h", so only on that bar has the new daily
        # value been merged. Every earlier bar of the day still reads the previous day's.
        step = day if t % HOURS_PER_DAY == HOURS_PER_DAY - 1 else day - 1
        d1[t] = prev_day_open(step, 0)
        d2[t] = prev_day_open(step, 1)
    return d1, d2


def events(bar_data: dict[str, np.ndarray]) -> list[tuple[int, str, str]]:
    """``(bar, side, action)`` of every Pine signal (script inputs at their defaults)."""
    open_ = bar_data["open"]
    hma1 = _hma(open_, PERIOD)
    hma2 = _shift(_hma(open_, PERIOD), 1)
    d1, d2 = _daily_pair(bar_data)
    high = bar_data["high"]
    low = bar_data["low"]
    lead1 = (_donchian(high, low, CONVERSION) + _donchian(high, low, BASE)) / 2.0
    lead2 = _donchian(high, low, SPAN)
    macd = _hma(open_, MACD_FAST) - _hma(open_, MACD_SLOW)
    signal = _hma(macd, MACD_SIGNAL)

    long_entry = (
        (hma1 > hma2) & (d1 > d2) & (open_ > hma2) & (lead1 > lead2) & (macd > signal)
    )
    short_entry = (
        (hma1 < hma2) & (d1 < d2) & (open_ < hma2) & (lead1 < lead2) & (macd < signal)
    )
    out: list[tuple[int, str, str]] = []
    for t in range(open_.size):
        if long_entry[t]:
            out.append((t, "BUY", "entry"))
        if short_entry[t]:
            out.append((t, "SELL", "entry"))
    return out
