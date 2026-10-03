"""Derived-output helpers for channel-style indicators.

These helpers compute scalar values that are derived at runtime from a live
NautilusTrader indicator instance plus the latest close price. They are
referenced by ``IndicatorSpec.computed_outputs`` (a mapping from output name
to helper name). The compiler generates code that imports and calls them.

Keeping them here (rather than inlining in compiler-generated code, which is
how the pre-P2 design worked) lets plugins declare custom derived outputs
without touching the compiler, and lets the formulas be unit-tested in
isolation.

All helpers are pure functions: no side effects, no NT imports at module
level (only inside TYPE_CHECKING), no hidden state. The ``ind_obj`` argument
is duck-typed as any object exposing ``.upper``, ``.middle``, ``.lower``
numeric attributes — the real type at runtime is an NT indicator, but we
accept any duck match so tests can pass a ``SimpleNamespace``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nautilus_trader.indicators.base import Indicator  # noqa: F401


def compute_percent_b(ind_obj: object, close: float) -> float:
    """Compute Bollinger %B = (close - lower) / (upper - lower).

    Range: typically 0-1 (close inside the bands) but can overshoot. When
    the band range collapses to zero (flat market, degenerate band), returns
    the neutral mid-value 0.5.

    Args:
        ind_obj: Any object exposing ``.upper`` and ``.lower`` numeric attrs
            (a live NT ``BollingerBands`` instance at runtime).
        close: Latest close price.

    Returns:
        %B scalar.
    """
    upper = float(ind_obj.upper)  # type: ignore[attr-defined]
    lower = float(ind_obj.lower)  # type: ignore[attr-defined]
    band_range = upper - lower
    if band_range > 0:
        return (close - lower) / band_range
    return 0.5


def compute_bandwidth(ind_obj: object, close: float) -> float:  # noqa: ARG001
    """Compute Bollinger bandwidth = (upper - lower) / middle.

    Close is unused but included in the signature so every derived-output
    helper has the same shape, which lets the compiler call them uniformly.

    Returns 0.0 when the middle band is non-positive (safety fallback).

    Args:
        ind_obj: Any object exposing ``.upper``, ``.middle``, ``.lower``
            numeric attrs (a live NT ``BollingerBands`` instance at runtime).
        close: Latest close price (unused, kept for signature uniformity).

    Returns:
        Bandwidth scalar.
    """
    upper = float(ind_obj.upper)  # type: ignore[attr-defined]
    lower = float(ind_obj.lower)  # type: ignore[attr-defined]
    middle = float(ind_obj.middle)  # type: ignore[attr-defined]
    if middle > 0:
        return (upper - lower) / middle
    return 0.0


def compute_position(ind_obj: object, close: float) -> float:
    """Compute Donchian channel position = (close - lower) / (upper - lower).

    Range: typically 0-1 (0 = at lower band, 1 = at upper band). When the
    channel collapses to zero width, returns the neutral mid-value 0.5.

    Args:
        ind_obj: Any object exposing ``.upper`` and ``.lower`` numeric attrs
            (a live NT ``DonchianChannel`` instance at runtime).
        close: Latest close price.

    Returns:
        Position scalar.
    """
    upper = float(ind_obj.upper)  # type: ignore[attr-defined]
    lower = float(ind_obj.lower)  # type: ignore[attr-defined]
    channel_range = upper - lower
    if channel_range > 0:
        return (close - lower) / channel_range
    return 0.5


def compute_natr_hourly(ind_obj: object, close: float, bar_minutes: int) -> float:
    """Normalized ATR in % of price, rescaled to a 1-hour-bar equivalent.

    ``100 * ATR / close * sqrt(60 / bar_minutes)``. On 1h bars this is the
    classic NATR; on other timeframes the square-root-of-time factor maps the
    value onto the 1h scale, so one threshold denotes the same volatility
    regime on every timeframe and every symbol (an absolute-price ATR
    threshold cannot: BTC 1m ATR ~40 USD vs SOL ~0.1 USD).

    Args:
        ind_obj: Object exposing ``.value`` (a live NT ``AverageTrueRange``).
        close: Latest close price; non-positive -> 0.0.
        bar_minutes: Bar length of the indicator's timeframe in minutes.

    Returns:
        1h-equivalent NATR (percent).
    """
    if close <= 0:
        return 0.0
    atr = float(ind_obj.value)  # type: ignore[attr-defined]
    return atr / close * 100.0 * math.sqrt(60.0 / bar_minutes)


def linear_weights(period: int) -> list[float]:
    """Linear WMA weights ``[1, 2, ..., period]`` (newest bar weighs most).

    NT's ``WeightedMovingAverage`` with ``weights=None`` is an equal-weight
    SMA; passing these weights makes it the standard (pandas-ta) WMA.
    """
    return [float(i) for i in range(1, int(period) + 1)]
