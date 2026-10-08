"""TREND_ENSEMBLE — stateful multi-lookback Donchian trend ensemble.

Zarattini, Pagani, Barbon (2025, "Catching Crypto Trends") is a long-only
Donchian breakout ensemble with trailing stops; the symmetric ``-1``
(short) state here is this project's extension. For every lookback ``L``
in ``(5, 10, 20, 30, 60, 90, 150, 250)`` filtered to
``L <= max_lookback``, with channels shifted by one bar so the value at
``t`` uses only bars ``<= t`` (no look-ahead)::

    upper_L[t] = rolling_max(high, L)[t-1]
    lower_L[t] = rolling_min(low, L)[t-1]

State ``s_L`` starts at 0 and is NaN while either channel is undefined
(``t < L``); at each bar ``t``::

    s_L[t] = +1              if close[t] > upper_L[t]
             -1              if close[t] < lower_L[t]
             s_L[t-1]        otherwise (carry)

The carry is bounded: ``s_L`` is the sign of the most recent breakout
within the last ``8 * L`` bars, else 0 — a breakout older than ``8 * L``
bars is forgotten::

    s_L[t] = 0               if t - last_breakout(t) >= 8 * L

The output is the mean of ``s_L`` over the lookbacks whose state is
defined at ``t`` (NaN if none) — a value in ``[-1, +1]``.

Warmup and buffering
--------------------
``pta_lookback_fn`` requests ``max_lookback + 1`` bars: the first bar at
which the longest included channel (and therefore every included state)
is defined. The engine trims each compute_fn bar buffer to
``max(400, 10 x (max_lookback + 1))`` bars (2510 for the default
``max_lookback=250``). The 8L state memory plus the L bars of channel
history a value at ``t`` can depend on is at most ``9L`` bars, and the cap
satisfies ``10 x (max_lookback + 1) > 9L`` for every included ``L`` — so
every value is independent of buffer trimming (a trimmed buffer always
holds all the history its tail depends on).

Usage::

    indicators:
      trend:
        type: TREND_ENSEMBLE
        max_lookback: 250
    entry_conditions:
      long:
        - trend > 0.5
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from vibe_quant.dsl.compute_builtins import int_param, nan_like
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry

if TYPE_CHECKING:
    import pandas as pd

_LOOKBACKS: tuple[int, ...] = (5, 10, 20, 30, 60, 90, 150, 250)
_DEFAULT_MAX_LOOKBACK = 250
_MIN_LOOKBACK = 5
_MEMORY_MULTIPLE = 8


def _shifted_donchian_channels(
    high: np.ndarray, low: np.ndarray, lookback: int
) -> tuple[np.ndarray, np.ndarray]:
    """Causal channels: ``upper[t] = max(high[t-L:t])``, NaN for ``t < L``.

    scipy.ndimage at C speed replaces the pandas rolling path (~5x the
    channel ops on a 2510-bar buffer).
    """
    from scipy.ndimage import maximum_filter1d, minimum_filter1d

    n = high.shape[0]
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    if n > lookback:
        # The engine buffer never contains NaN high/low (scipy and pandas
        # rolling differ on NaN).
        mx = maximum_filter1d(high, size=lookback, origin=(lookback - 1) // 2)
        mn = minimum_filter1d(low, size=lookback, origin=(lookback - 1) // 2)
        upper[lookback:] = mx[lookback - 1 : n - 1]
        lower[lookback:] = mn[lookback - 1 : n - 1]
    return upper, lower


def compute_trend_ensemble(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """Mean breakout state over the included Donchian lookbacks."""
    import pandas as pd

    max_lookback = int_param(params, "max_lookback", _DEFAULT_MAX_LOOKBACK)
    if max_lookback < _MIN_LOOKBACK:
        raise ValueError("max_lookback must be >= 5")
    lookbacks = [lookback for lookback in _LOOKBACKS if lookback <= max_lookback]
    n = len(df.index)
    if n == 0:
        return nan_like(df)

    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    states = np.full((len(lookbacks), n), np.nan, dtype=np.float64)
    bar_index = np.arange(n)
    for row, lookback in enumerate(lookbacks):
        if n <= lookback:
            continue
        upper, lower = _shifted_donchian_channels(high, low, lookback)
        signal = np.zeros(n, dtype=np.float64)
        signal[close > upper] = 1.0
        signal[close < lower] = -1.0
        # Carry each signal forward: the state at t is the sign of the most
        # recent breakout at or before t, and 0 before the first one.
        last_breakout = np.maximum.accumulate(
            np.where(signal != 0.0, bar_index, -1)
        )
        carried = signal[np.maximum(last_breakout, 0)].copy()
        carried[last_breakout < 0] = 0.0
        carried[(bar_index - last_breakout) >= _MEMORY_MULTIPLE * lookback] = 0.0
        carried[:lookback] = np.nan
        states[row] = carried

    count = np.count_nonzero(~np.isnan(states), axis=0)
    total = np.nansum(states, axis=0)
    value = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return cast("pd.Series", pd.Series(value, index=df.index, name="TREND_ENSEMBLE"))


def _trend_ensemble_lookback(params: dict[str, object]) -> int:
    """Bars required before every included channel — and the output — is defined."""
    return int_param(params, "max_lookback", _DEFAULT_MAX_LOOKBACK) + 1


indicator_registry.register_spec(
    IndicatorSpec(
        name="TREND_ENSEMBLE",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"max_lookback": 250},
        param_schema={"max_lookback": int},
        compute_fn=compute_trend_ensemble,
        pta_lookback_fn=_trend_ensemble_lookback,
        display_name="Trend Ensemble (multi-lookback Donchian)",
        description=(
            "Mean Donchian breakout state (+1 / -1 / carry) across fixed "
            "lookbacks 5-250, filtered by max_lookback. Ensemble trend "
            "signal in [-1, 1]."
        ),
        category="Trend",
        chart_placement="oscillator",
        param_ranges={"max_lookback": (60.0, 250.0)},
        threshold_range=(-1.0, 1.0),
    )
)
