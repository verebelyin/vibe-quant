"""Adaptive RSI — example indicator plugin.

Demonstrates the full plugin contract: a single file that declares a
``compute_fn``, registers an ``IndicatorSpec``, and auto-enrolls in both
the GA indicator pool (via ``param_ranges`` + ``threshold_range``) and
the frontend catalog API (via ``display_name``, ``description``,
``category``).

Algorithm
---------
Standard RSI smoothed by a Kaufman-style adaptive alpha that scales the
EMA smoothing factor by the *efficiency ratio* (ER) of the close series.
When the market is trending (ER close to 1) the effective period
shortens, making the indicator more responsive; in choppy conditions
(ER close to 0) the period lengthens, filtering noise.

Reference: Perry Kaufman, *Trading Systems and Methods*, 5th ed. (2013),
chapter on Adaptive Moving Average — the same ER concept applied here to
RSI's internal EMA smoothing.

Usage in a strategy YAML::

    indicators:
      arsi:
        type: ADAPTIVE_RSI
        period: 14
    entry_conditions:
      long:
        - arsi < 30
    exit_conditions:
      long:
        - arsi > 70
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry

if TYPE_CHECKING:
    import pandas as pd


def _efficiency_ratio(close: np.ndarray, period: int) -> np.ndarray:
    """Kaufman efficiency ratio: abs(direction) / volatility."""
    direction = np.abs(close[period:] - close[:-period])
    volatility = np.zeros_like(direction)
    abs_diff = np.abs(np.diff(close))
    for i in range(len(direction)):
        volatility[i] = abs_diff[i : i + period].sum()
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        er = np.where(volatility > 0, direction / volatility, 0.0)
    return er


def compute_adaptive_rsi(
    df: pd.DataFrame, params: dict[str, object]
) -> pd.Series:
    """Compute Adaptive RSI from OHLCV DataFrame.

    Args:
        df: DataFrame with at least a ``close`` column.
        params: Must contain ``period`` (int) and ``alpha`` (float, 0-1).

    Returns:
        Series of Adaptive RSI values (0-100 scale).
    """
    import pandas as pd

    period = int(params.get("period", 14))  # type: ignore[arg-type]
    alpha = float(params.get("alpha", 0.5))  # type: ignore[arg-type]

    close = df["close"].to_numpy(dtype=np.float64)
    n = len(close)
    if n < period + 1:
        return pd.Series(np.full(n, np.nan), index=df.index)

    # Price changes
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    # Efficiency ratio for the adaptive smoothing
    er = np.full(n, 0.5)
    er_values = _efficiency_ratio(close, period)
    er[period:] = er_values[: n - period]

    # Adaptive smoothing constant: blend between fast (2/(period+1)) and
    # slow (2/(2*period+1)) using ER, then raise to alpha power to
    # control aggressiveness.
    fast_sc = 2.0 / (period + 1)
    slow_sc = 2.0 / (2 * period + 1)
    sc = np.full(n - 1, fast_sc)
    for i in range(period, n - 1):
        sc[i] = (er[i + 1] * (fast_sc - slow_sc) + slow_sc) ** (1.0 / max(alpha, 0.01))

    # EMA of gains and losses with adaptive smoothing
    avg_gain = np.full(n - 1, np.nan)
    avg_loss = np.full(n - 1, np.nan)

    # Seed with SMA
    avg_gain[period - 1] = gains[:period].mean()
    avg_loss[period - 1] = losses[:period].mean()

    for i in range(period, n - 1):
        k = sc[i]
        avg_gain[i] = avg_gain[i - 1] * (1 - k) + gains[i] * k
        avg_loss[i] = avg_loss[i - 1] * (1 - k) + losses[i] * k

    # RSI formula
    rsi = np.full(n, np.nan)
    for i in range(period - 1, n - 1):
        ag = avg_gain[i]
        al = avg_loss[i]
        if np.isnan(ag) or np.isnan(al):
            continue
        if al == 0:
            rsi[i + 1] = 100.0
        else:
            rs = ag / al
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return pd.Series(rsi, index=df.index)


# ---------------------------------------------------------------------------
# Register the spec — this is the entire "plugin contract".
# ---------------------------------------------------------------------------

indicator_registry.register_spec(
    IndicatorSpec(
        name="ADAPTIVE_RSI",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"period": 14, "alpha": 0.5},
        param_schema={"period": int, "alpha": float},
        compute_fn=compute_adaptive_rsi,
        pta_lookback_fn=lambda p: int(p.get("period", 14)) * 2,
        display_name="Adaptive RSI",
        description=(
            "RSI with Kaufman-style adaptive smoothing. Responsive in "
            "trends, filtered in chop. alpha controls aggressiveness."
        ),
        category="Momentum",
        param_ranges={"period": (5.0, 50.0), "alpha": (0.1, 0.9)},
        threshold_range=(20.0, 80.0),
    )
)
