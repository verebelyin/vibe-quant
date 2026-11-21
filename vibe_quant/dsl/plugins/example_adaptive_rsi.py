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

import math
from itertools import repeat
from typing import TYPE_CHECKING

import numpy as np

from vibe_quant.dsl.compute_builtins import float_param, int_param
from vibe_quant.dsl.indicators import IndicatorParamError, IndicatorSpec, indicator_registry
from vibe_quant.dsl.warmup_class import SEED_PARITY_REASON, refuse

if TYPE_CHECKING:
    import pandas as pd


def _efficiency_ratio(close: np.ndarray, period: int) -> np.ndarray:
    """Kaufman efficiency ratio: abs(direction) / volatility.

    Vectorized: uses ``np.convolve`` for the rolling sum of absolute
    price changes instead of a Python-level loop.
    """
    direction = np.abs(close[period:] - close[:-period])
    abs_diff = np.abs(np.diff(close))
    # Rolling sum of abs_diff over `period` bars via convolve.
    volatility = np.convolve(abs_diff, np.ones(period), mode="full")[
        period - 1 : period - 1 + len(direction)
    ]
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

    period = int_param(params, "period", 14)
    alpha = float_param(params, "alpha", 0.7)

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
    #
    # Plain-float loops over lists, op for op with the original per-element numpy
    # loops (bit-identical; a catalog-origin replay runs over millions of 1m bars):
    # elementwise numpy +, -, *, / round like the scalar ops, but the power stays a
    # scalar libm ``pow`` call (``np.power`` turns an exponent of 2.0 or 0.5, i.e.
    # alpha 0.5 or 2, into square / sqrt) and each recurrence step stays
    # ``prev * (1 - k) + x * k`` (CPython never contracts it to an FMA).
    fast_sc = 2.0 / (period + 1)
    slow_sc = 2.0 / (2 * period + 1)
    power = 1.0 / max(alpha, 0.01)
    sc = np.full(n - 1, fast_sc)
    base = (er[period + 1 :] * (fast_sc - slow_sc) + slow_sc).tolist()
    sc[period:] = list(map(math.pow, base, repeat(power, len(base))))

    # EMA of gains and losses with adaptive smoothing, seeded with an SMA
    keep = (1 - sc).tolist()
    gain_k = (gains * sc).tolist()
    loss_k = (losses * sc).tolist()
    avg_gain = [math.nan] * (n - 1)
    avg_loss = [math.nan] * (n - 1)
    gain = avg_gain[period - 1] = float(gains[:period].mean())
    loss = avg_loss[period - 1] = float(losses[:period].mean())
    for i in range(period, n - 1):
        k = keep[i]
        gain = gain * k + gain_k[i]
        loss = loss * k + loss_k[i]
        avg_gain[i] = gain
        avg_loss[i] = loss

    # RSI formula (elementwise): NaN where either average is NaN, 100 on a zero loss
    ag = np.asarray(avg_gain[period - 1 :], dtype=np.float64)
    al = np.asarray(avg_loss[period - 1 :], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        body = np.where(al == 0, 100.0, 100.0 - 100.0 / (1.0 + ag / al))
    body[np.isnan(ag) | np.isnan(al)] = np.nan
    rsi = np.full(n, np.nan)
    rsi[period:] = body

    return pd.Series(rsi, index=df.index)


def _alpha_check(params: dict[str, object]) -> str | None:
    """Refuse alpha outside [0.7, 1.0] at strategy save (user ruling, bead vibe-quant-b9uet).

    ADAPTIVE_RSI(14, alpha 0.3) needs ~154,000 bars to settle, so a low alpha makes
    the value depend on how much history exists.
    """
    alpha = float_param(params, "alpha", 0.7)
    if 0.7 <= alpha <= 1.0:
        return None
    return (
        f"ADAPTIVE_RSI alpha must be between 0.7 and 1.0 (got {alpha}): lower values "
        "take ~154,000+ bars to settle, so results depend on how much history exists"
    )


def _lookback(params: dict[str, object]) -> int:
    """Bars buffered before the first value; also the run-time floor check.

    The strategy computes it from its EFFECTIVE params when it is built, so a sweep or
    override value that never passed strategy validation is refused here too.
    """
    error = _alpha_check(params)
    if error is not None:
        raise IndicatorParamError(error)
    return int_param(params, "period", 14) * 2


# ---------------------------------------------------------------------------
# Register the spec — this is the entire "plugin contract".
# ---------------------------------------------------------------------------

indicator_registry.register_spec(
    IndicatorSpec(
        name="ADAPTIVE_RSI",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"period": 14, "alpha": 0.7},
        param_schema={"period": int, "alpha": float},
        compute_fn=compute_adaptive_rsi,
        pta_lookback_fn=_lookback,
        paper_warmup=refuse(SEED_PARITY_REASON),
        param_check=_alpha_check,
        catalog_origin=True,
        display_name="Adaptive RSI",
        description=(
            "RSI with Kaufman-style adaptive smoothing. Responsive in "
            "trends, filtered in chop. alpha controls aggressiveness."
        ),
        category="Momentum",
        param_ranges={"period": (5.0, 50.0), "alpha": (0.7, 0.9)},
        threshold_range=(20.0, 80.0),
    )
)
