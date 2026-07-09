"""KAMA — Kaufman Adaptive Moving Average.

Perry Kaufman's adaptive smoothing applied to price. Uses the efficiency
ratio (ER) — abs(direction) / sum(abs(deltas)) — to blend between a
fast and slow EMA smoothing constant. In a trend, ER approaches 1 and
smoothing tightens; in chop, ER approaches 0 and smoothing widens.

Reference: Perry Kaufman, *Trading Systems and Methods*, 5th ed. (2013).

Bit-identical numpy port of ``pandas_ta_classic.kama``: same ER/smoothing
math (pandas ops reused from the library), but the O(n) recurrence runs on
numpy float64 scalars instead of the library's ``.iloc``-per-element Python
loop, which dominated screening evals (~60% of per-bar cost — profiled at
~3ms/call on a 250-row buffer vs ~0.05ms for this port). Exactness is
enforced by ``tests/unit/test_plugins/test_kama_exactness.py``.

Usage::

    indicators:
      kama_fast:
        type: KAMA
        period: 10
    entry_conditions:
      long:
        - close > kama_fast
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from vibe_quant.dsl.compute_builtins import int_param
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry

if TYPE_CHECKING:
    import pandas as pd


def compute_kama(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """KAMA over ``close`` at the given period. Fast/slow periods follow
    Kaufman's canonical 2/30 defaults (as in pandas-ta-classic)."""
    import numpy as np
    import pandas as pd
    from pandas_ta_classic.utils import non_zero_range, verify_series

    period = int_param(params, "period", 10)
    fast, slow, drift = 2, 30, 1

    close = verify_series(df["close"], max(fast, slow, period))
    if close is None:
        return cast("pd.Series", df["close"] * 0)

    # Smoothing constant — identical pandas ops to pandas_ta_classic.kama.
    fr = 2 / (fast + 1)
    sr = 2 / (slow + 1)
    abs_diff = non_zero_range(close, close.shift(period)).abs()
    peer_diff = non_zero_range(close, close.shift(drift)).abs()
    er = abs_diff / peer_diff.rolling(period).sum()
    x = er * (fr - sr) + sr
    sc = (x * x).to_numpy(dtype=np.float64)

    # Recurrence on float64 scalars — same operation order as the library's
    # `sc.iloc[i] * close.iloc[i] + (1 - sc.iloc[i]) * result[i - 1]`.
    c = close.to_numpy(dtype=np.float64)
    m = c.size
    result = np.full(m, np.nan, dtype=np.float64)
    prev = c[period - 1]
    result[period - 1] = prev
    for i in range(period, m):
        prev = sc[i] * c[i] + (1 - sc[i]) * prev
        result[i] = prev

    out = pd.Series(result, index=close.index)
    out.name = f"KAMA_{period}_{fast}_{slow}"
    return out


indicator_registry.register_spec(
    IndicatorSpec(
        name="KAMA",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"period": 10},
        param_schema={"period": int},
        compute_fn=compute_kama,
        pta_lookback_fn=lambda p: int_param(p, "period", 10) * 3,
        display_name="Kaufman Adaptive MA",
        description=(
            "Adaptive moving average that tightens in trends and widens "
            "in chop via the Kaufman efficiency ratio."
        ),
        category="Trend",
        chart_placement="overlay",
        param_ranges={"period": (5.0, 50.0)},
        threshold_range=None,
        ma_kind=True,
    )
)
