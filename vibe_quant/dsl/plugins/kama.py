"""KAMA — Kaufman Adaptive Moving Average.

Perry Kaufman's adaptive smoothing applied to price. Uses the efficiency
ratio (ER) — abs(direction) / sum(abs(deltas)) — to blend between a
fast and slow EMA smoothing constant. In a trend, ER approaches 1 and
smoothing tightens; in chop, ER approaches 0 and smoothing widens.

Reference: Perry Kaufman, *Trading Systems and Methods*, 5th ed. (2013).

Bit-identical numpy port of ``pandas_ta_classic.kama`` (0.9): the pandas prep
(missing-bar skip, shift/diff/flat-window guard/efficiency ratio/smoothing
constant) runs on numpy arrays and the O(n) recurrence on float64 scalars,
instead of the library's per-call Series churn. One op stays on pandas by
design: ``peer_diff.rolling(period).sum()`` -- pandas computes it with a
compensated running sum whose rounding order is not reproducible by numpy
reductions (or by a Python replica that would be slower than pandas' compiled
loop). Exactness is enforced by ``tests/unit/test_plugins/test_kama_exactness.py``
plus a one-shot runtime self-check (``_kama_port_ok``) that falls back to
``ta.kama`` with a warning if the port ever disagrees here.

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

from functools import cache
from typing import TYPE_CHECKING, Any

from vibe_quant.dsl.compute_builtins import int_param, nan_like
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry
from vibe_quant.dsl.warmup_class import windowed

if TYPE_CHECKING:
    import pandas as pd


# Kaufman's canonical fast/slow periods (pandas_ta_classic.kama defaults).
_FAST, _SLOW = 2, 30


def compute_kama(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """KAMA over ``close`` at the given period. Fast/slow periods follow
    Kaufman's canonical 2/30 defaults (as in pandas-ta-classic).

    Falls back to the pandas path for non-float64 input, periods < 1 (the
    library raises ``ValueError``) or if the one-shot self-check
    (``_kama_port_ok``) fails on this platform.
    """
    period = int_param(params, "period", 10)
    if period < 1 or df["close"].dtype != "float64" or not _kama_port_ok():
        return _kama_pandas(df, period)
    return _kama_port(df, period)


def _kama_pandas(df: pd.DataFrame, period: int) -> pd.Series:
    """Reference path: ``ta.kama`` itself (None below max(fast, slow, period)
    rows -> NaN, never 0). The runtime fallback and the zero-tolerance test reference."""
    import pandas_ta_classic as ta

    result = ta.kama(df["close"], length=period, fast=_FAST, slow=_SLOW)
    if result is None:
        return nan_like(df)  # insufficient data: not ready (NaN), never 0
    return result  # type: ignore[no-any-return]


def _kama_port(df: pd.DataFrame, period: int) -> pd.Series:
    """Numpy port of ``ta.kama``; see
    ``tests/unit/test_plugins/test_kama_exactness.py``. Any arithmetic change
    must keep that test (zero tolerance) green.
    """
    import numpy as np
    import pandas as pd

    if period < 1:
        return _kama_pandas(df, period)
    fast, slow = _FAST, _SLOW

    close = df["close"]
    full_c = close.to_numpy(dtype=np.float64)
    # skip_leading_nan(interior=True): the library computes on the finite rows
    # only, then pads the result back by position (NaN on the skipped rows).
    finite = np.isfinite(full_c)
    skipped = not finite.all()
    c = full_c[finite] if skipped else full_c
    m = c.size
    if m < max(fast, slow, period):
        return nan_like(df)

    fr = 2 / (fast + 1)
    sr = 2 / (slow + 1)
    with np.errstate(all="ignore"):
        # (close - close.shift(period)).abs() on arrays.
        abs_diff = np.empty(m, dtype=np.float64)
        abs_diff[:period] = np.nan
        np.subtract(c[period:], c[:-period], out=abs_diff[period:])
        abs_diff = np.abs(abs_diff)

        # (close - close.shift(drift)).abs() on arrays.
        peer = np.empty(m, dtype=np.float64)
        peer[0] = np.nan
        peer[1:] = c[1:] - c[:-1]
        peer = np.abs(peer)

        # Kept on pandas: its compensated running sum is not reproducible
        # bit-exactly by any numpy reduction order.
        peer_sum = pd.Series(peer).rolling(period).sum().to_numpy(dtype=np.float64)
        # In place, same IEEE ops in the same order as `x = er*(fr-sr)+sr; sc = x*x`;
        # a window with no movement has er 1.0 (TA-Lib's KAMA value).
        sc = abs_diff
        np.divide(sc, peer_sum, out=sc)
        sc[peer_sum == 0] = 1.0
        np.multiply(sc, fr - sr, out=sc)
        np.add(sc, sr, out=sc)
        np.multiply(sc, sc, out=sc)

    # The recurrence is pure in (sc, c), so it is memoized on those arrays
    # (computed AFTER the guards and the rolling sum above, so nothing upstream
    # of sc is keyed: a trimmed buffer changes the running sum and simply misses).
    from vibe_quant.dsl.prefix_memo import memo_run

    def full(inputs: Any) -> tuple[Any, float]:
        # Python floats + list indexing: ~4x faster than per-element numpy scalars.
        cl = inputs[1].tolist()
        sl = inputs[0].tolist()
        out = [float("nan")] * len(cl)
        prev = cl[period - 1]
        out[period - 1] = prev
        for i in range(period, len(cl)):
            si = sl[i]
            prev = si * cl[i] + (1.0 - si) * prev
            out[i] = prev
        return np.array(out, dtype=np.float64), prev

    # m <= period: no recurrence step yet, nothing to extend
    res = full((sc, c))[0] if m <= period else memo_run(("kama", period), (sc, c), full, _kama_step)
    if skipped:
        padded = np.full(full_c.size, np.nan)
        padded[finite] = res
        res = padded

    return pd.Series(
        res,
        index=close.index,
        name=f"KAMA_{period}_{fast}_{slow}",
    )


def _kama_step(prev: float, inputs: Any, i: int) -> tuple[float, float]:
    """One recurrence step, `s*c + (1-s)*prev` (no FMA, op-for-op with the
    reference loop). The state is the previous output. The full path inlines
    the same expression on Python floats; both must stay in lockstep."""
    si = float(inputs[0][i])
    v = si * float(inputs[1][i]) + (1.0 - si) * prev
    return v, v


@cache
def _kama_port_ok() -> bool:
    """One-shot runtime check that the numpy port is bit-identical here.

    The port reproduces the library's operation order on numpy/Python float64
    scalars; FMA contraction in a compiled layer (numba) or a pandas version
    change could shift results by ~1 ulp per step. Verify once per process and
    fall back to the pandas path (with a warning) on any mismatch.
    """
    import logging

    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(12345)
    close = 100.0 + np.cumsum(rng.randn(120) * 0.5)
    close[40:60] = close[40]  # flat stretch -> er 1.0 path
    close[60] = close[59]  # one exactly-equal consecutive pair
    close[80] = np.nan  # NaN gap
    df = pd.DataFrame({"close": close})
    ok = bool(
        np.array_equal(
            _kama_port(df, 10).to_numpy(),
            _kama_pandas(df, 10).to_numpy(),
            equal_nan=True,
        )
    )
    if not ok:
        logging.getLogger(__name__).warning(
            "numpy KAMA port differs from pandas-ta here (FMA/pandas version); "
            "falling back to the slower pandas path"
        )
    return ok


def _lookback(params: dict[str, object]) -> int:
    """Bars KAMA needs before its value is defined.

    pandas_ta rejects (warns + returns all NaN for) series shorter than
    max(fast, slow, period) = max(30, period): never call it earlier, or
    periods 5..9 (3 * period < 30) warn once per bar until bar 30.
    """
    return max(int_param(params, "period", 10) * 3, _SLOW)


indicator_registry.register_spec(
    IndicatorSpec(
        name="KAMA",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"period": 10},
        param_schema={"period": int},
        compute_fn=compute_kama,
        pta_lookback_fn=_lookback,
        paper_warmup=windowed(_lookback),
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
