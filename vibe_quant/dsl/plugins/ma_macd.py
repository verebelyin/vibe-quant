"""MA_MACD — MACD built from any TradingView moving-average kind.

``macd = ma(src, fast) - ma(src, slow)``; ``signal = ma(macd, signal_period)`` (recursing on the
macd line from its first Pine-defined bar, not from the bar where the exposed ``macd`` is ready);
``histogram = macd - signal``, all with the same ``kind``: 02_macd_sma200.pine
uses ``sma`` for all three, 14_ichimoku_hull_macd.pine uses the built-in ``hma``
(``MACD = hma(price,12) - hma(price,26)``, ``aMACD = hma(MACD, 9)``): that is Pine's
BUILT-IN ``hma``, kind ``hma_tv`` (floored root, UNVERIFIED vs TradingView output; ``hma``
rounds it as the Hull Suite script does). ``offset``
lags all three outputs like Pine ``x[offset]``. Each output is NaN until it is ready (``signal`` and
``histogram`` ``signal_period`` bars after ``macd``).

The spec sets ``requires_full_history`` because recursive kinds (ema, ...) exist
(see ``ma.py``).

Usage::

    indicators:
      mm:
        type: MA_MACD
        kind: sma
        fast_period: 12
        slow_period: 26
        signal_period: 9
    entry_conditions:
      long:
        - mm.histogram crosses_above 0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vibe_quant.dsl.compute_builtins import int_param
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry
from vibe_quant.dsl.plugins._tv_ma import (
    SOURCES,
    shift_back,
    source_array,
    str_param,
    tv_ma,
    tv_ma_raw,
    tv_ma_ready,
)

if TYPE_CHECKING:
    import pandas as pd


def compute_ma_macd(df: pd.DataFrame, params: dict[str, object]) -> dict[str, pd.Series]:
    """``macd``, ``signal`` and ``histogram`` of the ``kind`` moving average, lagged by ``offset``."""
    import pandas as pd

    kind = str_param(params, "kind", "sma")
    src = source_array(df, str_param(params, "src", "close"))
    fast = int_param(params, "fast_period", 12)
    slow = int_param(params, "slow_period", 26)
    sig = int_param(params, "signal_period", 9)
    macd = tv_ma(kind, src, fast) - tv_ma(kind, src, slow)  # exposed: NaN until ready
    # Pine's ``signal = ma(macd, n)`` recurses on the macd line from its first defined bar: feed it the
    # RAW macd (``tv_ma_raw`` of both legs) and mask only the exposed signal until ``sig`` bars after
    # the exposed macd line is ready.
    signal = tv_ma_raw(kind, tv_ma_raw(kind, src, fast) - tv_ma_raw(kind, src, slow), sig)
    finite = np.flatnonzero(np.isfinite(macd))
    signal[
        : min(
            int(finite[0]) + tv_ma_ready(kind, sig) - 1 if finite.size else signal.size, signal.size
        )
    ] = np.nan
    hist = macd - signal
    offset = int_param(params, "offset", 0)
    return {
        "macd": pd.Series(shift_back(macd, offset), index=df.index),
        "signal": pd.Series(shift_back(signal, offset), index=df.index),
        "histogram": pd.Series(shift_back(hist, offset), index=df.index),
    }


def _lookback(params: dict[str, object]) -> int:
    src = str_param(params, "src", "close")
    if src not in SOURCES:
        msg = f"MA_MACD src {src!r} unknown; valid: {', '.join(SOURCES)}"
        raise ValueError(msg)
    offset = int_param(params, "offset", 0)
    if offset < 0:
        msg = f"MA_MACD offset must be >= 0, got {offset}"
        raise ValueError(msg)
    kind = str_param(params, "kind", "sma")
    # The gate opens when the macd line is ready; ``signal`` and ``histogram`` stay NaN (not
    # stored) until ``signal_period`` more bars, exactly as in Pine.
    tv_ma_ready(kind, int_param(params, "signal_period", 9))  # validates the signal length
    return (
        max(
            tv_ma_ready(kind, int_param(params, "fast_period", 12)),
            tv_ma_ready(kind, int_param(params, "slow_period", 26)),
        )
        + offset
    )


indicator_registry.register_spec(
    IndicatorSpec(
        name="MA_MACD",
        nt_class=None,
        pandas_ta_func=None,
        default_params={
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "kind": "sma",
            "src": "close",
            "offset": 0,
        },
        param_schema={
            "fast_period": int,
            "slow_period": int,
            "signal_period": int,
            "kind": str,
            "src": str,
            "offset": int,
        },
        compute_fn=compute_ma_macd,
        pta_lookback_fn=_lookback,
        output_names=("macd", "signal", "histogram"),
        primary_output="macd",
        requires_high_low=True,
        requires_full_history=True,
        display_name="MACD (any MA kind)",
        description=(
            "MACD whose fast, slow and signal lines use one TradingView moving-average kind "
            "(sma, ema, hma, ...), optionally lagged."
        ),
        category="Momentum",
        chart_placement="oscillator",
    )
)
