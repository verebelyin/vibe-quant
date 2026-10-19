"""SQUEEZE_RATIO / SQUEEZE_MOM — John Carter's TTM Squeeze family.

The TTM Squeeze flags volatility contraction: Bollinger Bands (SMA basis,
``bb_mult`` deviations) pulled inside Keltner Channels (EMA basis,
``kc_mult`` Wilder ATRs). This module exposes the family as two
single-output indicators:

- ``SQUEEZE_RATIO`` = BB width / KC width. Widths are the full channel
  widths — ``2 * bb_mult * sd`` and ``2 * kc_mult * ATR`` — so the basis
  (and any zero-crossing of the midline) cancels out and the result is
  dimensionless. Squeeze ON (contraction) when < 1. NaN while warm or
  when the KC width is 0 (no division by zero).
- ``SQUEEZE_MOM`` = linreg endpoint of the squeeze momentum source
  ``close - (donchian_mid + SMA(close, length)) / 2``, normalised by close
  (``value / close * 100``). The normalisation is deliberate and NOT in
  Carter's original: the raw source is price-scaled, so a GA threshold
  would not transfer across symbols/prices; expressed as a percent of
  close the threshold is a scale-free fraction (``threshold_range``
  ±5%). Positive = upward momentum, negative = downward.

Formula notes (matching ``pandas_ta_classic``):

- ``sd`` is the **population** rolling standard deviation (``ddof=0``) —
  the default of ``pandas_ta_classic.bbands`` (``ddof: int = 0``).
- ``ATR`` is ``pandas_ta_classic.atr``: Wilder's RMA of true range,
  ``alpha = 1 / length``, ``adjust=False``, SMA-seeded at bar ``length``
  (seed = mean of the first ``length`` true ranges; true range has none at
  bar 0, so the seed window starts at bar 1 -- pandas-ta-classic >= 0.8).
  KC midline (EMA) cancels in the width ratio and is not computed.
- ``SQUEEZE_MOM`` uses ``pandas_ta_classic.linreg``'s default output —
  the fitted value at the end of each window (``m * (length - 1) + b``).
  The source's leading ``length - 1`` NaNs make the first valid momentum
  bar ``2 * length - 2``.

Every window is causal: the value at bar t uses bars <= t only.

Usage::

    indicators:
      squeeze_ratio:
        type: SQUEEZE_RATIO
        length: 20
      squeeze_mom:
        type: SQUEEZE_MOM
        length: 20
    entry_conditions:
      long:
        - squeeze_ratio < 1
        - squeeze_mom > 0
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, cast

from vibe_quant.dsl.compute_builtins import (
    _rma_np,
    float_param,
    int_param,
    nan_like,
)
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry

if TYPE_CHECKING:
    import pandas as pd


def _squeeze_ratio_library(
    df: pd.DataFrame, length: int, bb_mult: float, kc_mult: float
) -> pd.Series:
    """Reference path: pandas rolling std + ``pandas_ta_classic.atr``."""
    import pandas as pd
    import pandas_ta_classic as ta

    close = df["close"]
    # Population std (ddof=0) — pandas_ta_classic.bbands' default.
    bb_width = 2.0 * bb_mult * close.rolling(length).std(ddof=0)

    # Wilder ATR: rma(true_range, length), alpha=1/length, adjust=False,
    # SMA-seeded — exactly pandas_ta_classic.atr.
    atr = ta.atr(df["high"], df["low"], close, length=length)
    if atr is None:
        return nan_like(df)  # insufficient data: not ready (NaN), never 0
    kc_width = 2.0 * kc_mult * atr

    # NaN when the KC width is 0 (flat market / no true range yet).
    ratio = bb_width / kc_width.where(kc_width != 0.0)
    return cast(
        "pd.Series",
        pd.Series(ratio, index=df.index, name=f"SQUEEZE_RATIO_{length}"),
    )


def _squeeze_ratio_port(
    df: pd.DataFrame, length: int, bb_mult: float, kc_mult: float
) -> pd.Series:
    """Numpy port of :func:`_squeeze_ratio_library` (same float operations).

    Bollinger width keeps pandas' own ``rolling(length).std(ddof=0)``; the
    Keltner width replaces ``pandas_ta_classic.atr`` with its exact parts —
    ``true_range`` (non-zero range epsilon, skipna row max, blanked warmup)
    fed to ``_rma_np``, the already-verified ``pandas_ta_classic.rma`` port.
    ``tests/unit/test_indicator_fast_paths.py`` is the zero-tolerance
    reference. Assumes ``length > 0`` and float64 columns (the public entry point routes
    everything else to the library).
    """
    import numpy as np
    import pandas as pd

    n = len(df)
    if n < length:
        return nan_like(df)  # library's verify_series returns None here
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)

    eps = float(np.finfo(np.float64).eps)
    high_low = high - low
    high_low = np.where(high_low != 0.0, high_low, eps)
    prev_close = np.empty(n, dtype=np.float64)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]
    # true_range: abs of the three ranges, row max skipping NaN, warmup blanked
    true_range = np.fmax(
        np.fmax(np.abs(high_low), np.abs(high - prev_close)),
        np.abs(prev_close - low),
    )
    finite = ~(np.isnan(high) | np.isnan(low) | np.isnan(close))
    start = int(finite.argmax()) if finite.any() else n
    true_range[: start + 1] = np.nan
    atr = _rma_np(true_range, length)

    bb_width = 2.0 * bb_mult * df["close"].rolling(length).std(ddof=0).to_numpy()
    kc_width = 2.0 * kc_mult * atr
    with np.errstate(all="ignore"):
        ratio = bb_width / kc_width
    ratio = np.where(kc_width != 0.0, ratio, np.nan)
    return pd.Series(ratio, index=df.index, name=f"SQUEEZE_RATIO_{length}")


def _squeeze_ratio_fast_ok(df: pd.DataFrame) -> bool:
    """True when the port's float64 arithmetic is exactly the library's own."""
    if not df.index.is_unique:  # Series alignment is label-based
        return False
    return all(df[col].dtype == "float64" for col in ("high", "low", "close"))


@cache
def _squeeze_port_ok() -> bool:
    """One-shot runtime check that the port is bit-identical on this platform.

    ``_rma_np`` relies on pandas' compiled ``ewm`` contracting the blend into
    an FMA (true on pandas 3.0.3 arm64 wheels); other platforms or versions may
    differ by ~1 ulp per step, so verify once per process and fall back to the
    library path (with a warning) on any mismatch.
    """
    import logging

    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(12345)
    close = 100.0 + np.cumsum(rng.randn(120) * 0.5)
    high = close + np.abs(rng.randn(120)) * 0.4
    low = close - np.abs(rng.randn(120)) * 0.4
    high[40:60] = low[40:60] = close[40:60] = close[40]  # flat stretch
    df = pd.DataFrame({"high": high, "low": low, "close": close})
    ok = all(
        np.array_equal(
            _squeeze_ratio_port(df, length, 2.0, 1.5).to_numpy(),
            _squeeze_ratio_library(df, length, 2.0, 1.5).to_numpy(),
            equal_nan=True,
        )
        for length in (10, 20)
    )
    if not ok:
        logging.getLogger(__name__).warning(
            "numpy SQUEEZE_RATIO port differs from pandas-ta here (FMA/pandas "
            "version); falling back to the slower library path"
        )
    return ok


def compute_squeeze_ratio(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """BB width / KC width over ``close``.

    ``length`` = rolling window (default 20), ``bb_mult`` = Bollinger
    deviation multiplier (2.0), ``kc_mult`` = Keltner ATR multiplier
    (1.5). Population std (ddof=0), Wilder ATR via pandas_ta_classic.atr
    (numpy fast path; the library remains the fallback).
    """
    length = int_param(params, "length", 20)
    bb_mult = float_param(params, "bb_mult", 2.0)
    kc_mult = float_param(params, "kc_mult", 1.5)
    if length > 0 and _squeeze_ratio_fast_ok(df) and _squeeze_port_ok():
        return _squeeze_ratio_port(df, length, bb_mult, kc_mult)
    return _squeeze_ratio_library(df, length, bb_mult, kc_mult)


def compute_squeeze_mom(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """Squeeze momentum, in percent of close.

    Source = ``close - (donchian_mid + SMA(close, length)) / 2`` with
    ``donchian_mid = (rolling_max(high, length) + rolling_min(low, length)) / 2``;
    value = linreg endpoint over ``length`` bars (pandas_ta_classic.linreg
    default), divided by close and scaled to percent.
    """
    import pandas as pd
    import pandas_ta_classic as ta

    length = int_param(params, "length", 20)
    close = df["close"]

    donchian_mid = (
        df["high"].rolling(length).max() + df["low"].rolling(length).min()
    ) / 2.0
    sma = close.rolling(length).mean()
    src = close - (donchian_mid + sma) / 2.0

    # Default linreg output = fitted value at the end of the window.
    # Windows touching the source's leading NaNs stay NaN.
    linreg = ta.linreg(src, length=length)
    if linreg is None:
        return nan_like(df)  # insufficient data: not ready (NaN), never 0

    # Scale-free threshold: momentum as a fraction of price, in percent.
    pct = linreg / close * 100.0
    return cast(
        "pd.Series",
        pd.Series(pct, index=df.index, name=f"SQUEEZE_MOM_{length}"),
    )


indicator_registry.register_spec(
    IndicatorSpec(
        name="SQUEEZE_RATIO",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"length": 20, "bb_mult": 2.0, "kc_mult": 1.5},
        param_schema={"length": int, "bb_mult": float, "kc_mult": float},
        compute_fn=compute_squeeze_ratio,
        # Wilder alpha=1/length decays ~2x slower than an EMA of the same span,
        # so a 2*length buffer keeps ATR trim drift negligible.
        pta_lookback_fn=lambda p: 2 * int_param(p, "length", 20),
        display_name="TTM Squeeze Ratio",
        description=(
            "John Carter TTM Squeeze: Bollinger width / Keltner width. "
            "Below 1 = volatility contraction (squeeze ON)."
        ),
        category="Volatility",
        chart_placement="oscillator",
        param_ranges={
            "length": (10.0, 40.0),
            "bb_mult": (1.5, 2.5),
            "kc_mult": (1.0, 2.0),
        },
        threshold_range=(0.3, 2.0),
        requires_high_low=True,
    )
)

indicator_registry.register_spec(
    IndicatorSpec(
        name="SQUEEZE_MOM",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"length": 20},
        param_schema={"length": int},
        compute_fn=compute_squeeze_mom,
        pta_lookback_fn=lambda p: 2 * int_param(p, "length", 20) - 1,
        display_name="TTM Squeeze Momentum (%)",
        description=(
            "Linreg endpoint of close minus the Donchian/SMA midline, "
            "in percent of close. Sign gives the squeeze break direction."
        ),
        category="Momentum",
        chart_placement="oscillator",
        param_ranges={"length": (10.0, 40.0)},
        threshold_range=(-5.0, 5.0),
        requires_high_low=True,
    )
)
