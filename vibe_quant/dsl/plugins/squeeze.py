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
  ``alpha = 1 / length``, ``adjust=False``, SMA-seeded at bar
  ``length - 1`` (seed = mean of the first ``length`` true ranges).
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

from typing import TYPE_CHECKING, cast

from vibe_quant.dsl.compute_builtins import float_param, int_param, nan_like
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry

if TYPE_CHECKING:
    import pandas as pd


def compute_squeeze_ratio(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """BB width / KC width over ``close``.

    ``length`` = rolling window (default 20), ``bb_mult`` = Bollinger
    deviation multiplier (2.0), ``kc_mult`` = Keltner ATR multiplier
    (1.5). Population std (ddof=0), Wilder ATR via pandas_ta_classic.atr.
    """
    import pandas as pd
    import pandas_ta_classic as ta

    length = int_param(params, "length", 20)
    bb_mult = float_param(params, "bb_mult", 2.0)
    kc_mult = float_param(params, "kc_mult", 1.5)
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
