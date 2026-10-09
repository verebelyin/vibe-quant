"""AUTO-GENERATED FROM EXTRACTION 476 ON 2026-05-21T22:56:18+00:00 — review before promoting.

RANGES: period=llm
Display: Price Position in Range
Description: Normalized 0..1 score expressing where the current close sits inside its recent high-low range. Used by the author to require a stock be in the upper part of its range (>= 0.70 'core' gate, >= 0.45 'premium override') before promotion. Conceptually similar to Stochastic %K but typically computed on a longer lookback (multi-week/52-week) rather than 14 bars.

Source quote:
    Core early-runner gate: RSI 55-68, price_position >= 0.70, 5d >= -3%
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # noqa: TC002

from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry


def _price_position_pandas(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """Reference implementation (pandas rolling) — kept for equivalence tests."""
    period_raw = params.get('period', 14)
    period = int(period_raw) if isinstance(period_raw, (int, float)) else 14
    if period < 1:
        period = 1
    low: pd.Series = df['low']
    high: pd.Series = df['high']
    close: pd.Series = df['close']
    lowest_low: pd.Series = low.rolling(window=period, min_periods=period).min()
    highest_high: pd.Series = high.rolling(window=period, min_periods=period).max()
    range_span: pd.Series = highest_high - lowest_low
    numerator: pd.Series = close - lowest_low
    result: pd.Series = numerator / range_span.where(range_span != 0, other=np.nan)
    result = result.where(range_span.notna(), other=np.nan)
    result.index = df.index
    return result


def compute_price_position(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    period_raw = params.get('period', 14)
    period = int(period_raw) if isinstance(period_raw, (int, float)) else 14
    if period < 1:
        period = 1
    close = df['close'].to_numpy(dtype=np.float64)
    high = df['high'].to_numpy(dtype=np.float64)
    low = df['low'].to_numpy(dtype=np.float64)
    n = low.shape[0]
    lowest_low = np.full(n, np.nan, dtype=np.float64)
    highest_high = np.full(n, np.nan, dtype=np.float64)
    if n >= period:
        if np.isnan(high).any() or np.isnan(low).any():
            # pandas rolling with min_periods=period makes any NaN-containing
            # window NaN; scipy min/max filters propagate NaN inconsistently.
            # NaN close needs no guard: it only poisons its own row's numerator
            # on both paths.
            return _price_position_pandas(df, params)

        from scipy.ndimage import maximum_filter1d, minimum_filter1d

        # Rolling min/max are exact selections (no floating summation), so the
        # scipy windows are bit-identical to pandas rolling.
        mn = minimum_filter1d(low, size=period, origin=(period - 1) // 2)
        mx = maximum_filter1d(high, size=period, origin=(period - 1) // 2)
        # Outputs >= period - 1 have full in-bounds windows; the first
        # period - 1 values stay NaN exactly as min_periods=period does.
        lowest_low[period - 1 :] = mn[period - 1 :]
        highest_high[period - 1 :] = mx[period - 1 :]
    range_span = highest_high - lowest_low
    numerator = close - lowest_low
    result = numerator / np.where(range_span != 0, range_span, np.nan)
    result = np.where(np.isnan(range_span), np.nan, result)
    return pd.Series(result, index=df.index)


indicator_registry.register_spec(
    IndicatorSpec(
        name='PRICE_POSITION',
        nt_class=None,
        pandas_ta_func=None,
        default_params={'period': 52},
        param_schema={'period': int},
        compute_fn=compute_price_position,
        display_name='Price Position in Range',
        description="Normalized 0..1 score expressing where the current close sits inside its recent high-low range. Used by the author to require a stock be in the upper part of its range (>= 0.70 'core' gate, >= 0.45 'premium override') before promotion. Conceptually similar to Stochastic %K but typically computed on a longer lookback (multi-week/52-week) rather than 14 bars.",
        category='Custom',
        param_ranges={'period': (20.0, 252.0)},
        threshold_range=(0.2, 0.8),
    )
)
