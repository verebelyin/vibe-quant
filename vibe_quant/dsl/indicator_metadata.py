"""Indicator metadata for UI display: categories, descriptions, default parameters.

This module provides human-readable information about each indicator type
for the indicator catalog, condition builder, and strategy wizard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class IndicatorParam:
    """A single parameter for an indicator."""

    name: str
    label: str
    default: int | float
    min_val: int | float
    max_val: int | float
    step: int | float = 1
    description: str = ""


@dataclass(frozen=True, slots=True)
class IndicatorMeta:
    """Metadata for one indicator type."""

    type_name: str
    display_name: str
    category: str  # "Trend", "Momentum", "Volatility", "Volume"
    description: str
    use_case: str
    params: tuple[IndicatorParam, ...] = ()
    default_period: int | None = None
    source_required: bool = True
    popular: bool = False

    def get_default_params(self) -> dict[str, Any]:
        """Return dict of parameter name -> default value."""
        result: dict[str, Any] = {}
        for p in self.params:
            result[p.name] = p.default
        if self.default_period is not None:
            result.setdefault("period", self.default_period)
        return result


# ---------------------------------------------------------------------------
# Indicator catalog
# ---------------------------------------------------------------------------

INDICATOR_CATALOG: dict[str, IndicatorMeta] = {}
INDICATOR_CATEGORIES: dict[str, list[str]] = {
    "Trend": [],
    "Momentum": [],
    "Volatility": [],
    "Volume": [],
}


def _register(meta: IndicatorMeta) -> None:
    INDICATOR_CATALOG[meta.type_name] = meta
    INDICATOR_CATEGORIES[meta.category].append(meta.type_name)


# ── Trend ──────────────────────────────────────────────────────────────────

_register(IndicatorMeta(
    type_name="EMA",
    display_name="Exponential Moving Average",
    category="Trend",
    description="Weighted moving average giving more weight to recent prices. Reacts faster than SMA.",
    use_case="Trend direction, dynamic support/resistance, signal line for crossovers.",
    default_period=20,
    popular=True,
    params=(
        IndicatorParam("period", "Period", 20, 2, 500, 1, "Number of bars to average"),
    ),
))

_register(IndicatorMeta(
    type_name="SMA",
    display_name="Simple Moving Average",
    category="Trend",
    description="Equal-weighted average of last N closing prices. Smooth but lagging.",
    use_case="Trend confirmation, moving average crossover strategies.",
    default_period=20,
    params=(
        IndicatorParam("period", "Period", 20, 2, 500, 1, "Number of bars to average"),
    ),
))

_register(IndicatorMeta(
    type_name="WMA",
    display_name="Weighted Moving Average",
    category="Trend",
    description="Linearly-weighted moving average. Middle ground between SMA and EMA.",
    use_case="Smoothed trend detection with moderate lag.",
    default_period=20,
    params=(
        IndicatorParam("period", "Period", 20, 2, 500, 1, "Number of bars to average"),
    ),
))

_register(IndicatorMeta(
    type_name="DEMA",
    display_name="Double EMA",
    category="Trend",
    description="Double-smoothed EMA that reduces lag while maintaining smoothness.",
    use_case="Fast trend detection where minimal lag is critical.",
    default_period=20,
    params=(
        IndicatorParam("period", "Period", 20, 2, 500, 1, "Number of bars to average"),
    ),
))

_register(IndicatorMeta(
    type_name="TEMA",
    display_name="Triple EMA",
    category="Trend",
    description="Triple-smoothed EMA with even less lag than DEMA.",
    use_case="Ultra-responsive trend following on short timeframes.",
    default_period=20,
    params=(
        IndicatorParam("period", "Period", 20, 2, 500, 1, "Number of bars to average"),
    ),
))

# ── Momentum ───────────────────────────────────────────────────────────────

_register(IndicatorMeta(
    type_name="RSI",
    display_name="Relative Strength Index",
    category="Momentum",
    description="Oscillator (0-100) measuring speed and magnitude of price changes. "
                "Classic overbought/oversold indicator.",
    use_case="Mean reversion entries (RSI < 30 = oversold, > 70 = overbought). "
             "Divergence detection.",
    default_period=14,
    popular=True,
    params=(
        IndicatorParam("period", "Period", 14, 2, 100, 1, "Lookback period"),
    ),
))

_register(IndicatorMeta(
    type_name="MACD",
    display_name="MACD (Moving Average Convergence Divergence)",
    category="Momentum",
    description="Trend-following momentum indicator showing the relationship between "
                "two EMAs. Signal line crossovers generate entries.",
    use_case="Trend direction + momentum. MACD crosses signal line for entries. "
             "Histogram shows momentum strength.",
    default_period=None,
    popular=True,
    source_required=True,
    params=(
        IndicatorParam("fast_period", "Fast Period", 12, 2, 100, 1, "Fast EMA period"),
        IndicatorParam("slow_period", "Slow Period", 26, 5, 200, 1, "Slow EMA period"),
        IndicatorParam("signal_period", "Signal Period", 9, 2, 50, 1, "Signal line EMA period"),
    ),
))

_register(IndicatorMeta(
    type_name="STOCH",
    display_name="Stochastic Oscillator",
    category="Momentum",
    description="Compares closing price to the range over N periods. "
                "Values 0-100 with 20/80 OB/OS levels.",
    use_case="Overbought/oversold detection. Works well in ranging markets.",
    default_period=14,
    params=(
        IndicatorParam("period", "Period", 14, 2, 100, 1, "%K lookback period"),
    ),
))

_register(IndicatorMeta(
    type_name="CCI",
    display_name="Commodity Channel Index",
    category="Momentum",
    description="Measures price deviation from statistical mean. "
                "Values typically between -200 and +200.",
    use_case="Identify cyclical trends. CCI > 100 = strong uptrend, < -100 = strong downtrend.",
    default_period=20,
    params=(
        IndicatorParam("period", "Period", 20, 5, 200, 1, "Lookback period"),
    ),
))

_register(IndicatorMeta(
    type_name="WILLR",
    display_name="Williams %R",
    category="Momentum",
    description="Momentum oscillator (-100 to 0). Similar to Stochastic but inverted scale.",
    use_case="Overbought (> -20) / oversold (< -80) signals.",
    default_period=14,
    params=(
        IndicatorParam("period", "Period", 14, 2, 100, 1, "Lookback period"),
    ),
))

_register(IndicatorMeta(
    type_name="ROC",
    display_name="Rate of Change",
    category="Momentum",
    description="Percentage change between current price and N periods ago.",
    use_case="Momentum confirmation. Positive ROC = upward momentum.",
    default_period=12,
    params=(
        IndicatorParam("period", "Period", 12, 1, 200, 1, "Lookback period"),
    ),
))

# ── Volatility ─────────────────────────────────────────────────────────────

_register(IndicatorMeta(
    type_name="ATR",
    display_name="Average True Range",
    category="Volatility",
    description="Measures market volatility by averaging the true range over N periods. "
                "Essential for position sizing and stop placement.",
    use_case="Stop-loss placement (N x ATR from entry). Position sizing based on volatility. "
             "Volatility filters.",
    default_period=14,
    popular=True,
    source_required=False,
    params=(
        IndicatorParam("period", "Period", 14, 2, 100, 1, "Lookback period"),
    ),
))

_register(IndicatorMeta(
    type_name="BBANDS",
    display_name="Bollinger Bands",
    category="Volatility",
    description="Upper/lower bands at N standard deviations from SMA. "
                "Bands expand in high volatility, contract in low volatility.",
    use_case="Bollinger squeeze (low vol breakout). Mean reversion to middle band. "
             "Band-walk trend following.",
    default_period=20,
    popular=True,
    params=(
        IndicatorParam("period", "Period", 20, 5, 200, 1, "SMA lookback period"),
        IndicatorParam("std_dev", "Std Deviation", 2.0, 0.5, 4.0, 0.1,
                       "Standard deviation multiplier for bands"),
    ),
))

_register(IndicatorMeta(
    type_name="KC",
    display_name="Keltner Channel",
    category="Volatility",
    description="ATR-based envelope around EMA. More stable than Bollinger Bands.",
    use_case="Trend direction (price above/below channel). Squeeze detection when "
             "BBands move inside KC.",
    default_period=20,
    params=(
        IndicatorParam("period", "Period", 20, 5, 200, 1, "EMA lookback period"),
        IndicatorParam("atr_multiplier", "ATR Multiplier", 1.5, 0.5, 5.0, 0.1,
                       "ATR multiplier for channel width"),
    ),
))

_register(IndicatorMeta(
    type_name="DONCHIAN",
    display_name="Donchian Channel",
    category="Volatility",
    description="Highest high and lowest low over N periods. "
                "Classic breakout indicator (turtle trading).",
    use_case="Breakout entries on new highs/lows. Trend following. "
             "Stop placement at channel boundary.",
    default_period=20,
    params=(
        IndicatorParam("period", "Period", 20, 5, 200, 1, "Lookback period for high/low"),
    ),
))

# ── Volume ─────────────────────────────────────────────────────────────────

_register(IndicatorMeta(
    type_name="OBV",
    display_name="On-Balance Volume",
    category="Volume",
    description="Cumulative volume indicator: adds volume on up days, "
                "subtracts on down days. Leading indicator.",
    use_case="Confirm price trends with volume. Divergence between OBV and price "
             "signals potential reversal.",
    default_period=None,
    source_required=False,
    params=(),
))

_register(IndicatorMeta(
    type_name="VWAP",
    display_name="Volume-Weighted Average Price",
    category="Volume",
    description="Average price weighted by volume. Resets daily. "
                "Institutional benchmark.",
    use_case="Intraday support/resistance. Price above VWAP = bullish, below = bearish. "
             "Mean reversion target.",
    default_period=None,
    source_required=False,
    popular=True,
    params=(),
))

_register(IndicatorMeta(
    type_name="MFI",
    display_name="Money Flow Index",
    category="Volume",
    description="Volume-weighted RSI. Oscillator 0-100 incorporating both price and volume.",
    use_case="Like RSI but with volume confirmation. MFI < 20 = oversold, > 80 = overbought.",
    default_period=14,
    params=(
        IndicatorParam("period", "Period", 14, 2, 100, 1, "Lookback period"),
    ),
))


def get_indicators_by_category() -> dict[str, list[IndicatorMeta]]:
    """Return indicators grouped by category, ordered for display."""
    result: dict[str, list[IndicatorMeta]] = {}
    for cat in ("Trend", "Momentum", "Volatility", "Volume"):
        result[cat] = [INDICATOR_CATALOG[t] for t in INDICATOR_CATEGORIES[cat]]
    return result


def get_popular_indicators() -> list[IndicatorMeta]:
    """Return indicators marked as popular."""
    return [m for m in INDICATOR_CATALOG.values() if m.popular]


def suggest_indicator_name(type_name: str, existing_names: set[str]) -> str:
    """Suggest a unique indicator name based on type and defaults.

    Examples: rsi_14, ema_20, macd, atr_14
    """
    meta = INDICATOR_CATALOG.get(type_name)
    if not meta:
        base = type_name.lower()
    elif meta.default_period:
        base = f"{type_name.lower()}_{meta.default_period}"
    else:
        base = type_name.lower()

    if base not in existing_names:
        return base

    # Append suffix
    for i in range(2, 100):
        candidate = f"{base}_{i}"
        if candidate not in existing_names:
            return candidate
    return f"{base}_99"
