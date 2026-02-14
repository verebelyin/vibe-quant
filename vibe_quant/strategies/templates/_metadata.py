"""Template metadata for the strategy template selector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_TEMPLATE_DIR = Path(__file__).parent

VALID_CATEGORIES: frozenset[str] = frozenset({
    "Momentum", "Trend", "Volatility", "Multi-Timeframe", "Volume",
})
VALID_DIFFICULTIES: frozenset[str] = frozenset({
    "Beginner", "Intermediate", "Advanced",
})


@dataclass(frozen=True, slots=True)
class TemplateMeta:
    """Metadata for a strategy template."""

    file_name: str
    display_name: str
    category: str
    difficulty: str
    description: str
    market_conditions: str
    instruments: str
    tags: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.category not in VALID_CATEGORIES:
            raise ValueError(f"Invalid category '{self.category}', must be one of {sorted(VALID_CATEGORIES)}")
        if self.difficulty not in VALID_DIFFICULTIES:
            raise ValueError(f"Invalid difficulty '{self.difficulty}', must be one of {sorted(VALID_DIFFICULTIES)}")

    @property
    def path(self) -> Path:
        return _TEMPLATE_DIR / self.file_name

    def load_yaml(self) -> str:
        """Load the template YAML content as a string."""
        return self.path.read_text()

    def load_dict(self) -> dict[str, object]:
        """Load the template as a parsed dict.

        Raises:
            FileNotFoundError: If the template file does not exist.
            yaml.YAMLError: If the YAML content is malformed.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Template file not found: {self.path}")
        text = self.path.read_text()
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise yaml.YAMLError(
                f"Malformed YAML in template {self.file_name}: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise yaml.YAMLError(
                f"Template {self.file_name} must be a YAML mapping, got {type(data).__name__}"
            )
        return data


# ── Template registry ──────────────────────────────────────────────────────

TEMPLATES: list[TemplateMeta] = [
    # -- Momentum --
    TemplateMeta(
        file_name="rsi_mean_reversion.yaml",
        display_name="RSI Mean Reversion",
        category="Momentum",
        difficulty="Beginner",
        description="Buy oversold (RSI < 30), sell overbought (RSI > 70). "
                    "Simple mean reversion with ATR trailing stops.",
        market_conditions="Ranging / sideways markets with clear support and resistance.",
        instruments="BTC, ETH, SOL - high liquidity pairs",
        tags=("RSI", "mean-reversion", "beginner-friendly"),
    ),
    TemplateMeta(
        file_name="rsi_divergence.yaml",
        display_name="RSI Divergence",
        category="Momentum",
        difficulty="Intermediate",
        description="Detects when price makes new lows but RSI makes higher lows "
                    "(bullish divergence) using EMA trend filter.",
        market_conditions="Trending markets approaching exhaustion points.",
        instruments="BTC, ETH - major pairs with clear trend structure",
        tags=("RSI", "divergence", "trend-reversal"),
    ),
    TemplateMeta(
        file_name="macd_crossover.yaml",
        display_name="MACD Crossover",
        category="Momentum",
        difficulty="Beginner",
        description="Enter when MACD crosses signal line with EMA trend confirmation. "
                    "Classic momentum strategy with session filters.",
        market_conditions="Trending markets with sustained directional moves.",
        instruments="All major perpetuals",
        tags=("MACD", "crossover", "trend-following"),
    ),
    TemplateMeta(
        file_name="stochastic_reversal.yaml",
        display_name="Stochastic Reversal",
        category="Momentum",
        difficulty="Beginner",
        description="Stochastic oscillator overbought/oversold strategy with "
                    "EMA filter to trade only in the direction of the trend.",
        market_conditions="Ranging markets with trend filter for direction.",
        instruments="BTC, ETH, BNB",
        tags=("stochastic", "mean-reversion", "filtered"),
    ),
    # -- Trend --
    TemplateMeta(
        file_name="ema_ribbon.yaml",
        display_name="EMA Ribbon Trend",
        category="Trend",
        difficulty="Intermediate",
        description="Three EMAs (fast/medium/slow) forming a ribbon. Enter when all "
                    "three align in order. Strong trend confirmation.",
        market_conditions="Trending markets. Avoids choppy sideways action.",
        instruments="All major perpetuals",
        tags=("EMA", "ribbon", "trend-following", "multi-indicator"),
    ),
    TemplateMeta(
        file_name="donchian_breakout.yaml",
        display_name="Donchian Channel Breakout",
        category="Trend",
        difficulty="Beginner",
        description="Classic turtle trading: enter on N-period high/low breakout. "
                    "Donchian channel with ATR-based stops.",
        market_conditions="Volatile markets with strong breakout potential.",
        instruments="BTC, ETH, SOL - volatile large-cap pairs",
        tags=("Donchian", "breakout", "turtle-trading"),
    ),
    TemplateMeta(
        file_name="dual_ema_crossover.yaml",
        display_name="Dual EMA Crossover",
        category="Trend",
        difficulty="Beginner",
        description="Fast EMA crosses slow EMA for entry. The simplest possible "
                    "trend-following strategy. Good starting point.",
        market_conditions="Any trending market. Whipsaws in ranges.",
        instruments="All major perpetuals",
        tags=("EMA", "crossover", "simple", "beginner-friendly"),
    ),
    # -- Volatility --
    TemplateMeta(
        file_name="bollinger_squeeze.yaml",
        display_name="Bollinger Band Squeeze",
        category="Volatility",
        difficulty="Intermediate",
        description="Enter when Bollinger Bands contract (low volatility) then "
                    "expand. RSI confirms direction. Captures volatility breakouts.",
        market_conditions="Markets transitioning from low to high volatility.",
        instruments="BTC, ETH - pairs with clear vol cycles",
        tags=("BBANDS", "squeeze", "volatility-breakout"),
    ),
    TemplateMeta(
        file_name="keltner_trend.yaml",
        display_name="Keltner Channel Trend",
        category="Volatility",
        difficulty="Intermediate",
        description="Trade in the direction of Keltner Channel breakouts. "
                    "More stable than Bollinger Bands due to ATR-based width.",
        market_conditions="Trending markets with sustained momentum.",
        instruments="All major perpetuals",
        tags=("KC", "trend", "ATR-based"),
    ),
    # -- Multi-Timeframe --
    TemplateMeta(
        file_name="mtf_trend_scalp.yaml",
        display_name="Multi-TF Trend Scalper",
        category="Multi-Timeframe",
        difficulty="Advanced",
        description="Uses 1h and 4h EMA for trend direction, enters on 5m RSI "
                    "oversold/overbought aligned with higher TF trend.",
        market_conditions="Strong trending markets on higher timeframes.",
        instruments="BTC, ETH - high liquidity for fast TF entries",
        tags=("multi-timeframe", "scalping", "RSI", "EMA"),
    ),
    TemplateMeta(
        file_name="mtf_momentum_filter.yaml",
        display_name="Multi-TF Momentum Filter",
        category="Multi-Timeframe",
        difficulty="Advanced",
        description="MACD on 1h for momentum direction, EMA on 4h for trend, "
                    "RSI on 15m for precise entries. Triple confirmation.",
        market_conditions="Strongly trending markets with clear momentum.",
        instruments="BTC, ETH",
        tags=("multi-timeframe", "MACD", "RSI", "triple-confirmation"),
    ),
    TemplateMeta(
        file_name="vwap_intraday.yaml",
        display_name="VWAP Intraday",
        category="Volume",
        difficulty="Intermediate",
        description="Trade mean reversion to VWAP with RSI confirmation. "
                    "Price below VWAP + oversold RSI = long.",
        market_conditions="Intraday ranging around VWAP. Best for high-volume pairs.",
        instruments="BTC, ETH - highest volume pairs",
        tags=("VWAP", "intraday", "mean-reversion", "volume"),
    ),
]


TEMPLATE_CATEGORIES = ["Momentum", "Trend", "Volatility", "Multi-Timeframe", "Volume"]


def get_templates_by_category() -> dict[str, list[TemplateMeta]]:
    """Return templates grouped by category."""
    result: dict[str, list[TemplateMeta]] = {}
    for cat in TEMPLATE_CATEGORIES:
        result[cat] = [t for t in TEMPLATES if t.category == cat]
    # Only include non-empty categories
    return {k: v for k, v in result.items() if v}


