"""DSL schema definitions using Pydantic v2 models.

Defines the structure for trading strategy DSL including indicators,
conditions, time filters, stop loss, take profit, and sweep parameters.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Valid timeframes supported by the system
VALID_TIMEFRAMES = frozenset({"1m", "5m", "15m", "1h", "4h"})

# Valid indicator types (MVP set from SPEC.md)
VALID_INDICATOR_TYPES = frozenset({
    # Trend
    "EMA",
    "SMA",
    "WMA",
    "DEMA",
    "TEMA",
    "ICHIMOKU",
    # Momentum
    "RSI",
    "MACD",
    "STOCH",
    "CCI",
    "WILLR",
    "ROC",
    # Volatility
    "ATR",
    "BBANDS",
    "KC",
    "DONCHIAN",
    # Volume
    "OBV",
    "VWAP",
    "MFI",
    "VOLSMA",
})

# Valid price sources for indicators
VALID_SOURCES = frozenset({"open", "high", "low", "close", "volume", "hl2", "hlc3", "ohlc4"})

# Valid stop loss types
VALID_STOP_LOSS_TYPES = frozenset({"fixed_pct", "atr_fixed", "atr_trailing"})

# Valid take profit types
VALID_TAKE_PROFIT_TYPES = frozenset({"fixed_pct", "atr_fixed", "risk_reward"})

# Valid days of week
VALID_DAYS = frozenset({
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
})


class IndicatorConfig(BaseModel):
    """Configuration for a single indicator.

    Attributes:
        type: Indicator type (RSI, EMA, MACD, etc.)
        period: Lookback period for the indicator
        source: Price source (close, high, low, etc.)
        timeframe: Optional timeframe override (defaults to strategy primary)
        fast_period: Fast period for MACD-like indicators
        slow_period: Slow period for MACD-like indicators
        signal_period: Signal period for MACD-like indicators
        std_dev: Standard deviation multiplier for Bollinger Bands
        atr_multiplier: ATR multiplier for ATR-based indicators
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(..., description="Indicator type (RSI, EMA, etc.)")
    period: int | None = Field(default=None, ge=1, le=2000, description="Lookback period")
    source: str = Field(default="close", description="Price source")
    timeframe: str | None = Field(default=None, description="Timeframe override")

    # MACD-specific
    fast_period: int | None = Field(default=None, ge=1, le=500)
    slow_period: int | None = Field(default=None, ge=1, le=2000)
    signal_period: int | None = Field(default=None, ge=1, le=100)

    # STOCH-specific
    d_period: int | None = Field(default=None, ge=1, le=100)

    # Bollinger Bands / Keltner
    std_dev: float | None = Field(default=None, ge=0.1, le=5.0)
    atr_multiplier: float | None = Field(default=None, ge=0.1, le=10.0)

    @field_validator("type")
    @classmethod
    def validate_indicator_type(cls, v: str) -> str:
        """Validate indicator type is supported."""
        upper_v = v.upper()
        if upper_v not in VALID_INDICATOR_TYPES:
            valid_list = ", ".join(sorted(VALID_INDICATOR_TYPES))
            msg = f"Invalid indicator type '{v}'. Valid types: {valid_list}"
            raise ValueError(msg)
        return upper_v

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        """Validate price source is valid."""
        lower_v = v.lower()
        if lower_v not in VALID_SOURCES:
            valid_list = ", ".join(sorted(VALID_SOURCES))
            msg = f"Invalid source '{v}'. Valid sources: {valid_list}"
            raise ValueError(msg)
        return lower_v

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, v: str | None) -> str | None:
        """Validate timeframe if provided."""
        if v is not None and v not in VALID_TIMEFRAMES:
            valid_list = ", ".join(sorted(VALID_TIMEFRAMES))
            msg = f"Invalid timeframe '{v}'. Valid timeframes: {valid_list}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_indicator_params(self) -> IndicatorConfig:
        """Validate indicator-specific parameters."""
        if self.type == "MACD":
            if self.fast_period is None:
                self.fast_period = 12
            if self.slow_period is None:
                self.slow_period = 26
            if self.signal_period is None:
                self.signal_period = 9
        elif self.type in {"RSI", "EMA", "SMA", "WMA", "DEMA", "TEMA", "ATR", "CCI", "ROC", "MFI", "VOLSMA"}:
            if self.period is None:
                self.period = 14  # Default period
        elif self.type == "BBANDS":
            if self.period is None:
                self.period = 20
            if self.std_dev is None:
                self.std_dev = 2.0
        elif self.type == "STOCH":
            if self.period is None:
                self.period = 14
            if self.d_period is None:
                self.d_period = 3
        elif self.type == "ICHIMOKU":
            # Ichimoku uses its own param names; period not required
            pass
        return self


class SessionConfig(BaseModel):
    """Configuration for a trading session window.

    Attributes:
        start: Session start time in HH:MM format
        end: Session end time in HH:MM format
        timezone: Timezone for the session (default UTC)
    """

    model_config = ConfigDict(extra="forbid")

    start: str = Field(..., pattern=r"^([01]\d|2[0-3]):[0-5]\d$", description="Start time HH:MM")
    end: str = Field(..., pattern=r"^([01]\d|2[0-3]):[0-5]\d$", description="End time HH:MM")
    timezone: str = Field(default="UTC", description="Timezone")

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        """Validate timezone is recognized."""
        # Common valid timezones for trading
        valid_timezones = {
            "UTC",
            "US/Eastern",
            "US/Pacific",
            "US/Central",
            "Europe/London",
            "Europe/Paris",
            "Europe/Berlin",
            "Asia/Tokyo",
            "Asia/Shanghai",
            "Asia/Singapore",
            "Asia/Hong_Kong",
            "Australia/Sydney",
        }
        if v not in valid_timezones:
            # Allow pytz-style timezones
            try:
                import zoneinfo

                zoneinfo.ZoneInfo(v)
            except (ImportError, KeyError) as e:
                msg = f"Invalid timezone '{v}'. Use IANA timezone names (e.g., 'UTC', 'US/Eastern')"
                raise ValueError(msg) from e
        return v


class FundingAvoidanceConfig(BaseModel):
    """Configuration for avoiding trades around funding settlement.

    Attributes:
        enabled: Whether to avoid trading around funding times
        minutes_before: Minutes before funding to stop entering
        minutes_after: Minutes after funding to resume entering
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False, description="Enable funding avoidance")
    minutes_before: int = Field(default=5, ge=0, le=60, description="Minutes before funding")
    minutes_after: int = Field(default=5, ge=0, le=60, description="Minutes after funding")


class TimeFilterConfig(BaseModel):
    """Configuration for time-based trading filters.

    Attributes:
        allowed_sessions: List of allowed trading sessions
        blocked_days: Days of week when trading is blocked
        avoid_around_funding: Funding settlement avoidance config
    """

    model_config = ConfigDict(extra="forbid")

    allowed_sessions: list[SessionConfig] = Field(default_factory=list)
    blocked_days: list[str] = Field(default_factory=list)
    avoid_around_funding: FundingAvoidanceConfig = Field(
        default_factory=FundingAvoidanceConfig
    )

    @field_validator("blocked_days")
    @classmethod
    def validate_blocked_days(cls, v: list[str]) -> list[str]:
        """Validate blocked days are valid day names."""
        for day in v:
            if day not in VALID_DAYS:
                valid_list = ", ".join(sorted(VALID_DAYS))
                msg = f"Invalid day '{day}'. Valid days: {valid_list}"
                raise ValueError(msg)
        return v


class StopLossConfig(BaseModel):
    """Configuration for stop loss.

    Attributes:
        type: Stop loss type (fixed_pct, atr_fixed, atr_trailing)
        percent: Percentage for fixed_pct type
        atr_multiplier: ATR multiplier for ATR-based types
        indicator: Indicator name for ATR reference
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(..., description="Stop loss type")
    percent: float | None = Field(default=None, ge=0.1, le=50.0)
    atr_multiplier: float | None = Field(default=None, ge=0.5, le=10.0)
    indicator: str | None = Field(default=None, description="ATR indicator name")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate stop loss type."""
        if v not in VALID_STOP_LOSS_TYPES:
            valid_list = ", ".join(sorted(VALID_STOP_LOSS_TYPES))
            msg = f"Invalid stop_loss type '{v}'. Valid types: {valid_list}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_stop_loss_params(self) -> StopLossConfig:
        """Validate stop loss parameters based on type."""
        if self.type == "fixed_pct":
            if self.percent is None:
                msg = "stop_loss.percent is required when type is 'fixed_pct'"
                raise ValueError(msg)
        elif self.type in {"atr_fixed", "atr_trailing"}:
            if self.atr_multiplier is None:
                msg = f"stop_loss.atr_multiplier is required when type is '{self.type}'"
                raise ValueError(msg)
            if self.indicator is None:
                msg = f"stop_loss.indicator (ATR indicator name) is required when type is '{self.type}'"
                raise ValueError(msg)
        return self


class TakeProfitConfig(BaseModel):
    """Configuration for take profit.

    Attributes:
        type: Take profit type (fixed_pct, atr_fixed, risk_reward)
        percent: Percentage for fixed_pct type
        atr_multiplier: ATR multiplier for ATR-based types
        risk_reward_ratio: Risk/reward ratio for risk_reward type
        indicator: Indicator name for ATR reference
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(..., description="Take profit type")
    percent: float | None = Field(default=None, ge=0.1, le=100.0)
    atr_multiplier: float | None = Field(default=None, ge=0.5, le=20.0)
    risk_reward_ratio: float | None = Field(default=None, ge=0.5, le=10.0)
    indicator: str | None = Field(default=None, description="ATR indicator name")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate take profit type."""
        if v not in VALID_TAKE_PROFIT_TYPES:
            valid_list = ", ".join(sorted(VALID_TAKE_PROFIT_TYPES))
            msg = f"Invalid take_profit type '{v}'. Valid types: {valid_list}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_take_profit_params(self) -> TakeProfitConfig:
        """Validate take profit parameters based on type."""
        if self.type == "fixed_pct":
            if self.percent is None:
                msg = "take_profit.percent is required when type is 'fixed_pct'"
                raise ValueError(msg)
        elif self.type == "atr_fixed":
            if self.atr_multiplier is None:
                msg = "take_profit.atr_multiplier is required when type is 'atr_fixed'"
                raise ValueError(msg)
            if self.indicator is None:
                msg = "take_profit.indicator (ATR indicator name) is required when type is 'atr_fixed'"
                raise ValueError(msg)
        elif self.type == "risk_reward" and self.risk_reward_ratio is None:
            msg = "take_profit.risk_reward_ratio is required when type is 'risk_reward'"
            raise ValueError(msg)
        return self


class PositionManagementConfig(BaseModel):
    """Configuration for position management (MVP: disabled).

    Attributes:
        scale_in: Scale-in configuration
        partial_exit: Partial exit configuration
    """

    model_config = ConfigDict(extra="forbid")

    scale_in: dict[str, bool] = Field(default_factory=lambda: {"enabled": False})
    partial_exit: dict[str, bool] = Field(default_factory=lambda: {"enabled": False})


class EntryConditions(BaseModel):
    """Entry conditions for long and short positions.

    Attributes:
        long: List of condition strings for long entry
        short: List of condition strings for short entry
    """

    model_config = ConfigDict(extra="forbid")

    long: list[str] = Field(default_factory=list, description="Long entry conditions")
    short: list[str] = Field(default_factory=list, description="Short entry conditions")

    @model_validator(mode="after")
    def validate_has_conditions(self) -> EntryConditions:
        """Ensure at least one entry condition exists."""
        if not self.long and not self.short:
            msg = "entry_conditions must have at least one long or short condition"
            raise ValueError(msg)
        return self


class ExitConditions(BaseModel):
    """Exit conditions for long and short positions.

    Attributes:
        long: List of condition strings for long exit
        short: List of condition strings for short exit
    """

    model_config = ConfigDict(extra="forbid")

    long: list[str] = Field(default_factory=list, description="Long exit conditions")
    short: list[str] = Field(default_factory=list, description="Short exit conditions")


class StrategyDSL(BaseModel):
    """Complete strategy DSL definition.

    This is the main model representing a complete trading strategy
    defined in YAML format.

    Attributes:
        name: Unique strategy name
        description: Human-readable description
        version: Strategy version number
        timeframe: Primary execution timeframe
        additional_timeframes: Additional timeframes for multi-TF strategies
        indicators: Dictionary of indicator configurations
        entry_conditions: Entry conditions for long/short
        exit_conditions: Exit conditions for long/short
        time_filters: Time-based trading filters
        stop_loss: Stop loss configuration
        take_profit: Take profit configuration
        position_management: Position management configuration
        sweep: Parameter sweep ranges for screening
    """

    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, Field(min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9_]*$")]
    description: str = Field(default="", max_length=1000)
    version: int = Field(default=1, ge=1, le=1000)

    timeframe: str = Field(..., description="Primary execution timeframe")
    additional_timeframes: list[str] = Field(
        default_factory=list, description="Additional timeframes for multi-TF"
    )

    indicators: dict[str, IndicatorConfig] = Field(
        ..., description="Indicator configurations keyed by name"
    )

    entry_conditions: EntryConditions = Field(..., description="Entry conditions")
    exit_conditions: ExitConditions = Field(
        default_factory=ExitConditions, description="Exit conditions"
    )

    time_filters: TimeFilterConfig = Field(
        default_factory=TimeFilterConfig, description="Time-based filters"
    )

    stop_loss: StopLossConfig = Field(..., description="Stop loss configuration")
    take_profit: TakeProfitConfig = Field(..., description="Take profit configuration")

    position_management: PositionManagementConfig = Field(
        default_factory=PositionManagementConfig, description="Position management"
    )

    sweep: dict[str, list[int] | list[float]] = Field(
        default_factory=dict, description="Parameter sweep ranges"
    )

    @field_validator("timeframe")
    @classmethod
    def validate_primary_timeframe(cls, v: str) -> str:
        """Validate primary timeframe."""
        if v not in VALID_TIMEFRAMES:
            valid_list = ", ".join(sorted(VALID_TIMEFRAMES))
            msg = f"Invalid timeframe '{v}'. Valid timeframes: {valid_list}"
            raise ValueError(msg)
        return v

    @field_validator("additional_timeframes")
    @classmethod
    def validate_additional_timeframes(cls, v: list[str]) -> list[str]:
        """Validate additional timeframes."""
        for tf in v:
            if tf not in VALID_TIMEFRAMES:
                valid_list = ", ".join(sorted(VALID_TIMEFRAMES))
                msg = f"Invalid additional timeframe '{tf}'. Valid timeframes: {valid_list}"
                raise ValueError(msg)
        return v

    @field_validator("indicators")
    @classmethod
    def validate_indicator_names(cls, v: dict[str, IndicatorConfig]) -> dict[str, IndicatorConfig]:
        """Validate indicator names follow conventions."""
        for name in v:
            if not name.replace("_", "").replace("0123456789", "").isalnum():
                # More permissive check
                pass
            if not name or name.startswith("_") or name[0].isdigit():
                msg = f"Invalid indicator name '{name!r}'. Must be non-empty and start with a letter."
                raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_indicator_timeframes(self) -> StrategyDSL:
        """Validate indicator timeframes are available."""
        all_timeframes = {self.timeframe} | set(self.additional_timeframes)
        for name, indicator in self.indicators.items():
            if indicator.timeframe and indicator.timeframe not in all_timeframes:
                msg = (
                    f"Indicator '{name}' references timeframe '{indicator.timeframe}' "
                    f"which is not in primary timeframe or additional_timeframes. "
                    f"Available: {sorted(all_timeframes)}"
                )
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_stop_loss_indicator(self) -> StrategyDSL:
        """Validate stop loss indicator reference exists."""
        if self.stop_loss.type in {"atr_fixed", "atr_trailing"}:
            indicator_name = self.stop_loss.indicator
            if indicator_name and indicator_name not in self.indicators:
                msg = (
                    f"stop_loss references indicator '{indicator_name}' "
                    f"which is not defined in indicators"
                )
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_take_profit_indicator(self) -> StrategyDSL:
        """Validate take profit indicator reference exists."""
        if self.take_profit.type == "atr_fixed":
            indicator_name = self.take_profit.indicator
            if indicator_name and indicator_name not in self.indicators:
                msg = (
                    f"take_profit references indicator '{indicator_name}' "
                    f"which is not defined in indicators"
                )
                raise ValueError(msg)
        return self

    def get_all_timeframes(self) -> set[str]:
        """Get all timeframes used by this strategy."""
        return {self.timeframe} | set(self.additional_timeframes)

    def get_indicator_names(self) -> set[str]:
        """Get all indicator names defined in this strategy."""
        return set(self.indicators.keys())
