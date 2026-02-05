"""Tests for DSL parser and validator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vibe_quant.dsl import (
    Condition,
    ConditionParseError,
    DSLParseError,
    DSLValidationError,
    IndicatorConfig,
    Operand,
    Operator,
    StrategyDSL,
    extract_indicator_refs,
    get_referenced_indicators,
    get_required_timeframes,
    parse_condition,
    parse_strategy,
    parse_strategy_string,
    strategy_to_dict,
    strategy_to_yaml,
    validate_conditions,
    validate_strategy_dict,
)

if TYPE_CHECKING:
    from pathlib import Path
    pass


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def minimal_strategy_yaml() -> str:
    """Minimal valid strategy YAML."""
    return """
name: test_strategy
timeframe: 5m
indicators:
  rsi:
    type: RSI
    period: 14
entry_conditions:
  long:
    - rsi < 30
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""


@pytest.fixture
def full_strategy_yaml() -> str:
    """Full strategy YAML with all features."""
    return """
name: rsi_mean_reversion_mtf
description: "Enter on RSI extremes with EMA trend confirmation"
version: 1
timeframe: 5m
additional_timeframes:
  - 1h
  - 4h
indicators:
  rsi:
    type: RSI
    period: 14
    source: close
    timeframe: 5m
  rsi_1h:
    type: RSI
    period: 14
    timeframe: 1h
  ema_trend:
    type: EMA
    period: 50
    timeframe: 4h
  atr:
    type: ATR
    period: 14
entry_conditions:
  long:
    - rsi < 30
    - rsi_1h > 40
    - close > ema_trend
  short:
    - rsi > 70
    - rsi_1h < 60
    - close < ema_trend
exit_conditions:
  long:
    - rsi > 50
  short:
    - rsi < 50
time_filters:
  allowed_sessions:
    - start: "08:00"
      end: "20:00"
      timezone: UTC
  blocked_days: []
  avoid_around_funding:
    enabled: true
    minutes_before: 5
    minutes_after: 5
stop_loss:
  type: atr_trailing
  atr_multiplier: 2.0
  indicator: atr
take_profit:
  type: fixed_pct
  percent: 3.0
sweep:
  rsi.period:
    - 7
    - 14
    - 21
  stop_loss.atr_multiplier:
    - 1.5
    - 2.0
    - 2.5
"""


# =============================================================================
# Condition Parser Tests
# =============================================================================


class TestOperandParsing:
    """Tests for Operand.parse()."""

    def test_parse_numeric_integer(self) -> None:
        """Parse integer value."""
        op = Operand.parse("30")
        assert op.value == 30.0
        assert not op.is_indicator
        assert not op.is_price

    def test_parse_numeric_float(self) -> None:
        """Parse float value."""
        op = Operand.parse("3.5")
        assert op.value == 3.5
        assert not op.is_indicator
        assert not op.is_price

    def test_parse_price_close(self) -> None:
        """Parse price reference."""
        op = Operand.parse("close")
        assert op.value == "close"
        assert not op.is_indicator
        assert op.is_price

    def test_parse_price_references(self) -> None:
        """Parse all price references."""
        for price in ["close", "open", "high", "low", "volume"]:
            op = Operand.parse(price)
            assert op.is_price
            assert op.value == price

    def test_parse_indicator_reference(self) -> None:
        """Parse indicator reference."""
        op = Operand.parse("rsi", valid_indicators=["rsi", "ema"])
        assert op.value == "rsi"
        assert op.is_indicator
        assert not op.is_price

    def test_parse_unknown_indicator_raises(self) -> None:
        """Unknown indicator should raise error."""
        with pytest.raises(ValueError, match="Unknown indicator 'unknown'"):
            Operand.parse("unknown", valid_indicators=["rsi", "ema"])


class TestConditionParsing:
    """Tests for parse_condition()."""

    def test_parse_less_than(self) -> None:
        """Parse less than condition."""
        cond = parse_condition("rsi < 30", ["rsi"])
        assert cond.left.value == "rsi"
        assert cond.operator == Operator.LT
        assert cond.right.value == 30.0
        assert cond.raw == "rsi < 30"

    def test_parse_greater_than(self) -> None:
        """Parse greater than condition."""
        cond = parse_condition("rsi > 70", ["rsi"])
        assert cond.operator == Operator.GT

    def test_parse_greater_equal(self) -> None:
        """Parse greater or equal condition."""
        cond = parse_condition("rsi >= 50", ["rsi"])
        assert cond.operator == Operator.GTE

    def test_parse_less_equal(self) -> None:
        """Parse less or equal condition."""
        cond = parse_condition("rsi <= 50", ["rsi"])
        assert cond.operator == Operator.LTE

    def test_parse_crosses_above(self) -> None:
        """Parse crosses_above condition."""
        cond = parse_condition("ema_fast crosses_above ema_slow", ["ema_fast", "ema_slow"])
        assert cond.operator == Operator.CROSSES_ABOVE
        assert cond.left.value == "ema_fast"
        assert cond.right.value == "ema_slow"

    def test_parse_crosses_below(self) -> None:
        """Parse crosses_below condition."""
        cond = parse_condition("ema_fast crosses_below ema_slow", ["ema_fast", "ema_slow"])
        assert cond.operator == Operator.CROSSES_BELOW

    def test_parse_between(self) -> None:
        """Parse between condition."""
        cond = parse_condition("rsi between 30 70", ["rsi"])
        assert cond.operator == Operator.BETWEEN
        assert cond.left.value == "rsi"
        assert cond.right.value == 30.0
        assert cond.right2 is not None
        assert cond.right2.value == 70.0

    def test_parse_close_vs_indicator(self) -> None:
        """Parse close vs indicator condition."""
        cond = parse_condition("close > ema_trend", ["ema_trend"])
        assert cond.left.value == "close"
        assert cond.left.is_price
        assert cond.right.value == "ema_trend"
        assert cond.right.is_indicator

    def test_parse_indicator_vs_indicator(self) -> None:
        """Parse indicator vs indicator condition."""
        cond = parse_condition("ema_fast > ema_slow", ["ema_fast", "ema_slow"])
        assert cond.left.is_indicator
        assert cond.right.is_indicator

    def test_parse_with_whitespace(self) -> None:
        """Parse condition with extra whitespace."""
        cond = parse_condition("  rsi   <   30  ", ["rsi"])
        assert cond.left.value == "rsi"
        assert cond.right.value == 30.0

    def test_parse_empty_string_raises(self) -> None:
        """Empty condition raises error."""
        with pytest.raises(ConditionParseError, match="Empty condition string"):
            parse_condition("")

    def test_parse_invalid_format_raises(self) -> None:
        """Invalid format raises error."""
        with pytest.raises(ConditionParseError, match="Invalid condition format"):
            parse_condition("rsi something 30", ["rsi"])

    def test_parse_unknown_indicator_raises(self) -> None:
        """Unknown indicator raises error."""
        with pytest.raises(ConditionParseError, match="Unknown indicator 'unknown'"):
            parse_condition("unknown < 30", ["rsi"])

    def test_between_invalid_range_raises(self) -> None:
        """BETWEEN with low >= high raises error."""
        with pytest.raises(ConditionParseError, match="BETWEEN requires low"):
            parse_condition("rsi between 70 30", ["rsi"])

    def test_get_indicator_refs(self) -> None:
        """Get indicator references from condition."""
        cond = parse_condition("ema_fast crosses_above ema_slow", ["ema_fast", "ema_slow"])
        refs = cond.get_indicator_refs()
        assert refs == {"ema_fast", "ema_slow"}


class TestValidateConditions:
    """Tests for validate_conditions()."""

    def test_validate_multiple_conditions(self) -> None:
        """Validate list of conditions."""
        conditions = ["rsi < 30", "close > ema"]
        result = validate_conditions(conditions, ["rsi", "ema"])
        assert len(result) == 2
        assert all(isinstance(c, Condition) for c in result)

    def test_validate_invalid_condition_raises(self) -> None:
        """Invalid condition in list raises error."""
        conditions = ["rsi < 30", "bad condition"]
        with pytest.raises(ConditionParseError):
            validate_conditions(conditions, ["rsi"])


class TestExtractIndicatorRefs:
    """Tests for extract_indicator_refs()."""

    def test_extract_refs_from_conditions(self) -> None:
        """Extract indicator refs from conditions."""
        conditions = ["rsi < 30", "close > ema_trend", "macd crosses_above signal"]
        refs = extract_indicator_refs(conditions)
        assert "rsi" in refs
        assert "ema_trend" in refs
        assert "macd" in refs
        assert "signal" in refs
        assert "close" not in refs  # Price ref, not indicator


# =============================================================================
# Schema Tests
# =============================================================================


class TestIndicatorConfig:
    """Tests for IndicatorConfig."""

    def test_valid_rsi_indicator(self) -> None:
        """Create valid RSI indicator."""
        config = IndicatorConfig(type="RSI", period=14, source="close")
        assert config.type == "RSI"
        assert config.period == 14
        assert config.source == "close"

    def test_indicator_type_normalized(self) -> None:
        """Indicator type should be uppercased."""
        config = IndicatorConfig(type="rsi", period=14)
        assert config.type == "RSI"

    def test_source_normalized(self) -> None:
        """Source should be lowercased."""
        config = IndicatorConfig(type="RSI", period=14, source="CLOSE")
        assert config.source == "close"

    def test_invalid_indicator_type_raises(self) -> None:
        """Invalid indicator type raises error."""
        with pytest.raises(ValueError, match="Invalid indicator type"):
            IndicatorConfig(type="INVALID", period=14)

    def test_invalid_source_raises(self) -> None:
        """Invalid source raises error."""
        with pytest.raises(ValueError, match="Invalid source"):
            IndicatorConfig(type="RSI", period=14, source="invalid")

    def test_invalid_timeframe_raises(self) -> None:
        """Invalid timeframe raises error."""
        with pytest.raises(ValueError, match="Invalid timeframe"):
            IndicatorConfig(type="RSI", period=14, timeframe="2m")

    def test_macd_defaults(self) -> None:
        """MACD gets default periods."""
        config = IndicatorConfig(type="MACD")
        assert config.fast_period == 12
        assert config.slow_period == 26
        assert config.signal_period == 9

    def test_bbands_defaults(self) -> None:
        """Bollinger Bands gets defaults."""
        config = IndicatorConfig(type="BBANDS")
        assert config.period == 20
        assert config.std_dev == 2.0


class TestStrategyDSL:
    """Tests for StrategyDSL model."""

    def test_minimal_strategy(self) -> None:
        """Create minimal valid strategy."""
        data = {
            "name": "test_strategy",
            "timeframe": "5m",
            "indicators": {"rsi": {"type": "RSI", "period": 14}},
            "entry_conditions": {"long": ["rsi < 30"]},
            "stop_loss": {"type": "fixed_pct", "percent": 2.0},
            "take_profit": {"type": "fixed_pct", "percent": 3.0},
        }
        strategy = StrategyDSL.model_validate(data)
        assert strategy.name == "test_strategy"
        assert strategy.timeframe == "5m"
        assert len(strategy.indicators) == 1

    def test_strategy_name_validation(self) -> None:
        """Strategy name must follow pattern."""
        data = {
            "name": "Invalid Name",  # Has space and capital
            "timeframe": "5m",
            "indicators": {"rsi": {"type": "RSI", "period": 14}},
            "entry_conditions": {"long": ["rsi < 30"]},
            "stop_loss": {"type": "fixed_pct", "percent": 2.0},
            "take_profit": {"type": "fixed_pct", "percent": 3.0},
        }
        with pytest.raises(ValueError):
            StrategyDSL.model_validate(data)

    def test_invalid_primary_timeframe_raises(self) -> None:
        """Invalid primary timeframe raises error."""
        data = {
            "name": "test_strategy",
            "timeframe": "2m",  # Invalid
            "indicators": {"rsi": {"type": "RSI", "period": 14}},
            "entry_conditions": {"long": ["rsi < 30"]},
            "stop_loss": {"type": "fixed_pct", "percent": 2.0},
            "take_profit": {"type": "fixed_pct", "percent": 3.0},
        }
        with pytest.raises(ValueError, match="Invalid timeframe"):
            StrategyDSL.model_validate(data)

    def test_indicator_timeframe_must_be_available(self) -> None:
        """Indicator timeframe must be in available timeframes."""
        data = {
            "name": "test_strategy",
            "timeframe": "5m",
            "indicators": {"rsi": {"type": "RSI", "period": 14, "timeframe": "1h"}},
            "entry_conditions": {"long": ["rsi < 30"]},
            "stop_loss": {"type": "fixed_pct", "percent": 2.0},
            "take_profit": {"type": "fixed_pct", "percent": 3.0},
        }
        with pytest.raises(ValueError, match="not in primary timeframe or additional_timeframes"):
            StrategyDSL.model_validate(data)

    def test_stop_loss_indicator_must_exist(self) -> None:
        """Stop loss indicator reference must exist."""
        data = {
            "name": "test_strategy",
            "timeframe": "5m",
            "indicators": {"rsi": {"type": "RSI", "period": 14}},
            "entry_conditions": {"long": ["rsi < 30"]},
            "stop_loss": {"type": "atr_trailing", "atr_multiplier": 2.0, "indicator": "atr"},
            "take_profit": {"type": "fixed_pct", "percent": 3.0},
        }
        with pytest.raises(ValueError, match="stop_loss references indicator 'atr'"):
            StrategyDSL.model_validate(data)

    def test_entry_conditions_required(self) -> None:
        """Must have at least one entry condition."""
        data = {
            "name": "test_strategy",
            "timeframe": "5m",
            "indicators": {"rsi": {"type": "RSI", "period": 14}},
            "entry_conditions": {"long": [], "short": []},
            "stop_loss": {"type": "fixed_pct", "percent": 2.0},
            "take_profit": {"type": "fixed_pct", "percent": 3.0},
        }
        with pytest.raises(ValueError, match="at least one long or short condition"):
            StrategyDSL.model_validate(data)

    def test_get_all_timeframes(self) -> None:
        """Get all timeframes used by strategy."""
        data = {
            "name": "test_strategy",
            "timeframe": "5m",
            "additional_timeframes": ["1h", "4h"],
            "indicators": {"rsi": {"type": "RSI", "period": 14}},
            "entry_conditions": {"long": ["rsi < 30"]},
            "stop_loss": {"type": "fixed_pct", "percent": 2.0},
            "take_profit": {"type": "fixed_pct", "percent": 3.0},
        }
        strategy = StrategyDSL.model_validate(data)
        assert strategy.get_all_timeframes() == {"5m", "1h", "4h"}

    def test_get_indicator_names(self) -> None:
        """Get all indicator names."""
        data = {
            "name": "test_strategy",
            "timeframe": "5m",
            "indicators": {
                "rsi": {"type": "RSI", "period": 14},
                "ema": {"type": "EMA", "period": 20},
            },
            "entry_conditions": {"long": ["rsi < 30"]},
            "stop_loss": {"type": "fixed_pct", "percent": 2.0},
            "take_profit": {"type": "fixed_pct", "percent": 3.0},
        }
        strategy = StrategyDSL.model_validate(data)
        assert strategy.get_indicator_names() == {"rsi", "ema"}


# =============================================================================
# Parser Tests
# =============================================================================


class TestParseStrategyString:
    """Tests for parse_strategy_string()."""

    def test_parse_minimal_strategy(self, minimal_strategy_yaml: str) -> None:
        """Parse minimal strategy YAML."""
        strategy = parse_strategy_string(minimal_strategy_yaml)
        assert strategy.name == "test_strategy"
        assert strategy.timeframe == "5m"
        assert "rsi" in strategy.indicators

    def test_parse_full_strategy(self, full_strategy_yaml: str) -> None:
        """Parse full strategy with all features."""
        strategy = parse_strategy_string(full_strategy_yaml)
        assert strategy.name == "rsi_mean_reversion_mtf"
        assert strategy.timeframe == "5m"
        assert set(strategy.additional_timeframes) == {"1h", "4h"}
        assert len(strategy.indicators) == 4
        assert len(strategy.entry_conditions.long) == 3
        assert len(strategy.entry_conditions.short) == 3
        assert strategy.stop_loss.type == "atr_trailing"
        assert strategy.time_filters.avoid_around_funding.enabled

    def test_parse_invalid_yaml_syntax(self) -> None:
        """Invalid YAML syntax raises DSLParseError."""
        invalid_yaml = """
name: test
indicators:
  rsi: [invalid
"""
        with pytest.raises(DSLParseError, match="Invalid YAML syntax"):
            parse_strategy_string(invalid_yaml)

    def test_parse_empty_yaml(self) -> None:
        """Empty YAML raises DSLParseError."""
        with pytest.raises(DSLParseError, match="Empty YAML document"):
            parse_strategy_string("")

    def test_parse_non_dict_yaml(self) -> None:
        """Non-dict YAML raises DSLParseError."""
        with pytest.raises(DSLParseError, match="Expected YAML mapping"):
            parse_strategy_string("- item1\n- item2")

    def test_parse_invalid_schema(self) -> None:
        """Invalid schema raises DSLValidationError."""
        invalid_yaml = """
name: test_strategy
timeframe: invalid_tf
indicators:
  rsi:
    type: RSI
entry_conditions:
  long:
    - rsi < 30
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""
        with pytest.raises(DSLValidationError, match="Strategy validation failed"):
            parse_strategy_string(invalid_yaml)

    def test_parse_undefined_indicator_in_condition(self) -> None:
        """Condition referencing undefined indicator raises error."""
        yaml_content = """
name: test_strategy
timeframe: 5m
indicators:
  rsi:
    type: RSI
    period: 14
entry_conditions:
  long:
    - undefined_indicator < 30
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""
        with pytest.raises(DSLValidationError, match="Condition validation failed"):
            parse_strategy_string(yaml_content)


class TestParseStrategyFile:
    """Tests for parse_strategy() from file."""

    def test_parse_nonexistent_file(self) -> None:
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_strategy("/nonexistent/path.yaml")

    def test_parse_strategy_from_file(
        self, tmp_path: Path, minimal_strategy_yaml: str
    ) -> None:
        """Parse strategy from file."""
        yaml_file = tmp_path / "strategy.yaml"
        yaml_file.write_text(minimal_strategy_yaml)

        strategy = parse_strategy(yaml_file)
        assert strategy.name == "test_strategy"


class TestValidateStrategyDict:
    """Tests for validate_strategy_dict()."""

    def test_validate_from_dict(self) -> None:
        """Validate strategy from dictionary."""
        data = {
            "name": "test_strategy",
            "timeframe": "5m",
            "indicators": {"rsi": {"type": "RSI", "period": 14}},
            "entry_conditions": {"long": ["rsi < 30"]},
            "stop_loss": {"type": "fixed_pct", "percent": 2.0},
            "take_profit": {"type": "fixed_pct", "percent": 3.0},
        }
        strategy = validate_strategy_dict(data)
        assert strategy.name == "test_strategy"

    def test_validate_invalid_dict(self) -> None:
        """Invalid dict raises DSLValidationError."""
        with pytest.raises(DSLValidationError):
            validate_strategy_dict({"name": "test"})  # Missing required fields


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_required_timeframes(self, full_strategy_yaml: str) -> None:
        """Get required timeframes from strategy."""
        strategy = parse_strategy_string(full_strategy_yaml)
        timeframes = get_required_timeframes(strategy)
        assert timeframes == {"5m", "1h", "4h"}

    def test_get_referenced_indicators(self, full_strategy_yaml: str) -> None:
        """Get referenced indicators by condition type."""
        strategy = parse_strategy_string(full_strategy_yaml)
        refs = get_referenced_indicators(strategy)
        assert "rsi" in refs["entry_long"]
        assert "rsi_1h" in refs["entry_long"]
        assert "ema_trend" in refs["entry_long"]

    def test_strategy_to_dict(self, minimal_strategy_yaml: str) -> None:
        """Convert strategy to dict."""
        strategy = parse_strategy_string(minimal_strategy_yaml)
        data = strategy_to_dict(strategy)
        assert data["name"] == "test_strategy"
        assert isinstance(data["indicators"], dict)

    def test_strategy_to_yaml(self, minimal_strategy_yaml: str) -> None:
        """Convert strategy to YAML."""
        strategy = parse_strategy_string(minimal_strategy_yaml)
        yaml_str = strategy_to_yaml(strategy)
        assert "name: test_strategy" in yaml_str
        assert "timeframe: 5m" in yaml_str


# =============================================================================
# Time Filter Tests
# =============================================================================


class TestTimeFilterValidation:
    """Tests for time filter validation."""

    def test_valid_session_config(self) -> None:
        """Parse valid session configuration."""
        yaml_content = """
name: test_strategy
timeframe: 5m
indicators:
  rsi:
    type: RSI
    period: 14
entry_conditions:
  long:
    - rsi < 30
time_filters:
  allowed_sessions:
    - start: "08:00"
      end: "20:00"
      timezone: UTC
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""
        strategy = parse_strategy_string(yaml_content)
        assert len(strategy.time_filters.allowed_sessions) == 1
        assert strategy.time_filters.allowed_sessions[0].start == "08:00"

    def test_invalid_session_time_format(self) -> None:
        """Invalid session time format raises error."""
        yaml_content = """
name: test_strategy
timeframe: 5m
indicators:
  rsi:
    type: RSI
    period: 14
entry_conditions:
  long:
    - rsi < 30
time_filters:
  allowed_sessions:
    - start: "8:00"
      end: "20:00"
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""
        with pytest.raises(DSLValidationError):
            parse_strategy_string(yaml_content)

    def test_invalid_blocked_day(self) -> None:
        """Invalid blocked day raises error."""
        yaml_content = """
name: test_strategy
timeframe: 5m
indicators:
  rsi:
    type: RSI
    period: 14
entry_conditions:
  long:
    - rsi < 30
time_filters:
  blocked_days:
    - Funday
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""
        with pytest.raises(DSLValidationError, match="Invalid day"):
            parse_strategy_string(yaml_content)


# =============================================================================
# Sweep Parameter Tests
# =============================================================================


class TestSweepValidation:
    """Tests for sweep parameter validation."""

    def test_valid_sweep_parameters(self, full_strategy_yaml: str) -> None:
        """Valid sweep parameters parse correctly."""
        strategy = parse_strategy_string(full_strategy_yaml)
        assert "rsi.period" in strategy.sweep
        assert strategy.sweep["rsi.period"] == [7, 14, 21]

    def test_empty_sweep_values(self) -> None:
        """Empty sweep values raise error."""
        yaml_content = """
name: test_strategy
timeframe: 5m
indicators:
  rsi:
    type: RSI
    period: 14
entry_conditions:
  long:
    - rsi < 30
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
sweep:
  rsi.period: []
"""
        with pytest.raises(DSLValidationError, match="Empty sweep range"):
            parse_strategy_string(yaml_content)


# =============================================================================
# Integration Tests
# =============================================================================


class TestRoundTrip:
    """Test YAML -> Strategy -> YAML round-trip."""

    def test_roundtrip_minimal(self, minimal_strategy_yaml: str) -> None:
        """Round-trip minimal strategy."""
        strategy1 = parse_strategy_string(minimal_strategy_yaml)
        yaml_out = strategy_to_yaml(strategy1)
        strategy2 = parse_strategy_string(yaml_out)

        assert strategy1.name == strategy2.name
        assert strategy1.timeframe == strategy2.timeframe
        assert set(strategy1.indicators.keys()) == set(strategy2.indicators.keys())

    def test_roundtrip_full(self, full_strategy_yaml: str) -> None:
        """Round-trip full strategy."""
        strategy1 = parse_strategy_string(full_strategy_yaml)
        yaml_out = strategy_to_yaml(strategy1)
        strategy2 = parse_strategy_string(yaml_out)

        assert strategy1.name == strategy2.name
        assert strategy1.timeframe == strategy2.timeframe
        assert set(strategy1.additional_timeframes) == set(strategy2.additional_timeframes)
        assert len(strategy1.entry_conditions.long) == len(strategy2.entry_conditions.long)
