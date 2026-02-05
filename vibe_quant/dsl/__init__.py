"""Strategy DSL parser, validator, and NautilusTrader compiler.

This module provides the DSL (Domain Specific Language) for defining
trading strategies in YAML format. It includes:

- Schema definitions (Pydantic models)
- Condition parsing
- YAML parsing and validation
- Indicator registry with NautilusTrader/pandas-ta mappings
- Error reporting with line numbers

Example usage:
    from vibe_quant.dsl import parse_strategy, StrategyDSL, indicator_registry

    # Parse from file
    strategy = parse_strategy("strategies/my_strategy.yaml")

    # Parse from string
    strategy = parse_strategy_string(yaml_content)

    # Access strategy properties
    print(strategy.name)
    print(strategy.indicators)

    # Get indicator spec
    rsi_spec = indicator_registry.get("RSI")
    print(rsi_spec.default_params)

    # List all available indicators
    print(indicator_registry.list_indicators())
"""

from vibe_quant.dsl.compiler import (
    CompilerError,
    IndicatorInfo,
    StrategyCompiler,
)
from vibe_quant.dsl.conditions import (
    Condition,
    ConditionParseError,
    Operand,
    Operator,
    extract_indicator_refs,
    parse_condition,
    validate_conditions,
)
from vibe_quant.dsl.indicators import (
    IndicatorRegistry,
    IndicatorSpec,
    indicator_registry,
)
from vibe_quant.dsl.parser import (
    DSLParseError,
    DSLValidationError,
    get_referenced_indicators,
    get_required_timeframes,
    parse_strategy,
    parse_strategy_string,
    strategy_to_dict,
    strategy_to_yaml,
    validate_strategy_dict,
)
from vibe_quant.dsl.schema import (
    VALID_INDICATOR_TYPES,
    VALID_SOURCES,
    VALID_STOP_LOSS_TYPES,
    VALID_TAKE_PROFIT_TYPES,
    VALID_TIMEFRAMES,
    EntryConditions,
    ExitConditions,
    FundingAvoidanceConfig,
    IndicatorConfig,
    PositionManagementConfig,
    SessionConfig,
    StopLossConfig,
    StrategyDSL,
    TakeProfitConfig,
    TimeFilterConfig,
)

__all__ = [
    # Schema models
    "StrategyDSL",
    "IndicatorConfig",
    "EntryConditions",
    "ExitConditions",
    "TimeFilterConfig",
    "SessionConfig",
    "FundingAvoidanceConfig",
    "StopLossConfig",
    "TakeProfitConfig",
    "PositionManagementConfig",
    # Schema constants
    "VALID_TIMEFRAMES",
    "VALID_INDICATOR_TYPES",
    "VALID_SOURCES",
    "VALID_STOP_LOSS_TYPES",
    "VALID_TAKE_PROFIT_TYPES",
    # Indicator registry
    "IndicatorSpec",
    "IndicatorRegistry",
    "indicator_registry",
    # Condition parsing
    "Condition",
    "Operand",
    "Operator",
    "ConditionParseError",
    "parse_condition",
    "validate_conditions",
    "extract_indicator_refs",
    # Parser functions
    "parse_strategy",
    "parse_strategy_string",
    "validate_strategy_dict",
    "get_required_timeframes",
    "get_referenced_indicators",
    "strategy_to_dict",
    "strategy_to_yaml",
    # Compiler
    "StrategyCompiler",
    "CompilerError",
    "IndicatorInfo",
    # Errors
    "DSLParseError",
    "DSLValidationError",
]
