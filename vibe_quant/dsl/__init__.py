"""Strategy DSL parser, validator, and NautilusTrader compiler.

This module provides the DSL (Domain Specific Language) for defining
trading strategies in YAML format. It includes:

- Schema definitions (Pydantic models)
- Condition parsing
- YAML parsing and validation
- Error reporting with line numbers

Example usage:
    from vibe_quant.dsl import parse_strategy, StrategyDSL

    # Parse from file
    strategy = parse_strategy("strategies/my_strategy.yaml")

    # Parse from string
    strategy = parse_strategy_string(yaml_content)

    # Access strategy properties
    print(strategy.name)
    print(strategy.indicators)
    for cond in strategy.entry_conditions.long:
        print(cond)
"""

from vibe_quant.dsl.conditions import (
    Condition,
    ConditionParseError,
    Operand,
    Operator,
    extract_indicator_refs,
    parse_condition,
    validate_conditions,
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
    # Errors
    "DSLParseError",
    "DSLValidationError",
]
