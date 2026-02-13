"""DSL parser for trading strategies.

Parses YAML strategy definitions into validated StrategyDSL objects
with comprehensive error reporting including line numbers.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from vibe_quant.dsl.conditions import (
    ConditionParseError,
    extract_indicator_refs,
    parse_condition,
)
from vibe_quant.dsl.indicators import indicator_registry
from vibe_quant.dsl.schema import StrategyDSL


class DSLParseError(Exception):
    """Error raised when DSL parsing fails.

    Attributes:
        message: Error message
        file_path: Path to the file being parsed (if from file)
        line_number: Line number where error occurred (if available)
        details: Additional error details
    """

    def __init__(
        self,
        message: str,
        file_path: Path | str | None = None,
        line_number: int | None = None,
        details: list[str] | None = None,
    ) -> None:
        """Initialize DSL parse error.

        Args:
            message: Primary error message
            file_path: Path to file being parsed
            line_number: Line number of error
            details: Additional error details
        """
        self.file_path = file_path
        self.line_number = line_number
        self.details = details or []

        # Build full message
        parts = []
        if file_path:
            parts.append(f"File: {file_path}")
        if line_number:
            parts.append(f"Line: {line_number}")
        parts.append(message)
        if details:
            parts.extend(f"  - {d}" for d in details)

        super().__init__("\n".join(parts))


class DSLValidationError(DSLParseError):
    """Error raised when DSL validation fails after parsing."""

    pass


def _format_pydantic_errors(error: ValidationError) -> list[str]:
    """Format Pydantic validation errors into readable messages.

    Args:
        error: Pydantic ValidationError

    Returns:
        List of formatted error messages
    """
    messages = []
    for err in error.errors():
        loc = ".".join(str(x) for x in err["loc"])
        msg = err["msg"]
        messages.append(f"{loc}: {msg}")
    return messages


def _find_yaml_line_number(yaml_str: str, key_path: list[str]) -> int | None:
    """Attempt to find line number for a YAML key path.

    This is a best-effort function that scans the YAML for the key.

    Args:
        yaml_str: Raw YAML string
        key_path: Path of keys to find (e.g., ["indicators", "rsi"])

    Returns:
        Line number (1-indexed) or None if not found
    """
    lines = yaml_str.splitlines()
    current_indent = -1
    path_index = 0

    for i, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(stripped)

        # Check if this line starts a key we're looking for
        if path_index < len(key_path):
            target_key = key_path[path_index]
            is_target_key = stripped.startswith(f"{target_key}:")
            valid_indent = indent > current_indent or (path_index == 0 and indent == 0)
            if is_target_key and valid_indent:
                current_indent = indent
                path_index += 1
                if path_index == len(key_path):
                    return i

    return None


def parse_strategy(yaml_path: Path | str) -> StrategyDSL:
    """Parse a strategy DSL from a YAML file.

    Args:
        yaml_path: Path to the YAML file

    Returns:
        Parsed and validated StrategyDSL

    Raises:
        DSLParseError: If YAML parsing fails
        DSLValidationError: If validation fails
        FileNotFoundError: If file doesn't exist
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Strategy file not found: {path}")

    yaml_content = path.read_text(encoding="utf-8")
    try:
        return parse_strategy_string(yaml_content, file_path=path)
    except DSLParseError as e:
        # Re-raise with file path context
        if e.file_path is None:
            e.file_path = path
        raise


def parse_strategy_string(
    yaml_content: str,
    file_path: Path | str | None = None,
) -> StrategyDSL:
    """Parse a strategy DSL from a YAML string.

    Args:
        yaml_content: YAML content as string
        file_path: Optional file path for error messages

    Returns:
        Parsed and validated StrategyDSL

    Raises:
        DSLParseError: If YAML parsing fails
        DSLValidationError: If validation fails
    """
    # Step 1: Parse YAML
    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        line = getattr(e, "problem_mark", None)
        line_num = line.line + 1 if line else None
        raise DSLParseError(
            f"Invalid YAML syntax: {e}",
            file_path=file_path,
            line_number=line_num,
        ) from e

    if data is None:
        raise DSLParseError("Empty YAML document", file_path=file_path)

    if not isinstance(data, dict):
        raise DSLParseError(
            f"Expected YAML mapping (dict), got {type(data).__name__}",
            file_path=file_path,
        )

    # Step 2: Validate with Pydantic
    try:
        strategy = StrategyDSL.model_validate(data)
    except ValidationError as e:
        errors = _format_pydantic_errors(e)
        # Try to find line numbers for first error
        first_error = e.errors()[0] if e.errors() else None
        line_num = None
        if first_error:
            key_path = [str(x) for x in first_error["loc"] if not isinstance(x, int)]
            line_num = _find_yaml_line_number(yaml_content, key_path)

        raise DSLValidationError(
            "Strategy validation failed",
            file_path=file_path,
            line_number=line_num,
            details=errors,
        ) from e

    # Step 3: Validate conditions reference valid indicators
    _validate_condition_indicators(strategy, yaml_content, file_path)

    # Step 4: Validate sweep parameters
    _validate_sweep_parameters(strategy, yaml_content, file_path)

    return strategy


def _build_valid_indicator_names(strategy: StrategyDSL) -> list[str]:
    """Build list of valid indicator names including multi-output sub-names.

    For multi-output indicators (e.g., MACD with outputs macd/signal/histogram),
    adds both the base name and ``{base}_{output}`` sub-names so conditions
    like ``macd.histogram > 0`` (resolved to ``macd_histogram``) validate.

    Args:
        strategy: Parsed strategy.

    Returns:
        List of valid indicator names for condition validation.
    """
    names: list[str] = list(strategy.indicators.keys())
    for name, config in strategy.indicators.items():
        spec = indicator_registry.get(config.type)
        if spec is not None and spec.output_names != ("value",):
            for output_name in spec.output_names:
                names.append(f"{name}_{output_name}")
    return names


def _validate_condition_indicators(
    strategy: StrategyDSL,
    yaml_content: str,
    file_path: Path | str | None,
) -> None:
    """Validate that all conditions reference defined indicators.

    Args:
        strategy: Parsed strategy
        yaml_content: Original YAML for line number lookup
        file_path: File path for error messages

    Raises:
        DSLValidationError: If conditions reference undefined indicators
    """
    valid_names = _build_valid_indicator_names(strategy)
    errors: list[str] = []

    # Check entry conditions
    for cond_str in strategy.entry_conditions.long:
        try:
            parse_condition(cond_str, valid_names)
        except ConditionParseError as e:
            errors.append(f"entry_conditions.long: {e}")

    for cond_str in strategy.entry_conditions.short:
        try:
            parse_condition(cond_str, valid_names)
        except ConditionParseError as e:
            errors.append(f"entry_conditions.short: {e}")

    # Check exit conditions
    for cond_str in strategy.exit_conditions.long:
        try:
            parse_condition(cond_str, valid_names)
        except ConditionParseError as e:
            errors.append(f"exit_conditions.long: {e}")

    for cond_str in strategy.exit_conditions.short:
        try:
            parse_condition(cond_str, valid_names)
        except ConditionParseError as e:
            errors.append(f"exit_conditions.short: {e}")

    if errors:
        line_num = _find_yaml_line_number(yaml_content, ["entry_conditions"])
        raise DSLValidationError(
            "Condition validation failed",
            file_path=file_path,
            line_number=line_num,
            details=errors,
        )


def _validate_sweep_parameters(
    strategy: StrategyDSL,
    yaml_content: str,
    file_path: Path | str | None,
) -> None:
    """Validate sweep parameter paths reference valid config fields.

    Args:
        strategy: Parsed strategy
        yaml_content: Original YAML for line number lookup
        file_path: File path for error messages

    Raises:
        DSLValidationError: If sweep parameters are invalid
    """
    if not strategy.sweep:
        return

    errors: list[str] = []
    indicator_names = set(strategy.indicators.keys())

    for param_path, values in strategy.sweep.items():
        # Validate values list
        if not values:
            errors.append(f"sweep.{param_path}: Empty sweep range")
            continue

        # Check if all values are numeric
        for v in values:
            if not isinstance(v, (int, float)):
                errors.append(f"sweep.{param_path}: Values must be numeric, got {type(v).__name__}")
                break

        # Parse the parameter path
        parts = param_path.split(".")
        if len(parts) == 1:
            # Simple param like "rsi_period" - these are strategy-level params
            # We allow any name here as they're passed to the compiled strategy
            continue

        # Compound path like "rsi.period" or "stop_loss.atr_multiplier"
        root = parts[0]
        field = parts[1] if len(parts) > 1 else None

        if root in indicator_names:
            # Indicator parameter sweep
            valid_indicator_fields = {
                "period", "source", "fast_period", "slow_period",
                "signal_period", "std_dev", "atr_multiplier"
            }
            if field and field not in valid_indicator_fields:
                errors.append(f"sweep.{param_path}: Invalid indicator field '{field}'")
        elif root == "stop_loss":
            valid_fields = {"percent", "atr_multiplier"}
            if field and field not in valid_fields:
                errors.append(f"sweep.{param_path}: Invalid stop_loss field '{field}'")
        elif root == "take_profit":
            valid_fields = {"percent", "atr_multiplier", "risk_reward_ratio"}
            if field and field not in valid_fields:
                errors.append(f"sweep.{param_path}: Invalid take_profit field '{field}'")
        elif root not in {"rsi_oversold_threshold", "rsi_overbought_threshold"}:
            # Allow common threshold parameter names
            pass  # Unknown roots are allowed as custom parameters

    if errors:
        line_num = _find_yaml_line_number(yaml_content, ["sweep"])
        raise DSLValidationError(
            "Sweep parameter validation failed",
            file_path=file_path,
            line_number=line_num,
            details=errors,
        )


def validate_strategy_dict(data: dict[str, object]) -> StrategyDSL:
    """Validate a strategy from a dictionary (e.g., from database).

    Args:
        data: Dictionary containing strategy configuration

    Returns:
        Validated StrategyDSL

    Raises:
        DSLValidationError: If validation fails
    """
    try:
        strategy = StrategyDSL.model_validate(data)
    except ValidationError as e:
        errors = _format_pydantic_errors(e)
        raise DSLValidationError(
            "Strategy validation failed",
            details=errors,
        ) from e

    # Validate conditions (including multi-output sub-names)
    valid_names = _build_valid_indicator_names(strategy)

    condition_errors: list[str] = []
    for cond_str in strategy.entry_conditions.long:
        try:
            parse_condition(cond_str, valid_names)
        except ConditionParseError as e:
            condition_errors.append(f"entry_conditions.long: {e}")

    for cond_str in strategy.entry_conditions.short:
        try:
            parse_condition(cond_str, valid_names)
        except ConditionParseError as e:
            condition_errors.append(f"entry_conditions.short: {e}")

    for cond_str in strategy.exit_conditions.long:
        try:
            parse_condition(cond_str, valid_names)
        except ConditionParseError as e:
            condition_errors.append(f"exit_conditions.long: {e}")

    for cond_str in strategy.exit_conditions.short:
        try:
            parse_condition(cond_str, valid_names)
        except ConditionParseError as e:
            condition_errors.append(f"exit_conditions.short: {e}")

    if condition_errors:
        raise DSLValidationError("Condition validation failed", details=condition_errors)

    return strategy


def get_required_timeframes(strategy: StrategyDSL) -> set[str]:
    """Get all timeframes required by a strategy.

    Args:
        strategy: Parsed strategy

    Returns:
        Set of timeframe strings (e.g., {"5m", "1h", "4h"})
    """
    timeframes = {strategy.timeframe}
    timeframes.update(strategy.additional_timeframes)

    # Also include indicator-specific timeframes
    for indicator in strategy.indicators.values():
        if indicator.timeframe:
            timeframes.add(indicator.timeframe)

    return timeframes


def get_referenced_indicators(strategy: StrategyDSL) -> dict[str, set[str]]:
    """Get indicators referenced in conditions, grouped by usage.

    Args:
        strategy: Parsed strategy

    Returns:
        Dictionary with keys 'entry_long', 'entry_short', 'exit_long', 'exit_short'
        containing sets of indicator names used in each context.
    """
    result: dict[str, set[str]] = {
        "entry_long": extract_indicator_refs(strategy.entry_conditions.long),
        "entry_short": extract_indicator_refs(strategy.entry_conditions.short),
        "exit_long": extract_indicator_refs(strategy.exit_conditions.long),
        "exit_short": extract_indicator_refs(strategy.exit_conditions.short),
    }
    return result


def strategy_to_dict(strategy: StrategyDSL) -> dict[str, object]:
    """Convert a strategy back to a dictionary.

    Args:
        strategy: Strategy to convert

    Returns:
        Dictionary representation suitable for YAML serialization
    """
    return strategy.model_dump(mode="python", exclude_none=True)


def strategy_to_yaml(strategy: StrategyDSL) -> str:
    """Convert a strategy to YAML string.

    Args:
        strategy: Strategy to convert

    Returns:
        YAML string representation
    """
    data = strategy_to_dict(strategy)
    result: str = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return result
