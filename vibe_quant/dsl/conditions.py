"""Condition parser for trading strategy DSL.

Parses condition strings like "rsi < 30" or "ema_fast crosses_above ema_slow"
into structured Condition objects for evaluation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class Operator(Enum):
    """Condition operators supported by the DSL."""

    GT = ">"  # Greater than
    LT = "<"  # Less than
    GTE = ">="  # Greater than or equal
    LTE = "<="  # Less than or equal
    CROSSES_ABOVE = "crosses_above"  # Crossover
    CROSSES_BELOW = "crosses_below"  # Crossunder
    BETWEEN = "between"  # Range check

    @classmethod
    def from_string(cls, s: str) -> Operator:
        """Parse operator from string."""
        s = s.strip().lower()
        mapping = {
            ">": cls.GT,
            "<": cls.LT,
            ">=": cls.GTE,
            "<=": cls.LTE,
            "crosses_above": cls.CROSSES_ABOVE,
            "crosses_below": cls.CROSSES_BELOW,
            "between": cls.BETWEEN,
        }
        if s not in mapping:
            valid = ", ".join(mapping.keys())
            msg = f"Unknown operator '{s}'. Valid operators: {valid}"
            raise ValueError(msg)
        return mapping[s]


@dataclass(frozen=True, slots=True)
class Operand:
    """An operand in a condition (indicator reference or literal value).

    Attributes:
        value: The operand value (indicator name or numeric literal)
        is_indicator: Whether this is an indicator reference
        is_price: Whether this is a price reference (close, open, etc.)
    """

    value: str | float
    is_indicator: bool = False
    is_price: bool = False

    @classmethod
    def parse(cls, s: str, valid_indicators: Sequence[str] | None = None) -> Operand:
        """Parse operand from string.

        Supports dot notation for multi-output indicators: ``macd.histogram``
        is resolved to the flat name ``macd_histogram`` which the compiler
        registers as a sub-output accessor.

        Args:
            s: String to parse
            valid_indicators: Optional list of valid indicator names for validation

        Returns:
            Parsed Operand

        Raises:
            ValueError: If operand is invalid
        """
        s = s.strip()

        # Check for price references
        price_refs = {"close", "open", "high", "low", "volume"}
        if s.lower() in price_refs:
            return cls(value=s.lower(), is_indicator=False, is_price=True)

        # Try to parse as numeric. We skip only when the string has BOTH
        # alpha chars AND a dot — that pattern is a dot-notation indicator
        # like "macd.histogram". Pure numbers ("3.5", "42") and plain alpha
        # names ("rsi") both go through float() — the latter fails harmlessly
        # and falls through to indicator resolution below.
        has_alpha = any(c.isalpha() for c in s)
        has_dot = "." in s
        if not has_alpha or not has_dot:
            try:
                return cls(value=float(s), is_indicator=False, is_price=False)
            except ValueError:
                pass

        # Handle dot notation: "macd.histogram" -> "macd_histogram"
        resolved = s.replace(".", "_") if "." in s else s

        # Must be an indicator reference
        if valid_indicators is not None and resolved not in valid_indicators:
            valid_list = ", ".join(sorted(valid_indicators)) if valid_indicators else "(none)"
            msg = f"Unknown indicator '{s}' in condition. Defined indicators: {valid_list}"
            raise ValueError(msg)

        return cls(value=resolved, is_indicator=True, is_price=False)


@dataclass(frozen=True, slots=True)
class Condition:
    """A parsed condition from the DSL.

    Represents conditions like:
    - "rsi < 30" -> Condition(left=rsi, operator=LT, right=30)
    - "close > ema_fast" -> Condition(left=close, operator=GT, right=ema_fast)
    - "rsi between 30 70" -> Condition(left=rsi, operator=BETWEEN, right=30, right2=70)

    Attributes:
        left: Left operand (usually indicator or price)
        operator: Comparison operator
        right: Right operand (value or indicator)
        right2: Second right operand (only for BETWEEN)
        raw: Original raw condition string
    """

    left: Operand
    operator: Operator
    right: Operand
    right2: Operand | None = None
    raw: str = ""

    def __post_init__(self) -> None:
        """Validate condition structure."""
        if self.operator == Operator.BETWEEN and self.right2 is None:
            msg = "BETWEEN operator requires two right operands"
            raise ValueError(msg)

    def get_indicator_refs(self) -> set[str]:
        """Get all indicator names referenced in this condition."""
        refs: set[str] = set()
        if self.left.is_indicator and isinstance(self.left.value, str):
            refs.add(self.left.value)
        if self.right.is_indicator and isinstance(self.right.value, str):
            refs.add(self.right.value)
        if self.right2 and self.right2.is_indicator and isinstance(self.right2.value, str):
            refs.add(self.right2.value)
        return refs


class ConditionParseError(ValueError):
    """Error raised when condition parsing fails."""

    def __init__(self, message: str, raw_condition: str, position: int | None = None) -> None:
        """Initialize parse error.

        Args:
            message: Error message
            raw_condition: Original condition string
            position: Character position where error occurred
        """
        self.raw_condition = raw_condition
        self.position = position
        if position is not None:
            full_msg = f"{message} at position {position}: '{raw_condition}'"
        else:
            full_msg = f"{message}: '{raw_condition}'"
        super().__init__(full_msg)


# Regex patterns for parsing
# Allow dot notation for multi-output indicators (e.g., macd.histogram)
_COMPARISON_PATTERN = re.compile(r"^\s*(\w+(?:\.\w+)?)\s*(>=|<=|>|<)\s*(-?\w+(?:\.\w+)?)\s*$")
_CROSS_PATTERN = re.compile(
    r"^\s*(\w+(?:\.\w+)?)\s+(crosses_above|crosses_below)\s+(-?\w+(?:\.\w+)?)\s*$",
    re.IGNORECASE,
)
_BETWEEN_PATTERN = re.compile(
    r"^\s*(\w+(?:\.\w+)?)\s+between\s+(-?\w+(?:\.\w+)?)\s+(-?\w+(?:\.\w+)?)\s*$",
    re.IGNORECASE,
)


def parse_condition(
    condition_str: str,
    valid_indicators: Sequence[str] | None = None,
) -> Condition:
    """Parse a condition string into a Condition object.

    Supported formats:
    - "<indicator> > <value|indicator>"
    - "<indicator> < <value|indicator>"
    - "<indicator> >= <value|indicator>"
    - "<indicator> <= <value|indicator>"
    - "<indicator> crosses_above <value|indicator>"
    - "<indicator> crosses_below <value|indicator>"
    - "<indicator> between <low> <high>"
    - "close > <indicator>"
    - "close < <indicator>"

    Args:
        condition_str: The condition string to parse
        valid_indicators: Optional list of valid indicator names for validation

    Returns:
        Parsed Condition object

    Raises:
        ConditionParseError: If the condition string is invalid
    """
    condition_str = condition_str.strip()
    if not condition_str:
        raise ConditionParseError("Empty condition string", condition_str)

    # Try BETWEEN pattern first (most specific)
    match = _BETWEEN_PATTERN.match(condition_str)
    if match:
        left_str, low_str, high_str = match.groups()
        try:
            left = Operand.parse(left_str, valid_indicators)
            low = Operand.parse(low_str, None)  # Values don't need indicator validation
            high = Operand.parse(high_str, None)
        except ValueError as e:
            raise ConditionParseError(str(e), condition_str) from e

        # Validate low < high for numeric values
        if (
            not low.is_indicator
            and not high.is_indicator
            and isinstance(low.value, (int, float))
            and isinstance(high.value, (int, float))
            and low.value >= high.value
        ):
            msg = f"BETWEEN requires low ({low.value}) < high ({high.value})"
            raise ConditionParseError(msg, condition_str)

        return Condition(
            left=left,
            operator=Operator.BETWEEN,
            right=low,
            right2=high,
            raw=condition_str,
        )

    # Try cross patterns
    match = _CROSS_PATTERN.match(condition_str)
    if match:
        left_str, op_str, right_str = match.groups()
        try:
            left = Operand.parse(left_str, valid_indicators)
            right = Operand.parse(right_str, valid_indicators)
            operator = Operator.from_string(op_str)
        except ValueError as e:
            raise ConditionParseError(str(e), condition_str) from e

        return Condition(
            left=left,
            operator=operator,
            right=right,
            raw=condition_str,
        )

    # Try comparison patterns
    match = _COMPARISON_PATTERN.match(condition_str)
    if match:
        left_str, op_str, right_str = match.groups()
        try:
            left = Operand.parse(left_str, valid_indicators)
            right = Operand.parse(right_str, valid_indicators)
            operator = Operator.from_string(op_str)
        except ValueError as e:
            raise ConditionParseError(str(e), condition_str) from e

        return Condition(
            left=left,
            operator=operator,
            right=right,
            raw=condition_str,
        )

    # No pattern matched
    raise ConditionParseError(
        "Invalid condition format. Expected: "
        "'<indicator> <op> <value>', "
        "'<indicator> crosses_above/below <value>', or "
        "'<indicator> between <low> <high>'",
        condition_str,
    )


def validate_conditions(
    conditions: Sequence[str],
    valid_indicators: Sequence[str],
) -> list[Condition]:
    """Parse and validate a list of condition strings.

    Args:
        conditions: List of condition strings
        valid_indicators: List of valid indicator names

    Returns:
        List of parsed Condition objects

    Raises:
        ConditionParseError: If any condition is invalid
    """
    parsed: list[Condition] = []
    for i, cond_str in enumerate(conditions):
        try:
            cond = parse_condition(cond_str, valid_indicators)
            parsed.append(cond)
        except ConditionParseError as e:
            # Add index context
            msg = f"Condition {i + 1}: {e}"
            raise ConditionParseError(msg, cond_str) from e
    return parsed


def extract_indicator_refs(conditions: Sequence[str]) -> set[str]:
    """Extract all indicator references from condition strings.

    This does NOT validate the indicators exist - it just extracts
    what look like indicator names for later validation.

    Args:
        conditions: List of condition strings

    Returns:
        Set of indicator names referenced
    """
    refs: set[str] = set()
    price_refs = {"close", "open", "high", "low", "volume"}

    for cond_str in conditions:
        # Parse without validation to extract refs
        try:
            cond = parse_condition(cond_str, valid_indicators=None)
            if (
                cond.left.is_indicator
                and isinstance(cond.left.value, str)
                and cond.left.value.lower() not in price_refs
            ):
                refs.add(cond.left.value)
            if (
                cond.right.is_indicator
                and isinstance(cond.right.value, str)
                and cond.right.value.lower() not in price_refs
            ):
                refs.add(cond.right.value)
            if (
                cond.right2
                and cond.right2.is_indicator
                and isinstance(cond.right2.value, str)
                and cond.right2.value.lower() not in price_refs
            ):
                refs.add(cond.right2.value)
        except ConditionParseError:
            # Skip unparseable conditions - they'll be caught during validation
            logger.debug("Skipping unparseable condition ref: %s", cond_str)

    return refs
