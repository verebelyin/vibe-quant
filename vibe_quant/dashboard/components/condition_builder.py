"""Visual condition builder component.

Replaces free-text condition editing with a row-based builder:
- Dropdown for left operand (indicator or price)
- Dropdown for operator
- Input for right operand (value or indicator)
- Add/remove row buttons
- Toggle to raw text mode
"""

from __future__ import annotations

import streamlit as st

# Operators supported by the DSL condition parser
OPERATORS = ["<", ">", "<=", ">=", "crosses_above", "crosses_below", "between"]

OPERATOR_LABELS = {
    "<": "less than (<)",
    ">": "greater than (>)",
    "<=": "less than or equal (<=)",
    ">=": "greater than or equal (>=)",
    "crosses_above": "crosses above",
    "crosses_below": "crosses below",
    "between": "between (range)",
}

# Price references that can be used as operands
PRICE_REFS = ["close", "open", "high", "low"]


def render_condition_builder(
    label: str,
    conditions: list[str],
    indicator_names: list[str],
    key_prefix: str,
) -> list[str]:
    """Render a visual condition builder and return the list of condition strings.

    Args:
        label: Section label (e.g., "Long Entry Conditions")
        conditions: Current condition strings
        indicator_names: Available indicator names from the strategy
        key_prefix: Unique key prefix for Streamlit widgets

    Returns:
        Updated list of condition strings
    """
    st.markdown(f"**{label}**")

    # Mode toggle: visual vs raw
    raw_mode_key = f"{key_prefix}_raw_mode"
    if raw_mode_key not in st.session_state:
        st.session_state[raw_mode_key] = False

    col_label, col_toggle = st.columns([3, 1])
    with col_toggle:
        if st.button(
            "Raw text" if not st.session_state[raw_mode_key] else "Visual",
            key=f"{key_prefix}_mode_toggle",
            help="Switch between visual builder and raw text editing",
        ):
            st.session_state[raw_mode_key] = not st.session_state[raw_mode_key]
            st.rerun()

    if st.session_state[raw_mode_key]:
        return _render_raw_mode(conditions, key_prefix)

    return _render_visual_mode(conditions, indicator_names, key_prefix)


def _render_raw_mode(conditions: list[str], key_prefix: str) -> list[str]:
    """Render raw text editing mode."""
    text = st.text_area(
        "Conditions (one per line)",
        value="\n".join(conditions),
        key=f"{key_prefix}_raw",
        height=120,
        help='e.g., "rsi < 30" or "ema_fast crosses_above ema_slow"',
        label_visibility="collapsed",
    )
    return [c.strip() for c in text.split("\n") if c.strip()]


def _render_visual_mode(
    conditions: list[str],
    indicator_names: list[str],
    key_prefix: str,
) -> list[str]:
    """Render visual builder mode with dropdowns per row."""
    # Initialize condition rows in session state
    rows_key = f"{key_prefix}_rows"
    if rows_key not in st.session_state:
        st.session_state[rows_key] = [_parse_condition_to_row(c) for c in conditions]
        # Ensure at least one empty row if no conditions
        if not st.session_state[rows_key]:
            st.session_state[rows_key] = [_empty_row()]

    all_operands = indicator_names + PRICE_REFS
    result_conditions: list[str] = []
    rows_to_remove: list[int] = []

    for i, row in enumerate(st.session_state[rows_key]):
        condition_str = _render_condition_row(
            row, i, all_operands, key_prefix, rows_to_remove
        )
        if condition_str:
            result_conditions.append(condition_str)

    # Remove flagged rows (in reverse to preserve indices)
    for idx in sorted(rows_to_remove, reverse=True):
        st.session_state[rows_key].pop(idx)
    if rows_to_remove:
        st.rerun()

    # Add condition button
    if st.button("+ Add condition", key=f"{key_prefix}_add"):
        st.session_state[rows_key].append(_empty_row())
        st.rerun()

    if not result_conditions:
        st.caption("No conditions defined")

    return result_conditions


def _render_condition_row(
    row: dict,
    idx: int,
    all_operands: list[str],
    key_prefix: str,
    rows_to_remove: list[int],
) -> str | None:
    """Render a single condition row and return the condition string."""
    # Always create 5 columns to avoid NameError when operator changes
    # between 'between' (needs c5) and other operators at runtime.
    c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 1])

    # Defaults for variables that may not be set in every branch
    right: str = str(row.get("right", 0))
    low: float = float(row.get("right_low", 0))
    high: float = float(row.get("right_high", 100))
    use_number: bool = row.get("right_is_numeric", True)

    # Left operand
    with c1:
        left_options = all_operands
        left_current = row.get("left", "")
        left_idx = left_options.index(left_current) if left_current in left_options else 0
        left = st.selectbox(
            "Left",
            options=left_options if left_options else ["(define indicators first)"],
            index=left_idx if left_options else 0,
            key=f"{key_prefix}_left_{idx}",
            label_visibility="collapsed",
        )

    # Operator
    with c2:
        stored_op = row.get("operator", "<")
        op_idx = OPERATORS.index(stored_op) if stored_op in OPERATORS else 0
        operator = st.selectbox(
            "Op",
            options=OPERATORS,
            index=op_idx,
            format_func=lambda x: OPERATOR_LABELS.get(x, x),
            key=f"{key_prefix}_op_{idx}",
            label_visibility="collapsed",
        )

    # Right operand(s) — use the *widget* value (operator) not the stale row value
    if operator == "between":
        with c3:
            low = st.number_input(
                "Low",
                value=float(row.get("right_low", 0)),
                key=f"{key_prefix}_low_{idx}",
                label_visibility="collapsed",
                step=1.0,
            )
        with c4:
            high = st.number_input(
                "High",
                value=float(row.get("right_high", 100)),
                key=f"{key_prefix}_high_{idx}",
                label_visibility="collapsed",
                step=1.0,
            )
    else:
        with c3:
            # Right side can be a number or another indicator
            right_is_numeric = row.get("right_is_numeric", True)

            if operator in ("crosses_above", "crosses_below"):
                # Cross operators always use indicator references
                right_idx = (
                    all_operands.index(row.get("right", ""))
                    if row.get("right", "") in all_operands
                    else 0
                )
                right = st.selectbox(
                    "Right",
                    options=all_operands if all_operands else ["(define indicators)"],
                    index=right_idx if all_operands else 0,
                    key=f"{key_prefix}_right_{idx}",
                    label_visibility="collapsed",
                )
            else:
                # Comparison: toggle between number and indicator
                rc1, rc2 = st.columns([1, 3])
                with rc1:
                    use_number = st.checkbox(
                        "#",
                        value=right_is_numeric,
                        key=f"{key_prefix}_rtype_{idx}",
                        help="Check for number, uncheck for indicator",
                    )
                with rc2:
                    if use_number:
                        right = _format_num(st.number_input(
                            "Value",
                            value=_try_float(row.get("right", 0)),
                            key=f"{key_prefix}_rval_{idx}",
                            label_visibility="collapsed",
                            step=1.0,
                        ))
                    else:
                        r_idx = (
                            all_operands.index(row.get("right", ""))
                            if row.get("right", "") in all_operands
                            else 0
                        )
                        right = st.selectbox(
                            "Right indicator",
                            options=all_operands if all_operands else ["(define indicators)"],
                            index=r_idx if all_operands else 0,
                            key=f"{key_prefix}_rind_{idx}",
                            label_visibility="collapsed",
                        )
        # c4 intentionally empty in non-between mode

    # Remove button (always in c5)
    with c5:
        if st.button("X", key=f"{key_prefix}_rm_{idx}", help="Remove this condition"):
            rows_to_remove.append(idx)
            return None

    # Update row state
    row["left"] = left
    row["operator"] = operator
    if operator == "between":
        row["right_low"] = low
        row["right_high"] = high
        return f"{left} between {_format_num(low)} {_format_num(high)}"
    elif operator in ("crosses_above", "crosses_below"):
        row["right"] = right
        row["right_is_numeric"] = False
        return f"{left} {operator} {right}"
    else:
        row["right"] = right
        row["right_is_numeric"] = use_number
        return f"{left} {operator} {right}"


def _format_num(val: float) -> str:
    """Format a number: drop .0 for integers."""
    if val == int(val):
        return str(int(val))
    return str(val)


def _parse_condition_to_row(condition: str) -> dict:
    """Parse a condition string into a row dict for the visual builder."""
    condition = condition.strip().strip('"').strip("'")
    parts = condition.split()

    if not parts:
        return _empty_row()

    if len(parts) >= 4 and parts[1] == "between":
        return {
            "left": parts[0],
            "operator": "between",
            "right_low": _try_float(parts[2]),
            "right_high": _try_float(parts[3]) if len(parts) > 3 else 100,
            "right_is_numeric": True,
        }

    if len(parts) >= 3 and parts[1] in ("crosses_above", "crosses_below"):
        return {
            "left": parts[0],
            "operator": parts[1],
            "right": parts[2],
            "right_is_numeric": False,
        }

    if len(parts) >= 3 and parts[1] in ("<", ">", "<=", ">="):
        right_val = parts[2]
        is_numeric = _is_numeric(right_val)
        return {
            "left": parts[0],
            "operator": parts[1],
            "right": _try_float(right_val) if is_numeric else right_val,
            "right_is_numeric": is_numeric,
        }

    # Fallback: store as-is in left with defaults
    return {
        "left": parts[0] if parts else "",
        "operator": parts[1] if len(parts) > 1 else "<",
        "right": parts[2] if len(parts) > 2 else 0,
        "right_is_numeric": _is_numeric(parts[2]) if len(parts) > 2 else True,
    }


def _empty_row() -> dict:
    """Return an empty condition row."""
    return {
        "left": "",
        "operator": "<",
        "right": 0,
        "right_is_numeric": True,
    }


def _try_float(val: str | float | int) -> float:
    """Try to convert to float, return 0 on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _is_numeric(val: str | float | int) -> bool:
    """Check if a value is numeric."""
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Human-readable condition formatting
# ---------------------------------------------------------------------------

OPERATOR_HUMAN = {
    "<": "is less than",
    ">": "is greater than",
    "<=": "is at most",
    ">=": "is at least",
    "crosses_above": "crosses above",
    "crosses_below": "crosses below",
    "between": "is between",
}


def format_condition_human(condition: str) -> str:
    """Format a condition string into human-readable text.

    Examples:
        "rsi < 30"  ->  "RSI is less than 30"
        "ema_fast crosses_above ema_slow"  ->  "EMA Fast crosses above EMA Slow"
    """
    parts = condition.strip().strip('"').strip("'").split()
    if not parts:
        return condition

    def _humanize_name(name: str) -> str:
        """Convert indicator name to human-readable form."""
        return name.replace("_", " ").title()

    if len(parts) >= 4 and parts[1] == "between":
        return f"{_humanize_name(parts[0])} is between {parts[2]} and {parts[3]}"

    if len(parts) >= 3:
        left = _humanize_name(parts[0])
        op = OPERATOR_HUMAN.get(parts[1], parts[1])
        right = _humanize_name(parts[2]) if not _is_numeric(parts[2]) else parts[2]
        return f"{left} {op} {right}"

    return condition


def render_conditions_human_readable(
    conditions: list[str],
    label: str = "",
) -> None:
    """Render conditions as human-readable text instead of raw syntax."""
    if label:
        st.markdown(f"**{label}:**")
    if not conditions:
        st.caption("None defined")
        return
    for cond in conditions:
        human = format_condition_human(cond)
        st.markdown(f"- {human}")
