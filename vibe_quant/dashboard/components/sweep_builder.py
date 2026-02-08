"""Structured sweep parameter builder component.

Replaces raw YAML textarea with:
- Per-parameter value list editors with sliders
- Total combination counter with color-coded warnings
- Quick presets (Narrow/Medium/Wide)
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from vibe_quant.dsl.indicator_metadata import INDICATOR_CATALOG


def render_sweep_builder(
    sweep_config: dict[str, list[int | float]],
    indicators: dict[str, dict[str, Any]],
    key_prefix: str,
) -> dict[str, list[int | float]]:
    """Render structured sweep parameter builder.

    Args:
        sweep_config: Current sweep configuration from DSL
        indicators: Indicator configs from the strategy
        key_prefix: Unique key prefix for widgets

    Returns:
        Updated sweep configuration dict
    """
    # Initialize sweep state
    state_key = f"{key_prefix}_sweep_params"
    if state_key not in st.session_state:
        st.session_state[state_key] = dict(sweep_config) if sweep_config else {}

    sweep = st.session_state[state_key]

    # Header with quick presets
    col_title, col_presets = st.columns([2, 3])
    with col_title:
        st.markdown("**Sweep Parameters**")
    with col_presets:
        p1, p2, p3 = st.columns(3)
        with p1:
            if st.button("Narrow (3)", key=f"{key_prefix}_preset_narrow",
                          help="3 values per parameter"):
                sweep = _apply_preset(sweep, indicators, 3)
                st.session_state[state_key] = sweep
                st.rerun()
        with p2:
            if st.button("Medium (5)", key=f"{key_prefix}_preset_medium",
                          help="5 values per parameter"):
                sweep = _apply_preset(sweep, indicators, 5)
                st.session_state[state_key] = sweep
                st.rerun()
        with p3:
            if st.button("Wide (10)", key=f"{key_prefix}_preset_wide",
                          help="10 values per parameter"):
                sweep = _apply_preset(sweep, indicators, 10)
                st.session_state[state_key] = sweep
                st.rerun()

    if not sweep:
        st.info("No sweep parameters. Add parameters to optimize below.")

    # Render existing sweep params
    params_to_remove: list[str] = []
    updated_sweep: dict[str, list[int | float]] = {}

    for param_name, values in sweep.items():
        result = _render_sweep_param_row(param_name, values, key_prefix)
        if result is None:
            params_to_remove.append(param_name)
        else:
            updated_sweep[param_name] = result

    # Remove flagged params
    if params_to_remove:
        for p in params_to_remove:
            st.session_state[state_key].pop(p, None)
        st.rerun()

    # Add parameter button
    st.divider()
    _render_add_sweep_param(indicators, sweep, key_prefix)

    # Total combinations display
    if updated_sweep:
        total = 1
        for values in updated_sweep.values():
            total *= max(len(values), 1)
        _render_combination_counter(total)

    st.session_state[state_key] = updated_sweep
    return updated_sweep


def _render_sweep_param_row(
    param_name: str,
    values: list[int | float],
    key_prefix: str,
) -> list[int | float] | None:
    """Render a single sweep parameter row. Returns None if should remove."""
    col1, col2, col3 = st.columns([2, 4, 1])

    with col1:
        st.markdown(f"`{param_name}`")
        st.caption(f"{len(values)} values")

    with col2:
        values_str = st.text_input(
            f"Values for {param_name}",
            value=", ".join(str(v) for v in values),
            key=f"{key_prefix}_sweep_vals_{param_name}",
            label_visibility="collapsed",
            help="Comma-separated values to test",
        )
        try:
            parsed = []
            for v in values_str.split(","):
                v = v.strip()
                if not v:
                    continue
                if "." in v:
                    parsed.append(float(v))
                else:
                    parsed.append(int(v))
            return parsed if parsed else values
        except ValueError:
            st.error(f"Invalid values for {param_name}")
            return values

    with col3:
        if st.button("X", key=f"{key_prefix}_sweep_rm_{param_name}",
                      help=f"Remove {param_name} from sweep"):
            return None

    return values


def _render_add_sweep_param(
    indicators: dict[str, dict[str, Any]],
    current_sweep: dict[str, list[int | float]],
    key_prefix: str,
) -> None:
    """Render the add sweep parameter section."""
    # Build list of sweepable parameters from indicator configs
    sweepable: list[tuple[str, str, int | float]] = []
    for ind_name, ind_config in indicators.items():
        ind_type = ind_config.get("type", "")
        meta = INDICATOR_CATALOG.get(ind_type)
        if meta:
            for param in meta.params:
                param_path = f"{ind_name}.{param.name}"
                if param_path not in current_sweep:
                    sweepable.append((param_path, f"{ind_name} → {param.label}", param.default))
        else:
            # Fallback for unknown types
            period = ind_config.get("period")
            if period is not None:
                param_path = f"{ind_name}.period"
                if param_path not in current_sweep:
                    sweepable.append((param_path, f"{ind_name} → Period", period))

    if not sweepable:
        st.caption("All indicator parameters are already in the sweep.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        options = [f"{label} (default: {default})" for _, label, default in sweepable]
        selected_idx = st.selectbox(
            "Add parameter to sweep",
            options=range(len(options)),
            format_func=lambda i: options[i],
            key=f"{key_prefix}_add_sweep_sel",
            label_visibility="collapsed",
        )
    with col2:
        if (
            st.button("+ Add", key=f"{key_prefix}_add_sweep_btn", type="primary")
            and selected_idx is not None
        ):
                path, _, default = sweepable[selected_idx]
                # Generate sensible default sweep values
                default_values = _generate_default_sweep_values(default)
                current_sweep[path] = default_values
                st.rerun()


def _render_combination_counter(total: int) -> None:
    """Render the total parameter combinations with color-coded warning."""
    if total <= 100:
        color = "green"
        label = "Good"
    elif total <= 1000:
        color = "orange"
        label = "Moderate"
    else:
        color = "red"
        label = "Large"

    st.markdown(
        f"**Total combinations:** :{color}[**{total:,}**] "
        f"({label})"
    )
    if total > 1000:
        st.warning(
            f"Testing {total:,} combinations may take a long time. "
            "Consider reducing parameter ranges or using genetic optimization."
        )
    elif total > 5000:
        st.error(
            f"Testing {total:,} combinations is very expensive. "
            "Strongly consider narrowing sweep ranges."
        )


def _generate_default_sweep_values(default: int | float) -> list[int | float]:
    """Generate sensible default sweep values around the default value."""
    if isinstance(default, float):
        step = max(0.1, round(default * 0.25, 1))
        return [
            round(default - step, 2),
            round(default, 2),
            round(default + step, 2),
        ]
    else:
        step = max(1, int(default * 0.3))
        return [
            max(1, default - step),
            default,
            default + step,
        ]


def _apply_preset(
    current: dict[str, list[int | float]],
    indicators: dict[str, dict[str, Any]],
    num_values: int,
) -> dict[str, list[int | float]]:
    """Apply a preset by expanding or contracting existing sweep params."""
    result: dict[str, list[int | float]] = {}

    for param_path, values in current.items():
        if not values:
            continue
        min_val = min(values)
        max_val = max(values)
        if min_val == max_val:
            # Expand around the single value
            if isinstance(min_val, float):
                step = max(0.1, min_val * 0.1)
                min_val = round(min_val - step * 2, 2)
                max_val = round(min_val + step * (num_values - 1), 2)
            else:
                step = max(1, int(min_val * 0.1))
                min_val = max(1, int(min_val - step * 2))
                max_val = int(min_val + step * (num_values - 1))

        if isinstance(min_val, float):
            step = round((max_val - min_val) / max(1, num_values - 1), 2)
            result[param_path] = [
                round(min_val + step * i, 2) for i in range(num_values)
            ]
        else:
            step = max(1, int((max_val - min_val) / max(1, num_values - 1)))
            result[param_path] = [
                int(min_val + step * i) for i in range(num_values)
            ]

    return result
