"""Searchable, categorized indicator catalog component.

Replaces flat dropdowns with a rich selection experience:
- Category tabs (Trend, Momentum, Volatility, Volume)
- Search by name and description
- Popular indicators section
- Auto-populated parameter fields per indicator type
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from vibe_quant.dsl.indicator_metadata import (
    INDICATOR_CATALOG,
    IndicatorMeta,
    IndicatorParam,
    get_indicators_by_category,
    get_popular_indicators,
    suggest_indicator_name,
)
from vibe_quant.dsl.schema import VALID_SOURCES, VALID_TIMEFRAMES


def render_indicator_selector(
    key_prefix: str,
    existing_indicator_names: set[str],
) -> dict[str, Any] | None:
    """Render the indicator selector and return the selected indicator config.

    Returns None if cancelled, or a dict with keys:
        {"name": str, "config": {"type": ..., "period": ..., ...}}
    """
    search = st.text_input(
        "Search indicators",
        placeholder="Type to search (e.g., RSI, bollinger, trend...)",
        key=f"{key_prefix}_search",
    )

    by_category = get_indicators_by_category()
    popular = get_popular_indicators()

    # Filter by search
    if search:
        query = search.lower()
        filtered: dict[str, list[IndicatorMeta]] = {}
        for cat, indicators in by_category.items():
            matches = [
                m for m in indicators
                if query in m.type_name.lower()
                or query in m.display_name.lower()
                or query in m.description.lower()
                or query in m.use_case.lower()
            ]
            if matches:
                filtered[cat] = matches
        by_category = filtered

    # Popular quick-access (only when not searching)
    if not search and popular:
        st.caption("**Popular**")
        pop_cols = st.columns(len(popular))
        for i, meta in enumerate(popular):
            with pop_cols[i]:
                if st.button(
                    f"{meta.type_name}",
                    key=f"{key_prefix}_pop_{meta.type_name}",
                    help=meta.description[:80],
                    use_container_width=True,
                ):
                    st.session_state[f"{key_prefix}_selected_type"] = meta.type_name

    # Category tabs
    if by_category:
        tabs = st.tabs(list(by_category.keys()))
        for tab, (_cat, indicators) in zip(tabs, by_category.items(), strict=False):
            with tab:
                for meta in indicators:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{meta.display_name}** (`{meta.type_name}`)")
                        st.caption(meta.description[:120])
                    with col2:
                        if st.button(
                            "Select",
                            key=f"{key_prefix}_sel_{meta.type_name}",
                            use_container_width=True,
                        ):
                            st.session_state[f"{key_prefix}_selected_type"] = meta.type_name
    elif search:
        st.info(f"No indicators matching '{search}'")

    # If a type is selected, show configuration form
    selected_type = st.session_state.get(f"{key_prefix}_selected_type")
    if not selected_type:
        return None

    selected_meta = INDICATOR_CATALOG.get(selected_type)
    if selected_meta is None:
        return None

    st.divider()
    st.markdown(f"#### Configure {selected_meta.display_name}")
    st.caption(selected_meta.use_case)

    # Name input with auto-suggestion
    suggested = suggest_indicator_name(selected_type, existing_indicator_names)
    name = st.text_input(
        "Indicator name",
        value=suggested,
        key=f"{key_prefix}_name",
        help="Unique name for this indicator (lowercase, letters/numbers/underscores)",
    )

    # Source selector (if applicable)
    config: dict[str, Any] = {"type": selected_type}

    if selected_meta.source_required:
        source = st.selectbox(
            "Price source",
            options=sorted(VALID_SOURCES),
            index=sorted(VALID_SOURCES).index("close"),
            key=f"{key_prefix}_source",
        )
        config["source"] = source

    # Dynamic parameter fields based on indicator type
    for param in selected_meta.params:
        config[param.name] = _render_param_input(param, key_prefix)

    # Optional timeframe override
    tf_override = st.selectbox(
        "Timeframe override (optional)",
        options=["Use strategy primary"] + sorted(VALID_TIMEFRAMES),
        index=0,
        key=f"{key_prefix}_tf_override",
        help="Leave as default to use the strategy's primary timeframe",
    )
    if tf_override != "Use strategy primary":
        config["timeframe"] = tf_override

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add indicator", key=f"{key_prefix}_confirm", type="primary"):
            if not name:
                st.error("Please enter an indicator name")
                return None
            if name in existing_indicator_names:
                st.error(f"Indicator name '{name}' already exists. Choose a different name.")
                return None
            # Clear selection state
            st.session_state.pop(f"{key_prefix}_selected_type", None)
            return {"name": name, "config": config}
    with col2:
        if st.button("Cancel", key=f"{key_prefix}_cancel"):
            st.session_state.pop(f"{key_prefix}_selected_type", None)
            st.rerun()

    return None


def render_indicator_card(
    name: str,
    config: dict[str, Any],
    key_prefix: str,
    all_timeframes: list[str],
) -> tuple[dict[str, Any] | None, bool, bool]:
    """Render an indicator configuration card with edit/remove controls.

    Returns:
        (updated_config, should_remove, should_duplicate)
    """
    meta = INDICATOR_CATALOG.get(config.get("type", ""))
    display = meta.display_name if meta else config.get("type", "Unknown")

    col_header, col_actions = st.columns([3, 1])
    with col_header:
        st.markdown(f"**{name}** ({display})")
    with col_actions:
        c1, c2 = st.columns(2)
        with c1:
            remove = st.button("Remove", key=f"{key_prefix}_remove_{name}")
        with c2:
            duplicate = st.button("Copy", key=f"{key_prefix}_dup_{name}")

    if remove:
        return None, True, False
    if duplicate:
        return config, False, True

    # Editable parameters
    updated: dict[str, Any] = {"type": config["type"]}

    if meta and meta.source_required:
        source_options = sorted(VALID_SOURCES)
        current_source = config.get("source", "close")
        source_idx = source_options.index(current_source) if current_source in source_options else 0
        updated["source"] = st.selectbox(
            "Source",
            options=source_options,
            index=source_idx,
            key=f"{key_prefix}_src_{name}",
        )

    if meta:
        for param in meta.params:
            current = config.get(param.name, param.default)
            updated[param.name] = _render_param_input(
                param, f"{key_prefix}_{name}", current_value=current
            )
    else:
        # Fallback for unknown indicator types
        period = config.get("period")
        if period is not None:
            updated["period"] = st.number_input(
                "Period", value=period, min_value=1, max_value=500,
                key=f"{key_prefix}_period_{name}",
            )

    # Timeframe override
    tf_options = ["Use strategy primary"] + sorted(all_timeframes)
    current_tf = config.get("timeframe")
    tf_idx = tf_options.index(current_tf) if current_tf in tf_options else 0
    tf = st.selectbox(
        "Timeframe",
        options=tf_options,
        index=tf_idx,
        key=f"{key_prefix}_tf_{name}",
    )
    if tf != "Use strategy primary":
        updated["timeframe"] = tf

    return updated, False, False


def _render_param_input(
    param: IndicatorParam,
    key_prefix: str,
    current_value: int | float | None = None,
) -> int | float:
    """Render a single parameter input field."""
    val = current_value if current_value is not None else param.default
    if isinstance(param.default, float) or isinstance(val, float):
        return st.number_input(
            param.label,
            value=float(val),
            min_value=float(param.min_val),
            max_value=float(param.max_val),
            step=float(param.step),
            key=f"{key_prefix}_p_{param.name}",
            help=param.description,
        )
    return st.number_input(
        param.label,
        value=int(val),
        min_value=int(param.min_val),
        max_value=int(param.max_val),
        step=int(param.step),
        key=f"{key_prefix}_p_{param.name}",
        help=param.description,
    )
