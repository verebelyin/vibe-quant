"""Step-by-step strategy creation wizard with progress indicator.

Provides a guided multi-step flow for creating trading strategies,
making it easier for new users to build well-structured strategies.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
import yaml

from vibe_quant.dashboard.components.condition_builder import render_condition_builder
from vibe_quant.dashboard.components.form_state import (
    build_dsl_from_form,
    cleanup_form_state,
    init_form_state,
    sync_form_state,
)
from vibe_quant.dashboard.components.indicator_catalog import (
    render_indicator_card,
    render_indicator_selector,
)
from vibe_quant.dashboard.components.risk_management import render_risk_section
from vibe_quant.dashboard.components.template_selector import render_template_selector
from vibe_quant.dashboard.components.time_filters import render_time_filters_section
from vibe_quant.dashboard.components.validation_summary import render_validation_summary
from vibe_quant.dsl.indicator_metadata import suggest_indicator_name
from vibe_quant.dsl.schema import VALID_TIMEFRAMES, StrategyDSL

WIZARD_STEPS = [
    ("Template", "Choose a starting point"),
    ("Basic Info", "Name, timeframe, description"),
    ("Indicators", "Add technical indicators"),
    ("Rules", "Entry & exit conditions"),
    ("Risk", "Stop loss & take profit"),
    ("Review", "Validate & save"),
]


def _render_progress_bar(current_step: int) -> None:
    """Render wizard progress indicator."""
    total = len(WIZARD_STEPS)
    progress = (current_step + 1) / total

    # Visual step indicator
    cols = st.columns(total)
    for i, (label, _desc) in enumerate(WIZARD_STEPS):
        with cols[i]:
            if i < current_step:
                st.markdown(f":green[**{i+1}. {label}**]")
            elif i == current_step:
                st.markdown(f":blue[**{i+1}. {label}**]")
            else:
                st.markdown(f":gray[{i+1}. {label}]")

    st.progress(progress, text=f"Step {current_step + 1}/{total}: {WIZARD_STEPS[current_step][1]}")


def _render_nav_buttons(current_step: int) -> tuple[bool, bool]:
    """Render Back/Next navigation buttons. Returns (go_back, go_next)."""
    c1, c2, c3 = st.columns([1, 2, 1])
    go_back = False
    go_next = False

    with c1:
        if current_step > 0:
            go_back = st.button("Back", use_container_width=True)
    with c3:
        if current_step < len(WIZARD_STEPS) - 1:
            go_next = st.button("Next", type="primary", use_container_width=True)

    return go_back, go_next


def _get_default_wizard_dsl() -> dict[str, Any]:
    """Return default DSL for wizard."""
    return {
        "name": "my_strategy",
        "description": "",
        "version": 1,
        "timeframe": "1h",
        "additional_timeframes": [],
        "indicators": {},
        "entry_conditions": {"long": [], "short": []},
        "exit_conditions": {"long": [], "short": []},
        "time_filters": {
            "allowed_sessions": [],
            "blocked_days": [],
            "avoid_around_funding": {"enabled": False},
        },
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "risk_reward", "risk_reward_ratio": 2.0},
        "position_management": {"scale_in": {"enabled": False}, "partial_exit": {"enabled": False}},
        "sweep": {},
    }


def _step_template() -> None:
    """Step 0: Template selection."""
    st.subheader("Choose a starting point")
    st.caption("Pick a template for common strategy patterns, or start from scratch.")

    template_yaml = render_template_selector()
    if template_yaml is not None:
        dsl = yaml.safe_load(template_yaml)
        if isinstance(dsl, dict):
            st.session_state["wizard_dsl"] = dsl
            sync_form_state(dsl)
            st.session_state["wizard_step"] = 1
            st.rerun()

    st.divider()
    if st.button("Start from scratch", use_container_width=False):
        dsl = _get_default_wizard_dsl()
        st.session_state["wizard_dsl"] = dsl
        init_form_state(dsl)
        st.session_state["wizard_step"] = 1
        st.rerun()


def _step_basic_info() -> None:
    """Step 1: Basic information."""
    st.subheader("Basic Information")
    dsl = st.session_state.get("wizard_dsl", _get_default_wizard_dsl())

    st.text_input(
        "Strategy Name",
        value=dsl.get("name", ""),
        key="form_name",
        help="Lowercase, letters/numbers/underscores, starts with letter",
    )
    st.text_area(
        "Description",
        value=dsl.get("description", ""),
        key="form_description",
        height=80,
        help="What does this strategy do? What market conditions does it target?",
    )

    c1, c2 = st.columns(2)
    with c1:
        tf_list = sorted(VALID_TIMEFRAMES)
        current_tf = dsl.get("timeframe", "1h")
        st.selectbox(
            "Primary Timeframe",
            options=tf_list,
            index=tf_list.index(current_tf) if current_tf in tf_list else 0,
            key="form_timeframe",
            help="Main candle timeframe for your strategy",
        )
    with c2:
        st.number_input("Version", value=dsl.get("version", 1), min_value=1, max_value=1000, key="form_version")

    st.multiselect(
        "Additional Timeframes (optional)",
        options=[tf for tf in sorted(VALID_TIMEFRAMES)
                 if tf != st.session_state.get("form_timeframe", "1h")],
        default=[tf for tf in dsl.get("additional_timeframes", [])
                 if tf != st.session_state.get("form_timeframe", "1h")],
        key="form_additional_tfs",
    )

    # Tip
    with st.container(border=True):
        st.caption(
            "**Tip:** Start with a single timeframe (1h or 4h) for simplicity. "
            "Multi-timeframe adds power but also complexity and overfitting risk."
        )


def _step_indicators() -> None:
    """Step 2: Indicator configuration."""
    st.subheader("Technical Indicators")
    st.caption("Add the indicators your strategy will use for signals.")

    indicators: dict[str, dict[str, Any]] = st.session_state.get("form_indicators", {})
    if not indicators:
        st.info("No indicators added yet. Choose from the catalog below.")
    else:
        st.caption(f"{len(indicators)} indicator(s) defined")

    all_tfs = sorted(VALID_TIMEFRAMES)
    to_remove: list[str] = []
    to_add: list[tuple[str, dict[str, Any]]] = []

    for name, config in list(indicators.items()):
        with st.expander(f"`{name}` ({config.get('type', '?')})", expanded=False):
            updated, should_remove, should_dup = render_indicator_card(
                name=name, config=config, key_prefix="wiz_ind", all_timeframes=all_tfs,
            )
            if should_remove:
                to_remove.append(name)
            elif should_dup:
                new_name = suggest_indicator_name(config.get("type", ""), set(indicators.keys()))
                to_add.append((new_name, dict(config)))
            elif updated is not None:
                indicators[name] = updated

    if to_remove or to_add:
        for name in to_remove:
            indicators.pop(name, None)
        for name, config in to_add:
            indicators[name] = config
        st.session_state["form_indicators"] = indicators
        st.rerun()

    st.divider()
    if st.session_state.get("show_indicator_catalog"):
        result = render_indicator_selector(
            key_prefix="wiz_add_ind", existing_indicator_names=set(indicators.keys()),
        )
        if result is not None:
            indicators[result["name"]] = result["config"]
            st.session_state["form_indicators"] = indicators
            st.session_state["show_indicator_catalog"] = False
            st.rerun()
    elif st.button("+ Add Indicator", type="primary"):
        st.session_state["show_indicator_catalog"] = True
        st.rerun()

    # Tip
    if not indicators:
        with st.container(border=True):
            st.caption(
                "**Suggested starting set:** RSI (momentum), EMA pair (trend), ATR (volatility). "
                "These cover the three key aspects of price behavior."
            )


def _step_rules() -> None:
    """Step 3: Entry & exit conditions."""
    st.subheader("Entry & Exit Rules")
    st.caption("Define when to enter and exit trades.")

    indicator_names = list(st.session_state.get("form_indicators", {}).keys())

    if not indicator_names:
        st.warning("No indicators defined. Go back and add indicators first.")
        return

    tab_entry, tab_exit = st.tabs(["Entry Conditions", "Exit Conditions"])
    with tab_entry:
        c1, c2 = st.columns(2)
        with c1:
            st.session_state["form_entry_long"] = render_condition_builder(
                "Long Entry", st.session_state.get("form_entry_long", []),
                indicator_names, "wiz_cond_entry_long")
        with c2:
            st.session_state["form_entry_short"] = render_condition_builder(
                "Short Entry", st.session_state.get("form_entry_short", []),
                indicator_names, "wiz_cond_entry_short")
    with tab_exit:
        c1, c2 = st.columns(2)
        with c1:
            st.session_state["form_exit_long"] = render_condition_builder(
                "Long Exit", st.session_state.get("form_exit_long", []),
                indicator_names, "wiz_cond_exit_long")
        with c2:
            st.session_state["form_exit_short"] = render_condition_builder(
                "Short Exit", st.session_state.get("form_exit_short", []),
                indicator_names, "wiz_cond_exit_short")

    # Tip
    with st.container(border=True):
        st.caption(
            "**Tip:** Use at least 2 entry conditions (trend filter + signal) to reduce "
            "false signals. Always define exit conditions -- don't rely only on stop loss."
        )


def _step_risk() -> None:
    """Step 4: Risk management."""
    st.subheader("Risk Management")
    st.caption("Configure stop loss and take profit levels.")

    dsl = st.session_state.get("wizard_dsl", _get_default_wizard_dsl())
    render_risk_section(dsl)

    st.divider()
    render_time_filters_section(dsl)


def _step_review(manager: Any) -> bool:
    """Step 5: Review and save. Returns True if saved."""
    st.subheader("Review & Save")

    dsl = st.session_state.get("wizard_dsl", _get_default_wizard_dsl())
    new_dsl = build_dsl_from_form(dsl)

    # Show YAML preview
    with st.expander("Strategy YAML Preview", expanded=False):
        st.code(yaml.dump(new_dsl, default_flow_style=False, sort_keys=False), language="yaml")

    # Validate
    from pydantic import ValidationError

    try:
        model = StrategyDSL.model_validate(new_dsl)
        error = None
    except ValidationError as e:
        model = None
        errors = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            errors.append(f"{loc}: {err['msg']}")
        error = "\n".join(errors)

    render_validation_summary(model, error)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if (st.button("Save Strategy", type="primary", use_container_width=True, disabled=model is None)
                and model):
            dumped = model.model_dump()
            manager.create_strategy(name=model.name, dsl_config=dumped, description=model.description)
            st.success(f"Created strategy '{model.name}'")
            # Cleanup wizard state
            cleanup_form_state()
            st.session_state.pop("wizard_step", None)
            st.session_state.pop("wizard_dsl", None)
            st.session_state.pop("wizard_active", None)
            st.session_state.show_editor = False
            return True
    with c2:
        if st.button("Edit in Full Editor", use_container_width=True):
            yaml_key = "yaml_content_new"
            st.session_state[yaml_key] = yaml.dump(new_dsl, default_flow_style=False, sort_keys=False)
            st.session_state["template_applied"] = True
            st.session_state.pop("wizard_step", None)
            st.session_state.pop("wizard_active", None)
            st.session_state["editor_mode"] = "YAML"
            st.rerun()

    return False


def render_strategy_wizard(manager: Any) -> bool:
    """Render the full strategy creation wizard.

    Args:
        manager: StateManager instance for saving strategies.

    Returns:
        True if a strategy was saved, False otherwise.
    """
    # Initialize wizard state
    if "wizard_step" not in st.session_state:
        st.session_state["wizard_step"] = 0
    if "wizard_dsl" not in st.session_state:
        st.session_state["wizard_dsl"] = _get_default_wizard_dsl()

    current_step = st.session_state["wizard_step"]

    # Render progress
    _render_progress_bar(current_step)
    st.divider()

    # Render current step
    saved = False
    if current_step == 0:
        _step_template()
    elif current_step == 1:
        _step_basic_info()
    elif current_step == 2:
        _step_indicators()
    elif current_step == 3:
        _step_rules()
    elif current_step == 4:
        _step_risk()
    elif current_step == 5:
        saved = _step_review(manager)
        if saved:
            return True

    # Navigation (not shown for template step or after save)
    if current_step > 0:
        st.divider()
        go_back, go_next = _render_nav_buttons(current_step)
        if go_back:
            st.session_state["wizard_step"] = max(0, current_step - 1)
            st.rerun()
        if go_next:
            # Save intermediate state before advancing
            dsl = st.session_state.get("wizard_dsl", _get_default_wizard_dsl())
            updated_dsl = build_dsl_from_form(dsl)
            st.session_state["wizard_dsl"] = updated_dsl
            st.session_state["wizard_step"] = min(len(WIZARD_STEPS) - 1, current_step + 1)
            st.rerun()

    return False
