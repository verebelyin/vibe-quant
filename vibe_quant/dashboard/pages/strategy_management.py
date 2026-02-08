"""Strategy Management Tab for vibe-quant dashboard.

Provides CRUD operations for trading strategies with:
- Strategy list with search/filter and strategy cards
- Template selector with categorized card grid
- Visual form editor (indicators, conditions, risk, time, sweep)
- Raw YAML editor with live validation preview
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import streamlit as st
import yaml
from pydantic import ValidationError

from vibe_quant.dashboard.components.condition_builder import render_condition_builder
from vibe_quant.dashboard.components.form_state import (
    build_dsl_from_form,
    cleanup_form_state,
    init_form_state,
)
from vibe_quant.dashboard.components.indicator_catalog import (
    render_indicator_card,
    render_indicator_selector,
)
from vibe_quant.dashboard.components.risk_management import render_risk_section
from vibe_quant.dashboard.components.strategy_card import render_strategy_card
from vibe_quant.dashboard.components.sweep_builder import render_sweep_builder
from vibe_quant.dashboard.components.template_selector import render_template_selector
from vibe_quant.dashboard.components.time_filters import render_time_filters_section
from vibe_quant.dashboard.components.validation_summary import render_validation_summary
from vibe_quant.dashboard.utils import get_state_manager
from vibe_quant.dsl.indicator_metadata import suggest_indicator_name
from vibe_quant.dsl.schema import VALID_TIMEFRAMES, StrategyDSL

if TYPE_CHECKING:
    from pathlib import Path

    from vibe_quant.db.state_manager import StateManager

# ── Helpers ────────────────────────────────────────────────────────────────


def _format_validation_errors(e: ValidationError) -> str:
    """Format pydantic ValidationError into a human-readable string."""
    errors = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"])
        errors.append(f"{loc}: {err['msg']}")
    return "\n".join(errors)


def _validate_dsl(yaml_str: str) -> tuple[StrategyDSL | None, str | None]:
    """Validate DSL YAML string. Returns (model, error)."""
    try:
        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            return None, "YAML must be a mapping"
        return StrategyDSL.model_validate(data), None
    except yaml.YAMLError as e:
        return None, f"YAML parse error: {e}"
    except ValidationError as e:
        return None, _format_validation_errors(e)


def _validate_dsl_dict(dsl_dict: dict[str, Any]) -> tuple[StrategyDSL | None, str | None]:
    """Validate a DSL dict directly (avoids YAML round-trip)."""
    try:
        return StrategyDSL.model_validate(dsl_dict), None
    except ValidationError as e:
        return None, _format_validation_errors(e)


def _get_default_dsl_yaml() -> str:
    """Return default DSL YAML template."""
    return yaml.dump({
        "name": "my_strategy",
        "description": "A simple RSI-based strategy",
        "version": 1,
        "timeframe": "1h",
        "additional_timeframes": [],
        "indicators": {
            "rsi_14": {"type": "RSI", "period": 14, "source": "close"},
            "atr_14": {"type": "ATR", "period": 14},
        },
        "entry_conditions": {"long": ["rsi_14 < 30"], "short": ["rsi_14 > 70"]},
        "exit_conditions": {"long": ["rsi_14 > 70"], "short": ["rsi_14 < 30"]},
        "time_filters": {
            "allowed_sessions": [], "blocked_days": [],
            "avoid_around_funding": {"enabled": False},
        },
        "stop_loss": {"type": "atr_fixed", "atr_multiplier": 2.0, "indicator": "atr_14"},
        "take_profit": {"type": "risk_reward", "risk_reward_ratio": 2.0},
        "position_management": {"scale_in": {"enabled": False}, "partial_exit": {"enabled": False}},
        "sweep": {},
    }, default_flow_style=False, sort_keys=False)


def _save_strategy(
    manager: StateManager,
    dsl_dict: dict[str, Any],
    existing: dict[str, Any] | None,
) -> bool:
    """Validate and save strategy. Returns True on success."""
    model, error = _validate_dsl_dict(dsl_dict)
    if error:
        st.error(f"Validation failed:\n{error}")
        return False
    if model is None:
        st.error("Validation returned no model")
        return False

    dumped = model.model_dump()
    if existing:
        manager.update_strategy(existing["id"], dsl_config=dumped, description=model.description)
        st.success(f"Updated strategy '{model.name}'")
    else:
        manager.create_strategy(name=model.name, dsl_config=dumped, description=model.description)
        st.success(f"Created strategy '{model.name}'")

    st.session_state.show_editor = False
    st.session_state.editing_strategy_id = None
    return True




# ── Strategy List ──────────────────────────────────────────────────────────


def render_strategy_list(
    manager: StateManager, search_query: str, show_inactive: bool,
) -> None:
    """Render strategy list with search/filter."""
    strategies = manager.list_strategies(active_only=not show_inactive)
    if search_query:
        q = search_query.lower()
        strategies = [
            s for s in strategies
            if q in s["name"].lower()
            or (s.get("description") and q in s["description"].lower())
        ]
    if not strategies:
        st.info("No strategies found. Create one using the **New Strategy** button above.")
        return
    for strategy in strategies:
        render_strategy_card(manager, strategy)


def render_delete_confirmation(manager: StateManager) -> None:
    """Render delete confirmation dialog."""
    sid = st.session_state.get("confirm_delete_id")
    sname = st.session_state.get("confirm_delete_name")
    if not sid:
        return
    st.warning(f"Delete strategy '{sname}'? This cannot be undone.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Confirm Delete"):
            manager.update_strategy(sid, is_active=False)
            st.success(f"Deleted strategy '{sname}'")
            st.session_state.confirm_delete_id = None
            st.session_state.confirm_delete_name = None
            st.rerun()
    with c2:
        if st.button("Cancel Delete"):
            st.session_state.confirm_delete_id = None
            st.session_state.confirm_delete_name = None
            st.rerun()


# ── Editor Entrypoint ─────────────────────────────────────────────────────


def render_strategy_editor(
    manager: StateManager, strategy_id: int | None = None,
) -> None:
    """Render strategy create/edit with template selector and multi-mode editor."""
    existing = None
    if strategy_id:
        existing = manager.get_strategy(strategy_id)
        if not existing:
            st.error(f"Strategy {strategy_id} not found")
            return

    st.subheader("Edit Strategy" if existing else "Create Strategy")

    yaml_key = f"yaml_content_{strategy_id or 'new'}"
    if yaml_key not in st.session_state:
        if existing:
            st.session_state[yaml_key] = yaml.dump(
                existing["dsl_config"], default_flow_style=False, sort_keys=False,
            )
        else:
            st.session_state[yaml_key] = _get_default_dsl_yaml()

    # Template selector (new strategies only)
    if not existing and not st.session_state.get("template_applied"):
        template_yaml = render_template_selector()
        if template_yaml is not None:
            st.session_state[yaml_key] = template_yaml
            st.session_state["template_applied"] = True
            st.rerun()
        st.divider()
        col_blank, _ = st.columns([1, 3])
        with col_blank:
            if st.button("Start from scratch", use_container_width=True):
                st.session_state["template_applied"] = True
                st.rerun()
        return

    # Mode selector
    col_mode, col_cancel = st.columns([3, 1])
    with col_mode:
        mode = st.radio(
            "Editor mode", ["Visual", "YAML"], horizontal=True,
            key="editor_mode", label_visibility="collapsed",
        )
    with col_cancel:
        if st.button("Back to list"):
            st.session_state.show_editor = False
            st.session_state.editing_strategy_id = None
            st.session_state.pop("template_applied", None)
            cleanup_form_state()
            st.rerun()

    st.divider()
    if mode == "YAML":
        _render_yaml_editor(manager, existing, yaml_key)
    else:
        _render_form_editor(manager, existing, yaml_key)


# ── YAML Editor ───────────────────────────────────────────────────────────


def _render_yaml_editor(
    manager: StateManager, existing: dict[str, Any] | None, yaml_key: str,
) -> None:
    """Render YAML editor with live validation panel."""
    col_editor, col_preview = st.columns([3, 2])
    with col_editor:
        yaml_content = st.text_area(
            "Strategy DSL (YAML)", value=st.session_state[yaml_key],
            height=550, key=f"yaml_editor_{yaml_key}",
        )
        uploaded = st.file_uploader("Or upload YAML file", type=["yaml", "yml"], key="yaml_upload")
        if uploaded:
            yaml_content = uploaded.read().decode("utf-8")
            st.session_state[yaml_key] = yaml_content
            st.rerun()

    with col_preview:
        st.markdown("**Validation Preview**")
        model, error = _validate_dsl(yaml_content)
        render_validation_summary(model, error)

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Save", type="primary", use_container_width=True):
            model, error = _validate_dsl(yaml_content)
            if error:
                st.error(f"Validation failed:\n{error}")
            elif model and _save_strategy(manager, model.model_dump(), existing):
                st.session_state.pop("template_applied", None)
                st.rerun()
    with c2:
        if st.button("Copy to Visual Editor", use_container_width=True):
            st.session_state[yaml_key] = yaml_content
            st.session_state["editor_mode"] = "Visual"
            st.rerun()
    with c3:
        if st.button("Reset", use_container_width=True):
            if existing:
                st.session_state[yaml_key] = yaml.dump(
                    existing["dsl_config"], default_flow_style=False, sort_keys=False,
                )
            else:
                st.session_state[yaml_key] = _get_default_dsl_yaml()
            st.rerun()


# ── Visual Form Editor ────────────────────────────────────────────────────


def _render_form_editor(
    manager: StateManager, existing: dict[str, Any] | None, yaml_key: str,
) -> None:
    """Render the visual form editor with all components."""
    try:
        dsl = yaml.safe_load(st.session_state[yaml_key])
        if not isinstance(dsl, dict):
            dsl = yaml.safe_load(_get_default_dsl_yaml())
    except yaml.YAMLError:
        dsl = yaml.safe_load(_get_default_dsl_yaml())

    init_form_state(dsl)

    with st.expander("**1. Basic Info**", expanded=True):
        _render_basic_info_section(dsl)
    with st.expander("**2. Indicators**", expanded=True):
        _render_indicators_section()
    with st.expander("**3. Entry & Exit Rules**", expanded=True):
        _render_conditions_section()
    with st.expander("**4. Risk Management** (Stop Loss & Take Profit)", expanded=True):
        render_risk_section(dsl)
    with st.expander("**5. Time Filters**", expanded=False):
        render_time_filters_section(dsl)
    with st.expander("**6. Parameter Sweep**", expanded=False):
        render_sweep_builder(
            sweep_config=dsl.get("sweep", {}),
            indicators=st.session_state.get("form_indicators", {}),
            key_prefix="sweep",
        )

    st.divider()
    _render_save_section(manager, existing, yaml_key, dsl)



def _render_basic_info_section(dsl: dict[str, Any]) -> None:
    """Render basic info: name, description, version, timeframes."""
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        st.text_input(
            "Name", value=dsl.get("name", ""), key="form_name",
            help="Lowercase, letters/numbers/underscores, starts with letter",
        )
    with c2:
        st.number_input("Version", value=dsl.get("version", 1), min_value=1,
                         max_value=1000, key="form_version")
    with c3:
        tf_list = sorted(VALID_TIMEFRAMES)
        current_tf = dsl.get("timeframe", "1h")
        st.selectbox("Primary Timeframe", options=tf_list,
                      index=tf_list.index(current_tf) if current_tf in tf_list else 0,
                      key="form_timeframe")

    st.text_area("Description", value=dsl.get("description", ""),
                  key="form_description", height=68)
    st.multiselect(
        "Additional Timeframes (for multi-TF strategies)",
        options=[tf for tf in sorted(VALID_TIMEFRAMES)
                 if tf != st.session_state.get("form_timeframe", "1h")],
        default=[tf for tf in dsl.get("additional_timeframes", [])
                 if tf != st.session_state.get("form_timeframe", "1h")],
        key="form_additional_tfs",
    )


def _render_indicators_section() -> None:
    """Render indicators section with catalog, add/remove/duplicate."""
    indicators: dict[str, dict[str, Any]] = st.session_state["form_indicators"]
    if not indicators:
        st.info("No indicators defined. Add one below.")
    else:
        st.caption(f"{len(indicators)} indicator(s) defined")

    all_tfs = sorted(VALID_TIMEFRAMES)
    to_remove: list[str] = []
    to_add: list[tuple[str, dict[str, Any]]] = []

    for name, config in list(indicators.items()):
        with st.expander(f"`{name}` ({config.get('type', '?')})", expanded=False):
            updated, should_remove, should_dup = render_indicator_card(
                name=name, config=config, key_prefix="form_ind", all_timeframes=all_tfs,
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
            key_prefix="form_add_ind", existing_indicator_names=set(indicators.keys()),
        )
        if result is not None:
            indicators[result["name"]] = result["config"]
            st.session_state["form_indicators"] = indicators
            st.session_state["show_indicator_catalog"] = False
            st.rerun()
    elif st.button("+ Add Indicator", type="primary"):
        st.session_state["show_indicator_catalog"] = True
        st.rerun()


def _render_conditions_section() -> None:
    """Render entry/exit conditions using visual condition builder."""
    indicator_names = list(st.session_state.get("form_indicators", {}).keys())
    tab_entry, tab_exit = st.tabs(["Entry Conditions", "Exit Conditions"])
    with tab_entry:
        c1, c2 = st.columns(2)
        with c1:
            st.session_state["form_entry_long"] = render_condition_builder(
                "Long Entry", st.session_state.get("form_entry_long", []),
                indicator_names, "cond_entry_long")
        with c2:
            st.session_state["form_entry_short"] = render_condition_builder(
                "Short Entry", st.session_state.get("form_entry_short", []),
                indicator_names, "cond_entry_short")
    with tab_exit:
        c1, c2 = st.columns(2)
        with c1:
            st.session_state["form_exit_long"] = render_condition_builder(
                "Long Exit", st.session_state.get("form_exit_long", []),
                indicator_names, "cond_exit_long")
        with c2:
            st.session_state["form_exit_short"] = render_condition_builder(
                "Short Exit", st.session_state.get("form_exit_short", []),
                indicator_names, "cond_exit_short")


def _render_save_section(
    manager: StateManager, existing: dict[str, Any] | None,
    yaml_key: str, original_dsl: dict[str, Any],
) -> None:
    """Render validation, save, and action buttons."""
    new_dsl = build_dsl_from_form(original_dsl)
    model, error = _validate_dsl_dict(new_dsl)
    render_validation_summary(model, error)

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if (st.button("Save Strategy", type="primary", use_container_width=True,
                       disabled=model is None)
                and model and _save_strategy(manager, model.model_dump(), existing)):
            cleanup_form_state()
            st.rerun()
    with c2:
        if st.button("View as YAML", use_container_width=True):
            st.session_state[yaml_key] = yaml.dump(new_dsl, default_flow_style=False, sort_keys=False)
            st.session_state["editor_mode"] = "YAML"
            cleanup_form_state()
            st.rerun()
    with c3:
        if st.button("Reset", use_container_width=True):
            cleanup_form_state()
            st.rerun()



# ── Main Entry Point ──────────────────────────────────────────────────────


def render_strategy_management_tab(db_path: Path | None = None) -> None:
    """Main entry point for strategy management tab."""
    st.header("Strategy Management")
    manager = get_state_manager(db_path)

    if st.session_state.get("confirm_delete_id"):
        render_delete_confirmation(manager)
        return

    if st.session_state.get("show_editor"):
        render_strategy_editor(manager, strategy_id=st.session_state.get("editing_strategy_id"))
        return

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        search_query = st.text_input(
            "Search strategies", placeholder="Name or description...",
            label_visibility="collapsed",
        )
    with c2:
        show_inactive = st.checkbox("Show inactive")
    with c3:
        if st.button("New Strategy", type="primary", use_container_width=True):
            st.session_state.show_editor = True
            st.session_state.editing_strategy_id = None
            st.session_state.pop("template_applied", None)
            st.rerun()

    st.divider()
    render_strategy_list(manager, search_query, show_inactive)


# Top-level call for st.navigation API
render_strategy_management_tab()
