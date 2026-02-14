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

from vibe_quant.dashboard.components.condition_builder import render_condition_builder
from vibe_quant.dashboard.components.form_state import (
    build_dsl_from_form,
    cleanup_form_state,
    init_form_state,
    sync_form_state,
    validate_dsl_dict,
    validate_dsl_yaml,
)
from vibe_quant.dashboard.components.indicator_catalog import (
    render_indicator_card,
    render_indicator_selector,
)
from vibe_quant.dashboard.components.risk_management import render_risk_section
from vibe_quant.dashboard.components.strategy_card import render_strategy_card
from vibe_quant.dashboard.components.strategy_wizard import render_strategy_wizard
from vibe_quant.dashboard.components.sweep_builder import render_sweep_builder
from vibe_quant.dashboard.components.template_selector import render_template_selector
from vibe_quant.dashboard.components.time_filters import render_time_filters_section
from vibe_quant.dashboard.components.validation_summary import render_validation_summary
from vibe_quant.dashboard.utils import get_state_manager
from vibe_quant.dsl.indicator_metadata import suggest_indicator_name
from vibe_quant.dsl.schema import VALID_TIMEFRAMES

if TYPE_CHECKING:
    from pathlib import Path

    from vibe_quant.db.state_manager import StateManager

# ── Helpers ────────────────────────────────────────────────────────────────

# Backward-compatible aliases for test compatibility
_validate_dsl = validate_dsl_yaml
_validate_dsl_dict = validate_dsl_dict


def _get_default_dsl_yaml() -> str:
    """Return default DSL YAML template."""
    return str(yaml.dump({
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
    }, default_flow_style=False, sort_keys=False))


def _save_strategy(
    manager: StateManager,
    dsl_dict: dict[str, Any],
    existing: dict[str, Any] | None,
) -> bool:
    """Validate and save strategy. Returns True on success."""
    model, error = validate_dsl_dict(dsl_dict)
    if error:
        st.error(f"Validation failed:\n{error}")
        return False
    if model is None:
        st.error("Validation returned no model")
        return False

    # Duplicate name check on create (or rename)
    if not existing or existing["name"] != model.name:
        dup = manager.get_strategy_by_name(model.name)
        if dup is not None:
            st.error(f"A strategy named '{model.name}' already exists. Choose a different name.")
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
    st.warning(f"Deactivate strategy '{sname}'? This will deactivate the strategy (it can be reactivated later).")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Confirm Deactivate"):
            manager.update_strategy(sid, is_active=False)
            st.success(f"Deactivated strategy '{sname}'")
            st.session_state.confirm_delete_id = None
            st.session_state.confirm_delete_name = None
            st.rerun()
    with c2:
        if st.button("Cancel"):
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
        if template_yaml == "":
            st.session_state.pop("editing_strategy", None)
            st.session_state.pop("template_applied", None)
            st.rerun()
        if template_yaml is not None:
            st.session_state[yaml_key] = template_yaml
            st.session_state["template_applied"] = True
            st.rerun()
        st.divider()
        col_blank, _ = st.columns([1, 3])
        with col_blank:
            if st.button("Start from scratch", width="stretch"):
                st.session_state["template_applied"] = True
                st.rerun()
        return

    # Mode selector - track previous mode to detect switches
    # Apply pending mode change (set by buttons AFTER the radio widget rendered)
    pending = st.session_state.pop("_pending_editor_mode", None)
    if pending:
        st.session_state["editor_mode"] = pending
    prev_mode = st.session_state.get("_prev_editor_mode")
    col_mode, col_cancel = st.columns([3, 1])
    with col_mode:
        mode = st.radio(
            "Editor mode", ["Visual", "YAML", "Split"], horizontal=True,
            key="editor_mode", label_visibility="collapsed",
        )
    # Handle mode switch -- clean up old state to prevent leaks
    if prev_mode and prev_mode != mode:
        if mode == "Visual":
            # Switching YAML -> Visual: clean old form state, parse YAML, sync
            cleanup_form_state()
            try:
                dsl = yaml.safe_load(st.session_state[yaml_key])
                if isinstance(dsl, dict):
                    sync_form_state(dsl)
            except yaml.YAMLError:
                pass
        elif mode == "YAML":
            # Switching Visual -> YAML: build from form and update YAML
            try:
                dsl = yaml.safe_load(st.session_state[yaml_key])
                if isinstance(dsl, dict):
                    new_dsl = build_dsl_from_form(dsl)
                    st.session_state[yaml_key] = yaml.dump(
                        new_dsl, default_flow_style=False, sort_keys=False,
                    )
            except yaml.YAMLError:
                pass
            cleanup_form_state()
        elif mode == "Split":
            # Switching to Split: clean form state to avoid stale keys
            cleanup_form_state()
    st.session_state["_prev_editor_mode"] = mode
    with col_cancel:
        if st.button("Back to list"):
            st.session_state.show_editor = False
            st.session_state.editing_strategy_id = None
            st.session_state.pop("template_applied", None)
            st.session_state.pop("_prev_editor_mode", None)
            cleanup_form_state()
            st.rerun()

    st.divider()
    if mode == "YAML":
        _render_yaml_editor(manager, existing, yaml_key)
    elif mode == "Split":
        _render_split_editor(manager, existing, yaml_key)
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
        model, error = validate_dsl_yaml(yaml_content)
        render_validation_summary(model, error)

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Save", type="primary", width="stretch"):
            model, error = validate_dsl_yaml(yaml_content)
            if error:
                st.error(f"Validation failed:\n{error}")
            elif model and _save_strategy(manager, model.model_dump(), existing):
                st.session_state.pop("template_applied", None)
                st.rerun()
    with c2:
        if st.button("Copy to Visual Editor", width="stretch"):
            st.session_state[yaml_key] = yaml_content
            try:
                dsl = yaml.safe_load(yaml_content)
                if isinstance(dsl, dict):
                    sync_form_state(dsl)
            except yaml.YAMLError:
                pass
            st.session_state["_pending_editor_mode"] = "Visual"
            st.rerun()
    with c3:
        if st.button("Reset", width="stretch"):
            if existing:
                st.session_state[yaml_key] = yaml.dump(
                    existing["dsl_config"], default_flow_style=False, sort_keys=False,
                )
            else:
                st.session_state[yaml_key] = _get_default_dsl_yaml()
            st.rerun()


# ── Split Editor ──────────────────────────────────────────────────────────


def _render_split_editor(
    manager: StateManager, existing: dict[str, Any] | None, yaml_key: str,
) -> None:
    """Render synchronized split-pane editor (YAML + form side-by-side)."""
    col_yaml, col_form = st.columns(2)

    with col_yaml:
        st.markdown("**YAML Editor**")
        yaml_content = st.text_area(
            "Strategy DSL",
            value=st.session_state[yaml_key],
            height=600,
            key=f"split_yaml_{yaml_key}",
            label_visibility="collapsed",
        )

        # Live validation
        model, error = validate_dsl_yaml(yaml_content)
        if error:
            st.error(f"Validation: {error[:200]}")
        else:
            st.success("Valid")

    with col_form:
        st.markdown("**Visual Preview**")
        try:
            dsl = yaml.safe_load(yaml_content)
            if not isinstance(dsl, dict):
                dsl = yaml.safe_load(_get_default_dsl_yaml())
        except yaml.YAMLError:
            dsl = yaml.safe_load(_get_default_dsl_yaml())

        # Read-only preview of key settings
        with st.container(border=True):
            st.caption(f"**Name:** {dsl.get('name', 'N/A')}")
            st.caption(f"**Timeframe:** {dsl.get('timeframe', 'N/A')}")
            st.caption(f"**Indicators:** {len(dsl.get('indicators', {}))}")

            indicators = dsl.get("indicators", {})
            if indicators:
                for name, cfg in indicators.items():
                    st.caption(f"  - `{name}`: {cfg.get('type', '?')}")

            entry = dsl.get("entry_conditions", {})
            st.caption(f"**Entry:** {len(entry.get('long', []))}L / {len(entry.get('short', []))}S")

            # Show conditions in human-readable form
            from vibe_quant.dashboard.components.condition_builder import format_condition_human
            for cond in entry.get("long", [])[:3]:
                st.caption(f"  L: {format_condition_human(cond)}")
            for cond in entry.get("short", [])[:3]:
                st.caption(f"  S: {format_condition_human(cond)}")

            sl = dsl.get("stop_loss", {})
            tp = dsl.get("take_profit", {})
            st.caption(f"**SL:** {sl.get('type', 'N/A')} | **TP:** {tp.get('type', 'N/A')}")

            sweep = dsl.get("sweep", {})
            if sweep:
                combos = 1
                for v in sweep.values():
                    combos *= len(v)
                st.caption(f"**Sweep:** {len(sweep)} params, {combos:,} combos")

        if model:
            render_validation_summary(model, None)

    # Save/cancel buttons
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Save", type="primary", width="stretch"):
            model, error = validate_dsl_yaml(yaml_content)
            if error:
                st.error(f"Validation failed:\n{error}")
            elif model and _save_strategy(manager, model.model_dump(), existing):
                st.session_state.pop("template_applied", None)
                st.rerun()
    with c2:
        if st.button("Sync to YAML", width="stretch"):
            st.session_state[yaml_key] = yaml_content
            st.rerun()
    with c3:
        if st.button("Reset", width="stretch"):
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
    model, error = validate_dsl_dict(new_dsl)
    render_validation_summary(model, error)

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if (st.button("Save Strategy", type="primary", width="stretch",
                       disabled=model is None)
                and model and _save_strategy(manager, model.model_dump(), existing)):
            cleanup_form_state()
            st.rerun()
    with c2:
        if st.button("View as YAML", width="stretch"):
            st.session_state[yaml_key] = yaml.dump(new_dsl, default_flow_style=False, sort_keys=False)
            st.session_state["_pending_editor_mode"] = "YAML"
            cleanup_form_state()
            st.rerun()
    with c3:
        if st.button("Reset", width="stretch"):
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

    if st.session_state.get("wizard_active"):
        col_back, _ = st.columns([1, 5])
        with col_back:
            if st.button("Back to list"):
                cleanup_form_state()
                st.session_state.pop("wizard_active", None)
                st.session_state.pop("wizard_step", None)
                st.session_state.pop("wizard_dsl", None)
                st.rerun()
        if render_strategy_wizard(manager):
            st.rerun()
        return

    if st.session_state.get("show_editor"):
        render_strategy_editor(manager, strategy_id=st.session_state.get("editing_strategy_id"))
        return

    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    with c1:
        search_query = st.text_input(
            "Search strategies", placeholder="Name or description...",
            label_visibility="collapsed",
        )
    with c2:
        show_inactive = st.checkbox("Show inactive")
    with c3:
        if st.button("New Strategy", type="primary", width="stretch"):
            st.session_state.show_editor = True
            st.session_state.editing_strategy_id = None
            st.session_state.pop("template_applied", None)
            st.rerun()
    with c4:
        if st.button("Wizard", width="stretch"):
            st.session_state["wizard_active"] = True
            st.rerun()

    st.divider()
    render_strategy_list(manager, search_query, show_inactive)

render = render_strategy_management_tab

if __name__ == "__main__":
    render_strategy_management_tab()
