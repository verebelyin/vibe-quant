"""Strategy Management Tab for vibe-quant dashboard.

Provides CRUD operations for trading strategies with best-in-class UX:
- Strategy list with search/filter and strategy cards
- Template selector with categorized card grid
- Visual form editor with:
  - Searchable indicator catalog with add/remove/duplicate
  - Visual condition builder with dropdown-based row editing
  - Conditional risk parameter fields (only relevant fields shown)
  - Structured sweep parameter builder with combination warnings
  - Strategy validation summary with smart warnings
- Raw YAML editor with syntax highlighting
- Three editor modes: Visual, YAML, Split
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import streamlit as st
import yaml
from pydantic import ValidationError

from vibe_quant.dashboard.components.condition_builder import render_condition_builder
from vibe_quant.dashboard.components.indicator_catalog import (
    render_indicator_card,
    render_indicator_selector,
)
from vibe_quant.dashboard.components.sweep_builder import render_sweep_builder
from vibe_quant.dashboard.components.template_selector import render_template_selector
from vibe_quant.dashboard.components.validation_summary import render_validation_summary
from vibe_quant.dashboard.utils import get_state_manager
from vibe_quant.dsl.indicator_metadata import suggest_indicator_name
from vibe_quant.dsl.schema import (
    VALID_DAYS,
    VALID_STOP_LOSS_TYPES,
    VALID_TAKE_PROFIT_TYPES,
    VALID_TIMEFRAMES,
    StrategyDSL,
)

if TYPE_CHECKING:
    from pathlib import Path

    from vibe_quant.db.state_manager import StateManager

# ── Helpers ────────────────────────────────────────────────────────────────


def _validate_dsl(yaml_str: str) -> tuple[StrategyDSL | None, str | None]:
    """Validate DSL YAML string.

    Returns:
        Tuple of (validated model, error message). One will be None.
    """
    try:
        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            return None, "YAML must be a mapping"
        model = StrategyDSL.model_validate(data)
        return model, None
    except yaml.YAMLError as e:
        return None, f"YAML parse error: {e}"
    except ValidationError as e:
        errors = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            errors.append(f"{loc}: {err['msg']}")
        return None, "\n".join(errors)


def _validate_dsl_dict(dsl_dict: dict[str, Any]) -> tuple[StrategyDSL | None, str | None]:
    """Validate a DSL dict directly (avoids extra YAML round-trip)."""
    try:
        model = StrategyDSL.model_validate(dsl_dict)
        return model, None
    except ValidationError as e:
        errors = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            errors.append(f"{loc}: {err['msg']}")
        return None, "\n".join(errors)


def _get_default_dsl_yaml() -> str:
    """Return default DSL YAML template."""
    return """name: my_strategy
description: A simple RSI-based strategy
version: 1
timeframe: 1h
additional_timeframes: []

indicators:
  rsi_14:
    type: RSI
    period: 14
    source: close
  atr_14:
    type: ATR
    period: 14

entry_conditions:
  long:
    - "rsi_14 < 30"
  short:
    - "rsi_14 > 70"

exit_conditions:
  long:
    - "rsi_14 > 70"
  short:
    - "rsi_14 < 30"

time_filters:
  allowed_sessions: []
  blocked_days: []
  avoid_around_funding:
    enabled: false

stop_loss:
  type: atr_fixed
  atr_multiplier: 2.0
  indicator: atr_14

take_profit:
  type: risk_reward
  risk_reward_ratio: 2.0

position_management:
  scale_in:
    enabled: false
  partial_exit:
    enabled: false

sweep: {}
"""


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
        manager.update_strategy(
            existing["id"],
            dsl_config=dumped,
            description=model.description,
        )
        st.success(f"Updated strategy '{model.name}'")
    else:
        manager.create_strategy(
            name=model.name,
            dsl_config=dumped,
            description=model.description,
        )
        st.success(f"Created strategy '{model.name}'")

    st.session_state.show_editor = False
    st.session_state.editing_strategy_id = None
    return True


def _get_atr_indicator_names(indicators: dict[str, dict[str, Any]]) -> list[str]:
    """Extract names of ATR indicators from the indicator config."""
    return [
        name for name, cfg in indicators.items()
        if cfg.get("type", "").upper() == "ATR"
    ]


# ── Strategy List ──────────────────────────────────────────────────────────


def render_strategy_list(
    manager: StateManager, search_query: str, show_inactive: bool
) -> None:
    """Render strategy list with search/filter and strategy cards."""
    strategies = manager.list_strategies(active_only=not show_inactive)

    if search_query:
        query_lower = search_query.lower()
        strategies = [
            s
            for s in strategies
            if query_lower in s["name"].lower()
            or (s.get("description") and query_lower in s["description"].lower())
        ]

    if not strategies:
        st.info("No strategies found. Create one using the **New Strategy** button above.")
        return

    for strategy in strategies:
        _render_strategy_card(manager, strategy)


def _render_strategy_card(manager: StateManager, strategy: dict[str, Any]) -> None:
    """Render a single strategy as a card with actions."""
    dsl = strategy.get("dsl_config", {})
    is_active = strategy["is_active"]
    indicators = dsl.get("indicators", {})
    entry = dsl.get("entry_conditions", {})
    sweep = dsl.get("sweep", {})

    status_icon = "" if is_active else " [inactive]"

    with st.container(border=True):
        # Header row
        col_name, col_tf, col_stats, col_actions = st.columns([3, 1, 2, 2])

        with col_name:
            st.markdown(f"**{strategy['name']}** v{strategy['version']}{status_icon}")
            st.caption(strategy.get("description") or "No description")

        with col_tf:
            st.metric("Timeframe", dsl.get("timeframe", "N/A"), label_visibility="collapsed")
            st.caption(f"TF: {dsl.get('timeframe', 'N/A')}")

        with col_stats:
            n_ind = len(indicators)
            n_long = len(entry.get("long", []))
            n_short = len(entry.get("short", []))
            n_sweep = len(sweep)
            st.caption(
                f"{n_ind} indicators | "
                f"{n_long}L/{n_short}S entries | "
                f"{n_sweep} sweep params"
            )
            if indicators:
                st.caption(f"**Indicators:** {', '.join(indicators.keys())}")

        with col_actions:
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Edit", key=f"edit_{strategy['id']}", use_container_width=True):
                    st.session_state.editing_strategy_id = strategy["id"]
                    st.session_state.show_editor = True
                    st.rerun()
            with c2:
                if is_active:
                    if st.button(
                        "Deactivate", key=f"deact_{strategy['id']}",
                        use_container_width=True,
                    ):
                        manager.update_strategy(strategy["id"], is_active=False)
                        st.rerun()
                else:
                    if st.button(
                        "Activate", key=f"act_{strategy['id']}",
                        use_container_width=True,
                    ):
                        manager.update_strategy(strategy["id"], is_active=True)
                        st.rerun()
            with c3:
                if st.button(
                    "Delete", key=f"del_{strategy['id']}",
                    use_container_width=True,
                ):
                    st.session_state.confirm_delete_id = strategy["id"]
                    st.session_state.confirm_delete_name = strategy["name"]
                    st.rerun()


# ── Delete Confirmation ───────────────────────────────────────────────────


def render_delete_confirmation(manager: StateManager) -> None:
    """Render delete confirmation dialog."""
    strategy_id = st.session_state.get("confirm_delete_id")
    strategy_name = st.session_state.get("confirm_delete_name")

    if not strategy_id:
        return

    st.warning(f"Delete strategy '{strategy_name}'? This cannot be undone.")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Confirm Delete"):
            manager.update_strategy(strategy_id, is_active=False)
            st.success(f"Deleted strategy '{strategy_name}'")
            st.session_state.confirm_delete_id = None
            st.session_state.confirm_delete_name = None
            st.rerun()

    with col2:
        if st.button("Cancel Delete"):
            st.session_state.confirm_delete_id = None
            st.session_state.confirm_delete_name = None
            st.rerun()


# ── Editor Entrypoint ─────────────────────────────────────────────────────


def render_strategy_editor(
    manager: StateManager,
    strategy_id: int | None = None,
) -> None:
    """Render strategy create/edit with template selector and multi-mode editor."""
    existing = None
    if strategy_id:
        existing = manager.get_strategy(strategy_id)
        if not existing:
            st.error(f"Strategy {strategy_id} not found")
            return

    # ── Header ──
    st.subheader("Edit Strategy" if existing else "Create Strategy")

    # Initialize YAML in session state
    yaml_key = f"yaml_content_{strategy_id or 'new'}"
    if yaml_key not in st.session_state:
        if existing:
            st.session_state[yaml_key] = yaml.dump(
                existing["dsl_config"], default_flow_style=False, sort_keys=False
            )
        else:
            st.session_state[yaml_key] = _get_default_dsl_yaml()

    # ── Template selector (only for new strategies) ──
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

    # ── Editor mode selector ──
    col_mode, col_cancel = st.columns([3, 1])
    with col_mode:
        mode = st.radio(
            "Editor mode",
            ["Visual", "YAML"],
            horizontal=True,
            key="editor_mode",
            label_visibility="collapsed",
        )
    with col_cancel:
        if st.button("Back to list"):
            st.session_state.show_editor = False
            st.session_state.editing_strategy_id = None
            st.session_state.pop("template_applied", None)
            # Clean up condition builder state
            for k in list(st.session_state.keys()):
                if k.startswith(("form_", "cond_", "sweep_")):
                    del st.session_state[k]
            st.rerun()

    st.divider()

    if mode == "YAML":
        _render_yaml_editor(manager, existing, yaml_key)
    else:
        _render_form_editor(manager, existing, yaml_key)


# ── YAML Editor ───────────────────────────────────────────────────────────


def _render_yaml_editor(
    manager: StateManager,
    existing: dict[str, Any] | None,
    yaml_key: str,
) -> None:
    """Render YAML editor with live validation panel."""
    col_editor, col_preview = st.columns([3, 2])

    with col_editor:
        yaml_content = st.text_area(
            "Strategy DSL (YAML)",
            value=st.session_state[yaml_key],
            height=550,
            key=f"yaml_editor_{yaml_key}",
        )

        # File upload
        uploaded_file = st.file_uploader(
            "Or upload YAML file", type=["yaml", "yml"], key="yaml_upload"
        )
        if uploaded_file:
            yaml_content = uploaded_file.read().decode("utf-8")
            st.session_state[yaml_key] = yaml_content
            st.rerun()

    # Live validation preview
    with col_preview:
        st.markdown("**Validation Preview**")
        model, error = _validate_dsl(yaml_content)
        render_validation_summary(model, error)

    # Action buttons
    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Save", type="primary", use_container_width=True):
            model, error = _validate_dsl(yaml_content)
            if error:
                st.error(f"Validation failed:\n{error}")
            elif model and _save_strategy(manager, model.model_dump(), existing):
                st.session_state.pop("template_applied", None)
                st.rerun()

    with col2:
        if st.button("Copy to Visual Editor", use_container_width=True):
            st.session_state[yaml_key] = yaml_content
            st.session_state["editor_mode"] = "Visual"
            st.rerun()

    with col3:
        if st.button("Reset", use_container_width=True):
            if existing:
                st.session_state[yaml_key] = yaml.dump(
                    existing["dsl_config"], default_flow_style=False, sort_keys=False
                )
            else:
                st.session_state[yaml_key] = _get_default_dsl_yaml()
            st.rerun()


# ── Visual Form Editor ────────────────────────────────────────────────────


def _render_form_editor(
    manager: StateManager,
    existing: dict[str, Any] | None,
    yaml_key: str,
) -> None:
    """Render the visual form editor with all new components."""
    # Load DSL from YAML state
    try:
        dsl = yaml.safe_load(st.session_state[yaml_key])
        if not isinstance(dsl, dict):
            dsl = yaml.safe_load(_get_default_dsl_yaml())
    except yaml.YAMLError:
        dsl = yaml.safe_load(_get_default_dsl_yaml())

    # Initialize form state from DSL
    _init_form_state(dsl)

    # ── Section 1: Basic Info ──
    with st.expander("**1. Basic Info**", expanded=True):
        _render_basic_info_section(dsl)

    # ── Section 2: Indicators ──
    with st.expander("**2. Indicators**", expanded=True):
        _render_indicators_section()

    # ── Section 3: Entry/Exit Conditions ──
    with st.expander("**3. Entry & Exit Rules**", expanded=True):
        _render_conditions_section()

    # ── Section 4: Risk Management ──
    with st.expander("**4. Risk Management** (Stop Loss & Take Profit)", expanded=True):
        _render_risk_section(dsl)

    # ── Section 5: Time Filters ──
    with st.expander("**5. Time Filters**", expanded=False):
        _render_time_filters_section(dsl)

    # ── Section 6: Sweep Parameters ──
    with st.expander("**6. Parameter Sweep**", expanded=False):
        _render_sweep_section(dsl)

    # ── Validation & Save ──
    st.divider()
    _render_save_section(manager, existing, yaml_key, dsl)


def _init_form_state(dsl: dict[str, Any]) -> None:
    """Initialize session state for form fields from DSL dict."""
    if "form_indicators" not in st.session_state:
        st.session_state["form_indicators"] = dict(dsl.get("indicators", {}))

    if "form_entry_long" not in st.session_state:
        entry = dsl.get("entry_conditions", {})
        st.session_state["form_entry_long"] = list(entry.get("long", []))
        st.session_state["form_entry_short"] = list(entry.get("short", []))
        exit_cond = dsl.get("exit_conditions", {})
        st.session_state["form_exit_long"] = list(exit_cond.get("long", []))
        st.session_state["form_exit_short"] = list(exit_cond.get("short", []))


# ── Section Renderers ─────────────────────────────────────────────────────


def _render_basic_info_section(dsl: dict[str, Any]) -> None:
    """Render basic info section: name, description, version, timeframes."""
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.text_input(
            "Name",
            value=dsl.get("name", ""),
            key="form_name",
            help="Lowercase, letters/numbers/underscores, starts with letter",
        )
    with col2:
        st.number_input(
            "Version",
            value=dsl.get("version", 1),
            min_value=1,
            max_value=1000,
            key="form_version",
        )
    with col3:
        tf_list = sorted(VALID_TIMEFRAMES)
        current_tf = dsl.get("timeframe", "1h")
        st.selectbox(
            "Primary Timeframe",
            options=tf_list,
            index=tf_list.index(current_tf) if current_tf in tf_list else 0,
            key="form_timeframe",
        )

    st.text_area(
        "Description",
        value=dsl.get("description", ""),
        key="form_description",
        height=68,
    )

    st.multiselect(
        "Additional Timeframes (for multi-TF strategies)",
        options=[
            tf for tf in sorted(VALID_TIMEFRAMES)
            if tf != st.session_state.get("form_timeframe", "1h")
        ],
        default=[
            tf for tf in dsl.get("additional_timeframes", [])
            if tf != st.session_state.get("form_timeframe", "1h")
        ],
        key="form_additional_tfs",
    )


def _render_indicators_section() -> None:
    """Render indicators section with catalog, add/remove/duplicate."""
    indicators: dict[str, dict[str, Any]] = st.session_state["form_indicators"]

    if not indicators:
        st.info("No indicators defined. Add one below.")
    else:
        st.caption(f"{len(indicators)} indicator(s) defined")

    # Render each indicator as a card
    all_tfs = sorted(VALID_TIMEFRAMES)
    indicators_to_remove: list[str] = []
    indicators_to_add: list[tuple[str, dict[str, Any]]] = []

    for name, config in list(indicators.items()):
        with st.expander(f"`{name}` ({config.get('type', '?')})", expanded=False):
            updated, should_remove, should_dup = render_indicator_card(
                name=name,
                config=config,
                key_prefix="form_ind",
                all_timeframes=all_tfs,
            )
            if should_remove:
                indicators_to_remove.append(name)
            elif should_dup:
                # Duplicate with new name
                new_name = suggest_indicator_name(
                    config.get("type", ""), set(indicators.keys())
                )
                indicators_to_add.append((new_name, dict(config)))
            elif updated is not None:
                indicators[name] = updated

    # Process removals and duplications
    if indicators_to_remove or indicators_to_add:
        for name in indicators_to_remove:
            indicators.pop(name, None)
        for name, config in indicators_to_add:
            indicators[name] = config
        st.session_state["form_indicators"] = indicators
        st.rerun()

    # Add new indicator
    st.divider()
    if st.session_state.get("show_indicator_catalog"):
        result = render_indicator_selector(
            key_prefix="form_add_ind",
            existing_indicator_names=set(indicators.keys()),
        )
        if result is not None:
            indicators[result["name"]] = result["config"]
            st.session_state["form_indicators"] = indicators
            st.session_state["show_indicator_catalog"] = False
            st.rerun()
    else:
        if st.button("+ Add Indicator", type="primary"):
            st.session_state["show_indicator_catalog"] = True
            st.rerun()


def _render_conditions_section() -> None:
    """Render entry/exit conditions using visual condition builder."""
    indicators = st.session_state.get("form_indicators", {})
    indicator_names = list(indicators.keys())

    tab_entry, tab_exit = st.tabs(["Entry Conditions", "Exit Conditions"])

    with tab_entry:
        col_long, col_short = st.columns(2)
        with col_long:
            st.session_state["form_entry_long"] = render_condition_builder(
                label="Long Entry",
                conditions=st.session_state.get("form_entry_long", []),
                indicator_names=indicator_names,
                key_prefix="cond_entry_long",
            )
        with col_short:
            st.session_state["form_entry_short"] = render_condition_builder(
                label="Short Entry",
                conditions=st.session_state.get("form_entry_short", []),
                indicator_names=indicator_names,
                key_prefix="cond_entry_short",
            )

    with tab_exit:
        col_long, col_short = st.columns(2)
        with col_long:
            st.session_state["form_exit_long"] = render_condition_builder(
                label="Long Exit",
                conditions=st.session_state.get("form_exit_long", []),
                indicator_names=indicator_names,
                key_prefix="cond_exit_long",
            )
        with col_short:
            st.session_state["form_exit_short"] = render_condition_builder(
                label="Short Exit",
                conditions=st.session_state.get("form_exit_short", []),
                indicator_names=indicator_names,
                key_prefix="cond_exit_short",
            )


def _render_risk_section(dsl: dict[str, Any]) -> None:
    """Render risk management with conditional fields and R:R visualization."""
    indicators = st.session_state.get("form_indicators", {})
    atr_names = _get_atr_indicator_names(indicators)

    col_sl, col_tp = st.columns(2)

    # ── Stop Loss ──
    with col_sl:
        st.markdown("**Stop Loss**")
        sl = dsl.get("stop_loss", {})

        sl_types_sorted = sorted(VALID_STOP_LOSS_TYPES)
        current_sl = sl.get("type", "fixed_pct")
        sl_type = st.selectbox(
            "Type",
            options=sl_types_sorted,
            index=sl_types_sorted.index(current_sl) if current_sl in sl_types_sorted else 0,
            key="form_sl_type",
        )

        # Conditional fields based on type
        if sl_type == "fixed_pct":
            st.number_input(
                "Stop Loss %",
                value=sl.get("percent") or 2.0,
                min_value=0.1,
                max_value=50.0,
                step=0.1,
                key="form_sl_pct",
                help="Fixed percentage stop loss from entry price",
            )
        elif sl_type in ("atr_fixed", "atr_trailing"):
            st.number_input(
                "ATR Multiplier",
                value=sl.get("atr_multiplier") or 2.0,
                min_value=0.5,
                max_value=10.0,
                step=0.1,
                key="form_sl_atr_mult",
                help=f"Stop distance = ATR x multiplier. "
                     f"{'Trailing stop follows price.' if sl_type == 'atr_trailing' else 'Fixed distance from entry.'}",
            )
            if atr_names:
                current_ind = sl.get("indicator", "")
                atr_idx = atr_names.index(current_ind) if current_ind in atr_names else 0
                st.selectbox(
                    "ATR Indicator",
                    options=atr_names,
                    index=atr_idx,
                    key="form_sl_indicator",
                )
            else:
                st.warning("No ATR indicator defined. Add one in the Indicators section.")
                st.text_input(
                    "ATR Indicator Name",
                    value=sl.get("indicator", ""),
                    key="form_sl_indicator",
                )

    # ── Take Profit ──
    with col_tp:
        st.markdown("**Take Profit**")
        tp = dsl.get("take_profit", {})

        tp_types_sorted = sorted(VALID_TAKE_PROFIT_TYPES)
        current_tp = tp.get("type", "risk_reward")
        tp_type = st.selectbox(
            "Type",
            options=tp_types_sorted,
            index=tp_types_sorted.index(current_tp) if current_tp in tp_types_sorted else 0,
            key="form_tp_type",
        )

        if tp_type == "fixed_pct":
            st.number_input(
                "Take Profit %",
                value=tp.get("percent") or 4.0,
                min_value=0.1,
                max_value=100.0,
                step=0.1,
                key="form_tp_pct",
                help="Fixed percentage take profit from entry price",
            )
        elif tp_type == "atr_fixed":
            st.number_input(
                "ATR Multiplier",
                value=tp.get("atr_multiplier") or 3.0,
                min_value=0.5,
                max_value=20.0,
                step=0.1,
                key="form_tp_atr_mult",
                help="Take profit distance = ATR x multiplier",
            )
            if atr_names:
                current_ind = tp.get("indicator", "")
                atr_idx = atr_names.index(current_ind) if current_ind in atr_names else 0
                st.selectbox(
                    "ATR Indicator",
                    options=atr_names,
                    index=atr_idx,
                    key="form_tp_indicator",
                )
            else:
                st.warning("No ATR indicator defined. Add one in the Indicators section.")
                st.text_input(
                    "ATR Indicator Name",
                    value=tp.get("indicator", ""),
                    key="form_tp_indicator",
                )
        elif tp_type == "risk_reward":
            st.slider(
                "Risk/Reward Ratio",
                min_value=0.5,
                max_value=5.0,
                value=tp.get("risk_reward_ratio") or 2.0,
                step=0.1,
                key="form_tp_rr",
                help="Take profit = stop loss distance x R:R ratio",
            )

    # ── Risk/Reward visualization ──
    _render_rr_visualization()

    # ── Risk presets ──
    st.caption("**Quick presets:**")
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        if st.button("Conservative", key="risk_preset_conservative",
                      help="1% SL, 2:1 R:R", use_container_width=True):
            _apply_risk_preset("fixed_pct", 1.0, "risk_reward", 2.0)
            st.rerun()
    with pc2:
        if st.button("Moderate", key="risk_preset_moderate",
                      help="2% SL, 1.5:1 R:R", use_container_width=True):
            _apply_risk_preset("fixed_pct", 2.0, "risk_reward", 1.5)
            st.rerun()
    with pc3:
        if st.button("Aggressive", key="risk_preset_aggressive",
                      help="3% SL, 1:1 R:R", use_container_width=True):
            _apply_risk_preset("fixed_pct", 3.0, "risk_reward", 1.0)
            st.rerun()


def _render_rr_visualization() -> None:
    """Render a simple risk/reward ratio visualization."""
    sl_type = st.session_state.get("form_sl_type", "fixed_pct")
    tp_type = st.session_state.get("form_tp_type", "risk_reward")

    # Calculate SL distance
    if sl_type == "fixed_pct":
        sl_dist = st.session_state.get("form_sl_pct", 2.0)
        sl_label = f"{sl_dist}%"
    else:
        sl_mult = st.session_state.get("form_sl_atr_mult", 2.0)
        sl_dist = sl_mult  # Normalized
        sl_label = f"{sl_mult}x ATR"

    # Calculate TP distance
    if tp_type == "fixed_pct":
        tp_dist = st.session_state.get("form_tp_pct", 4.0)
        tp_label = f"{tp_dist}%"
    elif tp_type == "risk_reward":
        rr = st.session_state.get("form_tp_rr", 2.0)
        tp_dist = sl_dist * rr
        tp_label = f"{rr}:1 R:R"
    else:
        tp_mult = st.session_state.get("form_tp_atr_mult", 3.0)
        tp_dist = tp_mult
        tp_label = f"{tp_mult}x ATR"

    # R:R ratio
    rr_ratio = tp_dist / sl_dist if sl_dist > 0 else 0
    rr_color = "green" if rr_ratio >= 2.0 else "orange" if rr_ratio >= 1.0 else "red"

    st.divider()
    col_sl_viz, col_rr, col_tp_viz = st.columns([2, 1, 2])
    with col_sl_viz:
        st.markdown(f":red[**Stop Loss:** {sl_label}]")
    with col_rr:
        st.markdown(f":{rr_color}[**R:R {rr_ratio:.1f}**]")
    with col_tp_viz:
        st.markdown(f":green[**Take Profit:** {tp_label}]")


def _apply_risk_preset(
    sl_type: str, sl_pct: float, tp_type: str, rr: float
) -> None:
    """Apply a risk management preset to session state."""
    st.session_state["form_sl_type"] = sl_type
    st.session_state["form_sl_pct"] = sl_pct
    st.session_state["form_tp_type"] = tp_type
    st.session_state["form_tp_rr"] = rr


def _render_time_filters_section(dsl: dict[str, Any]) -> None:
    """Render time filters section."""
    time_filters = dsl.get("time_filters", {})

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Trading Days**")
        st.multiselect(
            "Blocked Days",
            options=sorted(VALID_DAYS),
            default=time_filters.get("blocked_days", []),
            key="form_blocked_days",
            help="Days when the strategy will not open new positions",
        )

    with col2:
        st.markdown("**Funding Avoidance**")
        funding = time_filters.get("avoid_around_funding", {})
        st.checkbox(
            "Avoid trading around funding settlement",
            value=funding.get("enabled", False),
            key="form_funding_enabled",
            help="Prevents entries near the 8h funding settlement (volatile period)",
        )
        if st.session_state.get("form_funding_enabled"):
            fc1, fc2 = st.columns(2)
            with fc1:
                st.number_input(
                    "Minutes before",
                    value=funding.get("minutes_before", 5),
                    min_value=0,
                    max_value=60,
                    key="form_funding_before",
                )
            with fc2:
                st.number_input(
                    "Minutes after",
                    value=funding.get("minutes_after", 5),
                    min_value=0,
                    max_value=60,
                    key="form_funding_after",
                )


def _render_sweep_section(dsl: dict[str, Any]) -> None:
    """Render sweep parameters section."""
    indicators = st.session_state.get("form_indicators", {})
    sweep = dsl.get("sweep", {})

    render_sweep_builder(
        sweep_config=sweep,
        indicators=indicators,
        key_prefix="sweep",
    )


def _render_save_section(
    manager: StateManager,
    existing: dict[str, Any] | None,
    yaml_key: str,
    original_dsl: dict[str, Any],
) -> None:
    """Render validation, save, and action buttons."""
    # Build DSL from form state
    new_dsl = _build_dsl_from_form(original_dsl)

    # Validate
    model, error = _validate_dsl_dict(new_dsl)

    # Show validation summary
    render_validation_summary(model, error)

    # Action buttons
    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        if (
            st.button("Save Strategy", type="primary", use_container_width=True,
                      disabled=model is None)
            and model
            and _save_strategy(manager, model.model_dump(), existing)
        ):
                # Clean up form state
                _cleanup_form_state()
                st.rerun()

    with col2:
        if st.button("View as YAML", use_container_width=True):
            yaml_str = yaml.dump(new_dsl, default_flow_style=False, sort_keys=False)
            st.session_state[yaml_key] = yaml_str
            st.session_state["editor_mode"] = "YAML"
            _cleanup_form_state()
            st.rerun()

    with col3:
        if st.button("Reset", use_container_width=True):
            _cleanup_form_state()
            st.rerun()


def _build_dsl_from_form(original_dsl: dict[str, Any]) -> dict[str, Any]:
    """Assemble a complete DSL dict from form session state."""
    indicators = st.session_state.get("form_indicators", {})

    # Build stop loss
    sl_type = st.session_state.get("form_sl_type", "fixed_pct")
    stop_loss: dict[str, Any] = {"type": sl_type}
    if sl_type == "fixed_pct":
        stop_loss["percent"] = st.session_state.get("form_sl_pct", 2.0)
    elif sl_type in ("atr_fixed", "atr_trailing"):
        stop_loss["atr_multiplier"] = st.session_state.get("form_sl_atr_mult", 2.0)
        stop_loss["indicator"] = st.session_state.get("form_sl_indicator", "")

    # Build take profit
    tp_type = st.session_state.get("form_tp_type", "risk_reward")
    take_profit: dict[str, Any] = {"type": tp_type}
    if tp_type == "fixed_pct":
        take_profit["percent"] = st.session_state.get("form_tp_pct", 4.0)
    elif tp_type == "atr_fixed":
        take_profit["atr_multiplier"] = st.session_state.get("form_tp_atr_mult", 3.0)
        take_profit["indicator"] = st.session_state.get("form_tp_indicator", "")
    elif tp_type == "risk_reward":
        take_profit["risk_reward_ratio"] = st.session_state.get("form_tp_rr", 2.0)

    # Build time filters
    time_filters: dict[str, Any] = {
        "allowed_sessions": original_dsl.get("time_filters", {}).get("allowed_sessions", []),
        "blocked_days": st.session_state.get("form_blocked_days", []),
        "avoid_around_funding": {
            "enabled": st.session_state.get("form_funding_enabled", False),
            "minutes_before": st.session_state.get("form_funding_before", 5),
            "minutes_after": st.session_state.get("form_funding_after", 5),
        },
    }

    # Build sweep
    sweep = st.session_state.get("sweep_sweep_params", {})

    return {
        "name": st.session_state.get("form_name", "my_strategy"),
        "description": st.session_state.get("form_description", ""),
        "version": st.session_state.get("form_version", 1),
        "timeframe": st.session_state.get("form_timeframe", "1h"),
        "additional_timeframes": st.session_state.get("form_additional_tfs", []),
        "indicators": indicators,
        "entry_conditions": {
            "long": st.session_state.get("form_entry_long", []),
            "short": st.session_state.get("form_entry_short", []),
        },
        "exit_conditions": {
            "long": st.session_state.get("form_exit_long", []),
            "short": st.session_state.get("form_exit_short", []),
        },
        "time_filters": time_filters,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "position_management": {
            "scale_in": {"enabled": False},
            "partial_exit": {"enabled": False},
        },
        "sweep": sweep,
    }


def _cleanup_form_state() -> None:
    """Remove form-specific session state keys."""
    prefixes = (
        "form_", "cond_", "sweep_", "show_indicator_catalog",
        "template_applied",
    )
    for key in list(st.session_state.keys()):
        if any(key.startswith(p) for p in prefixes):
            del st.session_state[key]


# ── Main Entry Point ──────────────────────────────────────────────────────


def render_strategy_management_tab(db_path: Path | None = None) -> None:
    """Main entry point for strategy management tab."""
    st.header("Strategy Management")

    manager = get_state_manager(db_path)

    # Check for delete confirmation
    if st.session_state.get("confirm_delete_id"):
        render_delete_confirmation(manager)
        return

    # Check for editor mode
    if st.session_state.get("show_editor"):
        render_strategy_editor(
            manager,
            strategy_id=st.session_state.get("editing_strategy_id"),
        )
        return

    # Main view: search + create button
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        search_query = st.text_input(
            "Search strategies",
            placeholder="Name or description...",
            label_visibility="collapsed",
        )

    with col2:
        show_inactive = st.checkbox("Show inactive")

    with col3:
        if st.button("New Strategy", type="primary", use_container_width=True):
            st.session_state.show_editor = True
            st.session_state.editing_strategy_id = None
            st.session_state.pop("template_applied", None)
            st.rerun()

    st.divider()

    render_strategy_list(manager, search_query, show_inactive)


# Top-level call for st.navigation API
render_strategy_management_tab()
