"""Strategy Management Tab for vibe-quant dashboard.

Provides CRUD operations for trading strategies:
- List strategies with search/filter
- Create new strategy (YAML editor or form)
- Edit existing strategy
- Delete strategy (with confirmation)
- DSL validation on save
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import streamlit as st
import yaml
from pydantic import ValidationError

from vibe_quant.dashboard.utils import get_state_manager
from vibe_quant.dsl.schema import (
    VALID_DAYS,
    VALID_INDICATOR_TYPES,
    VALID_SOURCES,
    VALID_STOP_LOSS_TYPES,
    VALID_TAKE_PROFIT_TYPES,
    VALID_TIMEFRAMES,
    StrategyDSL,
)


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


def render_strategy_list(manager: StateManager, search_query: str, show_inactive: bool) -> None:
    """Render strategy list with search/filter."""
    strategies = manager.list_strategies(active_only=not show_inactive)

    # Filter by search query
    if search_query:
        query_lower = search_query.lower()
        strategies = [
            s for s in strategies
            if query_lower in s["name"].lower()
            or (s.get("description") and query_lower in s["description"].lower())
        ]

    if not strategies:
        st.info("No strategies found")
        return

    for strategy in strategies:
        with st.expander(
            f"**{strategy['name']}** (v{strategy['version']}) "
            f"{'[inactive]' if not strategy['is_active'] else ''}"
        ):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(f"**Description:** {strategy.get('description') or 'N/A'}")
                st.write(f"**Type:** {strategy.get('strategy_type') or 'N/A'}")
                st.write(f"**Created:** {strategy['created_at']}")
                st.write(f"**Updated:** {strategy['updated_at']}")

                # Show DSL details
                dsl = strategy.get("dsl_config", {})
                st.write(f"**Timeframe:** {dsl.get('timeframe', 'N/A')}")

                indicators = dsl.get("indicators", {})
                if indicators:
                    st.write(f"**Indicators:** {', '.join(indicators.keys())}")

                entry = dsl.get("entry_conditions", {})
                if entry.get("long"):
                    st.write(f"**Long entries:** {len(entry['long'])} conditions")
                if entry.get("short"):
                    st.write(f"**Short entries:** {len(entry['short'])} conditions")

                sweep = dsl.get("sweep", {})
                if sweep:
                    st.write(f"**Sweep params:** {', '.join(sweep.keys())}")

            with col2:
                if st.button("Edit", key=f"edit_{strategy['id']}"):
                    st.session_state.editing_strategy_id = strategy["id"]
                    st.session_state.show_editor = True
                    st.rerun()

                if st.button("Delete", key=f"delete_{strategy['id']}"):
                    st.session_state.confirm_delete_id = strategy["id"]
                    st.session_state.confirm_delete_name = strategy["name"]
                    st.rerun()

                if strategy["is_active"]:
                    if st.button("Deactivate", key=f"deact_{strategy['id']}"):
                        manager.update_strategy(strategy["id"], is_active=False)
                        st.success(f"Deactivated {strategy['name']}")
                        st.rerun()
                else:
                    if st.button("Activate", key=f"act_{strategy['id']}"):
                        manager.update_strategy(strategy["id"], is_active=True)
                        st.success(f"Activated {strategy['name']}")
                        st.rerun()


def render_strategy_editor(
    manager: StateManager,
    strategy_id: int | None = None,
    raw_mode: bool = False,
) -> None:
    """Render strategy create/edit form.

    Args:
        manager: StateManager instance.
        strategy_id: If provided, edit existing strategy.
        raw_mode: If True, show raw YAML editor only.
    """
    existing = None
    if strategy_id:
        existing = manager.get_strategy(strategy_id)
        if not existing:
            st.error(f"Strategy {strategy_id} not found")
            return

    st.subheader("Edit Strategy" if existing else "Create Strategy")

    # Initialize yaml_content in session state if not present
    yaml_key = f"yaml_content_{strategy_id or 'new'}"
    if yaml_key not in st.session_state:
        if existing:
            st.session_state[yaml_key] = yaml.dump(
                existing["dsl_config"],
                default_flow_style=False,
                sort_keys=False,
            )
        else:
            st.session_state[yaml_key] = _get_default_dsl_yaml()

    if raw_mode:
        _render_yaml_editor(manager, existing, yaml_key)
    else:
        _render_form_editor(manager, existing, yaml_key)


def _render_yaml_editor(
    manager: StateManager,
    existing: dict[str, Any] | None,
    yaml_key: str,
) -> None:
    """Render raw YAML editor."""
    yaml_content = st.text_area(
        "Strategy DSL (YAML)",
        value=st.session_state[yaml_key],
        height=500,
        key=f"yaml_editor_{yaml_key}",
    )

    # File upload
    uploaded_file = st.file_uploader("Or upload YAML file", type=["yaml", "yml"])
    if uploaded_file:
        yaml_content = uploaded_file.read().decode("utf-8")
        st.session_state[yaml_key] = yaml_content
        st.rerun()

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Validate"):
            model, error = _validate_dsl(yaml_content)
            if error:
                st.error(f"Validation failed:\n{error}")
            else:
                st.success("Valid DSL!")

    with col2:
        if st.button("Save"):
            model, error = _validate_dsl(yaml_content)
            if error:
                st.error(f"Validation failed:\n{error}")
                return

            if model is None:
                st.error("Validation returned no model")
                return

            dsl_dict = model.model_dump()

            if existing:
                manager.update_strategy(
                    existing["id"],
                    dsl_config=dsl_dict,
                    description=model.description,
                )
                st.success(f"Updated strategy '{model.name}'")
            else:
                manager.create_strategy(
                    name=model.name,
                    dsl_config=dsl_dict,
                    description=model.description,
                )
                st.success(f"Created strategy '{model.name}'")

            st.session_state.show_editor = False
            st.session_state.editing_strategy_id = None
            st.rerun()

    with col3:
        if st.button("Cancel"):
            st.session_state.show_editor = False
            st.session_state.editing_strategy_id = None
            st.rerun()


def _render_form_editor(
    manager: StateManager,
    existing: dict[str, Any] | None,
    yaml_key: str,
) -> None:
    """Render form-based editor with structured inputs."""
    dsl = existing["dsl_config"] if existing else yaml.safe_load(_get_default_dsl_yaml())

    with st.form("strategy_form"):
        # Basic info
        st.markdown("### Basic Info")
        name = st.text_input(
            "Name",
            value=dsl.get("name", ""),
            help="Lowercase, letters/numbers/underscores, starts with letter",
        )
        description = st.text_area(
            "Description",
            value=dsl.get("description", ""),
        )
        version = st.number_input(
            "Version",
            value=dsl.get("version", 1),
            min_value=1,
            max_value=1000,
        )

        # Timeframes
        st.markdown("### Timeframes")
        timeframe = st.selectbox(
            "Primary Timeframe",
            options=sorted(VALID_TIMEFRAMES),
            index=list(sorted(VALID_TIMEFRAMES)).index(dsl.get("timeframe", "1h")),
        )
        additional_tfs = st.multiselect(
            "Additional Timeframes",
            options=[tf for tf in sorted(VALID_TIMEFRAMES) if tf != timeframe],
            default=dsl.get("additional_timeframes", []),
        )

        # Indicators
        st.markdown("### Indicators")
        st.info("Configure indicators below. Each indicator needs a unique name.")

        indicators = dsl.get("indicators", {})
        indicator_configs: dict[str, dict[str, Any]] = {}

        for i, (ind_name, ind_config) in enumerate(indicators.items()):
            with st.expander(f"Indicator: {ind_name}", expanded=i == 0):
                ind_type = st.selectbox(
                    "Type",
                    options=sorted(VALID_INDICATOR_TYPES),
                    index=list(sorted(VALID_INDICATOR_TYPES)).index(ind_config.get("type", "RSI")),
                    key=f"ind_type_{ind_name}",
                )
                ind_source = st.selectbox(
                    "Source",
                    options=sorted(VALID_SOURCES),
                    index=list(sorted(VALID_SOURCES)).index(ind_config.get("source", "close")),
                    key=f"ind_source_{ind_name}",
                )
                ind_period = st.number_input(
                    "Period",
                    value=ind_config.get("period") or 14,
                    min_value=1,
                    max_value=500,
                    key=f"ind_period_{ind_name}",
                )
                ind_tf = st.selectbox(
                    "Timeframe Override (optional)",
                    options=[""] + list(sorted(VALID_TIMEFRAMES)),
                    index=0 if not ind_config.get("timeframe") else
                    ([""] + list(sorted(VALID_TIMEFRAMES))).index(ind_config["timeframe"]),
                    key=f"ind_tf_{ind_name}",
                )

                indicator_configs[ind_name] = {
                    "type": ind_type,
                    "source": ind_source,
                    "period": ind_period,
                }
                if ind_tf:
                    indicator_configs[ind_name]["timeframe"] = ind_tf

        # Entry/Exit conditions
        st.markdown("### Entry Conditions")
        entry = dsl.get("entry_conditions", {})
        long_entries = st.text_area(
            "Long Conditions (one per line)",
            value="\n".join(entry.get("long", [])),
            help='e.g., "rsi_14 < 30"',
        )
        short_entries = st.text_area(
            "Short Conditions (one per line)",
            value="\n".join(entry.get("short", [])),
        )

        st.markdown("### Exit Conditions")
        exit_cond = dsl.get("exit_conditions", {})
        long_exits = st.text_area(
            "Long Exits (one per line)",
            value="\n".join(exit_cond.get("long", [])),
        )
        short_exits = st.text_area(
            "Short Exits (one per line)",
            value="\n".join(exit_cond.get("short", [])),
        )

        # Time filters
        st.markdown("### Time Filters")
        time_filters = dsl.get("time_filters", {})
        blocked_days = st.multiselect(
            "Blocked Days",
            options=sorted(VALID_DAYS),
            default=time_filters.get("blocked_days", []),
        )

        funding = time_filters.get("avoid_around_funding", {})
        funding_enabled = st.checkbox(
            "Avoid Around Funding",
            value=funding.get("enabled", False),
        )
        funding_before = st.number_input(
            "Minutes Before Funding",
            value=funding.get("minutes_before", 5),
            min_value=0,
            max_value=60,
        )
        funding_after = st.number_input(
            "Minutes After Funding",
            value=funding.get("minutes_after", 5),
            min_value=0,
            max_value=60,
        )

        # Stop loss
        st.markdown("### Stop Loss")
        sl = dsl.get("stop_loss", {})
        sl_type = st.selectbox(
            "Stop Loss Type",
            options=sorted(VALID_STOP_LOSS_TYPES),
            index=list(sorted(VALID_STOP_LOSS_TYPES)).index(sl.get("type", "fixed_pct")),
        )
        sl_percent = st.number_input(
            "Stop Loss Percent (for fixed_pct)",
            value=sl.get("percent") or 2.0,
            min_value=0.1,
            max_value=50.0,
        )
        sl_atr_mult = st.number_input(
            "ATR Multiplier (for atr_fixed/atr_trailing)",
            value=sl.get("atr_multiplier") or 2.0,
            min_value=0.5,
            max_value=10.0,
        )
        sl_indicator = st.text_input(
            "ATR Indicator Name",
            value=sl.get("indicator") or "",
        )

        # Take profit
        st.markdown("### Take Profit")
        tp = dsl.get("take_profit", {})
        tp_type = st.selectbox(
            "Take Profit Type",
            options=sorted(VALID_TAKE_PROFIT_TYPES),
            index=list(sorted(VALID_TAKE_PROFIT_TYPES)).index(tp.get("type", "fixed_pct")),
        )
        tp_percent = st.number_input(
            "Take Profit Percent (for fixed_pct)",
            value=tp.get("percent") or 4.0,
            min_value=0.1,
            max_value=100.0,
        )
        tp_atr_mult = st.number_input(
            "ATR Multiplier (for atr_fixed)",
            value=tp.get("atr_multiplier") or 3.0,
            min_value=0.5,
            max_value=20.0,
        )
        tp_rr = st.number_input(
            "Risk/Reward Ratio (for risk_reward)",
            value=tp.get("risk_reward_ratio") or 2.0,
            min_value=0.5,
            max_value=10.0,
        )
        tp_indicator = st.text_input(
            "ATR Indicator Name (for atr_fixed TP)",
            value=tp.get("indicator") or "",
        )

        # Sweep params
        st.markdown("### Parameter Sweep")
        sweep = dsl.get("sweep", {})
        sweep_yaml = st.text_area(
            "Sweep Parameters (YAML)",
            value=yaml.dump(sweep, default_flow_style=False) if sweep else "",
            help="e.g., rsi_period: [10, 14, 20]",
        )

        # Submit
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("Save Strategy")
        with col2:
            cancelled = st.form_submit_button("Cancel")

    if cancelled:
        st.session_state.show_editor = False
        st.session_state.editing_strategy_id = None
        st.rerun()

    if submitted:
        # Build DSL dict
        new_dsl: dict[str, Any] = {
            "name": name,
            "description": description,
            "version": version,
            "timeframe": timeframe,
            "additional_timeframes": additional_tfs,
            "indicators": indicator_configs,
            "entry_conditions": {
                "long": [c.strip() for c in long_entries.split("\n") if c.strip()],
                "short": [c.strip() for c in short_entries.split("\n") if c.strip()],
            },
            "exit_conditions": {
                "long": [c.strip() for c in long_exits.split("\n") if c.strip()],
                "short": [c.strip() for c in short_exits.split("\n") if c.strip()],
            },
            "time_filters": {
                "allowed_sessions": time_filters.get("allowed_sessions", []),
                "blocked_days": blocked_days,
                "avoid_around_funding": {
                    "enabled": funding_enabled,
                    "minutes_before": funding_before,
                    "minutes_after": funding_after,
                },
            },
            "stop_loss": {"type": sl_type},
            "take_profit": {"type": tp_type},
            "position_management": {
                "scale_in": {"enabled": False},
                "partial_exit": {"enabled": False},
            },
        }

        # Stop loss params
        if sl_type == "fixed_pct":
            new_dsl["stop_loss"]["percent"] = sl_percent
        elif sl_type in {"atr_fixed", "atr_trailing"}:
            new_dsl["stop_loss"]["atr_multiplier"] = sl_atr_mult
            new_dsl["stop_loss"]["indicator"] = sl_indicator

        # Take profit params
        if tp_type == "fixed_pct":
            new_dsl["take_profit"]["percent"] = tp_percent
        elif tp_type == "atr_fixed":
            new_dsl["take_profit"]["atr_multiplier"] = tp_atr_mult
            new_dsl["take_profit"]["indicator"] = tp_indicator
        elif tp_type == "risk_reward":
            new_dsl["take_profit"]["risk_reward_ratio"] = tp_rr

        # Parse sweep
        if sweep_yaml.strip():
            try:
                new_dsl["sweep"] = yaml.safe_load(sweep_yaml)
            except yaml.YAMLError as e:
                st.error(f"Invalid sweep YAML: {e}")
                return
        else:
            new_dsl["sweep"] = {}

        # Validate
        model, error = _validate_dsl(yaml.dump(new_dsl))
        if error:
            st.error(f"Validation failed:\n{error}")
            return

        if model is None:
            st.error("Validation returned no model")
            return

        dsl_dict = model.model_dump()

        if existing:
            manager.update_strategy(
                existing["id"],
                dsl_config=dsl_dict,
                description=model.description,
            )
            st.success(f"Updated strategy '{model.name}'")
        else:
            manager.create_strategy(
                name=model.name,
                dsl_config=dsl_dict,
                description=model.description,
            )
            st.success(f"Created strategy '{model.name}'")

        st.session_state.show_editor = False
        st.session_state.editing_strategy_id = None
        st.rerun()


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
            # Soft delete by deactivating
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


def render_strategy_management_tab(db_path: Path | None = None) -> None:
    """Main entry point for strategy management tab.

    Args:
        db_path: Optional database path. Uses default if not specified.
    """
    st.header("Strategy Management")

    manager = get_state_manager(db_path)

    # Check for delete confirmation
    if st.session_state.get("confirm_delete_id"):
        render_delete_confirmation(manager)
        return

    # Check for editor mode
    if st.session_state.get("show_editor"):
        col1, col2 = st.columns([3, 1])
        with col2:
            raw_mode = st.toggle("Raw YAML Mode", value=True)
        render_strategy_editor(
            manager,
            strategy_id=st.session_state.get("editing_strategy_id"),
            raw_mode=raw_mode,
        )
        return

    # Main view
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        search_query = st.text_input("Search strategies", placeholder="Name or description...")

    with col2:
        show_inactive = st.checkbox("Show inactive")

    with col3:
        if st.button("New Strategy", type="primary"):
            st.session_state.show_editor = True
            st.session_state.editing_strategy_id = None
            st.rerun()

    st.divider()

    render_strategy_list(manager, search_query, show_inactive)


# Top-level call for st.navigation API
render_strategy_management_tab()
