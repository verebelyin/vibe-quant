"""Strategy card component for strategy list view.

Renders a strategy as a bordered card with:
- Name, version, timeframe, indicator/entry stats
- Edit / Activate-Deactivate / Delete action buttons
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import streamlit as st

if TYPE_CHECKING:
    from vibe_quant.db.state_manager import StateManager


def render_strategy_card(manager: StateManager, strategy: dict[str, Any]) -> None:
    """Render a single strategy as a card with actions."""
    dsl = strategy.get("dsl_config", {})
    is_active = strategy["is_active"]
    indicators = dsl.get("indicators", {})
    entry = dsl.get("entry_conditions", {})
    sweep = dsl.get("sweep", {})

    status_icon = "" if is_active else " [inactive]"

    with st.container(border=True):
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
                if st.button("Edit", key=f"edit_{strategy['id']}", width="stretch"):
                    st.session_state.editing_strategy_id = strategy["id"]
                    st.session_state.show_editor = True
                    st.rerun()
            with c2:
                if is_active:
                    if st.button("Deactivate", key=f"deact_{strategy['id']}",
                                 width="stretch"):
                        manager.update_strategy(strategy["id"], is_active=False)
                        st.rerun()
                else:
                    if st.button("Activate", key=f"act_{strategy['id']}",
                                 width="stretch"):
                        manager.update_strategy(strategy["id"], is_active=True)
                        st.rerun()
            with c3:
                if st.button("Delete", key=f"del_{strategy['id']}",
                             width="stretch"):
                    st.session_state.confirm_delete_id = strategy["id"]
                    st.session_state.confirm_delete_name = strategy["name"]
                    st.rerun()
