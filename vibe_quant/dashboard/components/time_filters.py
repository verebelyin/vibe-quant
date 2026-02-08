"""Time filters component for strategy editor.

Provides:
- Blocked trading days multi-select
- Funding avoidance toggle with minute-based controls
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from vibe_quant.dsl.schema import VALID_DAYS


def render_time_filters_section(dsl: dict[str, Any]) -> None:
    """Render time filters: blocked days and funding avoidance."""
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
                    min_value=0, max_value=60,
                    key="form_funding_before",
                )
            with fc2:
                st.number_input(
                    "Minutes after",
                    value=funding.get("minutes_after", 5),
                    min_value=0, max_value=60,
                    key="form_funding_after",
                )
