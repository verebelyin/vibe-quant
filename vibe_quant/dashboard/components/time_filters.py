"""Time filters component for strategy editor.

Provides:
- Visual weekly schedule grid with session presets
- Allowed sessions with start/end time inputs
- Blocked trading days multi-select
- Funding avoidance toggle with minute-based controls
- Funding settlement time overlay
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from vibe_quant.dsl.schema import VALID_DAYS

# Session presets for common trading patterns
SESSION_PRESETS = {
    "24/7 (All Sessions)": [],
    "Asia (00:00-08:00 UTC)": [{"start": "00:00", "end": "08:00"}],
    "Europe (08:00-16:00 UTC)": [{"start": "08:00", "end": "16:00"}],
    "US (13:00-21:00 UTC)": [{"start": "13:00", "end": "21:00"}],
    "Asia + Europe": [{"start": "00:00", "end": "08:00"}, {"start": "08:00", "end": "16:00"}],
    "Europe + US": [{"start": "08:00", "end": "21:00"}],
    "High Volume Only": [{"start": "08:00", "end": "16:00"}, {"start": "13:00", "end": "21:00"}],
}

# Funding settlement times (Binance perps: every 8h)
FUNDING_TIMES = ["00:00", "08:00", "16:00"]

ALL_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def render_time_filters_section(dsl: dict[str, Any]) -> None:
    """Render time filters: sessions, blocked days, and funding avoidance."""
    time_filters = dsl.get("time_filters", {})

    # Visual weekly schedule
    _render_weekly_schedule(time_filters)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Trading Days**")
        st.multiselect(
            "Blocked Days",
            options=[d for d in ALL_DAYS if d in VALID_DAYS],
            default=time_filters.get("blocked_days", []),
            key="form_blocked_days",
            help="Days when the strategy will not open new positions",
        )

        # Quick day presets
        st.caption("**Presets:**")
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            if st.button("Weekdays only", key="tf_preset_weekdays", use_container_width=True):
                st.session_state["form_blocked_days"] = ["Saturday", "Sunday"]
                st.rerun()
        with dc2:
            if st.button("All days", key="tf_preset_alldays", use_container_width=True):
                st.session_state["form_blocked_days"] = []
                st.rerun()
        with dc3:
            if st.button("Low vol days", key="tf_preset_lowvol", use_container_width=True):
                st.session_state["form_blocked_days"] = ["Saturday", "Sunday", "Monday"]
                st.rerun()

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

            # Funding time display
            st.caption(
                f"Funding settlements at: {', '.join(FUNDING_TIMES)} UTC. "
                f"Strategy will avoid entries within "
                f"{st.session_state.get('form_funding_before', 5)}min before and "
                f"{st.session_state.get('form_funding_after', 5)}min after."
            )


def _render_weekly_schedule(time_filters: dict[str, Any]) -> None:
    """Render visual weekly schedule grid with session presets."""
    st.markdown("**Trading Sessions**")

    # Session presets
    st.caption("**Quick session presets:**")
    preset_names = list(SESSION_PRESETS.keys())
    cols = st.columns(len(preset_names))
    for i, name in enumerate(preset_names):
        with cols[i]:
            if st.button(
                name.split("(")[0].strip(),
                key=f"session_preset_{i}",
                use_container_width=True,
                help=name,
            ):
                sessions = SESSION_PRESETS[name]
                st.session_state["form_sessions"] = sessions
                st.rerun()

    # Current sessions display — deep-copy to avoid mutating original DSL dict
    raw_sessions = st.session_state.get(
        "form_sessions",
        time_filters.get("allowed_sessions", []),
    )
    sessions = [dict(s) for s in raw_sessions]

    if sessions:
        for i, session in enumerate(sessions):
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                new_start = st.text_input(
                    "Start",
                    value=session.get("start", "00:00"),
                    key=f"session_start_{i}",
                    help="HH:MM UTC format",
                )
            with c2:
                new_end = st.text_input(
                    "End",
                    value=session.get("end", "08:00"),
                    key=f"session_end_{i}",
                    help="HH:MM UTC format",
                )
            # Read back edited values into the session dict
            session["start"] = new_start
            session["end"] = new_end
            with c3:
                if st.button("X", key=f"session_rm_{i}"):
                    sessions.pop(i)
                    st.session_state["form_sessions"] = sessions
                    st.rerun()
        # Persist any edits back to session state
        st.session_state["form_sessions"] = sessions
    else:
        st.caption("No session restrictions (24/7 trading)")

    if st.button("+ Add Session", key="add_session"):
        sessions.append({"start": "00:00", "end": "08:00"})
        st.session_state["form_sessions"] = sessions
        st.rerun()

    # Visual schedule grid (text-based representation)
    blocked_days = st.session_state.get("form_blocked_days", [])
    if sessions or blocked_days:
        _render_schedule_summary(sessions, blocked_days)


def _render_schedule_summary(
    sessions: list[dict[str, str]],
    blocked_days: list[str],
) -> None:
    """Render a text summary of the trading schedule."""
    active_days = [d for d in ALL_DAYS if d not in blocked_days]

    with st.container(border=True):
        st.caption("**Schedule Summary**")

        # Days row
        day_labels = []
        for day in ALL_DAYS:
            short = day[:3]
            if day in blocked_days:
                day_labels.append(f":red[~~{short}~~]")
            else:
                day_labels.append(f":green[**{short}**]")
        st.markdown(" | ".join(day_labels))

        # Sessions info
        if sessions:
            session_strs = [f"{s['start']}-{s['end']} UTC" for s in sessions]
            st.caption(f"Active sessions: {', '.join(session_strs)}")
        else:
            st.caption("Sessions: 24/7 (no restriction)")

        st.caption(f"Active days: {len(active_days)}/7")
