"""Form state helpers for strategy editor.

Pure data logic: assembles DSL dicts from Streamlit session state
and manages form lifecycle (init / cleanup).
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def init_form_state(dsl: dict[str, Any]) -> None:
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


def cleanup_form_state() -> None:
    """Remove form-specific session state keys."""
    prefixes = ("form_", "cond_", "sweep_", "show_indicator_catalog", "template_applied")
    for key in list(st.session_state.keys()):
        if any(key.startswith(p) for p in prefixes):
            del st.session_state[key]


def build_dsl_from_form(original_dsl: dict[str, Any]) -> dict[str, Any]:
    """Assemble a complete DSL dict from form session state."""
    indicators = st.session_state.get("form_indicators", {})

    sl_type = st.session_state.get("form_sl_type", "fixed_pct")
    stop_loss: dict[str, Any] = {"type": sl_type}
    if sl_type == "fixed_pct":
        stop_loss["percent"] = st.session_state.get("form_sl_pct", 2.0)
    elif sl_type in ("atr_fixed", "atr_trailing"):
        stop_loss["atr_multiplier"] = st.session_state.get("form_sl_atr_mult", 2.0)
        stop_loss["indicator"] = st.session_state.get("form_sl_indicator", "")

    tp_type = st.session_state.get("form_tp_type", "risk_reward")
    take_profit: dict[str, Any] = {"type": tp_type}
    if tp_type == "fixed_pct":
        take_profit["percent"] = st.session_state.get("form_tp_pct", 4.0)
    elif tp_type == "atr_fixed":
        take_profit["atr_multiplier"] = st.session_state.get("form_tp_atr_mult", 3.0)
        take_profit["indicator"] = st.session_state.get("form_tp_indicator", "")
    elif tp_type == "risk_reward":
        take_profit["risk_reward_ratio"] = st.session_state.get("form_tp_rr", 2.0)

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
        "time_filters": {
            "allowed_sessions": original_dsl.get("time_filters", {}).get("allowed_sessions", []),
            "blocked_days": st.session_state.get("form_blocked_days", []),
            "avoid_around_funding": {
                "enabled": st.session_state.get("form_funding_enabled", False),
                "minutes_before": st.session_state.get("form_funding_before", 5),
                "minutes_after": st.session_state.get("form_funding_after", 5),
            },
        },
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "position_management": {"scale_in": {"enabled": False}, "partial_exit": {"enabled": False}},
        "sweep": st.session_state.get("sweep_sweep_params", {}),
    }
