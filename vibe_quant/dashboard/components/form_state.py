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


def sync_form_state(dsl: dict[str, Any]) -> None:
    """Force-sync session state for form fields from DSL dict.

    Unlike init_form_state, this ALWAYS overwrites current form state.
    Used when switching from YAML to Visual mode.
    """
    st.session_state["form_indicators"] = dict(dsl.get("indicators", {}))
    entry = dsl.get("entry_conditions", {})
    st.session_state["form_entry_long"] = list(entry.get("long", []))
    st.session_state["form_entry_short"] = list(entry.get("short", []))
    exit_cond = dsl.get("exit_conditions", {})
    st.session_state["form_exit_long"] = list(exit_cond.get("long", []))
    st.session_state["form_exit_short"] = list(exit_cond.get("short", []))
    # Also sync risk fields — use ``is not None`` to preserve 0.0 values.
    # Clear stale keys when the DSL key is absent so a previous strategy's
    # values don't leak through.
    sl = dsl.get("stop_loss", {})
    st.session_state["form_sl_type"] = sl.get("type", "fixed_pct")
    if sl.get("percent") is not None:
        st.session_state["form_sl_pct"] = sl["percent"]
    else:
        st.session_state.pop("form_sl_pct", None)
    if sl.get("atr_multiplier") is not None:
        st.session_state["form_sl_atr_mult"] = sl["atr_multiplier"]
    else:
        st.session_state.pop("form_sl_atr_mult", None)
    if sl.get("indicator") is not None:
        st.session_state["form_sl_indicator"] = sl["indicator"]
    else:
        st.session_state.pop("form_sl_indicator", None)
    tp = dsl.get("take_profit", {})
    st.session_state["form_tp_type"] = tp.get("type", "risk_reward")
    if tp.get("percent") is not None:
        st.session_state["form_tp_pct"] = tp["percent"]
    else:
        st.session_state.pop("form_tp_pct", None)
    if tp.get("atr_multiplier") is not None:
        st.session_state["form_tp_atr_mult"] = tp["atr_multiplier"]
    else:
        st.session_state.pop("form_tp_atr_mult", None)
    if tp.get("indicator") is not None:
        st.session_state["form_tp_indicator"] = tp["indicator"]
    else:
        st.session_state.pop("form_tp_indicator", None)
    if tp.get("risk_reward_ratio") is not None:
        st.session_state["form_tp_rr"] = tp["risk_reward_ratio"]
    else:
        st.session_state.pop("form_tp_rr", None)
    # Sync time filters (including allowed_sessions for session editor)
    tf = dsl.get("time_filters", {})
    st.session_state["form_sessions"] = tf.get("allowed_sessions", [])
    st.session_state["form_blocked_days"] = tf.get("blocked_days", [])
    funding = tf.get("avoid_around_funding", {})
    st.session_state["form_funding_enabled"] = funding.get("enabled", False)
    st.session_state["form_funding_before"] = funding.get("minutes_before", 5)
    st.session_state["form_funding_after"] = funding.get("minutes_after", 5)
    # Sync sweep params
    st.session_state["sweep_sweep_params"] = dsl.get("sweep", {})
    # Clear any condition builder visual row caches so they reinitialize
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith("cond_") and key.endswith("_rows"):
            del st.session_state[key]


def cleanup_form_state() -> None:
    """Remove form-specific session state keys."""
    prefixes = ("form_", "cond_", "sweep_", "show_indicator_catalog", "template_applied")
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and any(key.startswith(p) for p in prefixes):
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
            "allowed_sessions": st.session_state.get(
                "form_sessions",
                original_dsl.get("time_filters", {}).get("allowed_sessions", []),
            ),
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


def validate_dsl_yaml(yaml_str: str) -> tuple[Any, str | None]:
    """Validate DSL YAML string. Returns (model, error).

    Shared validation helper used by both YAML and Visual editors.
    """
    import yaml as _yaml
    from pydantic import ValidationError as _ValidationError

    from vibe_quant.dsl.schema import StrategyDSL as _StrategyDSL

    try:
        data = _yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            return None, "YAML must be a mapping"
        return _StrategyDSL.model_validate(data), None
    except _yaml.YAMLError as e:
        return None, f"YAML parse error: {e}"
    except _ValidationError as e:
        return None, _format_validation_errors(e)


def validate_dsl_dict(dsl_dict: dict[str, Any]) -> tuple[Any, str | None]:
    """Validate a DSL dict directly (avoids YAML round-trip).

    Shared validation helper used by both YAML and Visual editors.
    """
    from pydantic import ValidationError as _ValidationError

    from vibe_quant.dsl.schema import StrategyDSL as _StrategyDSL

    try:
        return _StrategyDSL.model_validate(dsl_dict), None
    except _ValidationError as e:
        return None, _format_validation_errors(e)


def _format_validation_errors(e: Any) -> str:
    """Format pydantic ValidationError into a human-readable string."""
    errors = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"])
        errors.append(f"{loc}: {err['msg']}")
    return "\n".join(errors)
