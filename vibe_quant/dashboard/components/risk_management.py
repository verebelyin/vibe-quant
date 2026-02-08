"""Risk management component for strategy editor.

Provides:
- Stop loss / take profit conditional form fields
- Risk/reward ratio visualization
- Quick risk presets (Conservative / Moderate / Aggressive)
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from vibe_quant.dsl.schema import VALID_STOP_LOSS_TYPES, VALID_TAKE_PROFIT_TYPES


def _get_atr_indicator_names(indicators: dict[str, dict[str, Any]]) -> list[str]:
    """Extract names of ATR indicators from the indicator config."""
    return [
        name for name, cfg in indicators.items()
        if cfg.get("type", "").upper() == "ATR"
    ]


def _safe_index(options: list[str], value: str) -> int:
    """Return index of *value* in *options*, or 0 if missing."""
    return options.index(value) if value in options else 0


def _apply_risk_preset(
    sl_type: str, sl_pct: float, tp_type: str, rr: float,
) -> None:
    """Apply a risk management preset to session state."""
    st.session_state["form_sl_type"] = sl_type
    st.session_state["form_sl_pct"] = sl_pct
    st.session_state["form_tp_type"] = tp_type
    st.session_state["form_tp_rr"] = rr


# ── Stop Loss ────────────────────────────────────────────────────────────


def _render_stop_loss(dsl: dict[str, Any], atr_names: list[str]) -> None:
    st.markdown("**Stop Loss**")
    sl = dsl.get("stop_loss", {})

    sl_types = sorted(VALID_STOP_LOSS_TYPES)
    sl_type = st.selectbox(
        "Type",
        options=sl_types,
        index=_safe_index(sl_types, sl.get("type", "fixed_pct")),
        key="form_sl_type",
    )

    if sl_type == "fixed_pct":
        st.number_input(
            "Stop Loss %", value=sl.get("percent") or 2.0,
            min_value=0.1, max_value=50.0, step=0.1, key="form_sl_pct",
            help="Fixed percentage stop loss from entry price",
        )
    elif sl_type in ("atr_fixed", "atr_trailing"):
        st.number_input(
            "ATR Multiplier", value=sl.get("atr_multiplier") or 2.0,
            min_value=0.5, max_value=10.0, step=0.1, key="form_sl_atr_mult",
            help=f"Stop distance = ATR x multiplier. "
                 f"{'Trailing stop follows price.' if sl_type == 'atr_trailing' else 'Fixed distance from entry.'}",
        )
        if atr_names:
            st.selectbox(
                "ATR Indicator", options=atr_names,
                index=_safe_index(atr_names, sl.get("indicator", "")),
                key="form_sl_indicator",
            )
        else:
            st.warning("No ATR indicator defined. Add one in the Indicators section.")
            st.text_input("ATR Indicator Name", value=sl.get("indicator", ""), key="form_sl_indicator")


# ── Take Profit ──────────────────────────────────────────────────────────


def _render_take_profit(dsl: dict[str, Any], atr_names: list[str]) -> None:
    st.markdown("**Take Profit**")
    tp = dsl.get("take_profit", {})

    tp_types = sorted(VALID_TAKE_PROFIT_TYPES)
    tp_type = st.selectbox(
        "Type",
        options=tp_types,
        index=_safe_index(tp_types, tp.get("type", "risk_reward")),
        key="form_tp_type",
    )

    if tp_type == "fixed_pct":
        st.number_input(
            "Take Profit %", value=tp.get("percent") or 4.0,
            min_value=0.1, max_value=100.0, step=0.1, key="form_tp_pct",
            help="Fixed percentage take profit from entry price",
        )
    elif tp_type == "atr_fixed":
        st.number_input(
            "ATR Multiplier", value=tp.get("atr_multiplier") or 3.0,
            min_value=0.5, max_value=20.0, step=0.1, key="form_tp_atr_mult",
            help="Take profit distance = ATR x multiplier",
        )
        if atr_names:
            st.selectbox(
                "ATR Indicator", options=atr_names,
                index=_safe_index(atr_names, tp.get("indicator", "")),
                key="form_tp_indicator",
            )
        else:
            st.warning("No ATR indicator defined. Add one in the Indicators section.")
            st.text_input("ATR Indicator Name", value=tp.get("indicator", ""), key="form_tp_indicator")
    elif tp_type == "risk_reward":
        st.slider(
            "Risk/Reward Ratio", min_value=0.5, max_value=5.0,
            value=tp.get("risk_reward_ratio") or 2.0, step=0.1,
            key="form_tp_rr",
            help="Take profit = stop loss distance x R:R ratio",
        )


# ── R:R Visualization ────────────────────────────────────────────────────


def _render_rr_visualization() -> None:
    """Render a simple risk/reward ratio visualization."""
    sl_type = st.session_state.get("form_sl_type", "fixed_pct")
    tp_type = st.session_state.get("form_tp_type", "risk_reward")

    # SL distance
    if sl_type == "fixed_pct":
        sl_dist = st.session_state.get("form_sl_pct", 2.0)
        sl_label = f"{sl_dist}%"
    else:
        sl_dist = st.session_state.get("form_sl_atr_mult", 2.0)
        sl_label = f"{sl_dist}x ATR"

    # TP distance
    if tp_type == "fixed_pct":
        tp_dist = st.session_state.get("form_tp_pct", 4.0)
        tp_label = f"{tp_dist}%"
    elif tp_type == "risk_reward":
        rr = st.session_state.get("form_tp_rr", 2.0)
        tp_dist = sl_dist * rr
        tp_label = f"{rr}:1 R:R"
    else:
        tp_dist = st.session_state.get("form_tp_atr_mult", 3.0)
        tp_label = f"{tp_dist}x ATR"

    rr_ratio = tp_dist / sl_dist if sl_dist > 0 else 0
    rr_color = "green" if rr_ratio >= 2.0 else "orange" if rr_ratio >= 1.0 else "red"

    st.divider()
    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        st.markdown(f":red[**Stop Loss:** {sl_label}]")
    with c2:
        st.markdown(f":{rr_color}[**R:R {rr_ratio:.1f}**]")
    with c3:
        st.markdown(f":green[**Take Profit:** {tp_label}]")


# ── Public API ───────────────────────────────────────────────────────────


def render_risk_section(dsl: dict[str, Any]) -> None:
    """Render full risk management section with conditional fields and R:R viz."""
    indicators = st.session_state.get("form_indicators", {})
    atr_names = _get_atr_indicator_names(indicators)

    col_sl, col_tp = st.columns(2)
    with col_sl:
        _render_stop_loss(dsl, atr_names)
    with col_tp:
        _render_take_profit(dsl, atr_names)

    _render_rr_visualization()

    # Quick presets
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
