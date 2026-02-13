"""Backtest configuration components extracted from backtest_launch page.

Provides:
- Strategy selector with summary card
- Symbol/timeframe selector
- Date range selector with presets
- Sweep parameters form
- Sizing/risk config selectors
- Latency preset selector
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

import streamlit as st

from vibe_quant.dsl.schema import VALID_TIMEFRAMES
from vibe_quant.validation.latency import LATENCY_PRESETS, LatencyPreset

if TYPE_CHECKING:
    from vibe_quant.db.state_manager import JsonDict, StateManager

# Common crypto perpetual symbols
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
]

LATENCY_OPTIONS = ["None (screening mode)"] + [p.value for p in LatencyPreset] + ["custom"]


def render_strategy_selector(manager: StateManager) -> JsonDict | None:
    """Render strategy selector with rich summary card."""
    strategies = manager.list_strategies(active_only=True)
    strategies = [s for s in strategies if not s["name"].startswith("__")]
    if not strategies:
        st.warning("No active strategies found. Create one in Strategy Management tab.")
        return None

    strategy_options = {s["name"]: s for s in strategies}
    selected_name = st.selectbox(
        "Select Strategy", options=list(strategy_options.keys()),
        key="strategy_select", help="Choose a strategy to backtest",
    )

    if selected_name:
        strategy = strategy_options[selected_name]
        dsl = strategy["dsl_config"]
        indicators = dsl.get("indicators", {})
        entry = dsl.get("entry_conditions", {})
        sweep = dsl.get("sweep", {})

        with st.container(border=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"**{selected_name}** v{strategy.get('version', 1)}")
                st.caption(strategy.get("description") or "No description")
            with c2:
                st.metric("Timeframe", dsl.get("timeframe", "N/A"))
            with c3:
                st.metric("Indicators", len(indicators))
                st.caption(f"{len(entry.get('long', []))}L / {len(entry.get('short', []))}S entries")
            with c4:
                n_sweep = len(sweep)
                combos = 1
                for v in sweep.values():
                    combos *= len(v)
                st.metric("Sweep Params", n_sweep)
                if n_sweep:
                    st.caption(f"{combos:,} combinations")
        return strategy
    return None


def render_symbol_timeframe_selector(strategy: JsonDict | None) -> tuple[list[str], str]:
    """Render symbol and timeframe selectors."""
    c1, c2 = st.columns([2, 1])

    with c1:
        # Build dynamic options list that includes any previously added custom symbols
        custom_syms: list[str] = st.session_state.get("custom_symbols_list", [])
        all_symbol_options = DEFAULT_SYMBOLS + [s for s in custom_syms if s not in DEFAULT_SYMBOLS]
        symbols = st.multiselect(
            "Symbols", options=all_symbol_options, default=["BTCUSDT", "ETHUSDT"],
            key="symbols_select", help="Select one or more perpetual futures symbols",
        )
        custom = st.text_input("Add custom symbol", placeholder="e.g., ARBUSDT", key="custom_symbol")
        if custom and custom.upper() not in all_symbol_options and st.button("Add Symbol", key="add_symbol"):
            custom_syms.append(custom.upper())
            st.session_state["custom_symbols_list"] = custom_syms
            st.session_state["symbols_select"] = symbols + [custom.upper()]
            st.rerun()

    with c2:
        default_tf = strategy["dsl_config"].get("timeframe", "1h") if strategy else "1h"
        tf_list = sorted(VALID_TIMEFRAMES)
        timeframe = st.selectbox(
            "Timeframe", options=tf_list,
            index=tf_list.index(default_tf) if default_tf in tf_list else 0,
            key="timeframe_select",
        )

    return symbols, timeframe


def render_date_range_selector() -> tuple[str, str]:
    """Render date range selector with presets."""
    default_end = date.today()

    st.caption("**Quick presets:**")
    preset_cols = st.columns(5)
    presets = [("30 days", 30), ("90 days", 90), ("6 months", 180), ("1 year", 365), ("Full history", None)]
    for col, (label, days) in zip(preset_cols, presets, strict=False):
        with col:
            if st.button(label, key=f"date_preset_{label}", width="stretch"):
                if days is not None:
                    st.session_state["start_date"] = default_end - timedelta(days=days)
                else:
                    st.session_state["start_date"] = date(2019, 1, 1)
                st.session_state["end_date"] = default_end
                st.rerun()

    c1, c2, c3 = st.columns([2, 2, 1])
    default_start = default_end - timedelta(days=365)
    with c1:
        start_date = st.date_input(
            "Start Date", value=st.session_state.get("start_date", default_start),
            min_value=date(2019, 1, 1), max_value=default_end, key="start_date",
        )
    with c2:
        end_date = st.date_input(
            "End Date", value=st.session_state.get("end_date", default_end),
            min_value=start_date, max_value=date.today(), key="end_date",
        )
    with c3:
        st.metric("Duration", f"{(end_date - start_date).days} days")

    return start_date.isoformat(), end_date.isoformat()


def render_latency_selector() -> str | None:
    """Render latency preset selector."""
    st.subheader("Latency Model")
    c1, c2 = st.columns([2, 1])

    with c1:
        selected = st.selectbox(
            "Latency Preset", options=LATENCY_OPTIONS, index=0, key="latency_preset",
            help="Use None for screening (fast), presets for validation, or custom.",
        )

    with c2:
        if selected == "custom":
            base_ms = st.number_input("Base latency (ms)", min_value=0, value=50, step=1, key="custom_base_ms")
            insert_ms = st.number_input("Insert latency (ms)", min_value=0, value=25, step=1, key="custom_insert_ms")
            st.metric("Total Insert Latency", f"{base_ms + insert_ms} ms")
            st.session_state["custom_latency"] = {
                "base_latency_nanos": base_ms * 1_000_000,
                "insert_latency_nanos": insert_ms * 1_000_000,
            }
        elif selected != "None (screening mode)":
            preset = LatencyPreset(selected)
            values = LATENCY_PRESETS[preset]
            st.metric("Total Insert Latency", f"{values.base_ms + values.insert_ms} ms")
        else:
            st.metric("Total Insert Latency", "0 ms (screening)")

    return None if selected == "None (screening mode)" else selected
