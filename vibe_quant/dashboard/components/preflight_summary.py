"""Pre-flight summary component for backtest launch.

Shows a summary card with strategy, symbols, period, mode,
total backtests, and active overfitting filters before launch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from vibe_quant.db.state_manager import JsonDict


def render_preflight_summary(
    strategy: JsonDict,
    symbols: list[str],
    timeframe: str,
    start_date: str,
    end_date: str,
    sweep_params: dict[str, list[int | float]],
    overfitting_filters: dict[str, bool],
    latency_preset: str | None,
) -> None:
    """Render pre-flight summary before launching backtest."""
    st.divider()
    st.subheader("Pre-flight Summary")

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"**Strategy:** {strategy['name']}")
            st.caption(f"Timeframe: {timeframe}")

        with col2:
            st.markdown(f"**Symbols:** {len(symbols)}")
            sym_display = ", ".join(symbols[:4])
            if len(symbols) > 4:
                sym_display += f" +{len(symbols) - 4}"
            st.caption(sym_display)

        with col3:
            st.markdown(f"**Period:** {start_date} to {end_date}")
            mode = "Screening" if latency_preset is None else "Validation"
            st.caption(f"Mode: {mode}")

        with col4:
            total_combos = 1
            for v in sweep_params.values():
                total_combos *= len(v)
            total_combos = max(1, total_combos)
            total_runs = total_combos * len(symbols)
            st.markdown(f"**Total backtests:** {total_runs:,}")
            filters_on = sum(1 for v in overfitting_filters.values() if v)
            st.caption(f"{filters_on}/3 overfitting filters active")
