"""Main Streamlit dashboard application for vibe-quant.

Run with: streamlit run vibe_quant/dashboard/app.py
"""

from __future__ import annotations

import streamlit as st

from vibe_quant.dashboard.pages.backtest_launch import render_backtest_launch_tab
from vibe_quant.dashboard.pages.data_management import render as render_data_management_tab
from vibe_quant.dashboard.pages.discovery import render_discovery_tab
from vibe_quant.dashboard.pages.paper_trading import render as render_paper_trading_tab
from vibe_quant.dashboard.pages.results_analysis import (
    render_results_tab as render_results_analysis_tab,
)
from vibe_quant.dashboard.pages.settings import render_settings_tab
from vibe_quant.dashboard.pages.strategy_management import render_strategy_management_tab


def main() -> None:
    """Main dashboard entry point."""
    st.set_page_config(
        page_title="vibe-quant Dashboard",
        page_icon="chart_with_upwards_trend",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("vibe-quant Dashboard")

    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select page",
            options=[
                "Strategy Management",
                "Backtest Launch",
                "Results Analysis",
                "Discovery",
                "Paper Trading",
                "Data Management",
                "Settings",
            ],
            index=0,
        )

    # Route to pages
    if page == "Strategy Management":
        render_strategy_management_tab()
    elif page == "Backtest Launch":
        render_backtest_launch_tab()
    elif page == "Results Analysis":
        render_results_analysis_tab()
    elif page == "Discovery":
        render_discovery_tab()
    elif page == "Paper Trading":
        render_paper_trading_tab()
    elif page == "Data Management":
        render_data_management_tab()
    elif page == "Settings":
        render_settings_tab()


if __name__ == "__main__":
    main()
