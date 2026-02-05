"""Main Streamlit dashboard application for vibe-quant.

Run with: streamlit run vibe_quant/dashboard/app.py
"""

from __future__ import annotations

import streamlit as st

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
                "Backtest Results",
                "Data Management",
                "Settings",
            ],
            index=0,
        )

    # Route to pages
    if page == "Strategy Management":
        render_strategy_management_tab()
    elif page == "Backtest Results":
        st.info("Backtest Results tab - coming soon")
    elif page == "Data Management":
        st.info("Data Management tab - coming soon")
    elif page == "Settings":
        st.info("Settings tab - coming soon")


if __name__ == "__main__":
    main()
