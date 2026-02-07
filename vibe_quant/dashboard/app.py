"""Main Streamlit dashboard application for vibe-quant.

Run with: streamlit run vibe_quant/dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_PAGES_DIR = Path(__file__).parent / "pages"


def main() -> None:
    """Main dashboard entry point using st.navigation API."""
    pages = {
        "Strategies": [
            st.Page(
                _PAGES_DIR / "strategy_management.py",
                title="Strategy Management",
                icon=":material/edit_note:",
                default=True,
            ),
            st.Page(
                _PAGES_DIR / "discovery.py",
                title="Discovery",
                icon=":material/psychology:",
            ),
        ],
        "Backtesting": [
            st.Page(
                _PAGES_DIR / "backtest_launch.py",
                title="Backtest Launch",
                icon=":material/rocket_launch:",
            ),
            st.Page(
                _PAGES_DIR / "results_analysis.py",
                title="Results Analysis",
                icon=":material/analytics:",
            ),
        ],
        "Trading": [
            st.Page(
                _PAGES_DIR / "paper_trading.py",
                title="Paper Trading",
                icon=":material/candlestick_chart:",
            ),
        ],
        "System": [
            st.Page(
                _PAGES_DIR / "data_management.py",
                title="Data Management",
                icon=":material/database:",
            ),
            st.Page(
                _PAGES_DIR / "settings.py",
                title="Settings",
                icon=":material/settings:",
            ),
        ],
    }

    st.set_page_config(
        page_title="vibe-quant Dashboard",
        page_icon=":material/show_chart:",
        layout="wide",
    )
    pg = st.navigation(pages)
    pg.run()


if __name__ == "__main__":
    main()
