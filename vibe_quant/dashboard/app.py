"""Main Streamlit dashboard application for vibe-quant.

Run with: streamlit run vibe_quant/dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_PAGES_DIR = Path(__file__).parent / "pages"


def main() -> None:
    """Main dashboard entry point using st.navigation API."""
    # st.set_page_config MUST be the first Streamlit command
    st.set_page_config(
        page_title="vibe-quant Dashboard",
        page_icon=":material/show_chart:",
        layout="wide",
    )

    from vibe_quant.dashboard.pages.backtest_launch import render_backtest_launch_tab
    from vibe_quant.dashboard.pages.data_management import render as render_data_management_tab
    from vibe_quant.dashboard.pages.discovery import render_discovery_tab
    from vibe_quant.dashboard.pages.paper_trading import render_paper_trading_tab
    from vibe_quant.dashboard.pages.results_analysis import render_results_tab
    from vibe_quant.dashboard.pages.settings import render_settings_tab
    from vibe_quant.dashboard.pages.strategy_management import render_strategy_management_tab

    pages = {
        "Strategies": [
            st.Page(
                render_strategy_management_tab,
                title="Strategy Management",
                icon=":material/edit_note:",
                default=True,
            ),
            st.Page(
                render_discovery_tab,
                title="Discovery",
                icon=":material/psychology:",
            ),
        ],
        "Backtesting": [
            st.Page(
                render_backtest_launch_tab,
                title="Backtest Launch",
                icon=":material/rocket_launch:",
            ),
            st.Page(
                render_results_tab,
                title="Results Analysis",
                icon=":material/analytics:",
            ),
        ],
        "Trading": [
            st.Page(
                render_paper_trading_tab,
                title="Paper Trading",
                icon=":material/candlestick_chart:",
            ),
        ],
        "System": [
            st.Page(
                render_data_management_tab,
                title="Data Management",
                icon=":material/database:",
            ),
            st.Page(
                render_settings_tab,
                title="Settings",
                icon=":material/settings:",
            ),
        ],
    }

    # Workaround: st.navigation() captures arrow-key events for page switching,
    # which interferes with sliders/number inputs. Hide the nav keyboard handler.
    st.html(
        "<style>"
        "[data-testid='stSidebarNav'] {pointer-events: auto;}"
        "</style>"
        "<script>"
        "window.addEventListener('keydown', function(e) {"
        "  var tag = e.target.tagName;"
        "  if (tag === 'INPUT' || tag === 'TEXTAREA' || "
        "      e.target.getAttribute('role') === 'slider') {"
        "    e.stopPropagation();"
        "  }"
        "}, true);"
        "</script>"
    )

    pg = st.navigation(pages)

    try:
        pg.run()
    except Exception as exc:
        # Prevent full server crash on page rendering errors (vibe-quant-4g6k)
        # Also handles "Page not found" from st.rerun() + navigation race (vibe-quant-g6ex)
        st.error(f"Page error: {exc}")
        st.info("Try refreshing the page or selecting a different page from the sidebar.")


if __name__ == "__main__":
    main()
