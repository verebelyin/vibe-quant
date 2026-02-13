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
