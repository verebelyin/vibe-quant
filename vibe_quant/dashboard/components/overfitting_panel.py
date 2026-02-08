"""Overfitting protection panel for backtest launch.

Provides:
- Overfitting risk indicator (Low / Medium / High)
- DSR, WFA, Purged K-Fold toggle checkboxes
- Educational expander explaining why it matters
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from vibe_quant.db.state_manager import JsonDict


def render_overfitting_panel(
    strategy: JsonDict | None,
    start_date: str,
    end_date: str,
) -> dict[str, bool]:
    """Render overfitting filter toggles with risk indicator.

    Returns dict with keys ``enable_dsr``, ``enable_wfa``, ``enable_purged_kfold``.
    """
    st.subheader("Overfitting Protection")

    # Risk indicator
    if strategy:
        dsl = strategy["dsl_config"]
        sweep = dsl.get("sweep", {})
        combos = 1
        for v in sweep.values():
            combos *= len(v)
        n_conditions = (
            len(dsl.get("entry_conditions", {}).get("long", []))
            + len(dsl.get("entry_conditions", {}).get("short", []))
        )

        if combos > 500 or n_conditions > 6:
            risk_level, risk_color = "High", "red"
        elif combos > 50 or n_conditions > 3:
            risk_level, risk_color = "Medium", "orange"
        else:
            risk_level, risk_color = "Low", "green"

        col_risk, col_detail = st.columns([1, 3])
        with col_risk:
            st.markdown(f"**Overfitting Risk:** :{risk_color}[**{risk_level}**]")
        with col_detail:
            st.caption(
                f"{combos:,} parameter combinations | "
                f"{n_conditions} entry conditions | "
                f"Data: {start_date} to {end_date}"
            )
            if combos > 20:
                st.caption(
                    f"With {combos} comparisons, there is a "
                    f"~{min(99, int((1 - 0.95**combos) * 100))}% chance of finding "
                    f"a spuriously profitable combination by chance alone."
                )

    # Filter toggles
    col1, col2, col3 = st.columns(3)

    with col1:
        enable_dsr = st.checkbox(
            "Deflated Sharpe Ratio (DSR)", value=True, key="filter_dsr",
            help="Tests statistical significance of Sharpe ratio "
                 "accounting for multiple comparisons.",
        )
    with col2:
        enable_wfa = st.checkbox(
            "Walk-Forward Analysis (WFA)", value=True, key="filter_wfa",
            help="Splits data into in-sample and out-of-sample periods. "
                 "The gold standard for overfitting detection.",
        )
    with col3:
        enable_pkfold = st.checkbox(
            "Purged K-Fold CV", value=True, key="filter_pkfold",
            help="Cross-validation with purging to prevent data leakage "
                 "between train and test folds.",
        )

    # Educational context
    with st.expander("Why overfitting protection matters", expanded=False):
        st.markdown(
            "Research shows a **100% average performance gap** between in-sample "
            "and out-of-sample results for trading strategies. Without these filters:\n"
            "- Strategies that look profitable in backtests often fail in live trading\n"
            "- Testing many parameter combinations guarantees finding false positives\n"
            "- The more you optimize, the worse real performance typically gets\n\n"
            "**Recommendation:** Keep all three filters enabled."
        )

    return {
        "enable_dsr": enable_dsr,
        "enable_wfa": enable_wfa,
        "enable_purged_kfold": enable_pkfold,
    }
