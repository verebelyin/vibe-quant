"""Results Analysis Tab for vibe-quant Streamlit dashboard.

Provides:
- Sweep results table (sortable, filterable)
- Pareto front scatter plot (Sharpe vs MaxDD, color=PF)
- Individual result detail: equity curve, drawdown chart, trade log
- Key metrics panel with cost breakdown
- Overfitting filter pass/fail badges
- Strategy comparison (side-by-side)
- CSV export
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from vibe_quant.db import StateManager

if TYPE_CHECKING:
    from typing import Any


def get_state_manager() -> StateManager:
    """Get or create StateManager in session state."""
    if "state_manager" not in st.session_state:
        st.session_state.state_manager = StateManager()
    mgr: StateManager = st.session_state.state_manager
    return mgr


def format_percent(val: float | None) -> str:
    """Format value as percentage."""
    if val is None:
        return "N/A"
    return f"{val * 100:.2f}%"


def format_number(val: float | None, decimals: int = 2) -> str:
    """Format numeric value."""
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def render_overfitting_badge(passed: bool | None, label: str) -> None:
    """Render pass/fail badge for overfitting filter."""
    if passed is None:
        st.markdown(f"**{label}**: :gray[N/A]")
    elif passed:
        st.markdown(f"**{label}**: :green[PASS]")
    else:
        st.markdown(f"**{label}**: :red[FAIL]")


def get_runs_for_dropdown(mgr: StateManager) -> list[dict[str, Any]]:
    """Get backtest runs with strategy names for dropdown."""
    runs = mgr.list_backtest_runs()
    result = []
    for run in runs:
        strategy = mgr.get_strategy(run["strategy_id"])
        strategy_name = strategy["name"] if strategy else "Unknown"
        label = f"{run['id']} - {strategy_name} ({run['run_mode']}) - {run['created_at'][:10]}"
        result.append({"id": run["id"], "label": label, "run": run})
    return result


def build_sweep_dataframe(sweep_results: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert sweep results to DataFrame for display."""
    if not sweep_results:
        return pd.DataFrame()

    rows = []
    for r in sweep_results:
        row = {
            "ID": r["id"],
            "Sharpe": r.get("sharpe_ratio"),
            "Sortino": r.get("sortino_ratio"),
            "Max DD": r.get("max_drawdown"),
            "Return": r.get("total_return"),
            "PF": r.get("profit_factor"),
            "Win Rate": r.get("win_rate"),
            "Trades": r.get("total_trades"),
            "Fees": r.get("total_fees"),
            "Funding": r.get("total_funding"),
            "Pareto": r.get("is_pareto_optimal", 0),
            "DSR Pass": r.get("passed_deflated_sharpe"),
            "WFA Pass": r.get("passed_walk_forward"),
            "KFold Pass": r.get("passed_purged_kfold"),
        }
        # Flatten parameters
        params = r.get("parameters", {})
        for k, v in params.items():
            row[f"param_{k}"] = v
        rows.append(row)

    return pd.DataFrame(rows)


def build_trades_dataframe(trades: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert trades to DataFrame for display."""
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    display_cols = [
        "symbol",
        "direction",
        "leverage",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "quantity",
        "gross_pnl",
        "net_pnl",
        "entry_fee",
        "exit_fee",
        "funding_fees",
        "slippage_cost",
        "roi_percent",
        "exit_reason",
    ]
    available = [c for c in display_cols if c in df.columns]
    return df[available]


def build_equity_curve(trades: list[dict[str, Any]]) -> pd.DataFrame:
    """Build equity curve from trades."""
    if not trades:
        return pd.DataFrame()

    # Sort by exit time, filter closed trades
    closed = [t for t in trades if t.get("exit_time")]
    if not closed:
        return pd.DataFrame()

    sorted_trades = sorted(closed, key=lambda x: x["exit_time"])
    cumulative_pnl = 0.0
    equity_points = [{"time": sorted_trades[0]["entry_time"], "equity": 0.0}]

    for trade in sorted_trades:
        pnl = trade.get("net_pnl", 0.0) or 0.0
        cumulative_pnl += pnl
        equity_points.append({"time": trade["exit_time"], "equity": cumulative_pnl})

    return pd.DataFrame(equity_points)


def compute_drawdown_series(equity_df: pd.DataFrame) -> pd.DataFrame:
    """Compute drawdown series from equity curve."""
    if equity_df.empty:
        return pd.DataFrame()

    equity = equity_df["equity"].values
    peak = equity[0]
    drawdowns = []

    for i, val in enumerate(equity):
        if val > peak:
            peak = val
        dd = (peak - val) / max(peak, 1e-9) if peak > 0 else 0
        drawdowns.append({"time": equity_df.iloc[i]["time"], "drawdown": dd})

    return pd.DataFrame(drawdowns)


def render_pareto_scatter(sweep_df: pd.DataFrame) -> None:
    """Render Pareto front scatter plot."""
    if sweep_df.empty:
        st.info("No sweep results to display")
        return

    # Filter out rows with missing data
    plot_df = sweep_df.dropna(subset=["Sharpe", "Max DD"])
    if plot_df.empty:
        st.info("No complete results for scatter plot")
        return

    # Invert Max DD for display (make it negative = worse)
    plot_df = plot_df.copy()
    plot_df["Max DD (%)"] = plot_df["Max DD"] * -100

    fig = px.scatter(
        plot_df,
        x="Sharpe",
        y="Max DD (%)",
        color="PF",
        size="Trades",
        hover_data=["ID", "Return", "Win Rate"],
        color_continuous_scale="Viridis",
        title="Pareto Front: Sharpe vs Max Drawdown",
    )

    # Highlight Pareto optimal points
    pareto_df = plot_df[plot_df["Pareto"] == 1]
    if not pareto_df.empty:
        fig.add_trace(
            go.Scatter(
                x=pareto_df["Sharpe"],
                y=pareto_df["Max DD (%)"],
                mode="markers",
                marker={"size": 15, "symbol": "diamond", "line": {"width": 2, "color": "red"}},
                name="Pareto Optimal",
                hoverinfo="skip",
            )
        )

    fig.update_layout(height=500)
    st.plotly_chart(fig, width="stretch")


def render_equity_chart(equity_df: pd.DataFrame) -> None:
    """Render equity curve chart."""
    if equity_df.empty:
        st.info("No equity data to display")
        return

    fig = px.line(
        equity_df, x="time", y="equity", title="Equity Curve", labels={"equity": "Cumulative P&L", "time": "Time"}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, width="stretch")


def render_drawdown_chart(dd_df: pd.DataFrame) -> None:
    """Render drawdown chart."""
    if dd_df.empty:
        st.info("No drawdown data to display")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd_df["time"],
            y=dd_df["drawdown"] * -100,
            fill="tozeroy",
            fillcolor="rgba(255, 0, 0, 0.3)",
            line={"color": "red"},
            name="Drawdown",
        )
    )
    fig.update_layout(
        title="Drawdown Chart",
        xaxis_title="Time",
        yaxis_title="Drawdown (%)",
        height=300,
    )
    st.plotly_chart(fig, width="stretch")


def render_metrics_panel(result: dict[str, Any]) -> None:
    """Render key metrics panel."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Return", format_percent(result.get("total_return")))
        st.metric("CAGR", format_percent(result.get("cagr")))
        st.metric("Volatility", format_percent(result.get("volatility_annual")))

    with col2:
        st.metric("Sharpe Ratio", format_number(result.get("sharpe_ratio")))
        st.metric("Sortino Ratio", format_number(result.get("sortino_ratio")))
        st.metric("Calmar Ratio", format_number(result.get("calmar_ratio")))

    with col3:
        st.metric("Max Drawdown", format_percent(result.get("max_drawdown")))
        st.metric("DD Duration (days)", format_number(result.get("max_drawdown_duration_days"), 0))
        st.metric("Win Rate", format_percent(result.get("win_rate")))

    with col4:
        st.metric("Total Trades", format_number(result.get("total_trades"), 0))
        st.metric("Profit Factor", format_number(result.get("profit_factor")))
        st.metric("Avg Trade Duration (hrs)", format_number(result.get("avg_trade_duration_hours"), 1))


def render_cost_breakdown(result: dict[str, Any]) -> None:
    """Render cost breakdown panel."""
    st.subheader("Cost Breakdown")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Fees", f"${format_number(result.get('total_fees'))}")

    with col2:
        st.metric("Total Funding", f"${format_number(result.get('total_funding'))}")

    with col3:
        st.metric("Total Slippage", f"${format_number(result.get('total_slippage'))}")


def render_overfitting_results(result: dict[str, Any]) -> None:
    """Render overfitting filter results."""
    st.subheader("Overfitting Filter Results")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_overfitting_badge(result.get("passed_deflated_sharpe"), "Deflated Sharpe")
        if result.get("deflated_sharpe") is not None:
            st.caption(f"DSR: {format_number(result.get('deflated_sharpe'))}")

    with col2:
        render_overfitting_badge(result.get("passed_walk_forward"), "Walk-Forward")
        if result.get("walk_forward_efficiency") is not None:
            st.caption(f"Efficiency: {format_percent(result.get('walk_forward_efficiency'))}")

    with col3:
        render_overfitting_badge(result.get("passed_purged_kfold"), "Purged K-Fold")
        if result.get("purged_kfold_mean_sharpe") is not None:
            st.caption(f"Mean Sharpe: {format_number(result.get('purged_kfold_mean_sharpe'))}")

    with col4:
        # Overall pass status
        filters = [
            result.get("passed_deflated_sharpe"),
            result.get("passed_walk_forward"),
            result.get("passed_purged_kfold"),
        ]
        active_filters = [f for f in filters if f is not None]
        if active_filters:
            all_pass = all(active_filters)
            if all_pass:
                st.success("All filters PASSED")
            else:
                st.error("Some filters FAILED")
        else:
            st.info("No filters applied")


def render_comparison_view(mgr: StateManager, run_ids: list[int]) -> None:
    """Render side-by-side comparison of runs."""
    if len(run_ids) < 2:
        st.info("Select at least 2 runs to compare")
        return

    results = []
    for run_id in run_ids[:3]:  # Max 3 comparisons
        result = mgr.get_backtest_result(run_id)
        run = mgr.get_backtest_run(run_id)
        if result and run:
            strategy = mgr.get_strategy(run["strategy_id"])
            result["strategy_name"] = strategy["name"] if strategy else "Unknown"
            result["run_mode"] = run["run_mode"]
            results.append(result)

    if not results:
        st.warning("No results found for selected runs")
        return

    # Build comparison DataFrame
    metrics = [
        ("Strategy", "strategy_name"),
        ("Mode", "run_mode"),
        ("Total Return", "total_return"),
        ("Sharpe", "sharpe_ratio"),
        ("Sortino", "sortino_ratio"),
        ("Max DD", "max_drawdown"),
        ("Win Rate", "win_rate"),
        ("Profit Factor", "profit_factor"),
        ("Total Trades", "total_trades"),
        ("Total Fees", "total_fees"),
        ("DSR Pass", "passed_deflated_sharpe"),
        ("WFA Pass", "passed_walk_forward"),
        ("KFold Pass", "passed_purged_kfold"),
    ]

    comparison_data: dict[str, dict[str, str]] = {}
    for i, result in enumerate(results):
        col_name = f"Run {run_ids[i]}"
        comparison_data[col_name] = {}
        for label, key in metrics:
            val = result.get(key)
            if key in ("total_return", "max_drawdown", "win_rate"):
                comparison_data[col_name][label] = format_percent(val)
            elif isinstance(val, bool):
                comparison_data[col_name][label] = "PASS" if val else "FAIL"
            elif val is None:
                comparison_data[col_name][label] = "N/A"
            elif isinstance(val, float):
                comparison_data[col_name][label] = format_number(val)
            else:
                comparison_data[col_name][label] = str(val)

    df = pd.DataFrame(comparison_data)
    st.dataframe(df, width="stretch")


def export_to_csv(df: pd.DataFrame, filename: str) -> bytes:
    """Export DataFrame to CSV bytes."""
    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue().encode("utf-8")


def render_results_tab() -> None:
    """Render the Results Analysis tab."""
    st.header("Results Analysis")

    mgr = get_state_manager()

    # Run selection
    runs = get_runs_for_dropdown(mgr)
    if not runs:
        st.info("No backtest runs found. Run a backtest first.")
        return

    # Create tabs for different views
    view_tab, compare_tab = st.tabs(["Single Run Analysis", "Compare Runs"])

    with view_tab:
        # Run selector
        run_options = {r["label"]: r["id"] for r in runs}
        selected_label = st.selectbox("Select Run", options=list(run_options.keys()), key="single_run_select")
        selected_run_id = run_options[selected_label]

        run = mgr.get_backtest_run(selected_run_id)
        if not run:
            st.error("Run not found")
            return

        # Show run info
        strategy = mgr.get_strategy(run["strategy_id"])
        strategy_name = strategy["name"] if strategy else "Unknown"

        with st.expander("Run Details", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Strategy:** {strategy_name}")
                st.write(f"**Mode:** {run['run_mode']}")
                st.write(f"**Symbols:** {', '.join(run['symbols'])}")
                st.write(f"**Timeframe:** {run['timeframe']}")
            with col2:
                st.write(f"**Period:** {run['start_date']} to {run['end_date']}")
                st.write(f"**Status:** {run['status']}")
                st.write(f"**Created:** {run['created_at']}")
                if run.get("latency_preset"):
                    st.write(f"**Latency:** {run['latency_preset']}")

        # Check if this is a sweep run
        sweep_results = mgr.get_sweep_results(selected_run_id)
        backtest_result = mgr.get_backtest_result(selected_run_id)

        if sweep_results:
            # Sweep results view
            st.subheader("Sweep Results")

            sweep_df = build_sweep_dataframe(sweep_results)

            # Filters
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            with filter_col1:
                show_pareto_only = st.checkbox("Pareto optimal only", value=False)
            with filter_col2:
                min_sharpe = st.number_input("Min Sharpe", value=0.0, step=0.1)
            with filter_col3:
                max_dd = st.number_input("Max Drawdown %", value=100.0, step=1.0)

            # Apply filters
            filtered_df = sweep_df.copy()
            if show_pareto_only:
                filtered_df = filtered_df[filtered_df["Pareto"] == 1]
            if min_sharpe > 0:
                filtered_df = filtered_df[filtered_df["Sharpe"] >= min_sharpe]
            if max_dd < 100:
                filtered_df = filtered_df[(filtered_df["Max DD"].abs() * 100) <= max_dd]

            # Pareto scatter plot
            render_pareto_scatter(filtered_df)

            # Results table
            st.dataframe(
                filtered_df.style.format(
                    {
                        "Sharpe": "{:.2f}",
                        "Sortino": "{:.2f}",
                        "Max DD": "{:.2%}",
                        "Return": "{:.2%}",
                        "PF": "{:.2f}",
                        "Win Rate": "{:.2%}",
                    },
                    na_rep="N/A",
                ),
                width="stretch",
                height=400,
            )

            # Export button
            csv_data = export_to_csv(filtered_df, f"sweep_results_{selected_run_id}.csv")
            st.download_button(
                label="Export to CSV",
                data=csv_data,
                file_name=f"sweep_results_{selected_run_id}.csv",
                mime="text/csv",
            )

            # Select individual sweep result for detail
            st.subheader("Individual Result Detail")
            sweep_ids = [r["id"] for r in sweep_results]
            selected_sweep_id = st.selectbox("Select sweep result", options=sweep_ids, key="sweep_detail_select")

            sweep_result = next((r for r in sweep_results if r["id"] == selected_sweep_id), None)
            if sweep_result:
                # Show parameters
                with st.expander("Parameters", expanded=True):
                    st.json(sweep_result.get("parameters", {}))

                # Show overfitting results for this sweep result
                col1, col2, col3 = st.columns(3)
                with col1:
                    render_overfitting_badge(sweep_result.get("passed_deflated_sharpe"), "Deflated Sharpe")
                with col2:
                    render_overfitting_badge(sweep_result.get("passed_walk_forward"), "Walk-Forward")
                with col3:
                    render_overfitting_badge(sweep_result.get("passed_purged_kfold"), "Purged K-Fold")

        elif backtest_result:
            # Single backtest result view
            render_metrics_panel(backtest_result)
            render_cost_breakdown(backtest_result)
            render_overfitting_results(backtest_result)

            # Get trades
            trades = mgr.get_trades(selected_run_id)

            if trades:
                # Equity curve
                equity_df = build_equity_curve(trades)
                render_equity_chart(equity_df)

                # Drawdown chart
                dd_df = compute_drawdown_series(equity_df)
                render_drawdown_chart(dd_df)

                # Trade log
                st.subheader("Trade Log")
                trades_df = build_trades_dataframe(trades)

                # Trade filters
                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    direction_filter = st.multiselect(
                        "Direction", options=["LONG", "SHORT"], default=["LONG", "SHORT"]
                    )
                with filter_col2:
                    if "symbol" in trades_df.columns:
                        symbols = trades_df["symbol"].unique().tolist()
                        symbol_filter = st.multiselect("Symbol", options=symbols, default=symbols)
                    else:
                        symbol_filter = []

                # Apply filters
                filtered_trades = trades_df.copy()
                if direction_filter and "direction" in filtered_trades.columns:
                    filtered_trades = filtered_trades[filtered_trades["direction"].isin(direction_filter)]
                if symbol_filter and "symbol" in filtered_trades.columns:
                    filtered_trades = filtered_trades[filtered_trades["symbol"].isin(symbol_filter)]

                st.dataframe(filtered_trades, width="stretch", height=400)

                # Export trades
                csv_data = export_to_csv(filtered_trades, f"trades_{selected_run_id}.csv")
                st.download_button(
                    label="Export Trades to CSV",
                    data=csv_data,
                    file_name=f"trades_{selected_run_id}.csv",
                    mime="text/csv",
                )
            else:
                st.info("No trades recorded for this run")

        else:
            st.warning(f"No results found for run {selected_run_id}. Status: {run['status']}")

    with compare_tab:
        st.subheader("Compare Multiple Runs")

        # Multi-select for comparison
        run_options_multi = {r["label"]: r["id"] for r in runs}
        selected_labels = st.multiselect(
            "Select runs to compare (max 3)", options=list(run_options_multi.keys()), max_selections=3
        )

        if selected_labels:
            selected_ids = [run_options_multi[label] for label in selected_labels]
            render_comparison_view(mgr, selected_ids)


# Top-level call for st.navigation API
render_results_tab()
