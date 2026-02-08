"""Results Analysis Tab for vibe-quant Streamlit dashboard.

Thin UI layer — data loading lives in :mod:`dashboard.data_builders`,
chart construction lives in :mod:`dashboard.charts`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from vibe_quant.dashboard import charts
from vibe_quant.dashboard.data_builders import (
    build_equity_curve,
    build_sweep_dataframe,
    build_trades_dataframe,
    compute_daily_returns,
    compute_drawdown_series,
    compute_long_short_split,
    compute_rolling_sharpe,
    compute_top_drawdown_periods,
    compute_yearly_returns,
    export_to_csv,
)
from vibe_quant.dashboard.utils import (
    format_dollar,
    format_number,
    format_percent,
    get_state_manager,
)

if TYPE_CHECKING:
    from typing import Any

    from vibe_quant.db.state_manager import StateManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _show_figure(fig: object | None, empty_msg: str = "") -> None:
    """Display a Plotly figure or show an info message when *None*."""
    if fig is None:
        if empty_msg:
            st.info(empty_msg)
    else:
        st.plotly_chart(fig, use_container_width=True)


def render_overfitting_badge(passed: bool | None, label: str) -> None:
    """Render pass/fail badge for overfitting filter."""
    if passed is None:
        st.markdown(f"**{label}**: :gray[N/A]")
    elif passed:
        st.markdown(f"**{label}**: :green[PASS]")
    else:
        st.markdown(f"**{label}**: :red[FAIL]")


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60)
def get_runs_for_dropdown(_mgr: StateManager) -> list[dict[str, Any]]:
    """Get backtest runs with strategy names for dropdown."""
    runs = _mgr.list_backtest_runs()
    result = []
    for run in runs:
        strategy = _mgr.get_strategy(run["strategy_id"])
        strategy_name = strategy["name"] if strategy else "Unknown"
        label = f"{run['id']} - {strategy_name} ({run['run_mode']}) - {run['created_at'][:10]}"
        result.append({"id": run["id"], "label": label, "run": run})
    return result


# ---------------------------------------------------------------------------
# Metrics panels
# ---------------------------------------------------------------------------


def render_metrics_panel(result: dict[str, Any]) -> None:
    """Render enhanced key metrics panel with derived metrics."""
    st.subheader("Performance Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)

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

    with col5:
        st.metric("Winning Trades", format_number(result.get("winning_trades"), 0))
        st.metric("Losing Trades", format_number(result.get("losing_trades"), 0))
        exec_time = result.get("execution_time_seconds")
        st.metric("Execution Time", f"{format_number(exec_time, 1)}s" if exec_time else "N/A")

    # Win/Loss detail
    st.subheader("Win/Loss Analysis")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Avg Win", format_dollar(result.get("avg_win")))
        st.metric("Avg Loss", format_dollar(result.get("avg_loss")))

    with col2:
        st.metric("Largest Win", format_dollar(result.get("largest_win")))
        st.metric("Largest Loss", format_dollar(result.get("largest_loss")))

    with col3:
        st.metric("Max Consecutive Wins", format_number(result.get("max_consecutive_wins"), 0))
        st.metric("Max Consecutive Losses", format_number(result.get("max_consecutive_losses"), 0))

    with col4:
        win_rate = result.get("win_rate")
        avg_win = result.get("avg_win")
        avg_loss = result.get("avg_loss")
        if win_rate is not None and avg_win is not None and avg_loss is not None:
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
            st.metric("Expectancy", format_dollar(expectancy))
        else:
            st.metric("Expectancy", "N/A")

        if avg_win is not None and avg_loss is not None and avg_loss != 0:
            payoff = abs(avg_win / avg_loss)
            st.metric("Payoff Ratio", format_number(payoff))
        else:
            st.metric("Payoff Ratio", "N/A")


def render_cost_breakdown(result: dict[str, Any]) -> None:
    """Render cost breakdown panel with cost drag percentage."""
    st.subheader("Cost Breakdown")
    col1, col2, col3, col4 = st.columns(4)

    total_fees = result.get("total_fees", 0.0) or 0.0
    total_funding = result.get("total_funding", 0.0) or 0.0
    total_slippage = result.get("total_slippage", 0.0) or 0.0
    total_costs = total_fees + total_funding + total_slippage

    with col1:
        st.metric("Total Fees", format_dollar(total_fees))
    with col2:
        st.metric("Total Funding", format_dollar(total_funding))
    with col3:
        st.metric("Total Slippage", format_dollar(total_slippage))
    with col4:
        total_return = result.get("total_return")
        starting_balance = result.get("starting_balance", 100000) or 100000
        if total_return is not None and total_return != 0:
            net_pnl = total_return * starting_balance
            gross_pnl = net_pnl + total_costs
            if gross_pnl != 0:
                cost_drag = (total_costs / abs(gross_pnl)) * 100
                st.metric("Cost Drag", f"{cost_drag:.1f}%")
            else:
                st.metric("Cost Drag", "N/A")
        else:
            st.metric("Cost Drag", "N/A")


def render_funding_impact(result: dict[str, Any], trades: list[dict[str, Any]]) -> None:
    """Render funding impact analysis for perpetual futures."""
    total_fees = result.get("total_fees", 0.0) or 0.0
    total_funding = result.get("total_funding", 0.0) or 0.0
    total_slippage = result.get("total_slippage", 0.0) or 0.0
    starting_balance = result.get("starting_balance", 100000) or 100000
    total_return = result.get("total_return", 0.0) or 0.0
    net_pnl = total_return * starting_balance
    gross_pnl = net_pnl + total_fees + total_funding + total_slippage

    st.subheader("Perpetual Futures Analytics")
    col1, col2 = st.columns(2)

    with col1:
        fig = charts.build_funding_pie(gross_pnl, total_fees, total_funding, total_slippage)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Gross P&L", format_dollar(gross_pnl))
        st.metric("Net P&L", format_dollar(net_pnl))
        if gross_pnl != 0:
            funding_pct = (abs(total_funding) / abs(gross_pnl)) * 100
            st.metric("Funding as % of Gross", f"{funding_pct:.1f}%")
        else:
            st.metric("Funding as % of Gross", "N/A")
        funded_trades = [t for t in trades if (t.get("funding_fees") or 0.0) != 0.0]
        st.metric("Trades with Funding", str(len(funded_trades)))


def render_long_short_split(trades: list[dict[str, Any]]) -> None:
    """Render long vs short performance split table."""
    split = compute_long_short_split(trades)

    st.subheader("Long vs Short Performance")
    metrics_list = [
        ("Trade Count", "count", "{:.0f}"),
        ("Win Rate", "win_rate", "{:.1%}"),
        ("Total P&L", "total_pnl", "${:,.2f}"),
        ("Avg P&L per Trade", "avg_pnl", "${:,.2f}"),
        ("Profit Factor", "profit_factor", "{:.2f}"),
        ("Avg Win", "avg_win", "${:,.2f}"),
        ("Avg Loss", "avg_loss", "${:,.2f}"),
    ]

    data: dict[str, list[str]] = {"Metric": [], "LONG": [], "SHORT": []}
    for label, key, fmt in metrics_list:
        data["Metric"].append(label)
        for direction in ("LONG", "SHORT"):
            val = split[direction][key]
            if val == float("inf"):
                data[direction].append("Inf")
            else:
                data[direction].append(fmt.format(val))

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_liquidation_summary(trades: list[dict[str, Any]]) -> None:
    """Render liquidation event summary."""
    liquidations = [t for t in trades if t.get("exit_reason") == "liquidation"]
    if not liquidations:
        return

    st.subheader("Liquidation Events")
    st.warning(f"{len(liquidations)} trades ended in liquidation")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Liquidation Count", str(len(liquidations)))
    with col2:
        total_loss = sum(float(t.get("net_pnl", 0.0) or 0.0) for t in liquidations)
        st.metric("Total Liquidation Loss", format_dollar(total_loss))
    with col3:
        pct = (len(liquidations) / len(trades)) * 100 if trades else 0
        st.metric("% of Trades Liquidated", f"{pct:.1f}%")


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
        filters = [
            result.get("passed_deflated_sharpe"),
            result.get("passed_walk_forward"),
            result.get("passed_purged_kfold"),
        ]
        active_filters = [f for f in filters if f is not None]
        if active_filters:
            if all(active_filters):
                st.success("All filters PASSED")
            else:
                st.error("Some filters FAILED")
        else:
            st.info("No filters applied")


# ---------------------------------------------------------------------------
# Comparison view
# ---------------------------------------------------------------------------


def render_comparison_view(mgr: StateManager, run_ids: list[int]) -> None:
    """Render side-by-side comparison of runs."""
    if len(run_ids) < 2:
        st.info("Select at least 2 runs to compare")
        return

    results = []
    all_trades: dict[int, list[dict[str, Any]]] = {}
    for run_id in run_ids[:3]:
        result = mgr.get_backtest_result(run_id)
        run = mgr.get_backtest_run(run_id)
        if result and run:
            strategy = mgr.get_strategy(run["strategy_id"])
            result["strategy_name"] = strategy["name"] if strategy else "Unknown"
            result["run_mode"] = run["run_mode"]
            result["run_id"] = run_id
            results.append(result)
            all_trades[run_id] = mgr.get_trades(run_id)

    if not results:
        st.warning("No results found for selected runs")
        return

    # Comparison metrics table
    metrics = [
        ("Strategy", "strategy_name", None, False),
        ("Mode", "run_mode", None, False),
        ("Total Return", "total_return", "pct", True),
        ("Sharpe", "sharpe_ratio", "num", True),
        ("Sortino", "sortino_ratio", "num", True),
        ("Max DD", "max_drawdown", "pct", False),
        ("Win Rate", "win_rate", "pct", True),
        ("Profit Factor", "profit_factor", "num", True),
        ("Total Trades", "total_trades", "int", False),
        ("Total Fees", "total_fees", "dollar", False),
        ("DSR Pass", "passed_deflated_sharpe", "bool", False),
        ("WFA Pass", "passed_walk_forward", "bool", False),
        ("KFold Pass", "passed_purged_kfold", "bool", False),
    ]

    comparison_data: dict[str, dict[str, str]] = {}
    for i, result in enumerate(results):
        col_name = f"Run {run_ids[i]}"
        comparison_data[col_name] = {}
        for label, key, fmt, _higher_better in metrics:
            val = result.get(key)
            if fmt == "pct":
                comparison_data[col_name][label] = format_percent(val)
            elif fmt == "num":
                comparison_data[col_name][label] = format_number(val)
            elif fmt == "int":
                comparison_data[col_name][label] = format_number(val, 0) if val else "N/A"
            elif fmt == "dollar":
                comparison_data[col_name][label] = format_dollar(val)
            elif fmt == "bool":
                if isinstance(val, bool):
                    comparison_data[col_name][label] = "PASS" if val else "FAIL"
                else:
                    comparison_data[col_name][label] = "N/A"
            elif val is None:
                comparison_data[col_name][label] = "N/A"
            else:
                comparison_data[col_name][label] = str(val)

    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)

    # Overlaid equity curves
    import plotly.graph_objects as go

    st.subheader("Equity Curve Comparison")
    fig_eq = go.Figure()
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    for i, result in enumerate(results):
        run_id = result["run_id"]
        trades = all_trades.get(run_id, [])
        if trades:
            starting_bal = result.get("starting_balance", 100000) or 100000
            eq_df = build_equity_curve(trades, starting_balance=starting_bal)
            if not eq_df.empty:
                fig_eq.add_trace(
                    go.Scatter(
                        x=eq_df["time"], y=eq_df["equity"], mode="lines",
                        name=f"Run {run_id} ({result.get('strategy_name', '')})",
                        line={"color": colors[i % len(colors)], "width": 2},
                    )
                )
    fig_eq.update_layout(title="Equity Curves Overlaid", height=400)
    st.plotly_chart(fig_eq, use_container_width=True)

    # Overlaid drawdowns
    st.subheader("Drawdown Comparison")
    fig_dd = go.Figure()
    for i, result in enumerate(results):
        run_id = result["run_id"]
        trades = all_trades.get(run_id, [])
        if trades:
            starting_bal = result.get("starting_balance", 100000) or 100000
            eq_df = build_equity_curve(trades, starting_balance=starting_bal)
            dd_df = compute_drawdown_series(eq_df)
            if not dd_df.empty:
                fig_dd.add_trace(
                    go.Scatter(
                        x=dd_df["time"], y=dd_df["drawdown"] * -100, mode="lines",
                        name=f"Run {run_id}",
                        line={"color": colors[i % len(colors)], "width": 1.5},
                        fill="tozeroy" if i == 0 else None,
                        fillcolor=f"rgba({','.join(str(int(colors[0][j:j+2], 16)) for j in (1,3,5))}, 0.1)" if i == 0 else None,
                    )
                )
    fig_dd.update_layout(title="Drawdowns Overlaid", yaxis_title="Drawdown (%)", height=300)
    st.plotly_chart(fig_dd, use_container_width=True)

    # Radar chart
    st.subheader("Strategy Profile Radar")
    _show_figure(charts.build_radar_chart(results))


# ---------------------------------------------------------------------------
# Page-level views
# ---------------------------------------------------------------------------


def _render_single_run_view(mgr: StateManager, runs: list[dict[str, Any]]) -> None:
    """Render single run analysis view."""
    run_options = {r["label"]: r["id"] for r in runs}
    selected_label = st.selectbox("Select Run", options=list(run_options.keys()), key="single_run_select")
    selected_run_id = run_options[selected_label]

    run = mgr.get_backtest_run(selected_run_id)
    if not run:
        st.error("Run not found")
        return

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

    sweep_results = mgr.get_sweep_results(selected_run_id)
    backtest_result = mgr.get_backtest_result(selected_run_id)

    if sweep_results:
        _render_sweep_view(mgr, selected_run_id, sweep_results)
    elif backtest_result:
        _render_backtest_result_view(mgr, selected_run_id, backtest_result)
    else:
        st.warning(f"No results found for run {selected_run_id}. Status: {run['status']}")


def _render_sweep_view(
    mgr: StateManager, run_id: int, sweep_results: list[dict[str, Any]],
) -> None:
    """Render sweep results view with Pareto charts."""
    st.subheader("Sweep Results")
    sweep_df = build_sweep_dataframe(sweep_results)

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        show_pareto_only = st.checkbox("Pareto optimal only", value=False)
    with filter_col2:
        min_sharpe = st.number_input("Min Sharpe", value=0.0, step=0.1)
    with filter_col3:
        max_dd = st.number_input("Max Drawdown %", value=100.0, step=1.0)

    filtered_df = sweep_df.copy()
    if show_pareto_only:
        filtered_df = filtered_df[filtered_df["Pareto"] == 1]
    if min_sharpe > 0:
        filtered_df = filtered_df[filtered_df["Sharpe"] >= min_sharpe]
    if max_dd < 100:
        filtered_df = filtered_df[(filtered_df["Max DD"].abs() * 100) <= max_dd]

    _show_figure(charts.build_pareto_scatter(filtered_df), "No sweep results to display")

    with st.expander("3D Pareto Surface", expanded=False):
        _show_figure(charts.build_3d_pareto_scatter(filtered_df))

    st.dataframe(
        filtered_df.style.format(
            {"Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Max DD": "{:.2%}",
             "Return": "{:.2%}", "PF": "{:.2f}", "Win Rate": "{:.2%}"},
            na_rep="N/A",
        ),
        use_container_width=True, height=400,
    )

    csv_data = export_to_csv(filtered_df, f"sweep_results_{run_id}.csv")
    st.download_button(
        label="Export to CSV", data=csv_data,
        file_name=f"sweep_results_{run_id}.csv", mime="text/csv",
    )

    st.subheader("Individual Result Detail")
    sweep_ids = [r["id"] for r in sweep_results]
    selected_sweep_id = st.selectbox("Select sweep result", options=sweep_ids, key="sweep_detail_select")

    sweep_result = next((r for r in sweep_results if r["id"] == selected_sweep_id), None)
    if sweep_result:
        with st.expander("Parameters", expanded=True):
            st.json(sweep_result.get("parameters", {}))
        col1, col2, col3 = st.columns(3)
        with col1:
            render_overfitting_badge(sweep_result.get("passed_deflated_sharpe"), "Deflated Sharpe")
        with col2:
            render_overfitting_badge(sweep_result.get("passed_walk_forward"), "Walk-Forward")
        with col3:
            render_overfitting_badge(sweep_result.get("passed_purged_kfold"), "Purged K-Fold")

        # Run Validation button
        st.divider()
        if st.button(
            "Run Validation for this candidate",
            key=f"run_val_{selected_sweep_id}",
            type="primary",
            help="Launch a full-fidelity validation backtest with these parameters",
        ):
            st.session_state["launch_validation_params"] = sweep_result.get("parameters", {})
            st.session_state["launch_validation_from_run"] = run_id
            st.info(
                "Parameters saved. Go to **Backtest Launch** tab and select "
                "'Validation' mode to run with these parameters."
            )


def _render_tearsheet_button(result: dict[str, Any], run_id: int) -> None:
    """Render tearsheet generation/download button."""
    import json as _json

    tearsheet_data = {
        "run_id": run_id,
        "total_return": result.get("total_return"),
        "sharpe_ratio": result.get("sharpe_ratio"),
        "sortino_ratio": result.get("sortino_ratio"),
        "max_drawdown": result.get("max_drawdown"),
        "win_rate": result.get("win_rate"),
        "profit_factor": result.get("profit_factor"),
        "total_trades": result.get("total_trades"),
        "calmar_ratio": result.get("calmar_ratio"),
        "cagr": result.get("cagr"),
        "volatility_annual": result.get("volatility_annual"),
        "avg_win": result.get("avg_win"),
        "avg_loss": result.get("avg_loss"),
        "largest_win": result.get("largest_win"),
        "largest_loss": result.get("largest_loss"),
        "max_consecutive_wins": result.get("max_consecutive_wins"),
        "max_consecutive_losses": result.get("max_consecutive_losses"),
        "total_fees": result.get("total_fees"),
        "total_funding": result.get("total_funding"),
        "total_slippage": result.get("total_slippage"),
        "passed_deflated_sharpe": result.get("passed_deflated_sharpe"),
        "passed_walk_forward": result.get("passed_walk_forward"),
        "passed_purged_kfold": result.get("passed_purged_kfold"),
    }

    json_bytes = _json.dumps(tearsheet_data, indent=2).encode("utf-8")
    st.download_button(
        label="Download Tearsheet (JSON)",
        data=json_bytes,
        file_name=f"tearsheet_run_{run_id}.json",
        mime="application/json",
    )


def _render_backtest_result_view(
    mgr: StateManager, run_id: int, backtest_result: dict[str, Any],
) -> None:
    """Render single backtest result view with all analysis panels."""
    render_metrics_panel(backtest_result)

    # Raw NT stats (persisted)
    nt_stats = backtest_result.get("raw_nt_stats")
    if nt_stats:
        with st.expander("Raw NautilusTrader Statistics", expanded=False):
            st.json(nt_stats)

    # Tearsheet generation
    with st.expander("Export Tearsheet", expanded=False):
        _render_tearsheet_button(backtest_result, run_id)

    render_cost_breakdown(backtest_result)
    render_overfitting_results(backtest_result)

    trades = mgr.get_trades(run_id)

    if trades:
        starting_balance = backtest_result.get("starting_balance", 100000) or 100000

        equity_df = build_equity_curve(trades, starting_balance=starting_balance)
        _show_figure(charts.build_equity_chart(equity_df), "No equity data to display")

        dd_df = compute_drawdown_series(equity_df)
        top_dd = compute_top_drawdown_periods(dd_df)
        _show_figure(charts.build_drawdown_chart(dd_df, top_periods=top_dd), "No drawdown data to display")

        rolling_df = compute_rolling_sharpe(trades)
        _show_figure(charts.build_rolling_sharpe_chart(rolling_df))

        yearly_df = compute_yearly_returns(trades)
        _show_figure(charts.build_yearly_returns_chart(yearly_df))

        # Monthly heatmap
        closed_trades = [t for t in trades if t.get("exit_time") and t.get("net_pnl") is not None]
        if closed_trades:
            st.subheader("Monthly Returns")
            _show_figure(charts.build_monthly_heatmap(closed_trades))

        render_long_short_split(trades)

        with st.expander("Perpetual Futures Analytics", expanded=False):
            render_funding_impact(backtest_result, trades)

        render_liquidation_summary(trades)

        with st.expander("Trade Analysis Charts", expanded=False):
            if closed_trades:
                st.subheader("Trade P&L Distribution")
                _show_figure(charts.build_pnl_distribution(closed_trades))
            _show_figure(charts.build_trade_scatter_roi_vs_duration(trades))
            _show_figure(charts.build_trade_scatter_size_vs_pnl(trades))

        with st.expander("Daily Returns", expanded=False):
            daily_df = compute_daily_returns(trades)
            _show_figure(charts.build_daily_returns_chart(daily_df))

        # Trade log
        st.subheader("Trade Log")
        trades_df = build_trades_dataframe(trades)

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

        filtered_trades = trades_df.copy()
        if direction_filter and "direction" in filtered_trades.columns:
            filtered_trades = filtered_trades[filtered_trades["direction"].isin(direction_filter)]
        if symbol_filter and "symbol" in filtered_trades.columns:
            filtered_trades = filtered_trades[filtered_trades["symbol"].isin(symbol_filter)]

        def _highlight_liquidations(row: pd.Series) -> list[str]:
            if row.get("exit_reason") == "liquidation":
                return ["background-color: #ffcccc"] * len(row)
            return [""] * len(row)

        if "exit_reason" in filtered_trades.columns:
            styled = filtered_trades.style.apply(_highlight_liquidations, axis=1)
            st.dataframe(styled, use_container_width=True, height=400)
        else:
            st.dataframe(filtered_trades, use_container_width=True, height=400)

        csv_data = export_to_csv(filtered_trades, f"trades_{run_id}.csv")
        st.download_button(
            label="Export Trades to CSV", data=csv_data,
            file_name=f"trades_{run_id}.csv", mime="text/csv",
        )
    else:
        st.info("No trades recorded for this run")

    # Notes
    st.subheader("Notes")
    current_notes = backtest_result.get("notes", "") or ""
    new_notes = st.text_area(
        "Add observations or annotations about this backtest run",
        value=current_notes, height=100, key=f"notes_{run_id}",
    )
    if new_notes != current_notes and st.button("Save Notes", key=f"save_notes_{run_id}"):
        mgr.update_result_notes(run_id, new_notes)
        st.success("Notes saved")
        st.rerun()


def _render_compare_view(mgr: StateManager, runs: list[dict[str, Any]]) -> None:
    """Render comparison view."""
    st.subheader("Compare Multiple Runs")
    run_options_multi = {r["label"]: r["id"] for r in runs}
    selected_labels = st.multiselect(
        "Select runs to compare (max 3)", options=list(run_options_multi.keys()), max_selections=3
    )
    if selected_labels:
        selected_ids = [run_options_multi[label] for label in selected_labels]
        render_comparison_view(mgr, selected_ids)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def render_results_tab() -> None:
    """Render the Results Analysis tab."""
    st.header("Results Analysis")
    mgr = get_state_manager()
    runs = get_runs_for_dropdown(mgr)
    if not runs:
        st.info("No backtest runs found. Run a backtest first.")
        return
    view_tab, compare_tab = st.tabs(["Single Run Analysis", "Compare Runs"])
    with view_tab:
        _render_single_run_view(mgr, runs)
    with compare_tab:
        _render_compare_view(mgr, runs)


# Top-level call for st.navigation API
render_results_tab()
