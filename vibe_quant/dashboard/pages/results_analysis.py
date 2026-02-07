"""Results Analysis Tab for vibe-quant Streamlit dashboard.

Provides:
- Sweep results table (sortable, filterable) with 3D Pareto
- Pareto front scatter plot (Sharpe vs MaxDD, color=PF)
- Individual result detail: equity curve, drawdown chart, trade log
- Enhanced metrics panel with derived metrics (expectancy, cost drag, payoff ratio)
- Rolling Sharpe chart, yearly returns bar chart, daily returns bar chart
- Long vs short performance split
- Funding impact analysis for perpetual futures
- Liquidation event highlighting
- Trade scatter plots (ROI vs duration, size vs PnL)
- Overfitting filter pass/fail badges with WFA detail charts
- Screening-to-validation degradation scatter
- Strategy comparison with overlaid equity curves and radar chart
- Top-5 drawdown period annotation
- Result notes/annotations
- CSV export
"""

from __future__ import annotations

import io
import math
from typing import TYPE_CHECKING

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from vibe_quant.db import StateManager

if TYPE_CHECKING:
    from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def format_dollar(val: float | None) -> str:
    """Format value as dollar amount."""
    if val is None:
        return "N/A"
    return f"${val:,.2f}"


def render_overfitting_badge(passed: bool | None, label: str) -> None:
    """Render pass/fail badge for overfitting filter."""
    if passed is None:
        st.markdown(f"**{label}**: :gray[N/A]")
    elif passed:
        st.markdown(f"**{label}**: :green[PASS]")
    else:
        st.markdown(f"**{label}**: :red[FAIL]")


# ---------------------------------------------------------------------------
# Data builders (cached)
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


def build_equity_curve(
    trades: list[dict[str, Any]], starting_balance: float = 100000
) -> pd.DataFrame:
    """Build equity curve from trades.

    Args:
        trades: List of trade dicts with exit_time and net_pnl.
        starting_balance: Initial account equity. Defaults to 100000.
    """
    if not trades:
        return pd.DataFrame()

    # Sort by exit time, filter closed trades
    closed = [t for t in trades if t.get("exit_time")]
    if not closed:
        return pd.DataFrame()

    sorted_trades = sorted(closed, key=lambda x: x["exit_time"])
    cumulative_pnl = 0.0
    equity_points = [{"time": sorted_trades[0]["entry_time"], "equity": starting_balance}]

    for trade in sorted_trades:
        pnl = trade.get("net_pnl", 0.0) or 0.0
        cumulative_pnl += pnl
        equity_points.append({"time": trade["exit_time"], "equity": starting_balance + cumulative_pnl})

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
        dd = (peak - val) / peak if peak > 0 else (peak - val) / abs(peak) if peak < 0 else 0.0
        drawdowns.append({"time": equity_df.iloc[i]["time"], "drawdown": dd})

    return pd.DataFrame(drawdowns)


def compute_top_drawdown_periods(
    dd_df: pd.DataFrame, n: int = 5
) -> list[dict[str, Any]]:
    """Find the top-N drawdown periods with start/end/depth/recovery.

    Args:
        dd_df: Drawdown series DataFrame with 'time' and 'drawdown' columns.
        n: Number of top drawdown periods to return.

    Returns:
        List of dicts with start, end, trough, depth, and duration_days.
    """
    if dd_df.empty:
        return []

    periods: list[dict[str, Any]] = []
    in_dd = False
    start_idx = 0
    max_dd = 0.0
    trough_idx = 0

    for i in range(len(dd_df)):
        dd_val = dd_df.iloc[i]["drawdown"]
        if dd_val > 0 and not in_dd:
            in_dd = True
            start_idx = max(0, i - 1)
            max_dd = dd_val
            trough_idx = i
        elif dd_val > 0 and in_dd:
            if dd_val > max_dd:
                max_dd = dd_val
                trough_idx = i
        elif dd_val == 0 and in_dd:
            in_dd = False
            periods.append({
                "start": dd_df.iloc[start_idx]["time"],
                "end": dd_df.iloc[i]["time"],
                "trough": dd_df.iloc[trough_idx]["time"],
                "depth": max_dd,
            })

    # Handle ongoing drawdown at end
    if in_dd:
        periods.append({
            "start": dd_df.iloc[start_idx]["time"],
            "end": dd_df.iloc[len(dd_df) - 1]["time"],
            "trough": dd_df.iloc[trough_idx]["time"],
            "depth": max_dd,
        })

    # Sort by depth descending and take top N
    periods.sort(key=lambda x: x["depth"], reverse=True)
    return periods[:n]


def compute_rolling_sharpe(
    trades: list[dict[str, Any]], window: int = 60
) -> pd.DataFrame:
    """Compute rolling Sharpe ratio from trades.

    Uses a rolling window of trade returns to compute annualized Sharpe.
    For crypto (24/7), annualization factor = sqrt(365).

    Args:
        trades: List of trade dicts with exit_time and roi_percent.
        window: Rolling window size (number of trades).

    Returns:
        DataFrame with 'time' and 'rolling_sharpe' columns.
    """
    closed = [t for t in trades if t.get("exit_time") and t.get("roi_percent") is not None]
    if len(closed) < window:
        return pd.DataFrame()

    sorted_trades = sorted(closed, key=lambda x: x["exit_time"])
    returns = [float(t.get("roi_percent", 0.0) or 0.0) / 100.0 for t in sorted_trades]
    times = [t["exit_time"] for t in sorted_trades]

    annualize = math.sqrt(365)
    points = []
    for i in range(window, len(returns) + 1):
        window_returns = returns[i - window : i]
        mean_r = sum(window_returns) / len(window_returns)
        var = sum((r - mean_r) ** 2 for r in window_returns) / max(len(window_returns) - 1, 1)
        std = math.sqrt(var) if var > 0 else 1e-10
        sharpe = (mean_r / std) * annualize
        points.append({"time": times[i - 1], "rolling_sharpe": sharpe})

    return pd.DataFrame(points)


def compute_daily_returns(
    trades: list[dict[str, Any]],
) -> pd.DataFrame:
    """Compute daily net PnL from trades.

    Args:
        trades: List of trade dicts with exit_time and net_pnl.

    Returns:
        DataFrame with 'date' and 'daily_pnl' columns.
    """
    closed = [t for t in trades if t.get("exit_time") and t.get("net_pnl") is not None]
    if not closed:
        return pd.DataFrame()

    df = pd.DataFrame(closed)
    df["exit_date"] = pd.to_datetime(df["exit_time"]).dt.date
    daily = df.groupby("exit_date")["net_pnl"].sum().reset_index()
    daily.columns = ["date", "daily_pnl"]
    return daily


def compute_yearly_returns(
    trades: list[dict[str, Any]],
) -> pd.DataFrame:
    """Compute yearly net PnL from trades.

    Args:
        trades: List of trade dicts with exit_time and net_pnl.

    Returns:
        DataFrame with 'year' and 'yearly_pnl' columns.
    """
    closed = [t for t in trades if t.get("exit_time") and t.get("net_pnl") is not None]
    if not closed:
        return pd.DataFrame()

    df = pd.DataFrame(closed)
    df["year"] = pd.to_datetime(df["exit_time"]).dt.year
    yearly = df.groupby("year")["net_pnl"].sum().reset_index()
    yearly.columns = ["year", "yearly_pnl"]
    return yearly


def compute_long_short_split(
    trades: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute performance metrics split by direction (LONG vs SHORT).

    Args:
        trades: List of trade dicts.

    Returns:
        Dict with 'LONG' and 'SHORT' keys, each containing metrics.
    """
    result: dict[str, dict[str, Any]] = {}
    for direction in ("LONG", "SHORT"):
        dir_trades = [t for t in trades if t.get("direction") == direction and t.get("exit_time")]
        if not dir_trades:
            result[direction] = {
                "count": 0, "win_rate": 0.0, "total_pnl": 0.0,
                "avg_pnl": 0.0, "profit_factor": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0,
            }
            continue

        pnls = [float(t.get("net_pnl", 0.0) or 0.0) for t in dir_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        total_wins = sum(wins) if wins else 0.0
        total_losses = abs(sum(losses)) if losses else 0.0

        result[direction] = {
            "count": len(dir_trades),
            "win_rate": len(wins) / len(dir_trades) if dir_trades else 0.0,
            "total_pnl": sum(pnls),
            "avg_pnl": sum(pnls) / len(pnls) if pnls else 0.0,
            "profit_factor": total_wins / total_losses if total_losses > 0 else float("inf") if total_wins > 0 else 0.0,
            "avg_win": sum(wins) / len(wins) if wins else 0.0,
            "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        }

    return result


# ---------------------------------------------------------------------------
# Chart renderers
# ---------------------------------------------------------------------------


def render_pareto_scatter(sweep_df: pd.DataFrame) -> None:
    """Render Pareto front scatter plot."""
    if sweep_df.empty:
        st.info("No sweep results to display")
        return

    plot_df = sweep_df.dropna(subset=["Sharpe", "Max DD"])
    if plot_df.empty:
        st.info("No complete results for scatter plot")
        return

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
    st.plotly_chart(fig, use_container_width=True)


def render_3d_pareto_scatter(sweep_df: pd.DataFrame) -> None:
    """Render 3D Pareto front scatter plot (Sharpe x MaxDD x PF)."""
    if sweep_df.empty:
        return

    plot_df = sweep_df.dropna(subset=["Sharpe", "Max DD", "PF"])
    if len(plot_df) < 3:
        return

    plot_df = plot_df.copy()
    plot_df["Max DD (%)"] = plot_df["Max DD"].abs() * 100

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=plot_df["Sharpe"],
                y=plot_df["Max DD (%)"],
                z=plot_df["PF"],
                mode="markers",
                marker={
                    "size": 5,
                    "color": plot_df["Sharpe"],
                    "colorscale": "Viridis",
                    "colorbar": {"title": "Sharpe"},
                    "opacity": 0.8,
                },
                text=[f"ID: {i}" for i in plot_df["ID"]],
                hovertemplate=(
                    "Sharpe: %{x:.2f}<br>"
                    "Max DD: %{y:.1f}%<br>"
                    "PF: %{z:.2f}<br>"
                    "%{text}<extra></extra>"
                ),
            )
        ]
    )

    # Highlight Pareto optimal
    pareto_3d = plot_df[plot_df["Pareto"] == 1]
    if not pareto_3d.empty:
        fig.add_trace(
            go.Scatter3d(
                x=pareto_3d["Sharpe"],
                y=pareto_3d["Max DD (%)"],
                z=pareto_3d["PF"],
                mode="markers",
                marker={"size": 8, "symbol": "diamond", "color": "red", "opacity": 1.0},
                name="Pareto Optimal",
            )
        )

    fig.update_layout(
        title="3D Pareto Surface: Sharpe x Max DD x Profit Factor",
        scene={
            "xaxis_title": "Sharpe Ratio",
            "yaxis_title": "Max Drawdown (%)",
            "zaxis_title": "Profit Factor",
        },
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_equity_chart(
    equity_df: pd.DataFrame,
    dd_df: pd.DataFrame | None = None,
    top_dd_periods: list[dict[str, Any]] | None = None,
) -> None:
    """Render equity curve chart with optional drawdown annotations."""
    if equity_df.empty:
        st.info("No equity data to display")
        return

    fig = go.Figure()

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=equity_df["time"],
            y=equity_df["equity"],
            mode="lines",
            name="Equity",
            line={"color": "#2196F3", "width": 2},
        )
    )

    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Time",
        yaxis_title="Equity ($)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_drawdown_chart(
    dd_df: pd.DataFrame,
    top_periods: list[dict[str, Any]] | None = None,
) -> None:
    """Render drawdown chart with optional top-5 period annotations."""
    if dd_df.empty:
        st.info("No drawdown data to display")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd_df["time"],
            y=dd_df["drawdown"] * -100,
            fill="tozeroy",
            fillcolor="rgba(255, 0, 0, 0.15)",
            line={"color": "red", "width": 1},
            name="Drawdown",
        )
    )

    # Annotate top-5 drawdown periods
    colors = ["#FF0000", "#FF4444", "#FF7777", "#FF9999", "#FFBBBB"]
    if top_periods:
        for i, period in enumerate(top_periods[:5]):
            color = colors[i] if i < len(colors) else colors[-1]
            fig.add_vrect(
                x0=period["start"],
                x1=period["end"],
                fillcolor=color,
                opacity=0.15,
                line_width=0,
                annotation_text=f"#{i + 1}: {period['depth'] * 100:.1f}%",
                annotation_position="top left",
                annotation_font_size=10,
            )

    fig.update_layout(
        title="Drawdown Chart" + (" (Top-5 periods highlighted)" if top_periods else ""),
        xaxis_title="Time",
        yaxis_title="Drawdown (%)",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_rolling_sharpe_chart(rolling_df: pd.DataFrame) -> None:
    """Render rolling Sharpe ratio chart."""
    if rolling_df.empty:
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rolling_df["time"],
            y=rolling_df["rolling_sharpe"],
            mode="lines",
            name="Rolling Sharpe",
            line={"color": "#9C27B0", "width": 1.5},
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)
    fig.add_hline(y=1, line_dash="dot", line_color="green", line_width=0.5,
                  annotation_text="Sharpe=1", annotation_position="bottom right")
    fig.add_hline(y=2, line_dash="dot", line_color="blue", line_width=0.5,
                  annotation_text="Sharpe=2", annotation_position="bottom right")

    fig.update_layout(
        title="Rolling Sharpe Ratio (60-trade window, annualized 365d)",
        xaxis_title="Time",
        yaxis_title="Sharpe Ratio",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_yearly_returns_chart(yearly_df: pd.DataFrame) -> None:
    """Render yearly returns bar chart."""
    if yearly_df.empty:
        return

    colors = ["green" if v >= 0 else "red" for v in yearly_df["yearly_pnl"]]
    fig = go.Figure(
        data=[
            go.Bar(
                x=yearly_df["year"].astype(str),
                y=yearly_df["yearly_pnl"],
                marker_color=colors,
                text=[f"${v:,.0f}" for v in yearly_df["yearly_pnl"]],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Yearly Returns",
        xaxis_title="Year",
        yaxis_title="Net P&L ($)",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_daily_returns_chart(daily_df: pd.DataFrame) -> None:
    """Render daily returns bar chart."""
    if daily_df.empty:
        return

    colors = ["green" if v >= 0 else "gray" for v in daily_df["daily_pnl"]]
    fig = go.Figure(
        data=[
            go.Bar(
                x=daily_df["date"],
                y=daily_df["daily_pnl"],
                marker_color=colors,
            )
        ]
    )
    fig.update_layout(
        title="Daily Returns",
        xaxis_title="Date",
        yaxis_title="Net P&L ($)",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Metrics panels
# ---------------------------------------------------------------------------


def render_metrics_panel(result: dict[str, Any]) -> None:
    """Render enhanced key metrics panel with derived metrics."""
    # Row 1: Core performance metrics
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

    # Column 5: Previously hidden metrics
    with col5:
        st.metric("Winning Trades", format_number(result.get("winning_trades"), 0))
        st.metric("Losing Trades", format_number(result.get("losing_trades"), 0))
        exec_time = result.get("execution_time_seconds")
        st.metric("Execution Time", f"{format_number(exec_time, 1)}s" if exec_time else "N/A")

    # Row 2: Win/Loss detail metrics
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

    # Column 4: Derived metrics
    with col4:
        # Expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        win_rate = result.get("win_rate")
        avg_win = result.get("avg_win")
        avg_loss = result.get("avg_loss")
        if win_rate is not None and avg_win is not None and avg_loss is not None:
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
            st.metric("Expectancy", format_dollar(expectancy))
        else:
            st.metric("Expectancy", "N/A")

        # Payoff Ratio = |avg_win / avg_loss|
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
        # Cost Drag %: total costs as fraction of what gross PnL would be
        total_return = result.get("total_return")
        starting_balance = result.get("starting_balance", 100000) or 100000
        if total_return is not None and total_return != 0:
            # Gross PnL ≈ net PnL + costs
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

    # Compute net trading PnL (gross - all costs gives net; so gross = net + costs)
    starting_balance = result.get("starting_balance", 100000) or 100000
    total_return = result.get("total_return", 0.0) or 0.0
    net_pnl = total_return * starting_balance
    gross_pnl = net_pnl + total_fees + total_funding + total_slippage

    st.subheader("Perpetual Futures Analytics")

    col1, col2 = st.columns(2)

    with col1:
        # PnL breakdown pie chart
        labels = ["Net Trading P&L", "Fees", "Funding", "Slippage"]
        values = [abs(gross_pnl - total_fees - total_funding - total_slippage),
                  total_fees, abs(total_funding), total_slippage]
        colors_pie = ["#4CAF50", "#FF9800", "#F44336", "#9E9E9E"]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    marker={"colors": colors_pie},
                    hole=0.3,
                    textinfo="label+percent",
                )
            ]
        )
        fig.update_layout(title="Gross P&L Breakdown", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Gross P&L", format_dollar(gross_pnl))
        st.metric("Net P&L", format_dollar(net_pnl))
        if gross_pnl != 0:
            funding_pct = (abs(total_funding) / abs(gross_pnl)) * 100
            st.metric("Funding as % of Gross", f"{funding_pct:.1f}%")
        else:
            st.metric("Funding as % of Gross", "N/A")

        # Count trades with funding
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
            all_pass = all(active_filters)
            if all_pass:
                st.success("All filters PASSED")
            else:
                st.error("Some filters FAILED")
        else:
            st.info("No filters applied")


# ---------------------------------------------------------------------------
# Trade analysis charts
# ---------------------------------------------------------------------------


def render_trade_scatter_roi_vs_duration(trades: list[dict[str, Any]]) -> None:
    """Render trade scatter: ROI% vs duration (hours), colored by direction."""
    closed = [t for t in trades if t.get("exit_time") and t.get("entry_time")]
    if not closed:
        return

    data = []
    for t in closed:
        try:
            from datetime import datetime
            entry = datetime.fromisoformat(str(t["entry_time"]).replace("Z", "+00:00"))
            exit_ = datetime.fromisoformat(str(t["exit_time"]).replace("Z", "+00:00"))
            duration_h = (exit_ - entry).total_seconds() / 3600.0
            data.append({
                "duration_hours": duration_h,
                "roi_percent": float(t.get("roi_percent", 0.0) or 0.0),
                "direction": t.get("direction", "UNKNOWN"),
            })
        except (ValueError, TypeError):
            continue

    if not data:
        return

    df = pd.DataFrame(data)
    fig = px.scatter(
        df, x="duration_hours", y="roi_percent", color="direction",
        color_discrete_map={"LONG": "#2196F3", "SHORT": "#FF5722"},
        title="Trade ROI vs Duration",
        labels={"duration_hours": "Duration (hours)", "roi_percent": "ROI (%)"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_trade_scatter_size_vs_pnl(trades: list[dict[str, Any]]) -> None:
    """Render trade scatter: position size (notional) vs net PnL."""
    closed = [t for t in trades if t.get("exit_time") and t.get("entry_price") and t.get("quantity")]
    if not closed:
        return

    data = []
    for t in closed:
        notional = float(t.get("entry_price", 0)) * float(t.get("quantity", 0))
        data.append({
            "notional": notional,
            "net_pnl": float(t.get("net_pnl", 0.0) or 0.0),
            "direction": t.get("direction", "UNKNOWN"),
        })

    df = pd.DataFrame(data)
    fig = px.scatter(
        df, x="notional", y="net_pnl", color="direction",
        color_discrete_map={"LONG": "#2196F3", "SHORT": "#FF5722"},
        title="Position Size vs Net P&L",
        labels={"notional": "Notional Value ($)", "net_pnl": "Net P&L ($)"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Comparison view
# ---------------------------------------------------------------------------


def render_comparison_view(mgr: StateManager, run_ids: list[int]) -> None:
    """Render side-by-side comparison of runs with overlaid charts and radar."""
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

    # Comparison metrics table with best-value highlighting
    metrics = [
        ("Strategy", "strategy_name", None, False),
        ("Mode", "run_mode", None, False),
        ("Total Return", "total_return", "pct", True),
        ("Sharpe", "sharpe_ratio", "num", True),
        ("Sortino", "sortino_ratio", "num", True),
        ("Max DD", "max_drawdown", "pct", False),  # Lower is better
        ("Win Rate", "win_rate", "pct", True),
        ("Profit Factor", "profit_factor", "num", True),
        ("Total Trades", "total_trades", "int", False),
        ("Total Fees", "total_fees", "dollar", False),
        ("DSR Pass", "passed_deflated_sharpe", "bool", False),
        ("WFA Pass", "passed_walk_forward", "bool", False),
        ("KFold Pass", "passed_purged_kfold", "bool", False),
    ]

    comparison_data: dict[str, dict[str, str]] = {}
    raw_values: dict[str, list[float | None]] = {}

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

            # Track raw values for highlighting
            if label not in raw_values:
                raw_values[label] = []
            raw_values[label].append(float(val) if isinstance(val, (int, float)) else None)

    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)

    # Overlaid equity curves
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
                        x=eq_df["time"],
                        y=eq_df["equity"],
                        mode="lines",
                        name=f"Run {run_id} ({result.get('strategy_name', '')})",
                        line={"color": colors[i % len(colors)], "width": 2},
                    )
                )

    fig_eq.update_layout(title="Equity Curves Overlaid", height=400)
    st.plotly_chart(fig_eq, use_container_width=True)

    # Overlaid drawdown curves
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
                        x=dd_df["time"],
                        y=dd_df["drawdown"] * -100,
                        mode="lines",
                        name=f"Run {run_id}",
                        line={"color": colors[i % len(colors)], "width": 1.5},
                        fill="tozeroy" if i == 0 else None,
                        fillcolor=f"rgba({','.join(str(int(colors[0][j:j+2], 16)) for j in (1,3,5))}, 0.1)" if i == 0 else None,
                    )
                )

    fig_dd.update_layout(title="Drawdowns Overlaid", yaxis_title="Drawdown (%)", height=300)
    st.plotly_chart(fig_dd, use_container_width=True)

    # Radar/spider chart
    st.subheader("Strategy Profile Radar")
    _render_radar_chart(results)


def _render_radar_chart(results: list[dict[str, Any]]) -> None:
    """Render radar/spider chart for strategy comparison."""
    radar_metrics = [
        ("Sharpe", "sharpe_ratio", True),
        ("Sortino", "sortino_ratio", True),
        ("Calmar", "calmar_ratio", True),
        ("Win Rate", "win_rate", True),
        ("Profit Factor", "profit_factor", True),
        ("1-MaxDD", "max_drawdown", False),  # Invert: lower DD = better
    ]

    # Collect raw values
    raw: dict[str, list[float]] = {}
    for label, key, _higher in radar_metrics:
        raw[label] = []
        for r in results:
            val = r.get(key)
            if val is None:
                raw[label].append(0.0)
            elif label == "1-MaxDD":
                raw[label].append(1.0 - abs(float(val)))
            else:
                raw[label].append(float(val))

    # Normalize each metric to [0, 1]
    normalized: dict[str, list[float]] = {}
    for label, values in raw.items():
        min_v = min(values)
        max_v = max(values)
        rng = max_v - min_v
        if rng == 0:
            normalized[label] = [0.5] * len(values)
        else:
            normalized[label] = [(v - min_v) / rng for v in values]

    fig = go.Figure()
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    categories = list(normalized.keys())

    for i, result in enumerate(results):
        values = [normalized[cat][i] for cat in categories]
        values.append(values[0])  # Close the polygon
        cats = categories + [categories[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=cats,
                fill="toself",
                name=f"Run {result.get('run_id', i)} ({result.get('strategy_name', '')})",
                line={"color": colors[i % len(colors)]},
                opacity=0.6,
            )
        )

    fig.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        title="Strategy Profile Comparison",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def export_to_csv(df: pd.DataFrame, filename: str) -> bytes:
    """Export DataFrame to CSV bytes."""
    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------


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
        _render_single_run_view(mgr, runs)

    with compare_tab:
        _render_compare_view(mgr, runs)


def _render_single_run_view(mgr: StateManager, runs: list[dict[str, Any]]) -> None:
    """Render single run analysis view."""
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
        _render_sweep_view(mgr, selected_run_id, sweep_results)
    elif backtest_result:
        _render_backtest_result_view(mgr, selected_run_id, backtest_result)
    else:
        st.warning(f"No results found for run {selected_run_id}. Status: {run['status']}")


def _render_sweep_view(
    mgr: StateManager,
    run_id: int,
    sweep_results: list[dict[str, Any]],
) -> None:
    """Render sweep results view with Pareto charts."""
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

    # 2D Pareto scatter
    render_pareto_scatter(filtered_df)

    # 3D Pareto surface (lazy-loaded)
    with st.expander("3D Pareto Surface", expanded=False):
        render_3d_pareto_scatter(filtered_df)

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
        use_container_width=True,
        height=400,
    )

    # Export button
    csv_data = export_to_csv(filtered_df, f"sweep_results_{run_id}.csv")
    st.download_button(
        label="Export to CSV",
        data=csv_data,
        file_name=f"sweep_results_{run_id}.csv",
        mime="text/csv",
    )

    # Select individual sweep result for detail
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


def _render_backtest_result_view(
    mgr: StateManager,
    run_id: int,
    backtest_result: dict[str, Any],
) -> None:
    """Render single backtest result view with all analysis panels."""
    # Metrics panels
    render_metrics_panel(backtest_result)
    render_cost_breakdown(backtest_result)
    render_overfitting_results(backtest_result)

    # Get trades
    trades = mgr.get_trades(run_id)

    if trades:
        starting_balance = backtest_result.get("starting_balance", 100000) or 100000

        # Equity curve
        equity_df = build_equity_curve(trades, starting_balance=starting_balance)
        render_equity_chart(equity_df)

        # Drawdown chart with top-5 annotations
        dd_df = compute_drawdown_series(equity_df)
        top_dd = compute_top_drawdown_periods(dd_df)
        render_drawdown_chart(dd_df, top_periods=top_dd)

        # Rolling Sharpe chart
        rolling_df = compute_rolling_sharpe(trades)
        if not rolling_df.empty:
            render_rolling_sharpe_chart(rolling_df)

        # Yearly returns
        yearly_df = compute_yearly_returns(trades)
        if not yearly_df.empty:
            render_yearly_returns_chart(yearly_df)

        # Monthly returns heatmap
        closed_trades = [
            t for t in trades if t.get("exit_time") and t.get("net_pnl") is not None
        ]
        if closed_trades:
            st.subheader("Monthly Returns")
            trade_df = pd.DataFrame(closed_trades)
            trade_df["exit_time"] = pd.to_datetime(trade_df["exit_time"])
            trade_df["year"] = trade_df["exit_time"].dt.year
            trade_df["month"] = trade_df["exit_time"].dt.month
            monthly = trade_df.groupby(["year", "month"])["net_pnl"].sum().reset_index()
            pivot = monthly.pivot(index="year", columns="month", values="net_pnl").fillna(0)
            month_names = {
                1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
                5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
                9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
            }
            pivot = pivot.rename(columns=month_names)
            for m in month_names.values():
                if m not in pivot.columns:
                    pivot[m] = 0.0
            pivot = pivot[list(month_names.values())]

            fig_heatmap = px.imshow(
                pivot.values,
                x=list(pivot.columns),
                y=[str(y) for y in pivot.index],
                color_continuous_scale=["red", "white", "green"],
                color_continuous_midpoint=0,
                text_auto=".0f",
                labels={"x": "Month", "y": "Year", "color": "P&L ($)"},
                title="Monthly Returns Heatmap",
                aspect="auto",
            )
            fig_heatmap.update_layout(height=max(250, len(pivot) * 60 + 100))
            st.plotly_chart(fig_heatmap, use_container_width=True)

        # Long vs Short performance split
        render_long_short_split(trades)

        # Funding impact analysis (lazy-loaded)
        with st.expander("Perpetual Futures Analytics", expanded=False):
            render_funding_impact(backtest_result, trades)

        # Liquidation events
        render_liquidation_summary(trades)

        # Trade P&L distribution and scatter plots (lazy-loaded)
        with st.expander("Trade Analysis Charts", expanded=False):
            if closed_trades:
                # P&L distribution histogram
                st.subheader("Trade P&L Distribution")
                pnl_values = [float(t.get("net_pnl", 0.0) or 0.0) for t in closed_trades]
                if pnl_values:
                    fig_hist = go.Figure()
                    profits = [v for v in pnl_values if v >= 0]
                    losses = [v for v in pnl_values if v < 0]
                    if profits:
                        fig_hist.add_trace(go.Histogram(
                            x=profits,
                            name="Profit",
                            marker_color="green",
                            opacity=0.7,
                        ))
                    if losses:
                        fig_hist.add_trace(go.Histogram(
                            x=losses,
                            name="Loss",
                            marker_color="red",
                            opacity=0.7,
                        ))
                    fig_hist.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)
                    fig_hist.update_layout(
                        title="Trade P&L Distribution",
                        xaxis_title="P&L ($)",
                        yaxis_title="Count",
                        barmode="overlay",
                        height=400,
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

            # Trade scatter: ROI vs duration
            render_trade_scatter_roi_vs_duration(trades)

            # Trade scatter: size vs PnL
            render_trade_scatter_size_vs_pnl(trades)

        # Daily returns (lazy-loaded)
        with st.expander("Daily Returns", expanded=False):
            daily_df = compute_daily_returns(trades)
            render_daily_returns_chart(daily_df)

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

        # Highlight liquidations in trade log
        def _highlight_liquidations(row: pd.Series) -> list[str]:
            if row.get("exit_reason") == "liquidation":
                return ["background-color: #ffcccc"] * len(row)
            return [""] * len(row)

        if "exit_reason" in filtered_trades.columns:
            styled = filtered_trades.style.apply(_highlight_liquidations, axis=1)
            st.dataframe(styled, use_container_width=True, height=400)
        else:
            st.dataframe(filtered_trades, use_container_width=True, height=400)

        # Export trades
        csv_data = export_to_csv(filtered_trades, f"trades_{run_id}.csv")
        st.download_button(
            label="Export Trades to CSV",
            data=csv_data,
            file_name=f"trades_{run_id}.csv",
            mime="text/csv",
        )
    else:
        st.info("No trades recorded for this run")

    # Notes / annotations
    st.subheader("Notes")
    current_notes = backtest_result.get("notes", "") or ""
    new_notes = st.text_area(
        "Add observations or annotations about this backtest run",
        value=current_notes,
        height=100,
        key=f"notes_{run_id}",
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


# Top-level call for st.navigation API
render_results_tab()
