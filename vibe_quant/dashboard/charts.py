"""Plotly figure builders for the dashboard.

Every public function returns a :class:`plotly.graph_objects.Figure`.
No Streamlit calls — rendering is the caller's responsibility.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

JsonMapping = Mapping[str, object]


def _to_float(value: object, default: float = 0.0) -> float:
    """Best-effort conversion to float with fallback."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default

# ---------------------------------------------------------------------------
# Pareto charts
# ---------------------------------------------------------------------------


def build_pareto_scatter(sweep_df: pd.DataFrame) -> go.Figure | None:
    """Build 2-D Pareto front scatter (Sharpe vs Max DD, color=PF)."""
    plot_df = sweep_df.dropna(subset=["Sharpe", "Max DD"])
    if plot_df.empty:
        return None

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
    return fig


def build_3d_pareto_scatter(sweep_df: pd.DataFrame) -> go.Figure | None:
    """Build 3-D Pareto surface (Sharpe x MaxDD x PF)."""
    plot_df = sweep_df.dropna(subset=["Sharpe", "Max DD", "PF"])
    if len(plot_df) < 3:
        return None

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
    return fig


# ---------------------------------------------------------------------------
# Equity / drawdown
# ---------------------------------------------------------------------------


def build_equity_chart(equity_df: pd.DataFrame) -> go.Figure | None:
    """Build equity curve line chart."""
    if equity_df.empty:
        return None

    fig = go.Figure()
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
    return fig


def build_drawdown_chart(
    dd_df: pd.DataFrame,
    top_periods: Sequence[JsonMapping] | None = None,
) -> go.Figure | None:
    """Build drawdown area chart with optional top-5 period annotations."""
    if dd_df.empty:
        return None

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

    colors = ["#FF0000", "#FF4444", "#FF7777", "#FF9999", "#FFBBBB"]
    if top_periods:
        for i, period in enumerate(top_periods[:5]):
            start = period.get("start")
            end = period.get("end")
            if start is None or end is None:
                continue
            depth = _to_float(period.get("depth", 0.0), 0.0)
            color = colors[i] if i < len(colors) else colors[-1]
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor=color,
                opacity=0.15,
                line_width=0,
                annotation_text=f"#{i + 1}: {depth * 100:.1f}%",
                annotation_position="top left",
                annotation_font_size=10,
            )

    fig.update_layout(
        title="Drawdown Chart" + (" (Top-5 periods highlighted)" if top_periods else ""),
        xaxis_title="Time",
        yaxis_title="Drawdown (%)",
        height=300,
    )
    return fig


# ---------------------------------------------------------------------------
# Rolling / periodic charts
# ---------------------------------------------------------------------------


def build_rolling_sharpe_chart(rolling_df: pd.DataFrame) -> go.Figure | None:
    """Build rolling Sharpe ratio line chart."""
    if rolling_df.empty:
        return None

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
    return fig


def build_yearly_returns_chart(yearly_df: pd.DataFrame) -> go.Figure | None:
    """Build yearly returns bar chart."""
    if yearly_df.empty:
        return None

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
    return fig


def build_daily_returns_chart(daily_df: pd.DataFrame) -> go.Figure | None:
    """Build daily returns bar chart."""
    if daily_df.empty:
        return None

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
    return fig


# ---------------------------------------------------------------------------
# Trade analysis
# ---------------------------------------------------------------------------


def build_trade_scatter_roi_vs_duration(trades: Sequence[JsonMapping]) -> go.Figure | None:
    """Build scatter: ROI% vs duration (hours), coloured by direction."""
    closed = [t for t in trades if t.get("exit_time") and t.get("entry_time")]
    if not closed:
        return None

    data = []
    for t in closed:
        try:
            entry = datetime.fromisoformat(str(t["entry_time"]).replace("Z", "+00:00"))
            exit_ = datetime.fromisoformat(str(t["exit_time"]).replace("Z", "+00:00"))
            duration_h = (exit_ - entry).total_seconds() / 3600.0
            data.append({
                "duration_hours": duration_h,
                "roi_percent": _to_float(t.get("roi_percent", 0.0), 0.0),
                "direction": t.get("direction", "UNKNOWN"),
            })
        except (ValueError, TypeError):
            continue

    if not data:
        return None

    df = pd.DataFrame(data)
    fig = px.scatter(
        df, x="duration_hours", y="roi_percent", color="direction",
        color_discrete_map={"LONG": "#2196F3", "SHORT": "#FF5722"},
        title="Trade ROI vs Duration",
        labels={"duration_hours": "Duration (hours)", "roi_percent": "ROI (%)"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)
    fig.update_layout(height=400)
    return fig


def build_trade_scatter_size_vs_pnl(trades: Sequence[JsonMapping]) -> go.Figure | None:
    """Build scatter: position size (notional) vs net PnL."""
    closed = [t for t in trades if t.get("exit_time") and t.get("entry_price") and t.get("quantity")]
    if not closed:
        return None

    data = []
    for t in closed:
        notional = _to_float(t.get("entry_price", 0), 0.0) * _to_float(t.get("quantity", 0), 0.0)
        data.append({
            "notional": notional,
            "net_pnl": _to_float(t.get("net_pnl", 0.0), 0.0),
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
    return fig


def build_pnl_distribution(closed_trades: Sequence[JsonMapping]) -> go.Figure | None:
    """Build histogram of trade P&L (profit vs loss)."""
    pnl_values = [_to_float(t.get("net_pnl", 0.0), 0.0) for t in closed_trades]
    if not pnl_values:
        return None

    fig = go.Figure()
    profits = [v for v in pnl_values if v >= 0]
    losses = [v for v in pnl_values if v < 0]
    if profits:
        fig.add_trace(go.Histogram(x=profits, name="Profit", marker_color="green", opacity=0.7))
    if losses:
        fig.add_trace(go.Histogram(x=losses, name="Loss", marker_color="red", opacity=0.7))
    fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)
    fig.update_layout(
        title="Trade P&L Distribution",
        xaxis_title="P&L ($)",
        yaxis_title="Count",
        barmode="overlay",
        height=400,
    )
    return fig


def build_monthly_heatmap(closed_trades: Sequence[JsonMapping]) -> go.Figure | None:
    """Build monthly returns heatmap."""
    if not closed_trades:
        return None

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

    fig = px.imshow(
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
    fig.update_layout(height=max(250, len(pivot) * 60 + 100))
    return fig


# ---------------------------------------------------------------------------
# Comparison charts
# ---------------------------------------------------------------------------


def build_radar_chart(results: Sequence[JsonMapping]) -> go.Figure | None:
    """Build radar/spider chart for strategy comparison."""
    if not results:
        return None

    radar_metrics = [
        ("Sharpe", "sharpe_ratio", True),
        ("Sortino", "sortino_ratio", True),
        ("Calmar", "calmar_ratio", True),
        ("Win Rate", "win_rate", True),
        ("Profit Factor", "profit_factor", True),
        ("1-MaxDD", "max_drawdown", False),
    ]

    raw: dict[str, list[float]] = {}
    for label, key, _higher in radar_metrics:
        raw[label] = []
        for r in results:
            val = r.get(key)
            if val is None:
                raw[label].append(0.0)
            elif label == "1-MaxDD":
                raw[label].append(1.0 - abs(_to_float(val, 0.0)))
            else:
                raw[label].append(_to_float(val, 0.0))

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
    return fig


def build_funding_pie(gross_pnl: float, total_fees: float,
                      total_funding: float, total_slippage: float) -> go.Figure:
    """Build gross P&L breakdown pie chart."""
    # Use signed arithmetic consistent with how the caller computes gross_pnl
    net_pnl = gross_pnl - total_fees - total_funding - total_slippage

    # Pie charts require positive values — label the net slice appropriately
    funding_label = "Funding Paid" if total_funding >= 0 else "Funding Received"
    labels = [
        "Net P&L" if net_pnl >= 0 else "Net Loss",
        "Fees", funding_label, "Slippage",
    ]
    values = [abs(net_pnl), total_fees, abs(total_funding), total_slippage]

    # Guard against all-zero values (Plotly renders an empty pie)
    if not any(v > 0 for v in values):
        values = [1]
        labels = ["No P&L data"]
        colors_pie = ["#9E9E9E"]
    else:
        colors_pie = [
            "#4CAF50" if net_pnl >= 0 else "#F44336",
            "#FF9800", "#E91E63", "#9E9E9E",
        ]

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
    return fig
