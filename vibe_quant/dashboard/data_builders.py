"""Cached data loading and metric computation for dashboard pages.

Pure functions that transform raw data into DataFrames and metrics.
No Streamlit display calls — those live in the page modules.
"""

from __future__ import annotations

import io
import math
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------


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


def build_equity_curve_daily(
    trades: list[dict[str, Any]], starting_balance: float = 100000
) -> pd.DataFrame:
    """Build equity curve from daily aggregated returns.

    Instead of plotting per-trade, this aggregates to daily P&L,
    producing a smoother and more accurate equity representation
    that handles overlapping trades correctly.

    Args:
        trades: List of trade dicts with exit_time and net_pnl.
        starting_balance: Initial account equity.

    Returns:
        DataFrame with 'time' and 'equity' columns at daily granularity.
    """
    daily = compute_daily_returns(trades)
    if daily.empty:
        return pd.DataFrame()

    daily = daily.sort_values("date").reset_index(drop=True)
    equity = starting_balance
    points = [{"time": daily.iloc[0]["date"], "equity": equity}]

    for _, row in daily.iterrows():
        equity += row["daily_pnl"]
        points.append({"time": row["date"], "equity": equity})

    return pd.DataFrame(points)


# ---------------------------------------------------------------------------
# Statistical computations
# ---------------------------------------------------------------------------


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
# Export helpers
# ---------------------------------------------------------------------------


def build_benchmark_equity(
    trades: list[dict[str, Any]],
    starting_balance: float = 100000,
    annual_return: float = 0.50,
) -> pd.DataFrame:
    """Build a simple BTC buy-and-hold benchmark equity curve.

    Approximates BTC buy-and-hold using a constant growth rate
    matched to the strategy's time range. In a real deployment,
    this would use actual BTC price data from the catalog.

    Args:
        trades: Strategy trades (used to determine time range).
        starting_balance: Initial equity.
        annual_return: Assumed annual return for BTC (default 50%).

    Returns:
        DataFrame with 'time' and 'equity' columns.
    """
    if not trades:
        return pd.DataFrame()

    closed = [t for t in trades if t.get("exit_time")]
    if not closed:
        return pd.DataFrame()

    sorted_trades = sorted(closed, key=lambda x: x["exit_time"])
    start_time = sorted_trades[0].get("entry_time", sorted_trades[0]["exit_time"])
    end_time = sorted_trades[-1]["exit_time"]

    try:
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
    except Exception:
        return pd.DataFrame()

    if start_dt >= end_dt:
        return pd.DataFrame()

    # Generate daily points
    date_range = pd.date_range(start=start_dt, end=end_dt, freq="D")
    if len(date_range) < 2:
        return pd.DataFrame()

    total_days = (end_dt - start_dt).total_seconds() / 86400.0  # noqa: F841
    daily_return = (1.0 + annual_return) ** (1.0 / 365.0) - 1.0

    equity_values = []
    for i, dt in enumerate(date_range):
        equity = starting_balance * (1.0 + daily_return) ** i
        equity_values.append({"time": dt, "equity": equity})

    return pd.DataFrame(equity_values)


def export_to_csv(df: pd.DataFrame, filename: str) -> bytes:
    """Export DataFrame to CSV bytes."""
    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue().encode("utf-8")
