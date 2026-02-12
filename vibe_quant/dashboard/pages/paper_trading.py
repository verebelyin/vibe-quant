"""Paper Trading Tab for vibe-quant dashboard.

Live monitoring and control of paper trading sessions:
- Strategy promotion: select validated strategy -> start paper trading
- Live P&L display
- Open positions table
- Recent trades list
- Strategy status (ACTIVE/HALTED)
- Manual controls: HALT, RESUME, CLOSE ALL
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st

from vibe_quant.dashboard.utils import format_pnl, get_job_manager
from vibe_quant.db.connection import get_connection
from vibe_quant.paper.persistence import StateCheckpoint, StatePersistence, recover_state

# Session state keys
SESSION_TRADER_ID = "paper_trading_trader_id"
SESSION_PERSISTENCE = "paper_trading_persistence"
SESSION_LAST_REFRESH = "paper_trading_last_refresh"


def _get_persistence(db_path: Path | None = None) -> StatePersistence | None:
    """Get or create StatePersistence in session state."""
    trader_id = st.session_state.get(SESSION_TRADER_ID)
    if not trader_id:
        return None

    if SESSION_PERSISTENCE not in st.session_state:
        st.session_state[SESSION_PERSISTENCE] = StatePersistence(
            db_path, trader_id=trader_id
        )
    persistence: StatePersistence = st.session_state[SESSION_PERSISTENCE]
    return persistence


def _get_validated_strategies(db_path: Path | None = None) -> list[dict[str, Any]]:
    """Get strategies with completed validation backtests.

    Returns strategies from backtest_results that have:
    - run_mode = 'validation'
    - status = 'completed'
    - sharpe_ratio IS NOT NULL
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT
                s.id as strategy_id,
                s.name as strategy_name,
                br.id as run_id,
                br.symbols,
                br.timeframe,
                res.sharpe_ratio,
                COALESCE(res.walk_forward_efficiency, 0.0) as walk_forward_efficiency,
                COALESCE(res.deflated_sharpe, 0.0) as deflated_sharpe,
                COALESCE(res.purged_kfold_mean_sharpe, 0.0) as purged_kfold_mean_sharpe,
                COALESCE(res.max_drawdown, 0.0) as max_drawdown,
                COALESCE(res.total_return, 0.0) as total_return
            FROM backtest_results res
            JOIN backtest_runs br ON res.run_id = br.id
            JOIN strategies s ON br.strategy_id = s.id
            WHERE br.run_mode = 'validation'
              AND br.status = 'completed'
              AND res.sharpe_ratio IS NOT NULL
            ORDER BY res.sharpe_ratio DESC
            """
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _get_active_paper_jobs(db_path: Path | None = None) -> list[dict[str, Any]]:
    """Get currently running paper trading jobs."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT * FROM background_jobs
            WHERE job_type = 'paper_trading' AND status = 'running'
            """
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _create_paper_config_file(
    trader_id: str,
    strategy_id: int,
    symbols: list[str],
    testnet: bool,
    db_path: Path | None,
) -> Path:
    """Create paper trading config JSON file for subprocess.

    API credentials are NOT written to disk -- they are passed via
    environment variables at subprocess launch time.
    """
    config_data = {
        "trader_id": trader_id,
        "strategy_id": strategy_id,
        "symbols": symbols,
        "binance": {
            "testnet": testnet,
            "account_type": "USDT_FUTURES",
        },
        "sizing": {
            "method": "fixed_fractional",
            "max_leverage": "10",
            "max_position_pct": "0.3",
            "risk_per_trade": "0.02",
        },
        "risk": {
            "max_drawdown_pct": "0.15",
            "max_daily_loss_pct": "0.05",
            "max_consecutive_losses": 5,
            "max_position_count": 3,
        },
        "db_path": str(db_path) if db_path else None,
        "logs_path": f"logs/paper/{trader_id}",
        "state_persistence_interval": 60,
    }

    config_path = Path(f"/tmp/paper_{trader_id}.json")
    with config_path.open("w") as f:
        json.dump(config_data, f, indent=2)

    return config_path


def _render_start_session(db_path: Path | None = None) -> None:
    """Render start session section for promoting validated strategies."""
    st.subheader("Start New Session")

    # Check for existing active jobs
    active_jobs = _get_active_paper_jobs(db_path)
    if active_jobs:
        st.warning(f"Paper trading session already running (PID: {active_jobs[0]['pid']})")
        if st.button("Stop Active Session", type="secondary"):
            manager = get_job_manager(db_path)
            manager.kill_job(active_jobs[0]["run_id"])
            st.success("Session stopped")
            st.rerun()
        return

    # Get validated strategies
    strategies = _get_validated_strategies(db_path)

    if not strategies:
        st.info("No validated strategies available. Run a validation backtest first.")
        return

    # Strategy selector
    strategy_options = {
        f"{s['strategy_name']} | Sharpe: {s['sharpe_ratio']:.2f}": s
        for s in strategies
    }

    selected_label = st.selectbox(
        "Select Validated Strategy",
        options=list(strategy_options.keys()),
        help="Strategies with completed validation backtests",
    )

    if not selected_label:
        return

    selected = strategy_options[selected_label]

    # Show strategy details
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sharpe Ratio", f"{selected['sharpe_ratio']:.2f}")
    with col2:
        st.metric("Max Drawdown", f"{selected['max_drawdown']:.1%}")
    with col3:
        st.metric("Total Return", f"{selected['total_return']:.1%}")

    # Binance config (collapsible)
    with st.expander("Binance Configuration", expanded=False):
        api_key = st.text_input(
            "API Key",
            type="password",
            value=os.environ.get("BINANCE_API_KEY", ""),
            help="Binance API key (from env BINANCE_API_KEY if set)",
        )
        api_secret = st.text_input(
            "API Secret",
            type="password",
            value=os.environ.get("BINANCE_API_SECRET", ""),
            help="Binance API secret (from env BINANCE_API_SECRET if set)",
        )
        testnet = st.checkbox("Use Testnet", value=True, help="Paper trade on Binance testnet")

    # Parse symbols from JSON
    symbols_json = selected.get("symbols", "[]")
    symbols = json.loads(symbols_json) if isinstance(symbols_json, str) else symbols_json

    # Start button
    if st.button("Start Paper Trading", type="primary", width="stretch"):
        if not api_key or not api_secret:
            st.error("API key and secret required")
            return

        # Generate trader ID
        trader_id = f"PAPER-{uuid.uuid4().hex[:8].upper()}"

        # Create config file (credentials passed via env, not written to disk)
        config_path = _create_paper_config_file(
            trader_id=trader_id,
            strategy_id=selected["strategy_id"],
            symbols=symbols,
            testnet=testnet,
            db_path=db_path,
        )

        # Create a run record for tracking
        conn = get_connection(db_path)
        try:
            cursor = conn.execute(
                """
                INSERT INTO backtest_runs
                (strategy_id, run_mode, symbols, timeframe, start_date, end_date, parameters, status)
                VALUES (?, 'paper_trading', ?, ?, date('now'), date('now', '+1 year'), '{}', 'pending')
                """,
                (selected["strategy_id"], json.dumps(symbols), selected.get("timeframe", "1h")),
            )
            conn.commit()
            run_id = cursor.lastrowid
            if run_id is None:
                st.error("Failed to create run record")
                return
        finally:
            conn.close()

        # Start subprocess with credentials in env (not on disk)
        manager = get_job_manager(db_path)
        command = [
            "python", "-m", "vibe_quant.paper.cli",
            "start",
            "--config", str(config_path),
            "--run-id", str(run_id),
        ]
        log_file = f"logs/paper/{trader_id}/paper_trading.log"

        env = os.environ.copy()
        env["BINANCE_API_KEY"] = api_key
        env["BINANCE_API_SECRET"] = api_secret

        pid = manager.start_job(
            run_id=run_id,
            job_type="paper_trading",
            command=command,
            log_file=log_file,
            env=env,
        )

        # Update session state with new trader ID
        st.session_state[SESSION_TRADER_ID] = trader_id

        st.success(f"Paper trading started! Trader ID: {trader_id} (PID: {pid})")
        st.rerun()


def _get_state_color(state: str) -> str:
    """Get color for state display."""
    colors = {
        "running": "green",
        "paused": "orange",
        "halted": "red",
        "stopped": "gray",
        "error": "red",
        "initializing": "blue",
    }
    return colors.get(state, "gray")




def _render_pnl_metrics(checkpoint: StateCheckpoint | None) -> None:
    """Render P&L metrics section."""
    st.subheader("P&L Summary")

    if checkpoint is None:
        st.info("No checkpoint data available. Start a paper trading session.")
        return

    balance = checkpoint.balance
    node_status = checkpoint.node_status

    # Extract values with defaults
    total_balance = float(balance.get("total", 0))
    available = float(balance.get("available", 0))
    margin_used = float(balance.get("margin_used", 0))
    daily_pnl = float(node_status.get("daily_pnl", 0))
    total_pnl = float(node_status.get("total_pnl", 0))

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Balance",
            f"${total_balance:,.2f}",
            delta=format_pnl(daily_pnl) if daily_pnl != 0 else None,
        )

    with col2:
        st.metric(
            "Available",
            f"${available:,.2f}",
        )

    with col3:
        st.metric(
            "Margin Used",
            f"${margin_used:,.2f}",
        )

    with col4:
        st.metric(
            "Total P&L",
            format_pnl(total_pnl),
            delta=format_pnl(daily_pnl) if daily_pnl != 0 else None,
            delta_color="normal",
        )


@st.fragment(run_every=5)
def _render_status_indicator(checkpoint: StateCheckpoint | None) -> None:
    """Render strategy status indicator. Auto-refreshes every 5s."""
    st.subheader("Strategy Status")

    if checkpoint is None:
        st.warning("No active session")
        return

    node_status = checkpoint.node_status
    state = node_status.get("state", "unknown")
    halt_reason = node_status.get("halt_reason")
    error_message = node_status.get("error_message")

    col1, col2, col3 = st.columns(3)

    with col1:
        color = _get_state_color(state)
        st.markdown(f"**Status:** :{color}[{state.upper()}]")

    with col2:
        trades_today = int(node_status.get("trades_today", 0))
        st.markdown(f"**Trades Today:** {trades_today}")

    with col3:
        consecutive_losses = int(node_status.get("consecutive_losses", 0))
        st.markdown(f"**Consecutive Losses:** {consecutive_losses}")

    if halt_reason:
        st.error(f"Halt Reason: {halt_reason}")
    if error_message:
        st.error(f"Error: {error_message}")


def _render_positions_table(checkpoint: StateCheckpoint | None) -> None:
    """Render open positions table."""
    st.subheader("Open Positions")

    if checkpoint is None or not checkpoint.positions:
        st.info("No open positions")
        return

    # Convert positions dict to list for display
    positions_list = []
    for pos_id, pos in checkpoint.positions.items():
        positions_list.append({
            "ID": pos_id[:8] + "..." if len(pos_id) > 8 else pos_id,
            "Symbol": pos.get("symbol", "N/A"),
            "Side": pos.get("side", "N/A"),
            "Quantity": pos.get("quantity", 0),
            "Entry Price": f"${float(pos.get('entry_price', 0)):,.2f}",
            "Current Price": f"${float(pos.get('current_price', 0)):,.2f}",
            "Unrealized P&L": format_pnl(float(pos.get("unrealized_pnl", 0))),
        })

    st.dataframe(
        positions_list,
        width="stretch",
        hide_index=True,
    )


def _render_orders_table(checkpoint: StateCheckpoint | None) -> None:
    """Render pending orders table."""
    st.subheader("Pending Orders")

    if checkpoint is None or not checkpoint.orders:
        st.info("No pending orders")
        return

    orders_list = []
    for order_id, order in checkpoint.orders.items():
        orders_list.append({
            "ID": order_id[:8] + "..." if len(order_id) > 8 else order_id,
            "Symbol": order.get("symbol", "N/A"),
            "Side": order.get("side", "N/A"),
            "Type": order.get("type", "N/A"),
            "Quantity": order.get("quantity", 0),
            "Price": f"${float(order.get('price', 0)):,.2f}",
            "Status": order.get("status", "N/A"),
        })

    st.dataframe(
        orders_list,
        width="stretch",
        hide_index=True,
    )


def _render_recent_trades(persistence: StatePersistence | None) -> None:
    """Render recent trades list from checkpoint history."""
    st.subheader("Recent Checkpoints")

    if persistence is None:
        st.info("Select a trader ID to view history")
        return

    checkpoints = persistence.list_checkpoints(limit=10)

    if not checkpoints:
        st.info("No checkpoint history")
        return

    # Show checkpoint timeline
    for cp in checkpoints[:5]:
        node_status = cp.node_status
        state = node_status.get("state", "unknown")
        daily_pnl = float(node_status.get("daily_pnl", 0))
        timestamp = cp.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        color = _get_state_color(state)
        st.markdown(
            f":{color}[{state.upper()}] | "
            f"{timestamp} | "
            f"P&L: {format_pnl(daily_pnl)}"
        )


def _render_controls(checkpoint: StateCheckpoint | None) -> dict[str, bool]:
    """Render manual control buttons.

    Returns:
        Dict with action keys (halt, resume, close_all) and bool values.
    """
    st.subheader("Manual Controls")

    actions: dict[str, bool] = {"halt": False, "resume": False, "close_all": False}

    col1, col2, col3 = st.columns(3)

    state = "unknown"
    if checkpoint is not None:
        state = checkpoint.node_status.get("state", "unknown")

    with col1:
        # Halt button - only enabled if running
        halt_disabled = state not in ("running", "paused")
        if st.button(
            "HALT",
            type="primary",
            disabled=halt_disabled,
            width="stretch",
            help="Halt trading immediately",
        ):
            actions["halt"] = True

    with col2:
        # Resume button - only enabled if halted due to error
        resume_disabled = state != "halted"
        if st.button(
            "RESUME",
            type="secondary",
            disabled=resume_disabled,
            width="stretch",
            help="Resume from error halt",
        ):
            actions["resume"] = True

    with col3:
        # Close All button - enabled if there are positions
        has_positions = checkpoint is not None and bool(checkpoint.positions)
        if st.button(
            "CLOSE ALL",
            type="secondary",
            disabled=not has_positions,
            width="stretch",
            help="Close all open positions",
        ):
            actions["close_all"] = True

    return actions


def _handle_actions(actions: dict[str, bool]) -> None:
    """Handle control button actions."""
    if actions["halt"]:
        st.warning("HALT action triggered. In production, this would halt the trading node.")
        # In real implementation: send halt signal to trading node

    if actions["resume"]:
        st.info("RESUME action triggered. In production, this would resume the trading node.")
        # In real implementation: send resume signal to trading node

    if actions["close_all"]:
        st.warning("CLOSE ALL action triggered. In production, this would close all positions.")
        # In real implementation: send close all signal to trading node


def _render_trader_selector() -> str | None:
    """Render trader ID selector."""
    st.sidebar.subheader("Paper Trading")

    trader_id: str = st.sidebar.text_input(
        "Trader ID",
        value=st.session_state.get(SESSION_TRADER_ID, ""),
        placeholder="e.g., PAPER-001",
    )

    if trader_id:
        st.session_state[SESSION_TRADER_ID] = trader_id
        return trader_id

    return None


def _render_refresh_button() -> bool:
    """Render refresh button in sidebar."""
    return st.sidebar.button("Refresh Data", width="stretch")


def render_paper_trading_tab(db_path: Path | None = None) -> None:
    """Render the complete paper trading tab.

    Args:
        db_path: Optional database path. Uses default if not provided.
    """
    st.title("Paper Trading")

    # Sidebar controls
    trader_id = _render_trader_selector()
    should_refresh = _render_refresh_button()

    # Start session section at top
    _render_start_session(db_path)
    st.divider()

    if not trader_id:
        st.info("Enter a Trader ID in the sidebar to view paper trading status.")
        return

    # Load checkpoint data
    checkpoint = recover_state(db_path, trader_id=trader_id)
    persistence = _get_persistence(db_path)

    # Update last refresh time
    if should_refresh:
        st.session_state[SESSION_LAST_REFRESH] = datetime.now(UTC)

    # Show last refresh time
    last_refresh = st.session_state.get(SESSION_LAST_REFRESH)
    if last_refresh:
        st.caption(f"Last refreshed: {last_refresh.strftime('%H:%M:%S')}")

    # Main content in columns
    left_col, right_col = st.columns([2, 1])

    with left_col:
        _render_pnl_metrics(checkpoint)
        st.divider()
        _render_positions_table(checkpoint)
        st.divider()
        _render_orders_table(checkpoint)

    with right_col:
        _render_status_indicator(checkpoint)
        st.divider()
        actions = _render_controls(checkpoint)
        _handle_actions(actions)
        st.divider()
        _render_recent_trades(persistence)


# Convenience alias for app.py imports
render = render_paper_trading_tab

# Top-level call for st.navigation API
render_paper_trading_tab()
