"""Paper Trading Tab for vibe-quant dashboard.

Live monitoring and control of paper trading sessions:
- Live P&L display
- Open positions table
- Recent trades list
- Strategy status (ACTIVE/HALTED)
- Manual controls: HALT, RESUME, CLOSE ALL
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import streamlit as st

from vibe_quant.paper.persistence import StateCheckpoint, StatePersistence, recover_state

if TYPE_CHECKING:
    from pathlib import Path


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


def _format_pnl(pnl: float) -> str:
    """Format P&L with sign and color indicator."""
    if pnl >= 0:
        return f"+${pnl:,.2f}"
    return f"-${abs(pnl):,.2f}"


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
            delta=_format_pnl(daily_pnl) if daily_pnl != 0 else None,
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
            _format_pnl(total_pnl),
            delta=_format_pnl(daily_pnl) if daily_pnl != 0 else None,
            delta_color="inverse" if total_pnl < 0 else "normal",
        )


def _render_status_indicator(checkpoint: StateCheckpoint | None) -> None:
    """Render strategy status indicator."""
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
            "Unrealized P&L": _format_pnl(float(pos.get("unrealized_pnl", 0))),
        })

    st.dataframe(
        positions_list,
        use_container_width=True,
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
        use_container_width=True,
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
            f"P&L: {_format_pnl(daily_pnl)}"
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
            use_container_width=True,
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
            use_container_width=True,
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
            use_container_width=True,
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
    return st.sidebar.button("Refresh Data", use_container_width=True)


def render_paper_trading_tab(db_path: Path | None = None) -> None:
    """Render the complete paper trading tab.

    Args:
        db_path: Optional database path. Uses default if not provided.
    """
    st.title("Paper Trading")

    # Sidebar controls
    trader_id = _render_trader_selector()
    should_refresh = _render_refresh_button()

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
