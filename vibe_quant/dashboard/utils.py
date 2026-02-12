"""Shared utilities for dashboard pages.

Centralises session-state singletons and formatting helpers so every
page uses the same instances and consistent display logic.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from vibe_quant.db.connection import DEFAULT_DB_PATH
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager

# ---------------------------------------------------------------------------
# Session-state singletons
# ---------------------------------------------------------------------------

def get_state_manager(db_path: Path | None = None) -> StateManager:
    """Get or create a single :class:`StateManager` in session state."""
    if "state_manager" not in st.session_state:
        resolved = Path(
            st.session_state.get("db_path", str(db_path or DEFAULT_DB_PATH))
        )
        st.session_state["state_manager"] = StateManager(resolved)
    manager: StateManager = st.session_state["state_manager"]
    return manager


def get_job_manager(db_path: Path | None = None) -> BacktestJobManager:
    """Get or create a single :class:`BacktestJobManager` in session state."""
    if "job_manager" not in st.session_state:
        resolved = Path(
            st.session_state.get("db_path", str(db_path or DEFAULT_DB_PATH))
        )
        st.session_state["job_manager"] = BacktestJobManager(resolved)
    manager: BacktestJobManager = st.session_state["job_manager"]
    return manager


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------

def format_percent(val: float | None) -> str:
    """Format *val* as a percentage string, e.g. ``'12.34%'``."""
    if val is None:
        return "N/A"
    return f"{val * 100:.2f}%"


def format_number(val: float | None, decimals: int = 2) -> str:
    """Format *val* to *decimals* decimal places."""
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def format_dollar(val: float | None) -> str:
    """Format *val* as a US-dollar string, e.g. ``'$1,234.56'``."""
    if val is None:
        return "N/A"
    return f"${val:,.2f}"


def format_pnl(pnl: float) -> str:
    """Format P&L with sign, e.g. ``'+$100.00'`` or ``'-$50.00'``."""
    if pnl >= 0:
        return f"+${pnl:,.2f}"
    return f"-${abs(pnl):,.2f}"


def format_bytes(size_bytes: int) -> str:
    """Format *size_bytes* to a human-readable string."""
    value: float = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"
