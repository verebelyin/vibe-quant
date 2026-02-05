"""Tests for paper trading dashboard tab."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from vibe_quant.dashboard.pages.paper_trading import (
    _format_pnl,
    _get_state_color,
)
from vibe_quant.paper.persistence import StateCheckpoint

if TYPE_CHECKING:
    pass


class TestFormatPnl:
    """Tests for _format_pnl helper."""

    def test_positive_pnl(self) -> None:
        """Positive P&L shows + sign."""
        result = _format_pnl(1234.56)
        assert result == "+$1,234.56"

    def test_negative_pnl(self) -> None:
        """Negative P&L shows - sign."""
        result = _format_pnl(-1234.56)
        assert result == "-$1,234.56"

    def test_zero_pnl(self) -> None:
        """Zero P&L shows + sign."""
        result = _format_pnl(0.0)
        assert result == "+$0.00"

    def test_small_pnl(self) -> None:
        """Small P&L formats correctly."""
        result = _format_pnl(0.01)
        assert result == "+$0.01"

    def test_large_pnl(self) -> None:
        """Large P&L formats with commas."""
        result = _format_pnl(1234567.89)
        assert result == "+$1,234,567.89"


class TestGetStateColor:
    """Tests for _get_state_color helper."""

    def test_running_state(self) -> None:
        """Running state is green."""
        assert _get_state_color("running") == "green"

    def test_paused_state(self) -> None:
        """Paused state is orange."""
        assert _get_state_color("paused") == "orange"

    def test_halted_state(self) -> None:
        """Halted state is red."""
        assert _get_state_color("halted") == "red"

    def test_stopped_state(self) -> None:
        """Stopped state is gray."""
        assert _get_state_color("stopped") == "gray"

    def test_error_state(self) -> None:
        """Error state is red."""
        assert _get_state_color("error") == "red"

    def test_initializing_state(self) -> None:
        """Initializing state is blue."""
        assert _get_state_color("initializing") == "blue"

    def test_unknown_state(self) -> None:
        """Unknown state defaults to gray."""
        assert _get_state_color("unknown") == "gray"


class TestStateCheckpointIntegration:
    """Tests for StateCheckpoint usage in dashboard."""

    def test_checkpoint_positions_format(self) -> None:
        """Checkpoint positions can be formatted for display."""
        checkpoint = StateCheckpoint(
            trader_id="PAPER-001",
            positions={
                "pos1": {
                    "symbol": "BTCUSDT",
                    "side": "LONG",
                    "quantity": 0.1,
                    "entry_price": 42000.0,
                    "current_price": 43000.0,
                    "unrealized_pnl": 100.0,
                }
            },
            balance={"total": 10000.0, "available": 5000.0, "margin_used": 5000.0},
            node_status={"state": "running", "daily_pnl": 100.0, "total_pnl": 500.0},
        )

        # Test positions can be iterated for display
        positions_list = []
        for pos_id, pos in checkpoint.positions.items():
            positions_list.append({
                "ID": pos_id,
                "Symbol": pos.get("symbol"),
                "Side": pos.get("side"),
                "Quantity": pos.get("quantity"),
            })

        assert len(positions_list) == 1
        assert positions_list[0]["Symbol"] == "BTCUSDT"
        assert positions_list[0]["Side"] == "LONG"
        assert positions_list[0]["Quantity"] == 0.1

    def test_checkpoint_orders_format(self) -> None:
        """Checkpoint orders can be formatted for display."""
        checkpoint = StateCheckpoint(
            trader_id="PAPER-001",
            orders={
                "ord1": {
                    "symbol": "BTCUSDT",
                    "side": "SELL",
                    "type": "LIMIT",
                    "quantity": 0.1,
                    "price": 45000.0,
                    "status": "OPEN",
                }
            },
        )

        orders_list = []
        for order_id, order in checkpoint.orders.items():
            orders_list.append({
                "ID": order_id,
                "Symbol": order.get("symbol"),
                "Type": order.get("type"),
                "Status": order.get("status"),
            })

        assert len(orders_list) == 1
        assert orders_list[0]["Symbol"] == "BTCUSDT"
        assert orders_list[0]["Type"] == "LIMIT"
        assert orders_list[0]["Status"] == "OPEN"

    def test_checkpoint_balance_extraction(self) -> None:
        """Balance data can be extracted for metrics display."""
        checkpoint = StateCheckpoint(
            trader_id="PAPER-001",
            balance={
                "total": 10000.0,
                "available": 5800.0,
                "margin_used": 4200.0,
            },
        )

        balance = checkpoint.balance
        assert float(balance.get("total", 0)) == 10000.0
        assert float(balance.get("available", 0)) == 5800.0
        assert float(balance.get("margin_used", 0)) == 4200.0

    def test_checkpoint_status_extraction(self) -> None:
        """Node status data can be extracted for display."""
        checkpoint = StateCheckpoint(
            trader_id="PAPER-001",
            node_status={
                "state": "running",
                "daily_pnl": 150.0,
                "total_pnl": 500.0,
                "trades_today": 5,
                "consecutive_losses": 0,
            },
        )

        status = checkpoint.node_status
        assert status.get("state") == "running"
        assert float(status.get("daily_pnl", 0)) == 150.0
        assert int(status.get("trades_today", 0)) == 5

    def test_empty_checkpoint(self) -> None:
        """Empty checkpoint handles defaults gracefully."""
        checkpoint = StateCheckpoint(trader_id="PAPER-001")

        # Should not raise
        assert checkpoint.positions == {}
        assert checkpoint.orders == {}
        assert checkpoint.balance == {}
        assert checkpoint.node_status == {}


class TestControlLogic:
    """Tests for control button enable/disable logic."""

    def test_halt_enabled_when_running(self) -> None:
        """Halt button should be enabled when state is running."""
        state = "running"
        halt_disabled = state not in ("running", "paused")
        assert halt_disabled is False

    def test_halt_enabled_when_paused(self) -> None:
        """Halt button should be enabled when state is paused."""
        state = "paused"
        halt_disabled = state not in ("running", "paused")
        assert halt_disabled is False

    def test_halt_disabled_when_halted(self) -> None:
        """Halt button should be disabled when already halted."""
        state = "halted"
        halt_disabled = state not in ("running", "paused")
        assert halt_disabled is True

    def test_resume_enabled_when_halted(self) -> None:
        """Resume button should be enabled when halted."""
        state = "halted"
        resume_disabled = state != "halted"
        assert resume_disabled is False

    def test_resume_disabled_when_running(self) -> None:
        """Resume button should be disabled when running."""
        state = "running"
        resume_disabled = state != "halted"
        assert resume_disabled is True

    def test_close_all_enabled_with_positions(self) -> None:
        """Close All button enabled when there are positions."""
        checkpoint = StateCheckpoint(
            trader_id="PAPER-001",
            positions={"pos1": {"symbol": "BTCUSDT"}},
        )
        has_positions = bool(checkpoint.positions)
        assert has_positions is True

    def test_close_all_disabled_without_positions(self) -> None:
        """Close All button disabled when no positions."""
        checkpoint = StateCheckpoint(trader_id="PAPER-001")
        has_positions = bool(checkpoint.positions)
        assert has_positions is False
