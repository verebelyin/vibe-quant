"""Tests for paper trading state persistence."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from vibe_quant.paper.persistence import (
    StateCheckpoint,
    StatePersistence,
    recover_state,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


class TestStateCheckpoint:
    """Tests for StateCheckpoint dataclass."""

    def test_default_values(self) -> None:
        """Should create checkpoint with defaults."""
        checkpoint = StateCheckpoint(trader_id="test")
        assert checkpoint.trader_id == "test"
        assert checkpoint.positions == {}
        assert checkpoint.orders == {}
        assert checkpoint.balance == {}
        assert checkpoint.node_status == {}
        assert checkpoint.timestamp is not None

    def test_to_dict(self) -> None:
        """Should convert to dict for storage."""
        checkpoint = StateCheckpoint(
            trader_id="PAPER-001",
            positions={"pos1": {"symbol": "BTCUSDT", "quantity": 0.1}},
            orders={"ord1": {"symbol": "ETHUSDT", "side": "BUY"}},
            balance={"total": 10000.0, "available": 5000.0},
            node_status={"state": "running"},
        )
        data = checkpoint.to_dict()
        assert data["trader_id"] == "PAPER-001"
        assert data["positions"]["pos1"]["symbol"] == "BTCUSDT"
        assert data["orders"]["ord1"]["side"] == "BUY"
        assert data["balance"]["total"] == 10000.0
        assert "timestamp" in data

    def test_from_dict(self) -> None:
        """Should create from dict."""
        data = {
            "trader_id": "PAPER-002",
            "positions": {"p1": {"qty": 1.0}},
            "orders": {},
            "balance": {"total": 5000.0},
            "node_status": {},
            "timestamp": "2024-01-15T10:00:00+00:00",
        }
        checkpoint = StateCheckpoint.from_dict(data)
        assert checkpoint.trader_id == "PAPER-002"
        assert checkpoint.positions["p1"]["qty"] == 1.0
        assert checkpoint.balance["total"] == 5000.0
        assert checkpoint.timestamp.year == 2024

    def test_roundtrip(self) -> None:
        """Should roundtrip through dict conversion."""
        original = StateCheckpoint(
            trader_id="test",
            positions={"p": {"x": 1}},
            orders={"o": {"y": 2}},
            balance={"z": 3},
            node_status={"s": 4},
        )
        restored = StateCheckpoint.from_dict(original.to_dict())
        assert restored.trader_id == original.trader_id
        assert restored.positions == original.positions
        assert restored.orders == original.orders
        assert restored.balance == original.balance
        assert restored.node_status == original.node_status


class TestStatePersistence:
    """Tests for StatePersistence class."""

    @pytest.fixture
    def persistence(self, tmp_path: Path) -> Generator[StatePersistence]:
        """Create StatePersistence with temp database."""
        db_path = tmp_path / "test_persistence.db"
        p = StatePersistence(db_path, trader_id="PAPER-TEST", checkpoint_interval=1)
        yield p
        p.close()

    def test_save_and_load_checkpoint(self, persistence: StatePersistence) -> None:
        """Should save and load checkpoint."""
        checkpoint = StateCheckpoint(
            trader_id="PAPER-TEST",
            positions={"btc": {"quantity": 0.5, "entry_price": 42000.0}},
            orders={"order1": {"side": "SELL", "price": 45000.0}},
            balance={"total": 10000.0, "available": 8000.0, "margin": 2000.0},
            node_status={"state": "running", "daily_pnl": 150.0},
        )
        checkpoint_id = persistence.save_checkpoint(checkpoint)
        assert checkpoint_id > 0

        loaded = persistence.load_checkpoint(checkpoint_id)
        assert loaded is not None
        assert loaded.trader_id == "PAPER-TEST"
        assert loaded.positions["btc"]["quantity"] == 0.5
        assert loaded.orders["order1"]["side"] == "SELL"
        assert loaded.balance["total"] == 10000.0
        assert loaded.node_status["daily_pnl"] == 150.0

    def test_load_nonexistent_checkpoint(self, persistence: StatePersistence) -> None:
        """Should return None for nonexistent checkpoint."""
        loaded = persistence.load_checkpoint(99999)
        assert loaded is None

    def test_load_latest_checkpoint(self, persistence: StatePersistence) -> None:
        """Should load most recent checkpoint."""
        # Save multiple checkpoints
        for i in range(3):
            checkpoint = StateCheckpoint(
                trader_id="PAPER-TEST",
                balance={"total": 10000.0 + i * 100},
            )
            persistence.save_checkpoint(checkpoint)

        latest = persistence.load_latest_checkpoint()
        assert latest is not None
        assert latest.balance["total"] == 10200.0  # Last saved

    def test_load_latest_no_checkpoints(self, persistence: StatePersistence) -> None:
        """Should return None when no checkpoints exist."""
        latest = persistence.load_latest_checkpoint()
        assert latest is None

    def test_load_latest_filters_by_trader(self, persistence: StatePersistence) -> None:
        """Should filter checkpoints by trader_id."""
        # Save for different traders
        persistence.save_checkpoint(
            StateCheckpoint(trader_id="PAPER-TEST", balance={"total": 1000.0})
        )
        persistence.save_checkpoint(
            StateCheckpoint(trader_id="OTHER-TRADER", balance={"total": 9999.0})
        )
        persistence.save_checkpoint(
            StateCheckpoint(trader_id="PAPER-TEST", balance={"total": 2000.0})
        )

        latest = persistence.load_latest_checkpoint("PAPER-TEST")
        assert latest is not None
        assert latest.balance["total"] == 2000.0

        other = persistence.load_latest_checkpoint("OTHER-TRADER")
        assert other is not None
        assert other.balance["total"] == 9999.0

    def test_list_checkpoints(self, persistence: StatePersistence) -> None:
        """Should list checkpoints in order."""
        for i in range(5):
            persistence.save_checkpoint(
                StateCheckpoint(trader_id="PAPER-TEST", balance={"seq": i})
            )

        checkpoints = persistence.list_checkpoints(limit=3)
        assert len(checkpoints) == 3
        # Most recent first
        assert checkpoints[0].balance["seq"] == 4
        assert checkpoints[1].balance["seq"] == 3
        assert checkpoints[2].balance["seq"] == 2

    def test_delete_old_checkpoints(self, persistence: StatePersistence) -> None:
        """Should delete old checkpoints keeping recent."""
        for i in range(10):
            persistence.save_checkpoint(
                StateCheckpoint(trader_id="PAPER-TEST", balance={"seq": i})
            )

        deleted = persistence.delete_old_checkpoints(keep_count=3)
        assert deleted == 7

        remaining = persistence.list_checkpoints(limit=100)
        assert len(remaining) == 3
        # Should keep most recent
        assert remaining[0].balance["seq"] == 9
        assert remaining[1].balance["seq"] == 8
        assert remaining[2].balance["seq"] == 7

    def test_checkpoint_interval_config(self, tmp_path: Path) -> None:
        """Should accept custom checkpoint interval."""
        db_path = tmp_path / "interval_test.db"
        p = StatePersistence(db_path, checkpoint_interval=120)
        assert p._checkpoint_interval == 120
        p.close()


class TestPeriodicCheckpointing:
    """Tests for periodic checkpointing."""

    @pytest.fixture
    def persistence(self, tmp_path: Path) -> Generator[StatePersistence]:
        """Create StatePersistence with short interval."""
        db_path = tmp_path / "periodic_test.db"
        p = StatePersistence(db_path, trader_id="PAPER-PERIODIC", checkpoint_interval=1)
        yield p
        p.close()

    @pytest.mark.asyncio
    async def test_periodic_checkpoint_creates_records(
        self, persistence: StatePersistence
    ) -> None:
        """Periodic checkpointing should create records at interval."""
        counter = 0

        def get_state() -> StateCheckpoint:
            nonlocal counter
            counter += 1
            return StateCheckpoint(
                trader_id="PAPER-PERIODIC",
                balance={"checkpoint_num": counter},
            )

        # Start with 1s interval
        await persistence.start_periodic_checkpointing(get_state)

        # Wait for ~2 checkpoints
        await asyncio.sleep(2.5)

        await persistence.stop_periodic_checkpointing()

        checkpoints = persistence.list_checkpoints()
        assert len(checkpoints) >= 2

    @pytest.mark.asyncio
    async def test_periodic_checkpoint_async_callback(
        self, persistence: StatePersistence
    ) -> None:
        """Should work with async state callback."""

        async def get_state_async() -> StateCheckpoint:
            await asyncio.sleep(0.01)
            return StateCheckpoint(
                trader_id="PAPER-PERIODIC",
                balance={"async": True},
            )

        await persistence.start_periodic_checkpointing(get_state_async)
        await asyncio.sleep(1.5)
        await persistence.stop_periodic_checkpointing()

        checkpoints = persistence.list_checkpoints()
        assert len(checkpoints) >= 1
        assert checkpoints[0].balance["async"] is True

    @pytest.mark.asyncio
    async def test_stop_periodic_checkpointing(
        self, persistence: StatePersistence
    ) -> None:
        """Should stop checkpointing when requested."""
        call_count = 0

        def get_state() -> StateCheckpoint:
            nonlocal call_count
            call_count += 1
            return StateCheckpoint(trader_id="PAPER-PERIODIC")

        await persistence.start_periodic_checkpointing(get_state)
        await asyncio.sleep(1.5)

        count_at_stop = call_count
        await persistence.stop_periodic_checkpointing()

        # Wait and verify no more calls
        await asyncio.sleep(1.5)
        assert call_count == count_at_stop


class TestRecovery:
    """Tests for crash recovery."""

    def test_recover_state_returns_latest(self, tmp_path: Path) -> None:
        """recover_state should return most recent checkpoint."""
        db_path = tmp_path / "recovery.db"

        # Save some state
        p = StatePersistence(db_path, trader_id="RECOVER-TEST")
        p.save_checkpoint(StateCheckpoint(trader_id="RECOVER-TEST", balance={"v": 1}))
        p.save_checkpoint(StateCheckpoint(trader_id="RECOVER-TEST", balance={"v": 2}))
        p.close()

        # Recover (simulating restart)
        checkpoint = recover_state(db_path, trader_id="RECOVER-TEST")
        assert checkpoint is not None
        assert checkpoint.balance["v"] == 2

    def test_recover_state_none_when_empty(self, tmp_path: Path) -> None:
        """recover_state should return None when no checkpoints."""
        db_path = tmp_path / "empty_recovery.db"
        checkpoint = recover_state(db_path, trader_id="NO-DATA")
        assert checkpoint is None

    def test_recovery_with_positions_and_orders(self, tmp_path: Path) -> None:
        """Should recover full state including positions and orders."""
        db_path = tmp_path / "full_recovery.db"

        # Save state with positions and orders
        p = StatePersistence(db_path, trader_id="FULL-TEST")
        p.save_checkpoint(
            StateCheckpoint(
                trader_id="FULL-TEST",
                positions={
                    "pos1": {
                        "symbol": "BTCUSDT",
                        "side": "LONG",
                        "quantity": 0.1,
                        "entry_price": 42000.0,
                        "unrealized_pnl": 500.0,
                    }
                },
                orders={
                    "ord1": {
                        "symbol": "BTCUSDT",
                        "side": "SELL",
                        "type": "LIMIT",
                        "price": 45000.0,
                        "quantity": 0.1,
                        "status": "OPEN",
                    }
                },
                balance={
                    "total": 10000.0,
                    "available": 5800.0,
                    "margin_used": 4200.0,
                },
                node_status={
                    "state": "running",
                    "daily_pnl": 500.0,
                    "consecutive_losses": 0,
                },
            )
        )
        p.close()

        # Recover
        checkpoint = recover_state(db_path, trader_id="FULL-TEST")
        assert checkpoint is not None

        # Verify positions
        assert "pos1" in checkpoint.positions
        pos = checkpoint.positions["pos1"]
        assert pos["symbol"] == "BTCUSDT"
        assert pos["quantity"] == 0.1
        assert pos["unrealized_pnl"] == 500.0

        # Verify orders
        assert "ord1" in checkpoint.orders
        order = checkpoint.orders["ord1"]
        assert order["price"] == 45000.0
        assert order["status"] == "OPEN"

        # Verify balance
        assert checkpoint.balance["total"] == 10000.0
        assert checkpoint.balance["margin_used"] == 4200.0

        # Verify node status
        assert checkpoint.node_status["daily_pnl"] == 500.0
