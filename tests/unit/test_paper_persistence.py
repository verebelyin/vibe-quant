"""Tests for paper trading state persistence."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import pytest

from vibe_quant.paper.persistence import (
    StateCheckpoint,
    StatePersistence,
    recover_state,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path


async def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Yield to the loop until ``predicate`` is true (bounded, no wall-clock sleep)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.005)
    return False


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
            persistence.save_checkpoint(StateCheckpoint(trader_id="PAPER-TEST", balance={"seq": i}))

        checkpoints = persistence.list_checkpoints(limit=3)
        assert len(checkpoints) == 3
        # Most recent first
        assert checkpoints[0].balance["seq"] == 4
        assert checkpoints[1].balance["seq"] == 3
        assert checkpoints[2].balance["seq"] == 2

    def test_delete_old_checkpoints(self, persistence: StatePersistence) -> None:
        """Should delete old checkpoints keeping recent."""
        for i in range(10):
            persistence.save_checkpoint(StateCheckpoint(trader_id="PAPER-TEST", balance={"seq": i}))

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


class TestHasAccountCheckpoint:
    """6ub67: does this account have ANY saved risk state (any trader)?"""

    @pytest.fixture
    def store(self, tmp_path: Path) -> Generator[StatePersistence]:
        p = StatePersistence(tmp_path / "accounts.db", trader_id="OWN-001")
        yield p
        p.close()

    @staticmethod
    def _save(store: StatePersistence, trader_id: str, node_status: dict[str, object]) -> None:
        store.save_checkpoint(StateCheckpoint(trader_id=trader_id, node_status=node_status))

    @staticmethod
    def _save_raw(store: StatePersistence, trader_id: str, node_status: str) -> None:
        store.conn.execute(
            """INSERT INTO paper_trading_checkpoints
               (trader_id, positions, orders, balance, node_status)
               VALUES (?, '{}', '{}', '{}', ?)""",
            (trader_id, node_status),
        )
        store.conn.commit()

    def test_empty_table_has_no_checkpoint(self, store: StatePersistence) -> None:
        assert store.has_account_checkpoint("binance:live:aaaa", "OWN-001") is False

    def test_another_trader_on_the_account_counts(self, store: StatePersistence) -> None:
        # A rotated key resumes the account through a DIFFERENT trader id.
        self._save(store, "OTHER-001", {"account": "binance:live:aaaa"})
        assert store.has_account_checkpoint("binance:live:aaaa", "OWN-001") is True

    def test_another_account_does_not_count(self, store: StatePersistence) -> None:
        self._save(store, "OTHER-001", {"account": "binance:live:bbbb"})
        self._save(store, "OWN-001", {"account": "binance:testnet:aaaa"})
        assert store.has_account_checkpoint("binance:live:aaaa", "OWN-001") is False

    def test_legacy_row_of_this_trader_counts(self, store: StatePersistence) -> None:
        """A row without the account key predates account identity: it is this trader's."""
        self._save(store, "OWN-001", {"state": "running"})
        assert store.has_account_checkpoint("binance:live:aaaa", "OWN-001") is True

    def test_legacy_row_of_another_trader_does_not_count(self, store: StatePersistence) -> None:
        self._save(store, "OTHER-001", {"state": "running"})
        assert store.has_account_checkpoint("binance:live:aaaa", "OWN-001") is False

    def test_invalid_json_of_another_trader_does_not_count(self, store: StatePersistence) -> None:
        self._save_raw(store, "OTHER-001", "{not json")
        assert store.has_account_checkpoint("binance:live:aaaa", "OWN-001") is False

    def test_invalid_json_of_this_trader_does_not_count(self, store: StatePersistence) -> None:
        """Conservative: an unparseable own row cannot prove the account, so refuse."""
        self._save_raw(store, "OWN-001", "{not json")
        assert store.has_account_checkpoint("binance:live:aaaa", "OWN-001") is False

    # neuvu.2: a key rotation by the same trader id carries its risk, so it counts.

    def test_own_row_on_another_live_key_is_a_rotation(self, store: StatePersistence) -> None:
        """R23: same trader, same venue + env, other key -> something carries."""
        self._save(store, "OWN-001", {"account": "binance:live:bbbb"})
        assert store.has_account_checkpoint("binance:live:aaaa", "OWN-001") is True

    def test_rotation_needs_the_own_row_not_another_traders(self, store: StatePersistence) -> None:
        self._save(store, "OTHER-001", {"account": "binance:live:bbbb"})
        assert store.has_account_checkpoint("binance:live:aaaa", "OWN-001") is False

    def test_rotation_reads_the_newest_own_row_only(self, store: StatePersistence) -> None:
        """R13/ae6ah (D1=A): an older live-key row under a newer testnet row carries.

        The gate reads the newest row IN the venue + env, so another env's newer row
        never hides the trader's own earlier live session on another key.
        """
        self._save(store, "OWN-001", {"account": "binance:live:bbbb"})
        self._save(store, "OWN-001", {"account": "binance:testnet:bbbb"})
        assert store.has_account_checkpoint("binance:live:aaaa", "OWN-001") is True
        # ... rotating back to the first key keeps it True (A -> B -> A)
        self._save(store, "OWN-001", {"account": "binance:live:bbbb"})
        assert store.has_account_checkpoint("binance:live:aaaa", "OWN-001") is True

    def test_rotation_stays_inside_the_venue_and_env(self, store: StatePersistence) -> None:
        self._save(store, "OWN-001", {"account": "binance:testnet:bbbb"})
        assert store.has_account_checkpoint("binance:live:aaaa", "OWN-001") is False

    def test_non_string_account_is_not_a_rotation_and_never_raises(
        self, store: StatePersistence
    ) -> None:
        """R32: a corrupt account value fails closed (no consent saved), no exception."""
        self._save_raw(store, "OWN-001", '{"account": 5}')
        assert store.has_account_checkpoint("binance:live:aaaa", "OWN-001") is False
        self._save_raw(store, "OWN-001", '{"account": ["binance:live:bbbb"]}')
        assert store.has_account_checkpoint("binance:live:aaaa", "OWN-001") is False

    def test_invalid_json_newest_row_is_not_a_rotation_and_never_raises(
        self, store: StatePersistence
    ) -> None:
        """R30: an unparseable newest own row cannot prove a rotation."""
        self._save(store, "OWN-001", {"account": "binance:live:bbbb"})
        self._save_raw(store, "OWN-001", "{not json")
        assert store.has_account_checkpoint("binance:live:aaaa", "OWN-001") is False


class TestLatestEnvCheckpoint:
    """R13 (ae6ah): the newest own row in one venue + env, whatever its key."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> Generator[StatePersistence]:
        p = StatePersistence(tmp_path / "env.db", trader_id="OWN-001")
        yield p
        p.close()

    @staticmethod
    def _save(store: StatePersistence, trader_id: str, node_status: dict[str, object]) -> None:
        store.save_checkpoint(StateCheckpoint(trader_id=trader_id, node_status=node_status))

    @staticmethod
    def _save_raw(store: StatePersistence, trader_id: str, node_status: str) -> None:
        store.conn.execute(
            """INSERT INTO paper_trading_checkpoints
               (trader_id, positions, orders, balance, node_status)
               VALUES (?, '{}', '{}', '{}', ?)""",
            (trader_id, node_status),
        )
        store.conn.commit()

    def test_same_env_and_legacy_rows_count(self, store: StatePersistence) -> None:
        """Any key of the env, a null account and a legacy row (no key) all count."""
        self._save(store, "OWN-001", {"account": "binance:live:aaaa"})
        self._save(store, "OWN-001", {"account": "binance:testnet:aaaa"})
        self._save(store, "OWN-001", {"account": "binance:live:bbbb"})
        latest = store.load_latest_env_checkpoint("OWN-001", "binance:live:aaaa")
        assert latest is not None
        assert latest.node_status == {"account": "binance:live:bbbb"}
        # a JSON null account is this env, and being newest it wins
        self._save(store, "OWN-001", {"account": None})
        latest = store.load_latest_env_checkpoint("OWN-001", "binance:live:aaaa")
        assert latest is not None
        assert latest.node_status == {"account": None}
        # so is a legacy row without the key
        self._save(store, "OWN-001", {"state": "running"})
        latest = store.load_latest_env_checkpoint("OWN-001", "binance:live:aaaa")
        assert latest is not None
        assert latest.node_status == {"state": "running"}

    def test_other_env_rows_skipped(self, store: StatePersistence) -> None:
        """Testnet and live (or another venue) never serve each other's rows."""
        self._save(store, "OWN-001", {"account": "binance:testnet:aaaa"})
        assert store.load_latest_env_checkpoint("OWN-001", "binance:live:aaaa") is None
        latest = store.load_latest_env_checkpoint("OWN-001", "binance:testnet:aaaa")
        assert latest is not None
        assert latest.node_status == {"account": "binance:testnet:aaaa"}
        self._save(store, "OWN-001", {"account": "other:live:aaaa"})
        assert store.load_latest_env_checkpoint("OWN-001", "binance:live:aaaa") is None

    def test_env_prefix_without_hash_is_skipped(self, store: StatePersistence) -> None:
        """'binance:live' and 'binance:livex:...' are not 'binance:live:' rows."""
        self._save_raw(store, "OWN-001", '{"account": "binance:live"}')
        assert store.load_latest_env_checkpoint("OWN-001", "binance:live:aaaa") is None
        self._save(store, "OWN-001", {"account": "binance:livex:aaaa"})
        assert store.load_latest_env_checkpoint("OWN-001", "binance:live:aaaa") is None
        self._save(store, "OWN-001", {"account": "binance:live:aaaa"})
        latest = store.load_latest_env_checkpoint("OWN-001", "binance:live:aaaa")
        assert latest is not None
        assert latest.node_status == {"account": "binance:live:aaaa"}

    def test_gate_rotation_read_uses_the_env_row(self, store: StatePersistence) -> None:
        """A->B->A: the gate finds the newest same-env key row, not the newest row."""
        self._save(store, "OWN-001", {"account": "binance:live:aaaa"})
        self._save(store, "OWN-001", {"account": "binance:live:bbbb"})
        # a third live key rotates from B's row
        assert store.has_account_checkpoint("binance:live:cccc", "OWN-001") is True
        # ... even when another env's row is newer (the D1=A flip)
        self._save(store, "OWN-001", {"account": "binance:testnet:cccc"})
        assert store.has_account_checkpoint("binance:live:cccc", "OWN-001") is True


class TestPeriodicCheckpointing:
    """Tests for periodic checkpointing."""

    @pytest.fixture
    def persistence(self, tmp_path: Path) -> Generator[StatePersistence]:
        """Create StatePersistence with a tiny (sub-second) test interval."""
        db_path = tmp_path / "periodic_test.db"
        p = StatePersistence(db_path, trader_id="PAPER-PERIODIC", checkpoint_interval=0.02)
        yield p
        p.close()

    @pytest.mark.asyncio
    async def test_periodic_checkpoint_creates_records(self, persistence: StatePersistence) -> None:
        """Periodic checkpointing should create records at interval."""
        counter = 0

        def get_state() -> StateCheckpoint:
            nonlocal counter
            counter += 1
            return StateCheckpoint(
                trader_id="PAPER-PERIODIC",
                balance={"checkpoint_num": counter},
            )

        await persistence.start_periodic_checkpointing(get_state)
        try:
            # At least two periodic checkpoints are written.
            assert await _wait_until(lambda: len(persistence.list_checkpoints()) >= 2)
        finally:
            await persistence.stop_periodic_checkpointing()

        checkpoints = persistence.list_checkpoints()
        assert len(checkpoints) >= 2

    @pytest.mark.asyncio
    async def test_periodic_checkpoint_async_callback(self, persistence: StatePersistence) -> None:
        """Should work with async state callback."""

        async def get_state_async() -> StateCheckpoint:
            await asyncio.sleep(0.01)
            return StateCheckpoint(
                trader_id="PAPER-PERIODIC",
                balance={"async": True},
            )

        await persistence.start_periodic_checkpointing(get_state_async)
        try:
            assert await _wait_until(lambda: len(persistence.list_checkpoints()) >= 1)
        finally:
            await persistence.stop_periodic_checkpointing()

        checkpoints = persistence.list_checkpoints()
        assert len(checkpoints) >= 1
        assert checkpoints[0].balance["async"] is True

    @pytest.mark.asyncio
    async def test_stop_periodic_checkpointing(self, persistence: StatePersistence) -> None:
        """Should stop checkpointing when requested."""
        call_count = 0

        def get_state() -> StateCheckpoint:
            nonlocal call_count
            call_count += 1
            return StateCheckpoint(trader_id="PAPER-PERIODIC")

        await persistence.start_periodic_checkpointing(get_state)
        assert await _wait_until(lambda: call_count >= 2)

        count_at_stop = call_count
        await persistence.stop_periodic_checkpointing()

        # Several more intervals' worth of time must pass with no callbacks.
        await asyncio.sleep(0.3)
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
