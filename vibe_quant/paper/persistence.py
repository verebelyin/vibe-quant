"""State persistence for paper trading.

Save/restore positions, orders, balance to SQLite with periodic checkpointing.
Recovery loads most recent checkpoint for reconciliation with exchange.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from vibe_quant.db.connection import get_connection

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Callable
    from pathlib import Path

logger = logging.getLogger(__name__)


# Type alias for JSON-serializable dict
JsonDict = dict[str, Any]

# SQL to create paper trading checkpoint table
CHECKPOINT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS paper_trading_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trader_id TEXT NOT NULL,
    positions JSON NOT NULL,
    orders JSON NOT NULL,
    balance JSON NOT NULL,
    node_status JSON NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_trader ON paper_trading_checkpoints(trader_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_created ON paper_trading_checkpoints(created_at);
"""


@dataclass
class StateCheckpoint:
    """Snapshot of paper trading state at a point in time.

    Attributes:
        trader_id: Unique identifier for the trading node.
        positions: Dict of position_id -> position data.
        orders: Dict of order_id -> order data.
        balance: Account balance data (total, available, margin).
        node_status: Node status dict (state, pnl, etc).
        timestamp: When checkpoint was created.
    """

    trader_id: str
    positions: JsonDict = field(default_factory=dict)
    orders: JsonDict = field(default_factory=dict)
    balance: JsonDict = field(default_factory=dict)
    node_status: JsonDict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> JsonDict:
        """Convert to dictionary for storage."""
        return {
            "trader_id": self.trader_id,
            "positions": self.positions,
            "orders": self.orders,
            "balance": self.balance,
            "node_status": self.node_status,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> StateCheckpoint:
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now(UTC)

        return cls(
            trader_id=data["trader_id"],
            positions=data.get("positions", {}),
            orders=data.get("orders", {}),
            balance=data.get("balance", {}),
            node_status=data.get("node_status", {}),
            timestamp=timestamp,
        )


class StatePersistence:
    """Manages paper trading state persistence to SQLite.

    Provides save/load/restore for checkpoints, periodic checkpointing,
    and recovery from most recent checkpoint.

    Example:
        persistence = StatePersistence(db_path, trader_id="PAPER-001")
        persistence.save_checkpoint(checkpoint)
        latest = persistence.load_latest_checkpoint()
        await persistence.start_periodic_checkpointing(get_state_callback)
    """

    def __init__(
        self,
        db_path: Path | None = None,
        trader_id: str = "default",
        checkpoint_interval: int = 60,
    ) -> None:
        """Initialize persistence manager.

        Args:
            db_path: Path to SQLite database.
            trader_id: Identifier for this trading node.
            checkpoint_interval: Seconds between periodic checkpoints.
        """
        self._db_path = db_path
        self._trader_id = trader_id
        self._checkpoint_interval = checkpoint_interval
        self._conn: sqlite3.Connection | None = None
        self._checkpoint_task: asyncio.Task[None] | None = None
        self._running = False

    @property
    def conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = get_connection(self._db_path)
            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        """Initialize checkpoint table if not exists."""
        if self._conn is not None:
            self._conn.executescript(CHECKPOINT_TABLE_SQL)
            self._conn.commit()

    def close(self) -> None:
        """Close database connection and stop periodic checkpointing."""
        self._running = False
        if self._checkpoint_task is not None:
            self._checkpoint_task.cancel()
            self._checkpoint_task = None
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def save_checkpoint(self, checkpoint: StateCheckpoint) -> int:
        """Save checkpoint to database.

        Args:
            checkpoint: StateCheckpoint to save.

        Returns:
            ID of saved checkpoint.
        """
        cursor = self.conn.execute(
            """INSERT INTO paper_trading_checkpoints
               (trader_id, positions, orders, balance, node_status)
               VALUES (?, ?, ?, ?, ?)""",
            (
                checkpoint.trader_id,
                json.dumps(checkpoint.positions),
                json.dumps(checkpoint.orders),
                json.dumps(checkpoint.balance),
                json.dumps(checkpoint.node_status),
            ),
        )
        self.conn.commit()
        return cursor.lastrowid or 0

    def load_checkpoint(self, checkpoint_id: int) -> StateCheckpoint | None:
        """Load checkpoint by ID.

        Args:
            checkpoint_id: ID of checkpoint to load.

        Returns:
            StateCheckpoint or None if not found.
        """
        cursor = self.conn.execute(
            "SELECT * FROM paper_trading_checkpoints WHERE id = ?",
            (checkpoint_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_checkpoint(row)

    def load_latest_checkpoint(self, trader_id: str | None = None) -> StateCheckpoint | None:
        """Load most recent checkpoint for trader.

        Args:
            trader_id: Trader ID to filter by. Defaults to instance trader_id.

        Returns:
            Most recent StateCheckpoint or None if none exist.
        """
        tid = trader_id or self._trader_id
        cursor = self.conn.execute(
            """SELECT * FROM paper_trading_checkpoints
               WHERE trader_id = ?
               ORDER BY id DESC
               LIMIT 1""",
            (tid,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_checkpoint(row)

    def list_checkpoints(
        self, trader_id: str | None = None, limit: int = 100
    ) -> list[StateCheckpoint]:
        """List checkpoints for trader.

        Args:
            trader_id: Trader ID to filter by. Defaults to instance trader_id.
            limit: Maximum number of checkpoints to return.

        Returns:
            List of StateCheckpoints, most recent first.
        """
        tid = trader_id or self._trader_id
        cursor = self.conn.execute(
            """SELECT * FROM paper_trading_checkpoints
               WHERE trader_id = ?
               ORDER BY id DESC
               LIMIT ?""",
            (tid, limit),
        )
        return [self._row_to_checkpoint(row) for row in cursor]

    def delete_old_checkpoints(self, keep_count: int = 100) -> int:
        """Delete old checkpoints, keeping most recent.

        Args:
            keep_count: Number of recent checkpoints to keep per trader.

        Returns:
            Number of deleted checkpoints.
        """
        # Get IDs to keep
        cursor = self.conn.execute(
            """SELECT id FROM paper_trading_checkpoints
               WHERE trader_id = ?
               ORDER BY id DESC
               LIMIT ?""",
            (self._trader_id, keep_count),
        )
        keep_ids = [row["id"] for row in cursor]

        if not keep_ids:
            return 0

        # Delete others
        placeholders = ",".join("?" * len(keep_ids))
        cursor = self.conn.execute(
            f"""DELETE FROM paper_trading_checkpoints
                WHERE trader_id = ? AND id NOT IN ({placeholders})""",
            [self._trader_id, *keep_ids],
        )
        self.conn.commit()
        return cursor.rowcount

    def _row_to_checkpoint(self, row: sqlite3.Row) -> StateCheckpoint:
        """Convert database row to StateCheckpoint."""
        return StateCheckpoint(
            trader_id=row["trader_id"],
            positions=json.loads(row["positions"]),
            orders=json.loads(row["orders"]),
            balance=json.loads(row["balance"]),
            node_status=json.loads(row["node_status"]),
            timestamp=datetime.fromisoformat(row["created_at"]),
        )

    async def start_periodic_checkpointing(
        self,
        state_callback: Callable[[], StateCheckpoint | None],
    ) -> None:
        """Start periodic checkpointing background task.

        Args:
            state_callback: Callable that returns current StateCheckpoint.
                Can be sync or async.
        """
        self._running = True
        self._checkpoint_task = asyncio.create_task(
            self._checkpoint_loop(state_callback)
        )

    async def stop_periodic_checkpointing(self) -> None:
        """Stop periodic checkpointing."""
        self._running = False
        if self._checkpoint_task is not None:
            self._checkpoint_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._checkpoint_task
            self._checkpoint_task = None

    async def _checkpoint_loop(
        self,
        state_callback: Callable[[], StateCheckpoint | None],
    ) -> None:
        """Periodic checkpoint loop.

        Args:
            state_callback: Callable returning StateCheckpoint.
        """
        while self._running:
            try:
                await asyncio.sleep(self._checkpoint_interval)
                if not self._running:
                    break

                # Get current state
                if asyncio.iscoroutinefunction(state_callback):
                    checkpoint = await state_callback()
                else:
                    checkpoint = state_callback()

                if checkpoint is not None:
                    self.save_checkpoint(checkpoint)
                    self.delete_old_checkpoints()

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("checkpoint loop error")


def recover_state(
    db_path: Path | None = None,
    trader_id: str = "default",
) -> StateCheckpoint | None:
    """Recover state from most recent checkpoint.

    Convenience function for crash recovery. Loads the most recent
    checkpoint for the given trader, which can then be reconciled
    with the exchange.

    Args:
        db_path: Path to SQLite database.
        trader_id: Trader ID to recover.

    Returns:
        Most recent StateCheckpoint or None if none exist.
    """
    persistence = StatePersistence(db_path, trader_id)
    try:
        return persistence.load_latest_checkpoint()
    finally:
        persistence.close()
