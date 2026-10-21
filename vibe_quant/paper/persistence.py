"""State persistence for paper trading.

Save/restore positions, orders, balance to SQLite with periodic checkpointing.
Recovery loads most recent checkpoint for reconciliation with exchange.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from vibe_quant.db.connection import get_connection
from vibe_quant.paper.config import account_env, is_key_rotation

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Callable
    from pathlib import Path

logger = logging.getLogger(__name__)


# Type alias for JSON-serializable dict
JsonDict = dict[str, Any]


class StoreWorker:
    """One worker thread that owns one store connection (R19d).

    Every call into the store goes through ``submit`` / ``run`` (FIFO, one at a
    time), so a hung statement stalls this worker only and never the trading event
    loop. ``begin_stop`` queues the store's ``close`` as the LAST job and shuts the
    executor down without joining: the worker closes its own connection after its
    last job (eventual close). Nothing closes a connection from another thread.
    """

    def __init__(self, thread_name: str) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name)
        self._close_future: Future[None] | None = None

    @property
    def executor(self) -> ThreadPoolExecutor:
        return self._executor

    def submit(self, fn: Callable[..., Any], *args: Any) -> Future[Any]:
        """Queue ``fn(*args)`` behind every earlier job (capture order = write order)."""
        return self._executor.submit(fn, *args)

    async def run(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Await ``fn(*args)`` on the worker without blocking the event loop."""
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    def begin_stop(self, close: Callable[[], None]) -> Future[None]:
        """Queue ``close`` as the last job and stop accepting work; never blocks. Idempotent."""
        if self._close_future is None:
            self._close_future = self._executor.submit(close)
            self._executor.shutdown(wait=False)
        return self._close_future

    def stop(self, close: Callable[[], None], timeout_s: float) -> bool:
        """``begin_stop`` and wait at most ``timeout_s``; True when the worker closed the store.

        On a timeout the worker is still busy (a hung statement): it closes the
        connection after the job returns.
        """
        fut = self.begin_stop(close)
        deadline = time.monotonic() + timeout_s
        while not fut.done():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return fut.exception() is None


async def wait_future(fut: Future[Any], timeout_s: float) -> bool:
    """Poll a worker future without blocking the loop; True when it finished in time.

    Polling (not ``wrap_future``) means a job finishing after the loop is gone never
    calls back into a closed loop.
    """
    deadline = time.monotonic() + timeout_s
    while not fut.done():
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(0.02)
    return True


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

#: ae6ah: the trader's rows in one venue + env: any key of that env, a legacy row (no /
#: null account), an unreadable row, or a non-string account (the last two fail closed
#: when parsed or checked). One predicate for the consent gate and the restore, so
#: "the gate asks for consent" and "nothing carries" cannot drift apart. Parameters:
#: ``(trader_id, account_env(account) + ":", account_env(account) + ":")``.
_ENV_ROW_WHERE = """trader_id = ?
  AND CASE WHEN json_valid(node_status)
           THEN CASE json_type(node_status, '$.account')
                    WHEN 'text' THEN substr(json_extract(node_status, '$.account'), 1, length(?)) = ?
                    ELSE 1 END
           ELSE 1 END"""


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
        checkpoint_interval: float = 60,
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
        """Get or create database connection.

        NOT thread-safe: the lazy ``_conn`` is created and returned without a
        lock, so one instance belongs to one thread / asyncio loop. Before
        sharing an instance across threads, add locking (see
        ``StateManager._write_lock``).
        """
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
        """Close database connection and stop periodic checkpointing.

        Note: For async contexts, prefer calling stop_periodic_checkpointing()
        first to ensure clean task cancellation before closing.
        """
        self._running = False
        if self._checkpoint_task is not None:
            self._checkpoint_task.cancel()
            # Cannot await in sync method — caller should use stop_periodic_checkpointing() first
            self._checkpoint_task = None
        if self._conn is not None:
            # Commit any pending writes before closing
            with contextlib.suppress(Exception):
                self._conn.commit()
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

    def load_latest_env_checkpoint(
        self, trader_id: str, account: str
    ) -> StateCheckpoint | None:
        """Newest checkpoint of ``trader_id`` in ``account``'s venue + env (any key).

        Ae6ah (D1=A): rotating back A -> B -> A restores the newest row of the env,
        so an older key's own row is never hidden by a row of the other env. Parsed
        strictly (an unreadable row raises, the restore fails closed).
        """
        prefix = account_env(account) + ":"
        cursor = self.conn.execute(
            f"""SELECT * FROM paper_trading_checkpoints
               WHERE {_ENV_ROW_WHERE}
               ORDER BY id DESC
               LIMIT 1""",
            (trader_id, prefix, prefix),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_checkpoint(row)

    def load_latest_account_checkpoint(
        self, trader_id: str, account: str
    ) -> StateCheckpoint | None:
        """Newest checkpoint of ``trader_id`` that belongs to ``account``.

        A row belongs to ``account`` when its ``account`` key equals it or is missing
        (legacy); rows of another account (testnet vs live, another key) are skipped.
        A row whose ``node_status`` is not valid JSON counts (it fails closed when
        parsed). Same predicate as the own-id threshold of
        :meth:`load_newer_account_checkpoint`.
        """
        cursor = self.conn.execute(
            """SELECT * FROM paper_trading_checkpoints
               WHERE trader_id = ?
                 AND CASE WHEN json_valid(node_status)
                          THEN COALESCE(json_extract(node_status, '$.account'), ?)
                          ELSE ? END = ?
               ORDER BY id DESC
               LIMIT 1""",
            (trader_id, account, account, account),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_checkpoint(row)

    def load_newer_account_checkpoint(
        self, account: str, trader_id: str
    ) -> StateCheckpoint | None:
        """Newest checkpoint of ANOTHER trader on ``account`` newer than ``trader_id``'s own.

        "Newer" is a higher row id than ``trader_id``'s newest row ON THIS
        ``account`` (0 when it has none; its rows without an ``account`` key count as
        this account, rows of another account do not). Rows whose ``node_status`` is not valid JSON are skipped (their
        account cannot be known; ``json_valid`` makes that independent of scan
        order). The CHOSEN row is parsed strictly: a parse failure raises, so the
        caller fails closed.

        Returns:
            The checkpoint, or None if no other trader on the account is newer.
        """
        cursor = self.conn.execute(
            """SELECT * FROM paper_trading_checkpoints
               WHERE trader_id != ?
                 AND json_valid(node_status)
                 AND json_extract(node_status, '$.account') = ?
                 AND id > (SELECT COALESCE(MAX(id), 0) FROM paper_trading_checkpoints
                           WHERE trader_id = ?
                             AND CASE WHEN json_valid(node_status)
                                      THEN COALESCE(json_extract(node_status, '$.account'), ?)
                                      ELSE ? END = ?)
               ORDER BY id DESC
               LIMIT 1""",
            (trader_id, account, trader_id, account, account, account),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_checkpoint(row)

    def has_account_checkpoint(self, account: str, trader_id: str) -> bool:
        """True if a LIVE start on ``account`` has saved risk state to carry.

        That is any trader's row carrying the account key, this trader's legacy row
        without the key, or a KEY ROTATION: this trader's newest row in the same venue
        + env (any key) is another key, whose risk and halt the restore carries
        (ae6ah, qa42q). The rotation read is the row ``load_latest_env_checkpoint``
        returns and the predicate is :func:`is_key_rotation`, the restore's own rule,
        so "the gate asks for consent" and "nothing carries" cannot drift apart
        (vibe-quant-neuvu.2). Rows of another trader whose ``node_status`` is not
        valid JSON are ignored: their account cannot be known, so counting them
        would let an unrelated row satisfy the gate. Never raises for row content.
        """
        cursor = self.conn.execute(
            """SELECT 1 FROM paper_trading_checkpoints
               WHERE (json_valid(node_status)
                      AND json_extract(node_status, '$.account') = ?)
                  OR (trader_id = ?
                      AND CASE WHEN json_valid(node_status)
                               THEN json_extract(node_status, '$.account') IS NULL
                               ELSE 0 END)
               LIMIT 1""",
            (account, trader_id),
        )
        if cursor.fetchone() is not None:
            return True
        prefix = account_env(account) + ":"
        newest = self.conn.execute(
            f"""SELECT CASE WHEN json_valid(node_status)
                           THEN json_extract(node_status, '$.account') END AS account
               FROM paper_trading_checkpoints
               WHERE {_ENV_ROW_WHERE}
               ORDER BY id DESC
               LIMIT 1""",
            (trader_id, prefix, prefix),
        ).fetchone()
        if newest is None:
            return False
        newest_account = newest["account"]
        return isinstance(newest_account, str) and is_key_rotation(newest_account, account)

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
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        """Start periodic checkpointing background task.

        Args:
            state_callback: Callable that returns current StateCheckpoint.
                Can be sync or async.
            executor: Single-worker executor that owns every write (R19). The
                capture runs on the loop, the save + prune on the executor, so
                capture order = write order.
        """
        self._running = True
        self._checkpoint_task = asyncio.create_task(self._checkpoint_loop(state_callback, executor))

    async def stop_periodic_checkpointing(self) -> None:
        """Stop periodic checkpointing."""
        self._running = False
        if self._checkpoint_task is not None:
            self._checkpoint_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._checkpoint_task
            self._checkpoint_task = None

    def _save_and_prune(self, checkpoint: StateCheckpoint) -> None:
        self.save_checkpoint(checkpoint)
        self.delete_old_checkpoints()

    async def _checkpoint_loop(
        self,
        state_callback: Callable[[], StateCheckpoint | None],
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        """Periodic checkpoint loop.

        Args:
            state_callback: Callable returning StateCheckpoint.
            executor: Runs save + prune off the loop (None: inline, as before).
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
                    if executor is None:
                        self._save_and_prune(checkpoint)
                    else:
                        # Submitted in the same loop step as the capture: no await in between.
                        await asyncio.get_running_loop().run_in_executor(
                            executor, self._save_and_prune, checkpoint
                        )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                # jf7c3 J4: class only. The text and the active exception's ``__context__``
                # chain can carry a fetched signed URL / api key, and ``logger.exception``
                # would print both plus the traceback.
                logger.error("checkpoint loop error (%s)", type(exc).__name__)


# ---------------------------------------------------------------------------
# Operator command queue (API -> running paper node)
# ---------------------------------------------------------------------------

COMMANDS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS paper_commands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trader_id TEXT NOT NULL,
    command TEXT NOT NULL,
    payload JSON NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    result JSON,
    error TEXT,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_paper_commands_pending
    ON paper_commands(trader_id, status);
"""

#: Commands a paper node understands.
PAPER_COMMANDS = frozenset({"halt", "pause", "resume", "close_all", "kill"})

#: Terminal command states.
COMMAND_DONE = "done"
COMMAND_FAILED = "failed"
COMMAND_EXPIRED = "expired"
_TERMINAL = frozenset({COMMAND_DONE, COMMAND_FAILED, COMMAND_EXPIRED})


@dataclass
class PaperCommand:
    """One operator command and its outcome."""

    id: int
    trader_id: str
    command: str
    payload: JsonDict
    status: str
    result: JsonDict | None
    error: str | None

    @property
    def finished(self) -> bool:
        return self.status in _TERMINAL


class PaperCommandQueue:
    """SQLite-backed command queue between the API and a paper node process.

    Replaces signal-based control (SIGUSR1/SIGWINCH): commands carry a payload,
    the node acknowledges with a result or an error, and the API can return that
    outcome to the caller instead of a fire-and-forget "signal sent".
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        """Get or create the command-queue connection.

        NOT thread-safe: the lazy ``_conn`` is created and returned without a
        lock, so one instance belongs to one thread / asyncio loop. Before
        sharing an instance across threads, add locking (see
        ``StateManager._write_lock``).
        """
        if self._conn is None:
            self._conn = get_connection(self._db_path)
            self._conn.executescript(COMMANDS_TABLE_SQL)
            self._conn.commit()
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def enqueue(self, trader_id: str, command: str, payload: JsonDict | None = None) -> int:
        if command not in PAPER_COMMANDS:
            raise ValueError(f"unknown paper command {command!r}")
        cursor = self.conn.execute(
            "INSERT INTO paper_commands (trader_id, command, payload) VALUES (?, ?, ?)",
            (trader_id, command, json.dumps(payload or {})),
        )
        self.conn.commit()
        return cursor.lastrowid or 0

    #: k8ygz: commands applied before NautilusTrader starts (only a leading run of these).
    STARTUP_COMMANDS = frozenset({"halt", "kill"})

    def claim_startup_halts(self, trader_id: str) -> list[PaperCommand]:
        """Claim this trader's pending halts/kills that no pending ``resume`` precedes.

        Rows are taken in id order; a pending ``resume`` stops the scan, and every row
        before it that is not halt/kill stays pending for the control loop. Only this
        trader's rows are touched; claimed rows move to ``running``. [] when none.
        """
        rows = self.conn.execute(
            "SELECT * FROM paper_commands WHERE trader_id = ? AND status = 'pending' ORDER BY id",
            (trader_id,),
        ).fetchall()
        claimed: list[PaperCommand] = []
        for row in rows:
            if row["command"] == "resume":
                break
            if row["command"] not in self.STARTUP_COMMANDS:
                continue
            cur = self.conn.execute(
                "UPDATE paper_commands SET status = 'running', "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
                "WHERE id = ? AND status = 'pending'",
                (row["id"],),
            )
            if cur.rowcount == 1:
                cmd = self._row_to_command(row)
                cmd.status = "running"
                claimed.append(cmd)
        self.conn.commit()
        return claimed

    def claim_pending(self, trader_id: str) -> list[PaperCommand]:
        """Atomically move this trader's pending commands to ``running``."""
        rows = self.conn.execute(
            "SELECT * FROM paper_commands WHERE trader_id = ? AND status = 'pending' ORDER BY id",
            (trader_id,),
        ).fetchall()
        claimed: list[PaperCommand] = []
        for row in rows:
            cur = self.conn.execute(
                "UPDATE paper_commands SET status = 'running', "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
                "WHERE id = ? AND status = 'pending'",
                (row["id"],),
            )
            if cur.rowcount == 1:
                cmd = self._row_to_command(row)
                cmd.status = "running"
                claimed.append(cmd)
        self.conn.commit()
        return claimed

    def fail_stale_running(self, trader_id: str, message: str) -> int:
        """Fail every row this trader left ``running`` (a previous process claimed it).

        A claimed row is finished only by the process that claimed it, so a row still
        ``running`` when a node starts can never reach an outcome and the API polls it
        forever. Other traders' rows and ``pending`` rows are untouched; returns the
        number of rows failed.
        """
        cur = self.conn.execute(
            "UPDATE paper_commands SET status = ?, error = ?, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
            "WHERE trader_id = ? AND status = 'running'",
            (COMMAND_FAILED, message, trader_id),
        )
        self.conn.commit()
        return cur.rowcount

    def complete(self, command_id: int, result: JsonDict) -> None:
        self._finish(command_id, COMMAND_DONE, result, None)

    def fail(self, command_id: int, error: str, result: JsonDict | None = None) -> None:
        self._finish(command_id, COMMAND_FAILED, result, error)

    def expire(self, command_id: int) -> bool:
        """Expire a command nobody claimed yet; False if the node already took it."""
        cur = self.conn.execute(
            "UPDATE paper_commands SET status = ?, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
            "WHERE id = ? AND status = 'pending'",
            (COMMAND_EXPIRED, command_id),
        )
        self.conn.commit()
        return cur.rowcount == 1

    def get(self, command_id: int) -> PaperCommand | None:
        row = self.conn.execute(
            "SELECT * FROM paper_commands WHERE id = ?", (command_id,)
        ).fetchone()
        return None if row is None else self._row_to_command(row)

    async def wait_for(
        self, command_id: int, timeout: float, poll_interval: float = 0.2
    ) -> PaperCommand | None:
        """Poll (non-blocking) until the command finishes; None on timeout."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            cmd = self.get(command_id)
            if cmd is not None and cmd.finished:
                return cmd
            if loop.time() >= deadline:
                return None
            await asyncio.sleep(poll_interval)

    def _finish(
        self, command_id: int, status: str, result: JsonDict | None, error: str | None
    ) -> None:
        self.conn.execute(
            "UPDATE paper_commands SET status = ?, result = ?, error = ?, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
            (status, json.dumps(result) if result is not None else None, error, command_id),
        )
        self.conn.commit()

    @staticmethod
    def _row_to_command(row: sqlite3.Row) -> PaperCommand:
        result_raw = row["result"]
        return PaperCommand(
            id=row["id"],
            trader_id=row["trader_id"],
            command=row["command"],
            payload=json.loads(row["payload"] or "{}"),
            status=row["status"],
            result=json.loads(result_raw) if result_raw else None,
            error=row["error"],
        )


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
