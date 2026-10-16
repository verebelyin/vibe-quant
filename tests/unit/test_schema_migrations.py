"""``_migrate_add_columns`` skips present columns and lets a real error escape.

bead vibe-quant-ovsok: the helper used to wrap every ``ALTER`` in
``except Exception: pass``, so a concurrent writer's SQLITE_BUSY left the
database half-migrated with no signal. It now reads each table's columns once
with ``PRAGMA table_info``, skips present columns and absent tables without
executing an ALTER, and lets an error from a needed ALTER propagate to the
caller -- ``init_schema`` and ``OverfittingPipeline.conn`` included.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.db.connection import get_connection
from vibe_quant.db.schema import SCHEMA_SQL, _migrate_add_columns, init_schema
from vibe_quant.db.state_manager import StateManager
from vibe_quant.overfitting.pipeline import OverfittingPipeline

if TYPE_CHECKING:
    from pathlib import Path

#: A column that a fresh SCHEMA_SQL carries; dropping it makes an "old" DB.
_DROPPED_COLUMN = "mtm_days"


def _columns(conn: object, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}  # type: ignore[attr-defined]


def _old_schema_db(tmp_path: Path) -> Path:
    """A full-schema DB rewound to before the ``sweep_results`` migration."""
    db = tmp_path / "old.db"
    conn = get_connection(db)
    conn.executescript(SCHEMA_SQL)
    conn.execute(f"ALTER TABLE sweep_results DROP COLUMN {_DROPPED_COLUMN}")
    conn.commit()
    assert _DROPPED_COLUMN not in _columns(conn, "sweep_results")
    conn.close()
    return db


class _AlterRejectingConnection:
    """Connection proxy that records (and rejects) any ``ALTER`` statement."""

    def __init__(self, real: sqlite3.Connection) -> None:
        self._real = real
        self.altered: list[str] = []

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
        if sql.lstrip().upper().startswith("ALTER"):
            self.altered.append(sql)
            raise AssertionError(f"ALTER executed on a migrated DB: {sql}")
        return self._real.execute(sql, *args, **kwargs)


class _AlterLockedConnection:
    """Connection proxy forwarding everything but raising SQLITE_BUSY on ALTER."""

    def __init__(self, real: sqlite3.Connection) -> None:
        self._real = real

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
        if sql.lstrip().upper().startswith("ALTER"):
            raise sqlite3.OperationalError("database is locked")
        return self._real.execute(sql, *args, **kwargs)

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)


class _AlterFailingConnection:
    """Connection proxy that raises ``message`` on the ALTER of one column.

    Models a racing process: the column was added between our ``PRAGMA
    table_info`` and our ``ALTER``, so the ALTER fails (M6) -- or, with a
    different message, any other OperationalError (M4).
    """

    def __init__(self, real: sqlite3.Connection, column: str, message: str) -> None:
        self._real = real
        self._column = column
        self._message = message

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
        if sql.lstrip().upper().startswith("ALTER") and self._column in sql:
            raise sqlite3.OperationalError(self._message)
        return self._real.execute(sql, *args, **kwargs)

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)


def _old_schema_db_missing(tmp_path: Path, columns: list[str]) -> Path:
    """A full-schema DB with ``columns`` of ``sweep_results`` dropped."""
    db = tmp_path / "old.db"
    conn = get_connection(db)
    conn.executescript(SCHEMA_SQL)
    for column in columns:
        conn.execute(f"ALTER TABLE sweep_results DROP COLUMN {column}")
    conn.commit()
    assert not set(columns) & _columns(conn, "sweep_results")
    conn.close()
    return db


@pytest.mark.parametrize(
    "template",
    [
        "duplicate column name: {col}",
        "Duplicate Column Name: {col}",
        "DUPLICATE COLUMN NAME: {col}",
    ],
)
def test_duplicate_column_from_a_racing_process_is_treated_as_present(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, template: str
) -> None:
    """M6: a column another process added first -> the losing ALTER's
    ``duplicate column name`` is skipped, not counted, not logged, not raised;
    the migrations after it still run."""
    db = _old_schema_db_missing(tmp_path, [_DROPPED_COLUMN, "exit_timing"])
    conn = get_connection(db)
    proxy = _AlterFailingConnection(conn, _DROPPED_COLUMN, template.format(col=_DROPPED_COLUMN))
    with caplog.at_level(logging.INFO, logger="vibe_quant.db.schema"):
        _migrate_add_columns(proxy)  # type: ignore[arg-type]
    conn.commit()
    columns = _columns(conn, "sweep_results")
    conn.close()

    assert "exit_timing" in columns, "a later needed ALTER must still be applied"
    assert _DROPPED_COLUMN not in columns  # the winner owns it, we did not add it
    messages = [record.getMessage() for record in caplog.records]
    applied_logs = [message for message in messages if message.startswith("Applied migration:")]
    assert not any(_DROPPED_COLUMN in message for message in applied_logs)  # not logged
    assert any("sweep_results.exit_timing" in message for message in applied_logs)
    # Every applied column logs once, so the summary count must match: the raced
    # column is skipped, NOT counted as an applied migration.
    assert any(
        message.startswith(f"Applied {len(applied_logs)} schema migration(s)")
        for message in messages
    )


def test_only_the_duplicate_error_is_forgiven(tmp_path: Path) -> None:
    """M6: every other OperationalError from the ALTER still propagates."""
    db = _old_schema_db_missing(tmp_path, [_DROPPED_COLUMN])
    conn = get_connection(db)
    proxy = _AlterFailingConnection(conn, _DROPPED_COLUMN, "database is locked")
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        _migrate_add_columns(proxy)  # type: ignore[arg-type]
    conn.commit()
    assert _DROPPED_COLUMN not in _columns(conn, "sweep_results")
    conn.close()


def test_present_columns_are_skipped_without_alter(tmp_path: Path) -> None:
    """M1: a fully migrated DB runs no ALTER at all."""
    conn = get_connection(tmp_path / "migrated.db")
    init_schema(conn)
    proxy = _AlterRejectingConnection(conn)
    _migrate_add_columns(proxy)  # type: ignore[arg-type]

    conn.close()
    assert proxy.altered == []


def test_absent_tables_are_skipped_without_error() -> None:
    """M2: a DB with only ``sweep_results`` migrates it and ignores the rest."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE sweep_results (id INTEGER PRIMARY KEY)")
    assert _columns(conn, "backtest_results") == set()

    _migrate_add_columns(conn)

    sweep_columns = _columns(conn, "sweep_results")
    conn.close()
    assert {_DROPPED_COLUMN, "exit_timing", "starting_balance"} <= sweep_columns


def test_missing_column_is_added_and_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """M3: a column missing from an existing table is added and logged."""
    db = _old_schema_db(tmp_path)
    conn = get_connection(db)
    with caplog.at_level(logging.INFO, logger="vibe_quant.db.schema"):
        _migrate_add_columns(conn)
    conn.commit()
    added = _columns(conn, "sweep_results")
    conn.close()

    assert _DROPPED_COLUMN in added
    assert any(f"sweep_results.{_DROPPED_COLUMN}" in r.getMessage() for r in caplog.records)


def test_held_writer_lock_propagates_operational_error(tmp_path: Path) -> None:
    """M4: with a REAL writer lock held, the needed ALTER raises SQLITE_BUSY."""
    db = _old_schema_db(tmp_path)
    holder = sqlite3.connect(str(db))
    holder.execute("PRAGMA journal_mode=WAL")
    holder.execute("BEGIN IMMEDIATE")  # RESERVED write lock, held until rollback
    migrating = sqlite3.connect(str(db), timeout=5.0)
    migrating.execute("PRAGMA busy_timeout=50")  # don't wait the 5 s default
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            _migrate_add_columns(migrating)
    finally:
        migrating.close()
        holder.rollback()
        holder.close()

    check = get_connection(db)
    still_missing = _DROPPED_COLUMN not in _columns(check, "sweep_results")
    check.close()
    assert still_missing  # the failed ALTER left the DB untouched, not half-migrated


def test_init_schema_propagates_the_failed_alter(tmp_path: Path) -> None:
    """M4: an ALTER error surfaces from ``init_schema`` too."""
    db = _old_schema_db(tmp_path)
    conn = get_connection(db)
    proxy = _AlterLockedConnection(conn)
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        init_schema(proxy)  # type: ignore[arg-type]
    conn.close()


def test_pipeline_conn_fails_closed_on_a_held_lock_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M5: the lock error escapes ``OverfittingPipeline.conn``; ``_conn`` stays None
    and the next call retries once the lock is released."""
    db = _old_schema_db(tmp_path)
    real = _migrate_add_columns

    def with_short_timeout(conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA busy_timeout=50")
        real(conn)

    monkeypatch.setattr(
        "vibe_quant.overfitting.pipeline._migrate_add_columns", with_short_timeout
    )

    holder = sqlite3.connect(str(db))
    holder.execute("PRAGMA journal_mode=WAL")
    holder.execute("BEGIN IMMEDIATE")
    pipeline = OverfittingPipeline(db)
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            pipeline.conn  # noqa: B018
        assert pipeline._conn is None
    finally:
        holder.rollback()
        holder.close()

    pipeline.conn  # noqa: B018  # lock gone -> the retry migrates and connects
    assert pipeline._conn is not None
    assert _DROPPED_COLUMN in _columns(pipeline._conn, "sweep_results")
    pipeline.close()


def test_absent_tables_log_one_debug_record_each(caplog: pytest.LogCaptureFixture) -> None:
    """M2b: an empty DB logs exactly one DEBUG record per absent migrated table."""
    from vibe_quant.db.schema import _migrate_add_columns as helper

    conn = sqlite3.connect(":memory:")
    with caplog.at_level(logging.DEBUG, logger="vibe_quant.db.schema"):
        helper(conn)
    conn.close()

    skipped = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.DEBUG and "absent table" in r.getMessage()
    ]
    expected = {
        "sweep_results",
        "backtest_results",
        "research_extractions",
        "research_extraction_jobs",
        "background_jobs",
    }
    assert len(skipped) == len(set(skipped)), "a table was logged more than once"
    for table in expected:
        assert sum(m.endswith(f": {table}") for m in skipped) == 1, table


def _hold_writer_lock(db: Path) -> sqlite3.Connection:
    holder = sqlite3.connect(str(db))
    holder.execute("PRAGMA journal_mode=WAL")
    holder.execute("BEGIN IMMEDIATE")
    return holder


def test_state_manager_conn_fails_closed_on_a_held_lock_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M7: a locked migration closes the new connection, caches nothing, and the
    next access (lock released) migrates instead of returning a half-migrated conn."""
    db = _old_schema_db(tmp_path)
    opened: list[sqlite3.Connection] = []

    def tracking(db_path: Path | None = None) -> sqlite3.Connection:
        conn = get_connection(db_path)
        conn.execute("PRAGMA busy_timeout=50")
        opened.append(conn)
        return conn

    monkeypatch.setattr("vibe_quant.db.state_manager.get_connection", tracking)
    holder = _hold_writer_lock(db)
    sm = StateManager(db)
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            sm.conn  # noqa: B018
        assert sm._conn is None
        with pytest.raises(sqlite3.ProgrammingError):
            opened[0].execute("SELECT 1")  # the failed connection was closed
    finally:
        holder.rollback()
        holder.close()

    conn = sm.conn
    assert _DROPPED_COLUMN in _columns(conn, "sweep_results")
    sm.close()
