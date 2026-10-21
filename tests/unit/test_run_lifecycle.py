"""Direct characterization of ``vibe_quant/db/run_lifecycle.py`` (rabmo.12 / T17).

The transition table itself is pinned through the public APIs in
``test_run_lifecycle_table.py``. These tests pin the moved module's own contract:
the re-exported ``JobStatus``, the exact refusal texts, the commit points, and the
status guard on the orphan-reap UPDATE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.db.connection import get_connection
from vibe_quant.db.run_lifecycle import JobStatus
from vibe_quant.db.schema import init_schema

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterator
    from pathlib import Path


# --- fixtures + row helpers --------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "run_lifecycle.db"


@pytest.fixture
def conn(db_path: Path) -> Iterator[sqlite3.Connection]:
    connection = get_connection(db_path)
    init_schema(connection)
    try:
        yield connection
    finally:
        connection.close()


def _run(
    conn: sqlite3.Connection,
    *,
    status: str = "running",
    error_message: str | None = None,
    pid: int | None = None,
) -> int:
    cursor = conn.execute(
        """INSERT INTO backtest_runs
           (run_mode, symbols, timeframe, start_date, end_date, parameters,
            status, error_message, pid)
           VALUES ('screening', '["BTCUSDT"]', '4h', '2024-01-01', '2025-01-01',
                   '{}', ?, ?, ?)""",
        (status, error_message, pid),
    )
    conn.commit()
    assert cursor.lastrowid is not None
    return int(cursor.lastrowid)


def _job(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    status: str = "running",
    pid: int = 4242,
    pid_start_time: str | None = "linux:1",
    error_message: str | None = None,
) -> None:
    conn.execute(
        """INSERT INTO background_jobs
           (run_id, pid, pid_start_time, job_type, status, error_message)
           VALUES (?, ?, ?, 'screening', ?, ?)""",
        (run_id, pid, pid_start_time, status, error_message),
    )
    conn.commit()


def _run_row(conn: sqlite3.Connection, run_id: int) -> dict[str, Any]:
    row = conn.execute(
        """SELECT status, pid, completed_at, error_message FROM backtest_runs
           WHERE id = ?""",
        (run_id,),
    ).fetchone()
    assert row is not None
    return dict(row)


def _job_row(conn: sqlite3.Connection, run_id: int) -> dict[str, Any]:
    row = conn.execute(
        """SELECT status, pid, pid_start_time, completed_at, error_message
           FROM background_jobs WHERE run_id = ?""",
        (run_id,),
    ).fetchone()
    assert row is not None
    return dict(row)


# --- JobStatus ownership -----------------------------------------------------


def test_job_status_is_reexported_identically() -> None:
    """``JobStatus`` lives in run_lifecycle; both old import paths keep working."""
    from vibe_quant.db import run_lifecycle
    from vibe_quant.jobs import JobStatus as package_status
    from vibe_quant.jobs.manager import JobStatus as manager_status

    assert manager_status is JobStatus
    assert package_status is JobStatus
    assert run_lifecycle.JobStatus is JobStatus
    assert [member.value for member in JobStatus] == [
        "pending",
        "running",
        "completed",
        "failed",
        "killed",
    ]


# --- finish ------------------------------------------------------------------


def test_finish_completed_adopts_the_runs_failure(conn: sqlite3.Connection) -> None:
    """R47/R48: a plain completed never overwrites a failed run; it adopts it."""
    run_id = _run(conn, status="failed", error_message="0 trades")
    _job(conn, run_id, status="running")

    from vibe_quant.db.run_lifecycle import finish

    assert finish(conn, run_id, JobStatus.COMPLETED, None, reconcile=False, owner=None) is True

    run = _run_row(conn, run_id)
    job = _job_row(conn, run_id)
    assert (run["status"], run["error_message"]) == ("failed", "0 trades")
    assert (job["status"], job["error_message"]) == ("failed", "0 trades")
    assert run["completed_at"] is not None
    assert job["completed_at"] is not None


def test_finish_failed_after_kill_keeps_the_kill_and_its_error(
    conn: sqlite3.Connection,
) -> None:
    """R47: failed after killed stays killed and keeps the run's error text."""
    run_id = _run(conn, status="killed", error_message="user kill")
    _job(conn, run_id, status="killed")

    from vibe_quant.db.run_lifecycle import finish

    assert finish(conn, run_id, JobStatus.FAILED, "boom", reconcile=False, owner=None) is True

    run = _run_row(conn, run_id)
    job = _job_row(conn, run_id)
    assert (run["status"], run["error_message"]) == ("killed", "user kill")
    assert (job["status"], job["error_message"]) == ("killed", None)


def test_finish_reconcile_on_non_running_job_writes_nothing(conn: sqlite3.Connection) -> None:
    """R47: reconcile only acts while the job row is still running."""
    run_id = _run(conn, status="running")
    _job(conn, run_id, status="completed")

    from vibe_quant.db.run_lifecycle import finish

    assert finish(conn, run_id, JobStatus.FAILED, "x", reconcile=True, owner=None) is False
    assert _run_row(conn, run_id)["status"] == "running"
    assert _run_row(conn, run_id)["completed_at"] is None
    assert _job_row(conn, run_id)["status"] == "completed"


def test_finish_ignores_a_row_another_attempt_owns(conn: sqlite3.Connection) -> None:
    """qgtdz: a re-launched row (different pid) is not written by the old process."""
    run_id = _run(conn, status="running")
    _job(conn, run_id, pid=4242)

    from vibe_quant.db.run_lifecycle import finish

    assert finish(conn, run_id, JobStatus.COMPLETED, None, reconcile=False, owner=999) is False
    assert _run_row(conn, run_id)["status"] == "running"
    assert _job_row(conn, run_id)["status"] == "running"


def test_finish_commits_before_returning(conn: sqlite3.Connection, db_path: Path) -> None:
    """The moved body still commits: another connection sees the terminal rows."""
    run_id = _run(conn, status="running")
    _job(conn, run_id)

    from vibe_quant.db.run_lifecycle import finish

    assert finish(conn, run_id, JobStatus.COMPLETED, None, reconcile=False, owner=None) is True

    other = get_connection(db_path)
    try:
        run = other.execute(
            "SELECT status FROM backtest_runs WHERE id = ?", (run_id,)
        ).fetchone()
        job = other.execute(
            "SELECT status FROM background_jobs WHERE run_id = ?", (run_id,)
        ).fetchone()
    finally:
        other.close()
    assert run["status"] == "completed"
    assert job["status"] == "completed"


# --- claim -------------------------------------------------------------------


def test_claim_updates_both_rows_and_returns_none(
    conn: sqlite3.Connection, db_path: Path
) -> None:
    """A successful claim records pid + identity on both rows, committed."""
    run_id = _run(conn, status="pending")
    _job(conn, run_id, status="pending", pid=0, pid_start_time=None)

    from vibe_quant.db.run_lifecycle import claim

    assert claim(conn, run_id, 5150, "linux:123") is None

    run = _run_row(conn, run_id)
    job = _job_row(conn, run_id)
    assert (run["status"], run["pid"]) == ("pending", 5150)
    assert (job["status"], job["pid"], job["pid_start_time"]) == ("running", 5150, "linux:123")

    other = get_connection(db_path)
    try:
        row = other.execute(
            "SELECT status, pid FROM background_jobs WHERE run_id = ?", (run_id,)
        ).fetchone()
    finally:
        other.close()
    assert (row["status"], row["pid"]) == ("running", 5150)


def test_claim_without_a_job_row_refuses_without_writing(conn: sqlite3.Connection) -> None:
    """R47: no job row -> the no-row refusal, and nothing is written."""
    run_id = _run(conn, status="running")

    from vibe_quant.db.run_lifecycle import claim

    refusal = (
        f"Refusing to trade: run {run_id} has no job or run row to record this process in"
    )
    assert claim(conn, run_id, 5150, "linux:123") == refusal
    assert _run_row(conn, run_id)["status"] == "running"


def test_claim_refuses_another_attempts_pid(conn: sqlite3.Connection) -> None:
    """R47: a live pid of another attempt is never overwritten."""
    run_id = _run(conn, status="running")
    _job(conn, run_id, pid=777, pid_start_time="other-attempt")

    from vibe_quant.db.run_lifecycle import claim

    refusal = (
        f"Refusing to trade: run {run_id} is recorded for pid 777, "
        "not this process (pid 5150)"
    )
    assert claim(conn, run_id, 5150, "linux:123") == refusal
    job = _job_row(conn, run_id)
    assert (job["pid"], job["pid_start_time"]) == (777, "other-attempt")


def test_claim_refuses_a_finished_run(conn: sqlite3.Connection) -> None:
    """R47: a finished run is never reopened."""
    run_id = _run(conn, status="completed")
    _job(conn, run_id, status="running")

    from vibe_quant.db.run_lifecycle import claim

    refusal = (
        f"Refusing to trade: run {run_id} is already completed (job running); "
        "a finished run is never reopened"
    )
    assert claim(conn, run_id, 4242, "linux:1") == refusal
    assert _run_row(conn, run_id)["status"] == "completed"


def test_claim_without_identity_fails_both_rows(conn: sqlite3.Connection) -> None:
    """R48: identity None overwrites both rows with the refusal text."""
    run_id = _run(conn, status="running")
    _job(conn, run_id, status="pending", pid=0, pid_start_time=None)

    from vibe_quant.db.run_lifecycle import claim

    refusal = (
        "Refusing to trade: cannot read this process's start time "
        "(pid 5150), so stop/kill could not verify it"
    )
    assert claim(conn, run_id, 5150, None) == refusal

    run = _run_row(conn, run_id)
    job = _job_row(conn, run_id)
    assert (run["status"], run["error_message"]) == ("failed", refusal)
    assert (job["status"], job["error_message"]) == ("failed", refusal)


# --- fail_orphan_run ---------------------------------------------------------


def test_fail_orphan_run_guards_on_the_current_status(conn: sqlite3.Connection) -> None:
    """Reap's UPDATE only flips a still pending/running run; true iff it wrote."""
    from vibe_quant.db.run_lifecycle import fail_orphan_run

    reaped = _run(conn, status="pending")
    finished = _run(conn, status="completed", error_message="done")

    reason = "reaped at startup: no live job (was pending)"
    assert fail_orphan_run(conn, reaped, reason) is True
    assert fail_orphan_run(conn, finished, "reaped at startup: no live job (was completed)") is False

    row = _run_row(conn, reaped)
    assert (row["status"], row["error_message"]) == ("failed", reason)
    assert row["completed_at"] is not None
    done = _run_row(conn, finished)
    assert (done["status"], done["error_message"]) == ("completed", "done")


def test_moved_sql_literals_are_byte_identical_to_base(conn: sqlite3.Connection) -> None:
    """The four moved multi-line UPDATEs execute exactly the base SQL bytes.

    Expected strings are the base (manager.py at 0eff3381) literal values with
    parameters expanded the way sqlite3's trace callback renders them, so any
    re-indentation of the source literal changes the compared bytes.
    """
    from vibe_quant.db.run_lifecycle import claim, fail_orphan_run, finish

    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    try:
        finished = _run(conn, status="running")
        _job(conn, finished)
        assert finish(conn, finished, JobStatus.COMPLETED, None, reconcile=False, owner=None)

        claimed = _run(conn, status="pending")
        _job(conn, claimed, status="pending", pid=0, pid_start_time=None)
        assert claim(conn, claimed, 5150, "linux:123") is None

        orphan = _run(conn, status="running")
        reason = "reaped at startup: no live job (was running)"
        assert fail_orphan_run(conn, orphan, reason) is True
    finally:
        conn.set_trace_callback(None)

    assert (
        "UPDATE background_jobs\n"
        "                       SET status = 'completed', completed_at = datetime('now'),\n"
        "                           error_message = COALESCE(NULL, error_message)\n"
        f"                       WHERE run_id = {finished}"
    ) in statements
    assert (
        "UPDATE backtest_runs\n"
        "                           SET status = 'completed', completed_at = datetime('now'),\n"
        "                               error_message = COALESCE(NULL, error_message)\n"
        f"                           WHERE id = {finished}"
    ) in statements
    assert (
        "UPDATE background_jobs\n"
        "                       SET pid = 5150, pid_start_time = 'linux:123', status = 'running',\n"
        "                           heartbeat_at = datetime('now')\n"
        f"                       WHERE run_id = {claimed}"
    ) in statements
    assert (
        "UPDATE backtest_runs\n"
        "                       SET status = 'failed', completed_at = datetime('now'),\n"
        f"                           error_message = '{reason}'\n"
        f"                       WHERE id = {orphan} AND status IN ('pending', 'running')"
    ) in statements


def test_fail_orphan_run_leaves_the_commit_to_its_caller(
    conn: sqlite3.Connection, db_path: Path
) -> None:
    """The reap UPDATE is not committed per row: the caller commits once."""
    from vibe_quant.db.run_lifecycle import fail_orphan_run

    run_id = _run(conn, status="running")
    assert fail_orphan_run(conn, run_id, "reaped at startup: no live job (was running)") is True

    other = get_connection(db_path)
    try:
        before = other.execute(
            "SELECT status FROM backtest_runs WHERE id = ?", (run_id,)
        ).fetchone()
        assert before["status"] == "running"
    finally:
        other.close()

    conn.commit()
    other = get_connection(db_path)
    try:
        after = other.execute(
            "SELECT status FROM backtest_runs WHERE id = ?", (run_id,)
        ).fetchone()
        assert after["status"] == "failed"
    finally:
        other.close()
