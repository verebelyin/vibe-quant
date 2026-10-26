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


def _run_full(conn: sqlite3.Connection, run_id: int) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM backtest_runs WHERE id = ?", (run_id,)).fetchone()
    assert row is not None
    return dict(row)


def _job_full(conn: sqlite3.Connection, run_id: int) -> dict[str, Any]:
    row = conn.execute(
        "SELECT * FROM background_jobs WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert row is not None
    return dict(row)


OLD_HEARTBEAT = "2000-01-01 00:00:00"


def _stamp_heartbeats(conn: sqlite3.Connection, run_id: int) -> None:
    conn.execute(
        "UPDATE background_jobs SET heartbeat_at = ? WHERE run_id = ?", (OLD_HEARTBEAT, run_id)
    )
    conn.execute(
        "UPDATE backtest_runs SET heartbeat_at = ? WHERE id = ?", (OLD_HEARTBEAT, run_id)
    )
    conn.commit()


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


# --- open_attempt ------------------------------------------------------------


def test_open_attempt_opens_a_fresh_attempt(conn: sqlite3.Connection) -> None:
    """exists=False inserts a pending job (pid 0) and reopens the run."""
    from vibe_quant.db import run_lifecycle

    screening = _run(conn, status="failed", error_message="boom", pid=111)
    validation = _run(conn, status="completed", error_message="done", pid=222)
    conn.execute(
        """UPDATE backtest_runs SET started_at = '2024-01-01 00:00:00',
           completed_at = '2024-01-02 00:00:00'"""
    )
    conn.commit()

    run_lifecycle.open_attempt(conn, screening, "screening", "/tmp/screening.log", exists=False)
    run_lifecycle.open_attempt(conn, validation, "validation", None, exists=False)

    job = _job_full(conn, screening)
    assert (job["pid"], job["pid_start_time"]) == (0, None)
    assert (job["job_type"], job["status"], job["log_file"]) == (
        "screening",
        "pending",
        "/tmp/screening.log",
    )
    assert job["started_at"] is not None
    assert job["heartbeat_at"] is not None
    assert (job["completed_at"], job["error_message"]) == (None, None)

    run = _run_full(conn, screening)
    assert (run["status"], run["pid"]) == ("running", None)
    assert run["started_at"] is not None
    assert run["heartbeat_at"] is not None
    assert (run["completed_at"], run["error_message"]) == (None, None)

    other_job = _job_full(conn, validation)
    assert (other_job["job_type"], other_job["status"], other_job["log_file"]) == (
        "validation",
        "pending",
        None,
    )
    other_run = _run_full(conn, validation)
    assert (other_run["status"], other_run["pid"]) == ("running", None)


def test_open_attempt_resets_an_existing_failed_attempt(
    conn: sqlite3.Connection, db_path: Path
) -> None:
    """exists=True clears the failure on both rows and resets the job to pending."""
    from vibe_quant.db import run_lifecycle

    run_id = _run(conn, status="failed", error_message="boom", pid=5150)
    _job(conn, run_id, status="failed", pid=5150, error_message="boom")
    conn.execute(
        """UPDATE background_jobs SET started_at = '2024-01-01 00:00:00',
           completed_at = '2024-01-02 00:00:00', log_file = '/tmp/old.log'"""
    )
    conn.execute("UPDATE backtest_runs SET completed_at = '2024-01-02 00:00:00'")
    conn.commit()

    run_lifecycle.open_attempt(conn, run_id, "validation", "/tmp/new.log", exists=True)

    job = _job_full(conn, run_id)
    assert (job["pid"], job["pid_start_time"]) == (0, None)
    assert (job["job_type"], job["status"], job["log_file"]) == (
        "validation",
        "pending",
        "/tmp/new.log",
    )
    assert job["started_at"] != "2024-01-01 00:00:00"
    assert (job["completed_at"], job["error_message"]) == (None, None)

    run = _run_full(conn, run_id)
    assert (run["status"], run["pid"]) == ("running", None)
    assert (run["completed_at"], run["error_message"]) == (None, None)

    other = get_connection(db_path)
    try:
        seen = other.execute(
            "SELECT status FROM background_jobs WHERE run_id = ?", (run_id,)
        ).fetchone()
    finally:
        other.close()
    assert seen["status"] == "pending"


# --- restore_attempt ---------------------------------------------------------


def test_restore_attempt_deletes_a_freshly_inserted_job(
    conn: sqlite3.Connection, db_path: Path
) -> None:
    """prior_job None removes the row open_attempt inserted and restores the run."""
    from vibe_quant.db import run_lifecycle

    run_id = _run(conn, status="pending")
    prior_run = _run_full(conn, run_id)
    run_lifecycle.open_attempt(conn, run_id, "screening", "/tmp/s.log", exists=False)
    assert _job_full(conn, run_id)["pid"] == 0

    run_lifecycle.restore_attempt(conn, run_id, None, prior_run)

    assert conn.execute(
        "SELECT COUNT(*) FROM background_jobs WHERE run_id = ?", (run_id,)
    ).fetchone()[0] == 0
    run = _run_full(conn, run_id)
    assert (run["status"], run["pid"]) == ("pending", None)
    assert (run["started_at"], run["completed_at"], run["error_message"]) == (None, None, None)

    other = get_connection(db_path)
    try:
        count = other.execute(
            "SELECT COUNT(*) FROM background_jobs WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
    finally:
        other.close()
    assert count == 0


@pytest.mark.parametrize(
    ("job_type", "log_file", "pid", "stamp"),
    [
        ("validation", "/tmp/v.log", 5150, ("2024-01-01 00:00:01", "2024-01-01 00:00:02")),
        ("data_update", None, 6160, ("2024-02-03 04:05:06", "2024-02-03 04:05:07")),
    ],
)
def test_restore_attempt_puts_back_every_prior_column(
    conn: sqlite3.Connection,
    job_type: str,
    log_file: str | None,
    pid: int,
    stamp: tuple[str, str],
) -> None:
    """Every job column except id/run_id and all six run columns come back."""
    from vibe_quant.db import run_lifecycle

    run_id = _run(conn, status="running", pid=pid)
    conn.execute(
        """INSERT INTO background_jobs
           (run_id, pid, pid_start_time, job_type, status, log_file,
            started_at, heartbeat_at, completed_at, error_message)
           VALUES (?, ?, 'linux:9', ?, 'failed', ?, ?, ?, ?, 'boom')""",
        (run_id, pid, job_type, log_file, stamp[0], stamp[1], stamp[0]),
    )
    conn.execute(
        """UPDATE backtest_runs SET error_message = 'boom', started_at = ?,
           heartbeat_at = ?, completed_at = ? WHERE id = ?""",
        (stamp[0], stamp[1], stamp[0], run_id),
    )
    conn.commit()
    prior_job = _job_full(conn, run_id)
    prior_run = _run_full(conn, run_id)

    run_lifecycle.open_attempt(conn, run_id, "screening", "/tmp/other.log", exists=True)
    assert _job_full(conn, run_id)["pid"] == 0

    run_lifecycle.restore_attempt(conn, run_id, prior_job, prior_run)

    restored = _job_full(conn, run_id)
    assert {k: v for k, v in restored.items() if k not in ("id", "run_id")} == {
        k: v for k, v in prior_job.items() if k not in ("id", "run_id")
    }
    run = _run_full(conn, run_id)
    for field in (
        "status",
        "pid",
        "started_at",
        "heartbeat_at",
        "completed_at",
        "error_message",
    ):
        assert run[field] == prior_run[field], field
    assert (run["status"], run["pid"], run["error_message"]) == ("running", pid, "boom")


# --- register_spawned --------------------------------------------------------


def test_register_spawned_starts_a_pending_job(conn: sqlite3.Connection) -> None:
    """pending -> running with the pid, the identity and a fresh heartbeat."""
    from vibe_quant.db import run_lifecycle

    run_id = _run(conn, status="running")
    _job(conn, run_id, status="pending", pid=0, pid_start_time=None)
    _stamp_heartbeats(conn, run_id)

    run_lifecycle.register_spawned(conn, run_id, 5150, "linux:123")

    job = _job_full(conn, run_id)
    assert (job["status"], job["pid"], job["pid_start_time"]) == ("running", 5150, "linux:123")
    assert job["heartbeat_at"] != OLD_HEARTBEAT
    run = _run_full(conn, run_id)
    assert run["pid"] == 5150
    assert run["heartbeat_at"] != OLD_HEARTBEAT


def test_register_spawned_records_the_pid_but_keeps_the_childs_result(
    conn: sqlite3.Connection,
) -> None:
    """A job the child already completed stays completed; only pid/pid_start_time move."""
    from vibe_quant.db import run_lifecycle

    run_id = _run(conn, status="running")
    _job(conn, run_id, status="completed", pid=0, pid_start_time=None)
    _stamp_heartbeats(conn, run_id)

    run_lifecycle.register_spawned(conn, run_id, 6160, "darwin:1.000002")

    job = _job_full(conn, run_id)
    assert (job["status"], job["pid"], job["pid_start_time"]) == (
        "completed",
        6160,
        "darwin:1.000002",
    )
    assert job["heartbeat_at"] == OLD_HEARTBEAT
    assert _run_full(conn, run_id)["pid"] == 6160


def test_register_spawned_refuses_another_attempts_pid(
    conn: sqlite3.Connection, db_path: Path
) -> None:
    """rowcount 0: the owner's pid is read, the text raised, nothing changes."""
    from vibe_quant.db import run_lifecycle

    run_id = _run(conn, status="running")
    _job(conn, run_id, status="running", pid=777, pid_start_time="other-attempt")
    before_job = _job_full(conn, run_id)
    before_run = _run_full(conn, run_id)

    with pytest.raises(
        ValueError, match=rf"^Run {run_id} already has an active job \(pid=777\)$"
    ):
        run_lifecycle.register_spawned(conn, run_id, 5150, "linux:123")

    assert _job_full(conn, run_id) == before_job
    assert _run_full(conn, run_id) == before_run
    other = get_connection(db_path)
    try:
        committed_job = other.execute(
            "SELECT pid, pid_start_time, status FROM background_jobs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        committed_run = other.execute(
            "SELECT pid FROM backtest_runs WHERE id = ?", (run_id,)
        ).fetchone()
    finally:
        other.close()
    assert (committed_job["pid"], committed_job["pid_start_time"]) == (777, "other-attempt")
    assert (committed_job["status"], committed_run["pid"]) == ("running", None)


# --- heartbeat ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("job", "run"),
    [(True, False), (False, True), (True, True)],
    ids=["job-only", "run-only", "both"],
)
def test_heartbeat_touches_only_the_requested_rows(
    conn: sqlite3.Connection, job: bool, run: bool
) -> None:
    """Each flag moves exactly its own row's heartbeat_at (seeded 2000-01-01)."""
    from vibe_quant.db import run_lifecycle

    run_id = _run(conn, status="running")
    _job(conn, run_id, status="running")
    _stamp_heartbeats(conn, run_id)

    run_lifecycle.heartbeat(conn, run_id, job=job, run=run)

    assert (_job_full(conn, run_id)["heartbeat_at"] != OLD_HEARTBEAT) is job
    assert (_run_full(conn, run_id)["heartbeat_at"] != OLD_HEARTBEAT) is run


# --- set_run_status ----------------------------------------------------------


def test_set_run_status_terminal_keeps_the_error_unless_one_is_given(
    conn: sqlite3.Connection, db_path: Path
) -> None:
    """completed keeps a prior error; passing one overwrites it on failed."""
    from vibe_quant.db import run_lifecycle

    completed = _run(conn, status="running", error_message="prior")
    failed = _run(conn, status="running")

    run_lifecycle.set_run_status(conn, completed, "completed", None, None)
    run_lifecycle.set_run_status(conn, failed, "failed", None, "boom")

    first = _run_row(conn, completed)
    assert (first["status"], first["error_message"]) == ("completed", "prior")
    assert first["completed_at"] is not None
    second = _run_row(conn, failed)
    assert (second["status"], second["error_message"]) == ("failed", "boom")
    assert second["completed_at"] is not None

    other = get_connection(db_path)
    try:
        seen = other.execute(
            "SELECT status, error_message FROM backtest_runs WHERE id = ?", (failed,)
        ).fetchone()
    finally:
        other.close()
    assert (seen["status"], seen["error_message"]) == ("failed", "boom")


@pytest.mark.parametrize("terminal", ["completed", "cancelled"])
def test_set_run_status_running_never_reopens_a_terminal_run(
    conn: sqlite3.Connection, terminal: str
) -> None:
    """The monotonic guard leaves status, pid, timestamps and error untouched."""
    from vibe_quant.db import run_lifecycle

    run_id = _run(conn, status=terminal, pid=4242, error_message="final")
    conn.execute(
        """UPDATE backtest_runs SET started_at = '2024-01-01 00:00:00',
           heartbeat_at = '2024-01-01 00:00:01', completed_at = '2024-01-01 00:00:02'
           WHERE id = ?""",
        (run_id,),
    )
    conn.commit()
    before = _run_full(conn, run_id)

    run_lifecycle.set_run_status(conn, run_id, "running", 5150, None)

    assert _run_full(conn, run_id) == before


def test_set_run_status_running_sets_the_pid_and_started_at(conn: sqlite3.Connection) -> None:
    """running opens a pending run and records the pid when one is given."""
    from vibe_quant.db import run_lifecycle

    with_pid = _run(conn, status="pending")
    without_pid = _run(conn, status="pending")

    run_lifecycle.set_run_status(conn, with_pid, "running", 5150, None)
    run_lifecycle.set_run_status(conn, without_pid, "running", None, None)

    first = _run_full(conn, with_pid)
    assert (first["status"], first["pid"]) == ("running", 5150)
    assert first["started_at"] is not None
    second = _run_full(conn, without_pid)
    assert (second["status"], second["pid"]) == ("running", None)
    assert second["started_at"] is not None


# --- public delegates --------------------------------------------------------


def test_state_manager_heartbeat_delegates_to_its_own_row(db_path: Path) -> None:
    """R3: StateManager.update_heartbeat is run-only; update_job_heartbeat is job-only."""
    from vibe_quant.db.state_manager import StateManager

    state = StateManager(db_path)
    try:
        run_id = _run(state.conn, status="running")
        _job(state.conn, run_id, status="running")

        _stamp_heartbeats(state.conn, run_id)
        state.update_heartbeat(run_id)
        assert _job_full(state.conn, run_id)["heartbeat_at"] == OLD_HEARTBEAT
        assert _run_full(state.conn, run_id)["heartbeat_at"] != OLD_HEARTBEAT

        _stamp_heartbeats(state.conn, run_id)
        state.update_job_heartbeat(run_id)
        assert _job_full(state.conn, run_id)["heartbeat_at"] != OLD_HEARTBEAT
        assert _run_full(state.conn, run_id)["heartbeat_at"] == OLD_HEARTBEAT
    finally:
        state.close()


def test_manager_heartbeat_delegates_to_both_rows(db_path: Path) -> None:
    """R3: BacktestJobManager.update_heartbeat still moves both rows."""
    from vibe_quant.jobs.manager import BacktestJobManager

    manager = BacktestJobManager(db_path)
    try:
        run_id = _run(manager.conn, status="running")
        _job(manager.conn, run_id, status="running")
        _stamp_heartbeats(manager.conn, run_id)

        manager.update_heartbeat(run_id)

        assert _job_full(manager.conn, run_id)["heartbeat_at"] != OLD_HEARTBEAT
        assert _run_full(manager.conn, run_id)["heartbeat_at"] != OLD_HEARTBEAT
    finally:
        manager.close()
