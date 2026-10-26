"""Single owner of the run/job lifecycle SQL (rabmo.12).

Every write to ``backtest_runs.status`` and ``background_jobs`` lives here.
These are pure functions over a ``sqlite3.Connection``: the caller holds that
connection's lock (``StateManager._write_lock``, ``BacktestJobManager._db_lock``)
and passes its own connection, so the locking contract, the connection PRAGMAs
and every commit point are unchanged. This module opens no connection and
imports nothing from ``vibe_quant.jobs``.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


class JobStatus(StrEnum):
    """Job status enum."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


# Terminal run/job statuses a later "completed" report must not overwrite
# (a run the runner marked failed — e.g. 0 trades — stays failed).
_STICKY_ON_COMPLETE = frozenset({"failed", "killed", "cancelled"})
# ... and that a later failure report must not overwrite (user intent wins).
_STICKY_ON_FAIL = frozenset({"killed", "cancelled"})

# Statuses in which a process may still claim its job row.
_OPEN = ("pending", "running")
# Fail both rows when the claim cannot establish a verifiable process identity.
_FAIL_JOB_SQL = (
    "UPDATE background_jobs SET status = 'failed', completed_at = datetime('now'), "
    "error_message = ? WHERE run_id = ?"
)
_FAIL_RUN_SQL = (
    "UPDATE backtest_runs SET status = 'failed', completed_at = datetime('now'), "
    "error_message = ? WHERE id = ?"
)


def finish(
    conn: sqlite3.Connection,
    run_id: int,
    requested: JobStatus,
    error: str | None,
    *,
    reconcile: bool,
    owner: int | None,
) -> bool:
    """Record a terminal status for the job row and its backtest run.

    Read-decide-write runs in one IMMEDIATE transaction, because the job
    subprocess and the API process report on the same rows concurrently.

    - ``completed`` never overwrites failed/killed/cancelled; the job row
      adopts the run's failure instead.
    - ``failed`` never overwrites killed/cancelled.
    - ``reconcile`` (process found dead/stale): only acts while the job row
      is still 'running', and only flips a run that is itself still
      running/pending (a run the runner already completed keeps its status).
    - ``owner``: pid this manager claimed, so a result from a process a
      re-launch has since replaced is not recorded (vibe-quant-qgtdz).

    Returns:
        False if nothing was written (reconcile on a non-running job, or the
        row now belongs to another attempt).
    """
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        job = conn.execute(
            "SELECT status, pid FROM background_jobs WHERE run_id = ?", (run_id,)
        ).fetchone()
        run = conn.execute(
            "SELECT status, error_message FROM backtest_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if owner is not None and (job is None or int(job["pid"]) != owner):
            # qgtdz: a re-launch owns the rows now; this process's result is not theirs.
            conn.rollback()
            logger.warning(
                "run %d: job row now records pid %s, not this process (pid %d); "
                "result not recorded",
                run_id,
                job["pid"] if job else None,
                owner,
            )
            return False
        job_status = job["status"] if job else None
        run_status = run["status"] if run else None
        if reconcile and job_status != JobStatus.RUNNING.value:
            conn.rollback()
            return False

        status: str = requested.value
        err = error
        if requested is JobStatus.COMPLETED:
            if run_status in _STICKY_ON_COMPLETE:
                status, err = run_status, run["error_message"]
            elif job_status in _STICKY_ON_COMPLETE:
                status = job_status
        elif requested is JobStatus.FAILED:
            if run_status in _STICKY_ON_FAIL:
                status, err = run_status, None
            elif job_status in _STICKY_ON_FAIL:
                status, err = job_status, None

        conn.execute(
            """UPDATE background_jobs
                       SET status = ?, completed_at = datetime('now'),
                           error_message = COALESCE(?, error_message)
                       WHERE run_id = ?""",
            (status, err, run_id),
        )
        update_run = run is not None and (
            not reconcile or run_status in ("running", "pending")
        )
        if update_run:
            conn.execute(
                """UPDATE backtest_runs
                           SET status = ?, completed_at = datetime('now'),
                               error_message = COALESCE(?, error_message)
                           WHERE id = ?""",
                (status, err, run_id),
            )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    return True


def claim(
    conn: sqlite3.Connection, run_id: int, pid: int, identity: str | None
) -> str | None:
    """The job's own process records its pid + start time (vibe-quant-qgtdz).

    Returns None once the job row is 'running' with (pid, identity); otherwise the
    refusal text, and the caller must not trade. Never revives a terminal row and
    never overwrites another live attempt's pid.
    """
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        job = conn.execute(
            "SELECT status, pid, pid_start_time FROM background_jobs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        run = conn.execute(
            "SELECT status FROM backtest_runs WHERE id = ?", (run_id,)
        ).fetchone()
        refusal: str | None = None
        if job is None or run is None:
            refusal = (
                f"Refusing to trade: run {run_id} has no job or run row "
                "to record this process in"
            )
        elif int(job["pid"]) not in (0, pid) or (
            int(job["pid"]) == pid and job["pid_start_time"] not in (None, identity)
        ):
            refusal = (
                f"Refusing to trade: run {run_id} is recorded for pid "
                f"{job['pid']}, not this process (pid {pid})"
            )
        elif job["status"] not in _OPEN or run["status"] not in _OPEN:
            refusal = (
                f"Refusing to trade: run {run_id} is already {run['status']} "
                f"(job {job['status']}); a finished run is never reopened"
            )
        elif identity is None:
            refusal = (
                "Refusing to trade: cannot read this process's start time "
                f"(pid {pid}), so stop/kill could not verify it"
            )
            for sql in (_FAIL_JOB_SQL, _FAIL_RUN_SQL):
                conn.execute(sql, (refusal, run_id))
            conn.commit()
            return refusal
        if refusal is not None:
            conn.rollback()
            return refusal
        conn.execute(
            """UPDATE background_jobs
                       SET pid = ?, pid_start_time = ?, status = 'running',
                           heartbeat_at = datetime('now')
                       WHERE run_id = ?""",
            (pid, identity, run_id),
        )
        conn.execute(
            "UPDATE backtest_runs SET pid = ?, heartbeat_at = datetime('now') WHERE id = ?",
            (pid, run_id),
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    return None


def fail_orphan_run(conn: sqlite3.Connection, run_id: int, error: str) -> bool:
    """Fail one orphaned run if it is still pending/running.

    Status guard: a job that finished between the caller's SELECT and this
    UPDATE already owns the row — never overwrite it. No commit here: the
    caller commits once for the whole reap sweep, as before.

    Returns:
        True if this call flipped the row.
    """
    cursor = conn.execute(
        """UPDATE backtest_runs
                       SET status = 'failed', completed_at = datetime('now'),
                           error_message = ?
                       WHERE id = ? AND status IN ('pending', 'running')""",
        (error, run_id),
    )
    return cursor.rowcount == 1


def open_attempt(
    conn: sqlite3.Connection,
    run_id: int,
    job_type: str,
    log_file: str | None,
    *,
    exists: bool,
) -> None:
    """Commit the fresh-attempt reset of the job and run rows (caller holds the lock)."""
    if exists:
        conn.execute(
                """UPDATE background_jobs
                   SET pid = 0, pid_start_time = NULL, job_type = ?, status = 'pending',
                       log_file = ?, started_at = datetime('now'),
                       heartbeat_at = datetime('now'), completed_at = NULL,
                       error_message = NULL
                   WHERE run_id = ?""",
                (job_type, log_file, run_id),
        )
    else:
        conn.execute(
                """INSERT INTO background_jobs
                   (run_id, pid, pid_start_time, job_type, status, log_file,
                    started_at, heartbeat_at)
                   VALUES (?, 0, NULL, ?, 'pending', ?, datetime('now'), datetime('now'))""",
                (run_id, job_type, log_file),
        )
    conn.execute(
            """UPDATE backtest_runs
               SET status = 'running', pid = NULL, started_at = datetime('now'),
                   heartbeat_at = datetime('now'), completed_at = NULL,
                   error_message = NULL
               WHERE id = ?""",
            (run_id,),
    )
    conn.commit()


def restore_attempt(
    conn: sqlite3.Connection,
    run_id: int,
    prior_job: Mapping[str, object] | None,
    prior_run: Mapping[str, object] | None,
) -> None:
    """Undo :func:`open_attempt` after a failed spawn (caller holds the lock)."""
    if prior_job is None:
        conn.execute("DELETE FROM background_jobs WHERE run_id = ?", (run_id,))
    else:
        cols = [c for c in prior_job if c not in ("id", "run_id")]
        conn.execute(
            f"UPDATE background_jobs SET {', '.join(f'{c} = ?' for c in cols)} "
            "WHERE run_id = ?",
            [prior_job[c] for c in cols] + [run_id],
        )
    if prior_run is not None:
        fields = (
            "status",
            "pid",
            "started_at",
            "heartbeat_at",
            "completed_at",
            "error_message",
        )
        conn.execute(
            f"UPDATE backtest_runs SET {', '.join(f'{c} = ?' for c in fields)} WHERE id = ?",
            [prior_run[c] for c in fields] + [run_id],
        )
    conn.commit()


def register_spawned(
    conn: sqlite3.Connection, run_id: int, pid: int, pid_start: str | None
) -> None:
    """Record the spawned pid on the job and run rows (caller holds the lock).

    Monotonic: only a still-'pending' job becomes 'running'; if the child
    already reported a terminal status, that result stands and only the pid
    (which identifies the exited process) is recorded.

    Raises:
        ValueError: If the job row belongs to another live attempt.
    """
    cursor = conn.execute(
            """UPDATE background_jobs
               SET pid = ?, pid_start_time = ?,
                   status = CASE WHEN status = 'pending' THEN 'running' ELSE status END,
                   heartbeat_at = CASE WHEN status = 'pending'
                                       THEN datetime('now') ELSE heartbeat_at END
               WHERE run_id = ? AND pid IN (0, ?)""",
            (pid, pid_start, run_id, pid),
    )
    if cursor.rowcount == 0:
        owner = conn.execute(
            "SELECT pid FROM background_jobs WHERE run_id = ?", (run_id,)
        ).fetchone()
        conn.commit()
        raise ValueError(
            f"Run {run_id} already has an active job (pid={owner['pid'] if owner else 0})"
        )
    conn.execute(
            """UPDATE backtest_runs
               SET pid = ?,
                   heartbeat_at = CASE WHEN status = 'running'
                                       THEN datetime('now') ELSE heartbeat_at END
               WHERE id = ?""",
            (pid, run_id),
    )
    conn.commit()


def heartbeat(conn: sqlite3.Connection, run_id: int, *, job: bool, run: bool) -> None:
    """Update the heartbeat timestamp of the job and/or run row (caller holds the lock)."""
    if job:
        conn.execute(
            """UPDATE background_jobs SET heartbeat_at = datetime('now')
               WHERE run_id = ?""",
            (run_id,),
        )
    if run:
        conn.execute(
            """UPDATE backtest_runs SET heartbeat_at = datetime('now')
               WHERE id = ?""",
            (run_id,),
        )
    conn.commit()


def set_run_status(
    conn: sqlite3.Connection,
    run_id: int,
    status: str,
    pid: int | None,
    error_message: str | None,
) -> None:
    """Update backtest run status (caller holds the lock).

    Args:
        run_id: Run ID.
        status: New status (pending, running, completed, failed).
        pid: Process ID if running.
        error_message: Error message if failed.

    ``running`` is monotonic: it only opens a pending/running run. A terminal
    run (failed/completed/cancelled/killed) keeps its status, pid and error —
    a launch that resumes after the child already reported must not erase the
    child's result. A deliberate re-launch goes through ``start_job``.
    """
    updates = ["status = ?"]
    params: list[object] = [status]
    guard = ""

    if status == "running":
        guard = " AND status NOT IN ('failed', 'completed', 'cancelled', 'killed')"
        updates.append("started_at = datetime('now')")
        if pid is not None:
            updates.append("pid = ?")
            params.append(pid)
    elif status in ("completed", "failed", "killed", "cancelled"):
        updates.append("completed_at = datetime('now')")
        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)

    params.append(run_id)
    conn.execute(
        f"UPDATE backtest_runs SET {', '.join(updates)} WHERE id = ?{guard}", params
    )
    conn.commit()
