"""BacktestJobManager for background subprocess management."""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vibe_quant.db.connection import DEFAULT_DB_PATH, get_connection
from vibe_quant.db.schema import init_schema

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Type alias for database row dict
RowDict = dict[str, Any]

# Heartbeat configuration per issue spec
HEARTBEAT_INTERVAL_SECONDS = 30
STALE_THRESHOLD_SECONDS = 120


class JobStatus(StrEnum):
    """Job status enum."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


def process_start_time(pid: int) -> str | None:
    """OS start time of ``pid``: the process identity token (None if no such process).

    PIDs are recycled, (pid, start time) is not. A job row stores both so a
    reused PID is never signalled or reported as the job's process. Read from
    the kernel (/proc on Linux, sysctl on macOS — psutil is not a dependency);
    ``ps`` only as a fallback elsewhere.
    """
    if pid <= 0:
        return None
    try:
        if sys.platform.startswith("linux"):
            try:
                stat = Path(f"/proc/{pid}/stat").read_text()
            except OSError:
                return None
            # Fields after the "(comm)" start at field 3; starttime is field 22.
            return f"linux:{stat.rsplit(')', 1)[1].split()[19]}"
        if sys.platform == "darwin":
            return _darwin_start_time(pid)
        proc = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
        )
        value = " ".join(proc.stdout.split()) if isinstance(proc.stdout, str) else ""
    except Exception:  # noqa: BLE001 — unknown identity degrades to the conservative path
        logger.debug("process_start_time(%d) failed", pid, exc_info=True)
        return None
    return f"ps:{value}" if value else None


def _darwin_start_time(pid: int) -> str | None:
    """``kinfo_proc.kp_proc.p_starttime`` via sysctl(CTL_KERN, KERN_PROC, KERN_PROC_PID)."""
    import ctypes
    import ctypes.util
    import struct

    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    mib = (ctypes.c_int * 4)(1, 14, 1, pid)  # CTL_KERN, KERN_PROC, KERN_PROC_PID
    size = ctypes.c_size_t(0)
    if libc.sysctl(mib, 4, None, ctypes.byref(size), None, 0) != 0 or size.value == 0:
        return None
    buf = ctypes.create_string_buffer(size.value)
    if libc.sysctl(mib, 4, buf, ctypes.byref(size), None, 0) != 0 or size.value == 0:
        return None  # process vanished between the two calls
    # extern_proc starts with a union whose struct timeval p_starttime is at offset 0.
    sec, usec = struct.unpack_from("qi", buf.raw, 0)
    return f"darwin:{sec}.{usec:06d}"


def _process_command(pid: int) -> str:
    """Command line of ``pid`` ('' if unknown). Identity fallback for legacy rows."""
    try:
        proc = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:  # noqa: BLE001 — unknown command → treated as not ours
        logger.debug("_process_command(%d) failed", pid, exc_info=True)
        return ""
    return proc.stdout.strip() if isinstance(proc.stdout, str) else ""


@dataclass
class JobInfo:
    """Information about a background job."""

    run_id: int
    pid: int
    job_type: str
    status: JobStatus
    heartbeat_at: datetime | None
    started_at: datetime | None
    completed_at: datetime | None
    log_file: str | None
    pid_start_time: str | None = None

    @property
    def is_stale(self) -> bool:
        """Check if job heartbeat is stale (>120s)."""
        if self.status != JobStatus.RUNNING:
            return False
        if self.heartbeat_at is None:
            # No heartbeat yet, check started_at
            if self.started_at is None:
                return True
            reference = self.started_at
        else:
            reference = self.heartbeat_at

        now = datetime.now(UTC)
        # Parse timestamps as UTC
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=UTC)
        return (now - reference).total_seconds() > STALE_THRESHOLD_SECONDS


# Regime-cross campaign runs share the discovery/validation entry points but
# carry dedicated run_modes so the matrix report can separate them from
# standalone runs.
_JOB_TO_MODES: dict[str, set[str]] = {
    "screening": {"screening"},
    "validation": {"validation", "regime_cross_oos"},
    "discovery": {"discovery", "regime_cross_discovery"},
}

# Terminal run/job statuses a later "completed" report must not overwrite
# (a run the runner marked failed — e.g. 0 trades — stays failed).
_STICKY_ON_COMPLETE = frozenset({"failed", "killed", "cancelled"})
# ... and that a later failure report must not overwrite (user intent wins).
_STICKY_ON_FAIL = frozenset({"killed", "cancelled"})


class BacktestJobManager:
    """Manages background backtest jobs with subprocess tracking.

    Provides:
    - Job spawning as subprocess with PID + process-identity tracking
    - Status monitoring via SQLite
    - Heartbeat protocol (30s updates, 120s stale threshold)
    - Job termination (kill) that never signals a recycled PID
    - Stale job cleanup and status reconciliation
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize job manager.

        Args:
            db_path: Path to database. Uses default if not specified.
        """
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._start_lock = threading.Lock()
        # The connection is shared by the heartbeat thread and API threads.
        self._db_lock = threading.RLock()

    @property
    def db_path(self) -> Path:
        """Effective database path; job subprocesses must be pointed at it (--db)."""
        return self._db_path if self._db_path is not None else DEFAULT_DB_PATH

    @property
    def conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        with self._db_lock:
            if self._conn is None:
                self._conn = get_connection(self._db_path)
                init_schema(self._conn)
            return self._conn

    def close(self) -> None:
        """Close database connection."""
        with self._db_lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def start_job(
        self,
        run_id: int,
        job_type: str,
        command: list[str],
        log_file: str | None = None,
        env: dict[str, str] | None = None,
    ) -> int:
        """Start a background job for a backtest run.

        Args:
            run_id: Backtest run ID to associate with job.
            job_type: Type of job (screening, validation, data_update).
            command: Command line arguments to spawn subprocess.
            log_file: Optional path to log file.
            env: Optional environment variables for subprocess.

        Returns:
            Process ID of spawned subprocess.

        Raises:
            ValueError: If run already has an active job.
        """
        expected_modes = _JOB_TO_MODES.get(job_type)

        with self._start_lock, self._db_lock:
            # Validate run_mode matches job_type to prevent cross-mode launches
            if expected_modes is not None:
                row = self.conn.execute(
                    "SELECT run_mode FROM backtest_runs WHERE id = ?", (run_id,)
                ).fetchone()
                if row and row["run_mode"] not in expected_modes:
                    raise ValueError(
                        f"Run {run_id} has mode '{row['run_mode']}', "
                        f"cannot launch as '{job_type}'"
                    )

            # Check for existing active job
            existing = self._get_job_record(run_id)
            if existing and existing["status"] == "running":
                raise ValueError(f"Run {run_id} already has an active job (pid={existing['pid']})")

            log_handle = None
            if log_file:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = log_path.open("w")

            try:
                proc = subprocess.Popen(
                    command,
                    stdout=log_handle or subprocess.DEVNULL,
                    stderr=subprocess.STDOUT if log_handle else subprocess.DEVNULL,
                    start_new_session=True,  # Detach from parent process group
                    env=env,
                )
            finally:
                # The child holds its own copy of the fd; ours would leak one
                # descriptor per job (it was only closed if this same manager
                # instance later saw the job finish, which it never does).
                if log_handle is not None:
                    log_handle.close()

            pid = proc.pid
            pid_start = process_start_time(pid)

            # Register job in database while still holding lock to prevent
            # race: two callers both passing the active-job check, both
            # spawning processes, second write clobbering first PID.
            existing_rec = self.conn.execute(
                "SELECT id FROM background_jobs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if existing_rec:
                self.conn.execute(
                    """UPDATE background_jobs
                       SET pid = ?, pid_start_time = ?, job_type = ?, status = 'running',
                           log_file = ?, started_at = datetime('now'),
                           heartbeat_at = datetime('now'), completed_at = NULL,
                           error_message = NULL
                       WHERE run_id = ?""",
                    (pid, pid_start, job_type, log_file, run_id),
                )
            else:
                self.conn.execute(
                    """INSERT INTO background_jobs
                       (run_id, pid, pid_start_time, job_type, status, log_file,
                        started_at, heartbeat_at)
                       VALUES (?, ?, ?, ?, 'running', ?, datetime('now'), datetime('now'))""",
                    (run_id, pid, pid_start, job_type, log_file),
                )
            # Also update backtest_runs (a re-launch starts a fresh attempt).
            self.conn.execute(
                """UPDATE backtest_runs
                   SET status = 'running', pid = ?, started_at = datetime('now'),
                       heartbeat_at = datetime('now'), completed_at = NULL,
                       error_message = NULL
                   WHERE id = ?""",
                (pid, run_id),
            )
            self.conn.commit()

        return pid

    def get_status(self, run_id: int) -> JobStatus | None:
        """Get job status for a run.

        Args:
            run_id: Backtest run ID.

        Returns:
            Job status or None if no job exists.
        """
        record = self._get_job_record(run_id)
        if record is None:
            return None
        return JobStatus(record["status"])

    def get_job_info(self, run_id: int) -> JobInfo | None:
        """Get full job info for a run.

        Args:
            run_id: Backtest run ID.

        Returns:
            JobInfo or None if no job exists.
        """
        record = self._get_job_record(run_id)
        if record is None:
            return None
        return self._record_to_info(record)

    def list_active_jobs(self) -> list[JobInfo]:
        """List all jobs whose DB status is running.

        Pure DB read; call :meth:`reconcile_jobs` first to drop jobs whose
        process is gone. Signal a job via :meth:`signal_job`, never by raw PID.

        Returns:
            List of JobInfo for running jobs.
        """
        with self._db_lock:
            rows = self.conn.execute(
                "SELECT * FROM background_jobs WHERE status = 'running'"
            ).fetchall()
        return [self._record_to_info(dict(row)) for row in rows]

    def list_all_jobs(self, job_type: str | None = None) -> list[JobInfo]:
        """List all jobs, optionally filtered by type.

        Args:
            job_type: Filter by job type (e.g. 'discovery'). None = all.

        Returns:
            List of JobInfo ordered by started_at descending.
        """
        with self._db_lock:
            if job_type:
                rows = self.conn.execute(
                    "SELECT * FROM background_jobs WHERE job_type = ? ORDER BY started_at DESC",
                    (job_type,),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM background_jobs ORDER BY started_at DESC"
                ).fetchall()
        return [self._record_to_info(dict(row)) for row in rows]

    def list_stale_jobs(self) -> list[JobInfo]:
        """List jobs with stale heartbeats (>120s old).

        Returns:
            List of JobInfo for stale jobs.
        """
        threshold = datetime.now(UTC) - timedelta(seconds=STALE_THRESHOLD_SECONDS)
        threshold_str = threshold.strftime("%Y-%m-%d %H:%M:%S")

        with self._db_lock:
            rows = self.conn.execute(
                """SELECT * FROM background_jobs
                   WHERE status = 'running'
                   AND (heartbeat_at IS NULL OR heartbeat_at < ?)""",
                (threshold_str,),
            ).fetchall()
        return [self._record_to_info(dict(row)) for row in rows]

    def kill_job(
        self,
        run_id: int,
        force: bool = False,
        graceful_timeout: float = 10.0,
    ) -> bool:
        """Terminate a running job without blocking the caller.

        Sends SIGTERM to the job's process group (SIGKILL if ``force``) and
        returns immediately; a daemon thread escalates to SIGKILL if the
        process is still alive after ``graceful_timeout`` seconds. Nothing is
        signalled unless the PID verifiably still belongs to the job.

        Args:
            run_id: Backtest run ID.
            force: If True, skip SIGTERM and send SIGKILL immediately.
            graceful_timeout: Seconds to wait after SIGTERM before SIGKILL.

        Returns:
            True if job was marked killed, False if job not found or not running.
        """
        record = self._get_job_record(run_id)
        if record is None or record["status"] != "running":
            return False

        pid = int(record["pid"])
        identity = record.get("pid_start_time")
        if self._owns_process(pid, identity):
            if force:
                self._signal_group(pid, signal.SIGKILL)
            else:
                self._signal_group(pid, signal.SIGTERM)
                threading.Thread(
                    target=self._escalate_kill,
                    args=(pid, identity, graceful_timeout),
                    name=f"kill-escalate-{run_id}",
                    daemon=True,
                ).start()
        else:
            logger.warning(
                "kill_job run_id=%d: pid %d not running or recycled; not signalled",
                run_id,
                pid,
            )

        self._finish(run_id, JobStatus.KILLED, None)
        return True

    def signal_job(self, run_id: int, sig: int) -> bool:
        """Send ``sig`` to a running job's process after verifying its identity.

        Use this instead of ``os.kill(job.pid, ...)`` — a stored PID may have
        been recycled by an unrelated process.

        Returns:
            True if the signal was delivered.
        """
        record = self._get_job_record(run_id)
        if record is None or record["status"] != "running":
            return False
        pid = int(record["pid"])
        if not self._owns_process(pid, record.get("pid_start_time")):
            return False
        try:
            os.kill(pid, sig)
        except OSError:
            return False
        return True

    def update_heartbeat(self, run_id: int) -> None:
        """Update heartbeat timestamp for a job.

        Called periodically by running job to indicate it's alive.

        Args:
            run_id: Backtest run ID.
        """
        with self._db_lock:
            self.conn.execute(
                """UPDATE background_jobs SET heartbeat_at = datetime('now')
                   WHERE run_id = ?""",
                (run_id,),
            )
            self.conn.execute(
                """UPDATE backtest_runs SET heartbeat_at = datetime('now')
                   WHERE id = ?""",
                (run_id,),
            )
            self.conn.commit()

    def mark_completed(self, run_id: int, error: str | None = None) -> None:
        """Mark a job as completed or failed.

        A plain "completed" never overwrites a failure the run itself recorded
        (e.g. the validation runner's 0-trade failure) nor a kill: the job row
        then adopts the run's failed status and error.

        Args:
            run_id: Backtest run ID.
            error: Error message if job failed.
        """
        status = JobStatus.FAILED if error else JobStatus.COMPLETED
        self._finish(run_id, status, error)

    def run_failure(self, run_id: int) -> str | None:
        """Error message if the backtest run is marked failed (e.g. 0 trades), else None."""
        with self._db_lock:
            row = self.conn.execute(
                "SELECT status, error_message FROM backtest_runs WHERE id = ?", (run_id,)
            ).fetchone()
        if row is None or row["status"] != JobStatus.FAILED.value:
            return None
        return str(row["error_message"] or "run marked failed")

    def cleanup_stale_jobs(self) -> int:
        """Detect and clean up stale jobs.

        Jobs with no heartbeat update for >120s are marked as failed. The
        process group is SIGKILLed only if the PID still verifiably belongs
        to the job (never a recycled PID).

        Returns:
            Number of stale jobs cleaned up.
        """
        cleaned = 0
        for job in self.list_stale_jobs():
            if self._owns_process(job.pid, job.pid_start_time):
                self._signal_group(job.pid, signal.SIGKILL)
            if self._finish(
                job.run_id,
                JobStatus.FAILED,
                f"Job stale - no heartbeat for >{STALE_THRESHOLD_SECONDS}s",
                reconcile=True,
            ):
                cleaned += 1
        return cleaned

    def reconcile_jobs(self) -> int:
        """Mark 'running' jobs whose process is gone (or PID recycled) as failed.

        Run at backend startup: jobs that outlived a backend restart keep
        running (their own heartbeat continues); jobs that died meanwhile are
        closed out instead of showing 'running' forever.

        Returns:
            Number of jobs marked failed.
        """
        fixed = 0
        for job in self.list_active_jobs():
            if self._job_alive(job.pid, job.pid_start_time):
                continue
            if self._finish(
                job.run_id,
                JobStatus.FAILED,
                "Process not running (exited without reporting status, or PID reused)",
                reconcile=True,
            ):
                fixed += 1
        return fixed

    def is_process_alive(self, pid: int) -> bool:
        """Check if a process is still running.

        Args:
            pid: Process ID to check.

        Returns:
            True if process exists and is running (not zombie).
        """
        try:
            # First check if process exists
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except OSError:
            # Permission denied means process exists but we can't signal it
            pass

        # Check if it's a zombie by trying waitpid with WNOHANG
        try:
            result_pid, _ = os.waitpid(pid, os.WNOHANG)
            # If result_pid == pid, process has terminated (zombie reaped)
            # If result_pid == 0, process is still running
            return result_pid != pid
        except ChildProcessError:
            # Not a child process, check /proc or use kill(0) result
            # On macOS/BSD, we can't waitpid non-child processes
            # If we got here after kill(0) succeeded, assume alive
            try:
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                return False
            except OSError:
                return True

    def sync_job_status(self, run_id: int) -> JobStatus | None:
        """Sync job status with actual process state.

        If job is marked running but its process is dead (or the PID now
        belongs to another process), updates status to failed.

        Args:
            run_id: Backtest run ID.

        Returns:
            Updated job status or None if no job exists.
        """
        record = self._get_job_record(run_id)
        if record is None:
            return None

        status = JobStatus(record["status"])
        if status == JobStatus.RUNNING and not self._job_alive(
            int(record["pid"]), record.get("pid_start_time")
        ):
            # Process died without marking complete
            self._finish(
                run_id, JobStatus.FAILED, "Process terminated unexpectedly", reconcile=True
            )
            return self.get_status(run_id)

        return status

    # --- process identity helpers ---

    def _job_alive(self, pid: int, identity: str | None) -> bool:
        """True if the job's process is alive (identity-checked when recorded)."""
        if not self.is_process_alive(pid):
            return False
        return identity is None or process_start_time(pid) == identity

    def _owns_process(self, pid: int, identity: str | None) -> bool:
        """True only if ``pid`` is alive AND verifiably the job's process (safe to signal)."""
        if not self.is_process_alive(pid):
            return False
        if identity is not None:
            return process_start_time(pid) == identity
        # Legacy row without identity: only signal one of our own CLIs.
        return "vibe_quant" in _process_command(pid)

    @staticmethod
    def _signal_group(pid: int, sig: int) -> None:
        """Signal the job's process group (start_new_session → PGID == pid)."""
        try:
            os.killpg(pid, sig)
        except OSError:
            # Not a group leader (legacy/foreign launch) — signal the PID only.
            with contextlib.suppress(OSError):
                os.kill(pid, sig)

    def _escalate_kill(self, pid: int, identity: str | None, timeout: float) -> None:
        """SIGKILL the group if it survives ``timeout`` seconds after SIGTERM."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_process_alive(pid):
                return
            time.sleep(0.1)
        if self._owns_process(pid, identity):
            self._signal_group(pid, signal.SIGKILL)

    # --- DB helpers ---

    def _get_job_record(self, run_id: int) -> RowDict | None:
        """Get raw job record from database."""
        with self._db_lock:
            row = self.conn.execute(
                "SELECT * FROM background_jobs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return dict(row) if row else None

    def _finish(
        self,
        run_id: int,
        requested: JobStatus,
        error: str | None,
        *,
        reconcile: bool = False,
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

        Returns:
            False if nothing was written (reconcile on a non-running job).
        """
        with self._db_lock:
            conn = self.conn
            conn.commit()
            conn.execute("BEGIN IMMEDIATE")
            try:
                job = conn.execute(
                    "SELECT status FROM background_jobs WHERE run_id = ?", (run_id,)
                ).fetchone()
                run = conn.execute(
                    "SELECT status, error_message FROM backtest_runs WHERE id = ?", (run_id,)
                ).fetchone()
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

    def _record_to_info(self, record: RowDict) -> JobInfo:
        """Convert database record to JobInfo."""
        return JobInfo(
            run_id=record["run_id"],
            pid=record["pid"],
            job_type=record["job_type"],
            status=JobStatus(record["status"]),
            heartbeat_at=self._parse_datetime(record.get("heartbeat_at")),
            started_at=self._parse_datetime(record.get("started_at")),
            completed_at=self._parse_datetime(record.get("completed_at")),
            log_file=record.get("log_file"),
            pid_start_time=record.get("pid_start_time"),
        )

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        """Parse SQLite datetime string."""
        if value is None:
            return None
        try:
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        except ValueError:
            return None


def run_with_heartbeat(
    run_id: int,
    db_path: Path | None = None,
    interval: float = HEARTBEAT_INTERVAL_SECONDS,
) -> tuple[BacktestJobManager, Callable[[], None]]:
    """Create job manager and start heartbeat thread for a running job.

    Utility for subprocess scripts to send periodic heartbeats.
    Returns both the manager and a stop function to cleanly shut down
    the heartbeat thread.

    Args:
        run_id: Backtest run ID.
        db_path: Path to database.
        interval: Heartbeat interval in seconds (default 30).

    Returns:
        Tuple of (BacktestJobManager, stop_fn). Call stop_fn() to terminate
        the heartbeat thread. The manager should be closed separately.
    """
    manager = BacktestJobManager(db_path)
    stop_event = threading.Event()

    def heartbeat_loop() -> None:
        while not stop_event.is_set():
            with contextlib.suppress(Exception):
                manager.update_heartbeat(run_id)
            stop_event.wait(interval)

    thread = threading.Thread(target=heartbeat_loop, daemon=True)
    thread.start()

    def stop() -> None:
        """Signal heartbeat thread to stop and wait for it to exit."""
        stop_event.set()
        thread.join(timeout=interval + 1)

    return manager, stop
