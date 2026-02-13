"""BacktestJobManager for background subprocess management."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vibe_quant.db.connection import get_connection
from vibe_quant.db.schema import init_schema

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Callable

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


class BacktestJobManager:
    """Manages background backtest jobs with subprocess tracking.

    Provides:
    - Job spawning as subprocess with PID tracking
    - Status monitoring via SQLite
    - Heartbeat protocol (30s updates, 120s stale threshold)
    - Job termination (kill)
    - Stale job cleanup
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize job manager.

        Args:
            db_path: Path to database. Uses default if not specified.
        """
        self._db_path = db_path
        self._log_handles: dict[int, Any] = {}  # run_id → file handle
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = get_connection(self._db_path)
            init_schema(self._conn)
        return self._conn

    def close(self) -> None:
        """Close database connection."""
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
        # Check for existing active job
        existing = self._get_job_record(run_id)
        if existing and existing["status"] == "running":
            raise ValueError(f"Run {run_id} already has an active job (pid={existing['pid']})")

        # Create log directory if needed
        log_handle = None
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("w")

        # Spawn subprocess
        proc = subprocess.Popen(
            command,
            stdout=log_handle or subprocess.DEVNULL,
            stderr=subprocess.STDOUT if log_handle else subprocess.DEVNULL,
            start_new_session=True,  # Detach from parent process group
            env=env,
        )

        pid = proc.pid

        # Track log handle for cleanup
        if log_handle is not None:
            self._log_handles[run_id] = log_handle

        # Register job in database (upsert: update if exists, insert otherwise)
        existing = self.conn.execute(
            "SELECT id FROM background_jobs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if existing:
            self.conn.execute(
                """UPDATE background_jobs
                   SET pid = ?, job_type = ?, status = 'running',
                       log_file = ?, started_at = datetime('now'),
                       heartbeat_at = datetime('now'), completed_at = NULL,
                       error_message = NULL
                   WHERE run_id = ?""",
                (pid, job_type, log_file, run_id),
            )
        else:
            self.conn.execute(
                """INSERT INTO background_jobs
                   (run_id, pid, job_type, status, log_file, started_at, heartbeat_at)
                   VALUES (?, ?, ?, 'running', ?, datetime('now'), datetime('now'))""",
                (run_id, pid, job_type, log_file),
            )
        # Also update backtest_runs table
        self.conn.execute(
            """UPDATE backtest_runs
               SET status = 'running', pid = ?, started_at = datetime('now'), heartbeat_at = datetime('now')
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
        """List all currently running jobs.

        Returns:
            List of JobInfo for running jobs.
        """
        cursor = self.conn.execute(
            "SELECT * FROM background_jobs WHERE status = 'running'"
        )
        return [self._record_to_info(dict(row)) for row in cursor]

    def list_stale_jobs(self) -> list[JobInfo]:
        """List jobs with stale heartbeats (>120s old).

        Returns:
            List of JobInfo for stale jobs.
        """
        threshold = datetime.now(UTC) - timedelta(seconds=STALE_THRESHOLD_SECONDS)
        threshold_str = threshold.strftime("%Y-%m-%d %H:%M:%S")

        cursor = self.conn.execute(
            """SELECT * FROM background_jobs
               WHERE status = 'running'
               AND (heartbeat_at IS NULL OR heartbeat_at < ?)""",
            (threshold_str,),
        )
        return [self._record_to_info(dict(row)) for row in cursor]

    def kill_job(
        self,
        run_id: int,
        force: bool = False,
        graceful_timeout: float = 10.0,
    ) -> bool:
        """Terminate a running job.

        Default behavior sends SIGTERM for graceful shutdown. If the process
        doesn't exit within ``graceful_timeout`` seconds, SIGKILL is sent.

        Args:
            run_id: Backtest run ID.
            force: If True, skip SIGTERM and send SIGKILL immediately.
            graceful_timeout: Seconds to wait after SIGTERM before SIGKILL.
                Only used when ``force=False``.

        Returns:
            True if job was killed, False if job not found or not running.
        """
        record = self._get_job_record(run_id)
        if record is None:
            return False

        if record["status"] != "running":
            return False

        pid = record["pid"]

        try:
            if force:
                os.kill(pid, signal.SIGKILL)
            else:
                # Graceful: SIGTERM first
                os.kill(pid, signal.SIGTERM)
                # Wait up to graceful_timeout for process to exit
                import time

                poll_interval = 0.1
                waited = 0.0
                while waited < graceful_timeout:
                    if not self.is_process_alive(pid):
                        break
                    time.sleep(poll_interval)
                    waited += poll_interval

                # If still alive, escalate to SIGKILL
                if self.is_process_alive(pid):
                    with contextlib.suppress(ProcessLookupError, OSError):
                        os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            # Process already dead
            pass
        except OSError:
            # Permission denied or other error
            pass

        # Update database status
        self._update_job_status(run_id, JobStatus.KILLED)
        return True

    def update_heartbeat(self, run_id: int) -> None:
        """Update heartbeat timestamp for a job.

        Called periodically by running job to indicate it's alive.

        Args:
            run_id: Backtest run ID.
        """
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

        Args:
            run_id: Backtest run ID.
            error: Error message if job failed.
        """
        status = JobStatus.FAILED if error else JobStatus.COMPLETED
        self._update_job_status(run_id, status, error)

    def cleanup_stale_jobs(self) -> int:
        """Detect and clean up stale jobs.

        Jobs with no heartbeat update for >120s are marked as failed.
        Also attempts to kill the process if still running.

        Returns:
            Number of stale jobs cleaned up.
        """
        stale = self.list_stale_jobs()
        for job in stale:
            # Try to kill the process
            with contextlib.suppress(ProcessLookupError, OSError):
                os.kill(job.pid, signal.SIGKILL)

            # Mark as failed
            self._update_job_status(
                job.run_id,
                JobStatus.FAILED,
                "Job stale - no heartbeat for >120s"
            )

        return len(stale)

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

        If job is marked running but process is dead, updates status to failed.

        Args:
            run_id: Backtest run ID.

        Returns:
            Updated job status or None if no job exists.
        """
        record = self._get_job_record(run_id)
        if record is None:
            return None

        status = JobStatus(record["status"])
        if status == JobStatus.RUNNING and not self.is_process_alive(record["pid"]):
            # Process died without marking complete
            self._update_job_status(
                run_id,
                JobStatus.FAILED,
                "Process terminated unexpectedly"
            )
            return JobStatus.FAILED

        return status

    def _get_job_record(self, run_id: int) -> RowDict | None:
        """Get raw job record from database."""
        cursor = self.conn.execute(
            "SELECT * FROM background_jobs WHERE run_id = ?",
            (run_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def _update_job_status(
        self,
        run_id: int,
        status: JobStatus,
        error: str | None = None
    ) -> None:
        """Update job status in database."""
        if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.KILLED):
            # Close log file handle to prevent FD leak
            handle = self._log_handles.pop(run_id, None)
            if handle is not None:
                with contextlib.suppress(Exception):
                    handle.close()
            self.conn.execute(
                """UPDATE background_jobs
                   SET status = ?, completed_at = datetime('now'),
                       error_message = ?
                   WHERE run_id = ?""",
                (status.value, error, run_id),
            )
            # Also update backtest_runs
            if error:
                self.conn.execute(
                    """UPDATE backtest_runs
                       SET status = ?, completed_at = datetime('now'), error_message = ?
                       WHERE id = ?""",
                    (status.value, error, run_id),
                )
            else:
                self.conn.execute(
                    """UPDATE backtest_runs
                       SET status = ?, completed_at = datetime('now')
                       WHERE id = ?""",
                    (status.value, run_id),
                )
        else:
            self.conn.execute(
                "UPDATE background_jobs SET status = ? WHERE run_id = ?",
                (status.value, run_id),
            )
            self.conn.execute(
                "UPDATE backtest_runs SET status = ? WHERE id = ?",
                (status.value, run_id),
            )
        self.conn.commit()

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
        )

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        """Parse SQLite datetime string."""
        if value is None:
            return None
        try:
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=UTC
            )
        except ValueError:
            return None


def run_with_heartbeat(
    run_id: int,
    db_path: Path | None = None,
    interval: int = HEARTBEAT_INTERVAL_SECONDS,
) -> tuple[BacktestJobManager, "Callable[[], None]"]:
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
    import threading

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
