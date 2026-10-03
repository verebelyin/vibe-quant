"""Audit e70tl.15/.17: job manager status semantics, PID identity, data-job heartbeats."""

from __future__ import annotations

import functools
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs import data_job
from vibe_quant.jobs.manager import (
    BacktestJobManager,
    JobStatus,
    process_start_time,
    run_with_heartbeat,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

ZERO_TRADE_ERR = "Validation produced 0 trades — likely missing/empty data"
SLEEPER = [sys.executable, "-c", "import time; time.sleep(30)"]


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "jobs.db"


@pytest.fixture
def jobs(db: Path) -> Iterator[BacktestJobManager]:
    m = BacktestJobManager(db)
    yield m
    for job in m.list_active_jobs():
        m.kill_job(job.run_id, force=True)
    m.close()


@pytest.fixture
def state(db: Path) -> Iterator[StateManager]:
    m = StateManager(db)
    yield m
    m.close()


@pytest.fixture
def foreign() -> Iterator[subprocess.Popen[bytes]]:
    """An unrelated process (own session → also a process-group leader)."""
    proc = subprocess.Popen(
        ["sleep", "30"], start_new_session=True, stdout=subprocess.DEVNULL
    )
    yield proc
    proc.kill()
    proc.wait()


def _run(state: StateManager, mode: str = "validation") -> int:
    sid = state.create_strategy(f"s{time.monotonic_ns()}", {"name": "s"})
    return state.create_backtest_run(sid, mode, ["BTCUSDT"], "4h", "2025-01-01", "2025-06-01", {})


def _insert_job(
    jobs: BacktestJobManager,
    run_id: int,
    pid: int,
    identity: str | None,
    heartbeat_age_s: int = 0,
    job_type: str = "validation",
) -> None:
    hb = (datetime.now(UTC) - timedelta(seconds=heartbeat_age_s)).strftime("%Y-%m-%d %H:%M:%S")
    jobs.conn.execute(
        """INSERT INTO background_jobs (run_id, pid, pid_start_time, job_type, status,
               started_at, heartbeat_at) VALUES (?, ?, ?, ?, 'running', ?, ?)""",
        (run_id, pid, identity, job_type, hb, hb),
    )
    jobs.conn.execute("UPDATE backtest_runs SET status = 'running' WHERE id = ?", (run_id,))
    jobs.conn.commit()


def _wait_dead(jobs: BacktestJobManager, pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not jobs.is_process_alive(pid):
            return True
        time.sleep(0.05)
    return False


class TestProcessIdentity:
    def test_start_time_stable_and_distinct(self, foreign: subprocess.Popen[bytes]) -> None:
        mine = process_start_time(os.getpid())
        assert mine is not None
        assert process_start_time(os.getpid()) == mine
        assert process_start_time(foreign.pid) not in (None, mine)
        assert process_start_time(99_999_999) is None

    def test_start_job_records_identity(
        self, jobs: BacktestJobManager, state: StateManager
    ) -> None:
        run_id = _run(state)
        pid = jobs.start_job(run_id, "validation", SLEEPER)
        info = jobs.get_job_info(run_id)
        assert info is not None
        assert info.pid_start_time == process_start_time(pid)


class TestPidReuse:
    """A stale row whose PID now belongs to another process must not touch it."""

    def test_cleanup_stale_spares_recycled_pid(
        self,
        jobs: BacktestJobManager,
        state: StateManager,
        foreign: subprocess.Popen[bytes],
    ) -> None:
        run_id = _run(state)
        _insert_job(jobs, run_id, foreign.pid, "darwin:1.000000", heartbeat_age_s=600)
        assert jobs.cleanup_stale_jobs() == 1
        time.sleep(0.2)
        assert foreign.poll() is None, "unrelated process was killed"
        assert jobs.get_status(run_id) == JobStatus.FAILED
        run = state.get_backtest_run(run_id)
        assert run is not None and run["status"] == "failed"

    def test_kill_job_spares_recycled_pid(
        self,
        jobs: BacktestJobManager,
        state: StateManager,
        foreign: subprocess.Popen[bytes],
    ) -> None:
        run_id = _run(state)
        _insert_job(jobs, run_id, foreign.pid, "darwin:1.000000")
        assert jobs.kill_job(run_id, force=True) is True
        time.sleep(0.2)
        assert foreign.poll() is None
        assert jobs.get_status(run_id) == JobStatus.KILLED

    def test_legacy_row_without_identity_spares_foreign_process(
        self,
        jobs: BacktestJobManager,
        state: StateManager,
        foreign: subprocess.Popen[bytes],
    ) -> None:
        """Rows written before pid_start_time existed: only vibe_quant CLIs are signalled."""
        run_id = _run(state)
        _insert_job(jobs, run_id, foreign.pid, None, heartbeat_age_s=600)
        assert jobs.cleanup_stale_jobs() == 1
        time.sleep(0.2)
        assert foreign.poll() is None

    def test_signal_job_refuses_recycled_pid(
        self,
        jobs: BacktestJobManager,
        state: StateManager,
        foreign: subprocess.Popen[bytes],
    ) -> None:
        run_id = _run(state)
        _insert_job(jobs, run_id, foreign.pid, "darwin:1.000000", job_type="paper")
        assert jobs.signal_job(run_id, signal.SIGUSR1) is False
        time.sleep(0.2)
        assert foreign.poll() is None

    def test_signal_job_delivers_to_own_process(
        self, jobs: BacktestJobManager, state: StateManager
    ) -> None:
        run_id = _run(state)
        pid = jobs.start_job(run_id, "validation", SLEEPER)
        assert jobs.signal_job(run_id, signal.SIGTERM) is True
        assert _wait_dead(jobs, pid)

    def test_sync_detects_recycled_pid(
        self,
        jobs: BacktestJobManager,
        state: StateManager,
        foreign: subprocess.Popen[bytes],
    ) -> None:
        run_id = _run(state)
        _insert_job(jobs, run_id, foreign.pid, "darwin:1.000000")
        assert jobs.sync_job_status(run_id) == JobStatus.FAILED


class TestReconcile:
    """Backend restart with jobs in flight → statuses reconciled correctly."""

    def test_reconcile_keeps_live_and_fails_dead(
        self, jobs: BacktestJobManager, state: StateManager, db: Path
    ) -> None:
        live_run, dead_run, done_run = _run(state), _run(state), _run(state)
        jobs.start_job(live_run, "validation", SLEEPER)
        dead_pid = jobs.start_job(dead_run, "validation", [sys.executable, "-c", "pass"])
        assert _wait_dead(jobs, dead_pid)
        # Runner finished and wrote results/status, but died before mark_completed.
        jobs.start_job(done_run, "validation", [sys.executable, "-c", "pass"])
        state.update_backtest_run_status(done_run, "completed")

        restarted = BacktestJobManager(db)  # fresh manager = new backend process
        try:
            assert _wait_dead(restarted, jobs.get_job_info(done_run).pid)  # type: ignore[union-attr]
            assert restarted.reconcile_jobs() == 2
            assert restarted.get_status(live_run) == JobStatus.RUNNING
            assert restarted.get_status(dead_run) == JobStatus.FAILED
            dead = state.get_backtest_run(dead_run)
            assert dead is not None and dead["status"] == "failed"
            done = state.get_backtest_run(done_run)
            assert done is not None and done["status"] == "completed"
        finally:
            restarted.close()


class TestStatusSemantics:
    def test_mark_completed_keeps_runner_failure(
        self, jobs: BacktestJobManager, state: StateManager
    ) -> None:
        """0-trade validation: runner marks failed, then __main__ calls mark_completed()."""
        run_id = _run(state)
        jobs.start_job(run_id, "validation", [sys.executable, "-c", "pass"])
        state.update_backtest_run_status(run_id, "failed", error_message=ZERO_TRADE_ERR)
        jobs.mark_completed(run_id)
        run = state.get_backtest_run(run_id)
        assert run is not None
        assert run["status"] == "failed"
        assert run["error_message"] == ZERO_TRADE_ERR
        job = state.get_job(run_id)
        assert job is not None
        assert job["status"] == "failed"
        assert job["error_message"] == ZERO_TRADE_ERR

    def test_killed_is_not_overwritten_by_dying_process(
        self, jobs: BacktestJobManager, state: StateManager
    ) -> None:
        run_id = _run(state)
        jobs.start_job(run_id, "validation", SLEEPER)
        jobs.kill_job(run_id, force=True)
        jobs.mark_completed(run_id, error="KeyboardInterrupt")
        jobs.mark_completed(run_id)
        assert jobs.get_status(run_id) == JobStatus.KILLED
        run = state.get_backtest_run(run_id)
        assert run is not None and run["status"] == "killed"

    def test_relaunch_clears_previous_error(
        self, jobs: BacktestJobManager, state: StateManager
    ) -> None:
        run_id = _run(state)
        jobs.start_job(run_id, "validation", [sys.executable, "-c", "pass"])
        jobs.mark_completed(run_id, error=ZERO_TRADE_ERR)
        jobs.start_job(run_id, "validation", [sys.executable, "-c", "pass"])
        jobs.mark_completed(run_id)
        run = state.get_backtest_run(run_id)
        assert run is not None
        assert run["status"] == "completed"
        assert run["error_message"] is None


class TestKillDoesNotBlock:
    def test_kill_returns_immediately_and_escalates(
        self, jobs: BacktestJobManager, state: StateManager
    ) -> None:
        run_id = _run(state)
        stubborn = [
            sys.executable,
            "-c",
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
        ]
        pid = jobs.start_job(run_id, "validation", stubborn)
        time.sleep(0.5)  # let the child install its SIGTERM handler
        t0 = time.monotonic()
        assert jobs.kill_job(run_id, graceful_timeout=0.5) is True
        assert time.monotonic() - t0 < 0.3
        assert jobs.get_status(run_id) == JobStatus.KILLED
        assert jobs.is_process_alive(pid)  # ignored SIGTERM...
        assert _wait_dead(jobs, pid, timeout=5)  # ...SIGKILLed by escalation


class TestLogHandles:
    def test_start_job_does_not_leak_log_fds(
        self, jobs: BacktestJobManager, state: StateManager, tmp_path: Path
    ) -> None:
        _ = jobs.conn, state.conn
        before = len(os.listdir("/dev/fd"))
        for i in range(5):
            run_id = _run(state)
            jobs.start_job(
                run_id,
                "validation",
                [sys.executable, "-c", "pass"],
                log_file=str(tmp_path / f"job{i}.log"),
            )
        assert len(os.listdir("/dev/fd")) - before <= 0


class TestDataJobs:
    """Data jobs heartbeat for their whole lifetime and record completion."""

    def test_long_ingest_survives_cleanup_and_completes(
        self, db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        job_id = -123
        api = BacktestJobManager(db)  # the backend's manager
        try:
            pid = api.start_job(job_id, "data_ingest", SLEEPER)
            # Pretend the download has been running for 10 minutes already.
            old = (datetime.now(UTC) - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S")
            api.conn.execute(
                "UPDATE background_jobs SET heartbeat_at = ? WHERE run_id = ?", (old, job_id)
            )
            api.conn.commit()

            cleaned: list[int] = []

            def fake_data_cli(args: list[str]) -> int:
                assert args == ["ingest", "--symbols", "BTCUSDT"]
                # Wait for the wrapper's heartbeat, then the Backtest page's cleanup runs.
                for _ in range(50):
                    info = api.get_job_info(job_id)
                    assert info is not None
                    if not info.is_stale:
                        break
                    time.sleep(0.05)
                cleaned.append(api.cleanup_stale_jobs())
                return 0

            monkeypatch.setattr(data_job, "_run_data_cli", fake_data_cli)
            monkeypatch.setattr(
                data_job, "run_with_heartbeat", functools.partial(run_with_heartbeat, interval=0.05)
            )
            rc = data_job.main(
                ["--run-id", str(job_id), "--db", str(db), "--", "ingest", "--symbols", "BTCUSDT"]
            )
            assert rc == 0
            assert cleaned == [0]
            assert api.get_status(job_id) == JobStatus.COMPLETED
        finally:
            os.killpg(pid, signal.SIGKILL)
            api.close()

    def test_wrapper_subprocess_records_failure_exit_code(
        self, jobs: BacktestJobManager, db: Path
    ) -> None:
        job_id = -456
        cmd = [
            sys.executable, "-m", "vibe_quant.jobs.data_job", "--run-id", str(job_id),
            "--db", str(db), "--", "ingest", "--symbols", "BTCUSDT", "--start", "not-a-date",
        ]
        pid = jobs.start_job(job_id, "data_ingest", cmd)
        assert _wait_dead(jobs, pid, timeout=60)
        info = jobs.get_job_info(job_id)
        assert info is not None and info.status == JobStatus.FAILED
        row = jobs.conn.execute(
            "SELECT error_message FROM background_jobs WHERE run_id = ?", (job_id,)
        ).fetchone()
        assert row[0] == "data command exited with code 1"

    def test_wrapper_subprocess_records_success(self, jobs: BacktestJobManager, db: Path) -> None:
        job_id = -789
        cmd = [
            sys.executable, "-m", "vibe_quant.jobs.data_job", "--run-id", str(job_id),
            "--db", str(db), "--", "--help",
        ]
        pid = jobs.start_job(job_id, "data_update", cmd)
        assert _wait_dead(jobs, pid, timeout=60)
        assert jobs.get_status(job_id) == JobStatus.COMPLETED


class TestCliStatus:
    """CLI entry points must leave runs in a terminal, truthful status."""

    def test_validation_cli_zero_trades_ends_failed(
        self, db: Path, state: StateManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from vibe_quant.validation import __main__ as validation_cli

        run_id = _run(state)
        api = BacktestJobManager(db)
        api.start_job(run_id, "validation", [sys.executable, "-c", "pass"])
        api.close()

        class ZeroTradeRunner:
            def __init__(self, db_path: Path) -> None:
                self._state = StateManager(db_path)

            def run(self, run_id: int, **_: object) -> object:
                # What ValidationRunner.run does for a 0-trade backtest.
                self._state.update_backtest_run_status(
                    run_id, "failed", error_message=ZERO_TRADE_ERR
                )

                class Result:
                    sharpe_ratio = 0.0
                    total_return = 0.0

                return Result()

            def close(self) -> None:
                self._state.close()

        monkeypatch.setattr("vibe_quant.validation.runner.ValidationRunner", ZeroTradeRunner)
        monkeypatch.setattr("sys.argv", ["prog", "--run-id", str(run_id), "--db", str(db)])
        assert validation_cli.main() == 1

        run = state.get_backtest_run(run_id)
        assert run is not None
        assert (run["status"], run["error_message"]) == ("failed", ZERO_TRADE_ERR)
        job = state.get_job(run_id)
        assert job is not None
        assert (job["status"], job["error_message"]) == ("failed", ZERO_TRADE_ERR)

    def test_screening_early_exit_marks_run_failed(self, db: Path, state: StateManager) -> None:
        from vibe_quant.screening.__main__ import main as screening_main

        run_id = state.create_backtest_run(
            None, "screening", ["BTCUSDT"], "4h", "2025-01-01", "2025-06-01", {}
        )
        state.update_backtest_run_status(run_id, "running")
        assert screening_main(["run", "--run-id", str(run_id), "--db", str(db)]) == 1
        run = state.get_backtest_run(run_id)
        assert run is not None
        assert run["status"] == "failed"
        assert "has no strategy_id" in run["error_message"]

    def test_root_validation_run_accepts_db(self) -> None:
        from vibe_quant.__main__ import build_parser

        args = build_parser().parse_args(["validation", "run", "--run-id", "3", "--db", "x.db"])
        assert args.db == "x.db"
