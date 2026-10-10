"""Startup reaper: fail backtest_runs stuck pending/running with no live job.

``reconcile_jobs`` only visits runs that have a ``background_jobs`` row, so a
run whose job row is missing (or is not running) stays pending/running forever
and shows as an in-flight job. ``reap_orphan_runs`` closes those out at startup.
"""

from __future__ import annotations

import functools
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

SLEEPER = [sys.executable, "-c", "import time; time.sleep(30)"]


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "reap.db"


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


def _run(state: StateManager, status: str = "pending") -> int:
    sid = state.create_strategy(f"s{time.monotonic_ns()}", {"name": "s"})
    run_id = state.create_backtest_run(
        sid, "validation", ["BTCUSDT"], "4h", "2025-01-01", "2025-06-01", {}
    )
    if status != "pending":
        state.update_backtest_run_status(run_id, status)
    return run_id


def _wait_dead(jobs: BacktestJobManager, pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not jobs.is_process_alive(pid):
            return True
        time.sleep(0.05)
    return False


class TestReapOrphanRuns:
    def test_reaps_only_runs_without_a_live_job(
        self, jobs: BacktestJobManager, state: StateManager
    ) -> None:
        pending_no_job = _run(state, "pending")
        running_no_job = _run(state, "running")

        dead = _run(state, "running")
        dead_pid = jobs.start_job(dead, "validation", [sys.executable, "-c", "pass"])
        assert _wait_dead(jobs, dead_pid)

        live = _run(state, "running")
        jobs.start_job(live, "validation", SLEEPER)

        completed = _run(state, "completed")

        # Lifespan order: reconcile first (closes the dead-but-running job),
        # then reap whatever is still pending/running with no live job.
        assert jobs.reconcile_jobs() == 1
        reaped = jobs.reap_orphan_runs()

        assert set(reaped) == {pending_no_job, running_no_job}

        pending = state.get_backtest_run(pending_no_job)
        assert pending is not None and pending["status"] == "failed"
        assert pending["error_message"] == "reaped at startup: no live job (was pending)"
        assert pending["completed_at"] is not None

        running = state.get_backtest_run(running_no_job)
        assert running is not None and running["status"] == "failed"
        assert running["error_message"] == "reaped at startup: no live job (was running)"

        # reconcile owned the dead job: the run is failed too, but by reconcile.
        dead_run = state.get_backtest_run(dead)
        assert dead_run is not None and dead_run["status"] == "failed"
        assert dead_run["error_message"]

        live_run = state.get_backtest_run(live)
        assert live_run is not None and live_run["status"] == "running"
        assert live_run["error_message"] is None
        assert jobs.get_status(live) is not None

        done = state.get_backtest_run(completed)
        assert done is not None and done["status"] == "completed"

    def test_reap_is_idempotent(self, jobs: BacktestJobManager, state: StateManager) -> None:
        run_id = _run(state, "pending")

        assert jobs.reap_orphan_runs() == [run_id]
        assert jobs.reap_orphan_runs() == []

        run = state.get_backtest_run(run_id)
        assert run is not None and run["status"] == "failed"

    def test_reap_appends_to_existing_error_message(
        self, jobs: BacktestJobManager, state: StateManager
    ) -> None:
        run_id = _run(state, "running")
        state.conn.execute(
            "UPDATE backtest_runs SET error_message = 'prior failure' WHERE id = ?", (run_id,)
        )
        state.conn.commit()

        assert jobs.reap_orphan_runs() == [run_id]

        run = state.get_backtest_run(run_id)
        assert run is not None
        assert run["error_message"] == "prior failure\nreaped at startup: no live job (was running)"

    def test_reap_spares_a_run_with_a_running_job(
        self, jobs: BacktestJobManager, state: StateManager
    ) -> None:
        run_id = _run(state, "running")
        jobs.start_job(run_id, "validation", SLEEPER)

        assert jobs.reap_orphan_runs() == []

        run = state.get_backtest_run(run_id)
        assert run is not None and run["status"] == "running"

    def test_reap_spares_run_with_live_own_pid(
        self, jobs: BacktestJobManager, state: StateManager
    ) -> None:
        """CLI/auto_screen/campaign runs write backtest_runs.pid with no job row.

        A backend restart while such a run is live must not fail it: the run's
        own PID is alive, so it is spared even without a background_jobs row.
        """
        run_id = _run(state, "running")
        state.conn.execute(
            "UPDATE backtest_runs SET pid = ?, heartbeat_at = NULL WHERE id = ?",
            (os.getpid(), run_id),
        )
        state.conn.commit()

        assert jobs.reap_orphan_runs() == []

        run = state.get_backtest_run(run_id)
        assert run is not None and run["status"] == "running"
        assert run["error_message"] is None

    def test_reap_spares_run_with_fresh_heartbeat(
        self, jobs: BacktestJobManager, state: StateManager
    ) -> None:
        """A run heartbeating within STALE_THRESHOLD_SECONDS is still live."""
        run_id = _run(state, "running")
        state.conn.execute(
            "UPDATE backtest_runs SET pid = NULL, heartbeat_at = datetime('now') WHERE id = ?",
            (run_id,),
        )
        state.conn.commit()

        assert jobs.reap_orphan_runs() == []

        run = state.get_backtest_run(run_id)
        assert run is not None and run["status"] == "running"

    def test_reap_reaps_run_with_dead_pid_and_stale_heartbeat(
        self, jobs: BacktestJobManager, state: StateManager
    ) -> None:
        """Sparing is not blanket: a dead PID + old heartbeat is still reaped."""
        stale = (datetime.now(UTC) - timedelta(seconds=10 * 120)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        run_id = _run(state, "running")
        state.conn.execute(
            "UPDATE backtest_runs SET pid = 99999999, heartbeat_at = ? WHERE id = ?",
            (stale, run_id),
        )
        state.conn.commit()

        assert jobs.reap_orphan_runs() == [run_id]

        run = state.get_backtest_run(run_id)
        assert run is not None and run["status"] == "failed"

    def test_reap_does_not_overwrite_a_run_completed_mid_loop(
        self, jobs: BacktestJobManager, state: StateManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TOCTOU: a job that finishes between the candidate SELECT and the
        per-run UPDATE must not be relabelled 'failed'."""
        run_id = _run(state, "running")
        # A running job row means the reaper reaches the liveness check.
        jobs.conn.execute(
            """INSERT INTO background_jobs (run_id, pid, pid_start_time, job_type, status)
               VALUES (?, 99999999, 'darwin:1.000000', 'validation', 'running')""",
            (run_id,),
        )
        jobs.conn.commit()

        def _alive_then_complete(
            _self: BacktestJobManager, _pid: int, _identity: str | None
        ) -> bool:
            # The surviving job finished while the reaper was scanning: the run
            # (and its own liveness read) is now completed/dead.
            state.update_backtest_run_status(run_id, "completed")
            return False

        monkeypatch.setattr(BacktestJobManager, "_job_alive", _alive_then_complete)

        assert jobs.reap_orphan_runs() == []

        run = state.get_backtest_run(run_id)
        assert run is not None and run["status"] == "completed"


class TestLifespanReapWiring:
    """The reaper must actually run from the app's startup lifespan."""

    def test_lifespan_reaps_orphan_run(
        self, db: Path, state: StateManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fastapi.testclient import TestClient

        from vibe_quant.api import app as app_module

        run_id = _run(state, "pending")

        monkeypatch.setattr(app_module, "StateManager", functools.partial(StateManager, db))
        monkeypatch.setattr(
            app_module, "BacktestJobManager", functools.partial(BacktestJobManager, db)
        )

        with TestClient(app_module.create_app()):
            run = state.get_backtest_run(run_id)
            assert run is not None and run["status"] == "failed"
            assert run["error_message"] == "reaped at startup: no live job (was pending)"
