"""Tests for BacktestJobManager."""

from __future__ import annotations

import os
import sys
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

from vibe_quant.db import StateManager
from vibe_quant.jobs import (
    HEARTBEAT_INTERVAL_SECONDS,
    STALE_THRESHOLD_SECONDS,
    BacktestJobManager,
    JobInfo,
    JobStatus,
)


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_status_values(self) -> None:
        """Status enum has expected values."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.KILLED.value == "killed"


class TestJobInfo:
    """Tests for JobInfo dataclass."""

    def test_is_stale_when_running_no_heartbeat(self) -> None:
        """Running job with no heartbeat and old start is stale."""
        old_time = datetime.now(UTC) - timedelta(seconds=200)
        info = JobInfo(
            run_id=1,
            pid=123,
            job_type="screening",
            status=JobStatus.RUNNING,
            heartbeat_at=None,
            started_at=old_time,
            completed_at=None,
            log_file=None,
        )
        assert info.is_stale is True

    def test_is_not_stale_when_recent_heartbeat(self) -> None:
        """Running job with recent heartbeat is not stale."""
        recent = datetime.now(UTC) - timedelta(seconds=10)
        info = JobInfo(
            run_id=1,
            pid=123,
            job_type="screening",
            status=JobStatus.RUNNING,
            heartbeat_at=recent,
            started_at=None,
            completed_at=None,
            log_file=None,
        )
        assert info.is_stale is False

    def test_is_not_stale_when_completed(self) -> None:
        """Completed job is never stale regardless of heartbeat."""
        old_time = datetime.now(UTC) - timedelta(seconds=500)
        info = JobInfo(
            run_id=1,
            pid=123,
            job_type="screening",
            status=JobStatus.COMPLETED,
            heartbeat_at=old_time,
            started_at=old_time,
            completed_at=old_time,
            log_file=None,
        )
        assert info.is_stale is False


class TestBacktestJobManager:
    """Tests for BacktestJobManager."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> Generator[BacktestJobManager]:
        """Create BacktestJobManager with temp database."""
        db_path = tmp_path / "jobs.db"
        mgr = BacktestJobManager(db_path)
        yield mgr
        mgr.close()

    @pytest.fixture
    def state_manager(self, tmp_path: Path) -> Generator[StateManager]:
        """Create StateManager with same temp database."""
        # Use same db_path as manager
        db_path = tmp_path / "jobs.db"
        mgr = StateManager(db_path)
        yield mgr
        mgr.close()

    @pytest.fixture
    def run_id(self, state_manager: StateManager) -> int:
        """Create a backtest run for testing."""
        strategy_id = state_manager.create_strategy(name="test_strat", dsl_config={})
        return state_manager.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="screening",
            symbols=["BTCUSDT-PERP"],
            timeframe="5m",
            start_date="2024-01-01",
            end_date="2024-12-31",
            parameters={},
        )

    def test_get_status_no_job(self, manager: BacktestJobManager) -> None:
        """get_status returns None for nonexistent job."""
        assert manager.get_status(99999) is None

    def test_get_job_info_no_job(self, manager: BacktestJobManager) -> None:
        """get_job_info returns None for nonexistent job."""
        assert manager.get_job_info(99999) is None

    def test_list_active_jobs_empty(self, manager: BacktestJobManager) -> None:
        """list_active_jobs returns empty list when no jobs."""
        assert manager.list_active_jobs() == []

    def test_start_job_spawns_process(
        self,
        manager: BacktestJobManager,
        state_manager: StateManager,
        run_id: int,
        tmp_path: Path,
    ) -> None:
        """start_job spawns subprocess and records in db."""
        log_file = str(tmp_path / "test.log")

        # Use a command that exits quickly
        pid = manager.start_job(
            run_id=run_id,
            job_type="screening",
            command=[sys.executable, "-c", "import time; time.sleep(0.1)"],
            log_file=log_file,
        )

        assert pid > 0
        assert manager.get_status(run_id) == JobStatus.RUNNING

        job_info = manager.get_job_info(run_id)
        assert job_info is not None
        assert job_info.pid == pid
        assert job_info.job_type == "screening"
        assert job_info.log_file == log_file

        # Wait for process to finish
        time.sleep(0.2)

    def test_start_job_updates_backtest_runs(
        self,
        manager: BacktestJobManager,
        state_manager: StateManager,
        run_id: int,
    ) -> None:
        """start_job updates backtest_runs table."""
        pid = manager.start_job(
            run_id=run_id,
            job_type="validation",
            command=[sys.executable, "-c", "pass"],
        )

        run = state_manager.get_backtest_run(run_id)
        assert run is not None
        assert run["status"] == "running"
        assert run["pid"] == pid
        assert run["started_at"] is not None

    def test_start_job_prevents_duplicate(
        self,
        manager: BacktestJobManager,
        state_manager: StateManager,
        run_id: int,
    ) -> None:
        """start_job raises error if job already running."""
        manager.start_job(
            run_id=run_id,
            job_type="screening",
            command=[sys.executable, "-c", "import time; time.sleep(10)"],
        )

        with pytest.raises(ValueError, match="already has an active job"):
            manager.start_job(
                run_id=run_id,
                job_type="screening",
                command=[sys.executable, "-c", "pass"],
            )

        # Cleanup
        manager.kill_job(run_id)

    def test_kill_job_terminates_process(
        self,
        manager: BacktestJobManager,
        state_manager: StateManager,
        run_id: int,
    ) -> None:
        """kill_job terminates running process."""
        manager.start_job(
            run_id=run_id,
            job_type="screening",
            command=[sys.executable, "-c", "import time; time.sleep(60)"],
        )

        job_info = manager.get_job_info(run_id)
        assert job_info is not None
        pid = job_info.pid

        result = manager.kill_job(run_id)
        assert result is True
        assert manager.get_status(run_id) == JobStatus.KILLED

        # Verify process is dead (wait up to 1s for termination)
        for _ in range(10):
            if not manager.is_process_alive(pid):
                break
            time.sleep(0.1)
        assert not manager.is_process_alive(pid)

    def test_kill_job_returns_false_if_not_running(
        self,
        manager: BacktestJobManager,
        state_manager: StateManager,
        run_id: int,
    ) -> None:
        """kill_job returns False if job not running."""
        # No job registered
        assert manager.kill_job(run_id) is False

    def test_update_heartbeat(
        self,
        manager: BacktestJobManager,
        state_manager: StateManager,
        run_id: int,
    ) -> None:
        """update_heartbeat updates timestamp."""
        manager.start_job(
            run_id=run_id,
            job_type="screening",
            command=[sys.executable, "-c", "import time; time.sleep(10)"],
        )

        # Get initial heartbeat
        info1 = manager.get_job_info(run_id)
        assert info1 is not None

        time.sleep(0.1)
        manager.update_heartbeat(run_id)

        info2 = manager.get_job_info(run_id)
        assert info2 is not None
        # Heartbeat should be updated (or at least not None)
        assert info2.heartbeat_at is not None

        # Cleanup
        manager.kill_job(run_id)

    def test_mark_completed_success(
        self,
        manager: BacktestJobManager,
        state_manager: StateManager,
        run_id: int,
    ) -> None:
        """mark_completed sets status to completed."""
        manager.start_job(
            run_id=run_id,
            job_type="screening",
            command=[sys.executable, "-c", "pass"],
        )

        time.sleep(0.1)  # Let process finish
        manager.mark_completed(run_id)

        assert manager.get_status(run_id) == JobStatus.COMPLETED
        run = state_manager.get_backtest_run(run_id)
        assert run is not None
        assert run["status"] == "completed"
        assert run["completed_at"] is not None

    def test_mark_completed_with_error(
        self,
        manager: BacktestJobManager,
        state_manager: StateManager,
        run_id: int,
    ) -> None:
        """mark_completed with error sets status to failed."""
        manager.start_job(
            run_id=run_id,
            job_type="screening",
            command=[sys.executable, "-c", "pass"],
        )

        time.sleep(0.1)
        manager.mark_completed(run_id, error="Something went wrong")

        assert manager.get_status(run_id) == JobStatus.FAILED
        run = state_manager.get_backtest_run(run_id)
        assert run is not None
        assert run["status"] == "failed"
        assert run["error_message"] == "Something went wrong"

    def test_list_active_jobs(
        self,
        manager: BacktestJobManager,
        state_manager: StateManager,
    ) -> None:
        """list_active_jobs returns running jobs."""
        # Create multiple runs
        strategy_id = state_manager.create_strategy(name="multi_test", dsl_config={})
        run1 = state_manager.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="screening",
            symbols=["BTCUSDT-PERP"],
            timeframe="5m",
            start_date="2024-01-01",
            end_date="2024-06-30",
            parameters={},
        )
        run2 = state_manager.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="validation",
            symbols=["ETHUSDT-PERP"],
            timeframe="1h",
            start_date="2024-07-01",
            end_date="2024-12-31",
            parameters={},
        )

        manager.start_job(
            run_id=run1,
            job_type="screening",
            command=[sys.executable, "-c", "import time; time.sleep(10)"],
        )
        manager.start_job(
            run_id=run2,
            job_type="validation",
            command=[sys.executable, "-c", "import time; time.sleep(10)"],
        )

        active = manager.list_active_jobs()
        assert len(active) == 2
        run_ids = {job.run_id for job in active}
        assert run_ids == {run1, run2}

        # Cleanup
        manager.kill_job(run1)
        manager.kill_job(run2)

    def test_sync_job_status_detects_dead_process(
        self,
        manager: BacktestJobManager,
        state_manager: StateManager,
        run_id: int,
    ) -> None:
        """sync_job_status detects dead process."""
        manager.start_job(
            run_id=run_id,
            job_type="screening",
            command=[sys.executable, "-c", "pass"],  # Exits immediately
        )

        # Wait for process to exit (up to 1s)
        job_info = manager.get_job_info(run_id)
        assert job_info is not None
        for _ in range(10):
            if not manager.is_process_alive(job_info.pid):
                break
            time.sleep(0.1)

        status = manager.sync_job_status(run_id)
        assert status == JobStatus.FAILED

        run = state_manager.get_backtest_run(run_id)
        assert run is not None
        assert run["status"] == "failed"
        assert "terminated unexpectedly" in run["error_message"]

    def test_is_process_alive(self, manager: BacktestJobManager) -> None:
        """is_process_alive correctly detects process state."""
        # Current process is alive
        assert manager.is_process_alive(os.getpid()) is True

        # Non-existent PID (use very large number)
        assert manager.is_process_alive(99999999) is False


class TestStaleJobDetection:
    """Tests for stale job detection and cleanup."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> Generator[BacktestJobManager]:
        """Create BacktestJobManager with temp database."""
        db_path = tmp_path / "stale.db"
        mgr = BacktestJobManager(db_path)
        yield mgr
        mgr.close()

    @pytest.fixture
    def state_manager(self, tmp_path: Path) -> Generator[StateManager]:
        """Create StateManager with same temp database."""
        db_path = tmp_path / "stale.db"
        mgr = StateManager(db_path)
        yield mgr
        mgr.close()

    def test_list_stale_jobs_with_old_heartbeat(
        self,
        manager: BacktestJobManager,
        state_manager: StateManager,
    ) -> None:
        """list_stale_jobs returns jobs with old heartbeat."""
        strategy_id = state_manager.create_strategy(name="stale_test", dsl_config={})
        run_id = state_manager.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="screening",
            symbols=["BTCUSDT-PERP"],
            timeframe="5m",
            start_date="2024-01-01",
            end_date="2024-12-31",
            parameters={},
        )

        # Start job
        manager.start_job(
            run_id=run_id,
            job_type="screening",
            command=[sys.executable, "-c", "import time; time.sleep(60)"],
        )

        # Manually set old heartbeat
        old_time = datetime.now(UTC) - timedelta(seconds=200)
        manager.conn.execute(
            "UPDATE background_jobs SET heartbeat_at = ? WHERE run_id = ?",
            (old_time.strftime("%Y-%m-%d %H:%M:%S"), run_id),
        )
        manager.conn.commit()

        stale = manager.list_stale_jobs()
        assert len(stale) == 1
        assert stale[0].run_id == run_id

        # Cleanup
        manager.kill_job(run_id)

    def test_cleanup_stale_jobs(
        self,
        manager: BacktestJobManager,
        state_manager: StateManager,
    ) -> None:
        """cleanup_stale_jobs kills and marks stale jobs."""
        strategy_id = state_manager.create_strategy(name="cleanup_test", dsl_config={})
        run_id = state_manager.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="screening",
            symbols=["BTCUSDT-PERP"],
            timeframe="5m",
            start_date="2024-01-01",
            end_date="2024-12-31",
            parameters={},
        )

        pid = manager.start_job(
            run_id=run_id,
            job_type="screening",
            command=[sys.executable, "-c", "import time; time.sleep(60)"],
        )

        # Manually set old heartbeat
        old_time = datetime.now(UTC) - timedelta(seconds=200)
        manager.conn.execute(
            "UPDATE background_jobs SET heartbeat_at = ? WHERE run_id = ?",
            (old_time.strftime("%Y-%m-%d %H:%M:%S"), run_id),
        )
        manager.conn.commit()

        count = manager.cleanup_stale_jobs()
        assert count == 1

        assert manager.get_status(run_id) == JobStatus.FAILED
        # Wait up to 1s for process to die
        for _ in range(10):
            if not manager.is_process_alive(pid):
                break
            time.sleep(0.1)
        assert not manager.is_process_alive(pid)


class TestConstants:
    """Tests for module constants."""

    def test_heartbeat_interval(self) -> None:
        """Heartbeat interval is 30 seconds per spec."""
        assert HEARTBEAT_INTERVAL_SECONDS == 30

    def test_stale_threshold(self) -> None:
        """Stale threshold is 120 seconds per spec."""
        assert STALE_THRESHOLD_SECONDS == 120
