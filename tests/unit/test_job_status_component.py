"""Tests for job status dashboard helpers."""

from __future__ import annotations

from vibe_quant.dashboard.components.job_status import _sync_recent_running_runs


class _FakeManager:
    def __init__(self, runs: list[dict]) -> None:
        self._runs = runs

    def list_backtest_runs(self) -> list[dict]:
        return [dict(r) for r in self._runs]

    def set_status(self, run_id: int, status: str) -> None:
        for run in self._runs:
            if run["id"] == run_id:
                run["status"] = status
                return


class _FakeJobManager:
    def __init__(self, manager: _FakeManager) -> None:
        self._manager = manager
        self.synced: list[int] = []

    def sync_job_status(self, run_id: int) -> None:
        self.synced.append(run_id)
        # Simulate stale running job being marked as failed after sync.
        self._manager.set_status(run_id, "failed")


def test_sync_recent_runs_updates_running_status() -> None:
    """Running rows should be synced and refreshed before rendering."""
    manager = _FakeManager(
        runs=[
            {"id": 1, "status": "running"},
            {"id": 2, "status": "completed"},
        ]
    )
    jobs = _FakeJobManager(manager)

    updated = _sync_recent_running_runs(manager, jobs, limit=10)

    assert jobs.synced == [1]
    assert updated[0]["status"] == "failed"
    assert updated[1]["status"] == "completed"


def test_sync_recent_runs_skips_non_running() -> None:
    """No sync calls should occur when nothing is running."""
    manager = _FakeManager(
        runs=[
            {"id": 3, "status": "failed"},
            {"id": 4, "status": "completed"},
        ]
    )
    jobs = _FakeJobManager(manager)

    _ = _sync_recent_running_runs(manager, jobs, limit=10)

    assert jobs.synced == []
