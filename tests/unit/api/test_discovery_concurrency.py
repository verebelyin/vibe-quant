"""Tests for discovery concurrency guard (bd-5br0).

Prevents launching new discovery batches while existing ones are running,
avoiding resource contention that kills running processes.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TCH003
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture()
async def client(tmp_db: Path):
    app = create_app()
    state_mgr = StateManager(db_path=tmp_db)
    _ = state_mgr.conn

    job_mgr = BacktestJobManager(db_path=tmp_db)
    ws_mgr = ConnectionManager()
    await ws_mgr.start()

    app.state.state_manager = state_mgr
    app.state.job_manager = job_mgr
    app.state.ws_manager = ws_mgr

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, state_mgr, job_mgr

    await ws_mgr.stop()
    job_mgr.close()
    state_mgr.close()


def _insert_running_discovery(state: StateManager, job_mgr: BacktestJobManager, run_id_offset: int) -> int:
    """Insert a fake running discovery job directly into DB."""
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="discovery",
        symbols=["BTCUSDT"],
        timeframe="4h",
        start_date="2025-03-07",
        end_date="2026-03-07",
        parameters={"population": 12, "generations": 10},
    )
    # Insert running job record
    job_mgr.conn.execute(
        """INSERT INTO background_jobs
           (run_id, pid, job_type, status, started_at, heartbeat_at)
           VALUES (?, ?, 'discovery', 'running', datetime('now'), datetime('now'))""",
        (run_id, 99900 + run_id_offset),
    )
    job_mgr.conn.commit()
    return run_id


_LAUNCH_BODY = {
    "symbols": ["BTCUSDT"],
    "timeframes": ["4h"],
    "population": 12,
    "generations": 10,
    "mutation_rate": 0.15,
    "crossover_rate": 0.7,
    "elite_count": 2,
    "tournament_size": 3,
    "convergence_generations": 5,
}


async def test_launch_blocked_when_5_running(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """Launch returns 409 when 5 discovery jobs are already running."""
    ac, state, job_mgr = client

    # Insert 5 running discovery jobs
    with patch.object(job_mgr, "is_process_alive", return_value=True):
        for i in range(5):
            _insert_running_discovery(state, job_mgr, i)

        r = await ac.post("/api/discovery/launch", json=_LAUNCH_BODY)

    assert r.status_code == 409
    assert "5 discovery jobs already running" in r.json()["detail"]


async def test_launch_allowed_when_under_limit(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """Launch succeeds when fewer than 5 discovery jobs are running."""
    ac, state, job_mgr = client

    # Insert 4 running discovery jobs
    with patch.object(job_mgr, "is_process_alive", return_value=True):
        for i in range(4):
            _insert_running_discovery(state, job_mgr, i)

    # 5th launch should succeed (mock subprocess to avoid real process)
    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 12345
        r = await ac.post("/api/discovery/launch", json=_LAUNCH_BODY)

    assert r.status_code == 201


async def test_dead_processes_dont_block_launch(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """Dead processes (marked running but actually dead) get synced and don't block."""
    ac, state, job_mgr = client

    # Insert 5 running discovery jobs but all processes are dead
    for i in range(5):
        _insert_running_discovery(state, job_mgr, i)

    # is_process_alive returns False → sync marks them as failed → launch allowed
    with (
        patch.object(job_mgr, "is_process_alive", return_value=False),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 12345
        r = await ac.post("/api/discovery/launch", json=_LAUNCH_BODY)

    assert r.status_code == 201


async def test_list_jobs_syncs_dead_processes(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """GET /jobs syncs status for dead processes before returning."""
    ac, state, job_mgr = client

    run_id = _insert_running_discovery(state, job_mgr, 0)

    # Process is dead
    with patch.object(job_mgr, "is_process_alive", return_value=False):
        r = await ac.get("/api/discovery/jobs")

    assert r.status_code == 200
    jobs = r.json()
    # Find our job — it should be marked failed, not running
    our_job = next((j for j in jobs if j["run_id"] == run_id), None)
    assert our_job is not None
    assert our_job["status"] == "failed"
