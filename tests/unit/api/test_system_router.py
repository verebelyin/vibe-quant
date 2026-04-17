"""Tests for the /api/system router (kill switch)."""

from __future__ import annotations

from pathlib import Path  # noqa: TCH003

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.data.catalog import CatalogManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "system_test.db"


@pytest.fixture()
async def client(tmp_db: Path):
    app = create_app()
    state_mgr = StateManager(db_path=tmp_db)
    _ = state_mgr.conn  # init schema
    job_mgr = BacktestJobManager()
    catalog_mgr = CatalogManager()
    ws_mgr = ConnectionManager()
    await ws_mgr.start()

    app.state.state_manager = state_mgr
    app.state.job_manager = job_mgr
    app.state.catalog_manager = catalog_mgr
    app.state.ws_manager = ws_mgr

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    await ws_mgr.stop()
    job_mgr.close()
    state_mgr.close()


async def test_status_starts_unkilled(client: AsyncClient) -> None:
    r = await client.get("/api/system/status")
    assert r.status_code == 200
    body = r.json()
    assert body["kill_switch"] is False
    assert body["reason"] is None
    assert body["killed_at"] is None


async def test_kill_sets_flag_and_reason(client: AsyncClient) -> None:
    r = await client.post(
        "/api/system/kill",
        json={"reason": "divergence detected", "killed_by": "operator"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["kill_switch"] is True
    assert body["reason"] == "divergence detected"
    assert body["killed_by"] == "operator"
    assert body["killed_at"] is not None

    # Status reflects kill across requests (DB-persisted)
    r2 = await client.get("/api/system/status")
    assert r2.json()["kill_switch"] is True


async def test_unlock_requires_acknowledge(client: AsyncClient) -> None:
    await client.post("/api/system/kill", json={"reason": "test"})

    r = await client.post("/api/system/unlock", json={"acknowledge": False})
    assert r.status_code == 400

    # Still killed
    assert (await client.get("/api/system/status")).json()["kill_switch"] is True


async def test_unlock_clears_flag(client: AsyncClient) -> None:
    await client.post("/api/system/kill", json={"reason": "test"})
    r = await client.post(
        "/api/system/unlock", json={"acknowledge": True, "cleared_by": "oncall"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["kill_switch"] is False
    assert body["reason"] is None


async def test_paper_start_refuses_when_killed(client: AsyncClient) -> None:
    """Engaging the kill switch must block new paper sessions (423 Locked)."""
    await client.post("/api/system/kill", json={"reason": "kill gate test"})

    # Minimal valid paper-start payload — fields beyond strategy_id are optional.
    r = await client.post("/api/paper/start", json={"strategy_id": 1})
    assert r.status_code == 423
    assert "kill gate test" in r.json()["detail"]


async def test_kill_idempotent(client: AsyncClient) -> None:
    """Re-killing updates the reason instead of stacking state."""
    await client.post("/api/system/kill", json={"reason": "first"})
    r = await client.post("/api/system/kill", json={"reason": "second"})
    assert r.json()["reason"] == "second"
