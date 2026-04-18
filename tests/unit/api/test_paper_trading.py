"""Tests for /api/paper positions/orders/checkpoints/restore endpoints."""

from __future__ import annotations

from pathlib import Path  # noqa: TCH003
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager
from vibe_quant.paper.persistence import StateCheckpoint, StatePersistence


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture(autouse=True)
def _redirect_default_db(tmp_db: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the connection-default DB at tmp_db for this test module."""
    monkeypatch.setattr("vibe_quant.db.connection.DEFAULT_DB_PATH", tmp_db)
    yield


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


def _write_checkpoint(
    trader_id: str,
    *,
    positions: dict[str, object] | None = None,
    orders: dict[str, object] | None = None,
    state: str = "running",
    halt_reason: str | None = None,
) -> None:
    persistence = StatePersistence(trader_id=trader_id)
    cp = StateCheckpoint(
        trader_id=trader_id,
        positions=positions or {},
        orders=orders or {},
        balance={"total": 1000.0, "available": 900.0},
        node_status={"state": state, **({"halt_reason": halt_reason} if halt_reason else {})},
    )
    persistence.save_checkpoint(cp)
    persistence.close()


async def test_positions_empty_when_no_trader_id(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    ac, _state, _jobs = client
    r = await ac.get("/api/paper/positions")
    assert r.status_code == 200
    assert r.json() == []


async def test_positions_returns_from_checkpoint(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    ac, _state, _jobs = client
    trader_id = "PAPER-TEST-POS"
    _write_checkpoint(
        trader_id,
        positions={
            "p1": {
                "symbol": "BTCUSDT",
                "direction": "long",
                "quantity": 0.5,
                "entry_price": 50000.0,
                "unrealized_pnl": 120.5,
                "leverage": 10.0,
            }
        },
    )

    r = await ac.get(f"/api/paper/positions?trader_id={trader_id}")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["symbol"] == "BTCUSDT"
    assert data[0]["direction"] == "long"
    assert data[0]["quantity"] == 0.5
    assert data[0]["leverage"] == 10.0


async def test_orders_returns_from_checkpoint(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    ac, _state, _jobs = client
    trader_id = "PAPER-TEST-ORD"
    _write_checkpoint(
        trader_id,
        orders={
            "o1": {
                "symbol": "ETHUSDT",
                "side": "BUY",
                "quantity": 1.0,
                "price": 2500.0,
                "status": "NEW",
            }
        },
    )
    r = await ac.get(f"/api/paper/orders?trader_id={trader_id}")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["order_id"] == "o1"
    assert data[0]["symbol"] == "ETHUSDT"
    assert data[0]["side"] == "BUY"


async def test_checkpoints_newest_first(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    ac, _state, _jobs = client
    trader_id = "PAPER-TEST-CP"
    _write_checkpoint(trader_id, state="running")
    _write_checkpoint(trader_id, state="halted", halt_reason="max_dd")

    r = await ac.get(f"/api/paper/checkpoints?trader_id={trader_id}")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 2
    # Newest first: halted checkpoint was written second
    assert data[0]["state"] == "halted"
    assert data[0]["halt_reason"] == "max_dd"
    assert data[1]["state"] == "running"


async def test_checkpoints_empty_when_no_trader_id(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    ac, _state, _jobs = client
    r = await ac.get("/api/paper/checkpoints")
    assert r.status_code == 200
    assert r.json() == []


async def test_restore_404_when_no_config(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    ac, _state, _jobs = client
    r = await ac.post("/api/paper/restore", json={"trader_id": "NONEXISTENT"})
    assert r.status_code == 404


async def test_restore_409_when_active(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    ac, state, job_mgr = client

    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="paper",
        symbols=[],
        timeframe="1m",
        start_date="",
        end_date="",
        parameters={},
    )
    job_mgr.conn.execute(
        """INSERT INTO background_jobs
           (run_id, pid, job_type, status, started_at, heartbeat_at)
           VALUES (?, ?, 'paper', 'running', datetime('now'), datetime('now'))""",
        (run_id, 99123),
    )
    job_mgr.conn.commit()

    with patch.object(job_mgr, "is_process_alive", return_value=True):
        r = await ac.post("/api/paper/restore", json={"trader_id": "ANY"})
    assert r.status_code == 409
