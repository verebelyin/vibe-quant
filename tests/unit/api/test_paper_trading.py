"""Tests for /api/paper positions/orders/checkpoints/restore endpoints."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TCH003
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager
from vibe_quant.paper.persistence import StateCheckpoint, StatePersistence

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


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


# ------------------------------------------------------ trade_alerts (ci7q9)

DSL = {
    "name": "test_strategy",
    "timeframe": "1h",
    "indicators": {"rsi_14": {"type": "RSI", "period": 14}},
    "entry_conditions": {"long": ["rsi_14 < 30"]},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 4.0},
}


class StartCtx:
    def __init__(
        self, ac: AsyncClient, state: StateManager, jobs: BacktestJobManager, tmp: Path
    ) -> None:
        self.ac = ac
        self.state = state
        self.jobs = jobs
        self.tmp = tmp
        self.launched: list[list[str]] = []
        self.strategy_id = state.create_strategy("test_strategy", DSL)
        self.validation_run = state.create_backtest_run(
            strategy_id=self.strategy_id,
            run_mode="validation",
            symbols=["BTCUSDT"],
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-12-31",
            parameters={},
        )
        state.update_backtest_run_status(self.validation_run, "completed")

    def config_for(self, run_id: int) -> dict[str, Any]:
        path = self.tmp / "paper_configs" / f"paper_{run_id}.json"
        data: dict[str, Any] = json.loads(path.read_text())
        return data


@pytest.fixture()
def api_env(tmp_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        "vibe_quant.api.routers.paper_trading.PAPER_CONFIG_DIR", tmp_path / "paper_configs"
    )
    monkeypatch.setenv("BINANCE_TESTNET_API_KEY", "tk")
    monkeypatch.setenv("BINANCE_TESTNET_API_SECRET", "ts")
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    return tmp_db


@pytest.fixture()
async def start_ctx(
    api_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[StartCtx]:
    app = create_app()
    state = StateManager(db_path=api_env)
    _ = state.conn
    jobs = BacktestJobManager(db_path=api_env)
    ws = ConnectionManager()
    await ws.start()
    app.state.state_manager = state
    app.state.job_manager = jobs
    app.state.ws_manager = ws
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        c = StartCtx(ac, state, jobs, tmp_path)

        def fake_start_job(
            run_id: int, job_type: str, command: list[str], log_file: str | None = None
        ) -> int:
            c.launched.append(command)
            return 4242

        monkeypatch.setattr(jobs, "start_job", fake_start_job)
        yield c
    await ws.stop()
    jobs.close()
    state.close()


@pytest.mark.parametrize("flag", ["start_fresh", "confirm_fresh_live", "trade_alerts"])
@pytest.mark.parametrize("bad", ["true", 1, None])
async def test_start_rejects_non_bool_flags(
    start_ctx: StartCtx, flag: str, bad: object
) -> None:
    """T6/R21: strict bool — a string, an int and an explicit null are 422.

    The absent default is the only "false" spelling: the three strict flags all
    refuse coercion instead of turning "true"/1 into a confirmation, and nothing
    is launched for a rejected body.
    """
    r = await start_ctx.ac.post(
        "/api/paper/start",
        json={"validation_run_id": start_ctx.validation_run, flag: bad},
    )
    assert r.status_code == 422, r.text
    assert start_ctx.launched == []


async def test_start_records_trade_alerts_in_params_and_config(start_ctx: StartCtx) -> None:
    """T6: trade_alerts travels into the run params AND the launched config JSON."""
    from vibe_quant.paper.cli import load_config_from_json

    r = await start_ctx.ac.post(
        "/api/paper/start",
        json={"validation_run_id": start_ctx.validation_run, "trade_alerts": True},
    )
    assert r.status_code == 201, r.text
    run_id = r.json()["run_id"]
    run = start_ctx.state.get_backtest_run(run_id)
    assert run is not None
    assert run["parameters"]["trade_alerts"] is True
    assert start_ctx.config_for(run_id)["trade_alerts"] is True
    assert load_config_from_json(start_ctx.tmp / "paper_configs" / f"paper_{run_id}.json").trade_alerts

    # Absent -> False everywhere: the field is not optional-true.
    r2 = await start_ctx.ac.post("/api/paper/start", json={"validation_run_id": start_ctx.validation_run})
    assert r2.status_code == 201, r2.text
    run_id2 = r2.json()["run_id"]
    run2 = start_ctx.state.get_backtest_run(run_id2)
    assert run2 is not None
    assert run2["parameters"]["trade_alerts"] is False
    assert start_ctx.config_for(run_id2)["trade_alerts"] is False


@pytest.mark.parametrize("saved", [True, False, None], ids=["true", "false", "old-config"])
async def test_restore_keeps_trade_alerts(start_ctx: StartCtx, saved: bool | None) -> None:
    """T6: /restore keeps the saved value; a config written before the key is False."""
    from vibe_quant.paper.cli import load_config_from_json

    config_dir = start_ctx.tmp / "paper_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "trader_id": "PAPER-900",
        "strategy_id": start_ctx.strategy_id,
        "validation_run_id": start_ctx.validation_run,
        "binance": {"testnet": True},
        "symbols": ["BTCUSDT"],
    }
    expected = False
    if saved is not None:
        data["trade_alerts"] = saved
        expected = saved
    (config_dir / "paper_900.json").write_text(json.dumps(data))

    r = await start_ctx.ac.post("/api/paper/restore", json={"trader_id": "PAPER-900"})
    assert r.status_code == 201, r.text
    run_id = r.json()["run_id"]
    assert start_ctx.config_for(run_id)["trade_alerts"] is expected
    path = start_ctx.tmp / "paper_configs" / f"paper_{run_id}.json"
    assert load_config_from_json(path).trade_alerts is expected
