"""/api/paper + /api/system paper control (vibe-quant-e70tl.4 / .19).

The paper node is simulated by a task that drains the ``paper_commands``
queue exactly like ``PaperTradingNode.process_commands`` does, so these tests
exercise the real API <-> node protocol without spawning a process.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.schemas.paper_trading import PaperStartRequest
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager
from vibe_quant.paper.config import is_valid_trader_id
from vibe_quant.paper.persistence import PaperCommandQueue, StateCheckpoint, StatePersistence

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

DSL = {
    "name": "test_strategy",
    "timeframe": "1h",
    "indicators": {"rsi_14": {"type": "RSI", "period": 14}},
    "entry_conditions": {"long": ["rsi_14 < 30"]},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 4.0},
}


@pytest.fixture()
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "api.db"
    monkeypatch.setattr("vibe_quant.db.connection.DEFAULT_DB_PATH", db)
    monkeypatch.setattr(
        "vibe_quant.api.routers.paper_trading.PAPER_CONFIG_DIR", tmp_path / "paper_configs"
    )
    monkeypatch.setenv("BINANCE_TESTNET_API_KEY", "tk")
    monkeypatch.setenv("BINANCE_TESTNET_API_SECRET", "ts")
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    return db


class Ctx:
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
            parameters={"rsi_14_period": 21},
        )
        state.update_backtest_run_status(self.validation_run, "completed")

    def config_for(self, run_id: int) -> dict[str, Any]:
        path = self.tmp / "paper_configs" / f"paper_{run_id}.json"
        data: dict[str, Any] = json.loads(path.read_text())
        return data

    def activate(self, run_id: int) -> None:
        """Mark a paper run's job as running (what start_job would record)."""
        self.jobs.conn.execute(
            """INSERT OR REPLACE INTO background_jobs
               (run_id, pid, job_type, status, started_at, heartbeat_at)
               VALUES (?, 4242, 'paper', 'running', datetime('now'), datetime('now'))""",
            (run_id,),
        )
        self.jobs.conn.commit()


@pytest.fixture()
async def ctx(tmp_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Ctx]:
    app = create_app()
    state = StateManager(db_path=tmp_db)
    _ = state.conn
    jobs = BacktestJobManager(db_path=tmp_db)
    ws = ConnectionManager()
    await ws.start()
    app.state.state_manager = state
    app.state.job_manager = jobs
    app.state.ws_manager = ws
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        c = Ctx(ac, state, jobs, tmp_path)

        def fake_start_job(
            run_id: int, job_type: str, command: list[str], log_file: str | None = None
        ) -> int:
            c.launched.append(command)
            c.activate(run_id)
            return 4242

        monkeypatch.setattr(jobs, "start_job", fake_start_job)
        yield c
    await ws.stop()
    jobs.close()
    state.close()


# ------------------------------------------------------------------- /start


def test_start_request_defaults_to_testnet() -> None:
    assert PaperStartRequest().testnet is True
    assert PaperStartRequest().confirm_live is False


async def test_start_from_validation_run_writes_runnable_config(ctx: Ctx) -> None:
    r = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})
    assert r.status_code == 201, r.text
    body = r.json()
    run_id = body["run_id"]
    assert body["trader_id"] == f"PAPER-{run_id:03d}"
    assert is_valid_trader_id(body["trader_id"])
    assert body["testnet"] is True

    cfg = ctx.config_for(run_id)
    assert cfg["symbols"] == ["BTCUSDT"], "audit: /start wrote symbols: [] (always failed)"
    assert cfg["strategy_id"] == ctx.strategy_id
    assert cfg["validation_run_id"] == ctx.validation_run
    assert cfg["binance"] == {"testnet": True}
    assert cfg["db_path"].endswith("api.db")
    assert "api_key" not in json.dumps(cfg)
    assert "--live" not in ctx.launched[0]

    # The CLI loads it into a config that validates (testnet env creds).
    from vibe_quant.paper.cli import load_config_from_json

    loaded = load_config_from_json(ctx.tmp / "paper_configs" / f"paper_{run_id}.json")
    assert loaded.validate() == []
    assert loaded.binance.api_key == "tk"

    run = ctx.state.get_backtest_run(run_id)
    assert run is not None and run["symbols"] == ["BTCUSDT"]


async def test_second_start_while_active_is_409(ctx: Ctx) -> None:
    r1 = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})
    assert r1.status_code == 201
    r2 = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})
    assert r2.status_code == 409
    assert len(ctx.launched) == 1


async def test_live_requires_confirmation_and_live_env(
    ctx: Ctx, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = await ctx.ac.post(
        "/api/paper/start", json={"validation_run_id": ctx.validation_run, "testnet": False}
    )
    assert r.status_code == 400 and "confirm_live" in r.text
    r = await ctx.ac.post(
        "/api/paper/start",
        json={"validation_run_id": ctx.validation_run, "testnet": False, "confirm_live": True},
    )
    assert r.status_code == 400 and "BINANCE_API_KEY" in r.text, r.text
    assert ctx.launched == []

    monkeypatch.setenv("BINANCE_API_KEY", "lk")
    monkeypatch.setenv("BINANCE_API_SECRET", "ls")
    r = await ctx.ac.post(
        "/api/paper/start",
        json={"validation_run_id": ctx.validation_run, "testnet": False, "confirm_live": True},
    )
    assert r.status_code == 201, r.text
    assert ctx.launched[0][-1] == "--live"
    assert ctx.config_for(r.json()["run_id"])["binance"] == {"testnet": False}


async def test_missing_testnet_env_is_400(ctx: Ctx, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BINANCE_TESTNET_API_KEY")
    r = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})
    assert r.status_code == 400 and "BINANCE_TESTNET_API_KEY" in r.text


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        ({"strategy_id": 1, "api_key": "K", "api_secret": "S", "symbols": ["BTCUSDT"]}, 422),
        ({"strategy_id": 1, "trader_id": "paper_1", "symbols": ["BTCUSDT"]}, 422),
        ({"strategy_id": 1, "symbols": ["BTCUSDT"], "risk_per_trade": 2}, 422),
        ({"strategy_id": 1, "symbols": ["BTCUSDT"], "sizing_method": "kelly"}, 422),
        ({"strategy_id": 1}, 400),
        ({}, 400),
    ],
)
async def test_start_rejects_bad_requests(ctx: Ctx, payload: dict[str, Any], status: int) -> None:
    if "strategy_id" in payload:
        payload = {**payload, "strategy_id": ctx.strategy_id}
    r = await ctx.ac.post("/api/paper/start", json=payload)
    assert r.status_code == status, r.text
    assert ctx.launched == []


async def test_start_without_validation_run_uses_given_symbols(ctx: Ctx) -> None:
    r = await ctx.ac.post(
        "/api/paper/start",
        json={"strategy_id": ctx.strategy_id, "symbols": ["ethusdt"], "risk_per_trade": 0.01},
    )
    assert r.status_code == 201, r.text
    cfg = ctx.config_for(r.json()["run_id"])
    assert cfg["symbols"] == ["ETHUSDT"]
    assert cfg["validation_run_id"] is None
    assert cfg["sizing"] == {"risk_per_trade": "0.01"}


# ------------------------------------------------------------- control


async def _fake_node(db: Path, trader_id: str, handler: Any, stop: asyncio.Event) -> None:
    q = PaperCommandQueue(db)
    try:
        while not stop.is_set():
            for cmd in q.claim_pending(trader_id):
                ok, result, error = handler(cmd)
                if ok:
                    q.complete(cmd.id, result)
                else:
                    q.fail(cmd.id, error, result)
            await asyncio.sleep(0.02)
    finally:
        q.close()


async def _start(ctx: Ctx) -> tuple[int, str]:
    r = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})
    assert r.status_code == 201
    return r.json()["run_id"], r.json()["trader_id"]


async def test_close_all_returns_node_errors_as_502(ctx: Ctx, tmp_db: Path) -> None:
    _run_id, trader_id = await _start(ctx)
    seen: list[str] = []

    def handler(cmd: Any) -> tuple[bool, dict[str, Any], str]:
        seen.append(cmd.command)
        return (
            False,
            {"targeted_positions": ["P-1"], "still_open": ["P-1"], "errors": ["venue down"]},
            "venue down; positions still open after 10s: ['P-1']",
        )

    stop = asyncio.Event()
    task = asyncio.create_task(_fake_node(tmp_db, trader_id, handler, stop))
    try:
        r = await ctx.ac.post("/api/paper/close-all-positions")
    finally:
        stop.set()
        await task
    assert seen == ["close_all"]
    assert r.status_code == 502
    detail = r.json()["detail"]
    assert "venue down" in detail["error"]
    assert detail["result"]["still_open"] == ["P-1"]


async def test_close_all_success(ctx: Ctx, tmp_db: Path) -> None:
    _run_id, trader_id = await _start(ctx)

    def handler(cmd: Any) -> tuple[bool, dict[str, Any], None]:
        return True, {"targeted_positions": ["P-1", "P-2"], "still_open": [], "errors": []}, None

    stop = asyncio.Event()
    task = asyncio.create_task(_fake_node(tmp_db, trader_id, handler, stop))
    try:
        r = await ctx.ac.post("/api/paper/close-all-positions")
    finally:
        stop.set()
        await task
    assert r.status_code == 200, r.text
    assert r.json()["targeted_positions"] == ["P-1", "P-2"]
    assert r.json()["still_open"] == []


async def test_halt_and_pause_commands(ctx: Ctx, tmp_db: Path) -> None:
    _run_id, trader_id = await _start(ctx)
    seen: list[str] = []

    def handler(cmd: Any) -> tuple[bool, dict[str, Any], None]:
        seen.append(cmd.command)
        return True, {"state": "halted" if cmd.command == "halt" else "paused"}, None

    stop = asyncio.Event()
    task = asyncio.create_task(_fake_node(tmp_db, trader_id, handler, stop))
    try:
        r1 = await ctx.ac.post("/api/paper/halt")
        r2 = await ctx.ac.post("/api/paper/halt?mode=pause")
    finally:
        stop.set()
        await task
    assert seen == ["halt", "pause"]
    assert r1.json()["state"] == "halted"
    assert r2.json()["state"] == "paused"


async def test_unacknowledged_command_times_out_and_expires(
    ctx: Ctx, tmp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("vibe_quant.api.routers.paper_trading.COMMAND_TIMEOUT_SECS", 0.3)
    _run_id, trader_id = await _start(ctx)
    r = await ctx.ac.post("/api/paper/halt")
    assert r.status_code == 504
    assert "cancelled" in r.json()["detail"]
    q = PaperCommandQueue(tmp_db)
    try:
        assert q.claim_pending(trader_id) == [], "expired command must not run later"
    finally:
        q.close()


async def test_resume_refused_while_killed(ctx: Ctx) -> None:
    await _start(ctx)
    ctx.state.set_kill_switch("ops")
    r = await ctx.ac.post("/api/paper/resume")
    assert r.status_code == 423


async def test_kill_queues_kill_for_active_session(ctx: Ctx, tmp_db: Path) -> None:
    _run_id, trader_id = await _start(ctx)
    r = await ctx.ac.post("/api/system/kill", json={"reason": "divergence"})
    assert r.status_code == 200
    q = PaperCommandQueue(tmp_db)
    try:
        cmds = q.claim_pending(trader_id)
    finally:
        q.close()
    assert [c.command for c in cmds] == ["kill"]
    assert "divergence" in cmds[0].payload["message"]


# ----------------------------------------------------------- read endpoints


async def test_status_reports_node_state_from_checkpoint(ctx: Ctx, tmp_db: Path) -> None:
    run_id, trader_id = await _start(ctx)
    r = await ctx.ac.get("/api/paper/status")
    assert r.json()["state"] == "starting"
    p = StatePersistence(tmp_db, trader_id)
    p.save_checkpoint(
        StateCheckpoint(
            trader_id=trader_id,
            node_status={
                "state": "halted",
                "halt_reason": "max_drawdown",
                "error_message": "drawdown 6.00% >= limit 5.00%",
                "risk": {"drawdown_pct": "0.06"},
            },
        )
    )
    p.close()
    body = (await ctx.ac.get("/api/paper/status")).json()
    assert body["state"] == "halted"
    assert body["halt_reason"] == "max_drawdown"
    assert body["run_id"] == run_id and body["trader_id"] == trader_id
    assert body["pnl_metrics"] == {"drawdown_pct": "0.06"}


async def test_positions_and_orders_map_nautilus_to_dict(ctx: Ctx, tmp_db: Path) -> None:
    """Audit: UI showed symbol '' and entry 0 for real Position/Order.to_dict() data."""
    p = StatePersistence(tmp_db, "PAPER-042")
    p.save_checkpoint(
        StateCheckpoint(
            trader_id="PAPER-042",
            positions={
                "BTCUSDT-PERP.BINANCE-S-000": {
                    "position_id": "BTCUSDT-PERP.BINANCE-S-000",
                    "instrument_id": "BTCUSDT-PERP.BINANCE",
                    "side": "LONG",
                    "quantity": "0.010",
                    "avg_px_open": 60000.0,
                    "unrealized_pnl": -12.5,
                    "leverage": 5,
                }
            },
            orders={
                "O-1": {
                    "instrument_id": "BTCUSDT-PERP.BINANCE",
                    "type": "STOP_MARKET",
                    "side": "SELL",
                    "quantity": "0.010",
                    "trigger_price": "59000.0",
                    "status": "ACCEPTED",
                    "is_reduce_only": True,
                }
            },
        )
    )
    p.close()
    pos = (await ctx.ac.get("/api/paper/positions?trader_id=PAPER-042")).json()
    assert pos == [
        {
            "symbol": "BTCUSDT-PERP.BINANCE",
            "direction": "LONG",
            "quantity": 0.01,
            "entry_price": 60000.0,
            "unrealized_pnl": -12.5,
            "leverage": 5.0,
        }
    ]
    orders = (await ctx.ac.get("/api/paper/orders?trader_id=PAPER-042")).json()
    assert orders == [
        {
            "order_id": "O-1",
            "symbol": "BTCUSDT-PERP.BINANCE",
            "side": "SELL",
            "quantity": 0.01,
            "price": 59000.0,
            "status": "ACCEPTED",
        }
    ]


async def test_restore_live_session_requires_confirmation(ctx: Ctx) -> None:
    cfg_dir = ctx.tmp / "paper_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "paper_1.json").write_text(
        json.dumps(
            {
                "trader_id": "LIVE-001",
                "strategy_id": ctx.strategy_id,
                "symbols": ["BTCUSDT"],
                "binance": {"testnet": False},
            }
        )
    )
    r = await ctx.ac.post("/api/paper/restore", json={"trader_id": "LIVE-001"})
    assert r.status_code == 400 and "confirm_live" in r.text
    assert ctx.launched == []


# ------------------------------------------------------------------- /stop


def _register_paper_job(ctx: Ctx, pid: int, identity: str | None) -> None:
    ctx.jobs.conn.execute(
        """INSERT OR REPLACE INTO background_jobs
           (run_id, pid, pid_start_time, job_type, status, started_at, heartbeat_at)
           VALUES (?, ?, ?, 'paper', 'running', datetime('now'), datetime('now'))""",
        (ctx.validation_run + 1000, pid, identity),
    )
    ctx.jobs.conn.commit()


async def test_stop_never_signals_a_recycled_pid(ctx: Ctx) -> None:
    """Stored PID now owned by an unrelated process -> /stop leaves it alone."""
    import subprocess

    victim = subprocess.Popen(["sleep", "30"])
    try:
        _register_paper_job(ctx, victim.pid, identity="not-this-process")
        r = await ctx.ac.post("/api/paper/stop")
        assert r.status_code == 200, r.text
        await asyncio.sleep(0.2)
        assert victim.poll() is None, "unrelated process was killed via a recycled PID"
    finally:
        victim.kill()
        victim.wait()


async def test_stop_terminates_the_owned_node_process(ctx: Ctx) -> None:
    import subprocess

    from vibe_quant.jobs.manager import process_start_time

    node = subprocess.Popen(["sleep", "30"])
    try:
        _register_paper_job(ctx, node.pid, identity=process_start_time(node.pid))
        r = await ctx.ac.post("/api/paper/stop")
        assert r.status_code == 200, r.text
        assert node.wait(timeout=5) == -15  # SIGTERM
    finally:
        if node.poll() is None:
            node.kill()
            node.wait()
