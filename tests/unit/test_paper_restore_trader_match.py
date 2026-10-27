"""/api/paper/restore picks the saved config of the requested trader, newest run first.

Never another trader's config, and never by file mtime (a touched or copied file
must not outrank the trader's latest session).
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

DSL = {
    "name": "restore_match",
    "timeframe": "1h",
    "indicators": {"rsi_14": {"type": "RSI", "period": 14}},
    "entry_conditions": {"long": ["rsi_14 < 30"]},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 4.0},
}


class Ctx:
    def __init__(
        self, ac: AsyncClient, state: StateManager, jobs: BacktestJobManager, cfg_dir: Path
    ) -> None:
        self.ac = ac
        self.state = state
        self.cfg_dir = cfg_dir
        self.strategy_a = state.create_strategy(
            "restore_match_a", {**DSL, "name": "restore_match_a"}
        )
        self.strategy_b = state.create_strategy(
            "restore_match_b", {**DSL, "name": "restore_match_b"}
        )

    def write(self, name: str, content: object, mtime: int) -> None:
        self.cfg_dir.mkdir(parents=True, exist_ok=True)
        path = self.cfg_dir / name
        path.write_text(content if isinstance(content, str) else json.dumps(content))
        os.utime(path, (mtime, mtime))

    def config(self, trader_id: str, strategy_id: int, symbols: list[str]) -> dict[str, Any]:
        return {
            "trader_id": trader_id,
            "strategy_id": strategy_id,
            "symbols": symbols,
            "binance": {"testnet": True},
        }

    def restored_config(self, run_id: int) -> dict[str, Any]:
        data: dict[str, Any] = json.loads((self.cfg_dir / f"paper_{run_id}.json").read_text())
        return data


@pytest.fixture()
async def ctx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Ctx]:
    db = tmp_path / "api.db"
    monkeypatch.setattr("vibe_quant.db.connection.DEFAULT_DB_PATH", db)
    monkeypatch.setattr(
        "vibe_quant.api.routers.paper_trading.PAPER_CONFIG_DIR", tmp_path / "paper_configs"
    )
    monkeypatch.setenv("BINANCE_TESTNET_API_KEY", "tk")
    monkeypatch.setenv("BINANCE_TESTNET_API_SECRET", "ts")
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    app = create_app()
    state = StateManager(db_path=db)
    _ = state.conn
    jobs = BacktestJobManager(db_path=db)
    ws = ConnectionManager()
    await ws.start()
    app.state.state_manager = state
    app.state.job_manager = jobs
    app.state.ws_manager = ws

    def fake_start_job(
        run_id: int, job_type: str, command: list[str], log_file: str | None = None
    ) -> int:
        jobs.conn.execute(
            """INSERT OR REPLACE INTO background_jobs
               (run_id, pid, job_type, status, started_at, heartbeat_at)
               VALUES (?, 4242, 'paper', 'running', datetime('now'), datetime('now'))""",
            (run_id,),
        )
        jobs.conn.commit()
        return 4242

    monkeypatch.setattr(jobs, "start_job", fake_start_job)
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield Ctx(ac, state, jobs, tmp_path / "paper_configs")
    await ws.stop()
    jobs.close()
    state.close()


async def test_restore_uses_the_requested_traders_config_not_the_newest_file(ctx: Ctx) -> None:
    ctx.write("paper_1.json", ctx.config("PAPER-001", ctx.strategy_a, ["BTCUSDT"]), 1_000)
    ctx.write("paper_2.json", ctx.config("PAPER-002", ctx.strategy_b, ["ETHUSDT"]), 2_000)

    r = await ctx.ac.post("/api/paper/restore", json={"trader_id": "PAPER-001"})

    assert r.status_code == 201, r.text
    cfg = ctx.restored_config(r.json()["run_id"])
    assert (cfg["trader_id"], cfg["strategy_id"], cfg["symbols"]) == (
        "PAPER-001",
        ctx.strategy_a,
        ["BTCUSDT"],
    )
    run = ctx.state.get_backtest_run(r.json()["run_id"])
    assert run is not None
    assert run["strategy_id"] == ctx.strategy_a


async def test_restore_of_the_other_trader_gets_its_own_config(ctx: Ctx) -> None:
    """Same two files, the other id: the older file is not preferred either."""
    ctx.write("paper_1.json", ctx.config("PAPER-001", ctx.strategy_a, ["BTCUSDT"]), 1_000)
    ctx.write("paper_2.json", ctx.config("PAPER-002", ctx.strategy_b, ["ETHUSDT"]), 2_000)

    r = await ctx.ac.post("/api/paper/restore", json={"trader_id": "PAPER-002"})

    assert r.status_code == 201, r.text
    cfg = ctx.restored_config(r.json()["run_id"])
    assert (cfg["strategy_id"], cfg["symbols"]) == (ctx.strategy_b, ["ETHUSDT"])


async def test_restore_picks_the_highest_run_id_of_the_trader_not_the_newest_mtime(
    ctx: Ctx,
) -> None:
    """The latest session (run 7) wins although run 3's file was touched afterwards."""
    ctx.write("paper_3.json", ctx.config("PAPER-001", ctx.strategy_a, ["BTCUSDT"]), 9_000)
    ctx.write("paper_7.json", ctx.config("PAPER-001", ctx.strategy_b, ["ETHUSDT"]), 1_000)
    ctx.write("paper_12.json", ctx.config("PAPER-002", ctx.strategy_a, ["SOLUSDT"]), 5_000)

    r = await ctx.ac.post("/api/paper/restore", json={"trader_id": "PAPER-001"})

    assert r.status_code == 201, r.text
    cfg = ctx.restored_config(r.json()["run_id"])
    assert (cfg["strategy_id"], cfg["symbols"]) == (ctx.strategy_b, ["ETHUSDT"])


async def test_restore_orders_run_ids_numerically(ctx: Ctx) -> None:
    """paper_10 is later than paper_9 (a name sort would pick paper_9)."""
    ctx.write("paper_9.json", ctx.config("PAPER-001", ctx.strategy_a, ["BTCUSDT"]), 5_000)
    ctx.write("paper_10.json", ctx.config("PAPER-001", ctx.strategy_b, ["ETHUSDT"]), 1_000)

    r = await ctx.ac.post("/api/paper/restore", json={"trader_id": "PAPER-001"})

    assert r.status_code == 201, r.text
    assert ctx.restored_config(r.json()["run_id"])["symbols"] == ["ETHUSDT"]


async def test_restore_without_a_config_for_the_trader_names_it_and_creates_no_run(
    ctx: Ctx,
) -> None:
    """Another trader's newer config is never used as a stand-in."""
    ctx.write("paper_2.json", ctx.config("PAPER-002", ctx.strategy_b, ["ETHUSDT"]), 2_000)
    before = len(ctx.state.list_backtest_runs())

    r = await ctx.ac.post("/api/paper/restore", json={"trader_id": "PAPER-001"})

    assert r.status_code == 404
    assert "PAPER-001" in r.json()["detail"]
    assert "PAPER-002" not in r.json()["detail"]
    assert len(ctx.state.list_backtest_runs()) == before


@pytest.mark.parametrize("junk", ["[1, 2]", "not json {", '"PAPER-001"', "null"])
async def test_restore_skips_a_garbage_config_file(ctx: Ctx, junk: str) -> None:
    """A non-object or unreadable file (even the newest) is skipped, never a 500."""
    ctx.write("paper_1.json", ctx.config("PAPER-001", ctx.strategy_a, ["BTCUSDT"]), 1_000)
    ctx.write("paper_2.json", junk, 2_000)

    r = await ctx.ac.post("/api/paper/restore", json={"trader_id": "PAPER-001"})

    assert r.status_code == 201, r.text
    assert ctx.restored_config(r.json()["run_id"])["symbols"] == ["BTCUSDT"]


async def test_restore_ignores_non_run_file_names(ctx: Ctx) -> None:
    """A hand-made backup file matching the glob has no run id and is not a session."""
    ctx.write("paper_1.json", ctx.config("PAPER-001", ctx.strategy_a, ["BTCUSDT"]), 1_000)
    ctx.write("paper_backup.json", ctx.config("PAPER-001", ctx.strategy_b, ["ETHUSDT"]), 9_000)

    r = await ctx.ac.post("/api/paper/restore", json={"trader_id": "PAPER-001"})

    assert r.status_code == 201, r.text
    assert ctx.restored_config(r.json()["run_id"])["symbols"] == ["BTCUSDT"]


async def test_restore_logs_a_skipped_file_by_name_without_its_content(
    ctx: Ctx, caplog: pytest.LogCaptureFixture
) -> None:
    ctx.write("paper_1.json", ctx.config("PAPER-001", ctx.strategy_a, ["BTCUSDT"]), 1_000)
    ctx.write("paper_2.json", '{"api_secret": "LEAKMARKER", ', 2_000)

    with caplog.at_level("WARNING", logger="vibe_quant.api.routers.paper_trading"):
        r = await ctx.ac.post("/api/paper/restore", json={"trader_id": "PAPER-001"})

    assert r.status_code == 201, r.text
    assert "paper_2.json" in caplog.text
    assert "LEAKMARKER" not in caplog.text


async def test_restore_skips_a_config_in_an_undecodable_encoding(ctx: Ctx) -> None:
    """Bytes that are not UTF-8 are an unreadable file like junk JSON, never a 500."""
    ctx.write("paper_1.json", ctx.config("PAPER-001", ctx.strategy_a, ["BTCUSDT"]), 1_000)
    (ctx.cfg_dir / "paper_2.json").write_bytes(b"\xff\xfe")

    r = await ctx.ac.post("/api/paper/restore", json={"trader_id": "PAPER-001"})

    assert r.status_code == 201, r.text
    assert ctx.restored_config(r.json()["run_id"])["symbols"] == ["BTCUSDT"]


async def test_restore_logs_an_undecodable_file_by_name_only(
    ctx: Ctx, caplog: pytest.LogCaptureFixture
) -> None:
    ctx.write("paper_1.json", ctx.config("PAPER-001", ctx.strategy_a, ["BTCUSDT"]), 1_000)
    (ctx.cfg_dir / "paper_2.json").write_bytes(b"\xff\xfe")

    with caplog.at_level("WARNING", logger="vibe_quant.api.routers.paper_trading"):
        r = await ctx.ac.post("/api/paper/restore", json={"trader_id": "PAPER-001"})

    assert r.status_code == 201, r.text
    assert "paper_2.json" in caplog.text
