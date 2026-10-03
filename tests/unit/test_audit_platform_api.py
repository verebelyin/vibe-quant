"""Audit e70tl.15-.18 + API mediums: notes, CSRF, input validation, DB pinning, WS, data jobs."""

from __future__ import annotations

import asyncio
import functools
import json
import sqlite3
import sys
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from vibe_quant.api import app as app_module
from vibe_quant.api.app import create_app
from vibe_quant.api.routers import data as data_router
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.data.archive import RawDataArchive
from vibe_quant.data.catalog import CatalogManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager, JobStatus

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

Client = tuple[AsyncClient, StateManager, BacktestJobManager]


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "api.db"


@pytest.fixture
async def client(db: Path) -> AsyncIterator[Client]:
    app = create_app()
    state = StateManager(db)
    jobs = BacktestJobManager(db)
    ws = ConnectionManager()
    await ws.start()
    app.state.state_manager = state
    app.state.job_manager = jobs
    app.state.catalog_manager = CatalogManager()
    app.state.ws_manager = ws
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, state, jobs
    await ws.stop()
    jobs.close()
    state.close()


def _strategy(state: StateManager, name: str = "s1") -> int:
    return state.create_strategy(name, {"name": name})


def _launch_body(strategy_id: int, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "strategy_id": strategy_id,
        "symbols": ["BTCUSDT"],
        "timeframe": "4h",
        "start_date": "2025-01-01",
        "end_date": "2025-06-01",
        "parameters": {"leverage": 5, "initial_balance": 1000},
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# e70tl.16 — user notes never clobber discovery JSON
# ---------------------------------------------------------------------------

_DISCOVERY_NOTES = json.dumps(
    {
        "type": "discovery",
        "cross_window_months": [1, 2],
        "cross_window_min_sharpe": 0.5,
        "bootstrap_min_sharpe": 0.0,
        "top_strategies": [
            {
                "dsl": {"name": "genome_abc"},
                "sharpe": 1.4,
                "cross_window": {
                    "passed": True,
                    "windows": [
                        {"sharpe": 1.4, "return_pct": 0.12, "max_dd": 0.05, "trades": 80},
                        {"sharpe": 0.9, "return_pct": -0.01, "max_dd": 0.07, "trades": 70},
                        {"sharpe": 0.6, "return_pct": 0.02, "max_dd": 0.06, "trades": 75},
                    ],
                },
                "wfa": {"consistency": 0.8, "sharpe_consistency": 0.6, "passed": True},
            }
        ],
    }
)


async def _discovery_run(state: StateManager) -> int:
    run_id = state.create_backtest_run(
        None, "discovery", ["BTCUSDT"], "4h", "2025-01-01", "2025-06-01", {}
    )
    state.save_backtest_result(run_id, {"total_trades": 80, "notes": _DISCOVERY_NOTES})
    return run_id


async def test_editing_notes_keeps_discovery_results(client: Client) -> None:
    ac, state, _ = client
    run_id = await _discovery_run(state)

    r = await ac.put(f"/api/results/runs/{run_id}/notes", json={"notes": "promising, check 1m"})
    assert r.status_code == 200
    body = r.json()
    assert body["user_notes"] == "promising, check 1m"
    assert body["notes"] == _DISCOVERY_NOTES

    strategies = (await ac.get(f"/api/discovery/results/{run_id}")).json()["strategies"]
    assert len(strategies) == 1
    assert strategies[0]["dsl"]["name"] == "genome_abc"

    again = (await ac.get(f"/api/results/runs/{run_id}")).json()
    assert again["user_notes"] == "promising, check 1m"


async def test_validation_run_notes_persist(client: Client) -> None:
    ac, state, _ = client
    run_id = state.create_backtest_run(
        _strategy(state), "validation", ["BTCUSDT"], "4h", "2025-01-01", "2025-06-01", {}
    )
    state.save_backtest_result(run_id, {"total_trades": 3})
    assert (await ac.put(f"/api/results/runs/{run_id}/notes", json={"notes": "x"})).status_code == 200
    assert (await ac.get(f"/api/results/runs/{run_id}")).json()["user_notes"] == "x"


async def test_cross_window_rows_labelled_and_gated_like_pipeline(client: Client) -> None:
    ac, state, _ = client
    run_id = await _discovery_run(state)
    body = (await ac.get(f"/api/results/runs/{run_id}")).json()
    rows = body["cross_window_results"]
    assert [r["offset"] for r in rows] == [0, 1, 2]
    assert [r["in_sample"] for r in rows] == [True, False, False]
    # window +1: sharpe 0.9 >= 0.5 but return < 0 → FAIL (pipeline requires both)
    assert [r["passed"] for r in rows] == [True, False, True]
    assert body["cross_window_passed"] is True
    assert body["wfa_consistency"] == 0.8
    assert body["wfa_passed"] is True
    assert body["bootstrap_min_sharpe"] == 0.0


# ---------------------------------------------------------------------------
# e70tl.15 — summary has no duplicate run ids
# ---------------------------------------------------------------------------


async def test_runs_summary_unique_after_rerun(client: Client) -> None:
    ac, state, _ = client
    run_id = state.create_backtest_run(
        _strategy(state), "validation", ["BTCUSDT"], "4h", "2025-01-01", "2025-06-01", {}
    )
    state.save_backtest_result(run_id, {"total_trades": 0})
    state.save_backtest_result(run_id, {"total_trades": 66, "sharpe_ratio": 2.3})
    runs = (await ac.get("/api/results/runs/summary")).json()["runs"]
    ids = [r["run_id"] for r in runs]
    assert ids.count(run_id) == 1
    assert next(r for r in runs if r["run_id"] == run_id)["total_trades"] == 66


# ---------------------------------------------------------------------------
# e70tl.17 — cross-site requests
# ---------------------------------------------------------------------------

_EVIL = {"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"}


@pytest.mark.parametrize(
    "path",
    [
        "/api/backtest/jobs/cleanup-stale",
        "/api/indicators/reload",
        "/api/backtest/jobs/1/heartbeat",
    ],
)
@pytest.mark.parametrize(
    "content_type",
    [None, "text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x"],
)
async def test_cross_site_simple_post_rejected(
    client: Client, path: str, content_type: str | None
) -> None:
    ac, _, _ = client
    headers = dict(_EVIL)
    if content_type:
        headers["Content-Type"] = content_type
    r = await ac.post(path, headers=headers, content=b"a=1" if content_type else None)
    assert r.status_code == 403


async def test_frontend_style_and_non_browser_posts_allowed(client: Client) -> None:
    ac, _, _ = client
    # Frontend (customInstance) always sends JSON content-type.
    r = await ac.post(
        "/api/backtest/jobs/cleanup-stale",
        headers={"Origin": "http://localhost:5173", "Content-Type": "application/json"},
    )
    assert r.status_code == 200
    r = await ac.post(
        "/api/backtest/jobs/cleanup-stale",
        headers={"Origin": "http://localhost:5175", "X-Requested-With": "fetch"},
    )
    assert r.status_code == 200
    # CLI/curl (no Origin) unaffected.
    assert (await ac.post("/api/backtest/jobs/cleanup-stale")).status_code == 200


async def test_cross_origin_json_preflight_rejected(client: Client) -> None:
    ac, _, _ = client
    r = await ac.options(
        "/api/backtest/jobs/cleanup-stale",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Input validation (mediums)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"start_date": "2025-06-01", "end_date": "2025-01-01"},
        {"start_date": "2025-01-01", "end_date": "2025-01-01"},
        {"start_date": "yesterday"},
        {"end_date": "2025-02-30"},
        {"symbols": []},
        {"symbols": ["  "]},
        {"timeframe": "7m"},
        {"parameters": {"leverage": 0}},
        {"parameters": {"leverage": -5}},
        {"parameters": {"leverage": 500}},
        {"parameters": {"leverage": True}},
        {"parameters": {"initial_balance": 0}},
        {"sizing_config_id": 1},
        {"risk_config_id": 1},
        {"overfitting_filters": {"deflated_sharpe_ratio": True}},
    ],
)
async def test_launch_rejects_garbage(client: Client, overrides: dict[str, Any]) -> None:
    ac, state, _ = client
    sid = _strategy(state)
    with patch("subprocess.Popen") as popen:
        r = await ac.post("/api/backtest/screening", json=_launch_body(sid, **overrides))
    assert r.status_code == 422, r.text
    popen.assert_not_called()


async def test_launch_unknown_strategy_404(client: Client) -> None:
    ac, _, _ = client
    with patch("subprocess.Popen") as popen:
        r = await ac.post("/api/backtest/validation", json=_launch_body(999))
    assert r.status_code == 404
    popen.assert_not_called()


async def test_compare_rejects_non_integer_ids(client: Client) -> None:
    ac, _, _ = client
    assert (await ac.get("/api/results/compare?run_ids=1,abc")).status_code == 422


async def test_recreate_soft_deleted_strategy_name_is_409(client: Client) -> None:
    ac, _, _ = client
    body = {"name": "dup", "dsl_config": {"name": "dup"}}
    sid = (await ac.post("/api/strategies", json=body)).json()["id"]
    assert (await ac.delete(f"/api/strategies/{sid}")).status_code == 204
    r = await ac.post("/api/strategies", json=body)
    assert r.status_code == 409
    assert "deleted" in r.json()["detail"]
    other = (await ac.post("/api/strategies", json={**body, "name": "other"})).json()["id"]
    r = await ac.put(f"/api/strategies/{other}", json={"name": "dup"})
    assert r.status_code == 409


async def test_ingest_rejects_garbage(client: Client) -> None:
    ac, _, _ = client
    bad = [
        {"symbols": [], "start_date": "2025-01-01", "end_date": "2025-02-01"},
        {"symbols": ["BTCUSDT"], "start_date": "2025-03-01", "end_date": "2025-02-01"},
        {"symbols": ["BTCUSDT"], "start_date": "garbage", "end_date": "2025-02-01"},
    ]
    with patch("subprocess.Popen") as popen:
        for body in bad:
            assert (await ac.post("/api/data/ingest", json=body)).status_code == 422
    popen.assert_not_called()


# ---------------------------------------------------------------------------
# e70tl.18 — every job is pinned to the API's DB; no runtime DB switch
# ---------------------------------------------------------------------------


async def test_launch_pins_subprocess_to_api_db(client: Client, db: Path) -> None:
    ac, state, jobs = client
    sid = _strategy(state)
    with patch("subprocess.Popen") as popen:
        popen.return_value.pid = 4242
        r = await ac.post("/api/backtest/screening", json=_launch_body(sid))
        r2 = await ac.post("/api/backtest/validation", json=_launch_body(sid))
    assert r.status_code == 201 and r2.status_code == 201
    for call, mode in zip(popen.call_args_list, ("screening", "validation"), strict=True):
        cmd = call.args[0]
        assert cmd[2:5] == ["vibe_quant", mode, "run"]
        assert cmd[cmd.index("--db") + 1] == str(db)
    run_id = r.json()["id"]
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT pid FROM background_jobs WHERE run_id = ?", (run_id,)).fetchone()
    assert conn.execute("SELECT id FROM backtest_runs WHERE id = ?", (run_id,)).fetchone()


async def test_database_switch_endpoint_removed(client: Client, db: Path) -> None:
    ac, _, _ = client
    r = await ac.put("/api/settings/database", json={"path": "data/state/other.db"})
    assert r.status_code == 405
    info = (await ac.get("/api/settings/database")).json()
    assert info["path"] == str(db)


# ---------------------------------------------------------------------------
# e70tl.17 — data jobs: wrapper command, job id, progress stream
# ---------------------------------------------------------------------------


async def test_ingest_spawns_heartbeating_wrapper(client: Client, db: Path) -> None:
    ac, _, jobs = client
    body = {"symbols": ["BTCUSDT"], "start_date": "2025-01-01", "end_date": "2025-02-01"}
    with patch("subprocess.Popen") as popen:
        popen.return_value.pid = 4243
        r = await ac.post("/api/data/ingest", json=body)
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert job_id < 0
    cmd = popen.call_args.args[0]
    assert cmd[1:3] == ["-m", "vibe_quant.jobs.data_job"]
    assert cmd[cmd.index("--run-id") + 1] == str(job_id)
    assert cmd[cmd.index("--db") + 1] == str(db)
    assert cmd[cmd.index("--") + 1 :] == [
        "ingest", "--symbols", "BTCUSDT", "--start", "2025-01-01", "--end", "2025-02-01",
    ]
    info = jobs.get_job_info(job_id)
    assert info is not None and info.job_type == "data_ingest"


async def test_update_and_rebuild_return_job_id(client: Client) -> None:
    ac, _, _ = client
    with patch("subprocess.Popen") as popen:
        popen.return_value.pid = 4244
        for path in ("/api/data/update", "/api/data/rebuild"):
            r = await ac.post(path)
            assert r.status_code == 202
            assert r.json()["job_id"] < 0


# ---------------------------------------------------------------------------
# Data quality endpoint (medium)
# ---------------------------------------------------------------------------


def test_data_quality_reports_gaps_and_filler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "raw.db"
    archive = RawDataArchive(path)
    t0 = 1704067200000
    m = 60_000
    rows = [
        (t0, 100.0, 101.0, 99.0, 100.5, 5.0, t0 + m - 1),
        (t0 + m, 100.5, 101.0, 100.0, 100.8, 5.0, t0 + 2 * m - 1),
        # 2-minute step: one missing bar (the old 5-minute threshold passed this)
        (t0 + 3 * m, 100.8, 101.5, 100.1, 101.0, 4.0, t0 + 4 * m - 1),
        # flat zero-volume filler run of 2 bars
        (t0 + 4 * m, 101.0, 101.0, 101.0, 101.0, 0.0, t0 + 5 * m - 1),
        (t0 + 5 * m, 101.0, 101.0, 101.0, 101.0, 0.0, t0 + 6 * m - 1),
        (t0 + 6 * m, 101.0, 102.0, 100.0, 101.2, 3.0, t0 + 7 * m - 1),
    ]
    archive.insert_klines("BTCUSDT", "1m", rows, "test")
    archive.close()
    monkeypatch.setattr(data_router, "_get_archive", lambda: RawDataArchive(path))

    body = data_router.data_quality("BTCUSDT").model_dump()
    assert body["error"] is None
    assert body["kline_count"] == 6
    assert body["gaps"] == [
        {"start": "2024-01-01 00:01", "end": "2024-01-01 00:03", "missing_bars": 1}
    ]
    assert body["missing_bars"] == 1
    assert body["zero_volume_runs"] == [
        {"start": "2024-01-01 00:04", "end": "2024-01-01 00:05", "bars": 2}
    ]
    # 7 expected bars, 1 missing + 2 filler → 4/7 clean
    assert body["quality_score"] == pytest.approx(4 / 7)


# ---------------------------------------------------------------------------
# WebSocket manager robustness (medium)
# ---------------------------------------------------------------------------


async def test_broadcast_survives_disconnect_during_send() -> None:
    mgr = ConnectionManager()
    slow = MagicMock()
    other = MagicMock()

    async def slow_send(_payload: str) -> None:
        mgr.disconnect(other, "jobs")  # mutates the live set mid-iteration
        await asyncio.sleep(0)

    async def other_send(_payload: str) -> None:
        return None

    dead = MagicMock()

    async def dead_send(_payload: str) -> None:
        raise RuntimeError("socket closed")

    slow.send_text = slow_send
    other.send_text = other_send
    dead.send_text = dead_send
    mgr._channels["jobs"] = {slow, other, dead}  # noqa: SLF001
    await mgr.broadcast("jobs", {"type": "job_started"})
    assert dead not in mgr._channels.get("jobs", set())  # noqa: SLF001
    await mgr.start()
    await mgr.stop()  # never raises


async def test_launch_ok_even_if_ws_broadcast_breaks(client: Client) -> None:
    ac, state, _ = client
    sid = _strategy(state)
    with (
        patch("subprocess.Popen") as popen,
        patch.object(ConnectionManager, "_send_all", side_effect=RuntimeError("boom")),
    ):
        popen.return_value.pid = 4245
        r = await ac.post("/api/backtest/screening", json=_launch_body(sid))
    assert r.status_code == 201


# ---------------------------------------------------------------------------
# Backend restart → lifespan reconciles dead 'running' jobs
# ---------------------------------------------------------------------------


def test_startup_reconciles_jobs(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = StateManager(db)
    sid = _strategy(state)
    dead_run, live_run = (
        state.create_backtest_run(sid, "validation", ["BTCUSDT"], "4h", "2025-01-01", "2025-06-01", {})
        for _ in range(2)
    )
    jobs = BacktestJobManager(db)
    jobs.start_job(live_run, "validation", [sys.executable, "-c", "import time; time.sleep(30)"])
    jobs.conn.execute(
        "INSERT INTO background_jobs (run_id, pid, pid_start_time, job_type, status) "
        "VALUES (?, 99999999, 'darwin:1.000000', 'validation', 'running')",
        (dead_run,),
    )
    jobs.conn.execute("UPDATE backtest_runs SET status='running' WHERE id = ?", (dead_run,))
    jobs.conn.commit()

    monkeypatch.setattr(app_module, "StateManager", functools.partial(StateManager, db))
    monkeypatch.setattr(app_module, "BacktestJobManager", functools.partial(BacktestJobManager, db))
    try:
        with TestClient(create_app()):
            assert jobs.get_status(dead_run) == JobStatus.FAILED
            assert jobs.get_status(live_run) == JobStatus.RUNNING
            run = state.get_backtest_run(dead_run)
            assert run is not None and run["status"] == "failed"
    finally:
        jobs.kill_job(live_run, force=True)
        jobs.close()
        state.close()
