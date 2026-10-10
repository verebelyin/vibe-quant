"""Launch-time discovery config validation (vibe-quant-zhcq7 part 2).

A bad window/gate config must be rejected with 422 BEFORE any run row exists.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TCH003
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.routers import discovery as discovery_router
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager

_OK_BODY = {
    "symbols": ["BTCUSDT"],
    "timeframes": ["4h"],
    "start_date": "2024-01-01",
    "end_date": "2025-01-01",
}


@pytest.fixture()
async def client(tmp_path: Path):
    app = create_app()
    state_mgr = StateManager(db_path=tmp_path / "test.db")
    _ = state_mgr.conn
    job_mgr = BacktestJobManager(db_path=tmp_path / "test.db")
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


def _run_count(state: StateManager) -> int:
    return int(state.conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0])


async def _post(ac: AsyncClient, job_mgr: BacktestJobManager, body: dict[str, object]):
    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 12345
        return await ac.post("/api/discovery/launch", json=body)


async def test_short_wfa_rejected_422_no_run_row(client) -> None:
    ac, state, job_mgr = client
    before = _run_count(state)
    r = await _post(
        ac, job_mgr,
        {**_OK_BODY, "start_date": "2025-01-01", "end_date": "2025-03-02",
         "train_test_split": 0.8, "wfa_oos_step_days": 90},
    )
    assert r.status_code == 422
    assert "WFA" in r.json()["detail"]
    assert _run_count(state) == before


async def test_eval_windows_omitted_uses_cli_default(client) -> None:
    """eval_windows=0 is not sent -> CLI default 3 applies (needs >= 21 train days)."""
    ac, state, job_mgr = client
    before = _run_count(state)
    r = await _post(
        ac, job_mgr,
        {**_OK_BODY, "start_date": "2025-01-01", "end_date": "2025-01-16",
         "train_test_split": 0, "eval_windows": 0},
    )
    assert r.status_code == 422
    assert "eval_windows" in r.json()["detail"]
    assert _run_count(state) == before


@pytest.mark.parametrize(
    "override",
    [
        {"population": 1},
        {"population": 5, "elite_count": 5},
        {"wfa_min_consistency": 1.5},
        {"cross_window_months": [0]},
        {"cross_window_months": [-15]},
        {"cross_window_months": [-15, 3]},
    ],
)
async def test_invalid_knobs_rejected_422(client, override: dict[str, object]) -> None:
    ac, state, job_mgr = client
    before = _run_count(state)
    r = await _post(ac, job_mgr, {**_OK_BODY, **override})
    assert r.status_code == 422
    assert _run_count(state) == before


async def test_valid_defaults_accepted(client) -> None:
    ac, state, job_mgr = client
    r = await _post(ac, job_mgr, _OK_BODY)
    assert r.status_code == 201
    assert _run_count(state) == 1


async def test_valid_cross_window_and_wfa_accepted(client) -> None:
    ac, _state, job_mgr = client
    r = await _post(
        ac, job_mgr,
        {**_OK_BODY, "cross_window_months": [1, 2], "wfa_oos_step_days": 60},
    )
    assert r.status_code == 201


async def test_malformed_own_command_is_500_with_stderr(client, monkeypatch) -> None:
    ac, state, job_mgr = client
    real = discovery_router._discovery_cli_args
    monkeypatch.setattr(
        discovery_router,
        "_discovery_cli_args",
        lambda *a, **k: [*real(*a, **k), "--no-such-flag"],
    )
    before = _run_count(state)
    r = await _post(ac, job_mgr, _OK_BODY)
    assert r.status_code == 500
    assert "--no-such-flag" in r.json()["detail"]
    assert _run_count(state) == before


async def test_real_command_carries_created_run_id(client) -> None:
    """The precheck's placeholder --run-id 0 must never reach the subprocess."""
    ac, _state, job_mgr = client
    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 12345
        r = await ac.post("/api/discovery/launch", json=_OK_BODY)
    assert r.status_code == 201
    run_id = r.json()["run_id"]
    argv = mock_popen.call_args.args[0]
    assert argv[argv.index("--run-id") + 1] == str(run_id)
    assert argv.count("--run-id") == 1
    assert run_id != 0
