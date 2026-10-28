"""``parameters.initial_balance`` on the backtest launch API (bead vibe-quant-zpbmo, Z1).

The input domain is tabled ONCE at its owner (``tests/unit/test_balance_settings.py``).
This file pins the WIRING: both routes take ``parameters.initial_balance`` through the ONE
shared rule, so an invalid value is a 422 whose detail names ``parameters.initial_balance``
and the reason, no run row is created, and ``subprocess.Popen`` is never called. (The
launch used to start a run that then silently ran at the 1000 default: a value like ``-5``
or ``"10000"`` was accepted and ignored.)

R1/R2: a valid value (or none at all) launches, and the stored parameters keep the value
the client sent.

NaN and Infinity are deliberately NOT in the API parametrization: they are not JSON
values, and a launch body that carries one is a 422 naming the path
(``must be a finite number``, see ``test_launch_non_finite.py``). Both are still rejected
by the shared parser (R5) and by the validation runner (R7, where a stored NaN reached
through the parameters JSON fails the run).
"""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TCH003
from typing import Any  # noqa: TCH003
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager

ENDPOINTS = ("screening", "validation")

BALANCE_MESSAGE = "parameters.initial_balance: initial_balance must be a whole number in"


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
        yield ac, state_mgr, job_mgr, tmp_db

    await ws_mgr.stop()
    job_mgr.close()
    state_mgr.close()


def _body(state: StateManager, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy_id": state.create_strategy(name="balance_api", dsl_config={}),
        "symbols": ["BTCUSDT"],
        "timeframe": "1m",
        "start_date": "2025-01-01",
        "end_date": "2025-02-01",
        "parameters": parameters,
    }


def _run_count(state: StateManager) -> int:
    return int(state.conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0])


async def _post(ac: AsyncClient, endpoint: str, body: dict[str, Any]):
    """POST the body with allow_nan JSON: httpx's json= refuses NaN/Infinity, but the
    contract (R5) requires those to be rejected by the API, not by the test client."""
    return await ac.post(
        f"/api/backtest/{endpoint}",
        content=json.dumps(body),
        headers={"content-type": "application/json"},
    )


@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_invalid_initial_balance_is_422_and_never_launches(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
    endpoint: str,
) -> None:
    """R6: 422 naming the parameter, no run row, and Popen is not called."""
    ac, state, job_mgr, _db = client
    before = _run_count(state)

    with patch.object(job_mgr, "is_process_alive", return_value=True), patch(
        "subprocess.Popen"
    ) as mock_popen:
        mock_popen.return_value.pid = 12345
        response = await _post(ac, endpoint, _body(state, {"initial_balance": 1000.5}))

    assert response.status_code == 422, response.text
    assert BALANCE_MESSAGE in response.text
    assert "got 1000.5" in response.text
    assert _run_count(state) == before
    assert mock_popen.call_count == 0


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize(
    ("value", "stored"),
    [
        pytest.param(1, 1, id="min"),
        pytest.param(10000.0, 10000.0, id="integral-float"),
    ],
)
async def test_valid_initial_balance_launches_and_is_stored_unchanged(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
    endpoint: str,
    value: object,
    stored: object,
) -> None:
    """R2: a valid balance is accepted and stored exactly as sent."""
    ac, state, job_mgr, _db = client

    with patch.object(job_mgr, "is_process_alive", return_value=True), patch(
        "subprocess.Popen"
    ) as mock_popen:
        mock_popen.return_value.pid = 12345
        response = await _post(ac, endpoint, _body(state, {"initial_balance": value}))

    assert response.status_code == 201, response.text
    run = state.get_backtest_run(response.json()["id"])
    assert run is not None
    assert run["parameters"]["initial_balance"] == stored


@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_absent_initial_balance_still_launches(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path], endpoint: str
) -> None:
    """R1: the key is optional; a missing key means the 1000 default at run time."""
    ac, state, job_mgr, _db = client

    with patch.object(job_mgr, "is_process_alive", return_value=True), patch(
        "subprocess.Popen"
    ) as mock_popen:
        mock_popen.return_value.pid = 12345
        response = await _post(ac, endpoint, _body(state, {}))

    assert response.status_code == 201, response.text
    run = state.get_backtest_run(response.json()["id"])
    assert run is not None
    assert "initial_balance" not in run["parameters"]
