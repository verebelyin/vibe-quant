"""``parameters.fee_tier`` / ``parameters.bnb_discount`` on the backtest launch API (R19).

The runner builds the fee schedule from the nested run parameters, so a bad tier used to
launch a run that failed later, and an unhashable tier (a list, a dict) was a 500. Both
routes now take the nested fee keys through the ONE shared rule (``FeeSchedule``): an
invalid value is a 422 naming the failing field, no run row is created and
``subprocess.Popen`` is never called.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TCH003
from typing import Any  # noqa: TCH003
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.fees import BINANCE_USDTM_TIERS
from vibe_quant.jobs.manager import BacktestJobManager

ENDPOINTS = ("screening", "validation")
KNOWN = sorted(BINANCE_USDTM_TIERS)


@pytest.fixture()
async def client(tmp_path: Path):
    db = tmp_path / "test.db"
    app = create_app()
    state_mgr = StateManager(db_path=db)
    _ = state_mgr.conn
    job_mgr = BacktestJobManager(db_path=db)
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


def _body(state: StateManager, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy_id": state.create_strategy(name="fee_api", dsl_config={}),
        "symbols": ["BTCUSDT"],
        "timeframe": "1m",
        "start_date": "2025-01-01",
        "end_date": "2025-02-01",
        "parameters": parameters,
    }


def _run_count(state: StateManager) -> int:
    return int(state.conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0])


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        pytest.param(
            {"fee_tier": ["VIP0"]},
            f"parameters.fee_tier: unknown fee tier ['VIP0']; known tiers: {KNOWN}",
            id="list",
        ),
        pytest.param(
            {"fee_tier": "VIP12"},
            f"parameters.fee_tier: unknown fee tier 'VIP12'; known tiers: {KNOWN}",
            id="unknown",
        ),
        pytest.param(
            {"fee_tier": {"a": 1}},
            f"parameters.fee_tier: unknown fee tier {{'a': 1}}; known tiers: {KNOWN}",
            id="dict",
        ),
        pytest.param(
            {"fee_tier": None},
            f"parameters.fee_tier: unknown fee tier None; known tiers: {KNOWN}",
            id="null",
        ),
        pytest.param(
            {"bnb_discount": "yes"},
            "parameters.bnb_discount: bnb_discount must be a bool, got 'yes'",
            id="bnb-not-bool",
        ),
    ],
)
async def test_invalid_fee_parameters_are_422_and_never_launch(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    endpoint: str,
    parameters: dict[str, Any],
    message: str,
) -> None:
    """R19: 422 naming the failing field, no run row, Popen never called."""
    ac, state, job_mgr = client
    before = _run_count(state)
    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 12345
        response = await ac.post(f"/api/backtest/{endpoint}", json=_body(state, parameters))

    assert response.status_code == 422, response.text
    assert [e["msg"] for e in response.json()["detail"]] == [f"Value error, {message}"]
    assert _run_count(state) == before
    assert mock_popen.call_count == 0


@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_valid_fee_parameters_launch_and_are_stored_unchanged(
    client: tuple[AsyncClient, StateManager, BacktestJobManager], endpoint: str
) -> None:
    """The known tier with the BNB discount launches; the stored params are what was sent."""
    ac, state, job_mgr = client
    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 12345
        response = await ac.post(
            f"/api/backtest/{endpoint}",
            json=_body(state, {"fee_tier": "VIP0", "bnb_discount": True}),
        )

    assert response.status_code == 201, response.text
    run = state.get_backtest_run(response.json()["id"])
    assert run is not None
    assert run["parameters"]["fee_tier"] == "VIP0"
    assert run["parameters"]["bnb_discount"] is True
