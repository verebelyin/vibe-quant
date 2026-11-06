"""BNB-discount launch field on the backtest API.

Validation and screening both accept ``bnb_discount`` and store it in the run
parameters; a non-bool value is rejected before any run is created. A post-only
``entry_order`` is validation-only and refused for screening (D3-2).
"""

from __future__ import annotations

import json
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
        yield ac, state_mgr, job_mgr, tmp_db

    await ws_mgr.stop()
    job_mgr.close()
    state_mgr.close()


def _body(state: StateManager, *, bnb: bool | None = None, parameters: dict | None = None) -> dict:
    strategy_id = state.create_strategy(name="bnb_api", dsl_config={})
    body: dict = {
        "strategy_id": strategy_id,
        "symbols": ["BTCUSDT"],
        "timeframe": "1m",
        "start_date": "2025-01-01",
        "end_date": "2025-02-01",
        "parameters": parameters if parameters is not None else {"initial_balance": 1000},
    }
    if bnb is not None:
        body["bnb_discount"] = bnb
    return body


async def test_launch_bnb_discount_in_params(client) -> None:
    ac, state, job_mgr, _db = client

    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 12345
        r = await ac.post("/api/backtest/validation", json=_body(state, bnb=True))

    assert r.status_code == 201, r.text
    run = state.get_backtest_run(r.json()["id"])
    assert run is not None
    assert run["parameters"]["bnb_discount"] is True


async def test_launch_default_omits_bnb_discount(client) -> None:
    ac, state, job_mgr, _db = client

    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 12345
        r = await ac.post("/api/backtest/validation", json=_body(state))

    assert r.status_code == 201, r.text
    run = state.get_backtest_run(r.json()["id"])
    assert run is not None
    assert "bnb_discount" not in run["parameters"]


async def test_screening_accepts_bnb_discount(client) -> None:
    """D3-2: screening stores top-level bnb_discount, and nothing when default."""
    ac, state, job_mgr, _db = client

    opted_body = _body(state, bnb=True)
    default_body = {k: v for k, v in opted_body.items() if k != "bnb_discount"}

    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 12345
        opted_in = await ac.post("/api/backtest/screening", json=opted_body)
        default = await ac.post("/api/backtest/screening", json=default_body)

    assert opted_in.status_code == 201, opted_in.text
    assert default.status_code == 201, default.text
    run = state.get_backtest_run(opted_in.json()["id"])
    assert run is not None
    assert run["parameters"]["bnb_discount"] is True
    default_run = state.get_backtest_run(default.json()["id"])
    assert default_run is not None
    assert "bnb_discount" not in default_run["parameters"]


async def test_screening_accepts_nested_bnb_discount_param(client) -> None:
    """D3-2 flips D3-3's refusal: a nested parameters.bnb_discount is applied too.

    The screening CLI derives the venue schedule from the run parameters, so the
    nested form reaches the runner exactly like the top-level field.
    """
    ac, state, _job_mgr, _db = client

    r = await ac.post(
        "/api/backtest/screening",
        json=_body(state, parameters={"initial_balance": 1000, "bnb_discount": True}),
    )

    assert r.status_code == 201, r.text
    run = state.get_backtest_run(r.json()["id"])
    assert run is not None
    assert run["parameters"]["bnb_discount"] is True


async def test_screening_api_refuses_post_only(client) -> None:
    """D3-2: a post-only entry_order is validation-only (422)."""
    ac, state, _job_mgr, _db = client

    r = await ac.post(
        "/api/backtest/screening",
        json=_body(state, parameters={"initial_balance": 1000, "entry_order": "post_only"}),
    )

    assert r.status_code == 422
    assert "validation-only" in r.json()["detail"]


async def test_screening_api_refuses_any_non_market_entry(client) -> None:
    """Any non-market entry_order is refused, not just the literal post_only."""
    ac, state, _job_mgr, _db = client

    r = await ac.post(
        "/api/backtest/screening",
        json=_body(state, parameters={"initial_balance": 1000, "entry_order": "limit"}),
    )

    assert r.status_code == 422
    assert "entry_order='limit'" in r.json()["detail"]


async def test_bnb_discount_strict_bool(client) -> None:
    ac, state, _job_mgr, _db = client

    r = await ac.post("/api/backtest/validation", json=_body(state, bnb="true"))

    assert r.status_code == 422


_VALID_DSL: dict = {
    "name": "bnb_final",
    "description": "test strategy",
    "version": 1,
    "timeframe": "5m",
    "additional_timeframes": [],
    "indicators": {"rsi": {"type": "RSI", "period": 14, "source": "close"}},
    "entry_conditions": {"long": ["rsi < 30"], "short": ["rsi > 70"]},
    "exit_conditions": {"long": ["rsi > 50"], "short": ["rsi < 50"]},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 4.0},
}


async def test_launch_bnb_discount_run_completes_with_fee_notes(
    client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launched BNB run reaches its FINAL status and records the fee schedule.

    The API only spawns a subprocess (mocked here), so the created run is driven
    to completion with the real runner (backtest itself stubbed): a 201 status
    alone proves nothing about the stored notes.
    """
    ac, state, job_mgr, db_path = client
    sid = state.create_strategy(name="bnb_final", dsl_config=_VALID_DSL)
    body = {
        "strategy_id": sid,
        "symbols": ["BTCUSDT"],
        "timeframe": "1m",
        "start_date": "2025-01-01",
        "end_date": "2025-01-31",
        "parameters": {"initial_balance": 1000},
        "bnb_discount": True,
    }
    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 12345
        r = await ac.post("/api/backtest/validation", json=body)
    assert r.status_code == 201, r.text
    run_id = r.json()["id"]

    from vibe_quant.validation.results import ValidationResult
    from vibe_quant.validation.runner import ValidationRunner

    monkeypatch.setattr(ValidationRunner, "_clamp_run_window", lambda self, rc, tf: (rc, None))
    monkeypatch.setattr(
        ValidationRunner, "_resolve_detail_timeframe", lambda self, rc, tf, ov: None
    )
    runner = ValidationRunner(db_path=db_path, logs_path=tmp_path / "logs")
    runner._run_backtest = lambda **kw: ValidationResult(  # type: ignore[method-assign]
        run_id=kw["run_id"], strategy_name=kw["strategy_name"], total_trades=1
    )
    try:
        runner.run(run_id=run_id)
    finally:
        runner.close()

    row = state.get_backtest_run(run_id)
    assert row is not None
    assert row["status"] == "completed"
    stored = state.get_result_notes_raw(run_id)
    assert stored is not None
    assert json.loads(stored)["fee_schedule"] == {"fee_tier": "VIP0", "bnb_discount": True}
