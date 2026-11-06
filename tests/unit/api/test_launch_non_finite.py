"""NaN / Infinity / -Infinity in a launch body -> readable 422, never 500, never 201.

The three launch routes (``POST /api/backtest/screening``,
``/api/backtest/validation``, ``/api/discovery/launch``) parse with Starlette's
``request.json()``, i.e. ``json.loads``, which accepts the bare literals ``NaN``,
``Infinity`` and ``-Infinity``. Measured on HEAD:

- ``parameters.rsi_period=NaN`` -> 201 (the run launched with a NaN parameter);
- ``parameters.leverage=NaN`` / top-level ``leverage=nan`` -> 500, because FastAPI
  cannot serialise the ``RequestValidationError`` whose ``input`` is NaN
  (``json.dumps(..., allow_nan=False)``).

Contract: a non-finite float anywhere in the body is a 422 whose detail names the
dotted path and ``must be a finite number``; a body that only has finite numbers
keeps today's outcome.

regression: vibe-quant-z9qzi
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator  # noqa: TCH003
from pathlib import Path  # noqa: TCH003
from typing import Any  # noqa: TCH003
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager

SUFFIX = "must be a finite number"


@pytest.fixture()
async def client(
    tmp_path: Path,
) -> AsyncIterator[tuple[AsyncClient, StateManager, BacktestJobManager]]:
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


async def _post(ac: AsyncClient, url: str, body: dict[str, Any]) -> Any:
    """Send a raw body: httpx's ``json=`` refuses NaN/Infinity, but the app's own
    parser (``json.loads``) accepts the bare literals — the API must reject them."""
    return await ac.post(
        url,
        content=json.dumps(body),
        headers={"content-type": "application/json"},
    )


async def _launch(
    ac: AsyncClient, job_mgr: BacktestJobManager, url: str, body: dict[str, Any]
) -> tuple[Any, Any]:
    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as popen,
    ):
        popen.return_value.pid = 12345
        return await _post(ac, url, body), popen


def _backtest_body(state: StateManager, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy_id": state.create_strategy(name="finite_api", dsl_config={}),
        "symbols": ["BTCUSDT"],
        "timeframe": "1m",
        "start_date": "2025-01-01",
        "end_date": "2025-02-01",
        "parameters": parameters,
    }


_DISCOVERY_BODY: dict[str, Any] = {
    "symbols": ["BTCUSDT"],
    "timeframes": ["4h"],
    "population": 12,
    "generations": 10,
}


# --- backtest screening + validation: non-finite anywhere in parameters ---

_BACKTEST_NON_FINITE = [
    pytest.param({"rsi_period": float("nan")}, "parameters.rsi_period", id="nan-param"),
    pytest.param({"leverage": float("inf")}, "parameters.leverage", id="pos-inf-leverage"),
    pytest.param(
        {"initial_balance": float("-inf")}, "parameters.initial_balance", id="neg-inf-balance"
    ),
    pytest.param(
        {"sma_weights": [1, 2, float("nan")]},
        "parameters.sma_weights.2",
        id="nan-in-list",
    ),
    pytest.param(
        {"nested": [{"x": float("inf")}]},
        "parameters.nested.0.x",
        id="inf-deep-nested",
    ),
]


@pytest.mark.parametrize("endpoint", ["screening", "validation"])
@pytest.mark.parametrize(("parameters", "path"), _BACKTEST_NON_FINITE)
async def test_backtest_non_finite_is_422_named_and_never_launches(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    endpoint: str,
    parameters: dict[str, Any],
    path: str,
) -> None:
    ac, state, job_mgr = client
    before = _run_count(state)

    r, popen = await _launch(
        ac, job_mgr, f"/api/backtest/{endpoint}", _backtest_body(state, parameters)
    )

    assert r.status_code == 422, r.text
    assert f"{path}: {SUFFIX}" in r.text
    assert _run_count(state) == before
    assert popen.call_count == 0


# --- discovery launch: non-finite in a top-level field or a nested list ---

_DISCOVERY_NON_FINITE = [
    pytest.param({"leverage": float("nan")}, "leverage", id="nan-leverage"),
    pytest.param({"leverage": float("inf")}, "leverage", id="pos-inf-leverage"),
    pytest.param({"liquidation_leverage": float("-inf")}, "liquidation_leverage", id="neg-inf-alias"),
    pytest.param(
        {"cross_window_months": [1, float("nan")]},
        "cross_window_months.1",
        id="nan-in-list",
    ),
]


@pytest.mark.parametrize(("override", "path"), _DISCOVERY_NON_FINITE)
async def test_discovery_non_finite_is_422_named_and_never_launches(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    override: dict[str, Any],
    path: str,
) -> None:
    ac, state, job_mgr = client
    before = _run_count(state)

    r, popen = await _launch(
        ac, job_mgr, "/api/discovery/launch", {**_DISCOVERY_BODY, **override}
    )

    assert r.status_code == 422, r.text
    assert f"{path}: {SUFFIX}" in r.text
    assert _run_count(state) == before
    assert popen.call_count == 0


# A non-finite number plus another before-validator error (null leverage) must still
# name the non-finite path: the finite diagnostic is never hidden (review B1).
_DISCOVERY_MIXED = [
    pytest.param(
        {"leverage": None, "mutation_rate": float("nan")}, "mutation_rate", id="null-lev-nan-rate"
    ),
    pytest.param(
        {"liquidation_leverage": None, "leverage": float("inf")}, "leverage", id="null-alias-inf-lev"
    ),
]


@pytest.mark.parametrize(("override", "path"), _DISCOVERY_MIXED)
async def test_discovery_non_finite_not_hidden_by_null_leverage(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    override: dict[str, Any],
    path: str,
) -> None:
    ac, state, job_mgr = client
    before = _run_count(state)

    r, popen = await _launch(
        ac, job_mgr, "/api/discovery/launch", {**_DISCOVERY_BODY, **override}
    )

    assert r.status_code == 422, r.text
    assert f"{path}: {SUFFIX}" in r.text
    assert _run_count(state) == before
    assert popen.call_count == 0


# --- finite / non-numeric bodies keep today's outcome ---


@pytest.mark.parametrize("endpoint", ["screening", "validation"])
async def test_backtest_finite_numbers_still_launch(
    client: tuple[AsyncClient, StateManager, BacktestJobManager], endpoint: str
) -> None:
    ac, state, job_mgr = client

    r, popen = await _launch(
        ac,
        job_mgr,
        f"/api/backtest/{endpoint}",
        _backtest_body(state, {"initial_balance": 1000, "leverage": 20, "rsi_period": 14}),
    )

    assert r.status_code == 201, r.text
    assert popen.call_count == 1


@pytest.mark.parametrize("endpoint", ["screening", "validation"])
async def test_backtest_non_numeric_param_values_unchanged(
    client: tuple[AsyncClient, StateManager, BacktestJobManager], endpoint: str
) -> None:
    """A numeric string / bool / None parameter is not a float and is untouched."""
    ac, state, job_mgr = client

    r, _popen = await _launch(
        ac,
        job_mgr,
        f"/api/backtest/{endpoint}",
        _backtest_body(
            state,
            {"initial_balance": 1000, "rsi_period": "14", "use_filter": True, "note": None},
        ),
    )

    assert r.status_code == 201, r.text


async def test_discovery_finite_numbers_still_launch(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    ac, _state, job_mgr = client

    r, popen = await _launch(
        ac, job_mgr, "/api/discovery/launch", {**_DISCOVERY_BODY, "leverage": 20}
    )

    assert r.status_code == 201, r.text
    assert popen.call_count == 1


# --- any other 422 whose error input holds a non-finite float must serialise ---


def test_validation_error_with_non_finite_input_serialises() -> None:
    """The handler makes a non-finite ``input``/``ctx`` JSON-safe (never a 500)."""
    from fastapi.exceptions import RequestValidationError

    from vibe_quant.api.app import _json_safe_validation_errors

    exc = RequestValidationError(
        [
            {
                "type": "greater_than_equal",
                "loc": ("body", "population"),
                "msg": "Input should be greater than or equal to 2",
                "input": float("nan"),
                "ctx": {"ge": 2},
            }
        ]
    )

    payload = _json_safe_validation_errors(exc)

    assert json.loads(json.dumps(payload, allow_nan=False))["detail"][0]["input"] == "nan"


def test_app_registers_the_non_finite_validation_handler() -> None:
    from fastapi.exceptions import RequestValidationError

    assert RequestValidationError in create_app().exception_handlers
