"""Run-level sizing settings on the backtest launch API (bead vibe-quant-2gyk1.2, task S5).

``POST /api/backtest/screening`` and ``/validation`` take ``parameters.max_position_pct`` /
``parameters.risk_per_trade`` through the ONE shared rule
(:mod:`vibe_quant.sizing_settings`, S1): an absent key means the shared default, an explicit
``null`` / bool / string / out-of-range number is a 422 that names ``parameters`` before any
run row exists, and the exposure cap may not exceed the run's resolved margin leverage (10.0
when ``parameters.leverage`` is absent) in every liquidation mode.

S2b: the run stores a sizing key only when it differs from the shared default, so an
explicit-default launch is byte-identical to a default launch.

S14: a sizing key in ``parameters.sweep`` is a 422 (the runner rejects it at run time too),
checked AFTER the sizing keys themselves (S13b order: leverage -> initial_balance -> sizing
-> sweep keys).

NaN/Infinity are not JSON values and are NOT tested through ``parameters`` here: the shared
``reject_non_finite`` body walker fires first with its own path (S8) -- see
``test_launch_non_finite.py``. This file pins that a sizing key carrying one still gets that
422 rather than a 201.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator  # noqa: TCH003
from itertools import count
from pathlib import Path  # noqa: TCH003
from typing import Any  # noqa: TCH003
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api import mark_precheck
from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager

ENDPOINTS = ("screening", "validation")
MODES = ("off", "isolated", "cross")

CAP_MESSAGE = "parameters: max_position_pct must be a number in (0, 125], got"
RISK_MESSAGE = (
    "parameters: risk_per_trade must be more than 0 and at most 0.5 "
    "(50% of equity per trade), got"
)
FINITE_MESSAGE = "parameters.max_position_pct: must be a finite number"

_STRATEGY_SEQ = count()


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture()
async def client(
    tmp_db: Path,
) -> AsyncIterator[tuple[AsyncClient, StateManager, BacktestJobManager, Path]]:
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


@pytest.fixture(autouse=True)
def mark_precheck_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, ...]]:
    """Stub the MARK-coverage precheck (a mode-on launch would open the real catalog).

    A 422 request never reaches it, and the accepted mode-on launches here only assert
    stored parameters, so the call log is not inspected.
    """
    calls: list[tuple[object, ...]] = []

    def _record(*args: object) -> None:
        calls.append(args)

    monkeypatch.setattr(mark_precheck, "check_mark_coverage", _record)
    return calls


def _body(
    state: StateManager,
    *,
    parameters: dict[str, Any],
    mode: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "strategy_id": state.create_strategy(
            name=f"sizing_api_{next(_STRATEGY_SEQ)}", dsl_config={}
        ),
        "symbols": ["BTCUSDT"],
        "timeframe": "1m",
        "start_date": "2025-01-01",
        "end_date": "2025-02-01",
        "parameters": parameters,
    }
    if mode is not None:
        body["liquidation_mode"] = mode
    return body


def _run_count(state: StateManager) -> int:
    return int(state.conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0])


async def _launch(
    ac: AsyncClient,
    job_mgr: BacktestJobManager,
    endpoint: str,
    body: dict[str, Any],
) -> tuple[Any, Any]:
    """POST a raw JSON body (httpx's ``json=`` refuses NaN) and return the Popen mock."""
    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as popen,
    ):
        popen.return_value.pid = 12345
        response = await ac.post(
            f"/api/backtest/{endpoint}",
            content=json.dumps(body),
            headers={"content-type": "application/json"},
        )
    return response, popen


def _stored(state: StateManager, response: Any) -> dict[str, Any]:
    run = state.get_backtest_run(response.json()["id"])
    assert run is not None
    parameters = run["parameters"]
    assert isinstance(parameters, dict)
    return parameters


# --- S5/S6/S7/S9/S10/S11/S12: the shared domain, rejected as a 422 naming parameters ---


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        pytest.param("max_position_pct", "0.5", CAP_MESSAGE, id="cap-str"),
        pytest.param("risk_per_trade", "0.02", RISK_MESSAGE, id="risk-str"),
        pytest.param("max_position_pct", None, CAP_MESSAGE, id="cap-null"),
        pytest.param("risk_per_trade", None, RISK_MESSAGE, id="risk-null"),
        pytest.param("max_position_pct", True, CAP_MESSAGE, id="cap-true"),
        pytest.param("risk_per_trade", False, RISK_MESSAGE, id="risk-false"),
        pytest.param("max_position_pct", 0, CAP_MESSAGE, id="cap-zero"),
        pytest.param("max_position_pct", -0.0, CAP_MESSAGE, id="cap-negative-zero"),
        pytest.param("risk_per_trade", -1, RISK_MESSAGE, id="risk-negative"),
        pytest.param("max_position_pct", 125.0000001, CAP_MESSAGE, id="cap-just-over-max"),
        pytest.param("risk_per_trade", 0.5000001, RISK_MESSAGE, id="risk-just-over-max"),
        pytest.param("max_position_pct", 10**400, CAP_MESSAGE, id="cap-huge-int"),
        pytest.param("max_position_pct", [], CAP_MESSAGE, id="cap-list"),
        pytest.param("risk_per_trade", {}, RISK_MESSAGE, id="risk-dict"),
    ],
)
async def test_invalid_sizing_is_422_and_never_launches(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
    endpoint: str,
    key: str,
    value: object,
    message: str,
) -> None:
    """S5/S7/S9-S12: a bad sizing value is a 422 naming ``parameters``, no run row, no job."""
    ac, state, job_mgr, _db = client
    before = _run_count(state)

    r, popen = await _launch(ac, job_mgr, endpoint, _body(state, parameters={key: value}))

    assert r.status_code == 422, r.text
    assert f"Value error, {message}" in r.text
    assert f"got {value!r}" in r.text
    assert _run_count(state) == before
    assert popen.call_count == 0


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="pos-inf"),
        pytest.param(float("-inf"), id="neg-inf"),
    ],
)
async def test_nan_is_422(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
    endpoint: str,
    value: float,
) -> None:
    """S8: the non-finite body walker fires first, naming the dotted path."""
    ac, state, job_mgr, _db = client
    before = _run_count(state)

    r, popen = await _launch(
        ac, job_mgr, endpoint, _body(state, parameters={"max_position_pct": value})
    )

    assert r.status_code == 422, r.text
    assert FINITE_MESSAGE in r.text
    assert _run_count(state) == before
    assert popen.call_count == 0


# --- S13: the cap may not exceed the run's resolved leverage, in EVERY mode ---


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("parameters", [{"max_position_pct": 20}, {"max_position_pct": 20.0}])
async def test_exposure_above_default_leverage_is_422(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
    endpoint: str,
    mode: str,
    parameters: dict[str, Any],
) -> None:
    """S13: cap 20 with ``parameters.leverage`` absent (10.0) -> 422 in every mode."""
    ac, state, job_mgr, _db = client
    before = _run_count(state)

    r, popen = await _launch(
        ac, job_mgr, endpoint, _body(state, parameters=parameters, mode=mode)
    )

    assert r.status_code == 422, r.text
    assert (
        "Value error, parameters: max_position_pct (20.0) must not exceed leverage (10.0)"
        in r.text
    )
    assert _run_count(state) == before
    assert popen.call_count == 0


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    ("parameters", "stored_cap"),
    [
        # the equality edge: a cap exactly at the run's leverage is fully marginable
        pytest.param({"max_position_pct": 10, "leverage": 10}, 10.0, id="cap-at-leverage"),
        pytest.param({"max_position_pct": 125, "leverage": 125}, 125.0, id="cap-at-max"),
        pytest.param({"max_position_pct": 10}, 10.0, id="cap-at-default-leverage"),
    ],
)
async def test_exposure_at_leverage_is_accepted(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
    endpoint: str,
    mode: str,
    parameters: dict[str, Any],
    stored_cap: float,
) -> None:
    """S13/S10: cap == leverage launches, and the cap is stored as a float."""
    ac, state, job_mgr, _db = client

    r, popen = await _launch(
        ac, job_mgr, endpoint, _body(state, parameters=parameters, mode=mode)
    )

    assert r.status_code == 201, r.text
    assert popen.call_count == 1
    assert _stored(state, r)["max_position_pct"] == stored_cap


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("mode", MODES)
async def test_exposure_just_above_leverage_is_422(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
    endpoint: str,
    mode: str,
) -> None:
    """S13: 10.000001 against a leverage of 10 is the first rejected step."""
    ac, state, job_mgr, _db = client
    before = _run_count(state)

    r, popen = await _launch(
        ac,
        job_mgr,
        endpoint,
        _body(state, parameters={"max_position_pct": 10.000001, "leverage": 10}, mode=mode),
    )

    assert r.status_code == 422, r.text
    assert "must not exceed leverage (10.0)" in r.text
    assert _run_count(state) == before
    assert popen.call_count == 0


# --- S13b/S14: the order of the checks inside the request model ---


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("cap", [20, -1, "abc"])
async def test_leverage_error_wins(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
    endpoint: str,
    cap: object,
) -> None:
    """S13b: an invalid leverage AND an invalid cap report the leverage only."""
    ac, state, job_mgr, _db = client
    before = _run_count(state)

    r, popen = await _launch(
        ac,
        job_mgr,
        endpoint,
        _body(state, parameters={"leverage": 200, "max_position_pct": cap}),
    )

    assert r.status_code == 422, r.text
    assert "parameters.leverage: leverage must be a whole number in [1, 125]" in r.text
    # the 422 body echoes the input, so assert on the MESSAGES: neither sizing
    # message (out-of-domain, nor above-leverage) is reported (S13b)
    assert "must be a number in (0, 125]" not in r.text
    assert "must not exceed leverage" not in r.text
    assert _run_count(state) == before
    assert popen.call_count == 0


@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_balance_error_wins(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
    endpoint: str,
) -> None:
    """S13b order: an invalid initial_balance AND an invalid cap report the balance only."""
    ac, state, job_mgr, _db = client
    before = _run_count(state)

    r, popen = await _launch(
        ac,
        job_mgr,
        endpoint,
        _body(state, parameters={"initial_balance": 0, "max_position_pct": 20}),
    )

    assert r.status_code == 422, r.text
    assert (
        "Value error, parameters.initial_balance: initial_balance must be a whole number "
        "in [1, 1000000000], got 0"
    ) in r.text
    # the 422 body echoes the input, so assert on the MESSAGES: neither sizing
    # diagnostic (out-of-domain, nor above-leverage) is reported (S13b)
    assert "must be a number in (0, 125]" not in r.text
    assert "must not exceed leverage" not in r.text
    assert _run_count(state) == before
    assert popen.call_count == 0


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("key", ["max_position_pct", "risk_per_trade"])
async def test_sweep_sizing_key_is_422(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
    endpoint: str,
    key: str,
) -> None:
    """S14: a run-level sizing setting in ``parameters.sweep`` is refused before any run."""
    ac, state, job_mgr, _db = client
    before = _run_count(state)

    r, popen = await _launch(
        ac, job_mgr, endpoint, _body(state, parameters={"sweep": {key: [1, 2]}})
    )

    assert r.status_code == 422, r.text
    assert (
        f"Value error, parameters.sweep: {key} is a run-level sizing setting "
        f"(parameters.{key}), not a sweep parameter"
    ) in r.text
    assert _run_count(state) == before
    assert popen.call_count == 0


@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_sizing_error_wins_over_sweep_error(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
    endpoint: str,
) -> None:
    """S13b order: the sizing keys themselves are checked BEFORE the sweep keys."""
    ac, state, job_mgr, _db = client
    before = _run_count(state)

    r, popen = await _launch(
        ac,
        job_mgr,
        endpoint,
        _body(state, parameters={"max_position_pct": 20, "sweep": {"risk_per_trade": [1]}}),
    )

    assert r.status_code == 422, r.text
    assert "must not exceed leverage (10.0)" in r.text
    assert "is a run-level sizing setting" not in r.text
    assert _run_count(state) == before
    assert popen.call_count == 0


@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_other_sweep_keys_still_launch(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
    endpoint: str,
) -> None:
    """S14: a sweep of ordinary strategy parameters is untouched."""
    ac, state, job_mgr, _db = client

    r, popen = await _launch(
        ac, job_mgr, endpoint, _body(state, parameters={"sweep": {"rsi_period": [14, 20]}})
    )

    assert r.status_code == 201, r.text
    assert popen.call_count == 1
    assert _stored(state, r)["sweep"] == {"rsi_period": [14, 20]}


# --- S2b/S4: what the run stores ---


@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_explicit_default_stores_nothing(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
    endpoint: str,
) -> None:
    """S2b: an explicit-default launch stores byte-identically to a default launch."""
    ac, state, job_mgr, _db = client

    default_response, _ = await _launch(
        ac, job_mgr, endpoint, _body(state, parameters={"initial_balance": 1000})
    )
    explicit_response, _ = await _launch(
        ac,
        job_mgr,
        endpoint,
        _body(
            state,
            parameters={
                "initial_balance": 1000,
                "max_position_pct": 0.5,
                "risk_per_trade": 0.02,
            },
        ),
    )

    assert default_response.status_code == 201, default_response.text
    assert explicit_response.status_code == 201, explicit_response.text
    assert _stored(state, default_response) == {"initial_balance": 1000}
    assert _stored(state, explicit_response) == {"initial_balance": 1000}


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize(
    ("parameters", "stored"),
    [
        pytest.param({"max_position_pct": 1}, {"max_position_pct": 1.0}, id="cap-int-1"),
        pytest.param({"max_position_pct": 2.0}, {"max_position_pct": 2.0}, id="cap-2.0"),
        pytest.param({"risk_per_trade": 1e-9}, {"risk_per_trade": 1e-9}, id="risk-tiny"),
        pytest.param(
            {"max_position_pct": 0.75, "risk_per_trade": 0.02},
            {"max_position_pct": 0.75},
            id="cap-only-moved",
        ),
    ],
)
async def test_non_default_sizing_is_stored_as_float(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
    endpoint: str,
    parameters: dict[str, Any],
    stored: dict[str, float],
) -> None:
    """S3/S4/S2b: a moved key is stored as a float; a default-valued key is not stored."""
    ac, state, job_mgr, _db = client
    body = {"initial_balance": 1000, **parameters}

    r, popen = await _launch(ac, job_mgr, endpoint, _body(state, parameters=body))

    assert r.status_code == 201, r.text
    assert popen.call_count == 1
    stored_params = _stored(state, r)
    assert stored_params == {"initial_balance": 1000, **stored}
    for key in stored:
        assert type(stored_params[key]) is float
