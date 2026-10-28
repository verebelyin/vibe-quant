"""Discovery launch sizing API (bead vibe-quant-2gyk1.2, task S4).

Binding rows: S27 (top-level ``max_position_pct`` / ``risk_per_trade`` on
:class:`DiscoveryLaunchRequest`; forwarded and stored only when non-default; a cap above the
default leverage is a 422), S27b (an explicit ``null`` on a sizing field, with an independently
bad leverage, reports BOTH field errors), S27c (a non-finite leverage is named alone, first) and
S27d (an explicit ``null`` risk is a 422).
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

_OK_BODY: dict[str, Any] = {
    "symbols": ["BTCUSDT"],
    "timeframes": ["4h"],
    "start_date": "2024-01-01",
    "end_date": "2025-01-01",
}


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


async def _launch(
    ac: AsyncClient, job_mgr: BacktestJobManager, body: dict[str, Any]
) -> tuple[Any, list[str] | None]:
    """POST a launch, capturing the exact argv the subprocess would get."""
    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 12345
        r = await ac.post("/api/discovery/launch", json=body)
        argv = mock_popen.call_args.args[0] if mock_popen.call_args else None
    return r, argv


def _stored_params(state: StateManager, run_id: int) -> dict[str, object]:
    run = state.get_backtest_run(run_id)
    assert run is not None
    params = run["parameters"]
    assert isinstance(params, dict)
    return params


async def test_forwarding_and_storage(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """S27: non-default sizing is forwarded as argv AND stored; defaults add neither."""
    ac, state, job_mgr = client

    r, argv = await _launch(ac, job_mgr, {**_OK_BODY, "max_position_pct": 1.0, "risk_per_trade": 0.01})
    assert r.status_code == 201, r.text
    assert argv is not None
    assert argv[argv.index("--max-position-pct") + 1] == "1.0"
    assert argv[argv.index("--risk-per-trade") + 1] == "0.01"
    stored = _stored_params(state, r.json()["run_id"])
    assert stored["max_position_pct"] == 1.0
    assert type(stored["max_position_pct"]) is float
    assert stored["risk_per_trade"] == 0.01
    assert type(stored["risk_per_trade"]) is float

    r2, argv2 = await _launch(ac, job_mgr, _OK_BODY)
    assert r2.status_code == 201, r2.text
    assert argv2 is not None
    assert "--max-position-pct" not in argv2
    assert "--risk-per-trade" not in argv2
    default_stored = _stored_params(state, r2.json()["run_id"])
    assert "max_position_pct" not in default_stored
    assert "risk_per_trade" not in default_stored


async def test_explicit_default_stores_nothing(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """S2: writing the defaults explicitly is byte-identical to omitting them."""
    ac, state, job_mgr = client

    r, argv = await _launch(
        ac, job_mgr, {**_OK_BODY, "max_position_pct": 0.5, "risk_per_trade": 0.02}
    )
    assert r.status_code == 201, r.text
    assert argv is not None
    assert "--max-position-pct" not in argv
    assert "--risk-per-trade" not in argv
    stored = _stored_params(state, r.json()["run_id"])
    assert "max_position_pct" not in stored
    assert "risk_per_trade" not in stored


async def test_exposure_above_default_leverage_is_422(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """S27: a cap above the default leverage 10 is a 422 before any run row."""
    ac, state, _job_mgr = client
    before = int(state.conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0])

    r = await ac.post("/api/discovery/launch", json={**_OK_BODY, "max_position_pct": 12.0})
    assert r.status_code == 422
    assert "Value error, max_position_pct (12.0) must not exceed leverage (10.0)" in r.text
    assert (
        int(state.conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0]) == before
    )


async def test_exposure_at_leverage_is_accepted(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """S13: a cap exactly at the leverage is fully marginable (equality allowed)."""
    ac, _state, job_mgr = client

    r, argv = await _launch(ac, job_mgr, {**_OK_BODY, "max_position_pct": 10.0})
    assert r.status_code == 201, r.text
    assert argv is not None
    assert argv[argv.index("--max-position-pct") + 1] == "10.0"


async def test_null_sizing_and_bad_leverage_both_reported(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """S27b: an explicit null sizing AND a bad leverage each report their own error."""
    ac, _state, _job_mgr = client

    r = await ac.post(
        "/api/discovery/launch", json={**_OK_BODY, "max_position_pct": None, "leverage": 2.5}
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert isinstance(detail, list)
    msgs = {tuple(err["loc"]): err["msg"] for err in detail}
    assert msgs[("body", "max_position_pct")] == (
        "Value error, max_position_pct must be a number in (0, 125], got None"
    )
    assert msgs[("body", "leverage")] == (
        "Value error, leverage must be a whole number in [1, 125], got 2.5"
    )


async def test_null_sizing_with_nan_leverage_names_nan(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """S27c: the non-finite guard aborts first, naming the leverage alone."""
    ac, _state, _job_mgr = client

    body = {**_OK_BODY, "max_position_pct": None, "leverage": float("nan")}
    r = await ac.post(
        "/api/discovery/launch",
        content=json.dumps(body),
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert isinstance(detail, list)
    assert len(detail) == 1
    assert detail[0]["msg"] == "Value error, leverage: must be a finite number"


async def test_null_risk_is_422(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """S27d: an explicit null risk is a 422, never a silent default."""
    ac, _state, _job_mgr = client

    r = await ac.post("/api/discovery/launch", json={**_OK_BODY, "risk_per_trade": None})
    assert r.status_code == 422
    assert (
        "Value error, risk_per_trade must be more than 0 and at most 0.5 "
        "(50% of equity per trade), got None"
    ) in r.text


# ---------------------------------------------------------------------------
# T6 (bead vibe-quant-t8nrv): the run-level margin limit on the discovery API
# ---------------------------------------------------------------------------

MARGIN_MESSAGE = "max_margin_pct must be a number in (0, 1], got"


async def test_margin_limit_forwarded_and_stored(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """T6/B5a: a run-level margin limit is forwarded as ``--max-margin-pct`` AND stored
    as a float; an omitted one adds neither argv nor a stored key."""
    ac, state, job_mgr = client

    r, argv = await _launch(ac, job_mgr, {**_OK_BODY, "max_margin_pct": 0.5})
    assert r.status_code == 201, r.text
    assert argv is not None
    assert argv[argv.index("--max-margin-pct") + 1] == "0.5"
    stored = _stored_params(state, r.json()["run_id"])
    assert stored["max_margin_pct"] == 0.5
    assert type(stored["max_margin_pct"]) is float

    r2, argv2 = await _launch(ac, job_mgr, _OK_BODY)
    assert r2.status_code == 201, r2.text
    assert argv2 is not None
    assert "--max-margin-pct" not in argv2
    assert "max_margin_pct" not in _stored_params(state, r2.json()["run_id"])


@pytest.mark.parametrize(
    ("value", "message"),
    [
        # the margin limit's domain is (0, 1] and its default is "off", so an
        # explicit null is a 422 rather than a silent off (the shared from_params rule)
        pytest.param(None, MARGIN_MESSAGE, id="null"),
        pytest.param(1.5000001, MARGIN_MESSAGE, id="just-over-max"),
    ],
)
async def test_bad_margin_limit_is_422_no_run(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    value: object,
    message: str,
) -> None:
    """T6: a value outside ``(0, 1]`` is a 422 naming the field, before any run row."""
    ac, state, _job_mgr = client
    before = int(state.conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0])

    r = await ac.post("/api/discovery/launch", json={**_OK_BODY, "max_margin_pct": value})

    assert r.status_code == 422, r.text
    assert f"Value error, {message}" in r.text
    assert f"got {value!r}" in r.text
    assert int(state.conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0]) == before


@pytest.mark.parametrize("value", [1e-9, 1])
async def test_margin_limit_domain_accepted(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    value: float,
) -> None:
    """T6: the equality edge is inclusive — a limit of exactly 1.0 (100% of equity) is
    accepted, and it is stored as the same float."""
    ac, state, job_mgr = client

    r, argv = await _launch(ac, job_mgr, {**_OK_BODY, "max_margin_pct": value})

    assert r.status_code == 201, r.text
    assert argv is not None
    assert argv[argv.index("--max-margin-pct") + 1] == str(float(value))
    assert _stored_params(state, r.json()["run_id"])["max_margin_pct"] == float(value)
