"""Indicator-param guard on the backtest launch endpoints (bead vibe-quant-9pj1w).

``POST /api/backtest/screening`` and ``/api/backtest/validation`` used to accept
a stored strategy whose indicator carries a param its type does not declare
(``EMA`` + ``std_dev``): the run row was created and a job started, and the run
only failed later. The launch now runs the same ``check_strategy_indicator_params``
guard the strategy save runs
(``strategies.py::_reject_unused_indicator_params``) right after
``_ensure_strategy`` and BEFORE the post-only / MARK prechecks, so the request is
a 422 carrying the guard's own message and no run row is created.

Contract rows:

* P1 both endpoints: unused param -> 422 with the save guard's exact text.
* P2 that 422 creates no ``backtest_runs`` row and starts no job.
* P3 a valid strategy still launches (201) and starts exactly one job.
* P4 the indicator check runs before the screening post-only 422.
"""

from __future__ import annotations

from collections.abc import AsyncIterator  # noqa: TCH003
from pathlib import Path  # noqa: TCH003
from typing import Any  # noqa: TCH003
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from vibe_quant.api import mark_precheck
from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager

# The exact text the save guard returns (tests/unit/test_indicator_used_params.py).
EMA_STD_DEV_MSG = "Indicator 'EMA' does not use param 'std_dev'. Valid params: ['period']"

_UNUSED_DSL: dict[str, Any] = {"indicators": {"ema": {"type": "EMA", "period": 20, "std_dev": 2}}}
_VALID_DSL: dict[str, Any] = {"indicators": {"ema": {"type": "EMA", "period": 20}}}


@pytest.fixture()
async def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[Any, ...]]:
    """A temp-DB app whose job manager's ``start_job`` is a counting fake."""
    app = create_app()
    state = StateManager(db_path=tmp_path / "test.db")
    _ = state.conn  # init schema
    jobs = BacktestJobManager(db_path=tmp_path / "test.db")
    ws = ConnectionManager()
    await ws.start()
    start_job = MagicMock(return_value=4242)
    monkeypatch.setattr(jobs, "start_job", start_job)

    app.state.state_manager = state
    app.state.job_manager = jobs
    app.state.ws_manager = ws

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, state, start_job

    await ws.stop()
    jobs.close()
    state.close()


def _body(
    state: StateManager,
    dsl: dict[str, Any],
    *,
    name: str,
    **extra: object,
) -> dict[str, object]:
    body: dict[str, object] = {
        "strategy_id": state.create_strategy(name=name, dsl_config=dsl),
        "symbols": ["BTCUSDT"],
        "timeframe": "1m",
        "start_date": "2025-01-01",
        "end_date": "2025-02-01",
        "parameters": {"initial_balance": 1000},
    }
    body.update(extra)
    return body


def _run_count(state: StateManager) -> int:
    return int(state.conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0])


@pytest.mark.parametrize("endpoint", ["screening", "validation"])
async def test_unused_indicator_param_is_422_before_any_run(client: tuple[Any, ...], endpoint: str) -> None:
    """P1/P2: an unused indicator param 422s with the save guard's text, no row, no job."""
    ac, state, start_job = client
    before = _run_count(state)

    r = await ac.post(
        f"/api/backtest/{endpoint}", json=_body(state, _UNUSED_DSL, name=f"unused_{endpoint}")
    )

    assert r.status_code == 422, r.text
    assert r.json()["detail"] == EMA_STD_DEV_MSG
    assert _run_count(state) == before
    assert start_job.call_count == 0


@pytest.mark.parametrize("endpoint", ["screening", "validation"])
async def test_valid_strategy_still_launches(client: tuple[Any, ...], endpoint: str) -> None:
    """P3: a strategy whose indicator params are all declared still launches."""
    ac, state, start_job = client

    r = await ac.post(
        f"/api/backtest/{endpoint}", json=_body(state, _VALID_DSL, name=f"valid_{endpoint}")
    )

    assert r.status_code == 201, r.text
    assert start_job.call_count == 1
    assert state.get_backtest_run(r.json()["id"]) is not None


async def test_indicator_check_precedes_post_only_422(client: tuple[Any, ...]) -> None:
    """P4: the indicator 422 wins over the screening post-only 422 (check order)."""
    ac, state, start_job = client
    before = _run_count(state)
    body = _body(state, _UNUSED_DSL, name="unused_precedence", entry_order="post_only")

    r = await ac.post("/api/backtest/screening", json=body)

    assert r.status_code == 422, r.text
    assert r.json()["detail"] == EMA_STD_DEV_MSG
    assert "validation-only" not in r.json()["detail"]
    assert _run_count(state) == before
    assert start_job.call_count == 0


@pytest.mark.parametrize("endpoint", ["screening", "validation"])
async def test_unused_indicator_param_422s_before_mark_precheck(
    client: tuple[Any, ...], endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N1: a mode-on launch of a bad-param strategy is the indicator 422, no MARK scan.

    The indicator guard's precedence over the screening post-only 422 is pinned by
    P4; this pins it over the liquidation-mode MARK precheck of BOTH endpoints
    (backtest.py ``launch_screening`` / ``launch_validation``), where the precheck
    is the expensive branch (a cold MARK decode).
    """
    ac, state, start_job = client
    before = _run_count(state)
    precheck_calls: list[tuple[object, ...]] = []

    def _indicator_guard_must_run_first(*args: object) -> None:
        precheck_calls.append(args)
        raise HTTPException(status_code=422, detail="MARK precheck ran before the indicator guard")

    monkeypatch.setattr(mark_precheck, "precheck_screening", _indicator_guard_must_run_first)
    monkeypatch.setattr(mark_precheck, "precheck_validation", _indicator_guard_must_run_first)

    r = await ac.post(
        f"/api/backtest/{endpoint}",
        json=_body(
            state,
            _UNUSED_DSL,
            name=f"unused_precheck_{endpoint}",
            liquidation_mode="isolated",
        ),
    )

    assert precheck_calls == []
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == EMA_STD_DEV_MSG
    assert _run_count(state) == before
    assert start_job.call_count == 0
