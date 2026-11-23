"""API responses carry raw stored Sharpe/Sortino plus computed √365 display twins.

User ruling Q4=A: the stored (√252) value is what the DB, gates and pins keep; the
``*_display`` twin is what a human reads on screen. A twin is computed on the way
out, never stored, and is null whenever its raw value is null or not finite.
"""

from __future__ import annotations

import json
import math
from collections.abc import AsyncIterator  # noqa: TCH003
from datetime import UTC, datetime
from pathlib import Path  # noqa: TCH003
from typing import Any
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager

#: The screening value of strategy 239 on main@efc19e47, and its display twin.
RAW_239 = 0.8004025908124951
DISPLAY_239 = 0.9632846671540376
FACTOR = math.sqrt(365 / 252)


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture()
async def api(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[AsyncClient, StateManager, BacktestJobManager]]:
    # The preflight reads the catalog for the train range; fake it like the sibling test.
    from vibe_quant.api import data_precheck
    from vibe_quant.data.catalog import CatalogManager

    monkeypatch.setattr(data_precheck, "DEFAULT_CATALOG_PATH", tmp_db.parent / "catalog")

    def _range(self: object, symbol: str, interval: str) -> tuple[datetime, datetime] | None:
        if symbol != "BTCUSDT":
            return None
        return datetime(2020, 1, 1, tzinfo=UTC), datetime(2026, 11, 10, tzinfo=UTC)

    monkeypatch.setattr(CatalogManager, "get_bar_date_range", _range)

    app = create_app()
    state = StateManager(db_path=tmp_db)
    _ = state.conn
    job_mgr = BacktestJobManager(db_path=tmp_db)
    ws_mgr = ConnectionManager()
    await ws_mgr.start()
    app.state.state_manager = state
    app.state.job_manager = job_mgr
    app.state.ws_manager = ws_mgr
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, state, job_mgr
    await ws_mgr.stop()
    job_mgr.close()
    state.close()


def _run_with_result(
    state: StateManager,
    *,
    sharpe: float | None,
    sortino: float | None,
    run_mode: str = "validation",
    notes: dict[str, Any] | None = None,
) -> int:
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode=run_mode,
        symbols=["BTCUSDT"],
        timeframe="4h",
        start_date="2024-01-01",
        end_date="2024-12-31",
        parameters={},
    )
    metrics: dict[str, Any] = {"sharpe_ratio": sharpe, "sortino_ratio": sortino}
    if notes is not None:
        metrics["notes"] = json.dumps(notes)
    state.save_backtest_result(run_id, metrics)
    return run_id


async def test_summary_carries_raw_and_display(
    api: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """R6: raw untouched, display = raw × √(365/252), null raw -> null twin, DB unchanged."""
    ac, state, _jobs = api
    run_id = _run_with_result(state, sharpe=RAW_239, sortino=None)
    before = state.get_backtest_result(run_id)

    r = await ac.get(f"/api/results/runs/{run_id}")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sharpe_ratio"] == RAW_239
    assert body["sharpe_display"] == DISPLAY_239
    assert body["sortino_ratio"] is None
    assert body["sortino_display"] is None
    # R6: the twin is computed on the way out; the stored row is untouched.
    assert state.get_backtest_result(run_id) == before


async def test_summary_twin_is_null_for_a_non_finite_raw_value(
    api: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """R7 (NaN raw): the row stores NaN, the response carries null, not null-shaped noise."""
    ac, state, _jobs = api
    run_id = _run_with_result(state, sharpe=math.nan, sortino=0.0)

    r = await ac.get(f"/api/results/runs/{run_id}")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sharpe_display"] is None
    # A finite sibling still converts: the twin rule is per field.
    assert body["sortino_display"] == 0.0


async def test_compare_rows_carry_the_twins(
    api: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    ac, state, _jobs = api
    a = _run_with_result(state, sharpe=RAW_239, sortino=0.5)
    b = _run_with_result(state, sharpe=-0.12, sortino=None)
    result_ids = [int(state.get_backtest_result(rid)["id"]) for rid in (a, b)]  # type: ignore[index]

    r = await ac.get("/api/results/compare", params={"run_ids": ",".join(map(str, result_ids))})

    assert r.status_code == 200, r.text
    rows = r.json()["runs"]
    assert [row["sharpe_display"] for row in rows] == [DISPLAY_239, -0.14442002235542986]
    assert [row["sortino_display"] for row in rows] == [0.6017500931476244, None]


async def test_sweep_rows_carry_the_twins(
    api: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """R7: the sweep list twin follows the same helper, per row."""
    ac, state, _jobs = api
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="screening",
        symbols=["BTCUSDT"],
        timeframe="4h",
        start_date="2024-01-01",
        end_date="2024-12-31",
        parameters={},
    )
    for sharpe, sortino in ((RAW_239, 2.0), (0.0, None)):
        state.save_sweep_result(
            run_id,
            {
                "parameters": {"x": 1},
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "is_pareto_optimal": False,
            },
        )

    r = await ac.get(f"/api/results/runs/{run_id}/sweeps")

    assert r.status_code == 200, r.text
    rows = r.json()
    assert [row["sharpe_ratio"] for row in rows] == [RAW_239, 0.0]
    assert [row["sharpe_display"] for row in rows] == [DISPLAY_239, 0.0]
    assert [row["sortino_display"] for row in rows] == [2.4070003725904976, None]


async def test_run_summary_list_carries_the_twins(
    api: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """R7: the runs table's rows get the same twin from the same helper."""
    ac, state, _jobs = api
    _run_with_result(state, sharpe=RAW_239, sortino=None)
    _run_with_result(state, sharpe=None, sortino=None)

    r = await ac.get("/api/results/runs/summary")

    assert r.status_code == 200, r.text
    rows = r.json()["runs"]
    assert [row["sharpe_ratio"] for row in rows] == [RAW_239, None]
    assert [row["sharpe_display"] for row in rows] == [DISPLAY_239, None]


def _seed_discovery_job(
    state: StateManager, job_mgr: BacktestJobManager, *, sharpe: float | None
) -> int:
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="discovery",
        symbols=["BTCUSDT"],
        timeframe="4h",
        start_date="2025-03-07",
        end_date="2026-03-07",
        parameters={"population": 12, "generations": 10},
    )
    notes = {"top_strategies": [{"sharpe": sharpe, "return_pct": 0.04}]}
    state.save_backtest_result(run_id, {"notes": json.dumps(notes)})
    job_mgr.conn.execute(
        """INSERT INTO background_jobs
           (run_id, pid, job_type, status, started_at, heartbeat_at)
           VALUES (?, ?, 'discovery', 'running', datetime('now'), datetime('now'))""",
        (run_id, 99900 + run_id),
    )
    job_mgr.conn.commit()
    return run_id


async def test_discovery_job_list_carries_the_twin(
    api: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """R7: a discovery job's best_sharpe twin follows the same helper."""
    ac, state, job_mgr = api
    rich = _seed_discovery_job(state, job_mgr, sharpe=RAW_239)
    empty = _seed_discovery_job(state, job_mgr, sharpe=None)

    with patch.object(job_mgr, "is_process_alive", return_value=True):
        r = await ac.get("/api/discovery/jobs")

    assert r.status_code == 200, r.text
    by_run = {job["run_id"]: job for job in r.json()}
    assert by_run[rich]["best_sharpe"] == RAW_239
    assert by_run[rich]["best_sharpe_display"] == DISPLAY_239
    assert by_run[empty]["best_sharpe"] is None
    assert by_run[empty]["best_sharpe_display"] is None


async def test_preflight_bars_carry_the_twin(
    api: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """R7: the power bars a launch form shows are annualised on screen too."""
    ac, _state, _jobs = api

    r = await ac.post(
        "/api/discovery/preflight",
        json={
            "symbols": ["BTCUSDT"],
            "timeframes": ["4h"],
            "start_date": "2023-06-01",
            "end_date": "2026-06-01",
            "population": 50,
            "generations": 20,
        },
    )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["bar_a"]["sharpe"] is not None
    assert body["bar_a"]["sharpe_display"] == body["bar_a"]["sharpe"] * FACTOR
    assert body["bar_b"]["sharpe_display"] == body["bar_b"]["sharpe"] * FACTOR
    # The stored bar and its basis stay √252 (nothing about the gate moves).
    assert body["sharpe_basis"] == "sqrt252"
