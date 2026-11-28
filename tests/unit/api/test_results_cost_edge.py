"""``GET /api/results/runs/{id}`` and ``/sweeps`` carry the run's cost edge (clmmc.11 T3).

Validation runs read ``notes.cost_edge``; screening runs read the best sweep row's four
columns. Missing or unreadable inputs are ``null``, never 0 bps and never NaN (R18, R27).
"""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TCH003
from typing import TYPE_CHECKING, Any

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.db.state_manager import StateManager

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

KEYS = ("turnover", "gross_edge_bps", "cost_paid_bps", "breakeven_fee_bps")
NONE = dict.fromkeys(KEYS)


@pytest.fixture()
async def api(tmp_path: Path) -> AsyncIterator[tuple[AsyncClient, StateManager]]:
    app = create_app()
    state = StateManager(db_path=tmp_path / "test.db")
    _ = state.conn
    app.state.state_manager = state
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, state
    state.close()


def _run(state: StateManager, mode: str) -> int:
    return state.create_backtest_run(
        strategy_id=None,
        run_mode=mode,
        symbols=["BTCUSDT"],
        timeframe="4h",
        start_date="2024-01-01",
        end_date="2024-12-31",
        parameters={},
    )


def _validation_run(state: StateManager, notes_text: str | None) -> int:
    run_id = _run(state, "validation")
    state.save_backtest_result(run_id, {"total_trades": 3, "sharpe_ratio": 0.4}, trades=[])
    if notes_text is not None:
        state.update_result_notes(run_id, notes_text)
    return run_id


async def _cost_edge(client: AsyncClient, run_id: int) -> Any:
    r = await client.get(f"/api/results/runs/{run_id}")
    assert r.status_code == 200, r.text
    return r.json()["cost_edge"]


@pytest.mark.parametrize(
    ("notes_text", "expected"),
    [
        (
            json.dumps({"fee_schedule": {"tier": "VIP0"}, "cost_edge": {
                "turnover": 2000.0, "gross_edge_bps": 55.0,
                "cost_paid_bps": 5.0, "breakeven_fee_bps": 55.0,
            }}),
            {"turnover": 2000.0, "gross_edge_bps": 55.0,
             "cost_paid_bps": 5.0, "breakeven_fee_bps": 55.0},
        ),
        (
            json.dumps({"cost_edge": {
                "turnover": 400.0, "gross_edge_bps": -62.5,
                "cost_paid_bps": 12.5, "breakeven_fee_bps": -61.0,
            }}),
            {"turnover": 400.0, "gross_edge_bps": -62.5,
             "cost_paid_bps": 12.5, "breakeven_fee_bps": -61.0},
        ),
        (  # no fills: turnover 0, every bps n/a
            json.dumps({"cost_edge": {
                "turnover": 0.0, "gross_edge_bps": None,
                "cost_paid_bps": None, "breakeven_fee_bps": None,
            }}),
            {**NONE, "turnover": 0.0},
        ),
        (None, None),
        (json.dumps({"fee_schedule": {"tier": "VIP0"}}), None),
        (json.dumps({"cost_edge": None}), None),
        (json.dumps({"cost_edge": [1]}), None),
        (json.dumps({"cost_edge": "x"}), None),
        (json.dumps({"cost_edge": {"turnover": "x", "cost_paid_bps": 5.0}}),
         {**NONE, "cost_paid_bps": 5.0}),
        ('{"cost_edge": {"turnover": 1e400, "gross_edge_bps": 3.0}}',
         {**NONE, "gross_edge_bps": 3.0}),
        ('{"cost_edge": {"turnover": NaN, "cost_paid_bps": -Infinity}}', NONE),
        ('{"cost_edge": {"turnover": 1' + "0" * 400 + ', "cost_paid_bps": 2.0}}',
         {**NONE, "cost_paid_bps": 2.0}),
        (json.dumps({"cost_edge": {"turnover": True, "breakeven_fee_bps": 7}}),
         {**NONE, "breakeven_fee_bps": 7.0}),
        ("not json", None),
    ],
    ids=[
        "notes-a", "notes-b", "no-fills", "no-notes", "key-absent", "null", "list", "string",
        "field-not-number", "field-overflow-inf", "field-nan", "field-huge-int", "field-bool", "unparsable",
    ],
)  # fmt: skip
async def test_validation_run_reads_notes_cost_edge(
    api: tuple[AsyncClient, StateManager], notes_text: str | None, expected: Any
) -> None:
    """R18/R27: the validation summary reads ``notes.cost_edge`` (no ``top_strategies``)."""
    client, state = api
    assert await _cost_edge(client, _validation_run(state, notes_text)) == expected


async def test_screening_run_reads_the_best_sweep_row(
    api: tuple[AsyncClient, StateManager],
) -> None:
    """R27: the screening summary is the best (Pareto) row's four columns; /sweeps lists
    every row's own values."""
    client, state = api
    run_id = _run(state, "screening")
    state.save_sweep_results_batch(
        run_id,
        [
            {
                "parameters": {"x": 1}, "sharpe_ratio": 0.5, "is_pareto_optimal": False,
                "turnover": 400.0, "gross_edge_bps": -62.5, "cost_paid_bps": 12.5,
                "breakeven_fee_bps": -62.5,
            },
            {
                "parameters": {"x": 2}, "sharpe_ratio": 1.0, "is_pareto_optimal": True,
                "turnover": 1250.0, "gross_edge_bps": 40.0, "cost_paid_bps": 8.0,
                "breakeven_fee_bps": 38.0,
            },
        ],
    )  # fmt: skip

    assert await _cost_edge(client, run_id) == {
        "turnover": 1250.0,
        "gross_edge_bps": 40.0,
        "cost_paid_bps": 8.0,
        "breakeven_fee_bps": 38.0,
    }
    r = await client.get(f"/api/results/runs/{run_id}/sweeps")
    assert r.status_code == 200
    assert [{k: row[k] for k in KEYS} for row in r.json()] == [
        {"turnover": 1250.0, "gross_edge_bps": 40.0, "cost_paid_bps": 8.0,
         "breakeven_fee_bps": 38.0},
        {"turnover": 400.0, "gross_edge_bps": -62.5, "cost_paid_bps": 12.5,
         "breakeven_fee_bps": -62.5},
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        ({}, None),  # written before the columns existed: NULL -> not recorded
        ({"turnover": 0.0}, {**NONE, "turnover": 0.0}),  # no fills
    ],
    ids=["pre-migration", "no-fills"],
)
async def test_screening_run_without_a_recorded_edge(
    api: tuple[AsyncClient, StateManager], columns: dict[str, float], expected: Any
) -> None:
    client, state = api
    run_id = _run(state, "screening")
    state.save_sweep_result(run_id, {"parameters": {"x": 1}, "sharpe_ratio": 1.0, **columns})

    assert await _cost_edge(client, run_id) == expected
