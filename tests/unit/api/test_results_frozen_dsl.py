"""``GET /api/results/runs/{id}/frozen-dsl``: the DSL a completed validation run executed.

Shadow enrolment (bead vibe-quant-clmmc.27) freezes this DSL. The run never stored its
DSL, only ``notes.validation_input.dsl_sha256``; the endpoint returns the strategy's
stored DSL only while it still hashes to that value, written here by the real writer
(``validation_input_identity``), so a strategy edited since the run is refused.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.bankroll.report import dsl_hasher
from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.dsl.translator import translate_dsl_config
from vibe_quant.validation.runner import validation_input_identity
from vibe_quant.validation.venue import create_venue_config_for_validation

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

EDITOR_DSL: dict[str, object] = {  # the frontend editor's nested format
    "general": {"strategy_type": "momentum", "symbols": ["BTCUSDT"], "timeframe": "1h"},
    "indicators": [{"type": "RSI", "params": {"period": 14}}],
    "conditions": {
        "entry": [{"left": "RSI(14)", "operator": "<", "right": "30"}],
        "exit": [{"left": "RSI(14)", "operator": ">", "right": "70", "logic": None}],
    },
    "risk": {
        "stop_loss": {"type": "fixed_pct", "value": 2},
        "take_profit": {"type": "risk_reward", "value": 2.5},
        "position_sizing": {"type": "percent_equity", "value": 10},
    },
    "time": {},
}
CANONICAL_DSL: dict[str, object] = {
    "name": "frozen_ema",
    "timeframe": "4h",
    "indicators": {"ema": {"type": "EMA", "period": 50}},
    "entry_conditions": {"long": ["close > ema"]},
    "exit_conditions": {"long": ["close < ema"]},
    "stop_loss": {"type": "fixed_pct", "percent": 3.0},
    "take_profit": {"type": "fixed_pct", "percent": 6.0},
}
PARAMS: dict[str, object] = {"leverage": 1, "risk_per_trade": 0.5, "liquidation_mode": "isolated"}
CHANGED = "strategy changed since this run: re-validate first"


class World:
    def __init__(self, client: AsyncClient, sm: StateManager) -> None:
        self.client = client
        self.sm = sm

    def run(
        self,
        name: str,
        dsl: dict[str, object],
        *,
        symbols: tuple[str, ...] = ("BTCUSDT",),
        mode: str = "validation",
        status: str = "completed",
        params: dict[str, object] = PARAMS,
        identity: bool = True,
    ) -> tuple[int, int]:
        sid = self.sm.create_strategy(name, dsl)
        run_id = self.sm.create_backtest_run(
            sid, mode, list(symbols), "1h", "2025-01-01", "2025-06-01", dict(params)
        )
        notes: dict[str, object] = {}
        if identity:  # what ValidationRunner writes: _validate_dsl, then the identity
            ran = validate_strategy_dict(translate_dsl_config(dsl, strategy_name=name))
            config: dict[str, object] = {"symbols": list(symbols), "parameters": dict(params),
                      "start_date": "2025-01-01", "end_date": "2025-06-01"}  # fmt: skip
            notes["validation_input"] = validation_input_identity(
                strategy_name=name, dsl=ran, run_config=config, detail_timeframe="1m",
                latency_preset="cloud", venue_config=create_venue_config_for_validation(),
                entry_order="market",
            )  # fmt: skip
        self.sm.save_backtest_result(run_id, {"total_return": 0.1, "notes": json.dumps(notes)})
        self.sm.update_backtest_run_status(run_id, status)
        return run_id, sid

    def recorded_sha(self, run_id: int) -> str:
        row = self.sm.get_backtest_result(run_id)
        assert row is not None
        return str(json.loads(row["notes"])["validation_input"]["dsl_sha256"])

    async def get(self, run_id: int) -> Any:
        return await self.client.get(f"/api/results/runs/{run_id}/frozen-dsl")


@pytest.fixture()
async def world(tmp_path: Path) -> AsyncIterator[World]:
    app = create_app()
    sm = StateManager(db_path=tmp_path / "state.db")
    _ = sm.conn
    app.state.state_manager = sm
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        try:
            yield World(client, sm)
        finally:
            sm.close()


@pytest.mark.parametrize(
    ("name", "dsl", "expected_name", "expected_tf"),
    [
        ("frozen_rsi", EDITOR_DSL, "frozen_rsi", "1h"),
        ("frozen_ema", CANONICAL_DSL, "frozen_ema", "4h"),
    ],
)
async def test_unedited_run_returns_the_dsl_whose_hash_the_run_recorded(
    world: World, name: str, dsl: dict[str, object], expected_name: str, expected_tf: str
) -> None:
    run_id, sid = world.run(name, dsl)
    resp = await world.get(run_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    sha = world.recorded_sha(run_id)
    assert (body["run_id"], body["strategy_id"], body["strategy_name"], body["dsl_sha256"]) == (
        run_id, sid, name, sha,
    )  # fmt: skip
    frozen = json.loads(body["dsl_json"])
    assert (frozen["name"], frozen["timeframe"]) == (expected_name, expected_tf)
    # The shadow tracker re-translates with ITS row name ("<name> (run N)"): the frozen
    # DSL keeps its own name, so it still hashes to what the run recorded.
    assert dsl_hasher(f"{name} (run {run_id})")(frozen) == sha


async def test_strategy_edited_since_the_run_is_refused(world: World) -> None:
    run_id, sid = world.run("frozen_rsi", EDITOR_DSL)
    edited = json.loads(json.dumps(EDITOR_DSL))
    edited["indicators"][0]["params"]["period"] = 21
    world.sm.update_strategy(sid, dsl_config=edited)
    resp = await world.get(run_id)
    assert (resp.status_code, resp.json()["detail"]) == (422, CHANGED)


@pytest.mark.parametrize("drop_strategy", [False, True])
async def test_run_without_its_strategy_is_not_called_changed(
    world: World, drop_strategy: bool
) -> None:
    """A deleted strategy (or a run with none) never matches: say it no longer exists."""
    run_id, sid = world.run("frozen_rsi", EDITOR_DSL)
    conn = world.sm.conn
    if drop_strategy:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("DELETE FROM strategies WHERE id = ?", (sid,))
        conn.execute("PRAGMA foreign_keys = ON")
        expected = f"strategy {sid} no longer exists"
    else:
        conn.execute("UPDATE backtest_runs SET strategy_id = NULL WHERE id = ?", (run_id,))
        expected = "strategy no longer exists"
    conn.commit()
    resp = await world.get(run_id)
    assert (resp.status_code, resp.json()["detail"]) == (422, expected)


async def test_dsl_override_is_ignored_because_validation_runs_the_stored_dsl(
    world: World,
) -> None:
    override = {**CANONICAL_DSL, "name": "the_override", "timeframe": "15m"}
    run_id, _ = world.run("frozen_rsi", EDITOR_DSL, params={**PARAMS, "dsl_override": override})
    resp = await world.get(run_id)
    assert resp.status_code == 200, resp.text
    frozen = json.loads(resp.json()["dsl_json"])
    assert (frozen["name"], frozen["timeframe"]) == ("frozen_rsi", "1h")


@pytest.mark.parametrize(
    ("kwargs", "detail"),
    [
        ({"mode": "screening"}, "Run {id} is a screening run, not a validation run"),
        ({"status": "failed"}, "Run {id} is failed, not completed"),
        ({"status": "running"}, "Run {id} is running, not completed"),
        ({"identity": False}, "Run {id} recorded no validation inputs: re-validate first"),
    ],
)
async def test_ineligible_runs_are_refused_with_their_reason(
    world: World, kwargs: dict[str, Any], detail: str
) -> None:
    run_id, _ = world.run("frozen_rsi", EDITOR_DSL, **kwargs)
    resp = await world.get(run_id)
    assert (resp.status_code, resp.json()["detail"]) == (422, detail.format(id=run_id))


async def test_missing_run_is_404(world: World) -> None:
    resp = await world.get(999_999)
    assert resp.status_code == 404
