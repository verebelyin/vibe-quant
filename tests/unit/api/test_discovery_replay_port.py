"""R35: ``/replay`` of a port discovery champion runs from its RECORDED DSL and matches.

A real (NautilusTrader) port discovery over two months of BTCUSDT 1h on a scratch DB,
one eval window; the champion is replayed through the API and its screening run
(executed in-process, the CLI's ``cmd_run``) must reproduce the discovery metrics.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import AsyncIterator  # noqa: TCH003
from pathlib import Path  # noqa: TCH003
from typing import Any

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery import run as discovery_run
from vibe_quant.jobs.manager import BacktestJobManager

_PORT: dict[str, Any] = {
    "name": "fast_ema_cross",
    "params": {
        "fast": {"kind": "int", "low": 3, "high": 8, "step": 1, "default": 5},
        "slow": {"kind": "int", "low": 9, "high": 20, "step": 1, "default": 12},
    },
    "constraints": ["fast < slow"],
    "dsl": {
        "name": "fast_ema_cross",
        "timeframe": "1h",
        "indicators": {
            "fast_ema": {"type": "EMA", "period": "{{fast}}"},
            "slow_ema": {"type": "EMA", "period": "{{slow}}"},
        },
        "entry_conditions": {"long": ["fast_ema crosses_above slow_ema"], "short": []},
        "exit_conditions": {"long": ["fast_ema crosses_below slow_ema"], "short": []},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 6.0},
    },
}


@pytest.fixture()
async def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[AsyncClient, Path]]:
    db_path = tmp_path / "scratch.db"
    app = create_app()
    state = StateManager(db_path=db_path)
    _ = state.conn
    jobs = BacktestJobManager(db_path=db_path)
    ws = ConnectionManager()
    await ws.start()
    app.state.state_manager = state
    app.state.job_manager = jobs
    app.state.ws_manager = ws
    # The router asks the job manager to spawn the screening subprocess; the test runs it in-process.
    monkeypatch.setattr(BacktestJobManager, "start_job", lambda self, *_a, **_k: 4242)
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, db_path
    await ws.stop()
    jobs.close()
    state.close()


def _finer_timeframe_port() -> dict[str, Any]:
    """The same EMA cross on 1h whose fast EMA reads 15m bars (a finer additional timeframe)."""
    doc = json.loads(json.dumps(_PORT))
    doc["dsl"]["additional_timeframes"] = ["15m"]
    doc["dsl"]["indicators"]["fast_ema"]["timeframe"] = "15m"
    return doc


def _port_discovery(
    tmp_path: Path, db_path: Path, port: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any]]:
    """Run a real port discovery on the scratch DB; return its run id and result notes."""
    from tests.unit.test_run_discovery import drive_run_discovery

    folder = tmp_path / "port"
    folder.mkdir()
    (folder / "port.yaml").write_text(yaml.safe_dump(port or _PORT, sort_keys=False), encoding="utf-8")
    state = StateManager(db_path)
    run_id = state.create_backtest_run(
        None, "discovery", ["BTCUSDT"], "1h", "2024-10-15", "2024-12-15", {}
    )
    state.close()
    argv = [
        "--run-id", str(run_id), "--population-size", "4", "--max-generations", "2",
        "--elite-count", "1", "--symbols", "BTCUSDT", "--timeframe", "1h",
        "--start-date", "2024-10-15", "--end-date", "2024-12-15",
        "--eval-windows", "1", "--train-test-split", "0", "--seed", "11",
        "--no-dsr", "--no-bootstrap-ci", "--no-buy-hold-gate",
        "--max-workers", "-1", "--db", str(db_path), "--port", str(folder),
    ]  # fmt: skip
    outcome = drive_run_discovery(db_path, run_id, argv, discovery_run.NTBacktests())
    assert outcome.exit_code == 0, outcome.message
    state = StateManager(db_path)
    result = state.get_backtest_result(run_id)
    state.close()
    assert result is not None
    return run_id, json.loads(result["notes"])


@pytest.mark.slow(reason="a real NT port discovery plus an in-process screening replay on BTCUSDT 1h")
@pytest.mark.parametrize("finer_timeframe", [False, True])
async def test_replay_of_a_port_champion_matches_its_discovery_result(
    api: tuple[AsyncClient, Path], tmp_path: Path, finer_timeframe: bool
) -> None:
    from vibe_quant.screening.__main__ import cmd_run

    client, db_path = api
    run_id, notes = _port_discovery(
        tmp_path, db_path, _finer_timeframe_port() if finer_timeframe else None
    )
    champion = notes["top_strategies"][0]
    assert champion["chromosome"]["kind"] == "param"

    response = await client.post(f"/api/discovery/results/{run_id}/replay/0")
    assert response.status_code == 201, response.text
    replay_id = response.json()["replay_run_id"]
    assert response.json()["metrics_note"] is None  # one eval window: comparable numbers

    state = StateManager(db_path)
    replay_row = state.get_backtest_run(replay_id)
    assert replay_row is not None
    override = replay_row["parameters"]["dsl_override"]
    recorded = champion["dsl"]
    # the replay executes the recorded DSL, whose indicator periods are the champion's own values
    values = champion["chromosome"]["values"]
    for dsl in (recorded, override):
        assert {name: spec["period"] for name, spec in dsl["indicators"].items()} == {
            "fast_ema": values["fast"],
            "slow_ema": values["slow"],
        }
    for key in ("entry_conditions", "exit_conditions", "stop_loss", "take_profit"):
        assert override[key] == recorded[key]

    assert cmd_run(argparse.Namespace(run_id=replay_id, db=str(db_path))) == 0
    rows = state.get_sweep_results(replay_id)
    state.close()
    assert len(rows) == 1
    replayed = rows[0]
    assert replayed["total_trades"] == champion["trades"]
    assert replayed["sharpe_ratio"] == pytest.approx(champion["sharpe"], abs=1e-12)
    assert replayed["total_return"] == pytest.approx(champion["return_pct"], abs=1e-12)
    assert replayed["max_drawdown"] == pytest.approx(champion["max_dd"], abs=1e-12)
    assert replayed["profit_factor"] == pytest.approx(champion["pf"], abs=1e-12)
