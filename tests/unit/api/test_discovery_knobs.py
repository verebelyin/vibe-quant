"""Discovery launch pass-through tests for E4 diversity + E5 warm-start knobs."""

from __future__ import annotations

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
        yield ac, state_mgr, job_mgr

    await ws_mgr.stop()
    job_mgr.close()
    state_mgr.close()


_BASE_BODY = {
    "symbols": ["BTCUSDT"],
    "timeframes": ["4h"],
    "population": 12,
    "generations": 10,
    "mutation_rate": 0.15,
    "crossover_rate": 0.7,
    "elite_count": 2,
    "tournament_size": 3,
    "convergence_generations": 5,
}


async def test_diversity_knobs_pass_through_to_params_and_command(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """E4: immigrant/entropy/crowding land in run params AND CLI command."""
    ac, state, job_mgr = client

    body = {
        **_BASE_BODY,
        "immigrant_fraction": 0.1,
        "entropy_threshold": 0.55,
        "crowding_enabled": False,
    }

    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 12345
        r = await ac.post("/api/discovery/launch", json=body)

    assert r.status_code == 201, r.text
    run = state.get_backtest_run(r.json()["run_id"])
    assert run is not None
    params = run["parameters"]
    assert params["immigrant_fraction"] == 0.1
    assert params["entropy_threshold"] == 0.55
    assert params["crowding_enabled"] is False

    # CLI command should include the custom values / no-crowding flag
    args = mock_popen.call_args.args[0]
    assert "--immigrant-fraction" in args
    assert args[args.index("--immigrant-fraction") + 1] == "0.1"
    assert "--entropy-threshold" in args
    assert args[args.index("--entropy-threshold") + 1] == "0.55"
    assert "--no-crowding" in args


async def test_diversity_defaults_dont_add_cli_flags(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """Defaults (0.15 / 0.4 / crowding on) don't inject redundant CLI flags."""
    ac, _state, job_mgr = client

    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 12345
        r = await ac.post("/api/discovery/launch", json=_BASE_BODY)

    assert r.status_code == 201, r.text
    args = mock_popen.call_args.args[0]
    assert "--immigrant-fraction" not in args
    assert "--entropy-threshold" not in args
    assert "--no-crowding" not in args


async def test_warm_start_400_when_seed_run_missing(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """E5: unknown seed_run_id rejected with 400."""
    ac, _state, job_mgr = client
    body = {**_BASE_BODY, "seed_run_id": 99999}

    with patch.object(job_mgr, "is_process_alive", return_value=True):
        r = await ac.post("/api/discovery/launch", json=body)

    assert r.status_code == 400
    assert "Seed run 99999" in r.json()["detail"]


async def test_warm_start_400_when_compiler_mismatch(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """E5: seed run with stale compiler_version rejected."""
    ac, state, job_mgr = client

    # Seed an existing discovery run with a stale compiler version
    seed_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="discovery",
        symbols=["BTCUSDT"],
        timeframe="4h",
        start_date="2025-01-01",
        end_date="2025-04-01",
        parameters={"compiler_version": "deadbeef-stale"},
    )
    state.update_backtest_run_status(seed_id, "completed")

    body = {**_BASE_BODY, "seed_run_id": seed_id}

    with patch.object(job_mgr, "is_process_alive", return_value=True):
        r = await ac.post("/api/discovery/launch", json=body)

    assert r.status_code == 400
    assert "compiler_version" in r.json()["detail"]


async def test_warm_start_passes_through_when_compiler_matches(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """E5: matching compiler_version launches and forwards --seed-from-run."""
    ac, state, job_mgr = client
    from vibe_quant.dsl.compiler import compiler_version_hash

    current = compiler_version_hash()
    seed_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="discovery",
        symbols=["BTCUSDT"],
        timeframe="4h",
        start_date="2025-01-01",
        end_date="2025-04-01",
        parameters={"compiler_version": current},
    )
    state.update_backtest_run_status(seed_id, "completed")

    body = {**_BASE_BODY, "seed_run_id": seed_id}

    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 12345
        r = await ac.post("/api/discovery/launch", json=body)

    assert r.status_code == 201, r.text
    run = state.get_backtest_run(r.json()["run_id"])
    assert run is not None
    assert run["parameters"]["seed_run_id"] == seed_id

    args = mock_popen.call_args.args[0]
    assert "--seed-from-run" in args
    assert args[args.index("--seed-from-run") + 1] == str(seed_id)
