"""Discovery launch pass-through tests for E4 diversity + E5 warm-start knobs."""

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


def _seed_chromosome() -> dict[str, object]:
    """A minimally valid serialized chromosome, via the real serializer."""
    from vibe_quant.discovery.genome import chromosome_to_serializable
    from vibe_quant.discovery.operators import (
        ConditionType,
        Direction,
        StrategyChromosome,
        StrategyGene,
    )

    chrom = StrategyChromosome(
        entry_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 30.0)],
        exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0)],
        stop_loss_pct=2.0,
        take_profit_pct=3.0,
        direction=Direction.LONG,
    )
    return chromosome_to_serializable(chrom)


def _seed_discovery_run(state: StateManager) -> int:
    """Create a completed discovery run whose compiler hash matches current."""
    from vibe_quant.dsl.compiler import compiler_version_hash

    seed_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="discovery",
        symbols=["BTCUSDT"],
        timeframe="4h",
        start_date="2025-01-01",
        end_date="2025-04-01",
        parameters={"compiler_version": compiler_version_hash()},
    )
    state.update_backtest_run_status(seed_id, "completed")
    return seed_id


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
    assert r.json()["detail"] == "Seed run 99999 not found"


async def test_warm_start_400_when_compiler_mismatch(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """E5: seed run with stale compiler_version rejected."""
    ac, state, job_mgr = client
    from vibe_quant.dsl.compiler import compiler_version_hash

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
    assert r.json()["detail"] == (
        f"Seed run {seed_id} compiler_version=deadbeef-stale does not match "
        f"current {compiler_version_hash()}; warm-start would be stale"
    )


async def test_bootstrap_ci_flags_pass_through_to_params_and_command(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """bd-l8ka: no_bootstrap_ci + bootstrap_min_sharpe land in run params AND CLI command."""
    ac, state, job_mgr = client

    body = {
        **_BASE_BODY,
        "no_bootstrap_ci": True,
        "bootstrap_min_sharpe": 0.75,
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
    assert params["no_bootstrap_ci"] is True
    assert params["bootstrap_min_sharpe"] == 0.75

    args = mock_popen.call_args.args[0]
    assert "--no-bootstrap-ci" in args
    assert "--bootstrap-min-sharpe" in args
    assert args[args.index("--bootstrap-min-sharpe") + 1] == "0.75"


async def test_bootstrap_ci_defaults_dont_add_cli_flags(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """Defaults (no_bootstrap_ci=False, bootstrap_min_sharpe=None) don't inject flags."""
    ac, _state, job_mgr = client

    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 12345
        r = await ac.post("/api/discovery/launch", json=_BASE_BODY)

    assert r.status_code == 201, r.text
    args = mock_popen.call_args.args[0]
    assert "--no-bootstrap-ci" not in args
    assert "--bootstrap-min-sharpe" not in args


async def test_warm_start_passes_through_when_compiler_matches(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """E5: matching compiler_version + a loadable champion launches & forwards --seed-from-run."""
    ac, state, job_mgr = client

    seed_id = _seed_discovery_run(state)
    state.save_backtest_result(
        seed_id,
        {"notes": json.dumps({"top_strategies": [{"chromosome": _seed_chromosome()}]})},
    )

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


@pytest.mark.parametrize(
    "notes",
    [
        None,  # no result row at all
        {"top_strategies": []},  # notes row, no persisted strategies
        {"top_strategies": [{"strategy_id": 1}]},  # entries without a chromosome
    ],
    ids=["no_result_row", "empty_top_strategies", "entry_without_chromosome"],
)
async def test_warm_start_400_when_seed_run_has_no_champions(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    notes: dict[str, object] | None,
) -> None:
    """W1: a matching-hash seed run with no loadable chromosome is refused at launch.

    Before the fix this returned 201, created the run row, and the subprocess
    died 1 s later with a warm-start ValueError.
    """
    ac, state, job_mgr = client

    seed_id = _seed_discovery_run(state)
    if notes is not None:
        state.save_backtest_result(seed_id, {"notes": json.dumps(notes)})

    before = state.conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0]
    body = {**_BASE_BODY, "seed_run_id": seed_id}

    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 1
        r = await ac.post("/api/discovery/launch", json=body)

    assert r.status_code == 400, r.text
    assert r.json()["detail"] == (
        f"Seed run {seed_id} has no persisted champions to seed from "
        "(guardrails may have rejected all candidates); launch without "
        "seed_run_id instead"
    )
    assert mock_popen.call_count == 0
    after = state.conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0]
    assert after == before


async def test_seed_passes_through_to_command(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """vibe-quant-s8d9y: seed=123 → `--seed 123`; absent → no flag."""
    ac, _state, job_mgr = client

    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 12345
        r = await ac.post("/api/discovery/launch", json={**_BASE_BODY, "seed": 123})
    assert r.status_code == 201, r.text
    args = mock_popen.call_args.args[0]
    assert args[args.index("--seed") + 1] == "123"

    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 12346
        r = await ac.post("/api/discovery/launch", json=_BASE_BODY)
    assert r.status_code == 201, r.text
    assert "--seed" not in mock_popen.call_args.args[0]


@pytest.mark.parametrize("bad", [-1, 2**32])
async def test_seed_out_of_range_422(
    client: tuple[AsyncClient, StateManager, BacktestJobManager], bad: int
) -> None:
    ac, _state, job_mgr = client
    with patch.object(job_mgr, "is_process_alive", return_value=True):
        r = await ac.post("/api/discovery/launch", json={**_BASE_BODY, "seed": bad})
    assert r.status_code == 422


async def test_job_response_exposes_notes_seed(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """Completed run with notes seed → response.seed; run without notes → None."""
    import json

    ac, state, job_mgr = client
    ids = []
    for pid in (12345, 12346):
        with (
            patch.object(job_mgr, "is_process_alive", return_value=True),
            patch("subprocess.Popen") as mock_popen,
        ):
            mock_popen.return_value.pid = pid
            r = await ac.post("/api/discovery/launch", json=_BASE_BODY)
        assert r.status_code == 201, r.text
        ids.append(r.json()["run_id"])
    state.save_backtest_result(ids[0], {"notes": json.dumps({"seed": 7, "strategies": []})})

    with patch.object(job_mgr, "is_process_alive", return_value=True):
        r = await ac.get("/api/discovery/jobs")
    assert r.status_code == 200, r.text
    by_id = {j["run_id"]: j for j in r.json()}
    assert by_id[ids[0]]["seed"] == 7
    assert by_id[ids[1]]["seed"] is None


async def test_max_workers_reaches_params_and_command(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """Parallel launches can size their pools (4 runs x 4 workers oversubscribed 10 cores)."""
    ac, state, job_mgr = client

    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 12345
        r = await ac.post("/api/discovery/launch", json={**_BASE_BODY, "max_workers": 2})
        default = await ac.post("/api/discovery/launch", json=_BASE_BODY)

    assert r.status_code == 201, r.text
    assert default.status_code == 201, default.text
    run = state.get_backtest_run(r.json()["run_id"])
    assert run is not None
    assert run["parameters"]["max_workers"] == 2
    first, second = (c.args[0] for c in mock_popen.call_args_list)
    assert first[first.index("--max-workers") + 1] == "2"
    assert second[second.index("--max-workers") + 1] == "4"  # default unchanged


async def test_max_workers_zero_rejected(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """0 would mean 'all cores' in the CLI — the API only accepts an explicit count."""
    ac, _state, job_mgr = client
    with patch.object(job_mgr, "is_process_alive", return_value=True):
        r = await ac.post("/api/discovery/launch", json={**_BASE_BODY, "max_workers": 0})
    assert r.status_code == 422


async def test_discovery_launch_bnb_in_params_and_argv(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """D3-2: bnb_discount=True lands in run params AND argv; default adds neither."""
    ac, state, job_mgr = client

    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 12345
        opted_in = await ac.post("/api/discovery/launch", json={**_BASE_BODY, "bnb_discount": True})
        default = await ac.post("/api/discovery/launch", json=_BASE_BODY)

    assert opted_in.status_code == 201, opted_in.text
    assert default.status_code == 201, default.text

    run = state.get_backtest_run(opted_in.json()["run_id"])
    assert run is not None
    assert run["parameters"]["bnb_discount"] is True
    default_run = state.get_backtest_run(default.json()["run_id"])
    assert default_run is not None
    assert "bnb_discount" not in default_run["parameters"]

    opted_args, default_args = (c.args[0] for c in mock_popen.call_args_list)
    assert "--bnb-discount" in opted_args
    assert "--bnb-discount" not in default_args


@pytest.mark.parametrize("value", ["post_only", "POST_ONLY", "limit", "", None])
async def test_discovery_launch_refuses_non_market_entry_order(
    client: tuple[AsyncClient, StateManager, BacktestJobManager], value: object
) -> None:
    """Discovery is market-only: any provided non-"market" entry_order is a 422, no run row."""
    ac, state, job_mgr = client
    before = state.conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0]
    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 1
        r = await ac.post("/api/discovery/launch", json={**_BASE_BODY, "entry_order": value})
    assert r.status_code == 422, r.text
    assert mock_popen.call_count == 0
    assert state.conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0] == before


async def test_discovery_launch_accepts_explicit_market_entry_order(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    ac, _state, job_mgr = client
    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 1
        r = await ac.post("/api/discovery/launch", json={**_BASE_BODY, "entry_order": "market"})
    assert r.status_code == 201, r.text


@pytest.mark.parametrize(
    "extra",
    [{"bnb_discounts": True}, {"bnb": True}, {"fee_tier": "VIP0"}, {"post_only": True}],
)
async def test_discovery_launch_rejects_unknown_fields(
    client: tuple[AsyncClient, StateManager, BacktestJobManager], extra: dict[str, object]
) -> None:
    """A mistyped knob (bnb_discounts) must 422, not run silently with defaults."""
    ac, state, job_mgr = client
    before = state.conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0]
    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 1
        r = await ac.post("/api/discovery/launch", json={**_BASE_BODY, **extra})
    assert r.status_code == 422, r.text
    assert mock_popen.call_count == 0
    assert state.conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0] == before


@pytest.mark.parametrize("value", ["true", 1, "yes", None])
async def test_discovery_launch_bnb_discount_must_be_a_bool(
    client: tuple[AsyncClient, StateManager, BacktestJobManager], value: object
) -> None:
    ac, _state, job_mgr = client
    with patch.object(job_mgr, "is_process_alive", return_value=True):
        r = await ac.post("/api/discovery/launch", json={**_BASE_BODY, "bnb_discount": value})
    assert r.status_code == 422, r.text


async def test_discovery_launch_fast_evaluator_in_params_and_argv(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """evaluator="fast" lands in run params AND argv; the default "nt" adds neither."""
    ac, state, job_mgr = client

    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 12345
        fast = await ac.post("/api/discovery/launch", json={**_BASE_BODY, "evaluator": "fast"})
        explicit_nt = await ac.post("/api/discovery/launch", json={**_BASE_BODY, "evaluator": "nt"})
        default = await ac.post("/api/discovery/launch", json=_BASE_BODY)

    for r in (fast, explicit_nt, default):
        assert r.status_code == 201, r.text
    fast_run = state.get_backtest_run(fast.json()["run_id"])
    assert fast_run is not None
    assert fast_run["parameters"]["evaluator"] == "fast"
    for r in (explicit_nt, default):
        run = state.get_backtest_run(r.json()["run_id"])
        assert run is not None
        assert "evaluator" not in run["parameters"]

    fast_args, nt_args, default_args = (c.args[0] for c in mock_popen.call_args_list)
    i = fast_args.index("--evaluator")
    assert fast_args[i + 1] == "fast"
    assert "--evaluator" not in nt_args
    assert "--evaluator" not in default_args


@pytest.mark.parametrize("value", ["FAST", "numpy", "", None])
async def test_discovery_launch_rejects_unknown_evaluator(
    client: tuple[AsyncClient, StateManager, BacktestJobManager], value: object
) -> None:
    ac, _state, job_mgr = client
    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 1
        r = await ac.post("/api/discovery/launch", json={**_BASE_BODY, "evaluator": value})
    assert r.status_code == 422, r.text
    assert mock_popen.call_count == 0
