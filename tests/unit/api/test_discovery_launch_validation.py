"""Launch-time discovery config validation.

A bad window/gate config must be rejected with 422 BEFORE any run row exists.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TCH003
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery.run_spec import DiscoveryRunSpec
from vibe_quant.jobs.manager import BacktestJobManager

_OK_BODY = {
    "symbols": ["BTCUSDT"],
    "timeframes": ["4h"],
    "start_date": "2024-01-01",
    "end_date": "2025-01-01",
}


@pytest.fixture()
async def client(tmp_path: Path):
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


async def _post(ac: AsyncClient, job_mgr: BacktestJobManager, body: dict[str, object]):
    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 12345
        return await ac.post("/api/discovery/launch", json=body)


async def test_short_wfa_rejected_422_no_run_row(client) -> None:
    ac, state, job_mgr = client
    before = _run_count(state)
    r = await _post(
        ac, job_mgr,
        {**_OK_BODY, "start_date": "2025-01-01", "end_date": "2025-03-02",
         "train_test_split": 0.8, "wfa_oos_step_days": 90},
    )
    assert r.status_code == 422
    assert "WFA" in r.json()["detail"]
    assert _run_count(state) == before


async def test_eval_windows_omitted_uses_cli_default(client) -> None:
    """eval_windows=0 is not sent -> CLI default 3 applies (needs >= 21 train days)."""
    ac, state, job_mgr = client
    before = _run_count(state)
    r = await _post(
        ac, job_mgr,
        {**_OK_BODY, "start_date": "2025-01-01", "end_date": "2025-01-16",
         "train_test_split": 0, "eval_windows": 0},
    )
    assert r.status_code == 422
    assert "eval_windows" in r.json()["detail"]
    assert _run_count(state) == before


@pytest.mark.parametrize(
    "override",
    [
        {"population": 1},
        {"population": 5, "elite_count": 5},
        {"wfa_min_consistency": 1.5},
        {"cross_window_months": [0]},
        {"cross_window_months": [-15]},
        {"cross_window_months": [-15, 3]},
    ],
)
async def test_invalid_knobs_rejected_422(client, override: dict[str, object]) -> None:
    ac, state, job_mgr = client
    before = _run_count(state)
    r = await _post(ac, job_mgr, {**_OK_BODY, **override})
    assert r.status_code == 422
    assert _run_count(state) == before


async def test_valid_defaults_accepted(client) -> None:
    ac, state, job_mgr = client
    r = await _post(ac, job_mgr, _OK_BODY)
    assert r.status_code == 201
    assert _run_count(state) == 1


async def test_valid_cross_window_and_wfa_accepted(client) -> None:
    ac, _state, job_mgr = client
    r = await _post(
        ac, job_mgr,
        {**_OK_BODY, "cross_window_months": [1, 2], "wfa_oos_step_days": 60},
    )
    assert r.status_code == 201


async def test_malformed_own_command_is_500_with_stderr(client, monkeypatch) -> None:
    ac, state, job_mgr = client
    real = DiscoveryRunSpec.to_argv
    monkeypatch.setattr(
        DiscoveryRunSpec,
        "to_argv",
        lambda spec: [*real(spec), "--no-such-flag"],
    )
    before = _run_count(state)
    r = await _post(ac, job_mgr, _OK_BODY)
    assert r.status_code == 500
    assert "--no-such-flag" in r.json()["detail"]
    assert _run_count(state) == before


async def test_real_command_carries_created_run_id(client) -> None:
    """The precheck's placeholder --run-id 0 must never reach the subprocess."""
    ac, _state, job_mgr = client
    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 12345
        r = await ac.post("/api/discovery/launch", json=_OK_BODY)
    assert r.status_code == 201
    run_id = r.json()["run_id"]
    argv = mock_popen.call_args.args[0]
    assert argv[argv.index("--run-id") + 1] == str(run_id)
    assert argv.count("--run-id") == 1
    assert run_id != 0


# --- --fitness v2 refusals (vibe-quant-clmmc.3 T4) ------------------------------------------


@pytest.mark.parametrize(
    ("extra", "detail"),
    [
        (
            {"fitness": "v2", "evaluator": "fast"},
            "--fitness v2 needs --evaluator nt (the fast-tier confirm compares every fitness "
            "field)",
        ),
        (
            {"fitness": "v2", "null_gate": True},
            "--fitness v2 cannot run with --null-gate (the timing-null re-run compares every "
            "fitness field)",
        ),
        (
            {"fitness": "v2", "evaluator": "fast", "null_gate": True},
            # The router's precheck order decides who refuses first: the null-gate precheck
            # runs before the fitness one and owns this combination (with its own text).
            "--null-gate needs real NT backtests (no --mock, no --evaluator fast)",
        ),
    ],
    ids=["fast", "null-gate", "null-gate-precheck-first"],
)  # fmt: skip
async def test_fitness_v2_launch_refusal_is_422_with_no_row(
    client, extra: dict[str, object], detail: str
) -> None:
    """R26: the CLI's own refusal texts, before any run row or job exists.

    Every ``--fitness v2`` launch whose extra flags a fitness-field comparer reads is refused:
    the fast-tier NT confirm and the timing-null re-run both compare every field of the score.
    Both problems in one body are refused by the earlier null-gate precheck, with its own text
    (the router's order, pinned here); the fixed "; " join order of the fitness texts is pinned
    in ``test_discovery_run_spec.py::test_fitness_launch_problems``.
    """
    ac, state, job_mgr = client
    before = _run_count(state)
    r = await _post(ac, job_mgr, {**_OK_BODY, **extra})
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == detail
    assert _run_count(state) == before


async def test_exit_genes_refusal_precedes_fitness_like_the_cli(client) -> None:
    """rev-c3t4-1 S1: one launch, one refusal text on API and CLI (run.py checks exit genes first)."""
    ac, state, job_mgr = client
    body = {**_OK_BODY, "fitness": "v2", "evaluator": "fast", "exit_gene_space": "extended"}
    r = await _post(ac, job_mgr, body)
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail.startswith("--exit-genes extended lets the stop pick the position size")
    assert "--fitness" not in detail
    assert _run_count(state) == 0


async def test_fitness_v2_mock_is_not_an_api_knob(client) -> None:
    """The API never mocks, so the mock refusal is the CLI's alone; a clean v2 body is a 201."""
    ac, state, job_mgr = client
    r = await _post(ac, job_mgr, {**_OK_BODY, "fitness": "v2"})
    assert r.status_code == 201, r.text
    run = state.get_backtest_run(r.json()["run_id"])
    assert run is not None
    assert run["parameters"]["fitness"] == "v2"


@pytest.mark.parametrize("value", ["v3", 1, "V2", None], ids=["v3", "int", "upper", "null"])
async def test_fitness_outside_the_literal_is_422(client, value: object) -> None:
    """R25: ``Literal["v1","v2"]`` rejects a value the pipeline could not run."""
    ac, state, job_mgr = client
    before = _run_count(state)
    r = await _post(ac, job_mgr, {**_OK_BODY, "fitness": value})
    assert r.status_code == 422, r.text
    assert "fitness" in r.text
    assert _run_count(state) == before


STOP_CAP_TEXT = (
    "--leverage 107 with liquidation off liquidates a BTCUSDT short 0.53% from entry; a 95% "
    "stop cap (0.50%) leaves no room above the smallest stop discovery draws (0.5%). Turn "
    "liquidation on (--liquidation-mode isolated or cross) or lower --leverage."
)


async def test_stop_cap_refusal_is_422_no_run_row(client) -> None:
    """R24: a liquidation-off leverage that leaves no room for a stop is refused like the CLI."""
    ac, state, job_mgr = client
    r = await _post(ac, job_mgr, {**_OK_BODY, "leverage": 107})
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == STOP_CAP_TEXT
    assert _run_count(state) == 0

    edge = await _post(ac, job_mgr, {**_OK_BODY, "leverage": 106})  # cap 0.51 still draws
    assert edge.status_code == 201, edge.text
    on = await _post(ac, job_mgr, {**_OK_BODY, "leverage": 107, "liquidation_mode": "isolated"})
    assert on.status_code == 201, on.text  # liquidation on books the liquidation itself
