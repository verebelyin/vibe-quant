"""Discovery launch-spec characterization goldens (bead vibe-quant-rabmo.5).

One discovery option is written in seven places today: the API schema, the launch
router's params ladder, the router's argv ladder, argparse, ``DiscoveryConfig``,
its fields and the provenance notes. The refactor moves them into
``DiscoveryRunSpec``; these goldens pin the OBSERVABLE spec so the move is proven
behaviour-preserving:

- ``run_spec_golden.json`` holds 16 API rows (router argv + stored parameters),
  10 CLI-only rows (``DiscoveryConfig`` + persisted notes for flags the API does
  not model) and the campaign subprocess argv. One API row has a negative
  cross-window month: the launch precheck refuses it (422, no run row), so that
  row pins the argv's ``--cross-window-months=-1,2`` "=" token form (a leading
  minus would otherwise parse as a flag) plus the exact refusal; its sibling
  ``cross_months_wfa`` carries the valid ``[1, 2]`` shape through POST.
- ``discovery_cli_help.txt`` is ``build_parser().format_help()`` at COLUMNS=80.

Characterization only: no production code is pinned at a new location here. The
fixtures were captured ONCE on base a557a612 via ``dump_golden()`` below.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from contextlib import ExitStack, asynccontextmanager
from dataclasses import asdict
from pathlib import Path  # noqa: TC003 - used at runtime for fixture paths
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient

from vibe_quant.api import data_precheck, mark_precheck
from vibe_quant.api.app import create_app
from vibe_quant.api.routers import discovery as discovery_router
from vibe_quant.api.schemas.discovery import DiscoveryLaunchRequest
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery import __main__ as discovery_main
from vibe_quant.discovery.campaign import RegimeCrossConfig, TrainWindow, run_discovery_subprocess
from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline, DiscoveryResult
from vibe_quant.discovery.run_spec import build_parser
from vibe_quant.jobs.manager import BacktestJobManager, JobStatus

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "discovery"
GOLDEN_PATH = FIXTURES / "run_spec_golden.json"
HELP_PATH = FIXTURES / "discovery_cli_help.txt"

START = "2024-01-01"
END = "2025-01-01"
MASK = "<masked>"


def _body(**overrides: Any) -> dict[str, Any]:
    return {
        "symbols": ["BTCUSDT"],
        "timeframes": ["4h"],
        "start_date": START,
        "end_date": END,
        **overrides,
    }


# The 16 API rows: one DiscoveryLaunchRequest body each (dates explicit).
API_ROWS: dict[str, dict[str, Any]] = {
    "default": _body(),
    "eval_windows_0": _body(eval_windows=0),
    "eval_windows_1_split_0": _body(eval_windows=1, train_test_split=0),
    "cross_months_negative": _body(cross_window_months=[-1, 2], wfa_oos_step_days=30),
    "cross_months_wfa": _body(cross_window_months=[1, 2], wfa_oos_step_days=30),
    "liquidation_isolated_sizing": _body(
        liquidation_mode="isolated",
        leverage=10,
        max_position_pct=0.5,
        risk_per_trade=0.01,
        max_margin_pct=0.3,
    ),
    "grid_isolated_cross": _body(
        bankroll_grid={
            "modes": ["isolated", "cross"],
            "leverages": [5, 10],
            "max_margin_pcts": [0.5],
        }
    ),
    "off_leverage_5": _body(liquidation_mode="off", leverage=5),
    "misc": _body(
        symbols=["BTCUSDT", "ETHUSDT"],
        seed=7,
        num_seeds=3,
        evaluator="fast",
        bnb_discount=True,
        crowding_enabled=False,
        immigrant_fraction=0.15,
        bootstrap_min_sharpe=0.2,
        no_bootstrap_ci=True,
        indicator_pool=["RSI", "KAMA"],
        direction="long",
        symbol_agg="worst",
        seed_run_id=5,
    ),
    "immigrant_entropy": _body(immigrant_fraction=0.2, entropy_threshold=0.3),
    "timeframe_1m": _body(timeframes=["1m"]),
    "direction_both": _body(direction="both"),
    "liquidation_cross_lev20": _body(liquidation_mode="cross", leverage=20),
    "sizing_cap_2_lev5": _body(leverage=5, max_position_pct=2.0),
    "num_seeds_grid": _body(
        num_seeds=2,
        bankroll_grid={
            "modes": ["isolated"],
            "leverages": [5, 10],
            "max_margin_pcts": [0.5, 0.25],
        },
    ),
    "split_holdout_three_symbols": _body(
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        train_test_split=0.5,
    ),
}

# Rows the launch precheck refuses before any run row (negative cross months);
# they pin the argv token form and the exact 422 instead of stored parameters.
API_REFUSED_ROWS = frozenset({"cross_months_negative"})

# The 10 CLI-only rows: no API field exists (``extra="forbid"`` in the schema), so
# the spec is the parsed DiscoveryConfig plus the persisted notes.
CLI_ROWS: dict[str, dict[str, Any]] = {
    "no_dsr": {"argv": ["--no-dsr"], "run_symbols": ["BTCUSDT"]},
    "dsr_rule_emax": {"argv": ["--dsr-rule", "emax"], "run_symbols": ["BTCUSDT"]},
    "dsr_rule_q95": {"argv": ["--dsr-rule", "q95"], "run_symbols": ["BTCUSDT"]},
    "no_buy_hold_gate": {"argv": ["--no-buy-hold-gate"], "run_symbols": ["BTCUSDT"]},
    "fee_tier_vip0": {"argv": ["--fee-tier", "VIP0"], "run_symbols": ["BTCUSDT"]},
    "holdout_min_trades_7": {"argv": ["--holdout-min-trades", "7"], "run_symbols": ["BTCUSDT"]},
    "holdout_min_sharpe_0_2": {"argv": ["--holdout-min-sharpe", "0.2"], "run_symbols": ["BTCUSDT"]},
    "bootstrap_ci_level_0_9": {
        "argv": ["--bootstrap-ci-level", "0.9"],
        "run_symbols": ["BTCUSDT"],
    },
    "liquidation_leverage_5": {
        "argv": ["--liquidation-leverage", "5"],
        "run_symbols": ["BTCUSDT"],
    },
    # F1: ``--symbols ""`` means "use the run row's symbols"; the notes and the
    # DiscoveryConfig must both carry the RESOLVED list, never the empty string.
    "symbols_empty": {"argv": ["--symbols", ""], "run_symbols": ["ETHUSDT", "SOLUSDT"]},
}


# --- capture helpers -------------------------------------------------------


@asynccontextmanager
async def _api(db_path: Path) -> AsyncIterator[tuple[AsyncClient, StateManager]]:
    app = create_app()
    state = StateManager(db_path=db_path)
    _ = state.conn
    jobs = BacktestJobManager(db_path=db_path)
    ws = ConnectionManager()
    await ws.start()
    app.state.state_manager = state
    app.state.job_manager = jobs
    app.state.ws_manager = ws
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, state
    finally:
        await ws.stop()
        jobs.close()
        state.close()


async def _capture_api_row(db_path: Path, name: str) -> dict[str, Any]:
    """Router argv + the parameters a real POST /launch stores (or its refusal)."""
    body = API_ROWS[name]
    request = DiscoveryLaunchRequest(**body)
    argv = discovery_router._discovery_cli_args(  # type: ignore[attr-defined]
        request, START, END, request.liquidation_settings()
    )
    async with _api(db_path) as (client, state):
        with (
            patch("subprocess.Popen") as popen,
            patch.object(BacktestJobManager, "is_process_alive", return_value=True),
            # The misc row warm-starts from run 5; warm-start validation is not
            # what this golden pins (argv/params are), and no such run exists here.
            patch.object(discovery_router, "_validate_seed_run_compiler", lambda *a, **k: None),
            patch.object(data_precheck, "precheck_discovery_data", lambda *a, **k: None),
            patch.object(mark_precheck, "precheck_discovery", lambda *a, **k: None),
        ):
            popen.return_value.pid = 4321
            response = await client.post("/api/discovery/launch", json=body)
            if name in API_REFUSED_ROWS:
                runs = state.conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0]
                assert response.status_code == 422 and runs == 0, response.text
                return {
                    "argv": argv,
                    "response": {"status": 422, "detail": response.json()["detail"]},
                }
            assert response.status_code == 201, response.text
            run = state.get_backtest_run(response.json()["run_id"])
            assert run is not None
            params = run["parameters"]
    return {"argv": argv, "params": params}


async def _capture_all_api_rows(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in API_ROWS:
        out[name] = await _capture_api_row(root / f"api_{name}.db", name)
    return out


def _create_run_row(db_path: Path, symbols: list[str]) -> int:
    state = StateManager(db_path)
    try:
        return state.create_backtest_run(
            strategy_id=None,
            run_mode="discovery",
            symbols=symbols,
            timeframe="4h",
            start_date=START,
            end_date="2026-02-24",
            parameters={},
        )
    finally:
        state.close()


def _fake_run(captured: list[DiscoveryConfig]) -> Callable[[DiscoveryPipeline], DiscoveryResult]:
    def run(pipeline: DiscoveryPipeline) -> DiscoveryResult:
        captured.append(pipeline.config)
        return DiscoveryResult(
            generations=[],
            top_strategies=[],
            total_candidates_evaluated=0,
            converged=False,
            convergence_generation=None,
        )

    return run


def _masked_notes(notes: dict[str, Any], argv: list[str]) -> dict[str, Any]:
    notes["compiler_version"] = MASK
    if "--seed" not in argv:
        notes["seed"] = MASK
    return notes


def _capture_cli_row(db_path: Path, name: str) -> dict[str, Any]:
    spec = CLI_ROWS[name]
    run_id = _create_run_row(db_path, spec["run_symbols"])
    captured: list[DiscoveryConfig] = []
    argv = ["x", "--run-id", str(run_id), "--db", str(db_path), "--mock", *spec["argv"]]
    with ExitStack() as stack:
        stack.enter_context(patch.object(sys, "argv", argv))
        stack.enter_context(patch.object(DiscoveryPipeline, "run", _fake_run(captured)))
        assert discovery_main.main() == 0
    assert len(captured) == 1
    config = asdict(captured[0])
    config.pop("prior_trials_fn")

    state = StateManager(db_path)
    try:
        result = state.get_backtest_result(run_id)
    finally:
        state.close()
    assert result is not None
    notes = _masked_notes(json.loads(result["notes"]), spec["argv"])
    return {
        "argv": spec["argv"],
        "run_symbols": spec["run_symbols"],
        "config": config,
        "notes": notes,
    }


def _capture_all_cli_rows(root: Path) -> dict[str, Any]:
    return {name: _capture_cli_row(root / f"cli_{name}.db", name) for name in CLI_ROWS}


class _FakeJobManager:
    """Just enough of BacktestJobManager to capture the campaign command."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def start_job(
        self, run_id: int, job_type: str, command: list[str], log_file: str | None = None
    ) -> int:
        self.commands.append(list(command))
        return 4242

    def is_process_alive(self, pid: int) -> bool:
        return False

    def sync_job_status(self, run_id: int) -> str:
        return JobStatus.COMPLETED


def _campaign_config() -> tuple[RegimeCrossConfig, TrainWindow]:
    config = RegimeCrossConfig(
        name="golden_campaign",
        symbol="BTCUSDT",
        timeframe="4h",
        train_windows=[TrainWindow(label="w1", start=START, end=END, direction="short")],
        oos_windows=[],
        population_size=100,
        max_generations=20,
        extra_discovery_args=["--eval-windows", "2", "--cross-window-months=1", "--no-dsr"],
    )
    return config, config.train_windows[0]


def _capture_campaign_argv() -> list[str]:
    config, window = _campaign_config()
    manager = _FakeJobManager()
    assert (
        run_discovery_subprocess(
            4242,
            config,
            window,
            manager,
            python_bin="python3",  # type: ignore[arg-type]
        )
        == 0
    )
    assert len(manager.commands) == 1
    return manager.commands[0]


def _help_text() -> str:
    with patch.dict(os.environ, {"COLUMNS": "80"}):
        return build_parser().format_help()


def dump_golden() -> None:
    """Write both fixtures from the code under test (run ONCE on the base commit)."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        payload = {
            "api": asyncio.run(_capture_all_api_rows(root)),
            "cli": _capture_all_cli_rows(root),
            "campaign": {"argv": _capture_campaign_argv()},
        }
    GOLDEN_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    HELP_PATH.write_text(_help_text())


# --- tests -----------------------------------------------------------------


def _golden() -> dict[str, Any]:
    return json.loads(GOLDEN_PATH.read_text())


def test_golden_matrix_size() -> None:
    """16 API rows + 10 CLI rows, both sides."""
    golden = _golden()
    assert len(API_ROWS) == 16
    assert len(CLI_ROWS) == 10
    assert len(golden["api"]) == 16
    assert len(golden["cli"]) == 10


def test_router_argv_matches_golden() -> None:
    """R1: the launch router's argv is byte-identical to the golden, every row."""
    golden = _golden()
    for name, body in API_ROWS.items():
        request = DiscoveryLaunchRequest(**body)
        argv = discovery_router._discovery_cli_args(  # type: ignore[attr-defined]
            request, START, END, request.liquidation_settings()
        )
        assert argv == golden["api"][name]["argv"], name


def test_stored_params_match_golden(tmp_path: Path) -> None:
    """R2: the stored parameters dict (and its key order) match the golden."""
    golden = _golden()
    for name in API_ROWS:
        captured = asyncio.run(_capture_api_row(tmp_path / f"{name}.db", name))
        expected = golden["api"][name]
        assert captured["argv"] == expected["argv"], name
        if name in API_REFUSED_ROWS:
            assert captured["response"] == expected["response"], name
            continue
        assert captured["params"] == expected["params"], name
        assert json.dumps(captured["params"]) == json.dumps(expected["params"]), name


def test_cli_help_matches_golden(monkeypatch: Any) -> None:
    """R14: ``--help`` is byte-identical at a fixed terminal width."""
    monkeypatch.setenv("COLUMNS", "80")
    assert build_parser().format_help() == HELP_PATH.read_text()


def test_to_config_matches_golden(tmp_path: Path) -> None:
    """R16: the CLI's DiscoveryConfig is identical to the golden, every CLI row."""
    golden = _golden()
    for name in CLI_ROWS:
        captured = _capture_cli_row(tmp_path / f"cli_{name}.db", name)
        assert captured["config"] == golden["cli"][name]["config"], name


def test_cli_notes_match_golden(tmp_path: Path) -> None:
    """The provenance notes the CLI persists are identical to the golden."""
    golden = _golden()
    for name in CLI_ROWS:
        captured = _capture_cli_row(tmp_path / f"notes_{name}.db", name)
        assert captured["notes"] == golden["cli"][name]["notes"], name


def test_empty_cli_symbols_use_run_row(tmp_path: Path) -> None:
    """R17b/F1: ``--symbols ""`` resolves to the run row's symbols everywhere."""
    captured = _capture_cli_row(tmp_path / "empty_symbols.db", "symbols_empty")
    symbols = ["ETHUSDT", "SOLUSDT"]
    assert captured["argv"] == ["--symbols", ""]
    assert captured["run_symbols"] == symbols
    assert captured["config"]["symbols"] == symbols
    assert captured["notes"]["buy_hold_gate"]["symbols"] == symbols
    assert captured == _golden()["cli"]["symbols_empty"]


def test_campaign_argv_golden() -> None:
    """R18: the campaign subprocess argv is byte-identical (campaign stays hand-built)."""
    assert _capture_campaign_argv() == _golden()["campaign"]["argv"]
