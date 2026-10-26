"""Discovery honours the run's ``SizingSettings`` end to end (bead vibe-quant-2gyk1.2, task S4).

Binding rows: S15 (a corrupt ``--max-position-pct`` fails the CLI before ``plan_from_args``),
S20 (every ``NTBacktestFn`` factory runner carries the run's :class:`SizingSettings`), S23
(promote/replay inherit the parent's ``notes.sizing``), S23b (a corrupt ``notes.sizing`` is a
409 before any strategy/backtest row), S23c (a legacy parent without ``notes.sizing`` adds no
sizing keys) and S23d (an inherited cap above the inherited leverage is a 409).

The rule itself (S1-S13) is pinned by ``tests/unit/test_sizing_settings.py``; the API surface
(S27-S27d) by ``tests/unit/api/test_discovery_sizing_api.py``.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator  # noqa: TCH003
from pathlib import Path  # noqa: TCH003
from typing import Any  # noqa: TCH003
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from tests.unit.test_run_discovery import drive_run_discovery
from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery.__main__ import main
from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.discovery.fitness import FitnessResult
from vibe_quant.discovery.pipeline import DiscoveryResult, HoldoutResult
from vibe_quant.jobs.manager import BacktestJobManager
from vibe_quant.sizing_settings import SizingSettings

# ---------------------------------------------------------------------------
# S20 -- every factory runner carries the run's sizing
# ---------------------------------------------------------------------------


def _record_runners(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture every ``NTScreeningRunner`` kwarg dict ``_make_runner`` builds."""
    captured: list[dict[str, Any]] = []

    class _Recorder:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    import vibe_quant.screening.nt_runner as ntr

    monkeypatch.setattr(ntr, "NTScreeningRunner", _Recorder)
    return captured


def test_every_factory_runner_gets_sizing(monkeypatch: pytest.MonkeyPatch) -> None:
    """S20: the one construction site passes the run's SizingSettings to every runner."""
    sizing = SizingSettings(max_position_pct=1.0, risk_per_trade=0.01)
    captured = _record_runners(monkeypatch)
    fn = NTBacktestFn(["BTCUSDT"], "4h", "2024-01-01", "2024-06-01", sizing=sizing)

    fn._make_runner({"name": "x"}, "2024-01-01", "2024-06-01")
    fn._make_runner({"name": "x"}, "2024-06-01", "2024-12-01", ["ETHUSDT"])

    assert [c["launch"].sizing for c in captured] == [sizing, sizing]
    assert captured[0]["launch"].sizing is sizing


def test_make_nt_backtest_fn_forwards_sizing() -> None:
    """The CLI factory threads the run's sizing into the NTBacktestFn it builds."""
    from vibe_quant.discovery.run import _make_nt_backtest_fn
    from vibe_quant.fees import FeeSchedule
    from vibe_quant.liquidation_settings import LiquidationSettings

    sizing = SizingSettings(max_position_pct=1.0)
    fn = _make_nt_backtest_fn(
        ["BTCUSDT"],
        "4h",
        "2024-01-01",
        "2024-06-01",
        fee_schedule=FeeSchedule(),
        entry_order="market",
        liquidation=LiquidationSettings(),
        sizing=sizing,
    )
    assert fn.sizing is sizing
    assert fn._make_runner({"name": "x"}, "2024-01-01", "2024-06-01") is not None


def test_nt_backtest_fn_defaults_to_default_sizing() -> None:
    """An omitted sizing is the run defaults, never a hidden non-default."""
    fn = NTBacktestFn(["BTCUSDT"], "4h", "2024-01-01", "2024-06-01")
    assert fn.sizing == SizingSettings()


# ---------------------------------------------------------------------------
# S20/S23e -- the CLI threads sizing to every fn it builds and records notes.sizing
# ---------------------------------------------------------------------------


def _reach_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    zero_champion: bool = False,
    extra_args: list[str] | None = None,
) -> tuple[int, dict[str, Any], dict[str, Any], list[NTBacktestFn]]:
    """Drive discovery with cap 1.0 / risk 0.01 and the pipeline's ``run`` replaced.

    ``extra_args`` are appended to argv verbatim.

    Returns ``(rc, run row, notes, every NTBacktestFn the CLI built)``.
    """
    import vibe_quant.discovery.run as discovery_run
    from vibe_quant.discovery.operators import (
        ConditionType,
        Direction,
        StrategyChromosome,
        StrategyGene,
    )
    from vibe_quant.discovery.pipeline import DiscoveryPipeline

    db = tmp_path / "reach.db"
    run_id = _create_discovery_run(db)
    chrom = StrategyChromosome(
        entry_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 30.0)],
        exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0)],
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        direction=Direction.SHORT,
    )
    fit = FitnessResult(
        sharpe_ratio=1.5, max_drawdown=0.05, profit_factor=1.8, total_trades=100,
        total_return=0.10, complexity_penalty=0.0, overtrade_penalty=0.0,
        sl_tp_penalty=0.0, raw_score=1.0, adjusted_score=1.0, passed_filters=True,
        filter_results={},
    )  # fmt: skip
    holdout = HoldoutResult(
        sharpe_ratio=0.8, max_drawdown=0.03, profit_factor=1.2, total_trades=30,
        total_return=0.02,
    )  # fmt: skip

    def run_fn(self: Any) -> DiscoveryResult:
        assert self._backtest_fn_factory is not None
        self._backtest_fn_factory("2024-02-01", "2024-03-01")
        if zero_champion:
            return DiscoveryResult(
                generations=[], top_strategies=[], total_candidates_evaluated=10,
                converged=True, convergence_generation=1,
            )  # fmt: skip
        return DiscoveryResult(
            generations=[], top_strategies=[(chrom, fit)], total_candidates_evaluated=10,
            converged=True, convergence_generation=1, holdout_results=[holdout],
            train_dates=("2024-01-01", "2024-11-01"),
            holdout_dates=("2024-11-01", "2025-01-01"),
        )  # fmt: skip

    built: list[NTBacktestFn] = []
    real_init = NTBacktestFn.__init__

    def recording_init(self: Any, *a: Any, **k: Any) -> None:
        real_init(self, *a, **k)
        built.append(self)

    monkeypatch.setattr(discovery_run, "_check_data_available", lambda syms: True)
    monkeypatch.setattr(NTBacktestFn, "__init__", recording_init)
    monkeypatch.setattr(
        NTBacktestFn,
        "__call__",
        lambda self, c: {
            "sharpe_ratio": 1.0, "total_return": 0.05, "max_drawdown": 0.05,
            "profit_factor": 1.5, "total_trades": 100,
        },  # fmt: skip
    )
    monkeypatch.setattr(DiscoveryPipeline, "run", lambda self: run_fn(self))
    argv = [
        "--run-id", str(run_id), "--db", str(db),
        "--start-date", "2024-01-01", "--end-date", "2025-01-01",
        "--eval-windows", "2", "--cross-window-months", "1",
        "--train-test-split", "0.8",
        "--max-position-pct", "1.0", "--risk-per-trade", "0.01",
        *(extra_args or []),
    ]  # fmt: skip
    rc = drive_run_discovery(db, run_id, argv, discovery_run.NTBacktests()).exit_code

    state = StateManager(db)
    run = state.get_backtest_run(run_id)
    result = state.get_backtest_result(run_id)
    state.close()
    assert run is not None
    assert result is not None
    return rc, run, json.loads(result["notes"]), built


def test_sizing_reaches_every_fn_the_cli_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S20: train GA, holdout, cross-window/WFA factory, full-range headline and the
    train-return fn all carry the run's non-default SizingSettings."""
    rc, _, _, built = _reach_run(tmp_path, monkeypatch)
    assert rc == 0
    expected = SizingSettings(max_position_pct=1.0, risk_per_trade=0.01)
    spans = {(fn.start_date, fn.end_date) for fn in built}
    assert ("2024-02-01", "2024-03-01") in spans  # cross-window / WFA factory
    assert ("2024-01-01", "2025-01-01") in spans  # full-range headline
    assert ("2024-01-01", "2024-11-01") in spans  # train-return fn
    assert len(spans) >= 5  # + the train GA fn and the holdout fn
    assert [fn.sizing for fn in built] == [expected] * len(built)


def test_cli_records_notes_sizing_not_run_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S23e: the CLI writes notes.sizing and leaves backtest_runs.parameters alone."""
    rc, run, notes, _ = _reach_run(tmp_path, monkeypatch)
    assert rc == 0
    assert notes["sizing"] == {"max_position_pct": 1.0, "risk_per_trade": 0.01}
    assert run["parameters"] == {}


def test_zero_champion_run_records_notes_sizing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The no-viable-strategies summary carries notes.sizing too."""
    rc, _, notes, _ = _reach_run(tmp_path, monkeypatch, zero_champion=True)
    assert rc == 0
    assert notes["outcome"] == "no_viable_strategies"
    assert notes["sizing"] == {"max_position_pct": 1.0, "risk_per_trade": 0.01}


def test_cli_help_says_notes_sizing(capsys: pytest.CaptureFixture[str]) -> None:
    """S23e: the help texts name notes.sizing as the CLI's record.

    Three flags now share that sentence: ``--max-position-pct``, ``--risk-per-trade``
    and (T6) ``--max-margin-pct``. argparse wraps help text to the terminal width, so
    whitespace is normalised before matching: the assertion is about the wording, not
    about where the wrap happens.
    """
    from vibe_quant.discovery.run_spec import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
    out = re.sub(r"\s+", " ", capsys.readouterr().out)
    assert out.count("recorded in notes.sizing (the API launch also stores it in the run parameters)") == 3


# ---------------------------------------------------------------------------
# T6 (bead vibe-quant-t8nrv) -- the CLI's --max-margin-pct lands in notes.sizing
# ---------------------------------------------------------------------------


def test_cli_records_notes_margin_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T6/B5a: ``--max-margin-pct`` is recorded in ``notes.sizing`` like the other keys.

    The limit applies to every screening backtest of the run (the same setup the
    champion is found under), so the CLI records it where promote/replay look.
    """
    rc, _run, notes, built = _reach_run(
        tmp_path, monkeypatch, extra_args=["--max-margin-pct", "0.5"]
    )

    assert rc == 0
    assert notes["sizing"] == {
        "max_position_pct": 1.0,
        "risk_per_trade": 0.01,
        "max_margin_pct": 0.5,
    }
    assert built
    assert all(fn.sizing.max_margin_pct == 0.5 for fn in built)


def test_cli_bad_margin_limit_fails_before_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T6/S15: a corrupt ``--max-margin-pct`` fails the run before ``plan_from_args``.

    The window below is itself too short for ``plan_from_args``; the sizing error
    winning proves the parse runs first, as it does for the other sizing keys.
    """
    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--run-id",
            str(run_id),
            "--symbols",
            "BTCUSDT",
            "--timeframe",
            "1h",
            "--start-date",
            "2025-01-01",
            "--end-date",
            "2025-01-02",
            "--db",
            str(db_path),
            "--max-margin-pct=0",
        ],
    )

    assert main() == 1

    state = StateManager(db_path)
    run = state.get_backtest_run(run_id)
    state.close()
    assert run is not None
    assert run["status"] == "failed"
    assert run["error_message"] == (
        "SizingConfigError: max_margin_pct must be a number in (0, 1], got 0.0"
    )


# ---------------------------------------------------------------------------
# S15 -- a corrupt CLI sizing fails before plan_from_args
# ---------------------------------------------------------------------------


def _create_discovery_run(db_path: Path, *, parameters: dict[str, object] | None = None) -> int:
    state = StateManager(db_path)
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="discovery",
        symbols=["BTCUSDT"],
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2025-01-01",
        parameters=parameters or {},
    )
    state.close()
    return run_id


def test_cli_garbage_fails_before_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """S15: ``--max-position-pct -1`` fails the run BEFORE ``plan_from_args``.

    The window range below is itself too short for ``plan_from_args``; the sizing error
    winning proves the parse runs first, exactly as the design places it.
    """
    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--run-id",
            str(run_id),
            "--symbols",
            "BTCUSDT",
            "--timeframe",
            "1h",
            "--start-date",
            "2025-01-01",
            "--end-date",
            "2025-01-02",
            "--db",
            str(db_path),
            "--max-position-pct=-1",
        ],
    )

    assert main() == 1

    state = StateManager(db_path)
    run = state.get_backtest_run(run_id)
    state.close()
    assert run is not None
    assert run["status"] == "failed"
    assert run["error_message"] == (
        "SizingConfigError: max_position_pct must be a number in (0, 125], got -1.0"
    )


def test_cli_cap_above_leverage_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cap > leverage is refused by the CLI before ``plan_from_args`` too (N4)."""
    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--run-id",
            str(run_id),
            "--symbols",
            "BTCUSDT",
            "--timeframe",
            "1h",
            "--start-date",
            "2025-01-01",
            "--end-date",
            "2025-01-02",
            "--db",
            str(db_path),
            "--max-position-pct=20",
            "--leverage",
            "10",
        ],
    )

    assert main() == 1

    state = StateManager(db_path)
    run = state.get_backtest_run(run_id)
    state.close()
    assert run is not None
    assert run["status"] == "failed"
    assert run["error_message"] == (
        "SizingConfigError: max_position_pct (20.0) must not exceed leverage (10.0)"
    )


# ---------------------------------------------------------------------------
# S23-S23d -- promote/replay inherit notes.sizing
# ---------------------------------------------------------------------------


@pytest.fixture()
async def client(tmp_path: Path) -> AsyncIterator[tuple[AsyncClient, StateManager]]:
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
        with patch.object(BacktestJobManager, "start_job", lambda *a, **k: 4242):
            yield ac, state_mgr

    await ws_mgr.stop()
    job_mgr.close()
    state_mgr.close()


def _create_discovery_run_with_results(
    state: StateManager, *, notes_extra: dict[str, object] | None = None
) -> int:
    """A completed discovery run carrying one genome entry in its notes."""
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="discovery",
        symbols=["BTCUSDT"],
        timeframe="4h",
        start_date="2024-01-01",
        end_date="2025-01-01",
        parameters={"population": 20, "generations": 10},
    )
    state.update_backtest_run_status(run_id, "completed")

    genome_dsl = {
        "name": "ga_winner_sizing",
        "strategy_type": "momentum",
        "entry": {
            "conditions": [
                {"indicator": "RSI", "params": {"period": 14}, "operator": "<", "value": 30}
            ]
        },
        "exit": {
            "conditions": [
                {"indicator": "RSI", "params": {"period": 14}, "operator": ">", "value": 70}
            ]
        },
    }
    notes: dict[str, object] = {
        "top_strategies": [{"dsl": genome_dsl, "score": 1.5, "trades": 42, "sharpe": 1.8}]
    }
    if notes_extra:
        notes.update(notes_extra)

    state.conn.execute(
        "INSERT INTO backtest_results (run_id, notes) VALUES (?, ?)",
        (run_id, json.dumps(notes)),
    )
    state.conn.commit()
    return run_id


def _child_params(state: StateManager, run_id: int) -> dict[str, object]:
    run = state.get_backtest_run(run_id)
    assert run is not None
    params = run["parameters"]
    assert isinstance(params, dict)
    return params


def _run_count(state: StateManager, table: str) -> int:
    return int(state.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


async def test_promote_and_replay_inherit_sizing(
    client: tuple[AsyncClient, StateManager],
) -> None:
    """S23: a parent recorded at cap 1.0 hands the cap to both children."""
    ac, state = client
    run_id = _create_discovery_run_with_results(
        state, notes_extra={"sizing": {"max_position_pct": 1.0}}
    )

    r = await ac.post(f"/api/discovery/results/{run_id}/promote/0")
    assert r.status_code == 201, r.text
    promoted = _child_params(state, r.json()["run_id"])
    assert promoted["max_position_pct"] == 1.0
    assert type(promoted["max_position_pct"]) is float
    assert "risk_per_trade" not in promoted

    rr = await ac.post(f"/api/discovery/results/{run_id}/replay/0")
    assert rr.status_code == 201, rr.text
    replayed = _child_params(state, rr.json()["replay_run_id"])
    assert replayed["max_position_pct"] == 1.0
    assert type(replayed["max_position_pct"]) is float
    assert "risk_per_trade" not in replayed


async def test_promote_and_replay_inherit_both_sizing_keys(
    client: tuple[AsyncClient, StateManager],
) -> None:
    """S23: both keys ride along, each stored as a float."""
    ac, state = client
    run_id = _create_discovery_run_with_results(
        state, notes_extra={"sizing": {"max_position_pct": 1.0, "risk_per_trade": 0.01}}
    )

    r = await ac.post(f"/api/discovery/results/{run_id}/promote/0")
    assert r.status_code == 201, r.text
    promoted = _child_params(state, r.json()["run_id"])
    assert promoted["max_position_pct"] == 1.0
    assert promoted["risk_per_trade"] == 0.01


async def test_legacy_parent_inherits_defaults(
    client: tuple[AsyncClient, StateManager],
) -> None:
    """S23c: a parent with no notes.sizing adds no sizing keys to its child."""
    ac, state = client
    run_id = _create_discovery_run_with_results(state)

    r = await ac.post(f"/api/discovery/results/{run_id}/promote/0")
    assert r.status_code == 201, r.text
    promoted = _child_params(state, r.json()["run_id"])
    assert "max_position_pct" not in promoted
    assert "risk_per_trade" not in promoted

    rr = await ac.post(f"/api/discovery/results/{run_id}/replay/0")
    assert rr.status_code == 201, rr.text
    replayed = _child_params(state, rr.json()["replay_run_id"])
    assert "max_position_pct" not in replayed
    assert "risk_per_trade" not in replayed


@pytest.mark.parametrize("bad", ["not-a-dict", None, 5, []])
async def test_corrupt_notes_sizing_409_no_rows(
    client: tuple[AsyncClient, StateManager], bad: object
) -> None:
    """S23b: a present-but-unreadable notes.sizing (incl. explicit null) is a 409 before
    any strategy/backtest row, on both endpoints."""
    ac, state = client
    run_id = _create_discovery_run_with_results(state, notes_extra={"sizing": bad})
    strategies_before = _run_count(state, "strategies")
    runs_before = _run_count(state, "backtest_runs")

    r = await ac.post(f"/api/discovery/results/{run_id}/promote/0")
    assert r.status_code == 409
    assert "unreadable notes.sizing" in r.json()["detail"]
    assert _run_count(state, "strategies") == strategies_before
    assert _run_count(state, "backtest_runs") == runs_before

    rr = await ac.post(f"/api/discovery/results/{run_id}/replay/0")
    assert rr.status_code == 409
    assert "unreadable notes.sizing" in rr.json()["detail"]
    assert _run_count(state, "strategies") == strategies_before
    assert _run_count(state, "backtest_runs") == runs_before


async def test_notes_sizing_invalid_value_409_no_rows(
    client: tuple[AsyncClient, StateManager],
) -> None:
    """S23b: a present-but-invalid value is also unreadable, not silently defaulted."""
    ac, state = client
    run_id = _create_discovery_run_with_results(
        state, notes_extra={"sizing": {"max_position_pct": -1}}
    )
    runs_before = _run_count(state, "backtest_runs")

    r = await ac.post(f"/api/discovery/results/{run_id}/promote/0")
    assert r.status_code == 409
    assert "unreadable notes.sizing" in r.json()["detail"]
    assert "max_position_pct must be a number in (0, 125], got -1" in r.json()["detail"]
    assert _run_count(state, "backtest_runs") == runs_before


async def test_notes_sizing_above_leverage_409(
    client: tuple[AsyncClient, StateManager],
) -> None:
    """S23d: an inherited cap above the inherited leverage is a 409, before any row."""
    ac, state = client
    run_id = _create_discovery_run_with_results(
        state, notes_extra={"sizing": {"max_position_pct": 20}}
    )
    runs_before = _run_count(state, "backtest_runs")

    r = await ac.post(f"/api/discovery/results/{run_id}/promote/0")
    assert r.status_code == 409
    assert r.json()["detail"] == (
        f"discovery run {run_id} has an unreadable notes.sizing: "
        "max_position_pct (20.0) must not exceed leverage (10.0)"
    )
    assert _run_count(state, "backtest_runs") == runs_before
