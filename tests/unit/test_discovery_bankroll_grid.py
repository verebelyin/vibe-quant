"""Discovery bankroll grid (bead vibe-quant-t8nrv, task T8).

A grid is margin mode x leverage x margin cushion. Every cell is its own discovery run (own
train / holdout / cross-window / WFA functions); the pick is made across ALL cells x seeds, so
the DSR trial count N is the sum over cells x seeds, each champion records its cell, and
promote / replay inherit it. Without a grid discovery is exactly what it was.

Binding rows: R50-R58 (design.md § Semantics contract, "Discovery grid").
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path  # noqa: TCH003
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from tests.unit.test_dsr_cumulative_n import _chrom, _fit, _mock_cfg, _series
from tests.unit.test_multi_seed_grouping import _distant_chrom
from vibe_quant.api import mark_precheck
from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.bankroll_grid import (
    MAX_CELLS,
    BankrollCell,
    BankrollGridError,
    build_cells,
)
from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery import __main__ as cli
from vibe_quant.discovery import pooling, record
from vibe_quant.discovery import run as discovery_run
from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.discovery.pipeline import DiscoveryPipeline, DiscoveryResult, HoldoutResult
from vibe_quant.jobs.manager import BacktestJobManager
from vibe_quant.liquidation_settings import LiquidationSettings
from vibe_quant.sizing_settings import SizingSettings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

GRID_ARGS = [
    "--bankroll-modes", "isolated,cross",
    "--bankroll-leverages", "1,5",
    "--bankroll-max-margin-pcts", "0.25,0.5",
]  # fmt: skip
X_CELL = ("isolated", 1.0, 0.25)
Y_CELL = ("cross", 5.0, 0.5)


def _cell(mode: str = "isolated", lev: float = 1.0, pct: float = 0.5) -> BankrollCell:
    return BankrollCell(mode, lev, pct)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# R54 -- the domain of a grid
# ---------------------------------------------------------------------------


def test_grid_domain() -> None:
    """R54: isolated/cross only, whole leverages from 1x, 1..12 cells; 1x is accepted."""
    cells = build_cells(["isolated"], [1], [0.5])
    assert cells == [_cell("isolated", 1.0, 0.5)]  # 1x isolated: the spot-like setup
    assert type(cells[0].leverage) is float

    with pytest.raises(BankrollGridError, match="'off'"):
        build_cells(["off"], [5], [0.5])
    with pytest.raises(BankrollGridError, match="mode"):
        build_cells(["Isolated"], [5], [0.5])
    for bad_lev in (1.5, 0, 126, True, "5"):
        with pytest.raises(BankrollGridError, match="leverage"):
            build_cells(["cross"], [bad_lev], [0.5])  # type: ignore[list-item]
    for bad_pct in (0, -0.1, 1.01, float("nan")):
        with pytest.raises(BankrollGridError, match="max_margin_pct"):
            build_cells(["cross"], [5], [bad_pct])
    assert build_cells(["cross"], [5], [1.0])  # equality edge: 1.0 is a valid cushion
    assert build_cells(["cross"], [125], [0.5])  # equality edge: 125x is valid
    assert build_cells(["cross"], [1], [0.5])  # equality edge: 1x is valid
    for field, args in (
        ("modes", ([], [5], [0.5])),
        ("leverages", (["cross"], [], [0.5])),
        ("max_margin_pcts", (["cross"], [5], [])),
    ):
        with pytest.raises(BankrollGridError, match=field):
            build_cells(*args)  # type: ignore[arg-type]

    # equality edge: exactly MAX_CELLS cells are fine, one more is not
    assert MAX_CELLS == 12
    twelve = build_cells(["isolated", "cross"], [1, 2, 3], [0.2, 0.4])
    assert len(twelve) == 12
    with pytest.raises(BankrollGridError, match="at most 12"):
        build_cells(["isolated", "cross"], [1, 2, 3, 4, 5, 6, 7], [0.5])  # 14 cells
    with pytest.raises(BankrollGridError, match="at most 12"):
        build_cells(["isolated", "cross"], [1, 2, 3, 4, 5, 6], [0.5, 0.6])  # 24 cells
    with pytest.raises(BankrollGridError, match="at most 12"):
        build_cells(["isolated"], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [0.5])  # 13


def test_grid_cells_are_ordered_and_deduplicated() -> None:
    """Cells = product(modes in (isolated, cross) order, sorted leverages, sorted pcts)."""
    cells = build_cells(["cross", "isolated", "cross"], [5, 1, 5.0], [0.5, 0.25, 0.5])
    assert [(c.mode, c.leverage, c.max_margin_pct) for c in cells] == [
        ("isolated", 1.0, 0.25), ("isolated", 1.0, 0.5),
        ("isolated", 5.0, 0.25), ("isolated", 5.0, 0.5),
        ("cross", 1.0, 0.25), ("cross", 1.0, 0.5),
        ("cross", 5.0, 0.25), ("cross", 5.0, 0.5),
    ]  # fmt: skip


def test_cell_note_round_trips_and_rejects_garbage() -> None:
    note = _cell("cross", 10.0, 0.5).to_note()
    assert note == {"liquidation_mode": "cross", "leverage": 10.0, "max_margin_pct": 0.5}
    assert BankrollCell.from_note(note) == _cell("cross", 10.0, 0.5)
    for bad in (
        None, "x", [], {}, {"leverage": 0},
        {"liquidation_mode": "off", "leverage": 5.0, "max_margin_pct": 0.5},
        {"liquidation_mode": "cross", "leverage": 5.0, "max_margin_pct": None},
        {"liquidation_mode": "cross", "leverage": 1.5, "max_margin_pct": 0.5},
        {"liquidation_mode": "cross", "leverage": 5.0, "max_margin_pct": 0},
    ):  # fmt: skip
        with pytest.raises(BankrollGridError):
            BankrollCell.from_note(bad)


# ---------------------------------------------------------------------------
# R52 / R53 -- launch-time rejection (rule, CLI, API)
# ---------------------------------------------------------------------------

R52_MESSAGE = "bankroll cell (isolated, 1.0, 0.5): max_position_pct (2.0) must not exceed leverage (1.0)"
R53_MESSAGE = "use either the bankroll grid or --liquidation-mode/--leverage/--max-margin-pct"


def _create_discovery_run(db_path: Path) -> int:
    state = StateManager(db_path)
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="discovery",
        symbols=["BTCUSDT"],
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2025-01-01",
        parameters={},
    )
    state.close()
    return run_id


def _run_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: list[str]
) -> tuple[int, dict[str, Any]]:
    db = tmp_path / "cli.db"
    run_id = _create_discovery_run(db)
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--run-id", str(run_id), "--db", str(db),
         "--start-date", "2024-01-01", "--end-date", "2025-01-01", *extra],
    )  # fmt: skip
    rc = cli.main()
    state = StateManager(db)
    run = state.get_backtest_run(run_id)
    state.close()
    assert run is not None
    return rc, run


def test_invalid_cell_rejects_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R52: a cap the cell's leverage cannot cover is refused before any search (CLI exit 2)."""
    with pytest.raises(BankrollGridError) as exc:
        build_cells(["isolated"], [1], [0.5], SizingSettings(max_position_pct=2.0))
    assert str(exc.value) == R52_MESSAGE

    def _no_pipeline(self: Any) -> DiscoveryResult:
        raise AssertionError("no run may start for an invalid grid")

    monkeypatch.setattr(DiscoveryPipeline, "run", _no_pipeline)
    rc, run = _run_cli(
        tmp_path, monkeypatch,
        ["--bankroll-modes", "isolated", "--bankroll-leverages", "1",
         "--bankroll-max-margin-pcts", "0.5", "--max-position-pct", "2.0"],
    )  # fmt: skip
    assert rc == 2
    assert run["status"] == "failed"
    assert R52_MESSAGE in run["error_message"]


@pytest.mark.parametrize(
    "run_level",
    [["--liquidation-mode", "isolated"], ["--leverage", "5"],
     ["--liquidation-leverage", "5"], ["--max-margin-pct", "0.5"]],
)  # fmt: skip
def test_grid_exclusive_with_run_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_level: list[str]
) -> None:
    """R53: a grid and a run-level setup never mix (CLI exit 2)."""
    rc, run = _run_cli(tmp_path, monkeypatch, [*GRID_ARGS, *run_level])
    assert rc == 2
    assert run["status"] == "failed"
    assert R53_MESSAGE in run["error_message"]


def test_cli_partial_grid_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The three grid flags come together: one or two alone are an operator error."""
    rc, run = _run_cli(tmp_path, monkeypatch, ["--bankroll-modes", "isolated"])
    assert rc == 2
    assert "all three" in run["error_message"]


@asynccontextmanager
async def _api(
    db_path: Path, *, fake_start: bool = True
) -> AsyncIterator[tuple[AsyncClient, StateManager]]:
    app = create_app()
    state_mgr = StateManager(db_path=db_path)
    _ = state_mgr.conn
    job_mgr = BacktestJobManager(db_path=db_path)
    ws_mgr = ConnectionManager()
    await ws_mgr.start()
    app.state.state_manager = state_mgr
    app.state.job_manager = job_mgr
    app.state.ws_manager = ws_mgr
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            if fake_start:
                with patch.object(BacktestJobManager, "start_job", lambda *a, **k: 4242):
                    yield ac, state_mgr
            else:
                yield ac, state_mgr
    finally:
        await ws_mgr.stop()
        job_mgr.close()
        state_mgr.close()


_LAUNCH: dict[str, Any] = {
    "symbols": ["BTCUSDT"], "timeframes": ["4h"],
    "start_date": "2024-01-01", "end_date": "2025-01-01",
}  # fmt: skip
_GRID_BODY: dict[str, Any] = {
    "modes": ["cross", "isolated"], "leverages": [5, 1], "max_margin_pcts": [0.5, 0.25],
}


def _run_count(state: StateManager, table: str = "backtest_runs") -> int:
    return int(state.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


async def _launch(ac: AsyncClient, body: dict[str, Any]) -> tuple[Any, list[str] | None]:
    """POST a launch, capturing the exact argv the subprocess would get."""
    with (
        patch.object(BacktestJobManager, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value.pid = 12345
        r = await ac.post("/api/discovery/launch", json=body)
        argv = mock_popen.call_args.args[0] if mock_popen.call_args else None
    return r, argv


async def test_api_launch_rejects_invalid_cell_and_mixing(tmp_path: Path) -> None:
    """R52/R53 at the API: 422 before any run row exists."""
    async with _api(tmp_path / "api.db") as (ac, state):
        r = await ac.post(
            "/api/discovery/launch",
            json={**_LAUNCH, "max_position_pct": 2.0,
                  "bankroll_grid": {**_GRID_BODY, "modes": ["isolated"], "leverages": [1],
                                    "max_margin_pcts": [0.5]}},
        )  # fmt: skip
        assert r.status_code == 422, r.text
        assert R52_MESSAGE in r.text

        for run_level in ({"leverage": 5}, {"liquidation_mode": "cross"},
                          {"liquidation_leverage": 5}, {"max_margin_pct": 0.5}):  # fmt: skip
            r = await ac.post(
                "/api/discovery/launch",
                json={**_LAUNCH, **run_level, "bankroll_grid": _GRID_BODY},
            )
            assert r.status_code == 422, (run_level, r.text)
            assert R53_MESSAGE in r.text
        assert _run_count(state) == 0


async def test_api_grid_domain(tmp_path: Path) -> None:
    """R54 at the API: mode off, a fractional leverage, 13 cells, an empty list are 422s."""
    async with _api(tmp_path / "api.db") as (ac, state):
        for grid in (
            {**_GRID_BODY, "modes": ["off"]},
            {**_GRID_BODY, "leverages": [1.5]},
            {**_GRID_BODY, "leverages": list(range(1, 14)), "modes": ["isolated"],
             "max_margin_pcts": [0.5]},
            {**_GRID_BODY, "max_margin_pcts": []},
            {**_GRID_BODY, "leverages": [True]},
            {**_GRID_BODY, "extra": 1},
        ):  # fmt: skip
            r = await ac.post("/api/discovery/launch", json={**_LAUNCH, "bankroll_grid": grid})
            assert r.status_code == 422, (grid, r.text)
        assert _run_count(state) == 0


async def test_api_grid_launch_forwards_and_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid grid reaches the CLI as the three flags, is stored, and MARK data is checked."""
    seen: list[tuple[Any, ...]] = []
    monkeypatch.setattr(mark_precheck, "precheck_discovery", lambda *a: seen.append(a))
    async with _api(tmp_path / "api.db", fake_start=False) as (ac, state):
        r, argv = await _launch(ac, {**_LAUNCH, "bankroll_grid": _GRID_BODY})
        assert r.status_code == 201, r.text
        assert argv is not None
        assert argv[argv.index("--bankroll-modes") + 1] == "cross,isolated"
        assert argv[argv.index("--bankroll-leverages") + 1] == "5.0,1.0"
        assert argv[argv.index("--bankroll-max-margin-pcts") + 1] == "0.5,0.25"
        for flag in ("--liquidation-mode", "--leverage", "--max-margin-pct"):
            assert flag not in argv
        run = state.get_backtest_run(r.json()["run_id"])
        assert run is not None
        assert run["parameters"]["bankroll_grid"] == {
            "modes": ["cross", "isolated"], "leverages": [5.0, 1.0],
            "max_margin_pcts": [0.5, 0.25],
        }  # fmt: skip
        assert len(seen) == 1  # a grid is always liquidation-on: MARK coverage is checked


async def test_api_no_grid_launch_unchanged(tmp_path: Path) -> None:
    """R57 at the API: a launch without a grid adds no grid flag and no stored key."""
    async with _api(tmp_path / "api.db", fake_start=False) as (ac, state):
        r, argv = await _launch(ac, _LAUNCH)
        assert r.status_code == 201, r.text
        assert argv is not None
        assert not [a for a in argv if a.startswith("--bankroll")]
        run = state.get_backtest_run(r.json()["run_id"])
        assert run is not None
        assert "bankroll_grid" not in run["parameters"]


# ---------------------------------------------------------------------------
# R50 / R58 / R55 -- the grid runner (own pipeline per cell x seed, pooled N)
# ---------------------------------------------------------------------------


class _TaggedFn:
    """A stand-in backtest fn that remembers the cell it was built for."""

    def __init__(self, cell: BankrollCell, role: str) -> None:
        self.cell = cell
        self.role = role

    def __call__(self, chrom: Any) -> dict[str, float | int]:  # pragma: no cover
        return {}


def _fns_for(cell: BankrollCell) -> Any:
    return pooling._CellFns(
        backtest_fn=_TaggedFn(cell, "train"),
        holdout_backtest_fn=_TaggedFn(cell, "holdout"),
        backtest_fn_factory=lambda s, e: _TaggedFn(cell, "factory"),
    )


def _patch_run(
    monkeypatch: pytest.MonkeyPatch, evaluated: int = 30, record: list[Any] | None = None
) -> None:
    def fake_run(self: Any) -> DiscoveryResult:
        cell = self._backtest_fn.cell
        if record is not None:
            record.append(
                (cell, self._backtest_fn.role, self._holdout_backtest_fn.cell,
                 self._backtest_fn_factory("a", "b").cell)
            )  # fmt: skip
        idx = [(c.mode, c.leverage, c.max_margin_pct) for c in ALL_CELLS].index(
            (cell.mode, cell.leverage, cell.max_margin_pct)
        )
        # one champion per cell run; the holdout Sharpe encodes the cell it came from
        champ = (_chrom(stop_loss_pct=1.0 + idx), _fit(sharpe_ratio=2.0 + idx, mtm_daily_returns=_series(5.0, seed=idx + 1)))  # noqa: E501
        return DiscoveryResult(
            generations=[], top_strategies=[champ], total_candidates_evaluated=evaluated,
            converged=False, convergence_generation=None,
            holdout_results=[HoldoutResult(
                sharpe_ratio=float(idx), max_drawdown=0.1, profit_factor=1.5,
                total_trades=50, total_return=0.1,
            )],
        )  # fmt: skip

    monkeypatch.setattr(DiscoveryPipeline, "run", fake_run)


ALL_CELLS = build_cells(["isolated", "cross"], [1, 5], [0.25, 0.5])


def test_pooled_n_counts_every_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    """R50: 2x2x2 grid, 1 seed, 30 evaluations per cell -> N = 240 (+ prior)."""
    _patch_run(monkeypatch)
    cfg = _mock_cfg(prior_trials_fn=lambda s, e: 30, top_k=8)
    res, _cells = pooling._run_bankroll_grid(ALL_CELLS, 1, cfg, _fns_for, "/tmp/vq_bg_progress.json")
    assert res.total_candidates_evaluated == 240
    assert res.dsr_trials is not None
    assert res.dsr_trials["raw"] == 240
    # no retained series in the pooled cache -> the estimator fails closed to raw
    assert res.dsr_trials["run_n_eff"] == 240
    assert res.dsr_trials["neff_ev_source"] == "fallback_raw"
    assert res.dsr_trials["gate_n"] == 270
    assert res.dsr_trials["prior_n"] == 30
    notes = record.provenance_notes(
        res, use_mock=False, eval_windows_count=1, eval_windows=None,
        symbol_agg="portfolio", split_ratio=0.0, direction=None, cross_window_months=[],
        cross_window_min_sharpe=0.0, wfa_oos_step_days=0, wfa_min_consistency=0.0,
        num_seeds=1, seed=42, bootstrap_min_sharpe=0.0, holdout_min_sharpe=0.0,
        symbols=["BTCUSDT"],
    )  # fmt: skip
    assert notes["evaluated"] == 240


def test_grid_times_seeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """R58: 2 cells x 3 seeds = 6 pipelines, N = sum over all of them."""
    record: list[Any] = []
    _patch_run(monkeypatch, evaluated=30, record=record)
    cells = ALL_CELLS[:2]
    cfg = _mock_cfg(prior_trials_fn=lambda s, e: 30, top_k=8)
    res, _ = pooling._run_bankroll_grid(cells, 3, cfg, _fns_for, "/tmp/vq_bg_progress.json")
    assert len(record) == 6
    assert [r[0] for r in record] == [cells[0]] * 3 + [cells[1]] * 3
    assert res.total_candidates_evaluated == 180
    assert res.dsr_trials is not None
    assert res.dsr_trials["raw"] == 180 and res.dsr_trials["gate_n"] == 210


def test_gates_use_champion_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    """R55: train / holdout / cross-window functions of a run all come from ITS cell, and a
    champion's holdout result is the one its own cell's run produced."""
    record: list[Any] = []
    _patch_run(monkeypatch, record=record)
    cfg = _mock_cfg(prior_trials_fn=lambda s, e: 0, top_k=8, min_diversity_distance=0.0)
    res, cells = pooling._run_bankroll_grid(ALL_CELLS, 1, cfg, _fns_for, "/tmp/vq_bg_progress.json")
    assert len(record) == len(ALL_CELLS)
    for train_cell, role, holdout_cell, factory_cell in record:
        assert role == "train"
        assert holdout_cell == train_cell == factory_cell
    assert len(res.top_strategies) == len(cells) == len(res.holdout_results) > 1
    for cell, hr in zip(cells, res.holdout_results, strict=True):
        assert ALL_CELLS.index(cell) == int(hr.sharpe_ratio)  # holdout of the champion's cell


def test_zero_champion_cell_does_not_shift_cells(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cell whose run finds nothing still counts its evaluations, and the returned cells stay
    aligned with the surviving champions."""
    def fake_run(self: Any) -> DiscoveryResult:
        cell = self._backtest_fn.cell
        tops = [] if cell.leverage == 1.0 else [
            (_chrom(stop_loss_pct=cell.leverage), _fit(mtm_daily_returns=_series(5.0)))
        ]
        return DiscoveryResult(
            generations=[], top_strategies=tops, total_candidates_evaluated=30,
            converged=False, convergence_generation=None,
        )  # fmt: skip

    monkeypatch.setattr(DiscoveryPipeline, "run", fake_run)
    cfg = _mock_cfg(prior_trials_fn=lambda s, e: 0, top_k=8)
    res, cells = pooling._run_bankroll_grid(ALL_CELLS, 1, cfg, _fns_for, "/tmp/vq_bg_progress.json")
    assert res.total_candidates_evaluated == 240
    assert len(cells) == len(res.top_strategies) >= 1
    assert {c.leverage for c in cells} == {5.0}


# ---------------------------------------------------------------------------
# main() -- champion cell (R51), per-cell full-range headline (R55b), no grid (R57)
# ---------------------------------------------------------------------------


def _stub_metrics(liq: LiquidationSettings, siz: SizingSettings) -> dict[str, float | int]:
    """Metrics that depend on the cell, so a wrong cell shows up as a wrong number."""
    pct = siz.max_margin_pct or 0.0
    mode_term = 100.0 if liq.mode == "cross" else 0.0
    return {
        "sharpe_ratio": liq.leverage + pct + mode_term,
        "total_return": liq.leverage / 100.0 + pct,
        "max_drawdown": pct / 10.0,
        "profit_factor": 1.0 + pct + liq.leverage,
        "total_trades": int(liq.leverage * 10 + pct * 100),
    }


def _drive_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: list[str],
    champions: dict[tuple[str, float, float], tuple[Any, Any]] | None,
) -> tuple[int, Path, int, dict[str, Any]]:
    """Run ``main()`` with a faked pipeline: each run returns its cell's champion (if any)."""
    db = tmp_path / "main.db"
    run_id = _create_discovery_run(db)
    holdout = HoldoutResult(
        sharpe_ratio=0.8, max_drawdown=0.03, profit_factor=1.2, total_trades=30, total_return=0.02,
    )  # fmt: skip

    def fake_run(self: Any) -> DiscoveryResult:
        liq = self._backtest_fn.liquidation
        siz = self._backtest_fn.sizing
        key = (liq.mode, liq.leverage, siz.max_margin_pct)
        tops = [champions[key]] if champions and key in champions else []
        if not champions:  # no-grid run: its single champion
            tops = [(_chrom(), _fit(mtm_daily_returns=_series(5.0)))]
        return DiscoveryResult(
            generations=[], top_strategies=tops, total_candidates_evaluated=30,
            converged=True, convergence_generation=1,
            holdout_results=[holdout] * len(tops),
            train_dates=("2024-01-01", "2024-11-01"), holdout_dates=("2024-11-01", "2025-01-01"),
        )  # fmt: skip

    monkeypatch.setattr("vibe_quant.discovery.run._check_data_available", lambda syms: True)
    monkeypatch.setattr(
        NTBacktestFn, "__call__", lambda self, c: _stub_metrics(self.liquidation, self.sizing)
    )
    monkeypatch.setattr(DiscoveryPipeline, "run", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--run-id", str(run_id), "--db", str(db),
         "--start-date", "2024-01-01", "--end-date", "2025-01-01",
         "--eval-windows", "2", "--train-test-split", "0.8", "--cross-window-months", "1",
         "--no-bootstrap-ci", "--seed", "7", *extra],
    )  # fmt: skip
    rc = cli.main()
    state = StateManager(db)
    result = state.get_backtest_result(run_id)
    state.close()
    assert result is not None
    return rc, db, run_id, json.loads(result["notes"])


def _two_champions() -> dict[tuple[str, float, float], tuple[Any, Any]]:
    return {
        X_CELL: (_chrom(), _fit(sharpe_ratio=2.0, mtm_daily_returns=_series(5.0))),
        Y_CELL: (_distant_chrom(), _fit(sharpe_ratio=3.0, mtm_daily_returns=_series(5.0))),
    }


def test_champion_records_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R51: each champion carries its cell; the run carries no run-level liquidation setup."""
    rc, _db, _rid, notes = _drive_main(
        tmp_path, monkeypatch, [*GRID_ARGS, "--max-position-pct", "1.0"], _two_champions()
    )
    assert rc == 0
    by_cell = {
        (e["bankroll"]["liquidation_mode"], e["bankroll"]["leverage"], e["bankroll"]["max_margin_pct"])
        for e in notes["top_strategies"]
    }
    assert by_cell == {X_CELL, Y_CELL}
    assert {tuple(sorted(e["bankroll"])) for e in notes["top_strategies"]} == {
        ("leverage", "liquidation_mode", "max_margin_pct")
    }
    assert "liquidation" not in notes and "leverage" not in notes
    assert notes["sizing"] == {"max_position_pct": 1.0}  # no run-level max_margin_pct
    assert notes["bankroll_grid"]["n_cells"] == 8
    assert notes["bankroll_grid"]["cells"][0] == {
        "liquidation_mode": "isolated", "leverage": 1.0, "max_margin_pct": 0.25,
    }  # fmt: skip
    assert len(notes["bankroll_grid"]["cells"]) == 8
    assert notes["evaluated"] == 240  # 8 cells x 30


async def test_full_range_headline_per_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R55b: each champion's saved full-range headline is measured at ITS cell, and a replay
    child of each reproduces that headline exactly (it inherits the same cell)."""
    rc, db, run_id, notes = _drive_main(tmp_path, monkeypatch, GRID_ARGS, _two_champions())
    assert rc == 0
    entries = {
        (e["bankroll"]["liquidation_mode"], e["bankroll"]["leverage"], e["bankroll"]["max_margin_pct"]): e
        for e in notes["top_strategies"]
    }
    assert set(entries) == {X_CELL, Y_CELL}
    for key, entry in entries.items():
        expected = _stub_metrics(
            LiquidationSettings(mode=key[0], leverage=key[1]),  # type: ignore[arg-type]
            SizingSettings(max_margin_pct=key[2]),
        )
        assert entry["full_range_sharpe"] == expected["sharpe_ratio"]
        assert entry["full_range_trades"] == expected["total_trades"]
        assert entry["full_range_pf"] == expected["profit_factor"]
        assert entry["full_range_return_pct"] == expected["total_return"]
    assert (
        entries[X_CELL]["full_range_sharpe"] != entries[Y_CELL]["full_range_sharpe"]
    )  # the two cells really differ

    async with _api(db) as (ac, state):
        order = [e["bankroll"] for e in notes["top_strategies"]]
        for i, bankroll in enumerate(order):
            key = (bankroll["liquidation_mode"], bankroll["leverage"], bankroll["max_margin_pct"])
            r = await ac.post(f"/api/discovery/results/{run_id}/replay/{i}")
            assert r.status_code == 201, r.text
            child = state.get_backtest_run(r.json()["replay_run_id"])
            assert child is not None
            params = child["parameters"]
            replayed = _stub_metrics(
                LiquidationSettings.from_params(params), SizingSettings.from_params(params)
            )
            assert replayed["sharpe_ratio"] == entries[key]["full_range_sharpe"]
            assert replayed["total_trades"] == entries[key]["full_range_trades"]
            assert replayed["profit_factor"] == entries[key]["full_range_pf"]
            assert replayed["total_return"] == entries[key]["full_range_return_pct"]


def test_wfa_efficiency_uses_best_champions_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The in-sample train-return fn behind walk_forward_efficiency is the best champion's cell."""
    built: list[NTBacktestFn] = []
    real_init = NTBacktestFn.__init__

    def recording_init(self: Any, *a: Any, **k: Any) -> None:
        real_init(self, *a, **k)
        built.append(self)

    monkeypatch.setattr(NTBacktestFn, "__init__", recording_init)
    rc, _db, _rid, notes = _drive_main(tmp_path, monkeypatch, GRID_ARGS, _two_champions())
    assert rc == 0
    best = notes["top_strategies"][0]["bankroll"]
    train_fns = [fn for fn in built if (fn.start_date, fn.end_date) == ("2024-01-01", "2024-11-01")]
    assert train_fns, "the train-return fn was never built"
    last = train_fns[-1]
    assert (last.liquidation.mode, last.liquidation.leverage, last.sizing.max_margin_pct) == (
        best["liquidation_mode"], best["leverage"], best["max_margin_pct"],
    )  # fmt: skip


def test_no_grid_notes_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R57: no grid -> no grid key anywhere in the notes, run-level setup recorded as before."""
    rc, _db, _rid, notes = _drive_main(
        tmp_path, monkeypatch, ["--liquidation-mode", "isolated", "--leverage", "5",
                                "--max-margin-pct", "0.5"], None,
    )  # fmt: skip
    assert rc == 0
    assert "bankroll_grid" not in notes
    assert all("bankroll" not in e for e in notes["top_strategies"])
    assert notes["liquidation"] == {"mode": "isolated", "leverage": 5.0}
    assert notes["sizing"] == {"max_margin_pct": 0.5}
    assert notes["evaluated"] == 30


# ---------------------------------------------------------------------------
# R56 / R56b -- promote / replay inherit the champion's cell
# ---------------------------------------------------------------------------

_GENOME_DSL = {
    "name": "ga_winner_grid",
    "strategy_type": "momentum",
    "entry": {"conditions": [{"indicator": "RSI", "params": {"period": 14}, "operator": "<", "value": 30}]},
    "exit": {"conditions": [{"indicator": "RSI", "params": {"period": 14}, "operator": ">", "value": 70}]},
}  # fmt: skip


def _grid_run(state: StateManager, bankrolls: list[object], *, sizing: object | None = None) -> int:
    run_id = state.create_backtest_run(
        strategy_id=None, run_mode="discovery", symbols=["BTCUSDT"], timeframe="4h",
        start_date="2024-01-01", end_date="2025-01-01", parameters={},
    )  # fmt: skip
    state.update_backtest_run_status(run_id, "completed")
    entries = []
    for i, b in enumerate(bankrolls):
        dsl = {**_GENOME_DSL, "name": f"ga_winner_grid_{i}"}
        entry: dict[str, object] = {"dsl": dsl, "score": 1.0, "trades": 40, "sharpe": 1.5}
        if b != "ABSENT":
            entry["bankroll"] = b
        entries.append(entry)
    notes: dict[str, object] = {
        "top_strategies": entries,
        "bankroll_grid": {"cells": [], "n_cells": 8},
    }
    if sizing is not None:
        notes["sizing"] = sizing
    state.conn.execute(
        "INSERT INTO backtest_results (run_id, notes) VALUES (?, ?)", (run_id, json.dumps(notes))
    )
    state.conn.commit()
    return run_id


async def test_promote_inherits_cell(tmp_path: Path) -> None:
    """R56: promote and replay of a grid champion hand the CHAMPION's cell to the child."""
    async with _api(tmp_path / "p.db") as (ac, state):
        run_id = _grid_run(
            state,
            [_cell("cross", 10.0, 0.5).to_note(), _cell("isolated", 1.0, 0.25).to_note()],
            sizing={"max_position_pct": 1.0},
        )
        for idx, (mode, lev, pct) in enumerate([("cross", 10.0, 0.5), ("isolated", 1.0, 0.25)]):
            r = await ac.post(f"/api/discovery/results/{run_id}/promote/{idx}?mode=screening")
            assert r.status_code == 201, r.text
            promoted = state.get_backtest_run(r.json()["run_id"])
            assert promoted is not None
            for params in (promoted["parameters"],):
                assert params["liquidation_mode"] == mode
                assert params["leverage"] == lev
                assert params["max_margin_pct"] == pct
                assert params["max_position_pct"] == 1.0  # run-level cap rides along
            rr = await ac.post(f"/api/discovery/results/{run_id}/replay/{idx}")
            assert rr.status_code == 201, rr.text
            replayed = state.get_backtest_run(rr.json()["replay_run_id"])
            assert replayed is not None
            rp = replayed["parameters"]
            assert (rp["liquidation_mode"], rp["leverage"], rp["max_margin_pct"]) == (mode, lev, pct)


@pytest.mark.parametrize(
    "bad",
    ["ABSENT", None, "x", {"leverage": 0},
     {"liquidation_mode": "off", "leverage": 5.0, "max_margin_pct": 0.5},
     {"liquidation_mode": "cross", "leverage": 5.0, "max_margin_pct": None}],
)  # fmt: skip
async def test_child_fails_closed_on_bad_bankroll(tmp_path: Path, bad: object) -> None:
    """R56b: a grid champion without a valid bankroll is a 422 on both endpoints, no rows."""
    async with _api(tmp_path / "b.db") as (ac, state):
        run_id = _grid_run(state, [_cell().to_note(), bad])
        runs_before = _run_count(state)
        strategies_before = _run_count(state, "strategies")
        for url in (
            f"/api/discovery/results/{run_id}/promote/1",
            f"/api/discovery/results/{run_id}/replay/1",
        ):
            r = await ac.post(url)
            assert r.status_code == 422, r.text
            assert r.json()["detail"] == f"discovery run {run_id} strategy 1 has no valid bankroll setup"
        assert _run_count(state) == runs_before
        assert _run_count(state, "strategies") == strategies_before
        # the intact neighbour still promotes
        ok = await ac.post(f"/api/discovery/results/{run_id}/replay/0")
        assert ok.status_code == 201, ok.text


async def test_non_grid_run_ignores_a_stray_bankroll_key(tmp_path: Path) -> None:
    """R57: only a run with notes.bankroll_grid reads a champion's bankroll."""
    async with _api(tmp_path / "n.db") as (ac, state):
        run_id = _grid_run(state, ["x"])
        raw = state.get_backtest_result(run_id)
        assert raw is not None
        notes = json.loads(raw["notes"])
        del notes["bankroll_grid"]
        state.conn.execute(
            "UPDATE backtest_results SET notes = ? WHERE run_id = ?", (json.dumps(notes), run_id)
        )
        state.conn.commit()
        r = await ac.post(f"/api/discovery/results/{run_id}/replay/0")
        assert r.status_code == 201, r.text
        child = state.get_backtest_run(r.json()["replay_run_id"])
        assert child is not None
        assert "liquidation_mode" not in child["parameters"]


# ---------------------------------------------------------------------------
# Round 2 -- shared seeds repeat chromosome uids / DSLs across cells
# ---------------------------------------------------------------------------


def test_champion_buy_hold_comes_from_its_own_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    """B1: cells share seeds, so the same uid recurs in every cell; the selected champion's
    buy-and-hold record must be the one ITS cell produced, not the last cell's."""
    shared = _chrom()
    cells = ALL_CELLS[:2]

    def fake_run(self: Any) -> DiscoveryResult:
        idx = cells.index(self._backtest_fn.cell)
        twin = _chrom()
        twin.uid = shared.uid  # what a shared seed regenerates in the next cell
        return DiscoveryResult(
            generations=[],
            top_strategies=[(twin, _fit(adjusted_score=0.9 - 0.4 * idx, sharpe_ratio=2.0 + idx,
                                        mtm_daily_returns=_series(5.0)))],
            total_candidates_evaluated=30, converged=False, convergence_generation=None,
            buy_hold_records={shared.uid: {"sharpe_ann": 12.0 / (idx + 1)}},
        )  # fmt: skip

    monkeypatch.setattr(DiscoveryPipeline, "run", fake_run)
    cfg = _mock_cfg(prior_trials_fn=lambda s, e: 0, top_k=8)
    res, picked = pooling._run_bankroll_grid(cells, 1, cfg, _fns_for, "/tmp/vq_bg_progress.json")
    assert [c for c in picked] == [cells[0]]  # the stronger (isolated) twin is the pick
    assert res.buy_hold_records[shared.uid] == {"sharpe_ann": 12.0}


def test_pooled_series_keep_both_cells_for_the_same_dsl(monkeypatch: pytest.MonkeyPatch) -> None:
    """SF1/T8-R2: the same DSL in two cells with different series is two non-flat series."""
    cells = ALL_CELLS[:2]

    def fake_run(self: Any) -> DiscoveryResult:
        idx = cells.index(self._backtest_fn.cell)
        self._fitness_cache["same-dsl"] = _fit(mtm_daily_returns=_series(1.0, seed=idx + 1))
        return DiscoveryResult(
            generations=[], top_strategies=[(_chrom(), _fit(mtm_daily_returns=_series(5.0)))],
            total_candidates_evaluated=30, converged=False, convergence_generation=None,
        )  # fmt: skip

    monkeypatch.setattr(DiscoveryPipeline, "run", fake_run)
    cfg = _mock_cfg(prior_trials_fn=lambda s, e: 0, top_k=8)
    res, _ = pooling._run_bankroll_grid(cells, 1, cfg, _fns_for, "/tmp/vq_bg_progress.json")
    assert res.dsr_trials is not None
    assert res.dsr_trials["raw"] == 60
    assert res.dsr_trials["non_flat"] == 2
    assert res.dsr_trials["neff_mean_corr"] is not None


# ---------------------------------------------------------------------------
# Round 2 -- the REAL per-cell functions carry the cell's settings (R55)
# ---------------------------------------------------------------------------


def test_real_cell_fns_carry_the_cells_settings() -> None:
    """SF2: train, holdout and the cross-window / WFA factory are built at the cell's mode,
    leverage and sizing (cap, risk and margin cushion), not at defaults."""
    from vibe_quant.fees import FeeSchedule

    base = SizingSettings(max_position_pct=1.0, risk_per_trade=0.01)
    seen = set()
    for cell in (_cell("isolated", 1.0, 0.25), _cell("cross", 5.0, 0.5)):
        fns = discovery_run._make_cell_fns(
            use_mock=False, symbols=["BTCUSDT"], timeframe="1h", symbol_agg="portfolio",
            fee_schedule=FeeSchedule(), entry_order="market",
            liquidation=cell.liquidation, sizing=cell.sizing(base),
            train_start="2024-01-01", train_end="2024-06-01", eval_windows=None, min_trades=0,
            holdout=("2024-06-01", "2024-08-01"), needs_factory=True,
        )  # fmt: skip
        built = [fns.backtest_fn, fns.holdout_backtest_fn, fns.backtest_fn_factory("a", "b")]  # type: ignore[operator]
        for fn in built:
            assert isinstance(fn, NTBacktestFn)
            assert fn.liquidation == LiquidationSettings(mode=cell.mode, leverage=cell.leverage)
            assert fn.sizing == SizingSettings(
                max_position_pct=1.0, risk_per_trade=0.01, max_margin_pct=cell.max_margin_pct
            )
        seen.add((built[0].liquidation, built[0].sizing))
    assert len(seen) == 2
