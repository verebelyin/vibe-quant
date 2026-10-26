"""Discovery adopts ``LaunchSettings`` (bead vibe-quant-rabmo.5 tail, T19/D6).

The adapter layer takes ONE :class:`~vibe_quant.launch_settings.LaunchSettings`
instead of the four loose kwargs ``fee_schedule, entry_order, liquidation,
sizing``. R15b is the load-bearing rule: a GRID run never builds a run-level
``LaunchSettings`` -- its ``__post_init__`` runs ``check_leverage(10)`` that the
grid path skips, so a grid with ``--max-position-pct 15`` would fail where it
used to complete. Each cell is checked at ITS leverage by ``build_cells``.

These tests drive the public entry point (``run_discovery``) with a recording
adapter, so every assertion is about the value the engine is actually handed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery import pooling, run
from vibe_quant.discovery.run_spec import build_parser
from vibe_quant.fees import FeeSchedule
from vibe_quant.jobs.manager import BacktestJobManager
from vibe_quant.launch_settings import LaunchSettings
from vibe_quant.liquidation_settings import LiquidationSettings
from vibe_quant.sizing_settings import SizingSettings

if TYPE_CHECKING:
    import argparse
    from pathlib import Path


_BASE_ARGV = [
    "--mock", "--seed", "7", "--population-size", "6", "--max-generations", "2",
    "--max-workers", "-1", "--start-date", "2024-01-01", "--end-date", "2025-01-01",
    "--timeframe", "4h", "--symbols", "BTCUSDT",
]  # fmt: skip


def _args(run_id: int, extra: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args([*_BASE_ARGV, *(extra or []), "--run-id", str(run_id)])


def _seeded_run(
    db: Path, parameters: dict[str, object] | None = None
) -> tuple[StateManager, BacktestJobManager, int]:
    state = StateManager(db)
    jobs = BacktestJobManager(db)
    run_id = state.create_backtest_run(
        None, "discovery", ["BTCUSDT"], "4h", "2024-01-01", "2024-06-01", parameters or {}
    )
    jobs.conn.execute(
        """INSERT INTO background_jobs (run_id, pid, job_type, status)
           VALUES (?, ?, 'discovery', 'running')""",
        (run_id, os.getpid()),
    )
    jobs.conn.commit()
    return state, jobs, run_id


@dataclass(frozen=True)
class _RecordingBacktests:
    """The mock adapter, recording every ``LaunchSettings`` it is handed."""

    launches: list[LaunchSettings] = field(default_factory=list)
    mock: bool = True

    def data_available(self, symbols: list[str]) -> bool:
        return True

    def cell_fns(
        self,
        *,
        symbols: list[str],
        timeframe: str,
        symbol_agg: object,
        launch: LaunchSettings,
        train_start: str,
        train_end: str,
        eval_windows: list[tuple[str, str]] | None,
        min_trades: int,
        holdout: tuple[str, str] | None,
        needs_factory: bool,
        evaluator: str,
    ) -> pooling._CellFns:
        self.launches.append(launch)
        return run.MockBacktests().cell_fns(
            symbols=symbols,
            timeframe=timeframe,
            symbol_agg=cast("run.SymbolAgg", symbol_agg),
            launch=launch,
            train_start=train_start,
            train_end=train_end,
            eval_windows=eval_windows,
            min_trades=min_trades,
            holdout=holdout,
            needs_factory=needs_factory,
            evaluator=evaluator,
        )

    def nt_fn(self, *, launch: LaunchSettings, **kwargs: object) -> object:
        self.launches.append(launch)
        return run.MockBacktests().nt_fn(launch=launch, **kwargs)  # type: ignore[arg-type]


def _drive(
    tmp_path: Path, extra: list[str], parameters: dict[str, object] | None = None
) -> tuple[int, list[LaunchSettings], StateManager, BacktestJobManager, int]:
    state, jobs, run_id = _seeded_run(tmp_path / "launch.db", parameters)
    adapter = _RecordingBacktests()
    outcome = run.run_discovery(
        _args(run_id, extra),
        adapter,  # type: ignore[arg-type]
        run.StateRecordStore(state, jobs),
        run_id=run_id,
        started_at=0.0,
    )
    return outcome.exit_code, adapter.launches, state, jobs, run_id


_GRID_ONE_CELL = [
    "--bankroll-modes", "isolated",
    "--bankroll-leverages", "20",
    "--bankroll-max-margin-pcts", "0.5",
    "--max-position-pct", "15",
]  # fmt: skip


def test_grid_with_cap_above_10_still_launches(tmp_path: Path) -> None:
    """R15b: a grid never builds a run-level LaunchSettings, so cap 15 at leverage 20 runs."""
    exit_code, launches, state, jobs, run_id = _drive(tmp_path, _GRID_ONE_CELL)
    try:
        assert exit_code == 0
        assert len(launches) == 1
        launch = launches[0]
        assert launch.liquidation == LiquidationSettings(mode="isolated", leverage=20.0)
        assert launch.sizing.max_position_pct == 15.0
        assert launch.sizing.max_margin_pct == 0.5
        row = state.get_backtest_run(run_id)
        assert row is not None and row["status"] == "completed"
    finally:
        state.close()
        jobs.close()


def test_grid_launch_carries_each_cells_leverage_and_cushion(tmp_path: Path) -> None:
    """Every cell's launch is the cell's own leverage/cushion, never another cell's."""
    exit_code, launches, state, jobs, _run_id = _drive(
        tmp_path,
        [
            "--bankroll-modes", "isolated",
            "--bankroll-leverages", "5,20",
            "--bankroll-max-margin-pcts", "0.25,0.5",
            "--max-position-pct", "2",
        ],
    )  # fmt: skip
    try:
        assert exit_code == 0
        got = {
            (launch.liquidation.mode, launch.liquidation.leverage, launch.sizing.max_margin_pct)
            for launch in launches
        }
        assert got == {
            ("isolated", 5.0, 0.25),
            ("isolated", 5.0, 0.5),
            ("isolated", 20.0, 0.25),
            ("isolated", 20.0, 0.5),
        }
        for launch in launches:
            assert launch.sizing.max_position_pct == 2.0
    finally:
        state.close()
        jobs.close()


def test_non_grid_cap_above_leverage_fails_before_the_engine(tmp_path: Path) -> None:
    """The run-level check stays the first raise and the adapter is never called."""
    exit_code, launches, state, jobs, run_id = _drive(tmp_path, ["--max-position-pct", "15"])
    try:
        assert exit_code == 1
        assert launches == []
        row = state.get_backtest_run(run_id)
        assert row is not None and row["status"] == "failed"
        assert row["error_message"] == (
            "SizingConfigError: max_position_pct (15.0) must not exceed leverage (10.0)"
        )
    finally:
        state.close()
        jobs.close()


def test_non_grid_forwards_the_run_launch(tmp_path: Path) -> None:
    """The non-grid launch is exactly the spec's settings plus the run row's entry order."""
    exit_code, launches, state, jobs, _run_id = _drive(
        tmp_path,
        [
            "--fee-tier", "VIP0",
            "--bnb-discount",
            "--leverage", "5",
            "--liquidation-mode", "isolated",
            "--max-position-pct", "2",
            "--risk-per-trade", "0.01",
        ],
        {"entry_order": "market"},
    )  # fmt: skip
    try:
        assert exit_code == 0
        assert launches == [
            LaunchSettings(
                fee_schedule=FeeSchedule(tier="VIP0", bnb_discount=True),
                liquidation=LiquidationSettings(mode="isolated", leverage=5.0),
                sizing=SizingSettings(max_position_pct=2.0, risk_per_trade=0.01),
                entry_order="market",
            )
        ]
    finally:
        state.close()
        jobs.close()


def test_non_grid_forwards_the_run_rows_entry_order(tmp_path: Path) -> None:
    """``entry_order`` comes from the run row's parameters, not from the spec."""
    exit_code, launches, state, jobs, _run_id = _drive(
        tmp_path,
        [
            "--fee-tier", "VIP0",
            "--bnb-discount",
            "--leverage", "5",
            "--liquidation-mode", "isolated",
            "--max-position-pct", "2",
            "--risk-per-trade", "0.01",
        ],
        {"entry_order": "limit"},
    )  # fmt: skip
    try:
        assert exit_code == 0
        assert launches == [
            LaunchSettings(
                fee_schedule=FeeSchedule(tier="VIP0", bnb_discount=True),
                liquidation=LiquidationSettings(mode="isolated", leverage=5.0),
                sizing=SizingSettings(max_position_pct=2.0, risk_per_trade=0.01),
                entry_order="limit",
            )
        ]
    finally:
        state.close()
        jobs.close()
