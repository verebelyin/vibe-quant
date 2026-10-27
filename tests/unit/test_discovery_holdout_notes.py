"""Outcome test: a discovery run persists holdout + cross-window results into notes.

Bead vibe-quant-nxmxd.18: deleting ``entry["holdout"] = ...`` or
``entry["cross_window"] = ...`` in ``vibe_quant/discovery/run.py`` survived every
test, because mock runs carry no holdout/cross-window results. The promote-time
holdout check reads these notes, so a silent delete would make every promoted
champion look unvalidated.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from tests.unit.test_run_discovery import drive_run_discovery
from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery import run as discovery_run
from vibe_quant.discovery.fitness import FitnessResult
from vibe_quant.discovery.gate_chain import CrossWindowResult, HoldoutResult
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)
from vibe_quant.discovery.pipeline import DiscoveryPipeline, DiscoveryResult

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _create_discovery_run(db_path: Path) -> int:
    state = StateManager(db_path)
    strategy_id = state.create_strategy(
        name="__test_discovery_holdout_notes__",
        dsl_config={"type": "discovery"},
        description="test strategy for discovery holdout notes",
        strategy_type="discovery",
    )
    run_id = state.create_backtest_run(
        strategy_id=strategy_id,
        run_mode="discovery",
        symbols=["BTCUSDT"],
        timeframe="1h",
        start_date="2025-01-01",
        end_date="2025-02-01",
        parameters={},
    )
    state.close()
    return run_id


def _chrom(period: float) -> StrategyChromosome:
    return StrategyChromosome(
        entry_genes=[StrategyGene("RSI", {"period": period}, ConditionType.LT, 30.0)],
        exit_genes=[StrategyGene("RSI", {"period": period}, ConditionType.GT, 70.0)],
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        direction=Direction.LONG,
    )


def _fit(sharpe: float) -> FitnessResult:
    return FitnessResult(
        sharpe_ratio=sharpe,
        max_drawdown=0.05,
        profit_factor=1.1,
        total_trades=40,
        total_return=0.03,
        complexity_penalty=0.0,
        overtrade_penalty=0.0,
        sl_tp_penalty=0.0,
        raw_score=sharpe,
        adjusted_score=sharpe,
        passed_filters=True,
        filter_results={},
    )


def _window(
    sharpe: float, max_dd: float, pf: float, trades: int, total_return: float
) -> HoldoutResult:
    return HoldoutResult(
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        profit_factor=pf,
        total_trades=trades,
        total_return=total_return,
    )


def test_holdout_and_cross_window_results_persist_into_champion_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two champions, two unrelated payload pairs: each entry carries its own."""
    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)

    holdouts = [
        _window(0.82, 0.03, 1.2, 30, 0.02),
        _window(1.71, 0.11, 1.9, 77, 0.13),
    ]
    cross_windows = [
        CrossWindowResult(
            window_results=[_window(0.91, 0.04, 1.3, 11, 0.07)],
            windows_passed=1,
            total_windows=1,
            passed=True,
            window_dates=[("2024-05-01", "2024-06-01")],
            offsets_months=[1],
            required=1,
        ),
        CrossWindowResult(
            window_results=[
                _window(0.55, 0.09, 1.05, 5, 0.01),
                _window(1.23, 0.06, 1.6, 23, 0.08),
            ],
            windows_passed=1,
            total_windows=2,
            passed=False,
            window_dates=[
                ("2024-06-01", "2024-07-01"),
                ("2024-07-01", "2024-08-01"),
            ],
            offsets_months=[1, 2],
            required=2,
        ),
    ]

    def fake_run(self: Any) -> DiscoveryResult:
        return DiscoveryResult(
            generations=[],
            top_strategies=[(_chrom(14.0), _fit(1.5)), (_chrom(21.0), _fit(2.2))],
            total_candidates_evaluated=10,
            converged=True,
            convergence_generation=1,
            holdout_results=holdouts,
            cross_window_results=cross_windows,
            train_dates=("2025-01-01", "2025-01-25"),
            holdout_dates=("2025-01-25", "2025-02-01"),
        )

    monkeypatch.setattr(DiscoveryPipeline, "run", fake_run)

    argv = [
        "--run-id", str(run_id),
        "--population-size", "6", "--max-generations", "2", "--elite-count", "1",
        "--symbols", "BTCUSDT", "--timeframe", "1h",
        "--start-date", "2024-01-01", "--end-date", "2025-01-01",
        "--eval-windows", "2", "--train-test-split", "0.8",
        "--cross-window-months", "1",
        "--db", str(db_path), "--mock", "--max-workers", "-1",
    ]  # fmt: skip

    outcome = drive_run_discovery(db_path, run_id, argv, discovery_run.MockBacktests())
    assert outcome.exit_code == 0

    state = StateManager(db_path)
    result = state.get_backtest_result(run_id)
    state.close()
    assert result is not None
    notes = json.loads(result["notes"])
    entries = notes["top_strategies"]
    assert len(entries) == 2

    assert entries[0]["holdout"] == {
        "sharpe": 0.82,
        "max_dd": 0.03,
        "pf": 1.2,
        "trades": 30,
        "return_pct": 0.02,
    }
    assert entries[0]["cross_window"] == {
        "windows_passed": 1,
        "total_windows": 1,
        "required": 1,
        "passed": True,
        "in_sample_excluded": True,
        "windows": [
            {
                "sharpe": 0.91,
                "max_dd": 0.04,
                "pf": 1.3,
                "trades": 11,
                "return_pct": 0.07,
                "offset_months": 1,
                "dates": ["2024-05-01", "2024-06-01"],
            }
        ],
    }

    assert entries[1]["holdout"] == {
        "sharpe": 1.71,
        "max_dd": 0.11,
        "pf": 1.9,
        "trades": 77,
        "return_pct": 0.13,
    }
    assert entries[1]["cross_window"] == {
        "windows_passed": 1,
        "total_windows": 2,
        "required": 2,
        "passed": False,
        "in_sample_excluded": True,
        "windows": [
            {
                "sharpe": 0.55,
                "max_dd": 0.09,
                "pf": 1.05,
                "trades": 5,
                "return_pct": 0.01,
                "offset_months": 1,
                "dates": ["2024-06-01", "2024-07-01"],
            },
            {
                "sharpe": 1.23,
                "max_dd": 0.06,
                "pf": 1.6,
                "trades": 23,
                "return_pct": 0.08,
                "offset_months": 2,
                "dates": ["2024-07-01", "2024-08-01"],
            },
        ],
    }
