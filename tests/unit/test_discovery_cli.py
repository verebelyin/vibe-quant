"""Tests for discovery CLI entrypoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery.__main__ import build_parser, main

if TYPE_CHECKING:
    from pathlib import Path


def _create_discovery_run(db_path: Path) -> int:
    state = StateManager(db_path)
    strategy_id = state.create_strategy(
        name="__test_discovery_cli__",
        dsl_config={"type": "discovery"},
        description="test strategy for discovery CLI",
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


def test_build_parser_requires_run_id() -> None:
    """CLI parser should require run id."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--run-id", "42",
            "--start-date", "2025-01-01",
            "--end-date", "2025-02-01",
        ]
    )
    assert args.run_id == 42


def test_main_runs_and_persists_result(tmp_path: Path, monkeypatch) -> None:
    """Running discovery CLI should complete run and persist summary metrics."""
    db_path = tmp_path / "state.db"
    run_id = _create_discovery_run(db_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--run-id", str(run_id),
            "--population-size", "6",
            "--max-generations", "2",
            "--mutation-rate", "0.1",
            "--elite-count", "1",
            "--symbols", "BTCUSDT",
            "--timeframe", "1h",
            "--start-date", "2025-01-01",
            "--end-date", "2025-02-01",
            "--db", str(db_path),
        ],
    )

    assert main() == 0

    state = StateManager(db_path)
    run = state.get_backtest_run(run_id)
    result = state.get_backtest_result(run_id)
    state.close()

    assert run is not None
    assert run["status"] == "completed"
    assert result is not None
    assert result["total_trades"] >= 0
