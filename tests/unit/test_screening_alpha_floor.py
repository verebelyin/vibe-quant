"""A screening sweep value below the ADAPTIVE_RSI alpha floor fails the whole run.

Chief ruling B3 (bead vibe-quant-b9uet): a refused combination never becomes a -inf /
zero-liquidation sweep row; the run ends ``failed`` with the refusal text and no
sweep_results row, and a mixed grid fails too (the user relaunches without the value).
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.screening.__main__ import cmd_run

if TYPE_CHECKING:
    from pathlib import Path

R1 = "ADAPTIVE_RSI alpha must be between 0.7 and 1.0 (got 0.3)"

_DSL: dict[str, Any] = {
    "name": "arsi_floor",
    "timeframe": "1m",
    "indicators": {"arsi": {"type": "ADAPTIVE_RSI", "period": 14, "alpha": 0.8}},
    "entry_conditions": {"long": ["arsi < 30"]},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 4.0},
}


@pytest.mark.slow(reason="real NT screening runs over one day of 1m BTCUSDT, ~5 s each")
@pytest.mark.parametrize(
    ("launch", "alphas", "sequential"),
    [
        ({}, [0.3], False),
        ({}, [0.3, 0.8], False),
        ({}, [0.8, 0.3], True),
        ({"liquidation_mode": "isolated", "leverage": 10}, [0.3], False),
    ],
    ids=["default-refused", "default-mixed", "sequential-mixed", "isolated-refused"],
)
def test_sweep_below_alpha_floor_fails_run_without_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    launch: dict[str, Any],
    alphas: list[float],
    sequential: bool,
) -> None:
    monkeypatch.setenv("VIBE_QUANT_LOG_DIR", str(tmp_path / "logs"))
    if sequential:
        # One worker: the pipeline runs the grid in-process, not in the pool.
        monkeypatch.setattr("vibe_quant.screening.pipeline.cpu_count", lambda: 2)
    db = tmp_path / "state.db"
    state = StateManager(db)
    try:
        strategy_id = state.create_strategy(name="arsi_floor", dsl_config=_DSL)
        run_id = state.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="screening",
            symbols=["BTCUSDT"],
            timeframe="1m",
            start_date="2024-01-06",
            end_date="2024-01-07",
            parameters={**launch, "sweep": {"arsi_alpha": alphas}},
        )
    finally:
        state.close()

    code = cmd_run(argparse.Namespace(run_id=run_id, db=str(db)))

    state = StateManager(db)
    try:
        run = state.get_backtest_run(run_id)
        rows = state.get_sweep_results(run_id)
    finally:
        state.close()
    assert run is not None
    assert code != 0
    assert run["status"] == "failed"
    assert R1 in (run.get("error_message") or "")
    assert rows == []
