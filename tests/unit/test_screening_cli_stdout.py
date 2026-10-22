"""The screening CLI's partial stdout (contract row R27c).

The run log names the launch settings the run will use, one field at a time, as each
field is parsed. A later field's failure must therefore still print the lines of the
fields before it: ``rabmo.3``'s ``iter_fields`` keeps that order, so the four lines of
a good fee schedule, liquidation mode, venue leverage and balance are already on
stdout when a bad sizing setting aborts the run.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl import parse_strategy_string
from vibe_quant.screening.__main__ import cmd_run

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

_MINIMAL_YAML = """
name: cli_stdout_probe
timeframe: 5m
indicators:
  rsi:
    type: RSI
    period: 14
entry_conditions:
  long:
    - rsi < 30
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""


def _dsl() -> Any:
    return parse_strategy_string(_MINIMAL_YAML)


def _run(tmp_path: Path, parameters: dict[str, object]) -> tuple[int, list[str], dict[str, Any]]:
    """Drive ``cmd_run`` on a one-run DB; return (rc, stdout lines, run row)."""
    db = tmp_path / "cli_stdout.db"
    state = StateManager(db)
    try:
        state.create_strategy("cli_stdout_probe", {"timeframe": "5m", "indicators": {}})
        run_id = state.create_backtest_run(
            strategy_id=None,
            run_mode="screening",
            symbols=["BTCUSDT"],
            timeframe="5m",
            start_date="2024-01-01",
            end_date="2024-02-01",
            parameters={"dsl_override": _dsl().model_dump(), **parameters},
        )
        state.update_backtest_run_status(run_id, "running")
    finally:
        state.close()

    rc = cmd_run(argparse.Namespace(run_id=run_id, db=str(db)))

    state = StateManager(db)
    try:
        run = state.get_backtest_run(run_id)
    finally:
        state.close()
    assert run is not None
    return rc, [], run


def test_screening_cli_partial_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A good fee schedule + a bad sizing setting -> exactly 4 lines, then the failure."""
    rc, _, run = _run(tmp_path, {"max_position_pct": -1})
    lines = capsys.readouterr().out.splitlines()

    assert rc == 1
    assert lines == [
        "Screening fee schedule: {'fee_tier': 'VIP0', 'bnb_discount': False}",
        "Screening liquidation: None",
        "Screening venue leverage: 10x",
        "Screening starting balance: 1000 USDT",
        "Screening failed: max_position_pct must be a number in (0, 125], got -1",
    ]
    assert run["status"] == "failed"
    assert run["error_message"] == (
        "SizingConfigError: max_position_pct must be a number in (0, 125], got -1"
    )
