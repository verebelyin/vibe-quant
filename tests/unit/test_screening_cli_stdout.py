"""The screening CLI's partial stdout (contract row R27c).

The run log names the launch settings the run will use, one field at a time, as each
field is parsed. A later field's failure must therefore still print the lines of the
fields before it: ``rabmo.3``'s ``iter_fields`` keeps that order, so the three lines of
a good starting balance, liquidation mode and venue leverage are already on stdout
when a bad sizing setting aborts the run.

S5b moved the CLI to the canonical field order (balance, liquidation, sizing, fee): a
complete run prints the five ``Screening`` field lines in that order, and a run with two
bad fields shows the earlier one's error, because the loop stops at the first bad field.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl import parse_strategy_string
from vibe_quant.screening.__main__ import cmd_run
from vibe_quant.screening.types import BacktestMetrics

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


def test_screening_cli_partial_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A good balance + liquidation, then a bad sizing -> exactly 3 lines, then failure."""
    rc, _, run = _run(tmp_path, {"max_position_pct": -1})
    lines = capsys.readouterr().out.splitlines()

    assert rc == 1
    assert lines == [
        "Screening starting balance: 1000 USDT",
        "Screening liquidation: None",
        "Screening venue leverage: 10x",
        "Screening failed: max_position_pct must be a number in (0, 125], got -1",
    ]
    assert run["status"] == "failed"
    assert run["error_message"] == (
        "SizingConfigError: max_position_pct must be a number in (0, 125], got -1"
    )


def _mock_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """A screening pipeline that runs with no engine and one worker (no ticks, no bars)."""
    import vibe_quant.screening.nt_runner as nt_runner
    from vibe_quant.screening import pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "resolve_parent_fill_ticks", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_mod, "cpu_count", lambda: 1)

    class _OkRunner:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def __call__(self, params: dict[str, float | int]) -> BacktestMetrics:
            return BacktestMetrics(parameters=params, sharpe_ratio=1.0, total_return=0.1)

    monkeypatch.setattr(nt_runner, "NTScreeningRunner", _OkRunner)


def test_screening_cli_prints_the_canonical_field_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete run: the five field lines in balance, liquidation, sizing, fee order."""
    _mock_pipeline(monkeypatch)
    rc, _, run = _run(
        tmp_path,
        {
            "initial_balance": 10000,
            "bnb_discount": True,
            "liquidation_mode": "isolated",
            "leverage": 3,
            "max_position_pct": 2,
        },
    )
    lines = capsys.readouterr().out.splitlines()

    assert rc == 0
    assert run["status"] == "completed"
    assert [line for line in lines if line.startswith("Screening ")] == [
        "Screening starting balance: 10000 USDT",
        "Screening liquidation: {'mode': 'isolated', 'leverage': 3.0}",
        "Screening venue leverage: 3x",
        "Screening sizing: max_position_pct=2 risk_per_trade=0.02",
        "Screening fee schedule: {'fee_tier': 'VIP0', 'bnb_discount': True}",
        "Screening complete: 1 combos, 0 Pareto-optimal",
    ]


def test_screening_cli_bad_balance_wins_over_a_bad_fee(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two bad fields: balance is parsed before the fee, so its error is the one shown."""
    rc, _, run = _run(tmp_path, {"initial_balance": 0, "fee_tier": "VIP99"})
    lines = capsys.readouterr().out.splitlines()

    assert rc == 1
    assert lines == [
        "Screening failed: initial_balance must be a whole number in [1, 1000000000], got 0"
    ]
    assert run["error_message"] == (
        "BalanceConfigError: initial_balance must be a whole number in [1, 1000000000], got 0"
    )


def test_screening_cli_bad_sizing_wins_over_a_bad_fee(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two bad fields: sizing is parsed before the fee, leaving 3 lines then its error."""
    rc, _, run = _run(tmp_path, {"max_position_pct": -1, "fee_tier": "VIP99"})
    lines = capsys.readouterr().out.splitlines()

    assert rc == 1
    assert lines == [
        "Screening starting balance: 1000 USDT",
        "Screening liquidation: None",
        "Screening venue leverage: 10x",
        "Screening failed: max_position_pct must be a number in (0, 125], got -1",
    ]
    assert run["error_message"] == (
        "SizingConfigError: max_position_pct must be a number in (0, 125], got -1"
    )
