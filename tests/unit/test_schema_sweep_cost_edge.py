"""Screening sweep rows persist the run's cost edge in four nullable columns (clmmc.11 T3).

``turnover, gross_edge_bps, cost_paid_bps, breakeven_fee_bps`` are written on EVERY row of a
batch (the batch writer binds positionally from row 0's keys): a ``-999`` sentinel stores four
NULLs, a no-fill run stores turnover 0.0 with NULL bps (never 0 bps, never NaN), and a DB from
before the columns migrates with its old rows reading NULL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.db.connection import get_connection
from vibe_quant.db.schema import SCHEMA_SQL
from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl import parse_strategy_string
from vibe_quant.liquidation_settings import LiquidationSettings
from vibe_quant.metrics import CostEdge
from vibe_quant.screening.pipeline import FAILED_BACKTEST_SHARPE, ScreeningPipeline
from vibe_quant.screening.types import BacktestMetrics, ScreeningResult

if TYPE_CHECKING:
    from pathlib import Path

COLUMNS = ("turnover", "gross_edge_bps", "cost_paid_bps", "breakeven_fee_bps")

_YAML = """
name: cost_edge_probe
timeframe: 4h
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


def _cols(state: StateManager, run_id: int) -> list[tuple[Any, ...]]:
    rows = state.conn.execute(
        f"SELECT {', '.join(COLUMNS)} FROM sweep_results WHERE run_id = ? ORDER BY id",
        (run_id,),
    ).fetchall()
    return [tuple(r) for r in rows]


def _run(state: StateManager) -> int:
    return state.create_backtest_run(
        strategy_id=None,
        run_mode="screening",
        symbols=["BTCUSDT"],
        timeframe="4h",
        start_date="2024-01-01",
        end_date="2024-03-01",
        parameters={},
    )


def _rows() -> list[BacktestMetrics]:
    return [
        BacktestMetrics(
            parameters={"a": 1},
            sharpe_ratio=FAILED_BACKTEST_SHARPE,
            # A sentinel never ran the engine: whatever it carries is not a measurement.
            cost_edge=CostEdge(9.0, 9.0, 9.0, 9.0),
        ),
        BacktestMetrics(
            parameters={"a": 2}, sharpe_ratio=1.0, cost_edge=CostEdge(2000.0, 10.0, 1.0, 0.0)
        ),
        BacktestMetrics(
            parameters={"a": 3}, sharpe_ratio=0.5, cost_edge=CostEdge(400.0, -3.0, 0.5, 0.0)
        ),
        BacktestMetrics(
            parameters={"a": 4}, sharpe_ratio=0.0, cost_edge=CostEdge(0.0, 0.0, 0.0, 0.0)
        ),
        BacktestMetrics(parameters={"a": 5}, sharpe_ratio=0.1),  # cost edge not recorded
    ]


@pytest.mark.parametrize("liquidation", ["off", "isolated"])
def test_save_results_writes_the_cost_edge_columns_on_every_row(
    tmp_path: Path, liquidation: str
) -> None:
    """R15: a mixed batch (sentinel + real + no-fill + unrecorded) binds without a
    ProgrammingError; the liquidation keys (mode on) do not shift the four columns."""
    settings = (
        LiquidationSettings()
        if liquidation == "off"
        else LiquidationSettings(mode="isolated", leverage=50.0)
    )
    state = StateManager(tmp_path / "ce.db")
    try:
        run_id = _run(state)
        ScreeningPipeline(dsl=parse_strategy_string(_YAML), liquidation=settings).save_results(
            ScreeningResult("cost_edge_probe", 5, 5, 0.1, _rows(), [1]), state, run_id
        )
        assert _cols(state, run_id) == [
            (None, None, None, None),
            (2000.0, 55.0, 5.0, 55.0),
            (400.0, -62.5, 12.5, -62.5),
            (0.0, None, None, None),
            (None, None, None, None),
        ]
    finally:
        state.close()


def test_old_db_gains_the_columns_and_old_rows_read_null(tmp_path: Path) -> None:
    """R15: a sweep_results table from before the columns migrates; its rows stay NULL."""
    db = tmp_path / "old.db"
    conn = get_connection(db)
    conn.executescript(SCHEMA_SQL)
    for column in COLUMNS:
        conn.execute(f"ALTER TABLE sweep_results DROP COLUMN {column}")
    conn.execute("PRAGMA foreign_keys = OFF")  # a bare row; no backtest_runs needed
    conn.execute(
        "INSERT INTO sweep_results (run_id, parameters, sharpe_ratio) VALUES (7, '{}', 1.5)"
    )
    conn.commit()
    conn.close()

    state = StateManager(db)
    try:
        (row,) = state.get_sweep_results(7)
    finally:
        state.close()
    assert row["sharpe_ratio"] == 1.5
    assert {c: row[c] for c in COLUMNS} == dict.fromkeys(COLUMNS)
