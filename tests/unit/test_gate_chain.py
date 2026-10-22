"""New gate-chain module boundary (lane 4 T6).

Pins the two properties rabmo.6 T6 must establish before anything moves: the
new module imports cleanly in a FRESH process (the PEP 695 ``Candidate`` alias
must not need the TYPE_CHECKING-only names at runtime), and the rejection
record built by the module keeps today's key order (R21).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from vibe_quant.discovery.fitness import FitnessResult
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)

_ROOT = Path(__file__).resolve().parents[2]


def _import_in_fresh_process(module: str) -> None:
    done = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr


def test_fresh_process_import() -> None:
    _import_in_fresh_process("vibe_quant.discovery.gate_chain")
    _import_in_fresh_process("vibe_quant.discovery.pipeline")


def test_rejection_record_key_order() -> None:
    from vibe_quant.discovery.gate_chain import rejection_record

    chrom = StrategyChromosome(
        entry_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 30.0, None)],
        exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0, None)],
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        direction=Direction.LONG,
    )
    fitness = FitnessResult(
        sharpe_ratio=1.23456,
        max_drawdown=0.1,
        profit_factor=1.5,
        total_trades=200,
        total_return=0.3,
        complexity_penalty=0.0,
        overtrade_penalty=0.0,
        sl_tp_penalty=0.0,
        raw_score=0.5,
        adjusted_score=0.123456,
        passed_filters=True,
        filter_results={},
    )

    rec = rejection_record(chrom, fitness, "guardrails", ["r"], timeframe="4h")

    assert list(rec) == [
        "uid", "stage", "score", "sharpe", "trades", "reasons", "dsl",
        "fee_drag", "fees_pct_equity_yr",
    ]  # fmt: skip
    assert rec["dsl"]["timeframe"] == "4h"  # type: ignore[index]
    assert rec["score"] == 0.1235 and rec["sharpe"] == 1.235
