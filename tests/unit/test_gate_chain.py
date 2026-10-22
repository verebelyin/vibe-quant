"""New gate-chain module boundary (lane 4 T6).

Pins the two properties rabmo.6 T6 must establish before anything moves: the
new module imports cleanly in a FRESH process (the PEP 695 ``Candidate`` alias
must not need the TYPE_CHECKING-only names at runtime), and the rejection
record built by the module keeps today's key order (R21).
"""

from __future__ import annotations

import math
import subprocess
import sys
from array import array
from pathlib import Path
from typing import Any

import numpy as np

from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.discovery.fitness import FitnessResult
from vibe_quant.discovery.gate_chain import GateChain, GateEvidence, TrialBudget
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)
from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline
from vibe_quant.metrics import DailyReturns

_ROOT = Path(__file__).resolve().parents[2]
_ANN = math.sqrt(252.0)


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


# ---------------------------------------------------------------------------
# GateChain screen / pooled re-check (T7)
# ---------------------------------------------------------------------------


def _cfg(**kw: Any) -> DiscoveryConfig:
    base: dict[str, Any] = {
        "min_trades": 50, "max_workers": None, "timeframe": "1d",
        "symbols": ["BTCUSDT"], "start_date": "2024-01-01", "end_date": "2025-01-01",
        "require_bootstrap_ci": False, "require_dsr": False, "min_diversity_distance": 0.0,
    }  # fmt: skip
    base.update(kw)
    return DiscoveryConfig(**base)


def _nt_fn() -> NTBacktestFn:
    return NTBacktestFn(["BTCUSDT"], "1d", "2024-01-01", "2025-01-01")


def _chrom(**kw: Any) -> StrategyChromosome:
    base: dict[str, Any] = {
        "entry_genes": [StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 30.0, None)],
        "exit_genes": [StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0, None)],
        "stop_loss_pct": 2.0, "take_profit_pct": 4.0, "direction": Direction.LONG,
    }  # fmt: skip
    base.update(kw)
    return StrategyChromosome(**base)


def _fit(**kw: Any) -> FitnessResult:
    base: dict[str, Any] = {
        "sharpe_ratio": 0.5, "max_drawdown": 0.1, "profit_factor": 1.5,
        "total_trades": 200, "total_return": 0.3, "complexity_penalty": 0.0,
        "overtrade_penalty": 0.0, "sl_tp_penalty": 0.0, "raw_score": 0.5,
        "adjusted_score": 0.5, "passed_filters": True, "filter_results": {},
    }  # fmt: skip
    base.update(kw)
    return FitnessResult(**base)


def _series(sr_ann: float, n: int = 400, seed: int = 1) -> DailyReturns:
    z = np.random.default_rng(seed).normal(size=n)
    z = (z - z.mean()) / z.std(ddof=1)
    sigma = 0.01
    x = z * sigma + sr_ann / _ANN * sigma
    return DailyReturns(19_000, array("d", x.tobytes()))


def test_non_nt_fn_requires_returns_true_rejects_missing_series() -> None:
    """R15c: the explicit ``GateEvidence.dsr_requires_returns`` flag decides the input."""
    cfg = _cfg(require_dsr=True)
    gates = GateChain(cfg, GateEvidence(lambda _chrom: {}, True))

    kept = gates.screen([(_chrom(), _fit())], TrialBudget(10, {}))

    assert kept == []
    assert gates.rejections[0]["reasons"] == [
        "DSR input missing: no usable daily return series (length 0)"
    ]


def test_pooled_recheck_rewrites_stage_and_prefix() -> None:
    """R23: the pooled re-check stamps ``pooled_guardrails`` + the RAW pooled N."""
    cfg = _cfg(require_dsr=True)
    gates = GateChain(cfg, GateEvidence(_nt_fn(), True))
    weak = _series(0.2, seed=5)
    worst_chrom = _chrom(stop_loss_pct=2.0)
    dsr_chrom = _chrom(stop_loss_pct=3.0)
    candidates = [
        (worst_chrom, _fit(sharpe_ratio=1.0, total_trades=60, symbol_stats={"A": (0.0, 60)})),
        (dsr_chrom, _fit(sharpe_ratio=0.5, mtm_daily_returns=weak)),
    ]

    kept = gates.pooled_recheck(candidates, TrialBudget(200, {}))

    assert kept == []
    assert [r["stage"] for r in gates.rejections] == ["pooled_guardrails", "pooled_guardrails"]
    assert gates.rejections[0]["reasons"] == [
        "Multi-seed pooled N=200: worst-symbol train return <= 0 (A)"
    ]
    dsr_reasons = gates.rejections[1]["reasons"]
    assert len(dsr_reasons) == 1
    assert dsr_reasons[0].startswith("Multi-seed pooled N=200: DSR ")
    assert gates.dsr_trials is not None
    assert gates.dsr_trials["raw"] == 200


def test_run_rebuilds_the_gate_chain() -> None:
    """Risk 2: a reused pipeline starts every run with a FRESH ledger."""
    from vibe_quant.discovery.mock_backtest import mock_backtest

    cfg = _cfg(
        population_size=4, max_generations=1, elite_count=1, top_k=1, max_workers=1,
        require_buy_hold=False, start_date="2024-01-01", end_date="2024-02-01",
    )  # fmt: skip
    pipe = DiscoveryPipeline(cfg, mock_backtest)
    pipe.gates.reject(_chrom(), _fit(), "guardrails", ["stale"])
    stale = pipe.gates
    assert stale.rejections

    pipe.run()

    assert pipe.gates is not stale
    assert pipe.gates.rejections == []
    assert pipe.gates.dsr_trials is None and pipe.gates.dsr_records == {}
