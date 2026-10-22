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
from typing import TYPE_CHECKING, Any

import numpy as np

from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.discovery.fitness import FitnessResult
from vibe_quant.discovery.gate_chain import (
    REGIME_RETURN_THRESHOLD,
    GateChain,
    GateEvidence,
    TrialBudget,
    passes_opposing_regime_gate,
    window_regime_sign,
)
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)
from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline
from vibe_quant.metrics import DailyReturns

if TYPE_CHECKING:
    import pytest

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


def test_validate_order_buy_hold_before_cross_window() -> None:
    """R30: buy-hold opens the ladder; a rejected candidate never reaches a window."""
    factory_calls: list[tuple[str, str]] = []
    closes_calls: list[list[str]] = []

    def metrics(_chrom: StrategyChromosome) -> dict[str, float | int]:
        return {
            "sharpe_ratio": 2.0,
            "max_drawdown": 0.1,
            "profit_factor": 2.0,
            "total_trades": 60,
            "total_return": 0.1,
        }

    def factory(ws: str, we: str) -> Any:
        factory_calls.append((ws, we))
        return metrics

    def closes(symbols: list[str]) -> dict[str, dict[int, float]]:
        closes_calls.append(list(symbols))
        return {}

    cfg = _cfg(
        require_buy_hold=True,
        cross_window_months=[1],
        wfa_oos_step_days=30,
        start_date="2024-01-01",
        end_date="2024-05-01",
    )
    gates = GateChain(
        cfg,
        GateEvidence(_nt_fn(), False, backtest_fn_factory=factory, buy_hold_closes_fn=closes),
    )

    outcome = gates.validate([(_chrom(), _fit())])

    assert outcome.survivors == []
    assert [r["stage"] for r in gates.rejections] == ["buy_hold"]
    assert gates.rejections[0]["reasons"] == ["Buy-hold: no daily series"]
    assert closes_calls == [["BTCUSDT"]]
    assert factory_calls == []


def test_run_gates_use_factory_rewrapped_by_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R41: ``_preflight_bars`` reassigns the factory; the run's gates must use it."""
    from vibe_quant.discovery.mock_backtest import mock_backtest

    original_calls: list[tuple[str, str]] = []
    wrapped_calls: list[tuple[str, str]] = []

    def metrics(_chrom: StrategyChromosome) -> dict[str, float | int]:
        return {
            "sharpe_ratio": 2.0,
            "max_drawdown": 0.1,
            "profit_factor": 2.0,
            "total_trades": 60,
            "total_return": 0.1,
        }

    def original_factory(ws: str, we: str) -> Any:
        original_calls.append((ws, we))
        return metrics

    def wrapped_factory(ws: str, we: str) -> Any:
        wrapped_calls.append((ws, we))
        return metrics

    cfg = _cfg(
        population_size=4,
        max_generations=1,
        elite_count=1,
        top_k=1,
        max_workers=1,
        require_buy_hold=False,
        cross_window_months=[1],
        wfa_oos_step_days=30,
        start_date="2024-01-01",
        end_date="2024-05-01",
    )
    pipe = DiscoveryPipeline(cfg, mock_backtest, backtest_fn_factory=original_factory)
    candidate = (_chrom(), _fit())
    monkeypatch.setattr(GateChain, "screen", lambda self, top, budget: [candidate])
    monkeypatch.setattr(
        pipe, "_preflight_bars", lambda: setattr(pipe, "_backtest_fn_factory", wrapped_factory)
    )

    result = pipe.run()

    assert wrapped_calls, "cross-window / WFA must call the reassigned factory"
    assert original_calls == []
    assert [c.uid for c, _ in result.top_strategies] == [candidate[0].uid]


def test_run_screen_receives_the_run_series_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Q2: ``_run`` hands screen the run's fitness cache, not an empty pool."""
    from vibe_quant.discovery.mock_backtest import mock_backtest

    seen: list[tuple[int, int]] = []
    real_screen = GateChain.screen

    def spy(self: Any, top: Any, budget: TrialBudget) -> Any:
        seen.append((budget.total_evaluated, len(budget.series_pool)))
        return real_screen(self, top, budget)

    monkeypatch.setattr(GateChain, "screen", spy)

    for population_size in (4, 6):
        cfg = _cfg(
            population_size=population_size, max_generations=1, elite_count=1,
            top_k=1, max_workers=1, require_buy_hold=False,
            start_date="2024-01-01", end_date="2024-02-01",
        )  # fmt: skip
        DiscoveryPipeline(cfg, mock_backtest).run()

    assert seen == [(4, 4), (6, 6)]


# ---------------------------------------------------------------------------
# Promote opposing-regime gate (T11)
# ---------------------------------------------------------------------------


def test_regime_gate_neutral_base_requires_bull() -> None:
    """R54: a neutral base demands a bull window; a bear window never passes."""
    calls: list[tuple[str, str, str]] = []
    payload: dict[str, object] = {
        "cross_window_months": [1],
        "train_dates": ["2024-01-01", "2025-01-01"],
        "cross_window_min_sharpe": 0.5,
    }
    discovery_run: dict[str, object] = {
        "timeframe": "1m",
        "symbols": ["BTCUSDT"],
        "start_date": "2023-01-01",
        "end_date": "2026-01-01",
    }
    entry: dict[str, object] = {
        "chromosome": {"direction": "short"},
        "cross_window": {
            "windows": [
                {"sharpe": 0.5, "return_pct": 0.1, "dates": ["2024-02-01", "2025-02-01"]},
            ]
        },
    }

    def classifier(sign: int) -> Any:
        def _sign(symbol: str, start: str, end: str) -> int:
            calls.append((symbol, start, end))
            return 0 if (start, end) == ("2024-01-01", "2025-01-01") else sign

        return _sign

    assert (
        passes_opposing_regime_gate(payload, discovery_run, entry, regime_sign=classifier(-1))
        is False
    )
    assert calls == [
        ("BTCUSDT", "2024-01-01", "2025-01-01"),
        ("BTCUSDT", "2024-02-01", "2025-02-01"),
    ]

    calls.clear()
    assert (
        passes_opposing_regime_gate(payload, discovery_run, entry, regime_sign=classifier(1))
        is True
    )
    assert calls == [
        ("BTCUSDT", "2024-01-01", "2025-01-01"),
        ("BTCUSDT", "2024-02-01", "2025-02-01"),
    ]


def test_regime_sign_band_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    """T11 note: +-5% is the edge; the band test is strict ``<``, so 5% is directional."""
    from types import SimpleNamespace

    import vibe_quant.validation.random_baseline as random_baseline

    def sign_for(bars: list[object]) -> int:
        monkeypatch.setattr(random_baseline, "load_ohlc", lambda *_args, **_kwargs: bars)
        return window_regime_sign("BTCUSDT", "2024-01-01", "2024-02-01")

    def two_bars(first_open: float, last_close: float) -> list[object]:
        return [
            SimpleNamespace(open=first_open, close=0.0),
            SimpleNamespace(open=0.0, close=last_close),
        ]

    assert REGIME_RETURN_THRESHOLD == 0.05
    assert sign_for(two_bars(100.0, 105.0)) == 1
    assert sign_for(two_bars(100.0, 104.99)) == 0
    assert sign_for(two_bars(100.0, 95.0)) == -1
    assert sign_for(two_bars(100.0, 95.01)) == 0
    assert sign_for(two_bars(100.0, 100.0)) == 0
    assert sign_for([]) == 0
    assert sign_for([SimpleNamespace(open=100.0, close=105.0)]) == 0
    assert sign_for(two_bars(0.0, 105.0)) == 0


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
