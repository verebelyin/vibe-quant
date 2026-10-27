"""Characterization contract for the discovery champion screen (lane 4 T2).

Pins today's behaviour of the guardrail screen and of the pooled multi-seed
re-check before rabmo.6 extracts ``vibe_quant/discovery/gate_chain.py``. Rows
R10-R25 (plus R15c) below call ONLY the ADAPTER block at the top; every private
touch of ``DiscoveryPipeline`` lives there, so T7 flips the adapter to
``GateChain(cfg, GateEvidence(fn, flag)).screen(top, TrialBudget(n, pool))``
without touching a row.

Gates fail closed: the reason texts are pinned verbatim.
"""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from vibe_quant.discovery import pooling
from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.discovery.fitness import FitnessResult
from vibe_quant.discovery.gate_chain import GateChain, GateEvidence, TrialBudget
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)
from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline, DiscoveryResult
from vibe_quant.metrics import DailyReturns

_ANN = math.sqrt(252.0)


# ---------------------------------------------------------------------------
# Plain helpers: no pipeline privates here.
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


def _plain(_chrom: StrategyChromosome) -> dict[str, Any]:
    return {}


def _series(sr_ann: float, n: int = 400, seed: int = 1) -> DailyReturns:
    z = np.random.default_rng(seed).normal(size=n)
    z = (z - z.mean()) / z.std(ddof=1)
    sigma = 0.01
    x = z * sigma + sr_ann / _ANN * sigma
    return DailyReturns(19_000, array("d", x.tobytes()))


def _chrom(**kw: Any) -> StrategyChromosome:
    base: dict[str, Any] = {
        "entry_genes": [StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 30.0, None)],
        "exit_genes": [StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0, None)],
        "stop_loss_pct": 2.0, "take_profit_pct": 4.0, "direction": Direction.LONG,
    }  # fmt: skip
    base.update(kw)
    return StrategyChromosome(**base)


def _genes(n: int) -> tuple[list[StrategyGene], list[StrategyGene]]:
    entry = [
        StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 20.0 + i, None) for i in range(n // 2)
    ]
    exit_ = [
        StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 60.0 + i, None)
        for i in range(n - n // 2)
    ]
    return entry, exit_


def _fit(**kw: Any) -> FitnessResult:
    base: dict[str, Any] = {
        "sharpe_ratio": 0.5, "max_drawdown": 0.1, "profit_factor": 1.5,
        "total_trades": 200, "total_return": 0.3, "complexity_penalty": 0.0,
        "overtrade_penalty": 0.0, "sl_tp_penalty": 0.0, "raw_score": 0.5,
        "adjusted_score": 0.5, "passed_filters": True, "filter_results": {},
    }  # fmt: skip
    base.update(kw)
    return FitnessResult(**base)


# ---------------------------------------------------------------------------
# ADAPTER: the only gate-internal access in this file (GateChain + GateEvidence
# + TrialBudget); the rows below must stay unchanged.
# ---------------------------------------------------------------------------


@dataclass
class _SeedOut:
    """One faked ``DiscoveryPipeline.run`` result of a pooled search."""

    champion: tuple[StrategyChromosome, FitnessResult]
    evaluated: int = 100
    cache: dict[str, FitnessResult] = field(default_factory=dict)
    buy_hold: dict[str, dict[str, object]] = field(default_factory=dict)


ScreenResult = tuple[
    list[str],
    list[dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, object] | None,
]


def _screen(
    cfg: DiscoveryConfig,
    fn: Any,
    requires_returns: bool,
    top: list[tuple[StrategyChromosome, FitnessResult]],
    n: int,
    pool: dict[str, FitnessResult] | None = None,
) -> ScreenResult:
    gates = GateChain(cfg, GateEvidence(fn, requires_returns))
    kept = gates.screen(top, TrialBudget(n, pool or {}))
    return (
        [chrom.uid for chrom, _ in kept],
        list(gates.rejections),
        dict(gates.dsr_records),
        gates.dsr_trials,
    )


def _record(
    cfg: DiscoveryConfig,
    fn: Any,
    chrom: StrategyChromosome,
    fitness: FitnessResult,
    stage: str,
    reasons: list[str],
    *,
    dsr: dict[str, object] | None = None,
    buy_hold: dict[str, object] | None = None,
) -> dict[str, object]:
    gates = GateChain(cfg, GateEvidence(fn, False))
    gates.reject(chrom, fitness, stage, reasons, dsr=dsr, buy_hold=buy_hold)
    return gates.rejections[-1]


def _pooled(
    monkeypatch: pytest.MonkeyPatch,
    cfg: DiscoveryConfig,
    fn: Any,
    seeds: list[_SeedOut],
) -> DiscoveryResult:
    pending = list(seeds)

    def fake_run(self: DiscoveryPipeline) -> DiscoveryResult:
        seed = pending.pop(0)
        self._fitness_cache.update(seed.cache)
        return DiscoveryResult(
            generations=[],
            top_strategies=[seed.champion],
            total_candidates_evaluated=seed.evaluated,
            converged=False,
            convergence_generation=None,
            buy_hold_records=dict(seed.buy_hold),
        )

    monkeypatch.setattr(DiscoveryPipeline, "run", fake_run)
    return pooling._run_multi_seed(len(seeds), cfg, fn, "/tmp/vq_rabmo_l4_t2_progress.json")


# ---------------------------------------------------------------------------
# R12 -- first test: the min-trades tie passes
# ---------------------------------------------------------------------------


def test_min_trades_tie_passes() -> None:
    cfg = _cfg()
    kept, rejections, _, _ = _screen(cfg, _nt_fn(), False, [(_chrom(), _fit(total_trades=50))], 10)
    assert len(kept) == 1 and rejections == []

    _, rejections, _, _ = _screen(cfg, _nt_fn(), False, [(_chrom(), _fit(total_trades=49))], 10)
    assert rejections[0]["reasons"] == ["Too few trades: 49 < 50"]


# ---------------------------------------------------------------------------
# R10-R15c: worst-symbol pre-gate and the guardrail checks
# ---------------------------------------------------------------------------


def test_worst_symbol_zero_return_rejects() -> None:
    cfg = _cfg(require_dsr=True)
    chrom = _chrom()
    fit = _fit(total_trades=60, symbol_stats={"A": (0.0, 60)})
    kept, rejections, records, _ = _screen(cfg, _nt_fn(), True, [(chrom, fit)], 10)

    assert kept == []
    assert rejections[0]["stage"] == "worst_symbol_train"
    assert rejections[0]["reasons"] == ["worst-symbol train return <= 0 (A)"]
    assert "dsr" not in rejections[0]
    assert chrom.uid not in records


def test_worst_symbol_nan_and_trades() -> None:
    cfg = _cfg(require_dsr=True)
    fit = _fit(total_trades=49, symbol_stats={"A": (float("nan"), 49)})
    _, rejections, _, _ = _screen(cfg, _nt_fn(), True, [(_chrom(), fit)], 10)
    assert rejections[0]["reasons"] == [
        "worst-symbol train return <= 0 (A)",
        "worst-symbol train trades < 50 (A)",
    ]


def test_min_return_rows() -> None:
    cfg = _cfg()
    kept, _, _, _ = _screen(cfg, _nt_fn(), False, [(_chrom(), _fit(total_return=0.0))], 10)
    assert len(kept) == 1

    _, rejections, _, _ = _screen(cfg, _nt_fn(), False, [(_chrom(), _fit(total_return=-0.01))], 10)
    assert rejections[0]["reasons"] == ["Return -1.00% below minimum 0.00%"]

    _, rejections, _, _ = _screen(
        cfg, _nt_fn(), False, [(_chrom(), _fit(total_return=float("nan")))], 10
    )
    assert rejections[0]["reasons"] == ["Return nan% below minimum 0.00%"]


def test_complexity_tie() -> None:
    cfg = _cfg()
    entry, exit_ = _genes(10)
    kept, _, _, _ = _screen(
        cfg, _nt_fn(), False, [(_chrom(entry_genes=entry, exit_genes=exit_), _fit())], 10
    )
    assert len(kept) == 1

    entry, exit_ = _genes(11)
    _, rejections, _, _ = _screen(
        cfg, _nt_fn(), False, [(_chrom(entry_genes=entry, exit_genes=exit_), _fit())], 10
    )
    assert rejections[0]["reasons"] == ["Too complex: 11 genes > 10 max"]


def test_dsr_missing_series_fails_closed() -> None:
    cfg = _cfg(require_dsr=True)
    chrom = _chrom()
    kept, rejections, records, _ = _screen(cfg, _nt_fn(), True, [(chrom, _fit())], 10)
    assert kept == []
    assert rejections[0]["reasons"] == [
        "DSR input missing: no usable daily return series (length 0)"
    ]
    assert records[chrom.uid]["missing"] is True

    non_finite = DailyReturns(19_000, array("d", [0.01, math.nan] + [0.01] * 28))
    chrom = _chrom(stop_loss_pct=3.0)
    _, rejections, records, _ = _screen(
        cfg, _nt_fn(), True, [(chrom, _fit(mtm_daily_returns=non_finite))], 10
    )
    assert rejections[0]["reasons"] == [
        "DSR input missing: no usable daily return series (length 30, non-finite values)"
    ]
    assert records[chrom.uid]["missing"] is True


def test_dsr_missing_symbol_series() -> None:
    cfg = _cfg(require_dsr=True)
    fit = _fit(
        mtm_daily_returns=_series(2.5),
        symbol_mtm_daily={"A": _series(2.5, seed=2), "B": None},
    )
    _, rejections, _, _ = _screen(cfg, _nt_fn(), True, [(_chrom(), fit)], 10)
    assert rejections[0]["reasons"] == ["DSR input missing: no daily return series for B"]


def test_requires_returns_flag_controls_input() -> None:
    cfg = _cfg(require_dsr=True)
    chrom = _chrom()
    kept, rejections, _, _ = _screen(cfg, _plain, True, [(chrom, _fit())], 10)
    assert kept == []
    assert rejections[0]["reasons"] == [
        "DSR input missing: no usable daily return series (length 0)"
    ]

    kept, rejections, records, _ = _screen(cfg, _plain, False, [(chrom, _fit())], 10)
    assert records[chrom.uid]["input"] == "fitness_sharpe"
    assert not any("DSR input missing" in r for row in rejections for r in row["reasons"])


# ---------------------------------------------------------------------------
# R16-R22: DSR inputs, trial counts, record shape
# ---------------------------------------------------------------------------


def test_dsr_legacy_fitness_input() -> None:
    cfg = _cfg(require_dsr=True)
    chrom = _chrom()
    kept, rejections, records, _ = _screen(cfg, _plain, False, [(chrom, _fit(sharpe_ratio=0.1))], 10)

    record = records[chrom.uid]
    assert list(record) == ["input", "trials", "observations", "p_value", "significant"]
    assert record["input"] == "fitness_sharpe"
    assert record["trials"] == 10
    assert record["observations"] == 366
    assert kept == []
    reason = rejections[0]["reasons"][0]
    assert reason.startswith("DSR not significant: p=") and reason.endswith(" >= 0.05")


def test_no_dsr_records_nothing() -> None:
    cfg = _cfg(require_dsr=False)
    chrom = _chrom()
    kept, rejections, records, trials = _screen(
        cfg, _nt_fn(), True, [(chrom, _fit(sharpe_ratio=0.1))], 10
    )
    assert kept == [chrom.uid] and rejections == []
    assert trials is None and records == {}


@pytest.mark.parametrize(
    ("prior_value", "prior_n", "gate_n", "prior_error"),
    [(1000, 1000, 1010, False), (None, None, 10, True), (-5, 0, 10, False)],
    ids=["returns-1000", "returns-none", "returns-negative"],
)
def test_gate_n_prior_rows(
    prior_value: int | None, prior_n: int | None, gate_n: int, prior_error: bool
) -> None:
    cfg = _cfg(require_dsr=True, prior_trials_fn=lambda _s, _e: prior_value)
    chrom = _chrom()
    _, _, records, trials = _screen(cfg, _plain, False, [(chrom, _fit())], 10)
    assert trials is not None
    assert records[chrom.uid]["trials"] == gate_n
    assert trials["gate_n"] == gate_n
    assert trials["prior_n"] == prior_n
    assert ("prior_error" in trials) is prior_error


def test_gate_n_prior_raises() -> None:
    def boom(_s: str, _e: str) -> int:
        raise RuntimeError("db down")

    cfg = _cfg(require_dsr=True, prior_trials_fn=boom)
    chrom = _chrom()
    _, _, records, trials = _screen(cfg, _plain, False, [(chrom, _fit())], 10)
    assert trials is not None
    assert records[chrom.uid]["trials"] == 10
    assert trials["gate_n"] == 10
    assert trials["prior_n"] is None
    assert trials["prior_error"] is True


def test_pool_order_invariant() -> None:
    cfg = _cfg(require_dsr=True)
    series = _series(2.5)
    items = [(str(i), _fit(mtm_daily_returns=series)) for i in range(3)]

    _, _, _, forward = _screen(
        cfg, _plain, False, [(_chrom(), _fit())], 40, dict(items)
    )
    _, _, _, backward = _screen(
        cfg, _plain, False, [(_chrom(), _fit())], 40, dict(reversed(items))
    )
    assert forward == backward
    assert forward is not None
    assert forward["raw"] == 40
    assert forward["run_n_eff"] == 14
    assert forward["gate_n"] == 14


def test_bootstrap_floor() -> None:
    cfg = _cfg(require_bootstrap_ci=True)
    _, rejections, _, _ = _screen(
        cfg, _nt_fn(), False, [(_chrom(), _fit(trade_returns=(0.01,) * 4))], 10
    )
    assert rejections[0]["reasons"] == [
        "Bootstrap CI required but insufficient trade returns (got 4, need ≥5)"
    ]

    _, rejections, _, _ = _screen(cfg, _nt_fn(), False, [(_chrom(), _fit())], 10)
    assert rejections[0]["reasons"] == [
        "Bootstrap CI required but insufficient trade returns (got 0, need ≥5)"
    ]


def test_rejection_record_shape() -> None:
    cfg = _cfg()
    chrom = _chrom()
    fit = _fit(adjusted_score=0.123456, sharpe_ratio=1.23456)

    bare = _record(cfg, _nt_fn(), chrom, fit, "guardrails", ["r"])
    assert list(bare) == [
        "uid", "stage", "score", "sharpe", "trades", "reasons", "dsl",
        "fee_drag", "fees_pct_equity_yr",
    ]  # fmt: skip
    assert bare["score"] == 0.1235 and bare["sharpe"] == 1.235
    assert bare["dsl"]["timeframe"] == "1d"  # type: ignore[index]
    assert "fee_drag" in bare and bare["fee_drag"] is None

    full = _record(
        cfg, _nt_fn(), chrom, fit, "guardrails", ["r"],
        dsr={"input": "x"}, buy_hold={"enforced": True},
    )  # fmt: skip
    assert list(full) == [
        "uid", "stage", "score", "sharpe", "trades", "reasons", "dsl", "dsr", "buy_hold",
        "fee_drag", "fees_pct_equity_yr",
    ]  # fmt: skip

    liquidated = _record(
        cfg, _nt_fn(), chrom, _fit(liquidation_count=3, liquidation_forfeited=0.5),
        "guardrails", ["r"],
    )  # fmt: skip
    assert list(liquidated)[-2:] == ["liquidation_count", "liquidation_forfeited"]
    assert liquidated["liquidation_count"] == 3
    assert liquidated["liquidation_forfeited"] == 0.5


def test_empty_candidates_still_record_trials() -> None:
    cfg = _cfg(require_dsr=True)
    kept, rejections, records, trials = _screen(cfg, _plain, False, [], 7)
    assert kept == [] and rejections == [] and records == {}
    assert trials is not None
    assert trials["gate_n"] == 7
    assert trials["raw"] == 7


# ---------------------------------------------------------------------------
# R23-R25: the pooled multi-seed re-check
# ---------------------------------------------------------------------------


def test_pooled_rewrite(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(require_dsr=True, top_k=2)
    weak = _series(0.2, seed=5)
    worst_chrom = _chrom(stop_loss_pct=2.0)
    dsr_chrom = _chrom(stop_loss_pct=3.0)
    worst = (worst_chrom, _fit(sharpe_ratio=1.0, total_trades=60, symbol_stats={"A": (0.0, 60)}))
    dsr_fail = (dsr_chrom, _fit(sharpe_ratio=0.5, mtm_daily_returns=weak))
    seeds = [
        _SeedOut(worst, evaluated=100),
        _SeedOut(
            dsr_fail,
            evaluated=100,
            cache={
                dsr_chrom.uid: _fit(mtm_daily_returns=weak),
                "extra-1": _fit(mtm_daily_returns=weak),
                "extra-2": _fit(mtm_daily_returns=weak),
            },
        ),
    ]

    merged = _pooled(monkeypatch, cfg, _nt_fn(), seeds)

    assert merged.dsr_trials is not None
    assert merged.dsr_trials["raw"] == 200
    assert merged.dsr_trials["gate_n"] == 67
    assert [r["stage"] for r in merged.guardrail_rejections] == [
        "pooled_guardrails",
        "pooled_guardrails",
    ]
    assert merged.guardrail_rejections[0]["reasons"] == [
        "Multi-seed pooled N=200: worst-symbol train return <= 0 (A)"
    ]
    dsr_reasons = merged.guardrail_rejections[1]["reasons"]
    assert len(dsr_reasons) == 1
    assert dsr_reasons[0] == (  # legacy spread -> the legacy rule decided (xw2en)
        "Multi-seed pooled N=200: DSR not significant: p=0.9836 >= 0.05 "
        "(pooled daily SR=0.200 ann, T=400, N=67)"
    )
    assert merged.top_strategies == []


def _boom(*_args: Any, **_kw: Any) -> Any:
    raise AssertionError("the pooled re-check must not run buy-and-hold")


def test_pooled_keeps_own_buy_hold_record(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(require_dsr=True, top_k=2)
    strong = _series(8.0, seed=7)
    a_chrom, b_chrom = _chrom(stop_loss_pct=2.0), _chrom(stop_loss_pct=3.0)
    a_fit = _fit(sharpe_ratio=2.0, mtm_daily_returns=strong)
    b_fit = _fit(sharpe_ratio=3.0, mtm_daily_returns=strong)
    own_a: dict[str, object] = {"enforced": True, "source": "seed-0"}
    own_b: dict[str, object] = {"enforced": True, "source": "seed-1"}
    seeds = [
        _SeedOut((a_chrom, a_fit), cache={a_chrom.uid: a_fit}, buy_hold={a_chrom.uid: own_a}),
        _SeedOut((b_chrom, b_fit), cache={b_chrom.uid: b_fit}, buy_hold={b_chrom.uid: own_b}),
    ]
    monkeypatch.setattr("vibe_quant.discovery.buyhold.load_daily_closes", _boom)

    merged = _pooled(monkeypatch, cfg, _nt_fn(), seeds)

    assert [chrom.uid for chrom, _ in merged.top_strategies] == [b_chrom.uid, a_chrom.uid]
    assert merged.buy_hold_records == {a_chrom.uid: own_a, b_chrom.uid: own_b}
    assert not any(r["stage"] == "buy_hold" for r in merged.guardrail_rejections)


def _bh_closes() -> dict[str, dict[int, float]]:
    prices = 100 * np.cumprod(1 + np.random.default_rng(5).normal(0.002, 0.02, 402))
    return {"BTCUSDT": {18_998 + k: float(v) for k, v in enumerate(prices)}}


def test_buy_hold_verdict_independent_of_trial_n() -> None:
    closes = _bh_closes()
    chrom = _chrom()
    fit = _fit(sharpe_ratio=2.0, mtm_daily_returns=_series(3.0, seed=7))

    def at_n(
        n: int | None,
    ) -> tuple[int | None, object, dict[str, object] | None, dict[str, object], int]:
        gates = GateChain(
            _cfg(require_dsr=True),
            GateEvidence(_nt_fn(), True, buy_hold_closes_fn=lambda _symbols: closes),
        )
        kept = None if n is None else gates.screen([(chrom, fit)], TrialBudget(n, {chrom.uid: fit}))
        kept_bh = gates.apply_buy_hold([(chrom, fit)])
        trials = gates.dsr_trials
        return (
            None if kept is None else len(kept),
            None if trials is None else trials["gate_n"],
            gates.buy_hold_gate,
            gates.buy_hold_records[chrom.uid],
            len(kept_bh),
        )

    n1, nmax, unscreened = at_n(1), at_n(1_000_000), at_n(None)

    # The DSR gate DOES read N: 1 trial passes, 1e6 rejects (flip proves the fixture has teeth).
    assert [n1[0], nmax[0]] == [1, 0]
    assert [n1[1], nmax[1]] == [1, 1_000_000]

    # Buy-and-hold takes no trial count, so every chain yields the same champion verdict
    # AND keeps the same champion (a B&H reject at 1e6 must fail this test, not just
    # change the record).
    records = [n1[3], nmax[3], unscreened[3]]
    assert records[0] == records[1] == records[2]
    assert records[0]["passed"] is True
    assert records[0]["sharpe_ann"] == 3.6105005588857453
    assert records[0]["bar_ann"] == 1.1155295831428362
    assert [n1[4], nmax[4], unscreened[4]] == [1, 1, 1]
    assert [gate for _, _, gate, _, _ in (n1, nmax, unscreened)] == [{"enforced": True}] * 3


def test_pooled_buy_hold_record_equals_own_run_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closes = _bh_closes()
    cfg = _cfg(require_dsr=True, top_k=2)
    fn = _nt_fn()
    a_chrom, b_chrom = _chrom(stop_loss_pct=2.0), _chrom(stop_loss_pct=3.0)
    strong = _series(8.0, seed=7)
    a_fit = _fit(sharpe_ratio=2.0, mtm_daily_returns=strong)
    b_fit = _fit(sharpe_ratio=3.0, mtm_daily_returns=strong)

    def own(chrom: StrategyChromosome, fitness: FitnessResult) -> dict[str, object]:
        gates = GateChain(
            cfg, GateEvidence(fn, True, buy_hold_closes_fn=lambda _symbols: closes)
        )
        gates.apply_buy_hold([(chrom, fitness)])
        return gates.buy_hold_records[chrom.uid]

    own_a, own_b = own(a_chrom, a_fit), own(b_chrom, b_fit)
    seeds = [
        _SeedOut(
            (a_chrom, a_fit), cache={a_chrom.uid: a_fit}, buy_hold={a_chrom.uid: own_a}
        ),
        _SeedOut(
            (b_chrom, b_fit), cache={b_chrom.uid: b_fit}, buy_hold={b_chrom.uid: own_b}
        ),
    ]
    calls: list[tuple[object, ...]] = []

    def spy(*args: Any, **_kw: Any) -> Any:
        calls.append(args)
        raise AssertionError("the pooled re-check must not run buy-and-hold")

    monkeypatch.setattr("vibe_quant.discovery.buyhold.load_daily_closes", spy)

    merged = _pooled(monkeypatch, cfg, fn, seeds)

    # apply_buy_hold catches EVERY load failure and pooled_recheck re-stages the
    # rejection as pooled_guardrails, so an exception sentinel cannot prove the
    # callee never ran: count the calls instead.
    assert calls == []
    assert merged.dsr_trials is not None
    assert merged.dsr_trials["raw"] == 200  # pooled N of the multi-seed re-check

    assert {chrom.uid for chrom, _ in merged.top_strategies} == {a_chrom.uid, b_chrom.uid}
    for chrom, fit in merged.top_strategies:
        assert merged.buy_hold_records[chrom.uid] == own(chrom, fit)
    assert merged.buy_hold_records == {a_chrom.uid: own_a, b_chrom.uid: own_b}
    assert not any(r["stage"] == "buy_hold" for r in merged.guardrail_rejections)


def test_pooled_requires_returns_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    missing = "Multi-seed pooled N=100: DSR input missing: no usable daily return series (length 0)"

    cfg = _cfg(require_dsr=True, top_k=1)
    chrom = _chrom()
    nt_merged = _pooled(
        monkeypatch, cfg, _nt_fn(), [_SeedOut((chrom, _fit(sharpe_ratio=3.0)), evaluated=100)]
    )
    assert nt_merged.guardrail_rejections[0]["reasons"] == [missing]

    plain_merged = _pooled(
        monkeypatch, cfg, _plain, [_SeedOut((chrom, _fit(sharpe_ratio=0.1)), evaluated=100)]
    )
    record = plain_merged.dsr_records[chrom.uid]
    assert record["input"] == "fitness_sharpe"
    assert record["trials"] == 100
    assert not any(
        "DSR input missing" in r for row in plain_merged.guardrail_rejections for r in row["reasons"]
    )

    no_dsr = _pooled(
        monkeypatch,
        _cfg(require_dsr=False, top_k=1),
        _nt_fn(),
        [_SeedOut((chrom, _fit()), evaluated=100)],
    )
    assert no_dsr.dsr_trials is None
    assert no_dsr.dsr_records == {}
    assert no_dsr.guardrail_rejections == []
    assert len(no_dsr.top_strategies) == 1
