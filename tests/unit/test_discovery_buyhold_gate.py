"""D2-3: the fail-closed buy-and-hold champion gate wired into discovery
(vibe-quant-wcn55).

The gate (D2-1, ``vibe_quant/discovery/buyhold.py``) compares a candidate's
mark-to-market daily Sharpe against the equal-weight daily-rebalanced
buy-and-hold of its symbols. D2-2 added ``FitnessResult.mtm_daily_returns`` /
``symbol_mtm_daily``. Here we prove the wiring: Gate 1b runs inside
``_apply_validation_gates`` BEFORE cross-window, skips only for a non-NT
backtest fn, fails closed on a closes-load error, and records telemetry.

``test_pooled_tp_sl_beta_fixture_passes_realized_fails_gate`` is the
pre-registered red fixture (design 06, task D2-3): a seeded long-only TP/SL beta
rule BEATS the buy-and-hold bar on the OLD realized series but the gate rejects
it on the mark-to-market series. Seed / params found by porting
``design-evidence/sim3.py`` and searching (seed 1372, tp 5%, sl 30%).
"""

from __future__ import annotations

import json
from array import array
from typing import TYPE_CHECKING, Any

import numpy as np

from vibe_quant.discovery import buyhold
from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.discovery.fitness import FitnessResult
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)
from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline, DiscoveryResult
from vibe_quant.metrics import (
    BarSeries,
    DailyReturns,
    LedgerEvent,
    dense_daily_returns,
    mark_to_market_daily_returns,
    pnl_path,
    pool_daily_returns,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from pytest import MonkeyPatch

_DAY = buyhold.DAY_NS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gene(ind: str, params: dict[str, float], cond: ConditionType = ConditionType.LT,
          thr: float = 30.0, sub: str | None = None) -> StrategyGene:
    return StrategyGene(ind, params, cond, thr, sub)


def _chrom(**kw: Any) -> StrategyChromosome:
    base: dict[str, Any] = {
        "entry_genes": [_gene("RSI", {"period": 14.0})],
        "exit_genes": [_gene("RSI", {"period": 14.0}, ConditionType.GT, 70.0)],
        "stop_loss_pct": 2.0,
        "take_profit_pct": 4.0,
        "direction": Direction.LONG,
    }
    base.update(kw)
    # inydc: a BOTH chromosome carries its own short-side genes
    if base["direction"] == Direction.BOTH:
        base.setdefault("short_entry_genes", [_gene("RSI", {"period": 14.0}, ConditionType.GT, 70.0)])
        base.setdefault("short_exit_genes", [_gene("RSI", {"period": 14.0}, ConditionType.LT, 30.0)])
    return StrategyChromosome(**base)


def _fit(**kw: Any) -> FitnessResult:
    base: dict[str, Any] = {
        "sharpe_ratio": 2.0, "max_drawdown": 0.1, "profit_factor": 1.8,
        "total_trades": 150, "total_return": 0.4, "complexity_penalty": 0.0,
        "overtrade_penalty": 0.0, "sl_tp_penalty": 0.0, "raw_score": 0.6,
        "adjusted_score": 0.6, "passed_filters": True, "filter_results": {},
    }
    base.update(kw)
    return FitnessResult(**base)


def _cfg(**kw: Any) -> DiscoveryConfig:
    base: dict[str, Any] = {
        "population_size": 6, "max_generations": 2, "elite_count": 1, "top_k": 5,
        "min_trades": 50, "max_workers": None, "timeframe": "1d",
        "start_date": "2024-01-01", "end_date": "2025-01-01",
        "require_bootstrap_ci": False, "require_dsr": False,
    }
    base.update(kw)
    return DiscoveryConfig(**base)


def _nt(symbols: Sequence[str], agg: str = "portfolio") -> NTBacktestFn:
    return NTBacktestFn(list(symbols), "1d", "2024-01-01", "2025-01-01", symbol_agg=agg)


def _gates(pipe: DiscoveryPipeline, top: list[tuple[StrategyChromosome, FitnessResult]]) -> Any:
    return pipe._apply_validation_gates(
        top, generations=[], total_evaluated=10, converged=False, convergence_gen=None,
    )


def _flat_trending_prices(symbols: Sequence[str], first: int, n: int, seed: int) -> dict[str, list[float]]:
    """Strong positive-drift closes -> a positive buy-and-hold bar."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[float]] = {}
    for s in symbols:
        r = rng.normal(0.010, 0.005, size=n + 1)
        out[s] = list(100.0 * np.cumprod(1.0 + r))
    return out


def _drift_closes(symbols: Sequence[str], mu: float, n: int = 60, seed: int = 7) -> dict[str, dict[int, float]]:
    """Daily closes with drift ``mu`` (positive -> B&H Sharpe > 0, negative -> < 0)."""
    rng = np.random.default_rng(seed)
    prices = {
        s: list(100.0 * np.cumprod(1.0 + rng.normal(mu, 0.005, size=n + 1))) for s in symbols
    }
    return _closes_from(prices, _FIX_BASE)


def _closes_from(prices: dict[str, list[float]], first: int) -> dict[str, dict[int, float]]:
    return {s: {first + k: float(p[k]) for k in range(len(p))} for s, p in prices.items()}


def _dr(first_day: int, values: Sequence[float]) -> DailyReturns:
    return DailyReturns(first_day, array("d", (float(v) for v in values)))


# ---------------------------------------------------------------------------
# Beta fixture: realized series passes, mark-to-market series fails
# ---------------------------------------------------------------------------

_FIX_SEED = 1372
_FIX_TP = 0.05
_FIX_SL = 0.3
_FIX_FEE = 0.0005
_FIX_SYMS = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
_FIX_BASE = 20000
_FIX_N = 400


def _fix_prices() -> list[np.ndarray]:
    rng = np.random.default_rng(_FIX_SEED * 7919)
    mus = rng.uniform(0.0005, 0.0040, size=3)
    sig = rng.uniform(0.010, 0.030, size=3)
    prng = np.random.default_rng(_FIX_SEED)
    return [
        100.0 * np.cumprod(1.0 + prng.normal(mu, sg, size=_FIX_N + 1))
        for mu, sg in zip(mus, sig, strict=True)
    ]


def _fix_sim(px: np.ndarray) -> list[tuple[int, float, int, float, float]]:
    """sim3's long-only TP/SL loop -> list of (entry_bar, entry_px, exit_bar, exit_px, qty)."""
    n = len(px) - 1
    eq = 1.0
    pos = False
    qty = ep = 0.0
    wait = 0
    ebar = 0
    trades: list[tuple[int, float, int, float, float]] = []
    for i in range(n):
        if not pos and wait == 0:
            qty = eq / px[i]
            ep = px[i]
            ebar = i
            pos = True
            eq -= _FIX_FEE * qty * px[i]
        elif not pos:
            wait -= 1
        if pos:
            ch = px[i + 1] / ep - 1
            if ch >= _FIX_TP or ch <= -_FIX_SL or i == n - 1:
                eq = eq + qty * (px[i + 1] - ep) - _FIX_FEE * qty * px[i + 1]
                trades.append((ebar, float(ep), i + 1, float(px[i + 1]), float(qty)))
                pos = False
                wait = 0
    return trades


def _fix_day(bar_i: int) -> int:
    return (_FIX_BASE + 1 + bar_i) * _DAY


def _beta_fixture() -> tuple[
    dict[str, dict[int, float]], DailyReturns, dict[str, DailyReturns], DailyReturns, float
]:
    """(closes, pooled MTM series, per-symbol MTM series, pooled realized series, bar)."""
    prices = _fix_prices()
    closes = {
        s: {_FIX_BASE + k: float(p[k]) for k in range(_FIX_N + 1)}
        for s, p in zip(_FIX_SYMS, prices, strict=True)
    }
    bh, err = buyhold.buy_hold_daily(closes, _FIX_SYMS, _FIX_BASE + 1, _FIX_N)
    assert err is None and bh is not None
    bar = buyhold.direction_bar(buyhold.daily_sharpe(bh), "long")

    sns, ens = _fix_day(0), _fix_day(_FIX_N)
    real_parts: list[DailyReturns | None] = []
    mtm_parts: list[DailyReturns | None] = []
    sym_mtm: dict[str, DailyReturns] = {}
    for s, p in zip(_FIX_SYMS, prices, strict=True):
        trades = _fix_sim(p)
        bars = BarSeries(
            ts=np.array([_fix_day(i) for i in range(_FIX_N)], dtype=np.int64),
            open=p[1:].copy(), high=p[1:].copy(), low=p[1:].copy(), close=p[1:].copy(),
        )
        events = []
        real_events = []
        for ebar, epx, xbar, xpx, q in trades:
            events.append(LedgerEvent(ts=_fix_day(ebar), qty=q, price=epx, cash=-_FIX_FEE * q * epx))
            events.append(LedgerEvent(ts=_fix_day(xbar), qty=-q, price=xpx, cash=-_FIX_FEE * q * xpx))
            real_events.append((_fix_day(ebar), -_FIX_FEE * q * epx))
            real_events.append((_fix_day(xbar), q * (xpx - epx) - _FIX_FEE * q * xpx))
        pc, pm = pnl_path(bars, events)
        real_parts.append(dense_daily_returns(1.0, real_events, sns, ens))
        m = mark_to_market_daily_returns(1.0, [(bars.ts, pc, pm)], sns, ens)
        assert m is not None
        mtm_parts.append(m)
        sym_mtm[s] = m

    mtm_pooled = pool_daily_returns(mtm_parts)
    real_pooled = pool_daily_returns(real_parts)
    assert mtm_pooled is not None and real_pooled is not None
    return closes, mtm_pooled, sym_mtm, real_pooled, bar


def test_pooled_tp_sl_beta_fixture_passes_realized_fails_gate() -> None:
    """The pre-registered red fixture: old series passes, MTM series is rejected."""
    closes, mtm_pooled, sym_mtm, realized_pooled, bar = _beta_fixture()

    # The OLD (realized) candidate series BEATS the bar -> pre-D2-3 it passes.
    realized_sharpe = buyhold.daily_sharpe(realized_pooled.values)
    mtm_sharpe = buyhold.daily_sharpe(mtm_pooled.values)
    assert realized_sharpe is not None and mtm_sharpe is not None
    assert realized_sharpe > bar
    assert mtm_sharpe <= bar

    # The gate, fed the mark-to-market series, rejects the same candidate.
    fit = _fit(mtm_daily_returns=mtm_pooled, symbol_mtm_daily=sym_mtm)
    chrom = _chrom()
    pipe = DiscoveryPipeline(
        _cfg(), backtest_fn=_nt(_FIX_SYMS), buy_hold_closes_fn=lambda syms: closes
    )
    res = _gates(pipe, [(chrom, fit)])
    assert res.top_strategies == []
    (rej,) = res.guardrail_rejections
    assert rej["stage"] == "buy_hold"
    assert any("Buy-hold:" in r for r in rej["reasons"])  # type: ignore[attr-defined]
    assert rej["buy_hold"]["passed"] is False  # type: ignore[index]


# ---------------------------------------------------------------------------
# Worst mode: a beta leg fails its own bar even when the pool passes
# ---------------------------------------------------------------------------


def test_worst_mode_beta_leg_rejected_despite_pooled_pass() -> None:
    symbols = ["AAAUSDT", "BBBUSDT"]
    prices = _flat_trending_prices(symbols, _FIX_BASE, 60, seed=7)
    closes = _closes_from(prices, _FIX_BASE)
    first = _FIX_BASE + 1
    n = 60

    pool = _dr(first, [0.030 + 0.001 * (i % 2) for i in range(n)])  # Sharpe ~ 30
    leg_a = _dr(first, [0.001 * (1 if i % 2 else -1) for i in range(n)])  # flat, Sharpe ~ 0
    leg_b = pool
    fit = _fit(mtm_daily_returns=pool, symbol_mtm_daily={"AAAUSDT": leg_a, "BBBUSDT": leg_b})
    cfg = _cfg(symbol_agg="worst")
    pipe = DiscoveryPipeline(cfg, backtest_fn=_nt(symbols, "worst"),
                             buy_hold_closes_fn=lambda syms: closes)
    res = _gates(pipe, [(_chrom(), fit)])

    assert res.top_strategies == []
    (rej,) = res.guardrail_rejections
    assert rej["stage"] == "buy_hold"
    reasons = rej["reasons"]  # type: ignore[assignment]
    assert any("AAAUSDT" in r for r in reasons)
    # The pooled side passed; only the leg failed.
    assert rej["buy_hold"]["passed"] is False  # type: ignore[index]
    assert "symbols" in rej["buy_hold"]  # type: ignore[operator]


def _worst_gate(fit: FitnessResult, symbols: Sequence[str], closes: dict[str, dict[int, float]]) -> Any:
    pipe = DiscoveryPipeline(_cfg(symbol_agg="worst"), backtest_fn=_nt(symbols, "worst"),
                             buy_hold_closes_fn=lambda syms: closes)
    return _gates(pipe, [(_chrom(), fit)])


def test_worst_mode_missing_symbol_map_rejects_despite_pooled_pass() -> None:
    symbols = ["AAAUSDT", "BBBUSDT"]
    closes = _closes_from(_flat_trending_prices(symbols, _FIX_BASE, 60, seed=7), _FIX_BASE)
    pool = _dr(_FIX_BASE + 1, [0.030 + 0.001 * (i % 2) for i in range(60)])  # Sharpe ~ 30

    # Control: same pooled series with a passing leg for every symbol survives.
    ok = _worst_gate(_fit(mtm_daily_returns=pool, symbol_mtm_daily={s: pool for s in symbols}),
                     symbols, closes)
    assert len(ok.top_strategies) == 1

    res = _worst_gate(_fit(mtm_daily_returns=pool, symbol_mtm_daily=None), symbols, closes)
    assert res.top_strategies == []
    (rej,) = res.guardrail_rejections
    assert rej["stage"] == "buy_hold"
    reasons = rej["reasons"]  # type: ignore[assignment]
    assert "Buy-hold: AAAUSDT has no daily series" in reasons
    assert "Buy-hold: BBBUSDT has no daily series" in reasons
    assert not any(r.startswith("Buy-hold: sharpe") for r in reasons)  # the pooled side passed


def test_worst_mode_missing_one_leg_rejects_despite_pooled_pass() -> None:
    symbols = ["AAAUSDT", "BBBUSDT"]
    closes = _closes_from(_flat_trending_prices(symbols, _FIX_BASE, 60, seed=7), _FIX_BASE)
    pool = _dr(_FIX_BASE + 1, [0.030 + 0.001 * (i % 2) for i in range(60)])

    res = _worst_gate(_fit(mtm_daily_returns=pool, symbol_mtm_daily={"BBBUSDT": pool}),
                      symbols, closes)
    assert res.top_strategies == []
    (rej,) = res.guardrail_rejections
    assert rej["reasons"] == ["Buy-hold: AAAUSDT has no daily series"]  # type: ignore[comparison-overlap]
    assert rej["buy_hold"]["symbols"]["BBBUSDT"]["passed"] is True  # type: ignore[index]
    assert rej["buy_hold"]["symbols"]["AAAUSDT"]["passed"] is False  # type: ignore[index]


# ---------------------------------------------------------------------------
# Direction-specific bars (long: bh, short: -bh, both: |bh|)
# ---------------------------------------------------------------------------

_FLAT = [0.001 * (1 if i % 2 else -1) for i in range(60)]  # Sharpe 0
_STRONG = [0.030 + 0.001 * (i % 2) for i in range(60)]  # Sharpe >> any bar


def _direction_gate(direction: Direction, mu: float, values: Sequence[float]) -> Any:
    closes = _drift_closes(["AAAUSDT"], mu)
    pipe = DiscoveryPipeline(
        _cfg(direction=direction.value), backtest_fn=_nt(["AAAUSDT"]),
        buy_hold_closes_fn=lambda syms: closes,
    )
    chrom = _chrom(direction=direction)
    res = _gates(pipe, [(chrom, _fit(mtm_daily_returns=_dr(_FIX_BASE + 1, values)))])
    return res, chrom


def test_short_candidate_faces_negated_bar() -> None:
    # Rising market (bh Sharpe > 0): a short has bar -bh < 0, so a flat series
    # (Sharpe 0) clears it. Forcing the long bar (+bh) would reject it.
    res, chrom = _direction_gate(Direction.SHORT, 0.010, _FLAT)
    assert [c.uid for c, _ in res.top_strategies] == [chrom.uid]
    record = res.buy_hold_records[chrom.uid]
    assert record["direction"] == "short"
    assert record["bar_daily"] == -record["bh_sharpe_daily"]  # type: ignore[operator]
    assert record["bh_sharpe_daily"] > 0  # type: ignore[operator]

    # Same short on the same market, but losing money: below -bh -> rejected.
    losing = [-0.030 + 0.001 * (i % 2) for i in range(60)]
    res, _ = _direction_gate(Direction.SHORT, 0.010, losing)
    assert res.top_strategies == []
    assert {r["stage"] for r in res.guardrail_rejections} == {"buy_hold"}


def test_both_candidate_faces_absolute_bar() -> None:
    # Falling market (bh Sharpe < 0): "both" faces |bh| > 0, a flat series
    # (Sharpe 0) must NOT clear it ...
    res, chrom = _direction_gate(Direction.BOTH, -0.010, _FLAT)
    assert res.top_strategies == []
    (rej,) = res.guardrail_rejections
    assert rej["buy_hold"]["direction"] == "both"  # type: ignore[index]
    assert rej["buy_hold"]["bh_sharpe_daily"] < 0  # type: ignore[index,operator]
    assert rej["buy_hold"]["bar_daily"] == -rej["buy_hold"]["bh_sharpe_daily"]  # type: ignore[index,operator]

    # ... while a strong series does ...
    res, chrom = _direction_gate(Direction.BOTH, -0.010, _STRONG)
    assert [c.uid for c, _ in res.top_strategies] == [chrom.uid]

    # ... and the same flat series passes as a LONG (bar = bh < 0): the bar differs by direction.
    res, chrom = _direction_gate(Direction.LONG, -0.010, _FLAT)
    assert [c.uid for c, _ in res.top_strategies] == [chrom.uid]


# ---------------------------------------------------------------------------
# Enforce / skip rules (fail closed)
# ---------------------------------------------------------------------------


def test_nt_fn_without_injected_closes_enforces(monkeypatch: MonkeyPatch) -> None:
    calls: list[list[str]] = []
    closes = _closes_from(_flat_trending_prices(_FIX_SYMS, _FIX_BASE, 60, seed=7), _FIX_BASE)

    def fake_load(symbols: Sequence[str], catalog: object) -> dict[str, dict[int, float]]:
        calls.append(list(symbols))
        return closes

    monkeypatch.setattr("vibe_quant.discovery.buyhold.load_daily_closes", fake_load)

    first = _FIX_BASE + 1
    n = 60
    flat = _dr(first, [0.001 * (1 if i % 2 else -1) for i in range(n)])  # Sharpe ~ 0 -> reject
    fit = _fit(mtm_daily_returns=flat)
    pipe = DiscoveryPipeline(_cfg(), backtest_fn=_nt(_FIX_SYMS))  # no injected closes fn
    res = _gates(pipe, [(_chrom(), fit)])

    assert calls == [_FIX_SYMS]  # loaded once, in the parent, from the catalog
    assert res.buy_hold_gate == {"enforced": True}
    assert res.top_strategies == []
    assert {r["stage"] for r in res.guardrail_rejections} == {"buy_hold"}


def test_closes_load_failure_rejects_all() -> None:
    def boom(symbols: Sequence[str]) -> dict[str, dict[int, float]]:
        msg = "corrupt catalog"
        raise ValueError(msg)

    pipe = DiscoveryPipeline(_cfg(), backtest_fn=_nt(_FIX_SYMS), buy_hold_closes_fn=boom)
    res = _gates(pipe, [(_chrom(stop_loss_pct=1.0), _fit()), (_chrom(stop_loss_pct=2.0), _fit())])

    assert res.top_strategies == []
    assert len(res.guardrail_rejections) == 2
    for rej in res.guardrail_rejections:
        assert rej["stage"] == "buy_hold"
        assert rej["reasons"][0].startswith("Buy-hold: benchmark closes unavailable: corrupt catalog")  # type: ignore[index]


def test_non_nt_fn_records_not_enforced() -> None:
    pipe = DiscoveryPipeline(_cfg(), backtest_fn=lambda c: {})  # non-NT
    top = [(_chrom(), _fit())]
    res = _gates(pipe, top)

    assert res.top_strategies == top
    assert res.guardrail_rejections == []
    assert res.buy_hold_gate == {"enforced": False, "reason": "non-NT backtest fn"}
    assert res.buy_hold_records == {}


# ---------------------------------------------------------------------------
# Placement, flag, series collection
# ---------------------------------------------------------------------------


def test_buy_hold_runs_before_cross_window() -> None:
    made: list[tuple[str, str]] = []

    def factory(s: str, e: str) -> Any:
        made.append((s, e))
        return lambda c: {}

    first = _FIX_BASE + 1
    flat = _dr(first, [0.001 * (1 if i % 2 else -1) for i in range(60)])
    closes = _closes_from(_flat_trending_prices(_FIX_SYMS, _FIX_BASE, 60, seed=7), _FIX_BASE)
    pipe = DiscoveryPipeline(
        _cfg(cross_window_months=[1]), backtest_fn=_nt(_FIX_SYMS),
        backtest_fn_factory=factory, buy_hold_closes_fn=lambda syms: closes,
    )
    res = _gates(pipe, [(_chrom(), _fit(mtm_daily_returns=flat))])

    assert res.top_strategies == []
    assert {r["stage"] for r in res.guardrail_rejections} == {"buy_hold"}
    assert made == []  # cross-window never reached the rejected candidate


def test_no_buy_hold_gate_flag_disables() -> None:
    first = _FIX_BASE + 1
    flat = _dr(first, [0.001 * (1 if i % 2 else -1) for i in range(60)])
    chrom, fit = _chrom(), _fit(mtm_daily_returns=flat)
    pipe = DiscoveryPipeline(
        _cfg(require_buy_hold=False), backtest_fn=_nt(_FIX_SYMS),
        buy_hold_closes_fn=lambda syms: {},
    )
    res = _gates(pipe, [(chrom, fit)])

    assert [c.uid for c, _ in res.top_strategies] == [chrom.uid]
    assert res.buy_hold_records == {}
    assert res.buy_hold_gate is None


def test_daily_returns_collected_when_dsr_off() -> None:
    # buy_hold on, dsr off -> collect series, but DSR's own fail-closed flag stays off
    pipe = DiscoveryPipeline(_cfg(require_dsr=False, require_buy_hold=True), backtest_fn=_nt(["A"]))
    assert isinstance(pipe._backtest_fn, NTBacktestFn)
    assert pipe._backtest_fn.collect_daily_returns is True
    assert pipe._dsr_requires_returns is False

    # both off -> nothing collected
    pipe2 = DiscoveryPipeline(_cfg(require_dsr=False, require_buy_hold=False), backtest_fn=_nt(["A"]))
    assert pipe2._backtest_fn.collect_daily_returns is False

    # dsr on -> collect + DSR flag
    pipe3 = DiscoveryPipeline(_cfg(require_dsr=True, require_buy_hold=False), backtest_fn=_nt(["A"]))
    assert pipe3._backtest_fn.collect_daily_returns is True
    assert pipe3._dsr_requires_returns is True


# ---------------------------------------------------------------------------
# Records: champion entry, notes, multi-seed union, DSL persistence
# ---------------------------------------------------------------------------


def test_champion_entry_and_notes_have_buy_hold() -> None:
    from vibe_quant.discovery.record import provenance_notes

    first = _FIX_BASE + 1
    n = 60
    strong = _dr(first, [0.030 + 0.001 * (i % 2) for i in range(n)])  # Sharpe >> bar
    closes = _closes_from(_flat_trending_prices(["AAAUSDT"], _FIX_BASE, n, seed=7), _FIX_BASE)
    chrom = _chrom()
    fit = _fit(mtm_daily_returns=strong)
    pipe = DiscoveryPipeline(
        _cfg(), backtest_fn=_nt(["AAAUSDT"]), buy_hold_closes_fn=lambda syms: closes
    )
    res = _gates(pipe, [(chrom, fit)])

    assert [c.uid for c, _ in res.top_strategies] == [chrom.uid]
    record = res.buy_hold_records[chrom.uid]
    assert record["passed"] is True
    assert record["series"] == "mtm"

    notes = provenance_notes(
        res, use_mock=False, eval_windows_count=1, eval_windows=None, symbol_agg="portfolio",
        split_ratio=0.0, direction=None, cross_window_months=[], cross_window_min_sharpe=0.5,
        wfa_oos_step_days=0, wfa_min_consistency=0.75, num_seeds=1, seed=42,
        bootstrap_min_sharpe=0.0, holdout_min_sharpe=0.0, symbols=["AAAUSDT"],
    )
    gate = notes["buy_hold_gate"]
    assert set(gate) == {  # type: ignore[arg-type]
        "enforced", "benchmark", "candidate_series", "symbols", "ann_factor", "per_symbol",
    }
    assert gate["enforced"] is True  # type: ignore[index]
    assert gate["benchmark"] == "equal_weight_daily_rebalanced_1d_close"  # type: ignore[index]
    assert gate["candidate_series"] == "mark_to_market_daily"  # type: ignore[index]
    assert gate["ann_factor"] == 365  # type: ignore[index]
    assert gate["per_symbol"] is False  # type: ignore[index]


def test_multi_seed_merges_records(monkeypatch: MonkeyPatch, tmp_path: Any) -> None:
    from vibe_quant.discovery.pooling import _run_multi_seed

    chrom_a, chrom_b = _chrom(stop_loss_pct=1.0), _chrom(stop_loss_pct=9.0)
    fit_a, fit_b = _fit(adjusted_score=0.9), _fit(adjusted_score=0.8)
    rec_a = {chrom_a.uid: {"series": "mtm", "passed": True}}
    rec_b = {chrom_b.uid: {"series": "mtm", "passed": False}}
    canned = [
        DiscoveryResult([], [(chrom_a, fit_a)], 1, False, None, buy_hold_records=rec_a,
                        buy_hold_gate={"enforced": True}),
        DiscoveryResult([], [(chrom_b, fit_b)], 1, False, None, buy_hold_records=rec_b,
                        buy_hold_gate={"enforced": True}),
    ]
    monkeypatch.setattr(DiscoveryPipeline, "run", lambda self: canned.pop(0))

    cfg = _cfg(require_dsr=False, require_bootstrap_ci=False)
    merged = _run_multi_seed(
        num_seeds=2, config=cfg, backtest_fn=lambda c: {}, progress_file=str(tmp_path / "p.json")
    )
    assert set(merged.buy_hold_records) == {chrom_a.uid, chrom_b.uid}
    assert merged.buy_hold_gate == {"enforced": True}


def test_rejected_buy_hold_candidate_keeps_dsl() -> None:
    first = _FIX_BASE + 1
    flat = _dr(first, [0.001 * (1 if i % 2 else -1) for i in range(60)])
    closes = _closes_from(_flat_trending_prices(["AAAUSDT"], _FIX_BASE, 60, seed=7), _FIX_BASE)
    chrom = _chrom(stop_loss_pct=3.5)
    pipe = DiscoveryPipeline(
        _cfg(timeframe="1h"), backtest_fn=_nt(["AAAUSDT"], "portfolio"),
        buy_hold_closes_fn=lambda syms: closes,
    )
    res = _gates(pipe, [(chrom, _fit(mtm_daily_returns=flat))])

    (rej,) = res.guardrail_rejections
    assert rej["stage"] == "buy_hold"
    assert rej["dsl"]["timeframe"] == "1h"  # type: ignore[index]
    assert rej["dsl"]["stop_loss"] == {"type": "fixed_pct", "percent": 3.5}  # type: ignore[index]
    assert rej["buy_hold"]["series"] == "mtm"  # type: ignore[index]


# ---------------------------------------------------------------------------
# CLI: default / opt-out reach the config; champion records are persisted
# ---------------------------------------------------------------------------


def test_cli_buy_hold_gate_defaults_on_with_opt_out() -> None:
    from vibe_quant.discovery.run_spec import build_parser

    assert build_parser().parse_args(["--run-id", "1"]).require_buy_hold is True
    args = build_parser().parse_args(["--run-id", "1", "--no-buy-hold-gate"])
    assert args.require_buy_hold is False


def _run_main(
    tmp_path: Path, monkeypatch: MonkeyPatch, extra: list[str], canned: DiscoveryResult
) -> tuple[dict[str, Any], list[DiscoveryConfig]]:
    """Run the discovery CLI (mock) with the pipeline canned; return (notes, configs)."""
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery import __main__ as cli

    db = tmp_path / "s.db"
    sm = StateManager(db)
    run_id = sm.create_backtest_run(None, "discovery", ["BTCUSDT"], "1d",
                                    "2024-01-01", "2024-07-01", {})
    sm.close()

    configs: list[DiscoveryConfig] = []
    real_init = DiscoveryPipeline.__init__

    def init(self: DiscoveryPipeline, *args: Any, **kwargs: Any) -> None:
        real_init(self, *args, **kwargs)
        configs.append(self.config)

    monkeypatch.setattr(DiscoveryPipeline, "__init__", init)
    monkeypatch.setattr(DiscoveryPipeline, "run", lambda self: canned)
    monkeypatch.setattr("sys.argv", [
        "prog", "--run-id", str(run_id), "--population-size", "6", "--max-generations", "2",
        "--elite-count", "1", "--max-workers", "-1", "--symbols", "BTCUSDT", "--timeframe", "1d",
        "--start-date", "2024-01-01", "--end-date", "2024-07-01", "--no-dsr", "--db", str(db),
        "--mock", *extra,
    ])
    assert cli.main() == 0
    sm = StateManager(db)
    row = sm.get_backtest_result(run_id)
    sm.close()
    assert row is not None
    return json.loads(row["notes"]), configs


def test_main_persists_champion_buy_hold_record_and_gate_notes(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    kept, dropped = _chrom(stop_loss_pct=1.0), _chrom(stop_loss_pct=9.0)
    record: dict[str, object] = {
        "series": "mtm", "sharpe_daily": 1.5, "bh_sharpe_daily": 0.5, "bar_daily": 0.5,
        "direction": "long", "observations": 60, "first_day": _FIX_BASE + 1, "passed": True,
    }
    canned = DiscoveryResult(
        [], [(kept, _fit())], 1, False, None,
        buy_hold_records={kept.uid: record, dropped.uid: {"series": "mtm", "passed": False}},
        buy_hold_gate={"enforced": True},
    )
    notes, _ = _run_main(tmp_path, monkeypatch, [], canned)

    (top,) = notes["top_strategies"]
    assert top["buy_hold"] == record  # the champion's own record, not the rejected one's
    assert notes["buy_hold_gate"] == {
        "enforced": True,
        "benchmark": "equal_weight_daily_rebalanced_1d_close",
        "candidate_series": "mark_to_market_daily",
        "symbols": ["BTCUSDT"],
        "ann_factor": 365,
        "per_symbol": False,
    }
