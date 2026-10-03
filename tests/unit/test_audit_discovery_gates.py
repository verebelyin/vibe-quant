"""Audit 2026-10-02 regression tests: discovery GA + champion gates
(vibe-quant-e70tl.1, .5, .6 and the discovery MEDIUM items).

Expected values are hand-computed; GA randomness is pinned with random.seed
where a test depends on it.
"""

from __future__ import annotations

import json
import random
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.discovery.fitness import (
    PF_MAX,
    FitnessResult,
    _evaluate_single,
    compute_fitness_score,
    compute_sl_tp_penalty,
)
from vibe_quant.discovery.genome import chromosome_to_dsl
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
    _random_chromosome,
    canonicalize_gene,
    crossover,
    initialize_population,
    mutate,
)
from vibe_quant.discovery.pipeline import (
    DiscoveryConfig,
    DiscoveryPipeline,
    assert_windows_outside_holdout,
)

if TYPE_CHECKING:
    from pathlib import Path


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
    return StrategyChromosome(**base)


def _fit(sharpe: float = 2.0, trades: int = 150, ret: float = 0.4, score: float = 0.6,
         trade_returns: tuple[float, ...] = ()) -> FitnessResult:
    return FitnessResult(
        sharpe_ratio=sharpe, max_drawdown=0.1, profit_factor=1.8, total_trades=trades,
        total_return=ret, complexity_penalty=0.0, overtrade_penalty=0.0, sl_tp_penalty=0.0,
        raw_score=score, adjusted_score=score, passed_filters=True, filter_results={},
        trade_returns=trade_returns,
    )


def _cfg(**kw: Any) -> DiscoveryConfig:
    base: dict[str, Any] = {
        "population_size": 6, "max_generations": 2, "elite_count": 1, "top_k": 5,
        "min_trades": 50, "max_workers": None, "timeframe": "4h",
        "start_date": "2024-01-01", "end_date": "2024-06-01",
        "require_bootstrap_ci": False, "require_dsr": False,
    }
    base.update(kw)
    return DiscoveryConfig(**base)


def _bt(sharpe: float = 2.0, ret: float = 0.3, trades: int = 120) -> dict[str, Any]:
    return {"sharpe_ratio": sharpe, "max_drawdown": 0.1, "profit_factor": 1.8,
            "total_trades": trades, "total_return": ret}


def _top(n: int) -> list[tuple[StrategyChromosome, FitnessResult]]:
    return [(_chrom(stop_loss_pct=1.0 + i), _fit(score=0.9 - i * 0.1)) for i in range(n)]


# ---------------------------------------------------------------------------
# e70tl.5 — gates fail closed
# ---------------------------------------------------------------------------


def test_all_top_k_fail_dsr_persists_zero_champions_with_reasons() -> None:
    """The old soft-fail path returned None -> 'keep unfiltered' (runs 854-867)."""
    pipe = DiscoveryPipeline(_cfg(require_dsr=True), backtest_fn=lambda c: _bt())
    top = [(_chrom(), _fit(sharpe=1.6, trades=150)) for _ in range(5)]
    out = pipe._validate_top_strategies(top, total_evaluated=8000)
    assert out == []
    assert len(pipe._guardrail_rejections) == 5
    for rej in pipe._guardrail_rejections:
        assert rej["stage"] == "guardrails"
        assert any("DSR not significant" in r for r in rej["reasons"])  # type: ignore[attr-defined]


def test_one_of_five_passes_only_that_one_survives() -> None:
    pipe = DiscoveryPipeline(_cfg(), backtest_fn=lambda c: _bt())
    top = [(_chrom(), _fit(trades=10)) for _ in range(4)] + [(_chrom(), _fit(trades=150))]
    out = pipe._validate_top_strategies(top, total_evaluated=100)
    assert [f.total_trades for _, f in out] == [150]
    assert len(pipe._guardrail_rejections) == 4


def _gates(pipe: DiscoveryPipeline, top: list[tuple[StrategyChromosome, FitnessResult]]) -> Any:
    return pipe._apply_validation_gates(
        top, generations=[], total_evaluated=10, converged=False, convergence_gen=None,
    )


def test_all_fail_cross_window_gives_zero_champions() -> None:
    pipe = DiscoveryPipeline(
        _cfg(cross_window_months=[1]), backtest_fn=lambda c: _bt(),
        backtest_fn_factory=lambda s, e: (lambda c: _bt(sharpe=-1.0, ret=-0.1)),
    )
    res = _gates(pipe, _top(3))
    assert res.top_strategies == []
    assert {r["stage"] for r in res.guardrail_rejections} == {"cross_window"}


def test_all_fail_wfa_gives_zero_champions() -> None:
    pipe = DiscoveryPipeline(
        _cfg(wfa_oos_step_days=30), backtest_fn=lambda c: _bt(),
        backtest_fn_factory=lambda s, e: (lambda c: _bt(ret=-0.01)),
    )
    res = _gates(pipe, _top(3))
    assert res.top_strategies == []
    assert {r["stage"] for r in res.guardrail_rejections} == {"wfa"}


def test_partial_cross_window_keeps_only_passers_with_aligned_results() -> None:
    good_sl = 2.0

    def factory(s: str, e: str) -> Any:
        return lambda c: _bt() if c.stop_loss_pct == good_sl else _bt(ret=-0.1)

    pipe = DiscoveryPipeline(_cfg(cross_window_months=[1]), backtest_fn=lambda c: _bt(),
                             backtest_fn_factory=factory)
    res = _gates(pipe, _top(3))  # SL 1.0, 2.0, 3.0
    assert [c.stop_loss_pct for c, _ in res.top_strategies] == [2.0]
    assert len(res.cross_window_results) == 1


def test_cross_window_in_sample_window_never_counts() -> None:
    """Audit P10: W0 (training) great, +3mo barely ok, +6mo big loss -> REJECT."""
    metrics = {"2024-04-01": (0.55, 0.01), "2024-07-01": (-1.2, -0.15)}
    pipe = DiscoveryPipeline(
        _cfg(end_date="2025-06-01", cross_window_months=[3, 6]),
        backtest_fn=lambda c: _bt(),
        backtest_fn_factory=lambda ws, we: (
            lambda c: _bt(sharpe=metrics[ws][0], ret=metrics[ws][1])
        ),
    )
    results, error = pipe._evaluate_cross_windows([(_chrom(), _fit(sharpe=3.0))])
    assert error is None
    cwr = results[0]
    assert (cwr.windows_passed, cwr.total_windows, cwr.required) == (1, 2, 2)
    assert cwr.passed is False
    # windows are sub-windows of the TRAIN range (end clipped, no holdout reach)
    assert cwr.window_dates == [("2024-04-01", "2025-06-01"), ("2024-07-01", "2025-06-01")]


def test_cross_window_min_pass_counts_shifted_windows_only() -> None:
    metrics = {"2024-02-01": (1.0, 0.05), "2024-03-01": (-1.0, -0.05)}
    pipe = DiscoveryPipeline(
        _cfg(cross_window_months=[1, 2], cross_window_min_pass=1),
        backtest_fn=lambda c: _bt(),
        backtest_fn_factory=lambda ws, we: (
            lambda c: _bt(sharpe=metrics[ws][0], ret=metrics[ws][1])
        ),
    )
    results, _ = pipe._evaluate_cross_windows([(_chrom(), _fit())])
    assert results[0].passed is True and results[0].windows_passed == 1


def test_no_cross_window_or_wfa_window_overlaps_holdout() -> None:
    cfg = _cfg(
        start_date="2024-01-01", end_date="2024-10-19",
        train_test_split=0.8, holdout_start_date="2024-10-19", holdout_end_date="2025-01-01",
        cross_window_months=[1, 3], wfa_oos_step_days=30,
    )
    pipe = DiscoveryPipeline(cfg, backtest_fn=lambda c: _bt())
    for _, ws, we in pipe.cross_window_ranges():
        assert we <= cfg.holdout_start_date and ws < cfg.holdout_start_date
    wfa_windows = pipe.wfa_window_ranges(cfg.start_date, cfg.end_date)
    assert wfa_windows
    assert all(we <= cfg.holdout_start_date for _, we in wfa_windows)
    with pytest.raises(ValueError, match="overlaps the holdout"):
        assert_windows_outside_holdout([("2024-09-01", "2024-11-01")], "2024-10-19", "WFA")
    assert_windows_outside_holdout([("2024-09-19", "2024-10-19")], "2024-10-19", "WFA")


def _holdout_cfg(**kw: Any) -> DiscoveryConfig:
    return _cfg(
        start_date="2024-01-01", end_date="2024-10-19", train_test_split=0.8,
        holdout_start_date="2024-10-19", holdout_end_date="2025-01-01", **kw,
    )


def test_holdout_is_the_final_gate() -> None:
    def holdout(c: StrategyChromosome) -> dict[str, Any]:
        return _bt(ret=0.05, trades=30) if c.stop_loss_pct == 1.0 else _bt(ret=-0.02, trades=30)

    pipe = DiscoveryPipeline(_holdout_cfg(), backtest_fn=lambda c: _bt(),
                             holdout_backtest_fn=holdout)
    res = _gates(pipe, _top(3))
    assert [c.stop_loss_pct for c, _ in res.top_strategies] == [1.0]
    assert len(res.holdout_results) == 1 and res.holdout_results[0].total_return == 0.05
    assert res.holdout_dates == ("2024-10-19", "2025-01-01")
    reasons = [r["reasons"] for r in res.guardrail_rejections]
    assert all(r["stage"] == "holdout" for r in res.guardrail_rejections)
    assert all("Holdout: return" in reason[0] for reason in reasons)  # type: ignore[index]


def test_holdout_min_trades_is_half_the_train_rate() -> None:
    pipe = DiscoveryPipeline(_holdout_cfg(), backtest_fn=lambda c: _bt())
    # train 292 days, holdout 74 days: 50 * 74 / (2 * 292) = 6.33 -> 6
    assert pipe.holdout_min_trades() == 6
    few = DiscoveryPipeline(
        _holdout_cfg(), backtest_fn=lambda c: _bt(),
        holdout_backtest_fn=lambda c: _bt(ret=0.05, trades=5),
    )
    res = _gates(few, _top(1))
    assert res.top_strategies == []
    assert res.guardrail_rejections[0]["reasons"] == ["Holdout: 5 trades < 6"]


def test_holdout_used_only_for_survivors_not_ranking() -> None:
    """Candidates rejected earlier are never run on the holdout."""
    seen: list[float] = []

    def holdout(c: StrategyChromosome) -> dict[str, Any]:
        seen.append(c.stop_loss_pct)
        return _bt(ret=0.05, trades=30)

    pipe = DiscoveryPipeline(
        _holdout_cfg(cross_window_months=[1]), backtest_fn=lambda c: _bt(),
        holdout_backtest_fn=holdout,
        backtest_fn_factory=lambda s, e: (
            lambda c: _bt() if c.stop_loss_pct == 3.0 else _bt(ret=-0.1)
        ),
    )
    res = _gates(pipe, _top(3))
    assert seen == [3.0]
    assert [c.stop_loss_pct for c, _ in res.top_strategies] == [3.0]


def test_cli_and_api_default_to_a_holdout() -> None:
    from vibe_quant.api.schemas.discovery import DiscoveryLaunchRequest
    from vibe_quant.discovery.__main__ import build_parser

    assert build_parser().parse_args(["--run-id", "1"]).train_test_split == 0.8
    assert DiscoveryLaunchRequest().train_test_split == 0.8


def test_full_run_all_fail_holdout_persists_zero_with_reasons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end CLI (mock): champions all fail the holdout -> 0 champions."""
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery import __main__ as cli

    db = tmp_path / "s.db"
    sm = StateManager(db)
    run_id = sm.create_backtest_run(None, "discovery", ["BTCUSDT"], "4h",
                                    "2024-01-01", "2024-07-01", {})
    sm.close()

    def mock_bt(chrom: StrategyChromosome) -> dict[str, Any]:
        r = random.Random(len(chrom.entry_genes))
        return {**_bt(sharpe=2.5, ret=0.4, trades=150),
                "trade_returns": tuple(r.gauss(0.003, 0.001) for _ in range(150))}

    monkeypatch.setattr(cli, "_mock_backtest", mock_bt)
    # the holdout uses the same mock fn object; make it lose instead
    real_pipeline_init = DiscoveryPipeline.__init__

    def init(self: DiscoveryPipeline, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("holdout_backtest_fn") is not None:
            kwargs["holdout_backtest_fn"] = lambda c: _bt(ret=-0.05, trades=40)
        real_pipeline_init(self, *args, **kwargs)

    monkeypatch.setattr(DiscoveryPipeline, "__init__", init)
    monkeypatch.setattr("sys.argv", [
        "prog", "--run-id", str(run_id), "--population-size", "6", "--max-generations", "2",
        "--elite-count", "1", "--max-workers", "-1", "--timeframe", "4h",
        "--start-date", "2024-01-01", "--end-date", "2024-07-01",
        "--no-dsr", "--db", str(db), "--mock",
    ])
    assert cli.main() == 0
    sm = StateManager(db)
    notes = json.loads(sm.get_backtest_result(run_id)["notes"])  # type: ignore[index]
    sm.close()
    assert notes["outcome"] == "no_viable_strategies"
    assert notes["top_strategies"] == []
    # 182 days * 0.8 = 145 train days -> holdout from 2024-05-25
    assert notes["holdout_dates"] == ["2024-05-25", "2024-07-01"]
    # 50 * 37 holdout days / (2 * 145 train days) = 6.38 -> 6
    assert notes["holdout_gate"]["min_trades"] == 6
    assert notes["guardrail_rejections"]
    assert {r["stage"] for r in notes["guardrail_rejections"]} == {"holdout"}
    assert "holdout" in notes["reason"]


# ---------------------------------------------------------------------------
# e70tl.6 — eval_windows worst-of-N
# ---------------------------------------------------------------------------


def _w(sharpe: float, ret: float, trades: int, dd: float = 0.1, pf: float = 1.5) -> dict[str, Any]:
    return {"sharpe_ratio": sharpe, "max_drawdown": dd, "profit_factor": pf,
            "total_trades": trades, "total_return": ret}


def test_worst_of_n_gates_one_great_window_two_bad() -> None:
    """Audit C6: Sharpe [6,-1,-1], trades [48,1,1] used to score 0.587."""
    agg = NTBacktestFn._aggregate_multi_window(
        [_w(6.0, 0.90, 48, 0.05, 3.0), _w(-1.0, -0.10, 1), _w(-1.0, -0.10, 1)], min_trades=50,
    )
    fr = _evaluate_single(_chrom(), lambda _c: agg, min_trades=50, timeframe="4h")
    assert fr.adjusted_score == 0.0


def test_worst_of_n_takes_min_max_and_sums_trades() -> None:
    agg = NTBacktestFn._aggregate_multi_window(
        [_w(2.0, 0.20, 30, 0.05, 2.0), _w(1.0, 0.05, 25, 0.12, 1.3), _w(1.5, 0.10, 20, 0.08, 1.6)],
        min_trades=50,
    )
    assert agg["sharpe_ratio"] == 1.0
    assert agg["total_return"] == 0.05
    assert agg["max_drawdown"] == 0.12
    assert agg["profit_factor"] == 1.3
    assert agg["total_trades"] == 75


def test_per_window_trade_gate() -> None:
    # min_trades 50, N=3 -> each window needs max(1, 50 // 6) = 8 trades
    assert NTBacktestFn.per_window_min_trades(50, 3) == 8
    ok = NTBacktestFn._aggregate_multi_window(
        [_w(1.0, 0.1, 30), _w(1.0, 0.1, 30), _w(1.0, 0.1, 8)], min_trades=50,
    )
    bad = NTBacktestFn._aggregate_multi_window(
        [_w(1.0, 0.1, 30), _w(1.0, 0.1, 30), _w(1.0, 0.1, 7)], min_trades=50,
    )
    assert ok["sharpe_ratio"] == 1.0
    assert bad["sharpe_ratio"] == -1.0 and bad["total_return"] == 0.0
    assert bad["total_trades"] == 67  # honest count, still gated by return <= 0


def test_eval_windows_one_runs_single_backtest_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_single(self: NTBacktestFn, chrom: Any, s: str, e: str) -> dict[str, Any]:
        calls.append((s, e))
        return _w(1.234, 0.1, 60)

    monkeypatch.setattr(NTBacktestFn, "_run_single", fake_single)
    fn = NTBacktestFn(["BTCUSDT"], "4h", "2024-01-01", "2024-06-01", windows=None, min_trades=50)
    assert fn(_chrom()) == _w(1.234, 0.1, 60)
    assert calls == [("2024-01-01", "2024-06-01")]


def test_replay_note_and_cli_help_say_worst_of_n() -> None:
    from vibe_quant.discovery.__main__ import build_parser

    help_text = build_parser().format_help()
    assert "WORST" in help_text and "min Sharpe" in help_text


# ---------------------------------------------------------------------------
# e70tl.1 — fresh uids
# ---------------------------------------------------------------------------


def test_mutant_and_crossover_children_get_fresh_uids() -> None:
    random.seed(1)
    parent = _random_chromosome()
    child = mutate(parent, mutation_rate=1.0)
    assert child.uid != parent.uid
    assert mutate(parent, mutation_rate=0.0).uid != parent.uid
    a, b = crossover(parent, _random_chromosome())
    assert len({a.uid, b.uid, parent.uid}) == 3
    assert parent.clone().uid == parent.uid  # unchanged copies keep it


def test_evolved_population_has_unique_names() -> None:
    """Elite + its mutants used to share genome_<uid> (13 runs affected)."""
    random.seed(7)
    for crowding in (True, False):
        cfg = _cfg(population_size=12, elite_count=2, use_crowding=crowding)
        pipe = DiscoveryPipeline(cfg, backtest_fn=lambda c: _bt())
        pop = initialize_population(12)
        fit = [_fit(score=0.1 * i) for i in range(12)]
        new_pop, _known = pipe._evolve_generation(pop, fit)
        names = [chromosome_to_dsl(c)["name"] for c in new_pop]
        assert len(set(names)) == len(names)


def test_seed_clones_get_fresh_uids() -> None:
    seed = _chrom()
    pop = initialize_population(4, seed_chromosomes=[seed])
    assert pop[0].uid != seed.uid


# ---------------------------------------------------------------------------
# Medium: GA immigrants / crowding / dead genes / fitness details
# ---------------------------------------------------------------------------


def test_immigrant_fraction_zero_injects_nothing() -> None:
    from vibe_quant.discovery.diversity import inject_random_immigrants

    random.seed(0)
    pop = initialize_population(10)
    after = inject_random_immigrants(pop, [0.5] * 10, fraction=0.0)
    assert all(a is b for a, b in zip(pop, after, strict=True))


def test_immigrants_never_replace_protected_elite_and_reject_stale_scores() -> None:
    from vibe_quant.discovery.diversity import inject_random_immigrants

    random.seed(3)
    pop = initialize_population(20)
    for _ in range(50):
        after = inject_random_immigrants(pop, None, fraction=0.5, protected=[0])
        assert after[0] is pop[0]
    with pytest.raises(ValueError, match="not parallel"):
        inject_random_immigrants(pop, [0.1] * 19, fraction=0.1)


def test_pipeline_immigrant_injection_keeps_elite() -> None:
    random.seed(5)
    cfg = _cfg(population_size=20, entropy_threshold=1.01, immigrant_fraction=0.5)
    pipe = DiscoveryPipeline(cfg, backtest_fn=lambda c: _bt())
    pop = initialize_population(20)
    known: list[FitnessResult | None] = [_fit(score=0.1)] * 20
    new_pop, new_known = pipe._maybe_inject_immigrants(pop, known)
    assert new_pop[0] is pop[0] and new_known[0] is known[0]
    replaced = [i for i in range(20) if new_pop[i] is not pop[i]]
    assert len(replaced) == 10
    assert all(new_known[i] is None for i in replaced)


def test_crowding_uses_real_offspring_fitness(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero-fitness offspring must not replace fit parents (old code passed the
    PARENTS' scores as offspring fitness, so every child won the tie)."""
    random.seed(11)
    cfg = _cfg(population_size=9, elite_count=1, use_crowding=True)
    pipe = DiscoveryPipeline(cfg, backtest_fn=lambda c: _bt())
    zero = _fit(score=0.0)
    monkeypatch.setattr(pipe, "_evaluate_new", lambda chroms: [zero] * len(chroms))
    pop = initialize_population(9)
    fit = [_fit(score=0.5 + 0.01 * i) for i in range(9)]
    new_pop, new_known = pipe._evolve_crowding(pop, fit)
    assert len(new_pop) == 9
    # every parent beats its zero-score offspring -> the population is the parents
    assert {c.uid for c in new_pop} == {c.uid for c in pop}
    assert all(k is not None and k.adjusted_score > 0 for k in new_known)


def test_fitness_cache_counts_distinct_strategies() -> None:
    calls: list[str] = []

    def bt(c: StrategyChromosome) -> dict[str, Any]:
        calls.append(c.uid)
        return _bt()

    pipe = DiscoveryPipeline(_cfg(), backtest_fn=bt)
    a = _chrom()
    twin = _chrom()  # same DSL, different uid/name
    other = _chrom(stop_loss_pct=3.0)
    pipe._evaluate_new([a, twin, other, a.clone()])
    assert len(calls) == 2
    assert pipe._total_evaluated == 2


def test_adaptive_rsi_alpha_reaches_dsl() -> None:
    c1 = _chrom(entry_genes=[_gene("ADAPTIVE_RSI", {"period": 14.0, "alpha": 0.1})])
    c2 = _chrom(entry_genes=[_gene("ADAPTIVE_RSI", {"period": 14.0, "alpha": 0.9})])
    d1, d2 = chromosome_to_dsl(c1), chromosome_to_dsl(c2)
    assert d1["indicators"]["adaptive_rsi_entry_0"] == {  # type: ignore[index]
        "type": "ADAPTIVE_RSI", "period": 14, "alpha": 0.1,
    }
    d1.pop("name")
    d2.pop("name")
    assert d1 != d2


def test_dead_and_fractional_params_are_canonicalized() -> None:
    stoch = canonicalize_gene(_gene("STOCH", {"period_k": 9.6, "period_d": 7.0}))
    assert stoch.parameters == {"period_k": 10.0, "period_d": 3.0}
    macd_line = canonicalize_gene(
        _gene("MACD", {"fast_period": 12.0, "slow_period": 26.0, "signal_period": 11.0})
    )
    assert macd_line.parameters["signal_period"] == 9.0  # unused by the MACD line
    macd_sig = canonicalize_gene(
        _gene("MACD", {"fast_period": 12.0, "slow_period": 26.0, "signal_period": 11.0},
              sub="signal")
    )
    assert macd_sig.parameters["signal_period"] == 11.0
    arsi = canonicalize_gene(_gene("ADAPTIVE_RSI", {"period": 14.4, "alpha": 0.3712}))
    assert arsi.parameters == {"period": 14.0, "alpha": 0.3712}


def test_random_and_mutated_genes_have_integer_periods() -> None:
    random.seed(2)
    from vibe_quant.discovery.operators import _INT_PARAMS

    pop = initialize_population(30)
    for _ in range(3):
        pop = [mutate(c, 1.0) for c in pop]
    for c in pop:
        for g in c.entry_genes + c.exit_genes:
            for p in _INT_PARAMS.get(g.indicator_type, frozenset()):
                if p in g.parameters:
                    assert g.parameters[p] == int(g.parameters[p])


def test_sl_tp_penalty_uses_per_direction_values_for_both() -> None:
    both = _chrom(
        direction=Direction.BOTH, stop_loss_pct=2.0, take_profit_pct=2.0,
        stop_loss_long_pct=10.0, take_profit_long_pct=0.5,
        stop_loss_short_pct=2.0, take_profit_short_pct=4.0,
    )
    fr = _evaluate_single(both, lambda _c: _bt(), min_trades=50, timeframe="4h")
    assert fr.sl_tp_penalty == compute_sl_tp_penalty(10.0, 0.5) == 0.15


def test_nan_profit_factor_with_profit_scores_at_cap() -> None:
    bt = {**_bt(sharpe=1.5, ret=0.2, trades=80), "profit_factor": float("nan")}
    fr = _evaluate_single(_chrom(), lambda _c: bt, min_trades=50, timeframe="4h")
    assert fr.profit_factor == PF_MAX
    assert fr.raw_score == compute_fitness_score(1.5, 0.1, PF_MAX, 0.2)


def test_missing_catalog_data_fails_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery import __main__ as cli

    db = tmp_path / "s.db"
    sm = StateManager(db)
    run_id = sm.create_backtest_run(None, "discovery", ["NOPEUSDT"], "4h",
                                    "2024-01-01", "2024-07-01", {})
    sm.close()
    monkeypatch.setattr(cli, "_check_data_available", lambda symbols: False)
    monkeypatch.setattr("sys.argv", [
        "prog", "--run-id", str(run_id), "--symbols", "NOPEUSDT",
        "--start-date", "2024-01-01", "--end-date", "2024-07-01", "--db", str(db),
    ])
    assert cli.main() == 1
    sm = StateManager(db)
    run = sm.get_backtest_run(run_id)
    sm.close()
    assert run is not None and run["status"] == "failed"
    assert "No catalog bar data" in str(run["error_message"])


def test_multi_seed_rechecks_dsr_with_pooled_trials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-seed N=400 passes, but the pick is across 3 seeds (N=1200)."""
    from vibe_quant.discovery import __main__ as cli
    from vibe_quant.discovery.guardrails import apply_discovery_dsr
    from vibe_quant.discovery.pipeline import DiscoveryResult
    from vibe_quant.utils import compute_day_count

    cfg = _cfg(require_dsr=True, top_k=1)
    days = compute_day_count(cfg.start_date, cfg.end_date)
    assert days is not None
    sharpe = next(
        s / 100 for s in range(100, 1000)
        if apply_discovery_dsr(s / 100, 400, days).is_significant
        and not apply_discovery_dsr(s / 100, 1200, days).is_significant
    )
    champ = (_chrom(), _fit(sharpe=sharpe))

    def fake_run(self: DiscoveryPipeline) -> DiscoveryResult:
        return DiscoveryResult(
            generations=[], top_strategies=[champ], total_candidates_evaluated=400,
            converged=False, convergence_generation=None,
        )

    monkeypatch.setattr(DiscoveryPipeline, "run", fake_run)
    res = cli._run_multi_seed(3, cfg, lambda c: _bt(), "/tmp/vq_disc_progress.json")
    assert res.total_candidates_evaluated == 1200
    assert res.top_strategies == []
    assert any(
        r["stage"] == "pooled_guardrails" and "N=1200" in str(r["reasons"])
        for r in res.guardrail_rejections
    )
