"""Discovery DSR on the pooled MARK-TO-MARKET daily series (yul7u.24, y9ywf).

The DSR gate used to test the worst-of-symbols x windows Sharpe with T = train
days and per-trade moments. It now tests one dense daily return series (the
shared account in portfolio mode, the equal-weight pool of the symbols in
worst mode) with its own T and moments at raw N, plus a worst-mode per-symbol
PSR floor and conservative (clamped) moments.

``vibe-quant-y9ywf``: that series is the MARK-TO-MARKET one (``mtm_daily_returns``
/ ``symbol_mtm_daily``), not the realized-balance series: the latter books a
multi-day trade's whole PnL on its exit day, so DSR was too lenient. A missing
MTM series now fails the gate closed.
"""

from __future__ import annotations

import math
from array import array
from typing import Any

import numpy as np
import pytest

from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.discovery.fitness import FitnessResult, _evaluate_single
from vibe_quant.discovery.guardrails import (
    SYMBOL_PSR_FLOOR,
    apply_discovery_dsr,
    apply_discovery_dsr_returns,
)
from vibe_quant.metrics import (
    DailyReturns,
    concat_daily_returns,
    dense_daily_returns,
    pool_daily_returns,
)
from vibe_quant.overfitting.dsr import DeflatedSharpeRatio, daily_sharpe_inputs
from vibe_quant.utils import compute_day_count, split_into_windows

DAY_NS = 86_400 * 1_000_000_000
ANN = math.sqrt(252.0)


def _series(sr_ann: float, n: int = 1264, seed: int = 1, first_day: int = 19_000) -> DailyReturns:
    """Gaussian daily returns standardized to EXACTLY ``sr_ann`` annualized Sharpe."""
    z = np.random.default_rng(seed).normal(size=n)
    z = (z - z.mean()) / z.std(ddof=1)
    sigma = 0.01
    x = z * sigma + sr_ann / ANN * sigma
    return DailyReturns(first_day, array("d", x.tobytes()))


def _chrom(**kw: Any) -> Any:
    from vibe_quant.discovery.operators import (
        ConditionType,
        Direction,
        StrategyChromosome,
        StrategyGene,
    )

    base: dict[str, Any] = {
        "entry_genes": [StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 30.0, None)],
        "exit_genes": [StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0, None)],
        "stop_loss_pct": 2.0, "take_profit_pct": 4.0, "direction": Direction.LONG,
    }  # fmt: skip
    base.update(kw)
    return StrategyChromosome(**base)


def _fit(sharpe: float = 0.1, **kw: Any) -> FitnessResult:
    base: dict[str, Any] = {
        "sharpe_ratio": sharpe, "max_drawdown": 0.1, "profit_factor": 1.5,
        "total_trades": 200, "total_return": 0.3, "complexity_penalty": 0.0,
        "overtrade_penalty": 0.0, "sl_tp_penalty": 0.0, "raw_score": 0.5,
        "adjusted_score": 0.5, "passed_filters": True, "filter_results": {},
    }  # fmt: skip
    base.update(kw)
    return FitnessResult(**base)


def _pipeline(symbols: list[str] | None = None, **cfg: Any) -> Any:
    from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline

    base: dict[str, Any] = {
        "population_size": 6, "max_generations": 1, "elite_count": 1, "top_k": 5,
        "min_trades": 50, "max_workers": None, "timeframe": "4h",
        "start_date": "2022-06-01", "end_date": "2025-11-17",
        "require_bootstrap_ci": False, "require_dsr": True,
    }  # fmt: skip
    base.update(cfg)
    fn = NTBacktestFn(symbols or ["BTCUSDT"], "4h", base["start_date"], base["end_date"])
    return DiscoveryPipeline(DiscoveryConfig(**base), backtest_fn=fn)


# ---------------------------------------------------------------------------
# T2/T3: gate input is the pooled daily series, fail closed when missing
# ---------------------------------------------------------------------------


def test_dsr_uses_pooled_daily_not_worst_of() -> None:
    pipe = _pipeline()
    assert pipe._backtest_fn.collect_daily_returns is True  # real fn told to collect
    # The OLD input (worst-of Sharpe 0.1, T = 1265 train days) cannot pass ...
    assert not apply_discovery_dsr(0.1, 79, 1265).is_significant

    strong = (_chrom(stop_loss_pct=2.0), _fit(0.1, mtm_daily_returns=_series(2.5)))
    weak = (_chrom(stop_loss_pct=3.0), _fit(0.1, mtm_daily_returns=_series(1.0)))
    empty = (_chrom(stop_loss_pct=4.0), _fit(0.1, mtm_daily_returns=None))
    kept = pipe._validate_top_strategies([strong, weak, empty], 79)

    # ... the pooled mark-to-market series with SR 2.5 passes at N=79, T=1264
    assert [c.uid for c, _ in kept] == [strong[0].uid]
    rec = pipe._dsr_records[strong[0].uid]
    assert rec["input"] == "pooled_mtm_daily_returns"
    assert rec["observations"] == 1264 and rec["trials"] == 79
    assert rec["sharpe_annualized"] == pytest.approx(2.5, rel=1e-9)
    assert rec["p_value"] < 0.05

    by_uid = {r["uid"]: r for r in pipe._guardrail_rejections}
    weak_reasons = by_uid[weak[0].uid]["reasons"]
    assert any(r.startswith("DSR not significant: p=") for r in weak_reasons)
    assert by_uid[weak[0].uid]["dsr"]["p_value"] >= 0.05
    empty_reasons = by_uid[empty[0].uid]["reasons"]
    assert any(r.startswith("DSR input missing") for r in empty_reasons)


def test_dsr_tests_the_mtm_series_not_the_realized_one() -> None:
    """The gate's pooled input is the mark-to-market series, not the reported Sharpe."""
    pipe = _pipeline()
    mtm_weak = _series(0.2, seed=2)
    mtm_strong = _series(2.5, seed=4)
    # the SAME reported Sharpe for both; only the MTM series differs
    weak = (_chrom(stop_loss_pct=2.0), _fit(0.1, mtm_daily_returns=mtm_weak))
    strong = (_chrom(stop_loss_pct=3.0), _fit(0.1, mtm_daily_returns=mtm_strong))
    kept = pipe._validate_top_strategies([weak, strong], 79)

    assert [c.uid for c, _ in kept] == [strong[0].uid]
    by_uid = {r["uid"]: r for r in pipe._guardrail_rejections}
    weak_rec = by_uid[weak[0].uid]["dsr"]
    assert weak_rec["input"] == "pooled_mtm_daily_returns"
    assert weak_rec["observations"] == 1264
    # the MTM series (0.2 ann), never the reported Sharpe 0.1
    assert weak_rec["sharpe_annualized"] == pytest.approx(0.2, rel=1e-9)
    sr_mtm, _skew, _kurt = daily_sharpe_inputs(list(mtm_weak.values))
    assert weak_rec["sharpe_daily"] == pytest.approx(sr_mtm)
    assert any(r.startswith("DSR not significant: p=") for r in by_uid[weak[0].uid]["reasons"])
    # and the passing candidate's verdict came from its MTM series too
    assert pipe._dsr_records[strong[0].uid]["sharpe_annualized"] == pytest.approx(2.5, rel=1e-9)


def test_missing_mtm_series_fails_dsr_closed() -> None:
    pipe = _pipeline()
    chrom = _chrom()
    # the reported Sharpe is strong, but the MTM series is missing -> fail closed
    fit = _fit(3.0, mtm_daily_returns=None)
    kept = pipe._validate_top_strategies([(chrom, fit)], 79)

    assert kept == []
    rej = pipe._guardrail_rejections[0]
    assert any(r.startswith("DSR input missing") for r in rej["reasons"])
    assert rej["dsr"]["input"] == "pooled_mtm_daily_returns" and rej["dsr"]["missing"] is True


def test_symbol_mtm_stats_helper_shape_and_fail_closed() -> None:
    from vibe_quant.discovery.pipeline import _symbol_mtm_stats

    a = _series(2.5, seed=31)
    out = _symbol_mtm_stats({"A": a, "B": None, "C": DailyReturns(19_000, array("d"))})
    assert out is not None
    assert out["B"] is None and out["C"] is None  # missing/empty -> None, floor fails closed
    sr, skew, kurt = daily_sharpe_inputs(a.values)
    assert out["A"] == {
        "sharpe": sr, "skewness": skew, "kurtosis": kurt,
        "observations": len(a), "mean": sum(a.values) / len(a),
    }
    assert _symbol_mtm_stats(None) is None  # portfolio mode: no per-symbol floor


def test_mock_backtest_fn_keeps_legacy_input() -> None:
    from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline

    cfg = DiscoveryConfig(
        population_size=6, max_generations=1, start_date="2022-06-01",
        end_date="2025-11-17", require_bootstrap_ci=False, require_dsr=True,
    )  # fmt: skip
    pipe = DiscoveryPipeline(cfg, backtest_fn=lambda c: {})
    chrom = _chrom()
    kept = pipe._validate_top_strategies([(chrom, _fit(3.0))], 79)
    assert len(kept) == 1
    assert pipe._dsr_records[chrom.uid]["input"] == "fitness_sharpe"


@pytest.mark.parametrize(
    "values",
    [[], [0.01], [0.01, float("nan"), 0.02]],
    ids=["empty", "one-day", "nan"],
)
def test_unusable_series_fails_closed(values: list[float]) -> None:
    result, record, reasons = apply_discovery_dsr_returns(values, num_trials=1)
    assert result is None and record["missing"] is True
    assert reasons and reasons[0].startswith("DSR input missing")


# ---------------------------------------------------------------------------
# SF1: worst-mode per-symbol floor
# ---------------------------------------------------------------------------


def _sym_stats(sr_ann: float, mean: float = 1e-4) -> dict[str, float]:
    return {
        "sharpe": sr_ann / ANN, "skewness": 0.0, "kurtosis": 3.0,
        "observations": 1264, "mean": mean,
    }  # fmt: skip


def test_one_strong_symbol_cannot_carry_null_symbols() -> None:
    pooled = _series(2.5).values  # the pool itself clears the DSR
    carried = {"A": _sym_stats(4.0, 3e-4), "B": _sym_stats(0.05, 1e-5), "C": _sym_stats(0.02, 1e-5)}
    result, record, reasons = apply_discovery_dsr_returns(pooled, 79, symbol_daily=carried)
    assert result is not None and result.is_significant  # pooled test alone would pass
    assert reasons == [r for r in reasons if r.startswith("DSR per-symbol floor")]
    assert {r.split(":")[1].split()[0] for r in reasons} == {"B", "C"}
    syms = record["symbols"]
    assert isinstance(syms, dict)
    assert syms["A"]["psr"] > SYMBOL_PSR_FLOOR > syms["B"]["psr"]
    assert syms["A"]["contribution_share"] == pytest.approx(3e-4 / 3.2e-4)
    assert syms["B"]["sharpe_annualized"] == pytest.approx(0.05)

    # Same pool, every symbol individually credible -> passes.
    even = {s: _sym_stats(1.0) for s in "ABC"}
    _, _, ok_reasons = apply_discovery_dsr_returns(pooled, 79, symbol_daily=even)
    assert ok_reasons == []


@pytest.mark.parametrize("n_symbols", [2, 3])
def test_worst_mode_symbol_floor_rejects_weak_mtm_symbol(n_symbols: int) -> None:
    """The per-symbol floor rejects the weak symbol for 2 AND 3 symbols.

    ``apply_discovery_dsr_returns`` applies the floor only when
    ``len(symbol_daily) >= 2``, so two symbols is the boundary: a regression
    that drops the floor for exactly two symbols must fail the ``n_symbols=2``
    case. The floor reads each symbol's MTM series only (the realized stats
    dict the pre-y9ywf gate used no longer exists, bd vibe-quant-f9zzq).
    """
    strong = [f"S{i}" for i in range(n_symbols - 1)]
    weak = "W"
    pipe = _pipeline([*strong, weak])
    chrom = _chrom()
    mtm: dict[str, DailyReturns] = {s: _series(4.0, seed=30 + i) for i, s in enumerate(strong)}
    mtm[weak] = _series(0.05, seed=39)
    fit = _fit(
        0.1,
        mtm_daily_returns=_series(2.5, seed=24),  # the pool itself clears the DSR
        symbol_mtm_daily=mtm,
    )
    assert pipe._validate_top_strategies([(chrom, fit)], 79) == []
    rej = pipe._guardrail_rejections[0]
    floor = [r for r in rej["reasons"] if r.startswith("DSR per-symbol floor")]
    assert {r.split(":")[1].split()[0] for r in floor} == {weak}, rej["reasons"]
    rec = rej["dsr"]
    assert isinstance(rec["symbols"], dict) and len(rec["symbols"]) == n_symbols
    assert rec["symbols"][weak]["psr"] < SYMBOL_PSR_FLOOR
    for s in strong:
        assert rec["symbols"][s]["psr"] > SYMBOL_PSR_FLOOR


def test_missing_symbol_series_fails_closed() -> None:
    stats: dict[str, dict[str, float] | None] = {"A": _sym_stats(2.0), "B": None}
    _, record, reasons = apply_discovery_dsr_returns(_series(2.5).values, 79, symbol_daily=stats)
    assert record["missing"] is True and reasons == ["DSR input missing: no daily return series for B"]


def test_single_symbol_has_no_extra_floor() -> None:
    _, record, reasons = apply_discovery_dsr_returns(
        _series(2.5).values, 79, symbol_daily={"A": _sym_stats(0.01)}
    )
    assert reasons == [] and "symbols" not in record


# ---------------------------------------------------------------------------
# SF3: conservative moments
# ---------------------------------------------------------------------------


def test_dsr_clamps_skew_and_kurtosis() -> None:
    rng = np.random.default_rng(3)
    # Positively skewed series (rare big up days): its raw moments would LOWER the bar.
    x = np.where(rng.random(1264) < 0.05, rng.exponential(0.05, 1264), rng.normal(-0.0005, 0.004, 1264))
    sr, skew, kurt = daily_sharpe_inputs(x)
    assert skew > 1.0
    result, record, _ = apply_discovery_dsr_returns(list(x), 79)
    assert result is not None
    assert record["skewness"] == skew and record["skewness_used"] == 0.0
    expected = DeflatedSharpeRatio().calculate(sr, 79, 1264, 0.0, max(kurt, 3.0)).p_value
    assert record["p_value"] == expected
    assert float(record["p_value"]) > float(record["p_value_unclamped"])  # type: ignore[arg-type]

    # Thin tails (uniform, kurtosis ~1.8) are floored at 3.
    u = rng.uniform(-0.01, 0.0112, 1264)
    sr_u, sk_u, ku = daily_sharpe_inputs(u)
    assert ku < 3.0
    _, rec_u, _ = apply_discovery_dsr_returns(list(u), 79)
    assert rec_u["kurtosis_used"] == 3.0
    dsr = DeflatedSharpeRatio()
    assert rec_u["p_value"] == dsr.calculate(sr_u, 79, 1264, min(sk_u, 0.0), 3.0).p_value
    assert rec_u["p_value"] != dsr.calculate(sr_u, 79, 1264, min(sk_u, 0.0), ku).p_value


# ---------------------------------------------------------------------------
# SF5: windows chain to exactly the days of one continuous run
# ---------------------------------------------------------------------------


def _ns(date: str) -> int:
    from vibe_quant.validation.extraction import date_to_ns

    ns = date_to_ns(date)
    assert ns is not None
    return ns


def test_three_window_chain_matches_continuous_days() -> None:
    start, end = "2022-06-01", "2025-11-17"
    rng = np.random.default_rng(5)
    s_ns, e_ns = _ns(start), _ns(end)
    events = [(int(rng.integers(s_ns, e_ns)), float(rng.normal(1.0, 20.0))) for _ in range(300)]
    windows = split_into_windows(start, end, 3)
    parts: list[dict[str, Any]] = [
        {"mtm_daily_returns": dense_daily_returns(1000.0, events, _ns(ws), _ns(we))}
        for ws, we in windows
    ]
    chain = NTBacktestFn._chain_window_series(parts)
    cont = dense_daily_returns(1000.0, events, s_ns, e_ns)
    assert chain is not None and cont is not None
    cont = cont.without_first_day()
    day_count = compute_day_count(start, end)
    assert day_count == 1265
    assert len(chain) == len(cont) == day_count - 1  # no boundary day lost or duplicated
    assert (chain.first_day, chain.end_day) == (cont.first_day, cont.end_day)
    # First window: same values as the continuous run (same starting balance),
    # except its last day, where later events are clamped in the window run
    w1 = len(parts[0]["mtm_daily_returns"]) - 1  # type: ignore[arg-type]
    assert list(chain.values[: w1 - 1]) == list(cont.values[: w1 - 1])


def test_eval_symbols_chains_windows_and_trims_day0(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_single(self: NTBacktestFn, chrom: Any, s: str, e: str, symbols: Any = None) -> Any:
        n = compute_day_count(s, e) or 0
        out: dict[str, Any] = {
            "sharpe_ratio": 1.0, "max_drawdown": 0.1, "profit_factor": 1.5,
            "total_trades": 60, "total_return": 0.1,
        }  # fmt: skip
        if self.collect_daily_returns:  # what the real _run_single does
            out["mtm_daily_returns"] = DailyReturns(_ns(s) // DAY_NS, array("d", [0.001] * n))
        return out

    monkeypatch.setattr(NTBacktestFn, "_run_single", fake_single)
    start, end = "2024-01-01", "2024-07-01"
    multi = NTBacktestFn(["X"], "4h", start, end, windows=split_into_windows(start, end, 3),
                         min_trades=10, collect_daily_returns=True)  # fmt: skip
    single = NTBacktestFn(["X"], "4h", start, end, collect_daily_returns=True)
    out_m = multi(_chrom())["mtm_daily_returns"]
    out_s = single(_chrom())["mtm_daily_returns"]
    assert isinstance(out_m, DailyReturns) and isinstance(out_s, DailyReturns)
    assert (out_m.first_day, len(out_m)) == (out_s.first_day, len(out_s))
    assert len(out_s) == (compute_day_count(start, end) or 0) - 1
    assert "mtm_daily_returns" not in NTBacktestFn(["X"], "4h", start, end)(_chrom())


def test_concat_fails_closed_on_gap_or_overlap() -> None:
    a = DailyReturns(100, array("d", [0.1, 0.2]))
    assert concat_daily_returns([a, DailyReturns(102, array("d", [0.3]))]) is not None
    assert concat_daily_returns([a, DailyReturns(103, array("d", [0.3]))]) is None  # gap
    assert concat_daily_returns([a, DailyReturns(101, array("d", [0.3]))]) is None  # dup day
    assert concat_daily_returns([a, None]) is None


# ---------------------------------------------------------------------------
# SF6: date-key alignment, fail closed, portfolio/single-symbol behaviour
# ---------------------------------------------------------------------------


def test_pool_aligns_by_date_key_and_fails_closed() -> None:
    a = DailyReturns(100, array("d", [0.01, 0.02, 0.03]))
    b = DailyReturns(100, array("d", [0.03, 0.00, -0.03]))
    pooled = pool_daily_returns([a, b])
    assert pooled is not None and pooled.first_day == 100
    assert list(pooled.values) == pytest.approx([0.02, 0.01, 0.0])
    # same length, shifted one day: must NOT be pooled index-by-index
    assert pool_daily_returns([a, DailyReturns(101, array("d", [0.0, 0.0, 0.0]))]) is None
    assert pool_daily_returns([a, DailyReturns(100, array("d", [0.0, 0.0]))]) is None  # length
    assert pool_daily_returns([a, DailyReturns(100, array("d", [0.0, math.nan, 0.0]))]) is None
    assert pool_daily_returns([a, DailyReturns(100, array("d"))]) is None  # empty
    assert pool_daily_returns([a, None]) is None


def test_dense_series_fails_closed_on_non_positive_balance() -> None:
    s, e = 19_000 * DAY_NS, 19_010 * DAY_NS
    assert dense_daily_returns(1000.0, [(s + 3 * DAY_NS, -1000.0)], s, e) is None
    assert dense_daily_returns(0.0, [], s, e) is None
    ok = dense_daily_returns(1000.0, [(s + 3 * DAY_NS, 100.0)], s, e)
    assert ok is not None and len(ok) == 10 and ok.values[3] == pytest.approx(0.1)


def _sym_result(daily: DailyReturns | None, ret: float = 0.2) -> dict[str, Any]:
    return {
        "sharpe_ratio": 1.0, "max_drawdown": 0.1, "profit_factor": 1.5,
        "total_trades": 100, "total_return": ret, "mtm_daily_returns": daily,
    }  # fmt: skip


def test_worst_mode_pools_symbols_and_strips_their_series() -> None:
    syms = ["A", "B", "C"]
    series = [_series(sr, seed=i) for i, sr in enumerate((2.0, 1.0, 0.5))]
    out = NTBacktestFn._aggregate_symbols([_sym_result(s) for s in series], syms)
    pooled = out["mtm_daily_returns"]
    assert isinstance(pooled, DailyReturns)
    expect = np.mean([np.asarray(s.values) for s in series], axis=0)
    assert list(pooled.values) == list(expect)
    sd = out["symbol_mtm_daily"]
    assert isinstance(sd, dict)
    assert sd["A"] is series[0]  # the raw series itself, not realized stats
    assert all(isinstance(v, DailyReturns) for v in sd.values())
    assert all("mtm_daily_returns" not in r for r in out["symbol_results"].values())  # type: ignore[union-attr]
    # one symbol without a series -> no pool, that symbol flagged missing
    broken = NTBacktestFn._aggregate_symbols(
        [_sym_result(series[0]), _sym_result(None), _sym_result(series[2])], syms
    )
    assert broken["mtm_daily_returns"] is None and broken["symbol_mtm_daily"]["B"] is None  # type: ignore[index]


def test_portfolio_mode_uses_shared_account_series(monkeypatch: pytest.MonkeyPatch) -> None:
    shared = DailyReturns(19_000, array("d", [0.0, 0.01, -0.005, 0.002]))
    seen: list[Any] = []

    def fake_single(self: NTBacktestFn, chrom: Any, s: str, e: str, symbols: Any = None) -> Any:
        seen.append(symbols)
        return _sym_result(shared)

    monkeypatch.setattr(NTBacktestFn, "_run_single", fake_single)
    fn = NTBacktestFn(["A", "B"], "4h", "2022-01-01", "2022-01-05", collect_daily_returns=True)
    out: dict[str, Any] = fn(_chrom())
    assert seen == [None]  # ONE shared-account run over both symbols
    assert out["mtm_daily_returns"] == shared.without_first_day()
    assert "symbol_mtm_daily" not in out  # no per-symbol floor in portfolio mode
    # worst mode with ONE symbol is a single run too (no pool, no floor)
    one: dict[str, Any] = NTBacktestFn(["A"], "4h", "2022-01-01", "2022-01-05", symbol_agg="worst",
                       collect_daily_returns=True)(_chrom())  # fmt: skip
    assert one["mtm_daily_returns"] == shared.without_first_day() and "symbol_mtm_daily" not in one


# ---------------------------------------------------------------------------
# SF2: only champion candidates keep a series
# ---------------------------------------------------------------------------


def _const(bt: dict[str, Any]) -> Any:
    return lambda _c: bt


def test_series_kept_only_for_gate_passing_genomes() -> None:
    d = _series(1.0)
    base = {"sharpe_ratio": 1.0, "max_drawdown": 0.1, "profit_factor": 1.5,
            "total_trades": 100, "total_return": 0.2, "mtm_daily_returns": d}  # fmt: skip
    assert _evaluate_single(_chrom(), _const(base), min_trades=50).mtm_daily_returns is d
    losing = {**base, "total_return": -0.1}
    assert _evaluate_single(_chrom(), _const(losing), min_trades=50).mtm_daily_returns is None
    few = {**base, "total_trades": 10}
    assert _evaluate_single(_chrom(), _const(few), min_trades=50).mtm_daily_returns is None

    ok_sym = {k: v for k, v in base.items() if k != "mtm_daily_returns"}
    worst = {**base, "symbol_mtm_daily": {"A": d, "B": d},
             "symbol_results": {"A": dict(ok_sym), "B": dict(ok_sym)}}  # fmt: skip
    kept = _evaluate_single(_chrom(), _const(worst), min_trades=50)
    assert kept.mtm_daily_returns is d and kept.symbol_mtm_daily == {"A": d, "B": d}
    one_loser = {**worst, "symbol_results": {"A": dict(ok_sym), "B": {**ok_sym, "total_return": -0.1}}}
    dropped = _evaluate_single(_chrom(), _const(one_loser), min_trades=50)
    assert dropped.adjusted_score > 0  # still ranks (soft score) ...
    assert dropped.mtm_daily_returns is None and dropped.symbol_mtm_daily is None  # ... keeps no series


# ---------------------------------------------------------------------------
# T4 / SF4: trial counts recorded; only raw N is used
# ---------------------------------------------------------------------------


def test_dsr_trials_recorded_not_used() -> None:
    """N_eff is recorded but never gates; the gate N is cumulative (yul7u.28).

    The old note logged ``p_value_at_cumulative_n`` while the gate used raw N;
    now the gate itself uses ``gate_n = run N + prior_n`` and the note records
    that rule. N_eff stays informational.
    """
    from vibe_quant.discovery.__main__ import _dsr_trials_note
    from vibe_quant.discovery.pipeline import DiscoveryResult

    pipe = _pipeline(prior_trials_fn=lambda s, e: 921)
    # Highly correlated retained series -> N_eff far below raw N
    rng = np.random.default_rng(11)
    common = rng.normal(0.0005, 0.01, 1264)
    for i in range(6):
        x = common + rng.normal(0, 0.001, 1264)
        series = DailyReturns(19_000, array("d", x.tobytes()))
        pipe._fitness_cache[f"k{i}"] = _fit(mtm_daily_returns=series)
    pipe._fitness_cache["flat"] = _fit()
    cand = _series(2.5)
    chrom = _chrom()
    pipe._validate_top_strategies([(chrom, _fit(mtm_daily_returns=cand))], 79)

    trials = pipe._dsr_trials
    assert trials is not None
    assert trials["raw"] == 79 and trials["non_flat"] == 6
    assert trials["prior_n"] == 921 and trials["gate_n"] == 1000
    neff_eig = trials["neff_eig_pr"]
    assert isinstance(neff_eig, float) and neff_eig < 79 / 3
    neff_mc = trials["neff_mean_corr"]
    assert isinstance(neff_mc, float) and 0 < neff_mc < 79
    sr, skew, kurt = daily_sharpe_inputs(cand.values)
    # Champion p_value uses gate_n (= run N + prior), NOT raw N and NOT N_eff.
    expected = DeflatedSharpeRatio().calculate(sr, 1000, 1264, min(skew, 0.0), max(kurt, 3.0))
    rec = pipe._dsr_records[chrom.uid]
    assert rec["trials"] == 1000 and rec["p_value"] == expected.p_value
    n_eff_p = DeflatedSharpeRatio().calculate(
        sr, int(neff_eig), 1264, min(skew, 0.0), max(kurt, 3.0)
    ).p_value
    assert rec["p_value"] != n_eff_p  # N_eff is recorded, never used

    result = DiscoveryResult(
        generations=[], top_strategies=[], total_candidates_evaluated=79, converged=False,
        convergence_generation=None, dsr_records=pipe._dsr_records, dsr_trials=trials,
    )  # fmt: skip
    note = _dsr_trials_note(result, [1, 2])
    assert note["gate_n"] == 1000 and note["prior_n"] == 921 and note["raw"] == 79
    assert note["prior_run_ids"] == [1, 2]
    assert note["n_rule"] == "cumulative_same_tf_overlapping_train"
    assert "cumulative_n" not in note and "p_value_at_cumulative_n" not in note


def test_prior_trials_sum_earlier_real_discovery_runs() -> None:
    import json
    import sqlite3

    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery.__main__ import _prior_discovery_trials

    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE backtest_runs (id INTEGER, run_mode TEXT, timeframe TEXT,"
        " start_date TEXT, end_date TEXT)"
    )
    conn.execute("CREATE TABLE backtest_results (run_id INTEGER, notes TEXT)")
    runs = [
        (1, "discovery", "4h", "2024-01-01", "2024-06-01", {"evaluated": 79}),
        (2, "discovery", "4h", "2024-01-01", "2024-06-01", {"evaluated": 500, "mock": True}),
        (3, "screening", "4h", "2024-01-01", "2024-06-01", {"evaluated": 40}),
        (4, "discovery", "4h", "2024-02-01", "2024-04-01", {"evaluated": 21}),
        # this run and later: excluded by b.id < run_id
        (5, "discovery", "4h", "2024-01-01", "2024-06-01", {"evaluated": 1000}),
    ]
    for rid, mode, tf, sd, ed, notes in runs:
        conn.execute("INSERT INTO backtest_runs VALUES (?, ?, ?, ?, ?)", (rid, mode, tf, sd, ed))
        conn.execute("INSERT INTO backtest_results VALUES (?, ?)", (rid, json.dumps(notes)))
    conn.execute("INSERT INTO backtest_results VALUES (?, ?)", (4, "not json"))
    state = StateManager()
    state._conn = conn  # noqa: SLF001 -- skip get_connection/init_schema
    n, ids = _prior_discovery_trials(state, 5, "4h", ("2024-03-01", "2024-09-01"))
    assert n == 100 and ids == [1, 4]


# ---------------------------------------------------------------------------
# vibe-quant-f9zzq: the realized (non-MTM) daily series left discovery
# ---------------------------------------------------------------------------


def test_fitness_result_has_no_realized_daily_series() -> None:
    """Only the market-valued series remains on ``FitnessResult`` (f9zzq).

    The realized-balance series fed nothing but the pre-y9ywf DSR gate, so its
    fields and producers were removed; the MTM pair stays.
    """
    import dataclasses

    names = {f.name for f in dataclasses.fields(FitnessResult)}
    assert "daily_returns" not in names
    assert "symbol_daily" not in names
    assert "mtm_daily_returns" in names
    assert "symbol_mtm_daily" in names
