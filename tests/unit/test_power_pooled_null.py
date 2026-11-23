"""G-pooled: the 6-symbol pooled selection null (scripts/research/power/neff_null.py --symbols 6).

Job 20261122-edge, task C-T4 (vibe-quant-clmmc.5), design ``arch-gates-2.md`` §5 G-pooled,
chief ruling R-gc4-1 (board 8561): the harness scores six coin series per candidate through
the PRODUCTION pooled reduction and gate chain, never a re-implementation. Nothing here runs
the binding 400-rep grid; the main() runs are tiny and draw the quick stream.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research" / "power"))

import gatechain  # noqa: E402
import layer1  # noqa: E402
import neff_null  # noqa: E402
import synth  # noqa: E402

from vibe_quant.discovery.backtest_fn import NTBacktestFn  # noqa: E402
from vibe_quant.discovery.fitness import _evaluate_single, passes_train_gate  # noqa: E402
from vibe_quant.discovery.gate_chain import (  # noqa: E402
    GateChain,
    GateEvidence,
    TrialBudget,
    holdout_min_trades,
    holdout_reasons,
    metrics_to_holdout_result,
)
from vibe_quant.discovery.genome import generate_random_chromosome  # noqa: E402
from vibe_quant.discovery.operators import (  # noqa: E402
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)

SYMS: list[str] = list(neff_null.POOLED_SYMBOLS)


def _closes() -> dict[str, dict[int, float]]:
    """Deterministic drift-free 1d closes for the six coins, covering both plans."""
    first = int(np.datetime64("2020-12-01", "D").astype("int64"))
    out: dict[str, dict[int, float]] = {}
    for k, sym in enumerate(SYMS):
        rng = np.random.default_rng(1000 + k)
        close, series = 100.0, {}
        for i, ret in enumerate(rng.normal(0.0, 0.03, 2200)):
            series[first + i] = close
            close *= 1.0 + float(ret)
        out[sym] = series
    return out


def test_grid_seeds_and_the_quick_stream_guard() -> None:
    # design §5: W3/W6 x coin_rho {0.3, 0.6} x rho {0.0, 0.5}, the index is the seed's cfg
    grid = neff_null.pooled_configs()
    assert [(c.index, c.key) for c in grid] == [
        (0, "W3:coin_rho=0.3:rho=0.0"),
        (1, "W3:coin_rho=0.3:rho=0.5"),
        (2, "W3:coin_rho=0.6:rho=0.0"),
        (3, "W3:coin_rho=0.6:rho=0.5"),
        (4, "W6:coin_rho=0.3:rho=0.0"),
        (5, "W6:coin_rho=0.3:rho=0.5"),
        (6, "W6:coin_rho=0.6:rho=0.0"),
        (7, "W6:coin_rho=0.6:rho=0.5"),
    ]
    assert neff_null.pooled_seq(grid[5], 17, 0x900D).entropy == [20261122, 0x900D, 5, 17]
    tag = neff_null.pooled_stream_tag
    # only the exact binding sizes draw the binding / top-up streams
    assert tag(reps=400, candidates=167, quick=False) == 0x900D
    assert tag(reps=1600, candidates=167, quick=False, topup=True) == 0x900E
    quick = neff_null.POOLED_QUICK_TAG
    assert quick not in (0x900D, 0x900E)
    assert tag(reps=400, candidates=167, quick=True) == quick
    assert tag(reps=399, candidates=167, quick=False) == quick
    assert tag(reps=401, candidates=167, quick=False) == quick
    assert tag(reps=400, candidates=166, quick=False) == quick
    assert tag(reps=400, candidates=167, quick=False, topup=True) == quick
    assert tag(reps=1600, candidates=167, quick=False) == quick
    # the grid is part of "binding": a --coin-rho subset at the binding sizes is quick too
    assert tag(reps=400, candidates=167, quick=False, coin_rhos=[0.3]) == quick
    assert tag(reps=400, candidates=167, quick=False, coin_rhos=[0.6]) == quick
    assert tag(reps=400, candidates=167, quick=False, coin_rhos=[0.3, 0.6]) == 0x900D
    assert neff_null.pooled_grid_rhos([0.6]) == [0.6]
    assert neff_null.pooled_grid_rhos([0.6, 0.3]) == [0.3, 0.6]


def _corrs(rows: list[np.ndarray]) -> np.ndarray:
    return np.corrcoef(np.array(rows))


@pytest.mark.parametrize(("index", "coin_rho", "rho"), [(2, 0.6, 0.0), (5, 0.3, 0.5)])
def test_pooled_draws_carry_the_coin_and_candidate_correlations(
    index: int, coin_rho: float, rho: float
) -> None:
    # One candidate's six coin series correlate at coin_rho; two candidates' series of the
    # same coin at rho (the shock). The series reach production via pooled_entry (tested below).
    cfg = neff_null.pooled_configs()[index]
    assert (cfg.coin_rho, cfg.rho) == (coin_rho, rho)
    series = [train for _c, train, _h in neff_null.pooled_draws(cfg, 0, 0x90A1, 16)]
    active = series[0][0] != 0.0
    within = [
        c[np.triu_indices(6, 1)].mean() for c in (_corrs([r[active] for r in s]) for s in series)
    ]
    across = [
        _corrs([s[coin][active] for s in series])[np.triu_indices(16, 1)].mean()
        for coin in range(6)
    ]
    assert float(np.mean(within)) == pytest.approx(coin_rho, abs=0.05)
    assert float(np.mean(across)) == pytest.approx(rho, abs=0.05)


def _coin(mask: np.ndarray, mean: float, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(len(mask))
    x[mask] = mean + 0.012 * rng.standard_normal(int(mask.sum()))
    return x


def _window_bt(
    sched_returns: np.ndarray, first_day: int, entries: Any, hold: int, ws: str, we: str
) -> dict[str, Any]:
    gen = synth.grid_window(sched_returns, first_day, entries, hold, ws, we)
    return {**gen.bt, "trade_pnls": tuple(t.pnl for t in gen.trades)}


def test_pooled_null_path_matches_a_direct_production_pooled_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # R-gc4-1: on a fixed 6-series input the harness and the production pooled path
    # (NTBacktestFn.__call__ split + GateChain.screen + GateChain.validate) agree on every
    # number and every verdict. Sign floor edge at N = 6: 4 positive coins pass, 3 do not.
    window = "W3"
    base, gen = neff_null._pooled_layout(window)
    plan = base.plan
    hs, he = plan.holdout
    h_entries = synth.place_entries((plan.holdout,), gen.n, gen.hold)
    h_first = synth.day_index(hs)
    train_mask = base.returns != 0.0
    hold_mask = synth.holdout_returns(gen) != 0.0
    rng = np.random.default_rng(20261123)
    means = {
        "four_up": [0.006, 0.006, 0.006, 0.006, -0.004, -0.004],
        "three_up": [0.012, 0.012, 0.012, -0.004, -0.004, -0.004],
        "flat": [0.0] * 6,
    }
    inputs = {
        name: (
            [_coin(train_mask, m, rng) for m in ms],
            [_coin(hold_mask, m, rng) for m in ms],
        )
        for name, ms in means.items()
    }
    chroms = {name: generate_random_chromosome(random.Random(k)) for k, name in enumerate(means)}
    entries = [neff_null.pooled_entry(window, chroms[name], *inputs[name]) for name in means]
    closes = _closes()
    harness = neff_null.pooled_chain(window, entries, closes)

    cfg = gatechain.settings_for(window).config
    table: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for name, (train, holdout) in inputs.items():
        uid = chroms[name].uid
        for sym, x, xh in zip(SYMS, train, holdout, strict=True):
            for ws, we in plan.eval_windows:
                table[(uid, sym, ws, we)] = _window_bt(
                    x, base.first_day, base.entries, base.hold, ws, we
                )
            table[(uid, sym, hs, he)] = _window_bt(xh, h_first, h_entries, gen.hold, hs, he)

    def fake_run(chrom: Any, ws: str, we: str, symbols: list[str] | None = None) -> dict[str, Any]:
        assert symbols is not None and len(symbols) == 1
        return dict(table[(chrom.uid, symbols[0], ws, we)])

    train_fn = NTBacktestFn(
        SYMS,
        cfg.timeframe,
        cfg.start_date,
        cfg.end_date,
        windows=list(plan.eval_windows),
        min_trades=cfg.min_trades,
        symbol_agg="pooled",
    )
    holdout_fn = NTBacktestFn(
        SYMS, cfg.timeframe, hs, he, min_trades=cfg.min_trades, symbol_agg="pooled"
    )
    monkeypatch.setattr(train_fn, "_run_single", fake_run)
    monkeypatch.setattr(holdout_fn, "_run_single", fake_run)
    fits = {
        name: _evaluate_single(chroms[name], train_fn, None, cfg.min_trades, cfg.timeframe)
        for name in means
    }
    for entry, name in zip(entries, means, strict=True):
        fit = fits[name]
        assert entry.fitness.adjusted_score == fit.adjusted_score
        assert entry.fitness.total_return == fit.total_return
        assert entry.fitness.sharpe_ratio == fit.sharpe_ratio
        assert entry.fitness.total_trades == fit.total_trades == 900
        assert entry.fitness.pooled_symbol_stats == fit.pooled_symbol_stats
        assert list(cast("Any", entry.fitness.mtm_daily_returns).values) == list(
            cast("Any", fit.mtm_daily_returns).values
        )
        direct_holdout = holdout_fn(chroms[name])
        for key in ("total_return", "sharpe_ratio", "total_trades"):
            assert entry.holdout_bt[key] == direct_holdout[key]

    pcfg = replace(
        cfg,
        symbols=SYMS,
        symbol_agg="pooled",
        dsr_rule="q95",
        cross_window_months=[],
        wfa_oos_step_days=0,
    )
    chain = GateChain(
        pcfg,
        GateEvidence(
            train_fn,
            True,
            holdout_backtest_fn=holdout_fn,
            buy_hold_closes_fn=lambda symbols: {s: closes[s] for s in symbols},
        ),
    )
    ranked = sorted(fits.items(), key=lambda kv: kv[1].adjusted_score, reverse=True)
    picks = [(chroms[n], f) for n, f in ranked if passes_train_gate(f, cfg.min_trades)]
    budget = TrialBudget(3, {f"{i:06d}": fits[n] for i, n in enumerate(means)})
    outcome = chain.validate(chain.screen(picks, budget))
    direct = {chrom.uid for chrom, _fit in outcome.survivors}

    assert harness["passed"] == [True, False, False]
    assert direct == {chroms["four_up"].uid}
    stages = {str(r["uid"]): str(r["stage"]) for r in chain.rejections}
    assert stages[chroms["three_up"].uid] == "pooled_symbol_train"
    assert cast("dict[str, int]", harness["stages"])["pooled_symbol_train"] == 1


def _pinned_chromosome() -> StrategyChromosome:
    return StrategyChromosome(
        entry_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 55.0, None)],
        exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 45.0, None)],
        stop_loss_pct=2.0,
        take_profit_pct=3.0,
        direction=Direction.LONG,
    )  # fmt: skip


def _trending_closes(growth: float, sd: float) -> dict[str, dict[int, float]]:
    """Six coins with the SAME steady drift, so the EW bench's Sharpe dwarfs a small edge."""
    first = int(np.datetime64("2020-12-01", "D").astype("int64"))
    rng = np.random.default_rng(7)
    out: dict[str, dict[int, float]] = {}
    for sym in SYMS:
        close, series = 100.0, {}
        for i, ret in enumerate(growth + sd * rng.standard_normal(2200)):
            series[first + i] = close
            close *= 1.0 + float(ret)
        out[sym] = series
    return out


def test_the_buy_and_hold_gate_decides_a_candidate_that_passes_every_other_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # S2 (rev-gc4-1): a candidate that clears the train gate, the sign floor, the guardrails
    # and the holdout but whose pooled daily Sharpe is BELOW the six-coin EW buy-and-hold bar
    # must be rejected at stage buy_hold with the benchmark reason. Dropping the
    # apply_buy_hold step would let this candidate pass, so the test pins the gate itself.
    window = "W3"
    base, gen = neff_null._pooled_layout(window)
    plan = base.plan
    hs, he = plan.holdout
    h_entries = synth.place_entries((plan.holdout,), gen.n, gen.hold)
    h_first = synth.day_index(hs)
    train_mask = base.returns != 0.0
    hold_mask = synth.holdout_returns(gen) != 0.0
    rng = np.random.default_rng(20261123)
    means = [0.006, 0.006, 0.006, 0.006, 0.004, 0.004]
    train = [_coin(train_mask, m, rng) for m in means]
    holdout = [_coin(hold_mask, m, rng) for m in means]
    chrom = _pinned_chromosome()
    entry = neff_null.pooled_entry(window, chrom, train, holdout)
    closes = _trending_closes(0.005, 0.002)

    cfg = gatechain.settings_for(window).config
    table: dict[tuple[str, str], dict[str, Any]] = {}
    for sym, x in zip(SYMS, train, strict=True):
        for ws, we in plan.eval_windows:
            table[(sym, ws)] = _window_bt(x, base.first_day, base.entries, base.hold, ws, we)
    for sym, xh in zip(SYMS, holdout, strict=True):
        table[(sym, hs)] = _window_bt(xh, h_first, h_entries, gen.hold, hs, he)

    def fake_run(chrom_: Any, ws: str, we: str, symbols: list[str] | None = None) -> dict[str, Any]:
        assert symbols is not None and len(symbols) == 1
        return dict(table[(symbols[0], ws)])

    pcfg = replace(
        cfg,
        symbols=SYMS,
        symbol_agg="pooled",
        dsr_rule="q95",
        cross_window_months=[],
        wfa_oos_step_days=0,
    )
    train_fn = NTBacktestFn(
        SYMS,
        cfg.timeframe,
        cfg.start_date,
        cfg.end_date,
        windows=list(plan.eval_windows),
        min_trades=cfg.min_trades,
        symbol_agg="pooled",
    )
    holdout_fn = NTBacktestFn(
        SYMS, cfg.timeframe, hs, he, min_trades=cfg.min_trades, symbol_agg="pooled"
    )

    # the fixture's premises: every gate BEFORE buy-hold lets it through, holdout included
    assert passes_train_gate(entry.fitness, cfg.min_trades)
    assert (
        holdout_reasons(
            metrics_to_holdout_result(entry.holdout_bt),
            holdout_min_trades(pcfg),
            pcfg.holdout_min_sharpe,
        )
        == []
    )

    harness = neff_null.pooled_chain(window, [entry], closes)
    assert harness["passed"] == [False]
    assert harness["stages"] == {"buy_hold": 1}

    # the direct production chain rejects at the same stage with the benchmark reason, so the
    # harness's verdict is the production one, and it is buy-hold that decides it
    monkeypatch.setattr(train_fn, "_run_single", fake_run)
    monkeypatch.setattr(holdout_fn, "_run_single", fake_run)
    chain = GateChain(
        pcfg,
        GateEvidence(
            train_fn,
            True,
            holdout_backtest_fn=holdout_fn,
            buy_hold_closes_fn=lambda symbols: {s: closes[s] for s in symbols},
        ),
    )
    picks = layer1.pick_pool([(chrom, entry.fitness)], "all")
    outcome = chain.validate(chain.screen(picks, TrialBudget(1, {"000000": entry.fitness})))
    assert outcome.survivors == []
    stages = [str(r["stage"]) for r in chain.rejections]
    assert stages == ["buy_hold"]
    reasons = cast("list[str]", chain.rejections[0]["reasons"])
    assert any("EW B&H" in r and "+".join(sorted(SYMS)) in r for r in reasons)

    # the counter-edge: a flat bench (no drift) lets the same candidate through buy-hold, so
    # the rejection above is the benchmark's doing, not a broken fixture
    flat = neff_null.pooled_chain(window, [entry], _trending_closes(0.0, 0.002))
    assert flat["passed"] == [True]
    assert "buy_hold" not in cast("dict[str, int]", flat["stages"])


def _pooled_args(out: Path, workers: int) -> list[str]:
    return [
        "--symbols",
        "6",
        "--quick",
        "--reps",
        "2",
        "--candidates",
        "6",
        "--coin-rho",
        "0.6",
        "--workers",
        str(workers),
        "--out",
        str(out),
    ]


@pytest.mark.slow(reason="three tiny pooled quick runs compared byte for byte")
def test_pooled_quick_run_is_deterministic_and_never_a_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The 6-symbol null determinism test (design C-T4 row): identical bytes across reruns and
    # for 1 vs 2 workers; a quick run draws the quick stream and exits 2 (MARGINAL).
    monkeypatch.setattr(neff_null, "load_pooled_closes", lambda _catalog: _closes())
    first, second, third = (tmp_path / f"{n}.json" for n in "abc")
    assert neff_null.main(_pooled_args(first, 1)) == 2
    assert neff_null.main(_pooled_args(second, 1)) == 2
    assert neff_null.main(_pooled_args(third, 2)) == 2
    assert first.read_bytes() == second.read_bytes() == third.read_bytes()
    out = capsys.readouterr().out
    assert "G-pooled neff (6 symbols, pick=all, q95): [" in out
    assert "(seed tag 0x90A1)" in out

    report = cast("dict[str, Any]", json.loads(first.read_text()))
    assert report["seed_tag"] == neff_null.POOLED_QUICK_TAG
    assert report["symbols"] == SYMS and report["symbol_agg"] == "pooled"
    assert (report["pick"], report["dsr_rule"]) == ("all", "q95")
    assert [c["key"] for c in report["configs"]] == [
        "W3:coin_rho=0.6:rho=0.0",
        "W3:coin_rho=0.6:rho=0.5",
        "W6:coin_rho=0.6:rho=0.0",
        "W6:coin_rho=0.6:rho=0.5",
    ]
    assert all(c["reps"] == 2 and c["candidates"] == 6 for c in report["configs"])
    assert report["verdict"]["status"] == "MARGINAL" and report["topup_configs"] == []
    assert "uid" not in first.read_text()

    with pytest.raises(SystemExit) as refused:
        neff_null.main([*_pooled_args(tmp_path / "x.json", 1), "--pick", "top5"])
    assert refused.value.code == 2


# ---------------------------------- S1: any non-binding grid draws the quick stream


def _config_row(key: str, count: int, reps: int) -> dict[str, Any]:
    """One pooled config row exactly as ``summarize_pooled`` shapes it (print reads these)."""
    return {
        "key": key,
        "reps": reps,
        "count_ev": count,
        "picks_per_rep": 0.0,
        "median_run_n_eff": None,
        "median_k_eff": None,
        "rejections": {},
    }


def _binding_args(out: Path | None, *, coin_rhos: list[str]) -> list[str]:
    args = ["--symbols", "6", "--reps", "400", "--candidates", "167", "--workers", "1"]
    args += ["--coin-rho", *coin_rhos]
    return args if out is None else [*args, "--out", str(out)]


def test_a_subset_grid_at_binding_size_is_forced_onto_the_quick_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # S1 (rev-gc4-1): only the FULL pre-registered grid (both coin_rhos) at the binding sizes
    # may draw 0x900D and write the binding file. One --coin-rho is a quick-stream run: exit 2,
    # never a verdict, its own default output name, and the report lists what actually ran.
    runs: list[tuple[list[str], int]] = []
    written: list[dict[str, Any]] = []
    paths: list[Path] = []

    def fake_run_pooled(
        cfgs: Any, reps: int, workers: int, closes: Any, *, candidates: int, tag: int
    ) -> list[dict[str, Any]]:
        runs.append(([c.key for c in cfgs], tag))
        return [_config_row(c.key, 0, reps) for c in cfgs]

    def fake_write(report: Any, out: Path) -> None:
        written.append(dict(report))
        paths.append(Path(out))

    monkeypatch.setattr(neff_null, "run_pooled", fake_run_pooled)
    monkeypatch.setattr(neff_null, "load_pooled_closes", lambda _catalog: _closes())
    monkeypatch.setattr(neff_null, "write_report", fake_write)

    subset = _binding_args(tmp_path / "subset.json", coin_rhos=["0.6"])
    # (a) a subset at the binding sizes: quick tag, exit 2, and the run's own grid reported
    assert neff_null.main(subset) == 2
    keys, tag = runs[-1]
    assert tag == neff_null.POOLED_QUICK_TAG
    assert keys == [
        "W3:coin_rho=0.6:rho=0.0",
        "W3:coin_rho=0.6:rho=0.5",
        "W6:coin_rho=0.6:rho=0.0",
        "W6:coin_rho=0.6:rho=0.5",
    ]
    assert written[-1]["coin_rhos"] == [0.6]
    assert written[-1]["seed_tag"] == neff_null.POOLED_QUICK_TAG
    assert written[-1]["verdict"]["status"] == "MARGINAL"
    assert paths[-1] == tmp_path / "subset.json"

    # (b) a subset with NO --out never lands on the binding default name
    assert neff_null.main(_binding_args(None, coin_rhos=["0.3"])) == 2
    assert runs[-1][1] == neff_null.POOLED_QUICK_TAG
    assert paths[-1] == Path(neff_null.GPOOLED_DIR) / "neff_null6.quick.json"
    assert written[-1]["coin_rhos"] == [0.3]

    # (c) the FULL grid at the binding reps is the only binding stream (stream choice and the
    # default name only: nothing here runs 400 replicates)
    neff_null.main(["--symbols", "6", "--reps", "400", "--candidates", "167", "--workers", "1"])
    assert runs[-1][1] == neff_null.POOLED_TAG
    assert paths[-1] == Path(neff_null.GPOOLED_DIR) / "neff_null6.json"
    assert written[-1]["coin_rhos"] == [0.3, 0.6]
