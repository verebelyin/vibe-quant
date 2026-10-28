"""Power test T2: the L1 statistical power grid (scripts/research/power/layer1.py).

Job 20261017-power-paper; contract rows R2, R4, R18, R19, R33 of
``data/swarm/20261017-power-paper/design-power.md``.

``layer1`` runs the whole L1 grid (S x window x trade count x N) through the REAL
chain, prints the pre-registered verdict lines and writes a JSON. Every literal
asserted here is the design's/chief's own number; nothing is recomputed by hand.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research" / "power"))

import gatechain  # noqa: E402
import layer1  # noqa: E402
import synth  # noqa: E402

from vibe_quant.discovery.buyhold import buy_hold_gate  # noqa: E402
from vibe_quant.discovery.distance import chromosome_distance  # noqa: E402
from vibe_quant.discovery.fitness import FitnessResult  # noqa: E402
from vibe_quant.discovery.operators import (  # noqa: E402
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)
from vibe_quant.overfitting.dsr import DeflatedSharpeRatio  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG = REPO_ROOT / "data" / "catalog"

#: R2: the design §2 critical-Sharpe table, annualized with sqrt(365), 4 dp.
R2_TABLE: dict[int, dict[int, float]] = {
    875: {1: 1.0638, 7: 1.9619, 9: 2.0488, 11: 2.1145, 12: 2.1422, 167: 2.8175, 22121: 3.6905},
    1689: {1: 0.7652, 7: 1.4108, 9: 1.4732, 11: 1.5204, 12: 1.5403, 167: 2.0253, 22121: 2.6521},
}

#: R18: the design §2 power table (kurtosis 3 = the upper bound), 4 dp.
R18_POWER_TABLE: dict[int, dict[int, float]] = {
    875: {1: 0.5834, 7: 0.1194, 9: 0.0947, 11: 0.0787, 12: 0.0726, 167: 0.0062, 22121: 0.0001},
    1689: {1: 0.8249, 7: 0.3253, 9: 0.2786, 11: 0.2456, 12: 0.2324, 167: 0.0381, 22121: 0.0009},
}

#: R4: the real BTC train buy-and-hold daily Sharpe on the chained day set.
R4_BAR: dict[str, float] = {
    "W3": 0.07840149276712642,
    "W6": 0.041810433685231255,
}


# --------------------------------------------------------------------------- R2


def test_analytic_dsr_critical_sharpe() -> None:
    # R2: the analytic critical S is the bisection root of DeflatedSharpeRatio at
    # p = 0.05, skew 0, kurtosis 3, annualized by sqrt(365). The design §2 table
    # is the literal contract (4 dp).
    for T, by_n in R2_TABLE.items():
        for N, expected in by_n.items():
            got = layer1.critical_sharpe_ann(T, N)
            assert got == pytest.approx(expected, abs=5e-5), (T, N, got, expected)
            # the bisection root really is p = 0.05 on the real repo function
            p = DeflatedSharpeRatio(0.05).calculate(
                observed_sharpe=layer1.critical_daily_sharpe(T, N),
                num_trials=N,
                num_observations=T,
                skewness=0.0,
                kurtosis=3.0,
            ).p_value
            assert p == pytest.approx(0.05, abs=1e-9), (T, N, p)


# ------------------------------------------------------------------- R18 (power)


def test_analytic_power_matches_design_table() -> None:
    # R18: the analytic DSR-alone power at kurtosis 3 reproduces the design §2
    # power table exactly (that table is the normal-tail UPPER bound).
    for T, by_n in R18_POWER_TABLE.items():
        for N, expected in by_n.items():
            got = layer1.analytic_power(1.2, T, N, 0.4, kurtosis=3.0)
            assert got == pytest.approx(expected, abs=5e-5), (T, N, got, expected)
    # the generator's own kurtosis k = 9/f lowers the power below the k = 3 bound
    assert layer1.analytic_power(1.2, 875, 167, 0.4) < layer1.analytic_power(
        1.2, 875, 167, 0.4, kurtosis=3.0
    )


def _r18_report(empirical: float, S: float, N: int, f_real: float, reps: int) -> dict[str, Any]:
    """A one-cell report as ``check_r18`` reads it."""
    band = {
        "kurt_9f": layer1.analytic_power(S, 875, N, f_real),
        "kurt3": layer1.analytic_power(S, 875, N, f_real, kurtosis=3.0),
        "crit_9f": 0.0,
        "crit3": 0.0,
    }
    return {
        "reps": reps,
        "cells": [
            {
                "window": "W3",
                "n": 150,
                "S": S,
                "dsr_alone": {str(N): empirical},
                "analytic": {str(N): band},
            }
        ],
    }


def test_check_r18_is_a_diagnostic_and_exit_is_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # W-R5 (chief ruling): --check prints per-cell FAIL lines plus the summary
    # "R18: <k> of <m> cells outside band" and ALWAYS exits 0 -- the check is a
    # sanity diagnostic, not a verdict gate. One reading is pushed out of band.
    def cell(window: str, S: float, f_real: float) -> dict[str, Any]:
        analytic = {
            str(N): {
                "kurt_9f": layer1.analytic_power(S, 875, N, f_real),
                "kurt3": layer1.analytic_power(S, 875, N, f_real, kurtosis=3.0),
                "crit_9f": 0.0,
                "crit3": 0.0,
            }
            for N in gatechain.DSR_TRIALS
        }
        return {
            "window": window,
            "n": 150,
            "S": S,
            "f_real": f_real,
            "floor": 0.0,
            "marginals": {name: 0.0 for name in layer1.MARGINAL_NAMES},
            "chain_recent": {str(N): 0.0 for N in gatechain.DSR_TRIALS},
            "chain_recent_minus_bh": {str(N): 0.0 for N in gatechain.DSR_TRIALS},
            "chain_recent_minus_dsr": {str(N): 0.0 for N in gatechain.DSR_TRIALS},
            "chain_recent_minus_dsr_bh": 0.0,
            # on the k = 3 edge of the band -> inside for every N
            "dsr_alone": {str(N): analytic[str(N)]["kurt3"] for N in gatechain.DSR_TRIALS},
            "analytic": analytic,
            "kurtosis_applied": {"mean": 17.0, "sd": 8.0, "n": 500},
            "kurtosis_9_over_f": 9.0 / f_real,
            "inposition_lag1": 0.0,
        }

    cells = [
        cell("W3", 1.2, 0.34285714285714286),
        cell("W3", 0.0, 0.34285714285714286),
        cell("W6", 1.2, 0.44404973357015987),
        cell("W6", 0.0, 0.44404973357015987),
    ]
    cells[1]["dsr_alone"]["1"] = 0.5  # one reading out of band
    report = layer1.build_report(
        cells, [{"window": "W3", "rho": 0.0, "fp_rate": 0.0}], reps=500, selection_reps=200
    )
    path = tmp_path / "l1.json"
    layer1.write_report(report, path)

    rc = layer1.main(["--check", str(path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "R18 FAIL W3 n=150 S=0.0 N=1:" in out
    assert "R18: 1 of 36 cells outside band" in out
    # the verdict lines a researcher reads come out of the same report: V1(a),
    # V2's nulls, V3's lowest marginal, S50 and the neff info column
    assert "V1(a) chain_recent: W3@N=22121=" in out
    assert " W6@N=167=" in out
    assert "V2 nulls: single max=" in out
    assert "V3 W3 (N=22121): lowest marginal" in out
    assert "S50 W3: N=1:none N=11:none" in out  # JSON round trip sorts the N keys
    assert "neff info W3 (S=1.2, n=150): N=11:0.0000 N=12:0.0000 N=7:0.0000 N=9:0.0000" in out


def test_main_does_not_leak_the_vibe_quant_log_level(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # verifier round 3: main() turns vibe_quant INFO down for the grid, but the
    # silencing must not outlive the call -- a later test in the same process
    # (test_telegram_start_gate captures a vibe_quant INFO line) would record
    # nothing. Level in == level out, on the grid path and the --check path, and
    # an INFO record from a vibe_quant child logger is still captured afterwards.
    # The grid is trimmed to the two verdict cells (plus one selection-null rho):
    # what is asserted is the logger level, not the grid's width.
    if not (CATALOG / "data").exists():
        pytest.skip("data/catalog absent")
    logger = logging.getLogger("vibe_quant")
    previous = logger.level
    verdict_cells = [
        c for c in layer1.cells() if c.n == layer1.RESULTS_N and c.S == layer1.RESULTS_S
    ]
    monkeypatch.setattr(layer1, "cells", lambda: verdict_cells)
    monkeypatch.setattr(layer1, "SELECTION_RHOS", (0.0,))

    args = [
        "--quick",
        "--reps",
        "2",
        "--selection-reps",
        "1",
        "--workers",
        "1",
        "--catalog",
        str(CATALOG),
        "--out",
        str(tmp_path / "q.json"),
    ]
    assert layer1.main(args) == 0
    assert logger.level == previous
    out = tmp_path / "q.json"
    assert out.exists()

    caplog.clear()
    with caplog.at_level(logging.INFO):
        logging.getLogger("vibe_quant.discovery.fitness").info("LAYER1-INFO-VISIBLE")
    assert "LAYER1-INFO-VISIBLE" in caplog.text

    assert layer1.main(["--check", str(out)]) == 0
    assert logger.level == previous


def test_check_r18_band_edges() -> None:
    # R18/§6: at S = 0, N = 1 the band is 0.05 +- 3*sqrt(0.05*0.95/reps) -- the
    # design's [0.0208, 0.0792] at 500 reps. W-R1 clamps p to [1/reps, 1-1/reps]
    # and includes both edges.
    reps, S, N, f_real = 500, 0.0, 1, 0.4
    lo = layer1.analytic_power(S, 875, N, f_real)
    hi = layer1.analytic_power(S, 875, N, f_real, kurtosis=3.0)
    band_lo = lo - 3.0 * layer1._binomial_se(lo, reps)
    band_hi = hi + 3.0 * layer1._binomial_se(hi, reps)
    assert band_lo == pytest.approx(0.0208, abs=1.5e-3)
    assert band_hi == pytest.approx(0.0792, abs=1.5e-3)
    assert band_lo == pytest.approx(0.01974133308099922, rel=1e-9)
    assert band_hi == pytest.approx(0.07907268169804006, rel=1e-9)

    def passes(empirical: float) -> bool:
        return layer1.check_r18(_r18_report(empirical, S, N, f_real, reps))[0]

    assert passes(band_lo) is True  # inclusive lower edge
    assert passes(band_lo - 1e-9) is False
    assert passes(band_hi) is True  # inclusive upper edge
    assert passes(band_hi + 1e-9) is False
    assert passes(0.05) is True

    ok, failures = layer1.check_r18(_r18_report(0.5, S, N, f_real, reps))
    assert ok is False and len(failures) == 1 and "W3 n=150 S=0.0 N=1" in failures[0]

    # W-R1: the SE's plug-in p is clamped to [1/reps, 1 - 1/reps]. At N = 22121 and
    # S = 0 the analytic power is far below 1/reps, so the clamp binds and the SE is
    # the clamped one -- a raw p would give a much narrower (never-reachable) band.
    tiny = layer1.analytic_power(0.0, 875, 22121, 0.4)
    assert 0.0 < tiny < 1.0 / reps
    assert layer1._clamp_probability(tiny, reps) == pytest.approx(1.0 / reps)
    assert layer1._clamp_probability(0.999, reps) == pytest.approx(1.0 - 1.0 / reps)
    assert layer1._clamp_probability(0.05, reps) == pytest.approx(0.05)
    assert layer1._binomial_se(tiny, reps) == pytest.approx(
        math.sqrt((1.0 / reps) * (1.0 - 1.0 / reps) / reps), rel=1e-12
    )


# ------------------------------------------------------------------- R4 (bar)


def _closes() -> dict[str, dict[int, float]]:
    """A deterministic BTC-like 1d close path covering both plan train ranges."""
    first = int(np.datetime64("2021-01-01", "D").astype("int64"))
    rng = np.random.default_rng(20261017)
    close = 20_000.0
    series: dict[int, float] = {}
    for i, ret in enumerate(rng.normal(0.0005, 0.03, 1800)):
        series[first - 1 + i] = close
        close *= 1.0 + float(ret)
    return {"BTCUSDT": series}


def test_bh_bar_btc_windows() -> None:
    # R4: the real BTC 1d closes over the chained train day set (skip if the
    # catalog is absent). Two unrelated inputs: W3 and W6.
    if not (CATALOG / "data").exists():
        pytest.skip("data/catalog absent")
    closes = layer1.load_closes(CATALOG)
    for window, expected in R4_BAR.items():
        assert layer1.buy_hold_bar(window, closes) == pytest.approx(expected, rel=1e-12)

    # the bar is what the buy-and-hold GATE builds: read the record of the synth
    # series the gates see, and confirm its day set is layer1's
    for window, S, n in (("W3", 0.0, 150), ("W6", 1.2, 70)):
        series = synth.evaluate(window, n, S, 11).mtm_daily_returns
        assert series is not None
        assert series.first_day == layer1.chained_first_day(window)
        _passed, record, _reasons = buy_hold_gate(series, "long", closes, ["BTCUSDT"])
        assert record["bh_sharpe_daily"] == pytest.approx(R4_BAR[window], rel=1e-12)


# --------------------------------------------------------------- R19 (seeds)


def test_quick_run_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # R19: SeedSequence([base, w, s, n]).spawn(reps), so the --quick JSON (minus
    # the wall clock) is identical for workers 1 vs 2 (two separate runs, so also a rerun).
    # The full grid is the contract's 30 cells; the run itself uses the two verdict
    # cells and one selection-null rho, because the seed stream -- not the grid's
    # width -- is what R19 pins.
    closes = _closes()
    full_grid = layer1.cells()
    assert len(full_grid) == 30
    grid = [c for c in full_grid if c.n == layer1.RESULTS_N and c.S == layer1.RESULTS_S]
    assert {(c.window, c.S) for c in grid} == {("W3", 1.2), ("W6", 1.2)}
    monkeypatch.setattr(layer1, "SELECTION_RHOS", (0.0,))

    def write(workers: int, name: str) -> dict[str, Any]:
        cell_results, null_results = layer1.run_grid(
            grid, 2, workers, closes, selection_reps=1
        )
        report = layer1.build_report(
            cell_results, null_results, reps=2, selection_reps=1
        )
        target = tmp_path / name
        layer1.write_report(report, target)
        return dict(json.loads(target.read_text()))

    raw = write(1, "w1.json")
    serial = cast("dict[str, Any]", layer1.strip_timestamps(raw))
    parallel = layer1.strip_timestamps(write(2, "w2.json"))

    assert json.dumps(serial, sort_keys=True) == json.dumps(parallel, sort_keys=True)
    # non-trivial content, and the only non-deterministic key is the wall clock
    assert len(serial["cells"]) == 2
    assert serial["reps"] == 2
    assert "generated_at" in raw and "generated_at" not in serial
    assert set(serial) == set(raw) - {"generated_at"}
    assert serial["verdicts"]["v1"]["production_n"] == {"W3": 22121, "W6": 167}

    # W-R5: every cell records the R18 reading, the applied-kurtosis mean/sd,
    # 9/f, and the in-position indicator's lag-1 autocorrelation. Two unrelated
    # inputs (W3 and W6) with the literal values of the seed-independent schedule.
    by_cell = {(c["window"], c["n"], c["S"]): c for c in serial["cells"]}
    assert layer1.inposition_lag1_autocorr("W3", 150) == pytest.approx(0.242012422360248, rel=1e-12)
    assert layer1.inposition_lag1_autocorr("W6", 150) == pytest.approx(0.6419126200920452, rel=1e-12)
    for window, expected in (("W3", 0.242012422360248), ("W6", 0.6419126200920452)):
        entry = by_cell[(window, 150, 1.2)]
        assert entry["inposition_lag1"] == pytest.approx(expected, rel=1e-12)
        assert entry["kurtosis_applied"]["n"] == 2
        assert entry["kurtosis_9_over_f"] == pytest.approx(9.0 / entry["f_real"], rel=1e-12)
        assert set(entry["r18"]) == {str(N) for N in gatechain.DSR_TRIALS}
        for reading in entry["r18"].values():
            assert reading["band_lo"] <= reading["band_hi"]
            assert set(reading) == {"empirical", "band_lo", "band_hi"}


# ----------------------------------------------------- R33 (selection null)


def _chrom(indicator: str) -> StrategyChromosome:
    return StrategyChromosome(
        entry_genes=[StrategyGene(indicator, {"period": 14.0}, ConditionType.LT, 30.0, None)],
        exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0, None)],
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        direction=Direction.LONG,
    )


def _fit(total_return: float = 0.3, adjusted_score: float = 0.5) -> FitnessResult:
    return FitnessResult(
        sharpe_ratio=1.0,
        max_drawdown=0.1,
        profit_factor=1.5,
        total_trades=60,
        total_return=total_return,
        complexity_penalty=0.0,
        overtrade_penalty=0.0,
        sl_tp_penalty=0.0,
        raw_score=0.5,
        adjusted_score=adjusted_score,
        passed_filters=True,
        filter_results={},
    )


def test_selection_uses_production_top_k() -> None:
    # R33: the selection null picks through the PRODUCTION top-K
    # (_select_diverse_top_k, top_k 5, min_distance 0.15) over adjusted_score > 0.
    assert layer1.SELECTION_TOP_K == 5
    assert layer1.SELECTION_MIN_DISTANCE == 0.15

    chroms = [_chrom(name) for name in ("RSI", "MACD", "BBANDS")]
    for i, a in enumerate(chroms):
        for b in chroms[i + 1 :]:
            assert chromosome_distance(a, b) >= layer1.SELECTION_MIN_DISTANCE

    # the R33 fixture: 3 chromosome-distant candidates with IDENTICAL
    # fingerprints -> 1 pick (the behaviour fingerprint dedups the other two)
    assert len(layer1.select_picks([(c, _fit()) for c in chroms])) == 1

    # different fingerprints -> all three survive
    distinct = [_fit(total_return=0.1 * (i + 1)) for i in range(3)]
    assert len(layer1.select_picks(list(zip(chroms, distinct, strict=True)))) == 3

    # top_k = 5 caps the pick
    many = [(_chrom(name), _fit(total_return=0.1 + 0.01 * i)) for i, name in enumerate(
        ("RSI", "MACD", "BBANDS", "ATR", "CCI", "ROC", "NATR", "DONCHIAN")
    )]
    assert len(layer1.select_picks(many)) == 5

    # adjusted_score == 0 is the equality edge: ineligible (never picked)
    scored = [(_chrom("RSI"), _fit(adjusted_score=0.0)), (_chrom("MACD"), _fit(adjusted_score=-0.5)), (_chrom("BBANDS"), _fit(adjusted_score=0.2))]
    picked = layer1.select_picks(scored)
    assert [c for c, _ in picked] == [scored[2][0]]


def test_cell_seeds_are_reproducible() -> None:
    # R19: the seed stream is the deterministic SeedSequence spawn.
    cell = layer1.cells()[0]
    assert layer1.cell_seeds(cell, 3) == layer1.cell_seeds(cell, 3)
    assert len(set(layer1.cell_seeds(cell, 5))) == 5
    other = [c for c in layer1.cells() if c.window != cell.window][0]
    assert layer1.cell_seeds(cell, 3) != layer1.cell_seeds(other, 3)


def test_s50_interpolates_the_crossing() -> None:
    # V3: S50 is the linear interpolation of the first >= 50% crossing per
    # (window, N); unreached N report None.
    def cell(S: float, rate: float) -> dict[str, Any]:
        return {
            "window": "W3",
            "n": 150,
            "S": S,
            "chain_recent": {"1": rate},
        }

    results = [cell(0.0, 0.0), cell(2.0, 0.25), cell(4.0, 0.75)]
    assert layer1._s50(results, "W3", 1) == pytest.approx(3.0)
    assert layer1._s50([cell(0.0, 0.0), cell(4.0, 0.4)], "W3", 1) is None


# ------------------------------------------------------- R33 selection null


def _stub_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gate chain stub: passes only at N=167, fails at the 4h production N."""

    def fake(*_args: Any, **_kwargs: Any) -> dict[str, object]:
        return {"chain_recent": {167: True, 22121: False}}

    monkeypatch.setattr(gatechain, "evaluate_candidate", fake)


@pytest.mark.parametrize("window", ["W3", "W6"])
def test_selection_null_counts_chain_recent_at_n167(
    monkeypatch: pytest.MonkeyPatch, window: str
) -> None:
    _stub_chain(monkeypatch)
    out = layer1.selection_null(
        window, 0.0, 2, _closes(), gatechain.settings_for(window), candidates=40
    )
    assert out["picks_per_rep"] > 0  # type: ignore[operator]
    assert out["fp_rate"] == 1.0


def test_selection_null_common_noise_is_fresh_per_replicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_chain(monkeypatch)
    seen: list[np.ndarray[Any, Any]] = []
    real = layer1._null_schedule

    def spy(plan_name: str, n: int, seed: int, common: Any, rho: float) -> Any:
        seen.append(np.asarray(common).copy())
        return real(plan_name, n, seed, common, rho)

    monkeypatch.setattr(layer1, "_null_schedule", spy)
    reps, cands = 3, 5
    layer1.selection_null(
        "W6", 0.5, reps, _closes(), gatechain.settings_for("W6"), candidates=cands
    )
    assert len(seen) == reps * cands
    per_rep = [seen[r * cands : (r + 1) * cands] for r in range(reps)]
    for group in per_rep:  # shared within a replicate
        assert all(np.array_equal(group[0], c) for c in group)
    for a in range(reps):  # fresh across replicates
        for b in range(a + 1, reps):
            assert not np.array_equal(per_rep[a][0], per_rep[b][0])
