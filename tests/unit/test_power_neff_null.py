"""Power test T2: the V2-neff selection null (scripts/research/power/neff_null.py).

Job 20261017-power-paper; ``design-neff.md`` section 4 (the null check), review
r1 B5 (the Clopper-Pearson escalation, the MARGINAL exit 2 and the 1600-rep
top-up) and N1 (the sparse-survivor configs). Every literal asserted here is the
design's/chief's own number or scipy's own ``beta.ppf`` value (the CP bounds at
400 reps are the design's: 12/400 -> 0.04815392102222603, 13/400 ->
0.05117557532347004).

The null harness is heavy (167 candidates x 12 configs x reps), so the grid
tests drive it through small fixtures (``--reps 2 --candidates 12``) or stub the
chain; nothing here runs the 400-rep gate run.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research" / "power"))

import gatechain  # noqa: E402
import neff_null  # noqa: E402
import synth  # noqa: E402

from vibe_quant.discovery.neff import (  # noqa: E402
    NEFF_MIN_DAYS,
    EffectiveTrials,
    effective_trials,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG = REPO_ROOT / "data" / "catalog"

_GRID: list[neff_null.Config] = neff_null.configs()

#: The sparse configs N1 accepted (W3/W6 x rho 0.5/0.8) and the design's 8 base configs.
SPARSE_KEYS = ("W3:rho=0.5:sparse", "W3:rho=0.8:sparse", "W6:rho=0.5:sparse", "W6:rho=0.8:sparse")


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


def _cell(key: str, count: int, reps: int) -> dict[str, Any]:
    """One config row as the grid reports it, with the extra columns print reads."""
    return {
        "key": key,
        "reps": reps,
        "count_ev": count,
        "rate_ev": count / reps,
        "count_raw167": 0,
        "rate_raw167": 0.0,
        "rate_mc": 0.0,
        "rate_pr": 0.0,
        "rate_maxmcpr": 0.0,
        "cp_upper_ev": neff_null.clopper_pearson_upper(count, reps),
        "median_run_n_eff": 150.0,
        "median_n_used": 25.0,
        "fallback_rate": 0.0,
        "survivor_rate": 0.25,
        "picks_per_rep": 1.0,
        "drift": 0.0,
        "sparse": False,
    }


def _stub_chain(monkeypatch: pytest.MonkeyPatch, passing: object = None) -> None:
    """Gate chain stub: every pick passes ``passing`` (True = every N)."""

    def fake(*args: Any, **_kwargs: Any) -> dict[str, object]:
        settings = cast("gatechain.ChainSettings", args[5])
        if passing is True:
            return {"chain_recent": {int(n): True for n in settings.dsr_trials}}
        return {"chain_recent": {int(n): bool(n == passing) for n in settings.dsr_trials}}

    monkeypatch.setattr(gatechain, "evaluate_candidate", fake)


# ------------------------------------------------------------------ B5 (verdict)


def test_clopper_pearson_upper_matches_scipy() -> None:
    # B5: the one-sided 95% Clopper-Pearson upper bound, against scipy's own
    # beta.ppf, at the design's two named counts (12/400 passes, 13/400 does not)
    # plus the degenerate edges (k = 0, k = n) and a small-reps case.
    assert neff_null.clopper_pearson_upper(12, 400) == pytest.approx(
        beta.ppf(0.95, 13, 388), rel=1e-12
    )
    assert neff_null.clopper_pearson_upper(13, 400) == pytest.approx(
        beta.ppf(0.95, 14, 387), rel=1e-12
    )
    assert neff_null.clopper_pearson_upper(12, 400) == pytest.approx(0.04815392102222603, rel=1e-12)
    assert neff_null.clopper_pearson_upper(13, 400) == pytest.approx(0.05117557532347004, rel=1e-12)
    assert neff_null.clopper_pearson_upper(0, 400) == pytest.approx(
        beta.ppf(0.95, 1, 400), rel=1e-12
    )
    assert neff_null.clopper_pearson_upper(0, 400) == pytest.approx(1.0 - 0.05 ** (1 / 400), rel=1e-12)
    assert neff_null.clopper_pearson_upper(0, 10) == pytest.approx(beta.ppf(0.95, 1, 10), rel=1e-12)
    assert neff_null.clopper_pearson_upper(400, 400) == 1.0
    assert neff_null.clopper_pearson_upper(10, 10) == 1.0
    # the pre-registered boundary: the CP bound is what separates 12 from 13
    assert neff_null.clopper_pearson_upper(12, 400) <= neff_null.THRESHOLD
    assert neff_null.clopper_pearson_upper(13, 400) > neff_null.THRESHOLD


def test_verdict_threshold_and_marginal_escalation() -> None:
    # design section 4 / B5: the verdict is max fp_rate_ev <= 0.05 -> PASS exit 0,
    # above -> FAIL exit 1, and a config whose Clopper-Pearson upper bound clears
    # 0.05 while its point estimate does not -> MARGINAL exit 2 (the chief tops it
    # up). Tested on the scalar rule first, then through the grid verdict at the
    # two equality-adjacent counts (20/400 = 0.05 exactly, and 21/400 just above).
    assert neff_null.threshold_exit_code(0.0) == 0
    assert neff_null.threshold_exit_code(neff_null.THRESHOLD) == 0  # 0.05 counts as pass
    assert neff_null.threshold_exit_code(0.0501) == 1

    keys = ["W3:rho=0.0", "W6:rho=0.8"]
    at_threshold = neff_null.build_verdict(
        [_cell(keys[0], 0, 400), _cell(keys[1], 20, 400)], reps=400
    )
    assert at_threshold.max_rate == pytest.approx(0.05)
    assert neff_null.threshold_exit_code(at_threshold.max_rate) == 0
    assert at_threshold.status == "MARGINAL"
    assert at_threshold.exit_code == 2

    above = neff_null.build_verdict([_cell(keys[0], 0, 400), _cell(keys[1], 21, 400)], reps=400)
    assert above.status == "FAIL"
    assert above.exit_code == 1
    assert above.failing == ("W6:rho=0.8",)
    assert neff_null.threshold_exit_code(above.max_rate) == 1

    # B5: a config at 13/400 (CP upper 0.0512 > 0.05) is MARGINAL and the run
    # exits 2 so the chief tops it up; 12/400 (0.0482 <= 0.05) passes outright.
    marginal = neff_null.build_verdict(
        [_cell("W3:rho=0.0", 0, 400), _cell("W3:rho=0.5", 13, 400)], reps=400
    )
    assert marginal.status == "MARGINAL"
    assert marginal.exit_code == 2
    assert marginal.marginal == ("W3:rho=0.5",)
    assert marginal.failing == ()

    clean = neff_null.build_verdict(
        [_cell("W3:rho=0.0", 0, 400), _cell("W3:rho=0.5", 12, 400)], reps=400
    )
    assert clean.status == "PASS"
    assert clean.exit_code == 0
    assert clean.marginal == ()

    # the top-up resolves a MARGINAL config: its own point estimate at 1600 fresh
    # reps decides (<= 0.05 -> PASS, above -> FAIL).
    resolved = neff_null.build_verdict(
        [_cell("W3:rho=0.5", 13, 400)], reps=400, topup={"W3:rho=0.5": 0.0281}
    )
    assert resolved.status == "PASS"
    assert resolved.exit_code == 0
    assert resolved.topup == {"W3:rho=0.5": 0.0281}

    failed = neff_null.build_verdict(
        [_cell("W3:rho=0.5", 13, 400)], reps=400, topup={"W3:rho=0.5": 0.0513}
    )
    assert failed.status == "FAIL"
    assert failed.exit_code == 1
    # the deciding top-up rate is named (the 400-rep point rate alone was 0.0325)
    assert failed.failing == ("W3:rho=0.5",)
    assert failed.line.endswith("(top-up decided: W3:rho=0.5=0.0513)")
    # a point-rate failure is a FAIL even if a top-up number is supplied
    assert neff_null.build_verdict(
        [_cell("W3:rho=0.5", 21, 400)], reps=400, topup={"W3:rho=0.5": 0.01}
    ).exit_code == 1


def test_topup_runs_only_for_marginal_configs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # B5: on the gate run the top-up re-runs ONLY the marginal configs, with
    # TOPUP_REPS reps on the fresh 0x4E3F0 seed stream, and the final verdict is
    # its own point estimate -- exit 0 when the top-up is clean.
    base_cfg = neff_null.configs()[0]  # bound before the monkeypatch below
    cfg = neff_null.Config(key="W3:rho=0.5", window="W3", rho=0.5, drift=0.0004, sparse=True)
    calls: list[tuple[tuple[str, ...], int, neff_null.SeedStreams]] = []

    def fake_run_grid(
        cfgs: list[neff_null.Config],
        reps: int,
        workers: int,
        closes: Any,
        *,
        candidates: int,
        streams: neff_null.SeedStreams,
        dsr_rule: str,
    ) -> list[dict[str, Any]]:
        calls.append((tuple(c.key for c in cfgs), reps, streams))
        if streams == neff_null.GRID_SEEDS:
            return [_cell(c.key, 13 if c.key == cfg.key else 0, reps) for c in cfgs]
        return [_cell(c.key, 20, reps) for c in cfgs]

    monkeypatch.setattr(neff_null, "run_grid", fake_run_grid)
    monkeypatch.setattr(neff_null, "load_closes", lambda catalog: _closes())
    monkeypatch.setattr(neff_null, "configs", lambda: [cfg, base_cfg])
    out = tmp_path / "topup.json"

    assert neff_null.main(["--reps", "400", "--workers", "1", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert calls[0] == (("W3:rho=0.5", "W3:rho=0.0"), 400, neff_null.GRID_SEEDS)
    assert calls[1] == (("W3:rho=0.5",), neff_null.TOPUP_REPS, neff_null.TOPUP_SEEDS)
    # the top-up runs on the design's fresh 0x4E3F0 stream, never the grid's
    assert neff_null.TOPUP_SEEDS.rep == 0x4E3F0
    assert neff_null.TOPUP_SEEDS != neff_null.GRID_SEEDS
    report = cast("dict[str, Any]", json.loads(out.read_text()))
    assert report["verdict"]["status"] == "PASS"
    assert report["verdict"]["topup"] == {"W3:rho=0.5": 0.0125}
    assert "neff topup W3:rho=0.5: 20/1600 rate=0.0125 seeds=0x4E3F0" in printed
    assert report["verdict"]["line"].endswith("max=0.0325 -> PASS")

    # a clean grid runs no top-up at all
    calls.clear()

    def clean_grid(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        cfgs = cast("list[neff_null.Config]", args[0])
        reps = cast("int", args[1])
        calls.append((tuple(c.key for c in cfgs), reps, cast("neff_null.SeedStreams", kwargs["streams"])))
        return [_cell(c.key, 0, reps) for c in cfgs]

    monkeypatch.setattr(neff_null, "run_grid", clean_grid)
    assert neff_null.main(["--reps", "400", "--workers", "1", "--out", str(out)]) == 0
    assert len(calls) == 1


# ------------------------------------------------------------ R14 (estimator)


def test_null_feeds_production_estimator(monkeypatch: pytest.MonkeyPatch) -> None:
    # R14: the rows handed to the estimator are exactly the candidates whose
    # series the production keep_series rule kept, in candidate-index order, with
    # raw = the candidate count -- and the estimator IS the production function
    # (T1's vibe_quant.discovery.neff.effective_trials), never a copy.
    assert neff_null.effective_trials is effective_trials

    seen: list[tuple[list[np.ndarray[Any, Any]], int]] = []
    real = effective_trials

    def spy(rows: Any, raw: int, **kwargs: Any) -> EffectiveTrials:
        seen.append(([np.asarray(r) for r in rows], raw))
        return real(rows, raw, **kwargs)

    monkeypatch.setattr(neff_null, "effective_trials", spy)
    _stub_chain(monkeypatch)

    cfg = neff_null.configs()[0]
    out = neff_null.run_config(cfg, 1, _closes(), candidates=40)
    assert len(seen) == 1
    rows, raw = seen[0]
    assert raw == 40

    common = neff_null.common_returns(cfg, 0, neff_null.GRID_SEEDS)
    timeframe = gatechain.settings_for(cfg.window).config.timeframe
    expected = [
        np.frombuffer(cand.fitness.mtm_daily_returns.values, dtype=np.float64)  # type: ignore[union-attr]
        for cand in (
            neff_null.build_candidate(cfg, 0, i, common, timeframe) for i in range(40)
        )
        if cand.fitness.mtm_daily_returns is not None
    ]
    # the production rule keeps only the positive-total-return survivors: strictly
    # fewer than all 40 rows, and exactly the kept ones in candidate-index order
    assert 0 < len(expected) < 40
    assert len(rows) == len(expected)
    for got, exp in zip(rows, expected, strict=True):
        assert np.array_equal(got, exp)
    assert out["kept_count"] == len(expected)
    assert out["reps"] == 1


def test_dsr_trials_are_the_per_rep_dedup(monkeypatch: pytest.MonkeyPatch) -> None:
    # design section 4: settings.dsr_trials = dedup of (167, run_n_eff, ceil mc,
    # ceil pr, max of the two), forwarded to the chain. Two unrelated inputs with
    # literal expected tuples; the legacy estimators are absent in the second.
    estimates = [
        EffectiveTrials(
            raw=167,
            non_flat=40,
            n_used=40,
            n_distinct=40,
            subsampled_from=None,
            neff_mean_corr=3.2,
            neff_eig_pr=4.7,
            neff_ev=8.5,
            ev_q95_spread=1.1,
            run_n_eff=9,
            source="ev_mc",
            fallback_reason=None,
        ),
        EffectiveTrials(
            raw=167,
            non_flat=1,
            n_used=1,
            n_distinct=1,
            subsampled_from=None,
            neff_mean_corr=None,
            neff_eig_pr=None,
            neff_ev=None,
            ev_q95_spread=None,
            run_n_eff=167,
            source="fallback_raw",
            fallback_reason="fewer_than_2_series",
        ),
    ]
    queue = list(estimates)
    seen: list[tuple[int, ...]] = []
    raw_seen: list[int] = []

    def spy(rows: Any, raw: int, **_kwargs: Any) -> EffectiveTrials:
        raw_seen.append(raw)
        return queue.pop(0)

    def fake_chain(*args: Any, **_kwargs: Any) -> dict[str, object]:
        settings = cast("gatechain.ChainSettings", args[5])
        seen.append(tuple(settings.dsr_trials))
        return {"chain_recent": {int(n): False for n in settings.dsr_trials}}

    monkeypatch.setattr(neff_null, "effective_trials", spy)
    monkeypatch.setattr(gatechain, "evaluate_candidate", fake_chain)

    base, other = neff_null.configs()[0], neff_null.configs()[4]
    neff_null.run_config(base, 1, _closes(), candidates=167)
    # every pick of the replicate is scored at the same forwarded N set
    assert seen and all(ns == (4, 5, 9, 167) for ns in seen), seen
    seen.clear()
    neff_null.run_config(other, 1, _closes(), candidates=167)
    assert raw_seen == [167, 167]
    # 3.2 -> 4, 4.7 -> 5, max 5; run_n_eff 9; raw 167
    # legacy estimators missing -> only the raw N survives (fail closed)
    assert seen and all(ns == (167,) for ns in seen), seen


def test_false_positive_is_any_pick_at_the_rep_n(monkeypatch: pytest.MonkeyPatch) -> None:
    # design section 4: a replicate is a false positive iff ANY of the top-5 picks
    # passes chain_recent[run_n_eff]. The OR reads every pick: first-only,
    # last-only and none all behave differently.
    passes = neff_null.any_pick_passed
    assert passes([], 9) is False
    assert passes([{9: False, 167: False}, {9: False, 167: False}], 9) is False
    assert passes([{9: True, 167: False}, {9: False, 167: False}], 9) is True
    assert passes([{9: False, 167: False}, {9: True, 167: False}], 9) is True
    # the OR is per-N: a pick passing at the raw 167 is not a pass at the run's N
    assert passes([{9: False, 167: True}, {9: False, 167: True}], 9) is False
    assert passes([{9: False, 167: True}, {9: False, 167: False}], 167) is True

    # through the harness: an always-passing chain is a false positive in every
    # rep, an always-failing one never is
    cfg = neff_null.configs()[0]
    _stub_chain(monkeypatch, passing=True)
    assert neff_null.run_config(cfg, 2, _closes(), candidates=40)["rate_ev"] == 1.0
    assert neff_null.run_config(cfg, 2, _closes(), candidates=40)["rate_raw167"] == 1.0
    _stub_chain(monkeypatch)
    assert neff_null.run_config(cfg, 2, _closes(), candidates=40)["rate_ev"] == 0.0


# --------------------------------------------------- N1 (sparse survivor configs)


def test_config_grid_and_window_lengths() -> None:
    # design section 4 + N1: W3/W6 x rho {0.0, 0.2, 0.5, 0.8} plus the 4 sparse
    # configs (W3/W6 x rho 0.5/0.8 with the negative drift) = 12.
    cfgs = neff_null.configs()
    assert len(cfgs) == 12
    assert [c.key for c in cfgs if c.sparse] == list(SPARSE_KEYS)
    assert {(c.window, c.rho) for c in cfgs if not c.sparse} == {
        (w, r) for w in ("W3", "W6") for r in (0.0, 0.2, 0.5, 0.8)
    }
    assert all(c.drift == 0.0 for c in cfgs if not c.sparse)
    # every window's chained train series clears NEFF_MIN_DAYS, so the estimator
    # runs the EV path rather than falling back too_short
    for cfg in cfgs:
        assert synth.PLANS[cfg.window].chained_days >= NEFF_MIN_DAYS
    assert (neff_null.SPARSE_RHOS, neff_null.SPARSE_TARGET_SURVIVAL) == ((0.5, 0.8), 0.10)


def _per_rep_kept(
    monkeypatch: pytest.MonkeyPatch, cfg: neff_null.Config, reps: int
) -> tuple[dict[str, Any], list[int]]:
    """Run ``cfg`` and return its row plus the retained-series count of each replicate."""
    _stub_chain(monkeypatch)
    kept: list[int] = []
    real = effective_trials

    def spy(rows: Any, raw: int, **kwargs: Any) -> EffectiveTrials:
        kept.append(len(rows))
        return real(rows, raw, **kwargs)

    monkeypatch.setattr(neff_null, "effective_trials", spy)
    row = cast("dict[str, Any]", neff_null.run_config(cfg, reps, _closes(), candidates=167))
    return row, kept


@pytest.mark.parametrize("key", ["W3:rho=0.8:sparse", "W6:rho=0.5:sparse"])
def test_sparse_config_per_replicate_survivors(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # N1 / F1: EVERY replicate keeps ~17 of 167 series (the production small-
    # survivor regime), not just the marginal mean. A replicate is either inside
    # the stated band 8..28, or an undershoot: its unshifted common path already
    # keeps fewer, and a negative-only shift leaves it at drift 0. Fixed seed.
    # One row per window x rho corner (W3/0.8 and W6/0.5): the other two configs'
    # existence and seeds are pinned by test_config_grid_and_window_lengths and
    # test_sparse_streams_are_separate_namespaces.
    cfg = next(c for c in _GRID if c.key == key)
    row, kept = _per_rep_kept(monkeypatch, cfg, 3)
    low, high = neff_null.SPARSE_BAND
    assert (low, high) == (8, 28)
    for index, count in enumerate(kept):
        common = neff_null.common_returns(cfg, index, neff_null.GRID_SEEDS)
        drift, achieved = neff_null.replicate_drift(cfg, index, common)
        assert achieved == count, (key, index, achieved, count)  # surrogate == real path
        assert count <= high, (key, index, count, kept)
        if count < low:
            assert drift == 0.0, (key, index, count, drift)  # disclosed undershoot
    assert row["kept_count"] == sum(kept)
    # the row reports what was drawn
    assert row["survivor_target"] == 17
    assert (row["survivor_min"], row["survivor_max"]) == (min(kept), max(kept))
    assert row["survivor_median"] == float(np.median(kept))
    assert row["survivor_undershoots"] == sum(1 for k in kept if k < 17)
    assert row["survivor_out_of_band"] == sum(1 for k in kept if not low <= k <= high)
    # base configs have no shift and no sparse bookkeeping
    plain = next(c for c in _GRID if c.key == f"{cfg.window}:rho={cfg.rho}")
    base_row, _ = _per_rep_kept(monkeypatch, plain, 1)
    assert base_row["drift"] == 0.0 and base_row["survivor_undershoots"] is None


def test_replicate_drift_is_negative_only_and_hits_the_target() -> None:
    # F1 ruling: the shift never creates edge. Replicates whose unshifted path
    # keeps 63..102 are pulled to the target 17 by a strictly positive shift
    # (W3 rho 0.8, index 4 kept 63 in the reviewer's probe on the old stream
    # family); an unshifted undershoot keeps drift exactly 0.
    cfg = next(c for c in _GRID if c.key == "W3:rho=0.8:sparse")
    seen_shift = seen_undershoot = False
    for index in range(5):  # indexes 0 and 3 undershoot, 1/2/4 are shifted down to 17
        common = neff_null.common_returns(cfg, index, neff_null.GRID_SEEDS)
        drift, count = neff_null.replicate_drift(cfg, index, common)
        assert drift >= 0.0
        schedules = [
            neff_null.candidate_schedule(cfg, index, i, common)[1] for i in range(167)
        ]
        unshifted = neff_null.survivor_count(schedules, 0.0, cfg.window)
        assert neff_null.survivor_count(schedules, drift, cfg.window) == count
        if unshifted > 17:
            assert drift > 0.0 and count == 17, (index, unshifted, drift, count)
            seen_shift = True
        else:
            assert drift == 0.0 and count == unshifted
            seen_undershoot = True
    assert seen_shift and seen_undershoot


def test_sparse_streams_are_separate_namespaces() -> None:
    # F2: a sparse config never shares chromosomes or common paths with the base
    # cell of the same window/rho, in any run mode (grid, top-up) or replicate; the
    # base streams keep their historical values.
    base = next(c for c in _GRID if c.key == "W3:rho=0.8")
    sparse = next(c for c in _GRID if c.key == "W3:rho=0.8:sparse")
    for streams in (neff_null.GRID_SEEDS, neff_null.TOPUP_SEEDS):
        for index in range(3):
            assert neff_null.stream_seed(base, index, streams.rep) != neff_null.stream_seed(
                sparse, index, streams.rep
            )
            assert neff_null.stream_seed(base, index, streams.common) != neff_null.stream_seed(
                sparse, index, streams.common
            )
    # the base stream is the unchanged [SEED_BASE, tag, w, rho, index] draw
    expected = int(
        np.random.SeedSequence([neff_null.SEED_BASE, 0x4E3FF, 0, 3, 0]).generate_state(
            1, dtype=np.uint32
        )[0]
    )
    assert neff_null.stream_seed(base, 0, neff_null.GRID_SEEDS.rep) == expected
    # distinct across sparse configs too
    seeds = {
        neff_null.stream_seed(c, 0, neff_null.GRID_SEEDS.rep) for c in _GRID if c.sparse
    }
    assert len(seeds) == 4


@pytest.mark.parametrize("key", ["W3:rho=0.0", "W6:rho=0.2"])
def test_survival_surrogate_matches_the_keep_rule(key: str) -> None:
    # The calibration reads survival off the schedule's own window arithmetic
    # (the worst eval window's compounded return), because bisecting on the real
    # fitness path is 100x too slow. It must agree with the production
    # keep_series flag of the REAL path candidate by candidate, not on average.
    cfg = next(c for c in _GRID if c.key == key)
    windows = synth.PLANS[cfg.window].eval_windows
    timeframe = gatechain.settings_for(cfg.window).config.timeframe
    kept = dropped = 0
    for rep in range(2):
        common = neff_null.common_returns(cfg, rep, neff_null.GRID_SEEDS)
        for i in range(60):
            candidate = neff_null.build_candidate(cfg, rep, i, common, timeframe)
            worst = min(neff_null.window_return(candidate.schedule, ws, we) for ws, we in windows)
            real = neff_null.kept_rows(candidate) is not None
            assert real == (worst > 0.0), (key, rep, i, real, worst)
            kept += int(real)
            dropped += int(not real)
    assert kept > 0 and dropped > 0  # both sides of the rule are exercised


# ------------------------------------------------------- F4 (top-up boundary)


@pytest.mark.parametrize(
    ("quick", "reps", "tops_up"),
    [(True, 2, False), (True, 400, False), (False, 10, False), (False, 400, True)],
)
def test_topup_boundary_matrix(
    quick: bool, reps: int, tops_up: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # F4: exactly one cell of the {--quick, --reps} matrix escalates -- a
    # nonquick DESIGN_REPS run. The marginal fixture is 13/400 (rate 0.0325, CP
    # upper 0.05117 > 0.05); every other cell is MARGINAL exit 2 with a single
    # run_grid call. In particular --quick with an EXPLICIT 400 reps must not
    # launch the 1600-rep top-up, which would let a plumbing run return PASS 0.
    calls: list[int] = []

    def fake_run_grid(
        cfgs: list[neff_null.Config], reps_: int, workers: int, closes: Any, **kwargs: Any
    ) -> list[dict[str, Any]]:
        calls.append(reps_)
        count = 13 if reps_ == neff_null.DESIGN_REPS else 0
        return [_cell(c.key, count, reps_) for c in cfgs]

    monkeypatch.setattr(neff_null, "run_grid", fake_run_grid)
    monkeypatch.setattr(neff_null, "load_closes", lambda catalog: _closes())
    out = tmp_path / f"matrix-{quick}-{reps}.json"
    args = ["--reps", str(reps), "--workers", "1", "--out", str(out)]
    if quick:
        args.insert(0, "--quick")

    assert neff_null.main(args) == (0 if tops_up else 2)
    assert calls == ([reps, neff_null.TOPUP_REPS] if tops_up else [reps])
    report = cast("dict[str, Any]", json.loads(out.read_text()))
    assert report["verdict"]["status"] == ("PASS" if tops_up else "MARGINAL")
    assert report["verdict"]["exit_code"] == (0 if tops_up else 2)
    assert bool(report["verdict"]["topup"]) is tops_up


# ------------------------------------------------------------------- --quick


def _quick_args(out: Path, workers: int, reps: int = 2, candidates: int = 4) -> list[str]:
    return [
        "--quick",
        "--reps",
        str(reps),
        "--candidates",
        str(candidates),
        "--workers",
        str(workers),
        "--catalog",
        str(CATALOG),
        "--out",
        str(out),
    ]


def test_quick_run_is_deterministic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # R19-style: the --quick report JSON is byte-identical across reruns and for
    # workers 1 vs 2, and it prints the V2-neff line. The JSON carries no wall
    # clock at all (the timestamp goes to stdout), so the files compare byte for
    # byte. A 2-rep run is MARGINAL: it cannot demonstrate FP <= 5% and the
    # 1600-rep top-up is --quick-skipped. Four candidates per config: what is
    # compared is the seed stream's reproducibility, not the candidate count.
    if not (CATALOG / "data").exists():
        pytest.skip("data/catalog absent")

    first = tmp_path / "a.json"
    rc = neff_null.main(_quick_args(first, 1))
    line = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("V2-neff:")]
    assert len(line) == 1
    assert line[0].endswith("max=0.0000 -> MARGINAL")
    assert rc == 2

    second = tmp_path / "b.json"
    assert neff_null.main(_quick_args(second, 2)) == 2
    assert first.read_bytes() == second.read_bytes()

    report = cast("dict[str, Any]", json.loads(first.read_text()))
    assert report["reps"] == 2
    assert report["candidates"] == 4
    assert report["verdict"]["status"] == "MARGINAL"
    assert report["verdict"]["exit_code"] == 2
    assert report["verdict"]["topup"] == {}
    assert len(report["configs"]) == 12
    assert all(c["reps"] == 2 for c in report["configs"])
    assert "generated_at" not in report
    # the sparse configs really are in the grid and really are drifted
    assert {c["key"] for c in report["configs"] if c["sparse"]} == set(SPARSE_KEYS)
    assert all(c["survivor_target"] == 1 for c in report["configs"] if c["sparse"])


def test_main_does_not_leak_the_vibe_quant_log_level(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # board 18df6f414b3f45f0: a logger level set for the run must not outlive it
    # (it leaks across pytest). Silence with layer1._quiet_vibe_quant only.
    if not (CATALOG / "data").exists():
        pytest.skip("data/catalog absent")
    logger = logging.getLogger("vibe_quant")
    previous = logger.level
    # the smallest run: only that main() returns and restores the level matters here
    assert neff_null.main(_quick_args(tmp_path / "q.json", 1, reps=1, candidates=3)) == 2
    assert logger.level == previous

    caplog.clear()
    with caplog.at_level(logging.INFO):
        logging.getLogger("vibe_quant.discovery.fitness").info("NEFF-NULL-INFO-VISIBLE")
    assert "NEFF-NULL-INFO-VISIBLE" in caplog.text
    assert logger.level == previous
