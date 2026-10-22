"""V2-neff: false-positive check of the EV-matched effective N (bd vibe-quant-megq1).

Job 20261017-power-paper, task T2 -- ``design-neff.md`` section 4 (the null
check: grid, command, columns, verdict line, threshold), review r1 B5 (the
Clopper-Pearson upper-bound column, the MARGINAL exit 2 and the 1600-rep top-up
on the fresh 0x4E3F0 seed stream) and N1 (``median_n_used`` plus the four
sparse-survivor configs W3/W6 x rho 0.5/0.8, each replicate shifted by its own
negative scalar until ~10% of its series survive).

What it measures
----------------
Each replicate draws ``SELECTION_K`` (167) random chromosomes at S = 0 with a
shared common noise path, scores them through the REAL ``synth``/``gatechain``
harness, and asks whether any of the production top-5 picks would pass the real
gate chain when the DSR trial count is this run's EV-matched effective N
(``vibe_quant.discovery.neff.effective_trials`` over the candidates the
production ``keep_series`` rule retained, in candidate-index order). The
reported ``fp_rate_ev`` is the verdict column; ``fp_rate_raw167``,
``fp_rate_mc``, ``fp_rate_pr`` and ``fp_rate_maxmcpr`` are the same picks
re-tested at the raw count and at the two legacy estimators, so the leniency
being fixed is visible in one table.

``rho`` is the correlation of the shared S = 0 shock every candidate of a
replicate is blended with (``layer1._null_schedule``), so it is a per-REPLICATE
shock and not a per-candidate one: the per-candidate keep probability is
rho-independent, but ``survivor_rate`` and ``median_n_used`` come out bimodal
while a replicate runs on a handful of them (a favourable common path keeps
nearly all candidates, an unfavourable one nearly none). Only the readings over
many replicates -- the 400-rep gate run -- estimate the config's true rate. The sparse shift is
therefore bisected PER REPLICATE toward 17 of 167 survivors (never positive; a
replicate already at or below the target is an undershoot, kept at drift 0), and
the per-config columns report the achieved min/median/max, the undershoots and
the replicates outside ``SPARSE_BAND``.

Verdict (design section 4 + B5)
------------------------------
``max fp_rate_ev <= 0.05`` over the 12 configs -> PASS (exit 0). A config whose
point rate is above 0.05 -> FAIL (exit 1). A config whose point rate clears
0.05 but whose one-sided 95% Clopper-Pearson upper bound does not (13/400 =
0.0512) -> MARGINAL: at the pre-registered 400 reps the run then tops the
marginal configs up to ``TOPUP_REPS`` (1600) fresh replicates and the top-up's
own point estimate decides, PASS (exit 0) or FAIL (exit 1). Any other MARGINAL
run -- in particular ``--quick``, which never tops up -- exits 2 so the caller
cannot mistake a short run for a conclusion.

Usage
-----
``cd <wt> && PYTHONPATH=$PWD .venv/bin/python scripts/research/power/neff_null.py \
--reps 400 --workers 8 --out data/swarm/20261017-power-paper/power/neff_null.json``
(the chief's G1 gate run). ``--quick`` (10 reps) is a plumbing check: it must
stay under 60 s and its numbers are not a verdict.

The JSON carries no wall clock and no runtime, so two identical runs write
byte-identical files (the timestamp and the runtime go to stdout instead).

Not on the mypy baseline (``files = ["vibe_quant"]``); kept typed and
ruff-clean. Logs are silenced only through ``layer1._quiet_vibe_quant`` (board
18df6f414b3f45f0: a bare ``setLevel`` leaks across pytest).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import gatechain
import layer1
import numpy as np
import synth
from scipy.stats import beta

from vibe_quant.discovery.genome import generate_random_chromosome
from vibe_quant.discovery.neff import effective_trials

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from vibe_quant.discovery.fitness import FitnessResult
    from vibe_quant.discovery.neff import EffectiveTrials
    from vibe_quant.discovery.operators import StrategyChromosome

#: The design's grid: W3/W6 x rho, plus the N1 sparse configs at rho 0.5/0.8.
WINDOWS: tuple[str, ...] = layer1.WINDOWS
RHO_GRID: tuple[float, ...] = (0.0, 0.2, 0.5, 0.8)
SPARSE_RHOS: tuple[float, ...] = (0.5, 0.8)
SEED_BASE: int = layer1.SEED_BASE
CONFIG_CANDIDATES: int = layer1.SELECTION_K
RESULTS_N: int = layer1.RESULTS_N
THRESHOLD: float = layer1.SIGNIFICANCE
CP_LEVEL: float = 0.95
DESIGN_REPS: int = 400
QUICK_REPS: int = 10
TOPUP_REPS: int = 1600
#: The sparse regime (N1, chief ruling on F1): EACH replicate gets its own scalar
#: negative shift, bisected until it retains ~``SPARSE_TARGET_SURVIVAL`` of the
#: candidates (17 of 167), the steady small-survivor regime. A shift is never
#: positive (true edge <= 0), so a replicate whose unshifted common path already
#: keeps fewer than the target is an UNDERSHOOT: it keeps drift 0 and is reported.
SPARSE_TARGET_SURVIVAL: float = 0.10
CALIBRATION_STEPS: int = 24
#: Bracket start for the per-active-day shift (doubled until it reaches the target).
SPARSE_DRIFT_HI: float = 0.002
#: Acceptance band on the achieved per-replicate survivor count (of 167).
SPARSE_BAND: tuple[int, int] = (8, 28)
#: The reported columns: the verdict column and the four comparison readings.
COLUMN_NAMES: tuple[str, ...] = ("ev", "raw167", "mc", "pr", "maxmcpr")

DEFAULT_OUT: str = "data/swarm/20261017-power-paper/power/neff_null.json"
DEFAULT_CATALOG: str = "data/catalog"
JOB: str = "20261017-power-paper"
TASK: str = "neff-null"
#: ``--workers`` default, matching layer1.
DEFAULT_WORKERS: int = 8


@dataclass(frozen=True, slots=True)
class SeedStreams:
    """The two seed tags one run mode draws its replicates from.

    ``rep`` seeds the candidate chromosomes of a replicate (and rotates them
    across replicates), ``common`` the shared S = 0 noise path of that
    replicate. Distinct objects per mode keep the top-up on a fresh stream:
    0x4E3FF grid, 0x4E3F0 top-up (B5).
    """

    rep: int
    common: int


GRID_SEEDS: SeedStreams = SeedStreams(0x4E3FF, 0x5E1EC7)
TOPUP_SEEDS: SeedStreams = SeedStreams(0x4E3F0, 0x4E3F3)


@dataclass(frozen=True, slots=True)
class Config:
    """One null configuration: a plan window, a candidate correlation, the drift."""

    key: str
    window: str
    rho: float
    drift: float
    sparse: bool


@dataclass(frozen=True, slots=True)
class Candidate:
    """One null candidate: its chromosome, its scheduled series and its fitness."""

    chromosome: StrategyChromosome
    schedule: synth.Schedule
    fitness: FitnessResult
    seed: int


def configs() -> list[Config]:
    """The 12-config grid: the 8 base cells, then the 4 N1 sparse ones."""
    base = [
        Config(key=f"{window}:rho={rho}", window=window, rho=rho, drift=0.0, sparse=False)
        for window in WINDOWS
        for rho in RHO_GRID
    ]
    sparse = [
        Config(
            key=f"{window}:rho={rho}:sparse",
            window=window,
            rho=rho,
            drift=0.0,
            sparse=True,
        )
        for window in WINDOWS
        for rho in SPARSE_RHOS
    ]
    return base + sparse


def stream_seed(cfg: Config, index: int, tag: int) -> int:
    """One draw from the config's seed stream: ``[SEED_BASE, tag, w, rho, index]``.

    A sparse config appends a trailing ``1``, a distinct namespace, so it never
    shares chromosomes or common paths with the base cell of the same window/rho;
    the base streams (and ``layer1``'s) are unchanged.
    """
    entropy = [SEED_BASE, tag, WINDOWS.index(cfg.window), RHO_GRID.index(cfg.rho), index]
    if cfg.sparse:
        entropy.append(1)
    seq = np.random.SeedSequence(entropy)
    return int(seq.generate_state(1, dtype=np.uint32)[0])


@dataclass(frozen=True, slots=True)
class Verdict:
    """The run's verdict (design section 4 + B5).

    ``status`` is PASS / MARGINAL / FAIL and ``exit_code`` 0 / 2 / 1; a run that
    resolved its MARGINAL configs with a top-up is reported as the top-up's own
    PASS or FAIL, never as MARGINAL.
    """

    status: str
    exit_code: int
    max_rate: float
    failing: tuple[str, ...]
    marginal: tuple[str, ...]
    topup: dict[str, float]
    line: str


def clopper_pearson_upper(count: int, reps: int, *, level: float = CP_LEVEL) -> float:
    """The one-sided Clopper-Pearson upper bound of ``count`` successes in ``reps`` (B5).

    ``beta.ppf(level, count + 1, reps - count)``; the bound is 1.0 when every
    replicate succeeded (the beta quantile is degenerate there).
    """
    if count >= reps:
        return 1.0
    return float(beta.ppf(level, count + 1, reps - count))


def threshold_exit_code(rate: float) -> int:
    """The design section 4 point rule on ONE config's rate: 0 at or below, else 1."""
    return 0 if rate <= THRESHOLD else 1


def build_verdict(
    results: Sequence[Mapping[str, object]],
    *,
    reps: int,
    topup: Mapping[str, float] | None = None,
) -> Verdict:
    """The grid verdict from the per-config rows (design section 4 + B5).

    ``topup`` is the marginal configs' top-up point rates, or None when no
    top-up ran (a short run or ``--quick``) -- which is what turns MARGINAL into
    exit 2 rather than a conclusion.
    """
    rates = {str(row["key"]): float(cast("float", row["rate_ev"])) for row in results}
    counts = {str(row["key"]): int(cast("int", row["count_ev"])) for row in results}
    failing = tuple(key for key, rate in rates.items() if threshold_exit_code(rate) == 1)
    marginal = tuple(
        key
        for key, rate in rates.items()
        if threshold_exit_code(rate) == 0
        and clopper_pearson_upper(counts[key], reps) > THRESHOLD
    )
    topup_failing: tuple[str, ...] = ()
    if failing:
        status, exit_code = "FAIL", 1
    elif marginal and topup is None:
        status, exit_code = "MARGINAL", 2
    elif marginal:
        topup_failing = tuple(key for key in marginal if topup[key] > THRESHOLD)
        status, exit_code = ("FAIL", 1) if topup_failing else ("PASS", 0)
        failing = topup_failing
    else:
        status, exit_code = "PASS", 0
    max_rate = max(rates.values(), default=0.0)
    suffix = {"PASS": "PASS", "MARGINAL": "MARGINAL", "FAIL": "FAIL (>0.05)"}[status]
    readings = ", ".join(f"{key}={rate:.4f}" for key, rate in rates.items())
    line = f"V2-neff: selection [{readings}] max={max_rate:.4f} -> {suffix}"
    if topup_failing and topup is not None:
        decided = ", ".join(f"{key}={topup[key]:.4f}" for key in topup_failing)
        line += f" (top-up decided: {decided})"
    return Verdict(
        status=status,
        exit_code=exit_code,
        max_rate=max_rate,
        failing=failing,
        marginal=marginal,
        topup=dict(topup or {}),
        line=line,
    )


def common_returns(cfg: Config, index: int, streams: SeedStreams) -> NDArray[np.float64]:
    """The replicate's shared S = 0 noise path (fresh across replicates)."""
    seed = stream_seed(cfg, index, streams.common)
    return synth.build_schedule(cfg.window, RESULTS_N, 0.0, seed).returns


def candidate_schedule(
    cfg: Config,
    index: int,
    i: int,
    common: NDArray[np.float64],
    *,
    streams: SeedStreams = GRID_SEEDS,
) -> tuple[StrategyChromosome, synth.Schedule, int]:
    """Candidate ``i`` of replicate ``index``: its chromosome, drifted series, seed.

    The chromosome stream is ``rep_seed + i`` (as ``layer1.selection_null``); the
    series is ``layer1._null_schedule``'s blend of the candidate's own S = 0 draw
    with the replicate's shared noise, minus the (replicate's) negative drift on the
    in-position days only (see ``shifted``).
    """
    rep_seed = stream_seed(cfg, index, streams.rep)
    cseed = (rep_seed + i) % (2**32)
    chromosome = generate_random_chromosome(random.Random(cseed))
    schedule = layer1._null_schedule(cfg.window, RESULTS_N, cseed, common, cfg.rho)
    return chromosome, shifted(schedule, cfg.drift), cseed


def shifted(schedule: synth.Schedule, drift: float) -> synth.Schedule:
    """``schedule`` with ``drift`` subtracted on the in-position days only.

    Flat days stay exactly flat, so the realized exposure is untouched.
    """
    if drift <= 0.0:
        return schedule
    active = schedule.returns != 0.0
    return replace(schedule, returns=np.where(active, schedule.returns - drift, schedule.returns))


def build_candidate(
    cfg: Config,
    index: int,
    i: int,
    common: NDArray[np.float64],
    timeframe: str,
    *,
    streams: SeedStreams = GRID_SEEDS,
) -> Candidate:
    """Candidate ``i`` of replicate ``index`` with its fitness through the real gate.

    The holdout window is NOT drifted: the drift models the retention regime the
    estimator sees, and the holdout is a separate noise window for the holdout
    gate.
    """
    chromosome, schedule, cseed = candidate_schedule(cfg, index, i, common, streams=streams)
    fitness = layer1._null_fitness(schedule, chromosome, timeframe)
    return Candidate(chromosome=chromosome, schedule=schedule, fitness=fitness, seed=cseed)


def kept_rows(candidate: Candidate) -> NDArray[np.float64] | None:
    """The candidate's retained MTM series, or None (the production reader's filter).

    Mirrors ``gate_chain.py::GateChain.trial_counts``: only a series the production
    ``keep_series`` rule kept is handed to the estimator (``None`` otherwise),
    as the float array production builds from the ``DailyReturns`` buffer.
    """
    series = candidate.fitness.mtm_daily_returns
    if series is None or len(series) < 2:
        return None
    return np.frombuffer(series.values, dtype=np.float64)


def column_n(eff: EffectiveTrials, candidates: int) -> dict[str, int | None]:
    """The DSR N each reported column scores a replicate's picks at (design section 4).

    ``ev`` is the run's effective N, ``raw167`` the raw candidate count and
    ``mc``/``pr``/``maxmcpr`` the two legacy estimators (ceil'd, clamped to >= 1
    like ``run_n_eff``); a legacy estimator that fell back has no N, so that
    column is scored over the replicates where it exists (``legacy_reps``).
    """
    mean_corr = None if eff.neff_mean_corr is None else max(1, math.ceil(eff.neff_mean_corr))
    eig_pr = None if eff.neff_eig_pr is None else max(1, math.ceil(eff.neff_eig_pr))
    return {
        "ev": eff.run_n_eff,
        "raw167": candidates,
        "mc": mean_corr,
        "pr": eig_pr,
        "maxmcpr": None if mean_corr is None or eig_pr is None else max(mean_corr, eig_pr),
    }


def any_pick_passed(pick_chains: Sequence[Mapping[int, bool]], num_trials: int) -> bool:
    """design section 4: a replicate is a false positive iff ANY top-5 pick passes."""
    return any(bool(chain[num_trials]) for chain in pick_chains)


def run_config(
    cfg: Config,
    reps: int,
    closes: Mapping[str, Mapping[int, float]],
    *,
    candidates: int = CONFIG_CANDIDATES,
    streams: SeedStreams = GRID_SEEDS,
    dsr_rule: str = "emax",
) -> dict[str, object]:
    """Run one config's ``reps`` replicates and report its columns.

    Per replicate: build ``candidates`` chromosomes at S = 0 on the config's own
    seed stream, hand the retained series (candidate-index order) to the
    PRODUCTION estimator, take the production top-5 pick, and score every pick
    through the real ``gatechain`` at each column's N. The DSR N set forwarded
    to the chain is the dedup of the columns that exist, as the design's
    ``settings.dsr_trials``.
    """
    settings = gatechain.settings_for(cfg.window)
    timeframe = settings.config.timeframe
    counts = dict.fromkeys(COLUMN_NAMES, 0)
    denoms = dict.fromkeys(COLUMN_NAMES, 0)
    n_eff: list[int] = []
    n_used: list[int] = []
    kept_count = 0
    fallbacks = 0
    picks_total = 0
    drifts: list[float] = []
    kept_per_rep: list[int] = []
    target = target_survivors(candidates)

    for index in range(reps):
        common = common_returns(cfg, index, streams)
        rep_cfg = cfg
        if cfg.sparse:
            drift, _achieved = replicate_drift(
                cfg, index, common, candidates=candidates, streams=streams
            )
            rep_cfg = replace(cfg, drift=drift)
        drifts.append(rep_cfg.drift)
        entries = [
            build_candidate(rep_cfg, index, i, common, timeframe, streams=streams)
            for i in range(candidates)
        ]
        rows = [row for row in (kept_rows(entry) for entry in entries) if row is not None]
        kept_count += len(rows)
        kept_per_rep.append(len(rows))
        eff = effective_trials(rows, candidates)
        n_eff.append(eff.run_n_eff)
        n_used.append(eff.n_used)
        fallbacks += int(eff.source == "fallback_raw")

        ranked = sorted(entries, key=lambda entry: entry.fitness.adjusted_score, reverse=True)
        picks = layer1.select_picks([(entry.chromosome, entry.fitness) for entry in ranked])
        picks_total += len(picks)
        picked = {id(chromosome) for chromosome, _ in picks}
        # design-dsr-q95: the q95 rule uses THIS replicate's spread; a fallback
        # (spread None) scores the legacy bar, as G2 pre-registers.
        spread = eff.ev_q95_spread if dsr_rule == "q95" else None
        column_ns = column_n(eff, candidates)
        trial_set = tuple(sorted({n for n in column_ns.values() if n is not None}))
        per_rep = replace(settings, dsr_trials=trial_set)
        chains = [
            cast(
                "dict[int, bool]",
                gatechain.evaluate_candidate(
                    entry.chromosome,
                    entry.fitness,
                    synth.holdout(cfg.window, RESULTS_N, 0.0, entry.seed).window.bt,
                    layer1._window_factory(entry.schedule),
                    closes,
                    per_rep,
                    null_spread=spread,
                )["chain_recent"],
            )
            for entry in entries
            if id(entry.chromosome) in picked
        ]
        for name, num_trials in column_ns.items():
            if num_trials is None:
                continue
            denoms[name] += 1
            counts[name] += int(any_pick_passed(chains, num_trials))

    rates = {
        name: (counts[name] / denoms[name] if denoms[name] else 0.0) for name in COLUMN_NAMES
    }
    cp_upper = clopper_pearson_upper(counts["ev"], reps)
    return {
        "key": cfg.key,
        "window": cfg.window,
        "rho": cfg.rho,
        "sparse": cfg.sparse,
        "drift": float(np.median(drifts)),
        "survivor_target": target if cfg.sparse else None,
        "survivor_min": min(kept_per_rep),
        "survivor_median": float(np.median(kept_per_rep)),
        "survivor_max": max(kept_per_rep),
        "survivor_undershoots": (
            sum(1 for d, k in zip(drifts, kept_per_rep, strict=True) if d == 0.0 and k < target)
            if cfg.sparse
            else None
        ),
        "survivor_out_of_band": (
            sum(1 for k in kept_per_rep if not SPARSE_BAND[0] <= k <= SPARSE_BAND[1])
            if cfg.sparse
            else None
        ),
        "reps": reps,
        "candidates": candidates,
        "count_ev": counts["ev"],
        "rate_ev": rates["ev"],
        "cp_upper_ev": cp_upper,
        "marginal": bool(threshold_exit_code(rates["ev"]) == 0 and cp_upper > THRESHOLD),
        "count_raw167": counts["raw167"],
        "rate_raw167": rates["raw167"],
        "rate_mc": rates["mc"],
        "rate_pr": rates["pr"],
        "rate_maxmcpr": rates["maxmcpr"],
        "legacy_reps": denoms["mc"],
        "median_run_n_eff": float(np.median(n_eff)),
        "median_n_used": float(np.median(n_used)),
        "fallback_rate": fallbacks / reps,
        "survivor_rate": kept_count / (reps * candidates),
        "kept_count": kept_count,
        "picks_per_rep": picks_total / reps,
        "threshold": THRESHOLD,
    }


def window_return(schedule: synth.Schedule, ws: str, we: str) -> float:
    """One eval window's compounded in-position return.

    The value ``synth.grid_window`` compounds to ``total_return`` for that window
    (flat days contribute a factor of 1), and the term
    ``NTBacktestFn._aggregate_multi_window`` takes the WORST of.
    """
    values = synth.window_returns(schedule, synth.day_index(ws), synth.day_index(we))
    return float(np.prod(1.0 + values) - 1.0)


def target_survivors(candidates: int) -> int:
    """The per-replicate survivor count the sparse shift aims at (17 of 167)."""
    return max(1, round(SPARSE_TARGET_SURVIVAL * candidates))


def survivor_count(schedules: Sequence[synth.Schedule], drift: float, plan: str) -> int:
    """How many of a replicate's schedules the production ``keep_series`` rule keeps.

    The rule is ``total_trades >= MIN_TRADES and total_return > 0``, the
    aggregate's ``total_return`` is the worst eval window, and the synthetic
    schedule always lays ``RESULTS_N`` trades -- so survival is exactly "every
    eval window's compounded in-position return is positive" (checked against
    the real fitness path candidate by candidate in the unit tests). Read off the
    schedules so the per-replicate bisection stays cheap.
    """
    windows = synth.PLANS[plan].eval_windows
    return sum(
        int(min(window_return(shifted(s, drift), ws, we) for ws, we in windows) > 0.0)
        for s in schedules
    )


def replicate_drift(
    cfg: Config,
    index: int,
    common: NDArray[np.float64],
    *,
    candidates: int = CONFIG_CANDIDATES,
    streams: SeedStreams = GRID_SEEDS,
    steps: int = CALIBRATION_STEPS,
) -> tuple[float, int]:
    """The replicate's scalar negative shift and the survivor count it achieves (N1).

    Survivors are non-increasing in the shift by construction, so the bracket
    reaches the target before the bisection starts. The shift is the bisected
    one whose count lands closest to ``target_survivors`` (the count is a step
    function, so the nearest of the two bracket ends). A replicate that keeps
    no more than the target unshifted gets drift 0 -- an undershoot, never a
    positive edge. Deterministic: the replicate's own stream, a fixed step count.
    """
    target = target_survivors(candidates)
    schedules = [
        candidate_schedule(replace(cfg, drift=0.0), index, i, common, streams=streams)[1]
        for i in range(candidates)
    ]
    count0 = survivor_count(schedules, 0.0, cfg.window)
    if count0 <= target:
        return 0.0, count0
    hi = SPARSE_DRIFT_HI
    count_hi = survivor_count(schedules, hi, cfg.window)
    while count_hi > target:
        hi *= 2.0
        count_hi = survivor_count(schedules, hi, cfg.window)
    lo, count_lo = 0.0, count0
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        count_mid = survivor_count(schedules, mid, cfg.window)
        if count_mid > target:
            lo, count_lo = mid, count_mid
        else:
            hi, count_hi = mid, count_mid
    if abs(count_lo - target) < abs(count_hi - target):
        return lo, count_lo
    return hi, count_hi


def load_closes(catalog: Path) -> dict[str, dict[int, float]]:
    """The real BTC 1d closes for the buy-and-hold gate (read-only)."""
    return layer1.load_closes(catalog)


_WORKER: dict[str, Any] = {}


def _init_worker(
    closes: Mapping[str, Mapping[int, float]],
    reps: int,
    candidates: int,
    streams: SeedStreams,
    dsr_rule: str,
) -> None:
    _WORKER.update(
        closes=closes, reps=reps, candidates=candidates, streams=streams, dsr_rule=dsr_rule
    )


def _worker(cfg: Config) -> dict[str, object]:
    with layer1._quiet_vibe_quant():
        return run_config(
            cfg,
            cast("int", _WORKER["reps"]),
            cast("Mapping[str, Mapping[int, float]]", _WORKER["closes"]),
            candidates=cast("int", _WORKER["candidates"]),
            streams=cast("SeedStreams", _WORKER["streams"]),
            dsr_rule=cast("str", _WORKER["dsr_rule"]),
        )


def run_grid(
    cfgs: Sequence[Config],
    reps: int,
    workers: int,
    closes: Mapping[str, Mapping[int, float]],
    *,
    candidates: int = CONFIG_CANDIDATES,
    streams: SeedStreams = GRID_SEEDS,
    dsr_rule: str = "emax",
) -> list[dict[str, object]]:
    """Run every config on a process pool (``workers <= 1`` runs inline).

    Identical seeds make the output identical for any worker count.
    """
    if workers <= 1:
        with layer1._quiet_vibe_quant():
            return [
                run_config(
                    cfg,
                    reps,
                    closes,
                    candidates=candidates,
                    streams=streams,
                    dsr_rule=dsr_rule,
                )
                for cfg in cfgs
            ]
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(closes, reps, candidates, streams, dsr_rule),
    ) as pool:
        return list(pool.map(_worker, cfgs))


def build_report(
    results: Sequence[Mapping[str, object]],
    *,
    reps: int,
    candidates: int,
    quick: bool,
    verdict: Verdict,
) -> dict[str, object]:
    """Assemble the JSON report (fully deterministic: no wall clock, no runtime)."""
    return {
        "job": JOB,
        "task": TASK,
        "seed_base": SEED_BASE,
        "quick": quick,
        "reps": reps,
        "candidates": candidates,
        "design_reps": DESIGN_REPS,
        "topup_reps": TOPUP_REPS,
        "windows": list(WINDOWS),
        "rho_grid": list(RHO_GRID),
        "sparse_rhos": list(SPARSE_RHOS),
        "sparse_target_survival": SPARSE_TARGET_SURVIVAL,
        "sparse_band": list(SPARSE_BAND),
        "threshold": THRESHOLD,
        "cp_level": CP_LEVEL,
        "columns": list(COLUMN_NAMES),
        "configs": list(results),
        "verdict": {
            "status": verdict.status,
            "exit_code": verdict.exit_code,
            "max_rate": verdict.max_rate,
            "failing": list(verdict.failing),
            "marginal": list(verdict.marginal),
            "topup": dict(verdict.topup),
            "line": verdict.line,
        },
    }


def write_report(report: Mapping[str, object], out: Path) -> None:
    """Write the report JSON to ``out`` (byte-identical for identical runs)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def print_report(report: Mapping[str, object], *, topup_rows: Sequence[Mapping[str, object]]) -> None:
    """Print the V2-neff verdict line, the per-config columns and the escalation."""
    verdict = cast("dict[str, Any]", report["verdict"])
    print(cast("str", verdict["line"]))
    for row in cast("list[dict[str, Any]]", report["configs"]):
        drift = f" drift={row['drift']:.6f}" if row["drift"] else ""
        if row["sparse"]:
            drift += (
                f" survivors/rep min={row['survivor_min']} "
                f"median={row['survivor_median']:.0f} max={row['survivor_max']} "
                f"target={row['survivor_target']} undershoots={row['survivor_undershoots']} "
                f"out_of_band={row['survivor_out_of_band']}"
            )
        print(
            f"neff column {row['key']}: count={row['count_ev']}/{row['reps']} "
            f"rate={row['rate_ev']:.4f} cp95={row['cp_upper_ev']:.4f} "
            f"run_n_eff={row['median_run_n_eff']:.0f} n_used={row['median_n_used']:.0f} "
            f"fallback={row['fallback_rate']:.2f} survivors={row['survivor_rate']:.2f} "
            f"picks={row['picks_per_rep']:.2f} raw167={row['rate_raw167']:.4f} "
            f"mc={row['rate_mc']:.4f} pr={row['rate_pr']:.4f} "
            f"maxmcpr={row['rate_maxmcpr']:.4f}{drift}"
        )
    for key in cast("list[str]", verdict["marginal"]):
        row = next(r for r in cast("list[dict[str, Any]]", report["configs"]) if r["key"] == key)
        print(
            f"neff marginal {key}: {row['count_ev']}/{row['reps']} "
            f"cp95={row['cp_upper_ev']:.4f} > {THRESHOLD}"
        )
    if topup_rows:
        for row in topup_rows:
            print(
                f"neff topup {row['key']}: {row['count_ev']}/{row['reps']} "
                f"rate={row['rate_ev']:.4f} seeds=0x{TOPUP_SEEDS.rep:X}"
            )
    elif verdict["status"] == "MARGINAL":
        print(
            f"neff marginal: only a {DESIGN_REPS}-rep run tops up to {TOPUP_REPS}; "
            f"this run used {report['reps']} reps"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: run the V2-neff grid (or a ``--quick`` plumbing check) and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT, type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--candidates", type=int, default=CONFIG_CANDIDATES)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG, type=Path)
    parser.add_argument(
        "--dsr-rule",
        choices=["emax", "q95"],
        default="emax",
        help="DSR critical value: emax (default, legacy) or q95 (this replicate's "
        "quantile-matched MC spread; G2 pre-registration).",
    )
    args = parser.parse_args(argv)

    reps = args.reps if args.reps is not None else (QUICK_REPS if args.quick else DESIGN_REPS)
    if args.workers is not None:
        workers = args.workers
    else:
        workers = min(DEFAULT_WORKERS, max(1, (os.cpu_count() or 2) - 2))
    start = time.perf_counter()
    with layer1._quiet_vibe_quant():
        closes = load_closes(Path(args.catalog))
        grid = configs()
        results = run_grid(
            grid,
            reps,
            workers,
            closes,
            candidates=args.candidates,
            streams=GRID_SEEDS,
            dsr_rule=args.dsr_rule,
        )
        base = build_verdict(results, reps=reps)
        topup_rows: list[dict[str, object]] = []
        topup: dict[str, float] = {}
        if base.status == "MARGINAL" and not args.quick and reps == DESIGN_REPS:
            marginal = [cfg for cfg in grid if cfg.key in base.marginal]
            topup_rows = run_grid(
                marginal,
                TOPUP_REPS,
                workers,
                closes,
                candidates=args.candidates,
                streams=TOPUP_SEEDS,
                dsr_rule=args.dsr_rule,
            )
            topup = {str(row["key"]): float(cast("float", row["rate_ev"])) for row in topup_rows}
        verdict = build_verdict(results, reps=reps, topup=topup or None)
    report = build_report(
        results, reps=reps, candidates=args.candidates, quick=args.quick, verdict=verdict
    )
    write_report(report, Path(args.out))
    print_report(report, topup_rows=topup_rows)
    print(f"neff-null wrote {args.out} in {time.perf_counter() - start:.1f}s")
    return verdict.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
