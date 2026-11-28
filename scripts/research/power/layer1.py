"""L1 statistical power grid: measured power of the REAL discovery gates.

Job 20261017-power-paper, task T2 (design-power.md §3 "L1 statistical layer",
§5 verdict rules, §6 rows R2/R4/R18/R19/R33, §7 row T2).

This module runs the whole L1 grid -- true Sharpe ``S`` x plan window
(``W3``/4h, ``W6``/1d) x train trade count ``n`` -- through the REAL chain
(``synth`` generates the series, ``gatechain`` scores every gate) and reports
the per-N pass rates the pre-registered verdicts read. It also runs the
selection null (R33) and prints the V1(a/b), V2, V3 and S50 lines from its own
JSON; ``--check`` re-derives the R18 analytic band from a saved JSON.

Analytic side (R2/R18): the DSR-alone critical Sharpe is the bisection root of
the repo's :class:`~vibe_quant.overfitting.dsr.DeflatedSharpeRatio` at p = 0.05,
annualized by ``sqrt(365)`` (the units of ``synth``). The R18 power is the
normal tail above that root at the generator's own kurtosis ``k = 9/f_real``
(lower bound) and at the normal ``k = 3`` (upper bound).

Not on the mypy baseline (``files = ["vibe_quant"]``); kept typed and
ruff-clean, imported by tests as a plain sibling (``import layer1``).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import multiprocessing
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import gatechain
import numpy as np
import synth
from scipy.stats import binom

from vibe_quant.discovery.buyhold import buy_hold_daily, daily_sharpe, load_daily_closes
from vibe_quant.discovery.fitness import (
    MIN_TRADES,
    FitnessResult,
    _const_backtest,
    _evaluate_single,
    passes_train_gate,
)
from vibe_quant.discovery.genome import generate_random_chromosome
from vibe_quant.discovery.neff import effective_trials
from vibe_quant.discovery.pipeline import _select_diverse_top_k
from vibe_quant.overfitting.dsr import DeflatedSharpeRatio

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from numpy.typing import NDArray

    from vibe_quant.discovery.operators import StrategyChromosome

#: The grid (design §7 row T2): S values, per-window train trade counts.
SEED_BASE: int = 20261017
S_VALUES: tuple[float, ...] = (0.0, 0.8, 1.2, 2.0, 3.0, 4.0)
WINDOW_N: dict[str, tuple[int, ...]] = {"W3": (70, 150, 500), "W6": (70, 150)}
#: The runs' own effective N (neff), reported as an info column (design §2).
NEFF_TRIALS: tuple[int, ...] = (7, 9, 11, 12)
#: Full-grid reps (design §7 row T2) and the --quick budget (< 60 s).
GRID_REPS: int = 500
QUICK_REPS: int = 50
#: Reps for the selection null: full run vs --quick.
SELECTION_REPS: int = 200
QUICK_SELECTION_REPS: int = 10
#: The selection null: K candidates, top-5 picks, any-of-5 (R33).
SELECTION_K: int = 167
SELECTION_TOP_K: int = 5
SELECTION_MIN_DISTANCE: float = 0.15
SELECTION_RHOS: tuple[float, ...] = (0.0, 0.5)
#: The verdict cell (design §5 V1/V3): S = 1.2, 150 train trades.
RESULTS_S: float = 1.2
RESULTS_N: int = 150
#: The production gate_n per plan window (design §2): 4h 22121, 1d 167.
PRODUCTION_N: dict[str, int] = {"W3": 22121, "W6": 167}
#: The single null (V2) sits at S = 0, N = 1.
SINGLE_NULL_N: int = 1
#: Significance and the ``sqrt(365)`` annualization of ``synth``'s units.
SIGNIFICANCE: float = 0.05
ANN_FACTOR: float = 365.0
#: The per-candidate gate marginals ``gatechain`` reports (R11).
MARGINAL_NAMES: tuple[str, ...] = (
    "min_trades",
    "min_return",
    "complexity",
    "bootstrap",
    "buy_hold",
    "holdout",
    "cross_window",
    "wfa",
)
WINDOWS: tuple[str, ...] = ("W3", "W6")
DEFAULT_OUT: str = "data/swarm/20261017-power-paper/power/layer1.json"
DEFAULT_CATALOG: str = "data/catalog"
JOB: str = "20261017-power-paper"
TASK: str = "pw-layer1"
#: ``--pick``: the production top-5 pick (default) or every train-passer, the
#: ``--gate-pool all`` pool (vibe-quant-clmmc.2, job 20261122-edge A-T3).
PICK_MODES: tuple[str, ...] = ("top5", "all")
#: The G-pool pre-registration's job-local outputs and labels (``--pick all``).
GPOOL_JOB: str = "20261122-edge"
GPOOL_DIR: str = "data/swarm/20261122-edge/prereg/gpool"
#: The selection top-up's tag, prefixed to BOTH seed derivations (R-ga3-1).
SELECTION_TOPUP_TAG: int = 0x5E1EC8
#: A seed-derivation prefix: one tag, or several (a ``--stream-tag`` run's top-up).
StreamTag = int | tuple[int, ...]
#: ``--fitness``: the selection null's fitness (vibe-quant-clmmc.3 T5a). The grid stays v1.
FITNESS_MODES: tuple[str, ...] = ("v1", "v2")
#: The G-fit pre-registration's job-local outputs (``--fitness v2``, arch-c3-1 §5 line (i)).
GFIT_DIR: str = "data/swarm/20261122-edge/prereg/gfit"

_INV_SQRT2: float = 1.0 / math.sqrt(2.0)


@dataclass(frozen=True, slots=True)
class SequentialRule:
    """A pre-registered two-look false-positive rule (R-ga3-1, dreview-b3-2 GATES-2).

    First look at ``reps``: ``<= early_pass`` PASS, ``>= fail_at`` FAIL, between
    -> an independent ``topup_reps`` run that PASSes iff ``<= topup_pass``. The
    cut-offs are a fixed table, not a CP95 bound on the first look: 11/400 and
    12/400 clear CP95 but would make P(PASS | true 0.05) too high, so they top up.
    """

    reps: int
    early_pass: int
    fail_at: int
    topup_reps: int
    topup_pass: int

    def first_look(self, count: int, reps: int) -> str:
        """PASS / FAIL / TOPUP, or NOT_BINDING for a sample that is not ``reps``."""
        if reps != self.reps:
            return "NOT_BINDING"
        if count <= self.early_pass:
            return "PASS"
        if count >= self.fail_at:
            return "FAIL"
        return "TOPUP"

    def topup_passes(self, count: int, reps: int) -> bool:
        """The top-up's own decision; a top-up of the wrong size fails closed."""
        return reps == self.topup_reps and count <= self.topup_pass

    def pass_probability(self, rate: float) -> float:
        """P(PASS) of one config whose true false-positive rate is ``rate``."""
        early = float(binom.cdf(self.early_pass, self.reps, rate))
        middle = float(binom.cdf(self.fail_at - 1, self.reps, rate)) - early
        return early + middle * float(binom.cdf(self.topup_pass, self.topup_reps, rate))


#: R-ga3-1: layer1's 4 selection configs (200 reps each) under ``--pick all``.
SELECTION_RULE: SequentialRule = SequentialRule(200, 3, 11, 800, 27)


@dataclass(frozen=True, slots=True)
class SequentialVerdict:
    """A grid's verdict under a :class:`SequentialRule` (exit 0 PASS, 1 FAIL, 2 MARGINAL)."""

    status: str
    exit_code: int
    failing: tuple[str, ...]
    marginal: tuple[str, ...]
    readings: str


def sequential_verdict(
    counts: Mapping[str, int],
    reps: int,
    rule: SequentialRule,
    topup: Mapping[str, tuple[int, int]] | None = None,
) -> SequentialVerdict:
    """PASS iff no config FAILs; a TOPUP config without its top-up -> MARGINAL.

    ``topup`` maps a config key to its top-up ``(count, reps)``. A run on a sample
    other than ``rule.reps`` (``--quick``) is MARGINAL whatever its counts.
    """
    stages = {key: rule.first_look(count, reps) for key, count in counts.items()}
    failing = tuple(key for key, stage in stages.items() if stage == "FAIL")
    marginal = tuple(key for key, stage in stages.items() if stage == "TOPUP")
    readings = ", ".join(f"{key}={count}/{reps}" for key, count in counts.items())
    if topup:
        readings += "; top-up " + ", ".join(f"{k}={c}/{r}" for k, (c, r) in topup.items())
    if failing:
        return SequentialVerdict("FAIL", 1, failing, marginal, readings)
    if "NOT_BINDING" in stages.values() or (marginal and topup is None):
        return SequentialVerdict("MARGINAL", 2, failing, marginal, readings)
    if marginal:
        assert topup is not None
        failing = tuple(
            key for key in marginal if key not in topup or not rule.topup_passes(*topup[key])
        )
        if failing:
            return SequentialVerdict("FAIL", 1, failing, marginal, readings)
    return SequentialVerdict("PASS", 0, failing, marginal, readings)


def _norm_sf(x: float) -> float:
    """Standard-normal survival function ``P(Z > x)`` via ``math.erfc``."""
    return 0.5 * math.erfc(x * _INV_SQRT2)


def critical_daily_sharpe(
    num_observations: int,
    num_trials: int,
    *,
    kurtosis: float = 3.0,
    skewness: float = 0.0,
    significance: float = SIGNIFICANCE,
) -> float:
    """The DSR critical Sharpe in PER-DAY units (R2), by bisection.

    The value at which the repo's :class:`DeflatedSharpeRatio` reports exactly
    ``significance``: the largest Sharpe a strategy may post and still be a
    plausible fluke after ``num_trials`` trials. ``p`` decreases monotonically
    in the Sharpe, so the root is bracketed and bisected to machine precision.
    """
    dsr = DeflatedSharpeRatio(significance)
    lo, hi = -1.0, 1.0
    if dsr.calculate(lo, num_trials, num_observations, skewness, kurtosis).p_value < significance:
        return lo
    if dsr.calculate(hi, num_trials, num_observations, skewness, kurtosis).p_value > significance:
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (
            dsr.calculate(mid, num_trials, num_observations, skewness, kurtosis).p_value
            > significance
        ):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def critical_sharpe_ann(
    num_observations: int,
    num_trials: int,
    *,
    kurtosis: float = 3.0,
    skewness: float = 0.0,
    significance: float = SIGNIFICANCE,
) -> float:
    """The R2 critical Sharpe annualized with ``sqrt(365)`` (the design §2 table)."""
    return critical_daily_sharpe(
        num_observations,
        num_trials,
        kurtosis=kurtosis,
        skewness=skewness,
        significance=significance,
    ) * math.sqrt(ANN_FACTOR)


def analytic_power(
    S: float,
    num_observations: int,
    num_trials: int,
    f_real: float,
    *,
    kurtosis: float | None = None,
    significance: float = SIGNIFICANCE,
) -> float:
    """Analytic DSR-alone power at true Sharpe ``S`` and realized exposure ``f_real`` (R18).

    The observed daily Sharpe is treated as ``N(S/sqrt(365), sd)`` with Lo's
    estimator variance ``sd = sqrt((1 + (k-1)/4 * s^2) / (T-1))`` at skew 0, and
    the test is ``SR_hat > crit`` with ``crit`` the DSR root at the same
    ``kurtosis``. ``kurtosis`` defaults to the generator's own ``9/f_real``
    (design §2/§6 R18); pass ``3.0`` for the normal-tail upper bound.
    """
    k = 9.0 / f_real if kurtosis is None else kurtosis
    s = S / math.sqrt(ANN_FACTOR)
    crit = critical_daily_sharpe(
        num_observations, num_trials, kurtosis=k, significance=significance
    )
    sd = math.sqrt((1.0 + (k - 1.0) / 4.0 * s * s) / (num_observations - 1))
    return _norm_sf((crit - s) / sd)


def _clamp_probability(p: float, reps: int) -> float:
    """R18/W-R1: clamp a plug-in probability to ``[1/reps, 1 - 1/reps]``."""
    return min(max(p, 1.0 / reps), 1.0 - 1.0 / reps)


def _binomial_se(p: float, reps: int) -> float:
    """The binomial standard error at ``p`` (clamped) for ``reps`` Bernoulli trials."""
    q = _clamp_probability(p, reps)
    return math.sqrt(q * (1.0 - q) / reps)


def inposition_lag1_autocorr(plan_name: str, n: int) -> float:
    """Lag-1 autocorrelation of the chained in-position indicator (info, W-R5).

    The trade schedule is seed-independent, so one schedule describes every rep.
    A multi-day hold makes consecutive active days likely, so the indicator is
    autocorrelated and Lo's iid variance understates the Sharpe dispersion.
    """
    indicator = synth.train_mtm_values(synth.build_schedule(plan_name, n, 0.0, SEED_BASE)) != 0.0
    x = indicator.astype(np.float64) - float(indicator.mean())
    denom = float(np.dot(x, x))
    return float(np.dot(x[:-1], x[1:]) / denom) if denom > 0.0 else 0.0


@dataclass(frozen=True, slots=True)
class Cell:
    """One grid cell: a plan window, a train trade count and a true Sharpe."""

    window: str
    timeframe: str
    n: int
    S: float
    w_idx: int
    s_idx: int
    n_idx: int


def cells() -> list[Cell]:
    """The 30-cell grid: S x (W3/4h n {70,150,500}; W6/1d n {70,150})."""
    out: list[Cell] = []
    for w_idx, window in enumerate(WINDOWS):
        timeframe = gatechain.settings_for(window).config.timeframe
        for n_idx, n in enumerate(WINDOW_N[window]):
            for s_idx, S in enumerate(S_VALUES):
                out.append(
                    Cell(
                        window=window,
                        timeframe=timeframe,
                        n=n,
                        S=S,
                        w_idx=w_idx,
                        s_idx=s_idx,
                        n_idx=n_idx,
                    )
                )
    return out


def cell_seeds(cell: Cell, reps: int, seed_base: int = SEED_BASE) -> list[int]:
    """R19: the cell's rep seeds, ``SeedSequence([base, w, s, n]).spawn(reps)``."""
    seq = np.random.SeedSequence([seed_base, cell.w_idx, cell.s_idx, cell.n_idx])
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in seq.spawn(reps)]


def _window_factory(
    sched: synth.Schedule,
) -> Callable[[str, str], Callable[[StrategyChromosome], dict[str, float | int]]]:
    """The cross-window / WFA backtest factory for a synthetic schedule.

    Range ``(start, end)`` -> the schedule's own window metrics through the real
    ``synth.make_window``, so ``gatechain`` scores cross-window and WFA on the
    same known-Sharpe series (never a stand-in).
    """

    def factory(start: str, end: str) -> Callable[[StrategyChromosome], dict[str, float | int]]:
        return _const_backtest(synth.make_window(sched, start, end).bt)

    return factory


def _rate(count: int, reps: int) -> float:
    return count / reps if reps else 0.0


def _analytic_block(
    S: float, T: int, trials: Sequence[int], f_real: float
) -> dict[str, dict[str, float]]:
    """The R18/R2 analytic readings of one cell, keyed by N."""
    k_gen = 9.0 / f_real
    return {
        str(N): {
            "kurt_9f": analytic_power(S, T, N, f_real),
            "kurt3": analytic_power(S, T, N, f_real, kurtosis=3.0),
            "crit_9f": critical_sharpe_ann(T, N, kurtosis=k_gen),
            "crit3": critical_sharpe_ann(T, N, kurtosis=3.0),
        }
        for N in trials
    }


def _r18_block(
    analytic: Mapping[str, Mapping[str, float]],
    dsr_alone: Mapping[int, int],
    trials: Sequence[int],
    reps: int,
) -> dict[str, dict[str, float]]:
    """W-R5: per-N empirical DSR-alone rate and the W-R1 analytic band."""
    out: dict[str, dict[str, float]] = {}
    for num_trials in trials:
        lo, hi = analytic[str(num_trials)]["kurt_9f"], analytic[str(num_trials)]["kurt3"]
        out[str(num_trials)] = {
            "empirical": _rate(dsr_alone[num_trials], reps),
            "band_lo": lo - 3.0 * _binomial_se(lo, reps),
            "band_hi": hi + 3.0 * _binomial_se(hi, reps),
        }
    return out


def run_cell(
    cell: Cell,
    reps: int,
    closes: Mapping[str, Mapping[int, float]],
    settings: gatechain.ChainSettings,
    seed_base: int = SEED_BASE,
) -> dict[str, object]:
    """Run one grid cell's ``reps`` independent draws through the real chain.

    Every rep is one synthetic train run (``synth``) scored by the real
    ``gatechain.evaluate_candidate``; the returned rates are the per-N pass
    rates of ``chain_recent`` and its leave-outs (R11), the per-gate marginals,
    the DSR-alone rate R18 checks, and the cell's analytic readings.
    """
    plan = synth.PLANS[cell.window]
    T = plan.chained_days
    hold = synth.hold_for(cell.n, T)
    f_real = synth.realized_exposure(cell.n, hold, T)
    trials = settings.dsr_trials
    prod_n = PRODUCTION_N[cell.window]

    chain_recent = dict.fromkeys(trials, 0)
    chain_minus_bh = dict.fromkeys(trials, 0)
    chain_minus_dsr = dict.fromkeys(trials, 0)
    dsr_alone = dict.fromkeys(trials, 0)
    marginals = dict.fromkeys(MARGINAL_NAMES, 0)
    chain_dsr_bh = 0
    floor_count = 0
    kurt_samples: list[float] = []

    for seed in cell_seeds(cell, reps, seed_base):
        chrom = generate_random_chromosome(random.Random(seed))
        gen = synth.generate(cell.window, cell.n, cell.S, seed)
        fit = _evaluate_single(chrom, _const_backtest(gen.bt), None, MIN_TRADES, cell.timeframe)
        holdout_bt = synth.holdout(cell.window, cell.n, cell.S, seed).window.bt
        out = gatechain.evaluate_candidate(
            chrom, fit, holdout_bt, _window_factory(gen.schedule), closes, settings
        )
        m = cast("dict[str, bool]", out["marginals"])
        recent = cast("dict[int, bool]", out["chain_recent"])
        minus_bh = cast("dict[int, bool]", out["chain_recent_minus_bh"])
        dsr = cast("dict[int, bool]", out["dsr"])
        records = cast("dict[int, dict[str, object]]", out["dsr_records"])
        floor = bool(out["floor"])
        floor_count += int(floor)
        for name in MARGINAL_NAMES:
            marginals[name] += int(m[name])
        kurt = records[prod_n].get("kurtosis_used")
        if isinstance(kurt, float):
            kurt_samples.append(kurt)
        no_dsr = bool(
            floor
            and m["min_trades"]
            and m["min_return"]
            and m["bootstrap"]
            and m["buy_hold"]
            and m["holdout"]
        )
        for n in trials:
            chain_recent[n] += int(recent[n])
            chain_minus_bh[n] += int(minus_bh[n])
            dsr_alone[n] += int(dsr[n])
            if no_dsr:
                chain_minus_dsr[n] += 1
        chain_dsr_bh += int(bool(out["chain_recent_minus_dsr_bh"]))

    analytic = _analytic_block(cell.S, T, trials, f_real)
    return {
        "window": cell.window,
        "timeframe": cell.timeframe,
        "n": cell.n,
        "S": cell.S,
        "chained_days": T,
        "hold": hold,
        "f_real": f_real,
        "kurtosis_9_over_f": 9.0 / f_real,
        "kurtosis_applied": {
            "mean": float(np.mean(kurt_samples)) if kurt_samples else None,
            "sd": float(np.std(kurt_samples)) if kurt_samples else None,
            "n": len(kurt_samples),
        },
        "inposition_lag1": inposition_lag1_autocorr(cell.window, cell.n),
        "reps": reps,
        "floor": _rate(floor_count, reps),
        "marginals": {name: _rate(count, reps) for name, count in marginals.items()},
        "chain_recent": {str(n): _rate(c, reps) for n, c in chain_recent.items()},
        "chain_recent_minus_bh": {str(n): _rate(c, reps) for n, c in chain_minus_bh.items()},
        "chain_recent_minus_dsr": {str(n): _rate(c, reps) for n, c in chain_minus_dsr.items()},
        "chain_recent_minus_dsr_bh": _rate(chain_dsr_bh, reps),
        "dsr_alone": {str(n): _rate(c, reps) for n, c in dsr_alone.items()},
        "analytic": analytic,
        "r18": _r18_block(analytic, dsr_alone, trials, reps),
    }


def _score(fitness: FitnessResult | float) -> float:
    return fitness.adjusted_score if isinstance(fitness, FitnessResult) else float(fitness)


def select_picks(
    scored: Sequence[tuple[StrategyChromosome, FitnessResult | float]],
    *,
    top_k: int = SELECTION_TOP_K,
    min_distance: float = SELECTION_MIN_DISTANCE,
    fitness: str = "v1",
) -> list[tuple[StrategyChromosome, FitnessResult | float]]:
    """R33: the production top-K pick over the positive-score candidates.

    ``scored`` is sorted by adjusted score descending; only ``adjusted_score > 0``
    candidates are eligible, and the pick is the repo's own
    :func:`~vibe_quant.discovery.pipeline._select_diverse_top_k` with the
    production ``top_k = 5`` / ``min_distance = 0.15``. ``fitness="v2"`` also
    requires ``passes_train_gate`` (R31): v2 scores a losing genome above 0.
    """
    positive = [
        (chrom, fit)
        for chrom, fit in scored
        if _score(fit) > 0.0
        and (
            fitness == "v1"
            or (isinstance(fit, FitnessResult) and passes_train_gate(fit, MIN_TRADES))
        )
    ]
    return _select_diverse_top_k(positive, top_k, min_distance)


def pick_pool(
    scored: Sequence[tuple[StrategyChromosome, FitnessResult]], pick: str, fitness: str = "v1"
) -> list[tuple[StrategyChromosome, FitnessResult]]:
    """The genomes the gates see: the production top-5 or, for ``all``, every train-passer.

    ``all`` mirrors ``--gate-pool all``: every ``passes_train_gate`` genome in
    ``scored`` order (both fitness modes, R16). Under v1 ``adjusted_score > 0``
    implies the train gate and under v2 the pick requires it, so the top-5 pick
    is always a subset of the ``all`` pool (A18).
    """
    if pick == "all":
        return [(chrom, fit) for chrom, fit in scored if passes_train_gate(fit, MIN_TRADES)]
    return cast(
        "list[tuple[StrategyChromosome, FitnessResult]]", select_picks(scored, fitness=fitness)
    )


def _null_schedule(
    plan_name: str, n: int, seed: int, common: NDArray[np.float64], rho: float
) -> synth.Schedule:
    """One S = 0 null candidate's schedule, with in-position noise correlation ``rho``.

    The candidate's own zero-Sharpe schedule is blended with a shared draw,
    ``x = sqrt(rho)*common + sqrt(1-rho)*own`` (both zero-mean, unit-variance on
    the same active days), so two candidates correlate at ``rho`` while the flat
    days stay flat.
    """
    sched = synth.build_schedule(plan_name, n, 0.0, seed)
    if rho > 0.0:
        sched = replace(sched, returns=np.sqrt(rho) * common + np.sqrt(1.0 - rho) * sched.returns)
    return sched


def _null_fitness(
    sched: synth.Schedule, chrom: StrategyChromosome, timeframe: str, fitness: str = "v1"
) -> FitnessResult:
    """Fitness of one S = 0 null candidate from its (already blended) schedule."""
    plan = sched.plan
    windows = tuple(synth.make_window(sched, ws, we) for ws, we in plan.eval_windows)
    bt = synth.aggregate_windows(windows)
    return _evaluate_single(
        chrom, _const_backtest(bt), None, MIN_TRADES, timeframe, fitness=fitness
    )


def selection_null(
    plan_name: str,
    rho: float,
    reps: int,
    closes: Mapping[str, Mapping[int, float]],
    settings: gatechain.ChainSettings,
    *,
    n: int = RESULTS_N,
    candidates: int = SELECTION_K,
    seed_base: int = SEED_BASE,
    dsr_rule: str = "emax",
    pick: str = "top5",
    stream_tag: StreamTag | None = None,
    fitness: str = "v1",
) -> dict[str, object]:
    """R33/V2: false-positive rate of one selection-null configuration.

    Each rep draws ``candidates`` random chromosomes (``random.Random(rep_seed+i)``),
    scores them with ``synth`` at S = 0, takes the production top-5 pick, and
    counts a false positive iff ANY pick passes ``chain_recent`` at N = 167.
    ``dsr_rule="q95"`` scores each rep with the effective-N estimator's spread
    over that rep's own retained series (design-dsr-q95); a fallback keeps the
    legacy bar.

    ``pick="all"`` scores every train-passer instead of the top-5 (the G-pool
    check) and adds ``fp_count``. ``stream_tag`` (``--stream-tag``, or the top-up's
    :func:`selection_topup_tag`) prefixes both seed derivations; None keeps them.
    ``fitness="v2"`` scores every candidate with v2 own, picks train-passers only
    (R31) and adds ``fitness`` + ``fp_count``.
    """
    timeframe = settings.config.timeframe
    tag = [] if stream_tag is None else list(_tag_tuple(stream_tag))
    w_idx = WINDOWS.index(plan_name)
    rho_idx = SELECTION_RHOS.index(rho)
    fp = 0
    picks_total = 0
    for rep in range(reps):
        rep_seed = int(
            np.random.SeedSequence([seed_base, *tag, w_idx, rho_idx, rep]).generate_state(
                1, dtype=np.uint32
            )[0]
        )
        # One shared noise path per replicate (fresh across replicates), so rho is
        # the correlation between candidates and not a conditional-on-one-shock FPR.
        common_seed = int(
            np.random.SeedSequence([seed_base, *tag, 0x5E1EC7, w_idx, rho_idx, rep]).generate_state(
                1, dtype=np.uint32
            )[0]
        )
        common = synth.build_schedule(plan_name, n, 0.0, common_seed).returns
        entries: list[tuple[StrategyChromosome, synth.Schedule, FitnessResult, int]] = []
        for i in range(candidates):
            cseed = (rep_seed + i) % (2**32)
            chrom = generate_random_chromosome(random.Random(cseed))
            sched = _null_schedule(plan_name, n, cseed, common, rho)
            fit = _null_fitness(sched, chrom, timeframe, fitness)
            entries.append((chrom, sched, fit, cseed))
        entries.sort(key=lambda entry: entry[2].adjusted_score, reverse=True)
        picks = pick_pool([(chrom, fit) for chrom, _, fit, _ in entries], pick, fitness)
        picks_total += len(picks)
        picked = {id(chrom) for chrom, _ in picks}
        spread: float | None = None
        if dsr_rule == "q95":
            rows = [
                np.frombuffer(fit.mtm_daily_returns.values, dtype=np.float64)
                for _, _, fit, _ in entries
                if fit.mtm_daily_returns is not None and len(fit.mtm_daily_returns) >= 2
            ]
            spread = effective_trials(rows, candidates).ev_q95_spread if rows else None
        for chrom, sched, fit, cseed in entries:
            if id(chrom) not in picked:
                continue
            holdout_bt = synth.holdout(plan_name, n, 0.0, cseed).window.bt
            out = gatechain.evaluate_candidate(
                chrom,
                fit,
                holdout_bt,
                _window_factory(sched),
                closes,
                settings,
                null_spread=spread,
            )
            recent = cast("dict[int, bool]", out["chain_recent"])
            if recent[SELECTION_K]:
                fp += 1
                break
    result: dict[str, object] = {
        "window": plan_name,
        "rho": rho,
        "reps": reps,
        "n": n,
        "candidates": candidates,
        "fp_rate": _rate(fp, reps),
        "picks_per_rep": _rate(picks_total, reps),
    }
    if pick == "all":
        result.update(pick=pick, fp_count=fp)
    if fitness == "v2":
        result.update(fitness=fitness, fp_count=fp)
    if stream_tag is not None:
        result["stream_tag"] = _tag_json(stream_tag)
    return result


_WORKER: dict[str, Any] = {}


def _init_worker(
    closes: Mapping[str, Mapping[int, float]],
    reps: int,
    seed_base: int,
    selection_reps: int,
    dsr_rule: str,
    pick: str = "top5",
    stream_tag: StreamTag | None = None,
    fitness: str = "v1",
) -> None:
    """Pool initializer: silence logs and cache the shared inputs per worker.

    The silencing applies only inside a pool CHILD (``workers <= 1`` runs the grid
    inline, where ``main``'s :func:`_quiet_vibe_quant` scopes the level and a bare
    ``run_grid`` must not leak it into the caller's process).
    """
    if multiprocessing.current_process().name != "MainProcess":
        logging.getLogger("vibe_quant").setLevel(logging.WARNING)
    _WORKER.update(
        closes=closes,
        reps=reps,
        seed_base=seed_base,
        selection_reps=selection_reps,
        dsr_rule=dsr_rule,
        pick=pick,
        stream_tag=stream_tag,
        fitness=fitness,
        settings={window: gatechain.settings_for(window) for window in WINDOWS},
    )


def _worker(cell: Cell) -> dict[str, object]:
    return run_cell(
        cell,
        cast("int", _WORKER["reps"]),
        cast("Mapping[str, Mapping[int, float]]", _WORKER["closes"]),
        cast("dict[str, gatechain.ChainSettings]", _WORKER["settings"])[cell.window],
        cast("int", _WORKER["seed_base"]),
    )


def _null_worker(key: tuple[str, float]) -> dict[str, object]:
    plan_name, rho = key
    return selection_null(
        plan_name,
        rho,
        cast("int", _WORKER["selection_reps"]),
        cast("Mapping[str, Mapping[int, float]]", _WORKER["closes"]),
        cast("dict[str, gatechain.ChainSettings]", _WORKER["settings"])[plan_name],
        seed_base=cast("int", _WORKER["seed_base"]),
        dsr_rule=cast("str", _WORKER["dsr_rule"]),
        pick=cast("str", _WORKER["pick"]),
        stream_tag=cast("StreamTag | None", _WORKER["stream_tag"]),
        fitness=cast("str", _WORKER["fitness"]),
    )


def run_grid(
    grid: Sequence[Cell],
    reps: int,
    workers: int,
    closes: Mapping[str, Mapping[int, float]],
    *,
    selection_reps: int = QUICK_SELECTION_REPS,
    seed_base: int = SEED_BASE,
    dsr_rule: str = "emax",
    pick: str = "top5",
    stream_tag: int | None = None,
    fitness: str = "v1",
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run the grid (and the selection null) on a process pool.

    ``workers <= 1`` runs inline; otherwise ``ProcessPoolExecutor`` with the
    shared ``closes`` cached in each worker. Identical seeds make the output
    identical for any worker count (R19). ``stream_tag`` and ``fitness`` reach
    the selection null only (the grid cells stay v1 on the untagged stream).
    """
    keys = [(window, rho) for window in WINDOWS for rho in SELECTION_RHOS]
    initargs = (closes, reps, seed_base, selection_reps, dsr_rule, pick, stream_tag, fitness)
    if workers <= 1:
        _init_worker(*initargs)
        return [_worker(cell) for cell in grid], [_null_worker(key) for key in keys]
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=initargs
    ) as pool:
        cell_results = list(pool.map(_worker, grid))
        null_results = list(pool.map(_null_worker, keys))
    return cell_results, null_results


def run_selection_topup(
    keys: Sequence[tuple[str, float]],
    reps: int,
    workers: int,
    closes: Mapping[str, Mapping[int, float]],
    *,
    dsr_rule: str = "emax",
    pick: str = "all",
    fitness: str = "v1",
    stream_tag: int | None = None,
) -> list[dict[str, object]]:
    """The selection configs' independent top-up on :func:`selection_topup_tag` (stream_tag)."""
    topup_tag = selection_topup_tag(stream_tag)
    initargs = (closes, 0, SEED_BASE, reps, dsr_rule, pick, topup_tag, fitness)
    if workers <= 1:
        _init_worker(*initargs)
        return [_null_worker(key) for key in keys]
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=initargs
    ) as pool:
        return list(pool.map(_null_worker, keys))


def selection_topup_tag(stream_tag: int | None) -> StreamTag:
    """The top-up's seed prefix: ``SELECTION_TOPUP_TAG``, or ``(stream_tag, it)`` (R-fv-3).

    A tagged run's top-up prefix is two entries long, so its seeds differ from the
    G-pool top-up's and from the tag's own selection reps (both one entry long).
    """
    return SELECTION_TOPUP_TAG if stream_tag is None else (stream_tag, SELECTION_TOPUP_TAG)


def _tag_tuple(tag: StreamTag) -> tuple[int, ...]:
    return (tag,) if isinstance(tag, int) else tag


def _tag_json(tag: StreamTag) -> int | list[int]:
    return tag if isinstance(tag, int) else list(tag)


def _selection_key(row: Mapping[str, object]) -> str:
    return f"{row['window']}:rho={row['rho']}"


def selection_verdict(
    null_results: Sequence[Mapping[str, object]],
    topup_results: Sequence[Mapping[str, object]] | None = None,
    *,
    label: str = "G-pool selection (pick=all)",
    topup_tag: StreamTag = SELECTION_TOPUP_TAG,
) -> dict[str, object]:
    """A selection line's verdict under ``SELECTION_RULE`` (``--pick all`` / ``--fitness v2``)."""
    counts = {_selection_key(row): int(cast("int", row["fp_count"])) for row in null_results}
    reps = {int(cast("int", row["reps"])) for row in null_results}
    topup = (
        None
        if topup_results is None
        else {
            _selection_key(row): (int(cast("int", row["fp_count"])), int(cast("int", row["reps"])))
            for row in topup_results
        }
    )
    # mixed sample sizes are not the pre-registered sample: NOT_BINDING via reps -1
    verdict = sequential_verdict(
        counts, reps.pop() if len(reps) == 1 else -1, SELECTION_RULE, topup
    )
    return {
        "status": verdict.status,
        "exit_code": verdict.exit_code,
        "failing": list(verdict.failing),
        "marginal": list(verdict.marginal),
        "counts": counts,
        "topup": {key: list(value) for key, value in (topup or {}).items()},
        "rule": {
            "reps": SELECTION_RULE.reps,
            "early_pass": SELECTION_RULE.early_pass,
            "fail_at": SELECTION_RULE.fail_at,
            "topup_reps": SELECTION_RULE.topup_reps,
            "topup_pass": SELECTION_RULE.topup_pass,
            "topup_tag": _tag_json(topup_tag),
        },
        "line": f"{label}: [{verdict.readings}] -> {verdict.status}",
    }


def _pick_cell(
    results: Sequence[Mapping[str, object]], window: str, n: int, S: float
) -> Mapping[str, object]:
    for cell in results:
        if cell["window"] == window and cell["n"] == n and cell["S"] == S:
            return cell
    msg = f"no cell for {window} n={n} S={S}"
    raise KeyError(msg)


def _rate_of(cell: Mapping[str, object], key: str, num_trials: int) -> float:
    block = cast("dict[str, float]", cell[key])
    return block[str(num_trials)]


def _s50(
    results: Sequence[Mapping[str, object]], window: str, num_trials: int, n: int = RESULTS_N
) -> float | None:
    """The true Sharpe where ``chain_recent`` crosses 50% (linear interp, R18 units).

    ``None`` when the grid never reaches 50% for that (window, N).
    """
    points = sorted(
        (cast("float", c["S"]), _rate_of(c, "chain_recent", num_trials))
        for c in results
        if c["window"] == window and c["n"] == n
    )
    prev: tuple[float, float] | None = None
    for S, rate in points:
        if rate >= 0.5:
            if prev is None or prev[1] >= 0.5:
                return S
            (s0, p0), (s1, p1) = prev, (S, rate)
            return s0 + (0.5 - p0) * (s1 - s0) / (p1 - p0)
        prev = (S, rate)
    return None


def _uplift(cell: Mapping[str, object], key: str, num_trials: int) -> float:
    return _rate_of(cell, key, num_trials) - _rate_of(cell, "chain_recent", num_trials)


def verdicts(report: Mapping[str, object]) -> dict[str, object]:
    """Design §5 V1(a/b), V2, V3 and S50, computed from the grid's own numbers."""
    results = cast("list[dict[str, object]]", report["cells"])
    selection = cast("list[dict[str, object]]", report["selection_null"])
    trials = cast("list[int]", report["dsr_trials"])

    v1a: dict[str, float] = {}
    v1b: dict[str, float] = {}
    v3: dict[str, object] = {}
    s50: dict[str, dict[str, float | None]] = {}
    neff: dict[str, dict[str, float]] = {}
    single_null: dict[str, float] = {}

    for window in WINDOWS:
        prod = PRODUCTION_N[window]
        result_cell = _pick_cell(results, window, RESULTS_N, RESULTS_S)
        v1a[window] = _rate_of(result_cell, "chain_recent", prod)
        v1b[window] = _rate_of(result_cell, "chain_recent_minus_bh", prod)

        marginal_rates: dict[str, float] = {
            name: cast("dict[str, float]", result_cell["marginals"])[name]
            for name in MARGINAL_NAMES
        }
        marginal_rates[f"dsr@{prod}"] = _rate_of(result_cell, "dsr_alone", prod)
        lowest = min(marginal_rates.items(), key=lambda kv: kv[1])
        dsr_bh = cast("float", result_cell["chain_recent_minus_dsr_bh"])
        uplifts = {
            "bh": _uplift(result_cell, "chain_recent_minus_bh", prod),
            "dsr": _uplift(result_cell, "chain_recent_minus_dsr", prod),
            "dsr_bh": dsr_bh - _rate_of(result_cell, "chain_recent", prod),
        }
        dominant = [name for name, value in uplifts.items() if value >= 0.20]
        v3[window] = {
            "production_n": prod,
            "lowest_marginal": {"gate": lowest[0], "rate": lowest[1]},
            "uplifts": uplifts,
            "dominant": dominant,
        }
        s50[window] = {str(N): _s50(results, window, N) for N in trials}
        neff[window] = {str(N): _rate_of(result_cell, "chain_recent", N) for N in NEFF_TRIALS}
        single_null[window] = max(
            (
                _rate_of(cell, "chain_recent", SINGLE_NULL_N)
                for cell in results
                if cell["window"] == window and cell["S"] == 0.0
            ),
            default=0.0,
        )

    selection_rates = {
        f"{null['window']}:rho={null['rho']}": cast("float", null["fp_rate"]) for null in selection
    }
    single_max = max(single_null.values())
    selection_max = max(selection_rates.values())
    v2 = {
        "single_null_max": single_max,
        "single_null_by_window": single_null,
        "selection_null": selection_rates,
        "selection_null_max": selection_max,
        "pass": single_max <= 0.05 and selection_max <= 0.05,
    }
    v1 = {
        "a": {
            "chain_recent": v1a,
            "adequate": all(rate >= 0.5 for rate in v1a.values()),
        },
        "b": {
            "chain_recent_minus_bh": v1b,
            "adequate": all(rate >= 0.5 for rate in v1b.values()),
        },
        "production_n": dict(PRODUCTION_N),
    }
    return {"v1": v1, "v2": v2, "v3": v3, "s50": s50, "neff": neff}


def check_r18(report: Mapping[str, object]) -> tuple[bool, list[str]]:
    """R18/W-R1: every cell's empirical DSR-alone power inside the analytic band.

    Pass iff ``analytic(k=9/f) - 3*SE <= empirical <= analytic(k=3) + 3*SE``,
    with each SE the binomial standard error at its own analytic probability
    (clamped to ``[1/reps, 1 - 1/reps]``).
    """
    reps = int(cast("int", report["reps"]))
    failures: list[str] = []
    for cell in cast("list[dict[str, object]]", report["cells"]):
        for num_trials, empirical in cast("dict[str, float]", cell["dsr_alone"]).items():
            block = cast("dict[str, dict[str, float]]", cell["analytic"])[num_trials]
            lo, hi = block["kurt_9f"], block["kurt3"]
            band_lo = lo - 3.0 * _binomial_se(lo, reps)
            band_hi = hi + 3.0 * _binomial_se(hi, reps)
            if not band_lo <= empirical <= band_hi:
                failures.append(
                    f"{cell['window']} n={cell['n']} S={cell['S']} N={num_trials}: "
                    f"empirical={empirical:.4f} band=[{band_lo:.4f}, {band_hi:.4f}]"
                )
    return (not failures, failures)


def build_report(
    cell_results: Sequence[dict[str, object]],
    null_results: Sequence[dict[str, object]],
    *,
    reps: int,
    selection_reps: int,
    seed_base: int = SEED_BASE,
) -> dict[str, object]:
    """Assemble the JSON report (deterministic except ``generated_at``)."""
    report: dict[str, object] = {
        "job": JOB,
        "task": TASK,
        "generated_at": datetime.now(UTC).isoformat(),
        "seed_base": seed_base,
        "reps": reps,
        "selection_reps": selection_reps,
        "s_values": list(S_VALUES),
        "window_n": {window: list(ns) for window, ns in WINDOW_N.items()},
        "dsr_trials": list(gatechain.DSR_TRIALS),
        "neff_trials": list(NEFF_TRIALS),
        "production_n": dict(PRODUCTION_N),
        "single_null_n": SINGLE_NULL_N,
        "results_cell": {"S": RESULTS_S, "n": RESULTS_N},
        "cells": list(cell_results),
        "selection_null": list(null_results),
    }
    report["verdicts"] = verdicts(report)
    return report


def strip_timestamps(report: Mapping[str, object]) -> dict[str, object]:
    """The report without its wall-clock fields, for the R19 determinism check."""
    return {key: value for key, value in report.items() if key != "generated_at"}


def print_verdicts(report: Mapping[str, object]) -> None:
    """Print the design §5 verdict lines from the report's own JSON."""
    verdict = cast("dict[str, Any]", report.get("verdicts") or verdicts(report))
    v1, v2, v3 = verdict["v1"], verdict["v2"], verdict["v3"]
    prod = cast("dict[str, int]", v1["production_n"])

    def _fmt(rates: Mapping[str, float], unit: str) -> str:
        return " ".join(f"{w}@N={prod[w]}={rates[w]:.4f}{unit}" for w in WINDOWS)

    print(
        f"V1(a) chain_recent: {_fmt(v1['a']['chain_recent'], '')} -> adequate={v1['a']['adequate']}"
    )
    print(
        f"V1(b) chain_recent_minus_bh: {_fmt(v1['b']['chain_recent_minus_bh'], '')} "
        f"-> adequate={v1['b']['adequate']}"
    )
    sel = ", ".join(f"{k}={v:.4f}" for k, v in sorted(v2["selection_null"].items()))
    print(
        f"V2 nulls: single max={v2['single_null_max']:.4f}; selection [{sel}] "
        f"max={v2['selection_null_max']:.4f} -> {'PASS' if v2['pass'] else 'FLAG (>0.05)'}"
    )
    for window in WINDOWS:
        entry = v3[window]
        uplift = entry["uplifts"]
        print(
            f"V3 {window} (N={entry['production_n']}): lowest marginal "
            f"{entry['lowest_marginal']['gate']}={entry['lowest_marginal']['rate']:.4f}; "
            f"d_bh={uplift['bh']:.4f} d_dsr={uplift['dsr']:.4f} d_dsr_bh={uplift['dsr_bh']:.4f}; "
            f"dominant={entry['dominant'] or 'none'} (>=0.20)"
        )
    for window in WINDOWS:
        line = " ".join(
            f"N={N}:{'none' if s is None else f'{s:.3f}'}"
            for N, s in verdict["s50"][window].items()
        )
        print(f"S50 {window}: {line}")
    for window in WINDOWS:
        line = " ".join(f"N={N}:{rate:.4f}" for N, rate in verdict["neff"][window].items())
        print(f"neff info {window} (S={RESULTS_S}, n={RESULTS_N}): {line}")


def _markdown(report: Mapping[str, object]) -> str:
    """A compact markdown view of the grid (design §3 "JSON + markdown table")."""
    lines = [
        f"# L1 power grid ({report['job']}, {report['reps']} reps)",
        "",
        "| window | n | S | f_real | N=1 | N=167 | N=22121 |",
        "|---|---|---|---|---|---|---|",
    ]
    for cell in cast("list[dict[str, object]]", report["cells"]):
        recent = cast("dict[str, float]", cell["chain_recent"])
        lines.append(
            f"| {cell['window']} | {cell['n']} | {cell['S']} | {cast('float', cell['f_real']):.4f} "
            f"| {recent.get('1', 0.0):.4f} | {recent.get('167', 0.0):.4f} "
            f"| {recent.get('22121', 0.0):.4f} |"
        )
    return "\n".join(lines) + "\n"


def chained_first_day(plan_name: str) -> int:
    """The chained train series' first day: window 1 drops its day-0 return (R4).

    ``NTBacktestFn._chain_window_series`` calls ``without_first_day`` on the
    first window, so the series the DSR and buy-and-hold gates read starts the
    day AFTER ``train_start`` and carries ``chained_days`` values.
    """
    plan = synth.PLANS[plan_name]
    return synth.day_index(plan.eval_windows[0][0]) + 1


def buy_hold_bar(plan_name: str, closes: Mapping[str, Mapping[int, float]]) -> float | None:
    """The W3/W6 train buy-and-hold daily Sharpe on the chained day set (R4).

    The exact benchmark :func:`~vibe_quant.discovery.buyhold.buy_hold_gate`
    builds for a synthetic candidate: the real BTC 1d closes over the chained
    train day set. ``None`` when the closes are unusable (fail closed).
    """
    plan = synth.PLANS[plan_name]
    values, _error = buy_hold_daily(
        closes, ["BTCUSDT"], chained_first_day(plan_name), plan.chained_days
    )
    return None if values is None else daily_sharpe(values)


def load_closes(catalog: Path) -> dict[str, dict[int, float]]:
    """The real BTC 1d closes for the buy-and-hold gate (read-only), R4."""
    return load_daily_closes(["BTCUSDT"], catalog)


def write_report(report: Mapping[str, object], out: Path) -> None:
    """Write the report JSON (+ its markdown view) to ``out``."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    out.with_suffix(".md").write_text(_markdown(report))


def _stream_tag(text: str) -> int:
    """``--stream-tag``: a decimal or ``0x`` hex seed tag, never a reserved stream's tag.

    0x5E1EC8 would replay the G-pool top-up stream and 0x5E1EC7 the common-shock stream.
    """
    tag = int(text, 0)
    if tag in (SELECTION_TOPUP_TAG, 0x5E1EC7):
        raise argparse.ArgumentTypeError(f"{text} is a reserved seed-stream tag")
    return tag


def default_out(pick: str, fitness: str, stream_tag: int | None, quick: bool) -> Path:
    """The report path without ``--out``: each non-default knob suffixes the name (R-fv-6).

    Default-knob runs keep their paths (P-1, G-pool, G-fit); ``--fitness v2`` with
    ``--pick all`` adds ``_fv2`` and a ``--stream-tag`` adds ``_tag<hex>``, so no run kind
    overwrites another's report.
    """
    if pick == "all":
        base = Path(GPOOL_DIR) / ("layer1_fv2" if fitness == "v2" else "layer1")
    elif fitness == "v2":
        base = Path(GFIT_DIR) / "layer1_v2"
    elif stream_tag is None:
        return Path(DEFAULT_OUT)  # the P-1 report; --quick never moved it
    else:
        base = Path(DEFAULT_OUT).with_suffix("")
    if stream_tag is not None:
        base = base.with_name(f"{base.name}_tag{stream_tag:x}")
    return base.with_suffix(".quick.json" if quick else ".json")


@contextlib.contextmanager
def _quiet_vibe_quant() -> Iterator[None]:
    """Silence the repo's INFO chatter for the enclosed run, then restore the level.

    The grid runs tens of thousands of gates; the level is saved and restored in
    ``finally`` so the silencing never outlives the call -- code running later in
    the same process (e.g. another test) still sees its own INFO records.
    """
    logger = logging.getLogger("vibe_quant")
    previous = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(previous)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: run the L1 grid (or ``--check`` a saved JSON) and print verdict lines."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, type=Path)
    parser.add_argument("--check", type=Path, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--selection-reps", type=int, default=None)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG, type=Path)
    parser.add_argument(
        "--dsr-rule",
        choices=["emax", "q95"],
        default="emax",
        help="V2 selection-null DSR rule: emax (default, legacy) or q95 (each "
        "rep's quantile-matched MC spread; design-dsr-q95).",
    )
    parser.add_argument(
        "--pick",
        choices=list(PICK_MODES),
        default="top5",
        help="Selection-null pick: top5 (default, production top-5) or all (every "
        "train-passer, the --gate-pool all pool; the G-pool rule and job-local output).",
    )
    parser.add_argument(
        "--fitness",
        choices=list(FITNESS_MODES),
        default="v1",
        help="Selection-null fitness: v1 (default) or v2 (v2 own scores, picks are train-"
        "passers with a positive score; the G-fit selection rule and job-local output). "
        "The grid cells stay v1.",
    )
    parser.add_argument(
        "--stream-tag",
        type=_stream_tag,
        default=None,
        help="Seed tag prefixed to both selection-null seed derivations (e.g. 0xF17E2; "
        "absent = the untagged stream). The v1 line on the same seeds is the same "
        "command with --fitness v1 and the same --stream-tag. The 800-rep top-up then "
        "draws its own stream [tag, 0x5E1EC8]. 0x5E1EC7/0x5E1EC8 are reserved; without "
        "--out the file name gains _tag<hex>.",
    )
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = default_out(args.pick, args.fitness, args.stream_tag, args.quick)

    with _quiet_vibe_quant():
        if args.check is not None:
            report = cast("dict[str, object]", json.loads(Path(args.check).read_text()))
            _ok, failures = check_r18(report)
            readings = len(cast("list[object]", report["cells"])) * len(
                cast("list[int]", report["dsr_trials"])
            )
            for line in failures:
                print(f"  R18 FAIL {line}")
            print(f"R18: {len(failures)} of {readings} cells outside band")
            print_verdicts(report)
            # W-R5: R18 is a sanity diagnostic, not a verdict gate -- always exit 0.
            return 0

        reps = args.reps if args.reps is not None else (QUICK_REPS if args.quick else GRID_REPS)
        selection_reps = (
            args.selection_reps
            if args.selection_reps is not None
            else (QUICK_SELECTION_REPS if args.quick else SELECTION_REPS)
        )
        workers = (
            args.workers if args.workers is not None else min(8, max(1, (os.cpu_count() or 2) - 2))
        )
        closes = load_closes(Path(args.catalog))
        cell_results, null_results = run_grid(
            cells(),
            reps,
            workers,
            closes,
            selection_reps=selection_reps,
            dsr_rule=args.dsr_rule,
            pick=args.pick,
            stream_tag=args.stream_tag,
            fitness=args.fitness,
        )
        report = build_report(cell_results, null_results, reps=reps, selection_reps=selection_reps)
        exit_code = 0
        # --pick all keeps its G-pool key and line; --fitness v2 (top5) is the G-fit line
        verdict_key = "gpool_selection" if args.pick == "all" else "gfit_selection"
        label = (
            "G-pool selection (pick=all)"
            if args.fitness == "v1"
            else f"G-fit selection (fitness={args.fitness}, pick={args.pick})"
        )
        if args.pick == "all" or args.fitness == "v2":
            topup_tag = selection_topup_tag(args.stream_tag)
            gpool = selection_verdict(null_results, label=label, topup_tag=topup_tag)
            if gpool["marginal"] and not args.quick and selection_reps == SELECTION_RULE.reps:
                keys = [
                    (str(row["window"]), float(cast("float", row["rho"])))
                    for row in null_results
                    if _selection_key(row) in cast("list[str]", gpool["marginal"])
                ]
                topup_rows = run_selection_topup(
                    keys,
                    SELECTION_RULE.topup_reps,
                    workers,
                    closes,
                    dsr_rule=args.dsr_rule,
                    pick=args.pick,
                    fitness=args.fitness,
                    stream_tag=args.stream_tag,
                )
                gpool = selection_verdict(
                    null_results, topup_rows, label=label, topup_tag=topup_tag
                )
            report.update(
                {"job": GPOOL_JOB, "pick": args.pick, "dsr_rule": args.dsr_rule, verdict_key: gpool}
            )
            if args.fitness == "v2":
                report["fitness"] = args.fitness
            exit_code = cast("int", gpool["exit_code"])
        out = Path(args.out)
        write_report(report, out)
        print_verdicts(report)
        if args.pick == "all" or args.fitness == "v2":
            print(cast("dict[str, str]", report[verdict_key])["line"])
        print(
            f"wrote {out} ({len(cell_results)} cells, {len(null_results)} selection-null configs)"
        )
        return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
