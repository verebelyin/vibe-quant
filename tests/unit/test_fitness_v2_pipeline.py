"""Fitness v2 in the discovery pipeline: the plateau step, neighbours as trials, own-vs-own selection.

vibe-quant-clmmc.3 T3. ``--fitness v2`` scores the top 5% of every generation a second time on
its one-step neighbours (``final = 0.5 * own + 0.5 * median(neighbour own)``). A neighbour is a
trial (it counts in the raw DSR N) but never a row of the effective-N series pool; crowding and
tournament compare OWN scores, the plateau final feeds elitism, top-K and export only.

The unit rows stub the evaluator (``pipeline.evaluate_population``) with controlled ``own``
values; the end-to-end rows use a deterministic MTM-series backtest through the real
``evaluate_population`` and, marked slow, real NautilusTrader.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from array import array
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from tests.unit.test_discovery_dsr_pooled import _fit
from tests.unit.test_discovery_gate_pool import _chrom as _gate_pool_chrom
from tests.unit.test_discovery_gate_pool import _FixedOps
from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH
from vibe_quant.discovery import pipeline as pipeline_mod
from vibe_quant.discovery import pooling, record
from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.discovery.fitness import FitnessResult, FitnessV2, _evaluate_single
from vibe_quant.discovery.neff import effective_trials
from vibe_quant.discovery.operators import StrategyChromosome, plateau_neighbours
from vibe_quant.discovery.pipeline import (
    DiscoveryConfig,
    DiscoveryEvaluationError,
    DiscoveryPipeline,
    DiscoveryResult,
)
from vibe_quant.errors import DataUnavailableError
from vibe_quant.metrics import DailyReturns

if TYPE_CHECKING:
    from collections.abc import Callable

key_of = DiscoveryPipeline._dsl_key
NO_FN: Any = object()  # the stubbed evaluator never calls it


def _ops(chroms: list[StrategyChromosome]) -> Any:
    return _FixedOps(chroms)


def _chrom(i: int) -> StrategyChromosome:
    """Genome ``i``: entry threshold 25 + i (below 25 the RSI genome has no valid neighbour)."""
    return _gate_pool_chrom(5 + i)


def _fr(
    own: float,
    *,
    trades: int = 200,
    ret: float | None = None,
    neighbours: tuple[float, ...] | None = None,
    error: str | None = None,
    adjusted: float | None = None,
    series: DailyReturns | None = None,
) -> FitnessResult:
    """A v2 result with the given own score; distinct ``own`` gives distinct behaviour fingerprints."""
    return FitnessResult(
        sharpe_ratio=2.0, max_drawdown=0.1, profit_factor=1.5, total_trades=trades,
        total_return=0.05 + own / 10 if ret is None else ret,
        complexity_penalty=0.0, overtrade_penalty=0.0, sl_tp_penalty=0.0, raw_score=own,
        adjusted_score=own if adjusted is None else adjusted, passed_filters=True,
        filter_results={}, error=error, mtm_daily_returns=series,
        fitness_v2=None if error else FitnessV2(own, 1.0, 1.0, neighbours),
    )  # fmt: skip


class _Eval:
    """Stand-in for ``pipeline.evaluate_population``: own score by DSL key, every call logged."""

    def __init__(
        self,
        own: dict[str, float] | None = None,
        default: float = 0.3,
        errors: set[str] | None = None,
        non_passers: set[str] | None = None,
        own_fn: Callable[[str], float] | None = None,
        fixed: dict[str, FitnessResult] | None = None,
    ) -> None:
        self.fixed = fixed or {}
        self.own = own or {}
        self.own_fn = own_fn
        self.default = default
        self.errors = errors or set()
        self.non_passers = non_passers or set()
        self.calls: list[list[str]] = []
        self.kwargs: list[dict[str, Any]] = []

    def __call__(self, chroms: list[StrategyChromosome], backtest_fn: Any, filter_fn: Any = None,
                 **kw: Any) -> list[FitnessResult]:  # fmt: skip
        keys = [key_of(c) for c in chroms]
        self.calls.append(keys)
        self.kwargs.append(kw)
        out: list[FitnessResult] = []
        for key in keys:
            if key in self.fixed:
                out.append(self.fixed[key])
            elif key in self.errors:
                out.append(_fr(0.0, error="RuntimeError: boom", trades=0, ret=-1.0))
            else:
                own = (
                    self.own[key]
                    if key in self.own
                    else (self.own_fn(key) if self.own_fn else self.default)
                )
                out.append(_fr(own, trades=10 if key in self.non_passers else 200))
        return out

    @property
    def evaluated(self) -> int:
        return sum(len(c) for c in self.calls)


def _config(population: int, **kw: Any) -> DiscoveryConfig:
    base: dict[str, Any] = {
        "population_size": population, "max_generations": 1, "elite_count": 1, "top_k": 5,
        "min_trades": 50, "max_workers": None, "timeframe": "4h", "start_date": "2024-01-01",
        "end_date": "2025-01-01", "require_bootstrap_ci": False, "require_buy_hold": False,
        "require_dsr": False, "min_diversity_distance": 0.0, "fitness": "v2", "gate_pool": "top_k",
    }  # fmt: skip
    base.update(kw)
    return DiscoveryConfig(**base)


def _seeded(
    monkeypatch: pytest.MonkeyPatch,
    population: int,
    own: dict[int, float] | None = None,
    *,
    ev: _Eval | None = None,
    **cfg: Any,
) -> tuple[DiscoveryPipeline, list[StrategyChromosome], _Eval]:
    """A v2 pipeline whose generation 0 is evaluated (cache seeded) but not yet plateau-scored."""
    chroms = [_chrom(i) for i in range(population)]
    if ev is None:
        ev = _Eval({key_of(chroms[i]): v for i, v in (own or {}).items()})
    monkeypatch.setattr(pipeline_mod, "evaluate_population", ev)
    pipe = DiscoveryPipeline(_config(population, **cfg), NO_FN, genome_ops=_ops(chroms))
    pipe._evaluate_new(chroms)
    return pipe, chroms, ev


def _scored_indices(pipe: DiscoveryPipeline, chroms: list[StrategyChromosome]) -> list[int]:
    return [
        i
        for i, c in enumerate(chroms)
        if (v2 := pipe.all_evaluated[key_of(c)].fitness_v2) is not None
        and v2.neighbours is not None
    ]


def _descending(n: int) -> dict[int, float]:
    return {i: round(0.9 - 0.01 * i, 2) for i in range(n)}


# --- config -----------------------------------------------------------------------------------


def test_config_fitness_is_validated_like_gate_pool() -> None:
    assert DiscoveryConfig().fitness == "v1"
    assert _config(10).fitness == "v2"
    with pytest.raises(ValueError, match="fitness must be 'v1' or 'v2'"):
        DiscoveryConfig(fitness="v3")  # type: ignore[arg-type]


def test_v2_sets_the_nt_train_fn_to_run_every_window_and_collect_series() -> None:
    fn = NTBacktestFn(["BTCUSDT"], "4h", "2024-01-01", "2024-02-01")
    DiscoveryPipeline(_config(10, require_dsr=False), fn)
    assert fn.full_windows is True and fn.collect_daily_returns is True
    # v1 with every series consumer off leaves the fn exactly as built
    v1 = NTBacktestFn(["BTCUSDT"], "4h", "2024-01-01", "2024-02-01")
    DiscoveryPipeline(_config(10, fitness="v1", require_dsr=False, require_buy_hold=False), v1)
    assert v1.full_windows is False and v1.collect_daily_returns is False


def test_v1_evaluator_call_is_unchanged_and_v2_forwards_the_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ev = _Eval()
    monkeypatch.setattr(pipeline_mod, "evaluate_population", ev)
    chroms = [_chrom(i) for i in range(4)]
    DiscoveryPipeline(_config(4, fitness="v1"), NO_FN, genome_ops=_ops(chroms))._evaluate_new(
        chroms
    )
    DiscoveryPipeline(_config(4), NO_FN, genome_ops=_ops(chroms))._evaluate_new(chroms)
    assert "fitness" not in ev.kwargs[0]
    assert ev.kwargs[1]["fitness"] == "v2"


# --- R18 / R18b: which genomes get a plateau score ----------------------------------------------


@pytest.mark.parametrize(
    ("population", "slots"), [(24, 2), (50, 3), (7, 1), (20, 1), (40, 2), (60, 3), (100, 5)]
)
def test_plateau_slots(monkeypatch: pytest.MonkeyPatch, population: int, slots: int) -> None:
    """ceil(0.05 * P) top keys, exactly (P = 20, 40, 60, 100 sit on the float edge)."""
    pipe, chroms, ev = _seeded(monkeypatch, population, _descending(population))
    pipe._apply_plateau(chroms, 0)
    assert _scored_indices(pipe, chroms) == list(range(slots))
    assert ev.evaluated == population + 4 * slots  # 4 neighbours per scored genome


def test_plateau_skips_non_passers_duplicates_and_ranks_ties_by_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chroms = [_chrom(i) for i in range(24)]
    keys = [key_of(c) for c in chroms]
    own = {keys[i]: v for i, v in _descending(24).items()}
    ev = _Eval(own, non_passers={keys[0], keys[1], keys[2]})
    pipe, population, _ = _seeded(monkeypatch, 24, ev=ev)
    pipe._apply_plateau(population, 0)
    assert _scored_indices(pipe, population) == [3, 4]  # the three best fail the train gate
    # a duplicated DSL is one key: slots 0 and 1 share it, so the second slot goes to genome 2
    pipe = DiscoveryPipeline(_config(24), NO_FN, genome_ops=_ops(chroms))
    monkeypatch.setattr(pipeline_mod, "evaluate_population", _Eval(own))
    clones = [_chrom(0), _chrom(0), *[_chrom(i) for i in range(2, 24)]]
    pipe._evaluate_new(clones)
    pipe._apply_plateau(clones, 0)
    assert _scored_indices(pipe, clones) == [0, 1, 2]
    # equal scores: the smaller DSL key takes the single slot
    pipe, population, _ = _seeded(monkeypatch, 7, ev=_Eval({keys[0]: 0.8, keys[1]: 0.8}))
    pipe._apply_plateau(population, 0)
    (winner,) = _scored_indices(pipe, population)
    assert key_of(population[winner]) == min(keys[0], keys[1])


def test_pick_then_skip_scores_nothing_when_the_top_slots_are_already_scored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P = 24: the two best are plateau-scored (neighbours as good as they are) and still rank above
    every unscored train-passer, so the next generation spends 0 neighbour evaluations."""
    pipe, chroms, ev = _seeded(monkeypatch, 24, _descending(24))
    for neighbour in (
        n for c in chroms[:2] for n in plateau_neighbours(c, key_of, max_neighbours=4)
    ):
        ev.own[key_of(neighbour)] = 0.88
    pipe._apply_plateau(chroms, 0)
    assert _scored_indices(pipe, chroms) == [0, 1]
    before = ev.evaluated
    results = pipe._apply_plateau(chroms, 1)
    assert ev.evaluated == before
    assert _scored_indices(pipe, chroms) == [0, 1]
    assert [r.adjusted_score for r in results[:3]] == [
        0.5 * 0.9 + 0.5 * 0.88,
        0.5 * 0.89 + 0.5 * 0.88,
        0.88,
    ]  # the scored two still outrank the unscored third


# --- R19: the plateau final ---------------------------------------------------------------------


def _plateau_one(
    monkeypatch: pytest.MonkeyPatch,
    values: list[float | None],
    *,
    own: float = 0.8,
    cached: dict[int, FitnessResult] | None = None,
) -> tuple[FitnessResult, DiscoveryPipeline]:
    """Plateau-score genome 0 (own ``own``) whose neighbours score ``values`` (None = error)."""
    chroms = [_chrom(i) for i in range(10)]
    neighbours = plateau_neighbours(chroms[0], key_of, max_neighbours=4)
    assert len(neighbours) == len(values)
    ev = _Eval({key_of(chroms[0]): own}, default=0.1)
    for n, v in zip(neighbours, values, strict=True):
        if v is None:
            ev.errors.add(key_of(n))
        else:
            ev.own[key_of(n)] = v
    for idx, fr in (cached or {}).items():
        ev.fixed[key_of(neighbours[idx])] = fr
    pipe, population, _ = _seeded(monkeypatch, 10, ev=ev)
    pipe._evaluate_new([neighbours[idx] for idx in (cached or {})])  # already known to the run
    pipe._apply_plateau(population, 0)
    return pipe.all_evaluated[key_of(population[0])], pipe


def test_plateau_final_is_half_own_half_median_of_neighbour_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fr, _ = _plateau_one(monkeypatch, [0.6, 0.7, None, 0.5])
    assert fr.adjusted_score == 0.7
    assert fr.fitness_v2 is not None
    assert fr.fitness_v2.own == 0.8
    assert fr.fitness_v2.neighbours == (0.6, 0.7, 0.5)
    assert fr.fitness_v2.neighbour_errors == 1
    even, _ = _plateau_one(monkeypatch, [0.6, 0.7, None, None])
    assert even.adjusted_score == 0.5 * 0.8 + 0.5 * (
        (0.6 + 0.7) / 2
    )  # even count: mean of the middle two


def test_plateau_with_every_neighbour_failing_keeps_own(monkeypatch: pytest.MonkeyPatch) -> None:
    fr, pipe = _plateau_one(monkeypatch, [None, None, None, None])
    assert fr.adjusted_score == 0.8
    assert fr.fitness_v2 is not None
    assert (fr.fitness_v2.neighbours, fr.fitness_v2.neighbour_errors) == ((), 4)
    assert pipe._plateau_stats["neighbour_errors"] == 4


def test_plateau_without_candidates_keeps_own(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_mod, "plateau_neighbours", lambda *a, **kw: [])
    chroms = [_chrom(i) for i in range(10)]
    pipe, population, ev = _seeded(monkeypatch, 10, {0: 0.8})
    pipe._apply_plateau(population, 0)
    fr = pipe.all_evaluated[key_of(chroms[0])]
    assert fr.adjusted_score == 0.8
    assert fr.fitness_v2 is not None and fr.fitness_v2.neighbours == ()
    assert ev.evaluated == 10  # nothing evaluated


def test_a_plateau_scored_neighbour_contributes_its_own_not_its_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = _fr(0.9, neighbours=(0.1,), adjusted=0.5)  # own 0.9, final 0.5
    fr, pipe = _plateau_one(monkeypatch, [0.6, 0.7, 0.5, 0.4], cached={0: prior})
    assert fr.fitness_v2 is not None
    assert fr.fitness_v2.neighbours == (0.9, 0.7, 0.5, 0.4)  # a cache hit: not re-evaluated
    assert fr.adjusted_score == 0.5 * 0.8 + 0.5 * 0.6  # median(0.9, 0.7, 0.5, 0.4) = 0.6
    assert pipe._total_evaluated == 10 + 1 + 3  # the known neighbour is not a second trial


# --- R20: neighbours are trials, never series-pool rows -----------------------------------------


def test_neighbours_are_counted_cached_pooled_and_kept_out_of_the_series_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe, chroms, ev = _seeded(monkeypatch, 10, {0: 0.9}, gate_pool="all")
    ga_keys = {key_of(c) for c in chroms}
    nbrs = plateau_neighbours(chroms[0], key_of, max_neighbours=4)
    nbr_keys = {key_of(n) for n in nbrs}
    pipe._apply_plateau(chroms, 0)
    assert pipe._total_evaluated == 10 + 4
    assert pipe._plateau_keys == nbr_keys
    assert set(pipe.all_evaluated) == ga_keys | nbr_keys
    assert set(pipe.fitness_cache) == ga_keys  # the series-pool source has no neighbour
    assert nbr_keys <= set(pipe._pool_chroms)  # a train-passing neighbour is still screened
    pipe._evaluate_new(nbrs)  # cached by DSL key: not a second trial
    assert pipe._total_evaluated == 14
    assert pipe._plateau_stats == {"scored": 1, "neighbour_evals": 4, "neighbour_errors": 0}


def test_a_neighbour_that_is_already_a_ga_genome_stays_in_the_series_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R-c3-1: only keys FIRST evaluated in a plateau batch leave the series pool."""
    leader = _chrom(0)
    twin = plateau_neighbours(leader, key_of, max_neighbours=4)[0]
    chroms = [leader, twin, *(_chrom(i) for i in range(2, 10))]
    ev = _Eval({key_of(leader): 0.9})
    monkeypatch.setattr(pipeline_mod, "evaluate_population", ev)
    pipe = DiscoveryPipeline(_config(10), NO_FN, genome_ops=_ops(chroms))
    pipe._evaluate_new(chroms)
    pipe._apply_plateau(chroms, 0)
    assert key_of(twin) not in pipe._plateau_keys
    assert key_of(twin) in pipe.fitness_cache
    assert pipe._total_evaluated == 10 + 3  # the twin was not a second trial


def test_an_all_failed_neighbour_batch_never_aborts_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chroms = [_chrom(i) for i in range(10)]
    nbrs = plateau_neighbours(chroms[0], key_of, max_neighbours=4)
    ev = _Eval({key_of(chroms[0]): 0.9}, errors={key_of(n) for n in nbrs})
    pipe, population, _ = _seeded(monkeypatch, 10, ev=ev)
    pipe._apply_plateau(population, 0)  # no DiscoveryEvaluationError
    assert pipe.all_evaluated[key_of(population[0])].adjusted_score == 0.9
    # the same batch through the GA path is a systemic failure
    fresh = [_chrom(i) for i in range(20, 24)]
    ev.errors.update(key_of(c) for c in fresh)
    with pytest.raises(DiscoveryEvaluationError, match="all 4 evaluations failed"):
        pipe._evaluate_new(fresh)


def test_missing_data_in_a_neighbour_batch_fails_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe, population, ev = _seeded(monkeypatch, 10, {0: 0.9})

    def missing(*a: Any, **kw: Any) -> list[FitnessResult]:
        raise DataUnavailableError("no bars")

    monkeypatch.setattr(pipeline_mod, "evaluate_population", missing)
    with pytest.raises(DataUnavailableError, match="no bars"):
        pipe._apply_plateau(population, 0)


def _correlated_rows(
    rng: np.random.Generator, n_ga: int, n_lead: int, per_leader: int, corr: float
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    t = 800
    common = rng.normal(0, 1, t)
    ga = [0.01 * (np.sqrt(0.2) * common + np.sqrt(0.8) * rng.normal(0, 1, t)) for _ in range(n_ga)]
    nbr = [
        0.01 * (corr * ga[i] / 0.01 + np.sqrt(1 - corr**2) * rng.normal(0, 1, t))
        for i in range(n_lead) for _ in range(per_leader)
    ]  # fmt: skip
    return ga, nbr


def _with_series(rows: list[np.ndarray]) -> list[FitnessResult]:
    return [_fit(mtm_daily_returns=DailyReturns(19000, array("d", r.tobytes()))) for r in rows]


EXPECTED_N_EFF = 151  # MEASURED: GA-only (raw 185) 102; neighbours in the series pool (raw 273) 86


def test_neighbour_series_do_not_lower_the_effective_n(monkeypatch: pytest.MonkeyPatch) -> None:
    """40 kept GA series + 88 train-passing neighbours at corr 0.9 to 11 leaders: with neighbours in
    the series pool the gate N DROPS below the GA-only N (fail open); the pipeline keeps them out."""
    ga_rows, nbr_rows = _correlated_rows(np.random.default_rng(7), 40, 11, 8, 0.9)
    ga_only = effective_trials(ga_rows, 185).run_n_eff
    in_pool = effective_trials(ga_rows + nbr_rows, 185 + 88).run_n_eff
    assert (ga_only, in_pool) == (
        102,
        86,
    )  # the premise: the mutant (neighbours in series_pool) fails open

    ga = [_chrom(i) for i in range(40)]
    quiet = [
        _chrom(200 + i) for i in range(145)
    ]  # the GA's other 145 genomes: no train pass, no series
    nbrs = [_chrom(100 + i) for i in range(88)]
    fixed = {
        key_of(c): _fit(mtm_daily_returns=fr.mtm_daily_returns)
        for c, fr in zip(ga + nbrs, _with_series(ga_rows + nbr_rows), strict=True)
    }
    ev = _Eval(fixed=fixed, non_passers={key_of(c) for c in quiet})
    monkeypatch.setattr(pipeline_mod, "evaluate_population", ev)
    pipe = DiscoveryPipeline(_config(10, require_dsr=True), NO_FN)
    pipe._evaluate_new(ga + quiet)
    pipe._evaluate_new(nbrs)
    pipe._plateau_keys.update(key_of(c) for c in nbrs)  # what a plateau batch records
    assert pipe._total_evaluated == 185 + 88
    counts = pipe.gates.trial_counts(pipe.trial_budget())
    expected = effective_trials(ga_rows, 185 + 88).run_n_eff
    assert counts["run_n_eff"] == expected == EXPECTED_N_EFF
    assert counts["run_n_eff"] >= ga_only
    assert (counts["raw"], counts["non_flat"]) == (273, 40)


def _independent_rows(seed: int, n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [0.01 * rng.normal(0, 1, 800) for _ in range(n)]


def test_pooled_budget_leaves_neighbours_out_of_the_series_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-seed: raw N is the sum of every seed's evaluations (neighbours incl.); the pooled
    effective-N series are the GA rows only."""
    seeds = iter([(1, 3, 2), (2, 3, 4)])  # (rng seed, GA series rows, neighbour series rows)

    def fake_run(self: DiscoveryPipeline) -> DiscoveryResult:
        seed, n_ga, n_nbr = next(seeds)
        for i, fr in enumerate(_with_series(_independent_rows(seed, n_ga + n_nbr))):
            self._fitness_cache[f"s{seed}-{i}"] = fr
            if i >= n_ga:
                self._plateau_keys.add(f"s{seed}-{i}")
        champion = _with_series(_independent_rows(99, 1))[0]
        return DiscoveryResult([], [(_chrom(0), champion)], 100 + n_nbr, False, None)

    monkeypatch.setattr(DiscoveryPipeline, "run", fake_run)
    cfg = _config(10, require_dsr=True, prior_trials_fn=lambda s, e: 0)
    merged = pooling._run_multi_seed(2, cfg, NO_FN, "/tmp/cc-c3t3-1/progress.json", base_seed=7)
    assert merged.total_candidates_evaluated == 102 + 104
    assert merged.dsr_trials is not None
    assert merged.dsr_trials["raw"] == 206
    assert merged.dsr_trials["non_flat"] == 6  # 3 + 3 GA rows; the 6 neighbour rows are not pooled


def test_v2_candidates_need_the_train_gate_and_a_positive_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under v2 ``_all_scored`` holds train-passers scoring > 0 (0.0 out, 0.0001 in); v1 keeps its rule."""
    chroms = [_chrom(i) for i in range(3)]
    keys = [key_of(c) for c in chroms]
    ev = _Eval({keys[0]: 0.0, keys[1]: 0.0001, keys[2]: 0.5}, non_passers={keys[2]})
    for fitness, expected in (("v2", [keys[1]]), ("v1", [keys[1], keys[2]])):
        monkeypatch.setattr(pipeline_mod, "evaluate_population", ev)
        pipe = DiscoveryPipeline(_config(3, fitness=fitness), NO_FN, genome_ops=_ops(chroms))
        pipe._evaluate_new(chroms)
        assert [key_of(c) for c, _ in pipe._all_scored] == expected


# --- R-c3-2: crowding and tournament compare OWN vs OWN --------------------------------------------


def _own_final(chrom: StrategyChromosome, own: float, final: float) -> FitnessResult:
    return _fr(own, neighbours=(0.0,), adjusted=final, ret=0.05 + own / 10)


def test_crowding_compares_own_against_own_and_elitism_reads_the_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parent own 0.62 / final 0.55 vs an unscored offspring (own 0.60): the PARENT stays (0.62 >
    0.60); under final-vs-own the offspring would win (0.60 > 0.55). Elitism keeps the best FINAL."""
    elite, parent, other = _chrom(0), _chrom(8), _chrom(16)
    pop = [elite, parent, other]
    fits = [
        _own_final(elite, 0.5, 0.65),
        _own_final(parent, 0.62, 0.55),
        _own_final(other, 0.4, 0.4),
    ]
    kid_p, kid_o = _chrom(9), _chrom(17)  # each next to its own parent
    kid_fit = {kid_p.uid: _fr(0.60), kid_o.uid: _fr(0.45)}
    pipe = DiscoveryPipeline(_config(3), NO_FN, genome_ops=_ops(pop))
    pipe._breed = lambda a, b: (
        [{parent.uid: kid_p, other.uid: kid_o}[g.uid] for g in (a, b)],
        0,
        0,
    )  # type: ignore[method-assign,assignment,misc]
    pipe._evaluate_new = lambda kids: [kid_fit[k.uid] for k in kids]  # type: ignore[method-assign,assignment,misc]
    random.seed(0)
    new_pop, new_known = pipe._evolve_crowding(pop, fits)
    uids = [c.uid for c in new_pop]
    assert uids[0] == elite.uid  # elitism: final 0.65 beats the parent's own 0.62
    assert parent.uid in uids and kid_p.uid not in uids  # own 0.62 > own 0.60
    assert (
        kid_o.uid in uids and other.uid not in uids
    )  # own 0.45 >= own 0.40: the child replaces it
    assert new_known[uids.index(parent.uid)] is fits[1]


def test_tournament_selects_on_own_and_elitism_on_the_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a, b = _chrom(0), _chrom(8)
    pop = [a, b]
    fits = [_own_final(a, 0.62, 0.55), _own_final(b, 0.60, 0.60)]
    pipe = DiscoveryPipeline(
        _config(2, use_crowding=False, tournament_size=2, elite_count=1),
        NO_FN,
        genome_ops=_ops(pop),
    )
    parents: list[str] = []

    def breed(
        x: StrategyChromosome, y: StrategyChromosome
    ) -> tuple[list[StrategyChromosome], int, int]:
        parents.extend([x.uid, y.uid])
        return [_chrom(30), _chrom(31)], 0, 0

    pipe._breed = breed  # type: ignore[method-assign,assignment,misc]
    new_pop, _ = pipe._evolve_tournament(pop, fits)
    assert new_pop[0].uid == b.uid  # the elite is the best FINAL (0.60 > 0.55)
    assert parents == [a.uid, a.uid]  # both tournaments pick the best OWN (0.62 > 0.60)


# --- R21 / R22 / R34: top-K, coherence, the best score after the plateau ------------------------------


def _run_one_generation(
    monkeypatch: pytest.MonkeyPatch, population: int, own: dict[int, float], **cfg: Any
) -> tuple[DiscoveryResult, DiscoveryPipeline, list[StrategyChromosome], _Eval]:
    chroms = [_chrom(i) for i in range(population)]
    ev = _Eval({key_of(chroms[i]): v for i, v in own.items()}, default=0.2)
    # genome 1's neighbours are as good as it is: its final stays 0.8
    for n in plateau_neighbours(chroms[1], key_of, max_neighbours=4):
        ev.own[key_of(n)] = own.get(1, 0.2)
    monkeypatch.setattr(pipeline_mod, "evaluate_population", ev)
    random.seed(1)
    pipe = DiscoveryPipeline(_config(population, **cfg), NO_FN, genome_ops=_ops(chroms))
    return pipe.run(), pipe, chroms, ev


def test_top_k_champions_come_from_plateau_scored_genomes_ranked_by_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P = 40 -> 2 plateau slots: genome 0 (own 0.9, weak neighbours: final 0.55) and genome 1 (own
    0.8, neighbours as good: final 0.8). Genome 2 (own 0.7, unscored) never reaches the gates."""
    res, pipe, chroms, _ = _run_one_generation(monkeypatch, 40, {0: 0.9, 1: 0.8, 2: 0.7})
    assert [c.uid for c, _ in res.top_strategies] == [chroms[1].uid, chroms[0].uid]
    assert [fr.adjusted_score for _, fr in res.top_strategies] == [0.8, 0.5 * 0.9 + 0.5 * 0.2]
    assert res.plateau == {
        "scored": 2, "neighbour_evals": 8, "neighbour_errors": 0,
        "top_fraction": 0.05, "max_neighbours": 4, "step_fraction": 0.05,
    }  # fmt: skip


def test_all_pool_still_screens_every_train_passer_including_neighbours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    res, pipe, chroms, _ = _run_one_generation(monkeypatch, 40, {0: 0.9, 1: 0.8}, gate_pool="all")
    assert {key_of(c) for c in chroms} <= set(pipe._pool_chroms)
    assert pipe._plateau_keys and pipe._plateau_keys <= set(pipe._pool_chroms)
    nbr_uids = {c.uid for k, c in pipe._pool_chroms.items() if k in pipe._plateau_keys}
    screened = {r["uid"] for r in res.guardrail_rejections} | {c.uid for c, _ in res.top_strategies}
    assert nbr_uids <= screened  # every train-passing neighbour reached the gates


def test_best_fitness_is_read_after_the_plateau(monkeypatch: pytest.MonkeyPatch) -> None:
    """Genome 0 (own 0.9) is plateau-scored down to 0.55: the generation's best is genome 1's 0.8."""
    res, _, _, _ = _run_one_generation(monkeypatch, 10, {0: 0.9, 1: 0.8})
    assert res.generations[0].best_fitness == 0.8


def test_cache_returns_the_final_after_a_plateau(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe, chroms, ev = _seeded(monkeypatch, 10, {0: 0.9})
    before = ev.evaluated
    slots = pipe._apply_plateau(chroms, 0)
    final = 0.5 * 0.9 + 0.5 * 0.3
    assert slots[0].adjusted_score == final and slots[0].fitness_v2 is not None
    assert slots[0].fitness_v2.neighbours is not None
    assert slots[0].fitness_v2.own == 0.9
    reproduced = pipe._evaluate_new([chroms[0].clone()])[0]  # the same DSL, generations later
    assert reproduced is slots[0]
    assert pipe._apply_plateau(chroms, 5)[0] is slots[0]  # scored once, never again
    assert ev.evaluated == before + 4


def test_one_info_line_per_generation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pipe, chroms, _ = _seeded(monkeypatch, 10, {0: 0.9})
    with caplog.at_level(logging.INFO, logger=pipeline_mod.__name__):
        pipe._apply_plateau(chroms, 0)
        pipe._apply_plateau(chroms, 1)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("plateau gen=")]
    assert lines == [
        "plateau gen=0 scored=1 neighbour_evals=4",
        "plateau gen=1 scored=0 neighbour_evals=0",
    ]


# --- R12 run-level / bankrupt --------------------------------------------------------------------------


class _NoSeriesFn:
    """A plain backtest with trades but no MTM series (v2 cannot score it)."""

    def __init__(self, total_return: float) -> None:
        self.total_return = total_return

    def __call__(self, chrom: StrategyChromosome) -> dict[str, float | int]:
        return {"sharpe_ratio": 1.0, "max_drawdown": 0.1, "profit_factor": 1.5,
                "total_trades": 80, "total_return": self.total_return}  # fmt: skip


def test_a_generation_where_every_v2_evaluation_errors_fails_the_run() -> None:
    chroms = [_chrom(i) for i in range(10)]
    pipe = DiscoveryPipeline(_config(10), _NoSeriesFn(0.05), genome_ops=_ops(chroms))
    with pytest.raises(DiscoveryEvaluationError, match="FitnessV2Error: no MTM series"):
        pipe.run()


def test_a_generation_of_bankrupt_genomes_is_a_quiet_zero_not_a_failed_run() -> None:
    chroms = [_chrom(i) for i in range(10)]
    pipe = DiscoveryPipeline(_config(10), _NoSeriesFn(-1.0), genome_ops=_ops(chroms))
    res = pipe.run()
    assert res.top_strategies == []
    assert res.generations[0].best_fitness == 0.0
    assert all(fr.error is None for fr in pipe.all_evaluated.values())


# --- R23 / R29 / R27 / R35: end to end with a deterministic MTM backtest ---------------------------------


class _SeriesFn:
    """Deterministic backtest: the series (and so the own score) is a function of the genome's DSL."""

    def __call__(self, chrom: StrategyChromosome) -> dict[str, Any]:
        seed = int.from_bytes(hashlib.sha256(key_of(chrom).encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0004 + 0.0003 * rng.random(), 0.01, 300)
        return {
            "sharpe_ratio": 1.0 + float(rng.random()), "max_drawdown": 0.1, "profit_factor": 1.5,
            "total_trades": 60 + int(rng.integers(0, 40)), "total_return": float(np.prod(1 + x) - 1),
            "mtm_daily_returns": DailyReturns(19000, array("d", x.tobytes())),
        }  # fmt: skip


def _notes_text(res: DiscoveryResult, seed: int) -> str:
    notes = record.provenance_notes(
        res, use_mock=True, eval_windows_count=1, eval_windows=None, symbol_agg="portfolio",
        split_ratio=0.0, direction=None, cross_window_months=[], cross_window_min_sharpe=0.5,
        wfa_oos_step_days=0, wfa_min_consistency=0.75, num_seeds=1, seed=seed,
        bootstrap_min_sharpe=0.0, holdout_min_sharpe=0.0, symbols=["BTCUSDT"],
    )  # fmt: skip
    return json.dumps(notes, sort_keys=True)


def _ga_run(seed: int = 7, **cfg: Any) -> tuple[DiscoveryResult, DiscoveryPipeline]:
    random.seed(seed)
    pipe = DiscoveryPipeline(
        _config(10, max_generations=3, elite_count=2, use_crowding=True, require_dsr=True, **cfg),
        _SeriesFn(),
    )
    return pipe.run(), pipe


def test_v2_run_twice_with_one_seed_is_identical() -> None:
    first, _ = _ga_run()
    second, _ = _ga_run()
    assert first.plateau is not None and cast("int", first.plateau["scored"]) >= 1
    assert _notes_text(first, 7) == _notes_text(second, 7)
    assert [key_of(c) for c, _ in first.top_strategies] == [
        key_of(c) for c, _ in second.top_strategies
    ]
    assert [g.best_fitness for g in first.generations] == [
        g.best_fitness for g in second.generations
    ]


def test_raw_dsr_trials_equal_evaluated_and_count_the_neighbours() -> None:
    res, pipe = _ga_run()
    notes = json.loads(_notes_text(res, 7))
    assert res.plateau is not None and cast("int", res.plateau["neighbour_evals"]) > 0
    assert notes["evaluated"] == notes["dsr_trials"]["raw"] == len(pipe.all_evaluated)
    assert notes["evaluated"] == len(pipe.fitness_cache) + cast(
        "int", res.plateau["neighbour_evals"]
    )
    assert notes["fitness"] == "v2" and notes["plateau"] == res.plateau
    assert notes["dsr_trials"]["non_flat"] <= len(pipe.fitness_cache)  # the pool has no neighbour


def test_multi_seed_pools_the_neighbour_trials_and_picks_on_final_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def own_of(key: str) -> float:
        return 0.3 + (int(hashlib.sha256(key.encode()).hexdigest(), 16) % 50) / 100

    ev = _Eval(own_fn=own_of)
    monkeypatch.setattr(pipeline_mod, "evaluate_population", ev)
    cfg = _config(10, max_generations=2)
    merged = pooling._run_multi_seed(2, cfg, NO_FN, "/tmp/cc-c3t3-1/progress.json", base_seed=7)
    assert merged.total_candidates_evaluated == ev.evaluated  # GA and neighbour trials, both seeds
    assert merged.plateau is not None and cast("int", merged.plateau["scored"]) >= 2
    assert cast("int", merged.plateau["neighbour_evals"]) > 0
    assert merged.top_strategies
    for _, fr in merged.top_strategies:  # champions are plateau-scored; their score is the final
        assert fr.fitness_v2 is not None and fr.fitness_v2.neighbours is not None


def test_bankroll_grid_cells_run_the_same_v2_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from vibe_quant.bankroll_grid import BankrollCell

    seen: list[str] = []

    class Seed:
        def __init__(self, config: DiscoveryConfig, **kw: Any) -> None:
            seen.append(config.fitness)
            self.fitness_cache: dict[str, FitnessResult] = {}

        def run(self) -> DiscoveryResult:
            return DiscoveryResult(
                [], [(_chrom(0), _fit())], 50, False, None,
                plateau={"scored": 2, "neighbour_evals": 8, "neighbour_errors": 1,
                         "top_fraction": 0.05, "max_neighbours": 4, "step_fraction": 0.05},
            )  # fmt: skip

    monkeypatch.setattr(pooling, "DiscoveryPipeline", Seed)
    cells = [
        BankrollCell.from_note(
            {"liquidation_mode": "isolated", "leverage": 5.0, "max_margin_pct": 0.5}
        ),
        BankrollCell.from_note(
            {"liquidation_mode": "isolated", "leverage": 3.0, "max_margin_pct": 0.5}
        ),
    ]
    merged, _ = pooling._run_bankroll_grid(
        cells, 2, _config(10), lambda cell: pooling._CellFns(NO_FN), "/tmp/cc-c3t3-1/progress.json"
    )
    assert seen == ["v2"] * 4
    assert merged.total_candidates_evaluated == 200
    assert merged.plateau == {
        "scored": 8, "neighbour_evals": 32, "neighbour_errors": 4,
        "top_fraction": 0.05, "max_neighbours": 4, "step_fraction": 0.05,
    }  # fmt: skip


# --- R28 / R23 slow: real NautilusTrader ---------------------------------------------------------------

needs_btc = pytest.mark.skipif(
    not (DEFAULT_CATALOG_PATH / "data" / "bar").exists(), reason="no data catalog"
)
START, END = "2024-01-01", "2024-04-01"


def _nt_run(seed: int) -> DiscoveryResult:
    from vibe_quant.discovery.operators import ConditionType, Direction, StrategyGene

    seeds = [
        StrategyChromosome(
            entry_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, thr, None)],
            exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 60.0, None)],
            stop_loss_pct=3.0, take_profit_pct=6.0, direction=Direction.LONG,
        )
        for thr in (35.0, 40.0, 45.0, 50.0)
    ]  # fmt: skip
    fn = NTBacktestFn(["BTCUSDT"], "4h", START, END, windows=None, min_trades=5)
    cfg = _config(
        6, max_generations=2, min_trades=5, start_date=START, end_date=END, eval_windows=1,
        gate_pool="all", direction="long", indicator_pool=["RSI"], max_workers=None, top_k=8,
        symbols=["BTCUSDT"],
    )  # fmt: skip
    random.seed(seed)
    return DiscoveryPipeline(cfg, fn, seed_chromosomes=seeds).run()


@needs_btc
@pytest.mark.slow(reason="two real NT discovery runs, 2 generations, pop 6, 3 months of 4h BTC")
def test_v2_nt_run_is_repeatable_and_its_champions_replay_exactly() -> None:
    """R23 (same seed twice -> same notes and champions) and R28 (a v2 champion of an
    eval_windows=1 run: a fresh screening run of its DSL reproduces sharpe, trades, return, PF and
    max DD bit for bit; v2 changes what is scored, never the engine's numbers)."""
    first = _nt_run(11)
    second = _nt_run(11)
    assert first.plateau is not None and cast("int", first.plateau["scored"]) >= 1
    assert first.top_strategies, "the seeded RSI genomes must produce a train-passing champion"
    assert _notes_text(first, 11) == _notes_text(second, 11)
    assert [key_of(c) for c, _ in first.top_strategies] == [
        key_of(c) for c, _ in second.top_strategies
    ]
    replay_fn = NTBacktestFn(["BTCUSDT"], "4h", START, END, windows=None, min_trades=5)
    for chrom, fr in first.top_strategies:
        again = _evaluate_single(
            chrom, replay_fn, None, 5, "4h"
        )  # v1 scoring: the engine numbers only
        assert (fr.sharpe_ratio, fr.total_trades, fr.total_return, fr.profit_factor, fr.max_drawdown) == (
            again.sharpe_ratio, again.total_trades, again.total_return, again.profit_factor, again.max_drawdown,
        )  # fmt: skip
