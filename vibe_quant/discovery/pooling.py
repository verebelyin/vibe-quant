"""Multi-seed and bankroll-grid pooling of discovery runs.

``_run_multi_seed``, ``_run_bankroll_grid`` and their shared ``_pool_runs`` core
moved verbatim from ``vibe_quant.discovery.__main__`` (bead vibe-quant-rabmo.7,
step T14). ``_CellFns`` lives here and ``run.py`` imports it from this module
(one direction only). Nothing here imports ``vibe_quant.discovery.run`` or
``vibe_quant.discovery.__main__``: the pipeline class is this module's own
global, so a patch of the pooled path targets
``vibe_quant.discovery.pooling.DiscoveryPipeline``.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vibe_quant.discovery.gate_chain import (
    GateChain,
    GateEvidence,
    TrialBudget,
    dsr_requires_returns,
)
from vibe_quant.discovery.pipeline import (
    DiscoveryConfig,
    DiscoveryPipeline,
    DiscoveryResult,
    FastTierConfirm,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from vibe_quant.bankroll_grid import BankrollCell
    from vibe_quant.discovery.fitness import FitnessResult
    from vibe_quant.discovery.operators import StrategyChromosome
    from vibe_quant.discovery.pipeline import GenerationResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CellFns:
    """The backtest functions of one discovery run: train, holdout and the cross-window / WFA factory."""

    backtest_fn: object
    holdout_backtest_fn: object = None
    backtest_fn_factory: object = None


@dataclass(frozen=True)
class _SeedRun:
    """One pipeline run of a pooled search: its seed, its functions and (grid) its cell."""

    seed: int
    fns: _CellFns
    cell: BankrollCell | None = None


def _run_multi_seed(
    num_seeds: int,
    config: DiscoveryConfig,
    backtest_fn: object,
    progress_file: str,
    holdout_backtest_fn: object = None,
    backtest_fn_factory: object = None,
    seed_chromosomes: list[StrategyChromosome] | None = None,
    base_seed: int = 42,
) -> DiscoveryResult:
    """Run the discovery pipeline multiple times with different random seeds.

    Aggregates results across all seeds:
    - Collects all top strategies from all runs
    - Applies diversity dedup across the merged pool
    - Reports per-seed distribution stats

    Args:
        num_seeds: Number of random seeds to run.
        config: Discovery configuration (shared across seeds).
        backtest_fn: Backtest callable.
        progress_file: Progress file path template.
        holdout_backtest_fn: Optional holdout backtest callable.
        backtest_fn_factory: Optional factory for cross-window.
        seed_chromosomes: Optional seed chromosomes for warm-start.

    Returns:
        Merged DiscoveryResult with aggregated stats.
    """
    fns = _CellFns(backtest_fn, holdout_backtest_fn, backtest_fn_factory)
    runs = [_SeedRun(base_seed + seed_idx * 7919, fns) for seed_idx in range(num_seeds)]
    result, _cells = _pool_runs(runs, config, progress_file, seed_chromosomes)
    return result


def _run_bankroll_grid(
    cells: list[BankrollCell],
    num_seeds: int,
    config: DiscoveryConfig,
    fns_for_cell: Callable[[BankrollCell], _CellFns],
    progress_file: str,
    seed_chromosomes: list[StrategyChromosome] | None = None,
    base_seed: int = 42,
) -> tuple[DiscoveryResult, list[BankrollCell]]:
    """Run one discovery per (cell, seed) and pool them like a multi-seed run.

    Every run gets the train / holdout / cross-window / WFA functions of ITS cell, so a champion
    is gated at the bankroll it was found under. The pick is made across all cells x seeds, so
    the DSR trial count N is the sum of every run's evaluations (bead vibe-quant-t8nrv, R50/R58).
    Seeds are shared across cells: cell ``c`` run ``i`` uses ``base_seed + i * 7919`` like any
    multi-seed run, which keeps the cells comparable and the whole grid replayable from one seed.

    Returns:
        ``(merged result, the cell of each merged top strategy)``, index-aligned with
        ``result.top_strategies``.
    """
    runs: list[_SeedRun] = []
    for cell in cells:
        fns = fns_for_cell(cell)
        runs.extend(
            _SeedRun(base_seed + seed_idx * 7919, fns, cell) for seed_idx in range(num_seeds)
        )
    result, picked = _pool_runs(runs, config, progress_file, seed_chromosomes)
    return result, [cell for cell in picked if cell is not None]


def _pool_runs(
    runs: list[_SeedRun],
    config: DiscoveryConfig,
    progress_file: str,
    seed_chromosomes: list[StrategyChromosome] | None,
) -> tuple[DiscoveryResult, list[BankrollCell | None]]:
    """Run every pipeline of ``runs``, group the champions and re-check them with the POOLED N.

    The shared core of :func:`_run_multi_seed` and :func:`_run_bankroll_grid`. Returns the merged
    result plus the cell (``None`` outside a grid) of each merged top strategy.
    """
    num_seeds = len(runs)
    backtest_fn = runs[0].fns.backtest_fn
    import math
    import statistics

    all_strategies: list[
        tuple[
            StrategyChromosome,
            FitnessResult,
            DiscoveryResult,
            int,
        ]
    ] = []
    all_generations: list[GenerationResult] = []
    all_rejections: list[dict[str, object]] = []
    buy_hold_records: dict[str, dict[str, object]] = {}
    total_evaluated = 0
    pooled_fitness_cache: dict[str, FitnessResult] = {}
    seed_stats: list[dict[str, float]] = []
    any_converged = False
    convergence_gen: int | None = None
    result_metadata: DiscoveryResult | None = None
    cell_of: dict[int, BankrollCell] = {}  # id(per-run result) -> the cell it ran under
    confirms: list[FastTierConfirm] = []  # --evaluator fast: every run's NT confirm

    for seed_idx, run in enumerate(runs):
        seed_val = run.seed  # Deterministic but varied seeds
        random.seed(seed_val)

        logger.info(
            "=== MULTI-SEED RUN %d/%d (seed=%d) ===",
            seed_idx + 1, num_seeds, seed_val,
        )
        if run.cell is not None:
            logger.info("  bankroll cell %s", run.cell.label)

        pipeline = DiscoveryPipeline(
            config=config,
            backtest_fn=run.fns.backtest_fn,  # type: ignore[arg-type]
            progress_file=progress_file,
            holdout_backtest_fn=run.fns.holdout_backtest_fn,  # type: ignore[arg-type]
            backtest_fn_factory=run.fns.backtest_fn_factory,  # type: ignore[arg-type]
            seed_chromosomes=seed_chromosomes,
        )
        result = pipeline.run()
        if result.fast_tier_confirm is not None:
            confirms.append(result.fast_tier_confirm)
        if run.cell is not None:
            cell_of[id(result)] = run.cell
        if run.cell is None:
            pooled_fitness_cache.update(pipeline.fitness_cache)
        else:
            # Cells share seeds, so the same DSL recurs in every cell with its own series:
            # keyed by (cell, DSL) so the informational non_flat / N_eff see each of them.
            pooled_fitness_cache.update(
                {f"{run.cell.label}|{key}": fit for key, fit in pipeline.fitness_cache.items()}
            )
        if result_metadata is None:
            result_metadata = result

        # Collect per-seed stats
        if result.top_strategies:
            sharpes = [f.sharpe_ratio for _, f in result.top_strategies]
            # ``max`` over a non-finite Sharpe is order-dependent (NaN
            # comparisons are False, so ``max([nan, 1.0])`` is nan while
            # ``max([1.0, nan])`` is 1.0). Pick the best FINITE value only; a
            # seed whose strategies are all non-finite gets a NaN best_sharpe.
            finite_seed_sharpes = [s for s in sharpes if math.isfinite(s)]
            best_sharpe = (
                max(finite_seed_sharpes) if finite_seed_sharpes else float("nan")
            )
            seed_stats.append({
                "seed": seed_val,
                "best_sharpe": best_sharpe,
                "best_score": result.top_strategies[0][1].adjusted_score,
                "num_strategies": len(result.top_strategies),
            })
        else:
            seed_stats.append({
                "seed": seed_val,
                "best_sharpe": 0.0,
                "best_score": 0.0,
                "num_strategies": 0,
            })

        all_strategies.extend(
            (chrom, fit, result, idx)
            for idx, (chrom, fit) in enumerate(result.top_strategies)
        )
        all_generations.extend(result.generations)
        all_rejections.extend(result.guardrail_rejections)
        buy_hold_records.update(result.buy_hold_records)
        total_evaluated += result.total_candidates_evaluated
        if result.converged:
            any_converged = True
            convergence_gen = result.convergence_generation

    # Log multi-seed distribution stats. A seed's best Sharpe can be non-finite
    # (a NaN/+-inf champion). ``statistics.stdev([nan, ...])`` raises
    # AttributeError ('float' object has no attribute 'numerator'), which would
    # abort the whole run BEFORE grouping/notes -- so compute mean/median/min/
    # max/stdev over FINITE values only and report the rest as a count.
    # Summary logging must never abort a run.
    sharpe_list = [s["best_sharpe"] for s in seed_stats]
    score_list = [s["best_score"] for s in seed_stats]
    finite_sharpes = [s for s in sharpe_list if math.isfinite(s)]
    non_finite_seeds = len(sharpe_list) - len(finite_sharpes)
    failure_count = sum(1 for s in seed_stats if s["num_strategies"] == 0)

    logger.info("=== MULTI-SEED SUMMARY (%d seeds) ===", num_seeds)
    if finite_sharpes:
        logger.info(
            "  Best Sharpe: mean=%.2f median=%.2f min=%.2f max=%.2f std=%.2f",
            statistics.mean(finite_sharpes),
            statistics.median(finite_sharpes),
            min(finite_sharpes),
            max(finite_sharpes),
            statistics.stdev(finite_sharpes) if len(finite_sharpes) > 1 else 0,
        )
    else:
        logger.info("  Best Sharpe: n/a (no finite seed Sharpe)")
    logger.info(
        "  Best Score: mean=%.4f median=%.4f",
        statistics.mean(score_list),
        statistics.median(score_list),
    )
    if non_finite_seeds:
        logger.info(
            "  Non-finite seed Sharpes: %d/%d", non_finite_seeds, num_seeds
        )
    logger.info(
        "  Seed failures: %d/%d (%.0f%%)",
        failure_count, num_seeds,
        failure_count / num_seeds * 100 if num_seeds else 0,
    )

    # Group strategies by structural similarity, rank by median Sharpe
    from vibe_quant.discovery.distance import chromosome_distance
    from vibe_quant.discovery.pipeline import _behaviour_fingerprint

    # Build groups: strategies within min_distance OR behaviourally identical to
    # the group representative (same non-None fitness fingerprint) are "the same"
    groups: list[
        list[
            tuple[
                StrategyChromosome,
                FitnessResult,
                DiscoveryResult,
                int,
            ]
        ]
    ] = []
    for chrom, fit, result, idx in all_strategies:
        placed = False
        candidate_fp = _behaviour_fingerprint(fit)
        for group in groups:
            rep_chrom = group[0][0]
            structurally_close = (
                chromosome_distance(chrom, rep_chrom) < config.min_diversity_distance
            )
            behaviourally_same = (
                candidate_fp is not None
                and candidate_fp == _behaviour_fingerprint(group[0][1])
            )
            if structurally_close or behaviourally_same:
                group.append((chrom, fit, result, idx))
                placed = True
                break
        if not placed:
            groups.append([(chrom, fit, result, idx)])

    # Rank groups by median Sharpe (not best single-run score)
    def _group_median_sharpe(
        group: list[
            tuple[
                StrategyChromosome,
                FitnessResult,
                DiscoveryResult,
                int,
            ]
        ]
    ) -> float:
        sharpes = [f.sharpe_ratio for _, f, _, _ in group]  # type: ignore[union-attr]
        return statistics.median(sharpes) if sharpes else 0.0

    groups.sort(key=_group_median_sharpe, reverse=True)

    # Select best representative from each top group (by adjusted_score)
    selected_entries: list[
        tuple[
            StrategyChromosome,
            FitnessResult,
            DiscoveryResult,
            int,
        ]
    ] = []
    for group in groups[:config.top_k]:
        best = max(group, key=lambda t: t[1].adjusted_score)  # type: ignore[union-attr]
        selected_entries.append(best)
        median_sr = _group_median_sharpe(group)
        logger.info(
            "  Group: %d seeds, median_sharpe=%.2f, representative=%s",
            len(group), median_sr, best[0].uid,  # type: ignore[union-attr]
        )

    # The pick is made across ALL seeds, so the multiple-testing burden is the
    # POOLED trial count -- per-seed DSR (N = one seed's evaluations)
    # undercounts it. Re-check the selected champions with pooled N; failures
    # are dropped (fail closed) and recorded.
    gates = GateChain(
        config,
        GateEvidence(backtest_fn, dsr_requires_returns(config, backtest_fn)),
    )
    candidates = [(chrom, fit) for chrom, fit, _, _ in selected_entries]
    # The seeds' retained daily series feed the POOLED effective-N record (T3).
    kept = gates.pooled_recheck(candidates, TrialBudget(total_evaluated, pooled_fitness_cache))
    kept_ids = {id(chrom) for chrom, _ in kept}
    all_rejections.extend(gates.rejections)
    selected_entries = [e for e in selected_entries if id(e[0]) in kept_ids]
    # Cells share seeds, so a uid recurs across cells (and seeds): the union above keeps the last
    # run's record. Each selected champion's buy-and-hold evidence is the one ITS run produced.
    for chrom, _fit, result, _idx in selected_entries:
        own = result.buy_hold_records.get(chrom.uid)  # type: ignore[union-attr]
        if own is not None:
            buy_hold_records[chrom.uid] = own  # type: ignore[union-attr]

    top_strategies = [(chrom, fit) for chrom, fit, _, _ in selected_entries]
    holdout_results = [
        result.holdout_results[idx]
        for _, _, result, idx in selected_entries
        if idx < len(result.holdout_results)
    ]
    cross_window_results = [
        result.cross_window_results[idx]
        for _, _, result, idx in selected_entries
        if idx < len(result.cross_window_results)
    ]
    wfa_results = [
        result.wfa_results[idx]
        for _, _, result, idx in selected_entries
        if idx < len(result.wfa_results)
    ]

    logger.info(
        "  Merged: %d groups from %d total candidates (%d groups)",
        len(top_strategies), len(all_strategies), len(groups),
    )

    merged = DiscoveryResult(
        generations=all_generations,
        top_strategies=top_strategies,  # type: ignore[arg-type]
        total_candidates_evaluated=total_evaluated,
        converged=any_converged,
        convergence_generation=convergence_gen,
        holdout_results=holdout_results,
        train_dates=result_metadata.train_dates if result_metadata else None,
        holdout_dates=result_metadata.holdout_dates if result_metadata else None,
        cross_window_results=cross_window_results,
        wfa_results=wfa_results,
        guardrail_rejections=all_rejections,
        holdout_min_trades=result_metadata.holdout_min_trades if result_metadata else None,
        # Pooled trial record: raw = pooled N, gate_n/prior_n = what the pooled
        # gate used (the per-seed results' own dsr_trials are not the pick's N).
        dsr_trials=gates.dsr_trials,
        # The pooled checker's own DSR records: their N is the pooled gate N
        # (per-seed records would understate it), so they are the ones the
        # champion entry (and the overfitting re-check) must read.
        dsr_records=dict(gates.dsr_records),
        # Per-seed buy-and-hold records unioned across seeds (the pooled champion
        # re-check above runs guardrails only, never the buy-hold gate).
        buy_hold_records=buy_hold_records,
        buy_hold_gate=result_metadata.buy_hold_gate if result_metadata else None,
        fast_tier_confirm=sum(confirms[1:], confirms[0]) if confirms else None,
    )
    return merged, [cell_of.get(id(result)) for _, _, result, _ in selected_entries]
