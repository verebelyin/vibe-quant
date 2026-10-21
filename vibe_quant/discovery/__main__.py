"""Discovery pipeline CLI entrypoint."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from vibe_quant.bankroll_grid import (
    BankrollCell,
    BankrollGridError,
    bankroll_grid_note,
)
from vibe_quant.discovery.config_check import (
    parse_cross_window_months,
)
from vibe_quant.discovery.mock_backtest import mock_backtest as _mock_backtest
from vibe_quant.discovery.neff import N_RULE_EFFECTIVE
from vibe_quant.discovery.pipeline import (
    DiscoveryConfig,
    DiscoveryPipeline,
    DiscoveryResult,
    FastTierConfirm,
)
from vibe_quant.discovery.run_spec import (
    _default_bootstrap_min_sharpe,
    _grid_cells_from_args,
    _grid_flags_given,
    build_parser,
    check_launch_leverage,
    plan_from_args,
)
from vibe_quant.discovery.seed import load_seed_chromosomes
from vibe_quant.fast_tier import fallbacks
from vibe_quant.fees import FeeSchedule
from vibe_quant.liquidation_settings import (
    DEFAULT_LEVERAGE,
    MISSING,
    LiquidationSettings,
    resolve_leverage,
)
from vibe_quant.sizing_settings import SizingSettings
from vibe_quant.utils import compute_day_count, log_dir

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery.fitness import FitnessResult
    from vibe_quant.discovery.operators import StrategyChromosome
    from vibe_quant.discovery.pipeline import (
        CrossWindowResult,
        GenerationResult,
        HoldoutResult,
        WFARollingResult,
    )


def _get_compiler_version() -> str:
    """Get compiler version hash for staleness detection."""
    try:
        from vibe_quant.dsl.compiler import compiler_version_hash

        return compiler_version_hash()
    except Exception:
        return "unknown"


# NTBacktestFn lives in vibe_quant.discovery.backtest_fn so worker
# processes can unpickle it. When this file is loaded as ``__main__``
# (via ``python -m vibe_quant.discovery``), classes defined here would
# pickle with ``__module__='__main__'`` — workers can't resolve that.
from vibe_quant.discovery.backtest_fn import (  # noqa: E402
    NTBacktestFn,
    SymbolAgg,
    full_range_headline,
)


def _make_nt_backtest_fn(
    symbols: list[str],
    timeframe: str,
    start_date: str,
    end_date: str,
    windows: list[tuple[str, str]] | None = None,
    min_trades: int = 0,
    symbol_agg: SymbolAgg = "portfolio",
    *,
    fee_schedule: FeeSchedule,
    entry_order: str,
    liquidation: LiquidationSettings,
    sizing: SizingSettings | None = None,
) -> NTBacktestFn:
    """Create a picklable backtest function using real NautilusTrader screening runner.

    ``fee_schedule``, ``entry_order`` and ``liquidation`` are required (no
    defaults) so a new call site cannot silently run at VIP0/10x while the
    rest of the run is on BNB or at the run's liquidation settings
    (bd vibe-quant-2gyk1.4, vibe-quant-2gyk1.1.9).

    ``sizing`` (bead vibe-quant-2gyk1.2) is the run's parsed
    :class:`~vibe_quant.sizing_settings.SizingSettings`. Every call site in
    ``main`` passes it; ``None`` means the defaults, which add no key and keep
    every pre-existing number byte-identical.
    """
    return NTBacktestFn(
        symbols, timeframe, start_date, end_date, windows=windows, min_trades=min_trades,
        symbol_agg=symbol_agg, fee_schedule=fee_schedule, entry_order=entry_order,
        liquidation=liquidation, sizing=sizing,
    )


def _log_data_catalog_info(symbols: list[str], timeframe: str) -> None:
    """Log data catalog details: available symbols, bar counts, date ranges."""
    try:
        from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH

        bar_dir = DEFAULT_CATALOG_PATH / "data" / "bar"
        if not bar_dir.exists():
            logger.info("Data catalog: no bar directory at %s", bar_dir)
            return

        for sym in symbols:
            instrument_id = f"{sym}-PERP.BINANCE"
            found_dirs = [
                d.name for d in bar_dir.iterdir()
                if d.is_dir() and instrument_id in d.name
            ]
            if found_dirs:
                # Count parquet files to estimate data volume
                for dname in found_dirs:
                    dpath = bar_dir / dname
                    parquet_files = list(dpath.glob("*.parquet"))
                    total_size = sum(f.stat().st_size for f in parquet_files)
                    logger.info(
                        "Data catalog: %s → %s (%d files, %.1f MB)",
                        sym,
                        dname,
                        len(parquet_files),
                        total_size / (1024 * 1024),
                    )
            else:
                logger.warning("Data catalog: %s NOT FOUND in catalog", sym)
    except Exception:
        logger.debug("Could not read data catalog info", exc_info=True)


def _check_data_available(symbols: list[str]) -> bool:
    """Check if ParquetDataCatalog has data for the given symbols."""
    try:
        from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH

        if not DEFAULT_CATALOG_PATH.exists():
            return False
        # Check if catalog directory has any bar data
        bar_dir = DEFAULT_CATALOG_PATH / "data" / "bar"
        if not bar_dir.exists():
            return False
        # Check for at least one symbol's data
        for sym in symbols:
            instrument_id = f"{sym}-PERP.BINANCE"
            # Look for any bar type directory containing this instrument
            found = False
            for d in bar_dir.iterdir():
                if d.is_dir() and instrument_id in d.name:
                    found = True
                    break
            if not found:
                return False
        return True
    except Exception:
        return False


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


def _make_cell_fns(
    *,
    use_mock: bool,
    symbols: list[str],
    timeframe: str,
    symbol_agg: SymbolAgg,
    fee_schedule: FeeSchedule,
    entry_order: str,
    liquidation: LiquidationSettings,
    sizing: SizingSettings,
    train_start: str,
    train_end: str,
    eval_windows: list[tuple[str, str]] | None,
    min_trades: int,
    holdout: tuple[str, str] | None,
    needs_factory: bool,
    evaluator: str = "nt",
) -> _CellFns:
    """The train / holdout / cross-window + WFA functions of ONE bankroll setup.

    ``main`` calls it once for a run-level setup and :func:`_run_bankroll_grid` once per cell, so
    every stage of a run is built from the same ``liquidation`` + ``sizing`` (bead
    vibe-quant-t8nrv). ``holdout`` is ``(start, end)`` or ``None`` (no split).
    """
    if use_mock:
        return _CellFns(
            backtest_fn=_mock_backtest,
            holdout_backtest_fn=_mock_backtest if holdout else None,
            backtest_fn_factory=(lambda s, e: _mock_backtest) if needs_factory else None,
        )
    backtest_fn = _make_nt_backtest_fn(
        symbols=symbols,
        timeframe=timeframe,
        start_date=train_start,
        end_date=train_end,
        windows=eval_windows,
        min_trades=min_trades,
        symbol_agg=symbol_agg,
        fee_schedule=fee_schedule,
        entry_order=entry_order,
        liquidation=liquidation,
        sizing=sizing,
    )
    if evaluator == "fast":
        # The fast tier scores the GA (NT for genomes it cannot replay); the pipeline
        # re-runs the top-K on NT before any gate (vibe-quant-z8vqv.5). Holdout,
        # cross-window and WFA below stay NT.
        from vibe_quant.fast_tier.backtest_fn import FastBacktestFn

        backtest_fn = FastBacktestFn(
            symbols, timeframe, train_start, train_end, windows=eval_windows,
            min_trades=min_trades, symbol_agg=symbol_agg, fee_schedule=fee_schedule,
            entry_order=entry_order, liquidation=liquidation, sizing=sizing, nt_fallback=True,
        )
    holdout_backtest_fn = (
        _make_nt_backtest_fn(
            symbols=symbols,
            timeframe=timeframe,
            start_date=holdout[0],
            end_date=holdout[1],
            symbol_agg=symbol_agg,
            fee_schedule=fee_schedule,
            entry_order=entry_order,
            liquidation=liquidation,
            sizing=sizing,
        )
        if holdout
        else None
    )
    backtest_fn_factory = None
    if needs_factory:
        _syms = symbols
        _tf = timeframe
        _agg: SymbolAgg = symbol_agg
        _fee = fee_schedule
        _entry = entry_order
        _liq = liquidation
        _sz = sizing

        def backtest_fn_factory(s: str, e: str) -> NTBacktestFn:
            return NTBacktestFn(
                _syms, _tf, s, e, symbol_agg=_agg, fee_schedule=_fee,
                entry_order=_entry, liquidation=_liq, sizing=_sz,
            )

    return _CellFns(backtest_fn, holdout_backtest_fn, backtest_fn_factory)


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
            pooled_fitness_cache.update(pipeline._fitness_cache)
        else:
            # Cells share seeds, so the same DSL recurs in every cell with its own series:
            # keyed by (cell, DSL) so the informational non_flat / N_eff see each of them.
            pooled_fitness_cache.update(
                {f"{run.cell.label}|{key}": fit for key, fit in pipeline._fitness_cache.items()}
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
    pooled_checker = DiscoveryPipeline(config=config, backtest_fn=backtest_fn)  # type: ignore[arg-type]
    # The seeds' retained daily series feed the POOLED effective-N record (T3).
    pooled_checker._fitness_cache.update(pooled_fitness_cache)
    candidates = [(chrom, fit) for chrom, fit, _, _ in selected_entries]
    kept = pooled_checker._validate_top_strategies(candidates, total_evaluated)
    kept_ids = {id(chrom) for chrom, _ in kept}
    for rejection in pooled_checker._guardrail_rejections:
        raw_reasons = rejection.get("reasons")
        reasons = raw_reasons if isinstance(raw_reasons, list) else []
        rejection["stage"] = "pooled_guardrails"
        rejection["reasons"] = [
            f"Multi-seed pooled N={total_evaluated}: {reason}" for reason in reasons
        ]
        all_rejections.append(rejection)
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
        dsr_trials=pooled_checker._dsr_trials,
        # The pooled checker's own DSR records: their N is the pooled gate N
        # (per-seed records would understate it), so they are the ones the
        # champion entry (and the overfitting re-check) must read.
        dsr_records=dict(pooled_checker._dsr_records),
        # Per-seed buy-and-hold records unioned across seeds (the pooled champion
        # re-check above runs guardrails only, never the buy-hold gate).
        buy_hold_records=buy_hold_records,
        buy_hold_gate=result_metadata.buy_hold_gate if result_metadata else None,
        fast_tier_confirm=sum(confirms[1:], confirms[0]) if confirms else None,
    )
    return merged, [cell_of.get(id(result)) for _, _, result, _ in selected_entries]




def walk_forward_efficiency(
    *,
    oos_return: float,
    oos_days: int | None,
    is_return: float,
    is_days: int | None,
) -> float | None:
    """Length-normalized walk-forward efficiency (Pardo).

    ``(oos_return / oos_days) / (is_return / is_days)``. A stationary edge
    scores ~1.0 regardless of how long each period is. ``None`` when the
    in-sample return isn't positive (no edge to retain) or a length is
    unknown -- a negative IS with a positive OOS is NOT "infinitely efficient".
    """
    if not oos_days or not is_days or is_days <= 0 or oos_days <= 0:
        return None
    is_per_day = is_return / is_days
    if not is_per_day > 0:
        return None
    return (oos_return / oos_days) / is_per_day


def _window_metrics(hr: HoldoutResult) -> dict[str, object]:
    return {
        "sharpe": hr.sharpe_ratio,
        "max_dd": hr.max_drawdown,
        "pf": hr.profit_factor,
        "trades": hr.total_trades,
        "return_pct": hr.total_return,
    }


def cross_window_entry(cwr: CrossWindowResult) -> dict[str, object]:
    """Persisted cross-window payload: SHIFTED windows only, with offsets + dates.

    ``windows[i]`` corresponds to ``cross_window_months[i]`` (the in-sample
    window is not listed -- it never counts as a pass).
    """
    return {
        "windows_passed": cwr.windows_passed,
        "total_windows": cwr.total_windows,
        "required": cwr.required,
        "passed": cwr.passed,
        "in_sample_excluded": True,
        "windows": [
            {
                **_window_metrics(w),
                "offset_months": cwr.offsets_months[i] if i < len(cwr.offsets_months) else None,
                "dates": list(cwr.window_dates[i]) if i < len(cwr.window_dates) else None,
            }
            for i, w in enumerate(cwr.window_results)
        ],
    }


def wfa_entry(wfa: WFARollingResult) -> dict[str, object]:
    """Persisted rolling-window (train range) stability payload."""
    return {
        "scope": "train_range_rolling",
        "windows_profitable": wfa.windows_profitable,
        "windows_sharpe_positive": wfa.windows_sharpe_positive,
        "total_windows": wfa.total_windows,
        "consistency": wfa.consistency,
        "sharpe_consistency": wfa.sharpe_consistency,
        "passed": wfa.passed,
        "windows": [
            {
                "dates": wfa.window_dates[j] if j < len(wfa.window_dates) else None,
                "sharpe": w.sharpe_ratio,
                "return_pct": w.total_return,
                "trades": w.total_trades,
            }
            for j, w in enumerate(wfa.oos_windows)
        ],
    }


def _champion_entry(
    chrom: StrategyChromosome,
    fitness: FitnessResult,
    *,
    timeframe: str,
    dsr: dict[str, object] | None,
    buy_hold: dict[str, object] | None,
) -> dict[str, object]:
    """Base champion entry (dsr/buy_hold/symbol scores + fee drag).

    ``fee_drag``/``fees_pct_equity_yr`` (bd vibe-quant-2gyk1.4) are reporting
    only -- they never influence selection. Full-range/holdout/cross-window/WFA
    payloads are added by the caller (they are index-dependent).
    """
    from vibe_quant.discovery.genome import chromosome_to_dsl, chromosome_to_serializable

    dsl = chromosome_to_dsl(chrom)
    dsl["timeframe"] = timeframe
    entry: dict[str, object] = {
        "dsl": dsl,
        "chromosome": chromosome_to_serializable(chrom),
        "score": fitness.adjusted_score,
        "sharpe": fitness.sharpe_ratio,
        "max_dd": fitness.max_drawdown,
        "pf": fitness.profit_factor,
        "trades": fitness.total_trades,
        "return_pct": fitness.total_return,
    }
    if dsr is not None:
        entry["dsr"] = dsr
    if buy_hold is not None:
        entry["buy_hold"] = buy_hold
    if fitness.symbol_scores is not None:
        entry["symbol_scores"] = {
            sym: round(sc, 6) for sym, sc in fitness.symbol_scores.items()
        }
    # Always persisted (JSON null when unavailable) so readers can tell
    # "no edge to attribute" from "key never written".
    entry["fee_drag"] = fitness.fee_drag
    entry["fees_pct_equity_yr"] = fitness.fees_pct_equity_yr
    # Report-only liquidation counts (bead vibe-quant-2gyk1.1.9, R1): written
    # only when the mode is on (the FitnessResult fields stay None off -> the
    # keys are absent, R4).
    if fitness.liquidation_count is not None:
        entry["liquidation_count"] = fitness.liquidation_count
    if fitness.liquidation_forfeited is not None:
        entry["liquidation_forfeited"] = fitness.liquidation_forfeited
    return entry


def _resolve_seed(arg: int | None) -> int:
    """Explicit --seed, else a fresh OS-entropy seed (recorded so runs can be replayed)."""
    return arg if arg is not None else random.SystemRandom().randrange(2**32)


def _attach_ind_memo(notes: dict[str, object], *, use_mock: bool) -> None:
    """Record the indicator-memo run summary in the persisted notes.

    Called right before each save so the full-range headline backtests are
    included; mock runs never touch the memo, so they carry no key.
    """
    if use_mock:
        return
    from vibe_quant.dsl import indicator_memo

    notes["ind_memo"] = indicator_memo.run_summary()


def _run_provenance_notes(
    result: DiscoveryResult,
    *,
    use_mock: bool,
    eval_windows_count: int,
    eval_windows: list[tuple[str, str]] | None,
    symbol_agg: str,
    split_ratio: float,
    direction: str | None,
    cross_window_months: list[int],
    cross_window_min_sharpe: float,
    wfa_oos_step_days: int,
    wfa_min_consistency: float,
    num_seeds: int,
    seed: int,
    bootstrap_min_sharpe: float,
    holdout_min_sharpe: float,
    symbols: list[str],
    fee_schedule: FeeSchedule | None = None,
    prior_run_ids: list[int] | None = None,
    prior_legacy: dict[str, int] | None = None,
    liquidation: LiquidationSettings | None = None,
    sizing: SizingSettings | None = None,
    indicator_pool: list[str] | None = None,
) -> dict[str, object]:
    """Run-level notes shared by the champion and zero-champion outcomes."""
    schedule = fee_schedule if fee_schedule is not None else FeeSchedule()
    liquidation_notes = liquidation.to_notes() if liquidation is not None else None
    notes: dict[str, object] = {
        "type": "discovery",
        "generations": len(result.generations),
        # Distinct strategies backtested == DSR trial count N (pooled across
        # seeds for multi-seed runs).
        "evaluated": result.total_candidates_evaluated,
        "converged": result.converged,
        "mock": use_mock,
        "synthetic": use_mock,
        "compiler_version": _get_compiler_version(),
        "eval_windows": eval_windows_count if eval_windows_count > 1 else None,
        "eval_window_aggregation": "worst_of_n" if eval_windows_count > 1 else None,
        "eval_window_ranges": eval_windows if eval_windows else None,
        # "worst": each symbol scored alone, genome keeps its worst symbol.
        "symbol_agg": symbol_agg,
        "train_test_split": split_ratio if split_ratio > 0 else None,
        "train_dates": list(result.train_dates) if result.train_dates else None,
        "holdout_dates": list(result.holdout_dates) if result.holdout_dates else None,
        "holdout_gate": (
            {
                "min_trades": result.holdout_min_trades,
                "min_sharpe": holdout_min_sharpe,
                "min_return": 0.0,
            }
            if result.holdout_dates
            else None
        ),
        "direction": direction,
        "cross_window_months": cross_window_months or None,
        "cross_window_min_sharpe": cross_window_min_sharpe if cross_window_months else None,
        "wfa_oos_step_days": wfa_oos_step_days if wfa_oos_step_days > 0 else None,
        "wfa_min_consistency": wfa_min_consistency if wfa_oos_step_days > 0 else None,
        "num_seeds": num_seeds if num_seeds > 1 else None,
        # Single-seed: the GA seed. Multi-seed: base; seed i = base + i * 7919.
        "seed": seed,
        "bootstrap_min_sharpe": bootstrap_min_sharpe,
        # Venue-level fee schedule (bd vibe-quant-2gyk1.4): the same schedule
        # every screening backtest of this run used.
        "fee_schedule": schedule.to_notes(),
    }
    # Run-level liquidation settings (bead vibe-quant-2gyk1.1.9, D2): the same
    # settings every screening backtest of this run used. The key is absent when
    # the mode is off, so existing runs stay unchanged (R4).
    if liquidation_notes is not None:
        notes["liquidation"] = liquidation_notes
    # Run-level sizing settings (bead vibe-quant-2gyk1.2): the settings every
    # screening backtest of this run used, recorded so promote/replay inherit
    # them. Only the keys moved off the default are written -- a default run
    # adds no key and its notes stay byte-identical to today (S23c).
    if sizing is not None:
        sizing_params = sizing.to_params()
        if sizing_params:
            notes["sizing"] = sizing_params
    # A non-default venue leverage is recorded so promote/replay inherit it
    # (bead vibe-quant-6wq6w). Only a mode-OFF run needs the top-level key: a
    # mode-on run already carries its leverage in notes.liquidation, and an
    # explicit default (10) writes nothing (R4/R7).
    if (
        liquidation is not None
        and not liquidation.on
        and liquidation.leverage != DEFAULT_LEVERAGE
    ):
        notes["leverage"] = liquidation.leverage
    # An explicit GA pool (scalar AND MA genes) is recorded so a harness can check what
    # the run used, not what it asked for (P-2, 2026-10-19: 96+ runs ignored the pool for
    # MA genes). A default-pool run adds no key.
    if indicator_pool is not None:
        notes["indicator_pool"] = list(indicator_pool)
    if symbol_agg == "worst":
        # GA rank score over the per-symbol adjusted fitness; champions still
        # need every symbol's train return > 0 and trades >= min (hard gate).
        notes["worst_mode_score"] = "0.5*min+0.5*median"
    if result.fast_tier_confirm is not None:
        notes["evaluator"] = {
            "name": "fast",
            "nt_confirmed": result.fast_tier_confirm.confirmed,
            "nt_fallbacks": fallbacks.summary(),
        }
    if result.dsr_trials is not None:
        notes["dsr_trials"] = _dsr_trials_note(result, prior_run_ids or [], prior_legacy)
    elif not use_mock:
        # B1 (bd vibe-quant-megq1): a --no-dsr run runs no gate, but it must still
        # record n_rule NEW at run_n_eff == raw, or A3 would treat it as a legacy
        # run and never charge its trials. Mock runs are not real trials at all.
        notes["dsr_trials"] = _dsr_trials_note(result, prior_run_ids or [], prior_legacy)
    # Buy-and-hold gate provenance (bd vibe-quant-wcn55). ``enforced`` comes from
    # the pipeline (False only for a non-NT backtest fn); the rest is the gate's
    # fixed configuration. A run with --no-buy-hold-gate has enforced False.
    notes["buy_hold_gate"] = {
        "enforced": bool((result.buy_hold_gate or {}).get("enforced", False)),
        "benchmark": "equal_weight_daily_rebalanced_1d_close",
        "candidate_series": "mark_to_market_daily",
        "symbols": list(symbols),
        "ann_factor": 365,
        "per_symbol": symbol_agg == "worst",
    }
    return notes


def _dsr_trials_note(
    result: DiscoveryResult,
    prior_run_ids: list[int],
    prior_legacy: dict[str, int] | None = None,
) -> dict[str, object]:
    """``notes.dsr_trials``: gate N and the effective-N rule that produced it.

    The gate N is ``this run's run_n_eff + prior_n`` -- the summed
    ``run_n_eff`` of earlier non-mock discovery runs with the same timeframe and
    an OVERLAPPING train window that were themselves recorded under
    ``N_RULE_EFFECTIVE`` (bd vibe-quant-megq1, decision A3). Earlier runs with no
    such rule are listed in ``prior_legacy_runs`` / ``prior_legacy_raw`` but are
    NOT charged. The pipeline records ``gate_n``/``prior_n`` when it runs the
    gate; ``prior_run_ids`` lists the contributing runs. A failed prior lookup
    fails open to the run-only N and sets ``prior_error``.

    A ``--no-dsr`` run (``require_dsr`` False) has ``result.dsr_trials`` None: it
    still records ``n_rule`` NEW with ``run_n_eff`` == the raw evaluated count and
    ``gate_n`` None (no gate ran), so the run is charged by later runs (B1).
    """
    note: dict[str, object] = dict(result.dsr_trials or {})
    if not note:
        raw = result.total_candidates_evaluated
        note["raw"] = raw
        note["run_n_eff"] = raw
        note["neff_ev_source"] = "dsr_disabled"
        note["gate_n"] = None
    legacy = prior_legacy or {}
    note["n_rule"] = N_RULE_EFFECTIVE
    note["prior_run_ids"] = list(prior_run_ids)
    if "prior_n" not in note:
        note["prior_n"] = 0
    if "gate_n" not in note:
        prior_raw = note["prior_n"]
        prior = prior_raw if isinstance(prior_raw, int) else 0
        run_side = note.get("run_n_eff")
        if not isinstance(run_side, int) or isinstance(run_side, bool):
            # Legacy/partial dsr_trials payload with no effective count: fall
            # back to the raw evaluated count (the pre-T3 run-side N).
            run_side = result.total_candidates_evaluated
        note["gate_n"] = run_side + prior
    note["prior_legacy_runs"] = int(legacy.get("runs", 0))
    note["prior_legacy_raw"] = int(legacy.get("raw", 0))
    return note


def _iso_day(value: object) -> str | None:
    """The ``YYYY-MM-DD`` of a date-ish value; None when unusable."""
    if not isinstance(value, str) or not value:
        return None
    day = value[:10]
    return day if len(day) == 10 else None


def _run_train_window(
    notes: dict[str, object],
    start_date: object,
    end_date: object,
) -> tuple[str, str] | None:
    """A prior run's train window: ``notes.train_dates`` when present, else run dates.

    Returns None (skip the row) when the window is missing or garbled.
    """
    if notes.get("train_dates"):
        train_dates = notes["train_dates"]
        if not isinstance(train_dates, (list, tuple)) or len(train_dates) != 2:
            return None
        start, end = _iso_day(train_dates[0]), _iso_day(train_dates[1])
    else:
        start, end = _iso_day(start_date), _iso_day(end_date)
    if start is None or end is None:
        return None
    return start, end


def _prior_discovery_trials(
    state: StateManager,
    run_id: int,
    timeframe: str,
    train_window: tuple[str, str],
) -> tuple[int | None, list[int], int, int]:
    """Cumulative prior trial count + contributing run ids (bd vibe-quant-megq1, A3).

    Walks earlier (``b.id < run_id``) non-mock discovery runs with the SAME
    timeframe whose train window overlaps ``train_window`` (counts only, no
    de-dup; symbols are not filtered) and charges this run for their EFFECTIVE
    trial counts:

    - A row whose ``notes.dsr_trials.n_rule`` is ``N_RULE_EFFECTIVE`` contributes
      its ``run_n_eff`` iff that is a plain int in ``1..evaluated``; a missing,
      null, bool, string, float, zero or over-evaluated value fails closed by
      contributing ``evaluated`` (and logs one warning naming the run).
    - Any other row is LEGACY (recorded before the switch): it is NOT charged,
      but is tallied in ``(legacy_runs, legacy_raw)`` so the note can name it.

    Returns ``(total, ids, legacy_runs, legacy_raw)``; ``(None, [], 0, 0)`` when
    the DB lookup fails or ``train_window`` is unusable -- the caller fails open
    to the run-only N and records ``prior_error``.
    """
    import json

    this_start, this_end = _iso_day(train_window[0]), _iso_day(train_window[1])
    if this_start is None or this_end is None:
        # Unusable current window: cannot tell which runs overlap. Fail open
        # (like a DB error) but loudly -- the caller records prior_error.
        logger.warning("Unusable train window %r; cannot count prior trials", train_window)
        return None, [], 0, 0
    try:
        rows = state.list_prior_discovery_notes(run_id, timeframe)
    except Exception:
        logger.warning("Could not count prior discovery trials", exc_info=True)
        return None, [], 0, 0
    total = 0
    ids: set[int] = set()
    legacy_runs = 0
    legacy_raw = 0
    for row_id, raw, start_date, end_date in rows:
        try:
            notes = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            logger.warning("prior trial count: run %s %s; skipped", row_id, "garbled notes")
            continue  # garbled notes -> skip the row
        if not isinstance(notes, dict) or notes.get("mock") or not notes.get("evaluated"):
            continue
        window = _run_train_window(notes, start_date, end_date)
        if window is None:
            continue  # missing/garbled train window -> skip the row
        prior_start, prior_end = window
        if not (prior_start < this_end and this_start < prior_end):
            continue  # disjoint windows -> not a prior trial
        try:
            evaluated = int(notes["evaluated"])  # type: ignore[call-overload]
        except (TypeError, ValueError):
            logger.warning(
                "prior trial count: run %s %s; skipped",
                row_id,
                "garbled evaluated count",
            )
            continue  # garbled evaluated count -> skip only this row

        dsr = notes.get("dsr_trials")
        if not isinstance(dsr, dict) or dsr.get("n_rule") != N_RULE_EFFECTIVE:
            # Legacy (pre-switch) run: tallied, never charged (decision A3).
            legacy_runs += 1
            legacy_raw += evaluated
            continue

        run_n_eff = dsr.get("run_n_eff")
        if (
            isinstance(run_n_eff, int)
            and not isinstance(run_n_eff, bool)
            and 1 <= run_n_eff <= evaluated
        ):
            total += run_n_eff
        else:
            logger.warning(
                "prior trial count: run %s has an invalid run_n_eff %r; charging evaluated %d",
                row_id,
                run_n_eff,
                evaluated,
            )
            total += evaluated
        ids.add(int(row_id))
    return total, sorted(ids), legacy_runs, legacy_raw


def _build_prior_trials_fn(
    state: StateManager,
    run_id: int,
    timeframe: str,
    *,
    mock: bool,
) -> tuple[Callable[[str, str], int | None] | None, list[int], dict[str, int]]:
    """``(DiscoveryConfig.prior_trials_fn, prior_run_ids, prior_legacy)`` for the DSR N.

    Mock runs get ``(None, [], {})`` (run-only N; their evaluations are not real
    trials). Otherwise the fn runs ``_prior_discovery_trials`` for the train
    window the pipeline hands it and refreshes the SHARED ``prior_run_ids`` list
    (so ``notes.dsr_trials`` can name the contributing runs) and the SHARED
    ``prior_legacy`` dict (``{"runs", "raw"}``, bd vibe-quant-megq1).
    """
    prior_run_ids: list[int] = []
    prior_legacy: dict[str, int] = {"runs": 0, "raw": 0}
    if mock:
        return None, prior_run_ids, {}

    def prior_trials(start: str, end: str) -> int | None:
        n, ids, legacy_runs, legacy_raw = _prior_discovery_trials(
            state, run_id, timeframe, (start, end)
        )
        prior_run_ids[:] = ids
        prior_legacy["runs"] = legacy_runs
        prior_legacy["raw"] = legacy_raw
        return n

    return prior_trials, prior_run_ids, prior_legacy


def main() -> int:
    """Run discovery pipeline and persist summary metrics for the run."""
    parser = build_parser()
    args = parser.parse_args()
    check_launch_leverage(parser, args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from vibe_quant.db.connection import DEFAULT_DB_PATH
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.jobs.manager import run_with_heartbeat

    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH
    state = StateManager(db_path)
    job_manager, stop_heartbeat = run_with_heartbeat(args.run_id, db_path)
    started_at = time.perf_counter()

    try:
        run = state.get_backtest_run(args.run_id)
        if run is None:
            error = f"Run {args.run_id} not found"
            job_manager.mark_completed(args.run_id, error=error)
            print(error)
            return 1

        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        if not symbols:
            symbols = run.get("symbols", [])

        max_workers = args.max_workers if args.max_workers >= 0 else None
        ind_pool = (
            [s.strip() for s in args.indicator_pool.split(",") if s.strip()]
            if args.indicator_pool
            else None
        )

        # Venue-level fee schedule (bd vibe-quant-2gyk1.4) shared by every
        # screening backtest of this run (train, holdout, cross-window, WFA).
        # The entry-order mode comes from the run parameters: post-only is
        # validation-only and NTBacktestFn refuses it before any engine runs.
        fee_schedule = FeeSchedule(tier=args.fee_tier, bnb_discount=args.bnb_discount)
        raw_run_params = run.get("parameters") or {}
        if isinstance(raw_run_params, str):
            import json as _json

            try:
                raw_run_params = _json.loads(raw_run_params)
            except (ValueError, TypeError):
                raw_run_params = {}
        if not isinstance(raw_run_params, dict):
            raw_run_params = {}
        entry_order = str(raw_run_params.get("entry_order", "market"))
        logger.info("Discovery fee schedule: %s", fee_schedule.to_notes())

        # Run-level liquidation settings (bead vibe-quant-6wq6w): one leverage
        # honoured in every mode. --leverage is the setting and
        # --liquidation-leverage a deprecated alias; the shared rule lets the
        # first given win and fails the run when the two disagree. Validated up
        # front so a bad leverage fails the run with the message rather than a
        # silent 10x.
        grid_run = _grid_flags_given(args)  # BankrollGridError -> exit 2, before anything runs
        resolved_leverage = resolve_leverage(
            MISSING if args.leverage is None else args.leverage,
            MISSING if args.liquidation_leverage is None else args.liquidation_leverage,
        )
        if resolved_leverage is MISSING:
            resolved_leverage = DEFAULT_LEVERAGE
        liquidation = LiquidationSettings(
            mode=args.liquidation_mode,
            leverage=cast("float", resolved_leverage),
        )
        logger.info("Discovery liquidation settings: %s", liquidation.to_notes())

        # Run-level sizing settings (bead vibe-quant-2gyk1.2): parsed and checked
        # against the resolved leverage BEFORE plan_from_args, so a corrupt or
        # unmarginable cap fails the run with its message rather than a silent
        # default. An omitted flag means the default; the keys are stored only
        # when non-default (SIP -> notes.sizing).
        sizing = SizingSettings.from_params(
            {
                key: value
                for key, value in (
                    ("max_position_pct", args.max_position_pct),
                    ("risk_per_trade", args.risk_per_trade),
                    ("max_margin_pct", args.max_margin_pct),
                )
                if value is not None
            }
        )
        # A grid has no run-level leverage: every cell checks the cap at its own leverage.
        cells = _grid_cells_from_args(args, sizing) if grid_run else None
        if cells is None:
            sizing.check_leverage(cast("float", resolved_leverage))
        else:
            logger.info("Discovery bankroll grid: %d cells", len(cells))
        logger.info("Discovery sizing settings: %s", sizing)

        # Train/test split + eval/cross/WFA windows: one shared check (the API
        # runs the same plan_from_args before creating a run row).
        plan = plan_from_args(args)
        train_start, train_end = plan.train_start, plan.train_end
        holdout_start, holdout_end = plan.holdout_start, plan.holdout_end
        split_ratio = args.train_test_split
        if split_ratio > 0:
            logger.info(
                "Train/test split: ratio=%.2f train=%s→%s holdout=%s→%s",
                split_ratio, train_start, train_end, holdout_start, holdout_end,
            )

        # Resolve timeframe-aware bootstrap_min_sharpe default.
        if args.bootstrap_min_sharpe is None:
            args.bootstrap_min_sharpe = _default_bootstrap_min_sharpe(args.timeframe)
            logger.info(
                "Bootstrap min t-stat default: %.1f (timeframe=%s)",
                args.bootstrap_min_sharpe, args.timeframe,
            )

        cross_window_months = parse_cross_window_months(args.cross_window_months or "")

        # Multi-window evaluation: sub-windows come from the plan
        eval_windows_count = max(1, args.eval_windows)
        eval_windows = plan.eval_windows
        if eval_windows is not None:
            logger.info(
                "Multi-window fitness: %d windows — %s",
                eval_windows_count,
                " | ".join(f"{s}→{e}" for s, e in eval_windows),
            )

        # Cumulative DSR gate N (bd vibe-quant-megq1): the gate adds the run_n_eff
        # of earlier non-mock discovery runs recorded under the effective rule,
        # with the same timeframe and an overlapping train window; earlier
        # (legacy) runs are tallied but not charged; mock runs keep run-only N.
        prior_trials_fn, prior_run_ids, prior_legacy = _build_prior_trials_fn(
            state, args.run_id, args.timeframe, mock=bool(args.mock)
        )

        config = DiscoveryConfig(
            prior_trials_fn=prior_trials_fn,
            population_size=args.population_size,
            max_generations=args.max_generations,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            elite_count=args.elite_count,
            tournament_size=args.tournament_size,
            convergence_generations=args.convergence_generations,
            max_workers=max_workers,
            symbols=symbols,
            timeframe=args.timeframe,
            start_date=train_start,
            end_date=train_end,
            indicator_pool=ind_pool,
            direction=args.direction,
            eval_windows=eval_windows_count,
            symbol_agg=args.symbol_agg,
            train_test_split=split_ratio,
            holdout_start_date=holdout_start or "",
            holdout_end_date=holdout_end or "",
            cross_window_months=cross_window_months,
            cross_window_min_sharpe=args.cross_window_min_sharpe,
            wfa_oos_step_days=args.wfa_oos_step_days,
            wfa_min_consistency=args.wfa_min_consistency,
            holdout_min_sharpe=args.holdout_min_sharpe,
            holdout_min_trades=args.holdout_min_trades,
            require_bootstrap_ci=args.require_bootstrap_ci,
            bootstrap_min_sharpe=args.bootstrap_min_sharpe,
            bootstrap_ci_level=args.bootstrap_ci_level,
            require_dsr=args.require_dsr,
            dsr_rule=args.dsr_rule,
            require_buy_hold=args.require_buy_hold,
            use_crowding=args.use_crowding,
            immigrant_fraction=args.immigrant_fraction,
            entropy_threshold=args.entropy_threshold,
        )

        # Log environment details for debugging and journal entries
        logger.info(
            "Environment: run_id=%d pid=%d compiler_version=%s",
            args.run_id,
            __import__("os").getpid(),
            _get_compiler_version(),
        )
        logger.info(
            "Data range: %s to %s | symbols=%s | timeframe=%s",
            args.start_date, args.end_date, symbols, args.timeframe,
        )
        _log_data_catalog_info(symbols, args.timeframe)

        # Choose backtest function: real NT, or synthetic ONLY when explicitly
        # requested. Missing catalog data used to silently fall back to mock
        # metrics that then got persisted like real champions.
        use_mock = bool(args.mock)
        if use_mock:
            logger.warning("Using MOCK backtest — forced via --mock. Results are synthetic.")
        else:
            if not _check_data_available(symbols):
                msg = (
                    f"No catalog bar data for symbols={symbols} — download data first "
                    "(Data Management) or pass --mock for a synthetic run"
                )
                raise RuntimeError(msg)
            logger.info("Using real NautilusTrader backtest for symbols=%s", symbols)
            if args.evaluator == "fast" and (
                args.timeframe != "1m" or (len(symbols) > 1 and args.symbol_agg == "portfolio")
            ):
                logger.warning(
                    "--evaluator fast replays 1m single-symbol accounts only: with "
                    "timeframe=%s symbols=%d symbol_agg=%s every genome runs on NT "
                    "(use --symbol-agg worst for multi-symbol)",
                    args.timeframe, len(symbols), args.symbol_agg,
                )
        if split_ratio <= 0:
            logger.warning(
                "No holdout (--train-test-split 0): champions get NO out-of-sample gate"
            )

        # Train, holdout (when the train/test split is on) and the factory for
        # cross-window and/or WFA validation: one set per bankroll setup.
        holdout_range = (
            (holdout_start, holdout_end)
            if split_ratio > 0 and holdout_start and holdout_end
            else None
        )
        needs_factory = bool(cross_window_months) or args.wfa_oos_step_days > 0

        def fns_for(liq: LiquidationSettings, siz: SizingSettings) -> _CellFns:
            return _make_cell_fns(
                use_mock=use_mock,
                symbols=symbols,
                timeframe=args.timeframe,
                symbol_agg=args.symbol_agg,
                fee_schedule=fee_schedule,
                entry_order=entry_order,
                liquidation=liq,
                sizing=siz,
                train_start=train_start,
                train_end=train_end,
                eval_windows=eval_windows,
                min_trades=config.min_trades,
                holdout=holdout_range,
                needs_factory=needs_factory,
                evaluator=args.evaluator,
            )

        # Load seed chromosomes from prior run if requested
        seed_chromosomes = None
        if args.seed_from_run is not None:
            seed_chromosomes = load_seed_chromosomes(state, args.seed_from_run)
            if seed_chromosomes:
                logger.info(
                    "Loaded %d seed chromosomes from run %d",
                    len(seed_chromosomes), args.seed_from_run,
                )
            else:
                # An explicitly requested warm-start that can't be honored is
                # an operator error — fail loudly instead of silently running
                # a random population under a warm-start label.
                raise ValueError(
                    f"warm-start run {args.seed_from_run} has no persisted "
                    "champions to seed from (guardrails may have rejected "
                    "all candidates); launch without seed_run_id instead"
                )

        num_seeds = max(1, args.num_seeds)
        top_cells: list[BankrollCell] | None = None  # grid: the cell of each top strategy
        progress_file = str(log_dir() / f"discovery_{args.run_id}_progress.json")
        # A leftover file (another DB's run with the same id) would be served by
        # the progress API as ours until our first generation overwrites it.
        Path(progress_file).unlink(missing_ok=True)

        if cells is not None:
            # Bankroll grid: one pipeline per (cell, seed), pooled like a multi-seed run;
            # each cell gets its own functions so every gate runs at the champion's bankroll.
            seed = _resolve_seed(args.seed)
            logger.info("Discovery seed: %d", seed)
            result, top_cells = _run_bankroll_grid(
                cells,
                num_seeds,
                config,
                lambda cell: fns_for(cell.liquidation, cell.sizing(sizing)),
                progress_file,
                seed_chromosomes=seed_chromosomes,
                base_seed=seed,
            )
        elif num_seeds == 1:
            # Single-seed run (default)
            fns = fns_for(liquidation, sizing)
            seed = _resolve_seed(args.seed)
            logger.info("Discovery seed: %d", seed)
            random.seed(seed)
            pipeline = DiscoveryPipeline(
                config=config,
                backtest_fn=fns.backtest_fn,  # type: ignore[arg-type]
                progress_file=progress_file,
                holdout_backtest_fn=fns.holdout_backtest_fn,  # type: ignore[arg-type]
                backtest_fn_factory=fns.backtest_fn_factory,  # type: ignore[arg-type]
                seed_chromosomes=seed_chromosomes,
            )
            result = pipeline.run()
        else:
            # Multi-seed ensemble: run N times with different seeds
            fns = fns_for(liquidation, sizing)
            seed = _resolve_seed(args.seed)
            logger.info("Discovery seed: %d", seed)
            result = _run_multi_seed(
                num_seeds=num_seeds,
                config=config,
                backtest_fn=fns.backtest_fn,
                progress_file=progress_file,
                holdout_backtest_fn=fns.holdout_backtest_fn,
                backtest_fn_factory=fns.backtest_fn_factory,
                seed_chromosomes=seed_chromosomes,
                base_seed=seed,
            )

        import json

        run_notes = _run_provenance_notes(
            result,
            use_mock=use_mock,
            eval_windows_count=eval_windows_count,
            eval_windows=eval_windows,
            symbol_agg=args.symbol_agg,
            split_ratio=split_ratio,
            direction=args.direction,
            cross_window_months=cross_window_months,
            cross_window_min_sharpe=args.cross_window_min_sharpe,
            wfa_oos_step_days=args.wfa_oos_step_days,
            wfa_min_consistency=args.wfa_min_consistency,
            num_seeds=num_seeds,
            seed=seed,
            bootstrap_min_sharpe=args.bootstrap_min_sharpe,
            holdout_min_sharpe=args.holdout_min_sharpe,
            symbols=symbols,
            fee_schedule=fee_schedule,
            prior_run_ids=prior_run_ids,
            prior_legacy=prior_legacy,
            liquidation=liquidation,
            sizing=sizing,
            indicator_pool=ind_pool,
        )
        if cells is not None:
            # A grid run has no run-level liquidation / leverage / max_margin_pct: the cells
            # are the setups, and each champion records its own (R51).
            run_notes["bankroll_grid"] = bankroll_grid_note(cells)

        if not result.top_strategies:
            # No champion passed every gate (guardrails / cross-window / WFA /
            # holdout -- all fail closed). Persist a structured summary with
            # the per-candidate rejection reasons and complete cleanly: with
            # honest gates this is a routine outcome, not a crash.
            execution_time = time.perf_counter() - started_at
            best_gen_fitness = (
                max((gr.best_fitness for gr in result.generations), default=0.0)
                if result.generations
                else 0.0
            )
            stages = sorted({
                str(r.get("stage", "guardrails")) for r in result.guardrail_rejections
            })
            summary_notes = {
                **run_notes,
                "outcome": "no_viable_strategies",
                "best_raw_score": best_gen_fitness,
                "reason": (
                    "No candidate passed every gate"
                    + (f" (rejected at: {', '.join(stages)})" if stages else
                       " (no candidate scored above zero)")
                    + ". See guardrail_rejections for per-candidate reasons. Consider: "
                    "longer date range, larger population, or a different indicator pool."
                ),
                "top_strategies": [],
                "guardrail_rejections": result.guardrail_rejections,
            }
            _attach_ind_memo(summary_notes, use_mock=use_mock)
            state.save_backtest_result(
                args.run_id,
                {
                    "total_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "profit_factor": 0.0,
                    "total_trades": 0,
                    "skewness": 0.0,
                    "kurtosis": 3.0,  # normal-distribution default (min valid is 1)
                    "execution_time_seconds": execution_time,
                    "notes": json.dumps(summary_notes),
                },
            )
            state.update_backtest_run_status(args.run_id, "completed")
            job_manager.mark_completed(args.run_id)
            print(
                "Discovery complete: no viable strategies "
                f"(evaluated={result.total_candidates_evaluated}, "
                f"best_raw_score={best_gen_fitness:.4f})"
            )
            return 0

        best_chrom, best_fitness = result.top_strategies[0]
        execution_time = time.perf_counter() - started_at

        # Save top strategies as DSL dicts in notes
        # bd vibe-quant-rewru: the GA's stored sharpe/trades are the multi-window
        # fitness aggregate (worst-of-windows / sum-of-windows) and, with a train/test
        # split, cover only the training slice. Promotion replays ONE continuous
        # screening backtest over the full discovery range, so those numbers are a
        # different statistic and replay_drift flags the gap by construction. Re-run
        # each champion ONCE over args.start_date..args.end_date (the exact range the
        # promote endpoint replays) and persist THAT as the headline so the drift
        # check compares like-for-like. Skip the extra backtest only when fitness
        # already IS that continuous full-range metric (single window, no split) or
        # in mock mode. Bounded cost: top-K champions, once each, at save time.
        worst_multi = args.symbol_agg == "worst" and len(symbols) > 1
        # worst mode always needs the run: it carries trades_sum/symbol_metrics
        needs_full_range = not use_mock and (
            eval_windows_count >= 2 or split_ratio > 0 or worst_multi
        )
        # One full-range function per distinct bankroll setup: a grid champion is measured at ITS
        # cell, never at the outer run settings (a replay child inherits that cell, so the
        # headline must be the number the replay reproduces). Without a grid it is the single
        # run-level function, built lazily so a zero-champion run builds none.
        full_range_fns: dict[BankrollCell | None, NTBacktestFn] = {}

        def full_range_fn_for(cell: BankrollCell | None) -> NTBacktestFn:
            if cell not in full_range_fns:
                full_range_fns[cell] = _make_nt_backtest_fn(
                    symbols=symbols,
                    timeframe=args.timeframe,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    symbol_agg=args.symbol_agg,
                    fee_schedule=fee_schedule,
                    entry_order=entry_order,
                    liquidation=cell.liquidation if cell is not None else liquidation,
                    sizing=cell.sizing(sizing) if cell is not None else sizing,
                )
            return full_range_fns[cell]

        if needs_full_range:
            logger.info(
                "Computing full-range headline for top-%d champions over %s..%s "
                "(multi-window/split fitness != continuous replay; bd-rewru)",
                len(result.top_strategies[:5]), args.start_date, args.end_date,
            )

        top_dsls = []
        for idx, (chrom, fitness) in enumerate(result.top_strategies[:5]):
            entry = _champion_entry(
                chrom,
                fitness,
                timeframe=args.timeframe,
                dsr=result.dsr_records.get(chrom.uid),
                buy_hold=result.buy_hold_records.get(chrom.uid),
            )
            cell = top_cells[idx] if top_cells is not None else None
            if cell is not None:
                entry["bankroll"] = cell.to_note()
            # Full-range headline (single continuous backtest) for like-for-like
            # promotion/replay_drift; sharpe/trades above stay as the multi-window
            # robustness aggregate. The function raises
            # DataUnavailableError on missing data (the run fails); other
            # backtest errors are swallowed into failure metrics. bd
            # vibe-quant-rewru.
            entry.update(
                full_range_headline(
                    full_range_fn_for(cell)(chrom) if needs_full_range else None,
                    fallback_sharpe=fitness.sharpe_ratio,
                    fallback_trades=fitness.total_trades,
                    fallback_max_dd=fitness.max_drawdown,
                    fallback_pf=fitness.profit_factor,
                    fallback_return=fitness.total_return,
                )
            )
            # Attach holdout metrics if available
            if idx < len(result.holdout_results):
                hr = result.holdout_results[idx]
                entry["holdout"] = {
                    "sharpe": hr.sharpe_ratio,
                    "max_dd": hr.max_drawdown,
                    "pf": hr.profit_factor,
                    "trades": hr.total_trades,
                    "return_pct": hr.total_return,
                }
                if hr.trades_sum is not None:
                    entry["holdout"]["trades_sum"] = hr.trades_sum  # type: ignore[index]
                    entry["holdout"]["symbol_metrics"] = hr.symbol_metrics  # type: ignore[index]
                # Report-only liquidation count (R5): written only when on.
                if hr.liquidation_count is not None:
                    entry["holdout"]["liquidation_count"] = hr.liquidation_count  # type: ignore[index]
            # Attach cross-window results if available
            if idx < len(result.cross_window_results):
                entry["cross_window"] = cross_window_entry(result.cross_window_results[idx])
            # Attach WFA rolling results if available
            if idx < len(result.wfa_results):
                entry["wfa"] = wfa_entry(result.wfa_results[idx])
            top_dsls.append(entry)

        # Walk-forward efficiency (Pardo): out-of-sample return per day over
        # in-sample return per day, for the best champion. OOS = the holdout,
        # IS = one continuous backtest over the train range (the worst-of-N
        # fitness return is not a train-range return). Undefined without a
        # holdout or with a non-positive IS return.
        wfa_efficiency: float | None = None
        if result.holdout_results and result.train_dates and result.holdout_dates:
            if eval_windows_count >= 2 and not use_mock:
                best_cell = top_cells[0] if top_cells is not None else None
                train_metrics = _make_nt_backtest_fn(
                    symbols=symbols,
                    timeframe=args.timeframe,
                    start_date=result.train_dates[0],
                    end_date=result.train_dates[1],
                    symbol_agg=args.symbol_agg,
                    fee_schedule=fee_schedule,
                    entry_order=entry_order,
                    liquidation=best_cell.liquidation if best_cell is not None else liquidation,
                    sizing=best_cell.sizing(sizing) if best_cell is not None else sizing,
                )(best_chrom)
                train_return = float(train_metrics.get("total_return", 0.0))
            else:
                train_return = best_fitness.total_return
            wfa_efficiency = walk_forward_efficiency(
                oos_return=result.holdout_results[0].total_return,
                oos_days=compute_day_count(*result.holdout_dates),
                is_return=train_return,
                is_days=compute_day_count(*result.train_dates),
            )

        run_notes["top_strategies"] = top_dsls
        run_notes["guardrail_rejections"] = result.guardrail_rejections or None
        _attach_ind_memo(run_notes, use_mock=use_mock)
        state.save_backtest_result(
            args.run_id,
            {
                "total_return": best_fitness.total_return,
                "sharpe_ratio": best_fitness.sharpe_ratio,
                "max_drawdown": best_fitness.max_drawdown,
                "profit_factor": best_fitness.profit_factor,
                "total_trades": best_fitness.total_trades,
                "skewness": best_fitness.skewness,
                "kurtosis": best_fitness.kurtosis,
                "execution_time_seconds": execution_time,
                "walk_forward_efficiency": wfa_efficiency,
                "notes": json.dumps(run_notes),
            },
        )
        state.update_backtest_run_status(args.run_id, "completed")
        job_manager.mark_completed(args.run_id)

        print(
            "Discovery complete: "
            f"top_score={best_fitness.adjusted_score:.4f}, "
            f"candidates={result.total_candidates_evaluated}, "
            f"mock={use_mock}"
        )
        return 0
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        try:
            state.update_backtest_run_status(args.run_id, "failed", error_message=error)
            job_manager.mark_completed(args.run_id, error=error)
        except Exception as write_exc:
            logger.error(
                "could not record run %s as failed (%s)",
                args.run_id,
                type(write_exc).__name__,
            )
        print(f"Discovery failed: {exc}")
        import traceback

        traceback.print_exc()
        # A bad bankroll grid is an operator error (argparse's own exit code), not a crash.
        return 2 if isinstance(exc, BankrollGridError) else 1
    finally:
        stop_heartbeat()
        state.close()
        job_manager.close()


if __name__ == "__main__":
    raise SystemExit(main())
