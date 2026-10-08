"""Discovery pipeline CLI entrypoint."""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

from vibe_quant.discovery.mock_backtest import mock_backtest as _mock_backtest
from vibe_quant.discovery.pipeline import (
    DiscoveryConfig,
    DiscoveryPipeline,
    DiscoveryResult,
)
from vibe_quant.utils import compute_day_count

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery.fitness import FitnessResult
    from vibe_quant.discovery.operators import StrategyChromosome
    from vibe_quant.discovery.pipeline import (
        CrossWindowResult,
        GenerationResult,
        HoldoutResult,
        WFARollingResult,
    )


# Timeframe-aware bootstrap-CI floor defaults (vibe-quant-gds1c):
# - 1m: high-frequency noise widens Sharpe CIs; 1.0 rejects nearly everything.
# - 4h/1d: ~50-180 trades/yr makes the CI lower bound structurally < 1.0 for
#   ANY strategy (Batch 41: lower bounds -0.77..-1.39 with DSR p=0.0000), so
#   the gate degrades to "must exclude negative Sharpe" instead of blocking
#   the entire timeframe. Explicit --bootstrap-min-sharpe always wins.
_BOOTSTRAP_FLOOR_BY_TIMEFRAME: dict[str, float] = {"1m": 0.5, "4h": 0.0, "1d": 0.0}
_BOOTSTRAP_FLOOR_DEFAULT = 1.0


def _default_bootstrap_min_sharpe(timeframe: str) -> float:
    return _BOOTSTRAP_FLOOR_BY_TIMEFRAME.get(timeframe, _BOOTSTRAP_FLOOR_DEFAULT)


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
) -> NTBacktestFn:
    """Create a picklable backtest function using real NautilusTrader screening runner."""
    return NTBacktestFn(
        symbols, timeframe, start_date, end_date, windows=windows, min_trades=min_trades,
        symbol_agg=symbol_agg,
    )


# Default TRAIN fraction: every discovery gets a 20% holdout that is used once,
# as the final pass/fail gate (vibe-quant-e70tl.5). Pass --train-test-split 0
# to opt out explicitly.
DEFAULT_TRAIN_TEST_SPLIT = 0.8


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
    total_evaluated = 0
    seed_stats: list[dict[str, float]] = []
    any_converged = False
    convergence_gen: int | None = None
    result_metadata: DiscoveryResult | None = None

    for seed_idx in range(num_seeds):
        seed_val = base_seed + seed_idx * 7919  # Deterministic but varied seeds
        random.seed(seed_val)

        logger.info(
            "=== MULTI-SEED RUN %d/%d (seed=%d) ===",
            seed_idx + 1, num_seeds, seed_val,
        )

        pipeline = DiscoveryPipeline(
            config=config,
            backtest_fn=backtest_fn,  # type: ignore[arg-type]
            progress_file=progress_file,
            holdout_backtest_fn=holdout_backtest_fn,  # type: ignore[arg-type]
            backtest_fn_factory=backtest_fn_factory,  # type: ignore[arg-type]
            seed_chromosomes=seed_chromosomes,
        )
        result = pipeline.run()
        if result_metadata is None:
            result_metadata = result

        # Collect per-seed stats
        if result.top_strategies:
            sharpes = [f.sharpe_ratio for _, f in result.top_strategies]
            best_sharpe = max(sharpes)
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
        total_evaluated += result.total_candidates_evaluated
        if result.converged:
            any_converged = True
            convergence_gen = result.convergence_generation

    # Log multi-seed distribution stats
    sharpe_list = [s["best_sharpe"] for s in seed_stats]
    score_list = [s["best_score"] for s in seed_stats]
    failure_count = sum(1 for s in seed_stats if s["num_strategies"] == 0)

    logger.info("=== MULTI-SEED SUMMARY (%d seeds) ===", num_seeds)
    if sharpe_list:
        logger.info(
            "  Best Sharpe: mean=%.2f median=%.2f min=%.2f max=%.2f std=%.2f",
            statistics.mean(sharpe_list),
            statistics.median(sharpe_list),
            min(sharpe_list),
            max(sharpe_list),
            statistics.stdev(sharpe_list) if len(sharpe_list) > 1 else 0,
        )
        logger.info(
            "  Best Score: mean=%.4f median=%.4f",
            statistics.mean(score_list),
            statistics.median(score_list),
        )
    logger.info(
        "  Seed failures: %d/%d (%.0f%%)",
        failure_count, num_seeds,
        failure_count / num_seeds * 100 if num_seeds else 0,
    )

    # Group strategies by structural similarity, rank by median Sharpe
    from vibe_quant.discovery.distance import chromosome_distance

    # Build groups: strategies within min_distance are "the same"
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
        for group in groups:
            rep_chrom = group[0][0]
            if chromosome_distance(chrom, rep_chrom) < config.min_diversity_distance:
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

    return DiscoveryResult(
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
    )


def _load_seed_chromosomes(
    state: StateManager,
    run_id: int,
) -> list[StrategyChromosome] | None:
    """Load top chromosomes from a prior discovery run for warm-starting.

    Reads the 'chromosome' field from stored top_strategies if available.
    Falls back to None if data is missing or unparseable.
    """
    import json

    from vibe_quant.discovery.genome import serializable_to_chromosome

    result = state.get_backtest_result(run_id)  # type: ignore[union-attr]
    if result is None:
        return None
    notes = result.get("notes", "")
    if not notes or not isinstance(notes, str):
        return None
    try:
        data = json.loads(notes)
        strategies = data.get("top_strategies", [])
        chromosomes: list[StrategyChromosome] = []
        for entry in strategies:
            chrom_data = entry.get("chromosome")
            if chrom_data and isinstance(chrom_data, dict):
                chromosomes.append(serializable_to_chromosome(chrom_data))
        return chromosomes if chromosomes else None
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        logger.warning("Failed to load seed chromosomes from run %d", run_id, exc_info=True)
        return None


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


def _resolve_seed(arg: int | None) -> int:
    """Explicit --seed, else a fresh OS-entropy seed (recorded so runs can be replayed)."""
    return arg if arg is not None else random.SystemRandom().randrange(2**32)


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
) -> dict[str, object]:
    """Run-level notes shared by the champion and zero-champion outcomes."""
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
    }
    if symbol_agg == "worst":
        # GA rank score over the per-symbol adjusted fitness; champions still
        # need every symbol's train return > 0 and trades >= min (hard gate).
        notes["worst_mode_score"] = "0.5*min+0.5*median"
    return notes


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for discovery jobs."""
    parser = argparse.ArgumentParser(
        prog="vibe-quant discovery",
        description="Run genetic strategy discovery",
    )
    parser.add_argument("--run-id", type=int, required=True, help="Backtest run ID")
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--max-generations", type=int, default=15)
    parser.add_argument("--mutation-rate", type=float, default=0.1)
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--convergence-generations", type=int, default=10)
    parser.add_argument(
        "--max-workers", type=int, default=0, help="Parallel workers (0=auto, -1=sequential)"
    )
    parser.add_argument("--symbols", type=str, default="BTCUSDT")
    parser.add_argument("--timeframe", type=str, default="4h")
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    parser.add_argument("--end-date", type=str, default="2026-02-24")
    parser.add_argument(
        "--indicator-pool",
        type=str,
        default=None,
        help="Comma-separated indicator names to use (default: all)",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default=None,
        help="Direction constraint: long, short, both, or omit for random",
    )
    parser.add_argument(
        "--eval-windows",
        type=int,
        default=3,
        help="Split the train range into N sub-windows; fitness is the WORST "
        "window (min Sharpe, min return, max drawdown, min PF; trades summed for "
        "the min-trades gate) and every window needs >= max(1, min_trades // (2N)) "
        "trades. Forces regime-robust strategies (default: 3; pass 1 for a "
        "single window).",
    )
    parser.add_argument(
        "--symbol-agg",
        choices=["portfolio", "worst"],
        default="portfolio",
        help="Multi-symbol scoring: 'portfolio' = one shared-account backtest "
        "over all symbols (default); 'worst' = each symbol backtested alone and "
        "the genome scored by its worst symbol (min Sharpe/return/PF, max DD, "
        "MIN trades per symbol) so no single symbol can carry the rest.",
    )
    parser.add_argument(
        "--train-test-split",
        type=float,
        default=DEFAULT_TRAIN_TEST_SPLIT,
        help="TRAIN fraction of the date range (default: 0.8 = last 20%% is a "
        "holdout used once as the final pass/fail gate; 0 disables the holdout).",
    )
    parser.add_argument(
        "--cross-window-months",
        type=str,
        default=None,
        help="Comma-separated month offsets for cross-window validation (e.g. '1,2'). "
        "Re-runs top strategies on sub-windows of the TRAIN range starting N months "
        "later; every shifted window must pass (the in-sample window never counts).",
    )
    parser.add_argument(
        "--holdout-min-sharpe",
        type=float,
        default=0.0,
        help="Holdout gate: holdout Sharpe must exceed this (default 0.0).",
    )
    parser.add_argument(
        "--holdout-min-trades",
        type=int,
        default=None,
        help="Holdout gate trade floor (default: half the train-gate trade rate, "
        "max(1, min_trades * holdout_days / (2 * train_days))).",
    )
    parser.add_argument(
        "--cross-window-min-sharpe",
        type=float,
        default=0.5,
        help="Min Sharpe on each shifted window to count as a pass (default: 0.5)",
    )
    parser.add_argument(
        "--seed-from-run",
        type=int,
        default=None,
        help="Seed initial population with top chromosomes from a prior discovery run ID",
    )
    parser.add_argument(
        "--wfa-oos-step-days",
        type=int,
        default=0,
        help="Rolling-window stability check step in days (0=disabled). Tiles the "
        "TRAIN range (never the holdout) with N-day windows.",
    )
    parser.add_argument(
        "--wfa-min-consistency",
        type=float,
        default=0.75,
        help="Min fraction of profitable rolling windows (default: 0.75 = 3/4)",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of random seeds to run. >1 enables multi-seed ensemble: "
        "runs GA N times, ranks by median Sharpe (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible GA runs. Default: random (recorded in "
        "notes). With --num-seeds >1 it is the base: seed i = base + i*7919 "
        "(default base 42).",
    )
    parser.add_argument(
        "--bootstrap-min-sharpe",
        type=float,
        default=None,
        help="Bootstrap CI lower bound threshold. Candidates whose bootstrap "
        "CI lower bound falls below this Sharpe are rejected. Timeframe-aware "
        "default: 0.5 for 1m (wider CIs on high-freq noise), 0.0 for 4h/1d "
        "(low trade counts make the lower bound structurally < 1.0), "
        "1.0 otherwise.",
    )
    parser.add_argument(
        "--bootstrap-ci-level",
        type=float,
        default=0.95,
        help="Confidence level for bootstrap Sharpe CI (default: 0.95).",
    )
    parser.add_argument(
        "--no-bootstrap-ci",
        dest="require_bootstrap_ci",
        action="store_false",
        default=True,
        help="Disable the bootstrap Sharpe CI hard guardrail.",
    )
    parser.add_argument(
        "--no-dsr",
        dest="require_dsr",
        action="store_false",
        default=True,
        help="Disable the Deflated Sharpe Ratio soft guardrail.",
    )
    parser.add_argument(
        "--immigrant-fraction",
        type=float,
        default=0.15,
        help=(
            "Fraction of population replaced when Shannon entropy drops below "
            "--entropy-threshold. 0 disables random-immigrant injection."
        ),
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=0.4,
        help="Population entropy below this triggers random-immigrant injection.",
    )
    parser.add_argument(
        "--no-crowding",
        dest="use_crowding",
        action="store_false",
        default=True,
        help="Disable deterministic crowding selection (falls back to classic tournament).",
    )
    parser.add_argument("--db", type=str, default=None, help="Database path")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Synthetic mock backtests (no NT). Results are flagged mock=true. "
        "Without --mock, missing catalog data is an error (never a silent mock).",
    )
    return parser


def main() -> int:
    """Run discovery pipeline and persist summary metrics for the run."""
    args = build_parser().parse_args()

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

        # Train/test split: split date range if requested
        train_start = args.start_date
        train_end = args.end_date
        holdout_start: str | None = None
        holdout_end: str | None = None
        split_ratio = args.train_test_split

        if split_ratio > 0:
            from vibe_quant.utils import split_date_range

            train_start, train_end, holdout_start, holdout_end = split_date_range(
                args.start_date, args.end_date, split_ratio,
            )
            logger.info(
                "Train/test split: ratio=%.2f train=%s→%s holdout=%s→%s",
                split_ratio, train_start, train_end, holdout_start, holdout_end,
            )

        # Resolve timeframe-aware bootstrap_min_sharpe default.
        if args.bootstrap_min_sharpe is None:
            args.bootstrap_min_sharpe = _default_bootstrap_min_sharpe(args.timeframe)
            logger.info(
                "Bootstrap min Sharpe default: %.1f (timeframe=%s)",
                args.bootstrap_min_sharpe, args.timeframe,
            )

        # Parse cross-window months
        cross_window_months: list[int] = []
        if args.cross_window_months:
            cross_window_months = [
                int(m.strip()) for m in args.cross_window_months.split(",") if m.strip()
            ]

        # Multi-window evaluation: split date range into sub-windows
        eval_windows_count = max(1, args.eval_windows)
        eval_windows: list[tuple[str, str]] | None = None
        if eval_windows_count >= 2:
            from vibe_quant.utils import split_into_windows

            eval_windows = split_into_windows(train_start, train_end, eval_windows_count)
            logger.info(
                "Multi-window fitness: %d windows — %s",
                eval_windows_count,
                " | ".join(f"{s}→{e}" for s, e in eval_windows),
            )

        config = DiscoveryConfig(
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
            backtest_fn = _mock_backtest
        else:
            if not _check_data_available(symbols):
                msg = (
                    f"No catalog bar data for symbols={symbols} — download data first "
                    "(Data Management) or pass --mock for a synthetic run"
                )
                raise RuntimeError(msg)
            logger.info("Using real NautilusTrader backtest for symbols=%s", symbols)
            backtest_fn = _make_nt_backtest_fn(
                symbols=symbols,
                timeframe=args.timeframe,
                start_date=train_start,
                end_date=train_end,
                windows=eval_windows,
                min_trades=config.min_trades,
                symbol_agg=args.symbol_agg,
            )

        # Create holdout backtest function if train/test split enabled
        holdout_backtest_fn = None
        if split_ratio > 0 and holdout_start and holdout_end:
            if use_mock:
                holdout_backtest_fn = _mock_backtest
            else:
                holdout_backtest_fn = _make_nt_backtest_fn(
                    symbols=symbols,
                    timeframe=args.timeframe,
                    start_date=holdout_start,
                    end_date=holdout_end,
                    symbol_agg=args.symbol_agg,
                )
        elif split_ratio <= 0:
            logger.warning(
                "No holdout (--train-test-split 0): champions get NO out-of-sample gate"
            )

        # Create backtest factory for cross-window and/or WFA validation
        backtest_fn_factory = None
        needs_factory = bool(cross_window_months) or args.wfa_oos_step_days > 0
        if needs_factory:
            if use_mock:
                backtest_fn_factory = lambda s, e: _mock_backtest  # noqa: E731
            else:
                _syms = symbols
                _tf = args.timeframe
                _agg: SymbolAgg = args.symbol_agg

                def backtest_fn_factory(s: str, e: str) -> NTBacktestFn:
                    return NTBacktestFn(_syms, _tf, s, e, symbol_agg=_agg)

        # Load seed chromosomes from prior run if requested
        seed_chromosomes = None
        if args.seed_from_run is not None:
            seed_chromosomes = _load_seed_chromosomes(state, args.seed_from_run)
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
        progress_file = f"logs/discovery_{args.run_id}_progress.json"

        if num_seeds == 1:
            # Single-seed run (default)
            seed = _resolve_seed(args.seed)
            logger.info("Discovery seed: %d", seed)
            random.seed(seed)
            pipeline = DiscoveryPipeline(
                config=config,
                backtest_fn=backtest_fn,
                progress_file=progress_file,
                holdout_backtest_fn=holdout_backtest_fn,
                backtest_fn_factory=backtest_fn_factory,
                seed_chromosomes=seed_chromosomes,
            )
            result = pipeline.run()
        else:
            # Multi-seed ensemble: run N times with different seeds
            seed = args.seed if args.seed is not None else 42
            result = _run_multi_seed(
                num_seeds=num_seeds,
                config=config,
                backtest_fn=backtest_fn,
                progress_file=progress_file,
                holdout_backtest_fn=holdout_backtest_fn,
                backtest_fn_factory=backtest_fn_factory,
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
        )

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
        from vibe_quant.discovery.genome import chromosome_to_dsl, chromosome_to_serializable

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
        full_range_fn = (
            _make_nt_backtest_fn(
                symbols=symbols,
                timeframe=args.timeframe,
                start_date=args.start_date,
                end_date=args.end_date,
                symbol_agg=args.symbol_agg,
            )
            if needs_full_range
            else None
        )
        if full_range_fn is not None:
            logger.info(
                "Computing full-range headline for top-%d champions over %s..%s "
                "(multi-window/split fitness != continuous replay; bd-rewru)",
                len(result.top_strategies[:5]), args.start_date, args.end_date,
            )

        top_dsls = []
        for idx, (chrom, fitness) in enumerate(result.top_strategies[:5]):
            dsl = chromosome_to_dsl(chrom)
            dsl["timeframe"] = args.timeframe
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
            if fitness.symbol_scores is not None:
                entry["symbol_scores"] = {
                    sym: round(sc, 6) for sym, sc in fitness.symbol_scores.items()
                }
            # Full-range headline (single continuous backtest) for like-for-like
            # promotion/replay_drift; sharpe/trades above stay as the multi-window
            # robustness aggregate. full_range_fn(chrom) never raises (NTBacktestFn
            # swallows backtest errors into failure metrics). bd vibe-quant-rewru.
            entry.update(
                full_range_headline(
                    full_range_fn(chrom) if full_range_fn is not None else None,
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
                train_metrics = _make_nt_backtest_fn(
                    symbols=symbols,
                    timeframe=args.timeframe,
                    start_date=result.train_dates[0],
                    end_date=result.train_dates[1],
                    symbol_agg=args.symbol_agg,
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
                "notes": json.dumps(
                    {
                        **run_notes,
                        "top_strategies": top_dsls,
                        "guardrail_rejections": result.guardrail_rejections or None,
                    }
                ),
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
        except Exception:
            pass
        print(f"Discovery failed: {exc}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        stop_heartbeat()
        state.close()
        job_manager.close()


if __name__ == "__main__":
    raise SystemExit(main())
