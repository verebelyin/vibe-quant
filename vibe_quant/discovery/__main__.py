"""Discovery pipeline CLI entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline, DiscoveryResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vibe_quant.discovery.operators import StrategyChromosome


def _get_compiler_version() -> str:
    """Get compiler version hash for staleness detection."""
    try:
        from vibe_quant.dsl.compiler import compiler_version_hash

        return compiler_version_hash()
    except Exception:
        return "unknown"


def _mock_backtest(chromosome: StrategyChromosome) -> dict[str, float | int]:
    """Generate deterministic pseudo-backtest metrics for testing GA loop."""
    seed_bytes = repr(chromosome).encode("utf-8")
    seed = int.from_bytes(hashlib.blake2b(seed_bytes, digest_size=8).digest(), "big")
    rng = random.Random(seed)

    genes = len(chromosome.entry_genes) + len(chromosome.exit_genes)
    complexity = min(1.0, genes / 20.0)

    sharpe = max(0.05, 1.8 - (0.7 * complexity) + rng.uniform(-0.35, 0.35))
    max_drawdown = min(0.95, max(0.02, 0.14 + (0.2 * complexity) + rng.uniform(-0.05, 0.08)))
    profit_factor = max(0.2, 1.6 - (0.5 * complexity) + rng.uniform(-0.25, 0.35))
    total_trades = max(60, int(120 + rng.randint(-30, 90) - (genes * 2)))

    # Estimate total return from metrics
    total_return = max(-0.5, (sharpe * 0.15) - (max_drawdown * 0.3) + rng.uniform(-0.1, 0.2))

    # Generate synthetic per-trade returns for bootstrap CI guardrail.
    # Std must be small relative to mean for bootstrap CI to pass.
    mean_ret = total_return / max(1, total_trades)
    trade_returns = tuple(
        rng.gauss(mean_ret, abs(mean_ret) * 0.5 + 0.0001)
        for _ in range(total_trades)
    )

    return {
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "total_trades": total_trades,
        "total_return": total_return,
        "trade_returns": trade_returns,
    }


class _NTBacktestFn:
    """Picklable backtest callable for ProcessPoolExecutor.

    Must be a top-level class (not a closure) so multiprocessing can pickle it.
    Supports multi-window evaluation: when ``windows`` has 2+ entries, runs
    the backtest on each window and returns worst-case metrics across all
    windows, forcing the GA to find regime-robust strategies.
    """

    def __init__(
        self,
        symbols: list[str],
        timeframe: str,
        start_date: str,
        end_date: str,
        windows: list[tuple[str, str]] | None = None,
    ) -> None:
        self.symbols = symbols
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.windows = windows

    def _run_single(
        self,
        chromosome: StrategyChromosome,
        start_date: str,
        end_date: str,
    ) -> dict[str, float | int]:
        from vibe_quant.discovery.genome import chromosome_to_dsl
        from vibe_quant.screening.nt_runner import NTScreeningRunner

        dsl_dict = chromosome_to_dsl(chromosome)
        dsl_dict["timeframe"] = self.timeframe

        runner = NTScreeningRunner(
            dsl_dict=dsl_dict,
            symbols=self.symbols,
            start_date=start_date,
            end_date=end_date,
        )
        result = runner({})

        return {
            "sharpe_ratio": result.sharpe_ratio
            if result.sharpe_ratio != float("-inf")
            else -1.0,
            "max_drawdown": result.max_drawdown,
            "profit_factor": result.profit_factor,
            "total_trades": result.total_trades,
            "total_return": getattr(result, "total_return", 0.0),
            "skewness": getattr(result, "skewness", 0.0),
            "kurtosis": getattr(result, "kurtosis", 3.0),
            "trade_returns": getattr(result, "trade_returns", ()),  # type: ignore[arg-type,dict-item]
        }

    @staticmethod
    def _aggregate_multi_window(
        results: list[dict[str, float | int]],
    ) -> dict[str, float | int]:
        """Aggregate metrics across multiple window results.

        Strategy: require ALL windows to produce trades. If any window
        has 0 trades, return failure metrics. Otherwise:
        - total_trades: sum (statistical significance across all data)
        - sharpe_ratio: mean (consistent performance)
        - max_drawdown: max (worst case)
        - profit_factor: trade-weighted mean
        - total_return: mean per-window return
        """
        n = len(results)
        per_window_trades = [int(r["total_trades"]) for r in results]

        # If any window has 0 trades, strategy doesn't cover that regime
        if any(t == 0 for t in per_window_trades):
            return {
                "sharpe_ratio": -1.0,
                "max_drawdown": 1.0,
                "profit_factor": 0.0,
                "total_trades": 0,
                "total_return": 0.0,
            }

        total_trades_sum = sum(per_window_trades)

        # Trade-weighted profit factor
        pf_weighted = sum(
            float(r["profit_factor"]) * t
            for r, t in zip(results, per_window_trades)
        ) / total_trades_sum

        return {
            "sharpe_ratio": sum(float(r["sharpe_ratio"]) for r in results) / n,
            "max_drawdown": max(float(r["max_drawdown"]) for r in results),
            "profit_factor": pf_weighted,
            "total_trades": total_trades_sum,
            "total_return": sum(float(r["total_return"]) for r in results) / n,
            "skewness": sum(float(r.get("skewness", 0.0)) for r in results) / n,  # type: ignore[arg-type]
            "kurtosis": max(float(r.get("kurtosis", 3.0)) for r in results),  # type: ignore[arg-type]
            "trade_returns": sum(
                (r.get("trade_returns", ()) for r in results), ()  # type: ignore[arg-type]
            ),
        }

    def __call__(self, chromosome: StrategyChromosome) -> dict[str, float | int]:
        try:
            if self.windows and len(self.windows) >= 2:
                results = [
                    self._run_single(chromosome, ws, we)
                    for ws, we in self.windows
                ]
                return self._aggregate_multi_window(results)
            return self._run_single(chromosome, self.start_date, self.end_date)
        except Exception as exc:
            logger.warning("NT backtest failed for chromosome %s: %s", chromosome.uid, exc)
            return {
                "sharpe_ratio": -1.0,
                "max_drawdown": 1.0,
                "profit_factor": 0.0,
                "total_trades": 0,
            }


def _make_nt_backtest_fn(
    symbols: list[str],
    timeframe: str,
    start_date: str,
    end_date: str,
    windows: list[tuple[str, str]] | None = None,
) -> _NTBacktestFn:
    """Create a picklable backtest function using real NautilusTrader screening runner."""
    return _NTBacktestFn(symbols, timeframe, start_date, end_date, windows=windows)


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
            object,
            DiscoveryResult,
            int,
        ]
    ] = []
    all_generations: list[object] = []
    total_evaluated = 0
    seed_stats: list[dict[str, float]] = []
    any_converged = False
    convergence_gen: int | None = None
    result_metadata: DiscoveryResult | None = None

    for seed_idx in range(num_seeds):
        seed_val = seed_idx * 7919 + 42  # Deterministic but varied seeds
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
                object,
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
                object,
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
            object,
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
    )


def _load_seed_chromosomes(
    state: object,
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
        default=1,
        help="Split date range into N sub-windows, evaluate fitness on worst-case. "
        "Forces GA to find regime-robust strategies (default: 1 = single window).",
    )
    parser.add_argument(
        "--train-test-split",
        type=float,
        default=0.0,
        help="Train/test split ratio (0=disabled, 0.5=50/50 split). "
        "GA trains on first portion, validates on remainder.",
    )
    parser.add_argument(
        "--cross-window-months",
        type=str,
        default=None,
        help="Comma-separated month offsets for cross-window validation (e.g. '1,2'). "
        "Re-runs top strategies on shifted windows; must pass on 2/3 to be promoted.",
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
        help="WFA rolling window step in days (0=disabled). "
        "Requires --train-test-split. Splits holdout into rolling windows.",
    )
    parser.add_argument(
        "--wfa-min-consistency",
        type=float,
        default=0.75,
        help="Min fraction of profitable WFA windows (default: 0.75 = 3/4)",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of random seeds to run. >1 enables multi-seed ensemble: "
        "runs GA N times, ranks by median Sharpe (default: 1)",
    )
    parser.add_argument("--db", type=str, default=None, help="Database path")
    parser.add_argument("--mock", action="store_true", help="Force mock backtest (no NT)")
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
            train_test_split=split_ratio,
            holdout_start_date=holdout_start or "",
            holdout_end_date=holdout_end or "",
            cross_window_months=cross_window_months,
            cross_window_min_sharpe=args.cross_window_min_sharpe,
            wfa_oos_step_days=args.wfa_oos_step_days,
            wfa_min_consistency=args.wfa_min_consistency,
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

        # Choose backtest function: real NT if data available, else mock
        use_mock = args.mock or not _check_data_available(symbols)
        if use_mock:
            logger.warning(
                "Using MOCK backtest — %s. Results are synthetic.",
                "forced via --mock" if args.mock else "no catalog data for symbols",
            )
            backtest_fn = _mock_backtest
        else:
            logger.info("Using real NautilusTrader backtest for symbols=%s", symbols)
            backtest_fn = _make_nt_backtest_fn(
                symbols=symbols,
                timeframe=args.timeframe,
                start_date=train_start,
                end_date=train_end,
                windows=eval_windows,
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

                def backtest_fn_factory(s: str, e: str) -> _NTBacktestFn:
                    return _NTBacktestFn(_syms, _tf, s, e)

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
                logger.warning(
                    "No chromosomes found in run %d — starting with random population",
                    args.seed_from_run,
                )

        num_seeds = max(1, args.num_seeds)
        progress_file = f"logs/discovery_{args.run_id}_progress.json"

        if num_seeds == 1:
            # Single-seed run (default)
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
            result = _run_multi_seed(
                num_seeds=num_seeds,
                config=config,
                backtest_fn=backtest_fn,
                progress_file=progress_file,
                holdout_backtest_fn=holdout_backtest_fn,
                backtest_fn_factory=backtest_fn_factory,
                seed_chromosomes=seed_chromosomes,
            )

        if not result.top_strategies:
            raise RuntimeError("Discovery produced no strategies")

        _best_chrom, best_fitness = result.top_strategies[0]
        execution_time = time.perf_counter() - started_at

        # Save top strategies as DSL dicts in notes
        import json

        from vibe_quant.discovery.genome import chromosome_to_dsl, chromosome_to_serializable

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
            # Attach cross-window results if available
            if idx < len(result.cross_window_results):
                cwr = result.cross_window_results[idx]
                entry["cross_window"] = {
                    "windows_passed": cwr.windows_passed,
                    "total_windows": cwr.total_windows,
                    "passed": cwr.passed,
                    "windows": [
                        {
                            "sharpe": w.sharpe_ratio,
                            "max_dd": w.max_drawdown,
                            "pf": w.profit_factor,
                            "trades": w.total_trades,
                            "return_pct": w.total_return,
                        }
                        for w in cwr.window_results
                    ],
                }
            # Attach WFA rolling results if available
            if idx < len(result.wfa_results):
                wfa = result.wfa_results[idx]
                entry["wfa"] = {
                    "windows_profitable": wfa.windows_profitable,
                    "total_windows": wfa.total_windows,
                    "consistency": wfa.consistency,
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
            top_dsls.append(entry)

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
                "notes": json.dumps(
                    {
                        "type": "discovery",
                        "generations": len(result.generations),
                        "evaluated": result.total_candidates_evaluated,
                        "converged": result.converged,
                        "mock": use_mock,
                        "compiler_version": _get_compiler_version(),
                        "eval_windows": eval_windows_count if eval_windows_count > 1 else None,
                        "eval_window_ranges": eval_windows if eval_windows else None,
                        "train_test_split": split_ratio if split_ratio > 0 else None,
                        "train_dates": list(result.train_dates) if result.train_dates else None,
                        "holdout_dates": list(result.holdout_dates) if result.holdout_dates else None,
                        "direction": args.direction,
                        "cross_window_months": cross_window_months or None,
                        "cross_window_min_sharpe": (
                            args.cross_window_min_sharpe if cross_window_months else None
                        ),
                        "wfa_oos_step_days": args.wfa_oos_step_days if args.wfa_oos_step_days > 0 else None,
                        "num_seeds": num_seeds if num_seeds > 1 else None,
                        "top_strategies": top_dsls,
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
