"""Discovery pipeline CLI entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline

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
    """

    def __init__(
        self,
        symbols: list[str],
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> None:
        self.symbols = symbols
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date

    def __call__(self, chromosome: StrategyChromosome) -> dict[str, float | int]:
        try:
            from vibe_quant.discovery.genome import chromosome_to_dsl
            from vibe_quant.screening.nt_runner import NTScreeningRunner

            dsl_dict = chromosome_to_dsl(chromosome)
            dsl_dict["timeframe"] = self.timeframe

            runner = NTScreeningRunner(
                dsl_dict=dsl_dict,
                symbols=self.symbols,
                start_date=self.start_date,
                end_date=self.end_date,
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
) -> _NTBacktestFn:
    """Create a picklable backtest function using real NautilusTrader screening runner."""
    return _NTBacktestFn(symbols, timeframe, start_date, end_date)


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
            train_test_split=split_ratio,
            cross_window_months=cross_window_months,
            cross_window_min_sharpe=args.cross_window_min_sharpe,
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

        # Create backtest factory for cross-window validation
        backtest_fn_factory = None
        if cross_window_months:
            if use_mock:
                backtest_fn_factory = lambda s, e: _mock_backtest  # noqa: E731
            else:
                _syms = symbols
                _tf = args.timeframe

                def backtest_fn_factory(s: str, e: str) -> _NTBacktestFn:
                    return _NTBacktestFn(_syms, _tf, s, e)

        progress_file = f"logs/discovery_{args.run_id}_progress.json"
        pipeline = DiscoveryPipeline(
            config=config,
            backtest_fn=backtest_fn,
            progress_file=progress_file,
            holdout_backtest_fn=holdout_backtest_fn,
            backtest_fn_factory=backtest_fn_factory,
        )
        result = pipeline.run()

        if not result.top_strategies:
            raise RuntimeError("Discovery produced no strategies")

        _best_chrom, best_fitness = result.top_strategies[0]
        execution_time = time.perf_counter() - started_at

        # Save top strategies as DSL dicts in notes
        import json

        from vibe_quant.discovery.genome import chromosome_to_dsl

        top_dsls = []
        for idx, (chrom, fitness) in enumerate(result.top_strategies[:5]):
            dsl = chromosome_to_dsl(chrom)
            dsl["timeframe"] = args.timeframe
            entry: dict[str, object] = {
                "dsl": dsl,
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
                        "train_test_split": split_ratio if split_ratio > 0 else None,
                        "train_dates": list(result.train_dates) if result.train_dates else None,
                        "holdout_dates": list(result.holdout_dates) if result.holdout_dates else None,
                        "cross_window_months": cross_window_months or None,
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
