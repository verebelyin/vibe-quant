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

    return {
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "total_trades": total_trades,
    }


def _make_nt_backtest_fn(
    symbols: list[str],
    timeframe: str,
    start_date: str,
    end_date: str,
) -> callable:
    """Create a backtest function using real NautilusTrader screening runner.

    Returns a callable that takes a StrategyChromosome and returns metrics dict.
    Falls back to mock if NT imports fail or data is unavailable.
    """
    from vibe_quant.discovery.genome import chromosome_to_dsl

    def _backtest(chromosome: StrategyChromosome) -> dict[str, float | int]:
        try:
            dsl_dict = chromosome_to_dsl(chromosome)
            # Override timeframe to match what we have data for
            dsl_dict["timeframe"] = timeframe

            from vibe_quant.screening.nt_runner import NTScreeningRunner

            runner = NTScreeningRunner(
                dsl_dict=dsl_dict,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
            )
            # Run with empty params (discovery doesn't sweep)
            result = runner({})

            return {
                "sharpe_ratio": result.sharpe_ratio if result.sharpe_ratio != float("-inf") else -1.0,
                "max_drawdown": result.max_drawdown,
                "profit_factor": result.profit_factor,
                "total_trades": result.total_trades,
            }
        except Exception as exc:
            logger.warning("NT backtest failed for chromosome %s: %s", chromosome.uid, exc)
            # Return poor metrics so GA deprioritizes failing strategies
            return {
                "sharpe_ratio": -1.0,
                "max_drawdown": 1.0,
                "profit_factor": 0.0,
                "total_trades": 0,
            }

    return _backtest


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
    parser.add_argument("--population-size", type=int, default=50)
    parser.add_argument("--max-generations", type=int, default=100)
    parser.add_argument("--mutation-rate", type=float, default=0.1)
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--convergence-generations", type=int, default=10)
    parser.add_argument("--symbols", type=str, default="BTCUSDT")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    parser.add_argument("--end-date", type=str, default="2026-02-24")
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

        config = DiscoveryConfig(
            population_size=args.population_size,
            max_generations=args.max_generations,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            elite_count=args.elite_count,
            tournament_size=args.tournament_size,
            convergence_generations=args.convergence_generations,
            symbols=symbols,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
        )

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
                start_date=args.start_date,
                end_date=args.end_date,
            )

        pipeline = DiscoveryPipeline(config=config, backtest_fn=backtest_fn)
        result = pipeline.run()

        if not result.top_strategies:
            raise RuntimeError("Discovery produced no strategies")

        _best_chrom, best_fitness = result.top_strategies[0]
        execution_time = time.perf_counter() - started_at

        # Save top strategies as DSL dicts in notes
        from vibe_quant.discovery.genome import chromosome_to_dsl
        import json

        top_dsls = []
        for chrom, fitness in result.top_strategies[:5]:
            dsl = chromosome_to_dsl(chrom)
            dsl["timeframe"] = args.timeframe
            top_dsls.append({
                "dsl": dsl,
                "score": fitness.adjusted_score,
                "sharpe": fitness.sharpe_ratio,
                "max_dd": fitness.max_drawdown,
                "pf": fitness.profit_factor,
                "trades": fitness.total_trades,
            })

        state.save_backtest_result(
            args.run_id,
            {
                "total_return": best_fitness.adjusted_score,
                "sharpe_ratio": best_fitness.sharpe_ratio,
                "max_drawdown": best_fitness.max_drawdown,
                "profit_factor": best_fitness.profit_factor,
                "total_trades": best_fitness.total_trades,
                "execution_time_seconds": execution_time,
                "notes": json.dumps({
                    "type": "discovery",
                    "generations": len(result.generations),
                    "evaluated": result.total_candidates_evaluated,
                    "converged": result.converged,
                    "mock": use_mock,
                    "top_strategies": top_dsls,
                }),
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
