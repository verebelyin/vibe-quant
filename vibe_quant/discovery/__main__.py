"""Discovery pipeline CLI entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vibe_quant.discovery.operators import StrategyChromosome


def _mock_backtest(chromosome: StrategyChromosome) -> dict[str, Any]:
    """PLACEHOLDER: Generate deterministic pseudo-backtest metrics.

    WARNING: This does NOT run actual backtests. It produces fake metrics
    derived from chromosome structure for testing the genetic evolution loop.
    Replace with real screening backtest integration before production use.
    """
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
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--symbols", type=str, default="BTCUSDT")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--db", type=str, default=None, help="Database path")
    return parser


def main() -> int:
    """Run discovery pipeline and persist summary metrics for the run."""
    args = build_parser().parse_args()

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
            elite_count=args.elite_count,
            symbols=symbols,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        logger.warning(
            "Using MOCK backtest function — results are synthetic. "
            "Replace _mock_backtest with real screening integration for production."
        )
        pipeline = DiscoveryPipeline(config=config, backtest_fn=_mock_backtest)
        result = pipeline.run()

        if not result.top_strategies:
            raise RuntimeError("Discovery produced no strategies")

        _best_chrom, best_fitness = result.top_strategies[0]
        execution_time = time.perf_counter() - started_at

        state.save_backtest_result(
            args.run_id,
            {
                "total_return": best_fitness.adjusted_score,
                "sharpe_ratio": best_fitness.sharpe_ratio,
                "max_drawdown": best_fitness.max_drawdown,
                "profit_factor": best_fitness.profit_factor,
                "total_trades": best_fitness.total_trades,
                "execution_time_seconds": execution_time,
                "notes": (
                    f"discovery: generations={len(result.generations)}, "
                    f"evaluated={result.total_candidates_evaluated}, "
                    f"converged={result.converged}"
                ),
            },
        )
        state.update_backtest_run_status(args.run_id, "completed")
        job_manager.mark_completed(args.run_id)

        print(
            "Discovery complete: "
            f"top_score={best_fitness.adjusted_score:.4f}, "
            f"candidates={result.total_candidates_evaluated}"
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
        return 1
    finally:
        stop_heartbeat()
        state.close()
        job_manager.close()


if __name__ == "__main__":
    raise SystemExit(main())
