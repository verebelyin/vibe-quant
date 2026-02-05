"""Screening pipeline CLI entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    """Run screening pipeline from CLI."""
    parser = argparse.ArgumentParser(description="Run screening pipeline")
    parser.add_argument("--run-id", type=int, required=True, help="Backtest run ID")
    parser.add_argument("--db", type=str, default=None, help="Database path")
    args = parser.parse_args()

    from vibe_quant.db.connection import DEFAULT_DB_PATH
    from vibe_quant.db.state_manager import StateManager

    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH
    state = StateManager(db_path)

    try:
        run_config = state.get_backtest_run(args.run_id)
        if run_config is None:
            print(f"Run {args.run_id} not found")
            return 1

        strategy_id = run_config["strategy_id"]
        strategy = state.get_strategy(int(str(strategy_id)))
        if strategy is None:
            print(f"Strategy {strategy_id} not found")
            return 1

        from vibe_quant.dsl.parser import parse_strategy
        from vibe_quant.screening.pipeline import create_screening_pipeline

        dsl = parse_strategy(strategy["dsl_config"])
        pipeline = create_screening_pipeline(dsl)
        result = pipeline.run(run_id=args.run_id, db_path=db_path)
        print(f"Screening complete: {result.total_combinations} combos, {len(result.pareto_optimal_indices)} Pareto-optimal")
        return 0
    except Exception as exc:
        print(f"Screening failed: {exc}")
        return 1
    finally:
        state.close()


if __name__ == "__main__":
    sys.exit(main())
