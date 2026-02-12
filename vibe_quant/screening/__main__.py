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
    from vibe_quant.jobs.manager import run_with_heartbeat

    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH
    state = StateManager(db_path)
    job_manager = run_with_heartbeat(args.run_id, db_path)

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

        from vibe_quant.dsl.schema import StrategyDSL
        from vibe_quant.screening.pipeline import create_screening_pipeline

        dsl = StrategyDSL.model_validate(strategy["dsl_config"])

        # Extract run parameters
        symbols = run_config["symbols"]
        start_date = run_config["start_date"]
        end_date = run_config["end_date"]
        parameters = run_config.get("parameters", {})
        sweep_params = parameters.get("sweep", {})

        # Apply sweep overrides to DSL if present
        if sweep_params:
            dsl_dict = dsl.model_dump()
            dsl_dict["sweep"] = sweep_params
            dsl = StrategyDSL.model_validate(dsl_dict)

        pipeline = create_screening_pipeline(
            dsl,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )
        result = pipeline.run()
        pipeline.save_results(result, state, args.run_id)
        state.update_backtest_run_status(args.run_id, "completed")
        job_manager.mark_completed(args.run_id)
        print(f"Screening complete: {result.total_combinations} combos, {len(result.pareto_optimal_indices)} Pareto-optimal")
        return 0
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        try:
            state.update_backtest_run_status(
                args.run_id, "failed", error_message=error_msg
            )
            job_manager.mark_completed(args.run_id, error=error_msg)
        except Exception:
            pass
        print(f"Screening failed: {exc}")
        return 1
    finally:
        state.close()
        job_manager.close()


if __name__ == "__main__":
    sys.exit(main())
