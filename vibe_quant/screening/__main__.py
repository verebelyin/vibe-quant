"""Screening pipeline CLI entrypoint.

Provides run/list/status subcommands for screening backtests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_run(args: argparse.Namespace) -> int:
    """Run screening pipeline for a given run_id.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 for success).
    """
    from vibe_quant.db.connection import DEFAULT_DB_PATH
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.jobs.manager import run_with_heartbeat

    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH
    state = StateManager(db_path)
    manager, stop_heartbeat = run_with_heartbeat(args.run_id, db_path)

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
        manager.mark_completed(args.run_id)
        print(f"Screening complete: {result.total_combinations} combos, {len(result.pareto_optimal_indices)} Pareto-optimal")
        return 0
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        try:
            state.update_backtest_run_status(
                args.run_id, "failed", error_message=error_msg
            )
            manager.mark_completed(args.run_id, error=error_msg)
        except Exception:
            pass
        print(f"Screening failed: {exc}")
        return 1
    finally:
        stop_heartbeat()
        state.close()
        manager.close()


def cmd_list(args: argparse.Namespace) -> int:
    """List screening backtest runs.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 for success).
    """
    from vibe_quant.db.connection import DEFAULT_DB_PATH
    from vibe_quant.db.state_manager import StateManager

    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH
    state = StateManager(db_path)

    try:
        runs = state.list_backtest_runs(status=args.status)
        screening_runs = [r for r in runs if r.get("run_mode") == "screening"]

        if args.limit:
            screening_runs = screening_runs[: args.limit]

        if not screening_runs:
            print("No screening runs found.")
            return 0

        print(f"{'ID':<6} {'Strategy':<12} {'Status':<12} {'Symbols':<20} {'Created'}")
        print("-" * 70)

        for run in screening_runs:
            run_id = run.get("id", "")
            strategy = run.get("strategy_id", "")
            status = run.get("status", "")
            symbols = run.get("symbols", [])
            symbols_str = ",".join(symbols[:2])
            if len(symbols) > 2:
                symbols_str += f"+{len(symbols) - 2}"
            created_raw = run.get("created_at", "")
            created = str(created_raw)[:16] if created_raw else ""

            print(f"{run_id:<6} {strategy:<12} {status:<12} {symbols_str:<20} {created}")

        return 0
    finally:
        state.close()


def cmd_status(args: argparse.Namespace) -> int:
    """Show status of a specific screening run.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 for success).
    """
    from vibe_quant.db.connection import DEFAULT_DB_PATH
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.jobs.manager import BacktestJobManager

    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH
    state = StateManager(db_path)
    job_mgr = BacktestJobManager(db_path)

    try:
        run_config = state.get_backtest_run(args.run_id)
        if run_config is None:
            print(f"Run {args.run_id} not found")
            return 1

        print(f"Run ID:     {args.run_id}")
        print(f"Strategy:   {run_config.get('strategy_id', '-')}")
        print(f"Mode:       {run_config.get('run_mode', '-')}")
        print(f"Status:     {run_config.get('status', '-')}")
        print(f"Symbols:    {run_config.get('symbols', [])}")
        print(f"Timeframe:  {run_config.get('timeframe', '-')}")
        print(f"Start:      {run_config.get('start_date', '-')}")
        print(f"End:        {run_config.get('end_date', '-')}")
        print(f"Created:    {run_config.get('created_at', '-')}")

        job_info = job_mgr.get_job_info(args.run_id)
        if job_info is not None:
            print(f"\nJob PID:    {job_info.pid}")
            print(f"Job Status: {job_info.status.value}")
            print(f"Heartbeat:  {job_info.heartbeat_at or '-'}")
            print(f"Stale:      {job_info.is_stale}")
            if job_info.log_file:
                print(f"Log File:   {job_info.log_file}")

        error = run_config.get("error_message")
        if error:
            print(f"\nError:      {error}")

        return 0
    finally:
        state.close()
        job_mgr.close()


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser with run/list/status subcommands.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="vibe-quant screening",
        description="Screening pipeline parameter sweep",
    )
    parser.add_argument("--db", type=str, default=None, help="Database path")

    subparsers = parser.add_subparsers(
        title="commands",
        dest="subcommand",
        help="Available screening commands",
    )

    # run subcommand
    run_parser = subparsers.add_parser("run", help="Run screening pipeline")
    run_parser.add_argument("--run-id", type=int, required=True, help="Backtest run ID")
    run_parser.add_argument("--strategy-id", type=int, default=None, help="Strategy ID (uses run's strategy if omitted)")
    run_parser.add_argument("--symbols", type=str, nargs="+", default=None, help="Override symbols")
    run_parser.add_argument("--timeframe", type=str, default=None, help="Override timeframe")
    run_parser.add_argument("--db", type=str, default=None, help="Database path")
    run_parser.set_defaults(func=cmd_run)

    # list subcommand
    list_parser = subparsers.add_parser("list", help="List screening runs")
    list_parser.add_argument("--limit", type=int, default=20, help="Max runs to show (default: 20)")
    list_parser.add_argument("--status", type=str, default=None, help="Filter by status")
    list_parser.add_argument("--db", type=str, default=None, help="Database path")
    list_parser.set_defaults(func=cmd_list)

    # status subcommand
    status_parser = subparsers.add_parser("status", help="Show run status")
    status_parser.add_argument("--run-id", type=int, required=True, help="Backtest run ID")
    status_parser.add_argument("--db", type=str, default=None, help="Database path")
    status_parser.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run screening CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand is None:
        parser.print_help()
        return 0

    func = getattr(args, "func", None)
    if callable(func):
        return int(func(args))

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
