"""CLI entry point for vibe-quant."""

from __future__ import annotations

import argparse
import sys


def _format_fraction_as_percent(value: float | int | None, *, decimals: int = 2) -> str:
    """Format a fraction (0.05) as a percentage string (5.00%)."""
    if not isinstance(value, (int, float)):
        return "-"
    return f"{float(value) * 100:.{decimals}f}%"


def cmd_validation_run(args: argparse.Namespace) -> int:
    """Run validation backtest for a given run_id.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 for success).
    """
    from pathlib import Path

    from vibe_quant.db.connection import DEFAULT_DB_PATH
    from vibe_quant.jobs.manager import run_with_heartbeat
    from vibe_quant.validation.runner import ValidationRunner

    run_id = args.run_id
    latency = args.latency
    db_path = Path(args.db) if getattr(args, "db", None) else DEFAULT_DB_PATH

    print(f"Running validation backtest for run_id={run_id}")
    if latency:
        print(f"  Latency preset: {latency}")

    job_manager, stop_heartbeat = run_with_heartbeat(run_id, db_path)
    runner: ValidationRunner | None = None
    try:
        runner = ValidationRunner(db_path=db_path)
        result = runner.run(run_id=run_id, latency_preset=latency)

        job_manager.mark_completed(run_id)

        print("\nValidation Results:")
        print(f"  Strategy: {result.strategy_name}")
        print(f"  Total Return: {_format_fraction_as_percent(result.total_return, decimals=2)}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio: {result.sortino_ratio:.2f}")
        print(f"  Max Drawdown: {_format_fraction_as_percent(result.max_drawdown, decimals=2)}")
        print(f"  Total Trades: {result.total_trades}")
        print(f"  Win Rate: {_format_fraction_as_percent(result.win_rate, decimals=1)}")
        print(f"  Profit Factor: {result.profit_factor:.2f}")
        print(f"  Total Fees: ${result.total_fees:.2f}")
        print(f"  Total Funding: ${result.total_funding:.2f}")
        print(f"  Total Slippage: ${result.total_slippage:.2f}")
        print(f"  Execution Time: {result.execution_time_seconds:.2f}s")

        return 0

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        import contextlib

        with contextlib.suppress(Exception):
            job_manager.mark_completed(run_id, error=error_msg)
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        stop_heartbeat()
        if runner is not None:
            runner.close()
        job_manager.close()


def cmd_validation_list(args: argparse.Namespace) -> int:
    """List recent validation runs.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 for success).
    """
    from vibe_quant.validation.runner import list_validation_runs

    limit = args.limit

    runs = list_validation_runs(limit=limit)

    if not runs:
        print("No validation runs found.")
        return 0

    print(
        f"{'ID':<6} {'Strategy':<12} {'Status':<12} {'Sharpe':<8} {'Return':<10} {'Trades':<8} {'Created'}"
    )
    print("-" * 80)

    for run in runs:
        run_id = run.get("run_id", "")
        strategy = run.get("strategy_id", "")
        status = run.get("status", "")
        sharpe_raw = run.get("sharpe_ratio")
        total_return_raw = run.get("total_return")
        trades = run.get("total_trades")
        created_raw = run.get("created_at", "")
        created = str(created_raw)[:16] if created_raw else ""

        sharpe = float(sharpe_raw) if isinstance(sharpe_raw, (int, float)) else None
        total_return = (
            float(total_return_raw) if isinstance(total_return_raw, (int, float)) else None
        )
        sharpe_str = f"{sharpe:.2f}" if sharpe is not None else "-"
        return_str = _format_fraction_as_percent(total_return, decimals=1)
        trades_str = str(trades) if trades is not None else "-"

        print(
            f"{run_id:<6} {strategy:<12} {status:<12} {sharpe_str:<8} {return_str:<10} {trades_str:<8} {created}"
        )

    return 0


def cmd_data(_args: argparse.Namespace, extra: list[str] | None = None) -> int:
    """Forward to data module CLI.

    Args:
        _args: Parsed CLI arguments (unused, forwarding via extra).
        extra: Remaining arguments for submodule.

    Returns:
        Exit code.
    """
    from vibe_quant.data.ingest import main as data_main

    return data_main(extra or [])


def cmd_screening(_args: argparse.Namespace, extra: list[str] | None = None) -> int:
    """Forward to screening module CLI.

    Args:
        _args: Parsed CLI arguments (unused, forwarding via extra).
        extra: Remaining arguments for submodule.

    Returns:
        Exit code.
    """
    from vibe_quant.screening.__main__ import main as screening_main

    return screening_main(extra or [])


def cmd_overfitting(_args: argparse.Namespace, extra: list[str] | None = None) -> int:
    """Forward to overfitting module CLI.

    Args:
        _args: Parsed CLI arguments (unused, forwarding via extra).
        extra: Remaining arguments for submodule.

    Returns:
        Exit code.
    """
    from vibe_quant.overfitting.__main__ import main as overfitting_main

    return overfitting_main(extra or [])


def cmd_discovery(_args: argparse.Namespace, extra: list[str] | None = None) -> int:
    """Forward to discovery module CLI.

    Args:
        _args: Parsed CLI arguments (unused, forwarding via extra).
        extra: Remaining arguments for submodule.

    Returns:
        Exit code.
    """
    import sys as _sys

    from vibe_quant.discovery.__main__ import main as discovery_main

    # Override sys.argv so discovery's argparse picks up the extra args
    old_argv = _sys.argv
    _sys.argv = ["vibe-quant discovery", *(extra or [])]
    try:
        return discovery_main()
    finally:
        _sys.argv = old_argv


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="vibe-quant",
        description="Algorithmic trading engine for crypto perpetual futures",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="Available commands",
    )

    # Data command - uses parse_known_args forwarding
    data_parser = subparsers.add_parser(
        "data",
        help="Data management commands",
    )
    data_parser.set_defaults(func=cmd_data)

    # Screening command - uses parse_known_args forwarding
    screening_parser = subparsers.add_parser(
        "screening",
        help="Parameter sweep screening",
    )
    screening_parser.set_defaults(func=cmd_screening)

    # Overfitting command - uses parse_known_args forwarding
    overfitting_parser = subparsers.add_parser(
        "overfitting",
        help="Overfitting prevention pipeline",
    )
    overfitting_parser.set_defaults(func=cmd_overfitting)

    # Discovery command - uses parse_known_args forwarding
    discovery_parser = subparsers.add_parser(
        "discovery",
        help="Genetic algorithm strategy discovery",
    )
    discovery_parser.set_defaults(func=cmd_discovery)

    # Validation command
    validation_parser = subparsers.add_parser(
        "validation",
        help="Full-fidelity validation backtesting",
    )
    validation_subparsers = validation_parser.add_subparsers(
        title="validation commands",
        dest="validation_command",
    )

    # validation run
    val_run_parser = validation_subparsers.add_parser(
        "run",
        help="Run validation backtest",
    )
    val_run_parser.add_argument(
        "--run-id",
        type=int,
        required=True,
        help="Backtest run ID from database",
    )
    val_run_parser.add_argument(
        "--latency",
        type=str,
        choices=["co_located", "domestic", "international", "retail"],
        default=None,
        help="Override latency preset (default: from database or retail)",
    )
    val_run_parser.set_defaults(func=cmd_validation_run)

    # validation list
    val_list_parser = validation_subparsers.add_parser(
        "list",
        help="List validation runs",
    )
    val_list_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum runs to show (default: 20)",
    )
    val_list_parser.set_defaults(func=cmd_validation_list)

    return parser


def main() -> int:
    """Main entry point.

    Returns:
        Exit code.
    """
    parser = build_parser()
    # parse_known_args so dashed args like --run-id can be forwarded
    args, extra = parser.parse_known_args()

    if args.command is None:
        parser.print_help()
        return 0

    # Handle validation subcommands
    if args.command == "validation" and (
        not hasattr(args, "validation_command") or args.validation_command is None
    ):
        parser.parse_args(["validation", "--help"])
        return 0

    if hasattr(args, "func"):
        # Forward extra args to submodule commands (data, screening)
        if args.command in ("data", "screening", "overfitting", "discovery"):
            result: int = args.func(args, extra=extra)
        else:
            result = args.func(args)
        return result

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
