"""CLI entry point for vibe-quant."""

from __future__ import annotations

import argparse
import sys


def cmd_validation_run(args: argparse.Namespace) -> int:
    """Run validation backtest for a given run_id.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 for success).
    """
    from vibe_quant.validation.runner import ValidationRunner, ValidationRunnerError

    run_id = args.run_id
    latency = args.latency

    print(f"Running validation backtest for run_id={run_id}")
    if latency:
        print(f"  Latency preset: {latency}")

    try:
        runner = ValidationRunner()
        result = runner.run(run_id=run_id, latency_preset=latency)
        runner.close()

        print("\nValidation Results:")
        print(f"  Strategy: {result.strategy_name}")
        print(f"  Total Return: {result.total_return * 100:.2f}%")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio: {result.sortino_ratio:.2f}")
        print(f"  Max Drawdown: {result.max_drawdown:.2f}%")
        print(f"  Total Trades: {result.total_trades}")
        print(f"  Win Rate: {result.win_rate * 100:.1f}%")
        print(f"  Profit Factor: {result.profit_factor:.2f}")
        print(f"  Total Fees: ${result.total_fees:.2f}")
        print(f"  Total Funding: ${result.total_funding:.2f}")
        print(f"  Total Slippage: ${result.total_slippage:.2f}")
        print(f"  Execution Time: {result.execution_time_seconds:.2f}s")

        return 0

    except ValidationRunnerError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


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

    print(f"{'ID':<6} {'Strategy':<12} {'Status':<12} {'Sharpe':<8} {'Return':<10} {'Trades':<8} {'Created'}")
    print("-" * 80)

    for run in runs:
        run_id = run.get("run_id", "")
        strategy = run.get("strategy_id", "")
        status = run.get("status", "")
        sharpe = run.get("sharpe_ratio")
        total_return = run.get("total_return")
        trades = run.get("total_trades")
        created_raw = run.get("created_at", "")
        created = str(created_raw)[:16] if created_raw else ""

        sharpe_str = f"{sharpe:.2f}" if sharpe is not None else "-"
        return_str = f"{total_return * 100:.1f}%" if total_return is not None else "-"
        trades_str = str(trades) if trades is not None else "-"

        print(f"{run_id:<6} {strategy:<12} {status:<12} {sharpe_str:<8} {return_str:<10} {trades_str:<8} {created}")

    return 0


def cmd_data(_args: argparse.Namespace, extra: list[str] | None = None) -> int:
    """Forward to data module CLI.

    Args:
        _args: Parsed CLI arguments (unused, forwarding via extra).
        extra: Remaining arguments for submodule.

    Returns:
        Exit code.
    """
    sys.argv = ["vibe_quant.data"] + (extra or [])
    from vibe_quant.data.ingest import main as data_main

    return data_main()


def cmd_screening(_args: argparse.Namespace, extra: list[str] | None = None) -> int:
    """Forward to screening module CLI.

    Args:
        _args: Parsed CLI arguments (unused, forwarding via extra).
        extra: Remaining arguments for submodule.

    Returns:
        Exit code.
    """
    sys.argv = ["vibe_quant.screening"] + (extra or [])
    from vibe_quant.screening.__main__ import main as screening_main

    return screening_main()


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
        choices=["colocated", "domestic", "international", "retail"],
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
        if args.command in ("data", "screening"):
            result: int = args.func(args, extra=extra)
        else:
            result = args.func(args)
        return result

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
