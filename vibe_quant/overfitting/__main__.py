"""CLI for overfitting prevention pipeline.

Usage:
    python -m vibe_quant.overfitting run --run-id 42
    python -m vibe_quant.overfitting run --run-id 42 --filters dsr,wfa
    python -m vibe_quant.overfitting run --run-id 42 --filters dsr --observations 500
    python -m vibe_quant.overfitting report --run-id 42
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from vibe_quant.overfitting.pipeline import OverfittingPipeline
from vibe_quant.overfitting.types import FilterConfig


def parse_filters(filters_str: str) -> FilterConfig:
    """Parse comma-separated filter names into FilterConfig.

    Args:
        filters_str: Comma-separated filter names (dsr, wfa, pkfold/cv).
            Empty string or "all" enables all filters.

    Returns:
        FilterConfig with specified filters enabled.
    """
    if not filters_str or filters_str.lower() == "all":
        return FilterConfig.default()

    filters = [f.strip().lower() for f in filters_str.split(",")]

    enable_dsr = "dsr" in filters
    enable_wfa = "wfa" in filters
    enable_cv = "pkfold" in filters or "cv" in filters

    return FilterConfig(
        enable_dsr=enable_dsr,
        enable_wfa=enable_wfa,
        enable_purged_kfold=enable_cv,
    )


def cmd_run(args: argparse.Namespace) -> int:
    """Run overfitting pipeline on sweep results.

    Args:
        args: Parsed command arguments.

    Returns:
        Exit code (0 = success).
    """
    # Parse filter config
    config = parse_filters(args.filters)

    # Parse dates if provided
    data_start = None
    data_end = None
    if args.start_date:
        data_start = date.fromisoformat(args.start_date)
    if args.end_date:
        data_end = date.fromisoformat(args.end_date)

    # Create pipeline
    db_path = Path(args.db) if args.db else None
    pipeline = OverfittingPipeline(db_path)

    try:
        # Run pipeline
        print(f"Running overfitting pipeline on run_id={args.run_id}...")
        print(f"  Filters: DSR={config.enable_dsr}, WFA={config.enable_wfa}, CV={config.enable_purged_kfold}")
        print()

        result = pipeline.run(
            run_id=args.run_id,
            config=config,
            num_observations=args.observations,
            data_start=data_start,
            data_end=data_end,
            n_samples=args.samples,
        )

        # Print summary
        print("=" * 60)
        print("OVERFITTING PIPELINE SUMMARY")
        print("=" * 60)
        print()

        # Flow summary: N in -> N passed DSR -> N passed WFA -> N passed CV -> N final
        summary_parts = [f"{result.total_candidates} candidates"]

        if config.enable_dsr:
            summary_parts.append(f"{result.passed_dsr} passed DSR")
        if config.enable_wfa:
            summary_parts.append(f"{result.passed_wfa} passed WFA")
        if config.enable_purged_kfold:
            summary_parts.append(f"{result.passed_cv} passed PKFOLD")

        summary_parts.append(f"{result.passed_all} final")

        print(" -> ".join(summary_parts))
        print()

        # Detailed stats
        print("-" * 60)
        print("FILTER STATISTICS")
        print("-" * 60)

        if config.enable_dsr:
            pct = (result.passed_dsr / result.total_candidates * 100) if result.total_candidates else 0
            print(f"  DSR (Deflated Sharpe):      {result.passed_dsr:4d} / {result.total_candidates:4d}  ({pct:5.1f}%)")

        if config.enable_wfa:
            pct = (result.passed_wfa / result.total_candidates * 100) if result.total_candidates else 0
            print(f"  WFA (Walk-Forward):         {result.passed_wfa:4d} / {result.total_candidates:4d}  ({pct:5.1f}%)")

        if config.enable_purged_kfold:
            pct = (result.passed_cv / result.total_candidates * 100) if result.total_candidates else 0
            print(f"  PKFOLD (Purged K-Fold CV):  {result.passed_cv:4d} / {result.total_candidates:4d}  ({pct:5.1f}%)")

        print()
        pct = (result.passed_all / result.total_candidates * 100) if result.total_candidates else 0
        print(f"  ALL FILTERS:                {result.passed_all:4d} / {result.total_candidates:4d}  ({pct:5.1f}%)")
        print()

        # Top candidates
        if result.filtered_candidates:
            print("-" * 60)
            print("TOP FILTERED CANDIDATES")
            print("-" * 60)
            print()
            print(f"{'#':>3}  {'Sharpe':>8}  {'Return':>8}  {'Parameters':<40}")
            print("-" * 60)

            for i, c in enumerate(result.filtered_candidates[:10]):
                params_short = c.parameters[:37] + "..." if len(c.parameters) > 40 else c.parameters
                print(f"{i+1:3d}  {c.sharpe_ratio:8.3f}  {c.total_return:7.2f}%  {params_short:<40}")

            if len(result.filtered_candidates) > 10:
                print(f"  ... and {len(result.filtered_candidates) - 10} more")

        else:
            print("-" * 60)
            print("NO CANDIDATES PASSED ALL FILTERS")
            print("-" * 60)
            print()
            print("Consider relaxing filter thresholds or increasing data range.")

        print()

        # Write detailed report if requested
        if args.output:
            report = pipeline.generate_report(result)
            Path(args.output).write_text(report)
            print(f"Detailed report written to: {args.output}")

        return 0

    finally:
        pipeline.close()


def cmd_report(args: argparse.Namespace) -> int:
    """Generate report for existing pipeline results.

    Args:
        args: Parsed command arguments.

    Returns:
        Exit code (0 = success).
    """
    db_path = Path(args.db) if args.db else None
    pipeline = OverfittingPipeline(db_path)

    try:
        # Get filtered candidates from database
        candidates = pipeline.get_filtered_candidates(
            args.run_id,
            require_all=not args.any_filter,
        )

        print("=" * 60)
        print(f"FILTERED CANDIDATES FOR RUN {args.run_id}")
        print("=" * 60)
        print()

        if not candidates:
            print("No candidates found matching criteria.")
            return 0

        print(f"Found {len(candidates)} candidates")
        print()

        print(f"{'#':>3}  {'ID':>5}  {'Sharpe':>8}  {'Return':>8}  {'DSR':>4}  {'WFA':>4}  {'CV':>4}")
        print("-" * 60)

        for i, c in enumerate(candidates[:20]):
            dsr = "Y" if c.get("passed_deflated_sharpe") == 1 else ("N" if c.get("passed_deflated_sharpe") == 0 else "-")
            wfa = "Y" if c.get("passed_walk_forward") == 1 else ("N" if c.get("passed_walk_forward") == 0 else "-")
            cv = "Y" if c.get("passed_purged_kfold") == 1 else ("N" if c.get("passed_purged_kfold") == 0 else "-")

            print(
                f"{i+1:3d}  {c['id']:5d}  {c.get('sharpe_ratio', 0):8.3f}  "
                f"{c.get('total_return', 0):7.2f}%  {dsr:>4}  {wfa:>4}  {cv:>4}"
            )

        if len(candidates) > 20:
            print(f"  ... and {len(candidates) - 20} more")

        print()
        return 0

    finally:
        pipeline.close()


def main() -> int:
    """Main entry point for overfitting CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m vibe_quant.overfitting",
        description="Overfitting prevention pipeline CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run overfitting filters")
    run_parser.add_argument(
        "--run-id",
        type=int,
        required=True,
        help="Backtest run ID to filter",
    )
    run_parser.add_argument(
        "--filters",
        type=str,
        default="all",
        help="Comma-separated filters: dsr,wfa,pkfold (default: all)",
    )
    run_parser.add_argument(
        "--db",
        type=str,
        help="Database path (default: data/state.db)",
    )
    run_parser.add_argument(
        "--observations",
        type=int,
        default=252,
        help="Number of observations for DSR (default: 252)",
    )
    run_parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for WFA (YYYY-MM-DD)",
    )
    run_parser.add_argument(
        "--end-date",
        type=str,
        help="End date for WFA (YYYY-MM-DD)",
    )
    run_parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples for Purged K-Fold (default: 1000)",
    )
    run_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file for detailed report",
    )

    # Report command
    report_parser = subparsers.add_parser("report", help="View filtered candidates")
    report_parser.add_argument(
        "--run-id",
        type=int,
        required=True,
        help="Backtest run ID",
    )
    report_parser.add_argument(
        "--db",
        type=str,
        help="Database path (default: data/state.db)",
    )
    report_parser.add_argument(
        "--any-filter",
        action="store_true",
        help="Show candidates passing ANY filter (default: ALL)",
    )

    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "report":
        return cmd_report(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
