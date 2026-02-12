"""CLI entry point for Ethereal data ingestion.

Usage: python -m vibe_quant.ethereal ingest|status|catalog
"""

from __future__ import annotations

import sys


def main() -> int:
    """CLI entry point for Ethereal data ingestion."""
    import argparse
    from datetime import UTC, datetime

    from vibe_quant.ethereal.archive import _safe_years_ago
    from vibe_quant.ethereal.ingestion import (
        ETHEREAL_TIMEFRAMES,
        archive_to_catalog,
        get_ethereal_status,
        ingest_all_ethereal,
    )
    from vibe_quant.ethereal.instruments import get_ethereal_symbols

    parser = argparse.ArgumentParser(
        description="Ethereal exchange data ingestion",
        prog="python -m vibe_quant.ethereal",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Download and ingest data")
    ingest_parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(get_ethereal_symbols()),
        help="Comma-separated symbols to ingest (default: BTCUSD,ETHUSD,SOLUSD)",
    )
    ingest_parser.add_argument(
        "--timeframes",
        type=str,
        default=",".join(ETHEREAL_TIMEFRAMES),
        help="Comma-separated timeframes to ingest (default: all)",
    )
    ingest_parser.add_argument(
        "--years", type=int, default=2, help="Years of history to download"
    )
    ingest_parser.add_argument(
        "--no-funding", action="store_true", help="Skip funding rate ingestion"
    )

    # Status command
    subparsers.add_parser("status", help="Show data status")

    # Catalog command
    catalog_parser = subparsers.add_parser(
        "catalog", help="Convert archive to ParquetDataCatalog"
    )
    catalog_parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols (default: all in archive)",
    )
    catalog_parser.add_argument(
        "--timeframes",
        type=str,
        default=",".join(ETHEREAL_TIMEFRAMES),
        help="Comma-separated timeframes (default: all)",
    )

    args = parser.parse_args()

    if args.command == "ingest":
        symbols = [s.strip() for s in args.symbols.split(",")]
        timeframes = [t.strip() for t in args.timeframes.split(",")]

        end_date = datetime.now(UTC)
        start_date = _safe_years_ago(end_date, args.years)

        ingest_all_ethereal(
            symbols=symbols,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            include_funding=not args.no_funding,
            verbose=True,
        )

    elif args.command == "status":
        status = get_ethereal_status()
        print("\nEthereal Data Status:")
        print("=" * 50)
        if not status["symbols"]:
            print("No data in archive")
        else:
            for symbol, info in status["symbols"].items():
                print(f"\n{symbol}:")
                if "archive" in info:
                    for key, value in info["archive"].items():
                        if isinstance(value, dict):
                            print(f"  {key}: {value['klines']} klines")
                            print(f"    Range: {value['start']} to {value['end']}")
                        else:
                            print(f"  {key}: {value}")

    elif args.command == "catalog":
        cat_symbols: list[str] | None = (
            [s.strip() for s in args.symbols.split(",")]
            if args.symbols
            else None
        )
        timeframes = [t.strip() for t in args.timeframes.split(",")]

        archive_to_catalog(
            symbols=cat_symbols,
            timeframes=timeframes,
            verbose=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
