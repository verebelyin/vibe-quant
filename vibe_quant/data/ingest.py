"""Data ingestion orchestration."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from vibe_quant.data.archive import RawDataArchive
from vibe_quant.data.catalog import (
    INTERVAL_TO_AGGREGATION,
    CatalogManager,
    aggregate_bars,
    create_instrument,
    get_bar_type,
    klines_to_bars,
)
from vibe_quant.data.downloader import (
    SUPPORTED_SYMBOLS,
    download_funding_rates,
    download_monthly_klines,
    get_years_months_to_download,
)

if TYPE_CHECKING:
    from pathlib import Path


def ingest_symbol(
    symbol: str,
    years: int = 2,
    archive: RawDataArchive | None = None,
    catalog: CatalogManager | None = None,
    verbose: bool = True,
) -> dict[str, int]:
    """Ingest historical data for a single symbol.

    Downloads data from Binance Vision, stores in archive, and writes to catalog.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT').
        years: Number of years of history to download.
        archive: Raw data archive (created if not provided).
        catalog: Catalog manager (created if not provided).
        verbose: Print progress messages.

    Returns:
        Dict with counts: {'klines': N, 'bars_1m': N, 'bars_5m': N, ...}
    """
    if archive is None:
        archive = RawDataArchive()
    if catalog is None:
        catalog = CatalogManager()

    counts: dict[str, int] = {"klines": 0}
    months = get_years_months_to_download(years)

    if verbose:
        print(f"Downloading {symbol} data for {len(months)} months...")

    # Download and archive 1m klines
    for year, month in months:
        klines = download_monthly_klines(symbol, "1m", year, month)
        if klines:
            archive.insert_klines(symbol, "1m", klines, "binance_vision")
            counts["klines"] += len(klines)
            if verbose:
                print(f"  {year}-{month:02d}: {len(klines)} klines")
        else:
            if verbose:
                print(f"  {year}-{month:02d}: no data available")

    if verbose:
        print(f"Total klines archived: {counts['klines']}")

    # Create and write instrument
    instrument = create_instrument(symbol)
    catalog.write_instrument(instrument)
    if verbose:
        print(f"Wrote instrument: {instrument.id}")

    # Get all klines from archive
    all_klines = archive.get_klines(symbol, "1m")
    if not all_klines:
        if verbose:
            print("No klines to process")
        return counts

    # Convert to 1m bars and write
    bar_type_1m = get_bar_type(symbol, "1m")
    bars_1m = klines_to_bars(all_klines, instrument.id, bar_type_1m)
    catalog.write_bars(bars_1m)
    counts["bars_1m"] = len(bars_1m)
    if verbose:
        print(f"Wrote {len(bars_1m)} 1m bars")

    # Aggregate to higher timeframes
    for interval in ["5m", "15m", "1h", "4h"]:
        step, aggregation = INTERVAL_TO_AGGREGATION[interval]
        bar_type = get_bar_type(symbol, interval)

        # Convert step to minutes
        if "m" in interval:
            minutes = int(interval.replace("m", ""))
        else:
            minutes = int(interval.replace("h", "")) * 60

        agg_bars = aggregate_bars(bars_1m, bar_type, minutes)
        catalog.write_bars(agg_bars)
        counts[f"bars_{interval}"] = len(agg_bars)
        if verbose:
            print(f"Wrote {len(agg_bars)} {interval} bars")

    return counts


def ingest_funding_rates(
    symbol: str,
    years: int = 2,
    archive: RawDataArchive | None = None,
    verbose: bool = True,
) -> int:
    """Ingest funding rate history for a symbol.

    Args:
        symbol: Trading symbol.
        years: Number of years of history.
        archive: Raw data archive.
        verbose: Print progress messages.

    Returns:
        Number of funding rates archived.
    """
    if archive is None:
        archive = RawDataArchive()

    now = datetime.now(UTC)
    start_time = int((now.timestamp() - years * 365 * 24 * 3600) * 1000)
    end_time = int(now.timestamp() * 1000)

    if verbose:
        print(f"Downloading {symbol} funding rates...")

    rates = download_funding_rates(symbol, start_time, end_time)
    if rates:
        archive.insert_funding_rates(symbol, rates, "binance_api")
        if verbose:
            print(f"Archived {len(rates)} funding rates")
        return len(rates)

    if verbose:
        print("No funding rates downloaded")
    return 0


def ingest_all(
    symbols: list[str] | None = None,
    years: int = 2,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """Ingest data for all symbols.

    Args:
        symbols: List of symbols to ingest. Uses SUPPORTED_SYMBOLS if not provided.
        years: Number of years of history.
        verbose: Print progress messages.

    Returns:
        Dict mapping symbol to counts dict.
    """
    if symbols is None:
        symbols = SUPPORTED_SYMBOLS

    archive = RawDataArchive()
    catalog = CatalogManager()
    results: dict[str, dict[str, Any]] = {}

    for symbol in symbols:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Processing {symbol}")
            print(f"{'='*50}")

        counts = ingest_symbol(
            symbol, years=years, archive=archive, catalog=catalog, verbose=verbose
        )
        funding_count = ingest_funding_rates(
            symbol, years=years, archive=archive, verbose=verbose
        )
        counts["funding_rates"] = funding_count
        results[symbol] = counts

    archive.close()

    if verbose:
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        for sym, counts in results.items():
            print(f"{sym}: {counts}")

    return results


def get_status(
    archive_path: Path | None = None,
    catalog_path: Path | None = None,
) -> dict[str, Any]:
    """Get status of data archive and catalog.

    Args:
        archive_path: Path to archive database.
        catalog_path: Path to catalog directory.

    Returns:
        Status dict with symbol info.
    """
    from vibe_quant.data.archive import DEFAULT_ARCHIVE_PATH
    from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH

    archive = RawDataArchive(archive_path or DEFAULT_ARCHIVE_PATH)
    catalog = CatalogManager(catalog_path or DEFAULT_CATALOG_PATH)

    status: dict[str, Any] = {"symbols": {}}

    for symbol in archive.get_symbols():
        sym_status: dict[str, Any] = {}

        # Archive info
        date_range = archive.get_date_range(symbol, "1m")
        if date_range:
            start_dt = datetime.fromtimestamp(date_range[0] / 1000, tz=UTC)
            end_dt = datetime.fromtimestamp(date_range[1] / 1000, tz=UTC)
            sym_status["archive"] = {
                "klines_1m": archive.get_kline_count(symbol, "1m"),
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
            }

        # Catalog info
        for interval in ["1m", "5m", "15m", "1h", "4h"]:
            bar_count = catalog.get_bar_count(symbol, interval)
            if bar_count > 0:
                if "catalog" not in sym_status:
                    sym_status["catalog"] = {}
                sym_status["catalog"][f"bars_{interval}"] = bar_count

        status["symbols"][symbol] = sym_status

    archive.close()
    return status


def main() -> int:
    """CLI entry point for data ingestion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="vibe-quant data ingestion",
        prog="python -m vibe_quant.data",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Download and ingest data")
    ingest_parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(SUPPORTED_SYMBOLS),
        help="Comma-separated symbols to ingest",
    )
    ingest_parser.add_argument(
        "--years", type=int, default=2, help="Years of history to download"
    )

    # Status command
    subparsers.add_parser("status", help="Show data status")

    args = parser.parse_args()

    if args.command == "ingest":
        symbols = [s.strip() for s in args.symbols.split(",")]
        ingest_all(symbols=symbols, years=args.years, verbose=True)
    elif args.command == "status":
        status = get_status()
        print("\nData Status:")
        print("=" * 50)
        for symbol, info in status["symbols"].items():
            print(f"\n{symbol}:")
            if "archive" in info:
                a = info["archive"]
                print(f"  Archive: {a['klines_1m']} klines")
                print(f"    Range: {a['start']} to {a['end']}")
            if "catalog" in info:
                for key, count in info["catalog"].items():
                    print(f"  Catalog {key}: {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
