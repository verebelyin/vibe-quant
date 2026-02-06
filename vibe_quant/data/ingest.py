"""Data ingestion orchestration."""

from __future__ import annotations

import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 (used at runtime in rebuild_from_archive)
from typing import Any

from vibe_quant.data.archive import RawDataArchive
from vibe_quant.data.catalog import (
    DEFAULT_CATALOG_PATH,
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
    download_recent_klines,
    get_months_in_range,
    get_years_months_to_download,
)
from vibe_quant.data.verify import verify_symbol


def get_download_preview(
    symbols: list[str],
    start_date: datetime,
    end_date: datetime,
    archive: RawDataArchive | None = None,
) -> list[dict[str, Any]]:
    """Preview what months will be downloaded vs skipped.

    Args:
        symbols: Symbols to check.
        start_date: Start date.
        end_date: End date.
        archive: Raw data archive.

    Returns:
        List of dicts with keys: symbol, year, month, status, kline_count, expected.
    """
    import calendar

    if archive is None:
        archive = RawDataArchive()

    months = get_months_in_range(start_date, end_date)
    preview: list[dict[str, Any]] = []

    for symbol in symbols:
        for year, month in months:
            actual = archive.get_month_kline_count(symbol, "1m", year, month)
            _, days = calendar.monthrange(year, month)
            expected = days * 24 * 60

            if actual >= expected * 0.9:
                status = "Archived"
            elif actual > 0:
                status = "Partial - will re-download"
            else:
                status = "Will download"

            preview.append({
                "Symbol": symbol,
                "Month": f"{year}-{month:02d}",
                "Status": status,
                "Klines": actual,
                "Expected": expected,
            })

    return preview


def update_symbol(
    symbol: str,
    archive: RawDataArchive | None = None,
    catalog: CatalogManager | None = None,
    verbose: bool = True,
) -> dict[str, int]:
    """Update data for a symbol by fetching missing candles.

    Detects last timestamp in archive, fetches gap via REST API,
    archives raw data, then rebuilds catalog.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT').
        archive: Raw data archive (created if not provided).
        catalog: Catalog manager (created if not provided).
        verbose: Print progress messages.

    Returns:
        Dict with counts: {'new_klines': N, 'bars_1m': N, ...}
    """
    if archive is None:
        archive = RawDataArchive()
    if catalog is None:
        catalog = CatalogManager()

    counts: dict[str, int] = {"new_klines": 0}

    # Get last timestamp from archive
    date_range = archive.get_date_range(symbol, "1m")
    if date_range is None:
        if verbose:
            print(f"{symbol}: no existing data, use 'ingest' command first")
        return counts

    last_timestamp = date_range[1]  # max open_time in ms
    now_ms = int(datetime.now(UTC).timestamp() * 1000)

    # Add 60000 (1 minute) to start after last candle
    start_time = last_timestamp + 60000

    if start_time >= now_ms:
        if verbose:
            print(f"{symbol}: already up to date")
        return counts

    if verbose:
        last_dt = datetime.fromtimestamp(last_timestamp / 1000, tz=UTC)
        print(f"{symbol}: fetching from {last_dt.isoformat()} to now...")

    # Download new klines via REST API
    new_klines = download_recent_klines(symbol, "1m", start_time, now_ms)
    if new_klines:
        archive.insert_klines(symbol, "1m", new_klines, "binance_api")
        counts["new_klines"] = len(new_klines)
        if verbose:
            print(f"  Archived {len(new_klines)} new klines")
    else:
        if verbose:
            print("  No new klines available")
        return counts

    # Rebuild catalog from all archived klines
    instrument = create_instrument(symbol)
    catalog.write_instrument(instrument)

    all_klines = archive.get_klines(symbol, "1m")
    if not all_klines:
        return counts

    # Convert to 1m bars
    bar_type_1m = get_bar_type(symbol, "1m")
    bars_1m = klines_to_bars(all_klines, instrument.id, bar_type_1m)
    catalog.write_bars(bars_1m)
    counts["bars_1m"] = len(bars_1m)
    if verbose:
        print(f"  Wrote {len(bars_1m)} 1m bars to catalog")

    # Aggregate to higher timeframes
    for interval in ["5m", "15m", "1h", "4h"]:
        if "m" in interval:
            minutes = int(interval.replace("m", ""))
        else:
            minutes = int(interval.replace("h", "")) * 60

        bar_type = get_bar_type(symbol, interval)
        agg_bars = aggregate_bars(bars_1m, bar_type, minutes)
        catalog.write_bars(agg_bars)
        counts[f"bars_{interval}"] = len(agg_bars)
        if verbose:
            print(f"  Wrote {len(agg_bars)} {interval} bars")

    return counts


def update_all(
    symbols: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, int]]:
    """Update data for all symbols.

    Args:
        symbols: List of symbols to update. Uses SUPPORTED_SYMBOLS if not provided.
        verbose: Print progress messages.

    Returns:
        Dict mapping symbol to counts dict.
    """
    if symbols is None:
        symbols = SUPPORTED_SYMBOLS

    archive = RawDataArchive()
    catalog = CatalogManager()
    results: dict[str, dict[str, int]] = {}

    session_id = archive.create_download_session(
        symbols=symbols, source="binance_api",
    )
    total_new = 0

    try:
        for symbol in symbols:
            if verbose:
                print(f"\n{'='*50}")
                print(f"Updating {symbol}")
                print(f"{'='*50}")

            counts = update_symbol(
                symbol, archive=archive, catalog=catalog, verbose=verbose
            )
            results[symbol] = counts
            total_new += counts.get("new_klines", 0)

        archive.complete_download_session(
            session_id, klines_fetched=total_new, klines_inserted=total_new,
        )
    except Exception as e:
        archive.complete_download_session(
            session_id, status="failed", error_message=str(e),
        )
        raise

    archive.close()

    if verbose:
        print(f"\n{'='*50}")
        print("UPDATE SUMMARY")
        print(f"{'='*50}")
        for sym, cnts in results.items():
            print(f"{sym}: {cnts.get('new_klines', 0)} new klines")

    return results


def ingest_symbol(
    symbol: str,
    years: int = 2,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    archive: RawDataArchive | None = None,
    catalog: CatalogManager | None = None,
    verbose: bool = True,
) -> dict[str, int]:
    """Ingest historical data for a single symbol.

    Downloads data from Binance Vision, stores in archive, and writes to catalog.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT').
        years: Number of years of history (used if start_date not given).
        start_date: Explicit start date (overrides years).
        end_date: Explicit end date (defaults to now).
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

    if start_date is not None:
        effective_end = end_date or datetime.now(UTC)
        months = get_months_in_range(start_date, effective_end)
    else:
        months = get_years_months_to_download(years)
        effective_end = datetime.now(UTC)

    # Smart skip: check which months already have sufficient coverage
    skipped: list[tuple[int, int]] = []
    to_download: list[tuple[int, int]] = []
    for year, month in months:
        if archive.has_month_coverage(symbol, "1m", year, month):
            skipped.append((year, month))
        else:
            to_download.append((year, month))

    if verbose:
        print(f"Downloading {symbol}: {len(to_download)} months to download, "
              f"{len(skipped)} months skipped (already archived)")

    # Download and archive 1m klines (only missing/partial months)
    for year, month in to_download:
        klines = download_monthly_klines(symbol, "1m", year, month)
        if klines:
            archive.insert_klines(symbol, "1m", klines, "binance_vision")
            counts["klines"] += len(klines)
            if verbose:
                print(f"  {year}-{month:02d}: {len(klines)} klines")
        else:
            if verbose:
                print(f"  {year}-{month:02d}: no data available")

    counts["months_skipped"] = len(skipped)
    counts["months_downloaded"] = len(to_download)

    if verbose:
        print(f"Total klines archived from Vision: {counts['klines']}")

    # Fill current incomplete month via REST API
    if months:
        last_year, last_month = months[-1]
        # Start of month after last complete month
        if last_month == 12:
            rest_start = datetime(last_year + 1, 1, 1, tzinfo=UTC)
        else:
            rest_start = datetime(last_year, last_month + 1, 1, tzinfo=UTC)

        rest_start_ms = int(rest_start.timestamp() * 1000)
        rest_end_ms = int(effective_end.timestamp() * 1000)

        if rest_start_ms < rest_end_ms:
            if verbose:
                print(f"Fetching {rest_start.strftime('%Y-%m-%d')} to "
                      f"{effective_end.strftime('%Y-%m-%d')} via REST API...")
            recent = download_recent_klines(symbol, "1m", rest_start_ms, rest_end_ms)
            if recent:
                count = archive.insert_klines(symbol, "1m", recent, "binance_api")
                counts["klines"] += len(recent)
                if verbose:
                    print(f"  REST API: {len(recent)} klines (inserted {count})")

    if verbose:
        print(f"Total klines: {counts['klines']}")

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

    # Clear existing catalog data to avoid disjoint interval errors
    # (archive is source of truth; catalog rebuilt from full archive each time)
    for interval in ["1m", "5m", "15m", "1h", "4h"]:
        catalog.clear_bar_data(symbol, interval)

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
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    archive: RawDataArchive | None = None,
    verbose: bool = True,
) -> int:
    """Ingest funding rate history for a symbol.

    Args:
        symbol: Trading symbol.
        years: Number of years of history (used if start_date not given).
        start_date: Explicit start date (overrides years).
        end_date: Explicit end date (defaults to now).
        archive: Raw data archive.
        verbose: Print progress messages.

    Returns:
        Number of funding rates archived.
    """
    if archive is None:
        archive = RawDataArchive()

    now = datetime.now(UTC)
    if start_date is not None:
        start_time = int(start_date.timestamp() * 1000)
    else:
        start_time = int((now.timestamp() - years * 365 * 24 * 3600) * 1000)
    end_time = int((end_date or now).timestamp() * 1000)

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
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """Ingest data for all symbols.

    Args:
        symbols: List of symbols to ingest. Uses SUPPORTED_SYMBOLS if not provided.
        years: Number of years of history (used if start_date not given).
        start_date: Explicit start date (overrides years).
        end_date: Explicit end date (defaults to now).
        verbose: Print progress messages.

    Returns:
        Dict mapping symbol to counts dict.
    """
    if symbols is None:
        symbols = SUPPORTED_SYMBOLS

    archive = RawDataArchive()
    catalog = CatalogManager()
    results: dict[str, dict[str, Any]] = {}

    # Create audit session
    session_id = archive.create_download_session(
        symbols=symbols,
        source="mixed",
        start_date=start_date.isoformat() if start_date else None,
        end_date=end_date.isoformat() if end_date else None,
    )

    total_klines_fetched = 0
    total_klines_inserted = 0
    total_funding = 0

    try:
        for symbol in symbols:
            if verbose:
                print(f"\n{'='*50}")
                print(f"Processing {symbol}")
                print(f"{'='*50}")

            counts = ingest_symbol(
                symbol,
                years=years,
                start_date=start_date,
                end_date=end_date,
                archive=archive,
                catalog=catalog,
                verbose=verbose,
            )
            funding_count = ingest_funding_rates(
                symbol,
                years=years,
                start_date=start_date,
                end_date=end_date,
                archive=archive,
                verbose=verbose,
            )
            counts["funding_rates"] = funding_count
            results[symbol] = counts
            total_klines_fetched += counts.get("klines", 0)
            total_klines_inserted += counts.get("bars_1m", 0)
            total_funding += funding_count

        archive.complete_download_session(
            session_id,
            klines_fetched=total_klines_fetched,
            klines_inserted=total_klines_inserted,
            funding_rates_fetched=total_funding,
        )
    except Exception as e:
        archive.complete_download_session(
            session_id, status="failed", error_message=str(e),
        )
        raise

    archive.close()

    if verbose:
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        for sym, cnts in results.items():
            print(f"{sym}: {cnts}")

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


def rebuild_from_archive(
    catalog_path: Path | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, int]]:
    """Rebuild ParquetDataCatalog from raw SQLite archive.

    Deletes existing catalog and recreates it from archived klines.

    Args:
        catalog_path: Path to catalog directory. Uses default if not specified.
        verbose: Print progress messages.

    Returns:
        Dict mapping symbol to bar counts.
    """
    catalog_path = catalog_path or DEFAULT_CATALOG_PATH

    # Delete existing catalog
    if catalog_path.exists():
        if verbose:
            print(f"Deleting existing catalog at {catalog_path}")
        shutil.rmtree(catalog_path)

    archive = RawDataArchive()
    catalog = CatalogManager(catalog_path)
    results: dict[str, dict[str, int]] = {}

    symbols = archive.get_symbols()
    if not symbols:
        if verbose:
            print("No symbols in archive")
        archive.close()
        return results

    if verbose:
        print(f"Rebuilding catalog for {len(symbols)} symbols...")

    for symbol in symbols:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Processing {symbol}")
            print(f"{'='*50}")

        counts: dict[str, int] = {}

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
            results[symbol] = counts
            continue

        # Convert to 1m bars and write
        bar_type_1m = get_bar_type(symbol, "1m")
        bars_1m = klines_to_bars(all_klines, instrument.id, bar_type_1m)
        catalog.write_bars(bars_1m)
        counts["bars_1m"] = len(bars_1m)
        if verbose:
            print(f"Wrote {len(bars_1m)} 1m bars")

        # Aggregate to higher timeframes (same logic as ingest_symbol)
        for interval in ["5m", "15m", "1h", "4h"]:
            bar_type = get_bar_type(symbol, interval)

            # Convert interval to minutes
            if "m" in interval:
                minutes = int(interval.replace("m", ""))
            else:
                minutes = int(interval.replace("h", "")) * 60

            agg_bars = aggregate_bars(bars_1m, bar_type, minutes)
            catalog.write_bars(agg_bars)
            counts[f"bars_{interval}"] = len(agg_bars)
            if verbose:
                print(f"Wrote {len(agg_bars)} {interval} bars")

        results[symbol] = counts

    archive.close()

    if verbose:
        print(f"\n{'='*50}")
        print("REBUILD COMPLETE")
        print(f"{'='*50}")
        for sym, counts in results.items():
            print(f"{sym}: {counts}")

    return results


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
        "--years", type=int, default=2, help="Years of history (ignored if --start given)"
    )
    ingest_parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD (overrides --years)",
    )
    ingest_parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (defaults to today)",
    )

    # Status command
    subparsers.add_parser("status", help="Show data status")

    # Update command
    update_parser = subparsers.add_parser(
        "update", help="Update data with recent candles"
    )
    update_parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(SUPPORTED_SYMBOLS),
        help="Comma-separated symbols to update",
    )

    # Rebuild command
    rebuild_parser = subparsers.add_parser(
        "rebuild", help="Rebuild catalog from archive"
    )
    rebuild_parser.add_argument(
        "--from-archive",
        action="store_true",
        required=True,
        help="Rebuild catalog from raw SQLite archive (required)",
    )

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify data quality")
    verify_parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols to verify (default: all in archive)",
    )

    args = parser.parse_args()

    if args.command == "ingest":
        symbols = [s.strip() for s in args.symbols.split(",")]
        start_date = (
            datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
            if args.start
            else None
        )
        end_date = (
            datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)
            if args.end
            else None
        )
        ingest_all(
            symbols=symbols,
            years=args.years,
            start_date=start_date,
            end_date=end_date,
            verbose=True,
        )
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
    elif args.command == "update":
        symbols = [s.strip() for s in args.symbols.split(",")]
        update_all(symbols=symbols, verbose=True)
    elif args.command == "rebuild":
        rebuild_from_archive(verbose=True)
    elif args.command == "verify":
        archive = RawDataArchive()
        symbols = (
            [s.strip() for s in args.symbols.split(",")]
            if args.symbols
            else archive.get_symbols()
        )

        if not symbols:
            print("No symbols to verify")
            archive.close()
            return 0

        print("\nData Verification Report")
        print("=" * 50)

        has_issues = False
        for symbol in symbols:
            result = verify_symbol(archive, symbol)
            gaps = result["gaps"]
            ohlc_errors = result["ohlc_errors"]
            kline_count = result["kline_count"]

            print(f"\n{symbol}: {kline_count} klines")

            if gaps:
                has_issues = True
                print(f"  GAPS ({len(gaps)}):")
                for start_ts, _end_ts, gap_min in gaps[:5]:  # Show first 5
                    start_dt = datetime.fromtimestamp(start_ts / 1000, tz=UTC)
                    print(f"    {start_dt.isoformat()} - {gap_min} min gap")
                if len(gaps) > 5:
                    print(f"    ... and {len(gaps) - 5} more")
            else:
                print("  Gaps: None")

            if ohlc_errors:
                has_issues = True
                print(f"  OHLC ERRORS ({len(ohlc_errors)}):")
                for ts, msg in ohlc_errors[:5]:  # Show first 5
                    err_dt = datetime.fromtimestamp(ts / 1000, tz=UTC)
                    print(f"    {err_dt.isoformat()}: {msg}")
                if len(ohlc_errors) > 5:
                    print(f"    ... and {len(ohlc_errors) - 5} more")
            else:
                print("  OHLC: Valid")

        archive.close()

        print("\n" + "=" * 50)
        if has_issues:
            print("ISSUES FOUND")
            return 1
        print("ALL CHECKS PASSED")

    return 0


if __name__ == "__main__":
    sys.exit(main())
