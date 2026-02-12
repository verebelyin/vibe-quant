"""Data ingestion for Ethereal exchange historical data."""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from datetime import UTC, datetime  # noqa: TC003 - used at runtime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import BarAggregation, PriceType
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from vibe_quant.ethereal.archive import (
    DEFAULT_ETHEREAL_ARCHIVE_PATH,
    EtherealArchive,
    _safe_years_ago,
)
from vibe_quant.ethereal.instruments import (
    ETHEREAL_INSTRUMENT_CONFIGS,
    ETHEREAL_VENUE,
    create_ethereal_instrument,
    get_ethereal_symbols,
)
from vibe_quant.utils import generate_month_range

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Sequence

# Ethereal archive base URL
ETHEREAL_ARCHIVE_BASE = "https://archive.ethereal.trade"

# Supported timeframes
ETHEREAL_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

# Bar aggregation mapping
TIMEFRAME_TO_AGGREGATION: dict[str, tuple[int, BarAggregation]] = {
    "1m": (1, BarAggregation.MINUTE),
    "5m": (5, BarAggregation.MINUTE),
    "15m": (15, BarAggregation.MINUTE),
    "1h": (1, BarAggregation.HOUR),
    "4h": (4, BarAggregation.HOUR),
    "1d": (1, BarAggregation.DAY),
}

# Timeframe to minutes mapping
TIMEFRAME_TO_MINUTES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}

# Default catalog path
DEFAULT_CATALOG_PATH = Path("data/catalog")


def download_bars(
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    timeout: float = 60.0,
) -> list[tuple[Any, ...]]:
    """Download OHLCV bars from Ethereal archive.

    Args:
        symbol: Trading symbol (e.g., 'ETHUSD', 'BTCUSD', 'SOLUSD').
        timeframe: Timeframe (e.g., '1m', '1h', '1d').
        start_date: Start date (inclusive).
        end_date: End date (inclusive).
        timeout: Request timeout in seconds.

    Returns:
        List of kline tuples:
        (open_time, open, high, low, close, volume, close_time, quote_volume, trade_count)
    """
    if timeframe not in ETHEREAL_TIMEFRAMES:
        msg = f"Unsupported timeframe: {timeframe}. Supported: {ETHEREAL_TIMEFRAMES}"
        raise ValueError(msg)

    all_klines: list[tuple[Any, ...]] = []
    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)

    # Download by month chunks
    with httpx.Client(timeout=timeout) as client:
        for year, month in generate_month_range(start_date, end_date):
            # Format: ETHUSD-1m-2024-01.csv.zip
            filename = f"{symbol}-{timeframe}-{year}-{month:02d}.csv.zip"
            url = f"{ETHEREAL_ARCHIVE_BASE}/klines/{symbol}/{timeframe}/{filename}"

            try:
                response = client.get(url)
                if response.status_code == 404:
                    continue
                response.raise_for_status()

                # Extract CSV from ZIP
                with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                    csv_filename = filename.replace(".zip", "")
                    with zf.open(csv_filename) as f:
                        reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
                        for row in reader:
                            # Skip header if present
                            if row[0] == "open_time" or not row[0].isdigit():
                                continue
                            # Ethereal kline format:
                            # open_time,open,high,low,close,volume,close_time,quote_volume,trade_count
                            open_time = int(row[0])
                            # Filter by date range
                            if start_ms <= open_time <= end_ms:
                                kline = (
                                    open_time,  # open_time (ms)
                                    float(row[1]),  # open
                                    float(row[2]),  # high
                                    float(row[3]),  # low
                                    float(row[4]),  # close
                                    float(row[5]),  # volume
                                    int(row[6]),  # close_time (ms)
                                    float(row[7]) if len(row) > 7 else 0.0,
                                    int(row[8]) if len(row) > 8 else 0,
                                )
                                all_klines.append(kline)
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "HTTP %d downloading bars %s %s/%d-%02d: %s",
                    exc.response.status_code, symbol, timeframe, year, month, exc,
                )
                continue
            except Exception:
                logger.exception("Unexpected error downloading bars %s %s/%s-%02d", symbol, timeframe, year, month)
                continue

    # Sort by open_time
    all_klines.sort(key=lambda x: x[0])
    return all_klines


def download_funding_rates(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    timeout: float = 60.0,
) -> list[tuple[Any, ...]]:
    """Download funding rate history from Ethereal archive.

    Ethereal uses hourly funding intervals.

    Args:
        symbol: Trading symbol (e.g., 'ETHUSD').
        start_date: Start date (inclusive).
        end_date: End date (inclusive).
        timeout: Request timeout in seconds.

    Returns:
        List of (funding_time, funding_rate, mark_price) tuples.
    """
    all_rates: list[tuple[Any, ...]] = []
    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)

    # Download by month chunks
    with httpx.Client(timeout=timeout) as client:
        for year, month in generate_month_range(start_date, end_date):
            filename = f"{symbol}-funding-{year}-{month:02d}.csv.zip"
            url = f"{ETHEREAL_ARCHIVE_BASE}/funding/{symbol}/{filename}"

            try:
                response = client.get(url)
                if response.status_code == 404:
                    continue
                response.raise_for_status()

                with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                    csv_filename = filename.replace(".zip", "")
                    with zf.open(csv_filename) as f:
                        reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
                        for row in reader:
                            # Skip header
                            if row[0] == "funding_time" or not row[0].isdigit():
                                continue
                            # Ethereal funding format:
                            # funding_time,funding_rate,mark_price
                            funding_time = int(row[0])
                            # Filter by date range
                            if start_ms <= funding_time <= end_ms:
                                rate = (
                                    funding_time,  # funding_time (ms)
                                    float(row[1]),  # funding_rate
                                    float(row[2]) if len(row) > 2 else 0.0,
                                )
                                all_rates.append(rate)
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "HTTP %d downloading funding %s %d-%02d: %s",
                    exc.response.status_code, symbol, year, month, exc,
                )
                continue
            except Exception:
                logger.exception("Unexpected error downloading funding %s %s/%s-%02d", symbol, year, month)
                continue

    all_rates.sort(key=lambda x: x[0])
    return all_rates


def get_ethereal_bar_type(symbol: str, timeframe: str) -> BarType:
    """Get NautilusTrader BarType for Ethereal symbol and timeframe.

    Args:
        symbol: Trading symbol (e.g., 'ETHUSD').
        timeframe: Timeframe (e.g., '1m', '1h').

    Returns:
        NautilusTrader BarType.
    """
    instrument_id = InstrumentId(Symbol(f"{symbol}-PERP"), ETHEREAL_VENUE)
    step, aggregation = TIMEFRAME_TO_AGGREGATION[timeframe]

    bar_spec = BarSpecification(step, aggregation, PriceType.LAST)
    return BarType(instrument_id=instrument_id, bar_spec=bar_spec)


def klines_to_bars(
    klines: Sequence[sqlite3.Row],
    bar_type: BarType,
) -> list[Bar]:
    """Convert raw klines to NautilusTrader Bar objects.

    Args:
        klines: Sequence of kline rows from archive.
        bar_type: NautilusTrader bar type.

    Returns:
        List of Bar objects.
    """
    bars = []
    for k in klines:
        # Convert ms timestamp to ns
        ts_event = int(k["open_time"]) * 1_000_000
        ts_init = int(k["close_time"]) * 1_000_000

        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str(str(k["open"])),
            high=Price.from_str(str(k["high"])),
            low=Price.from_str(str(k["low"])),
            close=Price.from_str(str(k["close"])),
            volume=Quantity.from_str(str(k["volume"])),
            ts_event=ts_event,
            ts_init=ts_init,
        )
        bars.append(bar)

    return bars


def archive_to_catalog(
    archive_path: Path | None = None,
    catalog_path: Path | None = None,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, int]]:
    """Convert archived raw data to ParquetDataCatalog.

    Args:
        archive_path: Path to Ethereal archive database.
        catalog_path: Path to output catalog directory.
        symbols: Symbols to convert (default: all in archive).
        timeframes: Timeframes to convert (default: all supported).
        verbose: Print progress messages.

    Returns:
        Dict mapping symbol to timeframe bar counts.
    """
    archive = EtherealArchive(archive_path)
    catalog_path = catalog_path or DEFAULT_CATALOG_PATH
    catalog_path.mkdir(parents=True, exist_ok=True)
    catalog = ParquetDataCatalog(str(catalog_path))

    symbols = symbols or archive.get_symbols()
    timeframes = timeframes or ETHEREAL_TIMEFRAMES

    results: dict[str, dict[str, int]] = {}

    for symbol in symbols:
        if symbol not in ETHEREAL_INSTRUMENT_CONFIGS:
            if verbose:
                print(f"Skipping unsupported symbol: {symbol}")
            continue

        results[symbol] = {}

        # Write instrument using existing create_ethereal_instrument
        instrument = create_ethereal_instrument(symbol)
        catalog.write_data([instrument])
        if verbose:
            print(f"Wrote instrument: {instrument.id}")

        for timeframe in timeframes:
            klines = archive.get_klines(symbol, timeframe)
            if not klines:
                continue

            bar_type = get_ethereal_bar_type(symbol, timeframe)
            bars = klines_to_bars(klines, bar_type)
            if bars:
                catalog.write_data(bars)
                results[symbol][timeframe] = len(bars)
                if verbose:
                    print(f"  {symbol} {timeframe}: {len(bars)} bars")

    archive.close()
    return results


def ingest_ethereal(
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    archive: EtherealArchive | None = None,
    verbose: bool = True,
) -> dict[str, int]:
    """Ingest Ethereal data: download and archive.

    Args:
        symbol: Trading symbol (e.g., 'ETHUSD').
        timeframe: Timeframe.
        start_date: Start date.
        end_date: End date.
        archive: Archive instance (created if not provided).
        verbose: Print progress.

    Returns:
        Counts dict with 'klines' key.
    """
    close_archive = archive is None
    if archive is None:
        archive = EtherealArchive()

    counts: dict[str, int] = {"klines": 0}

    if verbose:
        print(f"Downloading {symbol} {timeframe} from {start_date.date()} to {end_date.date()}...")

    klines = download_bars(symbol, timeframe, start_date, end_date)
    if klines:
        archive.insert_klines(symbol, timeframe, klines, "ethereal_archive")
        counts["klines"] = len(klines)
        if verbose:
            print(f"  Archived {len(klines)} klines")
    else:
        if verbose:
            print("  No klines downloaded")

    if close_archive:
        archive.close()

    return counts


def ingest_ethereal_funding(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    archive: EtherealArchive | None = None,
    verbose: bool = True,
) -> int:
    """Ingest Ethereal funding rates.

    Args:
        symbol: Trading symbol.
        start_date: Start date.
        end_date: End date.
        archive: Archive instance.
        verbose: Print progress.

    Returns:
        Number of funding rates archived.
    """
    close_archive = archive is None
    if archive is None:
        archive = EtherealArchive()

    if verbose:
        print(f"Downloading {symbol} funding rates...")

    rates = download_funding_rates(symbol, start_date, end_date)
    count = 0
    if rates:
        archive.insert_funding_rates(symbol, rates, "ethereal_archive")
        count = len(rates)
        if verbose:
            print(f"  Archived {count} funding rates")
    else:
        if verbose:
            print("  No funding rates downloaded")

    if close_archive:
        archive.close()

    return count


def ingest_all_ethereal(
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    include_funding: bool = True,
    verbose: bool = True,
) -> dict[str, dict[str, int]]:
    """Ingest data for multiple Ethereal symbols.

    Args:
        symbols: Symbols to ingest (default: all Ethereal symbols).
        timeframes: Timeframes to ingest (default: all supported).
        start_date: Start date (default: 2 years ago).
        end_date: End date (default: now).
        include_funding: Also download funding rates.
        verbose: Print progress.

    Returns:
        Dict mapping symbol to counts dict.
    """
    symbols = symbols or get_ethereal_symbols()
    timeframes = timeframes or ETHEREAL_TIMEFRAMES

    if end_date is None:
        end_date = datetime.now(UTC)
    if start_date is None:
        start_date = _safe_years_ago(end_date, 2)

    archive = EtherealArchive()
    results: dict[str, dict[str, int]] = {}

    for symbol in symbols:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Processing {symbol}")
            print(f"{'='*50}")

        counts: dict[str, int] = {}

        for timeframe in timeframes:
            tf_counts = ingest_ethereal(
                symbol, timeframe, start_date, end_date, archive=archive, verbose=verbose
            )
            counts[f"klines_{timeframe}"] = tf_counts.get("klines", 0)

        if include_funding:
            funding_count = ingest_ethereal_funding(
                symbol, start_date, end_date, archive=archive, verbose=verbose
            )
            counts["funding_rates"] = funding_count

        results[symbol] = counts

    archive.close()

    if verbose:
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        for sym, cnts in results.items():
            print(f"{sym}: {cnts}")

    return results


def get_ethereal_status(
    archive_path: Path | None = None,
    catalog_path: Path | None = None,
) -> dict[str, Any]:
    """Get status of Ethereal data archive and catalog.

    Args:
        archive_path: Path to archive database.
        catalog_path: Path to catalog directory.

    Returns:
        Status dict with symbol info.
    """
    archive = EtherealArchive(archive_path or DEFAULT_ETHEREAL_ARCHIVE_PATH)
    catalog_path = catalog_path or DEFAULT_CATALOG_PATH

    status: dict[str, Any] = {"symbols": {}}

    for symbol in archive.get_symbols():
        sym_status: dict[str, Any] = {}

        # Archive info for each timeframe
        for timeframe in ETHEREAL_TIMEFRAMES:
            date_range = archive.get_date_range(symbol, timeframe)
            if date_range:
                start_dt = datetime.fromtimestamp(date_range[0] / 1000, tz=UTC)
                end_dt = datetime.fromtimestamp(date_range[1] / 1000, tz=UTC)
                kline_count = archive.get_kline_count(symbol, timeframe)
                if "archive" not in sym_status:
                    sym_status["archive"] = {}
                sym_status["archive"][timeframe] = {
                    "klines": kline_count,
                    "start": start_dt.isoformat(),
                    "end": end_dt.isoformat(),
                }

        # Funding rates info
        funding_rates = archive.get_funding_rates(symbol)
        if funding_rates:
            if "archive" not in sym_status:
                sym_status["archive"] = {}
            sym_status["archive"]["funding_rates"] = len(funding_rates)

        status["symbols"][symbol] = sym_status

    archive.close()
    return status
