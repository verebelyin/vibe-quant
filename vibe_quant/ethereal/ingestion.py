"""Data ingestion for Ethereal exchange historical data."""

from __future__ import annotations

import csv
import io
import logging
import sqlite3
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

logger = logging.getLogger(__name__)
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import BarAggregation, PriceType
from nautilus_trader.model.identifiers import InstrumentId, Symbol
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from vibe_quant.ethereal.instruments import (
    ETHEREAL_INSTRUMENT_CONFIGS,
    ETHEREAL_VENUE,
    create_ethereal_instrument,
    get_ethereal_symbols,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Sequence

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

# Default archive path for Ethereal
DEFAULT_ETHEREAL_ARCHIVE_PATH = Path("data/archive/ethereal_raw.db")

# Default catalog path
DEFAULT_CATALOG_PATH = Path("data/catalog")

# SQLite schema for Ethereal raw data
ETHEREAL_ARCHIVE_SCHEMA = """
-- Raw klines data from Ethereal
CREATE TABLE IF NOT EXISTS raw_klines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open_time INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    close_time INTEGER NOT NULL,
    quote_volume REAL,
    trade_count INTEGER,
    source TEXT NOT NULL,
    downloaded_at TEXT DEFAULT (datetime('now')),
    UNIQUE(symbol, timeframe, open_time)
);

-- Raw funding rates from Ethereal (hourly intervals)
CREATE TABLE IF NOT EXISTS raw_funding_rates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    funding_time INTEGER NOT NULL,
    funding_rate REAL NOT NULL,
    mark_price REAL,
    source TEXT NOT NULL,
    downloaded_at TEXT DEFAULT (datetime('now')),
    UNIQUE(symbol, funding_time)
);

CREATE INDEX IF NOT EXISTS idx_ethereal_klines_symbol_time
    ON raw_klines(symbol, timeframe, open_time);
CREATE INDEX IF NOT EXISTS idx_ethereal_funding_symbol_time
    ON raw_funding_rates(symbol, funding_time);
"""


def _safe_years_ago(d: datetime, years: int) -> datetime:
    """Subtract years from date, handling leap day.

    Args:
        d: Date to subtract from.
        years: Number of years to subtract.

    Returns:
        Date ``years`` years before ``d``. Feb 29 falls back to Feb 28.
    """
    try:
        return d.replace(year=d.year - years)
    except ValueError:
        # Feb 29 on non-leap target year -> Feb 28
        return d.replace(year=d.year - years, day=28)


class EtherealArchive:
    """SQLite archive for raw Ethereal market data."""

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize archive.

        Args:
            db_path: Path to archive database. Uses default if not specified.
        """
        self._db_path = db_path or DEFAULT_ETHEREAL_ARCHIVE_PATH
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA busy_timeout=5000;")
            self._conn.executescript(ETHEREAL_ARCHIVE_SCHEMA)
            self._conn.commit()
        return self._conn

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def insert_klines(
        self,
        symbol: str,
        timeframe: str,
        klines: Iterable[tuple[Any, ...]],
        source: str,
    ) -> int:
        """Insert kline data into archive.

        Args:
            symbol: Trading symbol (e.g., 'ETHUSD').
            timeframe: Timeframe (e.g., '1m', '1h').
            klines: Iterable of kline tuples:
                (open_time, open, high, low, close, volume, close_time,
                 quote_volume, trade_count)
            source: Data source identifier.

        Returns:
            Number of rows inserted.
        """
        rows = [
            (
                symbol,
                timeframe,
                k[0],  # open_time
                k[1],  # open
                k[2],  # high
                k[3],  # low
                k[4],  # close
                k[5],  # volume
                k[6],  # close_time
                k[7] if len(k) > 7 else None,  # quote_volume
                k[8] if len(k) > 8 else None,  # trade_count
                source,
            )
            for k in klines
        ]

        before = self.conn.execute(
            "SELECT COUNT(*) FROM raw_klines WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe),
        ).fetchone()[0]

        self.conn.executemany(
            """INSERT OR IGNORE INTO raw_klines
               (symbol, timeframe, open_time, open, high, low, close, volume,
                close_time, quote_volume, trade_count, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self.conn.commit()

        after = self.conn.execute(
            "SELECT COUNT(*) FROM raw_klines WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe),
        ).fetchone()[0]
        return after - before

    def insert_funding_rates(
        self,
        symbol: str,
        rates: Iterable[tuple[Any, ...]],
        source: str,
    ) -> int:
        """Insert funding rate data into archive.

        Args:
            symbol: Trading symbol.
            rates: Iterable of (funding_time, funding_rate, mark_price) tuples.
            source: Data source identifier.

        Returns:
            Number of rows inserted.
        """
        rows = [
            (symbol, r[0], r[1], r[2] if len(r) > 2 else None, source) for r in rates
        ]

        before = self.conn.execute(
            "SELECT COUNT(*) FROM raw_funding_rates WHERE symbol = ?",
            (symbol,),
        ).fetchone()[0]

        self.conn.executemany(
            """INSERT OR IGNORE INTO raw_funding_rates
               (symbol, funding_time, funding_rate, mark_price, source)
               VALUES (?, ?, ?, ?, ?)""",
            rows,
        )
        self.conn.commit()

        after = self.conn.execute(
            "SELECT COUNT(*) FROM raw_funding_rates WHERE symbol = ?",
            (symbol,),
        ).fetchone()[0]
        return after - before

    def get_klines(
        self,
        symbol: str,
        timeframe: str,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[sqlite3.Row]:
        """Get klines from archive.

        Args:
            symbol: Trading symbol.
            timeframe: Timeframe.
            start_time: Start timestamp (ms), inclusive.
            end_time: End timestamp (ms), inclusive.

        Returns:
            List of kline rows.
        """
        query = "SELECT * FROM raw_klines WHERE symbol = ? AND timeframe = ?"
        params: list[Any] = [symbol, timeframe]

        if start_time is not None:
            query += " AND open_time >= ?"
            params.append(start_time)
        if end_time is not None:
            query += " AND open_time <= ?"
            params.append(end_time)

        query += " ORDER BY open_time"
        return list(self.conn.execute(query, params))

    def get_funding_rates(
        self,
        symbol: str,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[sqlite3.Row]:
        """Get funding rates from archive.

        Args:
            symbol: Trading symbol.
            start_time: Start timestamp (ms), inclusive.
            end_time: End timestamp (ms), inclusive.

        Returns:
            List of funding rate rows.
        """
        query = "SELECT * FROM raw_funding_rates WHERE symbol = ?"
        params: list[Any] = [symbol]

        if start_time is not None:
            query += " AND funding_time >= ?"
            params.append(start_time)
        if end_time is not None:
            query += " AND funding_time <= ?"
            params.append(end_time)

        query += " ORDER BY funding_time"
        return list(self.conn.execute(query, params))

    def get_date_range(
        self, symbol: str, timeframe: str
    ) -> tuple[int, int] | None:
        """Get the date range of stored klines.

        Args:
            symbol: Trading symbol.
            timeframe: Timeframe.

        Returns:
            (min_open_time, max_open_time) tuple or None if no data.
        """
        row = self.conn.execute(
            """SELECT MIN(open_time), MAX(open_time) FROM raw_klines
               WHERE symbol = ? AND timeframe = ?""",
            (symbol, timeframe),
        ).fetchone()

        if row and row[0] is not None:
            return (row[0], row[1])
        return None

    def get_kline_count(self, symbol: str, timeframe: str) -> int:
        """Get count of stored klines.

        Args:
            symbol: Trading symbol.
            timeframe: Timeframe.

        Returns:
            Number of klines stored.
        """
        row = self.conn.execute(
            "SELECT COUNT(*) FROM raw_klines WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe),
        ).fetchone()
        return row[0] if row else 0

    def get_symbols(self) -> list[str]:
        """Get list of symbols in archive.

        Returns:
            List of unique symbols.
        """
        rows = self.conn.execute(
            "SELECT DISTINCT symbol FROM raw_klines ORDER BY symbol"
        )
        return [r[0] for r in rows]


def generate_month_range(
    start_date: datetime, end_date: datetime
) -> Generator[tuple[int, int]]:
    """Generate (year, month) tuples between two dates.

    Args:
        start_date: Start date (inclusive).
        end_date: End date (inclusive).

    Yields:
        (year, month) tuples.
    """
    current = start_date.replace(day=1)
    while current <= end_date:
        yield (current.year, current.month)
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)


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
            except httpx.HTTPStatusError:
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
    for year, month in generate_month_range(start_date, end_date):
        filename = f"{symbol}-funding-{year}-{month:02d}.csv.zip"
        url = f"{ETHEREAL_ARCHIVE_BASE}/funding/{symbol}/{filename}"

        try:
            with httpx.Client(timeout=timeout) as client:
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
        except httpx.HTTPStatusError:
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


def main() -> int:
    """CLI entry point for Ethereal data ingestion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ethereal exchange data ingestion",
        prog="python -m vibe_quant.ethereal.ingestion",
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
    import sys
    sys.exit(main())
