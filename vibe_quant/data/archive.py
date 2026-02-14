"""Raw data archive for storing downloaded market data before processing."""

from __future__ import annotations

import calendar
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterable

DEFAULT_ARCHIVE_PATH = Path("data/archive/raw_data.db")

ARCHIVE_SCHEMA = """
-- Raw klines data from Binance
CREATE TABLE IF NOT EXISTS raw_klines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    open_time INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    close_time INTEGER NOT NULL,
    quote_volume REAL,
    trade_count INTEGER,
    taker_buy_volume REAL,
    taker_buy_quote_volume REAL,
    source TEXT NOT NULL,
    downloaded_at TEXT DEFAULT (datetime('now')),
    UNIQUE(symbol, interval, open_time)
);

-- Raw funding rates from Binance
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

CREATE INDEX IF NOT EXISTS idx_raw_klines_symbol_time
    ON raw_klines(symbol, interval, open_time);
CREATE INDEX IF NOT EXISTS idx_raw_funding_symbol_time
    ON raw_funding_rates(symbol, funding_time);

-- Download session audit log
CREATE TABLE IF NOT EXISTS download_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    symbols TEXT NOT NULL,
    start_date TEXT,
    end_date TEXT,
    source TEXT NOT NULL,
    klines_fetched INTEGER DEFAULT 0,
    klines_inserted INTEGER DEFAULT 0,
    funding_rates_fetched INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running',
    error_message TEXT
);
"""


class RawDataArchive:
    """SQLite archive for raw downloaded market data."""

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize archive.

        Args:
            db_path: Path to archive database. Uses default if not specified.
        """
        self._db_path = db_path or DEFAULT_ARCHIVE_PATH
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            from vibe_quant.db.connection import get_connection

            self._conn = get_connection(self._db_path)
            self._conn.executescript(ARCHIVE_SCHEMA)
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
        interval: str,
        klines: Iterable[tuple[Any, ...]],
        source: str,
    ) -> int:
        """Insert kline data into archive.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT').
            interval: Candle interval (e.g., '1m').
            klines: Iterable of kline tuples:
                (open_time, open, high, low, close, volume, close_time,
                 quote_volume, trade_count, taker_buy_volume, taker_buy_quote_volume)
            source: Data source identifier (e.g., 'binance_vision').

        Returns:
            Number of rows inserted.
        """
        rows = [
            (
                symbol,
                interval,
                k[0],  # open_time
                k[1],  # open
                k[2],  # high
                k[3],  # low
                k[4],  # close
                k[5],  # volume
                k[6],  # close_time
                k[7] if len(k) > 7 else None,  # quote_volume
                k[8] if len(k) > 8 else None,  # trade_count
                k[9] if len(k) > 9 else None,  # taker_buy_volume
                k[10] if len(k) > 10 else None,  # taker_buy_quote_volume
                source,
            )
            for k in klines
        ]

        before = self.conn.execute(
            "SELECT COUNT(*) FROM raw_klines WHERE symbol = ? AND interval = ?",
            (symbol, interval),
        ).fetchone()[0]

        self.conn.executemany(
            """INSERT OR IGNORE INTO raw_klines
               (symbol, interval, open_time, open, high, low, close, volume,
                close_time, quote_volume, trade_count, taker_buy_volume,
                taker_buy_quote_volume, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self.conn.commit()

        after = self.conn.execute(
            "SELECT COUNT(*) FROM raw_klines WHERE symbol = ? AND interval = ?",
            (symbol, interval),
        ).fetchone()[0]
        return int(after - before)

    def insert_funding_rates(
        self,
        symbol: str,
        rates: Iterable[tuple[Any, ...]],
        source: str,
    ) -> int:
        """Insert funding rate data into archive.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT').
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
        return int(after - before)

    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[sqlite3.Row]:
        """Get klines from archive.

        Args:
            symbol: Trading symbol.
            interval: Candle interval.
            start_time: Start timestamp (ms), inclusive.
            end_time: End timestamp (ms), inclusive.

        Returns:
            List of kline rows.
        """
        query = "SELECT * FROM raw_klines WHERE symbol = ? AND interval = ?"
        params: list[Any] = [symbol, interval]

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

    def get_date_range(self, symbol: str, interval: str) -> tuple[int, int] | None:
        """Get the date range of stored klines.

        Args:
            symbol: Trading symbol.
            interval: Candle interval.

        Returns:
            (min_open_time, max_open_time) tuple or None if no data.
        """
        row = self.conn.execute(
            """SELECT MIN(open_time), MAX(open_time) FROM raw_klines
               WHERE symbol = ? AND interval = ?""",
            (symbol, interval),
        ).fetchone()

        if row and row[0] is not None:
            return (row[0], row[1])
        return None

    def get_kline_count(self, symbol: str, interval: str) -> int:
        """Get count of stored klines.

        Args:
            symbol: Trading symbol.
            interval: Candle interval.

        Returns:
            Number of klines stored.
        """
        row = self.conn.execute(
            "SELECT COUNT(*) FROM raw_klines WHERE symbol = ? AND interval = ?",
            (symbol, interval),
        ).fetchone()
        return row[0] if row else 0

    def get_month_kline_count(
        self, symbol: str, interval: str, year: int, month: int,
    ) -> int:
        """Get kline count for a specific month.

        Args:
            symbol: Trading symbol.
            interval: Candle interval.
            year: Year.
            month: Month (1-12).

        Returns:
            Number of klines in that month.
        """
        # First ms of month
        start_ms = int(
            datetime(year, month, 1, tzinfo=UTC).timestamp() * 1000
        )
        if month == 12:
            end_ms = int(
                datetime(year + 1, 1, 1, tzinfo=UTC).timestamp() * 1000
            )
        else:
            end_ms = int(
                datetime(year, month + 1, 1, tzinfo=UTC).timestamp() * 1000
            )

        row = self.conn.execute(
            """SELECT COUNT(*) FROM raw_klines
               WHERE symbol = ? AND interval = ?
               AND open_time >= ? AND open_time < ?""",
            (symbol, interval, start_ms, end_ms),
        ).fetchone()
        return row[0] if row else 0

    def has_month_coverage(
        self, symbol: str, interval: str, year: int, month: int,
        threshold: float = 0.9,
    ) -> bool:
        """Check if a month has sufficient kline coverage.

        A month is considered covered if it has >= threshold of expected klines.
        For 1m interval, a 30-day month has ~43200 expected klines.

        Args:
            symbol: Trading symbol.
            interval: Candle interval.
            year: Year.
            month: Month (1-12).
            threshold: Coverage ratio threshold (0-1). Default 0.9.

        Returns:
            True if month is sufficiently covered.
        """
        _, days_in_month = calendar.monthrange(year, month)

        # Expected klines per interval
        if interval == "1m":
            expected = days_in_month * 24 * 60
        elif interval == "5m":
            expected = days_in_month * 24 * 12
        elif interval == "15m":
            expected = days_in_month * 24 * 4
        elif interval == "1h":
            expected = days_in_month * 24
        elif interval == "4h":
            expected = days_in_month * 6
        else:
            return False

        actual = self.get_month_kline_count(symbol, interval, year, month)
        return actual >= expected * threshold

    def get_symbols(self) -> list[str]:
        """Get list of symbols in archive.

        Returns:
            List of unique symbols.
        """
        rows = self.conn.execute(
            "SELECT DISTINCT symbol FROM raw_klines ORDER BY symbol"
        )
        return [r[0] for r in rows]

    # -- Audit log methods --

    def create_download_session(
        self,
        symbols: list[str],
        source: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> int:
        """Create a new download session audit record.

        Args:
            symbols: List of symbols being downloaded.
            source: Data source ('binance_vision', 'binance_api', 'mixed').
            start_date: Requested start date (ISO format).
            end_date: Requested end date (ISO format).

        Returns:
            Session ID.
        """
        cursor = self.conn.execute(
            """INSERT INTO download_sessions (symbols, source, start_date, end_date)
               VALUES (?, ?, ?, ?)""",
            (",".join(symbols), source, start_date, end_date),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def complete_download_session(
        self,
        session_id: int,
        klines_fetched: int = 0,
        klines_inserted: int = 0,
        funding_rates_fetched: int = 0,
        status: str = "completed",
        error_message: str | None = None,
    ) -> None:
        """Mark a download session as completed.

        Args:
            session_id: Session ID from create_download_session.
            klines_fetched: Total klines downloaded.
            klines_inserted: New klines inserted (excludes duplicates).
            funding_rates_fetched: Total funding rates downloaded.
            status: Final status ('completed' or 'failed').
            error_message: Error message if failed.
        """
        self.conn.execute(
            """UPDATE download_sessions
               SET completed_at = datetime('now'),
                   klines_fetched = ?,
                   klines_inserted = ?,
                   funding_rates_fetched = ?,
                   status = ?,
                   error_message = ?
               WHERE id = ?""",
            (klines_fetched, klines_inserted, funding_rates_fetched,
             status, error_message, session_id),
        )
        self.conn.commit()

    def get_download_sessions(self, limit: int = 50) -> list[sqlite3.Row]:
        """Get recent download sessions.

        Args:
            limit: Max number of sessions to return.

        Returns:
            List of session rows, newest first.
        """
        return list(self.conn.execute(
            "SELECT * FROM download_sessions ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ))
