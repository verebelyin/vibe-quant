"""Raw data archive for storing downloaded market data before processing."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA busy_timeout=5000;")
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
        return after - before

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
        return after - before

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

    def get_symbols(self) -> list[str]:
        """Get list of symbols in archive.

        Returns:
            List of unique symbols.
        """
        rows = self.conn.execute(
            "SELECT DISTINCT symbol FROM raw_klines ORDER BY symbol"
        )
        return [r[0] for r in rows]
