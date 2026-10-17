"""Raw data archive for storing downloaded market data before processing."""

from __future__ import annotations

import calendar
from datetime import UTC, date, datetime, timedelta
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

-- Raw MARK-price klines from Binance (liquidation trigger price). Only OHLC
-- and the two timestamps: mark-price volume/taker columns are always 0.
CREATE TABLE IF NOT EXISTS raw_mark_klines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    open_time INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    close_time INTEGER NOT NULL,
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

-- Binance Vision funding intervals: the settlement interval (hours) actually
-- in effect at each archived funding_time. Append-only and joined to
-- raw_funding_rates on ``funding_time``; a row with no entry here inherits the
-- previous row's interval (see FundingCalculator._load_intervals).
CREATE TABLE IF NOT EXISTS raw_funding_intervals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    funding_time INTEGER NOT NULL,
    interval_hours INTEGER NOT NULL,
    source TEXT NOT NULL,
    downloaded_at TEXT DEFAULT (datetime('now')),
    UNIQUE(symbol, funding_time)
);

CREATE INDEX IF NOT EXISTS idx_raw_klines_symbol_time
    ON raw_klines(symbol, interval, open_time);
CREATE INDEX IF NOT EXISTS idx_raw_mark_klines_symbol_time
    ON raw_mark_klines(symbol, interval, open_time);
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


# Milliseconds per UTC day, the bucket width of ``missing_days``.
_DAY_MS = 24 * 60 * 60 * 1000

# Days since the Unix epoch of ``_DAY_MS`` buckets (open_time // _DAY_MS).
_EPOCH_DATE = date(1970, 1, 1)


def _month_bounds_ms(year: int, month: int) -> tuple[int, int]:
    """UTC ``[start, end)`` open_time bounds (ms) of a calendar month."""
    start_ms = int(datetime(year, month, 1, tzinfo=UTC).timestamp() * 1000)
    if month == 12:
        end_ms = int(datetime(year + 1, 1, 1, tzinfo=UTC).timestamp() * 1000)
    else:
        end_ms = int(datetime(year, month + 1, 1, tzinfo=UTC).timestamp() * 1000)
    return start_ms, end_ms


# Bars a fully covered month holds, per interval (absent = unsupported interval).
_KLINES_PER_DAY: dict[str, int] = {
    "1s": 24 * 3600,
    "5s": 24 * 720,
    "1m": 24 * 60,
    "5m": 24 * 12,
    "15m": 24 * 4,
    "1h": 24,
    "4h": 6,
}


def _expected_klines_in_month(interval: str, year: int, month: int) -> int | None:
    """Expected bar count for a month, or None for an unsupported interval."""
    per_day = _KLINES_PER_DAY.get(interval)
    if per_day is None:
        return None
    _, days_in_month = calendar.monthrange(year, month)
    return days_in_month * per_day


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
        """Get or create database connection.

        NOT thread-safe: the lazy ``_conn`` is created and returned without a
        lock, so one instance belongs to one thread / asyncio loop. Before
        sharing an instance across threads, add locking (see
        ``StateManager._write_lock``).
        """
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

    def insert_mark_klines(
        self,
        symbol: str,
        interval: str,
        klines: Iterable[tuple[Any, ...]],
        source: str,
    ) -> int:
        """Insert mark-price kline data into the ``raw_mark_klines`` archive.

        Mark-price klines carry zero volume/taker columns, so only OHLC and the
        two timestamps are stored — one row per minute is already a lot of disk.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT').
            interval: Candle interval (e.g., '1m').
            klines: Iterable of tuples in Binance kline order:
                (open_time, open, high, low, close, volume, close_time, ...);
                only indices 0-4 and 6 are stored.
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
                k[6],  # close_time
                source,
            )
            for k in klines
        ]

        before = self.conn.execute(
            "SELECT COUNT(*) FROM raw_mark_klines WHERE symbol = ? AND interval = ?",
            (symbol, interval),
        ).fetchone()[0]

        self.conn.executemany(
            """INSERT OR IGNORE INTO raw_mark_klines
               (symbol, interval, open_time, open, high, low, close, close_time, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self.conn.commit()

        after = self.conn.execute(
            "SELECT COUNT(*) FROM raw_mark_klines WHERE symbol = ? AND interval = ?",
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
        rows = [(symbol, r[0], r[1], r[2] if len(r) > 2 else None, source) for r in rates]

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

    def insert_funding_intervals(
        self,
        symbol: str,
        rows: Iterable[tuple[int, int]],
        source: str,
    ) -> int:
        """Insert Binance Vision funding intervals into the archive.

        Append-only: rows are keyed by ``(symbol, funding_time)`` and an
        already-archived row is left untouched, so a re-run is idempotent.

        Args:
            symbol: Trading symbol (e.g., 'SOLUSDT').
            rows: Iterable of (funding_time_ms, interval_hours) tuples.
            source: Data source identifier (e.g., 'binance_vision').

        Returns:
            Number of rows inserted.
        """
        values = [(symbol, int(r[0]), int(r[1]), source) for r in rows]

        before = self.conn.execute(
            "SELECT COUNT(*) FROM raw_funding_intervals WHERE symbol = ?",
            (symbol,),
        ).fetchone()[0]

        self.conn.executemany(
            """INSERT OR IGNORE INTO raw_funding_intervals
               (symbol, funding_time, interval_hours, source)
               VALUES (?, ?, ?, ?)""",
            values,
        )
        self.conn.commit()

        after = self.conn.execute(
            "SELECT COUNT(*) FROM raw_funding_intervals WHERE symbol = ?",
            (symbol,),
        ).fetchone()[0]
        return int(after - before)

    def get_funding_intervals(self, symbol: str) -> dict[int, int]:
        """Get archived funding intervals for a symbol.

        Args:
            symbol: Trading symbol.

        Returns:
            ``{funding_time_ms: interval_hours}``, empty when none is archived.
        """
        return {
            int(row[0]): int(row[1])
            for row in self.conn.execute(
                "SELECT funding_time, interval_hours FROM raw_funding_intervals WHERE symbol = ?",
                (symbol,),
            )
        }

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

    def get_mark_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[sqlite3.Row]:
        """Get mark-price klines from the archive, oldest first.

        Args:
            symbol: Trading symbol.
            interval: Candle interval.
            start_ms: Start timestamp (ms), inclusive.
            end_ms: End timestamp (ms), inclusive.

        Returns:
            List of mark kline rows ordered by open_time.
        """
        query = "SELECT * FROM raw_mark_klines WHERE symbol = ? AND interval = ?"
        params: list[Any] = [symbol, interval]

        if start_ms is not None:
            query += " AND open_time >= ?"
            params.append(start_ms)
        if end_ms is not None:
            query += " AND open_time <= ?"
            params.append(end_ms)

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

    def get_mark_date_range(self, symbol: str, interval: str) -> tuple[int, int] | None:
        """Get the date range of stored MARK-price klines.

        Args:
            symbol: Trading symbol.
            interval: Candle interval.

        Returns:
            (min_open_time, max_open_time) tuple or None if no data.
        """
        row = self.conn.execute(
            """SELECT MIN(open_time), MAX(open_time) FROM raw_mark_klines
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
        self,
        symbol: str,
        interval: str,
        year: int,
        month: int,
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
        return self._count_month_rows("raw_klines", symbol, interval, year, month)

    def get_month_mark_kline_count(
        self,
        symbol: str,
        interval: str,
        year: int,
        month: int,
    ) -> int:
        """Get mark-price kline count for a specific month.

        Args:
            symbol: Trading symbol.
            interval: Candle interval.
            year: Year.
            month: Month (1-12).

        Returns:
            Number of mark klines in that month.
        """
        return self._count_month_rows("raw_mark_klines", symbol, interval, year, month)

    def _count_month_rows(
        self, table: str, symbol: str, interval: str, year: int, month: int
    ) -> int:
        """Count *table* rows whose open_time falls inside a calendar month.

        *table* is always one of this module's table-name constants, never user
        input, so interpolating it is safe (values stay parameterized).
        """
        start_ms, end_ms = _month_bounds_ms(year, month)
        row = self.conn.execute(
            f"""SELECT COUNT(*) FROM {table}
                WHERE symbol = ? AND interval = ?
                AND open_time >= ? AND open_time < ?""",
            (symbol, interval, start_ms, end_ms),
        ).fetchone()
        return row[0] if row else 0

    def has_month_coverage(
        self,
        symbol: str,
        interval: str,
        year: int,
        month: int,
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
        return self._has_month_coverage("raw_klines", symbol, interval, year, month, threshold)

    def has_month_mark_coverage(
        self,
        symbol: str,
        interval: str,
        year: int,
        month: int,
        threshold: float = 0.9,
    ) -> bool:
        """Check if a month has sufficient MARK-price kline coverage.

        Same threshold semantics as :meth:`has_month_coverage`.

        Args:
            symbol: Trading symbol.
            interval: Candle interval.
            year: Year.
            month: Month (1-12).
            threshold: Coverage ratio threshold (0-1). Default 0.9.

        Returns:
            True if month is sufficiently covered.
        """
        return self._has_month_coverage("raw_mark_klines", symbol, interval, year, month, threshold)

    def _has_month_coverage(
        self,
        table: str,
        symbol: str,
        interval: str,
        year: int,
        month: int,
        threshold: float,
    ) -> bool:
        """Coverage test shared by the kline and mark-kline tables."""
        expected = _expected_klines_in_month(interval, year, month)
        if expected is None:
            return False
        actual = self._count_month_rows(table, symbol, interval, year, month)
        return actual >= expected * threshold

    def missing_days(
        self,
        symbol: str,
        interval: str,
        start_day: date | None = None,
        end_day: date | None = None,
        table: str = "raw_klines",
    ) -> list[date]:
        """UTC days in [start_day, end_day] with ZERO rows in *table*.

        Only *interior* days are reported: the window is clamped to the symbol's
        own first and last archived day, so neither a not-yet-listed start nor a
        not-yet-published tail (both legitimately empty) shows up. Binance
        Vision monthly zips silently drop whole days while still clearing the
        0.9 ``has_month_coverage`` bar (SOLUSDT 1m 2022-02 ships 25/28), so this
        is the only way to see such a hole.

        A short day (>0 rows, <1440) is real exchange maintenance and is *not*
        reported: the test is "no row at all".

        Args:
            symbol: Trading symbol.
            interval: Candle interval.
            start_day: Lower bound (inclusive); None = the symbol's first day.
            end_day: Upper bound (inclusive); None = the symbol's last day.
            table: ``raw_klines`` or ``raw_mark_klines``. Always one of this
                module's table constants, never user input, so interpolating it
                is safe (values stay parameterized).

        Returns:
            Missing UTC days, oldest first.
        """
        row = self.conn.execute(
            f"SELECT MIN(open_time), MAX(open_time) FROM {table} "
            "WHERE symbol = ? AND interval = ?",
            (symbol, interval),
        ).fetchone()
        if row is None or row[0] is None:
            return []

        first_day = datetime.fromtimestamp(row[0] / 1000, tz=UTC).date()
        last_day = datetime.fromtimestamp(row[1] / 1000, tz=UTC).date()
        lo = max(start_day, first_day) if start_day else first_day
        hi = min(end_day, last_day) if end_day else last_day
        if lo > hi:
            return []

        lo_ms = int(datetime(lo.year, lo.month, lo.day, tzinfo=UTC).timestamp() * 1000)
        hi_ms = int(datetime(hi.year, hi.month, hi.day, tzinfo=UTC).timestamp() * 1000) + _DAY_MS
        present = {
            bucket[0]
            for bucket in self.conn.execute(
                f"""SELECT DISTINCT open_time / {_DAY_MS} FROM {table}
                    WHERE symbol = ? AND interval = ?
                    AND open_time >= ? AND open_time < ?""",
                (symbol, interval, lo_ms, hi_ms),
            )
        }

        missing: list[date] = []
        day = lo
        while day <= hi:
            if (day - _EPOCH_DATE).days not in present:
                missing.append(day)
            day += timedelta(days=1)
        return missing

    def get_symbols(self) -> list[str]:
        """Get list of symbols in archive.

        Returns:
            List of unique symbols.
        """
        rows = self.conn.execute("SELECT DISTINCT symbol FROM raw_klines ORDER BY symbol")
        return [r[0] for r in rows]

    def get_symbols_with_mark(self) -> list[str]:
        """Symbols present in ``raw_klines`` OR ``raw_mark_klines``.

        ``get_symbols`` reads only ``raw_klines``, so a symbol archived solely
        through ``download-mark`` is invisible to it.

        Returns:
            Sorted unique symbols from both tables.
        """
        rows = self.conn.execute(
            "SELECT symbol FROM raw_klines UNION SELECT symbol FROM raw_mark_klines "
            "ORDER BY symbol"
        )
        return [r[0] for r in rows]

    def get_coverage_rows(
        self, interval: str = "1m"
    ) -> list[tuple[str, int, int | None, int | None, int]]:
        """Batched per-symbol coverage stats, one scan per source table.

        Returns ``(symbol, kline_count, min_open_time, max_open_time,
        funding_rate_count)`` for every symbol present in ``raw_klines``,
        ordered by symbol. Symbols whose rows are for other intervals only
        get ``kline_count`` 0 and a ``None`` range.

        Args:
            interval: Candle interval the kline stats are counted for.
        """
        kline_stats = {
            row[0]: (row[1], row[2], row[3])
            for row in self.conn.execute(
                """SELECT symbol, COUNT(*), MIN(open_time), MAX(open_time)
                   FROM raw_klines WHERE interval = ? GROUP BY symbol""",
                (interval,),
            )
        }
        funding_counts = {
            row[0]: row[1]
            for row in self.conn.execute(
                "SELECT symbol, COUNT(*) FROM raw_funding_rates GROUP BY symbol"
            )
        }
        # Skip-scan over the (symbol, interval, open_time) index: one seek per
        # distinct symbol instead of a full DISTINCT scan of every kline row.
        symbol_rows = self.conn.execute(
            """WITH RECURSIVE symbols(x) AS (
                   SELECT MIN(symbol) FROM raw_klines
                   UNION ALL
                   SELECT (SELECT MIN(symbol) FROM raw_klines WHERE symbol > symbols.x)
                     FROM symbols WHERE symbols.x IS NOT NULL
               )
               SELECT x FROM symbols WHERE x IS NOT NULL"""
        )
        result: list[tuple[str, int, int | None, int | None, int]] = []
        for (symbol,) in symbol_rows:
            kline_count, min_time, max_time = kline_stats.get(symbol, (0, None, None))
            result.append(
                (symbol, kline_count, min_time, max_time, funding_counts.get(symbol, 0))
            )
        return result

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
            (
                klines_fetched,
                klines_inserted,
                funding_rates_fetched,
                status,
                error_message,
                session_id,
            ),
        )
        self.conn.commit()

    def get_download_sessions(self, limit: int = 50) -> list[sqlite3.Row]:
        """Get recent download sessions.

        Args:
            limit: Max number of sessions to return.

        Returns:
            List of session rows, newest first.
        """
        return list(
            self.conn.execute(
                "SELECT * FROM download_sessions ORDER BY started_at DESC LIMIT ?",
                (limit,),
            )
        )
