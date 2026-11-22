"""Archive-direct 1h LAST bars of the research tier (arch-c9-2 R41-R44, dreview C9-4).

One source per tier: a 1h run reads ``raw_klines`` interval ``'1h'`` from the
archive, read-only, and never the catalog. Each row is validated before any
engine exists (:class:`CarryDataError` ``"<SYM> 1h row <open_ms>: <reason>"``):
open on the hour, ``close_time == open + 3,599,999 ms``, no duplicate
``close_time``, finite positive OHLC, ``high >= max(open, close)``,
``low <= min(open, close)`` and the close on the instrument snapshot's price
grid. A frozen bar (zero volume and O=H=L=C, as ``data/verify.py`` defines
Binance's filler) is dropped and counted.

Listing evidence comes from two queries OUTSIDE the bounded window read: the
symbol's first archived 1h bar (its listing) and first funding row. A funding
row before the first bar is an incomplete download when that bar opens after the
snapshot's onboard hour, or, without an onboard date, more than 48 h after the
funding row (Binance posts funding up to a day before the first kline: SOL 15 h,
NEAR 16 h, ZEC 24 h); a first bar at or after the
window end means the symbol is not yet listed (an empty :class:`HourBars`), and
a run in which no symbol is listed is refused (:class:`CarryNotListed`).
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from vibe_quant.portfolio.spec import DAY_NS, CarryDataError

if TYPE_CHECKING:
    from collections.abc import Mapping

HOUR_NS = 3_600 * 10**9
WARMUP_DAYS = 161  # trend closes reach B0 - 159 d; one day of slack (R44)
_HOUR_MS = 3_600_000
_LISTING_TOLERANCE_MS = 48 * _HOUR_MS  # funding before the first kline, no onboard date
_NS_PER_MS = 1_000_000
_SELECT = (
    "SELECT open_time, open, high, low, close, volume, close_time FROM raw_klines "
    "WHERE symbol = ? AND interval = '1h' AND open_time >= ? AND open_time < ? "
    "ORDER BY open_time"
)


class CarryNotListed(CarryDataError):
    """No symbol of the run is listed before its end: 0 covered coins (C9-4).

    Raised by the shared loader before any engine is built, so both executors
    stop with the same outcome; ``listings`` maps every symbol to its first
    archived 1h bar (ns).
    """

    def __init__(self, end_ns: int, listings: Mapping[str, int]) -> None:
        self.end_ns = end_ns
        self.listings = dict(listings)
        detail = ", ".join(f"{symbol} listed {ts}" for symbol, ts in sorted(listings.items()))
        super().__init__(f"no symbol listed before end {end_ns}: 0 covered coins ({detail})")


@dataclass(frozen=True)
class HourBars:
    """One symbol's validated 1h LAST closes, ``ts`` = open + 1h - 1 ms (ascending).

    ``listing_ns`` is the open of the symbol's first archived 1h bar, wherever it
    lies; ``frozen_hours`` counts the frozen bars dropped from the window.
    """

    symbol: str
    ts: np.ndarray
    close: np.ndarray
    listing_ns: int
    frozen_hours: int

    def __len__(self) -> int:
        return len(self.ts)


def _first(conn: sqlite3.Connection, query: str, symbol: str) -> int | None:
    row = conn.execute(query, (symbol,)).fetchone()
    return None if row is None or row[0] is None else int(row[0])


def _row_error(
    row: tuple[int, float, float, float, float, float, int],
    previous_close_time: int | None,
    increment: Decimal,
) -> str | None:
    """The first R41 check a row fails, or ``None``."""
    open_ms, o, h, low, c, _volume, close_time = row
    if close_time == previous_close_time:
        return f"duplicate close_time {close_time}"
    if open_ms % _HOUR_MS:
        return "open not on the hour"
    if close_time - open_ms != _HOUR_MS - 1:
        return f"close_time {close_time} is not open + {_HOUR_MS - 1} ms"
    if not all(math.isfinite(v) and v > 0.0 for v in (o, h, low, c)):
        return "non-finite or non-positive OHLC"
    if h < max(o, c):
        return "high below open/close"
    if low > min(o, c):
        return "low above open/close"
    if Decimal(repr(c)) % increment:
        return f"close {c!r} off the price grid {increment}"
    return None


def _incomplete_download(first_bar: int, first_funding: int | None, onboard: int | None) -> bool:
    """Whether 1h bars are missing before ``first_bar`` (all ms; R42 as ruled R-c9t3-2)."""
    if first_funding is None or first_funding >= first_bar:
        return False
    if onboard is not None:
        return first_bar > onboard - onboard % _HOUR_MS
    return first_bar - first_funding > _LISTING_TOLERANCE_MS


def load_hour_bars(
    archive: Path,
    symbol: str,
    from_ns: int,
    to_ns: int,
    increment: Decimal,
    *,
    onboard_ms: int | None = None,
) -> HourBars:
    """The symbol's validated 1h bars opening in ``[from_ns, to_ns)`` (R41-R44).

    ``onboard_ms`` is the snapshot's listing evidence (Binance ``onboardDate``).

    Raises:
        CarryDataError: the archive cannot be read, has no 1h row for the symbol,
            starts its 1h bars after the first funding row and after the onboard
            hour (or, with no onboard date, more than 48 h after it: incomplete
            download), or holds a row that fails a check. A symbol whose first bar
            opens at or after ``to_ns`` returns empty bars (``not_listed``), unchecked.
    """
    try:
        conn = sqlite3.connect(f"{Path(archive).resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise CarryDataError(f"archive read failed: {symbol}: {error}") from error
    try:
        first_bar = _first(
            conn,
            "SELECT MIN(open_time) FROM raw_klines WHERE symbol = ? AND interval = '1h'",
            symbol,
        )
        if first_bar is None:
            raise CarryDataError(f"{symbol}: no 1h klines in archive {archive}")
        first_funding = _first(
            conn, "SELECT MIN(funding_time) FROM raw_funding_rates WHERE symbol = ?", symbol
        )
        if _incomplete_download(first_bar, first_funding, onboard_ms):
            raise CarryDataError(f"{symbol}: 1h bars start after funding (incomplete download)")
        listing_ns = first_bar * _NS_PER_MS
        if listing_ns >= to_ns:
            empty = np.array([], dtype=np.int64)
            return HourBars(symbol, empty, np.array([], dtype=np.float64), listing_ns, 0)
        rows = conn.execute(
            _SELECT, (symbol, from_ns // _NS_PER_MS, -(-to_ns // _NS_PER_MS))
        ).fetchall()
    except sqlite3.Error as error:
        raise CarryDataError(f"archive read failed: {symbol}: {error}") from error
    finally:
        conn.close()
    ts: list[int] = []
    close: list[float] = []
    frozen = 0
    previous: int | None = None
    for raw in rows:
        row = (
            int(raw[0]),
            float(raw[1]),
            float(raw[2]),
            float(raw[3]),
            float(raw[4]),
            float(raw[5]),
            int(raw[6]),
        )
        reason = _row_error(row, previous, increment)
        if reason is not None:
            raise CarryDataError(f"{symbol} 1h row {row[0]}: {reason}")
        previous = row[6]
        if row[5] == 0.0 and row[1] == row[2] == row[3] == row[4]:
            frozen += 1
            continue
        ts.append(row[6] * _NS_PER_MS)
        close.append(row[4])
    return HourBars(
        symbol,
        np.array(ts, dtype=np.int64),
        np.array(close, dtype=np.float64),
        listing_ns,
        frozen,
    )


def warmup_start(start_ns: int) -> int:
    """First bar open a 1h run reads: ``start - 161 days`` (R44)."""
    return start_ns - WARMUP_DAYS * DAY_NS
