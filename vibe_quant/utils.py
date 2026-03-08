"""Shared utilities for vibe_quant."""

from __future__ import annotations

from datetime import datetime as _datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator
    from datetime import datetime

# Timeframe string to minutes mapping (shared by screening + discovery)
TIMEFRAME_MINUTES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
}


def compute_bar_count(
    start_date: str | None,
    end_date: str | None,
    timeframe: str,
) -> int | None:
    """Compute number of bars from date range and timeframe.

    Returns None if dates are missing or timeframe is unknown.
    """
    if not start_date or not end_date:
        return None
    tf_minutes = TIMEFRAME_MINUTES.get(timeframe)
    if not tf_minutes:
        return None
    start = _datetime.strptime(start_date, "%Y-%m-%d")
    end = _datetime.strptime(end_date, "%Y-%m-%d")
    total_minutes = (end - start).total_seconds() / 60
    return max(int(total_minutes / tf_minutes), 1)


def generate_month_range(start_date: datetime, end_date: datetime) -> Generator[tuple[int, int]]:
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
