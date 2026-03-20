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


def split_date_range(
    start_date: str,
    end_date: str,
    split_ratio: float,
) -> tuple[str, str, str, str]:
    """Split a date range into train and holdout periods.

    Args:
        start_date: Start date (ISO format YYYY-MM-DD).
        end_date: End date (ISO format YYYY-MM-DD).
        split_ratio: Fraction of total range for training (0-1).
            E.g. 0.5 = first 50% train, last 50% holdout.

    Returns:
        (train_start, train_end, holdout_start, holdout_end) as ISO strings.

    Raises:
        ValueError: If split_ratio not in (0, 1) or dates invalid.
    """
    if not (0.0 < split_ratio < 1.0):
        raise ValueError(f"split_ratio must be in (0, 1), got {split_ratio}")
    start = _datetime.strptime(start_date, "%Y-%m-%d")
    end = _datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days
    if total_days < 2:
        raise ValueError(f"Date range too short for split: {total_days} days")
    train_days = max(1, int(total_days * split_ratio))
    split_point = start + __import__("datetime").timedelta(days=train_days)
    return (
        start_date,
        split_point.strftime("%Y-%m-%d"),
        split_point.strftime("%Y-%m-%d"),
        end_date,
    )


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
