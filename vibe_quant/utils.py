"""Shared utilities for vibe_quant."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator
    from datetime import datetime


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
