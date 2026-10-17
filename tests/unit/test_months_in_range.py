"""Tests for get_months_in_range month-selection contract.

A range whose ``end`` is the last calendar day of month M includes M when M is
strictly before the current UTC month, so ingest preview and ingest agree and a
finished month is actually counted. The current (unfinished) month is never
included.
"""

import calendar
from datetime import UTC, datetime

import pytest

from vibe_quant.data.downloader import get_months_in_range

# Fixed "now" for every contract row: the bug report's reference instant.
ROW_NOW = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)
START = datetime(2025, 1, 1, tzinfo=UTC)


def _months_through(last: tuple[int, int]) -> list[tuple[int, int]]:
    """Every (year, month) from (2025, 1) through ``last`` inclusive."""
    months: list[tuple[int, int]] = []
    year, mon = 2025, 1
    while (year, mon) <= last:
        months.append((year, mon))
        year, mon = (year + 1, 1) if mon == 12 else (year, mon + 1)
    return months


@pytest.mark.parametrize(
    ("end", "expected_last"),
    [
        (datetime(2025, 6, 30, 0, 0, tzinfo=UTC), (2025, 6)),
        (datetime(2025, 6, 30, 23, 59, tzinfo=UTC), (2025, 6)),
        (datetime(2025, 6, 29, tzinfo=UTC), (2025, 5)),
        (datetime(2025, 7, 1, 0, 0, tzinfo=UTC), (2025, 6)),
        (datetime(2025, 12, 31, tzinfo=UTC), (2025, 12)),
        (datetime(2026, 9, 30, tzinfo=UTC), (2026, 9)),
        (datetime(2026, 10, 31, tzinfo=UTC), (2026, 9)),
        (datetime(2026, 10, 17, tzinfo=UTC), (2026, 9)),
    ],
)
def test_months_in_range_contract_rows(end: datetime, expected_last: tuple[int, int]) -> None:
    assert get_months_in_range(START, end, now=ROW_NOW) == _months_through(expected_last)


def test_naive_endpoints_are_supported() -> None:
    end = datetime(2025, 6, 30, 23, 59)
    assert get_months_in_range(datetime(2025, 1, 1), end, now=ROW_NOW) == _months_through(
        (2025, 6)
    )


def test_start_after_end_is_empty() -> None:
    months = get_months_in_range(
        datetime(2025, 7, 15, tzinfo=UTC),
        datetime(2025, 6, 30, tzinfo=UTC),
        now=ROW_NOW,
    )
    assert months == []


def test_current_month_never_included_with_default_now() -> None:
    now = datetime.now(UTC)
    end = now.replace(day=calendar.monthrange(now.year, now.month)[1])
    months = get_months_in_range(datetime(end.year, 1, 1, tzinfo=UTC), end)

    assert (now.year, now.month) not in months
    expected_prev = (now.year - 1, 12) if now.month == 1 else (now.year, now.month - 1)
    assert months[-1] == expected_prev


def test_now_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        get_months_in_range(START, datetime(2025, 6, 30, tzinfo=UTC), ROW_NOW)  # type: ignore[misc]
