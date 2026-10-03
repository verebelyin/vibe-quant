"""Shared request-field validators (garbage in → 422, never a silent default)."""

from __future__ import annotations

import re
from datetime import date

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Timeframes the catalog/aggregation layer supports.
KNOWN_TIMEFRAMES = frozenset({"1m", "5m", "15m", "30m", "1h", "4h", "1d"})


def parse_iso_date(value: str, field: str) -> date:
    """Parse a strict ``YYYY-MM-DD`` date or raise ValueError naming ``field``."""
    if not isinstance(value, str) or not _ISO_DATE.match(value):
        raise ValueError(f"{field} must be YYYY-MM-DD, got {value!r}")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} is not a valid date: {value!r}") from exc


def check_date_range(start: str, end: str, *, allow_equal: bool) -> None:
    """Both dates valid and start before end (or equal if ``allow_equal``)."""
    start_d = parse_iso_date(start, "start_date")
    end_d = parse_iso_date(end, "end_date")
    if end_d < start_d or (end_d == start_d and not allow_equal):
        raise ValueError(f"start_date {start} must be before end_date {end}")


def clean_symbols(symbols: list[str]) -> list[str]:
    """Non-empty list of non-blank symbols (whitespace stripped)."""
    cleaned = [s.strip() for s in symbols]
    if not cleaned:
        raise ValueError("symbols must not be empty")
    if any(not s for s in cleaned):
        raise ValueError("symbols must not contain blank entries")
    return cleaned
