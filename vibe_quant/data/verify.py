"""Data verification functions for kline data quality checks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vibe_quant.data.archive import RawDataArchive


@runtime_checkable
class KlineRow(Protocol):
    """Protocol for kline row access (sqlite3.Row or dict-like)."""

    def __getitem__(self, key: str) -> int | float: ...


class VerifyResult(TypedDict):
    """Result of verify_symbol function."""

    gaps: list[tuple[int, int, int]]
    ohlc_errors: list[tuple[int, str]]
    kline_count: int


# Gap detection threshold: 5 minutes in milliseconds
MAX_GAP_MS = 5 * 60 * 1000


def detect_gaps(
    klines: Sequence[KlineRow],
    max_gap_minutes: int = 5,
) -> list[tuple[int, int, int]]:
    """Detect time gaps in kline data.

    Args:
        klines: Sequence of kline rows from archive (sorted by open_time).
        max_gap_minutes: Maximum allowed gap in minutes before flagging.

    Returns:
        List of (start_ts, end_ts, gap_minutes) tuples for each gap found.
    """
    if len(klines) < 2:
        return []

    max_gap_ms = max_gap_minutes * 60 * 1000
    gaps: list[tuple[int, int, int]] = []

    for i in range(1, len(klines)):
        prev_time = int(klines[i - 1]["open_time"])
        curr_time = int(klines[i]["open_time"])
        gap_ms = curr_time - prev_time

        # Expected gap is 1 minute (60000 ms) for 1m data
        if gap_ms > max_gap_ms:
            gap_minutes = gap_ms // 60000
            gaps.append((prev_time, curr_time, gap_minutes))

    return gaps


def check_ohlc_consistency(
    klines: Sequence[KlineRow],
) -> list[tuple[int, str]]:
    """Check OHLC data consistency.

    Validates:
    - high >= low
    - high >= open and high >= close
    - low <= open and low <= close

    Args:
        klines: Sequence of kline rows from archive.

    Returns:
        List of (open_time, error_message) tuples for each inconsistency.
    """
    errors: list[tuple[int, str]] = []

    for k in klines:
        open_time = int(k["open_time"])
        open_price = k["open"]
        high = k["high"]
        low = k["low"]
        close = k["close"]

        if high < low:
            errors.append((open_time, f"high ({high}) < low ({low})"))

        if high < open_price:
            errors.append((open_time, f"high ({high}) < open ({open_price})"))

        if high < close:
            errors.append((open_time, f"high ({high}) < close ({close})"))

        if low > open_price:
            errors.append((open_time, f"low ({low}) > open ({open_price})"))

        if low > close:
            errors.append((open_time, f"low ({low}) > close ({close})"))

    return errors


def validate_row_count(
    actual: int,
    expected: int,
    tolerance: float = 0.01,
) -> bool:
    """Validate row count is within tolerance.

    Args:
        actual: Actual row count.
        expected: Expected row count.
        tolerance: Allowed tolerance as fraction (default 1%).

    Returns:
        True if actual count is within tolerance of expected.
    """
    if expected == 0:
        return actual == 0

    deviation = abs(actual - expected) / expected
    return deviation <= tolerance


def verify_symbol(
    archive: RawDataArchive,
    symbol: str,
    interval: str = "1m",
    max_gap_minutes: int = 5,
) -> VerifyResult:
    """Run full verification on symbol data.

    Args:
        archive: RawDataArchive instance.
        symbol: Trading symbol (e.g., 'BTCUSDT').
        interval: Candle interval (default '1m').
        max_gap_minutes: Maximum allowed gap in minutes.

    Returns:
        Dict with:
            - gaps: list of (start_ts, end_ts, gap_minutes)
            - ohlc_errors: list of (open_time, error_message)
            - kline_count: int
    """
    klines = archive.get_klines(symbol, interval)

    return {
        "gaps": detect_gaps(klines, max_gap_minutes),
        "ohlc_errors": check_ohlc_consistency(klines),
        "kline_count": len(klines),
    }
