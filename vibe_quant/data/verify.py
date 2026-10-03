"""Data verification functions for kline data quality checks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from vibe_quant.data.archive import RawDataArchive


@runtime_checkable
class KlineRow(Protocol):
    """Protocol for kline row access (sqlite3.Row or dict-like)."""

    def __getitem__(self, key: str) -> int | float: ...


class VerifyResult(TypedDict):
    """Result of verify_symbol function."""

    gaps: list[tuple[int, int, int]]
    ohlc_errors: list[tuple[int, str]]
    zero_volume_runs: list[tuple[int, int, int]]
    kline_count: int


_MINUTE_MS = 60 * 1000


def scan_klines(
    klines: Iterable[KlineRow],
    max_gap_minutes: int = 1,
    min_flat_run: int = 1,
) -> VerifyResult:
    """Single pass over 1m klines (sorted by open_time): gaps, OHLC errors, filler runs.

    Constant memory apart from the findings, so it can stream a DB cursor
    over millions of bars.

    - gaps: any step other than <= ``max_gap_minutes`` whole minutes (default
      1 = exact continuity; also flags duplicates/off-grid timestamps). The old
      5-minute threshold passed every gap of up to 4 missing bars.
    - zero_volume_runs: runs of flat (O=H=L=C) zero-volume bars. Binance Vision
      fills exchange outages with such candles (e.g. BTC 2024-10-28
      20:00-21:14, 74 bars) although the market traded; they pass the OHLC
      checks but are not real prices.
    """
    max_gap_ms = max_gap_minutes * _MINUTE_MS
    gaps: list[tuple[int, int, int]] = []
    errors: list[tuple[int, str]] = []
    runs: list[tuple[int, int, int]] = []
    prev_time: int | None = None
    run_start: int | None = None
    run_last = 0
    run_len = 0
    count = 0

    for k in klines:
        count += 1
        open_time = int(k["open_time"])
        open_price = k["open"]
        high = k["high"]
        low = k["low"]
        close = k["close"]

        if prev_time is not None:
            gap_ms = open_time - prev_time
            if gap_ms > max_gap_ms or gap_ms <= 0 or gap_ms % _MINUTE_MS != 0:
                gaps.append((prev_time, open_time, gap_ms // _MINUTE_MS))
        prev_time = open_time

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

        flat_zero = open_price == high == low == close and _volume(k) == 0.0
        if flat_zero:
            if run_start is None:
                run_start, run_len = open_time, 0
            run_last = open_time
            run_len += 1
        else:
            if run_start is not None and run_len >= min_flat_run:
                runs.append((run_start, run_last, run_len))
            run_start = None
    if run_start is not None and run_len >= min_flat_run:
        runs.append((run_start, run_last, run_len))

    return {"gaps": gaps, "ohlc_errors": errors, "zero_volume_runs": runs, "kline_count": count}


def _volume(k: KlineRow) -> float | None:
    try:
        return float(k["volume"])
    except (KeyError, IndexError):
        return None


def detect_gaps(
    klines: Sequence[KlineRow],
    max_gap_minutes: int = 1,
) -> list[tuple[int, int, int]]:
    """Breaks in the open_time sequence (see :func:`scan_klines`).

    Returns:
        List of (start_ts, end_ts, gap_minutes) tuples, gap_minutes being the
        step between the two bars (2 = one missing 1m bar).
    """
    return scan_klines(klines, max_gap_minutes=max_gap_minutes)["gaps"]


def detect_flat_zero_volume_runs(
    klines: Sequence[KlineRow],
    min_run: int = 1,
) -> list[tuple[int, int, int]]:
    """Runs of flat zero-volume filler bars: (first_ts, last_ts, bars) (see scan_klines)."""
    return scan_klines(klines, min_flat_run=min_run)["zero_volume_runs"]


def check_ohlc_consistency(
    klines: Sequence[KlineRow],
) -> list[tuple[int, str]]:
    """Check OHLC data consistency.

    Validates high >= low, high >= open/close, low <= open/close.

    Returns:
        List of (open_time, error_message) tuples for each inconsistency.
    """
    return scan_klines(klines)["ohlc_errors"]


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
    max_gap_minutes: int = 1,
) -> VerifyResult:
    """Run full verification on symbol data.

    Args:
        archive: RawDataArchive instance.
        symbol: Trading symbol (e.g., 'BTCUSDT').
        interval: Candle interval (default '1m').
        max_gap_minutes: Maximum allowed step in minutes (1 = exact continuity).

    Returns:
        Dict with:
            - gaps: list of (start_ts, end_ts, gap_minutes)
            - ohlc_errors: list of (open_time, error_message)
            - zero_volume_runs: list of (first_ts, last_ts, bars) flat filler runs
            - kline_count: int
    """
    return scan_klines(archive.get_klines(symbol, interval), max_gap_minutes)
