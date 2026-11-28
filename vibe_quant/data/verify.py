"""Data verification functions for kline data quality checks."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

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


class MissingDays(TypedDict):
    """Interior UTC days with zero archived rows, per archive table."""

    klines: list[str]
    mark_klines: list[str]


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
    - zero_volume_runs: runs of flat (O=H=L=C) zero-volume bars, split at every
      gap so a run's bars are consecutive. Binance Vision
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
                # a filler run never spans a gap: (first, last, bars) stays contiguous
                if run_start is not None and run_len >= min_flat_run:
                    runs.append((run_start, run_last, run_len))
                run_start = None
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


def interior_missing_days(
    archive: RawDataArchive,
    symbol: str,
    interval: str = "1m",
) -> MissingDays:
    """Interior UTC days with zero archived rows, for klines and mark klines.

    Binance Vision monthly zips silently omit whole days (and the 0.9
    ``has_month_coverage`` smart skip hides that), so a gap in day coverage is
    invisible to :func:`verify_symbol`'s minute-step gap scan — a 3-day hole is
    just one gap tuple. These are the same days the ingest daily-fill pass
    requests, reported as ISO date strings.

    Args:
        archive: RawDataArchive instance.
        symbol: Trading symbol (e.g., 'BTCUSDT').
        interval: Candle interval (default '1m').

    Returns:
        Dict with ``klines`` / ``mark_klines``: missing UTC days, oldest first.
    """
    return {
        "klines": [
            day.isoformat() for day in archive.missing_days(symbol, interval, table="raw_klines")
        ],
        "mark_klines": [
            day.isoformat()
            for day in archive.missing_days(symbol, interval, table="raw_mark_klines")
        ],
    }


class ResearchVerifyResult(TypedDict):
    """Research-tier (1h/1d LAST + funding) checks for one symbol."""

    kline_counts: dict[str, int]
    gaps: dict[str, list[tuple[int, int, int]]]
    accepted_gaps: dict[str, list[tuple[int, int, int]]]
    ohlc_errors: list[tuple[int, str]]
    no_data_days: list[str]
    ticker_reuse: list[tuple[int, str]]
    coverage: list[str]


# A daily open this far from the previous daily close is not one asset trading
# continuously (perps trade 24/7, so open ~= previous close): the ticker was
# reused, redenominated or relisted.
TICKER_REUSE_RATIO = 2.0

_HOUR_MS = 3_600_000
_DAY_MS = 86_400_000
# Longest Binance funding interval (8h) plus settlement-time jitter.
_FUNDING_MAX_STEP_MS = 8 * _HOUR_MS + _MINUTE_MS


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).strftime("%Y-%m-%dT%H:%M")


def _month(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).strftime("%Y-%m")


def _research_coverage(archive: RawDataArchive, symbol: str, now: datetime) -> list[str]:
    """Required streams present and spanning the same history.

    1h and 1d start on the same day and end within a day of each other (the
    daily bar of the current day is not closed yet); funding starts within one
    8h settlement of the first price bar, ends within one of the last closed
    hour and has no step above 8h; funding intervals exist for every month of
    the price/funding span except the current one (Vision publishes a month's
    file after it ends).
    """
    issues: list[str] = []
    hourly = archive.get_date_range(symbol, "1h")
    daily = archive.get_date_range(symbol, "1d")
    times = [int(r["funding_time"]) for r in archive.get_funding_rates(symbol)]
    for name, present in (("1h klines", hourly), ("1d klines", daily), ("funding rates", times)):
        if not present:
            issues.append(f"MISSING {name}")
    if hourly and daily:
        if hourly[0] // _DAY_MS * _DAY_MS != daily[0]:
            issues.append(f"1h starts {_iso(hourly[0])} but 1d starts {_iso(daily[0])}")
        h_end, d_end = hourly[1] + _HOUR_MS, daily[1] + _DAY_MS
        if not d_end <= h_end < d_end + _DAY_MS:
            issues.append(f"1d ends {_iso(d_end)} but 1h ends {_iso(h_end)}")
    if hourly and times:
        if times[0] > hourly[0] + _FUNDING_MAX_STEP_MS:
            issues.append(
                f"funding starts {_iso(times[0])}, more than 8h after the first 1h bar "
                f"{_iso(hourly[0])}"
            )
        if times[-1] < hourly[1] + _HOUR_MS - _FUNDING_MAX_STEP_MS:
            issues.append(
                f"funding ends {_iso(times[-1])}, more than 8h before the last 1h bar closes"
            )
    issues += [
        f"funding gap {_iso(a)} .. {_iso(b)} ({(b - a) / _HOUR_MS:g}h)"
        for a, b in zip(times, times[1:], strict=False)
        if b - a > _FUNDING_MAX_STEP_MS
    ]
    spans = [t for r in (hourly, daily) if r for t in r] + times[:1] + times[-1:]
    if spans:
        current = now.strftime("%Y-%m")
        expected = {m for m in _months_between(min(spans), max(spans)) if m < current}
        have = {_month(t) for t in archive.get_funding_intervals(symbol)}
        if missing := sorted(expected - have):
            issues.append(f"MISSING funding intervals for {', '.join(missing)}")
    return issues


def _months_between(lo_ms: int, hi_ms: int) -> list[str]:
    lo = datetime.fromtimestamp(lo_ms / 1000, UTC).date().replace(day=1)
    hi = datetime.fromtimestamp(hi_ms / 1000, UTC).date()
    months: list[str] = []
    while lo <= hi:
        months.append(lo.strftime("%Y-%m"))
        lo = date(lo.year + lo.month // 12, lo.month % 12 + 1, 1)
    return months


def _accepted_gap_matcher(
    accepted: list[tuple[int, int]], step_ms: int
) -> Callable[[tuple[int, int, int]], bool]:
    """Predicate: is a ``scan_klines`` gap's *missing bars* fully inside an accepted range?

    ``scan_klines`` reports ``(prev_open_time, next_open_time, gap_minutes)``;
    the missing bars are the open times ``prev + step .. next - step``. A gap
    of zero or negative span (duplicate) has no missing bar, and an off-grid
    gap (not a whole number of bars) hides an off-grid bar: neither matches.
    """

    def matches(gap: tuple[int, int, int]) -> bool:
        prev_ts, next_ts, _gap_min = gap
        lo, hi = prev_ts + step_ms, next_ts - step_ms
        on_grid = (next_ts - prev_ts) % step_ms == 0
        return on_grid and lo <= hi and any(a <= lo and hi <= b for a, b in accepted)

    return matches


def verify_research_symbol(
    archive: RawDataArchive, symbol: str, now: datetime | None = None
) -> ResearchVerifyResult:
    """Coverage, gaps, OHLC errors, frozen days and ticker reuse of a research coin.

    - coverage: required streams absent or not spanning the same history
      (see :func:`_research_coverage`).
    - gaps: a step longer than one bar (60 / 1440 min), duplicates, off-grid.
      A gap whose missing bars all lie inside a recorded accepted gap is
      reported in ``accepted_gaps`` instead.
    - no_data_days: flat (O=H=L=C) zero-volume daily bars, Binance's filler for
      a day with no trading: reported as no data, not as prices.
    - ticker_reuse: a daily open outside [prev close / 2, prev close * 2].
    """
    counts: dict[str, int] = {}
    gaps: dict[str, list[tuple[int, int, int]]] = {}
    accepted_gaps: dict[str, list[tuple[int, int, int]]] = {}
    errors: list[tuple[int, str]] = []
    no_data: list[str] = []
    for interval, minutes in (("1h", 60), ("1d", 1440)):
        scan = scan_klines(archive.get_klines(symbol, interval), max_gap_minutes=minutes)
        counts[interval] = scan["kline_count"]
        is_accepted = _accepted_gap_matcher(
            archive.get_accepted_gaps(symbol, interval), minutes * _MINUTE_MS
        )
        gaps[interval] = [g for g in scan["gaps"] if not is_accepted(g)]
        accepted_gaps[interval] = [g for g in scan["gaps"] if is_accepted(g)]
        errors += scan["ohlc_errors"]
        if interval == "1d":
            no_data = [
                datetime.fromtimestamp((first + i * _DAY_MS) / 1000, UTC).date().isoformat()
                for first, _last, bars in scan["zero_volume_runs"]
                for i in range(bars)
            ]

    reuse: list[tuple[int, str]] = []
    prev_close: float | None = None
    for k in archive.get_klines(symbol, "1d"):
        open_price = float(k["open"])
        if prev_close and not (
            prev_close / TICKER_REUSE_RATIO <= open_price <= prev_close * TICKER_REUSE_RATIO
        ):
            reuse.append((int(k["open_time"]), f"open {open_price} vs previous close {prev_close}"))
        prev_close = float(k["close"])
    return {
        "kline_counts": counts,
        "gaps": gaps,
        "accepted_gaps": accepted_gaps,
        "ohlc_errors": errors,
        "no_data_days": no_data,
        "ticker_reuse": reuse,
        "coverage": _research_coverage(archive, symbol, now or datetime.now(UTC)),
    }


class MetricsVerifyResult(TypedDict):
    """Positioning (Vision ``metrics``) checks of one symbol.

    Only the newest files row of a day counts: an older malformed/drift row a
    later file superseded is resolved. ``drift``/``malformed``/``same`` are
    ``(day, detail)`` of newest rows without an ack; ``acked`` is
    ``(status, day, detail, reason)``.
    """

    files: int
    first_day: str | None
    last_day: str | None
    missing_days: list[str]
    missing_rows_by_day: dict[str, int]
    misaligned: int
    null_counts: dict[str, int]
    drift: list[tuple[str, str | None]]
    malformed: list[tuple[str, str | None]]
    acked: list[tuple[str, str, str | None, str]]
    same: int


class DvolVerifyResult(TypedDict):
    """Hourly DVOL candles of one currency; ``gaps`` = ``(first missing open, candles)``."""

    candles: int
    first: int | None
    last: int | None
    gaps: list[tuple[int, int]]


METRICS_STEP_MS = 300_000
METRICS_ROWS_PER_DAY = _DAY_MS // METRICS_STEP_MS
_METRICS_VALUE_COLUMNS = (
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)


def _day_iso(day_index: int) -> str:
    return datetime.fromtimestamp(day_index * _DAY_MS / 1000, UTC).date().isoformat()


def verify_metrics(archive: RawDataArchive, symbol: str) -> MetricsVerifyResult:
    """Coverage, holes and file outcomes of *symbol*'s archived metrics.

    Holes (missing days, missing 5-minute rows, off-grid rows, NULL fields)
    are information: the indicators read them as NaN. A day whose newest file
    is drift or malformed and has no ack is an issue for the caller.
    """
    days = list(
        archive.conn.execute(
            f"""SELECT create_time / {_DAY_MS}, COUNT(*), SUM(create_time % {METRICS_STEP_MS} = 0)
                FROM raw_metrics WHERE symbol = ? GROUP BY 1 ORDER BY 1""",
            (symbol,),
        )
    )
    nulls = archive.conn.execute(
        "SELECT "
        + ", ".join(f"COALESCE(SUM({c} IS NULL), 0)" for c in _METRICS_VALUE_COLUMNS)
        + " FROM raw_metrics WHERE symbol = ?",
        (symbol,),
    ).fetchone()
    have = {int(d) for d, _n, _on_grid in days}
    files = archive.get_metrics_files(symbol)
    newest = {row["day"]: row for row in files}  # ordered by day, id: the last row wins
    acks = archive.get_metrics_acks(symbol)
    result: MetricsVerifyResult = {
        "files": len(files),
        "first_day": _day_iso(min(have)) if have else None,
        "last_day": _day_iso(max(have)) if have else None,
        "missing_days": [_day_iso(d) for d in range(min(have), max(have)) if d not in have]
        if have
        else [],
        "missing_rows_by_day": {
            _day_iso(int(d)): METRICS_ROWS_PER_DAY - int(g)
            for d, _n, g in days
            if int(g) < METRICS_ROWS_PER_DAY
        },
        "misaligned": sum(int(n) - int(g) for _d, n, g in days),
        "null_counts": {c: int(v) for c, v in zip(_METRICS_VALUE_COLUMNS, nulls, strict=True) if v},
        "drift": [],
        "malformed": [],
        "acked": [],
        "same": 0,
    }
    for day, row in newest.items():
        status, detail = row["status"], row["detail"]
        if status in ("drift", "malformed"):
            ack = acks.get(int(row["id"]))
            if ack is not None:
                result["acked"].append((status, day, detail, ack["reason"]))
            elif status == "drift":
                result["drift"].append((day, detail))
            else:
                result["malformed"].append((day, detail))
        elif status == "same":
            result["same"] += 1
    return result


def verify_dvol(archive: RawDataArchive, currency: str) -> DvolVerifyResult:
    """Count, span and interior holes of *currency*'s hourly DVOL candles (information)."""
    opens = [
        int(r[0])
        for r in archive.conn.execute(
            f"SELECT open_time FROM raw_dvol WHERE currency = ? AND resolution_s = {_HOUR_MS // 1000} "
            "ORDER BY open_time",
            (currency,),
        )
    ]
    return {
        "candles": len(opens),
        "first": opens[0] if opens else None,
        "last": opens[-1] if opens else None,
        "gaps": [
            (a + _HOUR_MS, (b - a) // _HOUR_MS - 1)
            for a, b in zip(opens, opens[1:], strict=False)
            if b - a > _HOUR_MS
        ],
    }
