"""Data ingestion orchestration."""

from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path  # noqa: TC003 (used at runtime in rebuild_from_archive)
from typing import TYPE_CHECKING, Any

import httpx

from vibe_quant.data.archive import RawDataArchive
from vibe_quant.data.catalog import (
    DEFAULT_CATALOG_PATH,
    INSTRUMENT_CONFIGS,
    WRITE_LOCK_NAME,
    CatalogManager,
    aggregate_bars,
    catalog_write_lock,
    create_instrument,
    get_bar_type,
    klines_to_bars,
)
from vibe_quant.data.downloader import (
    SUPPORTED_SYMBOLS,
    DailyDownloadError,
    DownloadError,
    MonthlyDownloadError,
    RestDownloadError,
    download_daily_klines,
    download_daily_mark_klines,
    download_funding_rates,
    download_monthly_funding_intervals,
    download_monthly_klines,
    download_monthly_mark_klines,
    download_recent_klines,
    get_months_in_range,
    get_years_months_to_download,
)
from vibe_quant.data.mark_bars import build_mark_bars
from vibe_quant.data.verify import (
    interior_missing_days,
    verify_dvol,
    verify_metrics,
    verify_research_symbol,
    verify_symbol,
)
from vibe_quant.utils import generate_month_range

if TYPE_CHECKING:
    from vibe_quant.data.metrics import VisionFile

logger = logging.getLogger(__name__)

# Timeframes aggregated from 1m bars (1d = UTC-midnight aligned; ts_event=open, ts_init=close).
AGGREGATION_INTERVALS = ["5m", "15m", "1h", "4h", "1d"]

# Cap for the download-session error summary so one bad symbol cannot bloat the row.
_MAX_SESSION_SUMMARY = 2000

# One UTC day in milliseconds (day-bucket queries in the tail pass).
_DAY_MS = 86_400_000


def _failed_downloads_summary(
    results: dict[str, dict[str, Any]],
) -> tuple[bool, str | None]:
    """Summarize per-symbol download holes for the session audit row.

    ``failed_months`` (Binance Vision months that errored), ``failed_ranges``
    (REST ranges that errored) and ``missing_days`` (interior days whose daily
    Vision zip was 404/failed, so still a hole in the archive) are the holes
    worth re-running. Returns ``(has_holes, compact JSON)``; no holes gives
    ``(False, None)``.
    """
    failed: dict[str, dict[str, list[str]]] = {}
    for symbol, counts in results.items():
        months = list(counts.get("failed_months", []))
        ranges = list(counts.get("failed_ranges", []))
        days = list(counts.get("missing_days", []))
        if months or ranges or days:
            failed[symbol] = {
                "failed_months": months,
                "failed_ranges": ranges,
                "missing_days": days,
            }
    if not failed:
        return False, None
    summary = json.dumps(failed, sort_keys=True, separators=(",", ":"))
    if len(summary) > _MAX_SESSION_SUMMARY:
        summary = summary[: _MAX_SESSION_SUMMARY - 3] + "..."
    return True, summary


def _interval_to_minutes(interval: str) -> int:
    """Convert interval string to minutes.

    Supports 's' (seconds), 'm' (minutes), 'h' (hours), 'd' (days) suffixes.

    Args:
        interval: Interval string like '5s', '1m', '5m', '1h', '4h', '1d'.

    Returns:
        Number of minutes. Sub-minute intervals return 0.

    Raises:
        ValueError: If interval format is unrecognized.
    """
    interval = interval.strip().lower()
    if interval.endswith("s"):
        return 0  # Sub-minute: cannot aggregate from 1m bars
    elif interval.endswith("m"):
        return int(interval[:-1])
    elif interval.endswith("h"):
        return int(interval[:-1]) * 60
    elif interval.endswith("d"):
        return int(interval[:-1]) * 1440
    else:
        msg = f"Unrecognized interval format '{interval}'. Use suffixes: s, m, h, d"
        raise ValueError(msg)


def _interval_to_seconds(interval: str) -> int:
    """Convert interval string to seconds.

    Args:
        interval: Interval string like '1s', '5s', '1m', '5m'.

    Returns:
        Number of seconds.

    Raises:
        ValueError: If interval format is unrecognized.
    """
    interval = interval.strip().lower()
    if interval.endswith("s"):
        return int(interval[:-1])
    elif interval.endswith("m"):
        return int(interval[:-1]) * 60
    elif interval.endswith("h"):
        return int(interval[:-1]) * 3600
    elif interval.endswith("d"):
        return int(interval[:-1]) * 86400
    else:
        msg = f"Unrecognized interval format '{interval}'. Use suffixes: s, m, h, d"
        raise ValueError(msg)


def _drop_unclosed_klines(klines: list[tuple[Any, ...]], now_ms: int) -> list[tuple[Any, ...]]:
    """Drop the currently-forming candle from a REST kline batch.

    Binance returns the in-progress candle regardless of ``endTime`` — its
    ``close_time`` (tuple index 6) lies in the future (``close_time >= now_ms``)
    and its OHLC is partial. Archiving it would freeze that partial OHLC
    permanently: the next incremental fetch starts at ``last_open_time +
    interval`` and never re-pulls the now-completed bar. Keep only closed bars.
    """
    return [k for k in klines if int(k[6]) < now_ms]


def get_download_preview(
    symbols: list[str],
    start_date: datetime,
    end_date: datetime,
    archive: RawDataArchive | None = None,
) -> list[dict[str, Any]]:
    """Preview what months will be downloaded vs skipped.

    Args:
        symbols: Symbols to check.
        start_date: Start date.
        end_date: End date.
        archive: Raw data archive.

    Returns:
        List of dicts with keys: symbol, year, month, status, kline_count, expected.
    """
    import calendar

    owns_archive = archive is None
    if archive is None:
        archive = RawDataArchive()

    months = get_months_in_range(start_date, end_date)
    preview: list[dict[str, Any]] = []

    try:
        for symbol in symbols:
            for year, month in months:
                actual = archive.get_month_kline_count(symbol, "1m", year, month)
                _, days = calendar.monthrange(year, month)
                expected = days * 24 * 60

                if actual >= expected * 0.9:
                    status = "Archived"
                elif actual > 0:
                    status = "Partial - will re-download"
                else:
                    status = "Will download"

                preview.append(
                    {
                        "Symbol": symbol,
                        "Month": f"{year}-{month:02d}",
                        "Status": status,
                        "Klines": actual,
                        "Expected": expected,
                    }
                )
    finally:
        # Close only the archive opened here; a caller-supplied one stays open.
        if owns_archive:
            archive.close()

    return preview


def _update_funding_rates(symbol: str, archive: RawDataArchive, verbose: bool) -> int:
    """Fetch funding rates after the last archived funding_time (+1 ms).

    Skipped when the archive has no funding yet (use ``ingest`` for backfill).
    """
    last = archive.conn.execute(
        "SELECT MAX(funding_time) FROM raw_funding_rates WHERE symbol = ?", (symbol,)
    ).fetchone()[0]
    if last is None:
        if verbose:
            print(f"{symbol}: no archived funding, use 'ingest' to backfill")
        return 0
    start_time = int(last) + 1
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    if start_time >= now_ms:
        return 0
    # strict: a 429/5xx page raises instead of passing a truncated tail off as complete
    try:
        rates = download_funding_rates(symbol, start_time, now_ms, strict=True)
    except RestDownloadError as e:
        # the pages before the failed one are contiguous from start_time: keep them
        if e.partial:
            archive.insert_funding_rates(symbol, e.partial, "binance_api")
        raise
    if not rates:
        if verbose:
            print("  No new funding rates")
        return 0
    inserted = archive.insert_funding_rates(symbol, rates, "binance_api")
    if verbose:
        print(f"  Archived {inserted} new funding rates")
    return inserted


def _fetch_rest_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    failed_ranges: list[str],
) -> list[tuple[Any, ...]]:
    """REST klines; a page failing after retries is recorded, not swallowed.

    Returns what was fetched before the failure (still worth archiving) and
    appends "<from>..<to>" (UTC ISO) of the unfetched remainder to
    *failed_ranges*.
    """
    try:
        return download_recent_klines(symbol, interval, start_ms, end_ms)
    except RestDownloadError as e:
        lo = datetime.fromtimestamp(e.failed_from / 1000, tz=UTC).isoformat()
        hi = datetime.fromtimestamp(end_ms / 1000, tz=UTC).isoformat()
        failed_ranges.append(f"{lo}..{hi}")
        logger.error("%s %s REST fetch FAILED from %s: %s", symbol, interval, lo, e)
        return e.partial


def update_symbol(
    symbol: str,
    archive: RawDataArchive | None = None,
    catalog: CatalogManager | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Update data for a symbol by fetching missing candles.

    Detects last timestamp in archive, fetches gap via REST API,
    archives raw data, then rebuilds catalog.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT').
        archive: Raw data archive (created if not provided).
        catalog: Catalog manager (created if not provided).
        verbose: Print progress messages.

    Returns:
        Dict with counts: {'new_klines': N, 'bars_1m': N, ...}

    Raises:
        ValueError: If symbol is not in INSTRUMENT_CONFIGS.
    """
    if symbol not in INSTRUMENT_CONFIGS:
        supported = sorted(INSTRUMENT_CONFIGS.keys())
        raise ValueError(f"Unsupported symbol '{symbol}'. Supported symbols: {supported}")

    if archive is None:
        archive = RawDataArchive()
    if catalog is None:
        catalog = CatalogManager()

    counts: dict[str, Any] = {"new_klines": 0}
    failed_ranges: list[str] = []
    counts["failed_ranges"] = failed_ranges

    # Funding is independent of kline freshness: refresh it first so an
    # "already up to date" klines archive never leaves funding stale.
    try:
        counts["new_funding_rates"] = _update_funding_rates(symbol, archive, verbose)
    except (httpx.HTTPError, RestDownloadError) as e:
        logger.warning("%s: funding refresh failed (%s); continuing with klines", symbol, e)
        counts["new_funding_rates"] = 0
        counts["funding_failed"] = True  # a failed required stream: the shadow row freezes

    # The interval backfill is its own best-effort step: an archive without
    # intervals still charges every row, it just cannot see non-8h holes.
    try:
        counts["new_funding_intervals"] = ingest_funding_intervals(symbol, archive, verbose=verbose)
    except (httpx.HTTPError, MonthlyDownloadError) as e:
        logger.warning("%s: funding interval fetch failed (%s); continuing", symbol, e)
        counts["new_funding_intervals"] = 0

    # Get last timestamp from archive
    date_range = archive.get_date_range(symbol, "1m")
    if date_range is None:
        if verbose:
            print(f"{symbol}: no existing data, use 'ingest' command first")
        return counts

    last_timestamp = date_range[1]  # max open_time in ms
    now_ms = int(datetime.now(UTC).timestamp() * 1000)

    # Add 60000 (1 minute) to start after last candle
    start_time = last_timestamp + 60000

    if start_time >= now_ms:
        if verbose:
            print(f"{symbol}: already up to date")
        return counts

    if verbose:
        last_dt = datetime.fromtimestamp(last_timestamp / 1000, tz=UTC)
        print(f"{symbol}: fetching from {last_dt.isoformat()} to now...")

    # Download new klines via REST API
    new_klines = _fetch_rest_klines(symbol, "1m", start_time, now_ms, failed_ranges)
    new_klines = _drop_unclosed_klines(new_klines, now_ms)
    if new_klines:
        archive.insert_klines(symbol, "1m", new_klines, "binance_api")
        counts["new_klines"] = len(new_klines)
        if verbose:
            print(f"  Archived {len(new_klines)} new klines")
    else:
        if verbose:
            print("  No new klines available")
        return counts

    # Rebuild catalog from all archived klines
    instrument = create_instrument(symbol)
    catalog.write_instrument(instrument)

    all_klines = archive.get_klines(symbol, "1m")
    if not all_klines:
        return counts

    # Clear existing catalog data to avoid disjoint interval errors
    for interval in ["1m", *AGGREGATION_INTERVALS]:
        catalog.clear_bar_data(symbol, interval)

    # Convert to 1m bars
    bar_type_1m = get_bar_type(symbol, "1m")
    size_prec = instrument.size_precision
    price_prec = instrument.price_precision
    bars_1m = klines_to_bars(all_klines, instrument.id, bar_type_1m, size_prec, price_prec)
    catalog.write_bars(bars_1m)
    counts["bars_1m"] = len(bars_1m)
    if verbose:
        print(f"  Wrote {len(bars_1m)} 1m bars to catalog")

    # Aggregate to higher timeframes
    for interval in AGGREGATION_INTERVALS:
        minutes = _interval_to_minutes(interval)

        bar_type = get_bar_type(symbol, interval)
        agg_bars = aggregate_bars(bars_1m, bar_type, minutes, size_prec, price_prec)
        catalog.write_bars(agg_bars)
        counts[f"bars_{interval}"] = len(agg_bars)
        if verbose:
            print(f"  Wrote {len(agg_bars)} {interval} bars")

    return counts


def update_all(
    symbols: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """Update data for all symbols.

    Args:
        symbols: List of symbols to update. Uses SUPPORTED_SYMBOLS if not provided.
        verbose: Print progress messages.

    Returns:
        Dict mapping symbol to counts dict.
    """
    if symbols is None:
        symbols = SUPPORTED_SYMBOLS

    archive = RawDataArchive()
    catalog = CatalogManager()
    results: dict[str, dict[str, Any]] = {}

    session_id = archive.create_download_session(
        symbols=symbols,
        source="binance_api",
    )
    total_new = 0
    total_funding = 0

    try:
        # catalog writers serialise with the shadow snapshot (and each other)
        with catalog_write_lock(catalog._catalog_path):
            for symbol in symbols:
                if verbose:
                    print(f"\n{'=' * 50}")
                    print(f"Updating {symbol}")
                    print(f"{'=' * 50}")

                counts = update_symbol(symbol, archive=archive, catalog=catalog, verbose=verbose)
                results[symbol] = counts
                total_new += counts.get("new_klines", 0)
                total_funding += counts.get("new_funding_rates", 0)

        has_failures, summary = _failed_downloads_summary(results)
        archive.complete_download_session(
            session_id,
            klines_fetched=total_new,
            klines_inserted=total_new,
            funding_rates_fetched=total_funding,
            status="partial" if has_failures else "completed",
            error_message=summary,
        )
    except Exception as e:
        archive.complete_download_session(
            session_id,
            status="failed",
            error_message=str(e),
        )
        raise

    archive.close()

    if verbose:
        print(f"\n{'=' * 50}")
        print("UPDATE SUMMARY")
        print(f"{'=' * 50}")
        for sym, cnts in results.items():
            print(
                f"{sym}: {cnts.get('new_klines', 0)} new klines, "
                f"{cnts.get('new_funding_rates', 0)} new funding rates"
            )

    return results


def _failed_streams(results: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
    """Symbol -> required streams that failed in this update (a holed 1m range, funding)."""
    failed: dict[str, set[str]] = {}
    for symbol, counts in results.items():
        streams = {"klines"} if counts.get("failed_ranges") else set()
        if counts.get("funding_failed"):
            streams.add("funding")
        if streams:
            failed[symbol] = streams
    return failed


def _shadow_mark_starts(rows: list[dict[str, Any]]) -> dict[str, date]:
    """Symbol -> earliest warm-up start over the live shadow rows with liquidation on."""
    from vibe_quant.liquidation_settings import LiquidationConfigError, LiquidationSettings

    starts: dict[str, date] = {}
    for row in rows:
        if row["state"] not in ("tracking", "stalled"):
            continue
        try:
            if not LiquidationSettings.from_params(row["parameters"]).on:
                continue
        except LiquidationConfigError:  # the tracker stalls the row on its own run error
            logger.info("shadow row %s: invalid liquidation settings, no MARK refresh", row["id"])
            continue
        start = date.fromisoformat(str(row["run_start"]))
        for symbol in row["symbols"]:
            starts[symbol] = min(start, starts.get(symbol, start))
    return starts


def _refresh_shadow_mark(rows: list[dict[str, Any]], failed: dict[str, set[str]]) -> None:
    """Refresh MARK (archive, then catalog) for every symbol a tracked liquidation row reads.

    User 2026-11-23: the requested update refreshes it before the shadow hook; a failure
    adds ``mark`` to that symbol's failed streams (its rows stall), never aborts the update.
    """
    starts = _shadow_mark_starts(rows)
    if not starts:
        return
    archive = RawDataArchive()
    catalog = CatalogManager()
    try:
        for symbol in sorted(starts):
            try:
                counts = ingest_mark_klines(
                    symbol,
                    start_date=datetime.combine(starts[symbol], datetime.min.time(), tzinfo=UTC),
                    archive=archive,
                    verbose=True,
                )
                holes = [*counts.get("failed_months", []), *counts.get("failed_days", [])]
                if not holes:
                    with catalog_write_lock(catalog._catalog_path):
                        build_mark_bars(catalog, archive, symbol)
            except Exception as exc:  # one symbol's MARK stalls its rows, never the update
                logger.warning("MARK refresh failed for %s (%s)", symbol, type(exc).__name__)
                failed.setdefault(symbol, set()).add("mark")
                continue
            if holes:
                logger.warning("MARK refresh failed for %s: %s", symbol, ", ".join(holes))
                failed.setdefault(symbol, set()).add("mark")
    finally:
        archive.close()


def _run_shadow_hook(state_db: Path, results: dict[str, dict[str, Any]]) -> None:
    """Shadow update at the end of a requested data update (design arch-c7-2 §2 Hook).

    Runs after :func:`update_all` returned, outside the catalog lock. No shadow rows ->
    returns before importing the tracker (and the validation runner it loads). Errors are
    reported and swallowed: the data update's exit code never depends on the shadow update.
    """
    from vibe_quant.db.state_manager import ShadowUpdateRunning, StateManager

    state = StateManager(state_db)
    try:
        rows = state.list_shadow_strategies()
        if not rows:
            return
        failed = _failed_streams(results)
        _refresh_shadow_mark(rows, failed)
        from vibe_quant.shadow import notify, tracker

        try:
            tracker.update_all(
                state,
                today=datetime.now(UTC).date(),
                trigger="data-update",
                failed_streams=failed,
            )
        except ShadowUpdateRunning as exc:
            print(str(exc), file=sys.stderr)
            return
        notify.send_pending(state)
    except Exception as exc:  # the data update already succeeded or failed on its own
        logger.warning("shadow update failed (%s)", type(exc).__name__)
    finally:
        state.close()


def _format_days(days: list[str], limit: int = 10) -> str:
    """Join ISO dates for a report line, eliding the tail."""
    if len(days) <= limit:
        return ", ".join(days)
    return f"{', '.join(days[:limit])} ... and {len(days) - limit} more"


def _final_missing_days(
    archive: RawDataArchive,
    symbol: str,
    interval: str,
    day_range: tuple[date, date],
    *,
    mark: bool,
) -> list[str]:
    """Interior days still without rows in the archive, as ISO dates.

    Read from the archive itself (not from what one fill pass reports) so the
    count reflects every pass that ran: a day repaired by REST drops out, a hole
    that REST exposed is added.
    """
    table = "raw_mark_klines" if mark else "raw_klines"
    return [
        day.isoformat()
        for day in archive.missing_days(symbol, interval, day_range[0], day_range[1], table=table)
    ]


def _fill_interior_missing_days(
    symbol: str,
    interval: str,
    day_range: tuple[date, date],
    archive: RawDataArchive,
    counts: dict[str, Any],
    *,
    mark: bool,
    verbose: bool,
) -> list[str]:
    """Fetch a daily Vision zip for every interior day with no archived rows.

    A Binance Vision monthly zip can silently omit whole UTC days while still
    clearing the 0.9 ``has_month_coverage`` skip (SOLUSDT 1m 2022-02 ships
    25/28 days), so a re-ingest that only re-downloads months never repairs the
    hole. This pass runs after the monthly loop over the whole requested range
    — skipped months included — and is what actually fills it.

    A 404 (not published) or a download that failed after retries is *returned*,
    never raised: it is a hole, not a crashed run. The caller sets
    ``counts['missing_days']`` from the archive once every fill pass (this one
    and, for klines, REST) has run, and a nonempty list makes the session
    'partial'.

    Args:
        symbol: Trading symbol.
        interval: Candle interval.
        day_range: Requested (first_day, last_day); clamped to the symbol's own
            archived span by :meth:`RawDataArchive.missing_days`.
        archive: Raw data archive (inserts go to ``raw_klines`` or
            ``raw_mark_klines``).
        counts: Per-symbol counts dict, updated in place.
        mark: Select the mark-price table/downloader/counters.
        verbose: Print one line per filled/missing day.

    Returns:
        ISO dates that could not be filled, oldest first.
    """
    table = "raw_mark_klines" if mark else "raw_klines"
    source = "binance_vision_daily"
    fetched_key = "mark_klines_fetched" if mark else "klines_fetched"
    inserted_key = "mark_klines_inserted" if mark else "klines_inserted"

    missing = archive.missing_days(symbol, interval, day_range[0], day_range[1], table=table)
    unresolved: list[str] = []
    for day in missing:
        iso = day.isoformat()
        try:
            rows = (
                download_daily_mark_klines(symbol, interval, day)
                if mark
                else download_daily_klines(symbol, interval, day)
            )
        except DailyDownloadError as e:
            unresolved.append(iso)
            logger.error("%s %s daily %s: FAILED: %s", symbol, interval, iso, e)
            if verbose:
                print(f"    {iso}: daily zip FAILED ({e})")
            continue
        if not rows:
            unresolved.append(iso)
            if verbose:
                print(f"    {iso}: daily zip unavailable (404)")
            continue
        inserted = (
            archive.insert_mark_klines(symbol, interval, rows, source)
            if mark
            else archive.insert_klines(symbol, interval, rows, source)
        )
        counts[fetched_key] += len(rows)
        counts[inserted_key] += inserted
        if verbose:
            print(f"    {iso}: {len(rows)} rows from daily zip ({inserted} new)")

    if verbose and missing:
        print(
            f"  Interior missing days: {len(missing)} found, "
            f"{len(missing) - len(unresolved)} filled, {len(unresolved)} unresolved"
        )
    return unresolved


def _fill_unpublished_tail_days(
    symbol: str,
    interval: str,
    day_range: tuple[date, date],
    unavailable_months: list[str],
    archive: RawDataArchive,
    counts: dict[str, Any],
    *,
    verbose: bool,
    now: datetime,
) -> None:
    """Fetch a daily Vision zip for every zero-row day in the requested tail.

    A month Binance Vision has not published yet (the monthly zip 404s) leaves
    every one of its UTC days with zero archived rows, and the interior pass
    cannot repair them: :meth:`RawDataArchive.missing_days` clamps to the
    symbol's own last archived day, so days past the current coverage end are
    never reported missing. Vision publishes a daily zip roughly a day after a
    UTC day closes, so the days between the coverage end and yesterday are
    fetched here — *before* the interior pass, so ``counts['missing_days']``
    read from the archive below already reflects what this pass filled.

    The month does not have to be *unavailable* for its tail to be missing. A
    month whose daily rows already clear the 0.9 ``has_month_mark_coverage`` bar
    is *skipped*: its monthly zip is never requested, so the month never reaches
    ``unavailable_months``, and its closed tail days still have to come from the
    daily zips. Two day sets are therefore walked, whatever the month's status:

    * every day after the symbol's last archived day, up to yesterday, and
    * every day of a month in *unavailable_months* — interior holes included,
      since those days can sit *before* the coverage end.

    Both are bounded by the requested range and by yesterday: the current UTC day
    is never requested (it is still forming). A day that already holds a row is
    skipped, so a re-run is idempotent (the archive INSERT OR IGNORE on
    ``(symbol, interval, open_time)`` also drops duplicates).

    A 404 (day not published yet) is recorded in ``counts['unpublished_days']``
    and a download that failed after retries in ``counts['failed_days']`` (both
    ISO dates, oldest first); neither is raised — a hole, not a crashed run.

    Args:
        symbol: Trading symbol.
        interval: Candle interval.
        day_range: Requested (first_day, last_day).
        unavailable_months: "YYYY-MM" months whose monthly zip was a 404.
        archive: Raw data archive (inserts go to ``raw_mark_klines``).
        counts: Per-symbol counts dict, updated in place.
        verbose: Print one line per requested day.
        now: Reference "today"; the current UTC day is skipped.
    """
    import calendar

    source = "binance_vision_daily"
    last_full_day = now.astimezone(UTC).date() - timedelta(days=1)
    start_day, end_day = day_range
    hi = min(end_day, last_full_day)

    coverage = archive.get_mark_date_range(symbol, interval)
    if coverage is None:
        # Nothing archived at all: the whole requested tail is missing.
        tail_lo = start_day
    else:
        last_archived_day = datetime.fromtimestamp(coverage[1] / 1000, tz=UTC).date()
        tail_lo = max(start_day, last_archived_day + timedelta(days=1))

    days: set[date] = set()
    day = tail_lo
    while day <= hi:
        days.add(day)
        day += timedelta(days=1)

    for month_key in unavailable_months:
        year, month = (int(part) for part in month_key.split("-"))
        month_days = calendar.monthrange(year, month)[1]
        lo = max(date(year, month, 1), start_day)
        month_hi = min(date(year, month, month_days), end_day, last_full_day)
        day = lo
        while day <= month_hi:
            days.add(day)
            day += timedelta(days=1)

    unpublished: list[str] = counts["unpublished_days"]
    failed: list[str] = counts["failed_days"]
    requested = 0
    filled = 0
    for day in sorted(days):
        iso = day.isoformat()
        day_start_ms = int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)
        has_rows = (
            archive.conn.execute(
                "SELECT 1 FROM raw_mark_klines WHERE symbol = ? AND interval = ? "
                "AND open_time >= ? AND open_time < ? LIMIT 1",
                (symbol, interval, day_start_ms, day_start_ms + _DAY_MS),
            ).fetchone()
            is not None
        )
        if has_rows:
            continue
        requested += 1
        try:
            rows = download_daily_mark_klines(symbol, interval, day)
        except DailyDownloadError as e:
            failed.append(iso)
            logger.error("%s %s daily %s: FAILED: %s", symbol, interval, iso, e)
            if verbose:
                print(f"    {iso}: daily zip FAILED ({e})")
            continue
        if not rows:
            unpublished.append(iso)
            if verbose:
                print(f"    {iso}: daily zip unavailable (404)")
            continue
        inserted = archive.insert_mark_klines(symbol, interval, rows, source)
        counts["mark_klines_fetched"] += len(rows)
        counts["mark_klines_inserted"] += inserted
        filled += 1
        if verbose:
            print(f"    {iso}: {len(rows)} rows from daily zip ({inserted} new)")

    if verbose and requested:
        print(
            f"  Unpublished-month tail: {requested} days requested, {filled} filled, "
            f"{len(unpublished)} unpublished, {len(failed)} failed"
        )


def ingest_symbol(
    symbol: str,
    years: int = 2,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    archive: RawDataArchive | None = None,
    catalog: CatalogManager | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Ingest historical data for a single symbol.

    Downloads data from Binance Vision, stores in archive, and writes to catalog.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT').
        years: Number of years of history (used if start_date not given).
        start_date: Explicit start date (overrides years).
        end_date: Explicit end date (defaults to now).
        archive: Raw data archive (created if not provided).
        catalog: Catalog manager (created if not provided).
        verbose: Print progress messages.

    Returns:
        Dict with counts: {'klines': N, 'bars_1m': N, 'bars_5m': N, ...} plus
        ``failed_months`` / ``unavailable_months`` (lists of "YYYY-MM"):
        failed = download errored after retries (a hole to re-run), unavailable
        = Binance Vision has no archive (404), not an error.
    """
    if symbol not in INSTRUMENT_CONFIGS:
        supported = sorted(INSTRUMENT_CONFIGS.keys())
        raise ValueError(f"Unsupported symbol '{symbol}'. Supported symbols: {supported}")

    if archive is None:
        archive = RawDataArchive()
    if catalog is None:
        catalog = CatalogManager()

    counts: dict[str, Any] = {"klines_fetched": 0, "klines_inserted": 0}
    failed_months: list[str] = []
    unavailable_months: list[str] = []
    failed_ranges: list[str] = []

    if start_date is not None:
        effective_end = end_date or datetime.now(UTC)
        months = get_months_in_range(start_date, effective_end)
        range_start_day = start_date.date()
    else:
        months = get_years_months_to_download(years)
        effective_end = datetime.now(UTC)
        range_start_day = (effective_end - timedelta(days=365 * years)).date()
    day_range = (range_start_day, effective_end.date())

    # Smart skip: check which months already have sufficient coverage
    skipped: list[tuple[int, int]] = []
    to_download: list[tuple[int, int]] = []
    for year, month in months:
        if archive.has_month_coverage(symbol, "1m", year, month):
            skipped.append((year, month))
        else:
            to_download.append((year, month))

    if verbose:
        print(
            f"Downloading {symbol}: {len(to_download)} months to download, "
            f"{len(skipped)} months skipped (already archived)"
        )

    # Download and archive 1m klines (only missing/partial months)
    for year, month in to_download:
        try:
            klines = download_monthly_klines(symbol, "1m", year, month)
        except MonthlyDownloadError as e:
            failed_months.append(f"{year}-{month:02d}")
            logger.error("%s %d-%02d: monthly download FAILED: %s", symbol, year, month, e)
            if verbose:
                print(f"  {year}-{month:02d}: FAILED ({e})")
            continue
        if klines:
            inserted = archive.insert_klines(symbol, "1m", klines, "binance_vision")
            counts["klines_fetched"] += len(klines)
            counts["klines_inserted"] += inserted
            if verbose:
                print(f"  {year}-{month:02d}: {len(klines)} klines ({inserted} new)")
        else:
            unavailable_months.append(f"{year}-{month:02d}")
            if verbose:
                print(f"  {year}-{month:02d}: unavailable (not published)")

    # Repair interior days the monthly zips omitted (also covers skipped months).
    # The final ``missing_days`` is set after the REST pass below: REST can both
    # repair a day the daily zip lacked and expose a new interior hole.
    _fill_interior_missing_days(
        symbol, "1m", day_range, archive, counts, mark=False, verbose=verbose
    )

    counts["months_skipped"] = len(skipped)
    counts["months_downloaded"] = len(to_download) - len(failed_months)
    counts["failed_months"] = failed_months
    counts["unavailable_months"] = unavailable_months
    counts["failed_ranges"] = failed_ranges

    if verbose:
        print(
            f"Total from Vision: {counts['klines_fetched']} fetched, "
            f"{counts['klines_inserted']} new"
        )

    # Fill current incomplete month via REST API
    rest_start: datetime | None = None
    if months:
        last_year, last_month = months[-1]
        # Start of month after last complete month
        if last_month == 12:
            rest_start = datetime(last_year + 1, 1, 1, tzinfo=UTC)
        else:
            rest_start = datetime(last_year, last_month + 1, 1, tzinfo=UTC)
    elif start_date is not None:
        # Range lies inside one (incomplete) month: no monthly zips apply
        rest_start = start_date

    if rest_start is not None:
        rest_start_ms = int(rest_start.timestamp() * 1000)
        rest_end_ms = int(effective_end.timestamp() * 1000)

        if rest_start_ms < rest_end_ms:
            if verbose:
                print(
                    f"Fetching {rest_start.strftime('%Y-%m-%d')} to "
                    f"{effective_end.strftime('%Y-%m-%d')} via REST API..."
                )
            recent = _fetch_rest_klines(symbol, "1m", rest_start_ms, rest_end_ms, failed_ranges)
            recent = _drop_unclosed_klines(recent, int(datetime.now(UTC).timestamp() * 1000))
            if recent:
                inserted = archive.insert_klines(symbol, "1m", recent, "binance_api")
                counts["klines_fetched"] += len(recent)
                counts["klines_inserted"] += inserted
                if verbose:
                    print(f"  REST API: {len(recent)} fetched ({inserted} new)")

    # Final unresolved interior days, read from the archive after ALL fill passes.
    counts["missing_days"] = _final_missing_days(archive, symbol, "1m", day_range, mark=False)

    # Create and write instrument
    instrument = create_instrument(symbol)
    catalog.write_instrument(instrument)
    if verbose:
        print(f"Wrote instrument: {instrument.id}")

    # Get all klines from archive
    all_klines = archive.get_klines(symbol, "1m")
    if not all_klines:
        if verbose:
            print("No klines to process")
        return counts

    # Clear existing catalog data to avoid disjoint interval errors
    # (archive is source of truth; catalog rebuilt from full archive each time)
    for interval in ["1m", *AGGREGATION_INTERVALS]:
        catalog.clear_bar_data(symbol, interval)

    # Convert to 1m bars and write
    bar_type_1m = get_bar_type(symbol, "1m")
    size_prec = instrument.size_precision
    price_prec = instrument.price_precision
    bars_1m = klines_to_bars(all_klines, instrument.id, bar_type_1m, size_prec, price_prec)
    catalog.write_bars(bars_1m)
    counts["bars_1m"] = len(bars_1m)
    if verbose:
        print(f"Wrote {len(bars_1m)} 1m bars")

    # Aggregate to higher timeframes
    for interval in AGGREGATION_INTERVALS:
        bar_type = get_bar_type(symbol, interval)
        minutes = _interval_to_minutes(interval)

        agg_bars = aggregate_bars(bars_1m, bar_type, minutes, size_prec, price_prec)
        catalog.write_bars(agg_bars)
        counts[f"bars_{interval}"] = len(agg_bars)
        if verbose:
            print(f"Wrote {len(agg_bars)} {interval} bars")

    return counts


def ingest_mark_klines(
    symbol: str,
    start_date: datetime,
    end_date: datetime | None = None,
    archive: RawDataArchive | None = None,
    verbose: bool = True,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Ingest Binance mark-price 1m klines from Vision archives.

    Liquidation triggers on the mark price, so it is archived in its own
    ``raw_mark_klines`` table and never mixed into the ``raw_klines`` path that
    feeds the catalog. Monthly Vision zips are the primary source; there is no
    REST fill. A month Vision has not published yet 404s (unavailable), and its
    trailing days are then fetched one by one from the DAILY zips — Vision
    publishes those about a day after each UTC day closes — so coverage tracks
    the current month instead of stopping at the last monthly zip.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT').
        start_date: Start date (required); earlier months are not requested.
        end_date: End date (defaults to now).
        archive: Raw data archive (created if not provided).
        verbose: Print progress messages.
        now: Reference "today" (defaults to the current UTC time); the current
            UTC day is never requested from a daily zip.

    Returns:
        Dict with counts: ``mark_klines_fetched``, ``mark_klines_inserted``,
        ``months_skipped``, ``months_downloaded`` plus the ``failed_months`` /
        ``unavailable_months`` lists ("YYYY-MM") on the same convention as
        :func:`ingest_symbol`: failed = download errored after retries (a hole
        to re-run), unavailable = not published (404), not an error. The daily
        tail pass adds ISO-date lists ``unpublished_days`` (day zip 404'd) and
        ``failed_days`` (day zip errored after retries); ``missing_days`` is the
        interior days still empty after every pass.

    Raises:
        ValueError: If symbol is not in INSTRUMENT_CONFIGS.
    """
    if symbol not in INSTRUMENT_CONFIGS:
        supported = sorted(INSTRUMENT_CONFIGS.keys())
        raise ValueError(f"Unsupported symbol '{symbol}'. Supported symbols: {supported}")

    if archive is None:
        archive = RawDataArchive()

    now = now or datetime.now(UTC)

    counts: dict[str, Any] = {"mark_klines_fetched": 0, "mark_klines_inserted": 0}
    failed_months: list[str] = []
    unavailable_months: list[str] = []

    # Every UTC month touching [start, end], including the one holding `end`:
    # unlike raw klines there is no REST fill for the trailing partial month, so
    # an unpublished final month must surface as "unavailable", not vanish.
    # Endpoints are normalised to UTC (Vision months are UTC; naive = UTC) and
    # snapped to midnight month starts, else a late start time-of-day (or a
    # non-UTC offset) drops the inclusive end month from generate_month_range.
    def _utc(dt: datetime) -> datetime:
        return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)

    utc_start = _utc(start_date)
    utc_end = _utc(end_date or now)
    months = list(
        generate_month_range(
            utc_start.replace(day=1, hour=0, minute=0, second=0, microsecond=0), utc_end
        )
    )
    day_range = (utc_start.date(), utc_end.date())

    # Smart skip: months already carrying most of their 1m mark price are done.
    skipped: list[tuple[int, int]] = []
    to_download: list[tuple[int, int]] = []
    for year, month in months:
        if archive.has_month_mark_coverage(symbol, "1m", year, month):
            skipped.append((year, month))
        else:
            to_download.append((year, month))

    if verbose:
        print(
            f"Downloading {symbol} mark-price 1m klines: {len(to_download)} months to "
            f"download, {len(skipped)} months skipped (already archived)"
        )

    for year, month in to_download:
        try:
            klines = download_monthly_mark_klines(symbol, "1m", year, month)
        except MonthlyDownloadError as e:
            failed_months.append(f"{year}-{month:02d}")
            logger.error("%s %d-%02d: mark-price download FAILED: %s", symbol, year, month, e)
            if verbose:
                print(f"  {year}-{month:02d}: FAILED ({e})")
            continue
        if klines:
            inserted = archive.insert_mark_klines(symbol, "1m", klines, "binance_vision")
            counts["mark_klines_fetched"] += len(klines)
            counts["mark_klines_inserted"] += inserted
            if verbose:
                print(f"  {year}-{month:02d}: {len(klines)} mark klines ({inserted} new)")
        else:
            unavailable_months.append(f"{year}-{month:02d}")
            if verbose:
                coverage = archive.get_mark_date_range(symbol, "1m")
                coverage_end = (
                    datetime.fromtimestamp(coverage[1] / 1000, tz=UTC).strftime("%Y-%m-%d")
                    if coverage
                    else "none archived"
                )
                print(
                    f"  {year}-{month:02d}: unavailable (not yet published) - "
                    f"mark coverage ends {coverage_end}"
                )

    # Fill the missing tail from the daily zips — every zero-row day after the
    # coverage end (a 404'd *or* 90%-skip-passed current month) plus the days of
    # a 404'd month — before the interior pass, so counts['missing_days'] (read
    # from the archive below) already reflects the days this pass filled.
    counts["unpublished_days"] = []
    counts["failed_days"] = []
    _fill_unpublished_tail_days(
        symbol,
        "1m",
        day_range,
        unavailable_months,
        archive,
        counts,
        verbose=verbose,
        now=now,
    )
    if verbose and unavailable_months:
        coverage = archive.get_mark_date_range(symbol, "1m")
        coverage_end = (
            datetime.fromtimestamp(coverage[1] / 1000, tz=UTC).strftime("%Y-%m-%d")
            if coverage
            else "none archived"
        )
        print(f"  mark coverage now ends {coverage_end}")

    # Repair interior days the monthly mark zips omitted (skipped months included).
    _fill_interior_missing_days(
        symbol, "1m", day_range, archive, counts, mark=True, verbose=verbose
    )
    counts["missing_days"] = _final_missing_days(archive, symbol, "1m", day_range, mark=True)

    counts["months_skipped"] = len(skipped)
    counts["months_downloaded"] = len(to_download) - len(failed_months)
    counts["failed_months"] = failed_months
    counts["unavailable_months"] = unavailable_months
    return counts


def ingest_funding_intervals(
    symbol: str,
    archive: RawDataArchive,
    *,
    verbose: bool = False,
    now: datetime | None = None,
) -> int:
    """Backfill Binance Vision funding intervals for an existing archive.

    Vision publishes one monthly ``fundingRate`` file per symbol; the stored
    ``funding_interval_hours`` is what tells a hole on a 2h/4h symbol apart
    from a legal 8h gap.

    The months fetched are every UTC calendar month that holds at least one
    archived ``raw_funding_rates`` row for *symbol* but no
    ``raw_funding_intervals`` row yet, strictly before the current month. That
    fills months older than the newest interval row and mid-range holes alike,
    and a month with no archived rates is never fetched. The current month is
    skipped because Vision publishes a month's file only after it ends.

    A month that 404s (not published) inserts nothing and therefore still has
    no interval rows, so the next run retries it.

    Args:
        symbol: Trading symbol.
        archive: Raw data archive.
        verbose: Print progress messages.
        now: Reference "today" (defaults to the current UTC time).

    Returns:
        Number of interval rows inserted. 0 when the archive holds no funding
        rates at all (nothing to date a fetch from).
    """
    now = now or datetime.now(UTC)
    current_month = (now.year, now.month)

    def _archived_months(table: str) -> set[tuple[int, int]]:
        keys = archive.conn.execute(
            f"SELECT DISTINCT strftime('%Y-%m', funding_time / 1000, 'unixepoch') "
            f"FROM {table} WHERE symbol = ?",
            (symbol,),
        ).fetchall()
        months: set[tuple[int, int]] = set()
        for (key,) in keys:
            if key is None:
                continue
            year, month = str(key).split("-")
            months.add((int(year), int(month)))
        return months

    missing = sorted(
        month
        for month in _archived_months("raw_funding_rates")
        - _archived_months("raw_funding_intervals")
        if month < current_month
    )

    inserted_total = 0
    for year, month in missing:
        rows = download_monthly_funding_intervals(symbol, year, month)
        if rows is None:  # not published; retried on the next run
            continue
        inserted = archive.insert_funding_intervals(symbol, rows, "binance_vision")
        inserted_total += inserted
        if verbose:
            print(f"  {year}-{month:02d}: {inserted} new funding intervals")
    return inserted_total


def ingest_funding_rates(
    symbol: str,
    years: int = 2,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    archive: RawDataArchive | None = None,
    verbose: bool = True,
) -> int:
    """Ingest funding rate history for a symbol.

    Args:
        symbol: Trading symbol.
        years: Number of years of history (used if start_date not given).
        start_date: Explicit start date (overrides years).
        end_date: Explicit end date (defaults to now).
        archive: Raw data archive.
        verbose: Print progress messages.

    Returns:
        Number of funding rates archived.
    """
    if archive is None:
        archive = RawDataArchive()

    now = datetime.now(UTC)
    if start_date is not None:
        start_time = int(start_date.timestamp() * 1000)
    else:
        start_time = int((now.timestamp() - years * 365 * 24 * 3600) * 1000)
    end_time = int((end_date or now).timestamp() * 1000)

    if verbose:
        print(f"Downloading {symbol} funding rates...")

    rates = download_funding_rates(symbol, start_time, end_time)
    if rates:
        archive.insert_funding_rates(symbol, rates, "binance_api")
        if verbose:
            print(f"Archived {len(rates)} funding rates")
        # A Vision failure must never lose the rates that just landed.
        try:
            ingest_funding_intervals(symbol, archive, verbose=verbose)
        except (httpx.HTTPError, MonthlyDownloadError) as e:
            logger.warning("%s: funding interval fetch failed (%s); continuing", symbol, e)
        return len(rates)

    if verbose:
        print("No funding rates downloaded")
    return 0


def ingest_detail_data(
    symbol: str,
    interval: str = "5s",
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    archive: RawDataArchive | None = None,
    catalog: CatalogManager | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Ingest sub-minute detail data for a symbol via REST API.

    Downloads 1s or 5s kline data from Binance REST API, archives it,
    and writes to ParquetDataCatalog. Used for sub-bar fill simulation
    in validation backtests.

    Note: Binance Vision monthly archives don't include sub-minute data,
    so this always uses the REST API. Download times are significantly
    longer than 1m data (12x for 5s, 60x for 1s).

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT').
        interval: Sub-minute interval ('1s' or '5s').
        start_date: Start date (required).
        end_date: End date (defaults to now).
        archive: Raw data archive (created if not provided).
        catalog: Catalog manager (created if not provided).
        verbose: Print progress messages.

    Returns:
        Dict with counts: {'klines_fetched': N, 'klines_inserted': N, 'bars': N}
    """
    if interval not in ("1s", "5s"):
        msg = f"Detail interval must be '1s' or '5s', got '{interval}'"
        raise ValueError(msg)

    if symbol not in INSTRUMENT_CONFIGS:
        supported = sorted(INSTRUMENT_CONFIGS.keys())
        raise ValueError(f"Unsupported symbol '{symbol}'. Supported: {supported}")

    if start_date is None:
        msg = "start_date is required for detail data ingestion"
        raise ValueError(msg)

    if archive is None:
        archive = RawDataArchive()
    if catalog is None:
        catalog = CatalogManager()

    effective_end = end_date or datetime.now(UTC)
    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(effective_end.timestamp() * 1000)
    counts: dict[str, Any] = {"klines_fetched": 0, "klines_inserted": 0, "bars": 0}

    if verbose:
        secs = _interval_to_seconds(interval)
        est_candles = (end_ms - start_ms) // (secs * 1000)
        est_requests = est_candles // 1000 + 1
        print(
            f"Downloading {symbol} {interval} data: ~{est_candles:,} candles, "
            f"~{est_requests:,} API requests"
        )

    # Download via REST API (paginated)
    failed_ranges: list[str] = []
    counts["failed_ranges"] = failed_ranges
    klines = _fetch_rest_klines(symbol, interval, start_ms, end_ms, failed_ranges)
    klines = _drop_unclosed_klines(klines, int(datetime.now(UTC).timestamp() * 1000))
    if not klines:
        if verbose:
            print(f"  No {interval} data available")
        return counts

    # Archive raw data
    inserted = archive.insert_klines(symbol, interval, klines, "binance_api")
    counts["klines_fetched"] = len(klines)
    counts["klines_inserted"] = inserted
    if verbose:
        print(f"  Archived {len(klines):,} klines ({inserted:,} new)")

    # Write to catalog
    instrument = create_instrument(symbol)
    catalog.write_instrument(instrument)

    # Clear existing detail data to avoid disjoint interval errors
    catalog.clear_bar_data(symbol, interval)

    # Get all archived klines for this interval (may include prior downloads)
    all_klines = archive.get_klines(symbol, interval)
    bar_type = get_bar_type(symbol, interval)
    size_prec = instrument.size_precision
    price_prec = instrument.price_precision
    bars = klines_to_bars(all_klines, instrument.id, bar_type, size_prec, price_prec)
    catalog.write_bars(bars)
    counts["bars"] = len(bars)
    if verbose:
        print(f"  Wrote {len(bars):,} {interval} bars to catalog")

    return counts


def ingest_all(
    symbols: list[str] | None = None,
    years: int = 2,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """Ingest data for all symbols.

    Args:
        symbols: List of symbols to ingest. Uses SUPPORTED_SYMBOLS if not provided.
        years: Number of years of history (used if start_date not given).
        start_date: Explicit start date (overrides years).
        end_date: Explicit end date (defaults to now).
        verbose: Print progress messages.

    Returns:
        Dict mapping symbol to counts dict.
    """
    if symbols is None:
        symbols = SUPPORTED_SYMBOLS

    archive = RawDataArchive()
    catalog = CatalogManager()
    results: dict[str, dict[str, Any]] = {}

    # Create audit session
    session_id = archive.create_download_session(
        symbols=symbols,
        source="mixed",
        start_date=start_date.isoformat() if start_date else None,
        end_date=end_date.isoformat() if end_date else None,
    )

    total_klines_fetched = 0
    total_klines_inserted = 0
    total_funding = 0

    try:
        with catalog_write_lock(catalog._catalog_path):
            for symbol in symbols:
                if verbose:
                    print(f"\n{'=' * 50}")
                    print(f"Processing {symbol}")
                    print(f"{'=' * 50}")

                counts = ingest_symbol(
                    symbol,
                    years=years,
                    start_date=start_date,
                    end_date=end_date,
                    archive=archive,
                    catalog=catalog,
                    verbose=verbose,
                )
                funding_count = ingest_funding_rates(
                    symbol,
                    years=years,
                    start_date=start_date,
                    end_date=end_date,
                    archive=archive,
                    verbose=verbose,
                )
                counts["funding_rates"] = funding_count
                results[symbol] = counts
                total_klines_fetched += counts.get("klines_fetched", 0)
                total_klines_inserted += counts.get("klines_inserted", 0)
                total_funding += funding_count

        has_failures, summary = _failed_downloads_summary(results)
        archive.complete_download_session(
            session_id,
            klines_fetched=total_klines_fetched,
            klines_inserted=total_klines_inserted,
            funding_rates_fetched=total_funding,
            status="partial" if has_failures else "completed",
            error_message=summary,
        )
    except Exception as e:
        archive.complete_download_session(
            session_id,
            status="failed",
            error_message=str(e),
        )
        raise

    archive.close()

    if verbose:
        _print_summary(
            results,
            archive._db_path,
            catalog._catalog_path,
            total_klines_fetched,
            total_klines_inserted,
            total_funding,
        )

    return results


def _format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _dir_size(path: Path) -> int:
    """Get total size of a directory recursively."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _print_summary(
    results: dict[str, dict[str, Any]],
    archive_path: Path | None,
    catalog_path: Path,
    total_fetched: int,
    total_inserted: int,
    total_funding: int,
) -> None:
    """Print formatted download summary."""
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    for sym, cnts in results.items():
        fetched = cnts.get("klines_fetched", 0)
        inserted = cnts.get("klines_inserted", 0)
        bars_1m = cnts.get("bars_1m", 0)
        funding = cnts.get("funding_rates", 0)
        skipped = cnts.get("months_skipped", 0)
        downloaded = cnts.get("months_downloaded", 0)

        print(f"\n  {sym}:")
        print(f"    Months: {downloaded} downloaded, {skipped} skipped")
        if cnts.get("unavailable_months"):
            print(f"    Unavailable (404): {', '.join(cnts['unavailable_months'])}")
        if cnts.get("failed_months"):
            print(f"    FAILED months (re-run ingest): {', '.join(cnts['failed_months'])}")
        if cnts.get("failed_ranges"):
            print(f"    FAILED REST ranges (re-run): {', '.join(cnts['failed_ranges'])}")
        if cnts.get("missing_days"):
            print(
                f"    UNFILLED interior days (re-run ingest): {_format_days(cnts['missing_days'])}"
            )
        print(f"    Klines: {fetched:,} fetched, {inserted:,} new")
        print(f"    Archive total: {bars_1m:,} 1m bars")

        # Higher timeframes
        htf = []
        for tf in AGGREGATION_INTERVALS:
            key = f"bars_{tf}"
            if key in cnts:
                htf.append(f"{cnts[key]:,} {tf}")
        if htf:
            print(f"    Aggregated: {', '.join(htf)}")

        if funding:
            print(f"    Funding rates: {funding:,}")

    # File sizes
    archive_size = archive_path.stat().st_size if archive_path and archive_path.exists() else 0
    catalog_size = _dir_size(catalog_path)

    print("\n  Storage:")
    print(f"    Archive DB:  {_format_size(archive_size)}")
    print(f"    Catalog dir: {_format_size(catalog_size)}")
    print(f"    Total:       {_format_size(archive_size + catalog_size)}")

    # Totals if multiple symbols
    if len(results) > 1:
        print(
            f"\n  Totals: {total_fetched:,} fetched, {total_inserted:,} new, "
            f"{total_funding:,} funding rates"
        )

    print()


def _iso_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat()


def _research_status(archive: RawDataArchive, symbol: str) -> dict[str, dict[str, Any]]:
    """1h/1d kline and funding ranges of a research-tier (no 1m) symbol."""
    research: dict[str, dict[str, Any]] = {}
    for interval in ("1h", "1d"):
        rng = archive.get_date_range(symbol, interval)
        if rng:
            research[interval] = {
                "count": archive.get_kline_count(symbol, interval),
                "start": _iso_ms(rng[0]),
                "end": _iso_ms(rng[1]),
            }
    funding = archive.get_funding_range(symbol)
    if research and funding:
        research["funding"] = {
            "count": funding[0],
            "start": _iso_ms(funding[1]),
            "end": _iso_ms(funding[2]),
        }
    return research


def get_status(
    archive_path: Path | None = None,
    catalog_path: Path | None = None,
) -> dict[str, Any]:
    """Get status of data archive and catalog.

    Args:
        archive_path: Path to archive database.
        catalog_path: Path to catalog directory.

    Returns:
        Status dict with symbol info.
    """
    from vibe_quant.data.archive import DEFAULT_ARCHIVE_PATH
    from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH

    archive = RawDataArchive(archive_path or DEFAULT_ARCHIVE_PATH)
    catalog = CatalogManager(catalog_path or DEFAULT_CATALOG_PATH)

    status: dict[str, Any] = {"symbols": {}}

    metrics_symbols = [
        r[0] for r in archive.conn.execute("SELECT DISTINCT symbol FROM raw_metrics_files")
    ]
    for symbol in sorted({*archive.get_symbols(), *metrics_symbols}):
        sym_status: dict[str, Any] = {}

        # Archive info
        date_range = archive.get_date_range(symbol, "1m")
        if date_range:
            start_dt = datetime.fromtimestamp(date_range[0] / 1000, tz=UTC)
            end_dt = datetime.fromtimestamp(date_range[1] / 1000, tz=UTC)
            sym_status["archive"] = {
                "klines_1m": archive.get_kline_count(symbol, "1m"),
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
            }
        else:
            research = _research_status(archive, symbol)
            if research:
                sym_status["research"] = research

        # Detail interval archive info (sub-minute data)
        for detail_interval in ["1s", "5s"]:
            detail_count = archive.get_kline_count(symbol, detail_interval)
            if detail_count > 0:
                detail_range = archive.get_date_range(symbol, detail_interval)
                if detail_range:
                    d_start = datetime.fromtimestamp(detail_range[0] / 1000, tz=UTC)
                    d_end = datetime.fromtimestamp(detail_range[1] / 1000, tz=UTC)
                    if "detail" not in sym_status:
                        sym_status["detail"] = {}
                    sym_status["detail"][detail_interval] = {
                        "klines": detail_count,
                        "start": d_start.isoformat(),
                        "end": d_end.isoformat(),
                    }

        # Catalog info
        for interval in ["1s", "5s", "1m", "5m", "15m", "1h", "4h"]:
            bar_count = catalog.get_bar_count(symbol, interval)
            if bar_count > 0:
                if "catalog" not in sym_status:
                    sym_status["catalog"] = {}
                sym_status["catalog"][f"bars_{interval}"] = bar_count

        metrics = archive.get_metrics_range(symbol)
        if metrics:
            sym_status["metrics"] = {
                "rows": metrics[0],
                "start": _iso_ms(metrics[1]),
                "end": _iso_ms(metrics[2]),
            }

        status["symbols"][symbol] = sym_status

    from vibe_quant.data.dvol import DVOL_RESOLUTION_S

    status["dvol"] = {
        currency: {"candles": count, "start": _iso_ms(lo), "end": _iso_ms(hi)}
        for currency, count, lo, hi in archive.conn.execute(
            "SELECT currency, COUNT(*), MIN(open_time), MAX(open_time) FROM raw_dvol"
            " WHERE resolution_s = ? GROUP BY currency ORDER BY currency",
            (DVOL_RESOLUTION_S,),
        )
    }
    archive.close()
    return status


def rebuild_from_archive(
    catalog_path: Path | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, int]]:
    """Rebuild ParquetDataCatalog from raw SQLite archive.

    Deletes existing catalog and recreates it from archived klines, holding
    ``catalog_write_lock`` throughout.

    Args:
        catalog_path: Path to catalog directory. Uses default if not specified.
        verbose: Print progress messages.

    Returns:
        Dict mapping symbol to bar counts.

    Raises:
        OSError: ``catalog_path`` is a symlink.
    """
    catalog_path = catalog_path or DEFAULT_CATALOG_PATH
    # a worktree's data/catalog links to the shared catalog other agents read unlocked
    if catalog_path.is_symlink():
        raise OSError(f"refusing to rebuild through symlink {catalog_path}")
    with catalog_write_lock(catalog_path):
        return _rebuild_locked(catalog_path, verbose)


def _rebuild_locked(catalog_path: Path, verbose: bool) -> dict[str, dict[str, int]]:
    """:func:`rebuild_from_archive`'s body; the caller holds ``catalog_write_lock``."""
    # Delete the existing catalog's contents. The lock file stays: every holder must
    # lock the same inode, or a snapshot could start mid-rebuild.
    if verbose:
        print(f"Deleting existing catalog at {catalog_path}")
    for child in catalog_path.iterdir():
        if child.name == WRITE_LOCK_NAME:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()

    archive = RawDataArchive()
    catalog = CatalogManager(catalog_path)
    results: dict[str, dict[str, int]] = {}

    symbols = archive.get_symbols()
    if not symbols:
        if verbose:
            print("No symbols in archive")
        archive.close()
        return results

    if verbose:
        print(f"Rebuilding catalog for {len(symbols)} symbols...")

    # MARK bars are rebuilt for the same symbol set (driven by raw_klines); a
    # symbol with mark rows but no raw_klines has no LAST series to stand next
    # to, so it is skipped here rather than silently given a half catalog.
    mark_only = sorted(
        {row[0] for row in archive.conn.execute("SELECT DISTINCT symbol FROM raw_mark_klines")}
        - set(symbols)
    )
    if mark_only:
        print(f"Skipping MARK-only symbols (no raw_klines): {', '.join(mark_only)}")

    for symbol in symbols:
        if verbose:
            print(f"\n{'=' * 50}")
            print(f"Processing {symbol}")
            print(f"{'=' * 50}")

        counts: dict[str, int] = {}

        # Create and write instrument
        try:
            instrument = create_instrument(symbol)
        except KeyError:
            if verbose:
                print(f"Skipping {symbol}: no instrument config (unknown symbol)")
            results[symbol] = counts
            continue
        catalog.write_instrument(instrument)
        if verbose:
            print(f"Wrote instrument: {instrument.id}")

        # Get all klines from archive
        all_klines = archive.get_klines(symbol, "1m")
        if not all_klines:
            if verbose:
                print("No klines to process")
            results[symbol] = counts
            continue

        # Convert to 1m bars and write
        bar_type_1m = get_bar_type(symbol, "1m")
        size_prec = instrument.size_precision
        price_prec = instrument.price_precision
        bars_1m = klines_to_bars(all_klines, instrument.id, bar_type_1m, size_prec, price_prec)
        catalog.write_bars(bars_1m)
        counts["bars_1m"] = len(bars_1m)
        if verbose:
            print(f"Wrote {len(bars_1m)} 1m bars")

        # Aggregate to higher timeframes (same logic as ingest_symbol)
        for interval in AGGREGATION_INTERVALS:
            bar_type = get_bar_type(symbol, interval)
            minutes = _interval_to_minutes(interval)

            agg_bars = aggregate_bars(bars_1m, bar_type, minutes, size_prec, price_prec)
            catalog.write_bars(agg_bars)
            counts[f"bars_{interval}"] = len(agg_bars)
            if verbose:
                print(f"Wrote {len(agg_bars)} {interval} bars")

        # Rebuild detail (sub-minute) data if present in archive
        for detail_interval in ["1s", "5s"]:
            detail_klines = archive.get_klines(symbol, detail_interval)
            if detail_klines:
                catalog.clear_bar_data(symbol, detail_interval)
                detail_bar_type = get_bar_type(symbol, detail_interval)
                detail_bars = klines_to_bars(
                    detail_klines, instrument.id, detail_bar_type, size_prec, price_prec
                )
                catalog.write_bars(detail_bars)
                counts[f"bars_{detail_interval}"] = len(detail_bars)
                if verbose:
                    print(f"Wrote {len(detail_bars):,} {detail_interval} bars")

        # MARK-price 1m bars (liquidation trigger price) live beside the LAST
        # series and share the symbol's instrument.
        mark_written = build_mark_bars(catalog, archive, symbol)
        if mark_written:
            counts["mark_bars_1m"] = mark_written
            if verbose:
                print(f"Wrote {mark_written:,} mark 1m bars")

        results[symbol] = counts

    archive.close()

    if verbose:
        print(f"\n{'=' * 50}")
        print("REBUILD COMPLETE")
        print(f"{'=' * 50}")
        for sym, counts in results.items():
            print(f"{sym}: {counts}")

    return results


# ---------------------------------------------------------------- research tier
#
# Universe coins outside the full tier get 1h + 1d LAST klines and funding only:
# no 1m, no MARK (those stay per coin on demand through ``ingest`` /
# ``download-mark``). Archive only; the catalog is not written here.

RESEARCH_INTERVALS: tuple[str, ...] = ("1h", "1d")
_INTERVAL_MS = {"1h": 3_600_000, "1d": _DAY_MS}

# Less history than this is flagged "short history" in the plan.
SHORT_HISTORY_DAYS = 365

# Free disk that must remain after the estimated download.
DISK_HEADROOM_BYTES = 5 * 1024**3

# The research tier's own cap, independent of free disk (user ruling 2026-11-19:
# disk is limited): the estimate must fit, and measured growth stops the run.
RESEARCH_BUDGET_BYTES = 2 * 1024**3

# MEASURED 2026-11-22 on a tmp archive (50k rows each, both indexes included):
# raw_klines 205 B/row, raw_funding_rates 125 B, raw_funding_intervals 86 B.
_EST_ROW_BYTES = {"kline": 210, "funding": 130, "interval": 90}
# Funding settlements per day assumed for the estimate (4h symbols; 8h = 3).
_EST_FUNDING_PER_DAY = 6


def _archived_counts(archive_path: Path, symbol: str) -> dict[str, int]:
    """Archived row counts for *symbol*, read-only (never creates or migrates)."""
    import sqlite3

    counts = {"1m": 0, "1h": 0, "1d": 0, "funding": 0, "interval": 0}
    if not archive_path.exists():
        return counts
    con = sqlite3.connect(f"file:{archive_path}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"raw_klines", "raw_funding_rates", "raw_funding_intervals"} <= tables:
            return counts  # a file without the archive tables yet
        for interval in ("1m", "1h", "1d"):
            counts[interval] = con.execute(
                "SELECT COUNT(*) FROM raw_klines WHERE symbol = ? AND interval = ?",
                (symbol, interval),
            ).fetchone()[0]
        counts["funding"] = con.execute(
            "SELECT COUNT(*) FROM raw_funding_rates WHERE symbol = ?", (symbol,)
        ).fetchone()[0]
        counts["interval"] = con.execute(
            "SELECT COUNT(*) FROM raw_funding_intervals WHERE symbol = ?", (symbol,)
        ).fetchone()[0]
    finally:
        con.close()
    return counts


def plan_research_tier(
    symbols: list[str],
    listings: dict[str, int],
    archive_path: Path,
    now: datetime,
    *,
    include_full: bool = False,
) -> list[dict[str, Any]]:
    """Per universe symbol: tier, listing, history length and estimated bytes.

    ``tier`` is ``full`` (a SUPPORTED_SYMBOLS coin or one with archived 1m:
    skipped), ``research`` (to download) or ``unlisted`` (absent from
    exchangeInfo: cannot be dated, fails the run). ``include_full`` plans the
    full-tier coins as ``research`` too (their 1h/1d/funding; 1m and MARK are
    never touched). The estimate counts only the rows not archived yet.
    """
    now_ms = int(now.timestamp() * 1000)
    plan: list[dict[str, Any]] = []
    for symbol in symbols:
        have = _archived_counts(archive_path, symbol)
        entry: dict[str, Any] = {"symbol": symbol, "est_bytes": 0}
        listing = listings.get(symbol)
        entry["listing_ms"] = listing
        if not include_full and (symbol in SUPPORTED_SYMBOLS or have["1m"] > 0):
            entry["tier"] = "full"
        elif listing is None:
            entry["tier"] = "unlisted"
        else:
            entry["tier"] = "research"
            span = max(0, now_ms - listing)
            days = span // _DAY_MS
            entry["history_days"] = days
            entry["short_history"] = days < SHORT_HISTORY_DAYS
            klines = max(0, span // _INTERVAL_MS["1h"] - have["1h"]) + max(0, days - have["1d"])
            settlements = days * _EST_FUNDING_PER_DAY
            entry["est_bytes"] = (
                klines * _EST_ROW_BYTES["kline"]
                + max(0, settlements - have["funding"]) * _EST_ROW_BYTES["funding"]
                + max(0, settlements - have["interval"]) * _EST_ROW_BYTES["interval"]
            )
        plan.append(entry)
    return plan


def _uncovered_ranges(
    archive: RawDataArchive, symbol: str, interval: str, first: int, last: int
) -> list[tuple[int, int]]:
    """``[lo, hi]`` open-time ranges on the *interval* grid in [first, last] with no bar."""
    step = _INTERVAL_MS[interval]
    ranges: list[tuple[int, int]] = []
    expect = first
    for row in archive.get_klines(symbol, interval, first, last):
        t = int(row["open_time"])
        if t > expect:
            ranges.append((expect, t - step))
        expect = max(expect, t + step)
    if expect <= last:
        ranges.append((expect, last))
    return ranges


def _bar_label(ms: int, interval: str) -> str:
    dt = datetime.fromtimestamp(ms / 1000, tz=UTC)
    return dt.date().isoformat() if interval == "1d" else dt.strftime("%Y-%m-%dT%H:%M")


def _parse_bar_label(text: str, interval: str) -> int:
    """Parse an open time in the exact text ``_bar_label`` prints for *interval*."""
    fmt, hint = (
        ("%Y-%m-%d", "YYYY-MM-DD") if interval == "1d" else ("%Y-%m-%dT%H:%M", "YYYY-MM-DDTHH:MM")
    )
    try:
        dt = datetime.strptime(text, fmt).replace(tzinfo=UTC)
    except ValueError:
        raise ValueError(f"{text!r}: {interval} open times are {hint}") from None
    return int(dt.timestamp() * 1000)


def _months_touched(lo_ms: int, hi_ms: int) -> list[tuple[int, int]]:
    """Every calendar month ``[lo_ms, hi_ms]`` touches, oldest first."""
    lo = datetime.fromtimestamp(lo_ms / 1000, tz=UTC).date().replace(day=1)
    hi = datetime.fromtimestamp(hi_ms / 1000, tz=UTC).date().replace(day=1)
    months: list[tuple[int, int]] = []
    while (lo.year, lo.month) <= (hi.year, hi.month):
        months.append((lo.year, lo.month))
        lo = date(lo.year + lo.month // 12, lo.month % 12 + 1, 1)
    return months


def _served_in(rows: list[tuple[Any, ...]], lo_ms: int, hi_ms: int) -> bool:
    return any(lo_ms <= int(r[0]) <= hi_ms for r in rows)


def accept_gap(
    symbol: str,
    interval: str,
    from_text: str,
    to_text: str,
    reason: str,
    *,
    now: datetime | None = None,
) -> int:
    """``accept-gap``: record a dated, evidenced exchange gap, or refuse.

    The range is the open time of the first and last missing bar (inclusive,
    UTC, in ``_bar_label`` text), ending at or before the last closed bar. It
    is written only when the archive holds no bar in it, no Vision month it
    touches (a 404 or a published file) and no REST answer has a bar opening
    in it, and no probe raised: a gap the exchange served is never accepted.
    Append-only, so an accepted range is never rewritten.

    Returns:
        CLI exit code: 0 accepted or already accepted, 1 refused (nothing written).
    """
    symbol = symbol.upper()
    if interval not in RESEARCH_INTERVALS:
        print(
            f"Error: {interval} is not a research interval ({', '.join(RESEARCH_INTERVALS)})",
            file=sys.stderr,
        )
        return 1
    step = _INTERVAL_MS[interval]
    try:
        from_ms, to_ms = _parse_bar_label(from_text, interval), _parse_bar_label(to_text, interval)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if from_ms % step or to_ms % step:
        print(f"Error: {from_text}..{to_text} is not on the {interval} grid", file=sys.stderr)
        return 1
    if from_ms > to_ms:
        print(f"Error: FROM {from_text} is after TO {to_text}", file=sys.stderr)
        return 1
    now_ms = int((now or datetime.now(UTC)).timestamp() * 1000)
    last_closed = now_ms // step * step - step
    if to_ms > last_closed:
        print(
            f"Error: TO {to_text} is after the last closed {interval} bar "
            f"{_bar_label(last_closed, interval)}",
            file=sys.stderr,
        )
        return 1
    if not reason.strip():
        print("Error: --reason must not be blank (it is the evidence record)", file=sys.stderr)
        return 1

    archive = RawDataArchive()
    try:
        if archive.get_klines(symbol, interval, from_ms, to_ms):
            print(
                f"Error: {symbol} {interval} already holds bars in {from_text}..{to_text}",
                file=sys.stderr,
            )
            return 1
        statuses: list[str] = []
        try:
            with httpx.Client(timeout=60.0) as client:
                for year, month in _months_touched(from_ms, to_ms):
                    rows = download_monthly_klines(symbol, interval, year, month, client=client)
                    if rows is None:
                        statuses.append(f"{year}-{month:02d}:404")
                        continue
                    if _served_in(rows, from_ms, to_ms):
                        print(
                            f"Error: Vision has data for {year}-{month:02d}: "
                            f"{symbol} {interval} is served in the range, refusing",
                            file=sys.stderr,
                        )
                        return 1
                    statuses.append(f"{year}-{month:02d}:no rows in range")
                # Binance endTime is inclusive on open time: TO + 1 stops at TO
                rest_rows = download_recent_klines(symbol, interval, from_ms, to_ms + 1)
                if _served_in(rest_rows, from_ms, to_ms):
                    print(
                        "Error: REST serves bars in the range, refusing",
                        file=sys.stderr,
                    )
                    return 1
        except (DownloadError, httpx.HTTPError) as e:
            print(f"Error: probe failed, refusing: {e}", file=sys.stderr)
            return 1
        inserted = archive.insert_accepted_gap(
            symbol, interval, from_ms, to_ms, ",".join(statuses), reason
        )
    finally:
        archive.close()
    range_text = f"{from_text}..{to_text}"
    if inserted:
        print(f"ACCEPTED GAP {symbol} {interval} {range_text} ({reason})")
    else:
        print(f"already accepted: {symbol} {interval} {range_text}")
    return 0


def ingest_research_symbol(
    symbol: str,
    listing_ms: int,
    archive: RawDataArchive,
    *,
    now: datetime,
    verbose: bool = True,
) -> dict[str, Any]:
    """Archive 1h/1d LAST klines and funding for one coin from its listing.

    Complete months come from Binance Vision monthly zips (a month already
    covered is skipped); every range still uncovered between the listing and
    the last closed bar (the current month, a month Vision 404s, a hole in a
    month past the 0.9 coverage skip) is then fetched from REST. A range REST
    does not serve either is listed in ``failed`` ("<interval> missing
    <from>..<to>"), as is a unit that fails after retries (a month, the REST
    backfill, the funding history: it writes nothing); a re-run fetches only
    what is still missing (INSERT OR IGNORE keeps it duplicate-free).

    Returns:
        ``{"klines_inserted": {interval: n}, "funding_rates": n,
        "funding_intervals": n, "failed": [...], "unavailable": [...]}``.
    """
    now_ms = int(now.timestamp() * 1000)
    listing_dt = datetime.fromtimestamp(listing_ms / 1000, tz=UTC)
    counts: dict[str, Any] = {
        "klines_inserted": dict.fromkeys(RESEARCH_INTERVALS, 0),
        "funding_rates": 0,
        "funding_intervals": 0,
        "failed": [],
        "unavailable": [],
    }
    failed: list[str] = counts["failed"]
    months = get_months_in_range(listing_dt, now, now=now)

    with httpx.Client(timeout=60.0) as client:
        for interval in RESEARCH_INTERVALS:
            failed_before = len(failed)
            for year, month in months:
                if archive.has_month_coverage(symbol, interval, year, month):
                    continue
                try:
                    rows = download_monthly_klines(symbol, interval, year, month, client=client)
                except MonthlyDownloadError as e:
                    failed.append(f"{interval} {year}-{month:02d}")
                    logger.error("%s %s %d-%02d FAILED: %s", symbol, interval, year, month, e)
                    continue
                if rows is None:
                    counts["unavailable"].append(f"{interval} {year}-{month:02d}")
                    continue
                counts["klines_inserted"][interval] += archive.insert_klines(
                    symbol, interval, rows, "binance_vision"
                )
            if len(failed) > failed_before:
                continue  # the REST backfill waits for a hole-free monthly pass
            step = _INTERVAL_MS[interval]
            first = listing_ms // step * step
            last = now_ms // step * step - step  # the last closed bar
            try:
                for lo, hi in _uncovered_ranges(archive, symbol, interval, first, last):
                    rows = download_recent_klines(symbol, interval, lo, hi + 1)
                    counts["klines_inserted"][interval] += archive.insert_klines(
                        symbol, interval, _drop_unclosed_klines(rows, now_ms), "binance_api"
                    )
            except RestDownloadError as e:
                failed.append(f"{interval} REST backfill")
                logger.error("%s %s REST backfill FAILED: %s", symbol, interval, e)
                continue
            # Bars before the first one the exchange served were never traded;
            # any other uncovered range is a hole: fail so a re-run retries it.
            # A hole fully inside a recorded accepted gap is the exchange's own
            # outage: report it as accepted, not as a failure.
            served = archive.get_date_range(symbol, interval)
            head = max(first, served[0]) if served else first
            accepted = archive.get_accepted_gaps(symbol, interval)
            for lo, hi in _uncovered_ranges(archive, symbol, interval, head, last):
                if any(a <= lo and hi <= b for a, b in accepted):
                    if verbose:
                        print(
                            f"  ACCEPTED GAP {interval} {_bar_label(lo, interval)}"
                            f"..{_bar_label(hi, interval)}"
                        )
                    continue
                failed.append(
                    f"{interval} missing {_bar_label(lo, interval)}..{_bar_label(hi, interval)}"
                )

    funding = archive.get_funding_range(symbol)
    try:
        rates = download_funding_rates(
            symbol, funding[2] + 1 if funding else listing_ms, now_ms, strict=True
        )
    except RestDownloadError as e:
        failed.append("funding rates")
        logger.error("%s funding rates FAILED: %s", symbol, e)
    else:
        counts["funding_rates"] = archive.insert_funding_rates(symbol, rates, "binance_api")
    try:
        counts["funding_intervals"] = ingest_funding_intervals(symbol, archive, now=now)
    except MonthlyDownloadError as e:
        failed.append("funding intervals")
        logger.error("%s funding intervals FAILED: %s", symbol, e)

    if verbose:
        k = counts["klines_inserted"]
        print(
            f"  {symbol}: {k['1h']:,} 1h + {k['1d']:,} 1d klines, "
            f"{counts['funding_rates']:,} funding rates, "
            f"{counts['funding_intervals']:,} funding intervals"
        )
        if counts["unavailable"]:
            print(f"    Unavailable (404): {', '.join(counts['unavailable'])}")
        if failed:
            print(f"    FAILED: {', '.join(failed)}")
    return counts


def _archive_bytes(path: Path) -> int:
    """Archive DB + WAL size on disk."""
    return sum(p.stat().st_size for p in (path, path.with_name(path.name + "-wal")) if p.exists())


def _existing_dir(path: Path) -> Path:
    """Nearest existing ancestor directory of *path* (for the free-space check)."""
    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent


def _confirm() -> bool:
    """y/yes proceeds; any other answer, or no stdin (EOF), aborts."""
    try:
        answer = input("Proceed? [y/N] ")
    except EOFError:
        logger.warning("no answer on stdin (EOF): treated as no; pass --yes to proceed")
        return False
    return answer.strip().lower() in ("y", "yes")


# EYEBALLED 2026-11-22 (du): data/catalog 3.1 GB next to data/archive 8.2 GB for
# the 8 full-tier coins, so catalog Parquet bars add ~0.38 byte per archive byte.
_CATALOG_PER_ARCHIVE_BYTE = 0.38


def _missing_1m_rows(
    archive_path: Path, table: str, symbol: str, months: list[tuple[int, int]]
) -> int:
    """1m rows a full-tier download could add (months below the 0.9 coverage skip).

    Read-only: never creates or migrates the archive. Pre-listing months count
    as missing (over), the current month's daily tail is left out (under).
    """
    import calendar
    import sqlite3

    expected = {(y, m): calendar.monthrange(y, m)[1] * 1440 for y, m in months}
    have = dict.fromkeys(expected, 0)
    if archive_path.exists():
        con = sqlite3.connect(f"file:{archive_path}?mode=ro", uri=True)
        try:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master")}
            for y, m in expected if table in tables else ():
                lo = datetime(y, m, 1, tzinfo=UTC)
                hi = datetime(y + m // 12, m % 12 + 1, 1, tzinfo=UTC)
                have[(y, m)] = con.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE symbol = ? AND interval = '1m' "
                    "AND open_time >= ? AND open_time < ?",
                    (symbol, int(lo.timestamp() * 1000), int(hi.timestamp() * 1000)),
                ).fetchone()[0]
        finally:
            con.close()
    return sum(n - have[k] for k, n in expected.items() if have[k] < n * 0.9)


def _full_tier_preflight(
    symbols: list[str],
    months: list[tuple[int, int]],
    *,
    table: str,
    with_catalog: bool,
    yes: bool,
) -> int | None:
    """``ingest`` / ``download-mark`` gate, before anything is opened for writing.

    Refuses a coin without an instrument spec, prints the per-coin 1m estimate
    and the free disk, refuses when free < estimate + 5 GB, and asks (unless
    *yes*). *table* is ``raw_klines`` (LAST) or ``raw_mark_klines`` (MARK), a
    module constant, never user input.

    Returns:
        None to proceed, else the CLI exit code (1 refused, 0 aborted).
    """
    from vibe_quant.data import archive as archive_mod

    unknown = [s for s in symbols if s not in INSTRUMENT_CONFIGS]
    if unknown:
        print(
            f"Error: {', '.join(unknown)}: no instrument spec in INSTRUMENT_CONFIGS "
            "(vibe_quant/data/catalog.py), so no full tier (1m LAST/MARK + catalog bars); "
            "research-tier data (1h/1d + funding) comes from download-research",
            file=sys.stderr,
        )
        return 1
    archive_path = archive_mod.DEFAULT_ARCHIVE_PATH
    kind = "LAST" if table == "raw_klines" else "MARK"
    print(f"\nFull tier: 1m {kind} estimate")
    estimate = 0
    for symbol in symbols:
        rows = _missing_1m_rows(archive_path, table, symbol, months)
        size = rows * _EST_ROW_BYTES["kline"]
        if with_catalog:
            size += int(size * _CATALOG_PER_ARCHIVE_BYTE)
        estimate += size
        print(f"  {symbol:<16} {rows:,} new 1m rows, {_format_size(size)}")
    _total, _used, free = shutil.disk_usage(_existing_dir(archive_path))
    print(
        f"Estimate: {_format_size(estimate)}{' incl. catalog bars' if with_catalog else ''}; "
        f"free disk: {_format_size(free)} (needs estimate + {_format_size(DISK_HEADROOM_BYTES)})"
    )
    if free < estimate + DISK_HEADROOM_BYTES:
        print(
            "Error: refusing to download: free disk is below estimate + 5 GB headroom",
            file=sys.stderr,
        )
        return 1
    if not yes and not _confirm():
        print("aborted: nothing written")
        return 0
    return None


def _print_research_plan(plan: list[dict[str, Any]]) -> None:
    for p in plan:
        sym = f"  {p['symbol']:<16}"
        if p["tier"] == "full":
            print(f"{sym} full tier: skipped (1m archive is updated by 'update')")
        elif p["tier"] == "unlisted":
            print(f"{sym} NOT in exchangeInfo: cannot date the listing, not downloaded")
        else:
            listed = datetime.fromtimestamp(p["listing_ms"] / 1000, tz=UTC).date()
            short = f"  short history ({p['history_days']} days)" if p["short_history"] else ""
            print(f"{sym} listed {listed}  estimate {_format_size(p['est_bytes'])}{short}")


def download_research_tier(*, yes: bool, dry_run: bool, include_full: bool = False) -> int:
    """``download-research``: plan, size check, confirm, then download.

    Returns:
        CLI exit code: 0 done/aborted/dry run, 1 refused or any unit failed.
    """
    from vibe_quant.data import archive as archive_mod
    from vibe_quant.data import universe

    symbols, as_of = universe.read_universe(universe.UNIVERSE_FILE)
    archive_path = archive_mod.DEFAULT_ARCHIVE_PATH
    try:
        with httpx.Client(timeout=30.0) as client:
            listings = universe.listing_dates(universe.fetch_exchange_info(client))
    except DownloadError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    now = datetime.now(UTC)
    plan = plan_research_tier(symbols, listings, archive_path, now, include_full=include_full)
    todo = [p for p in plan if p["tier"] == "research"]
    unlisted = [p["symbol"] for p in plan if p["tier"] == "unlisted"]
    estimate = sum(p["est_bytes"] for p in todo)
    _total, _used, free = shutil.disk_usage(_existing_dir(archive_path))

    print(f"\nResearch tier (1h + 1d LAST klines, funding) from {universe.UNIVERSE_FILE}:")
    _print_research_plan(plan)
    print(
        f"\nEstimate: {_format_size(estimate)} for {len(todo)} coin(s); "
        f"free disk: {_format_size(free)} (needs estimate + {_format_size(DISK_HEADROOM_BYTES)})"
    )
    if as_of is not None:
        print(universe.honesty_note(as_of))
    if free < estimate + DISK_HEADROOM_BYTES:
        print(
            "Error: refusing to download: free disk is below estimate + 5 GB headroom",
            file=sys.stderr,
        )
        return 1
    if estimate > RESEARCH_BUDGET_BYTES:
        print(
            f"Error: refusing to download: estimate {_format_size(estimate)} is above the "
            f"research-tier budget of {_format_size(RESEARCH_BUDGET_BYTES)}",
            file=sys.stderr,
        )
        return 1
    if dry_run:
        print("Dry run: nothing written.")
        return 0
    if todo and not yes and not _confirm():
        print("aborted: nothing written")
        return 0

    failures = [f"{s}: not in exchangeInfo" for s in unlisted]
    if todo:
        archive = RawDataArchive(archive_path)
        session_id = archive.create_download_session(
            [p["symbol"] for p in todo], "mixed", start_date=None, end_date=now.date().isoformat()
        )
        inserted = 0
        start_size = _archive_bytes(archive_path)
        try:
            for i, p in enumerate(todo):
                counts = ingest_research_symbol(p["symbol"], p["listing_ms"], archive, now=now)
                inserted += sum(counts["klines_inserted"].values())
                failures += [f"{p['symbol']}: {f}" for f in counts["failed"]]
                grown = _archive_bytes(archive_path) - start_size
                if i == 0:
                    print(
                        f"  MEASURED on {p['symbol']}: {_format_size(grown)} archive growth "
                        f"(estimate {_format_size(p['est_bytes'])})"
                    )
                rest = todo[i + 1 :]
                if rest and grown + sum(r["est_bytes"] for r in rest) > RESEARCH_BUDGET_BYTES:
                    failures += [
                        f"{r['symbol']}: not downloaded: measured growth {_format_size(grown)} "
                        f"+ remaining estimate exceeds the {_format_size(RESEARCH_BUDGET_BYTES)} "
                        "research-tier budget"
                        for r in rest
                    ]
                    break
            archive.complete_download_session(
                session_id,
                klines_fetched=inserted,
                klines_inserted=inserted,
                status="partial" if failures else "completed",
                error_message="; ".join(failures)[:_MAX_SESSION_SUMMARY] or None,
            )
        except Exception as e:
            archive.complete_download_session(session_id, status="failed", error_message=str(e))
            raise
        finally:
            archive.close()

    if failures:
        print("\nERROR: research-tier downloads failed:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        print("Re-run download-research to resume (only missing data is fetched).", file=sys.stderr)
        return 1
    return 0


# Positioning data (design arch-posdata-1): Vision daily metrics files hold 288
# five-minute rows; MEASURED 77.1 B/row WITHOUT ROWID, estimated at 80 B (DVOL too).
_METRICS_ROWS_PER_FILE = 288
_EST_POSITIONING_ROW_BYTES = 80
DVOL_CURRENCIES = ("BTC", "ETH")
DVOL_FIRST_DAY = date(2021, 3, 24)  # Deribit's first BTC/ETH DVOL candle


def _read_archive(archive_path: Path, table: str, sql: str, params: tuple[Any, ...]) -> list[Any]:
    """Rows of *sql* on the archive opened read-only; [] when the file or *table* is absent."""
    import sqlite3

    if not archive_path.exists():
        return []
    con = sqlite3.connect(f"file:{archive_path}?mode=ro", uri=True)
    try:
        if con.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone() is None:
            return []
        return list(con.execute(sql, params))
    finally:
        con.close()


def _metrics_days_to_fetch(
    archive_path: Path, symbol: str, listed: list[VisionFile], recheck: bool
) -> int:
    """Listed days ``ingest_metrics`` will GET (its rule, read from the archive read-only)."""
    recorded: dict[str, list[tuple[str, str]]] = {}
    for day, status, md5 in _read_archive(
        archive_path,
        "raw_metrics_files",
        "SELECT day, status, md5 FROM raw_metrics_files WHERE symbol = ?",
        (symbol,),
    ):
        recorded.setdefault(day, []).append((status, md5))
    todo = 0
    for vision_file in listed:
        files = recorded.get(vision_file.day.isoformat(), [])
        has_ok = any(status == "ok" for status, _ in files)
        todo += int(not any(md5 == vision_file.md5 for _, md5 in files) and has_ok == recheck)
    return todo


def _disk_gate(archive_path: Path, estimate: int, *, yes: bool, dry_run: bool) -> int | None:
    """Free-disk refusal, dry run and confirmation; None to proceed, else the exit code."""
    _total, _used, free = shutil.disk_usage(_existing_dir(archive_path))
    print(f"free disk: {_format_size(free)} (needs estimate + {_format_size(DISK_HEADROOM_BYTES)})")
    if free < estimate + DISK_HEADROOM_BYTES:
        print(
            "Error: refusing to download: free disk is below estimate + 5 GB headroom",
            file=sys.stderr,
        )
        return 1
    if dry_run:
        print("Dry run: nothing written.")
        return 0
    if not yes and not _confirm():
        print("aborted: nothing written")
        return 0
    return None


def download_metrics(
    symbols: list[str],
    start: date | None,
    end: date,
    *,
    recheck: bool,
    yes: bool,
    dry_run: bool,
) -> int:
    """``download-metrics``: list, estimate, size check, confirm, then download.

    The listings and the estimate read the archive read-only, so a listing
    failure, a refusal, a dry run or a "no" leaves it unchanged (not even created).

    Returns:
        CLI exit code: 0 done/aborted/dry run (malformed or drift files are
        recorded and printed), 1 refused, listing failed or a day failed.
    """
    from vibe_quant.data import archive as archive_mod
    from vibe_quant.data.metrics import ingest_metrics, list_metrics_files

    archive_path = archive_mod.DEFAULT_ARCHIVE_PATH
    with httpx.Client(timeout=60.0) as client:
        todo: dict[str, int] = {}
        try:
            for symbol in symbols:
                listed = [
                    f
                    for f in list_metrics_files(symbol, client)
                    if (start is None or f.day >= start) and f.day <= end
                ]
                todo[symbol] = _metrics_days_to_fetch(archive_path, symbol, listed, recheck)
        except DownloadError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        print(f"\nBinance Vision metrics{' recheck' if recheck else ''}: files, rows, est bytes")
        for symbol, files in todo.items():
            rows = files * _METRICS_ROWS_PER_FILE
            size = _format_size(rows * _EST_POSITIONING_ROW_BYTES)
            print(f"  {symbol:<16} {files} file(s), {rows:,} rows, {size}")
        estimate = sum(todo.values()) * _METRICS_ROWS_PER_FILE * _EST_POSITIONING_ROW_BYTES
        print(f"Estimate: {_format_size(estimate)} for {sum(todo.values())} file(s)")
        gate = _disk_gate(archive_path, estimate, yes=yes, dry_run=dry_run)
        if gate is not None:
            return gate

        failures: list[str] = []
        flagged = False
        archive = RawDataArchive(archive_path)
        try:
            for symbol in symbols:
                seen = {row["id"] for row in archive.get_metrics_files(symbol)}
                try:
                    counts = ingest_metrics(
                        symbol, archive, start=start, end=end, recheck=recheck, client=client
                    )
                except DownloadError as e:
                    failures.append(f"{symbol}: {e}")
                    continue
                print(
                    f"{symbol}: {counts['files_ok']} file(s) ok, {counts['rows_inserted']:,} rows "
                    f"inserted, {counts['skipped']} skipped, {counts['same']} same, "
                    f"{counts['drift']} drift, {counts['malformed']} malformed, "
                    f"{counts['failed']} failed"
                )
                for row in archive.get_metrics_files(symbol):
                    if row["id"] not in seen and row["status"] in ("malformed", "drift"):
                        flagged = True
                        print(f"  {row['status'].upper()} {symbol} {row['day']} ({row['detail']})")
                if counts["failed"]:
                    failures.append(f"{symbol}: {counts['failed']} day(s) failed")
        finally:
            archive.close()
    if flagged:
        print("Malformed or drifted metrics files recorded: run verify")
    if failures:
        print("\nERROR: metrics downloads failed (re-run download-metrics):", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


def download_dvol(currencies: list[str], start: date, *, yes: bool) -> int:
    """``download-dvol``: estimate, size check, confirm, then download each currency.

    Returns:
        CLI exit code: 0 done/aborted, 1 refused or any currency failed (its
        earlier pages stay archived; a re-run resumes).
    """
    from vibe_quant.data import archive as archive_mod
    from vibe_quant.data.dvol import DVOL_RESOLUTION_S, DvolDownloadError, ingest_dvol

    archive_path = archive_mod.DEFAULT_ARCHIVE_PATH
    start_dt = datetime(start.year, start.month, start.day, tzinfo=UTC)
    hours = max(0, int((datetime.now(UTC) - start_dt).total_seconds()) // DVOL_RESOLUTION_S)
    archived = dict(
        _read_archive(
            archive_path,
            "raw_dvol",
            "SELECT currency, COUNT(*) FROM raw_dvol WHERE resolution_s = ? GROUP BY currency",
            (DVOL_RESOLUTION_S,),
        )
    )
    print(f"\nDeribit DVOL hourly candles from {start}: candles, est bytes")
    estimate = 0
    for currency in currencies:
        candles = max(0, hours - archived.get(currency, 0))
        estimate += candles * _EST_POSITIONING_ROW_BYTES
        print(
            f"  {currency:<16} {candles:,} candles, {_format_size(candles * _EST_POSITIONING_ROW_BYTES)}"
        )
    print(f"Estimate: {_format_size(estimate)}")
    gate = _disk_gate(archive_path, estimate, yes=yes, dry_run=False)
    if gate is not None:
        return gate

    failures: list[str] = []
    archive = RawDataArchive(archive_path)
    try:
        for currency in currencies:
            try:
                result = ingest_dvol(currency, archive, start=start)
            except DvolDownloadError as e:
                failures.append(str(e))
                continue
            first, last = result["first"], result["last"]
            span = (
                f"{_iso_ms(first)} to {_iso_ms(last)}"
                if first is not None and last is not None
                else "nothing"
            )
            print(f"DVOL {currency}: {result['inserted']:,} new candle(s), archived {span}")
    finally:
        archive.close()
    if failures:
        print("\nERROR: DVOL downloads failed:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


def _print_research_verify(archive: RawDataArchive, symbol: str) -> bool:
    """Print the research-tier checks of one symbol; True when an issue was found."""
    r = verify_research_symbol(archive, symbol)
    counts = r["kline_counts"]
    print(f"\n{symbol}: research tier, {counts['1h']} 1h + {counts['1d']} 1d klines")
    issues = bool(r["coverage"])
    if r["coverage"]:
        print(f"  COVERAGE ({len(r['coverage'])}):")
        for line in r["coverage"][:10]:
            print(f"    {line}")
        if len(r["coverage"]) > 10:
            print(f"    ... and {len(r['coverage']) - 10} more")
    for interval, gaps in r["gaps"].items():
        if gaps:
            issues = True
            print(f"  GAPS {interval} ({len(gaps)}):")
            for start_ts, _end_ts, gap_min in gaps[:5]:
                print(f"    {_iso_ms(start_ts)} - {gap_min} min step")
            if len(gaps) > 5:
                print(f"    ... and {len(gaps) - 5} more")
    if not any(r["gaps"].values()):
        print("  Gaps: None")
    for interval, gaps in r["accepted_gaps"].items():
        if gaps:
            print(f"  ACCEPTED GAPS {interval} ({len(gaps)}):")
            for start_ts, _end_ts, gap_min in gaps[:5]:
                print(f"    {_iso_ms(start_ts)} - {gap_min} min step")
            if len(gaps) > 5:
                print(f"    ... and {len(gaps) - 5} more")
    if r["ohlc_errors"]:
        issues = True
        print(f"  OHLC ERRORS ({len(r['ohlc_errors'])}):")
        for ts, msg in r["ohlc_errors"][:5]:
            print(f"    {_iso_ms(ts)}: {msg}")
    if r["ticker_reuse"]:
        issues = True
        print(f"  POSSIBLE TICKER REUSE ({len(r['ticker_reuse'])}):")
        for ts, msg in r["ticker_reuse"][:5]:
            print(f"    {_iso_ms(ts)[:10]}: {msg}")
    if r["no_data_days"]:
        print(
            f"  NO DATA (flat zero-volume daily bars, not prices): {len(r['no_data_days'])} "
            f"day(s): {_format_days(r['no_data_days'])}"
        )
    return issues


def _print_kline_verify(archive: RawDataArchive, symbol: str) -> bool:
    """Print the full-tier 1m checks of one symbol; True when an issue was found."""
    result = verify_symbol(archive, symbol)
    gaps = result["gaps"]
    ohlc_errors = result["ohlc_errors"]
    kline_count = result["kline_count"]

    print(f"\n{symbol}: {kline_count} klines")

    has_issues = False
    missing = interior_missing_days(archive, symbol)
    missing_klines = missing["klines"]
    missing_mark = missing["mark_klines"]
    if missing_klines or missing_mark:
        has_issues = True
        if missing_klines:
            print(
                f"  MISSING 1m DAYS (klines): {len(missing_klines)} interior "
                f"missing day(s): {_format_days(missing_klines)}"
            )
        if missing_mark:
            print(
                f"  MISSING 1m DAYS (mark klines): {len(missing_mark)} interior "
                f"missing day(s): {_format_days(missing_mark)}"
            )
    else:
        print("  Missing 1m days: None")

    if kline_count == 0:
        print("  NO DATA: symbol not found in archive")
        return True

    if gaps:
        has_issues = True
        print(f"  GAPS ({len(gaps)}):")
        for start_ts, _end_ts, gap_min in gaps[:5]:  # Show first 5
            start_dt = datetime.fromtimestamp(start_ts / 1000, tz=UTC)
            print(f"    {start_dt.isoformat()} - {gap_min} min gap")
        if len(gaps) > 5:
            print(f"    ... and {len(gaps) - 5} more")
    else:
        print("  Gaps: None")

    if ohlc_errors:
        has_issues = True
        print(f"  OHLC ERRORS ({len(ohlc_errors)}):")
        for ts, msg in ohlc_errors[:5]:  # Show first 5
            err_dt = datetime.fromtimestamp(ts / 1000, tz=UTC)
            print(f"    {err_dt.isoformat()}: {msg}")
        if len(ohlc_errors) > 5:
            print(f"    ... and {len(ohlc_errors) - 5} more")
    else:
        print("  OHLC: Valid")
    return has_issues


def _print_positioning_verify(archive: RawDataArchive, symbol: str) -> bool:
    """Print *symbol*'s metrics block; True when a newest file is unacked drift/malformed.

    Holes are information (the indicators read them as NaN), so only the file
    outcomes can fail verify.
    """
    r = verify_metrics(archive, symbol)
    span = archive.get_metrics_range(symbol)
    if span is None:
        print("  Metrics: none archived (download-metrics)")
    else:
        print(
            f"  Metrics: {span[0]} rows {r['first_day']} to {r['last_day']}, {r['files']} file(s)"
        )
    if r["missing_days"]:
        n = len(r["missing_days"])
        print(
            f"  metrics gaps: {n} missing day{'s' if n > 1 else ''} "
            f"({_format_days(r['missing_days'])})"
        )
    if r["missing_rows_by_day"]:
        print(
            "  metrics missing rows: "
            + _format_days([f"{day} {n}" for day, n in r["missing_rows_by_day"].items()])
        )
    if r["misaligned"]:
        print(f"  metrics misaligned: {r['misaligned']} row(s) off the 5-minute grid")
    if r["null_counts"]:
        print(
            "  metrics null values: "
            + ", ".join(f"{column} {n}" for column, n in r["null_counts"].items())
        )
    for status, day, detail, reason in r["acked"]:
        print(f"  metrics {status} acknowledged: {day} ({detail}; {reason})")
    if r["same"]:
        print(f"  metrics: {r['same']} regenerated file(s), values identical")
    for day, detail in r["drift"]:
        print(f"  METRICS DRIFT {symbol} {day} ({detail})")
    for day, detail in r["malformed"]:
        print(f"  METRICS MALFORMED {symbol} {day} ({detail})")
    return bool(r["drift"] or r["malformed"])


def _print_dvol_verify(archive: RawDataArchive) -> None:
    """Print each DVOL currency's span and interior holes (information, never an issue)."""
    print()
    for currency in DVOL_CURRENCIES:
        r = verify_dvol(archive, currency)
        if r["first"] is None or r["last"] is None:
            print(f"DVOL {currency}: none archived (download-dvol)")
            continue
        print(
            f"DVOL {currency}: {r['candles']} candles {_iso_ms(r['first'])} to {_iso_ms(r['last'])}"
        )
        if r["gaps"]:
            missing = sum(n for _start, n in r["gaps"])
            print(
                f"  dvol gaps: {missing} missing candle(s) ("
                + _format_days([f"{_iso_ms(start)} {n}" for start, n in r["gaps"]])
                + ")"
            )


def ack_metrics(
    symbol: str, day_text: str, reason: str, *, client: httpx.Client | None = None
) -> int:
    """``ack-metrics``: acknowledge a drift/malformed metrics file, or refuse.

    Acknowledges the day's newest files row (the one ``verify`` judges) when
    it is drift/malformed, only while Vision lists that exact file or an
    earlier ok/same file of the day (listing ETag == its md5; recheck skips a
    re-served known file): any other file must be rechecked first. Records
    the listed md5. Never automatic.

    Returns:
        CLI exit code: 0 acknowledged or already acknowledged, 1 refused
        (nothing written).
    """
    from vibe_quant.data import archive as archive_mod
    from vibe_quant.data.metrics import list_metrics_files

    symbol = symbol.upper()
    try:
        day = date.fromisoformat(day_text)
    except ValueError:
        print(f"Error: DAY {day_text} is not YYYY-MM-DD", file=sys.stderr)
        return 1
    if not reason.strip():
        print("Error: --reason must not be blank (it is the evidence record)", file=sys.stderr)
        return 1
    rows = list(
        _read_archive(
            archive_mod.DEFAULT_ARCHIVE_PATH,
            "raw_metrics_files",
            "SELECT f.id, f.status, f.md5, f.detail, a.id FROM raw_metrics_files f "
            "LEFT JOIN raw_metrics_acks a ON a.file_id = f.id "
            "WHERE f.symbol = ? AND f.day = ? ORDER BY f.id DESC",
            (symbol, day.isoformat()),
        )
    )
    if not rows or rows[0][1] not in ("drift", "malformed"):
        print(
            f"Error: no unacknowledged drift or malformed metrics file for {symbol} {day}",
            file=sys.stderr,
        )
        return 1
    file_id, status, md5, detail, ack_id = rows[0]
    label = f"METRICS {status.upper()} {symbol} {day} file #{file_id} ({detail})"
    if ack_id is not None:
        print(f"already acknowledged: {label}")
        return 0
    accepted = {md5} | {r[2] for r in rows[1:] if r[1] in ("ok", "same")}
    try:
        if client is None:
            with httpx.Client(timeout=60.0) as own:
                listed = list_metrics_files(symbol, own)
        else:
            listed = list_metrics_files(symbol, client)
    except (DownloadError, httpx.HTTPError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    listed_md5 = next((f.md5 for f in listed if f.day == day and f.md5 in accepted), None)
    if listed_md5 is None:
        print(
            f"Error: {symbol} {day} changed again since it was recorded; "
            "run download-metrics --recheck first",
            file=sys.stderr,
        )
        return 1
    archive = RawDataArchive(archive_mod.DEFAULT_ARCHIVE_PATH)
    try:
        inserted = archive.insert_metrics_ack(file_id, listed_md5, detail, reason)
    finally:
        archive.close()
    print(f"ACKNOWLEDGED {label} ({reason})" if inserted else f"already acknowledged: {label}")
    return 0


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise ValueError(text)
    return value


def _run_universe(top: int, as_of: date | None) -> int:
    """``universe``: rank, print, write the frozen file and its SHA-256.

    A file for the same as-of day and N is kept unchanged without a request:
    exchangeInfo membership is read at run time, so re-ranking a past as-of
    later could select different coins (a delisted winner drops out).
    """
    from vibe_quant.data import universe

    today = datetime.now(UTC).date()
    day = as_of or today - timedelta(days=1)
    if day >= today:
        print(
            f"Error: --as-of {day} is not a complete UTC day "
            f"(latest complete day: {today - timedelta(days=1)})",
            file=sys.stderr,
        )
        return 1
    if top > universe.USER_CAP:
        print(
            f"WARNING: --top {top} is above the user's cap of {universe.USER_CAP} coins "
            "(ruling 2026-11-19: disk is limited)",
            file=sys.stderr,
        )
    path = universe.UNIVERSE_FILE
    if path.exists():
        kept, kept_as_of = universe.read_universe(path)
        if kept_as_of == day and len(kept) == top:
            print(
                f"Frozen universe for as-of {day} with {top} symbols exists: kept unchanged "
                "(no request; a re-rank would use today's exchangeInfo membership)"
            )
            for rank, symbol in enumerate(kept, 1):
                print(f"{rank:>3}. {symbol}")
            print(f"\nKept {path} sha256 {universe.file_sha256(path)}")
            print(universe.honesty_note(day))
            return 0
    try:
        selected, excluded = universe.build_universe(top, day)
    except DownloadError as e:
        print(f"Error: {e}; nothing written, re-run to retry", file=sys.stderr)
        return 1
    print(f"Top {top} by {universe.WINDOW_DAYS}-day quote volume, as-of {day}:")
    for line in universe.format_ranking(selected, excluded):
        print(line)
    sha = universe.write_universe(universe.render_universe(selected, day), path)
    print(f"\nWrote {path} ({len(selected)} symbols) sha256 {sha}")
    print(universe.honesty_note(day))
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for data ingestion."""
    import argparse

    from vibe_quant.data.universe import parse_as_of

    parser = argparse.ArgumentParser(
        description="vibe-quant data ingestion",
        prog="python -m vibe_quant.data",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Download and ingest data")
    ingest_parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(SUPPORTED_SYMBOLS),
        help="Comma-separated symbols to ingest",
    )
    ingest_parser.add_argument(
        "--years", type=int, default=2, help="Years of history (ignored if --start given)"
    )
    ingest_parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD (overrides --years)",
    )
    ingest_parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (defaults to today)",
    )
    ingest_parser.add_argument(
        "--yes",
        action="store_true",
        help="Proceed without the confirmation prompt (the 1m size estimate and the "
        "free-disk check still run)",
    )

    # Status command
    subparsers.add_parser("status", help="Show data status")

    # Update command
    update_parser = subparsers.add_parser("update", help="Update data with recent candles")
    update_parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(SUPPORTED_SYMBOLS),
        help="Comma-separated symbols to update",
    )
    update_parser.add_argument(
        "--state-db",
        type=Path,
        default=None,
        help="State DB whose shadow rows the update advances (default: VIBE_QUANT_DB or "
        "data/state/vibe_quant.db)",
    )
    update_parser.add_argument(
        "--skip-shadow",
        action="store_true",
        help="Do not run the shadow tracker update after the data update",
    )

    # Rebuild command
    rebuild_parser = subparsers.add_parser("rebuild", help="Rebuild catalog from archive")
    rebuild_parser.add_argument(
        "--from-archive",
        action="store_true",
        required=True,
        help="Rebuild catalog from raw SQLite archive (required)",
    )

    # Detail data command (sub-minute data for validation)
    detail_parser = subparsers.add_parser(
        "detail", help="Download sub-minute (1s/5s) detail data for validation"
    )
    detail_parser.add_argument(
        "--symbols",
        type=str,
        default="BTCUSDT",
        help="Comma-separated symbols",
    )
    detail_parser.add_argument(
        "--interval",
        type=str,
        default="5s",
        choices=["1s", "5s"],
        help="Detail interval (default: 5s)",
    )
    detail_parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date YYYY-MM-DD (required)",
    )
    detail_parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (defaults to today)",
    )

    # Mark-price klines command (liquidation trigger price, own archive table)
    mark_parser = subparsers.add_parser(
        "download-mark", help="Download Binance mark-price 1m klines into the archive"
    )
    mark_parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(SUPPORTED_SYMBOLS),
        help="Comma-separated symbols",
    )
    mark_parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date YYYY-MM-DD (required)",
    )
    mark_parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (defaults to today)",
    )
    mark_parser.add_argument(
        "--catalog",
        action="store_true",
        help="Also rebuild MARK 1m bars in the ParquetDataCatalog from the archive",
    )
    mark_parser.add_argument(
        "--yes",
        action="store_true",
        help="Proceed without the confirmation prompt (the size estimate and the "
        "free-disk check still run)",
    )

    # Funding-intervals command (Binance Vision interval backfill)
    intervals_parser = subparsers.add_parser(
        "funding-intervals",
        help="Backfill Binance Vision funding intervals into the archive",
    )
    intervals_parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(SUPPORTED_SYMBOLS),
        help="Comma-separated symbols",
    )

    # Research universe: rank + write the frozen top-N file
    universe_parser = subparsers.add_parser(
        "universe",
        help="Rank Binance USDT-M perps by 30-day quote volume and write the frozen "
        "universe file (vibe_quant/data/universe_top20.txt)",
    )
    universe_parser.add_argument(
        "--top", type=_positive_int, default=20, help="Number of coins (default: 20)"
    )
    universe_parser.add_argument(
        "--as-of",
        type=parse_as_of,
        default=None,
        help="Last day of the 30-day window, YYYY-MM-DD; a complete UTC day "
        "(default: yesterday UTC). A file for the same as-of and --top is kept unchanged",
    )

    # Research-tier download for the universe coins outside the full tier
    research_parser = subparsers.add_parser(
        "download-research",
        help="Download 1h + 1d LAST klines and funding (no 1m, no MARK) from each "
        "listing date for the universe-file coins not in the full tier; prints the "
        "size estimate and free disk and asks before writing",
    )
    research_parser.add_argument(
        "--yes", action="store_true", help="Proceed without the confirmation prompt"
    )
    research_parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan and estimate only"
    )
    research_parser.add_argument(
        "--include-full",
        action="store_true",
        help="Also fetch 1h + 1d klines and funding for the full-tier coins "
        "(their 1m and MARK are not touched)",
    )

    # Accept-gap: record a dated, evidenced exchange outage the probe confirms
    accept_parser = subparsers.add_parser(
        "accept-gap",
        help="Record a dated, evidenced exchange gap (Vision 404 + REST empty) so "
        "download-research and verify stop failing on it; re-probes and refuses "
        "when any data is served",
    )
    accept_parser.add_argument("symbol", help="Trading symbol, uppercase (e.g. XRPUSDT)")
    accept_parser.add_argument(
        "interval", help=f"Research interval: {', '.join(RESEARCH_INTERVALS)}"
    )
    accept_parser.add_argument(
        "from_time", metavar="FROM", help="Open time of the first missing bar (UTC)"
    )
    accept_parser.add_argument(
        "to_time", metavar="TO", help="Open time of the last missing bar (UTC)"
    )
    accept_parser.add_argument("--reason", required=True, help="Evidence record (non-blank)")

    # Positioning data: Binance Vision daily metrics + Deribit DVOL (archive only)
    metrics_parser = subparsers.add_parser(
        "download-metrics",
        help="Download Binance Vision daily metrics (open interest, long/short ratios) "
        "into the archive; prints the estimate and free disk and asks before writing",
    )
    metrics_parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols (default: SUPPORTED_SYMBOLS + the universe file)",
    )
    metrics_parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=None,
        help="First day YYYY-MM-DD (default: the first listed file)",
    )
    metrics_parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=None,
        help="Last day YYYY-MM-DD (default: yesterday UTC)",
    )
    metrics_parser.add_argument(
        "--yes", action="store_true", help="Proceed without the confirmation prompt"
    )
    metrics_parser.add_argument("--dry-run", action="store_true", help="Print the estimate only")
    metrics_parser.add_argument(
        "--recheck",
        action="store_true",
        help="Re-fetch archived days whose listed file changed; records same/drift, "
        "never overwrites archived rows",
    )
    dvol_parser = subparsers.add_parser(
        "download-dvol",
        help="Download Deribit DVOL hourly candles into the archive (resumable)",
    )
    dvol_parser.add_argument(
        "--currencies",
        type=str,
        default=",".join(DVOL_CURRENCIES),
        help="Comma-separated currencies (default: BTC,ETH)",
    )
    dvol_parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=DVOL_FIRST_DAY,
        help=f"First day YYYY-MM-DD (default: {DVOL_FIRST_DAY})",
    )
    dvol_parser.add_argument(
        "--yes", action="store_true", help="Proceed without the confirmation prompt"
    )
    ack_metrics_parser = subparsers.add_parser(
        "ack-metrics",
        help="Acknowledge a drifted or malformed metrics file so verify stops failing on it "
        "(refuses when Vision now lists a different file)",
    )
    ack_metrics_parser.add_argument("symbol", help="Trading symbol (e.g. BTCUSDT)")
    ack_metrics_parser.add_argument("day", metavar="DAY", help="UTC day YYYY-MM-DD")
    ack_metrics_parser.add_argument("--reason", required=True, help="Evidence record (non-blank)")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify data quality")
    verify_parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols to verify (default: all in archive)",
    )

    args = parser.parse_args(argv)

    if args.command == "ingest":
        symbols = [s.strip() for s in args.symbols.split(",")]
        try:
            start_date = (
                datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
                if args.start
                else None
            )
            end_date = (
                datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC) if args.end else None
            )
        except ValueError as e:
            print(f"Error: invalid date format (expected YYYY-MM-DD): {e}", file=sys.stderr)
            return 1
        months = (
            get_months_in_range(start_date, end_date or datetime.now(UTC))
            if start_date is not None
            else get_years_months_to_download(args.years)
        )
        gate = _full_tier_preflight(
            symbols, months, table="raw_klines", with_catalog=True, yes=args.yes
        )
        if gate is not None:
            return gate
        ingest_results = ingest_all(
            symbols=symbols,
            years=args.years,
            start_date=start_date,
            end_date=end_date,
            verbose=True,
        )
        failed = {
            sym: [*cnts.get("failed_months", []), *cnts.get("failed_ranges", [])]
            for sym, cnts in ingest_results.items()
            if cnts.get("failed_months") or cnts.get("failed_ranges")
        }
        if failed:
            print("\nERROR: downloads failed (holes in archive):", file=sys.stderr)
            for sym, fm in failed.items():
                print(f"  {sym}: {', '.join(fm)}", file=sys.stderr)
            return 1
    elif args.command == "status":
        status = get_status()
        print("\nData Status:")
        print("=" * 50)
        for symbol, info in status["symbols"].items():
            print(f"\n{symbol}:")
            if "archive" in info:
                a = info["archive"]
                print(f"  Archive: {a['klines_1m']} klines")
                print(f"    Range: {a['start']} to {a['end']}")
            if "research" in info:
                print("  Archive: research tier (1h/1d LAST + funding, no 1m)")
                for key, r in info["research"].items():
                    print(f"    {key}: {r['count']:,}  {r['start']} to {r['end']}")
            if "detail" in info:
                for intv, d in info["detail"].items():
                    print(f"  Detail {intv}: {d['klines']:,} klines")
                    print(f"    Range: {d['start']} to {d['end']}")
            if "catalog" in info:
                for key, count in info["catalog"].items():
                    print(f"  Catalog {key}: {count}")
            if "metrics" in info:
                m = info["metrics"]
                print(f"  Metrics: {m['rows']:,} rows {m['start']} to {m['end']}")
        for currency, d in status["dvol"].items():
            print(f"\nDVOL {currency}: {d['candles']:,} candles {d['start']} to {d['end']}")
    elif args.command == "detail":
        symbols = [s.strip() for s in args.symbols.split(",")]
        try:
            start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
            end_date = (
                datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC) if args.end else None
            )
        except ValueError as e:
            print(f"Error: invalid date format (expected YYYY-MM-DD): {e}", file=sys.stderr)
            return 1
        archive = RawDataArchive()
        catalog = CatalogManager()
        detail_failed: list[str] = []
        with catalog_write_lock(catalog._catalog_path):
            for symbol in symbols:
                print(f"\n{'=' * 50}")
                print(f"Detail data: {symbol} @ {args.interval}")
                print(f"{'=' * 50}")
                detail_counts = ingest_detail_data(
                    symbol=symbol,
                    interval=args.interval,
                    start_date=start_date,
                    end_date=end_date,
                    archive=archive,
                    catalog=catalog,
                    verbose=True,
                )
                detail_failed += [f"{symbol}: {r}" for r in detail_counts.get("failed_ranges", [])]
        archive.close()
        if detail_failed:
            print("\nERROR: downloads failed (holes in archive):", file=sys.stderr)
            for line in detail_failed:
                print(f"  {line}", file=sys.stderr)
            return 1
    elif args.command == "download-mark":
        symbols = [s.strip() for s in args.symbols.split(",")]
        try:
            mark_start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
            mark_end = (
                datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC) if args.end else None
            )
        except ValueError as e:
            print(f"Error: invalid date format (expected YYYY-MM-DD): {e}", file=sys.stderr)
            return 1
        gate = _full_tier_preflight(
            symbols,
            get_months_in_range(mark_start, mark_end or datetime.now(UTC)),
            table="raw_mark_klines",
            with_catalog=args.catalog,
            yes=args.yes,
        )
        if gate is not None:
            return gate
        mark_archive = RawDataArchive()
        mark_failed: list[str] = []
        for symbol in symbols:
            print(f"\n{'=' * 50}")
            print(f"Mark-price 1m klines: {symbol}")
            print(f"{'=' * 50}")
            try:
                mark_counts = ingest_mark_klines(
                    symbol,
                    start_date=mark_start,
                    end_date=mark_end,
                    archive=mark_archive,
                    verbose=True,
                )
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                mark_failed.append(f"{symbol}: {e}")
                continue
            print(
                f"  {symbol}: {mark_counts['months_downloaded']} months downloaded, "
                f"{mark_counts['months_skipped']} skipped, "
                f"{len(mark_counts['unavailable_months'])} unavailable, "
                f"{len(mark_counts['failed_months'])} failed, "
                f"{len(mark_counts['unpublished_days'])} unpublished days, "
                f"{len(mark_counts['failed_days'])} failed days, "
                f"{mark_counts['mark_klines_inserted']:,} rows inserted"
            )
            if mark_counts["failed_days"]:
                print(f"    FAILED days: {', '.join(mark_counts['failed_days'])}")
            mark_failed += [f"{symbol}: {m}" for m in mark_counts["failed_months"]]
        if args.catalog:
            mark_catalog = CatalogManager()
            with catalog_write_lock(mark_catalog._catalog_path):
                for symbol in symbols:
                    mark_bars = build_mark_bars(mark_catalog, mark_archive, symbol)
                    print(f"  {symbol}: {mark_bars:,} MARK 1m bars in catalog")
        mark_archive.close()
        if mark_failed:
            print("\nERROR: mark-price downloads failed (holes in archive):", file=sys.stderr)
            for line in mark_failed:
                print(f"  {line}", file=sys.stderr)
            return 1
    elif args.command == "update":
        if args.state_db is not None and not Path(args.state_db).exists():
            print(f"ERROR: state DB {args.state_db} does not exist", file=sys.stderr)
            return 1
        symbols = [s.strip() for s in args.symbols.split(",")]
        update_results = update_all(symbols=symbols, verbose=True)
        if not args.skip_shadow:
            from vibe_quant.db.connection import DEFAULT_DB_PATH

            _run_shadow_hook(args.state_db or DEFAULT_DB_PATH, update_results)
        update_failed = [
            f"{sym}: {r}" for sym, c in update_results.items() for r in c.get("failed_ranges", [])
        ]
        if update_failed:
            print("\nERROR: downloads failed (holes in archive):", file=sys.stderr)
            for line in update_failed:
                print(f"  {line}", file=sys.stderr)
            return 1
    elif args.command == "funding-intervals":
        symbols = [s.strip() for s in args.symbols.split(",")]
        intervals_archive = RawDataArchive()
        interval_failed = False
        for symbol in symbols:
            try:
                inserted = ingest_funding_intervals(symbol, intervals_archive, verbose=True)
            except MonthlyDownloadError as e:
                print(f"{symbol}: funding interval fetch failed ({e})", file=sys.stderr)
                interval_failed = True
                continue
            print(f"{symbol}: archived {inserted} funding intervals")
        intervals_archive.close()
        if interval_failed:
            return 1
    elif args.command == "universe":
        return _run_universe(args.top, args.as_of)
    elif args.command == "download-research":
        return download_research_tier(
            yes=args.yes, dry_run=args.dry_run, include_full=args.include_full
        )
    elif args.command == "download-metrics":
        if args.symbols:
            metric_symbols = [s.strip() for s in args.symbols.split(",")]
        else:
            from vibe_quant.data import universe

            universe_symbols, _as_of = universe.read_universe(universe.UNIVERSE_FILE)
            metric_symbols = list(dict.fromkeys([*SUPPORTED_SYMBOLS, *universe_symbols]))
        return download_metrics(
            metric_symbols,
            args.start,
            args.end or datetime.now(UTC).date() - timedelta(days=1),
            recheck=args.recheck,
            yes=args.yes,
            dry_run=args.dry_run,
        )
    elif args.command == "download-dvol":
        currencies = [c.strip().upper() for c in args.currencies.split(",")]
        return download_dvol(currencies, args.start, yes=args.yes)
    elif args.command == "ack-metrics":
        return ack_metrics(args.symbol, args.day, args.reason)
    elif args.command == "accept-gap":
        return accept_gap(args.symbol, args.interval, args.from_time, args.to_time, args.reason)
    elif args.command == "rebuild":
        rebuild_from_archive(verbose=True)
    elif args.command == "verify":
        archive = RawDataArchive()
        symbols = (
            [s.strip() for s in args.symbols.split(",")]
            if args.symbols
            else archive.get_symbols_with_mark()
        )

        if not symbols:
            print("No symbols to verify")
            archive.close()
            return 0

        print("\nData Verification Report")
        print("=" * 50)

        has_issues = False
        for symbol in symbols:
            if archive.get_kline_count(symbol, "1m") == 0 and (
                archive.get_kline_count(symbol, "1h") or archive.get_kline_count(symbol, "1d")
            ):
                has_issues |= _print_research_verify(archive, symbol)
            else:
                has_issues |= _print_kline_verify(archive, symbol)
            has_issues |= _print_positioning_verify(archive, symbol)
        _print_dvol_verify(archive)

        archive.close()

        print("\n" + "=" * 50)
        if has_issues:
            print("ISSUES FOUND")
            return 1
        print("ALL CHECKS PASSED")

    return 0


if __name__ == "__main__":
    sys.exit(main())
