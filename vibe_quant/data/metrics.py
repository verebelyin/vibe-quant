"""Binance Vision daily ``metrics`` (open interest + long/short ratios) downloader.

Layout: ``<VISION_METRICS_BASE>/<SYM>/<SYM>-metrics-YYYY-MM-DD.zip`` (daily
only; there is no monthly archive), one CSV per zip with 288 five-minute rows,
not always sorted. The S3 listing's ETag is the zip's md5, so a day is fetched
again only when the listing names bytes the archive has not recorded:

- a normal run GETs a day iff the archive holds no ``ok`` file for it and no
  file with the listed md5;
- ``recheck`` GETs a day iff an ``ok`` file exists and no file has the listed
  md5, then records ``same``/``drift`` against the archived rows without ever
  touching them.

Bytes whose md5 != ETag after :data:`MD5_ATTEMPTS` GETs leave the day
``failed`` with nothing recorded, so the next run retries it.

Zips are fetched by :data:`FETCH_WORKERS` threads (each file is ~6 KB, so a
serial run waits ~0.5 s of latency per day); the archive is written by the
calling thread only, in listing order.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import math
import re
import xml.etree.ElementTree as ET
import zipfile
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, Literal

import httpx

from vibe_quant.data.archive import MetricsFileMeta, MetricsRow
from vibe_quant.data.downloader import DownloadError, _call_with_retry

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vibe_quant.data.archive import RawDataArchive

logger = logging.getLogger(__name__)

VISION_METRICS_BASE = "https://data.binance.vision/data/futures/um/daily/metrics"
VISION_S3_LISTING = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
METRICS_PREFIX = "data/futures/um/daily/metrics"
FETCH_WORKERS = 8  # concurrent zip GETs (Binance's own shell example fetches all at once)

METRICS_HEADER = (
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)

# GETs of one day before bytes whose md5 != listing ETag fail it.
MD5_ATTEMPTS = 3

COUNT_KEYS = ("files_ok", "rows_inserted", "skipped", "failed", "malformed", "same", "drift")

_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_FILENAME_RE = re.compile(r"^(?P<symbol>[A-Z0-9]+)-metrics-(?P<day>\d{4}-\d{2}-\d{2})\.zip$")
_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class VisionFile:
    """One daily metrics zip as the S3 listing names it (``md5`` = ETag)."""

    day: date
    filename: str
    md5: str
    size: int
    last_modified: str | None


@dataclass(frozen=True)
class ParsedMetrics:
    """A parsed zip: ``rows`` sorted by create_time and empty unless ``ok``."""

    status: Literal["ok", "malformed"]
    csv_rows: int
    rows: list[MetricsRow]
    detail: str | None = None


class _MalformedError(Exception):
    """A CSV row breaks the file; the message is the files row's ``detail``."""


def list_metrics_files(symbol: str, client: httpx.Client) -> list[VisionFile]:
    """Every daily metrics zip Vision lists for *symbol*, by day.

    Raises:
        DownloadError: a listing page failed after retries.
    """
    prefix = f"{METRICS_PREFIX}/{symbol}/"
    files: list[VisionFile] = []
    marker = ""
    while True:
        root = _call_with_retry(
            partial(_listing_page, client, {"prefix": prefix, "marker": marker}),
            f"Vision metrics listing {symbol}",
        )
        contents = list(root.iter(f"{_S3_NS}Contents"))
        keys = [c.findtext(f"{_S3_NS}Key", "") for c in contents]
        for content, key in zip(contents, keys, strict=True):
            match = _FILENAME_RE.match(key.rsplit("/", 1)[-1])
            if match is None or match["symbol"] != symbol:
                continue
            files.append(
                VisionFile(
                    day=date.fromisoformat(match["day"]),
                    filename=match.group(0),
                    md5=content.findtext(f"{_S3_NS}ETag", "").strip('"'),
                    size=int(content.findtext(f"{_S3_NS}Size", "0")),
                    last_modified=content.findtext(f"{_S3_NS}LastModified"),
                )
            )
        if root.findtext(f"{_S3_NS}IsTruncated") != "true":
            return sorted(files, key=lambda f: f.day)
        if not keys:
            raise DownloadError(f"Vision metrics listing {symbol}: truncated page with no keys")
        marker = root.findtext(f"{_S3_NS}NextMarker") or keys[-1]


def _listing_page(client: httpx.Client, params: dict[str, str]) -> ET.Element:
    response = client.get(VISION_S3_LISTING, params=params)
    response.raise_for_status()
    return ET.fromstring(response.content)


def parse_metrics_zip(content: bytes, filename: str, symbol: str) -> ParsedMetrics:
    """Parse one metrics zip; any defect makes the whole file ``malformed``.

    Rows are sorted by create_time; a blank field is None; a stamp repeated
    with identical values collapses to one row. Malformed: an unreadable zip,
    a bad header, no data rows, a bad field, a row outside the file's UTC day,
    a symbol other than *symbol*, or a repeated stamp with different values.

    Raises:
        ValueError: *filename* is not ``<SYMBOL>-metrics-YYYY-MM-DD.zip``.
    """
    match = _FILENAME_RE.match(filename)
    if match is None:
        raise ValueError(f"not a Vision metrics filename: {filename}")
    day = date.fromisoformat(match["day"])
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            text = zf.read(filename.removesuffix(".zip") + ".csv").decode("utf-8")
    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError, zlib.error, EOFError) as exc:
        return ParsedMetrics("malformed", 0, [], f"unreadable zip: {exc}")
    lines = [line for line in csv.reader(io.StringIO(text)) if line]
    body = lines[1:]
    if not lines or tuple(lines[0]) != METRICS_HEADER:
        header = ",".join(lines[0]) if lines else ""
        return ParsedMetrics("malformed", len(body), [], f"bad header {header[:120]!r}")
    if not body:
        return ParsedMetrics("malformed", 0, [], "no data rows")
    try:
        rows = _parse_rows(body, day, symbol)
    except _MalformedError as exc:
        return ParsedMetrics("malformed", len(body), [], str(exc))
    return ParsedMetrics("ok", len(body), rows)


def _parse_rows(body: list[list[str]], day: date, symbol: str) -> list[MetricsRow]:
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int((start + timedelta(days=1)).timestamp() * 1000)
    seen: dict[int, MetricsRow] = {}
    for line in body:
        if len(line) != len(METRICS_HEADER):
            raise _MalformedError(f"{len(line)} fields at {line[0]!r}")
        stamp, row_symbol = line[0], line[1]
        if row_symbol != symbol:
            raise _MalformedError(f"symbol {row_symbol} != {symbol}")
        try:
            ts = int(datetime.strptime(stamp, _TIME_FORMAT).replace(tzinfo=UTC).timestamp() * 1000)
        except ValueError:
            raise _MalformedError(f"bad create_time {stamp!r}") from None
        if not start_ms <= ts < end_ms:
            raise _MalformedError(f"create_time {stamp} outside {day.isoformat()}")
        v = [_field(value, stamp) for value in line[2:]]
        row: MetricsRow = (ts, v[0], v[1], v[2], v[3], v[4], v[5])
        previous = seen.setdefault(ts, row)
        if previous != row:
            raise _MalformedError(f"duplicate create_time {stamp} with different values")
    return [seen[ts] for ts in sorted(seen)]


def _field(value: str, stamp: str) -> float | None:
    if not value.strip():
        return None
    try:
        number = float(value)
    except ValueError:
        raise _MalformedError(f"bad value {value!r} at {stamp}") from None
    if not math.isfinite(number):
        raise _MalformedError(f"non-finite value {value!r} at {stamp}")
    return number


def ingest_metrics(
    symbol: str,
    archive: RawDataArchive,
    *,
    start: date | None = None,
    end: date | None = None,
    recheck: bool = False,
    client: httpx.Client | None = None,
) -> dict[str, int]:
    """Download *symbol*'s listed metrics days in ``[start, end]`` into *archive*.

    The listing is read before the archive is opened, so a listing failure
    leaves the archive untouched (not even created).

    Returns:
        Counts keyed by :data:`COUNT_KEYS`. ``skipped`` = days that needed no
        GET plus files already recorded; ``failed`` days are retried next run.

    Raises:
        DownloadError: the listing failed after retries.
    """
    own_client = client is None
    http = client if client is not None else httpx.Client(timeout=60.0)
    try:
        listed = [
            f
            for f in list_metrics_files(symbol, http)
            if (start is None or f.day >= start) and (end is None or f.day <= end)
        ]
        counts = dict.fromkeys(COUNT_KEYS, 0)
        if not listed:
            return counts
        recorded: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for row in archive.get_metrics_files(symbol):
            recorded[row["day"]].append((row["status"], row["md5"]))
        todo: list[VisionFile] = []
        for vision_file in listed:
            files = recorded[vision_file.day.isoformat()]
            has_ok = any(status == "ok" for status, _ in files)
            if any(md5 == vision_file.md5 for _, md5 in files) or has_ok != recheck:
                counts["skipped"] += 1
                continue
            todo.append(vision_file)
        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
            fetched = pool.map(partial(_fetch_or_error, symbol, client=http), todo)
            for vision_file, content in zip(todo, fetched, strict=True):
                _ingest_day(symbol, archive, vision_file, content, recheck, counts)
        return counts
    finally:
        if own_client:
            http.close()


def _fetch_or_error(
    symbol: str, vision_file: VisionFile, client: httpx.Client
) -> bytes | DownloadError | None:
    """:func:`_fetch_verified` with its DownloadError returned (worker threads never raise)."""
    try:
        return _fetch_verified(symbol, vision_file, client)
    except DownloadError as exc:
        return exc


def _ingest_day(
    symbol: str,
    archive: RawDataArchive,
    vision_file: VisionFile,
    content: bytes | DownloadError | None,
    recheck: bool,
    counts: dict[str, int],
) -> None:
    if isinstance(content, DownloadError):
        logger.warning("FAILED %s %s: %s", symbol, vision_file.day, content)
        counts["failed"] += 1
        return
    if content is None:
        counts["failed"] += 1
        return
    parsed = parse_metrics_zip(content, vision_file.filename, symbol)
    status: str = parsed.status
    detail = parsed.detail
    if status == "ok" and recheck:
        differ = _count_differences(archive, symbol, vision_file.day, parsed.rows)
        status, detail = ("drift", f"{differ} rows differ") if differ else ("same", None)
    meta = MetricsFileMeta(
        filename=vision_file.filename,
        md5=vision_file.md5,
        sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
        vision_last_modified=vision_file.last_modified,
        csv_rows=parsed.csv_rows,
        status=status,
        detail=detail,
    )
    inserted = archive.insert_metrics_file(symbol, vision_file.day, meta, parsed.rows)
    if status != "ok":
        logger.warning("%s %s %s (%s)", status.upper(), symbol, vision_file.day, detail)
        counts[status] += 1
    elif inserted:
        counts["files_ok"] += 1
        counts["rows_inserted"] += inserted
    else:
        counts["skipped"] += 1


def _fetch_verified(symbol: str, vision_file: VisionFile, client: httpx.Client) -> bytes | None:
    """The zip's bytes once their md5 equals the listing ETag; None after MD5_ATTEMPTS.

    A multipart ETag (``<hex>-<parts>``) is no md5: the zip's sha256 must
    equal the first token of its ``.CHECKSUM`` file instead.

    Raises:
        DownloadError: a GET failed after retries.
    """
    url = f"{VISION_METRICS_BASE}/{symbol}/{vision_file.filename}"
    multipart = "-" in vision_file.md5
    expected = vision_file.md5
    if multipart:
        text = _call_with_retry(lambda: _get(client, f"{url}.CHECKSUM"), f"Vision {url}.CHECKSUM")
        expected = (text.decode("ascii", "replace").split() or [""])[0]
    for attempt in range(1, MD5_ATTEMPTS + 1):
        content = _call_with_retry(lambda: _get(client, url), f"Vision metrics {url}")
        if multipart:
            digest = hashlib.sha256(content).hexdigest()
        else:
            digest = hashlib.md5(content, usedforsecurity=False).hexdigest()
        if digest == expected:
            return content
        logger.warning(
            "digest %s != expected %s for %s (attempt %d/%d)",
            digest,
            expected,
            vision_file.filename,
            attempt,
            MD5_ATTEMPTS,
        )
    return None


def _get(client: httpx.Client, url: str) -> bytes:
    response = client.get(url)
    response.raise_for_status()
    return response.content


def _count_differences(
    archive: RawDataArchive, symbol: str, day: date, rows: Sequence[MetricsRow]
) -> int:
    """Stamps whose values differ between *rows* and the archived day (either side missing)."""
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    start_ms = int(start.timestamp() * 1000)
    archived = {
        int(row["create_time"]): (int(row["create_time"]), *(row[c] for c in METRICS_HEADER[2:]))
        for row in archive.get_metrics(symbol, start_ms, start_ms + 86_400_000)
    }
    new = {row[0]: row for row in rows}
    return sum(1 for ts in archived.keys() | new.keys() if archived.get(ts) != new.get(ts))
