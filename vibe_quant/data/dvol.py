"""Deribit DVOL (implied-volatility index) hourly candle downloader.

``public/get_volatility_index_data`` serves ``[open_ms, open, high, low, close]``
candles, at most 1000 per call, paged backward through ``continuation``; the
newest candle is still in progress. A candle is stored only once it closed at
least 60 s before the response's ``usOut``.

Resume: with candles archived, a run fetches range 1 = (C, now] and range 2 =
[start, min archived open), each paged backward, where C is the contiguous
coverage end: the last candle of the unbroken hourly run that starts at the
first archived candle. Range 1 thus restarts at the first missing candle, so a
run that failed after archiving newer pages leaves a hole the next run fills
(it re-fetches the newer candles; storage ignores them). A permanent gap at
the source makes every run re-fetch from that gap. Storage is ``INSERT OR
IGNORE`` with one commit per page: a candle re-served with different values
keeps its archived values (DVOL drift is not detected).
"""

from __future__ import annotations

import time
from datetime import UTC, date, datetime
from functools import partial
from typing import TYPE_CHECKING, Any

import httpx

from vibe_quant.data.downloader import DownloadError, _call_with_retry

if TYPE_CHECKING:
    from collections.abc import Iterator

    from vibe_quant.data.archive import RawDataArchive

DERIBIT_DVOL_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
DVOL_RESOLUTION_S = 3600
DVOL_SOURCE = "deribit"
# A candle is final this long after its end (server clock, usOut).
IN_PROGRESS_MARGIN_MS = 60_000

type Candle = tuple[int, float, float, float, float]


class DvolDownloadError(DownloadError):
    """A DVOL page failed; candles of the earlier pages stay archived."""


class _RpcError(Exception):
    """Deribit answered with a JSON-RPC ``error`` or a malformed reply (not retried)."""


def fetch_dvol(
    currency: str, start_ms: int, end_ms: int, client: httpx.Client
) -> Iterator[list[Candle]]:
    """Yield *currency*'s finished hourly candles in ``[start_ms, end_ms]``, newest page first.

    Raises:
        DownloadError: a page failed after retries, or Deribit returned an error.
    """
    end = end_ms
    while end >= start_ms:
        params = {
            "currency": currency,
            "start_timestamp": str(start_ms),
            "end_timestamp": str(end),
            "resolution": str(DVOL_RESOLUTION_S),
        }
        candles, continuation, now_ms = _call_with_retry(
            partial(_page, client, params), f"Deribit DVOL {currency}"
        )
        horizon = now_ms - DVOL_RESOLUTION_S * 1000 - IN_PROGRESS_MARGIN_MS
        yield [c for c in candles if c[0] <= horizon]
        if continuation is None:
            return
        if continuation >= end:
            raise DownloadError(f"Deribit DVOL {currency}: continuation {continuation} >= {end}")
        end = continuation


def _page(client: httpx.Client, params: dict[str, str]) -> tuple[list[Candle], int | None, int]:
    """Fetch one page: ``(candles, continuation, usOut in ms)``."""
    response = client.get(DERIBIT_DVOL_URL, params=params)
    if response.status_code >= 500 or response.status_code == 429:
        response.raise_for_status()  # retryable
    try:
        payload = response.json()
    except ValueError as exc:
        response.raise_for_status()
        raise _RpcError(f"non-JSON response {response.text[:200]!r}") from exc
    if isinstance(payload, dict) and "error" in payload:
        error = payload["error"]
        raise _RpcError(f"JSON-RPC error {error.get('code')}: {error.get('message')}")
    response.raise_for_status()
    result = _field(payload, "result")
    data = _field(result, "data")
    us_out = _field(payload, "usOut")
    continuation = result.get("continuation")
    try:
        candles = [(int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4])) for c in data]
    except (TypeError, ValueError, IndexError) as exc:
        raise _RpcError(f"reply field 'data' malformed: {exc!r}") from exc
    try:
        return candles, None if continuation is None else int(continuation), int(us_out) // 1000
    except (TypeError, ValueError) as exc:
        raise _RpcError(f"reply field 'continuation' or 'usOut' malformed: {exc!r}") from exc


def _field(obj: object, name: str) -> Any:
    if not isinstance(obj, dict) or name not in obj:
        raise _RpcError(f"reply field {name!r} missing")
    return obj[name]


def ingest_dvol(
    currency: str,
    archive: RawDataArchive,
    *,
    start: date,
    client: httpx.Client | None = None,
    sleep: float = 0.1,
) -> dict[str, int | None]:
    """Archive *currency*'s finished hourly DVOL candles from *start* to now.

    ``sleep`` seconds between pages keeps the rate at or below 10 req/s.

    Returns:
        ``inserted`` (new candles), ``first`` / ``last`` (archived open_time
        range in ms after the run, None when nothing is archived).

    Raises:
        DvolDownloadError: a page failed or a reply was malformed; earlier
            pages stay archived and the message names the contiguous coverage
            end (where a re-run resumes).
    """
    start_ms = int(datetime(start.year, start.month, start.day, tzinfo=UTC).timestamp() * 1000)
    now_ms = int(time.time() * 1000)
    archived = _archived_range(archive, currency)
    if archived is None:
        ranges = [(start_ms, now_ms)]
    else:
        ranges = [(_coverage_end(archive, currency) + 1, now_ms), (start_ms, archived[0] - 1)]
    own_client = client is None
    http = client if client is not None else httpx.Client(timeout=60.0)
    inserted = 0
    try:
        for lo, hi in ranges:
            for candles in fetch_dvol(currency, lo, hi, http):
                inserted += archive.insert_dvol(currency, DVOL_RESOLUTION_S, candles, DVOL_SOURCE)
                time.sleep(sleep)
    except DownloadError as exc:
        through = (
            _iso(_coverage_end(archive, currency))
            if _archived_range(archive, currency) is not None
            else "nothing"
        )
        raise DvolDownloadError(
            f"Deribit DVOL {currency} failed: {exc}; archived through {through};"
            " re-run download-dvol to resume"
        ) from exc
    finally:
        if own_client:
            http.close()
    final = _archived_range(archive, currency)
    return {
        "inserted": inserted,
        "first": final[0] if final is not None else None,
        "last": final[1] if final is not None else None,
    }


def _archived_range(archive: RawDataArchive, currency: str) -> tuple[int, int] | None:
    row = archive.conn.execute(
        "SELECT MIN(open_time), MAX(open_time) FROM raw_dvol"
        " WHERE currency = ? AND resolution_s = ?",
        (currency, DVOL_RESOLUTION_S),
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return int(row[0]), int(row[1])


def _coverage_end(archive: RawDataArchive, currency: str) -> int:
    """Open time of the last candle in the unbroken run from the first archived one."""
    row = archive.conn.execute(
        "SELECT MIN(a.open_time) FROM raw_dvol a"
        " WHERE a.currency = ? AND a.resolution_s = ? AND NOT EXISTS ("
        " SELECT 1 FROM raw_dvol b WHERE b.currency = a.currency"
        " AND b.resolution_s = a.resolution_s AND b.open_time = a.open_time + ?)",
        (currency, DVOL_RESOLUTION_S, DVOL_RESOLUTION_S * 1000),
    ).fetchone()
    return int(row[0])


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat()
