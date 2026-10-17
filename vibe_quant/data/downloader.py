"""Download historical data from Binance Vision and REST API."""

from __future__ import annotations

import csv
import io
import logging
import time
import zipfile
import zlib
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx

from vibe_quant.utils import generate_month_range

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Binance Vision base URL for futures data
BINANCE_VISION_BASE = "https://data.binance.vision/data/futures/um/monthly/klines"

# Binance Vision base URL for futures MARK-price data. Mark price is what
# liquidations trigger on; the zip layout and CSV format match normal klines
# (volume/taker columns are always 0, count is 60).
BINANCE_VISION_MARK_BASE = "https://data.binance.vision/data/futures/um/monthly/markPriceKlines"

# DAILY Vision archives. A monthly zip can silently omit whole UTC days (e.g.
# SOLUSDT-1m-2022-02 ships 25 of 28), so a per-day zip is the only way to fill
# such a hole. Same zip/CSV layout as the monthly files (2020 files have no
# header row, 2023+ do).
BINANCE_VISION_DAILY_BASE = "https://data.binance.vision/data/futures/um/daily/klines"
BINANCE_VISION_DAILY_MARK_BASE = (
    "https://data.binance.vision/data/futures/um/daily/markPriceKlines"
)

# Binance Vision base URL for funding-rate archives. Unlike the kline layouts
# there is no interval path segment: <base>/<SYMBOL>/<SYMBOL>-fundingRate-YYYY-MM.zip,
# whose CSV is ``calc_time,funding_interval_hours,last_funding_rate``.
BINANCE_VISION_FUNDING_BASE = "https://data.binance.vision/data/futures/um/monthly/fundingRate"

# Binance Futures REST API
BINANCE_FUTURES_API = "https://fapi.binance.com"

# Supported symbols (USDT-M perpetuals)
SUPPORTED_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "SUIUSDT",
    "ZECUSDT",
    "NEARUSDT",
    "HYPEUSDT",
]

# Transient-failure retry policy: delays between attempts (s).
MONTHLY_RETRY_DELAYS: tuple[float, ...] = (1.0, 2.0, 4.0)


class DownloadError(Exception):
    """A download failed after retries / with a non-retryable error."""


class MonthlyDownloadError(DownloadError):
    """A monthly archive could not be fetched (not a 404: the month may exist)."""


class DailyDownloadError(DownloadError):
    """A daily archive could not be fetched (not a 404: the day may exist)."""


class RestDownloadError(DownloadError):
    """A REST klines page failed; ``partial`` holds what was fetched before it.

    ``failed_from`` is the open_time (ms) the failed page started at.
    """

    def __init__(
        self, message: str, partial: list[tuple[Any, ...]], failed_from: int
    ) -> None:
        super().__init__(message)
        self.partial = partial
        self.failed_from = failed_from

    def __reduce__(self) -> tuple[Any, ...]:
        # Exception.args only carries ``message``; without this, unpickling would
        # call RestDownloadError(message) and fail on the two missing args.
        return (type(self), (str(self), self.partial, self.failed_from))


def _is_retryable(exc: Exception) -> bool:
    """Transient: transport errors, 5xx, 429 and truncated/corrupt zips."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code >= 500 or code == 429
    return isinstance(
        exc, (httpx.TransportError, zipfile.BadZipFile, zlib.error, EOFError)
    )


def _call_with_retry[T](fn: Callable[[], T], label: str) -> T:
    """Run *fn*, retrying transient errors on MONTHLY_RETRY_DELAYS backoff.

    Raises:
        DownloadError: non-retryable error, or retries exhausted.
    """
    attempts = len(MONTHLY_RETRY_DELAYS) + 1
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if not _is_retryable(exc):
                raise DownloadError(f"{label}: {exc!r}") from exc
            if attempt == attempts - 1:
                raise DownloadError(
                    f"{label}: gave up after {attempts} attempts: {exc!r}"
                ) from exc
            delay = MONTHLY_RETRY_DELAYS[attempt]
            logger.warning(
                "%s: transient error %r, retry %d/%d in %.1fs",
                label,
                exc,
                attempt + 1,
                attempts - 1,
                delay,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def download_monthly_klines(
    symbol: str,
    interval: str,
    year: int,
    month: int,
    timeout: float = 60.0,
    client: httpx.Client | None = None,
) -> list[tuple[Any, ...]] | None:
    """Download monthly klines from Binance Vision.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT').
        interval: Candle interval (e.g., '1m').
        year: Year to download.
        month: Month to download (1-12).
        timeout: Request timeout in seconds.
        client: Optional shared httpx.Client for connection reuse.

    Returns:
        List of kline tuples, or None when the month is not published (404).

    Raises:
        MonthlyDownloadError: Transient failures persisted through the retry
            schedule, or a non-retryable error (e.g. 403) occurred.
    """
    return _download_monthly_zip(
        BINANCE_VISION_BASE, symbol, interval, year, month, timeout, client
    )


def download_monthly_mark_klines(
    symbol: str,
    interval: str,
    year: int,
    month: int,
    timeout: float = 60.0,
    client: httpx.Client | None = None,
) -> list[tuple[Any, ...]] | None:
    """Download monthly MARK-price klines from Binance Vision.

    Same contract, zip layout and CSV format as :func:`download_monthly_klines`;
    only the ``markPriceKlines`` path differs. Available roughly from each
    symbol's listing month (BTCUSDT 2020-01, ETHUSDT 2020-01, BNBUSDT 2020-03,
    SOLUSDT 2020-10, SUIUSDT 2023-05, ZECUSDT 2020-02, NEARUSDT 2020-10,
    HYPEUSDT 2025-05); earlier months are a 404.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT').
        interval: Candle interval (e.g., '1m').
        year: Year to download.
        month: Month to download (1-12).
        timeout: Request timeout in seconds.
        client: Optional shared httpx.Client for connection reuse.

    Returns:
        List of kline tuples, or None when the month is not published (404).

    Raises:
        MonthlyDownloadError: Transient failures persisted through the retry
            schedule, or a non-retryable error (e.g. 403) occurred.
    """
    return _download_monthly_zip(
        BINANCE_VISION_MARK_BASE, symbol, interval, year, month, timeout, client
    )


def download_daily_klines(
    symbol: str,
    interval: str,
    day: date,
    timeout: float = 60.0,
    client: httpx.Client | None = None,
) -> list[tuple[Any, ...]] | None:
    """Download one UTC day of klines from Binance Vision DAILY archives.

    Monthly zips can silently omit whole days, so a day the monthly pass left
    empty has to be refetched individually. Same zip layout, CSV format and
    retry policy as :func:`download_monthly_klines`.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT').
        interval: Candle interval (e.g., '1m').
        day: UTC day to download.
        timeout: Request timeout in seconds.
        client: Optional shared httpx.Client for connection reuse.

    Returns:
        List of kline tuples, or None when the day is not published (404).

    Raises:
        DailyDownloadError: Transient failures persisted through the retry
            schedule, or a non-retryable error (e.g. 403) occurred.
    """
    return _download_daily_zip(
        BINANCE_VISION_DAILY_BASE, symbol, interval, day, timeout, client
    )


def download_daily_mark_klines(
    symbol: str,
    interval: str,
    day: date,
    timeout: float = 60.0,
    client: httpx.Client | None = None,
) -> list[tuple[Any, ...]] | None:
    """Download one UTC day of MARK-price klines from Binance Vision.

    Same contract as :func:`download_daily_klines`; only the
    ``markPriceKlines`` path differs.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT').
        interval: Candle interval (e.g., '1m').
        day: UTC day to download.
        timeout: Request timeout in seconds.
        client: Optional shared httpx.Client for connection reuse.

    Returns:
        List of kline tuples, or None when the day is not published (404).

    Raises:
        DailyDownloadError: Transient failures persisted through the retry
            schedule, or a non-retryable error (e.g. 403) occurred.
    """
    return _download_daily_zip(
        BINANCE_VISION_DAILY_MARK_BASE, symbol, interval, day, timeout, client
    )


def download_monthly_funding_intervals(
    symbol: str,
    year: int,
    month: int,
    timeout: float = 60.0,
    client: httpx.Client | None = None,
) -> list[tuple[int, int]] | None:
    """Download one month of funding intervals from Binance Vision.

    Vision publishes the ``funding_interval_hours`` in effect at each funding
    settlement, which is the only source that can tell a legitimate 8h gap from
    a hole on a 2h/4h symbol. The layout differs from klines (no interval path
    segment), so this has its own fetch, but it shares the retry policy.

    Args:
        symbol: Trading symbol (e.g., 'SOLUSDT').
        year: Year to download.
        month: Month to download (1-12).
        timeout: Request timeout in seconds.
        client: Optional shared httpx.Client for connection reuse.

    Returns:
        List of ``(calc_time_ms, interval_hours)`` tuples, or None when the
        month is not published (404).

    Raises:
        MonthlyDownloadError: Transient failures persisted through the retry
            schedule, or a non-retryable error (e.g. 403) occurred.
    """
    filename = f"{symbol}-fundingRate-{year}-{month:02d}.zip"
    url = f"{BINANCE_VISION_FUNDING_BASE}/{symbol}/{filename}"

    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=timeout)
    assert client is not None  # narrowing for mypy
    try:
        return _call_with_retry(
            lambda: _fetch_funding_intervals_zip(client, url, filename, symbol, year, month),
            f"{symbol} fundingRate {year}-{month:02d}",
        )
    except DownloadError as e:
        raise MonthlyDownloadError(str(e)) from e.__cause__
    finally:
        if own_client:
            client.close()


def _download_monthly_zip(
    base_url: str,
    symbol: str,
    interval: str,
    year: int,
    month: int,
    timeout: float,
    client: httpx.Client | None,
) -> list[tuple[Any, ...]] | None:
    """Fetch one monthly Vision zip from *base_url* under the shared contract."""
    # Format: BTCUSDT-1m-2024-01.zip
    filename = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    try:
        return _download_vision_zip(
            base_url,
            symbol,
            interval,
            filename,
            f"{symbol} {interval} {year}-{month:02d}",
            timeout,
            client,
        )
    except DownloadError as e:
        raise MonthlyDownloadError(str(e)) from e.__cause__


def _download_daily_zip(
    base_url: str,
    symbol: str,
    interval: str,
    day: date,
    timeout: float,
    client: httpx.Client | None,
) -> list[tuple[Any, ...]] | None:
    """Fetch one daily Vision zip from *base_url* under the shared contract."""
    # Format: BTCUSDT-1m-2024-01-04.zip
    filename = f"{symbol}-{interval}-{day.isoformat()}.zip"
    try:
        return _download_vision_zip(
            base_url,
            symbol,
            interval,
            filename,
            f"{symbol} {interval} {day.isoformat()}",
            timeout,
            client,
        )
    except DownloadError as e:
        raise DailyDownloadError(str(e)) from e.__cause__


def _download_vision_zip(
    base_url: str,
    symbol: str,
    interval: str,
    filename: str,
    label: str,
    timeout: float,
    client: httpx.Client | None,
) -> list[tuple[Any, ...]] | None:
    """GET + parse one Vision zip (monthly or daily), retrying transient errors.

    The URL layout is ``<base>/<symbol>/<interval>/<filename>``, and the CSV
    inside the zip shares the zip's basename. 404 stays None (not published);
    any other failure is retried on the shared backoff and then raised as
    :class:`DownloadError` for the caller to wrap.
    """
    url = f"{base_url}/{symbol}/{interval}/{filename}"

    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=timeout)
    assert client is not None  # narrowing for mypy
    try:
        return _call_with_retry(
            lambda: _fetch_vision_csv_zip(client, url, filename),
            label,
        )
    finally:
        if own_client:
            client.close()


def _fetch_vision_csv_zip(
    client: httpx.Client, url: str, filename: str
) -> list[tuple[Any, ...]] | None:
    """One attempt: GET + parse. None on 404."""
    response = client.get(url)
    if response.status_code == 404:
        return None
    response.raise_for_status()

    # Extract CSV from ZIP
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        csv_filename = filename.replace(".zip", ".csv")
        with zf.open(csv_filename) as f:
            reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
            klines: list[tuple[Any, ...]] = []
            for row in reader:
                # Skip header row if present
                if row[0] == "open_time":
                    continue
                # Binance kline format:
                # open_time, open, high, low, close, volume, close_time,
                # quote_volume, count, taker_buy_volume, taker_buy_quote_volume, ignore
                klines.append(
                    (
                        int(row[0]),  # open_time
                        float(row[1]),  # open
                        float(row[2]),  # high
                        float(row[3]),  # low
                        float(row[4]),  # close
                        float(row[5]),  # volume
                        int(row[6]),  # close_time
                        float(row[7]),  # quote_volume
                        int(row[8]),  # trade_count
                        float(row[9]),  # taker_buy_volume
                        float(row[10]),  # taker_buy_quote_volume
                    )
                )
            return klines


def _fetch_funding_intervals_zip(
    client: httpx.Client, url: str, filename: str, symbol: str, year: int, month: int
) -> list[tuple[int, int]] | None:
    """One attempt: GET + parse a fundingRate zip. None on 404.

    The CSV is ``calc_time,funding_interval_hours,last_funding_rate`` with a
    header. A row is kept only when its ``funding_interval_hours`` is a
    positive integer; the header, a blank/non-integer interval and ``0`` are
    dropped, and the dropped rows are reported once per file (never fatal).
    """
    response = client.get(url)
    if response.status_code == 404:
        return None
    response.raise_for_status()

    rows: list[tuple[int, int]] = []
    skipped = 0
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        csv_filename = filename.replace(".zip", ".csv")
        with zf.open(csv_filename) as f:
            reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
            for row in reader:
                # Skip the header and any malformed line (calc_time is digits).
                if not row or not row[0].isdigit():
                    continue
                try:
                    hours = int(row[1])
                except (IndexError, ValueError):
                    skipped += 1
                    continue
                if hours <= 0:
                    skipped += 1
                    continue
                rows.append((int(row[0]), hours))
    if skipped:
        logger.warning(
            "skipped %d funding interval rows for %s %d-%02d", skipped, symbol, year, month
        )
    return rows


def _to_float(value: Any, default: float = 0.0) -> float:
    """Parse an API numeric field, falling back to *default* when blank.

    Binance returns "" for missing numeric fields on older records.
    """
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    return float(value)


def download_funding_rates(
    symbol: str,
    start_time: int,
    end_time: int,
    timeout: float = 30.0,
) -> list[tuple[Any, ...]]:
    """Download funding rate history from Binance REST API.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT').
        start_time: Start timestamp in milliseconds.
        end_time: End timestamp in milliseconds.
        timeout: Request timeout in seconds.

    Returns:
        List of (funding_time, funding_rate, mark_price) tuples.
    """
    url = f"{BINANCE_FUTURES_API}/fapi/v1/fundingRate"
    all_rates: list[tuple[Any, ...]] = []
    skipped = 0
    limit = 1000  # Max limit

    with httpx.Client(timeout=timeout) as client:
        current_start = start_time

        while current_start < end_time:
            params: dict[str, str | int] = {
                "symbol": symbol,
                "startTime": current_start,
                "endTime": end_time,
                "limit": limit,
            }

            try:
                response = client.get(url, params=params)
                response.raise_for_status()
            except httpx.HTTPStatusError:
                logger.warning(
                    "Funding rate request failed for %s at %d: %s",
                    symbol,
                    current_start,
                    response.status_code,
                )
                break
            data = response.json()

            if not data:
                break

            for item in data:
                raw_rate = item.get("fundingRate")
                if raw_rate is None or (isinstance(raw_rate, str) and not raw_rate.strip()):
                    skipped += 1
                    continue
                all_rates.append(
                    (
                        item["fundingTime"],
                        float(raw_rate),
                        _to_float(item.get("markPrice")),
                    )
                )

            # A short page means there are no more records in the window
            if len(data) < limit:
                break

            # Move start time past the last received rate
            current_start = data[-1]["fundingTime"] + 1

    if skipped:
        logger.warning("skipped %d funding records with empty fundingRate", skipped)

    return all_rates


def download_recent_klines(
    symbol: str,
    interval: str,
    start_time: int,
    end_time: int,
    timeout: float = 30.0,
) -> list[tuple[Any, ...]]:
    """Download recent klines from Binance REST API.

    For filling gaps or getting data not yet in Binance Vision.

    Args:
        symbol: Trading symbol.
        interval: Candle interval.
        start_time: Start timestamp in milliseconds.
        end_time: End timestamp in milliseconds.
        timeout: Request timeout in seconds.

    Returns:
        List of kline tuples.

    Raises:
        RestDownloadError: a page failed after retries (carries the partial
            result and the start of the failed page).
    """
    url = f"{BINANCE_FUTURES_API}/fapi/v1/klines"
    all_klines: list[tuple[Any, ...]] = []

    with httpx.Client(timeout=timeout) as client:
        current_start = start_time

        while current_start < end_time:
            params: dict[str, str | int] = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_time,
                "limit": 1500,  # Max limit
            }

            def fetch_page(params: dict[str, str | int] = params) -> Any:
                response = client.get(url, params=params)
                response.raise_for_status()
                return response.json()

            try:
                data = _call_with_retry(
                    fetch_page, f"{symbol}/{interval} klines from {current_start}"
                )
            except DownloadError as e:
                raise RestDownloadError(str(e), all_klines, current_start) from e.__cause__

            if not data:
                break

            for k in data:
                all_klines.append(
                    (
                        int(k[0]),  # open_time
                        float(k[1]),  # open
                        float(k[2]),  # high
                        float(k[3]),  # low
                        float(k[4]),  # close
                        float(k[5]),  # volume
                        int(k[6]),  # close_time
                        float(k[7]),  # quote_volume
                        int(k[8]),  # trade_count
                        float(k[9]),  # taker_buy_volume
                        float(k[10]),  # taker_buy_quote_volume
                    )
                )

            # Move start time past the last candle
            current_start = data[-1][0] + 1

    return all_klines


def get_years_months_to_download(years: int = 2) -> list[tuple[int, int]]:
    """Get list of (year, month) to download for N years of history.

    Args:
        years: Number of years of history to download.

    Returns:
        List of (year, month) tuples.
    """
    now = datetime.now(UTC)
    start = now - timedelta(days=365 * years)
    return get_months_in_range(start, now)


def get_months_in_range(start: datetime, end: datetime) -> list[tuple[int, int]]:
    """Get list of (year, month) for complete months in a date range.

    Only includes months that are fully completed (before the current month).
    Binance Vision publishes monthly archives after the month ends.

    Args:
        start: Start date (inclusive).
        end: End date (upper bound).

    Returns:
        List of (year, month) tuples for complete months.
    """
    # End at the previous complete month relative to end date
    if end.month == 1:
        last_complete = end.replace(year=end.year - 1, month=12, day=1)
    else:
        last_complete = end.replace(month=end.month - 1, day=1)

    return list(generate_month_range(start, last_complete))
