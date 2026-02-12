"""Download historical data from Binance Vision and REST API."""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Generator

# Binance Vision base URL for futures data
BINANCE_VISION_BASE = "https://data.binance.vision/data/futures/um/monthly/klines"

# Binance Futures REST API
BINANCE_FUTURES_API = "https://fapi.binance.com"

# Supported symbols (USDT-M perpetuals)
SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


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
        List of kline tuples or None if not available.
    """
    # Format: BTCUSDT-1m-2024-01.zip
    filename = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    url = f"{BINANCE_VISION_BASE}/{symbol}/{interval}/{filename}"

    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=timeout)
    assert client is not None  # narrowing for mypy
    try:
        response = client.get(url)
        if response.status_code == 404:
            return None
        response.raise_for_status()

        # Extract CSV from ZIP
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            csv_filename = filename.replace(".zip", ".csv")
            with zf.open(csv_filename) as f:
                reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
                klines = []
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
    except httpx.HTTPStatusError:
        return None
    except Exception:
        logger.exception("Unexpected error downloading %s %s/%s-%02d", symbol, interval, year, month)
        return None
    finally:
        if own_client:
            client.close()


def generate_month_range(
    start_date: datetime, end_date: datetime
) -> Generator[tuple[int, int]]:
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
        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)


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

    with httpx.Client(timeout=timeout) as client:
        current_start = start_time

        while current_start < end_time:
            params: dict[str, str | int] = {
                "symbol": symbol,
                "startTime": current_start,
                "endTime": end_time,
                "limit": 1000,  # Max limit
            }

            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            for item in data:
                all_rates.append(
                    (
                        item["fundingTime"],
                        float(item["fundingRate"]),
                        float(item.get("markPrice", 0)),
                    )
                )

            # Move start time past the last received rate
            current_start = data[-1]["fundingTime"] + 1

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

            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

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


def get_months_in_range(
    start: datetime, end: datetime
) -> list[tuple[int, int]]:
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
