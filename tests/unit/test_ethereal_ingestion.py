"""Tests for Ethereal data ingestion module."""

from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from vibe_quant.ethereal.ingestion import (
    ETHEREAL_TIMEFRAMES,
    EtherealArchive,
    _safe_years_ago,
    archive_to_catalog,
    download_bars,
    download_funding_rates,
    generate_month_range,
    get_ethereal_bar_type,
    ingest_ethereal,
    ingest_ethereal_funding,
    klines_to_bars,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


class TestEtherealArchive:
    """Tests for EtherealArchive."""

    @pytest.fixture
    def archive(self, tmp_path: Path) -> Generator[EtherealArchive]:
        """Create archive with temp database."""
        db_path = tmp_path / "test_ethereal_archive.db"
        arc = EtherealArchive(db_path)
        yield arc
        arc.close()

    def test_insert_and_get_klines(self, archive: EtherealArchive) -> None:
        """Should insert and retrieve klines."""
        klines = [
            (1704067200000, 2500.0, 2550.0, 2480.0, 2530.0, 100.0, 1704067259999),
            (1704067260000, 2530.0, 2560.0, 2520.0, 2545.0, 150.0, 1704067319999),
        ]
        count = archive.insert_klines("ETHUSD", "1m", klines, "test")
        assert count == 2

        result = archive.get_klines("ETHUSD", "1m")
        assert len(result) == 2
        assert result[0]["open_time"] == 1704067200000
        assert result[0]["open"] == 2500.0
        assert result[0]["close"] == 2530.0

    def test_insert_duplicate_klines_ignored(self, archive: EtherealArchive) -> None:
        """Should ignore duplicate klines (same symbol/timeframe/open_time)."""
        klines = [
            (1704067200000, 2500.0, 2550.0, 2480.0, 2530.0, 100.0, 1704067259999),
        ]
        archive.insert_klines("ETHUSD", "1m", klines, "test")
        archive.insert_klines("ETHUSD", "1m", klines, "test")

        result = archive.get_klines("ETHUSD", "1m")
        assert len(result) == 1

    def test_insert_and_get_funding_rates(self, archive: EtherealArchive) -> None:
        """Should insert and retrieve funding rates."""
        rates = [
            (1704067200000, 0.0001, 2500.0),
            (1704070800000, -0.0002, 2520.0),  # 1 hour later (Ethereal hourly funding)
        ]
        count = archive.insert_funding_rates("ETHUSD", rates, "test")
        assert count == 2

        result = archive.get_funding_rates("ETHUSD")
        assert len(result) == 2
        assert result[0]["funding_rate"] == 0.0001
        assert result[1]["funding_rate"] == -0.0002

    def test_get_date_range(self, archive: EtherealArchive) -> None:
        """Should return correct date range."""
        klines = [
            (1704067200000, 2500.0, 2550.0, 2480.0, 2530.0, 100.0, 1704067259999),
            (1704153600000, 2530.0, 2560.0, 2520.0, 2545.0, 150.0, 1704153659999),
        ]
        archive.insert_klines("ETHUSD", "1m", klines, "test")

        date_range = archive.get_date_range("ETHUSD", "1m")
        assert date_range is not None
        assert date_range[0] == 1704067200000
        assert date_range[1] == 1704153600000

    def test_get_date_range_empty(self, archive: EtherealArchive) -> None:
        """Should return None for empty data."""
        date_range = archive.get_date_range("ETHUSD", "1m")
        assert date_range is None

    def test_get_kline_count(self, archive: EtherealArchive) -> None:
        """Should return correct kline count."""
        klines = [
            (1704067200000, 2500.0, 2550.0, 2480.0, 2530.0, 100.0, 1704067259999),
            (1704067260000, 2530.0, 2560.0, 2520.0, 2545.0, 150.0, 1704067319999),
            (1704067320000, 2545.0, 2570.0, 2540.0, 2555.0, 120.0, 1704067379999),
        ]
        archive.insert_klines("ETHUSD", "1m", klines, "test")

        count = archive.get_kline_count("ETHUSD", "1m")
        assert count == 3

    def test_get_symbols(self, archive: EtherealArchive) -> None:
        """Should list all symbols in archive."""
        archive.insert_klines(
            "BTCUSD",
            "1m",
            [(1704067200000, 42000.0, 42500.0, 41800.0, 42300.0, 100.0, 1704067259999)],
            "test",
        )
        archive.insert_klines(
            "ETHUSD",
            "1m",
            [(1704067200000, 2500.0, 2550.0, 2480.0, 2530.0, 500.0, 1704067259999)],
            "test",
        )

        symbols = archive.get_symbols()
        assert "BTCUSD" in symbols
        assert "ETHUSD" in symbols
        assert len(symbols) == 2

    def test_filter_klines_by_time(self, archive: EtherealArchive) -> None:
        """Should filter klines by time range."""
        klines = [
            (1704067200000, 2500.0, 2550.0, 2480.0, 2530.0, 100.0, 1704067259999),
            (1704067260000, 2530.0, 2560.0, 2520.0, 2545.0, 150.0, 1704067319999),
            (1704067320000, 2545.0, 2570.0, 2540.0, 2555.0, 120.0, 1704067379999),
        ]
        archive.insert_klines("ETHUSD", "1m", klines, "test")

        # Filter by start time
        result = archive.get_klines("ETHUSD", "1m", start_time=1704067260000)
        assert len(result) == 2

        # Filter by end time
        result = archive.get_klines("ETHUSD", "1m", end_time=1704067260000)
        assert len(result) == 2

        # Filter by range
        result = archive.get_klines(
            "ETHUSD", "1m", start_time=1704067260000, end_time=1704067260000
        )
        assert len(result) == 1

    def test_multiple_timeframes(self, archive: EtherealArchive) -> None:
        """Should store data for different timeframes separately."""
        klines_1m = [
            (1704067200000, 2500.0, 2550.0, 2480.0, 2530.0, 100.0, 1704067259999),
        ]
        klines_1h = [
            (1704067200000, 2500.0, 2600.0, 2450.0, 2580.0, 1000.0, 1704070799999),
        ]

        archive.insert_klines("ETHUSD", "1m", klines_1m, "test")
        archive.insert_klines("ETHUSD", "1h", klines_1h, "test")

        assert archive.get_kline_count("ETHUSD", "1m") == 1
        assert archive.get_kline_count("ETHUSD", "1h") == 1

        result_1m = archive.get_klines("ETHUSD", "1m")
        result_1h = archive.get_klines("ETHUSD", "1h")

        assert result_1m[0]["volume"] == 100.0
        assert result_1h[0]["volume"] == 1000.0


class TestGenerateMonthRange:
    """Tests for generate_month_range."""

    def test_single_month(self) -> None:
        """Should return single month for same-month range."""
        start = datetime(2024, 1, 15, tzinfo=UTC)
        end = datetime(2024, 1, 20, tzinfo=UTC)

        months = list(generate_month_range(start, end))
        assert months == [(2024, 1)]

    def test_multiple_months(self) -> None:
        """Should return all months in range."""
        start = datetime(2024, 1, 15, tzinfo=UTC)
        end = datetime(2024, 4, 20, tzinfo=UTC)

        months = list(generate_month_range(start, end))
        assert months == [(2024, 1), (2024, 2), (2024, 3), (2024, 4)]

    def test_year_boundary(self) -> None:
        """Should handle year boundary correctly."""
        start = datetime(2023, 11, 1, tzinfo=UTC)
        end = datetime(2024, 2, 1, tzinfo=UTC)

        months = list(generate_month_range(start, end))
        assert months == [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]


class TestDownloadBars:
    """Tests for download_bars with mocked HTTP responses."""

    def _create_mock_zip(self, csv_content: str) -> bytes:
        """Create a mock ZIP file with CSV content."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("ETHUSD-1m-2024-01.csv", csv_content)
        return buffer.getvalue()

    def test_download_bars_parses_csv(self) -> None:
        """Should parse CSV data from ZIP archive."""
        csv_data = """open_time,open,high,low,close,volume,close_time,quote_volume,trade_count
1704067200000,2500.0,2550.0,2480.0,2530.0,100.0,1704067259999,250000.0,50
1704067260000,2530.0,2560.0,2520.0,2545.0,150.0,1704067319999,380000.0,75
"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = self._create_mock_zip(csv_data)

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = (
                mock_response
            )

            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 1, 31, tzinfo=UTC)

            klines = download_bars("ETHUSD", "1m", start, end)

            assert len(klines) == 2
            assert klines[0][0] == 1704067200000  # open_time
            assert klines[0][1] == 2500.0  # open
            assert klines[0][4] == 2530.0  # close

    def test_download_bars_handles_404(self) -> None:
        """Should return empty list for 404 responses."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = (
                mock_response
            )

            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 1, 31, tzinfo=UTC)

            klines = download_bars("ETHUSD", "1m", start, end)
            assert klines == []

    def test_download_bars_invalid_timeframe(self) -> None:
        """Should raise ValueError for unsupported timeframe."""
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 31, tzinfo=UTC)

        with pytest.raises(ValueError, match="Unsupported timeframe"):
            download_bars("ETHUSD", "2m", start, end)


class TestDownloadFundingRates:
    """Tests for download_funding_rates with mocked HTTP responses."""

    def _create_mock_zip(self, csv_content: str) -> bytes:
        """Create a mock ZIP file with CSV content."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("ETHUSD-funding-2024-01.csv", csv_content)
        return buffer.getvalue()

    def test_download_funding_rates_parses_csv(self) -> None:
        """Should parse funding rate CSV data."""
        csv_data = """funding_time,funding_rate,mark_price
1704067200000,0.0001,2500.0
1704070800000,-0.0002,2520.0
"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = self._create_mock_zip(csv_data)

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = (
                mock_response
            )

            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 1, 31, tzinfo=UTC)

            rates = download_funding_rates("ETHUSD", start, end)

            assert len(rates) == 2
            assert rates[0][0] == 1704067200000  # funding_time
            assert rates[0][1] == 0.0001  # funding_rate
            assert rates[1][1] == -0.0002


class TestGetEtherealBarType:
    """Tests for get_ethereal_bar_type."""

    def test_bar_type_1m(self) -> None:
        """Should create correct bar type for 1m."""
        bar_type = get_ethereal_bar_type("ETHUSD", "1m")
        assert "ETHUSD-PERP" in str(bar_type.instrument_id)
        assert "ETHEREAL" in str(bar_type.instrument_id)

    def test_bar_type_1h(self) -> None:
        """Should create correct bar type for 1h."""
        bar_type = get_ethereal_bar_type("BTCUSD", "1h")
        assert "BTCUSD-PERP" in str(bar_type.instrument_id)

    def test_bar_type_1d(self) -> None:
        """Should create correct bar type for 1d."""
        bar_type = get_ethereal_bar_type("SOLUSD", "1d")
        assert "SOLUSD-PERP" in str(bar_type.instrument_id)


class TestKlinesToBars:
    """Tests for klines_to_bars."""

    def test_converts_klines_to_bars(self, tmp_path: Path) -> None:
        """Should convert archive klines to NT Bar objects."""
        archive = EtherealArchive(tmp_path / "test.db")
        klines = [
            (1704067200000, 2500.0, 2550.0, 2480.0, 2530.0, 100.0, 1704067259999),
            (1704067260000, 2530.0, 2560.0, 2520.0, 2545.0, 150.0, 1704067319999),
        ]
        archive.insert_klines("ETHUSD", "1m", klines, "test")
        rows = archive.get_klines("ETHUSD", "1m")

        bar_type = get_ethereal_bar_type("ETHUSD", "1m")
        bars = klines_to_bars(rows, bar_type)

        assert len(bars) == 2
        assert float(bars[0].open) == 2500.0
        assert float(bars[0].close) == 2530.0
        assert float(bars[1].volume) == 150.0

        # Check timestamps (ms to ns conversion)
        assert bars[0].ts_event == 1704067200000 * 1_000_000
        assert bars[0].ts_init == 1704067259999 * 1_000_000

        archive.close()


class TestIngestEthereal:
    """Tests for ingest_ethereal."""

    def test_ingest_stores_to_archive(self, tmp_path: Path) -> None:
        """Should download and store klines in archive."""
        archive = EtherealArchive(tmp_path / "test.db")

        # Mock the download_bars function
        mock_klines = [
            (1704067200000, 2500.0, 2550.0, 2480.0, 2530.0, 100.0, 1704067259999),
        ]

        with patch(
            "vibe_quant.ethereal.ingestion.download_bars", return_value=mock_klines
        ):
            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 1, 31, tzinfo=UTC)

            counts = ingest_ethereal("ETHUSD", "1m", start, end, archive=archive, verbose=False)

            assert counts["klines"] == 1
            assert archive.get_kline_count("ETHUSD", "1m") == 1

        archive.close()


class TestIngestEtherealFunding:
    """Tests for ingest_ethereal_funding."""

    def test_ingest_funding_stores_to_archive(self, tmp_path: Path) -> None:
        """Should download and store funding rates in archive."""
        archive = EtherealArchive(tmp_path / "test.db")

        mock_rates = [
            (1704067200000, 0.0001, 2500.0),
            (1704070800000, -0.0002, 2520.0),
        ]

        with patch(
            "vibe_quant.ethereal.ingestion.download_funding_rates",
            return_value=mock_rates,
        ):
            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 1, 31, tzinfo=UTC)

            count = ingest_ethereal_funding("ETHUSD", start, end, archive=archive, verbose=False)

            assert count == 2

            rates = archive.get_funding_rates("ETHUSD")
            assert len(rates) == 2

        archive.close()


class TestArchiveToCatalog:
    """Tests for archive_to_catalog."""

    def test_writes_to_catalog(self, tmp_path: Path) -> None:
        """Should convert archive data to catalog format."""
        archive_path = tmp_path / "archive.db"
        catalog_path = tmp_path / "catalog"

        # Set up archive with test data
        archive = EtherealArchive(archive_path)
        klines = [
            (1704067200000, 2500.0, 2550.0, 2480.0, 2530.0, 100.0, 1704067259999),
            (1704067260000, 2530.0, 2560.0, 2520.0, 2545.0, 150.0, 1704067319999),
        ]
        archive.insert_klines("ETHUSD", "1m", klines, "test")
        archive.close()

        # Convert to catalog
        results = archive_to_catalog(
            archive_path=archive_path,
            catalog_path=catalog_path,
            timeframes=["1m"],
            verbose=False,
        )

        assert "ETHUSD" in results
        assert results["ETHUSD"]["1m"] == 2
        assert catalog_path.exists()


class TestSafeYearsAgo:
    """Tests for _safe_years_ago helper."""

    def test_normal_date(self) -> None:
        """Normal date subtraction works."""
        d = datetime(2026, 6, 15, tzinfo=UTC)
        result = _safe_years_ago(d, 2)
        assert result == datetime(2024, 6, 15, tzinfo=UTC)

    def test_leap_day(self) -> None:
        """Leap day falls back to Feb 28."""
        d = datetime(2024, 2, 29, tzinfo=UTC)  # Leap year
        result = _safe_years_ago(d, 2)
        assert result == datetime(2022, 2, 28, tzinfo=UTC)

    def test_leap_to_leap(self) -> None:
        """Leap day to leap day preserves Feb 29."""
        d = datetime(2024, 2, 29, tzinfo=UTC)
        result = _safe_years_ago(d, 4)
        assert result == datetime(2020, 2, 29, tzinfo=UTC)


class TestSupportedTimeframes:
    """Tests for supported timeframes constant."""

    def test_all_timeframes_present(self) -> None:
        """Should have all expected timeframes."""
        expected = ["1m", "5m", "15m", "1h", "4h", "1d"]
        assert expected == ETHEREAL_TIMEFRAMES
