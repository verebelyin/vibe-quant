"""Tests for data ingestion module."""

from pathlib import Path

import pytest

from vibe_quant.data.archive import RawDataArchive


class TestRawDataArchive:
    """Tests for RawDataArchive."""

    @pytest.fixture
    def archive(self, tmp_path: Path) -> RawDataArchive:
        """Create archive with temp database."""
        db_path = tmp_path / "test_archive.db"
        arc = RawDataArchive(db_path)
        yield arc
        arc.close()

    def test_insert_and_get_klines(self, archive: RawDataArchive) -> None:
        """Should insert and retrieve klines."""
        klines = [
            (1704067200000, 42000.0, 42500.0, 41800.0, 42300.0, 100.0, 1704067259999),
            (1704067260000, 42300.0, 42600.0, 42200.0, 42500.0, 150.0, 1704067319999),
        ]
        count = archive.insert_klines("BTCUSDT", "1m", klines, "test")
        assert count == 2

        result = archive.get_klines("BTCUSDT", "1m")
        assert len(result) == 2
        assert result[0]["open_time"] == 1704067200000
        assert result[0]["open"] == 42000.0
        assert result[0]["close"] == 42300.0

    def test_insert_klines_returns_actual_count(self, archive: RawDataArchive) -> None:
        """insert_klines should return actual inserted count, not input count."""
        klines = [
            (1000, 100.0, 110.0, 90.0, 105.0, 1000.0, 2000, 50000.0, 100, 500.0, 25000.0),
            (2000, 105.0, 115.0, 95.0, 110.0, 1100.0, 3000, 55000.0, 110, 550.0, 27500.0),
        ]
        count1 = archive.insert_klines("BTCUSDT", "1m", klines, "test")
        assert count1 == 2

        # Insert same data again - should return 0 (all ignored)
        count2 = archive.insert_klines("BTCUSDT", "1m", klines, "test")
        assert count2 == 0

    def test_insert_funding_rates_returns_actual_count(self, archive: RawDataArchive) -> None:
        """insert_funding_rates should return actual inserted count, not input count."""
        rates = [
            (1704067200000, 0.0001, 42000.0),
            (1704096000000, -0.0002, 42500.0),
        ]
        count1 = archive.insert_funding_rates("BTCUSDT", rates, "test")
        assert count1 == 2

        # Insert same data again - should return 0 (all ignored)
        count2 = archive.insert_funding_rates("BTCUSDT", rates, "test")
        assert count2 == 0

    def test_insert_duplicate_klines_ignored(self, archive: RawDataArchive) -> None:
        """Should ignore duplicate klines (same symbol/interval/open_time)."""
        klines = [
            (1704067200000, 42000.0, 42500.0, 41800.0, 42300.0, 100.0, 1704067259999),
        ]
        archive.insert_klines("BTCUSDT", "1m", klines, "test")
        archive.insert_klines("BTCUSDT", "1m", klines, "test")

        result = archive.get_klines("BTCUSDT", "1m")
        assert len(result) == 1

    def test_insert_and_get_funding_rates(self, archive: RawDataArchive) -> None:
        """Should insert and retrieve funding rates."""
        rates = [
            (1704067200000, 0.0001, 42000.0),
            (1704096000000, -0.0002, 42500.0),
        ]
        count = archive.insert_funding_rates("BTCUSDT", rates, "test")
        assert count == 2

        result = archive.get_funding_rates("BTCUSDT")
        assert len(result) == 2
        assert result[0]["funding_rate"] == 0.0001
        assert result[1]["funding_rate"] == -0.0002

    def test_get_date_range(self, archive: RawDataArchive) -> None:
        """Should return correct date range."""
        klines = [
            (1704067200000, 42000.0, 42500.0, 41800.0, 42300.0, 100.0, 1704067259999),
            (1704153600000, 42300.0, 42600.0, 42200.0, 42500.0, 150.0, 1704153659999),
        ]
        archive.insert_klines("BTCUSDT", "1m", klines, "test")

        date_range = archive.get_date_range("BTCUSDT", "1m")
        assert date_range is not None
        assert date_range[0] == 1704067200000
        assert date_range[1] == 1704153600000

    def test_get_date_range_empty(self, archive: RawDataArchive) -> None:
        """Should return None for empty data."""
        date_range = archive.get_date_range("BTCUSDT", "1m")
        assert date_range is None

    def test_get_kline_count(self, archive: RawDataArchive) -> None:
        """Should return correct kline count."""
        klines = [
            (1704067200000, 42000.0, 42500.0, 41800.0, 42300.0, 100.0, 1704067259999),
            (1704067260000, 42300.0, 42600.0, 42200.0, 42500.0, 150.0, 1704067319999),
            (1704067320000, 42500.0, 42700.0, 42400.0, 42600.0, 120.0, 1704067379999),
        ]
        archive.insert_klines("BTCUSDT", "1m", klines, "test")

        count = archive.get_kline_count("BTCUSDT", "1m")
        assert count == 3

    def test_get_symbols(self, archive: RawDataArchive) -> None:
        """Should list all symbols in archive."""
        archive.insert_klines(
            "BTCUSDT",
            "1m",
            [(1704067200000, 42000.0, 42500.0, 41800.0, 42300.0, 100.0, 1704067259999)],
            "test",
        )
        archive.insert_klines(
            "ETHUSDT",
            "1m",
            [(1704067200000, 2200.0, 2250.0, 2180.0, 2230.0, 500.0, 1704067259999)],
            "test",
        )

        symbols = archive.get_symbols()
        assert "BTCUSDT" in symbols
        assert "ETHUSDT" in symbols
        assert len(symbols) == 2

    def test_filter_klines_by_time(self, archive: RawDataArchive) -> None:
        """Should filter klines by time range."""
        klines = [
            (1704067200000, 42000.0, 42500.0, 41800.0, 42300.0, 100.0, 1704067259999),
            (1704067260000, 42300.0, 42600.0, 42200.0, 42500.0, 150.0, 1704067319999),
            (1704067320000, 42500.0, 42700.0, 42400.0, 42600.0, 120.0, 1704067379999),
        ]
        archive.insert_klines("BTCUSDT", "1m", klines, "test")

        # Filter by start time
        result = archive.get_klines("BTCUSDT", "1m", start_time=1704067260000)
        assert len(result) == 2

        # Filter by end time
        result = archive.get_klines("BTCUSDT", "1m", end_time=1704067260000)
        assert len(result) == 2

        # Filter by range
        result = archive.get_klines(
            "BTCUSDT", "1m", start_time=1704067260000, end_time=1704067260000
        )
        assert len(result) == 1
