"""Tests for data verification module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vibe_quant.data.archive import RawDataArchive
from vibe_quant.data.verify import (
    check_ohlc_consistency,
    detect_gaps,
    validate_row_count,
    verify_symbol,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from pathlib import Path


class MockRow(dict[str, int | float]):
    """Mock sqlite3.Row for testing."""

    def __getitem__(self, key: str) -> int | float:
        return dict.__getitem__(self, key)


def make_klines(
    data: Sequence[tuple[int, float, float, float, float]],
) -> list[MockRow]:
    """Create mock kline rows from (open_time, open, high, low, close) tuples."""
    return [
        MockRow(
            {
                "open_time": d[0],
                "open": d[1],
                "high": d[2],
                "low": d[3],
                "close": d[4],
            }
        )
        for d in data
    ]


class TestDetectGaps:
    """Tests for detect_gaps function."""

    def test_detect_gaps_none(self) -> None:
        """No gaps in consecutive 1m data."""
        # 3 consecutive 1-minute klines (60000ms apart)
        klines = make_klines(
            [
                (1704067200000, 42000.0, 42100.0, 41900.0, 42050.0),
                (1704067260000, 42050.0, 42150.0, 41950.0, 42100.0),
                (1704067320000, 42100.0, 42200.0, 42000.0, 42150.0),
            ]
        )
        gaps = detect_gaps(klines, max_gap_minutes=5)
        assert gaps == []

    def test_detect_gaps_found(self) -> None:
        """Detects 10-minute gap."""
        # First kline, then 10-minute gap, then second kline
        klines = make_klines(
            [
                (1704067200000, 42000.0, 42100.0, 41900.0, 42050.0),
                (1704067800000, 42050.0, 42150.0, 41950.0, 42100.0),  # 10 min later
            ]
        )
        gaps = detect_gaps(klines, max_gap_minutes=5)
        assert len(gaps) == 1
        assert gaps[0][0] == 1704067200000  # start
        assert gaps[0][1] == 1704067800000  # end
        assert gaps[0][2] == 10  # gap_minutes

    def test_detect_gaps_boundary(self) -> None:
        """5-minute gap exactly at threshold should not be flagged."""
        # Exactly 5 minutes (300000 ms) apart
        klines = make_klines(
            [
                (1704067200000, 42000.0, 42100.0, 41900.0, 42050.0),
                (1704067500000, 42050.0, 42150.0, 41950.0, 42100.0),  # 5 min later
            ]
        )
        gaps = detect_gaps(klines, max_gap_minutes=5)
        assert gaps == []

    def test_detect_gaps_empty_data(self) -> None:
        """Empty data should return no gaps."""
        gaps = detect_gaps([], max_gap_minutes=5)
        assert gaps == []

    def test_detect_gaps_single_kline(self) -> None:
        """Single kline should return no gaps."""
        klines = make_klines(
            [
                (1704067200000, 42000.0, 42100.0, 41900.0, 42050.0),
            ]
        )
        gaps = detect_gaps(klines, max_gap_minutes=5)
        assert gaps == []

    def test_detect_gaps_multiple(self) -> None:
        """Detects multiple gaps."""
        klines = make_klines(
            [
                (1704067200000, 42000.0, 42100.0, 41900.0, 42050.0),
                (1704068000000, 42050.0, 42150.0, 41950.0, 42100.0),  # ~13 min gap
                (1704068060000, 42100.0, 42200.0, 42000.0, 42150.0),  # 1 min gap (ok)
                (1704069000000, 42150.0, 42250.0, 42050.0, 42200.0),  # ~15 min gap
            ]
        )
        gaps = detect_gaps(klines, max_gap_minutes=5)
        assert len(gaps) == 2


class TestCheckOhlcConsistency:
    """Tests for check_ohlc_consistency function."""

    def test_ohlc_consistency_valid(self) -> None:
        """Valid OHLC data passes."""
        klines = make_klines(
            [
                (1704067200000, 42000.0, 42500.0, 41800.0, 42300.0),
                (1704067260000, 42300.0, 42600.0, 42200.0, 42500.0),
            ]
        )
        errors = check_ohlc_consistency(klines)
        assert errors == []

    def test_ohlc_consistency_high_equals_low(self) -> None:
        """High equals low (doji) is valid."""
        klines = make_klines(
            [
                (1704067200000, 42000.0, 42000.0, 42000.0, 42000.0),
            ]
        )
        errors = check_ohlc_consistency(klines)
        assert errors == []

    def test_ohlc_consistency_high_less_than_low(self) -> None:
        """High < low fails."""
        klines = make_klines(
            [
                (1704067200000, 42000.0, 41500.0, 42500.0, 42300.0),  # high < low
            ]
        )
        errors = check_ohlc_consistency(klines)
        assert len(errors) >= 1
        assert errors[0][0] == 1704067200000
        assert "high" in errors[0][1] and "low" in errors[0][1]

    def test_ohlc_consistency_high_less_than_open(self) -> None:
        """High < open fails."""
        klines = make_klines(
            [
                (1704067200000, 42500.0, 42000.0, 41800.0, 42300.0),  # high < open
            ]
        )
        errors = check_ohlc_consistency(klines)
        assert len(errors) >= 1
        assert any("open" in e[1] for e in errors)

    def test_ohlc_consistency_high_less_than_close(self) -> None:
        """High < close fails."""
        klines = make_klines(
            [
                (1704067200000, 42000.0, 42100.0, 41800.0, 42500.0),  # high < close
            ]
        )
        errors = check_ohlc_consistency(klines)
        assert len(errors) >= 1
        assert any("close" in e[1] for e in errors)

    def test_ohlc_consistency_low_greater_than_open(self) -> None:
        """Low > open fails."""
        klines = make_klines(
            [
                (1704067200000, 41500.0, 42500.0, 42000.0, 42300.0),  # low > open
            ]
        )
        errors = check_ohlc_consistency(klines)
        assert len(errors) >= 1
        assert any("open" in e[1] for e in errors)

    def test_ohlc_consistency_low_greater_than_close(self) -> None:
        """Low > close fails."""
        klines = make_klines(
            [
                (1704067200000, 42000.0, 42500.0, 42400.0, 42100.0),  # low > close
            ]
        )
        errors = check_ohlc_consistency(klines)
        assert len(errors) >= 1
        assert any("close" in e[1] for e in errors)

    def test_ohlc_consistency_empty(self) -> None:
        """Empty data returns no errors."""
        errors = check_ohlc_consistency([])
        assert errors == []


class TestValidateRowCount:
    """Tests for validate_row_count function."""

    def test_validate_row_count_exact(self) -> None:
        """Exact count passes."""
        assert validate_row_count(1000, 1000) is True

    def test_validate_row_count_within_tolerance(self) -> None:
        """Count within 1% tolerance passes."""
        assert validate_row_count(995, 1000) is True
        assert validate_row_count(1005, 1000) is True
        assert validate_row_count(1010, 1000) is True

    def test_validate_row_count_outside_tolerance(self) -> None:
        """Count outside tolerance fails."""
        assert validate_row_count(950, 1000) is False
        assert validate_row_count(1050, 1000) is False

    def test_validate_row_count_custom_tolerance(self) -> None:
        """Custom tolerance works."""
        assert validate_row_count(900, 1000, tolerance=0.10) is True
        assert validate_row_count(850, 1000, tolerance=0.10) is False

    def test_validate_row_count_zero_expected(self) -> None:
        """Zero expected only passes if actual is also zero."""
        assert validate_row_count(0, 0) is True
        assert validate_row_count(1, 0) is False


class TestVerifySymbol:
    """Tests for verify_symbol function."""

    @pytest.fixture
    def archive(self, tmp_path: Path) -> Generator[RawDataArchive]:
        """Create archive with temp database."""
        db_path = tmp_path / "test_verify.db"
        arc = RawDataArchive(db_path)
        yield arc
        arc.close()

    def test_verify_symbol_clean_data(self, archive: RawDataArchive) -> None:
        """Clean data returns no errors."""
        klines = [
            (1704067200000, 42000.0, 42500.0, 41800.0, 42300.0, 100.0, 1704067259999),
            (1704067260000, 42300.0, 42600.0, 42200.0, 42500.0, 150.0, 1704067319999),
            (1704067320000, 42500.0, 42700.0, 42400.0, 42600.0, 120.0, 1704067379999),
        ]
        archive.insert_klines("BTCUSDT", "1m", klines, "test")

        result = verify_symbol(archive, "BTCUSDT")
        assert result["gaps"] == []
        assert result["ohlc_errors"] == []
        assert result["kline_count"] == 3

    def test_verify_symbol_with_gap(self, archive: RawDataArchive) -> None:
        """Data with gap is detected."""
        klines = [
            (1704067200000, 42000.0, 42500.0, 41800.0, 42300.0, 100.0, 1704067259999),
            (1704068000000, 42300.0, 42600.0, 42200.0, 42500.0, 150.0, 1704068059999),
        ]
        archive.insert_klines("BTCUSDT", "1m", klines, "test")

        result = verify_symbol(archive, "BTCUSDT")
        assert len(result["gaps"]) == 1
        assert result["kline_count"] == 2

    def test_verify_symbol_empty(self, archive: RawDataArchive) -> None:
        """Empty data returns zeros."""
        result = verify_symbol(archive, "BTCUSDT")
        assert result["gaps"] == []
        assert result["ohlc_errors"] == []
        assert result["kline_count"] == 0
