"""Tests for screening-to-validation consistency checker."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from vibe_quant.db.connection import DEFAULT_DB_PATH
from vibe_quant.screening.consistency import (
    ConsistencyChecker,
    ConsistencyResult,
    check_consistency,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Create temp database with test data using real schema."""
    from vibe_quant.db.connection import get_connection
    from vibe_quant.db.schema import init_schema

    db_file = tmp_path / "test_consistency.db"
    conn = get_connection(db_file)
    init_schema(conn)

    # Create strategies
    conn.execute("INSERT INTO strategies (id, name, dsl_config) VALUES (1, 'test_strategy', '{}')")
    conn.execute("INSERT INTO strategies (id, name, dsl_config) VALUES (2, 'improved_strategy', '{}')")
    conn.execute("INSERT INTO strategies (id, name, dsl_config) VALUES (3, 'sensitive_strategy', '{}')")

    # Create screening runs (run_ids 1-3)
    for i in range(1, 4):
        conn.execute(
            """INSERT INTO backtest_runs (id, strategy_id, run_mode, symbols, timeframe,
               start_date, end_date, parameters, status)
               VALUES (?, ?, 'screening', '["BTCUSDT"]', '5m', '2024-01-01', '2024-12-31', '{}', 'completed')""",
            (i, i),
        )

    # Create validation runs (run_ids 4-6, same strategy_id 1-3)
    for i in range(4, 7):
        conn.execute(
            """INSERT INTO backtest_runs (id, strategy_id, run_mode, symbols, timeframe,
               start_date, end_date, parameters, status)
               VALUES (?, ?, 'validation', '["BTCUSDT"]', '5m', '2024-01-01', '2024-12-31', '{}', 'completed')""",
            (i, i - 3),
        )

    # Sweep results (screening) - run_ids 1,2,3
    conn.execute("INSERT INTO sweep_results (run_id, parameters, sharpe_ratio, total_return) VALUES (1, '{\"rsi_period\": 14}', 2.0, 50.0)")
    conn.execute("INSERT INTO sweep_results (run_id, parameters, sharpe_ratio, total_return) VALUES (2, '{\"ema_period\": 20}', 1.5, 30.0)")
    conn.execute("INSERT INTO sweep_results (run_id, parameters, sharpe_ratio, total_return) VALUES (3, '{\"period\": 10}', 3.0, 100.0)")

    # Backtest results (validation) - run_ids 4,5,6
    conn.execute("INSERT INTO backtest_results (run_id, total_return, sharpe_ratio) VALUES (4, 45.0, 1.8)")
    conn.execute("INSERT INTO backtest_results (run_id, total_return, sharpe_ratio) VALUES (5, 40.0, 2.0)")
    conn.execute("INSERT INTO backtest_results (run_id, total_return, sharpe_ratio) VALUES (6, 30.0, 1.0)")

    conn.commit()
    conn.close()
    return db_file


class TestConsistencyResult:
    """Tests for ConsistencyResult dataclass."""

    def test_create_result(self) -> None:
        """Can create consistency result."""
        result = ConsistencyResult(
            strategy_name="test",
            screening_run_id=1,
            validation_run_id=1,
            screening_sharpe=2.0,
            validation_sharpe=1.5,
            sharpe_degradation=0.25,
            screening_return=50.0,
            validation_return=40.0,
            return_degradation=0.20,
            is_execution_sensitive=False,
            parameters='{"period": 14}',
            checked_at="2026-01-01T00:00:00",
        )
        assert result.strategy_name == "test"
        assert result.sharpe_degradation == 0.25
        assert not result.is_execution_sensitive

    def test_result_is_frozen(self) -> None:
        """Result should be immutable."""
        result = ConsistencyResult(
            strategy_name="test",
            screening_run_id=1,
            validation_run_id=1,
            screening_sharpe=2.0,
            validation_sharpe=1.5,
            sharpe_degradation=0.25,
            screening_return=50.0,
            validation_return=40.0,
            return_degradation=0.20,
            is_execution_sensitive=False,
            parameters='{}',
            checked_at="2026-01-01T00:00:00",
        )
        with pytest.raises(AttributeError):
            result.strategy_name = "changed"  # type: ignore[misc]


class TestConsistencyChecker:
    """Tests for ConsistencyChecker class."""

    def test_init_default_db_path(self) -> None:
        """Default db_path uses canonical DEFAULT_DB_PATH."""
        checker = ConsistencyChecker()
        assert checker.db_path == DEFAULT_DB_PATH

    def test_init_creates_table(self, db_path: Path) -> None:
        """Checker creates consistency_checks table."""
        checker = ConsistencyChecker(db_path)

        # Table should exist after accessing conn
        _ = checker.conn

        # Verify table exists
        row = checker.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='consistency_checks'"
        ).fetchone()
        assert row is not None

        checker.close()

    def test_check_consistency_slight_degradation(self, db_path: Path) -> None:
        """Check detects slight degradation."""
        checker = ConsistencyChecker(db_path)
        result = checker.check_consistency(1, 4)

        assert result.strategy_name == "test_strategy"
        assert result.screening_sharpe == 2.0
        assert result.validation_sharpe == 1.8
        assert result.sharpe_degradation == pytest.approx(0.1)  # 10% degradation
        assert not result.is_execution_sensitive  # Below 50% threshold

        checker.close()

    def test_check_consistency_improved(self, db_path: Path) -> None:
        """Check detects improvement."""
        checker = ConsistencyChecker(db_path)
        result = checker.check_consistency(2, 5)

        assert result.strategy_name == "improved_strategy"
        assert result.screening_sharpe == 1.5
        assert result.validation_sharpe == 2.0
        assert result.sharpe_degradation < 0  # Negative = improved
        assert not result.is_execution_sensitive

        checker.close()

    def test_check_consistency_execution_sensitive(self, db_path: Path) -> None:
        """Check flags execution-sensitive strategies."""
        checker = ConsistencyChecker(db_path)
        result = checker.check_consistency(3, 6)

        assert result.strategy_name == "sensitive_strategy"
        assert result.screening_sharpe == 3.0
        assert result.validation_sharpe == 1.0
        # (3.0 - 1.0) / 3.0 = 0.667 > 0.50
        assert result.sharpe_degradation > 0.50
        assert result.is_execution_sensitive

        checker.close()

    def test_check_invalid_screening_id(self, db_path: Path) -> None:
        """Check raises for invalid screening ID."""
        checker = ConsistencyChecker(db_path)

        with pytest.raises(ValueError, match="Screening run 999 not found"):
            checker.check_consistency(999, 4)

        checker.close()

    def test_check_invalid_validation_id(self, db_path: Path) -> None:
        """Check raises for invalid validation ID."""
        checker = ConsistencyChecker(db_path)

        with pytest.raises(ValueError, match="Validation run 999 not found"):
            checker.check_consistency(1, 999)

        checker.close()

    def test_check_saves_to_database(self, db_path: Path) -> None:
        """Check saves result to consistency_checks table."""
        checker = ConsistencyChecker(db_path)
        checker.check_consistency(1, 4)

        row = checker.conn.execute(
            "SELECT * FROM consistency_checks WHERE screening_run_id = 1"
        ).fetchone()

        assert row is not None
        assert row["strategy_name"] == "test_strategy"
        assert row["screening_sharpe"] == 2.0
        assert row["validation_sharpe"] == 1.8

        checker.close()

    def test_check_batch(self, db_path: Path) -> None:
        """Can check multiple pairs at once."""
        checker = ConsistencyChecker(db_path)

        results = checker.check_batch([
            (1, 4),
            (2, 5),
            (3, 6),
        ])

        assert len(results) == 3
        assert results[0].strategy_name == "test_strategy"
        assert results[1].strategy_name == "improved_strategy"
        assert results[2].strategy_name == "sensitive_strategy"

        checker.close()

    def test_get_execution_sensitive(self, db_path: Path) -> None:
        """Can retrieve execution-sensitive strategies."""
        checker = ConsistencyChecker(db_path)

        # First run checks to populate database
        checker.check_batch([(1, 4), (2, 5), (3, 6)])

        sensitive = checker.get_execution_sensitive()

        assert len(sensitive) == 1
        assert sensitive[0].strategy_name == "sensitive_strategy"
        assert sensitive[0].is_execution_sensitive

        checker.close()

    def test_get_improved(self, db_path: Path) -> None:
        """Can retrieve improved strategies."""
        checker = ConsistencyChecker(db_path)

        # First run checks to populate database
        checker.check_batch([(1, 4), (2, 5), (3, 6)])

        improved = checker.get_improved()

        assert len(improved) == 1
        assert improved[0].strategy_name == "improved_strategy"
        assert improved[0].sharpe_degradation < 0

        checker.close()


class TestDegradationCalculation:
    """Tests for degradation calculation edge cases."""

    def test_zero_screening_positive_validation(self, db_path: Path) -> None:
        """Handle screening=0, validation>0 case."""
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")

        # Insert strategy + backtest_runs + results
        conn.execute("INSERT INTO strategies (id, name, dsl_config) VALUES (10, 'zero_screen', '{}')")
        conn.execute(
            """INSERT INTO backtest_runs (id, strategy_id, run_mode, symbols, timeframe,
               start_date, end_date, parameters, status)
               VALUES (10, 10, 'screening', '["BTCUSDT"]', '5m', '2024-01-01', '2024-12-31', '{}', 'completed')"""
        )
        conn.execute(
            """INSERT INTO backtest_runs (id, strategy_id, run_mode, symbols, timeframe,
               start_date, end_date, parameters, status)
               VALUES (11, 10, 'validation', '["BTCUSDT"]', '5m', '2024-01-01', '2024-12-31', '{}', 'completed')"""
        )
        conn.execute(
            "INSERT INTO sweep_results (run_id, parameters, sharpe_ratio, total_return) VALUES (10, '{}', 0.0, 0.0)"
        )
        conn.execute(
            "INSERT INTO backtest_results (run_id, sharpe_ratio, total_return) VALUES (11, 1.0, 10.0)"
        )
        conn.commit()
        conn.close()

        checker = ConsistencyChecker(db_path)
        result = checker.check_consistency(10, 11)

        # Improved from 0 to positive
        assert result.sharpe_degradation == -1.0
        assert not result.is_execution_sensitive

        checker.close()

    def test_zero_screening_negative_validation(self, db_path: Path) -> None:
        """Handle screening=0, validation<0 case."""
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")

        # Insert strategy + backtest_runs + results
        conn.execute("INSERT INTO strategies (id, name, dsl_config) VALUES (11, 'zero_neg', '{}')")
        conn.execute(
            """INSERT INTO backtest_runs (id, strategy_id, run_mode, symbols, timeframe,
               start_date, end_date, parameters, status)
               VALUES (12, 11, 'screening', '["BTCUSDT"]', '5m', '2024-01-01', '2024-12-31', '{}', 'completed')"""
        )
        conn.execute(
            """INSERT INTO backtest_runs (id, strategy_id, run_mode, symbols, timeframe,
               start_date, end_date, parameters, status)
               VALUES (13, 11, 'validation', '["BTCUSDT"]', '5m', '2024-01-01', '2024-12-31', '{}', 'completed')"""
        )
        conn.execute(
            "INSERT INTO sweep_results (run_id, parameters, sharpe_ratio, total_return) VALUES (12, '{}', 0.0, 0.0)"
        )
        conn.execute(
            "INSERT INTO backtest_results (run_id, sharpe_ratio, total_return) VALUES (13, -1.0, -10.0)"
        )
        conn.commit()
        conn.close()

        checker = ConsistencyChecker(db_path)
        result = checker.check_consistency(12, 13)

        # Degraded from 0 to negative
        assert result.sharpe_degradation == 1.0
        assert result.is_execution_sensitive

        checker.close()


class TestGenerateReport:
    """Tests for report generation."""

    def test_empty_report(self, db_path: Path) -> None:
        """Empty list produces minimal report."""
        checker = ConsistencyChecker(db_path)
        report = checker.generate_report([])

        assert "No consistency checks" in report
        checker.close()

    def test_report_with_results(self, db_path: Path) -> None:
        """Report includes all categories."""
        checker = ConsistencyChecker(db_path)
        results = checker.check_batch([(1, 4), (2, 5), (3, 6)])

        report = checker.generate_report(results)

        assert "CONSISTENCY REPORT" in report
        assert "Total checked: 3" in report
        assert "Execution-sensitive" in report
        assert "sensitive_strategy" in report
        assert "improved_strategy" in report

        checker.close()


class TestConvenienceFunction:
    """Tests for check_consistency convenience function."""

    def test_check_consistency_function(self, db_path: Path) -> None:
        """Convenience function works."""
        result = check_consistency(1, 4, db_path)

        assert result.strategy_name == "test_strategy"
        assert result.screening_run_id == 1
        assert result.validation_run_id == 4
