"""Tests for overfitting CLI module."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from vibe_quant.overfitting.__main__ import (
    cmd_report,
    cmd_run,
    main,
    parse_filters,
)
from vibe_quant.overfitting.pipeline import FilterConfig


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Create temp database with test data."""
    db_file = tmp_path / "test_cli.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("PRAGMA journal_mode=WAL")

    # Create tables
    conn.execute("""
        CREATE TABLE backtest_runs (
            id INTEGER PRIMARY KEY,
            strategy_id INTEGER,
            run_mode TEXT NOT NULL,
            symbols JSON NOT NULL,
            timeframe TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            parameters JSON NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE sweep_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            parameters TEXT NOT NULL,
            sharpe_ratio REAL,
            sortino_ratio REAL,
            max_drawdown REAL,
            total_return REAL,
            profit_factor REAL,
            win_rate REAL,
            total_trades INTEGER,
            total_fees REAL,
            total_funding REAL,
            is_pareto_optimal BOOLEAN DEFAULT 0,
            passed_deflated_sharpe BOOLEAN,
            passed_walk_forward BOOLEAN,
            passed_purged_kfold BOOLEAN
        )
    """)

    # Insert test data
    conn.execute(
        "INSERT INTO backtest_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (1, 1, "screening", '["BTCUSDT"]', "1h", "2024-01-01", "2024-12-31", "{}"),
    )

    # Insert sweep results
    for i, (sharpe, ret) in enumerate([(2.5, 50.0), (1.5, 30.0), (0.5, 10.0)]):
        conn.execute(
            """INSERT INTO sweep_results
               (run_id, parameters, sharpe_ratio, total_return)
               VALUES (?, ?, ?, ?)""",
            (1, f'{{"period": {14 + i * 7}}}', sharpe, ret),
        )

    conn.commit()
    conn.close()
    return db_file


class TestParseFilters:
    """Tests for parse_filters function."""

    def test_empty_string_enables_all(self) -> None:
        """Empty string enables all filters."""
        config = parse_filters("")
        assert config.enable_dsr is True
        assert config.enable_wfa is True
        assert config.enable_purged_kfold is True

    def test_all_keyword(self) -> None:
        """'all' keyword enables all filters."""
        config = parse_filters("all")
        assert config.enable_dsr is True
        assert config.enable_wfa is True
        assert config.enable_purged_kfold is True

    def test_dsr_only(self) -> None:
        """'dsr' enables only DSR filter."""
        config = parse_filters("dsr")
        assert config.enable_dsr is True
        assert config.enable_wfa is False
        assert config.enable_purged_kfold is False

    def test_wfa_only(self) -> None:
        """'wfa' enables only WFA filter."""
        config = parse_filters("wfa")
        assert config.enable_dsr is False
        assert config.enable_wfa is True
        assert config.enable_purged_kfold is False

    def test_pkfold_alias(self) -> None:
        """'pkfold' enables Purged K-Fold filter."""
        config = parse_filters("pkfold")
        assert config.enable_dsr is False
        assert config.enable_wfa is False
        assert config.enable_purged_kfold is True

    def test_cv_alias(self) -> None:
        """'cv' enables Purged K-Fold filter."""
        config = parse_filters("cv")
        assert config.enable_purged_kfold is True

    def test_multiple_filters(self) -> None:
        """Can enable multiple filters."""
        config = parse_filters("dsr,wfa")
        assert config.enable_dsr is True
        assert config.enable_wfa is True
        assert config.enable_purged_kfold is False

    def test_case_insensitive(self) -> None:
        """Filter names are case-insensitive."""
        config = parse_filters("DSR,WFA,PKFOLD")
        assert config.enable_dsr is True
        assert config.enable_wfa is True
        assert config.enable_purged_kfold is True

    def test_whitespace_handling(self) -> None:
        """Handles whitespace around filter names."""
        config = parse_filters(" dsr , wfa ")
        assert config.enable_dsr is True
        assert config.enable_wfa is True


class TestCmdRun:
    """Tests for cmd_run function."""

    def test_run_dsr_only(self, db_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Run with DSR filter only."""
        from argparse import Namespace

        args = Namespace(
            run_id=1,
            filters="dsr",
            db=str(db_path),
            observations=252,
            start_date=None,
            end_date=None,
            samples=1000,
            output=None,
        )

        result = cmd_run(args)
        assert result == 0

        captured = capsys.readouterr()
        assert "OVERFITTING PIPELINE SUMMARY" in captured.out
        assert "3 candidates" in captured.out
        assert "passed DSR" in captured.out

    def test_run_all_filters(self, db_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Run with all filters."""
        from argparse import Namespace

        args = Namespace(
            run_id=1,
            filters="all",
            db=str(db_path),
            observations=252,
            start_date="2024-01-01",
            end_date="2025-12-31",
            samples=1000,
            output=None,
        )

        result = cmd_run(args)
        assert result == 0

        captured = capsys.readouterr()
        assert "DSR (Deflated Sharpe)" in captured.out
        assert "WFA (Walk-Forward)" in captured.out
        assert "PKFOLD (Purged K-Fold CV)" in captured.out

    def test_run_with_output(self, db_path: Path, tmp_path: Path) -> None:
        """Run with output file."""
        from argparse import Namespace

        output_file = tmp_path / "report.txt"
        args = Namespace(
            run_id=1,
            filters="dsr",
            db=str(db_path),
            observations=252,
            start_date=None,
            end_date=None,
            samples=1000,
            output=str(output_file),
        )

        result = cmd_run(args)
        assert result == 0
        assert output_file.exists()

        content = output_file.read_text()
        assert "OVERFITTING PREVENTION PIPELINE REPORT" in content

    def test_run_no_candidates(self, db_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Run with no candidates."""
        from argparse import Namespace

        args = Namespace(
            run_id=999,  # Non-existent
            filters="dsr",
            db=str(db_path),
            observations=252,
            start_date=None,
            end_date=None,
            samples=1000,
            output=None,
        )

        result = cmd_run(args)
        assert result == 0

        captured = capsys.readouterr()
        assert "0 candidates" in captured.out


class TestCmdReport:
    """Tests for cmd_report function."""

    def test_report_basic(self, db_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Basic report shows candidates."""
        from argparse import Namespace

        # First run pipeline to set flags
        from vibe_quant.overfitting.pipeline import OverfittingPipeline

        pipeline = OverfittingPipeline(db_path)
        pipeline.run(run_id=1, config=FilterConfig.dsr_only())
        pipeline.close()

        args = Namespace(
            run_id=1,
            db=str(db_path),
            any_filter=False,
        )

        result = cmd_report(args)
        assert result == 0

        captured = capsys.readouterr()
        assert "FILTERED CANDIDATES" in captured.out

    def test_report_no_candidates(self, db_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Report with no matching candidates."""
        from argparse import Namespace

        args = Namespace(
            run_id=999,
            db=str(db_path),
            any_filter=False,
        )

        result = cmd_report(args)
        assert result == 0

        captured = capsys.readouterr()
        assert "No candidates found" in captured.out


class TestMain:
    """Tests for main entry point."""

    def test_no_command(self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        """No command prints help."""
        monkeypatch.setattr(sys, "argv", ["prog"])
        result = main()
        assert result == 1

    def test_run_command(
        self, db_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Run command works."""
        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "run", "--run-id", "1", "--db", str(db_path), "--filters", "dsr"],
        )
        result = main()
        assert result == 0

    def test_report_command(
        self, db_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Report command works."""
        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "report", "--run-id", "1", "--db", str(db_path)],
        )
        result = main()
        assert result == 0


class TestModuleExecution:
    """Test module can be executed directly."""

    def test_module_help(self) -> None:
        """Module shows help with --help."""
        result = subprocess.run(
            [sys.executable, "-m", "vibe_quant.overfitting", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Overfitting prevention pipeline CLI" in result.stdout

    def test_run_help(self) -> None:
        """Run subcommand shows help."""
        result = subprocess.run(
            [sys.executable, "-m", "vibe_quant.overfitting", "run", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--run-id" in result.stdout
        assert "--filters" in result.stdout

    def test_report_help(self) -> None:
        """Report subcommand shows help."""
        result = subprocess.run(
            [sys.executable, "-m", "vibe_quant.overfitting", "report", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--run-id" in result.stdout
        assert "--any-filter" in result.stdout
