"""Tests for overfitting prevention pipeline orchestrator."""

from __future__ import annotations

import sqlite3
from datetime import date
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from vibe_quant.db.connection import DEFAULT_DB_PATH
from vibe_quant.overfitting.pipeline import (
    CandidateResult,
    FilterConfig,
    MockBacktestRunner,
    OverfittingPipeline,
    PipelineResult,
    run_overfitting_pipeline,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Create temp database with test data."""
    db_file = tmp_path / "test_overfitting.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("PRAGMA journal_mode=WAL")

    # Create backtest_runs table
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

    # Create sweep_results table
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

    # Insert test backtest run
    conn.execute(
        """INSERT INTO backtest_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (1, 1, "screening", '["BTCUSDT"]', "1h", "2024-01-01", "2024-12-31", "{}"),
    )

    # Insert test sweep results
    # High Sharpe - should pass DSR
    conn.execute(
        """INSERT INTO sweep_results
           (run_id, parameters, sharpe_ratio, total_return, profit_factor, max_drawdown)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (1, '{"rsi_period": 14}', 2.5, 50.0, 2.0, 0.15),
    )
    # Medium Sharpe - might pass
    conn.execute(
        """INSERT INTO sweep_results
           (run_id, parameters, sharpe_ratio, total_return, profit_factor, max_drawdown)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (1, '{"rsi_period": 21}', 1.5, 30.0, 1.5, 0.20),
    )
    # Low Sharpe - unlikely to pass
    conn.execute(
        """INSERT INTO sweep_results
           (run_id, parameters, sharpe_ratio, total_return, profit_factor, max_drawdown)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (1, '{"rsi_period": 7}', 0.5, 10.0, 1.1, 0.30),
    )

    conn.commit()
    conn.close()
    return db_file


class TestFilterConfig:
    """Tests for FilterConfig dataclass."""

    def test_default_config(self) -> None:
        """Default config enables all filters."""
        config = FilterConfig.default()
        assert config.enable_dsr is True
        assert config.enable_wfa is True
        assert config.enable_purged_kfold is True

    def test_dsr_only_config(self) -> None:
        """DSR-only config disables WFA and CV."""
        config = FilterConfig.dsr_only()
        assert config.enable_dsr is True
        assert config.enable_wfa is False
        assert config.enable_purged_kfold is False

    def test_wfa_only_config(self) -> None:
        """WFA-only config disables DSR and CV."""
        config = FilterConfig.wfa_only()
        assert config.enable_dsr is False
        assert config.enable_wfa is True
        assert config.enable_purged_kfold is False

    def test_cv_only_config(self) -> None:
        """CV-only config disables DSR and WFA."""
        config = FilterConfig.cv_only()
        assert config.enable_dsr is False
        assert config.enable_wfa is False
        assert config.enable_purged_kfold is True

    def test_custom_config(self) -> None:
        """Can create custom config."""
        config = FilterConfig(
            enable_dsr=True,
            enable_wfa=False,
            enable_purged_kfold=True,
            dsr_significance=0.01,
            dsr_confidence_threshold=0.99,
        )
        assert config.dsr_significance == 0.01
        assert config.dsr_confidence_threshold == 0.99


class TestMockBacktestRunner:
    """Tests for MockBacktestRunner."""

    def test_optimize(self) -> None:
        """Mock optimize returns params and metrics."""
        runner = MockBacktestRunner(oos_sharpe=1.0, oos_return=10.0)
        params, sharpe, ret = runner.optimize(
            "test",
            date(2024, 1, 1),
            date(2024, 6, 30),
            {"period": [14, 21]},
        )
        assert params == {"period": 14}
        assert sharpe == 1.2  # oos_sharpe * 1.2
        assert ret == 11.0  # oos_return * 1.1

    def test_backtest(self) -> None:
        """Mock backtest returns OOS metrics."""
        runner = MockBacktestRunner(oos_sharpe=1.5, oos_return=20.0)
        sharpe, ret = runner.backtest(
            "test",
            date(2024, 7, 1),
            date(2024, 12, 31),
            {"period": 14},
        )
        assert sharpe == 1.5
        assert ret == 20.0

    def test_run_fold(self) -> None:
        """Mock run returns fold result."""
        runner = MockBacktestRunner(oos_sharpe=1.0, oos_return=10.0)
        result = runner.run(list(range(100)), list(range(100, 150)))
        assert result.train_size == 100
        assert result.test_size == 50
        assert result.test_sharpe == 1.0


class TestOverfittingPipeline:
    """Tests for OverfittingPipeline class."""

    def test_init_default_path(self) -> None:
        """Pipeline uses canonical DEFAULT_DB_PATH."""
        pipeline = OverfittingPipeline()
        assert pipeline.db_path == DEFAULT_DB_PATH
        pipeline.close()

    def test_init_custom_path(self, db_path: Path) -> None:
        """Pipeline uses custom db path."""
        pipeline = OverfittingPipeline(db_path)
        assert pipeline.db_path == db_path
        pipeline.close()

    def test_conn_enables_wal(self, db_path: Path) -> None:
        """Connection has WAL mode enabled."""
        pipeline = OverfittingPipeline(db_path)
        _ = pipeline.conn
        result = pipeline.conn.execute("PRAGMA journal_mode").fetchone()
        assert result[0] == "wal"
        pipeline.close()

    def test_load_candidates(self, db_path: Path) -> None:
        """Can load candidates from database."""
        pipeline = OverfittingPipeline(db_path)
        candidates = pipeline._load_candidates(1)
        assert len(candidates) == 3
        assert candidates[0]["sharpe_ratio"] == 2.5  # Highest first
        pipeline.close()

    def test_run_no_candidates(self, db_path: Path) -> None:
        """Run with no candidates returns empty result."""
        pipeline = OverfittingPipeline(db_path)
        result = pipeline.run(run_id=999)  # Non-existent run
        assert result.total_candidates == 0
        assert result.passed_all == 0
        pipeline.close()

    def test_run_dsr_only(self, db_path: Path) -> None:
        """Run with DSR filter only."""
        pipeline = OverfittingPipeline(db_path)
        result = pipeline.run(
            run_id=1,
            config=FilterConfig.dsr_only(),
            num_observations=252,
        )
        assert result.total_candidates == 3
        # DSR results depend on num_trials and observed sharpe
        assert result.passed_dsr >= 0
        assert result.passed_wfa == 0  # Disabled
        assert result.passed_cv == 0  # Disabled
        pipeline.close()

    def test_run_wfa_only(self, db_path: Path) -> None:
        """Run with WFA filter only."""
        pipeline = OverfittingPipeline(db_path)
        result = pipeline.run(
            run_id=1,
            config=FilterConfig.wfa_only(),
            data_start=date(2024, 1, 1),
            data_end=date(2025, 12, 31),
        )
        assert result.total_candidates == 3
        assert result.passed_dsr == 0  # Disabled
        # WFA with mock runner should pass
        assert result.passed_wfa >= 0
        assert result.passed_cv == 0  # Disabled
        pipeline.close()

    def test_run_cv_only(self, db_path: Path) -> None:
        """Run with Purged K-Fold filter only."""
        pipeline = OverfittingPipeline(db_path)
        result = pipeline.run(
            run_id=1,
            config=FilterConfig.cv_only(),
            n_samples=1000,
        )
        assert result.total_candidates == 3
        assert result.passed_dsr == 0  # Disabled
        assert result.passed_wfa == 0  # Disabled
        assert result.passed_cv >= 0
        pipeline.close()

    def test_run_all_filters(self, db_path: Path) -> None:
        """Run with all filters enabled."""
        pipeline = OverfittingPipeline(db_path)
        result = pipeline.run(
            run_id=1,
            config=FilterConfig.default(),
            num_observations=252,
            data_start=date(2024, 1, 1),
            data_end=date(2025, 12, 31),
            n_samples=1000,
        )
        assert result.total_candidates == 3
        assert result.config.enable_dsr is True
        assert result.config.enable_wfa is True
        assert result.config.enable_purged_kfold is True
        pipeline.close()

    def test_update_candidate(self, db_path: Path) -> None:
        """Updates candidate with filter results."""
        pipeline = OverfittingPipeline(db_path)

        # Get first candidate ID
        candidates = pipeline._load_candidates(1)
        candidate_id = candidates[0]["id"]

        # Update with filter results
        pipeline._update_candidate(candidate_id, True, False, None)

        # Verify update
        row = pipeline.conn.execute(
            "SELECT passed_deflated_sharpe, passed_walk_forward, passed_purged_kfold "
            "FROM sweep_results WHERE id = ?",
            (candidate_id,),
        ).fetchone()

        assert row[0] == 1  # passed_dsr = True
        assert row[1] == 0  # passed_wfa = False
        assert row[2] is None  # passed_cv = None (not run)

        pipeline.close()

    def test_filtered_candidates_property(self, db_path: Path) -> None:
        """PipelineResult.filtered_candidates returns passing candidates."""
        pipeline = OverfittingPipeline(db_path)
        result = pipeline.run(run_id=1, config=FilterConfig.dsr_only())

        # filtered_candidates should match passed_all count
        assert len(result.filtered_candidates) == result.passed_all
        for c in result.filtered_candidates:
            assert c.passed_all is True

        pipeline.close()

    def test_get_filtered_candidates_require_all(self, db_path: Path) -> None:
        """get_filtered_candidates with require_all=True."""
        pipeline = OverfittingPipeline(db_path)

        # Run pipeline to set flags
        pipeline.run(run_id=1, config=FilterConfig.dsr_only())

        # Query filtered
        filtered = pipeline.get_filtered_candidates(1, require_all=True)

        # Should only include those that passed DSR (or have NULL which means not evaluated)
        for f in filtered:
            assert f["passed_deflated_sharpe"] in (1, None)

        pipeline.close()


class TestCandidateResult:
    """Tests for CandidateResult dataclass."""

    def test_create_result(self) -> None:
        """Can create candidate result."""
        result = CandidateResult(
            sweep_result_id=1,
            run_id=1,
            strategy_name="test",
            parameters='{"period": 14}',
            sharpe_ratio=2.0,
            total_return=50.0,
            passed_dsr=True,
            passed_wfa=False,
            passed_cv=None,
            passed_all=False,
        )
        assert result.sweep_result_id == 1
        assert result.passed_dsr is True
        assert result.passed_wfa is False
        assert result.passed_cv is None
        assert result.passed_all is False


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_filtered_candidates_empty(self) -> None:
        """Empty result has no filtered candidates."""
        result = PipelineResult(
            config=FilterConfig.default(),
            total_candidates=0,
            passed_dsr=0,
            passed_wfa=0,
            passed_cv=0,
            passed_all=0,
        )
        assert result.filtered_candidates == []

    def test_filtered_candidates_filters(self) -> None:
        """filtered_candidates only returns passing candidates."""
        candidates = [
            CandidateResult(
                sweep_result_id=1, run_id=1, strategy_name="pass",
                parameters="{}", sharpe_ratio=2.0, total_return=50.0,
                passed_dsr=True, passed_wfa=True, passed_cv=True, passed_all=True,
            ),
            CandidateResult(
                sweep_result_id=2, run_id=1, strategy_name="fail",
                parameters="{}", sharpe_ratio=1.0, total_return=20.0,
                passed_dsr=True, passed_wfa=False, passed_cv=True, passed_all=False,
            ),
        ]
        result = PipelineResult(
            config=FilterConfig.default(),
            total_candidates=2,
            passed_dsr=2,
            passed_wfa=1,
            passed_cv=2,
            passed_all=1,
            candidates=candidates,
        )
        assert len(result.filtered_candidates) == 1
        assert result.filtered_candidates[0].strategy_name == "pass"


class TestGenerateReport:
    """Tests for report generation."""

    def test_report_with_results(self, db_path: Path) -> None:
        """Report includes filter statistics."""
        pipeline = OverfittingPipeline(db_path)
        result = pipeline.run(run_id=1, config=FilterConfig.dsr_only())
        report = pipeline.generate_report(result)

        assert "OVERFITTING PREVENTION PIPELINE REPORT" in report
        assert "Total candidates: 3" in report
        assert "DSR (Deflated Sharpe):" in report
        assert "WFA (Walk-Forward):" in report
        assert "DISABLED" in report or "PASSED ALL FILTERS" in report

        pipeline.close()

    def test_report_empty_result(self, db_path: Path) -> None:
        """Report handles empty result."""
        pipeline = OverfittingPipeline(db_path)
        result = pipeline.run(run_id=999)  # No candidates
        report = pipeline.generate_report(result)

        assert "Total candidates: 0" in report
        pipeline.close()


class TestConvenienceFunction:
    """Tests for run_overfitting_pipeline convenience function."""

    def test_convenience_function(self, db_path: Path) -> None:
        """Convenience function works."""
        result = run_overfitting_pipeline(
            run_id=1,
            db_path=db_path,
            config=FilterConfig.dsr_only(),
        )
        assert result.total_candidates == 3
        assert isinstance(result, PipelineResult)


class TestIntegration:
    """Integration tests for full pipeline flow."""

    def test_full_pipeline_flow(self, db_path: Path) -> None:
        """Full pipeline flow from candidates to filtered output."""
        # Create pipeline
        pipeline = OverfittingPipeline(db_path)

        # Run with all filters
        result = pipeline.run(
            run_id=1,
            config=FilterConfig.default(),
            num_observations=252,
            data_start=date(2024, 1, 1),
            data_end=date(2025, 12, 31),
            n_samples=1000,
        )

        # Verify structure
        assert result.total_candidates == 3
        assert len(result.candidates) == 3

        # Each candidate should have filter results
        for c in result.candidates:
            assert c.passed_dsr is not None
            assert c.passed_wfa is not None
            assert c.passed_cv is not None

        # Database should be updated
        for c in result.candidates:
            row = pipeline.conn.execute(
                "SELECT passed_deflated_sharpe, passed_walk_forward, passed_purged_kfold "
                "FROM sweep_results WHERE id = ?",
                (c.sweep_result_id,),
            ).fetchone()
            assert row[0] is not None
            assert row[1] is not None
            assert row[2] is not None

        # Report should be generated
        report = pipeline.generate_report(result)
        assert len(report) > 0

        pipeline.close()

    def test_sequential_filter_runs(self, db_path: Path) -> None:
        """Can run filters sequentially with different configs."""
        pipeline = OverfittingPipeline(db_path)

        # First run DSR only
        result1 = pipeline.run(run_id=1, config=FilterConfig.dsr_only())

        # Then run WFA only
        result2 = pipeline.run(run_id=1, config=FilterConfig.wfa_only())

        # Both should have same total candidates
        assert result1.total_candidates == result2.total_candidates

        # But different filter counts
        assert result1.passed_wfa == 0  # Disabled
        assert result2.passed_dsr == 0  # Disabled

        pipeline.close()
