"""Tests for DSR calculation fixes.

Covers beads:
- vibe-quant-edhju: pipeline loads stored skewness/kurtosis into DSR
- vibe-quant-zmzzh: annualized Sharpe de-annualized to match observation units
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import pytest

from vibe_quant.overfitting.dsr import (
    TRADING_DAYS_PER_YEAR,
    calculate_dsr,
    deannualize_sharpe,
)
from vibe_quant.overfitting.pipeline import OverfittingPipeline
from vibe_quant.overfitting.types import FilterConfig
from vibe_quant.utils import compute_day_count


class TestDeannualizeSharpe:
    def test_scaling(self) -> None:
        assert deannualize_sharpe(2.0) == pytest.approx(2.0 / math.sqrt(252))

    def test_custom_periods(self) -> None:
        assert deannualize_sharpe(3.0, periods_per_year=365) == pytest.approx(
            3.0 / math.sqrt(365)
        )

    def test_constant(self) -> None:
        assert TRADING_DAYS_PER_YEAR == 252.0


class TestComputeDayCount:
    def test_one_year(self) -> None:
        assert compute_day_count("2025-01-01", "2026-01-01") == 365

    def test_missing_dates(self) -> None:
        assert compute_day_count(None, "2026-01-01") is None
        assert compute_day_count("2025-01-01", None) is None

    def test_invalid_dates(self) -> None:
        assert compute_day_count("not-a-date", "2026-01-01") is None

    def test_minimum_one(self) -> None:
        assert compute_day_count("2025-01-01", "2025-01-01") == 1


class TestDsrUsesStoredMoments:
    """The offline pipeline must feed stored return moments into DSR."""

    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        path = tmp_path / "test.db"
        conn = sqlite3.connect(str(path))
        conn.executescript("""
            CREATE TABLE strategies (
                id INTEGER PRIMARY KEY, name TEXT, dsl_config TEXT
            );
            CREATE TABLE backtest_runs (
                id INTEGER PRIMARY KEY, strategy_id INTEGER, run_mode TEXT,
                symbols TEXT, timeframe TEXT, start_date TEXT, end_date TEXT,
                parameters TEXT
            );
            CREATE TABLE sweep_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER, parameters TEXT NOT NULL,
                sharpe_ratio REAL, sortino_ratio REAL, max_drawdown REAL,
                total_return REAL, profit_factor REAL, win_rate REAL,
                total_trades INTEGER, total_fees REAL, total_funding REAL,
                execution_time_seconds REAL, skewness REAL, kurtosis REAL,
                is_pareto_optimal BOOLEAN DEFAULT 0,
                passed_deflated_sharpe BOOLEAN, passed_walk_forward BOOLEAN,
                passed_purged_kfold BOOLEAN
            );
        """)
        conn.execute(
            "INSERT INTO strategies (id, name, dsl_config) VALUES (1, 's', '{}')"
        )
        conn.execute(
            "INSERT INTO backtest_runs VALUES (1, 1, 'screening', '[]', '4h',"
            " '2025-01-01', '2026-01-01', '{}')"
        )
        conn.commit()
        conn.close()
        return path

    def _run_dsr(
        self, db_path: Path, skewness: float | None, kurtosis: float | None
    ) -> float:
        conn = sqlite3.connect(str(db_path))
        conn.execute("DELETE FROM sweep_results")
        conn.execute(
            "INSERT INTO sweep_results (run_id, parameters, sharpe_ratio,"
            " total_return, skewness, kurtosis) VALUES (1, '{}', 3.0, 0.5, ?, ?)",
            (skewness, kurtosis),
        )
        conn.commit()
        conn.close()

        pipeline = OverfittingPipeline(db_path)
        try:
            result = pipeline.run(
                run_id=1,
                config=FilterConfig(
                    enable_dsr=True, enable_wfa=False, enable_purged_kfold=False
                ),
                num_observations=365,
                total_trials=100,
            )
        finally:
            pipeline.close()
        assert result.candidates
        dsr_result = result.candidates[0].dsr_result
        assert dsr_result is not None
        return dsr_result.p_value

    def test_fat_tails_change_dsr(self, db_path: Path) -> None:
        """Stored non-normal moments must produce a different p-value."""
        p_normal = self._run_dsr(db_path, skewness=0.0, kurtosis=3.0)
        p_fat = self._run_dsr(db_path, skewness=-1.5, kurtosis=9.0)
        assert p_fat != pytest.approx(p_normal)
        # Negative skew + fat tails widen the estimator variance -> less
        # significant (higher p) for a positive Sharpe.
        assert p_fat > p_normal

    def test_null_moments_coalesce(self, db_path: Path) -> None:
        """NULL skewness/kurtosis must not crash and match normal defaults."""
        p_null = self._run_dsr(db_path, skewness=None, kurtosis=None)
        p_normal = self._run_dsr(db_path, skewness=0.0, kurtosis=3.0)
        assert p_null == pytest.approx(p_normal)

    def test_deannualized_units(self, db_path: Path) -> None:
        """Pipeline p-value must match a direct daily-units DSR calculation."""
        p_pipeline = self._run_dsr(db_path, skewness=0.0, kurtosis=3.0)
        direct = calculate_dsr(
            observed_sharpe=deannualize_sharpe(3.0),
            num_trials=100,
            num_observations=365,
        )
        assert p_pipeline == pytest.approx(direct.p_value)
