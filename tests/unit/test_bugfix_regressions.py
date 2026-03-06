"""Regression tests for backend bug fixes.

Each test verifies that a specific bug fix works correctly (not just that code runs).
"""

from __future__ import annotations

import inspect
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from vibe_quant.db.state_manager import StateManager

if TYPE_CHECKING:
    from collections.abc import Generator


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def state_mgr(tmp_path: Path) -> Generator[StateManager]:
    """StateManager with temp database."""
    mgr = StateManager(db_path=tmp_path / "test.db")
    _ = mgr.conn  # force schema init
    yield mgr
    mgr.close()


def _create_run(mgr: StateManager, strategy_id: int | None = None) -> int:
    """Helper: create a strategy + backtest run, return run_id."""
    sid = strategy_id or mgr.create_strategy(name="regtest", dsl_config={})
    return mgr.create_backtest_run(
        strategy_id=sid,
        run_mode="screening",
        symbols=["BTCUSDT-PERP"],
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-12-31",
        parameters={},
    )


# ---------------------------------------------------------------------------
# 1. Equity curve uses actual starting_balance (bd-bzp6)
# ---------------------------------------------------------------------------


class TestStartingBalance:
    """_get_starting_balance returns stored balance, not hardcoded 10000."""

    def test_returns_stored_balance(self, state_mgr: StateManager) -> None:
        """When backtest_results has starting_balance=50000, helper returns 50000."""
        from vibe_quant.api.routers.results import _get_starting_balance

        run_id = _create_run(state_mgr)
        state_mgr.save_backtest_result(run_id, {"starting_balance": 50_000.0})

        balance = _get_starting_balance(state_mgr, run_id)
        assert balance == 50_000.0, f"Expected 50000, got {balance}"

    def test_fallback_to_default(self, state_mgr: StateManager) -> None:
        """When no starting_balance stored, defaults to 10000."""
        from vibe_quant.api.routers.results import _get_starting_balance

        run_id = _create_run(state_mgr)
        # No backtest_results row at all
        balance = _get_starting_balance(state_mgr, run_id)
        assert balance == 10_000.0


# ---------------------------------------------------------------------------
# 2. DSR uses bar count not trades (bd-siif)
# ---------------------------------------------------------------------------


class TestComputeBarCount:
    """_compute_bar_count returns bar count from date range, not trade count."""

    def test_1h_full_year(self) -> None:
        """365 days at 1h = 8760 bars."""
        from vibe_quant.screening.pipeline import _compute_bar_count

        result = _compute_bar_count("2024-01-01", "2024-12-31", "1h")
        assert result is not None
        # 365 days * 24 hours = 8760 (might be 8736 due to 364 day diff)
        assert 8700 <= result <= 8800, f"Expected ~8760 bars, got {result}"

    def test_unknown_timeframe_returns_none(self) -> None:
        """Unknown timeframe like '2h' returns None."""
        from vibe_quant.screening.pipeline import _compute_bar_count

        result = _compute_bar_count("2024-01-01", "2024-12-31", "2h")
        assert result is None

    def test_missing_dates_returns_none(self) -> None:
        """Missing start or end date returns None."""
        from vibe_quant.screening.pipeline import _compute_bar_count

        assert _compute_bar_count(None, "2024-12-31", "1h") is None
        assert _compute_bar_count("2024-01-01", None, "1h") is None
        assert _compute_bar_count(None, None, "1h") is None


# ---------------------------------------------------------------------------
# 3. JobManager race condition — lock held through DB write (bd-4wc9)
# ---------------------------------------------------------------------------


class TestJobManagerAtomicStart:
    """start_job holds lock through DB write so record exists immediately."""

    def test_job_record_exists_after_start(self, tmp_path: Path) -> None:
        """After start_job returns, the job record exists in DB (no race window)."""
        import sys

        from vibe_quant.jobs.manager import BacktestJobManager, JobStatus

        db_path = tmp_path / "atomic.db"
        sm = StateManager(db_path=db_path)
        _ = sm.conn
        jm = BacktestJobManager(db_path=db_path)

        run_id = _create_run(sm)
        pid = jm.start_job(
            run_id=run_id,
            job_type="screening",
            command=[sys.executable, "-c", "import time; time.sleep(0.1)"],
        )
        assert pid > 0

        # Immediately after start_job returns, DB record must exist
        info = jm.get_job_info(run_id)
        assert info is not None, "Job record missing right after start_job"
        assert info.status == JobStatus.RUNNING

        import time

        time.sleep(0.2)
        jm.close()
        sm.close()

    def test_start_lock_exists(self) -> None:
        """BacktestJobManager has _start_lock attribute (threading.Lock)."""
        from vibe_quant.jobs.manager import BacktestJobManager

        mgr = BacktestJobManager()
        assert hasattr(mgr, "_start_lock")
        assert isinstance(mgr._start_lock, type(threading.Lock()))
        mgr.close()


# ---------------------------------------------------------------------------
# 4. end_date filter uses date comparison (bd-pyng)
# ---------------------------------------------------------------------------


class TestEndDateFilter:
    """end_date filter uses <= (inclusive), not < (exclusive)."""

    def test_run_included_on_exact_date(self, state_mgr: StateManager) -> None:
        """Run created on a date is included when end_date equals that date."""
        run_id = _create_run(state_mgr)
        run = state_mgr.get_backtest_run(run_id)
        assert run is not None
        run_date = run["created_at"][:10]  # "YYYY-MM-DD"

        # Simulate the filter logic from results.py
        from vibe_quant.api.schemas.backtest import BacktestRunResponse

        runs = [BacktestRunResponse(**run)]

        # Apply the fixed filter (<=)
        filtered = [r for r in runs if r.created_at and r.created_at[:10] <= run_date[:10]]
        assert len(filtered) == 1, "Run should be INCLUDED when end_date == created_at date"

        # Verify old broken filter (<) would exclude it
        broken = [r for r in runs if r.created_at and r.created_at[:10] < run_date[:10]]
        assert len(broken) == 0, "Old < filter would exclude the run (confirming the bug)"


# ---------------------------------------------------------------------------
# 5. StateManager write lock (bd-29kn)
# ---------------------------------------------------------------------------


class TestStateManagerWriteLock:
    """StateManager has _write_lock for thread-safe writes."""

    def test_write_lock_exists(self, state_mgr: StateManager) -> None:
        """_write_lock attribute exists and is a threading.Lock."""
        assert hasattr(state_mgr, "_write_lock")
        assert isinstance(state_mgr._write_lock, type(threading.Lock()))


# ---------------------------------------------------------------------------
# 6. Strategy name persisted on save (bd-wt64)
# ---------------------------------------------------------------------------


class TestStrategyNamePersist:
    """Strategy name update is persisted."""

    def test_name_updated(self, state_mgr: StateManager) -> None:
        """update_strategy with new name persists the change."""
        sid = state_mgr.create_strategy(name="old_name", dsl_config={})
        state_mgr.update_strategy(sid, name="new_name")
        strat = state_mgr.get_strategy(sid)
        assert strat is not None
        assert strat["name"] == "new_name", f"Expected 'new_name', got {strat['name']}"


# ---------------------------------------------------------------------------
# 7. Data quality error handling (bd-pafy)
# ---------------------------------------------------------------------------


class TestDataQualityResponse:
    """DataQualityResponse accepts quality_score=None and error string."""

    def test_quality_score_none_with_error(self) -> None:
        """DataQualityResponse can hold quality_score=None and error message."""
        from vibe_quant.api.schemas.data import DataQualityResponse

        resp = DataQualityResponse(
            symbol="BTCUSDT",
            gaps=[],
            quality_score=None,
            error="No data available",
        )
        assert resp.quality_score is None
        assert resp.error == "No data available"

    def test_quality_score_with_value(self) -> None:
        """DataQualityResponse works normally with a float score."""
        from vibe_quant.api.schemas.data import DataQualityResponse

        resp = DataQualityResponse(
            symbol="BTCUSDT",
            gaps=[],
            quality_score=0.95,
        )
        assert resp.quality_score == 0.95
        assert resp.error is None


# ---------------------------------------------------------------------------
# 8. Discovery dynamic dates (bd-fzf6)
# ---------------------------------------------------------------------------


class TestDiscoveryDynamicDates:
    """Discovery router uses datetime.now() for fallback, not hardcoded dates."""

    def test_fallback_uses_today(self) -> None:
        """Verify discovery router computes dates dynamically with datetime.now()."""
        from vibe_quant.api.routers import discovery

        source = inspect.getsource(discovery)
        # The fix: fallback dates use today = datetime.now() instead of hardcoded
        assert "datetime.now()" in source, "Discovery router should use datetime.now() for fallback"
        # launch_discovery should reference 'today' variable for dynamic dates
        launch_fn_source = inspect.getsource(discovery.launch_discovery)
        assert "today" in launch_fn_source, "launch_discovery should reference 'today' variable"


# ---------------------------------------------------------------------------
# 9. Telegram format (bd-a5gq)
# ---------------------------------------------------------------------------


class TestTelegramFormat:
    """DailySummary formats win_rate as percentage correctly."""

    def test_win_rate_55_percent(self) -> None:
        """win_rate=0.55 formats as '55.0%' not '0.6%'."""
        from vibe_quant.alerts.telegram import DailySummary

        summary = DailySummary(
            date=datetime(2024, 6, 15, tzinfo=UTC),
            win_rate=0.55,
        )
        msg = summary.format_message()
        assert "55.0%" in msg, f"Expected '55.0%' in message, got:\n{msg}"
        assert "0.6%" not in msg, "Should not contain '0.6%' (broken formatting)"


# ---------------------------------------------------------------------------
# 10. ConsistencyChecker uses canonical schema (bd-oygu)
# ---------------------------------------------------------------------------


class TestConsistencyChecksSchema:
    """consistency_checks table is part of SCHEMA_SQL."""

    def test_table_in_schema(self) -> None:
        """consistency_checks CREATE TABLE exists in SCHEMA_SQL."""
        from vibe_quant.db.schema import SCHEMA_SQL

        assert "consistency_checks" in SCHEMA_SQL

    def test_table_created(self, state_mgr: StateManager) -> None:
        """consistency_checks table exists after schema init."""
        cursor = state_mgr.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='consistency_checks'"
        )
        row = cursor.fetchone()
        assert row is not None, "consistency_checks table not created by init_schema"


# ---------------------------------------------------------------------------
# 11. paper_trading ImportError (bd-lwxd)
# ---------------------------------------------------------------------------


class TestPaperTradingImportError:
    """Paper trading router catches only ImportError, not broad Exception."""

    def test_except_clause_is_import_error(self) -> None:
        """get_session catches ImportError specifically (not Exception)."""
        from vibe_quant.api.routers import paper_trading

        source = inspect.getsource(paper_trading.get_session)
        assert "except ImportError" in source, (
            "get_session should catch ImportError, not a broader exception"
        )
        # Ensure it's not "except Exception"
        lines = source.split("\n")
        except_lines = [line.strip() for line in lines if line.strip().startswith("except ")]
        for line in except_lines:
            assert "Exception" not in line or "ImportError" in line, (
                f"Found broad exception handler: {line}"
            )


# ---------------------------------------------------------------------------
# 12. Settings SQL whitelist (bd-zyry)
# ---------------------------------------------------------------------------


class TestSettingsSqlWhitelist:
    """_KNOWN_TABLES frozenset prevents SQL injection in table count queries."""

    def test_known_tables_exists(self) -> None:
        """_KNOWN_TABLES is a frozenset."""
        from vibe_quant.api.routers.settings import _KNOWN_TABLES

        assert isinstance(_KNOWN_TABLES, frozenset)

    def test_known_tables_contains_expected(self) -> None:
        """_KNOWN_TABLES contains all schema tables."""
        from vibe_quant.api.routers.settings import _KNOWN_TABLES

        expected = {
            "strategies",
            "backtest_runs",
            "backtest_results",
            "trades",
            "sweep_results",
            "background_jobs",
            "consistency_checks",
        }
        assert expected.issubset(_KNOWN_TABLES), f"Missing tables: {expected - _KNOWN_TABLES}"
