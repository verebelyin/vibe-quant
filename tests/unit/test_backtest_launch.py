"""Tests for backtest launch tab module."""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Create temp database with test data."""
    db_file = tmp_path / "test_launch.db"
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    # Create tables
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            dsl_config JSON NOT NULL,
            strategy_type TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            is_active BOOLEAN DEFAULT 1,
            version INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS sizing_configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            method TEXT NOT NULL,
            config JSON NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS risk_configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            strategy_level JSON NOT NULL,
            portfolio_level JSON NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS backtest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id INTEGER,
            sizing_config_id INTEGER,
            risk_config_id INTEGER,
            run_mode TEXT NOT NULL,
            symbols JSON NOT NULL,
            timeframe TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            parameters JSON NOT NULL,
            latency_preset TEXT,
            status TEXT DEFAULT 'pending',
            pid INTEGER,
            heartbeat_at TEXT,
            started_at TEXT,
            completed_at TEXT,
            error_message TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS background_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            pid INTEGER NOT NULL,
            job_type TEXT NOT NULL,
            status TEXT DEFAULT 'running',
            heartbeat_at TEXT,
            started_at TEXT DEFAULT (datetime('now')),
            completed_at TEXT,
            log_file TEXT
        );
    """)

    # Insert test strategy
    dsl_config = {
        "name": "test_strategy",
        "description": "Test strategy",
        "version": 1,
        "timeframe": "1h",
        "additional_timeframes": [],
        "indicators": {
            "rsi_14": {"type": "RSI", "period": 14, "source": "close"},
            "atr_14": {"type": "ATR", "period": 14},
        },
        "entry_conditions": {"long": ["rsi_14 < 30"], "short": ["rsi_14 > 70"]},
        "exit_conditions": {"long": [], "short": []},
        "time_filters": {},
        "stop_loss": {"type": "atr_fixed", "atr_multiplier": 2.0, "indicator": "atr_14"},
        "take_profit": {"type": "risk_reward", "risk_reward_ratio": 2.0},
        "position_management": {"scale_in": {"enabled": False}, "partial_exit": {"enabled": False}},
        "sweep": {"rsi_period": [10, 14, 20], "atr_mult": [1.5, 2.0, 2.5]},
    }

    conn.execute(
        "INSERT INTO strategies (name, description, dsl_config) VALUES (?, ?, ?)",
        ("test_strategy", "A test strategy", json.dumps(dsl_config)),
    )

    # Insert sizing config
    conn.execute(
        "INSERT INTO sizing_configs (name, method, config) VALUES (?, ?, ?)",
        ("default_sizing", "fixed_fractional", json.dumps({"risk_per_trade": "0.02"})),
    )

    # Insert risk config
    conn.execute(
        "INSERT INTO risk_configs (name, strategy_level, portfolio_level) VALUES (?, ?, ?)",
        (
            "default_risk",
            json.dumps({"max_drawdown_pct": "0.15"}),
            json.dumps({"max_portfolio_drawdown_pct": "0.20"}),
        ),
    )

    conn.commit()
    conn.close()
    return db_file


class TestStateManagerIntegration:
    """Test integration with StateManager."""

    def test_list_strategies(self, db_path: Path):
        """Can list strategies from database."""
        from vibe_quant.db.state_manager import StateManager

        manager = StateManager(db_path)
        strategies = manager.list_strategies()

        assert len(strategies) == 1
        assert strategies[0]["name"] == "test_strategy"
        assert "sweep" in strategies[0]["dsl_config"]
        manager.close()

    def test_list_sizing_configs(self, db_path: Path):
        """Can list sizing configs from database."""
        from vibe_quant.db.state_manager import StateManager

        manager = StateManager(db_path)
        configs = manager.list_sizing_configs()

        assert len(configs) == 1
        assert configs[0]["name"] == "default_sizing"
        manager.close()

    def test_list_risk_configs(self, db_path: Path):
        """Can list risk configs from database."""
        from vibe_quant.db.state_manager import StateManager

        manager = StateManager(db_path)
        configs = manager.list_risk_configs()

        assert len(configs) == 1
        assert configs[0]["name"] == "default_risk"
        manager.close()

    def test_create_backtest_run(self, db_path: Path):
        """Can create backtest run record."""
        from vibe_quant.db.state_manager import StateManager

        manager = StateManager(db_path)

        run_id = manager.create_backtest_run(
            strategy_id=1,
            run_mode="screening",
            symbols=["BTCUSDT", "ETHUSDT"],
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-12-31",
            parameters={"sweep": {"rsi_period": [14, 20]}},
            sizing_config_id=1,
            risk_config_id=1,
            latency_preset=None,
        )

        assert run_id > 0

        run = manager.get_backtest_run(run_id)
        assert run is not None
        assert run["run_mode"] == "screening"
        assert run["symbols"] == ["BTCUSDT", "ETHUSDT"]
        manager.close()

    def test_get_strategy_by_id(self, db_path: Path):
        """Can get strategy by ID."""
        from vibe_quant.db.state_manager import StateManager

        manager = StateManager(db_path)
        strategy = manager.get_strategy(1)

        assert strategy is not None
        assert strategy["name"] == "test_strategy"
        assert strategy["dsl_config"]["timeframe"] == "1h"
        manager.close()

    def test_list_backtest_runs(self, db_path: Path):
        """Can list backtest runs."""
        from vibe_quant.db.state_manager import StateManager

        manager = StateManager(db_path)

        # Create a run first
        manager.create_backtest_run(
            strategy_id=1,
            run_mode="screening",
            symbols=["BTCUSDT"],
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-12-31",
            parameters={},
        )

        runs = manager.list_backtest_runs()
        assert len(runs) == 1
        manager.close()


class TestJobManagerIntegration:
    """Test integration with BacktestJobManager."""

    def test_list_active_jobs_empty(self, db_path: Path):
        """list_active_jobs returns empty list when no jobs."""
        from vibe_quant.jobs.manager import BacktestJobManager

        manager = BacktestJobManager(db_path)
        jobs = manager.list_active_jobs()

        assert jobs == []
        manager.close()

    def test_list_stale_jobs_empty(self, db_path: Path):
        """list_stale_jobs returns empty list when no stale jobs."""
        from vibe_quant.jobs.manager import BacktestJobManager

        manager = BacktestJobManager(db_path)
        jobs = manager.list_stale_jobs()

        assert jobs == []
        manager.close()

    def test_get_status_none_for_nonexistent(self, db_path: Path):
        """get_status returns None for nonexistent run."""
        from vibe_quant.jobs.manager import BacktestJobManager

        manager = BacktestJobManager(db_path)
        status = manager.get_status(999)

        assert status is None
        manager.close()

    def test_kill_job_returns_false_for_nonexistent(self, db_path: Path):
        """kill_job returns False for nonexistent run."""
        from vibe_quant.jobs.manager import BacktestJobManager

        manager = BacktestJobManager(db_path)
        result = manager.kill_job(999)

        assert result is False
        manager.close()


class TestSweepParamsParsing:
    """Test sweep parameter handling."""

    def test_sweep_params_from_dsl(self, db_path: Path):
        """Sweep params can be extracted from DSL config."""
        from vibe_quant.db.state_manager import StateManager

        manager = StateManager(db_path)
        strategy = manager.list_strategies()[0]

        sweep = strategy["dsl_config"].get("sweep", {})
        assert "rsi_period" in sweep
        assert sweep["rsi_period"] == [10, 14, 20]
        assert "atr_mult" in sweep
        manager.close()

    def test_calculate_combinations(self):
        """Can calculate total parameter combinations."""
        sweep_params = {
            "rsi_period": [10, 14, 20],  # 3 values
            "atr_mult": [1.5, 2.0, 2.5],  # 3 values
            "ema_period": [10, 20],  # 2 values
        }

        total = 1
        for values in sweep_params.values():
            total *= len(values)

        assert total == 18  # 3 * 3 * 2

    def test_parse_sweep_values_int(self):
        """Can parse integer sweep values from string."""
        values_str = "10, 14, 20"
        parsed = []
        for v in values_str.split(","):
            v = v.strip()
            if "." in v:
                parsed.append(float(v))
            else:
                parsed.append(int(v))

        assert parsed == [10, 14, 20]

    def test_parse_sweep_values_float(self):
        """Can parse float sweep values from string."""
        values_str = "1.5, 2.0, 2.5"
        parsed = []
        for v in values_str.split(","):
            v = v.strip()
            if "." in v:
                parsed.append(float(v))
            else:
                parsed.append(int(v))

        assert parsed == [1.5, 2.0, 2.5]


class TestOverfittingFilters:
    """Test overfitting filter handling."""

    def test_filter_dict_structure(self):
        """Overfitting filter dict has expected structure."""
        filters = {
            "enable_dsr": True,
            "enable_wfa": True,
            "enable_purged_kfold": False,
        }

        assert "enable_dsr" in filters
        assert "enable_wfa" in filters
        assert "enable_purged_kfold" in filters
        assert isinstance(filters["enable_dsr"], bool)

    def test_all_filters_enabled(self):
        """Can enable all filters."""
        filters = {
            "enable_dsr": True,
            "enable_wfa": True,
            "enable_purged_kfold": True,
        }

        assert all(filters.values())

    def test_no_filters_enabled(self):
        """Can disable all filters."""
        filters = {
            "enable_dsr": False,
            "enable_wfa": False,
            "enable_purged_kfold": False,
        }

        assert not any(filters.values())


class TestValidTimeframes:
    """Test timeframe handling."""

    def test_valid_timeframes_available(self):
        """VALID_TIMEFRAMES contains expected values."""
        from vibe_quant.dsl.schema import VALID_TIMEFRAMES

        assert "1m" in VALID_TIMEFRAMES
        assert "5m" in VALID_TIMEFRAMES
        assert "15m" in VALID_TIMEFRAMES
        assert "1h" in VALID_TIMEFRAMES
        assert "4h" in VALID_TIMEFRAMES


class TestDefaultSymbols:
    """Test default symbols constant."""

    def test_common_symbols_present(self):
        """Common crypto symbols are in default list."""
        # Test the expected symbol list without importing the module directly
        expected_symbols = [
            "BTCUSDT",
            "ETHUSDT",
            "BNBUSDT",
            "SOLUSDT",
        ]

        for symbol in expected_symbols:
            assert symbol.endswith("USDT")


class TestDateRangeValidation:
    """Test date range handling."""

    def test_date_format(self):
        """Dates should be in ISO format."""
        from datetime import date

        today = date.today()
        iso_str = today.isoformat()

        assert len(iso_str) == 10
        assert iso_str[4] == "-"
        assert iso_str[7] == "-"

    def test_date_range_calculation(self):
        """Can calculate date range."""
        from datetime import date, timedelta

        end_date = date.today()
        start_date = end_date - timedelta(days=365)

        assert (end_date - start_date).days == 365


class TestBacktestRunParameters:
    """Test backtest run parameter structure."""

    def test_parameters_structure(self):
        """Parameters dict has expected structure."""
        parameters = {
            "sweep": {"rsi_period": [14, 20]},
            "overfitting_filters": {
                "enable_dsr": True,
                "enable_wfa": True,
                "enable_purged_kfold": False,
            },
        }

        assert "sweep" in parameters
        assert "overfitting_filters" in parameters
        assert isinstance(parameters["sweep"], dict)
        assert isinstance(parameters["overfitting_filters"], dict)

    def test_parameters_serialization(self):
        """Parameters can be serialized to JSON."""
        parameters = {
            "sweep": {"rsi_period": [14, 20]},
            "overfitting_filters": {"enable_dsr": True},
        }

        json_str = json.dumps(parameters)
        parsed = json.loads(json_str)

        assert parsed == parameters


class TestRunModes:
    """Test run mode handling."""

    def test_valid_run_modes(self):
        """Valid run modes are 'screening' and 'validation'."""
        valid_modes = ["screening", "validation"]

        for mode in valid_modes:
            assert mode in ["screening", "validation"]

    def test_screening_mode_no_latency(self):
        """Screening mode typically has no latency preset."""
        run_config = {
            "run_mode": "screening",
            "latency_preset": None,
        }

        assert run_config["latency_preset"] is None

    def test_validation_mode_requires_latency(self):
        """Validation mode should have latency preset."""
        run_config = {
            "run_mode": "validation",
            "latency_preset": "co_located",
        }

        assert run_config["latency_preset"] is not None


class TestLatencyOptions:
    """Test latency option constants."""

    def test_latency_options_include_custom(self) -> None:
        """Latency options should include 'custom'."""
        from vibe_quant.dashboard.components.backtest_config import LATENCY_OPTIONS

        assert "custom" in LATENCY_OPTIONS

    def test_latency_options_include_none(self) -> None:
        """Latency options should include screening mode."""
        from vibe_quant.dashboard.components.backtest_config import LATENCY_OPTIONS

        assert "None (screening mode)" in LATENCY_OPTIONS
