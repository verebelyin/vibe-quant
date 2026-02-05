"""Tests for SQLite state database."""

from pathlib import Path

import pytest

from vibe_quant.db import StateManager, get_connection


class TestConnection:
    """Tests for database connection factory."""

    def test_connection_creates_file(self, tmp_path: Path) -> None:
        """Connection should create database file."""
        db_path = tmp_path / "test.db"
        conn = get_connection(db_path)
        assert db_path.exists()
        conn.close()

    def test_connection_enables_wal_mode(self, tmp_path: Path) -> None:
        """Connection should enable WAL journal mode."""
        db_path = tmp_path / "test.db"
        conn = get_connection(db_path)
        cursor = conn.execute("PRAGMA journal_mode;")
        mode = cursor.fetchone()[0]
        assert mode == "wal"
        conn.close()

    def test_connection_enables_foreign_keys(self, tmp_path: Path) -> None:
        """Connection should enable foreign keys."""
        db_path = tmp_path / "test.db"
        conn = get_connection(db_path)
        cursor = conn.execute("PRAGMA foreign_keys;")
        enabled = cursor.fetchone()[0]
        assert enabled == 1
        conn.close()


class TestStateManager:
    """Tests for StateManager CRUD operations."""

    @pytest.fixture
    def state_manager(self, tmp_path: Path) -> StateManager:
        """Create StateManager with temp database."""
        db_path = tmp_path / "test.db"
        manager = StateManager(db_path)
        yield manager
        manager.close()

    def test_create_and_get_strategy(self, state_manager: StateManager) -> None:
        """Should create and retrieve a strategy."""
        dsl_config = {
            "name": "test_strategy",
            "timeframe": "5m",
            "indicators": {"rsi": {"type": "RSI", "period": 14}},
        }
        strategy_id = state_manager.create_strategy(
            name="test_strategy",
            dsl_config=dsl_config,
            description="Test description",
            strategy_type="technical",
        )
        assert strategy_id > 0

        strategy = state_manager.get_strategy(strategy_id)
        assert strategy is not None
        assert strategy["name"] == "test_strategy"
        assert strategy["dsl_config"] == dsl_config
        assert strategy["description"] == "Test description"
        assert strategy["strategy_type"] == "technical"
        assert strategy["version"] == 1
        assert strategy["is_active"] == 1

    def test_get_strategy_by_name(self, state_manager: StateManager) -> None:
        """Should retrieve strategy by name."""
        dsl_config = {"name": "named_strategy"}
        state_manager.create_strategy(name="named_strategy", dsl_config=dsl_config)

        strategy = state_manager.get_strategy_by_name("named_strategy")
        assert strategy is not None
        assert strategy["name"] == "named_strategy"

    def test_list_strategies(self, state_manager: StateManager) -> None:
        """Should list all active strategies."""
        state_manager.create_strategy(name="strategy1", dsl_config={})
        state_manager.create_strategy(name="strategy2", dsl_config={})

        strategies = state_manager.list_strategies()
        assert len(strategies) == 2

    def test_update_strategy(self, state_manager: StateManager) -> None:
        """Should update strategy and increment version."""
        strategy_id = state_manager.create_strategy(name="update_test", dsl_config={})

        new_config = {"updated": True}
        state_manager.update_strategy(strategy_id, dsl_config=new_config)

        strategy = state_manager.get_strategy(strategy_id)
        assert strategy is not None
        assert strategy["dsl_config"] == new_config
        assert strategy["version"] == 2

    def test_create_and_get_sizing_config(self, state_manager: StateManager) -> None:
        """Should create and retrieve sizing config."""
        config_id = state_manager.create_sizing_config(
            name="default_sizing",
            method="fixed_fractional",
            config={"risk_per_trade": 0.02, "max_position_pct": 0.10},
        )
        assert config_id > 0

        config = state_manager.get_sizing_config(config_id)
        assert config is not None
        assert config["method"] == "fixed_fractional"
        assert config["config"]["risk_per_trade"] == 0.02

    def test_create_and_get_risk_config(self, state_manager: StateManager) -> None:
        """Should create and retrieve risk config."""
        config_id = state_manager.create_risk_config(
            name="default_risk",
            strategy_level={"max_drawdown_halt_pct": 0.15},
            portfolio_level={"max_portfolio_drawdown_pct": 0.20},
        )
        assert config_id > 0

        config = state_manager.get_risk_config(config_id)
        assert config is not None
        assert config["strategy_level"]["max_drawdown_halt_pct"] == 0.15
        assert config["portfolio_level"]["max_portfolio_drawdown_pct"] == 0.20

    def test_create_and_get_backtest_run(self, state_manager: StateManager) -> None:
        """Should create and retrieve backtest run."""
        strategy_id = state_manager.create_strategy(name="bt_strategy", dsl_config={})

        run_id = state_manager.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="screening",
            symbols=["BTCUSDT-PERP", "ETHUSDT-PERP"],
            timeframe="5m",
            start_date="2024-01-01",
            end_date="2024-12-31",
            parameters={"rsi_period": 14},
            latency_preset="retail",
        )
        assert run_id > 0

        run = state_manager.get_backtest_run(run_id)
        assert run is not None
        assert run["run_mode"] == "screening"
        assert run["symbols"] == ["BTCUSDT-PERP", "ETHUSDT-PERP"]
        assert run["parameters"]["rsi_period"] == 14
        assert run["status"] == "pending"

    def test_update_backtest_run_status(self, state_manager: StateManager) -> None:
        """Should update backtest run status."""
        strategy_id = state_manager.create_strategy(name="status_test", dsl_config={})
        run_id = state_manager.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="validation",
            symbols=["BTCUSDT-PERP"],
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-06-30",
            parameters={},
        )

        state_manager.update_backtest_run_status(run_id, "running", pid=12345)
        run = state_manager.get_backtest_run(run_id)
        assert run is not None
        assert run["status"] == "running"
        assert run["pid"] == 12345
        assert run["started_at"] is not None

        state_manager.update_backtest_run_status(run_id, "completed")
        run = state_manager.get_backtest_run(run_id)
        assert run is not None
        assert run["status"] == "completed"
        assert run["completed_at"] is not None

    def test_save_and_get_backtest_result(self, state_manager: StateManager) -> None:
        """Should save and retrieve backtest results."""
        strategy_id = state_manager.create_strategy(name="result_test", dsl_config={})
        run_id = state_manager.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="validation",
            symbols=["BTCUSDT-PERP"],
            timeframe="5m",
            start_date="2024-01-01",
            end_date="2024-12-31",
            parameters={},
        )

        result_id = state_manager.save_backtest_result(
            run_id,
            {
                "total_return": 0.25,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.10,
                "total_trades": 100,
                "win_rate": 0.55,
            },
        )
        assert result_id > 0

        result = state_manager.get_backtest_result(run_id)
        assert result is not None
        assert result["total_return"] == 0.25
        assert result["sharpe_ratio"] == 1.5

    def test_save_and_get_trades(self, state_manager: StateManager) -> None:
        """Should save and retrieve trades."""
        strategy_id = state_manager.create_strategy(name="trades_test", dsl_config={})
        run_id = state_manager.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="validation",
            symbols=["BTCUSDT-PERP"],
            timeframe="5m",
            start_date="2024-01-01",
            end_date="2024-12-31",
            parameters={},
        )

        trades = [
            {
                "symbol": "BTCUSDT-PERP",
                "direction": "LONG",
                "entry_time": "2024-01-15T10:00:00",
                "entry_price": 42000.0,
                "quantity": 0.1,
            },
            {
                "symbol": "BTCUSDT-PERP",
                "direction": "SHORT",
                "entry_time": "2024-01-20T14:00:00",
                "entry_price": 43000.0,
                "quantity": 0.1,
            },
        ]
        state_manager.save_trades_batch(run_id, trades)

        saved_trades = state_manager.get_trades(run_id)
        assert len(saved_trades) == 2
        assert saved_trades[0]["direction"] == "LONG"
        assert saved_trades[1]["direction"] == "SHORT"

    def test_save_and_get_sweep_results(self, state_manager: StateManager) -> None:
        """Should save and retrieve sweep results."""
        strategy_id = state_manager.create_strategy(name="sweep_test", dsl_config={})
        run_id = state_manager.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="screening",
            symbols=["BTCUSDT-PERP"],
            timeframe="5m",
            start_date="2024-01-01",
            end_date="2024-12-31",
            parameters={},
        )

        results = [
            {"parameters": {"rsi_period": 14}, "sharpe_ratio": 1.2, "max_drawdown": 0.08},
            {"parameters": {"rsi_period": 21}, "sharpe_ratio": 1.5, "max_drawdown": 0.12},
        ]
        state_manager.save_sweep_results_batch(run_id, results)

        saved = state_manager.get_sweep_results(run_id)
        assert len(saved) == 2
        assert saved[0]["sharpe_ratio"] == 1.5  # Sorted by Sharpe DESC
        assert saved[0]["parameters"]["rsi_period"] == 21

    def test_mark_pareto_optimal(self, state_manager: StateManager) -> None:
        """Should mark sweep results as Pareto optimal."""
        strategy_id = state_manager.create_strategy(name="pareto_test", dsl_config={})
        run_id = state_manager.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="screening",
            symbols=["BTCUSDT-PERP"],
            timeframe="5m",
            start_date="2024-01-01",
            end_date="2024-12-31",
            parameters={},
        )

        result_id = state_manager.save_sweep_result(
            run_id, {"parameters": {"rsi": 14}, "sharpe_ratio": 1.5}
        )
        state_manager.mark_pareto_optimal([result_id])

        pareto = state_manager.get_sweep_results(run_id, pareto_only=True)
        assert len(pareto) == 1
        assert pareto[0]["is_pareto_optimal"] == 1

    def test_register_and_get_job(self, state_manager: StateManager) -> None:
        """Should register and retrieve background job."""
        strategy_id = state_manager.create_strategy(name="job_test", dsl_config={})
        run_id = state_manager.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="screening",
            symbols=["BTCUSDT-PERP"],
            timeframe="5m",
            start_date="2024-01-01",
            end_date="2024-12-31",
            parameters={},
        )

        job_id = state_manager.register_job(
            run_id=run_id,
            pid=99999,
            job_type="screening",
            log_file="/tmp/test.log",
        )
        assert job_id > 0

        job = state_manager.get_job(run_id)
        assert job is not None
        assert job["pid"] == 99999
        assert job["job_type"] == "screening"
        assert job["status"] == "running"

    def test_get_running_jobs(self, state_manager: StateManager) -> None:
        """Should get all running jobs."""
        strategy_id = state_manager.create_strategy(name="running_test", dsl_config={})
        run_id = state_manager.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="screening",
            symbols=["BTCUSDT-PERP"],
            timeframe="5m",
            start_date="2024-01-01",
            end_date="2024-12-31",
            parameters={},
        )

        state_manager.register_job(run_id, 11111, "screening")

        running = state_manager.get_running_jobs()
        assert len(running) == 1
        assert running[0]["pid"] == 11111
