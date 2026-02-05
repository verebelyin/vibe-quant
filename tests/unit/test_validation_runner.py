"""Tests for validation runner."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.validation.latency import LatencyPreset
from vibe_quant.validation.runner import (
    TradeRecord,
    ValidationResult,
    ValidationRunner,
    ValidationRunnerError,
    list_validation_runs,
)


@pytest.fixture
def temp_db() -> Path:
    """Create temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        return Path(f.name)


@pytest.fixture
def temp_logs() -> Path:
    """Create temporary logs directory."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_dsl_config() -> dict[str, object]:
    """Sample valid DSL configuration."""
    return {
        "name": "test_strategy",
        "description": "Test strategy for validation",
        "version": 1,
        "timeframe": "5m",
        "additional_timeframes": [],
        "indicators": {
            "rsi": {
                "type": "RSI",
                "period": 14,
                "source": "close",
            },
            "ema": {
                "type": "EMA",
                "period": 20,
                "source": "close",
            },
        },
        "entry_conditions": {
            "long": ["rsi < 30"],
            "short": ["rsi > 70"],
        },
        "exit_conditions": {
            "long": ["rsi > 50"],
            "short": ["rsi < 50"],
        },
        "stop_loss": {
            "type": "fixed_pct",
            "percent": 2.0,
        },
        "take_profit": {
            "type": "fixed_pct",
            "percent": 4.0,
        },
    }


@pytest.fixture
def state_with_strategy(temp_db: Path, sample_dsl_config: dict[str, object]) -> StateManager:
    """Create state manager with a test strategy and validation run."""
    state = StateManager(temp_db)

    # Create strategy
    strategy_id = state.create_strategy(
        name="test_strategy",
        dsl_config=sample_dsl_config,
        description="Test",
    )

    # Create validation run
    state.create_backtest_run(
        strategy_id=strategy_id,
        run_mode="validation",
        symbols=["BTCUSDT-PERP"],
        timeframe="5m",
        start_date="2025-01-01",
        end_date="2025-01-31",
        parameters={"rsi_period": 14},
        latency_preset="retail",
    )

    return state


class TestTradeRecord:
    """Tests for TradeRecord dataclass."""

    def test_trade_record_creation(self) -> None:
        """Create trade record with all fields."""
        trade = TradeRecord(
            symbol="BTCUSDT-PERP",
            direction="LONG",
            leverage=10,
            entry_time="2025-01-15T10:00:00Z",
            exit_time="2025-01-15T14:00:00Z",
            entry_price=42000.0,
            exit_price=42500.0,
            quantity=0.1,
            entry_fee=5.0,
            exit_fee=5.0,
            funding_fees=2.0,
            slippage_cost=1.5,
            gross_pnl=50.0,
            net_pnl=36.5,
            roi_percent=1.2,
            exit_reason="take_profit",
        )

        assert trade.symbol == "BTCUSDT-PERP"
        assert trade.direction == "LONG"
        assert trade.leverage == 10
        assert trade.net_pnl == 36.5

    def test_trade_record_to_dict(self) -> None:
        """Convert trade record to dictionary."""
        trade = TradeRecord(
            symbol="ETHUSDT-PERP",
            direction="SHORT",
            leverage=5,
            entry_time="2025-01-15T10:00:00Z",
            exit_time="2025-01-15T12:00:00Z",
            entry_price=2500.0,
            exit_price=2450.0,
            quantity=1.0,
        )

        d = trade.to_dict()

        assert d["symbol"] == "ETHUSDT-PERP"
        assert d["direction"] == "SHORT"
        assert d["leverage"] == 5
        assert d["entry_price"] == 2500.0
        assert d["exit_price"] == 2450.0

    def test_trade_record_defaults(self) -> None:
        """Trade record has sensible defaults."""
        trade = TradeRecord(
            symbol="BTCUSDT-PERP",
            direction="LONG",
            leverage=1,
            entry_time="2025-01-15T10:00:00Z",
            exit_time=None,
            entry_price=42000.0,
            exit_price=None,
            quantity=0.01,
        )

        assert trade.entry_fee == 0.0
        assert trade.exit_fee == 0.0
        assert trade.funding_fees == 0.0
        assert trade.slippage_cost == 0.0
        assert trade.gross_pnl == 0.0
        assert trade.net_pnl == 0.0


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_result_creation(self) -> None:
        """Create validation result."""
        result = ValidationResult(
            run_id=42,
            strategy_name="test_strategy",
            total_return=10.5,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=5.0,
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
        )

        assert result.run_id == 42
        assert result.strategy_name == "test_strategy"
        assert result.total_return == 10.5
        assert result.sharpe_ratio == 1.5

    def test_validation_result_to_metrics_dict(self) -> None:
        """Convert validation result to metrics dictionary."""
        result = ValidationResult(
            run_id=1,
            strategy_name="test",
            total_return=5.0,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            max_drawdown=8.0,
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
            win_rate=60.0,
            profit_factor=1.5,
            total_fees=100.0,
            total_funding=25.0,
            total_slippage=15.0,
            execution_time_seconds=3.5,
        )

        metrics = result.to_metrics_dict()

        assert metrics["total_return"] == 5.0
        assert metrics["sharpe_ratio"] == 1.2
        assert metrics["sortino_ratio"] == 1.5
        assert metrics["max_drawdown"] == 8.0
        assert metrics["total_trades"] == 50
        assert metrics["win_rate"] == 60.0
        assert metrics["profit_factor"] == 1.5
        assert metrics["total_fees"] == 100.0
        assert metrics["total_funding"] == 25.0
        assert metrics["total_slippage"] == 15.0
        assert metrics["execution_time_seconds"] == 3.5

    def test_validation_result_with_trades(self) -> None:
        """Validation result can hold trade records."""
        trade = TradeRecord(
            symbol="BTCUSDT-PERP",
            direction="LONG",
            leverage=10,
            entry_time="2025-01-15T10:00:00Z",
            exit_time="2025-01-15T14:00:00Z",
            entry_price=42000.0,
            exit_price=42500.0,
            quantity=0.1,
        )

        result = ValidationResult(
            run_id=1,
            strategy_name="test",
            trades=[trade],
        )

        assert len(result.trades) == 1
        assert result.trades[0].symbol == "BTCUSDT-PERP"


class TestValidationRunner:
    """Tests for ValidationRunner class."""

    def test_runner_initialization(self, temp_db: Path, temp_logs: Path) -> None:
        """Initialize validation runner."""
        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)
        assert runner is not None
        runner.close()

    def test_runner_run_not_found(self, temp_db: Path, temp_logs: Path) -> None:
        """Run with non-existent run_id raises error."""
        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)

        with pytest.raises(ValidationRunnerError, match="not found"):
            runner.run(run_id=9999)

        runner.close()

    def test_runner_run_wrong_mode(
        self,
        temp_db: Path,
        temp_logs: Path,
        sample_dsl_config: dict[str, object],
    ) -> None:
        """Run with screening mode raises error."""
        state = StateManager(temp_db)
        strategy_id = state.create_strategy(
            name="test_strategy",
            dsl_config=sample_dsl_config,
        )
        run_id = state.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="screening",  # Wrong mode
            symbols=["BTCUSDT-PERP"],
            timeframe="5m",
            start_date="2025-01-01",
            end_date="2025-01-31",
            parameters={},
        )
        state.close()

        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)
        with pytest.raises(ValidationRunnerError, match="not a validation run"):
            runner.run(run_id=run_id)
        runner.close()

    def test_runner_run_success(
        self,
        temp_db: Path,
        temp_logs: Path,
        state_with_strategy: StateManager,
    ) -> None:
        """Successfully run validation backtest."""
        state_with_strategy.close()

        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)
        result = runner.run(run_id=1)  # First run has ID 1

        assert result is not None
        assert result.run_id == 1
        assert result.strategy_name == "test_strategy"
        assert result.total_trades > 0
        assert result.execution_time_seconds > 0
        assert len(result.trades) > 0

        runner.close()

    def test_runner_run_with_latency_override(
        self,
        temp_db: Path,
        temp_logs: Path,
        state_with_strategy: StateManager,
    ) -> None:
        """Run with latency preset override."""
        state_with_strategy.close()

        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)
        result = runner.run(
            run_id=1,
            latency_preset=LatencyPreset.COLOCATED,
        )

        assert result is not None
        runner.close()

    def test_runner_run_with_string_latency(
        self,
        temp_db: Path,
        temp_logs: Path,
        state_with_strategy: StateManager,
    ) -> None:
        """Run with string latency preset."""
        state_with_strategy.close()

        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)
        result = runner.run(run_id=1, latency_preset="domestic")

        assert result is not None
        runner.close()

    def test_runner_stores_results(
        self,
        temp_db: Path,
        temp_logs: Path,
        state_with_strategy: StateManager,
    ) -> None:
        """Verify results are stored in database."""
        state_with_strategy.close()

        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)
        result = runner.run(run_id=1)
        runner.close()

        # Verify results in database
        state = StateManager(temp_db)
        stored_result = state.get_backtest_result(1)

        assert stored_result is not None
        assert stored_result["sharpe_ratio"] == result.sharpe_ratio
        assert stored_result["total_return"] == result.total_return
        assert stored_result["max_drawdown"] == result.max_drawdown

        # Verify trades stored
        trades = state.get_trades(1)
        assert len(trades) == len(result.trades)

        state.close()

    def test_runner_updates_status(
        self,
        temp_db: Path,
        temp_logs: Path,
        state_with_strategy: StateManager,
    ) -> None:
        """Verify run status is updated to completed."""
        state_with_strategy.close()

        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)
        runner.run(run_id=1)
        runner.close()

        # Verify status
        state = StateManager(temp_db)
        run_config = state.get_backtest_run(1)

        assert run_config is not None
        assert run_config["status"] == "completed"
        state.close()

    def test_runner_writes_events(
        self,
        temp_db: Path,
        temp_logs: Path,
        state_with_strategy: StateManager,
    ) -> None:
        """Verify events are written to log file."""
        state_with_strategy.close()

        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)
        runner.run(run_id=1)
        runner.close()

        # Check log file exists
        log_file = temp_logs / "1.jsonl"
        assert log_file.exists()

        # Read and verify events
        with open(log_file) as f:
            lines = f.readlines()

        assert len(lines) > 0

        # First event should be start marker
        first_event = json.loads(lines[0])
        assert "BACKTEST_START" in str(first_event.get("data", {}))

        # Last event should be completion marker
        last_event = json.loads(lines[-1])
        assert "BACKTEST_COMPLETE" in str(last_event.get("data", {}))

    def test_runner_sets_failed_on_exception(
        self,
        temp_db: Path,
        temp_logs: Path,
        state_with_strategy: StateManager,
    ) -> None:
        """Exception during run sets status to 'failed' and raises ValidationRunnerError."""
        state_with_strategy.close()

        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)

        # Monkey-patch _run_backtest_mock to raise
        def _exploding_mock(*args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated engine crash")

        runner._run_backtest_mock = _exploding_mock  # type: ignore[assignment]

        with pytest.raises(ValidationRunnerError, match="simulated engine crash"):
            runner.run(run_id=1)

        runner.close()

        # Verify status set to failed with error message
        state = StateManager(temp_db)
        run = state.get_backtest_run(1)
        assert run is not None
        assert run["status"] == "failed"
        assert "simulated engine crash" in str(run.get("error_message", ""))
        state.close()

    def test_runner_invalid_strategy(
        self,
        temp_db: Path,
        temp_logs: Path,
        sample_dsl_config: dict[str, object],
    ) -> None:
        """Run with deleted strategy raises error."""
        state = StateManager(temp_db)

        # Create a strategy and a run
        strategy_id = state.create_strategy(
            name="to_delete",
            dsl_config=sample_dsl_config,
        )
        run_id = state.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="validation",
            symbols=["BTCUSDT-PERP"],
            timeframe="5m",
            start_date="2025-01-01",
            end_date="2025-01-31",
            parameters={},
        )

        # Disable foreign keys temporarily to delete the strategy
        state.conn.execute("PRAGMA foreign_keys = OFF")
        state.conn.execute("DELETE FROM strategies WHERE id = ?", (strategy_id,))
        state.conn.execute("PRAGMA foreign_keys = ON")
        state.conn.commit()
        state.close()

        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)
        with pytest.raises(ValidationRunnerError, match="not found"):
            runner.run(run_id=run_id)
        runner.close()

    def test_mock_metrics_use_decimal_format(
        self,
        temp_db: Path,
        temp_logs: Path,
        state_with_strategy: StateManager,
    ) -> None:
        """Mock metrics should use decimal format (0.052 not 5.2)."""
        state_with_strategy.close()

        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)
        result = runner.run(run_id=1)
        runner.close()

        # All percentage-like values should be in decimal form
        assert 0 <= result.win_rate <= 1.0
        assert result.total_return < 1.0  # Not percentage points
        assert result.max_drawdown < 1.0  # Not percentage points


class TestListValidationRuns:
    """Tests for list_validation_runs function."""

    def test_list_empty(self, temp_db: Path) -> None:
        """List returns empty when no runs exist."""
        runs = list_validation_runs(db_path=temp_db)
        assert runs == []

    def test_list_filters_validation_only(
        self,
        temp_db: Path,
        sample_dsl_config: dict[str, object],
    ) -> None:
        """List returns only validation runs."""
        state = StateManager(temp_db)

        strategy_id = state.create_strategy(
            name="test_strategy",
            dsl_config=sample_dsl_config,
        )

        # Create screening run
        state.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="screening",
            symbols=["BTCUSDT-PERP"],
            timeframe="5m",
            start_date="2025-01-01",
            end_date="2025-01-31",
            parameters={},
        )

        # Create validation run
        state.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="validation",
            symbols=["BTCUSDT-PERP"],
            timeframe="5m",
            start_date="2025-01-01",
            end_date="2025-01-31",
            parameters={},
            latency_preset="retail",
        )

        state.close()

        runs = list_validation_runs(db_path=temp_db)
        assert len(runs) == 1
        assert runs[0]["latency_preset"] == "retail"

    def test_list_with_results(
        self,
        temp_db: Path,
        temp_logs: Path,
        state_with_strategy: StateManager,
    ) -> None:
        """List includes result metrics when available."""
        state_with_strategy.close()

        # Run validation to create results
        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)
        result = runner.run(run_id=1)
        runner.close()

        runs = list_validation_runs(db_path=temp_db)

        assert len(runs) == 1
        assert runs[0]["run_id"] == 1
        assert runs[0]["sharpe_ratio"] == result.sharpe_ratio
        assert runs[0]["total_return"] == result.total_return
        assert runs[0]["max_drawdown"] == result.max_drawdown
        assert runs[0]["total_trades"] == result.total_trades

    def test_list_limit(
        self,
        temp_db: Path,
        sample_dsl_config: dict[str, object],
    ) -> None:
        """List respects limit parameter."""
        state = StateManager(temp_db)

        strategy_id = state.create_strategy(
            name="test_strategy",
            dsl_config=sample_dsl_config,
        )

        # Create multiple runs
        for _ in range(5):
            state.create_backtest_run(
                strategy_id=strategy_id,
                run_mode="validation",
                symbols=["BTCUSDT-PERP"],
                timeframe="5m",
                start_date="2025-01-01",
                end_date="2025-01-31",
                parameters={},
            )

        state.close()

        runs = list_validation_runs(db_path=temp_db, limit=3)
        assert len(runs) == 3
