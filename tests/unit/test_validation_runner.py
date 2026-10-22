"""Tests for validation runner."""

from __future__ import annotations

import json
import math
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

import vibe_quant.data.catalog as catalog_mod
from vibe_quant.data.catalog import CatalogManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.validation.latency import LatencyPreset
from vibe_quant.validation.runner import (
    TradeRecord,
    ValidationResult,
    ValidationRunner,
    ValidationRunnerError,
    list_validation_runs,
)
from vibe_quant.validation.venue import DEFAULT_STARTING_BALANCE_USDT

#: Captured before the autouse ``_no_catalog_coverage_check`` fixture replaces it: the
#: launch-error precedence tests below need the REAL clamp (rabmo.3 S5, R27a/R27b).
_REAL_CLAMP_RUN_WINDOW = ValidationRunner._clamp_run_window


@pytest.fixture(autouse=True)
def _no_catalog_coverage_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests mock the backtest and use a fake symbol ("BTCUSDT-PERP")
    with no catalog data; data-window clamping/failure is covered by
    tests/unit/test_audit_metrics_window.py."""
    monkeypatch.setattr(
        ValidationRunner, "_clamp_run_window", lambda self, run_config, tf: (run_config, None)
    )


def _make_mock_result(
    run_id: int = 1,
    strategy_name: str = "test_strategy",
    starting_balance: float = 1000.0,
) -> ValidationResult:
    """Create a realistic mock ValidationResult for unit tests."""
    return ValidationResult(
        run_id=run_id,
        strategy_name=strategy_name,
        starting_balance=starting_balance,
        total_return=0.052,
        sharpe_ratio=1.4,
        sortino_ratio=1.8,
        max_drawdown=0.08,
        total_trades=15,
        winning_trades=9,
        losing_trades=6,
        win_rate=0.6,
        profit_factor=1.5,
        total_fees=45.0,
        total_funding=12.0,
        total_slippage=8.0,
        trades=[
            TradeRecord(
                symbol="BTCUSDT-PERP",
                direction="LONG",
                leverage=10,
                entry_time="2025-01-10T08:00:00Z",
                exit_time="2025-01-10T14:00:00Z",
                entry_price=42000.0,
                exit_price=42500.0,
                quantity=0.1,
                entry_fee=2.1,
                exit_fee=2.125,
                net_pnl=45.775,
                gross_pnl=50.0,
                exit_reason="take_profit",
            ),
            TradeRecord(
                symbol="BTCUSDT-PERP",
                direction="SHORT",
                leverage=10,
                entry_time="2025-01-12T10:00:00Z",
                exit_time="2025-01-12T16:00:00Z",
                entry_price=43000.0,
                exit_price=42800.0,
                quantity=0.1,
                entry_fee=2.15,
                exit_fee=2.14,
                net_pnl=15.71,
                gross_pnl=20.0,
                exit_reason="take_profit",
            ),
        ],
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

    def test_sub_bar_validation_adds_execution_delay_param(
        self,
        temp_db: Path,
        temp_logs: Path,
    ) -> None:
        """Sub-5m validation without finer data defers EVERY fill one bar (e70tl.13)."""
        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)

        params = runner._augment_strategy_params_for_validation({}, timeframe="1m")

        assert params["execution_delay_probability"] == 1.0
        runner.close()

    def test_validation_param_override_is_preserved(
        self,
        temp_db: Path,
        temp_logs: Path,
    ) -> None:
        """Explicit run parameters should beat the default sub-bar delay."""
        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)

        params = runner._augment_strategy_params_for_validation(
            {"execution_delay_probability": 0.45},
            timeframe="1m",
        )

        assert params["execution_delay_probability"] == 0.45
        runner.close()

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
        runner._run_backtest = lambda **kwargs: _make_mock_result(  # type: ignore[assignment]
            run_id=kwargs["run_id"], strategy_name=kwargs["strategy_name"]
        )
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
        runner._run_backtest = lambda **kwargs: _make_mock_result(  # type: ignore[assignment]
            run_id=kwargs["run_id"], strategy_name=kwargs["strategy_name"]
        )
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
        runner._run_backtest = lambda **kwargs: _make_mock_result(  # type: ignore[assignment]
            run_id=kwargs["run_id"], strategy_name=kwargs["strategy_name"]
        )
        result = runner.run(run_id=1, latency_preset="domestic")

        assert result is not None
        runner.close()

    def test_build_walk_forward_windows(self) -> None:
        """Walk-forward windows are generated with expected rolling boundaries."""
        windows = ValidationRunner._build_walk_forward_windows(
            range_start=date(2025, 1, 1),
            range_end=date(2025, 1, 31),
            train_days=7,
            test_days=7,
            step_days=7,
        )

        assert len(windows) == 3
        assert windows[0].train_start == "2025-01-01"
        assert windows[0].train_end == "2025-01-08"
        assert windows[0].test_start == "2025-01-08"
        assert windows[0].test_end == "2025-01-15"
        assert windows[2].test_start == "2025-01-22"
        assert windows[2].test_end == "2025-01-29"

    def test_run_walk_forward_success(
        self,
        temp_db: Path,
        temp_logs: Path,
        state_with_strategy: StateManager,
    ) -> None:
        """Walk-forward should execute windows and persist aggregated metrics."""
        state_with_strategy.close()
        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)

        called_windows: list[tuple[str, str]] = []

        def _fake_backtest(**kwargs: object) -> ValidationResult:
            run_cfg = kwargs["run_config"]
            assert isinstance(run_cfg, dict)
            start = str(run_cfg["start_date"])
            end = str(run_cfg["end_date"])
            called_windows.append((start, end))

            idx = len(called_windows)
            result = _make_mock_result(
                run_id=int(kwargs["run_id"]),
                strategy_name=str(kwargs["strategy_name"]),
            )
            result.total_return = 0.01 * idx
            result.sharpe_ratio = 1.0 + idx
            return result

        runner._run_backtest = _fake_backtest  # type: ignore[assignment]
        window_results = runner.run_walk_forward(
            run_id=1,
            train_days=7,
            test_days=7,
            step_days=7,
        )
        runner.close()

        assert len(window_results) == 3
        assert called_windows == [
            ("2025-01-08", "2025-01-15"),
            ("2025-01-15", "2025-01-22"),
            ("2025-01-22", "2025-01-29"),
        ]

        state = StateManager(temp_db)
        stored_result = state.get_backtest_result(1)
        run = state.get_backtest_run(1)
        trades = state.get_trades(1)
        state.close()

        assert stored_result is not None
        assert run is not None
        assert run["status"] == "completed"
        # Compounded: (1+0.01)*(1+0.02)*(1+0.03) - 1 = 0.061106
        assert stored_result["total_return"] == pytest.approx(0.061106, rel=1e-4)
        # Trade-weighted avg (equal trades per window): (2+3+4)*15/45 = 3.0
        assert stored_result["sharpe_ratio"] == pytest.approx(3.0)
        assert stored_result["total_trades"] == 45
        assert len(trades) == 6

    def test_run_walk_forward_no_windows_raises(
        self,
        temp_db: Path,
        temp_logs: Path,
        state_with_strategy: StateManager,
    ) -> None:
        """Invalid train/test sizing for run range should fail clearly."""
        state_with_strategy.close()
        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)
        with pytest.raises(ValidationRunnerError, match="No walk-forward windows fit"):
            runner.run_walk_forward(run_id=1, train_days=30, test_days=30, step_days=30)
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
        runner._run_backtest = lambda **kwargs: _make_mock_result(  # type: ignore[assignment]
            run_id=kwargs["run_id"], strategy_name=kwargs["strategy_name"]
        )
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
        runner._run_backtest = lambda **kwargs: _make_mock_result(  # type: ignore[assignment]
            run_id=kwargs["run_id"], strategy_name=kwargs["strategy_name"]
        )
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
        runner._run_backtest = lambda **kwargs: _make_mock_result(  # type: ignore[assignment]
            run_id=kwargs["run_id"], strategy_name=kwargs["strategy_name"]
        )
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

        # Monkey-patch _run_backtest to raise
        def _exploding_mock(*args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated engine crash")

        runner._run_backtest = _exploding_mock  # type: ignore[assignment]

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
        runner._run_backtest = lambda **kwargs: _make_mock_result(  # type: ignore[assignment]
            run_id=kwargs["run_id"], strategy_name=kwargs["strategy_name"]
        )
        result = runner.run(run_id=1)
        runner.close()

        # All percentage-like values should be in decimal form
        assert 0 <= result.win_rate <= 1.0
        assert result.total_return < 1.0  # Not percentage points
        assert result.max_drawdown < 1.0  # Not percentage points


class TestDetailTimeframe:
    """Tests for sub-bar detail timeframe resolution."""

    def test_resolve_detail_explicit_override(
        self, temp_db: Path, temp_logs: Path
    ) -> None:
        """Explicit detail_timeframe override is used directly."""
        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)
        result = runner._resolve_detail_timeframe(
            run_config={
                "symbols": '["BTCUSDT"]',
                "start_date": "2025-01-01",
                "end_date": "2025-01-31",
            },
            strategy_timeframe="1m",
            override="1s",
        )
        assert result == "1s"
        runner.close()

    def test_resolve_detail_from_run_config_parameters(
        self, temp_db: Path, temp_logs: Path
    ) -> None:
        """detail_timeframe in run parameters is used."""
        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)
        result = runner._resolve_detail_timeframe(
            run_config={
                "symbols": '["BTCUSDT"]',
                "start_date": "2025-01-01",
                "end_date": "2025-01-31",
                "parameters": {"detail_timeframe": "5s"},
            },
            strategy_timeframe="1m",
            override=None,
        )
        assert result == "5s"
        runner.close()

    def test_resolve_detail_no_dates_returns_none(
        self, temp_db: Path, temp_logs: Path
    ) -> None:
        """Missing run dates prevents auto-detection (returns None)."""
        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)
        result = runner._resolve_detail_timeframe(
            run_config={"symbols": '["BTCUSDT"]'},
            strategy_timeframe="1m",
            override=None,
        )
        assert result is None
        runner.close()

    def test_venue_config_with_detail_data_drops_latency_keeps_preset_fill(
        self, temp_db: Path, temp_logs: Path, state_with_strategy: StateManager
    ) -> None:
        """With detail data the preset picks the fill model but no LatencyModel
        runs: orders release on each strategy's own detail bar (yul7u.11)."""
        state_with_strategy.close()

        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)

        # Without detail data: latency should be None for 1m
        config_no_detail, _launch = runner._create_venue_config(
            {"starting_balance": 1000}, "cloud", timeframe="1m", has_detail_data=False
        )
        assert config_no_detail.latency_preset is None

        # With detail data: no latency model, fill model of the latency path
        config_with_detail, _launch = runner._create_venue_config(
            {"starting_balance": 1000}, "cloud", timeframe="1m", has_detail_data=True
        )
        assert config_with_detail.latency_preset is None
        assert config_with_detail.latency_config is None
        from vibe_quant.validation.fill_model import VolumeSlippageFillModelConfig

        assert isinstance(config_with_detail.fill_config, VolumeSlippageFillModelConfig)
        assert config_with_detail.fill_config.prob_best_price_fill == 1.0
        assert isinstance(config_no_detail.fill_config, VolumeSlippageFillModelConfig)
        assert config_no_detail.fill_config.prob_best_price_fill == 0.7


class TestInitialBalanceRule:
    """R1/R2/R7 (bead vibe-quant-zpbmo, task Z1): the venue balance comes from
    ``parameters.initial_balance`` through the ONE shared rule in
    ``vibe_quant.balance_settings``. An invalid stored value fails the run instead of
    silently running at 1000, and the dead top-level
    ``run_config["starting_balance"]`` fallback is gone."""

    def test_venue_config_uses_the_launch_balance(
        self, temp_db: Path, temp_logs: Path
    ) -> None:
        """R2: 10000.0 reaches the venue as the int 10000 ("10000 USDT" then)."""
        from vibe_quant.validation.venue import create_backtest_venue_config

        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)
        config, _launch = runner._create_venue_config(
            {"parameters": {"initial_balance": 10000.0}}, None, timeframe="1m"
        )
        runner.close()

        assert config.starting_balance_usdt == 10000
        assert create_backtest_venue_config(config).starting_balances == ["10000 USDT"]

    def test_venue_config_normal_branch_uses_the_launch_balance(
        self, temp_db: Path, temp_logs: Path
    ) -> None:
        """R2: the normal branch (``timeframe`` above sub-bar) keeps the launch 10000.

        The other direct assertions use ``timeframe="1m"``, which takes the sub-bar
        early return; this pins the ``_create_venue_config`` branch a runner.py:887
        ``starting_balance_usdt=1000`` mutant would otherwise hide in.
        """
        from vibe_quant.validation.venue import create_backtest_venue_config

        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)
        config, _launch = runner._create_venue_config(
            {"parameters": {"initial_balance": 10000.0}}, None, timeframe="15m"
        )
        runner.close()

        assert config.starting_balance_usdt == 10000
        assert create_backtest_venue_config(config).starting_balances == ["10000 USDT"]

    def test_venue_config_detail_branch_uses_the_launch_balance(
        self, temp_db: Path, temp_logs: Path
    ) -> None:
        """R2: detail data also takes the normal branch and keeps the launch 10000."""
        from vibe_quant.validation.venue import create_backtest_venue_config

        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)
        config, _launch = runner._create_venue_config(
            {"parameters": {"initial_balance": 10000.0}},
            None,
            timeframe="1m",
            has_detail_data=True,
        )
        runner.close()

        assert config.starting_balance_usdt == 10000
        assert create_backtest_venue_config(config).starting_balances == ["10000 USDT"]

    def test_venue_config_ignores_a_top_level_starting_balance_column(
        self, temp_db: Path, temp_logs: Path
    ) -> None:
        """R1: the dead ``run_config["starting_balance"]`` fallback is gone."""
        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)
        config, _launch = runner._create_venue_config(
            {"starting_balance": 50_000, "parameters": {}}, None, timeframe="1m"
        )
        runner.close()

        assert config.starting_balance_usdt == DEFAULT_STARTING_BALANCE_USDT == 1000

    def test_invalid_initial_balance_is_a_validation_runner_error(
        self, temp_db: Path, temp_logs: Path
    ) -> None:
        """R7: the shared BalanceConfigError becomes a ValidationRunnerError."""
        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)
        with pytest.raises(ValidationRunnerError) as excinfo:
            runner._create_venue_config(
                {"parameters": {"initial_balance": 1000.5}}, None, timeframe="1m"
            )
        runner.close()

        assert str(excinfo.value) == (
            "invalid initial_balance: initial_balance must be a whole number in "
            "[1, 1000000000], got 1000.5"
        )

    @pytest.mark.parametrize("bad", [0, -5, 1_000_000_001, "10000", True, None, math.nan, math.inf])
    def test_invalid_stored_initial_balance_fails_the_run(
        self,
        temp_db: Path,
        temp_logs: Path,
        sample_dsl_config: dict[str, object],
        bad: object,
    ) -> None:
        """R7: a bad stored value fails the run (status failed, no engine, no silent 1000)."""
        state = StateManager(temp_db)
        strategy_id = state.create_strategy(name="bad_balance", dsl_config=sample_dsl_config)
        run_id = state.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="validation",
            symbols=["BTCUSDT-PERP"],
            timeframe="5m",
            start_date="2025-01-01",
            end_date="2025-01-31",
            parameters={"initial_balance": bad},
        )
        state.close()

        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)
        with pytest.raises(ValidationRunnerError, match="invalid initial_balance"):
            runner.run(run_id=run_id)
        runner.close()

        state = StateManager(temp_db)
        run = state.get_backtest_run(run_id)
        state.close()
        assert run is not None
        assert run["status"] == "failed"
        assert str(run.get("error_message", "")).startswith("invalid initial_balance")

    def test_valid_initial_balance_reaches_the_stored_result(
        self,
        temp_db: Path,
        temp_logs: Path,
        sample_dsl_config: dict[str, object],
    ) -> None:
        """R14: validation is unchanged for a valid value — the run completes and stores.

        The fake derives the result's ``starting_balance`` from the venue config the
        runner built (as ``extract_results`` does), so the persisted
        ``backtest_results.starting_balance`` proves the venue carried ``10000.0`` and
        not the 1000 default. The DSL timeframe is non-sub-bar, so the venue is built on
        the ``_create_venue_config`` branch a 1000 mutant would hide in.
        """
        state = StateManager(temp_db)
        strategy_id = state.create_strategy(
            name="good_balance",
            dsl_config={**sample_dsl_config, "timeframe": "15m"},
        )
        run_id = state.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="validation",
            symbols=["BTCUSDT-PERP"],
            timeframe="15m",
            start_date="2025-01-01",
            end_date="2025-01-31",
            parameters={"initial_balance": 10000.0},
        )
        state.close()

        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)
        runner._run_backtest = lambda **kwargs: _make_mock_result(  # type: ignore[assignment]
            run_id=kwargs["run_id"],
            strategy_name=kwargs["strategy_name"],
            starting_balance=float(kwargs["venue_config"].starting_balance_usdt),
        )
        result = runner.run(run_id=run_id)
        runner.close()

        state = StateManager(temp_db)
        stored = state.get_backtest_result(run_id)
        run = state.get_backtest_run(run_id)
        state.close()
        assert stored is not None
        assert stored["sharpe_ratio"] == result.sharpe_ratio
        assert stored["starting_balance"] == 10000.0
        assert run is not None
        assert run["status"] == "completed"

    def test_stored_balance_is_the_venue_config_balance(
        self, temp_db: Path, temp_logs: Path
    ) -> None:
        """R14: ``backtest_results.starting_balance`` is the venue the run built (10000.0)."""
        from vibe_quant.validation.extraction import extract_results

        runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)
        venue_config, _launch = runner._create_venue_config(
            {"parameters": {"initial_balance": 10000.0}}, None, timeframe="1m"
        )
        runner.close()

        # bt_result=None: the venue stamp happens before any engine work.
        result = extract_results(1, "test_strategy", None, None, venue_config)  # type: ignore[arg-type]

        assert result.starting_balance == 10000.0

        runner.close()


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
        runner._run_backtest = lambda **kwargs: _make_mock_result(  # type: ignore[assignment]
            run_id=kwargs["run_id"], strategy_name=kwargs["strategy_name"]
        )
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


class TestLaunchErrorPrecedence:
    """Where each launch error is parsed, relative to the detail/clamp block (R26/R27a/R27b).

    The launch settings are parsed in two steps (rabmo.3 S5): liquidation/sizing/leverage
    in ``run`` before the detail check, balance/fee inside ``_create_venue_config`` after
    it. These characterisation tests pin today's precedence for both entry points.
    """

    @pytest.mark.parametrize(
        "parameters",
        [{"leverage": 0}, {"initial_balance": 0, "leverage": 0}],
        ids=["leverage-only", "balance-and-leverage"],
    )
    def test_bad_leverage_and_missing_detail_reports_leverage(
        self,
        tmp_path: Path,
        sample_dsl_config: dict[str, object],
        monkeypatch: pytest.MonkeyPatch,
        parameters: dict[str, object],
    ) -> None:
        """R27a: the leverage fails BEFORE the missing-1m-data check."""
        run_id = _launch_error_run(tmp_path, sample_dsl_config, parameters)
        _no_one_minute_data(monkeypatch, tmp_path)

        runner = _launch_runner(tmp_path)
        try:
            with pytest.raises(ValidationRunnerError) as excinfo:
                runner.run(run_id=run_id)
        finally:
            runner.close()

        assert str(excinfo.value) == "leverage must be a whole number in [1, 125], got 0"
        assert _stored_run(tmp_path, run_id)["status"] == "failed"

    @pytest.mark.parametrize(
        "parameters",
        [{"leverage": 0}, {"initial_balance": 0, "leverage": 0}],
        ids=["leverage-only", "balance-and-leverage"],
    )
    def test_walk_forward_bad_leverage_and_missing_detail_reports_leverage(
        self,
        tmp_path: Path,
        sample_dsl_config: dict[str, object],
        monkeypatch: pytest.MonkeyPatch,
        parameters: dict[str, object],
    ) -> None:
        """R27a for ``run_walk_forward``: same position, same text."""
        run_id = _launch_error_run(tmp_path, sample_dsl_config, parameters)
        _no_one_minute_data(monkeypatch, tmp_path)

        runner = _launch_runner(tmp_path)
        try:
            with pytest.raises(ValidationRunnerError) as excinfo:
                runner.run_walk_forward(run_id=run_id, train_days=7, test_days=7, step_days=7)
        finally:
            runner.close()

        assert str(excinfo.value) == "leverage must be a whole number in [1, 125], got 0"
        assert _stored_run(tmp_path, run_id)["status"] == "failed"

    @pytest.mark.parametrize(
        "parameters",
        [
            {"initial_balance": 0},
            {"fee_tier": "VIP99"},
            {"fee_tier": "VIP99", "initial_balance": 0},
        ],
        ids=["balance-only", "fee-only", "balance-and-fee"],
    )
    def test_bad_balance_and_missing_detail_reports_detail(
        self,
        tmp_path: Path,
        sample_dsl_config: dict[str, object],
        monkeypatch: pytest.MonkeyPatch,
        parameters: dict[str, object],
    ) -> None:
        """R27b: balance and fee are parsed AFTER the detail check, so it wins."""
        run_id = _launch_error_run(tmp_path, sample_dsl_config, parameters)
        _no_one_minute_data(monkeypatch, tmp_path)

        runner = _launch_runner(tmp_path)
        try:
            with pytest.raises(ValidationRunnerError) as excinfo:
                runner.run(run_id=run_id)
        finally:
            runner.close()

        assert str(excinfo.value).startswith(_DETAIL_MISSING)
        assert _stored_run(tmp_path, run_id)["status"] == "failed"

    @pytest.mark.parametrize(
        "parameters",
        [
            {"initial_balance": 0},
            {"fee_tier": "VIP99"},
            {"fee_tier": "VIP99", "initial_balance": 0},
        ],
        ids=["balance-only", "fee-only", "balance-and-fee"],
    )
    def test_walk_forward_bad_balance_and_missing_detail_reports_detail(
        self,
        tmp_path: Path,
        sample_dsl_config: dict[str, object],
        monkeypatch: pytest.MonkeyPatch,
        parameters: dict[str, object],
    ) -> None:
        """R27b for ``run_walk_forward``: the detail-coverage message wins there too."""
        run_id = _launch_error_run(tmp_path, sample_dsl_config, parameters)
        _no_one_minute_data(monkeypatch, tmp_path)

        runner = _launch_runner(tmp_path)
        try:
            with pytest.raises(ValidationRunnerError) as excinfo:
                runner.run_walk_forward(run_id=run_id, train_days=7, test_days=7, step_days=7)
        finally:
            runner.close()

        assert str(excinfo.value).startswith(_DETAIL_MISSING)
        assert _stored_run(tmp_path, run_id)["status"] == "failed"

    @pytest.mark.parametrize(
        ("parameters", "message"),
        [
            (
                {"initial_balance": 0},
                "invalid initial_balance: initial_balance must be a whole number in "
                "[1, 1000000000], got 0",
            ),
            ({"leverage": 0}, "leverage must be a whole number in [1, 125], got 0"),
            (
                {"liquidation_mode": "margin"},
                "liquidation_mode must be one of ('off', 'isolated', 'cross'), got 'margin'",
            ),
            (
                {"max_position_pct": -1},
                "max_position_pct must be a number in (0, 125], got -1",
            ),
            (
                {"fee_tier": "VIP99"},
                "invalid fee schedule parameters: unknown fee tier 'VIP99'; "
                "known tiers: ['VIP0']",
            ),
        ],
        ids=["balance", "leverage", "liquidation", "sizing", "fee"],
    )
    def test_each_launch_error_keeps_its_exact_text(
        self,
        tmp_path: Path,
        sample_dsl_config: dict[str, object],
        monkeypatch: pytest.MonkeyPatch,
        parameters: dict[str, object],
        message: str,
    ) -> None:
        """R26: 1m data present, so the field parser (not the coverage check) decides."""
        run_id = _launch_error_run(tmp_path, sample_dsl_config, parameters)
        _one_minute_data(monkeypatch, tmp_path)

        runner = _launch_runner(tmp_path)
        try:
            with pytest.raises(ValidationRunnerError) as excinfo:
                runner.run(run_id=run_id)
        finally:
            runner.close()

        assert str(excinfo.value) == message
        stored = _stored_run(tmp_path, run_id)
        assert stored["status"] == "failed"
        assert stored["error_message"] == message

    def test_bad_fee_tier_fails_run_with_prefixed_message(
        self,
        tmp_path: Path,
        sample_dsl_config: dict[str, object],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """R26: the fee wrapper is byte-identical and fails the run before ``running``."""
        run_id = _launch_error_run(tmp_path, sample_dsl_config, {"fee_tier": "VIP99"})
        _one_minute_data(monkeypatch, tmp_path)
        statuses: list[str] = []
        original = StateManager.update_backtest_run_status

        def recording(
            self: StateManager,
            rid: int,
            status: str,
            pid: int | None = None,
            error_message: str | None = None,
        ) -> None:
            statuses.append(status)
            original(self, rid, status, pid=pid, error_message=error_message)

        monkeypatch.setattr(StateManager, "update_backtest_run_status", recording)

        runner = _launch_runner(tmp_path)
        try:
            with pytest.raises(ValidationRunnerError) as excinfo:
                runner.run(run_id=run_id)
        finally:
            runner.close()

        expected = (
            "invalid fee schedule parameters: unknown fee tier 'VIP99'; "
            "known tiers: ['VIP0']"
        )
        assert str(excinfo.value) == expected
        assert "running" not in statuses
        stored = _stored_run(tmp_path, run_id)
        assert stored["status"] == "failed"
        assert stored["error_message"] == expected


_RUN_START = "2025-01-01"
_RUN_END = "2025-02-01"

#: The clamp's missing-detail text (runner.clamp_run_window); the tail is not pinned.
_DETAIL_MISSING = (
    "No 1m bars for BTCUSDT in the catalog. Validation needs 1m data for "
    "intrabar fills/latency"
)

_COVERED = (datetime(2024, 1, 1, tzinfo=UTC), datetime(2026, 3, 17, tzinfo=UTC))


def _launch_error_run(
    tmp_path: Path, sample_dsl_config: dict[str, object], parameters: dict[str, object]
) -> int:
    """A 4h BTCUSDT validation run over a window, with ``parameters`` stored."""
    state = StateManager(tmp_path / "launch.db")
    try:
        strategy_id = state.create_strategy(
            name="launch_error_probe",
            dsl_config={**sample_dsl_config, "timeframe": "4h"},
        )
        return state.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="validation",
            symbols=["BTCUSDT"],
            timeframe="4h",
            start_date=_RUN_START,
            end_date=_RUN_END,
            parameters=parameters,
        )
    finally:
        state.close()


def _launch_runner(tmp_path: Path) -> ValidationRunner:
    return ValidationRunner(db_path=tmp_path / "launch.db", logs_path=tmp_path / "logs")


def _stored_run(tmp_path: Path, run_id: int) -> dict[str, object]:
    state = StateManager(tmp_path / "launch.db")
    try:
        row = state.get_backtest_run(run_id)
    finally:
        state.close()
    assert row is not None
    return dict(row)


def _real_clamp_and_catalog(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Undo the autouse no-op clamp and point the catalog at ``tmp_path``."""
    monkeypatch.setattr(ValidationRunner, "_clamp_run_window", _REAL_CLAMP_RUN_WINDOW)
    monkeypatch.setattr(catalog_mod, "DEFAULT_CATALOG_PATH", tmp_path / "catalog")


def _no_one_minute_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The catalog has no 1m bars: the clamp fails with its coverage message."""
    _real_clamp_and_catalog(monkeypatch, tmp_path)

    def no_bars(
        self: CatalogManager, symbol: str, interval: str
    ) -> tuple[datetime, datetime] | None:
        return None

    monkeypatch.setattr(CatalogManager, "get_bar_date_range", no_bars)


def _one_minute_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """1m bars cover the run window, so the detail check passes and the fields decide."""
    _real_clamp_and_catalog(monkeypatch, tmp_path)

    def covered(
        self: CatalogManager, symbol: str, interval: str
    ) -> tuple[datetime, datetime] | None:
        return _COVERED if interval == "1m" else None

    monkeypatch.setattr(CatalogManager, "get_bar_date_range", covered)
