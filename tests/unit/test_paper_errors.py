"""Tests for paper trading error handling."""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from vibe_quant.paper.config import BinanceTestnetConfig, PaperTradingConfig
from vibe_quant.paper.errors import (
    ErrorCategory,
    ErrorContext,
    ErrorHandler,
    RetryConfig,
    classify_error,
)
from vibe_quant.paper.node import HaltReason, NodeState, PaperTradingNode


class TestClassifyError:
    """Tests for classify_error function."""

    def test_timeout_is_transient(self):
        """Timeout errors are transient."""
        err = TimeoutError("connection timed out")
        assert classify_error(err) == ErrorCategory.TRANSIENT

    def test_connection_error_is_transient(self):
        """Connection errors are transient."""
        err = ConnectionError("connection reset by peer")
        assert classify_error(err) == ErrorCategory.TRANSIENT

    def test_rate_limit_is_transient(self):
        """Rate limit errors are transient."""
        err = Exception("rate limit exceeded")
        assert classify_error(err) == ErrorCategory.TRANSIENT

    def test_socket_error_is_transient(self):
        """Socket errors are transient."""
        err = OSError("socket error occurred")
        assert classify_error(err) == ErrorCategory.TRANSIENT

    def test_service_unavailable_is_transient(self):
        """Service unavailable is transient."""
        err = Exception("service temporarily unavailable")
        assert classify_error(err) == ErrorCategory.TRANSIENT

    def test_auth_failure_is_fatal(self):
        """Auth failures are fatal."""
        err = Exception("authentication failed")
        assert classify_error(err) == ErrorCategory.FATAL

    def test_invalid_api_key_is_fatal(self):
        """Invalid API key is fatal."""
        err = Exception("Invalid API key")
        assert classify_error(err) == ErrorCategory.FATAL

    def test_insufficient_balance_is_fatal(self):
        """Insufficient balance is fatal."""
        err = Exception("insufficient balance")
        assert classify_error(err) == ErrorCategory.FATAL

    def test_liquidation_is_fatal(self):
        """Liquidation is fatal."""
        err = Exception("position liquidation triggered")
        assert classify_error(err) == ErrorCategory.FATAL

    def test_unknown_error_is_strategy(self):
        """Unknown errors default to strategy category."""
        err = Exception("some unknown error")
        assert classify_error(err) == ErrorCategory.STRATEGY

    def test_value_error_is_strategy(self):
        """Generic ValueError is strategy error."""
        err = ValueError("invalid value")
        assert classify_error(err) == ErrorCategory.STRATEGY


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_values(self):
        """Default values are reasonable."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay_ms == 1000
        assert config.max_delay_ms == 30000

    def test_exponential_backoff(self):
        """Delay increases exponentially."""
        config = RetryConfig(base_delay_ms=100, exponential_base=2.0)

        assert config.get_delay_ms(0) == 100
        assert config.get_delay_ms(1) == 200
        assert config.get_delay_ms(2) == 400
        assert config.get_delay_ms(3) == 800

    def test_delay_capped_at_max(self):
        """Delay is capped at max_delay_ms."""
        config = RetryConfig(base_delay_ms=1000, max_delay_ms=2000)

        assert config.get_delay_ms(0) == 1000
        assert config.get_delay_ms(1) == 2000
        assert config.get_delay_ms(10) == 2000  # capped


class TestErrorContext:
    """Tests for ErrorContext dataclass."""

    def test_basic_creation(self):
        """Can create error context."""
        err = ValueError("test error")
        ctx = ErrorContext(
            error=err,
            category=ErrorCategory.STRATEGY,
            operation="test_op",
            symbol="BTCUSDT",
        )

        assert ctx.error is err
        assert ctx.category == ErrorCategory.STRATEGY
        assert ctx.operation == "test_op"
        assert ctx.symbol == "BTCUSDT"
        assert ctx.retry_count == 0

    def test_timestamp_auto_set(self):
        """Timestamp is auto-set."""
        err = ValueError("test")
        ctx = ErrorContext(error=err, category=ErrorCategory.TRANSIENT)

        assert ctx.timestamp is not None


class TestErrorHandler:
    """Tests for ErrorHandler class."""

    def test_transient_error_logged(self, caplog):
        """Transient errors are logged with retry info."""
        handler = ErrorHandler()
        err = TimeoutError("connection timeout")

        ctx = handler.handle_error(err, operation="connect")

        assert ctx.category == ErrorCategory.TRANSIENT
        assert ctx.retry_count == 1
        assert "Transient error" in caplog.text

    def test_transient_increments_count(self):
        """Transient errors increment retry count."""
        handler = ErrorHandler()
        err = TimeoutError("timeout")

        ctx1 = handler.handle_error(err, operation="connect")
        ctx2 = handler.handle_error(err, operation="connect")

        assert ctx1.retry_count == 1
        assert ctx2.retry_count == 2

    def test_transient_escalates_to_fatal(self):
        """Transient errors escalate to fatal after max retries."""
        handler = ErrorHandler(
            retry_config=RetryConfig(max_retries=2),
            on_halt=MagicMock(),
        )
        err = TimeoutError("timeout")

        ctx1 = handler.handle_error(err, operation="connect")
        assert ctx1.category == ErrorCategory.TRANSIENT

        ctx2 = handler.handle_error(err, operation="connect")
        assert ctx2.category == ErrorCategory.FATAL  # escalated
        assert handler._on_halt.called

    def test_fatal_error_triggers_halt(self):
        """Fatal errors trigger halt callback."""
        on_halt = MagicMock()
        handler = ErrorHandler(on_halt=on_halt)
        err = Exception("authentication failed")

        handler.handle_error(err, operation="auth")

        on_halt.assert_called_once()
        reason, message = on_halt.call_args[0]
        assert "fatal_error" in reason

    def test_fatal_error_triggers_alert(self):
        """Fatal errors trigger alert callback."""
        on_alert = MagicMock()
        handler = ErrorHandler(on_alert=on_alert)
        err = Exception("insufficient balance")

        handler.handle_error(err, operation="order")

        on_alert.assert_called_once()
        alert_type, context = on_alert.call_args[0]
        assert alert_type == "fatal_error"
        assert context.category == ErrorCategory.FATAL

    def test_strategy_error_halts_no_alert(self):
        """Strategy errors halt but don't alert."""
        on_halt = MagicMock()
        on_alert = MagicMock()
        handler = ErrorHandler(on_halt=on_halt, on_alert=on_alert)
        err = ValueError("invalid parameter")

        handler.handle_error(err, operation="validate")

        on_halt.assert_called_once()
        on_alert.assert_not_called()

    def test_reset_retry_count(self):
        """Can reset retry count after success."""
        handler = ErrorHandler()
        err = TimeoutError("timeout")

        handler.handle_error(err, operation="connect")
        handler.handle_error(err, operation="connect")

        handler.reset_retry_count(operation="connect")

        ctx = handler.handle_error(err, operation="connect")
        assert ctx.retry_count == 1  # Reset

    def test_should_retry(self):
        """should_retry returns correct value."""
        handler = ErrorHandler(retry_config=RetryConfig(max_retries=3))
        err = TimeoutError("timeout")

        ctx = handler.handle_error(err, operation="connect")
        assert handler.should_retry(ctx) is True

        # Exhaust retries
        handler.handle_error(err, operation="connect")
        ctx = handler.handle_error(err, operation="connect")
        assert handler.should_retry(ctx) is False

    def test_get_retry_delay_ms(self):
        """get_retry_delay_ms returns correct delay."""
        handler = ErrorHandler(retry_config=RetryConfig(base_delay_ms=100))
        ctx = ErrorContext(
            error=Exception("test"),
            category=ErrorCategory.TRANSIENT,
            retry_count=2,
        )

        delay = handler.get_retry_delay_ms(ctx)
        # retry_count=2 means second attempt, so delay = 100 * 2^(2-1) = 200
        assert delay == 200


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Create temp database with test strategy."""
    db_file = tmp_path / "test_errors.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("PRAGMA journal_mode=WAL")
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
    """)
    dsl_config = {
        "name": "test_strategy",
        "description": "Test",
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
        "sweep": {},
    }
    conn.execute(
        "INSERT INTO strategies (name, description, dsl_config) VALUES (?, ?, ?)",
        ("test_strategy", "Test", json.dumps(dsl_config)),
    )
    conn.commit()
    conn.close()
    return db_file


class TestNodeErrorIntegration:
    """Tests for PaperTradingNode error handling integration."""

    def test_handle_error_classifies_and_handles(self, db_path: Path):
        """Node.handle_error classifies and handles errors."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=1,
            db_path=db_path,
        )
        node = PaperTradingNode(config)

        err = TimeoutError("timeout")
        ctx = node.handle_error(err, operation="connect")

        assert ctx.category == ErrorCategory.TRANSIENT

    def test_fatal_error_halts_node(self, db_path: Path):
        """Fatal error halts the node."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=1,
            db_path=db_path,
        )
        node = PaperTradingNode(config)
        node._status.state = NodeState.RUNNING

        err = Exception("authentication failed")
        node.handle_error(err, operation="auth")

        assert node.status.state == NodeState.HALTED
        assert node.status.halt_reason == HaltReason.ERROR

    def test_resume_from_halt_after_error(self, db_path: Path):
        """Can resume from halt after error."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=1,
            db_path=db_path,
        )
        node = PaperTradingNode(config)
        node._status.state = NodeState.RUNNING

        # Trigger error halt
        err = Exception("authentication failed")
        node.handle_error(err, operation="auth")
        assert node.status.state == NodeState.HALTED

        # Resume
        result = node.resume_from_halt()
        assert result is True
        assert node.status.state == NodeState.RUNNING
        assert node.status.halt_reason is None

    def test_cannot_resume_from_drawdown_halt(self, db_path: Path):
        """Cannot resume from drawdown halt (not error halt)."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=1,
            db_path=db_path,
        )
        node = PaperTradingNode(config)

        # Halt due to drawdown
        node.halt(HaltReason.MAX_DRAWDOWN, "Hit 15% drawdown")
        assert node.status.state == NodeState.HALTED

        # Cannot resume
        result = node.resume_from_halt()
        assert result is False
        assert node.status.state == NodeState.HALTED

    def test_cannot_resume_when_not_halted(self, db_path: Path):
        """Cannot resume when not halted."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=1,
            db_path=db_path,
        )
        node = PaperTradingNode(config)
        node._status.state = NodeState.RUNNING

        result = node.resume_from_halt()
        assert result is False
        assert node.status.state == NodeState.RUNNING


class TestStateTransitions:
    """Tests for strategy state machine transitions."""

    def test_running_to_halted_via_error(self, db_path: Path):
        """RUNNING -> HALTED via error."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=1,
            db_path=db_path,
        )
        node = PaperTradingNode(config)
        node._status.state = NodeState.RUNNING

        err = Exception("insufficient balance")
        node.handle_error(err, operation="order")

        assert node.status.state == NodeState.HALTED

    def test_halted_to_running_via_resume(self, db_path: Path):
        """HALTED -> RUNNING via resume_from_halt."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=1,
            db_path=db_path,
        )
        node = PaperTradingNode(config)
        node._status.state = NodeState.RUNNING
        node._status.state = NodeState.HALTED
        node._status.halt_reason = HaltReason.ERROR
        node._previous_state = NodeState.RUNNING

        result = node.resume_from_halt()

        assert result is True
        assert node.status.state == NodeState.RUNNING

    def test_paused_to_halted_via_error(self, db_path: Path):
        """PAUSED -> HALTED via error, preserves previous state."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=1,
            db_path=db_path,
        )
        node = PaperTradingNode(config)
        node._status.state = NodeState.PAUSED

        err = Exception("authentication failed")
        node.handle_error(err, operation="reconnect")

        assert node.status.state == NodeState.HALTED
        assert node._previous_state == NodeState.PAUSED

    def test_halted_resumes_to_previous_paused_state(self, db_path: Path):
        """HALTED -> PAUSED when previous state was PAUSED."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=1,
            db_path=db_path,
        )
        node = PaperTradingNode(config)
        node._status.state = NodeState.PAUSED

        # Halt via error
        err = Exception("authentication failed")
        node.handle_error(err, operation="reconnect")
        assert node.status.state == NodeState.HALTED

        # Resume back to paused
        result = node.resume_from_halt()
        assert result is True
        assert node.status.state == NodeState.PAUSED
