"""Tests for paper trading module."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import time
from decimal import Decimal
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from vibe_quant.paper.config import (
    BinanceTestnetConfig,
    ConfigurationError,
    PaperTradingConfig,
    RiskModuleConfig,
    SizingModuleConfig,
    create_trading_node_config,
)
from vibe_quant.paper.node import (
    HaltReason,
    NodeState,
    NodeStatus,
    PaperTradingNode,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Create temp database with test strategy."""
    db_file = tmp_path / "test_paper.db"
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
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
        "sweep": {},
    }

    conn.execute(
        "INSERT INTO strategies (name, description, dsl_config) VALUES (?, ?, ?)",
        ("test_strategy", "A test strategy", json.dumps(dsl_config)),
    )
    conn.commit()
    conn.close()
    return db_file


class TestBinanceTestnetConfig:
    """Tests for BinanceTestnetConfig."""

    def test_create_directly(self):
        """Can create config directly."""
        config = BinanceTestnetConfig(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True,
        )

        assert config.api_key == "test_key"
        assert config.api_secret == "test_secret"
        assert config.testnet is True
        assert config.account_type == "USDT_FUTURES"

    def test_from_env_missing_key(self):
        """Raises error if API key missing."""
        with patch.dict(os.environ, {}, clear=True), pytest.raises(
            ConfigurationError, match="BINANCE_API_KEY"
        ):
            BinanceTestnetConfig.from_env()

    def test_from_env_missing_secret(self):
        """Raises error if API secret missing."""
        with patch.dict(os.environ, {"BINANCE_API_KEY": "key"}, clear=True), pytest.raises(
            ConfigurationError, match="BINANCE_API_SECRET"
        ):
            BinanceTestnetConfig.from_env()

    def test_from_env_success(self):
        """Creates config from env vars."""
        env = {
            "BINANCE_API_KEY": "test_key",
            "BINANCE_API_SECRET": "test_secret",
        }
        with patch.dict(os.environ, env, clear=True):
            config = BinanceTestnetConfig.from_env(testnet=True)

        assert config.api_key == "test_key"
        assert config.api_secret == "test_secret"
        assert config.testnet is True


class TestSizingModuleConfig:
    """Tests for SizingModuleConfig."""

    def test_default_values(self):
        """Default values are sensible."""
        config = SizingModuleConfig()

        assert config.method == "fixed_fractional"
        assert config.max_leverage == Decimal("20")
        assert config.max_position_pct == Decimal("0.5")
        assert config.risk_per_trade == Decimal("0.02")

    def test_custom_values(self):
        """Can set custom values."""
        config = SizingModuleConfig(
            method="kelly",
            max_leverage=Decimal("10"),
            risk_per_trade=Decimal("0.01"),
        )

        assert config.method == "kelly"
        assert config.max_leverage == Decimal("10")
        assert config.risk_per_trade == Decimal("0.01")


class TestRiskModuleConfig:
    """Tests for RiskModuleConfig."""

    def test_default_values(self):
        """Default values are sensible."""
        config = RiskModuleConfig()

        assert config.max_drawdown_pct == Decimal("0.15")
        assert config.max_daily_loss_pct == Decimal("0.02")
        assert config.max_consecutive_losses == 10
        assert config.max_position_count == 5

    def test_custom_values(self):
        """Can set custom values."""
        config = RiskModuleConfig(
            max_drawdown_pct=Decimal("0.10"),
            max_consecutive_losses=5,
        )

        assert config.max_drawdown_pct == Decimal("0.10")
        assert config.max_consecutive_losses == 5


class TestPaperTradingConfig:
    """Tests for PaperTradingConfig."""

    def test_validate_missing_trader_id(self):
        """Validates trader_id required."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=1,
        )

        errors = config.validate()
        assert any("trader_id" in e for e in errors)

    def test_validate_missing_symbols(self):
        """Validates symbols required."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=[],
            strategy_id=1,
        )

        errors = config.validate()
        assert any("symbol" in e.lower() for e in errors)

    def test_validate_missing_strategy_id(self):
        """Validates strategy_id required."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=None,
        )

        errors = config.validate()
        assert any("strategy_id" in e for e in errors)

    def test_validate_invalid_risk_per_trade(self):
        """Validates risk_per_trade bounds."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=1,
            sizing=SizingModuleConfig(risk_per_trade=Decimal("0.6")),
        )

        errors = config.validate()
        assert any("risk_per_trade" in e for e in errors)

    def test_validate_success(self):
        """Valid config passes validation."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=1,
        )

        errors = config.validate()
        assert len(errors) == 0

    def test_create_from_env(self):
        """Can create from env vars."""
        env = {
            "BINANCE_API_KEY": "test_key",
            "BINANCE_API_SECRET": "test_secret",
        }
        with patch.dict(os.environ, env, clear=True):
            config = PaperTradingConfig.create(
                trader_id="PAPER-001",
                symbols=["BTCUSDT"],
                strategy_id=1,
            )

        assert config.trader_id == "PAPER-001"
        assert config.symbols == ["BTCUSDT"]
        assert config.strategy_id == 1
        assert config.binance.testnet is True


class TestCreateTradingNodeConfig:
    """Tests for create_trading_node_config function."""

    def test_creates_config_dict(self):
        """Creates complete config dictionary."""
        binance = BinanceTestnetConfig("key", "secret", testnet=True)
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT", "ETHUSDT"],
            strategy_id=42,
        )

        node_config = create_trading_node_config(config)

        assert node_config["trader_id"] == "PAPER-001"
        assert "BINANCE" in node_config["data_clients"]
        assert "BINANCE" in node_config["exec_clients"]
        assert node_config["data_clients"]["BINANCE"]["testnet"] is True
        assert node_config["symbols"] == ["BTCUSDT", "ETHUSDT"]
        assert node_config["strategy_id"] == 42

    def test_includes_sizing_config(self):
        """Includes sizing configuration."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=1,
            sizing=SizingModuleConfig(method="kelly", risk_per_trade=Decimal("0.01")),
        )

        node_config = create_trading_node_config(config)

        assert node_config["sizing"]["method"] == "kelly"
        assert node_config["sizing"]["risk_per_trade"] == "0.01"

    def test_includes_risk_config(self):
        """Includes risk configuration."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=1,
            risk=RiskModuleConfig(max_drawdown_pct=Decimal("0.10")),
        )

        node_config = create_trading_node_config(config)

        assert node_config["risk"]["max_drawdown_pct"] == "0.10"


class TestNodeStatus:
    """Tests for NodeStatus."""

    def test_default_state(self):
        """Default state is initializing."""
        status = NodeStatus()

        assert status.state == NodeState.INITIALIZING
        assert status.halt_reason is None
        assert status.positions == 0

    def test_to_dict(self):
        """Can serialize to dict."""
        status = NodeStatus(
            state=NodeState.RUNNING,
            positions=2,
            daily_pnl=100.0,
        )

        data = status.to_dict()

        assert data["state"] == "running"
        assert data["positions"] == 2
        assert data["daily_pnl"] == 100.0


class TestPaperTradingNode:
    """Tests for PaperTradingNode."""

    def test_init_validates_config(self):
        """Init validates configuration."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="",  # Invalid
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=1,
        )

        with pytest.raises(ConfigurationError, match="trader_id"):
            PaperTradingNode(config)

    def test_init_success(self, db_path: Path):
        """Can create node with valid config."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=1,
            db_path=db_path,
        )

        node = PaperTradingNode(config)

        assert node.config == config
        assert node.status.state == NodeState.INITIALIZING

    def test_load_strategy(self, db_path: Path):
        """Can load strategy from database."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=1,
            db_path=db_path,
        )

        node = PaperTradingNode(config)
        try:
            strategy = node._load_strategy()

            assert strategy.name == "test_strategy"
            assert strategy.timeframe == "1h"
        finally:
            node._state_manager.close()

    def test_load_strategy_not_found(self, db_path: Path):
        """Raises error if strategy not found."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=999,  # Non-existent
            db_path=db_path,
        )

        node = PaperTradingNode(config)
        try:
            with pytest.raises(ConfigurationError, match="not found"):
                node._load_strategy()
        finally:
            node._state_manager.close()

    def test_halt_changes_state(self, db_path: Path):
        """Halt changes node state."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=1,
            db_path=db_path,
        )

        node = PaperTradingNode(config)
        node.halt(HaltReason.MAX_DRAWDOWN, "Hit 15% drawdown")

        assert node.status.state == NodeState.HALTED
        assert node.status.halt_reason == HaltReason.MAX_DRAWDOWN
        assert node.status.error_message == "Hit 15% drawdown"
        assert not node._shutdown_event.is_set()

    def test_halt_cancels_open_orders(self, db_path: Path):
        """Halt cancels currently open orders via loaded runtime strategies."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=1,
            db_path=db_path,
        )

        class _Strategy:
            def __init__(self, strategy_id: str) -> None:
                self.id = strategy_id
                self.cancelled: list[object] = []

            def cancel_order(self, order: object) -> None:
                self.cancelled.append(order)

        class _Trader:
            def __init__(self, strategies: list[_Strategy]) -> None:
                self._strategies = strategies

            def strategies(self) -> list[_Strategy]:
                return self._strategies

        class _Cache:
            def __init__(self, orders_by_strategy: dict[str, list[object]]) -> None:
                self._orders_by_strategy = orders_by_strategy

            def orders_open(self, strategy_id: object | None = None) -> list[object]:
                if strategy_id is None:
                    return []
                return list(self._orders_by_strategy.get(str(strategy_id), []))

        class _TradingNode:
            def __init__(self, trader: _Trader, cache: _Cache) -> None:
                self.trader = trader
                self.cache = cache

        strategy_a = _Strategy("S-A")
        strategy_b = _Strategy("S-B")
        order_1 = object()
        order_2 = object()
        order_3 = object()

        node = PaperTradingNode(config)
        node._trading_node = _TradingNode(
            trader=_Trader([strategy_a, strategy_b]),
            cache=_Cache({"S-A": [order_1, order_2], "S-B": [order_3]}),
        )

        node.halt(HaltReason.ERROR, "Risk threshold breached")

        assert strategy_a.cancelled == [order_1, order_2]
        assert strategy_b.cancelled == [order_3]

    def test_halt_ignores_cancel_failures(self, db_path: Path):
        """Halt continues even if individual order cancellations fail."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=1,
            db_path=db_path,
        )

        class _Strategy:
            def __init__(self) -> None:
                self.id = "S-ERR"
                self.calls = 0

            def cancel_order(self, order: object) -> None:
                self.calls += 1
                raise RuntimeError("cancel failed")

        class _TradingNode:
            def __init__(self) -> None:
                strategy = _Strategy()
                self.strategy = strategy
                self.trader = type("Trader", (), {"strategies": lambda _self: [strategy]})()
                self.cache = type(
                    "Cache",
                    (),
                    {"orders_open": lambda _self, strategy_id=None: [object()]},
                )()

        node = PaperTradingNode(config)
        fake_runtime = _TradingNode()
        node._trading_node = fake_runtime

        node.halt(HaltReason.ERROR, "halt")

        assert fake_runtime.strategy.calls == 1
        assert node.status.state == NodeState.HALTED

    def test_pause_resume(self, db_path: Path):
        """Can pause and resume node."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=1,
            db_path=db_path,
        )

        node = PaperTradingNode(config)

        # Set to running first
        node._status.state = NodeState.RUNNING

        node.pause()
        assert node.status.state == NodeState.PAUSED

        node.resume()
        assert node.status.state == NodeState.RUNNING

    @pytest.mark.asyncio
    async def test_initialize_creates_live_node_object(self, db_path: Path, monkeypatch: pytest.MonkeyPatch):
        """_initialize should attach a lifecycle node object, not a dict placeholder."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=1,
            db_path=db_path,
        )

        class _FakeTradingNode:
            def __init__(self) -> None:
                self.stop_calls = 0
                self.dispose_calls = 0

            def run(self, raise_exception: bool = False) -> None:
                return

            def stop(self) -> None:
                self.stop_calls += 1

            def dispose(self) -> None:
                self.dispose_calls += 1

        fake_node = _FakeTradingNode()
        monkeypatch.setattr(
            PaperTradingNode,
            "_create_live_trading_node",
            lambda _self: fake_node,
        )

        node = PaperTradingNode(config)
        await node._initialize()
        try:
            assert node._trading_node is fake_node
            assert not isinstance(node._trading_node, dict)
        finally:
            await node._shutdown()

    @pytest.mark.asyncio
    async def test_initialize_saves_initial_checkpoint(self, db_path: Path, monkeypatch: pytest.MonkeyPatch):
        """_initialize should create persistence and store initial checkpoint."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=1,
            db_path=db_path,
        )

        class _FakeTradingNode:
            def run(self, raise_exception: bool = False) -> None:
                return

            def stop(self) -> None:
                return

            def dispose(self) -> None:
                return

        monkeypatch.setattr(
            PaperTradingNode,
            "_create_live_trading_node",
            lambda _self: _FakeTradingNode(),
        )

        node = PaperTradingNode(config)
        await node._initialize()
        try:
            assert node._persistence is not None
            checkpoint = node._persistence.load_latest_checkpoint()
            assert checkpoint is not None
            assert checkpoint.trader_id == "PAPER-001"
            assert checkpoint.node_status["state"] == NodeState.INITIALIZING.value
        finally:
            await node._shutdown()

    @pytest.mark.asyncio
    async def test_run_loop_uses_trading_node_lifecycle(self, db_path: Path):
        """_run_loop should call node.run and stop the node when shutdown is signaled."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=1,
            db_path=db_path,
        )
        node = PaperTradingNode(config)

        class _BlockingTradingNode:
            def __init__(self) -> None:
                self.run_started = threading.Event()
                self.stop_called = threading.Event()

            def run(self, raise_exception: bool = False) -> None:
                self.run_started.set()
                while not self.stop_called.is_set():
                    time.sleep(0.01)

            def stop(self) -> None:
                self.stop_called.set()

            def dispose(self) -> None:
                return

        fake_node = _BlockingTradingNode()
        node._trading_node = fake_node

        run_task = asyncio.create_task(node._run_loop())
        try:
            for _ in range(100):
                if fake_node.run_started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert fake_node.run_started.is_set()
            node._shutdown_event.set()
            await asyncio.wait_for(run_task, timeout=2.0)
            assert fake_node.stop_called.is_set()
        finally:
            await node._shutdown()

    @pytest.mark.asyncio
    async def test_halt_keeps_run_loop_alive_until_explicit_shutdown(self, db_path: Path):
        """halt() should not terminate _run_loop without explicit shutdown signal."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=1,
            db_path=db_path,
        )
        node = PaperTradingNode(config)

        class _BlockingTradingNode:
            def __init__(self) -> None:
                self.run_started = threading.Event()
                self.stop_called = threading.Event()

            def run(self, raise_exception: bool = False) -> None:
                self.run_started.set()
                while not self.stop_called.is_set():
                    time.sleep(0.01)

            def stop(self) -> None:
                self.stop_called.set()

            def dispose(self) -> None:
                return

        fake_node = _BlockingTradingNode()
        node._trading_node = fake_node

        run_task = asyncio.create_task(node._run_loop())
        try:
            for _ in range(100):
                if fake_node.run_started.is_set():
                    break
                await asyncio.sleep(0.01)

            node.halt(HaltReason.ERROR, "risk halt")
            await asyncio.sleep(0.05)

            assert node.status.state == NodeState.HALTED
            assert not run_task.done()

            node._shutdown_event.set()
            await asyncio.wait_for(run_task, timeout=2.0)
            assert fake_node.stop_called.is_set()
        finally:
            await node._shutdown()

    @pytest.mark.asyncio
    async def test_run_loop_starts_and_shutdown_stops_persistence(self, db_path: Path):
        """Persistence lifecycle should run with node lifecycle."""
        binance = BinanceTestnetConfig("key", "secret")
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=binance,
            symbols=["BTCUSDT"],
            strategy_id=1,
            db_path=db_path,
        )
        node = PaperTradingNode(config)

        class _BlockingTradingNode:
            def __init__(self) -> None:
                self.run_started = threading.Event()
                self.stop_called = threading.Event()

            def run(self, raise_exception: bool = False) -> None:
                self.run_started.set()
                while not self.stop_called.is_set():
                    time.sleep(0.01)

            def stop(self) -> None:
                self.stop_called.set()

            def dispose(self) -> None:
                return

        class _FakePersistence:
            def __init__(self) -> None:
                self.start_calls = 0
                self.stop_calls = 0
                self.save_calls = 0
                self.close_calls = 0
                self.callback = None

            async def start_periodic_checkpointing(self, state_callback):
                self.start_calls += 1
                self.callback = state_callback

            async def stop_periodic_checkpointing(self):
                self.stop_calls += 1

            def save_checkpoint(self, checkpoint):
                self.save_calls += 1
                return 1

            def close(self):
                self.close_calls += 1

        fake_node = _BlockingTradingNode()
        fake_persistence = _FakePersistence()
        node._trading_node = fake_node
        node._persistence = fake_persistence

        run_task = asyncio.create_task(node._run_loop())
        try:
            for _ in range(100):
                if fake_node.run_started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert fake_node.run_started.is_set()
            assert fake_persistence.start_calls == 1
            assert callable(fake_persistence.callback)

            node._shutdown_event.set()
            await asyncio.wait_for(run_task, timeout=2.0)
        finally:
            await node._shutdown()

        assert fake_persistence.stop_calls == 1
        assert fake_persistence.save_calls >= 1
        assert fake_persistence.close_calls == 1


class TestNodeStateEnum:
    """Tests for NodeState enum."""

    def test_all_states_defined(self):
        """All expected states are defined."""
        assert NodeState.INITIALIZING.value == "initializing"
        assert NodeState.RUNNING.value == "running"
        assert NodeState.PAUSED.value == "paused"
        assert NodeState.HALTED.value == "halted"
        assert NodeState.STOPPED.value == "stopped"
        assert NodeState.ERROR.value == "error"


class TestHaltReasonEnum:
    """Tests for HaltReason enum."""

    def test_all_reasons_defined(self):
        """All expected halt reasons are defined."""
        assert HaltReason.MAX_DRAWDOWN.value == "max_drawdown"
        assert HaltReason.MAX_DAILY_LOSS.value == "max_daily_loss"
        assert HaltReason.MAX_CONSECUTIVE_LOSSES.value == "max_consecutive_losses"
        assert HaltReason.MANUAL.value == "manual"
        assert HaltReason.ERROR.value == "error"
        assert HaltReason.SIGNAL.value == "signal"
