"""Tests for paper trading CLI module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vibe_quant.paper.cli import (
    _decimal_from_str,
    load_config_from_json,
    run_with_config,
    save_config_to_json,
)
from vibe_quant.paper.config import (
    BinanceTestnetConfig,
    ConfigurationError,
    PaperTradingConfig,
    RiskModuleConfig,
    SizingModuleConfig,
)


class TestDecimalFromStr:
    """Tests for _decimal_from_str helper."""

    def test_from_string(self) -> None:
        """Convert string to Decimal."""
        from decimal import Decimal

        result = _decimal_from_str("10.5")
        assert result == Decimal("10.5")

    def test_from_int(self) -> None:
        """Convert int to Decimal."""
        from decimal import Decimal

        result = _decimal_from_str(10)
        assert result == Decimal("10")

    def test_from_float(self) -> None:
        """Convert float to Decimal."""
        from decimal import Decimal

        result = _decimal_from_str(10.5)
        assert result == Decimal("10.5")


class TestLoadConfigFromJson:
    """Tests for load_config_from_json."""

    def test_load_minimal_config(self, tmp_path: Path) -> None:
        """Load minimal valid config."""
        config_data = {
            "trader_id": "PAPER-001",
            "binance": {
                "api_key": "test_key",
                "api_secret": "test_secret",
                "testnet": True,
            },
            "symbols": ["BTCUSDT"],
            "strategy_id": 1,
        }
        config_path = tmp_path / "config.json"
        with config_path.open("w") as f:
            json.dump(config_data, f)

        config = load_config_from_json(config_path)

        assert config.trader_id == "PAPER-001"
        assert config.binance.api_key == "test_key"
        assert config.binance.api_secret == "test_secret"
        assert config.binance.testnet is True
        assert config.symbols == ["BTCUSDT"]
        assert config.strategy_id == 1

    def test_load_full_config(self, tmp_path: Path) -> None:
        """Load config with all fields."""
        config_data = {
            "trader_id": "PAPER-002",
            "binance": {
                "api_key": "key",
                "api_secret": "secret",
                "testnet": False,
                "account_type": "COIN_FUTURES",
            },
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "strategy_id": 42,
            "sizing": {
                "method": "kelly",
                "max_leverage": "15",
                "max_position_pct": "0.3",
                "risk_per_trade": "0.01",
                "kelly_fraction": "0.25",
                "atr_multiplier": "1.5",
            },
            "risk": {
                "max_drawdown_pct": "0.10",
                "max_daily_loss_pct": "0.03",
                "max_consecutive_losses": 5,
                "max_position_count": 3,
            },
            "db_path": "/tmp/test.db",
            "logs_path": "/tmp/logs",
            "state_persistence_interval": 120,
        }
        config_path = tmp_path / "config.json"
        with config_path.open("w") as f:
            json.dump(config_data, f)

        config = load_config_from_json(config_path)

        assert config.trader_id == "PAPER-002"
        assert config.binance.testnet is False
        assert config.binance.account_type == "COIN_FUTURES"
        assert config.symbols == ["BTCUSDT", "ETHUSDT"]
        assert config.sizing.method == "kelly"
        assert config.sizing.max_leverage == 15
        assert config.risk.max_consecutive_losses == 5
        assert config.db_path == Path("/tmp/test.db")
        assert config.state_persistence_interval == 120

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """Error for missing config file."""
        config_path = tmp_path / "nonexistent.json"

        with pytest.raises(ConfigurationError, match="Config file not found"):
            load_config_from_json(config_path)

    def test_load_defaults(self, tmp_path: Path) -> None:
        """Missing optional fields use defaults."""
        config_data = {
            "trader_id": "PAPER-001",
            "binance": {
                "api_key": "key",
                "api_secret": "secret",
            },
        }
        config_path = tmp_path / "config.json"
        with config_path.open("w") as f:
            json.dump(config_data, f)

        config = load_config_from_json(config_path)

        # Defaults should be applied
        assert config.binance.testnet is True
        assert config.binance.account_type == "USDT_FUTURES"
        assert config.sizing.method == "fixed_fractional"
        assert config.state_persistence_interval == 60


class TestSaveConfigToJson:
    """Tests for save_config_to_json."""

    def test_save_and_reload(self, tmp_path: Path) -> None:
        """Save config and reload it."""
        from decimal import Decimal

        config = PaperTradingConfig(
            trader_id="PAPER-TEST",
            binance=BinanceTestnetConfig(
                api_key="test_key",
                api_secret="test_secret",
                testnet=True,
            ),
            symbols=["BTCUSDT", "ETHUSDT"],
            strategy_id=123,
            sizing=SizingModuleConfig(
                method="fixed_fractional",
                max_leverage=Decimal("10"),
            ),
            risk=RiskModuleConfig(
                max_drawdown_pct=Decimal("0.15"),
            ),
            db_path=Path("/tmp/test.db"),
        )

        config_path = tmp_path / "saved_config.json"
        save_config_to_json(config, config_path)

        # Verify file exists
        assert config_path.exists()

        # Reload and verify
        reloaded = load_config_from_json(config_path)
        assert reloaded.trader_id == "PAPER-TEST"
        assert reloaded.symbols == ["BTCUSDT", "ETHUSDT"]
        assert reloaded.strategy_id == 123

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        """Save creates parent directory if needed."""
        from decimal import Decimal

        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=BinanceTestnetConfig(api_key="k", api_secret="s"),
            sizing=SizingModuleConfig(max_leverage=Decimal("10")),
            risk=RiskModuleConfig(max_drawdown_pct=Decimal("0.15")),
        )

        nested_path = tmp_path / "nested" / "dir" / "config.json"
        save_config_to_json(config, nested_path)

        assert nested_path.exists()


class TestConfigRoundTrip:
    """Tests for config serialization round-trip."""

    def test_decimal_values_preserved(self, tmp_path: Path) -> None:
        """Decimal values are preserved through round-trip."""
        from decimal import Decimal

        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=BinanceTestnetConfig(api_key="k", api_secret="s"),
            symbols=["BTCUSDT"],
            strategy_id=1,
            sizing=SizingModuleConfig(
                max_leverage=Decimal("12.5"),
                risk_per_trade=Decimal("0.015"),
            ),
            risk=RiskModuleConfig(
                max_drawdown_pct=Decimal("0.123"),
            ),
        )

        config_path = tmp_path / "config.json"
        save_config_to_json(config, config_path)
        reloaded = load_config_from_json(config_path)

        assert reloaded.sizing.max_leverage == Decimal("12.5")
        assert reloaded.sizing.risk_per_trade == Decimal("0.015")
        assert reloaded.risk.max_drawdown_pct == Decimal("0.123")


class TestRunWithConfig:
    """Tests for run_with_config lifecycle behavior."""

    @staticmethod
    def _write_minimal_config(path: Path) -> None:
        data = {
            "trader_id": "PAPER-001",
            "binance": {
                "api_key": "test_key",
                "api_secret": "test_secret",
                "testnet": True,
            },
            "symbols": ["BTCUSDT"],
            "strategy_id": 1,
        }
        with path.open("w") as f:
            json.dump(data, f)

    @pytest.mark.asyncio
    async def test_run_with_config_cleans_up_heartbeat_on_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = tmp_path / "paper_config.json"
        self._write_minimal_config(config_path)

        class _FakeManager:
            def __init__(self) -> None:
                self.completed_calls: list[tuple[int, str | None]] = []
                self.closed = False

            def mark_completed(self, run_id: int, error: str | None = None) -> None:
                self.completed_calls.append((run_id, error))

            def close(self) -> None:
                self.closed = True

        fake_manager = _FakeManager()
        stop_calls = {"count": 0}

        def _stop() -> None:
            stop_calls["count"] += 1

        monkeypatch.setattr(
            "vibe_quant.paper.cli.run_with_heartbeat",
            lambda run_id, db_path: (fake_manager, _stop),
        )

        async def _fake_run_paper_trading(_config: PaperTradingConfig) -> None:
            return

        monkeypatch.setattr("vibe_quant.paper.cli.run_paper_trading", _fake_run_paper_trading)

        rc = await run_with_config(config_path, run_id=7)

        assert rc == 0
        assert fake_manager.completed_calls == [(7, None)]
        assert stop_calls["count"] == 1
        assert fake_manager.closed

    @pytest.mark.asyncio
    async def test_run_with_config_marks_failed_and_cleans_up_on_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = tmp_path / "paper_config.json"
        self._write_minimal_config(config_path)

        class _FakeManager:
            def __init__(self) -> None:
                self.completed_calls: list[tuple[int, str | None]] = []
                self.closed = False

            def mark_completed(self, run_id: int, error: str | None = None) -> None:
                self.completed_calls.append((run_id, error))

            def close(self) -> None:
                self.closed = True

        fake_manager = _FakeManager()
        stop_calls = {"count": 0}

        def _stop() -> None:
            stop_calls["count"] += 1

        monkeypatch.setattr(
            "vibe_quant.paper.cli.run_with_heartbeat",
            lambda run_id, db_path: (fake_manager, _stop),
        )

        async def _failing_run(_config: PaperTradingConfig) -> None:
            raise RuntimeError("kaboom")

        monkeypatch.setattr("vibe_quant.paper.cli.run_paper_trading", _failing_run)

        rc = await run_with_config(config_path, run_id=7)

        assert rc == 1
        assert len(fake_manager.completed_calls) == 1
        assert fake_manager.completed_calls[0][0] == 7
        assert fake_manager.completed_calls[0][1] is not None
        assert "RuntimeError: kaboom" in fake_manager.completed_calls[0][1]
        assert stop_calls["count"] == 1
        assert fake_manager.closed
