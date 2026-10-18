"""Tests for paper trading CLI module."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from vibe_quant.alerts.telegram import (
    ConfigurationError as TelegramConfigurationError,
)
from vibe_quant.alerts.telegram import (
    TelegramUnavailableError,
)
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


@pytest.fixture(autouse=True)
def _binance_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Testnet and live credentials live in DIFFERENT env vars."""
    monkeypatch.setenv("BINANCE_TESTNET_API_KEY", "test_key")
    monkeypatch.setenv("BINANCE_TESTNET_API_SECRET", "test_secret")
    monkeypatch.setenv("BINANCE_API_KEY", "live_key")
    monkeypatch.setenv("BINANCE_API_SECRET", "live_secret")


class TestLoadConfigFromJson:
    """Tests for load_config_from_json."""

    def test_load_minimal_config(self, tmp_path: Path) -> None:
        """Load minimal valid config; testnet creds come from BINANCE_TESTNET_*."""
        config_data = {
            "trader_id": "PAPER-001",
            "binance": {
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
        # Live reads BINANCE_* (never the testnet vars).
        assert config.binance.api_key == "live_key"
        assert config.binance.api_secret == "live_secret"
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
            "binance": {},
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
        # Sizing overrides default to None = keep the validated strategy values.
        assert config.sizing.risk_per_trade is None
        assert config.sizing.max_position_pct is None
        assert config.binance.api_key == "test_key"

    def test_rejects_credentials_in_file(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps({"trader_id": "PAPER-001", "binance": {"api_key": "K", "api_secret": "S"}})
        )
        with pytest.raises(ConfigurationError, match="environment"):
            load_config_from_json(config_path)

    def test_missing_testnet_env_fails_validation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Testnet never silently falls back to the live BINANCE_* keys."""
        monkeypatch.delenv("BINANCE_TESTNET_API_KEY")
        monkeypatch.delenv("BINANCE_TESTNET_API_SECRET")
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps({"trader_id": "PAPER-001", "symbols": ["BTCUSDT"], "strategy_id": 1})
        )
        config = load_config_from_json(config_path)
        assert config.binance.api_key == ""
        assert any("BINANCE_TESTNET_API_KEY" in e for e in config.validate())


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
        assert reloaded.start_fresh is False

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
                "testnet": True,
            },
            "symbols": ["BTCUSDT"],
            "strategy_id": 1,
        }
        with path.open("w") as f:
            json.dump(data, f)

    @pytest.mark.asyncio
    async def test_live_config_requires_live_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = tmp_path / "live.json"
        config_path.write_text(
            json.dumps(
                {
                    "trader_id": "LIVE-001",
                    "binance": {"testnet": False},
                    "symbols": ["BTCUSDT"],
                    "strategy_id": 1,
                }
            )
        )
        started: list[PaperTradingConfig] = []

        async def _fake_run(config: PaperTradingConfig) -> None:
            started.append(config)

        monkeypatch.setattr("vibe_quant.paper.cli.run_paper_trading", _fake_run)
        assert await run_with_config(config_path) == 1
        assert started == []
        assert await run_with_config(config_path, allow_live=True) == 0
        assert started[0].binance.testnet is False
        assert started[0].binance.api_key == "live_key"

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


class TestMarkCompletedFailure:
    """R29: a heartbeat failure never changes the run's exit code, and logs class only."""

    @staticmethod
    def _write_minimal_config(path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "trader_id": "PAPER-001",
                    "binance": {"testnet": True},
                    "symbols": ["BTCUSDT"],
                    "strategy_id": 1,
                }
            )
        )

    @pytest.mark.asyncio
    async def test_success_path_failure_logs_and_keeps_zero(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        config_path = tmp_path / "paper_config.json"
        self._write_minimal_config(config_path)

        class _FakeManager:
            def mark_completed(self, run_id: int, error: str | None = None) -> None:
                raise RuntimeError("mark failed: HEARTBEAT_SECRET_MARKER")

            def close(self) -> None:
                return

        monkeypatch.setattr(
            "vibe_quant.paper.cli.run_with_heartbeat",
            lambda run_id, db_path: (_FakeManager(), lambda: None),
        )

        async def _ok_run(_config: PaperTradingConfig) -> None:
            return

        monkeypatch.setattr("vibe_quant.paper.cli.run_paper_trading", _ok_run)

        with caplog.at_level(logging.ERROR):
            rc = await run_with_config(config_path, run_id=7)

        assert rc == 0
        records = [r for r in caplog.records if "heartbeat mark_completed failed" in r.getMessage()]
        assert len(records) == 1
        assert "RuntimeError" in records[0].getMessage()
        assert records[0].exc_info is None
        assert "HEARTBEAT_SECRET_MARKER" not in caplog.text

    @pytest.mark.asyncio
    async def test_error_path_failure_logs_keeps_one_and_hides_chain(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        config_path = tmp_path / "paper_config.json"
        self._write_minimal_config(config_path)

        class _FakeManager:
            def mark_completed(self, run_id: int, error: str | None = None) -> None:
                raise SystemError("mark failed: HEARTBEAT_SECRET_MARKER")

            def close(self) -> None:
                return

        monkeypatch.setattr(
            "vibe_quant.paper.cli.run_with_heartbeat",
            lambda run_id, db_path: (_FakeManager(), lambda: None),
        )

        async def _failing_run(_config: PaperTradingConfig) -> None:
            raise RuntimeError("run failed: CHAINED_SECRET_MARKER")

        monkeypatch.setattr("vibe_quant.paper.cli.run_paper_trading", _failing_run)

        with caplog.at_level(logging.ERROR):
            rc = await run_with_config(config_path, run_id=7)

        assert rc == 1
        records = [r for r in caplog.records if "heartbeat mark_completed failed" in r.getMessage()]
        assert len(records) == 1
        assert "SystemError" in records[0].getMessage()
        assert records[0].exc_info is None
        assert "HEARTBEAT_SECRET_MARKER" not in caplog.text
        assert "CHAINED_SECRET_MARKER" not in caplog.text

    @pytest.mark.asyncio
    async def test_stderr_shows_class_and_run_id_not_exception_text(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        config_path = tmp_path / "paper_config.json"
        self._write_minimal_config(config_path)

        class _FakeManager:
            def mark_completed(self, run_id: int, error: str | None = None) -> None:
                return

            def close(self) -> None:
                return

        monkeypatch.setattr(
            "vibe_quant.paper.cli.run_with_heartbeat",
            lambda run_id, db_path: (_FakeManager(), lambda: None),
        )

        async def _failing_run(_config: PaperTradingConfig) -> None:
            try:
                raise ValueError("inner: CHAIN_SECRET_MARKER")
            except ValueError as exc:
                raise RuntimeError("boom: STDERR_SECRET_MARKER") from exc

        monkeypatch.setattr("vibe_quant.paper.cli.run_paper_trading", _failing_run)

        rc = await run_with_config(config_path, run_id=7)

        err = capsys.readouterr().err
        assert rc == 1
        assert err.splitlines() == ["Paper trading error: RuntimeError (run 7)"]
        assert "STDERR_SECRET_MARKER" not in err
        assert "CHAIN_SECRET_MARKER" not in err

    @pytest.mark.asyncio
    async def test_stderr_shows_class_only_without_run_id(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        config_path = tmp_path / "paper_config.json"
        self._write_minimal_config(config_path)

        async def _failing_run(_config: PaperTradingConfig) -> None:
            raise KeyError("STDERR_SECRET_MARKER")

        monkeypatch.setattr("vibe_quant.paper.cli.run_paper_trading", _failing_run)

        rc = await run_with_config(config_path, run_id=None)

        err = capsys.readouterr().err
        assert rc == 1
        assert err.splitlines() == ["Paper trading error: KeyError"]
        assert "STDERR_SECRET_MARKER" not in err


class TestMarginCushionConfig:
    """sizing.max_margin_pct in the node's config file: absent = paper default, null = error."""

    @staticmethod
    def _write(tmp_path: Path, sizing: dict[str, object]) -> Path:
        path = tmp_path / "c.json"
        path.write_text(
            json.dumps(
                {
                    "trader_id": "PAPER-001",
                    "binance": {"testnet": True},
                    "symbols": ["BTCUSDT"],
                    "strategy_id": 1,
                    "sizing": sizing,
                }
            )
        )
        return path

    def test_absent_is_none_and_value_round_trips(self, tmp_path: Path) -> None:
        from decimal import Decimal

        assert load_config_from_json(self._write(tmp_path, {})).sizing.max_margin_pct is None
        loaded = load_config_from_json(self._write(tmp_path, {"max_margin_pct": 0.3}))
        assert loaded.sizing.max_margin_pct == Decimal("0.3")
        out = tmp_path / "out.json"
        save_config_to_json(loaded, out)
        assert json.loads(out.read_text())["sizing"]["max_margin_pct"] == 0.3
        again = load_config_from_json(out)
        assert again.sizing.max_margin_pct == Decimal("0.3")
        save_config_to_json(load_config_from_json(self._write(tmp_path, {})), out)
        assert "max_margin_pct" not in json.loads(out.read_text())["sizing"]

    @pytest.mark.parametrize("bad", [None, 0, -0.5, 1.5, True, "0.3", float("nan")])
    def test_invalid_is_an_error(self, tmp_path: Path, bad: object) -> None:
        with pytest.raises(ConfigurationError, match="max_margin_pct"):
            load_config_from_json(self._write(tmp_path, {"max_margin_pct": bad}))

    def test_one_is_allowed(self, tmp_path: Path) -> None:
        from decimal import Decimal

        loaded = load_config_from_json(self._write(tmp_path, {"max_margin_pct": 1}))
        assert loaded.sizing.max_margin_pct == Decimal("1")


class TestStartFresh:
    """R20/R20b: the ``start_fresh`` config key (skip inheriting the last session)."""

    @staticmethod
    def _write(tmp_path: Path, extra: dict[str, object]) -> Path:
        path = tmp_path / "c.json"
        path.write_text(
            json.dumps(
                {
                    "trader_id": "PAPER-001",
                    "binance": {"testnet": True},
                    "symbols": ["BTCUSDT"],
                    "strategy_id": 1,
                    **extra,
                }
            )
        )
        return path

    def test_load_start_fresh_rows(self, tmp_path: Path) -> None:
        assert load_config_from_json(self._write(tmp_path, {})).start_fresh is False
        assert load_config_from_json(self._write(tmp_path, {"start_fresh": True})).start_fresh
        assert (
            load_config_from_json(self._write(tmp_path, {"start_fresh": False})).start_fresh
            is False
        )

    @pytest.mark.parametrize("bad", [None, "yes", 1, 0, "true"])
    def test_load_start_fresh_rejects_non_bool(self, tmp_path: Path, bad: object) -> None:
        with pytest.raises(ConfigurationError, match="start_fresh must be true or false"):
            load_config_from_json(self._write(tmp_path, {"start_fresh": bad}))

    def test_start_fresh_round_trip(self, tmp_path: Path) -> None:
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=BinanceTestnetConfig(api_key="k", api_secret="s", testnet=True),
            start_fresh=True,
        )
        out = tmp_path / "out.json"
        save_config_to_json(config, out)
        assert json.loads(out.read_text())["start_fresh"] is True
        assert load_config_from_json(out).start_fresh is True


class TestTradeAlerts:
    """ci7q9: the ``trade_alerts`` config key (per-session Telegram trade messages)."""

    @staticmethod
    def _write(tmp_path: Path, extra: dict[str, object]) -> Path:
        path = tmp_path / "c.json"
        path.write_text(
            json.dumps(
                {
                    "trader_id": "PAPER-001",
                    "binance": {"testnet": True},
                    "symbols": ["BTCUSDT"],
                    "strategy_id": 1,
                    **extra,
                }
            )
        )
        return path

    def test_load_trade_alerts_rows(self, tmp_path: Path) -> None:
        assert load_config_from_json(self._write(tmp_path, {})).trade_alerts is False
        assert load_config_from_json(self._write(tmp_path, {"trade_alerts": True})).trade_alerts
        assert (
            load_config_from_json(self._write(tmp_path, {"trade_alerts": False})).trade_alerts
            is False
        )

    @pytest.mark.parametrize("bad", [None, "yes", 1, 0, "true"])
    def test_load_trade_alerts_rejects_non_bool(self, tmp_path: Path, bad: object) -> None:
        with pytest.raises(ConfigurationError, match="trade_alerts must be true or false"):
            load_config_from_json(self._write(tmp_path, {"trade_alerts": bad}))

    def test_trade_alerts_round_trip(self, tmp_path: Path) -> None:
        config = PaperTradingConfig(
            trader_id="PAPER-001",
            binance=BinanceTestnetConfig(api_key="k", api_secret="s", testnet=True),
            trade_alerts=True,
        )
        out = tmp_path / "out.json"
        save_config_to_json(config, out)
        assert json.loads(out.read_text())["trade_alerts"] is True
        assert load_config_from_json(out).trade_alerts is True


class TestAllowlistedRefusalMessage:
    """u52nu: the curated refusal message prints for allowlisted errors only.

    The three allowlisted classes are ``paper.config.ConfigurationError``,
    ``alerts.telegram.ConfigurationError`` and ``alerts.telegram.TelegramUnavailableError``;
    every other exception keeps the class-only stderr line (R1-R5).
    """

    @staticmethod
    def _write_minimal_config(path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "trader_id": "PAPER-001",
                    "binance": {"testnet": True},
                    "symbols": ["BTCUSDT"],
                    "strategy_id": 1,
                }
            )
        )

    @staticmethod
    def _fake_heartbeat(monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeManager:
            def mark_completed(self, run_id: int, error: str | None = None) -> None:
                return

            def close(self) -> None:
                return

        monkeypatch.setattr(
            "vibe_quant.paper.cli.run_with_heartbeat",
            lambda run_id, db_path: (_FakeManager(), lambda: None),
        )

    @staticmethod
    def _raise_with(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
        async def _failing_run(_config: PaperTradingConfig) -> None:
            raise exc

        monkeypatch.setattr("vibe_quant.paper.cli.run_paper_trading", _failing_run)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("cls", "message"),
        [
            (ConfigurationError, "sizing.max_margin_pct: must be in (0, 1], got 1.5"),
            (TelegramConfigurationError, "Missing TELEGRAM_CHAT_ID environment variable"),
            (
                TelegramUnavailableError,
                "Telegram alerts are configured but not working (HTTP 401 Unauthorized); "
                "refusing to start",
            ),
        ],
    )
    async def test_allowlisted_message_without_run_id(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        cls: type[Exception],
        message: str,
    ) -> None:
        config_path = tmp_path / "paper_config.json"
        self._write_minimal_config(config_path)
        self._raise_with(monkeypatch, cls(message))

        rc = await run_with_config(config_path, run_id=None)

        err = capsys.readouterr().err
        assert rc == 1
        assert err.splitlines() == [f"Paper trading error: {cls.__name__}: {message}"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("cls", "message"),
        [
            (ConfigurationError, "sizing.max_margin_pct: must be in (0, 1], got 1.5"),
            (TelegramConfigurationError, "Missing TELEGRAM_CHAT_ID environment variable"),
            (
                TelegramUnavailableError,
                "Telegram alerts are configured but not working (HTTP 401 Unauthorized); "
                "refusing to start",
            ),
        ],
    )
    async def test_allowlisted_message_with_run_id(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        cls: type[Exception],
        message: str,
    ) -> None:
        config_path = tmp_path / "paper_config.json"
        self._write_minimal_config(config_path)
        self._fake_heartbeat(monkeypatch)
        self._raise_with(monkeypatch, cls(message))

        rc = await run_with_config(config_path, run_id=7)

        err = capsys.readouterr().err
        assert rc == 1
        assert err.splitlines() == [f"Paper trading error: {cls.__name__}: {message} (run 7)"]

    @pytest.mark.asyncio
    async def test_allowlisted_message_is_redacted(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        config_path = tmp_path / "paper_config.json"
        self._write_minimal_config(config_path)
        token = "123456789:AAFakeBotTokenAbcdefghij"
        self._raise_with(
            monkeypatch,
            TelegramUnavailableError(
                "Telegram alerts are configured but not working (HTTP 401 against "
                f"https://api.telegram.org/bot{token}/sendChatAction); refusing to start"
            ),
        )

        rc = await run_with_config(config_path, run_id=None)

        err = capsys.readouterr().err
        assert rc == 1
        assert err.splitlines() == [
            "Paper trading error: TelegramUnavailableError: Telegram alerts are configured "
            "but not working (HTTP 401 against "
            "https://api.telegram.org/bot<redacted>/sendChatAction); refusing to start"
        ]
        assert token not in err

    @pytest.mark.asyncio
    async def test_non_allowlisted_stays_class_only(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        config_path = tmp_path / "paper_config.json"
        self._write_minimal_config(config_path)
        self._raise_with(monkeypatch, RuntimeError("secret-xyz"))

        rc = await run_with_config(config_path, run_id=None)

        err = capsys.readouterr().err
        assert rc == 1
        assert err.splitlines() == ["Paper trading error: RuntimeError"]
        assert "secret-xyz" not in err

    @pytest.mark.asyncio
    async def test_subclass_of_allowlisted_shows_message(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        class _PaperConfigError(ConfigurationError):
            pass

        config_path = tmp_path / "paper_config.json"
        self._write_minimal_config(config_path)
        self._fake_heartbeat(monkeypatch)
        self._raise_with(monkeypatch, _PaperConfigError("derived refusal reason"))

        rc = await run_with_config(config_path, run_id=7)

        err = capsys.readouterr().err
        assert rc == 1
        assert err.splitlines() == [
            "Paper trading error: _PaperConfigError: derived refusal reason (run 7)"
        ]

    @pytest.mark.asyncio
    async def test_subclass_of_non_allowlisted_stays_class_only(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        class NotAllowlistedConfigurationError(Exception):
            pass

        config_path = tmp_path / "paper_config.json"
        self._write_minimal_config(config_path)
        self._raise_with(monkeypatch, NotAllowlistedConfigurationError("nope-xyz"))

        rc = await run_with_config(config_path, run_id=None)

        err = capsys.readouterr().err
        assert rc == 1
        assert err.splitlines() == ["Paper trading error: NotAllowlistedConfigurationError"]
        assert "nope-xyz" not in err
