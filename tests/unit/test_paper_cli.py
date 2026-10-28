"""Tests for paper trading CLI module."""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from pathlib import Path

import pytest

from vibe_quant.alerts.telegram import (
    ConfigurationError as TelegramConfigurationError,
)
from vibe_quant.alerts.telegram import (
    TelegramUnavailableError,
)
from vibe_quant.paper import account_lock as account_lock_module
from vibe_quant.paper import lineage
from vibe_quant.paper.account_lock import acquire_account_lock
from vibe_quant.paper.cli import (
    load_config_from_json,
    main,
    run_with_config,
    save_config_to_json,
)
from vibe_quant.paper.config import (
    BinanceTestnetConfig,
    ConfigurationError,
    PaperTradingConfig,
    RiskModuleConfig,
    SizingModuleConfig,
    account_id,
)
from vibe_quant.paper.persistence import StateCheckpoint, StatePersistence


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
    """Tests for save_config_to_json and the load round-trip."""

    def test_save_creates_the_directory_and_round_trips_decimals(self, tmp_path: Path) -> None:
        """The parent directory is created; Decimal sizing/risk values survive the trip."""
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
                max_leverage=Decimal("12.5"),
                risk_per_trade=Decimal("0.015"),
            ),
            risk=RiskModuleConfig(
                max_drawdown_pct=Decimal("0.123"),
            ),
            db_path=Path("/tmp/test.db"),
        )

        config_path = tmp_path / "nested" / "dir" / "saved_config.json"
        save_config_to_json(config, config_path)

        assert config_path.exists()
        reloaded = load_config_from_json(config_path)
        assert reloaded.trader_id == "PAPER-TEST"
        assert reloaded.symbols == ["BTCUSDT", "ETHUSDT"]
        assert reloaded.strategy_id == 123
        assert reloaded.start_fresh is False
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
    async def test_run_with_config_cleans_up_heartbeat_on_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = tmp_path / "paper_config.json"
        self._write_minimal_config(config_path)

        class _FakeManager:
            def __init__(self) -> None:
                self.completed_calls: list[tuple[int, str | None]] = []
                self.closed = False

            def claim_process(
                self, run_id: int, pid: int, identity: str | None
            ) -> str | None:
                return None

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


def _testnet_account() -> str:
    """The account id the minimal config's ``test_key`` resolves to."""
    return account_id(
        BinanceTestnetConfig(api_key="test_key", api_secret="test_secret", testnet=True)
    )


def _live_account(api_key: str = "live_key", *, testnet: bool = False) -> str:
    """The account id the live config's ``live_key`` (or another key) resolves to."""
    return account_id(
        BinanceTestnetConfig(api_key=api_key, api_secret="live_secret", testnet=testnet)
    )


class TestAccountLock:
    """rw1g8: two CLI runs on one account must not run with separate guards."""

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
    def _fake_heartbeat(
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[list[tuple[int, str | None]], dict[str, int]]:
        completed: list[tuple[int, str | None]] = []
        stops = {"count": 0}

        class _FakeManager:
            def claim_process(
                self, run_id: int, pid: int, identity: str | None
            ) -> str | None:
                return None

            def mark_completed(self, run_id: int, error: str | None = None) -> None:
                completed.append((run_id, error))

            def close(self) -> None:
                return

        def _stop() -> None:
            stops["count"] += 1

        monkeypatch.setattr(
            "vibe_quant.paper.cli.run_with_heartbeat",
            lambda run_id, db_path: (_FakeManager(), _stop),
        )
        return completed, stops

    @pytest.mark.asyncio
    async def test_success_releases_the_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = tmp_path / "paper_config.json"
        self._write_minimal_config(config_path)

        async def _fake_run(_config: PaperTradingConfig) -> None:
            return

        monkeypatch.setattr("vibe_quant.paper.cli.run_paper_trading", _fake_run)

        assert await run_with_config(config_path) == 0

        # A second session on the same account may start immediately: the first released.
        lock = acquire_account_lock(_testnet_account(), "PAPER-002")
        lock.release()

    @pytest.mark.asyncio
    async def test_failure_releases_the_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = tmp_path / "paper_config.json"
        self._write_minimal_config(config_path)

        async def _failing_run(_config: PaperTradingConfig) -> None:
            raise RuntimeError("kaboom")

        monkeypatch.setattr("vibe_quant.paper.cli.run_paper_trading", _failing_run)

        assert await run_with_config(config_path) == 1

        lock = acquire_account_lock(_testnet_account(), "PAPER-002")
        lock.release()

    @pytest.mark.asyncio
    async def test_a_different_account_is_not_blocked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = tmp_path / "paper_config.json"
        self._write_minimal_config(config_path)
        started: list[PaperTradingConfig] = []

        async def _fake_run(config: PaperTradingConfig) -> None:
            started.append(config)

        monkeypatch.setattr("vibe_quant.paper.cli.run_paper_trading", _fake_run)

        other = account_id(
            BinanceTestnetConfig(api_key="other_key", api_secret="s", testnet=True)
        )
        held = acquire_account_lock(other, "PAPER-OTHER")
        try:
            assert await run_with_config(config_path) == 0
        finally:
            held.release()
        assert len(started) == 1


def _write_trader_config(
    path: Path, *, trader_id: str = "PAPER-001", testnet: bool = True
) -> None:
    path.write_text(
        json.dumps(
            {
                "trader_id": trader_id,
                "binance": {"testnet": testnet},
                "symbols": ["BTCUSDT"],
                "strategy_id": 1,
            }
        )
    )


def _trader_lock_is_free(trader_id: str) -> bool:
    """Independently take, then release, the trader lock in the patched lock dir.

    A second open file description contends with a leaked one even inside this
    process, so a returned run that never released its fd still holds the lock
    here. False (not ``TraderBusyError``) so the caller's assertion names the leak.
    """
    lock_dir = account_lock_module.PAPER_LOCK_DIR
    assert lock_dir is not None
    try:
        probe = account_lock_module.acquire_trader_lock(trader_id, "PAPER-PROBE", lock_dir)
    except account_lock_module.TraderBusyError:
        return False
    probe.release()
    return True


class TestTraderLock:
    """2q7cv: one paper session per trader id, across keys and across venues.

    The refusal itself (text and exit code, on another key and beside a live
    session) is a row of ``tests/unit/test_paper_admission_matrix.py``
    (``test_cli_trader_busy_exits_one_with_the_holder``); these tests own the
    lock's lifecycle, a leaked lock being the failure mode the refusal hides.
    """

    @pytest.mark.asyncio
    async def test_successful_run_releases_the_trader_lock(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A returned session releases its own trader lock (F1: a leaked fd still holds it)."""
        config_path = tmp_path / "paper_config.json"
        _write_trader_config(config_path)
        TestAccountLock._fake_heartbeat(monkeypatch)
        started: list[PaperTradingConfig] = []

        async def _fake_run(config: PaperTradingConfig) -> None:
            started.append(config)

        monkeypatch.setattr("vibe_quant.paper.cli.run_paper_trading", _fake_run)

        assert await run_with_config(config_path) == 0
        assert [c.trader_id for c in started] == ["PAPER-001"]
        assert _trader_lock_is_free("PAPER-001"), "the returned run left the trader lock held"

    @pytest.mark.asyncio
    async def test_failed_run_releases_the_trader_lock(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A trading exception still releases the trader lock before run_with_config returns."""
        config_path = tmp_path / "paper_config.json"
        _write_trader_config(config_path)
        completed, stops = TestAccountLock._fake_heartbeat(monkeypatch)
        started: list[PaperTradingConfig] = []

        async def _boom(config: PaperTradingConfig) -> None:
            started.append(config)
            raise RuntimeError("trading failed")

        monkeypatch.setattr("vibe_quant.paper.cli.run_paper_trading", _boom)

        assert await run_with_config(config_path, run_id=11) == 1
        assert [c.trader_id for c in started] == ["PAPER-001"]
        assert capsys.readouterr().err.splitlines() == [
            "Paper trading error: RuntimeError (run 11)"
        ]
        assert completed == [(11, "RuntimeError: trading failed")]
        assert stops["count"] == 1
        assert _trader_lock_is_free("PAPER-001"), (
            "the failed run left the trader lock held"
        )

    @pytest.mark.asyncio
    async def test_unopenable_trader_lock_exits_one_with_the_class_only(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """R41 at the CLI: an OSError from the trader lock prints its class only."""
        config_path = tmp_path / "paper_config.json"
        _write_trader_config(config_path)
        started: list[PaperTradingConfig] = []

        async def _fake_run(config: PaperTradingConfig) -> None:
            started.append(config)

        monkeypatch.setattr("vibe_quant.paper.cli.run_paper_trading", _fake_run)

        lock_dir = account_lock_module.PAPER_LOCK_DIR
        assert lock_dir is not None
        lock_dir.mkdir(parents=True, exist_ok=True)
        path = lock_dir / "trader_PAPER-001.lock"
        path.write_text("")
        path.chmod(0)
        try:
            rc = await run_with_config(config_path)
        finally:
            path.chmod(0o600)

        assert rc == 1
        assert started == []
        assert capsys.readouterr().err.splitlines() == ["Paper trading error: PermissionError"]


class TestLiveStartFresh:
    """v4l2n + 30vki: a LIVE start_fresh skips another trader id's risk carry.

    The LIVE refusal and the confirmed LIVE run are rows of
    ``tests/unit/test_paper_admission_matrix.py`` (``test_cli_live_start_fresh_*``
    and ``test_admission_preserves_checkpoint_lookup_short_circuit``).
    """

    @staticmethod
    def _write(path: Path, *, testnet: bool, start_fresh: bool) -> None:
        path.write_text(
            json.dumps(
                {
                    "trader_id": "LIVE-001" if not testnet else "PAPER-001",
                    "binance": {"testnet": testnet},
                    "symbols": ["BTCUSDT"],
                    "strategy_id": 1,
                    "start_fresh": start_fresh,
                }
            )
        )

    @pytest.mark.asyncio
    async def test_testnet_start_fresh_needs_no_extra_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A testnet start_fresh is not a LIVE consent: it runs without the flag."""
        config_path = tmp_path / "testnet_fresh.json"
        self._write(config_path, testnet=True, start_fresh=True)
        started: list[PaperTradingConfig] = []

        async def _fake_run(config: PaperTradingConfig) -> None:
            started.append(config)

        monkeypatch.setattr("vibe_quant.paper.cli.run_paper_trading", _fake_run)

        assert await run_with_config(config_path) == 0
        assert [(c.binance.testnet, c.start_fresh) for c in started] == [(True, True)]


class TestLiveAccountWithoutCheckpoint:
    """6ub67 + neuvu.2: a LIVE start with no saved risk state to carry refuses unless consented.

    A first LIVE session, or a new trader id on a new key (no checkpoint on the
    account, no earlier live session of this trader on another key): the drawdown,
    daily-loss and losing-streak limits would silently start from zero. A key
    rotation by the SAME trader id carries its own risk and halt and needs no consent.
    """

    _REFUSAL = (
        "Refusing LIVE start: no saved risk state to carry (no checkpoint on this "
        "account and no earlier live session of this trader on another key): "
        "drawdown, daily-loss and losing-streak limits would start from zero; "
        "re-run with --confirm-fresh-live to confirm"
    )

    @staticmethod
    def _write(path: Path, *, db_path: Path, testnet: bool = False) -> None:
        path.write_text(
            json.dumps(
                {
                    "trader_id": "PAPER-001" if testnet else "LIVE-001",
                    "binance": {"testnet": testnet},
                    "symbols": ["BTCUSDT"],
                    "strategy_id": 1,
                    "db_path": str(db_path),
                }
            )
        )

    @staticmethod
    def _seed(db_path: Path, account: str, trader_id: str = "OTHER-001") -> None:
        store = StatePersistence(db_path, trader_id=trader_id)
        try:
            store.save_checkpoint(
                StateCheckpoint(trader_id=trader_id, node_status={"account": account})
            )
        finally:
            store.close()

    @staticmethod
    def _patch(
        monkeypatch: pytest.MonkeyPatch,
        started: list[PaperTradingConfig],
        heartbeats: list[int | None],
    ) -> None:
        async def _fake_run(config: PaperTradingConfig) -> None:
            started.append(config)

        class _Manager:
            def mark_completed(self, run_id: int, error: str | None = None) -> None:
                raise AssertionError("a refused start must not reach mark_completed")

            def close(self) -> None:
                raise AssertionError("a refused start must not open a manager")

        def _fake_heartbeat(run_id: int | None, db_path: Path | None) -> tuple[_Manager, object]:
            heartbeats.append(run_id)
            return _Manager(), lambda: None

        monkeypatch.setattr("vibe_quant.paper.cli.run_paper_trading", _fake_run)
        monkeypatch.setattr("vibe_quant.paper.cli.run_with_heartbeat", _fake_heartbeat)

    @pytest.mark.asyncio
    async def test_a_checkpoint_on_the_account_runs_without_the_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db_path = tmp_path / "state.db"
        self._seed(db_path, _live_account())
        config_path = tmp_path / "live.json"
        self._write(config_path, db_path=db_path)
        started: list[PaperTradingConfig] = []
        heartbeats: list[int | None] = []
        self._patch(monkeypatch, started, heartbeats)

        assert await run_with_config(config_path, allow_live=True) == 0
        assert [c.binance.testnet for c in started] == [False]

    @pytest.mark.asyncio
    async def test_a_failed_lookup_refuses_and_leaks_no_text(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """R12: fail closed - a lookup that cannot answer is "no checkpoint"."""
        marker = "LOOKUP-TEXT-MARKER-42"

        def _boom(store: StatePersistence, account: str, trader_id: str) -> bool:
            raise RuntimeError(marker)

        monkeypatch.setattr(lineage, "has_prior_state", _boom)
        config_path = tmp_path / "live.json"
        self._write(config_path, db_path=tmp_path / "state.db")
        started: list[PaperTradingConfig] = []
        heartbeats: list[int | None] = []
        self._patch(monkeypatch, started, heartbeats)

        with caplog.at_level(logging.DEBUG):
            assert await run_with_config(config_path, allow_live=True) == 1

        assert started == []
        assert capsys.readouterr().err.splitlines() == [self._REFUSAL]
        assert "paper account checkpoint lookup failed (RuntimeError)" in caplog.text
        assert marker not in caplog.text


class TestNoCheckpointRefusalIsPersisted:
    """6ub67 F1: with --run-id the consent refusal is the run's terminal failure.

    Without it the API-launched run (e.g. /restore of a trader with no risk state to
    carry) stays
    ``running`` with no error after the child exits, and a later job sync only
    records "Process terminated unexpectedly", hiding the operator action.
    """

    @staticmethod
    def _live_run(db_path: Path) -> int:
        """A paper run + running job row, as the API leaves them after launching the CLI."""
        from vibe_quant.db.state_manager import StateManager

        state = StateManager(db_path=db_path)
        try:
            strategy_id = state.create_strategy(
                "s",
                {
                    "name": "s",
                    "timeframe": "1h",
                    "indicators": {"rsi_14": {"type": "RSI", "period": 14}},
                    "entry_conditions": {"long": ["rsi_14 < 30"]},
                    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
                    "take_profit": {"type": "fixed_pct", "percent": 4.0},
                },
            )
            run_id = state.create_backtest_run(
                strategy_id=strategy_id,
                run_mode="paper",
                symbols=["BTCUSDT"],
                timeframe="",
                start_date="",
                end_date="",
                parameters={},
            )
            state.update_backtest_run_status(run_id, "running", pid=4242)
            state.conn.execute(
                """INSERT INTO background_jobs
                   (run_id, pid, job_type, status, started_at, heartbeat_at)
                   VALUES (?, 4242, 'paper', 'running', datetime('now'), datetime('now'))""",
                (run_id,),
            )
            state.conn.commit()
            return run_id
        finally:
            state.close()

    @staticmethod
    def _rows(db_path: Path, run_id: int) -> tuple[tuple[str, str | None], tuple[str, str | None]]:

        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            run = conn.execute(
                "SELECT status, error_message FROM backtest_runs WHERE id = ?", (run_id,)
            ).fetchone()
            job = conn.execute(
                "SELECT status, error_message FROM background_jobs WHERE run_id = ?", (run_id,)
            ).fetchone()
        finally:
            conn.close()
        return (run[0], run[1]), (job[0], job[1])

    @pytest.mark.asyncio
    async def test_run_and_job_fail_with_the_curated_text(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        db_path = tmp_path / "state.db"
        run_id = self._live_run(db_path)
        config_path = tmp_path / "live.json"
        TestLiveAccountWithoutCheckpoint._write(config_path, db_path=db_path)
        started: list[PaperTradingConfig] = []
        heartbeats: list[int | None] = []
        TestLiveAccountWithoutCheckpoint._patch(monkeypatch, started, heartbeats)

        rc = await run_with_config(config_path, run_id=run_id, allow_live=True)

        refusal = TestLiveAccountWithoutCheckpoint._REFUSAL
        assert rc == 1
        assert started == []
        assert heartbeats == []
        assert capsys.readouterr().err.splitlines() == [refusal]
        assert self._rows(db_path, run_id) == (("failed", refusal), ("failed", refusal))
        assert not (tmp_path / "paper_locks").exists()

    @pytest.mark.asyncio
    async def test_a_persistence_failure_keeps_the_refusal_and_leaks_no_text(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from vibe_quant.jobs.manager import BacktestJobManager

        marker = "PERSIST-TEXT-MARKER-77"

        def _boom(self: BacktestJobManager, run_id: int, error: str | None = None) -> None:
            raise RuntimeError(marker)

        monkeypatch.setattr(BacktestJobManager, "mark_completed", _boom)
        db_path = tmp_path / "state.db"
        run_id = self._live_run(db_path)
        config_path = tmp_path / "live.json"
        TestLiveAccountWithoutCheckpoint._write(config_path, db_path=db_path)
        TestLiveAccountWithoutCheckpoint._patch(monkeypatch, [], [])

        with caplog.at_level(logging.DEBUG):
            rc = await run_with_config(config_path, run_id=run_id, allow_live=True)

        assert rc == 1
        assert capsys.readouterr().err.splitlines() == [TestLiveAccountWithoutCheckpoint._REFUSAL]
        assert "RuntimeError" in caplog.text
        assert marker not in caplog.text

    @pytest.mark.asyncio
    async def test_no_run_id_touches_no_database(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Direct CLI use (no --run-id) never opens a job manager."""
        from vibe_quant.jobs.manager import BacktestJobManager

        def _forbidden(self: BacktestJobManager, run_id: int, error: str | None = None) -> None:
            raise AssertionError("no run id, nothing to persist")

        monkeypatch.setattr(BacktestJobManager, "mark_completed", _forbidden)
        config_path = tmp_path / "live.json"
        TestLiveAccountWithoutCheckpoint._write(config_path, db_path=tmp_path / "state.db")
        TestLiveAccountWithoutCheckpoint._patch(monkeypatch, [], [])

        assert await run_with_config(config_path, allow_live=True) == 1


def test_main_passes_both_live_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--confirm-fresh-live`` reaches ``run_with_config`` as ``allow_fresh_live``."""
    seen: list[tuple[Path, int | None, dict[str, bool]]] = []

    async def _fake(
        config_path: Path, run_id: int | None = None, **kwargs: bool
    ) -> int:
        seen.append((config_path, run_id, kwargs))
        return 0

    monkeypatch.setattr("vibe_quant.paper.cli.run_with_config", _fake)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "start",
            "--config",
            "config.json",
            "--run-id",
            "9",
            "--live",
            "--confirm-fresh-live",
        ],
    )

    assert main() == 0
    assert seen == [(Path("config.json"), 9, {"allow_live": True, "allow_fresh_live": True})]


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
            def claim_process(
                self, run_id: int, pid: int, identity: str | None
            ) -> str | None:
                return None

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
            def claim_process(
                self, run_id: int, pid: int, identity: str | None
            ) -> str | None:
                return None

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
            def claim_process(
                self, run_id: int, pid: int, identity: str | None
            ) -> str | None:
                return None

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
        # The upper bound is inclusive; the owner's table is
        # ``tests/unit/test_sizing_settings.py``.
        loaded = load_config_from_json(self._write(tmp_path, {"max_margin_pct": 1}))
        assert loaded.sizing.max_margin_pct == Decimal("1")

    @pytest.mark.parametrize("bad", [None, 0, 1.5, True, float("nan")])
    def test_invalid_is_an_error(self, tmp_path: Path, bad: object) -> None:
        """Only the class-guarded rows the loader adds: absent, a bool and a NaN value.

        The value bounds themselves are pinned by ``tests/unit/test_sizing_settings.py``.
        """
        with pytest.raises(ConfigurationError, match="max_margin_pct"):
            load_config_from_json(self._write(tmp_path, {"max_margin_pct": bad}))


class TestBoolConfigKeys:
    """start_fresh (R20/R20b) and trade_alerts (ci7q9): boolean config keys."""

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

    def test_load_rows(self, tmp_path: Path) -> None:
        """Absent is False; both JSON booleans load as themselves."""
        for key in ("start_fresh", "trade_alerts"):
            assert getattr(load_config_from_json(self._write(tmp_path, {})), key) is False
            assert getattr(load_config_from_json(self._write(tmp_path, {key: True})), key) is True
            assert getattr(load_config_from_json(self._write(tmp_path, {key: False})), key) is False

    @pytest.mark.parametrize("key", ["start_fresh", "trade_alerts"])
    @pytest.mark.parametrize("bad", [None, 1, "true"])
    def test_rejects_non_bool(self, tmp_path: Path, key: str, bad: object) -> None:
        with pytest.raises(ConfigurationError, match=f"{key} must be true or false"):
            load_config_from_json(self._write(tmp_path, {key: bad}))

    def test_round_trip(self, tmp_path: Path) -> None:
        for key in ("start_fresh", "trade_alerts"):
            config = PaperTradingConfig(
                trader_id="PAPER-001",
                binance=BinanceTestnetConfig(api_key="k", api_secret="s", testnet=True),
                start_fresh=key == "start_fresh",
                trade_alerts=key == "trade_alerts",
            )
            out = tmp_path / f"out_{key}.json"
            save_config_to_json(config, out)
            assert json.loads(out.read_text())[key] is True
            assert getattr(load_config_from_json(out), key) is True


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
            def claim_process(
                self, run_id: int, pid: int, identity: str | None
            ) -> str | None:
                return None

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
    @pytest.mark.parametrize("run_id", [None, 7], ids=["no-run-id", "run-id"])
    async def test_allowlisted_message(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        cls: type[Exception],
        message: str,
        run_id: int | None,
    ) -> None:
        """The curated text prints for each allowlisted class, with or without a run id."""
        config_path = tmp_path / "paper_config.json"
        self._write_minimal_config(config_path)
        if run_id is not None:
            self._fake_heartbeat(monkeypatch)
        self._raise_with(monkeypatch, cls(message))

        rc = await run_with_config(config_path, run_id=run_id)

        err = capsys.readouterr().err
        suffix = "" if run_id is None else f" (run {run_id})"
        assert rc == 1
        assert err.splitlines() == [f"Paper trading error: {cls.__name__}: {message}{suffix}"]

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
