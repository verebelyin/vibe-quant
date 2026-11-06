"""Paper margin type derives from the validated run's liquidation mode (T6 / P2a).

Contract rows R39, R39b (incl. the API -> JSON -> loader -> node end-to-end case),
R40, R41 and R43.

``PaperTradingConfig.margin_type`` is ``None`` by default: the margin type is then
derived from the validated run's ``liquidation_mode`` (isolated -> ISOLATED, cross
and off -> CROSSED). An explicit ``CROSSED``/``ISOLATED`` in the config file is an
override and keeps producing the existing diff entry.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from httpx import ASGITransport, AsyncClient

from tests.unit.test_audit_paper_node import LIQ_OFF_ENTRY
from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager
from vibe_quant.paper.cli import load_config_from_json, save_config_to_json
from vibe_quant.paper.config import (
    BinanceTestnetConfig,
    ConfigurationError,
    PaperTradingConfig,
)
from vibe_quant.paper.node import PaperTradingNode

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

DSL: dict[str, Any] = {
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

ISOLATED_PARAMS: dict[str, Any] = {
    "rsi_14_period": 21,
    "take_profit_risk_reward": 3.0,
    "liquidation_mode": "isolated",
}
OFF_PARAMS: dict[str, Any] = {
    "rsi_14_period": 21,
    "take_profit_risk_reward": 3.0,
    "bnb_discount": True,
}
CROSS_PARAMS: dict[str, Any] = {
    "rsi_14_period": 21,
    "take_profit_risk_reward": 3.0,
    "liquidation_mode": "cross",
}


class _FakeKernel:
    class _RiskEngine:
        def execute(self, command: object) -> None:
            pass

    def __init__(self) -> None:
        self.risk_engine = self._RiskEngine()


class _FakeTrader:
    def __init__(self) -> None:
        self.actors: list[object] = []
        self.strategies: list[object] = []

    def add_actor(self, actor: object) -> None:
        self.actors.append(actor)

    def add_strategy(self, strategy: object) -> None:
        self.strategies.append(strategy)


class _FakeTradingNode:
    instances: list[_FakeTradingNode] = []

    def __init__(self, config: Any) -> None:
        self.config = config
        self.kernel = _FakeKernel()
        self.trader = _FakeTrader()
        self.built = False
        _FakeTradingNode.instances.append(self)

    def add_data_client_factory(self, name: str, factory: object) -> None:
        pass

    def add_exec_client_factory(self, name: str, factory: object) -> None:
        pass

    def build(self) -> None:
        self.built = True

    async def run_async(self) -> None:
        return

    async def stop_async(self) -> None:
        return


@pytest.fixture(autouse=True)
def _binance_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BINANCE_TESTNET_API_KEY", "tk")
    monkeypatch.setenv("BINANCE_TESTNET_API_SECRET", "ts")


@pytest.fixture()
def fake_tradingnode(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTradingNode]:
    import nautilus_trader.live.node as live_node

    _FakeTradingNode.instances = []
    monkeypatch.setattr(live_node, "TradingNode", _FakeTradingNode)
    return _FakeTradingNode


def _write_config(path: Path, **extra: Any) -> Path:
    data: dict[str, Any] = {
        "trader_id": "PAPER-001",
        "symbols": ["BTCUSDT"],
        "strategy_id": 1,
    }
    data.update(extra)
    path.write_text(json.dumps(data))
    return path


class Ctx:
    def __init__(
        self, ac: AsyncClient, state: StateManager, jobs: BacktestJobManager, tmp: Path
    ) -> None:
        self.ac = ac
        self.state = state
        self.jobs = jobs
        self.tmp = tmp
        self.config_dir = tmp / "paper_configs"
        self.launched: list[list[str]] = []
        self.strategy_id = state.create_strategy("test_strategy", DSL)
        self.isolated_run = self._completed_run(ISOLATED_PARAMS)
        self.off_run = self._completed_run(OFF_PARAMS)
        self.cross_run = self._completed_run(CROSS_PARAMS)

    def _completed_run(self, parameters: dict[str, Any]) -> int:
        rid = self.state.create_backtest_run(
            strategy_id=self.strategy_id,
            run_mode="validation",
            symbols=["BTCUSDT"],
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-12-31",
            parameters=dict(parameters),
        )
        self.state.update_backtest_run_status(rid, "completed")
        return rid

    def config_path(self, run_id: int) -> Path:
        return self.config_dir / f"paper_{run_id}.json"

    async def launch(self, payload: dict[str, Any]) -> Path:
        """POST /paper/start and return the config JSON the paper CLI would read."""
        r = await self.ac.post("/api/paper/start", json=payload)
        assert r.status_code == 201, r.text
        return self.config_path(r.json()["run_id"])


@pytest.fixture()
async def ctx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Ctx]:
    db = tmp_path / "api.db"
    monkeypatch.setattr("vibe_quant.db.connection.DEFAULT_DB_PATH", db)
    monkeypatch.setattr(
        "vibe_quant.api.routers.paper_trading.PAPER_CONFIG_DIR", tmp_path / "paper_configs"
    )

    app = create_app()
    state = StateManager(db_path=db)
    _ = state.conn
    jobs = BacktestJobManager(db_path=db)
    ws = ConnectionManager()
    await ws.start()
    app.state.state_manager = state
    app.state.job_manager = jobs
    app.state.ws_manager = ws
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        c = Ctx(ac, state, jobs, tmp_path)

        def fake_start_job(
            run_id: int, job_type: str, command: list[str], log_file: str | None = None
        ) -> int:
            c.launched.append(command)
            return 4242

        monkeypatch.setattr(jobs, "start_job", fake_start_job)
        yield c
    await ws.stop()
    jobs.close()
    state.close()




def test_loader_maps_absent_or_null_to_derive_and_keeps_both_explicit_types(
    tmp_path: Path,
) -> None:
    """Absent/null -> None (derive); ``CROSSED``/``ISOLATED`` are kept."""
    for explicit in ("CROSSED", "ISOLATED"):
        path = _write_config(tmp_path / f"{explicit}.json", margin_type=explicit)
        assert load_config_from_json(path).margin_type == explicit
    assert load_config_from_json(_write_config(tmp_path / "absent.json")).margin_type is None
    null_config = load_config_from_json(_write_config(tmp_path / "null.json", margin_type=None))
    assert null_config.margin_type is None


@pytest.mark.parametrize("written", ["None", "crossed", 5, True, ""])
def test_loader_refuses_a_margin_type_it_cannot_trust(tmp_path: Path, written: object) -> None:
    """Anything that is not CROSSED/ISOLATED/null fails the load closed."""
    with pytest.raises(ConfigurationError, match="margin_type"):
        load_config_from_json(_write_config(tmp_path / "bad.json", margin_type=written))


def test_validate_accepts_derive_and_still_rejects_an_unknown_margin_type() -> None:
    """``validate()`` accepts ``None`` (derive) but not a garbage string."""

    def _config(margin_type: str | None) -> PaperTradingConfig:
        return PaperTradingConfig(
            trader_id="PAPER-001",
            binance=BinanceTestnetConfig(api_key="k", api_secret="s"),
            symbols=["BTCUSDT"],
            strategy_id=1,
            margin_type=margin_type,
        )

    assert _config(None).validate() == []
    assert _config("ISOLATED").validate() == []
    assert any("margin_type" in e for e in _config("crossed").validate())


def test_save_writes_null_for_derive_and_round_trips_both_values(tmp_path: Path) -> None:
    """``save_config_to_json`` writes null and reads it back as ``None``."""
    config = PaperTradingConfig(
        trader_id="PAPER-001",
        binance=BinanceTestnetConfig(api_key="k", api_secret="s"),
        symbols=["BTCUSDT"],
        strategy_id=1,
    )
    path = tmp_path / "derive.json"
    save_config_to_json(config, path)
    assert json.loads(path.read_text())["margin_type"] is None
    assert load_config_from_json(path).margin_type is None

    config.margin_type = "ISOLATED"
    save_config_to_json(config, path)
    assert json.loads(path.read_text())["margin_type"] == "ISOLATED"
    assert load_config_from_json(path).margin_type == "ISOLATED"




async def test_api_launch_of_isolated_validation_run_trades_isolated_end_to_end(
    ctx: Ctx, fake_tradingnode: type[_FakeTradingNode]
) -> None:
    """API start -> config JSON -> loader -> resolve -> venue ISOLATED."""
    from nautilus_trader.adapters.binance.common.symbol import BinanceSymbol
    from nautilus_trader.adapters.binance.futures.enums import BinanceFuturesMarginType

    path = await ctx.launch({"validation_run_id": ctx.isolated_run})
    written = json.loads(path.read_text())
    assert written["validation_run_id"] == ctx.isolated_run
    assert "margin_type" not in written, "the API must not pin a margin type on the launch path"

    config = load_config_from_json(path)
    assert config.margin_type is None, "an absent margin_type means 'derive from the run'"

    node = PaperTradingNode(config)
    await node._initialize()
    try:
        resolved = node.resolved_config
        assert resolved is not None
        assert resolved.margin_type == "ISOLATED"
        assert resolved.diff == [], "a paper launch of the very run it validated must not diff"

        exec_cfg = fake_tradingnode.instances[0].config.exec_clients["BINANCE"]
        assert exec_cfg.futures_margin_types == {
            BinanceSymbol("BTCUSDT"): BinanceFuturesMarginType.ISOLATED
        }
    finally:
        await node._shutdown()


async def test_cross_mode_validation_resolves_crossed_without_a_liquidation_entry(
    ctx: Ctx, fake_tradingnode: type[_FakeTradingNode]
) -> None:
    """Mode cross -> CROSSED, its margin matches, so no diff entry at all."""
    path = await ctx.launch({"validation_run_id": ctx.cross_run})
    node = PaperTradingNode(load_config_from_json(path))
    await node._initialize()
    try:
        resolved = node.resolved_config
        assert resolved is not None
        assert resolved.margin_type == "CROSSED"
        assert resolved.diff == []
    finally:
        await node._shutdown()


async def test_launch_without_a_validation_run_stays_crossed(
    ctx: Ctx, fake_tradingnode: type[_FakeTradingNode]
) -> None:
    """With nothing to derive from, the margin type is CROSSED (no validation metadata)."""
    from nautilus_trader.adapters.binance.common.symbol import BinanceSymbol
    from nautilus_trader.adapters.binance.futures.enums import BinanceFuturesMarginType

    path = await ctx.launch({"strategy_id": ctx.strategy_id, "symbols": ["BTCUSDT"]})
    node = PaperTradingNode(load_config_from_json(path))
    await node._initialize()
    try:
        resolved = node.resolved_config
        assert resolved is not None
        assert resolved.margin_type == "CROSSED"
        assert resolved.validated is None
        assert resolved.diff == []
        exec_cfg = fake_tradingnode.instances[0].config.exec_clients["BINANCE"]
        assert exec_cfg.futures_margin_types == {
            BinanceSymbol("BTCUSDT"): BinanceFuturesMarginType.CROSS
        }
    finally:
        await node._shutdown()




async def test_explicit_margin_type_override_is_used_and_diffed(
    ctx: Ctx, fake_tradingnode: type[_FakeTradingNode]
) -> None:
    """An explicit override differing from validation is used and named in the diff."""
    from nautilus_trader.adapters.binance.common.symbol import BinanceSymbol
    from nautilus_trader.adapters.binance.futures.enums import BinanceFuturesMarginType

    path = await ctx.launch({"validation_run_id": ctx.isolated_run})
    data = json.loads(path.read_text())
    data["margin_type"] = "CROSSED"
    path.write_text(json.dumps(data))

    config = load_config_from_json(path)
    assert config.margin_type == "CROSSED", "an explicit type is an override, not a hint"

    node = PaperTradingNode(config)
    await node._initialize()
    try:
        resolved = node.resolved_config
        assert resolved is not None
        assert resolved.margin_type == "CROSSED"
        assert [d for d in resolved.diff if d["field"] == "margin_type"] == [
            {"field": "margin_type", "validated": "ISOLATED", "paper": "CROSSED"}
        ]
        exec_cfg = fake_tradingnode.instances[0].config.exec_clients["BINANCE"]
        assert exec_cfg.futures_margin_types == {
            BinanceSymbol("BTCUSDT"): BinanceFuturesMarginType.CROSS
        }
    finally:
        await node._shutdown()




async def test_off_mode_validation_names_unverified_liquidation_before_fee_schedule(
    ctx: Ctx, fake_tradingnode: type[_FakeTradingNode]
) -> None:
    """Mode off -> exactly one liquidation entry, after symbols, before fee_schedule."""
    path = await ctx.launch({"validation_run_id": ctx.off_run})
    node = PaperTradingNode(load_config_from_json(path))
    await node._initialize()
    try:
        resolved = node.resolved_config
        assert resolved is not None
        assert resolved.margin_type == "CROSSED"
        assert [d["field"] for d in resolved.diff] == ["liquidation", "fee_schedule"]
        assert resolved.diff[0] == LIQ_OFF_ENTRY
        assert resolved.diff[1]["paper"] == "exchange commission (Binance live fee model)"
    finally:
        await node._shutdown()


async def test_off_mode_with_an_explicit_isolated_override_names_isolated_liquidation(
    ctx: Ctx, fake_tradingnode: type[_FakeTradingNode]
) -> None:
    """The off-mode note must name the *resolved* margin type, not CROSSED.

    An explicit ``ISOLATED`` override on an off-mode run makes paper apply the
    exchange's isolated-margin liquidation, so the unverified-parity entry has to
    read ``ISOLATED``. Regression against hard-coding the R43 text to CROSSED:
    here the derived value would be CROSSED and only the override differs.
    """
    path = await ctx.launch({"validation_run_id": ctx.off_run})
    data = json.loads(path.read_text())
    data["margin_type"] = "ISOLATED"
    path.write_text(json.dumps(data))

    config = load_config_from_json(path)
    assert config.margin_type == "ISOLATED", "an explicit type is an override, not a hint"

    node = PaperTradingNode(config)
    await node._initialize()
    try:
        resolved = node.resolved_config
        assert resolved is not None
        assert resolved.margin_type == "ISOLATED"
        # margin_type entry, then the R43 liquidation entry after symbols and
        # before fee_schedule.
        assert [d["field"] for d in resolved.diff] == [
            "margin_type",
            "liquidation",
            "fee_schedule",
        ]
        assert resolved.diff[0] == {
            "field": "margin_type",
            "validated": "CROSSED",
            "paper": "ISOLATED",
        }
        assert resolved.diff[1] == {
            "field": "liquidation",
            "validated": "off",
            "paper": "exchange liquidation (ISOLATED)",
            "note": (
                "validation did not simulate liquidation; paper liquidation "
                "parity is unverified"
            ),
        }
    finally:
        await node._shutdown()
