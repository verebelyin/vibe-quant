"""PaperTradingNode integration (vibe-quant-e70tl.4 / .19 / .21).

Covers: validated-run params + leverage reach the live config (empty diff),
external order claims, command-queue control (close-all errors surface),
kill-switch polling, restart risk-state restore, and venue order rejection ->
ErrorHandler -> halt + Telegram alert through a real NT BacktestEngine.
"""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.objects import Currency, Money

from tests.unit.test_audit_paper_guard import (
    _ENGINES,
    NaiveConfig,
    NaiveStrategy,
    _flat_bars,
    _new_engine,
)
from vibe_quant.data.catalog import create_instrument, get_bar_type
from vibe_quant.db.state_manager import StateManager
from vibe_quant.nt_compat import retain_log_guard
from vibe_quant.paper.config import (
    BinanceTestnetConfig,
    ConfigurationError,
    PaperTradingConfig,
    SizingModuleConfig,
)
from vibe_quant.paper.guard import GuardState, HaltReason, RiskState
from vibe_quant.paper.node import NodeState, PaperTradingNode
from vibe_quant.paper.persistence import PaperCommandQueue, StateCheckpoint, StatePersistence
from vibe_quant.reconciliation import load_trades

if TYPE_CHECKING:
    from pathlib import Path

    from nautilus_trader.model.data import Bar

DSL = {
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


@pytest.fixture(autouse=True)
def _dispose_engines() -> Any:
    yield
    while _ENGINES:
        eng = _ENGINES.pop()
        retain_log_guard(eng)
        eng.dispose()


@pytest.fixture()
def seeded(tmp_path: Path) -> dict[str, Any]:
    db = tmp_path / "state.db"
    sm = StateManager(db)
    sid = sm.create_strategy("test_strategy", DSL)
    other = sm.create_strategy("other_strategy", {**DSL, "name": "other_strategy"})
    rid = sm.create_backtest_run(
        strategy_id=sid,
        run_mode="validation",
        symbols=["BTCUSDT"],
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-12-31",
        parameters={"rsi_14_period": 21, "take_profit_risk_reward": 3.0, "leverage": 5},
    )
    sm.update_backtest_run_status(rid, "completed")
    pending = sm.create_backtest_run(
        strategy_id=sid,
        run_mode="validation",
        symbols=["BTCUSDT"],
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-12-31",
        parameters={},
    )
    sm.close()
    return {"db": db, "sid": sid, "other": other, "rid": rid, "pending": pending, "tmp": tmp_path}


def _config(seeded: dict[str, Any], **kw: Any) -> PaperTradingConfig:
    base: dict[str, Any] = {
        "trader_id": "PAPER-007",
        "binance": BinanceTestnetConfig("key", "secret", testnet=True),
        "strategy_id": seeded["sid"],
        "validation_run_id": seeded["rid"],
        "db_path": seeded["db"],
        "logs_path": seeded["tmp"] / "logs",
    }
    base.update(kw)
    return PaperTradingConfig(**base)


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


@pytest.fixture()
def fake_tradingnode(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTradingNode]:
    import nautilus_trader.live.node as live_node

    _FakeTradingNode.instances = []
    monkeypatch.setattr(live_node, "TradingNode", _FakeTradingNode)
    return _FakeTradingNode


# --------------------------------------------------------------- e70tl.19


async def test_validated_run_params_and_leverage_reach_live_config(
    seeded: dict[str, Any], fake_tradingnode: type[_FakeTradingNode]
) -> None:
    from nautilus_trader.adapters.binance.common.enums import BinanceEnvironment
    from nautilus_trader.adapters.binance.common.symbol import BinanceSymbol
    from nautilus_trader.adapters.binance.futures.enums import BinanceFuturesMarginType

    node = PaperTradingNode(_config(seeded))
    await node._initialize()
    try:
        resolved = node.resolved_config
        assert resolved is not None
        assert resolved.symbols == ["BTCUSDT"]
        assert resolved.strategy_params == {"rsi_14_period": 21, "take_profit_risk_reward": 3.0}
        assert resolved.leverage == 5
        assert resolved.diff == [], "paper from a validated run must trade it exactly"

        strat: Any = node._compiled_strategies[0]
        assert strat.config.rsi_14_period == 21
        assert strat.config.take_profit_risk_reward == 3.0
        # Compiled defaults untouched where validation used defaults.
        assert strat.config.risk_per_trade == 0.02
        assert strat.config.execution_delay_probability == 0.0
        assert strat.external_order_claims == [InstrumentId.from_str("BTCUSDT-PERP.BINANCE")]

        tn = fake_tradingnode.instances[0]
        exec_cfg = tn.config.exec_clients["BINANCE"]
        assert exec_cfg.environment == BinanceEnvironment.TESTNET
        assert exec_cfg.futures_leverages == {BinanceSymbol("BTCUSDT"): 5}
        assert exec_cfg.futures_margin_types == {
            BinanceSymbol("BTCUSDT"): BinanceFuturesMarginType.CROSS
        }
        assert tn.trader.actors == [node.guard], "guard actor registered before strategies"
        assert tn.trader.strategies == node._compiled_strategies

        assert node._event_writer is not None
        node._event_writer.flush()
        start = [
            json.loads(line)
            for line in (seeded["tmp"] / "logs" / "PAPER-007.jsonl").read_text().splitlines()
            if '"action":"start"' in line
        ][0]
        assert start["data"]["config_diff"] == []
        assert start["data"]["leverage"] == 5
    finally:
        await node._shutdown()


async def test_sizing_override_shows_in_config_diff(
    seeded: dict[str, Any], fake_tradingnode: type[_FakeTradingNode]
) -> None:
    cfg = _config(seeded, sizing=SizingModuleConfig(risk_per_trade=Decimal("0.01")), leverage=3)
    node = PaperTradingNode(cfg)
    await node._initialize()
    try:
        assert node.resolved_config is not None
        diff = {d["field"]: (d["validated"], d["paper"]) for d in node.resolved_config.diff}
        assert diff == {
            "strategy_params.risk_per_trade": (None, 0.01),
            "leverage": (5.0, 3),
        }
    finally:
        await node._shutdown()


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"strategy_id": "other"}, "is for strategy"),
        ({"validation_run_id": "pending"}, "not completed"),
        ({"validation_run_id": 99999}, "not found"),
    ],
)
async def test_validation_run_mismatch_refuses_to_start(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    override: dict[str, Any],
    match: str,
) -> None:
    kw = {k: seeded[v] if isinstance(v, str) else v for k, v in override.items()}
    node = PaperTradingNode(_config(seeded, **kw))
    try:
        with pytest.raises(ConfigurationError, match=match):
            await node._initialize()
    finally:
        await node._shutdown()


async def test_kill_switch_blocks_start(
    seeded: dict[str, Any], fake_tradingnode: type[_FakeTradingNode]
) -> None:
    sm = StateManager(seeded["db"])
    sm.set_kill_switch("audit test")
    sm.close()
    node = PaperTradingNode(_config(seeded))
    try:
        with pytest.raises(ConfigurationError, match="kill switch"):
            await node._initialize()
    finally:
        await node._shutdown()
    assert fake_tradingnode.instances == []


async def test_restart_restores_risk_state_from_checkpoint(
    seeded: dict[str, Any], fake_tradingnode: type[_FakeTradingNode]
) -> None:
    persistence = StatePersistence(db_path=seeded["db"], trader_id="PAPER-007")
    persistence.save_checkpoint(
        StateCheckpoint(
            trader_id="PAPER-007",
            positions={"BTCUSDT-PERP.BINANCE-TestStrategyStrategy-000": {"side": "LONG"}},
            node_status={
                "state": "running",
                "risk": RiskState(high_water_mark=Decimal("10600"), consecutive_losses=3).to_dict(),
            },
        )
    )
    persistence.close()
    node = PaperTradingNode(_config(seeded))
    await node._initialize()
    try:
        guard = node.guard
        assert guard is not None
        assert guard.risk_state.high_water_mark == Decimal("10600")
        assert guard.risk_state.consecutive_losses == 3
    finally:
        await node._shutdown()


# --------------------------------------------------------------- commands


class _FakeGuard:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.guard_state = GuardState.RUNNING
        self.halt_reason: HaltReason | None = None
        self.close_errors: list[str] = []

    def halt(self, reason: HaltReason, message: str) -> bool:
        self.calls.append(("halt", reason))
        self.guard_state = GuardState.HALTED
        self.halt_reason = reason
        return True

    def pause(self) -> bool:
        self.calls.append(("pause", None))
        return True

    def resume(self, *, kill_switch_engaged: bool = False) -> tuple[bool, str]:
        self.calls.append(("resume", kill_switch_engaged))
        if kill_switch_engaged:
            return False, "system kill switch is engaged; unlock it before resuming"
        return True, "resumed"

    def close_all(self) -> Any:
        from vibe_quant.paper.guard import CloseAllResult

        self.calls.append(("close_all", None))
        return CloseAllResult(targeted_positions=["P-1"], errors=list(self.close_errors))

    def status(self) -> dict[str, object]:
        return {"risk": {}, "state": self.guard_state.value}


class _Pos:
    def __init__(self, pid: str) -> None:
        self.id = pid


class _Cache:
    def __init__(self) -> None:
        self.open: list[_Pos] = []

    def positions_open(self) -> list[_Pos]:
        return list(self.open)

    def orders_open(self) -> list[object]:
        return []

    def accounts(self) -> list[object]:
        return []


class _RuntimeNode:
    def __init__(self) -> None:
        self.cache = _Cache()


def _node_with_fake_guard(seeded: dict[str, Any]) -> tuple[PaperTradingNode, _FakeGuard]:
    node = PaperTradingNode(_config(seeded))
    guard = _FakeGuard()
    node._guard = guard  # type: ignore[assignment]
    node._trading_node = _RuntimeNode()  # type: ignore[assignment]
    node._commands = PaperCommandQueue(seeded["db"])
    return node, guard


async def test_command_queue_halt_pause_resume(seeded: dict[str, Any]) -> None:
    node, guard = _node_with_fake_guard(seeded)
    q = PaperCommandQueue(seeded["db"])
    try:
        ids = [q.enqueue("PAPER-007", c) for c in ("pause", "resume", "halt")]
        other = q.enqueue("PAPER-999", "halt")  # different session: untouched
        await node.process_commands()
        assert [c[0] for c in guard.calls] == ["pause", "resume", "halt"]
        assert guard.calls[2] == ("halt", HaltReason.MANUAL)
        for cid in ids:
            cmd = q.get(cid)
            assert cmd is not None and cmd.status == "done", cmd
        other_cmd = q.get(other)
        assert other_cmd is not None and other_cmd.status == "pending"
    finally:
        q.close()
        await node._shutdown()


async def test_close_all_errors_surface_in_command_result(seeded: dict[str, Any]) -> None:
    node, guard = _node_with_fake_guard(seeded)
    assert node._trading_node is not None
    node._trading_node.cache.open = [_Pos("BTCUSDT-PERP.BINANCE-S-000")]  # type: ignore[attr-defined]
    guard.close_errors = ["S-000: RuntimeError: venue down"]
    q = PaperCommandQueue(seeded["db"])
    try:
        cid = q.enqueue("PAPER-007", "close_all", {"wait_secs": 0.3})
        await node.process_commands()
        cmd = q.get(cid)
        assert cmd is not None and cmd.status == "failed"
        assert cmd.error is not None
        assert "venue down" in cmd.error
        assert "still open" in cmd.error
        assert cmd.result is not None
        assert cmd.result["still_open"] == ["BTCUSDT-PERP.BINANCE-S-000"]
        assert cmd.result["targeted_positions"] == ["P-1"]
    finally:
        q.close()
        await node._shutdown()


async def test_close_all_success_when_flat(seeded: dict[str, Any]) -> None:
    node, _guard = _node_with_fake_guard(seeded)
    q = PaperCommandQueue(seeded["db"])
    try:
        cid = q.enqueue("PAPER-007", "close_all", {"wait_secs": 0.3})
        await node.process_commands()
        cmd = q.get(cid)
        assert cmd is not None and cmd.status == "done", cmd
        assert cmd.result is not None and cmd.result["still_open"] == []
    finally:
        q.close()
        await node._shutdown()


async def test_kill_switch_is_polled_and_blocks_resume(seeded: dict[str, Any]) -> None:
    node, guard = _node_with_fake_guard(seeded)
    sm = StateManager(seeded["db"])
    try:
        node._check_kill_switch()
        assert guard.calls == []
        sm.set_kill_switch("ops kill")
        node._check_kill_switch()
        assert guard.calls == [("halt", HaltReason.KILL_SWITCH)]
        node._check_kill_switch()  # already killed: no repeat
        assert guard.calls == [("halt", HaltReason.KILL_SWITCH)]
        ok, msg = node.resume()
        assert not ok and "kill switch" in msg
        assert guard.calls[-1] == ("resume", True)
        sm.clear_kill_switch("ops")
        ok, _ = node.resume()
        assert ok
    finally:
        sm.close()
        await node._shutdown()


# ------------------------------------------- real engine: rejection -> alert


class _FakeTelegram:
    def __init__(self) -> None:
        self.errors: list[tuple[str, bool]] = []
        self.breakers: list[tuple[str, bool]] = []

    async def send_error(self, message: str, *, bypass_rate_limit: bool = False) -> bool:
        self.errors.append((message, bypass_rate_limit))
        return True

    async def send_circuit_breaker(
        self, reason: str, details: str = "", *, bypass_rate_limit: bool = False
    ) -> bool:
        self.breakers.append((reason, bypass_rate_limit))
        return True

    async def close(self) -> None:
        return None


class _RejectOnce(NaiveStrategy):
    """Holds a long, then sends a reduce-only BUY the venue must reject."""

    def on_bar(self, bar: Bar) -> None:
        if not self.entered_once:
            super().on_bar(bar)
            return
        if getattr(self, "_rejected_sent", False):
            return
        self._rejected_sent = True
        self.submit_order(
            self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=OrderSide.BUY,
                quantity=self.instrument.make_qty(Decimal("0.100")),
                reduce_only=True,
            )
        )


def _engine_node(seeded: dict[str, Any], strategy: NaiveStrategy) -> tuple[Any, PaperTradingNode]:
    from vibe_quant.logging.writer import EventWriter

    engine = _new_engine()
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(10_000, Currency.from_str("USDT"))],
        default_leverage=Decimal("10"),
        bar_execution=True,
    )
    engine.add_instrument(create_instrument("BTCUSDT"))
    node = PaperTradingNode(_config(seeded))
    node._telegram = _FakeTelegram()  # type: ignore[assignment]
    node._event_writer = EventWriter(run_id="PAPER-007", base_path=seeded["tmp"] / "logs")
    node._compiled_strategies = [strategy]
    guard = node._build_guard(engine.kernel.risk_engine)
    node._guard = guard
    engine.add_actor(guard)
    engine.add_strategy(strategy)
    return engine, node


async def test_order_rejection_halts_and_sends_telegram_alert(seeded: dict[str, Any]) -> None:
    strat = _RejectOnce(
        NaiveConfig(
            instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
            bar_type=get_bar_type("BTCUSDT", "1m"),
            mode="once",
            trade_size="0.100",
        )
    )
    engine, node = _engine_node(seeded, strat)
    engine.add_data(_flat_bars(get_bar_type("BTCUSDT", "1m"), [10000.0] * 6))
    engine.run()
    await asyncio.sleep(0)  # let fire-and-forget alert tasks run
    tg: Any = node._telegram
    try:
        assert any("order rejected" in m for m, _ in tg.errors), tg.errors
        # Fatal rejection -> ERROR halt -> critical (rate-limit-bypassing) breaker alert.
        assert node.status.state == NodeState.HALTED
        assert node.status.halt_reason == HaltReason.ERROR
        assert any("HALT (error)" in m and bypass for m, bypass in tg.breakers), tg.breakers
        assert engine.cache.positions_open() == [], "halt flattened the open long"
    finally:
        assert node._event_writer is not None
        node._event_writer.close()
        node._event_writer = None
        await node._shutdown()


async def test_paper_position_events_feed_reconciliation(seeded: dict[str, Any]) -> None:
    """Paper POSITION_OPEN/CLOSE events carry ids + venue timestamps -> load_trades works."""
    strat = NaiveStrategy(
        NaiveConfig(
            instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
            bar_type=get_bar_type("BTCUSDT", "1m"),
            mode="once",
            trade_size="0.100",
            tp_pct=0.01,
        )
    )
    engine, node = _engine_node(seeded, strat)
    engine.add_data(_flat_bars(get_bar_type("BTCUSDT", "1m"), [10000.0, 10000.0, 10200.0, 10200.0]))
    engine.run()
    assert node._event_writer is not None
    node._event_writer.close()
    node._event_writer = None
    trades = load_trades("PAPER-007", base_path=seeded["tmp"] / "logs")
    try:
        assert len(trades) == 1
        t = trades[0]
        assert t.symbol == "BTCUSDT-PERP.BINANCE"
        assert t.side == "LONG"
        assert t.entry_price == 10000.0
        assert t.exit_price == 10100.0
        assert t.exit_reason == "take_profit"
        # Venue (simulated) time, not wall-clock time.
        assert t.entry_time.year == 2026 and t.entry_time.month == 1
        # 0.1 BTC * +100 = +10 gross; fees 0.0005*1000 + 0.0002*1010 = 0.702
        assert t.gross_pnl == pytest.approx(10.0)
        assert t.net_pnl == pytest.approx(10.0 - 0.5 - 0.202)
    finally:
        await node._shutdown()
