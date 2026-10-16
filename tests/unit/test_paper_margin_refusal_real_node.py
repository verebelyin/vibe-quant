"""R42 on the REAL NautilusTrader ``TradingNode`` (bead vibe-quant-q7hcn).

This file is R42's ONLY test: the fake-node version in
``test_paper_startup_and_estimate.py`` modelled a ``ConnectionError`` real NT never
raises and was removed (bead vibe-quant-qf58o). Here a real ``TradingNode`` (no
network) runs a stub exec client whose ``_connect`` fails the way
``BinanceFuturesExecutionClient._connect`` does when the venue refuses the margin-type
change (``set_margin_type`` raises a ``BinanceClientError`` out of its TaskGroup).

What real NT does (probed): ``LiveExecutionClient.connect`` runs ``_connect`` in a task
and only LOGS its exception; ``start_async`` waits ``timeout_connection`` for the engines
to connect and RETURNS without ``trader.start()``, and ``run_async`` keeps idling (no raise). So the guarantee
is "no strategy starts, no order exists", while the failure is silent - that is why
R42b's startup-connect timeout alert exists.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from typing import TYPE_CHECKING, Any

import pytest
from nautilus_trader.adapters.binance.http.error import BinanceClientError
from nautilus_trader.cache.cache import Cache  # noqa: TCH002
from nautilus_trader.common.component import (
    LiveClock,  # noqa: TCH002
    MessageBus,  # noqa: TCH002
)
from nautilus_trader.common.config import PositiveFloat  # noqa: TCH002
from nautilus_trader.common.providers import InstrumentProvider
from nautilus_trader.config import LoggingConfig
from nautilus_trader.execution.config import ExecEngineConfig  # noqa: TCH002
from nautilus_trader.live.config import (
    LiveExecClientConfig,
    LiveExecEngineConfig,
    TradingNodeConfig,
)
from nautilus_trader.live.data_client import LiveMarketDataClient
from nautilus_trader.live.execution_client import LiveExecutionClient
from nautilus_trader.live.factories import LiveDataClientFactory, LiveExecClientFactory
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import AccountId, ClientId, Venue
from nautilus_trader.trading.strategy import Strategy

from tests.unit.test_audit_paper_node import DSL, _config
from vibe_quant.db.state_manager import StateManager
from vibe_quant.logging.events import EventType
from vibe_quant.nt_compat import retain_log_guard
from vibe_quant.paper.node import PaperTradingNode

if TYPE_CHECKING:
    from pathlib import Path

#: Short so the refused-connect case (where NT waits the whole timeout) stays fast.
_CONNECT_TIMEOUT_S = 1.0
_VENUE = Venue("BINANCE")


class _StubExecClient(LiveExecutionClient):
    """Exec client whose ``_connect`` raises like a refused margin-type change."""

    refuse = True

    def __init__(self, loop: Any, msgbus: MessageBus, cache: Cache, clock: LiveClock) -> None:
        super().__init__(
            loop=loop,
            client_id=ClientId("BINANCE"),
            venue=_VENUE,
            oms_type=OmsType.HEDGING,
            account_type=AccountType.MARGIN,
            base_currency=USDT,
            instrument_provider=InstrumentProvider(),
            msgbus=msgbus,
            cache=cache,
            clock=clock,
        )
        self.connect_attempts = 0

    async def _connect(self) -> None:
        self.connect_attempts += 1
        if _StubExecClient.refuse:
            # Binance -4046-style 400: "No need to change margin type" is benign, but
            # any refused change (open position / orders) raises out of the TaskGroup.
            raise BinanceClientError(  # type: ignore[no-untyped-call]
                400, {"code": -4048, "msg": "Margin type cannot be changed"}, {}
            )
        # Healthy venue: register the account so the exec engine counts as connected.
        self._set_account_id(AccountId("BINANCE-001"))

    async def _disconnect(self) -> None:
        return


class _StubExecFactory(LiveExecClientFactory):
    created: list[_StubExecClient] = []

    @staticmethod
    def create(  # type: ignore[override]
        loop: Any,
        name: str,
        config: LiveExecClientConfig,
        msgbus: MessageBus,
        cache: Cache,
        clock: LiveClock,
    ) -> _StubExecClient:
        client = _StubExecClient(loop, msgbus, cache, clock)
        _StubExecFactory.created.append(client)
        return client


class _RecordingStrategy(Strategy):  # type: ignore[misc]
    started = 0

    def on_start(self) -> None:
        type(self).started += 1


def _build_node() -> TradingNode:
    config = TradingNodeConfig(
        trader_id="PAPER-007",
        timeout_connection=_CONNECT_TIMEOUT_S,
        timeout_disconnection=1.0,
        timeout_post_stop=1.0,
        timeout_shutdown=1.0,
        logging=LoggingConfig(log_level="ERROR"),
        exec_engine=LiveExecEngineConfig(reconciliation=False),
        exec_clients={"BINANCE": LiveExecClientConfig()},
    )
    node = TradingNode(config)
    # Same order as PaperTradingNode._build_node: strategies before build().
    node.trader.add_strategy(_RecordingStrategy())
    node.add_exec_client_factory("BINANCE", _StubExecFactory)
    node.build()
    return node


async def _run_node(node: TradingNode, *, until_started: bool) -> None:
    """Run like ``PaperTradingNode._run_loop``: ``run_async`` as a task on our loop.

    Ends like ``PaperTradingNode._shutdown``: ``stop_async`` and no ``dispose()``
    (it stops the running loop; node.py skips it for the same reason).
    """
    run_task = asyncio.create_task(node.run_async())
    try:
        if until_started:
            # Healthy venue: run_async never returns (engine queues); wait for the start.
            for _ in range(200):
                if _RecordingStrategy.started or run_task.done():
                    break
                await asyncio.sleep(0.05)
        else:
            # Refused connect: NT logs the error, gives up on the start after
            # timeout_connection and run_async keeps idling (no raise, no trader).
            await asyncio.sleep(_CONNECT_TIMEOUT_S + 1.5)
            assert not run_task.done(), "run_async idles; it neither raises nor returns"
    finally:
        await node.stop_async()
        if not run_task.done():
            run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)
        # Rust logger inits once per process: keep the guard alive (nautilus-gotchas).
        retain_log_guard(node)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _reset() -> None:
    _RecordingStrategy.started = 0
    _StubExecFactory.created = []
    _StubExecClient.refuse = True


async def test_refused_margin_change_never_starts_trader() -> None:
    """R42 (real NT): ``_connect`` raises -> ``trader.start()`` never runs."""
    node = _build_node()
    trader = node.trader
    await _run_node(node, until_started=False)
    client = _StubExecFactory.created[0]
    assert client.connect_attempts == 1, "the stub exec client must really be connected"
    assert _RecordingStrategy.started == 0, "a refused connect must not start any strategy"
    assert client.is_connected is False, "the refused connect must leave the client down"
    assert trader.is_running is False
    assert all(state != "RUNNING" for state in trader.strategy_states().values())


async def test_healthy_connect_does_start_trader() -> None:
    """Teeth: with the same stub NOT raising, the strategy DOES start."""
    _StubExecClient.refuse = False
    node = _build_node()
    await _run_node(node, until_started=True)
    assert _StubExecFactory.created[0].connect_attempts == 1
    assert _RecordingStrategy.started == 1, "the control run must reach trader.start()"


# ------------------------------------------------- R42c: through PaperTradingNode


#: Its own trader id: checkpoints and the event log are keyed by it.
_LIVE_TRADER_ID = "PAPER-008"


class _StubDataClient(LiveMarketDataClient):
    """Minimal no-op market data client: the test must not touch the network."""

    def __init__(self, loop: Any, msgbus: MessageBus, cache: Cache, clock: LiveClock) -> None:
        super().__init__(
            loop=loop,
            client_id=ClientId("BINANCE"),
            venue=_VENUE,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            instrument_provider=InstrumentProvider(),
        )

    async def _connect(self) -> None:
        return

    async def _disconnect(self) -> None:
        return

    async def _subscribe(self, command: Any) -> None:
        return

    async def _unsubscribe(self, command: Any) -> None:
        return

    async def _request(self, request: Any) -> None:
        return


class _StubDataFactory(LiveDataClientFactory):
    @staticmethod
    def create(  # type: ignore[override]
        loop: Any,
        name: str,
        config: Any,
        msgbus: MessageBus,
        cache: Cache,
        clock: LiveClock,
    ) -> _StubDataClient:
        return _StubDataClient(loop, msgbus, cache, clock)


class _FastNodeConfig(TradingNodeConfig):
    """Bounded NT waits (its defaults run to 60 s) and no reconciliation.

    ``PaperTradingNode._create_live_trading_node`` imports ``TradingNodeConfig``
    at call time, so patching the ``nautilus_trader.live.config`` name redirects
    the REAL build path. Reconciliation is off for the same reason R42's own
    real-node config turns it off: the stub has no venue reports to serve.
    """

    timeout_connection: PositiveFloat = _CONNECT_TIMEOUT_S
    timeout_disconnection: PositiveFloat = _CONNECT_TIMEOUT_S
    timeout_post_stop: PositiveFloat = _CONNECT_TIMEOUT_S
    timeout_shutdown: PositiveFloat = _CONNECT_TIMEOUT_S
    exec_engine: ExecEngineConfig | None = LiveExecEngineConfig(reconciliation=False)


@pytest.fixture()
def seeded(tmp_path: Path) -> dict[str, Any]:
    """A strategy plus a completed validation run (liquidation off, leverage 5)."""
    db = tmp_path / "state.db"
    sm = StateManager(db)
    sid = sm.create_strategy("test_strategy", DSL)
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
    sm.close()
    return {"db": db, "sid": sid, "rid": rid, "tmp": tmp_path}


async def test_paper_node_refused_connect_starts_nothing_and_shuts_down_clean(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R42c: the refused margin change through the REAL ``PaperTradingNode`` lifecycle.

    ``_initialize -> _run_loop -> _shutdown`` (the paper CLI's ``start()`` path) with
    the exec client refusing the way ``BinanceFuturesExecutionClient._connect`` does
    on a refused margin-type change. Guarantees: (a) the compiled strategy never
    starts, (b) no order exists in the NT cache and no order row / ORDER event is
    written, (c) shutdown completes within a bounded time with nothing left running.
    """
    import nautilus_trader.adapters.binance.factories as binance_factories

    monkeypatch.setattr(binance_factories, "BinanceLiveExecClientFactory", _StubExecFactory)
    monkeypatch.setattr(binance_factories, "BinanceLiveDataClientFactory", _StubDataFactory)
    monkeypatch.setattr("nautilus_trader.live.config.TradingNodeConfig", _FastNodeConfig)

    async def _idle_watch() -> None:
        # The report-only liquidation watch fetches Binance over HTTP (R47); out of scope.
        await asyncio.Event().wait()

    node = PaperTradingNode(_config(seeded, trader_id=_LIVE_TRADER_ID))
    monkeypatch.setattr(node, "_liquidation_loop", _idle_watch)
    strategy_starts: list[str] = []

    await node._initialize()
    trading_node: Any = node._trading_node
    assert trading_node is not None, "the real build path must yield a TradingNode"
    for strategy in node._compiled_strategies:
        original = type(strategy).on_start

        def _counting_start(self: Any, _original: Any = original) -> None:
            strategy_starts.append(type(self).__name__)
            _original(self)

        monkeypatch.setattr(type(strategy), "on_start", _counting_start)

    run_loop = asyncio.create_task(node._run_loop())
    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5.0
        while not (_StubExecFactory.created and _StubExecFactory.created[0].connect_attempts):
            assert loop.time() < deadline, "the stub exec client was never asked to connect"
            await asyncio.sleep(0.01)
        # R42 facts: the failure is swallowed, start_async gives up after
        # timeout_connection and run_async keeps idling (no raise, no trader).
        await asyncio.sleep(_CONNECT_TIMEOUT_S + 0.5)
        assert not run_loop.done(), "the node must still be alive when the refusal lands"
        node._shutdown_event.set()
        await asyncio.wait_for(run_loop, timeout=10.0)  # (c): bounded
    finally:
        node._shutdown_event.set()
        if not run_loop.done():
            run_loop.cancel()
            await asyncio.gather(run_loop, return_exceptions=True)
        await asyncio.wait_for(node._shutdown(), timeout=5.0)  # (c): bounded
        retain_log_guard(trading_node)  # type: ignore[arg-type]

    # (a) no strategy ever started
    assert strategy_starts == [], "a refused connect must not start any strategy"
    assert trading_node.trader.is_running is False
    assert all(state != "RUNNING" for state in trading_node.trader.strategy_states().values())
    client = _StubExecFactory.created[0]
    assert client.connect_attempts == 1, "the stub exec client must really have been refused"
    assert client.is_connected is False

    # (b) no order anywhere: the NT cache, checkpoint rows (sqlite), ORDER events
    assert list(trading_node.kernel.cache.orders()) == []
    con = sqlite3.connect(f"file:{seeded['db']}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT orders FROM paper_trading_checkpoints WHERE trader_id = ?",
            (_LIVE_TRADER_ID,),
        ).fetchall()
    finally:
        con.close()
    assert rows, "the refused run must still have checkpointed"
    assert all(json.loads(row[0]) == {} for row in rows), "a checkpoint recorded an order"
    log_path = seeded["tmp"] / "logs" / f"{_LIVE_TRADER_ID}.jsonl"
    assert log_path.exists(), "the paper event log must exist"
    written = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [e for e in written if e["event"] == EventType.ORDER.value] == []

    # (c) nothing left running
    assert node._trading_node is None, "shutdown must drop the trading node"
    assert node._control_task is None
    assert node._liquidation_task is None
    assert node._persistence is None
    assert node._event_writer is None
    assert trading_node.is_running() is False
