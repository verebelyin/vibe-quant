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
from vibe_quant.data.catalog import create_instrument
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
    assert node._journal.writer is None
    assert trading_node.is_running() is False


# ------------------------------------- R30d: same-step entries on a LIVE TradingNode

_REACHED: list[str] = []


class _VerifiedExecClient(_StubExecClient):
    """Healthy venue: 1000 USDT, verified 10x on BTC/ETH, and every submit is recorded."""

    async def _connect(self) -> None:
        from decimal import Decimal

        from nautilus_trader.model.identifiers import InstrumentId
        from nautilus_trader.model.objects import AccountBalance, Money

        self.connect_attempts += 1
        for symbol in ("BTCUSDT", "ETHUSDT"):
            self._cache.add_instrument(create_instrument(symbol))
        self._set_account_id(AccountId("BINANCE-001"))
        balance = AccountBalance(Money(1000, USDT), Money(0, USDT), Money(1000, USDT))
        self.generate_account_state(balances=[balance], margins=[], reported=True, ts_event=0)
        account = self._cache.account_for_venue(_VENUE)
        for symbol in ("BTCUSDT", "ETHUSDT"):
            account.set_leverage(InstrumentId.from_str(f"{symbol}-PERP.BINANCE"), Decimal(10))

    async def _submit_order(self, command: Any) -> None:
        from nautilus_trader.model.identifiers import VenueOrderId

        order = command.order
        _REACHED.append(str(order.instrument_id))
        ts = self._clock.timestamp_ns()
        self.generate_order_submitted(
            order.strategy_id, order.instrument_id, order.client_order_id, ts
        )
        self.generate_order_accepted(
            order.strategy_id,
            order.instrument_id,
            order.client_order_id,
            VenueOrderId(f"V{len(_REACHED)}"),
            ts,
        )


class _VerifiedExecFactory(LiveExecClientFactory):
    @staticmethod
    def create(  # type: ignore[override]
        loop: Any,
        name: str,
        config: LiveExecClientConfig,
        msgbus: MessageBus,
        cache: Cache,
        clock: LiveClock,
    ) -> _VerifiedExecClient:
        return _VerifiedExecClient(loop, msgbus, cache, clock)


async def test_same_step_entries_are_margin_checked_on_a_live_node(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R30d: on a LIVE TradingNode a gate-admitted entry stays INITIALIZED (the live risk
    engine only enqueues it), in neither orders_open nor orders_inflight. Two entries
    submitted in ONE synchronous step (equity 1000, verified 10x, cushion 0.5: BTC margin
    400 + ETH margin 200 = 600 > 500) must still see each other: ETH is DENIED and never
    reaches the exec client."""
    from decimal import Decimal

    import nautilus_trader.adapters.binance.factories as binance_factories
    from nautilus_trader.model.enums import OrderSide, OrderStatus
    from nautilus_trader.model.identifiers import InstrumentId

    _REACHED.clear()
    monkeypatch.setattr(binance_factories, "BinanceLiveExecClientFactory", _VerifiedExecFactory)
    monkeypatch.setattr(binance_factories, "BinanceLiveDataClientFactory", _StubDataFactory)
    monkeypatch.setattr("nautilus_trader.live.config.TradingNodeConfig", _FastNodeConfig)

    async def _idle_watch() -> None:
        await asyncio.Event().wait()

    node = PaperTradingNode(_config(seeded, trader_id="PAPER-009"))
    monkeypatch.setattr(node, "_liquidation_loop", _idle_watch)
    submitted: list[str] = []

    def _submit(strategy: Any, symbol: str, qty: str, price: str) -> None:
        iid = InstrumentId.from_str(f"{symbol}-PERP.BINANCE")
        inst = strategy.cache.instrument(iid)
        order = strategy.order_factory.limit(
            instrument_id=iid,
            order_side=OrderSide.BUY,
            quantity=inst.make_qty(Decimal(qty)),
            price=inst.make_price(Decimal(price)),
        )
        submitted.append(symbol)
        strategy.submit_order(order)

    def _entries_in_one_step(self: Any, _original: Any = None) -> None:
        assert _original is not None
        _original(self)
        _submit(self, "BTCUSDT", "0.1", "40000")  # 4000 notional -> 400
        _submit(self, "ETHUSDT", "1", "2000")  # 2000 notional -> 200 (600 > 500)

    await node._initialize()
    trading_node: Any = node._trading_node
    assert trading_node is not None
    strategy_type = type(node._compiled_strategies[0])
    monkeypatch.setattr(
        strategy_type,
        "on_start",
        lambda self, _o=strategy_type.on_start: _entries_in_one_step(self, _o),
    )

    run_loop = asyncio.create_task(node._run_loop())
    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 15.0
        while len(submitted) < 2:
            assert loop.time() < deadline, "the strategy never submitted both entries"
            await asyncio.sleep(0.05)
        await asyncio.sleep(1.0)  # let the queued risk engine / exec client drain
        statuses = {str(o.instrument_id): o.status for o in trading_node.cache.orders()}
        guard = node._guard
        assert guard is not None
        node._shutdown_event.set()
        await asyncio.wait_for(run_loop, timeout=10.0)
    finally:
        node._shutdown_event.set()
        if not run_loop.done():
            run_loop.cancel()
            await asyncio.gather(run_loop, return_exceptions=True)
        await asyncio.wait_for(node._shutdown(), timeout=5.0)
        retain_log_guard(trading_node)  # type: ignore[arg-type]

    assert submitted == ["BTCUSDT", "ETHUSDT"]
    assert _REACHED == ["BTCUSDT-PERP.BINANCE"], "the over-limit ETH entry reached execution"
    assert statuses["ETHUSDT-PERP.BINANCE"] == OrderStatus.DENIED
    assert statuses["BTCUSDT-PERP.BINANCE"] != OrderStatus.DENIED
    assert guard.gate_denials == 1
