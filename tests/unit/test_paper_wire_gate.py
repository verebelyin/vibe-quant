"""Venue hold at the Binance wire (opspt P1, rows R28-R32, R19', R31d).

Harness: an offline NautilusTrader ``TradingNode`` with the REAL
``BinanceFuturesExecutionClient`` and ``BinanceHttpClient``; only the pyo3 transport under
the write gate is a recording fake (no network, the node is never started or connected).
The live exec engine's command and event queues are run by hand (``drain``), so every
command takes NT's own path to the transport and every refusal comes back as NT's own
rejection event.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

import pytest
from nautilus_trader.accounting.accounts.margin import MarginAccount
from nautilus_trader.adapters.binance.common.enums import BinanceAccountType, BinanceEnvironment
from nautilus_trader.adapters.binance.config import BinanceExecClientConfig
from nautilus_trader.adapters.binance.futures.execution import BinanceFuturesExecutionClient
from nautilus_trader.adapters.binance.futures.providers import BinanceFuturesInstrumentProvider
from nautilus_trader.adapters.binance.http.client import BinanceHttpClient
from nautilus_trader.adapters.binance.http.error import BinanceClientError
from nautilus_trader.config import LoggingConfig
from nautilus_trader.core.nautilus_pyo3 import HttpMethod
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.execution.messages import SubmitOrder, SubmitOrderList
from nautilus_trader.live.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.enums import OrderSide, OrderStatus
from nautilus_trader.model.events import (
    OrderCancelRejected,
    OrderDenied,
    OrderModifyRejected,
    OrderRejected,
)
from nautilus_trader.model.identifiers import AccountId, VenueOrderId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.orders.list import OrderList
from nautilus_trader.test_kit.stubs.events import TestEventStubs
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

from tests.unit.test_audit_paper_node import _config
from tests.unit.test_audit_paper_node import seeded as seeded  # noqa: F401  # fixture
from vibe_quant.data.catalog import create_instrument
from vibe_quant.paper.config import ConfigurationError
from vibe_quant.paper.guard import (
    GuardListener,
    GuardState,
    HaltReason,
    TradingGuard,
    TradingGuardConfig,
)
from vibe_quant.paper.halt_state import RESTORED_HALT_PREFIX, RestoredHalt
from vibe_quant.paper.node import PaperTradingNode
from vibe_quant.paper.ownership import OwnershipLedger
from vibe_quant.paper.wire_gate import WriteGate, install_write_gate

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

#: NT's real ``send_request``, captured before tests/conftest.py's network guard replaces it
#: per test: every request here reaches a fake transport, never a socket.
_REAL_SEND_REQUEST = BinanceHttpClient.send_request

OPEN_ORDERS = "/fapi/v1/openOrders"
OPEN_ALGO = "/fapi/v1/openAlgoOrders"
H = "venue position BTCUSDT 0 != session 1"
HELD = f"VQ venue hold: {H}; not sent"
_ACCOUNT = AccountId("BINANCE-USDT_FUTURES-master")


@pytest.fixture(autouse=True)
def _real_send_to_fake_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore NT's own ``send_request`` so requests reach the gate and the fake transport
    under it (the socket guards of tests/conftest.py stay on)."""
    monkeypatch.setattr(BinanceHttpClient, "send_request", _REAL_SEND_REQUEST)


class _Transport:
    """The pyo3 ``HttpClient`` stand-in: records ``(method, path)`` of every request that
    reached it and answers a venue rejection (no order is ever accepted here). The two
    open-order reads (opspt P2) answer the rows a test rests on Binance (default none)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.after: Callable[[], None] | None = None
        #: GET path -> rows answered (status 200); every other request is rejected.
        self.rows: dict[str, list[object]] = {OPEN_ORDERS: [], OPEN_ALGO: []}
        #: GET path -> an HTTP error status answered instead of its rows.
        self.errors: dict[str, int] = {}

    async def request(
        self,
        method: Any,
        url: str,
        params: Any = None,
        headers: Any = None,
        body: Any = None,
        keys: Any = None,
        timeout_secs: Any = None,
    ) -> Any:
        verb, path = str(method).rsplit(".", 1)[-1], urlparse(url).path
        self.calls.append((verb, path))
        if self.after is not None:
            self.after()
        if verb == "GET" and path in self.errors:
            error = json.dumps({"code": -1000, "msg": "test venue error"}).encode()
            return SimpleNamespace(status=self.errors[path], body=error, headers={})
        if verb == "GET" and path in self.rows:
            return SimpleNamespace(
                status=200, body=json.dumps(self.rows[path]).encode(), headers={}
            )
        reject = json.dumps({"code": -2010, "msg": "test venue reject"}).encode()
        return SimpleNamespace(status=400, body=reject, headers={})

    def writes(self) -> list[tuple[str, str]]:
        return [c for c in self.calls if c[0] != "GET"]


class _HoldListener(GuardListener):
    def __init__(self, live: _Live) -> None:
        self._live = live
        #: The metrics payload and the message of every guard state change.
        self.payloads: list[dict[str, object]] = []
        self.messages: list[str] = []

    def on_state_change(
        self, state: GuardState, reason: HaltReason | None, message: str, metrics: dict[str, object]
    ) -> None:
        self.payloads.append(dict(metrics))
        self.messages.append(message)

    def venue_hold(self) -> str | None:
        return self._live.hold()

    def venue_verified(self) -> bool:
        return self._live.verified


@dataclass
class _Live:
    node: TradingNode
    client: BinanceFuturesExecutionClient
    http: BinanceHttpClient
    transport: _Transport
    strategy: Strategy
    held: str | None = None
    raises: BaseException | None = None
    #: What the guard's ``venue_verified`` answers (the node's first venue check agreed).
    verified: bool = True
    guard: TradingGuard | None = None
    iid: Any = None
    orders: int = field(default=0)

    def hold(self) -> str | None:
        if self.raises is not None:
            raise self.raises
        return self.held

    def unchecked(self) -> str | None:
        """The node's P1-R8 predicate: a reason until its first venue check agreed."""
        return None if self.verified else "Binance not checked yet"

    @property
    def kernel(self) -> Any:
        return self.node.kernel

    async def drain(self) -> None:
        """Run the live risk and exec engines' queues until commands and events settled."""
        tasks = [
            asyncio.create_task(queue())
            for engine in (self.kernel.risk_engine, self.kernel.exec_engine)
            for queue in (engine._run_cmd_queue, engine._run_evt_queue)
        ]
        for _ in range(20):
            await asyncio.sleep(0)
        await asyncio.sleep(0.05)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def accepted(self, order: Any) -> Any:
        """``order`` resting on the venue (SUBMITTED then ACCEPTED, as reconciliation sees it)."""
        self.orders += 1
        self.kernel.cache.add_order(order)
        order.apply(TestEventStubs.order_submitted(order, account_id=_ACCOUNT))
        order.apply(
            TestEventStubs.order_accepted(
                order, account_id=_ACCOUNT, venue_order_id=VenueOrderId(str(self.orders))
            )
        )
        self.kernel.cache.update_order(order)
        return order

    def sl_tp(self, strategy: Any = None) -> tuple[Any, Any]:
        factory = (strategy or self.strategy).order_factory
        qty = Quantity.from_str("1.000")
        sl = factory.stop_market(
            self.iid, OrderSide.SELL, qty, trigger_price=Price.from_str("9000.0"), reduce_only=True
        )
        tp = factory.limit(
            self.iid, OrderSide.SELL, qty, price=Price.from_str("11000.0"), reduce_only=True
        )
        return self.accepted(sl), self.accepted(tp)


@contextlib.asynccontextmanager
async def _live(
    *,
    guard: bool = False,
    strategy: Strategy | None = None,
    restored: RestoredHalt | None = None,
    verified: bool = True,
    before_start: Callable[[_Live], Awaitable[object]] | None = None,
    listener: GuardListener | None = None,
) -> AsyncIterator[_Live]:
    loop = asyncio.get_running_loop()
    node = TradingNode(
        TradingNodeConfig(trader_id="PAPER-007", logging=LoggingConfig(log_level="ERROR")),
        loop=loop,
    )
    kernel = node.kernel
    instrument = create_instrument("BTCUSDT")
    kernel.cache.add_instrument(instrument)
    transport = _Transport()
    http = BinanceHttpClient(
        kernel.clock, api_key="k", api_secret="s", base_url="https://fapi.binance.com"
    )
    http._client = transport
    provider = BinanceFuturesInstrumentProvider(
        client=http, clock=kernel.clock, account_type=BinanceAccountType.USDT_FUTURES
    )
    client = BinanceFuturesExecutionClient(
        loop=loop,
        client=http,
        msgbus=kernel.msgbus,
        cache=kernel.cache,
        clock=kernel.clock,
        instrument_provider=provider,
        base_url_ws="wss://example.invalid",
        config=BinanceExecClientConfig(
            api_key="k", api_secret="s", account_type=BinanceAccountType.USDT_FUTURES
        ),
        environment=BinanceEnvironment.LIVE,
        api_key="k",
        api_secret="s",
    )
    client._set_account_id(_ACCOUNT)
    kernel.exec_engine.register_client(client)
    kernel.cache.add_account(MarginAccount(TestEventStubs.margin_account_state(_ACCOUNT)))
    strat = strategy or Strategy(StrategyConfig(order_id_tag="000"))
    live = _Live(
        node=node, client=client, http=http, transport=transport, strategy=strat, verified=verified
    )
    live.iid = instrument.id
    install_write_gate(client, live.hold, live.unchecked)
    if guard:
        live.guard = TradingGuard(
            TradingGuardConfig(check_interval_secs=3600),
            strategies=[strat],
            risk_engine=kernel.risk_engine,
            listener=listener or _HoldListener(live),
            restored_halt=restored,
        )
        node.trader.add_actor(live.guard)
    node.trader.add_strategy(strat)
    if live.guard is not None:
        live.guard.start()
    if before_start is not None:
        await before_start(live)  # what NT reconciled into the cache before a restart
    strat.start()
    try:
        yield live
    finally:
        kernel.clock.cancel_timers()
        for component in (strat, live.guard):
            if component is not None and component.is_running:
                component.stop()


def _reasons(order: Any, kind: type) -> list[str]:
    return [str(e.reason) for e in order.events if isinstance(e, kind)]


# ----------------------------------------------------------------------- R28-R32


async def test_held_cancel_never_reaches_binance() -> None:
    """R28: the strategy's cancel-all is queued while clean and the latch begins before the
    queue runs: nothing reaches Binance and NT restores both resting orders."""
    async with _live() as live:
        sl, tp = live.sl_tp()
        live.strategy.cancel_all_orders(live.iid)
        assert (sl.status, tp.status) == (OrderStatus.PENDING_CANCEL,) * 2
        live.held = H
        await live.drain()

        assert live.transport.writes() == []
        assert (sl.status, tp.status) == (OrderStatus.ACCEPTED, OrderStatus.ACCEPTED)
        assert _reasons(sl, OrderCancelRejected) == [HELD]
        assert _reasons(tp, OrderCancelRejected) == [HELD]


@pytest.mark.parametrize("state", ["submitted", "triggered"])
async def test_held_cancel_of_a_submitted_order_stays_unresolved(state: str) -> None:
    """R32 / S1: a cancel while held sends nothing. NT restores neither a SUBMITTED order
    nor a TRIGGERED stop-limit from PENDING_CANCEL on the refusal (its FSM), so the guard's
    status lists each as unresolved."""
    async with _live(guard=True) as live:
        factory = live.strategy.order_factory
        qty = Quantity.from_str("1.000")
        if state == "submitted":
            order = factory.limit(
                live.iid, OrderSide.SELL, qty, price=Price.from_str("12000.0"), reduce_only=True
            )
            live.kernel.cache.add_order(order)
            order.apply(TestEventStubs.order_submitted(order, account_id=_ACCOUNT))
        else:
            order = live.accepted(
                factory.stop_limit(
                    live.iid,
                    OrderSide.SELL,
                    qty,
                    price=Price.from_str("8990.0"),
                    trigger_price=Price.from_str("9000.0"),
                    reduce_only=True,
                )
            )
            order.apply(TestEventStubs.order_triggered(order))
        live.kernel.cache.update_order(order)
        live.held = H
        live.strategy.cancel_order(order)
        await live.drain()

        assert live.transport.writes() == []
        assert order.status == OrderStatus.PENDING_CANCEL
        assert live.guard is not None
        assert live.guard.status()["hold_unresolved"] == [str(order.client_order_id)]
        live.held = None
        assert live.guard.status()["hold_unresolved"] == []


async def test_mixed_list_is_denied_at_the_held_risk_gate() -> None:
    """R5' / S3 (rev control): a list of an entry and a reduce-only exit sent through the
    RiskEngine while held: both DENIED with the hold text, nothing reaches Binance."""
    async with _live(guard=True) as live:
        factory = live.strategy.order_factory
        qty = Quantity.from_str("1.000")
        orders = [
            factory.market(live.iid, OrderSide.BUY, qty),
            factory.market(live.iid, OrderSide.SELL, qty, reduce_only=True),
        ]
        for order in orders:
            live.kernel.cache.add_order(order)
        live.held = H
        live.kernel.msgbus.send(
            "RiskEngine.execute",
            SubmitOrderList(
                trader_id=live.kernel.trader_id,
                strategy_id=live.strategy.id,
                order_list=OrderList(factory.generate_order_list_id(), orders),
                command_id=UUID4(),
                ts_init=0,
            ),
        )
        await live.drain()

        assert live.transport.writes() == []
        assert [order.status for order in orders] == [OrderStatus.DENIED, OrderStatus.DENIED]
        for order in orders:
            assert _reasons(order, OrderDenied) == [
                f"VQ_GUARD: venue hold: {H}; nothing is sent until restart"
            ]


@pytest.mark.parametrize("cause", ["latched", "raises"])
async def test_held_submit_past_the_risk_gate_is_rejected_unsent(cause: str) -> None:
    """R29 / R31c: a reduce-only MARKET exit already past the risk gate reaches the client
    while held: 0 writes, REJECTED with the hold text. A raising hold check refuses too,
    named by class only."""
    async with _live() as live:
        exit_order = live.strategy.order_factory.market(
            live.iid, OrderSide.SELL, Quantity.from_str("1.000"), reduce_only=True
        )
        live.kernel.cache.add_order(exit_order)
        if cause == "latched":
            live.held = H
            expected = HELD
        else:
            live.raises = RuntimeError("ZZSECRET")
            expected = "VQ venue hold: venue hold check failed (RuntimeError); not sent"
        live.kernel.msgbus.send(
            "ExecEngine.execute",
            SubmitOrder(
                trader_id=live.kernel.trader_id,
                strategy_id=live.strategy.id,
                order=exit_order,
                command_id=UUID4(),
                ts_init=0,
            ),
        )
        await live.drain()

        assert live.transport.writes() == []
        assert exit_order.status == OrderStatus.REJECTED
        assert _reasons(exit_order, OrderRejected) == [expected]


async def test_list_checked_per_request() -> None:
    """R30: a two-order list whose hold begins right after the first request left: that
    one request reached Binance; the second order is refused with the hold text."""
    async with _live() as live:
        factory = live.strategy.order_factory
        qty = Quantity.from_str("1.000")
        first = factory.limit(
            live.iid, OrderSide.SELL, qty, price=Price.from_str("11000.0"), reduce_only=True
        )
        second = factory.stop_market(
            live.iid, OrderSide.SELL, qty, trigger_price=Price.from_str("9000.0"), reduce_only=True
        )
        for order in (first, second):
            live.kernel.cache.add_order(order)

        def latch() -> None:
            live.held = H

        live.transport.after = latch
        live.kernel.msgbus.send(
            "ExecEngine.execute",
            SubmitOrderList(
                trader_id=live.kernel.trader_id,
                strategy_id=live.strategy.id,
                order_list=OrderList(factory.generate_order_list_id(), [first, second]),
                command_id=UUID4(),
                ts_init=0,
            ),
        )
        await live.drain()

        assert live.transport.writes() == [("POST", "/fapi/v1/order")]
        assert _reasons(second, OrderRejected) == [HELD]
        assert _reasons(first, OrderRejected) != [HELD], "the first one was sent"


async def test_held_modify_is_rejected_and_price_kept() -> None:
    """R31: modifying the resting TP while held sends nothing; NT's OrderModifyRejected
    carries the hold text and the TP keeps its price."""
    async with _live() as live:
        _sl, tp = live.sl_tp()
        live.held = H
        live.strategy.modify_order(tp, price=Price.from_str("10500.0"))
        await live.drain()

        assert live.transport.writes() == []
        assert _reasons(tp, OrderModifyRejected) == [HELD]
        assert tp.status == OrderStatus.ACCEPTED
        assert tp.price == Price.from_str("11000.0")


async def test_reads_and_user_stream_keepalive_pass_while_held() -> None:
    """R31b: while held, a signed GET (positionRisk) and the user-data listenKey POST/PUT
    still reach the transport; a leverage POST does not."""
    async with _live() as live:
        live.held = H
        for method, path in (
            (HttpMethod.GET, "/fapi/v3/positionRisk"),
            (HttpMethod.POST, "/fapi/v1/listenKey"),
            (HttpMethod.PUT, "/fapi/v1/listenKey"),
        ):
            with pytest.raises(BinanceClientError, match="test venue reject"):
                await live.http.send_request(method, path)
        with pytest.raises(BinanceClientError, match="VQ venue hold"):
            await live.client._futures_http_account.set_leverage("BTCUSDT", 2)

        assert live.transport.calls == [
            ("GET", "/fapi/v3/positionRisk"),
            ("POST", "/fapi/v1/listenKey"),
            ("PUT", "/fapi/v1/listenKey"),
        ]


# ----------------------------------------------------------------------- R19'


async def test_stop_during_hold_compiled_strategy() -> None:
    """R19': SIGTERM while latched runs the compiled DSL ``on_stop`` (cancel-all, then
    close-all). The cancel is refused at the wire (SL/TP back to ACCEPTED), the close is
    denied at the guard's gate; nothing reaches Binance and the position stays open."""
    from vibe_quant.dsl.compiler import StrategyCompiler
    from vibe_quant.dsl.parser import validate_strategy_dict

    dsl = validate_strategy_dict(
        {
            "name": "wire_hold_probe",
            "timeframe": "1m",
            "indicators": {"sma_2": {"type": "SMA", "period": 2}},
            "entry_conditions": {"long": ["sma_2 > 0"]},
            "stop_loss": {"type": "fixed_pct", "percent": 10.0},
            "take_profit": {"type": "fixed_pct", "percent": 10.0},
        }
    )
    module = StrategyCompiler().compile_to_module(dsl)
    cfg = module.WireHoldProbeConfig.parse(
        json.dumps({"instrument_id": "BTCUSDT-PERP.BINANCE", "order_id_tag": "000"})
    )
    async with _live(guard=True, strategy=module.WireHoldProbeStrategy(config=cfg)) as live:
        instrument = live.kernel.cache.instrument(live.iid)
        entry = live.accepted(
            live.strategy.order_factory.market(live.iid, OrderSide.BUY, Quantity.from_str("1.000"))
        )
        live.kernel.exec_engine.process(
            TestEventStubs.order_filled(
                entry, instrument=instrument, account_id=_ACCOUNT, last_px=Price.from_str("10000.0")
            )
        )
        await live.drain()
        [position] = live.kernel.cache.positions_open()
        sl, tp = live.sl_tp()
        writes_before = list(live.transport.writes())
        live.held = H

        live.strategy.stop()
        await live.drain()

        assert live.transport.writes() == writes_before
        assert (sl.status, tp.status) == (OrderStatus.ACCEPTED, OrderStatus.ACCEPTED)
        assert _reasons(sl, OrderCancelRejected) == [HELD]
        closes = [
            o
            for o in live.kernel.cache.orders()
            if o.is_reduce_only and o.order_type.name == "MARKET"
        ]
        assert [o.status for o in closes] == [OrderStatus.DENIED]
        assert _reasons(closes[0], OrderDenied) == [
            f"VQ_GUARD: venue hold: {H}; nothing is sent until restart"
        ]
        assert position.is_open
        assert live.kernel.cache.positions_open() == [position]


# ----------------------------------------------------------------------- R31d


def test_non_binance_client_refuses_install() -> None:
    """R31d: a client that is not NT's Binance execution client, or whose transport has no
    ``request`` (an NT bump renamed it), cannot be gated: the install refuses."""
    stand_in = SimpleNamespace(_http_client=SimpleNamespace(_client=_Transport()))
    with pytest.raises(ConfigurationError, match="write gate not installable on SimpleNamespace"):
        install_write_gate(stand_in, lambda: None)


async def test_install_refuses_a_binance_client_without_its_transport() -> None:
    """R31d: the real Binance client whose ``_http_client._client`` lacks ``request``."""
    async with _live() as live:
        live.http._client = object()
        with pytest.raises(ConfigurationError, match="write gate not installable"):
            install_write_gate(live.client, lambda: None)


async def test_production_build_gates_the_real_binance_client(
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """The REAL ``PaperTradingNode._create_live_trading_node`` (real TradingNode, real
    Binance exec client factory, offline) installs the gate on the node's one exec client:
    a leverage write is refused once the node latches and reaches the transport before."""
    paper = PaperTradingNode(_config(seeded))
    paper._strategy = paper._load_strategy()
    paper._compiled_module = paper._compile_strategy(paper._strategy)
    paper._resolved = paper._resolve_run_config(paper._compiled_module, paper._strategy)
    paper._compiled_strategies = paper._instantiate_strategies(
        paper._compiled_module, paper._strategy
    )
    trading_node = paper._create_live_trading_node()
    try:
        [client] = trading_node.kernel.exec_engine._clients.values()
        gate = client._http_client._client
        assert isinstance(gate, WriteGate)
        transport = _Transport()
        gate._inner = transport

        with pytest.raises(BinanceClientError, match="test venue reject"):
            await client._futures_http_account.set_leverage("BTCUSDT", 2)
        paper._latch(H, kind="position_drift")
        with pytest.raises(BinanceClientError, match=f"VQ venue hold: {H}; not sent"):
            await client._futures_http_account.set_leverage("BTCUSDT", 2)
        assert transport.writes() == [("POST", "/fapi/v1/leverage")]
    finally:
        await paper._shutdown()


async def test_production_build_refuses_a_node_without_one_exec_client(
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """R31d: no (or more than one) execution client after build: start refused."""
    paper = PaperTradingNode(_config(seeded))
    node = SimpleNamespace(kernel=SimpleNamespace(exec_engine=SimpleNamespace(_clients={})))
    with pytest.raises(ConfigurationError, match="0 execution clients"):
        paper._install_write_gate(node)


# ------------------------------------------------- B1: the native rate-limit queue
#
# NT's pyo3 HttpClient waits for quota INSIDE ``request``. These run the REAL pyo3
# transport against a loopback receiver (the reviewers' repros, ported): a write that
# passed an early check while clean and waits for quota when the hold begins must
# never reach the receiver.


@contextlib.asynccontextmanager
async def _loopback(
    on_request: Callable[[list[tuple[str, str, bool]]], None] | None = None,
    answer: bytes = b"HTTP/1.1 200 OK",
) -> AsyncIterator[tuple[str, list[tuple[str, str, bool]], _Flag]]:
    """A loopback HTTP receiver recording ``(method, path, held at arrival)``; it answers
    ``answer`` (status line) with a ``{}`` body."""
    received: list[tuple[str, str, bool]] = []
    flag = _Flag()

    async def respond(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            verb, target, _ = head.splitlines()[0].decode().split()
            received.append((verb, target.split("?", 1)[0], flag.held is not None))
            if on_request is not None:
                on_request(received)
            writer.write(answer + b"\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    async with server:
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}", received, flag


@dataclass
class _Flag:
    held: str | None = None
    checks: int = 0

    def hold(self) -> str | None:
        self.checks += 1
        return self.held


async def test_write_queued_for_quota_is_not_sent_after_the_hold() -> None:
    """B1 (rev late_write.py): the order quota is used up, a second signed POST passes the
    early check while clean and waits for quota; the hold begins: it never arrives."""
    from nautilus_trader.common.component import LiveClock
    from nautilus_trader.core.nautilus_pyo3 import Quota

    key = "binance:fapi/v1/order"
    async with _loopback() as (url, received, flag):
        http = BinanceHttpClient(
            LiveClock(),
            api_key="offline-fixture",
            api_secret="offline-fixture",
            base_url=url,
            ratelimiter_quotas=[(key, Quota.rate_per_second(1))],
        )
        http._client = WriteGate(http._client, flag.hold, {key: (1, 1.0)})

        async def submit() -> Any:
            return await http.sign_request(
                HttpMethod.POST, "/fapi/v1/order", ratelimiter_keys=[key]
            )

        await submit()
        queued = asyncio.create_task(submit())
        while flag.checks < 2:
            await asyncio.sleep(0)
        flag.held = H
        with pytest.raises(BinanceClientError, match=f"VQ venue hold: {H}; not sent"):
            await asyncio.wait_for(queued, 5)

    assert received == [("POST", "/fapi/v1/order", False)]


async def test_raw_transport_queue_refuses_after_the_hold() -> None:
    """B1 (ro quota.py): the bare pyo3 HttpClient with an order quota; the request queued
    before the hold and one created after it are both refused, only the first arrives."""
    from nautilus_trader.core.nautilus_pyo3 import HttpClient, Quota

    async with _loopback() as (url, received, flag):
        raw = HttpClient(keyed_quotas=[("orders", Quota.rate_per_second(1))])
        client = WriteGate(raw, flag.hold, {"orders": (1, 1.0)})
        await client.request(HttpMethod.POST, url=f"{url}/prime", keys=["orders"])
        queued = asyncio.ensure_future(
            client.request(HttpMethod.POST, url=f"{url}/queued-before-latch", keys=["orders"])
        )
        await asyncio.sleep(0)
        assert not queued.done()
        flag.held = "position mismatch"
        with pytest.raises(BinanceClientError, match="VQ venue hold: position mismatch"):
            await queued
        with pytest.raises(BinanceClientError, match="VQ venue hold: position mismatch"):
            await client.request(HttpMethod.POST, url=f"{url}/created-after-latch", keys=["orders"])

    assert received == [("POST", "/prime", False)]


async def test_default_futures_quotas_send_nothing_after_the_hold() -> None:
    """B1 (ro quota_default.py): 54 signed order POSTs through NT's own futures client
    (factory quotas: 30 per second on the order-count key) and endpoint keys. The hold
    begins when the 31st arrives (the 1st one spaced by the quota; the next leaves 1/30 s
    later): the 23 still queued are refused and none arrives after the hold."""
    from nautilus_trader.adapters.binance.common.constants import (
        BINANCE_FUTURES_ORDER_COUNT_KEYS,
    )
    from nautilus_trader.adapters.binance.common.enums import BinanceSecurityType
    from nautilus_trader.adapters.binance.factories import get_cached_binance_http_client
    from nautilus_trader.adapters.binance.http.endpoint import BinanceHttpEndpoint
    from nautilus_trader.common.component import LiveClock

    from vibe_quant.paper.wire_gate import BINANCE_FUTURES_DEFAULT_QUOTA, BINANCE_FUTURES_QUOTAS

    flag_box: list[_Flag] = []

    def latch_at_31(received: list[tuple[str, str, bool]]) -> None:
        if len(received) == 31:
            flag_box[0].held = "position mismatch"

    async with _loopback(latch_at_31) as (url, received, flag):
        flag_box.append(flag)
        client = get_cached_binance_http_client(
            LiveClock(),
            BinanceAccountType.USDT_FUTURES,
            api_key="synthetic",
            api_secret="synthetic",
            base_url=url,
        )
        client._client = WriteGate(
            client._client, flag.hold, BINANCE_FUTURES_QUOTAS, BINANCE_FUTURES_DEFAULT_QUOTA
        )
        endpoint = BinanceHttpEndpoint(
            client, {HttpMethod.POST: BinanceSecurityType.TRADE}, "/fapi/v1/order"
        )
        results = await asyncio.wait_for(
            asyncio.gather(
                *(
                    endpoint._method(
                        HttpMethod.POST,
                        {"symbol": "BTCUSDT"},
                        list(BINANCE_FUTURES_ORDER_COUNT_KEYS),
                    )
                    for _ in range(54)
                ),
                return_exceptions=True,
            ),
            10,
        )

    refused = [r for r in results if isinstance(r, BinanceClientError)]
    assert len(received) == 31
    assert [after for *_, after in received] == [False] * 31
    assert len(refused) == 23
    assert {r.message for r in refused} == {"VQ venue hold: position mismatch; not sent"}


def test_quota_table_mirrors_the_nt_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate's quota table equals what NT's futures factory gives its HttpClient (an NT
    bump that changes a quota fails here, never silently doubles the request rate)."""
    import nautilus_trader.adapters.binance.factories as factories
    from nautilus_trader.common.component import LiveClock

    from vibe_quant.paper.wire_gate import BINANCE_FUTURES_DEFAULT_QUOTA, BINANCE_FUTURES_QUOTAS

    class _Quota:
        @staticmethod
        def rate_per_second(n: int) -> tuple[int, float]:
            return (n, 1.0)

        @staticmethod
        def rate_per_minute(n: int) -> tuple[int, float]:
            return (n, 60.0)

        @staticmethod
        def rate_per_hour(n: int) -> tuple[int, float]:
            return (n, 3600.0)

    captured: dict[str, Any] = {}
    monkeypatch.setattr(factories, "Quota", _Quota)
    monkeypatch.setattr(factories, "BinanceHttpClient", lambda **kw: captured.update(kw))
    factories.get_cached_binance_http_client.__wrapped__(
        LiveClock(), BinanceAccountType.USDT_FUTURES, api_key="k", api_secret="s"
    )

    assert dict(captured["ratelimiter_quotas"]) == BINANCE_FUTURES_QUOTAS
    assert len(captured["ratelimiter_quotas"]) == len(BINANCE_FUTURES_QUOTAS)
    assert captured["ratelimiter_default_quota"] == BINANCE_FUTURES_DEFAULT_QUOTA


async def test_quota_limiter_spaces_like_the_native_one() -> None:
    """The Python limiter's admissions, on a fake clock, equal NT's measured native ones:
    burst N then one per period/N; an unquoted key uses the default per key; a request
    with two keys waits for both."""
    from vibe_quant.paper.wire_gate import QuotaLimiter

    now = [0.0]
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay
        await real_sleep(0)

    limiter = QuotaLimiter({"k": (3, 1.0), "a": (1, 1.0)}, default=(1, 1.0), clock=lambda: now[0])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(asyncio, "sleep", fake_sleep)
        times = []
        for _ in range(5):
            await limiter.acquire(["k"])
            times.append(round(now[0], 6))
        assert times == [0.0, 0.0, 0.0, round(1 / 3, 6), round(2 / 3, 6)]
        now[0] = 10.0
        await limiter.acquire(["u1"])
        await limiter.acquire(["u2"])
        assert now[0] == 10.0, "each unquoted key has its own default quota"
        await limiter.acquire(["u1"])
        assert now[0] == 11.0
        await limiter.acquire(["a", "k"])
        await limiter.acquire(["a", "k"])
        assert now[0] == 12.0, "the second waits for 'a' (1/s) though 'k' is ready"
        await limiter.acquire(None)
        assert now[0] == 12.0


# ------------------------------------- P1-R6: a restored halt waits for the venue check

_RESTORED = RestoredHalt(
    HaltReason.MANUAL, f"{RESTORED_HALT_PREFIX}halt via API", "2026-11-21T00:00:00+00:00"
)
_CLOSE = {("POST", "/fapi/v1/order")}


@pytest.mark.parametrize(
    ("first_check", "flatten_at"),
    [
        pytest.param("verified_at_start", "first_ticks", id="control"),
        pytest.param("agrees_later", "after_check", id="pending_then_agrees"),
        pytest.param("disagrees", None, id="first_check_latches"),
        pytest.param("never", None, id="never_verified"),
    ],
)
async def test_restored_halt_flatten_waits_for_an_agreeing_venue_check(
    first_check: str, flatten_at: str | None
) -> None:
    """P1-R6 (user Q3): a restored HALTED session with a long open sends NOTHING on its
    ticks while the first venue check is pending (no close, no cancel). An agreeing check
    releases the deferred flatten once: the same reduce-only close the control sends on
    its first ticks. A disagreeing first check latches the hold and the flatten never
    runs; a check that never passes keeps it deferred."""
    verified = first_check == "verified_at_start"
    async with _live(guard=True, restored=_RESTORED, verified=verified) as live:
        guard = live.guard
        assert guard is not None and guard.guard_state == GuardState.HALTED
        flattens: list[int] = []
        flatten = guard._flatten

        def counted(**kwargs: Any) -> Any:
            flattens.append(1)
            return flatten(**kwargs)

        guard._flatten = counted  # type: ignore[method-assign]
        listener = guard._listener
        assert isinstance(listener, _HoldListener)
        # The restored HALT notice (on_start) says the close waits for the check.
        from vibe_quant.paper.guard import FLATTEN_DEFERRED

        [restored] = listener.payloads
        assert restored.get("flatten") == (None if verified else FLATTEN_DEFERRED)
        entry = live.accepted(
            live.strategy.order_factory.market(live.iid, OrderSide.BUY, Quantity.from_str("1.000"))
        )
        live.kernel.exec_engine.process(
            TestEventStubs.order_filled(
                entry,
                instrument=live.kernel.cache.instrument(live.iid),
                account_id=_ACCOUNT,
                last_px=Price.from_str("10000.0"),
            )
        )
        await live.drain()
        for _ in range(3):
            guard._on_tick(None)
        await live.drain()
        sent_before = live.transport.writes()
        flattens_before = len(flattens)

        if first_check == "agrees_later":
            live.verified = True
        elif first_check == "disagrees":
            live.held = H
        for _ in range(3):
            guard._on_tick(None)
        await live.drain()
        sent_after = live.transport.writes()[len(sent_before) :]
        closes = [o for o in live.kernel.cache.orders() if o.is_reduce_only and o is not entry]

    expected_before = _CLOSE if flatten_at == "first_ticks" else set()
    expected_after = _CLOSE if flatten_at == "after_check" else set()
    assert set(sent_before) == expected_before
    if flatten_at != "first_ticks":
        assert sent_before == [], "nothing at all reaches Binance before the check agrees"
        assert set(sent_after) == expected_after
    assert (flattens_before, len(flattens)) == (
        (1, 1) if flatten_at == "first_ticks" else (0, 1 if flatten_at else 0)
    )
    assert [(o.order_type.name, o.side.name, str(o.quantity)) for o in closes][:1] == (
        [("MARKET", "SELL", "1.000")] if flatten_at else []
    )
    assert guard.guard_state == GuardState.HALTED


_DEFERRED_ALERT = "; positions not closed yet: closing waits for the Binance check"


@pytest.mark.parametrize(
    ("requests", "then"),
    [
        pytest.param(("halt",), "agrees", id="halt"),
        pytest.param(("kill_switch",), "agrees", id="kill_switch"),
        pytest.param(("close_all",), "agrees", id="close_all_refused"),
        pytest.param(("halt", "kill_switch", "close_all"), "agrees", id="two_halts_one_flatten"),
        pytest.param(("halt", "close_all"), "disagrees", id="first_check_latches"),
        pytest.param(("kill_switch",), "never", id="never_verified"),
    ],
)
async def test_flatten_requested_before_the_venue_check_waits_for_it(
    requests: tuple[str, ...], then: str
) -> None:
    """P1-R7 (user Q3): a halt or kill-switch halt requested while the node's first venue
    check has not agreed changes the state now (entries denied) and sends NOTHING; the
    alert says closing waits for the check. On the first agreeing check ONE flatten runs
    (several halts merge), the same reduce-only close as today; a disagreeing first check
    latches the hold and it never runs; never verified = stays deferred. B7a: a close-all
    before the check is refused at once and owes nothing."""
    from vibe_quant.paper.guard import CLOSE_ALL_NOT_CHECKED, FLATTEN_DEFERRED

    async with _live(guard=True, verified=False) as live:
        guard = live.guard
        assert guard is not None
        listener = guard._listener
        assert isinstance(listener, _HoldListener)
        flattens: list[int] = []
        flatten = guard._flatten

        def counted(**kwargs: Any) -> Any:
            flattens.append(1)
            return flatten(**kwargs)

        guard._flatten = counted  # type: ignore[method-assign]
        entry = live.accepted(
            live.strategy.order_factory.market(live.iid, OrderSide.BUY, Quantity.from_str("1.000"))
        )
        live.kernel.exec_engine.process(
            TestEventStubs.order_filled(
                entry,
                instrument=live.kernel.cache.instrument(live.iid),
                account_id=_ACCOUNT,
                last_px=Price.from_str("10000.0"),
            )
        )
        await live.drain()
        close_all_errors: list[list[str]] = []
        for request in requests:
            if request == "close_all":
                close_all_errors.append(guard.close_all().errors)
            else:
                reason = HaltReason.MANUAL if request == "halt" else HaltReason.KILL_SWITCH
                assert guard.halt(reason, f"{request} probe") is True
        for _ in range(3):
            guard._on_tick(None)
        await live.drain()
        sent_before = live.transport.writes()
        flattens_before = len(flattens)
        halted = guard.guard_state

        if then == "agrees":
            live.verified = True
        elif then == "disagrees":
            live.held = H
        for _ in range(3):
            guard._on_tick(None)
        await live.drain()
        sent_after = live.transport.writes()

    halt_count = len(requests) - requests.count("close_all")
    assert (sent_before, flattens_before) == ([], 0), "nothing reaches Binance before the check"
    assert close_all_errors == [[CLOSE_ALL_NOT_CHECKED]] * requests.count("close_all")
    halts = [p for p in listener.payloads if "flatten" in p]
    assert [p["flatten"] for p in halts] == [FLATTEN_DEFERRED] * halt_count
    for payload in halts:
        text = PaperTradingNode._halt_alert_text(HaltReason.MANUAL, "m", payload)
        assert text == f"HALT (manual): m{_DEFERRED_ALERT}"
    expected = GuardState.HALTED if halt_count else GuardState.RUNNING
    assert halted == expected, "the state changes now; only the close waits"
    if then == "agrees" and halt_count:
        assert len(flattens) == 1, "one flatten for every deferred halt"
        assert set(sent_after) == _CLOSE
    else:
        assert (sent_after, flattens) == ([], []), "a refused close-all owes nothing"


async def _long(live: _Live) -> Any:
    entry = live.accepted(
        live.strategy.order_factory.market(live.iid, OrderSide.BUY, Quantity.from_str("1.000"))
    )
    live.kernel.exec_engine.process(
        TestEventStubs.order_filled(
            entry,
            instrument=live.kernel.cache.instrument(live.iid),
            account_id=_ACCOUNT,
            last_px=Price.from_str("10000.0"),
        )
    )
    await live.drain()
    return entry


def _past_utc_midnight(guard: TradingGuard, monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard's next risk pass sees a new UTC day (equity readable)."""
    from datetime import date

    monkeypatch.setattr(guard, "current_equity", lambda: Decimal("1000"))
    guard._risk.day = date(2020, 1, 1)


@pytest.mark.parametrize("lift", ["resume", "utc_rollover"])
async def test_halt_lift_cancels_its_waiting_close_and_says_so(
    lift: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B7b: a halt lifted (operator resume, daily-loss UTC rollover) before the venue
    check agreed drops its owed flatten; the lift message (alert + status) says the
    waiting close was cancelled and positions stay open. After agreement: 0 writes."""
    from vibe_quant.paper.guard import WAITING_CLOSE_CANCELLED

    async with _live(guard=True, verified=False) as live:
        guard = live.guard
        assert guard is not None
        listener = guard._listener
        assert isinstance(listener, _HoldListener)
        await _long(live)
        reason = HaltReason.MANUAL if lift == "resume" else HaltReason.MAX_DAILY_LOSS
        assert guard.halt(reason, "halt while the check is pending")
        if lift == "resume":
            assert guard.resume() == (True, "resumed from halt")
        else:
            _past_utc_midnight(guard, monkeypatch)
            guard.evaluate_risk()
        live.verified = True
        for _ in range(3):
            guard._on_tick(None)
        await live.drain()

        assert guard.guard_state == GuardState.RUNNING
        assert listener.messages[-1].endswith(f"; {WAITING_CLOSE_CANCELLED}")
        assert live.transport.writes() == []
        assert len(live.kernel.cache.positions_open()) == 1


async def test_timer_lifts_an_expired_halt_before_releasing_its_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B7c (ro midnight case): a daily-loss halt's close waits for the check; the check
    agrees after UTC midnight. The first tick lifts the expired halt first, so the
    prior day's close is cancelled, never sent."""
    from vibe_quant.paper.guard import WAITING_CLOSE_CANCELLED

    async with _live(guard=True, verified=False) as live:
        guard = live.guard
        assert guard is not None
        listener = guard._listener
        assert isinstance(listener, _HoldListener)
        await _long(live)
        assert guard.halt(HaltReason.MAX_DAILY_LOSS, "daily loss")
        _past_utc_midnight(guard, monkeypatch)
        live.verified = True
        guard._on_tick(None)
        await live.drain()

        assert guard.guard_state == GuardState.RUNNING
        assert WAITING_CLOSE_CANCELLED in listener.messages[-1]
        assert live.transport.writes() == []


async def test_close_all_after_agreement_takes_over_the_owed_close() -> None:
    """B7d (ro retry case): a halt's close is owed, the operator's close-all before the
    check is refused, the check agrees and the operator retries before the guard tick:
    that close-all closes once and consumes the owed close; the tick sends nothing more."""
    from vibe_quant.paper.guard import CLOSE_ALL_NOT_CHECKED

    async with _live(guard=True, verified=False) as live:
        guard = live.guard
        assert guard is not None
        await _long(live)
        live.strategy.stop()  # the blank strategy's on_stop trades nothing
        assert guard.halt(HaltReason.MANUAL, "halt while the check is pending")
        assert guard.close_all().errors == [CLOSE_ALL_NOT_CHECKED]
        await live.drain()
        assert live.transport.writes() == []

        live.verified = True
        assert guard.close_all().errors == []
        await live.drain()
        after_retry = live.transport.writes()
        guard._on_tick(None)
        await live.drain()

    assert after_retry == [("POST", "/fapi/v1/order")]
    assert live.transport.writes() == after_retry, "one close in total"


def _compiled() -> Strategy:
    from vibe_quant.dsl.compiler import StrategyCompiler
    from vibe_quant.dsl.parser import validate_strategy_dict

    dsl = validate_strategy_dict(
        {
            "name": "wire_hold_probe",
            "timeframe": "1m",
            "indicators": {"sma_2": {"type": "SMA", "period": 2}},
            "entry_conditions": {"long": ["sma_2 > 0"]},
            "stop_loss": {"type": "fixed_pct", "percent": 10.0},
            "take_profit": {"type": "fixed_pct", "percent": 10.0},
        }
    )
    module = StrategyCompiler().compile_to_module(dsl)
    cfg = module.WireHoldProbeConfig.parse(
        json.dumps({"instrument_id": "BTCUSDT-PERP.BINANCE", "order_id_tag": "000"})
    )
    strategy: Strategy = module.WireHoldProbeStrategy(config=cfg)
    return strategy


_EXIT_WRITES = [
    ("DELETE", "/fapi/v1/algoOpenOrders"),
    ("DELETE", "/fapi/v1/allOpenOrders"),
    ("POST", "/fapi/v1/order"),
]


@pytest.mark.parametrize("verified", [False, True], ids=["unchecked", "agreed"])
@pytest.mark.parametrize("operation", ["ordinary_exit", "on_stop"])
async def test_compiled_exit_and_stop_send_nothing_before_the_venue_check(
    operation: str, verified: bool
) -> None:
    """P1-R8 / B5: before the node's first agreeing venue check, the compiled strategy's
    ordinary exit and its ``on_stop`` (SIGTERM) cancel-all + close reach the wire gate and
    are refused there: 0 writes, SL/TP stay ACCEPTED on Binance, the position stays open,
    no latch. After agreement both send today's two cancels and the close."""
    from vibe_quant.paper.guard import WIRE_UNCHECKED_PREFIX

    async with _live(guard=True, strategy=_compiled(), verified=verified) as live:
        await _long(live)
        sl, tp = live.sl_tp()
        start = len(live.transport.writes())
        if operation == "ordinary_exit":
            live.strategy._submit_exit(None)  # type: ignore[attr-defined]
        else:
            live.strategy.stop()
        await live.drain()
        sent = live.transport.writes()[start:]
        closes = [
            o
            for o in live.kernel.cache.orders()
            if o.is_reduce_only and o.order_type.name == "MARKET"
        ]

    if verified:
        assert sent == _EXIT_WRITES
        return
    assert sent == []
    # No cancel reached Binance: SL/TP rest there. Locally NT restores them after the
    # refused on_stop cancel; the exit's refused cancel-all leaves them PENDING_CANCEL
    # (the same as under the latch, S1).
    local = OrderStatus.ACCEPTED if operation == "on_stop" else OrderStatus.PENDING_CANCEL
    assert (sl.status, tp.status) == (local, local)
    assert [o.status for o in closes] == [OrderStatus.REJECTED]
    [reason] = _reasons(closes[0], OrderRejected)
    assert reason.startswith(WIRE_UNCHECKED_PREFIX)
    assert len(live.kernel.cache.positions_open()) == 1
    assert live.held is None, "recoverable: nothing latched"


async def test_unchecked_gate_passes_only_account_config_writes() -> None:
    """P1-R8: before agreement the gate refuses every order/cancel/modify write (single,
    batch, algo, cancel-all) and passes only NT's connect-time leverage and margin-type
    writes, the listenKey keep-alive and reads; agreement opens it. After a latch the
    config writes are refused too (R31b)."""
    refused = [
        (HttpMethod.POST, "/fapi/v1/order"),
        (HttpMethod.PUT, "/fapi/v1/order"),
        (HttpMethod.DELETE, "/fapi/v1/order"),
        (HttpMethod.POST, "/fapi/v1/batchOrders"),
        (HttpMethod.DELETE, "/fapi/v1/batchOrders"),
        (HttpMethod.DELETE, "/fapi/v1/allOpenOrders"),
        (HttpMethod.POST, "/fapi/v1/algoOrder"),
        (HttpMethod.DELETE, "/fapi/v1/algoOpenOrders"),
    ]
    passed = [
        (HttpMethod.POST, "/fapi/v1/leverage"),
        (HttpMethod.POST, "/fapi/v1/marginType"),
        (HttpMethod.PUT, "/fapi/v1/listenKey"),
        (HttpMethod.GET, "/fapi/v3/positionRisk"),
    ]
    async with _live(verified=False) as live:
        for method, path in refused:
            with pytest.raises(BinanceClientError, match="VQ venue not checked yet"):
                await live.http.send_request(method, path)
        for method, path in passed:
            with pytest.raises(BinanceClientError, match="test venue reject"):
                await live.http.send_request(method, path)
        assert live.transport.calls == [(str(m).rsplit(".", 1)[-1], p) for m, p in passed]
        live.verified = True
        with pytest.raises(BinanceClientError, match="test venue reject"):
            await live.http.send_request(HttpMethod.POST, "/fapi/v1/order")
        live.held = H
        with pytest.raises(BinanceClientError, match="VQ venue hold"):
            await live.http.send_request(HttpMethod.POST, "/fapi/v1/leverage")


# ------------------------------------- B8: an adopted position's SL/TP after agreement

_LEGS = [("POST", "/fapi/v1/algoOrder"), ("POST", "/fapi/v1/order")]


async def _protected_long(live: _Live) -> None:
    await _long(live)
    live.sl_tp()


def _new_legs(live: _Live, before: set[Any]) -> list[tuple[str, str, str]]:
    """Reduce-only SL/TP the strategy built after ``before``: (type, side, qty @ level)."""
    return sorted(
        (
            o.order_type.name,
            o.side.name,
            f"{o.quantity} @ {o.trigger_price if o.has_trigger_price else o.price}",
        )
        for o in live.kernel.cache.orders()
        if o.is_reduce_only and o.order_type.name != "MARKET" and o.client_order_id not in before
    )


@contextlib.asynccontextmanager
async def _restarted(
    seeded: dict[str, Any],
    before_start: Callable[[_Live], Awaitable[object]],
    *,
    strategy: Strategy | None = None,
    settle: bool = True,
) -> AsyncIterator[tuple[PaperTradingNode, _Live]]:
    """A restart: the compiled strategy starts on an adopted BTC long before the node's
    first venue check. The REAL node predicates gate the REAL Binance client, and the
    check is applied through the node's own ``_venue_apply``. ``settle=False`` yields
    with the start-up SL/TP still queued (INITIALIZED, not yet at the wire)."""
    from vibe_quant.paper.node import ResolvedRunConfig

    paper = PaperTradingNode(_config(seeded))
    try:
        async with _live(
            guard=True,
            strategy=strategy or _compiled(),
            verified=False,
            before_start=before_start,
        ) as live:
            install_write_gate(live.client, paper._venue_hold, paper._venue_unchecked)
            paper._guard = live.guard
            paper._resolved = ResolvedRunConfig(["BTCUSDT"], {}, 5, "CROSSED")
            paper._compiled_strategies = [live.strategy]
            if settle:
                await live.drain()
            yield paper, live
    finally:
        await paper._shutdown()


def _agree(paper: PaperTradingNode) -> bool:
    """One matching leverage/margin answer through the node's real apply path."""
    from vibe_quant.paper.venue_config import classify_symbol_configs

    body = [{"symbol": "BTCUSDT", "leverage": 5, "marginType": "CROSSED"}]
    return paper._venue_apply(classify_symbol_configs(body, ["BTCUSDT"], 5, "CROSSED"), None)


@pytest.mark.parametrize(
    ("adopted", "first_check", "sent", "legs"),
    [
        pytest.param(
            _long,
            "agrees",
            _LEGS,
            [("LIMIT", "SELL", "1.000 @ 11000.0"), ("STOP_MARKET", "SELL", "1.000 @ 9000.0")],
            id="unprotected",
        ),
        pytest.param(_protected_long, "agrees", [], [], id="already_protected"),
        pytest.param(_long, "disagrees", [], [], id="first_check_latches"),
    ],
)
async def test_adopted_position_is_protected_once_after_the_first_agreeing_check(
    seeded: dict[str, Any],  # noqa: F811
    adopted: Callable[[_Live], Awaitable[object]],
    first_check: str,
    sent: list[tuple[str, str]],
    legs: list[tuple[str, str, str]],
) -> None:
    """B8 (P1-R8): on restart the strategy re-arms an adopted position's SL/TP in
    ``on_start``; before the first agreeing venue check the wire refuses both legs. The
    first agreeing check re-evaluates the position once: an unprotected one gets both
    legs (sized to it, today's 10 % levels), a protected one gets none (no duplicate),
    a second agreeing check adds nothing. A disagreeing first check latches: no leg,
    ever, even when a later answer would agree."""
    from vibe_quant.paper.guard import WIRE_UNCHECKED_PREFIX
    from vibe_quant.paper.venue_config import parse_positions

    async with _restarted(seeded, adopted) as (paper, live):
        startup = [
            o for o in live.kernel.cache.orders() if o.is_reduce_only and o.venue_order_id is None
        ]
        before = {o.client_order_id for o in live.kernel.cache.orders()}
        assert live.transport.writes() == [], "nothing reaches Binance before the check"
        if first_check == "disagrees":
            flat = no_orders(parse_positions([], ["BTCUSDT"]))
            paper._venue_apply_positions(flat, None)
        agreed = [_agree(paper), _agree(paper)]
        await live.drain()
        writes = live.transport.writes()
        built = _new_legs(live, before)
        open_positions = len(live.kernel.cache.positions_open())

    assert agreed == ([True, True] if first_check == "agrees" else [False, False])
    assert writes == sent
    assert built == legs
    assert open_positions == 1
    if adopted is _long:
        # The start-up re-arm was refused at the wire (recoverable, no latch).
        assert sorted(o.order_type.name for o in startup) == ["LIMIT", "STOP_MARKET"]
        assert all(o.status == OrderStatus.REJECTED for o in startup)
        assert all(_reasons(o, OrderRejected)[0].startswith(WIRE_UNCHECKED_PREFIX) for o in startup)
    else:
        assert startup == []


async def _inflight_legs_of_another_size(live: _Live) -> None:
    """An adopted long 1.000 whose SL/TP (0.500 each) were in flight when the process
    stopped: NT reconciles them as SUBMITTED."""
    await _long(live)
    factory = live.strategy.order_factory
    qty = Quantity.from_str("0.500")
    for order in (
        factory.stop_market(
            live.iid, OrderSide.SELL, qty, trigger_price=Price.from_str("9000.0"), reduce_only=True
        ),
        factory.limit(
            live.iid, OrderSide.SELL, qty, price=Price.from_str("11000.0"), reduce_only=True
        ),
    ):
        live.kernel.cache.add_order(order)
        order.apply(TestEventStubs.order_submitted(order, account_id=_ACCOUNT))
        live.kernel.cache.update_order(order)


async def test_unresolved_refused_cancel_places_no_second_pair_and_alerts(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B8: at start the strategy resizes the in-flight legs (cancel + replace); before
    agreement the wire refuses both. NT keeps the SUBMITTED legs PENDING_CANCEL (they
    may rest on Binance), the replacements are REJECTED. The first agreeing check places
    no second pair over them; one critical alert asks for a manual check."""
    alerts: list[Any] = []
    async with _restarted(seeded, _inflight_legs_of_another_size) as (paper, live):
        monkeypatch.setattr(paper._journal, "alert", alerts.append)
        states = sorted(
            (o.order_type.name, str(o.quantity), o.status.name)
            for o in live.kernel.cache.orders()
            if o.is_reduce_only
        )
        before = {o.client_order_id for o in live.kernel.cache.orders()}
        assert _agree(paper) is True
        await live.drain()
        writes = live.transport.writes()
        built = _new_legs(live, before)

    assert states == [
        ("LIMIT", "0.500", "PENDING_CANCEL"),
        ("LIMIT", "1.000", "REJECTED"),
        ("STOP_MARKET", "0.500", "PENDING_CANCEL"),
        ("STOP_MARKET", "1.000", "REJECTED"),
    ]
    assert (writes, built) == ([], [])
    [alert] = alerts
    assert alert.critical
    assert alert.message == (
        "paper PAPER-007: SL/TP check after the Binance check was skipped: a cancel refused "
        f"before it is unresolved for {live.strategy.id}; check its protection on Binance"
    )


async def test_protection_re_evaluation_failure_alerts_and_the_check_stands(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """B8: a re-evaluation that raises does not latch the session or undo the pass; one
    critical alert names the strategy and asks for a manual check; the log carries the
    error class only (no exception text)."""
    alerts: list[Any] = []
    async with _restarted(seeded, _long) as (paper, live):
        monkeypatch.setattr(paper._journal, "alert", alerts.append)

        def boom() -> bool:
            raise RuntimeError("SECRET-MARKER")

        monkeypatch.setattr(live.strategy, "_ensure_protection", boom)
        with caplog.at_level("ERROR"):
            agreed = _agree(paper)
        passed, latched = paper._venue_passed, paper._venue_latched

    assert (agreed, passed, latched) == (True, True, False)
    [alert] = alerts
    assert alert.critical
    assert alert.message == (
        "paper PAPER-007: SL/TP check after the Binance check failed (RuntimeError) for "
        f"{live.strategy.id}; check its protection on Binance"
    )
    assert "RuntimeError" in caplog.text
    assert "SECRET-MARKER" not in caplog.text


# ------------------- B9/B10: the first agreement while start-up SL/TP are still on their way


def _healthy_venue(live: _Live, *, fill: bool = True) -> list[Any]:
    """Binance answering like a healthy venue under the gate: a reduce-only SL (algo
    order) or TP (limit) is accepted, a reduce-only market close fills at 10200 (with
    ``fill=False`` it is only accepted: the fill is the test's); any other write gets
    the fake's rejection. Returns every order Binance took, in order (its user-stream
    events are queued into NT's exec engine)."""
    from urllib.parse import parse_qs

    from nautilus_trader.model.enums import OrderType
    from nautilus_trader.model.identifiers import ClientOrderId

    fake = live.transport.request
    kernel = live.kernel
    took: list[Any] = []

    async def request(method: Any, url: str, **kwargs: Any) -> Any:
        path = urlparse(url).path
        query = parse_qs(urlparse(url).query)
        cids = query.get("newClientOrderId") or query.get("clientAlgoId") or []
        cid = cids[0] if cids else None
        order = None if cid is None else kernel.cache.order(ClientOrderId(cid))
        if str(method).endswith("POST") and order is not None and order.is_reduce_only:
            live.transport.calls.append(("POST", path))
            venue_id = VenueOrderId(str(200 + len(took)))
            kernel.exec_engine.process(
                TestEventStubs.order_accepted(order, account_id=_ACCOUNT, venue_order_id=venue_id)
            )
            if order.order_type == OrderType.MARKET and fill:
                _close_fill(live, order, venue_id)
            took.append(order)
            body: dict[str, Any] = {
                "symbol": "BTCUSDT",
                "side": order.side.name,
                "positionSide": "BOTH",
                "timeInForce": "GTC",
                "reduceOnly": True,
            }
            if path.endswith("/algoOrder"):
                body |= {
                    "algoId": 200 + len(took),
                    "clientAlgoId": cid,
                    "algoType": "CONDITIONAL",
                    "orderType": "STOP_MARKET",
                    "quantity": str(order.quantity),
                    "algoStatus": "NEW",
                    "triggerPrice": str(order.trigger_price),
                    "price": "0",
                    "icebergQuantity": None,
                    "selfTradePreventionMode": "NONE",
                    "workingType": "CONTRACT_PRICE",
                    "priceMatch": "NONE",
                    "closePosition": False,
                    "priceProtect": False,
                    "createTime": 0,
                    "updateTime": 0,
                    "triggerTime": 0,
                    "goodTillDate": 0,
                }
            else:
                body |= {
                    "orderId": 200 + len(took),
                    "clientOrderId": cid,
                    "price": str(order.price) if order.has_price else "0",
                    "origQty": str(order.quantity),
                    "executedQty": "0",
                    "status": "NEW",
                    "type": order.order_type.name,
                }
            return SimpleNamespace(status=200, body=json.dumps(body).encode(), headers={})
        return await fake(method, url, **kwargs)

    live.transport.request = request  # type: ignore[method-assign,assignment]
    return took


def _close_fill(live: _Live, order: Any, venue_id: Any = None) -> None:
    """The market close ``order`` filled at 10200 (queued into NT's exec engine)."""
    kernel = live.kernel
    [position] = kernel.cache.positions_open()
    kernel.exec_engine.process(
        TestEventStubs.order_filled(
            order,
            instrument=kernel.cache.instrument(live.iid),
            account_id=_ACCOUNT,
            venue_order_id=venue_id or order.venue_order_id,
            position_id=position.id,
            last_px=Price.from_str("10200.0"),
        )
    )


async def _settle(live: _Live, paper: PaperTradingNode) -> None:
    """Drain NT's queues until the node's deferred SL/TP re-check (if any) has run."""
    for _ in range(40):
        await live.drain()
        if getattr(paper, "_rearm_timer", None) is None:
            break
    await live.drain()


def _protection(live: _Live) -> list[tuple[str, str]]:
    """``(type, status)`` of every reduce-only SL/TP the session holds, sorted."""
    return sorted(
        (o.order_type.name, o.status.name)
        for o in live.kernel.cache.orders()
        if o.is_reduce_only and o.order_type.name in ("STOP_MARKET", "LIMIT")
    )


async def _queue_rejections(live: _Live) -> None:
    """Run NT's queues until the gate refused both start-up legs, then hold the
    ``OrderRejected`` events back in the exec engine's event queue (legs SUBMITTED)."""
    engine = live.kernel.exec_engine
    tasks = [
        asyncio.create_task(engine._run_cmd_queue()),
        asyncio.create_task(live.kernel.risk_engine._run_cmd_queue()),
    ]
    try:
        for _ in range(80):
            if engine.evt_qsize() >= 4:
                break
            await asyncio.sleep(0)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    queued = [engine._evt_queue.get_nowait() for _ in range(engine.evt_qsize())]
    for event in queued:
        if isinstance(event, OrderRejected):
            engine._evt_queue.put_nowait(event)
        else:
            engine._handle_event_with_tracking(event)


async def _adopted_long(live: _Live) -> None:
    live.venue = _healthy_venue(live)  # type: ignore[attr-defined]
    await _long(live)


@pytest.mark.parametrize("stage", ["initialized", "rejection_queued"])
async def test_first_agreement_waits_for_startup_protection_in_flight(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    """B9/B10: the first agreeing check lands while the start-up SL/TP of the adopted long
    are still on their way: queued, not yet at the wire (B9: they go out after the
    agreement), or refused at the wire with the refusal not yet processed (B10). The
    re-check waits until they are answered: Binance ends with exactly one SL and one TP,
    never two pairs, never none."""
    alerts: list[Any] = []
    async with _restarted(seeded, _adopted_long, settle=False) as (paper, live):
        monkeypatch.setattr(paper._journal, "alert", alerts.append)
        if stage == "rejection_queued":
            await _queue_rejections(live)
        at_agreement = _protection(live)
        assert _agree(paper) is True
        await _settle(live, paper)
        took = sorted((o.order_type.name, o.status.name) for o in live.venue)  # type: ignore[attr-defined]
        writes = live.transport.writes()
        held = _protection(live)

    status = "INITIALIZED" if stage == "initialized" else "SUBMITTED"
    assert at_agreement == [("LIMIT", status), ("STOP_MARKET", status)]
    assert took == [("LIMIT", "ACCEPTED"), ("STOP_MARKET", "ACCEPTED")]
    assert sorted(writes) == sorted(_LEGS)
    accepted = [leg for leg in held if leg[1] == "ACCEPTED"]
    assert accepted == [("LIMIT", "ACCEPTED"), ("STOP_MARKET", "ACCEPTED")]
    assert alerts == []


@pytest.mark.parametrize(
    ("agree_at", "refused", "alert", "legs"),
    [
        pytest.param(None, {0, 1}, None, ["LIMIT", "STOP_MARKET"], id="both_refused"),
        pytest.param(0, set(), None, ["LIMIT", "STOP_MARKET"], id="both_accepted"),
        pytest.param(1, {0}, "stop-loss", ["LIMIT"], id="sl_refused_tp_accepted"),
        pytest.param(0, {2}, "take-profit", ["STOP_MARKET"], id="tp_refused_sl_accepted"),
    ],
)
async def test_first_agreement_across_the_startup_legs_leaves_no_silent_one_legged_position(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    agree_at: int | None,
    refused: set[int],
    alert: str | None,
    legs: list[str],
) -> None:
    """B1: the first agreement can land BETWEEN the two start-up legs at the wire. The
    gate asks the unchecked predicate once per check (a write passes it twice, before
    and after its quota wait; a refused write once): ``agree_at`` = the check that
    already sees the agreement, ``refused`` = the checks answered "unchecked" (check 0
    is the SL's, check 2 the TP's when the SL passed). Both refused: the re-check places
    both. Both accepted: nothing is added. One leg each way: the re-check only resizes
    the resting leg, so one critical alert names the missing leg (completing it is not
    done): never a one-legged position without the alert, never a leg placed twice."""
    alerts: list[Any] = []
    async with _restarted(seeded, _adopted_long, settle=False) as (paper, live):
        monkeypatch.setattr(paper._journal, "alert", alerts.append)
        unchecked_text = paper._venue_unchecked()
        assert unchecked_text is not None
        writes_seen: list[int] = []

        def unchecked() -> str | None:
            index = len(writes_seen)
            writes_seen.append(index)
            if index == agree_at:
                assert _agree(paper) is True
            if index in refused:
                return unchecked_text
            return paper._venue_unchecked()

        install_write_gate(live.client, paper._venue_hold, unchecked)
        if agree_at is None:
            await _settle(live, paper)
            assert _agree(paper) is True
        await _settle(live, paper)
        await _settle(live, paper)
        held = _protection(live)
        accepted_legs = sorted(t for t, state in held if state == "ACCEPTED")
        wire = [o.order_type.name for o in live.venue]  # type: ignore[attr-defined]

    if alert is None:
        assert accepted_legs == ["LIMIT", "STOP_MARKET"]
        assert alerts == []
    else:
        assert accepted_legs == legs, held
        [sent] = alerts
        assert sent.critical
        other = "take-profit" if alert == "stop-loss" else "stop-loss"
        assert sent.message == (
            f"paper PAPER-007: {alert} missing on Binance for {live.strategy.id} "
            f"({other} rests); protect it by hand on Binance"
        )
    assert sorted(wire) == sorted(accepted_legs)  # no leg reached the venue twice


async def test_first_agreement_alerts_nothing_for_a_position_closed_while_cancels_pend(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exit's close filled while its cancels are still PENDING_CANCEL: the position is
    flat, so the first-agreement re-check sends no "cancel refused before" alert and no
    order."""
    alerts: list[Any] = []
    async with _restarted(seeded, _inflight_legs_of_another_size) as (paper, live):
        monkeypatch.setattr(paper._journal, "alert", alerts.append)
        close = live.accepted(
            live.strategy.order_factory.market(
                live.iid, OrderSide.SELL, Quantity.from_str("1.000"), reduce_only=True
            )
        )
        _close_fill(live, close)
        await live.drain()
        flat = live.kernel.cache.positions_open() == []
        pending = [o for o in live.kernel.cache.orders() if o.is_pending_cancel]
        assert _agree(paper) is True
        await _settle(live, paper)
        writes = live.transport.writes()

    assert flat
    assert len(pending) == 2
    assert (alerts, writes) == ([], [])


def _exiting() -> Strategy:
    """``_compiled`` with an exit: close > 10100 sends the market close."""
    from vibe_quant.dsl.compiler import StrategyCompiler
    from vibe_quant.dsl.parser import validate_strategy_dict

    dsl = validate_strategy_dict(
        {
            "name": "wire_exit_probe",
            "timeframe": "1m",
            "indicators": {"sma_2": {"type": "SMA", "period": 2}},
            "entry_conditions": {"long": ["sma_2 > 0"]},
            "exit_conditions": {"long": ["close > 10100"]},
            "stop_loss": {"type": "fixed_pct", "percent": 10.0},
            "take_profit": {"type": "fixed_pct", "percent": 10.0},
        }
    )
    module = StrategyCompiler().compile_to_module(dsl)
    cfg = module.WireExitProbeConfig.parse(
        json.dumps({"instrument_id": "BTCUSDT-PERP.BINANCE", "order_id_tag": "000"})
    )
    strategy: Strategy = module.WireExitProbeStrategy(config=cfg)
    return strategy


def _bar(live: _Live, close: str, minute: int) -> None:
    from nautilus_trader.model.data import Bar

    px = Price.from_str(close)
    ts = minute * 60_000_000_000
    live.strategy.handle_bar(
        Bar(live.strategy.primary_bar_type, px, px, px, px, Quantity.from_str("1"), ts, ts)  # type: ignore[attr-defined]
    )


@pytest.mark.parametrize("startup", ["refused", "queued", "close_unfilled"])
async def test_first_agreement_places_no_leg_for_a_closing_position(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    startup: str,
) -> None:
    """B9/B10 exit race: the exit signal fires on the bar that lands with the first
    agreement (its market close still queued). The re-check waits for the close; the
    position is flat after it, so no SL/TP is placed. ``refused``: the start-up legs
    were already refused (none resting); ``queued``: they are still queued too;
    ``close_unfilled``: Binance accepted the close and its fill comes later (the
    re-check waits while the close works)."""
    fill = startup != "close_unfilled"

    async def adopted(live: _Live) -> None:
        live.venue = _healthy_venue(live, fill=fill)  # type: ignore[attr-defined]
        await _long(live)

    async with _restarted(seeded, adopted, strategy=_exiting(), settle=startup != "queued") as (
        paper,
        live,
    ):
        before = {o.client_order_id for o in live.kernel.cache.orders()}
        _bar(live, "10000.0", 1)
        _bar(live, "10000.0", 2)
        _bar(live, "10200.0", 3)  # exit signal: cancel-all + reduce-only market close
        assert _agree(paper) is True
        if not fill:
            await live.drain()
            await live.drain()
            [close] = live.venue  # type: ignore[attr-defined]
            assert close.status == OrderStatus.ACCEPTED
            assert _new_legs(live, before) == []
            assert paper._rearm_timer is not None  # still waiting on the working close
            _close_fill(live, close)
        await _settle(live, paper)
        built = _new_legs(live, before)
        took = [(o.order_type.name, o.status.name) for o in live.venue]  # type: ignore[attr-defined]
        writes = live.transport.writes()
        flat = live.kernel.cache.positions_open() == []

    assert built == []  # the re-check placed no leg
    assert flat
    if startup != "queued":
        assert (took, writes) == ([("MARKET", "FILLED")], [("POST", "/fapi/v1/order")])
    else:
        # The start-up legs leave first, then the close and its cancel-all (this fake
        # venue refuses the cancel-all; Binance would cancel both legs).
        assert took == [("STOP_MARKET", "ACCEPTED"), ("LIMIT", "ACCEPTED"), ("MARKET", "FILLED")]
        assert sorted(writes) == sorted(
            [
                *_LEGS,
                ("POST", "/fapi/v1/order"),
                ("DELETE", "/fapi/v1/algoOpenOrders"),
                ("DELETE", "/fapi/v1/allOpenOrders"),
            ]
        )


async def _unanswered_stop(live: _Live) -> None:
    """An adopted long whose resting TP is known and whose SL NT holds as SUBMITTED
    (sent by the previous process, never answered)."""
    await _long(live)
    factory = live.strategy.order_factory
    qty = Quantity.from_str("1.000")
    stop = factory.stop_market(
        live.iid, OrderSide.SELL, qty, trigger_price=Price.from_str("9000.0"), reduce_only=True
    )
    live.kernel.cache.add_order(stop)
    stop.apply(TestEventStubs.order_submitted(stop, account_id=_ACCOUNT))
    live.kernel.cache.update_order(stop)
    live.accepted(
        factory.limit(
            live.iid, OrderSide.SELL, qty, price=Price.from_str("11000.0"), reduce_only=True
        )
    )
    live.stop = stop  # type: ignore[attr-defined]


async def test_first_agreement_alerts_once_while_an_order_stays_unanswered(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B9/B10: a re-check that keeps waiting on an unanswered SL sends nothing and, past
    REARM_WAIT_ALERT_S, one critical alert asking for a manual check; it keeps waiting
    and decides once Binance answers (here: the SL rests, so nothing is added)."""
    import vibe_quant.paper.node as node_module

    monkeypatch.setattr(node_module, "REARM_WAIT_ALERT_S", 0.0, raising=False)
    alerts: list[Any] = []
    async with _restarted(seeded, _unanswered_stop) as (paper, live):
        monkeypatch.setattr(paper._journal, "alert", alerts.append)
        before = {o.client_order_id for o in live.kernel.cache.orders()}
        assert _agree(paper) is True
        for _ in range(3):
            await live.drain()
        waiting = paper._rearm_timer is not None
        stop = live.stop  # type: ignore[attr-defined]
        live.kernel.exec_engine.process(
            TestEventStubs.order_accepted(
                stop, account_id=_ACCOUNT, venue_order_id=VenueOrderId("9")
            )
        )
        await _settle(live, paper)
        writes = live.transport.writes()
        built = _new_legs(live, before)
        done = paper._rearm_timer is None

    assert (waiting, done) == (True, True)
    assert (writes, built) == ([], [])
    [alert] = alerts
    assert alert.critical
    assert alert.message == (
        "paper PAPER-007: SL/TP check after the Binance check still waits after 0 s for "
        f"orders Binance has not answered for {live.strategy.id}; check its protection on "
        "Binance"
    )


# ------------------------------------------------- B2: what the HOLD alert claims


def _protect(position: str) -> str:
    return f"NO STOP on Binance for {position}: protect it by hand on Binance, then restart."


def _verify(position: str) -> str:
    return (
        f"Stop unconfirmed for {position}: check its stop on Binance before adding or "
        "changing one, then restart."
    )


def _alert_node(live: _Live) -> SimpleNamespace:
    """What ``PaperTradingNode._hold_alert_text`` reads: the guard, the symbol map, the id."""
    return SimpleNamespace(
        _guard=live.guard,
        _symbol=PaperTradingNode._symbol,
        _config=SimpleNamespace(trader_id="PAPER-007"),
        _hold_action=None,
    )


def _leg(live: _Live, state: str, quantity: str = "1.000") -> Any:
    """A BTC exit leg for a long 1.000 in the NT state ``state`` (real order FSM)."""
    factory = live.strategy.order_factory
    qty = Quantity.from_str(quantity)
    if state == "take_profit_only":
        return live.accepted(
            factory.limit(
                live.iid, OrderSide.SELL, qty, price=Price.from_str("11000.0"), reduce_only=True
            )
        )
    side = OrderSide.BUY if state == "wrong_side" else OrderSide.SELL
    if state == "triggered":
        stop = factory.stop_limit(
            live.iid,
            side,
            qty,
            price=Price.from_str("8990.0"),
            trigger_price=Price.from_str("9000.0"),
            reduce_only=True,
        )
    else:
        stop = factory.stop_market(
            live.iid, side, qty, trigger_price=Price.from_str("9000.0"), reduce_only=True
        )
    if state in ("submitted", "submitted_cancel"):
        live.kernel.cache.add_order(stop)
        stop.apply(TestEventStubs.order_submitted(stop, account_id=_ACCOUNT))
        live.kernel.cache.update_order(stop)
    else:
        live.accepted(stop)
    if state == "triggered":
        stop.apply(TestEventStubs.order_triggered(stop))
        live.kernel.cache.update_order(stop)
    if state in ("submitted_cancel", "accepted_cancel"):
        live.strategy.cancel_order(stop)  # held: never sent, NT marks PENDING_CANCEL
    return stop


_RESTING, _UNCONFIRMED, _NO_STOP = "stop resting", "stop unconfirmed", "NO STOP"


@pytest.mark.parametrize(
    ("state", "leg_qty", "venue", "status", "stop"),
    [
        pytest.param("accepted", "1.000", "1.000", OrderStatus.ACCEPTED, _RESTING, id="accepted"),
        pytest.param(
            "accepted", "1.000", None, OrderStatus.ACCEPTED, _RESTING, id="accepted_not_read"
        ),
        pytest.param("accepted", "1.001", "1.000", OrderStatus.ACCEPTED, _RESTING, id="qty_above"),
        pytest.param(
            "accepted", "0.999", "1.000", OrderStatus.ACCEPTED, _UNCONFIRMED, id="qty_below"
        ),
        pytest.param(
            "accepted", "1.000", "2.000", OrderStatus.ACCEPTED, _UNCONFIRMED, id="stale_more"
        ),
        # ro resting_stop.py: a partial close the bot missed; the accepted 1.000 SL still
        # rests for the 0.400 left. The records are stale: unconfirmed, never NO STOP.
        pytest.param(
            "accepted", "1.000", "0.400", OrderStatus.ACCEPTED, _UNCONFIRMED, id="stale_partial"
        ),
        pytest.param(
            "submitted", "1.000", "1.000", OrderStatus.SUBMITTED, _UNCONFIRMED, id="submitted"
        ),
        pytest.param(
            "submitted_cancel",
            "1.000",
            "1.000",
            OrderStatus.PENDING_CANCEL,
            _UNCONFIRMED,
            id="pending_cancel_no_venue_id",
        ),
        pytest.param(
            "accepted_cancel",
            "1.000",
            "1.000",
            OrderStatus.PENDING_CANCEL,
            _UNCONFIRMED,
            id="pending_cancel",
        ),
        pytest.param(
            "triggered", "1.000", "1.000", OrderStatus.TRIGGERED, _UNCONFIRMED, id="triggered"
        ),
        pytest.param(
            "wrong_side", "1.000", "1.000", OrderStatus.ACCEPTED, _NO_STOP, id="wrong_side"
        ),
        pytest.param(
            "take_profit_only", "1.000", "1.000", OrderStatus.ACCEPTED, _NO_STOP, id="tp_only"
        ),
    ],
)
async def test_hold_alert_names_each_stop_state(
    state: str, leg_qty: str, venue: str | None, status: OrderStatus, stop: str
) -> None:
    """B3: three protection states per exposed position. ``stop resting`` only for a
    reduce-only exit-side stop Binance ACCEPTED (venue id known) covering the position
    while the reading agrees with the bot. ``NO STOP`` only when the bot knows of no
    open exit-side stop (wrong side or a TP alone): protect it by hand. Any other leg
    (submitted, pending cancel, triggered, too small, records stale) is ``stop
    unconfirmed``: check it on Binance before adding or changing one. 0 writes."""
    from decimal import Decimal

    async with _live(guard=True) as live:
        entry = live.accepted(
            live.strategy.order_factory.market(live.iid, OrderSide.BUY, Quantity.from_str("1.000"))
        )
        live.kernel.exec_engine.process(
            TestEventStubs.order_filled(
                entry,
                instrument=live.kernel.cache.instrument(live.iid),
                account_id=_ACCOUNT,
                last_px=Price.from_str("10000.0"),
            )
        )
        await live.drain()
        live.held = H
        leg = _leg(live, state, leg_qty)
        node = _alert_node(live)
        reading = None if venue is None else {"BTCUSDT": Decimal(venue)}
        text = PaperTradingNode._hold_alert_text(node, H, reading)  # type: ignore[arg-type]

    assert leg.status == status
    assert live.transport.writes() == []
    binance = "not read" if venue is None else f"long {venue}"
    assert (
        f"Positions (Binance {'positions not read by this check' if venue is None else 'as read by the check that froze it'}; stops per the bot's records): BTCUSDT Binance {binance}, bot long 1.000, {stop}. "
        in text
    )
    exposed = f"BTCUSDT long {venue or '1.000'}"
    action = {
        _RESTING: "Check Binance, then restart.",
        _UNCONFIRMED: _verify(exposed),
        _NO_STOP: _protect(exposed),
    }[stop]
    assert text.endswith(action)
    assert node._hold_action == action


async def test_entry_fill_racing_the_check_is_reported_unprotected(
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """B2 (ro fill_race.py): the entry is accepted, Binance's positionRisk already shows
    the long, the bot has not booked the fill yet: that first reading latches (Q7=A). The
    one HOLD alert names Binance long 1.000 vs bot flat, NO STOP (no leg exists yet), and
    tells the operator to protect it by hand. When the fill lands, the compiled strategy's SL/TP are
    denied at the gate: nothing reaches Binance, the position stays open."""
    from vibe_quant.dsl.compiler import StrategyCompiler
    from vibe_quant.dsl.parser import validate_strategy_dict
    from vibe_quant.paper.venue_config import parse_positions

    dsl = validate_strategy_dict(
        {
            "name": "entry_race",
            "timeframe": "1m",
            "indicators": {"sma_2": {"type": "SMA", "period": 2}},
            "entry_conditions": {"long": ["sma_2 > 0"]},
            "stop_loss": {"type": "fixed_pct", "percent": 10.0},
            "take_profit": {"type": "fixed_pct", "percent": 10.0},
        }
    )
    module = StrategyCompiler().compile_to_module(dsl)
    cfg = module.EntryRaceConfig.parse(
        json.dumps({"instrument_id": "BTCUSDT-PERP.BINANCE", "order_id_tag": "000"})
    )
    strat = module.EntryRaceStrategy(config=cfg)
    paper = PaperTradingNode(_config(seeded))
    alerts: list[Any] = []
    real_record = paper._journal.record

    def record(event_type: Any, data: dict[str, Any], **kwargs: Any) -> Any:
        if kwargs.get("alert") is not None:
            alerts.append(kwargs["alert"])
        return real_record(event_type, data, **{**kwargs, "alert": None})

    paper._journal.record = record  # type: ignore[method-assign]
    try:
        async with _live(guard=True, strategy=strat) as live:
            paper._guard = live.guard
            paper._resolved = SimpleNamespace(symbols=["BTCUSDT"])  # type: ignore[assignment]
            paper._compiled_strategies = [strat]
            entry = live.accepted(
                strat.order_factory.market(live.iid, OrderSide.BUY, Quantity.from_str("1.000"))
            )
            # A saved record (nothing owned yet); the gate admitted the entry.
            paper._ownership = OwnershipLedger(note=None)
            paper._ownership.admit(str(entry.client_order_id))
            reading = no_orders(
                parse_positions(
                    [{"symbol": "BTCUSDT", "positionAmt": "1.000", "positionSide": "BOTH"}],
                    ["BTCUSDT"],
                )
            )
            assert live.kernel.cache.positions_open() == []
            paper._venue_apply_positions(reading, None)
            live.held = paper._venue_hold()
            live.kernel.exec_engine.process(
                TestEventStubs.order_filled(
                    entry,
                    instrument=live.kernel.cache.instrument(live.iid),
                    account_id=_ACCOUNT,
                    last_px=Price.from_str("10000.0"),
                )
            )
            await live.drain()

            assert live.held == "venue position BTCUSDT 1.000 != session 0"
            assert len(live.kernel.cache.positions_open()) == 1
            protective = [o for o in live.kernel.cache.orders() if o.is_reduce_only]
            assert sorted((o.order_type.name, o.status) for o in protective) == [
                ("LIMIT", OrderStatus.DENIED),
                ("STOP_MARKET", OrderStatus.DENIED),
            ]
            assert live.transport.writes() == []
    finally:
        await paper._shutdown()

    [alert] = [a for a in alerts if a.message.startswith("HOLD ")]
    assert (
        "Positions (Binance as read by the check that froze it; stops per the bot's records): "
        "BTCUSDT Binance long 1.000, bot flat, NO STOP. "
    ) in alert.message
    assert alert.message.endswith(_protect("BTCUSDT long 1.000"))


@pytest.mark.parametrize(
    ("answer", "status"),
    [
        pytest.param(b"HTTP/1.1 429 Too Many Requests", 429, id="429"),
        pytest.param(b"HTTP/1.1 418 I'm a teapot", 418, id="418"),
    ],
)
async def test_binance_rate_limit_answer_surfaces_unretried(answer: bytes, status: int) -> None:
    """P1-R3 (3): a Binance 429/418 through the gate (clean) raises NT's
    BinanceClientError with that status, as without the gate; one request, no retry."""
    from nautilus_trader.common.component import LiveClock

    from vibe_quant.paper.wire_gate import BINANCE_FUTURES_DEFAULT_QUOTA, BINANCE_FUTURES_QUOTAS

    async with _loopback(answer=answer) as (url, received, flag):
        http = BinanceHttpClient(LiveClock(), api_key="k", api_secret="s", base_url=url)
        http._client = WriteGate(
            http._client, flag.hold, BINANCE_FUTURES_QUOTAS, BINANCE_FUTURES_DEFAULT_QUOTA
        )
        with pytest.raises(BinanceClientError) as raised:
            await http.sign_request(
                HttpMethod.POST, "/fapi/v1/order", ratelimiter_keys=["binance:fapi/v1/order"]
            )

    assert raised.value.status == status
    assert received == [("POST", "/fapi/v1/order", False)]


# ------------------------------------------------------- R40-R43: bot-owned evidence (P2)

_OWN_SL = "O-20261121-000000-007-000-90"
_HAND_SL = "web_hand_stop_1"
_PRIOR_HALT = {"state": "halted", "halt_reason": "manual", "halt_message": "halt via API"}


class _ClaimConfig(StrategyConfig, frozen=True):  # type: ignore[misc]
    """A plain strategy's config with the instrument the node reads (as compiled ones have)."""

    instrument_id: str = "BTCUSDT-PERP.BINANCE"


def _claiming() -> Strategy:
    """A plain strategy claiming BTC's external orders and positions, as paper's do."""
    from nautilus_trader.model.identifiers import InstrumentId

    iid = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
    return Strategy(_ClaimConfig(order_id_tag="000", external_order_claims=[iid]))


def _write_prior(db: Any, node_status: dict[str, object]) -> None:
    from vibe_quant.paper.persistence import StateCheckpoint, StatePersistence

    store = StatePersistence(db_path=db, trader_id="PAPER-007")
    try:
        store.save_checkpoint(StateCheckpoint(trader_id="PAPER-007", node_status=node_status))
    finally:
        store.close()


def venue_long(amount: str) -> Callable[[_Live], Awaitable[object]]:
    """NT start-up reconciliation of a venue-only BTC position (no order events)."""

    async def reconcile(live: _Live) -> None:
        from nautilus_trader.execution.reports import PositionStatusReport
        from nautilus_trader.model.enums import PositionSide

        if Decimal(amount) == 0:
            return
        report = PositionStatusReport(
            account_id=_ACCOUNT,
            instrument_id=live.iid,
            position_side=PositionSide.LONG,
            quantity=Quantity.from_str(amount),
            report_id=UUID4(),
            ts_last=1,
            ts_init=1,
            avg_px_open=Decimal("10000"),
        )
        assert live.kernel.exec_engine._reconcile_position_report_netting(report)

    return reconcile


def algo_row(cid: str, amount: str | None = "1.000") -> dict[str, object]:
    """One Binance openAlgoOrders row: a BTC SELL stop ``cid`` (``amount`` None = a
    closePosition stop, which has no quantity)."""
    return {
        "algoId": 777,
        "clientAlgoId": cid,
        "algoType": "CONDITIONAL",
        "orderType": "STOP_MARKET",
        "symbol": "BTCUSDT",
        "side": "SELL",
        "algoStatus": "NEW",
        "quantity": amount,
        "closePosition": amount is None,
        "triggerPrice": "9000.0",
        "reduceOnly": amount is not None,
    }


def resting_sl(cid: str, amount: str = "1.000") -> Callable[[_Live], Awaitable[object]]:
    """``venue_long(amount)`` plus a resting reduce-only stop ``cid`` that NT reconciled and
    Binance's open-order read shows."""

    async def reconcile(live: _Live) -> None:
        from nautilus_trader.model.identifiers import ClientOrderId

        await venue_long(amount)(live)
        sl = live.strategy.order_factory.stop_market(
            live.iid,
            OrderSide.SELL,
            Quantity.from_str(amount),
            trigger_price=Price.from_str("9000.0"),
            reduce_only=True,
            client_order_id=ClientOrderId(cid),
        )
        live.accepted(sl)
        live.transport.rows[OPEN_ALGO].append(algo_row(cid, amount))

    return reconcile


@contextlib.asynccontextmanager
async def owned_restart(
    seeded: dict[str, Any],  # noqa: F811
    prior: dict[str, object] | None,
    before_start: Callable[[_Live], Awaitable[object]],
) -> AsyncIterator[tuple[PaperTradingNode, _Live]]:
    """A restart through the REAL restore path: ``prior`` (None = no checkpoint) is the
    saved checkpoint, NT reconciles ``before_start`` into the cache, the guard reports to
    the REAL node bridge and the REAL node predicates gate the REAL Binance client."""
    from vibe_quant.paper.node import ResolvedRunConfig, _GuardBridge
    from vibe_quant.paper.persistence import StatePersistence

    if prior is not None:
        _write_prior(seeded["db"], prior)
    paper = PaperTradingNode(_config(seeded))
    paper._persistence = StatePersistence(db_path=seeded["db"], trader_id="PAPER-007")
    try:
        paper._restore_from_checkpoint()
    finally:
        paper._persistence.close()
        paper._persistence = None
    paper._resolved = ResolvedRunConfig(["BTCUSDT"], {}, 5, "CROSSED")
    bridge = _GuardBridge(paper)
    from vibe_quant.paper.venue_config import open_orders_read

    listen = bridge.on_order_event

    async def reconcile(live: _Live) -> None:
        # NT reconciles before the trader (and the guard) starts: no order event of it
        # reaches the node.
        bridge.on_order_event = lambda event: None  # type: ignore[method-assign]
        try:
            await before_start(live)
            await live.drain()
        finally:
            bridge.on_order_event = listen  # type: ignore[method-assign]

    try:
        async with _live(
            guard=True,
            strategy=_claiming(),
            restored=paper._restored_halt,
            verified=False,
            before_start=reconcile,
            listener=bridge,
        ) as live:
            install_write_gate(live.client, paper._venue_hold, paper._venue_unchecked)
            paper._guard = live.guard
            paper._compiled_strategies = [live.strategy]
            # The node's own open-order reads go through the gated client's transport.
            read = open_orders_read(live.http, live.kernel.clock, ["BTCUSDT"])
            paper._build_open_orders_read = lambda: read  # type: ignore[method-assign]
            await live.drain()
            yield paper, live
    finally:
        await paper._shutdown()


def no_orders(reading: Any) -> Any:
    """``reading`` with both open-order reads complete and empty (no order on Binance)."""
    from dataclasses import replace

    from vibe_quant.paper.venue_config import OpenOrders

    return replace(reading, orders=OpenOrders([], None))


def _btc_rows(amount: str) -> list[dict[str, str]]:
    if Decimal(amount) == 0:
        return []
    return [{"symbol": "BTCUSDT", "positionSide": "BOTH", "positionAmt": amount}]


def btc_reading(amount: str) -> Any:
    """A positionRisk reading of BTC ``amount`` and no open order on Binance."""
    from vibe_quant.paper.venue_config import parse_positions

    return no_orders(parse_positions(_btc_rows(amount), ["BTCUSDT"]))


async def venue_reading(
    paper: PaperTradingNode, amount: str, rows: list[object] | None = None
) -> tuple[str | None, str | None]:
    """One venue position reading through the node's real reads: positionRisk answers BTC
    ``amount`` (or ``rows``); the open-order reads go through the gated client to the
    fake transport. Returns what keeps the cycle from passing (None, None = nothing)."""

    async def position_risk() -> object:
        return _btc_rows(amount) if rows is None else rows

    reading, failure = await paper._venue_read_positions(
        position_risk, paper._build_open_orders_read(), cast("Any", paper._resolved)
    )
    return paper._venue_apply_positions(reading, failure)


async def first_check(
    paper: PaperTradingNode, live: _Live, amount: str, rows: list[object] | None = None
) -> bool:
    """The node's first venue check, as one cycle concludes it: one reading (positionRisk
    of BTC ``amount`` or ``rows``, plus Binance's open orders) and a matching
    leverage/margin answer, then the guard's ticks. True = it passed."""
    from vibe_quant.paper.venue_config import classify_symbol_configs

    problem, unpublishable = await venue_reading(paper, amount, rows)
    body = [{"symbol": "BTCUSDT", "leverage": 5, "marginType": "CROSSED"}]
    verdict = classify_symbol_configs(body, ["BTCUSDT"], 5, "CROSSED")
    agreed = paper._venue_conclude(verdict, None, problem, None, unpublishable)
    guard = live.guard
    assert guard is not None
    for _ in range(3):
        guard._on_tick(None)
    await live.drain()
    return agreed


def _market_closes(live: _Live) -> list[tuple[str, str, bool]]:
    from nautilus_trader.model.enums import OrderType

    return [
        (o.side.name, str(o.quantity), o.is_reduce_only)
        for o in live.kernel.cache.orders()
        if o.order_type == OrderType.MARKET
    ]


@pytest.mark.parametrize(
    ("owned_net", "own_orders", "before_start", "passes", "hold"),
    [
        pytest.param(
            {"BTCUSDT": "0"},
            {},
            venue_long("1.000"),
            False,
            "venue position BTCUSDT 1.000 != session 0",
            id="manual_position",
        ),
        pytest.param(
            {"BTCUSDT": "1.000"},
            {_OWN_SL: "0"},
            resting_sl(_OWN_SL),
            True,
            None,
            id="bot_own_control",
        ),
        pytest.param(
            {"BTCUSDT": "1.000"},
            {},
            resting_sl(_HAND_SL),
            False,
            f"foreign order {_HAND_SL} SELL 1.000 on BTCUSDT",
            id="manual_resting_order",
        ),
    ],
)
async def test_manual_position_on_bot_coin_is_held(
    seeded: dict[str, Any],  # noqa: F811
    owned_net: dict[str, str],
    own_orders: dict[str, str],
    before_start: Callable[[_Live], Awaitable[object]],
    passes: bool,
    hold: str | None,
) -> None:
    """R40/R41 (user Q3): a restored HALTED session; NT reconciled a BTC long 1.000. The
    bot's saved record says what it owns. Owned 0 (a hand position) or a resting order the
    bot never placed: the first check holds, the restored halt never closes it, nothing
    reaches Binance. Owned 1.000 with its own stop (control): the check passes and the
    halt closes the bot's position (one reduce-only SELL 1.000)."""
    prior = {**_PRIOR_HALT, "owned_net": owned_net, "own_orders": own_orders}
    async with owned_restart(seeded, prior, before_start) as (paper, live):
        assert paper._restored_halt is not None
        assert live.transport.writes() == [], "nothing before the check"
        agreed = await first_check(paper, live, "1.000")
        writes = live.transport.writes()
        held = paper._venue_hold()
        closes = _market_closes(live)

    assert agreed is passes
    if hold is None:
        assert held is None
        assert closes == [("SELL", "1.000", True)]
        assert ("POST", "/fapi/v1/order") in writes
    else:
        assert held is not None and held.startswith(hold), held
        assert (writes, closes) == ([], [])


def _ws_order(
    cid: str,
    *,
    side: str = "BUY",
    status: str = "FILLED",
    execution: str = "TRADE",
    order_type: str = "MARKET",
    price: str = "0",
    filled: str = "1.000",
) -> bytes:
    """One Binance futures ORDER_TRADE_UPDATE user-stream message for BTCUSDT."""
    last = filled if execution in ("TRADE", "CALCULATED") else "0"
    return json.dumps(
        {
            "e": "ORDER_TRADE_UPDATE",
            "E": 1_700_000_000_000,
            "T": 1_700_000_000_000,
            "o": {
                "s": "BTCUSDT",
                "c": cid,
                "S": side,
                "o": order_type,
                "f": "GTC",
                "q": "1.000",
                "p": price,
                "ap": "10000" if last != "0" else "0",
                "sp": "0",
                "x": execution,
                "X": status,
                "i": 777,
                "l": last,
                "z": filled,
                "L": "10000" if last != "0" else "0",
                "N": "USDT",
                "n": "0",
                "T": 1_700_000_000_000,
                "t": 4242,
                "b": "0",
                "a": "0",
                "m": False,
                "R": False,
                "wt": "CONTRACT_PRICE",
                "ot": order_type,
                "ps": "BOTH",
                "pP": False,
                "si": 0,
                "ss": 0,
                "rp": "0",
                "gtd": 0,
            },
        }
    ).encode()


@pytest.mark.parametrize(
    ("message", "hold"),
    [
        pytest.param(
            _ws_order("web_hand_buy"), "foreign order web_hand_buy BUY 1.000 on BTCUSDT", id="fill"
        ),
        pytest.param(
            _ws_order(
                "web_hand_limit",
                status="NEW",
                execution="NEW",
                order_type="LIMIT",
                price="9000",
                filled="0",
            ),
            "foreign order web_hand_limit BUY 1.000 on BTCUSDT",
            id="resting_limit",
        ),
        pytest.param(
            _ws_order("autoclose-1700000000000", side="SELL", execution="CALCULATED"),
            "foreign order autoclose-1700000000000 SELL 1.000 on BTCUSDT",
            id="binance_liquidation",
        ),
    ],
)
async def test_foreign_fill_latches(
    seeded: dict[str, Any],  # noqa: F811
    message: bytes,
    hold: str,
) -> None:
    """R42/R43: a running, checked session; Binance's user stream reports an order the
    bot never placed on its coin (a hand market buy, a hand limit, a Binance liquidation).
    The hold latches at once and names it; the bot's own record does not move; a bot
    cancel-all afterwards (Binance's is symbol-wide: it would cancel the user's order)
    sends nothing."""
    prior = {"state": "running", "owned_net": {"BTCUSDT": "0"}, "own_orders": {}}
    async with owned_restart(seeded, prior, venue_long("0")) as (paper, live):
        assert await first_check(paper, live, "0") is True
        assert paper._venue_hold() is None
        live.client._handle_user_ws_message(message)
        await live.drain()
        held = paper._venue_hold()
        before = list(live.transport.writes())
        live.strategy.cancel_all_orders(live.iid)
        await live.drain()
        after = live.transport.writes()[len(before) :]
        owned = paper._ownership.owned("BTCUSDT")

    assert held is not None and held.startswith(hold), held
    assert after == []
    assert owned == 0


async def test_hand_position_gets_no_stop_and_no_close(
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """P2: the compiled strategy restarts on an adopted BTC long the bot has no record of
    (a hand position), with a halt owed. The first check holds: the first-agreement SL/TP
    re-check places no leg and the owed flatten never closes it; nothing reaches Binance."""
    async with _restarted(seeded, _long) as (paper, live):
        guard = live.guard
        assert guard is not None
        guard.halt(HaltReason.MANUAL, "owed close")
        paper._venue_apply_positions(btc_reading("1.000"), None)
        agreed = _agree(paper)
        for _ in range(3):
            guard._on_tick(None)
        await _settle(live, paper)
        writes = live.transport.writes()
        held = paper._venue_hold()

    assert agreed is False
    assert held is not None and held.startswith("venue position BTCUSDT 1.000 != session 0")
    assert writes == []
