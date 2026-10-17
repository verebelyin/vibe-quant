"""Unit tests can never reach the network (bead vibe-quant-j9siz).

A session-wide autouse guard in ``tests/unit/conftest.py`` patches
``socket.socket.connect``, ``socket.socket.connect_ex`` and ``socket.getaddrinfo``
so that a connect (or a DNS resolution) for any address that is not loopback
(``127.0.0.0/8``, ``::1``, ``localhost``) and not an ``AF_UNIX`` path raises
``RuntimeError("unit tests must not use the network: <host>:<port>")``. Loopback
stays allowed so ``TestClient``, local servers and the event loop's self-pipe keep
working.

Contract rows (brief cc-gr-hermetic-1, task gr-hermetic):

* **N1** the guard blocks a non-loopback ``connect`` / ``connect_ex``; loopback and
  ``AF_UNIX`` are untouched (``socket.create_connection`` resolves through
  ``getaddrinfo`` and then calls the patched ``socket.socket.connect``, so no
  separate patch is needed).
* **N2** a test marked ``@pytest.mark.allow_network`` skips the guard entirely: the
  real ``socket`` functions stay in place.
* **N3** ``socket.getaddrinfo`` for a non-loopback host is blocked too (DNS leak).
* **N4** the paper node's startup path -- the one risk-sb-t2a-1 drove with a fake
  trading node into a real Binance call -- builds and polls the liquidation watcher
  without a network connection once the fetch is stubbed (the same seam
  ``tests/unit/test_paper_liquidation_detector.py`` uses).
"""

from __future__ import annotations

import asyncio
import logging
import socket
from typing import TYPE_CHECKING, Any

import pytest

import vibe_quant.paper.node as node_module
from tests.unit.test_audit_paper_node import DSL, _config
from vibe_quant.db.state_manager import StateManager
from vibe_quant.paper.node import PaperTradingNode

if TYPE_CHECKING:
    from pathlib import Path

#: Captured at collection time, before any fixture patches them: the allow_network
#: test asserts the guard leaves these exact objects in place.
_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex

#: TEST-NET-3 (RFC 5737): reserved for documentation, never routable.
_TEST_NET_HOST = "203.0.113.1"
_GUARD_MESSAGE = f"unit tests must not use the network: {_TEST_NET_HOST}:443"


# --------------------------------------------------------------------------- N1/N3


def test_connect_to_a_public_address_raises_the_guard_error() -> None:
    """N1: a non-loopback connect is refused with the guard's RuntimeError."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    try:
        with pytest.raises(RuntimeError, match=_GUARD_MESSAGE):
            sock.connect((_TEST_NET_HOST, 443))
    finally:
        sock.close()


def test_connect_ex_to_a_public_address_raises_the_guard_error() -> None:
    """N1: ``connect_ex`` is guarded the same way (it must not return an errno)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    try:
        with pytest.raises(RuntimeError, match=_GUARD_MESSAGE):
            sock.connect_ex((_TEST_NET_HOST, 443))
    finally:
        sock.close()


def test_getaddrinfo_for_a_public_host_raises_the_guard_error() -> None:
    """N3: resolving a non-loopback host (a DNS leak) is refused too."""
    with pytest.raises(RuntimeError, match=_GUARD_MESSAGE):
        socket.getaddrinfo(_TEST_NET_HOST, 443)


def test_loopback_connect_and_resolution_are_allowed() -> None:
    """N1: loopback stays usable -- local servers, TestClient, the loop's self-pipe."""
    assert socket.getaddrinfo("127.0.0.1", 0)
    assert socket.getaddrinfo("localhost", 0)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(0.5)
    try:
        client.connect(server.getsockname())
    finally:
        client.close()
        server.close()


def test_the_guard_is_installed_for_an_ordinary_test() -> None:
    """Control: an unmarked test really runs under the guard."""
    assert socket.socket.connect is not _REAL_CONNECT
    assert socket.socket.connect_ex is not _REAL_CONNECT_EX


# ---------------------------------------------------------------------- N2


@pytest.mark.allow_network
def test_allow_network_marker_skips_the_guard() -> None:
    """N2: a marked test keeps the real socket functions (no real connect here)."""
    assert socket.socket.connect is _REAL_CONNECT
    assert socket.socket.connect_ex is _REAL_CONNECT_EX


# ----------------------------------------------------------------- fixture: node


@pytest.fixture()
def seeded(tmp_path: Path) -> dict[str, Any]:
    """A strategy plus a completed validation run (same shape as the other tests)."""
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


class _Engine:
    def check_connected(self) -> bool:
        return True


class _Kernel:
    class _RiskEngine:
        def execute(self, command: object) -> None:
            pass

    def __init__(self) -> None:
        self.risk_engine = self._RiskEngine()
        self.data_engine = _Engine()
        self.exec_engine = _Engine()


class _Trader:
    def __init__(self) -> None:
        self.actors: list[object] = []
        self.strategies: list[object] = []

    def add_actor(self, actor: object) -> None:
        self.actors.append(actor)

    def add_strategy(self, strategy: object) -> None:
        self.strategies.append(strategy)


class _FakeTradingNode:
    """Stand-in for NT's ``TradingNode`` (copied from test_paper_startup_and_estimate).

    Building it must never open a socket: only ``run_async`` connects, and the paper
    startup path under test stops before that.
    """

    instances: list[_FakeTradingNode] = []

    def __init__(self, config: Any) -> None:
        self.config = config
        self.kernel = _Kernel()
        self.trader = _Trader()
        self.built = False
        _FakeTradingNode.instances.append(self)

    def add_data_client_factory(self, name: str, factory: object) -> None:
        pass

    def add_exec_client_factory(self, name: str, factory: object) -> None:
        pass

    def build(self) -> None:
        self.built = True

    async def run_async(self) -> None:
        self.trader.start()

    async def stop_async(self) -> None:
        return


@pytest.fixture()
def fake_tradingnode(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTradingNode]:
    import nautilus_trader.live.node as live_node

    _FakeTradingNode.instances = []
    monkeypatch.setattr(live_node, "TradingNode", _FakeTradingNode)
    return _FakeTradingNode


class _FakeFetch:
    """Stands in for the forceOrders fetcher bound by ``build_binance_fetch``."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, int, int]] = []

    async def __call__(self, symbol: str, start_ms: int, end_ms: int) -> list[object]:
        self.requests.append((symbol, start_ms, end_ms))
        return []


class _FetchFactory:
    """Stands in for ``build_binance_fetch`` (the detector test's seam)."""

    def __init__(self, fetch: _FakeFetch) -> None:
        self.fetch = fetch
        self.credentials: list[tuple[str, str, str, bool]] = []

    def __call__(
        self, api_key: str, api_secret: str, account_type: str, testnet: bool
    ) -> _FakeFetch:
        self.credentials.append((api_key, api_secret, account_type, testnet))
        return self.fetch


# ---------------------------------------------------------------------- N4


async def test_paper_node_startup_polls_the_watcher_without_using_the_network(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """N4(d): the paper startup path opens no network connection.

    The fakes (``_FakeTradingNode``, the fetch seam) are copied from
    ``tests/unit/test_paper_startup_and_estimate.py`` and
    ``tests/unit/test_paper_liquidation_detector.py``. ``build_binance_fetch`` is
    stubbed because the real one builds NautilusTrader's pyo3 ``HttpClient`` (Rust
    ``reqwest``): its connect bypasses Python's ``socket`` module, so the guard
    cannot see it -- a unit test must never build it. With the fetch stubbed the
    node's own ``_build_liquidation_watcher`` and watch loop run for real and reach
    the fetch, and the guard logs nothing.
    """
    fetch = _FakeFetch()
    factory = _FetchFactory(fetch)
    monkeypatch.setattr(node_module, "build_binance_fetch", factory)

    node = PaperTradingNode(_config(seeded))
    await node._initialize()
    try:
        node._start_liquidation_task()
        task = node._liquidation_task
        assert task is not None, "the watch task must exist beside the control loop"
        for _ in range(500):
            if fetch.requests:
                break
            await asyncio.sleep(0)
        assert fetch.requests, "the watch loop must reach the stubbed fetch"
        assert factory.credentials == [("key", "secret", "USDT_FUTURES", True)]
    finally:
        await node._shutdown()

    assert node._liquidation_watcher is not None, "the node built its own watcher"
    assert _FakeTradingNode.instances, "the startup path built the (fake) trading node"
    assert "must not use the network" not in caplog.text


# ---------------------------------------------------------------------- N6


async def test_binance_http_client_send_request_raises_the_guard_error() -> None:
    """N6: NT's pyo3-backed client is blocked -- the exact path risk-sb-t2a-1 took."""
    from nautilus_trader.adapters.binance.http.client import BinanceHttpClient
    from nautilus_trader.common.component import LiveClock
    from nautilus_trader.core.nautilus_pyo3 import HttpMethod

    client = BinanceHttpClient(
        clock=LiveClock(),
        api_key="dummy",
        api_secret="dummy",
        base_url="https://fapi.binance.com",
    )
    with pytest.raises(
        RuntimeError, match="unit tests must not use the network: BinanceHttpClient"
    ):
        await client.send_request(HttpMethod.GET, "/fapi/v1/ping")


async def test_paper_start_path_without_a_fetch_stub_is_blocked_by_the_guard(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """N6: with no fetch stub the node's own watcher hits the guard, not Binance.

    This is the N4(d) path with ``build_binance_fetch`` left real: the watcher builds
    NT's pyo3 client and its first fetch is refused, so the guard now covers the
    original leak instead of only documenting it.
    """
    node = PaperTradingNode(_config(seeded))
    await node._initialize()
    try:
        with caplog.at_level(logging.WARNING):
            node._start_liquidation_task()
            for _ in range(500):
                if "must not use the network" in caplog.text:
                    break
                await asyncio.sleep(0)
    finally:
        await node._shutdown()

    assert node._liquidation_watcher is not None, "the node built the real watcher"
    assert "unit tests must not use the network: BinanceHttpClient" in caplog.text
