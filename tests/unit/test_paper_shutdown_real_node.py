"""Stop-requested disconnect on a REAL NautilusTrader ``TradingNode`` (vibe-quant-yejfx).

The fake-node cases in ``test_paper_shutdown_connectivity_alert.py`` pin the node's
rule; these run the real ``kernel.stop_async`` with real data/exec client objects (no
network) to prove the rule against what NT actually does: ``stop_async`` sets
``kernel._is_stopping`` and then disconnects the clients, and a venue lost BEFORE the
stop must still page even when the operator stops the node right after.

Rule (chief R2, fail open): a disconnect during a stop is silent only when the node saw
the venue CONNECTED after the stop began.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from nautilus_trader.live.factories import LiveExecClientFactory

from tests.unit.test_audit_paper_node import DSL, _config
from tests.unit.test_paper_margin_refusal_real_node import (
    _FastNodeConfig,
    _StubDataFactory,
    _StubExecClient,
)
from vibe_quant.db.state_manager import StateManager
from vibe_quant.nt_compat import retain_log_guard
from vibe_quant.paper.node import PaperTradingNode

if TYPE_CHECKING:
    from pathlib import Path

_TRANSIENT = "transient_error: venue connection lost"


class _GatedExecClient(_StubExecClient):
    """Healthy exec client whose ``_disconnect`` parks until the test opens the gate."""

    gate: asyncio.Event | None = None

    async def _connect(self) -> None:
        from nautilus_trader.model.identifiers import AccountId

        self.connect_attempts += 1
        self._set_account_id(AccountId("BINANCE-001"))

    async def _disconnect(self) -> None:
        if _GatedExecClient.gate is not None:
            await _GatedExecClient.gate.wait()


class _GatedExecFactory(LiveExecClientFactory):
    created: list[_GatedExecClient] = []

    @staticmethod
    def create(loop: Any, name: str, config: Any, msgbus: Any, cache: Any, clock: Any) -> Any:  # type: ignore[override]
        client = _GatedExecClient(loop, msgbus, cache, clock)
        _GatedExecFactory.created.append(client)
        return client


class _Sent:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bool]] = []

    def __call__(self, kind: str, message: str, *, critical: bool = False) -> None:
        self.calls.append((kind, message, critical))


@pytest.fixture()
def seeded(tmp_path: Path) -> dict[str, Any]:
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


async def _wait(predicate: Any, what: str, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, f"timed out waiting for {what}"
        await asyncio.sleep(0.01)


class _Venue:
    """A running paper node over a real TradingNode, already seen connected."""

    node: PaperTradingNode
    trading_node: Any
    client: _GatedExecClient
    sent: _Sent
    run_task: asyncio.Task[None]
    stop_task: asyncio.Task[None] | None = None

    def stopping(self) -> bool:
        return bool(self.trading_node.kernel._is_stopping)

    async def begin_stop(self) -> None:
        """Operator stop: ``stop_async`` runs; its client disconnect parks on the gate."""
        self.stop_task = asyncio.create_task(self.trading_node.stop_async())
        await _wait(self.stopping, "kernel._is_stopping")

    def venue_connected(self) -> bool:
        kernel = self.trading_node.kernel
        return bool(kernel.data_engine.check_connected() and kernel.exec_engine.check_connected())


@pytest.fixture()
async def venue(seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> Any:
    import nautilus_trader.adapters.binance.factories as binance_factories

    monkeypatch.setattr(binance_factories, "BinanceLiveExecClientFactory", _GatedExecFactory)
    monkeypatch.setattr(binance_factories, "BinanceLiveDataClientFactory", _StubDataFactory)
    monkeypatch.setattr("nautilus_trader.live.config.TradingNodeConfig", _FastNodeConfig)
    _GatedExecFactory.created = []
    _GatedExecClient.gate = asyncio.Event()

    v = _Venue()
    v.node = PaperTradingNode(_config(seeded, trader_id="PAPER-010"))
    v.sent = _Sent()
    monkeypatch.setattr(v.node, "_send_telegram_alert", v.sent)
    await v.node._initialize()
    v.trading_node = v.node._trading_node
    assert v.trading_node is not None
    v.run_task = asyncio.create_task(v.trading_node.run_async())
    try:
        await _wait(v.venue_connected, "real engines connected")
        v.client = _GatedExecFactory.created[0]
        v.node._check_connectivity()
        assert v.node._connected is True
        yield v
    finally:
        assert _GatedExecClient.gate is not None
        _GatedExecClient.gate.set()
        if v.stop_task is not None:
            await asyncio.gather(v.stop_task, return_exceptions=True)
        v.run_task.cancel()
        await asyncio.gather(v.run_task, return_exceptions=True)
        v.node._shutdown_event.set()
        await asyncio.wait_for(v.node._shutdown(), timeout=10.0)
        retain_log_guard(v.trading_node)


def _lost(v: _Venue) -> None:
    """The venue drops: the real exec client flips to disconnected."""
    v.client._set_connected(False)  # type: ignore[no-untyped-call]
    assert v.venue_connected() is False


async def test_disconnect_then_stop_still_alerts(venue: _Venue) -> None:
    """(a) venue lost while running, operator stops before the next check: alert."""
    _lost(venue)
    await venue.begin_stop()
    venue.node._check_connectivity()

    assert venue.node._connected is False
    assert [m for _, m, _ in venue.sent.calls if _TRANSIENT in m], venue.sent.calls


async def test_stop_teardown_disconnect_is_silent(venue: _Venue) -> None:
    """(b) stop begins with the venue up; the teardown disconnect is not a failure."""
    await venue.begin_stop()
    venue.node._check_connectivity()  # observed connected after the stop began
    assert venue.node._connected is True
    _lost(venue)  # NT's own client teardown
    venue.node._check_connectivity()

    assert venue.node._connected is False
    assert [m for _, m, _ in venue.sent.calls if "connection lost" in m] == []


async def test_disconnect_while_running_alerts(venue: _Venue) -> None:
    """(c) the plain case, no stop: alert."""
    _lost(venue)
    assert venue.stopping() is False
    venue.node._check_connectivity()

    assert [m for _, m, _ in venue.sent.calls if _TRANSIENT in m], venue.sent.calls
