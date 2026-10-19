"""A stop-requested disconnect is not a venue failure (bead vibe-quant-yejfx).

The paper control loop polls connectivity every tick. On SIGTERM Nautilus runs its own
graceful shutdown: ``kernel.stop_async`` sets ``_is_stopping`` and then disconnects the
data and execution clients while ``run_async`` is still returning. A control tick in
that window used to report ``transient_error: venue connection lost`` and page the
operator even though the disconnect is exactly the stop that was requested (observed
on testnet PAPER-901: two ``sendMessage`` calls after SIGTERM).

Both sides are pinned here:

* a disconnect while the node is running still alerts (once, through ErrorHandler);
* a disconnect while shutdown is in progress is silent -- for either marker,
  ``kernel._is_stopping`` (the NT-internal signal, set BEFORE the clients disconnect)
  and the node's own shutdown event.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_audit_paper_node import DSL
from tests.unit.test_paper_startup_and_estimate import (
    _actions,
    _Alerts,
    _connect_venue,
    _FakeTradingNode,
    _initialized,
)
from vibe_quant.db.state_manager import StateManager

if TYPE_CHECKING:
    from pathlib import Path

    from vibe_quant.paper.node import PaperTradingNode

#: ErrorHandler's first transient alert for the connectivity probe (node.py
#: ``_on_error_alert``: ``f"{alert_type}: {context.error}\\nOp: {context.operation}"``).
_TRANSIENT_MESSAGE = (
    "transient_error: venue connection lost (data or execution client disconnected)"
    "\nOp: connectivity"
)


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


@pytest.fixture()
def fake_tradingnode(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTradingNode]:
    import nautilus_trader.live.node as live_node

    _FakeTradingNode.instances = []
    monkeypatch.setattr(live_node, "TradingNode", _FakeTradingNode)
    return _FakeTradingNode


async def _connected_node(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[PaperTradingNode, _Alerts]:
    """An initialized node that has already seen its first healthy connectivity tick."""
    node = await _initialized(seeded)
    alerts = _Alerts()
    monkeypatch.setattr(node, "_send_telegram_alert", alerts)
    _connect_venue(fake_tradingnode)
    node._check_connectivity()
    assert node._connected is True
    return node, alerts


async def test_disconnect_while_running_alerts(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real disconnect while running still pages: one RISK_CHECK alert + Telegram."""
    node, alerts = await _connected_node(seeded, fake_tradingnode, monkeypatch)
    try:
        _connect_venue(fake_tradingnode, connected=False)
        node._check_connectivity()

        assert node._connected is False
        assert alerts.calls == [("error", _TRANSIENT_MESSAGE, False)]
        payloads = _actions(node, seeded, "alert")
        assert len(payloads) == 1
        assert payloads[0]["alert_type"] == "transient_error"
        assert payloads[0]["operation"] == "connectivity"
    finally:
        await node._shutdown()


@pytest.mark.parametrize("marker", ["kernel-stopping", "shutdown-requested"])
async def test_disconnect_during_shutdown_is_silent(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
    marker: str,
) -> None:
    """A stop-requested disconnect sends no alert and writes no RISK_CHECK alert row."""
    node, alerts = await _connected_node(seeded, fake_tradingnode, monkeypatch)
    try:
        if marker == "kernel-stopping":
            # NT 1.231: set at the top of kernel.stop_async, before either client
            # disconnects, and cleared only when stop_async returns.
            fake_tradingnode.instances[0].kernel._is_stopping = True  # type: ignore[attr-defined]
        else:
            node._shutdown_event.set()
        node._check_connectivity()  # R2: the venue is seen CONNECTED after the stop began
        _connect_venue(fake_tradingnode, connected=False)
        node._check_connectivity()

        assert node._connected is False
        assert alerts.calls == []
        assert _actions(node, seeded, "alert") == []
    finally:
        node._shutdown_event.set()
        await node._shutdown()


async def test_stopping_marker_cleared_alerts_again(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The suppression is scoped to the stop: a disconnect after it clears still alerts."""
    node, alerts = await _connected_node(seeded, fake_tradingnode, monkeypatch)
    try:
        kernel = fake_tradingnode.instances[0].kernel
        kernel._is_stopping = True  # type: ignore[attr-defined]
        node._check_connectivity()  # connected after the stop began
        _connect_venue(fake_tradingnode, connected=False)
        node._check_connectivity()
        assert alerts.calls == []

        kernel._is_stopping = False  # type: ignore[attr-defined]
        _connect_venue(fake_tradingnode)
        node._check_connectivity()
        assert node._connected is True
        alerts.calls.clear()  # the "venue connection restored" message is not the subject
        _connect_venue(fake_tradingnode, connected=False)
        node._check_connectivity()

        assert alerts.calls == [("error", _TRANSIENT_MESSAGE, False)]
    finally:
        await node._shutdown()


@pytest.mark.parametrize("marker", ["kernel-stopping", "shutdown-requested"])
async def test_disconnect_before_any_connected_stop_observation_alerts(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
    marker: str,
) -> None:
    """Fail open: the first post-stop observation is already disconnected -> alert."""
    node, alerts = await _connected_node(seeded, fake_tradingnode, monkeypatch)
    try:
        if marker == "kernel-stopping":
            fake_tradingnode.instances[0].kernel._is_stopping = True  # type: ignore[attr-defined]
        else:
            node._shutdown_event.set()
        _connect_venue(fake_tradingnode, connected=False)
        node._check_connectivity()

        assert alerts.calls == [("error", _TRANSIENT_MESSAGE, False)]
    finally:
        node._shutdown_event.set()
        await node._shutdown()
