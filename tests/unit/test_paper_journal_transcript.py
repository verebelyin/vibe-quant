"""Golden transcript of one paper node session (bead vibe-quant-rabmo.11a, T11).

Characterization only: pins the paper event-log JSONL lines (timestamps stripped) and
the ordered Telegram alert calls of a full session - start, trade open/close, halt,
resume, event-log outage and recovery, stop - before ``rabmo.11`` moves the journal
out of ``PaperTradingNode`` into its own module. The seams driven here are the ones
T13a keeps: ``_write_event``, ``_write_then_alert``, ``_event_writer``, ``_telegram``
and the guard bridge callbacks. The same transcript must stay green after the move.

No production code is touched and nothing here can reach the network: the start probe
returns a config and ``TelegramBot`` is a recording fake (the repo's no-network guard).
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_audit_paper_node import _config
from tests.unit.test_audit_paper_node import seeded as _base_seeded
from tests.unit.test_paper_liquidation_detector import (
    fake_tradingnode,  # noqa: F401  # fixture
)
from tests.unit.test_paper_trade_alerts import _Cache, _Order, _Trip
from vibe_quant.alerts.telegram import AlertType, TelegramConfig
from vibe_quant.logging.events import EventType
from vibe_quant.paper import node as node_module
from vibe_quant.paper.guard import GuardState, HaltReason
from vibe_quant.paper.node import PaperTradingNode

if TYPE_CHECKING:
    from pathlib import Path

#: One alert call: (method, alert type for ``send_alert`` else None, text, bypass).
_AlertCall = tuple[str, str | None, str, bool]


@pytest.fixture()
def seeded(tmp_path: Path) -> dict[str, Any]:
    return _base_seeded.__wrapped__(tmp_path)  # type: ignore[attr-defined,no-any-return]


class _RecordingTelegram:
    """Recording stand-in for ``TelegramBot``: every send is kept in order."""

    instances: list[_RecordingTelegram] = []

    def __init__(self, config: TelegramConfig) -> None:
        self.config = config
        self.calls: list[_AlertCall] = []
        self.closed = False
        _RecordingTelegram.instances.append(self)

    async def send_alert(
        self, alert_type: AlertType, message: str, *, bypass_rate_limit: bool = False
    ) -> bool:
        self.calls.append(("send_alert", alert_type.value, message, bypass_rate_limit))
        return True

    async def send_error(self, message: str, *, bypass_rate_limit: bool = False) -> bool:
        self.calls.append(("send_error", None, message, bypass_rate_limit))
        return True

    async def send_circuit_breaker(
        self, reason: str, details: str = "", *, bypass_rate_limit: bool = False
    ) -> bool:
        message = f"{reason}\n{details}" if details else reason
        self.calls.append(("send_circuit_breaker", None, message, bypass_rate_limit))
        return True

    async def close(self) -> None:
        self.closed = True


async def _drain(node: PaperTradingNode) -> None:
    """Wait for every fire-and-forget alert task to reach its final state."""
    while node._alert_tasks:
        await asyncio.gather(*list(node._alert_tasks), return_exceptions=True)


def _break_writer_once(monkeypatch: pytest.MonkeyPatch, node: PaperTradingNode) -> None:
    """Make the real event writer fail the ``during_outage`` append once, then work again.

    Earlier events may still sit in the background writer queue, so wait for it to settle
    first and fail only the event that names the outage, never a stray earlier write.
    """
    node._event_log.barrier().result(10)
    writer = node._event_writer
    assert writer is not None, "the transcript needs the real writer from _initialize"
    real_write = writer.write
    failed = {"done": False}

    def write(event: Any) -> None:
        if not failed["done"] and event.data.get("action") == "during_outage":
            failed["done"] = True
            raise OSError(28, "No space left on device")
        real_write(event)

    monkeypatch.setattr(writer, "write", write)


def _normalized(path: Path) -> list[dict[str, Any]]:
    """The JSONL lines with the timestamp removed (it is wall clock, not contract)."""
    lines = path.read_text().splitlines() if path.exists() else []
    entries = [json.loads(line) for line in lines if line.strip()]
    for entry in entries:
        del entry["ts"]
    return entries


async def _session(
    monkeypatch: pytest.MonkeyPatch, seeded: dict[str, Any]
) -> tuple[PaperTradingNode, _RecordingTelegram, Path]:
    """A node started through ``_initialize`` with a recording Telegram bot."""
    _RecordingTelegram.instances = []
    monkeypatch.setattr(node_module, "TelegramBot", _RecordingTelegram)

    async def working_telegram() -> TelegramConfig:
        return TelegramConfig(bot_token="123:ABC", chat_id="456")

    monkeypatch.setattr(node_module, "require_working_telegram", working_telegram)
    monkeypatch.setattr(node_module, "TRADE_ALERT_DELAY_S", 0.0)
    node = PaperTradingNode(_config(seeded, trade_alerts=True))
    await node._initialize()
    bot = node._telegram
    assert isinstance(bot, _RecordingTelegram)
    log = node.config.logs_path / f"{node.config.trader_id}.jsonl"
    return node, bot, log


async def test_full_session_transcript(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    fake_tradingnode: type[Any],  # noqa: F811
) -> None:
    """One ordered transcript: JSONL lines minus timestamps and the alert calls."""
    node, bot, log = await _session(monkeypatch, seeded)
    try:
        guard = node._guard
        assert guard is not None
        bridge = guard._listener  # the real guard's _GuardBridge, its notify seam
        trip = _Trip()
        # The guard itself is not exercised here (its table lives in
        # test_paper_halt_restore): its cache is unregistered under the fake kernel, so
        # the trade-text callbacks read a flat fake cache instead. ``status`` still comes
        # from the real guard, so the checkpoints stay meaningful.
        node._guard = SimpleNamespace(  # type: ignore[assignment]
            cache=_Cache(
                open_orders=(
                    _Order("STOP_MARKET", trigger_price="67498.70"),
                    _Order("LIMIT", price="67701.50"),
                ),
                positions={"P-1": trip.position},
                orders={str(trip.closed.closing_order_id): _Order("LIMIT")},
            ),
            status=guard.status,
        )

        bridge.on_position_opened(trip.opened)
        await _drain(node)
        bridge.on_position_closed(trip.closed)
        await _drain(node)
        bridge.on_state_change(GuardState.HALTED, HaltReason.MANUAL, "halt via API", {})
        await _drain(node)
        bridge.on_state_change(GuardState.RUNNING, None, "resumed by operator", {})
        await _drain(node)
        node._check_event_log()  # the first control tick only sets the baseline
        _break_writer_once(monkeypatch, node)
        with pytest.raises(OSError):
            future = node._write_event(EventType.SIGNAL, {"action": "during_outage"})
            assert future is not None
            future.result(10)
        node._check_event_log()  # outage tick: one CRITICAL alert
        node._write_event(EventType.SIGNAL, {"action": "after_outage"})
        node._event_log.barrier().result(10)
        node._check_event_log()  # recovery tick: one line + one non-critical alert
    finally:
        await node._shutdown()

    lines = _normalized(log)
    expected_lines: list[dict[str, Any]] = [
        {
            "event": "SIGNAL",
            "run_id": "PAPER-007",
            "strategy": "test_strategy",
            "data": {
                "action": "start",
                "trader_id": "PAPER-007",
                "strategy_id": 1,
                "validation_run_id": 1,
                "symbols": ["BTCUSDT"],
                "testnet": True,
                "leverage": 5,
                "margin_type": "CROSSED",
                "strategy_params": {"rsi_14_period": 21, "take_profit_risk_reward": 3.0},
                "config_diff": [
                    {
                        "field": "liquidation",
                        "validated": "off",
                        "paper": "exchange liquidation (CROSSED)",
                        "note": (
                            "validation did not simulate liquidation; "
                            "paper liquidation parity is unverified"
                        ),
                    }
                ],
                "max_margin_pct": 0.5,
                "start_fresh": False,
                "account": "binance:testnet:2c70e12b7a0646f9",
            },
        },
        {
            "event": "POSITION_OPEN",
            "run_id": "PAPER-007",
            "strategy": "test_strategy",
            "data": {
                "position_id": "P-1",
                "symbol": "BTCUSDT-PERP.BINANCE",
                "side": "LONG",
                "quantity": 0.002,
                "entry_price": 67600.09999999999,
                "leverage": 5,
            },
        },
        {
            "event": "RISK_CHECK",
            "run_id": "PAPER-007",
            "strategy": "test_strategy",
            "data": {
                "action": "liquidation_estimate",
                "position_id": "P-1",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "quantity": 0.002,
                "entry_price": 67600.09999999999,
                "leverage": 5,
                "margin_type": "CROSSED",
                "isolated_margin": None,
                "est_liquidation_price": None,
                "tier_table_version": None,
                "basis": None,
                "passed": True,
                "reason": (
                    "cross LP depends on the account wallet and other positions; not estimated"
                ),
            },
        },
        {
            "event": "POSITION_CLOSE",
            "run_id": "PAPER-007",
            "strategy": "test_strategy",
            "data": {
                "position_id": "P-1",
                "symbol": "BTCUSDT-PERP.BINANCE",
                "exit_price": 67498.7,
                "gross_pnl": -0.20279999999999998,
                "net_pnl": -0.25143557,
                "exit_reason": "take_profit",
            },
        },
        {
            "event": "RISK_CHECK",
            "run_id": "PAPER-007",
            "strategy": "test_strategy",
            "data": {
                "action": "halt",
                "check_type": "manual",
                "reason": "manual",
                "message": "halt via API",
                "passed": False,
            },
        },
        {
            "event": "SIGNAL",
            "run_id": "PAPER-007",
            "strategy": "test_strategy",
            "data": {
                "action": "state_change",
                "from_state": "halted",
                "to_state": "running",
                "message": "resumed by operator",
            },
        },
        {
            "event": "SIGNAL",
            "run_id": "PAPER-007",
            "strategy": "test_strategy",
            "data": {"action": "after_outage"},
        },
        {
            "event": "RISK_CHECK",
            "run_id": "PAPER-007",
            "strategy": "test_strategy",
            "data": {
                "action": "event_log_recovered",
                "failed": 1,
                "dropped": 0,
                "passed": True,
            },
        },
        {
            "event": "SIGNAL",
            "run_id": "PAPER-007",
            "strategy": "test_strategy",
            "data": {
                "action": "stop",
                "trader_id": "PAPER-007",
                "reason": "normal",
                "total_pnl": 0.0,
            },
        },
    ]
    assert lines == expected_lines, json.dumps(lines, indent=1)

    expected_alerts: list[_AlertCall] = [
        (
            "send_alert",
            "TRADE",
            "🟢 OPENED LONG 📈 BTCUSDT\n"
            "💰 Size: 0.002 BTC (≈ $135.20)\n"
            "⚡ Leverage: 5x\n"
            "🎯 Entry: 67,600.10\n"
            "🛑 Stop: 67,498.70 (−0.15%)\n"
            "✅ Target: 67,701.50 (+0.15%)\n"
            "🤖 PAPER-007 · 🧪 testnet",
            True,
        ),
        (
            "send_alert",
            "TRADE",
            "❌ CLOSED LONG 📈 BTCUSDT · LOSS\n"
            "📉 P&L: −0.25 USDT (−0.19%)\n"
            "🚪 Exit: 67,498.70 · 🎯 take profit hit\n"
            "🎯 Entry: 67,600.10\n"
            "⏱️ Held: 0s\n"
            "💸 Fees: 0.05 USDT\n"
            "🤖 PAPER-007 · 🧪 testnet",
            True,
        ),
        ("send_circuit_breaker", None, "HALT (manual): halt via API", True),
        ("send_circuit_breaker", None, "RESUMED: resumed by operator", True),
        (
            "send_error",
            None,
            "paper PAPER-007 event log not writing: 1 failed, 0 dropped (OSError); "
            "trading continues",
            True,
        ),
        (
            "send_error",
            None,
            "paper PAPER-007 event log writing again (0 dropped, 1 failed)",
            False,
        ),
    ]
    assert bot.calls == expected_alerts, json.dumps(bot.calls, indent=1)
    assert bot.closed is True
