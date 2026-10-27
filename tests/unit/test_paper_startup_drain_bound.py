"""A hung start-up command drain alerts within a bound (bead vibe-quant-wi1m0, part b).

The pre-NT drain (``process_commands(startup_halts_only=True)``) awaited its claim without
a bound: a claim that never answers (busy DB lock, NFS) kept the node from starting and
nothing told the operator. It now waits ``COMMAND_CLAIM_STALL_S`` like every control claim;
a stall counts as the start-up failure (the existing start-up block rule: entries blocked at
the first failure) and raises ONE critical alert. Pinned on a real ``TelegramBot`` over
``httpx.MockTransport``, a real ``SessionJournal`` and the real command queue.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import TYPE_CHECKING, Any

import httpx

import vibe_quant.paper.node as node_module
from tests.unit.test_audit_paper_node import _config, _FakeTradingNode
from tests.unit.test_paper_halt_restore import (
    fake_tradingnode,  # noqa: F401  # fixture
    seeded,  # noqa: F401  # fixture
)
from tests.unit.test_paper_startup_halt import _Guard, _remove_handlers
from tests.unit.test_paper_startup_queued_halt import _enqueue, _fresh_kill_snapshot, _row
from vibe_quant.alerts.telegram import TelegramBot, TelegramConfig
from vibe_quant.paper.command_channel import (
    STALL_LABEL,
    STARTUP_CHECKPOINT_STEP,
    STARTUP_SWEEP_STEP,
    failed_alert_text,
    restored_alert_text,
    startup_checkpoint_failed_alert_text,
    startup_stall_alert_text,
)
from vibe_quant.paper.guard import GuardState, HaltReason
from vibe_quant.paper.node import PaperTradingNode
from vibe_quant.paper.persistence import PaperCommand, PaperCommandQueue, StatePersistence

if TYPE_CHECKING:
    import pytest

_TRADER = "PAPER-007"
_REAL_DRAIN = PaperCommandQueue.claim_startup_halts
_REAL_SWEEP = PaperCommandQueue.fail_stale_running
_REAL_CHECKPOINT = StatePersistence.save_checkpoint

_HUNG_ALERT = failed_alert_text(_TRADER, 1, STALL_LABEL, "new entries blocked")


class _Chat:
    """A real TelegramBot whose HTTP goes to an in-memory recorder."""

    def __init__(self) -> None:
        self.bodies: list[str] = []

    def bot(self) -> TelegramBot:
        def record(request: httpx.Request) -> httpx.Response:
            # Each text is "<header · time>\n<alert text>".
            self.bodies.append(json.loads(request.content)["text"].split("\n", 1)[1])
            return httpx.Response(200, json={"ok": True})

        bot = TelegramBot(TelegramConfig(bot_token="T0K", chat_id="42"))

        async def client() -> httpx.AsyncClient:
            return httpx.AsyncClient(transport=httpx.MockTransport(record))

        bot._get_client = client  # type: ignore[method-assign]
        return bot


def _slow_drain(monkeypatch: pytest.MonkeyPatch, hold: threading.Event, seconds: float) -> None:
    """The start-up drain answers (really) after ``seconds``, or when ``hold`` is set."""

    def drain(queue: PaperCommandQueue, trader_id: str) -> list[PaperCommand]:
        hold.wait(seconds)
        return _REAL_DRAIN(queue, trader_id)

    monkeypatch.setattr(PaperCommandQueue, "claim_startup_halts", drain)


def _node(
    monkeypatch: pytest.MonkeyPatch, seeded_run: dict[str, Any], chat: _Chat
) -> PaperTradingNode:
    monkeypatch.setattr(node_module, "TradingGuard", _Guard)
    node = PaperTradingNode(_config(seeded_run))
    node._journal.attach(None, chat.bot(), "s")
    monkeypatch.setattr(node._journal, "_write_event", lambda *_a, **_k: None)
    _fresh_kill_snapshot(monkeypatch, node)
    node._setup_signal_handlers()
    return node


async def test_hung_startup_drain_starts_blocked_and_alerts_once(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A drain that never answers: the start continues within the bound, entries are
    blocked, ONE critical alert reaches Telegram; the queued kill still halts when the
    claim finally returns, the channel lifts nothing past it and says so."""
    monkeypatch.setattr(node_module, "COMMAND_CLAIM_STALL_S", 0.05)
    hold = threading.Event()
    _slow_drain(monkeypatch, hold, 30)
    (kill_id,) = _enqueue(seeded["db"], ("kill", {"message": "kill switch: test"}))
    chat = _Chat()
    node = _node(monkeypatch, seeded, chat)
    try:
        began = time.monotonic()
        await asyncio.wait_for(node._initialize(), 10)
        assert time.monotonic() - began < 5, "the start is bounded by the stall, not the claim"
        await node._journal.wait_alerts(None)

        assert node._guard is not None
        assert node._guard.guard_state == GuardState.PAUSED
        assert node._cmd_channel.owns_pause is True
        assert chat.bodies == [_HUNG_ALERT]
        assert _row(seeded["db"], kill_id).status in ("pending", "running")

        # The control loop keeps polling the SAME claim; one alert per streak, no more.
        await node.process_commands()
        await node._journal.wait_alerts(None)
        assert chat.bodies == [_HUNG_ALERT]

        hold.set()
        deadline = time.monotonic() + 15
        while _row(seeded["db"], kill_id).status in ("pending", "running"):
            assert time.monotonic() < deadline, "the returned claim's kill must run"
            await node.process_commands()
            await asyncio.sleep(0.01)
        await node._journal.wait_alerts(None)

        assert _row(seeded["db"], kill_id).status == "done"
        guard_after: Any = node._guard
        assert guard_after.guard_state == GuardState.HALTED
        assert node._guard.halt_reason == HaltReason.KILL_SWITCH
        restored = [b for b in chat.bodies if "channel restored" in b]
        assert restored == [
            restored_alert_text(_TRADER, 2, "entries stay blocked (halted/kill_switch)")
        ]
    finally:
        hold.set()
        _remove_handlers()
        await node._shutdown()


async def test_startup_drain_inside_the_bound_stays_silent(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow-but-answering drain (inside the stall bound) neither blocks nor alerts."""
    monkeypatch.setattr(node_module, "COMMAND_CLAIM_STALL_S", 5.0)
    _slow_drain(monkeypatch, threading.Event(), 0.1)
    (kill_id,) = _enqueue(seeded["db"], ("kill", {"message": "kill switch: test"}))
    chat = _Chat()
    node = _node(monkeypatch, seeded, chat)
    try:
        await asyncio.wait_for(node._initialize(), 10)
        await node._journal.wait_alerts(None)
        assert node._guard is not None
        assert node._cmd_channel.failures == 0
        assert node._cmd_channel.owns_pause is False
        assert _row(seeded["db"], kill_id).status == "done"
        assert node._guard.halt_reason == HaltReason.KILL_SWITCH
        assert not [b for b in chat.bodies if "command channel" in b]
    finally:
        _remove_handlers()
        await node._shutdown()


async def test_hung_stale_command_sweep_starts_bounded_and_alerts_once(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The start-up sweep of rows a dead process left 'running' shares the command
    worker; one that never answers used to hold the start for good with no word."""
    monkeypatch.setattr(node_module, "COMMAND_CLAIM_STALL_S", 0.05)
    hold = threading.Event()

    def sweep(queue: PaperCommandQueue, trader_id: str, message: str) -> int:
        hold.wait(30)
        return _REAL_SWEEP(queue, trader_id, message)

    monkeypatch.setattr(PaperCommandQueue, "fail_stale_running", sweep)
    chat = _Chat()
    node = _node(monkeypatch, seeded, chat)
    try:
        began = time.monotonic()
        await asyncio.wait_for(node._initialize(), 10)
        assert time.monotonic() - began < 5, "the start is bounded by the stall"
        await node._journal.wait_alerts(None)
        stall = startup_stall_alert_text(_TRADER, STARTUP_SWEEP_STEP)
        assert chat.bodies.count(stall) == 1, "ONE critical alert for the hung sweep"
        # The drain queues behind the hung sweep on the same worker: entries stay blocked.
        assert node._guard is not None
        assert node._guard.guard_state == GuardState.PAUSED
    finally:
        hold.set()
        _remove_handlers()
        await node._shutdown()


def _slow_checkpoint(
    monkeypatch: pytest.MonkeyPatch, hold: threading.Event, fail: bool = False
) -> None:
    """The FIRST checkpoint write answers when ``hold`` is set (raising if ``fail``)."""
    calls: list[int] = []

    def checkpoint(persistence: StatePersistence, checkpoint: Any) -> int:
        calls.append(1)
        if len(calls) == 1:
            hold.wait(30)
            if fail:
                raise OSError("disk I/O error at /secret/path token=T0K")
        return _REAL_CHECKPOINT(persistence, checkpoint)

    monkeypatch.setattr(StatePersistence, "save_checkpoint", checkpoint)


async def _until(predicate: Any, what: str) -> None:
    deadline = time.monotonic() + 10
    while not predicate():
        assert time.monotonic() < deadline, what
        await asyncio.sleep(0.01)


_CHECKPOINT_STALL = startup_stall_alert_text(
    _TRADER, STARTUP_CHECKPOINT_STEP, "new entries blocked until it is saved"
)


async def test_hung_initial_checkpoint_blocks_entries_until_it_is_saved(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first checkpoint write waited on the state store without a bound; bounded, it
    let the node trade with no crash persistence. Entries now stay paused (ONE critical
    alert) until it is saved; a good command claim lifts nothing, the late save does."""
    monkeypatch.setattr(node_module, "COMMAND_CLAIM_STALL_S", 0.05)
    hold = threading.Event()
    _slow_checkpoint(monkeypatch, hold)
    chat = _Chat()
    node = _node(monkeypatch, seeded, chat)
    try:
        began = time.monotonic()
        await asyncio.wait_for(node._initialize(), 10)
        assert time.monotonic() - began < 5, "the start is bounded by the stall"
        await node._journal.wait_alerts(None)
        assert chat.bodies == [_CHECKPOINT_STALL], "ONE critical alert for the hung write"
        guard: Any = node._guard
        assert guard.guard_state == GuardState.PAUSED, "no trading before the first checkpoint"

        await node.process_commands()  # claims answer: the channel's recovery lifts nothing
        assert guard.guard_state == GuardState.PAUSED

        hold.set()
        await _until(lambda: guard.guard_state == GuardState.RUNNING, "the save lifts the block")
        await node._journal.wait_alerts(None)
        assert [b for b in chat.bodies if "start-up" in b] == [_CHECKPOINT_STALL]
        assert node._cmd_channel.owns_pause is False
    finally:
        hold.set()
        _remove_handlers()
        await node._shutdown()


async def test_late_initial_checkpoint_keeps_a_channel_block(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checkpoint and drain both hang: the late save lifts only its own block; entries
    resume once the command channel answers too."""
    monkeypatch.setattr(node_module, "COMMAND_CLAIM_STALL_S", 0.05)
    hold_checkpoint, hold_drain = threading.Event(), threading.Event()
    _slow_checkpoint(monkeypatch, hold_checkpoint)
    _slow_drain(monkeypatch, hold_drain, 30)
    chat = _Chat()
    node = _node(monkeypatch, seeded, chat)
    try:
        await asyncio.wait_for(node._initialize(), 10)
        guard: Any = node._guard
        assert guard.guard_state == GuardState.PAUSED
        hold_checkpoint.set()
        await _until(lambda: not node._cmd_channel.checkpoint_blocking, "the save lands")
        await asyncio.sleep(0.05)
        assert guard.guard_state == GuardState.PAUSED, "the channel is still failing"

        hold_drain.set()
        await _until(lambda: node._claim_inflight is None or node._claim_inflight.done(), "claim")
        await node.process_commands()
        assert guard.guard_state == GuardState.RUNNING
    finally:
        hold_checkpoint.set()
        hold_drain.set()
        _remove_handlers()
        await node._shutdown()


async def test_failed_late_initial_checkpoint_keeps_entries_blocked_and_alerts(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hung first write that then raises: still unsaved, so entries stay blocked (fail
    closed) and ONE more critical alert names the error class, never its text."""
    monkeypatch.setattr(node_module, "COMMAND_CLAIM_STALL_S", 0.05)
    hold = threading.Event()
    _slow_checkpoint(monkeypatch, hold, fail=True)
    chat = _Chat()
    node = _node(monkeypatch, seeded, chat)
    try:
        await asyncio.wait_for(node._initialize(), 10)
        hold.set()
        failed = startup_checkpoint_failed_alert_text(
            _TRADER, "OSError", "new entries stay blocked until an operator resume"
        )
        await _until(lambda: failed in chat.bodies, "the failure is reported")
        await node.process_commands()
        await node._journal.wait_alerts(None)
        guard: Any = node._guard
        assert guard.guard_state == GuardState.PAUSED
        assert chat.bodies == [_CHECKPOINT_STALL, failed]
        assert not [b for b in chat.bodies if "secret" in b or "T0K" in b]
    finally:
        hold.set()
        _remove_handlers()
        await node._shutdown()


async def test_startup_steps_inside_the_bound_stay_silent(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(node_module, "COMMAND_CLAIM_STALL_S", 5.0)
    chat = _Chat()
    node = _node(monkeypatch, seeded, chat)
    try:
        await asyncio.wait_for(node._initialize(), 10)
        await node._journal.wait_alerts(None)
        assert not [b for b in chat.bodies if "start-up" in b]
    finally:
        _remove_handlers()
        await node._shutdown()
