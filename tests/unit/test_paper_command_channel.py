"""The paper command channel fails closed (bead vibe-quant-3fg9d).

Contract rows R1-R21 of design arch-cmdchan section 6. ``F`` = a command claim raises
``sqlite3.OperationalError``, ``S`` = it returns. After 5 consecutive failures the node
blocks new entries through the guard's pause and raises ONE critical alert per streak;
the pause is lifted only by the channel that set it (T2), and a stalled claim counts as a
failure (T3). A claim failure never carries exception text into a log, event or alert.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import signal
import sqlite3
import threading
import time
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import httpx
import pytest

import tests.unit.test_audit_paper_guard as harness
import vibe_quant.paper.node as node_module
from tests.unit.test_audit_paper_node import _config, _FakeTradingNode
from tests.unit.test_paper_halt_restore import (
    _bars,
    _latest_checkpoint,
    _trace_node,
    _write_checkpoint,
    fake_tradingnode,  # noqa: F401  # fixture
    seeded,  # noqa: F401  # fixture
)
from tests.unit.test_paper_startup_halt import (
    _Guard,
    _remove_handlers,
    _startup_harness,
)
from tests.unit.test_paper_startup_queued_halt import (
    _enqueue,
    _fresh_kill_snapshot,
    _row,
)
from vibe_quant.alerts.telegram import TelegramBot, TelegramConfig
from vibe_quant.data.catalog import get_bar_type
from vibe_quant.logging.events import EventType
from vibe_quant.paper.command_channel import (
    COMMAND_CLAIM_FAILURE_LIMIT,
    CommandChannelHealth,
    block_message,
    failed_alert_text,
    guard_state_text,
    restored_alert_text,
)
from vibe_quant.paper.guard import GuardState, HaltReason, RiskState
from vibe_quant.paper.node import NodeState, PaperTradingNode, _GuardBridge
from vibe_quant.paper.persistence import PaperCommand, PaperCommandQueue

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from tests.unit.test_paper_halt_restore import _Trace

_TRADER = "PAPER-007"
_BLOCK_5 = (
    "command channel failing (5 consecutive claim failures, last OperationalError); "
    "new entries blocked"
)
_FAILED_5 = (
    f"paper {_TRADER}: command channel failing (5 consecutive claim failures, last "
    "OperationalError); new entries blocked; queued halt/kill/close_all commands are not "
    "being read - use the system kill switch or SIGUSR1"
)


# ------------------------------------------------------------------ helpers

_REAL_CLAIM = PaperCommandQueue.claim_pending


class _Claims:
    """Patches ``claim_pending`` with a script, one letter per claim (tail repeats ``tail``).

    ``F`` raises; ``S`` runs the real claim; ``H`` hangs on ``hold`` then runs the real
    claim; ``R`` hangs on ``hold`` then raises; ``C`` runs the real claim (rows go
    ``running``), then hangs on ``hold`` before returning them. ``guard_before[i]`` is the
    guard state as claim ``i`` started, i.e. after ``i`` completed polls.
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        node: PaperTradingNode,
        script: str,
        *,
        tail: str = "S",
    ) -> None:
        self.calls = 0
        self.guard_before: list[GuardState | None] = []
        self.hold = threading.Event()

        def claim(queue: PaperCommandQueue, trader_id: str) -> list[PaperCommand]:
            index = self.calls
            self.calls += 1
            guard = node._guard
            self.guard_before.append(guard.guard_state if guard is not None else None)
            kind = script[index] if index < len(script) else tail
            if kind == "F":
                raise sqlite3.OperationalError("ZZLEAK1") from RuntimeError("ZZLEAK2")
            if kind in ("H", "R"):
                self.hold.wait(30)
                if kind == "R":
                    raise sqlite3.OperationalError("database is locked")
            rows = _REAL_CLAIM(queue, trader_id)
            if kind == "C":
                self.hold.wait(30)
            return rows

        monkeypatch.setattr(PaperCommandQueue, "claim_pending", claim)


async def _until(predicate: Callable[[], bool], timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        await asyncio.sleep(0.01)


async def _polls(node: PaperTradingNode, count: int) -> int:
    """``count`` control polls through the public entry; returns how many raised."""
    raised = 0
    for _ in range(count):
        try:
            await node.process_commands()
        except sqlite3.OperationalError:
            raised += 1
    return raised


@contextlib.asynccontextmanager
async def _initialized(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],  # noqa: F811
    **config_kw: Any,
) -> AsyncIterator[tuple[PaperTradingNode, _Trace]]:
    """A traced node past ``_initialize`` (real guard, claims working)."""
    monkeypatch.setattr(node_module, "TradingGuard", _Guard)
    node, trace = _trace_node(monkeypatch, seeded, **config_kw)
    _fresh_kill_snapshot(monkeypatch, node)
    node._setup_signal_handlers()
    try:
        await node._initialize()
        yield node, trace
    finally:
        _remove_handlers()
        await node._shutdown()


@contextlib.asynccontextmanager
async def _started(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],  # noqa: F811
    script: str,
    *,
    tail: str = "S",
    interval: float = 0.01,
) -> AsyncIterator[tuple[PaperTradingNode, _Trace, _Claims]]:
    """``node.start()`` as a task (real control loop) on a fresh kill-switch read."""
    monkeypatch.setattr(node_module, "TradingGuard", _Guard)
    node, trace = _trace_node(monkeypatch, seeded, control_poll_interval=interval)
    _fresh_kill_snapshot(monkeypatch, node)
    release = asyncio.Event()

    async def run_async(self: Any) -> None:
        await release.wait()

    monkeypatch.setattr(_FakeTradingNode, "run_async", run_async)
    claims = _Claims(monkeypatch, node, script, tail=tail)
    task = asyncio.create_task(node.start())
    try:
        yield node, trace, claims
    finally:
        release.set()
        _remove_handlers()
        await asyncio.wait_for(task, 15)


def _channel_calls(trace: _Trace, action: str | None = None) -> list[tuple[Any, ...]]:
    """Channel events (``action`` given) or the critical/non-critical channel alerts."""
    if action is not None:
        return [c for c in trace.calls if c[0] == "write" and c[2].get("action") == action]
    return [c for c in trace.calls if c[0] == "alert" and "command channel" in c[2]]


# ------------------------------------------------------- pure table (R3b, R21)


def test_health_table() -> None:
    """Edges of the streak counter: 4 failures do not block, the 5th blocks once, a
    success resets the count and owes the restored alert only for a blocked streak."""
    health = CommandChannelHealth()
    assert [health.record_failure("E") for _ in range(4)] == [False] * 4
    assert (health.failures, health.blocking) == (4, False)

    health.record_success()
    assert (health.failures, health.blocking, health.unreported_recovery) == (0, False, None)

    assert health.record_failure("E") is False
    assert (health.failures, health.blocking) == (1, False), "F x4, S, F restarts at 1"

    health = CommandChannelHealth()
    results = [health.record_failure("OperationalError") for _ in range(7)]
    assert results == [False] * (COMMAND_CLAIM_FAILURE_LIMIT - 1) + [True, False, False]
    assert (health.failures, health.blocking, health.last_error) == (7, True, "OperationalError")

    health.record_success()
    assert (health.failures, health.blocking, health.last_error) == (0, False, "")
    assert health.unreported_recovery == 7


def test_health_limit_one_blocks_on_the_first_failure() -> None:
    health = CommandChannelHealth()
    assert health.record_failure("E", limit=1) is True
    assert health.blocking is True
    assert health.record_failure("E", limit=1) is False


def test_health_texts_are_exact() -> None:
    assert guard_state_text(GuardState.PAUSED, None) == "paused"
    assert guard_state_text(GuardState.RUNNING, None) == "running"
    assert guard_state_text(GuardState.HALTED, HaltReason.KILL_SWITCH) == "halted/kill_switch"
    assert guard_state_text(GuardState.HALTED, HaltReason.MAX_DRAWDOWN) == "halted/max_drawdown"
    assert block_message(5, "OperationalError") == _BLOCK_5
    assert block_message(1, "TimeoutError") == (
        "command channel failing (1 consecutive claim failures, last TimeoutError); "
        "new entries blocked"
    )
    assert failed_alert_text(_TRADER, 5, "OperationalError", "new entries blocked") == _FAILED_5
    assert failed_alert_text("P-9", 7, "no answer within 1 s", "entries already blocked (paused)") == (
        "paper P-9: command channel failing (7 consecutive claim failures, last no answer "
        "within 1 s); entries already blocked (paused); queued halt/kill/close_all commands "
        "are not being read - use the system kill switch or SIGUSR1"
    )
    assert restored_alert_text(_TRADER, 5, "entries resumed") == (
        f"paper {_TRADER}: command channel restored after 5 failed claims; entries resumed"
    )
    assert restored_alert_text("P-9", 7, "entries stay blocked (halted/kill_switch)") == (
        "paper P-9: command channel restored after 7 failed claims; "
        "entries stay blocked (halted/kill_switch)"
    )


# ------------------------------------------------------------------ R1


async def test_persistent_claim_failure_blocks_entries_at_fifth_poll(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R1: through ``node.start()`` with a claim that always raises: the guard runs for
    polls 1-5's start, is paused from poll 5 on, and exactly ONE event, ONE critical alert
    and ONE pause transition exist however long the streak lasts."""
    with caplog.at_level(logging.ERROR):
        async with _started(monkeypatch, seeded, "F" * 1000) as (node, trace, claims):
            await _until(lambda: claims.calls >= 20)
            states = list(claims.guard_before)
            assert states[:5] == [GuardState.RUNNING] * 5, "polls 1-5 start unblocked"
            assert set(states[5:]) == {GuardState.PAUSED}, "paused from the 5th failure on"
            assert node.status.state == NodeState.PAUSED
            assert node.status.error_message == _BLOCK_5

    assert _channel_calls(trace, "command_channel_failed") == [
        (
            "write",
            EventType.RISK_CHECK,
            {
                "action": "command_channel_failed",
                "failures": 5,
                "error_type": "OperationalError",
                "entries": "blocked",
                "state": "paused",
                "passed": False,
            },
            None,
        )
    ]
    assert _channel_calls(trace) == [("alert", "error", _FAILED_5, True)]
    pauses = [
        c
        for c in trace.calls
        if c[0] == "write"
        and c[2].get("action") == "state_change"
        and c[2].get("to_state") == "paused"
    ]
    assert len(pauses) == 1
    failed_logs = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage() == "paper control loop iteration failed (OperationalError)"
    ]
    assert len(failed_logs) >= claims.calls - 1, "every failed poll is still logged"


# ------------------------------------------------------------------ R2


async def test_four_failures_then_success_never_blocks(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R2: F x4, S, F x4, S never pauses and writes no channel event or alert."""
    async with _initialized(monkeypatch, seeded) as (node, trace):
        claims = _Claims(monkeypatch, node, "FFFFSFFFFS")
        assert await _polls(node, 10) == 8
        assert claims.guard_before == [GuardState.RUNNING] * 10
        assert node._guard is not None
        assert node._guard.guard_state == GuardState.RUNNING
        assert node._cmd_channel.failures == 0
        assert node._cmd_channel.owns_pause is False
        assert _channel_calls(trace, "command_channel_failed") == []
        assert _channel_calls(trace) == []


# ------------------------------------------------------------------ R12


async def test_failed_alert_once_per_streak_and_flapping(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R12: one failed alert per streak whatever its length; (F x5, S) x3 -> 3 failed
    alerts (each with its own streak), (F x4, S) x4 -> none."""
    async with _initialized(monkeypatch, seeded) as (node, trace):
        _Claims(monkeypatch, node, "F" * 12)
        await _polls(node, 12)
        assert len(_channel_calls(trace, "command_channel_failed")) == 1
        assert [c[2] for c in _channel_calls(trace)] == [_FAILED_5]

    async with _initialized(monkeypatch, seeded) as (node, trace):
        _Claims(monkeypatch, node, ("FFFFFS" * 3))
        await _polls(node, 18)
        assert len(_channel_calls(trace, "command_channel_failed")) == 3

    async with _initialized(monkeypatch, seeded) as (node, trace):
        _Claims(monkeypatch, node, ("FFFFS" * 4))
        await _polls(node, 20)
        assert _channel_calls(trace, "command_channel_failed") == []
        assert _channel_calls(trace) == []


# ------------------------------------------------- R6 / R7 / R7b / R7c (never weakened)


def _halt_kill(node: PaperTradingNode) -> None:
    assert node._guard is not None
    node._guard.halt(HaltReason.KILL_SWITCH, "kill switch: test")


def _halt_drawdown(node: PaperTradingNode) -> None:
    assert node._guard is not None
    node._guard.halt(HaltReason.MAX_DRAWDOWN, "drawdown: test")


def _operator_pause(node: PaperTradingNode) -> None:
    node.pause()


@pytest.mark.parametrize(
    ("prepare", "expected_state"),
    [
        (_halt_kill, "halted/kill_switch"),
        (_halt_drawdown, "halted/max_drawdown"),
        (_operator_pause, "paused"),
    ],
    ids=["kill_switch", "max_drawdown", "operator_pause"],
)
async def test_existing_halt_is_never_weakened(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    prepare: Callable[[PaperTradingNode], None],
    expected_state: str,
) -> None:
    """R6/R7/R7b: a halt (or an operator pause) that is already active stays exactly as it
    was: the channel does not own it, the alert says entries were already blocked."""
    async with _initialized(monkeypatch, seeded) as (node, trace):
        assert node._guard is not None
        _Claims(monkeypatch, node, "F" * 5)
        prepare(node)
        before = (node._guard.guard_state, node._guard.halt_reason, node._guard.message)

        await _polls(node, 5)

        assert (node._guard.guard_state, node._guard.halt_reason, node._guard.message) == before
        assert node._cmd_channel.owns_pause is False
        events = _channel_calls(trace, "command_channel_failed")
        assert [e[2]["entries"] for e in events] == ["already_blocked"]
        assert [e[2]["state"] for e in events] == [expected_state]
        assert [c[2] for c in _channel_calls(trace)] == [
            f"paper {_TRADER}: command channel failing (5 consecutive claim failures, last "
            f"OperationalError); entries already blocked ({expected_state}); queued "
            "halt/kill/close_all commands are not being read - use the system kill switch or "
            "SIGUSR1"
        ]


async def test_restored_error_halt_is_never_weakened_by_a_failing_startup_drain(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R7c: a restored ERROR halt plus a failing start-up drain stays halted/error."""
    _write_checkpoint(
        seeded["db"],
        {"state": "stopped", "halt_reason": "error", "error_message": "boom earlier"},
    )

    def boom(self: PaperCommandQueue, trader_id: str) -> list[PaperCommand]:
        raise sqlite3.OperationalError("ZZLEAK1")

    monkeypatch.setattr(PaperCommandQueue, "claim_startup_halts", boom)
    async with _initialized(monkeypatch, seeded) as (node, trace):
        assert node._guard is not None
        assert node._guard.guard_state == GuardState.HALTED
        assert node._guard.halt_reason == HaltReason.ERROR
        assert node._cmd_channel.owns_pause is False
        events = _channel_calls(trace, "command_channel_failed")
        assert [(e[2]["entries"], e[2]["state"], e[2]["failures"]) for e in events] == [
            ("already_blocked", "halted/error", 1)
        ]


# ------------------------------------------------------------------ R8


class _EngagedSnapshot:
    error_class = None
    engaged = True
    reason = "operator test"
    last_ok_mono = 0.0
    consecutive_failures = 0

    @staticmethod
    def age(_now: float) -> float:
        return 0.0


async def test_system_kill_switch_halts_while_blocked(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R8: the system kill switch supersedes the channel pause: halted/kill_switch."""
    async with _initialized(monkeypatch, seeded) as (node, _trace):
        assert node._guard is not None
        _Claims(monkeypatch, node, "F" * 5)
        await _polls(node, 5)
        assert node._guard.guard_state == GuardState.PAUSED

        monkeypatch.setattr(node._kill_monitor, "snapshot", _EngagedSnapshot)
        node._check_kill_switch()

        assert node._guard.guard_state == GuardState.HALTED
        assert node._guard.halt_reason == HaltReason.KILL_SWITCH
        assert node.status.state == NodeState.HALTED


# ------------------------------------------------------------------ R16


async def test_health_without_guard(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R16: with no guard the streak is still counted and alerted; nothing is paused and
    nothing but the claim error is raised."""
    node, trace = _trace_node(monkeypatch, seeded)
    node._commands = PaperCommandQueue(seeded["db"])
    _Claims(monkeypatch, node, "F" * 5)
    try:
        assert node._guard is None
        assert await _polls(node, 5) == 5
        assert node._cmd_channel.blocking is True
        assert node._cmd_channel.owns_pause is False
        events = _channel_calls(trace, "command_channel_failed")
        assert [(e[2]["failures"], e[2]["entries"]) for e in events] == [(5, "not_blocked")]
        assert [c[2] for c in _channel_calls(trace)] == [
            _FAILED_5.replace("new entries blocked", "new entries NOT blocked (no trading guard)")
        ]
        assert node.status.state != NodeState.PAUSED
    finally:
        await node._shutdown()


# ------------------------------------------------------------------ R17


async def test_channel_texts_are_class_only(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R17: ``OperationalError("ZZLEAK1") from RuntimeError("ZZLEAK2")`` leaks into no log,
    event payload, alert text or status text; no record carries ``exc_info``."""
    async with _initialized(monkeypatch, seeded) as (node, trace):
        _Claims(monkeypatch, node, "F" * 6)
        with caplog.at_level(logging.DEBUG):
            await _polls(node, 6)

        assert node._guard is not None and node._guard.guard_state == GuardState.PAUSED
        everything = repr(trace.calls) + (node.status.error_message or "") + caplog.text
        assert "ZZLEAK1" not in everything
        assert "ZZLEAK2" not in everything
        assert all(r.exc_info is None for r in caplog.records)


# ------------------------------------------------------------------ R18


async def test_alert_write_failure_keeps_claim_error(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R18: when the event write of the failed alert raises, the alert is still sent, the
    claim error still reaches the caller, and one class-only log names the write error."""
    async with _initialized(monkeypatch, seeded) as (node, trace):
        real_write = node._journal._write_event

        def write(event_type: EventType, data: dict[str, object], timestamp: Any = None) -> Any:
            if data.get("action") == "command_channel_failed":
                raise PermissionError("ZZWRITE1")
            return real_write(event_type, data, timestamp)

        monkeypatch.setattr(node._journal, "_write_event", write)
        _Claims(monkeypatch, node, "F" * 5)
        with caplog.at_level(logging.ERROR):
            assert await _polls(node, 5) == 5, "the claim error was raised on every poll"

        assert [c[2] for c in _channel_calls(trace)] == [_FAILED_5]
        mine = [
            r.getMessage()
            for r in caplog.records
            if r.getMessage().startswith("command channel alert failed")
        ]
        assert mine == ["command channel alert failed (PermissionError)"]
        assert "ZZWRITE1" not in caplog.text
        assert all(r.exc_info is None for r in caplog.records)


# ------------------------------------------- listener raises during the pause


async def test_pause_listener_failure_still_owns_and_counts(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The guard is PAUSED before its listener runs: a raising state-change write must
    still leave the channel owning that pause, and never replace the claim error."""
    async with _initialized(monkeypatch, seeded) as (node, trace):
        assert node._guard is not None
        real_write = node._journal._write_event

        def write(event_type: EventType, data: dict[str, object], timestamp: Any = None) -> Any:
            if data.get("action") == "state_change" and data.get("to_state") == "paused":
                raise PermissionError("ZZWRITE2")
            return real_write(event_type, data, timestamp)

        monkeypatch.setattr(node._journal, "_write_event", write)
        _Claims(monkeypatch, node, "F" * 5)
        with caplog.at_level(logging.ERROR):
            assert await _polls(node, 5) == 5

        assert node._guard.guard_state == GuardState.PAUSED
        assert node._cmd_channel.owns_pause is True
        assert len(_channel_calls(trace, "command_channel_failed")) == 1
        assert "command channel pause failed (PermissionError)" in caplog.text
        assert "ZZWRITE2" not in caplog.text


async def test_failed_pause_is_not_owned(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A pause that never happened is not owned: recovery must not ``resume`` it later."""
    async with _initialized(monkeypatch, seeded) as (node, trace):
        assert node._guard is not None

        def refuse(_message: str = "") -> bool:
            raise RuntimeError("ZZPAUSE")

        monkeypatch.setattr(node._guard, "pause", refuse)
        _Claims(monkeypatch, node, "F" * 5)
        with caplog.at_level(logging.ERROR):
            assert await _polls(node, 5) == 5

        assert node._guard.guard_state == GuardState.RUNNING
        assert node._cmd_channel.owns_pause is False
        assert "command channel pause failed (RuntimeError)" in caplog.text
        assert "ZZPAUSE" not in caplog.text
        assert [e[2]["entries"] for e in _channel_calls(trace, "command_channel_failed")] == [
            "not_blocked"
        ]
        assert [c[2] for c in _channel_calls(trace)] == [
            _FAILED_5.replace("new entries blocked", "new entries NOT blocked (pause failed)")
        ], "the alert never claims a block that did not happen"


# ------------------------------------------------- R9 / R13b (real NT, start-up)


async def test_startup_drain_failure_starts_blocked_real_nt(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R9/R13b: a failing start-up drain pauses the guard BEFORE NT starts: on a real NT
    engine the strategies' on_start and every-bar entries are all denied by the pause
    gate, none fills; the start continues, one critical alert says so, owns_pause holds."""
    h = _startup_harness(monkeypatch)
    node, trace = _trace_node(monkeypatch, seeded)
    monkeypatch.setattr(node, "_build_guard", lambda *_args, **_kwargs: h.guard)

    def boom(self: PaperCommandQueue, trader_id: str) -> list[PaperCommand]:
        raise sqlite3.OperationalError("ZZLEAK1") from RuntimeError("ZZLEAK2")

    monkeypatch.setattr(PaperCommandQueue, "claim_startup_halts", boom)
    node._setup_signal_handlers()
    try:
        with caplog.at_level(logging.ERROR):
            await node._initialize()
        assert node._commands is not None, "the start continued past the failed drain"
        assert h.guard.guard_state == GuardState.PAUSED
        assert node._cmd_channel.owns_pause is True
        assert [r.getMessage() for r in caplog.records if "drain failed" in r.getMessage()] == [
            "start-up command drain failed (OperationalError)"
        ]
        assert [c[2] for c in _channel_calls(trace)] == [
            f"paper {_TRADER}: command channel failing (1 consecutive claim failures, last "
            "OperationalError); new entries blocked; queued halt/kill/close_all commands are "
            "not being read - use the system kill switch or SIGUSR1"
        ]
        assert _channel_calls(trace, "command_channel_failed")[0][2]["failures"] == 1

        h.engine.add_data(_bars(h, [100.0] * 5))
        h.engine.run()

        submitted = len(h.entry_orders())
        assert submitted >= 1, "the gate must really be exercised"
        assert len(h.filled_entries()) == 0
        assert [by_gate for _e, by_gate in h.listener.denied] == [True] * submitted
        assert all(
            str(e.reason).startswith("VQ_GUARD: trading paused (paused)")
            for e, _by_gate in h.listener.denied
        )
    finally:
        _remove_handlers()
        await node._shutdown()
        h.engine.dispose()


# =============================================================== T2: recovery

_RESTORED_5 = f"paper {_TRADER}: command channel restored after 5 failed claims; entries resumed"


def _restored_calls(trace: _Trace) -> list[tuple[Any, ...]]:
    return [c for c in trace.calls if c[0] == "alert" and "channel restored" in c[2]]


def _transitions(trace: _Trace) -> list[str]:
    """``to_state`` of every state_change event, in order."""
    return [
        c[2]["to_state"]
        for c in trace.calls
        if c[0] == "write" and c[2].get("action") == "state_change"
    ]


class _StaleSnapshot:
    """A kill-switch read that is too old for ``resume`` (limit 2.0 s)."""

    error_class = None
    engaged = False
    reason = None
    last_ok_mono = 0.0
    consecutive_failures = 0

    @staticmethod
    def age(_now: float) -> float:
        return 30.0


# ------------------------------------------------------------- R3 / R10


async def test_block_lifts_after_recovered_batch(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3/R10 through ``start()``: F x5 then S. The recovered rows run in id order while the
    streak is already reset and entries are still blocked; AFTER the batch the channel lifts
    its own pause: guard running, owns False, one failed + one restored event and alert."""
    ids = _enqueue(seeded["db"], ("pause", None), ("close_all", None))
    seen: list[tuple[str, int, bool, GuardState]] = []
    async with _started(monkeypatch, seeded, "FFFFF") as (node, trace, claims):

        async def execute(cmd: PaperCommand) -> tuple[bool, dict[str, Any], None]:
            assert node._guard is not None
            seen.append(
                (
                    cmd.command,
                    node._cmd_channel.failures,
                    node._cmd_channel.blocking,
                    node._guard.guard_state,
                )
            )
            return True, {}, None

        monkeypatch.setattr(node, "_execute_command", execute)
        await _until(lambda: len(_restored_calls(trace)) == 1)
        await _until(lambda: claims.calls >= 12)

        assert seen == [
            ("pause", 0, False, GuardState.PAUSED),
            ("close_all", 0, False, GuardState.PAUSED),
        ], "id order, streak reset first, still blocked while the batch runs"
        assert node._guard is not None
        assert node._guard.guard_state == GuardState.RUNNING
        assert node._cmd_channel.owns_pause is False
        assert [_row(seeded["db"], cid).status for cid in ids] == ["done", "done"]
        assert [c[2] for c in _channel_calls(trace)] == [_FAILED_5, _RESTORED_5]
        assert [c[3] for c in _channel_calls(trace)] == [True, True], "restored bypasses the 60 s limit"
        restored = _channel_calls(trace, "command_channel_restored")
        assert [c[2] for c in restored] == [
            {
                "action": "command_channel_restored",
                "failures": 5,
                "entries": "resumed",
                "state": "running",
                "passed": True,
            }
        ]
        assert _transitions(trace)[-2:] == ["paused", "running"]
        assert node.status.state == NodeState.RUNNING


# ------------------------------------------------------------------ R5 / R5b


async def test_queued_kill_runs_on_recovery_and_stays_halted(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R5: a kill queued during the outage is unread until a claim works; then it halts
    (the channel never resumes over a halt) and the restored alert says entries stay blocked."""
    async with _initialized(monkeypatch, seeded) as (node, trace):
        assert node._guard is not None
        (kill_id,) = _enqueue(seeded["db"], ("kill", {"message": "kill switch: test"}))
        _Claims(monkeypatch, node, "F" * 7)
        resumes: list[int] = []
        real_resume = node.resume
        monkeypatch.setattr(node, "resume", lambda: (resumes.append(1), real_resume())[1])

        assert await _polls(node, 7) == 7
        assert _row(seeded["db"], kill_id).status == "pending"
        assert node._guard.guard_state == GuardState.PAUSED

        await _polls(node, 1)

        assert _row(seeded["db"], kill_id).status == "done"
        assert node._guard.guard_state == GuardState.HALTED
        assert node._guard.halt_reason == HaltReason.KILL_SWITCH
        assert resumes == [], "a halt is never lifted by the channel"
        assert node._cmd_channel.owns_pause is True
        assert [c[2] for c in _restored_calls(trace)] == [
            f"paper {_TRADER}: command channel restored after 7 failed claims; "
            "entries stay blocked (halted/kill_switch)"
        ]


async def test_resume_after_recovered_kill_returns_running(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R5b: after R5 the operator resumes the kill halt: the guard returns to the PAUSED it
    was in before the halt, and the same tick's after-batch step lifts it to running."""
    async with _initialized(monkeypatch, seeded) as (node, trace):
        assert node._guard is not None
        _enqueue(seeded["db"], ("kill", None))
        _Claims(monkeypatch, node, "F" * 5)
        await _polls(node, 5)
        assert node._guard.guard_state == GuardState.PAUSED
        await _polls(node, 1)
        assert node._guard.guard_state == GuardState.HALTED
        assert node._cmd_channel.owns_pause is True
        (resume_id,) = _enqueue(seeded["db"], ("resume", None))

        await _polls(node, 1)

        assert _row(seeded["db"], resume_id).status == "done"
        assert node._guard.guard_state == GuardState.RUNNING
        assert node._cmd_channel.owns_pause is False
        assert len(_restored_calls(trace)) == 1, "no second restored alert"


# ------------------------------------------- R6 / R7 / R7b: no lift of a foreign block


@pytest.mark.parametrize(
    ("prepare", "expected_state"),
    [
        (_halt_kill, "halted/kill_switch"),
        (_halt_drawdown, "halted/max_drawdown"),
        (_operator_pause, "paused"),
    ],
    ids=["kill_switch", "max_drawdown", "operator_pause"],
)
async def test_recovery_never_lifts_a_block_it_does_not_own(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    prepare: Callable[[PaperTradingNode], None],
    expected_state: str,
) -> None:
    """R6/R7/R7b: a halt or an operator pause present when the channel failed stays; on
    recovery ``resume()`` is never called and the alert says entries stay blocked."""
    async with _initialized(monkeypatch, seeded) as (node, trace):
        assert node._guard is not None
        _Claims(monkeypatch, node, "F" * 5)
        prepare(node)
        before = (node._guard.guard_state, node._guard.halt_reason, node._guard.message)
        resumes: list[int] = []
        monkeypatch.setattr(node, "resume", lambda: (resumes.append(1), (True, "x"))[1])

        await _polls(node, 5)
        await _polls(node, 3)

        assert resumes == []
        assert (node._guard.guard_state, node._guard.halt_reason, node._guard.message) == before
        assert [c[2] for c in _restored_calls(trace)] == [
            f"paper {_TRADER}: command channel restored after 5 failed claims; "
            f"entries stay blocked ({expected_state})"
        ]


# ------------------------------------------------------------------ R10b


async def test_recovery_lift_refused_when_kill_snapshot_stale(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R10b: ``resume`` refuses without a fresh kill-switch read: the pause stays (owned),
    the restored alert says so; a later poll with a fresh read lifts it with no new alert."""
    async with _initialized(monkeypatch, seeded) as (node, trace):
        assert node._guard is not None
        _Claims(monkeypatch, node, "F" * 5)
        await _polls(node, 5)
        monkeypatch.setattr(node._kill_monitor, "snapshot", _StaleSnapshot)

        await _polls(node, 2)

        assert node._guard.guard_state == GuardState.PAUSED
        assert node._cmd_channel.owns_pause is True
        assert [c[2] for c in _restored_calls(trace)] == [
            f"paper {_TRADER}: command channel restored after 5 failed claims; "
            "entries stay blocked (paused)"
        ]

        _fresh_kill_snapshot(monkeypatch, node)
        await _polls(node, 1)

        assert node._guard.guard_state == GuardState.RUNNING
        assert node._cmd_channel.owns_pause is False
        assert len(_restored_calls(trace)) == 1
        assert len(_channel_calls(trace)) == 2, "one failed + one restored alert in total"


# ------------------------------------------------------- R10c / R10d / R10e


async def test_recovery_batch_pause_keeps_operator_pause(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R10c: a recovered [pause] is an operator decision: entries stay blocked, the channel
    no longer owns the pause, and no later poll lifts it."""
    async with _initialized(monkeypatch, seeded) as (node, trace):
        assert node._guard is not None
        _Claims(monkeypatch, node, "F" * 5)
        await _polls(node, 5)
        (pause_id,) = _enqueue(seeded["db"], ("pause", None))

        await _polls(node, 4)

        assert _row(seeded["db"], pause_id).status == "done"
        assert node._guard.guard_state == GuardState.PAUSED
        assert node._cmd_channel.owns_pause is False
        assert [c[2] for c in _restored_calls(trace)] == [
            f"paper {_TRADER}: command channel restored after 5 failed claims; "
            "entries stay blocked (paused)"
        ]


async def test_recovery_batch_resume(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R10d: a recovered [resume] resumes the channel pause itself; ownership is cleared
    (observed RUNNING) and the alert says entries resumed."""
    async with _initialized(monkeypatch, seeded) as (node, trace):
        assert node._guard is not None
        _Claims(monkeypatch, node, "F" * 5)
        await _polls(node, 5)
        (resume_id,) = _enqueue(seeded["db"], ("resume", None))

        await _polls(node, 1)

        assert _row(seeded["db"], resume_id).status == "done"
        assert node._guard.guard_state == GuardState.RUNNING
        assert node._cmd_channel.owns_pause is False
        assert [c[2] for c in _restored_calls(trace)] == [_RESTORED_5]


async def test_recovery_close_all_runs_blocked(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R10e: a recovered [close_all] waits for flat with entries still blocked; the lift
    comes after the batch."""
    async with _initialized(monkeypatch, seeded) as (node, trace):
        assert node._guard is not None
        _Claims(monkeypatch, node, "F" * 5)
        await _polls(node, 5)
        (close_id,) = _enqueue(seeded["db"], ("close_all", {"wait_secs": 10}))
        gate: asyncio.Future[list[str]] = asyncio.get_running_loop().create_future()
        entered = asyncio.Event()

        async def wait_flat(_timeout: float) -> list[str]:
            entered.set()
            return await gate

        monkeypatch.setattr(node, "_wait_flat", wait_flat)

        tick = asyncio.create_task(node.process_commands())
        await asyncio.wait_for(entered.wait(), 5)
        assert _row(seeded["db"], close_id).status == "running"
        assert node._guard.guard_state == GuardState.PAUSED, "blocked while it flattens"
        assert _restored_calls(trace) == []

        gate.set_result([])
        await asyncio.wait_for(tick, 5)

        assert _row(seeded["db"], close_id).status == "done"
        assert node._guard.guard_state == GuardState.RUNNING
        assert [c[2] for c in _restored_calls(trace)] == [_RESTORED_5]


# ------------------------------------------------------------------ R10f


async def test_recovery_owed_alert_after_outcome_write_failure(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R10f: the recovered batch raises in ``commands.complete``: the lift and the restored
    alert wait for the next good claim, and exactly one restored alert exists overall."""
    async with _initialized(monkeypatch, seeded) as (node, trace):
        assert node._guard is not None
        _Claims(monkeypatch, node, "F" * 5)
        await _polls(node, 5)
        _enqueue(seeded["db"], ("pause", None))

        async def execute(_cmd: PaperCommand) -> tuple[bool, dict[str, Any], None]:
            return True, {}, None

        monkeypatch.setattr(node, "_execute_command", execute)
        real_complete = PaperCommandQueue.complete

        def broken_complete(self: PaperCommandQueue, command_id: int, result: Any) -> None:
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(PaperCommandQueue, "complete", broken_complete)

        assert await _polls(node, 1) == 1
        assert node._guard.guard_state == GuardState.PAUSED, "lift deferred"
        assert _restored_calls(trace) == []

        monkeypatch.setattr(PaperCommandQueue, "complete", real_complete)
        await _polls(node, 3)

        assert node._guard.guard_state == GuardState.RUNNING
        assert [c[2] for c in _restored_calls(trace)] == [_RESTORED_5]


async def test_restored_alert_reaches_telegram_after_the_failed_alert(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R-c4: real SessionJournal + TelegramBot on mocked HTTP. The restored alert follows the
    streak's own critical alert within the bot's 60 s per-type limit and is still delivered."""
    sent: list[str] = []

    def record(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content)["text"])
        return httpx.Response(200, json={"ok": True})

    bot = TelegramBot(TelegramConfig(bot_token="T0K", chat_id="42"))

    async def client() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(record))

    monkeypatch.setattr(bot, "_get_client", client)
    monkeypatch.setattr(node_module, "TradingGuard", _Guard)
    node = PaperTradingNode(_config(seeded))
    node._journal.attach(None, bot, "s")
    monkeypatch.setattr(node._journal, "_write_event", lambda *_a, **_k: None)
    _fresh_kill_snapshot(monkeypatch, node)
    node._setup_signal_handlers()
    try:
        await node._initialize()
        _Claims(monkeypatch, node, "F" * 5)
        await _polls(node, 6)
        await node._journal.wait_alerts(None)
    finally:
        _remove_handlers()
        await node._shutdown()

    # Each Telegram text is "<header · time>\n<alert text>".
    assert [text.split("\n", 1)[1] for text in sent] == [_FAILED_5, _RESTORED_5]


@pytest.mark.parametrize("variant", ["complete_raises", "fail_result", "execute_raises"])
async def test_unrun_tail_runs_before_any_lift_after_outcome_write_failure(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    """R-c3: a recovered [close_all, kill] batch whose first outcome write raises (complete,
    or fail for a failed / raising command) leaves the kill claimed. It runs on the next tick,
    before any lift: entries are never reopened, the kill halts, nothing runs twice."""
    async with _initialized(monkeypatch, seeded) as (node, trace):
        assert node._guard is not None
        _Claims(monkeypatch, node, "F" * 5)
        await _polls(node, 5)
        close_id, kill_id = _enqueue(seeded["db"], ("close_all", None), ("kill", None))
        executed: list[str] = []
        real_execute = node._execute_command

        async def execute(cmd: PaperCommand) -> tuple[bool, dict[str, Any], str | None]:
            executed.append(cmd.command)
            if cmd.command == "close_all" and variant == "fail_result":
                return False, {}, "positions still open"
            if cmd.command == "close_all" and variant == "execute_raises":
                raise RuntimeError("flatten blew up")
            return await real_execute(cmd)

        monkeypatch.setattr(node, "_execute_command", execute)
        real_complete, real_fail = PaperCommandQueue.complete, PaperCommandQueue.fail

        def complete(self: PaperCommandQueue, command_id: int, result: Any) -> None:
            if command_id == close_id:
                raise sqlite3.OperationalError("disk I/O error")
            real_complete(self, command_id, result)

        def fail(self: PaperCommandQueue, command_id: int, error: str, result: Any = None) -> None:
            if command_id == close_id:
                raise sqlite3.OperationalError("disk I/O error")
            real_fail(self, command_id, error, result)

        monkeypatch.setattr(PaperCommandQueue, "complete", complete)
        monkeypatch.setattr(PaperCommandQueue, "fail", fail)

        assert await _polls(node, 1) == 1
        assert node._guard.guard_state == GuardState.PAUSED
        assert _row(seeded["db"], kill_id).status == "running", "claimed, not yet executed"

        await _polls(node, 3)

        assert node._guard.guard_state == GuardState.HALTED
        assert node._guard.halt_reason == HaltReason.KILL_SWITCH
        assert _row(seeded["db"], kill_id).status == "done"
        assert executed == ["close_all", "kill"]
        assert [c[2] for c in _restored_calls(trace)] == [
            f"paper {_TRADER}: command channel restored after 5 failed claims; "
            "entries stay blocked (halted/kill_switch)"
        ]


async def test_unrun_tail_survives_a_second_outcome_write_failure(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R-c3: [close_all, halt, kill], the first TWO outcome writes raise. Every raising tick
    keeps the entries blocked and the rest queued; the kill still runs, each command once."""
    async with _initialized(monkeypatch, seeded) as (node, trace):
        assert node._guard is not None
        _Claims(monkeypatch, node, "F" * 5)
        await _polls(node, 5)
        ids = _enqueue(seeded["db"], ("close_all", None), ("halt", None), ("kill", None))
        executed: list[str] = []
        real_execute = node._execute_command

        async def execute(cmd: PaperCommand) -> tuple[bool, dict[str, Any], str | None]:
            executed.append(cmd.command)
            return await real_execute(cmd)

        monkeypatch.setattr(node, "_execute_command", execute)
        real_complete = PaperCommandQueue.complete
        broken = {ids[0], ids[1]}

        def complete(self: PaperCommandQueue, command_id: int, result: Any) -> None:
            if command_id in broken:
                raise sqlite3.OperationalError("disk I/O error")
            real_complete(self, command_id, result)

        monkeypatch.setattr(PaperCommandQueue, "complete", complete)

        assert await _polls(node, 1) == 1
        assert node._guard.guard_state == GuardState.PAUSED
        assert await _polls(node, 1) == 1
        assert node._guard.guard_state == GuardState.HALTED, "halt ran on the retry"
        assert node._guard.halt_reason == HaltReason.MANUAL
        assert _restored_calls(trace) == []
        assert _row(seeded["db"], ids[2]).status == "running"

        await _polls(node, 2)

        assert node._guard.halt_reason == HaltReason.KILL_SWITCH
        assert _row(seeded["db"], ids[2]).status == "done"
        assert executed == ["close_all", "halt", "kill"]


async def test_stop_fails_unrun_tail_rows(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R-c3: the node stops while a claimed tail waits for its retry: the row ends failed
    (never left running), the executed command is not run again."""
    async with _initialized(monkeypatch, seeded) as (node, _trace):
        close_id, kill_id = _enqueue(seeded["db"], ("close_all", None), ("kill", None))
        executed: list[str] = []

        async def execute(cmd: PaperCommand) -> tuple[bool, dict[str, Any], None]:
            executed.append(cmd.command)
            return True, {}, None

        monkeypatch.setattr(node, "_execute_command", execute)

        def complete(self: PaperCommandQueue, command_id: int, result: Any) -> None:
            raise sqlite3.OperationalError("disk I/O error")

        real_complete = PaperCommandQueue.complete
        monkeypatch.setattr(PaperCommandQueue, "complete", complete)
        assert await _polls(node, 1) == 1
        monkeypatch.setattr(PaperCommandQueue, "complete", real_complete)
        await node._stop_control_loop()
        await _until(lambda: _row(seeded["db"], kill_id).status == "failed")

        row = _row(seeded["db"], kill_id)
        assert (row.status, row.error) == ("failed", "node stopping; command not executed")
        assert executed == ["close_all"]


# ------------------------------------------------------------------ R11


async def test_daily_loss_rollover_while_blocked_reblocks(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R11a: on a real NT engine the daily-loss halt lifts at the UTC rollover while the
    channel is still failing; the lift is re-paused inside the same guard transition, so
    not one entry is filled before or after midnight and the node ends PAUSED."""
    risk = RiskState(
        high_water_mark=Decimal(10000),
        day=date(2026, 1, 5),
        day_start_equity=Decimal(10000),
        last_equity=Decimal(10000),
    )
    h = harness.build(
        strategy_kwargs={"mode": "when_flat"},
        limits={"max_drawdown_pct": Decimal("0.5"), "max_daily_loss_pct": Decimal("0.5")},
        risk_state=risk,
    )
    node, trace = _trace_node(monkeypatch, seeded)
    node._guard = h.guard
    h.guard._listener = _GuardBridge(node)
    node._commands = PaperCommandQueue(seeded["db"])
    _Claims(monkeypatch, node, "F" * 5)
    try:
        assert h.guard.halt(HaltReason.MAX_DAILY_LOSS, "daily loss: test")
        assert await _polls(node, 5) == 5
        assert node._cmd_channel.blocking is True
        assert node._cmd_channel.owns_pause is False, "the daily-loss halt came first"

        bt = get_bar_type("BTCUSDT", "1m")
        h.engine.add_data(harness._flat_bars(bt, [10000.0] * 140))
        h.engine.run()

        midnight_ns = int(datetime(2026, 1, 6, tzinfo=UTC).timestamp() * 1e9)
        assert len(h.entry_orders()) >= 1, "the strategy really tried to enter"
        assert any(o.ts_init >= midnight_ns for o in h.entry_orders()), "also after midnight"
        assert h.filled_entries() == []
        assert h.guard.guard_state == GuardState.PAUSED
        assert node.status.state == NodeState.PAUSED
        assert node._cmd_channel.owns_pause is True
        assert _transitions(trace) == ["running", "paused"], "lifted, then re-paused at once"
    finally:
        await node._shutdown()
        h.engine.dispose()


async def test_sigusr2_resume_while_blocked_reblocks(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R11b: a real SIGUSR2 resume while the channel still fails is accepted, then paused
    again in the same call: SIGUSR2 cannot reopen entries during an outage."""
    async with _initialized(monkeypatch, seeded) as (node, trace):
        assert node._guard is not None
        _Claims(monkeypatch, node, "F" * 5)
        await _polls(node, 5)
        assert _transitions(trace) == ["paused"]

        with caplog.at_level(logging.WARNING):
            os.kill(os.getpid(), signal.SIGUSR2)
            await asyncio.sleep(0.05)

        # Accepted, then re-paused: the log says so, never that it resumed or was refused.
        assert "resume via SIGUSR2 refused" not in caplog.text
        assert "resume via SIGUSR2 not applied: resume re-paused, still paused: " in caplog.text
        assert _transitions(trace) == ["paused", "running", "paused"]
        assert node._guard.guard_state == GuardState.PAUSED
        assert node.status.state == NodeState.PAUSED
        assert node._cmd_channel.owns_pause is True


# ------------------------------------------------------------------ R12 / R13


async def test_each_streak_has_its_own_alert_pair(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R12: (F x5, S) x3 -> 3 failed + 3 restored alerts, in order, one pair per streak."""
    async with _initialized(monkeypatch, seeded) as (node, trace):
        assert node._guard is not None
        _Claims(monkeypatch, node, "FFFFFS" * 3)
        await _polls(node, 18)

        assert [c[2] for c in _channel_calls(trace)] == [_FAILED_5, _RESTORED_5] * 3
        assert node._guard.guard_state == GuardState.RUNNING


async def test_two_restarts_with_failing_channel(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R13a/R13c: restart #1 fails its drain (paused, then the run crashes); the pause is
    not restored, so restart #2 with a working drain starts RUNNING without channel
    alerts, and restart #3 with a failing drain pauses again with one alert."""
    drain_fails = [True]
    real_drain = PaperCommandQueue.claim_startup_halts

    def drain(self: PaperCommandQueue, trader_id: str) -> list[PaperCommand]:
        if drain_fails[0]:
            raise sqlite3.OperationalError("ZZLEAK1")
        return real_drain(self, trader_id)

    monkeypatch.setattr(PaperCommandQueue, "claim_startup_halts", drain)
    monkeypatch.setattr(node_module, "TradingGuard", _Guard)
    node1, trace1 = _trace_node(monkeypatch, seeded)

    async def crash(self: Any) -> None:
        raise RuntimeError("venue dropped")

    monkeypatch.setattr(_FakeTradingNode, "run_async", crash)
    try:
        with pytest.raises(RuntimeError, match="venue dropped"):
            await node1.start()
    finally:
        _remove_handlers()
    assert len(_channel_calls(trace1, "command_channel_failed")) == 1
    failed = _latest_checkpoint(seeded["db"])
    assert failed is not None
    assert failed.node_status.get("halt_reason") is None, "a channel pause is not persisted"

    drain_fails[0] = False
    async with _initialized(monkeypatch, seeded) as (node2, trace2):
        assert node2._guard is not None
        assert node2._guard.guard_state == GuardState.RUNNING
        assert node2._cmd_channel.failures == 0
        assert _channel_calls(trace2) == []

    drain_fails[0] = True
    async with _initialized(monkeypatch, seeded) as (node3, trace3):
        assert node3._guard is not None
        assert node3._guard.guard_state == GuardState.PAUSED
        assert len(_channel_calls(trace3)) == 1


async def test_startup_block_lifts_at_first_good_control_claim(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R13b: the start-up block (a failed drain) is lifted by the first good claim: one
    failed alert says 1 failure, one restored alert says 1 failed claim."""

    def boom(self: PaperCommandQueue, trader_id: str) -> list[PaperCommand]:
        raise sqlite3.OperationalError("ZZLEAK1")

    monkeypatch.setattr(PaperCommandQueue, "claim_startup_halts", boom)
    async with _initialized(monkeypatch, seeded) as (node, trace):
        assert node._guard is not None
        assert node._guard.guard_state == GuardState.PAUSED

        await _polls(node, 1)

        assert node._guard.guard_state == GuardState.RUNNING
        assert [c[2] for c in _restored_calls(trace)] == [
            f"paper {_TRADER}: command channel restored after 1 failed claims; entries resumed"
        ]


# =============================================================== T3: stalled claims


async def test_stalled_claim_counts_and_is_consumed_later(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R14: a claim that never answers counts one failure per poll (``no answer within 1 s``),
    only that ONE claim is ever submitted, entries block after 5; when it returns its rows
    run that tick (none left running) and the channel lifts."""
    monkeypatch.setattr(node_module, "COMMAND_CLAIM_STALL_S", 0.05)
    ids = _enqueue(seeded["db"], ("close_all", None), ("pause", None))
    executed: list[str] = []
    async with _started(monkeypatch, seeded, "H") as (node, trace, claims):

        async def execute(cmd: PaperCommand) -> tuple[bool, dict[str, Any], None]:
            executed.append(cmd.command)
            return True, {}, None

        monkeypatch.setattr(node, "_execute_command", execute)
        await _until(lambda: node._cmd_channel.blocking)
        assert claims.calls == 1, "a stalled claim is awaited, never queued behind"
        assert node._guard is not None
        assert node._guard.guard_state == GuardState.PAUSED
        assert [c[2] for c in _channel_calls(trace)] == [
            _FAILED_5.replace("OperationalError", "no answer within 1 s")
        ]
        assert [_row(seeded["db"], cid).status for cid in ids] == ["pending", "pending"]

        claims.hold.set()
        await _until(lambda: len(_restored_calls(trace)) == 1)

        assert executed == ["close_all", "pause"]
        assert [_row(seeded["db"], cid).status for cid in ids] == ["done", "done"]
        assert node._guard.guard_state == GuardState.RUNNING
        await _until(lambda: claims.calls >= 2)  # polling resumed with a fresh claim
        (restored,) = _restored_calls(trace)
        assert re.fullmatch(
            rf"paper {_TRADER}: command channel restored after \d+ failed claims; entries resumed",
            restored[2],
        )


async def test_slow_raising_claim_blocks_within_five_polls(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R14b: a claim that raises only after a busy-lock wait counts as stalls and then one
    failure: the 5th poll still blocks, and the raised claim is consumed (a new one follows)."""
    monkeypatch.setattr(node_module, "COMMAND_CLAIM_STALL_S", 0.05)
    async with _started(monkeypatch, seeded, "R", tail="F") as (node, trace, claims):
        await _until(lambda: node._cmd_channel.failures == 4)
        assert node._cmd_channel.blocking is False
        assert node._guard is not None
        assert node._guard.guard_state == GuardState.RUNNING
        assert claims.calls == 1

        claims.hold.set()
        await _until(lambda: node._cmd_channel.blocking)

        assert node._cmd_channel.failures == 5
        assert node._guard.guard_state == GuardState.PAUSED
        assert len(_channel_calls(trace)) == 1
        await _until(lambda: claims.calls >= 2)


@pytest.mark.parametrize(
    ("stall_s", "interval"),
    [(0.05, 30.0), (30.0, 0.01)],
    ids=["between_ticks", "mid_wait"],
)
async def test_stop_with_inflight_claim_fails_rows(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    stall_s: float,
    interval: float,
) -> None:
    """R14c: the node stops while a claim that already took the rows is still unanswered
    (control loop asleep, or inside its wait): the rows end ``failed``, never ``running``."""
    monkeypatch.setattr(node_module, "COMMAND_CLAIM_STALL_S", stall_s)
    (cid,) = _enqueue(seeded["db"], ("close_all", None))
    async with _started(monkeypatch, seeded, "C", interval=interval) as (node, _trace, claims):
        await _until(lambda: claims.calls == 1 and _row(seeded["db"], cid).status == "running")
        if stall_s < 1:
            await _until(lambda: node._claim_inflight is not None and node._cmd_channel.failures == 1)
        threading.Timer(0.5, claims.hold.set).start()

    row = _row(seeded["db"], cid)
    assert (row.status, row.error) == ("failed", "node stopping; command not executed")
