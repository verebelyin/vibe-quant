"""The kill-switch reader thread and the node's cached kill paths (bead vibe-quant-dt065, T2a).

Contract rows R10-R18 of job 20261016-safety-bankroll (design B2). The node never
reads the kill switch on its event loop: ``KillSwitchMonitor`` owns a reader thread
and a snapshot, and the watcher / ``resume()`` / startup read only that snapshot.

* R10  deleted row -> the monitor reads it engaged -> one KILL_SWITCH halt, one
       CRITICAL alert (fake guard that reports like the real one)
* R11  last clean read 7.999 s old -> no halt; R11b 8.000 s -> ERROR halt, exact text
* R12  one 5 s read that succeeds -> no halt
* R13  REAL thread, every read after the first sleeps 5 s then raises -> one ERROR
       halt <= 15 s after the first failing read started (timing)
* R13b REAL thread, the read hangs HOLDING the reader's ``_write_lock`` -> the halt
       still comes <= 15 s, ``resume()`` and ``process_commands()`` answer at once,
       ``stop(1.0)`` returns <= 1.2 s, and the reader thread closes its own handle
       once the read is released (timing)
* R14  engaged switch -> KILL_SWITCH halt <= 1.5 s (poll 1.0 + watcher tick 0.5)
* R15  stale latch, then a clean read -> latch cleared, still HALTED, operator resume
* R15b resume with a 2.0 s old clean read allowed, 2.001 s refused
* R16  startup without a clean read -> refuse "(no read within 10 s)", loop not blocked
* R17  startup with the switch engaged -> refuse with the engaged text
* R18  failure logs carry the exception class only (no text, no chain, no traceback)
"""

# ruff: noqa: F811
from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
import threading
import time
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_audit_paper_node import _config, _FakeTradingNode
from tests.unit.test_paper_kill_switch_fail_closed import (
    _CTX_LEAK,
    _LEAK,
    FakeClock,
    _node_with_guard,
    _raise_read_error,
    _real_guard,
    _RecordGuard,
    _rig,
    _ScriptedReads,
    fake_tradingnode,  # noqa: F401  # fixture
    seeded,  # noqa: F401  # fixture
)
from vibe_quant.db.state_manager import StateManager
from vibe_quant.paper import kill_switch_monitor as ksm
from vibe_quant.paper.config import ConfigurationError
from vibe_quant.paper.guard import GuardState, HaltReason
from vibe_quant.paper.kill_switch_monitor import KillSwitchMonitor
from vibe_quant.paper.node import PaperTradingNode
from vibe_quant.paper.persistence import PaperCommandQueue

if TYPE_CHECKING:
    from collections.abc import Callable

_RESEEDED = "kill-switch record was missing; re-created engaged"


def test_monitor_constants_and_old_count_rule_gone() -> None:
    """The cadence the bound is derived from; the 3-read count rule no longer exists."""
    import vibe_quant.paper.node as node_module

    assert ksm.KILL_SWITCH_POLL_S == 1.0
    assert ksm.KILL_SWITCH_STALE_LIMIT_S == 8.0
    assert ksm.KILL_SWITCH_WATCH_TICK_S == 0.5
    assert ksm.KILL_SWITCH_RESUME_FRESH_S == 2.0
    assert ksm.KILL_SWITCH_STARTUP_TIMEOUT_S == 10.0
    assert not hasattr(node_module, "KILL_SWITCH_READ_FAILURE_LIMIT")
    assert not hasattr(PaperTradingNode, "_kill_switch")


# ------------------------------------------------------------ R10


class _AlertingGuard(_RecordGuard):
    """Reports halts to the node like the real guard's state-change listener."""

    def __init__(self, node: PaperTradingNode) -> None:
        super().__init__()
        self._node = node

    def halt(self, reason: HaltReason, message: str) -> bool:
        super().halt(reason, message)
        self._node._on_guard_state_change(GuardState.HALTED, reason, message, {})
        return True


def _wait_until(predicate: Callable[[], bool], timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


async def test_deleted_row_halts_kill_switch_once(
    seeded: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R10: a row deleted under a running node is re-created engaged by the
    reader's next read; the watcher halts KILL_SWITCH once with the reseed reason
    and sends exactly one CRITICAL alert."""
    node = PaperTradingNode(_config(seeded))
    guard = _AlertingGuard(node)
    node._guard = guard  # type: ignore[assignment]
    alerts: list[tuple[str, str, bool]] = []
    monkeypatch.setattr(
        node._journal,
        "alert",
        lambda alert: alerts.append((alert.kind, alert.message, alert.critical)),
    )
    node._kill_monitor = KillSwitchMonitor(lambda: StateManager(seeded["db"]), poll_s=0.05)
    node._kill_monitor.start()
    try:
        snap = await node._kill_monitor.wait_first_read(5.0)
        assert snap.engaged is False
        node._check_kill_switch()
        assert guard.calls == []
        other = StateManager(seeded["db"])  # a fresh connection, like the real deleter
        try:
            other.conn.execute("DELETE FROM system_state")
            other.conn.commit()
        finally:
            other.close()
        assert _wait_until(lambda: node._kill_monitor.snapshot().engaged, 5.0)
        for _ in range(3):
            node._check_kill_switch()
        assert guard.calls == [("halt", HaltReason.KILL_SWITCH)]
        assert guard.messages == [f"system kill switch: {_RESEEDED}"]
        assert alerts == [
            ("circuit_breaker", f"HALT (kill_switch): system kill switch: {_RESEEDED}", True)
        ]
    finally:
        await node._shutdown()


# ------------------------------------------------------------ R11 / R11b / R12


async def test_stale_edge_below_limit(seeded: dict[str, Any]) -> None:
    """R11: the last clean read is 7.999 s old - no halt."""
    guard = _RecordGuard()
    node = _node_with_guard(seeded, guard)
    rig = _rig(node, "FE")
    try:
        rig.read(0.0)
        rig.read(0.001)
        rig.check(7.998)  # 7.999 s after the clean read at t=0.0
        assert rig.clock.now - 0.0 == pytest.approx(7.999)
        assert guard.calls == []
    finally:
        await node._shutdown()


async def test_stale_edge_at_limit_halts(seeded: dict[str, Any]) -> None:
    """R11b: exactly 8.000 s old - ERROR halt with the exact text. The edge is
    ``>=``: a ``>`` would wait for the next tick."""
    guard = _RecordGuard()
    node = _node_with_guard(seeded, guard)
    rig = _rig(node, "FE")
    try:
        rig.read(0.0)
        rig.read(1.0)
        rig.check(7.0)
        assert rig.clock.now == 8.0
        assert guard.calls == [("halt", HaltReason.ERROR)]
        assert guard.messages == [
            "system kill switch unreadable for 8.0 s (OperationalError); failing closed"
        ]
    finally:
        await node._shutdown()


async def test_single_slow_read_tolerated(seeded: dict[str, Any]) -> None:
    """R12: one read that takes 5 s and then succeeds - the snapshot age peaks at
    5 s while it runs and no tick halts; after it the age is back to 0."""
    guard = _RecordGuard()
    node = _node_with_guard(seeded, guard)
    rig = _rig(node, "F")
    try:
        rig.read(0.0)
        for _ in range(10):  # watcher ticks while the second read is in flight
            rig.check(0.5)
        assert rig.clock.now == 5.0
        rig.read(0.0)  # the slow read completes cleanly
        for _ in range(10):  # polling resumes at the normal 1 s cadence
            rig.read(1.0)
            rig.check()
        assert guard.calls == []
    finally:
        await node._shutdown()


# ------------------------------------------------------------ R13 / R13b (REAL thread)


class _SlowFailingReader:
    """The first read is clean; every later read sleeps 5 s, then raises."""

    def __init__(self, release: threading.Event) -> None:
        self.reads = 0
        self.first_fail_start: float | None = None
        self.release = release

    def get_system_state(self) -> dict[str, Any]:
        self.reads += 1
        if self.reads == 1:
            return {"kill_switch": False}
        if self.first_fail_start is None:
            self.first_fail_start = time.monotonic()
        self.release.wait(5.0)
        raise sqlite3.OperationalError("database is locked")

    def close(self) -> None:
        return None


class _TimedGuard(_RecordGuard):
    def __init__(self) -> None:
        super().__init__()
        self.halt_times: list[float] = []

    def halt(self, reason: HaltReason, message: str) -> bool:
        self.halt_times.append(time.monotonic())
        return super().halt(reason, message)


async def _run_watcher_until_halt(node: PaperTradingNode, guard: _TimedGuard) -> None:
    watch = asyncio.create_task(node._kill_watch_loop())
    try:
        deadline = time.monotonic() + 20.0
        while not guard.calls and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
    finally:
        node._shutdown_event.set()
        await asyncio.wait_for(watch, 5)


@pytest.mark.timing
async def test_slow_failures_halt_within_15s(seeded: dict[str, Any]) -> None:
    """R13: REAL reader thread; after one clean read every read sleeps 5 s and
    raises. One ERROR halt comes at last_ok + 8.0 + the watcher tick, i.e. well
    within 15 s of the first failing read's start (the old 3-read rule needed ~33 s)."""
    guard = _TimedGuard()
    node = _node_with_guard(seeded, guard)
    release = threading.Event()
    reader = _SlowFailingReader(release)
    node._kill_monitor = KillSwitchMonitor(lambda: reader)
    node._kill_monitor.start()
    try:
        first = await node._kill_monitor.wait_first_read(5.0)
        assert first.last_ok_mono is not None
        await _run_watcher_until_halt(node, guard)
        assert guard.calls == [("halt", HaltReason.ERROR)]
        assert reader.first_fail_start is not None
        elapsed = guard.halt_times[0] - reader.first_fail_start
        assert 6.5 <= elapsed <= 15.0, elapsed
        assert re.fullmatch(
            r"system kill switch unreadable for \d+\.\d s \(OperationalError\); failing closed",
            guard.messages[0],
        ), guard.messages[0]
    finally:
        release.set()
        node._kill_monitor.stop(2.0)
        await node._shutdown()


@pytest.mark.timing
async def test_hung_reader_resume_nonblocking_and_stop_bounded(
    seeded: dict[str, Any],
) -> None:
    """R13b: REAL thread whose read hangs forever HOLDING the reader state
    manager's ``_write_lock``. The halt still comes <= 15 s; ``resume()`` and
    ``process_commands()`` with a queued resume answer from the cache at once
    (refused: not fresh); ``stop(1.0)`` returns <= 1.2 s although the thread is
    stuck; once the read is released the reader thread closes its own handle,
    exactly once, on the reader thread."""
    release = threading.Event()
    closes: list[str] = []
    hang_start: list[float] = []

    class _HungStateManager(StateManager):
        def __init__(self, db_path: Any) -> None:
            super().__init__(db_path)
            self.reads = 0

        def get_system_state(self) -> Any:
            self.reads += 1
            if self.reads == 1:
                return super().get_system_state()
            hang_start.append(time.monotonic())
            with self._write_lock:
                release.wait()
            return super().get_system_state()

        def close(self) -> None:
            closes.append(threading.current_thread().name)
            super().close()

    guard = _TimedGuard()
    node = _node_with_guard(seeded, guard)
    node._commands = PaperCommandQueue(seeded["db"])
    node._kill_monitor = KillSwitchMonitor(lambda: _HungStateManager(seeded["db"]))
    node._kill_monitor.start()
    queue = PaperCommandQueue(seeded["db"])
    try:
        first = await node._kill_monitor.wait_first_read(5.0)
        assert first.last_ok_mono is not None
        await _run_watcher_until_halt(node, guard)
        assert guard.calls == [("halt", HaltReason.ERROR)]
        assert guard.halt_times[0] - hang_start[0] <= 15.0

        started = time.monotonic()
        ok, msg = node.resume()
        assert time.monotonic() - started <= 0.1
        assert ok is False
        assert re.fullmatch(
            r"system kill switch state not fresh \(\d+\.\d s old\);"
            r" cannot resume until it reads cleanly",
            msg,
        ), msg

        cid = queue.enqueue(node._config.trader_id, "resume")
        started = time.monotonic()
        await node.process_commands()
        assert time.monotonic() - started <= 1.0
        cmd = queue.get(cid)
        assert cmd is not None and cmd.status == "failed", cmd
        assert cmd.error is not None and cmd.error.startswith("system kill switch state not fresh")

        started = time.monotonic()
        node._kill_monitor.stop(join_timeout_s=1.0)
        assert time.monotonic() - started <= 1.2
        assert closes == [], "a hung read keeps the handle open: stop() never closes it"
    finally:
        release.set()
        assert _wait_until(lambda: closes != [], 5.0)
        queue.close()
        await node._shutdown()
    assert closes == ["vq-kill-switch-reader"]


# ------------------------------------------------------------ R14


async def test_engaged_detected_within_poll_plus_tick(seeded: dict[str, Any]) -> None:
    """R14: the switch engages right after a clean read. Worst case the watcher
    tick runs just before the next poll, so the halt comes at the tick after:
    poll 1.0 + tick 0.5 = 1.5 s (fake clock, events in time order)."""
    guard = _RecordGuard()
    node = _node_with_guard(seeded, guard)
    rig = _rig(node, "FT")
    try:
        rig.read(0.0)  # t=0.0 clean, not engaged (the switch engages at t=0.0+)
        halt_at: float | None = None
        # watcher ticks every 0.5 s; the next read is at t=1.0, ordered AFTER the t=1.0 tick
        for t in (0.5, 1.0):
            rig.check(t - rig.clock.now)
            assert guard.calls == []
        rig.read(0.0)  # t=1.0: the engaged read
        for t in (1.5,):
            rig.check(t - rig.clock.now)
            if guard.calls:
                halt_at = rig.clock.now
        assert guard.calls == [("halt", HaltReason.KILL_SWITCH)]
        assert halt_at is not None and halt_at <= 1.5
    finally:
        await node._shutdown()


# ------------------------------------------------------------ R15 / R15b


async def test_recovery_needs_operator_resume(seeded: dict[str, Any]) -> None:
    """R15: the stale latch clears on a clean not-engaged read, the guard stays
    HALTED (no auto-resume) and an operator resume is allowed once fresh."""
    guard = _real_guard()
    node = _node_with_guard(seeded, guard)
    rig = _rig(node, "FEF")
    try:
        rig.read(0.0)
        rig.read(1.0)
        rig.check(7.0)
        assert node._kill_read_halted is True
        assert guard.halt_reason == HaltReason.ERROR
        rig.read(1.0)
        rig.check()
        assert node._kill_read_halted is False
        assert guard.guard_state == GuardState.HALTED
        ok, _msg = node.resume()
        assert ok
        assert guard.guard_state == GuardState.RUNNING
    finally:
        await node._shutdown()


async def test_resume_freshness_edge(seeded: dict[str, Any]) -> None:
    """R15b: a clean read 2.0 s old allows resume, 2.001 s old refuses it."""
    guard = _RecordGuard()
    node = _node_with_guard(seeded, guard)
    rig = _rig(node, "F")
    try:
        rig.read(0.0)
        rig.clock.now = 2.0
        assert node.resume() == (True, "resumed")
        rig.clock.now = 2.001
        ok, msg = node.resume()
        assert ok is False
        assert msg == (
            "system kill switch state not fresh (2.0 s old); cannot resume until it reads cleanly"
        )
        assert guard.calls == [("resume", False)]
    finally:
        await node._shutdown()


async def test_resume_refused_when_engaged_from_cache(seeded: dict[str, Any]) -> None:
    """A fresh engaged snapshot refuses resume through the guard (today's text)."""
    guard = _RecordGuard()
    node = _node_with_guard(seeded, guard)
    rig = _rig(node, "T")
    try:
        rig.read(0.0)
        assert node.resume() == (
            False,
            "system kill switch is engaged; unlock it before resuming",
        )
        assert guard.calls == [("resume", True)]
    finally:
        await node._shutdown()


# ------------------------------------------------------------ R16 / R17


async def test_startup_waits_nonblocking_then_refuses(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R16: no clean read within the startup wait -> ConfigurationError with the
    exact text, no TradingNode built, and the loop keeps ticking while it waits."""
    monkeypatch.setattr("vibe_quant.paper.node.KILL_SWITCH_STARTUP_TIMEOUT_S", 0.4)
    release = threading.Event()

    class _Hung:
        def get_system_state(self) -> Any:
            release.wait()

        def close(self) -> None:
            return None

    node = PaperTradingNode(_config(seeded))
    node._kill_monitor = KillSwitchMonitor(lambda: _Hung())
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.01)

    task = asyncio.create_task(ticker())
    try:
        with pytest.raises(ConfigurationError) as exc_info:
            await node._initialize()
        assert (
            str(exc_info.value)
            == "System kill switch state unreadable (no read within 10 s); refusing to start"
        )
        assert ticks >= 10, "the startup wait must not block the event loop"
        assert fake_tradingnode.instances == []
    finally:
        task.cancel()
        release.set()
        await node._shutdown()


async def test_startup_refuses_engaged(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """R17: an engaged switch at startup refuses with today's text."""
    sm = StateManager(seeded["db"])
    sm.set_kill_switch("audit test")
    sm.close()
    node = PaperTradingNode(_config(seeded))
    try:
        with pytest.raises(ConfigurationError) as exc_info:
            await node._initialize()
        assert str(exc_info.value) == "System kill switch engaged (audit test); refusing to start"
        assert fake_tradingnode.instances == []
    finally:
        await node._shutdown()


# ------------------------------------------------------------ R18


def test_failure_log_class_only(caplog: pytest.LogCaptureFixture) -> None:
    """R18: neither the error text (ZZLEAK9), the ``__context__`` chain (QQLEAK7) nor a
    traceback reaches the log, for a failed read, an unopenable reader and a failing close."""

    class _Failing:
        def get_system_state(self) -> Any:
            _raise_read_error()

        def close(self) -> None:
            try:
                raise RuntimeError(f"fetch failed {_CTX_LEAK}")
            except RuntimeError:
                raise OSError(f"close {_LEAK}")  # noqa: B904  # keep __context__

    def open_failing() -> Any:
        try:
            raise RuntimeError(f"fetch failed {_CTX_LEAK}")
        except RuntimeError:
            raise sqlite3.OperationalError(f"unable to open {_LEAK}")  # noqa: B904

    monitor = KillSwitchMonitor(_Failing, poll_s=0.01)
    with caplog.at_level(logging.ERROR):
        monitor.poll_once(_Failing())
        monitor.start()
        assert _wait_until(lambda: monitor.snapshot().consecutive_failures >= 2, 5.0)
        monitor.stop(2.0)
        assert _wait_until(lambda: "close failed" in caplog.text, 5.0)
        opener = KillSwitchMonitor(open_failing, poll_s=0.01)
        opener.start()
        assert _wait_until(lambda: opener.snapshot().consecutive_failures >= 1, 5.0)
        opener.stop(2.0)
    assert "OperationalError" in caplog.text
    assert "kill-switch reader close failed (OSError)" in caplog.text
    assert _LEAK not in caplog.text
    assert _CTX_LEAK not in caplog.text
    assert "Traceback" not in caplog.text
    assert all(r.exc_info is None for r in caplog.records)


# ------------------------------------------------------------ monitor mechanics


def test_reader_is_opened_and_closed_on_the_reader_thread() -> None:
    """B2: the reader thread opens its handle and, after ``stop()``, closes it
    exactly once on the same thread; ``stop()`` itself never closes it."""
    events: list[tuple[str, str]] = []

    class _Reader:
        def get_system_state(self) -> dict[str, Any]:
            return {"kill_switch": True, "reason": "ops"}

        def close(self) -> None:
            events.append(("close", threading.current_thread().name))

    def open_reader() -> _Reader:
        events.append(("open", threading.current_thread().name))
        return _Reader()

    monitor = KillSwitchMonitor(open_reader, poll_s=0.01)
    monitor.stop()  # never started: harmless
    monitor.start()
    monitor.start()  # idempotent
    assert _wait_until(lambda: monitor.snapshot().engaged, 5.0)
    snap = monitor.snapshot()
    assert (snap.reason, snap.error_class, snap.consecutive_failures) == ("ops", None, 0)
    monitor.stop(2.0)
    monitor.stop(2.0)
    assert events == [("open", "vq-kill-switch-reader"), ("close", "vq-kill-switch-reader")]


def test_unopenable_reader_is_retried() -> None:
    """A reader that cannot be opened is a failed read and is retried next poll."""
    attempts: list[int] = []

    def open_reader() -> _ScriptedReads:
        attempts.append(1)
        if len(attempts) == 1:
            raise sqlite3.OperationalError("unable to open")
        return _ScriptedReads("F")

    monitor = KillSwitchMonitor(open_reader, poll_s=0.01)
    monitor.start()
    try:
        assert _wait_until(lambda: monitor.snapshot().last_ok_mono is not None, 5.0)
        assert monitor.snapshot().error_class is None
        assert len(attempts) == 2
    finally:
        monitor.stop(2.0)


async def test_wait_first_read_returns_on_first_clean_read_or_timeout() -> None:
    """``wait_first_read`` returns the clean snapshot at once, or the empty one at
    the timeout, and does not hold the loop in between."""
    clock = FakeClock(5.0)
    fresh = KillSwitchMonitor(lambda: _ScriptedReads("F"), clock=clock)
    fresh.poll_once(_ScriptedReads("F"))
    snap = await fresh.wait_first_read(10.0)
    assert snap.last_ok_mono == 5.0

    empty = KillSwitchMonitor(lambda: _ScriptedReads("F"), clock=clock)
    started = time.monotonic()
    snap = await empty.wait_first_read(0.25)
    assert snap.last_ok_mono is None
    assert 0.2 <= time.monotonic() - started < 2.0


# ------------------------------------------------------------ stop()/start() contract


def test_second_stop_does_not_join_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """K1: with a hung read (it blocks on an event) ``stop()`` joins at most once across
    repeated calls, so the node's two stop sites (``_run_loop`` finally + ``_shutdown``)
    pay one join timeout, not two."""
    release = threading.Event()
    reading = threading.Event()

    class _Hung:
        def get_system_state(self) -> dict[str, Any]:
            reading.set()
            release.wait(5)
            return {"kill_switch": False, "reason": None}

        def close(self) -> None:
            return None

    monitor = KillSwitchMonitor(_Hung, poll_s=0.01)
    monitor.start()
    assert reading.wait(5)
    thread = monitor._thread
    assert thread is not None
    joins: list[float | None] = []
    real_join = thread.join

    def spy(timeout: float | None = None) -> None:
        joins.append(timeout)
        real_join(timeout)

    monkeypatch.setattr(thread, "join", spy)
    try:
        started = time.monotonic()
        monitor.stop(join_timeout_s=0.2)
        monitor.stop(join_timeout_s=0.2)
        assert joins == [0.2], "the second stop must not join the thread again"
        assert time.monotonic() - started < 0.6
        assert thread.is_alive(), "a hung read outlives the bounded join"
    finally:
        release.set()
        real_join(5)


def test_start_after_stop_refuses_while_double_start_is_idempotent() -> None:
    """K2: ``start()`` twice without a stop stays idempotent (one reader thread), but a
    same-object restart after ``stop()`` is refused explicitly - the node must never pass
    startup on a snapshot taken before the stop."""
    monitor = KillSwitchMonitor(lambda: _ScriptedReads("F"), poll_s=0.01)
    monitor.start()
    first = monitor._thread
    monitor.start()
    assert monitor._thread is first
    assert first is not None and first.is_alive()

    monitor.stop(2.0)
    with pytest.raises(RuntimeError, match="KillSwitchMonitor cannot be restarted"):
        monitor.start()
