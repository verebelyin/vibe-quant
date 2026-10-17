"""Kill switch fail-closed reads (beads vibe-quant-s40h9 K1, vibe-quant-dt065 T2a).

The node never reads the kill switch on its event loop: a reader thread
publishes a snapshot (``vibe_quant/paper/kill_switch_monitor.py``) and the
watcher (``_check_kill_switch``), ``resume()`` and startup read ONLY that
snapshot. A failed read is not engaged; the node fails closed (halt ERROR) when
the last CLEAN read is 8.0 s old, startup refuses without a clean read, and
resume needs a clean read at most 2.0 s old. These tests drive the monitor with
a fake clock and a scripted reader (outcomes ``E`` raises, ``F`` clean and not
engaged, ``T`` clean and engaged, ``P`` an unparseable state); the R-row timing
tests with a REAL thread live in ``test_paper_kill_switch_monitor.py``.

* K1  one failed read after a clean one -> no halt, one class-only ERROR log
* K2  two failed reads, snapshot still fresh -> no halt (no read COUNT rule)
* K3  last clean read 8.0 s old -> one ERROR halt with the exact text
* K4  stale for longer -> still one halt (the latch)
* K5  a clean read resets the failure count and the latch
* K6  clean read after the halt -> HALTED/ERROR stays, then an operator resume works
* K7  resume while the snapshot has a failed read -> refused, guard never reached
* K8  startup without a clean read -> ConfigurationError, no TradingNode
* K10 failing process_commands never skips the kill check (own watcher task)
* K11 failed reads then an engaged read -> halt(KILL_SWITCH)
* K12 ERROR halt then an engaged read -> the real guard escalates to KILL_SWITCH
* K16 a raising state parse counts as a failed read
* M4  second stale streak after an operator resume halts ERROR again
* F2  the reader asks ``get_system_state()`` with no kwargs (the T2a SF2 completion)
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from typing import TYPE_CHECKING, Any, NoReturn

import pytest

from tests.unit.test_audit_paper_node import DSL, _config, _FakeTradingNode
from vibe_quant.db.state_manager import StateManager
from vibe_quant.paper.config import ConfigurationError
from vibe_quant.paper.guard import GuardState, HaltReason, TradingGuard, TradingGuardConfig
from vibe_quant.paper.kill_switch_monitor import KillSwitchMonitor
from vibe_quant.paper.node import PaperTradingNode

if TYPE_CHECKING:
    from pathlib import Path

#: Leak markers: contain none of the redacted credential names, so a mutant that
#: drops the class-only rule cannot hide behind them (impl-sb-watchlog-2).
_LEAK = "ZZLEAK9"
_CTX_LEAK = "QQLEAK7"


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
    """Redirect NT's TradingNode to the recording fake (copy of the audit fixture)."""
    import nautilus_trader.live.node as live_node

    _FakeTradingNode.instances = []
    monkeypatch.setattr(live_node, "TradingNode", _FakeTradingNode)
    return _FakeTradingNode


# --------------------------------------------------------------- fakes


def _raise_read_error() -> NoReturn:
    """The failed read: SQLITE_BUSY with a chained fetch error kept as __context__."""
    try:
        raise RuntimeError(f"fetch failed {_CTX_LEAK}")
    except RuntimeError:
        raise sqlite3.OperationalError(f"database is locked {_LEAK}")  # noqa: B904  # keep __context__


class _BadParseState:
    """A state row whose parse raises (K16)."""

    def get(self, key: str, default: Any = None) -> Any:
        raise RuntimeError(f"parse {_CTX_LEAK}")


class _ScriptedReads:
    """The reader thread's DB handle following a script of outcomes: E raises, F is
    clean and not engaged, T is clean and engaged, P returns an unparseable state.

    When the script runs out, its last outcome repeats. ``reads`` records one entry
    per call. ``get_system_state`` takes NO kwargs (F2: the ``require_row`` flag is
    gone), so a caller passing one raises ``TypeError``."""

    def __init__(self, outcomes: str) -> None:
        self._outcomes = list(outcomes)
        self.reads: list[str] = []
        self.closed = 0

    def get_system_state(self) -> dict[str, Any]:
        index = min(len(self.reads), len(self._outcomes) - 1)
        outcome = self._outcomes[index]
        self.reads.append(outcome)
        if outcome == "E":
            _raise_read_error()
        if outcome == "P":
            return _BadParseState()  # type: ignore[return-value]
        if outcome == "T":
            return {"kill_switch": True, "reason": "ops"}
        return {"kill_switch": False}

    def close(self) -> None:
        self.closed += 1


class FakeClock:
    """A monotonic clock the test moves by hand."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class Rig:
    """A node wired to a fake-clock monitor over a scripted reader (no thread).

    ``read(dt)`` moves the clock ``dt`` seconds and runs one read cycle (the
    snapshot's ``last_ok_mono`` is the clock after the read); ``check(dt)`` moves
    the clock and runs the watcher once."""

    def __init__(self, node: PaperTradingNode, outcomes: str) -> None:
        self.node = node
        self.clock = FakeClock()
        self.script = _ScriptedReads(outcomes)
        self.monitor = KillSwitchMonitor(lambda: self.script, clock=self.clock)
        node._kill_monitor = self.monitor
        node._mono = self.clock  # type: ignore[method-assign]

    def read(self, dt: float = 1.0) -> None:
        self.clock.now += dt
        self.monitor.poll_once(self.script)

    def check(self, dt: float = 0.0) -> None:
        self.clock.now += dt
        self.node._check_kill_switch()


def _rig(node: PaperTradingNode, outcomes: str) -> Rig:
    return Rig(node, outcomes)


class _RecordGuard:
    """Fake guard recording ``(kind, detail)`` calls like the audit ``_FakeGuard``,
    plus every halt message (K3 pins the exact fail-closed text)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.messages: list[str] = []
        self.guard_state = GuardState.RUNNING
        self.halt_reason: HaltReason | None = None

    def halt(self, reason: HaltReason, message: str) -> bool:
        self.calls.append(("halt", reason))
        self.messages.append(message)
        self.guard_state = GuardState.HALTED
        self.halt_reason = reason
        return True

    def resume(self, *, kill_switch_engaged: bool = False) -> tuple[bool, str]:
        self.calls.append(("resume", kill_switch_engaged))
        if kill_switch_engaged:
            return False, "system kill switch is engaged; unlock it before resuming"
        self.guard_state = GuardState.RUNNING
        self.halt_reason = None
        return True, "resumed"


class _EmptyCache:
    def positions_open(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    def orders_open(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


class _RiskEngine:
    def execute(self, command: object) -> None:
        return None


class _RealGuard(TradingGuard):
    """A real :class:`TradingGuard` with an injected cache: an unregistered Actor
    gets ``cache = None`` and ``_flatten`` would crash. Halt priorities here are
    the real ones (K12)."""

    @property
    def cache(self) -> Any:  # type: ignore[override]
        return _EmptyCache()


def _real_guard() -> TradingGuard:
    return _RealGuard(TradingGuardConfig(), strategies=[], risk_engine=_RiskEngine())


def _node_with_guard(seeded: dict[str, Any], guard: Any) -> PaperTradingNode:
    node = PaperTradingNode(_config(seeded))
    node._guard = guard  # type: ignore[assignment]
    return node


# --------------------------------------------------------------- K1/K2/K3/K4/K5


async def test_single_failure_tolerated(
    seeded: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """K1: one failed read after a clean one - no halt, one class-only ERROR log,
    no exc_info (the snapshot is 1 s old, far below the 8 s limit)."""
    guard = _RecordGuard()
    node = _node_with_guard(seeded, guard)
    rig = _rig(node, "FE")
    try:
        rig.read()
        with caplog.at_level(logging.ERROR):
            rig.read()
        rig.check()
        assert rig.script.reads == ["F", "E"]
        assert guard.calls == []
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert (
            errors[0].getMessage()
            == "could not read system kill switch (OperationalError); 1 consecutive failure(s)"
        )
        assert errors[0].exc_info is None
    finally:
        await node._shutdown()


async def test_two_failures_tolerated(seeded: dict[str, Any]) -> None:
    """K2: two failed reads while the last clean read is still fresh - no halt:
    the rule is the age of the last clean read, not a count of failed reads."""
    guard = _RecordGuard()
    node = _node_with_guard(seeded, guard)
    rig = _rig(node, "FEE")
    try:
        for _ in range(3):
            rig.read()
            rig.check()
        assert rig.script.reads == ["F", "E", "E"]
        assert node._kill_monitor.snapshot().consecutive_failures == 2
        assert guard.calls == []
    finally:
        await node._shutdown()


async def test_stale_snapshot_halts_error_with_exact_text(seeded: dict[str, Any]) -> None:
    """K3: the last clean read is 8.0 s old - ERROR halt with the exact text."""
    guard = _RecordGuard()
    node = _node_with_guard(seeded, guard)
    rig = _rig(node, "FE")
    try:
        rig.read()
        rig.read()
        rig.check(7.0)
        assert guard.calls == [("halt", HaltReason.ERROR)]
        assert guard.messages == [
            "system kill switch unreadable for 8.0 s (OperationalError); failing closed"
        ]
    finally:
        await node._shutdown()


async def test_halt_once_per_failure_streak(seeded: dict[str, Any]) -> None:
    """K4: stale for five more watcher ticks - still exactly one halt (the latch
    is set before the halt, so the streak never repeats it)."""
    guard = _RecordGuard()
    node = _node_with_guard(seeded, guard)
    rig = _rig(node, "FE")
    try:
        rig.read()
        rig.read()
        rig.check(7.0)
        for _ in range(5):
            rig.check(0.5)
        assert guard.calls == [("halt", HaltReason.ERROR)]
        assert len(guard.messages) == 1
    finally:
        await node._shutdown()


async def test_success_resets_counter_and_latch(seeded: dict[str, Any]) -> None:
    """K5: a clean read after the stale halt clears the failure state and the
    latch (the node stays halted; only an operator resume lifts it)."""
    guard = _RecordGuard()
    node = _node_with_guard(seeded, guard)
    rig = _rig(node, "FEEF")
    try:
        rig.read()
        rig.read()
        rig.read()
        rig.check(6.0)
        assert node._kill_read_halted is True
        assert node._kill_monitor.snapshot().consecutive_failures == 2
        rig.read()
        rig.check()
        snap = node._kill_monitor.snapshot()
        assert (snap.consecutive_failures, snap.error_class) == (0, None)
        assert node._kill_read_halted is False
        assert guard.calls == [("halt", HaltReason.ERROR)]
    finally:
        await node._shutdown()


# --------------------------------------------------------------- K6/K7


async def test_recovery_needs_operator_resume(seeded: dict[str, Any]) -> None:
    """K6: after the fail-closed ERROR halt a clean read does NOT auto-resume;
    the guard stays HALTED/ERROR until an operator resumes (which then works)."""
    guard = _real_guard()
    node = _node_with_guard(seeded, guard)
    rig = _rig(node, "FEF")
    try:
        rig.read()
        rig.read()
        rig.check(7.0)
        assert guard.guard_state == GuardState.HALTED
        assert guard.halt_reason == HaltReason.ERROR
        rig.read()  # clean read: recovery alone resumes nothing
        rig.check()
        assert guard.guard_state == GuardState.HALTED
        assert guard.halt_reason == HaltReason.ERROR
        ok, _msg = node.resume()
        assert ok
        assert guard.guard_state == GuardState.RUNNING
    finally:
        await node._shutdown()


async def test_resume_refused_while_unreadable(seeded: dict[str, Any]) -> None:
    """K7: resume while the snapshot carries a failed read is refused with the
    exact (cached, fresh-based) message and never reaches the guard."""
    guard = _RecordGuard()
    node = _node_with_guard(seeded, guard)
    rig = _rig(node, "FE")
    try:
        rig.read()
        rig.read()
        ok, msg = node.resume()
        assert ok is False
        assert (
            msg == "system kill switch state not fresh (OperationalError); cannot resume until it"
            " reads cleanly"
        )
        assert guard.calls == []
    finally:
        await node._shutdown()


# --------------------------------------------------------------- K8


async def test_startup_refuses_when_unreadable(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """K8: every read failing at _initialize refuses to start with the failing
    class, before any TradingNode is built (the wait is shortened for the test)."""
    monkeypatch.setattr("vibe_quant.paper.node.KILL_SWITCH_STARTUP_TIMEOUT_S", 0.3)
    node = PaperTradingNode(_config(seeded))
    script = _ScriptedReads("E")
    node._kill_monitor = KillSwitchMonitor(lambda: script, poll_s=0.02)
    try:
        with pytest.raises(ConfigurationError) as exc_info:
            await node._initialize()
        assert (
            str(exc_info.value)
            == "System kill switch state unreadable (OperationalError); refusing to start"
        )
        assert fake_tradingnode.instances == []
    finally:
        await node._shutdown()
    assert script.closed == 1, "shutdown stops the monitor and the reader thread closes its handle"


# --------------------------------------------------------------- K10


async def test_control_loop_does_not_poll_the_kill_switch(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """K10: the kill check moved to its own watcher task, so a control-loop
    iteration (commands + connectivity) never calls it."""
    node = _node_with_guard(seeded, _RecordGuard())
    monkeypatch.setattr(node._config, "control_poll_interval", 0.0)
    calls: list[str] = []
    monkeypatch.setattr(node, "_check_kill_switch", lambda: calls.append("kill"))

    iterations = 0

    def connectivity() -> None:
        nonlocal iterations
        iterations += 1
        if iterations >= 2:
            node._shutdown_event.set()

    monkeypatch.setattr(node, "_check_connectivity", connectivity)
    try:
        await asyncio.wait_for(node._control_loop(), 5)
        assert iterations == 2
        assert calls == []
    finally:
        await node._shutdown()


async def test_kill_check_runs_when_commands_fail(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """K10: ``process_commands`` raising every iteration must not skip the
    kill-switch poll: the watcher task halts KILL_SWITCH on its FIRST iteration
    while the control loop keeps logging one class-only error per failing step."""
    guard = _RecordGuard()
    node = _node_with_guard(seeded, guard)
    rig = _rig(node, "T")
    rig.read()
    monkeypatch.setattr(node._config, "control_poll_interval", 0.0)
    monkeypatch.setattr("vibe_quant.paper.node.KILL_SWITCH_WATCH_TICK_S", 0.0)
    monkeypatch.setattr(node, "_check_connectivity", lambda: None)
    commands = 0

    async def failing_commands() -> None:
        nonlocal commands
        commands += 1
        raise OSError("disk full")

    monkeypatch.setattr(node, "process_commands", failing_commands)
    try:
        with caplog.at_level(logging.ERROR):
            control = asyncio.create_task(node._control_loop())
            watch = asyncio.create_task(node._kill_watch_loop())
            for _ in range(200):
                if guard.calls:
                    break
                await asyncio.sleep(0)
            node._shutdown_event.set()
            await asyncio.wait_for(asyncio.gather(control, watch), 5)
        assert guard.calls == [("halt", HaltReason.KILL_SWITCH)]
        assert commands >= 1
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert errors
        for record in errors:
            assert record.getMessage() == "paper control loop iteration failed (OSError)"
            assert record.exc_info is None
    finally:
        await node._shutdown()


async def test_watch_loop_survives_a_raising_check(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """K10: the watcher has its own try - a raising check is logged by class
    only and the next tick still runs."""
    node = _node_with_guard(seeded, _RecordGuard())
    monkeypatch.setattr("vibe_quant.paper.node.KILL_SWITCH_WATCH_TICK_S", 0.0)
    ticks = 0

    def check() -> None:
        nonlocal ticks
        ticks += 1
        if ticks == 1:
            try:
                raise RuntimeError(f"fetch failed {_CTX_LEAK}")
            except RuntimeError:
                raise OSError(f"disk {_LEAK}")  # noqa: B904  # keep __context__
        node._shutdown_event.set()

    monkeypatch.setattr(node, "_check_kill_switch", check)
    try:
        with caplog.at_level(logging.ERROR):
            await asyncio.wait_for(node._kill_watch_loop(), 5)
        assert ticks == 2
        assert _LEAK not in caplog.text
        assert _CTX_LEAK not in caplog.text
        assert "Traceback" not in caplog.text
        assert "OSError" in caplog.text
    finally:
        await node._shutdown()


# --------------------------------------------------------------- K11/K12


async def test_engaged_read_after_failures_halts_kill(seeded: dict[str, Any]) -> None:
    """K11: failed reads then an engaged read go down the engaged path
    (KILL_SWITCH) and reset the failure count."""
    guard = _RecordGuard()
    node = _node_with_guard(seeded, guard)
    rig = _rig(node, "FEET")
    try:
        for _ in range(4):
            rig.read()
            rig.check()
        assert rig.script.reads == ["F", "E", "E", "T"]
        assert guard.calls == [("halt", HaltReason.KILL_SWITCH)]
        assert node._kill_monitor.snapshot().consecutive_failures == 0
    finally:
        await node._shutdown()


async def test_engaged_switch_escalates_fail_closed_halt(seeded: dict[str, Any]) -> None:
    """K12: a stale ERROR halt, then an engaged read - the REAL guard replaces
    the fail-closed ERROR halt with KILL_SWITCH (priority 6 > 2)."""
    guard = _real_guard()
    node = _node_with_guard(seeded, guard)
    rig = _rig(node, "FET")
    try:
        rig.read()
        rig.read()
        rig.check(7.0)
        assert guard.guard_state == GuardState.HALTED
        assert guard.halt_reason == HaltReason.ERROR
        rig.read()
        rig.check()  # engaged switch escalates past the ERROR halt
        assert guard.guard_state == GuardState.HALTED
        assert guard.halt_reason == HaltReason.KILL_SWITCH
    finally:
        await node._shutdown()


# --------------------------------------------------------------- K16


async def test_state_parse_error_counts_as_failure(
    seeded: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """K16: a raising state parse (``state.get``) is a failed read - recorded and
    logged like a read error, never raised out of the monitor, and the previous
    clean view stays."""
    guard = _RecordGuard()
    node = _node_with_guard(seeded, guard)
    rig = _rig(node, "FP")
    try:
        rig.read()
        with caplog.at_level(logging.ERROR):
            rig.read()
        snap = node._kill_monitor.snapshot()
        assert (snap.error_class, snap.consecutive_failures) == ("RuntimeError", 1)
        assert snap.engaged is False
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert (
            errors[0].getMessage()
            == "could not read system kill switch (RuntimeError); 1 consecutive failure(s)"
        )
    finally:
        await node._shutdown()


# --------------------------------------------------------------- M4


async def test_second_streak_re_halts_after_operator_resume(seeded: dict[str, Any]) -> None:
    """M4: stale -> ERROR halt; the clean read releases the per-streak latch; an
    operator resume returns the real guard to RUNNING; a second stale streak must
    halt ERROR again. A success path that stopped resetting ``_kill_read_halted``
    would leave the node silently failed-open here."""
    guard = _real_guard()
    node = _node_with_guard(seeded, guard)
    rig = _rig(node, "FEFE")
    try:
        rig.read()
        rig.read()
        rig.check(7.0)
        assert guard.guard_state == GuardState.HALTED
        assert guard.halt_reason == HaltReason.ERROR
        assert node._kill_read_halted is True

        rig.read()  # clean read
        rig.check()  # releases the latch
        assert node._kill_read_halted is False

        ok, _msg = node.resume()
        assert ok
        assert guard.guard_state == GuardState.RUNNING
        assert guard.halt_reason is None

        rig.read()  # E: the last clean read is now 1 s old
        rig.check(6.5)
        assert guard.guard_state == GuardState.RUNNING
        rig.check(0.5)  # 8.0 s since the clean read
        assert rig.script.reads == ["F", "E", "F", "E"]
        assert guard.guard_state == GuardState.HALTED
        assert guard.halt_reason == HaltReason.ERROR
        assert node._kill_read_halted is True
    finally:
        await node._shutdown()


# --------------------------------------------------------------- F2


def test_reader_asks_for_the_row_without_a_kwarg(tmp_path: Path) -> None:
    """F2/SF2 completion: ``get_system_state`` has no ``require_row`` kwarg any
    more (a missing row is re-created engaged by T1, so the flag meant nothing)
    and the monitor reads a real state manager through the plain call."""
    import inspect

    assert list(inspect.signature(StateManager.get_system_state).parameters) == ["self"]
    sm = StateManager(tmp_path / "state.db")
    try:
        monitor = KillSwitchMonitor(lambda: sm)
        monitor.poll_once(sm)
        snap = monitor.snapshot()
    finally:
        sm.close()
    assert (snap.engaged, snap.error_class, snap.last_ok_mono is not None) == (False, None, True)
