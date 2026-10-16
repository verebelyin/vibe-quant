"""Kill switch fail-closed reads (bead vibe-quant-s40h9, task K1).

A raising kill-switch state read counts as engaged for that check and as a
consecutive read failure: three consecutive failures halt the node with reason
ERROR, startup and resume refuse while the state is unreadable, and the control
loop runs each step in its own try so a failing command claim (same state DB)
can never skip the kill-switch poll.

Contract rows (brief cc-sz-kill; K16/K17 from the chief's post-review rulings):

* K1  E        -> no halt; one ERROR log ``could not read system kill switch
                  (OperationalError); 1 consecutive failure(s)``, no exc_info
* K2  E,E      -> no halt (the limit is 3, counted per read)
* K3  E,E,E    -> ``guard.calls == [("halt", ERROR)]`` with message
                  ``system kill switch unreadable for 3 consecutive reads
                  (OperationalError); failing closed``
* K4  E x5     -> still one halt call (the latch is set before the halt)
* K5  E,E,F,E,E -> no halt (a clean read resets the counter)
* K6  E,E,E,F  -> the guard stays HALTED/ERROR (no auto-resume), then an
                  operator resume succeeds
* K7  resume while the last read failed -> refused with the exact message and
                  ``guard.resume`` is never called
* K8  E at _initialize -> ``ConfigurationError`` with the exact message and no
                  TradingNode built
* K9  log hygiene -> the exception class is logged; the message text, the
                  ``__context__`` chain and any traceback are not
* K10 control loop -> a ``process_commands`` failure must not skip the kill
                  check (its own try): ``halt(KILL_SWITCH)`` on the FIRST
                  iteration, one ``paper control loop iteration failed
                  (OSError)`` per failing step
* K11 E,E,T    -> ``halt(KILL_SWITCH)`` at the third read, counter reset
* K12 E,E,E (ERROR halt) then T -> the engaged switch escalates: the real guard
                  replaces ERROR with KILL_SWITCH (priority 6 > 2)
* K15 ``KILL_SWITCH_READ_FAILURE_LIMIT == 3``; failures are counted per read
* K16 a raising state parse (``sys_state.get``) counts as a failed read
* M4  E,E,E -> ERROR halt, a clean read releases the latch, an operator
                  resume, then E,E,E again -> a SECOND ERROR halt
* M5  ``_kill_switch`` on a failed read returns exactly ``(True, "kill switch
                  state unreadable (<Class>)")`` - never a bare ``False``
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
from vibe_quant.paper.node import KILL_SWITCH_READ_FAILURE_LIMIT, PaperTradingNode

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
    """``get_system_state`` following a script of outcomes: E raises, F is clean
    and not engaged, T is clean and engaged, P returns an unparseable state.

    When the script runs out, its last outcome repeats. ``reads`` records one
    entry per call, so a test can pin that each check consumes exactly one read
    (failures are counted per read, not per second)."""

    def __init__(self, outcomes: str) -> None:
        self._outcomes = list(outcomes)
        self.reads: list[str] = []

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


def _patch_reads(
    monkeypatch: pytest.MonkeyPatch, node: PaperTradingNode, outcomes: str
) -> _ScriptedReads:
    script = _ScriptedReads(outcomes)
    monkeypatch.setattr(node._state_manager, "get_system_state", script.get_system_state)
    return script


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


# --------------------------------------------------------------- K15


def test_limit_constant() -> None:
    """K15: three consecutive read failures before the node fails closed; the
    cadence rationale (worst case ~33 s: 3 iterations x (1 s poll + up to 5 s
    busy_timeout on claim_pending + up to 5 s on the kill-switch read)) lives in
    the constant's comment."""
    assert KILL_SWITCH_READ_FAILURE_LIMIT == 3


# --------------------------------------------------------------- K1/K2/K4/K5


async def test_single_failure_tolerated(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """K1: one failed read - no halt, one class-only ERROR log, no exc_info."""
    guard = _RecordGuard()
    node = _node_with_guard(seeded, guard)
    script = _patch_reads(monkeypatch, node, "E")
    try:
        with caplog.at_level(logging.ERROR):
            node._check_kill_switch()
        assert script.reads == ["E"]
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


async def test_two_failures_tolerated(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """K2: two consecutive failures - still no halt (the limit is 3, counted per
    read: two calls must consume exactly two reads)."""
    guard = _RecordGuard()
    node = _node_with_guard(seeded, guard)
    script = _patch_reads(monkeypatch, node, "E")
    try:
        node._check_kill_switch()
        node._check_kill_switch()
        assert script.reads == ["E", "E"]
        assert guard.calls == []
        assert node._kill_read_failures == 2
    finally:
        await node._shutdown()


async def test_third_consecutive_failure_halts_error(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """K3: the third consecutive failure halts with reason ERROR and the exact
    fail-closed message."""
    guard = _RecordGuard()
    node = _node_with_guard(seeded, guard)
    script = _patch_reads(monkeypatch, node, "E")
    try:
        for _ in range(3):
            node._check_kill_switch()
        assert script.reads == ["E", "E", "E"]
        assert guard.calls == [("halt", HaltReason.ERROR)]
        assert guard.messages == [
            "system kill switch unreadable for 3 consecutive reads (OperationalError); failing closed"
        ]
    finally:
        await node._shutdown()


async def test_halt_once_per_failure_streak(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """K4: five consecutive failures - still exactly one halt (the latch is set
    before the halt, so the streak never repeats it)."""
    guard = _RecordGuard()
    node = _node_with_guard(seeded, guard)
    script = _patch_reads(monkeypatch, node, "E")
    try:
        for _ in range(5):
            node._check_kill_switch()
        assert script.reads == ["E"] * 5
        assert guard.calls == [("halt", HaltReason.ERROR)]
        assert len(guard.messages) == 1
    finally:
        await node._shutdown()


async def test_success_resets_counter(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """K5: E,E,F,E,E - the clean read resets the counter, so the streak never
    reaches the limit and nothing halts."""
    guard = _RecordGuard()
    node = _node_with_guard(seeded, guard)
    script = _patch_reads(monkeypatch, node, "EEFEE")
    try:
        for _ in range(5):
            node._check_kill_switch()
        assert script.reads == ["E", "E", "F", "E", "E"]
        assert guard.calls == []
    finally:
        await node._shutdown()


# --------------------------------------------------------------- K6/K7


async def test_recovery_needs_operator_resume(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """K6: after the fail-closed ERROR halt a clean read does NOT auto-resume;
    the guard stays HALTED/ERROR until an operator resumes (which then works)."""
    guard = _real_guard()
    node = _node_with_guard(seeded, guard)
    _patch_reads(monkeypatch, node, "EEEF")
    try:
        node._check_kill_switch()
        node._check_kill_switch()
        node._check_kill_switch()
        assert guard.guard_state == GuardState.HALTED
        assert guard.halt_reason == HaltReason.ERROR
        node._check_kill_switch()  # clean read: recovery alone resumes nothing
        assert guard.guard_state == GuardState.HALTED
        assert guard.halt_reason == HaltReason.ERROR
        ok, _msg = node.resume()
        assert ok
        assert guard.guard_state == GuardState.RUNNING
    finally:
        await node._shutdown()


async def test_resume_refused_while_unreadable(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """K7: resume while the last read failed is refused with the exact message
    and never reaches the guard."""
    guard = _RecordGuard()
    node = _node_with_guard(seeded, guard)
    _patch_reads(monkeypatch, node, "E")
    try:
        node._check_kill_switch()
        ok, msg = node.resume()
        assert ok is False
        assert (
            msg
            == "system kill switch state unreadable (OperationalError); cannot resume until it"
            " reads cleanly"
        )
        assert guard.calls == []
        assert ("resume", False) not in guard.calls
    finally:
        await node._shutdown()


# --------------------------------------------------------------- K8


async def test_startup_refuses_when_unreadable(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """K8: a failed read at _initialize refuses to start with the exact message,
    before any TradingNode is built."""
    node = PaperTradingNode(_config(seeded))
    _patch_reads(monkeypatch, node, "E")
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


# --------------------------------------------------------------- K9


async def test_failure_log_is_class_only(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """K9: the failure log carries the exception class only - the error text
    (``ZZLEAK9``), the ``__context__`` chain's text (``QQLEAK7``) and any
    traceback stay out of the log."""
    guard = _RecordGuard()
    node = _node_with_guard(seeded, guard)
    _patch_reads(monkeypatch, node, "E")
    try:
        with caplog.at_level(logging.ERROR):
            node._check_kill_switch()
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert errors[0].getMessage() == (
            "could not read system kill switch (OperationalError); 1 consecutive failure(s)"
        )
        assert errors[0].exc_info is None
        assert "OperationalError" in caplog.text
        assert _LEAK not in caplog.text
        assert _CTX_LEAK not in caplog.text
        assert "Traceback" not in caplog.text
    finally:
        await node._shutdown()


# --------------------------------------------------------------- K10


async def test_kill_check_runs_when_commands_fail(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """K10: ``process_commands`` raising every iteration must not skip the
    kill-switch poll (its own try): the switch is engaged and the node halts
    KILL_SWITCH on the FIRST iteration, with one control-loop error log per
    failing step."""
    guard = _RecordGuard()
    node = _node_with_guard(seeded, guard)
    _patch_reads(monkeypatch, node, "T")
    monkeypatch.setattr(node._config, "control_poll_interval", 0.0)
    snapshots: list[list[tuple[str, Any]]] = []

    async def failing_commands() -> None:
        snapshots.append(list(guard.calls))
        raise OSError("disk full")

    checks = 0

    def connectivity() -> None:
        nonlocal checks
        checks += 1
        if checks >= 2:
            node._shutdown_event.set()

    monkeypatch.setattr(node, "process_commands", failing_commands)
    monkeypatch.setattr(node, "_check_connectivity", connectivity)
    try:
        with caplog.at_level(logging.ERROR):
            await asyncio.wait_for(node._control_loop(), 5)
        assert snapshots == [[], [("halt", HaltReason.KILL_SWITCH)]], (
            "the kill check must halt on the first iteration despite the failing commands"
        )
        assert guard.calls == [("halt", HaltReason.KILL_SWITCH)]
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 2
        for record in errors:
            assert record.getMessage() == "paper control loop iteration failed (OSError)"
            assert record.exc_info is None
    finally:
        await node._shutdown()


# --------------------------------------------------------------- K11/K12


async def test_engaged_read_after_failures_halts_kill(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """K11: E,E,T - the engaged read after two failures goes down the engaged
    path (KILL_SWITCH) and resets the failure counter."""
    guard = _RecordGuard()
    node = _node_with_guard(seeded, guard)
    script = _patch_reads(monkeypatch, node, "EET")
    try:
        node._check_kill_switch()
        node._check_kill_switch()
        node._check_kill_switch()
        assert script.reads == ["E", "E", "T"]
        assert guard.calls == [("halt", HaltReason.KILL_SWITCH)]
        assert node._kill_read_failures == 0
    finally:
        await node._shutdown()


async def test_engaged_switch_escalates_fail_closed_halt(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """K12: E,E,E (ERROR halt) then T - the engaged switch still halts
    KILL_SWITCH and the REAL guard replaces the fail-closed ERROR halt
    (priority 6 > 2)."""
    guard = _real_guard()
    node = _node_with_guard(seeded, guard)
    _patch_reads(monkeypatch, node, "EEET")
    try:
        node._check_kill_switch()
        node._check_kill_switch()
        node._check_kill_switch()
        assert guard.guard_state == GuardState.HALTED
        assert guard.halt_reason == HaltReason.ERROR
        node._check_kill_switch()  # engaged switch escalates past the ERROR halt
        assert guard.guard_state == GuardState.HALTED
        assert guard.halt_reason == HaltReason.KILL_SWITCH
    finally:
        await node._shutdown()


# --------------------------------------------------------------- K16


async def test_state_parse_error_counts_as_failure(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """K16: a raising state parse (``sys_state.get``) is a failed read - it is
    counted and logged like a read error, never raised out of the check."""
    guard = _RecordGuard()
    node = _node_with_guard(seeded, guard)
    script = _patch_reads(monkeypatch, node, "P")
    try:
        with caplog.at_level(logging.ERROR):
            node._check_kill_switch()
        assert script.reads == ["P"]
        assert node._kill_read_failures == 1
        assert guard.calls == []
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert (
            errors[0].getMessage()
            == "could not read system kill switch (RuntimeError); 1 consecutive failure(s)"
        )
    finally:
        await node._shutdown()


# --------------------------------------------------------------- M4/M5


async def test_second_streak_re_halts_after_operator_resume(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """M4: E,E,E -> ERROR halt; the clean read releases the per-streak latch; an
    operator resume returns the real guard to RUNNING; a second E,E,E streak
    must halt ERROR again. A success path that stopped resetting
    ``_kill_read_halted`` would leave the node silently failed-open here."""
    guard = _real_guard()
    node = _node_with_guard(seeded, guard)
    script = _patch_reads(monkeypatch, node, "EEEFFFEEE")
    try:
        for _ in range(3):
            node._check_kill_switch()
        assert guard.guard_state == GuardState.HALTED
        assert guard.halt_reason == HaltReason.ERROR
        assert node._kill_read_halted is True

        node._check_kill_switch()  # clean read: releases the latch
        assert node._kill_read_failures == 0

        ok, _msg = node.resume()  # consumes one more clean read
        assert ok
        assert guard.guard_state == GuardState.RUNNING
        assert guard.halt_reason is None

        node._check_kill_switch()  # clean read
        for _ in range(3):
            node._check_kill_switch()
        assert script.reads == ["E", "E", "E", "F", "F", "F", "E", "E", "E"]
        assert guard.guard_state == GuardState.HALTED
        assert guard.halt_reason == HaltReason.ERROR
        assert node._kill_read_halted is True
    finally:
        await node._shutdown()


@pytest.mark.parametrize(
    ("outcome", "cls"),
    [("E", "OperationalError"), ("P", "RuntimeError")],
)
async def test_failed_read_returns_engaged_tuple(
    seeded: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    cls: str,
) -> None:
    """M5: ``_kill_switch`` answers a failed read with exactly
    ``(True, "kill switch state unreadable (<Class>)")`` - engaged, with the
    exception class in the reason. Returning ``False`` there fails open."""
    guard = _RecordGuard()
    node = _node_with_guard(seeded, guard)
    script = _patch_reads(monkeypatch, node, outcome)
    try:
        result = node._kill_switch()
        assert script.reads == [outcome]
        assert result == (True, f"kill switch state unreadable ({cls})")
        assert result[0] is True
        assert node._kill_read_failures == 1
        assert node._kill_read_error == cls
    finally:
        await node._shutdown()
