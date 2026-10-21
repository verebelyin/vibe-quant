"""Pin the paper halt/pause/resume table for both machines (rabmo.9a, before rabmo.9).

Contract: design lane1-paper.md r2 F8, "T6 rows" (G1-G15 for ``TradingGuard``,
P1-P12 for the pre-guard ``PaperTradingNode``). Chief ruling D6: the refactor
keeps TWO halt-state instances (the guard's machine and the node's pre-guard
record), so every pre-guard difference is PINNED here, never unified:

* engaged is checked before "already running": pre-guard RUNNING + engaged
  refuses with the kill-switch text, the guard answers "already running";
* ``"<reason> halt cannot be resumed by the operator"`` (pre-guard) vs
  ``"<reason> halt requires manual review; restart the session to trade again"``
  (guard);
* the pre-guard resume has no MAX_DAILY_LOSS branch (that halt is not resumable
  before the guard exists);
* a pre-guard resume of a restored halt is overridden by the guard at build (P12).

Characterization only: no production code change. The severity table and every
expected string are literal here, so a refactor cannot silently re-derive them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_audit_paper_guard import RecordingListener
from tests.unit.test_paper_guard_prestart_halt import _unit_guard
from tests.unit.test_paper_halt_restore import (
    _ISO,
    _PRIOR_MANUAL,
    _RESTORED_MESSAGE,
    _Trace,
    _write_checkpoint,
    fake_tradingnode,  # noqa: F401
    seeded,  # noqa: F401
)
from tests.unit.test_paper_startup_halt import _arm, _drive, _prepare
from vibe_quant.logging.events import EventType
from vibe_quant.paper.config import BinanceTestnetConfig, PaperTradingConfig
from vibe_quant.paper.guard import (
    RESTORED_HALT_PREFIX,
    GuardState,
    HaltReason,
    RestoredHalt,
    RiskState,
)
from vibe_quant.paper.kill_switch_monitor import KillSwitchSnapshot
from vibe_quant.paper.node import NodeState, PaperTradingNode

if TYPE_CHECKING:
    from pathlib import Path

_MSG = "m"
_PRIOR = "prior"
_ENGAGED_TEXT = "system kill switch is engaged; unlock it before resuming"
_ALREADY_RUNNING = "already running"
_RESUMED_FROM_PAUSE = "resumed from pause"
_RESUMED_FROM_HALT = "resumed from halt"
_GUARD_CHANGED = "{reason} halt requires manual review; restart the session to trade again"
_NODE_CHANGED = "{reason} halt cannot be resumed by the operator"
_STARTUP_HALT_WARNING = (
    "paper PAPER-007: {reason} halt during start-up recorded; "
    "the trading guard applies it when built"
)

#: Literal severity order (independent of ``guard._HALT_PRIORITY``): the table has
#: teeth against a changed comparison, not just against a renamed constant.
_PRIORITY: dict[HaltReason, int] = {
    HaltReason.MAX_DAILY_LOSS: 1,
    HaltReason.ERROR: 2,
    HaltReason.MANUAL: 3,
    HaltReason.SIGNAL: 3,
    HaltReason.MAX_CONSECUTIVE_LOSSES: 4,
    HaltReason.MAX_DRAWDOWN: 5,
    HaltReason.KILL_SWITCH: 6,
}
_REASONS: list[HaltReason] = list(HaltReason)
_OPERATOR_RESUMABLE = frozenset(
    {HaltReason.MANUAL, HaltReason.SIGNAL, HaltReason.ERROR, HaltReason.KILL_SWITCH}
)
_MANUAL_REVIEW = frozenset({HaltReason.MAX_DRAWDOWN, HaltReason.MAX_CONSECUTIVE_LOSSES})


def _supersedes(new: HaltReason, current: HaltReason | None) -> bool:
    return current is None or _PRIORITY[new] > _PRIORITY[current]


# ------------------------------------------------------------------ first test


def test_first_guard_manual_over_signal_returns_false() -> None:
    """G3: an equal-severity halt never replaces the active one (MANUAL over SIGNAL)."""
    guard, listener = _guard_in(GuardState.HALTED, HaltReason.SIGNAL)

    assert guard.halt(HaltReason.MANUAL, _MSG) is False

    assert guard.guard_state is GuardState.HALTED
    assert guard.halt_reason is HaltReason.SIGNAL
    assert guard.message == _PRIOR
    assert listener.changes == [], "a refused halt notifies nobody"


# ------------------------------------------------------- (a) TradingGuard table


def _guard_in(
    state: GuardState,
    reason: HaltReason | None = None,
    prior: GuardState = GuardState.RUNNING,
) -> tuple[Any, RecordingListener]:
    """The ``_unit_guard`` subject in ``state`` with a recording listener."""
    guard = _unit_guard([])
    listener = RecordingListener()
    guard._listener = listener
    if state is GuardState.PAUSED or prior is GuardState.PAUSED:
        assert guard.pause() is True
    if state is GuardState.HALTED:
        assert guard.halt(reason, _PRIOR) is True
    listener.changes.clear()
    return guard, listener


def _changes(listener: RecordingListener) -> list[tuple[GuardState, HaltReason | None, str]]:
    return [(state, reason, message) for _ns, state, reason, message in listener.changes]


@dataclass(frozen=True)
class _GuardStart:
    label: str
    state: GuardState
    reason: HaltReason | None = None


_GUARD_HALT_STARTS: list[_GuardStart] = [
    _GuardStart("running", GuardState.RUNNING),
    _GuardStart("paused", GuardState.PAUSED),
    *[_GuardStart(f"halted-{r.value}", GuardState.HALTED, r) for r in _REASONS],
]


@pytest.mark.parametrize("reason", _REASONS, ids=lambda r: r.value)
@pytest.mark.parametrize("start", _GUARD_HALT_STARTS, ids=lambda s: s.label)
def test_table_guard_halt(reason: HaltReason, start: _GuardStart) -> None:
    """G1-G6: halt applies from RUNNING/PAUSED, and replaces a HALTED reason only
    when strictly more severe (MANUAL and SIGNAL are both priority 3)."""
    guard, listener = _guard_in(start.state, start.reason)
    applied = _supersedes(reason, start.reason) if start.state is GuardState.HALTED else True

    assert guard.halt(reason, _MSG) is applied

    if not applied:
        assert guard.guard_state is GuardState.HALTED
        assert guard.halt_reason is start.reason
        assert guard.message == _PRIOR
        assert guard._state_before_halt is GuardState.RUNNING
        assert _changes(listener) == []
        assert guard.status()["state"] == "halted"
        assert guard.status()["halt_reason"] == start.reason.value
        assert guard.status()["message"] == _PRIOR
        return
    assert guard.guard_state is GuardState.HALTED
    assert guard.halt_reason is reason
    assert guard.message == _MSG
    expected_before = start.state if start.state is not GuardState.HALTED else GuardState.RUNNING
    assert guard._state_before_halt is expected_before
    assert _changes(listener) == [(GuardState.HALTED, reason, _MSG)]
    assert guard.status()["state"] == "halted"
    assert guard.status()["halt_reason"] == reason.value
    assert guard.status()["message"] == _MSG
    assert guard.status()["gate"] == "reduce_only"


@pytest.mark.parametrize("start", _GUARD_HALT_STARTS, ids=lambda s: s.label)
def test_table_guard_pause(start: _GuardStart) -> None:
    """G7: pause is a no-op unless RUNNING; its default message is pinned."""
    guard, listener = _guard_in(start.state, start.reason)

    applied = guard.pause()

    if start.state is not GuardState.RUNNING:
        assert applied is False
        assert guard.guard_state is start.state
        assert guard.halt_reason is start.reason
        expected = _PRIOR if start.state is GuardState.HALTED else "paused by operator"
        assert guard.message == expected
        assert _changes(listener) == []
        return
    assert applied is True
    assert guard.guard_state is GuardState.PAUSED
    assert guard.halt_reason is None
    assert guard.message == "paused by operator"
    assert _changes(listener) == [(GuardState.PAUSED, None, "paused by operator")]


@dataclass(frozen=True)
class _GuardResumeStart:
    label: str
    state: GuardState
    reason: HaltReason | None = None
    prior: GuardState = GuardState.RUNNING


_GUARD_RESUME_STARTS: list[_GuardResumeStart] = [
    _GuardResumeStart("running", GuardState.RUNNING),
    _GuardResumeStart("paused", GuardState.PAUSED),
    *[
        _GuardResumeStart(f"halted-{r.value}-from-{p.value}", GuardState.HALTED, r, p)
        for r in _REASONS
        for p in (GuardState.RUNNING, GuardState.PAUSED)
    ],
]


@pytest.mark.parametrize("engaged", [False, True], ids=["disengaged", "engaged"])
@pytest.mark.parametrize("start", _GUARD_RESUME_STARTS, ids=lambda s: s.label)
def test_table_guard_resume(start: _GuardResumeStart, engaged: bool) -> None:
    """G8-G13: RUNNING wins over the kill switch; engaged refuses everything else;
    daily loss lifts by itself; drawdown/streak need a restart; the rest resume."""
    guard, listener = _guard_in(start.state, start.reason, start.prior)

    ok, text = guard.resume(kill_switch_engaged=engaged)

    if start.state is GuardState.RUNNING:
        assert (ok, text) == (True, _ALREADY_RUNNING)
        assert guard.guard_state is GuardState.RUNNING
        assert _changes(listener) == []
        return
    if engaged:
        assert (ok, text) == (False, _ENGAGED_TEXT)
        assert guard.guard_state is start.state
        assert guard.halt_reason is start.reason
        assert _changes(listener) == []
        return
    if start.state is GuardState.PAUSED:
        assert (ok, text) == (True, _RESUMED_FROM_PAUSE)
        assert guard.guard_state is GuardState.RUNNING
        assert guard.halt_reason is None
        assert guard.message == "resumed"
        assert _changes(listener) == [(GuardState.RUNNING, None, "resumed")]
        return
    assert start.reason is not None
    if start.reason is HaltReason.MAX_DAILY_LOSS:
        assert (ok, text) == (
            False,
            "daily-loss halt lifts automatically at the next UTC day",
        )
        assert guard.guard_state is GuardState.HALTED
        assert guard.halt_reason is HaltReason.MAX_DAILY_LOSS
        assert _changes(listener) == []
        return
    if start.reason in _MANUAL_REVIEW:
        assert (ok, text) == (False, _GUARD_CHANGED.format(reason=start.reason.value))
        assert guard.guard_state is GuardState.HALTED
        assert guard.halt_reason is start.reason
        assert _changes(listener) == []
        return
    assert start.reason in _OPERATOR_RESUMABLE
    assert (ok, text) == (True, _RESUMED_FROM_HALT)
    assert guard.guard_state is start.prior
    assert guard.halt_reason is None
    assert guard.message == "resumed by operator"
    assert guard._state_before_halt is GuardState.RUNNING
    assert _changes(listener) == [(start.prior, None, "resumed by operator")]


@pytest.mark.parametrize("reason", _REASONS, ids=lambda r: r.value)
def test_table_guard_daily_rollover(reason: HaltReason) -> None:
    """G14: a UTC-day rollover lifts ONLY a MAX_DAILY_LOSS halt."""
    risk = RiskState(
        high_water_mark=Decimal("10000"),
        day=date(2026, 1, 4),
        day_start_equity=Decimal("10000"),
    )
    guard = _unit_guard([], risk_state=risk)
    listener = RecordingListener()
    guard._listener = listener
    assert guard.halt(reason, _PRIOR) is True
    listener.changes.clear()

    guard.evaluate_risk()

    if reason is HaltReason.MAX_DAILY_LOSS:
        assert guard.guard_state is GuardState.RUNNING
        assert guard.halt_reason is None
        assert guard.message == "UTC day rollover to 2026-01-05"
        assert _changes(listener) == [(GuardState.RUNNING, None, "UTC day rollover to 2026-01-05")]
        return
    assert guard.guard_state is GuardState.HALTED
    assert guard.halt_reason is reason
    assert guard.message == _PRIOR
    assert _changes(listener) == []


@pytest.mark.parametrize("reason", _REASONS, ids=lambda r: r.value)
def test_table_guard_restored_halt_at_construction(reason: HaltReason) -> None:
    """G15: a restored halt comes up HALTED with the prefixed message."""
    restored = RestoredHalt(reason, f"{RESTORED_HALT_PREFIX}halt via API", _ISO)
    guard = _unit_guard([], restored_halt=restored)
    listener = RecordingListener()
    guard._listener = listener

    assert guard.guard_state is GuardState.HALTED
    assert guard.halt_reason is reason
    assert guard.message == "restored after restart: halt via API"
    assert guard._state_before_halt is GuardState.RUNNING
    assert guard.status()["state"] == "halted"
    assert guard.status()["halt_reason"] == reason.value
    assert listener.changes == []


# --------------------------------------------------- (b) pre-guard node table


def _node(tmp_path: Path) -> PaperTradingNode:
    return PaperTradingNode(
        PaperTradingConfig(
            trader_id="PAPER-007",
            binance=BinanceTestnetConfig("key", "secret", testnet=True),
            symbols=["BTCUSDT"],
            strategy_id=1,
            db_path=tmp_path / "state.db",
        )
    )


def _traced_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[PaperTradingNode, _Trace]:
    node = _node(tmp_path)
    trace = _Trace()
    monkeypatch.setattr(node, "_write_event", trace.write)
    monkeypatch.setattr(node, "_write_then_alert", trace.write_then_alert)
    monkeypatch.setattr(node, "_send_telegram_alert", trace.alert)
    return node, trace


def _set_node_state(
    node: PaperTradingNode,
    state: NodeState,
    reason: HaltReason | None = None,
    prior: NodeState = NodeState.RUNNING,
) -> None:
    if state is NodeState.INITIALIZING:
        return
    node._status.state = NodeState.RUNNING
    if state is NodeState.RUNNING:
        return
    if state is NodeState.PAUSED or prior is NodeState.PAUSED:
        node.pause()
        assert node.status.state is NodeState.PAUSED
    if state is NodeState.HALTED:
        node.halt(reason, _PRIOR)
        assert node.status.state is NodeState.HALTED


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "vibe_quant.paper.node" and record.levelno == logging.WARNING
    ]


def _node_status(node: PaperTradingNode) -> tuple[NodeState, HaltReason | None, str | None]:
    return (node.status.state, node.status.halt_reason, node.status.error_message)


def _fresh_kill_snapshot(engaged: bool = False) -> KillSwitchSnapshot:
    return KillSwitchSnapshot(
        engaged=engaged,
        reason="ops" if engaged else None,
        last_ok_mono=0.0,
        error_class=None,
        consecutive_failures=0,
    )


@dataclass(frozen=True)
class _NodeStart:
    label: str
    state: NodeState
    reason: HaltReason | None = None
    previous: NodeState | None = None


_NODE_HALT_STARTS: list[_NodeStart] = [
    _NodeStart("initializing", NodeState.INITIALIZING),
    _NodeStart("running", NodeState.RUNNING),
    _NodeStart("paused", NodeState.PAUSED),
    *[_NodeStart(f"halted-{r.value}", NodeState.HALTED, r, NodeState.RUNNING) for r in _REASONS],
]


@pytest.mark.parametrize("reason", _REASONS, ids=lambda r: r.value)
@pytest.mark.parametrize("start", _NODE_HALT_STARTS, ids=lambda s: s.label)
def test_table_node_halt(
    reason: HaltReason,
    start: _NodeStart,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """P1-P4: the pre-guard halt records severity, status, the replayed message and
    the previous state; it writes NO event and NO alert before the guard exists."""
    node, trace = _traced_node(tmp_path, monkeypatch)
    _set_node_state(node, start.state, start.reason)
    trace.calls.clear()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="vibe_quant.paper.node"):
        result = node.halt(reason, _MSG)

    assert result is None
    applied = _supersedes(reason, start.reason) if start.state is NodeState.HALTED else True
    expected_previous = {
        NodeState.INITIALIZING: None,
        NodeState.RUNNING: NodeState.RUNNING,
        NodeState.PAUSED: NodeState.PAUSED,
        NodeState.HALTED: NodeState.RUNNING,
    }[start.state]
    assert node._previous_state is expected_previous
    if not applied:
        assert _node_status(node) == (NodeState.HALTED, start.reason, _PRIOR)
        assert node._startup_halt == (start.reason, _PRIOR)
        assert trace.calls == []
        assert _warnings(caplog) == []
        return
    assert _node_status(node) == (NodeState.HALTED, reason, _MSG)
    assert node._startup_halt == (reason, _MSG)
    assert node._state_payload() == {
        "state": "halted",
        "halt_reason": reason.value,
        "message": _MSG,
    }
    assert trace.calls == [], "no event and no alert before the guard exists"
    assert _warnings(caplog) == [_STARTUP_HALT_WARNING.format(reason=reason.value)]


@pytest.mark.parametrize("reason", _REASONS, ids=lambda r: r.value)
def test_table_node_halt_defaults_the_message_to_the_reason(
    reason: HaltReason, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1: ``halt(reason)`` without a message records the reason's own text."""
    node, _trace = _traced_node(tmp_path, monkeypatch)

    node.halt(reason)

    assert node.status.error_message == reason.value
    assert node._startup_halt == (reason, reason.value)


@pytest.mark.parametrize("start", _NODE_HALT_STARTS, ids=lambda s: s.label)
def test_table_node_pause(
    start: _NodeStart, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P5: only a RUNNING node pauses, and only that one write is emitted."""
    node, trace = _traced_node(tmp_path, monkeypatch)
    _set_node_state(node, start.state, start.reason)
    trace.calls.clear()
    before = _node_status(node)

    result = node.pause()

    assert result is None
    if start.state is not NodeState.RUNNING:
        assert _node_status(node) == before
        assert trace.calls == []
        return
    assert _node_status(node) == (NodeState.PAUSED, None, None)
    assert trace.calls == [
        (
            "write",
            EventType.SIGNAL,
            {
                "action": "state_change",
                "from_state": "running",
                "to_state": "paused",
            },
            None,
        )
    ]


@dataclass(frozen=True)
class _NodeResumeStart:
    label: str
    state: NodeState
    reason: HaltReason | None = None
    previous: NodeState | None = None


_NODE_RESUME_STARTS: list[_NodeResumeStart] = [
    _NodeResumeStart("initializing", NodeState.INITIALIZING),
    _NodeResumeStart("running", NodeState.RUNNING),
    _NodeResumeStart("paused", NodeState.PAUSED),
    *[
        _NodeResumeStart(
            f"halted-{r.value}-from-{previous.value}",
            NodeState.HALTED,
            r,
            previous,
        )
        for r in _REASONS
        for previous in (NodeState.RUNNING, NodeState.PAUSED)
    ],
]


@pytest.mark.parametrize("engaged", [False, True], ids=["disengaged", "engaged"])
@pytest.mark.parametrize("start", _NODE_RESUME_STARTS, ids=lambda s: s.label)
def test_table_node_resume(
    start: _NodeResumeStart,
    engaged: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P7-P11: engaged is checked BEFORE "already running" (a pre-guard difference);
    the pre-guard refusal text differs from the guard's, and there is no
    daily-loss branch: MAX_DAILY_LOSS is simply not operator-resumable here."""
    node, trace = _traced_node(tmp_path, monkeypatch)
    _set_node_state(node, start.state, start.reason, start.previous or NodeState.RUNNING)
    trace.calls.clear()
    monkeypatch.setattr(node, "_mono", lambda: 0.0)
    monkeypatch.setattr(node._kill_monitor, "snapshot", lambda: _fresh_kill_snapshot(engaged))

    ok, text = node.resume()

    if engaged:
        assert (ok, text) == (False, _ENGAGED_TEXT)
        assert trace.calls == []
        return
    if start.state is NodeState.INITIALIZING:
        assert (ok, text) == (False, "trading node not running")
        assert trace.calls == []
        return
    if start.state is NodeState.RUNNING:
        assert (ok, text) == (True, _ALREADY_RUNNING)
        assert trace.calls == []
        return
    if start.state is NodeState.PAUSED:
        assert (ok, text) == (True, _RESUMED_FROM_PAUSE)
        assert _node_status(node) == (NodeState.RUNNING, None, None)
        assert trace.calls == [
            (
                "write",
                EventType.SIGNAL,
                {
                    "action": "state_change",
                    "from_state": "paused",
                    "to_state": "running",
                },
                None,
            )
        ]
        return
    assert start.reason is not None
    if start.reason not in _OPERATOR_RESUMABLE:
        assert (ok, text) == (False, _NODE_CHANGED.format(reason=start.reason.value))
        assert _node_status(node) == (NodeState.HALTED, start.reason, _PRIOR)
        assert trace.calls == []
        return
    target = start.previous or NodeState.RUNNING
    assert (ok, text) == (True, _RESUMED_FROM_HALT)
    assert _node_status(node) == (target, None, None)
    assert node._previous_state is None
    assert node._startup_halt is None
    assert trace.calls == [
        (
            "write",
            EventType.SIGNAL,
            {
                "action": "state_change",
                "from_state": "halted",
                "to_state": target.value,
                "resumed": True,
            },
            None,
        )
    ]


@pytest.mark.parametrize(
    ("last_ok_mono", "error_class", "now", "expected"),
    [
        (
            None,
            None,
            0.0,
            (
                False,
                "system kill switch state not fresh (no clean read); "
                "cannot resume until it reads cleanly",
            ),
        ),
        (
            0.0,
            "OperationalError",
            0.0,
            (
                False,
                "system kill switch state not fresh (OperationalError); "
                "cannot resume until it reads cleanly",
            ),
        ),
        (
            0.0,
            None,
            2.0,
            (True, "already running"),
        ),
        (
            0.0,
            None,
            3.0,
            (
                False,
                "system kill switch state not fresh (3.0 s old); "
                "cannot resume until it reads cleanly",
            ),
        ),
    ],
    ids=["no-read", "error-class", "two-seconds-exactly", "three-seconds"],
)
def test_table_node_resume_requires_a_fresh_clean_kill_read(
    last_ok_mono: float | None,
    error_class: str | None,
    now: float,
    expected: tuple[bool, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P6: the freshness gate forwards the read's detail verbatim; its edge is
    ``age > 2.0`` (2.0 s exactly still resumes)."""
    node, _trace = _traced_node(tmp_path, monkeypatch)
    node._status.state = NodeState.RUNNING
    monkeypatch.setattr(node, "_mono", lambda: now)
    monkeypatch.setattr(
        node._kill_monitor,
        "snapshot",
        lambda: KillSwitchSnapshot(
            engaged=False,
            reason=None,
            last_ok_mono=last_ok_mono,
            error_class=error_class,
            consecutive_failures=0,
        ),
    )

    assert node.resume() == expected


# ------------------------------------------------------------- P12 (lifecycle)


async def test_restored_halt_resume_is_overridden_by_the_guard_at_build(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P12: a pre-guard resume of a restored halt reports success and clears the
    status, but the guard is built from ``_restored_halt``: HALTED/MANUAL again in
    ``_run_loop``. The two instances are separate (D6), so the resume is not lost
    silently: it is simply re-decided by the guard."""
    _write_checkpoint(seeded["db"], _PRIOR_MANUAL)
    node, trace, seen = _prepare(monkeypatch, seeded)
    results: list[tuple[bool, str]] = []

    _arm(monkeypatch, node, "W4", action=lambda: results.append(node.resume()))
    await _drive(node, seen)

    assert results == [(True, _RESUMED_FROM_HALT)]
    assert seen["in_run"] == {
        "state": "halted",
        "reason": HaltReason.MANUAL,
        "message": _RESTORED_MESSAGE,
        "guard": GuardState.HALTED,
        "guard_reason": HaltReason.MANUAL,
        "guard_msg": _RESTORED_MESSAGE,
    }
    assert seen["after_run"] == seen["in_run"], "the guard's halt survives the run loop"
    assert (
        "write",
        EventType.SIGNAL,
        {
            "action": "state_change",
            "from_state": "halted",
            "to_state": "running",
            "resumed": True,
        },
        None,
    ) in trace.calls
    assert [c for c in trace.calls if c[0] == "alert"] == [], "pre-guard resume never alerts"
