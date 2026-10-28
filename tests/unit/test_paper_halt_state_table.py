"""Pin the paper halt/pause/resume table at its two public edges (rabmo.9, qlx46).

Contract: design lane1-paper.md r2 F8, "T6 rows" (G1-G15 for ``TradingGuard``,
P1-P12 for the pre-guard ``PaperTradingNode``). This is the only copy of each
row: the guard and the node are the edges an operator's halt/resume reaches, so
every row is driven through them, and ``test_halt_state.py`` keeps only the
behaviours of the extracted classes that have no row here (verbatim message
forwarding, ``PrestartHalt.take`` once).

Rows are boundary rows, not the full reason x start product: every distinct
outcome text and severity branch appears once, and every ``HaltReason`` appears
somewhere. Within one branch the rows differ only where the expected text,
severity comparison or returned state differs.

Chief ruling D6: the refactor keeps TWO halt-state instances (the guard's
machine and the node's pre-guard record), so every pre-guard difference is
PINNED here, never unified:

* engaged is checked before "already running": pre-guard RUNNING + engaged
  refuses with the kill-switch text, the guard answers "already running";
* ``"<reason> halt cannot be resumed by the operator"`` (pre-guard) vs
  ``"<reason> halt requires manual review; restart the session to trade again"``
  (guard);
* the pre-guard resume has no MAX_DAILY_LOSS branch (that halt is not resumable
  before the guard exists);
* a pre-guard resume of a restored halt is overridden by the guard at build (P12).

Every expected string is literal here, so a refactor cannot silently re-derive it.
"""

from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Any, NamedTuple

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
from vibe_quant.paper.journal import FakeJournal
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
_DAILY_LOSS_TEXT = "daily-loss halt lifts automatically at the next UTC day"
_NOT_RUNNING = "trading node not running"
_STARTUP_HALT_WARNING = (
    "paper PAPER-007: {reason} halt during start-up recorded; "
    "the trading guard applies it when built"
)


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
        assert reason is not None
        assert guard.halt(reason, _PRIOR) is True
    listener.changes.clear()
    return guard, listener


def _changes(listener: RecordingListener) -> list[tuple[GuardState, HaltReason | None, str]]:
    return [(state, reason, message) for _ns, state, reason, message in listener.changes]


def test_table_guard_starts_running_with_the_gate_open() -> None:
    """A guard built without a restored halt is RUNNING: entries are allowed."""
    guard = _unit_guard([])

    status = guard.status()

    assert (status["state"], status["halt_reason"], status["message"]) == ("running", None, "")
    assert status["gate"] == "active"


@pytest.mark.parametrize(
    ("start", "expected"),
    [(GuardState.RUNNING, GuardState.RUNNING), (GuardState.PAUSED, GuardState.PAUSED)],
    ids=["running", "paused"],
)
def test_table_guard_halt_from_running_or_paused(
    start: GuardState, expected: GuardState
) -> None:
    """G1-G2: halt applies from RUNNING/PAUSED and remembers which state to return to."""
    guard, listener = _guard_in(start)

    assert guard.halt(HaltReason.MANUAL, _MSG) is True

    assert (guard.guard_state, guard.halt_reason, guard.message) == (
        GuardState.HALTED,
        HaltReason.MANUAL,
        _MSG,
    )
    assert guard.status()["gate"] == "reduce_only"
    assert _changes(listener) == [(GuardState.HALTED, HaltReason.MANUAL, _MSG)]
    assert guard.resume() == (True, _RESUMED_FROM_HALT)
    assert guard.guard_state is expected


#: Adjacent pairs of the literal severity order, both ways: MAX_DAILY_LOSS < ERROR
#: < MANUAL == SIGNAL < MAX_CONSECUTIVE_LOSSES < MAX_DRAWDOWN < KILL_SWITCH. A halt
#: replaces a HALTED reason only when strictly more severe, so the tie at
#: MANUAL/SIGNAL refuses in both directions.
_SEVERITY_PAIRS: list[tuple[HaltReason, HaltReason, bool]] = [
    (HaltReason.ERROR, HaltReason.MAX_DAILY_LOSS, True),
    (HaltReason.MAX_DAILY_LOSS, HaltReason.ERROR, False),
    (HaltReason.MANUAL, HaltReason.ERROR, True),
    (HaltReason.ERROR, HaltReason.MANUAL, False),
    (HaltReason.SIGNAL, HaltReason.MANUAL, False),
    (HaltReason.MANUAL, HaltReason.SIGNAL, False),
    (HaltReason.MAX_CONSECUTIVE_LOSSES, HaltReason.MANUAL, True),
    (HaltReason.MANUAL, HaltReason.MAX_CONSECUTIVE_LOSSES, False),
    (HaltReason.MAX_DRAWDOWN, HaltReason.MAX_CONSECUTIVE_LOSSES, True),
    (HaltReason.MAX_CONSECUTIVE_LOSSES, HaltReason.MAX_DRAWDOWN, False),
    (HaltReason.KILL_SWITCH, HaltReason.MAX_DRAWDOWN, True),
    (HaltReason.MAX_DRAWDOWN, HaltReason.KILL_SWITCH, False),
]


@pytest.mark.parametrize(
    ("new", "current", "applied"),
    _SEVERITY_PAIRS,
    ids=[f"{new.value}-over-{current.value}" for new, current, _applied in _SEVERITY_PAIRS],
)
def test_table_guard_halt_replaces_a_halted_reason_only_when_more_severe(
    new: HaltReason, current: HaltReason, applied: bool
) -> None:
    """G3-G6: equal severity never replaces the active halt (MANUAL/SIGNAL tie)."""
    guard, listener = _guard_in(GuardState.HALTED, current)

    assert guard.halt(new, _MSG) is applied

    if not applied:
        assert (guard.guard_state, guard.halt_reason, guard.message) == (
            GuardState.HALTED,
            current,
            _PRIOR,
        )
        assert guard.status()["halt_reason"] == current.value
        assert guard.status()["message"] == _PRIOR
        assert _changes(listener) == [], "a refused halt notifies nobody"
        return
    assert (guard.guard_state, guard.halt_reason, guard.message) == (
        GuardState.HALTED,
        new,
        _MSG,
    )
    assert guard.status()["halt_reason"] == new.value
    assert guard.status()["gate"] == "reduce_only"
    assert _changes(listener) == [(GuardState.HALTED, new, _MSG)]


@pytest.mark.parametrize(
    "start", [GuardState.RUNNING, GuardState.PAUSED, GuardState.HALTED], ids=lambda s: s.value
)
def test_table_guard_pause(start: GuardState) -> None:
    """G7: pause is a no-op unless RUNNING; its default message is pinned."""
    guard, listener = _guard_in(
        start, HaltReason.MANUAL if start is GuardState.HALTED else None
    )

    applied = guard.pause()

    assert applied is (start is GuardState.RUNNING)
    if applied:
        assert (guard.guard_state, guard.halt_reason, guard.message) == (
            GuardState.PAUSED,
            None,
            "paused by operator",
        )
        assert _changes(listener) == [(GuardState.PAUSED, None, "paused by operator")]
        return
    assert guard.guard_state is start
    kept_message = _PRIOR if start is GuardState.HALTED else "paused by operator"
    assert guard.message == kept_message
    assert _changes(listener) == []


class _GuardResumeRow(NamedTuple):
    label: str
    start: GuardState
    ok: bool
    text: str
    engaged: bool = False
    reason: HaltReason | None = None
    prior: GuardState = GuardState.RUNNING
    after: GuardState = GuardState.RUNNING
    message: str = ""
    notifies: bool = False


_GUARD_RESUME_ROWS: list[_GuardResumeRow] = [
    _GuardResumeRow("running", GuardState.RUNNING, True, _ALREADY_RUNNING),
    _GuardResumeRow("running-engaged", GuardState.RUNNING, True, _ALREADY_RUNNING, engaged=True),
    _GuardResumeRow(
        "paused-engaged",
        GuardState.PAUSED,
        False,
        _ENGAGED_TEXT,
        engaged=True,
        after=GuardState.PAUSED,
        message="paused by operator",
    ),
    _GuardResumeRow(
        "halted-engaged",
        GuardState.HALTED,
        False,
        _ENGAGED_TEXT,
        engaged=True,
        reason=HaltReason.MANUAL,
        after=GuardState.HALTED,
        message=_PRIOR,
    ),
    _GuardResumeRow(
        "paused",
        GuardState.PAUSED,
        True,
        _RESUMED_FROM_PAUSE,
        message="resumed",
        notifies=True,
    ),
    _GuardResumeRow(
        "halted-daily-loss",
        GuardState.HALTED,
        False,
        _DAILY_LOSS_TEXT,
        reason=HaltReason.MAX_DAILY_LOSS,
        after=GuardState.HALTED,
        message=_PRIOR,
    ),
    _GuardResumeRow(
        "halted-drawdown",
        GuardState.HALTED,
        False,
        _GUARD_CHANGED.format(reason=HaltReason.MAX_DRAWDOWN.value),
        reason=HaltReason.MAX_DRAWDOWN,
        after=GuardState.HALTED,
        message=_PRIOR,
    ),
    _GuardResumeRow(
        "halted-streak",
        GuardState.HALTED,
        False,
        _GUARD_CHANGED.format(reason=HaltReason.MAX_CONSECUTIVE_LOSSES.value),
        reason=HaltReason.MAX_CONSECUTIVE_LOSSES,
        after=GuardState.HALTED,
        message=_PRIOR,
    ),
    _GuardResumeRow(
        "halted-manual",
        GuardState.HALTED,
        True,
        _RESUMED_FROM_HALT,
        reason=HaltReason.MANUAL,
        message="resumed by operator",
        notifies=True,
    ),
    _GuardResumeRow(
        "halted-error",
        GuardState.HALTED,
        True,
        _RESUMED_FROM_HALT,
        reason=HaltReason.ERROR,
        message="resumed by operator",
        notifies=True,
    ),
    _GuardResumeRow(
        "halted-signal",
        GuardState.HALTED,
        True,
        _RESUMED_FROM_HALT,
        reason=HaltReason.SIGNAL,
        message="resumed by operator",
        notifies=True,
    ),
    _GuardResumeRow(
        "halted-kill-switch-from-paused",
        GuardState.HALTED,
        True,
        _RESUMED_FROM_HALT,
        reason=HaltReason.KILL_SWITCH,
        prior=GuardState.PAUSED,
        after=GuardState.PAUSED,
        message="resumed by operator",
        notifies=True,
    ),
]


@pytest.mark.parametrize("row", _GUARD_RESUME_ROWS, ids=lambda r: r.label)
def test_table_guard_resume(row: _GuardResumeRow) -> None:
    """G8-G13: RUNNING wins over the kill switch; engaged refuses everything else;
    daily loss lifts by itself; drawdown/streak need a restart; the rest resume."""
    guard, listener = _guard_in(row.start, row.reason, row.prior)

    assert guard.resume(kill_switch_engaged=row.engaged) == (row.ok, row.text)

    assert guard.guard_state is row.after
    assert guard.message == row.message
    expected_changes = [(row.after, None, row.message)] if row.notifies else []
    assert _changes(listener) == expected_changes
    if row.start is GuardState.HALTED:
        assert guard.halt_reason is (None if row.ok else row.reason)
        if row.ok:
            # The lift resets the pivot so the next halt returns to RUNNING again.
            assert guard._halt.before_halt is GuardState.RUNNING


@pytest.mark.parametrize(
    ("reason", "lifts"),
    [
        (HaltReason.MAX_DAILY_LOSS, True),
        (HaltReason.MANUAL, False),
        (HaltReason.MAX_DRAWDOWN, False),
        (HaltReason.ERROR, False),
        (HaltReason.KILL_SWITCH, False),
    ],
    ids=["daily-loss", "manual", "drawdown", "error", "kill-switch"],
)
def test_table_guard_daily_rollover(reason: HaltReason, lifts: bool) -> None:
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

    if lifts:
        assert guard.guard_state is GuardState.RUNNING
        assert guard.halt_reason is None
        assert guard.message == "UTC day rollover to 2026-01-05"
        assert _changes(listener) == [
            (GuardState.RUNNING, None, "UTC day rollover to 2026-01-05")
        ]
        return
    assert guard.guard_state is GuardState.HALTED
    assert guard.halt_reason is reason
    assert guard.message == _PRIOR
    assert _changes(listener) == []


@pytest.mark.parametrize("reason", [HaltReason.MANUAL, HaltReason.KILL_SWITCH], ids=lambda r: r.value)
def test_table_guard_restored_halt_at_construction(reason: HaltReason) -> None:
    """G15: a restored halt comes up HALTED with the prefixed message."""
    restored = RestoredHalt(reason, f"{RESTORED_HALT_PREFIX}halt via API", _ISO)
    guard = _unit_guard([], restored_halt=restored)
    listener = RecordingListener()
    guard._listener = listener

    assert guard.guard_state is GuardState.HALTED
    assert guard.halt_reason is reason
    assert guard.message == "restored after restart: halt via API"
    assert guard._halt.before_halt is GuardState.RUNNING
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
    fake = FakeJournal()
    node._journal = fake
    return node, _Trace(calls=fake.calls)


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
        assert reason is not None
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


class _NodeHaltRow(NamedTuple):
    label: str
    start: NodeState
    new: HaltReason
    applied: bool
    current: HaltReason | None = None


#: The severity comparison is the guard's own (``halt_supersedes``); these rows
#: pin it once from the pre-guard edge: applied from INITIALIZING/RUNNING/PAUSED
#: and on an escalation, refused on an equal or less severe halt.
_NODE_HALT_ROWS: list[_NodeHaltRow] = [
    _NodeHaltRow("initializing", NodeState.INITIALIZING, HaltReason.MANUAL, True),
    _NodeHaltRow("running", NodeState.RUNNING, HaltReason.MANUAL, True),
    _NodeHaltRow("paused", NodeState.PAUSED, HaltReason.MANUAL, True),
    _NodeHaltRow(
        "halted-equal-severity",
        NodeState.HALTED,
        HaltReason.SIGNAL,
        False,
        current=HaltReason.MANUAL,
    ),
    _NodeHaltRow(
        "halted-escalates",
        NodeState.HALTED,
        HaltReason.MANUAL,
        True,
        current=HaltReason.ERROR,
    ),
    _NodeHaltRow(
        "halted-more-severe",
        NodeState.HALTED,
        HaltReason.KILL_SWITCH,
        True,
        current=HaltReason.MAX_DRAWDOWN,
    ),
]


@pytest.mark.parametrize("row", _NODE_HALT_ROWS, ids=lambda r: r.label)
def test_table_node_halt(
    row: _NodeHaltRow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """P1-P4: the pre-guard halt records severity, status and the replayed message;
    it writes NO event and NO alert before the guard exists."""
    node, trace = _traced_node(tmp_path, monkeypatch)
    _set_node_state(node, row.start, row.current)
    trace.calls.clear()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="vibe_quant.paper.node"):
        node.halt(row.new, _MSG)

    if not row.applied:
        assert _node_status(node) == (NodeState.HALTED, row.current, _PRIOR)
        assert trace.calls == []
        assert _warnings(caplog) == []
        return
    assert _node_status(node) == (NodeState.HALTED, row.new, _MSG)
    assert node._state_payload() == {
        "state": "halted",
        "halt_reason": row.new.value,
        "message": _MSG,
    }
    assert trace.calls == [], "no event and no alert before the guard exists"
    assert _warnings(caplog) == [_STARTUP_HALT_WARNING.format(reason=row.new.value)]


@pytest.mark.parametrize(
    "reason", [HaltReason.SIGNAL, HaltReason.MAX_DRAWDOWN], ids=lambda r: r.value
)
def test_table_node_halt_defaults_the_message_to_the_reason(
    reason: HaltReason, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1: ``halt(reason)`` without a message records the reason's own text."""
    node, _trace = _traced_node(tmp_path, monkeypatch)

    node.halt(reason)

    assert _node_status(node) == (NodeState.HALTED, reason, reason.value)


@pytest.mark.parametrize(
    "start", [NodeState.INITIALIZING, NodeState.RUNNING, NodeState.PAUSED], ids=lambda s: s.value
)
def test_table_node_pause(
    start: NodeState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P5: only a RUNNING node pauses, and only that one write is emitted."""
    node, trace = _traced_node(tmp_path, monkeypatch)
    _set_node_state(node, start)
    trace.calls.clear()
    before = _node_status(node)

    node.pause()

    if start is not NodeState.RUNNING:
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


class _NodeResumeRow(NamedTuple):
    label: str
    start: NodeState
    ok: bool
    text: str
    engaged: bool = False
    reason: HaltReason | None = None
    previous: NodeState = NodeState.RUNNING
    after: NodeState = NodeState.RUNNING


_NODE_RESUME_ROWS: list[_NodeResumeRow] = [
    # engaged wins over BOTH earlier returns that exist here: "already running"
    # (guard order) and "trading node not running" (pre-guard order).
    _NodeResumeRow("engaged-before-already-running", NodeState.RUNNING, False, _ENGAGED_TEXT, engaged=True),
    _NodeResumeRow("engaged-before-not-running", NodeState.INITIALIZING, False, _ENGAGED_TEXT, engaged=True),
    _NodeResumeRow("initializing", NodeState.INITIALIZING, False, _NOT_RUNNING),
    _NodeResumeRow("running", NodeState.RUNNING, True, _ALREADY_RUNNING),
    _NodeResumeRow("paused", NodeState.PAUSED, True, _RESUMED_FROM_PAUSE),
    _NodeResumeRow(
        "halted-daily-loss",
        NodeState.HALTED,
        False,
        _NODE_CHANGED.format(reason=HaltReason.MAX_DAILY_LOSS.value),
        reason=HaltReason.MAX_DAILY_LOSS,
    ),
    _NodeResumeRow(
        "halted-drawdown",
        NodeState.HALTED,
        False,
        _NODE_CHANGED.format(reason=HaltReason.MAX_DRAWDOWN.value),
        reason=HaltReason.MAX_DRAWDOWN,
    ),
    _NodeResumeRow(
        "halted-manual", NodeState.HALTED, True, _RESUMED_FROM_HALT, reason=HaltReason.MANUAL
    ),
    _NodeResumeRow("halted-error", NodeState.HALTED, True, _RESUMED_FROM_HALT, reason=HaltReason.ERROR),
    _NodeResumeRow("halted-signal", NodeState.HALTED, True, _RESUMED_FROM_HALT, reason=HaltReason.SIGNAL),
    _NodeResumeRow(
        "halted-kill-switch-from-paused",
        NodeState.HALTED,
        True,
        _RESUMED_FROM_HALT,
        reason=HaltReason.KILL_SWITCH,
        previous=NodeState.PAUSED,
        after=NodeState.PAUSED,
    ),
]


@pytest.mark.parametrize("row", _NODE_RESUME_ROWS, ids=lambda r: r.label)
def test_table_node_resume(
    row: _NodeResumeRow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P7-P11: engaged is checked BEFORE "already running" (a pre-guard difference);
    the pre-guard refusal text differs from the guard's, and there is no
    daily-loss branch: MAX_DAILY_LOSS is simply not operator-resumable here."""
    node, trace = _traced_node(tmp_path, monkeypatch)
    _set_node_state(node, row.start, row.reason, row.previous)
    trace.calls.clear()
    before = _node_status(node)
    monkeypatch.setattr(node, "_mono", lambda: 0.0)
    monkeypatch.setattr(node._kill_monitor, "snapshot", lambda: _fresh_kill_snapshot(row.engaged))

    assert node.resume() == (row.ok, row.text)

    if row.start is NodeState.PAUSED:
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
    if row.start is NodeState.HALTED and row.ok:
        assert _node_status(node) == (row.after, None, None)
        assert trace.calls == [
            (
                "write",
                EventType.SIGNAL,
                {
                    "action": "state_change",
                    "from_state": "halted",
                    "to_state": row.after.value,
                    "resumed": True,
                },
                None,
            )
        ]
        return
    assert _node_status(node) == before
    assert trace.calls == []


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
