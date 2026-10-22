"""The halt machines of ``vibe_quant.paper.halt_state`` (rabmo.9 T7a).

Contract: design lane1-paper.md r2 F8 T7a. ``test_paper_halt_state_table.py`` is
the verbatim golden of both machines (T6 rows G1-G15 for ``TradingGuard``, P1-P12
for the pre-guard node) and must stay green UNCHANGED; this file re-runs the rows
the two classes own directly against :class:`HaltState` and
:class:`PrestartHalt`, with the same literal texts, so the extraction cannot
change a word or an order.

Chief ruling D6 keeps TWO machines: the pre-guard one checks the kill switch
before "already running", has no MAX_DAILY_LOSS branch and refuses with
``"<reason> halt cannot be resumed by the operator"``; the guard answers
"already running" first and refuses with
``"<reason> halt requires manual review; restart the session to trade again"``.

Rows that are decisions of a caller, not of a class, stay pinned only by the T6
table: G14 (UTC-day rollover in ``TradingGuard.evaluate_risk``), P5 (node pause
before the guard exists), P6 (kill-snapshot freshness in ``PaperTradingNode``)
and P12 (a pre-guard resume overridden by the guard at build).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

import vibe_quant.paper.guard as guard_module
from tests.unit.test_audit_paper_guard import RecordingListener
from tests.unit.test_audit_paper_node import (
    _config,
    seeded,  # noqa: F401
)
from tests.unit.test_paper_guard_prestart_halt import _unit_guard
from vibe_quant.paper.halt_state import (
    OPERATOR_RESUMABLE_HALTS,
    RESTORED_HALT_PREFIX,
    RESTORED_HALTS,
    GuardState,
    HaltReason,
    HaltState,
    PrestartHalt,
    RestoredHalt,
    halt_supersedes,
)
from vibe_quant.paper.node import NodeState, PaperTradingNode

_MSG = "m"
_PRIOR = "prior"
_PAUSED_BY_OPERATOR = "paused by operator"
_ENGAGED_TEXT = "system kill switch is engaged; unlock it before resuming"
_ALREADY_RUNNING = "already running"
_RESUMED_FROM_PAUSE = "resumed from pause"
_RESUMED_FROM_HALT = "resumed from halt"
_GUARD_CHANGED = "{reason} halt requires manual review; restart the session to trade again"
_NODE_CHANGED = "{reason} halt cannot be resumed by the operator"
_DAILY_LOSS_TEXT = "daily-loss halt lifts automatically at the next UTC day"
_NOT_RUNNING = "trading node not running"
_ISO = "2026-01-04T00:00:00+00:00"

#: Literal severity order (independent of ``halt_state._HALT_PRIORITY``): the table
#: has teeth against a changed comparison, not just against a renamed constant.
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
    halt = _halt_in(GuardState.HALTED, HaltReason.SIGNAL)

    assert halt.apply(HaltReason.MANUAL, _MSG) is False

    assert halt.state is GuardState.HALTED
    assert halt.reason is HaltReason.SIGNAL
    assert halt.message == _PRIOR


# ---------------------------------------------------------------- HaltState


def test_guard_defaults_are_running_and_empty() -> None:
    halt = HaltState()

    assert (halt.state, halt.reason, halt.message, halt.before_halt) == (
        GuardState.RUNNING,
        None,
        "",
        GuardState.RUNNING,
    )


def _halt_in(
    state: GuardState,
    reason: HaltReason | None = None,
    prior: GuardState = GuardState.RUNNING,
) -> HaltState:
    """A ``HaltState`` in ``state``, mirroring the T6 guard helper."""
    halt = HaltState()
    if state is GuardState.PAUSED or prior is GuardState.PAUSED:
        assert halt.pause(_PAUSED_BY_OPERATOR) is True
    if state is GuardState.HALTED:
        assert halt.apply(reason, _PRIOR) is True
    return halt


@dataclass(frozen=True)
class _Start:
    label: str
    state: GuardState
    reason: HaltReason | None = None


_STARTS: list[_Start] = [
    _Start("running", GuardState.RUNNING),
    _Start("paused", GuardState.PAUSED),
    *[_Start(f"halted-{r.value}", GuardState.HALTED, r) for r in _REASONS],
]


@pytest.mark.parametrize("reason", _REASONS, ids=lambda r: r.value)
@pytest.mark.parametrize("start", _STARTS, ids=lambda s: s.label)
def test_table_guard_halt(reason: HaltReason, start: _Start) -> None:
    """G1-G6: halt applies from RUNNING/PAUSED, and replaces a HALTED reason only
    when strictly more severe (MANUAL and SIGNAL are both priority 3)."""
    halt = _halt_in(start.state, start.reason)
    applied = _supersedes(reason, start.reason) if start.state is GuardState.HALTED else True

    assert halt.apply(reason, _MSG) is applied

    if not applied:
        assert (halt.state, halt.reason, halt.message) == (
            GuardState.HALTED,
            start.reason,
            _PRIOR,
        )
        assert halt.before_halt is GuardState.RUNNING
        return
    assert (halt.state, halt.reason, halt.message) == (GuardState.HALTED, reason, _MSG)
    expected_before = start.state if start.state is not GuardState.HALTED else GuardState.RUNNING
    assert halt.before_halt is expected_before


@pytest.mark.parametrize("start", _STARTS, ids=lambda s: s.label)
def test_table_guard_pause(start: _Start) -> None:
    """G7: pause is a no-op unless RUNNING."""
    halt = _halt_in(start.state, start.reason)

    applied = halt.pause(_PAUSED_BY_OPERATOR)

    if start.state is not GuardState.RUNNING:
        assert applied is False
        assert (halt.state, halt.reason) == (start.state, start.reason)
        expected = _PRIOR if start.state is GuardState.HALTED else _PAUSED_BY_OPERATOR
        assert halt.message == expected
        return
    assert applied is True
    assert (halt.state, halt.reason, halt.message) == (
        GuardState.PAUSED,
        None,
        _PAUSED_BY_OPERATOR,
    )


@dataclass(frozen=True)
class _ResumeStart:
    label: str
    state: GuardState
    reason: HaltReason | None = None
    prior: GuardState = GuardState.RUNNING


_RESUME_STARTS: list[_ResumeStart] = [
    _ResumeStart("running", GuardState.RUNNING),
    _ResumeStart("paused", GuardState.PAUSED),
    *[
        _ResumeStart(f"halted-{r.value}-from-{p.value}", GuardState.HALTED, r, p)
        for r in _REASONS
        for p in (GuardState.RUNNING, GuardState.PAUSED)
    ],
]


@pytest.mark.parametrize("engaged", [False, True], ids=["disengaged", "engaged"])
@pytest.mark.parametrize("start", _RESUME_STARTS, ids=lambda s: s.label)
def test_table_guard_resume(start: _ResumeStart, engaged: bool) -> None:
    """G8-G13: RUNNING wins over the kill switch; engaged refuses everything else;
    daily loss lifts by itself; drawdown/streak need a restart; the rest resume
    (the caller lifts, which restores the pre-halt state)."""
    halt = _halt_in(start.state, start.reason, start.prior)

    ok, text, lift = halt.resume(kill_switch_engaged=engaged)

    if start.state is GuardState.RUNNING:
        assert (ok, text, lift) == (True, _ALREADY_RUNNING, False)
        assert halt.state is GuardState.RUNNING
        return
    if engaged:
        assert (ok, text, lift) == (False, _ENGAGED_TEXT, False)
        assert (halt.state, halt.reason, halt.message) == (
            start.state,
            start.reason,
            _PAUSED_BY_OPERATOR if start.state is GuardState.PAUSED else _PRIOR,
        )
        return
    if start.state is GuardState.PAUSED:
        assert (ok, text, lift) == (True, _RESUMED_FROM_PAUSE, False)
        assert (halt.state, halt.reason, halt.message) == (GuardState.RUNNING, None, "resumed")
        return
    assert start.reason is not None
    if start.reason is HaltReason.MAX_DAILY_LOSS:
        assert (ok, text, lift) == (False, _DAILY_LOSS_TEXT, False)
        assert (halt.state, halt.reason, halt.message) == (
            GuardState.HALTED,
            HaltReason.MAX_DAILY_LOSS,
            _PRIOR,
        )
        return
    if start.reason in _MANUAL_REVIEW:
        assert (ok, text, lift) == (False, _GUARD_CHANGED.format(reason=start.reason.value), False)
        assert (halt.state, halt.reason, halt.message) == (GuardState.HALTED, start.reason, _PRIOR)
        return
    assert start.reason in _OPERATOR_RESUMABLE
    assert (ok, text, lift) == (True, _RESUMED_FROM_HALT, True)
    halt.lift("resumed by operator")
    assert (halt.state, halt.reason, halt.message) == (
        start.prior,
        None,
        "resumed by operator",
    )
    assert halt.before_halt is GuardState.RUNNING


def test_lift_restores_the_state_before_the_halt_once() -> None:
    """G2: a halt out of PAUSED resumes to PAUSED, and lift resets before_halt."""
    halt = _halt_in(GuardState.PAUSED)
    assert halt.apply(HaltReason.MANUAL, _MSG) is True

    assert halt.resume(kill_switch_engaged=False) == (True, _RESUMED_FROM_HALT, True)
    halt.lift("resumed by operator")

    assert (halt.state, halt.reason) == (GuardState.PAUSED, None)
    assert halt.before_halt is GuardState.RUNNING


@pytest.mark.parametrize("message", ["halt via SIGUSR1", "drawdown 12.5% >= limit 10%"])
def test_apply_forwards_the_message_verbatim(message: str) -> None:
    halt = HaltState()

    assert halt.apply(HaltReason.SIGNAL, message) is True

    assert halt.message == message


@pytest.mark.parametrize("message", ["api pause", "maintenance window"])
def test_pause_forwards_the_message_verbatim(message: str) -> None:
    halt = HaltState()

    assert halt.pause(message) is True

    assert halt.message == message


@pytest.mark.parametrize("reason", _REASONS, ids=lambda r: r.value)
def test_table_guard_restored_halt_at_construction(reason: HaltReason) -> None:
    """G15: ``from_restored`` comes up HALTED with the prefixed message verbatim."""
    restored = RestoredHalt(reason, f"{RESTORED_HALT_PREFIX}halt via API", _ISO)

    halt = HaltState.from_restored(restored)

    assert (halt.state, halt.reason, halt.message) == (
        GuardState.HALTED,
        reason,
        "restored after restart: halt via API",
    )
    assert halt.before_halt is GuardState.RUNNING


def test_from_restored_none_is_a_fresh_running_machine() -> None:
    assert HaltState.from_restored(None) == HaltState()


def test_restored_halts_are_exactly_the_operator_resumable_ones() -> None:
    """H13: fail-closed after a restart only for the halts an operator may lift."""
    assert RESTORED_HALTS == OPERATOR_RESUMABLE_HALTS


# ------------------------------------------------------------- PrestartHalt


def test_node_defaults_have_no_pending_halt() -> None:
    halt = PrestartHalt()

    assert (halt.pending, halt.previous) == (None, None)
    assert halt.take() is None


def _prestart_in(
    state: GuardState | None,
    reason: HaltReason | None = None,
    previous: GuardState = GuardState.RUNNING,
) -> PrestartHalt:
    """A pre-guard record as the node leaves it after halt/pause."""
    halt = PrestartHalt()
    if state is GuardState.HALTED:
        assert halt.record(previous, None, reason, _PRIOR) is True
    return halt


@dataclass(frozen=True)
class _NodeStart:
    label: str
    state: GuardState | None
    reason: HaltReason | None = None
    previous: GuardState = GuardState.RUNNING


_NODE_STARTS: list[_NodeStart] = [
    _NodeStart("initializing", None),
    _NodeStart("running", GuardState.RUNNING),
    _NodeStart("paused", GuardState.PAUSED),
    *[_NodeStart(f"halted-{r.value}", GuardState.HALTED, r) for r in _REASONS],
]


@pytest.mark.parametrize("reason", _REASONS, ids=lambda r: r.value)
@pytest.mark.parametrize("start", _NODE_STARTS, ids=lambda s: s.label)
def test_table_node_halt(reason: HaltReason, start: _NodeStart) -> None:
    """P1-P4: the pre-guard halt records severity and the replayed message, keeping
    the state to return to; INITIALIZING (``None``) records no previous state."""
    halt = _prestart_in(start.state, start.reason)
    applied = _supersedes(reason, start.reason) if start.state is GuardState.HALTED else True

    assert halt.record(start.state, start.reason, reason, _MSG) is applied

    expected_previous = start.previous if start.state is GuardState.HALTED else start.state
    assert halt.previous is expected_previous
    if not applied:
        assert halt.pending == (start.reason, _PRIOR)
        return
    assert halt.pending == (reason, _MSG)


@pytest.mark.parametrize("engaged", [False, True], ids=["disengaged", "engaged"])
@pytest.mark.parametrize("start", _NODE_STARTS, ids=lambda s: s.label)
@pytest.mark.parametrize("previous", [GuardState.RUNNING, GuardState.PAUSED], ids=lambda p: p.value)
def test_table_node_resume(
    start: _NodeStart, engaged: bool, previous: GuardState
) -> None:
    """P7-P11: engaged is checked BEFORE "already running" (a pre-guard difference);
    the refusal text differs from the guard's and there is no daily-loss branch:
    MAX_DAILY_LOSS is simply not operator-resumable here."""
    halt = _prestart_in(start.state, start.reason, previous)
    before = (halt.pending, halt.previous)

    ok, text, target = halt.resume(
        start.state, start.reason, kill_switch_engaged=engaged
    )

    if engaged:
        assert (ok, text, target) == (False, _ENGAGED_TEXT, None)
        assert (halt.pending, halt.previous) == before
        return
    if start.state is None:
        assert (ok, text, target) == (False, _NOT_RUNNING, None)
        assert (halt.pending, halt.previous) == before
        return
    if start.state is GuardState.RUNNING:
        assert (ok, text, target) == (True, _ALREADY_RUNNING, None)
        assert (halt.pending, halt.previous) == before
        return
    if start.state is GuardState.PAUSED:
        assert (ok, text, target) == (True, _RESUMED_FROM_PAUSE, GuardState.RUNNING)
        assert (halt.pending, halt.previous) == before
        return
    assert start.reason is not None
    if start.reason not in _OPERATOR_RESUMABLE:
        assert (ok, text, target) == (False, _NODE_CHANGED.format(reason=start.reason.value), None)
        assert (halt.pending, halt.previous) == before
        return
    assert (ok, text, target) == (True, _RESUMED_FROM_HALT, previous)
    assert (halt.pending, halt.previous) == (None, None)


def test_take_consumes_the_pending_halt_once() -> None:
    """``_apply_startup_halt`` pops the record: the guard gets it exactly once."""
    halt = PrestartHalt()
    assert halt.record(None, None, HaltReason.SIGNAL, "halt via SIGUSR1") is True

    assert halt.take() == (HaltReason.SIGNAL, "halt via SIGUSR1")
    assert halt.take() is None


def test_a_refused_halt_keeps_the_earlier_pending_halt() -> None:
    """P3: refusal changes nothing, including the message that will be replayed."""
    halt = _prestart_in(GuardState.HALTED, HaltReason.SIGNAL)

    assert halt.record(GuardState.HALTED, HaltReason.SIGNAL, HaltReason.ERROR, _MSG) is False

    assert halt.take() == (HaltReason.SIGNAL, _PRIOR)


# ------------------------------------------- TradingGuard holds one HaltState (T7b)


def test_guard_owns_one_halt_state_and_its_props_read_it() -> None:
    """T7b: the four loose fields are gone; ``guard._halt`` is the only state."""
    guard = _unit_guard([])

    assert isinstance(guard._halt, HaltState)
    assert (guard._halt.state, guard._halt.reason, guard._halt.message) == (
        GuardState.RUNNING,
        None,
        "",
    )
    assert (guard.guard_state, guard.halt_reason, guard.message) == (
        GuardState.RUNNING,
        None,
        "",
    )

    assert guard.halt(HaltReason.MANUAL, _MSG) is True
    assert guard.guard_state is guard._halt.state is GuardState.HALTED
    assert guard.halt_reason is guard._halt.reason is HaltReason.MANUAL
    assert guard.message == guard._halt.message == _MSG
    assert guard._halt.before_halt is GuardState.RUNNING

    assert guard.resume() == (True, _RESUMED_FROM_HALT)
    assert guard.guard_state is guard._halt.state is GuardState.RUNNING
    assert guard.halt_reason is guard._halt.reason is None
    assert guard.message == guard._halt.message == "resumed by operator"
    assert guard._halt.before_halt is GuardState.RUNNING


def test_guard_starts_from_from_restored_halt_state() -> None:
    """T7b: a restored halt seeds the machine, exactly as the old init block did."""
    restored = RestoredHalt(HaltReason.MANUAL, f"{RESTORED_HALT_PREFIX}halt via API", _ISO)
    guard = _unit_guard([], restored_halt=restored)

    assert (
        guard._halt.state,
        guard._halt.reason,
        guard._halt.message,
        guard._halt.before_halt,
    ) == (
        GuardState.HALTED,
        HaltReason.MANUAL,
        "restored after restart: halt via API",
        GuardState.RUNNING,
    )
    assert (guard.guard_state, guard.halt_reason, guard.message) == (
        GuardState.HALTED,
        HaltReason.MANUAL,
        "restored after restart: halt via API",
    )

    assert guard.pause() is False
    assert (guard._halt.state, guard._halt.message) == (
        GuardState.HALTED,
        "restored after restart: halt via API",
    )


def test_guard_pause_and_resume_transition_the_halt_state() -> None:
    """T7b: pause/resume delegate the transition; the listener still fires once."""
    guard = _unit_guard([])
    listener = RecordingListener()
    guard._listener = listener

    assert guard.pause() is True
    assert (guard._halt.state, guard._halt.reason, guard._halt.message) == (
        GuardState.PAUSED,
        None,
        _PAUSED_BY_OPERATOR,
    )
    assert guard.resume() == (True, _RESUMED_FROM_PAUSE)
    assert (guard._halt.state, guard._halt.reason, guard._halt.message) == (
        GuardState.RUNNING,
        None,
        "resumed",
    )
    assert [(state, reason, message) for _ns, state, reason, message in listener.changes] == [
        (GuardState.PAUSED, None, _PAUSED_BY_OPERATOR),
        (GuardState.RUNNING, None, "resumed"),
    ]


# ------------------------------------ the node holds one PrestartHalt (T7c)


def test_node_maps_status_to_a_guard_state_before_recording(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T7c: ``halt`` forwards the node status through ``_NODE_TO_GUARD``: a RUNNING
    node records GuardState.RUNNING, an INITIALIZING one records None (nothing to
    return to). Two unrelated inputs, literal expectations."""
    calls: list[tuple[GuardState | None, HaltReason | None]] = []
    original = PrestartHalt.record

    def record(
        self: PrestartHalt,
        node_state: GuardState | None,
        current: HaltReason | None,
        reason: HaltReason,
        message: str,
    ) -> bool:
        calls.append((node_state, current))
        return original(self, node_state, current, reason, message)

    monkeypatch.setattr(PrestartHalt, "record", record)
    initializing = PaperTradingNode(_config(seeded))
    running = PaperTradingNode(_config(seeded))
    running._status.state = NodeState.RUNNING

    initializing.halt(HaltReason.SIGNAL, "from initializing")
    running.halt(HaltReason.SIGNAL, "from running")

    assert calls == [(None, None), (GuardState.RUNNING, None)]


def test_node_keeps_its_prestart_halt_in_one_record(
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """T7c: halt writes ``PrestartHalt.previous``/``pending``; resume clears both, and
    the halt*status* still mirrors the record."""
    node = PaperTradingNode(_config(seeded))
    node._status.state = NodeState.PAUSED

    node.halt(HaltReason.SIGNAL, "halt via SIGUSR1")

    assert isinstance(node._prestart, PrestartHalt)
    assert (node._prestart.pending, node._prestart.previous) == (
        (HaltReason.SIGNAL, "halt via SIGUSR1"),
        GuardState.PAUSED,
    )
    assert (node.status.state, node.status.halt_reason, node.status.error_message) == (
        NodeState.HALTED,
        HaltReason.SIGNAL,
        "halt via SIGUSR1",
    )

    node._kill_monitor.poll_once(node._state_manager)  # fresh clean kill-switch snapshot
    assert node.resume() == (True, _RESUMED_FROM_HALT)

    assert (node._prestart.pending, node._prestart.previous) == (None, None)
    assert node.status.state is NodeState.PAUSED


# --------------------------------------------------------------- re-exports


def test_guard_reexports_every_moved_name_by_identity() -> None:
    """T7a: guard.py swaps its definitions for imports, so importers are unchanged."""
    assert guard_module.GuardState is GuardState
    assert guard_module.HaltReason is HaltReason
    assert guard_module.RestoredHalt is RestoredHalt
    assert guard_module.halt_supersedes is halt_supersedes
    assert guard_module.OPERATOR_RESUMABLE_HALTS is OPERATOR_RESUMABLE_HALTS
    assert guard_module.RESTORED_HALTS is RESTORED_HALTS
    assert guard_module.RESTORED_HALT_PREFIX == RESTORED_HALT_PREFIX
