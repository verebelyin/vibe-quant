"""Behaviours of ``vibe_quant.paper.halt_state`` that no table row can show.

Contract: design lane1-paper.md r2 F8 T7a. The halt/pause/resume rows themselves
live at the guard and node edges in ``test_paper_halt_state_table.py`` (G1-G15,
P1-P12) and are NOT re-run here. What stays is what an edge row cannot observe:
the edges always pass a message of their own, so message forwarding needs
unrelated literal inputs, and ``PrestartHalt.take`` is consumed once.

Chief ruling D6 keeps TWO machines: the pre-guard one checks the kill switch
before "already running", has no MAX_DAILY_LOSS branch and refuses with
``"<reason> halt cannot be resumed by the operator"``; the guard answers
"already running" first and refuses with
``"<reason> halt requires manual review; restart the session to trade again"``.
Both are pinned at the edges in the table file.
"""

from __future__ import annotations

import pytest

from vibe_quant.paper.halt_state import GuardState, HaltReason, HaltState, PrestartHalt

_ENGAGED_TEXT = "system kill switch is engaged; unlock it before resuming"


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


def test_take_consumes_the_pending_halt_once() -> None:
    """``_apply_startup_halt`` pops the record: the guard gets it exactly once."""
    halt = PrestartHalt()
    assert halt.record(None, None, HaltReason.SIGNAL, "halt via SIGUSR1") is True

    assert halt.take() == (HaltReason.SIGNAL, "halt via SIGUSR1")
    assert halt.take() is None


@pytest.mark.parametrize(
    "state", [GuardState.RUNNING, GuardState.PAUSED], ids=lambda s: s.value
)
def test_prestart_resume_refuses_while_the_kill_switch_is_engaged(state: GuardState) -> None:
    """P7: the record checks the kill switch before "already running"/"from pause".

    Only reachable on the class: ``PaperTradingNode.resume`` refuses an engaged
    kill switch itself before it calls this, so the branch is the record's own
    guarantee (D6: neither placement may be dropped alone).
    """
    halt = PrestartHalt()

    assert halt.resume(state, None, kill_switch_engaged=True) == (
        False,
        _ENGAGED_TEXT,
        None,
    )
    assert (halt.pending, halt.previous) == (None, None)
