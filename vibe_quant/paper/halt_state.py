"""The paper node's halt family: states, reasons and the two halt state machines.

Behaviour-preserving extraction of rabmo.9. Everything here lived in
:mod:`vibe_quant.paper.guard` (the enum family) and in
:class:`vibe_quant.paper.guard.TradingGuard` / :class:`vibe_quant.paper.node.PaperTradingNode`
(the machines); the names are re-exported from ``guard`` so no importer changes.

Chief ruling D6: there are TWO instances, never one shared object handed to the
guard. :class:`HaltState` is the trading guard's machine and
:class:`PrestartHalt` is the node's pre-guard record; the guard is built from the
node's ``RestoredHalt`` afterwards, so a pre-guard resume is simply re-decided by
the guard at build. Their differing texts and check order are deliberate and
pinned by ``tests/unit/test_paper_halt_state_table.py``:

* the pre-guard record checks the kill switch before "already running", has no
  ``MAX_DAILY_LOSS`` branch and refuses with ``"<reason> halt cannot be resumed
  by the operator"``;
* the guard answers "already running" first and refuses with ``"<reason> halt
  requires manual review; restart the session to trade again"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class GuardState(StrEnum):
    """Trading state owned by the guard."""

    RUNNING = "running"
    PAUSED = "paused"
    HALTED = "halted"


class HaltReason(StrEnum):
    """Reason for halting the trading node."""

    MAX_DRAWDOWN = "max_drawdown"
    MAX_DAILY_LOSS = "max_daily_loss"
    MAX_CONSECUTIVE_LOSSES = "max_consecutive_losses"
    MANUAL = "manual"
    ERROR = "error"
    SIGNAL = "signal"
    KILL_SWITCH = "kill_switch"


# A new halt only replaces the current one when it is at least as severe.
_HALT_PRIORITY: dict[HaltReason, int] = {
    HaltReason.MAX_DAILY_LOSS: 1,
    HaltReason.ERROR: 2,
    HaltReason.MANUAL: 3,
    HaltReason.SIGNAL: 3,
    HaltReason.MAX_CONSECUTIVE_LOSSES: 4,
    HaltReason.MAX_DRAWDOWN: 5,
    HaltReason.KILL_SWITCH: 6,
}


def halt_supersedes(new: HaltReason, current: HaltReason | None) -> bool:
    """True when ``new`` may replace ``current`` (strictly more severe, or nothing active)."""
    return current is None or _HALT_PRIORITY[new] > _HALT_PRIORITY[current]


# Halts that flatten and block entries but keep strategies running so they can
# resume automatically (daily loss lifts at the next UTC midnight).
_KEEP_STRATEGIES_RUNNING: frozenset[HaltReason] = frozenset({HaltReason.MAX_DAILY_LOSS})

# Halts an operator may resume from without restarting the session.
OPERATOR_RESUMABLE_HALTS: frozenset[HaltReason] = frozenset(
    {HaltReason.MANUAL, HaltReason.SIGNAL, HaltReason.ERROR, HaltReason.KILL_SWITCH}
)

# Halts restored to HALTED after a restart: fail-closed until the operator
# resumes (D-H1). Risk halts are re-derived from the restored bookkeeping.
RESTORED_HALTS: frozenset[HaltReason] = OPERATOR_RESUMABLE_HALTS

#: Prefix on every restored-halt message (stripped before re-prefixing, H13).
RESTORED_HALT_PREFIX = "restored after restart: "


@dataclass(frozen=True)
class RestoredHalt:
    """A halt restored from the prior session's checkpoint."""

    reason: HaltReason
    message: str  # already prefixed with RESTORED_HALT_PREFIX
    checkpoint_time: str  # prior checkpoint timestamp (isoformat)


@dataclass
class HaltState:
    """The trading guard's halt/pause machine (was ``_state`` and friends).

    State only: the caller still owns the gate, the flatten, the listener and the
    deferred-flatten flag. ``resume`` returns ``lift`` so the caller can run its
    own lift side effects and then call :meth:`lift` for the state transition.
    """

    state: GuardState = GuardState.RUNNING
    reason: HaltReason | None = None
    message: str = ""
    before_halt: GuardState = GuardState.RUNNING

    @classmethod
    def from_restored(cls, restored: RestoredHalt | None) -> HaltState:
        """The machine a guard starts with (mirrors ``TradingGuard.__init__``)."""
        if restored is None:
            return cls()
        return cls(
            state=GuardState.HALTED,
            reason=restored.reason,
            message=restored.message,
            before_halt=GuardState.RUNNING,
        )

    def apply(self, reason: HaltReason, message: str) -> bool:
        """Enter HALTED (mirrors ``TradingGuard.halt`` lines 941-948).

        Returns False when an equal-or-more-severe halt is already active.
        """
        if self.state == GuardState.HALTED and not halt_supersedes(reason, self.reason):
            return False
        if self.state != GuardState.HALTED:
            # A paused session that gets halted returns to PAUSED on resume.
            self.before_halt = self.state
        self.state = GuardState.HALTED
        self.reason = reason
        self.message = message
        return True

    def pause(self, message: str) -> bool:
        """Enter PAUSED; no-op unless RUNNING (mirrors ``TradingGuard.pause`` 965-968)."""
        if self.state != GuardState.RUNNING:
            return False
        self.state = GuardState.PAUSED
        self.message = message
        return True

    def resume(self, *, kill_switch_engaged: bool) -> tuple[bool, str, bool]:
        """Decide a resume (mirrors ``TradingGuard.resume`` lines 974-990).

        Returns ``(ok, text, lift)``: ``lift`` asks the caller to run its own
        side effects and call :meth:`lift` with "resumed by operator".
        """
        if self.state == GuardState.RUNNING:
            return True, "already running", False
        if kill_switch_engaged:
            return False, "system kill switch is engaged; unlock it before resuming", False
        if self.state == GuardState.PAUSED:
            self.state = GuardState.RUNNING
            self.message = "resumed"
            return True, "resumed from pause", False
        reason = self.reason
        if reason == HaltReason.MAX_DAILY_LOSS:
            return False, "daily-loss halt lifts automatically at the next UTC day", False
        if reason not in OPERATOR_RESUMABLE_HALTS:
            why = reason.value if reason else "unknown"
            return (
                False,
                f"{why} halt requires manual review; restart the session to trade again",
                False,
            )
        return True, "resumed from halt", True

    def lift(self, message: str) -> None:
        """Leave HALTED for the pre-halt state (mirrors ``_lift_halt`` lines 1009-1012)."""
        self.state = self.before_halt
        self.before_halt = GuardState.RUNNING
        self.reason = None
        self.message = message


@dataclass
class PrestartHalt:
    """The node's halt record before the trading guard exists.

    Replaces ``PaperTradingNode._previous_state`` + ``_startup_halt``. The node
    keeps its status writes, log line and event/alert suppression; this object
    only owns the record the guard later replays plus the resume decision.
    """

    pending: tuple[HaltReason, str] | None = None
    previous: GuardState | None = None

    def record(
        self,
        node_state: GuardState | None,
        current: HaltReason | None,
        reason: HaltReason,
        message: str,
    ) -> bool:
        """Record a pre-guard halt (mirrors ``PaperTradingNode.halt`` 3236-3246).

        ``node_state`` is None for INITIALIZING/STOPPED/ERROR (no state to
        return to). Returns False when a more severe halt is already recorded.
        """
        if node_state == GuardState.HALTED and not halt_supersedes(reason, current):
            return False
        if node_state in (GuardState.RUNNING, GuardState.PAUSED):
            self.previous = node_state
        self.pending = (reason, message)
        return True

    def resume(
        self,
        node_state: GuardState | None,
        current: HaltReason | None,
        *,
        kill_switch_engaged: bool,
    ) -> tuple[bool, str, GuardState | None]:
        """Decide a pre-guard resume (mirrors ``PaperTradingNode.resume`` 3312-3350).

        The kill switch is still checked first (as in the node); returns
        ``(ok, text, target)`` where ``target`` is the state the caller must
        write to its status, or None when nothing changes.
        """
        if kill_switch_engaged:
            return False, "system kill switch is engaged; unlock it before resuming", None
        if node_state == GuardState.PAUSED:
            return True, "resumed from pause", GuardState.RUNNING
        if node_state == GuardState.RUNNING:
            return True, "already running", None
        reason = current
        if node_state == GuardState.HALTED and reason in OPERATOR_RESUMABLE_HALTS:
            target = self.previous or GuardState.RUNNING
            self.previous = None
            self.pending = None
            return True, "resumed from halt", target
        if node_state == GuardState.HALTED:
            why = reason.value if reason else "unknown"
            return False, f"{why} halt cannot be resumed by the operator", None
        return False, "trading node not running", None

    def take(self) -> tuple[HaltReason, str] | None:
        """Pop the pending halt for ``_apply_startup_halt`` (consumed exactly once)."""
        pending, self.pending = self.pending, None
        return pending


__all__ = [
    "OPERATOR_RESUMABLE_HALTS",
    "RESTORED_HALTS",
    "RESTORED_HALT_PREFIX",
    "GuardState",
    "HaltReason",
    "HaltState",
    "PrestartHalt",
    "RestoredHalt",
    "halt_supersedes",
]
