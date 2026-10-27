"""Health of the operator command channel (bead vibe-quant-3fg9d).

The node claims queued operator commands (halt/kill/close_all/...) once per control tick.
While every claim fails, a queued halt is never read, so after ``COMMAND_CLAIM_FAILURE_LIMIT``
consecutive failures the node blocks new entries through the guard's existing pause and
tells the operator once. This module is the pure part: the streak counter, the ownership
flag and the user-visible texts. No I/O, no exception text (class names only).
"""

from __future__ import annotations

from dataclasses import dataclass

from vibe_quant.paper.halt_state import GuardState, HaltReason

#: Consecutive failed claims after which new entries stay blocked.
COMMAND_CLAIM_FAILURE_LIMIT = 5

#: A claim that gives no answer within this many seconds counts as one failed claim.
COMMAND_CLAIM_STALL_S = 1.0

#: ``last_error`` of a stalled claim (an exception class name stands for a raised one).
STALL_LABEL = f"no answer within {COMMAND_CLAIM_STALL_S:g} s"


@dataclass
class CommandChannelHealth:
    """Consecutive failed command claims and who owns the entry block.

    Attributes:
        failures: Consecutive failed claims (raised or stalled).
        blocking: The streak reached its limit; entries must stay blocked.
        last_error: Exception CLASS name or ``STALL_LABEL``, never ``str(exc)``.
        owns_pause: The guard's PAUSED was set by this channel, so only then may a
            recovery lift it (an operator pause or a halt is never ours to lift).
        unreported_recovery: Failures of a blocked streak whose "restored" alert is owed.
    """

    failures: int = 0
    blocking: bool = False
    last_error: str = ""
    owns_pause: bool = False
    unreported_recovery: int | None = None

    def record_failure(self, error: str, *, limit: int = COMMAND_CLAIM_FAILURE_LIMIT) -> bool:
        """Count one failed claim; True exactly once per streak (the call that blocks)."""
        self.failures += 1
        self.last_error = error
        if self.blocking or self.failures < limit:
            return False
        self.blocking = True
        return True

    def record_success(self) -> None:
        """A claim returned: a blocked streak owes its restored alert; the count restarts."""
        if self.blocking:
            self.unreported_recovery = self.failures
        self.failures = 0
        self.blocking = False
        self.last_error = ""


def guard_state_text(state: GuardState, reason: HaltReason | None) -> str:
    """``paused``, ``running`` or ``halted/<reason>`` (the guard state in operator words)."""
    if state == GuardState.HALTED and reason is not None:
        return f"halted/{reason.value}"
    return state.value


def block_message(failures: int, error: str) -> str:
    """The guard pause message (also the status text while entries are blocked)."""
    return (
        f"command channel failing ({failures} consecutive claim failures, last {error}); "
        "new entries blocked"
    )


def failed_alert_text(trader_id: str, failures: int, error: str, entries: str) -> str:
    """The critical alert of a streak. ``entries``: what became of new entries."""
    return (
        f"paper {trader_id}: command channel failing ({failures} consecutive claim failures, "
        f"last {error}); {entries}; queued halt/kill/close_all commands are not being read "
        "- use the system kill switch or SIGUSR1"
    )


def restored_alert_text(trader_id: str, failures: int, entries: str) -> str:
    """The alert once claims work again. ``entries``: resumed, or why they stay blocked."""
    return (
        f"paper {trader_id}: command channel restored after {failures} failed claims; {entries}"
    )
