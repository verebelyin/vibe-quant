"""Paper start admission: one pure decision over every fact a start gathers.

A paper start is refused at several points: ``POST /api/paper/start`` and
``/restore`` in :mod:`vibe_quant.api.routers.paper_trading`, ``run_with_config``
in :mod:`vibe_quant.paper.cli`, and ``PaperTradingNode._initialize`` in
:mod:`vibe_quant.paper.node`. This module holds that decision once. A caller
probes the facts it has gathered so far and passes them as :class:`StartFacts`
(``None`` = not probed yet / not applicable, so that gate is skipped), then maps
the returned :class:`Refusal` onto its own status code, exit code or
``ConfigurationError``. The texts differ per caller, so they stay in the
adapters; :attr:`Refusal.detail` is the fact value the adapter formats, unchanged.

The order of the rules is the declaration order of :class:`RefusalCode`, which is
the order the API already applied when every fact is present: kill switch,
already-running session, LIVE consent, LIVE ``start_fresh`` consent, account
lock, no saved risk state, Telegram. Behaviour-preserving move
(vibe-quant-rabmo.10): every predicate is the one its inline site used, and a
fact counts as present when it ``is not None`` (so an empty engaged reason and a
zero run id are refusals).

Nothing here does I/O, logs or catches: callers probe and pass the facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True)
class StartIntent:
    """What the caller asked for; the consent flags are the operator's answers."""

    live: bool
    confirm_live: bool
    start_fresh: bool
    confirm_fresh_live: bool


@dataclass(frozen=True)
class StartFacts:
    """What the caller probed before asking. ``None`` = not probed / not applicable."""

    kill_missing: bool | None = None
    kill_unreadable: str | None = None
    kill_engaged: str | None = None
    active_run_id: int | None = None
    lock_dir_error: str | None = None
    lock_probe_error: str | None = None
    account_holder: str | None = None
    trader_holder: str | None = None
    has_prior_state: bool | None = None
    telegram_error: str | None = None


class RefusalCode(StrEnum):
    """Every way a paper start can be refused, in the order the checks apply."""

    KILL_MISSING = "kill_missing"
    KILL_UNREADABLE = "kill_unreadable"
    KILL_ENGAGED = "kill_engaged"
    SESSION_ACTIVE = "session_active"
    LIVE_NOT_CONFIRMED = "live_not_confirmed"
    FRESH_NOT_CONFIRMED = "fresh_not_confirmed"
    LOCK_DIR_UNAVAILABLE = "lock_dir_unavailable"
    LOCK_PROBE_FAILED = "lock_probe_failed"
    ACCOUNT_BUSY = "account_busy"
    TRADER_BUSY = "trader_busy"
    NO_PRIOR_STATE = "no_prior_state"
    TELEGRAM_UNAVAILABLE = "telegram_unavailable"


@dataclass(frozen=True)
class Refusal:
    """A refused start: which gate, and the fact value the adapter formats."""

    code: RefusalCode
    detail: str | int | None


@dataclass(frozen=True)
class Admitted:
    """An admitted start, and whether it forwards the operator's fresh consent."""

    forward_fresh_consent: bool


def check(intent: StartIntent, facts: StartFacts) -> Refusal | Admitted:
    """Decide one paper start; the first refusal in declaration order wins.

    ``forward_fresh_consent`` is the API's ``fresh_confirmed``: a LIVE start that
    confirmed fresh limits forwards that consent only when it asked for a fresh
    start or the account has no saved risk state to carry.
    """
    if facts.kill_missing is True:
        return Refusal(RefusalCode.KILL_MISSING, None)
    if facts.kill_unreadable is not None:
        return Refusal(RefusalCode.KILL_UNREADABLE, facts.kill_unreadable)
    if facts.kill_engaged is not None:
        return Refusal(RefusalCode.KILL_ENGAGED, facts.kill_engaged)
    if facts.active_run_id is not None:
        return Refusal(RefusalCode.SESSION_ACTIVE, facts.active_run_id)
    if intent.live and not intent.confirm_live:
        return Refusal(RefusalCode.LIVE_NOT_CONFIRMED, None)
    if intent.live and intent.start_fresh and not intent.confirm_fresh_live:
        return Refusal(RefusalCode.FRESH_NOT_CONFIRMED, None)
    if facts.lock_dir_error is not None:
        return Refusal(RefusalCode.LOCK_DIR_UNAVAILABLE, facts.lock_dir_error)
    if facts.lock_probe_error is not None:
        return Refusal(RefusalCode.LOCK_PROBE_FAILED, facts.lock_probe_error)
    if facts.account_holder is not None:
        return Refusal(RefusalCode.ACCOUNT_BUSY, facts.account_holder)
    if facts.trader_holder is not None:
        return Refusal(RefusalCode.TRADER_BUSY, facts.trader_holder)
    if intent.live and not intent.confirm_fresh_live and facts.has_prior_state is False:
        return Refusal(RefusalCode.NO_PRIOR_STATE, None)
    if facts.telegram_error is not None:
        return Refusal(RefusalCode.TELEGRAM_UNAVAILABLE, facts.telegram_error)
    return Admitted(
        forward_fresh_consent=(
            intent.live
            and intent.confirm_fresh_live
            and (intent.start_fresh or facts.has_prior_state is False)
        )
    )
