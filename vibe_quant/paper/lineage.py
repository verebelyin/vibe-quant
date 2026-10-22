"""Checkpoint lineage: which saved session a start carries.

The restore (:meth:`~vibe_quant.paper.node.PaperTradingNode._restore_from_checkpoint`)
and the LIVE consent gate ask the same questions of the same rows, so they live in
one place, over the caller's :class:`~vibe_quant.paper.persistence.StatePersistence`
connection. Behaviour-preserving move (vibe-quant-rabmo.1): every SQL text, ORDER
BY, LIMIT and parameter order is the one the store used before the move, and the
SQL literals keep their original spelling so a diff against the old methods is empty.

:func:`resolve` is the whole restore decision as one pure function: the node maps the
returned :class:`Inheritance` onto its ``_restored_*`` wiring and writes the event.

Callers import the module (``from vibe_quant.paper import lineage``) and call
``lineage.has_prior_state(...)``: a monkeypatch on a module attribute then reaches
every caller at once.
"""

from __future__ import annotations

import contextlib
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from vibe_quant.paper.guard import RiskState
from vibe_quant.paper.halt_state import (
    RESTORED_HALT_PREFIX,
    RESTORED_HALTS,
    HaltReason,
    RestoredHalt,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from vibe_quant.paper.persistence import StateCheckpoint, StatePersistence


#: The restore's log lines are part of the paper transcript, byte for byte, and they
#: named this logger before the move (vibe-quant-rabmo.1c): keep the name so an
#: operator's log filter and every pinned line stay valid.
logger = logging.getLogger("vibe_quant.paper.node")


#: ae6ah: the trader's rows in one venue + env: any key of that env, a legacy row (no /
#: null account), an unreadable row, or a non-string account (the last two fail closed
#: when parsed or checked). One predicate for the consent gate and the restore, so
#: "the gate asks for consent" and "nothing carries" cannot drift apart. Parameters:
#: ``(trader_id, account_env(account) + ":", account_env(account) + ":")``.
_ENV_ROW_WHERE = """trader_id = ?
  AND CASE WHEN json_valid(node_status)
           THEN CASE json_type(node_status, '$.account')
                    WHEN 'text' THEN substr(json_extract(node_status, '$.account'), 1, length(?)) = ?
                    ELSE 1 END
           ELSE 1 END"""


def account_env(account: str) -> str:
    """``binance:testnet:<hash>`` -> ``binance:testnet`` (qa42q: same venue + env)."""
    return account.rpartition(":")[0]


def is_key_rotation(newest_account: object, account: str) -> bool:
    """Is the trader's newest checkpoint account a KEY ROTATION into ``account``?

    True when ``newest_account`` is another account of the same venue + env (another
    key). The one rule the LIVE consent gate and the restore share (vibe-quant-neuvu.2),
    so "the gate asks for consent" and "nothing carries" cannot drift apart.

    Raises:
        TypeError: ``newest_account`` is neither None nor a string (a corrupt row);
            the restore fails closed on it, the gate treats it as "no rotation".
    """
    if newest_account is None:
        return False
    if not isinstance(newest_account, str):
        raise TypeError(f"checkpoint account must be a string, not {type(newest_account).__name__}")
    return newest_account != account and account_env(newest_account) == account_env(account)


def latest_own(
    store: StatePersistence, trader_id: str, account: str
) -> StateCheckpoint | None:
    """Newest checkpoint of ``trader_id`` that belongs to ``account``.

    A row belongs to ``account`` when its ``account`` key equals it or is missing
    (legacy); rows of another account (testnet vs live, another key) are skipped.
    A row whose ``node_status`` is not valid JSON counts (it fails closed when
    parsed). Same predicate as the own-id threshold of :func:`newer_other`.
    """
    cursor = store.conn.execute(
        """SELECT * FROM paper_trading_checkpoints
               WHERE trader_id = ?
                 AND CASE WHEN json_valid(node_status)
                          THEN COALESCE(json_extract(node_status, '$.account'), ?)
                          ELSE ? END = ?
               ORDER BY id DESC
               LIMIT 1""",
        (trader_id, account, account, account),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return store.row_to_checkpoint(row)


def latest_in_env(
    store: StatePersistence, trader_id: str, account: str
) -> StateCheckpoint | None:
    """Newest checkpoint of ``trader_id`` in ``account``'s venue + env (any key).

    Ae6ah (D1=A): rotating back A -> B -> A restores the newest row of the env,
    so an older key's own row is never hidden by a row of the other env. Parsed
    strictly (an unreadable row raises, the restore fails closed).
    """
    prefix = account_env(account) + ":"
    cursor = store.conn.execute(
        f"""SELECT * FROM paper_trading_checkpoints
               WHERE {_ENV_ROW_WHERE}
               ORDER BY id DESC
               LIMIT 1""",
        (trader_id, prefix, prefix),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return store.row_to_checkpoint(row)


def newer_other(
    store: StatePersistence, account: str, trader_id: str
) -> StateCheckpoint | None:
    """Newest checkpoint of ANOTHER trader on ``account`` newer than ``trader_id``'s own.

    "Newer" is a higher row id than ``trader_id``'s newest row ON THIS
    ``account`` (0 when it has none; its rows without an ``account`` key count as
    this account, rows of another account do not). Rows whose ``node_status`` is not valid JSON are skipped (their
    account cannot be known; ``json_valid`` makes that independent of scan
    order). The CHOSEN row is parsed strictly: a parse failure raises, so the
    caller fails closed.

    Returns:
        The checkpoint, or None if no other trader on the account is newer.
    """
    cursor = store.conn.execute(
        """SELECT * FROM paper_trading_checkpoints
               WHERE trader_id != ?
                 AND json_valid(node_status)
                 AND json_extract(node_status, '$.account') = ?
                 AND id > (SELECT COALESCE(MAX(id), 0) FROM paper_trading_checkpoints
                           WHERE trader_id = ?
                             AND CASE WHEN json_valid(node_status)
                                      THEN COALESCE(json_extract(node_status, '$.account'), ?)
                                      ELSE ? END = ?)
               ORDER BY id DESC
               LIMIT 1""",
        (trader_id, account, trader_id, account, account, account),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return store.row_to_checkpoint(row)


def has_prior_state(store: StatePersistence, account: str, trader_id: str) -> bool:
    """True if a LIVE start on ``account`` has saved risk state to carry.

    That is any trader's row carrying the account key, this trader's legacy row
    without the key, or a KEY ROTATION: this trader's newest row in the same venue
    + env (any key) is another key, whose risk and halt the restore carries
    (ae6ah, qa42q). The rotation read is the row :func:`latest_in_env`
    returns and the predicate is :func:`is_key_rotation`, the restore's own rule,
    so "the gate asks for consent" and "nothing carries" cannot drift apart
    (vibe-quant-neuvu.2). Rows of another trader whose ``node_status`` is not
    valid JSON are ignored: their account cannot be known, so counting them
    would let an unrelated row satisfy the gate. Never raises for row content.
    """
    cursor = store.conn.execute(
        """SELECT 1 FROM paper_trading_checkpoints
               WHERE (json_valid(node_status)
                      AND json_extract(node_status, '$.account') = ?)
                  OR (trader_id = ?
                      AND CASE WHEN json_valid(node_status)
                               THEN json_extract(node_status, '$.account') IS NULL
                               ELSE 0 END)
               LIMIT 1""",
        (account, trader_id),
    )
    if cursor.fetchone() is not None:
        return True
    prefix = account_env(account) + ":"
    newest = store.conn.execute(
        f"""SELECT CASE WHEN json_valid(node_status)
                           THEN json_extract(node_status, '$.account') END AS account
               FROM paper_trading_checkpoints
               WHERE {_ENV_ROW_WHERE}
               ORDER BY id DESC
               LIMIT 1""",
        (trader_id, prefix, prefix),
    ).fetchone()
    if newest is None:
        return False
    newest_account = newest["account"]
    return isinstance(newest_account, str) and is_key_rotation(newest_account, account)


# ------------------------------------------------------------ restore pieces


def _int_map(value: object) -> dict[str, int]:
    """The int-valued entries of a stored map (a checkpoint round-trips through JSON)."""
    result: dict[str, int] = {}
    if isinstance(value, dict):
        for key, entry in value.items():
            if isinstance(entry, int) and not isinstance(entry, bool):
                result[str(key)] = entry
    return result


def checkpoint_ms(timestamp: datetime) -> int:
    """The epoch milliseconds of a checkpoint row's timestamp (R50b).

    ``StatePersistence`` stores ``datetime('now')``, i.e. NAIVE UTC text
    (persistence.py:39), so the timestamp is taken as UTC and the distance to the
    epoch is computed in whole seconds: reading a naive timestamp through
    ``.timestamp()`` would shift every restored cursor by the host's UTC offset.
    """
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    delta = timestamp.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
    return int(delta.total_seconds()) * 1_000


def restored_watch_state(value: object) -> dict[str, dict[str, int]]:
    """Read ``node_status["liquidation_watch"]`` from a prior checkpoint (R50).

    A missing or malformed payload is an empty state rather than an error: every
    traded symbol then starts at the checkpoint timestamp (R50b), which only ever
    costs a duplicate alert.
    """
    payload = value if isinstance(value, dict) else {}
    return {
        "cursors": _int_map(payload.get("cursors")),
        "seen": _int_map(payload.get("seen")),
    }


def watch_cursors(
    watch: dict[str, dict[str, int]], watch_ms: int, symbols: list[str]
) -> dict[str, int]:
    """The cursors a watcher starts from: the stored ones, plus every traded symbol.

    R50b/R50c: a symbol the checkpoint does not name starts at the checkpoint itself,
    which covers the downtime gap. 0m3tf/1cvw6: stored cursors of symbols this session
    does NOT trade are handed over as well. One rule for the real watcher and for the
    state reported before it exists (R50d), so the two cannot disagree.
    """
    cursors = dict(watch["cursors"])
    for symbol in symbols:
        cursors.setdefault(symbol, watch_ms)
    return cursors


def restored_halt_from_status(node_status: object, checkpoint_time: str) -> RestoredHalt | None:
    """Decide what a prior checkpoint's ``node_status`` restores (D-H1/H22).

    MANUAL/ERROR/SIGNAL/KILL_SWITCH restore HALTED with a
    ``RESTORED_HALT_PREFIX`` message (idempotent across restarts); risk halts are
    left to the restored bookkeeping (they re-trip in ``evaluate_risk``);
    anything unrecognised fails closed as ERROR. ``None`` means start RUNNING.
    """
    if not isinstance(node_status, dict):
        return None
    raw = node_status.get("halt_reason")
    if raw is None:
        return None
    reason: HaltReason | None = None
    if isinstance(raw, str):
        with contextlib.suppress(ValueError):
            reason = HaltReason(raw)
    if reason is not None:
        if reason not in RESTORED_HALTS:
            return None
        text_raw = node_status.get("error_message")
        text = text_raw if isinstance(text_raw, str) and text_raw.strip() else reason.value
        if text.startswith(RESTORED_HALT_PREFIX):
            text = text[len(RESTORED_HALT_PREFIX) :]
        return RestoredHalt(
            reason=reason,
            message=RESTORED_HALT_PREFIX + text,
            checkpoint_time=checkpoint_time,
        )
    return RestoredHalt(
        reason=HaltReason.ERROR,
        message=f"{RESTORED_HALT_PREFIX}unrecognised prior halt_reason {raw!r}",
        checkpoint_time=checkpoint_time,
    )


#: One leading prefix of a stored halt message: this module's own restore prefix or an
#: inherited one. Stripped in a loop so messages do not grow along a chain of sessions.
_HALT_PREFIX_RE = re.compile(rf"(?:{re.escape(RESTORED_HALT_PREFIX)}|inherited from [A-Za-z0-9_-]+: )")


def inherited_halt_message(trader_id: str, message: str) -> str:
    """Message of an ERROR halt inherited from ``trader_id``'s last checkpoint.

    Strips every leading restored/inherited prefix, then prefixes exactly one
    ``inherited from <trader_id>: `` (R14e: no growth along a chain of sessions).
    """
    rest = message
    while match := _HALT_PREFIX_RE.match(rest):
        rest = rest[match.end() :]
    return f"inherited from {trader_id}: {rest}"


@dataclass(frozen=True)
class Inheritance:
    """Everything one start inherits from the saved rows, and nothing else.

    ``unreadable`` is the exception CLASS NAME when the resolve failed closed (the
    node mirrors status HALTED/ERROR for it); every field then holds the fail-closed
    value, including the watch captured before the raise. ``risk_source`` is the
    trader id whose checkpoint supplied the risk when it is not our own.

    ``restore_event`` and ``log_restored`` are deferred: the caller installs the risk,
    halt and watch first, then builds the one restore event (its ``sorted(positions)``
    can raise on a corrupt column) and only then logs, the order the node always had.
    A failure there must leave the installed halt and risk in place.
    """

    risk: RiskState | None
    risk_problems: tuple[str, ...]
    halt: RestoredHalt | None
    risk_source: str | None
    watch: dict[str, dict[str, int]] | None
    watch_ms: int
    restore_event: Callable[[], dict[str, object]] | None
    log_restored: Callable[[], None] | None
    unreadable: str | None


def resolve(
    store: StatePersistence,
    trader_id: str,
    account: str,
    *,
    start_fresh: bool,
    now: Callable[[], datetime],
) -> Inheritance:
    """Resolve what a start on ``account`` inherits, from the rows alone.

    The trader's own latest checkpoint (restart / /restore) is read as before.
    The newest checkpoint of ANOTHER trader on the same account that is newer than
    the own one is ALWAYS looked up. Unless ``start_fresh``, it is the source of
    the risk bookkeeping (high water mark, UTC-day baseline, consecutive losses),
    so a new session id cannot reset the drawdown/daily-loss limits. The
    liquidation watch (forceOrders cursors + dedupe keys) is inherited from that
    newer row regardless of ``start_fresh`` -- it is report-only, never a risk
    limit -- merged with the own row's cursors and bounded by the older of the two
    timestamps, so no venue liquidation between the two sessions is skipped. The
    own watch is captured BEFORE the newer-row lookup, so an unreadable newer row
    (which halts ERROR) never discards it.
    Halts: the own halt always wins; from another session only an ERROR halt is
    inherited (an unrecognised reason counts as ERROR). ``start_fresh`` never
    discards the own checkpoint; it resets only the risk/halt inheritance
    (vibe-quant-1ghod).

    Any raise below fails closed (R18/R19): a blank :class:`RiskState`, an ERROR
    halt carrying the exception class only, and the failed restore event with nulls.
    """
    watch: dict[str, dict[str, int]] | None = None
    watch_ms = 0
    try:
        # P-R7/P-R7b: only a row of THIS account (or a legacy row without the key)
        # is ours; the same trader_id on another account (testnet vs live, another
        # key) carries nothing, but its older matching history is still restored.
        own = latest_own(store, trader_id, account)
        # W5/1ghod: the OWN watch is captured BEFORE the newer-row lookup, so an
        # unreadable newer row still halts ERROR (W3) without discarding the own
        # cursors and dedupe keys -- a duplicate alert is acceptable, a miss is not.
        if own is not None:
            watch_ms = checkpoint_ms(own.timestamp)
            watch = restored_watch_state(own.node_status.get("liquidation_watch"))
        # qa42q/ae6ah: the trader's own NEWEST row in the same venue + env (any
        # key) is a key rotation; its risk and halt carry (below), so a rotation
        # can neither re-arm a halted trader nor reset its drawdown/streak
        # bookkeeping, and rotating back A -> B -> A restores B's newer row.
        latest = latest_in_env(store, trader_id, account)
        rotated = None
        if latest is not None and is_key_rotation(latest.node_status.get("account"), account):
            rotated = latest
            logger.warning(
                "paper %s: key rotation from account %s to %s; "
                "own risk and halt carried",
                trader_id,
                latest.node_status.get("account"),
                account,
            )
        newest = store.load_latest_checkpoint(trader_id)
        if newest is not None:
            newest_account = newest.node_status.get("account")
            if isinstance(newest_account, str) and account_env(newest_account) != account_env(account):
                logger.warning(
                    "paper %s: own checkpoint is for account %s, not %s; ignored",
                    trader_id,
                    newest_account,
                    account,
                )
        newer = newer_other(store, account, trader_id)
        # 1ghod: the lookup ALWAYS runs. ``start_fresh`` only drops the newer row
        # as the RISK/halt source -- the liquidation watch is report-only and is
        # still inherited, so a venue liquidation in the gap is never silent.
        inherited = None if start_fresh else newer
        source = inherited if inherited is not None else own
        # 1zh0m + 1ghod: the watch source is the newer row whatever start_fresh
        # says, merged over the own row captured above. The newer row wins for a
        # symbol both name; a symbol only the own (older) row names keeps ITS
        # cursor, and every symbol neither names starts at the older of the two
        # timestamps, so no venue liquidation in the gap is skipped. With no own row
        # the watch source bounds the gap by itself.
        if newer is not None:
            if own is None:
                watch_ms = checkpoint_ms(newer.timestamp)
            merged = watch if watch is not None else {"cursors": {}, "seen": {}}
            source_watch = restored_watch_state(newer.node_status.get("liquidation_watch"))
            merged["cursors"].update(source_watch["cursors"])
            merged["seen"].update(source_watch["seen"])
            watch = merged
        if rotated is not None:
            # The watch above stays account-scoped; only risk and halt rotate.
            own = rotated
            if inherited is None:
                source = rotated
        if source is None:
            if not start_fresh:
                logger.warning(
                    "paper %s: no prior session for account %s; "
                    "limits start at current equity",
                    trader_id,
                    account,
                )
            return Inheritance(None, (), None, None, watch, watch_ms, None, None, None)
        risk, problems = RiskState.from_checkpoint(source.node_status.get("risk"))
        # P-R8: a new trader id silently arming the old limits would surprise; this
        # is the newer-other-row source only, never a rotation of our own state.
        risk_source = source.trader_id if inherited is not None else None
        halt = (
            restored_halt_from_status(own.node_status, own.timestamp.isoformat())
            if own is not None
            else None
        )
        if halt is None and inherited is not None:
            other = restored_halt_from_status(
                inherited.node_status, inherited.timestamp.isoformat()
            )
            if other is not None and other.reason == HaltReason.ERROR:
                halt = RestoredHalt(
                    reason=other.reason,
                    message=inherited_halt_message(inherited.trader_id, other.message),
                    checkpoint_time=other.checkpoint_time,
                )
    except Exception as exc:
        # R18/R19: an unreadable checkpoint row (failed load / parse, non-dict
        # node_status, or any raise below) fails closed. A non-None risk state and
        # a restored ERROR halt keep BOTH in the failure checkpoint the shutdown
        # writes, so the next restart is HALTED, never a blank RUNNING state (H19).
        # The risk state really is blank here, and the guard re-baselines at the
        # first equity read; only the exception CLASS reaches the log and the event.
        risk = RiskState()
        halt = RestoredHalt(
            reason=HaltReason.ERROR,
            message=(
                f"{RESTORED_HALT_PREFIX}checkpoint unreadable ({type(exc).__name__}); "
                "limits re-baselined at current equity"
            ),
            checkpoint_time=now().isoformat(),
        )
        logger.error(
            "paper %s: checkpoint unreadable (%s); halting",
            trader_id,
            type(exc).__name__,
        )
        return Inheritance(
            risk,
            (),
            halt,
            None,
            watch,
            watch_ms,
            lambda: {
                "action": "restore",
                "checkpoint_time": None,
                "prior_state": None,
                "prior_halt_reason": None,
                "prior_positions": [],
                "prior_orders": [],
                "restored_risk": risk.to_dict(),
                "restored_halt": HaltReason.ERROR.value,
                "risk_problems": [],
                "risk_source_trader_id": None,
                "risk_source_checkpoint_time": None,
            },
            None,
            type(exc).__name__,
        )
    if halt is None and problems:
        # R12/R12m: an unreadable or partly missing risk state restarts HALTED
        # with an ERROR halt through the existing restored-halt path (guard
        # starts HALTED, on_start arms the deferred flatten, the first tick
        # flattens owned+claimed positions and stops strategies). The risk
        # limits are re-baselined at the current equity (the restored values
        # could not be trusted).
        halt = RestoredHalt(
            reason=HaltReason.ERROR,
            message=(
                f"{RESTORED_HALT_PREFIX}unreadable risk state in checkpoint "
                f"({', '.join(problems)}); limits re-baselined at current equity; "
                "owned and claimed positions flattened"
            ),
            checkpoint_time=source.timestamp.isoformat(),
        )
    if problems:
        # R12/R13: one logger.error, values only (never exc_info - this can run
        # under an active fetch error; see _check_connectivity / _write_then_alert).
        logger.error(
            "paper %s: unreadable risk state in checkpoint (%s); halting",
            trader_id,
            ", ".join(problems),
        )
    def log_restored() -> None:
        if risk_source is not None:
            # P-R8: a new trader id silently arming the old limits would surprise.
            logger.warning(
                "paper %s: inherited risk state from %s (high water mark %s, "
                "%d consecutive losses); a limit breach halts at the first equity read; "
                "start fresh skips this other trader id's risk state; "
                "this trader id's own saved limits and halt still carry",
                trader_id,
                risk_source,
                risk.high_water_mark,
                risk.consecutive_losses,
            )
        logger.info(
            "restored risk state for %s from checkpoint %s of %s: %s",
            trader_id,
            source.timestamp.isoformat(),
            source.trader_id,
            risk.to_dict(),
        )

    def restore_event() -> dict[str, object]:
        return {
            "action": "restore",
            "checkpoint_time": own.timestamp.isoformat() if own is not None else None,
            "prior_state": own.node_status.get("state") if own is not None else None,
            "prior_halt_reason": own.node_status.get("halt_reason") if own is not None else None,
            "prior_positions": sorted(own.positions) if own is not None else [],
            "prior_orders": sorted(own.orders) if own is not None else [],
            "restored_risk": risk.to_dict(),
            "restored_halt": halt.reason.value if halt is not None else None,
            "risk_problems": list(problems),
            "risk_source_trader_id": source.trader_id,
            "risk_source_checkpoint_time": source.timestamp.isoformat(),
        }

    return Inheritance(
        risk,
        tuple(problems),
        halt,
        risk_source,
        watch,
        watch_ms,
        restore_event,
        log_restored,
        None,
    )
