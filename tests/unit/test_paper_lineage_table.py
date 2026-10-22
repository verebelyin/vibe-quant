"""The checkpoint restore lineage table (vibe-quant-rabmo.1a, group C).

One characterization table for ``PaperTradingNode._restore_from_checkpoint``: for
every rotation / env / corrupt-row seed of ``test_paper_rotation_consent.CASES``
(G1-G19) it pins the restored risk, the restored halt, the source trader, the
liquidation watch and its gap boundary, the FULL restore-event payload, the node
status and the consent gate. The W1-W5 rows pin the watch merge on its own: the
own cursors survive, the newer row wins a shared symbol, the gap boundary is the
own timestamp (the newer one when there is no own row), ``start_fresh`` still
inherits the watch, and an unreadable newer row keeps the own watch while it
fails closed.

Characterization only: no production code changes. Every row tests the base
(swarm/job-rabmo) as it is, so rabmo.1b/c can move the code and keep this table
green. Rows use the fixture's own seed helpers; timestamps are normalized after
seeding so every ISO string and epoch millisecond below is a literal.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from tests.unit.test_paper_halt_restore import seeded  # noqa: F401  (fixture)
from tests.unit.test_paper_risk_inherit import _put, _risk, _sql
from tests.unit.test_paper_rotation_consent import (
    CASES,
    K_NEW,
    TRADER,
    _restore,
    _seed,
)
from vibe_quant.paper import lineage
from vibe_quant.paper.guard import HaltReason, RestoredHalt
from vibe_quant.paper.node import NodeState
from vibe_quant.paper.persistence import StatePersistence

#: ``_ESCAPED``: every seeded CASES row is stamped by row id, one second apart, so
#: the restore event's ISO strings and the watch's epoch milliseconds are literals.
_STAMP = "2026-01-01 00:00:{second:02d}"
_STAMP_MS = 1_767_225_600_000
_TS = ("2026-01-01T00:00:00", "2026-01-01T00:00:01", "2026-01-01T00:00:02")
_MS = (_STAMP_MS, _STAMP_MS + 1_000, _STAMP_MS + 2_000)
_ERROR = "restored after restart: checkpoint unreadable ({kind}); limits re-baselined at current equity"


def _expected_risk(high_water_mark: str) -> dict[str, object]:
    """``RiskState.to_dict()`` of the fixture's clean risk payload (hwm overridable)."""
    mark = Decimal(high_water_mark)
    day_start = Decimal("10400")
    last = Decimal("10300")
    return {
        "high_water_mark": high_water_mark,
        "day": datetime.now(UTC).date().isoformat(),
        "day_start_equity": "10400",
        "consecutive_losses": 3,
        "last_equity": "10300",
        "drawdown_pct": str((mark - last) / mark),
        "daily_loss_pct": str((day_start - last) / day_start),
    }


def _empty_risk() -> dict[str, object]:
    """``RiskState().to_dict()``: the fail-closed blank state."""
    return {
        "high_water_mark": None,
        "day": None,
        "day_start_equity": None,
        "consecutive_losses": 0,
        "last_equity": None,
        "drawdown_pct": None,
        "daily_loss_pct": None,
    }


def _stamp_rows(db: Any) -> None:
    """Stamp the seeded rows by id: id 1 -> second 0, id 2 -> second 1, ..."""
    con = sqlite3.connect(db)
    try:
        table = con.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'paper_trading_checkpoints'"
        ).fetchone()
        ids = (
            [row[0] for row in con.execute("SELECT id FROM paper_trading_checkpoints ORDER BY id")]
            if table is not None
            else []
        )
    finally:
        con.close()
    for second, row_id in enumerate(ids):
        _sql(
            db,
            "UPDATE paper_trading_checkpoints SET created_at = ? WHERE id = ?",
            _STAMP.format(second=second),
            row_id,
        )


def _event(
    *,
    checkpoint: str | None,
    prior_state: str | None,
    prior_halt: str | None,
    risk: dict[str, object],
    halt: str | None,
    source: str | None,
    source_ts: str | None,
) -> dict[str, object]:
    """The full restore-event payload, every key of the one write on that path."""
    return {
        "action": "restore",
        "checkpoint_time": checkpoint,
        "prior_state": prior_state,
        "prior_halt_reason": prior_halt,
        "prior_positions": [],
        "prior_orders": [],
        "restored_risk": risk,
        "restored_halt": halt,
        "risk_problems": [],
        "risk_source_trader_id": source,
        "risk_source_checkpoint_time": source_ts,
    }


_FAILED_EVENT = _event(
    checkpoint=None,
    prior_state=None,
    prior_halt=None,
    risk=_empty_risk(),
    halt="error",
    source=None,
    source_ts=None,
)


@dataclass(frozen=True)
class Expected:
    """Everything ``_restore_from_checkpoint`` must leave behind for one seed."""

    risk: dict[str, object] | None = None
    halt: RestoredHalt | None = None
    halt_error: str | None = None  # fail-closed message; its checkpoint_time is the clock
    inherited: str | None = None
    watch: dict[str, dict[str, int]] | None = None
    watch_ms: int = 0
    event: dict[str, object] | None = None
    state: NodeState = NodeState.INITIALIZING
    gate: bool = False


def _manual_halt(second: int) -> RestoredHalt:
    return RestoredHalt(
        reason=HaltReason.MANUAL,
        message="restored after restart: halt via API",
        checkpoint_time=_TS[second],
    )


_EXPECTED: dict[str, Expected] = {
    # No row on this account: nothing to restore, no event.
    "G1": Expected(),
    # Own row on the new key.
    "G2": Expected(
        risk=_expected_risk("10600"),
        watch={"cursors": {}, "seen": {}},
        watch_ms=_MS[0],
        event=_event(
            checkpoint=_TS[0],
            prior_state="stopped",
            prior_halt=None,
            risk=_expected_risk("10600"),
            halt=None,
            source=TRADER,
            source_ts=_TS[0],
        ),
        gate=True,
    ),
    # Own old-key row only: its risk and halt rotate in, its watch does not.
    "G3": Expected(
        risk=_expected_risk("10600"),
        halt=_manual_halt(0),
        event=_event(
            checkpoint=_TS[0],
            prior_state="halted",
            prior_halt="manual",
            risk=_expected_risk("10600"),
            halt="manual",
            source=TRADER,
            source_ts=_TS[0],
        ),
        state=NodeState.HALTED,
        gate=True,
    ),
    # Own old-key row under a newer testnet row: the rotation still carries.
    "G4": Expected(
        risk=_expected_risk("10600"),
        halt=_manual_halt(0),
        event=_event(
            checkpoint=_TS[0],
            prior_state="halted",
            prior_halt="manual",
            risk=_expected_risk("10600"),
            halt="manual",
            source=TRADER,
            source_ts=_TS[0],
        ),
        state=NodeState.HALTED,
        gate=True,
    ),
    # Own testnet row only: a live start sees nothing.
    "G5": Expected(),
    # Another trader's old-key row: not on this account, nothing to carry.
    "G6": Expected(),
    # Another trader on the new key: no own row, its risk is the source.
    "G7": Expected(
        risk=_expected_risk("10600"),
        inherited="PAPER-041",
        watch={"cursors": {}, "seen": {}},
        watch_ms=_MS[0],
        event=_event(
            checkpoint=None,
            prior_state=None,
            prior_halt=None,
            risk=_expected_risk("10600"),
            halt=None,
            source="PAPER-041",
            source_ts=_TS[0],
        ),
        gate=True,
    ),
    # Own legacy row without an account key is ours.
    "G8": Expected(
        risk=_expected_risk("10600"),
        watch={"cursors": {}, "seen": {}},
        watch_ms=_MS[0],
        event=_event(
            checkpoint=_TS[0],
            prior_state="stopped",
            prior_halt=None,
            risk=_expected_risk("10600"),
            halt=None,
            source=TRADER,
            source_ts=_TS[0],
        ),
        gate=True,
    ),
    # Newest own row is not JSON: fail closed before the watch is captured.
    "G9": Expected(
        risk=_empty_risk(),
        halt_error=_ERROR.format(kind="JSONDecodeError"),
        event=_FAILED_EVENT,
        state=NodeState.HALTED,
    ),
    # Newest own row is JSON but not an object: the watch read raises AFTER the
    # gap boundary was set from its timestamp, so watch_ms survives and watch does not.
    "G10": Expected(
        risk=_empty_risk(),
        halt_error=_ERROR.format(kind="AttributeError"),
        watch_ms=_MS[0],
        event=_FAILED_EVENT,
        state=NodeState.HALTED,
        gate=True,
    ),
    # Newest own row has a non-string account: the rotation predicate raises.
    "G11": Expected(
        risk=_empty_risk(),
        halt_error=_ERROR.format(kind="TypeError"),
        event=_FAILED_EVENT,
        state=NodeState.HALTED,
    ),
    # Own old-key halt + another trader's newer row: other risk, own halt.
    "G12": Expected(
        risk=_expected_risk("12000"),
        halt=_manual_halt(0),
        inherited="PAPER-041",
        watch={"cursors": {}, "seen": {}},
        watch_ms=_MS[1],
        event=_event(
            checkpoint=_TS[0],
            prior_state="halted",
            prior_halt="manual",
            risk=_expected_risk("12000"),
            halt="manual",
            source="PAPER-041",
            source_ts=_TS[1],
        ),
        state=NodeState.HALTED,
        gate=True,
    ),
    # Rotate back: the OLDER key's newer row wins (ae6ah, D1=A).
    "G13": Expected(
        risk=_expected_risk("10600"),
        halt=_manual_halt(1),
        watch={"cursors": {}, "seen": {}},
        watch_ms=_MS[0],
        event=_event(
            checkpoint=_TS[1],
            prior_state="halted",
            prior_halt="manual",
            risk=_expected_risk("10600"),
            halt="manual",
            source=TRADER,
            source_ts=_TS[1],
        ),
        state=NodeState.HALTED,
        gate=True,
    ),
    # The same rotation with a stopped old-key row: risk carries, no halt.
    "G14": Expected(
        risk=_expected_risk("10600"),
        watch={"cursors": {}, "seen": {}},
        watch_ms=_MS[0],
        event=_event(
            checkpoint=_TS[1],
            prior_state="stopped",
            prior_halt=None,
            risk=_expected_risk("10600"),
            halt=None,
            source=TRADER,
            source_ts=_TS[1],
        ),
        gate=True,
    ),
    # Own newest row on this key wins over the older same-env row.
    "G15": Expected(
        risk=_expected_risk("10000"),
        watch={"cursors": {}, "seen": {}},
        watch_ms=_MS[1],
        event=_event(
            checkpoint=_TS[1],
            prior_state="stopped",
            prior_halt=None,
            risk=_expected_risk("10000"),
            halt=None,
            source=TRADER,
            source_ts=_TS[1],
        ),
        gate=True,
    ),
    # Another trader's newer row supplies the risk, the own rotated halt wins.
    "G16": Expected(
        risk=_expected_risk("12000"),
        halt=_manual_halt(2),
        inherited="PAPER-041",
        watch={"cursors": {}, "seen": {}},
        watch_ms=_MS[0],
        event=_event(
            checkpoint=_TS[2],
            prior_state="halted",
            prior_halt="manual",
            risk=_expected_risk("12000"),
            halt="manual",
            source="PAPER-041",
            source_ts=_TS[1],
        ),
        state=NodeState.HALTED,
        gate=True,
    ),
    # start_fresh skips the other trader's risk: the own rotation carries.
    "G16-fresh": Expected(
        risk=_expected_risk("10600"),
        halt=_manual_halt(2),
        watch={"cursors": {}, "seen": {}},
        watch_ms=_MS[0],
        event=_event(
            checkpoint=_TS[2],
            prior_state="halted",
            prior_halt="manual",
            risk=_expected_risk("10600"),
            halt="manual",
            source=TRADER,
            source_ts=_TS[2],
        ),
        state=NodeState.HALTED,
        gate=True,
    ),
    # Newer own row with a non-string account fails closed; the own watch survives.
    "G17": Expected(
        risk=_empty_risk(),
        halt_error=_ERROR.format(kind="TypeError"),
        watch={"cursors": {}, "seen": {}},
        watch_ms=_MS[0],
        event=_FAILED_EVENT,
        state=NodeState.HALTED,
        gate=True,
    ),
    # Newest own row is not JSON: the first load raises, before the watch.
    "G18": Expected(
        risk=_empty_risk(),
        halt_error=_ERROR.format(kind="JSONDecodeError"),
        event=_FAILED_EVENT,
        state=NodeState.HALTED,
        gate=True,
    ),
    # Newest own row is legacy (no account): it is the source, no rotation.
    "G19": Expected(
        risk=_expected_risk("11000"),
        watch={"cursors": {}, "seen": {}},
        watch_ms=_MS[1],
        event=_event(
            checkpoint=_TS[1],
            prior_state="stopped",
            prior_halt=None,
            risk=_expected_risk("11000"),
            halt=None,
            source=TRADER,
            source_ts=_TS[1],
        ),
        gate=True,
    ),
}


def _check_restore(
    node: Any,
    events: list[dict[str, object]],
    persistence: Any,
    expected: Expected,
) -> None:
    if expected.risk is None:
        assert node._restored_risk is None
    else:
        assert node._restored_risk is not None
        assert node._restored_risk.to_dict() == expected.risk
    assert node._risk_inherited_from == expected.inherited
    if expected.halt is not None:
        assert node._restored_halt == expected.halt
    elif expected.halt_error is not None:
        assert node._restored_halt is not None
        assert node._restored_halt.reason == HaltReason.ERROR
        assert node._restored_halt.message == expected.halt_error
        assert datetime.fromisoformat(node._restored_halt.checkpoint_time).tzinfo is not None
    else:
        assert node._restored_halt is None
    assert node._restored_watch == expected.watch
    assert node._restored_watch_ms == expected.watch_ms
    assert events == ([] if expected.event is None else [expected.event])
    assert node.status.state == expected.state
    halted = expected.halt.reason if expected.halt is not None else None
    if expected.halt_error is not None:
        halted = HaltReason.ERROR
    assert node.status.halt_reason == halted
    message = expected.halt.message if expected.halt is not None else expected.halt_error
    assert node.status.error_message == message
    assert lineage.has_prior_state(persistence, K_NEW, TRADER) is expected.gate


def test_fresh_account_restores_nothing_and_writes_no_event(
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """G1: no prior row at all -> no restore event, no risk, no halt, no watch."""
    _seed(seeded["db"], CASES["G1"].rows)

    node, events, persistence = _restore(seeded)
    try:
        assert events == []
        assert node._restored_risk is None
        assert node._restored_halt is None
        assert node._risk_inherited_from is None
        assert node._restored_watch is None
        assert node._restored_watch_ms == 0
        assert node.status.state == NodeState.INITIALIZING
        assert node.status.error_message is None
        assert lineage.has_prior_state(persistence, K_NEW, TRADER) is False
    finally:
        persistence.close()
        node._state_manager.close()


@pytest.mark.parametrize("name", list(_EXPECTED))
def test_rotation_and_corrupt_rows_pin_the_restore_lineage(
    name: str,
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """G1-G19 through ``_restore_from_checkpoint``: risk, halt, source, watch, event."""
    case = CASES[name]
    _seed(seeded["db"], case.rows)
    _stamp_rows(seeded["db"])

    node, events, persistence = _restore(seeded, start_fresh=case.start_fresh)
    try:
        _check_restore(node, events, persistence, _EXPECTED[name])
    finally:
        persistence.close()
        node._state_manager.close()


def _check_inheritance(inheritance: lineage.Inheritance, expected: Expected) -> None:
    """The same expectations, read off one ``Inheritance`` instead of a node."""
    if expected.risk is None:
        assert inheritance.risk is None
    else:
        assert inheritance.risk is not None
        assert inheritance.risk.to_dict() == expected.risk
    assert inheritance.risk_source == expected.inherited
    if expected.halt is not None:
        assert inheritance.halt == expected.halt
    elif expected.halt_error is not None:
        assert inheritance.halt is not None
        assert inheritance.halt.reason == HaltReason.ERROR
        assert inheritance.halt.message == expected.halt_error
        assert datetime.fromisoformat(inheritance.halt.checkpoint_time).tzinfo is not None
    else:
        assert inheritance.halt is None
    assert inheritance.watch == expected.watch
    assert inheritance.watch_ms == expected.watch_ms
    assert inheritance.restore_event == expected.event
    if expected.halt_error is not None:
        # Fail closed: the class NAME of the raise, never its text.
        assert inheritance.unreadable == expected.halt_error.split("(")[1].split(")")[0]
    else:
        assert inheritance.unreadable is None


@pytest.mark.parametrize("name", list(_EXPECTED))
def test_resolve_reaches_the_same_lineage_without_a_node(
    name: str,
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """G1-G19 through ``lineage.resolve()`` alone: no node, the same row-for-row result."""
    case = CASES[name]
    _seed(seeded["db"], case.rows)
    _stamp_rows(seeded["db"])

    persistence = StatePersistence(seeded["db"], trader_id=TRADER)
    try:
        inheritance = lineage.resolve(
            persistence,
            TRADER,
            K_NEW,
            start_fresh=case.start_fresh,
            now=lambda: datetime.now(UTC),
        )
    finally:
        persistence.close()

    _check_inheritance(inheritance, _EXPECTED[name])


# ------------------------------------------------------- W1-W5: watch merge


def _own_row(
    db: Any,
    *,
    cursors: dict[str, int],
    seen: dict[str, int],
    stamp: str,
    high_water_mark: str = "10600",
) -> None:
    row_id = _put(
        db,
        TRADER,
        account=K_NEW,
        risk=_risk(high_water_mark=high_water_mark),
        extra={"liquidation_watch": {"cursors": cursors, "seen": seen}},
    )
    _sql(db, "UPDATE paper_trading_checkpoints SET created_at = ? WHERE id = ?", stamp, row_id)


def _other_row(
    db: Any,
    *,
    cursors: dict[str, int],
    seen: dict[str, int],
    stamp: str,
    high_water_mark: str = "12000",
) -> None:
    row_id = _put(
        db,
        "PAPER-041",
        account=K_NEW,
        risk=_risk(high_water_mark=high_water_mark),
        extra={"liquidation_watch": {"cursors": cursors, "seen": seen}},
    )
    _sql(db, "UPDATE paper_trading_checkpoints SET created_at = ? WHERE id = ?", stamp, row_id)


_OWN_STAMP = "2026-01-01 00:00:03"
_OWN_MS = _STAMP_MS + 3_000
_OTHER_STAMP = "2026-01-01 00:00:07"
_OTHER_MS = _STAMP_MS + 7_000
_OWN_WATCH = {"cursors": {"BTCUSDT": 100}, "seen": {"k": 1}}
_OTHER_WATCH = {"cursors": {"BTCUSDT": 200, "ETHUSDT": 50}, "seen": {"j": 2}}
_MERGED_WATCH = {"cursors": {"BTCUSDT": 200, "ETHUSDT": 50}, "seen": {"k": 1, "j": 2}}


def _watch_event(
    *,
    checkpoint: str | None,
    risk: dict[str, object],
    source: str | None,
    source_ts: str | None,
) -> dict[str, object]:
    return _event(
        checkpoint=checkpoint,
        prior_state="stopped" if checkpoint is not None else None,
        prior_halt=None,
        risk=risk,
        halt=None,
        source=source,
        source_ts=source_ts,
    )


def test_w1_own_watch_only(seeded: dict[str, Any]) -> None:  # noqa: F811
    """W1: own watch survives unchanged, bounded by the own timestamp."""
    _own_row(seeded["db"], cursors={"BTCUSDT": 100}, seen={"k": 1}, stamp=_OWN_STAMP)

    node, events, persistence = _restore(seeded)
    try:
        _check_restore(
            node,
            events,
            persistence,
            Expected(
                risk=_expected_risk("10600"),
                watch=_OWN_WATCH,
                watch_ms=_OWN_MS,
                event=_watch_event(
                    checkpoint="2026-01-01T00:00:03",
                    risk=_expected_risk("10600"),
                    source=TRADER,
                    source_ts="2026-01-01T00:00:03",
                ),
                gate=True,
            ),
        )
    finally:
        persistence.close()
        node._state_manager.close()


def test_w2_newer_other_row_wins_a_shared_symbol(seeded: dict[str, Any]) -> None:  # noqa: F811
    """W2: newer cursors win shared symbols, own-only symbols keep theirs, gap = own."""
    _own_row(seeded["db"], cursors={"BTCUSDT": 100}, seen={"k": 1}, stamp=_OWN_STAMP)
    _other_row(
        seeded["db"],
        cursors={"BTCUSDT": 200, "ETHUSDT": 50},
        seen={"j": 2},
        stamp=_OTHER_STAMP,
    )

    node, events, persistence = _restore(seeded)
    try:
        _check_restore(
            node,
            events,
            persistence,
            Expected(
                risk=_expected_risk("12000"),
                inherited="PAPER-041",
                watch=_MERGED_WATCH,
                watch_ms=_OWN_MS,
                event=_watch_event(
                    checkpoint="2026-01-01T00:00:03",
                    risk=_expected_risk("12000"),
                    source="PAPER-041",
                    source_ts="2026-01-01T00:00:07",
                ),
                gate=True,
            ),
        )
    finally:
        persistence.close()
        node._state_manager.close()


def test_w3_no_own_row_bounds_the_watch_by_the_newer_timestamp(
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """W3: with no own row the newer row's cursor state and timestamp stand alone."""
    _other_row(
        seeded["db"],
        cursors={"BTCUSDT": 200, "ETHUSDT": 50},
        seen={"j": 2},
        stamp=_OTHER_STAMP,
    )

    node, events, persistence = _restore(seeded)
    try:
        _check_restore(
            node,
            events,
            persistence,
            Expected(
                risk=_expected_risk("12000"),
                inherited="PAPER-041",
                watch=_OTHER_WATCH,
                watch_ms=_OTHER_MS,
                event=_watch_event(
                    checkpoint=None,
                    risk=_expected_risk("12000"),
                    source="PAPER-041",
                    source_ts="2026-01-01T00:00:07",
                ),
                gate=True,
            ),
        )
    finally:
        persistence.close()
        node._state_manager.close()


def test_w4_start_fresh_still_merges_the_watch(seeded: dict[str, Any]) -> None:  # noqa: F811
    """W4: start_fresh keeps the own risk source but still inherits the watch."""
    _own_row(seeded["db"], cursors={"BTCUSDT": 100}, seen={"k": 1}, stamp=_OWN_STAMP)
    _other_row(
        seeded["db"],
        cursors={"BTCUSDT": 200, "ETHUSDT": 50},
        seen={"j": 2},
        stamp=_OTHER_STAMP,
    )

    node, events, persistence = _restore(seeded, start_fresh=True)
    try:
        _check_restore(
            node,
            events,
            persistence,
            Expected(
                risk=_expected_risk("10600"),
                watch=_MERGED_WATCH,
                watch_ms=_OWN_MS,
                event=_watch_event(
                    checkpoint="2026-01-01T00:00:03",
                    risk=_expected_risk("10600"),
                    source=TRADER,
                    source_ts="2026-01-01T00:00:03",
                ),
                gate=True,
            ),
        )
    finally:
        persistence.close()
        node._state_manager.close()


def test_w5_unreadable_newer_row_keeps_the_own_watch(
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """W5: a garbage newer timestamp fails closed but never discards the own cursors."""
    _own_row(seeded["db"], cursors={"BTCUSDT": 100}, seen={"k": 1}, stamp=_OWN_STAMP)
    _other_row(
        seeded["db"],
        cursors={"BTCUSDT": 200, "ETHUSDT": 50},
        seen={"j": 2},
        stamp="garbage",
    )

    node, events, persistence = _restore(seeded)
    try:
        _check_restore(
            node,
            events,
            persistence,
            Expected(
                risk=_empty_risk(),
                halt_error=_ERROR.format(kind="ValueError"),
                watch=_OWN_WATCH,
                watch_ms=_OWN_MS,
                event=_FAILED_EVENT,
                state=NodeState.HALTED,
                gate=True,
            ),
        )
    finally:
        persistence.close()
        node._state_manager.close()
