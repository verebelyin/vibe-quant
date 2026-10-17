"""Stale ``running`` paper commands are failed when the node starts (bead vibe-quant-jf7c3).

A claimed command row only leaves ``running`` from the node that claimed it. A node
killed mid-command leaves its rows ``running`` forever, so the API polls an outcome
that can never arrive. J2: ``_initialize`` sweeps THIS trader's rows through the
command queue's own worker before anything else touches the queue, and a sweep that
raises is logged by class only - it must never block a start.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING, Any

import pytest  # noqa: TC002  # annotations only; no runtime pytest API in this file

from tests.unit.test_audit_paper_node import _config, _FakeTradingNode
from tests.unit.test_paper_kill_switch_fail_closed import (
    fake_tradingnode,  # noqa: F401  # fixture
    seeded,  # noqa: F401  # fixture
)
from vibe_quant.paper.node import PaperTradingNode
from vibe_quant.paper.persistence import COMMAND_FAILED, PaperCommandQueue

if TYPE_CHECKING:
    from pathlib import Path

_TRADER_ID = "PAPER-007"
_OTHER_TRADER = "PAPER-999"
#: J2: what a swept row carries - the outcome it was waiting for can never arrive.
_STALE_TEXT = "node restarted; command outcome unknown (was running when the node stopped)"
#: Leak marker: contains none of the redacted credential names, so a mutant that drops
#: a name cannot hide it (impl-sb-watchlog-2).
_LEAK = "ZZLEAK9"


def _row(db: Path, cid: int) -> tuple[str, str | None]:
    """``(status, error)`` of one command row, read on its own short-lived connection."""
    queue = PaperCommandQueue(db)
    try:
        cmd = queue.get(cid)
    finally:
        queue.close()
    assert cmd is not None
    return cmd.status, cmd.error


def _seed_running(db: Path, trader_id: str, count: int = 1) -> list[int]:
    """Leave ``count`` command rows ``running``, the state a killed process leaves behind."""
    queue = PaperCommandQueue(db)
    try:
        cids = [queue.enqueue(trader_id, "halt") for _ in range(count)]
        assert [c.id for c in queue.claim_pending(trader_id)] == cids
        return cids
    finally:
        queue.close()


# ------------------------------------------------------------------ J1: the sweep


def test_fail_stale_running_fails_only_this_traders_running_rows(tmp_path: Path) -> None:
    """J1: one UPDATE over this trader's ``running`` rows, committed; another trader's
    running row and this trader's ``pending`` row are untouched, and the return value is
    the rowcount (0 when nothing is left running)."""
    db = tmp_path / "state.db"
    queue = PaperCommandQueue(db)
    try:
        mine = queue.enqueue(_TRADER_ID, "halt")
        theirs = queue.enqueue(_OTHER_TRADER, "halt")
        assert [c.id for c in queue.claim_pending(_TRADER_ID)] == [mine]
        assert [c.id for c in queue.claim_pending(_OTHER_TRADER)] == [theirs]
        # Enqueued after the claim, so it is still pending when the sweep runs.
        left_pending = queue.enqueue(_TRADER_ID, "pause")

        assert queue.fail_stale_running(_TRADER_ID, _STALE_TEXT) == 1
        assert _row(db, mine) == (COMMAND_FAILED, _STALE_TEXT)
        assert _row(db, left_pending) == ("pending", None)
        assert _row(db, theirs) == ("running", None)
        # The sweep already took every running row of this trader: nothing left to do.
        assert queue.fail_stale_running(_TRADER_ID, _STALE_TEXT) == 0
        assert _row(db, left_pending) == ("pending", None)
    finally:
        queue.close()


# ------------------------------------------------------- J2/J3: at node startup


@pytest.mark.parametrize("stale_count", [1, 2])
async def test_initialize_fails_stale_running_commands(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    caplog: pytest.LogCaptureFixture,
    stale_count: int,
) -> None:
    """J2/J3, node fixture: ``seeded`` + ``fake_tradingnode`` from
    ``tests/unit/test_paper_kill_switch_fail_closed.py``. Rows left ``running`` by a dead
    process are failed with the fixed text when the node starts, and one WARNING carries
    the count - the count 1 case is the equality edge of ``>= 1``."""
    db = seeded["db"]
    cids = _seed_running(db, _TRADER_ID, stale_count)
    node = PaperTradingNode(_config(seeded))
    with caplog.at_level(logging.WARNING):
        await node._initialize()
    try:
        assert [_row(db, cid) for cid in cids] == [(COMMAND_FAILED, _STALE_TEXT)] * stale_count
    finally:
        await node._shutdown()
    warnings = [
        r.getMessage() for r in caplog.records if "stale running paper command" in r.getMessage()
    ]
    assert warnings == [f"failed {stale_count} stale running paper command(s) at startup"]


async def test_initialize_without_stale_rows_logs_no_warning(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    """J3 edge: zero swept rows -> no count warning at all (the warning is gated on
    ``>= 1``)."""
    node = PaperTradingNode(_config(seeded))
    with caplog.at_level(logging.WARNING):
        await node._initialize()
    try:
        assert node._commands is not None
    finally:
        await node._shutdown()
    assert [
        r.getMessage() for r in caplog.records if "stale running paper command" in r.getMessage()
    ] == []


async def test_initialize_continues_when_the_sweep_raises(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """J2: a sweep that raises is logged by class only and never blocks the start."""
    db = seeded["db"]
    cids = _seed_running(db, _TRADER_ID)

    def boom(self: PaperCommandQueue, trader_id: str, message: str) -> int:
        raise sqlite3.OperationalError(f"sweep {_LEAK}")

    monkeypatch.setattr(PaperCommandQueue, "fail_stale_running", boom)
    node = PaperTradingNode(_config(seeded))
    with caplog.at_level(logging.ERROR):
        await node._initialize()
    try:
        assert node._event_writer is not None, "startup continued past the failed sweep"
        assert node._commands is not None
        assert _row(db, cids[0]) == ("running", None), "the failed sweep left the row alone"
    finally:
        await node._shutdown()
    errors = [
        r.getMessage() for r in caplog.records if "stale command sweep failed" in r.getMessage()
    ]
    assert errors == ["stale command sweep failed (OperationalError)"]
    assert _LEAK not in caplog.text
    assert all(r.exc_info is None for r in caplog.records)
