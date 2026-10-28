"""A bad command row never blocks or leaks a claim (bead vibe-quant-wo6q8).

Claiming moves rows to ``running`` one UPDATE at a time. A row whose payload cannot be
read must neither abort the claim (a later halt/kill would never run: the risk-officer
veto on the first fix) nor leave earlier UPDATEs open for the connection's next commit.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

from vibe_quant.paper.persistence import PaperCommandQueue

if TYPE_CHECKING:
    from pathlib import Path

_TRADER = "PAPER-007"
_UNREADABLE = "stored command payload unreadable"


def _rows(db: Path) -> list[tuple[str, str, str | None]]:
    """Every row as ``(command, status, error)``, read on a fresh connection."""
    queue = PaperCommandQueue(db)
    try:
        rows = queue.conn.execute(
            "SELECT command, status, error FROM paper_commands ORDER BY id"
        ).fetchall()
        return [(r["command"], r["status"], r["error"]) for r in rows]
    finally:
        queue.close()


def _seed(queue: PaperCommandQueue, commands: list[tuple[str, str | None]]) -> None:
    """Enqueue ``(command, raw payload)``; a raw payload overwrites the stored JSON."""
    for command, raw in commands:
        cid = queue.enqueue(_TRADER, command, {"message": "ok"})
        if raw is not None:
            queue.conn.execute("UPDATE paper_commands SET payload = ? WHERE id = ?", (raw, cid))
    queue.conn.commit()


@pytest.mark.parametrize("claim", ["claim_pending", "claim_startup_halts"])
@pytest.mark.parametrize("raw", ["{bad", "[1]"])
def test_an_unreadable_halt_still_runs_and_never_blocks_a_later_kill(
    tmp_path: Path, claim: str, raw: str
) -> None:
    db = tmp_path / "state.db"
    queue = PaperCommandQueue(db)
    try:
        _seed(queue, [("halt", None), ("halt", raw), ("kill", None)])
        claimed = getattr(queue, claim)(_TRADER)
    finally:
        queue.close()

    assert [(c.command, c.payload) for c in claimed] == [
        ("halt", {"message": "ok"}),
        ("halt", {"message": f"halt via API ({_UNREADABLE})"}),
        ("kill", {"message": "ok"}),
    ]
    assert _rows(db) == [("halt", "running", None)] * 2 + [("kill", "running", None)]


@pytest.mark.parametrize("raw", ["{bad", "", 0, "null", '""', "[1]"])
def test_an_unreadable_resume_is_failed_and_the_rest_claimed(tmp_path: Path, raw: object) -> None:
    """Empty and numeric payloads are corrupt too: none may decode to ``{}``."""
    db = tmp_path / "state.db"
    queue = PaperCommandQueue(db)
    try:
        _seed(queue, [("resume", "{}"), ("kill", None)])
        queue.conn.execute("UPDATE paper_commands SET payload = ? WHERE id = 1", (raw,))
        queue.conn.commit()
        claimed = queue.claim_pending(_TRADER)
    finally:
        queue.close()

    assert [c.command for c in claimed] == ["kill"]
    assert _rows(db) == [("resume", "failed", _UNREADABLE), ("kill", "running", None)]


def test_an_unreadable_close_all_runs_on_its_defaults(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    queue = PaperCommandQueue(db)
    try:
        _seed(queue, [("close_all", "{bad")])
        claimed = queue.claim_pending(_TRADER)
    finally:
        queue.close()

    assert [(c.command, c.payload) for c in claimed] == [
        ("close_all", {"message": f"close_all via API ({_UNREADABLE})"})
    ]


class _FailingCommit:
    """The queue's connection, except the first ``commit`` raises (then delegates)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.failed = False

    def commit(self) -> None:
        if not self.failed:
            self.failed = True
            raise sqlite3.OperationalError("disk I/O error")
        self._conn.commit()

    def __getattr__(self, name: str) -> object:
        return getattr(self._conn, name)


@pytest.mark.parametrize("claim", ["claim_pending", "claim_startup_halts"])
def test_a_failed_commit_rolls_the_claim_back(tmp_path: Path, claim: str) -> None:
    db = tmp_path / "state.db"
    queue = PaperCommandQueue(db)
    try:
        _seed(queue, [("halt", None), ("kill", None)])
        real = queue.conn
        queue._conn = _FailingCommit(real)  # type: ignore[assignment]
        with pytest.raises(sqlite3.OperationalError):
            getattr(queue, claim)(_TRADER)
        queue._conn = real
        queue.enqueue("PAPER-OTHER", "halt")  # the connection's next commit
    finally:
        queue.close()

    assert _rows(db) == [("halt", "pending", None), ("kill", "pending", None), ("halt", "pending", None)]


@pytest.mark.parametrize("claim", ["claim_pending", "claim_startup_halts"])
def test_a_failure_mid_claim_rolls_every_row_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, claim: str
) -> None:
    """Any exception mid-claim (not only sqlite3/RecursionError) rolls every row back.

    Private seam on purpose: no public payload makes the reader raise an arbitrary error.
    """
    db = tmp_path / "state.db"
    queue = PaperCommandQueue(db)
    calls: list[str | None] = []

    def read(raw: str | None) -> dict[str, object]:
        calls.append(raw)
        if len(calls) == 2:
            raise RuntimeError("disk I/O error")
        return {}

    try:
        _seed(queue, [("halt", None), ("kill", None)])
        monkeypatch.setattr(PaperCommandQueue, "_readable_payload", staticmethod(read))
        with pytest.raises(RuntimeError):
            getattr(queue, claim)(_TRADER)
        queue.enqueue("PAPER-OTHER", "halt")  # the connection's next commit
    finally:
        queue.close()

    assert _rows(db) == [("halt", "pending", None), ("kill", "pending", None), ("halt", "pending", None)]
