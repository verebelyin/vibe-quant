"""A missing ``system_state`` row is re-created ENGAGED (bead vibe-quant-tdjop, task T1).

Before this change ``init_schema`` seeded the singleton row with
``INSERT OR IGNORE INTO system_state (id, kill_switch) VALUES (1, 0)``, so a
deleted row came back disengaged on every fresh connection, ``set_kill_switch``
was UPDATE-only (a silent no-op on a missing row) and both ``/api/system/status``
and the header widget showed the switch unlocked. A deleted kill-switch record
is now treated as evidence of tampering: it is re-created ENGAGED with the
reason below, logged once at CRITICAL, and every reader (paper API refusal, node
poll, header) sees the halt.

Contract rows:

* R1  new DB file, first ``get_system_state()`` -> ``kill_switch False``,
      ``reason None``; no critical log.
* R2  existing DB, row deleted, NEW StateManager -> row
      ``(1, 1, <reason>, now, "system")``; one critical log; a second fresh
      StateManager logs nothing.
* R3  row deleted under a cached conn -> the same row, returned engaged.
* R3b reseed INSERT raises ``OperationalError`` -> ``SystemStateMissingError``
      with the exact "could not be re-created" text.
* R3c ``require_row=True`` and ``require_row=False`` are accepted and ignored:
      both return the same (engaged) result.
* R4  ``set_kill_switch("r", "op")`` on a missing row upserts it.
* R5  ``clear_kill_switch("op")`` on a missing row re-seeds it engaged and
      raises with the "unlock again" text.
* R6  GET /status with the row missing -> 200, engaged, the R2 reason.
* R6b GET /status raising ``SystemStateMissingError`` -> 503
      ``"kill switch state unavailable: <msg>"``.
* R7  POST /kill when the write raises ``OperationalError`` -> the cascade is
      still enqueued and the response is 503 naming the class and the count.
* R7b POST /unlock with the row missing -> 409 with the R5 text, row engaged.
* R8  POST /api/paper/start, row deleted, fresh StateManager -> 423 with the
      R2 reason in the existing "System kill switch engaged" text.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.schema import SYSTEM_STATE_RESEEDED_REASON, SYSTEM_STATE_UNLOCK_ACK_BY
from vibe_quant.db.state_manager import StateManager, SystemStateMissingError
from vibe_quant.jobs.manager import BacktestJobManager
from vibe_quant.paper.persistence import PaperCommandQueue

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path
    from typing import Any

REASON = SYSTEM_STATE_RESEEDED_REASON
_UNRESEEDABLE = "system_state row id=1 is missing and could not be re-created"
_CLEAR_MISSING = "system_state row id=1 was missing; re-created engaged - unlock again"
_TRADER = "PAPER-KS"

DSL = {
    "name": "ks_strategy",
    "timeframe": "1h",
    "indicators": {"rsi_14": {"type": "RSI", "period": 14}},
    "entry_conditions": {"long": ["rsi_14 < 30"]},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 4.0},
}


def _read_row(state: StateManager) -> tuple[Any, ...]:
    """``(id, kill_switch, reason, killed_at, killed_by)`` of the singleton row."""
    row = state.conn.execute(
        "SELECT id, kill_switch, reason, killed_at, killed_by FROM system_state WHERE id = 1"
    ).fetchone()
    assert row is not None, "the singleton row must exist"
    return tuple(row)


def _delete_row(state: StateManager) -> None:
    """Delete the singleton row through ``state``'s own (cached) connection."""
    _ = state.conn  # open + init schema before deleting the row
    state.conn.execute("DELETE FROM system_state")
    state.conn.commit()
    count = state.conn.execute("SELECT COUNT(*) FROM system_state").fetchone()[0]
    assert count == 0, "the row must really be gone for the read under test"


def _critical_records(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.CRITICAL]


# --------------------------------------------------------------- R1


def test_fresh_db_seeds_disengaged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """R1: a brand-new database gets a disengaged row and no critical log."""
    with caplog.at_level(logging.CRITICAL):
        sm = StateManager(tmp_path / "state.db")
        try:
            state = sm.get_system_state()
        finally:
            sm.close()
    assert state["kill_switch"] is False
    assert state["reason"] is None
    assert _critical_records(caplog) == []


# --------------------------------------------------------------- R2


def test_fresh_manager_reseeds_engaged_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """R2: a fresh StateManager on a DB whose row was deleted re-creates it
    ENGAGED (reason + killed_at + killed_by='system') and logs CRITICAL once;
    the next fresh StateManager finds the row present and logs nothing."""
    db = tmp_path / "state.db"
    first = StateManager(db)
    try:
        _delete_row(first)
    finally:
        first.close()

    caplog.clear()
    with caplog.at_level(logging.CRITICAL):
        second = StateManager(db)
        try:
            state = second.get_system_state()
            row = _read_row(second)
        finally:
            second.close()

    assert state["kill_switch"] is True
    assert state["reason"] == REASON
    assert row[0] == 1
    assert row[1] == 1
    assert row[2] == REASON
    assert row[3] is not None, "killed_at is stamped when the row is re-created engaged"
    assert row[4] == "system"
    assert _critical_records(caplog) == ["system_state row was missing; re-created ENGAGED"]

    caplog.clear()
    with caplog.at_level(logging.CRITICAL):
        third = StateManager(db)
        try:
            again = third.get_system_state()
        finally:
            third.close()
    assert again["kill_switch"] is True
    assert again["reason"] == REASON
    assert _critical_records(caplog) == [], "an already re-created row logs nothing further"


# --------------------------------------------------------------- R3


def test_cached_conn_reseeds_engaged(tmp_path: Path) -> None:
    """R3: the row deleted under a live connection is re-created engaged by the
    very next read on that same connection."""
    sm = StateManager(tmp_path / "state.db")
    try:
        _delete_row(sm)
        state = sm.get_system_state()
        row = _read_row(sm)
    finally:
        sm.close()
    assert state["kill_switch"] is True
    assert state["reason"] == REASON
    assert (row[0], row[1], row[2], row[4]) == (1, 1, REASON, "system")
    assert row[3] is not None


# --------------------------------------------------------------- R3b


def test_unreseedable_row_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R3b: when the re-creating INSERT itself fails the read raises
    ``SystemStateMissingError`` instead of inventing a state."""
    sm = StateManager(tmp_path / "state.db")
    try:
        _delete_row(sm)

        def boom(conn: object, *, engaged: bool) -> bool:
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr("vibe_quant.db.state_manager.reseed_system_state", boom)
        with pytest.raises(SystemStateMissingError) as exc_info:
            sm.get_system_state()
        assert str(exc_info.value) == _UNRESEEDABLE
    finally:
        sm.close()


# --------------------------------------------------------------- R3c


def test_require_row_kwarg_ignored(tmp_path: Path) -> None:
    """R3c: T1 keeps the ``require_row`` kwarg accepted and ignored - both
    values are a plain read and return the same engaged result."""
    sm = StateManager(tmp_path / "state.db")
    try:
        _delete_row(sm)
        required = sm.get_system_state(require_row=True)
        default = sm.get_system_state(require_row=False)
    finally:
        sm.close()
    assert required == default
    assert required["kill_switch"] is True
    assert required["reason"] == REASON


# --------------------------------------------------------------- R4


def test_set_kill_upserts_missing_row(tmp_path: Path) -> None:
    """R4: engaging the switch on a missing row inserts it (not a silent no-op)."""
    sm = StateManager(tmp_path / "state.db")
    try:
        _delete_row(sm)
        sm.set_kill_switch("r", "op")
        row = _read_row(sm)
        state = sm.get_system_state()
    finally:
        sm.close()
    assert (row[0], row[1], row[2], row[4]) == (1, 1, "r", "op")
    assert row[3] is not None
    assert state["kill_switch"] is True
    assert state["reason"] == "r"


# --------------------------------------------------------------- R5


def test_clear_on_missing_row_reseeds_engaged_and_raises(tmp_path: Path) -> None:
    """R5: unlocking a missing row cannot fail open: the row is re-created
    engaged and the caller is told to unlock again."""
    sm = StateManager(tmp_path / "state.db")
    try:
        _delete_row(sm)
        with pytest.raises(SystemStateMissingError) as exc_info:
            sm.clear_kill_switch("op")
        assert str(exc_info.value) == _CLEAR_MISSING
        state = sm.get_system_state()
        row = _read_row(sm)
    finally:
        sm.close()
    assert state["kill_switch"] is True
    assert state["reason"] == REASON
    assert (row[1], row[2], row[4]) == (1, REASON, SYSTEM_STATE_UNLOCK_ACK_BY)


_TRIGGER_SQL = (
    "CREATE TRIGGER ks_block_reseed BEFORE INSERT ON system_state "
    "BEGIN SELECT missing_sql_function(); END"
)


def _delete_row_and_block_reseed(db: Path) -> None:
    """Delete the row and make every INSERT into system_state fail (real SQL error)."""
    raw = sqlite3.connect(db)
    try:
        raw.execute("DELETE FROM system_state")
        raw.execute(_TRIGGER_SQL)
        raw.commit()
    finally:
        raw.close()


def _raw_row(db: Path) -> tuple[Any, ...] | None:
    """The singleton row as seen by an INDEPENDENT connection (committed state only)."""
    raw = sqlite3.connect(db)
    try:
        row = raw.execute(
            "SELECT kill_switch, reason, killed_by FROM system_state WHERE id = 1"
        ).fetchone()
    finally:
        raw.close()
    return tuple(row) if row is not None else None


# --------------------------------------------------------------- SF1


def test_cached_reseed_is_committed_for_other_connections(tmp_path: Path) -> None:
    """R3: the cached-connection reseed is COMMITTED -- an independent raw
    connection sees the engaged row right after the read, before any later
    helper could commit it."""
    db = tmp_path / "state.db"
    sm = StateManager(db)
    try:
        _delete_row(sm)
        sm.get_system_state()
        seen = _raw_row(db)
    finally:
        sm.close()
    assert seen == (1, REASON, "system")


# --------------------------------------------------------------- R5 / R7b through a fresh manager


def test_fresh_manager_first_unlock_refused_then_succeeds(tmp_path: Path) -> None:
    """R5 unqualified: after a deleted row the FIRST unlock is refused even
    through a NEW StateManager whose schema init re-created the row; the next
    unlock releases it."""
    db = tmp_path / "state.db"
    first = StateManager(db)
    try:
        _delete_row(first)
    finally:
        first.close()
    fresh = StateManager(db)
    try:
        with pytest.raises(SystemStateMissingError) as exc_info:
            fresh.clear_kill_switch("op")
        assert str(exc_info.value) == _CLEAR_MISSING
        assert _raw_row(db) == (1, REASON, SYSTEM_STATE_UNLOCK_ACK_BY), "still engaged"
        fresh.clear_kill_switch("op")
        assert _raw_row(db) == (0, None, "op")
    finally:
        fresh.close()


def test_unlock_refused_when_another_connection_recreated_row(tmp_path: Path) -> None:
    """The re-creation may have been done by a different connection (job manager,
    CLI) than the one that unlocks: the evidence lives in the row."""
    db = tmp_path / "state.db"
    first = StateManager(db)
    try:
        _delete_row(first)
    finally:
        first.close()
    jobs = BacktestJobManager(db_path=db)
    try:
        _ = jobs.conn  # its init_schema re-creates the row engaged
    finally:
        jobs.close()
    sm = StateManager(db)
    try:
        with pytest.raises(SystemStateMissingError):
            sm.clear_kill_switch("op")
    finally:
        sm.close()


def test_operator_kill_then_unlock_is_not_refused(tmp_path: Path) -> None:
    """A normal engaged row (operator reason) unlocks on the first attempt."""
    sm = StateManager(tmp_path / "state.db")
    try:
        sm.set_kill_switch("divergence", "op")
        sm.clear_kill_switch("op")
        assert sm.get_system_state()["kill_switch"] is False
    finally:
        sm.close()


# --------------------------------------------------------------- R3b with a real failing INSERT


def test_fresh_manager_unreseedable_raises_missing_error(tmp_path: Path) -> None:
    """R3b on the FRESH path: the schema-init reseed INSERT fails for real
    (trigger) -> SystemStateMissingError with the exact text and the original
    sqlite error as the cause, not a raw OperationalError."""
    db = tmp_path / "state.db"
    first = StateManager(db)
    try:
        _ = first.conn
    finally:
        first.close()
    _delete_row_and_block_reseed(db)
    fresh = StateManager(db)
    try:
        with pytest.raises(SystemStateMissingError) as exc_info:
            fresh.get_system_state()
    finally:
        fresh.close()
    assert str(exc_info.value) == _UNRESEEDABLE
    assert isinstance(exc_info.value.__cause__, sqlite3.OperationalError)


def test_cached_unreseedable_cause_is_original_error(tmp_path: Path) -> None:
    """R3b on the CACHED path with the same real failure: same text, same cause."""
    db = tmp_path / "state.db"
    sm = StateManager(db)
    try:
        _delete_row(sm)
        raw = sqlite3.connect(db)
        raw.execute(_TRIGGER_SQL)
        raw.commit()
        raw.close()
        with pytest.raises(SystemStateMissingError) as exc_info:
            sm.get_system_state()
    finally:
        sm.close()
    assert str(exc_info.value) == _UNRESEEDABLE
    assert isinstance(exc_info.value.__cause__, sqlite3.OperationalError)


# --------------------------------------------------------------- R6-R8 (API)


@pytest.fixture()
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "ks.db"
    monkeypatch.setattr("vibe_quant.db.connection.DEFAULT_DB_PATH", db)
    monkeypatch.setattr(
        "vibe_quant.api.routers.paper_trading.PAPER_CONFIG_DIR", tmp_path / "paper_configs"
    )
    monkeypatch.setenv("BINANCE_TESTNET_API_KEY", "tk")
    monkeypatch.setenv("BINANCE_TESTNET_API_SECRET", "ts")
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    return db


@pytest.fixture()
async def api(tmp_db: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[dict[str, Any]]:
    app = create_app()
    state = StateManager(db_path=tmp_db)
    _ = state.conn  # init schema
    jobs = BacktestJobManager(db_path=tmp_db)
    ws = ConnectionManager()
    await ws.start()
    app.state.state_manager = state
    app.state.job_manager = jobs
    app.state.ws_manager = ws
    monkeypatch.setattr(jobs, "start_job", lambda *args, **kwargs: 4242)
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield {"ac": ac, "app": app, "state": state, "jobs": jobs, "db": tmp_db}
    await ws.stop()
    jobs.close()
    state.close()


def _seed_active_paper_job(state: StateManager, jobs: BacktestJobManager) -> int:
    """A validation run with a LIVE paper job row (what POST /kill cascades to)."""
    sid = state.create_strategy(DSL["name"], DSL)
    rid = state.create_backtest_run(
        strategy_id=sid,
        run_mode="validation",
        symbols=["BTCUSDT"],
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-12-31",
        parameters={"trader_id": _TRADER},
    )
    state.update_backtest_run_status(rid, "completed")
    jobs.conn.execute(
        """INSERT OR REPLACE INTO background_jobs
           (run_id, pid, job_type, status, started_at, heartbeat_at)
           VALUES (?, 4242, 'paper', 'running', datetime('now'), datetime('now'))""",
        (rid,),
    )
    jobs.conn.commit()
    return rid


async def test_status_missing_row_reads_engaged(api: dict[str, Any]) -> None:
    """R6: with the row missing, the status endpoint answers 200/engaged (it
    re-creates the row rather than reporting the old disengaged defaults)."""
    _delete_row(api["state"])
    r = await api["ac"].get("/api/system/status")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kill_switch"] is True
    assert body["reason"] == REASON


async def test_status_unavailable_503(api: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """R6b: an unreadable-and-unreseatable state is a 503, never a 200 default."""

    def boom(*, require_row: bool = True) -> dict[str, Any]:
        raise SystemStateMissingError(_UNRESEEDABLE)

    monkeypatch.setattr(api["state"], "get_system_state", boom)
    r = await api["ac"].get("/api/system/status")
    assert r.status_code == 503, r.text
    assert r.json()["detail"] == f"kill switch state unavailable: {_UNRESEEDABLE}"


async def test_kill_not_persisted_still_cascades_503(
    api: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R7: a failed persist still cascades kill commands to live paper sessions
    and reports the write failure (503) with the class and the session count."""
    _seed_active_paper_job(api["state"], api["jobs"])

    def boom(reason: str, killed_by: str | None = None) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(api["state"], "set_kill_switch", boom)
    r = await api["ac"].post("/api/system/kill", json={"reason": "divergence"})
    assert r.status_code == 503, r.text
    assert r.json()["detail"] == (
        "kill switch NOT persisted (OperationalError); kill commands sent to 1 paper session(s)"
    )
    queue = PaperCommandQueue(api["db"])
    try:
        cmds = queue.claim_pending(_TRADER)
    finally:
        queue.close()
    assert [c.command for c in cmds] == ["kill"]


async def test_unlock_missing_row_409(api: dict[str, Any]) -> None:
    """R7b: unlocking a missing row is a 409 and leaves the row engaged."""
    _delete_row(api["state"])
    r = await api["ac"].post(
        "/api/system/unlock", json={"acknowledge": True, "cleared_by": "op"}
    )
    assert r.status_code == 409, r.text
    assert r.json()["detail"] == _CLEAR_MISSING
    state = api["state"].get_system_state()
    assert state["kill_switch"] is True
    assert state["reason"] == REASON


async def test_paper_start_refused_after_reseed(api: dict[str, Any], tmp_db: Path) -> None:
    """R8: a fresh StateManager on a DB whose row was deleted re-creates it
    engaged, so starting a paper session is refused 423 with that reason."""
    _delete_row(api["state"])
    api["state"].close()
    fresh = StateManager(db_path=tmp_db)  # fresh manager: init re-seeds engaged
    api["app"].state.state_manager = fresh
    try:
        r = await api["ac"].post("/api/paper/start", json={"strategy_id": 1})
        assert r.status_code == 423, r.text
        assert r.json()["detail"] == f"System kill switch engaged: {REASON}"
    finally:
        fresh.close()


async def test_unlock_fresh_manager_409_then_200(api: dict[str, Any], tmp_db: Path) -> None:
    """R7b through the API with a NEW StateManager (schema init re-created the
    row): first unlock 409 and still engaged, second unlock 200 and released."""
    _delete_row(api["state"])
    api["state"].close()
    fresh = StateManager(db_path=tmp_db)
    api["app"].state.state_manager = fresh
    try:
        body = {"acknowledge": True, "cleared_by": "op"}
        r1 = await api["ac"].post("/api/system/unlock", json=body)
        assert r1.status_code == 409, r1.text
        assert r1.json()["detail"] == _CLEAR_MISSING
        assert fresh.get_system_state()["kill_switch"] is True
        r2 = await api["ac"].post("/api/system/unlock", json=body)
        assert r2.status_code == 200, r2.text
        assert r2.json()["kill_switch"] is False
    finally:
        fresh.close()


async def test_fresh_unreseedable_status_and_paper_503(api: dict[str, Any], tmp_db: Path) -> None:
    """R6b with a really failing fresh connection: /status is a 503 with the
    exact text (not a 500) and paper start is refused 503."""
    api["state"].close()
    _delete_row_and_block_reseed(tmp_db)
    fresh = StateManager(db_path=tmp_db)
    api["app"].state.state_manager = fresh
    try:
        r = await api["ac"].get("/api/system/status")
        assert r.status_code == 503, r.text
        assert r.json()["detail"] == f"kill switch state unavailable: {_UNRESEEDABLE}"
        p = await api["ac"].post("/api/paper/start", json={"strategy_id": 1})
        assert p.status_code == 503, p.text
    finally:
        fresh.close()


async def test_kill_cascades_when_state_initializer_fails(api: dict[str, Any], tmp_db: Path) -> None:
    """R7 with the REAL failing fresh connection: the StateManager cannot
    initialise (reseed INSERT fails), yet /kill still queues exactly one kill
    command for the active paper job (trader id read from the run) and answers
    the promised 503."""
    _seed_active_paper_job(api["state"], api["jobs"])
    api["state"].close()
    _delete_row_and_block_reseed(tmp_db)
    fresh = StateManager(db_path=tmp_db)
    api["app"].state.state_manager = fresh
    try:
        r = await api["ac"].post("/api/system/kill", json={"reason": "divergence"})
        assert r.status_code == 503, r.text
        assert r.json()["detail"] == (
            "kill switch NOT persisted (SystemStateMissingError); "
            "kill commands sent to 1 paper session(s)"
        )
    finally:
        fresh.close()
    queue = PaperCommandQueue(tmp_db)
    try:
        cmds = queue.claim_pending(_TRADER)
    finally:
        queue.close()
    assert [c.command for c in cmds] == ["kill"]
