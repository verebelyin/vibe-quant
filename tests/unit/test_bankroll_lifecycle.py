"""Bankroll report DB lifecycle: one-running admission, claim/fence, publish, reconcile.

Contract rows C26 (DB half) and C29 of design 2gyk1.3. Job rows are driven through
the real ``run_lifecycle`` functions on a second connection, as the job manager does.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from vibe_quant.db import run_lifecycle, state_manager
from vibe_quant.db.connection import get_connection
from vibe_quant.db.schema import _USER_VERSION_V16
from vibe_quant.db.state_manager import (
    BANKROLL_JOB_ID_BASE,
    BankrollAdmission,
    BankrollAdmissionError,
    StateManager,
)
from vibe_quant.jobs.manager import BacktestJobManager

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

DSL = {"name": "s", "timeframe": "4h", "indicators": {"rsi": {"period": 14}}}


def dsl_hash(dsl: object) -> str | None:
    if not isinstance(dsl, dict):
        return None
    return hashlib.sha256(json.dumps(dsl, sort_keys=True).encode()).hexdigest()


def alive(pid: int, identity: str | None) -> bool:
    return True


def dead(pid: int, identity: str | None) -> bool:
    return False


def never_signal(job_id: int) -> None:
    raise AssertionError(f"job {job_id} signalled")


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "state.db"


@pytest.fixture
def sm(db_path: Path) -> Iterator[StateManager]:
    mgr = StateManager(db_path)
    yield mgr
    mgr.close()


@pytest.fixture
def jobs_conn(db_path: Path, sm: StateManager) -> Iterator[sqlite3.Connection]:
    _ = sm.conn  # schema first
    conn = get_connection(db_path)
    yield conn
    conn.close()


def make_parent(
    sm: StateManager,
    *,
    name: str = "s",
    mode: str = "validation",
    status: str = "completed",
    identity: bool = True,
    dsl: dict[str, Any] | None = None,
) -> int:
    dsl = dsl or dict(DSL)
    strategy_id = sm.create_strategy(name, dsl)
    run_id = sm.create_backtest_run(
        strategy_id,
        mode,
        ["BTCUSDT"],
        "4h",
        "2024-01-01",
        "2024-06-01",
        {"risk_per_trade": 0.02, "max_position_pct": 0.5},
    )
    notes: dict[str, object] = {"data_window": "x"}
    if identity:
        notes["validation_input"] = {"v": 1, "strategy_name": name, "dsl_sha256": dsl_hash(dsl)}
    sm.save_backtest_result(run_id, {"total_return": 0.1, "notes": json.dumps(notes)})
    sm.update_backtest_run_status(run_id, status)
    return run_id


def admit(
    sm: StateManager, run_id: int, job_alive: Callable[[int, str | None], bool] = alive
) -> BankrollAdmission:
    return sm.admit_bankroll_report(
        run_id,
        {"multiples": [1.0]},
        dsl_hash=dsl_hash,
        job_alive=job_alive,
        terminate=never_signal,
    )


def open_job(conn: sqlite3.Connection, job_id: int) -> None:
    run_lifecycle.open_attempt(conn, job_id, "bankroll_report", None, exists=False)


def register(conn: sqlite3.Connection, job_id: int, pid: int) -> None:
    run_lifecycle.register_spawned(conn, job_id, pid, f"start-{pid}")


def kill(conn: sqlite3.Connection, job_id: int) -> None:
    run_lifecycle.finish(
        conn, job_id, run_lifecycle.JobStatus.KILLED, None, reconcile=False, owner=None
    )


def _fetch(db_path: Path, sql: str, key: int) -> sqlite3.Row | None:
    conn = get_connection(db_path)
    try:
        row: sqlite3.Row | None = conn.execute(sql, (key,)).fetchone()
        return row
    finally:
        conn.close()


def maybe_report_row(db_path: Path, report_id: int) -> sqlite3.Row | None:
    return _fetch(db_path, "SELECT * FROM bankroll_reports WHERE id = ?", report_id)


def report_row(db_path: Path, report_id: int) -> sqlite3.Row:
    row = maybe_report_row(db_path, report_id)
    assert row is not None
    return row


def job_row(db_path: Path, job_id: int) -> sqlite3.Row:
    row = _fetch(db_path, "SELECT * FROM background_jobs WHERE run_id = ?", job_id)
    assert row is not None
    return row


def launched(
    sm: StateManager, jobs_conn: sqlite3.Connection, run_id: int, pid: int = 4242
) -> BankrollAdmission:
    """Admit, open + register the job, and claim it as ``pid``."""
    report_id, job_id = admit(sm, run_id)
    open_job(jobs_conn, job_id)
    register(jobs_conn, job_id, pid)
    assert sm.claim_bankroll_report(report_id, job_id, pid, f"start-{pid}") is None
    return BankrollAdmission(report_id, job_id)


REPORT = {"recommendation": {"k": 0.5}, "rows": [{"k": 0.5}]}


# --- C26: cancel / publish ordering (red-first) ---------------------------------------


def test_lifecycle(sm: StateManager, jobs_conn: sqlite3.Connection, db_path: Path) -> None:
    """A cancel that wins before publication prevents any completed report."""
    run_id = make_parent(sm)
    report_id, job_id = launched(sm, jobs_conn, run_id)
    assert sm.bankroll_report_owned(report_id, 4242)

    kill(jobs_conn, job_id)
    assert not sm.bankroll_report_owned(report_id, 4242)  # per-cell fence stops the worker
    refusal = sm.publish_bankroll_report(report_id, 4242, REPORT, dsl_hash=dsl_hash)
    assert refusal is not None and "killed" in refusal
    row = report_row(db_path, report_id)
    assert row["status"] == "running" and row["report"] is None

    assert sm.reconcile_bankroll_reports(job_alive=alive, terminate=never_signal) == [report_id]
    row = report_row(db_path, report_id)
    assert (row["status"], row["error_message"]) == ("failed", "cancelled")
    # the slot is free again: a retry is admitted
    assert admit(sm, run_id)[0] == report_id + 1


def test_lifecycle_publish_wins_first(
    sm: StateManager, jobs_conn: sqlite3.Connection, db_path: Path
) -> None:
    run_id = make_parent(sm)
    report_id, job_id = launched(sm, jobs_conn, run_id)
    assert sm.publish_bankroll_report(report_id, 4242, REPORT, dsl_hash=dsl_hash) is None
    kill(jobs_conn, job_id)
    assert sm.reconcile_bankroll_reports(job_alive=alive, terminate=never_signal) == []
    row = report_row(db_path, report_id)
    assert row["status"] == "completed"
    assert json.loads(row["report"]) == REPORT
    got = sm.get_latest_bankroll_report(run_id, dsl_hash=dsl_hash)
    assert got is not None
    assert (got["status"], got["freshness"], got["report"]) == ("completed", "fresh", REPORT)


def test_lifecycle_failure_after_cancel_stays_cancelled(
    sm: StateManager, jobs_conn: sqlite3.Connection, db_path: Path
) -> None:
    run_id = make_parent(sm)
    report_id, job_id = launched(sm, jobs_conn, run_id)
    kill(jobs_conn, job_id)
    assert not sm.end_bankroll_report(report_id, "failed", "boom", owner_pid=4242)
    sm.reconcile_bankroll_reports(job_alive=alive, terminate=never_signal)
    assert report_row(db_path, report_id)["error_message"] == "cancelled"


def test_lifecycle_owner_failure_and_stale(
    sm: StateManager, jobs_conn: sqlite3.Connection, db_path: Path
) -> None:
    run_id = make_parent(sm)
    report_id, _ = launched(sm, jobs_conn, run_id)
    assert not sm.end_bankroll_report(report_id, "stale_parent", "x", owner_pid=999)
    assert sm.end_bankroll_report(
        report_id, "stale_parent", "inputs_mismatch: end_date", owner_pid=4242
    )
    row = report_row(db_path, report_id)
    assert (row["status"], row["error_message"]) == ("stale_parent", "inputs_mismatch: end_date")
    assert row["completed_at"] is not None
    with pytest.raises(ValueError, match="completed"):
        sm.end_bankroll_report(report_id, "completed", "x", owner_pid=4242)


# --- C26: one global running report --------------------------------------------------


def test_lifecycle_second_parent_on_another_connection_is_refused(
    sm: StateManager, db_path: Path
) -> None:
    first = make_parent(sm, name="a")
    second = make_parent(sm, name="b")
    report_id, job_id = admit(sm, first)
    assert job_id == BANKROLL_JOB_ID_BASE - report_id == -(10**12) - report_id
    other = StateManager(db_path)
    try:
        with pytest.raises(BankrollAdmissionError) as exc:
            admit(other, second)
        assert exc.value.code == "report_running"
    finally:
        other.close()
    assert maybe_report_row(db_path, report_id + 1) is None


def test_lifecycle_concurrent_admissions_admit_exactly_one(db_path: Path) -> None:
    setup = StateManager(db_path)
    parents = [make_parent(setup, name=f"p{i}") for i in range(4)]
    setup.close()
    results: list[object] = []
    barrier = threading.Barrier(len(parents))

    def worker(run_id: int) -> None:
        mgr = StateManager(db_path)
        _ = mgr.conn
        barrier.wait()
        try:
            results.append(admit(mgr, run_id))
        except BankrollAdmissionError as exc:
            results.append(exc.code)
        finally:
            mgr.close()

    threads = [threading.Thread(target=worker, args=(p,)) for p in parents]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(r for r in results if isinstance(r, str)) == ["report_running"] * 3
    assert len([r for r in results if isinstance(r, tuple)]) == 1


def test_lifecycle_open_report_job_holds_the_slot_after_parent_delete(
    sm: StateManager, jobs_conn: sqlite3.Connection, db_path: Path
) -> None:
    first = make_parent(sm, name="a")
    second = make_parent(sm, name="b")
    report_id, job_id = launched(sm, jobs_conn, first)
    with sm._write_lock:
        sm.conn.execute("DELETE FROM backtest_results WHERE run_id = ?", (first,))
        sm.conn.execute("DELETE FROM backtest_runs WHERE id = ?", (first,))
        sm.conn.commit()
    assert maybe_report_row(db_path, report_id) is None  # cascade
    with pytest.raises(BankrollAdmissionError) as exc:
        admit(sm, second)
    assert exc.value.code == "job_open"
    assert not sm.bankroll_report_owned(report_id, 4242)
    assert sm.publish_bankroll_report(report_id, 4242, REPORT, dsl_hash=dsl_hash) is not None
    # the job ends -> the slot is released
    run_lifecycle.finish(
        jobs_conn, job_id, run_lifecycle.JobStatus.FAILED, "gone", reconcile=False, owner=None
    )
    assert admit(sm, second)[0] == report_id + 1


def test_lifecycle_wrong_job_id_held_by_data_job(
    sm: StateManager, jobs_conn: sqlite3.Connection, db_path: Path
) -> None:
    run_id = make_parent(sm)
    held = BANKROLL_JOB_ID_BASE - 1  # the id the first report would get
    jobs_conn.execute(
        "INSERT INTO background_jobs (run_id, pid, job_type, status, error_message) "
        "VALUES (?, 77, 'data_ingest', 'completed', 'data job row')",
        (held,),
    )
    jobs_conn.commit()
    before = dict(job_row(db_path, held))
    with pytest.raises(BankrollAdmissionError) as exc:
        admit(sm, run_id)
    assert exc.value.code == "job_id_taken"
    assert dict(job_row(db_path, held)) == before
    with sm._write_lock:
        assert sm.conn.execute("SELECT COUNT(*) FROM bankroll_reports").fetchone()[0] == 0


def test_lifecycle_spawn_failure_releases_the_slot(sm: StateManager, db_path: Path) -> None:
    run_id = make_parent(sm)
    report_id, _ = admit(sm, run_id)
    assert sm.end_bankroll_report(report_id, "failed", "spawn failed: OSError", owner_pid=None)
    assert report_row(db_path, report_id)["status"] == "failed"
    assert admit(sm, run_id)[0] == report_id + 1


# --- C26: launch gaps, deadline, fencing ---------------------------------------------


def _expire(db_path: Path, report_id: int) -> None:
    conn = get_connection(db_path)
    conn.execute(
        "UPDATE bankroll_reports SET launch_deadline = datetime('now', '-2 seconds') WHERE id = ?",
        (report_id,),
    )
    conn.commit()
    conn.close()


def test_lifecycle_launch_gaps_are_not_dead(
    sm: StateManager, jobs_conn: sqlite3.Connection, db_path: Path
) -> None:
    run_id = make_parent(sm)
    report_id, job_id = admit(sm, run_id)
    deadline_s = (
        get_connection(db_path)
        .execute(
            "SELECT CAST(strftime('%s', launch_deadline) AS INTEGER) "
            "- CAST(strftime('%s', created_at) AS INTEGER) FROM bankroll_reports"
        )
        .fetchone()[0]
    )
    assert deadline_s == 120
    # gap 1: no job row yet
    assert sm.reconcile_bankroll_reports(job_alive=dead, terminate=never_signal) == []
    assert report_row(db_path, report_id)["status"] == "running"
    # gap 2: job row pending with pid 0
    open_job(jobs_conn, job_id)
    assert sm.reconcile_bankroll_reports(job_alive=dead, terminate=never_signal) == []
    assert report_row(db_path, report_id)["status"] == "running"
    assert job_row(db_path, job_id)["status"] == "pending"


def test_lifecycle_launcher_death_before_job_creation(
    sm: StateManager, jobs_conn: sqlite3.Connection, db_path: Path
) -> None:
    run_id = make_parent(sm)
    report_id, job_id = admit(sm, run_id)
    _expire(db_path, report_id)
    assert sm.reconcile_bankroll_reports(job_alive=alive, terminate=never_signal) == [report_id]
    row = report_row(db_path, report_id)
    assert row["status"] == "failed" and "launch" in row["error_message"]
    # a late launcher + child is fenced before any engine work
    open_job(jobs_conn, job_id)
    register(jobs_conn, job_id, 4242)
    refusal = sm.claim_bankroll_report(report_id, job_id, 4242, "start-4242")
    assert refusal is not None and "failed" in refusal
    assert report_row(db_path, report_id)["owner_pid"] is None


def test_lifecycle_launcher_death_before_pid_registration(
    sm: StateManager, jobs_conn: sqlite3.Connection, db_path: Path
) -> None:
    run_id = make_parent(sm)
    report_id, job_id = admit(sm, run_id)
    open_job(jobs_conn, job_id)
    _expire(db_path, report_id)
    assert sm.reconcile_bankroll_reports(job_alive=alive, terminate=never_signal) == [report_id]
    assert report_row(db_path, report_id)["status"] == "failed"
    assert job_row(db_path, job_id)["status"] == "failed"  # the job row frees the slot too
    register(jobs_conn, job_id, 4242)  # late launcher
    assert job_row(db_path, job_id)["status"] == "failed"
    assert sm.claim_bankroll_report(report_id, job_id, 4242, "start-4242") is not None
    assert admit(sm, run_id)[0] == report_id + 1


@pytest.mark.parametrize("job_first", [True, False])
@pytest.mark.parametrize(
    ("offset", "refused"), [("-2 seconds", True), ("+0 seconds", True), ("+30 seconds", False)]
)
def test_lifecycle_first_claim_needs_an_open_launch_window(
    sm: StateManager,
    jobs_conn: sqlite3.Connection,
    db_path: Path,
    job_first: bool,
    offset: str,
    refused: bool,
) -> None:
    """No reconcile runs between expiry and the claim: the claim checks the deadline."""
    run_id = make_parent(sm)
    report_id, job_id = admit(sm, run_id)
    if job_first:
        open_job(jobs_conn, job_id)  # pending, pid 0 at expiry
    jobs_conn.execute(
        "UPDATE bankroll_reports SET launch_deadline = datetime('now', ?) WHERE id = ?",
        (offset, report_id),
    )
    jobs_conn.commit()
    if not job_first:
        open_job(jobs_conn, job_id)  # the launcher creates and registers after expiry
        register(jobs_conn, job_id, 4242)
    before = dict(job_row(db_path, job_id))
    refusal = sm.claim_bankroll_report(report_id, job_id, 4242, "start-4242")
    if not refused:
        assert refusal is None
        assert report_row(db_path, report_id)["owner_pid"] == 4242
        return
    assert refusal is not None and "launch deadline" in refusal
    assert report_row(db_path, report_id)["owner_pid"] is None
    assert dict(job_row(db_path, job_id)) == before
    assert not sm.bankroll_report_owned(report_id, 4242)


def test_lifecycle_claimed_child_survives_an_expired_deadline(
    sm: StateManager, jobs_conn: sqlite3.Connection, db_path: Path
) -> None:
    """A live child that claimed before registration is not failed by the deadline."""
    run_id = make_parent(sm)
    report_id, job_id = admit(sm, run_id)
    open_job(jobs_conn, job_id)
    assert sm.claim_bankroll_report(report_id, job_id, 4242, "start-4242") is None
    job = job_row(db_path, job_id)
    assert (job["status"], job["pid"], job["pid_start_time"]) == ("running", 4242, "start-4242")
    _expire(db_path, report_id)
    # a valid claim survives the deadline: re-claim and the fence still pass
    assert sm.claim_bankroll_report(report_id, job_id, 4242, "start-4242") is None
    assert sm.bankroll_report_owned(report_id, 4242)
    assert sm.reconcile_bankroll_reports(job_alive=alive, terminate=never_signal) == []
    assert report_row(db_path, report_id)["status"] == "running"


def test_lifecycle_dead_child_before_heartbeat(
    sm: StateManager, jobs_conn: sqlite3.Connection, db_path: Path
) -> None:
    run_id = make_parent(sm)
    report_id, job_id = admit(sm, run_id)
    open_job(jobs_conn, job_id)
    register(jobs_conn, job_id, 4242)
    seen: list[tuple[int, str | None]] = []

    def probe(pid: int, identity: str | None) -> bool:
        seen.append((pid, identity))
        return False

    assert sm.reconcile_bankroll_reports(job_alive=probe, terminate=never_signal) == [report_id]
    assert seen == [(4242, "start-4242")]
    assert report_row(db_path, report_id)["status"] == "failed"
    assert job_row(db_path, job_id)["status"] == "failed"


@pytest.mark.parametrize(("age_s", "stale"), [(110, False), (130, True)])
def test_lifecycle_stale_heartbeat(
    sm: StateManager, jobs_conn: sqlite3.Connection, db_path: Path, age_s: int, stale: bool
) -> None:
    run_id = make_parent(sm)
    report_id, job_id = launched(sm, jobs_conn, run_id)
    jobs_conn.execute(
        "UPDATE background_jobs SET heartbeat_at = datetime('now', ?) WHERE run_id = ?",
        (f"-{age_s} seconds", job_id),
    )
    jobs_conn.commit()
    signalled: list[tuple[int, str]] = []

    def terminate(signalled_job: int) -> None:
        # the job row is still open when the verified process is signalled
        signalled.append((signalled_job, job_row(db_path, signalled_job)["status"]))

    flipped = sm.reconcile_bankroll_reports(job_alive=alive, terminate=terminate)
    assert flipped == ([report_id] if stale else [])
    assert signalled == ([(job_id, "running")] if stale else [])
    report = report_row(db_path, report_id)
    job = job_row(db_path, job_id)
    if not stale:
        assert (report["status"], job["status"]) == ("running", "running")
        return
    assert report["status"] == "failed" and "heartbeat" in report["error_message"]
    assert job["status"] == "failed" and "heartbeat" in job["error_message"]
    # the terminal rows free the slot; the fenced worker can no longer publish
    assert admit(sm, run_id)[0] == report_id + 1
    assert not sm.bankroll_report_owned(report_id, 4242)
    assert sm.publish_bankroll_report(report_id, 4242, REPORT, dsl_hash=dsl_hash) is not None
    assert report_row(db_path, report_id)["report"] is None


def test_lifecycle_claim_refusals(
    sm: StateManager, jobs_conn: sqlite3.Connection, db_path: Path
) -> None:
    run_id = make_parent(sm)
    report_id, job_id = admit(sm, run_id)
    # no job row yet
    assert sm.claim_bankroll_report(report_id, job_id, 4242, "start-4242") is not None
    open_job(jobs_conn, job_id)
    register(jobs_conn, job_id, 5555)
    assert "5555" in (sm.claim_bankroll_report(report_id, job_id, 4242, "start-4242") or "")
    assert sm.claim_bankroll_report(report_id, job_id - 1, 5555, "start-5555") is not None
    assert sm.claim_bankroll_report(report_id, job_id, 5555, None) is not None
    assert sm.claim_bankroll_report(report_id, job_id, 5555, "recycled") is not None
    assert report_row(db_path, report_id)["owner_pid"] is None
    assert sm.claim_bankroll_report(report_id, job_id, 5555, "start-5555") is None
    assert report_row(db_path, report_id)["owner_pid"] == 5555
    assert sm.claim_bankroll_report(report_id + 7, job_id, 5555, "start-5555") is not None


# --- C26 / C20 / C22: admission snapshot and freshness ---------------------------------


def test_lifecycle_admission_refusals(sm: StateManager) -> None:
    cases = {
        "run_missing": 9999,
        "not_validation": make_parent(sm, name="scr", mode="screening"),
        "not_completed": make_parent(sm, name="run", status="running"),
        "inputs_unverified": make_parent(sm, name="old", identity=False),
    }
    edited = make_parent(sm, name="edit")
    sm.update_strategy(
        sm.get_strategy_id_by_name("edit") or 0,
        dsl_config={**DSL, "indicators": {"rsi": {"period": 15}}},
    )
    cases["stale_parent"] = edited
    for code, run_id in cases.items():
        with pytest.raises(BankrollAdmissionError) as exc:
            admit(sm, run_id)
        assert exc.value.code == code
        if code in ("inputs_unverified", "stale_parent"):
            assert "re-validate first" in str(exc.value)
    with sm._write_lock:
        assert sm.conn.execute("SELECT COUNT(*) FROM bankroll_reports").fetchone()[0] == 0


def test_lifecycle_admission_stores_the_input_snapshot(sm: StateManager) -> None:
    run_id = make_parent(sm)
    report_id, job_id = admit(sm, run_id)
    row = sm.get_bankroll_report(report_id)
    assert row is not None
    assert (row["run_id"], row["job_id"], row["status"]) == (run_id, job_id, "running")
    assert row["request"] == {"multiples": [1.0]}
    inputs = row["inputs"]
    result = sm.get_backtest_result(run_id)
    assert result is not None
    assert inputs["parent_result_id"] == result["id"]
    assert inputs["strategy_name"] == "s"
    assert inputs["dsl_sha256"] == dsl_hash(DSL)
    assert json.loads(inputs["dsl_json"]) == DSL
    assert json.loads(inputs["run_row"]["parameters"]) == {
        "risk_per_trade": 0.02,
        "max_position_pct": 0.5,
    }
    assert inputs["run_row"]["symbols"] == '["BTCUSDT"]'
    assert inputs["parent_identity"] == {"v": 1, "strategy_name": "s", "dsl_sha256": dsl_hash(DSL)}
    assert sm.list_running_bankroll_report_ids() == [report_id]


def _rerun_parent(sm: StateManager, run_id: int) -> None:
    result = sm.get_backtest_result(run_id)
    assert result is not None
    sm.save_backtest_result(run_id, {"total_return": 0.1, "notes": result["notes"]})


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("dsl", "dsl_changed"),
        ("result", "result_replaced"),
        ("running", "parent_not_completed"),
        ("row", "run_row_changed"),
        ("no_result", "result_missing"),
    ],
)
def test_lifecycle_publish_freshness(
    sm: StateManager, jobs_conn: sqlite3.Connection, db_path: Path, change: str, reason: str
) -> None:
    run_id = make_parent(sm)
    report_id, _ = launched(sm, jobs_conn, run_id)
    if change == "dsl":
        sm.update_strategy(
            sm.get_strategy_id_by_name("s") or 0, dsl_config={**DSL, "indicators": {}}
        )
    elif change == "result":
        _rerun_parent(sm, run_id)
    elif change == "running":
        jobs_conn.execute("UPDATE backtest_runs SET status = 'running' WHERE id = ?", (run_id,))
        jobs_conn.commit()
    elif change == "row":
        jobs_conn.execute(
            "UPDATE backtest_runs SET end_date = '2024-07-01' WHERE id = ?", (run_id,)
        )
        jobs_conn.commit()
    else:
        jobs_conn.execute("DELETE FROM backtest_results WHERE run_id = ?", (run_id,))
        jobs_conn.commit()
    got = sm.get_latest_bankroll_report(run_id, dsl_hash=dsl_hash)
    assert got is not None and got["freshness"] == "stale"
    assert any(r.startswith(reason) for r in got["stale_reasons"])
    refusal = sm.publish_bankroll_report(report_id, 4242, REPORT, dsl_hash=dsl_hash)
    assert refusal is not None and reason in refusal
    row = report_row(db_path, report_id)
    assert row["status"] == "stale_parent" and row["report"] is None
    assert reason in row["error_message"]


def test_lifecycle_publish_holds_the_write_lock_while_checking(
    sm: StateManager, jobs_conn: sqlite3.Connection, db_path: Path
) -> None:
    """An edit racing the freshness check cannot land before the report is written."""
    run_id = make_parent(sm)
    report_id, _ = launched(sm, jobs_conn, run_id)
    racer = sqlite3.connect(db_path, timeout=0.05)
    errors: list[str] = []

    def hash_and_race(dsl: object) -> str | None:
        try:
            racer.execute("UPDATE strategies SET dsl_config = '{}'")
            racer.commit()
        except sqlite3.OperationalError as exc:
            errors.append(str(exc))
        return dsl_hash(dsl)

    assert sm.publish_bankroll_report(report_id, 4242, REPORT, dsl_hash=hash_and_race) is None
    racer.close()
    assert errors == ["database is locked"]
    assert report_row(db_path, report_id)["status"] == "completed"


def test_lifecycle_get_masks_a_stale_recommendation(
    sm: StateManager, jobs_conn: sqlite3.Connection
) -> None:
    run_id = make_parent(sm)
    report_id, _ = launched(sm, jobs_conn, run_id)
    assert sm.publish_bankroll_report(report_id, 4242, REPORT, dsl_hash=dsl_hash) is None
    _rerun_parent(sm, run_id)
    got = sm.get_latest_bankroll_report(run_id, dsl_hash=dsl_hash)
    assert got is not None
    assert (got["status"], got["freshness"]) == ("completed", "stale")
    assert got["stale_reasons"] == ["result_replaced"]
    assert got["report"] == {"recommendation": None, "rows": [{"k": 0.5}]}
    assert sm.get_latest_bankroll_report(run_id + 50, dsl_hash=dsl_hash) is None


def test_lifecycle_unreadable_dsl_is_stale(sm: StateManager, jobs_conn: sqlite3.Connection) -> None:
    run_id = make_parent(sm)
    launched(sm, jobs_conn, run_id)
    got = sm.get_latest_bankroll_report(run_id, dsl_hash=lambda dsl: None)
    assert got is not None and got["stale_reasons"] == ["dsl_unreadable"]


def test_lifecycle_late_registration_keeps_the_claimed_identity(
    sm: StateManager, db_path: Path
) -> None:
    """Child claims first; the launcher's start-time lookup then fails (None)."""
    run_id = make_parent(sm)
    report_id, job_id = admit(sm, run_id)
    jobs = BacktestJobManager(db_path)

    def spawn(*args: object, **kwargs: object) -> SimpleNamespace:
        assert sm.claim_bankroll_report(report_id, job_id, 4242, "child-token") is None
        return SimpleNamespace(pid=4242)

    try:
        with (
            patch("vibe_quant.jobs.manager.subprocess.Popen", spawn),
            patch("vibe_quant.jobs.manager.process_start_time", return_value=None),
        ):
            assert jobs.start_job(job_id, "bankroll_report", ["unused"]) == 4242
        job = job_row(db_path, job_id)
        assert (job["status"], job["pid"], job["pid_start_time"]) == (
            "running",
            4242,
            "child-token",
        )
        # pid 4242 now belongs to an unrelated process: the report job is dead
        with (
            patch.object(jobs, "is_process_alive", return_value=True),
            patch("vibe_quant.jobs.manager.process_start_time", return_value="recycled-token"),
        ):
            flipped = sm.reconcile_bankroll_reports(
                job_alive=jobs._job_alive, terminate=never_signal
            )
        assert flipped == [report_id]
    finally:
        jobs.close()


def test_register_spawned_identity(jobs_conn: sqlite3.Connection, db_path: Path) -> None:
    open_job(jobs_conn, -31)
    run_lifecycle.register_spawned(jobs_conn, -31, 61, None)  # nothing recorded before
    assert job_row(db_path, -31)["pid_start_time"] is None
    run_lifecycle.register_spawned(jobs_conn, -31, 61, "launcher-61")
    assert job_row(db_path, -31)["pid_start_time"] == "launcher-61"
    run_lifecycle.register_spawned(jobs_conn, -31, 61, None)
    assert job_row(db_path, -31)["pid_start_time"] == "launcher-61"
    open_job(jobs_conn, -32)
    jobs_conn.execute("UPDATE background_jobs SET pid_start_time = 'old' WHERE run_id = -32")
    jobs_conn.commit()
    run_lifecycle.register_spawned(jobs_conn, -32, 62, None)  # pid 0 -> 62: no carry-over
    assert job_row(db_path, -32)["pid_start_time"] is None


def test_lifecycle_refused_admission_keeps_the_recovery(
    sm: StateManager, jobs_conn: sqlite3.Connection, db_path: Path
) -> None:
    run_id = make_parent(sm)
    report_id, _ = admit(sm, run_id)
    _expire(db_path, report_id)
    with pytest.raises(BankrollAdmissionError) as exc:
        admit(sm, 9999)
    assert exc.value.code == "run_missing"
    row = report_row(db_path, report_id)  # read on a second connection
    assert row["status"] == "failed" and "launch" in row["error_message"]
    # a refusal after the reservation insert keeps the recovery, drops the reservation
    second, _ = admit(sm, run_id)
    _expire(db_path, second)
    held = BANKROLL_JOB_ID_BASE - (second + 1)
    jobs_conn.execute(
        "INSERT INTO background_jobs (run_id, pid, job_type, status) "
        "VALUES (?, 77, 'data_ingest', 'completed')",
        (held,),
    )
    jobs_conn.commit()
    with pytest.raises(BankrollAdmissionError) as exc:
        admit(sm, run_id)
    assert exc.value.code == "job_id_taken"
    assert report_row(db_path, second)["status"] == "failed"
    assert maybe_report_row(db_path, second + 1) is None
    jobs_conn.execute("DELETE FROM background_jobs WHERE run_id = ?", (held,))
    jobs_conn.commit()
    assert admit(sm, run_id)[0] == second + 1


@pytest.mark.parametrize("run_id", [2**63, -(2**63) - 1, 10**400])
def test_lifecycle_out_of_range_ids_are_missing(sm: StateManager, run_id: int) -> None:
    with pytest.raises(BankrollAdmissionError) as exc:
        admit(sm, run_id)
    assert exc.value.code == "run_missing"
    assert sm.get_latest_bankroll_report(run_id, dsl_hash=dsl_hash) is None
    assert sm.get_bankroll_report(run_id) is None


def test_lifecycle_in_range_extreme_ids_are_missing(sm: StateManager) -> None:
    for run_id in (2**63 - 1, -(2**63)):
        with pytest.raises(BankrollAdmissionError) as exc:
            admit(sm, run_id)
        assert exc.value.code == "run_missing"
        assert sm.get_bankroll_report(run_id) is None


def test_lifecycle_split_owner_is_refused(
    sm: StateManager, jobs_conn: sqlite3.Connection, db_path: Path
) -> None:
    """The job row names pid 5555 but the report is still owned by 4242."""
    run_id = make_parent(sm)
    report_id, job_id = launched(sm, jobs_conn, run_id)
    jobs_conn.execute(
        "UPDATE background_jobs SET pid = 5555, pid_start_time = 'start-5555' WHERE run_id = ?",
        (job_id,),
    )
    jobs_conn.commit()
    before = dict(report_row(db_path, report_id))
    assert not sm.bankroll_report_owned(report_id, 5555)
    refusal = sm.publish_bankroll_report(report_id, 5555, REPORT, dsl_hash=dsl_hash)
    assert refusal is not None and "owned by pid 4242" in refusal
    assert not sm.end_bankroll_report(report_id, "failed", "x", owner_pid=5555)
    assert dict(report_row(db_path, report_id)) == before


def test_lifecycle_get_reads_one_snapshot(
    sm: StateManager,
    jobs_conn: sqlite3.Connection,
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A parent edit landing between the report read and the freshness read."""
    run_id = make_parent(sm)
    report_id, _ = launched(sm, jobs_conn, run_id)
    assert sm.publish_bankroll_report(report_id, 4242, REPORT, dsl_hash=dsl_hash) is None
    original = state_manager._bankroll_stale_reasons
    edits: list[int] = []

    def edit_then_check(conn: sqlite3.Connection, inputs: Any, h: Any) -> list[str]:
        if not edits:
            jobs_conn.execute(
                "UPDATE strategies SET dsl_config = ? WHERE name = 's'",
                (json.dumps({**DSL, "indicators": {}}),),
            )
            jobs_conn.commit()
            edits.append(1)
        return original(conn, inputs, h)

    monkeypatch.setattr(state_manager, "_bankroll_stale_reasons", edit_then_check)
    got = sm.get_latest_bankroll_report(run_id, dsl_hash=dsl_hash)
    assert edits == [1]
    assert got is not None
    assert (got["freshness"], got["report"]) == ("fresh", REPORT)
    again = sm.get_latest_bankroll_report(run_id, dsl_hash=dsl_hash)
    assert again is not None
    assert (again["freshness"], again["stale_reasons"]) == ("stale", ["dsl_changed"])
    assert again["report"] == {"recommendation": None, "rows": [{"k": 0.5}]}


# --- C29: schema ----------------------------------------------------------------------


def _drop_bankroll(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("DROP INDEX IF EXISTS one_running_bankroll")
    conn.execute("DROP TABLE IF EXISTS bankroll_reports")
    conn.commit()
    conn.close()


def test_schema(db_path: Path) -> None:
    """A populated pre-change DB opens twice and concurrently and keeps every row."""
    setup = StateManager(db_path)
    parents = [make_parent(setup, name=f"p{i}") for i in range(3)]
    setup.close()
    _drop_bankroll(db_path)
    raw = sqlite3.connect(db_path)
    tables = [
        r[0]
        for r in raw.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    before = {t: raw.execute(f"SELECT * FROM {t} ORDER BY rowid").fetchall() for t in tables}
    user_version = raw.execute("PRAGMA user_version").fetchone()[0]
    raw.close()
    assert user_version == _USER_VERSION_V16

    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def open_db() -> None:
        try:
            mgr = StateManager(db_path)
            barrier.wait()
            _ = mgr.conn
            mgr.close()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=open_db) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    StateManager(db_path).close()  # third, serial open

    mgr = StateManager(db_path)
    with mgr._write_lock:
        conn = mgr.conn
        for t in tables:
            rows = conn.execute(f"SELECT * FROM {t} ORDER BY rowid").fetchall()
            assert [tuple(r) for r in rows] == before[t]
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("PRAGMA user_version").fetchone()[0] == user_version
        index = {
            r["name"]: (r["unique"], r["partial"])
            for r in conn.execute("PRAGMA index_list('bankroll_reports')")
        }
        columns = [r["name"] for r in conn.execute("PRAGMA index_info('one_running_bankroll')")]
    assert index["one_running_bankroll"] == (1, 1) and columns == ["status"]
    first, *_ = (admit(mgr, p) for p in parents[:1])
    assert first[0] == 1
    mgr.close()


def _insert_report(conn: sqlite3.Connection, run_id: int, job_id: int, status: str) -> None:
    conn.execute(
        "INSERT INTO bankroll_reports (run_id, job_id, status, launch_deadline, inputs, request)"
        " VALUES (?, ?, ?, datetime('now'), '{}', '{}')",
        (run_id, job_id, status),
    )
    conn.commit()


def test_schema_slot_and_cascade(sm: StateManager, jobs_conn: sqlite3.Connection) -> None:
    a = make_parent(sm, name="a")
    b = make_parent(sm, name="b")
    _insert_report(jobs_conn, a, -1, "running")
    with pytest.raises(sqlite3.IntegrityError, match="bankroll_reports.status"):
        _insert_report(jobs_conn, b, -2, "running")
    jobs_conn.rollback()
    jobs_conn.execute("UPDATE bankroll_reports SET status = 'completed'")
    jobs_conn.commit()
    _insert_report(jobs_conn, b, -2, "running")  # a terminal row released the slot
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        _insert_report(jobs_conn, b, -3, "cancelled")
    jobs_conn.rollback()
    with pytest.raises(sqlite3.IntegrityError, match="job_id"):
        _insert_report(jobs_conn, a, -2, "failed")
    jobs_conn.rollback()
    jobs_conn.execute("DELETE FROM backtest_results WHERE run_id = ?", (a,))
    jobs_conn.execute("DELETE FROM backtest_runs WHERE id = ?", (a,))
    jobs_conn.commit()
    rows = jobs_conn.execute("SELECT run_id, job_id FROM bankroll_reports").fetchall()
    assert [tuple(r) for r in rows] == [(b, -2)]


# --- run_lifecycle job-row helpers used by the bankroll lifecycle ----------------------


def _in_txn(conn: sqlite3.Connection, fn: Callable[[], bool]) -> bool:
    conn.execute("BEGIN IMMEDIATE")
    out = fn()
    conn.commit()
    return out


def test_fail_open_job(jobs_conn: sqlite3.Connection, db_path: Path) -> None:
    open_job(jobs_conn, -11)
    assert _in_txn(jobs_conn, lambda: run_lifecycle.fail_open_job(jobs_conn, -11, "dead"))
    job = job_row(db_path, -11)
    assert (job["status"], job["error_message"]) == ("failed", "dead")
    assert job["completed_at"] is not None
    # running row keeps an existing error text
    open_job(jobs_conn, -12)
    register(jobs_conn, -12, 31)
    jobs_conn.execute("UPDATE background_jobs SET error_message = 'first' WHERE run_id = -12")
    jobs_conn.commit()
    assert _in_txn(jobs_conn, lambda: run_lifecycle.fail_open_job(jobs_conn, -12, "second"))
    assert (job_row(db_path, -12)["status"], job_row(db_path, -12)["error_message"]) == (
        "failed",
        "first",
    )
    # terminal and missing rows: no-op
    jobs_conn.execute("UPDATE background_jobs SET status = 'killed' WHERE run_id = -12")
    jobs_conn.commit()
    before = dict(job_row(db_path, -12))
    assert not _in_txn(jobs_conn, lambda: run_lifecycle.fail_open_job(jobs_conn, -12, "x"))
    assert dict(job_row(db_path, -12)) == before
    assert not _in_txn(jobs_conn, lambda: run_lifecycle.fail_open_job(jobs_conn, -99, "x"))


def test_claim_job_row(jobs_conn: sqlite3.Connection, db_path: Path) -> None:
    open_job(jobs_conn, -21)
    assert _in_txn(jobs_conn, lambda: run_lifecycle.claim_job_row(jobs_conn, -21, 51, "s-51"))
    job = job_row(db_path, -21)
    assert (job["status"], job["pid"], job["pid_start_time"]) == ("running", 51, "s-51")
    assert job["heartbeat_at"] is not None
    # another attempt's pid is never overwritten
    before = dict(job)
    assert not _in_txn(jobs_conn, lambda: run_lifecycle.claim_job_row(jobs_conn, -21, 52, "s"))
    assert dict(job_row(db_path, -21)) == before
    # a terminal row is never reopened
    run_lifecycle.finish(
        jobs_conn, -21, run_lifecycle.JobStatus.FAILED, "done", reconcile=False, owner=None
    )
    before = dict(job_row(db_path, -21))
    assert not _in_txn(jobs_conn, lambda: run_lifecycle.claim_job_row(jobs_conn, -21, 51, "s-51"))
    assert dict(job_row(db_path, -21)) == before


@pytest.mark.parametrize(
    ("helper", "status"),
    [
        (lambda c: run_lifecycle.fail_open_job(c, -41, "dead"), "failed"),
        (lambda c: run_lifecycle.claim_job_row(c, -41, 71, "s-71"), "running"),
    ],
    ids=["fail_open_job", "claim_job_row"],
)
def test_job_row_helpers_leave_the_commit_to_the_caller(
    jobs_conn: sqlite3.Connection,
    db_path: Path,
    helper: Callable[[sqlite3.Connection], bool],
    status: str,
) -> None:
    open_job(jobs_conn, -41)
    before = dict(job_row(db_path, -41))
    jobs_conn.execute("BEGIN IMMEDIATE")
    assert helper(jobs_conn)
    assert dict(job_row(db_path, -41)) == before  # a second connection sees nothing yet
    jobs_conn.rollback()
    assert dict(job_row(db_path, -41)) == before
    assert _in_txn(jobs_conn, lambda: helper(jobs_conn))
    assert job_row(db_path, -41)["status"] == status
