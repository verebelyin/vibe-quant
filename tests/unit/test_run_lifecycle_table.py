"""Characterization of the run-lifecycle transition table (rabmo.12 / T4).

Pins every write to ``backtest_runs.status`` and ``background_jobs`` (status,
completed_at, error_message, pid) through the public APIs of ``StateManager``
and ``BacktestJobManager`` on one temp DB file, before rabmo.12 moves the SQL
into ``vibe_quant/db/run_lifecycle.py``.

Read-only helpers read the job row through ``BacktestJobManager.conn`` because
``JobInfo`` does not expose ``error_message``; every transition under test goes
through a public method. Where a row state cannot be produced by the public API
alone (an old heartbeat, pre-existing error text) the fixture pre-state is
written with SQL on the public ``conn``.
"""

from __future__ import annotations

import ast
import contextlib
import os
import signal
import sqlite3
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

import vibe_quant.db.state_manager as state_manager_module
import vibe_quant.jobs.manager as job_manager_module
from vibe_quant.db import StateManager
from vibe_quant.jobs import BacktestJobManager, JobStatus

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
SLEEPER = [sys.executable, "-c", "import time; time.sleep(30)"]
QUICK_EXIT = [sys.executable, "-c", "pass"]
STALE_HEARTBEAT = "2020-01-01 00:00:00"


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "lifecycle.db"


@pytest.fixture
def state_manager(db_path: Path) -> Iterator[StateManager]:
    manager = StateManager(db_path)
    try:
        yield manager
    finally:
        manager.close()


@pytest.fixture
def manager(db_path: Path) -> Iterator[BacktestJobManager]:
    manager = BacktestJobManager(db_path)
    try:
        yield manager
    finally:
        manager.close()


@pytest.fixture
def run_id(state_manager: StateManager) -> int:
    strategy_id = state_manager.create_strategy(name="lifecycle", dsl_config={})
    return state_manager.create_backtest_run(
        strategy_id=strategy_id,
        run_mode="screening",
        symbols=["BTCUSDT-PERP"],
        timeframe="5m",
        start_date="2024-01-01",
        end_date="2024-12-31",
        parameters={},
    )


@pytest.fixture
def spawned(manager: BacktestJobManager) -> Iterator[list[int]]:
    """Pids spawned by a test; every one is killed (and reaped) in teardown."""
    pids: list[int] = []
    try:
        yield pids
    finally:
        for pid in pids:
            with contextlib.suppress(OSError):
                os.killpg(pid, signal.SIGKILL)
        deadline = time.monotonic() + 5
        for pid in pids:
            while time.monotonic() < deadline:
                try:
                    reaped, _ = os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    break
                if reaped == pid:
                    break
                time.sleep(0.01)
        assert not any(pid for pid in pids if manager.is_process_alive(pid) is True)


# --- helpers ----------------------------------------------------------------


def _child_env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(REPO_ROOT)}


def _spawn(
    manager: BacktestJobManager,
    spawned: list[int],
    run_id: int,
    command: list[str] | None = None,
) -> int:
    pid = manager.start_job(
        run_id=run_id,
        job_type="screening",
        command=command or SLEEPER,
    )
    spawned.append(pid)
    return pid


def _job_row(manager: BacktestJobManager, run_id: int) -> dict[str, Any]:
    row = manager.conn.execute(
        "SELECT status, completed_at, error_message, pid, heartbeat_at "
        "FROM background_jobs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    assert row is not None, f"no background_jobs row for run {run_id}"
    return dict(row)


def _run_row(state_manager: StateManager, run_id: int) -> dict[str, Any]:
    row = state_manager.get_backtest_run(run_id)
    assert row is not None, f"no backtest_runs row for run {run_id}"
    return {key: row[key] for key in ("status", "completed_at", "error_message", "pid")}


def _assert_both(
    state_manager: StateManager,
    manager: BacktestJobManager,
    run_id: int,
    *,
    status: str,
    error_message: str | None,
    pid: int | None,
    completed: bool = True,
) -> None:
    """Both rows agree on (status, completed_at, error_message, pid)."""
    for label, row in (
        ("run", _run_row(state_manager, run_id)),
        ("job", _job_row(manager, run_id)),
    ):
        assert row["status"] == status, (label, row)
        assert (row["completed_at"] is not None) is completed, (label, row)
        assert row["error_message"] == error_message, (label, row)
        assert row["pid"] == pid, (label, row)


def _wait_for(predicate: Any, what: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


def _wait_exited(manager: BacktestJobManager, pid: int) -> None:
    _wait_for(lambda: not manager.is_process_alive(pid), f"pid {pid} to exit")


# --- R47: transitions -------------------------------------------------------


def test_start_job_opens_both_rows(
    state_manager: StateManager,
    manager: BacktestJobManager,
    run_id: int,
    spawned: list[int],
) -> None:
    """start_job: run running (pid set); job pending -> running."""
    assert _run_row(state_manager, run_id)["status"] == "pending"

    pid = _spawn(manager, spawned, run_id)

    _assert_both(
        state_manager,
        manager,
        run_id,
        status="running",
        error_message=None,
        pid=pid,
        completed=False,
    )


def test_mark_completed_marks_both_completed(
    state_manager: StateManager,
    manager: BacktestJobManager,
    run_id: int,
    spawned: list[int],
) -> None:
    pid = _spawn(manager, spawned, run_id)

    manager.mark_completed(run_id)

    _assert_both(state_manager, manager, run_id, status="completed", error_message=None, pid=pid)


def test_mark_completed_with_error_fails_both(
    state_manager: StateManager,
    manager: BacktestJobManager,
    run_id: int,
    spawned: list[int],
) -> None:
    pid = _spawn(manager, spawned, run_id)

    manager.mark_completed(run_id, error="boom")

    _assert_both(state_manager, manager, run_id, status="failed", error_message="boom", pid=pid)


def test_complete_never_overwrites_failed_run(
    state_manager: StateManager,
    manager: BacktestJobManager,
    run_id: int,
    spawned: list[int],
) -> None:
    """R47: completed after run failed — job adopts the run's failed + error."""
    pid = _spawn(manager, spawned, run_id)
    state_manager.update_backtest_run_status(run_id, "failed", error_message="runner failure")

    manager.mark_completed(run_id)

    _assert_both(
        state_manager,
        manager,
        run_id,
        status="failed",
        error_message="runner failure",
        pid=pid,
    )


def test_failed_after_kill_stays_killed(
    state_manager: StateManager,
    manager: BacktestJobManager,
    run_id: int,
    spawned: list[int],
) -> None:
    """R47: failed after killed stays killed; error_message unchanged."""
    pid = _spawn(manager, spawned, run_id)
    assert manager.kill_job(run_id, force=True) is True

    manager.mark_completed(run_id, error="late failure")

    _assert_both(state_manager, manager, run_id, status="killed", error_message=None, pid=pid)


def test_kill_job_marks_both_killed(
    state_manager: StateManager,
    manager: BacktestJobManager,
    run_id: int,
    spawned: list[int],
) -> None:
    pid = _spawn(manager, spawned, run_id)

    assert manager.kill_job(run_id, force=True) is True

    _assert_both(state_manager, manager, run_id, status="killed", error_message=None, pid=pid)


def test_sync_job_status_dead_process_fails_both(
    state_manager: StateManager,
    manager: BacktestJobManager,
    run_id: int,
    spawned: list[int],
) -> None:
    """sync_job_status on a dead pid: failed 'Process terminated unexpectedly'."""
    pid = _spawn(manager, spawned, run_id, QUICK_EXIT)
    _wait_exited(manager, pid)

    assert manager.sync_job_status(run_id) == JobStatus.FAILED

    _assert_both(
        state_manager,
        manager,
        run_id,
        status="failed",
        error_message="Process terminated unexpectedly",
        pid=pid,
    )


def test_cleanup_stale_jobs_fails_both(
    state_manager: StateManager,
    manager: BacktestJobManager,
    run_id: int,
    spawned: list[int],
) -> None:
    """cleanup_stale: failed 'Job stale - no heartbeat for >120s'."""
    pid = _spawn(manager, spawned, run_id)
    manager.conn.execute(
        "UPDATE background_jobs SET heartbeat_at = ? WHERE run_id = ?",
        (STALE_HEARTBEAT, run_id),
    )
    manager.conn.commit()

    assert manager.cleanup_stale_jobs() == 1

    _assert_both(
        state_manager,
        manager,
        run_id,
        status="failed",
        error_message="Job stale - no heartbeat for >120s",
        pid=pid,
    )


def test_reconcile_dead_process_fails_both(
    state_manager: StateManager,
    manager: BacktestJobManager,
    run_id: int,
    spawned: list[int],
) -> None:
    """reconcile: failed 'Process not running (exited without reporting status, or PID reused)'."""
    pid = _spawn(manager, spawned, run_id, QUICK_EXIT)
    _wait_exited(manager, pid)

    assert manager.reconcile_jobs() == 1

    _assert_both(
        state_manager,
        manager,
        run_id,
        status="failed",
        error_message=("Process not running (exited without reporting status, or PID reused)"),
        pid=pid,
    )


def test_reconcile_ignores_non_running_job(
    state_manager: StateManager,
    manager: BacktestJobManager,
    run_id: int,
    spawned: list[int],
) -> None:
    """R47: reconcile on a non-running job writes nothing (returns 0)."""
    _spawn(manager, spawned, run_id)
    manager.mark_completed(run_id)
    before_run = _run_row(state_manager, run_id)
    before_job = _job_row(manager, run_id)

    assert manager.reconcile_jobs() == 0

    assert _run_row(state_manager, run_id) == before_run
    assert _job_row(manager, run_id) == before_job


def test_update_run_status_running_on_terminal_run_is_noop(
    state_manager: StateManager,
    manager: BacktestJobManager,
    run_id: int,
) -> None:
    """R47: update_backtest_run_status(running) on a terminal run changes nothing."""
    state_manager.update_backtest_run_status(run_id, "failed", error_message="first")
    before = _run_row(state_manager, run_id)

    state_manager.update_backtest_run_status(run_id, "running", pid=4242)

    assert _run_row(state_manager, run_id) == before


# --- R48: error_message composition per writer ------------------------------


def test_error_message_composition(
    state_manager: StateManager,
    manager: BacktestJobManager,
    run_id: int,
    spawned: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # update_backtest_run_status: overwrite only when an error is given.
    state_manager.update_backtest_run_status(run_id, "failed", error_message="e1")
    assert _run_row(state_manager, run_id)["error_message"] == "e1"

    state_manager.update_backtest_run_status(run_id, "completed")
    assert _run_row(state_manager, run_id)["status"] == "completed"
    assert _run_row(state_manager, run_id)["error_message"] == "e1"

    state_manager.update_backtest_run_status(run_id, "failed", error_message="e2")
    assert _run_row(state_manager, run_id)["error_message"] == "e2"

    # finish: COALESCE(new, old) — a None error keeps the run's existing text.
    pid = _spawn(manager, spawned, run_id)
    state_manager.update_backtest_run_status(run_id, "failed", error_message="runner failure")
    assert manager.kill_job(run_id, force=True) is True
    run = _run_row(state_manager, run_id)
    job = _job_row(manager, run_id)
    assert (run["status"], run["error_message"]) == ("killed", "runner failure")
    assert (job["status"], job["error_message"]) == ("killed", None)
    assert pid == run["pid"] == job["pid"]

    # reap: reason alone, then old text + "\n" + reason.
    reaped_run = _create_run(state_manager)
    assert manager.reap_orphan_runs() == [reaped_run]
    assert _run_row(state_manager, reaped_run)["error_message"] == (
        "reaped at startup: no live job (was pending)"
    )

    older_run = _create_run(state_manager)
    state_manager.update_backtest_run_status(older_run, "running")
    state_manager.conn.execute(
        "UPDATE backtest_runs SET error_message = 'old' WHERE id = ?", (older_run,)
    )
    state_manager.conn.commit()
    assert manager.reap_orphan_runs() == [older_run]
    assert _run_row(state_manager, older_run)["error_message"] == (
        "old\nreaped at startup: no live job (was running)"
    )

    # claim with identity None: the refusal text overwrites both rows. A job row
    # whose start time is NULL is the only public route into that refusal.
    claimed_run = _create_run(state_manager)
    original_start_time = job_manager_module.process_start_time
    monkeypatch.setattr(job_manager_module, "process_start_time", lambda pid: None)
    claimed_pid = _spawn(manager, spawned, claimed_run)
    monkeypatch.setattr(job_manager_module, "process_start_time", original_start_time)
    refusal = (
        f"Refusing to trade: cannot read this process's start time (pid {claimed_pid}), "
        "so stop/kill could not verify it"
    )
    assert manager.claim_process(claimed_run, claimed_pid, None) == refusal
    _assert_both(
        state_manager,
        manager,
        claimed_run,
        status="failed",
        error_message=refusal,
        pid=claimed_pid,
    )

    # open_attempt: NULL.
    resumed_run = _create_run(state_manager)
    state_manager.update_backtest_run_status(resumed_run, "failed", error_message="previous")
    resumed_pid = _spawn(manager, spawned, resumed_run)
    _assert_both(
        state_manager,
        manager,
        resumed_run,
        status="running",
        error_message=None,
        pid=resumed_pid,
        completed=False,
    )


def _create_run(state_manager: StateManager) -> int:
    strategy_id = state_manager.create_strategy(name=f"lifecycle-{time.time_ns()}", dsl_config={})
    return state_manager.create_backtest_run(
        strategy_id=strategy_id,
        run_mode="screening",
        symbols=["BTCUSDT-PERP"],
        timeframe="5m",
        start_date="2024-01-01",
        end_date="2024-12-31",
        parameters={},
    )


# --- R49: child reports completed before parent registration ----------------


def test_child_reports_completed_before_parent_registration(
    state_manager: StateManager,
    manager: BacktestJobManager,
    run_id: int,
    spawned: list[int],
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R49: the registration is monotonic — a child's terminal result stands."""
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from vibe_quant.jobs.manager import BacktestJobManager\n"
        "db, run_id = Path(sys.argv[1]), int(sys.argv[2])\n"
        "job_manager = BacktestJobManager(db)\n"
        "job_manager.mark_completed(run_id)\n"
        "job_manager.close()\n"
    )
    original = job_manager_module.process_start_time

    def delayed(pid: int) -> str | None:
        # Runs after the spawn, before the parent's registration UPDATE.
        manager.conn.execute(
            "UPDATE background_jobs SET heartbeat_at = ? WHERE run_id = ?",
            (STALE_HEARTBEAT, run_id),
        )
        manager.conn.commit()
        _wait_for(
            lambda: _job_row(manager, run_id)["status"] == "completed",
            "the child to report completed",
        )
        return original(pid)

    monkeypatch.setattr(job_manager_module, "process_start_time", delayed)

    pid = manager.start_job(
        run_id=run_id,
        job_type="screening",
        command=[sys.executable, "-c", script, str(db_path), str(run_id)],
        env=_child_env(),
    )
    spawned.append(pid)

    assert pid > 0
    _assert_both(state_manager, manager, run_id, status="completed", error_message=None, pid=pid)
    assert _job_row(manager, run_id)["heartbeat_at"] == STALE_HEARTBEAT


# --- R49b: registration refusal reads the owner before COMMIT ---------------


def test_registration_refusal_reads_owner_before_commit(
    state_manager: StateManager,
    manager: BacktestJobManager,
    run_id: int,
    spawned: list[int],
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R49b: UPDATE background_jobs -> SELECT pid -> COMMIT, ValueError pid=777."""
    _spawn(manager, spawned, run_id)
    assert manager.kill_job(run_id, force=True) is True

    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from vibe_quant.jobs.manager import BacktestJobManager\n"
        "db, run_id = Path(sys.argv[1]), int(sys.argv[2])\n"
        "job_manager = BacktestJobManager(db)\n"
        "job_manager.claim_process(run_id, 777, 'other-attempt')\n"
        "job_manager.close()\n"
    )
    original = job_manager_module.process_start_time

    def delayed(pid: int) -> str | None:
        _wait_for(
            lambda: _job_row(manager, run_id)["pid"] == 777,
            "the other attempt to claim the job row",
        )
        return original(pid)

    monkeypatch.setattr(job_manager_module, "process_start_time", delayed)

    statements: list[str] = []
    manager.conn.set_trace_callback(statements.append)
    try:
        with pytest.raises(
            ValueError, match=rf"^Run {run_id} already has an active job \(pid=777\)$"
        ):
            manager.start_job(
                run_id=run_id,
                job_type="screening",
                command=[sys.executable, "-c", script, str(db_path), str(run_id)],
                env=_child_env(),
            )
    finally:
        manager.conn.set_trace_callback(None)

    # set_trace_callback expands parameters, so match the statement shapes.
    normalized = [" ".join(statement.split()) for statement in statements]
    registration = [
        index
        for index, statement in enumerate(normalized)
        if statement.startswith("UPDATE background_jobs SET pid = ")
        and "pid_start_time = NULL" not in statement
    ]
    assert registration, normalized
    tail = normalized[registration[-1] :]
    assert f"WHERE run_id = {run_id} AND pid IN (0, " in tail[0], tail[0]
    assert tail[1] == f"SELECT pid FROM background_jobs WHERE run_id = {run_id}"
    assert tail[2] == "COMMIT"
    assert len(tail) == 3


# --- R52: locking -----------------------------------------------------------


def _method_holds_lock(method: ast.FunctionDef, attr: str) -> bool:
    for node in ast.walk(method):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            context = item.context_expr
            if (
                isinstance(context, ast.Attribute)
                and context.attr == attr
                and isinstance(context.value, ast.Name)
                and context.value.id == "self"
            ):
                return True
    return False


def test_lifecycle_writes_hold_write_lock() -> None:
    """R52: StateManager lifecycle writes run inside ``with self._write_lock``."""
    source = Path(state_manager_module.__file__).read_text()
    tree = ast.parse(source)
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "StateManager"
    )
    methods = {node.name: node for node in class_node.body if isinstance(node, ast.FunctionDef)}
    for name in ("update_backtest_run_status", "update_heartbeat"):
        assert name in methods, name
        assert _method_holds_lock(methods[name], "_write_lock"), name


# --- R53: dead methods / unique run_id --------------------------------------


def test_duplicate_job_run_id_rejected(state_manager: StateManager, run_id: int) -> None:
    """R53: a second background_jobs row for one run still raises IntegrityError."""
    conn = state_manager.conn
    conn.execute(
        "INSERT INTO background_jobs (run_id, pid, job_type) VALUES (?, 1, 'screening')",
        (run_id,),
    )
    conn.commit()
    try:
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
            conn.execute(
                "INSERT INTO background_jobs (run_id, pid, job_type) VALUES (?, 2, 'screening')",
                (run_id,),
            )
    finally:
        conn.rollback()
