"""qgtdz: the paper child records its own pid + start time before trading.

The API parent's ``start_job`` writes the job row only after ``Popen``; a backend
death in that window leaves it ``pending`` with pid 0 and every stop/kill/startup
path (which only look at ``running``) never sees the child. The child therefore
claims the rows itself, and refuses to trade when it cannot.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import sqlite3
import subprocess
from typing import TYPE_CHECKING

import pytest

from vibe_quant.db import StateManager
from vibe_quant.jobs import manager as manager_module
from vibe_quant.jobs.manager import BacktestJobManager, process_start_time
from vibe_quant.paper.account_lock import account_lock_holder
from vibe_quant.paper.cli import run_with_config
from vibe_quant.paper.config import BinanceTestnetConfig, account_id

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@pytest.fixture(autouse=True)
def _binance_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BINANCE_TESTNET_API_KEY", "test_key")
    monkeypatch.setenv("BINANCE_TESTNET_API_SECRET", "test_secret")


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "jobs.db"


@pytest.fixture
def manager(db_path: Path) -> Generator[BacktestJobManager]:
    mgr = BacktestJobManager(db_path)
    yield mgr
    mgr.close()


@pytest.fixture
def state_manager(db_path: Path) -> Generator[StateManager]:
    mgr = StateManager(db_path)
    yield mgr
    mgr.close()


@pytest.fixture
def paper_run(state_manager: StateManager) -> int:
    strategy_id = state_manager.create_strategy(name="paper_strat", dsl_config={})
    return state_manager.create_backtest_run(
        strategy_id=strategy_id,
        run_mode="paper",
        symbols=["BTCUSDT"],
        timeframe="",
        start_date="",
        end_date="",
        parameters={},
    )


def _job_row(manager: BacktestJobManager, run_id: int) -> tuple[str, int, str | None]:
    row = manager.conn.execute(
        "SELECT status, pid, pid_start_time FROM background_jobs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    assert row is not None
    return str(row["status"]), int(row["pid"]), row["pid_start_time"]


def _job_error(manager: BacktestJobManager, run_id: int) -> tuple[str, str | None]:
    row = manager.conn.execute(
        "SELECT status, error_message FROM background_jobs WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert row is not None
    return str(row["status"]), row["error_message"]


def _run_row(manager: BacktestJobManager, run_id: int) -> tuple[str, int | None]:
    row = manager.conn.execute(
        "SELECT status, pid FROM backtest_runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert row is not None
    return str(row["status"]), (None if row["pid"] is None else int(row["pid"]))


def _run_error(manager: BacktestJobManager, run_id: int) -> tuple[str, str | None]:
    row = manager.conn.execute(
        "SELECT status, error_message FROM backtest_runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert row is not None
    return str(row["status"]), row["error_message"]


def _open_pending(manager: BacktestJobManager, run_id: int) -> None:
    with manager._db_lock:
        manager._open_attempt(run_id, "paper", None, exists=False)


def _start_sleep() -> subprocess.Popen[bytes]:
    return subprocess.Popen(["sleep", "30"], start_new_session=True)


def _kill(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.kill()
        proc.wait(5)


def test_child_claims_the_pending_row_a_dead_parent_left(
    manager: BacktestJobManager,
    paper_run: int,
    db_path: Path,
) -> None:
    """R3: pending pid 0 is claimed, then every operator path reaches the child."""
    victim = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        with manager._db_lock:
            manager._open_attempt(paper_run, "paper", None, exists=False)
        identity = process_start_time(victim.pid)
        assert identity is not None

        assert manager.claim_process(paper_run, victim.pid, identity) is None

        assert _job_row(manager, paper_run) == ("running", victim.pid, identity)
        assert _run_row(manager, paper_run) == ("running", victim.pid)

        fresh = BacktestJobManager(db_path)
        try:
            assert fresh.reconcile_jobs() == 0
            assert fresh.reap_orphan_runs() == []
            assert [job.run_id for job in fresh.list_active_jobs()] == [paper_run]
            assert fresh.kill_job(paper_run, force=True) is True
        finally:
            fresh.close()

        assert _job_row(manager, paper_run)[0] == "killed"
        assert _run_row(manager, paper_run)[0] == "killed"
        assert victim.wait(5) == -9
    finally:
        if victim.poll() is None:
            victim.kill()
            victim.wait(5)


def test_claiming_twice_is_idempotent(manager: BacktestJobManager, paper_run: int) -> None:
    """R2: the parent's later UPDATE records the same values the child already wrote."""
    victim = _start_sleep()
    try:
        _open_pending(manager, paper_run)
        identity = process_start_time(victim.pid)
        assert identity is not None

        assert manager.claim_process(paper_run, victim.pid, identity) is None
        rows = (_job_row(manager, paper_run), _run_row(manager, paper_run))
        assert rows == (("running", victim.pid, identity), ("running", victim.pid))

        assert manager.claim_process(paper_run, victim.pid, identity) is None
        assert (_job_row(manager, paper_run), _run_row(manager, paper_run)) == rows
    finally:
        _kill(victim)


@pytest.mark.parametrize("status", ["failed", "completed", "killed", "cancelled"])
def test_refuses_a_run_that_is_already_terminal(
    manager: BacktestJobManager,
    state_manager: StateManager,
    paper_run: int,
    status: str,
) -> None:
    """R6: a terminal run row is never reopened by a claim."""
    victim = _start_sleep()
    try:
        _open_pending(manager, paper_run)
        state_manager.update_backtest_run_status(paper_run, status, error_message="old")
        identity = process_start_time(victim.pid)
        assert identity is not None

        refusal = manager.claim_process(paper_run, victim.pid, identity)

        assert refusal == (
            f"Refusing to trade: run {paper_run} is already {status} (job pending); "
            "a finished run is never reopened"
        )
        assert _job_row(manager, paper_run) == ("pending", 0, None)
        assert _run_row(manager, paper_run) == (status, None)
        assert _run_error(manager, paper_run) == (status, "old")
    finally:
        _kill(victim)


def test_refuses_a_job_row_that_is_already_terminal(
    manager: BacktestJobManager, paper_run: int
) -> None:
    """R6: a terminal job row refuses too, even while the run still reads running."""
    victim = _start_sleep()
    try:
        _open_pending(manager, paper_run)
        manager.conn.execute(
            "UPDATE background_jobs SET status = 'killed' WHERE run_id = ?", (paper_run,)
        )
        manager.conn.commit()
        identity = process_start_time(victim.pid)
        assert identity is not None

        refusal = manager.claim_process(paper_run, victim.pid, identity)

        assert refusal == (
            f"Refusing to trade: run {paper_run} is already running (job killed); "
            "a finished run is never reopened"
        )
        assert _job_row(manager, paper_run) == ("killed", 0, None)
        assert _run_row(manager, paper_run) == ("running", None)
    finally:
        _kill(victim)


def test_refuses_a_row_recorded_for_another_pid(
    manager: BacktestJobManager, paper_run: int
) -> None:
    """R7: the row belongs to another live attempt; this process must stand down."""
    victim = _start_sleep()
    try:
        _open_pending(manager, paper_run)
        manager.conn.execute(
            """UPDATE background_jobs
               SET status = 'running', pid = 4242, pid_start_time = 'darwin:1.000000'
               WHERE run_id = ?""",
            (paper_run,),
        )
        manager.conn.execute("UPDATE backtest_runs SET pid = 4242 WHERE id = ?", (paper_run,))
        manager.conn.commit()
        identity = process_start_time(victim.pid)
        assert identity is not None

        refusal = manager.claim_process(paper_run, victim.pid, identity)

        assert refusal == (
            f"Refusing to trade: run {paper_run} is recorded for pid 4242, "
            f"not this process (pid {victim.pid})"
        )
        assert _job_row(manager, paper_run) == ("running", 4242, "darwin:1.000000")
        assert _run_row(manager, paper_run) == ("running", 4242)
    finally:
        _kill(victim)


def test_parent_registration_never_overwrites_a_child_pid(
    manager: BacktestJobManager,
    paper_run: int,
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R8: the child claimed between ``_open_attempt`` and the parent's UPDATE."""
    victim = _start_sleep()
    identity = process_start_time(victim.pid)
    assert identity is not None
    spawned: list[int] = []
    real_popen = subprocess.Popen

    def _popen_then_child_claims(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        proc = real_popen(*args, **kwargs)  # type: ignore[arg-type]
        spawned.append(proc.pid)
        child = BacktestJobManager(db_path)
        try:
            assert child.claim_process(paper_run, victim.pid, identity) is None
        finally:
            child.close()
        return proc

    monkeypatch.setattr(manager_module.subprocess, "Popen", _popen_then_child_claims)
    try:
        with pytest.raises(
            ValueError, match=rf"already has an active job \(pid={victim.pid}\)"
        ):
            manager.start_job(paper_run, "paper", ["sleep", "30"])

        assert _job_row(manager, paper_run) == ("running", victim.pid, identity)
        assert _run_row(manager, paper_run) == ("running", victim.pid)
    finally:
        for pid in spawned:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGKILL)
        _kill(victim)


def test_parent_registration_accepts_a_child_that_claimed_its_own_pid(
    manager: BacktestJobManager,
    paper_run: int,
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R2: a child claiming its own pid does not make the parent's UPDATE a no-op."""
    spawned: list[int] = []
    identities: dict[int, str] = {}
    real_popen = subprocess.Popen

    def _popen_then_child_claims(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        proc = real_popen(*args, **kwargs)  # type: ignore[arg-type]
        spawned.append(proc.pid)
        identity = process_start_time(proc.pid)
        assert identity is not None
        identities[proc.pid] = identity
        child = BacktestJobManager(db_path)
        try:
            assert child.claim_process(paper_run, proc.pid, identity) is None
        finally:
            child.close()
        return proc

    monkeypatch.setattr(manager_module.subprocess, "Popen", _popen_then_child_claims)
    try:
        pid = manager.start_job(paper_run, "paper", ["sleep", "30"])

        assert pid in spawned
        assert _job_row(manager, paper_run) == ("running", pid, identities[pid])
        assert _run_row(manager, paper_run) == ("running", pid)
    finally:
        for spawned_pid in spawned:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(spawned_pid, signal.SIGKILL)


def test_late_completion_never_touches_a_relaunched_row(
    manager: BacktestJobManager,
    paper_run: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R10: after a re-launch the old process's result is not the new attempt's."""
    victim = _start_sleep()
    new_pid: int | None = None
    try:
        _open_pending(manager, paper_run)
        identity = process_start_time(victim.pid)
        assert identity is not None
        assert manager.claim_process(paper_run, victim.pid, identity) is None
        assert manager.kill_job(paper_run, force=True) is True

        new_pid = manager.start_job(paper_run, "paper", ["sleep", "30"])

        with caplog.at_level(logging.WARNING):
            manager.mark_completed(paper_run)

        warnings = [r for r in caplog.records if "result not recorded" in r.getMessage()]
        assert len(warnings) == 1
        assert warnings[0].exc_info is None
        assert _job_row(manager, paper_run)[:2] == ("running", new_pid)
        assert _run_row(manager, paper_run) == ("running", new_pid)
    finally:
        if new_pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(new_pid, signal.SIGKILL)
        _kill(victim)


def test_refuses_without_a_job_row(manager: BacktestJobManager, paper_run: int) -> None:
    """R12: a ``--run-id`` with no API launch has no job row to record into."""
    victim = _start_sleep()
    try:
        identity = process_start_time(victim.pid)
        assert identity is not None

        refusal = manager.claim_process(paper_run, victim.pid, identity)

        assert refusal == (
            f"Refusing to trade: run {paper_run} has no job or run row "
            "to record this process in"
        )
        count = manager.conn.execute(
            "SELECT COUNT(*) FROM background_jobs WHERE run_id = ?", (paper_run,)
        ).fetchone()[0]
        assert count == 0
        assert _run_row(manager, paper_run) == ("pending", None)
    finally:
        _kill(victim)


def test_unreadable_own_start_time_fails_both_rows(
    manager: BacktestJobManager, paper_run: int
) -> None:
    """R13: a claim that cannot verify this process fails closed, not open."""
    victim = _start_sleep()
    try:
        _open_pending(manager, paper_run)

        refusal = manager.claim_process(paper_run, victim.pid, None)

        message = (
            "Refusing to trade: cannot read this process's start time "
            f"(pid {victim.pid}), so stop/kill could not verify it"
        )
        assert refusal == message
        assert _job_error(manager, paper_run) == ("failed", message)
        assert _run_error(manager, paper_run) == ("failed", message)
        assert manager.get_status(paper_run) == "failed"
    finally:
        _kill(victim)


def test_refuses_a_recycled_pid_with_a_different_identity(
    manager: BacktestJobManager, paper_run: int
) -> None:
    """R15: the stored start time belongs to a previous process that held this pid."""
    victim = _start_sleep()
    try:
        _open_pending(manager, paper_run)
        manager.conn.execute(
            """UPDATE background_jobs
               SET status = 'running', pid = ?, pid_start_time = 'darwin:0.000000'
               WHERE run_id = ?""",
            (victim.pid, paper_run),
        )
        manager.conn.commit()
        identity = process_start_time(victim.pid)
        assert identity is not None
        assert identity != "darwin:0.000000"

        refusal = manager.claim_process(paper_run, victim.pid, identity)

        assert refusal == (
            f"Refusing to trade: run {paper_run} is recorded for pid "
            f"{victim.pid}, not this process (pid {victim.pid})"
        )
        assert _job_row(manager, paper_run) == ("running", victim.pid, "darwin:0.000000")
        assert _run_row(manager, paper_run) == ("running", None)
    finally:
        _kill(victim)


def _write_config(path: Path, db_path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "trader_id": "PAPER-001",
                "binance": {"testnet": True},
                "symbols": ["BTCUSDT"],
                "strategy_id": 1,
                "db_path": str(db_path),
            }
        )
    )


def _testnet_account() -> str:
    return account_id(
        BinanceTestnetConfig(api_key="test_key", api_secret="test_secret", testnet=True)
    )


@pytest.mark.asyncio
async def test_cli_records_its_own_pid_before_trading(
    db_path: Path,
    paper_run: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R1/R3: at the moment trading starts the row already names this process."""
    config_path = tmp_path / "paper_config.json"
    _write_config(config_path, db_path)
    seed = BacktestJobManager(db_path)
    try:
        _open_pending(seed, paper_run)
    finally:
        seed.close()

    observed: list[tuple[str, int, str | None]] = []

    async def _fake_run(_config: object) -> None:
        probe = BacktestJobManager(db_path)
        try:
            observed.append(_job_row(probe, paper_run))
        finally:
            probe.close()

    monkeypatch.setattr("vibe_quant.paper.cli.run_paper_trading", _fake_run)

    assert await run_with_config(config_path, run_id=paper_run) == 0

    identity = process_start_time(os.getpid())
    assert identity is not None
    assert observed == [("running", os.getpid(), identity)]


@pytest.mark.asyncio
async def test_cli_refuses_a_row_owned_by_another_pid(
    db_path: Path,
    paper_run: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """R7 through the CLI: refuse, touch nothing, never take the account lock."""
    config_path = tmp_path / "paper_config.json"
    _write_config(config_path, db_path)
    seed = BacktestJobManager(db_path)
    try:
        _open_pending(seed, paper_run)
        seed.conn.execute(
            """UPDATE background_jobs
               SET status = 'running', pid = 4242, pid_start_time = 'darwin:1.000000'
               WHERE run_id = ?""",
            (paper_run,),
        )
        seed.conn.execute("UPDATE backtest_runs SET pid = 4242 WHERE id = ?", (paper_run,))
        seed.conn.commit()
    finally:
        seed.close()

    started: list[object] = []

    async def _fake_run(config: object) -> None:
        started.append(config)

    monkeypatch.setattr("vibe_quant.paper.cli.run_paper_trading", _fake_run)

    assert await run_with_config(config_path, run_id=paper_run) == 1

    err = capsys.readouterr().err
    assert err.splitlines() == [
        f"Refusing to trade: run {paper_run} is recorded for pid 4242, "
        f"not this process (pid {os.getpid()})"
    ]
    assert started == []
    probe = BacktestJobManager(db_path)
    try:
        assert _job_row(probe, paper_run) == ("running", 4242, "darwin:1.000000")
        assert _run_row(probe, paper_run) == ("running", 4242)
    finally:
        probe.close()
    assert account_lock_holder(_testnet_account()) is None


@pytest.mark.asyncio
async def test_cli_claim_error_refuses_and_hides_the_exception_text(
    db_path: Path,
    paper_run: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R14: a claim that raises refuses with the class name only."""
    config_path = tmp_path / "paper_config.json"
    _write_config(config_path, db_path)
    seed = BacktestJobManager(db_path)
    try:
        _open_pending(seed, paper_run)
    finally:
        seed.close()

    def _boom(self: BacktestJobManager, run_id: int, pid: int, identity: str | None) -> None:
        raise sqlite3.OperationalError("SECRET-X")

    monkeypatch.setattr(BacktestJobManager, "claim_process", _boom)
    started: list[object] = []

    async def _fake_run(config: object) -> None:
        started.append(config)

    monkeypatch.setattr("vibe_quant.paper.cli.run_paper_trading", _fake_run)

    with caplog.at_level(logging.ERROR):
        rc = await run_with_config(config_path, run_id=paper_run)

    assert rc == 1
    assert capsys.readouterr().err.splitlines() == [
        f"Refusing to trade: could not record this process for run {paper_run} "
        "(OperationalError)"
    ]
    assert started == []
    assert account_lock_holder(_testnet_account()) is None
    records = [r for r in caplog.records if "paper process claim failed" in r.getMessage()]
    assert len(records) == 1
    assert records[0].getMessage() == "paper process claim failed (OperationalError)"
    assert records[0].exc_info is None
    assert "SECRET-X" not in caplog.text
    probe = BacktestJobManager(db_path)
    try:
        assert _run_row(probe, paper_run) == ("running", None)
        assert _job_row(probe, paper_run) == ("pending", 0, None)
    finally:
        probe.close()
