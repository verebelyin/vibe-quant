"""A child's terminal result survives the parent's launch bookkeeping (6ub67 F1, I7).

The real ``/restore`` child (``vibe_quant.paper.cli``) refuses a rotated key and
records a curated terminal failure on the run and job rows while the API parent
is still inside ``start_job`` / ``_launch``. The probes below force both parent/
child interleavings with the actual CLI child and database; the terminal write
is never replaced. Status transitions are monotonic: terminal never goes back to
``running`` and its error text is never cleared by a later launch write.
"""

from __future__ import annotations

import functools
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.unit import test_audit_paper_api as audit
from vibe_quant.jobs import manager as manager_module
from vibe_quant.jobs.manager import BacktestJobManager

if TYPE_CHECKING:
    from vibe_quant.db.state_manager import StateManager

ctx = audit.ctx
tmp_db = audit.tmp_db

CURATED = (
    "Refusing LIVE start: no saved risk state for this account (new or rotated "
    "API key, or first live session): drawdown, daily-loss and losing-streak "
    "limits would start from zero; re-run with --confirm-fresh-live to confirm"
)


def _prepare_rotated_key_restore(ctx: audit.Ctx, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BINANCE_API_KEY", "rotated-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "rotated-secret")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    monkeypatch.setattr(
        ctx.jobs, "start_job", functools.partial(BacktestJobManager.start_job, ctx.jobs)
    )
    audit._seed_account_checkpoint(ctx.tmp / "api.db", api_key="old-key", testnet=False)
    cfg_dir = ctx.tmp / "paper_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "paper_1.json").write_text(
        json.dumps(
            {
                "trader_id": "LIVE-001",
                "strategy_id": ctx.strategy_id,
                "symbols": ["BTCUSDT"],
                "binance": {"testnet": False},
            }
        )
    )


def _assert_curated_failure(ctx: audit.Ctx, run_id: int) -> None:
    ctx.jobs.sync_job_status(run_id)
    run = ctx.state.get_backtest_run(run_id)
    assert run is not None
    job = ctx.jobs.conn.execute(
        "SELECT status, error_message FROM background_jobs WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert (run["status"], run["error_message"]) == ("failed", CURATED)
    assert (job["status"], job["error_message"]) == ("failed", CURATED)


async def test_restore_terminal_refusal_survives_delayed_parent(
    ctx: audit.Ctx, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parent resumes at the router's running update AFTER the child refused."""
    original = ctx.state.update_backtest_run_status
    seen_child_failure: list[bool] = []

    def delayed_update(
        run_id: int, status: str, pid: int | None = None, error_message: str | None = None
    ) -> None:
        if status == "running":
            deadline = time.monotonic() + 30
            row = None
            while time.monotonic() < deadline:
                row = ctx.jobs.conn.execute(
                    "SELECT status FROM background_jobs WHERE run_id = ?", (run_id,)
                ).fetchone()
                if row and row["status"] == "failed":
                    break
                time.sleep(0.02)
            assert row and row["status"] == "failed", "real child did not refuse"
            seen_child_failure.append(True)
        original(run_id, status, pid=pid, error_message=error_message)

    monkeypatch.setattr(ctx.state, "update_backtest_run_status", delayed_update)
    _prepare_rotated_key_restore(ctx, monkeypatch)

    response = await ctx.ac.post(
        "/api/paper/restore", json={"trader_id": "LIVE-001", "confirm_live": True}
    )
    assert response.status_code == 201, response.text
    assert seen_child_failure == [True]  # the interleaving really happened
    _assert_curated_failure(ctx, response.json()["run_id"])


async def test_restore_terminal_refusal_survives_delayed_registration(
    ctx: audit.Ctx, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parent registers the job AFTER the child already refused."""
    original = manager_module.process_start_time
    seen_child_failure: list[bool] = []

    def delayed_identity(pid: int) -> str | None:
        deadline = time.monotonic() + 30
        row = None
        while time.monotonic() < deadline:
            row = ctx.state.conn.execute(
                "SELECT status FROM backtest_runs WHERE run_mode = 'paper' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row and row["status"] == "failed":
                break
            time.sleep(0.02)
        assert row and row["status"] == "failed", "real child did not refuse"
        seen_child_failure.append(True)
        return original(pid)

    monkeypatch.setattr(manager_module, "process_start_time", delayed_identity)
    _prepare_rotated_key_restore(ctx, monkeypatch)

    response = await ctx.ac.post(
        "/api/paper/restore", json={"trader_id": "LIVE-001", "confirm_live": True}
    )
    assert response.status_code == 201, response.text
    assert seen_child_failure == [True]
    monkeypatch.setattr(manager_module, "process_start_time", original)
    _assert_curated_failure(ctx, response.json()["run_id"])


# --- the shared helper itself (I7) ---


def _paper_run(state: StateManager, status: str, error: str | None) -> int:
    sid = state.create_strategy(f"s_{status}", audit.DSL)
    run_id = state.create_backtest_run(
        strategy_id=sid,
        run_mode="paper",
        symbols=["BTCUSDT"],
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-12-31",
        parameters={},
    )
    if status != "pending":
        state.update_backtest_run_status(run_id, status, error_message=error)
    return run_id


@pytest.mark.parametrize("terminal", ["failed", "completed", "cancelled", "killed"])
def test_running_update_never_reopens_a_terminal_run(ctx: audit.Ctx, terminal: str) -> None:
    run_id = _paper_run(ctx.state, terminal, "curated text")

    ctx.state.update_backtest_run_status(run_id, "running", pid=999)

    run = ctx.state.get_backtest_run(run_id)
    assert run is not None
    assert (run["status"], run["error_message"], run["pid"]) == (terminal, "curated text", None)


def test_running_update_still_opens_a_pending_run(ctx: audit.Ctx) -> None:
    run_id = _paper_run(ctx.state, "pending", None)

    ctx.state.update_backtest_run_status(run_id, "running", pid=999)

    run = ctx.state.get_backtest_run(run_id)
    assert run is not None
    assert (run["status"], run["pid"]) == ("running", 999)


def test_relaunch_of_a_failed_run_is_a_fresh_attempt(ctx: audit.Ctx, tmp_path: Path) -> None:
    """A deliberate re-launch (start_job) still reopens a terminal run and clears its error."""
    run_id = _paper_run(ctx.state, "failed", "old failure")
    start = functools.partial(BacktestJobManager.start_job, ctx.jobs)

    pid = start(run_id, "paper", ["sleep", "5"])
    try:
        run = ctx.state.get_backtest_run(run_id)
        job = ctx.jobs.conn.execute(
            "SELECT status, pid, error_message FROM background_jobs WHERE run_id = ?", (run_id,)
        ).fetchone()
        assert run is not None
        assert (run["status"], run["error_message"], run["pid"]) == ("running", None, pid)
        assert (job["status"], job["pid"], job["error_message"]) == ("running", pid, None)
    finally:
        ctx.jobs.kill_job(run_id, force=True)


def test_failed_spawn_leaves_the_prior_rows_untouched(ctx: audit.Ctx, tmp_path: Path) -> None:
    run_id = _paper_run(ctx.state, "failed", "old failure")
    start = functools.partial(BacktestJobManager.start_job, ctx.jobs)

    with pytest.raises(FileNotFoundError):
        start(run_id, "paper", [str(tmp_path / "no-such-binary")])

    run = ctx.state.get_backtest_run(run_id)
    assert run is not None
    assert (run["status"], run["error_message"]) == ("failed", "old failure")
    n = ctx.jobs.conn.execute(
        "SELECT COUNT(*) FROM background_jobs WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    assert n == 0


class _CloseFailsAfterChildRefused:
    """Log handle whose ``close`` raises once the real child has refused."""

    def __init__(self, handle: object, conn: object) -> None:
        self._handle = handle
        self._conn = conn
        self.saw_child_failure = False

    def fileno(self) -> int:
        return self._handle.fileno()  # type: ignore[attr-defined, no-any-return]

    def close(self) -> None:
        deadline = time.monotonic() + 30
        row = None
        while time.monotonic() < deadline:
            row = self._conn.execute(  # type: ignore[attr-defined]
                "SELECT status FROM backtest_runs WHERE run_mode = 'paper' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row and row["status"] == "failed":
                break
            time.sleep(0.02)
        assert row and row["status"] == "failed", "real child did not refuse"
        self.saw_child_failure = True
        self._handle.close()  # type: ignore[attr-defined]
        raise OSError("injected close failure")


async def test_restore_terminal_refusal_survives_log_close_error(
    ctx: audit.Ctx, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Closing the parent's log fd fails AFTER Popen succeeded and the child refused (I8)."""
    handles: list[_CloseFailsAfterChildRefused] = []

    class _Path(type(Path())):  # type: ignore[misc]
        def open(self, *args: object, **kwargs: object) -> object:  # type: ignore[override]
            handle = _CloseFailsAfterChildRefused(super().open(*args, **kwargs), ctx.state.conn)
            handles.append(handle)
            return handle

    monkeypatch.setattr(manager_module, "Path", _Path)
    _prepare_rotated_key_restore(ctx, monkeypatch)

    with caplog.at_level(logging.WARNING, logger=manager_module.__name__):
        response = await ctx.ac.post(
            "/api/paper/restore", json={"trader_id": "LIVE-001", "confirm_live": True}
        )
    assert [h.saw_child_failure for h in handles] == [True]  # the close really failed
    monkeypatch.setattr(manager_module, "Path", Path)
    run_id = ctx.state.conn.execute(
        "SELECT id FROM backtest_runs WHERE run_mode = 'paper' ORDER BY id DESC LIMIT 1"
    ).fetchone()["id"]

    warnings = [
        record
        for record in caplog.records
        if record.name == manager_module.__name__ and record.levelno >= logging.WARNING
    ]
    assert [record.getMessage() for record in warnings] == [
        f"start_job run_id={run_id}: closing the log handle failed (OSError)"
    ]
    assert all(record.exc_info is None for record in warnings)
    assert "injected close failure" not in caplog.text

    _assert_curated_failure(ctx, run_id)
    assert response.status_code == 201, response.text
    assert response.json()["run_id"] == run_id
