"""Cancel a RUNNING extraction job (vibe-quant-ma1j).

Proves the per-job interrupt path without ever launching the real ``claude``
CLI: a small fake executable plays the role of ``claude -p`` and exposes its
argv/liveness through files + environment variables.

Covered:
- extractor: the cancellable path spawns the same argv as the legacy call and
  SIGTERMs only *that* subprocess, raising ``ExtractionCancelled``.
- extractor: a child that ignores SIGTERM is SIGKILLed after the grace period
  (``grace_expired=True`` -> the worker records ``cancelled-after-timeout``).
- extractor: the timeout path still surfaces as a ``failed`` result.
- worker: two concurrent jobs; cancelling one kills only its subprocess, the
  job ends ``cancelled`` (attempts untouched, item restored) and the other
  job still completes.
- worker: re-extract after a cancel enqueues and runs again.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.research import extractor as extractor_mod
from vibe_quant.research import worker as worker_mod
from vibe_quant.research.extractor import ClaudePExtractor, ExtractionCancelled
from vibe_quant.research.schema import RawItem

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path


_FAKE_CLAUDE_PY = '''#!/usr/bin/env python3
"""Fake `claude -p` for cancel tests. Never touches the real CLI."""
import json
import os
import signal
import sys
import time

argv = sys.argv
prompt = argv[-1] if len(argv) > 1 else ""


def _touch(path):
    if path:
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("1")
        except OSError:
            pass


def _write(path, text):
    if path:
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(str(text))
        except OSError:
            pass


# Per-mode marker so two CONCURRENT jobs can be told apart in one test
# (same env, different prompt).
if "SLOWJOB" in prompt:
    mode = "slow"
elif "GATEDJOB" in prompt:
    mode = "gated"
else:
    mode = "fast"

_state_dir = os.environ.get("VQ_FAKE_STATE_DIR")


def _sp(suffix):
    return os.path.join(_state_dir, f"{suffix}_{mode}") if _state_dir else None


argv_file = os.environ.get("VQ_FAKE_ARGV_FILE")
if argv_file:
    with open(argv_file, "w", encoding="utf-8") as fh:
        json.dump(argv, fh)

slow = mode == "slow" or os.environ.get("VQ_FAKE_ALWAYS_SLOW") == "1"
gated = mode == "gated"
if slow or gated:
    _touch(os.environ.get("VQ_FAKE_STARTED_FILE"))
    _touch(_sp("started"))
    _write(os.environ.get("VQ_FAKE_PID_FILE"), os.getpid())
    _write(_sp("pid"), os.getpid())
    if os.environ.get("VQ_FAKE_IGNORE_SIGTERM") == "1":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        def _on_term(_signum, _frame):
            _touch(os.environ.get("VQ_FAKE_SIGTERM_FILE"))
            _touch(_sp("sigterm"))
            sys.exit(0)

        signal.signal(signal.SIGTERM, _on_term)
    gate = os.environ.get("VQ_FAKE_GATE_FILE")
    # GATED jobs stay alive until the gate file appears (a slow sibling that
    # must survive another job's cancel); SLOW jobs never finish on their own.
    while True:
        if gated and gate and os.path.exists(gate):
            break
        time.sleep(0.2)

if gated:
    _touch(_sp("released"))

# Fast path: a valid, empty claude envelope -> extractor yields a skipped batch.
print(json.dumps({"result": json.dumps([])}))
sys.exit(0)
'''


@pytest.fixture
def sm(tmp_path: Path) -> Generator[StateManager]:
    mgr = StateManager(tmp_path / "cancel.db")
    yield mgr
    mgr.close()


@pytest.fixture(autouse=True)
def _redirect_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_mod, "DEFAULT_LOG_ROOT", tmp_path / "worker-logs")
    from vibe_quant.research import extraction_log

    monkeypatch.setattr(extraction_log, "DEFAULT_LOG_ROOT", tmp_path / "extraction-logs")


def _fake_claude(tmp_path: Path) -> Path:
    path = tmp_path / "fake_claude.py"
    path.write_text(_FAKE_CLAUDE_PY, encoding="utf-8")
    path.chmod(0o755)
    return path


def _item(body: str = "some body text") -> RawItem:
    return RawItem(
        source="reddit",
        external_id="x",
        url="https://reddit.com/x",
        title="a strategy",
        body=body,
        author=None,
        posted_at=None,
        score=None,
        extras={"comments": []},
    )


def _seed_item(sm: StateManager, ext_id: str, body: str = "b") -> int:
    return sm.create_research_item(
        source="reddit",
        external_id=ext_id,
        url=f"https://reddit.com/{ext_id}",
        title="t",
        body=body,
        author=None,
        posted_at=None,
        score=None,
    )


def _wait_until(pred: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.05)
    return False


def _run_in_thread(fn: Callable[[], object]) -> tuple[threading.Thread, dict[str, Any]]:
    box: dict[str, Any] = {}

    def _target() -> None:
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - test harness captures all
            box["error"] = exc

    th = threading.Thread(target=_target, daemon=True)
    th.start()
    return th, box


# ---------------------------------------------------------------------------
# extractor: cancellable Popen path
# ---------------------------------------------------------------------------


def test_cancellable_extract_terminates_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake_claude(tmp_path)
    started = tmp_path / "started"
    sigterm = tmp_path / "sigterm"
    monkeypatch.setenv("VQ_FAKE_STARTED_FILE", str(started))
    monkeypatch.setenv("VQ_FAKE_SIGTERM_FILE", str(sigterm))

    ext = ClaudePExtractor(claude_path=str(fake), timeout_seconds=30)
    event = threading.Event()

    def _work() -> object:
        extractor_mod.set_thread_cancel_event(event)
        try:
            return ext.extract_all(_item("SLOWJOB"))
        finally:
            extractor_mod.clear_thread_cancel_event()

    th, box = _run_in_thread(_work)
    assert _wait_until(started.exists, timeout=5.0), "fake claude never started"
    event.set()
    th.join(timeout=10.0)
    assert not th.is_alive()

    err = box.get("error")
    assert isinstance(err, ExtractionCancelled)
    assert err.grace_expired is False
    assert sigterm.exists(), "the targeted subprocess was never SIGTERMed"


def test_cancellable_extract_uses_same_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake_claude(tmp_path)
    argv_file = tmp_path / "argv.json"
    monkeypatch.setenv("VQ_FAKE_ARGV_FILE", str(argv_file))

    ext = ClaudePExtractor(claude_path=str(fake), timeout_seconds=30)
    event = threading.Event()
    extractor_mod.set_thread_cancel_event(event)
    try:
        batch = ext.extract_all(_item())
    finally:
        extractor_mod.clear_thread_cancel_event()

    argv = json.loads(argv_file.read_text(encoding="utf-8"))
    assert argv[:4] == [str(fake), "-p", "--output-format", "json"]
    assert argv[4:] == [batch.prompt]


def test_cancel_ignoring_sigterm_is_killed_after_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake_claude(tmp_path)
    started = tmp_path / "started"
    monkeypatch.setenv("VQ_FAKE_STARTED_FILE", str(started))
    monkeypatch.setenv("VQ_FAKE_ALWAYS_SLOW", "1")
    monkeypatch.setenv("VQ_FAKE_IGNORE_SIGTERM", "1")
    monkeypatch.setattr(extractor_mod, "CANCEL_GRACE_SECONDS", 1.0)

    ext = ClaudePExtractor(claude_path=str(fake), timeout_seconds=30)
    event = threading.Event()

    def _work() -> object:
        extractor_mod.set_thread_cancel_event(event)
        try:
            return ext.extract_all(_item())
        finally:
            extractor_mod.clear_thread_cancel_event()

    th, box = _run_in_thread(_work)
    assert _wait_until(started.exists, timeout=5.0), "fake claude never started"
    event.set()
    th.join(timeout=10.0)
    assert not th.is_alive()

    err = box.get("error")
    assert isinstance(err, ExtractionCancelled)
    assert err.grace_expired is True


def test_cancellable_timeout_still_fails_as_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake_claude(tmp_path)
    monkeypatch.setenv("VQ_FAKE_ALWAYS_SLOW", "1")
    ext = ClaudePExtractor(claude_path=str(fake), timeout_seconds=0.3)

    event = threading.Event()
    extractor_mod.set_thread_cancel_event(event)
    try:
        batch = ext.extract_all(_item())
    finally:
        extractor_mod.clear_thread_cancel_event()

    assert batch.results[0].status == "failed"
    assert "timeout" in (batch.results[0].parse_error or "")


# ---------------------------------------------------------------------------
# worker: cancel one of two concurrent jobs
# ---------------------------------------------------------------------------


def _run_worker(
    sm: StateManager,
    stop: threading.Event,
    ext: ClaudePExtractor,
    *,
    log_name: str,
    concurrency: int = 1,
) -> threading.Thread:
    sink = worker_mod._JsonlSink(worker_mod.DEFAULT_LOG_ROOT / log_name)

    def runner() -> None:
        with patch(
            "vibe_quant.research.extractor.get_default_extractor",
            return_value=ext,
        ):
            worker_mod.run_forever(
                sm,
                poll_interval=0.05,
                stop_event=stop,
                sink=sink,
                concurrency=concurrency,
            )

    th = threading.Thread(target=runner, daemon=True)
    th.start()
    return th


def _job_status(sm: StateManager, job_id: int) -> str | None:
    row = sm.get_extraction_job(job_id)
    return None if row is None else str(row["status"])


def test_worker_cancels_one_running_job_other_completes(
    sm: StateManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F2: the cancel must be confined to the ONE targeted job.

    Both jobs are deliberately SLOW here (B is gated, not fast): B is still
    in flight when A is cancelled, so a shared/global cancel Event would
    also kill B. With the per-job Event B survives, keeps its child process
    alive, and only completes once we release its gate.
    """
    fake = _fake_claude(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    gate = tmp_path / "b-gate"
    monkeypatch.setenv("VQ_FAKE_STATE_DIR", str(state))
    monkeypatch.setenv("VQ_FAKE_GATE_FILE", str(gate))
    monkeypatch.setattr(worker_mod, "DEFAULT_CANCEL_POLL_INTERVAL_S", 0.05)

    iid_a = _seed_item(sm, "a", body="SLOWJOB")
    iid_b = _seed_item(sm, "b", body="GATEDJOB")
    job_a = sm.enqueue_extraction_job(iid_a)
    job_b = sm.enqueue_extraction_job(iid_b)

    stop = threading.Event()
    ext = ClaudePExtractor(claude_path=str(fake), timeout_seconds=60)
    th = _run_worker(sm, stop, ext, log_name="cancel-one.log", concurrency=2)
    try:
        # Both children must be live and both jobs running BEFORE we cancel A.
        assert _wait_until(lambda: (state / "started_slow").exists(), timeout=10.0), (
            "slow child never started"
        )
        assert _wait_until(lambda: (state / "started_gated").exists(), timeout=10.0), (
            "gated sibling never started"
        )
        assert _wait_until(lambda: _job_status(sm, job_a) == "running", timeout=5.0)
        assert _wait_until(lambda: _job_status(sm, job_b) == "running", timeout=5.0)
        pid_b = int((state / "pid_gated").read_text(encoding="utf-8"))

        requested = sm.request_cancel_extraction_job(job_a)
        assert requested is not None
        assert requested["status"] == "running"
        assert requested["cancel_requested_at"] is not None

        assert _wait_until(lambda: _job_status(sm, job_a) == "cancelled", timeout=15.0), (
            sm.get_extraction_job(job_a)
        )

        # The sibling is untouched: still running, its child still alive.
        assert _job_status(sm, job_b) == "running"
        assert (state / "sigterm_slow").exists(), "the targeted child was not signalled"
        assert not (state / "sigterm_gated").exists(), "the sibling child was signalled too"
        os.kill(pid_b, 0)  # raises OSError if the sibling process is gone

        # Release B: it must run to completion, not be collateral damage.
        gate.write_text("go", encoding="utf-8")
        assert _wait_until(lambda: _job_status(sm, job_b) == "done", timeout=15.0), (
            sm.get_extraction_job(job_b)
        )
        assert (state / "released_gated").exists()
    finally:
        stop.set()
        th.join(timeout=10.0)

    job_a_row = sm.get_extraction_job(job_a)
    job_b_row = sm.get_extraction_job(job_b)
    assert job_a_row is not None and job_b_row is not None
    assert job_a_row["status"] == "cancelled"
    # A cancel is not a failure: no retry, attempts unchanged.
    assert job_a_row["attempts"] == 0
    assert "timeout" not in (job_a_row["error_message"] or "")
    assert job_b_row["status"] == "done"
    # The cancelled item is popped out of 'running' so it can be re-extracted.
    item_a = sm.get_research_item(iid_a)
    assert item_a is not None
    assert item_a["extraction_status"] == "pending"


def test_worker_hung_subprocess_records_cancelled_after_timeout(
    sm: StateManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake_claude(tmp_path)
    started = tmp_path / "started"
    monkeypatch.setenv("VQ_FAKE_STARTED_FILE", str(started))
    monkeypatch.setenv("VQ_FAKE_ALWAYS_SLOW", "1")
    monkeypatch.setenv("VQ_FAKE_IGNORE_SIGTERM", "1")
    monkeypatch.setattr(worker_mod, "DEFAULT_CANCEL_POLL_INTERVAL_S", 0.05)
    monkeypatch.setattr(extractor_mod, "CANCEL_GRACE_SECONDS", 1.0)

    iid = _seed_item(sm, "hung", body="SLOWJOB")
    job_id = sm.enqueue_extraction_job(iid)

    stop = threading.Event()
    ext = ClaudePExtractor(claude_path=str(fake), timeout_seconds=30)
    th = _run_worker(sm, stop, ext, log_name="cancel-hung.log")
    try:
        assert _wait_until(started.exists, timeout=10.0), "slow child never started"
        assert _wait_until(lambda: _job_status(sm, job_id) == "running", timeout=5.0)
        sm.request_cancel_extraction_job(job_id)
        assert _wait_until(lambda: _job_status(sm, job_id) == "cancelled", timeout=15.0), (
            sm.get_extraction_job(job_id)
        )
    finally:
        stop.set()
        th.join(timeout=10.0)

    row = sm.get_extraction_job(job_id)
    assert row is not None
    assert row["status"] == "cancelled"
    assert row["error_message"] == "cancelled-after-timeout"


def test_worker_reextract_after_cancel_runs_again(
    sm: StateManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake_claude(tmp_path)
    started = tmp_path / "started"
    monkeypatch.setenv("VQ_FAKE_STARTED_FILE", str(started))
    monkeypatch.setattr(worker_mod, "DEFAULT_CANCEL_POLL_INTERVAL_S", 0.05)

    iid = _seed_item(sm, "retry", body="SLOWJOB")
    first_job = sm.enqueue_extraction_job(iid)

    stop = threading.Event()
    ext = ClaudePExtractor(claude_path=str(fake), timeout_seconds=30)
    th = _run_worker(sm, stop, ext, log_name="cancel-retry.log")
    try:
        assert _wait_until(started.exists, timeout=10.0)
        assert _wait_until(lambda: _job_status(sm, first_job) == "running", timeout=5.0)
        sm.request_cancel_extraction_job(first_job)
        assert _wait_until(lambda: _job_status(sm, first_job) == "cancelled", timeout=15.0)

        # Re-enqueue the same item; the fast path (no SLOWJOB marker now).
        sm.conn.execute(
            "UPDATE research_items SET body = 'FASTJOB' WHERE id = ?", (iid,)
        )
        sm.conn.commit()
        second_job = sm.enqueue_extraction_job(iid)
        assert _wait_until(lambda: _job_status(sm, second_job) == "done", timeout=15.0), (
            sm.get_extraction_job(second_job)
        )
    finally:
        stop.set()
        th.join(timeout=10.0)

    assert _job_status(sm, second_job) == "done"
    item = sm.get_research_item(iid)
    assert item is not None
    assert item["extraction_status"] != "running"


# ---------------------------------------------------------------------------
# F1: shared-connection reads must be serialized with writers
# ---------------------------------------------------------------------------


def test_get_extraction_job_serializes_on_write_lock(sm: StateManager) -> None:
    """The watcher and the router poll read the shared sqlite connection from
    their own threads while a drain thread writes. If the read is not under
    the write lock it can corrupt the connection (sqlite3.InterfaceError) and
    break the writer. Prove the read takes the lock: holding it must block the
    read."""
    iid = _seed_item(sm, "lock", body="b")
    job_id = sm.enqueue_extraction_job(iid)
    sm.claim_next_extraction_job()

    read_done = threading.Event()

    def _read() -> None:
        sm.get_extraction_job(job_id)
        read_done.set()

    sm._write_lock.acquire()
    try:
        th = threading.Thread(target=_read, daemon=True)
        th.start()
        assert not read_done.wait(0.3), (
            "get_extraction_job read the shared connection without the write lock"
        )
    finally:
        sm._write_lock.release()
    assert read_done.wait(2.0), "get_extraction_job never returned after the lock released"


def test_watcher_cancel_read_serializes_on_write_lock(sm: StateManager) -> None:
    """The per-job watcher's cancel poll must use the same locked read path."""
    iid = _seed_item(sm, "watch-lock", body="b")
    job_id = sm.enqueue_extraction_job(iid)
    sm.claim_next_extraction_job()

    read_done = threading.Event()

    def _read() -> None:
        sm.extraction_job_cancel_requested(job_id)
        read_done.set()

    sm._write_lock.acquire()
    try:
        th = threading.Thread(target=_read, daemon=True)
        th.start()
        assert not read_done.wait(0.3), (
            "extraction_job_cancel_requested read without the write lock"
        )
    finally:
        sm._write_lock.release()
    assert read_done.wait(2.0)


# ---------------------------------------------------------------------------
# F4: cancel UPDATEs carry status guards + rowcount checks
# ---------------------------------------------------------------------------


def test_mark_cancelled_guard_refuses_terminal_job(sm: StateManager) -> None:
    """A cancel must never overwrite a real outcome: the guarded UPDATE is a
    no-op on a job that crossed into 'done' after the cancel SELECTed it."""
    iid = _seed_item(sm, "guard-done", body="b")
    job_id = sm.enqueue_extraction_job(iid)
    sm.claim_next_extraction_job()
    sm.complete_extraction_job(job_id, status="done")

    with sm._write_lock:
        changed = sm._mark_cancelled_locked(job_id, iid, error_message=None)

    assert changed is False, "the guarded cancel UPDATE matched a terminal job"
    row = sm.get_extraction_job(job_id)
    assert row is not None
    assert row["status"] == "done"
    assert row["completed_at"] is not None
    # The failed cancel must not have restored the item either.
    item = sm.get_research_item(iid)
    assert item is not None
    assert item["extraction_status"] == "running"


def test_stamp_cancel_requested_guard_only_matches_running(sm: StateManager) -> None:
    """The running-job cancel stamp must not touch a job that already finished
    between the SELECT and the UPDATE (cross-process race)."""
    iid = _seed_item(sm, "guard-stamp", body="b")
    job_id = sm.enqueue_extraction_job(iid)
    sm.claim_next_extraction_job()
    sm.complete_extraction_job(job_id, status="done")

    with sm._write_lock:
        stamped = sm._stamp_cancel_requested_locked(job_id)

    assert stamped is False
    row = sm.get_extraction_job(job_id)
    assert row is not None
    assert row["status"] == "done"
    assert row["cancel_requested_at"] is None


def test_finish_cancelled_refuses_job_that_finished(sm: StateManager) -> None:
    """finish_cancelled_extraction_job must raise (not silently overwrite) when
    the worker races the job to a real terminal outcome before it can finalize
    the cancel."""
    iid = _seed_item(sm, "guard-finish", body="b")
    job_id = sm.enqueue_extraction_job(iid)
    sm.claim_next_extraction_job()
    sm.complete_extraction_job(job_id, status="failed", error_message="boom")

    with pytest.raises(ValueError):
        sm.finish_cancelled_extraction_job(job_id, error_message="cancelled by user")
    row = sm.get_extraction_job(job_id)
    assert row is not None
    assert row["status"] == "failed"
    assert row["error_message"] == "boom"
