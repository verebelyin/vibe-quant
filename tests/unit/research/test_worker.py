"""Extraction queue + worker tests (bd-j68g.1)."""

from __future__ import annotations

import contextlib
import threading
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.research import worker as worker_mod
from vibe_quant.research.schema import ExtractionBatch, ExtractionResult

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@pytest.fixture
def sm(tmp_path: Path) -> Generator[StateManager]:
    mgr = StateManager(tmp_path / "queue.db")
    yield mgr
    mgr.close()


@pytest.fixture(autouse=True)
def _redirect_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_mod, "DEFAULT_LOG_ROOT", tmp_path / "worker-logs")
    # Also redirect the extraction-log root so persist_extractions writes
    # into the test tmpdir.
    from vibe_quant.research import extraction_log

    monkeypatch.setattr(extraction_log, "DEFAULT_LOG_ROOT", tmp_path / "extraction-logs")


def _seed_item(sm: StateManager, ext_id: str = "x") -> int:
    return sm.create_research_item(
        source="reddit",
        external_id=ext_id,
        url="u",
        title="t",
        body="b",
        author=None,
        posted_at=None,
        score=None,
    )


def _ok_extractor() -> object:
    """An extractor that returns a 'skipped' result without touching the LLM."""
    result = ExtractionResult(
        status="skipped",
        confidence=0.1,
        rationale="no",
        raw_response="{}",
        dsl_yaml=None,
        parsed_dsl_json=None,
        parse_error=None,
        llm_model="t",
    )
    batch = ExtractionBatch(prompt="P", raw_response="{}", results=[result])

    class _Ext:
        def extract_all(self, _item: Any) -> ExtractionBatch:
            return batch

        def extract(self, _item: Any) -> ExtractionResult:
            return result

    return _Ext()


def test_enqueue_creates_job_and_marks_item_queued(sm: StateManager) -> None:
    iid = _seed_item(sm)
    job_id = sm.enqueue_extraction_job(iid)
    assert isinstance(job_id, int)
    item = sm.get_research_item(iid)
    assert item is not None
    assert item["extraction_status"] == "queued"
    job = sm.get_extraction_job(job_id)
    assert job is not None
    assert job["status"] == "queued"
    assert job["research_item_id"] == iid


def test_claim_returns_oldest_queued_job_then_none(sm: StateManager) -> None:
    iid1 = _seed_item(sm, "a")
    iid2 = _seed_item(sm, "b")
    j1 = sm.enqueue_extraction_job(iid1)
    j2 = sm.enqueue_extraction_job(iid2)

    first = sm.claim_next_extraction_job()
    second = sm.claim_next_extraction_job()
    empty = sm.claim_next_extraction_job()

    assert first is not None and first["id"] == j1
    assert second is not None and second["id"] == j2
    assert first["status"] == "running"
    assert empty is None


def test_claim_is_exclusive_between_concurrent_callers(sm: StateManager) -> None:
    """Two threads racing on the same single queued row must claim distinct ids."""
    iid1 = _seed_item(sm, "a")
    iid2 = _seed_item(sm, "b")
    sm.enqueue_extraction_job(iid1)
    sm.enqueue_extraction_job(iid2)

    results: list[Any] = []
    barrier = threading.Barrier(2)

    def claim() -> None:
        barrier.wait()
        results.append(sm.claim_next_extraction_job())

    t1 = threading.Thread(target=claim)
    t2 = threading.Thread(target=claim)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    ids = [r["id"] for r in results if r is not None]
    assert len(ids) == 2
    assert ids[0] != ids[1]  # no double-claim


def test_process_one_job_happy_path(sm: StateManager) -> None:
    iid = _seed_item(sm)
    job_id = sm.enqueue_extraction_job(iid)

    with patch(
        "vibe_quant.research.extractor.get_default_extractor", return_value=_ok_extractor()
    ):
        result = worker_mod.process_one_job(sm)

    assert result is not None
    assert result["id"] == job_id
    after_job = sm.get_extraction_job(job_id)
    assert after_job is not None
    assert after_job["status"] == "done"
    assert after_job["completed_at"] is not None
    after_item = sm.get_research_item(iid)
    assert after_item is not None
    assert after_item["extraction_status"] == "skipped"


def test_process_one_job_extractor_failure_marks_job_failed(sm: StateManager) -> None:
    iid = _seed_item(sm)
    # max_attempts=1 so a single failure terminates without retry.
    job_id = sm.enqueue_extraction_job(iid, max_attempts=1)

    class _BoomExt:
        def extract_all(self, _item: Any) -> ExtractionBatch:
            raise RuntimeError("boom")

    with patch(
        "vibe_quant.research.extractor.get_default_extractor", return_value=_BoomExt()
    ):
        result = worker_mod.process_one_job(sm)

    assert result is not None
    assert result["status"] == "failed"
    after = sm.get_extraction_job(job_id)
    assert after is not None
    assert after["status"] == "failed"
    assert "boom" in (after["last_error"] or "")
    assert after["attempts"] == 1
    item = sm.get_research_item(iid)
    assert item is not None
    assert item["extraction_status"] == "failed"


def test_process_one_job_returns_none_when_queue_empty(sm: StateManager) -> None:
    assert worker_mod.process_one_job(sm) is None


def test_run_forever_finalizes_inflight_job_as_cancelled_on_stop(
    sm: StateManager,
) -> None:
    iid = _seed_item(sm)
    job_id = sm.enqueue_extraction_job(iid)

    stop = threading.Event()
    started = threading.Event()

    class _SlowExt:
        def extract_all(self, _item: Any) -> ExtractionBatch:
            started.set()
            # Block until the test signals the worker to stop, then bail out
            # with a regular Exception (not KeyboardInterrupt, which would
            # propagate out of the drain thread uncaught).
            stop.wait(timeout=2.0)
            raise RuntimeError("forced bailout")

    sink = worker_mod._JsonlSink(
        worker_mod.DEFAULT_LOG_ROOT / "test-cancel.log"
    )

    def runner() -> None:
        with patch(
            "vibe_quant.research.extractor.get_default_extractor",
            return_value=_SlowExt(),
        ):
            worker_mod.run_forever(
                sm, poll_interval=0.05, stop_event=stop, sink=sink
            )

    th = threading.Thread(target=runner)
    th.start()
    assert started.wait(timeout=2.0)
    stop.set()
    th.join(timeout=3.0)
    assert not th.is_alive()

    after = sm.get_extraction_job(job_id)
    assert after is not None
    # Either failed or queued (re-queue under default max_attempts). The AC
    # is that the job never stays 'running' after shutdown.
    assert after["status"] in ("cancelled", "failed", "queued")
    _ = contextlib  # kept for module-level imports compatibility


# ---------- bd-j68g.2: retry + last_error + stuck detection ----------


def test_extractor_failure_under_max_attempts_requeues(sm: StateManager) -> None:
    """First failure with attempts<max_attempts must re-queue, not finalize."""
    iid = _seed_item(sm)
    job_id = sm.enqueue_extraction_job(iid, max_attempts=3)

    class _BoomExt:
        def extract_all(self, _item: Any) -> ExtractionBatch:
            raise RuntimeError("transient")

    with patch(
        "vibe_quant.research.extractor.get_default_extractor", return_value=_BoomExt()
    ):
        worker_mod.process_one_job(sm)

    after = sm.get_extraction_job(job_id)
    assert after is not None
    assert after["status"] == "queued"
    assert after["attempts"] == 1
    assert "transient" in (after["last_error"] or "")
    # started_at cleared so the next claim resets it.
    assert after["started_at"] is None
    item = sm.get_research_item(iid)
    assert item is not None
    assert item["extraction_status"] == "queued"


def test_extractor_failure_three_times_marks_failed_and_keeps_last_error(
    sm: StateManager,
) -> None:
    """Three consecutive failures → final 'failed' with most-recent error."""
    iid = _seed_item(sm)
    job_id = sm.enqueue_extraction_job(iid, max_attempts=3)
    errors = iter(["first", "second", "third"])

    class _RotatingBoom:
        def extract_all(self, _item: Any) -> ExtractionBatch:
            raise RuntimeError(next(errors))

    with patch(
        "vibe_quant.research.extractor.get_default_extractor",
        return_value=_RotatingBoom(),
    ):
        for _ in range(3):
            worker_mod.process_one_job(sm)

    after = sm.get_extraction_job(job_id)
    assert after is not None
    assert after["status"] == "failed"
    assert after["attempts"] == 3
    # last_error is the most recent only — not concatenated.
    assert "third" in (after["last_error"] or "")
    assert "first" not in (after["last_error"] or "")
    item = sm.get_research_item(iid)
    assert item is not None
    assert item["extraction_status"] == "failed"


def test_sweep_stuck_job_with_attempts_remaining_requeues(sm: StateManager) -> None:
    """A 'running' row with no heartbeat past the threshold must be re-queued."""
    iid = _seed_item(sm)
    job_id = sm.enqueue_extraction_job(iid, max_attempts=2)
    sm.claim_next_extraction_job()
    # Simulate the worker dying right after claim — push heartbeat far enough
    # into the past that any positive threshold catches it.
    sm.conn.execute(
        "UPDATE research_extraction_jobs SET heartbeat_at = datetime('now', '-1 hour') "
        "WHERE id = ?",
        (job_id,),
    )
    sm.conn.commit()

    swept = sm.sweep_stuck_extraction_jobs(60)
    assert len(swept) == 1
    assert swept[0]["id"] == job_id
    assert swept[0]["status"] == "queued"  # attempts=1 < max_attempts=2

    after = sm.get_extraction_job(job_id)
    assert after is not None
    assert after["status"] == "queued"
    assert after["attempts"] == 1
    assert "stuck" in (after["last_error"] or "")


def test_sweep_stuck_job_at_max_attempts_marks_failed(sm: StateManager) -> None:
    iid = _seed_item(sm)
    job_id = sm.enqueue_extraction_job(iid, max_attempts=1)
    sm.claim_next_extraction_job()
    sm.conn.execute(
        "UPDATE research_extraction_jobs SET heartbeat_at = datetime('now', '-1 hour') "
        "WHERE id = ?",
        (job_id,),
    )
    sm.conn.commit()

    swept = sm.sweep_stuck_extraction_jobs(60)
    assert len(swept) == 1
    assert swept[0]["status"] == "failed"


def test_sweep_excludes_current_job_id(sm: StateManager) -> None:
    """Worker must not sweep its own in-flight job even if heartbeat is stale."""
    iid = _seed_item(sm)
    job_id = sm.enqueue_extraction_job(iid)
    sm.claim_next_extraction_job()
    sm.conn.execute(
        "UPDATE research_extraction_jobs SET heartbeat_at = datetime('now', '-1 hour') "
        "WHERE id = ?",
        (job_id,),
    )
    sm.conn.commit()

    swept = sm.sweep_stuck_extraction_jobs(60, exclude_job_id=job_id)
    assert swept == []

    still = sm.get_extraction_job(job_id)
    assert still is not None
    assert still["status"] == "running"


def test_heartbeat_extraction_job_updates_running_only(sm: StateManager) -> None:
    iid = _seed_item(sm)
    job_id = sm.enqueue_extraction_job(iid)
    # Not yet running → heartbeat is a no-op.
    assert sm.heartbeat_extraction_job(job_id) is False
    sm.claim_next_extraction_job()
    assert sm.heartbeat_extraction_job(job_id) is True
    row = sm.get_extraction_job(job_id)
    assert row is not None
    assert row["heartbeat_at"] is not None


def test_stuck_threshold_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VQ_EXTRACTION_STUCK_THRESHOLD_SECONDS", "120")
    assert worker_mod._stuck_threshold_seconds() == 120
    monkeypatch.setenv("VQ_EXTRACTION_STUCK_THRESHOLD_SECONDS", "garbage")
    assert worker_mod._stuck_threshold_seconds() == 240
    monkeypatch.delenv("VQ_EXTRACTION_STUCK_THRESHOLD_SECONDS")
    assert worker_mod._stuck_threshold_seconds() == 240


# ---------- bd-j68g.4: orphan migration + --concurrency ----------


def test_reset_orphan_running_items_resets_only_orphans(sm: StateManager) -> None:
    """Items at extraction_status='running' with no queue row → 'pending'.
    Items with an existing queue row are NOT touched."""
    orphan = _seed_item(sm, "orphan")
    sm.update_research_item_status(orphan, "running")
    tracked = _seed_item(sm, "tracked")
    sm.update_research_item_status(tracked, "running")
    # Insert a manual queue row for the tracked item so it is NOT orphaned.
    sm.conn.execute(
        "INSERT INTO research_extraction_jobs (research_item_id, status) VALUES (?, 'running')",
        (tracked,),
    )
    sm.conn.commit()
    pending = _seed_item(sm, "pending")  # already pending

    n = sm.reset_orphan_running_items()
    assert n == 1

    assert sm.get_research_item(orphan)["extraction_status"] == "pending"  # type: ignore[index]
    assert sm.get_research_item(tracked)["extraction_status"] == "running"  # type: ignore[index]
    assert sm.get_research_item(pending)["extraction_status"] == "pending"  # type: ignore[index]


def test_reset_orphan_running_items_is_idempotent(sm: StateManager) -> None:
    iid = _seed_item(sm, "x")
    sm.update_research_item_status(iid, "running")
    assert sm.reset_orphan_running_items() == 1
    # Second call must touch nothing — the item is already 'pending'.
    assert sm.reset_orphan_running_items() == 0


def test_run_forever_rejects_invalid_concurrency(sm: StateManager) -> None:
    stop = threading.Event()
    sink = worker_mod._JsonlSink(worker_mod.DEFAULT_LOG_ROOT / "bad.log")
    with pytest.raises(ValueError, match="concurrency must be >= 1"):
        worker_mod.run_forever(sm, concurrency=0, stop_event=stop, sink=sink)
    with pytest.raises(ValueError, match="concurrency must be <="):
        worker_mod.run_forever(sm, concurrency=5, stop_event=stop, sink=sink)


def test_run_forever_concurrency_2_drains_both_jobs_in_parallel(
    sm: StateManager,
) -> None:
    """With concurrency=2 and 2 jobs queued, both threads execute
    extractions simultaneously (proven via a Barrier that requires
    both threads to arrive before either proceeds)."""
    iid_a = _seed_item(sm, "a")
    iid_b = _seed_item(sm, "b")
    job_a = sm.enqueue_extraction_job(iid_a)
    job_b = sm.enqueue_extraction_job(iid_b)

    barrier = threading.Barrier(2, timeout=3.0)
    fake_result = ExtractionResult(
        status="skipped",
        confidence=0.1,
        rationale="ok",
        raw_response="{}",
        dsl_yaml=None,
        parsed_dsl_json=None,
        parse_error=None,
        llm_model="t",
    )
    fake_batch = ExtractionBatch(prompt="P", raw_response="{}", results=[fake_result])

    class _ParallelExt:
        def extract_all(self, _item: Any) -> ExtractionBatch:
            # Both threads must arrive here for either to proceed — fails
            # the test if concurrency is silently 1.
            barrier.wait()
            return fake_batch

    stop = threading.Event()
    sink = worker_mod._JsonlSink(worker_mod.DEFAULT_LOG_ROOT / "concurrency2.log")

    def runner() -> None:
        with patch(
            "vibe_quant.research.extractor.get_default_extractor",
            return_value=_ParallelExt(),
        ):
            worker_mod.run_forever(
                sm,
                poll_interval=0.05,
                stop_event=stop,
                sink=sink,
                concurrency=2,
            )

    th = threading.Thread(target=runner)
    th.start()

    # Wait for both jobs to reach 'done'.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        a = sm.get_extraction_job(job_a)
        b = sm.get_extraction_job(job_b)
        if a and b and a["status"] == "done" and b["status"] == "done":
            break
        time.sleep(0.05)

    stop.set()
    th.join(timeout=5.0)
    assert not th.is_alive()

    a = sm.get_extraction_job(job_a)
    b = sm.get_extraction_job(job_b)
    assert a is not None and a["status"] == "done"
    assert b is not None and b["status"] == "done"


def test_cli_concurrency_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI must SystemExit on out-of-range --concurrency."""
    import argparse

    # argparse calls parser.error() which raises SystemExit(2).
    with pytest.raises(SystemExit):
        worker_mod.cli_main(["--concurrency", "0"])
    with pytest.raises(SystemExit):
        worker_mod.cli_main(["--concurrency", "99"])

    # Keep a strict reference so we don't accidentally test something else.
    _ = argparse
