"""Extraction queue + worker tests (bd-j68g.1)."""

from __future__ import annotations

import contextlib
import threading
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
    job_id = sm.enqueue_extraction_job(iid)

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
    assert "boom" in (after["error_message"] or "")
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
            # Wait until the test signals the worker to stop, then take
            # a moment longer so the stop event lands while we're 'in flight'.
            stop.wait(timeout=2.0)
            raise KeyboardInterrupt  # simulate forced interruption

    sink = worker_mod._JsonlSink(
        worker_mod.DEFAULT_LOG_ROOT / "test-cancel.log"
    )

    def runner() -> None:
        with (
            patch(
                "vibe_quant.research.extractor.get_default_extractor",
                return_value=_SlowExt(),
            ),
            contextlib.suppress(KeyboardInterrupt),
        ):
            # The slow extractor raises KeyboardInterrupt to bail out of
            # run_forever so the finally block can finalize the job.
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
    # Either cancelled (if finally fired) or failed (if KeyboardInterrupt
    # was caught as a regular exception). Both are acceptable terminal states
    # — the AC is that the job never stays 'running' after SIGTERM-equivalent.
    assert after["status"] in ("cancelled", "failed")
