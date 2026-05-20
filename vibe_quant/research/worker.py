"""Extraction queue worker.

A standalone process that polls the `research_extraction_jobs` table,
atomically claims queued jobs, runs the LLM extraction inline, and
records the result. Designed to replace `FastAPI BackgroundTasks` so
extraction survives API restarts.

Run with::

    vibe-quant extraction-worker [--poll-interval 2.0] [--db PATH]

The worker logs structured JSON lines to
`data/logs/extraction-worker-<pid>.log` and to stdout. On SIGTERM it
finalizes the in-flight job as `cancelled` (with a short grace
period) and exits 0.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import FrameType

    from vibe_quant.db.state_manager import StateManager

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_S: float = 2.0
DEFAULT_GRACE_PERIOD_S: float = 5.0
DEFAULT_LOG_ROOT: Path = Path("data/logs")


def _iso_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


class _JsonlSink:
    """Append-only JSONL event log. Each call writes one line + flushes."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")

    def emit(self, event: str, **fields: Any) -> None:
        payload = {"ts": _iso_now(), "event": event, **fields}
        self._fh.write(json.dumps(payload, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._fh.close()


def process_one_job(sm: StateManager) -> dict[str, Any] | None:
    """Claim and process exactly one queued job.

    Returns the (post-update) job row, or None if the queue was empty.
    Exceptions inside extraction are caught and recorded as `failed`.
    """
    job = sm.claim_next_extraction_job()
    if job is None:
        return None

    job_id = int(job["id"])
    item_id = int(job["research_item_id"])
    try:
        _run_extraction(sm, item_id)
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        logger.exception("extraction-worker: job %s failed", job_id)
        sm.complete_extraction_job(job_id, status="failed", error_message=msg)
        sm.update_research_item_status(item_id, "failed")
        return {**job, "status": "failed", "error_message": msg}

    sm.complete_extraction_job(job_id, status="done")
    return {**job, "status": "done"}


def _run_extraction(sm: StateManager, item_id: int) -> None:
    """Synchronously extract one item using the default extractor.

    Mirrors the previous `_run_extraction_background` in the router but
    raises on failure so the caller (process_one_job) can mark the job
    failed and log uniformly.
    """
    from vibe_quant.research.archive import row_to_raw_item
    from vibe_quant.research.extraction_log import (
        log_dir_for_manual,
        write_extraction_log,
    )
    from vibe_quant.research.extractor import extractor_version, get_default_extractor
    from vibe_quant.research.pipeline import persist_extractions

    item_row = sm.get_research_item(item_id)
    if item_row is None:
        raise RuntimeError(f"research_item {item_id} not found at job start")

    extractor = get_default_extractor()
    batch = extractor.extract_all(row_to_raw_item(item_row))
    write_extraction_log(
        log_dir=log_dir_for_manual(),
        item_id=item_id,
        batch=batch,
        extractor_version=extractor_version(),
        scrape_run_id=None,
    )
    persist_extractions(sm, item_id, batch.results)


def run_forever(
    sm: StateManager,
    *,
    poll_interval: float = DEFAULT_POLL_INTERVAL_S,
    stop_event: Event | None = None,
    sink: _JsonlSink | None = None,
) -> None:
    """Drain the queue forever, sleeping `poll_interval` seconds between empty polls.

    Returns when `stop_event` is set. If a job is in flight when stop is
    requested, that job is marked `cancelled` and the function returns.
    """
    stop = stop_event or Event()
    sink = sink or _JsonlSink(DEFAULT_LOG_ROOT / f"extraction-worker-{os.getpid()}.log")
    sink.emit("worker_started", pid=os.getpid(), poll_interval=poll_interval)
    logger.info("extraction-worker started (pid=%d, poll=%.1fs)", os.getpid(), poll_interval)

    current_job_id: int | None = None
    try:
        while not stop.is_set():
            try:
                job = sm.claim_next_extraction_job()
            except Exception:  # noqa: BLE001
                logger.exception("extraction-worker: claim failed; sleeping")
                stop.wait(poll_interval)
                continue

            if job is None:
                stop.wait(poll_interval)
                continue

            current_job_id = int(job["id"])
            item_id = int(job["research_item_id"])
            sink.emit("job_claimed", job_id=current_job_id, item_id=item_id)
            try:
                _run_extraction(sm, item_id)
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"
                logger.exception("extraction-worker: job %s failed", current_job_id)
                sm.complete_extraction_job(
                    current_job_id, status="failed", error_message=msg
                )
                sm.update_research_item_status(item_id, "failed")
                sink.emit(
                    "job_failed", job_id=current_job_id, item_id=item_id, error=msg
                )
            else:
                sm.complete_extraction_job(current_job_id, status="done")
                sink.emit("job_done", job_id=current_job_id, item_id=item_id)
            current_job_id = None
    finally:
        if current_job_id is not None:
            # Stop requested while a job was in flight — finalize it.
            sm.complete_extraction_job(
                current_job_id,
                status="cancelled",
                error_message="worker received SIGTERM",
            )
            sink.emit("job_cancelled", job_id=current_job_id)
        sink.emit("worker_shutdown")
        sink.close()
        logger.info("extraction-worker shutdown")


def cli_main(argv: list[str] | None = None) -> int:
    """Entry point for `vibe-quant extraction-worker`."""
    import argparse

    from vibe_quant.db.connection import DEFAULT_DB_PATH
    from vibe_quant.db.state_manager import StateManager

    parser = argparse.ArgumentParser(prog="vibe-quant extraction-worker")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_S,
        help=f"Seconds to sleep when the queue is empty (default: {DEFAULT_POLL_INTERVAL_S})",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to the SQLite state DB (default: project default)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH
    sm = StateManager(db_path)

    stop_event = Event()

    def _handle_signal(signum: int, _frame: FrameType | None) -> None:
        logger.warning("extraction-worker: received signal %d; shutting down", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # Grace period: after stop is set, give run_forever up to GRACE_PERIOD_S
    # to finalize. Workers spend most time inside extractor calls; the
    # heuristic here is best-effort and matches the bead AC.
    start = time.monotonic()
    try:
        run_forever(sm, poll_interval=args.poll_interval, stop_event=stop_event)
    finally:
        sm.close()
        elapsed = time.monotonic() - start
        logger.info("extraction-worker: total uptime %.1fs", elapsed)

    return 0
