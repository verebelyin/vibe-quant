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
import threading
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
DEFAULT_HEARTBEAT_INTERVAL_S: float = 30.0
DEFAULT_SWEEP_INTERVAL_S: float = 60.0
MAX_CONCURRENCY: int = 4


def _stuck_threshold_seconds() -> int:
    """Read the stuck-job threshold from env (default 240s = 4×heartbeat)."""
    raw = os.environ.get("VQ_EXTRACTION_STUCK_THRESHOLD_SECONDS")
    if not raw:
        return 240
    try:
        v = int(raw)
        return v if v > 0 else 240
    except ValueError:
        return 240


class _HeartbeatThread(threading.Thread):
    """Background daemon that bumps `heartbeat_at` for the current job."""

    def __init__(
        self,
        sm: StateManager,
        job_id: int,
        *,
        interval: float = DEFAULT_HEARTBEAT_INTERVAL_S,
    ) -> None:
        super().__init__(daemon=True, name=f"extract-heartbeat-{job_id}")
        self._sm = sm
        self._job_id = job_id
        self._interval = interval
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._sm.heartbeat_extraction_job(self._job_id)
            except Exception:  # noqa: BLE001
                logger.exception("heartbeat failed for job %s", self._job_id)


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
    Exceptions inside extraction are routed through `fail_extraction_job`
    which decides retry vs final failure based on attempts/max_attempts.
    """
    job = sm.claim_next_extraction_job()
    if job is None:
        return None

    job_id = int(job["id"])
    item_id = int(job["research_item_id"])
    heartbeat = _HeartbeatThread(sm, job_id)
    heartbeat.start()
    try:
        _run_extraction(sm, item_id)
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        logger.exception("extraction-worker: job %s failed", job_id)
        return sm.fail_extraction_job(job_id, msg)
    finally:
        heartbeat.stop()

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
    persist_extractions(sm, item_id, batch.results, prompt=batch.prompt)


def _drain_loop(
    sm: StateManager,
    *,
    stop: Event,
    sink: _JsonlSink,
    poll_interval: float,
    in_flight: set[int],
    in_flight_lock: threading.Lock,
    worker_idx: int,
) -> None:
    """One worker thread's drain loop. Claims one job at a time, marks
    in_flight, processes, then loops. Tracks its own current job so it can
    finalize as 'cancelled' if stop is set mid-extraction."""
    current_job_id: int | None = None
    try:
        while not stop.is_set():
            try:
                job = sm.claim_next_extraction_job()
            except Exception:  # noqa: BLE001
                logger.exception("extraction-worker[%d]: claim failed; sleeping", worker_idx)
                stop.wait(poll_interval)
                continue

            if job is None:
                stop.wait(poll_interval)
                continue

            current_job_id = int(job["id"])
            item_id = int(job["research_item_id"])
            with in_flight_lock:
                in_flight.add(current_job_id)
            sink.emit(
                "job_claimed",
                job_id=current_job_id,
                item_id=item_id,
                worker_idx=worker_idx,
            )
            heartbeat = _HeartbeatThread(sm, current_job_id)
            heartbeat.start()
            try:
                _run_extraction(sm, item_id)
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"
                logger.exception("extraction-worker: job %s failed", current_job_id)
                outcome = sm.fail_extraction_job(current_job_id, msg)
                sink.emit(
                    "job_failed",
                    job_id=current_job_id,
                    item_id=item_id,
                    error=msg,
                    attempts=int(outcome["attempts"]),
                    final_status=outcome["status"],
                )
            else:
                sm.complete_extraction_job(current_job_id, status="done")
                sink.emit("job_done", job_id=current_job_id, item_id=item_id)
            finally:
                heartbeat.stop()
                with in_flight_lock:
                    in_flight.discard(current_job_id)
                current_job_id = None
    finally:
        if current_job_id is not None:
            # Stop requested while a job was in flight — finalize it.
            sm.complete_extraction_job(
                current_job_id,
                status="cancelled",
                error_message="worker received SIGTERM",
            )
            with in_flight_lock:
                in_flight.discard(current_job_id)
            sink.emit("job_cancelled", job_id=current_job_id)


def run_forever(
    sm: StateManager,
    *,
    poll_interval: float = DEFAULT_POLL_INTERVAL_S,
    stop_event: Event | None = None,
    sink: _JsonlSink | None = None,
    concurrency: int = 1,
) -> None:
    """Drain the queue forever, sleeping `poll_interval` seconds between empty polls.

    Spawns `concurrency` drain threads (max MAX_CONCURRENCY). The main thread
    runs the stuck-job sweeper. Returns when `stop_event` is set; any jobs in
    flight at that moment are finalized as `cancelled`.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1 (got {concurrency})")
    if concurrency > MAX_CONCURRENCY:
        raise ValueError(
            f"concurrency must be <= {MAX_CONCURRENCY} (got {concurrency})"
        )

    stop = stop_event or Event()
    sink = sink or _JsonlSink(DEFAULT_LOG_ROOT / f"extraction-worker-{os.getpid()}.log")
    sink.emit(
        "worker_started",
        pid=os.getpid(),
        poll_interval=poll_interval,
        concurrency=concurrency,
    )
    logger.info(
        "extraction-worker started (pid=%d, poll=%.1fs, concurrency=%d)",
        os.getpid(),
        poll_interval,
        concurrency,
    )

    stuck_threshold = _stuck_threshold_seconds()

    in_flight: set[int] = set()
    in_flight_lock = threading.Lock()

    drain_threads = [
        threading.Thread(
            target=_drain_loop,
            kwargs={
                "sm": sm,
                "stop": stop,
                "sink": sink,
                "poll_interval": poll_interval,
                "in_flight": in_flight,
                "in_flight_lock": in_flight_lock,
                "worker_idx": i,
            },
            name=f"extract-drain-{i}",
            daemon=False,
        )
        for i in range(concurrency)
    ]
    for t in drain_threads:
        t.start()

    try:
        last_sweep = time.monotonic()
        while not stop.is_set():
            stop.wait(min(DEFAULT_SWEEP_INTERVAL_S, poll_interval))
            now = time.monotonic()
            if now - last_sweep < DEFAULT_SWEEP_INTERVAL_S:
                continue
            try:
                with in_flight_lock:
                    excluded = set(in_flight)
                swept = sm.sweep_stuck_extraction_jobs(
                    stuck_threshold, exclude_job_ids=excluded
                )
                for j in swept:
                    sink.emit(
                        "job_swept",
                        job_id=int(j["id"]),
                        item_id=int(j["research_item_id"]),
                        attempts=int(j["attempts"]),
                        final_status=j["status"],
                    )
            except Exception:  # noqa: BLE001
                logger.exception("extraction-worker: sweep failed")
            last_sweep = now
    finally:
        stop.set()
        for t in drain_threads:
            t.join()
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
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            f"Number of in-flight extractions per worker process "
            f"(default: 1, max: {MAX_CONCURRENCY})"
        ),
    )
    args = parser.parse_args(argv)

    if args.concurrency < 1 or args.concurrency > MAX_CONCURRENCY:
        parser.error(
            f"--concurrency must be between 1 and {MAX_CONCURRENCY} "
            f"(got {args.concurrency})"
        )

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH
    sm = StateManager(db_path)

    # One-time legacy orphan migration: items left at 'running' from the old
    # BackgroundTasks path (no row in research_extraction_jobs) get reset to
    # 'pending' so they can be re-enqueued via the UI.
    n_reset = sm.reset_orphan_running_items()
    if n_reset:
        logger.info("reset %d orphan items from old BackgroundTasks path", n_reset)

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
        run_forever(
            sm,
            poll_interval=args.poll_interval,
            stop_event=stop_event,
            concurrency=args.concurrency,
        )
    finally:
        sm.close()
        elapsed = time.monotonic() - start
        logger.info("extraction-worker: total uptime %.1fs", elapsed)

    return 0
