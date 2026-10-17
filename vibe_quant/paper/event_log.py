"""Bounded single-thread FIFO worker for paper event-log writes.

Paper trading writes structured events (``logs/events/{run_id}.jsonl``) from the
trading loop. A blocking or hung disk write must never stall the loop, so every
write is handed to one worker thread (a :class:`~vibe_quant.paper.persistence.StoreWorker`)
and the caller only holds a future. The queue is bounded: when ``max_pending``
jobs are outstanding (submitted but not finished, including the running one), a
new submit is *dropped* rather than queued, so a stuck writer cannot make memory
grow without bound while the loop keeps producing events.

``submit`` never blocks and never raises for a full queue: it returns an
already-finished future whose exception is :class:`EventLogFullError` and counts
the drop in :meth:`health`. A single WARNING per full streak (one warning while
the queue stays saturated, none until a submit is accepted again) keeps the log
from being flooded. Failures inside the writer are counted and logged by class
name only (the exception's message may carry payload data) and re-raised into
the caller's future.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vibe_quant.paper.persistence import StoreWorker

if TYPE_CHECKING:
    from vibe_quant.logging.events import Event
    from vibe_quant.logging.writer import EventWriter

logger = logging.getLogger(__name__)

# Maximum number of outstanding (submitted, not finished) write jobs.
EVENT_QUEUE_MAX: int = 10_000


class EventLogFullError(RuntimeError):
    """Returned (on the future) when a submit is dropped because the queue is full."""


class EventWriteTimeoutError(TimeoutError):
    """Raised when a paper event write is not confirmed within the timeout."""


@dataclass(frozen=True)
class EventLogHealth:
    """Lock-protected snapshot of the worker's counters.

    Attributes:
        written: Jobs whose ``writer.write`` returned successfully.
        failed: Jobs whose ``writer.write`` raised.
        dropped: Submits refused because the queue was full.
        pending: Submitted-but-unfinished jobs, including the running one.
        last_error_class: Class name of the most recent write failure, or ``None``.
    """

    written: int
    failed: int
    dropped: int
    pending: int
    last_error_class: str | None


class EventLogWorker:
    """One thread that owns paper event-log writes, with a bounded FIFO queue.

    Attributes:
        worker: The underlying :class:`StoreWorker`; ``_shutdown`` hands it to
            ``_close_store`` so the log file is closed after the last queued write.
    """

    def __init__(
        self,
        thread_name: str = "vq-paper-events",
        max_pending: int = EVENT_QUEUE_MAX,
    ) -> None:
        self._max_pending = max_pending
        self._lock = threading.Lock()
        self._pending = 0
        self._written = 0
        self._failed = 0
        self._dropped = 0
        self._last_error_class: str | None = None
        self._full_streak = False
        self.worker = StoreWorker(thread_name)

    def submit(self, writer: EventWriter, event: Event) -> Future[None]:
        """Queue ``writer.write(event)`` behind every earlier job (FIFO).

        Never blocks and never raises for a full queue: when ``max_pending`` jobs
        are outstanding a completed future carrying :class:`EventLogFullError` is
        returned, ``dropped`` is incremented and one WARNING is emitted per full
        streak. After the worker's executor is stopped this raises the executor's
        ``RuntimeError`` synchronously (and does not leak the pending count).
        """
        with self._lock:
            if self._pending >= self._max_pending:
                self._dropped += 1
                if not self._full_streak:
                    self._full_streak = True
                    logger.warning(
                        "paper event log full (%d pending); dropping events", self._pending
                    )
                full: Future[None] = Future()
                full.set_exception(
                    EventLogFullError(
                        f"paper event log full ({self._pending} pending); event dropped"
                    )
                )
                return full
            self._pending += 1
        try:
            future: Future[None] = self.worker.submit(self._write_job, writer, event)
        except RuntimeError:
            with self._lock:
                self._pending -= 1
            raise
        # The full streak ends only on an *accepted* submit (chief nit, rev-e4-evlog-1):
        # a submit refused by a stopped executor accepts nothing, so it must leave the
        # streak flag untouched.
        with self._lock:
            self._full_streak = False
        return future

    def barrier(self) -> Future[None]:
        """Queue a no-op behind every earlier job; NOT counted against the bound.

        Returns a future that completes once every job submitted before it has
        finished, letting a caller wait for the current queue to drain.
        """
        return self.worker.submit(_noop)

    def health(self) -> EventLogHealth:
        """Return a lock-protected snapshot of the counters."""
        with self._lock:
            return EventLogHealth(
                written=self._written,
                failed=self._failed,
                dropped=self._dropped,
                pending=self._pending,
                last_error_class=self._last_error_class,
            )

    def _write_job(self, writer: EventWriter, event: Event) -> None:
        """Worker-side job: write one event, counting success/failure.

        The pending decrement happens here, under the lock, before the future
        resolves (see chief ruling D-SF2), so ``pending`` is already up to date
        when ``future.result()`` returns.
        """
        try:
            writer.write(event)
        except Exception as exc:
            error_class = type(exc).__name__
            with self._lock:
                self._failed += 1
                self._last_error_class = error_class
            logger.error("paper event write failed (%s)", error_class)
            raise
        else:
            with self._lock:
                self._written += 1
        finally:
            with self._lock:
                self._pending -= 1


def _noop() -> None:
    """Barrier job: does nothing, only preserves FIFO ordering."""
