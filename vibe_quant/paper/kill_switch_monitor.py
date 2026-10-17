"""Kill-switch reader thread for the paper node (bead vibe-quant-dt065, task T2a).

The paper event loop never reads the system kill switch itself: a locked or
hung state DB would freeze trading and, worse, the fail-closed check with it.
One daemon thread owns its own state-DB handle, reads the switch every
``KILL_SWITCH_POLL_S`` and publishes a :class:`KillSwitchSnapshot`. The node's
watcher, ``resume()`` and startup only ever look at the snapshot (a lock-guarded
copy, no DB access), so a hung read shows up as a growing snapshot age and the
node halts ERROR when it reaches ``KILL_SWITCH_STALE_LIMIT_S``.

The reader thread owns its handle: it is opened on that thread and closed on
that thread after the loop exits and any in-flight read has returned. ``stop()``
never closes it (the state manager's ``close()`` waits for the lock a hung read
holds), it only signals and waits a bounded time.

Stdlib only: nothing here imports the node, the guard or the state manager.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

logger = logging.getLogger(__name__)

#: Seconds between kill-switch reads on the reader thread.
KILL_SWITCH_POLL_S = 1.0
#: Halt ERROR when the last clean read is this old (``>=``). A recovering 5 s
#: read peaks at ~7 s, below the limit, so one slow read halts nothing.
KILL_SWITCH_STALE_LIMIT_S = 8.0
#: How often the node's watcher looks at the snapshot.
KILL_SWITCH_WATCH_TICK_S = 0.5
#: ``resume()`` needs a clean read at most this old (``>`` refuses).
KILL_SWITCH_RESUME_FRESH_S = 2.0
#: Startup waits this long for the first clean read.
KILL_SWITCH_STARTUP_TIMEOUT_S = 10.0

_WAIT_FIRST_READ_STEP_S = 0.1


class StateReader(Protocol):
    """What the reader thread needs from the state DB handle."""

    def get_system_state(self) -> Mapping[str, object]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class KillSwitchSnapshot:
    """The reader thread's latest view; ``engaged``/``reason`` are the last CLEAN read."""

    engaged: bool
    reason: str | None
    last_ok_mono: float | None  # clock() when the last clean read COMPLETED
    error_class: str | None  # latest failed read's class; None after a clean read
    consecutive_failures: int

    def age(self, now: float) -> float:
        """Seconds since the last clean read completed; ``inf`` before the first."""
        if self.last_ok_mono is None:
            return math.inf
        return now - self.last_ok_mono


_EMPTY = KillSwitchSnapshot(False, None, None, None, 0)


class KillSwitchMonitor:
    """Polls the system kill switch on a daemon thread; readers use ``snapshot()``."""

    def __init__(
        self,
        open_reader: Callable[[], StateReader],
        *,
        poll_s: float = KILL_SWITCH_POLL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._open_reader = open_reader
        self._poll_s = poll_s
        self._clock = clock
        self._lock = threading.Lock()
        self._snapshot = _EMPTY
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        #: Single-use (K2): set by the first ``stop`` of a started monitor.
        self._stopped = False
        #: K1: ``stop`` joins at most once, however many times the node calls it.
        self._join_done = False

    def start(self) -> None:
        """Start the reader thread (idempotent while running; a stopped monitor is spent).

        Raises:
            RuntimeError: If this monitor was already stopped. Its snapshot is stale by
                definition, so a restart would let the node pass startup on it: build a
                new monitor instead.
        """
        if self._stopped:
            raise RuntimeError("KillSwitchMonitor cannot be restarted; build a new one")
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="vq-kill-switch-reader", daemon=True)
        self._thread.start()

    def stop(self, join_timeout_s: float = 1.0) -> None:
        """Signal the thread and wait at most ``join_timeout_s`` ONCE; never closes the reader.

        The node stops the monitor twice (``_run_loop``'s finally and ``_shutdown``): only
        the first call joins, the second returns at once. A read hung inside the DB keeps
        the thread alive past the timeout; the thread closes its own handle once that read
        returns.
        """
        self._stop.set()
        thread = self._thread
        if thread is None:  # never started: nothing to signal and nothing to join
            return
        self._stopped = True
        if self._join_done:
            return
        self._join_done = True
        if thread is not threading.current_thread():
            thread.join(join_timeout_s)

    def snapshot(self) -> KillSwitchSnapshot:
        """A copy of the latest view. Never touches the DB."""
        with self._lock:
            return self._snapshot

    async def wait_first_read(self, timeout_s: float) -> KillSwitchSnapshot:
        """Wait (without blocking the loop) for the first clean read or the timeout."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while True:
            snap = self.snapshot()
            if snap.last_ok_mono is not None or loop.time() >= deadline:
                return snap
            await asyncio.sleep(min(_WAIT_FIRST_READ_STEP_S, max(deadline - loop.time(), 0.0)))

    def poll_once(self, reader: StateReader) -> None:
        """One read cycle: update the snapshot from a clean read or record the failure.

        Errors are logged by class only: the text can carry SQL/paths and the
        active exception's ``__context__`` chain may hold a fetch error (signed
        URL / api key), so no ``exc_info`` either.
        """
        try:
            state = reader.get_system_state()
            engaged = bool(state.get("kill_switch"))
            raw_reason = state.get("reason")
            reason = None if raw_reason is None else str(raw_reason)
        except Exception as exc:
            self._record_failure(type(exc).__name__)
            return
        now = self._clock()
        with self._lock:
            self._snapshot = KillSwitchSnapshot(engaged, reason, now, None, 0)

    def _record_failure(self, cls: str) -> None:
        with self._lock:
            snap = self._snapshot
            self._snapshot = replace(
                snap, error_class=cls, consecutive_failures=snap.consecutive_failures + 1
            )
            failures = self._snapshot.consecutive_failures
        logger.error(
            "could not read system kill switch (%s); %d consecutive failure(s)", cls, failures
        )

    def _run(self) -> None:
        reader: StateReader | None = None
        try:
            while not self._stop.is_set():
                if reader is None:
                    try:
                        reader = self._open_reader()
                    except Exception as exc:
                        self._record_failure(type(exc).__name__)
                if reader is not None:
                    self.poll_once(reader)
                self._stop.wait(self._poll_s)
        finally:
            # The reader owns its handle: closed here, on this thread, only after
            # the loop exited and any in-flight read returned (B2: no cross-thread close).
            if reader is not None:
                try:
                    reader.close()
                except Exception as exc:
                    logger.error("kill-switch reader close failed (%s)", type(exc).__name__)
