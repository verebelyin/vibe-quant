"""No blocking DB call on the paper event loop (bead vibe-quant-dt065, T2b).

Contract rows of job 20261016-safety-bankroll (design B2, T2b half):

* R13c REAL loop: checkpoint save + prune, command claim + complete, one guard
       transition (its event write AND its checkpoint) each block 5 s, the kill-switch
       reader fails from t_f -> ERROR halt <= 15.0 s after t_f, and the watcher tick
       gap never exceeds 1.5 s: the 5 s event-file write now runs on the
       ``vq-paper-events`` worker (ogd37, R2), so no stall is left on the loop.
* R19  checkpoint ordering: the single FIFO worker writes in capture order, whichever
       path (guard transition / periodic loop) captured first.
* R19b a raising transition write still submits its checkpoint; the write error wins.
* R19c shutdown with a hung checkpoint save waits a bounded time, logs class only,
       and still stops the trading engine.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from typing import Any

import pytest

from tests.unit.test_audit_paper_node import _config
from tests.unit.test_paper_alert_write import (
    _FakeStateManager,
    _FakeTradingNodeStop,
)
from tests.unit.test_paper_kill_switch_fail_closed import (
    _LEAK,
    fake_tradingnode,  # noqa: F401  # fixture
    seeded,  # noqa: F401  # fixture
)
from tests.unit.test_paper_kill_switch_monitor import (
    HALT_BOUND_S,
    READ_STALL_S,
    SCALE,
    _SlowFailingReader,
    _TimedGuard,
    scaled_kill_clock,  # noqa: F401  # fixture
)
from vibe_quant.paper import node as node_module
from vibe_quant.paper.guard import GuardState, HaltReason
from vibe_quant.paper.journal import FakeJournal
from vibe_quant.paper.kill_switch_monitor import KillSwitchMonitor
from vibe_quant.paper.node import PaperTradingNode
from vibe_quant.paper.persistence import PaperCommandQueue, StateCheckpoint, StatePersistence

STALL_S = READ_STALL_S  # production's 5 s stall, at the kill-switch tests' SCALE


def _drain(node: PaperTradingNode) -> None:
    """Barrier: every checkpoint submitted so far is written (the worker is FIFO)."""
    node._db_worker.submit(lambda: None).result(10)


# ------------------------------------------------------------------ R13c


class _StallingEventWriter:
    """An event writer whose FIRST write blocks STALL_S (a stuck disk); all are recorded."""

    def __init__(self) -> None:
        self.durations: list[float] = []
        self.threads: list[str] = []
        self.sequence: list[Any] = []

    def write(self, event: Any) -> None:
        self.threads.append(threading.current_thread().name)
        started = time.monotonic()
        if not self.durations:
            time.sleep(STALL_S)
        self.durations.append(time.monotonic() - started)
        self.sequence.append(event.timestamp)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class _StallCounters:
    def __init__(self) -> None:
        self.unstall = threading.Event()
        self.started: dict[str, int] = {}
        self._lock = threading.Lock()

    def stall(self, name: str) -> None:
        with self._lock:
            self.started[name] = self.started.get(name, 0) + 1
        self.unstall.wait(STALL_S)


class _SlowPersistence(StatePersistence):
    stalls: _StallCounters

    def save_checkpoint(self, checkpoint: StateCheckpoint) -> int:
        self.stalls.stall("save_checkpoint")
        return super().save_checkpoint(checkpoint)

    def delete_old_checkpoints(self, keep_count: int = 100) -> int:
        self.stalls.stall("delete_old_checkpoints")
        return super().delete_old_checkpoints(keep_count)


class _SlowQueue(PaperCommandQueue):
    stalls: _StallCounters

    def claim_pending(self, trader_id: str) -> Any:
        claimed = super().claim_pending(trader_id)
        if claimed:
            self.stalls.stall("claim_pending")
        return claimed

    def complete(self, command_id: int, result: Any) -> None:
        self.stalls.stall("complete")
        super().complete(command_id, result)


@pytest.mark.timing
async def test_full_callback_sequence_halts_within_15s(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    scaled_kill_clock: float,  # noqa: F811
) -> None:
    """R13c/R13d: REAL loop, REAL reader thread and workers. Every former sync DB call
    blocks 5 s (periodic save + prune, command claim + complete, the guard-transition
    checkpoint) and the first event write blocks for 5 s on the event worker; the
    reader fails from t_f. The ERROR halt still lands <= 15.0 s after t_f and the
    watcher never goes quiet for more than 1.5 s. All at ``SCALE`` of those seconds."""
    # This test is about the kill halt: its 5 s claim stall must not count as failed claims
    # (3fg9d) and reach the guard fake, which has no ``pause``.
    monkeypatch.setattr(node_module, "COMMAND_CLAIM_STALL_S", 10.0)
    stalls = _StallCounters()
    guard = _TimedGuard()
    node = PaperTradingNode(_config(seeded, control_poll_interval=0.1))
    node._guard = guard  # type: ignore[assignment]

    persistence = _SlowPersistence(seeded["db"], node._config.trader_id, checkpoint_interval=1)
    persistence.stalls = stalls
    queue = _SlowQueue(seeded["db"])
    queue.stalls = stalls
    node._persistence = persistence
    node._commands = queue
    cid = PaperCommandQueue(seeded["db"]).enqueue(node._config.trader_id, "pause")

    async def executed(_cmd: Any) -> tuple[bool, dict[str, Any], None]:
        return True, {}, None

    monkeypatch.setattr(node, "_execute_command", executed)

    writer = _StallingEventWriter()
    node._journal.writer = writer  # type: ignore[assignment]
    monkeypatch.setattr(
        node, "_capture_checkpoint", lambda: StateCheckpoint(trader_id=node._config.trader_id)
    )

    ticks: list[float] = []
    real_check = node._check_kill_switch

    def recording_check() -> None:
        ticks.append(time.monotonic())
        real_check()

    monkeypatch.setattr(node, "_check_kill_switch", recording_check)

    reader = _SlowFailingReader(stalls.unstall)
    node._kill_monitor = KillSwitchMonitor(lambda: reader, poll_s=scaled_kill_clock)
    node._kill_monitor.start()
    tasks: list[asyncio.Task[None]] = []
    try:
        first = await node._kill_monitor.wait_first_read(5.0)
        assert first.last_ok_mono is not None
        await persistence.start_periodic_checkpointing(
            node._capture_checkpoint, node._db_worker.executor
        )
        tasks = [
            asyncio.create_task(node._control_loop()),
            asyncio.create_task(node._kill_watch_loop()),
        ]
        # t_f + ~1 s: the transition's event write blocks the loop for 5 s, then its
        # checkpoint is queued behind the periodic save + prune.
        asyncio.get_running_loop().call_later(
            1.0 * SCALE,
            node._on_guard_state_change,
            GuardState.PAUSED,
            None,
            "paused",
            {},
        )
        deadline = time.monotonic() + 25.0
        while not guard.calls and time.monotonic() < deadline:
            await asyncio.sleep(0.05)

        assert guard.calls == [("halt", HaltReason.ERROR)]
        assert reader.first_fail_start is not None
        assert guard.halt_times[0] - reader.first_fail_start <= HALT_BOUND_S
        gaps = [b - a for a, b in zip(ticks, ticks[1:], strict=False)]
        assert max(gaps) <= 1.5 * SCALE, max(gaps)
        # R2: the first event write really stalled 5 s, but on the event worker; the
        # lines behind it are written in submit order once it returns.
        node._journal.event_log.barrier().result(15)
        assert writer.durations[0] >= 0.9 * STALL_S, writer.durations
        assert all(name.startswith("vq-paper-events") for name in writer.threads), writer.threads
        assert writer.sequence == sorted(writer.sequence)
        assert node._journal.event_log.health().written == len(writer.sequence) >= 1

        # Every stalled call was real and the work still completes afterwards.
        stalls.unstall.set()
        reader_deadline = time.monotonic() + 15.0
        status = None
        while time.monotonic() < reader_deadline:
            cmd = PaperCommandQueue(seeded["db"]).get(cid)
            status = cmd.status if cmd else None
            # Save and prune are ONE worker job: the 2nd save starts a moment before its
            # prune, so wait on the prune too (the last call this block asserts on).
            if (
                status == "done"
                and stalls.started.get("save_checkpoint", 0) >= 2
                and stalls.started.get("delete_old_checkpoints", 0) >= 1
            ):
                break
            await asyncio.sleep(0.1)
        assert status == "done"
        assert stalls.started["delete_old_checkpoints"] >= 1
        assert stalls.started["claim_pending"] == 1
        assert stalls.started["complete"] == 1
        assert stalls.started["save_checkpoint"] >= 2, "periodic + guard transition"
    finally:
        stalls.unstall.set()
        node._shutdown_event.set()
        for task in tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        node._kill_monitor.stop(2.0)
        await node._shutdown()


# ------------------------------------------------------------------ R19


def _checkpoint(trader_id: str, tag: str) -> StateCheckpoint:
    return StateCheckpoint(trader_id=trader_id, node_status={"tag": tag})


class _OrderedPersistence(StatePersistence):
    """Records the write order; the FIRST save is slow so a second worker would pass it."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.order: list[str] = []
        self._first = True

    def save_checkpoint(self, checkpoint: StateCheckpoint) -> int:
        if self._first:
            self._first = False
            time.sleep(0.3)
        self.order.append(checkpoint.node_status["tag"])
        return super().save_checkpoint(checkpoint)


@pytest.mark.parametrize("first", ["transition", "periodic"])
async def test_checkpoint_fifo_order(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    first: str,
) -> None:
    """R19: whichever path captured first is written first; load_latest is the
    one captured last (a newer snapshot is never overwritten by an older one)."""
    node = PaperTradingNode(_config(seeded))
    persistence = _OrderedPersistence(
        seeded["db"], node._config.trader_id, checkpoint_interval=0.05
    )
    node._persistence = persistence
    trader_id = node._config.trader_id
    periodic_called = threading.Event()

    def periodic_capture() -> StateCheckpoint | None:
        if periodic_called.is_set():
            return None
        periodic_called.set()
        return _checkpoint(trader_id, "periodic")

    monkeypatch.setattr(node, "_capture_checkpoint", lambda: _checkpoint(trader_id, "transition"))
    try:
        if first == "transition":
            node._save_checkpoint_now()
            await persistence.start_periodic_checkpointing(
                periodic_capture, node._db_worker.executor
            )
            while not periodic_called.is_set():
                await asyncio.sleep(0.01)
        else:
            await persistence.start_periodic_checkpointing(
                periodic_capture, node._db_worker.executor
            )
            while not periodic_called.is_set():
                await asyncio.sleep(0.01)
            node._save_checkpoint_now()
        await asyncio.get_running_loop().run_in_executor(None, _drain, node)
        expected = (
            ["transition", "periodic"] if first == "transition" else ["periodic", "transition"]
        )
        assert persistence.order == expected
        latest = persistence.load_latest_checkpoint()
        assert latest is not None and latest.node_status["tag"] == expected[-1]
    finally:
        await persistence.stop_periodic_checkpointing()
        persistence.close()


# ------------------------------------------------------------------ R19b


class _RecordingPersistence:
    def __init__(self) -> None:
        self.saved: list[StateCheckpoint] = []

    def save_checkpoint(self, checkpoint: StateCheckpoint) -> int:
        self.saved.append(checkpoint)
        return 1


def test_transition_checkpoint_in_finally(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R19b (bykze): the transition's event write raises; the checkpoint is STILL
    submitted to the worker (finally) and written, and the write's error propagates."""
    node = PaperTradingNode(_config(seeded))
    persistence = _RecordingPersistence()
    node._persistence = persistence  # type: ignore[assignment]
    error = OSError("disk full")

    def failing_write(*_args: Any, **_kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(node._journal, "record", failing_write)

    with pytest.raises(OSError) as caught:
        node._on_guard_state_change(GuardState.HALTED, HaltReason.MANUAL, "drawdown", {})

    assert caught.value is error
    _drain(node)
    assert len(persistence.saved) == 1
    assert persistence.saved[0].node_status["state"] == "halted"
    assert persistence.saved[0].node_status["halt_reason"] == "manual"


# ------------------------------------------------------------------ R19c


class _HungPersistence(StatePersistence):
    """A save that never returns until released (a locked state DB)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.entered = threading.Event()
        self.release = threading.Event()

    def save_checkpoint(self, checkpoint: StateCheckpoint) -> int:
        self.entered.set()
        self.release.wait(30)
        return super().save_checkpoint(checkpoint)


async def test_shutdown_bounded_with_hung_db(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R19c: a hung executor save costs the shutdown the bounded wait only; the wait
    is logged class-only and the trading engine is still stopped."""
    assert node_module.SHUTDOWN_CHECKPOINT_WAIT_S == 10.0
    monkeypatch.setattr(node_module, "SHUTDOWN_CHECKPOINT_WAIT_S", 0.3)
    monkeypatch.setattr(node_module, "STORE_CLOSE_WAIT_S", 0.2)
    node = PaperTradingNode(_config(seeded))
    persistence = _HungPersistence(seeded["db"], node._config.trader_id)
    node._persistence = persistence
    node._journal.writer = None
    node._journal = FakeJournal()
    trading = _FakeTradingNodeStop()
    node._trading_node = trading  # type: ignore[assignment]
    node._state_manager = _FakeStateManager()  # type: ignore[assignment]
    node._save_checkpoint_now()  # the hung save, already inside the worker
    assert await asyncio.get_running_loop().run_in_executor(None, persistence.entered.wait, 5)

    started = time.monotonic()
    with caplog.at_level(logging.ERROR):
        await asyncio.wait_for(node._shutdown(), 5)
    elapsed = time.monotonic() - started
    try:
        assert 0.3 <= elapsed < 1.5, elapsed
        assert "final checkpoint not saved within" in caplog.text
        assert _LEAK not in caplog.text
        assert all(record.exc_info is None for record in caplog.records)
        assert trading.stopped == 1, "stop_async still runs"
        assert node._persistence is None
    finally:
        persistence.release.set()
        await asyncio.sleep(0.2)
