"""Store connections are closed by their own worker, never from the loop (R19d / DR3).

Bead vibe-quant-dt065 (T2b). ``StatePersistence`` (checkpoint saves) and
``PaperCommandQueue`` (claim_pending / complete / fail) are each owned by one worker
thread. A shutdown signals the worker and waits a bounded time; with a statement
still held on the real connection the wait ends, and the worker closes its own
connection after the job returns.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

import pytest

from tests.unit.test_audit_paper_node import _config
from tests.unit.test_paper_alert_write import _FakeStateManager, _FakeTradingNodeStop
from tests.unit.test_paper_kill_switch_fail_closed import (
    seeded,  # noqa: F401  # fixture
)
from vibe_quant.paper import node as node_module
from vibe_quant.paper.node import PaperTradingNode
from vibe_quant.paper.persistence import (
    PaperCommandQueue,
    StateCheckpoint,
    StatePersistence,
    StoreWorker,
)


class _HeldPersistence(StatePersistence):
    """Holds the REAL connection mid-save: the INSERT ran, the commit has not."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.in_save = threading.Event()
        self.release = threading.Event()
        self.close_threads: list[str] = []

    def save_checkpoint(self, checkpoint: StateCheckpoint) -> int:
        if not self.in_save.is_set():  # only the first save is held
            self.conn.execute("BEGIN IMMEDIATE")
            self.in_save.set()
            self.release.wait(30)
            self.conn.commit()
        return super().save_checkpoint(checkpoint)

    def close(self) -> None:
        self.close_threads.append(threading.current_thread().name)
        super().close()


class _HeldQueue(PaperCommandQueue):
    """Holds the REAL connection mid-claim: the SELECT ran, the UPDATEs have not."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.in_claim = threading.Event()
        self.release = threading.Event()
        self.close_threads: list[str] = []

    def claim_pending(self, trader_id: str) -> Any:
        self.conn.execute("BEGIN IMMEDIATE")
        self.in_claim.set()
        self.release.wait(30)
        self.conn.commit()
        return super().claim_pending(trader_id)

    def close(self) -> None:
        self.close_threads.append(threading.current_thread().name)
        super().close()


def _wait(event: threading.Event) -> Any:
    return asyncio.get_running_loop().run_in_executor(None, event.wait, 5)


def _fast_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(node_module, "SHUTDOWN_CHECKPOINT_WAIT_S", 0.3)
    monkeypatch.setattr(node_module, "STORE_CLOSE_WAIT_S", 0.3)


def _wait_closed(store: Any, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if store._conn is None and store.close_threads:
            return True
        time.sleep(0.01)
    return False


async def test_stop_bounded_while_save_hung(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R19d: a save holds the real checkpoint connection; shutdown returns within its
    bound with the connection still open; after release the WORKER closes it."""
    _fast_bounds(monkeypatch)
    node = PaperTradingNode(_config(seeded))
    persistence = _HeldPersistence(seeded["db"], node._config.trader_id)
    node._persistence = persistence
    monkeypatch.setattr(node, "_write_event", lambda *a, **k: None)
    node._trading_node = _FakeTradingNodeStop()  # type: ignore[assignment]
    node._state_manager = _FakeStateManager()  # type: ignore[assignment]
    node._save_checkpoint_now()
    assert await _wait(persistence.in_save)

    started = time.monotonic()
    await asyncio.wait_for(node._shutdown(), 5)
    assert time.monotonic() - started < 1.5
    assert persistence.close_threads == [], "stop never closes a connection that is in use"
    assert persistence._conn is not None

    persistence.release.set()
    assert _wait_closed(persistence), "the worker closes its connection after the held job"
    assert persistence.close_threads == ["vq-paper-db_0"]
    # Nothing was lost: the held save and the final shutdown save both landed.
    reader = StatePersistence(seeded["db"], node._config.trader_id)
    try:
        count = reader.conn.execute("SELECT COUNT(*) FROM paper_trading_checkpoints").fetchone()[0]
    finally:
        reader.close()
    assert count == 2


async def test_stop_bounded_while_claim_hung(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R19d: ``claim_pending`` holds the real queue connection; shutdown returns within
    its bound and the worker closes the connection once the claim returns."""
    _fast_bounds(monkeypatch)
    node = PaperTradingNode(_config(seeded))
    queue = _HeldQueue(seeded["db"])
    node._commands = queue
    node._persistence = None
    node._trading_node = _FakeTradingNodeStop()  # type: ignore[assignment]
    node._state_manager = _FakeStateManager()  # type: ignore[assignment]
    node._control_task = asyncio.create_task(node.process_commands())
    assert await _wait(queue.in_claim)

    started = time.monotonic()
    await asyncio.wait_for(node._shutdown(), 5)
    assert time.monotonic() - started < 1.5
    assert queue.close_threads == []
    assert queue._conn is not None

    queue.release.set()
    assert _wait_closed(queue)
    assert queue.close_threads == ["vq-paper-cmd_0"]


def test_worker_stop_is_bounded_idempotent_and_closes_last() -> None:
    """The worker contract itself: close is the LAST job, stop(timeout) returns False
    while a job is running, a second stop does not queue another close."""
    worker = StoreWorker("vq-test-worker")
    release = threading.Event()
    order: list[str] = []
    worker.submit(lambda: (release.wait(5), order.append("job")))

    started = time.monotonic()
    assert worker.stop(lambda: order.append("close"), timeout_s=0.2) is False
    assert 0.2 <= time.monotonic() - started < 0.6
    assert worker.stop(lambda: order.append("second-close"), timeout_s=0.0) is False

    release.set()
    assert worker.stop(lambda: order.append("third-close"), timeout_s=5.0) is True
    assert order == ["job", "close"]
    with pytest.raises(RuntimeError):
        worker.submit(lambda: None)


async def test_shutdown_still_stops_engine_when_store_close_raises(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Risk F1 (18def66b680c20f8-4926): a raising store stop before ``stop_async`` is
    logged class-only and never skips stopping the trading engine (live orders)."""
    node = PaperTradingNode(_config(seeded))
    node._persistence = StatePersistence(seeded["db"], node._config.trader_id)
    node._commands = PaperCommandQueue(seeded["db"])
    monkeypatch.setattr(node, "_write_event", lambda *a, **k: None)
    trading = _FakeTradingNodeStop()
    node._trading_node = trading  # type: ignore[assignment]
    node._state_manager = _FakeStateManager()  # type: ignore[assignment]

    async def boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("ZZLEAK9 close")

    monkeypatch.setattr(node, "_close_store", boom)
    with caplog.at_level(logging.ERROR):
        await asyncio.wait_for(node._shutdown(), 5)

    assert trading.stopped == 1
    assert "RuntimeError" in caplog.text
    assert "ZZLEAK9" not in caplog.text
    assert node._persistence is None and node._commands is None
    node._db_worker.stop(lambda: None, 2)
    node._cmd_worker.stop(lambda: None, 2)


def _status(db: Any, cid: int) -> str:
    queue = PaperCommandQueue(db)
    try:
        cmd = queue.get(cid)
        assert cmd is not None
        return cmd.status
    finally:
        queue.close()


async def test_cancel_mid_claim_never_leaves_running(
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """Cancelling the control loop while the worker is inside ``claim_pending``: the
    worker still claims, so those rows must end ``failed`` (fixed text), not ``running``."""
    node = PaperTradingNode(_config(seeded))
    setup = PaperCommandQueue(seeded["db"])
    cids = [setup.enqueue(node._config.trader_id, "halt") for _ in range(2)]
    setup.close()
    queue = _HeldQueue(seeded["db"])
    node._commands = queue
    task = asyncio.create_task(node.process_commands())
    assert await _wait(queue.in_claim)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    queue.release.set()
    await asyncio.get_running_loop().run_in_executor(
        None, lambda: node._cmd_worker.submit(lambda: None).result(5)
    )
    assert [_status(seeded["db"], c) for c in cids] == ["failed", "failed"]
    queue2 = PaperCommandQueue(seeded["db"])
    cmd = queue2.get(cids[0])
    assert cmd is not None and cmd.error == "node stopping; command not executed"
    queue2.close()
    node._cmd_worker.stop(queue.close, 2)


async def test_cancel_mid_execute_fails_current_and_rest(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel while the first claimed command executes: it and the unexecuted one end
    ``failed``; an already-finished command is never rewritten."""
    node = PaperTradingNode(_config(seeded))
    setup = PaperCommandQueue(seeded["db"])
    cids = [setup.enqueue(node._config.trader_id, "halt") for _ in range(2)]
    setup.close()
    queue = PaperCommandQueue(seeded["db"])
    node._commands = queue
    started = asyncio.Event()

    async def hang(_cmd: Any) -> Any:
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(node, "_execute_command", hang)
    task = asyncio.create_task(node.process_commands())
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.get_running_loop().run_in_executor(
        None, lambda: node._cmd_worker.submit(lambda: None).result(5)
    )
    assert [_status(seeded["db"], c) for c in cids] == ["failed", "failed"]
    node._cmd_worker.stop(queue.close, 2)
