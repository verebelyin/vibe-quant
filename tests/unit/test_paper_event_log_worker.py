"""Tests for the bounded paper event-log FIFO worker (EventLogWorker)."""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from vibe_quant.logging.events import Event, EventType
from vibe_quant.logging.writer import EventWriter
from vibe_quant.paper.event_log import (
    EventLogFullError,
    EventLogWorker,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _event(ts: datetime, marker: str) -> Event:
    return Event(
        timestamp=ts,
        event_type=EventType.SIGNAL,
        run_id="run-1",
        strategy_name="s1",
        data={"marker": marker},
    )


def _read_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class _GateWriter:
    """Delegates to a real EventWriter, blocking inside write() while the gate is closed."""

    def __init__(self, inner: EventWriter) -> None:
        self._inner = inner
        self._gate = threading.Event()
        self._entered = threading.Event()

    def close(self) -> None:
        self._gate.clear()

    def open(self) -> None:
        self._gate.set()

    def wait_entered(self, timeout: float = 5.0) -> None:
        assert self._entered.wait(timeout), "worker never entered write()"
        self._entered.clear()

    def write(self, event: Event) -> None:
        self._entered.set()
        # Bounded wait so a failing test can never hang the worker thread at teardown.
        self._gate.wait(timeout=2.0)
        self._inner.write(event)


def _full_warnings(caplog: pytest.LogCaptureFixture) -> int:
    return sum(
        1
        for record in caplog.records
        if record.levelname == "WARNING" and "paper event log full" in record.getMessage()
    )


@pytest.fixture
def worker_factory() -> Iterator[object]:
    created: list[EventLogWorker] = []

    def _make(**kwargs: object) -> EventLogWorker:
        worker = EventLogWorker(**kwargs)  # type: ignore[arg-type]
        created.append(worker)
        return worker

    yield _make
    for worker in created:
        worker.worker.begin_stop(lambda: None)


class TestEventLogWorkerQueueBound:
    def test_queue_exactly_full_drops_the_next_and_counts_it(
        self, worker_factory: object, tmp_path: Path
    ) -> None:
        writer = EventWriter("run-a", base_path=tmp_path)
        gate = _GateWriter(writer)
        worker = worker_factory(max_pending=3)  # type: ignore[operator]
        t0 = datetime(2026, 1, 2, 3, 0, tzinfo=UTC)

        gate.close()
        worker.submit(gate, _event(t0, "e1"))
        gate.wait_entered()
        worker.submit(gate, _event(t0, "e2"))
        worker.submit(gate, _event(t0, "e3"))

        health = worker.health()
        assert health.pending == 3
        assert health.dropped == 0

        dropped = worker.submit(gate, _event(t0, "e4"))
        assert dropped.done()
        assert isinstance(dropped.exception(), EventLogFullError)
        health = worker.health()
        assert health.dropped == 1
        assert health.pending == 3

        gate.open()
        worker.barrier().result(timeout=5)

        lines = _read_lines(writer.file_path)
        assert [line["data"]["marker"] for line in lines] == ["e1", "e2", "e3"]  # type: ignore[index]
        assert worker.health().pending == 0

        worker.submit(gate, _event(t0, "e5")).result(timeout=5)
        lines = _read_lines(writer.file_path)
        assert [line["data"]["marker"] for line in lines] == ["e1", "e2", "e3", "e5"]  # type: ignore[index]

    def test_full_warning_once_per_streak(
        self, worker_factory: object, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        writer = EventWriter("run-b", base_path=tmp_path)
        gate = _GateWriter(writer)
        worker = worker_factory(max_pending=3)  # type: ignore[operator]
        t0 = datetime(2026, 1, 2, 3, 0, tzinfo=UTC)

        with caplog.at_level(logging.WARNING):
            gate.close()
            worker.submit(gate, _event(t0, "e1"))
            gate.wait_entered()
            accepted = [worker.submit(gate, _event(t0, f"e{i}")) for i in (2, 3)]
            for i in range(4, 7):
                worker.submit(gate, _event(t0, f"d{i}"))
            assert worker.health().dropped == 3
            assert _full_warnings(caplog) == 1

            gate.open()
            worker.barrier().result(timeout=5)
            for future in accepted:
                future.result(timeout=5)

            gate.close()
            worker.submit(gate, _event(t0, "ok4"))
            gate.wait_entered()
            worker.submit(gate, _event(t0, "ok5"))
            worker.submit(gate, _event(t0, "ok6"))
            worker.submit(gate, _event(t0, "d7"))
            assert worker.health().dropped == 4
            assert _full_warnings(caplog) == 2
            gate.open()
            worker.barrier().result(timeout=5)

    def test_barrier_bypasses_bound_and_waits_fifo(
        self, worker_factory: object, tmp_path: Path
    ) -> None:
        writer = EventWriter("run-c", base_path=tmp_path)
        gate = _GateWriter(writer)
        worker = worker_factory(max_pending=3)  # type: ignore[operator]
        t0 = datetime(2026, 1, 2, 3, 0, tzinfo=UTC)

        gate.close()
        worker.submit(gate, _event(t0, "e1"))
        gate.wait_entered()
        worker.submit(gate, _event(t0, "e2"))
        worker.submit(gate, _event(t0, "e3"))
        assert worker.health().pending == 3

        barrier = worker.barrier()
        assert not barrier.done()

        gate.open()
        assert barrier.result(timeout=5) is None
        lines = _read_lines(writer.file_path)
        assert [line["data"]["marker"] for line in lines] == ["e1", "e2", "e3"]  # type: ignore[index]
        assert worker.health().pending == 0


class TestEventLogWorkerOrdering:
    def test_lines_keep_submit_order(self, worker_factory: object, tmp_path: Path) -> None:
        writer = EventWriter("run-d", base_path=tmp_path)
        gate = _GateWriter(writer)
        worker = worker_factory()  # type: ignore[operator]
        t0 = datetime(2026, 1, 2, 3, 0, tzinfo=UTC)
        t1 = t0
        t2 = t0 + timedelta(seconds=1)
        t3 = t0 + timedelta(seconds=2)

        gate.close()
        worker.submit(gate, _event(t1, "e1"))
        gate.wait_entered()
        worker.submit(gate, _event(t2, "e2"))
        worker.submit(gate, _event(t3, "e3"))

        gate.open()
        worker.barrier().result(timeout=5)

        lines = _read_lines(writer.file_path)
        assert [line["ts"] for line in lines] == [t1.isoformat(), t2.isoformat(), t3.isoformat()]
        assert [line["data"]["marker"] for line in lines] == ["e1", "e2", "e3"]  # type: ignore[index]


class TestEventLogWorkerAfterStop:
    def test_submit_after_stop_raises_and_leaks_no_pending(
        self, worker_factory: object, tmp_path: Path
    ) -> None:
        """S1 (rev-e4-evlog-1): a submit refused by the stopped executor must re-raise.

        ``StoreWorker.begin_stop`` shuts the executor down, so ``executor.submit``
        raises ``RuntimeError``. ``EventLogWorker.submit`` reserves a slot
        (``pending += 1``) *before* handing the job over, so the refusal path must
        give that slot back: otherwise ``pending`` leaks forever and every later
        submit is refused as "full".
        """
        writer = EventWriter("run-f", base_path=tmp_path)
        worker = worker_factory()  # type: ignore[operator]
        t0 = datetime(2026, 1, 2, 3, 0, tzinfo=UTC)
        event = _event(t0, "after-stop")

        worker.worker.begin_stop(lambda: None)

        with pytest.raises(RuntimeError):
            worker.submit(writer, event)
        assert worker.health().pending == 0


class TestEventLogWorkerFailure:
    def test_writer_error_counted_and_logged_class_only(
        self, worker_factory: object, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        worker = worker_factory()  # type: ignore[operator]
        boom = OSError("disk ZZLEAK9")

        class _BoomWriter:
            def write(self, event: Event) -> None:
                raise boom

        t0 = datetime(2026, 1, 2, 3, 0, tzinfo=UTC)
        with caplog.at_level(logging.ERROR):
            future = worker.submit(_BoomWriter(), _event(t0, "bad"))
            assert future.exception(timeout=5) is boom
            # The pending decrement is in the job's finally (ruling D-SF2): a failed
            # write must release its slot too, before the future resolves.
            assert worker.health().pending == 0

        health = worker.health()
        assert health.failed == 1
        assert health.last_error_class == "OSError"
        errors = [record for record in caplog.records if record.levelname == "ERROR"]
        assert len(errors) == 1
        assert errors[0].getMessage() == "paper event write failed (OSError)"
        assert errors[0].exc_info is None
        assert "ZZLEAK9" not in caplog.text

        writer = EventWriter("run-e", base_path=tmp_path)
        worker.submit(writer, _event(t0, "ok")).result(timeout=5)
        lines = _read_lines(writer.file_path)
        assert [line["data"]["marker"] for line in lines] == ["ok"]  # type: ignore[index]
        assert worker.health().written == 1
        assert worker.health().pending == 0
