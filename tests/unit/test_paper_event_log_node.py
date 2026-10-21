"""Paper event-file writes run on their own worker thread (bead vibe-quant-ogd37, T2).

Contract rows of job 20261017-easy4 (design-ogd37.md § 4), node half:

* R1   a stalled ``writer.write`` never holds the caller: ``_write_event`` returns a
       not-done future at once and the append runs on a ``vq-paper-events`` thread.
* R3b  the default timestamp is taken at submit, not when the worker writes.
* R6   a sync alert site (pre-guard ``halt``) alerts without waiting for the disk.
* R7   a failing disk never skips risk evaluation (position open/close through the guard).
* R8c  liquidation report, write not confirmed within EVENT_WRITE_WAIT_S: alert, then
       ``EventWriteTimeoutError``; the watcher retries the row.
* R8d  liquidation report, queue full: alert, then ``EventLogFullError``; row retried.
* R8e  liquidation report cancelled mid-wait: ``CancelledError`` propagates, the alert is
       sent once, the line still lands, the watcher cursor is unchanged.
* R9   an unwritable / stalled event log refuses to start.
* R10  shutdown drains queued lines in order, then closes the file on the event worker,
       bounded by STORE_CLOSE_WAIT_S; later writes are no-ops.
* R11  an event write is not queued behind a hung checkpoint save.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

import vibe_quant.paper.guard as guard_module
import vibe_quant.paper.node as node_module
from tests.unit.test_audit_paper_node import _config
from tests.unit.test_paper_db_off_loop import _HungPersistence
from tests.unit.test_paper_liquidation_detector import (
    _Alerts,
    _FakeTradingNode,
    _harness,
    _order,
    _until,
    _validation_run,
    fake_tradingnode,  # noqa: F401  # fixture
    seeded,  # noqa: F401  # fixture
)
from tests.unit.test_paper_startup_and_estimate import (
    _guarded_event,
)
from vibe_quant.logging.events import EventType
from vibe_quant.logging.writer import EventWriter
from vibe_quant.paper.config import ConfigurationError
from vibe_quant.paper.event_log import EventLogFullError, EventLogWorker, EventWriteTimeoutError
from vibe_quant.paper.guard import GuardState, HaltReason
from vibe_quant.paper.node import PaperTradingNode
from vibe_quant.paper.persistence import StateCheckpoint

if TYPE_CHECKING:
    from pathlib import Path

    from vibe_quant.logging.events import Event

_EVENT_THREAD = "vq-paper-events"
_ISOLATED_10X = {"liquidation_mode": "isolated", "leverage": 10}
#: The watcher state before any row was delivered: both cursors at the injected node start.
_UNCHANGED_WATCH_STATE = {
    "cursors": {"BTCUSDT": 1_767_322_800_000, "ETHUSDT": 1_767_322_800_000},
    "seen": {},
}


class _GateWriter:
    """An ``EventWriter`` stand-in whose ``write`` blocks until ``release`` is set."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.entered = threading.Event()
        self.events: list[Event] = []
        self.threads: list[str] = []

    def write(self, event: Event) -> None:
        self.entered.set()
        self.threads.append(threading.current_thread().name)
        self.release.wait(10)
        self.events.append(event)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def _gate_real_writer(
    monkeypatch: pytest.MonkeyPatch, writer: EventWriter
) -> tuple[threading.Event, threading.Event]:
    """Make a real writer's ``write`` block until ``release``; returns (entered, release)."""
    original = writer.write
    entered, release = threading.Event(), threading.Event()

    def write(event: Event) -> None:
        entered.set()
        release.wait(10)
        original(event)

    monkeypatch.setattr(writer, "write", write)
    return entered, release


def _node(seeded: dict[str, Any], **kw: Any) -> PaperTradingNode:  # noqa: F811
    return PaperTradingNode(_config(seeded, **kw))


def _stop_event_worker(node: PaperTradingNode) -> None:
    node._event_log.worker.begin_stop(lambda: None)


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _liquidation_alert(order: Any) -> tuple[str, str, bool]:
    return (
        "error",
        f"EXCHANGE {order.auto_close_type} {order.symbol} {order.side} "
        f"qty {order.executed_qty} @ {order.avg_price} (order {order.order_id}); "
        "paper position state may be stale; NO paper action taken",
        True,
    )


# ------------------------------------------------------------------- R1 / R3b


def test_write_event_returns_before_a_stalled_write_finishes(
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """R1: the caller gets a not-done future in < 0.05 s; the line lands on the worker."""
    node = _node(seeded)
    writer = _GateWriter()
    node._event_writer = writer  # type: ignore[assignment]
    try:
        started = time.monotonic()
        future = node._write_event(EventType.SIGNAL, {"action": "x"})
        elapsed = time.monotonic() - started

        assert future is not None
        assert elapsed < 0.05, elapsed
        assert writer.entered.wait(5), "the worker did start the write"
        assert not future.done(), "the stalled write is still pending"
        assert writer.events == []

        writer.release.set()
        future.result(5)
        assert [e.data for e in writer.events] == [{"action": "x"}]
        assert writer.threads == [writer.threads[0]]
        assert writer.threads[0].startswith(_EVENT_THREAD), writer.threads
    finally:
        writer.release.set()
        _stop_event_worker(node)


def test_default_timestamp_is_taken_at_submit(
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """R3b: a write stalled 0.5 s still carries the submit-time ``ts``."""
    node = _node(seeded)
    writer = _GateWriter()
    node._event_writer = writer  # type: ignore[assignment]
    try:
        before = datetime.now(UTC)
        future = node._write_event(EventType.SIGNAL, {"action": "x"})
        after = datetime.now(UTC)
        assert future is not None
        time.sleep(0.5)
        writer.release.set()
        future.result(5)

        assert before <= writer.events[0].timestamp <= after
    finally:
        writer.release.set()
        _stop_event_worker(node)


@pytest.mark.parametrize("event_type", [EventType.SIGNAL, EventType.LIFECYCLE])
def test_write_event_snapshots_the_payload_at_submit(
    event_type: EventType,
    seeded: dict[str, Any],  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3c: the payload is captured at submit, so a caller mutating its dict while the
    write is stalled (value, new key, nested list, nested dict) cannot change the line."""
    node = _node(seeded)
    writer = EventWriter(run_id=node._config.trader_id, base_path=tmp_path / "logs")
    node._event_writer = writer
    _entered, release = _gate_real_writer(monkeypatch, writer)
    levels: list[int] = [1, 2]
    nested: dict[str, str] = {"side": "long"}
    payload: dict[str, object] = {"action": "signal", "levels": levels, "nested": nested}
    try:
        future = node._write_event(event_type, payload)
        assert future is not None
        assert _entered.wait(5), "the write is stalled and has not serialized yet"

        payload["action"] = "changed"
        payload["added"] = "later"
        levels.append(3)
        nested["side"] = "short"

        release.set()
        node._event_log.barrier().result(10)
        line = _lines(tmp_path / "logs" / f"{node._config.trader_id}.jsonl")[-1]
        assert line["event"] == event_type.value
        assert line["data"] == {"action": "signal", "levels": [1, 2], "nested": {"side": "long"}}
    finally:
        release.set()
        monkeypatch.undo()
        _stop_event_worker(node)


async def test_write_after_shutdown_is_a_no_op(
    seeded: dict[str, Any],  # noqa: F811
    tmp_path: Path,
) -> None:
    """R10d: after ``_shutdown`` a write returns None and touches nothing."""
    node = _node(seeded)
    node._event_writer = EventWriter(run_id=node._config.trader_id, base_path=tmp_path / "logs")
    node._write_event(EventType.SIGNAL, {"action": "early"})
    await node._shutdown()
    path = tmp_path / "logs" / f"{node._config.trader_id}.jsonl"
    before = path.read_text()

    assert node._write_event(EventType.SIGNAL, {"action": "late"}) is None
    assert path.read_text() == before
    assert "early" in before
    assert "late" not in before


# ------------------------------------------------------------------------ R6


def test_sync_alert_site_never_waits_for_the_disk(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R6: a guard halt alerts at once while the event write is still stalled."""
    node = _node(seeded)
    writer = _GateWriter()
    node._event_writer = writer  # type: ignore[assignment]
    alerts = _Alerts()
    monkeypatch.setattr(node, "_send_telegram_alert", alerts)
    try:
        started = time.monotonic()
        node._on_guard_state_change(GuardState.HALTED, HaltReason.MANUAL, "x", {})
        elapsed = time.monotonic() - started

        assert elapsed < 0.05, elapsed
        assert alerts.calls == [("circuit_breaker", "HALT (manual): x", True)]
        assert writer.events == [], "the line has not landed yet"

        writer.release.set()
        node._event_log.barrier().result(5)
        assert [e.data["action"] for e in writer.events] == ["halt"]
    finally:
        writer.release.set()
        _stop_event_worker(node)


# ------------------------------------------------------------------------ R7


async def test_failing_disk_never_skips_risk_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
) -> None:
    """R7: with every append failing, a position open and a losing close through the
    real guard path still reach ``evaluate_risk`` (and the loss is counted)."""
    harness = await _harness(
        monkeypatch, seeded, validation_run_id=_validation_run(seeded, _ISOLATED_10X)
    )
    node = harness.node
    guard = node._guard
    writer = node._event_writer
    assert guard is not None and writer is not None
    attempted: list[EventType] = []

    def write(event: Event) -> None:
        attempted.append(event.event_type)
        raise OSError("disk full")

    monkeypatch.setattr(writer, "write", write)
    calls: list[str] = []
    monkeypatch.setattr(guard, "evaluate_risk", lambda: calls.append("evaluate_risk"))
    monkeypatch.setattr(guard, "_is_guarded_strategy", lambda _sid: True)
    monkeypatch.setattr(guard, "_stop_exited_strategies", lambda: None)
    try:
        opened = _guarded_event()
        monkeypatch.setattr(guard_module, "PositionOpened", type(opened))
        guard._on_position_event(opened)
        node._event_log.barrier().result(10)

        assert calls == ["evaluate_risk"], "the failed append must not skip the risk check"
        assert attempted == [EventType.POSITION_OPEN, EventType.RISK_CHECK], (
            "the open line and the liquidation estimate were both submitted"
        )

        closed = _closed_event()
        monkeypatch.setattr(guard_module, "PositionClosed", type(closed))
        before = guard._risk.consecutive_losses
        guard._on_position_event(closed)
        node._event_log.barrier().result(10)

        assert calls == ["evaluate_risk", "evaluate_risk"]
        assert guard._risk.consecutive_losses == before + 1, "the losing close is counted"
        assert attempted[-1] == EventType.POSITION_CLOSE
        assert node._event_log.health().failed == 3
    finally:
        monkeypatch.undo()
        await node._shutdown()


class _Money:
    currency = "USDT"

    def __init__(self, value: float) -> None:
        self.value = value

    def as_decimal(self) -> Any:
        from decimal import Decimal

        return Decimal(str(self.value))

    def __float__(self) -> float:
        return self.value


def _closed_event() -> Any:
    from types import SimpleNamespace

    from nautilus_trader.model.identifiers import InstrumentId, PositionId

    class _Closed(SimpleNamespace):
        pass

    return _Closed(
        strategy_id="S-1",
        position_id=PositionId("P-1"),
        instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
        realized_pnl=_Money(-5.0),
        avg_px_close=59000.0,
        closing_order_id=None,
        ts_event=1_713_365_300_000_000_000,
    )


# --------------------------------------------------------------------- R8c-e


async def test_liquidation_write_timeout_alerts_raises_and_retries(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
) -> None:
    """R8c: an unconfirmed write alerts, raises the timeout, leaves the row for the
    next cycle, which writes and alerts again (a duplicate is accepted, R47b)."""
    order = _order()
    harness = await _harness(monkeypatch, seeded, orders={"BTCUSDT": [order]})
    node = harness.node
    writer = node._event_writer
    assert writer is not None
    entered, release = _gate_real_writer(monkeypatch, writer)
    monkeypatch.setattr(node_module, "EVENT_WRITE_WAIT_S", 0.3)
    watcher = node._build_liquidation_watcher()
    try:
        started = time.monotonic()
        with pytest.raises(EventWriteTimeoutError):
            await node._on_exchange_liquidation(order)
        assert time.monotonic() - started >= 0.3
        assert harness.alerts.calls == [_liquidation_alert(order)]

        # The watcher's own cycle: the same timeout fails the symbol, nothing is marked seen.
        await watcher.poll_once()
        assert watcher.state() == _UNCHANGED_WATCH_STATE
        assert len(harness.alerts.calls) == 2

        release.set()
        node._event_log.barrier().result(10)
        assert entered.is_set()
        assert len(harness.events(EventType.LIQUIDATION)) == 2, "both stalled writes landed"

        harness.clock.advance(61)
        await watcher.poll_once()
        assert len(harness.events(EventType.LIQUIDATION)) == 3
        assert len(harness.alerts.calls) == 3
        assert list(watcher.state()["seen"]) == [f"BTCUSDT:{order.order_id}"]
    finally:
        release.set()
        monkeypatch.undo()
        await node._shutdown()


async def test_liquidation_dropped_write_alerts_and_retries(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
) -> None:
    """R8d: a full queue drops the line; the alert still goes out, the error is raised
    and the row is left for the next cycle."""
    order = _order()
    harness = await _harness(monkeypatch, seeded, orders={"BTCUSDT": [order]})
    node = harness.node
    writer = node._event_writer
    assert writer is not None
    entered, release = _gate_real_writer(monkeypatch, writer)
    node._event_log = EventLogWorker(_EVENT_THREAD, max_pending=1)
    watcher = node._build_liquidation_watcher()
    try:
        node._write_event(EventType.SIGNAL, {"action": "fill"})
        assert entered.wait(5)
        assert node._event_log.health().pending == 1

        with pytest.raises(EventLogFullError):
            await node._on_exchange_liquidation(order)
        assert harness.alerts.calls == [_liquidation_alert(order)]
        assert node._event_log.health().dropped == 1

        await watcher.poll_once()
        assert watcher.state()["seen"] == {}, "the dropped row is not marked seen"
        assert watcher.state()["cursors"]["BTCUSDT"] == 1_767_322_800_000
    finally:
        release.set()
        monkeypatch.undo()
        await node._shutdown()


async def test_cancel_during_liquidation_wait_keeps_the_row(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
) -> None:
    """R8e: a shutdown cancel mid-wait propagates, alerts exactly once with today's
    text, the line still lands, and the watcher cursor / seen set are unchanged."""
    order = _order()
    harness = await _harness(monkeypatch, seeded, orders={"BTCUSDT": [order]})
    node = harness.node
    writer = node._event_writer
    assert writer is not None
    entered, release = _gate_real_writer(monkeypatch, writer)
    watcher = node._build_liquidation_watcher()
    try:
        task = asyncio.create_task(watcher.poll_once())
        await _until(entered.is_set)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert harness.alerts.calls == [_liquidation_alert(order)]
        assert watcher.state() == _UNCHANGED_WATCH_STATE

        release.set()
        node._event_log.barrier().result(10)
        assert len(harness.events(EventType.LIQUIDATION)) == 1
    finally:
        release.set()
        monkeypatch.undo()
        await node._shutdown()


# ------------------------------------------------------------------------ R9


class _FailingWriter(EventWriter):
    def write(self, event: Event) -> None:
        raise OSError("disk ZZLEAK9")


async def test_unwritable_event_log_refuses_start(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
) -> None:
    """R9: a log whose first lines fail never lets the engine be built."""
    monkeypatch.setattr(node_module, "EventWriter", _FailingWriter)
    node = _node(seeded)
    try:
        with pytest.raises(ConfigurationError) as caught:
            await node._initialize()

        assert str(caught.value) == "paper event log not writable (OSError); refusing to start"
        assert node._trading_node is None
        assert fake_tradingnode.instances == []
    finally:
        await node._shutdown()


async def test_stalled_event_log_refuses_start(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
) -> None:
    """R9b: a log that does not answer within EVENT_STARTUP_WAIT_S refuses to start."""
    release = threading.Event()

    class _StalledWriter(EventWriter):
        def write(self, event: Event) -> None:
            release.wait(10)

    monkeypatch.setattr(node_module, "EventWriter", _StalledWriter)
    monkeypatch.setattr(node_module, "EVENT_STARTUP_WAIT_S", 0.3)
    node = _node(seeded)
    try:
        started = time.monotonic()
        with pytest.raises(ConfigurationError) as caught:
            await node._initialize()

        assert time.monotonic() - started < 5
        assert str(caught.value) == "paper event log not writable within 0.3 s; refusing to start"
        assert node._trading_node is None
    finally:
        release.set()
        monkeypatch.undo()
        await node._shutdown()


# ------------------------------------------------------------------------ R10


async def test_shutdown_drains_in_order_then_closes(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
) -> None:
    """R10: queued lines + the stop line land in submit order, the file is flushed and
    closed once on the event worker, after the trading engine stopped."""
    harness = await _harness(monkeypatch, seeded)
    node = harness.node
    writer = node._event_writer
    assert writer is not None
    trace: list[str] = []
    threads: dict[str, str] = {}
    original_flush, original_close = writer.flush, writer.close

    def flush() -> None:
        trace.append("flush")
        threads["flush"] = threading.current_thread().name
        original_flush()

    def close() -> None:
        trace.append("close")
        threads["close"] = threading.current_thread().name
        original_close()

    async def stop_async(_self: object) -> None:
        trace.append("stop_async")

    monkeypatch.setattr(writer, "flush", flush)
    monkeypatch.setattr(writer, "close", close)
    monkeypatch.setattr(fake_tradingnode, "stop_async", stop_async)
    release = threading.Event()
    original_write = writer.write

    def slow_write(event: Event) -> None:
        release.wait(10)
        original_write(event)

    monkeypatch.setattr(writer, "write", slow_write)
    for i in range(5):
        node._write_event(EventType.SIGNAL, {"action": "queued", "i": i})
    release.set()

    await node._shutdown()

    entries = _lines(harness.log)
    actions = [(e["data"].get("action"), e["data"].get("i")) for e in entries]
    assert actions[-6:] == [("queued", i) for i in range(5)] + [("stop", None)]
    assert trace == ["stop_async", "flush", "close"]
    assert threads["flush"].startswith(_EVENT_THREAD)
    assert threads["close"].startswith(_EVENT_THREAD)
    assert node._event_writer is None


async def test_shutdown_bounded_with_hung_event_write(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R10b: a hung append costs the shutdown the bounded wait only; the engine is
    stopped, and the worker writes the queued lines, then closes, once released."""
    harness = await _harness(monkeypatch, seeded)
    node = harness.node
    writer = node._event_writer
    assert writer is not None
    _entered, release = _gate_real_writer(monkeypatch, writer)
    closed = threading.Event()
    original_close = writer.close

    def close() -> None:
        original_close()
        closed.set()

    monkeypatch.setattr(writer, "close", close)
    stopped: list[int] = []

    async def stop_async(_self: object) -> None:
        stopped.append(1)

    monkeypatch.setattr(fake_tradingnode, "stop_async", stop_async)
    monkeypatch.setattr(node_module, "STORE_CLOSE_WAIT_S", 0.2)
    node._write_event(EventType.SIGNAL, {"action": "queued", "i": 0})
    node._write_event(EventType.SIGNAL, {"action": "queued", "i": 1})
    try:
        started = time.monotonic()
        with caplog.at_level(logging.ERROR):
            await node._shutdown()
        assert time.monotonic() - started < 1.5
        assert "paper event writer still busy; its worker closes it when the job returns" in (
            caplog.text
        )
        assert stopped == [1]
        assert not closed.is_set()
    finally:
        release.set()
    assert closed.wait(10), "the worker closes the file after its last queued job"
    actions = [e["data"].get("action") for e in _lines(harness.log)]
    assert actions[-3:] == ["queued", "queued", "stop"]


# ------------------------------------------------------------------------ R11


async def test_event_write_not_behind_a_hung_checkpoint(
    seeded: dict[str, Any],  # noqa: F811
    tmp_path: Path,
) -> None:
    """R11: a checkpoint save hung on the DB worker does not delay an event line."""
    node = _node(seeded)
    node._event_writer = EventWriter(run_id=node._config.trader_id, base_path=tmp_path / "logs")
    persistence = _HungPersistence(seeded["db"], node._config.trader_id)
    node._persistence = persistence
    try:
        node._db_worker.submit(
            persistence.save_checkpoint, StateCheckpoint(trader_id=node._config.trader_id)
        )
        assert persistence.entered.wait(5)

        started = time.monotonic()
        future = node._write_event(EventType.SIGNAL, {"action": "x"})
        assert future is not None
        future.result(1)
        assert time.monotonic() - started < 1.0
    finally:
        persistence.release.set()
        node._db_worker.begin_stop(persistence.close)
        node._event_log.worker.begin_stop(node._event_writer.close)


# ------------------------------------------------------------------- R12 / Q2


class _FlakyWriter(EventWriter):
    """A real writer that raises while ``broken`` is set, so the worker counts a failure."""

    def __init__(self, run_id: str, base_path: Path) -> None:
        super().__init__(run_id=run_id, base_path=base_path)
        self.broken = False

    def write(self, event: Event) -> None:
        if self.broken:
            raise OSError("disk full")
        super().write(event)


def _flaky_node(
    seeded: dict[str, Any], tmp_path: Path  # noqa: F811
) -> tuple[PaperTradingNode, _FlakyWriter]:
    node = _node(seeded, logs_path=tmp_path / "logs")
    writer = _FlakyWriter(run_id=node._config.trader_id, base_path=tmp_path / "logs")
    node._event_writer = writer
    return node, writer


def _submit_and_settle(node: PaperTradingNode, action: str) -> None:
    """Submit one event and wait for the worker to finish it, so ``health()`` is current."""
    node._write_event(EventType.SIGNAL, {"action": action})
    node._event_log.barrier().result(10)


def test_event_log_degraded_alert_once_per_streak(
    seeded: dict[str, Any],  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R12/Q2: one CRITICAL alert per outage, one recovery line + non-critical alert at
    the end, and a second critical alert only when a second streak opens."""
    node, writer = _flaky_node(seeded, tmp_path)
    alerts = _Alerts()
    monkeypatch.setattr(node, "_send_telegram_alert", alerts)
    trader = node._config.trader_id
    critical = (
        "error",
        f"paper {trader} event log not writing: 1 failed, 0 dropped (OSError); trading continues",
        True,
    )
    try:
        node._check_event_log()  # first tick only takes the baseline
        assert alerts.calls == []

        writer.broken = True
        _submit_and_settle(node, "a")
        node._check_event_log()
        assert alerts.calls == [critical]

        # Three more failures inside the SAME streak: one alert; unchanged ticks are silent.
        for action in ("b", "c", "d"):
            _submit_and_settle(node, action)
            node._check_event_log()
            node._check_event_log()  # counters unchanged since the tick above
        assert alerts.calls == [critical]
        health = node._event_log.health()
        assert (health.failed, health.dropped, health.pending) == (4, 0, 0)

        # The disk writes again: exactly one recovery line and one non-critical alert.
        writer.broken = False
        _submit_and_settle(node, "ok")
        node._check_event_log()
        node._event_log.barrier().result(10)
        assert alerts.calls == [
            critical,
            ("error", f"paper {trader} event log writing again (0 dropped, 4 failed)", False),
        ]
        recovered = [
            e
            for e in _lines(tmp_path / "logs" / f"{trader}.jsonl")
            if e["data"].get("action") == "event_log_recovered"
        ]
        assert [(e["event"], e["data"]) for e in recovered] == [
            (
                "RISK_CHECK",
                {"action": "event_log_recovered", "failed": 4, "dropped": 0, "passed": True},
            )
        ]

        # A fresh outage opens a fresh streak with its own critical alert.
        writer.broken = True
        _submit_and_settle(node, "e")
        node._check_event_log()
        assert alerts.calls[-1] == critical
        assert len(alerts.calls) == 3
    finally:
        _stop_event_worker(node)


def test_event_log_drops_only_alert_says_queue_full(
    seeded: dict[str, Any],  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R12/Q2: a full queue with no failed write reads "(queue full)", not a class name."""
    node = _node(seeded, logs_path=tmp_path / "logs")
    writer = _GateWriter()
    node._event_writer = writer  # type: ignore[assignment]
    node._event_log = EventLogWorker(_EVENT_THREAD, max_pending=1)
    alerts = _Alerts()
    monkeypatch.setattr(node, "_send_telegram_alert", alerts)
    try:
        node._check_event_log()  # baseline
        node._write_event(EventType.SIGNAL, {"action": "first"})
        assert writer.entered.wait(5), "the first write is in flight, so pending is 1"
        node._write_event(EventType.SIGNAL, {"action": "second"})  # refused, counted as dropped
        assert node._event_log.health().dropped == 1

        node._check_event_log()
        assert alerts.calls == [
            (
                "error",
                f"paper {node._config.trader_id} event log not writing: "
                "0 failed, 1 dropped (queue full); trading continues",
                True,
            )
        ]
    finally:
        writer.release.set()
        _stop_event_worker(node)


class _RaisingEventLog:
    """A stand-in event log whose counter read always raises (carries a leak marker)."""

    def health(self) -> Any:
        raise RuntimeError("boom ZZLEAK9")


async def test_raising_event_log_health_does_not_break_the_control_loop(
    seeded: dict[str, Any],  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R12/Q2: a raising counter read is logged by class only and the loop survives it."""
    node = _node(seeded, control_poll_interval=0.01)
    node._event_log = _RaisingEventLog()  # type: ignore[assignment]
    with caplog.at_level(logging.ERROR):
        loop_task = asyncio.create_task(node._control_loop())
        try:
            await _until(
                lambda: "paper control loop iteration failed (RuntimeError)" in caplog.text
            )
            assert not loop_task.done(), "the loop survived the raising health()"
        finally:
            node._shutdown_event.set()
            await loop_task
    assert "ZZLEAK9" not in caplog.text, "class-only log must not print the message"
