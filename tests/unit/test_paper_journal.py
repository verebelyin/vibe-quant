"""Unit tests for the paper session journal (bead vibe-quant-rabmo.11b, T12).

``vibe_quant.paper.journal`` holds verbatim copies of the node's event-write and
alert code, so every rule pinned here is a rule the node already enforces: the write
is submitted before the alert, the alert goes out even when the submit raises and only
then the error propagates, the payload is deep-copied at submit, failures are logged
by class only, TRADE alerts bypass the rate limit, an event-log outage alerts once per
streak, and teardown closes the writer on its worker before draining alerts and
closing Telegram.

The node keeps its own copies until T13a-d migrate callers, so these tests drive the
journal directly with recording fakes (no node, no network, no threads).
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import pytest

from tests.unit.test_paper_halt_restore import _Trace
from vibe_quant.logging.events import EventType
from vibe_quant.paper import journal as journal_module
from vibe_quant.paper.event_log import EventLogHealth, EventWriteTimeoutError
from vibe_quant.paper.journal import Alert, FakeJournal, SessionJournal

if TYPE_CHECKING:
    from collections.abc import Callable

    from vibe_quant.alerts.telegram import AlertType, TelegramBot
    from vibe_quant.logging.writer import EventWriter
    from vibe_quant.paper.event_log import EventLogWorker
    from vibe_quant.paper.persistence import StoreWorker

_LOGGER = "vibe_quant.paper.node"
_TS = datetime(2026, 10, 22, 9, 0, tzinfo=UTC)


class _FakeWorker:
    """The store worker seam: ``begin_stop`` runs the close job like the real one."""

    def __init__(self, order: list[str]) -> None:
        self._order = order

    def begin_stop(self, close: Callable[[], None]) -> Future[None]:
        self._order.append("worker.begin_stop")
        close()
        future: Future[None] = Future()
        future.set_result(None)
        return future


class _FakeWriter:
    """Records ``flush``/``close`` in the shared teardown order."""

    def __init__(self, order: list[str]) -> None:
        self._order = order

    def flush(self) -> None:
        self._order.append("writer.flush")

    def close(self) -> None:
        self._order.append("writer.close")


class _FakeEventLog:
    """The ``EventLogWorker`` seam: counters, submits and the worker handoff."""

    def __init__(
        self,
        order: list[str],
        *,
        submit_error: BaseException | None = None,
        pending: bool = False,
    ) -> None:
        self.worker = _FakeWorker(order)
        self._order = order
        self.submitted: list[Any] = []
        self.futures: list[Future[None]] = []
        self._submit_error = submit_error
        self._pending = pending
        self.failed = 0
        self.dropped = 0
        self.written = 0
        self.last_error_class: str | None = None

    def submit(self, writer: object, event: Any) -> Future[None]:
        if self._submit_error is not None:
            raise self._submit_error
        self._order.append("submit")
        self.submitted.append(event)
        future: Future[None] = Future()
        if not self._pending:
            future.set_result(None)
        self.futures.append(future)
        return future

    def health(self) -> EventLogHealth:
        return EventLogHealth(
            written=self.written,
            failed=self.failed,
            dropped=self.dropped,
            pending=0,
            last_error_class=self.last_error_class,
        )


class _RecordingTelegram:
    """Records every send in the shared teardown order; can raise a marker."""

    def __init__(
        self,
        order: list[str],
        *,
        send_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.calls: list[tuple[str, object, str, bool]] = []
        self._order = order
        self._send_error = send_error
        self._close_error = close_error

    async def send_alert(
        self, alert_type: AlertType, message: str, *, bypass_rate_limit: bool = False
    ) -> bool:
        self.calls.append(("send_alert", alert_type.value, message, bypass_rate_limit))
        self._order.append("alert")
        return True

    async def send_error(self, message: str, *, bypass_rate_limit: bool = False) -> bool:
        if self._send_error is not None:
            raise self._send_error
        self.calls.append(("send_error", None, message, bypass_rate_limit))
        self._order.append("alert")
        return True

    async def send_circuit_breaker(
        self, reason: str, details: str = "", *, bypass_rate_limit: bool = False
    ) -> bool:
        message = f"{reason}\n{details}" if details else reason
        self.calls.append(("send_circuit_breaker", None, message, bypass_rate_limit))
        self._order.append("alert")
        return True

    async def close(self) -> None:
        self._order.append("telegram.close")
        if self._close_error is not None:
            raise self._close_error


@dataclass
class _Harness:
    journal: SessionJournal
    event_log: _FakeEventLog
    order: list[str]
    telegram: _RecordingTelegram | None


def _harness(
    *,
    writer: bool = True,
    telegram: bool = True,
    submit_error: BaseException | None = None,
    pending: bool = False,
    send_error: BaseException | None = None,
    close_error: BaseException | None = None,
    trader_id: str = "PAPER-007",
    strategy_name: str = "test_strategy",
) -> _Harness:
    order: list[str] = []
    event_log = _FakeEventLog(order, submit_error=submit_error, pending=pending)
    bot = (
        _RecordingTelegram(order, send_error=send_error, close_error=close_error)
        if telegram
        else None
    )
    journal = SessionJournal(trader_id, cast("EventLogWorker", event_log))
    journal.attach(
        cast("EventWriter", _FakeWriter(order)) if writer else None,
        cast("TelegramBot", bot),
        strategy_name,
    )
    return _Harness(journal=journal, event_log=event_log, order=order, telegram=bot)


# --------------------------------------------------------------- writes


async def test_record_writes_then_alerts() -> None:
    h = _harness()
    future = h.journal.record(
        EventType.SIGNAL, {"action": "halt"}, alert=Alert("error", "halted"), timestamp=_TS
    )

    assert future is None
    await h.journal.wait_alerts(None)
    assert h.order == ["submit", "alert"]
    assert h.telegram is not None
    assert h.telegram.calls == [("send_error", None, "halted", False)]
    assert h.event_log.submitted[0].timestamp == _TS
    assert h.event_log.submitted[0].data == {"action": "halt"}


async def test_record_without_alert_returns_the_submit_future() -> None:
    h = _harness()
    future = h.journal.record(EventType.SIGNAL, {"action": "x"})

    assert future is h.event_log.futures[0]
    await h.journal.wait_alerts(None)
    assert h.telegram is not None
    assert h.telegram.calls == []


async def test_record_passes_the_critical_flag_through() -> None:
    h = _harness()
    h.journal.record(
        EventType.RISK_CHECK,
        {"passed": False},
        alert=Alert("circuit_breaker", "breaker", critical=True),
    )
    await h.journal.wait_alerts(None)

    assert h.telegram is not None
    assert h.telegram.calls == [("send_circuit_breaker", None, "breaker", True)]


async def test_record_alerts_even_when_the_submit_raises_then_reraises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _Chained(ValueError):
        pass

    cause = RuntimeError("MARK2")
    error = _Chained("MARK1")
    error.__cause__ = cause
    h = _harness(submit_error=error)

    with caplog.at_level(logging.ERROR, logger=_LOGGER):
        with pytest.raises(_Chained):
            h.journal.record(
                EventType.SIGNAL, {"a": 1}, alert=Alert("error", "boom", critical=True)
            )
        await h.journal.wait_alerts(None)

    assert h.order == ["alert"]
    assert h.telegram is not None
    assert h.telegram.calls == [("send_error", None, "boom", True)]
    assert "paper event write failed (_Chained); sending the alert anyway" in caplog.text
    # Class only: the exception text (which can carry URLs/keys on a real failure) never
    # reaches the log, and neither does a chained exception's text.
    assert "MARK1" not in caplog.text
    assert "MARK2" not in caplog.text


async def test_record_without_writer_returns_none_and_still_alerts() -> None:
    h = _harness(writer=False)

    assert h.journal.has_writer is False
    assert h.journal.record(EventType.SIGNAL, {"a": 1}) is None
    assert h.event_log.submitted == []

    h.journal.record(EventType.SIGNAL, {"a": 2}, alert=Alert("error", "boom"))
    await h.journal.wait_alerts(None)
    assert h.event_log.submitted == []
    assert h.telegram is not None
    assert h.telegram.calls == [("send_error", None, "boom", False)]


async def test_record_deep_copies_the_payload_at_submit() -> None:
    h = _harness()
    first = {"nested": {"k": [1]}, "scalar": 1}
    second = {"list": [{"deep": True}]}

    h.journal.record(EventType.SIGNAL, first)
    first["nested"]["k"].append(2)
    h.journal.record(EventType.SIGNAL, second)
    second["list"][0]["deep"] = False

    assert h.event_log.submitted[0].data == {"nested": {"k": [1]}, "scalar": 1}
    assert h.event_log.submitted[1].data == {"list": [{"deep": True}]}


async def test_record_uses_typed_position_events_and_base_events() -> None:
    h = _harness(strategy_name="alpha")

    h.journal.record(
        EventType.POSITION_OPEN,
        {
            "position_id": "P-1",
            "symbol": "BTCUSDT-PERP.BINANCE",
            "side": "LONG",
            "quantity": 0.5,
            "entry_price": 100.0,
            "leverage": 5,
        },
    )
    h.journal.record(EventType.SIGNAL, {"action": "x"})

    assert [type(event).__name__ for event in h.event_log.submitted] == [
        "PositionOpenEvent",
        "Event",
    ]
    assert [event.run_id for event in h.event_log.submitted] == ["PAPER-007", "PAPER-007"]
    assert [event.strategy_name for event in h.event_log.submitted] == ["alpha", "alpha"]


async def test_attach_replaces_trader_and_strategy_metadata() -> None:
    h = _harness(trader_id="PAPER-001", strategy_name="alpha")
    h.journal.record(EventType.SIGNAL, {"a": 1})
    h.journal.attach(
        cast("EventWriter", _FakeWriter(h.order)),
        cast("TelegramBot", h.telegram),
        "beta",
    )
    h.journal.record(EventType.SIGNAL, {"a": 2})

    assert [(e.run_id, e.strategy_name) for e in h.event_log.submitted] == [
        ("PAPER-001", "alpha"),
        ("PAPER-001", "beta"),
    ]


async def test_has_writer_and_telegram_properties() -> None:
    h = _harness()
    assert h.journal.has_writer is True
    assert h.journal.telegram is h.telegram
    assert h.journal.pending_alerts == 0


# ---------------------------------------------------------- durable writes


async def test_record_durable_writes_then_alerts() -> None:
    h = _harness()

    await h.journal.record_durable(
        EventType.RISK_CHECK, {"action": "liquidation", "passed": False}, Alert("error", "liq")
    )
    await h.journal.wait_alerts(None)

    assert h.order == ["submit", "alert"]
    assert h.event_log.submitted[0].data == {"action": "liquidation", "passed": False}


async def test_record_durable_timeout_alerts_then_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(journal_module, "EVENT_WRITE_WAIT_S", 0.01)
    h = _harness(pending=True)

    with pytest.raises(EventWriteTimeoutError) as excinfo:
        await h.journal.record_durable(
            EventType.RISK_CHECK, {"p": 1}, Alert("error", "boom", critical=True)
        )

    assert str(excinfo.value) == "paper event write not confirmed within 0.01 s"
    await h.journal.wait_alerts(None)
    assert h.order == ["submit", "alert"]
    assert h.telegram is not None
    assert h.telegram.calls == [("send_error", None, "boom", True)]


async def test_record_durable_cancellation_alerts_once_and_propagates() -> None:
    h = _harness(pending=True)
    task = asyncio.create_task(
        h.journal.record_durable(
            EventType.RISK_CHECK, {"p": 1}, Alert("error", "boom", critical=True)
        )
    )
    while not h.event_log.submitted:
        await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    await h.journal.wait_alerts(None)
    assert h.order == ["submit", "alert"]
    assert h.telegram is not None
    assert h.telegram.calls == [("send_error", None, "boom", True)]


async def test_record_durable_completed_worker_error_alerts_once_then_reraises() -> None:
    h = _harness(pending=True)
    error = OSError("disk full")
    original_submit = h.event_log.submit

    def failing_submit(writer: object, event: Any) -> Future[None]:
        # The worker has already failed by the time the journal looks at the future.
        future = original_submit(writer, event)
        future.set_exception(error)
        return future

    h.event_log.submit = failing_submit  # type: ignore[method-assign]

    with pytest.raises(OSError) as excinfo:
        await h.journal.record_durable(
            EventType.RISK_CHECK, {"p": 1}, Alert("error", "liq", critical=True)
        )

    assert excinfo.value is error
    await h.journal.wait_alerts(None)
    assert h.telegram is not None
    assert h.telegram.calls == [("send_error", None, "liq", True)]


async def test_record_durable_pending_write_failing_later_alerts_once_then_reraises() -> None:
    h = _harness(pending=True)
    error = OSError("disk full")
    task = asyncio.create_task(
        h.journal.record_durable(
            EventType.RISK_CHECK, {"p": 1}, Alert("error", "liq", critical=True)
        )
    )
    while not h.event_log.submitted:
        await asyncio.sleep(0)
    await asyncio.sleep(0)

    # Still waiting on the disk: no alert yet, the loop is free.
    assert not task.done()
    assert h.telegram is not None
    assert h.telegram.calls == []

    h.event_log.futures[0].set_exception(error)
    with pytest.raises(OSError) as excinfo:
        await task

    assert excinfo.value is error
    await h.journal.wait_alerts(None)
    assert h.telegram.calls == [("send_error", None, "liq", True)]


async def test_record_durable_forwards_the_explicit_timestamps() -> None:
    h = _harness()
    first = datetime(2024, 2, 3, 4, 5, tzinfo=UTC)
    second = datetime(2026, 9, 10, 11, 12, tzinfo=UTC)

    await h.journal.record_durable(
        EventType.RISK_CHECK, {"n": 1}, Alert("error", "a"), timestamp=first
    )
    await h.journal.record_durable(
        EventType.RISK_CHECK, {"n": 2}, Alert("error", "b"), timestamp=second
    )

    assert [e.timestamp.isoformat() for e in h.event_log.submitted] == [
        "2024-02-03T04:05:00+00:00",
        "2026-09-10T11:12:00+00:00",
    ]


# ---------------------------------------------------------- alert funnel (R2)
# ``journal.alert`` is the single funnel: every composer sends through it, so a patch
# of that one method sees every alert the journal makes (including its own).


def _funnel(monkeypatch: pytest.MonkeyPatch, h: _Harness) -> list[Alert]:
    """Patch ``journal.alert`` with a collector and return it."""
    raised: list[Alert] = []
    monkeypatch.setattr(h.journal, "alert", raised.append)
    return raised


async def test_record_sends_its_alert_through_the_funnel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(a) ``record(..., alert=A)`` collects ``[A]`` once."""
    h = _harness()
    raised = _funnel(monkeypatch, h)

    h.journal.record(EventType.RISK_CHECK, {"a": 1}, alert=Alert("error", "m", critical=True))

    assert raised == [Alert("error", "m", True)]


async def test_record_sends_its_alert_through_the_funnel_when_the_write_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(b) a raising write still collects ``[A]`` once and re-raises the same object."""
    error = OSError("disk full")
    h = _harness(submit_error=error)
    raised = _funnel(monkeypatch, h)

    with pytest.raises(OSError) as caught:
        h.journal.record(EventType.SIGNAL, {"a": 1}, alert=Alert("error", "boom", critical=True))

    assert caught.value is error
    assert raised == [Alert("error", "boom", True)]


async def test_record_durable_sends_its_alert_through_the_funnel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(c) a confirmed durable write collects ``[A]`` once."""
    h = _harness()
    raised = _funnel(monkeypatch, h)

    await h.journal.record_durable(EventType.RISK_CHECK, {"p": 1}, Alert("error", "ok"))

    assert raised == [Alert("error", "ok", False)]


async def test_record_durable_sends_its_alert_through_the_funnel_on_a_write_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(c) a worker error collects ``[A]`` once and re-raises the same object."""
    error = OSError("disk full")
    h = _harness(submit_error=error)
    raised = _funnel(monkeypatch, h)

    with pytest.raises(OSError) as caught:
        await h.journal.record_durable(
            EventType.RISK_CHECK, {"p": 1}, Alert("error", "boom", critical=True)
        )

    assert caught.value is error
    assert raised == [Alert("error", "boom", True)]


async def test_record_durable_sends_its_alert_through_the_funnel_on_a_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(c) a timed-out durable write collects ``[A]`` once before raising."""
    monkeypatch.setattr(journal_module, "EVENT_WRITE_WAIT_S", 0.01)
    h = _harness(pending=True)
    raised = _funnel(monkeypatch, h)

    with pytest.raises(EventWriteTimeoutError):
        await h.journal.record_durable(
            EventType.RISK_CHECK, {"p": 1}, Alert("error", "boom", critical=True)
        )

    assert raised == [Alert("error", "boom", True)]


async def test_record_durable_sends_its_alert_through_the_funnel_on_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(c) a cancelled durable write collects ``[A]`` once and propagates."""
    h = _harness(pending=True)
    raised = _funnel(monkeypatch, h)

    task = asyncio.create_task(
        h.journal.record_durable(
            EventType.RISK_CHECK, {"p": 1}, Alert("error", "boom", critical=True)
        )
    )
    while not h.event_log.submitted:
        await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert raised == [Alert("error", "boom", True)]


async def test_check_health_sends_both_alerts_through_the_funnel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(d) the outage and the recovery alert are both collected, with their flags."""
    h = _harness()
    raised = _funnel(monkeypatch, h)

    h.journal.check_health()  # first tick only records the baseline
    assert raised == []

    h.event_log.failed = 1
    h.event_log.last_error_class = "OSError"
    h.journal.check_health()
    assert raised == [
        Alert(
            "error",
            "paper PAPER-007 event log not writing: 1 failed, 0 dropped (OSError); "
            "trading continues",
            True,
            "eventlog:failed",
        )
    ]

    h.event_log.failed = 3
    h.event_log.dropped = 1
    h.journal.check_health()  # same streak: silent
    assert len(raised) == 1

    h.event_log.written = 1
    h.journal.check_health()
    assert raised == [
        Alert(
            "error",
            "paper PAPER-007 event log not writing: 1 failed, 0 dropped (OSError); "
            "trading continues",
            True,
            "eventlog:failed",
        ),
        Alert(
            "error",
            "paper PAPER-007 event log writing again (1 dropped, 3 failed)",
            True,
            "eventlog:recovered",
        ),
    ]


# --------------------------------------------------------------- alerts


async def test_alert_routes_error_and_circuit_breaker() -> None:
    h = _harness()

    h.journal.alert(Alert("error", "plain"))
    h.journal.alert(Alert("circuit_breaker", "reason", critical=True))
    await h.journal.wait_alerts(None)

    assert h.telegram is not None
    assert h.telegram.calls == [
        ("send_error", None, "plain", False),
        ("send_circuit_breaker", None, "reason", True),
    ]


async def test_alert_without_telegram_does_nothing() -> None:
    h = _harness(telegram=False)

    h.journal.alert(Alert("error", "plain"))
    await h.journal.wait_alerts(None)

    assert h.journal.pending_alerts == 0
    assert h.order == []


def test_alert_without_a_running_loop_warns_by_class(caplog: pytest.LogCaptureFixture) -> None:
    h = _harness()

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        h.journal.alert(Alert("error", "plain"))

    assert h.journal.pending_alerts == 0
    assert "telegram alert not sent (no running event loop): error" in caplog.text


async def test_alert_task_failure_logs_the_class_only(caplog: pytest.LogCaptureFixture) -> None:
    send_error = RuntimeError("MARK2")
    send_error.__cause__ = ValueError("MARK4")
    h = _harness(send_error=send_error)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        h.journal.alert(Alert("error", "plain"))
        await h.journal.wait_alerts(None)

    assert "telegram alert task failed: RuntimeError" in caplog.text
    assert "MARK2" not in caplog.text
    assert "MARK4" not in caplog.text


async def test_wait_alerts_ignores_a_cancelled_alert_task(
    caplog: pytest.LogCaptureFixture,
) -> None:
    h = _harness()
    h.journal.alert(Alert("error", "plain"))
    task = next(iter(h.journal._alert_tasks))  # type: ignore[attr-defined]
    task.cancel()

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        await h.journal.wait_alerts(None)

    assert h.journal.pending_alerts == 0
    assert "telegram alert task failed" not in caplog.text


# ---------------------------------------------------------- trade alerts


async def test_trade_alert_bypasses_the_rate_limit() -> None:
    h = _harness()

    h.journal.trade_alert(lambda: "OPENED LONG", delay=0.0, enabled=True)
    assert h.journal.pending_alerts == 1
    await h.journal.wait_alerts(None)

    assert h.telegram is not None
    assert h.telegram.calls == [("send_alert", "TRADE", "OPENED LONG", True)]
    assert h.journal.pending_alerts == 0


async def test_trade_alert_disabled_or_no_telegram_schedules_nothing() -> None:
    disabled = _harness()
    disabled.journal.trade_alert(lambda: "T", delay=0.0, enabled=False)
    silent = _harness(telegram=False)
    silent.journal.trade_alert(lambda: "T", delay=0.0, enabled=True)

    assert disabled.journal.pending_alerts == 0
    assert disabled.order == []
    assert silent.journal.pending_alerts == 0


async def test_trade_alert_waits_the_entry_delay_before_building(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = _harness()
    real_sleep = asyncio.sleep
    release = asyncio.Event()
    slept: list[float] = []
    built: list[str] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        await release.wait()

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    def build() -> str:
        built.append("build")
        return "OPENED LONG sl=1 tp=2"

    h.journal.trade_alert(build, delay=0.25, enabled=True)
    for _ in range(5):
        await real_sleep(0)

    assert slept == [0.25]
    assert built == []
    assert h.telegram is not None
    assert h.telegram.calls == []

    release.set()
    await h.journal.wait_alerts(None)

    assert built == ["build"]
    assert h.telegram.calls == [("send_alert", "TRADE", "OPENED LONG sl=1 tp=2", True)]


def test_trade_alert_without_a_running_loop_warns_by_class(
    caplog: pytest.LogCaptureFixture,
) -> None:
    h = _harness()

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        h.journal.trade_alert(lambda: "T", delay=0.0, enabled=True)

    assert h.journal.pending_alerts == 0
    assert "trade alert not sent (RuntimeError)" in caplog.text


async def test_trade_alert_failure_logs_the_class_only(caplog: pytest.LogCaptureFixture) -> None:
    h = _harness()

    def build() -> str:
        raise RuntimeError("MARK3")

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        h.journal.trade_alert(build, delay=0.0, enabled=True)
        await h.journal.wait_alerts(None)

    assert "trade alert not sent (RuntimeError)" in caplog.text
    assert "MARK3" not in caplog.text
    assert h.telegram is not None
    assert h.telegram.calls == []


# --------------------------------------------------------------- health


async def test_check_health_alerts_once_per_streak_then_reports_recovery() -> None:
    h = _harness()

    h.journal.check_health()  # first tick only records the baseline
    await h.journal.wait_alerts(None)
    assert h.order == []

    h.event_log.failed = 1
    h.event_log.last_error_class = "OSError"
    h.journal.check_health()
    await h.journal.wait_alerts(None)
    assert h.telegram is not None
    assert h.telegram.calls == [
        (
            "send_error",
            None,
            "paper PAPER-007 event log not writing: 1 failed, 0 dropped (OSError); "
            "trading continues",
            True,
        )
    ]

    h.event_log.failed = 3
    h.event_log.dropped = 1
    h.journal.check_health()  # same streak: silent
    await h.journal.wait_alerts(None)
    assert len(h.telegram.calls) == 1
    assert h.event_log.submitted == []

    h.event_log.written = 1
    h.journal.check_health()
    await h.journal.wait_alerts(None)
    assert h.telegram.calls[-1] == (
        "send_error",
        None,
        "paper PAPER-007 event log writing again (1 dropped, 3 failed)",
        True,
    )
    assert h.event_log.submitted[0].data == {
        "action": "event_log_recovered",
        "failed": 3,
        "dropped": 1,
        "passed": True,
    }
    assert h.event_log.submitted[0].event_type is EventType.RISK_CHECK


async def test_check_health_reports_a_full_queue_without_a_write_failure() -> None:
    h = _harness()
    h.journal.check_health()

    h.event_log.dropped = 2
    h.journal.check_health()
    await h.journal.wait_alerts(None)

    assert h.telegram is not None
    assert h.telegram.calls == [
        (
            "send_error",
            None,
            "paper PAPER-007 event log not writing: 0 failed, 2 dropped (queue full); "
            "trading continues",
            True,
        )
    ]


async def test_check_health_unchanged_counters_are_silent() -> None:
    h = _harness()
    h.event_log.written = 4
    h.journal.check_health()
    h.journal.check_health()
    await h.journal.wait_alerts(None)

    assert h.order == []
    assert h.event_log.submitted == []


# -------------------------------------------------------------- teardown


async def test_close_orders_writer_alerts_then_telegram() -> None:
    h = _harness()
    seen: list[tuple[object, str]] = []

    async def close_store(worker: StoreWorker, close: Callable[[], None], label: str) -> None:
        seen.append((worker, label))
        h.order.append("close_store")
        worker.begin_stop(close)

    async def slow_alert() -> bool:
        await asyncio.sleep(0)
        h.order.append("alert.finished")
        return True

    task = asyncio.get_running_loop().create_task(slow_alert())
    h.journal._alert_tasks.add(task)  # type: ignore[attr-defined]
    # like production's _track: a finished alert leaves the set (else wait_alerts spins 10 s)
    task.add_done_callback(h.journal._alert_tasks.discard)  # type: ignore[attr-defined]

    await h.journal.close(close_store)

    assert seen == [(h.event_log.worker, "paper event writer")]
    assert h.order == [
        "close_store",
        "worker.begin_stop",
        "writer.flush",
        "writer.close",
        "alert.finished",
        "telegram.close",
    ]
    assert h.journal.has_writer is False


async def test_close_logs_a_raising_close_store_and_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    h = _harness()

    async def close_store(worker: StoreWorker, close: Callable[[], None], label: str) -> None:
        raise RuntimeError("MARK4")

    with caplog.at_level(logging.ERROR, logger=_LOGGER):
        await h.journal.close(close_store)

    assert "paper event writer shutdown failed (RuntimeError); continuing shutdown" in caplog.text
    assert "MARK4" not in caplog.text
    assert h.order == ["telegram.close"]


async def test_close_swallows_a_raising_telegram_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    h = _harness(close_error=RuntimeError("MARK5"))

    async def close_store(worker: StoreWorker, close: Callable[[], None], label: str) -> None:
        worker.begin_stop(close)

    with caplog.at_level(logging.ERROR, logger=_LOGGER):
        await h.journal.close(close_store)

    assert h.order == [
        "worker.begin_stop",
        "writer.flush",
        "writer.close",
        "telegram.close",
    ]
    assert "MARK5" not in caplog.text


# ----------------------------------------------------------- seam accessors


def test_writer_and_strategy_name_accessors() -> None:
    order: list[str] = []
    writer = cast("EventWriter", _FakeWriter(order))
    h = _harness(strategy_name="alpha")
    h.journal.attach(writer, cast("TelegramBot", h.telegram), "alpha")

    assert h.journal.writer is writer
    assert h.journal.strategy_name == "alpha"

    h.journal.strategy_name = "beta"
    assert h.journal.strategy_name == "beta"
    assert h.journal.writer is writer
    assert h.journal.telegram is h.telegram


@pytest.mark.parametrize("name", ["alpha_probe", "beta_probe"])
def test_sink_setters_change_one_sink_and_keep_the_others(name: str) -> None:
    """(R5) ``writer``/``telegram`` setters keep the other two journal fields."""
    h = _harness()
    bot = cast("TelegramBot", _RecordingTelegram(h.order))
    first_writer = cast("EventWriter", object())
    second_writer = cast("EventWriter", object())

    h.journal.strategy_name = name
    h.journal.writer = first_writer
    h.journal.telegram = bot
    assert h.journal.writer is first_writer
    assert h.journal.telegram is bot
    assert h.journal.strategy_name == name

    h.journal.writer = second_writer
    assert h.journal.writer is second_writer
    assert h.journal.telegram is bot
    assert h.journal.strategy_name == name

    h.journal.telegram = None
    assert h.journal.telegram is None
    assert h.journal.writer is second_writer
    assert h.journal.strategy_name == name

    h.journal.telegram = bot
    assert h.journal.strategy_name == name

    h.journal.writer = None
    assert h.journal.writer is None
    assert h.journal.telegram is bot
    assert h.journal.strategy_name == name


def test_event_log_accessor_rebinds_the_submit_worker() -> None:
    h = _harness()
    other = _FakeEventLog(h.order)

    h.journal.event_log = cast("EventLogWorker", other)
    h.journal.record(EventType.SIGNAL, {"a": 1})

    assert h.journal.event_log is other
    assert [event.data for event in other.submitted] == [{"a": 1}]
    assert h.event_log.submitted == []


async def test_alert_tasks_accessor_is_the_live_set() -> None:
    h = _harness()
    h.journal.alert(Alert("error", "plain"))

    assert h.journal.alert_tasks is h.journal._alert_tasks
    assert h.journal.pending_alerts == 1
    task = next(iter(h.journal.alert_tasks))

    await h.journal.wait_alerts(None)
    assert task not in h.journal.alert_tasks
    assert h.journal.alert_tasks == set()


# ----------------------------------------------------------- fake journal


async def test_fake_journal_calls_match_the_transcript_trace() -> None:
    trace = _Trace()
    fake = FakeJournal()

    trace.write(EventType.SIGNAL, {"a": 1}, _TS)
    fake.record(EventType.SIGNAL, {"a": 1}, timestamp=_TS)
    trace.write_then_alert(EventType.RISK_CHECK, {"b": 2}, "error", "boom", critical=True)
    fake.record(EventType.RISK_CHECK, {"b": 2}, alert=Alert("error", "boom", critical=True))
    trace.write_then_alert(EventType.RISK_CHECK, {"c": 3}, "error", "liq")
    await fake.record_durable(EventType.RISK_CHECK, {"c": 3}, Alert("error", "liq"))
    trace.alert("circuit_breaker", "reason", critical=True)
    fake.alert(Alert("circuit_breaker", "reason", critical=True))

    assert fake.calls == trace.calls
    assert fake.calls == [
        ("write", EventType.SIGNAL, {"a": 1}, _TS),
        ("write", EventType.RISK_CHECK, {"b": 2}, None),
        ("alert", "error", "boom", True),
        ("write", EventType.RISK_CHECK, {"c": 3}, None),
        ("alert", "error", "liq", False),
        ("alert", "circuit_breaker", "reason", True),
    ]
