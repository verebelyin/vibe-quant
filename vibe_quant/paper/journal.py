"""Paper session journal: event writes, Telegram alerts, event-log health, teardown.

T12 of bead vibe-quant-rabmo.11b: every body here is a copy of the code in
``vibe_quant.paper.node``, so the journal writes the same lines and sends the same
alerts as the node. ``node`` keeps its own copies until T13a-d move every caller, and
the rabmo.11 contract is behaviour-preserving: no log line, event line or alert text
may change.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from vibe_quant.alerts.telegram import AlertType, TelegramBot
from vibe_quant.logging.events import Event, EventType, create_event
from vibe_quant.paper.event_log import EventLogWorker, EventWriteTimeoutError
from vibe_quant.paper.persistence import StoreWorker, wait_future

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from concurrent.futures import Future

    from vibe_quant.logging.writer import EventWriter

# The node's logger name, not this module's (r2 F2): the moved lines are the node's
# lines, and an operator or a test that filters on ``vibe_quant.paper.node`` must keep
# seeing them.
logger = logging.getLogger("vibe_quant.paper.node")

#: ogd37: a durable report waits at most this long for its line to land.
EVENT_WRITE_WAIT_S = 5.0


@dataclass(frozen=True)
class Alert:
    """One Telegram alert: the node's ``(kind, message, critical)`` triple."""

    kind: str  # "circuit_breaker" | "error"
    message: str
    critical: bool = False


class SessionJournal:
    """Everything one paper session writes or alerts, behind one object.

    Attributes:
        has_writer: Whether ``attach`` installed an event writer.
        telegram: The bot ``attach`` installed, if any.
        pending_alerts: Alert tasks still in flight.
    """

    def __init__(self, trader_id: str, event_log: EventLogWorker) -> None:
        self._trader_id = trader_id
        self._event_log = event_log
        self._writer: EventWriter | None = None
        self._telegram: TelegramBot | None = None
        self._strategy_name = "unknown"
        self._alert_tasks: set[asyncio.Task[bool]] = set()
        # R12/Q2: the counters as of the previous control tick, and the (failed, dropped)
        # baseline of the outage streak now in progress (None while the log is healthy).
        self._event_log_seen: tuple[int, int, int] | None = None
        self._event_log_streak: tuple[int, int] | None = None

    def attach(
        self, writer: EventWriter | None, telegram: TelegramBot | None, strategy_name: str
    ) -> None:
        """Install the sinks a started session writes to (the node's wiring)."""
        self._writer = writer
        self._telegram = telegram
        self._strategy_name = strategy_name

    @property
    def has_writer(self) -> bool:
        """True when an event writer is installed (the liquidation report gate)."""
        return self._writer is not None

    @property
    def telegram(self) -> TelegramBot | None:
        """The installed bot, if any (read-only)."""
        return self._telegram

    @property
    def pending_alerts(self) -> int:
        """Number of fire-and-forget alert tasks still in flight."""
        return len(self._alert_tasks)

    # ---------------------------------------------------------------- record

    def record(
        self,
        event_type: EventType,
        data: dict[str, object],
        *,
        alert: Alert | None = None,
        timestamp: datetime | None = None,
    ) -> Future[None] | None:
        """Submit one event; with ``alert``, write it then alert (``_write_then_alert``)."""
        if alert is None:
            return self._write_event(event_type, data, timestamp=timestamp)
        self._write_then_alert(event_type, data, alert, timestamp=timestamp)
        return None

    async def record_durable(
        self,
        event_type: EventType,
        data: dict[str, object],
        alert: Alert,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """Write one event, wait for it (bounded), then alert (``_awrite_then_alert``)."""
        await self._awrite_then_alert(event_type, data, alert, timestamp=timestamp)

    def alert(self, alert: Alert) -> None:
        """Send one alert (``_send_telegram_alert``). Fire-and-forget, never raises."""
        self._send_telegram_alert(alert.kind, alert.message, critical=alert.critical)

    def trade_alert(self, build: Callable[[], str], *, delay: float, enabled: bool) -> None:
        """Queue an entry/exit alert; ``enabled`` is the node's trade-alert setting."""
        self._send_trade_alert(build, delay=delay, enabled=enabled)

    def check_health(self) -> None:
        """One control tick's event-log health check (``_check_event_log``)."""
        self._check_event_log()

    async def wait_alerts(self, timeout: float | None) -> None:
        """Wait until no alert task is left; ``None`` waits without a bound (tests)."""
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        while self._alert_tasks:
            remaining = None if deadline is None else deadline - loop.time()
            if remaining is not None and remaining <= 0:
                return
            await asyncio.wait(set(self._alert_tasks), timeout=remaining)

    # --------------------------------------------------------------- alerts

    def _send_telegram_alert(self, kind: str, message: str, *, critical: bool = False) -> None:
        """Send alert via Telegram if configured. Fire-and-forget, never raises.

        ``critical`` alerts (halts, kill switch, risk breaches) bypass the
        per-type rate limit so they are never silently dropped.
        """
        if self._telegram is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("telegram alert not sent (no running event loop): %s", kind)
            return
        tg = self._telegram
        if kind == "circuit_breaker":
            coro = tg.send_circuit_breaker(message, bypass_rate_limit=critical)
        else:
            coro = tg.send_error(message, bypass_rate_limit=critical)
        task = loop.create_task(coro)
        self._alert_tasks.add(task)
        task.add_done_callback(self._alert_done)

    def _alert_done(self, task: asyncio.Task[bool]) -> None:
        self._alert_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            # TelegramBot already redacts its own failures; never feed alert
            # failures back into ErrorHandler (that would alert about alerts).
            logger.warning("telegram alert task failed: %s", type(exc).__name__)

    # --------------------------------------------------------- trade alerts

    def _send_trade_alert(self, build: Callable[[], str], *, delay: float, enabled: bool) -> None:
        """Fire-and-forget entry/exit alert; the text is built inside the alert task.

        ci7q9, report-only: the text is built lazily so a raising cache lookup lands in
        the task's own handler, never on the NT fill callback (a raise there would also
        skip the guard's risk evaluation). An entry alert waits ``delay`` before
        building, because the compiled strategy submits its SL/TP orders only AFTER the
        entry fill. No Telegram / setting off -> nothing is scheduled.
        """
        telegram = self._telegram
        if not enabled or telegram is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            logger.warning("trade alert not sent (%s)", type(exc).__name__)
            return
        task = loop.create_task(self._trade_alert_task(telegram, build, delay))
        self._alert_tasks.add(task)
        task.add_done_callback(self._alert_done)

    async def _trade_alert_task(
        self, telegram: TelegramBot, build: Callable[[], str], delay: float
    ) -> bool:
        """Send one TRADE alert; every failure is swallowed with a class-only log.

        R31: the TRADE per-type rate limit is bypassed. OPEN and CLOSE share
        ``AlertType.TRADE`` with a 60 s limit, so a fast round trip would silently drop
        the CLOSE; the trade count is the natural bound and trade alerts are opt-in.
        """
        try:
            if delay:
                await asyncio.sleep(delay)
            return await telegram.send_alert(AlertType.TRADE, build(), bypass_rate_limit=True)
        except Exception as exc:
            # Class only, never the text: a chained cache/httpx error can carry secrets.
            logger.warning("trade alert not sent (%s)", type(exc).__name__)
            return False

    # ---------------------------------------------------------------- events

    def _write_then_alert(
        self,
        event_type: EventType,
        data: dict[str, object],
        alert: Alert,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """Submit the event, then alert, even when the submit itself raises (R1/irdig).

        The append runs on the event-log worker (ogd37), so the alert never waits for the
        disk: the line may land after the alert. A submit-time error (an unbuildable
        event, a stopped worker) still sends the alert with the same text and
        ``critical`` flag and only then propagates unchanged. Sites that must know the
        line landed use :meth:`record_durable`.
        """
        try:
            self._write_event(event_type, data, timestamp=timestamp)
        except Exception as exc:
            # R1b: class only. ``logger.exception`` would print the active exception's
            # ``__context__`` chain too (a fetch error with signature/apiKey text).
            logger.error(
                "paper event write failed (%s); sending the alert anyway", type(exc).__name__
            )
            self._send_telegram_alert(alert.kind, alert.message, critical=alert.critical)
            raise
        self._send_telegram_alert(alert.kind, alert.message, critical=alert.critical)

    async def _awrite_then_alert(
        self,
        event_type: EventType,
        data: dict[str, object],
        alert: Alert,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """Write the event, wait for it (bounded), then alert, even when the write fails.

        For the liquidation reports (R47b: a failed write fails the watcher's cycle so the
        row is re-delivered). A timeout raises ``EventWriteTimeoutError`` and a worker
        error is re-raised as is, each after the alert; a cancellation (shutdown) sends the
        alert once and propagates. The loop keeps running while it waits.
        """
        try:
            future = self._write_event(event_type, data, timestamp=timestamp)
            if future is not None:
                if not await wait_future(future, EVENT_WRITE_WAIT_S):
                    raise EventWriteTimeoutError(
                        f"paper event write not confirmed within {EVENT_WRITE_WAIT_S:g} s"
                    )
                future.result()
        except asyncio.CancelledError:
            self._send_telegram_alert(alert.kind, alert.message, critical=alert.critical)
            raise
        except Exception as exc:
            logger.error(
                "paper event write failed (%s); sending the alert anyway", type(exc).__name__
            )
            self._send_telegram_alert(alert.kind, alert.message, critical=alert.critical)
            raise
        self._send_telegram_alert(alert.kind, alert.message, critical=alert.critical)

    def _write_event(
        self,
        event_type: EventType,
        data: dict[str, object],
        timestamp: datetime | None = None,
    ) -> Future[None] | None:
        """Queue an event for the paper event log, preserving the full payload.

        The line is appended on the event-log worker (ogd37): the returned future
        resolves when it is on disk (or carries the write error / ``EventLogFullError``),
        ``None`` when there is no writer. The timestamp is taken here, at submit.

        The payload is deep-copied here, at submit: the worker serializes the event
        later, so a caller that mutates its dict after this returns (a new key, a new
        value, a nested list/dict) must not change the written line.

        Position events use the typed classes (their fields are the payload);
        everything else is written as a base ``Event`` because the typed
        SIGNAL/RISK_CHECK classes drop unknown keys (and default ``passed``
        to True, which would record a halt as a passed risk check).
        """
        if self._writer is None:
            return None
        strategy_name = self._strategy_name
        ts = timestamp or datetime.now(UTC)
        payload = copy.deepcopy(data)
        event: Event
        if event_type in (EventType.POSITION_OPEN, EventType.POSITION_CLOSE):
            event = create_event(
                event_type=event_type,
                run_id=self._trader_id,
                strategy_name=strategy_name,
                data=payload,
                timestamp=ts,
            )
        else:
            event = Event(
                timestamp=ts,
                event_type=event_type,
                run_id=self._trader_id,
                strategy_name=strategy_name,
                data=payload,
            )
        return self._event_log.submit(self._writer, event)

    # ---------------------------------------------------------------- health

    def _check_event_log(self) -> None:
        """Alert once per event-log outage, report once when it writes again (R12/Q2).

        ``health()`` is a lock-protected snapshot, so this never blocks the loop. An
        increase in ``failed + dropped`` since the previous tick opens a streak: one
        CRITICAL alert, with the counts accumulated since the streak began, is sent and
        no later rising tick repeats it (writing an event line here would be pointless -
        the log is exactly what is broken). The first tick that sees a new successful
        write and no new failure closes the streak: one ``RISK_CHECK``
        ``event_log_recovered`` line through the normal write path plus one
        non-critical alert. Unchanged counters are silent in both directions.
        """
        health = self._event_log.health()
        failed, dropped, written = health.failed, health.dropped, health.written
        previous = self._event_log_seen
        # Recorded before anything can raise: a failed alert must not re-fire next tick.
        self._event_log_seen = (failed, dropped, written)
        if previous is None:
            return
        prev_failed, prev_dropped, prev_written = previous
        if failed + dropped > prev_failed + prev_dropped:
            if self._event_log_streak is not None:
                return
            self._event_log_streak = (prev_failed, prev_dropped)
            error_class = health.last_error_class if failed > prev_failed else "queue full"
            self._send_telegram_alert(
                "error",
                f"paper {self._trader_id} event log not writing: "
                f"{failed - prev_failed} failed, {dropped - prev_dropped} dropped "
                f"({error_class}); trading continues",
                critical=True,
            )
            return
        if written > prev_written and self._event_log_streak is not None:
            base_failed, base_dropped = self._event_log_streak
            # Cleared before the write: if the recovery line itself fails to land, the
            # next failing tick opens a fresh streak rather than repeating this one.
            self._event_log_streak = None
            recovered_failed = failed - base_failed
            recovered_dropped = dropped - base_dropped
            self._write_then_alert(
                EventType.RISK_CHECK,
                {
                    "action": "event_log_recovered",
                    "failed": recovered_failed,
                    "dropped": recovered_dropped,
                    "passed": True,
                },
                Alert(
                    "error",
                    f"paper {self._trader_id} event log writing again "
                    f"({recovered_dropped} dropped, {recovered_failed} failed)",
                ),
            )

    # -------------------------------------------------------------- shutdown

    async def close(
        self,
        close_store: Callable[[StoreWorker, Callable[[], None], str], Awaitable[None]],
    ) -> None:
        """The node's shutdown sequence for the writer, the alerts and the bot.

        ogd37: later writes are no-ops; the event worker flushes and closes the file
        after its last queued line. R6 (k8oei): a close() failure is logged by class and
        the remaining teardown (alerts, telegram) still runs.
        """
        writer = self._writer
        self._writer = None

        def close_writer() -> None:
            if writer is None:
                return
            with contextlib.suppress(Exception):
                writer.flush()
            writer.close()

        try:
            await close_store(self._event_log.worker, close_writer, "paper event writer")
        except Exception as exc:
            logger.error(
                "paper event writer shutdown failed (%s); continuing shutdown",
                type(exc).__name__,
            )

        # Let in-flight alerts (e.g. the halt that caused shutdown) go out.
        with contextlib.suppress(Exception):
            await self.wait_alerts(10)
        if self._telegram is not None:
            with contextlib.suppress(Exception):
                await self._telegram.close()


class FakeJournal(SessionJournal):
    """Test double recording the transcript trace's exact tuples (r3 F12).

    ``.calls`` holds ``("write", event_type, data, timestamp)`` and
    ``("alert", kind, message, critical)`` entries in call order, the same tuples
    ``test_paper_halt_restore._Trace`` records. Writes are accepted unconditionally so
    a node's liquidation path is not gated by the fake.
    """

    def __init__(self) -> None:
        super().__init__("fake", EventLogWorker("vq-fake-journal"))
        self.calls: list[tuple[Any, ...]] = []

    @property
    def has_writer(self) -> bool:
        return True

    def _write_event(
        self,
        event_type: EventType,
        data: dict[str, object],
        timestamp: datetime | None = None,
    ) -> None:
        self.calls.append(("write", event_type, data, timestamp))
        return None

    def _send_telegram_alert(self, kind: str, message: str, *, critical: bool = False) -> None:
        self.calls.append(("alert", kind, message, critical))
