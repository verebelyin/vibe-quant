"""Paper session journal: event writes, Telegram alerts, event-log health, teardown.

T12 of bead vibe-quant-rabmo.11b moved every body here from ``vibe_quant.paper.node``,
so the journal writes the same lines and sends the same alerts as the node did. T13d
moved the last callers over: the node holds ``self._journal`` and makes no write or
alert of its own, and every composer sends through :meth:`SessionJournal.alert` (the
single funnel). The rabmo.11 contract is behaviour-preserving: no log line, event line
or alert text may change.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from vibe_quant.alerts.telegram import RATE_LIMIT_SECONDS, AlertType, TelegramBot
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

#: wi1m0: a monitor alert (``Alert.monitor``) bypasses the ERROR limit at most once per
#: this many seconds per monitor key; a flapping monitor is then held back instead of
#: flooding the chat and starving a HALT behind Telegram's own 429.
MONITOR_BYPASS_SECONDS = float(RATE_LIMIT_SECONDS)

#: 084l4: a critical alert that exhausted its Telegram attempts is owed and retried on the
#: control tick, at most once per this many seconds ...
FAILED_ALERT_RETRY_S = 30.0
#: ... until it is older than this (then it is dropped with a log line; the owed current
#: HALT/RESUMED/kill state never expires) ...
FAILED_ALERT_MAX_AGE_S = 3_600.0
#: ... and at most this many are owed at once, in flight included (the oldest alert that
#: is not the guard state is dropped first).
FAILED_ALERT_QUEUE_MAX = 10
#: Shutdown waits this long for the in-flight alerts, then this long again for the one
#: final retry of everything still owed; what remains is logged as abandoned.
SHUTDOWN_ALERT_WAIT_S = 10.0

#: Every critical ``circuit_breaker`` alert (HALT, RESUMED, kill) reports the guard's state;
#: a newer one replaces the older, so the chat always ends on the current state.
_GUARD_KEY = "state:guard"


@dataclass
class _Owed:
    """One critical alert still owed to the chat after its send exhausted its attempts."""

    kind: str
    message: str
    since: float  # monotonic time of the first failure; the age bound counts from here
    key: str  # what a newer send replaces: the guard state, a monitor pair or the same text
    seq: int  # the send number of ``key`` this alert had
    protected: bool = False  # the guard state: kept on overflow, exempt from the age bound
    live: bool = True  # False once a newer send of ``key`` (or a drop) made it moot


@dataclass(frozen=True)
class Alert:
    """One Telegram alert: the node's ``(kind, message, critical)`` triple.

    ``monitor`` names one direction of a safety monitor's transition as
    ``"<pair>:<state>"`` (``"venue:lost"``, ``"venue:restored"``,
    ``"liquidation:degraded"`` ...). A monitor alert is built ``critical=True`` and
    :meth:`SessionJournal.alert` sends it only when the same key last bypassed the
    limiter at least ``MONITOR_BYPASS_SECONDS`` ago; otherwise it is held as the pair's
    latest state until that window reopens.
    """

    kind: str  # "circuit_breaker" | "error"
    message: str
    critical: bool = False
    monitor: str | None = None


class SessionJournal:
    """Everything one paper session writes or alerts, behind one object.

    Attributes:
        has_writer: Whether ``attach`` installed an event writer.
        writer: The installed event writer, if any.
        strategy_name: The strategy name stamped on event lines.
        event_log: The worker event writes are submitted through.
        telegram: The bot ``attach`` installed, if any.
        alert_tasks: The live set of in-flight alert tasks.
        pending_alerts: Alert tasks still in flight.
    """

    def __init__(self, trader_id: str, event_log: EventLogWorker) -> None:
        self._trader_id = trader_id
        self._event_log = event_log
        self._writer: EventWriter | None = None
        self._telegram: TelegramBot | None = None
        self._strategy_name = "unknown"
        self._alert_tasks: set[asyncio.Task[bool]] = set()
        # wi1m0: monotonic time of each monitor key's last limiter bypass; per pair, the
        # state last sent and the latest alert held back by its key's window (C5).
        self._monitor_bypass_at: dict[str, float] = {}
        self._monitor_sent: dict[str, str] = {}
        self._monitor_held: dict[str, tuple[Alert, str]] = {}
        # 084l4: the newest send number per critical key (a newer send makes every older
        # one of its key moot), the owed alerts (insertion order = oldest first), the one
        # being retried, the earliest next retry, the retry task, and the critical sends
        # still running (a retry waits for them: a fresh alert goes first).
        self._send_seq = 0
        self._latest: dict[str, int] = {}
        self._owed: dict[str, _Owed] = {}
        self._inflight: _Owed | None = None
        self._owed_next_at = 0.0
        self._owed_task: asyncio.Task[bool] | None = None
        self._fresh_critical: dict[asyncio.Task[bool], tuple[str, str]] = {}  # (kind, key)
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
    def writer(self) -> EventWriter | None:
        """The installed event writer, if any."""
        return self._writer

    @writer.setter
    def writer(self, value: EventWriter | None) -> None:
        """Replace ONE sink, keeping the bot and the strategy name (R5)."""
        self._writer = value

    @property
    def strategy_name(self) -> str:
        """The strategy name stamped on every event line ("unknown" before it loads)."""
        return self._strategy_name

    @strategy_name.setter
    def strategy_name(self, value: str) -> None:
        self._strategy_name = value

    @property
    def event_log(self) -> EventLogWorker:
        """The event-log worker every write is submitted through."""
        return self._event_log

    @event_log.setter
    def event_log(self, value: EventLogWorker) -> None:
        self._event_log = value

    @property
    def telegram(self) -> TelegramBot | None:
        """The installed bot, if any."""
        return self._telegram

    @telegram.setter
    def telegram(self, value: TelegramBot | None) -> None:
        """Replace ONE sink, keeping the writer and the strategy name (R5)."""
        self._telegram = value

    @property
    def alert_tasks(self) -> set[asyncio.Task[bool]]:
        """The LIVE set of in-flight alert tasks (callers inspect and mutate it)."""
        return self._alert_tasks

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
        """Send one alert (``_send_telegram_alert``). Fire-and-forget, never raises.

        A critical monitor alert goes out at most once per ``MONITOR_BYPASS_SECONDS`` per
        key (wi1m0); inside that window it is held (:meth:`_send_monitor_alert`).
        """
        if alert.critical and alert.monitor is not None:
            self._send_monitor_alert(alert, alert.monitor)
            return
        self._send_telegram_alert(alert.kind, alert.message, critical=alert.critical)

    def _send_monitor_alert(self, alert: Alert, monitor: str) -> None:
        """Send critically if ``monitor``'s window is open, else hold it as the pair's latest.

        C5: a held alert replaces any older one of its pair; :meth:`flush_monitors`
        sends it once its key's window reopens. It is never routed through the limiter:
        the key's own bypass reset the ERROR window under 60 s ago, so it would drop it.
        """
        pair, _, state = monitor.partition(":")
        if not self._claim_monitor_bypass(monitor):
            self._monitor_held[pair] = (alert, monitor)
            return
        self._monitor_held.pop(pair, None)
        self._dispatch_monitor(alert, pair, state)

    def _dispatch_monitor(self, alert: Alert, pair: str, state: str) -> None:
        """Send ``alert`` as ``pair``'s newest state; any owed older one is superseded (084l4)."""
        self._monitor_sent[pair] = state
        self._send_telegram_alert(alert.kind, alert.message, critical=True, key=f"monitor:{pair}")

    def flush_monitors(self, *, final: bool = False) -> None:
        """Send each held monitor alert whose window reopened (one per pair, C5).

        Run on the control tick. A held state equal to the one the chat already shows is
        dropped: the operator's chat ends on every monitor pair's current state.
        ``final`` (shutdown: no tick follows) sends every other held state now.
        """
        for pair, (alert, monitor) in list(self._monitor_held.items()):
            state = monitor.partition(":")[2]
            if state == self._monitor_sent.get(pair):
                del self._monitor_held[pair]
            elif final:
                del self._monitor_held[pair]
                self._dispatch_monitor(alert, pair, state)
            else:
                self._send_monitor_alert(alert, monitor)

    def _claim_monitor_bypass(self, monitor: str) -> bool:
        """True (and the window restarts) if ``monitor`` has not bypassed for 60 s."""
        now = time.monotonic()
        last = self._monitor_bypass_at.get(monitor)
        if last is not None and now - last < MONITOR_BYPASS_SECONDS:
            return False
        self._monitor_bypass_at[monitor] = now
        return True

    def trade_alert(self, build: Callable[[], str], *, delay: float, enabled: bool) -> None:
        """Queue an entry/exit alert; ``enabled`` is the node's trade-alert setting."""
        self._send_trade_alert(build, delay=delay, enabled=enabled)

    def check_health(self) -> None:
        """One control tick: the event-log check, the held monitor alerts, owed retries."""
        self._check_event_log()
        self.flush_monitors()
        self._retry_owed()

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

    def _send_telegram_alert(
        self,
        kind: str,
        message: str,
        *,
        critical: bool = False,
        key: str | None = None,
    ) -> None:
        """Send alert via Telegram if configured. Fire-and-forget, never raises.

        ``critical`` alerts (halts, kill switch, risk breaches) bypass the
        per-type rate limit so they are never silently dropped, and one that fails on
        every attempt is owed and retried later (084l4). ``key`` names what a newer send
        replaces: a monitor pair, the guard state (any critical ``circuit_breaker``) or,
        by default, the same text. A replaced send stops retrying and is never owed.
        """
        if self._telegram is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("telegram alert not sent (no running event loop): %s", kind)
            return
        tg = self._telegram
        if not critical:
            if kind == "circuit_breaker":
                plain = tg.send_circuit_breaker(message, bypass_rate_limit=False)
            else:
                plain = tg.send_error(message, bypass_rate_limit=False)
            self._track(loop.create_task(plain))
            return
        if key is None:
            key = _GUARD_KEY if kind == "circuit_breaker" else f"{kind}:{message}"
        self._send_seq += 1
        seq = self._send_seq
        self._latest[key] = seq
        self._retire(key)

        def current() -> bool:
            return self._latest.get(key) == seq

        task = loop.create_task(
            self._deliver_critical(
                self._start_send(tg, kind, message, current), kind, message, key, seq
            )
        )
        self._fresh_critical[task] = (kind, key)
        self._track(task)

    @staticmethod
    def _start_send(
        tg: TelegramBot, kind: str, message: str, current: Callable[[], bool]
    ) -> Awaitable[bool]:
        """The bot's critical send of ``message``; ``current`` ends its retries when False.

        A stand-in bot (tests) has no ``current``; it is called the old way.
        """
        if not isinstance(tg, TelegramBot):
            if kind == "circuit_breaker":
                return tg.send_circuit_breaker(message, bypass_rate_limit=True)
            return tg.send_error(message, bypass_rate_limit=True)
        if kind == "circuit_breaker":
            return tg.send_circuit_breaker(message, bypass_rate_limit=True, current=current)
        return tg.send_error(message, bypass_rate_limit=True, current=current)

    def _track(self, task: asyncio.Task[bool]) -> None:
        self._alert_tasks.add(task)
        task.add_done_callback(self._alert_done)

    async def _deliver_critical(
        self, send: Awaitable[bool], kind: str, message: str, key: str, seq: int
    ) -> bool:
        """Await a critical send; if it exhausted its attempts the alert is owed (084l4)."""
        sent = await send
        if not sent:
            self._owe(kind, message, key, seq)
        self._settle(key, seq)
        return sent

    def _retire(self, key: str) -> None:
        """A newer send of ``key`` went out: the owed (or retrying) older one is moot."""
        queued = self._owed.pop(key, None)
        if queued is not None:
            queued.live = False
        inflight = self._inflight
        if inflight is not None and inflight.key == key:
            inflight.live = False

    def _settle(self, key: str, seq: int) -> None:
        """Forget a same-text key's send number once nothing about it is pending."""
        if key.startswith(("monitor:", "state:")) or key in self._owed:
            return
        if self._latest.get(key) == seq:
            del self._latest[key]

    def _owe(self, kind: str, message: str, key: str, seq: int) -> None:
        """Queue a critical alert whose send failed on every attempt (bounded).

        A rejection (bad token/chat) is not owed: retrying cannot help. An alert already
        replaced by a newer send of its key is not owed either, so the chat never ends on
        a stale state (and a repeat that was delivered is never sent a second time).
        """
        tg = self._telegram
        if not isinstance(tg, TelegramBot) or not tg.take_exhausted(message):
            return
        if self._latest.get(key) != seq:
            return
        if not self._owed and self._inflight is None:
            self._owed_next_at = time.monotonic() + FAILED_ALERT_RETRY_S
        self._add_owed(
            _Owed(kind, message, time.monotonic(), key, seq, protected=key == _GUARD_KEY)
        )

    def _add_owed(self, owed: _Owed, *, front: bool = False) -> None:
        """Put ``owed`` in the queue (``front``: a failed retry keeps its place) and cap it."""
        if front:
            self._owed = {owed.key: owed, **self._owed}
        else:
            self._owed[owed.key] = owed
        inflight = self._inflight
        # The alert being retried counts: the bound holds for queue + retry together.
        while len(self._owed) + (inflight is not None and inflight.live) > FAILED_ALERT_QUEUE_MAX:
            candidates = ([inflight] if inflight is not None and inflight.live else []) + list(
                self._owed.values()
            )
            victim = next((o for o in candidates if not o.protected), None)
            if victim is None:
                logger.warning(
                    "owed alert queue over its bound of %d: only guard-state alerts left, "
                    "none dropped",
                    FAILED_ALERT_QUEUE_MAX,
                )
                return
            victim.live = False
            self._owed.pop(victim.key, None)
            logger.warning(
                "owed %s alert dropped: %d alerts already owed",
                victim.kind,
                FAILED_ALERT_QUEUE_MAX,
            )

    def _retry_owed(self, *, force: bool = False) -> None:
        """Start one retry round if alerts are owed and the interval passed (or ``force``).

        One round runs at a time, on its own task beside any newer alert: a new HALT is
        never queued behind it.
        """
        if not self._owed or self._telegram is None:
            return
        if self._owed_task is not None and not self._owed_task.done():
            return
        if not force and time.monotonic() < self._owed_next_at:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("owed telegram alerts not retried (no running event loop)")
            return
        task = loop.create_task(self._retry_owed_round())
        self._owed_task = task
        self._track(task)

    async def _retry_owed_round(self) -> bool:
        """Send the owed alerts (guard state first, then oldest first); stop at the first failure.

        Each is taken out of the queue before its send, so it cannot go out twice; one that
        failed again is put back at the front (keeping its age) unless a newer send of its key
        or the queue bound made it moot. A critical alert sent meanwhile goes first: the round
        waits for it before every owed send, so a backlog never spends the chat's budget ahead
        of a fresh HALT. A failure is logged by class only.
        """
        while self._owed:
            tg = self._telegram
            if not isinstance(tg, TelegramBot):
                return False
            await self._wait_fresh_critical()
            owed = self._take_next_owed()
            if owed is None:
                continue
            sent, retry_later = await self._try_owed(tg, owed)
            if sent:
                self._settle(owed.key, owed.seq)
                continue
            if retry_later and owed.live:
                self._add_owed(owed, front=True)
                self._owed_next_at = time.monotonic() + FAILED_ALERT_RETRY_S
            return False
        return True

    async def _wait_fresh_critical(self) -> None:
        """Return once no freshly dispatched critical alert is still being sent."""
        while True:
            pending = [task for task in self._fresh_critical if not task.done()]
            if not pending:
                return
            await asyncio.wait(pending)

    def _take_next_owed(self) -> _Owed | None:
        """Pop the next alert to retry; None when it aged out (the guard state never does)."""
        key = next((k for k, o in self._owed.items() if o.protected), None)
        if key is None:
            key = next(iter(self._owed), None)
        if key is None:
            return None
        owed = self._owed.pop(key)
        if not owed.protected and time.monotonic() - owed.since > FAILED_ALERT_MAX_AGE_S:
            owed.live = False
            logger.warning(
                "owed %s alert dropped: older than %.0f s", owed.kind, FAILED_ALERT_MAX_AGE_S
            )
            return None
        return owed

    async def _try_owed(self, tg: TelegramBot, owed: _Owed) -> tuple[bool, bool]:
        """One send of an owed alert: ``(sent, worth retrying later)``. Never raises."""
        self._inflight = owed
        try:
            if owed.kind == "circuit_breaker":
                sent = await tg.send_circuit_breaker(
                    owed.message, bypass_rate_limit=True, current=lambda: owed.live
                )
            else:
                sent = await tg.send_error(
                    owed.message, bypass_rate_limit=True, current=lambda: owed.live
                )
            return sent, not sent and owed.live and tg.take_exhausted(owed.message)
        except Exception as exc:
            logger.warning("owed %s alert retry failed (%s)", owed.kind, type(exc).__name__)
            return False, True
        finally:
            self._inflight = None

    def _alert_done(self, task: asyncio.Task[bool]) -> None:
        self._alert_tasks.discard(task)
        self._fresh_critical.pop(task, None)
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
            self.alert(alert)
            raise
        self.alert(alert)

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
            self.alert(alert)
            raise
        except Exception as exc:
            logger.error(
                "paper event write failed (%s); sending the alert anyway", type(exc).__name__
            )
            self.alert(alert)
            raise
        self.alert(alert)

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
        critical alert. Unchanged counters are silent in both directions.
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
            self.alert(
                Alert(
                    "error",
                    f"paper {self._trader_id} event log not writing: "
                    f"{failed - prev_failed} failed, {dropped - prev_dropped} dropped "
                    f"({error_class}); trading continues",
                    critical=True,
                    monitor="eventlog:failed",
                )
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
                # Critical: sent within 60 s of the outage's own critical alert, so the
                # limit would drop it and leave the operator on "not writing" (wi1m0).
                Alert(
                    "error",
                    f"paper {self._trader_id} event log writing again "
                    f"({recovered_dropped} dropped, {recovered_failed} failed)",
                    critical=True,
                    monitor="eventlog:recovered",
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

        # C5: at most one held state per monitor pair, queued behind any in-flight halt.
        self.flush_monitors(final=True)
        # Let in-flight alerts (e.g. the halt that caused shutdown) go out ...
        with contextlib.suppress(Exception):
            await self.wait_alerts(SHUTDOWN_ALERT_WAIT_S)
        # ... then no tick follows: everything still owed gets one last try (084l4).
        try:
            await self._final_owed_retry()
        except Exception as exc:
            logger.error("owed alert final retry failed (%s)", type(exc).__name__)
        if self._telegram is not None:
            with contextlib.suppress(Exception):
                await self._telegram.close()

    async def _final_owed_retry(self) -> None:
        """Shutdown: one bounded retry of every owed alert; log each one never delivered.

        084l4 (chief R3): an alert still being sent after the wait, one whose retry is cut
        off, and one that fails once more are logged as abandoned by type and a fixed label
        of its key (never its text: ``Paper node failed: {e}`` carries exception text), so
        the operator log says what the chat never got.
        """
        abandoned: list[tuple[str, str]] = []  # (kind, key)
        for task, sent in list(self._fresh_critical.items()):
            if not task.done():
                abandoned.append(sent)
                task.cancel()
        round_task = self._owed_task
        if round_task is not None and not round_task.done():
            inflight = self._inflight
            if inflight is not None and inflight.live:
                abandoned.append((inflight.kind, inflight.key))
            round_task.cancel()
        owed = sorted((o for o in self._owed.values() if o.live), key=lambda o: not o.protected)
        self._owed = {}
        tg = self._telegram
        undelivered = list(owed)
        if isinstance(tg, TelegramBot):

            async def flush() -> None:
                for item in owed:
                    sent, _ = await self._try_owed(tg, item)
                    if sent:
                        undelivered.remove(item)

            try:
                await asyncio.wait_for(flush(), SHUTDOWN_ALERT_WAIT_S)
            except TimeoutError:
                logger.warning("owed alert final retry cut off after %g s", SHUTDOWN_ALERT_WAIT_S)
        abandoned.extend((o.kind, o.key) for o in undelivered)
        for kind, key in abandoned:
            logger.error(
                "%s alert abandoned at shutdown, never delivered (%s)", kind, _abandoned_label(key)
            )


def _abandoned_label(key: str) -> str:
    """A fixed label for an alert key: ``guard state``, ``<pair> monitor`` or ``single alert``."""
    if key == _GUARD_KEY:
        return "guard state"
    if key.startswith("monitor:"):
        return f"{key.removeprefix('monitor:')} monitor"
    return "single alert"


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

    def _send_telegram_alert(
        self,
        kind: str,
        message: str,
        *,
        critical: bool = False,
        key: str | None = None,
    ) -> None:
        self.calls.append(("alert", kind, message, critical))
