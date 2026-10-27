"""A recovery alert reaches Telegram after its own failure alert (bead vibe-quant-wi1m0a).

TelegramBot keeps ONE ERROR alert per 60 s (``RATE_LIMIT_SECONDS``); ``critical=True``
is the only bypass. Each recovery below is sent straight after the failure alert of the
SAME episode, so a non-critical recovery is dropped and the operator is left on
"venue disconnected" / "liquidation detector degraded" / "event log not writing".

Pinned here on a real ``TelegramBot`` with mocked HTTP (``httpx.MockTransport``), the
pattern of ``test_paper_command_channel.test_restored_alert_reaches_telegram_after_the_failed_alert``:

* venue connection lost -> restored (``_check_connectivity``);
* liquidation detector degraded -> recovered (the real ``LiquidationWatcher`` loop that
  owns the one-per-episode latch, driven through the node's own handlers);
* event log not writing -> writing again (``SessionJournal._check_event_log``).

Every pair is also driven over repeated ticks: the recovery is delivered once per
failure episode, never on every tick.

Both sides of every pair also survive an unrelated ERROR inside the last 60 s, and a
flapping monitor is bounded: each monitor direction bypasses the ERROR limit at most once
per 60 s (``SessionJournal.alert``), so a 30-cycle flap cannot starve a HALT behind
Telegram's own 429. An alert held back by that window is the pair's latest state and goes
out on the first tick after the window reopens, unless the chat already shows it, so the
chat always ends on each pair's current state.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest

from tests.unit.test_audit_paper_node import DSL
from tests.unit.test_paper_journal import _FakeEventLog, _FakeWriter
from tests.unit.test_paper_startup_and_estimate import (
    _connect_venue,
    _FakeTradingNode,
    _initialized,
)
from vibe_quant.alerts.telegram import TelegramBot, TelegramConfig
from vibe_quant.db.state_manager import StateManager
from vibe_quant.paper import journal as journal_module
from vibe_quant.paper.journal import Alert, SessionJournal

if TYPE_CHECKING:
    from pathlib import Path

    from vibe_quant.logging.writer import EventWriter
    from vibe_quant.paper.event_log import EventLogWorker
    from vibe_quant.paper.liquidation_watch import LiquidationWatcher

_TRADER_ID = "PAPER-007"

#: ErrorHandler's first transient alert (node.py ``_on_error_alert``).
_LOST = (
    "transient_error: venue connection lost (data or execution client disconnected)"
    "\nOp: connectivity"
)
_RESTORED = "venue connection restored"


@pytest.fixture()
def seeded(tmp_path: Path) -> dict[str, Any]:
    db = tmp_path / "state.db"
    sm = StateManager(db)
    sid = sm.create_strategy("test_strategy", DSL)
    rid = sm.create_backtest_run(
        strategy_id=sid,
        run_mode="validation",
        symbols=["BTCUSDT"],
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-12-31",
        parameters={"rsi_14_period": 21, "take_profit_risk_reward": 3.0, "leverage": 5},
    )
    sm.update_backtest_run_status(rid, "completed")
    sm.close()
    return {"db": db, "sid": sid, "rid": rid, "tmp": tmp_path}


@pytest.fixture()
def fake_tradingnode(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTradingNode]:
    import nautilus_trader.live.node as live_node

    _FakeTradingNode.instances = []
    monkeypatch.setattr(live_node, "TradingNode", _FakeTradingNode)
    return _FakeTradingNode


class _Sent:
    """A real TelegramBot whose HTTP goes to an in-memory recorder."""

    def __init__(self) -> None:
        self.bodies: list[str] = []

    def bot(self) -> TelegramBot:
        def record(request: httpx.Request) -> httpx.Response:
            # Each text is "<header · time>\n<alert text>".
            self.bodies.append(json.loads(request.content)["text"].split("\n", 1)[1])
            return httpx.Response(200, json={"ok": True})

        bot = TelegramBot(TelegramConfig(bot_token="T0K", chat_id="42"))

        async def client() -> httpx.AsyncClient:
            return httpx.AsyncClient(transport=httpx.MockTransport(record))

        bot._get_client = client  # type: ignore[method-assign]
        return bot


async def test_venue_restored_reaches_telegram_after_the_lost_alert(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """lost then restored both delivered within the 60 s ERROR window; restored once."""
    sent = _Sent()
    bot = sent.bot()
    node = await _initialized(seeded)
    node._journal.telegram = bot
    try:
        _connect_venue(fake_tradingnode)
        node._check_connectivity()
        assert node._connected is True
        assert sent.bodies == []
        # The pair starts on a clean 60 s window: the failure below is the first ERROR.
        bot._last_alert_times.clear()

        _connect_venue(fake_tradingnode, connected=False)
        node._check_connectivity()
        _connect_venue(fake_tradingnode)
        node._check_connectivity()
        # Repeated ticks on a healthy venue never repeat the recovery.
        node._check_connectivity()
        node._check_connectivity()
        await node._journal.wait_alerts(None)

        assert sent.bodies == [_LOST, _RESTORED]
    finally:
        await node._shutdown()


class _Clock:
    """Injected ``now_ms``: only the test moves time (the watcher's backoff is real)."""

    def __init__(self, ms: int = 1_700_000_000_000) -> None:
        self.ms = ms

    def __call__(self) -> int:
        return self.ms

    def advance(self, seconds: int) -> None:
        self.ms += seconds * 1_000


class _Fetch:
    """Scripted forceOrders fetch: ``error`` raises for that symbol, else no rows."""

    def __init__(self) -> None:
        self.error: Exception | None = None

    async def __call__(self, symbol: str, start_ms: int, end_ms: int) -> list[Any]:
        if self.error is not None:
            raise self.error
        return []


async def test_liquidation_recovered_reaches_telegram_after_the_degraded_alert(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """degraded then recovered both delivered; recovery once per failure episode."""
    sent = _Sent()
    bot = sent.bot()
    node = await _initialized(seeded)
    node._journal.telegram = bot
    clock = _Clock()
    monkeypatch.setattr(node, "_now_ms", clock)
    fetch = _Fetch()
    try:
        watcher: LiquidationWatcher = node._build_liquidation_watcher()
        watcher._fetch = fetch  # type: ignore[method-assign]
        bot._last_alert_times.clear()

        # Three failures reach degraded_after=3, each after its own backoff.
        fetch.error = RuntimeError("forceOrders blew up")
        await watcher.poll_once()
        clock.advance(60)
        await watcher.poll_once()
        clock.advance(120)
        await watcher.poll_once()
        await node._journal.wait_alerts(None)
        assert sent.bodies == ["WARNING: liquidation detector degraded: BTCUSDT RuntimeError"]

        fetch.error = None
        clock.advance(240)
        await watcher.poll_once()
        clock.advance(60)
        await watcher.poll_once()
        clock.advance(60)
        await watcher.poll_once()
        await node._journal.wait_alerts(None)

        assert sent.bodies == [
            "WARNING: liquidation detector degraded: BTCUSDT RuntimeError",
            "liquidation detector recovered: BTCUSDT after 3 failures",
        ]
    finally:
        await node._shutdown()


async def test_event_log_recovered_reaches_telegram_after_the_outage_alert() -> None:
    """not-writing then writing-again both delivered; recovery once per episode."""
    sent = _Sent()
    bot = sent.bot()
    event_log = _FakeEventLog([])
    journal = SessionJournal(_TRADER_ID, cast("EventLogWorker", event_log))
    journal.attach(cast("EventWriter", _FakeWriter([])), bot, "test_strategy")

    journal.check_health()  # only records the baseline
    event_log.failed = 5
    event_log.last_error_class = "OSError"
    journal.check_health()
    event_log.written = 1
    journal.check_health()
    journal.check_health()  # unchanged counters: silent in both directions
    await journal.wait_alerts(None)

    assert sent.bodies == [
        f"paper {_TRADER_ID} event log not writing: 5 failed, 0 dropped (OSError); "
        "trading continues",
        f"paper {_TRADER_ID} event log writing again (0 dropped, 5 failed)",
    ]


# ------------------------------------------- failure side, bounded bypass (wi1m0)

_UNRELATED = "recent unrelated ERROR"
_DEGRADED = "WARNING: liquidation detector degraded: BTCUSDT RuntimeError"
_RECOVERED = "liquidation detector recovered: BTCUSDT after 3 failures"
_RECOVERED_5 = "liquidation detector recovered: BTCUSDT after 5 failures"
_RECOVERED_4 = "liquidation detector recovered: BTCUSDT after 4 failures"
_TIMEOUT_CONNECTED = f"paper {_TRADER_ID} venue connected after startup timeout"
_LOG_FAILED = (
    f"paper {_TRADER_ID} event log not writing: 5 failed, 0 dropped (OSError); trading continues"
)
_LOG_AGAIN = f"paper {_TRADER_ID} event log writing again (0 dropped, 5 failed)"


async def _warm(bot: TelegramBot) -> None:
    """An unrelated ERROR just went out: the 60 s ERROR window is now closed."""
    assert await bot.send_error(_UNRELATED, bypass_rate_limit=True)


def _age(journal: SessionJournal, bot: TelegramBot, seconds: float) -> None:
    """Move the ERROR window and every monitor bypass ``seconds`` into the past."""
    bypassed = journal._monitor_bypass_at
    for key in list(bypassed):
        bypassed[key] -= seconds
    for kind in list(bot._last_alert_times):
        bot._last_alert_times[kind] -= seconds


async def test_venue_failure_and_recovery_survive_a_recent_error(
    seeded: dict[str, Any], fake_tradingnode: type[_FakeTradingNode]
) -> None:
    """lost + restored after an unrelated ERROR; repeat ticks silent; a 2nd episode too."""
    sent = _Sent()
    bot = sent.bot()
    node = await _initialized(seeded)
    node._journal.telegram = bot
    try:
        _connect_venue(fake_tradingnode)
        node._check_connectivity()
        await _warm(bot)
        for _ in range(2):
            _connect_venue(fake_tradingnode, connected=False)
            node._check_connectivity()
            node._check_connectivity()  # still down: the latch keeps it one alert
            _connect_venue(fake_tradingnode)
            node._check_connectivity()
            node._check_connectivity()
            await node._journal.wait_alerts(None)
            _age(node._journal, bot, 61)
        assert sent.bodies == [_UNRELATED, _LOST, _RESTORED, _LOST, _RESTORED]
    finally:
        await node._shutdown()


async def test_startup_timeout_then_connected_reaches_telegram(
    seeded: dict[str, Any], fake_tradingnode: type[_FakeTradingNode]
) -> None:
    """The late-connect notice follows the timeout's own critical ERROR, and a recent one."""
    sent = _Sent()
    bot = sent.bot()
    node = await _initialized(seeded)
    node._journal.telegram = bot
    node.status.started_at = datetime.now(UTC) - timedelta(seconds=121)
    try:
        await _warm(bot)
        node._check_connectivity()  # startup timeout: critical
        await node._journal.wait_alerts(None)
        timeout = node._startup_timeout_message()
        assert sent.bodies == [_UNRELATED, timeout]
        _connect_venue(fake_tradingnode)
        node._check_connectivity()
        node._check_connectivity()
        await node._journal.wait_alerts(None)
        assert sent.bodies == [_UNRELATED, timeout, _TIMEOUT_CONNECTED]
    finally:
        await node._shutdown()


async def test_liquidation_degraded_and_recovered_survive_a_recent_error(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """degraded after an unrelated ERROR (4th/5th failures add nothing), recovered, 2nd episode."""
    sent = _Sent()
    bot = sent.bot()
    node = await _initialized(seeded)
    node._journal.telegram = bot
    clock = _Clock()
    monkeypatch.setattr(node, "_now_ms", clock)
    fetch = _Fetch()
    try:
        watcher: LiquidationWatcher = node._build_liquidation_watcher()
        watcher._fetch = fetch  # type: ignore[method-assign]
        await _warm(bot)
        for _ in range(2):
            fetch.error = RuntimeError("forceOrders blew up")
            for wait in (0, 60, 120, 240, 480):
                clock.advance(wait)
                await watcher.poll_once()
            await node._journal.wait_alerts(None)
            fetch.error = None
            clock.advance(960)
            await watcher.poll_once()
            await node._journal.wait_alerts(None)
            _age(node._journal, bot, 61)
        assert sent.bodies == [_UNRELATED, _DEGRADED, _RECOVERED_5, _DEGRADED, _RECOVERED_4]
    finally:
        await node._shutdown()


async def test_liquidation_clamp_warning_stays_under_the_limiter(
    seeded: dict[str, Any], fake_tradingnode: type[_FakeTradingNode]
) -> None:
    """Only the degraded notice bypasses; a lookback-clamp warning is still rate limited."""
    sent = _Sent()
    bot = sent.bot()
    node = await _initialized(seeded)
    node._journal.telegram = bot
    try:
        await _warm(bot)
        await node._on_liquidation_warning("liquidation lookback clamped: BTCUSDT")
        await node._journal.wait_alerts(None)
        assert sent.bodies == [_UNRELATED]
    finally:
        await node._shutdown()


async def test_event_log_outage_and_recovery_survive_a_recent_error() -> None:
    sent = _Sent()
    bot = sent.bot()
    event_log = _FakeEventLog([])
    journal = SessionJournal(_TRADER_ID, cast("EventLogWorker", event_log))
    journal.attach(cast("EventWriter", _FakeWriter([])), bot, "test_strategy")
    await _warm(bot)
    journal.check_health()
    for _ in range(2):
        event_log.failed += 5
        event_log.last_error_class = "OSError"
        journal.check_health()
        event_log.written += 1
        journal.check_health()
        journal.check_health()
        await journal.wait_alerts(None)
        _age(journal, bot, 61)
        event_log.failed = 0  # the fake's counters restart; the next streak is 5 again
        journal.check_health()
    assert sent.bodies == [_UNRELATED, _LOG_FAILED, _LOG_AGAIN, _LOG_FAILED, _LOG_AGAIN]


# ------------------------------------------------ the bypass bound (C2) and HALT (C3)


class _Telegram429:
    """A real TelegramBot on a 20-messages-per-60-s chat: the 21st send is a 429."""

    def __init__(self, limit: int = 20) -> None:
        self.limit = limit
        self.accepted: list[str] = []
        self.attempts: list[tuple[int, str]] = []

    def bot(self) -> TelegramBot:
        def record(request: httpx.Request) -> httpx.Response:
            text = json.loads(request.content)["text"].split("\n", 1)[1]
            if len(self.accepted) >= self.limit:
                self.attempts.append((429, text))
                return httpx.Response(
                    429,
                    json={
                        "ok": False,
                        "error_code": 429,
                        "description": "Too Many Requests",
                        "parameters": {"retry_after": 0.01},
                    },
                )
            self.accepted.append(text)
            self.attempts.append((200, text))
            return httpx.Response(200, json={"ok": True})

        bot = TelegramBot(TelegramConfig(bot_token="T0K", chat_id="42"))

        async def client() -> httpx.AsyncClient:
            return httpx.AsyncClient(transport=httpx.MockTransport(record))

        bot._get_client = client  # type: ignore[method-assign]
        return bot


def _halt_attempts(chat: _Telegram429) -> list[int]:
    return [code for code, text in chat.attempts if "HALT" in text]


async def test_venue_flap_never_starves_a_halt(
    seeded: dict[str, Any], fake_tradingnode: type[_FakeTradingNode]
) -> None:
    """30 lost/restored cycles on a 20/min chat: <= 2 restored, HALT delivered first try."""
    chat = _Telegram429()
    node = await _initialized(seeded)
    node._journal.telegram = chat.bot()
    try:
        _connect_venue(fake_tradingnode)
        node._check_connectivity()
        for _ in range(30):
            _connect_venue(fake_tradingnode, connected=False)
            node._check_connectivity()
            _connect_venue(fake_tradingnode)
            node._check_connectivity()
        node._journal.alert(Alert("circuit_breaker", "HALT (max_drawdown): probe", True))
        await node._journal.wait_alerts(None)
        assert _halt_attempts(chat) == [200]
        assert sum(text == _RESTORED for text in chat.accepted) <= 2
        assert sum(text == _LOST for text in chat.accepted) <= 2
    finally:
        await node._shutdown()


async def test_event_log_flap_never_starves_a_halt() -> None:
    chat = _Telegram429()
    event_log = _FakeEventLog([])
    journal = SessionJournal(_TRADER_ID, cast("EventLogWorker", event_log))
    journal.attach(cast("EventWriter", _FakeWriter([])), chat.bot(), "test_strategy")
    journal.check_health()
    for _ in range(30):
        event_log.failed += 5
        event_log.last_error_class = "OSError"
        journal.check_health()
        event_log.written += 1
        journal.check_health()
    journal.alert(Alert("circuit_breaker", "HALT (max_drawdown): probe", True))
    await journal.wait_alerts(None)
    assert _halt_attempts(chat) == [200]
    assert sum("writing again" in text for text in chat.accepted) <= 2
    assert sum("not writing" in text for text in chat.accepted) <= 2


class _Recorder:
    """send_error stand-in that records each call's bypass flag."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    async def send_error(self, message: str, *, bypass_rate_limit: bool = False) -> bool:
        self.calls.append((message, bypass_rate_limit))
        return True


async def test_monitor_bypass_window_edges_and_independent_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same key: bypass again only at >= 60.0 s; another key and non-monitor alerts unaffected."""
    now = [1_000.0]
    monkeypatch.setattr(journal_module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    recorder = _Recorder()
    journal = SessionJournal(_TRADER_ID, cast("EventLogWorker", _FakeEventLog([])))
    journal.attach(cast("EventWriter", _FakeWriter([])), cast("TelegramBot", recorder), "s")

    def send(text: str, *, monitor: str | None, critical: bool = True) -> None:
        journal.alert(Alert("error", text, critical, monitor))

    send("a1", monitor="m:a")
    now[0] += 59.999
    send("a2", monitor="m:a")  # inside the window: held, not sent
    send("b1", monitor="m:b")  # another key: its own window
    send("plain", monitor=None)  # a critical non-monitor alert never rides the bound
    now[0] += 0.001  # exactly 60.0 s after a1
    send("a3", monitor="m:a")
    send("quiet", monitor="m:a", critical=False)  # non-critical never claims
    await journal.wait_alerts(None)
    assert recorder.calls == [
        ("a1", True),
        ("b1", True),
        ("plain", True),
        ("a3", True),
        ("quiet", False),
    ]


# ------------------------------------------- the chat ends on the CURRENT state (C5)


def _clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The journal's monotonic clock, moved only by the test."""
    now = [1_000.0]
    monkeypatch.setattr(journal_module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    return now


async def test_venue_chat_ends_on_lost_after_a_quick_refailure(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """lost, restored, lost inside 60 s: the held "lost" goes out at 60.0 s, once."""
    now = _clock(monkeypatch)
    sent = _Sent()
    node = await _initialized(seeded)
    node._journal.telegram = sent.bot()
    try:
        _connect_venue(fake_tradingnode)
        node._check_connectivity()
        for connected in (False, True, False):
            _connect_venue(fake_tradingnode, connected=connected)
            node._check_connectivity()
            now[0] += 0.5
        now[0] = 1_059.999  # "lost" bypassed at 1000.0: its window is still closed
        node._check_connectivity()
        await node._journal.wait_alerts(None)
        assert sent.bodies == [_LOST, _RESTORED]
        now[0] = 1_060.0
        node._check_connectivity()
        node._check_connectivity()
        now[0] += 120
        node._check_connectivity()
        await node._journal.wait_alerts(None)
        assert node._connected is False
        assert sent.bodies == [_LOST, _RESTORED, _LOST]
    finally:
        await node._shutdown()


async def test_venue_held_state_equal_to_the_chat_is_never_resent(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """lost, restored, lost, restored inside 60 s: the chat already says restored."""
    now = _clock(monkeypatch)
    sent = _Sent()
    node = await _initialized(seeded)
    node._journal.telegram = sent.bot()
    try:
        _connect_venue(fake_tradingnode)
        node._check_connectivity()
        for connected in (False, True, False, True):
            _connect_venue(fake_tradingnode, connected=connected)
            node._check_connectivity()
            now[0] += 0.5
        now[0] += 120
        node._check_connectivity()
        node._journal.check_health()
        await node._journal.wait_alerts(None)
        assert sent.bodies == [_LOST, _RESTORED]
    finally:
        await node._shutdown()


async def test_event_log_chat_ends_on_not_writing_after_a_quick_refailure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _clock(monkeypatch)
    sent = _Sent()
    event_log = _FakeEventLog([])
    journal = SessionJournal(_TRADER_ID, cast("EventLogWorker", event_log))
    journal.attach(cast("EventWriter", _FakeWriter([])), sent.bot(), "test_strategy")
    journal.check_health()
    event_log.failed += 5
    event_log.last_error_class = "OSError"
    journal.check_health()
    event_log.written += 1
    journal.check_health()
    event_log.failed += 5
    journal.check_health()  # inside the window: held
    for _ in range(3):
        now[0] += 0.5
        event_log.failed += 1
        journal.check_health()  # still failing: the latch keeps it one alert
    now[0] = 1_059.999
    journal.check_health()
    await journal.wait_alerts(None)
    assert sent.bodies == [_LOG_FAILED, _LOG_AGAIN]
    now[0] = 1_060.0
    journal.check_health()
    journal.check_health()
    await journal.wait_alerts(None)
    assert sent.bodies == [_LOG_FAILED, _LOG_AGAIN, _LOG_FAILED]


async def test_liquidation_chat_ends_on_degraded_after_a_quick_refailure(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _clock(monkeypatch)
    sent = _Sent()
    node = await _initialized(seeded)
    node._journal.telegram = sent.bot()
    try:
        await node._on_liquidation_warning("liquidation detector degraded: BTCUSDT RuntimeError")
        await node._on_liquidation_recovered("BTCUSDT", 3)
        await node._on_liquidation_warning("liquidation detector degraded: BTCUSDT RuntimeError")
        # The control loop's connectivity tick flushes every pair, venue up or not.
        now[0] = 1_059.999
        node._check_connectivity()
        await node._journal.wait_alerts(None)
        assert sent.bodies == [_DEGRADED, _RECOVERED]
        now[0] = 1_060.0
        node._check_connectivity()
        node._journal.check_health()
        await node._journal.wait_alerts(None)
        assert sent.bodies == [_DEGRADED, _RECOVERED, _DEGRADED]
    finally:
        await node._shutdown()


async def test_venue_flap_ending_down_adds_one_catch_up_and_never_starves_a_halt(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """30 flaps then down on a 20/min chat: lost, restored, one catch-up lost; HALT first try."""
    now = _clock(monkeypatch)
    chat = _Telegram429()
    node = await _initialized(seeded)
    node._journal.telegram = chat.bot()
    try:
        _connect_venue(fake_tradingnode)
        node._check_connectivity()
        for _ in range(30):
            for connected in (False, True):
                _connect_venue(fake_tradingnode, connected=connected)
                node._check_connectivity()
                now[0] += 0.1
        _connect_venue(fake_tradingnode, connected=False)
        node._check_connectivity()
        for _ in range(5):
            now[0] += 30
            node._check_connectivity()
        node._journal.alert(Alert("circuit_breaker", "HALT (max_drawdown): probe", True))
        await node._journal.wait_alerts(None)
        assert _halt_attempts(chat) == [200]
        assert chat.accepted[:3] == [_LOST, _RESTORED, _LOST]
        assert len(chat.accepted) == 4
    finally:
        await node._shutdown()


async def test_held_monitor_alert_window_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    """A held alert goes out at exactly 60.0 s after its key's bypass, once, critical."""
    now = _clock(monkeypatch)
    recorder = _Recorder()
    journal = SessionJournal(_TRADER_ID, cast("EventLogWorker", _FakeEventLog([])))
    journal.attach(cast("EventWriter", _FakeWriter([])), cast("TelegramBot", recorder), "s")
    journal.alert(Alert("error", "a1", True, "m:a"))
    journal.alert(Alert("error", "b1", True, "m:b"))
    now[0] += 1
    journal.alert(Alert("error", "a2", True, "m:a"))  # held: the pair's latest state
    now[0] = 1_059.999
    journal.check_health()
    await journal.wait_alerts(None)
    assert recorder.calls == [("a1", True), ("b1", True)]
    now[0] = 1_060.0
    journal.check_health()
    journal.check_health()
    await journal.wait_alerts(None)
    assert recorder.calls == [("a1", True), ("b1", True), ("a2", True)]


async def test_shutdown_sends_a_held_failure_state(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No tick follows shutdown: a held "lost" goes out then, inside its window."""
    now = _clock(monkeypatch)
    sent = _Sent()
    node = await _initialized(seeded)
    node._journal.telegram = sent.bot()
    _connect_venue(fake_tradingnode)
    node._check_connectivity()
    for connected in (False, True, False):
        _connect_venue(fake_tradingnode, connected=connected)
        node._check_connectivity()
        now[0] += 0.5
    await node._shutdown()
    assert sent.bodies == [_LOST, _RESTORED, _LOST]


async def test_shutdown_never_resends_the_state_the_chat_shows(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _clock(monkeypatch)
    sent = _Sent()
    node = await _initialized(seeded)
    node._journal.telegram = sent.bot()
    _connect_venue(fake_tradingnode)
    node._check_connectivity()
    for connected in (False, True, False, True):
        _connect_venue(fake_tradingnode, connected=connected)
        node._check_connectivity()
        now[0] += 0.5
    await node._shutdown()
    assert sent.bodies == [_LOST, _RESTORED]
