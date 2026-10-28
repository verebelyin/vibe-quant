"""Tests for the Telegram alert sender (ci7q9/21qug, vibe-quant-3ljxh, 084l4).

One row per refusal/retry branch: the sender is the operator's only page, so
the in-flight cap, the 429 handling and the retry budget keep their own rows.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import httpx
import pytest

from vibe_quant.alerts import telegram as telegram_mod
from vibe_quant.alerts.telegram import (
    HALF_MSG,
    RATE_LIMIT_SECONDS,
    AlertType,
    ConfigurationError,
    DailySummary,
    TelegramBot,
    TelegramConfig,
    TelegramUnavailableError,
)

if TYPE_CHECKING:
    from collections.abc import Callable


# ------------------------------------------------------------------ configuration


@pytest.mark.parametrize(
    ("env", "missing"),
    [
        ({}, "TELEGRAM_BOT_TOKEN"),
        ({"TELEGRAM_BOT_TOKEN": "123:ABC"}, "TELEGRAM_CHAT_ID"),
        ({"TELEGRAM_BOT_TOKEN": "123:ABC", "TELEGRAM_CHAT_ID": "456"}, None),
    ],
    ids=["no-token", "no-chat", "ok"],
)
def test_config_from_env(env: dict[str, str], missing: str | None) -> None:
    with patch.dict(os.environ, env, clear=True):
        if missing is not None:
            with pytest.raises(ConfigurationError, match=missing):
                TelegramConfig.from_env()
            return
        config = TelegramConfig.from_env()

    assert (config.bot_token, config.chat_id) == ("123:ABC", "456")


def test_configured_env_returns_config() -> None:
    env = {"TELEGRAM_BOT_TOKEN": "123:ABC", "TELEGRAM_CHAT_ID": "456"}
    with patch.dict(os.environ, env, clear=True):
        assert telegram_mod.telegram_config_or_none() == TelegramConfig(
            bot_token="123:ABC", chat_id="456"
        )


@pytest.mark.parametrize(
    "env",
    [{}, {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""}],
    ids=["unset", "empty"],
)
def test_unconfigured_env_is_none(env: dict[str, str]) -> None:
    with patch.dict(os.environ, env, clear=True):
        assert telegram_mod.telegram_config_or_none() is None


@pytest.mark.parametrize(
    ("env", "var"),
    [
        ({"TELEGRAM_BOT_TOKEN": "123:ABC"}, "TELEGRAM_CHAT_ID"),
        ({"TELEGRAM_CHAT_ID": "456"}, "TELEGRAM_BOT_TOKEN"),
    ],
    ids=["token-only", "chat-only"],
)
def test_half_configured_refuses_naming_the_missing_var(
    env: dict[str, str], var: str
) -> None:
    """vibe-quant-u2r9r: half-configured refuses, and names the var to fix."""
    with (
        patch.dict(os.environ, env, clear=True),
        pytest.raises(TelegramUnavailableError) as excinfo,
    ):
        telegram_mod.telegram_config_or_none()

    assert str(excinfo.value) == HALF_MSG.format(VAR=var)


def test_space_only_token_counts_as_set() -> None:
    """Only ``""`` is unset: a whitespace token is configured and must be probed."""
    env = {"TELEGRAM_BOT_TOKEN": " ", "TELEGRAM_CHAT_ID": "42"}
    with patch.dict(os.environ, env, clear=True):
        config = telegram_mod.telegram_config_or_none()

    assert config is not None and config.bot_token == " "


# ---------------------------------------------------------------- message bodies


def test_daily_summary_formats_both_pnl_signs() -> None:
    positive = DailySummary(
        date=datetime(2024, 1, 15, tzinfo=UTC),
        total_pnl=Decimal("150.50"),
        realized_pnl=Decimal("100.00"),
        unrealized_pnl=Decimal("50.50"),
        open_positions=2,
        trades_executed=5,
        win_rate=0.60,
        max_drawdown=0.025,
        total_fees=Decimal("1.25"),
    ).format_message()

    for expected in (
        "2024-01-15",
        "+150.50",
        "Realized: 100.00",
        "Unrealized: 50.50",
        "Positions: 2",
        "Trades: 5",
        "Win Rate: 60.0%",
        "Max DD: 2.50%",
        "Fees: 1.2500",
    ):
        assert expected in positive

    negative = DailySummary(
        date=datetime(2024, 1, 15, tzinfo=UTC), total_pnl=Decimal("-50.00")
    ).format_message()
    assert "-50.00" in negative and "+-50" not in negative


def test_format_message_prefix_clock_and_escaping() -> None:
    """Every alert type gets its own header; dynamic text is HTML-escaped."""
    bot = TelegramBot(TelegramConfig(bot_token="123:ABC", chat_id="456"))
    expected = {
        AlertType.ERROR: "⚠️ ERROR",
        AlertType.CIRCUIT_BREAKER: "🚨 CIRCUIT BREAKER",
        AlertType.TRADE: "💹 TRADE",
        AlertType.DAILY_SUMMARY: "📊 DAILY",
    }
    for alert_type, prefix in expected.items():
        header = bot._format_message(alert_type, "").split("\n")[0]
        assert header.startswith(f"{prefix} · ")
        assert header.endswith(" UTC")

    fallback = bot._format_message(cast("AlertType", "WAT"), "")
    assert fallback.startswith("🔔 ALERT · ")

    body = bot._format_message(AlertType.ERROR, "<b>&").split("\n")[1]
    assert body == "&lt;b&gt;&amp;"


# ------------------------------------------------------------------- rate limits


async def test_rate_limit_is_per_type_and_expires(
    bot: TelegramBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [1_000.0]
    monkeypatch.setattr(telegram_mod, "time", SimpleNamespace(time=lambda: clock[0]))
    posts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posts.append(json.loads(request.content)["text"])
        return _ok_response()

    bot._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    assert await bot.send_alert(AlertType.ERROR, "first") is True
    assert await bot.send_alert(AlertType.ERROR, "second") is False, "same type is throttled"
    assert await bot.send_alert(AlertType.TRADE, "trade") is True, "types are independent"

    clock[0] += RATE_LIMIT_SECONDS
    assert await bot.send_alert(AlertType.ERROR, "third") is True, "the window expires"

    assert len(posts) == 3


# ---------------------------------------------------------------------- sending


@pytest.fixture()
def bot() -> TelegramBot:
    return TelegramBot(TelegramConfig(bot_token="123:ABC", chat_id="456"))


def _ok_response() -> httpx.Response:
    return httpx.Response(200, json={"ok": True})


def _too_many_response(retry_after: float = 1) -> httpx.Response:
    """The real Telegram 429 body: ``parameters.retry_after`` is the server's delay."""
    return httpx.Response(
        429,
        json={
            "ok": False,
            "error_code": 429,
            "description": f"Too Many Requests: retry after {retry_after:g}",
            "parameters": {"retry_after": retry_after},
        },
    )


def _raw_429(retry_after_json: str) -> httpx.Response:
    """A 429 whose ``retry_after`` is spliced in as RAW JSON (``1e309``, ``10**400``)."""
    body = '{"ok": false, "error_code": 429, "parameters": {"retry_after": ' + retry_after_json + "}}"
    return httpx.Response(429, content=body.encode(), headers={"content-type": "application/json"})


async def test_send_alert_posts_to_send_message(bot: TelegramBot) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok_response()

    bot._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    assert await bot.send_alert(AlertType.ERROR, "Test error") is True
    assert len(seen) == 1
    assert seen[0].url.path.endswith("/sendMessage")
    payload = json.loads(seen[0].content)
    assert payload["chat_id"] == "456"
    assert "Test error" in payload["text"]


async def test_bypass_rate_limit_sends(
    bot: TelegramBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(telegram_mod.time, "time", lambda: 1_000.0)
    bot._update_rate_limit(AlertType.ERROR)  # 21qug: criticals pass bypass_rate_limit
    posts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posts.append(1)
        return _ok_response()

    bot._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    assert await bot.send_alert(AlertType.ERROR, "x", bypass_rate_limit=True) is True
    assert len(posts) == 1


def _status_503(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(503, json={"ok": False})


def _connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("Connection failed", request=request)


@pytest.mark.parametrize("handler", [_status_503, _connect_error], ids=["http-503", "transport"])
async def test_retryable_failure_retries_full_budget_then_gives_up(
    bot: TelegramBot, monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> None:
    """Retries 3 attempts, backing off 1 s then 2 s, then returns False."""
    delays: list[float] = []

    async def _record_sleep(delay: float) -> None:
        delays.append(delay)

    # Record (don't wait out) the retry backoff. Patch only the telegram
    # module's asyncio view so the global asyncio.sleep stays real.
    monkeypatch.setattr(telegram_mod, "asyncio", SimpleNamespace(sleep=_record_sleep))
    posts: list[int] = []

    def counting(request: httpx.Request) -> httpx.Response:
        posts.append(1)
        return handler(request)

    bot._client = httpx.AsyncClient(transport=httpx.MockTransport(counting))

    assert await bot.send_alert(AlertType.ERROR, "Test error") is False
    assert len(posts) == 3
    assert delays == [1.0, 2.0]


async def test_send_error_uses_the_error_type(bot: TelegramBot) -> None:
    texts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts.append(json.loads(request.content)["text"])
        return _ok_response()

    bot._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    await bot.send_error("Connection lost")
    assert "⚠️ ERROR" in texts[0] and "Connection lost" in texts[0]


@pytest.mark.parametrize("details", ["Hit 15% threshold", ""], ids=["details", "no-details"])
async def test_send_circuit_breaker_carries_reason_and_details(
    bot: TelegramBot, details: str
) -> None:
    texts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts.append(json.loads(request.content)["text"])
        return _ok_response()

    bot._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    await bot.send_circuit_breaker("Max drawdown", details)
    assert "🚨 CIRCUIT BREAKER" in texts[0] and "Max drawdown" in texts[0]
    assert (details in texts[0]) if details else texts[0].count("\n") == 1


async def test_send_trade_formats_the_trade(bot: TelegramBot) -> None:
    texts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts.append(json.loads(request.content)["text"])
        return _ok_response()

    bot._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    assert await bot.send_trade(
        symbol="BTCUSDT", side="BUY", quantity=Decimal("0.5"), price=Decimal("50000.00")
    )
    assert "💹 TRADE" in texts[0]
    assert "BUY 0.5 BTCUSDT @ 50000.00" in texts[0]

    # A non-positive quantity/price is refused before the wire.
    assert await bot.send_trade(
        symbol="BTCUSDT", side="BUY", quantity=Decimal("0"), price=Decimal("50000")
    ) is False
    assert len(texts) == 1


async def test_send_daily_summary_formats_the_day(bot: TelegramBot) -> None:
    texts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts.append(json.loads(request.content)["text"])
        return _ok_response()

    bot._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    await bot.send_daily_summary(
        DailySummary(date=datetime(2024, 1, 15, tzinfo=UTC), total_pnl=Decimal("100.00"))
    )
    assert "📊 DAILY" in texts[0] and "2024-01-15" in texts[0] and "+100.00" in texts[0]


async def test_close_releases_the_client(bot: TelegramBot) -> None:
    bot._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _r: _ok_response()))

    await bot.close()
    assert bot._client is None
    await bot.close()  # idempotent: a second close must not raise


# --------------------------------------- ci7q9/21qug: burst backpressure


class TestTradeBurstBackpressure:
    """A trade-alert burst must never cost a CIRCUIT_BREAKER alert (vibe-quant-21qug).

    Trade alerts bypass the per-type rate limit (R31), so a fast strategy can post
    dozens of messages and make Telegram answer 429 for the whole chat. The circuit
    breaker then burned ~3 s of retries and was dropped, and every failed send closed
    the shared httpx client under the other in-flight requests.
    """

    @pytest.mark.asyncio
    async def test_in_flight_trade_alerts_are_capped_and_counted(
        self, bot: TelegramBot, caplog: pytest.LogCaptureFixture
    ) -> None:
        """At most 5 TRADE sends are in flight; the rest are dropped and counted."""
        caplog.set_level(logging.WARNING)
        release = asyncio.Event()
        five_in_flight = asyncio.Event()
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if len(requests) == 5:
                five_in_flight.set()
            await release.wait()
            return _ok_response()

        bot._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            tasks = [
                asyncio.create_task(
                    bot.send_trade(
                        "BTCUSDT",
                        "BUY",
                        Decimal("0.5"),
                        Decimal("50000"),
                        bypass_rate_limit=True,
                    )
                )
                for _ in range(50)
            ]
            await asyncio.wait_for(five_in_flight.wait(), 1)
            for _ in range(3):
                await asyncio.sleep(0)  # let the droppers reach their return

            try:
                assert len(requests) == 5, "only 5 trade sends may be in flight"
                assert bot.trade_alerts_dropped == 45
                assert "trade alert dropped" in caplog.text
            finally:
                release.set()  # never leave the in-flight handlers parked
                results = await asyncio.gather(*tasks)

            assert results.count(True) == 5
            assert bot.trade_alerts_dropped == 45, "the drop count does not move on success"
        finally:
            await bot.close()

    @pytest.mark.asyncio
    async def test_circuit_breaker_is_delivered_through_the_trade_burst(
        self,
        bot: TelegramBot,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """50 trades + one breaker, chat answering 429 retry_after=1: the breaker arrives."""
        caplog.set_level(logging.WARNING)
        sleeps: list[float] = []

        async def _record_sleep(delay: float) -> None:
            sleeps.append(delay)

        monkeypatch.setattr(telegram_mod, "asyncio", SimpleNamespace(sleep=_record_sleep))

        delivered: list[str] = []
        breaker_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal breaker_calls
            text = request.content.decode()
            if "TRADE" in text:
                return _too_many_response()
            breaker_calls += 1
            if breaker_calls == 1:
                return _too_many_response()  # the breaker itself is caught in the 429
            delivered.append(text)
            return _ok_response()

        bot._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            breaker = asyncio.create_task(
                bot.send_circuit_breaker(
                    "Max drawdown", "Hit 15%", bypass_rate_limit=True
                )
            )
            trades = [
                asyncio.create_task(
                    bot.send_trade(
                        "BTCUSDT",
                        "BUY",
                        Decimal("0.5"),
                        Decimal("50000"),
                        bypass_rate_limit=True,
                    )
                )
                for _ in range(50)
            ]
            results = await asyncio.gather(breaker, *trades)
        finally:
            await bot.close()

        assert results[0] is True, "the CIRCUIT_BREAKER alert must be delivered"
        assert len(delivered) == 1
        assert "🚨 CIRCUIT BREAKER" in delivered[0]
        assert "Max drawdown" in delivered[0]
        assert sleeps == [1.0], "Telegram's retry_after is honoured, not 1s/2s backoff"
        assert results[1:].count(True) == 0, "no trade alert got through the 429 chat"
        assert bot.trade_alerts_dropped == 50
        assert "trade alert dropped" in caplog.text

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("handler", "posts"),
        [(lambda _r: httpx.Response(400, json={"ok": False}), 1), (_connect_error, 3)],
        ids=["rejected", "transport"],
    )
    async def test_failure_leaves_the_shared_client_open(
        self, bot: TelegramBot, monkeypatch: pytest.MonkeyPatch,
        handler: Callable[[httpx.Request], httpx.Response], posts: int,
    ) -> None:
        """A 400 (one shot) or a transport error (retried) must not close the client:
        peers are in flight on it, and closing it turns one hiccup into several losses."""

        async def _record_sleep(_delay: float) -> None:
            return None

        monkeypatch.setattr(telegram_mod, "asyncio", SimpleNamespace(sleep=_record_sleep))
        seen: list[int] = []

        def counting(request: httpx.Request) -> httpx.Response:
            seen.append(1)
            return handler(request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(counting))
        bot._client = client
        try:
            assert await bot.send_alert(AlertType.ERROR, "x") is False
            assert len(seen) == posts
            assert bot._client is client
            assert not client.is_closed
        finally:
            await bot.close()


# ------------------------- retry_after bounds, critical priority (rev round 1)


class _Sleeper:
    """Yielding ``asyncio.sleep`` stand-in: records delays, parks the ones in ``park``.

    The older fixtures' sleep never yields, so a second sender could never run in a
    sleeping sender's window; this one hands control back like the real loop does.
    """

    def __init__(self, park: tuple[float, ...] = ()) -> None:
        self.delays: list[float] = []
        self.parked = {d: asyncio.Event() for d in park}
        self.gates = {d: asyncio.Event() for d in park}

    @property
    def gate(self) -> asyncio.Event:
        """The (single) parked delay's gate, for tests that park one delay."""
        return next(iter(self.gates.values()))

    @property
    def parked_event(self) -> asyncio.Event:
        return next(iter(self.parked.values()))

    def open_all(self) -> None:
        for gate in self.gates.values():
            gate.set()

    async def sleep(self, delay: float) -> None:
        self.delays.append(delay)
        if delay in self.gates:
            self.parked[delay].set()
            await self.gates[delay].wait()
        else:
            await asyncio.sleep(0)


def _install_sleeper(monkeypatch: pytest.MonkeyPatch, sleeper: _Sleeper) -> None:
    monkeypatch.setattr(telegram_mod, "asyncio", SimpleNamespace(sleep=sleeper.sleep))


class TestCriticalRetryAfterAndPriority:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("retry_after_json", "expected_delay"),
        [
            ("7", 7.0),  # nondefault: kills "use 1.0 instead of retry_after"
            ("0", 0.0),  # lower edge, allowed
            ("60", 60.0),  # cap edge, allowed
            ("60.5", 1.0),  # just over the cap -> default backoff
            ("-1", 1.0),  # negative -> default backoff
            ("1e309", 1.0),  # JSON -> float inf
            ("1" + "0" * 400, 1.0),  # int too large for float() -> OverflowError
            ("NaN", 1.0),  # JSON NaN passes `< 0` / `> 60`; only isfinite rejects it
        ],
        ids=["7s", "zero", "cap", "over-cap", "negative", "inf", "overflow", "nan"],
    )
    async def test_breaker_429_delay(
        self,
        bot: TelegramBot,
        monkeypatch: pytest.MonkeyPatch,
        retry_after_json: str,
        expected_delay: float,
    ) -> None:
        """The breaker is delivered after Telegram's bounded delay, never lost to a crash."""
        sleeper = _Sleeper()
        _install_sleeper(monkeypatch, sleeper)
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return _raw_429(retry_after_json) if calls == 1 else _ok_response()

        bot._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            assert await bot.send_circuit_breaker("DD", bypass_rate_limit=True) is True
        finally:
            await bot.close()
        assert sleeper.delays == [expected_delay]

    @pytest.mark.asyncio
    async def test_trade_429_is_dropped_without_waiting(
        self, bot: TelegramBot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A trade alert is never worth waiting out the chat's 429 window."""
        _install_sleeper(monkeypatch, _Sleeper())
        posts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal posts
            posts += 1
            return _raw_429("1e309")

        bot._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            sent = await bot.send_trade(
                "BTCUSDT", "BUY", Decimal("1"), Decimal("50000"), bypass_rate_limit=True
            )
        finally:
            await bot.close()
        assert sent is False
        assert posts == 1
        assert bot.trade_alerts_dropped == 1

    @pytest.mark.asyncio
    async def test_trade_arriving_during_breaker_backoff_is_dropped(
        self, bot: TelegramBot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Breaker parked in its retry_after wait: a new trade never reaches the wire."""
        sleeper = _Sleeper(park=(7.0,))
        _install_sleeper(monkeypatch, sleeper)
        trade_posts = 0
        breaker_posts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal trade_posts, breaker_posts
            if "TRADE" in request.content.decode():
                trade_posts += 1
                return _ok_response()
            breaker_posts += 1
            return _raw_429("7") if breaker_posts == 1 else _ok_response()

        bot._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            breaker = asyncio.create_task(bot.send_circuit_breaker("DD", bypass_rate_limit=True))
            await asyncio.wait_for(sleeper.parked_event.wait(), 2)
            sent = await bot.send_trade(
                "BTCUSDT", "BUY", Decimal("1"), Decimal("50000"), bypass_rate_limit=True
            )
            sleeper.gate.set()
            assert await breaker is True
        finally:
            sleeper.gate.set()
            await bot.close()
        assert sent is False
        assert trade_posts == 0
        assert bot.trade_alerts_dropped == 1

    @pytest.mark.asyncio
    async def test_admitted_trade_retry_yields_to_breaker(
        self, bot: TelegramBot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Trade FIRST: it 503s and waits to retry; the breaker (429) arrives meanwhile.

        The pending trade retry must be dropped and counted, not sent ahead of the
        breaker's retry (a one-message-per-window chat would starve the breaker).
        """
        sleeper = _Sleeper(park=(1.0, 7.0))
        _install_sleeper(monkeypatch, sleeper)
        trade_posts = 0
        breaker_posts = 0
        order: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal trade_posts, breaker_posts
            if "TRADE" in request.content.decode():
                trade_posts += 1
                order.append("trade")
                return httpx.Response(503) if trade_posts == 1 else _ok_response()
            breaker_posts += 1
            order.append("breaker")
            return _raw_429("7") if breaker_posts == 1 else _ok_response()

        bot._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            trade = asyncio.create_task(
                bot.send_trade(
                    "BTCUSDT", "BUY", Decimal("1"), Decimal("50000"), bypass_rate_limit=True
                )
            )
            await asyncio.wait_for(sleeper.parked[1.0].wait(), 2)
            breaker = asyncio.create_task(bot.send_circuit_breaker("DD", bypass_rate_limit=True))
            await asyncio.wait_for(sleeper.parked[7.0].wait(), 2)  # 429, waiting 7 s
            sleeper.gates[1.0].set()  # the trade's backoff ends while the breaker is in flight
            for _ in range(10):
                await asyncio.sleep(0)
            assert trade.done(), "the pending trade retry resolves (dropped) at once"
            sleeper.gates[7.0].set()
            results = await asyncio.gather(trade, breaker)
        finally:
            sleeper.open_all()
            await bot.close()
        assert results == [False, True]
        assert trade_posts == 1, "the pending trade retry must not be sent"
        assert order == ["trade", "breaker", "breaker"]
        assert bot.trade_alerts_dropped == 1
        assert 7.0 in sleeper.delays
