"""Tests for Telegram alerts module."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

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


class TestTelegramConfig:
    """Tests for TelegramConfig."""

    def test_create_directly(self) -> None:
        """Can create config directly."""
        config = TelegramConfig(
            bot_token="123:ABC",
            chat_id="456",
        )

        assert config.bot_token == "123:ABC"
        assert config.chat_id == "456"

    def test_from_env_missing_token(self) -> None:
        """Raises error if bot token missing."""
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ConfigurationError, match="TELEGRAM_BOT_TOKEN"),
        ):
            TelegramConfig.from_env()

    def test_from_env_missing_chat_id(self) -> None:
        """Raises error if chat ID missing."""
        with (
            patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "123:ABC"}, clear=True),
            pytest.raises(ConfigurationError, match="TELEGRAM_CHAT_ID"),
        ):
            TelegramConfig.from_env()

    def test_from_env_success(self) -> None:
        """Creates config from env vars."""
        env = {
            "TELEGRAM_BOT_TOKEN": "123:ABC",
            "TELEGRAM_CHAT_ID": "456",
        }
        with patch.dict(os.environ, env, clear=True):
            config = TelegramConfig.from_env()

        assert config.bot_token == "123:ABC"
        assert config.chat_id == "456"


class TestTelegramConfigOrNone:
    """The start-gate loader (vibe-quant-u2r9r): unset -> None, half-configured -> refuse."""

    def test_configured_returns_config(self) -> None:
        env = {"TELEGRAM_BOT_TOKEN": "123:ABC", "TELEGRAM_CHAT_ID": "456"}
        with patch.dict(os.environ, env, clear=True):
            config = telegram_mod.telegram_config_or_none()

        assert config == TelegramConfig(bot_token="123:ABC", chat_id="456")

    @pytest.mark.parametrize(
        "env",
        [{}, {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""}],
        ids=["unset", "empty"],
    )
    def test_unconfigured_is_none(self, env: dict[str, str]) -> None:
        with patch.dict(os.environ, env, clear=True):
            assert telegram_mod.telegram_config_or_none() is None

    @pytest.mark.parametrize(
        ("env", "var"),
        [
            ({"TELEGRAM_BOT_TOKEN": "123:ABC"}, "TELEGRAM_CHAT_ID"),
            ({"TELEGRAM_CHAT_ID": "456"}, "TELEGRAM_BOT_TOKEN"),
            ({"TELEGRAM_BOT_TOKEN": "123:ABC", "TELEGRAM_CHAT_ID": ""}, "TELEGRAM_CHAT_ID"),
        ],
        ids=["token-only", "chat-only", "empty-chat"],
    )
    def test_half_configured_raises(self, env: dict[str, str], var: str) -> None:
        with (
            patch.dict(os.environ, env, clear=True),
            pytest.raises(TelegramUnavailableError) as excinfo,
        ):
            telegram_mod.telegram_config_or_none()

        assert str(excinfo.value) == HALF_MSG.format(VAR=var)

    def test_space_only_token_counts_as_set(self) -> None:
        """Only ``""`` is unset: a whitespace token is configured and must be probed."""
        env = {"TELEGRAM_BOT_TOKEN": " ", "TELEGRAM_CHAT_ID": "42"}
        with patch.dict(os.environ, env, clear=True):
            config = telegram_mod.telegram_config_or_none()

        assert config is not None and config.bot_token == " "


class TestAlertType:
    """Tests for AlertType enum."""

    def test_all_types_defined(self) -> None:
        """All expected alert types are defined."""
        assert AlertType.ERROR.value == "ERROR"
        assert AlertType.CIRCUIT_BREAKER.value == "CIRCUIT_BREAKER"
        assert AlertType.TRADE.value == "TRADE"
        assert AlertType.DAILY_SUMMARY.value == "DAILY_SUMMARY"


class TestDailySummary:
    """Tests for DailySummary."""

    def test_default_values(self) -> None:
        """Default values are zeroed."""
        now = datetime.now(UTC)
        summary = DailySummary(date=now)

        assert summary.total_pnl == Decimal("0")
        assert summary.open_positions == 0
        assert summary.trades_executed == 0
        assert summary.win_rate == 0.0

    def test_format_message_positive_pnl(self) -> None:
        """Formats message with positive P&L."""
        summary = DailySummary(
            date=datetime(2024, 1, 15, tzinfo=UTC),
            total_pnl=Decimal("150.50"),
            realized_pnl=Decimal("100.00"),
            unrealized_pnl=Decimal("50.50"),
            open_positions=2,
            trades_executed=5,
            win_rate=0.60,
            max_drawdown=0.025,
            total_fees=Decimal("1.25"),
        )

        msg = summary.format_message()

        assert "2024-01-15" in msg
        assert "+150.50" in msg
        assert "Realized: 100.00" in msg
        assert "Unrealized: 50.50" in msg
        assert "Positions: 2" in msg
        assert "Trades: 5" in msg
        assert "Win Rate: 60.0%" in msg
        assert "Max DD: 2.50%" in msg
        assert "Fees: 1.2500" in msg

    def test_format_message_negative_pnl(self) -> None:
        """Formats message with negative P&L (no plus sign)."""
        summary = DailySummary(
            date=datetime(2024, 1, 15, tzinfo=UTC),
            total_pnl=Decimal("-50.00"),
        )

        msg = summary.format_message()

        assert "-50.00" in msg
        assert "+-50" not in msg


class TestTelegramBot:
    """Tests for TelegramBot."""

    @pytest.fixture
    def config(self) -> TelegramConfig:
        """Create test config."""
        return TelegramConfig(bot_token="123:ABC", chat_id="456")

    @pytest.fixture
    def bot(self, config: TelegramConfig) -> TelegramBot:
        """Create test bot."""
        return TelegramBot(config)

    def test_init(self, config: TelegramConfig) -> None:
        """Bot initializes with config."""
        bot = TelegramBot(config)

        assert bot.config == config
        assert bot._client is None
        assert bot._last_alert_times == {}

    def test_from_env(self) -> None:
        """Creates bot from env vars."""
        env = {
            "TELEGRAM_BOT_TOKEN": "123:ABC",
            "TELEGRAM_CHAT_ID": "456",
        }
        with patch.dict(os.environ, env, clear=True):
            bot = TelegramBot.from_env()

        assert bot.config.bot_token == "123:ABC"
        assert bot.config.chat_id == "456"

    def test_rate_limit_not_set(self, bot: TelegramBot) -> None:
        """Not rate limited if never sent."""
        assert not bot._is_rate_limited(AlertType.ERROR)
        assert not bot._is_rate_limited(AlertType.TRADE)

    def test_rate_limit_after_send(self, bot: TelegramBot) -> None:
        """Rate limited after update."""
        bot._update_rate_limit(AlertType.ERROR)

        assert bot._is_rate_limited(AlertType.ERROR)
        # Other types not affected
        assert not bot._is_rate_limited(AlertType.TRADE)

    def test_rate_limit_expires(self, bot: TelegramBot) -> None:
        """Rate limit expires after threshold."""
        # Set a time in the past
        bot._last_alert_times[AlertType.ERROR] = time.time() - RATE_LIMIT_SECONDS - 1

        assert not bot._is_rate_limited(AlertType.ERROR)

    def test_format_message(self, bot: TelegramBot) -> None:
        """Formats message with an emoji header and a UTC clock time."""
        msg = bot._format_message(AlertType.ERROR, "Test error")

        header = msg.split("\n")[0]
        assert re.fullmatch(r"⚠️ ERROR · \d{2}:\d{2}:\d{2} UTC", header)
        assert "Test error" in msg

    def test_format_message_all_types(self, bot: TelegramBot) -> None:
        """All alert types have proper emoji headers."""
        assert "⚠️ ERROR · " in bot._format_message(AlertType.ERROR, "")
        assert "🚨 CIRCUIT BREAKER · " in bot._format_message(AlertType.CIRCUIT_BREAKER, "")
        assert "💹 TRADE · " in bot._format_message(AlertType.TRADE, "")
        assert "📊 DAILY · " in bot._format_message(AlertType.DAILY_SUMMARY, "")

    def test_format_message_unknown_type_uses_default(self, bot: TelegramBot) -> None:
        """An unmapped type falls back to the generic bell header."""
        msg = bot._format_message(cast("AlertType", "WAT"), "")

        assert "🔔 ALERT · " in msg

    @pytest.mark.asyncio
    async def test_send_alert_success(self, bot: TelegramBot) -> None:
        """Successfully sends alert."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        bot._client = mock_client

        result = await bot.send_alert(AlertType.ERROR, "Test error")

        assert result is True
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert "sendMessage" in call_kwargs.args[0]
        assert call_kwargs.kwargs["json"]["chat_id"] == "456"
        assert "Test error" in call_kwargs.kwargs["json"]["text"]

    @pytest.mark.asyncio
    async def test_send_alert_rate_limited(self, bot: TelegramBot) -> None:
        """Returns False when rate limited."""
        bot._update_rate_limit(AlertType.ERROR)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        bot._client = mock_client

        result = await bot.send_alert(AlertType.ERROR, "Test error")

        assert result is False
        mock_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_alert_bypass_rate_limit(self, bot: TelegramBot) -> None:
        """Can bypass rate limit."""
        bot._update_rate_limit(AlertType.ERROR)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        bot._client = mock_client

        result = await bot.send_alert(AlertType.ERROR, "Test error", bypass_rate_limit=True)

        assert result is True
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_alert_http_error(
        self, bot: TelegramBot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retries the full budget, backing off 1s then 2s, then returns False."""
        delays: list[float] = []

        async def _record_sleep(delay: float) -> None:
            delays.append(delay)

        # Record (don't wait out) the retry backoff. Patch only the telegram
        # module's asyncio view so the global asyncio.sleep stays real.
        monkeypatch.setattr(telegram_mod, "asyncio", SimpleNamespace(sleep=_record_sleep))

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Error", request=MagicMock(), response=MagicMock()
        )

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        mock_client.aclose = AsyncMock()

        # Patch _get_client to always return mock (avoids real httpx on retry)
        bot._get_client = AsyncMock(return_value=mock_client)

        result = await bot.send_alert(AlertType.ERROR, "Test error")

        assert result is False
        assert mock_client.post.await_count == 3
        assert delays == [1.0, 2.0]

    @pytest.mark.asyncio
    async def test_send_alert_request_error(
        self, bot: TelegramBot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retries the full budget with the same backoff, then returns False."""
        delays: list[float] = []

        async def _record_sleep(delay: float) -> None:
            delays.append(delay)

        monkeypatch.setattr(telegram_mod, "asyncio", SimpleNamespace(sleep=_record_sleep))

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(side_effect=httpx.RequestError("Connection failed"))
        mock_client.is_closed = False
        mock_client.aclose = AsyncMock()

        # Patch _get_client to always return mock (avoids real httpx on retry)
        bot._get_client = AsyncMock(return_value=mock_client)

        result = await bot.send_alert(AlertType.ERROR, "Test error")

        assert result is False
        assert mock_client.post.await_count == 3
        assert delays == [1.0, 2.0]

    @pytest.mark.asyncio
    async def test_send_error(self, bot: TelegramBot) -> None:
        """send_error uses ERROR type."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        bot._client = mock_client

        await bot.send_error("Connection lost")

        call_kwargs = mock_client.post.call_args
        assert "⚠️ ERROR" in call_kwargs.kwargs["json"]["text"]

    @pytest.mark.asyncio
    async def test_send_circuit_breaker(self, bot: TelegramBot) -> None:
        """send_circuit_breaker uses CIRCUIT_BREAKER type."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        bot._client = mock_client

        await bot.send_circuit_breaker("Max drawdown", "Hit 15% threshold")

        call_kwargs = mock_client.post.call_args
        text = call_kwargs.kwargs["json"]["text"]
        assert "🚨 CIRCUIT BREAKER" in text
        assert "Max drawdown" in text
        assert "15% threshold" in text

    @pytest.mark.asyncio
    async def test_send_circuit_breaker_no_details(self, bot: TelegramBot) -> None:
        """send_circuit_breaker works without details."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        bot._client = mock_client

        await bot.send_circuit_breaker("Max daily loss")

        call_kwargs = mock_client.post.call_args
        text = call_kwargs.kwargs["json"]["text"]
        assert "Max daily loss" in text

    @pytest.mark.asyncio
    async def test_send_trade(self, bot: TelegramBot) -> None:
        """send_trade formats trade info."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        bot._client = mock_client

        await bot.send_trade(
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("0.5"),
            price=Decimal("50000.00"),
        )

        call_kwargs = mock_client.post.call_args
        text = call_kwargs.kwargs["json"]["text"]
        assert "💹 TRADE" in text
        assert "BUY" in text
        assert "0.5" in text
        assert "BTCUSDT" in text
        assert "50000.00" in text

    @pytest.mark.asyncio
    async def test_send_daily_summary(self, bot: TelegramBot) -> None:
        """send_daily_summary formats summary."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        bot._client = mock_client

        summary = DailySummary(
            date=datetime(2024, 1, 15, tzinfo=UTC),
            total_pnl=Decimal("100.00"),
            trades_executed=10,
        )

        await bot.send_daily_summary(summary)

        call_kwargs = mock_client.post.call_args
        text = call_kwargs.kwargs["json"]["text"]
        assert "📊 DAILY" in text
        assert "2024-01-15" in text
        assert "+100.00" in text

    @pytest.mark.asyncio
    async def test_close_client(self, bot: TelegramBot) -> None:
        """close() closes the HTTP client."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.aclose = AsyncMock()
        bot._client = mock_client

        await bot.close()

        mock_client.aclose.assert_called_once()
        assert bot._client is None

    @pytest.mark.asyncio
    async def test_close_already_closed(self, bot: TelegramBot) -> None:
        """close() handles already closed client."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = True
        bot._client = mock_client

        await bot.close()

        # Should not call aclose on already closed client
        mock_client.aclose.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_no_client(self, bot: TelegramBot) -> None:
        """close() handles no client."""
        assert bot._client is None
        await bot.close()  # Should not raise


class TestRateLimitIntegration:
    """Integration tests for rate limiting across alert types."""

    @pytest.fixture
    def bot(self) -> TelegramBot:
        """Create test bot."""
        config = TelegramConfig(bot_token="123:ABC", chat_id="456")
        return TelegramBot(config)

    @pytest.mark.asyncio
    async def test_different_types_independent(self, bot: TelegramBot) -> None:
        """Rate limits are independent per alert type."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        bot._client = mock_client

        # Send ERROR
        result1 = await bot.send_alert(AlertType.ERROR, "Error 1")
        assert result1 is True

        # ERROR rate limited, but TRADE should work
        result2 = await bot.send_alert(AlertType.ERROR, "Error 2")
        result3 = await bot.send_alert(AlertType.TRADE, "Trade 1")

        assert result2 is False  # Rate limited
        assert result3 is True  # Different type, not limited

        # Verify post was called twice (ERROR + TRADE)
        assert mock_client.post.call_count == 2


# --------------------------------------- ci7q9/21qug: burst backpressure


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


class TestTradeBurstBackpressure:
    """A trade-alert burst must never cost a CIRCUIT_BREAKER alert (vibe-quant-21qug).

    Trade alerts bypass the per-type rate limit (R31), so a fast strategy can post
    dozens of messages and make Telegram answer 429 for the whole chat. The circuit
    breaker then burned ~3 s of retries and was dropped, and every failed send closed
    the shared httpx client under the other in-flight requests.
    """

    @pytest.fixture()
    def bot(self) -> TelegramBot:
        return TelegramBot(TelegramConfig(bot_token="123:ABC", chat_id="456"))

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
    async def test_rejected_send_does_not_close_the_shared_client(
        self, bot: TelegramBot
    ) -> None:
        """A 400 leaves the shared client open for the other in-flight requests."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"ok": False, "description": "Bad Request"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        bot._client = client
        try:
            assert await bot.send_alert(AlertType.ERROR, "x") is False
            assert bot._client is client
            assert not client.is_closed
        finally:
            await bot.close()

    @pytest.mark.asyncio
    async def test_retryable_failure_retries_on_the_same_client(
        self, bot: TelegramBot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A transport error retries on the SAME client (closing it would break peers)."""

        async def _record_sleep(_delay: float) -> None:
            return None

        monkeypatch.setattr(telegram_mod, "asyncio", SimpleNamespace(sleep=_record_sleep))
        posts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            posts.append(1)
            raise httpx.ConnectError("boom", request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        bot._client = client
        try:
            assert await bot.send_alert(AlertType.ERROR, "x") is False
            assert len(posts) == 3
            assert bot._client is client
            assert not client.is_closed
        finally:
            await bot.close()


# ------------------------- retry_after bounds, critical priority (rev round 1)


def _raw_429(retry_after_json: str) -> httpx.Response:
    """A 429 whose ``retry_after`` is spliced in as RAW JSON (``1e309``, ``10**400``)."""
    body = '{"ok": false, "error_code": 429, "parameters": {"retry_after": ' + retry_after_json + "}}"
    return httpx.Response(429, content=body.encode(), headers={"content-type": "application/json"})


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
    @pytest.fixture()
    def bot(self) -> TelegramBot:
        return TelegramBot(TelegramConfig(bot_token="123:ABC", chat_id="456"))

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("retry_after_json", "expected_delay"),
        [
            ("7", 7.0),  # nondefault: kills "use 1.0 instead of retry_after"
            ("2.5", 2.5),
            ("0", 0.0),  # lower edge, allowed
            ("60", 60.0),  # cap edge, allowed
            ("60.5", 1.0),  # just over the cap -> default backoff
            ("-1", 1.0),
            ("1e309", 1.0),  # JSON -> float inf
            ("1" + "0" * 400, 1.0),  # int too large for float()
            ("NaN", 1.0),
            ("Infinity", 1.0),
        ],
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
    async def test_unbounded_retry_after_is_logged_once(
        self, bot: TelegramBot, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING)
        _install_sleeper(monkeypatch, _Sleeper())
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return _raw_429("1e309") if calls == 1 else _ok_response()

        bot._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            assert await bot.send_error("x", bypass_rate_limit=True) is True
        finally:
            await bot.close()
        assert caplog.text.count("retry_after outside") == 1

    @pytest.mark.asyncio
    async def test_trade_429_with_unbounded_retry_after_still_drops(
        self, bot: TelegramBot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
