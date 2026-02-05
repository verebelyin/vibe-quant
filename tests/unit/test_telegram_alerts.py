"""Tests for Telegram alerts module."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from vibe_quant.alerts.telegram import (
    RATE_LIMIT_SECONDS,
    AlertType,
    ConfigurationError,
    DailySummary,
    TelegramBot,
    TelegramConfig,
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
            win_rate=60.0,
            max_drawdown=2.5,
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
        bot._last_alert_times[AlertType.ERROR] = (
            time.monotonic() - RATE_LIMIT_SECONDS - 1
        )

        assert not bot._is_rate_limited(AlertType.ERROR)

    def test_format_message(self, bot: TelegramBot) -> None:
        """Formats message with prefix and timestamp."""
        msg = bot._format_message(AlertType.ERROR, "Test error")

        assert "[ERROR]" in msg
        assert "UTC" in msg
        assert "Test error" in msg

    def test_format_message_all_types(self, bot: TelegramBot) -> None:
        """All alert types have proper prefixes."""
        assert "[ERROR]" in bot._format_message(AlertType.ERROR, "")
        assert "[CIRCUIT BREAKER]" in bot._format_message(AlertType.CIRCUIT_BREAKER, "")
        assert "[TRADE]" in bot._format_message(AlertType.TRADE, "")
        assert "[DAILY]" in bot._format_message(AlertType.DAILY_SUMMARY, "")

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

        result = await bot.send_alert(
            AlertType.ERROR, "Test error", bypass_rate_limit=True
        )

        assert result is True
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_alert_http_error(self, bot: TelegramBot) -> None:
        """Returns False on HTTP error."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Error", request=MagicMock(), response=MagicMock()
        )

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        bot._client = mock_client

        result = await bot.send_alert(AlertType.ERROR, "Test error")

        assert result is False

    @pytest.mark.asyncio
    async def test_send_alert_request_error(self, bot: TelegramBot) -> None:
        """Returns False on request error."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(
            side_effect=httpx.RequestError("Connection failed")
        )
        mock_client.is_closed = False
        bot._client = mock_client

        result = await bot.send_alert(AlertType.ERROR, "Test error")

        assert result is False

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
        assert "[ERROR]" in call_kwargs.kwargs["json"]["text"]

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
        assert "[CIRCUIT BREAKER]" in text
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
        assert "[TRADE]" in text
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
        assert "[DAILY]" in text
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
