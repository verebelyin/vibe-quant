"""Telegram bot for paper trading alerts.

Provides async alerts for errors, circuit breakers, trades, and daily summaries
with rate limiting to prevent notification spam.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum

import httpx

# Environment variable names
ENV_TELEGRAM_BOT_TOKEN = "TELEGRAM_BOT_TOKEN"
ENV_TELEGRAM_CHAT_ID = "TELEGRAM_CHAT_ID"

# Telegram API base URL
TELEGRAM_API_BASE = "https://api.telegram.org/bot"

# Rate limit: 1 alert per type per minute
RATE_LIMIT_SECONDS = 60


class ConfigurationError(Exception):
    """Error in Telegram configuration."""

    pass


class AlertType(StrEnum):
    """Types of alerts the bot can send."""

    ERROR = "ERROR"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    TRADE = "TRADE"
    DAILY_SUMMARY = "DAILY_SUMMARY"


@dataclass(frozen=True)
class TelegramConfig:
    """Configuration for Telegram bot.

    Attributes:
        bot_token: Telegram bot token from BotFather.
        chat_id: Target chat ID for alerts.
    """

    bot_token: str
    chat_id: str

    @classmethod
    def from_env(cls) -> TelegramConfig:
        """Create config from environment variables.

        Returns:
            TelegramConfig instance.

        Raises:
            ConfigurationError: If required env vars are missing.
        """
        bot_token = os.getenv(ENV_TELEGRAM_BOT_TOKEN)
        chat_id = os.getenv(ENV_TELEGRAM_CHAT_ID)

        if not bot_token:
            raise ConfigurationError(
                f"Missing {ENV_TELEGRAM_BOT_TOKEN} environment variable"
            )
        if not chat_id:
            raise ConfigurationError(
                f"Missing {ENV_TELEGRAM_CHAT_ID} environment variable"
            )

        return cls(bot_token=bot_token, chat_id=chat_id)


@dataclass
class DailySummary:
    """Daily trading summary data.

    Attributes:
        date: Summary date.
        total_pnl: Total P&L for the day.
        realized_pnl: Realized P&L.
        unrealized_pnl: Unrealized P&L.
        open_positions: Number of open positions.
        trades_executed: Number of trades executed.
        win_rate: Win rate percentage (0-100).
        max_drawdown: Maximum drawdown percentage.
        total_fees: Total fees paid.
    """

    date: datetime
    total_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    realized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    unrealized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    open_positions: int = 0
    trades_executed: int = 0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    total_fees: Decimal = field(default_factory=lambda: Decimal("0"))

    def format_message(self) -> str:
        """Format summary as Telegram message.

        Returns:
            Formatted message string.
        """
        pnl_emoji = "+" if self.total_pnl >= 0 else ""
        date_str = self.date.strftime("%Y-%m-%d")

        return (
            f"Daily Summary - {date_str}\n"
            f"{'=' * 25}\n"
            f"P&L: {pnl_emoji}{self.total_pnl:.2f} USDT\n"
            f"  Realized: {self.realized_pnl:.2f}\n"
            f"  Unrealized: {self.unrealized_pnl:.2f}\n"
            f"Positions: {self.open_positions}\n"
            f"Trades: {self.trades_executed}\n"
            f"Win Rate: {self.win_rate:.1f}%\n"
            f"Max DD: {self.max_drawdown:.2f}%\n"
            f"Fees: {self.total_fees:.4f} USDT"
        )


class TelegramBot:
    """Telegram bot for sending paper trading alerts.

    Implements rate limiting (1 alert per type per minute) to prevent spam.
    Uses httpx for async HTTP requests.

    Attributes:
        config: Telegram configuration.
    """

    def __init__(self, config: TelegramConfig) -> None:
        """Initialize bot with configuration.

        Args:
            config: TelegramConfig instance.
        """
        self.config = config
        self._last_alert_times: dict[AlertType, float] = {}
        self._client: httpx.AsyncClient | None = None

    @classmethod
    def from_env(cls) -> TelegramBot:
        """Create bot from environment variables.

        Returns:
            TelegramBot instance.

        Raises:
            ConfigurationError: If required env vars are missing.
        """
        return cls(TelegramConfig.from_env())

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client.

        Returns:
            httpx.AsyncClient instance.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    def _is_rate_limited(self, alert_type: AlertType) -> bool:
        """Check if alert type is rate limited.

        Args:
            alert_type: Type of alert to check.

        Returns:
            True if rate limited, False if can send.
        """
        now = time.monotonic()
        last_time = self._last_alert_times.get(alert_type)

        if last_time is None:
            return False

        return (now - last_time) < RATE_LIMIT_SECONDS

    def _update_rate_limit(self, alert_type: AlertType) -> None:
        """Update last alert time for rate limiting.

        Args:
            alert_type: Type of alert sent.
        """
        self._last_alert_times[alert_type] = time.monotonic()

    def _format_message(self, alert_type: AlertType, message: str) -> str:
        """Format message with alert type prefix.

        Args:
            alert_type: Type of alert.
            message: Message content.

        Returns:
            Formatted message string.
        """
        timestamp = datetime.now(UTC).strftime("%H:%M:%S UTC")
        prefix_map = {
            AlertType.ERROR: "[ERROR]",
            AlertType.CIRCUIT_BREAKER: "[CIRCUIT BREAKER]",
            AlertType.TRADE: "[TRADE]",
            AlertType.DAILY_SUMMARY: "[DAILY]",
        }
        prefix = prefix_map.get(alert_type, "[ALERT]")
        return f"{prefix} {timestamp}\n{message}"

    async def send_alert(
        self,
        alert_type: AlertType,
        message: str,
        *,
        bypass_rate_limit: bool = False,
    ) -> bool:
        """Send alert to Telegram.

        Args:
            alert_type: Type of alert.
            message: Message content.
            bypass_rate_limit: If True, skip rate limit check.

        Returns:
            True if sent successfully, False if rate limited or failed.
        """
        if not bypass_rate_limit and self._is_rate_limited(alert_type):
            return False

        formatted = self._format_message(alert_type, message)
        url = f"{TELEGRAM_API_BASE}{self.config.bot_token}/sendMessage"

        payload = {
            "chat_id": self.config.chat_id,
            "text": formatted,
            "parse_mode": "HTML",
        }

        try:
            client = await self._get_client()
            response = await client.post(url, json=payload)
            response.raise_for_status()
            self._update_rate_limit(alert_type)
            return True
        except httpx.HTTPStatusError:
            return False
        except httpx.RequestError:
            return False

    async def send_error(self, message: str, *, bypass_rate_limit: bool = False) -> bool:
        """Send error alert.

        Args:
            message: Error message.
            bypass_rate_limit: If True, skip rate limit check.

        Returns:
            True if sent successfully.
        """
        return await self.send_alert(
            AlertType.ERROR, message, bypass_rate_limit=bypass_rate_limit
        )

    async def send_circuit_breaker(
        self,
        reason: str,
        details: str = "",
        *,
        bypass_rate_limit: bool = False,
    ) -> bool:
        """Send circuit breaker alert.

        Args:
            reason: Reason for circuit breaker activation.
            details: Additional details.
            bypass_rate_limit: If True, skip rate limit check.

        Returns:
            True if sent successfully.
        """
        message = reason
        if details:
            message = f"{reason}\n{details}"
        return await self.send_alert(
            AlertType.CIRCUIT_BREAKER, message, bypass_rate_limit=bypass_rate_limit
        )

    async def send_trade(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        *,
        bypass_rate_limit: bool = False,
    ) -> bool:
        """Send trade executed alert.

        Args:
            symbol: Trading symbol.
            side: Trade side (BUY/SELL).
            quantity: Trade quantity.
            price: Trade price.
            bypass_rate_limit: If True, skip rate limit check.

        Returns:
            True if sent successfully.
        """
        message = f"{side} {quantity} {symbol} @ {price:.2f}"
        return await self.send_alert(
            AlertType.TRADE, message, bypass_rate_limit=bypass_rate_limit
        )

    async def send_daily_summary(
        self,
        summary: DailySummary,
        *,
        bypass_rate_limit: bool = False,
    ) -> bool:
        """Send daily summary alert.

        Args:
            summary: Daily trading summary.
            bypass_rate_limit: If True, skip rate limit check.

        Returns:
            True if sent successfully.
        """
        return await self.send_alert(
            AlertType.DAILY_SUMMARY,
            summary.format_message(),
            bypass_rate_limit=bypass_rate_limit,
        )

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
