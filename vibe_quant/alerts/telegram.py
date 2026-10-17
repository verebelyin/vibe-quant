"""Telegram bot for paper trading alerts.

Provides async alerts for errors, circuit breakers, trades, and daily summaries
with rate limiting to prevent notification spam.

Secrets: the Bot API puts the token in the URL path
(``https://api.telegram.org/bot<TOKEN>/sendMessage``) and httpx logs every
request URL at INFO. A redaction filter on the ``httpx`` logger and on this
module's logger replaces ``bot<TOKEN>`` with ``bot<redacted>``, and failure
messages are sanitized before logging, so the token never reaches a log.
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum

import httpx

logger = logging.getLogger(__name__)

# Environment variable names
ENV_TELEGRAM_BOT_TOKEN = "TELEGRAM_BOT_TOKEN"
ENV_TELEGRAM_CHAT_ID = "TELEGRAM_CHAT_ID"

# Telegram API base URL
TELEGRAM_API_BASE = "https://api.telegram.org/bot"

# Rate limit: 1 alert per type per minute
RATE_LIMIT_SECONDS = 60

#: How long the start-up probe may take before the start is refused.
TELEGRAM_PROBE_TIMEOUT_S = 5.0

#: Longest ``reason`` inside a refusal (a Telegram description can be arbitrary).
_REASON_MAX_CHARS = 200

#: Shortest whitespace-separated piece of a token that is redacted on its own.
_MIN_TOKEN_PART_CHARS = 8

#: Only one of the two env vars set: the operator made a mistake, refuse loudly.
HALF_MSG = (
    "Telegram alerts half-configured: {VAR} is not set (set both "
    "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID, or neither); refusing to start"
)

#: Configured but unusable -- every CRITICAL alert would be silently lost.
FAIL_MSG = (
    "Telegram alerts are configured but not working ({reason}); refusing to start. "
    "Fix TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID, or unset both to run without alerts "
    "(backend env; restart the backend)"
)

_TOKEN_IN_URL = re.compile(r"(/bot)[^/\s\"'<>]+")


class TelegramUnavailableError(Exception):
    """Telegram alerts configured but unusable. ``str()`` is token-free by construction."""


def redact_token(text: str) -> str:
    """Replace any ``/bot<token>`` URL segment with ``/bot<redacted>``."""
    return _TOKEN_IN_URL.sub(r"\1<redacted>", text)


class TelegramTokenFilter(logging.Filter):
    """Logging filter that strips Telegram bot tokens from records."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        redacted = redact_token(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


_TOKEN_FILTER = TelegramTokenFilter()


def install_token_redaction() -> None:
    """Attach the redaction filter to httpx/httpcore and this module (idempotent)."""
    for name in ("httpx", "httpcore", __name__):
        target = logging.getLogger(name)
        if _TOKEN_FILTER not in target.filters:
            target.addFilter(_TOKEN_FILTER)


install_token_redaction()


def _is_retryable(exc: Exception) -> bool:
    """Transport errors, 429 and 5xx are retryable; other 4xx are permanent."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = getattr(exc.response, "status_code", None)
        if isinstance(status, int):
            return status == 429 or status >= 500
    return True


def _describe_failure(exc: Exception) -> str:
    """Token-free description of a failed send (status + Telegram description)."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = getattr(exc.response, "status_code", "?")
        description = ""
        try:
            body = exc.response.json()
            if isinstance(body, dict):
                description = str(body.get("description", ""))
        except Exception:
            description = ""
        return redact_token(f"HTTP {status} {description}".strip())
    return redact_token(f"{type(exc).__name__}: {exc}")


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

    bot_token: str = field(repr=False)
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
            raise ConfigurationError(f"Missing {ENV_TELEGRAM_BOT_TOKEN} environment variable")
        if not chat_id:
            raise ConfigurationError(f"Missing {ENV_TELEGRAM_CHAT_ID} environment variable")

        return cls(bot_token=bot_token, chat_id=chat_id)


def telegram_config_or_none() -> TelegramConfig | None:
    """Config from the environment, or ``None`` when alerts are not configured.

    Returns:
        ``None`` when both vars are unset/empty, else the config.

    Raises:
        TelegramUnavailableError: When exactly one of the two vars is set.
    """
    bot_token = os.getenv(ENV_TELEGRAM_BOT_TOKEN)
    chat_id = os.getenv(ENV_TELEGRAM_CHAT_ID)
    if bot_token and chat_id:
        return TelegramConfig(bot_token=bot_token, chat_id=chat_id)
    if not bot_token and not chat_id:
        return None
    missing = ENV_TELEGRAM_CHAT_ID if bot_token else ENV_TELEGRAM_BOT_TOKEN
    raise TelegramUnavailableError(HALF_MSG.format(VAR=missing))


def _make_probe_client(timeout_s: float) -> httpx.AsyncClient:
    """Build the probe's HTTP client (test seam: ``MockTransport``).

    The httpx timeout sits ABOVE ``timeout_s`` so ``asyncio.timeout(timeout_s)``
    always governs and reports ``TimeoutError``, never ``ReadTimeout``.
    """
    return httpx.AsyncClient(timeout=timeout_s + 1.0)


def _description_from_body(response: httpx.Response) -> str:
    """Telegram's ``description`` field, or ``""`` when the body is not a JSON object."""
    try:
        body = response.json()
    except Exception:
        return ""
    if not isinstance(body, dict):
        return ""
    description = body.get("description", "")
    return description if isinstance(description, str) else str(description)


def _clean_reason(text: str, config: TelegramConfig) -> str:
    """Redact the token (literal first, then the ``/bot<token>`` URL form), cut to 200 chars.

    A token with surrounding or inner whitespace is redacted raw, stripped and per
    whitespace-separated part (>= 8 chars), longest first. An all-whitespace token
    (``TELEGRAM_BOT_TOKEN=" "``) is skipped: it would turn every space in the
    message into ``<redacted>``.
    """
    token = config.bot_token
    stripped = token.strip()
    if stripped:
        parts = {part for part in stripped.split() if len(part) >= _MIN_TOKEN_PART_CHARS}
        for secret in sorted({token, stripped, *parts}, key=len, reverse=True):
            text = text.replace(secret, "<redacted>")
    return redact_token(text)[:_REASON_MAX_CHARS]


def _probe_reason(exc: Exception, config: TelegramConfig) -> str:
    """Token-free reason for a failed probe: ``HTTP <status> <description>`` or the class name.

    An httpx error's ``str()`` holds the full URL (and the token), so it is never used.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        status = getattr(exc.response, "status_code", "?")
        description = _description_from_body(exc.response)
        return _clean_reason(f"HTTP {status} {description}".strip(), config)
    return type(exc).__name__


def _response_failure(response: httpx.Response, config: TelegramConfig) -> str | None:
    """``HTTP 200 ok=false`` / ``HTTP 200 invalid body``, or ``None`` on success.

    Only exactly HTTP 200 is a success: another 2xx is refused as ``HTTP <status> <description>``.
    """
    if response.status_code != 200:
        description = _description_from_body(response) or "invalid body"
        return _clean_reason(f"HTTP {response.status_code} {description}", config)
    try:
        body = response.json()
    except Exception:
        return "HTTP 200 invalid body"
    if not isinstance(body, dict):
        return "HTTP 200 invalid body"
    if body.get("ok") is not True:
        return "HTTP 200 ok=false"
    return None


async def verify_telegram(
    config: TelegramConfig, *, timeout_s: float = TELEGRAM_PROBE_TIMEOUT_S
) -> None:
    """Probe a configured bot once with a silent ``sendChatAction``; raise if unusable.

    ``sendChatAction`` is silent (a ≤5 s "typing…" indicator) and checks the token,
    the chat id AND the permission to post (a blocked bot returns 403), unlike
    ``getMe``, which passes with a wrong chat id.

    Raises:
        TelegramUnavailableError: On any failure. ``str()`` is token-free.
    """
    url = f"{TELEGRAM_API_BASE}{config.bot_token}/sendChatAction"
    payload = {"chat_id": config.chat_id, "action": "typing"}
    try:
        # P-R6: the try wraps the WHOLE timed block, so a TimeoutError is reported
        # as the class name "TimeoutError". A client is closed on success and failure.
        async with _make_probe_client(timeout_s) as client:
            async with asyncio.timeout(timeout_s):
                attempt = await client.post(url, json=payload)
                attempt.raise_for_status()
    except Exception as exc:
        raise TelegramUnavailableError(
            FAIL_MSG.format(reason=_probe_reason(exc, config))
        ) from None

    failure = _response_failure(attempt, config)
    if failure is not None:
        raise TelegramUnavailableError(FAIL_MSG.format(reason=failure)) from None


async def require_working_telegram(
    *, timeout_s: float = TELEGRAM_PROBE_TIMEOUT_S
) -> TelegramConfig | None:
    """``None`` when Telegram is unconfigured, else the config after a working probe.

    Raises:
        TelegramUnavailableError: When half-configured, or when the probe fails.
    """
    config = telegram_config_or_none()
    if config is not None:
        await verify_telegram(config, timeout_s=timeout_s)
    return config


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
        win_rate: Win rate as decimal fraction (0.55 = 55%).
        max_drawdown: Maximum drawdown as decimal fraction (0.05 = 5%).
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
        q2 = Decimal("0.01")
        q4 = Decimal("0.0001")
        pnl_emoji = "+" if self.total_pnl >= 0 else ""
        date_str = self.date.strftime("%Y-%m-%d")

        return (
            f"Daily Summary - {date_str}\n"
            f"{'=' * 25}\n"
            f"P&L: {pnl_emoji}{self.total_pnl.quantize(q2)} USDT\n"
            f"  Realized: {self.realized_pnl.quantize(q2)}\n"
            f"  Unrealized: {self.unrealized_pnl.quantize(q2)}\n"
            f"Positions: {self.open_positions}\n"
            f"Trades: {self.trades_executed}\n"
            f"Win Rate: {self.win_rate * 100:.1f}%\n"
            f"Max DD: {self.max_drawdown * 100:.2f}%\n"
            f"Fees: {self.total_fees.quantize(q4)} USDT"
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
            if self._client is not None:
                logger.info("Recreating Telegram client (previous was closed)")
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    def _is_rate_limited(self, alert_type: AlertType) -> bool:
        """Check if alert type is rate limited.

        Args:
            alert_type: Type of alert to check.

        Returns:
            True if rate limited, False if can send.
        """
        # Wall-clock time keeps rate limits aligned with real elapsed time.
        now = time.time()
        last_time = self._last_alert_times.get(alert_type)

        if last_time is None:
            return False

        return (now - last_time) < RATE_LIMIT_SECONDS

    def _update_rate_limit(self, alert_type: AlertType) -> None:
        """Update last alert time for rate limiting.

        Args:
            alert_type: Type of alert sent.
        """
        self._last_alert_times[alert_type] = time.time()

    def _format_message(self, alert_type: AlertType, message: str) -> str:
        """Format message with alert type prefix.

        Args:
            alert_type: Type of alert.
            message: Message content.

        Returns:
            Formatted message string.
        """
        timestamp = datetime.now(UTC).strftime("%H:%M:%S UTC")
        # parse_mode=HTML: dynamic text (exception messages like "<Response
        # [503]>") must be escaped or Telegram rejects the whole message.
        message = html.escape(message, quote=False)
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

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                client = await self._get_client()
                response = await client.post(url, json=payload)
                response.raise_for_status()
                self._update_rate_limit(alert_type)
                return True
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                # Reset client on transport/protocol failures so next send
                # starts from a known-good connection state.
                await self.close()
                reason = _describe_failure(exc)
                if not _is_retryable(exc):
                    # 400/401/403/404: retrying cannot help (bad token, bad
                    # chat id, unparseable text) -- fail once, loudly.
                    logger.warning("Alert send rejected by Telegram (%s); not retrying", reason)
                    return False
                if attempt < max_attempts - 1:
                    delay = 1.0 * (2**attempt)  # 1s, 2s
                    logger.warning(
                        "Alert send attempt %d/%d failed: %s, retrying in %.0fs",
                        attempt + 1,
                        max_attempts,
                        reason,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.warning("Alert send failed after %d attempts: %s", max_attempts, reason)
                    return False
        return False

    async def send_error(self, message: str, *, bypass_rate_limit: bool = False) -> bool:
        """Send error alert.

        Args:
            message: Error message.
            bypass_rate_limit: If True, skip rate limit check.

        Returns:
            True if sent successfully.
        """
        return await self.send_alert(AlertType.ERROR, message, bypass_rate_limit=bypass_rate_limit)

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
            True if sent successfully, False if invalid or failed.
        """
        if quantity <= 0 or price <= 0:
            logger.warning("Invalid trade alert: quantity=%s price=%s", quantity, price)
            return False
        message = f"{side} {quantity} {symbol} @ {price.quantize(Decimal('0.01'))}"
        return await self.send_alert(AlertType.TRADE, message, bypass_rate_limit=bypass_rate_limit)

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

    async def __aenter__(self) -> TelegramBot:
        """Enter async context manager."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Exit async context manager, ensuring client cleanup."""
        await self.close()
