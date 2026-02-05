"""Alert system for paper trading notifications.

Provides Telegram bot integration for real-time alerts on errors,
circuit breakers, trades, and daily summaries.
"""

from vibe_quant.alerts.telegram import (
    AlertType,
    ConfigurationError,
    DailySummary,
    TelegramBot,
    TelegramConfig,
)

__all__ = [
    "AlertType",
    "ConfigurationError",
    "DailySummary",
    "TelegramBot",
    "TelegramConfig",
]
