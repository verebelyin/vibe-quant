"""Telegram alerts for shadow state changes (design arch-c7-2 §2 Alerts, Q5).

Only ``state_change`` events alert (notes never do). An event is stamped sent only when
``send_alert`` returns exactly ``True``; ``False``, an exception or no Telegram config
leave it unsent, and the next update retries it. A crash after delivery and before the
stamp re-sends that alert once. Sending holds the shadow update lock, so two concurrent
senders never send the same event twice (clmmc.24); the one that finds it held skips.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Final

from vibe_quant.alerts.telegram import (
    AlertType,
    TelegramBot,
    TelegramUnavailableError,
    telegram_config_or_none,
)
from vibe_quant.db.connection import DEFAULT_DB_PATH
from vibe_quant.db.state_manager import ShadowUpdateRunning, shadow_update_lock

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vibe_quant.db.state_manager import StateManager

logger = logging.getLogger(__name__)

TEST_TAG: Final = "[TEST] "


def alert_text(event: Mapping[str, object]) -> str:
    """One alert line; a test row's text starts with ``[TEST] `` exactly once."""
    body = str(event["message"])
    tag = TEST_TAG if event["is_test"] else ""
    body = body.removeprefix(TEST_TAG) if tag else body
    return f"{tag}Shadow '{event['name']}': {event['from_state']} -> {event['to_state']}. {body}"


async def _send_all(state: StateManager, bot: TelegramBot, events: list[dict[str, object]]) -> int:
    stamped = 0
    try:
        for event in events:
            try:
                ok = await bot.send_alert(AlertType.INFO, alert_text(event), bypass_rate_limit=True)
            except Exception as exc:  # unsent: retried by the next update
                logger.warning("shadow alert %s not sent (%s)", event["id"], type(exc).__name__)
                continue
            if ok is True:  # stamp each at once: a later crash re-sends only the rest
                stamped += state.mark_shadow_alert_sent([int(str(event["id"]))])
    finally:
        await bot.close()
    return stamped


def send_pending(state: StateManager) -> int:
    """Send every unsent ``state_change`` alert; returns how many were stamped sent.

    Skips (returns 0) while another update or sender holds the shadow update lock: the
    holder's own send, or the next update, sends what is pending.
    """
    try:
        with shadow_update_lock(state.db_path or DEFAULT_DB_PATH):
            return _send_pending_locked(state)
    except ShadowUpdateRunning:
        logger.info("shadow alerts skipped: another shadow update or sender holds the lock")
        return 0


def _send_pending_locked(state: StateManager) -> int:
    events = state.pending_shadow_alerts()
    if not events:
        return 0
    try:
        config = telegram_config_or_none()
    except TelegramUnavailableError as exc:
        logger.warning("shadow alerts not sent: %s", exc)
        return 0
    if config is None:
        return 0
    return asyncio.run(_send_all(state, TelegramBot(config), events))
