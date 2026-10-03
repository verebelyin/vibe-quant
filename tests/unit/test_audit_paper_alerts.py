"""Telegram secret hygiene + alert delivery + ErrorHandler semantics (vibe-quant-e70tl.21)."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import httpx
import pytest

from vibe_quant.alerts.telegram import TelegramBot, TelegramConfig, redact_token
from vibe_quant.paper.errors import ErrorCategory, ErrorHandler, RetryConfig

if TYPE_CHECKING:
    from collections.abc import Callable

TOKEN = "123456789:AAHsecretTOKEN-abcdefghijklmnopqrstu"


def _bot(handler: Callable[[httpx.Request], httpx.Response]) -> tuple[TelegramBot, list[Any]]:
    seen: list[Any] = []

    def _record(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.read()))
        return handler(request)

    bot = TelegramBot(TelegramConfig(bot_token=TOKEN, chat_id="42"))

    async def _client() -> httpx.AsyncClient:
        # A REAL httpx client (MockTransport) so httpx's own INFO request
        # logging runs exactly as in production.
        return httpx.AsyncClient(transport=httpx.MockTransport(_record))

    bot._get_client = _client  # type: ignore[method-assign]
    return bot, seen


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _instant(_delay: float) -> None:
        return None

    monkeypatch.setattr("vibe_quant.alerts.telegram.asyncio.sleep", _instant)


async def test_token_never_logged_on_success(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    bot, seen = _bot(lambda _r: httpx.Response(200, json={"ok": True}))
    assert await bot.send_error("boom", bypass_rate_limit=True) is True
    assert len(seen) == 1
    # httpx logged the request line -- with the token redacted.
    assert "api.telegram.org/bot<redacted>/sendMessage" in caplog.text
    assert TOKEN not in caplog.text
    assert "AAHsecretTOKEN" not in caplog.text


async def test_token_never_logged_on_failures(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    bot, seen = _bot(lambda _r: httpx.Response(503, json={"ok": False, "description": "busy"}))
    assert await bot.send_error("boom", bypass_rate_limit=True) is False
    assert len(seen) == 3, "5xx is retried"
    assert "HTTP 503 busy" in caplog.text
    assert TOKEN not in caplog.text

    def _transport_error(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"connect failed for https://api.telegram.org/bot{TOKEN}/x")

    bot2, _ = _bot(_transport_error)
    assert await bot2.send_error("boom", bypass_rate_limit=True) is False
    assert TOKEN not in caplog.text


async def test_permanent_4xx_is_not_retried(caplog: pytest.LogCaptureFixture) -> None:
    """Audit repro: a 400 'can't parse entities' was retried 3x with sleeps."""
    bot, seen = _bot(
        lambda _r: httpx.Response(
            400, json={"ok": False, "description": "Bad Request: can't parse entities"}
        )
    )
    assert await bot.send_error("x", bypass_rate_limit=True) is False
    assert len(seen) == 1
    assert "not retrying" in caplog.text


async def test_html_special_chars_are_escaped_and_delivered() -> None:
    def _strict(request: httpx.Request) -> httpx.Response:
        text = json.loads(request.read())["text"]
        # Telegram HTML parse mode rejects a bare '<' / '&'.
        if "<Response" in text or "& " in text:
            return httpx.Response(400, json={"ok": False, "description": "can't parse"})
        return httpx.Response(200, json={"ok": True})

    bot, seen = _bot(_strict)
    ok = await bot.send_error("fatal_error: <Response [503]> & retry", bypass_rate_limit=True)
    assert ok is True
    assert "&lt;Response [503]&gt; &amp; retry" in seen[0]["text"]
    assert seen[0]["parse_mode"] == "HTML"


def test_config_repr_hides_token() -> None:
    assert TOKEN not in repr(TelegramConfig(bot_token=TOKEN, chat_id="1"))


def test_redact_token_helper() -> None:
    assert redact_token(f"POST https://api.telegram.org/bot{TOKEN}/sendMessage") == (
        "POST https://api.telegram.org/bot<redacted>/sendMessage"
    )


# ------------------------------------------------------------- ErrorHandler


def test_transient_error_alerts_once_per_burst() -> None:
    on_alert = MagicMock()
    handler = ErrorHandler(retry_config=RetryConfig(max_retries=5), on_alert=on_alert)
    handler.handle_error(ConnectionError("venue connection lost"), operation="connectivity")
    handler.handle_error(ConnectionError("venue connection lost"), operation="connectivity")
    assert on_alert.call_count == 1
    alert_type, ctx = on_alert.call_args[0]
    assert alert_type == "transient_error"
    assert ctx.category == ErrorCategory.TRANSIENT


def test_transient_count_resets_after_quiet_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three disconnects a day apart must not add up to a fatal halt."""
    now = [1000.0]
    monkeypatch.setattr("vibe_quant.paper.errors.time.monotonic", lambda: now[0])
    on_halt = MagicMock()
    handler = ErrorHandler(
        retry_config=RetryConfig(max_retries=3, reset_after_secs=600), on_halt=on_halt
    )
    for _ in range(3):
        ctx = handler.handle_error(ConnectionError("lost"), operation="connectivity")
        assert ctx.retry_count == 1
        now[0] += 86_400
    on_halt.assert_not_called()
    # A real burst (3 within the window) still escalates to fatal -> halt.
    for _ in range(3):
        handler.handle_error(ConnectionError("lost"), operation="connectivity")
        now[0] += 5
    on_halt.assert_called_once()
