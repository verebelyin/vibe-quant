"""Shadow alerts: Telegram on state changes only, stamped only when the sender says True.

Design arch-c7-2 §2 Alerts (Q5), contract row R41.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING

import pytest

from vibe_quant.alerts.telegram import AlertType, TelegramBot
from vibe_quant.db.state_manager import StateManager
from vibe_quant.shadow import notify

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_PREREG = "Pre-registered: forward Sharpe > 1 on BTC 4h after costs."
_MARKER = "SEND-EXPLODED-7f3a"


def _row(name: str, *, is_test: bool) -> dict[str, object]:
    return {
        "name": name,
        "source_kind": "strategy",
        "dsl_json": "{}",
        "dsl_sha256": "ab" * 32,
        "compiler_hash": "c0ffee",
        "symbols": ["BTCUSDT"],
        "timeframe": "4h",
        "parameters": {},
        "frozen_identity": {},
        "enrolled_at": "2026-11-22T18:00:00.000000+00:00",
        "first_day": "2026-11-23",
        "run_start": "2026-05-27",
        "is_test": int(is_test),
        "family": None if is_test else "F1",
        "pre_registration": _PREREG,
        "max_dd_limit": 0.2,
        "engine_epoch": 1,
        "engine_epoch_last": 1,
    }


@pytest.fixture
def sm(tmp_path: Path) -> Iterator[StateManager]:
    mgr = StateManager(tmp_path / "state.db")
    live = mgr.insert_shadow_strategy(_row("rsi-btc", is_test=False))
    test = mgr.insert_shadow_strategy(_row("ema-eth", is_test=True))
    mgr.commit_shadow_update(
        live,
        new_days=[],
        fields={},
        events=[
            {"kind": "note", "message": "no new data since 2026-11-30"},
            {
                "kind": "state_change",
                "from_state": "tracking",
                "to_state": "stalled",
                "message": "data gap at 2026-12-01 (BTCUSDT)",
            },
        ],
    )
    mgr.commit_shadow_update(
        test,
        new_days=[],
        fields={},
        events=[
            {
                "kind": "state_change",
                "from_state": "tracking",
                "to_state": "futile",
                # tracker._commit already prefixes a test row's event text
                "message": "[TEST] max drawdown 0.31 > limit 0.2",
            }
        ],
    )
    yield mgr
    mgr.close()


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[tuple[AlertType, str, bool]]:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:fake")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    return []


def _fake_send(sent: list[tuple[AlertType, str, bool]], outcome: object) -> object:
    async def send_alert(
        self: TelegramBot,
        alert_type: AlertType,
        message: str,
        *,
        bypass_rate_limit: bool = False,
        current: object = None,
    ) -> bool:
        sent.append((alert_type, message, bypass_rate_limit))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # type: ignore[return-value]

    return send_alert


EXPECTED = [
    "Shadow 'rsi-btc': tracking -> stalled. data gap at 2026-12-01 (BTCUSDT)",
    "[TEST] Shadow 'ema-eth': tracking -> futile. max drawdown 0.31 > limit 0.2",
]


@pytest.mark.parametrize(
    ("outcome", "stamped"),
    [(False, 0), (RuntimeError(_MARKER), 0), (1, 0), (True, 2)],
    ids=["false", "raises", "truthy-not-true", "true"],
)
def test_alert_bool(
    sm: StateManager,
    sent: list[tuple[AlertType, str, bool]],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    outcome: object,
    stamped: int,
) -> None:
    """R41: False / raise / non-True leave the event unsent; True stamps it once."""
    monkeypatch.setattr(TelegramBot, "send_alert", _fake_send(sent, outcome))
    with caplog.at_level(logging.DEBUG):
        assert notify.send_pending(sm) == stamped
    # both state changes tried, in event order, note never sent, [TEST] exactly once
    assert sent == [(AlertType.INFO, text, True) for text in EXPECTED]
    assert _MARKER not in caplog.text
    unsent = [e["message"] for e in sm.pending_shadow_alerts()]
    assert len(unsent) == 2 - stamped
    sent.clear()
    notify.send_pending(sm)  # a retry sends exactly what is still unsent
    assert len(sent) == 2 - stamped


def test_unconfigured_sends_nothing(sm: StateManager, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[AlertType, str, bool]] = []
    monkeypatch.setattr(TelegramBot, "send_alert", _fake_send(calls, True))
    assert notify.send_pending(sm) == 0
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:fake")  # half-configured: still nothing
    assert notify.send_pending(sm) == 0
    assert calls == []
    assert len(sm.pending_shadow_alerts()) == 2


def test_concurrent_senders_send_each_event_once(
    sm: StateManager, sent: list[tuple[AlertType, str, bool]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """clmmc.24: a second sender while the first is mid-send skips; each alert goes out once."""
    in_send = threading.Event()
    release = threading.Event()
    lock = threading.Lock()

    async def slow_send(
        self: TelegramBot, alert_type: AlertType, message: str, **_: object
    ) -> bool:
        with lock:
            sent.append((alert_type, message, True))
        in_send.set()
        await asyncio.to_thread(release.wait, 10)
        return True

    monkeypatch.setattr(TelegramBot, "send_alert", slow_send)
    first: list[int] = []
    worker = threading.Thread(target=lambda: first.append(notify.send_pending(sm)))
    worker.start()
    assert in_send.wait(10)
    other = StateManager(sm.db_path)  # a second process's view of the same DB
    try:
        assert notify.send_pending(other) == 0
    finally:
        other.close()
        release.set()
        worker.join(10)
    assert first == [2]
    assert sorted(m for _, m, _ in sent) == sorted(EXPECTED)
    assert sm.pending_shadow_alerts() == []
