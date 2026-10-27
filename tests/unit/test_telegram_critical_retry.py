"""A critical alert whose send failed past its attempts is retried later (bead vibe-quant-084l4).

``TelegramBot`` gives up after 3 attempts (sustained 429, network loss), and the alert was
gone: HALT, kill switch and the monitor state alerts alike. A CRITICAL alert (journal
``critical=True``) that exhausted its attempts is now owed: the journal keeps it in a small
bounded queue and the control tick (``check_health``) retries it, at most once per
``FAILED_ALERT_RETRY_S``, until it lands, ages out (``FAILED_ALERT_MAX_AGE_S``) or the queue
overflows (``FAILED_ALERT_QUEUE_MAX``, oldest first). Never duplicated, never retried for a
non-critical alert or a rejection (400/401/403/404), and a retry yields to a newer state of
the same monitor so the chat ends on the CURRENT state.

Every test drives a real ``TelegramBot`` over ``httpx.MockTransport`` and a real
``SessionJournal``; the journal's monotonic clock is moved only by the test.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest

from tests.unit.test_paper_journal import _FakeEventLog
from tests.unit.test_paper_recovery_alert_delivery import _clock, _halt_attempts, _Telegram429
from vibe_quant.alerts import telegram as telegram_module
from vibe_quant.alerts.telegram import AlertType, TelegramBot, TelegramConfig
from vibe_quant.paper import journal as journal_module
from vibe_quant.paper.journal import Alert, SessionJournal

if TYPE_CHECKING:
    from collections.abc import Callable

    from vibe_quant.paper.event_log import EventLogWorker

_TRADER_ID = "PAPER-007"
_HALT = "HALT (max_drawdown): probe"
_LOST = "venue connection lost"
_RESTORED = "venue connection restored"


class _Chat:
    """A real TelegramBot on a scripted chat: ``script(attempt_number, text) -> status``."""

    def __init__(self, script: Callable[[int, str], int] = lambda _n, _t: 200) -> None:
        self.script = script
        self.retry_after = 0.001
        self.attempts: list[tuple[int, str]] = []
        self.accepted: list[str] = []

    def bot(self) -> TelegramBot:
        def record(request: httpx.Request) -> httpx.Response:
            text = json.loads(request.content)["text"].split("\n", 1)[1]
            status = self.script(len(self.attempts) + 1, text)
            self.attempts.append((status, text))
            if status == 200:
                self.accepted.append(text)
                return httpx.Response(200, json={"ok": True})
            if status == 0:
                raise httpx.ConnectError("network down")
            body: dict[str, Any] = {"ok": False, "error_code": status, "description": "no"}
            if status == 429:
                body["parameters"] = {"retry_after": self.retry_after}
            return httpx.Response(status, json=body)

        bot = TelegramBot(TelegramConfig(bot_token="T0K", chat_id="42"))

        async def client() -> httpx.AsyncClient:
            return httpx.AsyncClient(transport=httpx.MockTransport(record))

        bot._get_client = client  # type: ignore[method-assign]
        return bot

    def tries(self, text: str) -> int:
        return sum(t == text for _s, t in self.attempts)


def _fail_first(count: int, status: int = 429) -> Callable[[int, str], int]:
    return lambda n, _t: status if n <= count else 200


def _journal(bot: TelegramBot | None) -> SessionJournal:
    journal = SessionJournal(_TRADER_ID, cast("EventLogWorker", _FakeEventLog([])))
    journal.attach(None, bot, "s")
    return journal


async def _tick(journal: SessionJournal, now: list[float], at: float) -> None:
    now[0] = at
    journal.check_health()
    await journal.wait_alerts(None)


def _halt(text: str = _HALT) -> Alert:
    return Alert("circuit_breaker", text, True)


def _critical(text: str) -> Alert:
    """A critical alert that is NOT the guard state (it never replaces a HALT)."""
    return Alert("error", text, True)


def _monitor(text: str, key: str) -> Alert:
    return Alert("error", text, True, monitor=key)


@pytest.fixture()
def fast_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bot's 1 s / 2 s backoff (network failures) costs no wall-clock time."""
    real_sleep = asyncio.sleep

    async def sleep(_delay: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", sleep)


# ------------------------------------------------------------------ TelegramBot seam


async def test_exhausted_critical_send_is_reported_once_per_message() -> None:
    chat = _Chat(lambda _n, _t: 429)
    bot = chat.bot()
    assert await bot.send_circuit_breaker("HALT (kill_switch): a") is False
    assert await bot.send_error("venue lost", bypass_rate_limit=True) is False
    assert len(chat.attempts) == 6
    assert bot.take_exhausted("HALT (kill_switch): a") is True
    assert bot.take_exhausted("HALT (kill_switch): a") is False
    assert bot.take_exhausted("venue lost") is True
    assert bot.take_exhausted("never sent") is False


async def test_only_an_exhausted_critical_send_is_reported() -> None:
    # A rejection is permanent; a success and a non-critical type never count.
    rejected = _Chat(lambda _n, _t: 400)
    bot = rejected.bot()
    assert await bot.send_circuit_breaker("bad chat id") is False
    assert len(rejected.attempts) == 1
    assert bot.take_exhausted("bad chat id") is False

    fine = _Chat()
    bot = fine.bot()
    assert await bot.send_error("ok") is True
    assert bot.take_exhausted("ok") is False

    down = _Chat(lambda _n, _t: 503)
    bot = down.bot()
    assert await bot.send_alert(AlertType.DAILY_SUMMARY, "daily") is False
    assert bot.take_exhausted("daily") is False


async def test_bounds_are_the_documented_ones() -> None:
    assert telegram_module.EXHAUSTED_MAX == 64
    assert journal_module.FAILED_ALERT_QUEUE_MAX == 10
    assert journal_module.FAILED_ALERT_RETRY_S == 30.0


async def test_exhausted_record_is_bounded() -> None:
    chat = _Chat(lambda _n, _t: 429)
    bot = chat.bot()
    for index in range(telegram_module.EXHAUSTED_MAX + 6):
        await bot.send_circuit_breaker(f"HALT {index}")
    assert bot.take_exhausted("HALT 0") is False, "the oldest is evicted"
    assert bot.take_exhausted(f"HALT {telegram_module.EXHAUSTED_MAX + 5}") is True


# ------------------------------------------------------------ delivered later, once


@pytest.mark.parametrize("status", [429, 503, 0])
async def test_halt_failing_past_three_attempts_is_delivered_once_later(
    status: int, monkeypatch: pytest.MonkeyPatch, fast_sleep: None
) -> None:
    now = _clock(monkeypatch)
    chat = _Chat(_fail_first(4, status))
    journal = _journal(chat.bot())

    journal.alert(_halt())
    await journal.wait_alerts(None)
    assert chat.accepted == [] and chat.tries(_HALT) == 3, "the first send exhausted 3 attempts"

    await _tick(journal, now, 1_029.999)
    assert chat.tries(_HALT) == 3, "owed, but the retry interval has not passed"

    await _tick(journal, now, 1_030.0)
    assert chat.accepted == [_HALT]
    assert chat.tries(_HALT) == 5

    for at in (1_031.0, 1_100.0, 5_000.0):
        await _tick(journal, now, at)
    assert chat.accepted == [_HALT], "delivered exactly once"
    assert chat.tries(_HALT) == 5


async def test_retry_that_fails_again_waits_a_full_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _clock(monkeypatch)
    chat = _Chat(_fail_first(6))
    journal = _journal(chat.bot())

    journal.alert(_halt())
    await journal.wait_alerts(None)
    await _tick(journal, now, 1_030.0)
    assert chat.accepted == [] and chat.tries(_HALT) == 6, "the retry also exhausted"

    await _tick(journal, now, 1_059.999)
    assert chat.tries(_HALT) == 6
    await _tick(journal, now, 1_060.0)
    assert chat.accepted == [_HALT] and chat.tries(_HALT) == 7


async def test_retry_traffic_is_bounded_while_the_chat_stays_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _clock(monkeypatch)
    chat = _Chat(lambda _n, _t: 429)
    journal = _journal(chat.bot())

    journal.alert(_halt())
    await journal.wait_alerts(None)
    for second in range(1, 601):
        await _tick(journal, now, 1_000.0 + second)
    # 3 attempts at once, then 3 per 30 s round: 20 rounds in 600 s.
    assert chat.tries(_HALT) == 3 + 3 * 20


# ------------------------------------------------------ what is never retried


async def test_non_critical_alert_is_never_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    now = _clock(monkeypatch)
    chat = _Chat(lambda _n, _t: 429)
    journal = _journal(chat.bot())

    journal.alert(Alert("error", "note", False))
    await journal.wait_alerts(None)
    for at in (1_030.0, 2_000.0, 9_000.0):
        await _tick(journal, now, at)
    assert chat.tries("note") == 3


async def test_rejected_critical_alert_is_never_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    now = _clock(monkeypatch)
    chat = _Chat(lambda _n, _t: 400)
    journal = _journal(chat.bot())

    journal.alert(_halt())
    await journal.wait_alerts(None)
    for at in (1_030.0, 2_000.0):
        await _tick(journal, now, at)
    assert chat.tries(_HALT) == 1


async def test_stand_in_bot_without_a_failure_record_is_not_queued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _clock(monkeypatch)
    calls: list[str] = []

    class _Down:
        async def send_circuit_breaker(self, message: str, *, bypass_rate_limit: bool) -> bool:
            calls.append(message)
            return False

        async def close(self) -> None:
            pass

    journal = _journal(cast("TelegramBot", _Down()))
    journal.alert(_halt())
    await journal.wait_alerts(None)
    await _tick(journal, now, 2_000.0)
    assert calls == [_HALT]


# --------------------------------------------------------------- bounds and order


async def test_same_alert_failing_twice_is_owed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    now = _clock(monkeypatch)
    state = {"up": False}
    chat = _Chat(lambda _n, _t: 200 if state["up"] else 429)
    journal = _journal(chat.bot())

    journal.alert(_halt())
    journal.alert(_halt())
    await journal.wait_alerts(None)
    state["up"] = True
    await _tick(journal, now, 1_030.0)
    await _tick(journal, now, 1_100.0)
    assert chat.accepted == [_HALT]


async def test_queue_keeps_the_newest_alerts_and_sends_oldest_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _clock(monkeypatch)
    state = {"up": False}
    chat = _Chat(lambda _n, _t: 200 if state["up"] else 429)
    journal = _journal(chat.bot())
    size = journal_module.FAILED_ALERT_QUEUE_MAX

    texts = [f"ADL {index}" for index in range(size + 2)]
    for text in texts:
        journal.alert(_critical(text))
        await journal.wait_alerts(None)
    state["up"] = True
    await _tick(journal, now, 1_030.0)
    assert chat.accepted == texts[2:], "the two oldest were dropped, the rest keep their order"


@pytest.mark.parametrize(("at", "delivered"), [(4_600.0, True), (4_600.001, False)])
async def test_owed_alert_ages_out_after_the_max_age_edge(
    at: float, delivered: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = _clock(monkeypatch)
    state = {"up": False}
    chat = _Chat(lambda _n, _t: 200 if state["up"] else 429)
    journal = _journal(chat.bot())
    assert journal_module.FAILED_ALERT_MAX_AGE_S == 3_600.0

    journal.alert(_critical("ADL"))  # fails at t = 1000.0
    await journal.wait_alerts(None)
    state["up"] = True
    await _tick(journal, now, at)
    assert chat.accepted == (["ADL"] if delivered else [])
    await _tick(journal, now, at + 100)
    assert len(chat.accepted) == (1 if delivered else 0)


async def test_a_newer_alert_is_never_held_behind_the_owed_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _clock(monkeypatch)
    state = {"up": False}
    chat = _Chat(lambda _n, _t: 200 if state["up"] else 429)
    journal = _journal(chat.bot())

    journal.alert(_critical("ADL old"))
    await journal.wait_alerts(None)
    state["up"] = True
    now[0] = 1_030.0
    journal.check_health()  # starts the retry of the old one ...
    journal.alert(_halt("HALT new"))  # ... and a new one fires in the same turn
    await journal.wait_alerts(None)
    assert chat.accepted == ["HALT new", "ADL old"], "the fresh HALT goes first"
    assert chat.tries("HALT new") == 1


async def test_halt_is_first_try_after_a_flap_and_the_chat_ends_on_the_current_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _clock(monkeypatch)
    chat = _Telegram429()
    journal = _journal(chat.bot())

    for _ in range(30):
        journal.alert(_monitor(_LOST, "venue:lost"))
        now[0] += 0.5
        journal.alert(_monitor(_RESTORED, "venue:restored"))
        now[0] += 0.5
        journal.check_health()
    journal.alert(_monitor(_LOST, "venue:lost"))  # the flap ends DOWN
    journal.alert(_halt())
    await journal.wait_alerts(None)
    assert _halt_attempts(chat) == [200], "HALT is delivered on its first try"

    async def close_store(_worker: object, close_writer: Callable[[], None], _name: str) -> None:
        close_writer()

    await journal.close(close_store)
    venue = [t for t in chat.accepted if t in (_LOST, _RESTORED)]
    assert venue[-1] == _LOST
    assert len(venue) <= 4


async def test_halt_lost_to_an_exhausted_chat_budget_arrives_once_when_it_reopens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """20/min chat (3 here), budget used up by other traffic: HALT exhausts 3 attempts at
    t=1000, the 30 s retry still hits the window, the 60 s retry lands - exactly once."""
    now = _clock(monkeypatch)
    sent_at: list[float] = []
    chat = _Chat()
    limit = 3

    def script(_n: int, _text: str) -> int:
        in_window = [t for t in sent_at if now[0] - t < 60.0]
        if len(in_window) >= limit:
            return 429
        sent_at.append(now[0])
        return 200

    chat.script = script
    bot = chat.bot()
    journal = _journal(bot)
    for index in range(limit):
        assert await bot.send_error(f"other {index}", bypass_rate_limit=True)

    journal.alert(_halt())
    await journal.wait_alerts(None)
    assert chat.tries(_HALT) == 3 and _HALT not in chat.accepted
    await _tick(journal, now, 1_030.0)
    assert _HALT not in chat.accepted and chat.tries(_HALT) == 6
    await _tick(journal, now, 1_059.999)
    assert chat.tries(_HALT) == 6
    await _tick(journal, now, 1_060.0)
    assert chat.accepted.count(_HALT) == 1
    await _tick(journal, now, 1_200.0)
    assert chat.accepted.count(_HALT) == 1


# ------------------------------------------------- the chat ends on the CURRENT state


async def test_owed_lost_is_dropped_once_restored_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    now = _clock(monkeypatch)
    chat = _Chat(lambda n, _t: 429 if n <= 3 else 200)
    journal = _journal(chat.bot())

    journal.alert(_monitor(_LOST, "venue:lost"))
    await journal.wait_alerts(None)
    assert chat.accepted == []
    now[0] = 1_061.0  # the "restored" window is open (its own key)
    journal.alert(_monitor(_RESTORED, "venue:restored"))
    await journal.wait_alerts(None)
    assert chat.accepted == [_RESTORED]
    for at in (1_100.0, 2_000.0):
        await _tick(journal, now, at)
    assert chat.accepted == [_RESTORED], "the stale 'lost' is never delivered after it"


async def test_owed_lost_is_retried_while_restored_is_held_then_restored_follows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _clock(monkeypatch)
    chat = _Chat(lambda n, _t: 429 if 2 <= n <= 4 else 200)
    journal = _journal(chat.bot())

    now[0] = 990.0
    journal.alert(_monitor(_RESTORED, "venue:restored"))  # an earlier episode; window to 1050
    await journal.wait_alerts(None)
    now[0] = 1_000.0
    journal.alert(_monitor(_LOST, "venue:lost"))  # 3 attempts fail: owed
    await journal.wait_alerts(None)
    assert chat.accepted == [_RESTORED]
    now[0] = 1_010.0
    journal.alert(_monitor(_RESTORED, "venue:restored"))  # window closed: held
    await journal.wait_alerts(None)
    assert chat.accepted == [_RESTORED]

    await _tick(journal, now, 1_030.0)
    assert chat.accepted == [_RESTORED, _LOST], "the owed 'lost' is delivered first"
    await _tick(journal, now, 1_049.999)
    assert chat.accepted == [_RESTORED, _LOST]
    await _tick(journal, now, 1_050.0)
    assert chat.accepted == [_RESTORED, _LOST, _RESTORED], "the chat ends on restored"
    await _tick(journal, now, 2_000.0)
    assert chat.accepted == [_RESTORED, _LOST, _RESTORED]


async def test_owed_lost_still_current_is_delivered_and_a_held_duplicate_is_not_resent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _clock(monkeypatch)
    chat = _Chat(lambda n, _t: 429 if n <= 3 else 200)
    journal = _journal(chat.bot())

    journal.alert(_monitor(_LOST, "venue:lost"))  # 3 attempts fail: owed
    await journal.wait_alerts(None)
    now[0] = 1_010.0
    journal.alert(_monitor(_LOST, "venue:lost"))  # inside the window: held duplicate
    await _tick(journal, now, 1_030.0)
    await _tick(journal, now, 1_100.0)
    assert chat.accepted == [_LOST], "the chat ends on lost, delivered once"


async def test_restored_sent_over_an_owed_lost_and_a_held_lost_ends_on_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _clock(monkeypatch)
    chat = _Chat(lambda n, _t: 429 if n <= 3 else 200)
    journal = _journal(chat.bot())

    journal.alert(_monitor(_LOST, "venue:lost"))  # owed
    await journal.wait_alerts(None)
    now[0] = 1_010.0
    journal.alert(_monitor(_LOST, "venue:lost"))  # held (window closed)
    journal.alert(_monitor(_RESTORED, "venue:restored"))  # newer state: replaces both
    await journal.wait_alerts(None)
    for at in (1_100.0, 1_200.0):
        await _tick(journal, now, at)
    assert chat.accepted == [_RESTORED]


async def test_retry_that_fails_while_a_newer_state_goes_out_is_not_put_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The "lost" retry is in flight (its 3 attempts all fail) when "restored" is sent:
    the failed retry must not come back later and end the chat on a stale "lost"."""
    now = _clock(monkeypatch)
    holder: list[SessionJournal] = []

    def script(n: int, text: str) -> int:
        if n == 4:  # the first attempt of the retry round: the venue recovers meanwhile
            holder[0].alert(_monitor(_RESTORED, "venue:restored"))
        return 200 if text == _RESTORED else 429

    chat = _Chat(script)
    journal = _journal(chat.bot())
    holder.append(journal)

    journal.alert(_monitor(_LOST, "venue:lost"))
    await journal.wait_alerts(None)
    await _tick(journal, now, 1_030.0)
    assert chat.accepted == [_RESTORED]
    for at in (1_060.0, 1_100.0, 3_000.0):
        await _tick(journal, now, at)
    assert chat.accepted == [_RESTORED]
    assert chat.tries(_LOST) == 4, "the superseded retry stops after its attempt in flight"


async def test_a_superseded_owed_alert_frees_its_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    now = _clock(monkeypatch)
    state = {"up": False}
    chat = _Chat(lambda _n, text: 200 if state["up"] or text == _RESTORED else 429)
    journal = _journal(chat.bot())
    size = journal_module.FAILED_ALERT_QUEUE_MAX

    halts = [f"ADL {index}" for index in range(size)]
    for text in halts[: size - 1]:
        journal.alert(_critical(text))
        await journal.wait_alerts(None)
    journal.alert(_monitor(_LOST, "venue:lost"))  # owed: the youngest of the full queue
    await journal.wait_alerts(None)
    journal.alert(_monitor(_RESTORED, "venue:restored"))  # supersedes "lost": slot is free
    await journal.wait_alerts(None)
    journal.alert(_critical(halts[-1]))
    await journal.wait_alerts(None)
    state["up"] = True
    await _tick(journal, now, 1_030.0)
    assert sorted(chat.accepted) == sorted([_RESTORED, *halts]), "no alert was pushed out"


# ------------------------------------------------------------------------- shutdown


async def test_shutdown_makes_one_last_attempt_at_an_owed_alert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _clock(monkeypatch)
    chat = _Chat(lambda n, _t: 429 if n <= 3 else 200)
    journal = _journal(chat.bot())

    async def close_store(_worker: object, close_writer: Callable[[], None], _name: str) -> None:
        close_writer()

    journal.alert(_halt())
    await journal.wait_alerts(None)
    now[0] = 1_005.0  # inside the retry interval: only shutdown forces the attempt
    await journal.close(close_store)
    assert chat.accepted == [_HALT]


async def test_shutdown_with_the_chat_still_down_ends_without_a_hang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clock(monkeypatch)
    chat = _Chat(lambda _n, _t: 429)
    journal = _journal(chat.bot())

    async def close_store(_worker: object, close_writer: Callable[[], None], _name: str) -> None:
        close_writer()

    journal.alert(_halt())
    await journal.wait_alerts(None)
    await asyncio.wait_for(journal.close(close_store), 10)
    assert chat.accepted == []


# ------------------------------------------------------------------------ log hygiene


async def test_failing_retry_logs_the_class_only(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    now = _clock(monkeypatch)
    chat = _Chat(lambda _n, _t: 429)
    bot = chat.bot()
    journal = _journal(bot)
    journal.alert(_halt())
    await journal.wait_alerts(None)

    async def boom(
        message: str, *, bypass_rate_limit: bool = False, current: Callable[[], bool] | None = None
    ) -> bool:
        raise RuntimeError("ZZLEAK1 bot T0K") from ValueError("ZZLEAK2")

    monkeypatch.setattr(bot, "send_circuit_breaker", boom)
    with caplog.at_level(logging.DEBUG):
        await _tick(journal, now, 1_030.0)
    assert "ZZLEAK" not in caplog.text and "T0K" not in caplog.text
    assert all(r.exc_info is None for r in caplog.records)
    assert any("RuntimeError" in r.getMessage() for r in caplog.records)


# ---------------------------------------------- bot: a replaced send stops its retries


async def test_replaced_send_stops_before_its_next_attempt_but_never_before_the_first() -> None:
    chat = _Chat(lambda _n, _t: 429)
    bot = chat.bot()
    assert await bot.send_error("a", bypass_rate_limit=True, current=lambda: False) is False
    assert chat.tries("a") == 1, "the first attempt is never skipped"
    assert bot.take_exhausted("a") is False, "a replaced send is not owed"

    checks = [True, False]
    assert await bot.send_error("b", bypass_rate_limit=True, current=lambda: checks.pop(0)) is False
    assert chat.tries("b") == 2

    assert await bot.send_error("c", bypass_rate_limit=True, current=lambda: True) is False
    assert chat.tries("c") == 3 and bot.take_exhausted("c") is True


# --------------------------------------------- fixes after review (084l4 round 2)


async def test_retry_that_succeeds_after_the_state_recovered_does_not_end_the_chat_on_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _clock(monkeypatch)
    holder: list[SessionJournal] = []
    lost_tries = 0

    def script(_n: int, text: str) -> int:
        nonlocal lost_tries
        if text == _RESTORED:
            return 200
        lost_tries += 1
        if lost_tries == 4:  # first attempt of the retry: the venue recovers meanwhile
            holder[0].alert(_monitor(_RESTORED, "venue:restored"))
        return 429 if lost_tries <= 4 else 200

    chat = _Chat(script)
    journal = _journal(chat.bot())
    holder.append(journal)
    journal.alert(_monitor(_LOST, "venue:lost"))
    await journal.wait_alerts(None)
    await _tick(journal, now, 1_030.0)
    assert chat.accepted == [_RESTORED], "the older retry never lands after the newer state"
    await _tick(journal, now, 2_000.0)
    assert chat.accepted == [_RESTORED]


@pytest.mark.parametrize("make", [_halt, _critical])
async def test_a_repeat_that_was_delivered_clears_the_owed_copy(
    make: Callable[[str], Alert], monkeypatch: pytest.MonkeyPatch
) -> None:
    now = _clock(monkeypatch)
    chat = _Chat(lambda n, _t: 429 if n <= 3 else 200)
    journal = _journal(chat.bot())
    journal.alert(make("X"))
    await journal.wait_alerts(None)  # exhausted: owed
    journal.alert(make("X"))  # the same text again, delivered
    await journal.wait_alerts(None)
    assert chat.accepted == ["X"]
    for at in (1_030.0, 1_100.0):
        await _tick(journal, now, at)
    assert chat.accepted == ["X"], "the owed copy was retired by the delivery"


async def test_a_repeat_that_overlaps_a_running_retry_is_not_sent_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _clock(monkeypatch)
    holder: list[SessionJournal] = []

    def script(n: int, text: str) -> int:
        if n == 4:  # first attempt of the owed copy's retry: the same text is sent afresh
            holder[0].alert(_critical("X"))
            return 429
        return 429 if n <= 3 else 200

    chat = _Chat(script)
    journal = _journal(chat.bot())
    holder.append(journal)
    journal.alert(_critical("X"))
    await journal.wait_alerts(None)
    await _tick(journal, now, 1_030.0)
    await _tick(journal, now, 1_100.0)
    assert chat.accepted == ["X"]


async def test_a_new_halt_goes_before_the_owed_backlog_spends_the_chat_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _clock(monkeypatch)
    state = {"up": False, "used": 0}

    def script(_n: int, _text: str) -> int:
        if not state["up"] or state["used"] >= 20:  # Telegram: 20 messages a minute
            return 429
        state["used"] += 1
        return 200

    chat = _Chat(script)
    bot = chat.bot()
    journal = _journal(bot)
    for index in range(10):
        journal.alert(_critical(f"ADL old {index}"))
        await journal.wait_alerts(None)
    state["up"] = True
    for index in range(11):
        assert await bot.send_error(f"other {index}", bypass_rate_limit=True)
    now[0] = 1_030.0
    journal.check_health()  # the backlog round starts ...
    journal.alert(_halt("HALT new"))  # ... and a HALT fires in the same turn
    await journal.wait_alerts(None)
    assert chat.tries("HALT new") == 1 and "HALT new" in chat.accepted
    assert chat.accepted.index("HALT new") == 11, "right after the 11 other messages"


async def test_a_retry_failing_while_newer_alerts_fill_the_queue_respects_the_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _clock(monkeypatch)
    holder: list[SessionJournal] = []
    release = asyncio.Event()
    handled = {"old": 0, "new": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        text = json.loads(request.content)["text"].split("\n", 1)[1]
        if text == "ADL old":
            handled["old"] += 1
            if handled["old"] == 4:  # the retry's first attempt: ten newer alerts fail meanwhile
                for index in range(10):
                    holder[0].alert(_critical(f"ADL new {index}"))
                await release.wait()
        else:
            handled["new"] += 1
            if handled["new"] == 30:
                release.set()
        return httpx.Response(
            429, json={"ok": False, "error_code": 429, "parameters": {"retry_after": 0.001}}
        )

    bot = TelegramBot(TelegramConfig(bot_token="T0K", chat_id="42"))

    async def client() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    bot._get_client = client  # type: ignore[method-assign]
    journal = _journal(bot)
    holder.append(journal)
    journal.alert(_critical("ADL old"))
    await journal.wait_alerts(None)
    await _tick(journal, now, 1_030.0)
    assert sorted(journal._owed) == sorted(f"error:ADL new {i}" for i in range(10))
    for at in (1_061.0, 1_100.0):
        await _tick(journal, now, at)
    assert len(journal._owed) <= journal_module.FAILED_ALERT_QUEUE_MAX


async def test_overflow_never_drops_the_owed_kill_state(monkeypatch: pytest.MonkeyPatch) -> None:
    now = _clock(monkeypatch)
    state = {"up": False}
    chat = _Chat(lambda _n, _t: 200 if state["up"] else 429)
    journal = _journal(chat.bot())
    kill = "HALT (kill_switch): kill switch engaged"
    journal.alert(_halt(kill))
    await journal.wait_alerts(None)
    size = journal_module.FAILED_ALERT_QUEUE_MAX
    adl = [f"EXCHANGE ADL {index}" for index in range(size)]
    for text in adl:
        journal.alert(_critical(text))
        await journal.wait_alerts(None)
    state["up"] = True
    await _tick(journal, now, 1_030.0)
    assert chat.accepted == [kill, *adl[1:]], "the oldest ADL went, the kill state stayed"


async def test_a_queue_of_only_guard_state_is_kept_and_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    now = _clock(monkeypatch)
    monkeypatch.setattr(journal_module, "FAILED_ALERT_QUEUE_MAX", 0)
    state = {"up": False}
    chat = _Chat(lambda _n, _t: 200 if state["up"] else 429)
    journal = _journal(chat.bot())
    with caplog.at_level(logging.WARNING):
        journal.alert(_halt())
        await journal.wait_alerts(None)
    assert len(journal._owed) == 1
    assert any("none dropped" in r.getMessage() for r in caplog.records)
    state["up"] = True
    await _tick(journal, now, 1_030.0)
    assert chat.accepted == [_HALT]


async def test_the_chat_ends_on_resumed_not_on_an_owed_halt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _clock(monkeypatch)
    state = {"up": False}
    chat = _Chat(lambda _n, _t: 200 if state["up"] else 429)
    journal = _journal(chat.bot())
    journal.alert(_halt())
    await journal.wait_alerts(None)  # HALT owed
    state["up"] = True
    now[0] = 1_010.0
    journal.alert(_halt("RESUMED: operator resume"))
    await journal.wait_alerts(None)
    for at in (1_031.0, 1_100.0):
        await _tick(journal, now, at)
    assert chat.accepted == ["RESUMED: operator resume"]


async def test_a_newer_halt_replaces_an_owed_older_one(monkeypatch: pytest.MonkeyPatch) -> None:
    now = _clock(monkeypatch)
    state = {"up": False}
    chat = _Chat(lambda _n, _t: 200 if state["up"] else 429)
    journal = _journal(chat.bot())
    kill = "HALT (kill_switch): kill switch engaged"
    journal.alert(_halt())
    await journal.wait_alerts(None)
    journal.alert(_halt(kill))
    await journal.wait_alerts(None)
    assert list(journal._owed) == ["state:guard"] and len(journal._owed) == 1
    state["up"] = True
    await _tick(journal, now, 1_030.0)
    assert chat.accepted == [kill]


@pytest.mark.parametrize("at", [4_600.0, 4_600.001, 90_000.0])
async def test_the_owed_guard_state_never_ages_out(
    at: float, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    now = _clock(monkeypatch)
    state = {"up": False}
    chat = _Chat(lambda _n, _t: 200 if state["up"] else 429)
    journal = _journal(chat.bot())
    journal.alert(_halt())
    await journal.wait_alerts(None)
    state["up"] = True
    with caplog.at_level(logging.WARNING):
        await _tick(journal, now, at)
    assert chat.accepted == [_HALT]
    assert not any("dropped" in r.getMessage() for r in caplog.records)


async def _close(journal: SessionJournal) -> None:
    async def close_store(_worker: object, close_writer: Callable[[], None], _name: str) -> None:
        close_writer()

    await journal.close(close_store)


def _abandoned(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "abandoned" in r.getMessage()]


async def test_shutdown_retries_an_alert_that_fails_while_it_waits_and_logs_it_abandoned(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _clock(monkeypatch)
    chat = _Chat(lambda _n, _t: 429)
    journal = _journal(chat.bot())
    journal.alert(_halt("HALT (max_drawdown): drawdown 5.20% exceeds the limit by far"))
    with caplog.at_level(logging.WARNING):
        await _close(journal)  # the HALT is still being sent when teardown starts
    text = "HALT (max_drawdown): drawdown 5.20% exceeds the limit by far"
    assert chat.tries(text) == 6, "3 attempts in flight, then one final retry of 3"
    assert _abandoned(caplog) == [
        "circuit_breaker alert abandoned at shutdown, never delivered (guard state)"
    ]


async def test_shutdown_tries_every_owed_alert_and_logs_only_the_undelivered(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _clock(monkeypatch)
    chat = _Chat()
    chat.script = lambda _n, text: 429 if text != "ADL a" or chat.tries(text) < 3 else 200
    journal = _journal(chat.bot())
    for text in ("ADL b", "ADL a", "ADL c"):
        journal.alert(_critical(text))
        await journal.wait_alerts(None)
    assert len(journal._owed) == 3
    with caplog.at_level(logging.WARNING):
        await _close(journal)
    assert chat.accepted == ["ADL a"], "a failing first alert does not stop the others"
    assert all(chat.tries(t) == 6 for t in ("ADL b", "ADL c"))
    assert (
        _abandoned(caplog)
        == ["error alert abandoned at shutdown, never delivered (single alert)"] * 2
    )
    assert not journal._owed


async def test_shutdown_final_retry_is_bounded_and_the_cut_off_alert_is_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _clock(monkeypatch)
    monkeypatch.setattr(journal_module, "SHUTDOWN_ALERT_WAIT_S", 0.3)
    chat = _Chat(lambda _n, _t: 429)
    journal = _journal(chat.bot())
    journal.alert(_halt())
    await journal.wait_alerts(None)
    chat.retry_after = 60.0  # Telegram now asks for a minute: the retry would outlast shutdown
    began = time.monotonic()
    with caplog.at_level(logging.WARNING):
        await _close(journal)
    assert time.monotonic() - began < 5
    assert _abandoned(caplog) == [
        "circuit_breaker alert abandoned at shutdown, never delivered (guard state)"
    ]


async def test_shutdown_abandoned_log_never_carries_the_alert_text(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An alert text can be built from an exception (``Paper node failed: {e}``); the
    abandoned line names the type and a fixed label only (R1b)."""
    _clock(monkeypatch)
    chat = _Chat(lambda _n, _t: 429)
    journal = _journal(chat.bot())
    journal.alert(_critical("Paper node failed: apiKey=SECRET123 signature=abc"))
    journal.alert(_monitor("venue connection lost: apiKey=SECRET123", "venue:lost"))
    await journal.wait_alerts(None)
    with caplog.at_level(logging.WARNING):
        await _close(journal)
    assert sorted(_abandoned(caplog)) == [
        "error alert abandoned at shutdown, never delivered (single alert)",
        "error alert abandoned at shutdown, never delivered (venue monitor)",
    ]
    assert not [r for r in caplog.records if "SECRET123" in r.getMessage()]


async def test_shutdown_logs_an_alert_still_hanging_after_the_wait_and_ends(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _clock(monkeypatch)
    monkeypatch.setattr(journal_module, "SHUTDOWN_ALERT_WAIT_S", 0.3)
    never = asyncio.Event()

    async def handler(_request: httpx.Request) -> httpx.Response:
        await never.wait()
        return httpx.Response(200, json={"ok": True})

    bot = TelegramBot(TelegramConfig(bot_token="T0K", chat_id="42"))

    async def client() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    bot._get_client = client  # type: ignore[method-assign]
    journal = _journal(bot)
    journal.alert(_halt())
    began = time.monotonic()
    with caplog.at_level(logging.WARNING):
        await _close(journal)
    assert time.monotonic() - began < 5
    assert _abandoned(caplog) == [
        "circuit_breaker alert abandoned at shutdown, never delivered (guard state)"
    ]
