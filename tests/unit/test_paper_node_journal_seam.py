"""The node's compatibility seam onto ``SessionJournal`` (bead vibe-quant-rabmo.11, T13a).

``PaperTradingNode`` owns ``self._journal``; the attributes the node's own code and the
existing tests touch are properties onto it:

* ``_event_writer`` / ``_telegram`` setters change ONE sink and keep the other two
  (writer, bot, strategy name) - the journal's ``attach`` is all-or-nothing;
* ``_alert_tasks`` is the journal's LIVE alert-task set (the trade-alert path adds to
  it and tests drain it);
* ``_event_log`` rebinds the journal's worker (tests swap in a bounded or raising one);
* the strategy name is pushed once, after ``_load_strategy``, so lines written before
  it carry "unknown" and later lines carry the strategy's name, like the base node;
* ``_write_then_alert`` / ``_awrite_then_alert`` keep their node bodies: the tests that
  patch ``_write_event`` / ``_send_telegram_alert`` keep routing through them.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, cast

import pytest

from tests.unit.test_audit_paper_node import _config
from tests.unit.test_paper_liquidation_detector import (  # noqa: F401
    fake_tradingnode,  # fixture
    seeded,  # fixture
)
from vibe_quant.logging.events import EventType
from vibe_quant.logging.writer import EventWriter
from vibe_quant.paper.event_log import EventLogWorker
from vibe_quant.paper.journal import Alert
from vibe_quant.paper.node import PaperTradingNode

if TYPE_CHECKING:
    from pathlib import Path


class _Bot:
    """A TelegramBot stand-in: records sends, optionally holds them pending."""

    def __init__(self, gate: asyncio.Event | None = None) -> None:
        self.calls: list[tuple[str, str, bool]] = []
        self._gate = gate

    async def _wait(self) -> None:
        if self._gate is not None:
            await self._gate.wait()

    async def send_error(self, message: str, *, bypass_rate_limit: bool = False) -> bool:
        await self._wait()
        self.calls.append(("send_error", message, bypass_rate_limit))
        return True

    async def send_circuit_breaker(
        self, reason: str, details: str = "", *, bypass_rate_limit: bool = False
    ) -> bool:
        message = f"{reason}\n{details}" if details else reason
        await self._wait()
        self.calls.append(("send_circuit_breaker", message, bypass_rate_limit))
        return True


def _node(seeded_data: dict[str, Any]) -> PaperTradingNode:
    return PaperTradingNode(_config(seeded_data))


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _stop(node: PaperTradingNode) -> None:
    node._event_log.worker.begin_stop(lambda: None)


def test_event_writer_setter_installs_and_clears_the_journal_writer(
    seeded: dict[str, Any],  # noqa: F811
    tmp_path: Path,
) -> None:
    """(a) ``_event_writer = w`` writes through the journal; ``None`` stops the writes."""
    node = _node(seeded)
    writer = EventWriter(run_id=node._config.trader_id, base_path=tmp_path / "logs")
    path = tmp_path / "logs" / f"{node._config.trader_id}.jsonl"
    try:
        node._event_writer = writer
        assert node._journal.has_writer is True

        future = node._write_event(EventType.SIGNAL, {"action": "x"})
        assert future is not None
        node._event_log.barrier().result(10)
        lines = _lines(path)
        assert [line["data"] for line in lines] == [{"action": "x"}]
        assert lines[0]["strategy"] == "unknown"

        node._event_writer = None
        assert node._journal.has_writer is False
        assert node._write_event(EventType.SIGNAL, {"action": "late"}) is None
        assert _lines(path) == lines
    finally:
        writer.close()
        _stop(node)


def test_sink_setters_change_one_sink_and_keep_the_others(seeded: dict[str, Any]) -> None:  # noqa: F811
    """(b) the all-or-nothing trap: each setter keeps the other two journal fields."""
    node = _node(seeded)
    bot = _Bot()
    first_writer = cast("EventWriter", object())
    second_writer = cast("EventWriter", object())
    try:
        node._event_writer = first_writer
        node._telegram = bot
        assert node._event_writer is first_writer
        assert node._telegram is bot

        node._event_writer = second_writer
        assert node._event_writer is second_writer
        assert node._telegram is bot

        node._telegram = None
        assert node._telegram is None
        assert node._event_writer is second_writer

        node._event_writer = None
        assert node._event_writer is None
        assert node._telegram is None
    finally:
        _stop(node)


async def test_alert_tasks_is_the_journals_live_set(seeded: dict[str, Any]) -> None:  # noqa: F811
    """(c) the node exposes the journal's set object; an alert lands in it."""
    node = _node(seeded)
    gate = asyncio.Event()
    bot = _Bot(gate)
    node._telegram = bot
    try:
        assert node._alert_tasks is node._journal.alert_tasks

        node._send_telegram_alert("error", "m", critical=True)
        assert node._journal.pending_alerts == 1
        assert len(node._alert_tasks) == 1

        gate.set()
        await node._journal.wait_alerts(None)
        assert node._alert_tasks == set()
        assert node._journal.pending_alerts == 0
        assert bot.calls == [("send_error", "m", True)]
    finally:
        _stop(node)


async def test_strategy_name_is_published_after_the_strategy_loads(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """(d) "unknown" before ``_load_strategy``, the loaded name after - two names."""
    names = ("alpha_probe", "beta_probe")
    node = _node(seeded)
    real = node._load_strategy()
    monkeypatch.setattr(node, "_load_strategy", lambda: real.model_copy(update={"name": names[0]}))

    early_path = tmp_path / "early"
    early_writer = EventWriter(run_id=node._config.trader_id, base_path=early_path)
    try:
        node._event_writer = early_writer
        node._write_event(EventType.SIGNAL, {"action": "early"})
        node._event_log.barrier().result(10)
        early_lines = _lines(early_path / f"{node._config.trader_id}.jsonl")
        assert [line["strategy"] for line in early_lines] == ["unknown"]

        await node._initialize()
        await node._shutdown()
        log_path = node._config.logs_path / f"{node._config.trader_id}.jsonl"
        starts = [line for line in _lines(log_path) if line["data"]["action"] == "start"]
        assert [line["strategy"] for line in starts] == [names[0]]
    finally:
        early_writer.close()

    second = PaperTradingNode(_config(seeded, trader_id="PAPER-008"))
    monkeypatch.setattr(
        second, "_load_strategy", lambda: real.model_copy(update={"name": names[1]})
    )
    try:
        await second._initialize()
        await second._shutdown()
        starts = [
            line
            for line in _lines(second._config.logs_path / f"{second._config.trader_id}.jsonl")
            if line["data"]["action"] == "start"
        ]
        assert [line["strategy"] for line in starts] == [names[1]]
    finally:
        _stop(second)


def test_event_log_rebinding_moves_the_submit(
    seeded: dict[str, Any],  # noqa: F811
    tmp_path: Path,
) -> None:
    """(e) a rebound ``_event_log`` is the worker the journal submits through."""
    node = _node(seeded)
    original = node._event_log
    other = EventLogWorker("vq-paper-events-seam")
    writer = EventWriter(run_id=node._config.trader_id, base_path=tmp_path / "logs")
    try:
        node._event_writer = writer
        node._event_log = other
        assert node._event_log is other

        future = node._write_event(EventType.SIGNAL, {"action": "x"})
        assert future is not None
        other.barrier().result(10)
        lines = _lines(tmp_path / "logs" / f"{node._config.trader_id}.jsonl")
        assert [line["data"] for line in lines] == [{"action": "x"}]
        assert node._journal.event_log is other
    finally:
        other.worker.begin_stop(lambda: None)
        original.worker.begin_stop(lambda: None)
        writer.close()


async def test_composers_still_route_through_the_node_pair(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(f) the patched pair keeps the base order and the same ``critical`` flag."""
    node = _node(seeded)
    order: list[str] = []
    seen: list[tuple[str, object, object]] = []

    def write(event_type: EventType, data: dict[str, object], timestamp: object = None) -> None:
        order.append("write")
        seen.append(("write", event_type, data))

    def alert(kind: str, message: str, *, critical: bool = False) -> None:
        order.append("alert")
        seen.append(("alert", kind, critical))

    monkeypatch.setattr(node, "_write_event", write)
    monkeypatch.setattr(node, "_send_telegram_alert", alert)
    try:
        node._write_then_alert(
            EventType.RISK_CHECK, {"a": 1}, "error", "m", critical=True, timestamp=None
        )
        assert order == ["write", "alert"]
        assert seen == [
            ("write", EventType.RISK_CHECK, {"a": 1}),
            ("alert", "error", True),
        ]

        order.clear()
        seen.clear()
        await node._awrite_then_alert(EventType.SIGNAL, {"b": 2}, "circuit_breaker", "m2")
        assert order == ["write", "alert"]
        assert seen == [
            ("write", EventType.SIGNAL, {"b": 2}),
            ("alert", "circuit_breaker", False),
        ]

        class _Boom(RuntimeError):
            pass

        def raising_write(event_type: EventType, data: dict[str, object], **_kw: object) -> None:
            order.append("write")
            raise _Boom("MARK7")

        monkeypatch.setattr(node, "_write_event", raising_write)
        order.clear()
        with pytest.raises(_Boom):
            node._write_then_alert(EventType.RISK_CHECK, {"c": 3}, "error", "m3")
        assert order == ["write", "alert"]
    finally:
        _stop(node)


async def test_send_telegram_alert_uses_the_journal_alert_shape(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(g) same positional shape and the same bot call as the journal's alert."""
    node = _node(seeded)
    raised: list[Alert] = []
    original = node._journal.alert

    def spy(alert: Alert) -> None:
        raised.append(alert)
        original(alert)

    monkeypatch.setattr(node._journal, "alert", spy)
    try:
        node._send_telegram_alert("circuit_breaker", "m")
        assert raised == [Alert("circuit_breaker", "m", False)]

        bot = _Bot()
        node._telegram = bot
        node._send_telegram_alert("circuit_breaker", "m")
        await node._journal.wait_alerts(None)
        assert bot.calls == [("send_circuit_breaker", "m", False)]
    finally:
        _stop(node)
