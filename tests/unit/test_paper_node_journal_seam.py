"""The node reaches ``SessionJournal`` for every write and alert (bead vibe-quant-rabmo.11).

``PaperTradingNode`` owns ``self._journal``; there are no compatibility properties on the
node any more (T13d). What stays here is the proof that a ``FakeJournal`` injected at
``_journal`` sees what the node writes: the strategy name is published once the strategy
loads, the trace helper shares the fake's live call list, and ``journal.record`` /
``record_durable`` leave the same ``("write", ...)`` / ``("alert", ...)`` tuples in order.
The health alert also goes through ``journal.alert`` (the single funnel, R2), so a patch of
that one method - the T13c point - sees the journal's own alerts.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from tests.unit.test_audit_paper_node import _config
from tests.unit.test_paper_halt_restore import _trace_node
from tests.unit.test_paper_liquidation_detector import (  # noqa: F401
    fake_tradingnode,  # fixture
    seeded,  # fixture
)
from vibe_quant.logging.events import EventType
from vibe_quant.logging.writer import EventWriter
from vibe_quant.paper.event_log import EventLogHealth
from vibe_quant.paper.journal import Alert, FakeJournal
from vibe_quant.paper.node import PaperTradingNode

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _node(seeded_data: dict[str, Any]) -> PaperTradingNode:
    return PaperTradingNode(_config(seeded_data))


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _stop(node: PaperTradingNode) -> None:
    node._journal.event_log.worker.begin_stop(lambda: None)


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
        node._journal.writer = early_writer
        node._journal.record(EventType.SIGNAL, {"action": "early"})
        node._journal.event_log.barrier().result(10)
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


def test_trace_node_injects_a_fake_journal_into_the_node(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(h) ``_trace_node`` swaps the journal and shares its live call list."""
    node, trace = _trace_node(monkeypatch, seeded)
    try:
        assert isinstance(node._journal, FakeJournal)
        assert trace.calls is node._journal.calls
        assert node._journal.record(EventType.SIGNAL, {"action": "x"}) is None
        assert trace.calls == [("write", EventType.SIGNAL, {"action": "x"}, None)]
    finally:
        _stop(node)


def test_recording_an_alert_leaves_the_write_and_alert_tuples(
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """(f) ``record(..., alert=...)`` leaves the write then the alert, in that order."""
    node = _node(seeded)
    fake = FakeJournal()
    node._journal = fake
    try:
        node._journal.record(
            EventType.RISK_CHECK, {"x": 1}, alert=Alert("error", "m", critical=True)
        )
        assert fake.calls == [
            ("write", EventType.RISK_CHECK, {"x": 1}, None),
            ("alert", "error", "m", True),
        ]

        fake.calls.clear()
        node._journal.record(EventType.SIGNAL, {"y": [1, 2]}, alert=Alert("circuit_breaker", "m2"))
        assert fake.calls == [
            ("write", EventType.SIGNAL, {"y": [1, 2]}, None),
            ("alert", "circuit_breaker", "m2", False),
        ]
    finally:
        _stop(node)


async def test_recording_a_durable_alert_leaves_the_write_and_alert_tuples(
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """(f) the durable route leaves the same two tuples in order."""
    node = _node(seeded)
    fake = FakeJournal()
    node._journal = fake
    try:
        await node._journal.record_durable(
            EventType.SIGNAL, {"b": 2}, Alert("circuit_breaker", "m2")
        )
        assert fake.calls == [
            ("write", EventType.SIGNAL, {"b": 2}, None),
            ("alert", "circuit_breaker", "m2", False),
        ]
    finally:
        _stop(node)


def test_record_forwards_the_timestamp_it_was_given(
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """(f) the caller's timestamp reaches the journal unchanged."""
    node = _node(seeded)
    fake = FakeJournal()
    node._journal = fake
    ts = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)
    try:
        node._journal.record(EventType.SIGNAL, {"action": "ts"}, timestamp=ts)
        assert fake.calls == [("write", EventType.SIGNAL, {"action": "ts"}, ts)]
    finally:
        _stop(node)


def test_the_journal_funnel_sees_its_own_health_alert(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(j) patching ``node._journal.alert`` (the T13c point) sees a journal alert."""
    node = _node(seeded)
    snapshot: dict[str, object] = {"failed": 0, "error": None}
    monkeypatch.setattr(
        node._journal.event_log,
        "health",
        lambda: EventLogHealth(
            written=0,
            failed=snapshot["failed"],  # type: ignore[arg-type]
            dropped=0,
            pending=0,
            last_error_class=snapshot["error"],  # type: ignore[arg-type]
        ),
    )
    raised: list[Alert] = []
    monkeypatch.setattr(node._journal, "alert", raised.append)
    try:
        node._journal.check_health()  # first tick only records the baseline
        assert raised == []

        snapshot["failed"] = 1
        snapshot["error"] = "OSError"
        node._journal.check_health()
        assert raised == [
            Alert(
                "error",
                "paper PAPER-007 event log not writing: 1 failed, 0 dropped (OSError); "
                "trading continues",
                True,
                "eventlog:failed",
            )
        ]
    finally:
        _stop(node)
