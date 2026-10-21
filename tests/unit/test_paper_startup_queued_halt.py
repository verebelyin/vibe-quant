"""Queued halts/kills are applied before NautilusTrader starts (bead vibe-quant-k8ygz).

Commands were claimed only by ``_control_loop``, started beside ``run_task``, so an API
halt/kill (or the system kill cascade, which also arrives as ``kill`` rows) queued during
start-up waited for the first control tick - after NT may have started the strategies.
D4=B: the node's last start-up step claims, in id order, every pending halt/kill that no
pending ``resume`` precedes, so a queued kill is never overtaken by strategy entries.
Preceding non-halt rows (pause, close_all) stay pending and run afterwards in order.

Rows R25-R35 + R27c + R35b of design lane1-paper (T4), plus the reviewer's hostile probe:
a kill enqueued during the drain's own claim is a residual (applied at the first control
tick, not bounded) and stays pending.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_audit_paper_node import _FakeTradingNode
from tests.unit.test_paper_halt_restore import (
    _ISO,
    _bars,
    _latest_checkpoint,
    _trace_node,
    fake_tradingnode,  # noqa: F401  # fixture
    seeded,  # noqa: F401  # fixture
)
from tests.unit.test_paper_startup_halt import (
    _arm,
    _drive,
    _halt_records,
    _prepare,
    _remove_handlers,
    _restored,
    _startup_harness,
)
from vibe_quant.logging.events import EventType
from vibe_quant.paper.guard import GuardState, HaltReason
from vibe_quant.paper.node import NodeState, PaperTradingNode
from vibe_quant.paper.persistence import PaperCommand, PaperCommandQueue

if TYPE_CHECKING:
    from pathlib import Path

_TRADER = "PAPER-007"
_OTHER_TRADER = "PAPER-999"


def _enqueue(db: Path, *rows: tuple[str, dict[str, Any] | None]) -> list[int]:
    """Enqueue rows on a short-lived connection; returns their ids in order."""
    queue = PaperCommandQueue(db)
    try:
        return [queue.enqueue(_TRADER, command, payload) for command, payload in rows]
    finally:
        queue.close()


def _row(db: Path, command_id: int) -> PaperCommand:
    queue = PaperCommandQueue(db)
    try:
        cmd = queue.get(command_id)
    finally:
        queue.close()
    assert cmd is not None
    return cmd


def _halted(reason: HaltReason, message: str) -> dict[str, Any]:
    return {
        "state": "halted",
        "reason": reason,
        "message": message,
        "guard": GuardState.HALTED,
        "guard_reason": reason,
        "guard_msg": message,
    }


def _halt_record(reason: HaltReason, message: str) -> tuple[Any, ...]:
    return (
        "write",
        EventType.RISK_CHECK,
        {
            "action": "halt",
            "check_type": reason.value,
            "reason": reason.value,
            "message": message,
            "passed": False,
        },
        None,
    )


def _halt_alert(reason: HaltReason, message: str) -> tuple[Any, ...]:
    return ("alert", "circuit_breaker", f"HALT ({reason.value}): {message}", True)


class _CleanSnapshot:
    """A fresh, clean kill-switch read: ``resume`` is never refused for a stale one."""

    error_class = None
    engaged = False
    reason = None
    last_ok_mono = 0.0
    consecutive_failures = 0

    @staticmethod
    def age(_now: float) -> float:
        return 0.0


def _fresh_kill_snapshot(monkeypatch: pytest.MonkeyPatch, node: PaperTradingNode) -> None:
    monkeypatch.setattr(node._kill_monitor, "snapshot", _CleanSnapshot)


# ---------------------------------------------------- R25 (first test) / R26


@pytest.mark.parametrize("where", ["W1", "W2", "W3", "W4", "W5"])
async def test_queued_kill_during_startup_await_halts_before_run(
    where: str,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R25: a ``kill`` row queued at any start-up await is drained before NT starts:
    guard HALTED/kill_switch at ``after_init`` and ``in_run``, the row is ``done`` with
    result state ``halted``, and exactly one RISK_CHECK + one CRITICAL alert."""
    node, trace, seen = _prepare(monkeypatch, seeded)
    ids: list[int] = []

    def queue_kill() -> None:
        ids.extend(_enqueue(seeded["db"], ("kill", {"message": "kill switch: test"})))

    fired = _arm(monkeypatch, node, where, action=queue_kill)
    await _drive(node, seen)

    assert fired == [where], "the row really was queued inside that await"
    assert len(ids) == 1
    expected = _halted(HaltReason.KILL_SWITCH, "kill switch: test")
    assert seen["after_init"] == expected, "the drain applied the kill before run_async"
    assert seen["in_run"] == expected, "the trader starts under a halted guard"
    assert _halt_records(trace.calls) == [
        _halt_record(HaltReason.KILL_SWITCH, "kill switch: test"),
        _halt_alert(HaltReason.KILL_SWITCH, "kill switch: test"),
    ]
    row = _row(seeded["db"], ids[0])
    assert row.status == "done"
    assert row.result == {
        "state": "halted",
        "halt_reason": "kill_switch",
        "message": "kill switch: test",
    }


async def test_queued_api_halt_halts_before_run(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R26: an API ``halt`` row (no payload message) uses the default text and one alert."""
    node, trace, seen = _prepare(monkeypatch, seeded)
    ids: list[int] = []

    def queue_halt() -> None:
        ids.extend(_enqueue(seeded["db"], ("halt", None)))

    _arm(monkeypatch, node, "W3", action=queue_halt)
    await _drive(node, seen)

    expected = _halted(HaltReason.MANUAL, "halt via API")
    assert seen["after_init"] == expected
    assert seen["in_run"] == expected
    assert _halt_records(trace.calls) == [
        _halt_record(HaltReason.MANUAL, "halt via API"),
        _halt_alert(HaltReason.MANUAL, "halt via API"),
    ]
    row = _row(seeded["db"], ids[0])
    assert row.status == "done"
    assert row.result == {
        "state": "halted",
        "halt_reason": "manual",
        "message": "halt via API",
    }


# ------------------------------------------------------------- R27 / R27c / R28


async def test_non_leading_halt_waits_for_control_tick(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R27: [resume, halt] - the pending resume precedes the halt, so the drain claims
    nothing; the first control tick runs them in today's order and ends HALTED/manual."""
    node, _trace, _seen = _prepare(monkeypatch, seeded)
    _fresh_kill_snapshot(monkeypatch, node)
    ids = _enqueue(seeded["db"], ("resume", None), ("halt", None))
    node._setup_signal_handlers()
    try:
        await node._initialize()
        assert node._guard is not None
        assert node._guard.guard_state == GuardState.RUNNING, "the drain claimed 0"
        assert [_row(seeded["db"], cid).status for cid in ids] == ["pending", "pending"]

        await node.process_commands()

        assert [_row(seeded["db"], cid).status for cid in ids] == ["done", "done"]
        assert node._guard.guard_state == GuardState.HALTED
        assert node._guard.halt_reason == HaltReason.MANUAL
        assert node._guard.status()["message"] == "halt via API"
    finally:
        _remove_handlers()
        await node._shutdown()


async def test_pause_then_halt_halt_drained_resume_returns_running(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R27c: [pause, halt] - the halt is drained pre-start and the pause stays pending;
    on the first control tick the pause is a no-op on HALTED, and a later operator
    resume returns to RUNNING (not the PAUSED it would have returned to under order A)."""
    node, _trace, _seen = _prepare(monkeypatch, seeded)
    _fresh_kill_snapshot(monkeypatch, node)
    pause_id, halt_id = _enqueue(seeded["db"], ("pause", None), ("halt", None))
    node._setup_signal_handlers()
    try:
        await node._initialize()
        assert node._guard is not None
        assert node._guard.guard_state == GuardState.HALTED
        assert node._guard.halt_reason == HaltReason.MANUAL
        assert _row(seeded["db"], halt_id).status == "done"
        assert _row(seeded["db"], pause_id).status == "pending"

        await node.process_commands()
        assert _row(seeded["db"], pause_id).status == "done"
        assert node._guard.guard_state == GuardState.HALTED, "guard.pause is a no-op"

        (resume_id,) = _enqueue(seeded["db"], ("resume", None))
        await node.process_commands()

        assert _row(seeded["db"], resume_id).status == "done"
        assert node._guard.guard_state == GuardState.RUNNING
    finally:
        _remove_handlers()
        await node._shutdown()


async def test_leading_halt_drained_rest_kept(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R28: [halt, pause] - the leading halt is drained, the pause stays pending; the
    first control tick's ``guard.pause`` no-ops on HALTED and the row ends done."""
    node, _trace, _seen = _prepare(monkeypatch, seeded)
    halt_id, pause_id = _enqueue(seeded["db"], ("halt", None), ("pause", None))
    node._setup_signal_handlers()
    try:
        await node._initialize()
        assert node._guard is not None
        assert node._guard.guard_state == GuardState.HALTED
        assert _row(seeded["db"], halt_id).status == "done"
        assert _row(seeded["db"], pause_id).status == "pending"

        await node.process_commands()

        assert _row(seeded["db"], pause_id).status == "done"
        assert node._guard.guard_state == GuardState.HALTED
    finally:
        _remove_handlers()
        await node._shutdown()


# --------------------------------------------------------------------- R29 / R30


async def test_two_leading_halts_apply_in_order(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R29: [halt, kill] - both are drained in id order: MANUAL then KILL_SWITCH
    (strictly more severe), so two halt records and two alerts."""
    node, trace, seen = _prepare(monkeypatch, seeded)
    ids = _enqueue(seeded["db"], ("halt", {"message": "first"}), ("kill", {"message": "second"}))
    await _drive(node, seen)

    expected = _halted(HaltReason.KILL_SWITCH, "second")
    assert seen["after_init"] == expected
    assert _halt_records(trace.calls) == [
        _halt_record(HaltReason.MANUAL, "first"),
        _halt_alert(HaltReason.MANUAL, "first"),
        _halt_record(HaltReason.KILL_SWITCH, "second"),
        _halt_alert(HaltReason.KILL_SWITCH, "second"),
    ]
    assert [_row(seeded["db"], cid).status for cid in ids] == ["done", "done"]


def _engine_node(
    monkeypatch: pytest.MonkeyPatch,
    state: dict[str, Any],
    h: Any,
) -> tuple[PaperTradingNode, Any]:
    """A traced node whose guard is the real NT harness guard (engine-backed)."""
    node, trace = _trace_node(monkeypatch, state)
    monkeypatch.setattr(node, "_build_guard", lambda *_args, **_kwargs: h.guard)
    return node, trace


async def test_queued_halt_over_restored_halt_alerts_once(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R30: a restored MANUAL halt plus a queued ``halt``: the command row is done and
    changes nothing (equal priority), so the only alert is the restored notify at
    ``on_start``."""
    h = _startup_harness(monkeypatch, restored=_restored("manual", "halt via API"))
    node, _trace = _engine_node(monkeypatch, seeded, h)
    ids: list[int] = []
    _arm(
        monkeypatch,
        node,
        "W3",
        action=lambda: ids.extend(_enqueue(seeded["db"], ("halt", None))),
    )
    node._setup_signal_handlers()
    try:
        await node._initialize()
        assert _row(seeded["db"], ids[0]).status == "done"
        assert h.guard.guard_state == GuardState.HALTED
        assert h.listener.changes == [], "no alert from the command"

        h.engine.add_data(_bars(h, [100.0] * 2))
        h.engine.run()

        assert [(c[1], c[2], c[3]) for c in h.listener.changes] == [
            (GuardState.HALTED, HaltReason.MANUAL, "restored after restart: halt via API")
        ]
        assert h.listener.state_metrics == [{"restored": True, "checkpoint_time": _ISO}]
    finally:
        _remove_handlers()
        await node._shutdown()
        h.engine.dispose()


# --------------------------------------------------------------------- R31


async def test_drain_failure_keeps_rows_for_control_loop(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R31: a raising ``claim_startup_halts`` never blocks the start; the ERROR log names
    the class only (no exc_info, no exception text), the row stays pending, and the
    first control tick applies it."""
    ids = _enqueue(seeded["db"], ("kill", {"message": "late"}))

    def boom(self: PaperCommandQueue, trader_id: str) -> list[PaperCommand]:
        raise sqlite3.OperationalError("ZZMARK1") from RuntimeError("ZZMARK2")

    monkeypatch.setattr(PaperCommandQueue, "claim_startup_halts", boom)
    node, _trace, _seen = _prepare(monkeypatch, seeded)
    node._setup_signal_handlers()
    try:
        with caplog.at_level(logging.ERROR):
            await node._initialize()
        assert node._commands is not None, "the start continued past the failed drain"
        assert _row(seeded["db"], ids[0]).status == "pending"
        assert node._guard is not None
        assert node._guard.guard_state == GuardState.RUNNING

        errors = [
            r for r in caplog.records if "start-up command drain failed" in r.getMessage()
        ]
        assert [r.getMessage() for r in errors] == [
            "start-up command drain failed (OperationalError)"
        ]
        assert all(r.exc_info is None for r in caplog.records)
        assert "ZZMARK1" not in caplog.text
        assert "ZZMARK2" not in caplog.text

        await node.process_commands()

        assert _row(seeded["db"], ids[0]).status == "done"
        assert node._guard.guard_state == GuardState.HALTED
        assert node._guard.halt_reason == HaltReason.KILL_SWITCH
    finally:
        _remove_handlers()
        await node._shutdown()


# --------------------------------------------------------------------- R32


async def test_queued_kill_survives_two_failed_restarts(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R32: the drained kill is written into the failure checkpoint; two later restarts
    come up HALTED/kill_switch from the restored halt, the row stays ``done`` (not
    re-applied) and no second command alert is raised."""
    node1, trace1, _seen = _prepare(monkeypatch, seeded)
    ids = _enqueue(seeded["db"], ("kill", {"message": "kill switch: test"}))

    async def crash(self: Any) -> None:
        raise RuntimeError("venue dropped")

    monkeypatch.setattr(_FakeTradingNode, "run_async", crash)
    try:
        with pytest.raises(RuntimeError, match="venue dropped"):
            await node1.start()
    finally:
        _remove_handlers()

    assert node1.status.halt_reason == HaltReason.KILL_SWITCH
    halt_writes = [
        c for c in trace1.calls if c[0] == "write" and c[2].get("action") == "halt"
    ]
    assert halt_writes == [_halt_record(HaltReason.KILL_SWITCH, "kill switch: test")]
    halt_alerts = [c for c in trace1.calls if c[0] == "alert" and "HALT (" in c[2]]
    assert halt_alerts == [_halt_alert(HaltReason.KILL_SWITCH, "kill switch: test")]
    failed = _latest_checkpoint(seeded["db"])
    assert failed is not None
    assert failed.node_status["halt_reason"] == "kill_switch"
    assert failed.node_status["error_message"] == "kill switch: test"
    assert _row(seeded["db"], ids[0]).status == "done"

    for _restart in range(2):
        node, trace = _trace_node(monkeypatch, seeded)
        await node._initialize()
        try:
            assert node.status.state == NodeState.HALTED
            assert node.status.halt_reason == HaltReason.KILL_SWITCH
            assert node.status.error_message == "restored after restart: kill switch: test"
            assert _row(seeded["db"], ids[0]).status == "done", "not re-applied"
            assert _halt_records(trace.calls) == [], "no second command alert"
        finally:
            await node._shutdown()
        checkpoint = _latest_checkpoint(seeded["db"])
        assert checkpoint is not None
        assert checkpoint.node_status["halt_reason"] == "kill_switch"


# ------------------------------------------------------------- R34 / R35 (SQL)


def test_claim_startup_halts_is_per_trader(tmp_path: Path) -> None:
    """R34: another trader's pending halt is untouched."""
    queue = PaperCommandQueue(tmp_path / "state.db")
    try:
        mine = queue.enqueue(_TRADER, "halt")
        theirs = queue.enqueue(_OTHER_TRADER, "halt")
        assert [c.id for c in queue.claim_startup_halts(_TRADER)] == [mine]
        assert queue.get(mine).status == "running"
        assert queue.get(theirs).status == "pending"
    finally:
        queue.close()


@pytest.mark.parametrize(
    ("commands", "claimed", "pending"),
    [
        (["halt", "kill", "pause", "halt"], [0, 1, 3], [2]),
        (["halt", "resume", "kill"], [0], [1, 2]),
        (["close_all", "kill"], [1], [0]),
    ],
)
def test_claim_startup_halts_skips_until_first_resume(
    tmp_path: Path,
    commands: list[str],
    claimed: list[int],
    pending: list[int],
) -> None:
    """R35: in id order every halt/kill is claimed, a row before the first pending
    ``resume`` is skipped and stays pending, and the scan stops at that ``resume``."""
    queue = PaperCommandQueue(tmp_path / "state.db")
    try:
        ids = [queue.enqueue(_TRADER, command) for command in commands]
        claimed_rows = queue.claim_startup_halts(_TRADER)
        assert [c.id for c in claimed_rows] == [ids[i] for i in claimed]
        assert [c.command for c in claimed_rows] == [commands[i] for i in claimed]
        assert [c.status for c in claimed_rows] == ["running"] * len(claimed)
        assert [queue.get(ids[i]).status for i in pending] == ["pending"] * len(pending)
    finally:
        queue.close()


# ------------------------------------------------------- R35b (r3 F1, D4=B)


async def test_startup_halt_overtakes_close_all_blocks_entries(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R35b: [close_all {wait_secs: 10}, kill] - the kill is drained before NT starts,
    so the strategies' ``on_start``/every-bar entries are all gate-denied and every
    strategy is recorded for stop. The close_all afterwards runs on the halted session
    and its ``_wait_flat`` is the controlled future."""
    h = _startup_harness(monkeypatch)
    node, _trace = _engine_node(monkeypatch, seeded, h)
    ids = _enqueue(
        seeded["db"], ("close_all", {"wait_secs": 10}), ("kill", None)
    )
    close_id, kill_id = ids
    gate: asyncio.Future[list[str]] = asyncio.get_running_loop().create_future()
    entered = asyncio.Event()

    async def wait_flat(_timeout: float) -> list[str]:
        entered.set()
        return await gate

    monkeypatch.setattr(node, "_wait_flat", wait_flat)

    node._setup_signal_handlers()
    try:
        await node._initialize()
        assert h.guard.guard_state == GuardState.HALTED
        assert h.guard.halt_reason == HaltReason.KILL_SWITCH
        assert _row(seeded["db"], kill_id).status == "done"
        assert _row(seeded["db"], close_id).status == "pending"
        assert h.guard.status()["pending_stop"] == sorted(str(s.id) for s in h.strategies)

        h.engine.add_data(_bars(h, [100.0] * 5))
        h.engine.run()

        assert len(h.filled_entries()) == 0, "the halt really blocked the entries"
        assert all(not strategy.is_running for strategy in h.strategies)

        tick = asyncio.create_task(node.process_commands())
        await asyncio.wait_for(entered.wait(), 5)
        assert _row(seeded["db"], close_id).status == "running"
        assert h.guard.guard_state == GuardState.HALTED

        gate.set_result([])
        await asyncio.wait_for(tick, 5)
        assert _row(seeded["db"], close_id).status == "done"
    finally:
        _remove_handlers()
        await node._shutdown()
        h.engine.dispose()


# ------------------------------------------------------- hostile probe (r2 F1)


async def test_kill_enqueued_during_the_drain_claim_waits_for_control_tick(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Residual, not bounded: a kill enqueued DURING the drain's own claim is not in
    the drain's SELECT - it stays pending and is applied at the first control tick."""
    real_claim = PaperCommandQueue.claim_startup_halts
    ids: list[int] = []

    def claim_then_enqueue(self: PaperCommandQueue, trader_id: str) -> list[PaperCommand]:
        claimed = real_claim(self, trader_id)
        ids.extend(_enqueue(seeded["db"], ("kill", {"message": "late kill"})))
        return claimed

    monkeypatch.setattr(PaperCommandQueue, "claim_startup_halts", claim_then_enqueue)
    node, _trace, _seen = _prepare(monkeypatch, seeded)
    node._setup_signal_handlers()
    try:
        await node._initialize()
        assert len(ids) == 1
        assert node._guard is not None
        assert node._guard.guard_state == GuardState.RUNNING, "the drain did not see it"
        assert _row(seeded["db"], ids[0]).status == "pending"

        await node.process_commands()

        assert _row(seeded["db"], ids[0]).status == "done"
        assert node._guard.guard_state == GuardState.HALTED
        assert node._guard.halt_reason == HaltReason.KILL_SWITCH
    finally:
        _remove_handlers()
        await node._shutdown()
