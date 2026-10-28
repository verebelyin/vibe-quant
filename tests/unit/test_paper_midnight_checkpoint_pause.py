"""A daily-loss halt during the start-up checkpoint pause survives the UTC-midnight lift.

A hung initial checkpoint pauses entries until it saves. A daily-loss halt landing in
that window used to lift to RUNNING at the UTC rollover, reopening entries before any
state was saved. The lift is now re-paused under the same checkpoint block, which the
late save lifts. Driven through the public ``PaperTradingNode.start()``.
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

import vibe_quant.paper.node as node_module
from tests.unit.test_audit_paper_node import _config, _FakeTradingNode
from tests.unit.test_paper_halt_restore import (
    _EmptyPortfolio,
    _FakeClock,
    _FakeMoney,
    _UnitGuard,
    fake_tradingnode,  # noqa: F401  # fixture
    seeded,  # noqa: F401  # fixture
)
from tests.unit.test_paper_startup_drain_bound import _Chat
from tests.unit.test_paper_startup_queued_halt import _enqueue, _fresh_kill_snapshot, _row
from vibe_quant.paper.command_channel import (
    CHECKPOINT_BLOCK_MESSAGE,
    RESUME_OWED_MESSAGE,
    block_message,
)
from vibe_quant.paper.guard import GuardState, HaltReason
from vibe_quant.paper.node import PaperTradingNode
from vibe_quant.paper.persistence import StatePersistence

_DAY1 = datetime(2026, 1, 5, 22, 0, tzinfo=UTC)

_world: dict[str, Any] = {}


class _Clock(_FakeClock):
    def utc_now(self) -> datetime:
        now: datetime = _world["now"]
        return now


class _Portfolio(_EmptyPortfolio):
    def equity(self, venue: Any) -> dict[Any, Any]:
        from nautilus_trader.model.objects import Currency

        return {Currency.from_str("USDT"): _FakeMoney(str(_world["equity"]))}


class _MidnightGuard(_UnitGuard):
    @property
    def portfolio(self) -> Any:  # type: ignore[override]
        return _Portfolio()

    @property
    def clock(self) -> Any:  # type: ignore[override]
        return _Clock()


async def _poll(cond: Any, limit: float = 10.0) -> bool:
    end = time.monotonic() + limit
    while time.monotonic() < end:
        if cond():
            return True
        await asyncio.sleep(0.05)
    return bool(cond())


async def _start_with_hung_checkpoint(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    hold_checkpoint: bool,
    halt_first: bool = False,
    *,
    fail_late: bool = False,
    interval: float = 60,
    saved: list[int] | None = None,
) -> tuple[PaperTradingNode, threading.Event, asyncio.Event, asyncio.Task[Any]]:
    """``fail_late``: the held first save raises once released; ``saved`` counts the
    saves that succeed; ``interval``: seconds between routine (periodic) saves."""
    hold = threading.Event()
    calls = {"n": 0}
    real = StatePersistence.save_checkpoint

    # With a halt first, save 1 is the halt's own; the start-up checkpoint queues behind it.
    fail_at = 2 if halt_first and fail_late else 1

    def save(p: StatePersistence, *a: Any) -> Any:
        calls["n"] += 1
        if hold_checkpoint and calls["n"] <= fail_at:
            hold.wait(40)
            if fail_late and calls["n"] == fail_at:
                raise OSError("disk I/O error")
        out = real(p, *a)
        if saved is not None:
            saved.append(1)
        return out

    monkeypatch.setattr(StatePersistence, "save_checkpoint", save)
    monkeypatch.setattr(node_module, "TradingGuard", _MidnightGuard)
    _world.update(now=_DAY1, equity=Decimal("10000"))
    reached = asyncio.Event()
    stop = asyncio.Event()

    async def run_async(self: Any) -> None:
        reached.set()
        await stop.wait()

    async def stop_async(self: Any) -> None:
        stop.set()

    monkeypatch.setattr(_FakeTradingNode, "run_async", run_async)
    monkeypatch.setattr(_FakeTradingNode, "stop_async", stop_async)
    node = PaperTradingNode(_config(seeded, state_persistence_interval=interval))
    _fresh_kill_snapshot(monkeypatch, node)
    if halt_first:
        # The daily-loss halt is already in force when the checkpoint write stalls, so the
        # checkpoint pause has no RUNNING guard to pause.
        apply_startup_halt = node._apply_startup_halt

        def halt_then_apply() -> None:
            apply_startup_halt()
            _daily_loss_halt(node)

        monkeypatch.setattr(node, "_apply_startup_halt", halt_then_apply)
    task = asyncio.create_task(node.start())
    await asyncio.wait_for(reached.wait(), 20)
    return node, hold, stop, task


def _daily_loss_halt(node: PaperTradingNode) -> None:
    guard = node._guard
    assert guard is not None
    _world["equity"] = Decimal("10000")
    guard.evaluate_risk()  # UTC-day baseline 10000
    _world["equity"] = Decimal("9000")  # 10% > 2% daily limit, 10% < 15% drawdown
    guard.evaluate_risk()
    assert guard.guard_state == GuardState.HALTED
    assert guard.halt_reason == HaltReason.MAX_DAILY_LOSS


async def _stop(node: PaperTradingNode, hold: threading.Event, task: asyncio.Task[Any]) -> None:
    hold.set()
    node._shutdown_event.set()
    await asyncio.wait_for(task, 60)


@pytest.mark.parametrize(
    "halt_first", [True, False], ids=["halt_before_stall", "halt_while_paused"]
)
async def test_midnight_lift_keeps_entries_blocked_until_checkpoint_saves(
    halt_first: bool,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node, hold, _stop_ev, task = await _start_with_hung_checkpoint(
        seeded, monkeypatch, True, halt_first
    )
    guard = node._guard
    assert guard is not None
    try:
        if not halt_first:
            assert guard.guard_state == GuardState.PAUSED
            _daily_loss_halt(node)
        halted: GuardState = guard.guard_state
        assert halted == GuardState.HALTED

        _world["now"] = _DAY1 + timedelta(hours=3)  # 01:00 UTC next day
        guard.evaluate_risk()
        blocked: GuardState = guard.guard_state
        assert blocked == GuardState.PAUSED, "entries must stay blocked"

        node._lift_channel_pause()  # a channel recovery tick while the save is pending
        still: GuardState = guard.guard_state
        assert still == GuardState.PAUSED, "only the checkpoint save may lift this pause"

        hold.set()  # the checkpoint finally saves
        assert await _poll(lambda: guard.guard_state == GuardState.RUNNING)
    finally:
        await _stop(node, hold, task)


async def test_midnight_lift_without_pending_checkpoint_pause_runs_normally(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node, hold, _stop_ev, task = await _start_with_hung_checkpoint(seeded, monkeypatch, False)
    guard = node._guard
    assert guard is not None
    try:
        assert guard.guard_state == GuardState.RUNNING
        _daily_loss_halt(node)
        _world["now"] = _DAY1 + timedelta(hours=3)
        guard.evaluate_risk()
        assert guard.guard_state == GuardState.RUNNING
    finally:
        await _stop(node, hold, task)


@pytest.mark.parametrize("cause", ["checkpoint", "channel"])
async def test_midnight_lift_still_blocked_alerts_still_paused_not_resumed(
    cause: str,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lift is re-paused inside the same transition: ONE alert says so, never RESUMED."""
    checkpoint = cause == "checkpoint"
    node, hold, _stop_ev, task = await _start_with_hung_checkpoint(
        seeded, monkeypatch, checkpoint, halt_first=checkpoint
    )
    chat = _Chat()
    node._journal.attach(None, chat.bot(), "s")
    guard = node._guard
    assert guard is not None
    try:
        if checkpoint:
            reason = CHECKPOINT_BLOCK_MESSAGE
        else:
            _daily_loss_halt(node)
            reason = block_message(2, "OSError")
            assert await _poll(lambda: any("HALT" in b for b in chat.bodies))
        before = len(chat.bodies)
        if not checkpoint:  # no await until the lift: the control loop would recover it
            node._cmd_channel.blocking = True
            node._cmd_channel.failures = 2
            node._cmd_channel.last_error = "OSError"
        _world["now"] = _DAY1 + timedelta(hours=3)
        guard.evaluate_risk()
        paused: GuardState = guard.guard_state
        assert paused == GuardState.PAUSED
        assert await _poll(lambda: len(chat.bodies) > before)  # the alert is sent
        await asyncio.sleep(0.3)  # a second (wrong) alert would land by now
        await node._journal.wait_alerts(None)
        assert chat.bodies[before:] == [f"day rolled over, still paused: {reason}"]
    finally:
        await _stop(node, hold, task)


# ------------------------------------------------- late failure: user decision 1A


async def _late_failure(node: PaperTradingNode, hold: threading.Event, chat: _Chat) -> None:
    """Release the held start-up save so it fails; wait for its alert."""
    hold.set()
    assert await _poll(lambda: node._startup_checkpoint_failed)
    assert await _poll(lambda: any("failed late" in b for b in chat.bodies))


async def _operator_resume(node: PaperTradingNode, db: Any) -> Any:
    """One operator resume through the API command row (the SIGUSR2 entry point is
    pinned in test_paper_startup_halt, the same queue row)."""
    cid = _enqueue(db, ("resume", {}))[0]
    assert await _poll(lambda: _row(db, cid).status not in ("pending", "running"))
    return _row(db, cid)


async def test_late_failure_block_clears_on_a_routine_save_then_one_resume_runs(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1A: a later ROUTINE (periodic) save lifts the block, the guard stays PAUSED and its
    text says what is true now; exactly ONE operator resume then reaches RUNNING."""
    saved: list[int] = []
    node, hold, _stop_ev, task = await _start_with_hung_checkpoint(
        seeded, monkeypatch, True, fail_late=True, interval=0.2, saved=saved
    )
    chat = _Chat()
    node._journal.attach(None, chat.bot(), "s")
    guard = node._guard
    assert guard is not None
    try:
        await _late_failure(node, hold, chat)
        before = len(saved)
        assert await _poll(lambda: len(saved) > before), "a routine save lands"
        assert await _poll(lambda: not node._cmd_channel.checkpoint_blocking)
        paused: GuardState = guard.guard_state
        assert paused == GuardState.PAUSED, "the save does not resume by itself"
        assert guard.message == RESUME_OWED_MESSAGE
        assert node.status.error_message == RESUME_OWED_MESSAGE
        node._lift_channel_pause()  # a channel recovery is not an operator resume
        still: GuardState = guard.guard_state
        assert still == GuardState.PAUSED

        row = await _operator_resume(node, seeded["db"])
        assert (row.status, row.result["message"]) == ("done", "resumed from pause")
        running: GuardState = guard.guard_state
        assert running == GuardState.RUNNING
    finally:
        await _stop(node, hold, task)


async def test_late_failure_resume_before_any_save_reports_repaused_and_alerts_once(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R2: a resume the unsaved checkpoint re-pauses says so (command fails with the current
    reason, one alert), never "resumed from pause"; once a save lands ONE resume runs."""
    node, hold, _stop_ev, task = await _start_with_hung_checkpoint(
        seeded, monkeypatch, True, fail_late=True
    )
    chat = _Chat()
    node._journal.attach(None, chat.bot(), "s")
    guard = node._guard
    assert guard is not None
    try:
        await _late_failure(node, hold, chat)
        await node._journal.wait_alerts(None)
        before = len(chat.bodies)
        repaused = f"resume re-paused, still paused: {CHECKPOINT_BLOCK_MESSAGE}"

        row = await _operator_resume(node, seeded["db"])
        assert (row.status, row.error) == ("failed", repaused)
        repaused_state: GuardState = guard.guard_state
        assert repaused_state == GuardState.PAUSED
        assert await _poll(lambda: len(chat.bodies) > before)
        await asyncio.sleep(0.3)  # a second alert would land by now
        await node._journal.wait_alerts(None)
        assert chat.bodies[before:] == [repaused]

        assert await _poll(lambda: not node._cmd_channel.checkpoint_blocking), "its save lands"
        assert guard.message == RESUME_OWED_MESSAGE
        row = await _operator_resume(node, seeded["db"])
        assert row.status == "done"
        running: GuardState = guard.guard_state
        assert running == GuardState.RUNNING
    finally:
        await _stop(node, hold, task)


async def test_late_failure_then_routine_save_midnight_lift_stays_paused_until_resume(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1A + R2: the daily-loss lift after the block cleared is re-paused (no operator resume
    yet) and its alert names the CURRENT reason; ONE resume then reaches RUNNING."""
    saved: list[int] = []
    node, hold, _stop_ev, task = await _start_with_hung_checkpoint(
        seeded, monkeypatch, True, halt_first=True, fail_late=True, interval=0.2, saved=saved
    )
    chat = _Chat()
    node._journal.attach(None, chat.bot(), "s")
    guard = node._guard
    assert guard is not None
    try:
        await _late_failure(node, hold, chat)
        assert await _poll(lambda: not node._cmd_channel.checkpoint_blocking), "routine save"
        halted: GuardState = guard.guard_state
        assert halted == GuardState.HALTED
        await node._journal.wait_alerts(None)
        before = len(chat.bodies)

        _world["now"] = _DAY1 + timedelta(hours=3)
        guard.evaluate_risk()
        paused: GuardState = guard.guard_state
        assert paused == GuardState.PAUSED, "no operator resume yet"
        assert await _poll(lambda: len(chat.bodies) > before)
        await asyncio.sleep(0.3)
        await node._journal.wait_alerts(None)
        assert chat.bodies[before:] == [f"day rolled over, still paused: {RESUME_OWED_MESSAGE}"]

        row = await _operator_resume(node, seeded["db"])
        assert row.status == "done"
        assert guard.guard_state == GuardState.RUNNING
    finally:
        await _stop(node, hold, task)
