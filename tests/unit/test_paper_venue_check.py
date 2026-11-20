"""Paper venue config check: the node's pause-text composition and lift guard.

Contract rows C1-C8, C7b, C7c of design 68m53 (task T2a, no venue I/O yet). The venue
blocker (``_venue_block``) is one more reason to keep new entries blocked:

* the status the API shows composes ``"<other message> | <venue blocker>"`` (F2);
* ``_entry_block_message`` names the venue blocker LAST (warm-up, checkpoint,
  resume-owed, channel first), so every resume/rollover re-pause keeps it;
* ``_retell_pause`` keeps the guard's pause text on the strongest live reason and
  says when a released blocker leaves an independent pause behind (R9);
* the channel's own pause is never lifted while the venue blocks, while a claimed
  command batch is still running, or while claimed rows are left unrun (R2).

Tests set ``_venue_block`` directly the way the venue check will (T2b): a block is
set, entries blocked on a RUNNING guard, the pause retold; a pass clears it, tries
the channel lift and retells with the released text.
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import json
import logging
import re
import signal
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId

import vibe_quant.paper.node as node_module
from tests.unit.paper_venue_harness import (
    no_network,  # noqa: F401  # fixture
    synthetic_warmup,  # noqa: F401  # fixture
    venue_pass,
)
from tests.unit.test_audit_paper_node import _config
from tests.unit.test_paper_command_channel import _Claims, _polls, _until
from tests.unit.test_paper_halt_restore import (
    _EmptyCache,
    _EmptyPortfolio,
    _trace_node,
    fake_tradingnode,  # noqa: F401  # fixture
    node_timing,  # noqa: F401  # fixture
    seeded,  # noqa: F401  # fixture
)
from tests.unit.test_paper_midnight_checkpoint_pause import _DAY1, _MidnightGuard, _world
from tests.unit.test_paper_startup_and_estimate import _Engine
from tests.unit.test_paper_startup_halt import _Guard, _remove_handlers
from tests.unit.test_paper_startup_queued_halt import _enqueue, _fresh_kill_snapshot, _row
from vibe_quant.alerts.telegram import TelegramBot, TelegramConfig
from vibe_quant.paper.command_channel import (
    CHECKPOINT_BLOCK_MESSAGE,
    RESUME_OWED_MESSAGE,
    STARTUP_CHECKPOINT_STEP,
    block_message,
)
from vibe_quant.paper.guard import RESTORED_HALT_PREFIX, GuardState, HaltReason
from vibe_quant.paper.journal import SessionJournal
from vibe_quant.paper.node import (
    STARTUP_CONNECT_TIMEOUT_S,
    VENUE_CHECK_PENDING,
    VENUE_CHECK_VERIFIED_STILL_PAUSED,
    NodeStatus,
    PaperTradingNode,
)
from vibe_quant.paper.persistence import PaperCommandQueue
from vibe_quant.paper.venue_config import parse_positions

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from tests.unit.test_audit_paper_node import _FakeTradingNode
    from tests.unit.test_paper_halt_restore import _Trace

# ro-vc-1: no venue test reaches Binance (warm-up) or Telegram.
pytestmark = pytest.mark.usefixtures("no_network", "synthetic_warmup")

_M = "venue config mismatch: ETHUSDT: venue leverage 5x != resolved 10x"
_K = "kill switch: operator test"
_BLOCK_5 = block_message(5, "OperationalError")
_BTC = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")


class _FlatPortfolio(_EmptyPortfolio):
    def net_position(self, instrument_id: Any) -> Decimal:
        return Decimal(0)


class _VGuard(_Guard):
    """The unit guard with a flat portfolio, so its entry gate can be asked directly."""

    @property
    def portfolio(self) -> Any:  # type: ignore[override]
        return _FlatPortfolio()


@contextlib.asynccontextmanager
async def _node(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],  # noqa: F811
    guard_cls: type = _VGuard,
) -> AsyncIterator[tuple[PaperTradingNode, _Trace]]:
    """A traced node past ``_initialize`` (real guard RUNNING, claims working)."""
    monkeypatch.setattr(node_module, "TradingGuard", guard_cls)
    node, trace = _trace_node(monkeypatch, seeded)
    _fresh_kill_snapshot(monkeypatch, node)
    node._setup_signal_handlers()
    try:
        await node._initialize()
        assert node._guard is not None
        assert node._guard.guard_state == GuardState.PAUSED, "armed: PENDING"
        venue_pass(node, monkeypatch)
        assert node._guard.guard_state == GuardState.RUNNING
        yield node, trace
    finally:
        _remove_handlers()
        await node._shutdown()


def _guard(node: PaperTradingNode) -> Any:
    guard = node._guard
    assert guard is not None
    return guard


def _shown(node: PaperTradingNode) -> object:
    """What the API shows (the checkpoint's ``node_status.error_message``)."""
    shown = node.status.to_dict()["error_message"]
    assert node._state_payload()["message"] == shown
    return shown


def _set_venue_block(node: PaperTradingNode, text: str) -> None:
    """What the venue check does on PENDING / a latch (§B.2, §B.7.2-3)."""
    node._venue_block = text
    node._block_entries_for_channel(text)
    node._retell_pause()


def _venue_pass(node: PaperTradingNode) -> None:
    """What a published venue pass does (§B.5.3): clear, try the lift, retell."""
    released = node._venue_block
    node._venue_block = None
    node._lift_channel_pause()
    node._retell_pause(released=released)


class _Qty:
    def as_decimal(self) -> Decimal:
        return Decimal("0.01")

    def __str__(self) -> str:
        return "0.01"


def _entry_denial(node: PaperTradingNode) -> str | None:
    """The guard's admission answer for a new BTC buy (not reduce-only)."""
    order = SimpleNamespace(
        is_reduce_only=False,
        instrument_id=_BTC,
        quantity=_Qty(),
        side=OrderSide.BUY,
        client_order_id=None,
        price=Decimal("100"),
    )
    denial: str | None = _guard(node)._checked_denial_reason(order)
    return denial


def _alerts(trace: _Trace) -> list[str]:
    return [c[2] for c in trace.calls if c[0] == "alert"]


def _late(result: BaseException | None) -> asyncio.Future[Any]:
    """A finished start-up job future: saved (None) or failed (the exception)."""
    future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    if result is None:
        future.set_result(None)
    else:
        future.set_exception(result)
    return future


def _checkpoint_block(node: PaperTradingNode) -> None:
    """The bounded start-up save gave no answer: entries paused until it saves."""
    node._cmd_channel.checkpoint_blocking = True
    assert node._block_entries_for_channel(CHECKPOINT_BLOCK_MESSAGE)


# ------------------------------------------------------------- composition (F2)


@pytest.mark.parametrize(
    ("error_message", "venue_block", "shown"),
    [
        (None, None, None),
        ("", None, ""),
        ("halted", None, "halted"),
        (None, "V", "V"),
        ("", "V", "V"),
        ("V", "V", "V"),
        ("K", "V", "K | V"),
        ("V ", "V", "V  | V"),
    ],
)
def test_shown_message_composition(
    error_message: str | None, venue_block: str | None, shown: str | None
) -> None:
    """The status shows the venue blocker beside, never instead of, the message."""
    status = NodeStatus(error_message=error_message, venue_block=venue_block)

    assert status.shown_message() == shown
    assert status.to_dict()["error_message"] == shown
    assert status.error_message == error_message, "no writer's message is changed"


def test_entry_block_message_names_venue_last() -> None:
    """Precedence: warm-up, checkpoint, resume-owed, channel, then the venue blocker."""
    node = PaperTradingNode.__new__(PaperTradingNode)
    node._status = NodeStatus()
    node._warmup_entries_off = None
    node._startup_resume_owed = False
    node._cmd_channel = node_module.CommandChannelHealth()
    assert node._entry_block_message() is None

    node._venue_block = _M
    assert node._entry_block_message() == _M
    assert node.status.venue_block == _M, "the status mirrors the node's venue blocker"
    node._cmd_channel.blocking = True
    node._cmd_channel.failures = 5
    node._cmd_channel.last_error = "OperationalError"
    assert node._entry_block_message() == _BLOCK_5
    node._startup_resume_owed = True
    assert node._entry_block_message() == RESUME_OWED_MESSAGE
    node._cmd_channel.checkpoint_blocking = True
    assert node._entry_block_message() == CHECKPOINT_BLOCK_MESSAGE
    node._warmup_entries_off = "warm-up off"
    assert node._entry_block_message() == "warm-up off"
    node._venue_block = None
    assert node.status.venue_block is None


# ------------------------------------------------------------------- C1-C8


async def test_warmup_block_survives_venue_pass(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C1 (R9): a warm-up failure while PENDING takes over the pause text; the pass
    leaves the warm-up block (and only it) shown, still PAUSED."""
    async with _node(monkeypatch, seeded) as (node, _trace):
        guard = _guard(node)
        _set_venue_block(node, VENUE_CHECK_PENDING)
        assert guard.guard_state == GuardState.PAUSED
        assert guard.message == VENUE_CHECK_PENDING
        assert _shown(node) == VENUE_CHECK_PENDING

        node._on_topup_failed("BTCUSDT", "1h")
        warm = node._warmup_entries_off
        assert warm is not None
        assert guard.message == warm
        assert _shown(node) == f"{warm} | {VENUE_CHECK_PENDING}"

        _venue_pass(node)
        assert guard.guard_state == GuardState.PAUSED
        assert guard.message == warm
        assert _shown(node) == warm
        assert _entry_denial(node) is not None


async def test_latch_survives_checkpoint_recovery(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2: a latch under a failed-late start-up checkpoint: once a later save lands the
    status shows the owed resume beside the latch; the operator resume is re-paused by
    the latch, which stays shown."""
    async with _node(monkeypatch, seeded) as (node, _trace):
        guard = _guard(node)
        _checkpoint_block(node)
        _set_venue_block(node, _M)
        node._startup_job_late(_late(OSError("disk I/O error")), STARTUP_CHECKPOINT_STEP)
        assert node._startup_resume_owed is True

        node._clear_failed_startup_checkpoint()  # a later checkpoint saved
        assert guard.message == RESUME_OWED_MESSAGE
        assert _shown(node) == f"{RESUME_OWED_MESSAGE} | {_M}"

        assert node.resume() == (False, f"resume re-paused, still paused: {_M}")
        assert guard.guard_state == GuardState.PAUSED
        assert guard.message == _M
        assert _shown(node) == _M
        assert _entry_denial(node) is not None


async def test_pass_under_operator_pause_retells(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C3: an operator pause while PENDING outlives the pass: the text says the venue is
    verified and an operator resume is still needed; the pause is not owned."""
    async with _node(monkeypatch, seeded) as (node, _trace):
        guard = _guard(node)
        _set_venue_block(node, VENUE_CHECK_PENDING)
        _enqueue(seeded["db"], ("pause", None))
        await node.process_commands()
        assert node._cmd_channel.owns_pause is False
        assert guard.message == VENUE_CHECK_PENDING

        _venue_pass(node)
        assert guard.guard_state == GuardState.PAUSED
        assert guard.message == VENUE_CHECK_VERIFIED_STILL_PAUSED
        assert _shown(node) == VENUE_CHECK_VERIFIED_STILL_PAUSED
        assert node._cmd_channel.owns_pause is False


async def test_latch_composes_with_halt(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C4: a kill halt over a latch shows both; unlock + resume lands PAUSED on M."""
    async with _node(monkeypatch, seeded) as (node, _trace):
        guard = _guard(node)
        _set_venue_block(node, _M)
        node.halt(HaltReason.KILL_SWITCH, _K)
        assert guard.guard_state == GuardState.HALTED
        assert _shown(node) == f"{_K} | {_M}"

        assert node.resume() == (False, f"resume re-paused, still paused: {_M}")
        assert guard.guard_state == GuardState.PAUSED
        assert guard.message == _M
        assert _shown(node) == _M
        assert _entry_denial(node) is not None


@pytest.mark.parametrize("order", ["halt_then_latch", "latch_then_halt"])
async def test_rollover_lift_repauses_with_latch(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    order: str,
) -> None:
    """C5: the UTC-rollover lift of a daily-loss halt keeps the latch. A latch landing on
    the halted guard: the lift to RUNNING is re-paused with the "day rolled over, still
    paused" alert. A latch before the halt: the lift returns to the pre-halt PAUSED
    (silently, as any HALTED->PAUSED lift) and the pause text is retold to M."""
    _world.update(now=_DAY1, equity=Decimal("10000"))
    async with _node(monkeypatch, seeded, _MidnightGuard) as (node, trace):
        guard = _guard(node)
        if order == "latch_then_halt":
            _set_venue_block(node, _M)
        guard.evaluate_risk()  # UTC-day baseline 10000
        _world["equity"] = Decimal("9000")  # 10% daily loss > 2% limit
        guard.evaluate_risk()
        assert guard.guard_state == GuardState.HALTED
        assert guard.halt_reason == HaltReason.MAX_DAILY_LOSS
        if order == "halt_then_latch":
            _set_venue_block(node, _M)
            assert guard.guard_state == GuardState.HALTED

        _world["now"] = _DAY1 + timedelta(hours=3)
        before = len(_alerts(trace))
        guard.evaluate_risk()

        assert guard.guard_state == GuardState.PAUSED
        assert guard.message == _M
        assert _shown(node) == _M
        rollover = [a for a in _alerts(trace)[before:] if "day rolled over" in a]
        if order == "halt_then_latch":
            assert rollover == [f"day rolled over, still paused: {_M}"]
        else:
            assert rollover == []


async def test_late_checkpoint_failure_keeps_latch(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C6: the start-up checkpoint failing late disowns the pause and owes a resume;
    the latch stays shown."""
    async with _node(monkeypatch, seeded) as (node, _trace):
        guard = _guard(node)
        _checkpoint_block(node)
        _set_venue_block(node, _M)
        assert _shown(node) == f"{CHECKPOINT_BLOCK_MESSAGE} | {_M}"

        node._startup_job_late(_late(OSError("disk I/O error")), STARTUP_CHECKPOINT_STEP)

        assert node._cmd_channel.owns_pause is False
        assert node._startup_resume_owed is True
        assert guard.guard_state == GuardState.PAUSED
        shown = _shown(node)
        assert isinstance(shown, str)
        assert _M in shown


async def test_channel_block_then_venue_pass(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C7 (R9): a channel block that starts on the PENDING pause takes over the text;
    the pass leaves the channel text; the channel's recovery lifts to RUNNING."""
    async with _node(monkeypatch, seeded) as (node, _trace):
        guard = _guard(node)
        _set_venue_block(node, VENUE_CHECK_PENDING)
        _Claims(monkeypatch, node, "F" * 5)
        assert await _polls(node, 5) == 5
        assert node._cmd_channel.blocking is True
        assert guard.message == _BLOCK_5
        assert _shown(node) == f"{_BLOCK_5} | {VENUE_CHECK_PENDING}"

        _venue_pass(node)
        assert guard.guard_state == GuardState.PAUSED
        assert guard.message == _BLOCK_5
        assert _shown(node) == _BLOCK_5

        await _polls(node, 1)  # the next claim works
        assert guard.guard_state == GuardState.RUNNING
        assert node._cmd_channel.owns_pause is False


async def test_channel_release_under_operator_pause_retells(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R9: the channel recovers under an operator pause that took its pause over: the
    channel text says it cleared and the operator still has to resume."""
    async with _node(monkeypatch, seeded) as (node, _trace):
        guard = _guard(node)
        _Claims(monkeypatch, node, "F" * 5)
        await _polls(node, 5)
        assert guard.message == _BLOCK_5
        node._cmd_channel.owns_pause = False  # what an operator ``pause`` command does

        await _polls(node, 1)

        text = f"{_BLOCK_5} cleared; still paused until an operator resumes"
        assert guard.guard_state == GuardState.PAUSED
        assert guard.message == text
        assert _shown(node) == text


async def test_refused_lift_keeps_the_channel_text(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R9: the channel recovers but its lift is refused (stale kill-switch read): the
    pause stays owned and the text never claims an operator must resume; the next
    good claim with a fresh read lifts it."""
    async with _node(monkeypatch, seeded) as (node, _trace):
        guard = _guard(node)
        _Claims(monkeypatch, node, "F" * 5)
        await _polls(node, 5)
        assert guard.message == _BLOCK_5
        real_resume = node.resume
        stale = (False, "system kill switch state not fresh (no clean read)")
        monkeypatch.setattr(node, "resume", lambda: stale)

        await _polls(node, 1)

        assert guard.guard_state == GuardState.PAUSED
        assert node._cmd_channel.owns_pause is True
        assert guard.message == _BLOCK_5
        assert _shown(node) == _BLOCK_5

        monkeypatch.setattr(node, "resume", real_resume)
        await _polls(node, 1)
        assert guard.guard_state == GuardState.RUNNING
        assert node._cmd_channel.owns_pause is False


async def test_refused_lift_after_late_save_keeps_its_text(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R9 sibling: the late start-up save lands but the lift is refused (stale
    kill-switch read): the pause stays owned, the text never claims an operator must
    resume, and the next good claim lifts it."""
    async with _node(monkeypatch, seeded) as (node, _trace):
        guard = _guard(node)
        _checkpoint_block(node)
        real_resume = node.resume
        stale = (False, "system kill switch state not fresh (no clean read)")
        monkeypatch.setattr(node, "resume", lambda: stale)

        node._startup_job_late(_late(None), STARTUP_CHECKPOINT_STEP)

        assert guard.guard_state == GuardState.PAUSED
        assert node._cmd_channel.owns_pause is True
        assert guard.message == CHECKPOINT_BLOCK_MESSAGE
        assert "operator" not in str(_shown(node))

        monkeypatch.setattr(node, "resume", real_resume)
        await _polls(node, 1)
        assert guard.guard_state == GuardState.RUNNING
        assert node._cmd_channel.owns_pause is False


@pytest.mark.parametrize("variant", ["kill", "no_kill", "unrun"])
async def test_no_lift_during_claimed_batch(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    """C7b (R2): after a good claim the channel streak is reset BEFORE the batch runs;
    a venue pass (and lift) during the batch must not reopen entries ahead of a queued
    kill. Control: without a kill the batch's own recovery lifts. An outcome write that
    raises leaves the tail unrun: still no lift until it ran."""
    async with _node(monkeypatch, seeded) as (node, _trace):
        guard = _guard(node)
        _set_venue_block(node, VENUE_CHECK_PENDING)
        _Claims(monkeypatch, node, "F" * 5)
        await _polls(node, 5)
        assert node._cmd_channel.blocking is True
        assert node._cmd_channel.owns_pause is True

        rows = [("close_all", None)] + ([("kill", None)] if variant != "no_kill" else [])
        ids = _enqueue(seeded["db"], *rows)
        close_id = ids[0]
        entered = threading.Event()
        hold = threading.Event()
        real_complete, real_fail = PaperCommandQueue.complete, PaperCommandQueue.fail

        def outcome(real: Any) -> Any:
            def write(self: PaperCommandQueue, command_id: int, *args: Any) -> None:
                if command_id == close_id:
                    entered.set()
                    if variant == "unrun":
                        raise sqlite3.OperationalError("disk I/O error")
                    hold.wait(30)
                real(self, command_id, *args)

            return write

        monkeypatch.setattr(PaperCommandQueue, "complete", outcome(real_complete))
        monkeypatch.setattr(PaperCommandQueue, "fail", outcome(real_fail))
        resumes: list[int] = []
        real_resume = node.resume
        monkeypatch.setattr(node, "resume", lambda: (resumes.append(1), real_resume())[1])

        task = asyncio.create_task(node.process_commands())
        try:
            await _until(entered.is_set)
            if variant == "unrun":
                with pytest.raises(sqlite3.OperationalError):
                    await task
                assert node._unrun_claimed, "the kill is claimed but not run"
            else:
                await asyncio.sleep(0.05)
                assert not task.done(), "the outcome write is held"
            assert node._cmd_channel.blocking is False, "the streak was reset by the claim"

            _venue_pass(node)
            node._lift_channel_pause()

            assert resumes == []
            assert guard.guard_state == GuardState.PAUSED
            assert node._cmd_channel.owns_pause is True
            denial = _entry_denial(node)
            assert denial is not None
            assert "trading paused" in denial
            if variant != "no_kill":
                assert _row(seeded["db"], ids[1]).status == "running", "kill not run yet"
        finally:
            hold.set()
            if not task.done():
                await asyncio.wait_for(task, 15)

        if variant == "unrun":
            await _polls(node, 1)  # the next tick runs the unrun tail first
        if variant == "no_kill":
            assert guard.guard_state == GuardState.RUNNING, "lifted once the batch finished"
            assert node._cmd_channel.owns_pause is False
        else:
            assert guard.guard_state == GuardState.HALTED
            assert guard.halt_reason == HaltReason.KILL_SWITCH
            assert _row(seeded["db"], ids[1]).status == "done"


async def test_no_lift_during_leftover_batch(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C7b (R2), leftover run: [close_all (write raises), close_all, kill]. The next tick
    runs the unrun tail first; while that run is held, a venue pass and a lift must not
    reopen entries ahead of the queued kill. After it: HALTED by the kill."""
    async with _node(monkeypatch, seeded) as (node, _trace):
        guard = _guard(node)
        _set_venue_block(node, VENUE_CHECK_PENDING)
        _Claims(monkeypatch, node, "F" * 5)
        await _polls(node, 5)
        assert node._cmd_channel.owns_pause is True

        ids = _enqueue(seeded["db"], ("close_all", None), ("close_all", None), ("kill", None))
        held = threading.Event()
        hold = threading.Event()
        real_complete, real_fail = PaperCommandQueue.complete, PaperCommandQueue.fail

        def outcome(real: Any) -> Any:
            def write(self: PaperCommandQueue, command_id: int, *args: Any) -> None:
                if command_id == ids[0]:
                    raise sqlite3.OperationalError("disk I/O error")
                if command_id == ids[1]:
                    held.set()
                    hold.wait(30)
                real(self, command_id, *args)

            return write

        monkeypatch.setattr(PaperCommandQueue, "complete", outcome(real_complete))
        monkeypatch.setattr(PaperCommandQueue, "fail", outcome(real_fail))
        resumes: list[int] = []
        real_resume = node.resume
        monkeypatch.setattr(node, "resume", lambda: (resumes.append(1), real_resume())[1])

        with pytest.raises(sqlite3.OperationalError):
            await node.process_commands()
        assert len(node._unrun_claimed) == 2, "close_all and kill claimed, not run"

        task = asyncio.create_task(node.process_commands())
        try:
            await _until(held.is_set)
            assert node._unrun_claimed == [], "the leftover is being run"

            _venue_pass(node)
            node._lift_channel_pause()

            assert resumes == []
            assert guard.guard_state == GuardState.PAUSED
            assert node._cmd_channel.owns_pause is True
            assert _entry_denial(node) is not None
            assert _row(seeded["db"], ids[2]).status == "running", "kill not run yet"
        finally:
            hold.set()
            await asyncio.wait_for(task, 15)

        assert guard.guard_state == GuardState.HALTED
        assert guard.halt_reason == HaltReason.KILL_SWITCH
        assert _row(seeded["db"], ids[2]).status == "done"


async def test_late_save_retells_pending(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C7c (R9): the bounded save landing late releases only its own block; the pause
    text becomes PENDING and the ownership is unchanged."""
    async with _node(monkeypatch, seeded) as (node, _trace):
        guard = _guard(node)
        _checkpoint_block(node)
        _set_venue_block(node, VENUE_CHECK_PENDING)
        assert guard.message == CHECKPOINT_BLOCK_MESSAGE
        assert _shown(node) == f"{CHECKPOINT_BLOCK_MESSAGE} | {VENUE_CHECK_PENDING}"

        node._startup_job_late(_late(None), STARTUP_CHECKPOINT_STEP)

        assert guard.guard_state == GuardState.PAUSED
        assert guard.message == VENUE_CHECK_PENDING
        assert _shown(node) == VENUE_CHECK_PENDING
        assert node._cmd_channel.owns_pause is True


async def test_checkpoint_release_under_operator_pause_retells(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A released non-venue blocker under an operator pause: the text says it cleared and
    the operator still has to resume (never a stale "not saved yet")."""
    async with _node(monkeypatch, seeded) as (node, _trace):
        guard = _guard(node)
        _checkpoint_block(node)
        node._cmd_channel.owns_pause = False  # an operator pause took the pause over

        node._startup_job_late(_late(None), STARTUP_CHECKPOINT_STEP)

        text = f"{CHECKPOINT_BLOCK_MESSAGE} cleared; still paused until an operator resumes"
        assert guard.guard_state == GuardState.PAUSED
        assert guard.message == text
        assert _shown(node) == text


async def test_release_of_another_text_keeps_the_pause_text(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Released text != the pause text (and no venue release): the text is kept."""
    async with _node(monkeypatch, seeded) as (node, _trace):
        guard = _guard(node)
        node.pause()
        assert guard.message == "paused by operator"

        node._retell_pause(released=CHECKPOINT_BLOCK_MESSAGE)
        assert guard.message == "paused by operator"
        node._retell_pause(released=VENUE_CHECK_PENDING)
        assert guard.message == "paused by operator"
        node._retell_pause()
        assert guard.message == "paused by operator"


async def test_venue_release_matches_any_venue_text(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R9 venue alias: a released venue blocker clears any ``"venue ..."`` pause text
    (the check may have retold a sharper venue text than the released one); an operator
    pause outliving it says verified, still paused. A non-venue text is kept."""
    async with _node(monkeypatch, seeded) as (node, _trace):
        guard = _guard(node)
        node.pause()
        sharper = "venue leverage not published (BTCUSDT); new entries blocked"
        assert guard.set_pause_message(sharper)

        node._retell_pause(released=VENUE_CHECK_PENDING)

        assert guard.guard_state == GuardState.PAUSED
        assert guard.message == VENUE_CHECK_VERIFIED_STILL_PAUSED
        assert _shown(node) == VENUE_CHECK_VERIFIED_STILL_PAUSED

        assert guard.set_pause_message("vent paused by operator")
        node._retell_pause(released=VENUE_CHECK_PENDING)
        assert guard.message == "vent paused by operator"


async def test_retell_never_touches_a_halt(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retell only rewrites a PAUSED guard's text: halt text, reason and state stay."""
    async with _node(monkeypatch, seeded) as (node, _trace):
        guard = _guard(node)
        node.halt(HaltReason.MANUAL, "operator halt")
        node._venue_block = _M
        node._retell_pause()
        node._retell_pause(released="operator halt")

        assert guard.guard_state == GuardState.HALTED
        assert guard.halt_reason == HaltReason.MANUAL
        assert guard.message == "operator halt"
        assert node.status.error_message == "operator halt"
        assert _shown(node) == f"operator halt | {_M}"


async def test_retell_failure_is_logged_by_class_only(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``_retell_pause`` never raises; its log names the class, never the text."""
    async with _node(monkeypatch, seeded) as (node, _trace):
        guard = _guard(node)
        _set_venue_block(node, VENUE_CHECK_PENDING)

        def boom(_message: str) -> bool:
            raise RuntimeError("ZZMARK1") from ValueError("ZZMARK2")

        monkeypatch.setattr(guard, "set_pause_message", boom)
        node._cmd_channel.checkpoint_blocking = True
        with caplog.at_level(logging.ERROR):
            node._retell_pause()

        assert "pause retell failed (RuntimeError)" in caplog.text
        assert "ZZMARK" not in caplog.text
        assert all(r.exc_info is None for r in caplog.records)
        assert guard.guard_state == GuardState.PAUSED


@pytest.mark.parametrize("holder", ["venue", "batch", "unrun", "none"])
async def test_lift_refused_while_venue_blocks(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    holder: str,
) -> None:
    """C8 (+R2 edges): the channel's own pause is not lifted while the venue blocks, a
    claimed batch runs, or claimed rows are unrun; with none of them it lifts."""
    async with _node(monkeypatch, seeded) as (node, _trace):
        guard = _guard(node)
        _set_venue_block(node, VENUE_CHECK_PENDING)
        assert node._cmd_channel.owns_pause is True
        if holder != "venue":
            node._venue_block = None
        if holder == "batch":
            node._cmd_batch_running = True
        if holder == "unrun":
            node._unrun_claimed = [object()]  # type: ignore[list-item]
        resumes: list[int] = []
        real_resume = node.resume
        monkeypatch.setattr(node, "resume", lambda: (resumes.append(1), real_resume())[1])

        node._lift_channel_pause()

        if holder == "none":
            assert resumes == [1]
            assert guard.guard_state == GuardState.RUNNING
            assert node._cmd_channel.owns_pause is False
        else:
            assert resumes == []
            assert guard.guard_state == GuardState.PAUSED
            assert node._cmd_channel.owns_pause is True
        node._unrun_claimed = []


async def test_batch_flag_clears_after_the_batch(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The channel's own pause is not lifted inside a batch; the window closes after a
    good batch (the batch's recovery lifts) and after a raising one (a lift works)."""
    async with _node(monkeypatch, seeded) as (node, _trace):
        guard = _guard(node)
        _set_venue_block(node, VENUE_CHECK_PENDING)
        node._venue_block = None  # only the batch window can hold the lift now
        seen: list[GuardState] = []

        async def execute(cmd: Any) -> tuple[bool, dict[str, Any], None]:
            node._lift_channel_pause()
            seen.append(guard.guard_state)
            return True, {}, None

        monkeypatch.setattr(node, "_execute_command", execute)
        _enqueue(seeded["db"], ("pause", None))
        await node.process_commands()
        assert seen == [GuardState.PAUSED], "no lift inside the batch"
        assert guard.guard_state == GuardState.RUNNING, "lifted once the batch ran"

        _set_venue_block(node, VENUE_CHECK_PENDING)
        node._venue_block = None

        async def raising(cmd: Any) -> Any:
            raise asyncio.CancelledError

        monkeypatch.setattr(node, "_execute_command", raising)
        _enqueue(seeded["db"], ("pause", None))
        with pytest.raises(asyncio.CancelledError):
            await node.process_commands()
        assert guard.guard_state == GuardState.PAUSED
        node._lift_channel_pause()
        assert guard.guard_state == GuardState.RUNNING, "the window closed on the raise"


async def test_timeout_text_with_venue_pending(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R42b: the start-up connect timeout text replaces PENDING (not any other message)
    and is shown beside PENDING, across command polls (N13); once the venue connects
    late (R42c) its own text is cleared and PENDING alone shows."""
    async with _node(monkeypatch, seeded) as (node, _trace):
        kernel = fake_tradingnode.instances[0].kernel
        kernel.data_engine = _Engine()  # not connected yet
        kernel.exec_engine = _Engine()
        node._connected = None
        _set_venue_block(node, VENUE_CHECK_PENDING)
        assert node.status.error_message == VENUE_CHECK_PENDING
        node.status.started_at = datetime.now(UTC) - timedelta(
            seconds=STARTUP_CONNECT_TIMEOUT_S + 1
        )
        node._check_connectivity()
        timeout = node._startup_timeout_message()
        assert node.status.error_message == timeout
        assert _shown(node) == f"{timeout} | {VENUE_CHECK_PENDING}"

        await node.process_commands()  # an empty poll must not retell the text away
        node._check_connectivity()
        assert _shown(node) == f"{timeout} | {VENUE_CHECK_PENDING}"

        kernel.data_engine.connected = True
        kernel.exec_engine.connected = True
        node._check_connectivity()
        assert node._connected is True
        assert node.status.error_message is None
        assert _shown(node) == VENUE_CHECK_PENDING


async def test_timeout_text_never_replaces_other_messages(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R42d edge: a message that is not PENDING (even one starting like it) is kept."""
    async with _node(monkeypatch, seeded) as (node, _trace):
        for kept in (f"{VENUE_CHECK_PENDING} ", "operator halt"):
            node._startup_timeout_reported = False
            node.status.error_message = kept
            node.status.started_at = datetime.now(UTC) - timedelta(
                seconds=node_module.STARTUP_CONNECT_TIMEOUT_S + 1
            )
            node._check_startup_connect_timeout()
            assert node.status.error_message == kept


async def test_restart_restores_the_raw_halt_text(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A kill halt over a venue latch: the checkpoint keeps the composed status text for
    the API and the raw halt text for the restore. Restarts restore the halt with the
    kill text only (the latch is not restored) and the text never grows."""
    async with _node(monkeypatch, seeded) as (node, _trace):
        _set_venue_block(node, _M)
        node.halt(HaltReason.KILL_SWITCH, _K)
        assert _shown(node) == f"{_K} | {_M}"
        assert node.status.to_dict()["halt_message"] == _K

    for _restart in range(2):
        again = PaperTradingNode(_config(seeded))
        await again._initialize()
        try:
            assert again.status.halt_reason == HaltReason.KILL_SWITCH
            assert again.status.error_message == f"{RESTORED_HALT_PREFIX}{_K}"
            _set_venue_block(again, VENUE_CHECK_PENDING)  # T2b arms PENDING (N1b)
            assert _shown(again) == f"{RESTORED_HALT_PREFIX}{_K} | {VENUE_CHECK_PENDING}"
        finally:
            await again._shutdown()


# ============================================ T2b: arm, one-shot check, publish, latch

_PAIR = ["BTCUSDT", "ETHUSDT"]
_HANG = object()  # a read that never answers (its wait_for times out)
_HOLD = object()  # the next answer is held until ``release`` is set


def _run_id(
    seeded: dict[str, Any],  # noqa: F811
    symbols: list[str],
    leverage: int,
    liquidation: str,
) -> int:
    """A completed validation run trading ``symbols`` at ``leverage`` (margin from mode)."""
    from vibe_quant.db.state_manager import StateManager

    sm = StateManager(seeded["db"])
    try:
        rid = sm.create_backtest_run(
            strategy_id=seeded["sid"],
            run_mode="validation",
            symbols=symbols,
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-12-31",
            parameters={"liquidation_mode": liquidation, "leverage": leverage},
        )
        sm.update_backtest_run_status(rid, "completed")
    finally:
        sm.close()
    return rid


def _cfg_row(symbol: str, leverage: object = 10, margin: object = "ISOLATED") -> dict[str, object]:
    return {"symbol": symbol, "leverage": leverage, "marginType": margin}


_MATCH = [_cfg_row("BTCUSDT"), _cfg_row("ETHUSDT")]


class _Account:
    """NT's leverage map on the guard's account; ``mode`` breaks it for N2b."""

    def __init__(self, order: list[str] | None = None) -> None:
        self.leverages: dict[Any, Decimal] = {}
        self.sets: list[tuple[str, Decimal]] = []
        self.mode = "ok"  # ok | raises | ignores
        self.order = order if order is not None else []

    def set_leverage(self, instrument_id: Any, leverage: Decimal) -> None:
        if self.mode == "raises":
            raise ValueError("ZZACCOUNT")
        self.sets.append((str(instrument_id), leverage))
        self.order.append(f"set {instrument_id} {leverage}")
        if self.mode == "ok":
            self.leverages[instrument_id] = leverage

    def leverage(self, instrument_id: Any) -> Decimal | None:
        if self.mode == "ignores":
            return Decimal(1)  # the setter had no effect: the stale cached value
        return self.leverages.get(instrument_id)


class _AccountPortfolio(_FlatPortfolio):
    def __init__(self, account: object, nets: dict[str, Decimal] | None = None) -> None:
        self._account = account
        self._nets = nets or {}

    def account(self, venue: Any) -> object:
        return self._account

    def net_position(self, instrument_id: Any) -> Decimal:
        """NT's reconciled net per instrument (T4); flat unless the test set one."""
        return self._nets.get(str(instrument_id), Decimal(0))


class _GateCache(_EmptyCache):
    """No orders, no instruments: enough for the margin gate to price a priced order."""

    def orders(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    def instrument(self, instrument_id: Any) -> None:
        return None


class _RunningGuard(_VGuard):
    """A unit guard whose actor runs (NT connected) unless a test says otherwise; its
    portfolio holds the test's account."""

    @property
    def cache(self) -> Any:  # type: ignore[override]
        return _GateCache()

    @property
    def is_running(self) -> bool:  # type: ignore[override]
        return bool(self.__dict__.get("actor_running", True))

    @property
    def portfolio(self) -> Any:  # type: ignore[override]
        if "test_portfolio" in self.__dict__:
            return self.__dict__["test_portfolio"]
        return _AccountPortfolio(self.__dict__.get("test_account"), self.__dict__.get("test_nets"))


class _Clock:
    """The verifier's monotonic clock (``_venue_now``), moved by the test or the script."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, node: PaperTradingNode) -> None:
        self.now = 0.0
        monkeypatch.setattr(node, "_venue_now", lambda: self.now)


class _Script:
    """The node's real verifier loop, answered by scripted reads.

    A wait with nothing left to answer parks the loop for good (``parked``); any other
    wait completes at once and advances ``clock`` (when given) by its seconds. The
    positionRisk read (T4) answers ``positions`` in turn (flat once used up); with
    ``positions_after_hold`` it answers a cycle whose symbolConfig read is held only
    after that read ended.
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        node: PaperTradingNode,
        *answers: Any,
        clock: _Clock | None = None,
        positions: list[Any] | None = None,
        positions_after_hold: bool = False,
    ):
        self.answers = list(answers)
        #: positionRisk answers (T4), one per cycle; flat (``[]``) once used up.
        self.positions = list(positions or [])
        self.position_reads = 0
        self.positions_after_hold = positions_after_hold
        self.config_ended = asyncio.Event()
        self.reads = 0
        self.builds = 0
        self.waits: list[float] = []
        self.release = asyncio.Event()
        self.held = asyncio.Event()
        self.holds = 0
        self.parked = asyncio.Event()
        self.clock = clock

        def build() -> Any:
            self.builds += 1
            return self.read

        monkeypatch.setattr(node, "_build_venue_read", build)
        monkeypatch.setattr(
            node, "_build_position_read", lambda: self.read_positions, raising=False
        )
        monkeypatch.setattr(node, "_venue_wait", self.wait)

    async def read_positions(self) -> object:
        self.position_reads += 1
        await asyncio.sleep(0)  # the same cycle's symbolConfig read starts first
        if self.positions_after_hold and self.held.is_set():
            await self.config_ended.wait()
        item = self.positions.pop(0) if self.positions else []
        if item is _HANG:
            await asyncio.Event().wait()
        if callable(item):
            item = await item()
        if isinstance(item, BaseException):
            raise item
        return item

    async def read(self) -> object:
        self.config_ended.clear()
        try:
            return await self._read_config()
        finally:
            self.config_ended.set()

    async def _read_config(self) -> object:
        self.reads += 1
        if not self.answers:
            await asyncio.Event().wait()
        item = self.answers.pop(0)
        if callable(item):
            item = await item()
        if item is _HOLD:
            self.holds += 1
            self.held.set()
            await self.release.wait()
            self.release.clear()
            self.held.clear()
            item = self.answers.pop(0)
        if item is _HANG:
            await asyncio.Event().wait()
        if isinstance(item, BaseException):
            raise item
        return item

    async def wait(self, seconds: float) -> None:
        self.waits.append(seconds)
        if not self.answers:
            self.parked.set()
            await asyncio.Event().wait()
        if self.clock is not None:
            self.clock.now += seconds
        await asyncio.sleep(0)


class _Spy:
    """Every Alert the node submits (``journal.record(alert=)`` and ``journal.alert``)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, node: PaperTradingNode) -> None:
        self.alerts: list[Any] = []
        real = node._journal.alert

        def alert(item: Any) -> None:
            self.alerts.append(item)
            real(item)

        monkeypatch.setattr(node._journal, "alert", alert)

    def texts(self) -> list[tuple[str, str, bool, str | None]]:
        return [(a.kind, a.message, a.critical, a.monitor) for a in self.alerts]


def _events(trace: _Trace, action: str = "venue_config_check") -> list[dict[str, Any]]:
    return [c[2] for c in trace.calls if c[0] == "write" and c[2].get("action") == action]


@contextlib.asynccontextmanager
async def _armed(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],  # noqa: F811
    *,
    symbols: list[str] | None = None,
    leverage: int = 10,
    account: object | None = None,
) -> AsyncIterator[tuple[PaperTradingNode, _Trace]]:
    """A traced node past ``_initialize`` (armed, PENDING), default BTC+ETH 10x ISOLATED."""
    monkeypatch.setattr(node_module, "TradingGuard", _RunningGuard)
    rid = _run_id(seeded, symbols or _PAIR, leverage, "isolated")
    node, trace = _trace_node(monkeypatch, seeded, validation_run_id=rid)
    _fresh_kill_snapshot(monkeypatch, node)
    node._setup_signal_handlers()
    try:
        await node._initialize()
        _guard(node).__dict__["test_account"] = _Account() if account is None else account
        yield node, trace
    finally:
        _remove_handlers()
        await node._shutdown()


async def _settle(node: PaperTradingNode, script: _Script) -> None:
    """Start the verifier and wait until it stopped by itself or parked with nothing
    left to read."""
    node._start_venue_task()
    task = node._venue_task
    assert task is not None
    await _until(lambda: task.done() or script.parked.is_set())
    if task.done():
        assert task.exception() is None


def _reduce_only_denial(node: PaperTradingNode) -> str | None:
    order = SimpleNamespace(
        is_reduce_only=True,
        instrument_id=_BTC,
        quantity=SimpleNamespace(as_decimal=lambda: Decimal("0.01")),
        side=OrderSide.SELL,
    )
    denial: str | None = _guard(node)._checked_denial_reason(order)
    return denial


def _no_flatten_spies(monkeypatch: pytest.MonkeyPatch, node: PaperTradingNode) -> list[str]:
    """Record any call that would flatten, cancel, halt or stop strategies."""
    calls: list[str] = []
    guard = _guard(node)
    for name in ("halt", "close_all", "cancel_all_orders"):
        if hasattr(guard, name):
            monkeypatch.setattr(guard, name, lambda *a, _n=name, **k: calls.append(_n))
    monkeypatch.setattr(node, "halt", lambda *a, **k: calls.append("node.halt"))
    monkeypatch.setattr(node, "close_all_positions", lambda: calls.append("close_all_positions"))
    for strategy in node._compiled_strategies:
        monkeypatch.setattr(strategy, "stop", lambda: calls.append("strategy.stop"))
        monkeypatch.setattr(strategy, "market_exit", lambda *a, **k: calls.append("market_exit"))
    return calls


def _assert_latched(
    node: PaperTradingNode, trace: _Trace, spy: _Spy, message: str, kind: str = "mismatch"
) -> None:
    """Latched(M): blocked, PAUSED, shown, one RISK_CHECK, exactly one latch alert."""
    guard = _guard(node)
    assert node._venue_block == message
    assert node._venue_latched is True
    assert guard.guard_state == GuardState.PAUSED
    shown = _shown(node)
    assert isinstance(shown, str)
    assert message in shown
    failed = [e for e in _events(trace) if e["passed"] is False and e["kind"] == kind]
    assert failed == [
        {"action": "venue_config_check", "passed": False, "kind": kind, "message": message}
    ]
    latch = [a for a in spy.texts() if a[0] == "circuit_breaker"]
    assert latch == [("circuit_breaker", f"ENTRIES BLOCKED (venue config): {message}", True, None)]
    denial = _entry_denial(node)
    assert denial is not None
    assert "trading paused" in denial
    assert _reduce_only_denial(node) is None, "reduce-only SL/TP stay eligible"


async def test_start_is_blocked_until_checked(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N1: ``_initialize`` arms the check: PAUSED on PENDING, owned, shown, nothing read.
    The bounded start-up checkpoint is the first save; the arm's pause is saved after it."""
    from vibe_quant.paper.persistence import StatePersistence

    saved: list[object] = []
    real_save = StatePersistence.save_checkpoint

    def save(self: StatePersistence, checkpoint: Any) -> Any:
        saved.append(checkpoint.node_status.get("error_message"))
        return real_save(self, checkpoint)

    monkeypatch.setattr(StatePersistence, "save_checkpoint", save)
    async with _armed(monkeypatch, seeded) as (node, _trace):
        script = _Script(monkeypatch, node)
        guard = _guard(node)
        assert guard.guard_state == GuardState.PAUSED
        assert guard.message == VENUE_CHECK_PENDING
        assert node._cmd_channel.owns_pause is True
        assert _shown(node) == VENUE_CHECK_PENDING
        assert node._venue_block == VENUE_CHECK_PENDING
        denial = _entry_denial(node)
        assert denial is not None
        assert "trading paused" in denial
        assert (script.builds, script.reads) == (0, 0)
        await _until(lambda: len(saved) >= 2)
        assert saved[0] is None, "the bounded start-up checkpoint is saved first"
        assert saved[1] == VENUE_CHECK_PENDING


async def test_restored_halt_then_resume_stays_blocked(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N1b: a restored MANUAL halt stays HALTED and shows PENDING beside it; a resume
    before the pass is re-paused on PENDING."""
    from tests.unit.test_paper_halt_restore import _PRIOR_MANUAL, _write_checkpoint

    _write_checkpoint(seeded["db"], _PRIOR_MANUAL)
    async with _armed(monkeypatch, seeded) as (node, _trace):
        guard = _guard(node)
        halt = f"{RESTORED_HALT_PREFIX}halt via API"
        assert guard.guard_state == GuardState.HALTED
        assert _shown(node) == f"{halt} | {VENUE_CHECK_PENDING}"

        assert node.resume() == (False, f"resume re-paused, still paused: {VENUE_CHECK_PENDING}")
        assert guard.guard_state == GuardState.PAUSED
        assert _entry_denial(node) is not None


async def test_pass_after_late_checkpoint_failure_needs_resume(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N1c: the bounded start-up save stalls, then fails: one late-failure alert, resume
    owed. A later venue pass does not lift (the pause is not owned); an operator resume
    after a checkpoint saved runs."""
    from tests.unit.paper_venue_harness import venue_pass
    from tests.unit.test_paper_startup_drain_bound import _Chat, _node, _slow_checkpoint

    monkeypatch.setattr(node_module, "COMMAND_CLAIM_STALL_S", 0.05)
    hold = threading.Event()
    _slow_checkpoint(monkeypatch, hold, fail=True)
    chat = _Chat()
    node = _node(monkeypatch, seeded, chat, verified=False)
    try:
        await asyncio.wait_for(node._initialize(), 10)
        hold.set()
        await _until(lambda: node._startup_resume_owed)
        await node._journal.wait_alerts(None)
        assert len([b for b in chat.bodies if "failed" in b and "OSError" in b]) == 1
        guard = _guard(node)

        venue_pass(node, monkeypatch)
        assert node._venue_block is None
        assert guard.guard_state == GuardState.PAUSED, "not owned: no lift"
        assert node._cmd_channel.owns_pause is False

        node._save_checkpoint_now()  # a routine save lands: only the resume is owed now
        await _until(lambda: not node._startup_checkpoint_failed)
        assert node.resume()[0] is True
        assert guard.guard_state == GuardState.RUNNING
    finally:
        hold.set()
        _remove_handlers()
        await node._shutdown()


async def test_coin_m_refused_before_seed(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N1d: any account type but USDT_FUTURES is refused by ``start()`` before a seed or
    an NT node."""
    from vibe_quant.paper import warmup
    from vibe_quant.paper.config import BinanceTestnetConfig, ConfigurationError

    seeds: list[str] = []
    monkeypatch.setattr(warmup, "seed_all", lambda *a, **k: seeds.append("seed"))
    node = PaperTradingNode(
        _config(
            seeded,
            binance=BinanceTestnetConfig(
                "key", "secret", testnet=True, account_type="COIN_FUTURES"
            ),
        )
    )
    with pytest.raises(ConfigurationError) as caught:
        await node.start()
    assert str(caught.value) == (
        "paper supports account_type USDT_FUTURES only (venue check reads /fapi/)"
    )
    assert fake_tradingnode.instances == []
    assert seeds == []


class _VenueHttp:
    """Stands in for NT's cached Binance HTTP client: the real builder signs through it."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *bodies: bytes) -> None:
        import nautilus_trader.adapters.binance.factories as factories

        self.bodies = list(bodies)
        self.requests: list[tuple[str, str, list[str]]] = []
        self.release = asyncio.Event()
        self.release.set()
        self.built: list[dict[str, Any]] = []

        def get_client(**kwargs: Any) -> _VenueHttp:
            self.built.append(kwargs)
            return self

        monkeypatch.setattr(factories, "get_cached_binance_http_client", get_client)

    async def sign_request(self, method: Any, path: str, payload: dict[str, str]) -> bytes:
        if path == "/fapi/v3/positionRisk":  # T4: the session holds no position
            return b"[]"
        if path != "/fapi/v1/symbolConfig":  # the liquidation watch shares the client
            raise RuntimeError(f"not served here: {path}")
        self.requests.append((str(method), path, sorted(payload)))
        await self.release.wait()
        if not self.bodies:
            await asyncio.Event().wait()
        return self.bodies.pop(0)


@pytest.mark.parametrize("case", ["stale_higher", "control"])
async def test_pass_publishes_leverage_before_entries_reopen(
    case: str,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N2 (R1): public ``start()``, the real compiled strategy configs (string ids), a real
    NT guard and margin account (cached 10x by the connect-time read), the real verifier
    loop, builder and classifier reading venue bytes. While the first read is held every
    entry is DENIED "trading paused". Venue 1x (resolved 1x): the guard reads 1x BEFORE the
    RUNNING transition, so every later 8,000-notional entry is DENIED by the 50 % margin
    cap and none fills. Control (10x): entries fill."""
    import json

    from nautilus_trader.model.identifiers import Venue

    from tests.unit import test_audit_paper_guard as harness
    from vibe_quant.nt_compat import retain_log_guard

    leverage = 1 if case == "stale_higher" else 10
    rid = _run_id(seeded, ["BTCUSDT"], leverage, "isolated")
    http = _VenueHttp(monkeypatch, json.dumps([_cfg_row("BTCUSDT", leverage)]).encode())
    http.release.clear()
    h = harness.build(limits={"max_margin_pct": Decimal("0.5")})
    harness._ENGINES.remove(h.engine)  # disposed here, not by the harness's own fixture
    node, _trace = _trace_node(monkeypatch, seeded, validation_run_id=rid)
    at_running: list[object] = []

    class _Tee(node_module._GuardBridge):
        """The node's bridge, plus the harness recorder and the leverage at RUNNING."""

        def on_state_change(
            self,
            state: GuardState,
            reason: HaltReason | None,
            message: str,
            metrics: dict[str, object],
        ) -> None:
            if state == GuardState.RUNNING:
                account = h.guard.portfolio.account(Venue("BINANCE"))
                at_running.append(h.guard._verified_leverage(account, _BTC))
            super().on_state_change(state, reason, message, metrics)

        def on_order_denied(self, event: Any, *, by_gate: bool) -> None:
            h.listener.on_order_denied(event, by_gate=by_gate)
            super().on_order_denied(event, by_gate=by_gate)

    def build_guard(_risk_engine: Any) -> Any:
        h.guard._listener = _Tee(node)
        return h.guard

    monkeypatch.setattr(node, "_build_guard", build_guard)
    reached, stop = asyncio.Event(), asyncio.Event()

    async def run_async(self: Any) -> None:
        reached.set()
        await stop.wait()

    async def stop_async(self: Any) -> None:
        stop.set()

    async def short_wait(_seconds: float) -> None:
        await asyncio.sleep(0.01)

    monkeypatch.setattr(fake_tradingnode, "run_async", run_async)
    monkeypatch.setattr(fake_tradingnode, "stop_async", stop_async)
    monkeypatch.setattr(node, "_venue_wait", short_wait)
    _fresh_kill_snapshot(monkeypatch, node)
    assert isinstance(node, PaperTradingNode)
    task = asyncio.create_task(node.start())
    try:
        await asyncio.wait_for(reached.wait(), 20)
        assert h.guard.guard_state == GuardState.PAUSED
        strategies: Any = node._compiled_strategies
        assert isinstance(strategies[0].config.instrument_id, str), "production config is a str"
        bars = harness._flat_bars(h.strategies[0].config.bar_type, [8000.0] * 6)
        h.engine.add_data(bars[:2])
        h.engine.run(streaming=True)  # NT connected: the guard actor runs from here on
        h.engine.clear_data()
        await _until(lambda: len(http.requests) == 1)
        assert http.requests[0][1] == "/fapi/v1/symbolConfig"
        assert h.guard.guard_state == GuardState.PAUSED, "the read is held"
        held = [o.status.name for o in h.entry_orders()]
        assert held, "the gate must really be exercised"
        assert set(held) == {"DENIED"}
        assert all("trading paused" in str(e.reason) for e, _ in h.listener.denied)

        http.release.set()
        await _until(lambda: h.guard.guard_state == GuardState.RUNNING)
        assert at_running == [Decimal(leverage)], "published before entries reopened"
        before = len(h.entry_orders())
        h.engine.add_data(bars[2:])
        h.engine.run(streaming=True)
        after = h.entry_orders()[before:]
        assert after, "entries were attempted after the pass"
        if case == "stale_higher":
            assert [o.status.name for o in after] == ["DENIED"] * len(after)
            margin = [e for e, _ in h.listener.denied if "margin limit" in str(e.reason)]
            assert len(margin) == len(after)
            assert h.filled_entries() == []
        else:
            assert len(h.filled_entries()) >= 1
    finally:
        http.release.set()
        node._shutdown_event.set()
        stop.set()
        await asyncio.wait_for(task, 30)
        retain_log_guard(h.engine)
        h.engine.dispose()


@pytest.mark.parametrize("when", ["first", "after_pass"])
@pytest.mark.parametrize("kind", ["map", "account", "setter", "readback"])
async def test_publish_failure_keeps_block(
    kind: str,
    when: str,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N2b: a matching answer whose leverage does not reach the guard keeps (first) or
    re-establishes (after an earlier pass, T3) the entry block at once with the publish
    diagnostic, writes no new event, keeps the old deadline and retries after 5 s; the
    next full published pass lifts and is a recovery."""
    account = _Account()
    async with _armed(monkeypatch, seeded, account=account) as (node, trace):
        clock = _Clock(monkeypatch, node)
        guard = _guard(node)
        strategies = node._compiled_strategies
        lead = [_MATCH, _HOLD] if when == "after_pass" else []
        script = _Script(
            monkeypatch,
            node,
            *lead,
            _MATCH,
            _HOLD,
            _MATCH,
            clock=clock,
            positions_after_hold=True,  # the session positions are read after the fix
        )
        node._start_venue_task()
        before: list[dict[str, Any]] = []
        if when == "after_pass":
            await _until(lambda: script.holds == 1)
            assert guard.guard_state == GuardState.RUNNING
            assert node._venue_deadline == 90.0
            before = _events(trace)
            assert [e["kind"] for e in before] == ["pass"]

        if kind == "map":
            node._compiled_strategies = strategies[:1]  # ETHUSDT has no strategy id
        if kind == "account":
            guard.__dict__["test_account"] = None
        account.mode = {"setter": "raises", "readback": "ignores"}.get(kind, "ok")
        if when == "after_pass":
            script.release.set()
        await _until(lambda: script.holds == len(lead) // 2 + 1)

        diagnostic = {
            "map": "ETHUSDT: no valid instrument id (not configured)",
            "account": "no venue account in the NT cache",
            "setter": "BTCUSDT: leverage publish failed (ValueError)",
            "readback": "BTCUSDT: guard reads leverage 1 after publishing 10",
        }[kind]
        text = f"venue leverage not published ({diagnostic}); new entries blocked"
        assert node._venue_block == text
        assert guard.guard_state == GuardState.PAUSED
        assert guard.message == text
        assert _shown(node) == text
        assert _entry_denial(node) is not None
        assert _events(trace) == before
        assert script.waits == ([60.0, 5.0] if when == "after_pass" else [5.0])
        assert node._venue_deadline == (90.0 if when == "after_pass" else None), "not refreshed"

        node._compiled_strategies = strategies
        account.mode = "ok"
        guard.__dict__["test_account"] = account
        script.release.set()
        await _until(script.parked.is_set)

        assert node._venue_block is None
        assert guard.guard_state == GuardState.RUNNING
        assert node._venue_deadline == clock.now + 90.0
        assert _events(trace) == [
            *before,
            {
                "action": "venue_config_check",
                "passed": True,
                "kind": "recovered",
                "recovered_from": "publish_failed",
                "failures": 1,
                "symbols": _PAIR,
                "leverage": 10,
                "margin_type": "ISOLATED",
            },
        ]


async def test_no_read_before_actor_runs(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N3: the guard is RUNNING-state but its actor is not running (NT still connecting):
    no read, the actor is polled every 1 s; once it runs, one read."""
    async with _armed(monkeypatch, seeded) as (node, _trace):
        guard = _guard(node)
        guard.__dict__["actor_running"] = False
        script = _Script(monkeypatch, node, _HANG)
        node._start_venue_task()
        await _until(lambda: len(script.waits) >= 3)
        assert script.reads == 0
        assert set(script.waits) == {1.0}
        guard.__dict__["actor_running"] = True
        await _until(lambda: script.reads == 1)


async def test_match_publishes_then_lifts_silently(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N4: a matching first answer publishes each symbol's leverage BEFORE the RUNNING
    transition, clears the blocker, writes one pass event, alerts nothing."""
    order: list[str] = []
    async with _armed(monkeypatch, seeded, account=_Account(order)) as (node, trace):
        spy = _Spy(monkeypatch, node)
        real = node._on_guard_state_change

        def change(state: GuardState, *args: Any) -> None:
            order.append(f"state {state.value}")
            real(state, *args)

        monkeypatch.setattr(node, "_on_guard_state_change", change)
        script = _Script(monkeypatch, node, _MATCH)
        await _settle(node, script)

        assert order == [
            "set BTCUSDT-PERP.BINANCE 10",
            "set ETHUSDT-PERP.BINANCE 10",
            "state running",
        ]
        assert node._venue_block is None
        assert _guard(node).guard_state == GuardState.RUNNING
        assert _entry_denial(node) is None
        assert script.reads == 1
        assert script.waits == [60.0], "T3: the next read follows a pass after 60 s"
        assert [(e["kind"], e["failures"]) for e in _events(trace)] == [("pass", 0)]
        assert spy.alerts == []


async def test_match_lift_waits_for_fresh_kill_read(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N4b: a pass on a stale kill-switch read cannot resume: still PAUSED and owned; the
    next good claim with a fresh read lifts it."""
    from tests.unit.test_paper_command_channel import _StaleSnapshot

    async with _armed(monkeypatch, seeded) as (node, _trace):
        guard = _guard(node)
        monkeypatch.setattr(node._kill_monitor, "snapshot", _StaleSnapshot)
        script = _Script(monkeypatch, node, _MATCH)
        await _settle(node, script)
        assert node._venue_block is None
        assert guard.guard_state == GuardState.PAUSED
        assert node._cmd_channel.owns_pause is True

        await node.process_commands()  # a good claim, the read still stale
        assert guard.guard_state == GuardState.PAUSED
        _fresh_kill_snapshot(monkeypatch, node)
        await node.process_commands()
        assert guard.guard_state == GuardState.RUNNING
        assert node._cmd_channel.owns_pause is False


async def test_mismatch_latches_without_flatten(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N5: ETH at 5x latches: entries blocked for the session, one event, one alert;
    nothing is flattened, cancelled, halted or stopped; nothing is published."""
    account = _Account()
    async with _armed(monkeypatch, seeded, account=account) as (node, trace):
        spy = _Spy(monkeypatch, node)
        calls = _no_flatten_spies(monkeypatch, node)
        script = _Script(monkeypatch, node, [_cfg_row("BTCUSDT"), _cfg_row("ETHUSDT", 5)])
        await _settle(node, script)

        _assert_latched(node, trace, spy, _M)
        assert calls == []
        assert account.sets == []
        assert script.reads == 1


async def test_resume_cannot_clear_latch(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N6: an operator resume while latched is re-paused on the latch."""
    async with _armed(monkeypatch, seeded) as (node, _trace):
        script = _Script(monkeypatch, node, [_cfg_row("BTCUSDT"), _cfg_row("ETHUSDT", 5)])
        await _settle(node, script)
        assert node.resume() == (False, f"resume re-paused, still paused: {_M}")
        assert _guard(node).guard_state == GuardState.PAUSED
        assert _guard(node).message == _M


async def test_transient_failures_retry_then_pass(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """N7: OSError, a read that times out, then a match: retries after 5 s each, the pass
    is a recovery from 2 failures, no alert below the paging threshold."""
    monkeypatch.setattr(node_module, "VENUE_CHECK_TIMEOUT_S", 0.05)
    async with _armed(monkeypatch, seeded) as (node, trace):
        spy = _Spy(monkeypatch, node)
        script = _Script(monkeypatch, node, OSError("ZZREAD"), _HANG, _MATCH)
        with caplog.at_level(logging.WARNING):
            await _settle(node, script)
        assert "venue config read failed (OSError)" in caplog.text
        assert "venue config read failed (TimeoutError)" in caplog.text
        assert "ZZREAD" not in caplog.text
        assert all(r.exc_info is None for r in caplog.records)

        assert script.waits == [5.0, 5.0, 60.0]
        assert script.reads == 3
        assert [(e["kind"], e["recovered_from"], e["failures"]) for e in _events(trace)] == [
            ("recovered", "failures", 2)
        ]
        assert spy.alerts == []
        assert _guard(node).guard_state == GuardState.RUNNING


@pytest.mark.parametrize("pause", [False, True], ids=["owned", "operator_pause"])
async def test_unverified_alerts_once_then_recovers(
    pause: bool,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N8 (R11): 3 failed reads before any pass page once (PENDING stays); 2 failures do
    not. The match then lifts and sends the recovery alert naming what entries do now
    and, since it replaces an owed page (ro-vc-1 F1), since when they were paused."""
    monkeypatch.setattr(node_module, "VENUE_CHECK_TIMEOUT_S", 0.02)
    _Wall(monkeypatch)
    async with _armed(monkeypatch, seeded) as (node, trace):
        spy = _Spy(monkeypatch, node)
        if pause:
            _enqueue(seeded["db"], ("pause", None))
            await node.process_commands()
        script = _Script(monkeypatch, node, _HANG, _HANG, _HOLD, _HANG, _HOLD, _MATCH)
        node._start_venue_task()
        await asyncio.wait_for(script.held.wait(), 10)
        assert spy.alerts == [], "2 failures do not page"
        script.release.set()
        await _until(lambda: len(spy.alerts) == 1 and script.held.is_set())
        unverified = (
            "error",
            "paper PAPER-007: venue leverage/margin unverified after 3 reads (last "
            "TimeoutError); new entries stay blocked (paused) since 2026-11-20 14:01:30 UTC",
            True,
            "venuecfg:unverified",
        )
        assert spy.texts() == [unverified]
        assert node._venue_block == VENUE_CHECK_PENDING
        assert [e["kind"] for e in _events(trace)] == ["unverified"]

        script.release.set()
        await _until(script.parked.is_set)
        entries = "entries stay blocked (paused)" if pause else "new entries allowed"
        assert spy.texts() == [
            unverified,
            (
                "error",
                "paper PAPER-007: venue leverage/margin verified after 3 failed reads (last "
                f"TimeoutError); entries were paused from 2026-11-20 14:01:30 UTC; {entries}",
                True,
                "venuecfg:verified",
            ),
        ]
        assert node._venue_alerted is False
        assert _guard(node).guard_state == (GuardState.PAUSED if pause else GuardState.RUNNING)
        assert [e["kind"] for e in _events(trace)] == ["unverified", "recovered"]


async def test_latch_is_terminal_after_unverified_alert(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N8c (R5): 3 failures (paged), then a mismatch latches; the later mismatch and match
    are never read, a match that still completes is discarded: one latch alert, no
    verified alert, no publish, no new read."""
    monkeypatch.setattr(node_module, "VENUE_CHECK_TIMEOUT_S", 0.02)
    account = _Account()
    bad = [_cfg_row("BTCUSDT"), _cfg_row("ETHUSDT", 5)]
    async with _armed(monkeypatch, seeded, account=account) as (node, trace):
        spy = _Spy(monkeypatch, node)
        script = _Script(monkeypatch, node, _HANG, _HANG, _HANG, bad, bad, _MATCH)
        await _settle(node, script)
        for _ in range(2):  # the poll advanced two more cycles
            await asyncio.sleep(0)
        resolved = node.resolved_config
        assert resolved is not None
        late = node_module.classify_symbol_configs(_MATCH, resolved.symbols, 10, "ISOLATED")
        assert node._venue_apply(late, None) is False
        node._latch("venue config mismatch: other", kind="mismatch")

        assert script.reads == 4
        assert account.sets == []
        latch = [a for a in spy.texts() if a[0] == "circuit_breaker"]
        assert len(latch) == 1
        assert [a for a in spy.texts() if a[3] == "venuecfg:verified"] == []
        _assert_latched(node, trace, spy, _M)


async def test_malformed_answer_is_a_failed_read(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N9: through the real builder, an undecodable body is a failed read (class only) and
    a body that is not a list leaves every symbol unverified; neither passes nor latches."""
    _Wall(monkeypatch)
    async with _armed(monkeypatch, seeded) as (node, _trace):
        spy = _Spy(monkeypatch, node)
        http = _VenueHttp(monkeypatch, b"{oops", b'{"ZZMARK": 1}', b"{oops")
        waits: list[float] = []

        async def wait(seconds: float) -> None:
            waits.append(seconds)
            await asyncio.sleep(0)

        monkeypatch.setattr(node, "_venue_wait", wait)
        node._start_venue_task()
        await _until(lambda: len(spy.alerts) == 1)
        assert spy.texts()[0][1] == (
            "paper PAPER-007: venue leverage/margin unverified after 3 reads (last "
            "JSONDecodeError); new entries stay blocked (paused) since 2026-11-20 14:01:30 UTC"
        )
        assert node._venue_latched is False
        assert node._venue_block == VENUE_CHECK_PENDING
        # the 4th request (no answer scripted) may already be out
        assert [r[1] for r in http.requests[:3]] == ["/fapi/v1/symbolConfig"] * 3
        assert waits[:2] == [5.0, 5.0]
        assert http.requests[0][2] == ["recvWindow", "timestamp"]
        # T4: one signed client per endpoint (symbolConfig, positionRisk), session key
        assert [(k["api_key"], k["api_secret"]) for k in http.built] == [("key", "secret")] * 2
        await node._stop_venue_check()

        node._venue_fails = 2
        node._venue_alerted = False
        node._venue_failed("x")  # reset the count's edge: the 3rd failure pages
        assert len(spy.alerts) == 2


@pytest.mark.parametrize("order", ["btc_first", "eth_first"])
async def test_mismatch_latches_despite_bad_sibling(
    order: str,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N9b: BTC at 5x latches although ETH's leverage is unreadable; both are named."""
    rows = [_cfg_row("BTCUSDT", 5), _cfg_row("ETHUSDT", None)]
    if order == "eth_first":
        rows.reverse()
    async with _armed(monkeypatch, seeded) as (node, trace):
        spy = _Spy(monkeypatch, node)
        script = _Script(monkeypatch, node, rows, _MATCH)
        await _settle(node, script)
        message = (
            "venue config mismatch: BTCUSDT: venue leverage 5x != resolved 10x | unverified: "
            "ETHUSDT (leverage is not a whole number in 0..125: None)"
        )
        _assert_latched(node, trace, spy, message)
        assert script.reads == 1


async def test_bug_fails_closed(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N10: an exception in the apply step latches (fail closed); the task ends cleanly."""
    async with _armed(monkeypatch, seeded) as (node, trace):
        spy = _Spy(monkeypatch, node)
        resolved = node._resolved
        script = _Script(monkeypatch, node)
        real_read = script.read

        async def read() -> object:
            node._resolved = None
            script.answers.append(_MATCH)
            return await real_read()

        monkeypatch.setattr(node, "_build_venue_read", lambda: read)
        node._start_venue_task()
        task = node._venue_task
        assert task is not None
        await _until(task.done)
        node._resolved = resolved
        assert task.exception() is None
        _assert_latched(node, trace, spy, "venue config check failed: AttributeError", "bug")


@pytest.mark.parametrize("where", ["signal", "risk_check"])
async def test_pass_survives_journal_failures(
    where: str,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """N10b (R6): (a) the resume's SIGNAL write raises: RUNNING, logged by class; (b) the
    pass event's submit raises: lifted, entries admitted, logged, one event-log alert.
    Never latched. (c) is ``test_pass_event_append_failure_uses_the_event_log_alert``."""
    async with _armed(monkeypatch, seeded) as (node, _trace):
        spy = _Spy(monkeypatch, node)
        journal = node._journal
        real_write = journal._write_event

        def write(event_type: Any, data: dict[str, object], timestamp: Any = None) -> Any:
            if where == "signal" and data.get("to_state") == "running":
                raise OSError("ZZWRITE")
            if where == "risk_check" and data.get("action") == "venue_config_check":
                raise OSError("ZZWRITE")
            return real_write(event_type, data, timestamp)

        monkeypatch.setattr(journal, "_write_event", write)
        script = _Script(monkeypatch, node, _MATCH)
        with caplog.at_level(logging.ERROR):
            await _settle(node, script)

        assert node._venue_latched is False
        assert node._venue_block is None
        assert _guard(node).guard_state == GuardState.RUNNING
        assert _entry_denial(node) is None, "a new entry is admitted"
        assert "ZZWRITE" not in caplog.text
        if where == "signal":
            assert "venue check lift failed (OSError)" in caplog.text
            assert spy.alerts == []
        else:
            assert "venue check pass event submit failed (OSError)" in caplog.text
            assert spy.texts() == [
                (
                    "error",
                    "paper PAPER-007: venue config verified but its event was not written "
                    "(OSError); verification stands",
                    True,
                    "eventlog:venuecfg",
                )
            ]


async def test_pass_event_append_failure_uses_the_event_log_alert(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N10b(c): the worker fails the pass line's append: the pass stands, the existing
    event-log health alert reports it, no venue alert."""
    monkeypatch.setattr(node_module, "TradingGuard", _RunningGuard)
    rid = _run_id(seeded, _PAIR, 10, "isolated")
    node = PaperTradingNode(_config(seeded, validation_run_id=rid))
    _fresh_kill_snapshot(monkeypatch, node)
    sent: list[str] = []
    monkeypatch.setattr(
        node._journal, "_send_telegram_alert", lambda kind, message, **kw: sent.append(message)
    )
    node._setup_signal_handlers()
    try:
        await node._initialize()
        _guard(node).__dict__["test_account"] = _Account()
        node._journal.event_log.barrier().result(10)
        node._journal.check_health()  # the baseline tick
        writer = node._journal.writer
        assert writer is not None
        real = writer.write

        def write(event: Any) -> None:
            if event.data.get("action") == "venue_config_check":
                raise OSError(28, "No space left on device")
            real(event)

        monkeypatch.setattr(writer, "write", write)
        script = _Script(monkeypatch, node, _MATCH)
        await _settle(node, script)
        node._journal.event_log.barrier().result(10)
        node._journal.check_health()

        assert _guard(node).guard_state == GuardState.RUNNING
        assert node._venue_latched is False
        assert len(sent) == 1
        assert "venue" not in sent[0]
    finally:
        _remove_handlers()
        await node._shutdown()


async def test_write_failure_keeps_block_and_alert(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """N12: the latch's event write raises: the alert is still sent once, the block
    stays, the text is logged by class only."""
    async with _armed(monkeypatch, seeded) as (node, _trace):
        spy = _Spy(monkeypatch, node)
        journal = node._journal
        real_write = journal._write_event

        def write(event_type: Any, data: dict[str, object], timestamp: Any = None) -> Any:
            if data.get("kind") == "mismatch":
                raise OSError("ZZWRITE")
            return real_write(event_type, data, timestamp)

        monkeypatch.setattr(journal, "_write_event", write)
        script = _Script(monkeypatch, node, [_cfg_row("BTCUSDT"), _cfg_row("ETHUSDT", 5)])
        with caplog.at_level(logging.ERROR):
            await _settle(node, script)

        assert spy.texts() == [
            ("circuit_breaker", f"ENTRIES BLOCKED (venue config): {_M}", True, None)
        ]
        assert node._venue_block == _M
        assert _guard(node).guard_state == GuardState.PAUSED
        assert "venue check latch event failed (OSError)" in caplog.text
        assert "ZZWRITE" not in caplog.text


async def test_latch_not_restored_on_restart(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N14: a latch is never restored: the next session re-checks from PENDING."""
    async with _armed(monkeypatch, seeded) as (node, _trace):
        script = _Script(monkeypatch, node, [_cfg_row("BTCUSDT"), _cfg_row("ETHUSDT", 5)])
        await _settle(node, script)
        assert node._venue_latched is True
        rid = node.config.validation_run_id

    again = PaperTradingNode(_config(seeded, validation_run_id=rid))
    _fresh_kill_snapshot(monkeypatch, again)
    try:
        await again._initialize()
        assert again._venue_latched is False
        assert again._venue_block == VENUE_CHECK_PENDING
        assert _guard(again).guard_state == GuardState.PAUSED
        assert _shown(again) == VENUE_CHECK_PENDING
        assert again.resume()[1] == f"resume re-paused, still paused: {VENUE_CHECK_PENDING}"
    finally:
        await again._shutdown()


async def test_nt_logging_not_debug(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L1: NT logs the signed URL only at DEBUG; the paper node never runs NT at DEBUG."""
    async with _armed(monkeypatch, seeded):
        config = fake_tradingnode.instances[0].config
        logging_config = config.logging
        assert logging_config is None or str(logging_config.log_level).upper() not in (
            "DEBUG",
            "TRACE",
        )


# ================================================ T2c: stop path, signal routing, veto


def _installed(sig: int) -> bool:
    handlers: dict[int, Any] = getattr(asyncio.get_running_loop(), "_signal_handlers", {})
    return sig in handlers


def _fire(sig: int) -> None:
    """Run the loop's installed handler for ``sig`` now (what a delivered signal does)."""
    handle = asyncio.get_running_loop()._signal_handlers[sig]  # type: ignore[attr-defined]
    handle._run()


class _VerifierTasks:
    """Every verifier task the node starts, and a positionRisk read that hangs and, once
    cancelled, takes several loop turns to clean up (a teardown that does not wait for
    the verifier sees it unfinished)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, node: PaperTradingNode) -> None:
        self.started: list[asyncio.Task[None]] = []
        self.cleaned = False
        real_start = node._start_venue_task

        def start() -> None:
            real_start()
            assert node._venue_task is not None
            self.started.append(node._venue_task)

        monkeypatch.setattr(node, "_start_venue_task", start)

    def done(self) -> bool:
        return bool(self.started) and all(task.done() for task in self.started)

    async def slow_cancel(self) -> object:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            for _ in range(5):
                await asyncio.sleep(0)
            self.cleaned = True
            raise
        return None


@pytest.mark.parametrize("read", ["match", "mismatch", "hung", "passed"])
@pytest.mark.parametrize("trigger", ["sigterm", "event", "run_raises", "init_fails"])
async def test_stop_cancels_before_nt_teardown(
    trigger: str,
    read: str,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N11 (R4): however the node stops (SIGTERM through its installed handler, the
    shutdown event, ``run_async`` raising, ``_initialize`` cancelled after the arm), with a
    venue answer already in hand, a hung read, or after a pass: from the stop mark on, no
    publish, lift, latch or venue alert; new entries stay denied (STOPPING once the guard
    runs), reduce-only orders pass; the verifier is done before NT's first ``stop_async``.
    SIGTERM is one-shot like NT's; both handlers are gone after shutdown."""
    monkeypatch.setattr(node_module, "TradingGuard", _RunningGuard)
    rid = _run_id(seeded, _PAIR, 10, "isolated")
    node, trace = _trace_node(monkeypatch, seeded, validation_run_id=rid)
    _fresh_kill_snapshot(monkeypatch, node)
    spy = _Spy(monkeypatch, node)
    account = _Account()
    answer = {
        "match": [_HOLD, _MATCH],
        "mismatch": [_HOLD, [_cfg_row("BTCUSDT"), _cfg_row("ETHUSDT", 5)]],
        "hung": [_HANG],
        "passed": [_MATCH],
    }[read]
    verifiers = _VerifierTasks(monkeypatch, node)
    hold = [verifiers.slow_cancel] if read == "hung" else None
    script = _Script(monkeypatch, node, *answer, positions=hold)
    resumes: list[int] = []
    real_resume = node.resume
    monkeypatch.setattr(node, "resume", lambda: (resumes.append(1), real_resume())[1])
    stops: list[dict[str, Any]] = []
    reached, release_run, stop = asyncio.Event(), asyncio.Event(), asyncio.Event()
    real_create = node._create_live_trading_node

    def create() -> Any:
        built = real_create()
        _guard(node).__dict__["test_account"] = account
        return built

    monkeypatch.setattr(node, "_create_live_trading_node", create)

    async def run_async(self: Any) -> None:
        reached.set()
        if trigger == "run_raises":
            await release_run.wait()
            raise RuntimeError("venue client died")
        await stop.wait()

    async def stop_async(self: Any) -> None:
        stops.append(
            {
                "venue_done": verifiers.done(),
                "cleaned": verifiers.cleaned,
                "entry": _entry_denial(node),
                "reduce_only": _reduce_only_denial(node),
                "guard": _guard(node).guard_state,
            }
        )
        stop.set()

    monkeypatch.setattr(fake_tradingnode, "run_async", run_async)
    monkeypatch.setattr(fake_tradingnode, "stop_async", stop_async)

    async def ready() -> None:
        """The first answer is in hand (or the read hangs, or the pass already ran)."""
        if read == "passed":
            await _until(script.parked.is_set)
            assert _guard(node).guard_state == GuardState.RUNNING
        elif read != "hung":
            await asyncio.wait_for(script.held.wait(), 10)
        else:
            await _until(lambda: script.reads == 1)

    marked: dict[str, Any] = {}

    def effects() -> dict[str, int]:
        venue_alerts = [
            a for a in spy.alerts if "venue config" in a.message or "venue leverage" in a.message
        ]
        return {
            "sets": len(account.sets),
            "resumes": len(resumes),
            "alerts": len(venue_alerts),
            "events": len(_events(trace)),
        }

    def mark() -> None:
        marked.update(effects())
        if read in ("match", "mismatch") and trigger != "run_raises":
            # Ready: the verifier resumes only after the (synchronous) stop below. A
            # raising run_async is seen a loop hop later, so its read stays held: the
            # verifier is cancelled at the mark.
            script.release.set()

    if trigger == "init_fails":
        real_process = node.process_commands

        async def drain_cancelled(*, startup_halts_only: bool = False) -> None:
            if not startup_halts_only:
                return await real_process()
            node._start_venue_task()  # a verifier already running when start-up unwinds
            await ready()
            mark()
            raise asyncio.CancelledError

        monkeypatch.setattr(node, "process_commands", drain_cancelled)

    task = asyncio.create_task(node.start())
    try:
        if trigger == "init_fails":
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 30)
        else:
            await asyncio.wait_for(reached.wait(), 20)
            await ready()
            assert _installed(signal.SIGTERM)
            assert _installed(signal.SIGINT)
            mark()
            if trigger == "sigterm":
                _fire(signal.SIGTERM)
                assert node._stop_requested is True, "the mark is set inside the handler"
                assert not _installed(signal.SIGTERM), "one-shot, like NT's"
                _fire(signal.SIGINT)  # a no-op now
            elif trigger == "event":
                node._shutdown_event.set()
            else:
                release_run.set()
            if trigger == "run_raises":
                with pytest.raises(RuntimeError, match="venue client died"):
                    await asyncio.wait_for(task, 30)
            else:
                await asyncio.wait_for(task, 30)
    finally:
        release_run.set()
        stop.set()
        script.release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert node._stop_requested is True
    assert stops, "NT was stopped"
    first = stops[0]
    assert first["venue_done"] is True, "the verifier is done before NT's first stop_async"
    assert first["cleaned"] is (read == "hung"), "teardown waited for the sibling's cleanup"
    assert first["entry"] is not None
    if first["guard"] == GuardState.RUNNING:
        assert first["entry"] == f"VQ_GUARD: {node_module.VENUE_CHECK_STOPPING}"
    assert (read == "passed") == (first["guard"] == GuardState.RUNNING)
    assert first["reduce_only"] is None
    assert effects() == marked, "nothing published, lifted, latched or alerted after the mark"
    assert node._venue_latched is False
    assert node._venue_expiry is None, "T3: no expiry timer outlives the stop"
    assert not _installed(signal.SIGTERM)
    assert not _installed(signal.SIGINT)


async def test_signal_during_init_skips_run(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N11b: SIGTERM lands during ``_initialize`` (after NT's build): ``run_async`` never
    starts, and ``_shutdown`` stops the verifier before ``stop_async``."""
    monkeypatch.setattr(node_module, "TradingGuard", _RunningGuard)
    rid = _run_id(seeded, _PAIR, 10, "isolated")
    node, _trace = _trace_node(monkeypatch, seeded, validation_run_id=rid)
    _fresh_kill_snapshot(monkeypatch, node)
    verifiers = _VerifierTasks(monkeypatch, node)
    _Script(monkeypatch, node, _HANG)
    runs: list[str] = []
    stops: list[bool] = []

    async def run_async(self: Any) -> None:
        runs.append("run")

    async def stop_async(self: Any) -> None:
        stops.append(verifiers.done())

    monkeypatch.setattr(fake_tradingnode, "run_async", run_async)
    monkeypatch.setattr(fake_tradingnode, "stop_async", stop_async)
    real_arm = node._arm_venue_check

    def arm_then_signal() -> None:
        real_arm()
        node._start_venue_task()
        _fire(signal.SIGTERM)

    monkeypatch.setattr(node, "_arm_venue_check", arm_then_signal)
    await asyncio.wait_for(node.start(), 30)

    assert runs == []
    assert stops
    assert stops[0] is True
    assert not _installed(signal.SIGTERM)
    assert not _installed(signal.SIGINT)


async def test_apply_bug_is_logged_by_class_only(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A raising apply step (chained marker exception) latches and logs its class only."""
    async with _armed(monkeypatch, seeded) as (node, trace):
        spy = _Spy(monkeypatch, node)

        def boom(_verdict: Any, _failure: Any) -> bool:
            raise RuntimeError("ZZBUG1") from ValueError("ZZBUG2")

        monkeypatch.setattr(node, "_venue_apply", boom)
        script = _Script(monkeypatch, node, _MATCH)
        with caplog.at_level(logging.ERROR):
            await _settle(node, script)
        assert "venue config check failed (RuntimeError)" in caplog.text
        assert "ZZBUG" not in caplog.text
        assert all(r.exc_info is None for r in caplog.records)
        _assert_latched(node, trace, spy, "venue config check failed: RuntimeError", "bug")


# ------------------------------------------- T3: periodic re-verify + 90 s expiry

_ETH5 = [_cfg_row("BTCUSDT"), _cfg_row("ETHUSDT", 5)]


def _expired(seconds: int, reason: str) -> str:
    return (
        f"venue leverage/margin not verified for {seconds} s (last: {reason}); new entries blocked"
    )


def _unverified(reason: str, entries: str) -> tuple[str, str, bool, str]:
    return (
        "error",
        f"paper PAPER-007: venue leverage/margin unverified after 3 reads (last {reason}); "
        f"{entries}",
        True,
        "venuecfg:unverified",
    )


async def test_periodic_match_is_silent(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P1: after the pass the venue is read again every 60 s; each matching read
    publishes, writes nothing, alerts nothing, replaces the expiry timer and moves the
    deadline to now + 90 s."""
    account = _Account()
    async with _armed(monkeypatch, seeded, account=account) as (node, trace):
        spy = _Spy(monkeypatch, node)
        clock = _Clock(monkeypatch, node)
        script = _Script(monkeypatch, node, _MATCH, _HOLD, _MATCH, _HOLD, _MATCH, clock=clock)
        node._start_venue_task()
        loop = asyncio.get_running_loop()
        handles: list[asyncio.TimerHandle] = []
        for holds, deadline in ((1, 90.0), (2, 150.0)):
            await _until(lambda h=holds: script.holds == h)  # type: ignore[misc]
            assert node._venue_deadline == deadline
            handle = node._venue_expiry
            assert handle is not None
            assert handle.when() - loop.time() == pytest.approx(90.0, abs=5.0)
            handles.append(handle)
            script.release.set()
        await _until(script.parked.is_set)

        assert node._venue_deadline == 210.0
        assert all(h.cancelled() for h in handles), "each pass replaced the old timer"
        assert node._venue_expiry is not None
        assert not node._venue_expiry.cancelled()
        assert script.waits == [60.0, 60.0, 60.0]
        assert len(account.sets) == 6, "every pass publishes"
        assert [e["kind"] for e in _events(trace)] == ["pass"]
        assert spy.alerts == []
        assert _guard(node).guard_state == GuardState.RUNNING


async def test_mid_session_change_latches(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P2: the user sets BTC to 20x in the app mid-session; the 5th read (4 intervals in)
    latches at once; nothing is flattened, nothing published after it."""
    account = _Account()
    async with _armed(monkeypatch, seeded, account=account) as (node, trace):
        spy = _Spy(monkeypatch, node)
        calls = _no_flatten_spies(monkeypatch, node)
        btc20 = [_cfg_row("BTCUSDT", 20), _cfg_row("ETHUSDT")]
        script = _Script(monkeypatch, node, _MATCH, _MATCH, _MATCH, _MATCH, btc20, _MATCH)
        await _settle(node, script)

        message = "venue config mismatch: BTCUSDT: venue leverage 20x != resolved 10x"
        _assert_latched(node, trace, spy, message)
        assert script.reads == 5
        assert script.waits == [60.0] * 4
        assert len(account.sets) == 8
        assert calls == []
        assert node._venue_deadline is None
        assert node._venue_expiry is None


async def test_latch_stops_the_verifier(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P2b (R5): pass, 3 failures (paged), then a mismatch latches: the verifier stops,
    the expiry timer is cancelled; a late match, a second latch and the old timer are
    no-ops; no recovery alert, no publish, no new read."""
    monkeypatch.setattr(node_module, "VENUE_CHECK_TIMEOUT_S", 0.02)
    account = _Account()
    async with _armed(monkeypatch, seeded, account=account) as (node, trace):
        spy = _Spy(monkeypatch, node)
        clock = _Clock(monkeypatch, node)
        script = _Script(
            monkeypatch, node, _MATCH, _HOLD, _HANG, _HANG, _HANG, _ETH5, _MATCH, clock=clock
        )
        node._start_venue_task()
        await _until(lambda: script.holds == 1)
        handle = node._venue_expiry
        assert handle is not None
        gen = node._venue_expiry_gen
        script.release.set()
        await _until(lambda: node._venue_task is not None and node._venue_task.done())
        for _ in range(2):  # the poll advanced two more cycles
            await asyncio.sleep(0)
        resolved = node.resolved_config
        assert resolved is not None
        late = node_module.classify_symbol_configs(_MATCH, resolved.symbols, 10, "ISOLATED")
        assert node._venue_apply(late, None) is False
        node._latch("venue config mismatch: other", kind="mismatch")
        clock.now = 500.0
        node._venue_expire(gen)

        assert script.reads == 5
        assert handle.cancelled()
        assert node._venue_expiry is None
        assert node._venue_deadline is None
        assert len(account.sets) == 2, "only the first pass published"
        assert spy.texts() == [
            _unverified("TimeoutError", "new entries still allowed until verification expires"),
            ("circuit_breaker", f"ENTRIES BLOCKED (venue config): {_M}", True, None),
        ]
        assert [e["kind"] for e in _events(trace)] == ["pass", "unverified", "mismatch"]
        _assert_latched(node, trace, spy, _M)


@pytest.mark.parametrize("pause", [False, True], ids=["running", "operator_pause"])
async def test_mid_session_failures_page_at_three(
    pause: bool,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P3 (R11): after a pass at t=0, reads fail at 60/65/70 s: the 3rd failure pages and
    says entries are still allowed until the verification expires (RUNNING) or stay
    blocked (operator pause); nothing blocks before 90 s."""
    monkeypatch.setattr(node_module, "VENUE_CHECK_TIMEOUT_S", 0.02)
    async with _armed(monkeypatch, seeded) as (node, trace):
        spy = _Spy(monkeypatch, node)
        clock = _Clock(monkeypatch, node)
        if pause:
            _enqueue(seeded["db"], ("pause", None))
            await node.process_commands()
        script = _Script(monkeypatch, node, _MATCH, _HANG, _HANG, _HANG, clock=clock)
        await _settle(node, script)

        assert clock.now == 70.0
        assert script.waits == [60.0, 5.0, 5.0, 5.0]
        entries = (
            "new entries stay blocked (paused)"
            if pause
            else "new entries still allowed until verification expires"
        )
        assert spy.texts() == [_unverified("TimeoutError", entries)]
        assert [e["kind"] for e in _events(trace)] == ["pass", "unverified"]
        assert node._venue_block is None
        guard = _guard(node)
        if pause:
            assert guard.guard_state == GuardState.PAUSED
            assert guard.message == VENUE_CHECK_VERIFIED_STILL_PAUSED
        else:
            assert guard.guard_state == GuardState.RUNNING
            clock.now = 89.999
            assert _entry_denial(node) is None, "no block before 90 s"


@pytest.mark.parametrize(("now", "entries"), [(89.999, "still"), (90.0, "blocked")])
async def test_outage_text_at_the_deadline(
    now: float,
    entries: str,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R11 edge: the page says entries are still allowed only strictly before the
    deadline (the admission veto denies from the deadline on)."""
    async with _armed(monkeypatch, seeded) as (node, _trace):
        spy = _Spy(monkeypatch, node)
        clock = _Clock(monkeypatch, node)
        script = _Script(monkeypatch, node, _MATCH, clock=clock)
        await _settle(node, script)
        clock.now = now
        node._venue_fails = 2
        node._venue_failed("OSError")
        expected = {
            "still": "new entries still allowed until verification expires",
            "blocked": "new entries stay blocked (running)",
        }[entries]
        assert spy.texts() == [_unverified("OSError", expected)]


@pytest.mark.parametrize("prior", ["none", "failure"])
async def test_deadline_denies_at_admission_then_pass_lifts(
    prior: str,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P4 (R3, R11): pass at t=0, the expiry timer withheld: a new entry passes the gate at
    89.999 s and is DENIED at 90.000 / 90.001 s with the expiry text (fallback reason
    without a completed failure); reduce-only stays eligible. The timer then blocks
    (PAUSED, one ``expired`` event below the paging threshold, no alert); the next full
    published pass lifts it as a recovery from ``expired``."""
    async with _armed(monkeypatch, seeded) as (node, trace):
        spy = _Spy(monkeypatch, node)
        clock = _Clock(monkeypatch, node)
        middle = [OSError("ZZP4")] if prior == "failure" else []
        script = _Script(monkeypatch, node, _MATCH, *middle, _HOLD, _MATCH, clock=clock)
        node._start_venue_task()
        await _until(lambda: script.holds == 1)
        reason = "OSError" if prior == "failure" else "no completed recheck"
        text = _expired(90, reason)

        clock.now = 89.999
        assert _entry_denial(node) is None
        assert _reduce_only_denial(node) is None
        for at in (90.0, 90.001):
            clock.now = at
            assert _entry_denial(node) == f"VQ_GUARD: {text}"
            assert _reduce_only_denial(node) is None
        clock.now = 125.4
        assert _entry_denial(node) == f"VQ_GUARD: {_expired(125, reason)}"

        clock.now = 90.001
        guard = _guard(node)
        assert guard.guard_state == GuardState.RUNNING, "the timer has not run yet"
        node._venue_expire(node._venue_expiry_gen)
        assert node._venue_block == text
        assert guard.guard_state == GuardState.PAUSED
        assert guard.message == text
        assert _shown(node) == text
        assert _reduce_only_denial(node) is None
        assert _events(trace)[1:] == [
            {"action": "venue_config_check", "passed": False, "kind": "expired", "message": text}
        ]
        assert spy.alerts == []

        script.release.set()
        await _until(script.parked.is_set)
        assert node._venue_block is None
        assert guard.guard_state == GuardState.RUNNING
        assert _entry_denial(node) is None
        assert node._venue_deadline == pytest.approx(clock.now + 90.0)
        recovered = _events(trace)[-1]
        assert (recovered["kind"], recovered["recovered_from"], recovered["failures"]) == (
            "recovered",
            "expired",
            1 if prior == "failure" else 0,
        )
        assert [e["kind"] for e in _events(trace)] == ["pass", "expired", "recovered"]
        assert spy.alerts == []


async def test_replaced_timer_is_ignored(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P4b (R3): pass at 0, pass at 60; the first pass's timer firing at 90 s is ignored
    (cancelled, and its generation is stale)."""
    async with _armed(monkeypatch, seeded) as (node, trace):
        clock = _Clock(monkeypatch, node)
        script = _Script(monkeypatch, node, _MATCH, _HOLD, _MATCH, clock=clock)
        node._start_venue_task()
        await _until(lambda: script.holds == 1)
        old_gen = node._venue_expiry_gen
        old = node._venue_expiry
        assert old is not None
        script.release.set()
        await _until(script.parked.is_set)
        assert old.cancelled()
        assert node._venue_expiry_gen == old_gen + 1

        clock.now = 90.0
        node._venue_expire(old_gen)
        assert _guard(node).guard_state == GuardState.RUNNING
        assert node._venue_block is None
        assert _entry_denial(node) is None
        assert [e["kind"] for e in _events(trace)] == ["pass"]


@pytest.mark.parametrize("state", ["latched", "stopping"])
async def test_expiry_noop_when_latched_or_stopping(
    state: str,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P5: once latched or stopping, the expiry timer is cancelled and a callback that
    still runs (current generation) changes nothing."""
    async with _armed(monkeypatch, seeded) as (node, trace):
        clock = _Clock(monkeypatch, node)
        script = _Script(monkeypatch, node, _MATCH, clock=clock)
        await _settle(node, script)
        handle = node._venue_expiry
        assert handle is not None
        gen = node._venue_expiry_gen
        if state == "latched":
            node._latch(_M, kind="mismatch")
        else:
            node._mark_stopping()
        assert handle.cancelled()
        assert node._venue_expiry is None
        clock.now = 100.0
        node._venue_expire(node._venue_expiry_gen)

        kinds = [e["kind"] for e in _events(trace)]
        if state == "latched":
            assert node._venue_expiry_gen == gen
            assert node._venue_block == _M
            assert node._venue_deadline is None
            assert node._entry_veto() is None, "the latch pause holds entries"
            assert kinds == ["pass", "mismatch"]
        else:
            assert node._venue_expiry_gen == gen + 1
            assert node._venue_block is None
            assert node._entry_veto() == node_module.VENUE_CHECK_STOPPING
            assert kinds == ["pass"]


async def test_second_outage_alerts_again(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N8b: under an operator pause, 3 failures page, the match recovers (entries stay
    blocked: paused); 3 more failures after that pass page a second time. Only the first
    episode had a venue block (PENDING since the arm): its page and recovery say since
    when; the second outage is under the operator pause alone."""
    monkeypatch.setattr(node_module, "VENUE_CHECK_TIMEOUT_S", 0.02)
    _Wall(monkeypatch)
    async with _armed(monkeypatch, seeded) as (node, trace):
        spy = _Spy(monkeypatch, node)
        clock = _Clock(monkeypatch, node)
        _enqueue(seeded["db"], ("pause", None))
        await node.process_commands()
        script = _Script(
            monkeypatch, node, _HANG, _HANG, _HANG, _MATCH, _HANG, _HANG, _HANG, clock=clock
        )
        await _settle(node, script)

        blocked = _unverified("TimeoutError", "new entries stay blocked (paused)")
        assert spy.texts() == [
            _unverified(
                "TimeoutError", "new entries stay blocked (paused) since 2026-11-20 14:01:30 UTC"
            ),
            (
                "error",
                "paper PAPER-007: venue leverage/margin verified after 3 failed reads (last "
                "TimeoutError); entries were paused from 2026-11-20 14:01:30 UTC; "
                "entries stay blocked (paused)",
                True,
                "venuecfg:verified",
            ),
            blocked,
        ]
        assert [e["kind"] for e in _events(trace)] == ["unverified", "recovered", "unverified"]
        assert node._venue_alerted is True


async def test_expiry_event_failure_is_logged_by_class_only(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The expiry event submit raises (chained marker): the block still holds and the
    log names the class only."""
    async with _armed(monkeypatch, seeded) as (node, _trace):
        clock = _Clock(monkeypatch, node)
        script = _Script(monkeypatch, node, _MATCH, clock=clock)
        await _settle(node, script)

        def record(*_a: Any, **_k: Any) -> None:
            raise RuntimeError("ZZEXP1") from ValueError("ZZEXP2")

        monkeypatch.setattr(node._journal, "record", record)
        clock.now = 95.0
        with caplog.at_level(logging.ERROR):
            node._venue_expire(node._venue_expiry_gen)
        assert node._venue_block == _expired(95, "no completed recheck")
        assert _guard(node).guard_state == GuardState.PAUSED
        assert "venue check expiry event failed (RuntimeError)" in caplog.text
        assert "ZZEXP" not in caplog.text
        assert all(r.exc_info is None for r in caplog.records)


# ============================================ T4: unknown positions + reconciled drift


def _prow(symbol: str, amount: str, side: str = "BOTH") -> dict[str, object]:
    """A positionRisk v3 row."""
    return {"symbol": symbol, "positionSide": side, "positionAmt": amount, "markPrice": "1"}


class _Snap:
    """A positionRisk answer that first records the node's state left by the previous
    cycle (block, latched, guard state, entry and reduce-only denials, waits so far)."""

    def __init__(self, node: PaperTradingNode, script: _Script) -> None:
        self.node = node
        self.script = script
        self.seen: list[dict[str, Any]] = []

    def __call__(self, body: object) -> Any:
        async def answer() -> object:
            node = self.node
            self.seen.append(
                {
                    "block": node._venue_block,
                    "latched": node._venue_latched,
                    "state": _guard(node).guard_state,
                    "entry": _entry_denial(node),
                    "reduce_only": _reduce_only_denial(node),
                    "waits": list(self.script.waits),
                }
            )
            return body

        return answer


def _confirming(*pairs: str) -> str:
    return f"venue position {', '.join(pairs)} (confirming); new entries blocked"


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        pytest.param(
            [_prow("SOLUSDT", "3.5"), _prow("BTCUSDT", "0")],
            "unknown position on venue: SOLUSDT 3.5 (not traded by this session)",
            id="sol",
        ),
        pytest.param(
            [_prow("XRPUSDT", "-120"), _prow("DOGEUSDT", "7000")],
            "unknown position on venue: DOGEUSDT 7000 (not traded by this session); "
            "XRPUSDT -120 (not traded by this session)",
            id="two_foreign",
        ),
    ],
)
async def test_foreign_symbol_position_latches(
    rows: list[dict[str, object]],
    message: str,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U1: a nonzero position on a symbol this session does not trade latches on the
    FIRST read (config matching); nothing is closed and no leverage is published."""
    account = _Account()
    async with _armed(monkeypatch, seeded, account=account) as (node, trace):
        spy = _Spy(monkeypatch, node)
        calls = _no_flatten_spies(monkeypatch, node)
        script = _Script(monkeypatch, node, _MATCH, _MATCH, positions=[rows])
        await _settle(node, script)

        _assert_latched(node, trace, spy, message, kind="unknown_position")
        assert calls == []
        assert account.sets == [], "a latched cycle publishes nothing"
        assert (script.reads, script.position_reads) == (1, 1), "no read after the latch"
        assert [e["kind"] for e in _events(trace)] == ["unknown_position"]


@pytest.mark.parametrize(
    ("row", "message"),
    [
        pytest.param(
            _prow("BTCUSDT", "0.010", "LONG"),
            "unknown position on venue: BTCUSDT LONG 0.010 (hedge-mode row)",
            id="long",
        ),
        pytest.param(
            _prow("ETHUSDT", "-2.5", "SHORT"),
            "unknown position on venue: ETHUSDT SHORT -2.5 (hedge-mode row)",
            id="short",
        ),
    ],
)
async def test_hedge_rows_are_unknown(
    row: dict[str, object],
    message: str,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U3: a nonzero LONG/SHORT row (hedge mode, which NT refuses) is unknown even on a
    session symbol: Latched."""
    async with _armed(monkeypatch, seeded) as (node, trace):
        spy = _Spy(monkeypatch, node)
        script = _Script(monkeypatch, node, _MATCH, positions=[[row, _prow("BTCUSDT", "0")]])
        await _settle(node, script)
        _assert_latched(node, trace, spy, message, kind="unknown_position")


@pytest.mark.parametrize("variant", ["confirm", "match_lifts"])
async def test_drift_blocks_first_then_latches(
    variant: str,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U2 (R8): after a pass, the venue shows BTC +0.010 while the session is flat. The
    first reading blocks at once (a buy is DENIED, reduce-only stays eligible) and the
    next read follows 5 s after that comparison; the same pair again latches. Variant:
    a matching published pass instead lifts only this blocker (a recovery event)."""
    async with _armed(monkeypatch, seeded) as (node, trace):
        spy = _Spy(monkeypatch, node)
        calls = _no_flatten_spies(monkeypatch, node)
        script = _Script(monkeypatch, node, _MATCH, _MATCH, _MATCH)
        snap = _Snap(node, script)
        second = [_prow("BTCUSDT", "0.010")] if variant == "confirm" else []
        script.positions = [[], [_prow("BTCUSDT", "0.010")], snap(second)]
        await _settle(node, script)

        [between] = snap.seen
        assert between["block"] == _confirming("BTCUSDT 0.010 != session 0")
        assert between["latched"] is False
        assert between["state"] == GuardState.PAUSED
        assert between["entry"] is not None
        assert "trading paused" in between["entry"]
        assert between["reduce_only"] is None
        assert between["waits"] == [60.0, 5.0]
        assert calls == []
        if variant == "confirm":
            _assert_latched(
                node, trace, spy, "venue position BTCUSDT 0.010 != session 0", "position_drift"
            )
            assert [e["kind"] for e in _events(trace)] == ["pass", "position_drift"]
            return
        assert node._venue_block is None
        assert node._venue_latched is False
        assert _guard(node).guard_state == GuardState.RUNNING
        assert _entry_denial(node) is None
        assert [(e["kind"], e["recovered_from"]) for e in _events(trace)] == [
            ("pass", None),
            ("recovered", "drift"),
        ]
        assert spy.alerts == []


async def test_drift_streak_rules(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U2b: a new differing pair restarts the streak at 1; an unverified or failed
    position comparison resets it but keeps the block; a match resets it (and the pass
    lifts). Only two consecutive equal pairs latch."""
    async with _armed(monkeypatch, seeded) as (node, trace):
        spy = _Spy(monkeypatch, node)
        script = _Script(monkeypatch, node, *([_MATCH] * 9))
        snap = _Snap(node, script)
        btc = [_prow("BTCUSDT", "0.010")]
        btc2 = [_prow("BTCUSDT", "0.020")]
        script.positions = [
            btc,  # 1: streak (0.010, 0) = 1
            snap(btc2),  # 2: new pair -> streak 1 again
            snap([_prow("BTCUSDT", "NaN")]),  # 3: unverified -> reset, block kept
            snap(btc2),  # 4: streak 1 (not a latch)
            snap(OSError("ZZPOS")),  # 5: failed read -> reset, block kept
            snap(btc2),  # 6: streak 1
            snap([]),  # 7: match -> reset, the pass lifts
            snap(btc2),  # 8: streak 1
            snap(btc2),  # 9: same pair -> latch
        ]
        await _settle(node, script)

        one = _confirming("BTCUSDT 0.010 != session 0")
        two = _confirming("BTCUSDT 0.020 != session 0")
        assert [(s["block"], s["latched"]) for s in snap.seen] == [
            (one, False),  # after 1
            (two, False),  # after 2
            (two, False),  # after 3
            (two, False),  # after 4
            (two, False),  # after 5
            (two, False),  # after 6
            (None, False),  # after 7: lifted
            (two, False),  # after 8
        ]
        assert [s["state"] for s in snap.seen][-2] == GuardState.RUNNING
        _assert_latched(
            node, trace, spy, "venue position BTCUSDT 0.020 != session 0", "position_drift"
        )
        assert script.position_reads == 9


@pytest.mark.parametrize(
    ("net", "venue", "pair"),
    [
        pytest.param(("BTCUSDT", "0.010"), [], "BTCUSDT 0 != session 0.010", id="btc_long"),
        pytest.param(
            ("ETHUSDT", "-1.5"),
            [_prow("BTCUSDT", "0")],
            "ETHUSDT 0 != session -1.5",
            id="eth_short",
        ),
    ],
)
async def test_omitted_row_is_flat(
    net: tuple[str, str],
    venue: list[dict[str, object]],
    pair: str,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U2c: a session symbol the venue omits is flat; NT's reconciled net differs: the
    first reading blocks (no pass), the same pair again latches."""
    async with _armed(monkeypatch, seeded) as (node, trace):
        spy = _Spy(monkeypatch, node)
        symbol, amount = net
        _guard(node).__dict__["test_nets"] = {f"{symbol}-PERP.BINANCE": Decimal(amount)}
        script = _Script(monkeypatch, node, _MATCH, _MATCH)
        snap = _Snap(node, script)
        script.positions = [venue, snap(venue)]
        await _settle(node, script)

        assert [s["block"] for s in snap.seen] == [_confirming(pair)]
        assert snap.seen[0]["waits"] == [5.0]
        _assert_latched(node, trace, spy, f"venue position {pair}", "position_drift")
        assert [e["kind"] for e in _events(trace)] == ["position_drift"], "never passed"


@pytest.mark.parametrize(
    ("symbol", "amount"),
    [pytest.param("BTCUSDT", "0.010", id="btc_long"), pytest.param("ETHUSDT", "-0.5", id="eth")],
)
async def test_claimed_restore_passes(
    symbol: str,
    amount: str,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U5: a restored, claimed position the session holds (NT's net) equals the venue's
    amount (written with a different scale): pass, lift, no event beyond the pass."""
    account = _Account()
    async with _armed(monkeypatch, seeded, account=account) as (node, trace):
        spy = _Spy(monkeypatch, node)
        _guard(node).__dict__["test_nets"] = {f"{symbol}-PERP.BINANCE": Decimal(amount)}
        venue_amount = f"{amount}00"  # "0.01000" == 0.010: equal amounts, other scale
        script = _Script(monkeypatch, node, _MATCH, positions=[[_prow(symbol, venue_amount)]])
        await _settle(node, script)

        assert node._venue_block is None
        assert node._venue_latched is False
        assert _guard(node).guard_state == GuardState.RUNNING
        assert len(account.sets) == 2, "published"
        assert [e["kind"] for e in _events(trace)] == ["pass"]
        assert spy.alerts == []


async def test_preexisting_same_symbol_position_is_adopted(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U6 (disclosed limitation, pinned): a manual BTC position opened BEFORE a fresh
    start is adopted by NT's start-up reconciliation, so venue and session agree and the
    check passes: same-symbol origin is unknowable (design T4 scope)."""
    async with _armed(monkeypatch, seeded) as (node, trace):
        # NT reconciliation adopted the manual short into the session's net position.
        _guard(node).__dict__["test_nets"] = {"BTCUSDT-PERP.BINANCE": Decimal("-0.250")}
        script = _Script(monkeypatch, node, _MATCH, positions=[[_prow("BTCUSDT", "-0.250")]])
        await _settle(node, script)
        assert node._venue_block is None
        assert _guard(node).guard_state == GuardState.RUNNING
        assert [e["kind"] for e in _events(trace)] == ["pass"]


class _ReconciledPortfolio(_AccountPortfolio):
    """The test account for the publish; nets from a real NT portfolio."""

    def __init__(self, account: object, portfolio: Any) -> None:
        super().__init__(account)
        self._portfolio = portfolio

    def net_position(self, instrument_id: Any) -> Decimal:
        return Decimal(self._portfolio.net_position(instrument_id))


async def _reconcile_venue_position(
    strategy: Any, symbol: str, side: str, quantity: str
) -> tuple[Any, Any, Any]:
    """NT start-up reconciliation of one venue position (no network): a real
    LiveExecutionEngine with the strategy's external order claims registered, fed by a
    mock Binance client's position report. Returns (engine, cache, portfolio)."""
    from nautilus_trader.cache.cache import Cache
    from nautilus_trader.common.component import LiveClock, MessageBus
    from nautilus_trader.common.providers import InstrumentProvider
    from nautilus_trader.core.uuid import UUID4
    from nautilus_trader.execution.reports import PositionStatusReport
    from nautilus_trader.live.config import LiveExecEngineConfig
    from nautilus_trader.live.execution_engine import LiveExecutionEngine
    from nautilus_trader.model.currencies import USDT
    from nautilus_trader.model.enums import AccountType, OmsType, PositionSide
    from nautilus_trader.model.identifiers import ClientId, TraderId, Venue
    from nautilus_trader.model.objects import Quantity
    from nautilus_trader.portfolio.portfolio import Portfolio
    from nautilus_trader.test_kit.mocks.exec_clients import MockLiveExecutionClient
    from nautilus_trader.test_kit.stubs.events import TestEventStubs

    from vibe_quant.data.catalog import create_instrument

    loop = asyncio.get_running_loop()
    clock = LiveClock()
    trader_id = TraderId("PAPER-007")
    msgbus = MessageBus(trader_id, clock)
    cache = Cache()
    portfolio = Portfolio(msgbus, cache, clock)
    engine = LiveExecutionEngine(loop, msgbus, cache, clock, LiveExecEngineConfig())
    client = MockLiveExecutionClient(
        loop,
        ClientId("BINANCE"),
        Venue("BINANCE"),
        AccountType.MARGIN,
        USDT,
        InstrumentProvider(),
        msgbus,
        cache,
        clock,
        oms_type=OmsType.NETTING,
    )
    engine.register_client(client)
    cache.add_instrument(create_instrument(symbol))
    portfolio.update_account(TestEventStubs.margin_account_state(client.account_id))
    strategy.register(trader_id, portfolio, msgbus, cache, clock)
    engine.register_external_order_claims(strategy)
    client.add_position_status_report(
        PositionStatusReport(
            account_id=client.account_id,
            instrument_id=InstrumentId.from_str(f"{symbol}-PERP.BINANCE"),
            position_side=PositionSide[side],
            quantity=Quantity.from_str(quantity),
            report_id=UUID4(),
            ts_last=0,
            ts_init=0,
            avg_px_open=Decimal("60000"),
        )
    )
    engine.start()
    assert await engine.reconcile_execution_state() is True
    assert client.commands == []
    return engine, cache, portfolio


@pytest.mark.parametrize(
    ("symbol", "side", "quantity", "venue_amount"),
    [
        pytest.param("BTCUSDT", "SHORT", "0.250", "-0.25000", id="btc_short"),
        pytest.param("ETHUSDT", "LONG", "0.500", "0.5", id="eth_long"),
    ],
)
async def test_reconciled_claimed_position_passes(
    symbol: str,
    side: str,
    quantity: str,
    venue_amount: str,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U5/U6 through NT: a position already on the venue at start is reconciled by a
    real LiveExecutionEngine and claimed by the session's own compiled strategy (its
    production ``external_order_claims``). The verifier compares NT's reconciled net
    with the venue's positionRisk amount and passes; the position stays open and no
    order is sent."""
    account = _Account()
    async with _armed(monkeypatch, seeded, account=account) as (node, trace):
        spy = _Spy(monkeypatch, node)
        strategy = node._compiled_strategies[_PAIR.index(symbol)]
        engine, cache, portfolio = await _reconcile_venue_position(strategy, symbol, side, quantity)
        try:
            [position] = cache.positions_open()
            assert position.strategy_id == strategy.id, "claimed, not EXTERNAL"
            _guard(node).__dict__["test_portfolio"] = _ReconciledPortfolio(account, portfolio)
            script = _Script(monkeypatch, node, _MATCH, positions=[[_prow(symbol, venue_amount)]])
            await _settle(node, script)

            assert node._venue_block is None
            assert node._venue_latched is False
            assert _guard(node).guard_state == GuardState.RUNNING
            assert len(account.sets) == 2, "published"
            assert [e["kind"] for e in _events(trace)] == ["pass"]
            assert spy.alerts == []
            [still] = cache.positions_open()
            assert still.id == position.id
            assert still.signed_decimal_qty() == Decimal(venue_amount)
            assert cache.orders_open() == []
        finally:
            engine.stop()
            engine.dispose()


@pytest.mark.parametrize("confirmed", ["config", "position"])
async def test_confirmed_result_beats_hung_sibling(
    confirmed: str,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U7 (R7): one endpoint confirms (config BTC 5x / foreign SOL position) while the
    other read hangs. The latch lands and a new entry is DENIED while the sibling is
    still held and long before any timeout; the sibling is then cancelled, and the
    exception it raises on cancel never erases the latch."""
    monkeypatch.setattr(node_module, "VENUE_CHECK_TIMEOUT_S", 1_000.0)
    async with _armed(monkeypatch, seeded) as (node, trace):
        spy = _Spy(monkeypatch, node)
        at_cancel: list[tuple[bool, str | None]] = []

        async def hung() -> object:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                at_cancel.append((node._venue_latched, _entry_denial(node)))
                raise OSError("ZZSIBLING") from None
            return None

        if confirmed == "config":
            bad = [_cfg_row("BTCUSDT", 5), _cfg_row("ETHUSDT")]
            _Script(monkeypatch, node, bad, positions=[hung])
            message = "venue config mismatch: BTCUSDT: venue leverage 5x != resolved 10x"
            kind = "mismatch"
        else:
            _Script(monkeypatch, node, hung, positions=[[_prow("SOLUSDT", "3.5")]])
            message = "unknown position on venue: SOLUSDT 3.5 (not traded by this session)"
            kind = "unknown_position"
        node._start_venue_task()
        task = node._venue_task
        assert task is not None
        await asyncio.wait_for(asyncio.shield(task), 10)
        assert task.exception() is None

        [(latched, denial)] = at_cancel
        assert latched is True, "the latch landed before the sibling was cancelled"
        assert denial is not None
        assert "trading paused" in denial
        _assert_latched(node, trace, spy, message, kind)


async def test_position_failure_blocks_pass(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U8: the positionRisk read fails while symbolConfig matches: no pass, no publish,
    PENDING stays, retry after 5 s; the next cycle with both endpoints verified passes
    as a recovery from failures."""
    account = _Account()
    async with _armed(monkeypatch, seeded, account=account) as (node, trace):
        script = _Script(monkeypatch, node, _MATCH, _MATCH)
        snap = _Snap(node, script)
        script.positions = [OSError("ZZPOS"), snap([])]
        await _settle(node, script)

        [between] = snap.seen
        assert between["block"] == VENUE_CHECK_PENDING
        assert between["state"] == GuardState.PAUSED
        assert between["waits"] == [5.0]
        assert node._venue_block is None
        assert _guard(node).guard_state == GuardState.RUNNING
        assert len(account.sets) == 2, "published once, by the second cycle"
        assert [(e["kind"], e["recovered_from"], e["failures"]) for e in _events(trace)] == [
            ("recovered", "failures", 1)
        ]
        assert node._venue_last_failure is None


# ============================================ venue alerts through the real journal + bot

_TELEGRAM_MAX = 4_096  # sendMessage text limit, counted after entity parsing


def _real_delivery(
    node: PaperTradingNode,
    monkeypatch: pytest.MonkeyPatch,
    respond: Callable[[httpx.Request], Awaitable[httpx.Response]],
) -> tuple[SessionJournal, list[dict[str, object]]]:
    """Replace the node's journal with a real one whose real bot posts to ``respond``;
    returns it and the event payloads it was given (no event log)."""
    journal = SessionJournal(node.config.trader_id, node._journal.event_log)
    bot = TelegramBot(TelegramConfig(bot_token="1:offline", chat_id="1"))
    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))

    async def get_client() -> httpx.AsyncClient:
        return client

    monkeypatch.setattr(bot, "_get_client", get_client)
    bot._client = client
    journal.attach(None, bot, "probe")
    writes: list[dict[str, object]] = []

    def write(_event_type: Any, data: dict[str, object], timestamp: Any = None) -> None:
        writes.append(data)

    monkeypatch.setattr(journal, "_write_event", write)
    node._journal = journal
    return journal, writes


def _sent_text(request: httpx.Request) -> str:
    return str(json.loads(request.content)["text"])


def _rendered_length(text: str) -> int:
    """The message length Telegram counts: HTML tags parsed away, entities decoded."""
    return len(html.unescape(re.sub(r"<[^>]*>", "", text)))


@pytest.mark.parametrize("first", ["latch", "halt"])
async def test_latch_alert_retries_independently_of_guard_state(
    first: str,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B1: the venue latch alert and a kill-switch HALT alert have independent delivery
    keys. Whichever goes first gets HTTP 503 once; the other is sent while it waits to
    retry; both reach the chat (neither supersedes the other)."""
    failed = asyncio.Event()
    attempts: list[str] = []
    delivered: list[str] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        text = _sent_text(request)
        key = (
            "latch"
            if "ENTRIES BLOCKED (venue config)" in text
            else "halt"
            if "HALT (kill_switch)" in text
            else "other"
        )
        attempts.append(key)
        if key == first and not failed.is_set():
            failed.set()
            return httpx.Response(503)
        delivered.append(key)
        return httpx.Response(200)

    async with _armed(monkeypatch, seeded) as (node, _trace):
        journal, _writes = _real_delivery(node, monkeypatch, respond)
        script = _Script(monkeypatch, node, _ETH5)
        if first == "latch":
            await _settle(node, script)
            await asyncio.wait_for(failed.wait(), 5)
            node.halt(HaltReason.KILL_SWITCH, "operator kill")
        else:
            node.halt(HaltReason.KILL_SWITCH, "operator kill")
            await asyncio.wait_for(failed.wait(), 5)
            await _settle(node, script)
        await asyncio.wait_for(journal.wait_alerts(None), 10)
        assert node._venue_latched
        second = "halt" if first == "latch" else "latch"
        assert [a for a in attempts if a != "other"] == [first, second, first]
        assert [d for d in delivered if d != "other"] == [second, first]


async def test_foreign_positions_alerts_fit_telegram(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B2: 100 foreign positions latch. The latch alert and the operator resume's
    re-pause alert each fit Telegram's 4,096 characters (a sample plus the count left
    out) and are delivered; status and the event keep every symbol."""
    lengths: list[int] = []
    delivered: list[str] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        text = _sent_text(request)
        lengths.append(_rendered_length(text))
        if lengths[-1] > _TELEGRAM_MAX:
            return httpx.Response(400, json={"description": "Bad Request: message is too long"})
        delivered.append(text)
        return httpx.Response(200)

    async with _armed(monkeypatch, seeded) as (node, _trace):
        journal, writes = _real_delivery(node, monkeypatch, respond)
        rows = [_prow(f"ALT{i:03}USDT", "1.000") for i in range(100)]
        script = _Script(monkeypatch, node, _MATCH, positions=[rows])
        await _settle(node, script)
        await asyncio.wait_for(journal.wait_alerts(None), 5)
        assert node._venue_latched
        resumed = node.resume()
        assert resumed[0] is False
        await asyncio.wait_for(journal.wait_alerts(None), 5)

        assert len(delivered) == 2, lengths
        latch, repause = delivered
        assert "ENTRIES BLOCKED (venue config): unknown position on venue: ALT000USDT" in latch
        assert "(+" in latch and "more; full text in status)" in latch
        assert "resume re-paused, still paused: unknown position on venue" in repause
        assert "more; full text in status)" in repause
        assert "ALT099USDT" not in latch + repause
        assert "ALT099USDT 1.000" in str(node._venue_block)
        assert "ALT099USDT 1.000" in _guard(node).message
        assert "ALT099USDT 1.000" in resumed[1]
        [event] = [w for w in writes if w.get("action") == "venue_config_check"]
        assert "ALT099USDT 1.000" in str(event["message"])


async def test_unverified_alert_fits_telegram(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B2: three cycles with 100 unreadable foreign positionRisk rows: the third-failure
    alert fits Telegram's limit and is delivered; the event keeps the full reason."""
    lengths: list[int] = []
    delivered: list[str] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        text = _sent_text(request)
        lengths.append(_rendered_length(text))
        if lengths[-1] > _TELEGRAM_MAX:
            return httpx.Response(400, json={"description": "Bad Request: message is too long"})
        delivered.append(text)
        return httpx.Response(200)

    async with _armed(monkeypatch, seeded) as (node, _trace):
        journal, writes = _real_delivery(node, monkeypatch, respond)
        rows = [_prow(f"ALT{i:03}USDT", "1.0.0") for i in range(100)]
        script = _Script(monkeypatch, node, _MATCH, _MATCH, _MATCH, positions=[rows] * 3)
        await _settle(node, script)
        await asyncio.wait_for(journal.wait_alerts(None), 5)

        assert node._venue_fails == 3
        [alert] = delivered
        assert "venue leverage/margin unverified after 3 reads (last positions unverified" in alert
        assert "more; full text in status)" in alert
        [event] = [w for w in writes if w.get("action") == "venue_config_check"]
        assert event["kind"] == "unverified"
        assert "ALT099USDT" in str(event["message"])


async def test_third_failure_publish_alert_reports_pause(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B3: a pass at t=0, failed reads at 60/65 s, a publish failure at 70 s (third
    failure): the publish-failed pause is installed before the alert is worded, so the
    alert says entries stay blocked and since when; the old verification deadline is kept."""
    account = _Account()
    _Wall(monkeypatch, _T1)
    async with _armed(monkeypatch, seeded, account=account) as (node, _trace):
        clock = _Clock(monkeypatch, node)
        spy = _Spy(monkeypatch, node)

        async def invalid_account() -> object:
            account.mode = "raises"
            return _MATCH

        script = _Script(
            monkeypatch, node, _MATCH, OSError(), OSError(), invalid_account, clock=clock
        )
        await _settle(node, script)
        assert clock.now == 70
        assert _guard(node).guard_state == GuardState.PAUSED
        assert node._venue_deadline == 90.0, "the pass at t=0 still sets the deadline"
        messages = [a.message for a in spy.alerts if a.monitor == "venuecfg:unverified"]
        assert messages == [
            "paper PAPER-007: venue leverage/margin unverified after 3 reads (last leverage "
            "not published: BTCUSDT: leverage publish failed (ValueError)); new entries stay "
            "blocked (paused) since 2026-11-20 14:07:05 UTC"
        ]


@pytest.mark.parametrize("mismatch", [False, True])
async def test_ready_result_is_discarded_after_engine_exits(
    mismatch: bool,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B4: ``run_async`` raises in the same loop turn the last venue answer becomes
    ready. The engine task marks stopping as it exits, so the answer is discarded: no
    publish, no latch, no venue event."""
    account = _Account()
    failed = asyncio.Event()
    async with _armed(monkeypatch, seeded, account=account) as (node, trace):

        async def run_async(_self: Any) -> None:
            await failed.wait()
            raise RuntimeError("engine failed")

        monkeypatch.setattr(fake_tradingnode, "run_async", run_async)
        real = node._venue_read_config

        async def last_result(*args: Any) -> Any:
            result = await real(*args)
            failed.set()  # the engine's failure and this answer are ready together
            return result

        monkeypatch.setattr(node, "_venue_read_config", last_result)
        _Script(monkeypatch, node, _ETH5 if mismatch else _MATCH)
        with pytest.raises(RuntimeError, match="engine failed"):
            await node._run_loop()
        assert node._stop_requested is True
        assert not node._venue_latched
        assert not account.sets, account.sets
        assert _events(trace) == []


@pytest.mark.parametrize("failed_endpoint", ["config", "positions"])
async def test_confirmed_result_beats_other_endpoint_build_failure(
    failed_endpoint: str,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B5: one endpoint's reader cannot be built (OSError); the other endpoint is still
    read in the same cycle and its confirmed result (foreign SOL position / ETH 5x)
    latches."""

    def fail_build() -> Any:
        raise OSError("offline builder failure")

    async with _armed(monkeypatch, seeded) as (node, trace):
        spy = _Spy(monkeypatch, node)
        script = _Script(monkeypatch, node, _ETH5, positions=[[_prow("SOLUSDT", "3.5")]])
        if failed_endpoint == "config":
            monkeypatch.setattr(node, "_build_venue_read", fail_build)
            message = "unknown position on venue: SOLUSDT 3.5 (not traded by this session)"
            kind = "unknown_position"
        else:
            monkeypatch.setattr(node, "_build_position_read", fail_build)
            message = "venue config mismatch: ETHUSDT: venue leverage 5x != resolved 10x"
            kind = "mismatch"

        async def park(_seconds: float) -> None:
            # A retry wait means neither endpoint latched in the first cycle.
            script.parked.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(node, "_venue_wait", park)
        await _settle(node, script)
        assert script.waits == []
        _assert_latched(node, trace, spy, message, kind)


# ============================================ ro-vc-1: F1 alert supersession, F2 drift page, F3 flag

_T0 = datetime(2026, 11, 20, 14, 1, 30, tzinfo=UTC)
_T1 = datetime(2026, 11, 20, 14, 7, 5, tzinfo=UTC)


class _Wall:
    """The node's wall clock (``_now``), moved by the test; set before ``_armed`` so the
    arm's PENDING block is stamped with it."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, at: datetime = _T0) -> None:
        self.at = at
        monkeypatch.setattr(PaperTradingNode, "_now", lambda _self: self.at)


def _throttle(retry_after: float) -> httpx.Response:
    """Telegram's 429 with ``parameters.retry_after``."""
    body = {
        "ok": False,
        "error_code": 429,
        "description": "Too Many Requests",
        "parameters": {"retry_after": retry_after},
    }
    return httpx.Response(429, json=body)


async def test_recovery_that_supersedes_an_owed_outage_alert_reports_the_pause(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ro-vc-1 F1, order 1: the third-failure page gets Telegram 429 and waits to retry;
    the recovery of the same monitor pair supersedes it (journal policy kept), so the
    recovery alone must say what the page said: entries were paused, since when, after
    how many failed reads and why."""
    monkeypatch.setattr(node_module, "VENUE_CHECK_TIMEOUT_S", 0.02)
    _Wall(monkeypatch)
    throttled = asyncio.Event()
    attempts: list[str] = []
    delivered: list[str] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        text = _sent_text(request)
        attempts.append(text)
        if "unverified" in text and not throttled.is_set():
            throttled.set()
            return _throttle(0.3)
        delivered.append(text)
        return httpx.Response(200)

    async with _armed(monkeypatch, seeded) as (node, _trace):
        journal, _writes = _real_delivery(node, monkeypatch, respond)
        script = _Script(monkeypatch, node, _HANG, _HANG, _HANG, _HOLD, _MATCH)
        node._start_venue_task()
        await asyncio.wait_for(throttled.wait(), 10)
        await _until(script.held.is_set)
        script.release.set()
        await _until(script.parked.is_set)
        await asyncio.wait_for(journal.wait_alerts(None), 10)
        assert _guard(node).guard_state == GuardState.RUNNING

    assert sum("unverified" in a for a in attempts) == 1, "the owed page was superseded"
    [recovery] = delivered
    assert (
        "paper PAPER-007: venue leverage/margin verified after 3 failed reads (last "
        "TimeoutError); entries were paused from 2026-11-20 14:01:30 UTC; new entries allowed"
    ) in recovery


async def test_outage_that_supersedes_an_owed_recovery_reports_the_pause(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ro-vc-1 F1, order 2: the recovery gets 429 and waits to retry; a drifting venue
    position then pages (third drifting cycle) and supersedes it. The chat ends on the
    current state, and that page says entries are paused and since when."""
    from tests.unit.test_paper_recovery_alert_delivery import _clock

    monkeypatch.setattr(node_module, "VENUE_CHECK_TIMEOUT_S", 0.02)
    wall = _Wall(monkeypatch)
    now = _clock(monkeypatch)
    throttled = asyncio.Event()
    attempts: list[str] = []
    delivered: list[str] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        text = _sent_text(request)
        attempts.append(text)
        if "margin verified after" in text and not throttled.is_set():
            throttled.set()
            return _throttle(0.3)
        delivered.append(text)
        return httpx.Response(200)

    async with _armed(monkeypatch, seeded) as (node, _trace):
        journal, _writes = _real_delivery(node, monkeypatch, respond)
        drift = [[_prow("BTCUSDT", amount)] for amount in ("0.010", "0.011", "0.010")]
        script = _Script(
            monkeypatch,
            node,
            *(_HANG, _HANG, _HANG, _MATCH, _HOLD, _MATCH, _MATCH, _MATCH),
            positions=[[], [], [], [], *drift],
            positions_after_hold=True,  # the first drift lands after the release, at _T1
        )
        node._start_venue_task()
        await asyncio.wait_for(throttled.wait(), 10)
        await _until(script.held.is_set)
        now[0] += 61.0  # the unverified page's monitor window reopened
        wall.at = _T1
        script.release.set()
        await _until(script.parked.is_set)
        await asyncio.wait_for(journal.wait_alerts(None), 10)
        assert _guard(node).guard_state == GuardState.PAUSED
        assert node._venue_latched is False

    assert sum("margin verified after" in a for a in attempts) == 1, "the owed recovery was superseded"
    first, current = delivered
    assert "unverified after 3 reads (last TimeoutError)" in first
    assert (
        "paper PAPER-007: venue leverage/margin unverified after 3 reads (last venue position "
        "BTCUSDT 0.010 != session 0); new entries stay blocked (paused) since "
        "2026-11-20 14:07:05 UTC"
    ) in current


@pytest.mark.parametrize(("cycles", "pages"), [(2, 0), (3, 1), (8, 1)])
async def test_changing_drift_pages_once_by_the_third_cycle(
    cycles: int,
    pages: int,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ro-vc-1 F2: after a pass the venue alternates 0.010 / 0.011 BTC against a flat
    session. No pair repeats, so nothing latches; each drifting cycle counts toward the
    three-failure page: none after 2 cycles, exactly one from the third on. Entries
    stay paused, reduce-only stays eligible."""
    wall = _Wall(monkeypatch)
    async with _armed(monkeypatch, seeded) as (node, trace):
        spy = _Spy(monkeypatch, node)
        amounts = [("0.010", "0.011")[i % 2] for i in range(cycles)]
        script = _Script(
            monkeypatch,
            node,
            *([_MATCH] * (cycles + 1)),
            positions=[[], *[[_prow("BTCUSDT", a)] for a in amounts]],
        )
        snap = _Snap(node, script)
        script.positions[1] = snap(script.positions[1])
        wall.at = _T1
        await _settle(node, script)

        assert snap.seen[0]["block"] is None, "the first cycle passed"
        assert node._venue_latched is False
        assert node._venue_fails == cycles
        assert _guard(node).guard_state == GuardState.PAUSED
        assert node._venue_block == _confirming(f"BTCUSDT {amounts[-1]} != session 0")
        assert _entry_denial(node) is not None
        assert _reduce_only_denial(node) is None
        page = _unverified(
            "venue position BTCUSDT 0.010 != session 0",
            "new entries stay blocked (paused) since 2026-11-20 14:07:05 UTC",
        )
        assert spy.texts() == [page] * pages
        kinds = [e["kind"] for e in _events(trace)]
        assert kinds == ["pass"] + ["unverified"] * pages


def _inferred_reduction_denial(node: PaperTradingNode) -> str | None:
    """The gate's answer for a BTC SELL 0.01 WITHOUT the reduce_only flag against a
    local net long 0.01 (it would only reduce the local net)."""
    _guard(node).__dict__["test_nets"] = {str(_BTC): Decimal("0.01")}
    order = SimpleNamespace(
        is_reduce_only=False,
        instrument_id=_BTC,
        quantity=_Qty(),
        side=OrderSide.SELL,
        client_order_id=None,
        price=Decimal("100"),
    )
    denial: str | None = _guard(node)._checked_denial_reason(order)
    return denial


@pytest.mark.parametrize(
    "state", ["pending", "latched", "drift", "expired", "stopping", "operator_pause", "running"]
)
async def test_inferred_reduction_needs_the_flag_while_the_venue_blocks(
    state: str,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ro-vc-1 F3: while a venue block (PENDING, latch, drift, expired verification) is
    active or the node is stopping, a SELL that only reduces NT's local net but lacks
    the reduce_only flag is denied (the venue check found that net unreliable or not
    verified); a flagged reduce-only order stays eligible in every state. An operator
    pause alone and a running guard keep today's admission of the inferred reduction."""
    async with _armed(monkeypatch, seeded) as (node, _trace):
        clock = _Clock(monkeypatch, node)
        if state != "pending":
            await _settle(node, _Script(monkeypatch, node, _MATCH))
            assert node._venue_block is None
        if state == "latched":
            node._latch(_M, kind="mismatch")
        elif state == "drift":
            _guard(node).__dict__["test_nets"] = {str(_BTC): Decimal("0.01")}
            resolved = node.resolved_config
            assert resolved is not None
            reading = parse_positions([_prow("BTCUSDT", "0")], resolved.symbols)
            node._venue_apply_positions(reading, None)
        elif state == "expired":
            clock.now = 90.0
        elif state == "stopping":
            node._mark_stopping()
        elif state == "operator_pause":
            _enqueue(seeded["db"], ("pause", None))
            await node.process_commands()

        denial = _inferred_reduction_denial(node)
        assert _reduce_only_denial(node) is None
        if state in ("operator_pause", "running"):
            assert denial is None
            return
        assert denial is not None
        assert denial.startswith("VQ_GUARD: ")
        assert denial.endswith("; only explicit reduce-only orders allowed")
        expected = {
            "pending": VENUE_CHECK_PENDING,
            "latched": _M,
            "drift": _confirming("BTCUSDT 0 != session 0.01"),
            "expired": _expired(90, "no completed recheck"),
            "stopping": node_module.VENUE_CHECK_STOPPING,
        }[state]
        assert expected in denial
