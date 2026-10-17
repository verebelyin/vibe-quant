"""The start path launches the kill-switch watcher (bead vibe-quant-dt065, T2a).

Review finding rev-sb-t2a-1 (should-fix): nothing pinned that the node actually
starts its ``_kill_watch_loop`` task. Every existing paper/kill/node test calls
``_kill_watch_loop`` or ``_check_kill_switch`` directly, so deleting the single
``asyncio.create_task`` that launches the watcher - ``PaperTradingNode._run_loop``,
the line right after ``self._control_task = asyncio.create_task(...)`` - left the
whole paper/kill/node suite green, while a node that never started its watcher
would run with no engaged-switch escalation and no stale-read halt.

This test drives the real start path (``_initialize`` then ``_run_loop``, the
same two steps ``start()`` runs) and asserts both that ``_kill_watch_task``
exists and is alive, and that an engaged snapshot reaches a KILL_SWITCH halt
through the watcher alone, with no direct ``_check_kill_switch`` call.
"""

# ruff: noqa: F811
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_audit_paper_node import _config, _FakeTradingNode
from tests.unit.test_paper_kill_switch_fail_closed import (
    _RecordGuard,
    _ScriptedReads,
    fake_tradingnode,  # noqa: F401  # fixture
    seeded,  # noqa: F401  # fixture
)
from vibe_quant.paper.guard import HaltReason
from vibe_quant.paper.kill_switch_monitor import KillSwitchMonitor
from vibe_quant.paper.node import PaperTradingNode

if TYPE_CHECKING:
    from collections.abc import Callable


async def _wait_until(predicate: Callable[[], bool], timeout_s: float) -> bool:
    """Poll ``predicate`` without blocking the loop; ``False`` on timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


@pytest.mark.timing
async def test_run_loop_launches_the_kill_switch_watcher(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_run_loop`` launches ``_kill_watch_task`` and it stays alive; an engaged
    snapshot then reaches a KILL_SWITCH halt through the watcher alone.

    Mutant A (delete the ``create_task`` line) leaves ``_kill_watch_task`` None
    and reddens this test."""
    release = asyncio.Event()

    async def gated_run_async(_self: _FakeTradingNode) -> None:
        await release.wait()

    # Hold ``_run_loop`` open like the real TradingNode does, so the watcher it
    # launches is observable (the fake's ``run_async`` otherwise returns at once
    # and ``_run_loop`` tears the watcher down before the test can look).
    monkeypatch.setattr(fake_tradingnode, "run_async", gated_run_async)

    node = PaperTradingNode(_config(seeded))
    # The reader's first read is clean and not engaged (startup proceeds); every
    # later read is engaged, so the switch flips right after startup.
    script = _ScriptedReads("FT")
    node._kill_monitor = KillSwitchMonitor(lambda: script, poll_s=0.05)
    guard = _RecordGuard()

    await node._initialize()
    # ``_initialize`` builds the real guard (whose cache is None under the fake
    # trader, so a real flatten would crash); record halts on a fake instead,
    # exactly as the other kill-switch tests do.
    node._guard = guard  # type: ignore[assignment]

    run = asyncio.create_task(node._run_loop())
    try:
        launched = await _wait_until(lambda: node._kill_watch_task is not None, 2.0)
        assert launched, "the start path must launch the kill-switch watcher task"
        assert node._kill_watch_task is not None
        assert not node._kill_watch_task.done()

        halted = await _wait_until(lambda: bool(guard.calls), 1.5)
        assert halted, "an engaged snapshot must reach a KILL_SWITCH halt via the watcher"
        assert guard.calls == [("halt", HaltReason.KILL_SWITCH)]
    finally:
        node._shutdown_event.set()
        release.set()
        await asyncio.wait_for(run, 5)
        await node._shutdown()
