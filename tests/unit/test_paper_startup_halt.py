"""A SIGUSR1 halt during paper start-up is replayed through the guard (vibe-quant-neuvu.1).

Before the trading guard exists ``PaperTradingNode.halt`` only recorded the halt in the
node status: the guard was built from ``_restored_halt`` alone, so it came up RUNNING and
traded while the status said "running" with a stale ``halt_reason``, and the pre-guard
alert was a silent no-op before Telegram was built. The halt is now recorded in the
node's pre-guard record (``_prestart``) and applied through ``TradingGuard.halt`` right
after the guard is built, so the severity rule, the flatten, the RISK_CHECK, the single
alert and the checkpoint all run the normal guard path. Contract rows R1-R19
(design-L3 section 6):
R1 handlers first, R2-R6 the five start-up awaits, R7/R8 NT replay, R9-R15 node
sequences, R17 status sync; R16/R18 live in ``test_paper_alert_write.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from datetime import date
from decimal import Decimal
from typing import Any

import pytest

import tests.unit.test_audit_paper_guard as harness
import vibe_quant.paper.node as node_module
from tests.unit.test_audit_paper_node import _config, _FakeTradingNode
from tests.unit.test_paper_halt_restore import (
    _ISO,
    _bars,
    _count_flatten,
    _latest_checkpoint,
    _MetricsListener,
    _raises_venue_build,
    _StartupStrategy,
    _Trace,
    _trace_node,
    _UnitGuard,
    _write_checkpoint,
    fake_tradingnode,  # noqa: F401
    seeded,  # noqa: F401
)
from vibe_quant.db.state_manager import StateManager
from vibe_quant.logging.events import EventType
from vibe_quant.paper.config import ConfigurationError
from vibe_quant.paper.guard import (
    RESTORED_HALT_PREFIX,
    GuardState,
    HaltReason,
    RestoredHalt,
    RiskState,
    halt_supersedes,
)
from vibe_quant.paper.node import NodeState, PaperTradingNode, restored_halt_from_status

_MSG = "halt via SIGUSR1"
_RESTORED_MSG = f"{RESTORED_HALT_PREFIX}{_MSG}"
_SIGNAL_ALERT = ("alert", "circuit_breaker", f"HALT (signal): {_MSG}", True)
_SIGNAL_RECORD = (
    "write",
    EventType.RISK_CHECK,
    {
        "action": "halt",
        "check_type": "signal",
        "reason": "signal",
        "message": _MSG,
        "passed": False,
    },
    None,
)
_HALTED_SIGNAL: dict[str, Any] = {
    "state": "halted",
    "reason": HaltReason.SIGNAL,
    "message": _MSG,
    "guard": GuardState.HALTED,
    "guard_reason": HaltReason.SIGNAL,
    "guard_msg": _MSG,
}
_NO_CLEAN_READ = (
    False,
    "system kill switch state not fresh (no clean read); cannot resume until it reads cleanly",
)


# ------------------------------------------------------------------ helpers


class _Guard(_UnitGuard):
    """The unit guard (injected cache/portfolio/clock) that records each halt() result."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.halt_results: list[bool] = []
        super().__init__(*args, **kwargs)

    def halt(self, reason: HaltReason, message: str, metrics: dict[str, object] | None = None) -> bool:
        applied = super().halt(reason, message, metrics)
        self.halt_results.append(applied)
        return applied


def _snap(node: PaperTradingNode) -> dict[str, Any]:
    """Node status plus guard view, compared with ``_HALTED_SIGNAL``."""
    guard = node._guard
    out: dict[str, Any] = {
        "state": node.status.state.value,
        "reason": node.status.halt_reason,
        "message": node.status.error_message,
    }
    if guard is not None:
        out.update(
            guard=guard.guard_state,
            guard_reason=guard.halt_reason,
            guard_msg=guard.status()["message"],
        )
    return out


def _halt_records(calls: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """Every halt RISK_CHECK write and every alert, in order."""
    return [c for c in calls if c[0] == "alert" or c[2].get("action") == "halt"]


def _prepare(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],  # noqa: F811
) -> tuple[PaperTradingNode, _Trace, dict[str, Any]]:
    """A traced node on the fake trader; ``seen`` records what the start-up looked like."""
    monkeypatch.setattr(node_module, "TradingGuard", _Guard)
    node, trace = _trace_node(monkeypatch, seeded)
    seen: dict[str, Any] = {}

    original_build = node._create_live_trading_node

    def build() -> Any:
        seen["pre_guard"] = list(trace.calls)
        return original_build()

    async def run_async(self: Any) -> None:
        seen["in_run"] = _snap(node)

    monkeypatch.setattr(node, "_create_live_trading_node", build)
    monkeypatch.setattr(_FakeTradingNode, "run_async", run_async)
    return node, trace, seen


def _remove_handlers() -> None:
    loop = asyncio.get_running_loop()
    loop.remove_signal_handler(signal.SIGUSR1)
    loop.remove_signal_handler(signal.SIGUSR2)


async def _drive(node: PaperTradingNode, seen: dict[str, Any]) -> None:
    """Handlers -> ``_initialize`` -> ``_run_loop`` -> shutdown, like ``start()``."""
    node._setup_signal_handlers()
    try:
        await node._initialize()
        seen["after_init"] = _snap(node)
        seen["init_checkpoint"] = _latest_checkpoint(node._config.db_path)
        await node._run_loop()
        seen["after_run"] = _snap(node)
    finally:
        _remove_handlers()
        await node._shutdown()


def _sigusr1() -> None:
    os.kill(os.getpid(), signal.SIGUSR1)


def _arm(
    monkeypatch: pytest.MonkeyPatch,
    node: PaperTradingNode,
    where: str,
    action: Any = _sigusr1,
) -> list[str]:
    """Run ``action`` (default: a real SIGUSR1) inside start-up await ``where`` once.

    W1 kill first read, W2 event-log barrier, W3 Telegram probe, W4 stale-command sweep
    (the signal was sent after the sync restore), W5 init checkpoint (guard exists).
    The loop sleeps 10 ms so the signal handler really runs before the await returns.
    """
    fired: list[str] = []

    async def fire() -> None:
        if fired:
            return
        fired.append(where)
        action()
        await asyncio.sleep(0.01)

    if where == "W1":
        orig_read = node._kill_monitor.wait_first_read

        async def read(timeout: float) -> Any:
            await fire()
            return await orig_read(timeout)

        monkeypatch.setattr(node._kill_monitor, "wait_first_read", read)
    elif where == "W2":
        orig_wait = node_module.wait_future

        async def wait(future: Any, timeout: float) -> Any:
            await fire()
            return await orig_wait(future, timeout)

        monkeypatch.setattr(node_module, "wait_future", wait)
    elif where == "W3":

        async def telegram() -> None:
            await fire()

        monkeypatch.setattr(node_module, "require_working_telegram", telegram)
    elif where == "W4":
        orig_cmd = node._cmd_worker.run

        async def sweep(*args: Any, **kwargs: Any) -> Any:
            await fire()
            return await orig_cmd(*args, **kwargs)

        monkeypatch.setattr(node._cmd_worker, "run", sweep)
    else:
        assert where == "W5"
        orig_db = node._db_worker.run

        async def save(*args: Any, **kwargs: Any) -> Any:
            await fire()
            return await orig_db(*args, **kwargs)

        monkeypatch.setattr(node._db_worker, "run", save)
    return fired


# ------------------------------------------------------------- R2-R6 (first test)


@pytest.mark.parametrize("where", ["W1", "W2", "W3", "W4", "W5"])
async def test_sigusr1_during_startup_await_halts_session(
    where: str,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R2-R6: a real SIGUSR1 at any start-up await ends HALTED/signal, the guard blocks
    entries, exactly one halt event and one CRITICAL alert, and nothing before the guard."""
    node, trace, seen = _prepare(monkeypatch, seeded)
    fired = _arm(monkeypatch, node, where)

    await _drive(node, seen)

    assert fired == [where], "the signal really landed in that await"
    assert seen["after_init"] == _HALTED_SIGNAL
    assert seen["in_run"] == _HALTED_SIGNAL, "L2: the trader starts under a halted guard"
    assert seen["after_run"] == _HALTED_SIGNAL, "L1 keeps the guard's halt (was 'running')"
    assert _halt_records(trace.calls) == [_SIGNAL_RECORD, _SIGNAL_ALERT]
    if where != "W5":
        assert _halt_records(seen["pre_guard"]) == [], "recorded at P5, no write or alert before"
        checkpoint = seen["init_checkpoint"]
        assert checkpoint is not None
        assert checkpoint.node_status["state"] == "halted"
        assert checkpoint.node_status["halt_reason"] == "signal"
        assert checkpoint.node_status["guard"]["state"] == "halted"
    final = _latest_checkpoint(node._config.db_path)
    assert final is not None
    assert final.node_status["halt_reason"] == "signal"
    changes = [c[2] for c in trace.calls if c[0] == "write" and c[2].get("action") == "state_change"]
    assert {"action": "state_change", "from_state": "initializing", "to_state": "halted"} in changes
    assert node._prestart.pending is None, "consumed at the replay"


# ------------------------------------------------------------------ R1


async def test_start_installs_handlers_before_first_await(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R1: SIGUSR1 can only be handled once ``start()`` installed the handlers, and that
    is before the first await (the kill monitor starts inside ``_initialize``)."""
    node = PaperTradingNode(_config(seeded))
    order: list[str] = []
    monkeypatch.setattr(node, "_setup_signal_handlers", lambda: order.append("handlers"))

    def refuse_to_start() -> None:
        order.append("kill_monitor.start")
        raise ConfigurationError("stop here")

    monkeypatch.setattr(node._kill_monitor, "start", refuse_to_start)
    with pytest.raises(ConfigurationError, match="stop here"):
        await node.start()
    assert order == ["handlers", "kill_monitor.start"]


# ------------------------------------------------------------------ R7 / R8 (NT)


def _startup_harness(
    monkeypatch: pytest.MonkeyPatch,
    restored: RestoredHalt | None = None,
    **build_kw: Any,
) -> harness.Harness:
    """Two-symbol NT harness; strategies enter in ``on_start`` and on every bar."""
    monkeypatch.setattr(harness, "RecordingListener", _MetricsListener)
    monkeypatch.setattr(harness, "NaiveStrategy", _StartupStrategy)
    return harness.build(
        symbols=("BTCUSDT", "ETHUSDT"),
        strategy_kwargs={"mode": "every_bar"},
        restored_halt=restored,
        **build_kw,
    )


def _restored(reason: str, message: str) -> RestoredHalt:
    restored = restored_halt_from_status(
        {"state": "stopped", "halt_reason": reason, "error_message": message}, _ISO
    )
    assert restored is not None
    return restored


@pytest.mark.parametrize(
    "reason", [HaltReason.SIGNAL, HaltReason.MANUAL, HaltReason.KILL_SWITCH]
)
def test_replayed_startup_halt_denies_on_start_entries(
    reason: HaltReason, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R33: the halt applied before the engine starts (the replay, or a drained queued
    halt/kill) gate-denies the entries submitted in ``on_start`` and on every bar for
    every stopping reason, and the strategies end up stopped."""
    h = _startup_harness(monkeypatch)
    assert h.guard.halt(reason, _MSG) is True
    snaps: dict[str, Any] = {}
    harness.run(h, _bars(h, [100.0] * 5), actions=[(h.at(3.5), h.snap(snaps, "t3.5"))])

    submitted = len(h.entry_orders())
    assert submitted >= 1, "the gate must really be exercised"
    assert len(h.filled_entries()) == 0
    assert [by_gate for _e, by_gate in h.listener.denied] == [True] * submitted
    assert [(c[1], c[2], c[3]) for c in h.listener.changes] == [
        (GuardState.HALTED, reason, _MSG)
    ]
    assert snaps["t3.5"]["running"] == [False, False]
    assert snaps["t3.5"]["state"] == GuardState.HALTED
    h.engine.dispose()


def test_replay_over_restored_error_flattens_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """R8a: a restored ERROR halt (prio 2) is superseded by the replayed SIGNAL (prio 3):
    the replay flattens once and ``on_start`` must NOT arm the deferred restored flatten
    again (it flattened twice), nor notify a second, 'restored' halt."""
    h = _startup_harness(monkeypatch, restored=_restored("error", "venue down"))
    calls = _count_flatten(h.guard)
    assert h.guard.halt(HaltReason.SIGNAL, _MSG) is True
    harness.run(h, _bars(h, [100.0] * 5))

    assert len(calls) == 1
    assert h.guard.halt_reason == HaltReason.SIGNAL
    assert h.guard.status()["message"] == _MSG
    assert [(c[1], c[2], c[3]) for c in h.listener.changes] == [
        (GuardState.HALTED, HaltReason.SIGNAL, _MSG)
    ], "one alert-bearing change, no restored notify"
    assert h.listener.state_metrics == [{}]
    assert h.engine.cache.positions_open() == []
    assert len(h.filled_entries()) == 0
    h.engine.dispose()


@pytest.mark.parametrize("reason", ["manual", "signal", "kill_switch"])
def test_restored_halt_outranks_startup_signal(
    reason: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R8b/R8c: a restored halt of equal or higher priority wins over the replayed SIGNAL:
    the replay returns False, the restored notify is the one alert, the deferred restored
    flatten still runs once on the first tick."""
    restored = _restored(reason, "halt via API")
    h = _startup_harness(monkeypatch, restored=restored)
    calls = _count_flatten(h.guard)
    assert h.guard.halt(HaltReason.SIGNAL, _MSG) is False
    harness.run(h, _bars(h, [100.0] * 5))

    assert h.guard.halt_reason == HaltReason(reason)
    assert h.guard.status()["message"] == "restored after restart: halt via API"
    assert [(c[1], c[2], c[3]) for c in h.listener.changes] == [
        (GuardState.HALTED, HaltReason(reason), "restored after restart: halt via API")
    ]
    assert h.listener.state_metrics == [{"restored": True, "checkpoint_time": _ISO}]
    assert len(calls) == 1, "the restored flatten is still owed and runs on the first tick"
    assert len(h.filled_entries()) == 0
    h.engine.dispose()


def test_startup_signal_then_drawdown_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    """R8d: no restored halt, but a restored high-water mark that breaches max drawdown:
    the replayed SIGNAL alerts at once, ``on_start`` escalates to MAX_DRAWDOWN (a second
    alert): each strictly more severe halt notifies, as live."""
    risk = RiskState(
        high_water_mark=Decimal("10600"),
        day=date(2026, 1, 5),
        day_start_equity=Decimal("10000"),
    )
    h = _startup_harness(
        monkeypatch,
        limits={"max_drawdown_pct": Decimal("0.05"), "max_daily_loss_pct": Decimal("0.5")},
        risk_state=risk,
    )
    assert h.guard.halt(HaltReason.SIGNAL, _MSG) is True
    harness.run(h, _bars(h, [10000.0] * 3))

    assert [(c[1], c[2]) for c in h.listener.changes] == [
        (GuardState.HALTED, HaltReason.SIGNAL),
        (GuardState.HALTED, HaltReason.MAX_DRAWDOWN),
    ]
    assert h.guard.halt_reason == HaltReason.MAX_DRAWDOWN
    h.engine.dispose()


# ------------------------------------------------- halt_supersedes (the shared rule)


@pytest.mark.parametrize(
    ("new", "current", "expected"),
    [
        (HaltReason.SIGNAL, None, True),
        (HaltReason.MAX_DAILY_LOSS, None, True),
        (HaltReason.SIGNAL, HaltReason.SIGNAL, False),
        (HaltReason.SIGNAL, HaltReason.MANUAL, False),
        (HaltReason.MANUAL, HaltReason.SIGNAL, False),
        (HaltReason.SIGNAL, HaltReason.ERROR, True),
        (HaltReason.ERROR, HaltReason.SIGNAL, False),
        (HaltReason.KILL_SWITCH, HaltReason.SIGNAL, True),
        (HaltReason.SIGNAL, HaltReason.KILL_SWITCH, False),
        (HaltReason.MAX_DRAWDOWN, HaltReason.MAX_CONSECUTIVE_LOSSES, True),
        (HaltReason.MAX_DAILY_LOSS, HaltReason.ERROR, False),
        (HaltReason.ERROR, HaltReason.MAX_DAILY_LOSS, True),
    ],
)
def test_halt_supersedes_is_strictly_more_severe(
    new: HaltReason, current: HaltReason | None, expected: bool
) -> None:
    """The guard's severity rule, exported for the pre-guard node: equal priority never
    supersedes (MANUAL and SIGNAL share 3), a lower one never, a higher one always."""
    assert halt_supersedes(new, current) is expected


# ------------------------------------------------------------------ R12


@pytest.mark.parametrize(
    ("sequence", "kept"),
    [
        ([(HaltReason.SIGNAL, "s"), (HaltReason.ERROR, "e")], (HaltReason.SIGNAL, "s")),
        ([(HaltReason.ERROR, "e"), (HaltReason.SIGNAL, "s")], (HaltReason.SIGNAL, "s")),
        ([(HaltReason.MANUAL, "m"), (HaltReason.SIGNAL, "s")], (HaltReason.MANUAL, "m")),
        ([(HaltReason.SIGNAL, "s"), (HaltReason.MANUAL, "m")], (HaltReason.SIGNAL, "s")),
        ([(HaltReason.MANUAL, "m"), (HaltReason.KILL_SWITCH, "k")], (HaltReason.KILL_SWITCH, "k")),
        ([(HaltReason.KILL_SWITCH, "k"), (HaltReason.MAX_DRAWDOWN, "d")], (HaltReason.KILL_SWITCH, "k")),
    ],
)
def test_pre_guard_halt_uses_guard_priority(
    sequence: list[tuple[HaltReason, str]],
    kept: tuple[HaltReason, str],
    seeded: dict[str, Any],  # noqa: F811
) -> None:
    """R12: before the guard exists the strongest recorded halt wins by the guard's rule
    (first wins on equal priority), in both orders; status and replay record agree."""
    node = PaperTradingNode(_config(seeded))
    for reason, message in sequence:
        node.halt(reason, message)
    assert node._prestart.pending == kept
    assert node.status.state == NodeState.HALTED
    assert (node.status.halt_reason, node.status.error_message) == kept


async def test_two_startup_signals_alert_once(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R10: a second pre-guard halt of equal priority does not replace the first; the
    session alerts once, with the first message."""
    node, trace, seen = _prepare(monkeypatch, seeded)

    def twice() -> None:
        node.halt(HaltReason.SIGNAL, "first")
        node.halt(HaltReason.SIGNAL, "second")
        assert node._prestart.pending == (HaltReason.SIGNAL, "first")

    fired = _arm(monkeypatch, node, "W1", action=twice)
    await _drive(node, seen)

    assert fired == ["W1"]
    assert [c for c in _halt_records(trace.calls) if c[0] == "alert"] == [
        ("alert", "circuit_breaker", "HALT (signal): first", True)
    ]
    assert seen["after_run"]["message"] == "first"


async def test_startup_and_post_guard_signal_alert_once(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R11: SIGUSR1 at W1 (replayed) and again at W5 (guard exists): the second
    ``guard.halt`` returns False and nothing is alerted twice."""
    node, trace, seen = _prepare(monkeypatch, seeded)
    first = _arm(monkeypatch, node, "W1")
    second = _arm(monkeypatch, node, "W5")

    await _drive(node, seen)

    assert (first, second) == (["W1"], ["W5"]), "both signals really landed"
    guard = node._guard
    assert isinstance(guard, _Guard)
    assert guard.halt_results == [True, False]
    assert _halt_records(trace.calls) == [_SIGNAL_RECORD, _SIGNAL_ALERT]


# ------------------------------------------------------------------ R8 (node level)


@pytest.mark.parametrize("where", ["W1", "W4"])
@pytest.mark.parametrize(
    ("prior", "prior_reason", "applied"),
    [
        ("error", HaltReason.ERROR, True),
        ("manual", HaltReason.MANUAL, False),
        ("signal", HaltReason.SIGNAL, False),
        ("kill_switch", HaltReason.KILL_SWITCH, False),
    ],
)
async def test_startup_signal_over_restored_halt(
    where: str,
    prior: str,
    prior_reason: HaltReason,
    applied: bool,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R8a-c at node level: the start-up SIGUSR1 (before or after the restore) beats a
    restored ERROR halt and is dropped against an equal-or-higher restored one; the
    session never ends with less than the restored halt."""
    _write_checkpoint(
        seeded["db"], {"state": "stopped", "halt_reason": prior, "error_message": "prior"}
    )
    node, trace, seen = _prepare(monkeypatch, seeded)
    _arm(monkeypatch, node, where)

    await _drive(node, seen)

    guard = node._guard
    assert isinstance(guard, _Guard)
    if applied:
        assert seen["after_run"] == _HALTED_SIGNAL
        assert guard.halt_results == [True]
        assert _halt_records(trace.calls) == [_SIGNAL_RECORD, _SIGNAL_ALERT]
    else:
        restored = "restored after restart: prior"
        assert seen["after_run"] == {
            "state": "halted",
            "reason": prior_reason,
            "message": restored,
            "guard": GuardState.HALTED,
            "guard_reason": prior_reason,
            "guard_msg": restored,
        }
        assert guard.halt_results == ([False] if where == "W1" else [])
        assert _halt_records(trace.calls) == [], "the restored notify belongs to on_start"
        assert node._prestart.pending is None


# ------------------------------------------------------------------ R9


async def test_startup_signal_over_unreadable_checkpoint(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R9: the restore raises (fail-closed ERROR halt, blank risk) after a W1 SIGUSR1:
    the replayed SIGNAL (prio 3) supersedes the ERROR halt, one SIGNAL alert, and the
    init checkpoint says 'signal' with the blank risk state."""

    def unreadable(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("row unreadable")

    monkeypatch.setattr(
        node_module.StatePersistence, "load_latest_account_checkpoint", unreadable
    )
    node, trace, seen = _prepare(monkeypatch, seeded)
    _arm(monkeypatch, node, "W1")

    await _drive(node, seen)

    assert seen["after_init"] == _HALTED_SIGNAL
    assert [c for c in _halt_records(trace.calls) if c[0] == "alert"] == [_SIGNAL_ALERT]
    checkpoint = seen["init_checkpoint"]
    assert checkpoint is not None
    assert checkpoint.node_status["halt_reason"] == "signal"
    assert checkpoint.node_status["risk"] == RiskState().to_dict()


# ------------------------------------------------------------------ R13


async def _resume_case(
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[str, Any],  # noqa: F811
    *,
    read_before_resume: bool,
) -> tuple[PaperTradingNode, _Trace, dict[str, Any], list[tuple[bool, str]]]:
    node, trace, seen = _prepare(monkeypatch, seeded)
    results: list[tuple[bool, str]] = []
    original_resume = node.resume

    def resume() -> tuple[bool, str]:
        results.append(original_resume())
        return results[-1]

    monkeypatch.setattr(node, "resume", resume)
    monkeypatch.setattr(node._kill_monitor, "start", lambda: None)

    async def read(_timeout: float) -> Any:
        _sigusr1()
        await asyncio.sleep(0.01)
        if read_before_resume:
            node._kill_monitor.poll_once(node._state_manager)
        os.kill(os.getpid(), signal.SIGUSR2)
        await asyncio.sleep(0.01)
        if not read_before_resume:
            node._kill_monitor.poll_once(node._state_manager)
        return node._kill_monitor.snapshot()

    monkeypatch.setattr(node._kill_monitor, "wait_first_read", read)
    await _drive(node, seen)
    return node, trace, seen, results


async def test_startup_resume_clears_recorded_halt(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R13: SIGUSR1 then SIGUSR2 pre-guard with a fresh clean kill read lifts the halt
    and forgets it (no replay, guard RUNNING, no halt alert)."""
    node, trace, seen, results = await _resume_case(
        monkeypatch, seeded, read_before_resume=True
    )
    assert results == [(True, "resumed from halt")]
    assert node._prestart.pending is None
    assert seen["in_run"] == {
        "state": "running",
        "reason": None,
        "message": None,
        "guard": GuardState.RUNNING,
        "guard_reason": None,
        "guard_msg": "",
    }
    assert _halt_records(trace.calls) == []


async def test_startup_resume_refused_without_clean_read_keeps_halt(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R13: no clean kill read yet -> resume refused, the halt stands and is replayed."""
    _node, trace, seen, results = await _resume_case(
        monkeypatch, seeded, read_before_resume=False
    )
    assert results == [_NO_CLEAN_READ]
    assert seen["in_run"] == _HALTED_SIGNAL
    assert _halt_records(trace.calls) == [_SIGNAL_RECORD, _SIGNAL_ALERT]


# ------------------------------------------------------------------ R14


async def test_refused_start_after_startup_signal_writes_no_checkpoint(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R14: SIGUSR1 at W1, then the engaged kill switch refuses the start: the original
    error propagates, the status keeps the signal halt, nothing is traded, no halt alert,
    and no checkpoint exists (persistence is not built yet: accepted residual)."""
    sm = StateManager(seeded["db"])
    sm.set_kill_switch("ops")
    sm.close()
    node, trace, _seen = _prepare(monkeypatch, seeded)
    _arm(monkeypatch, node, "W1")
    try:
        with pytest.raises(
            ConfigurationError, match=r"System kill switch engaged \(ops\); refusing to start"
        ):
            await node.start()
    finally:
        _remove_handlers()

    assert node.status.state == NodeState.ERROR
    assert node.status.halt_reason == HaltReason.SIGNAL
    assert node.status.error_message == _MSG
    assert _latest_checkpoint(seeded["db"]) is None
    assert [c for c in trace.calls if c[0] == "alert" and c[1] == "circuit_breaker"] == []
    assert node._guard is None
    assert fake_tradingnode.instances == []


# ------------------------------------------------------------------ R15


async def _failed_restart(
    seeded: dict[str, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    *,
    signal_at: str | None,
) -> PaperTradingNode:
    node = PaperTradingNode(_config(seeded))
    monkeypatch.setattr(node, "_create_live_trading_node", _raises_venue_build)
    if signal_at is None:
        monkeypatch.setattr(node, "_setup_signal_handlers", lambda: None)
    else:
        _arm(monkeypatch, node, signal_at)
    try:
        with pytest.raises(RuntimeError, match="venue build failed"):
            await node.start()
    finally:
        _remove_handlers()
    return node


async def test_startup_signal_survives_two_failed_restarts(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R15: a start-up SIGUSR1 whose session then fails at the node build is not lost:
    the failure checkpoint keeps 'signal' and its message, and every later restart comes
    up HALTED/SIGNAL with one 'restored after restart' prefix, failed or not."""
    monkeypatch.setattr(node_module, "TradingGuard", _Guard)
    node1 = await _failed_restart(seeded, monkeypatch, signal_at="W4")
    assert node1.status.state == NodeState.ERROR
    first = _latest_checkpoint(seeded["db"])
    assert first is not None
    assert first.node_status["state"] == "error"
    assert first.node_status["halt_reason"] == "signal"
    assert first.node_status["error_message"] == _MSG

    node2 = PaperTradingNode(_config(seeded))
    await node2._initialize()
    try:
        assert _snap(node2) == {
            **_HALTED_SIGNAL,
            "message": _RESTORED_MSG,
            "guard_msg": _RESTORED_MSG,
        }
        init = _latest_checkpoint(seeded["db"])
        assert init is not None
        assert init.node_status["halt_reason"] == "signal"
    finally:
        await node2._shutdown()

    await _failed_restart(seeded, monkeypatch, signal_at=None)
    failed = _latest_checkpoint(seeded["db"])
    assert failed is not None
    assert failed.node_status["halt_reason"] == "signal"
    assert failed.node_status["error_message"] == _RESTORED_MSG, "one prefix, not two"

    node4 = PaperTradingNode(_config(seeded))
    await node4._initialize()
    try:
        assert node4.status.state == NodeState.HALTED
        assert node4.status.halt_reason == HaltReason.SIGNAL
        assert node4.status.error_message == _RESTORED_MSG
    finally:
        await node4._shutdown()


# ------------------------------------------------------------------ R17


@pytest.mark.parametrize("case", ["running", "halted", "paused"])
async def test_run_loop_status_follows_guard(
    case: str,
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R17: the status ``_run_loop`` publishes is the guard's own, not a guess."""
    node, trace, seen = _prepare(monkeypatch, seeded)
    await node._initialize()
    try:
        guard = node._guard
        assert guard is not None
        if case == "halted":
            guard.halt(HaltReason.MANUAL, "boom")
        elif case == "paused":
            guard.pause()
        # Scrub what the listener wrote: _run_loop must read the guard, not the status.
        node._status.state = NodeState.INITIALIZING
        node._status.halt_reason = HaltReason.SIGNAL if case == "running" else None
        node._status.error_message = "keep me" if case == "running" else None
        await node._run_loop()
        expected = {
            "running": (NodeState.RUNNING, None, "keep me"),
            "halted": (NodeState.HALTED, HaltReason.MANUAL, "boom"),
            "paused": (NodeState.PAUSED, None, None),
        }[case]
        assert (
            node.status.state,
            node.status.halt_reason,
            node.status.error_message,
        ) == expected
        to_state = [
            c[2]["to_state"]
            for c in trace.calls
            if c[0] == "write" and c[2].get("from_state") == "initializing"
        ]
        assert to_state[-1] == expected[0].value, "the event _run_loop writes"
    finally:
        await node._shutdown()


async def test_run_loop_without_guard_keeps_restored_halt_branch(
    seeded: dict[str, Any],  # noqa: F811
    fake_tradingnode: type[_FakeTradingNode],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R17: with no guard (unit fakes) the old restored-halt branch applies verbatim."""
    node, _trace, _seen = _prepare(monkeypatch, seeded)
    await node._initialize()
    try:
        node._guard = None
        node._restored_halt = RestoredHalt(HaltReason.MANUAL, "restored after restart: x", _ISO)
        await node._run_loop()
        assert node.status.state == NodeState.HALTED
        assert node.status.halt_reason == HaltReason.MANUAL
        assert node.status.error_message == "restored after restart: x"
    finally:
        await node._shutdown()


# ------------------------------------------------------------------ guard.message


def test_guard_message_property_is_the_halt_message() -> None:
    """``TradingGuard.message`` is the status message (what ``_run_loop`` publishes)."""
    guard = _Guard(
        node_module.TradingGuardConfig(), strategies=[], risk_engine=_NoRiskEngine()
    )
    assert guard.message == ""
    assert guard.halt(HaltReason.MANUAL, "boom") is True
    assert guard.message == "boom"
    assert guard.status()["message"] == "boom"


class _NoRiskEngine:
    def execute(self, command: object) -> None:
        return None


def test_pre_guard_halt_logs_one_warning_without_traceback(
    seeded: dict[str, Any],  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The recorded halt is logged once (money path: no exc_info, no message text)."""
    node = PaperTradingNode(_config(seeded))
    with caplog.at_level(logging.WARNING):
        node.halt(HaltReason.SIGNAL, "ZZSECRET9")
    records = [r for r in caplog.records if "during start-up" in r.getMessage()]
    assert [r.getMessage() for r in records] == [
        "paper PAPER-007: signal halt during start-up recorded; "
        "the trading guard applies it when built"
    ]
    assert records[0].exc_info is None
    assert "ZZSECRET9" not in caplog.text
