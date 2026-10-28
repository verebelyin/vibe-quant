"""Halt restore after a paper restart (vibe-quant-oe9yc H1, contract H1-H24).

A node whose last checkpoint records a MANUAL / ERROR / SIGNAL / KILL_SWITCH
halt comes back HALTED (fail-closed) until the operator resumes (D-H1/H22):

* ``restored_halt_from_status`` derives the restored halt from the prior
  ``node_status``; risk halts are NOT restored from the field and re-trip from
  the restored risk bookkeeping, unknown values fail closed as ERROR, and a
  legacy checkpoint without ``halt_reason`` starts RUNNING.
* ``TradingGuard`` applies the restored state in ``__init__`` (H20);
  ``on_start`` arms the deferred flatten and notifies LAST (H21).
* the first HALTED tick flattens owned positions and stops strategies exactly
  as the original halt reason would (D-H2/H23); ``_lift_halt`` disarms the
  deferred flatten, so a resume before the first tick never flattens (H18).
* every checkpoint written before ``_run_loop`` keeps the halt (H19).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

import tests.unit.test_audit_paper_guard as harness
from tests.unit.test_audit_paper_node import DSL, _config, _FakeTradingNode
from vibe_quant.db.state_manager import StateManager
from vibe_quant.logging.events import EventType
from vibe_quant.paper import lineage
from vibe_quant.paper.guard import (
    RESTORED_HALT_PREFIX,
    GuardState,
    HaltReason,
    RestoredHalt,
    RiskState,
    TradingGuard,
    TradingGuardConfig,
)
from vibe_quant.paper.journal import FakeJournal
from vibe_quant.paper.node import NodeState, PaperTradingNode, restored_halt_from_status
from vibe_quant.paper.persistence import StateCheckpoint, StatePersistence

if TYPE_CHECKING:
    from pathlib import Path

from nautilus_trader.model.enums import OrderSide

#: The prior checkpoint of contract row H1 and its restored message.
_ISO = "2026-10-15T10:00:00+00:00"
#: R12: the exact restored-halt message for an unreadable risk state (R12m: the
#: risk-officer's flatten note is part of the message the alert text is built from).
_GARBLED_HWM_MESSAGE = (
    f"{RESTORED_HALT_PREFIX}unreadable risk state in checkpoint "
    "(high_water_mark='10600x'); limits re-baselined at current equity; "
    "owned and claimed positions flattened"
)
_PRIOR_MANUAL: dict[str, object] = {
    "state": "stopped",
    "halt_reason": "manual",
    "error_message": "halt via API",
}
_RESTORED_MESSAGE = "restored after restart: halt via API"


# ------------------------------------------------------------------ fixtures


@pytest.fixture()
def seeded(tmp_path: Path) -> dict[str, Any]:
    """A strategy plus a completed validation run (liquidation off, leverage 5)."""
    db = tmp_path / "state.db"
    sm = StateManager(db)
    sid = sm.create_strategy("test_strategy", DSL)
    rid = sm.create_backtest_run(
        strategy_id=sid,
        run_mode="validation",
        symbols=["BTCUSDT"],
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-12-31",
        parameters={"rsi_14_period": 21, "take_profit_risk_reward": 3.0, "leverage": 5},
    )
    sm.update_backtest_run_status(rid, "completed")
    sm.close()
    return {"db": db, "sid": sid, "rid": rid, "tmp": tmp_path}


@pytest.fixture()
def fake_tradingnode(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTradingNode]:
    """Redirect NT's TradingNode to the recording fake (copy of the audit fixture)."""
    import nautilus_trader.live.node as live_node

    _FakeTradingNode.instances = []
    monkeypatch.setattr(live_node, "TradingNode", _FakeTradingNode)
    return _FakeTradingNode


# ------------------------------------------------------------------ helpers


def _write_checkpoint(
    db: Path,
    node_status: dict[str, object],
    *,
    trader_id: str = "PAPER-007",
    risk: dict[str, object] | None = None,
) -> None:
    """Store one prior-session checkpoint (the restart input)."""
    status = dict(node_status)
    if risk is not None:
        status["risk"] = risk
    persistence = StatePersistence(db_path=db, trader_id=trader_id)
    try:
        persistence.save_checkpoint(StateCheckpoint(trader_id=trader_id, node_status=status))
    finally:
        persistence.close()


def _latest_checkpoint(db: Path, trader_id: str = "PAPER-007") -> StateCheckpoint | None:
    persistence = StatePersistence(db_path=db, trader_id=trader_id)
    try:
        return persistence.load_latest_checkpoint()
    finally:
        persistence.close()


class _EmptyCache:
    def positions_open(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    def orders_open(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    def orders_inflight(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    def order_exists(self, *args: Any, **kwargs: Any) -> bool:
        return False


class _RiskEngine:
    def execute(self, command: object) -> None:
        return None


class _FakeMoney:
    def __init__(self, value: str) -> None:
        self._value = Decimal(value)

    def as_decimal(self) -> Decimal:
        return self._value


class _EmptyPortfolio:
    def account(self, venue: Any) -> object:
        return object()

    def missing_price_instruments(self, venue: Any) -> list[Any]:
        return []

    def equity(self, venue: Any) -> dict[Any, Any]:
        from nautilus_trader.model.objects import Currency

        return {Currency.from_str("USDT"): _FakeMoney("10000")}


class _FakeClock:
    def utc_now(self) -> datetime:
        return datetime(2026, 1, 5, 22, 0, tzinfo=UTC)

    def timestamp_ns(self) -> int:
        return 0


class _UnitGuard(TradingGuard):
    """A real :class:`TradingGuard` with an injected cache (an unregistered
    actor gets ``cache = None`` and ``_flatten`` would crash)."""

    @property
    def cache(self) -> Any:  # type: ignore[override]
        return _EmptyCache()

    @property
    def portfolio(self) -> Any:  # type: ignore[override]
        return _EmptyPortfolio()

    @property
    def clock(self) -> Any:  # type: ignore[override]
        return _FakeClock()


def _unit_guard(restored: RestoredHalt | None) -> TradingGuard:
    return _UnitGuard(
        TradingGuardConfig(),
        strategies=[],
        risk_engine=_RiskEngine(),
        restored_halt=restored,
    )


class _StartupStrategy(harness.NaiveStrategy):
    """Submits one entry in ``on_start`` and one on every bar (the probe shape)."""

    def on_start(self) -> None:
        super().on_start()
        self.submit_order(
            self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=OrderSide.BUY,
                quantity=self.instrument.make_qty(Decimal("1")),
            )
        )


@dataclass
class _MetricsListener(harness.RecordingListener):
    """RecordingListener that also keeps the full ``on_state_change`` metrics."""

    state_metrics: list[dict[str, object]] = field(default_factory=list)

    def on_state_change(
        self,
        state: GuardState,
        reason: HaltReason | None,
        message: str,
        metrics: dict[str, object],
    ) -> None:
        self.state_metrics.append(dict(metrics))
        super().on_state_change(state, reason, message, metrics)


def _count_flatten(guard: TradingGuard) -> list[int]:
    """Wrap ``guard._flatten`` with a call counter (H8: exactly once)."""
    calls: list[int] = []
    original = guard._flatten

    def counted(**kwargs: Any) -> Any:
        calls.append(1)
        return original(**kwargs)

    guard._flatten = counted  # type: ignore[method-assign]
    return calls


def _restored_harness(monkeypatch: pytest.MonkeyPatch, **build_kw: Any) -> harness.Harness:
    """NT harness whose guard is built through the real restore parse."""
    restored = restored_halt_from_status(_PRIOR_MANUAL, _ISO)
    assert restored is not None
    monkeypatch.setattr(harness, "RecordingListener", _MetricsListener)
    monkeypatch.setattr(harness, "NaiveStrategy", _StartupStrategy)
    return harness.build(restored_halt=restored, **build_kw)


def _bars(h: harness.Harness, closes: list[float]) -> list[Any]:
    return [b for s in h.strategies for b in harness._flat_bars(s.config.bar_type, closes)]


@dataclass
class _Trace:
    """Every event write and alert in ONE ordered trace (gotcha 18dece9fd1f154a0-1926)."""

    calls: list[tuple[Any, ...]] = field(default_factory=list)

    def write(self, event_type: Any, data: dict[str, object], timestamp: Any = None) -> None:
        self.calls.append(("write", event_type, data, timestamp))

    def alert(self, kind: str, message: str, *, critical: bool = False) -> None:
        self.calls.append(("alert", kind, message, critical))

    def write_then_alert(
        self,
        event_type: Any,
        data: dict[str, object],
        kind: str,
        message: str,
        *,
        critical: bool = False,
        timestamp: Any = None,
    ) -> None:
        self.write(event_type, data, timestamp)
        self.alert(kind, message, critical=critical)


def _trace_node(
    monkeypatch: pytest.MonkeyPatch, seeded: dict[str, Any], **config_kw: Any
) -> tuple[PaperTradingNode, _Trace]:
    """A node whose event writes and alerts land in one ordered trace.

    The journal injected at ``node._journal`` records the same tuples; ``_Trace``
    only carries the fake's LIVE list, so ``trace.calls`` grows as the node writes.
    ``monkeypatch`` is kept for the 33 callers that pass it.
    """
    fake = FakeJournal()
    node = PaperTradingNode(_config(seeded, **config_kw))
    node._journal = fake
    return node, _Trace(calls=fake.calls)


def _raises_venue_build() -> Any:
    raise RuntimeError("venue build failed")


# --------------------------------------------------------------- H1 (NT harness)


def test_manual_halt_is_restored_and_blocks_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """H1: a prior MANUAL halt restores HALTED/MANUAL and every entry (submitted
    in ``on_start`` and on every bar by 2 strategies) is gate-denied: 0 filled
    (H25: the submitted count is timing-dependent, only >= 1 is pinned).
    The notify carries the full restored metrics (H12's event payload source)."""
    h = _restored_harness(
        monkeypatch, symbols=("BTCUSDT", "ETHUSDT"), strategy_kwargs={"mode": "every_bar"}
    )
    harness.run(h, _bars(h, [100.0] * 5))

    submitted = len(h.entry_orders())
    assert submitted >= 1, "the gate must really be exercised"
    assert len(h.filled_entries()) == 0
    assert [by_gate for _e, by_gate in h.listener.denied] == [True] * submitted
    assert [(c[1], c[2], c[3]) for c in h.listener.changes] == [
        (GuardState.HALTED, HaltReason.MANUAL, _RESTORED_MESSAGE)
    ]
    assert h.listener.state_metrics == [{"restored": True, "checkpoint_time": _ISO}]
    assert h.guard.guard_state == GuardState.HALTED
    assert h.guard.halt_reason == HaltReason.MANUAL
    assert h.guard.status()["message"] == _RESTORED_MESSAGE
    h.engine.dispose()


# --------------------------------------------------------------- H2/H3


def test_error_halt_is_restored() -> None:
    """H2: an ERROR halt (incl. K3's fail-closed halt) restores HALTED/ERROR."""
    message = (
        "system kill switch unreadable for 3 consecutive reads (OperationalError); failing closed"
    )
    restored = restored_halt_from_status(
        {"state": "error", "halt_reason": "error", "error_message": message}, _ISO
    )
    assert restored == RestoredHalt(
        reason=HaltReason.ERROR,
        message=f"{RESTORED_HALT_PREFIX}{message}",
        checkpoint_time=_ISO,
    )
    guard = _unit_guard(restored)
    assert guard.guard_state == GuardState.HALTED
    assert guard.halt_reason == HaltReason.ERROR
    assert guard.status()["message"] == f"{RESTORED_HALT_PREFIX}{message}"


@pytest.mark.parametrize(
    ("reason", "expected"), [("signal", HaltReason.SIGNAL), ("kill_switch", HaltReason.KILL_SWITCH)]
)
def test_operator_resumable_halts_are_restored(reason: str, expected: HaltReason) -> None:
    """H3/H22: SIGNAL and KILL_SWITCH (switch since unlocked) restore HALTED."""
    restored = restored_halt_from_status(
        {"state": "stopped", "halt_reason": reason, "error_message": "halt via SIGUSR1"}, _ISO
    )
    assert restored == RestoredHalt(
        reason=expected,
        message=f"{RESTORED_HALT_PREFIX}halt via SIGUSR1",
        checkpoint_time=_ISO,
    )
    guard = _unit_guard(restored)
    assert guard.guard_state == GuardState.HALTED
    assert guard.halt_reason == expected


# --------------------------------------------------------------- H4


def test_risk_halts_retrip_from_bookkeeping_not_field() -> None:
    """H4/H22: risk halts are NOT restored from the field - the guard starts
    RUNNING and re-trips from restored bookkeeping (HWM 10600 vs equity 10000
    = 5.66% >= 5% -> MAX_DRAWDOWN); a daily-loss halt from a previous UTC day
    re-baselines and stays RUNNING."""
    for reason in ("max_drawdown", "max_consecutive_losses", "max_daily_loss"):
        assert (
            restored_halt_from_status(
                {"state": "stopped", "halt_reason": reason, "error_message": "breach"}, _ISO
            )
            is None
        )

    risk = RiskState(
        high_water_mark=Decimal("10600"),
        day=date(2026, 1, 5),
        day_start_equity=Decimal("10000"),
    )
    h = harness.build(
        limits={"max_drawdown_pct": Decimal("0.05"), "max_daily_loss_pct": Decimal("0.5")},
        risk_state=risk,
    )
    harness.run(h, harness._flat_bars(h.strategies[0].config.bar_type, [10000.0] * 3))
    assert h.guard.halt_reason == HaltReason.MAX_DRAWDOWN
    assert h.guard.guard_state == GuardState.HALTED
    h.engine.dispose()

    stale = RiskState(
        high_water_mark=Decimal("10000"),
        day=date(2026, 1, 4),
        day_start_equity=Decimal("10600"),
    )
    h2 = harness.build(
        limits={"max_drawdown_pct": Decimal("0.5"), "max_daily_loss_pct": Decimal("0.02")},
        risk_state=stale,
    )
    harness.run(h2, harness._flat_bars(h2.strategies[0].config.bar_type, [10000.0] * 3))
    assert h2.guard.guard_state == GuardState.RUNNING, "a previous-UTC-day baseline must reset"
    assert h2.guard.risk_state.day == date(2026, 1, 5)
    h2.engine.dispose()


# --------------------------------------------------------------- H5/H15/H7


async def test_legacy_checkpoint_starts_running(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H5: a checkpoint without ``halt_reason`` (or None) - and even a
    non-dict/unreadable ``node_status`` - starts RUNNING with no halt event, and
    the restore event reports ``restored_halt: None``."""
    assert restored_halt_from_status({"state": "stopped"}, _ISO) is None
    assert restored_halt_from_status({"state": "paused"}, _ISO) is None
    assert restored_halt_from_status({"state": "stopped", "halt_reason": None}, _ISO) is None
    assert restored_halt_from_status(None, _ISO) is None
    assert restored_halt_from_status("not a dict", _ISO) is None

    _write_checkpoint(seeded["db"], {"state": "stopped"})
    node, trace = _trace_node(monkeypatch, seeded)
    await node._initialize()
    try:
        await node._run_loop()
        assert node.status.state == NodeState.RUNNING
        restores = [c for c in trace.calls if c[0] == "write" and c[2].get("action") == "restore"]
        assert restores and restores[0][2]["restored_halt"] is None
        assert not [c for c in trace.calls if c[2].get("action") == "halt"], "no halt event"
    finally:
        await node._shutdown()


async def test_no_checkpoint_runs(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """H7: no checkpoint at all -> RUNNING."""
    node = PaperTradingNode(_config(seeded))
    await node._initialize()
    try:
        await node._run_loop()
        assert node.status.state == NodeState.RUNNING
        assert node._restored_halt is None
    finally:
        await node._shutdown()


# --------------------------------------------------------------- H6


def test_unknown_halt_reason_fails_closed() -> None:
    """H6: an unrecognised ``halt_reason`` fails closed as HALTED/ERROR with the
    exact message, for both unknown strings and non-strings (error_message is
    not trusted for them)."""
    restored = restored_halt_from_status(
        {"state": "stopped", "halt_reason": "foo", "error_message": "x"}, _ISO
    )
    assert restored == RestoredHalt(
        reason=HaltReason.ERROR,
        message=f"{RESTORED_HALT_PREFIX}unrecognised prior halt_reason 'foo'",
        checkpoint_time=_ISO,
    )
    guard = _unit_guard(restored)
    assert guard.guard_state == GuardState.HALTED
    assert guard.halt_reason == HaltReason.ERROR

    raw_int = restored_halt_from_status({"halt_reason": 5}, _ISO)
    assert raw_int == RestoredHalt(
        reason=HaltReason.ERROR,
        message=f"{RESTORED_HALT_PREFIX}unrecognised prior halt_reason 5",
        checkpoint_time=_ISO,
    )


# --------------------------------------------------------------- H8 (NT harness)


def test_first_tick_flattens_and_stops_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """H8/H23/H25: the first HALTED tick flattens and stops the strategies
    exactly once (``stop_strategies = reason not in _KEEP_STRATEGIES_RUNNING``
    -> True for MANUAL); the end state equals the live halt's: flat, strategies
    stopped, still HALTED with the restored reason and message. The deferred
    flatten is consumed once, so later ticks never repeat it."""
    h = _restored_harness(
        monkeypatch, symbols=("BTCUSDT", "ETHUSDT"), strategy_kwargs={"mode": "every_bar"}
    )
    calls = _count_flatten(h.guard)
    snaps: dict[str, Any] = {}
    actions = [(h.at(m), h.snap(snaps, f"t{m}")) for m in (0.5, 1.5, 2.5, 3.5)]
    harness.run(h, _bars(h, [100.0] * 5), actions=actions)

    assert len(calls) == 1, "the restored flatten runs exactly once"
    assert snaps["t0.5"]["running"] == [True, True], "strategies run until the first tick"
    assert snaps["t3.5"]["running"] == [False, False], "strategies end up stopped"
    assert snaps["t3.5"]["state"] == GuardState.HALTED
    assert h.engine.cache.positions_open() == []
    assert len(h.filled_entries()) == 0
    assert h.guard.guard_state == GuardState.HALTED
    assert h.guard.halt_reason == HaltReason.MANUAL
    assert h.guard.status()["message"] == _RESTORED_MESSAGE
    h.engine.dispose()


class _BoomStrategy:
    """A guarded strategy whose market exit raises (H8's flatten-error branch)."""

    id = "strat-boom"
    is_running = True
    is_stopped = False

    def __init__(self) -> None:
        self.exit_calls = 0

    def is_exiting(self) -> bool:
        return False

    def market_exit(self) -> None:
        self.exit_calls += 1
        raise RuntimeError("boom ZZLEAK9")


def test_restored_flatten_error_reports_warning() -> None:
    """H8: a flatten error becomes the warning ``restored halt flatten: <error>``
    and never aborts the tick or the halt."""
    guard = _unit_guard(restored_halt_from_status(_PRIOR_MANUAL, _ISO))
    guard._restored_flatten_pending = True  # on_start arms it; the unit guard skips on_start
    warnings: list[str] = []
    guard._listener.on_warning = warnings.append  # type: ignore[assignment]
    boom = _BoomStrategy()
    guard._strategies = [boom]  # type: ignore[assignment]

    guard._on_tick(None)

    assert boom.exit_calls == 1
    assert warnings == ["restored halt flatten: strat-boom: RuntimeError: boom ZZLEAK9"]
    assert guard._restored_flatten_pending is False, "the deferred flatten is consumed once"
    assert guard.guard_state == GuardState.HALTED
    assert guard.halt_reason == HaltReason.MANUAL


# --------------------------------------------------------------- H9/H10


async def test_resume_lifts_restored_halt_and_clears_checkpoint(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """H9: operator resume returns ``(True, "resumed from halt")`` and the next
    checkpoint drops the halt, so the next restart comes up RUNNING."""
    _write_checkpoint(seeded["db"], _PRIOR_MANUAL)
    node = PaperTradingNode(_config(seeded))
    await node._initialize()
    try:
        assert node.status.state == NodeState.HALTED
        assert node.status.halt_reason == HaltReason.MANUAL
        assert node.resume() == (True, "resumed from halt")
        assert node.status.state == NodeState.RUNNING
        cp = node._capture_checkpoint()
        assert cp.node_status["halt_reason"] is None
        assert restored_halt_from_status(cp.node_status, _ISO) is None
    finally:
        await node._shutdown()


async def test_resume_refused_while_killed(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H10: resume while the kill switch is engaged is refused with the exact
    message and the restored halt stays in place."""
    _write_checkpoint(seeded["db"], _PRIOR_MANUAL)
    node = PaperTradingNode(_config(seeded))
    await node._initialize()
    try:
        node._kill_monitor.stop()  # the thread would overwrite the injected read

        class _Engaged:
            def get_system_state(self) -> dict[str, object]:
                return {"kill_switch": True, "reason": "ops"}

            def close(self) -> None:
                return None

        node._kill_monitor.poll_once(_Engaged())
        assert node.resume() == (
            False,
            "system kill switch is engaged; unlock it before resuming",
        )
        assert node.status.state == NodeState.HALTED
        assert node.status.halt_reason == HaltReason.MANUAL
    finally:
        await node._shutdown()


# --------------------------------------------------------------- H11


def test_restored_halt_escalates_by_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    """H11: a restored MANUAL halt (priority 3) escalates to MAX_DRAWDOWN
    (priority 5) when the restored HWM shows a breached drawdown at on_start;
    the escalated halt's own flatten is the only flatten (halt() disarms the
    deferred one) and resume reports the escalation's refusal."""
    risk = RiskState(
        high_water_mark=Decimal("10600"),
        day=date(2026, 1, 5),
        day_start_equity=Decimal("10000"),
    )
    h = _restored_harness(
        monkeypatch,
        limits={"max_drawdown_pct": Decimal("0.05"), "max_daily_loss_pct": Decimal("0.5")},
        risk_state=risk,
    )
    calls = _count_flatten(h.guard)
    harness.run(h, _bars(h, [10000.0] * 3))

    assert h.guard.halt_reason == HaltReason.MAX_DRAWDOWN
    assert h.guard.guard_state == GuardState.HALTED
    assert len(calls) == 1
    assert h.guard.resume() == (
        False,
        "max_drawdown halt requires manual review; restart the session to trade again",
    )
    h.engine.dispose()


# --------------------------------------------------------------- H12


async def test_restored_halt_event_and_alert_payload(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H12: the restore SIGNAL event gains ``"restored_halt": "manual"`` and the
    restored halt notify writes the full RISK_CHECK payload (restored +
    checkpoint_time merged in) followed by the critical circuit_breaker alert."""
    _write_checkpoint(seeded["db"], _PRIOR_MANUAL)
    node, trace = _trace_node(monkeypatch, seeded)
    await node._initialize()
    try:
        restores = [c for c in trace.calls if c[0] == "write" and c[2].get("action") == "restore"]
        assert restores and restores[0][2]["restored_halt"] == "manual"
        trace.calls.clear()

        node._on_guard_state_change(
            GuardState.HALTED,
            HaltReason.MANUAL,
            _RESTORED_MESSAGE,
            {"restored": True, "checkpoint_time": _ISO},
        )
        assert trace.calls == [
            (
                "write",
                EventType.RISK_CHECK,
                {
                    "action": "halt",
                    "check_type": "manual",
                    "reason": "manual",
                    "message": _RESTORED_MESSAGE,
                    "passed": False,
                    "restored": True,
                    "checkpoint_time": _ISO,
                },
                None,
            ),
            ("alert", "circuit_breaker", f"HALT (manual): {_RESTORED_MESSAGE}", True),
        ]
    finally:
        await node._shutdown()


# --------------------------------------------------------------- H13


def test_prefix_is_idempotent() -> None:
    """H13: an already-prefixed prior message is not prefixed twice (even across
    a second restart); a blank or non-str ``error_message`` falls back to the
    reason value."""
    prior = {"state": "stopped", "halt_reason": "manual", "error_message": _RESTORED_MESSAGE}
    restored = restored_halt_from_status(prior, _ISO)
    assert restored is not None
    assert restored.message == _RESTORED_MESSAGE

    again = restored_halt_from_status(
        {"state": "halted", "halt_reason": "manual", "error_message": restored.message}, _ISO
    )
    assert again is not None
    assert again.message == _RESTORED_MESSAGE

    for blank in ("", "   ", 5, None):
        fallback = restored_halt_from_status(
            {"halt_reason": "manual", "error_message": blank}, _ISO
        )
        assert fallback == RestoredHalt(
            reason=HaltReason.MANUAL,
            message="restored after restart: manual",
            checkpoint_time=_ISO,
        )


# --------------------------------------------------------------- H14


async def test_status_halted_before_trader_start(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H14: when the venue never connects (the trader never starts), the status
    is HALTED/MANUAL straight from ``_run_loop`` and the INITIALIZING
    state_change event says ``to_state: "halted"``."""
    _write_checkpoint(seeded["db"], _PRIOR_MANUAL)
    node, trace = _trace_node(monkeypatch, seeded)
    await node._initialize()
    try:
        await node._run_loop()
        assert node.status.state == NodeState.HALTED
        assert node.status.halt_reason == HaltReason.MANUAL
        assert node.status.error_message == _RESTORED_MESSAGE
        changes = [
            c[2] for c in trace.calls if c[0] == "write" and c[2].get("action") == "state_change"
        ]
        assert {"action": "state_change", "from_state": "initializing", "to_state": "halted"} in (
            changes
        )
    finally:
        await node._shutdown()


# --------------------------------------------------------------- H18 (NT harness)


def test_resume_before_first_tick_does_not_flatten(monkeypatch: pytest.MonkeyPatch) -> None:
    """H18: a resume inside the tick window (before the first guard tick)
    disarms the deferred flatten in ``_lift_halt``; after 2+ ticks nothing was
    flattened, no strategy was stopped and the guard is RUNNING."""
    h = _restored_harness(
        monkeypatch, symbols=("BTCUSDT", "ETHUSDT"), strategy_kwargs={"mode": "every_bar"}
    )
    calls = _count_flatten(h.guard)
    flag_after_resume: list[bool] = []

    def resume_now() -> None:
        assert h.guard.resume() == (True, "resumed from halt")
        flag_after_resume.append(h.guard._restored_flatten_pending)

    snaps: dict[str, Any] = {}
    actions = [
        (h.at(0.5), resume_now),
        (h.at(1.5), h.snap(snaps, "t1.5")),
        (h.at(2.5), h.snap(snaps, "t2.5")),
        (h.at(3.5), h.snap(snaps, "t3.5")),
    ]
    harness.run(h, _bars(h, [100.0] * 5), actions=actions)

    assert flag_after_resume == [False], "_lift_halt must clear the pending flatten"
    assert calls == [], "a resumed book must never be flattened"
    for key in ("t1.5", "t2.5", "t3.5"):
        assert snaps[key]["running"] == [True, True], key
    assert h.guard.guard_state == GuardState.RUNNING
    h.engine.dispose()


# --------------------------------------------------------------- H19


async def test_failed_restart_keeps_restored_risk(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H19: ``_capture_checkpoint`` without a guard persists the restored risk
    bookkeeping (HWM / daily baseline / consecutive losses are not wiped)."""
    risk = {
        "high_water_mark": "10600",
        "day": "2026-10-15",
        "day_start_equity": "10000",
        "consecutive_losses": 3,
        "last_equity": "10000",
    }
    _write_checkpoint(seeded["db"], _PRIOR_MANUAL, risk=risk)
    node = PaperTradingNode(_config(seeded))
    monkeypatch.setattr(node, "_create_live_trading_node", _raises_venue_build)
    with pytest.raises(RuntimeError):
        await node._initialize()
    try:
        cp = node._capture_checkpoint()
        assert cp.node_status["risk"] == {
            "high_water_mark": "10600",
            "day": "2026-10-15",
            "day_start_equity": "10000",
            "consecutive_losses": 3,
            "last_equity": "10000",
            "drawdown_pct": "0.05660377358490566037735849057",
            "daily_loss_pct": "0",
        }
    finally:
        await node._shutdown()


# --------------------------------------------------------------- H19b


async def test_failed_start_keeps_restored_halt_message(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """H19b: a failing ``start()`` during a restored halt must NOT overwrite the
    halt message with the exception text. The failure shutdown's checkpoint keeps
    the restored message, a second restart comes up with the same message, and
    the failure is logged by class only."""
    _write_checkpoint(seeded["db"], _PRIOR_MANUAL)
    node = PaperTradingNode(_config(seeded))
    monkeypatch.setattr(node, "_setup_signal_handlers", lambda: None)
    monkeypatch.setattr(node, "_create_live_trading_node", _raises_venue_build)

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="venue build failed"):
        await node.start()

    assert node.status.state == NodeState.ERROR
    assert node.status.error_message == _RESTORED_MESSAGE, "the exception text overwrote the halt"
    latest = _latest_checkpoint(seeded["db"])
    assert latest is not None
    assert latest.node_status["halt_reason"] == "manual"
    assert latest.node_status["error_message"] == _RESTORED_MESSAGE
    failures = [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR and "start failed while halted" in r.getMessage()
    ]
    assert [r.getMessage() for r in failures] == [
        "paper node start failed while halted (RuntimeError)"
    ]
    assert failures[0].exc_info is None
    assert "venue build failed" not in caplog.text

    node2 = PaperTradingNode(_config(seeded))
    await node2._initialize()
    try:
        assert node2.status.state == NodeState.HALTED
        assert node2.status.halt_reason == HaltReason.MANUAL
        assert node2.status.error_message == _RESTORED_MESSAGE
        # H19: the halt survives every checkpoint written before ``_run_loop`` too.
        after_init = _latest_checkpoint(seeded["db"])
        assert after_init is not None
        assert after_init.node_status["halt_reason"] == "manual"
        assert after_init.node_status["error_message"] == _RESTORED_MESSAGE
    finally:
        await node2._shutdown()


async def test_failed_start_without_halt_records_the_error(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H19b (pin): a non-halted failed start keeps today's behaviour - the
    exception text IS recorded as ``error_message`` (only a halted failure
    protects its message)."""
    node = PaperTradingNode(_config(seeded))
    monkeypatch.setattr(node, "_setup_signal_handlers", lambda: None)
    monkeypatch.setattr(node, "_create_live_trading_node", _raises_venue_build)

    with pytest.raises(RuntimeError, match="venue build failed"):
        await node.start()

    assert node.status.state == NodeState.ERROR
    assert node.status.error_message == "venue build failed"


# --------------------------------------------------------------- H20 (NT harness)


def test_resume_before_trader_start_lifts_restored_halt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H20: the restored state is applied in ``__init__`` (state, reason,
    message, ``_halt.before_halt``); ``on_start`` arms the flatten flag and
    notifies only while still HALTED, so a resume before start gives a real
    RUNNING session: no flatten, no halt notify, strategies trade normally."""
    h = _restored_harness(monkeypatch, strategy_kwargs={"mode": "every_bar"})
    guard = h.guard
    assert guard.guard_state == GuardState.HALTED
    assert guard.halt_reason == HaltReason.MANUAL
    assert guard.status()["message"] == _RESTORED_MESSAGE
    assert guard._halt.before_halt == GuardState.RUNNING
    calls = _count_flatten(guard)

    assert guard.resume() == (True, "resumed from halt")
    assert guard._restored_flatten_pending is False
    changes_after_resume = list(h.listener.changes)

    snaps: dict[str, Any] = {}
    harness.run(h, _bars(h, [10000.0] * 4), actions=[(h.at(2.5), h.snap(snaps, "t2.5"))])

    assert calls == []
    assert h.listener.changes == changes_after_resume, "on_start must not notify once resumed"
    assert guard.guard_state == GuardState.RUNNING
    assert snaps["t2.5"]["running"] == [True], "strategies start normally after a pre-start resume"
    assert h.filled_entries(), "strategies start normally after a pre-start resume"
    h.engine.dispose()


# --------------------------------------------------------------- H21 (NT harness)


class _RaisingListener(_MetricsListener):
    """Listener whose restored-halt notify raises (H21)."""

    leak = "ZZLEAK9"

    def on_state_change(
        self,
        state: GuardState,
        reason: HaltReason | None,
        message: str,
        metrics: dict[str, object],
    ) -> None:
        super().on_state_change(state, reason, message, metrics)
        if metrics.get("restored"):
            raise RuntimeError(f"listener boom {self.leak}")


def test_restored_notify_failure_does_not_abort_start(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """H21: a raising restored-halt notify is caught LAST inside on_start with a
    class-only log; the strategies still start and entries stay gate-denied."""
    monkeypatch.setattr(harness, "RecordingListener", _RaisingListener)
    restored = restored_halt_from_status(_PRIOR_MANUAL, _ISO)
    assert restored is not None
    monkeypatch.setattr(harness, "NaiveStrategy", _StartupStrategy)
    h = harness.build(
        restored_halt=restored,
        symbols=("BTCUSDT", "ETHUSDT"),
        strategy_kwargs={"mode": "every_bar"},
    )
    snaps: dict[str, Any] = {}
    with caplog.at_level(logging.ERROR):
        harness.run(h, _bars(h, [100.0] * 5), actions=[(h.at(0.5), h.snap(snaps, "t0.5"))])

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert [r.getMessage() for r in errors] == ["restored halt notify failed (RuntimeError)"]
    assert all(r.exc_info is None for r in errors)
    assert "listener boom" not in caplog.text
    assert "ZZLEAK9" not in caplog.text
    assert h.listener.state_metrics == [{"restored": True, "checkpoint_time": _ISO}]
    assert snaps["t0.5"]["running"] == [True, True], "strategies must still start"
    assert h.filled_entries() == []
    assert h.entry_orders(), "entries were attempted ..."
    assert all(by_gate for _e, by_gate in h.listener.denied), "... and gate-denied"
    h.engine.dispose()


# --------------------------------------------------------------- H24


async def test_explicit_trader_id_inherits_halt(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """H24: a start with an explicit trader_id whose latest checkpoint is halted
    inherits the halt (fail-closed); a different trader_id is unaffected."""
    _write_checkpoint(seeded["db"], _PRIOR_MANUAL, trader_id="PAPER-EXPLICIT")
    node = PaperTradingNode(_config(seeded, trader_id="PAPER-EXPLICIT"))
    await node._initialize()
    try:
        assert node.status.state == NodeState.HALTED
        assert node.status.halt_reason == HaltReason.MANUAL
    finally:
        await node._shutdown()

    other = PaperTradingNode(_config(seeded, trader_id="PAPER-OTHER"))
    await other._initialize()
    try:
        assert other._restored_halt is None
    finally:
        await other._shutdown()


# --------------------------------------------------------------- R12


async def test_garbled_high_water_mark_restores_error_halt(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R12: a checkpoint whose risk high_water_mark is unreadable ("10600x")
    restarts HALTED/ERROR with the exact message (R12m: the flatten note is in the
    alert text), the restore event carries ``risk_problems``, exactly one
    logger.error is emitted (class/values only, no exc_info), and the halt notify
    produces the one CRITICAL circuit_breaker alert."""
    risk = {
        "high_water_mark": "10600x",
        "day": "2026-10-15",
        "day_start_equity": "10000",
        "consecutive_losses": 3,
        "last_equity": "10000",
    }
    _write_checkpoint(seeded["db"], {"state": "running"}, risk=risk)
    node, trace = _trace_node(monkeypatch, seeded)
    try:
        with caplog.at_level(logging.ERROR):
            await node._initialize()

        assert node.status.state == NodeState.HALTED
        assert node.status.halt_reason == HaltReason.ERROR
        assert node.status.error_message == _GARBLED_HWM_MESSAGE

        restores = [c for c in trace.calls if c[0] == "write" and c[2].get("action") == "restore"]
        assert restores and restores[0][2]["risk_problems"] == ["high_water_mark='10600x'"]

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert [r.getMessage() for r in errors] == [
            "paper PAPER-007: unreadable risk state in checkpoint "
            "(high_water_mark='10600x'); halting"
        ]
        assert errors[0].exc_info is None

        trace.calls.clear()
        node._on_guard_state_change(
            GuardState.HALTED,
            HaltReason.ERROR,
            _GARBLED_HWM_MESSAGE,
            {"restored": True, "checkpoint_time": _ISO},
        )
        alerts = [c for c in trace.calls if c[0] == "alert"]
        assert alerts == [
            ("alert", "circuit_breaker", f"HALT (error): {_GARBLED_HWM_MESSAGE}", True)
        ]
    finally:
        await node._shutdown()


# ------------------------------------------------------- RiskState.from_checkpoint


def _full_risk(**overrides: object) -> dict[str, object]:
    """A full, clean ``to_dict`` payload with one field overridable."""
    base: dict[str, object] = {
        "high_water_mark": "10600",
        "day": "2026-10-15",
        "day_start_equity": "10000",
        "consecutive_losses": 3,
        "last_equity": "10000",
        "drawdown_pct": "0.05660377358490566037735849057",
        "daily_loss_pct": "0",
    }
    base.update(overrides)
    return base


_NULL_RISK: dict[str, object] = {
    "high_water_mark": None,
    "day": None,
    "day_start_equity": None,
    "consecutive_losses": 0,
    "last_equity": None,
}


def test_r1_no_risk_is_fresh_and_clean() -> None:
    """R1: no risk / null / {} / an all-null dict -> fresh state, no problems."""
    for data in (None, {}, _NULL_RISK):
        state, problems = RiskState.from_checkpoint(data)
        assert problems == (), data
        assert state == RiskState(), data


def test_r2_full_to_dict_restores_exactly() -> None:
    """R2: a fully-populated ``to_dict`` round-trips without problems."""
    original = RiskState(
        high_water_mark=Decimal("10600"),
        day=date(2026, 10, 15),
        day_start_equity=Decimal("10000"),
        consecutive_losses=3,
        last_equity=Decimal("10000"),
    )
    state, problems = RiskState.from_checkpoint(original.to_dict())
    assert problems == ()
    assert state == original
    assert state.to_dict() == original.to_dict()


def test_r3_null_field_is_none_no_problem() -> None:
    """R3: a null field restores as None with no problem."""
    state, problems = RiskState.from_checkpoint({"high_water_mark": None})
    assert state.high_water_mark is None
    assert problems == ()


def test_r4_garbled_high_water_mark_is_a_problem() -> None:
    """R4: "10600x" -> None + the exact problem string."""
    state, problems = RiskState.from_checkpoint(_full_risk(high_water_mark="10600x"))
    assert state.high_water_mark is None
    assert problems == ("high_water_mark='10600x'",)


@pytest.mark.parametrize("raw", ["NaN", "Infinity", "-Infinity", "sNaN"])
def test_r5_non_finite_decimal_is_a_problem(raw: str) -> None:
    """R5: a non-finite Decimal value (today: start raises InvalidOperation) is a
    problem and restores None; "sNaN" is the row Decimal() would otherwise raise on."""
    state, problems = RiskState.from_checkpoint(_full_risk(last_equity=raw))
    assert state.last_equity is None
    assert problems == (f"last_equity={raw!r}",)


@pytest.mark.parametrize("raw", [[1], True])
def test_r6_wrong_type_decimal_is_a_problem(raw: object) -> None:
    """R6: a non-scalar or a bool in a decimal field is a problem and restores None."""
    state, problems = RiskState.from_checkpoint(_full_risk(high_water_mark=raw))
    assert state.high_water_mark is None
    assert problems == (f"high_water_mark={raw!r}",)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (10600.5, Decimal("10600.5")),
        ("1E+4", Decimal("1E+4")),
        (" 10600 ", Decimal("10600")),
    ],
)
def test_r7_usable_decimal_values_restore(raw: object, expected: Decimal) -> None:
    """R7: float/scientific/padded strings parse to Decimals with no problem."""
    state, problems = RiskState.from_checkpoint(_full_risk(high_water_mark=raw))
    assert state.high_water_mark == expected
    assert problems == ()


def test_r7b_sub_unit_values_are_accepted_then_rejected_as_baselines() -> None:
    """R7b (as overridden by R20): "-5"/"0" still parse to Decimals, but a non-positive
    high_water_mark / day_start_equity is a problem (last_equity stays unrestricted)."""
    state, problems = RiskState.from_checkpoint(_full_risk(last_equity="-5"))
    assert state.last_equity == Decimal("-5")
    assert problems == ()

    rejected, bad = RiskState.from_checkpoint(
        _full_risk(high_water_mark="10600", day_start_equity="0")
    )
    assert rejected.day_start_equity is None
    assert bad == ("day_start_equity='0'",)


@pytest.mark.parametrize("raw", ["2026-13-45", 20261017])
def test_r8_bad_day_is_a_problem(raw: object) -> None:
    """R8: an impossible date and a non-string day are both problems."""
    state, problems = RiskState.from_checkpoint(_full_risk(day=raw))
    assert state.day is None
    assert problems == (f"day={raw!r}",)


@pytest.mark.parametrize("raw", [3, "3"])
def test_r9_consecutive_losses_accepts_int_and_digits(raw: object) -> None:
    """R9: an int or a digit string restores as 3."""
    state, problems = RiskState.from_checkpoint(_full_risk(consecutive_losses=raw))
    assert state.consecutive_losses == 3
    assert problems == ()


@pytest.mark.parametrize("raw", [3.0, True, -1, "3.0", " 3", "1_0", "\u00b3"])
def test_r9b_consecutive_losses_rejects_non_int(raw: object) -> None:
    """R9b: bool/negative/padded/dot/underscore/non-digit -> 0 + a problem.
    F1 (round 2): ``'³'.isdigit()`` is True but ``'³'.isdecimal()`` is False and
    ``int('³')`` raises, so the parser must reject it (never raise). ``'1_0'`` and
    ``' 3'`` are the guard's teeth: ``int()`` would happily read them as 10 and 3."""
    state, problems = RiskState.from_checkpoint(_full_risk(consecutive_losses=raw))
    assert state.consecutive_losses == 0
    assert problems == (f"consecutive_losses={raw!r}",)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("\u0661\u0662", 12),
        ("1" * 5000, 0),
    ],
)
def test_r9b_decimal_strings_never_raise(raw: str, expected: int) -> None:
    """R9b (F1, round 2): a string of Unicode decimal digits parses (Arabic-Indic
    ``"١٢"`` -> 12, since ``isdecimal()`` and ``int()`` agree), while one past
    ``int()``'s digit limit is a problem with value 0 - never an exception."""
    state, problems = RiskState.from_checkpoint(_full_risk(consecutive_losses=raw))
    assert state.consecutive_losses == expected
    if expected == 0:
        assert problems == ("consecutive_losses=" + repr(raw)[:40],)
    else:
        assert problems == ()


def test_r10_missing_key_is_a_problem_and_derived_keys_ignored() -> None:
    """R10: a stored key absent from a non-empty dict -> default + "<key> missing";
    the derived drawdown_pct/daily_loss_pct never count as state."""
    state, problems = RiskState.from_checkpoint({"high_water_mark": "10600"})
    assert state.high_water_mark == Decimal("10600")
    assert problems == (
        "day missing",
        "day_start_equity missing",
        "consecutive_losses missing",
        "last_equity missing",
    )

    derived_only, derived_problems = RiskState.from_checkpoint(
        {"drawdown_pct": "0.5", "daily_loss_pct": "0.1"}
    )
    assert (derived_only, derived_problems) == (RiskState(), ())


@pytest.mark.parametrize(("raw", "kind"), [("garbage", "str")])
def test_r11_non_dict_risk_is_a_problem(raw: object, kind: str) -> None:
    """R11: a non-dict risk value -> fresh state + "risk=<type>"."""
    state, problems = RiskState.from_checkpoint(raw)
    assert state == RiskState()
    assert problems == (f"risk={kind}",)


@pytest.mark.parametrize(
    ("key", "raw"),
    [("high_water_mark", "0"), ("high_water_mark", "-5"), ("day_start_equity", "0")],
)
def test_r20_non_positive_baseline_is_a_problem(key: str, raw: str) -> None:
    """R20: a non-positive high_water_mark / day_start_equity is a problem."""
    state, problems = RiskState.from_checkpoint(_full_risk(**{key: raw}))
    assert getattr(state, key) is None
    assert problems == (f"{key}={raw!r}",)


def test_r20_last_equity_non_positive_is_accepted() -> None:
    """R20 pins only high_water_mark/day_start_equity; last_equity <= 0 still restores."""
    state, problems = RiskState.from_checkpoint(_full_risk(last_equity="0"))
    assert state.last_equity == Decimal("0")
    assert problems == ()


def test_r21_partial_dict_is_a_problem_but_all_null_is_clean() -> None:
    """R21: a non-empty dict with some fields set and some null is partial -> a
    "<key> missing" problem per null field; an all-null dict stays clean (R1/R3)."""
    state, problems = RiskState.from_checkpoint(
        _full_risk(day=None, last_equity=None, consecutive_losses=0)
    )
    assert state.high_water_mark == Decimal("10600")
    assert state.day_start_equity == Decimal("10000")
    assert problems == ("day missing", "last_equity missing")

    clean, clean_problems = RiskState.from_checkpoint(_NULL_RISK)
    assert (clean, clean_problems) == (RiskState(), ())


# --------------------------------------------------------- node restore rows


async def test_r13_problems_with_prior_halt_keeps_the_prior_halt(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R13: problems alongside a restored MANUAL halt keep the prior reason+message;
    the problems go to the event and one logger.error, with no extra alert."""
    status = {"state": "halted", "halt_reason": "manual", "error_message": "halt via API"}
    _write_checkpoint(seeded["db"], status, risk=_full_risk(high_water_mark="10600x"))
    node, trace = _trace_node(monkeypatch, seeded)
    try:
        with caplog.at_level(logging.ERROR):
            await node._initialize()

        assert node.status.state == NodeState.HALTED
        assert node.status.halt_reason == HaltReason.MANUAL
        assert node.status.error_message == _RESTORED_MESSAGE

        restores = [c for c in trace.calls if c[0] == "write" and c[2].get("action") == "restore"]
        assert restores[0][2]["restored_halt"] == "manual"
        assert restores[0][2]["risk_problems"] == ["high_water_mark='10600x'"]

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert [r.getMessage() for r in errors] == [
            "paper PAPER-007: unreadable risk state in checkpoint "
            "(high_water_mark='10600x'); halting"
        ]
        assert [c for c in trace.calls if c[0] == "alert"] == []
    finally:
        await node._shutdown()


async def test_r15b_pre_first_tick_death_rearms_flatten(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R15b: a process that dies before the first guard tick (the checkpoint R12's
    restart writes in ``_initialize``) still restores ERROR on the next restart, and
    the guard built from it re-arms the deferred flatten."""
    _write_checkpoint(seeded["db"], {"state": "running"}, risk=_full_risk(high_water_mark="10600x"))
    node = PaperTradingNode(_config(seeded))
    await node._initialize()
    # "dies" here: read the checkpoint written before any tick, then shut down.
    latest = _latest_checkpoint(seeded["db"])
    assert latest is not None
    assert latest.node_status["halt_reason"] == "error"
    await node._shutdown()

    node2 = PaperTradingNode(_config(seeded))
    await node2._initialize()
    try:
        assert node2.status.state == NodeState.HALTED
        assert node2.status.halt_reason == HaltReason.ERROR
        # R15: restart 2 keeps the exact message (the prefix is not doubled).
        assert node2.status.error_message == _GARBLED_HWM_MESSAGE
    finally:
        await node2._shutdown()

    restored = restored_halt_from_status(latest.node_status, _ISO)
    assert restored is not None
    assert restored.reason == HaltReason.ERROR
    monkeypatch.setattr(harness, "NaiveStrategy", _StartupStrategy)
    h = harness.build(restored_halt=restored, strategy_kwargs={"mode": "every_bar"})
    calls = _count_flatten(h.guard)
    harness.run(h, _bars(h, [100.0] * 4))
    assert len(calls) == 1, "the restored flatten must be re-armed and run once"
    assert h.guard.guard_state == GuardState.HALTED
    assert h.guard.halt_reason == HaltReason.ERROR
    assert h.filled_entries() == []
    h.engine.dispose()


async def test_r16_failed_restart_keeps_error_halt_and_risk(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R16: R12 then a venue build failure on restart 1 -> the failure checkpoint keeps
    halt_reason=error AND the risk dict; restart 2 is HALTED/ERROR."""
    _write_checkpoint(seeded["db"], {"state": "running"}, risk=_full_risk(high_water_mark="10600x"))
    node = PaperTradingNode(_config(seeded))
    monkeypatch.setattr(node, "_create_live_trading_node", _raises_venue_build)
    with pytest.raises(RuntimeError, match="venue build failed"):
        await node._initialize()
    await node._shutdown()

    latest = _latest_checkpoint(seeded["db"])
    assert latest is not None
    assert latest.node_status["halt_reason"] == "error"
    # The checkpoint keeps whatever risk was restored (the readable fields survive the
    # garbled high_water_mark), not a blank state.
    restored_risk, problems = RiskState.from_checkpoint(_full_risk(high_water_mark="10600x"))
    assert problems == ("high_water_mark='10600x'",)
    assert latest.node_status["risk"] == restored_risk.to_dict()

    node2 = PaperTradingNode(_config(seeded))
    await node2._initialize()
    try:
        assert node2.status.state == NodeState.HALTED
        assert node2.status.halt_reason == HaltReason.ERROR
    finally:
        await node2._shutdown()


@pytest.mark.parametrize(
    ("column", "value", "expected_class"),
    [
        ("node_status", "not json", "JSONDecodeError"),
        ("created_at", "not a date", "ValueError"),
    ],
)
async def test_r18_unreadable_checkpoint_row_halts_and_keeps_risk(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    column: str,
    value: str,
    expected_class: str,
) -> None:
    """R18: an unreadable checkpoint row (bad JSON node_status, a non-dict node_status,
    a bad created_at) -> ERROR halt "checkpoint unreadable (<class>)" with a non-None
    risk state, so restart 2 is still HALTED/ERROR."""
    import sqlite3

    _write_checkpoint(seeded["db"], {"state": "running"}, risk=_full_risk())
    con = sqlite3.connect(seeded["db"])
    with con:
        con.execute(f"UPDATE paper_trading_checkpoints SET {column} = ?", (value,))
    con.close()

    node = PaperTradingNode(_config(seeded))
    await node._initialize()
    try:
        assert node.status.state == NodeState.HALTED
        assert node.status.halt_reason == HaltReason.ERROR
        assert node.status.error_message == (
            f"{RESTORED_HALT_PREFIX}checkpoint unreadable ({expected_class}); "
            "limits re-baselined at current equity"
        )
        assert node._restored_risk is not None
    finally:
        await node._shutdown()

    latest = _latest_checkpoint(seeded["db"])
    assert latest is not None
    assert latest.node_status["halt_reason"] == "error"
    node2 = PaperTradingNode(_config(seeded))
    await node2._initialize()
    try:
        assert node2.status.state == NodeState.HALTED
        assert node2.status.halt_reason == HaltReason.ERROR
    finally:
        await node2._shutdown()


async def test_r19_raise_before_risk_is_set_halts_and_keeps_risk(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R19: any raise inside ``_restore_from_checkpoint`` before ``_restored_risk`` is
    set behaves like R18 (fresh risk state, ERROR halt, restart 2 HALTED).

    F1 (round 3): the class-only ``logger.error`` on that path must carry NO
    ``exc_info`` and no exception text. The raise is chained (``ZZLEAK9`` from
    ``ZZINNER4``) because a traceback log prints both the raised message and its
    ``from`` cause: neither marker may reach the captured log, the restore event
    payload, or ``status.error_message``.
    """
    _write_checkpoint(seeded["db"], {"state": "running"}, risk=_full_risk())
    node, trace = _trace_node(monkeypatch, seeded)

    def _boom(_timestamp: datetime) -> int:
        raise RuntimeError("restore boom ZZLEAK9") from ValueError("ZZINNER4")

    monkeypatch.setattr(lineage, "checkpoint_ms", _boom)
    with caplog.at_level(logging.ERROR):
        await node._initialize()
    try:
        assert node.status.state == NodeState.HALTED
        assert node.status.halt_reason == HaltReason.ERROR
        assert node.status.error_message == (
            f"{RESTORED_HALT_PREFIX}checkpoint unreadable (RuntimeError); "
            "limits re-baselined at current equity"
        )
        assert node._restored_risk == RiskState()

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert [r.getMessage() for r in errors] == [
            "paper PAPER-007: checkpoint unreadable (RuntimeError); halting"
        ]
        assert all(r.exc_info is None for r in errors), "the failure is logged with a traceback"
        assert "ZZLEAK9" not in caplog.text, "the raised message leaked into the log"
        assert "ZZINNER4" not in caplog.text, "the chained cause leaked into the log"

        restores = [c for c in trace.calls if c[0] == "write" and c[2].get("action") == "restore"]
        assert [c[2] for c in restores] == [
            {
                "action": "restore",
                "checkpoint_time": None,
                "prior_state": None,
                "prior_halt_reason": None,
                "prior_positions": [],
                "prior_orders": [],
                "restored_risk": RiskState().to_dict(),
                "restored_halt": "error",
                "risk_problems": [],
                "risk_source_trader_id": None,
                "risk_source_checkpoint_time": None,
            }
        ]
        assert "ZZLEAK9" not in repr(restores[0][2]), "the raised message leaked into the event"
        assert "ZZINNER4" not in repr(restores[0][2]), "the chained cause leaked into the event"
        assert "ZZLEAK9" not in node.status.error_message
        assert "ZZINNER4" not in node.status.error_message
    finally:
        await node._shutdown()

    latest = _latest_checkpoint(seeded["db"])
    assert latest is not None
    assert latest.node_status["halt_reason"] == "error"
    assert latest.node_status["risk"] == RiskState().to_dict()
    node2 = PaperTradingNode(_config(seeded))
    await node2._initialize()
    try:
        assert node2.status.state == NodeState.HALTED
        assert node2.status.halt_reason == HaltReason.ERROR
    finally:
        await node2._shutdown()


class _SettablePortfolio:
    """A portfolio whose USDT equity the test can move between ticks."""

    def __init__(self, equity: str) -> None:
        self.equity_value = Decimal(equity)

    def account(self, venue: Any) -> object:
        return object()

    def missing_price_instruments(self, venue: Any) -> list[Any]:
        return []

    def equity(self, venue: Any) -> dict[Any, Any]:
        return {self._currency(): _FakeMoney(str(self.equity_value))}

    @staticmethod
    def _currency() -> Any:
        from nautilus_trader.model.objects import Currency

        return Currency.from_str("USDT")


class _SettableGuard(_UnitGuard):
    def __init__(self, *args: Any, equity: str = "9000", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._settled = _SettablePortfolio(equity)

    @property
    def portfolio(self) -> Any:  # type: ignore[override]
        return self._settled


def test_r17_operator_resume_after_r12_rebaselines_and_re_halts() -> None:
    """R17: after an operator resume of an R12 ERROR halt at equity 9000 the HWM is
    re-baselined to 9000; the next drawdown halt fires at <= 7650, not at 7651."""
    guard = _SettableGuard(
        TradingGuardConfig(max_drawdown_pct=Decimal("0.15"), max_daily_loss_pct=Decimal("0.5")),
        strategies=[],
        risk_engine=_RiskEngine(),
        restored_halt=RestoredHalt(HaltReason.ERROR, _GARBLED_HWM_MESSAGE, _ISO),
        equity="9000",
    )
    assert guard.guard_state == GuardState.HALTED
    assert guard.halt_reason == HaltReason.ERROR

    guard.evaluate_risk()
    assert guard.risk_state.high_water_mark == Decimal("9000")
    assert guard.guard_state == GuardState.HALTED, "no breach yet, the ERROR halt stands"

    assert guard.resume() == (True, "resumed from halt")
    assert guard.guard_state == GuardState.RUNNING

    guard._settled.equity_value = Decimal("7651")
    guard.evaluate_risk()
    assert guard.guard_state == GuardState.RUNNING, "7651 is one unit above the limit"

    guard._settled.equity_value = Decimal("7650")
    guard.evaluate_risk()
    assert guard.guard_state == GuardState.HALTED
    assert guard.halt_reason == HaltReason.MAX_DRAWDOWN


@pytest.mark.parametrize("column", ["positions", "orders"])
async def test_restore_event_failure_keeps_halt_and_risk(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
    column: str,
) -> None:
    """A corrupt positions or orders column (JSON null) breaks only the restore event.

    The risk, halt and HALTED status are installed before the event is built, so the
    TypeError escaping ``start()`` still leaves them in the failure checkpoint and the
    next restart comes up HALTED with the full risk.
    """
    import sqlite3

    _write_checkpoint(
        seeded["db"],
        {"state": "halted", "halt_reason": "manual", "error_message": "halt via API"},
        risk=_full_risk(),
    )
    with sqlite3.connect(seeded["db"]) as conn:
        conn.execute(f"UPDATE paper_trading_checkpoints SET {column} = ?", ("null",))
    node = PaperTradingNode(_config(seeded))
    monkeypatch.setattr(node, "_setup_signal_handlers", lambda: None)
    with pytest.raises(TypeError, match="not iterable"):
        await node.start()
    saved = _latest_checkpoint(seeded["db"])
    assert saved is not None
    assert saved.node_status.get("halt_reason") == "manual"
    assert saved.node_status.get("risk") == _full_risk()

    node2 = PaperTradingNode(_config(seeded))
    try:
        await node2._initialize()
        assert node2.status.state == NodeState.HALTED
        assert node2.status.halt_reason is not None
        assert node2._restored_risk is not None
        assert node2._restored_risk.to_dict() == _full_risk()
    finally:
        await node2._shutdown()
