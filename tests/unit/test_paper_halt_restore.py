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
from functools import partial
from typing import TYPE_CHECKING, Any

import pytest

import tests.unit.test_audit_paper_guard as harness
from tests.unit.test_audit_paper_node import DSL, _config, _FakeTradingNode
from vibe_quant.db.state_manager import StateManager
from vibe_quant.logging.events import EventType
from vibe_quant.paper.guard import (
    RESTORED_HALT_PREFIX,
    GuardState,
    HaltReason,
    RestoredHalt,
    RiskState,
    TradingGuard,
    TradingGuardConfig,
)
from vibe_quant.paper.node import NodeState, PaperTradingNode, restored_halt_from_status
from vibe_quant.paper.persistence import StateCheckpoint, StatePersistence

if TYPE_CHECKING:
    from pathlib import Path

from nautilus_trader.model.enums import OrderSide

#: The prior checkpoint of contract row H1 and its restored message.
_ISO = "2026-10-15T10:00:00+00:00"
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


def _restored_harness(
    monkeypatch: pytest.MonkeyPatch, **build_kw: Any
) -> harness.Harness:
    """NT harness whose guard is built through the real restore parse."""
    restored = restored_halt_from_status(_PRIOR_MANUAL, _ISO)
    assert restored is not None
    monkeypatch.setattr(harness, "RecordingListener", _MetricsListener)
    monkeypatch.setattr(harness, "TradingGuard", partial(TradingGuard, restored_halt=restored))
    monkeypatch.setattr(harness, "NaiveStrategy", _StartupStrategy)
    return harness.build(**build_kw)


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
    """A node whose event writes and alerts land in one ordered trace."""
    trace = _Trace()
    node = PaperTradingNode(_config(seeded, **config_kw))
    monkeypatch.setattr(node, "_write_event", trace.write)
    monkeypatch.setattr(node, "_write_then_alert", trace.write_then_alert)
    monkeypatch.setattr(node, "_send_telegram_alert", trace.alert)
    return node, trace


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
        "system kill switch unreadable for 3 consecutive reads "
        "(OperationalError); failing closed"
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


async def test_pause_not_restored(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
) -> None:
    """H15: a prior PAUSED state (no halt_reason) comes back RUNNING."""
    _write_checkpoint(seeded["db"], {"state": "paused"})
    node = PaperTradingNode(_config(seeded))
    await node._initialize()
    try:
        await node._run_loop()
        assert node.status.state == NodeState.RUNNING
        assert node._restored_halt is None
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
        monkeypatch.setattr(
            node._state_manager,
            "get_system_state",
            lambda **_: {"kill_switch": True, "reason": "ops"},
        )
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
        fallback = restored_halt_from_status({"halt_reason": "manual", "error_message": blank}, _ISO)
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


async def test_restored_halt_survives_failed_restart(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H19: the halt survives every checkpoint written before ``_run_loop`` -
    a restart whose TradingNode build fails still checkpoints
    ``halt_reason == "manual"``, and a second restart comes up HALTED with its
    own ``_initialize`` checkpoint keeping the halt too."""
    _write_checkpoint(seeded["db"], _PRIOR_MANUAL)
    node = PaperTradingNode(_config(seeded))
    monkeypatch.setattr(node, "_create_live_trading_node", _raises_venue_build)
    with pytest.raises(RuntimeError, match="venue build failed"):
        await node._initialize()
    await node._shutdown()

    latest = _latest_checkpoint(seeded["db"])
    assert latest is not None
    assert latest.node_status["halt_reason"] == "manual"

    node2 = PaperTradingNode(_config(seeded))
    await node2._initialize()
    try:
        assert node2.status.state == NodeState.HALTED
        assert node2.status.halt_reason == HaltReason.MANUAL
        after_init = _latest_checkpoint(seeded["db"])
        assert after_init is not None
        assert after_init.node_status["halt_reason"] == "manual"
    finally:
        await node2._shutdown()


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
    message, ``_state_before_halt``); ``on_start`` arms the flatten flag and
    notifies only while still HALTED, so a resume before start gives a real
    RUNNING session: no flatten, no halt notify, strategies trade normally."""
    h = _restored_harness(monkeypatch, strategy_kwargs={"mode": "every_bar"})
    guard = h.guard
    assert guard.guard_state == GuardState.HALTED
    assert guard.halt_reason == HaltReason.MANUAL
    assert guard.status()["message"] == _RESTORED_MESSAGE
    assert guard._state_before_halt == GuardState.RUNNING
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
    monkeypatch.setattr(harness, "TradingGuard", partial(TradingGuard, restored_halt=restored))
    monkeypatch.setattr(harness, "NaiveStrategy", _StartupStrategy)
    h = harness.build(symbols=("BTCUSDT", "ETHUSDT"), strategy_kwargs={"mode": "every_bar"})
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
