"""A stopping halt applied before a guarded strategy is running stops it (F1).

``TradingGuard`` is built by ``PaperTradingNode._build_guard`` and started by NT
*before* the strategies. When a stopping halt trips inside ``TradingGuard.on_start``
(``evaluate_risk`` breaching a restored bookkeeping limit, or a restored halt
escalated there), every strategy is still READY: there is nothing to flatten yet,
but each one must still be stopped on its first running tick (C1). A strategy
already STOPPED keeps the ``_flatten_stopped_strategy`` path (C2), a resume before
that tick clears the pending entry (C3), and a keep-running halt records nothing
(C4). An escalated restored halt notifies once, not twice (C6/C7), and the whole
pre-start halt is pinned against the real NT kernel (C8).

Contract F1 rows C1-C8 (beads vibe-quant-cplc9, vibe-quant-msocy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from nautilus_trader.core.rust.common import ComponentState
from nautilus_trader.model.enums import OrderSide

import tests.unit.test_audit_paper_guard as harness
from tests.unit.test_paper_halt_restore import (
    _ISO,
    _RESTORED_MESSAGE,
    _trace_node,
)
from tests.unit.test_paper_halt_restore import (
    seeded as _restore_seeded,
)
from vibe_quant.paper.guard import (
    GuardState,
    HaltReason,
    RestoredHalt,
    RiskState,
    TradingGuard,
    TradingGuardConfig,
)

# --------------------------------------------------------------- unit fakes


@pytest.fixture()
def seeded(tmp_path: Any) -> dict[str, Any]:
    """The H-test seeded strategy/validation fixture (single source of truth)."""
    return _restore_seeded.__wrapped__(tmp_path)  # type: ignore[attr-defined]


class _Cache:
    """Fake cache: no positions unless seeded, plus an optional open order."""

    def __init__(self, orders: list[Any] | None = None) -> None:
        self._orders = orders or []

    def positions_open(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    def orders_open(self, *args: Any, **kwargs: Any) -> list[Any]:
        return list(self._orders)

    def orders_inflight(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    def order_exists(self, *args: Any, **kwargs: Any) -> bool:
        return False


class _Money:
    def as_decimal(self) -> Decimal:
        return Decimal("10000")


class _Portfolio:
    def account(self, venue: Any) -> object:
        return object()

    def missing_price_instruments(self, venue: Any) -> list[Any]:
        return []

    def equity(self, venue: Any) -> dict[Any, Any]:
        from nautilus_trader.model.objects import Currency

        return {Currency.from_str("USDT"): _Money()}


class _Clock:
    def utc_now(self) -> datetime:
        return datetime(2026, 1, 5, 22, 0, tzinfo=UTC)

    def timestamp_ns(self) -> int:
        return 0

    def set_timer(self, name: str, interval: timedelta, callback: Any) -> None:
        return None


class _Bus:
    def register(self, endpoint: str, handler: Any) -> None:
        return None

    def deregister(self, endpoint: str, handler: Any) -> None:
        return None

    def subscribe(self, topic: str, handler: Any) -> None:
        return None

    def is_subscribed(self, topic: str, handler: Any) -> bool:
        return False


class _RiskEngine:
    def execute(self, command: object) -> None:
        return None


@dataclass
class _Strategy:
    """Guarded strategy double whose NT component state the test drives."""

    id: str
    is_running: bool = False
    is_stopped: bool = False
    state: ComponentState = ComponentState.READY
    exits: int = 0
    stops: int = 0
    cancels: list[Any] = field(default_factory=list)
    closes: list[Any] = field(default_factory=list)

    def is_exiting(self) -> bool:
        return False

    def market_exit(self) -> None:
        self.exits += 1

    def stop(self) -> None:
        self.stops += 1
        self.is_running = False
        self.is_stopped = True
        self.state = ComponentState.STOPPED

    def start(self) -> None:
        self.is_running = True
        self.is_stopped = False
        self.state = ComponentState.RUNNING

    def resume(self) -> None:
        self.is_running = True
        self.is_stopped = False
        self.state = ComponentState.RUNNING

    def cancel_all_orders(self, instrument_id: Any) -> None:
        self.cancels.append(instrument_id)

    def close_all_positions(self, instrument_id: Any, *, reduce_only: bool = False) -> None:
        self.closes.append((instrument_id, reduce_only))


@dataclass
class _Order:
    instrument_id: str


class _UnitGuard(TradingGuard):
    """Real guard with fake cache/portfolio/clock/msgbus: an unregistered Actor
    gets ``cache = None`` (``_flatten`` would crash) and a None msgbus."""

    def __init__(self, *args: Any, cache: Any = None, **kwargs: Any) -> None:
        self._cache = cache if cache is not None else _Cache()
        super().__init__(*args, **kwargs)

    @property
    def cache(self) -> Any:  # type: ignore[override]
        return self._cache

    @property
    def portfolio(self) -> Any:  # type: ignore[override]
        return _Portfolio()

    @property
    def clock(self) -> Any:  # type: ignore[override]
        return _Clock()

    @property
    def msgbus(self) -> Any:  # type: ignore[override]
        return _Bus()


def _unit_guard(
    strategies: list[Any],
    *,
    cache: Any = None,
    restored_halt: RestoredHalt | None = None,
    risk_state: RiskState | None = None,
    limits: dict[str, Any] | None = None,
) -> TradingGuard:
    return _UnitGuard(
        TradingGuardConfig(**(limits or {})),
        strategies=strategies,
        risk_engine=_RiskEngine(),
        risk_state=risk_state,
        restored_halt=restored_halt,
        cache=cache,
    )


# --------------------------------------------------------------- C1


@pytest.mark.parametrize("reason", [HaltReason.MANUAL, HaltReason.KILL_SWITCH])
def test_stopping_halt_before_start_stops_strategy_on_first_running_tick(
    reason: HaltReason,
) -> None:
    """C1: a stopping halt while the strategy is READY (never started) records it
    for stopping; a tick that still finds it not running must NOT discard it; the
    first tick where it runs gives it ``market_exit`` and stops it once exited -
    the same ``_pending_stop`` / ``_stop_exited_strategies`` path as a running
    strategy."""
    strategy = _Strategy("strat-1")
    guard = _unit_guard([strategy])
    assert guard.halt(reason, "pre-start halt") is True
    assert guard.guard_state == GuardState.HALTED
    assert guard._pending_stop == {"strat-1"}, "the ready strategy must be recorded"
    assert guard._pending_exit == {"strat-1"}, "and it still owes a market exit"
    assert strategy.exits == 0, "nothing to exit before it runs"

    guard._on_tick(None)  # not started yet: the pending entry must survive
    assert guard._pending_stop == {"strat-1"}
    assert guard._pending_exit == {"strat-1"}
    assert strategy.stops == 0

    strategy.start()
    guard._on_tick(None)
    assert strategy.exits == 1, "market_exit runs on the first running tick (no-op when flat)"
    assert guard._pending_exit == set()
    assert strategy.stops == 0, "not stopped until that exit has finished"
    assert guard._pending_stop == {"strat-1"}

    guard._on_tick(None)
    assert guard._pending_stop == set()
    assert strategy.stops == 1, "and the strategy is stopped once exited"


# --------------------------------------------------------------- C2


def test_flatten_leaves_a_stopped_strategy_on_the_stopped_path() -> None:
    """C2: a STOPPED strategy keeps today's path - ``_flatten_stopped_strategy``
    cancels/closes through its order API and it is never re-added to
    ``_pending_stop``."""
    cache = _Cache(orders=[_Order("BTCUSDT-PERP.BINANCE")])
    strategy = _Strategy("strat-1", is_stopped=True, state=ComponentState.STOPPED)
    guard = _unit_guard([strategy], cache=cache)

    result = guard._flatten(stop_strategies=True)

    assert result.errors == []
    assert guard._pending_stop == set(), "a stopped strategy is not coming back"
    assert guard._pending_exit == set()
    assert strategy.exits == 0
    assert strategy.cancels == ["BTCUSDT-PERP.BINANCE"]
    assert strategy.closes == [("BTCUSDT-PERP.BINANCE", True)]


@pytest.mark.parametrize(
    "state",
    [ComponentState.PRE_INITIALIZED, ComponentState.READY, ComponentState.STARTING],
)
def test_every_never_started_state_is_recorded(state: ComponentState) -> None:
    """C1/C2: every NT state in which a strategy has not started yet is recorded
    (not sent down the stopped path), so a mutant dropping one from
    ``_STRATEGY_NOT_STARTED`` fails here."""
    strategy = _Strategy("strat-1", state=state)
    guard = _unit_guard([strategy])
    guard.halt(HaltReason.MANUAL, "pre-start halt")
    assert guard._pending_stop == {"strat-1"}
    assert guard._pending_exit == {"strat-1"}


def test_a_faulted_or_degraded_strategy_keeps_the_stopped_path() -> None:
    """C2: a strategy that is neither running nor ever going to start (DEGRADED/
    FAULTED) is not "not started" - it keeps ``_flatten_stopped_strategy``."""
    cache = _Cache(orders=[_Order("BTCUSDT-PERP.BINANCE")])
    strategy = _Strategy("strat-1", state=ComponentState.DEGRADED)
    guard = _unit_guard([strategy], cache=cache)

    guard.halt(HaltReason.MANUAL, "halt")

    assert guard._pending_stop == set()
    assert guard._pending_exit == set()
    assert strategy.cancels == ["BTCUSDT-PERP.BINANCE"]
    assert strategy.closes == [("BTCUSDT-PERP.BINANCE", True)]


# --------------------------------------------------------------- C3


def test_resume_before_the_first_running_tick_clears_the_pending_stop() -> None:
    """C3: lifting a pre-start halt (resume -> ``_lift_halt``) before the first
    running tick clears the pending entry, so a later tick never stops or exits
    the strategy."""
    strategy = _Strategy("strat-1")
    guard = _unit_guard([strategy])
    guard.halt(HaltReason.MANUAL, "pre-start halt")
    assert guard._pending_stop == {"strat-1"}

    assert guard.resume() == (True, "resumed from halt")
    assert guard.guard_state == GuardState.RUNNING
    assert guard._pending_stop == set(), "_lift_halt must discard the pending stop"
    assert guard._pending_exit == set(), "... including the deferred market exit"

    strategy.start()
    guard._on_tick(None)
    assert strategy.exits == 0
    assert strategy.stops == 0


# --------------------------------------------------------------- C4


def test_keep_running_halt_does_not_record_a_prestart_strategy() -> None:
    """C4: a MAX_DAILY_LOSS halt (in ``_KEEP_STRATEGIES_RUNNING``) records no
    pending stop for a strategy that has not started, so it runs normally."""
    strategy = _Strategy("strat-1")
    guard = _unit_guard([strategy])
    guard.halt(HaltReason.MAX_DAILY_LOSS, "daily loss")
    assert guard.guard_state == GuardState.HALTED
    assert guard._pending_stop == set()
    assert guard._pending_exit == set()

    strategy.start()
    guard._on_tick(None)
    assert strategy.exits == 0
    assert strategy.stops == 0


# --------------------------------------------------------------- C6/C7


def _prestart_guard(node: Any, *, risk: RiskState | None) -> TradingGuard:
    """A guard wired to a node bridge, halted MANUAL, risk as given."""
    guard = _unit_guard(
        [],
        restored_halt=RestoredHalt(
            reason=HaltReason.MANUAL, message=_RESTORED_MESSAGE, checkpoint_time=_ISO
        ),
        risk_state=risk,
        limits={"max_drawdown_pct": Decimal("0.05"), "max_daily_loss_pct": Decimal("0.5")},
    )
    guard._listener.on_state_change = node._on_guard_state_change  # type: ignore[assignment]
    return guard


def test_escalated_restored_halt_notifies_once(
    monkeypatch: pytest.MonkeyPatch, seeded: dict[str, Any]
) -> None:
    """C6 (msocy): the restored MANUAL halt escalated by ``evaluate_risk`` inside
    ``on_start`` produces exactly ONE RISK_CHECK event and ONE critical alert -
    the escalating MAX_DRAWDOWN halt's - never the restored one too."""
    risk = RiskState(
        high_water_mark=Decimal("10600"),
        day=date(2026, 1, 5),
        day_start_equity=Decimal("10000"),
    )
    node, trace = _trace_node(monkeypatch, seeded)
    guard = _prestart_guard(node, risk=risk)

    guard.on_start()

    assert guard.guard_state == GuardState.HALTED
    assert guard.halt_reason == HaltReason.MAX_DRAWDOWN
    assert guard._restored_flatten_pending is False, "the escalated halt owns the flatten"
    halts = [c for c in trace.calls if c[0] == "write" and c[2].get("action") == "halt"]
    assert len(halts) == 1
    assert halts[0][2]["reason"] == "max_drawdown"
    assert [c[2]["restored"] for c in halts if "restored" in c[2]] == [], (
        "the escalated halt must not repeat the restored notify"
    )
    alerts = [c for c in trace.calls if c[0] == "alert"]
    assert len(alerts) == 1
    assert alerts[0][1] == "circuit_breaker"
    assert alerts[0][3] is True
    assert "max_drawdown" in alerts[0][2]


def test_unescalated_restored_halt_still_notifies_once(
    monkeypatch: pytest.MonkeyPatch, seeded: dict[str, Any]
) -> None:
    """C6: a restored halt that is NOT escalated still notifies exactly once,
    with ``restored: True`` and the checkpoint time."""
    node, trace = _trace_node(monkeypatch, seeded)
    guard = _prestart_guard(node, risk=None)

    guard.on_start()

    assert guard.halt_reason == HaltReason.MANUAL
    assert guard._restored_flatten_pending is True
    halts = [c for c in trace.calls if c[0] == "write" and c[2].get("action") == "halt"]
    assert len(halts) == 1
    assert halts[0][2]["check_type"] == "manual"
    assert halts[0][2]["restored"] is True
    assert halts[0][2]["checkpoint_time"] == _ISO
    assert [c for c in trace.calls if c[0] == "alert"] == [
        ("alert", "circuit_breaker", f"HALT (manual): {_RESTORED_MESSAGE}", True)
    ]


# --------------------------------------------------------------- C8 (NT harness)


class _StartupStrategy(harness.NaiveStrategy):
    """Submits one entry in ``on_start`` and one on every bar."""

    def on_start(self) -> None:
        super().on_start()
        self.submit_order(
            self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=OrderSide.BUY,
                quantity=self.instrument.make_qty(Decimal("1")),
            )
        )


def _bars(h: harness.Harness, closes: list[float]) -> list[Any]:
    return [b for s in h.strategies for b in harness._flat_bars(s.config.bar_type, closes)]


def test_prestart_drawdown_halt_stops_and_blocks_the_strategies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C8: drawdown breached in the guard's ``on_start`` (before the strategies
    are running) with strategies entering on start and every bar: every entry is
    gate-denied (0 filled), the strategies are stopped by the first ticks after
    they start, and the end state is flat + stopped + HALTED/MAX_DRAWDOWN."""
    risk = RiskState(
        high_water_mark=Decimal("10600"),
        day=date(2026, 1, 5),
        day_start_equity=Decimal("10000"),
    )
    monkeypatch.setattr(harness, "NaiveStrategy", _StartupStrategy)
    h = harness.build(
        symbols=("BTCUSDT", "ETHUSDT"),
        strategy_kwargs={"mode": "every_bar"},
        limits={"max_drawdown_pct": Decimal("0.05"), "max_daily_loss_pct": Decimal("0.5")},
        risk_state=risk,
    )
    snaps: dict[str, Any] = {}
    harness.run(
        h,
        _bars(h, [10000.0] * 5),
        actions=[(h.at(0.5), h.snap(snaps, "t0.5")), (h.at(3.5), h.snap(snaps, "t3.5"))],
    )

    assert h.guard.guard_state == GuardState.HALTED
    assert h.guard.halt_reason == HaltReason.MAX_DRAWDOWN
    assert snaps["t0.5"]["running"] == [True, True], "the strategies really started"
    assert snaps["t3.5"]["running"] == [False, False], "and were stopped after starting"
    assert h.engine.cache.positions_open() == []
    assert len(h.filled_entries()) == 0
    submitted = len(h.entry_orders())
    assert submitted >= 1, "the gate must really be exercised"
    assert [by_gate for _e, by_gate in h.listener.denied] == [True] * submitted
    h.engine.dispose()
