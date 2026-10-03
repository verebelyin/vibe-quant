"""TradingGuard against a real NautilusTrader BacktestEngine (vibe-quant-e70tl.4).

Every scenario runs the real NT kernel (RiskEngine, ExecEngine, Portfolio,
matching engine) with a deliberately naive strategy that tries to trade on
every bar, so the guard -- not strategy discipline -- is what keeps the
account safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import FillModel, LatencyModel
from nautilus_trader.common.actor import Actor
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import (
    AccountType,
    OmsType,
    OrderSide,
    OrderStatus,
    TimeInForce,
)
from nautilus_trader.model.events import PositionClosed, PositionOpened
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.objects import Currency, Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

from vibe_quant.data.catalog import create_instrument, get_bar_type
from vibe_quant.nt_compat import retain_log_guard
from vibe_quant.paper.guard import (
    GATE_DENIAL_PREFIX,
    GuardListener,
    GuardState,
    HaltReason,
    RiskState,
    TradingGuard,
    TradingGuardConfig,
)

if TYPE_CHECKING:
    from collections.abc import Callable

START = datetime(2026, 1, 5, 22, 0, tzinfo=UTC)
_ENGINES: list[BacktestEngine] = []


def _new_engine() -> BacktestEngine:
    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    retain_log_guard(engine)
    _ENGINES.append(engine)
    return engine


# --------------------------------------------------------------------------- strategy


class NaiveConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: str = "1.000"
    side: str = "BUY"
    mode: str = "when_flat"  # when_flat | every_bar | once
    sl_pct: float | None = None
    tp_pct: float | None = None


class NaiveStrategy(Strategy):
    """Enters per ``mode``; places reduce-only SL/TP after the entry fills."""

    def __init__(self, config: NaiveConfig) -> None:
        super().__init__(config)
        self.entries_submitted = 0
        self.entered_once = False
        self.instrument: Any = None

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        mode = self.config.mode
        if mode == "once" and self.entered_once:
            return
        if mode == "when_flat" and not self.portfolio.is_flat(self.config.instrument_id):
            return
        side = OrderSide.BUY if self.config.side == "BUY" else OrderSide.SELL
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=side,
            quantity=self.instrument.make_qty(Decimal(self.config.trade_size)),
        )
        self.entries_submitted += 1
        self.entered_once = True
        self.submit_order(order)

    def on_event(self, event: Any) -> None:
        if isinstance(event, PositionOpened) and (self.config.sl_pct or self.config.tp_pct):
            pos = self.cache.position(event.position_id)
            px = float(pos.avg_px_open)
            is_long = pos.is_long
            exit_side = OrderSide.SELL if is_long else OrderSide.BUY
            if self.config.sl_pct:
                trig = px * (1 - self.config.sl_pct) if is_long else px * (1 + self.config.sl_pct)
                self.submit_order(
                    self.order_factory.stop_market(
                        instrument_id=self.config.instrument_id,
                        order_side=exit_side,
                        quantity=pos.quantity,
                        trigger_price=self.instrument.make_price(trig),
                        time_in_force=TimeInForce.GTC,
                        reduce_only=True,
                    )
                )
            if self.config.tp_pct:
                lim = px * (1 + self.config.tp_pct) if is_long else px * (1 - self.config.tp_pct)
                self.submit_order(
                    self.order_factory.limit(
                        instrument_id=self.config.instrument_id,
                        order_side=exit_side,
                        quantity=pos.quantity,
                        price=self.instrument.make_price(lim),
                        time_in_force=TimeInForce.GTC,
                        reduce_only=True,
                    )
                )
        elif isinstance(event, PositionClosed):
            self.cancel_all_orders(self.config.instrument_id)


class Scheduler(Actor):
    """Runs callbacks at fixed simulated times (operator actions mid-backtest)."""

    def __init__(self, actions: list[tuple[datetime, Callable[[], None]]]) -> None:
        super().__init__()
        self._actions = actions

    def on_start(self) -> None:
        for idx, (when, fn) in enumerate(self._actions):
            self.clock.set_time_alert(
                name=f"sched-{idx}", alert_time=when, callback=lambda _e, fn=fn: fn()
            )


# --------------------------------------------------------------------------- harness


@dataclass
class RecordingListener(GuardListener):
    clock_ns: Callable[[], int] = lambda: 0
    changes: list[tuple[int, GuardState, HaltReason | None, str]] = field(default_factory=list)
    rejected: list[Any] = field(default_factory=list)
    denied: list[tuple[Any, bool]] = field(default_factory=list)
    opened: list[Any] = field(default_factory=list)
    closed: list[Any] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def on_state_change(
        self,
        state: GuardState,
        reason: HaltReason | None,
        message: str,
        metrics: dict[str, object],
    ) -> None:
        self.changes.append((self.clock_ns(), state, reason, message))

    def on_order_rejected(self, event: Any) -> None:
        self.rejected.append(event)

    def on_order_denied(self, event: Any, *, by_gate: bool) -> None:
        self.denied.append((event, by_gate))

    def on_position_opened(self, event: Any) -> None:
        self.opened.append(event)

    def on_position_closed(self, event: Any) -> None:
        self.closed.append(event)

    def on_warning(self, message: str) -> None:
        self.warnings.append(message)


def _bar(bar_type: BarType, minute: int, o: float, h: float, lo: float, c: float) -> Bar:
    ts_open = int((START + timedelta(minutes=minute)).timestamp() * 1e9)
    fmt = "{:.1f}" if "BTC" in str(bar_type) else "{:.2f}"
    return Bar(
        bar_type=bar_type,
        open=Price.from_str(fmt.format(o)),
        high=Price.from_str(fmt.format(h)),
        low=Price.from_str(fmt.format(lo)),
        close=Price.from_str(fmt.format(c)),
        volume=Quantity.from_str("1000.000"),
        ts_event=ts_open,
        ts_init=ts_open + 59_999_000_000,
    )


def _flat_bars(bar_type: BarType, closes: list[float]) -> list[Bar]:
    bars = []
    prev = closes[0]
    for i, c in enumerate(closes):
        o = prev
        bars.append(_bar(bar_type, i, o, max(o, c) + 1, min(o, c) - 1, c))
        prev = c
    return bars


@dataclass
class Harness:
    engine: BacktestEngine
    guard: TradingGuard
    strategies: list[NaiveStrategy]
    listener: RecordingListener

    def at(self, minute: float) -> datetime:
        return START + timedelta(minutes=minute)

    def orders(self) -> list[Any]:
        return list(self.engine.cache.orders())

    def entry_orders(self) -> list[Any]:
        return [o for o in self.orders() if not o.is_reduce_only]

    def filled_entries(self) -> list[Any]:
        return [o for o in self.entry_orders() if o.status == OrderStatus.FILLED]

    def snap(self, store: dict[str, Any], key: str) -> Callable[[], None]:
        """Action recording strategy/position state mid-run (engine end stops all)."""

        def _record() -> None:
            store[key] = {
                "running": [s.is_running for s in self.strategies],
                "positions_open": len(self.engine.cache.positions_open()),
                "orders_open": [
                    (o.order_type, o.is_reduce_only) for o in self.engine.cache.orders_open()
                ],
                "state": self.guard.guard_state,
            }

        return _record

    def accepted_entries_after(self, ts_ns: int) -> list[Any]:
        """Opening orders after ``ts_ns`` that were NOT denied (i.e. reached the venue)."""
        return [
            o for o in self.entry_orders() if o.ts_init >= ts_ns and o.status != OrderStatus.DENIED
        ]

    def halt_ns(self) -> int:
        halts = [c for c in self.listener.changes if c[1] == GuardState.HALTED]
        assert halts, f"no halt recorded: {self.listener.changes}"
        return halts[0][0]


def build(
    *,
    symbols: tuple[str, ...] = ("BTCUSDT",),
    strategy_kwargs: dict[str, Any] | None = None,
    limits: dict[str, Any] | None = None,
    latency_ns: int | None = None,
    starting_usdt: int = 10_000,
    risk_state: RiskState | None = None,
) -> Harness:
    engine = _new_engine()
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(starting_usdt, Currency.from_str("USDT"))],
        default_leverage=Decimal("10"),
        fill_model=FillModel(prob_fill_on_limit=1.0, prob_slippage=0.0),
        latency_model=LatencyModel(base_latency_nanos=latency_ns) if latency_ns else None,
        bar_execution=True,
    )
    strategies: list[NaiveStrategy] = []
    for idx, symbol in enumerate(symbols):
        engine.add_instrument(create_instrument(symbol))
        cfg = NaiveConfig(
            instrument_id=InstrumentId.from_str(f"{symbol}-PERP.BINANCE"),
            bar_type=get_bar_type(symbol, "1m"),
            order_id_tag=f"{idx:03d}",
            **(strategy_kwargs or {}),
        )
        strategies.append(NaiveStrategy(cfg))
    listener = RecordingListener()
    guard = TradingGuard(
        TradingGuardConfig(check_interval_secs=60, **(limits or {})),
        strategies=strategies,
        risk_engine=engine.kernel.risk_engine,
        listener=listener,
        risk_state=risk_state,
    )
    listener.clock_ns = lambda: guard.clock.timestamp_ns()
    engine.add_actor(guard)
    for s in strategies:
        engine.add_strategy(s)
    return Harness(engine=engine, guard=guard, strategies=strategies, listener=listener)


def run(
    h: Harness, bars: list[Bar], actions: list[tuple[datetime, Callable[[], None]]] | None = None
) -> None:
    if actions:
        h.engine.add_actor(Scheduler(actions))
    h.engine.add_data(bars)
    h.engine.run()


@pytest.fixture(autouse=True)
def _dispose_engines() -> Any:
    yield
    while _ENGINES:
        eng = _ENGINES.pop()
        retain_log_guard(eng)
        eng.dispose()


# --------------------------------------------------------------------------- tests


def test_max_drawdown_halts_flattens_and_blocks_new_entries() -> None:
    """max_drawdown_pct=0.05, price slides 6% -> halt, flat, strategy stopped, no new entries."""
    h = build(limits={"max_drawdown_pct": Decimal("0.05"), "max_daily_loss_pct": Decimal("0.5")})
    bt = get_bar_type("BTCUSDT", "1m")
    # 1 BTC long at 10000 (bar 0 close). 1% per bar slide: equity DD crosses 5%
    # at close 9500 (loss 500 + 5 fee = 5.05%), the 6% bar (9400) never trades.
    closes = [10000, 9900, 9800, 9700, 9600, 9500, 9400, 9400, 9400, 9400, 9400, 9400]
    snaps: dict[str, Any] = {}
    run(h, _flat_bars(bt, closes), actions=[(h.at(9.5), h.snap(snaps, "after"))])

    assert h.guard.guard_state == GuardState.HALTED
    assert h.guard.halt_reason == HaltReason.MAX_DRAWDOWN
    assert h.engine.cache.positions_open() == []
    assert snaps["after"]["running"] == [False], "strategy must be stopped after a DD halt"
    assert snaps["after"]["positions_open"] == 0
    halt_ns = h.halt_ns()
    # Exactly one entry ever filled; nothing opening reached the venue after the halt
    # (the naive strategy's attempts during the market exit were gate-denied).
    assert len(h.filled_entries()) == 1
    assert h.accepted_entries_after(halt_ns) == []
    assert all(by_gate for _, by_gate in h.listener.denied)
    # The flattening order is reduce-only and filled.
    closers = [o for o in h.orders() if o.is_reduce_only and o.status == OrderStatus.FILLED]
    assert len(closers) == 1
    dd = h.guard.risk_state.drawdown_pct
    assert dd is not None and dd >= Decimal("0.05")


def test_daily_loss_halts_until_utc_rollover() -> None:
    """2% daily loss on day 1 -> flat + entries denied until 00:00 UTC, then trading resumes."""
    h = build(
        strategy_kwargs={"mode": "when_flat"},
        limits={"max_drawdown_pct": Decimal("0.5"), "max_daily_loss_pct": Decimal("0.02")},
    )
    bt = get_bar_type("BTCUSDT", "1m")
    # 22:00 entry @10000; slide to 9700 by 22:03 (loss 300 + 5 fee = 3.05%) -> halt.
    # Then flat 9700 until 00:20 next day (140 bars total).
    closes = [10000.0, 9900.0, 9800.0, 9700.0] + [9700.0] * 136
    snaps: dict[str, Any] = {}
    run(
        h,
        _flat_bars(bt, closes),
        actions=[(h.at(60.5), h.snap(snaps, "halted")), (h.at(135.5), h.snap(snaps, "next_day"))],
    )

    halts = [c for c in h.listener.changes if c[1] == GuardState.HALTED]
    assert halts and halts[0][2] == HaltReason.MAX_DAILY_LOSS
    halt_ns = halts[0][0]
    midnight_ns = int(datetime(2026, 1, 6, tzinfo=UTC).timestamp() * 1e9)
    assert halt_ns < midnight_ns

    resumes = [c for c in h.listener.changes if c[1] == GuardState.RUNNING]
    assert resumes, "daily-loss halt must lift at UTC rollover"
    assert resumes[0][0] >= midnight_ns
    assert "rollover" in resumes[0][3]

    filled = h.filled_entries()
    # First entry on day 1, nothing filled between halt and midnight, re-entry after.
    assert filled[0].ts_init < halt_ns
    assert [o for o in filled if halt_ns < o.ts_init < midnight_ns] == []
    assert any(o.ts_init >= midnight_ns for o in filled)
    # The strategy kept trying while halted -> those were gate denials.
    gate_denials = [e for e, by_gate in h.listener.denied if by_gate]
    assert gate_denials
    assert all(GATE_DENIAL_PREFIX in e.reason for e, _ in h.listener.denied)
    # Strategies stay running (warm buffers) but flat while daily-loss halted.
    assert snaps["halted"]["running"] == [True]
    assert snaps["halted"]["positions_open"] == 0
    assert snaps["halted"]["state"] == GuardState.HALTED
    assert snaps["next_day"]["state"] == GuardState.RUNNING
    assert snaps["next_day"]["positions_open"] == 1


def test_single_manual_halt_flattens_and_stops_everything() -> None:
    """One halt() call: no new orders, position flattened, strategy stopped."""
    h = build(strategy_kwargs={"mode": "when_flat", "sl_pct": 0.10, "tp_pct": 0.10})
    bt = get_bar_type("BTCUSDT", "1m")
    closes = [10000.0] * 30
    snaps: dict[str, Any] = {}
    run(
        h,
        _flat_bars(bt, closes),
        actions=[
            (h.at(10.5), h.snap(snaps, "before")),
            (h.at(10.5), lambda: h.guard.halt(HaltReason.MANUAL, "operator")),
            (h.at(13.5), h.snap(snaps, "after")),
        ],
    )

    assert snaps["before"]["positions_open"] == 1
    assert len(snaps["before"]["orders_open"]) == 2  # SL + TP live before the halt
    assert h.guard.guard_state == GuardState.HALTED
    assert snaps["after"] == {
        "running": [False],
        "positions_open": 0,
        "orders_open": [],
        "state": GuardState.HALTED,
    }
    halt_ns = h.halt_ns()
    assert h.accepted_entries_after(halt_ns) == []
    assert len(h.filled_entries()) == 1
    # Flatten was a reduce-only market order.
    closers = [o for o in h.orders() if o.is_reduce_only and o.status == OrderStatus.FILLED]
    assert len(closers) == 1 and closers[0].ts_init >= halt_ns


def test_pause_blocks_entries_but_keeps_sl_tp_working() -> None:
    """Pause: SL/TP stay live (TP later fills), no new entry fills while paused."""
    h = build(strategy_kwargs={"mode": "when_flat", "sl_pct": 0.05, "tp_pct": 0.02})
    bt = get_bar_type("BTCUSDT", "1m")
    # Entry @10000; TP @10200 hit at minute 12 (after the pause at 5.5).
    closes = [10000.0] * 12 + [10250.0] + [10250.0] * 10
    snaps: dict[str, Any] = {}
    run(
        h,
        _flat_bars(bt, closes),
        actions=[
            (h.at(5.5), lambda: h.guard.pause()),
            (h.at(8.5), h.snap(snaps, "paused")),
            (h.at(16.5), h.snap(snaps, "after_tp")),
        ],
    )

    assert h.guard.guard_state == GuardState.PAUSED
    paused = snaps["paused"]
    assert paused["positions_open"] == 1 and paused["running"] == [True]
    assert len(paused["orders_open"]) == 2 and all(ro for _, ro in paused["orders_open"])
    assert snaps["after_tp"]["positions_open"] == 0
    tp_fills = [o for o in h.orders() if o.is_reduce_only and o.status == OrderStatus.FILLED]
    assert len(tp_fills) == 1, "TP placed before the pause must still fill"
    pause_ns = int(h.at(5.5).timestamp() * 1e9)
    assert [o for o in h.filled_entries() if o.ts_init > pause_ns] == []
    assert any(by_gate for _, by_gate in h.listener.denied), "re-entry after TP must be denied"


def test_resume_after_pause_allows_entries_again() -> None:
    h = build(strategy_kwargs={"mode": "when_flat", "tp_pct": 0.01})
    bt = get_bar_type("BTCUSDT", "1m")
    closes = [10000.0] * 3 + [10150.0] + [10150.0] * 10
    run(
        h,
        _flat_bars(bt, closes),
        actions=[(h.at(1.5), lambda: h.guard.pause()), (h.at(8.5), lambda: h.guard.resume())],
    )
    resume_ns = int(h.at(8.5).timestamp() * 1e9)
    assert any(o.ts_init > resume_ns for o in h.filled_entries())


def test_close_all_closes_every_position_and_reports_them() -> None:
    h = build(
        symbols=("BTCUSDT", "ETHUSDT"),
        strategy_kwargs={"mode": "once", "sl_pct": 0.2, "trade_size": "0.100"},
    )
    btc = _flat_bars(get_bar_type("BTCUSDT", "1m"), [10000.0] * 10)
    eth = _flat_bars(get_bar_type("ETHUSDT", "1m"), [2000.0] * 10)
    result: dict[str, Any] = {}

    def close_all() -> None:
        result["r"] = h.guard.close_all()

    snaps: dict[str, Any] = {}
    run(
        h,
        btc + eth,
        actions=[
            (h.at(5.5), h.snap(snaps, "before")),
            (h.at(5.5), close_all),
            (h.at(7.5), h.snap(snaps, "after")),
        ],
    )

    r = result["r"]
    assert snaps["before"]["positions_open"] == 2
    assert len(r.targeted_positions) == 2
    assert r.errors == [] and r.unowned_positions == []
    assert h.engine.cache.positions_open() == []
    assert h.engine.cache.orders_open() == []
    assert len(h.engine.cache.positions_closed()) == 2
    # Strategies keep running after close-all (not a halt).
    assert snaps["after"]["running"] == [True, True]
    assert snaps["after"]["positions_open"] == 0
    assert h.guard.guard_state == GuardState.RUNNING


def test_gate_blocks_second_entry_on_existing_position() -> None:
    """A strategy that ignores its own state (restart, no sync) cannot double up."""
    h = build(strategy_kwargs={"mode": "every_bar", "trade_size": "0.500"})
    run(h, _flat_bars(get_bar_type("BTCUSDT", "1m"), [10000.0] * 10))
    positions = h.engine.cache.positions()
    assert len(positions) == 1
    assert positions[0].quantity == Quantity.from_str("0.500")
    assert len(h.filled_entries()) == 1
    assert h.strategies[0].entries_submitted == 10
    reasons = [e.reason for e, by_gate in h.listener.denied if by_gate]
    assert len(reasons) == 9
    assert all("already has net position" in r for r in reasons)


def test_gate_blocks_second_entry_while_first_is_pending() -> None:
    """Entry still in flight (latency) when the next signal fires -> denied."""
    h = build(
        strategy_kwargs={"mode": "every_bar", "trade_size": "0.500"}, latency_ns=90_000_000_000
    )
    run(h, _flat_bars(get_bar_type("BTCUSDT", "1m"), [10000.0] * 8))
    reasons = [e.reason for e, by_gate in h.listener.denied if by_gate]
    assert any("still pending" in r for r in reasons), reasons
    assert len(h.filled_entries()) == 1
    assert h.engine.cache.positions()[0].quantity == Quantity.from_str("0.500")


def test_max_position_count_enforced_across_instruments() -> None:
    h = build(
        symbols=("BTCUSDT", "ETHUSDT"),
        strategy_kwargs={"mode": "when_flat", "trade_size": "0.100"},
        limits={"max_position_count": 1},
    )
    btc = _flat_bars(get_bar_type("BTCUSDT", "1m"), [10000.0] * 5)
    eth = _flat_bars(get_bar_type("ETHUSDT", "1m"), [2000.0] * 5)
    run(h, btc + eth)
    assert len(h.engine.cache.positions_open()) == 1
    assert any("max_position_count" in e.reason for e, _ in h.listener.denied)


def test_consecutive_losses_halt() -> None:
    h = build(
        strategy_kwargs={"mode": "when_flat", "sl_pct": 0.005, "trade_size": "0.010"},
        limits={"max_consecutive_losses": 2, "max_drawdown_pct": Decimal("0.5")},
    )
    # Each 1% dip stops out the long (0.5% SL), then it re-enters.
    closes = [10000.0, 9900.0, 9900.0, 9800.0, 9800.0, 9700.0, 9700.0, 9600.0]
    run(h, _flat_bars(get_bar_type("BTCUSDT", "1m"), closes))
    assert h.guard.halt_reason == HaltReason.MAX_CONSECUTIVE_LOSSES
    assert len(h.listener.closed) == 2
    assert h.engine.cache.positions_open() == []


def test_resume_rules() -> None:
    h = build()
    g = h.guard
    out: dict[str, Any] = {}

    def scenario() -> None:
        g.halt(HaltReason.KILL_SWITCH, "kill")
        out["kill_blocked"] = g.resume(kill_switch_engaged=True)
        out["state_after_blocked"] = g.guard_state
        # A lower-priority halt never downgrades the kill.
        out["downgrade"] = g.halt(HaltReason.ERROR, "late error")
        out["reason_after_downgrade"] = g.halt_reason
        out["kill_cleared"] = g.resume(kill_switch_engaged=False)
        out["state_after_resume"] = g.guard_state
        g.halt(HaltReason.MAX_DRAWDOWN, "dd")
        out["dd_resume"] = g.resume()

    run(
        h,
        _flat_bars(get_bar_type("BTCUSDT", "1m"), [10000.0] * 6),
        actions=[(h.at(2.5), scenario)],
    )
    assert out["kill_blocked"][0] is False and "kill switch" in out["kill_blocked"][1]
    assert out["state_after_blocked"] == GuardState.HALTED
    assert out["downgrade"] is False
    assert out["reason_after_downgrade"] == HaltReason.KILL_SWITCH
    assert out["kill_cleared"][0] is True
    assert out["state_after_resume"] == GuardState.RUNNING
    assert out["dd_resume"][0] is False and "manual review" in out["dd_resume"][1]

    h2 = build()
    out2: dict[str, Any] = {}

    def daily() -> None:
        h2.guard.halt(HaltReason.MAX_DAILY_LOSS, "daily")
        out2["r"] = h2.guard.resume()

    run(
        h2,
        _flat_bars(get_bar_type("BTCUSDT", "1m"), [10000.0] * 4),
        actions=[(h2.at(1.5), daily)],
    )
    assert out2["r"][0] is False and "UTC day" in out2["r"][1]


def test_order_rejection_reaches_listener() -> None:
    """Venue rejection (reduce-only order with no position) is reported."""

    class RejectMe(NaiveStrategy):
        def on_bar(self, bar: Bar) -> None:
            if self.entered_once:
                return
            self.entered_once = True
            self.submit_order(
                self.order_factory.market(
                    instrument_id=self.config.instrument_id,
                    order_side=OrderSide.SELL,
                    quantity=self.instrument.make_qty(Decimal("0.100")),
                    reduce_only=True,
                )
            )

    engine = _new_engine()
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(10_000, Currency.from_str("USDT"))],
        default_leverage=Decimal("10"),
        bar_execution=True,
    )
    engine.add_instrument(create_instrument("BTCUSDT"))
    strat = RejectMe(
        NaiveConfig(
            instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
            bar_type=get_bar_type("BTCUSDT", "1m"),
        )
    )
    listener = RecordingListener()
    guard = TradingGuard(
        TradingGuardConfig(check_interval_secs=60),
        strategies=[strat],
        risk_engine=engine.kernel.risk_engine,
        listener=listener,
    )
    engine.add_actor(guard)
    engine.add_strategy(strat)
    engine.add_data(_flat_bars(get_bar_type("BTCUSDT", "1m"), [10000.0] * 3))
    engine.run()
    assert len(listener.rejected) == 1
    assert "reduce" in listener.rejected[0].reason.lower()


def test_risk_state_restored_from_checkpoint_keeps_drawdown_limit() -> None:
    """A restart must not reset the high water mark (DD measured from pre-restart peak)."""
    restored = RiskState.from_dict(
        RiskState(high_water_mark=Decimal("10600"), consecutive_losses=0).to_dict()
    )
    h = build(limits={"max_drawdown_pct": Decimal("0.05")}, risk_state=restored)
    run(h, _flat_bars(get_bar_type("BTCUSDT", "1m"), [10000.0] * 3))
    # 10000 equity vs 10600 HWM = 5.66% drawdown -> halted before any trade fills.
    assert h.guard.halt_reason == HaltReason.MAX_DRAWDOWN
    assert h.filled_entries() == []


def test_compiled_dsl_strategy_dd_halt_flattens_and_stops() -> None:
    """Same DD scenario with a real StrategyCompiler strategy (entry always true)."""
    import json

    from vibe_quant.dsl.compiler import StrategyCompiler
    from vibe_quant.dsl.parser import validate_strategy_dict

    dsl = validate_strategy_dict(
        {
            "name": "audit_guard_probe",
            "timeframe": "1m",
            "indicators": {"sma_2": {"type": "SMA", "period": 2}},
            "entry_conditions": {"long": ["sma_2 > 0"]},
            "stop_loss": {"type": "fixed_pct", "percent": 20.0},
            "take_profit": {"type": "fixed_pct", "percent": 50.0},
        }
    )
    module = StrategyCompiler().compile_to_module(dsl)
    cfg = module.AuditGuardProbeConfig.parse(
        json.dumps(
            {
                "instrument_id": "BTCUSDT-PERP.BINANCE",
                "order_id_tag": "000",
                "external_order_claims": ["BTCUSDT-PERP.BINANCE"],
                "max_position_pct": 2.0,
                "risk_per_trade": 0.5,
            }
        )
    )
    strat = module.AuditGuardProbeStrategy(config=cfg)
    engine = _new_engine()
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(10_000, Currency.from_str("USDT"))],
        default_leverage=Decimal("10"),
        bar_execution=True,
    )
    engine.add_instrument(create_instrument("BTCUSDT"))
    listener = RecordingListener()
    guard = TradingGuard(
        TradingGuardConfig(
            check_interval_secs=60,
            max_drawdown_pct=Decimal("0.05"),
            max_daily_loss_pct=Decimal("0.5"),
        ),
        strategies=[strat],
        risk_engine=engine.kernel.risk_engine,
        listener=listener,
    )
    listener.clock_ns = lambda: guard.clock.timestamp_ns()
    engine.add_actor(guard)
    engine.add_strategy(strat)
    h = Harness(engine=engine, guard=guard, strategies=[strat], listener=listener)
    closes = [10000.0] * 5 + [9900.0, 9800.0, 9700.0, 9600.0, 9500.0] + [9400.0] * 12
    snaps: dict[str, Any] = {}
    run(
        h,
        _flat_bars(get_bar_type("BTCUSDT", "1m"), closes),
        actions=[(h.at(19.5), h.snap(snaps, "after"))],
    )

    assert guard.halt_reason == HaltReason.MAX_DRAWDOWN
    assert snaps["after"]["running"] == [False]
    assert snaps["after"]["positions_open"] == 0
    assert snaps["after"]["orders_open"] == []
    assert h.accepted_entries_after(h.halt_ns()) == []
    exits = [o for o in h.orders() if o.tags and "MARKET_EXIT" in o.tags]
    assert exits and all(o.is_reduce_only for o in exits)
