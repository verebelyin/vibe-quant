"""Paper/live margin limit in the TradingGuard gate (bead vibe-quant-9r0m5, rows R27-R35c).

Guard-level rows run the real ``TradingGuard._denial_reason`` against fake cache/portfolio/
account objects (the account's ``leverage`` is the venue-VERIFIED value, DR1); R35c runs the
real NT BacktestEngine with strategies that submit entries in ``on_start``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from nautilus_trader.model.enums import OrderSide, PriceType
from nautilus_trader.model.identifiers import InstrumentId

from tests.unit.test_audit_paper_node import fake_tradingnode  # noqa: F401
from tests.unit.test_paper_halt_restore import _trace_node
from tests.unit.test_paper_halt_restore import seeded as _restore_seeded
from vibe_quant.db.state_manager import StateManager
from vibe_quant.logging.events import EventType
from vibe_quant.paper.config import SizingModuleConfig, effective_max_margin_pct
from vibe_quant.paper.guard import (
    GATE_DENIAL_PREFIX,
    GuardListener,
    GuardState,
    HaltReason,
    TradingGuard,
    TradingGuardConfig,
)
from vibe_quant.paper.node import PaperTradingNode, _GuardBridge

if TYPE_CHECKING:
    from pathlib import Path

BTC = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
ETH = InstrumentId.from_str("ETHUSDT-PERP.BINANCE")
SOL = InstrumentId.from_str("SOLUSDT-PERP.BINANCE")
PREFIX = f"{GATE_DENIAL_PREFIX}: margin limit: "


class _Qty:
    def __init__(self, value: str) -> None:
        self._v = Decimal(value)

    def as_decimal(self) -> Decimal:
        return self._v

    def __str__(self) -> str:
        return str(self._v)


class _Account:
    def __init__(self, leverages: dict[InstrumentId, Decimal | None]) -> None:
        self._l = leverages

    def leverage(self, instrument_id: InstrumentId) -> Decimal | None:
        return self._l.get(instrument_id)


@dataclass
class _Order:
    client_order_id: str
    instrument_id: InstrumentId
    qty: str = "1"
    leaves: str | None = None
    price: Any = None  # None -> attribute absent (market order)
    reduce_only: bool = False
    side: OrderSide = OrderSide.BUY
    is_closed: bool = False

    def __post_init__(self) -> None:
        if self.price is None:
            del self.price
        else:
            self.price = Decimal(str(self.price))

    @property
    def is_reduce_only(self) -> bool:
        return self.reduce_only

    @property
    def quantity(self) -> _Qty:
        return _Qty(self.qty)

    @property
    def leaves_qty(self) -> _Qty:
        return _Qty(self.leaves if self.leaves is not None else self.qty)


@dataclass
class _Position:
    instrument_id: InstrumentId
    quantity: str
    avg_px_open: float


@dataclass
class _Cache:
    positions: list[Any] = field(default_factory=list)
    open_orders: list[Any] = field(default_factory=list)
    inflight: list[Any] = field(default_factory=list)
    initialized: list[Any] = field(default_factory=list)
    prices: dict[tuple[InstrumentId, PriceType], Decimal] = field(default_factory=dict)

    def positions_open(self, *a: Any, **k: Any) -> list[Any]:
        return list(self.positions)

    def orders(self, *a: Any, **k: Any) -> list[Any]:
        iid = k.get("instrument_id")
        seen: dict[str, Any] = {}
        for o in (*self.open_orders, *self.inflight, *self.initialized):
            seen.setdefault(o.client_order_id, o)
        return [o for o in seen.values() if iid is None or o.instrument_id == iid]

    def orders_open(self, *a: Any, **k: Any) -> list[Any]:
        iid = k.get("instrument_id")
        return [o for o in self.open_orders if iid is None or o.instrument_id == iid]

    def orders_inflight(self, *a: Any, **k: Any) -> list[Any]:
        iid = k.get("instrument_id")
        return [o for o in self.inflight if iid is None or o.instrument_id == iid]

    def order_exists(self, *a: Any, **k: Any) -> bool:
        return False

    def instrument(self, iid: Any) -> Any:
        return SimpleNamespace(multiplier=Decimal(1))

    def price(self, iid: InstrumentId, price_type: PriceType) -> Decimal | None:
        return self.prices.get((iid, price_type))


class _Money:
    def __init__(self, v: str) -> None:
        self._v = Decimal(v)

    def as_decimal(self) -> Decimal:
        return self._v


class _Portfolio:
    def __init__(self, account: Any, equity: str | None) -> None:
        self._account = account
        self._equity = equity

    def account(self, venue: Any) -> Any:
        return self._account

    def missing_price_instruments(self, venue: Any) -> list[Any]:
        return []

    def equity(self, venue: Any) -> dict[Any, Any]:
        from nautilus_trader.model.objects import Currency

        if self._equity is None:
            return {}
        return {Currency.from_str("USDT"): _Money(self._equity)}

    def net_position(self, iid: Any) -> Decimal:
        return Decimal(0)


class _Clock:
    def utc_now(self) -> datetime:
        return datetime(2026, 1, 5, 22, 0, tzinfo=UTC)

    def timestamp_ns(self) -> int:
        return 0

    def set_timer(self, name: str, interval: timedelta, callback: Any) -> None:
        return None


class _Listener(GuardListener):
    def __init__(self) -> None:
        self.allowed: list[str] = []

    def on_entry_allowed(self, symbol: str) -> None:
        self.allowed.append(symbol)


class _UnitGuard(TradingGuard):
    def __init__(self, *args: Any, cache: Any, portfolio: Any, **kwargs: Any) -> None:
        self._c = cache
        self._p = portfolio
        super().__init__(*args, **kwargs)

    @property
    def cache(self) -> Any:  # type: ignore[override]
        return self._c

    @property
    def portfolio(self) -> Any:  # type: ignore[override]
        return self._p

    @property
    def clock(self) -> Any:  # type: ignore[override]
        return _Clock()


def _guard(
    cache: _Cache,
    leverages: dict[InstrumentId, Decimal | None],
    *,
    equity: str | None = "1000",
    cushion: float = 0.5,
    account: bool = True,
    listener: GuardListener | None = None,
) -> _UnitGuard:
    acct = _Account(leverages) if account else None
    return _UnitGuard(
        TradingGuardConfig(max_margin_pct=cushion),
        strategies=[],
        risk_engine=SimpleNamespace(execute=lambda c: None),
        listener=listener,
        cache=cache,
        portfolio=_Portfolio(acct, equity),
    )


L10 = Decimal(10)


def test_reduce_only_never_margin_checked() -> None:
    """R27: a reduce-only exit passes even when the book is far over the limit."""
    cache = _Cache(positions=[_Position(BTC, "10", 100000.0)])
    g = _guard(cache, {BTC: L10, ETH: L10})
    exit_order = _Order("O-1", ETH, qty="50", price=100000, reduce_only=True)
    assert g._denial_reason(exit_order) is None


def test_state_denial_precedes_margin() -> None:
    """R28: HALTED + over the limit -> the halted reason wins."""
    cache = _Cache(positions=[_Position(BTC, "10", 100000.0)])
    g = _guard(cache, {BTC: L10, ETH: L10})
    g._state = GuardState.HALTED
    g._halt_reason = HaltReason.MANUAL
    reason = g._denial_reason(_Order("O-1", ETH, qty="50", price=100000))
    assert reason is not None
    assert "trading halted (manual)" in reason
    assert "margin limit" not in reason


def test_counts_all_account_positions() -> None:
    """R29: 2 guarded + 1 unguarded positions are all counted, each at its own leverage."""
    cache = _Cache(
        positions=[
            _Position(BTC, "1", 10000.0),  # 10000 @10 -> 1000
            _Position(ETH, "10", 1000.0),  # 10000 @5 -> 2000
            _Position(SOL, "60", 100.0),  # 6000 @2 -> 3000 (unguarded)
        ]
    )
    bnb = InstrumentId.from_str("BNBUSDT-PERP.BINANCE")
    g = _guard(
        cache,
        {BTC: L10, ETH: Decimal(5), SOL: Decimal(2), bnb: L10},
        equity="10000",
    )
    reason = g._denial_reason(_Order("O-9", bnb, qty="1", price=1000))
    assert reason == (PREFIX + "margin in use 6000 + new 100 > limit 5000.0 (50% of equity 10000)")


def test_uses_verified_leverage() -> None:
    """R29b: requested 10x but the venue verified 5x on BTC -> in_use 800, not 400."""
    cache = _Cache(positions=[_Position(BTC, "0.04", 100000.0)])  # 4000
    g = _guard(cache, {BTC: Decimal(5), ETH: L10}, equity="1000")
    reason = g._denial_reason(_Order("O-1", ETH, qty="1", price=1000))
    assert reason == PREFIX + "margin in use 800 + new 100 > limit 500.0 (50% of equity 1000)"


def test_unverified_symbol_refused() -> None:
    """R29c: unverified ETH leverage refuses ETH; a verified BTC entry is checked normally."""
    g = _guard(_Cache(), {BTC: L10, ETH: None}, equity="1000")
    assert g._denial_reason(_Order("O-1", ETH, qty="1", price=100)) == (
        PREFIX + "leverage unverified for ETHUSDT-PERP.BINANCE"
    )
    assert g._denial_reason(_Order("O-2", BTC, qty="1", price=100)) is None


def test_partial_fill_counts_leaves_qty() -> None:
    """R30: position 0.4 + an open entry with 0.6 left at its limit price."""
    cache = _Cache(
        positions=[_Position(BTC, "0.4", 20000.0)],  # 8000 -> 800
        open_orders=[_Order("P-1", BTC, qty="1", leaves="0.6", price=21000)],  # 12600 -> 1260
    )
    g = _guard(cache, {BTC: L10, ETH: L10}, equity="4000")
    reason = g._denial_reason(_Order("O-1", ETH, qty="1", price=1000))
    assert reason == PREFIX + "margin in use 2060 + new 100 > limit 2000.0 (50% of equity 4000)"


def test_pending_cancel_counted_once() -> None:
    """R30b: a PENDING_CANCEL entry is in orders_open AND
    orders_inflight; it is counted once (200 + 200 = 400 <= 500), not at 600."""
    pending = _Order("P-1", BTC, qty="1", price=2000)
    cache = _Cache(open_orders=[pending], inflight=[pending])
    g = _guard(cache, {BTC: L10, ETH: L10}, equity="1000")
    assert g._denial_reason(_Order("O-1", ETH, qty="1", price=2000)) is None


def test_pending_update_counted_once() -> None:
    """R30c: the PENDING_UPDATE twin (same union rule), ending exactly on the limit."""
    pending = _Order("P-1", BTC, qty="1", price=2500)  # 250
    cache = _Cache(open_orders=[pending], inflight=[pending])
    g = _guard(cache, {BTC: L10, ETH: L10}, equity="1000")
    # 250 + 250 == 500 == limit: equality is allowed
    assert g._denial_reason(_Order("O-1", ETH, qty="1", price=2500)) is None
    over = g._denial_reason(_Order("O-2", ETH, qty="1", price=2500.1))
    assert over is not None and over.startswith(PREFIX + "margin in use 250 + new")


def test_unpriced_pending_denies() -> None:
    """R31: a pending MARKET entry with no LAST/MID denies the new entry."""
    cache = _Cache(open_orders=[_Order("P-1", ETH, qty="1")])  # market: no price
    g = _guard(cache, {BTC: L10, ETH: L10}, equity="1000")
    assert g._denial_reason(_Order("O-1", BTC, qty="1", price=100)) == (
        PREFIX + "no reference price for ETHUSDT-PERP.BINANCE"
    )
    cache.prices[(ETH, PriceType.MID)] = Decimal(100)
    assert g._denial_reason(_Order("O-2", BTC, qty="1", price=100)) is None


def test_reference_price_order_is_price_trigger_last_mid() -> None:
    cache = _Cache(open_orders=[_Order("P-1", ETH, qty="1")])
    cache.prices[(ETH, PriceType.LAST)] = Decimal(4000)  # 400 margin
    cache.prices[(ETH, PriceType.MID)] = Decimal(1)
    g = _guard(cache, {BTC: L10, ETH: L10}, equity="1000")
    reason = g._denial_reason(_Order("O-1", BTC, qty="1", price=1500))  # 150 -> 550 > 500
    assert reason is not None and "margin in use 400 + new 150" in reason


def test_no_account_denies_entry() -> None:
    """R32: no account -> equity unknown; an account without USDT also refuses."""
    g = _guard(_Cache(), {BTC: L10}, account=False)
    assert g._denial_reason(_Order("O-1", BTC, price=100)) == PREFIX + "equity unknown"
    g2 = _guard(_Cache(), {BTC: L10}, equity=None)
    assert g2._denial_reason(_Order("O-2", BTC, price=100)) == PREFIX + "equity unknown"


def test_unverified_leverage_outranks_unknown_equity_when_account_exists() -> None:
    """Shared-rule order (R21c): with an account, an unverified leverage is reported before
    an unknown equity (only a missing ACCOUNT short-circuits to equity unknown)."""
    g = _guard(_Cache(), {BTC: None}, equity=None)
    assert g._denial_reason(_Order("O-1", BTC, price=100)) == (
        PREFIX + "leverage unverified for BTCUSDT-PERP.BINANCE"
    )


def test_equality_edge_allowed_and_over_denied() -> None:
    g = _guard(_Cache(), {BTC: L10}, equity="1000")
    assert g._denial_reason(_Order("O-1", BTC, qty="1", price=5000)) is None  # 500 == 500
    assert g._denial_reason(_Order("O-2", BTC, qty="1", price=5000.1)) is not None


def test_entry_allowed_callback_only_when_margin_passes() -> None:
    listener = _Listener()
    g = _guard(_Cache(), {BTC: L10}, equity="1000", listener=listener)
    assert g._denial_reason(_Order("O-1", BTC, qty="1", price=5000.1)) is not None
    assert listener.allowed == []
    assert g._denial_reason(_Order("O-2", BTC, qty="1", price=5000)) is None
    assert listener.allowed == ["BTCUSDT-PERP.BINANCE"]


def test_effective_cushion_source() -> None:
    """R33: none/none -> 0.5; run 0.3 -> 0.3; override 0.2 wins."""
    assert effective_max_margin_pct(None, None) == Decimal("0.5")
    assert effective_max_margin_pct({}, None) == Decimal("0.5")
    assert effective_max_margin_pct({"max_margin_pct": 0.3}, None) == Decimal("0.3")
    assert effective_max_margin_pct({"max_margin_pct": 0.3}, Decimal("0.2")) == Decimal("0.2")


# ------------------------------------------------------------------ node: R34 / R35 / R35b


@pytest.fixture()
def seeded(tmp_path: Path) -> dict[str, Any]:
    return _restore_seeded.__wrapped__(tmp_path)  # type: ignore[attr-defined,no-any-return]


def _denied(symbol: str, reason: str) -> Any:
    return SimpleNamespace(
        client_order_id="C-1",
        instrument_id=symbol,
        strategy_id="S-1",
        reason=reason,
        ts_init=1_700_000_000_000_000_000,
    )


def test_margin_alert_once_per_symbol(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R34: ETH denial, ETH denial, ETH allowed, ETH denial -> alerts 1, 0, -, 1 (BTC separate)."""
    node, trace = _trace_node(monkeypatch, seeded)
    bridge = _GuardBridge(node)
    eth, btc = "ETHUSDT-PERP.BINANCE", "BTCUSDT-PERP.BINANCE"
    reason = PREFIX + "margin in use 2060 + new 100 > limit 2000.0 (50% of equity 4000)"

    def alerts() -> list[Any]:
        return [c for c in trace.calls if c[0] == "alert"]

    node._on_order_denied(_denied(eth, reason), by_gate=True)
    assert alerts() == [
        ("alert", "warning", f"Entry refused (margin limit): {reason}\n{eth}", False)
    ]
    assert trace.calls[0][:2] == ("write", EventType.ORDER)
    assert trace.calls[0][2]["by_guard"] is True and trace.calls[0][2]["reason"] == reason

    node._on_order_denied(_denied(eth, reason), by_gate=True)
    assert len(alerts()) == 1, "the second refusal of the same symbol does not alert"
    assert len([c for c in trace.calls if c[0] == "write"]) == 2, "but it is still written"

    node._on_order_denied(_denied(btc, reason), by_gate=True)
    assert len(alerts()) == 2, "another symbol has its own latch"

    bridge.on_entry_allowed(eth)
    node._on_order_denied(_denied(eth, reason), by_gate=True)
    assert len(alerts()) == 3, "an allowed entry re-arms the symbol"


def test_non_margin_gate_denial_still_does_not_alert(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    node, trace = _trace_node(monkeypatch, seeded)
    node._on_order_denied(
        _denied("ETHUSDT-PERP.BINANCE", f"{GATE_DENIAL_PREFIX}: trading halted (manual)"),
        by_gate=True,
    )
    assert [c[0] for c in trace.calls] == ["write"]


def _module_with_fields(node: PaperTradingNode, *fields: str) -> Any:
    dsl = node._load_strategy()
    _, config_name = node._class_names(dsl)
    cfg_cls = type(config_name, (), {"__struct_fields__": fields})
    return dsl, SimpleNamespace(**{config_name: cfg_cls})


def test_paper_strips_backtest_margin_params(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R35: a run recorded with max_margin_pct 0.3 + margin_leverage reaches neither the
    resolved strategy params nor the strategy config, even if the config declares the fields;
    the cushion itself still resolves to 0.3."""
    sm = StateManager(seeded["db"])
    rid = sm.create_backtest_run(
        strategy_id=seeded["sid"],
        run_mode="validation",
        symbols=["BTCUSDT"],
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-12-31",
        parameters={
            "rsi_14_period": 21,
            "leverage": 5,
            "max_margin_pct": 0.3,
            "margin_leverage": 5.0,
        },
    )
    sm.update_backtest_run_status(rid, "completed")
    sm.close()
    node, _trace = _trace_node(monkeypatch, seeded, validation_run_id=rid)
    dsl, module = _module_with_fields(
        node, "rsi_14_period", "take_profit_risk_reward", "max_margin_pct", "margin_leverage"
    )
    resolved = node._resolve_run_config(module, dsl)
    assert resolved.strategy_params == {"rsi_14_period": 21}
    assert "max_margin_pct" not in resolved.strategy_params
    assert "margin_leverage" not in resolved.strategy_params
    assert resolved.max_margin_pct == Decimal("0.3")


async def test_start_warning_not_refusal(
    seeded: dict[str, Any],
    fake_tradingnode: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R35b: 3 symbols x cap 0.5 / lev 1 = 150% > cushion 50% -> warning log + event, the
    start proceeds and the start event shows the cushion; 1 symbol (== 50%) is silent."""
    node, trace = _trace_node(
        monkeypatch,
        seeded,
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        leverage=1,
        sizing=SizingModuleConfig(max_position_pct=Decimal("0.5")),
    )
    with caplog.at_level(logging.WARNING):
        await node._initialize()
        try:
            await node._run_loop()
        finally:
            await node._shutdown()
    text = "max possible margin 150% exceeds cushion 50%; entries beyond it will be refused"
    assert text in caplog.text
    warns = [
        c for c in trace.calls if c[0] == "write" and c[2].get("action") == "margin_cushion_warning"
    ]
    assert len(warns) == 1 and warns[0][2]["message"] == text
    start = next(c for c in trace.calls if c[0] == "write" and c[2].get("action") == "start")
    assert start[2]["max_margin_pct"] == 0.5
    assert node.status.state.value != "error"

    node2, trace2 = _trace_node(
        monkeypatch,
        seeded,
        symbols=["BTCUSDT"],
        leverage=1,
        sizing=SizingModuleConfig(max_position_pct=Decimal("0.5")),
    )
    await node2._initialize()
    try:
        await node2._run_loop()
    finally:
        await node2._shutdown()
    assert not [
        c
        for c in trace2.calls
        if c[0] == "write" and c[2].get("action") == "margin_cushion_warning"
    ]


# ------------------------------------------------------------------------ R35c (real NT)

from fractions import Fraction  # noqa: E402

from nautilus_trader.model.enums import AccountType, OmsType, OrderStatus  # noqa: E402
from nautilus_trader.model.identifiers import Venue  # noqa: E402
from nautilus_trader.model.objects import Currency, Money  # noqa: E402
from nautilus_trader.trading.strategy import Strategy, StrategyConfig  # noqa: E402

import tests.unit.test_audit_paper_guard as harness  # noqa: E402
from vibe_quant.data.catalog import create_instrument, get_bar_type  # noqa: E402


class _OnStartConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    qty: str
    price: str


class _OnStartEntry(Strategy):
    """Submits ONE resting limit entry in ``on_start`` (before any price exists)."""

    def __init__(self, config: _OnStartConfig) -> None:
        super().__init__(config)
        self.submitted = 0

    def on_start(self) -> None:
        inst = self.cache.instrument(self.config.instrument_id)
        order = self.order_factory.limit(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY,
            quantity=inst.make_qty(Decimal(self.config.qty)),
            price=inst.make_price(Decimal(self.config.price)),
        )
        self.submitted += 1
        self.submit_order(order)


def test_on_start_entries_gated() -> None:
    """R35c: 3 strategies submit resting entries in on_start (margins 2000, 2000, 1100
    at 10x on 10000 equity, cushion 50% = 5000): the third takes the sum to 5100. Counts are measured on the real
    NT flow: denied == over-limit submitted, and no over-limit entry reaches execution."""
    entries = [
        ("BTCUSDT", "1", "20000", 30000.0),  # 20000 notional -> 2000
        ("ETHUSDT", "10", "2000", 3000.0),  # 20000 -> 2000
        ("BNBUSDT", "10", "1100", 1500.0),  # 11000 -> 1100: 5100 > 5000
    ]
    engine = harness._new_engine()
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(10_000, Currency.from_str("USDT"))],
        default_leverage=Decimal("10"),
        bar_execution=True,
    )
    strategies: list[_OnStartEntry] = []
    ids: list[InstrumentId] = []
    for idx, (symbol, qty, price, _mkt) in enumerate(entries):
        engine.add_instrument(create_instrument(symbol))
        iid = InstrumentId.from_str(f"{symbol}-PERP.BINANCE")
        ids.append(iid)
        strategies.append(
            _OnStartEntry(
                _OnStartConfig(instrument_id=iid, qty=qty, price=price, order_id_tag=f"{idx:03d}")
            )
        )
    listener = harness.RecordingListener()
    guard = TradingGuard(
        TradingGuardConfig(check_interval_secs=60, max_margin_pct=0.5),
        strategies=strategies,
        risk_engine=engine.kernel.risk_engine,
        listener=listener,
    )
    engine.add_actor(harness.VerifiedLeverage(ids, Decimal(10)))
    engine.add_actor(guard)
    for s in strategies:
        engine.add_strategy(s)
    for symbol, _qty, _price, mkt in entries[:2]:  # BNB needs no bars: it never matches
        engine.add_data(harness._flat_bars(get_bar_type(symbol, "1m"), [mkt] * 3))
    engine.run()

    limit = Fraction(1, 2) * 10_000
    running = Fraction(0)
    expected_over: list[str] = []
    for symbol, qty, price, _mkt in entries:
        margin = Fraction(Decimal(qty) * Decimal(price)) / 10
        if running + margin > limit:
            expected_over.append(symbol)
        else:
            running += margin
    assert expected_over == ["BNBUSDT"]

    orders = {str(o.instrument_id): o for o in engine.cache.orders()}
    assert sum(s.submitted for s in strategies) == 3
    denied = [k for k, o in orders.items() if o.status == OrderStatus.DENIED]
    reached = [k for k, o in orders.items() if o.status != OrderStatus.DENIED]
    assert denied == [f"{symbol}-PERP.BINANCE" for symbol in expected_over]
    assert len(reached) == 2
    assert [by_gate for _e, by_gate in listener.denied] == [True]
    assert str(listener.denied[0][0].reason).startswith(f"{GATE_DENIAL_PREFIX}: margin limit: ")
    engine.dispose()


# ------------------------------------------------------- R30d: gate-admitted INITIALIZED entries


def test_initialized_entry_is_pending_margin() -> None:
    """R30d: an admitted entry that is in neither orders_open nor orders_inflight (live
    INITIALIZED) still counts: 400 + 200 = 600 > 500 (equity 1000, 10x, cushion 0.5)."""
    admitted = _Order("P-1", BTC, qty="0.1", price=40000)
    g = _guard(_Cache(initialized=[admitted]), {BTC: L10, ETH: L10}, equity="1000")
    reason = g._denial_reason(_Order("O-1", ETH, qty="1", price=2000))
    assert reason == PREFIX + "margin in use 400 + new 200 > limit 500.0 (50% of equity 1000)"
    # the equality edge: 400 + 100 == 500 is allowed
    assert g._denial_reason(_Order("O-2", ETH, qty="1", price=1000)) is None


def test_closed_and_reduce_only_orders_are_not_pending_margin() -> None:
    closed = _Order("P-1", BTC, qty="0.1", price=40000, is_closed=True)
    exit_order = _Order("P-2", BTC, qty="0.1", price=40000, reduce_only=True)
    g = _guard(_Cache(initialized=[closed, exit_order]), {BTC: L10, ETH: L10}, equity="1000")
    assert g._denial_reason(_Order("O-1", ETH, qty="1", price=2000)) is None


def test_initialized_entry_counts_as_pending_for_the_instrument() -> None:
    """R30d: the single-pending-entry check sees an INITIALIZED sibling too."""
    admitted = _Order("P-1", BTC, qty="0.1", price=1000)
    g = _guard(_Cache(initialized=[admitted]), {BTC: L10}, equity="100000")
    reason = g._denial_reason(_Order("O-1", BTC, qty="0.1", price=1000))
    assert reason is not None and "entry order P-1 for BTCUSDT-PERP.BINANCE still pending" in reason


# ----------------------------------------------- R29d: verified-leverage bounds [1, 125]


@pytest.mark.parametrize(
    ("leverage", "allowed"),
    [
        (Decimal(1), True),
        (Decimal(125), True),
        (Decimal("0.5"), False),
        (Decimal(126), False),
    ],
)
def test_verified_leverage_bounds(leverage: Decimal, allowed: bool) -> None:
    """R29d: exactly 1x and 125x are verified; 0.5x and 126x are as good as unverified."""
    g = _guard(_Cache(), {BTC: leverage}, equity="1000000")
    reason = g._denial_reason(_Order("O-1", BTC, qty="1", price=100))
    if allowed:
        assert reason is None
    else:
        assert reason == PREFIX + "leverage unverified for BTCUSDT-PERP.BINANCE"
