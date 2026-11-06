"""Post-only maker entries in VALIDATION runs.

Three layers:

* per-leg fees from the actual ``OrderFilled`` commissions (maker entry +
  taker stop), with the market-mode per-trade TOTAL preserved bit-for-bit;
* ``maker_entry_stats`` counted from the engine cache after a post-only run;
* the runner wiring: post-only params reach the generated config, a market-mode
  TWIN is run for the notes (never stored), bad maker params fail the run
  before the engine starts.

The engine-level tests run the real generated strategy on synthetic bars; the
runner-level tests drive ``ValidationRunner.run`` with ``_run_backtest``
stubbed, because the twin/notes/storage wiring is what they assert.
"""

from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import (
    AccountType,
    LiquiditySide,
    OmsType,
    OrderSide,
    OrderStatus,
    OrderType,
)
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Currency, Money, Price, Quantity

from tests.unit.fake_engine import fake_slippage
from vibe_quant.data.catalog import create_instrument
from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl.compiler import StrategyCompiler, _to_class_name
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.nt_compat import retain_log_guard
from vibe_quant.validation.extraction import all_positions as _all_positions
from vibe_quant.validation.extraction import extract_trades, maker_entry_stats
from vibe_quant.validation.results import TradeRecord, ValidationResult
from vibe_quant.validation.runner import ValidationRunner, ValidationRunnerError
from vibe_quant.validation.venue import create_venue_config_for_validation

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import Any

_S = 1_000_000_000
_M = 60 * _S
_M5 = 5 * _M
_VOL = "100000.000"
_TICK = 0.1
_BTC = "BTCUSDT-PERP.BINANCE"


# ---------------------------------------------------------------------------
# Synthetic engine harness (real generated strategy, real NT fee model)
# ---------------------------------------------------------------------------


@pytest.fixture
def engines() -> Iterator[list[BacktestEngine]]:
    created: list[BacktestEngine] = []
    yield created
    for engine in created:
        engine.dispose()


def _bar(
    spec: str, step: int, i: int, o: float, h: float, low: float, c: float
) -> Bar:
    """Bar ``i`` of length ``step``; ts_init = bar close (like the catalog)."""
    return Bar(
        bar_type=BarType.from_str(f"{_BTC}-{spec}-LAST-EXTERNAL"),
        open=Price.from_str(f"{o:.1f}"),
        high=Price.from_str(f"{h:.1f}"),
        low=Price.from_str(f"{low:.1f}"),
        close=Price.from_str(f"{c:.1f}"),
        volume=Quantity.from_str(_VOL),
        ts_event=i * step,
        ts_init=(i + 1) * step,
    )


def _strategy_bars(closes: list[float]) -> list[Bar]:
    return [_bar("5-MINUTE", _M5, i, c, c, c, c) for i, c in enumerate(closes)]


def _one_minute(n: int, *, ohlc: dict[int, tuple[float, float, float, float]] | None = None) -> list[Bar]:
    ohlc = ohlc or {}
    return [_bar("1-MINUTE", _M, i, *ohlc.get(i, (100.5,) * 4)) for i in range(n)]


def _first_release(signal_bar: int) -> int:
    """First 1m bar whose ts_init is strictly after 5m bar ``signal_bar`` closes."""
    return 5 * (signal_bar + 1)


def _engine() -> BacktestEngine:
    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    retain_log_guard(engine)
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(1_000_000, Currency.from_str("USDT"))],
        default_leverage=Decimal("10"),
        fill_model=FillModel(prob_fill_on_limit=1.0, prob_slippage=0.0, random_seed=42),
        bar_execution=True,
        use_reduce_only=True,
    )
    engine.add_instrument(create_instrument("BTCUSDT"))
    return engine


def _dsl(name: str, **overrides: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "name": name,
        "timeframe": "5m",
        "indicators": {"sma": {"type": "SMA", "period": 2}},
        "entry_conditions": {"long": ["close > 100.5"]},
        "exit_conditions": {"long": ["close < 50"]},
        "stop_loss": {"type": "fixed_pct", "percent": 1.0},
        "take_profit": {"type": "fixed_pct", "percent": 20.0},
    }
    d.update(overrides)
    return d


def _run(
    engines: list[BacktestEngine],
    *,
    data: list[Any],
    dsl: dict[str, Any],
    **config: Any,
) -> tuple[BacktestEngine, Any]:
    module = StrategyCompiler().compile_to_module(validate_strategy_dict(dsl))
    cn = _to_class_name(dsl["name"])
    strategy_cls = getattr(module, f"{cn}Strategy")
    config_cls = getattr(module, f"{cn}Config")
    engine = _engine()
    engines.append(engine)
    engine.add_data(data)
    kw: dict[str, Any] = {
        "instrument_id": _BTC,
        "command_release": "bar",
        "command_release_bar_type": f"{_BTC}-1-MINUTE-LAST-EXTERNAL",
        "entry_order": "post_only",
    }
    kw.update(config)
    strat = strategy_cls(config_cls(**kw))
    engine.add_strategy(strat)
    engine.run()
    return engine, strat


def _closed_position(engine: BacktestEngine) -> Any:
    closed = [p for p in _all_positions(engine) if p.is_closed]
    assert len(closed) == 1, f"expected one closed position, got {len(closed)}"
    return closed[0]


def _fills(pos: Any, side: OrderSide) -> list[Any]:
    return [e for e in pos.events if type(e).__name__ == "OrderFilled" and e.order_side == side]


def _extract(engine: BacktestEngine, name: str) -> ValidationResult:
    venue = create_venue_config_for_validation()
    result = ValidationResult(
        run_id=1, strategy_name=name, starting_balance=float(venue.starting_balance_usdt)
    )
    extract_trades(
        result,
        engine,
        venue,
        slippage=fake_slippage(venue.fill_config.prob_slippage),
        cost_ledger=None,
    )
    return result


# Long signal on 5m bar 2 (close 101 > 100.5; SMA(2) ready); the rest sit at 100.
_SIGNAL_CLOSES = [100.0, 100.0, 101.0] + [100.0] * 6
_REL = _first_release(2)


# ---------------------------------------------------------------------------
# 1. Per-leg fees from the OrderFilled commissions
# ---------------------------------------------------------------------------


def test_trade_entry_exit_fee_from_fill_events(engines: list[BacktestEngine]) -> None:
    """Maker entry + taker stop: each leg's fee is that fill's commission."""
    ohlc = {i: (100.5,) * 4 for i in range(_REL, 40)}
    ohlc[_REL + 1] = (100.5, 100.5, 100.3, 100.4)  # trades through the 100.4 limit
    ohlc[_REL + 9] = (100.4, 100.4, 98.0, 98.2)  # through the 1% stop at 99.396
    engine, _ = _run(
        engines,
        data=_strategy_bars(_SIGNAL_CLOSES[:4]) + _one_minute(40, ohlc=ohlc),
        dsl=_dsl("fee_legs"),
    )
    result = _extract(engine, "fee_legs")
    assert len(result.trades) == 1
    trade = result.trades[0]

    pos = _closed_position(engine)
    entries = _fills(pos, OrderSide.BUY)
    exits = _fills(pos, OrderSide.SELL)
    assert entries and exits
    assert all(f.liquidity_side == LiquiditySide.MAKER for f in entries)
    assert all(f.liquidity_side == LiquiditySide.TAKER for f in exits)

    expected_entry = sum(abs(float(f.commission)) for f in entries)
    expected_exit = sum(abs(float(f.commission)) for f in exits)
    assert trade.entry_fee == expected_entry
    assert trade.exit_fee == expected_exit

    inst = engine.cache.instrument(pos.instrument_id)
    entry_notional = sum(float(f.last_qty) * float(f.last_px) for f in entries)
    exit_notional = sum(float(f.last_qty) * float(f.last_px) for f in exits)
    assert trade.entry_fee == pytest.approx(
        entry_notional * float(inst.maker_fee), rel=1e-6
    )
    assert trade.exit_fee == pytest.approx(exit_notional * float(inst.taker_fee), rel=1e-6)
    # not the old 50/50 split: a maker entry pays strictly less than a taker exit
    assert trade.entry_fee < trade.exit_fee


def test_market_mode_trade_fees_unchanged(engines: list[BacktestEngine]) -> None:
    """A market run's per-trade TOTAL is exactly the old (sum-of-commissions) one."""
    ohlc = {i: (100.5,) * 4 for i in range(_REL, 40)}
    ohlc[_REL + 5] = (100.5, 100.5, 98.0, 98.2)  # through the 1% stop at 99.495
    engine, _ = _run(
        engines,
        data=_strategy_bars(_SIGNAL_CLOSES[:4]) + _one_minute(40, ohlc=ohlc),
        dsl=_dsl("fee_market"),
        entry_order="market",
    )
    result = _extract(engine, "fee_market")
    assert len(result.trades) == 1
    trade = result.trades[0]

    pos = _closed_position(engine)
    old_total = abs(sum(float(c) for c in pos.commissions()))  # what D3-1's split used
    assert trade.entry_fee + trade.exit_fee == old_total
    assert result.total_fees == old_total


# ---------------------------------------------------------------------------
# 2. Maker entry statistics from the engine cache
# ---------------------------------------------------------------------------


def test_maker_entry_stats_counts_fill(engines: list[BacktestEngine]) -> None:
    ohlc = {i: (100.5,) * 4 for i in range(_REL, 40)}
    ohlc[_REL + 1] = (100.5, 100.5, 100.3, 100.4)
    engine, _ = _run(
        engines,
        data=_strategy_bars(_SIGNAL_CLOSES[:4]) + _one_minute(40, ohlc=ohlc),
        dsl=_dsl("stats_fill"),
    )
    stats = maker_entry_stats(engine, primary_bar_ns=_M5, maker_entry_timeout_bars=3)
    assert stats["submitted"] == 1
    assert stats["filled"] == 1
    assert stats["fill_rate"] == 1.0
    assert stats["partial"] == 0
    assert stats["expired"] == 0


def test_maker_entry_stats_timeout_is_expired(engines: list[BacktestEngine]) -> None:
    engine, _ = _run(
        engines,
        data=_strategy_bars(_SIGNAL_CLOSES) + _one_minute(60),
        dsl=_dsl("stats_timeout"),
        maker_entry_timeout_bars=3,
    )
    stats = maker_entry_stats(engine, primary_bar_ns=_M5, maker_entry_timeout_bars=3)
    assert stats["submitted"] == 1
    assert stats["filled"] == 0
    assert stats["expired"] == 1
    assert stats["cancelled_by_exit"] == 0
    assert stats["rejected_post_only"] == 0
    assert stats["partial"] == 0
    assert stats["fill_rate"] == 0.0


# ---------------------------------------------------------------------------
# 3. Runner wiring: params, market twin, notes, fail-closed validation
# ---------------------------------------------------------------------------

_VALID_DSL: dict[str, Any] = {
    "name": "maker_runner",
    "description": "test strategy",
    "version": 1,
    "timeframe": "1h",
    "additional_timeframes": [],
    "indicators": {"rsi": {"type": "RSI", "period": 14, "source": "close"}},
    "entry_conditions": {"long": ["rsi < 30"], "short": ["rsi > 70"]},
    "exit_conditions": {"long": ["rsi > 50"], "short": ["rsi < 50"]},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 4.0},
}

_MAKER_KEYS = (
    "submitted",
    "filled",
    "partial",
    "expired",
    "rejected_post_only",
    "cancelled_by_exit",
    "denied",
    "fill_rate",
    "market_twin",
)


@pytest.fixture()
def case(tmp_path: Path) -> dict[str, Any]:
    db = tmp_path / "state.db"
    sm = StateManager(db)
    sid = sm.create_strategy("maker_runner", _VALID_DSL)
    sm.close()
    return {"db": db, "sid": sid, "tmp": tmp_path}


def _make_run(case: dict[str, Any], **params: Any) -> int:
    sm = StateManager(case["db"])
    try:
        return sm.create_backtest_run(
            strategy_id=case["sid"],
            run_mode="validation",
            symbols=["BTCUSDT"],
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-31",
            parameters=params,
        )
    finally:
        sm.close()


def _runner(case: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> ValidationRunner:
    monkeypatch.setattr(ValidationRunner, "_clamp_run_window", lambda self, rc, tf: (rc, None))
    monkeypatch.setattr(
        ValidationRunner, "_resolve_detail_timeframe", lambda self, rc, tf, ov: None
    )
    return ValidationRunner(db_path=case["db"], logs_path=case["tmp"] / "logs")


def _primary_result(**kw: Any) -> ValidationResult:
    r = ValidationResult(
        run_id=kw["run_id"],
        strategy_name=kw["strategy_name"],
        sharpe_ratio=0.22,
        total_return=0.03,
        total_trades=1,
        total_fees=1.0,
    )
    r.notes["maker_entries"] = {
        "submitted": 4,
        "filled": 3,
        "partial": 0,
        "expired": 1,
        "rejected_post_only": 0,
        "denied": 0,
        "cancelled_by_exit": 0,
        "fill_rate": 0.75,
    }
    r.trades.append(
        TradeRecord(
            symbol=_BTC,
            direction="LONG",
            leverage=5,
            entry_time="2024-01-02T00:00:00+00:00",
            exit_time="2024-01-02T05:00:00+00:00",
            entry_price=100.0,
            exit_price=101.0,
            quantity=1.0,
        )
    )
    return r


def test_validation_post_only_notes_have_market_twin_and_fill_rate(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    rid = _make_run(case, entry_order="post_only", maker_offset_ticks=2, maker_entry_timeout_bars=5)
    calls: list[dict[str, Any]] = []

    def fake(**kw: Any) -> ValidationResult:
        calls.append(kw)
        if kw.get("entry_order_override") == "market":
            twin = ValidationResult(
                run_id=kw["run_id"],
                strategy_name=kw["strategy_name"],
                sharpe_ratio=0.11,
                total_return=0.02,
                total_trades=7,
                total_fees=3.0,
            )
            twin.trades.append(
                TradeRecord(
                    symbol="TWIN-PERP.BINANCE",
                    direction="LONG",
                    leverage=1,
                    entry_time="2024-01-02T00:00:00+00:00",
                    exit_time=None,
                    entry_price=1.0,
                    exit_price=None,
                    quantity=1.0,
                )
            )
            return twin
        return _primary_result(**kw)

    runner = _runner(case, monkeypatch)
    runner._run_backtest = fake  # type: ignore[method-assign]
    try:
        result = runner.run(run_id=rid)
    finally:
        runner.close()

    # the identical market twin ran, and it may not write events/trades
    assert len(calls) == 2
    assert calls[1]["entry_order_override"] == "market"
    assert calls[1]["write_trade_events"] is False
    assert calls[0].get("entry_order_override") is None

    maker = result.notes["maker_entries"]
    assert isinstance(maker, dict)
    for key in _MAKER_KEYS:
        assert key in maker, key
    assert 0.0 <= maker["fill_rate"] <= 1.0
    assert maker["market_twin"] == {
        "sharpe_ratio": 0.11,
        "total_return": 0.02,
        "total_trades": 7,
        "total_fees": 3.0,
    }

    sm = StateManager(case["db"])
    try:
        row = sm.get_backtest_run(rid)
        assert row is not None and row["status"] == "completed"
        result_rows = sm.conn.execute(
            "SELECT COUNT(*) FROM backtest_results WHERE run_id = ?", (rid,)
        ).fetchone()[0]
        assert result_rows == 1  # the twin wrote no result row
        stored = [r[0] for r in sm.conn.execute(
            "SELECT symbol FROM trades WHERE run_id = ?", (rid,)
        ).fetchall()]
        assert stored == [_BTC]  # ... and no twin trades
        notes_raw = sm.get_result_notes_raw(rid)
    finally:
        sm.close()
    assert notes_raw is not None
    assert json.loads(notes_raw)["maker_entries"]["market_twin"]["total_trades"] == 7


def test_market_run_has_no_twin_and_no_maker_notes(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    rid = _make_run(case)
    calls: list[dict[str, Any]] = []

    def fake(**kw: Any) -> ValidationResult:
        calls.append(kw)
        r = ValidationResult(
            run_id=kw["run_id"],
            strategy_name=kw["strategy_name"],
            sharpe_ratio=0.22,
            total_return=0.03,
            total_trades=1,
            total_fees=1.0,
        )
        r.trades.append(
            TradeRecord(
                symbol=_BTC,
                direction="LONG",
                leverage=5,
                entry_time="2024-01-02T00:00:00+00:00",
                exit_time="2024-01-02T05:00:00+00:00",
                entry_price=100.0,
                exit_price=101.0,
                quantity=1.0,
            )
        )
        return r

    runner = _runner(case, monkeypatch)
    runner._run_backtest = fake  # type: ignore[method-assign]
    try:
        result = runner.run(run_id=rid)
    finally:
        runner.close()

    assert len(calls) == 1
    assert calls[0].get("entry_order_override") is None
    assert "maker_entries" not in result.notes


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"entry_order": "postonly"}, "entry_order"),
        ({"entry_order": "post_only", "maker_offset_ticks": 0}, "maker_offset_ticks"),
        (
            {"entry_order": "post_only", "maker_entry_timeout_bars": 0},
            "maker_entry_timeout_bars",
        ),
        ({"entry_order": "post_only", "maker_offset_ticks": True}, "maker_offset_ticks"),
    ],
    ids=["typo_mode", "zero_offset", "zero_timeout", "bool_offset"],
)
def test_validation_bad_maker_params_fail_run(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, params: dict[str, Any], match: str
) -> None:
    rid = _make_run(case, **params)
    started: list[int] = []

    def fake(**kw: Any) -> ValidationResult:
        started.append(1)
        raise AssertionError("the engine must not start with invalid maker params")

    runner = _runner(case, monkeypatch)
    runner._run_backtest = fake  # type: ignore[method-assign]
    try:
        with pytest.raises(ValidationRunnerError) as excinfo:
            runner.run(run_id=rid)
    finally:
        runner.close()

    assert match in str(excinfo.value)
    assert started == []
    sm = StateManager(case["db"])
    try:
        row = sm.get_backtest_run(rid)
    finally:
        sm.close()
    assert row is not None
    assert row["status"] == "failed"
    assert match in str(row["error_message"])

    # sanity: the generated config declares the three fields this brief wires
    src = StrategyCompiler().compile(validate_strategy_dict(_dsl("maker_src")))
    assert 'entry_order: str = "market"' in src
    assert "maker_offset_ticks: int = 1" in src
    assert "maker_entry_timeout_bars: int = 3" in src
    assert OrderType.LIMIT is not None


# ---------------------------------------------------------------------------
# 4. Fix round 2: exact per-trade fee reconciliation (review F1)
# ---------------------------------------------------------------------------


class OrderFilled:  # named like the NT event: _leg_fees filters on the type name
    def __init__(self, side: OrderSide, commission: float, liquidity: LiquiditySide) -> None:
        self.order_side = side
        self.commission = commission
        self.liquidity_side = liquidity


class _StubPos:
    entry = OrderSide.BUY

    def __init__(self, *fills: OrderFilled) -> None:
        self.events = list(fills)


def _usdt(x: float) -> float:
    """A commission as NT stores it (Money, 8 decimals) read back as a float."""
    return float(Money(x, Currency.from_str("USDT")))


def test_market_trade_fee_total_exact_with_reviewer_numbers() -> None:
    """qty 0.077 BUY@100.5 / SELL@101.5, both taker: the pair sums to the old total EXACTLY.

    pos.commissions() is 0.007777; the two raw commissions add up to
    0.007776999999999999 -- a per-fill split must not leak that into the total.
    """
    from vibe_quant.validation.extraction import trade_fee_legs

    entry_c, exit_c = _usdt(0.00386925), _usdt(0.00390775)
    pos = _StubPos(
        OrderFilled(OrderSide.BUY, entry_c, LiquiditySide.TAKER),
        OrderFilled(OrderSide.SELL, exit_c, LiquiditySide.TAKER),
    )
    pos_fees = _usdt(0.007777)
    assert entry_c + exit_c != pos_fees  # the trap itself
    entry_fee, exit_fee = trade_fee_legs(pos, pos_fees)
    assert entry_fee + exit_fee == pos_fees
    # pure-taker (market) trades keep the pre-D3-6 even split, byte for byte
    assert (entry_fee, exit_fee) == (abs(pos_fees) / 2.0, abs(pos_fees) / 2.0)


def test_maker_trade_legs_reconcile_exactly_to_position_total() -> None:
    """Maker entry + taker exit: legs follow the fills, the pair still sums to the total."""
    import random

    from vibe_quant.validation.extraction import trade_fee_legs

    rng = random.Random(7)
    for _ in range(3000):
        entry_c = _usdt(rng.uniform(1e-4, 40.0))
        exit_c = _usdt(rng.uniform(1e-4, 40.0))
        pos_fees = _usdt(entry_c + exit_c)
        pos = _StubPos(
            OrderFilled(OrderSide.BUY, entry_c, LiquiditySide.MAKER),
            OrderFilled(OrderSide.SELL, exit_c, LiquiditySide.TAKER),
        )
        entry_fee, exit_fee = trade_fee_legs(pos, pos_fees)
        assert entry_fee + exit_fee == pos_fees, (entry_c, exit_c)
        assert entry_fee == pytest.approx(entry_c, rel=1e-12)
        assert exit_fee == pytest.approx(exit_c, rel=1e-12)


# ---------------------------------------------------------------------------
# 5. Fix round 2: cancel cause is the primary-bar count, not wall-clock (review F3)
# ---------------------------------------------------------------------------


def _stub_order(
    status: OrderStatus, *, init: int, last: int, filled: float = 0.0, qty: float = 1.0
) -> SimpleNamespace:
    return SimpleNamespace(
        is_reduce_only=False,
        is_post_only=True,
        order_type=OrderType.LIMIT,
        status=status,
        filled_qty=filled,
        quantity=qty,
        ts_init=init,
        ts_last=last,
    )


def _stub_engine(*orders: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(cache=SimpleNamespace(orders=lambda: list(orders)))


@pytest.mark.parametrize(
    ("init_min", "cancel_min", "expired", "by_exit"),
    [
        (19, 31, 1, 0),  # review F3: three 5m bars closed (20, 25, 30) but only 12 wall-clock minutes
        (19, 26, 0, 1),  # two bars closed -> an exit/opposite signal cancelled it
        (20, 35, 1, 0),  # placed exactly on a bar close: closes 25, 30, 35 are three bars
        (16, 21, 0, 1),
    ],
    ids=["three_bars_in_12_minutes", "two_bars_exit", "placed_on_close", "one_bar_exit"],
)
def test_cancel_cause_counts_primary_bars_not_minutes(
    init_min: int, cancel_min: int, expired: int, by_exit: int
) -> None:
    order = _stub_order(OrderStatus.CANCELED, init=init_min * _M, last=cancel_min * _M)
    stats = maker_entry_stats(
        _stub_engine(order), primary_bar_ns=_M5, maker_entry_timeout_bars=3
    )
    assert (stats["expired"], stats["cancelled_by_exit"]) == (expired, by_exit)
    assert stats["submitted"] == 1


def test_denied_expired_and_rejected_orders_are_counted_apart() -> None:
    orders = [
        _stub_order(OrderStatus.DENIED, init=0, last=0),
        _stub_order(OrderStatus.REJECTED, init=0, last=_M),
        _stub_order(OrderStatus.EXPIRED, init=0, last=3 * _M5),
        _stub_order(OrderStatus.CANCELED, init=0, last=3 * _M5),
        _stub_order(OrderStatus.CANCELED, init=0, last=_M5),
        _stub_order(OrderStatus.ACCEPTED, init=0, last=0),  # still resting at the end
        _stub_order(OrderStatus.CANCELED, init=0, last=_M5, filled=0.4),  # partial then cancelled
        _stub_order(OrderStatus.FILLED, init=0, last=_M, filled=1.0),
    ]
    stats = maker_entry_stats(
        _stub_engine(*orders), primary_bar_ns=_M5, maker_entry_timeout_bars=3
    )
    assert stats["submitted"] == 8
    assert stats["denied"] == 1
    assert stats["rejected_post_only"] == 1
    assert stats["expired"] == 3  # NT-expired + timed-out cancel + still resting
    assert stats["cancelled_by_exit"] == 1
    assert stats["partial"] == 1
    assert stats["filled"] == 1
    assert stats["fill_rate"] == pytest.approx(1 / 8)


def test_maker_entry_stats_exit_signal_is_cancelled_by_exit(engines: list[BacktestEngine]) -> None:
    """Real engine: an exit signal one bar after placement withdraws the resting entry."""
    engine, _ = _run(
        engines,
        data=_strategy_bars(_SIGNAL_CLOSES) + _one_minute(60),
        dsl=_dsl("stats_exit", exit_conditions={"long": ["close < 100.5"]}),
    )
    stats = maker_entry_stats(engine, primary_bar_ns=_M5, maker_entry_timeout_bars=3)
    assert stats["submitted"] == 1
    assert stats["cancelled_by_exit"] == 1
    assert stats["expired"] == 0
    assert stats["fill_rate"] == 0.0


# ---------------------------------------------------------------------------
# 6. Fix round 2: a failing market twin never costs the primary run (review F2)
# ---------------------------------------------------------------------------


def test_twin_failure_keeps_primary_run_completed(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    rid = _make_run(case, entry_order="post_only")

    def fake(**kw: Any) -> ValidationResult:
        if kw.get("entry_order_override") == "market":
            msg = "twin engine blew up"
            raise RuntimeError(msg)
        return _primary_result(**kw)

    runner = _runner(case, monkeypatch)
    runner._run_backtest = fake  # type: ignore[method-assign]
    try:
        with caplog.at_level("ERROR"):
            result = runner.run(run_id=rid)
    finally:
        runner.close()

    maker = result.notes["maker_entries"]
    assert isinstance(maker, dict)
    assert maker["market_twin"] == {"error": "RuntimeError: twin engine blew up"}
    assert maker["submitted"] == 4  # the primary stats survive
    assert "twin engine blew up" in caplog.text

    sm = StateManager(case["db"])
    try:
        row = sm.get_backtest_run(rid)
        assert row is not None and row["status"] == "completed"
        assert sm.conn.execute(
            "SELECT COUNT(*) FROM backtest_results WHERE run_id = ?", (rid,)
        ).fetchone()[0] == 1
        notes_raw = sm.get_result_notes_raw(rid)
    finally:
        sm.close()
    assert notes_raw is not None
    assert json.loads(notes_raw)["maker_entries"]["market_twin"] == {
        "error": "RuntimeError: twin engine blew up"
    }


def test_fee_schedule_source_is_not_forwarded_to_the_strategy_config(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner(case, monkeypatch)
    try:
        params = runner._build_strategy_params(
            {"parameters": {"rsi_period": 14, "fee_schedule_source": "inherited:42"}}
        )
    finally:
        runner.close()
    assert params == {"rsi_period": 14}


# ---------------------------------------------------------------------------
# 7. Synthetic catalog for REAL ValidationRunner runs (review F4)
# ---------------------------------------------------------------------------

_DAY0_MS = 1_704_067_200_000  # 2024-01-01T00:00Z
SYNTHETIC_DSL: dict[str, Any] = {
    "name": "maker_real_run",
    "description": "sine-wave trend follower for the maker-entry real-run tests",
    "version": 1,
    "timeframe": "1h",
    "additional_timeframes": [],
    "indicators": {"sma": {"type": "SMA", "period": 3}},
    "entry_conditions": {"long": ["close > sma"]},
    "exit_conditions": {"long": ["close < sma"]},
    "stop_loss": {"type": "fixed_pct", "percent": 1.0},
    "take_profit": {"type": "fixed_pct", "percent": 1.0},
}


def write_synthetic_catalog(path: Path, *, days: int = 4) -> None:
    """BTCUSDT 1m + 1h bars of a noisy sine wave (period 12h, +-2%) in a fresh catalog.

    Each 1m bar spans +-0.3 around its path, so post-only limits one tick inside
    the release price are traded through; the 1h bars aggregate the 1m bars.
    """
    import math

    from vibe_quant.data.catalog import CatalogManager, get_bar_type, klines_to_bars

    inst = create_instrument("BTCUSDT")
    cm = CatalogManager(path)
    cm.write_instrument(inst)
    minutes = days * 24 * 60
    closes = [
        round(100.0 + 2.0 * math.sin(2 * math.pi * m / (12 * 60)) + 0.2 * math.sin(m / 7.0), 1)
        for m in range(minutes)
    ]
    one_min: list[dict[str, float]] = []
    prev = closes[0]
    for m, close in enumerate(closes):
        open_ms = _DAY0_MS + m * 60_000
        one_min.append(
            {
                "open_time": open_ms,
                "open": prev,
                "high": max(prev, close) + 0.3,
                "low": min(prev, close) - 0.3,
                "close": close,
                "volume": 1000.0,
                "close_time": open_ms + 59_999,
            }
        )
        prev = close
    hourly = []
    for h in range(days * 24):
        chunk = one_min[h * 60 : (h + 1) * 60]
        hourly.append(
            {
                "open_time": chunk[0]["open_time"],
                "open": chunk[0]["open"],
                "high": max(k["high"] for k in chunk),
                "low": min(k["low"] for k in chunk),
                "close": chunk[-1]["close"],
                "volume": sum(k["volume"] for k in chunk),
                "close_time": chunk[-1]["close_time"],
            }
        )
    for interval, klines in (("1m", one_min), ("1h", hourly)):
        cm.write_bars(
            klines_to_bars(  # type: ignore[arg-type]
                klines,
                inst.id,
                get_bar_type("BTCUSDT", interval),
                inst.size_precision,
                inst.price_precision,
            )
        )
