"""NT carry engine: outbox, strategy, runner lifecycle (task T3).

Contract rows R14c (flip = close then open, same release tick), R15 (fill on the
symbol's own next tick), R15b (stale release), R15c (unreleased order), R21
(open positions not closed at the end), R29 (silent NT stop), R29b (NT-state
cross-checks), R31 (unplanned fill) and R34 (same-ts ordering) of the
funding-carry semantics contract (``design.md`` §1.3, §7, §8-T3).

The fixture is synthetic on purpose: two catalog instruments (BTCUSDT/ETHUSDT),
a tmp archive whose 8h rows make BTC the low-funding leg in week 1 and the high
one in week 2 (so week 2 flips the book), a flat MARK series at every settlement
minute, and hand-built ``PriceTick``s -- a sizing tick at ``D-1 ms`` and a fill
tick at ``D+59.999 s`` per symbol-week. Every fill number is hand-checkable
(R15: BUY 0.098 @ 51010.0, commission 2.49949).
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pytest
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import ClientOrderId, StrategyId, TraderId
from nautilus_trader.model.orders import MarketOrder

from vibe_quant.data.catalog import create_instrument
from vibe_quant.data.fill_ticks import FILL_TICK_SIZE
from vibe_quant.data.mark_bars import MarkSeries
from vibe_quant.portfolio.ledger import steps_held
from vibe_quant.portfolio.outbox import InstrumentOutbox
from vibe_quant.portfolio.reference import simulate_reference
from vibe_quant.portfolio.runner import run_carry_engine
from vibe_quant.portfolio.signal import Settlements
from vibe_quant.portfolio.sizing import InstrumentSpec, OrderIntent
from vibe_quant.portfolio.spec import CarryExecutionError, CarrySpec, decision_times
from vibe_quant.portfolio.strategy import FundingCarryStrategy
from vibe_quant.portfolio.ticks import CarryInputs, PriceTick, sizing_price, to_trade_ticks
from vibe_quant.validation.funding import FundingCalculator, clear_rate_cache

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from nautilus_trader.model.data import TradeTick

_MIN = 60 * 10**9
_HOUR = 3_600 * 10**9
_DAY = 86_400 * 10**9
_WEEK = 7 * _DAY
_BAR_TS_INIT = 59_999_000_000  # catalog 1m bar: ts_init = open + 59.999 s
_EIGHT_HOURS = 8 * _HOUR
_SETTLEMENT_JITTER_MS = 3  # archived funding_time carries a few ms after the boundary

_SYMBOLS = ("BTCUSDT", "ETHUSDT")
_SIZING = {"BTCUSDT": (51000.0, 52000.0), "ETHUSDT": (3000.0, 3100.0)}
_FILL = {"BTCUSDT": (51010.0, 52010.0), "ETHUSDT": (3010.0, 3110.0)}
_MARK = {"BTCUSDT": 51000.0, "ETHUSDT": 3000.0}


def _ns(date: str) -> int:
    """``YYYY-MM-DD`` 00:00 UTC -> epoch ns."""
    return int(datetime.fromisoformat(date).replace(tzinfo=UTC).timestamp()) * 10**9


START = _ns("2025-01-06")  # a Monday 00:00 UTC
END = START + 2 * _WEEK  # Monday 2025-01-20 00:00 UTC
D0 = START + 5 * _MIN  # first decision, Monday 00:05
D1 = D0 + _WEEK  # second decision
_SWITCH = D0 + 2 * _DAY  # funding rates swap after this boundary (week 2 flips)
_BOUNDARY_FIRST = START - 9 * _DAY  # the earliest 8h boundary the signal window needs


def _boundaries() -> list[int]:
    return list(range(_BOUNDARY_FIRST, END, _EIGHT_HOURS))


def _rate(symbol: str, boundary: int) -> float:
    """BTC pays least before ``_SWITCH`` (week 1 long) and most after it (week 2 short)."""
    before, after = (0.0001, 0.0002) if symbol == "BTCUSDT" else (0.0002, 0.0001)
    return before if boundary < _SWITCH else after


def _settlement(symbol: str, boundaries: Sequence[int]) -> Settlements:
    return Settlements(
        times_ns=tuple(boundary + _SETTLEMENT_JITTER_MS * 10**6 for boundary in boundaries),
        rates=tuple(_rate(symbol, boundary) for boundary in boundaries),
    )


def _archive(
    path: Path,
    boundaries: Sequence[int],
    funded: Sequence[str] = _SYMBOLS,
    gaps: dict[str, frozenset[int]] | None = None,
) -> FundingCalculator:
    """A tmp funding archive: 8h rows for the ``funded`` symbols, ms-jittered like Binance."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        """CREATE TABLE raw_funding_rates (
            id INTEGER PRIMARY KEY, symbol TEXT NOT NULL, funding_time INTEGER NOT NULL,
            funding_rate REAL NOT NULL, mark_price REAL, source TEXT,
            UNIQUE(symbol, funding_time))"""
    )
    for symbol in funded:
        for boundary in boundaries:
            if boundary in (gaps or {}).get(symbol, frozenset()):
                continue
            conn.execute(
                "INSERT INTO raw_funding_rates (symbol, funding_time, funding_rate) VALUES (?,?,?)",
                (
                    symbol,
                    boundary // 1_000_000 + _SETTLEMENT_JITTER_MS,
                    _rate(symbol, boundary),
                ),
            )
    conn.commit()
    conn.close()
    return FundingCalculator(path)


def _marks(price: float) -> MarkSeries:
    """A flat 1m MARK series over the fixture window (one bar opening per minute)."""
    ts = np.arange(START - _EIGHT_HOURS, END, _MIN, dtype=np.int64)
    values = np.full(ts.shape, price, dtype=np.float64)
    return MarkSeries(ts=ts, low=values, high=values, close=values, open=values)


def _ticks(decisions: Sequence[int] = (D0, D1)) -> dict[str, tuple[PriceTick, ...]]:
    """Per decision: the sizing tick (``D-1 ms``) then the fill tick (``D+59.999 s``)."""
    return {
        symbol: tuple(
            tick
            for index, decision_ns in enumerate(decisions)
            for tick in (
                PriceTick(decision_ns - 1_000_000, _SIZING[symbol][index]),
                PriceTick(decision_ns + _BAR_TS_INIT, _FILL[symbol][index]),
            )
        )
        for symbol in _SYMBOLS
    }


def _inputs(
    tmp_path: Path,
    *,
    name: str = "a",
    order: Sequence[str] = _SYMBOLS,
    funded: Sequence[str] = _SYMBOLS,
    ticks: dict[str, tuple[PriceTick, ...]] | None = None,
    gaps: dict[str, frozenset[int]] | None = None,
) -> CarryInputs:
    """A synthetic two-symbol carry run: two decisions, a flip in the second week."""
    spec = CarrySpec(
        symbols=_SYMBOLS,
        start_ns=START,
        end_ns=END,
        initial_balance=10_000.0,
        legs_divisor=3,
        gross=1.0,
        band=0.10,
    )
    boundaries = _boundaries()
    all_ticks = _ticks() if ticks is None else ticks
    instruments = {symbol: create_instrument(symbol) for symbol in order}
    return CarryInputs(
        spec=spec,
        decisions=tuple(decision_times(START, END)),
        instruments=instruments,
        specs={symbol: InstrumentSpec.from_instrument(instrument) for symbol, instrument in instruments.items()},
        ticks={symbol: all_ticks[symbol] for symbol in order},
        settlements={
            symbol: (
                _settlement(
                    symbol, [b for b in boundaries if b not in (gaps or {}).get(symbol, frozenset())]
                )
                if symbol in funded
                else Settlements((), ())
            )
            for symbol in order
        },
        marks={symbol: _marks(_MARK[symbol]) for symbol in order},
        calc=_archive(tmp_path / f"raw_{name}.db", boundaries, funded, gaps),
        boundary_prices={symbol: {} for symbol in order},
    )


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    clear_rate_cache()


@pytest.fixture
def inputs(tmp_path: Path) -> CarryInputs:
    return _inputs(tmp_path)


# ---------------------------------------------------------------------------
# InstrumentOutbox: strict release, FIFO, per instrument
# ---------------------------------------------------------------------------


def _order(symbol: str, qty: str) -> MarketOrder:
    instrument = create_instrument(symbol)
    return MarketOrder(
        trader_id=TraderId("TESTER-001"),
        strategy_id=StrategyId("S-001"),
        instrument_id=instrument.id,
        client_order_id=ClientOrderId(f"O-{symbol}-{qty}"),
        order_side=OrderSide.BUY,
        quantity=instrument.make_qty(qty),
        init_id=UUID4(),
        ts_init=0,
        time_in_force=TimeInForce.IOC,
    )


def test_outbox_release_is_strict_fifo_and_per_instrument() -> None:
    """A queued order is released by its OWN symbol's next tick, strict ``queue_ts < ts``."""
    outbox = InstrumentOutbox()
    first = _order("BTCUSDT", "0.001")
    second = _order("BTCUSDT", "0.002")
    other = _order("ETHUSDT", "0.001")
    outbox.queue("BTCUSDT", 1_000, first)
    outbox.queue("BTCUSDT", 1_000, second)
    outbox.queue("ETHUSDT", 1_000, other)
    assert outbox.pending() == 3

    assert outbox.release("BTCUSDT", 1_000) == []  # strict: a datum at the queue ts never releases
    assert outbox.release("ETHUSDT", 1_001) == [(1_000, other)]  # another symbol's tick never does
    assert outbox.release("ETHUSDT", 1_002) == []  # released once, gone
    released = outbox.release("BTCUSDT", 1_001)
    assert [ts for ts, _ in released] == [1_000, 1_000]  # FIFO
    assert [order.client_order_id for _, order in released] == [
        first.client_order_id,
        second.client_order_id,
    ]
    assert outbox.pending() == 0
    assert outbox.release("BTCUSDT", 2_000) == []


def test_outbox_release_only_before_the_tick() -> None:
    """A tick before the queue ts releases nothing; the boundary is exclusive."""
    outbox = InstrumentOutbox()
    order = _order("BTCUSDT", "0.001")
    outbox.queue("BTCUSDT", 1_000, order)
    assert outbox.release("BTCUSDT", 999) == []
    assert outbox.pending() == 1
    assert outbox.release("BTCUSDT", 1_001) == [(1_000, order)]


# ---------------------------------------------------------------------------
# The engine run on the synthetic book
# ---------------------------------------------------------------------------


def test_decision_fills_on_next_own_tick(inputs: CarryInputs) -> None:
    """Queued at D, filled on the symbol's own next tick -- never at the sizing tick."""
    run = run_carry_engine(inputs)

    assert run.engine == "nt"
    assert [decision.ts for decision in run.decisions] == [D0, D1]
    assert [(o.symbol, o.steps, o.reduce_only) for o in run.decisions[0].orders] == [
        ("BTCUSDT", 98, False),
        ("ETHUSDT", -1666, False),
    ]
    week1 = [fill for fill in run.fills if fill.ts < D1]
    assert [(f.symbol, f.ts, f.steps, f.price, f.fee, f.reduce_only) for f in week1] == [
        ("BTCUSDT", D0 + _BAR_TS_INIT, 98, 51010.0, 2.49949, False),
        ("ETHUSDT", D0 + _BAR_TS_INIT, -1666, 3010.0, 2.50733, False),
    ]
    assert all(fill.ts != D0 - 1_000_000 for fill in run.fills)
    assert len(run.fills) == 6  # 2 week-1 opens + 2 week-2 flips x (close, open)


def test_flip_is_close_then_open_same_tick(inputs: CarryInputs) -> None:
    """A sign flip closes (reduce-only) then opens, in that order, on one tick."""
    run = run_carry_engine(inputs)

    assert run.decisions[0].signals["BTCUSDT"] == pytest.approx(0.0021)
    assert run.decisions[0].signals["ETHUSDT"] == pytest.approx(0.0042)
    assert run.decisions[0].weights == {"BTCUSDT": 0.5, "ETHUSDT": -0.5}
    assert run.decisions[1].signals["BTCUSDT"] == pytest.approx(0.0036)
    assert run.decisions[1].signals["ETHUSDT"] == pytest.approx(0.0027)
    assert run.decisions[1].weights == {"BTCUSDT": -0.5, "ETHUSDT": 0.5}

    flipped = [fill for fill in run.fills if fill.symbol == "BTCUSDT" and fill.ts == D1 + _BAR_TS_INIT]
    assert [(fill.steps, fill.reduce_only) for fill in flipped] == [
        (-98, True),
        (run.decisions[1].target_steps["BTCUSDT"], False),
    ]
    assert flipped[0].steps == -steps_held(run.fills, "BTCUSDT", D1)
    assert flipped[1].steps < 0
    assert [fill.ts for fill in flipped] == [D1 + _BAR_TS_INIT, D1 + _BAR_TS_INIT]


def test_funding_hole_excludes_that_symbol(tmp_path: Path) -> None:
    """An uncovered 8h boundary excludes that symbol only, with its reason."""
    run = run_carry_engine(_inputs(tmp_path, funded=("BTCUSDT",)))

    assert run.decisions[0].excluded == {"ETHUSDT": "funding_hole"}
    assert run.decisions[0].signals == {"BTCUSDT": pytest.approx(0.0021), "ETHUSDT": None}
    assert run.decisions[0].weights == {"BTCUSDT": 0.0, "ETHUSDT": 0.0}
    assert run.decisions[0].orders == ()
    assert run.fills == ()
    assert run.final_steps == dict.fromkeys(_SYMBOLS, 0)


def test_funding_hole_wins_over_missing_price(tmp_path: Path) -> None:
    """A symbol with both a funding hole and no sizing tick keeps ``funding_hole``."""
    all_ticks = _ticks()
    inputs = _inputs(
        tmp_path,
        funded=("BTCUSDT",),
        ticks={"BTCUSDT": all_ticks["BTCUSDT"], "ETHUSDT": ()},
    )

    run = run_carry_engine(inputs)

    assert run.decisions[0].excluded == {"ETHUSDT": "funding_hole"}
    assert run.decisions[0].signals == {"BTCUSDT": pytest.approx(0.0021), "ETHUSDT": None}


def test_open_positions_not_closed_at_end(inputs: CarryInputs) -> None:
    """Nothing is closed at the end; the final book is still held."""
    run = run_carry_engine(inputs)

    assert len(run.fills) == 6
    assert max(fill.ts for fill in run.fills) == D1 + _BAR_TS_INIT < END
    assert all(fill.ts < END for fill in run.fills)
    assert run.final_steps == {symbol: steps_held(run.fills, symbol, END) for symbol in _SYMBOLS}
    assert run.final_steps["BTCUSDT"] == run.decisions[1].target_steps["BTCUSDT"]
    assert run.final_steps["ETHUSDT"] == run.decisions[1].target_steps["ETHUSDT"]
    assert all(steps != 0 for steps in run.final_steps.values())


# ---------------------------------------------------------------------------
# Release staleness and unreleased orders
# ---------------------------------------------------------------------------


def test_stale_release_fails(tmp_path: Path) -> None:
    """BTC's next own tick is a week late -> ``CarryExecutionError``."""
    ticks = dict(_ticks())
    ticks["BTCUSDT"] = tuple(t for t in ticks["BTCUSDT"] if t.ts != D0 + _BAR_TS_INIT)

    with pytest.raises(CarryExecutionError, match="stale release BTCUSDT"):
        run_carry_engine(_inputs(tmp_path, ticks=ticks))


def test_unreleased_order_fails(tmp_path: Path) -> None:
    """No later own tick before the end -> the queued order is never released."""
    ticks = dict(_ticks())
    ticks["BTCUSDT"] = tuple(t for t in ticks["BTCUSDT"] if t.ts != D1 + _BAR_TS_INIT)

    with pytest.raises(CarryExecutionError, match="unreleased orders"):
        run_carry_engine(_inputs(tmp_path, ticks=ticks))


# ---------------------------------------------------------------------------
# The engine did not execute the planned book
# ---------------------------------------------------------------------------


def test_counts_check_catches_silent_stop(tmp_path: Path) -> None:
    """``engine.run()`` returns normally when a scheduled decision never ran."""
    inputs = _inputs(tmp_path)
    phantom = inputs.decisions[-1] + _WEEK  # beyond the run window: the alert never fires

    with pytest.raises(CarryExecutionError, match="decisions 2 != 3 scheduled"):
        run_carry_engine(replace(inputs, decisions=(*inputs.decisions, phantom)))


def test_ticks_seen_check_catches_unprocessed_tick(tmp_path: Path) -> None:
    """A tick the engine never processed fails the count check too."""
    ticks = dict(_ticks())
    ticks["BTCUSDT"] = (*ticks["BTCUSDT"], PriceTick(END + _MIN, 51000.0))

    with pytest.raises(CarryExecutionError, match="ticks seen"):
        run_carry_engine(_inputs(tmp_path, ticks=ticks))


def test_denied_order_fails(inputs: CarryInputs, monkeypatch: pytest.MonkeyPatch) -> None:
    """A reduce-only BUY while flat is rejected by the NT venue -> fail closed."""

    def _reduce_only_buy(
        current: object, targets: object, prices: object, specs: object, band: float
    ) -> tuple[tuple[OrderIntent, ...], int, int]:
        return (OrderIntent("BTCUSDT", 5, reduce_only=True),), 0, 0

    monkeypatch.setattr("vibe_quant.portfolio.sizing.plan_orders", _reduce_only_buy)

    with pytest.raises(CarryExecutionError, match="rejected|denied"):
        run_carry_engine(inputs)


def test_partial_fill_fails(inputs: CarryInputs, monkeypatch: pytest.MonkeyPatch) -> None:
    """A partial fill (small tick liquidity) is not the planned book."""

    def _small_ticks(
        instrument: object, ticks: Sequence[PriceTick], size: float = FILL_TICK_SIZE
    ) -> list[TradeTick]:
        return to_trade_ticks(instrument, ticks, size=0.001)  # type: ignore[arg-type]

    monkeypatch.setattr("vibe_quant.portfolio.runner.to_trade_ticks", _small_ticks)

    with pytest.raises(CarryExecutionError, match="partial fill"):
        run_carry_engine(inputs)


def test_nt_state_mismatch_fails(inputs: CarryInputs, monkeypatch: pytest.MonkeyPatch) -> None:
    """The ledger's held steps and NT's booked net position must agree at every D."""
    real = steps_held
    monkeypatch.setattr(
        "vibe_quant.portfolio.strategy.steps_held",
        lambda fills, symbol, before_ns: real(fills, symbol, before_ns) + 1,
    )

    with pytest.raises(CarryExecutionError, match="net position"):
        run_carry_engine(inputs)


def test_unfilled_order_fails(inputs: CarryInputs, monkeypatch: pytest.MonkeyPatch) -> None:
    """After the run every planned order must be FILLED."""
    monkeypatch.setattr(FundingCarryStrategy, "submit_order", lambda self, order: None)

    with pytest.raises(CarryExecutionError, match="not filled"):
        run_carry_engine(inputs)


def test_unplanned_fill_fails(inputs: CarryInputs, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fill whose order the strategy never planned fails closed."""

    class _ExtraOrder(FundingCarryStrategy):
        def on_trade_tick(self, tick: TradeTick) -> None:
            super().on_trade_tick(tick)
            btc = self.inputs.instruments["BTCUSDT"]
            self.submit_order(
                self.order_factory.market(
                    btc.id,
                    OrderSide.SELL,
                    btc.make_qty("0.001"),
                    time_in_force=TimeInForce.IOC,
                )
            )

    monkeypatch.setattr("vibe_quant.portfolio.runner.FundingCarryStrategy", _ExtraOrder)

    with pytest.raises(CarryExecutionError, match="unplanned fill"):
        run_carry_engine(inputs)


# ---------------------------------------------------------------------------
# Same-ts ordering + the double-run lifecycle
# ---------------------------------------------------------------------------


def _record_add_data(monkeypatch: pytest.MonkeyPatch) -> list[list[tuple[str, int, float]]]:
    """Capture every ``engine.add_data`` argument as (instrument, ts_init, price) in call order."""
    seen: list[list[tuple[str, int, float]]] = []

    class _Recording(BacktestEngine):  # type: ignore[misc]
        def add_data(self, data: list[TradeTick], *args: object, **kwargs: object) -> None:
            seen.append([(str(t.instrument_id), int(t.ts_init), float(t.price)) for t in data])
            super().add_data(data, *args, **kwargs)

    monkeypatch.setattr("vibe_quant.portfolio.runner.BacktestEngine", _Recording)
    return seen


def test_add_data_order_invariant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reversed ``add_data`` order and reversed input dicts give the same run."""
    seen = _record_add_data(monkeypatch)
    forward = run_carry_engine(_inputs(tmp_path, name="a"))
    backward = run_carry_engine(
        _inputs(tmp_path, name="b", order=tuple(reversed(_SYMBOLS))), reverse_data=True
    )

    assert len(seen) == 2
    assert seen[1] != seen[0]  # the order really changed ...
    assert seen[1] == seen[0][::-1]  # ... and is exactly the reversed feed
    assert backward.decisions == forward.decisions
    assert backward.fills == forward.fills
    assert backward.payments == forward.payments
    assert backward.final_steps == forward.final_steps


# ---------------------------------------------------------------------------
# Cache-tick check and NT == reference parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("ts_shift", "price_shift"), [(1, 0.0), (0, 1.0)])
def test_cached_tick_not_sizing_tick_fails(
    inputs: CarryInputs, monkeypatch: pytest.MonkeyPatch, ts_shift: int, price_shift: float
) -> None:
    """NT's cached last trade tick must be the sizing tick (ts and price)."""
    real = sizing_price

    def _shifted(ticks: Sequence[PriceTick], decision_ns: int) -> PriceTick | None:
        tick = real(ticks, decision_ns)
        if tick is None or tick.price not in _SIZING["BTCUSDT"]:
            return tick
        return PriceTick(tick.ts + ts_shift, tick.price + price_shift)

    monkeypatch.setattr("vibe_quant.portfolio.strategy.sizing_price", _shifted)

    with pytest.raises(CarryExecutionError, match="not the sizing tick"):
        run_carry_engine(inputs)


def test_engine_matches_reference_on_fixture(inputs: CarryInputs) -> None:
    """NT == reference, exactly: short leg, week-2 flip, same inputs (decisions, fills, funding)."""
    nt = run_carry_engine(inputs)
    ref = simulate_reference(inputs)

    assert nt.decisions == ref.decisions
    assert nt.fills == ref.fills
    assert nt.payments == ref.payments
    assert nt.final_steps == ref.final_steps
    assert any(fill.steps < 0 and fill.qty < 0 for fill in nt.fills)  # a real short leg


def test_engine_matches_reference_with_held_funding_hole(tmp_path: Path) -> None:
    """A held symbol that loses funding coverage is closed, not a data error, in both executors."""
    hole = frozenset({D0 + 3 * _DAY - D0 % _EIGHT_HOURS + _EIGHT_HOURS})
    assert all(boundary % _EIGHT_HOURS == 0 for boundary in hole)
    inputs = _inputs(tmp_path, gaps={"ETHUSDT": hole})

    nt = run_carry_engine(inputs)
    ref = simulate_reference(inputs)

    assert nt.decisions[1].excluded == {"ETHUSDT": "funding_hole"}
    assert nt.decisions[1].prices.keys() == {"BTCUSDT", "ETHUSDT"}  # priced though excluded
    assert [(o.symbol, o.steps, o.reduce_only) for o in nt.decisions[1].orders] == [
        ("BTCUSDT", -98, True),
        ("ETHUSDT", 1666, True),
    ]
    assert nt.final_steps == dict.fromkeys(_SYMBOLS, 0)
    assert nt.decisions == ref.decisions
    assert nt.fills == ref.fills
    assert nt.payments == ref.payments
    assert nt.final_steps == ref.final_steps


def test_two_engine_runs_in_one_process(tmp_path: Path) -> None:
    """Two ``run_carry_engine`` calls in one process both succeed (log guard retained)."""
    first = run_carry_engine(_inputs(tmp_path, name="a"))
    second = run_carry_engine(_inputs(tmp_path, name="b"))

    assert first.fills == second.fills
    assert first.payments == second.payments
    assert first.final_steps == second.final_steps
