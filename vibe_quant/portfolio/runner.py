"""The carry engine runner: a low-level NT ``BacktestEngine`` over in-memory ticks.

Task T3 of the funding-carry epic (``design.md`` §1.3, §1.4, §8-T3). The engine
gets Python objects only -- the catalog's instruments and the decision-time 1m
trade ticks built by ``ticks.to_trade_ticks`` -- so there is no tick catalog, no
``BacktestDataConfig`` and no epoch-parquet cleanup. The venue is a NETTING
MARGIN BINANCE account with the design's fill model, taker fee model, leverage
and ``use_reduce_only`` (MARGIN accounts skip NT's pre-trade risk checks, so
Binance minimums stay ``sizing.py``'s job).

NT force-stops a run silently on an ``AccountError`` (``engine.pyx:1742``) and
stops at data exhaustion, so a run that did not execute the planned book looks
exactly like a successful one: ``_require_executed`` re-derives the book from
counts and NT state (R29/R29b) and raises ``CarryExecutionError`` instead.
"""

from __future__ import annotations

import contextlib
from decimal import Decimal
from typing import TYPE_CHECKING

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import FillModel, MakerTakerFeeModel
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.enums import AccountType, BookType, OmsType, OrderStatus
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Currency, Money

from vibe_quant.nt_compat import retain_log_guard
from vibe_quant.portfolio.ledger import CarryRun, steps_held
from vibe_quant.portfolio.spec import VENUE_LEVERAGE, CarryExecutionError
from vibe_quant.portfolio.strategy import FundingCarryStrategy
from vibe_quant.portfolio.ticks import to_trade_ticks

if TYPE_CHECKING:
    from vibe_quant.portfolio.ticks import CarryInputs

_VENUE = "BINANCE"
_QUOTE = "USDT"


def run_carry_engine(inputs: CarryInputs, *, reverse_data: bool = False) -> CarryRun:
    """Run one carry book on a low-level NT engine and return its :class:`CarryRun`.

    Args:
        inputs: The preflighted run inputs (instruments, ticks, marks, archive).
        reverse_data: Reverse the tick list handed to ``engine.add_data``; NT's
            stable sort by ``ts_init`` then reorders same-ts ticks (R34 test only).

    Raises:
        CarryExecutionError: if the engine did not execute exactly the planned
            book (stale/unreleased/unfilled orders, unprocessed ticks or
            decisions, NT-state drift).
    """
    spec = inputs.spec
    engine = BacktestEngine(BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    try:
        engine.add_venue(
            Venue(_VENUE),
            OmsType.NETTING,
            AccountType.MARGIN,
            [Money(spec.initial_balance, Currency.from_str(_QUOTE))],
            default_leverage=Decimal(VENUE_LEVERAGE),
            fill_model=FillModel(prob_fill_on_limit=1.0, prob_slippage=0.0, random_seed=42),
            fee_model=MakerTakerFeeModel(),
            book_type=BookType.L1_MBP,
            use_reduce_only=True,
        )
        for symbol in spec.symbols:
            engine.add_instrument(inputs.instruments[symbol])
        data = [
            tick
            for symbol in spec.symbols
            for tick in to_trade_ticks(inputs.instruments[symbol], inputs.ticks[symbol])
        ]
        if not data:  # NT add_data([]) raises; the loader refuses an all-unlisted run first
            raise CarryExecutionError("no decision ticks for any symbol: nothing to run")
        if reverse_data:
            data.reverse()
        engine.add_data(data)
        strategy = FundingCarryStrategy(inputs)
        engine.add_strategy(strategy)
        engine.run(start=spec.start_ns, end=spec.end_ns - 1)

        _require_executed(strategy, inputs, len(data))
        strategy.funding.advance(strategy.fills, spec.end_ns)
        return CarryRun(
            spec=spec,
            decisions=tuple(strategy.decisions),
            fills=tuple(sorted(strategy.fills, key=lambda fill: (fill.ts, fill.symbol))),
            payments=tuple(strategy.funding.payments),
            final_steps={
                symbol: steps_held(strategy.fills, symbol, spec.end_ns) for symbol in spec.symbols
            },
            engine="nt",
            funding_switch_holes_upper=strategy.funding.funding_switch_holes_upper,
            funding_fallback_charges=strategy.funding.funding_fallback_charges,
        )
    finally:
        retain_log_guard(engine)
        with contextlib.suppress(Exception):
            engine.reset()
        engine.dispose()  # type: ignore[no-untyped-call]


def _require_executed(
    strategy: FundingCarryStrategy, inputs: CarryInputs, total_ticks: int
) -> None:
    """R29/R29b: raise unless NT executed every tick, decision and planned order.

    ``engine.run()`` returns normally after a silent force-stop or a short data
    feed, so the plan is re-derived from the counts and from NT's own state.
    """
    if strategy.errors:
        raise CarryExecutionError("; ".join(strategy.errors))
    strategy.require_released(inputs.spec.end_ns)
    if strategy.outbox.pending():
        raise CarryExecutionError(
            f"unreleased orders: {strategy.outbox.pending()} still queued at the end"
        )
    if strategy.ticks_seen != total_ticks:
        raise CarryExecutionError(
            f"ticks seen {strategy.ticks_seen} != {total_ticks} added to the engine"
        )
    if len(strategy.decisions) != len(inputs.decisions):
        raise CarryExecutionError(
            f"decisions {len(strategy.decisions)} != {len(inputs.decisions)} scheduled"
        )
    if len(strategy._submitted) != len(strategy.planned):
        raise CarryExecutionError(
            f"submitted {len(strategy._submitted)} of {len(strategy.planned)} planned "
            "orders (not filled)"
        )
    for client_order_id, intent in strategy.planned.items():
        order = strategy._submitted.get(client_order_id)
        if order is None or order.status != OrderStatus.FILLED:
            status = "missing" if order is None else order.status.name
            raise CarryExecutionError(
                f"planned order {client_order_id} for {intent.symbol} not filled (status {status})"
            )
