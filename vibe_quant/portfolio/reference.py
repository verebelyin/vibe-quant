"""Funding-carry reference simulator: the shared rules, no NautilusTrader (§1.2, §5).

:func:`simulate_reference` walks a run's decisions in order and executes them
with exactly the same rule functions the NT engine uses — :func:`decide`,
:class:`FundingAccumulator`, :class:`BookIndex` (the indexed :func:`equity_at`),
:func:`ticks.release_tick`/:func:`ticks.check_release` and
:func:`sizing.nt_commission` — so the two executors produce bitwise-equal
outputs (the parity/exactness task proves it, R30). It re-implements no rule:

* a planned order fills at the symbol's first tick strictly after its decision
  (:func:`release_tick`, staleness via :func:`check_release`, R15b); a symbol
  with no later tick can never be released and fails closed (R15c);
* the fill price is that tick's price and the fee is the NT taker formula
  (:func:`nt_commission`, §5) on the exact quantity
  ``float(Decimal(steps) * step)``;
* funding is accrued window by window by the shared
  :class:`FundingAccumulator` (:func:`run_accumulator`), up to ``spec.end_ns``;
* a trend run (arch-c9-2) reads its daily closes, listings and regime through the
  same :func:`decision_signals`/:func:`decide` calls, and every fill pays the
  spread cost :func:`sizing.spread_cost` (0.0 on the funding path).

The book is one pooled account: ``decide`` sizes every symbol against the
equity the previous windows wrote (:meth:`BookIndex.equity`), keyed off each symbol's
sizing tick; orders of one decision release in plan order (a flip's close
before its open) at the same tick.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from vibe_quant.portfolio.ledger import BookIndex, CarryFill, CarryRun, run_accumulator
from vibe_quant.portfolio.signal import decision_signals
from vibe_quant.portfolio.sizing import decide, nt_commission, spread_cost
from vibe_quant.portfolio.ticks import release_tick, sizing_price, unreleased_error

if TYPE_CHECKING:
    from vibe_quant.portfolio.research import Ranker
    from vibe_quant.portfolio.sizing import Decision
    from vibe_quant.portfolio.ticks import CarryInputs


def _fee_multiplier(fee_multiplier: float) -> Decimal:
    """``Decimal(1)`` exactly for the default, so the fee hook is a no-op (R35)."""
    return Decimal(1) if fee_multiplier == 1.0 else Decimal(str(fee_multiplier))


def _fill(
    inputs: CarryInputs,
    symbol: str,
    steps: int,
    price: float,
    ts: int,
    reduce_only: bool,
    multiplier: Decimal,
) -> CarryFill:
    """One execution: price = the release tick, fee = the NT taker formula (§5), spread
    cost = :func:`spread_cost` on the exact quantity (R48)."""
    instrument = inputs.instruments[symbol]
    step = inputs.specs[symbol].step
    quantity = instrument.make_qty(Decimal(abs(steps)) * step)
    fee = nt_commission(instrument, quantity, instrument.make_price(price), multiplier)
    qty = float(Decimal(steps) * step)
    return CarryFill(
        symbol=symbol,
        ts=ts,
        steps=steps,
        qty=qty,
        price=price,
        fee=fee,
        reduce_only=reduce_only,
        spread=spread_cost(qty, price, inputs.spec.spread),
    )


def simulate_reference(
    inputs: CarryInputs,
    *,
    ranker: Ranker | None = None,
    fee_multiplier: float = 1.0,
) -> CarryRun:
    """Simulate a carry run from the shared rules alone (``engine="reference"``).

    Args:
        inputs: The preflighted run inputs.
        ranker: Optional signal replacement (T8 research null/placebo); ``None``
            uses the archived trailing funding as the signal (byte-identical to
            the no-hook run, R35). It runs after eligibility and must keep every
            excluded symbol ``None``.
        fee_multiplier: Multiplier on the NT taker commission (T8 fee stress);
            ``1.0`` is the shipped fee (byte-identical).

    Raises:
        ValueError: if ``ranker`` returns a signal for an excluded symbol.
    """
    spec = inputs.spec
    multiplier = _fee_multiplier(fee_multiplier)
    funding = run_accumulator(inputs)
    fills: list[CarryFill] = []
    book = BookIndex(fills, funding.payments)
    decisions: list[Decision] = []
    for decision_ns in inputs.decisions:
        sizing = {
            symbol: sizing_price(inputs.ticks[symbol], decision_ns) for symbol in spec.symbols
        }
        eligible = decision_signals(
            spec,
            inputs.settlements,
            sizing,
            decision_ns,
            vol_closes=inputs.vol_closes,
            quote_volume=inputs.quote_volume,
            trend_closes=inputs.trend_closes,
            listings=inputs.listings,
        )
        prices = eligible.prices
        excluded = eligible.excluded
        funding.advance(fills, decision_ns, book=book)
        equity = book.equity(spec.initial_balance, prices, decision_ns)
        signals = eligible.signals
        if ranker is not None:
            signals = ranker(decision_ns, signals)
            for symbol in excluded:
                if signals.get(symbol) is not None:
                    raise ValueError(f"ranker returned a signal for excluded {symbol}")
        current = {symbol: book.held(symbol, decision_ns) for symbol in spec.symbols}
        decision = decide(
            decision_ns,
            equity,
            signals,
            excluded,
            prices,
            current,
            inputs.specs,
            spec,
            vols=eligible.vols,
            trend_closes=inputs.trend_closes,
            regime=eligible.regime,
        )
        decisions.append(decision)
        for order in decision.orders:
            tick = release_tick(order.symbol, inputs.ticks[order.symbol], decision_ns)
            if tick is None:
                raise unreleased_error(order.symbol, decision_ns)
            fills.append(
                _fill(
                    inputs,
                    order.symbol,
                    order.steps,
                    tick.price,
                    tick.ts,
                    order.reduce_only,
                    multiplier,
                )
            )
    funding.advance(fills, spec.end_ns, book=book)
    # Stable sort on (ts, symbol): ties keep release order (a flip's close before its open).
    ordered = tuple(sorted(fills, key=lambda fill: (fill.ts, fill.symbol)))
    final_steps = {
        symbol: sum(fill.steps for fill in ordered if fill.symbol == symbol)
        for symbol in spec.symbols
    }
    return CarryRun(
        spec=spec,
        decisions=tuple(decisions),
        fills=ordered,
        payments=tuple(funding.payments),
        final_steps=final_steps,
        engine="reference",
        funding_switch_holes_upper=funding.funding_switch_holes_upper,
        funding_fallback_charges=funding.funding_fallback_charges,
    )
