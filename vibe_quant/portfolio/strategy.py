"""The funding-carry NT strategy: one time alert per decision, orders via the outbox.

Task T3 of the funding-carry epic (``design.md`` §1.3, §8-T3). Every money rule
lives in ``signal``/``sizing``/``ledger``; this class only supplies the NT side:

* one ``clock.set_time_alert_ns`` per scheduled decision ``D`` (no re-arm), whose
  callback runs with the clock at ``D`` -- after every datum with
  ``ts_init <= D`` and before any datum with ``ts_init > D`` (MEASURED);
* an NT-state cross-check at each ``D`` (R29b): ``portfolio.net_position`` must
  equal the ledger's exact ``Decimal(steps_held) * step`` and ``cache.trade_tick``
  must be the sizing tick, else ``CarryExecutionError``;
* orders queued at ``D`` in the per-symbol :class:`InstrumentOutbox` and
  submitted by that symbol's own next tick, with ``ticks.check_release`` applied.

There is no ``try``/``except`` anywhere: a raise inside an alert callback or a
trade-tick handler propagates out of ``engine.run()`` (MEASURED), which is what
the runner's count checks and the CLI rely on.
"""

from __future__ import annotations

from decimal import Decimal
from functools import partial
from typing import TYPE_CHECKING

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.trading.strategy import Strategy

from vibe_quant.portfolio.ledger import BookIndex, CarryFill, FundingAccumulator, steps_held
from vibe_quant.portfolio.outbox import InstrumentOutbox
from vibe_quant.portfolio.signal import decision_signals
from vibe_quant.portfolio.sizing import OrderIntent, decide
from vibe_quant.portfolio.spec import CarryExecutionError
from vibe_quant.portfolio.ticks import check_release, sizing_price

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nautilus_trader.common.events import TimeEvent
    from nautilus_trader.model.data import TradeTick
    from nautilus_trader.model.events import OrderDenied, OrderFilled, OrderRejected
    from nautilus_trader.model.identifiers import ClientOrderId, InstrumentId
    from nautilus_trader.model.orders import MarketOrder

    from vibe_quant.portfolio.sizing import Decision
    from vibe_quant.portfolio.ticks import CarryInputs, PriceTick


class FundingCarryConfig(StrategyConfig, frozen=True):
    """Run-level knobs live on the venue (balance, leverage), never on the strategy."""


class FundingCarryStrategy(Strategy):  # type: ignore[misc]
    """One pooled funding-carry book over N symbols, decided once a week at ``D``."""

    def __init__(self, inputs: CarryInputs, config: FundingCarryConfig | None = None) -> None:
        super().__init__(FundingCarryConfig() if config is None else config)
        self.inputs = inputs
        self.decisions: list[Decision] = []
        self.fills: list[CarryFill] = []
        self.errors: list[str] = []
        self.ticks_seen = 0
        self.outbox = InstrumentOutbox()
        self.funding = FundingAccumulator(
            inputs.calc, inputs.marks, inputs.specs, inputs.spec.start_ns
        )
        self.book = BookIndex(self.fills, self.funding.payments)
        self.planned: dict[ClientOrderId, OrderIntent] = {}
        self._submitted: dict[ClientOrderId, MarketOrder] = {}
        self._symbols: dict[InstrumentId, str] = {
            instrument.id: symbol for symbol, instrument in inputs.instruments.items()
        }

    def on_start(self) -> None:
        """Subscribe to every symbol's ticks and arm one time alert per decision."""
        for instrument in self.inputs.instruments.values():
            self.subscribe_trade_ticks(instrument.id)
        for decision_ns in self.inputs.decisions:
            self.clock.set_time_alert_ns(
                f"carry-{decision_ns}",
                decision_ns,
                partial(self._on_decision, decision_ns),
                allow_past=False,
            )

    def on_trade_tick(self, tick: TradeTick) -> None:
        """Count the tick, then release and submit this symbol's queued orders."""
        self.ticks_seen += 1
        symbol = self._symbols[tick.instrument_id]
        for queue_ns, order in self.outbox.release(symbol, tick.ts_init):
            check_release(symbol, queue_ns, tick.ts_init)
            self._submitted[order.client_order_id] = order
            self.submit_order(order)

    def on_order_filled(self, event: OrderFilled) -> None:
        """Record the fill against its planned intent; an unplanned fill fails closed (R31)."""
        intent = self.planned.get(event.client_order_id)
        if intent is None:
            raise CarryExecutionError(
                f"unplanned fill {event.client_order_id} on {event.instrument_id}"
            )
        step = self.inputs.specs[intent.symbol].step
        quantity = Decimal(str(event.last_qty))
        if quantity != Decimal(abs(intent.steps)) * step:
            self.errors.append(
                f"partial fill {event.client_order_id}: {quantity} != {abs(intent.steps)} x {step}"
            )
        steps = int(quantity / step)
        signed = steps if intent.steps > 0 else -steps
        self.fills.append(
            CarryFill(
                symbol=intent.symbol,
                ts=int(event.ts_event),
                steps=signed,
                qty=float(Decimal(signed) * step),
                price=float(event.last_px),
                fee=float(event.commission),
                reduce_only=intent.reduce_only,
            )
        )

    def on_order_denied(self, event: OrderDenied) -> None:
        self.errors.append(f"order {event.client_order_id} denied: {event.reason}")

    def on_order_rejected(self, event: OrderRejected) -> None:
        self.errors.append(f"order {event.client_order_id} rejected: {event.reason}")

    def _on_decision(self, decision_ns: int, _event: TimeEvent) -> None:
        """One decision at ``D``: cross-check NT, run the shared rules, queue the orders at ``D``."""
        sizing = {
            symbol: sizing_price(self.inputs.ticks[symbol], decision_ns)
            for symbol in self.inputs.spec.symbols
        }
        current = self._check_nt_state(decision_ns, sizing)
        eligible = decision_signals(
            self.inputs.spec,
            self.inputs.settlements,
            sizing,
            decision_ns,
            vol_closes=self.inputs.vol_closes,
            quote_volume=self.inputs.quote_volume,
        )
        prices = eligible.prices
        self.funding.advance(self.fills, decision_ns, book=self.book)
        equity = self.book.equity(self.inputs.spec.initial_balance, prices, decision_ns)
        decision = decide(
            decision_ns,
            equity,
            eligible.signals,
            eligible.excluded,
            prices,
            current,
            self.inputs.specs,
            self.inputs.spec,
            vols=eligible.vols,
        )
        self.decisions.append(decision)
        for intent in decision.orders:
            self._queue(decision_ns, intent)

    def _check_nt_state(
        self, decision_ns: int, sizing: Mapping[str, PriceTick | None]
    ) -> dict[str, int]:
        """R29b: NT's booked net position and last trade tick must match the ledger at ``D``.

        Returns:
            The signed steps held per symbol, the ledger's view the decision sizes from.

        Raises:
            CarryExecutionError: if NT's position or cached tick disagrees with the ledger.
        """
        current: dict[str, int] = {}
        for symbol in self.inputs.spec.symbols:
            instrument = self.inputs.instruments[symbol]
            steps = steps_held(self.fills, symbol, decision_ns)  # the oracle: an independent check
            current[symbol] = steps
            expected = Decimal(steps) * self.inputs.specs[symbol].step
            booked = self.portfolio.net_position(instrument.id)
            if booked != expected:
                raise CarryExecutionError(
                    f"NT net position for {symbol} at {decision_ns}: {booked} != {expected}"
                )
            tick = sizing[symbol]
            cached = self.cache.trade_tick(instrument.id)
            if tick is None:
                continue
            if (
                cached is None
                or int(cached.ts_event) != tick.ts
                or float(cached.price) != tick.price
            ):
                raise CarryExecutionError(
                    f"NT last trade tick for {symbol} at {decision_ns} is not the sizing "
                    f"tick {tick.ts} @ {tick.price}"
                )
        return current

    def _queue(self, decision_ns: int, intent: OrderIntent) -> None:
        """Build the market order for ``intent``, record it, and queue it for release."""
        instrument = self.inputs.instruments[intent.symbol]
        order = self.order_factory.market(
            instrument.id,
            OrderSide.BUY if intent.steps > 0 else OrderSide.SELL,
            instrument.make_qty(Decimal(abs(intent.steps)) * self.inputs.specs[intent.symbol].step),
            time_in_force=TimeInForce.IOC,
            reduce_only=intent.reduce_only,
        )
        self.planned[order.client_order_id] = intent
        self.outbox.queue(intent.symbol, decision_ns, order)
