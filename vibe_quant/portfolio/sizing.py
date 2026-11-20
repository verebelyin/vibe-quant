"""Sizing, order planning, decisions and the NT fee formula (design §2, §5).

Every money rule of the book lives here: the Decimal target-step floor, the
fail-closed min-equity check, the exact band compare, the Binance
min-notional/min-qty filter and the copy of NautilusTrader's taker commission.
Both executors (NT engine and reference simulator) call these functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from typing import TYPE_CHECKING

from nautilus_trader.model.objects import Money

from vibe_quant.portfolio.signal import inverse_vol_weights, rank_weights
from vibe_quant.portfolio.spec import (
    MIN_LEG_NOTIONAL_MULTIPLE,
    MIN_LEG_STEPS,
    CarryDataError,
    CarryEquityTooSmall,
    CarrySpec,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nautilus_trader.model.instruments import CryptoPerpetual
    from nautilus_trader.model.objects import Price, Quantity


@dataclass(frozen=True)
class InstrumentSpec:
    """The Binance filters and fee of one instrument, in Decimal where compared."""

    symbol: str
    step: Decimal
    min_quantity: Decimal
    min_notional: float
    taker_fee: Decimal

    @classmethod
    def from_instrument(cls, instrument: CryptoPerpetual) -> InstrumentSpec:
        return cls(
            symbol=str(instrument.raw_symbol),
            step=instrument.size_increment.as_decimal(),
            min_quantity=instrument.min_quantity.as_decimal(),
            min_notional=float(instrument.min_notional.as_decimal()),
            taker_fee=instrument.taker_fee,
        )


@dataclass(frozen=True)
class OrderIntent:
    """A signed order in steps (+buy / -sell); ``steps`` is never 0."""

    symbol: str
    steps: int
    reduce_only: bool


@dataclass(frozen=True)
class Decision:
    ts: int
    equity: float
    signals: dict[str, float | None]
    excluded: dict[str, str]
    weights: dict[str, float]
    prices: dict[str, float]
    target_steps: dict[str, int]
    orders: tuple[OrderIntent, ...]
    band_skips: int
    min_notional_skips: int
    rounding_drift: float
    vols: dict[str, float] | None = None


def _required(price: float, spec: InstrumentSpec) -> float:
    """The smallest tradable leg notional: 10 steps or 2x min notional (R12)."""
    price_step = Decimal(price) * spec.step
    return max(MIN_LEG_STEPS * float(price_step), MIN_LEG_NOTIONAL_MULTIPLE * spec.min_notional)


def target_steps(
    weights: Mapping[str, float],
    equity: float,
    prices: Mapping[str, float],
    specs: Mapping[str, InstrumentSpec],
) -> dict[str, int]:
    """``sign(w) * floor(Decimal(|w| E) / (Decimal(p) step))``; R11, fail-closed R12."""
    targets: dict[str, int] = {}
    for symbol, weight in weights.items():
        if weight == 0.0:
            targets[symbol] = 0
            continue
        price_step = Decimal(prices[symbol]) * specs[symbol].step
        required = _required(prices[symbol], specs[symbol])
        leg_notional = abs(weight) * equity
        if leg_notional < required:
            raise CarryEquityTooSmall(symbol, None, leg_notional, required)
        steps = int((Decimal(abs(weight) * equity) / price_step).to_integral_value(rounding=ROUND_FLOOR))
        targets[symbol] = steps if weight > 0.0 else -steps
    return targets


def _below_min(order: OrderIntent, spec: InstrumentSpec, price: float | None) -> bool:
    quantity = Decimal(abs(order.steps)) * spec.step
    if quantity < spec.min_quantity:
        return True
    if price is None:
        return False
    return quantity * Decimal(price) < Decimal(str(spec.min_notional))


def plan_orders(
    current: Mapping[str, int],
    targets: Mapping[str, int],
    prices: Mapping[str, float],
    specs: Mapping[str, InstrumentSpec],
    band: float,
) -> tuple[tuple[OrderIntent, ...], int, int]:
    """Orders (sorted by symbol, close before open), band skips, min-notional skips."""
    orders: list[OrderIntent] = []
    band_skips = 0
    min_notional_skips = 0
    band_decimal = Decimal(str(band))
    for symbol in sorted(set(current) | set(targets)):
        held = current.get(symbol, 0)
        target = targets.get(symbol, 0)
        delta = target - held
        if delta == 0:
            continue
        spec = specs[symbol]
        price = prices.get(symbol)
        if held != 0 and target != 0 and (held > 0) == (target > 0):
            if Decimal(abs(delta)) < band_decimal * abs(target):
                band_skips += 1
                continue
            resize = OrderIntent(symbol, delta, reduce_only=abs(target) < abs(held))
            if not resize.reduce_only and _below_min(resize, spec, price):
                min_notional_skips += 1
                continue
            orders.append(resize)
        elif held == 0:
            opened = OrderIntent(symbol, delta, reduce_only=False)
            if _below_min(opened, spec, price):
                min_notional_skips += 1
                continue
            orders.append(opened)
        elif target == 0:
            orders.append(OrderIntent(symbol, delta, reduce_only=True))
        else:
            orders.append(OrderIntent(symbol, -held, reduce_only=True))
            flipped = OrderIntent(symbol, target, reduce_only=False)
            if _below_min(flipped, spec, price):
                min_notional_skips += 1
                continue
            orders.append(flipped)
    return tuple(orders), band_skips, min_notional_skips


def _rounding_drift(
    weights: Mapping[str, float],
    targets: Mapping[str, int],
    prices: Mapping[str, float],
    specs: Mapping[str, InstrumentSpec],
    equity: float,
) -> float:
    terms = [
        abs(
            float(Decimal(targets.get(symbol, 0)) * specs[symbol].step) * prices.get(symbol, 0.0)
            - weight * equity
        )
        for symbol, weight in weights.items()
    ]
    return math.fsum(terms) / equity


def decide(
    ts: int,
    equity: float,
    signals: Mapping[str, float | None],
    excluded: Mapping[str, str],
    prices: Mapping[str, float],
    current: Mapping[str, int],
    specs: Mapping[str, InstrumentSpec],
    spec: CarrySpec,
    *,
    vols: Mapping[str, float] | None = None,
) -> Decision:
    """One decision: fail closed on non-positive equity (R44c), exclude unpriced
    legs (R9), rank, weight, size, plan, drift (R44).

    The ``not (equity > 0.0)`` guard runs first, for every signal state, so
    ``equity <= 0`` and ``nan`` never reach the per-leg check or the drift ratio.
    With ``spec.weighting == "inverse_vol"`` the ranked weights are re-split per
    side by :func:`inverse_vol_weights`; a picked leg whose weight underflows to
    exactly 0.0 fails closed (``CarryEquityTooSmall``), never silently unselected.
    """
    if not (equity > 0.0):
        raise CarryEquityTooSmall("<book>", ts, equity, 0.0)
    reasons = dict(excluded)
    usable: dict[str, float | None] = {}
    for symbol, value in signals.items():
        if value is None:
            usable[symbol] = None
        elif symbol in prices:
            usable[symbol] = value
        else:
            if current.get(symbol, 0) != 0:
                raise CarryDataError(f"no price for held {symbol} at {ts}")
            usable[symbol] = None
            reasons[symbol] = "no_price"
    weights = rank_weights(usable, spec.legs_divisor, spec.gross)
    if spec.weighting == "inverse_vol":
        picked = [symbol for symbol, weight in weights.items() if weight != 0.0]
        weights = inverse_vol_weights(weights, vols, spec.gross)
        for symbol in picked:
            if weights[symbol] == 0.0:
                raise CarryEquityTooSmall(symbol, ts, 0.0, _required(prices[symbol], specs[symbol]))
    try:
        targets = target_steps(weights, equity, prices, specs)
    except CarryEquityTooSmall as error:
        raise CarryEquityTooSmall(
            error.symbol, ts, error.leg_notional, error.required
        ) from error
    orders, band_skips, min_notional_skips = plan_orders(
        current, targets, prices, specs, spec.band
    )
    return Decision(
        ts=ts,
        equity=equity,
        signals=dict(signals),
        excluded=reasons,
        weights=weights,
        prices={symbol: prices[symbol] for symbol in spec.symbols if symbol in prices},
        target_steps=targets,
        orders=orders,
        band_skips=band_skips,
        min_notional_skips=min_notional_skips,
        rounding_drift=_rounding_drift(weights, targets, prices, specs, equity),
        vols=None if vols is None else dict(vols),
    )


def nt_commission(
    instrument: CryptoPerpetual,
    quantity: Quantity,
    price: Price,
    multiplier: Decimal = Decimal(1),
) -> float:
    """Copy of NT's ``MakerTakerFeeModel.get_commission`` taker branch (fee.pyx:89-106)."""
    notional = instrument.notional_value(
        quantity=quantity,
        price=price,
        use_quote_for_inverse=False,
    )
    commission = Money(
        notional.as_decimal() * instrument.taker_fee * multiplier,
        instrument.quote_currency,
    )
    return float(commission.as_decimal())
