"""Sizing, order planning, decisions and the NT fee formula (design §2, §5).

Every money rule of the book lives here: the Decimal target-step floor, the
fail-closed min-equity check, the exact band compare, the Binance
min-notional/min-qty filter and the copy of NautilusTrader's taker commission.
Both executors (NT engine and reference simulator) call these functions.

The trend variant (arch-c9-2) sizes with :func:`time_series_weights` (a causal
portfolio-vol scalar under the gross cap) and keeps post-plan exposure inside
one Decimal budget ``Decimal(gross) * Decimal(E)``: allocation is fitted to it
and the band yields to it.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from typing import TYPE_CHECKING

from nautilus_trader.model.objects import Money

from vibe_quant.portfolio.signal import inverse_vol_weights, rank_weights
from vibe_quant.portfolio.spec import (
    DAY_NS,
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
    book_state: str | None = None
    min_leg_skips: int = 0
    exposure_after_plan: Decimal | None = None


@dataclass(frozen=True)
class TimeSeriesBook:
    """One decision's time-series weights plus the vol-scalar audit values.

    ``state`` is ``None`` for a sized book, else the reason every weight is 0:
    ``filter_off`` (regime None or <= 0), ``flat`` (no eligible symbol, or no
    nonzero unit) or ``book_no_vol`` (book sigma 0, not finite, or a guarded
    intermediate). ``sigma`` is the daily book sigma, ``lam`` the vol scalar.
    """

    weights: dict[str, float]
    state: str | None
    sigma: float | None = None
    lam: float | None = None


def _scale(lam: float, total: float, gross: float) -> float:
    """The vol scalar under the gross cap: ``lam`` if ``lam * total <= gross``."""
    return lam if lam * total <= gross else gross / total


def _bounded(values: list[float], limit: float) -> bool:
    return all(math.isfinite(value) and abs(value) <= limit for value in values)


def _book_sigma(
    units: Mapping[str, float],
    closes: Mapping[str, Mapping[int, float]],
    first_ns: int,
    lookback: int,
) -> float | None:
    """Sample stdev of the unscaled book's ``lookback`` daily returns ending at ``B0``.

    ``R_j = fsum(u_i (C_i[B0-(j-1)D] / C_i[B0-jD] - 1))`` over the sorted legs
    with ``u_i != 0``. ``None`` on a missing, non-positive or non-finite close,
    or on a term too large for the next fsum: each product is bounded by
    ``max / (legs + 1)`` and each return and square by ``max / (lookback + 1)``,
    so no fsum can raise (opposite infinite products, or an intermediate overflow).
    """
    live = sorted(symbol for symbol, unit in units.items() if unit != 0.0)
    product_limit = sys.float_info.max / (len(live) + 1)
    limit = sys.float_info.max / (lookback + 1)
    returns: list[float] = []
    for back in range(1, lookback + 1):
        products: list[float] = []
        for symbol in live:
            path = closes.get(symbol, {})
            now = path.get(first_ns - (back - 1) * DAY_NS)
            before = path.get(first_ns - back * DAY_NS)
            if now is None or before is None or not (before > 0.0 and now > 0.0):
                return None
            products.append(units[symbol] * (now / before - 1.0))
        if not _bounded(products, product_limit):
            return None
        returns.append(math.fsum(products))
    if not _bounded(returns, limit):
        return None
    mean = math.fsum(returns) / lookback
    squares = [(value - mean) * (value - mean) for value in returns]
    if not _bounded(squares, limit):
        return None
    return math.sqrt(math.fsum(squares) / (lookback - 1))


def time_series_weights(
    signals: Mapping[str, float | None],
    closes: Mapping[str, Mapping[int, float]],
    decision_ns: int,
    spec: CarrySpec,
    regime: float | None = None,
) -> TimeSeriesBook:
    """Time-series weights with a causal portfolio-vol scalar (arch-c9-2 §2, R22-R28).

    E = the symbols with a signal, n = |E|; ``d = s`` (long_short) or
    ``max(s, 0)`` (long_flat), ``u = d / n``, ``total = fsum|u|``. The book
    sigma is today's unscaled book over the trailing ``vol_lookback_days``
    closes (only ``B0`` and earlier), ``lam = vol_target / (sigma sqrt 365)``
    and ``w = _scale(lam, total, gross) * u``. With a ``trend_filter`` a regime
    that is None or <= 0 makes every weight 0 first.
    """
    zero = dict.fromkeys(signals, 0.0)
    if spec.trend_filter is not None and not (regime is not None and regime > 0.0):
        return TimeSeriesBook(zero, "filter_off")
    eligible = sorted(symbol for symbol, value in signals.items() if value is not None)
    long_short = spec.direction == "long_short"
    units: dict[str, float] = {}
    for symbol in eligible:
        value = signals[symbol]
        assert value is not None  # eligible
        units[symbol] = (value if long_short else max(value, 0.0)) / len(eligible)
    total = math.fsum(abs(unit) for unit in units.values())
    if total == 0.0:
        return TimeSeriesBook(zero, "flat")
    first_ns = decision_ns - decision_ns % DAY_NS
    sigma = _book_sigma(units, closes, first_ns, spec.vol_lookback_days)
    if sigma is None or not (math.isfinite(sigma) and sigma > 0.0):
        return TimeSeriesBook(zero, "book_no_vol", sigma)
    assert spec.vol_target is not None  # CarrySpec requires it for trend
    lam = spec.vol_target / (sigma * math.sqrt(365.0))
    scale = _scale(lam, total, spec.gross)
    weights = dict(zero)
    for symbol, unit in units.items():
        weights[symbol] = scale * unit
    return TimeSeriesBook(weights, None, sigma, lam)


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
    trend_closes: Mapping[str, Mapping[int, float]] | None = None,
    regime: float | None = None,
) -> Decision:
    """One decision: fail closed on non-positive equity (R44c), exclude unpriced
    legs (R9), rank, weight, size, plan, drift (R44).

    The ``not (equity > 0.0)`` guard runs first, for every signal state, so
    ``equity <= 0`` and ``nan`` never reach the per-leg check or the drift ratio.
    With ``spec.weighting == "inverse_vol"`` the ranked weights are re-split per
    side by :func:`inverse_vol_weights`; a picked leg whose weight underflows to
    exactly 0.0 fails closed (``CarryEquityTooSmall``), never silently unselected.
    With ``spec.weighting == "time_series"`` the book is sized by
    :func:`_decide_time_series` from ``trend_closes`` and ``regime``.
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
    if spec.weighting == "time_series":
        if trend_closes is None:
            raise ValueError("weighting='time_series' needs trend_closes")
        return _decide_time_series(
            ts, equity, signals, reasons, usable, prices, current, specs, spec, trend_closes, regime
        )
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


def _exposure(
    holdings: Mapping[str, int], prices: Mapping[str, float], specs: Mapping[str, InstrumentSpec]
) -> Decimal:
    """``sum(Decimal(|q|) * step * Decimal(p))``: the one Decimal exposure rule (C9-9)."""
    return sum(
        (
            Decimal(abs(steps)) * specs[symbol].step * Decimal(prices[symbol])
            for symbol, steps in sorted(holdings.items())
            if steps != 0
        ),
        Decimal(0),
    )


def _after_plan(current: Mapping[str, int], orders: tuple[OrderIntent, ...]) -> dict[str, int]:
    """Holdings once every planned order fills: current plus the signed deltas."""
    after = dict(current)
    for order in orders:
        after[order.symbol] = after.get(order.symbol, 0) + order.steps
    return after


def _time_series_targets(
    weights: Mapping[str, float],
    equity: float,
    prices: Mapping[str, float],
    specs: Mapping[str, InstrumentSpec],
) -> tuple[dict[str, int], int]:
    """Target steps ``floor(Decimal(|w|) Decimal(E) / (Decimal(p) step))`` and min-leg skips.

    A leg with ``|w| E`` below its tradable size gets target 0 and counts as a
    skip (R30); the book-level size check runs before (``_decide_time_series``).
    """
    targets: dict[str, int] = {}
    skips = 0
    for symbol, weight in weights.items():
        if weight == 0.0:
            targets[symbol] = 0
            continue
        if abs(weight) * equity < _required(prices[symbol], specs[symbol]):
            targets[symbol] = 0
            skips += 1
            continue
        price_step = Decimal(prices[symbol]) * specs[symbol].step
        notional = Decimal(abs(weight)) * Decimal(equity)
        steps = int((notional / price_step).to_integral_value(rounding=ROUND_FLOOR))
        targets[symbol] = steps if weight > 0.0 else -steps
    return targets, skips


def _fit_budget(
    targets: Mapping[str, int],
    prices: Mapping[str, float],
    specs: Mapping[str, InstrumentSpec],
    budget: Decimal,
) -> dict[str, int]:
    """Targets whose :func:`_exposure` is <= ``budget`` (C9-9).

    Float weights can sum a few ulps above gross; while the floored targets
    exceed the budget, the leg with the largest exposure (ties: last symbol)
    gives up one step. Binds only at those ulp edges.
    """
    fitted = dict(targets)
    while _exposure(fitted, prices, specs) > budget:
        symbol = max(
            (name for name, steps in fitted.items() if steps != 0),
            key=lambda name: (_exposure({name: fitted[name]}, prices, specs), name),
        )
        fitted[symbol] -= 1 if fitted[symbol] > 0 else -1
    return fitted


def _cap_bypass(
    current: Mapping[str, int],
    targets: Mapping[str, int],
    orders: tuple[OrderIntent, ...],
    band_skips: int,
    prices: Mapping[str, float],
    specs: Mapping[str, InstrumentSpec],
    budget: Decimal,
) -> tuple[tuple[OrderIntent, ...], int]:
    """Emit every band-skipped same-sign reduction when post-plan exposure > budget (R31).

    A same-sign reduction is never min-notional skipped, so an unordered one was
    band-skipped. Band-skipped increases stay skipped; orders stay sorted by
    symbol (close before open).
    """
    if _exposure(_after_plan(current, orders), prices, specs) <= budget:
        return orders, band_skips
    ordered = {order.symbol for order in orders}
    extra = [
        OrderIntent(symbol, targets.get(symbol, 0) - held, reduce_only=True)
        for symbol, held in sorted(current.items())
        if symbol not in ordered
        and held != 0
        and targets.get(symbol, 0) != 0
        and (held > 0) == (targets[symbol] > 0)
        and abs(targets[symbol]) < abs(held)
    ]
    merged = tuple(sorted((*orders, *extra), key=lambda order: order.symbol))
    return merged, band_skips - len(extra)


def _decide_time_series(
    ts: int,
    equity: float,
    signals: Mapping[str, float | None],
    reasons: dict[str, str],
    usable: Mapping[str, float | None],
    prices: Mapping[str, float],
    current: Mapping[str, int],
    specs: Mapping[str, InstrumentSpec],
    spec: CarrySpec,
    trend_closes: Mapping[str, Mapping[int, float]],
    regime: float | None,
) -> Decision:
    """The time-series decision (arch-c9-2 R22-R32).

    Fails closed when ``E * gross / n`` is below any eligible leg's tradable
    size; then weights, min-leg skips, a target fit to the Decimal budget, the
    band plan, and the cap bypass. ``exposure_after_plan`` is the post-plan
    Decimal exposure / E at the sizing prices.
    """
    eligible = sorted(symbol for symbol, value in usable.items() if value is not None)
    if eligible:
        per_leg = equity * spec.gross / len(eligible)
        for symbol in eligible:
            required = _required(prices[symbol], specs[symbol])
            if per_leg < required:
                raise CarryEquityTooSmall(symbol, ts, per_leg, required)
    book = time_series_weights(usable, trend_closes, ts, spec, regime)
    budget = Decimal(spec.gross) * Decimal(equity)
    targets, min_leg_skips = _time_series_targets(book.weights, equity, prices, specs)
    targets = _fit_budget(targets, prices, specs, budget)
    orders, band_skips, min_notional_skips = plan_orders(
        current, targets, prices, specs, spec.band
    )
    orders, band_skips = _cap_bypass(
        current, targets, orders, band_skips, prices, specs, budget
    )
    exposure = _exposure(_after_plan(current, orders), prices, specs)
    return Decision(
        ts=ts,
        equity=equity,
        signals=dict(signals),
        excluded=reasons,
        weights=book.weights,
        prices={symbol: prices[symbol] for symbol in spec.symbols if symbol in prices},
        target_steps=targets,
        orders=orders,
        band_skips=band_skips,
        min_notional_skips=min_notional_skips,
        rounding_drift=_rounding_drift(book.weights, targets, prices, specs, equity),
        book_state=book.state,
        min_leg_skips=min_leg_skips,
        exposure_after_plan=exposure / Decimal(equity),
    )


DEFAULT_SPREAD = 0.0001  # full quoted spread a trend run charges until the spread table lands


def spread_cost(qty: float, price: float, spread: float) -> float:
    """The spread cost of one fill: ``|q p| * spread / 2`` (arch-c9-2 R48, ruling R-c9-4).

    ``spread`` is the FULL quoted spread as a fraction of price; the taker crosses half
    of it, through the validation convention ``SlippageEstimator.calculate`` (no volume,
    no volatility: the spread-only branch). ``spread == 0.0`` costs exactly ``0.0``.
    """
    from vibe_quant.validation.fill_model import SlippageEstimator

    return abs(qty * price) * SlippageEstimator().calculate(abs(qty), 0.0, spread=spread)


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
