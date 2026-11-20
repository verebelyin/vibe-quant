"""Trailing funding signal and the cross-sectional ranking (design §2).

``trailing_funding`` sums the raw per-settlement rates over the fixed calendar
window ``(L - 7 d, L]``, ``L = D - 60 s``: Binance already scales each rate by
``8 / N``, so a 4h symbol's half-size rates and an 8h symbol's rates give
comparable 7-day totals (R4). A symbol is rankable only when every settlement
its OWN interval expects in the window has an archived row within 60 s and a
sizing tick exists; otherwise it is excluded that week, never forward-filled.
The interval rule is the funding accrual's own
(:func:`vibe_quant.validation.funding.missing_settlements`), so a missing row on
a 2h/4h symbol is a ``funding_hole`` here too (R34).

:func:`decision_signals` is the one eligibility rule per decision: both
executors call it, so neither keeps its own copy of the exclusion loop.
"""

from __future__ import annotations

import bisect
import math
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vibe_quant.portfolio.spec import DAY_NS, LOOKBACK_NS, SIGNAL_SAFETY_NS
from vibe_quant.validation.funding import FUNDING_PERIOD_HOURS, missing_settlements

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vibe_quant.portfolio.spec import CarrySpec
    from vibe_quant.portfolio.ticks import PriceTick


@dataclass(frozen=True)
class Settlements:
    """Archived settlement times (int ns, ascending), their rates and intervals.

    ``hours`` is the per-row settlement interval in hours, aligned with
    ``times_ns``. ``None`` means 8h for every row -- today's fixed-grid rule --
    so a ``Settlements`` built without it (every fixture that predates the
    field) keeps that rule.
    """

    times_ns: tuple[int, ...]
    rates: tuple[float, ...]
    hours: tuple[int, ...] | None = None
    # trailing_funding memo by decision_ns: the fields above are immutable, so the value is
    # a pure function of the decision and every research draw re-asks the same ~225.
    _trailing: dict[int, float | None] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.hours is not None and len(self.hours) != len(self.times_ns):
            raise ValueError(
                f"Settlements.hours has {len(self.hours)} entries "
                f"for {len(self.times_ns)} settlements"
            )


def carry_hours(hours: tuple[int, ...]) -> tuple[int, ...] | None:
    """``hours`` as a ``Settlements`` field: None when every row is 8h (or none).

    Collapsing the all-8h case keeps ``hours=None`` -- the documented default
    rule -- for the real archive, for a stubbed or empty one, and for every test
    fixture built before the field existed.
    """
    if not hours or all(h == FUNDING_PERIOD_HOURS for h in hours):
        return None
    return hours


def trailing_funding(settlements: Settlements, decision_ns: int) -> float | None:
    """7-day funding sum for ``decision_ns``; ``None`` = an expected row is missing."""
    memo = settlements._trailing
    if decision_ns not in memo:
        memo[decision_ns] = _trailing_funding(settlements, decision_ns)
    return memo[decision_ns]


def _trailing_funding(settlements: Settlements, decision_ns: int) -> float | None:
    upper_ns = decision_ns - SIGNAL_SAFETY_NS
    lower_ns = upper_ns - LOOKBACK_NS
    hours = settlements.hours or (FUNDING_PERIOD_HOURS,) * len(settlements.times_ns)
    if missing_settlements(settlements.times_ns, hours, lower_ns, upper_ns, SIGNAL_SAFETY_NS):
        return None
    first = bisect.bisect_right(settlements.times_ns, lower_ns)
    last = bisect.bisect_right(settlements.times_ns, upper_ns)
    return math.fsum(settlements.rates[first:last])


def rank_weights(
    signals: Mapping[str, float | None],
    legs_divisor: int,
    gross: float,
) -> dict[str, float]:
    """Long the lowest ``k`` signals, short the highest; ``k = max(1, n // divisor)``."""
    ranked = sorted((value, symbol) for symbol, value in signals.items() if value is not None)
    weights: dict[str, float] = dict.fromkeys(signals, 0.0)
    if len(ranked) < 2:
        return weights
    legs = max(1, len(ranked) // legs_divisor)
    each = gross / (2 * legs)
    for _, symbol in ranked[:legs]:
        weights[symbol] = each
    for _, symbol in ranked[-legs:]:
        weights[symbol] = -each
    return weights


def trailing_vol(closes: Mapping[int, float], decision_ns: int, lookback_days: int) -> float | None:
    """Sample stdev of the ``L`` daily close-to-close returns ending at ``B0 = D - D % DAY``.

    ``closes`` maps a UTC day boundary (ns) to its close (``ticks.boundary_prices``),
    so no key after ``B0`` is read. ``None`` -- the symbol is ``no_vol`` -- on a
    missing, non-positive or non-finite close, a non-finite (overflowed)
    intermediate (one that would overflow an fsum included), or a sigma that is
    not finite and > 0. Never forward-filled.
    """
    first_ns = decision_ns - decision_ns % DAY_NS
    path: list[float] = []
    for back in range(lookback_days, -1, -1):
        close = closes.get(first_ns - back * DAY_NS)
        if close is None or not math.isfinite(close) or close <= 0.0:
            return None
        path.append(close)
    # each fsum term bounded by max / (L + 1) (max / L can round up), so neither fsum can raise OverflowError
    limit = sys.float_info.max / (lookback_days + 1)
    returns = [path[i] / path[i - 1] - 1.0 for i in range(1, len(path))]
    if not all(math.isfinite(value) and abs(value) <= limit for value in returns):
        return None
    mean = math.fsum(returns) / lookback_days
    squares = [(value - mean) * (value - mean) for value in returns]
    if not all(math.isfinite(value) and value <= limit for value in squares):
        return None
    sigma = math.sqrt(math.fsum(squares) / (lookback_days - 1))
    if not math.isfinite(sigma) or sigma <= 0.0:
        return None
    return sigma


QUOTE_VOLUME_DAYS = 7


def trailing_quote_volume(days: Mapping[int, float], decision_ns: int) -> float | None:
    """Mean daily 1m quote volume over the 7 UTC days ``[B0 - 7 d, B0)``, ``B0 = D - D % DAY``.

    ``days`` maps a UTC day start (ns) to its archived 1m quote-volume sum; a
    partial day (a listing day) counts its observed sum. ``None`` -- the
    symbol is a ``volume_hole`` -- when any of the 7 days is absent. A sum
    that overflows falls back to ``fsum(s / 7)``, so ordinary inputs stay
    bit-identical.
    """
    first_ns = decision_ns - decision_ns % DAY_NS
    sums: list[float] = []
    for back in range(QUOTE_VOLUME_DAYS, 0, -1):
        total = days.get(first_ns - back * DAY_NS)
        if total is None:
            return None
        sums.append(total)
    try:
        return math.fsum(sums) / QUOTE_VOLUME_DAYS
    except OverflowError:
        return math.fsum(total / QUOTE_VOLUME_DAYS for total in sums)


def inverse_vol_weights(
    weights: Mapping[str, float],
    vols: Mapping[str, float] | None,
    gross: float,
) -> dict[str, float]:
    """Per side, re-split ``gross / 2`` in proportion to ``1 / sigma`` (dollar-neutral).

    A side with one leg keeps its ``rank_weights`` value bit for bit. A side with
    two or more legs gets ``w_i = (gross/2) * sign * (s_min/s_i) / fsum(s_min/s_j)``;
    the ratios are in ``[0, 1]``, so nothing overflows. A picked leg without a
    finite sigma > 0 is a programming error (``decision_signals`` excludes it as
    ``no_vol`` first).
    """
    result = dict(weights)
    for sign in (1.0, -1.0):
        side = [symbol for symbol, weight in weights.items() if weight * sign > 0.0]
        for symbol in side:
            sigma = None if vols is None else vols.get(symbol)
            if sigma is None or not math.isfinite(sigma) or sigma <= 0.0:
                raise ValueError(f"inverse_vol weighting needs a vol for {symbol}")
        if len(side) < 2:
            continue
        assert vols is not None  # checked per leg above
        smallest = min(vols[symbol] for symbol in side)
        ratios = {symbol: smallest / vols[symbol] for symbol in side}
        total = math.fsum(ratios.values())
        half = gross / 2
        for symbol in side:
            result[symbol] = sign * (half * ratios[symbol] / total)
    return result


@dataclass(frozen=True)
class DecisionSignals:
    """One decision's ranking inputs: signal, exclusion reason and sizing price per symbol.

    ``vols`` is ``None`` unless the spec weights by inverse vol; then it holds the
    trailing sigma of every symbol that stays rankable.
    """

    signals: dict[str, float | None]
    excluded: dict[str, str]
    prices: dict[str, float]
    vols: dict[str, float] | None = None


def decision_signals(
    spec: CarrySpec,
    settlements: Mapping[str, Settlements],
    sizing: Mapping[str, PriceTick | None],
    decision_ns: int,
    *,
    vol_closes: Mapping[str, Mapping[int, float]] | None = None,
    quote_volume: Mapping[str, Mapping[int, float]] | None = None,
) -> DecisionSignals:
    """The eligibility rule of one decision, shared by every executor.

    Call contract (stable for any caller, paper/live included)::

        sizing = {s: sizing_price(ticks[s], decision_ns) for s in spec.symbols}
        decision_signals(spec, settlements, sizing, decision_ns)

    These four positional parameters never change; variant inputs arrive as
    keyword-only arguments.

    Per symbol, in ``spec.symbols`` order: a missing expected settlement is a
    ``funding_hole``, checked first; otherwise no sizing tick is ``no_price``.
    An excluded symbol's signal is ``None``. ``prices`` keeps every symbol with
    a sizing tick, excluded or not, so a held excluded leg can still be marked
    and closed. Total over data gaps: they give ``None`` or exclusions, never
    an exception.

    With ``spec.min_quote_volume`` set, a still-rankable symbol without all 7
    days of ``quote_volume`` is a ``volume_hole``, and one whose
    :func:`trailing_quote_volume` is below the threshold is ``illiquid``
    (equality passes). With ``spec.weighting == "inverse_vol"`` a still-rankable
    symbol without a :func:`trailing_vol` over ``vol_closes`` is ``no_vol``
    (checked last).

    Raises:
        ValueError: if a variant is on and its keyword map is ``None``.
    """
    inverse_vol = spec.weighting == "inverse_vol"
    if inverse_vol and vol_closes is None:
        raise ValueError("inverse_vol weighting needs CarryInputs.vol_closes")
    threshold = spec.min_quote_volume
    if threshold is not None and quote_volume is None:
        raise ValueError("min_quote_volume needs CarryInputs.quote_volume")
    signals: dict[str, float | None] = {}
    excluded: dict[str, str] = {}
    prices: dict[str, float] = {}
    vols: dict[str, float] | None = {} if inverse_vol else None
    for symbol in spec.symbols:
        value = trailing_funding(settlements[symbol], decision_ns)
        tick = sizing[symbol]
        if tick is not None:
            prices[symbol] = tick.price
        if value is None:
            signals[symbol] = None
            excluded[symbol] = "funding_hole"
        elif tick is None:
            signals[symbol] = None
            excluded[symbol] = "no_price"
        elif threshold is not None and quote_volume is not None and (
            reason := _liquidity(quote_volume.get(symbol, {}), decision_ns, threshold)
        ):
            signals[symbol] = None
            excluded[symbol] = reason
        else:
            signals[symbol] = value
            if vols is not None and vol_closes is not None:
                sigma = trailing_vol(vol_closes.get(symbol, {}), decision_ns, spec.vol_lookback_days)
                if sigma is None:
                    signals[symbol] = None
                    excluded[symbol] = "no_vol"
                else:
                    vols[symbol] = sigma
    return DecisionSignals(signals, excluded, prices, vols)


def _liquidity(days: Mapping[int, float], decision_ns: int, threshold: float) -> str | None:
    """``volume_hole``, ``illiquid`` (mean < threshold) or ``None`` when liquid."""
    mean = trailing_quote_volume(days, decision_ns)
    if mean is None:
        return "volume_hole"
    return "illiquid" if mean < threshold else None
