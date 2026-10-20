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
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vibe_quant.portfolio.spec import LOOKBACK_NS, SIGNAL_SAFETY_NS
from vibe_quant.validation.funding import FUNDING_PERIOD_HOURS, missing_settlements

if TYPE_CHECKING:
    from collections.abc import Mapping


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
