"""The ONE margin rule shared by the paper guard and the compiled backtest template.

Stdlib only, and deliberately outside :mod:`vibe_quant.paper` and :mod:`vibe_quant.dsl`: the
paper guard and the generated strategy code both run this rule, and neither may pull the
other's stack through it (import isolation, like :mod:`vibe_quant.sizing_settings`).

Margin is ``notional / leverage`` per open position, per pending entry and for the candidate
entry, and the rule is:

* ``sum(open + pending margins) + new margin <= max_margin_pct * equity``. Equality is
  ALLOWED: an entry that lands exactly on the limit is fully marginable.
* The leverage of an item is the VENUE-VERIFIED value -- paper/live: what
  ``account.leverage(instrument_id)`` recorded from the venue's ``symbolConfig``; a backtest:
  the run leverage. An item whose leverage is unverified (``None``) makes the check REFUSE:
  an unknown leverage is never guessed (DR1), and a request/resolved leverage is not a
  substitute for a verified one.
* Arithmetic is EXACT. Margins are summed as :class:`~fractions.Fraction` and compared against
  ``Fraction(max_margin_pct) * Fraction(equity)``, so a limit no decimal can represent --
  ``1000/3 + 500/3`` against ``500`` -- still lands exactly on the boundary instead of a
  rounded one. The :class:`Decimal` values carried on the decision are for display only.

The same rule serves two very different callers, so it takes plain records and imports
nothing engine-specific:

* :func:`check_entry_margin` takes :class:`MarginItem` values for the open positions, the
  pending entries and the candidate entry;
* :func:`unique_pending_entries` is the ONE dedup of the pending set (B4): Nautilus keeps
  ``PENDING_CANCEL``/``PENDING_UPDATE`` orders in BOTH ``orders_open`` and
  ``orders_inflight``, so summing those two lists double-counts them. It is duck-typed (no
  NT import) and drops reduce-only orders, which were never entry margin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Any

DEFAULT_PAPER_MAX_MARGIN_PCT = Decimal("0.5")
"""The paper/live cushion when neither the run nor the launch request set one: 50% of equity."""

MIN_LEVERAGE = Decimal(1)
"""Leverage below 1x is a venue error, not a margin input."""

MAX_LEVERAGE = Decimal(125)
"""The highest leverage the venues behind this platform expose."""

Number = Decimal | int | float
"""A margin input. Callers pass :class:`Decimal`; ints/floats are normalized exactly."""


class MarginLimitError(ValueError):
    """A margin input is outside its domain (never silently clamped or defaulted)."""


@dataclass(frozen=True)
class MarginItem:
    """One open position, pending entry or the candidate entry.

    ``notional`` is the position/order notional (``abs(qty) * price * multiplier``),
    ``leverage`` is the verified leverage or ``None`` when it could not be verified (which
    refuses the entry), and ``label`` names the instrument in the refusal reason.
    """

    notional: Number
    leverage: Number | None
    label: str = ""


@dataclass(frozen=True)
class MarginDecision:
    """The verdict plus the numbers behind it (``in_use``/``limit`` for display only)."""

    allowed: bool
    in_use: Decimal
    new_margin: Decimal
    limit: Decimal
    reason: str | None


def check_entry_margin(
    *,
    open_items: Sequence[MarginItem],
    pending_items: Sequence[MarginItem],
    new_item: MarginItem,
    equity: Number | None,
    max_margin_pct: Number,
) -> MarginDecision:
    """Whether one candidate entry fits inside the run's margin limit.

    Refusal reasons, in the order they are checked: an unverified leverage (DR1) outranks
    everything, then an unknown equity, then a non-positive equity, then the limit itself.
    A refusal still reports the margins it could measure; an unmeasurable item (unverified
    leverage) contributes nothing rather than a guess.

    Raises:
        MarginLimitError: ``max_margin_pct`` is not in ``(0, 1]``, a notional is negative, or
            a leverage is present but outside ``[1, 125]``.
    """
    pct = _validated_pct(max_margin_pct)
    equity_value = _validated_equity(equity)
    open_values = list(open_items)
    pending_values = list(pending_items)
    for item in (*open_values, *pending_values, new_item):
        _check_notional(item.notional)
        _check_leverage(item.leverage)

    in_use_fraction = _margin_fraction(open_values) + _margin_fraction(pending_values)
    new_fraction = _item_margin_fraction(new_item)
    in_use = _display(in_use_fraction)
    new_margin = _display(new_fraction)
    limit = Decimal(0) if equity_value is None else pct * equity_value

    for item in (*open_values, *pending_values, new_item):
        if item.leverage is None:
            return MarginDecision(
                allowed=False,
                in_use=in_use,
                new_margin=new_margin,
                limit=limit,
                reason=f"leverage unverified for {item.label}",
            )
    if equity_value is None:
        return MarginDecision(False, in_use, new_margin, limit, "equity unknown")
    if equity_value <= 0:
        return MarginDecision(False, in_use, new_margin, limit, f"equity {equity_value} <= 0")
    if in_use_fraction + new_fraction > Fraction(pct) * Fraction(equity_value):
        reason = (
            f"margin in use {in_use} + new {new_margin} > limit {limit} "
            f"({_percent_display(pct)}% of equity {equity_value})"
        )
        return MarginDecision(False, in_use, new_margin, limit, reason)
    return MarginDecision(True, in_use, new_margin, limit, None)


def unique_pending_entries(
    open_orders: Iterable[Any],
    inflight_orders: Iterable[Any],
    *,
    exclude_client_order_id: object,
) -> list[Any]:
    """The ONE union of the pending entry sets: each ``client_order_id`` counted once.

    Nautilus reports a ``PENDING_CANCEL``/``PENDING_UPDATE`` order in both ``orders_open``
    and ``orders_inflight`` (``model/orders/base.pyx``), so the two sets are unioned rather
    than concatenated. The first occurrence wins, reduce-only orders are dropped (they are
    exits, never entry margin) and ``exclude_client_order_id`` drops the entry being checked.
    Duck-typed on ``.client_order_id``/``.is_reduce_only``: no Nautilus import.
    """
    seen: set[Any] = set()
    pending: list[Any] = []
    for order in (*open_orders, *inflight_orders):
        if order.is_reduce_only:
            continue
        client_order_id = order.client_order_id
        if client_order_id == exclude_client_order_id or client_order_id in seen:
            continue
        seen.add(client_order_id)
        pending.append(order)
    return pending


def _as_decimal(value: object) -> Decimal | None:
    """``value`` as an exact Decimal, or ``None`` when it is not a finite number.

    The type is checked BEFORE any conversion: a bool is an int subclass but is not a number
    here, and ``float('nan')``/``float('inf')`` are rejected rather than turned into a NaN
    Decimal (which would make every later comparison silently False). A float goes through
    ``str``, so ``0.1`` means the decimal a caller wrote, not its binary expansion.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value)) if math.isfinite(value) else None
    return None


def _validated_pct(value: object) -> Decimal:
    pct = _as_decimal(value)
    if pct is None or not Decimal(0) < pct <= Decimal(1):
        raise MarginLimitError(f"max_margin_pct must be a number in (0, 1], got {value!r}")
    return pct


def _validated_equity(value: object) -> Decimal | None:
    if value is None:
        return None
    equity = _as_decimal(value)
    if equity is None:
        raise MarginLimitError(f"equity must be a number or None, got {value!r}")
    return equity


def _check_notional(value: object) -> None:
    notional = _as_decimal(value)
    if notional is None or notional < 0:
        raise MarginLimitError(f"notional must be a number >= 0, got {value!r}")


def _check_leverage(value: object) -> None:
    if value is None:
        return
    leverage = _as_decimal(value)
    if leverage is None or not MIN_LEVERAGE <= leverage <= MAX_LEVERAGE:
        raise MarginLimitError(f"leverage must be a number in [1, 125] or None, got {value!r}")


def _item_margin_fraction(item: MarginItem) -> Fraction:
    """``notional / leverage`` exactly; 0 for an item whose leverage is unverified."""
    notional = _as_decimal(item.notional)
    leverage = _as_decimal(item.leverage)
    if notional is None or leverage is None:
        return Fraction(0)
    return Fraction(notional) / Fraction(leverage)


def _margin_fraction(items: Sequence[MarginItem]) -> Fraction:
    total = Fraction(0)
    for item in items:
        total += _item_margin_fraction(item)
    return total


def _display(fraction: Fraction) -> Decimal:
    """The display value of an exact margin total (rounded only if non-terminating)."""
    return Decimal(fraction.numerator) / Decimal(fraction.denominator)


def _percent_display(pct: Decimal) -> str:
    """A percentage without a trailing ``.0``: ``0.5`` -> ``50``, ``0.005`` -> ``0.5``."""
    percent = pct * 100
    if percent == percent.to_integral_value():
        return str(int(percent))
    return str(percent.normalize())
