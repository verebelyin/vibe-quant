"""Liquidating move of one margin setting: how far price must move against the book.

Modeled scenario (design arch-bankroll-1 §2.2): every leg is entered at price 1.0 at one
moment at equity = the starting balance ``W``; funding, fees and equity drift are ignored.
The book is the largest one the margin limit admits: ``m`` legs at the exposure cap
``N = cap * W`` plus at most one remainder leg ``r`` for the margin left
(:func:`margin_book`). The move is the adverse fraction ``x`` of the entry price at which
the book liquidates, every leg moving against it by the same ``x``.

- Isolated: each leg liquidates on its own, so the cell is the worst single leg.
- Cross, one leg: :func:`~vibe_quant.liquidation.tiers.cross_liq_price`, bit for bit.
- Cross, >= 2 legs: the exact root of the piecewise-linear margin balance
  ``D(x) = W - sum(N_i * x + MM_i(N_i * (1 -/+ x)))`` over the bracket breakpoints. ``D`` is
  concave in the notionals, so the worst admitted book is a vertex: ``m`` full legs and one
  remainder leg, searched over every choice of symbols.

Cross books of >= 2 symbols are ``path_dependent``: in the backtest a rally on an open leg
lets the margin limit admit a larger later entry, so the simultaneous book is not the worst
the run can hold. Ties keep the first book in sorted-symbol order, the short side first (as
:func:`vibe_quant.discovery.stop_cap.stop_cap`). Imports only the tier math and the margin
limit (import isolation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from itertools import combinations
from typing import TYPE_CHECKING, Literal

from vibe_quant.liquidation.tiers import (
    InsolventPosition,
    TierTable,
    TierTableError,
    TierTableExceeded,
    UnknownSymbol,
    cross_liq_price,
    isolated_liq_price,
)
from vibe_quant.margin_limit import MarginItem, check_entry_margin

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vibe_quant.liquidation.tiers import TierTables

MoveStatus = Literal["ok", "never", "at_entry", "unavailable"]
Outcome = tuple[MoveStatus, float | None, str | None]
"""(status, adverse move or None, unavailable reason or None) of one book."""

_SIDES: tuple[tuple[int, Literal["short", "long"]], ...] = ((-1, "short"), (1, "long"))
_TIER_OVERFLOW: Outcome = ("unavailable", None, "tier_overflow")
_AT_ENTRY: Outcome = ("at_entry", 0.0, None)
_NEVER: Outcome = ("never", None, None)


@dataclass(frozen=True)
class Book:
    """``positions`` legs at the exposure cap plus one leg of ``remainder`` (0.0 = none)."""

    positions: int
    remainder: float


@dataclass(frozen=True)
class Move:
    """The cell's liquidating move: ok > 0, at_entry 0.0, never / unavailable None.

    ``legs`` is the binding book as (symbol, notional): full legs, then the remainder.
    ``reason``: tier_overflow | no_tier_table | no_starting_balance | "move_error: T: msg".
    """

    status: MoveStatus
    value: float | None
    side: Literal["long", "short"] | None
    legs: tuple[tuple[str, float], ...]
    reason: str | None
    path_dependent: bool


def _dec(x: float) -> Decimal:
    return Decimal(str(x))


def margin_book(
    *, n_symbols: int, notional: float, leverage: float, max_margin_pct: float, equity: float
) -> Book:
    """The largest book the margin limit admits at ``equity``: cap-sized legs, then a remainder.

    Legs of ``notional`` are added through :func:`check_entry_margin` until one is refused or
    every symbol holds one; the remainder ``min(notional, L * p * W - m * notional)`` (exact
    Fractions, equality admitted) takes the margin left when it is positive.

    Raises:
        MarginLimitError: ``max_margin_pct``, a notional or ``leverage`` is out of domain.
    """
    leg = MarginItem(_dec(notional), _dec(leverage))
    admitted: list[MarginItem] = []
    while len(admitted) < n_symbols:
        decision = check_entry_margin(
            open_items=admitted,
            pending_items=[],
            new_item=leg,
            equity=_dec(equity),
            max_margin_pct=_dec(max_margin_pct),
        )
        if not decision.allowed:
            break
        admitted.append(leg)
    m = len(admitted)
    if m == n_symbols:
        return Book(m, 0.0)
    full = Fraction(_dec(notional))
    room = Fraction(_dec(leverage)) * Fraction(_dec(max_margin_pct)) * Fraction(_dec(equity))
    remainder = min(full, room - m * full)
    return Book(m, float(remainder) if remainder > 0 else 0.0)


def _from_liq_price(side: int, lp: float) -> Outcome:
    if side == 1 and lp <= 0:
        return _NEVER
    move = 1.0 - lp if side == 1 else lp - 1.0
    return _AT_ENTRY if move <= 0 else ("ok", move, None)


def isolated_move(side: int, notional: float, leverage: float, table: TierTable) -> Outcome:
    """One isolated leg with margin ``notional / leverage``, entered at 1.0."""
    if notional >= table.brackets[-1].cap:
        return _TIER_OVERFLOW
    try:
        lp = isolated_liq_price(side, notional, 1.0, notional / leverage, table)
    except InsolventPosition:
        return _AT_ENTRY
    except TierTableExceeded:
        return _TIER_OVERFLOW
    return _from_liq_price(side, lp)


def _adverse_notional(side: int, notional: float, x: float) -> float:
    return notional * (1.0 - x) if side == 1 else notional * (1.0 + x)


def cross_move(wallet: float, legs: Sequence[tuple[int, float, TierTable]]) -> Outcome:
    """A cross book of ``(side, notional, table)`` legs, all moving adversely by one ``x``.

    Raises:
        ValueError: a mixed long/short book still stands at ``x = 1``, where its long legs
            are worthless (the uniform-move model ends there; :func:`cell_move` builds
            single-side books only).
    """
    if any(notional >= table.brackets[-1].cap for _, notional, table in legs):
        return _TIER_OVERFLOW
    if len(legs) == 1:
        side, notional, table = legs[0]
        try:
            lp = cross_liq_price(side, notional, 1.0, wallet, 0.0, 0.0, table)
        except InsolventPosition:
            return _AT_ENTRY
        except TierTableExceeded:
            return _TIER_OVERFLOW
        return _from_liq_price(side, lp)
    if wallet - math.fsum(table.maintenance_margin(n) for _, n, table in legs) <= 0:
        return _AT_ENTRY
    all_long = all(side == 1 for side, _, _ in legs)
    has_long = any(side == 1 for side, _, _ in legs)
    if all_long and wallet - math.fsum(n for _, n, _ in legs) >= 0:
        return _NEVER
    # D(x) is linear between the x where a leg's adverse notional crosses a bracket edge.
    breaks: set[float] = {1.0} if has_long else set()
    for side, notional, table in legs:
        for bracket in table.brackets:
            for edge in (bracket.floor, bracket.cap):
                x = 1.0 - edge / notional if side == 1 else edge / notional - 1.0
                if x > 0 and (not has_long or x < 1.0):
                    breaks.add(x)
    points = [0.0, *sorted(breaks)]
    for lo, hi in zip(points, points[1:], strict=False):
        mid = (lo + hi) / 2
        c0, c1 = wallet, 0.0
        for side, notional, table in legs:
            try:
                bracket = table.bracket_for(_adverse_notional(side, notional, mid))
            except TierTableExceeded:
                return _TIER_OVERFLOW
            c0 -= notional * bracket.mmr - bracket.cum
            c1 -= notional + notional * (1 if side == -1 else -1) * bracket.mmr
        if c0 + c1 * hi <= 0:
            return ("ok", -c0 / c1, None)
    if has_long and not all_long:
        msg = "a mixed long/short cross book survives a full adverse move (x = 1)"
        raise ValueError(msg)
    return _TIER_OVERFLOW


def _rank(outcome: Outcome) -> float:
    status, value, _ = outcome
    if status == "ok" and value is not None:
        return value
    return math.inf if status == "never" else 0.0


def _books(symbols: list[str], notional: float, book: Book) -> list[tuple[tuple[str, float], ...]]:
    """Every vertex book: m full legs (combinations order) x the remainder's symbol."""
    out: list[tuple[tuple[str, float], ...]] = []
    for full in combinations(symbols, book.positions):
        legs = tuple((s, notional) for s in full)
        rest = [s for s in symbols if s not in full]
        if book.remainder > 0 and rest:
            out += [(*legs, (s, book.remainder)) for s in rest]
        else:
            out.append(legs)
    return out


def cell_move(
    *,
    mode: Literal["isolated", "cross"],
    leverage: float,
    max_margin_pct: float,
    cap: float,
    starting_balance: float | None,
    symbols: Sequence[str],
    sides: Sequence[int],
    tables: TierTables,
) -> tuple[Book | None, Move]:
    """The worst liquidating move of one margin cell over its books and allowed sides.

    Order for the min: at_entry (0.0) < ok(value) < never; strict ``<`` keeps the first.
    An ``unavailable`` book makes the cell unavailable with its reason. A missing symbol
    table is ``no_tier_table``; any other tier or domain error is ``move_error: Type: msg``
    (book None); nothing else is caught.
    """
    path_dependent = mode == "cross" and len(set(symbols)) >= 2

    def unavailable(reason: str) -> Move:
        return Move("unavailable", None, None, (), reason, path_dependent)

    if starting_balance is None or not math.isfinite(starting_balance) or starting_balance <= 0:
        return None, unavailable("no_starting_balance")
    try:
        syms = sorted(set(symbols))
        allowed = [(v, name) for v, name in _SIDES if v in sides]
        if not syms or not allowed:
            msg = f"a liquidating move needs symbols and sides, got {syms!r}, {list(sides)!r}"
            raise ValueError(msg)
        by_symbol = {s: tables[s] for s in syms}
        notional = cap * starting_balance
        book = margin_book(
            n_symbols=len(syms),
            notional=notional,
            leverage=leverage,
            max_margin_pct=max_margin_pct,
            equity=starting_balance,
        )
        books: list[tuple[tuple[str, float], ...]]
        if mode == "isolated":
            leg = notional if book.positions >= 1 else book.remainder
            books = [((s, leg),) for s in syms]
        else:
            books = _books(syms, notional, book)
        best: tuple[Outcome, Literal["long", "short"], tuple[tuple[str, float], ...]] | None
        best = None
        for legs in books:
            for side, name in allowed:
                if mode == "isolated":
                    ((symbol, leg_notional),) = legs
                    outcome = isolated_move(side, leg_notional, leverage, by_symbol[symbol])
                else:
                    outcome = cross_move(
                        starting_balance, [(side, n, by_symbol[s]) for s, n in legs]
                    )
                if outcome[0] == "unavailable":
                    return book, unavailable(str(outcome[2]))
                if best is None or _rank(outcome) < _rank(best[0]):
                    best = (outcome, name, legs)
    except UnknownSymbol:
        return None, unavailable("no_tier_table")
    except (TierTableError, ValueError) as exc:  # MarginLimitError is a ValueError
        return None, unavailable(f"move_error: {type(exc).__name__}: {exc}")
    assert best is not None  # books and sides are non-empty above
    (status, value, _), side_name, legs = best
    return book, Move(status, value, side_name, legs, None, path_dependent)
