"""Funding-carry book ledger: fills, funding accrual, equity (design §3, §4).

One money-ledger implementation shared by both executors (the NT engine and the
no-NT reference simulator, §1.2):

* :class:`CarryFill` — one execution of the book (signed ``steps``, so the
  exact quantity is ``float(Decimal(steps) * step)``, never a float delta).
* :func:`steps_held` — the signed step count held strictly before an instant.
* :func:`mark_at` — the MARK price a settlement is charged at: the OPEN of the
  1m MARK bar opening at ``floor_minute(S)`` (R17, fail closed).
* :func:`hour_mark_at` — the 1h tier's as-of mark: the close of the 1h LAST bar
  completing in ``(S - 1h, S]`` (arch-c9-2 R33, fail closed).
* :class:`FundingAccumulator` — windowed, seeded accrual (§3): one window per
  decision, each seeded with the EXACT cumulative quantity, so a resized
  NETTING leg leaves no float residue and nothing is charged after it is flat
  (R18b); a settlement on a window edge is counted once, in the later window
  (R18c). Cost is O(total settlements), not O(decisions x settlements).
* :func:`equity_at` — the cash-form mark-to-market equity of the book (R10,
  R22): only events strictly before the instant count.
* :class:`BookIndex` — the incremental index both executors and the metrics read
  (arch-c9-2 R65): the same terms as :func:`equity_at` / :func:`steps_held`, found
  by bisect instead of a full rescan; those two stay as the oracles.
"""

from __future__ import annotations

import bisect
import functools
import math
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np

from vibe_quant.portfolio.spec import CarryDataError, CarryExecutionError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from vibe_quant.data.mark_bars import MarkSeries
    from vibe_quant.portfolio.bars import HourBars
    from vibe_quant.portfolio.sizing import Decision, InstrumentSpec
    from vibe_quant.portfolio.spec import CarrySpec
    from vibe_quant.validation.funding import FundingCalculator

_NS_PER_MINUTE = 60_000_000_000
_NS_PER_HOUR = 60 * _NS_PER_MINUTE
_NS_PER_MS = 1_000_000
_INSTRUMENT_SUFFIX = "-PERP.BINANCE"


@dataclass(frozen=True)
class CarryFill:
    """One execution: ``steps`` is the signed delta, ``qty`` its exact size."""

    symbol: str
    ts: int
    steps: int
    qty: float
    price: float
    fee: float
    reduce_only: bool


@dataclass(frozen=True)
class FundingPayment:
    """One settlement of one leg; ``amount`` positive = paid by the trader."""

    symbol: str
    ts: int
    amount: float
    fallback: bool


@dataclass(frozen=True)
class CarryRun:
    """The output of both executors, consumed by the metrics and parity tasks."""

    spec: CarrySpec
    decisions: tuple[Decision, ...]
    fills: tuple[CarryFill, ...]
    payments: tuple[FundingPayment, ...]
    final_steps: dict[str, int]
    engine: str


def steps_held(fills: Sequence[CarryFill], symbol: str, before_ns: int) -> int:
    """Signed step count of ``symbol`` from every fill strictly before ``before_ns``."""
    return sum(f.steps for f in fills if f.symbol == symbol and f.ts < before_ns)


def mark_at(series: MarkSeries, symbol: str, ts_ns: int) -> float:
    """MARK price at ``ts_ns``: OPEN of the 1m bar opening at ``floor_minute(ts_ns)``.

    Raises:
        CarryDataError: if no MARK bar opens at that minute (fail closed, R17).
    """
    minute = ts_ns - ts_ns % _NS_PER_MINUTE
    index = int(np.searchsorted(series.ts, minute))
    if index >= len(series.ts) or int(series.ts[index]) != minute:
        raise CarryDataError(f"no MARK bar for {symbol} opening at {minute} (settlement {ts_ns})")
    return float(series.open[index])


def hour_mark_at(bars: HourBars, symbol: str, ts_ns: int) -> float:
    """Close of the 1h bar completing at ``B`` with ``S - 1h < B <= S`` (R33).

    A bar's completion is ``ts + 1 ms`` (``ts`` = open + 1h - 1 ms). The bar ending
    exactly at ``S`` counts; one completing an hour or more before ``S`` does not.

    Raises:
        CarryDataError: no 1h close completes in ``(S - 1h, S]`` (fail closed).
    """
    index = int(np.searchsorted(bars.ts, ts_ns - _NS_PER_MS, side="right")) - 1
    if index < 0 or int(bars.ts[index]) + _NS_PER_MS <= ts_ns - _NS_PER_HOUR:
        raise CarryDataError(f"no 1h close for {symbol} in ({ts_ns - _NS_PER_HOUR}, {ts_ns}]")
    return float(bars.close[index])


def hour_mark_rule(bars: Mapping[str, HourBars]) -> Callable[[str, int], float]:
    """A :class:`FundingAccumulator` ``mark_rule`` charging at :func:`hour_mark_at`."""

    def rule(symbol: str, ts_ns: int) -> float:
        return hour_mark_at(bars[symbol], symbol, ts_ns)

    return rule


class FundingAccumulator:
    """Windowed, seeded funding accrual over one run (§3, R18b/R18c).

    ``mark_rule(symbol, ts)`` replaces the 1m MARK lookup (:func:`mark_at` over
    ``marks``); the 1h tier passes :func:`hour_mark_rule`. ``causal=True`` uses the
    causal fallback grid and counts held settlements the grid cannot see at an
    interval switch in :attr:`funding_switch_holes_upper` (never charged), reading
    no archived row at or after ``end_ns`` (the run end, required). Both default to
    the legacy funding path, unchanged. :attr:`funding_fallback_charges` counts every
    settlement charged at the fallback rate; on the causal grid that includes the
    pessimistic charges after a finer-to-coarser switch, whose backward-looking labels
    keep the finer interval until the first coarser row (SOL 2022-11-18 10/12/14:00).

    Raises:
        ValueError: ``causal`` without ``end_ns``.
    """

    def __init__(
        self,
        calc: FundingCalculator,
        marks: Mapping[str, MarkSeries],
        specs: Mapping[str, InstrumentSpec],
        start_ns: int,
        *,
        mark_rule: Callable[[str, int], float] | None = None,
        causal: bool = False,
        end_ns: int | None = None,
    ) -> None:
        if causal and end_ns is None:
            raise ValueError("causal accrual needs the run end (end_ns)")
        self._calc = calc
        self._marks = marks
        self._specs = specs
        self._last = start_ns
        self._mark_rule = mark_rule
        self._causal = causal
        self._end = end_ns
        self.payments: list[FundingPayment] = []
        self.funding_switch_holes_upper = 0
        self.funding_fallback_charges = 0

    def advance(
        self, fills: Sequence[CarryFill], before_ns: int, *, book: BookIndex | None = None
    ) -> list[FundingPayment]:
        """Accrue the window ``[self._last, before_ns)``; return its payments.

        ``fills`` is the run's fill list (any window): the seed of each window is
        the exact quantity held before it, so no residue crosses a window.
        ``book``, an index over that same list, finds the seed and the window's
        fills by bisect instead of a scan (same fills, same order).

        Raises:
            ValueError: if ``before_ns`` does not advance the window, or ``book``
                indexes another fill list.
        """
        if before_ns <= self._last:
            raise ValueError(f"before_ns {before_ns} must be after {self._last}")
        if book is not None and book.fills is not fills:
            raise ValueError("book indexes another fill list")
        window: list[FundingPayment] = []
        for symbol in sorted(self._specs):
            window.extend(self._advance_symbol(fills, symbol, before_ns, book))
        window.sort(key=lambda payment: (payment.ts, payment.symbol))
        self.payments.extend(window)
        self._last = before_ns
        return window

    def _advance_symbol(
        self, fills: Sequence[CarryFill], symbol: str, before_ns: int, book: BookIndex | None
    ) -> list[FundingPayment]:
        """One symbol's window: a seed fill carrying the exact held quantity, then its fills."""
        if book is None:
            held = steps_held(fills, symbol, self._last)
            window_fills = [
                fill for fill in fills if fill.symbol == symbol and self._last <= fill.ts < before_ns
            ]
        else:
            held = book.held(symbol, self._last)
            window_fills = book.window(symbol, self._last, before_ns)
        seed_qty = float(Decimal(held) * self._specs[symbol].step)
        if seed_qty == 0.0 and not window_fills:
            return []
        window_fills.sort(key=lambda fill: fill.ts)
        accrual = self._calc.accrue_book(
            f"{symbol}{_INSTRUMENT_SUFFIX}",
            [(self._last - 1, seed_qty), *((fill.ts, fill.qty) for fill in window_fills)],
            self._mark(symbol, before_ns),
            end_ns=before_ns - 1,
            causal=self._causal,
            horizon_ns=self._end,
        )
        self.funding_switch_holes_upper += len(accrual.switch_hole_timestamps)
        self.funding_fallback_charges += len(accrual.fallback_timestamps)
        fallback_ts = set(accrual.fallback_timestamps)
        return [
            FundingPayment(symbol, ts, amount, ts in fallback_ts)
            for ts, amount in accrual.payments
        ]

    def _mark(self, symbol: str, before_ns: int) -> Callable[[int], float]:
        """The settlement mark of ``symbol``: ``mark_rule`` or the 1m MARK series."""
        rule = self._mark_rule
        if rule is not None:
            return lambda ts: rule(symbol, ts)
        series = self._marks.get(symbol)
        if series is None:
            raise CarryDataError(f"no MARK series for {symbol} at {before_ns}")
        return lambda ts: mark_at(series, symbol, ts)


@functools.lru_cache(maxsize=65_536)
def _size_increment(qty: float, steps: int) -> Decimal:
    """The size increment a fill implies: ``repr`` round-trips ``float(Decimal(steps) * step)``."""
    return Decimal(repr(qty)) / Decimal(steps)


def _payment_ts(payment: FundingPayment) -> int:
    return payment.ts


def equity_at(
    initial_balance: float,
    fills: Sequence[CarryFill],
    payments: Sequence[FundingPayment],
    prices: Mapping[str, float],
    at_ns: int,
) -> float:
    """Cash-form equity at ``at_ns``; only events strictly before it count.

    A leg is held iff its integer step sum is non-zero; its quantity is
    ``float(Decimal(step_sum) * step)`` (R47), never a float sum of fill sizes.

    Raises:
        CarryDataError: if a leg is held and no price is given for it (fail closed).
        CarryExecutionError: if one symbol's fills imply different size increments.
    """
    cash_terms: list[float] = []
    step_sums: dict[str, int] = {}
    steps: dict[str, Decimal] = {}
    for fill in fills:
        if fill.ts >= at_ns:
            continue
        cash_terms.append(-fill.qty * fill.price - fill.fee)
        step_sums[fill.symbol] = step_sums.get(fill.symbol, 0) + fill.steps
        if fill.steps:
            # qty was built as float(Decimal(steps) * step): repr round-trips the exact step
            step = _size_increment(fill.qty, fill.steps)
            if steps.setdefault(fill.symbol, step) != step:
                raise CarryExecutionError(
                    f"inconsistent size increment for {fill.symbol}: {steps[fill.symbol]} vs {step}"
                )
    cash = math.fsum(cash_terms)
    # payments ascend by ts (FundingAccumulator writes windows in order), so those strictly
    # before at_ns are a prefix
    due = bisect.bisect_left(payments, at_ns, key=_payment_ts)
    funding = math.fsum([payment.amount for payment in payments[:due]])
    marked: list[float] = []
    for symbol, step_sum in step_sums.items():
        if step_sum == 0:
            continue
        if symbol not in prices:
            raise CarryDataError(f"no price for held {symbol} at {at_ns}")
        marked.append(float(Decimal(step_sum) * steps[symbol]) * prices[symbol])
    return initial_balance + cash - funding + math.fsum(marked)


@dataclass
class _SymbolRows:
    """One symbol's indexed fills, in append order."""

    positions: list[int]  # index of each fill in the run's fill list
    ts: list[int]
    cumulative: list[int]  # cumulative[i] = step sum of the first i fills (cumulative[0] = 0)
    fills: list[CarryFill]
    ordered: bool = True  # ts non-decreasing: the bisect paths are valid


class BookIndex:
    """Incremental equity/held index over one run's growing fill and payment lists (R65).

    ``fills`` and ``payments`` are read lazily and may only grow by appending (the
    executors' own lists, or a finished run's tuples). :meth:`equity` hands
    ``math.fsum`` exactly the term lists :func:`equity_at` builds, in the same order,
    so it is bit-identical -- order-sensitive overflow included -- and :meth:`held`
    equals :func:`steps_held`. The index paths need ordered appends; otherwise they
    fall back to the oracles (C9-13):

    * :meth:`equity`: fills globally non-decreasing in ts -> the prefix before
      ``at_ns`` by bisect (the metrics' sorted ``run.fills``); else every fill when
      all are before ``at_ns`` (an executor at its next decision); else :func:`equity_at`.
    * :meth:`held` / :meth:`window`: the symbol's fills non-decreasing in ts ->
      bisect; else the full scan.

    A fill whose size increment differs from its symbol's earlier one raises
    :class:`CarryExecutionError` (the oracle's text) on the first read after it is
    appended -- earlier than :func:`equity_at`, which raises only once the fill is
    before the instant.
    """

    def __init__(self, fills: Sequence[CarryFill], payments: Sequence[FundingPayment]) -> None:
        self.fills = fills
        self.payments = payments
        self._cash: list[float] = []
        self._ts: list[int] = []
        self._sorted = True
        self._first: dict[str, int] = {}  # symbol -> its first fill's position, in that order
        self._steps: dict[str, Decimal] = {}
        self._rows: dict[str, _SymbolRows] = {}
        self._payment_ts: list[int] = []
        self._amounts: list[float] = []
        self._sync()

    def _sync(self) -> None:
        """Index the fills and payments appended since the last read."""
        for position in range(len(self._cash), len(self.fills)):
            fill = self.fills[position]
            if fill.steps:
                step = _size_increment(fill.qty, fill.steps)
                if self._steps.setdefault(fill.symbol, step) != step:
                    raise CarryExecutionError(
                        f"inconsistent size increment for {fill.symbol}: "
                        f"{self._steps[fill.symbol]} vs {step}"
                    )
            rows = self._rows.get(fill.symbol)
            if rows is None:
                rows = self._rows[fill.symbol] = _SymbolRows([], [], [0], [])
                self._first[fill.symbol] = position
            elif fill.ts < rows.ts[-1]:
                rows.ordered = False
            if self._ts and fill.ts < self._ts[-1]:
                self._sorted = False
            rows.positions.append(position)
            rows.ts.append(fill.ts)
            rows.cumulative.append(rows.cumulative[-1] + fill.steps)
            rows.fills.append(fill)
            self._ts.append(fill.ts)
            self._cash.append(-fill.qty * fill.price - fill.fee)  # last: marks it indexed
        for payment in self.payments[len(self._amounts) :]:
            self._payment_ts.append(payment.ts)
            self._amounts.append(payment.amount)

    def equity(self, initial_balance: float, prices: Mapping[str, float], at_ns: int) -> float:
        """:func:`equity_at` over the indexed lists, bit for bit.

        Raises:
            CarryDataError: if a leg is held and no price is given for it (fail closed).
            CarryExecutionError: if one symbol's fills imply different size increments.
        """
        self._sync()
        if self._sorted:
            count = bisect.bisect_left(self._ts, at_ns)
        elif max(self._ts) < at_ns:
            count = len(self._ts)
        else:
            return equity_at(initial_balance, self.fills, self.payments, prices, at_ns)
        cash = math.fsum(self._cash[:count])
        due = bisect.bisect_left(self._payment_ts, at_ns)
        funding = math.fsum(self._amounts[:due])
        marked: list[float] = []
        for symbol, first in self._first.items():
            if first >= count:
                break
            rows = self._rows[symbol]
            step_sum = rows.cumulative[bisect.bisect_left(rows.positions, count)]
            if step_sum == 0:
                continue
            if symbol not in prices:
                raise CarryDataError(f"no price for held {symbol} at {at_ns}")
            marked.append(float(Decimal(step_sum) * self._steps[symbol]) * prices[symbol])
        return initial_balance + cash - funding + math.fsum(marked)

    def held(self, symbol: str, before_ns: int) -> int:
        """:func:`steps_held`: the signed step count of ``symbol`` before ``before_ns``."""
        self._sync()
        rows = self._rows.get(symbol)
        if rows is None:
            return 0
        if not rows.ordered:
            return steps_held(self.fills, symbol, before_ns)
        return rows.cumulative[bisect.bisect_left(rows.ts, before_ns)]

    def window(self, symbol: str, from_ns: int, before_ns: int) -> list[CarryFill]:
        """``symbol``'s fills with ``from_ns <= ts < before_ns``, in fill-list order."""
        self._sync()
        rows = self._rows.get(symbol)
        if rows is None:
            return []
        if not rows.ordered:
            return [
                fill for fill in self.fills if fill.symbol == symbol and from_ns <= fill.ts < before_ns
            ]
        return rows.fills[
            bisect.bisect_left(rows.ts, from_ns) : bisect.bisect_left(rows.ts, before_ns)
        ]
