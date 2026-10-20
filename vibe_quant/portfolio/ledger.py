"""Funding-carry book ledger: fills, funding accrual, equity (design §3, §4).

One money-ledger implementation shared by both executors (the NT engine and the
no-NT reference simulator, §1.2):

* :class:`CarryFill` — one execution of the book (signed ``steps``, so the
  exact quantity is ``float(Decimal(steps) * step)``, never a float delta).
* :func:`steps_held` — the signed step count held strictly before an instant.
* :func:`mark_at` — the MARK price a settlement is charged at: the OPEN of the
  1m MARK bar opening at ``floor_minute(S)`` (R17, fail closed).
* :class:`FundingAccumulator` — windowed, seeded accrual (§3): one window per
  decision, each seeded with the EXACT cumulative quantity, so a resized
  NETTING leg leaves no float residue and nothing is charged after it is flat
  (R18b); a settlement on a window edge is counted once, in the later window
  (R18c). Cost is O(total settlements), not O(decisions x settlements).
* :func:`equity_at` — the cash-form mark-to-market equity of the book (R10,
  R22): only events strictly before the instant count.
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
    from collections.abc import Mapping, Sequence

    from vibe_quant.data.mark_bars import MarkSeries
    from vibe_quant.portfolio.sizing import Decision, InstrumentSpec
    from vibe_quant.portfolio.spec import CarrySpec
    from vibe_quant.validation.funding import FundingCalculator

_NS_PER_MINUTE = 60_000_000_000
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


class FundingAccumulator:
    """Windowed, seeded funding accrual over one run (§3, R18b/R18c)."""

    def __init__(
        self,
        calc: FundingCalculator,
        marks: Mapping[str, MarkSeries],
        specs: Mapping[str, InstrumentSpec],
        start_ns: int,
    ) -> None:
        self._calc = calc
        self._marks = marks
        self._specs = specs
        self._last = start_ns
        self.payments: list[FundingPayment] = []

    def advance(self, fills: Sequence[CarryFill], before_ns: int) -> list[FundingPayment]:
        """Accrue the window ``[self._last, before_ns)``; return its payments.

        ``fills`` is the run's fill list (any window): the seed of each window is
        the exact quantity held before it, so no residue crosses a window.

        Raises:
            ValueError: if ``before_ns`` does not advance the window.
        """
        if before_ns <= self._last:
            raise ValueError(f"before_ns {before_ns} must be after {self._last}")
        window: list[FundingPayment] = []
        for symbol in sorted(self._specs):
            window.extend(self._advance_symbol(fills, symbol, before_ns))
        window.sort(key=lambda payment: (payment.ts, payment.symbol))
        self.payments.extend(window)
        self._last = before_ns
        return window

    def _advance_symbol(
        self, fills: Sequence[CarryFill], symbol: str, before_ns: int
    ) -> list[FundingPayment]:
        """One symbol's window: a seed fill carrying the exact held quantity, then its fills."""
        seed_qty = float(Decimal(steps_held(fills, symbol, self._last)) * self._specs[symbol].step)
        window_fills = [
            fill for fill in fills if fill.symbol == symbol and self._last <= fill.ts < before_ns
        ]
        if seed_qty == 0.0 and not window_fills:
            return []
        window_fills.sort(key=lambda fill: fill.ts)
        series = self._marks.get(symbol)
        if series is None:
            raise CarryDataError(f"no MARK series for {symbol} at {before_ns}")
        accrual = self._calc.accrue_book(
            f"{symbol}{_INSTRUMENT_SUFFIX}",
            [(self._last - 1, seed_qty), *((fill.ts, fill.qty) for fill in window_fills)],
            lambda ts: mark_at(series, symbol, ts),
            end_ns=before_ns - 1,
        )
        fallback_ts = set(accrual.fallback_timestamps)
        return [
            FundingPayment(symbol, ts, amount, ts in fallback_ts)
            for ts, amount in accrual.payments
        ]


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
