"""Cross-account liquidation overlay of the carry book (design x0axf.14 §2, T6).

``liquidation_mode="cross"`` runs the Binance USDT-M cross scanner
(:func:`vibe_quant.liquidation.scan.scan_liquidations`) over a COMPLETED carry run:
one :class:`~vibe_quant.liquidation.scan.PositionView` per symbol (fills in
``run.fills`` order with cash = -fee, funding as ``(ts, -amount)``), starting balance
``spec.initial_balance``, leverage ``VENUE_LEVERAGE`` (unused by cross), the shipped
tier table, and the run's MARK bars that open before ``end_ns``. A hit is an account
bust: :class:`~vibe_quant.portfolio.spec.CarryLiquidated` (a ``liquidated`` trial);
no hit returns the scanner's notes for ``summary["liquidation"]``.

Scanner model, stated as is (a conservative model verdict, not a venue replay):

* The carry MARK preflight (60-min max gap) and the scanner's gap rule hold: a leg
  inside an allowed gap is valued at its previous close. A required missing mark or
  any scanner data error (``MissingMarkSeries``, a symbol absent from the tier table,
  an out-of-tier mark) is a ``CarryDataError`` with the original text, not a trial.
* Cross values every leg at its adverse extreme of the same minute together (low for
  longs, high for shorts), even when the extremes were not simultaneous.
* A wick in the fill minute counts and is stamped at the fill timestamp.
* A segment-wide tier error can mask an earlier hit in that segment; the scanner stops
  at the first hit and validates nothing after it.
* The overlay runs only after execution completes: an executor abort
  (``CarryEquityTooSmall``) propagates as itself, never re-classified as a bust.

``vibe_quant.liquidation`` is imported only when the overlay runs, so liquidation off
never loads it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from vibe_quant.portfolio.spec import VENUE_LEVERAGE, CarryDataError, CarryLiquidated

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from vibe_quant.data.mark_bars import MarkSeries
    from vibe_quant.liquidation.scan import PositionView
    from vibe_quant.portfolio.ledger import CarryRun
    from vibe_quant.portfolio.ticks import CarryInputs

_INSTRUMENT_SUFFIX = "-PERP.BINANCE"


class CarryEvent(NamedTuple):
    """One carry fill as the scanner's ``EventLike``: signed qty, cash = -fee."""

    ts: int
    qty: float
    price: float
    cash: float


def carry_views(run: CarryRun) -> list[PositionView]:
    """One position view per spec symbol: its fills in run order, its funding negated."""
    from vibe_quant.liquidation.scan import PositionView

    return [
        PositionView(
            key=symbol,
            symbol=symbol,
            instrument_id=f"{symbol}{_INSTRUMENT_SUFFIX}",
            events=[
                CarryEvent(fill.ts, fill.qty, fill.price, -fill.fee)
                for fill in run.fills
                if fill.symbol == symbol
            ],
            funding=[
                (payment.ts, -payment.amount)
                for payment in run.payments
                if payment.symbol == symbol
            ],
        )
        for symbol in run.spec.symbols
    ]


def require_tier_symbols(symbols: Sequence[str]) -> None:
    """Preflight: every symbol has a shipped tier table (else ``CarryDataError``)."""
    from vibe_quant.liquidation.tiers import load_tier_tables

    tables = load_tier_tables()
    for symbol in symbols:
        if symbol not in tables:
            raise CarryDataError(
                f"no liquidation tier table for {symbol} in {tables.version}"
            )


def _before(series: MarkSeries, end_ns: int) -> MarkSeries:
    """The MARK bars opening strictly before ``end_ns``."""
    stop = int(np.searchsorted(series.ts, end_ns, side="left"))
    return series._replace(
        ts=series.ts[:stop],
        low=series.low[:stop],
        high=series.high[:stop],
        close=series.close[:stop],
        open=series.open[:stop],
    )


def carry_liquidation(run: CarryRun, inputs: CarryInputs) -> dict[str, Any]:
    """Cross-scan a completed run: the scanner's notes, or ``CarryLiquidated`` on a bust.

    Raises:
        CarryLiquidated: the account's margin balance reached maintenance (``decision_ns``
            = bust ts, ``symbols`` = the sorted legs held then).
        CarryDataError: a scanner data error, with its original text.
    """
    from vibe_quant.liquidation.scan import MissingMarkSeries, scan_liquidations
    from vibe_quant.liquidation.tiers import TierTableError, UnknownSymbol, load_tier_tables

    spec = run.spec
    marks = {symbol: _before(series, spec.end_ns) for symbol, series in inputs.marks.items()}
    try:
        result = scan_liquidations(
            carry_views(run),
            marks,
            mode="cross",
            starting_balance=spec.initial_balance,
            leverage=float(VENUE_LEVERAGE),
            tables=load_tier_tables(),
        )
    except UnknownSymbol as error:
        raise CarryDataError(str(error.args[0])) from error
    except (MissingMarkSeries, TierTableError) as error:
        raise CarryDataError(str(error)) from error
    if result.bust_ts is not None:
        hit = result.hits[0]
        raise CarryLiquidated(
            sorted({hit.symbol for hit in result.hits}),
            result.bust_ts,
            hit.margin_balance,
            hit.maint_margin,
        )
    return result.to_notes(len(run.fills))
