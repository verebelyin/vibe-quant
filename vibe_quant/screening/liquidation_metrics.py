"""Per-trade accounting of a screening run after the liquidation overlay hit something.

``NTScreeningRunner._extract_metrics`` keeps its NT-based path (byte-identical) for
``liquidation_mode="off"`` and for an identity overlay. When the overlay liquidated or
dropped a position, NT's own figures (``total_positions``, ``PnL%``, win rate, return
moments) still describe the untruncated run, so every trade-level number is rebuilt from
the same inputs as the NT path -- the trade lists and sums here, the profit factor and win
rate in the runner from the rebuilt ``trade_pnls``:

* a dropped position (a cross bust happened before it opened) is not a trade at all: no
  fees, funding, PnL or count;
* a liquidated position is a trade closed at the venue fill (``OverlayResult.exit_override``,
  never the ledger's synthetic price). Its net PnL is the kept fills' realized PnL
  (commissions of the kept fills only) minus the liquidation loss
  (``LiquidationHit.loss`` = price loss + clearance fee, with the insurance fund's part
  already excluded). Its funding is the scan's funding input truncated at the hit (the same
  ``<= ts`` rule as the overlay's truncation), never re-accrued: an open position was funded
  to its valuation horizon, so a hit after the horizon is not charged the settlements in
  between. A liquidated position that NT still holds open is never marked to market;
* every other position is accounted exactly as in the NT path (closed realized PnL, or the
  end-of-run mark of a still-open one) with the scan's funding input (to its close or its
  mark, which is what the NT path accrues).

``total_fees`` = commissions + the venue's clearance fee of each liquidation (a real cost the
account paid), so gross PnL, fee drag and the fee rate see it. The net PnL is unchanged by this
classification (``LiquidationHit.loss`` already contains the clearance fee). The runner also
reports the clearance total separately as ``liquidation_forfeited``: it OVERLAPS
``total_fees``, never add the two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from vibe_quant.liquidation.exit_timing import RestingExit
from vibe_quant.liquidation.overlay import default_symbol, enum_name, position_views
from vibe_quant.metrics import LedgerBook
from vibe_quant.validation.extraction import accrue_position_funding
from vibe_quant.validation.funding import FundingAccrual

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from vibe_quant.liquidation.overlay import OverlayResult
    from vibe_quant.metrics import LedgerEvent
    from vibe_quant.validation.extraction import OpenPositionMark
    from vibe_quant.validation.funding import FundingCalculator

__all__ = [
    "LiquidationTrades",
    "liquidation_trades",
    "position_key",
    "resting_exit_inputs",
]

_ONE_MS = 1_000_000
# NautilusTrader's OrderEvent subclass names that mark an order live (R21). The order's
# ``events`` carry no order-type enum, only the concrete class name.
_ACTIVE_EVENT_NAMES = frozenset({"OrderAccepted", "OrderUpdated"})
_MARKET = "MARKET"
_REFINED_ORDER_TYPES = ("STOP_MARKET", "LIMIT")


def position_key(pos: Any) -> str:
    """The overlay's key of a position (``<position id>@<ts_opened>``)."""
    return f"{pos.id}@{int(pos.ts_opened)}"


def resting_exit_inputs(
    engine: Any, positions: Iterable[Any], execution_bar_ns: int
) -> list[RestingExit]:
    """The boundary resting fills of ``positions``, with the order details R19/R21 need.

    A *boundary* fill is a STOP_MARKET or LIMIT whose ``ts_event`` sits on the execution
    (strategy) bar's ``ts_init``: ``(ts_event + 1 ms) % execution_bar_ns == 0`` (R18). The
    screening engine fills a resting exit at the bar close, a whole bar after the 1m bar that
    reached it, which is what :func:`~vibe_quant.liquidation.exit_timing.refine_resting_exits`
    corrects. MARKET fills (the entry) and off-boundary resting fills (triggered by fill ticks)
    are left alone (R18).

    ``level`` is the order's ``trigger_price`` (STOP_MARKET) or ``price`` (LIMIT), never the
    fill price (R19); ``active_from`` is its last ``OrderAccepted``/``OrderUpdated`` ``ts_event``
    at or before the fill (R21). ``order_id`` is the client order id string (R28b).

    Raises:
        ValueError: a boundary fill whose order is missing from the engine cache, whose order
            has no accepted/updated event at or before the fill, or whose order type is an
            unsupported resting type (R27). The message names the position key and the client
            order id; the caller fails the run closed.
    """
    # ``execution_bar_ns`` needs no check here: only mode-on screening with a coarser-than-1m
    # execution bar (E > 60 s, an int) reaches this call (nt_runner._boundary_exit_scan_ts, R17).
    cache = engine.kernel.cache
    out: list[RestingExit] = []
    for pos in positions:
        key = position_key(pos)
        symbol = default_symbol(str(pos.instrument_id))
        for index, fill in enumerate(pos.events):
            ts = int(fill.ts_event)
            if (ts + _ONE_MS) % execution_bar_ns != 0:
                continue
            order_type = enum_name(fill.order_type)
            if order_type == _MARKET:
                continue
            order_id = str(fill.client_order_id)
            if order_type not in _REFINED_ORDER_TYPES:
                raise ValueError(
                    f"{key} fill {index}: order {order_id} is a {order_type} fill on the "
                    f"execution-bar boundary; only {' and '.join(_REFINED_ORDER_TYPES)} are "
                    f"refined to their 1m bar"
                )
            order = cache.order(fill.client_order_id)
            if order is None:
                raise ValueError(
                    f"{key} fill {index}: order {order_id} is not in the engine cache"
                )
            active_from = _active_from(order, ts)
            if active_from is None:
                raise ValueError(
                    f"{key} fill {index}: order {order_id} has no OrderAccepted/OrderUpdated "
                    f"at or before the fill ts {ts}"
                )
            kind: Literal["stop", "limit"] = "stop" if order_type == "STOP_MARKET" else "limit"
            side = enum_name(fill.order_side)
            if side not in ("BUY", "SELL"):
                raise ValueError(f"{key} fill {index}: unknown order side {side!r}")
            # True -> the trigger watches the bar LOW (a SELL stop / a BUY limit)
            falls = (side == "SELL") == (kind == "stop")
            level = float(order.trigger_price if kind == "stop" else order.price)
            out.append(
                RestingExit(
                    key=key,
                    fill_index=index,
                    order_id=order_id,
                    symbol=symbol,
                    ts=ts,
                    kind=kind,
                    falls=falls,
                    level=level,
                    active_from=active_from,
                )
            )
    return out


def _active_from(order: Any, ts: int) -> int | None:
    """The order's last ``OrderAccepted``/``OrderUpdated`` ``ts_event`` at or before ``ts`` (R21)."""
    active: int | None = None
    for event in order.events:
        if type(event).__name__ not in _ACTIVE_EVENT_NAMES:
            continue
        event_ts = int(event.ts_event)
        if event_ts <= ts and (active is None or event_ts > active):
            active = event_ts
    return active


@dataclass
class LiquidationTrades:
    """Trade-level sums and lists of a run with the liquidation overlay applied."""

    total_fees: float = 0.0
    total_funding: float = 0.0
    funding_fallbacks: int = 0
    trade_pnls: list[float] = field(default_factory=list)  # net of costs incl. funding
    closed_net: list[tuple[int, float]] = field(default_factory=list)
    cash_events: list[tuple[int, float]] = field(default_factory=list)
    # per-trade return on entry notional for the return moments (closed / liquidated only)
    returns: list[float] = field(default_factory=list)
    n_closed: int = 0


def liquidation_trades(
    overlay: OverlayResult,
    trade_inputs: Iterable[tuple[Any, OpenPositionMark | None]],
    scan_funding: Mapping[str, FundingAccrual],
    funding_calc: FundingCalculator,
) -> LiquidationTrades:
    """Rebuild the per-trade accounting of the screening run from the overlay result.

    ``scan_funding`` is the funding the scan was given per position key (paid > 0);
    ``funding_calc`` only recounts the fallback settlements a hit's truncation keeps.
    """
    out = LiquidationTrades()
    for pos, mark in trade_inputs:
        key = position_key(pos)
        if key in overlay.dropped_keys:
            continue
        hit = overlay.hits_by_key.get(key)
        accrual = scan_funding[key]
        if hit is not None:
            exit_ns, _venue_price = overlay.exit_override[key]
            # position_views builds LedgerEvents (the EventLike protocol is their duck type)
            kept = [
                cast("LedgerEvent", e) for e in position_views([pos])[0].events if e.ts <= hit.ts
            ]
            book = LedgerBook()
            for event in kept:
                book.apply(event)
            fees = sum(abs(e.cash) for e in kept) + hit.clearance_fee
            realized = book.realized - hit.loss
            closed = True
            accrual = _funding_until(accrual, hit.ts, pos, funding_calc)
        else:
            fees = sum(abs(float(c)) for c in pos.commissions())
            realized = float(pos.realized_pnl)
            if mark is None:
                exit_ns = int(pos.ts_closed)
            else:
                exit_ns = mark.ts
                fees += mark.exit_fee
                realized += mark.unrealized - mark.exit_fee
            closed = mark is None
        out.total_fees += fees
        out.total_funding += accrual.total
        out.funding_fallbacks += accrual.fallback_settlements
        out.trade_pnls.append(realized - accrual.total)
        out.closed_net.append((exit_ns, realized - accrual.total))
        out.cash_events.append((exit_ns, realized))
        out.cash_events.extend((ts, -amount) for ts, amount in accrual.payments)
        if closed:
            out.n_closed += 1
            entry_val = abs(float(pos.peak_qty) * float(pos.avg_px_open))
            if entry_val > 0:
                out.returns.append(realized / entry_val)
    return out


def _funding_until(
    accrual: FundingAccrual, ts: int, pos: Any, funding_calc: FundingCalculator
) -> FundingAccrual:
    """The scan's funding of ``pos`` truncated at ``ts`` (the overlay's ``<= ts`` rule).

    The payments are the scan's own; only the fallback count is recounted, through the last
    kept settlement (exactly the kept ones: ``accrue`` counts settlements in (entry, exit]).
    """
    kept = [(t, amount) for t, amount in accrual.payments if t <= ts]
    if len(kept) == len(accrual.payments):
        return accrual
    fallbacks = (
        accrue_position_funding(funding_calc, pos, exit_ns=kept[-1][0]).fallback_settlements
        if kept
        else 0
    )
    return FundingAccrual(
        total=sum(amount for _, amount in kept), payments=kept, fallback_settlements=fallbacks
    )
