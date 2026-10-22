"""Per-trade accounting of a screening run after the liquidation overlay hit something.

The rebuild itself lives in :func:`vibe_quant.trade_ledger.screening_liquidation_trades`
(bead vibe-quant-rabmo.2 L6); this module keeps the resting-exit inputs and ``position_key``.
The rules below describe that rebuild.

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

from typing import TYPE_CHECKING, Any, Literal

from vibe_quant.liquidation.exit_timing import RestingExit
from vibe_quant.liquidation.overlay import default_symbol, enum_name

if TYPE_CHECKING:
    from collections.abc import Iterable


__all__ = [
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
