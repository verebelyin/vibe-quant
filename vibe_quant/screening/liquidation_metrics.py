"""Per-trade accounting of a screening run after the liquidation overlay hit something.

The rebuild itself lives in :func:`vibe_quant.trade_ledger.screening_liquidation_trades`
(bead vibe-quant-rabmo.2 L6); this module keeps the resting-exit inputs, the shared
boundary-exit refinement core and ``position_key``. The rules below describe that rebuild.

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

from vibe_quant.errors import DataUnavailableError
from vibe_quant.liquidation.exit_timing import ExitTiming, RestingExit, refine_resting_exits
from vibe_quant.liquidation.overlay import default_symbol, enum_name

if TYPE_CHECKING:
    from collections.abc import Iterable


__all__ = [
    "funding_exit_ts",
    "position_key",
    "refined_exit_ts",
    "resting_exit_inputs",
    "scan_fill_ts_map",
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
    cache: Any, positions: Iterable[Any], execution_bar_ns: int
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
    # execution bar (E > 60 s, an int) reaches this call (refined_exit_ts, R17).
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


def refined_exit_ts(
    cache: Any, positions: Iterable[Any], bar_ns: int, catalog_path: Any
) -> ExitTiming | None:
    """Refine the boundary resting exits of ``positions`` to their real 1m instant (R17-R30).

    The ONE refinement core: the liquidation scan and the funding-exit rule both read it
    (design-costs.md F1), and the modeled-cost ledger reuses it per closed position. It loads no
    other series and logs nothing -- warnings and the R25b notes stay with the caller.

    R17: only a coarser-than-1m execution with at least one boundary STOP_MARKET/LIMIT fill
    refines anything; anything else returns ``None`` (byte-identical scan). R17b: the 1m LAST
    series is loaded through :func:`~vibe_quant.validation.extraction.load_bar_windows` per
    SYMBOL, only for symbols that have such a fill -- never for a symbol without one, never from
    the engine cache and never the MARK series (R19). A symbol whose 1m LAST bars are absent
    leaves no series behind, so :func:`~vibe_quant.liquidation.exit_timing.refine_resting_exits`
    raises ``MissingExitBars`` (R26); the caller fails the run closed.
    """
    if bar_ns <= 60_000_000_000:  # 1m (or finer) execution: nothing to refine
        return None
    exits = resting_exit_inputs(cache, positions, bar_ns)
    if not exits:
        return None
    from vibe_quant.data.catalog import INTERVAL_TO_AGGREGATION
    from vibe_quant.validation.extraction import load_bar_windows

    step, agg = INTERVAL_TO_AGGREGATION["1m"]
    windows: dict[str, list[tuple[int, int]]] = {}
    for exit_ in exits:
        # R19: the exit's own bar window, [ts + 1 ms - E, ts], as an inclusive ns pair
        windows.setdefault(exit_.symbol, []).append((exit_.ts + _ONE_MS - bar_ns, exit_.ts))
    bars_1m: dict[str, Any] = {}
    if catalog_path is not None:  # R10c: no catalog -> no 1m LAST -> MissingExitBars
        for symbol, spans in windows.items():
            series = load_bar_windows(
                catalog_path,
                f"{symbol}-PERP.BINANCE-{step}-{agg.name}-LAST-EXTERNAL",
                spans,
            )
            if series is not None:
                bars_1m[symbol] = series
    return refine_resting_exits(exits, bars_1m, bar_ns)


def scan_fill_ts_map(timing: ExitTiming | None) -> dict[str, dict[int, int]] | None:
    """``ExitTiming.scan_ts`` as ``position key -> {fill index: scan ts}`` (``None`` for ``None``)."""
    if timing is None:
        return None
    scan_fill_ts: dict[str, dict[int, int]] = {}
    for (key, index), ts in timing.scan_ts.items():
        scan_fill_ts.setdefault(key, {})[index] = ts
    return scan_fill_ts


def funding_exit_ts(
    cache: Any,
    positions: Iterable[Any],
    execution_tf: str | None,
    catalog_path: Any,
    refine: frozenset[str] = frozenset(),
) -> ExitTiming | None:
    """The refined exits of the ``refine`` instruments' positions, for the funding exit rule.

    R9/R14: no refined instrument (an empty ``refine``) or an unsupported execution timeframe is
    excluded before any input is read, so no boundary scan runs and no ``vibe_quant.liquidation``
    module is imported (R9d); only refined instruments' positions are scanned, so another coin's
    boundary exit never needs 1m LAST. A ``MissingExitBars`` is a ``DataUnavailableError`` and
    re-raised unchanged (R10/R10c); any other failure becomes one naming the timeframe (R10b).
    """
    from vibe_quant.validation.extraction import _TIMEFRAME_SECONDS

    if execution_tf not in _TIMEFRAME_SECONDS:
        return None
    if not refine:
        return None
    bar_ns = _TIMEFRAME_SECONDS[execution_tf] * 1_000_000_000
    try:
        return refined_exit_ts(
            cache,
            [p for p in positions if str(p.instrument_id) in refine],
            bar_ns,
            catalog_path,
        )
    except DataUnavailableError:
        raise
    except Exception as exc:
        raise DataUnavailableError(
            f"funding exit refinement ({execution_tf}): {type(exc).__name__}: {exc}"
        ) from exc
