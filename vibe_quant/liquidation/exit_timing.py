"""1m refinement of resting exits whose NT fill landed on the execution-bar boundary.

vjno2: the screening liquidation scan reads a resting exit at the execution (strategy)
bar's ``ts_init``, so a STOP_MARKET / LIMIT that validation's 1m simulation filled
mid-bar closes the position a whole bar too late and the liquidation that should have
happened stays hidden. This module recomputes, per boundary fill, the 1m bar that
actually reached the order's level and returns the timestamp the scan must use instead.

Pure functions over data: no NT, no I/O. :func:`refine_resting_exits` maps the resting
fills plus a per-symbol 1m LAST series to an :class:`ExitTiming`. V2 builds the
:class:`RestingExit` inputs, loads the series (V1c) and hands ``ExitTiming.scan_ts`` to
the overlay (V1b).

Semantics (contract rows R20-R32, design r2). ``E = execution_bar_ns``; a boundary fill
is one with ``(ts_event + 1 ms) % E == 0`` (R18, filtered by the caller), so the strategy
bar is ``[bar_open, bar_open + E)`` with ``bar_open = ts + 1 ms - E`` (R21).

* ``falls`` is the direction the trigger watches: True -> the bar LOW (a SELL STOP_MARKET
  protecting a long, or a BUY LIMIT taking profit on a short), False -> the bar HIGH (a
  BUY STOP_MARKET, a SELL LIMIT). ``level`` is the order's ``trigger_price`` (stop) or
  ``price`` (limit), never the fill price (R19).
* Trigger (R20/R20b): a STOP refines on equality (``low <= level`` / ``high >= level``); a
  LIMIT only on STRICT penetration (``low < level`` / ``high > level``). A bar that opens
  already past the level is a candidate like any other (R32).
* Candidate (R21): 1m bars whose ``ts_init`` (bar close - 1 ms) lies in
  ``[bar_open + 59.999 s, ts + 60 s)`` AND is strictly greater than
  ``active_from + 59.999 s``. That is a bar OPENING in ``[bar_open, bar_open + E)`` and
  strictly after ``active_from`` (the last ``OrderAccepted``/``OrderUpdated`` ``ts_event``
  at or before the fill). The refined timestamp is the first candidate's ``ts_init``
  (R22); it is always ``<= ts``.
* No candidate: the fill keeps its NT timestamp and is counted in ``kept_bar_close``
  (R25). A symbol with a boundary fill but no 1m series raises ``MissingExitBars`` (R26);
  an existing series with an empty window is the R25 case.
* Per position (``key``): timestamps never go backwards. Fills of ONE order (``order_id``)
  at the same NT timestamp share one refined timestamp; otherwise a later ``fill_index``
  must refine strictly after the previous fill's scan timestamp (R28/R28b), and distinct
  orders at the same (``key``, ``ts``) refine independently (an order's own ``active_from``
  applies, a kept fill stays kept). Anything else raises ``ValueError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np

from vibe_quant.errors import DataUnavailableError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "BarLike",
    "ExitTiming",
    "MissingExitBars",
    "RestingExit",
    "refine_resting_exits",
]

_ONE_MS = 1_000_000
_MINUTE_NS = 60_000_000_000
_ACTIVE_EPS = _MINUTE_NS - _ONE_MS  # 59.999 s: the R21 "strictly after active_from" offset


class MissingExitBars(DataUnavailableError):
    """A symbol with a boundary resting fill has no 1m LAST series to refine it (R26).

    A :class:`~vibe_quant.errors.DataUnavailableError` (hence still a ``ValueError``), so a
    MODE-OFF screening run that hits it fails closed instead of scoring a silent ``-inf`` row
    (R18).
    """


class BarLike(Protocol):
    """A 1m OHLC series (``metrics.BarSeries``-like); ``ts`` is the bar ``ts_init`` (ns).

    ``ts`` increases; only ``low`` and ``high`` are read (the trigger's extremes).
    """

    @property
    def ts(self) -> np.ndarray: ...
    @property
    def low(self) -> np.ndarray: ...
    @property
    def high(self) -> np.ndarray: ...


@dataclass(frozen=True)
class RestingExit:
    """One resting fill that landed on the execution-bar boundary (R18).

    ``key`` identifies the position and ``fill_index`` its fill within that position.
    ``order_id`` is the client order id: only fills of the SAME order at the same NT
    timestamp share a refined timestamp (R28b). ``ts`` is the NT fill ``ts_event``
    (equal to the strategy bar's ``ts_init``). ``kind`` is the NT order-type family;
    ``falls`` the direction the trigger watches (module docstring). ``level`` is the
    order's ``trigger_price`` / ``price`` and ``active_from`` the last
    ``OrderAccepted``/``OrderUpdated`` ``ts_event`` at or before the fill.
    """

    key: str
    fill_index: int
    order_id: str
    symbol: str
    ts: int
    kind: Literal["stop", "limit"]
    falls: bool
    level: float
    active_from: int


@dataclass(frozen=True)
class ExitTiming:
    """Refined scan timestamps and counts for one run's boundary resting fills.

    ``scan_ts[(key, fill_index)]`` is the timestamp the scan must use for that fill (the
    refined ``r``, equal to the NT timestamp for a kept fill; R24b reads ``r`` and ``n``
    from here). ``refined`` / ``kept_bar_close`` are fill counts and ``by_symbol`` holds
    the same two counts per symbol.
    """

    scan_ts: dict[tuple[str, int], int]
    refined: int
    kept_bar_close: int
    by_symbol: dict[str, dict[str, int]]

    def to_notes(self) -> dict[str, Any]:
        """JSON-safe summary for ``last_liquidation_notes["exit_timing"]`` (R25b)."""
        return {
            "refined": self.refined,
            "kept_bar_close": self.kept_bar_close,
            "by_symbol": {symbol: dict(counts) for symbol, counts in self.by_symbol.items()},
        }


def refine_resting_exits(
    exits: Sequence[RestingExit],
    bars_1m: Mapping[str, BarLike],
    execution_bar_ns: int,
) -> ExitTiming:
    """Refine boundary resting fills to the 1m bar that reached their level (R20-R32).

    ``bars_1m`` maps a symbol to its 1m LAST series; a symbol with an exit but no entry
    raises ``MissingExitBars`` (R26). Raises ``ValueError`` when a position's refined
    timestamps would go backwards or fall after the NT fill (R28).
    """
    if execution_bar_ns <= 0:
        raise ValueError(f"execution_bar_ns must be > 0, got {execution_bar_ns!r}")
    scan_ts: dict[tuple[str, int], int] = {}
    by_symbol: dict[str, dict[str, int]] = {}
    shared: dict[tuple[str, str, int], tuple[int, bool]] = {}
    last: dict[str, tuple[str, int, int]] = {}  # key -> (order_id, ts, refined)
    refined = kept = 0
    for ex in sorted(exits, key=lambda e: (e.key, e.fill_index)):
        series = bars_1m.get(ex.symbol)
        if series is None:
            raise MissingExitBars(
                f"{ex.symbol}: no 1m LAST series to refine the boundary fill "
                f"{ex.key} fill {ex.fill_index}"
            )
        n = int(ex.ts)
        found = shared.get((ex.key, ex.order_id, n))
        if found is None:
            r = _first_candidate(ex, series, execution_bar_ns)
            # fills of ONE order at the same NT timestamp share one refined timestamp (R28b)
            found = (n if r is None else r, r is not None)
            shared[(ex.key, ex.order_id, n)] = found
        r, is_refined = found
        if r > n:
            raise ValueError(
                f"{ex.key} fill {ex.fill_index}: refined {r} is after the NT fill {n}"
            )
        prev = last.get(ex.key)
        # distinct orders refine independently and must advance; only one order's fills at
        # one NT timestamp may share a scan timestamp (R28/R28b)
        if prev is not None and not (prev[0] == ex.order_id and prev[1] == n) and r <= prev[2]:
            raise ValueError(
                f"{ex.key} fill {ex.fill_index}: refined {r} is not after the previous "
                f"fill's scan {prev[2]} (NT {n}, previous NT {prev[1]})"
            )
        last[ex.key] = (ex.order_id, n, r)
        scan_ts[(ex.key, ex.fill_index)] = r
        if is_refined:
            refined += 1
        else:
            kept += 1
        counts = by_symbol.setdefault(ex.symbol, {"refined": 0, "kept_bar_close": 0})
        counts["refined" if is_refined else "kept_bar_close"] += 1
    return ExitTiming(scan_ts, refined, kept, by_symbol)


def _first_candidate(ex: RestingExit, series: BarLike, execution_bar_ns: int) -> int | None:
    """``ts_init`` of the first candidate 1m bar that reaches ``ex.level``, else None."""
    ts = np.asarray(series.ts, dtype=np.int64)
    n = int(ex.ts)
    bar_open = n + _ONE_MS - execution_bar_ns
    lo = int(np.searchsorted(ts, bar_open + _ACTIVE_EPS, side="left"))
    hi = int(np.searchsorted(ts, n + _MINUTE_NS, side="left"))
    lo = max(lo, int(np.searchsorted(ts, ex.active_from + _ACTIVE_EPS, side="right")))
    if lo >= hi:
        return None
    low = np.asarray(series.low, dtype=np.float64)[lo:hi]
    high = np.asarray(series.high, dtype=np.float64)[lo:hi]
    if ex.kind == "stop":
        reached = low <= ex.level if ex.falls else high >= ex.level
    elif ex.kind == "limit":
        reached = low < ex.level if ex.falls else high > ex.level
    else:
        raise ValueError(f"{ex.key} fill {ex.fill_index}: unknown resting exit kind {ex.kind!r}")
    if not bool(reached.any()):
        return None
    return int(ts[lo + int(reached.argmax())])
