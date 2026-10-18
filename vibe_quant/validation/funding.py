"""Post-hoc funding accrual for backtest trades (screening AND validation).

NT's backtest engine applies no funding, so funding is reconstructed per
trade from the archived Binance funding-rate history:

    payment_i = funding_rate_i x entry_notional x direction_sign

where direction_sign is +1 for longs (pay when rate > 0) and -1 for shorts.
The entry notional is used as a constant approximation of position value at
each settlement (mark-price notional is not available post-hoc).

Coverage is checked PER SETTLEMENT on the symbol's OWN settlement interval
(bd vibe-quant-x0axf.13), not on the fixed 8h UTC grid and not just at the
archive's end points (bd vibe-quant-e70tl.20). Each archived row carries the
``funding_interval_hours`` Binance Vision recorded for it (a row without one
inherits the previous row's, 8 before any known), and the gap between two
archived rows is scanned for the epoch-aligned multiples of
``max(h_prev, h_next)`` hours. A settlement the interval expects but the archive
lacks within :data:`_SETTLEMENT_TOLERANCE_NS` -- before/after the archive or
inside an interior hole such as BTC 2026-02-23 -> 03-10 -- is charged the
documented fallback rate (Binance baseline 0.01% per 8h) SCALED by ``h / 8``
and counted in ``FundingAccrual.fallback_settlements`` so callers can record
that the number is partly estimated. With all-8h intervals this is exactly the
old fixed 8h UTC grid, boundary for boundary. Archived rates at non-8h times
(symbols on 4h/1h funding) are charged as archived.

Rates are cached per (archive path, symbol) at module level, i.e. once per
worker process: discovery/screening workers evaluate thousands of genomes
over the same symbol.
"""

from __future__ import annotations

import logging
import os
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

# Binance baseline funding rate: 0.01% per 8h settlement
DEFAULT_FUNDING_RATE_PER_PERIOD = 0.0001
FUNDING_PERIOD_HOURS = 8
_NS_PER_HOUR = 3_600 * 1_000_000_000
_FUNDING_PERIOD_NS = FUNDING_PERIOD_HOURS * _NS_PER_HOUR
# Archived funding_time carries a few ms of jitter after the boundary.
_SETTLEMENT_TOLERANCE_NS = 60 * 1_000_000_000

# Process-level rate cache: (archive path, symbol) -> (times_ns, rates)
_RATE_CACHE: dict[tuple[str, str], tuple[list[int], list[float]]] = {}
# Process-level interval cache: (archive path, symbol) -> (times_ns, hours), keyed
# on the IDENTITY of the cached times list so a hit costs no scan of the archive.
_INTERVAL_CACHE: dict[tuple[str, str], tuple[list[int], list[int]]] = {}
# (archive path, symbol, hole start ns) already warned about
_WARNED_HOLES: set[tuple[str, str, int]] = set()
# (archive path, symbol) already warned about rows without a Vision interval
_WARNED_INTERVAL_MISSING: set[tuple[str, str]] = set()


def clear_rate_cache() -> None:
    """Drop cached funding series (tests / after an archive update)."""
    _RATE_CACHE.clear()
    _INTERVAL_CACHE.clear()
    _WARNED_HOLES.clear()
    _WARNED_INTERVAL_MISSING.clear()


def _multiples(step_ns: int, lo_exclusive: int, hi_inclusive: int) -> list[int]:
    """Epoch-aligned multiples of ``step_ns`` inside ``(lo_exclusive, hi_inclusive]``."""
    if hi_inclusive <= lo_exclusive:
        return []
    out: list[int] = []
    k = lo_exclusive // step_ns + 1
    while k * step_ns <= hi_inclusive:
        out.append(k * step_ns)
        k += 1
    return out


def missing_settlements(
    times_ns: Sequence[int],
    hours: Sequence[int],
    lo_ns: int,
    hi_ns: int,
    tol_ns: int = _SETTLEMENT_TOLERANCE_NS,
) -> list[tuple[int, int]]:
    """Settlements the symbol's interval expects in ``(lo_ns, hi_ns]`` with no row.

    The expected grid is the epoch-aligned multiples of ``H`` hours, where ``H``
    is the interval in effect around each settlement: ``hours[0]`` before the
    first archived row, ``max(hours[i], hours[i + 1])`` between two consecutive
    rows, and ``hours[-1]`` after the last one. ``max`` -- never ``min`` -- is
    what makes an interval switch hole-free: the recorded value lags at a
    switch, so the gap between two rows is never longer than ``max``. For an
    all-8h series this reduces to the pre-T4 fixed 8h UTC grid, boundary for
    boundary; with no rows at all ``H = 8`` everywhere.

    Rows are scanned from one neighbour on each side of the window
    (``O(rows in window + 2)``), so a gap spanning the whole window is measured
    with its true neighbouring rows rather than with a padded time window.

    Args:
        times_ns: Archived settlement times (ns, ascending).
        hours: Per-row settlement interval in hours, aligned with ``times_ns``.
        lo_ns: Window start (exclusive).
        hi_ns: Window end (inclusive).
        tol_ns: A row within this distance of a settlement covers it.

    Returns:
        ``(settlement_ts_ns, interval_hours)`` per missing settlement, ascending.

    Raises:
        ValueError: ``hours`` and ``times_ns`` differ in length.

    Known limitation (documented, measured): a deleted row exactly at an
    interval switch is masked by the ``max`` rule -- 104 of the 105 single-row
    deletions of SOL 2022-11-09..19 are detected; the masked one is 11-09 20:00,
    an 8h row whose neighbours' 4h grid does not expect it.
    """
    times = list(times_ns)
    n = len(times)
    if len(hours) != n:
        raise ValueError(f"hours has {len(hours)} entries for {len(times)} settlements")
    if n == 0:
        return [
            (ts, FUNDING_PERIOD_HOURS)
            for ts in _multiples(FUNDING_PERIOD_HOURS * _NS_PER_HOUR, lo_ns, hi_ns)
        ]
    aligned = [int(h) for h in hours]
    i0 = max(0, bisect_left(times, lo_ns - tol_ns) - 1)
    i1 = min(n, bisect_right(times, hi_ns + tol_ns) + 1)
    found: list[tuple[int, int]] = []
    if i0 == 0:
        # Before the first row the interval is the first row's own.
        head = min(hi_ns, times[0] - tol_ns - 1)
        found.extend((ts, aligned[0]) for ts in _multiples(aligned[0] * _NS_PER_HOUR, lo_ns, head))
    for i in range(i0, i1 - 1):
        h = max(aligned[i], aligned[i + 1])
        low = max(lo_ns, times[i] + tol_ns)
        high = min(hi_ns, times[i + 1] - tol_ns - 1)
        found.extend((ts, h) for ts in _multiples(h * _NS_PER_HOUR, low, high))
    if i1 == n:
        # After the last row the interval is the last row's own.
        found.extend(
            (ts, aligned[-1])
            for ts in _multiples(
                aligned[-1] * _NS_PER_HOUR, max(lo_ns, times[-1] + tol_ns), hi_ns
            )
        )
    found.sort()
    return found


@dataclass
class FundingAccrual:
    """Funding charged to one trade.

    Attributes:
        total: Quote-currency cost; positive = paid, negative = received.
        payments: ``(settlement_ts_ns, amount)`` per settlement crossed.
        fallback_settlements: Settlements charged at the fallback rate
            because the archive has no rate for them.
        fallback_timestamps: Their settlement timestamps (provenance, so a truncated
            accrual can recount them).
    """

    total: float = 0.0
    payments: list[tuple[int, float]] = field(default_factory=list)
    fallback_settlements: int = 0
    fallback_timestamps: list[int] = field(default_factory=list)

    def truncated(self, cutoff_ns: int) -> FundingAccrual:
        """The accrual up to and including ``cutoff_ns``: same payments, never re-accrued."""
        payments = [p for p in self.payments if p[0] <= cutoff_ns]
        fallbacks = [ts for ts in self.fallback_timestamps if ts <= cutoff_ns]
        return FundingAccrual(
            total=sum(amount for _, amount in payments),
            payments=payments,
            fallback_settlements=len(fallbacks),
            fallback_timestamps=fallbacks,
        )


class FundingCalculator:
    """Computes per-trade funding cost from archived funding rates."""

    def __init__(
        self,
        archive_path: Path | str | None = None,
        default_rate: float = DEFAULT_FUNDING_RATE_PER_PERIOD,
    ) -> None:
        from vibe_quant.data.archive import DEFAULT_ARCHIVE_PATH

        # Absolute, computed once: the caches below are process-wide, so a relative key
        # ("data/archive/raw_data.db") would serve one root's rates to another cwd.
        self._archive_path = Path(os.path.abspath(archive_path or DEFAULT_ARCHIVE_PATH))  # noqa: PTH100
        self._default_rate = default_rate

    @property
    def default_rate(self) -> float:
        """Fallback rate charged per uncovered 8h settlement."""
        return self._default_rate

    @staticmethod
    def symbol_from_instrument_id(instrument_id: str) -> str:
        """"BTCUSDT-PERP.BINANCE" -> "BTCUSDT" (archive symbol key)."""
        return instrument_id.split(".")[0].removesuffix("-PERP")

    def _load_rates(self, symbol: str) -> tuple[list[int], list[float]]:
        key = (str(self._archive_path), symbol)
        cached = _RATE_CACHE.get(key)
        if cached is not None:
            return cached
        times: list[int] = []
        rates: list[float] = []
        if not self._archive_path.exists():
            # Never create an empty archive DB as a side effect of a lookup.
            logger.warning(
                "Funding archive %s not found — %s funding charged at the fallback rate",
                self._archive_path,
                symbol,
            )
        else:
            try:
                from vibe_quant.data.archive import RawDataArchive

                archive = RawDataArchive(self._archive_path)
                try:
                    for row in archive.get_funding_rates(symbol):
                        # funding_time is stored in ms; convert to ns
                        times.append(int(row["funding_time"]) * 1_000_000)
                        rates.append(float(row["funding_rate"]))
                finally:
                    archive.close()
            except Exception:
                logger.warning("Could not load funding rates for %s", symbol, exc_info=True)
                times, rates = [], []
        _RATE_CACHE[key] = (times, rates)
        return times, rates

    def _load_intervals(self, symbol: str) -> list[int]:
        """Per-row settlement intervals (hours) aligned with :meth:`_load_rates`.

        A row takes the exact-ms ``raw_funding_intervals`` entry for it; a row
        without one inherits the previous row's interval, and rows before any
        known interval are 8h. Cached per ``(path, symbol)`` on the IDENTITY of
        the cached times list, so a repeated call costs no scan and
        ``CappedFundingCalculator`` still hits the base calculator's cache.
        """
        times, _ = self._load_rates(symbol)
        key = (str(self._archive_path), symbol)
        cached = _INTERVAL_CACHE.get(key)
        if cached is not None and cached[0] is times:
            return cached[1]
        hours = self._resolve_intervals(symbol, times)
        _INTERVAL_CACHE[key] = (times, hours)
        return hours

    def _resolve_intervals(self, symbol: str, times: list[int]) -> list[int]:
        """Read ``symbol``'s Vision intervals and align them to ``times`` (ns)."""
        if not times:
            return []
        known: dict[int, int] = {}
        if self._archive_path.exists():
            try:
                from vibe_quant.data.archive import RawDataArchive

                archive = RawDataArchive(self._archive_path)
                try:
                    known = archive.get_funding_intervals(symbol)
                finally:
                    archive.close()
            except Exception:
                logger.warning("Could not load funding intervals for %s", symbol, exc_info=True)
                known = {}
        hours: list[int] = []
        previous = FUNDING_PERIOD_HOURS
        unknown = 0
        for ts in times:
            entry = known.get(ts // 1_000_000)
            if entry is None:
                unknown += 1
            else:
                previous = int(entry)
            hours.append(previous)
        if unknown:
            key = (str(self._archive_path), symbol)
            if key not in _WARNED_INTERVAL_MISSING:
                _WARNED_INTERVAL_MISSING.add(key)
                logger.warning(
                    "%s: %d of %d archived funding rows have no Vision interval "
                    "(run 'python -m vibe_quant.data funding-intervals'); "
                    "using the previous row's interval",
                    symbol,
                    unknown,
                    len(times),
                )
        return hours

    def settlements(self, symbol: str) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """Read-only copies of a symbol's archived settlement times (ns) and rates.

        Returns:
            ``(times_ns, rates)`` as tuples, so a caller can neither mutate the
            process-level rate cache nor observe a later load of the same symbol.
        """
        times, rates = self._load_rates(symbol)
        return tuple(times), tuple(rates)

    def settlement_hours(self, symbol: str) -> tuple[int, ...]:
        """Read-only copy of a symbol's per-row settlement intervals (hours)."""
        return tuple(self._load_intervals(symbol))

    def accrue(
        self,
        instrument_id: str,
        direction: str,
        entry_notional: float,
        entry_ns: int,
        exit_ns: int | None,
    ) -> FundingAccrual:
        """Funding for one trade, settlement by settlement.

        Settlements strictly after entry and up to and including exit are
        counted. Positive amounts are paid by the trader.
        """
        accrual = FundingAccrual()
        if exit_ns is None or exit_ns <= entry_ns or entry_notional <= 0:
            return accrual

        symbol = self.symbol_from_instrument_id(instrument_id)
        sign = 1.0 if direction.upper() == "LONG" else -1.0
        times, rates = self._load_rates(symbol)
        hours = self._load_intervals(symbol)

        # 1) Archived settlements inside (entry, exit]
        lo = bisect_right(times, entry_ns)
        hi = bisect_right(times, exit_ns)
        payments = [(times[i], sign * rates[i] * entry_notional) for i in range(lo, hi)]

        # 2) Every settlement the symbol's own interval expects must be covered by
        #    an archived rate; uncovered ones (archive edges or interior holes) use
        #    the fallback, scaled by the interval (a 4h hole pays 4/8 of it).
        missing = missing_settlements(times, hours, entry_ns, exit_ns)
        if missing:
            payments.extend(
                (ts, sign * self._default_rate * (h / FUNDING_PERIOD_HOURS) * entry_notional)
                for ts, h in missing
            )
            payments.sort(key=lambda p: p[0])
            accrual.fallback_settlements = len(missing)
            accrual.fallback_timestamps = [ts for ts, _ in missing]
            self._warn_missing(symbol, [ts for ts, _ in missing], has_archive=bool(times))

        accrual.payments = payments
        accrual.total = sum(amount for _, amount in payments)
        return accrual

    def accrue_book(
        self,
        instrument_id: str,
        fills: Sequence[tuple[int, float]],
        mark: Callable[[int], float],
        end_ns: int,
    ) -> FundingAccrual:
        """Funding for one leg of a resized (NETTING) book, settlement by settlement.

        Each settlement in ``(first fill, end_ns]`` is paid on the signed
        quantity held strictly before it (fills at or after the settlement time
        do not count) at ``mark(settlement_ns)``: ``qty x mark x rate``,
        positive = paid. Uncovered settlements of the leg's own interval while it
        is non-flat use the fallback rate (scaled by the interval) and are
        counted, as in :meth:`accrue`.

        Args:
            instrument_id: e.g. "BTCUSDT-PERP.BINANCE".
            fills: ``(ts_ns, signed quantity change)`` sorted by time.
            mark: Mark price at a settlement timestamp (ns).
            end_ns: Last instant funding is accrued for (inclusive).
        """
        accrual = FundingAccrual()
        if not fills or end_ns <= fills[0][0]:
            return accrual
        fill_ts = [ts for ts, _ in fills]
        held: list[float] = []
        running = 0.0
        for _, delta in fills:
            running += delta
            held.append(running)

        def qty_before(ts: int) -> float:
            i = bisect_left(fill_ts, ts)  # fills strictly before ts
            return held[i - 1] if i else 0.0

        symbol = self.symbol_from_instrument_id(instrument_id)
        times, rates = self._load_rates(symbol)
        hours = self._load_intervals(symbol)
        start = fill_ts[0]
        payments: list[tuple[int, float]] = []
        for i in range(bisect_right(times, start), bisect_right(times, end_ns)):
            qty = qty_before(times[i])
            if qty:
                payments.append((times[i], qty * mark(times[i]) * rates[i]))

        charged: list[int] = []
        for ts, h in missing_settlements(times, hours, start, end_ns):
            qty = qty_before(ts)
            if qty:
                charged.append(ts)
                payments.append(
                    (ts, qty * mark(ts) * self._default_rate * (h / FUNDING_PERIOD_HOURS))
                )
        if charged:
            payments.sort(key=lambda p: p[0])
            accrual.fallback_settlements = len(charged)
            accrual.fallback_timestamps = list(charged)
            self._warn_missing(symbol, charged, has_archive=bool(times))

        accrual.payments = payments
        accrual.total = sum(amount for _, amount in payments)
        return accrual

    def compute_funding(
        self,
        instrument_id: str,
        direction: str,
        entry_notional: float,
        entry_ns: int,
        exit_ns: int | None,
    ) -> float:
        """Total funding cost for one closed trade (see :meth:`accrue`).

        Args:
            instrument_id: e.g. "BTCUSDT-PERP.BINANCE".
            direction: "LONG" or "SHORT".
            entry_notional: entry_price x quantity (quote currency).
            entry_ns: Position open timestamp (ns).
            exit_ns: Position close timestamp (ns); None -> 0.0.

        Returns:
            Funding cost in quote currency. Positive = paid by the trader,
            negative = received.
        """
        return self.accrue(instrument_id, direction, entry_notional, entry_ns, exit_ns).total

    def _warn_missing(self, symbol: str, missing: list[int], *, has_archive: bool) -> None:
        """Warn once per contiguous run of missing settlements per process."""
        from datetime import UTC, datetime

        runs: list[tuple[int, int]] = []
        for ts in missing:
            if runs and ts - runs[-1][1] <= _FUNDING_PERIOD_NS:
                runs[-1] = (runs[-1][0], ts)
            else:
                runs.append((ts, ts))
        for start, end in runs:
            key = (str(self._archive_path), symbol, start)
            if key in _WARNED_HOLES:
                continue
            _WARNED_HOLES.add(key)
            logger.warning(
                "No archived funding for %s %s -> %s (%s) — charging fallback %.4f%% "
                "per 8h, scaled to the interval",
                symbol,
                datetime.fromtimestamp(start / 1e9, tz=UTC).isoformat(),
                datetime.fromtimestamp(end / 1e9, tz=UTC).isoformat(),
                "archive hole/edge" if has_archive else "symbol not in archive",
                self._default_rate * 100,
            )
