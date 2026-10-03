"""Post-hoc funding accrual for backtest trades (screening AND validation).

NT's backtest engine applies no funding, so funding is reconstructed per
trade from the archived Binance funding-rate history:

    payment_i = funding_rate_i x entry_notional x direction_sign

where direction_sign is +1 for longs (pay when rate > 0) and -1 for shorts.
The entry notional is used as a constant approximation of position value at
each settlement (mark-price notional is not available post-hoc).

Coverage is checked PER SETTLEMENT, not just at the archive's end points
(bd vibe-quant-e70tl.20): every 8h UTC boundary the trade crosses must have an
archived rate within :data:`_SETTLEMENT_TOLERANCE_NS`. Boundaries without one
-- before/after the archive or inside an interior hole such as BTC
2026-02-23 -> 03-10 -- are charged the documented fallback rate (Binance
baseline 0.01% per 8h) and counted in ``FundingAccrual.fallback_settlements``
so callers can record that the number is partly estimated. Archived rates at
non-8h times (symbols on 4h/1h funding) are charged as archived.

Rates are cached per (archive path, symbol) at module level, i.e. once per
worker process: discovery/screening workers evaluate thousands of genomes
over the same symbol.
"""

from __future__ import annotations

import logging
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Binance baseline funding rate: 0.01% per 8h settlement
DEFAULT_FUNDING_RATE_PER_PERIOD = 0.0001
FUNDING_PERIOD_HOURS = 8
_FUNDING_PERIOD_NS = FUNDING_PERIOD_HOURS * 3_600 * 1_000_000_000
# Archived funding_time carries a few ms of jitter after the boundary.
_SETTLEMENT_TOLERANCE_NS = 60 * 1_000_000_000

# Process-level rate cache: (archive path, symbol) -> (times_ns, rates)
_RATE_CACHE: dict[tuple[str, str], tuple[list[int], list[float]]] = {}
# (archive path, symbol, hole start ns) already warned about
_WARNED_HOLES: set[tuple[str, str, int]] = set()


def clear_rate_cache() -> None:
    """Drop cached funding series (tests / after an archive update)."""
    _RATE_CACHE.clear()
    _WARNED_HOLES.clear()


@dataclass
class FundingAccrual:
    """Funding charged to one trade.

    Attributes:
        total: Quote-currency cost; positive = paid, negative = received.
        payments: ``(settlement_ts_ns, amount)`` per settlement crossed.
        fallback_settlements: Settlements charged at the fallback rate
            because the archive has no rate for them.
    """

    total: float = 0.0
    payments: list[tuple[int, float]] = field(default_factory=list)
    fallback_settlements: int = 0


class FundingCalculator:
    """Computes per-trade funding cost from archived funding rates."""

    def __init__(
        self,
        archive_path: Path | str | None = None,
        default_rate: float = DEFAULT_FUNDING_RATE_PER_PERIOD,
    ) -> None:
        from vibe_quant.data.archive import DEFAULT_ARCHIVE_PATH

        self._archive_path = Path(archive_path) if archive_path else DEFAULT_ARCHIVE_PATH
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

        # 1) Archived settlements inside (entry, exit]
        lo = bisect_right(times, entry_ns)
        hi = bisect_right(times, exit_ns)
        payments = [(times[i], sign * rates[i] * entry_notional) for i in range(lo, hi)]

        # 2) Every 8h boundary crossed must be covered by an archived rate;
        #    uncovered ones (archive edges or interior holes) use the fallback.
        missing: list[int] = []
        boundary = (entry_ns // _FUNDING_PERIOD_NS + 1) * _FUNDING_PERIOD_NS
        while boundary <= exit_ns:
            j = bisect_left(times, boundary - _SETTLEMENT_TOLERANCE_NS)
            if j >= len(times) or times[j] > boundary + _SETTLEMENT_TOLERANCE_NS:
                missing.append(boundary)
            boundary += _FUNDING_PERIOD_NS
        if missing:
            fallback = sign * self._default_rate * entry_notional
            payments.extend((ts, fallback) for ts in missing)
            payments.sort(key=lambda p: p[0])
            accrual.fallback_settlements = len(missing)
            self._warn_missing(symbol, missing, has_archive=bool(times))

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
            if runs and ts - runs[-1][1] == _FUNDING_PERIOD_NS:
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
                "per %dh settlement",
                symbol,
                datetime.fromtimestamp(start / 1e9, tz=UTC).isoformat(),
                datetime.fromtimestamp(end / 1e9, tz=UTC).isoformat(),
                "archive hole/edge" if has_archive else "symbol not in archive",
                self._default_rate * 100,
                FUNDING_PERIOD_HOURS,
            )

    @staticmethod
    def _count_settlements(entry_ns: int, exit_ns: int) -> int:
        """Number of 8h UTC settlement boundaries in (entry, exit]."""
        first = (entry_ns // _FUNDING_PERIOD_NS + 1) * _FUNDING_PERIOD_NS
        if first > exit_ns:
            return 0
        return int((exit_ns - first) // _FUNDING_PERIOD_NS) + 1
