"""Post-hoc funding accrual for validation trades.

NT's Position API exposes no cumulative funding, so validation previously
reported total_funding=0 (a systematic bias for perps held across the 8h
settlements). This module reconstructs funding per closed trade from the
archived Binance funding-rate history:

    payment_i = funding_rate_i x entry_notional x direction_sign

where direction_sign is +1 for longs (pay when rate > 0) and -1 for shorts.
The entry notional is used as a constant approximation of position value at
each settlement (mark-price notional is not available post-hoc).

When the archive has no rates covering a trade's window, a flat default
rate (Binance baseline 0.01% per 8h period) is applied per settlement and
a warning is logged once per symbol.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from pathlib import Path

logger = logging.getLogger(__name__)

# Binance baseline funding rate: 0.01% per 8h settlement
DEFAULT_FUNDING_RATE_PER_PERIOD = 0.0001
FUNDING_PERIOD_HOURS = 8
_FUNDING_PERIOD_NS = FUNDING_PERIOD_HOURS * 3_600 * 1_000_000_000


class FundingCalculator:
    """Computes per-trade funding cost from archived funding rates.

    Rates are lazily loaded per symbol and cached as parallel sorted lists
    of (settlement time ns, rate) for bisect lookups.
    """

    def __init__(
        self,
        archive_path: Path | None = None,
        default_rate: float = DEFAULT_FUNDING_RATE_PER_PERIOD,
    ) -> None:
        self._archive_path = archive_path
        self._default_rate = default_rate
        self._cache: dict[str, tuple[list[int], list[float]]] = {}
        self._warned_symbols: set[str] = set()

    @staticmethod
    def symbol_from_instrument_id(instrument_id: str) -> str:
        """"BTCUSDT-PERP.BINANCE" -> "BTCUSDT" (archive symbol key)."""
        return instrument_id.split(".")[0].removesuffix("-PERP")

    def _load_rates(self, symbol: str) -> tuple[list[int], list[float]]:
        cached = self._cache.get(symbol)
        if cached is not None:
            return cached
        times: list[int] = []
        rates: list[float] = []
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
        self._cache[symbol] = (times, rates)
        return times, rates

    def compute_funding(
        self,
        instrument_id: str,
        direction: str,
        entry_notional: float,
        entry_ns: int,
        exit_ns: int | None,
    ) -> float:
        """Funding cost for one closed trade.

        Args:
            instrument_id: e.g. "BTCUSDT-PERP.BINANCE".
            direction: "LONG" or "SHORT".
            entry_notional: entry_price x quantity (quote currency).
            entry_ns: Position open timestamp (ns).
            exit_ns: Position close timestamp (ns); None -> 0.0.

        Returns:
            Funding cost in quote currency. Positive = paid by the trader,
            negative = received. Settlements strictly after entry and up to
            and including exit are counted.
        """
        if exit_ns is None or exit_ns <= entry_ns or entry_notional <= 0:
            return 0.0

        symbol = self.symbol_from_instrument_id(instrument_id)
        sign = 1.0 if direction.upper() == "LONG" else -1.0

        times, rates = self._load_rates(symbol)
        if times and times[0] <= entry_ns and exit_ns <= times[-1] + _FUNDING_PERIOD_NS:
            lo = bisect_right(times, entry_ns)
            hi = bisect_right(times, exit_ns)
            rate_sum = sum(rates[lo:hi])
            return sign * rate_sum * entry_notional

        # Fallback: flat default rate per 8h settlement boundary crossed
        if symbol not in self._warned_symbols:
            self._warned_symbols.add(symbol)
            logger.warning(
                "No archived funding rates covering trade window for %s — "
                "using flat default %.4f%% per %dh settlement",
                symbol,
                self._default_rate * 100,
                FUNDING_PERIOD_HOURS,
            )
        n_settlements = self._count_settlements(entry_ns, exit_ns)
        return sign * self._default_rate * n_settlements * entry_notional

    @staticmethod
    def _count_settlements(entry_ns: int, exit_ns: int) -> int:
        """Number of 8h UTC settlement boundaries in (entry, exit]."""
        first = (entry_ns // _FUNDING_PERIOD_NS + 1) * _FUNDING_PERIOD_NS
        if first > exit_ns:
            return 0
        return int((exit_ns - first) // _FUNDING_PERIOD_NS) + 1
