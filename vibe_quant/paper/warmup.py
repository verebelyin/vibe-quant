"""Paper warm-up: the REST seed, the refusal rule and the handover top-up (design.md R2.2-R2.3).

Before the NautilusTrader node exists, every strategy is seeded with the last
``SEED_BARS`` closed bars per bar type from Binance REST (``fetch_closed_klines``)
and replays them through ``strategy._warm_from``. A start whose indicators cannot
reach parity from that seed is REFUSED (``ConfigurationError``), never traded cold.

After the node runs, the first live bar may sit past the seed (a gap at handover):
the strategy asks the node for the missing bars through ``_warm_request_topup``;
``TopUpScheduler`` fetches them off the loop and feeds them back ON the loop. A
top-up that cannot be completed never stops the node (a stop flattens positions in
paper): ``on_failure`` keeps the session with entries off.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from nautilus_trader.core.datetime import millis_to_nanos
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.objects import Price, Quantity

from vibe_quant.data.downloader import KlineRaw, RestDownloadError, fetch_closed_klines
from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.warmup_class import WarmupUnsupported, rma
from vibe_quant.paper.config import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from nautilus_trader.model.data import BarType

    from vibe_quant.dsl.compiler import IndicatorInfo

logger = logging.getLogger(__name__)

#: Closed bars fetched per (symbol, bar type): user ruling Q2 (design.md).
SEED_BARS: Final = 8_000

#: A hole in a top-up is asked for at most this many times (a fetch that raises is retried
#: this many times too); then the session keeps running with entries off.
TOPUP_ATTEMPTS: Final = 3

#: Pause between two failed fetch attempts of one top-up (a test sets it to 0).
RETRY_SLEEP_S: Final = 1.0

_INTERVAL_SECONDS: Final[dict[str, int]] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3_600,
    "4h": 14_400,
    "1d": 86_400,
}

_AGGREGATION_UNIT: Final[dict[str, str]] = {"MINUTE": "m", "HOUR": "h", "DAY": "d"}


class WarmupRefused(ConfigurationError):
    """The seed cannot be built or cannot reach parity: the paper START is refused."""


def interval_of(bar_type: BarType) -> str:
    """Binance interval (``15m``) of a bar type (``...-15-MINUTE-LAST-EXTERNAL``)."""
    spec = bar_type.spec
    unit = _AGGREGATION_UNIT[BarAggregation(spec.aggregation).name]
    return f"{spec.step}{unit}"


def interval_ms(interval: str) -> int:
    """Bar length of a Binance interval, in milliseconds."""
    return _INTERVAL_SECONDS[interval] * 1_000


def last_closed_boundary_ms(now_ms: int, interval: str) -> int:
    """Open time of the bar still forming at ``now_ms``: every earlier bar is closed."""
    step = interval_ms(interval)
    return now_ms - now_ms % step


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1_000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------- classification


@dataclass(frozen=True)
class WarmupNeed:
    """What one indicator needs on one bar interval: ``bars``, or a ``refusal`` reason."""

    name: str
    type: str
    interval: str
    bars: int | None = None
    refusal: str | None = None


def indicator_needs(
    infos: list[IndicatorInfo], config: object, primary_interval: str
) -> list[WarmupNeed]:
    """Warm-up need of every indicator of one strategy (its effective params), sizing ATR last.

    ``config`` is the strategy's own config: a value the run overrode (``<name>_<param>``)
    wins over the spec default, exactly as the generated strategy reads it.
    """
    needs: list[WarmupNeed] = []
    for info in infos:
        interval = info.timeframe
        spec = info.spec
        if spec.paper_warmup is None:
            needs.append(
                WarmupNeed(
                    info.name, info.config.type, interval, refusal="no paper warm-up class"
                )
            )
            continue
        params = StrategyCompiler._merge_effective_params(info)
        for key in list(params):
            field = f"{info.name}_{key}"
            if hasattr(config, field):
                params[key] = getattr(config, field)
        try:
            bars = spec.paper_warmup(params, _INTERVAL_SECONDS[interval])
        except WarmupUnsupported as exc:
            needs.append(WarmupNeed(info.name, info.config.type, interval, refusal=str(exc)))
        else:
            needs.append(WarmupNeed(info.name, info.config.type, interval, bars=bars))
    if getattr(config, "sizing_mode", "risk") == "atr":
        period = int(getattr(config, "sizing_atr_period"))  # noqa: B009
        bars = rma()({"period": period}, _INTERVAL_SECONDS[primary_interval])
        needs.append(WarmupNeed("sizing_atr", "ATR", primary_interval, bars=bars))
    return needs


def required_intervals(infos: list[IndicatorInfo], primary_interval: str) -> list[str]:
    """Intervals that need a seed: the primary one first, then each indicator's, in order."""
    out = [primary_interval]
    for info in infos:
        if info.timeframe not in out:
            out.append(info.timeframe)
    return out


def _refused(symbol: str, interval: str, label: str, reason: str) -> WarmupRefused:
    return WarmupRefused(f"paper warm-up refused: {symbol} {interval} {label}: {reason}")


def check_warmup(
    symbol: str,
    needs: list[WarmupNeed],
    rows: Mapping[str, int] | None = None,
    caps: Mapping[str, int] | None = None,
) -> None:
    """Refuse a start whose indicators cannot reach parity from the seed.

    Without ``rows`` only the classification is judged (a refused or unclassified kind never
    converges, whatever the seed length), so nothing is fetched for a start that is refused
    anyway. With ``rows`` (bars the seed holds per interval) the bound is compared too: a need
    above the rows refuses, equality passes. ``caps`` are the strategy's ``_pta_buffer_cap``
    per interval; the buffer's steady-state size must fit in the seed as well.

    Raises:
        WarmupRefused: ``paper warm-up refused: <SYMBOL> <tf> <indicator> (<TYPE>): <reason>``.
    """
    for need in needs:
        if need.refusal is not None:
            raise _refused(symbol, need.interval, f"{need.name} ({need.type})", need.refusal)
    if rows is None:
        return
    for need in needs:
        held = rows.get(need.interval, 0)
        assert need.bars is not None
        if need.bars > held:
            raise _refused(
                symbol,
                need.interval,
                f"{need.name} ({need.type})",
                f"needs {need.bars} bars, seed holds {held}",
            )
    for interval, cap in (caps or {}).items():
        held = rows.get(interval, 0)
        if cap > held:
            raise _refused(
                symbol,
                interval,
                "compute_fn buffer",
                f"needs {cap} bars (buffer cap), seed holds {held}",
            )


# ----------------------------------------------------------------------------- the seed


class SeedError(Exception):
    """The seed of one bar type cannot be built; ``reason`` is the refusal text."""

    def __init__(self, reason: str, interval: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.interval = interval


def kline_to_bar(bar_type: BarType, row: KlineRaw) -> Bar:
    """A closed REST kline as an NT bar, stamped like the live adapter stamps it.

    NautilusTrader's Binance adapter sets ``ts_event`` to the kline's ``close_time``
    (its last millisecond) through ``millis_to_nanos`` and ``ts_init`` to the receive time;
    the seed uses the SAME conversion for both (it goes through a double, so it is not
    ``close_time * 1_000_000``: 64 ns off at today's epoch) so the strategy's watermark
    compares like with like.
    """
    ts = millis_to_nanos(row[6])
    return Bar(
        bar_type,
        Price.from_str(row[1]),
        Price.from_str(row[2]),
        Price.from_str(row[3]),
        Price.from_str(row[4]),
        Quantity.from_str(row[5]),
        ts,
        ts,
    )


def fetch_seed(symbol: str, bar_type: BarType, end_ms: int) -> list[Bar]:
    """The last ``SEED_BARS`` closed bars of ``bar_type`` before ``end_ms``, oldest first.

    A young listing returns fewer rows; ``check_warmup`` judges the count.

    Raises:
        SeedError: the fetch failed, no bar exists, or the rows are not contiguous.
    """
    interval = interval_of(bar_type)
    step = interval_ms(interval)
    try:
        rows = fetch_closed_klines(symbol, interval, end_ms - SEED_BARS * step, end_ms)
    except RestDownloadError as exc:
        raise SeedError(f"fetch failed: {type(exc).__name__}") from None
    if not rows:
        raise SeedError("fetch failed: no bars returned")
    for prev, row in zip(rows, rows[1:], strict=False):
        if row[6] - prev[6] != step:
            raise SeedError(f"seed gap at {_iso(prev[6] + 1)}")
    return [kline_to_bar(bar_type, row) for row in rows]


# ---------------------------------------------------------------------------- the top-up


def fetch_gap(symbol: str, bar_type: BarType, edge_ns: int, ts_ns: int) -> list[Bar]:
    """Closed bars strictly between two ``ts_event`` values (the watermark and the first queued bar)."""
    interval = interval_of(bar_type)
    rows = fetch_closed_klines(symbol, interval, edge_ns // 1_000_000 + 1, ts_ns // 1_000_000)
    bars = [kline_to_bar(bar_type, row) for row in rows]
    return [b for b in bars if edge_ns < b.ts_event < ts_ns]


class TopUpScheduler:
    """Runs the strategy's handover top-ups off the loop; the result is fed back ON the loop.

    ``request`` is the strategy seam ``_warm_request_topup(bar_type, edge, ts)`` (bound to
    its strategy and symbol by the node); it is called on the loop. A fetch that raises is
    attempted ``TOPUP_ATTEMPTS`` times. A response that leaves the hole open makes the strategy
    ask for it again: the SAME hole (same bar type and watermark) is fetched at most
    ``TOPUP_ATTEMPTS`` times, then ``on_failure(reason)`` runs and nothing is fetched or fed
    for it again. Failure never stops the node.
    """

    def __init__(
        self,
        on_failure: Callable[[str, str], None],
        *,
        fetch: Callable[[str, BarType, int, int], list[Bar]] | None = None,
    ) -> None:
        self._on_failure = on_failure
        self._fetch = fetch or fetch_gap
        self._asked: dict[tuple[int, BarType, int], int] = {}
        self._failed: set[int] = set()
        self.tasks: set[asyncio.Task[None]] = set()

    def request(
        self, strategy: Any, symbol: str, bar_type: BarType, edge_ns: int, ts_ns: int
    ) -> None:
        """Schedule the fetch of the bars missing in (edge, ts) for ``bar_type``."""
        key = id(strategy)
        if key in self._failed:
            return
        hole = (key, bar_type, edge_ns)
        self._asked[hole] = self._asked.get(hole, 0) + 1
        interval = interval_of(bar_type)
        if self._asked[hole] > TOPUP_ATTEMPTS:
            self._fail(key, symbol, interval)
            return
        task = asyncio.get_running_loop().create_task(
            self._run(strategy, symbol, bar_type, edge_ns, ts_ns)
        )
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def cancel(self) -> None:
        for task in list(self.tasks):
            task.cancel()

    def _fail(self, key: int, symbol: str, interval: str) -> None:
        self._failed.add(key)
        self._on_failure(symbol, interval)

    async def _run(
        self, strategy: Any, symbol: str, bar_type: BarType, edge_ns: int, ts_ns: int
    ) -> None:
        interval = interval_of(bar_type)
        key = id(strategy)
        bars = await asyncio.to_thread(self._fetch_with_retries, symbol, bar_type, edge_ns, ts_ns)
        if bars is None:
            self._fail(key, symbol, interval)
            return
        try:
            strategy._warm_feed_topup(bar_type, bars)
        except Exception as exc:
            logger.error("warm-up top-up feed failed (%s)", type(exc).__name__)
            self._fail(key, symbol, interval)

    def _fetch_with_retries(
        self, symbol: str, bar_type: BarType, edge_ns: int, ts_ns: int
    ) -> list[Bar] | None:
        """Off the loop: the bars, or None after ``TOPUP_ATTEMPTS`` raising fetches."""
        for attempt in range(TOPUP_ATTEMPTS):
            try:
                return self._fetch(symbol, bar_type, edge_ns, ts_ns)
            except Exception as exc:
                logger.warning(
                    "warm-up top-up fetch %s %s failed (%s), attempt %d/%d",
                    symbol,
                    interval_of(bar_type),
                    type(exc).__name__,
                    attempt + 1,
                    TOPUP_ATTEMPTS,
                )
                if attempt + 1 < TOPUP_ATTEMPTS:
                    time.sleep(RETRY_SLEEP_S)
        return None


def seed_all(
    symbol: str, bar_types: Mapping[str, BarType], now_ms: int
) -> dict[str, list[Bar]]:
    """Seed of every needed interval (blocking; run off the loop): ``{interval: bars}``.

    Raises:
        SeedError: carrying the interval the failure belongs to.
    """
    out: dict[str, list[Bar]] = {}
    for interval, bar_type in bar_types.items():
        try:
            out[interval] = fetch_seed(symbol, bar_type, last_closed_boundary_ms(now_ms, interval))
        except SeedError as exc:
            raise SeedError(exc.reason, interval) from None
    return out

