"""Result extraction helpers for validation runner.

Extracts metrics and trades from NautilusTrader backtest output
into vibe-quant's ValidationResult format. The cost/equity helpers in the
"Shared metric helpers" section are also used by the screening runner so
both tiers compute PF, funding and Sharpe the same way.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from vibe_quant.liquidation_settings import LIQUIDATION_MODES  # one definition, shared
from vibe_quant.metrics import (
    BarSeries,
    LedgerBook,  # average-cost replay (as liquidation.overlay)
    LedgerEvent,
    daily_balance_returns,
    gross_pnl,
    mark_to_market_max_drawdown,
    pnl_path,
    profit_factor,
    trade_win_rate,
)
from vibe_quant.trade_ledger import (
    apply_validation_moments,
    parse_nt_stats,
    split_positions,
    validation_engine_moments,
)
from vibe_quant.validation.fill_model import SlippageEstimator
from vibe_quant.validation.results import TradeRecord, ValidationResult

if TYPE_CHECKING:
    from collections.abc import Iterable
    from decimal import Decimal
    from pathlib import Path

    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.backtest.results import BacktestResult

    from vibe_quant.validation.funding import FundingAccrual, FundingCalculator
    from vibe_quant.validation.venue import VenueConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared metric helpers (screening + validation)
# ---------------------------------------------------------------------------


def all_positions(engine: Any) -> list[Any]:
    """Every position incarnation in the engine cache.

    NT netting mode reuses position IDs: a closed-then-reopened position is
    removed from the main index and kept as a snapshot, so positions() +
    position_snapshots() is needed to see all of them.
    """
    cache = engine.kernel.cache
    return list(cache.positions()) + list(cache.position_snapshots())


def position_direction(pos: Any) -> str:
    """'LONG' / 'SHORT' from a position's entry side."""
    return "LONG" if getattr(pos.entry, "name", str(pos.entry)).upper() == "BUY" else "SHORT"


def _leg_fees(pos: Any) -> tuple[float, float] | None:
    """Entry/exit fees (USDT) from each fill's own commission, maker entries only.

    ``None`` unless the position was opened by at least one MAKER fill (a
    post-only entry): every other trade, i.e. every market-mode entry, keeps
    the even split of the position total, byte-identical to before D3-6.
    Fills on the position's entry side are entry legs; the rest are exits.
    """
    fills = [
        e for e in (getattr(pos, "events", None) or []) if type(e).__name__ == "OrderFilled"
    ]
    entry_side = getattr(pos, "entry", None)
    entry = 0.0
    exit_fee = 0.0
    maker_entry = False
    for fill in fills:
        fee = abs(float(fill.commission))
        if fill.order_side == entry_side:
            entry += fee
            maker_entry = maker_entry or getattr(fill.liquidity_side, "name", "") == "MAKER"
        else:
            exit_fee += fee
    return (entry, exit_fee) if maker_entry else None


def _exact_split(total: float, entry: float) -> tuple[float, float]:
    """``(entry', total - entry')`` whose float sum is exactly ``total``.

    ``entry + (total - entry)`` can miss ``total`` by one ulp (~3% of random
    commission pairs, when ``entry`` sits on a finer binary grid than
    ``total``); the entry leg is then nudged by the fewest ulps (~1e-18 USDT)
    that restore the exact sum, so a trade's legs always add up to the
    position total that feeds every metric.
    """
    exit_fee = total - entry
    if entry + exit_fee == total:
        return entry, exit_fee
    for ulps in range(1, 8):
        for direction in (math.inf, -math.inf):
            nudged = entry
            for _ in range(ulps):
                nudged = math.nextafter(nudged, direction)
            exit_fee = total - nudged
            if nudged + exit_fee == total:
                return nudged, exit_fee
    return entry, total - entry  # unreachable in practice; keep the per-fill entry


def trade_fee_legs(pos: Any, pos_fees: float) -> tuple[float, float]:
    """(entry_fee, exit_fee) of one trade; the pair sums exactly to ``abs(pos_fees)``.

    Maker-entry trades take their entry leg from the fill commissions and the
    exit leg is the remainder (see :func:`_exact_split`). Everything else --
    market-mode trades, positions without fill events -- is split evenly like
    before, so existing per-trade rows do not move.
    """
    total = abs(pos_fees)
    split = _leg_fees(pos)
    if split is None or not 0.0 <= split[0] <= total:
        return total / 2.0, total / 2.0
    return _exact_split(total, split[0])


def maker_entry_stats(
    engine: Any,
    *,
    primary_bar_ns: int,
    maker_entry_timeout_bars: int,
) -> dict[str, object]:
    """Post-only entry statistics from the engine cache (vibe-quant-2gyk1.4 D3-6).

    Entry orders are the non-reduce-only post-only LIMIT orders the generated
    strategy placed. Terminal outcomes are kept apart: ``rejected_post_only``
    (the venue refused a would-be-taker), ``denied`` (the risk engine refused
    it) and ``expired`` (NT-expired, timed out, or still resting at the end).

    A cancelled, never-filled order is told apart by the number of PRIMARY BARS
    that closed between its placement and its cancel -- the strategy's own
    ``maker_entry_timeout_bars`` counter, not elapsed minutes (an order placed
    at a detail bar mid-way through a primary bar rests for fewer minutes than
    ``timeout x bar length``). The strategy evaluates its timeout before any
    exit/opposite-signal withdrawal on the same bar, so a cancel after
    ``maker_entry_timeout_bars`` closes is a timeout (``expired``) and an
    earlier one came from an exit/opposite signal (``cancelled_by_exit``).
    ``fill_rate`` is ``filled / submitted`` (``None`` when nothing was submitted).
    """
    from nautilus_trader.model.enums import OrderStatus, OrderType

    submitted = 0
    filled = 0
    partial = 0
    expired = 0
    rejected = 0
    denied = 0
    cancelled_by_exit = 0
    for order in engine.cache.orders():
        if order.is_reduce_only or not order.is_post_only or order.order_type != OrderType.LIMIT:
            continue
        submitted += 1
        status = order.status
        if status == OrderStatus.REJECTED:
            rejected += 1
            continue
        if status == OrderStatus.DENIED:
            denied += 1
            continue
        if status == OrderStatus.FILLED:
            filled += 1
            continue
        filled_qty = float(order.filled_qty)
        how_much = float(order.quantity)
        if 0.0 < filled_qty < how_much:
            partial += 1
            continue
        if filled_qty > 0.0:
            filled += 1  # fully filled but the status has not settled
            continue
        if status != OrderStatus.CANCELED:
            expired += 1  # NT-expired, or still resting when the run ended
            continue
        bars_closed = int(order.ts_last) // primary_bar_ns - int(order.ts_init) // primary_bar_ns
        if bars_closed >= maker_entry_timeout_bars:
            expired += 1
        else:
            cancelled_by_exit += 1
    return {
        "submitted": submitted,
        "filled": filled,
        "partial": partial,
        "expired": expired,
        "rejected_post_only": rejected,
        "denied": denied,
        "cancelled_by_exit": cancelled_by_exit,
        "fill_rate": (filled / submitted) if submitted else None,
    }


def accrue_position_funding(
    calculator: FundingCalculator,
    pos: Any,
    exit_ns: int | None = None,
) -> FundingAccrual:
    """Funding for one position (entry notional, held until close or ``exit_ns``)."""
    entry_price = float(pos.avg_px_open)
    quantity = float(pos.peak_qty)
    close_ns = exit_ns if exit_ns is not None else (int(pos.ts_closed) or None)
    return calculator.accrue(
        instrument_id=str(pos.instrument_id),
        direction=position_direction(pos),
        entry_notional=entry_price * quantity,
        entry_ns=int(pos.ts_opened),
        exit_ns=close_ns,
    )


def date_to_ns(value: str | None) -> int | None:
    """'YYYY-MM-DD' (or ISO datetime) as UTC nanoseconds; None if unparseable."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp()) * 1_000_000_000


def sharpe_sortino_from_returns(returns: dict[int, float]) -> tuple[float, float]:
    """NT's own SharpeRatio / SortinoRatio (252-day) on a daily returns dict.

    Using NT's pyo3 statistics keeps the math identical to what NT reported
    before (mean / std(ddof=1) * sqrt(252)); non-finite results (flat
    equity, fewer than 2 days) map to 0.0 instead of NaN.
    """
    from nautilus_trader.core.nautilus_pyo3 import SharpeRatio, SortinoRatio

    def _finite(value: object) -> float:
        try:
            f = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0
        return f if math.isfinite(f) else 0.0

    if len(returns) < 2:
        return 0.0, 0.0
    return (
        _finite(SharpeRatio().calculate_from_returns(returns)),
        _finite(SortinoRatio().calculate_from_returns(returns)),
    )


def daily_sharpe_sortino(
    starting_balance: float,
    cash_events: Iterable[tuple[int, float]],
    start_ns: int,
    end_ns: int,
) -> tuple[float, float]:
    """Sharpe/Sortino of the daily realized balance incl. modeled costs."""
    return sharpe_sortino_from_returns(
        daily_balance_returns(starting_balance, cash_events, start_ns, end_ns)
    )


# --- Mark-to-market drawdown (bd vibe-quant-e70tl.11) ----------------------

# Timeframe -> seconds, to pick the finest bar type the venue executed on
_TIMEFRAME_SECONDS: dict[str, int] = {
    "1s": 1,
    "5s": 5,
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3_600,
    "4h": 14_400,
    "1d": 86_400,
}

# Process-level cache of decoded catalog bars: bar dir -> (signature, series)
_BAR_CACHE: dict[str, tuple[tuple[tuple[str, int, int], ...], BarSeries]] = {}


def finest_timeframe(timeframes: Iterable[str]) -> str | None:
    """The finest known timeframe: NT's matching engine executes on it."""
    known = [tf for tf in timeframes if tf in _TIMEFRAME_SECONDS]
    return min(known, key=_TIMEFRAME_SECONDS.__getitem__) if known else None


def _decode_price_column(column: Any) -> np.ndarray:
    """NT fixed-point price column (fixed_size_binary 8/16) -> float64."""
    from nautilus_trader.model.objects import FIXED_PRECISION

    arr = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    width = arr.type.byte_width
    buf = arr.buffers()[1]
    raw = np.frombuffer(buf, dtype=np.uint8)[arr.offset * width : (arr.offset + len(arr)) * width]
    if width == 16:
        words = raw.view("<u8").reshape(-1, 2)
        value = words[:, 1].view("<i8").astype(np.float64) * 18446744073709551616.0 + words[
            :, 0
        ].astype(np.float64)
        return np.asarray(value / 10.0 ** int(FIXED_PRECISION), dtype=np.float64)
    if width == 8:
        return np.asarray(raw.view("<i8").astype(np.float64) / 1e9, dtype=np.float64)
    raise ValueError(f"Unsupported NT price encoding: {width}-byte fixed binary")


def load_catalog_bars(catalog_path: Path | str, bar_type: str) -> BarSeries | None:
    """Decode a catalog bar type into numpy arrays (cached per process).

    Reads the parquet files directly (no NT objects): discovery evaluates
    thousands of genomes per worker over the same bars, so this is decoded
    once per worker and sliced per evaluation.
    """
    from pathlib import Path as _Path

    import pyarrow.parquet as pq

    # Absolute: ``_BAR_CACHE`` is process-wide, so a relative catalog path
    # ("data/catalog") would serve one root's bars under another cwd's spelling
    # of the same key. abspath, not resolve(): a symlinked data/catalog must keep
    # working and keep its own key.
    bar_dir = _Path(os.path.abspath(catalog_path)) / "data" / "bar" / bar_type  # noqa: PTH100
    files = sorted(bar_dir.glob("*.parquet")) if bar_dir.is_dir() else []
    if not files:
        return None
    signature = tuple((f.name, f.stat().st_mtime_ns, f.stat().st_size) for f in files)
    cached = _BAR_CACHE.get(str(bar_dir))
    if cached is not None and cached[0] == signature:
        return cached[1]

    parts: list[tuple[np.ndarray, ...]] = []
    for f in files:
        table = pq.read_table(  # type: ignore[no-untyped-call]
            f, columns=["open", "high", "low", "close", "ts_init"]
        )
        # Not rounded to the instrument precision: raw / 10**FIXED_PRECISION
        # is bit-identical to NT's float(Price) (aggregated bars carry raw
        # values a hair off the tick grid, and fills use them as-is).
        cols = [
            _decode_price_column(table.column(name)) for name in ("open", "high", "low", "close")
        ]
        ts = table.column("ts_init").to_numpy().astype(np.int64)
        parts.append((ts, *cols))
    ts_all = np.concatenate([p[0] for p in parts])
    order = np.argsort(ts_all, kind="stable")
    ts_sorted = ts_all[order]
    keep = np.concatenate(([True], ts_sorted[1:] != ts_sorted[:-1]))
    sel = order[keep]
    series = BarSeries(
        ts=ts_all[sel],
        open=np.concatenate([p[1] for p in parts])[sel],
        high=np.concatenate([p[2] for p in parts])[sel],
        low=np.concatenate([p[3] for p in parts])[sel],
        close=np.concatenate([p[4] for p in parts])[sel],
    )
    _BAR_CACHE[str(bar_dir)] = (signature, series)
    return series


_INT64_MAX = 2**63 - 1


def _checked_window(index: int, window: tuple[int, int]) -> tuple[int, int]:
    """Validate one ``(start_ns, end_ns)`` window before any pruning or cast (R17g/R17h).

    Bounds are signed int64 ns, so they must be integers (not bool, not float)
    in ``[0, 2**63-1]`` with ``start <= end``. Nothing is clamped: a window
    outside the domain would otherwise silently match nothing or overflow the
    int64 comparison against the row-group statistics.
    """
    label = f"window #{index} {window!r}"
    start, end = window
    for bound in (start, end):
        if isinstance(bound, bool | np.bool_) or not isinstance(bound, int | np.integer):
            raise ValueError(f"{label}: bounds must be integer nanoseconds, got {bound!r}")
        if not 0 <= int(bound) <= _INT64_MAX:
            raise ValueError(f"{label}: bounds must be in [0, {_INT64_MAX}] (int64 ns)")
    if int(start) > int(end):
        raise ValueError(f"{label}: start > end")
    return int(start), int(end)


def _window_spans(windows: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Materialise and merge the inclusive ``(start_ns, end_ns)`` windows (R17d).

    Merging keeps the row mask and the row-group overlap test linear in the
    number of distinct ranges, however many windows the caller passes.
    """
    spans = sorted(_checked_window(i, window) for i, window in enumerate(windows))
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _in_windows(ts: np.ndarray, spans: list[tuple[int, int]]) -> np.ndarray:
    """Boolean mask of ``ts`` inside any merged inclusive span (R17f)."""
    if not spans:
        return np.zeros(len(ts), dtype=bool)
    starts = np.array([start for start, _ in spans], dtype=np.int64)
    ends = np.array([end for _, end in spans], dtype=np.int64)
    slot = np.searchsorted(starts, ts, side="right") - 1
    in_window = slot >= 0
    return np.asarray(in_window & (ts <= ends[np.where(in_window, slot, 0)]), dtype=bool)


def _overlapping_row_groups(
    meta: Any, ts_column: int, spans: list[tuple[int, int]]
) -> list[int]:
    """Row groups whose ``ts_init`` statistics overlap some window (R17c).

    A row group without usable statistics cannot be proven disjoint from the
    windows, so it is read anyway: the pruning must never drop a bar the full
    read would return.
    """
    if not spans:
        return []
    total = meta.num_row_groups
    if ts_column < 0 or ts_column >= meta.num_columns:
        return list(range(total))
    groups: list[int] = []
    for rg in range(total):
        stats = meta.row_group(rg).column(ts_column).statistics
        if stats is None or stats.min is None or stats.max is None:
            groups.append(rg)
            continue
        lo, hi = int(stats.min), int(stats.max)
        if any(lo <= end and hi >= start for start, end in spans):
            groups.append(rg)
    return groups


def load_bar_windows(
    catalog_path: Path | str,
    bar_type: str,
    windows: Iterable[tuple[int, int]],
) -> BarSeries | None:
    """Decode catalog bars for a set of ``ts_init`` windows (R17c/R17d/R17e).

    ``windows`` are inclusive ``(start_ns, end_ns)`` pairs (R17d); a bound that
    is not an integer in ``[0, 2**63-1]`` or ``start > end`` raises ``ValueError``
    naming the window, before any file is looked at (R17g/R17h). Only the
    parquet row groups whose ``ts_init`` min/max statistics overlap some window
    are read (R17c), rows outside the windows are masked out (R17f) and every
    ``BarSeries`` column is decoded, so the result is exactly the matching slice
    of a full :func:`load_catalog_bars` read, bit for bit. Segments are
    concatenated, sorted by ``ts_init`` and deduplicated the same way (stable
    sort, first occurrence wins), so overlapping windows are harmless.

    The read is deliberately independent of :func:`load_catalog_bars`:
    ``_BAR_CACHE`` is neither read nor written (no ``cache_bars`` fallback),
    because a windowed read must not evict or fill the full-read cache of the
    worker.

    Returns:
        ``None`` when the bar directory or the bar files are absent; an empty
        ``BarSeries`` when files exist but no row group overlaps (R17e).
    """
    from pathlib import Path as _Path

    import pyarrow.parquet as pq

    spans = _window_spans(windows)
    bar_dir = _Path(catalog_path) / "data" / "bar" / bar_type
    files = sorted(bar_dir.glob("*.parquet")) if bar_dir.is_dir() else []
    if not files:
        return None

    parts: list[tuple[np.ndarray, ...]] = []
    for f in files:
        parquet = pq.ParquetFile(f)  # type: ignore[no-untyped-call]
        groups = _overlapping_row_groups(
            parquet.metadata, parquet.schema_arrow.get_field_index("ts_init"), spans
        )
        if not groups:
            continue
        table = parquet.read_row_groups(  # type: ignore[no-untyped-call]
            groups, columns=["open", "high", "low", "close", "ts_init"]
        )
        ts = table.column("ts_init").to_numpy().astype(np.int64)
        cols = [
            _decode_price_column(table.column(name)) for name in ("open", "high", "low", "close")
        ]
        inside = _in_windows(ts, spans)
        if not inside.any():
            continue
        parts.append((ts[inside], *(col[inside] for col in cols)))

    if not parts:
        empty = np.array([], dtype=np.int64)
        return BarSeries(ts=empty, open=empty, high=empty, low=empty, close=empty)

    ts_all = np.concatenate([p[0] for p in parts])
    order = np.argsort(ts_all, kind="stable")
    ts_sorted = ts_all[order]
    keep = np.concatenate(([True], ts_sorted[1:] != ts_sorted[:-1]))
    sel = order[keep]
    return BarSeries(
        ts=ts_all[sel],
        open=np.concatenate([p[1] for p in parts])[sel],
        high=np.concatenate([p[2] for p in parts])[sel],
        low=np.concatenate([p[3] for p in parts])[sel],
        close=np.concatenate([p[4] for p in parts])[sel],
    )


def cache_bars(engine: Any, bar_type: str) -> BarSeries | None:
    """Execution bars from the engine cache (tests / no catalog).

    The NT cache keeps only the most recent ``bar_capacity`` bars per type,
    so this is only complete for short runs.
    """
    from nautilus_trader.model.data import BarType

    try:
        bars = list(engine.kernel.cache.bars(BarType.from_str(bar_type)))
    except Exception:
        return None
    if not bars:
        return None
    bars.sort(key=lambda b: int(b.ts_init))
    return BarSeries(
        ts=np.array([int(b.ts_init) for b in bars], dtype=np.int64),
        open=np.array([float(b.open) for b in bars]),
        high=np.array([float(b.high) for b in bars]),
        low=np.array([float(b.low) for b in bars]),
        close=np.array([float(b.close) for b in bars]),
    )


def position_ledger_events(
    positions: Iterable[Any],
    extra_cash: dict[str, list[tuple[int, float]]] | None = None,
) -> dict[str, list[LedgerEvent]]:
    """Per-instrument account events from positions' fill events.

    Positions are replayed in open order (DSL strategies hold one position
    per instrument at a time); within a position NT keeps fills in order.
    ``extra_cash`` adds modeled cash flows (funding, slippage) per
    instrument; they are merged in by timestamp.

    Raises:
        AttributeError: if a position exposes no fill events (stub engines).
    """
    from nautilus_trader.model.enums import OrderSide, OrderType

    by_instrument: dict[str, list[tuple[int, int, int, LedgerEvent]]] = {}
    ordered = sorted(
        positions,
        key=lambda p: (int(p.ts_opened), int(p.ts_closed) or 2**63 - 1),
    )
    seq = 0
    for pos in ordered:
        fills = pos.events
        if not fills:
            raise AttributeError("position has no fill events")
        bucket = by_instrument.setdefault(str(pos.instrument_id), [])
        for fill in fills:
            sign = 1.0 if fill.order_side == OrderSide.BUY else -1.0
            bucket.append(
                (
                    int(fill.ts_event),
                    0,
                    seq,
                    LedgerEvent(
                        ts=int(fill.ts_event),
                        qty=sign * float(fill.last_qty),
                        price=float(fill.last_px),
                        cash=-float(fill.commission),
                        resting=fill.order_type != OrderType.MARKET,
                    ),
                )
            )
            seq += 1
    for instrument_id, flows in (extra_cash or {}).items():
        bucket = by_instrument.setdefault(instrument_id, [])
        for ts, amount in flows:
            # cash flows sort after fills at the same timestamp
            bucket.append((int(ts), 1, seq, LedgerEvent(ts=int(ts), cash=amount)))
            seq += 1
    return {
        iid: [item[3] for item in sorted(items, key=lambda x: (x[0], x[1], x[2]))]
        for iid, items in by_instrument.items()
    }


@dataclass(frozen=True)
class OpenPositionMark:
    """A position still open at the end of the run, marked to market."""

    position: Any
    ts: int
    price: float
    unrealized: float
    exit_fee: float


def last_execution_bar(
    engine: Any,
    pos: Any,
    *,
    execution_timeframe: str | None,
    catalog_path: Path | str | None = None,
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> tuple[int, float] | None:
    """``(ts, close)`` of the last execution bar of ``pos``'s instrument inside the window.

    The bar a still-open position is valued at (:func:`mark_open_positions`); ``None`` when
    the instrument has no bars to value it. The liquidation path asks for it BEFORE the scan
    to know each open position's valuation horizon.
    """
    from vibe_quant.data.catalog import INTERVAL_TO_AGGREGATION

    instrument_id = str(pos.instrument_id)
    series = None
    if execution_timeframe in INTERVAL_TO_AGGREGATION:
        step, agg = INTERVAL_TO_AGGREGATION[execution_timeframe]
        bar_type = f"{instrument_id}-{step}-{agg.name}-LAST-EXTERNAL"
        series = load_catalog_bars(catalog_path, bar_type) if catalog_path else None
        if series is None:
            series = cache_bars(engine, bar_type)
    if series is not None:
        series = series.window(start_ns, end_ns)
    if series is None or len(series) == 0:
        return None
    return int(series.ts[-1]), float(series.close[-1])


def mark_open_positions(
    engine: Any,
    positions: Iterable[Any],
    *,
    execution_timeframe: str | None,
    catalog_path: Path | str | None = None,
    start_ns: int | None = None,
    end_ns: int | None = None,
    taker_fee: Decimal | float | None = None,
    log: bool = True,
) -> list[OpenPositionMark]:
    """Mark open positions at the last execution-bar close in the window.

    The exit fee is estimated at ``taker_fee`` when given (the run's fee
    schedule), else at the instrument's taker rate -- either way the flatten
    would be a market order. Positions that cannot be priced are skipped with
    a WARNING. ``log=False`` suppresses the per-mark WARNING (the liquidation path
    marks first, then logs only the survivors via :func:`log_open_mark`).
    """
    marks: list[OpenPositionMark] = []
    for pos in positions:
        last = last_execution_bar(
            engine,
            pos,
            execution_timeframe=execution_timeframe,
            catalog_path=catalog_path,
            start_ns=start_ns,
            end_ns=end_ns,
        )
        if last is None:
            logger.warning(
                "Position %s still open at end of run but no %s bars to mark it -- "
                "excluded from trades and return",
                str(pos.instrument_id),
                execution_timeframe,
            )
            continue
        last_ts, price = last
        qty = float(pos.signed_qty)
        if taker_fee is not None:
            taker = float(taker_fee)
        else:
            try:
                taker = float(engine.kernel.cache.instrument(pos.instrument_id).taker_fee)
            except Exception as exc:
                logger.warning(
                    "open-position exit fee: no taker rate for %s (%s); using 0.0005",
                    pos.instrument_id,
                    type(exc).__name__,
                )
                taker = 0.0005
        mark = OpenPositionMark(
            position=pos,
            ts=last_ts,
            price=price,
            unrealized=qty * (price - float(pos.avg_px_open)),
            exit_fee=abs(qty) * price * taker,
        )
        if log:
            log_open_mark(mark)
        marks.append(mark)
    return marks


def log_open_mark(mark: OpenPositionMark) -> None:
    """WARNING for a position still open at the end of the run and marked to market."""
    logger.warning(
        "Position %s (qty %s @ %s) still open at end of run -- marked at %s "
        "(unrealized %.2f, est. exit fee %.4f)",
        mark.position.instrument_id,
        float(mark.position.signed_qty),
        float(mark.position.avg_px_open),
        mark.price,
        mark.unrealized,
        mark.exit_fee,
    )


def mark_to_market_paths(
    engine: Any,
    *,
    execution_timeframe: str | None,
    catalog_path: Path | str | None = None,
    start_ns: int | None = None,
    end_ns: int | None = None,
    extra_cash: dict[str, list[tuple[int, float]]] | None = None,
    adaptive: bool = True,
    ledgers: dict[str, list[LedgerEvent]] | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None:
    """Per-instrument MTM PnL paths ``(ts, pnl_close, pnl_min)`` for the run.

    ``[]`` when the run has no positions; ``None`` when a path cannot be built
    (no fill events / no bars for a traded instrument). Shared by the
    mark-to-market drawdown and the daily MTM return series so the execution
    bars are loaded once (bd vibe-quant-wcn55).

    ``ledgers`` (the liquidation overlay's truncated per-instrument events,
    ``liquidation.overlay.OverlayResult.ledgers``) replaces the ledger rebuilt
    from the engine's positions. It already carries every per-position cash
    flow (funding, slippage), so combining it with ``extra_cash`` raises
    ``ValueError`` (cash would be booked twice or after a bust). ``None``
    (default) rebuilds from the engine.
    """
    from vibe_quant.data.catalog import INTERVAL_TO_AGGREGATION

    if ledgers is None:
        positions = all_positions(engine)
        if not positions:
            return []
        try:
            ledgers = position_ledger_events(positions, extra_cash)
        except AttributeError:
            return None
    else:
        if extra_cash:
            raise ValueError(
                "extra_cash cannot be combined with ledgers=: pass per-position cash "
                "to liquidation_overlay (cash_by_position / funding_by_position)"
            )
        if not ledgers:
            return []
    if execution_timeframe not in INTERVAL_TO_AGGREGATION:
        return None
    step, agg = INTERVAL_TO_AGGREGATION[execution_timeframe]

    paths: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for instrument_id, events in ledgers.items():
        bar_type = f"{instrument_id}-{step}-{agg.name}-LAST-EXTERNAL"
        series = load_catalog_bars(catalog_path, bar_type) if catalog_path else None
        if series is None:
            series = cache_bars(engine, bar_type)
        if series is None:
            logger.warning("No %s bars for mark-to-market drawdown", bar_type)
            return None
        series = series.window(start_ns, end_ns)
        if len(series) == 0 or (events and events[0].ts < int(series.ts[0]) - 1):
            logger.warning(
                "%s bars do not cover %s's first fill — cannot mark to market",
                bar_type,
                instrument_id,
            )
            return None
        pnl_close, pnl_min = pnl_path(series, events, adaptive=adaptive)
        paths.append((series.ts, pnl_close, pnl_min))
    return paths


def mark_to_market_drawdown(
    engine: Any,
    starting_balance: float,
    *,
    execution_timeframe: str | None,
    catalog_path: Path | str | None = None,
    start_ns: int | None = None,
    end_ns: int | None = None,
    extra_cash: dict[str, list[tuple[int, float]]] | None = None,
    adaptive: bool = True,
    ledgers: dict[str, list[LedgerEvent]] | None = None,
    floor_from_ns: int | None = None,
) -> float | None:
    """Mark-to-market max drawdown of the run (intrabar adverse extremes).

    Thin wrapper over :func:`mark_to_market_paths`: ``None`` when the paths
    cannot be computed (no fill events / no bars for a traded instrument),
    ``0.0`` for a run with no positions. Callers fall back to the closed-trade
    curve on ``None`` and MUST say so in logs. ``floor_from_ns`` (a cross
    liquidation's ``bust_ts``) see :func:`vibe_quant.metrics.mark_to_market_equity`.
    """
    paths = mark_to_market_paths(
        engine,
        execution_timeframe=execution_timeframe,
        catalog_path=catalog_path,
        start_ns=start_ns,
        end_ns=end_ns,
        extra_cash=extra_cash,
        adaptive=adaptive,
        ledgers=ledgers,
    )
    if paths is None:
        return None
    if not paths:
        return 0.0
    return mark_to_market_max_drawdown(starting_balance, paths, floor_from_ns=floor_from_ns)


def _ns_to_isoformat(ns_timestamp: int | float | str) -> str:
    """Convert a nanosecond Unix timestamp to ISO 8601 string.

    NautilusTrader Position.ts_opened / ts_closed are uint64
    nanosecond timestamps.  ``str()`` gives a bare integer string
    which breaks ``datetime.fromisoformat()``.
    """
    ns = int(ns_timestamp)
    return datetime.fromtimestamp(ns / 1e9, tz=UTC).isoformat()


def extract_results(
    run_id: int,
    strategy_name: str,
    bt_result: BacktestResult,
    engine: BacktestEngine,
    venue_config: VenueConfig,
    primary_timeframe: str | None = None,
    funding_calculator: FundingCalculator | None = None,
    run_start_date: str | None = None,
    run_end_date: str | None = None,
    execution_timeframe: str | None = None,
    catalog_path: Path | str | None = None,
    liquidation_mode: str = "off",
) -> ValidationResult:
    """Extract ValidationResult from NautilusTrader backtest output.

    Args:
        funding_calculator: When provided, per-trade funding is accrued
            from archived rates and charged into net PnL / headline return.
        run_start_date / run_end_date: Backtest window (YYYY-MM-DD) used
            for CAGR — measuring over first-trade..last-trade instead
            inflates CAGR for sparse traders.
        execution_timeframe: Finest bar timeframe the venue executed on
            (detail TF when loaded); drives the mark-to-market drawdown.
        catalog_path: Catalog holding those bars (engine cache if None).
        liquidation_mode: "off" (default), "isolated" or "cross": see :func:`extract_trades`.
    """
    result = ValidationResult(
        run_id=run_id,
        strategy_name=strategy_name,
        starting_balance=venue_config.starting_balance_usdt,
    )

    if bt_result is None:
        return result

    result.execution_time_seconds = bt_result.elapsed_time
    result.total_trades = bt_result.total_positions

    extract_stats(result, bt_result)
    moments_done = extract_trades(
        result,
        engine,
        venue_config,
        primary_timeframe=primary_timeframe,
        funding_calculator=funding_calculator,
        run_start_date=run_start_date,
        run_end_date=run_end_date,
        execution_timeframe=execution_timeframe,
        catalog_path=catalog_path,
        liquidation_mode=liquidation_mode,
    )
    if not moments_done:
        validation_engine_moments(result, engine)

    return result


def extract_stats(
    result: ValidationResult,
    bt_result: BacktestResult,
) -> None:
    """Extract aggregate statistics from BacktestResult into ValidationResult.

    NT's PortfolioAnalyzer populates stats_pnls with PnL and any
    registered statistics keyed by their ``name`` attribute, and
    stats_returns with the same registered statistics.

    Known key names from NT 1.222 (Rust statistics):
        stats_pnls:  "PnL (total)", "PnL% (total)", "Sharpe Ratio (252 days)",
                     "Sortino Ratio (252 days)", "Max Drawdown", "Win Rate",
                     "Profit Factor", "Expectancy", "Avg Winner", "Avg Loser"
        stats_returns: same statistic names

    Profit factor is deliberately NOT taken from NT: its realized-PnL PF is
    unimplemented and its returns PF is a daily-return statistic. The trade
    PF is computed from net trade PnLs in :func:`extract_trades`.

    Args:
        result: ValidationResult to populate (mutated in place).
        bt_result: NautilusTrader BacktestResult.
    """
    stats = parse_nt_stats(bt_result, tier="validation")
    if stats.total_return is not None:
        result.total_return = stats.total_return
    if stats.sharpe is not None:
        result.sharpe_ratio = stats.sharpe
    if stats.sortino is not None:
        result.sortino_ratio = stats.sortino
    if stats.max_drawdown is not None:
        result.max_drawdown = stats.max_drawdown
    if stats.win_rate is not None:
        result.win_rate = stats.win_rate
    if stats.avg_win is not None:
        result.avg_win = stats.avg_win
    if stats.avg_loss is not None:
        result.avg_loss = stats.avg_loss


def liquidation_execution_bar_ns(execution_timeframe: str | None) -> int:
    """The overlay's execution grid: the finest timeframe the run executed on, in ns.

    Validation executes on its detail timeframe (1m by default, an explicit 5s / 4h / ...
    override otherwise): the overlay values a liquidation's execution bar on that grid, so a
    wrong grid would accept ledgers it cannot replay. An unknown timeframe fails closed.
    """
    seconds = _TIMEFRAME_SECONDS.get(execution_timeframe) if execution_timeframe else None
    if seconds is None:
        raise ValueError(
            f"liquidation needs a known execution timeframe, got {execution_timeframe!r} "
            f"(one of {sorted(_TIMEFRAME_SECONDS)})"
        )
    return seconds * 1_000_000_000


def position_key(pos: Any) -> str:
    """Overlay key of an NT position: ``<id>@<ts_opened>`` (netting reuses position ids)."""
    return f"{pos.id}@{int(pos.ts_opened)}"


@dataclass
class _PositionCosts:
    """One trade's realized numbers before they become a :class:`TradeRecord`.

    ``realized_pnl`` is NT's (commissions included, modeled slippage/funding not); a
    liquidated position carries the overlay's account loss in it instead and sets the
    ``fee_legs`` / ``gross_pnl`` overrides. ``slippage_cost`` = ``entry_slippage`` (booked
    at the entry) + ``exit_slippage`` (booked at the exit).
    """

    pos: Any
    mark: OpenPositionMark | None
    realized_pnl: float
    exit_price: float
    exit_ns: int
    exit_reason: str
    pos_fees: float
    slippage_cost: float
    accrual: FundingAccrual | None
    fee_legs: tuple[float, float] | None = None
    gross_pnl: float | None = None
    entry_slippage: float = 0.0
    exit_slippage: float = 0.0


def _liquidated_costs(
    costs: _PositionCosts,
    hit: Any,
    exit_override: tuple[int, float],
) -> _PositionCosts:
    """Trade numbers of a liquidated position, from the venue fill and the account loss.

    The account keeps the fills up to the liquidation (their commissions stay booked and
    partial closes before it keep their PnL), then the liquidating close: price PnL
    ``hit.realized_pnl`` and the clearance fee. Funding is the scan's, cut at the liquidation;
    the strategy's own later exit never happened, so its modeled slippage leg is gone while
    the entry leg's stays.
    """
    ts, price = exit_override
    realized = LedgerBook()
    kept_fees = 0.0
    for fill in costs.pos.events:
        if int(fill.ts_event) > hit.ts:
            continue
        sign = 1.0 if getattr(fill.order_side, "name", str(fill.order_side)).upper() == "BUY" else -1.0
        commission = float(fill.commission)
        kept_fees += commission
        realized.apply(
            LedgerEvent(
                ts=int(fill.ts_event),
                qty=sign * float(fill.last_qty),
                price=float(fill.last_px),
                cash=-commission,
            )
        )
    realized_pnl = realized.realized + hit.realized_pnl - hit.clearance_fee
    pos_fees = kept_fees + hit.clearance_fee
    # the scan's funding input cut at the hit (open positions were accrued to their valuation
    # horizon, so this is min(hit, horizon)); never re-accrued: the ledger carries exactly this
    accrual = costs.accrual.truncated(ts) if costs.accrual is not None else None
    return _PositionCosts(
        pos=costs.pos,
        mark=None,  # a liquidated position is never marked at the end of the run
        realized_pnl=realized_pnl,
        exit_price=price,
        exit_ns=ts,
        exit_reason="liquidation",
        pos_fees=pos_fees,
        slippage_cost=costs.entry_slippage,
        accrual=accrual,
        fee_legs=(kept_fees, hit.clearance_fee),
        gross_pnl=realized_pnl + abs(pos_fees),
        entry_slippage=costs.entry_slippage,
    )


def _with_valuation_cash(
    ledgers: dict[str, list[LedgerEvent]], survivors: Iterable[_PositionCosts]
) -> dict[str, list[LedgerEvent]]:
    """Overlay ledgers plus the survivors' end-of-run valuation costs.

    The exit fee (and the exit leg of the modeled slippage) of a position still open at the
    run end is booked AFTER the scan, at its mark instant: it is a cost of valuing the
    position, never a liquidation trigger, so the scan that decided the hits did not see it.
    It still counts in the equity the ledger replays (no floor, no extra liquidation).
    """
    out = {iid: list(events) for iid, events in ledgers.items()}
    for item in survivors:
        mark = item.mark
        if mark is None:
            continue
        events = out.setdefault(str(item.pos.instrument_id), [])
        events.append(LedgerEvent(ts=mark.ts, cash=-mark.exit_fee))
        if item.exit_slippage:
            events.append(LedgerEvent(ts=mark.ts, cash=-item.exit_slippage))
    # stable: the new cash sorts behind the events already at its instant
    return {iid: sorted(events, key=lambda e: e.ts) for iid, events in out.items()}


class _SlippageModel:
    """Post-fill SPEC slippage of one leg (entry or exit fill), as the extraction charges it.

    Off when the engine already models slippage (``prob_slippage`` > 0). Market stats are
    estimated lazily, once, on the first charged leg.
    """

    def __init__(
        self, engine: BacktestEngine, venue_config: VenueConfig, primary_timeframe: str | None
    ) -> None:
        fill_cfg = venue_config.fill_config
        impact_k = getattr(fill_cfg, "impact_coefficient", 0.1) if fill_cfg else 0.1
        engine_prob_slippage = (
            float(getattr(fill_cfg, "prob_slippage", 0.0)) if fill_cfg is not None else 0.0
        )
        self._engine = engine
        self._primary_timeframe = primary_timeframe
        self._estimator = (
            SlippageEstimator(impact_coefficient=impact_k) if engine_prob_slippage <= 0.0 else None
        )
        self._stats: dict[str, tuple[float, float]] | None = None
        if self._estimator is None:
            logger.info(
                "Skipping post-fill SPEC slippage estimation because engine "
                "prob_slippage=%s is enabled",
                engine_prob_slippage,
            )

    def leg(self, pos: Any, price: float) -> float:
        """Slippage cost of one fill of ``pos`` at ``price`` (``peak_qty`` units)."""
        if self._estimator is None:
            return 0.0
        if self._stats is None:
            self._stats = estimate_market_stats(self._engine, self._primary_timeframe)
        avg_bar_volume, bar_volatility = self._stats.get(
            str(pos.instrument_id),
            (DEFAULT_AVG_BAR_VOLUME, DEFAULT_BAR_VOLATILITY),
        )
        return self._estimator.estimate_cost(
            entry_price=price,
            order_size=float(pos.peak_qty),
            avg_volume=avg_bar_volume,
            volatility=bar_volatility,
            spread=0.0001,
        )


def _scan_liquidation(
    nt_positions: list[Any],
    *,
    mode: str,
    leverage: float,
    starting_balance: float,
    catalog_path: Path | str,
    start_ns: int,
    end_ns: int,
    execution_bar_ns: int,
    funding_by_position: dict[str, list[tuple[int, float]]],
    cash_by_position: dict[str, list[tuple[int, float]]],
) -> Any:
    """Scan the raw positions of the run on MARK bars; returns the ``OverlayResult``.

    Funding and the other modeled cash arrive untruncated in ledger sign (negative = paid),
    per position key; the overlay truncates them at a liquidation.
    """
    from vibe_quant.liquidation.apply import apply_liquidation

    return apply_liquidation(
        nt_positions,
        mode=mode,  # type: ignore[arg-type]
        leverage=leverage,
        starting_balance=starting_balance,
        catalog_path=catalog_path,
        start_ns=start_ns,
        end_ns=end_ns,
        execution_bar_ns=execution_bar_ns,
        funding_by_position=funding_by_position,
        cash_by_position=cash_by_position,
    )


def _liquidated_run(
    costed: list[_PositionCosts],
    still_open: list[Any],
    overlay: Any,
    *,
    slip: _SlippageModel,
    funding_by_key: dict[str, FundingAccrual],
) -> tuple[list[_PositionCosts], dict[str, list[LedgerEvent]]]:
    """Trade numbers and ledgers of a run the overlay touched (a hit or a cross bust).

    ``costed`` holds the closed positions and the marked survivors; the open positions the
    venue liquidated or dropped come straight from the raw NT position (they were never
    marked): liquidated ones get their venue-fill numbers, dropped ones vanish. Order: closed
    first, then the open ones in position order, as without liquidation. The survivors'
    end-of-run valuation cash is added to the overlay ledgers (see
    :func:`_with_valuation_cash`).
    """
    ordered = [item for item in costed if item.mark is None]
    marked = {position_key(item.pos): item for item in costed if item.mark is not None}
    for pos in still_open:
        key = position_key(pos)
        if key in marked:
            ordered.append(marked[key])
            continue
        entry_slip = slip.leg(pos, float(pos.avg_px_open))
        ordered.append(
            _PositionCosts(  # placeholder: every number is replaced by _liquidated_costs
                pos=pos,
                mark=None,
                realized_pnl=0.0,
                exit_price=0.0,
                exit_ns=0,
                exit_reason="",
                pos_fees=0.0,
                slippage_cost=entry_slip,
                accrual=funding_by_key.get(key),  # the scan's input; cut at the hit later
                entry_slippage=entry_slip,
            )
        )
    adjusted: list[_PositionCosts] = []
    for item in ordered:
        key = position_key(item.pos)
        if key in overlay.dropped_keys:
            continue
        hit = overlay.hits_by_key.get(key)
        if hit is None:
            adjusted.append(item)
        else:
            adjusted.append(
                _liquidated_costs(item, hit, overlay.exit_override[key])
            )
    return adjusted, _with_valuation_cash(overlay.ledgers, adjusted)


def extract_trades(
    result: ValidationResult,
    engine: BacktestEngine,
    venue_config: VenueConfig,
    primary_timeframe: str | None = None,
    funding_calculator: FundingCalculator | None = None,
    run_start_date: str | None = None,
    run_end_date: str | None = None,
    execution_timeframe: str | None = None,
    catalog_path: Path | str | None = None,
    liquidation_mode: str = "off",
) -> bool:
    """Extract individual trade records from the engine's closed positions.

    Uses the Position objects from the engine cache directly, since the
    positions report DataFrame column names can vary across NT versions.

    Cost accounting: NT's realized_pnl already includes commissions. The
    post-fill SPEC slippage estimate and (when a calculator is provided)
    accrued funding are ADDITIONALLY charged into each trade's net_pnl, into
    the headline total_return, the trade profit factor and -- when the run
    window is known -- the daily-balance Sharpe/Sortino (same computation as
    screening, bd vibe-quant-e70tl.20).

    Liquidation (opt-in, bd vibe-quant-2gyk1.1.7): ``liquidation_mode`` "isolated" or
    "cross" replays the run's positions on 1m MARK bars
    (:func:`vibe_quant.liquidation.apply.apply_liquidation`). Without a liquidation the
    numbers are those of the default path (plus ``result.notes["liquidation"]``). With
    one, every metric is recomputed from the truncated ledgers: a liquidated position
    becomes a trade with ``exit_reason="liquidation"`` at the venue fill and the account
    loss (clearance fee included) as its net PnL, the clearance fee counted in
    ``total_fees`` / the trade's exit fee (a real cost: ``result.gross_pnl`` = net + fees +
    slippage + funding, so fee drag includes it), positions dropped by a cross bust
    produce no trade, ``total_return`` is the sum of the trade net PnLs, and win/loss
    counts, PF, Sharpe/Sortino, skew/kurtosis and the mark-to-market drawdown (floored
    from ``bust_ts`` in cross) follow. "off" (default) never imports the liquidation
    package.

    The scan runs on the raw positions before anything is marked: liquidated and dropped
    positions are valued by the overlay only, survivors are marked afterwards (an open
    survivor that cannot be priced fails the run closed instead of leaving the accounting).
    A survivor's end-of-run exit fee (and exit slippage leg) is a valuation cost booked after
    the last scanned instant: it counts in equity, ``total_fees`` and the trade's net, but
    never decides a hit; the equity may go below 0 honestly (no floor).

    Args:
        result: ValidationResult to populate trades on (mutated in place).
        engine: BacktestEngine after run.
        venue_config: Venue config for default leverage.
        primary_timeframe: Strategy timeframe for market-stat selection.
        funding_calculator: Optional post-hoc funding accrual.
        run_start_date / run_end_date: Backtest window for CAGR / Sharpe (required
            when ``liquidation_mode`` is on).
        execution_timeframe / catalog_path: Execution bars for the
            mark-to-market max drawdown (bd vibe-quant-e70tl.11); ``catalog_path``
            also holds the MARK bars for the liquidation scan.
        liquidation_mode: "off", "isolated" or "cross".

    Returns:
        True when the per-trade return moments (skewness/kurtosis) were recomputed from
        the liquidation ledgers; False when the caller still derives them from the engine.

    Raises:
        ValueError: unknown ``liquidation_mode``, or a mode without a run window /
            ``catalog_path`` / known ``execution_timeframe`` (the overlay's execution
            grid), or an open survivor with no LAST bar to value it; whatever
            ``apply_liquidation`` raises (missing MARK data fails the run closed).
    """
    if liquidation_mode not in LIQUIDATION_MODES:
        raise ValueError(
            f"liquidation_mode must be one of {LIQUIDATION_MODES}, got {liquidation_mode!r}"
        )
    window_start_ns = date_to_ns(run_start_date)
    window_end_ns = date_to_ns(run_end_date)
    has_window = (
        window_start_ns is not None
        and window_end_ns is not None
        and window_end_ns > window_start_ns
    )
    if liquidation_mode != "off" and (
        not has_window or window_start_ns is None or window_end_ns is None
    ):
        raise ValueError(
            f"liquidation_mode={liquidation_mode!r} needs a valid run window "
            f"(got {run_start_date!r}..{run_end_date!r})"
        )
    if liquidation_mode != "off" and catalog_path is None:
        raise ValueError(f"liquidation_mode={liquidation_mode!r} needs the catalog_path of the run")
    liquidation_bar_ns = (
        liquidation_execution_bar_ns(execution_timeframe) if liquidation_mode != "off" else 0
    )
    positions, still_open = split_positions(engine)

    # Positions still open when the engine stopped (with latency the on_stop
    # flatten order is still in flight) used to vanish from trades and
    # return. Mark them at the last execution-bar close instead, charging an
    # estimated taker exit fee, priced at the run's fee schedule (a real
    # VenueConfig exposes fee_schedule; fall back to the venue's taker rate).
    fee_schedule = getattr(venue_config, "fee_schedule", None)
    open_taker_fee = (
        fee_schedule.taker_rate
        if fee_schedule is not None
        else getattr(venue_config, "taker_fee", None)
    )
    liquidating = liquidation_mode != "off"
    default_leverage = int(venue_config.default_leverage)
    slip = _SlippageModel(engine, venue_config, primary_timeframe)
    funding_by_key: dict[str, FundingAccrual] = {}  # scan-time accruals (liquidation only)
    overlay: Any = None
    if liquidating and (positions or still_open):
        assert window_start_ns is not None and window_end_ns is not None
        assert catalog_path is not None
        # ONE scan, on the RAW positions, BEFORE any run-end marking: a liquidated or dropped
        # position is valued by the overlay, never by LAST bars, so no missing bar can make it
        # vanish. Its inputs are the cash that really happened: funding, and the modeled
        # slippage legs at their own instants. An open position is funded through the ts
        # mode-off would mark it at (its instrument's last execution bar), so a survivor
        # carries exactly the funding mode-off charges; the overlay truncates it at a hit.
        # The survivors' end-of-run exit fee is a valuation cost booked after the scan:
        # it never decides a hit and nothing is rescanned.
        final_bar_floor = window_end_ns - liquidation_bar_ns
        stale_last: dict[str, int | None] = {}  # open positions with no final-bar horizon
        scan_funding: dict[str, list[tuple[int, float]]] = {}
        scan_cash: dict[str, list[tuple[int, float]]] = {}
        for pos in (*positions, *still_open):
            key = position_key(pos)
            is_open = not pos.is_closed
            # window is [start, end): a position without a final-bar horizon (a data gap; a
            # survivor of it fails closed below) is funded to the end, short of end_ns itself
            horizon = window_end_ns - 1
            if is_open:
                last = last_execution_bar(
                    engine,
                    pos,
                    execution_timeframe=execution_timeframe,
                    catalog_path=catalog_path,
                    start_ns=window_start_ns,
                    end_ns=window_end_ns,
                )
                if last is not None and last[0] >= final_bar_floor:
                    horizon = last[0]
                else:
                    stale_last[key] = None if last is None else last[0]
            else:
                horizon = int(pos.ts_closed)
            if funding_calculator is not None:
                accrual = accrue_position_funding(funding_calculator, pos, exit_ns=horizon)
                funding_by_key[key] = accrual
                if accrual.payments:
                    scan_funding[key] = [(ts, -amount) for ts, amount in accrual.payments]
            cash: list[tuple[int, float]] = []
            entry_slip = slip.leg(pos, float(pos.avg_px_open))
            if entry_slip:
                cash.append((int(pos.ts_opened), -entry_slip))
            if not is_open:
                exit_slip = slip.leg(pos, float(pos.avg_px_close))
                if exit_slip:
                    cash.append((int(pos.ts_closed), -exit_slip))
            if cash:
                scan_cash[key] = cash
        overlay = _scan_liquidation(
            [*positions, *still_open],
            mode=liquidation_mode,
            leverage=float(venue_config.default_leverage),
            starting_balance=result.starting_balance,
            catalog_path=catalog_path,
            start_ns=window_start_ns,
            end_ns=window_end_ns,
            execution_bar_ns=liquidation_bar_ns,
            funding_by_position=scan_funding,
            cash_by_position=scan_cash,
        )
        touched = set(overlay.hits_by_key) | set(overlay.dropped_keys)
        survivors = [p for p in still_open if position_key(p) not in touched]
        for pos in survivors:
            key = position_key(pos)
            if key in stale_last:
                last_end = stale_last[key]
                raise ValueError(
                    f"liquidation_mode={liquidation_mode!r}: open position "
                    f"{pos.instrument_id} cannot be valued at the run end: its last "
                    f"{execution_timeframe} bar "
                    + ("is missing" if last_end is None else f"ends at ts {last_end}")
                    + f", before the run's final execution bar (run end ts {window_end_ns}). "
                    "Refusing to drop or mis-fund it in the liquidation accounting."
                )
    else:
        survivors = still_open
    # With liquidation on, only the survivors are marked (and logged below, in pass 2).
    open_marks = mark_open_positions(
        engine,
        survivors,
        execution_timeframe=execution_timeframe,
        catalog_path=catalog_path,
        start_ns=window_start_ns,
        end_ns=window_end_ns,
        taker_fee=open_taker_fee,
        log=not liquidating,
    )

    recomputed = overlay is not None and not overlay.is_identity
    if not positions and not open_marks and not recomputed:
        result.profit_factor = 0.0
        # No trades: 0.0, never NT's stat (bd vibe-quant-neuvu.4). Same rule as
        # the computed path below; this branch returns before it.
        result.win_rate = 0.0
        result.max_drawdown = 0.0
        if has_window:
            result.sharpe_ratio, result.sortino_ratio = 0.0, 0.0
        if liquidating:
            from vibe_quant.liquidation.scan import LiquidationResult

            result.notes["liquidation"] = LiquidationResult(
                mode=liquidation_mode, leverage=float(venue_config.default_leverage)
            ).to_notes(0)
        return False

    winning = 0
    losing = 0
    total_fees = 0.0
    total_slippage = 0.0

    total_funding = 0.0
    funding_fallbacks = 0
    # Realized balance changes for the daily Sharpe series
    cash_events: list[tuple[int, float]] = []
    # Modeled cash flows per instrument for the mark-to-market equity curve
    modeled_cash: dict[str, list[tuple[int, float]]] = {}

    unrealized_at_end = 0.0
    trade_inputs: list[tuple[Any, OpenPositionMark | None]] = [(p, None) for p in positions]
    trade_inputs.extend((mark.position, mark) for mark in open_marks)

    # Pass 1: every position's realized numbers and modeled costs (no accumulation).
    costed: list[_PositionCosts] = []
    for pos, mark in trade_inputs:
        entry_price = float(pos.avg_px_open)
        pos_fees = sum(float(c) for c in pos.commissions())
        if mark is None:
            realized_pnl = float(pos.realized_pnl)
            exit_price = float(pos.avg_px_close)
            exit_ns = int(pos.ts_closed)
            exit_reason = "signal"
        else:
            # NT's open-position realized_pnl is minus the entry commission
            realized_pnl = float(pos.realized_pnl) + mark.unrealized - mark.exit_fee
            exit_price = mark.price
            exit_ns = mark.ts
            exit_reason = "end_of_data"
            pos_fees += mark.exit_fee

        # SPEC slippage on BOTH legs (entry and exit fill); it used to be
        # charged on the entry notional only.
        entry_slip = slip.leg(pos, entry_price)
        exit_slip = slip.leg(pos, exit_price)
        slippage_cost = entry_slip + exit_slip

        # Post-hoc funding accrual from archived rates (positive = paid). A recomputed run
        # reuses the scan's accrual: the ledger the overlay built carries exactly that cash.
        # a liquidation scan already accrued it (to this very exit ts for closed positions
        # and marked survivors): reuse, the ledger carries exactly that funding
        pos_accrual: FundingAccrual | None = None
        if funding_by_key:
            pos_accrual = funding_by_key.get(position_key(pos))
        if pos_accrual is None and funding_calculator is not None:
            pos_accrual = accrue_position_funding(funding_calculator, pos, exit_ns=exit_ns)
        costed.append(
            _PositionCosts(
                pos=pos,
                mark=mark,
                realized_pnl=realized_pnl,
                exit_price=exit_price,
                exit_ns=exit_ns,
                exit_reason=exit_reason,
                pos_fees=pos_fees,
                slippage_cost=slippage_cost,
                accrual=pos_accrual,
                entry_slippage=entry_slip,
                exit_slippage=exit_slip,
            )
        )

    ledgers: dict[str, list[LedgerEvent]] | None = None
    if recomputed:
        costed, ledgers = _liquidated_run(
            costed,
            still_open,
            overlay,
            slip=slip,
            funding_by_key=funding_by_key,
        )

    # Pass 2: trade records and the accumulators (same order as before liquidation existed).
    return_moments: list[float] = []
    for item in costed:
        pos, mark = item.pos, item.mark
        entry_price = float(pos.avg_px_open)
        quantity = float(pos.peak_qty)
        realized_pnl = item.realized_pnl
        exit_price, exit_ns, exit_reason = item.exit_price, item.exit_ns, item.exit_reason
        pos_fees, slippage_cost = item.pos_fees, item.slippage_cost
        if mark is not None:
            unrealized_at_end += mark.unrealized - mark.exit_fee
            modeled_cash.setdefault(str(pos.instrument_id), []).append(
                (mark.ts, -mark.exit_fee)
            )
            if liquidating:
                log_open_mark(mark)
        total_fees += abs(pos_fees)
        total_slippage += slippage_cost

        entry_time = _ns_to_isoformat(pos.ts_opened)
        exit_time = _ns_to_isoformat(exit_ns) if exit_ns else None

        direction = position_direction(pos)
        instrument_id = str(pos.instrument_id)

        if item.accrual is not None:
            funding_fees = item.accrual.total
            funding_fallbacks += item.accrual.fallback_settlements
            cash_events.extend((ts, -amount) for ts, amount in item.accrual.payments)
            modeled_cash.setdefault(instrument_id, []).extend(
                (ts, -amount) for ts, amount in item.accrual.payments
            )
        else:
            funding_fees = 0.0
        total_funding += funding_fees

        # Net PnL: NT realized (fees included) minus modeled slippage/funding
        net_pnl = realized_pnl - slippage_cost - funding_fees
        if recomputed and item.entry_slippage:
            # the overlay ledger books each leg at its own instant: so does the daily series
            cash_events.append((int(pos.ts_opened), -item.entry_slippage))
            cash_events.append((exit_ns, realized_pnl - item.exit_slippage))
        else:
            cash_events.append((exit_ns, realized_pnl - slippage_cost))
        if slippage_cost:
            modeled_cash.setdefault(instrument_id, []).append((exit_ns, -slippage_cost))

        if net_pnl > 0:
            winning += 1
        elif net_pnl < 0:
            losing += 1

        if entry_price > 0 and quantity > 0:
            notional = entry_price * quantity
            roi_pct = (net_pnl / notional) * 100.0
        else:
            roi_pct = 0.0

        # Per-leg fees: a post-only maker entry pays the maker rate while its
        # stop-market exit pays the taker rate (NT's MakerTakerFeeModel charges
        # order.liquidity_side per fill). The pair always sums exactly to
        # pos_fees, so the per-trade TOTAL and every metric are unchanged.
        if item.fee_legs is not None:
            entry_fee, exit_fee = item.fee_legs  # liquidated: kept fills vs clearance fee
        else:
            entry_fee, exit_fee = trade_fee_legs(pos, pos_fees)

        # TODO: NT Position does not expose which child order (SL/TP/signal)
        # triggered the close. Detecting exit_reason from price vs SL/TP
        # levels requires correlating with OrderFilled events, which is not
        # readily available from the Position object alone. Closed trades
        # default to "signal"; open-at-end trades are "end_of_data"; liquidated
        # ones (opt-in overlay) are "liquidation".

        trade = TradeRecord(
            symbol=instrument_id,
            direction=direction,
            leverage=default_leverage,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            entry_fee=entry_fee,
            exit_fee=exit_fee,
            funding_fees=funding_fees,
            slippage_cost=slippage_cost,
            gross_pnl=(
                item.gross_pnl if item.gross_pnl is not None else realized_pnl + abs(pos_fees)
            ),
            net_pnl=net_pnl,
            roi_percent=roi_pct,
            exit_reason=exit_reason,
        )
        result.trades.append(trade)
        if mark is None and abs(quantity * entry_price) > 0:
            # same per-trade return as trade_ledger.closed_returns (closed positions only)
            return_moments.append(realized_pnl / abs(quantity * entry_price))

    result.total_trades = len(result.trades)
    result.trades.sort(key=lambda t: t.entry_time)
    if recomputed:
        # NT's counters/stats know nothing of the liquidation: nothing from extract_stats
        # that depends on the (untruncated) positions survives
        result.avg_win = result.avg_loss = 0.0
    result.winning_trades = winning
    result.losing_trades = losing
    result.total_fees = total_fees
    result.total_slippage = total_slippage
    result.total_funding = total_funding
    result.funding_fallback_settlements = funding_fallbacks
    if funding_fallbacks:
        logger.warning(
            "Validation funding: %d settlement(s) charged at the fallback rate "
            "(no archived rate)",
            funding_fallbacks,
        )
    # Trade win rate and PF on net PnL (fees, modeled slippage and funding
    # included) — same definitions as screening (bd vibe-quant-e70tl.7,
    # bd vibe-quant-neuvu.4). Zero trades -> 0.0.
    result.win_rate = trade_win_rate(t.net_pnl for t in result.trades)
    result.profit_factor = profit_factor(t.net_pnl for t in result.trades)

    if recomputed:
        # The headline is the sum of the (truncated) trades: NT's PnL% still holds the
        # strategy's own exits of liquidated positions and the dropped positions' PnL.
        result.total_return = (
            sum(t.net_pnl for t in result.trades) / result.starting_balance
            if result.starting_balance > 0
            else 0.0
        )
    else:
        # Charge modeled slippage + funding into the headline return so it is
        # consistent with the per-trade net PnL (NT's stats know nothing of
        # either post-fill cost).
        extra_costs = total_slippage + total_funding - unrealized_at_end
        if extra_costs != 0.0 and result.starting_balance > 0:
            result.total_return -= extra_costs / result.starting_balance

    # Daily realized-balance Sharpe/Sortino including slippage + funding,
    # spanning the whole run window (shared with screening).
    if window_start_ns is not None and window_end_ns is not None and has_window:
        result.sharpe_ratio, result.sortino_ratio = daily_sharpe_sortino(
            result.starting_balance, cash_events, window_start_ns, window_end_ns
        )

    # Mark-to-market max drawdown incl. open-trade intrabar losses, funding
    # and slippage — same function as screening (bd vibe-quant-e70tl.11).
    # NT's MaxDrawdown stat is a daily realized-balance figure: not used.
    # A liquidation replaces the rebuilt ledger by the overlay's truncated one (it already
    # carries every modeled cash flow) and floors the account at 0 from a cross bust on.
    mtm_dd = mark_to_market_drawdown(
        engine,
        result.starting_balance,
        execution_timeframe=execution_timeframe,
        catalog_path=catalog_path,
        start_ns=window_start_ns,
        end_ns=window_end_ns,
        extra_cash=None if recomputed else modeled_cash,
        ledgers=ledgers,
        floor_from_ns=overlay.result.bust_ts if recomputed and liquidation_mode == "cross" else None,
    )
    if mtm_dd is None:
        logger.warning(
            "Mark-to-market drawdown unavailable (timeframe=%s) — using "
            "closed-trade drawdown, which ignores open-trade losses",
            execution_timeframe,
        )
        mtm_dd = _compute_max_drawdown_from_trades(result.trades, result.starting_balance)
    result.max_drawdown = mtm_dd

    compute_extended_metrics(result, run_start_date=run_start_date, run_end_date=run_end_date)

    # Cost-free PnL = net + every cost paid (it was never set before: always 0.0). With
    # liquidation, fees INCLUDE the clearance fee (``total_fees`` = commissions + clearance;
    # ``notes["liquidation"]["total_forfeited"]`` is the clearance part on its own and
    # overlaps it: never add the two).
    result.gross_pnl = gross_pnl(
        sum(t.net_pnl for t in result.trades), total_fees, total_slippage, total_funding
    )
    if overlay is not None:
        result.notes["liquidation"] = overlay.result.to_notes(len(result.trades))
        result.liquidation_events = [
            {**hit.to_dict(), "mode": overlay.result.mode} for hit in overlay.result.hits
        ]
    if recomputed:
        apply_validation_moments(result, return_moments)
    return recomputed


#: Conservative fallbacks when no usable bar data is in the cache
DEFAULT_AVG_BAR_VOLUME = 1000.0
DEFAULT_BAR_VOLATILITY = 0.02


def estimate_market_stats(
    engine: BacktestEngine,
    primary_timeframe: str | None = None,
) -> dict[str, tuple[float, float]]:
    """Estimate per-instrument bar volume and bar-level volatility.

    Bars in the engine cache span multiple instruments AND timeframes
    (strategy bars, additional-TF bars, sub-bar detail data). Computing
    log returns over that interleaved series produces spurious jumps
    (BTC bar -> ETH bar), wildly inflating volatility and therefore the
    SPEC slippage estimate. Instead, group bars by bar_type, compute
    stats per group, and pick one group per instrument — preferring the
    strategy's primary timeframe, else the group with the most bars.

    NOTE: The volatility returned is per-bar (std of log returns between
    consecutive bars of the SAME bar type), NOT annualized or daily.
    SlippageEstimator uses it as a relative magnitude input.

    Args:
        engine: BacktestEngine after run.
        primary_timeframe: Strategy primary timeframe (e.g. "1h") used to
            select the representative bar group per instrument.

    Returns:
        Mapping of instrument_id string -> (avg_bar_volume, bar_volatility).
        Missing instruments should fall back to
        (DEFAULT_AVG_BAR_VOLUME, DEFAULT_BAR_VOLATILITY).
    """
    # Expected bar-spec fragment for the primary timeframe, e.g. "1-HOUR"
    primary_spec: str | None = None
    if primary_timeframe:
        try:
            from vibe_quant.data.catalog import INTERVAL_TO_AGGREGATION

            if primary_timeframe in INTERVAL_TO_AGGREGATION:
                step, agg = INTERVAL_TO_AGGREGATION[primary_timeframe]
                primary_spec = f"{step}-{agg.name}"
        except Exception:
            primary_spec = None

    try:
        bars = engine.kernel.cache.bars()
    except Exception:
        logger.debug("Could not read bars from engine cache", exc_info=True)
        return {}
    if not bars:
        return {}

    # Group by full bar type (instrument + spec), chronologically ordered
    groups: dict[str, list[Any]] = {}
    for bar in bars:
        groups.setdefault(str(bar.bar_type), []).append(bar)
    for group in groups.values():
        group.sort(key=lambda b: int(b.ts_init))

    def _group_stats(group: list[Any]) -> tuple[float, float]:
        volumes = [float(b.volume) for b in group if float(b.volume) > 0]
        closes = [float(b.close) for b in group if float(b.close) > 0]
        avg_volume = sum(volumes) / len(volumes) if volumes else DEFAULT_AVG_BAR_VOLUME
        volatility = DEFAULT_BAR_VOLATILITY
        if len(closes) >= 3:
            log_returns = [
                math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))
            ]
            mean_r = sum(log_returns) / len(log_returns)
            var = sum((r - mean_r) ** 2 for r in log_returns) / (len(log_returns) - 1)
            if var > 0:
                volatility = math.sqrt(var)
        return avg_volume, volatility

    # Pick one representative group per instrument
    result: dict[str, tuple[float, float]] = {}
    chosen: dict[str, tuple[str, int]] = {}  # instrument -> (bar_type, size)
    for bar_type_str, group in groups.items():
        # "BTCUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL" -> instrument is the
        # part before the bar spec; instrument ids contain exactly one dot.
        instrument_id = str(group[0].bar_type.instrument_id)
        is_primary = primary_spec is not None and f"-{primary_spec}-" in bar_type_str
        prev = chosen.get(instrument_id)
        prev_is_primary = prev is not None and primary_spec is not None and (
            f"-{primary_spec}-" in prev[0]
        )
        if prev is None or (is_primary and not prev_is_primary) or (
            is_primary == prev_is_primary and len(group) > prev[1]
        ):
            chosen[instrument_id] = (bar_type_str, len(group))
            result[instrument_id] = _group_stats(group)

    return result


def _compute_max_drawdown_from_trades(
    trades: list[TradeRecord],
    starting_balance: float,
) -> float:
    """Compute max drawdown from trade PnLs as equity curve.

    NT 1.222+ removed the MaxDrawdown indicator, so stats may not contain it.
    This reconstructs the equity curve from cumulative net PnL and computes
    max drawdown = max((peak - current) / peak) over the curve.

    Args:
        trades: Sorted list of trade records.
        starting_balance: Starting account balance.

    Returns:
        Max drawdown as a positive fraction (e.g. 0.12 for 12%).
    """
    if not trades or starting_balance <= 0:
        return 0.0

    equity = starting_balance
    peak = equity
    max_dd = 0.0

    for trade in trades:
        equity += trade.net_pnl
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

    return max_dd


def compute_extended_metrics(
    result: ValidationResult,
    run_start_date: str | None = None,
    run_end_date: str | None = None,
) -> None:
    """Compute SPEC-required extended metrics from trades.

    Populates: largest_win/loss, avg_win/loss, max_consecutive_wins/losses,
    avg_trade_duration_hours, cagr, volatility_annual, calmar_ratio.

    Args:
        result: ValidationResult to populate (mutated in place).
        run_start_date / run_end_date: Backtest window (YYYY-MM-DD). When
            provided, CAGR annualizes over the run window; the old
            first-trade..last-trade span inflates CAGR for strategies that
            trade sparsely within a longer window.
    """
    if not result.trades:
        return

    wins: list[float] = []
    losses: list[float] = []
    durations_hours: list[float] = []

    max_con_wins = 0
    max_con_losses = 0
    cur_wins = 0
    cur_losses = 0

    for trade in result.trades:
        pnl = trade.net_pnl
        if pnl > 0:
            wins.append(pnl)
            cur_wins += 1
            max_con_wins = max(max_con_wins, cur_wins)
            cur_losses = 0
        elif pnl < 0:
            losses.append(pnl)
            cur_losses += 1
            max_con_losses = max(max_con_losses, cur_losses)
            cur_wins = 0
        else:
            cur_wins = 0
            cur_losses = 0

        if trade.entry_time and trade.exit_time:
            try:
                entry_dt = datetime.fromisoformat(trade.entry_time.replace("Z", "+00:00"))
                exit_dt = datetime.fromisoformat(trade.exit_time.replace("Z", "+00:00"))
                duration_h = (exit_dt - entry_dt).total_seconds() / 3600.0
                if duration_h >= 0:
                    durations_hours.append(duration_h)
            except (ValueError, TypeError):
                logger.warning(
                    "Trade duration not computed for entry %s / exit %s",
                    trade.entry_time,
                    trade.exit_time,
                )

    result.max_consecutive_wins = max_con_wins
    result.max_consecutive_losses = max_con_losses

    if wins:
        result.largest_win = max(wins)
        result.avg_win = sum(wins) / len(wins)
    if losses:
        result.largest_loss = min(losses)
        result.avg_loss = sum(losses) / len(losses)

    if durations_hours:
        result.avg_trade_duration_hours = sum(durations_hours) / len(durations_hours)

    if result.total_return != 0.0 and result.trades:
        try:
            days: float | None = None
            if run_start_date and run_end_date:
                try:
                    window_start = datetime.fromisoformat(run_start_date)
                    window_end = datetime.fromisoformat(run_end_date)
                    days = max((window_end - window_start).total_seconds() / 86400.0, 1.0)
                except ValueError:
                    logger.warning(
                        "CAGR run window unparseable (%s..%s); using trade span",
                        run_start_date,
                        run_end_date,
                    )
                    days = None
            if days is None:
                # Fallback: trade span (inflates CAGR for sparse traders)
                first_entry = datetime.fromisoformat(
                    result.trades[0].entry_time.replace("Z", "+00:00")
                )
                last_exit_str = result.trades[-1].exit_time or result.trades[-1].entry_time
                last_exit = datetime.fromisoformat(last_exit_str.replace("Z", "+00:00"))
                days = max((last_exit - first_entry).total_seconds() / 86400.0, 1.0)
            # total_return is stored as a decimal fraction from NT stats
            # (e.g. 0.12 = 12%). Use directly — no heuristic conversion.
            total_return_frac = result.total_return
            if total_return_frac == -1.0:
                # 100% loss: CAGR is -1.0 regardless of duration
                result.cagr = -1.0
            elif total_return_frac > -1.0:
                result.cagr = ((1.0 + total_return_frac) ** (365.0 / days)) - 1.0
        except (ValueError, TypeError) as exc:
            logger.warning("CAGR not computed (%s)", type(exc).__name__)

    # Note: computes volatility from individual trade returns, not daily
    # equity returns. This may differ from standard annual volatility
    # measures that use daily mark-to-market returns.
    if len(result.trades) >= 2:
        trade_returns = [t.roi_percent / 100.0 for t in result.trades if t.roi_percent != 0.0]
        if len(trade_returns) >= 2:
            mean_r = sum(trade_returns) / len(trade_returns)
            var = sum((r - mean_r) ** 2 for r in trade_returns) / (len(trade_returns) - 1)
            if durations_hours:
                avg_dur_days = max(sum(durations_hours) / len(durations_hours) / 24.0, 0.01)
                trades_per_year = 365.0 / avg_dur_days
            else:
                trades_per_year = 252.0
            result.volatility_annual = math.sqrt(var * trades_per_year) if var > 0 else 0.0

    if result.max_drawdown > 0 and result.cagr != 0:
        result.calmar_ratio = result.cagr / result.max_drawdown
