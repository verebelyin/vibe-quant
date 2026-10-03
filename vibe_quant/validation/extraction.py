"""Result extraction helpers for validation runner.

Extracts metrics and trades from NautilusTrader backtest output
into vibe-quant's ValidationResult format. The cost/equity helpers in the
"Shared metric helpers" section are also used by the screening runner so
both tiers compute PF, funding and Sharpe the same way.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from vibe_quant.metrics import (
    BarSeries,
    LedgerEvent,
    daily_balance_returns,
    mark_to_market_max_drawdown,
    pnl_path,
    profit_factor,
)
from vibe_quant.validation.fill_model import SlippageEstimator
from vibe_quant.validation.results import TradeRecord, ValidationResult

if TYPE_CHECKING:
    from collections.abc import Iterable
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

    bar_dir = _Path(catalog_path) / "data" / "bar" / bar_type
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


def mark_open_positions(
    engine: Any,
    positions: Iterable[Any],
    *,
    execution_timeframe: str | None,
    catalog_path: Path | str | None = None,
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> list[OpenPositionMark]:
    """Mark open positions at the last execution-bar close in the window.

    The exit fee is estimated at the instrument's taker rate (the flatten
    would be a market order). Positions that cannot be priced are skipped
    with a WARNING.
    """
    from vibe_quant.data.catalog import INTERVAL_TO_AGGREGATION

    marks: list[OpenPositionMark] = []
    for pos in positions:
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
            logger.warning(
                "Position %s still open at end of run but no %s bars to mark it -- "
                "excluded from trades and return",
                instrument_id,
                execution_timeframe,
            )
            continue
        price = float(series.close[-1])
        qty = float(pos.signed_qty)
        try:
            taker = float(engine.kernel.cache.instrument(pos.instrument_id).taker_fee)
        except Exception:
            taker = 0.0005
        mark = OpenPositionMark(
            position=pos,
            ts=int(series.ts[-1]),
            price=price,
            unrealized=qty * (price - float(pos.avg_px_open)),
            exit_fee=abs(qty) * price * taker,
        )
        logger.warning(
            "Position %s (qty %s @ %s) still open at end of run -- marked at %s "
            "(unrealized %.2f, est. exit fee %.4f)",
            instrument_id,
            qty,
            float(pos.avg_px_open),
            price,
            mark.unrealized,
            mark.exit_fee,
        )
        marks.append(mark)
    return marks


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
) -> float | None:
    """Mark-to-market max drawdown of the run (intrabar adverse extremes).

    Returns None when it cannot be computed (no fill events / no bars for a
    traded instrument); callers then fall back to the closed-trade curve and
    MUST say so in logs.
    """
    from vibe_quant.data.catalog import INTERVAL_TO_AGGREGATION

    positions = all_positions(engine)
    if not positions:
        return 0.0
    try:
        ledgers = position_ledger_events(positions, extra_cash)
    except AttributeError:
        return None
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
    return mark_to_market_max_drawdown(starting_balance, paths)


def _ns_to_isoformat(ns_timestamp: int | float | str) -> str:
    """Convert a nanosecond Unix timestamp to ISO 8601 string.

    NautilusTrader Position.ts_opened / ts_closed are uint64
    nanosecond timestamps.  ``str()`` gives a bare integer string
    which breaks ``datetime.fromisoformat()``.
    """
    ns = int(ns_timestamp)
    return datetime.fromtimestamp(ns / 1e9, tz=UTC).isoformat()


def _extract_return_moments(result: ValidationResult, engine: BacktestEngine) -> None:
    """Compute skewness/kurtosis from closed positions and set on result."""
    try:
        cache = engine.kernel.cache
        all_positions = list(cache.positions()) + list(cache.position_snapshots())
        closed = [p for p in all_positions if p.is_closed]
        if len(closed) < 4:
            return

        returns: list[float] = []
        for pos in closed:
            entry_val = abs(float(pos.peak_qty) * float(pos.avg_px_open))
            if entry_val > 0:
                returns.append(float(pos.realized_pnl) / entry_val)

        n = len(returns)
        if n < 4:
            return

        mean = sum(returns) / n
        diffs = [r - mean for r in returns]
        m2 = sum(d * d for d in diffs) / n
        if m2 == 0:
            return

        m3 = sum(d**3 for d in diffs) / n
        m4 = sum(d**4 for d in diffs) / n

        result.skewness = round(
            (math.sqrt(n * (n - 1)) / (n - 2)) * (m3 / m2**1.5), 4
        )
        excess = ((n + 1) * (n - 1) * m4) / (
            (n - 2) * (n - 3) * m2**2
        ) - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
        result.kurtosis = round(max(1.0, excess + 3.0), 4)
    except Exception:
        logger.warning("Could not compute return moments", exc_info=True)


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
    extract_trades(
        result,
        engine,
        venue_config,
        primary_timeframe=primary_timeframe,
        funding_calculator=funding_calculator,
        run_start_date=run_start_date,
        run_end_date=run_end_date,
        execution_timeframe=execution_timeframe,
        catalog_path=catalog_path,
    )
    _extract_return_moments(result, engine)

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
    stats_returns = bt_result.stats_returns or {}
    stats_pnls = bt_result.stats_pnls or {}

    _known_pnl_keys = {
        "pnl (total)",
        "pnl% (total)",
        "sharpe",
        "sortino",
        "max drawdown",
        "win rate",
        "profit factor",
        "expectancy",
        "avg winner",
        "avg loser",
        "long ratio",
    }

    # Track which fields were populated from stats_pnls so we only
    # fall back to stats_returns for fields that weren't set.
    _populated: set[str] = set()

    for _currency, pnl_stats in stats_pnls.items():
        for key, value in pnl_stats.items():
            if value is None:
                continue
            key_lower = key.lower()
            try:
                fval = float(value)
            except (ValueError, TypeError):
                logger.debug("Non-numeric PnL stat skipped: %s = %r", key, value)
                continue
            if key_lower == "pnl% (total)":
                # NT reports as percentage (e.g. -13.06 for -13.06%);
                # we store as fraction (e.g. -0.1306)
                result.total_return = fval / 100.0
                _populated.add("total_return")
            elif "sharpe" in key_lower:
                result.sharpe_ratio = fval
                _populated.add("sharpe_ratio")
            elif "sortino" in key_lower:
                result.sortino_ratio = fval
                _populated.add("sortino_ratio")
            elif key_lower == "max drawdown":
                result.max_drawdown = abs(fval)
                _populated.add("max_drawdown")
            elif key_lower == "win rate":
                result.win_rate = fval
                _populated.add("win_rate")
            elif key_lower == "avg winner":
                result.avg_win = fval
            elif key_lower == "avg loser":
                result.avg_loss = fval
            elif not any(k in key_lower for k in _known_pnl_keys):
                logger.debug("Unmatched PnL stats key: %s = %s", key, value)

    _known_returns_keys = {
        "sharpe",
        "sortino",
        "max drawdown",
        "win rate",
        "profit factor",
        "expectancy",
        "avg winner",
        "avg loser",
        "long ratio",
    }
    for key, value in stats_returns.items():
        if value is None:
            continue
        key_lower = key.lower()
        try:
            fval = float(value)
        except (ValueError, TypeError):
            logger.debug("Non-numeric returns stat skipped: %s = %r", key, value)
            continue
        if "sharpe" in key_lower and "sharpe_ratio" not in _populated:
            result.sharpe_ratio = fval
        elif "sortino" in key_lower and "sortino_ratio" not in _populated:
            result.sortino_ratio = fval
        elif "max drawdown" in key_lower and "max_drawdown" not in _populated:
            result.max_drawdown = abs(fval)
        elif key_lower == "win rate" and "win_rate" not in _populated:
            result.win_rate = fval
        elif not any(k in key_lower for k in _known_returns_keys):
            logger.debug("Unmatched returns stats key: %s = %s", key, value)


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
) -> None:
    """Extract individual trade records from the engine's closed positions.

    Uses the Position objects from the engine cache directly, since the
    positions report DataFrame column names can vary across NT versions.

    Cost accounting: NT's realized_pnl already includes commissions. The
    post-fill SPEC slippage estimate and (when a calculator is provided)
    accrued funding are ADDITIONALLY charged into each trade's net_pnl, into
    the headline total_return, the trade profit factor and -- when the run
    window is known -- the daily-balance Sharpe/Sortino (same computation as
    screening, bd vibe-quant-e70tl.20).

    Args:
        result: ValidationResult to populate trades on (mutated in place).
        engine: BacktestEngine after run.
        venue_config: Venue config for default leverage.
        primary_timeframe: Strategy timeframe for market-stat selection.
        funding_calculator: Optional post-hoc funding accrual.
        run_start_date / run_end_date: Backtest window for CAGR / Sharpe.
        execution_timeframe / catalog_path: Execution bars for the
            mark-to-market max drawdown (bd vibe-quant-e70tl.11).
    """
    window_start_ns = date_to_ns(run_start_date)
    window_end_ns = date_to_ns(run_end_date)
    has_window = (
        window_start_ns is not None
        and window_end_ns is not None
        and window_end_ns > window_start_ns
    )
    try:
        # NT netting mode reuses position IDs: when a position closes and reopens,
        # it's removed from _index_positions_closed. The closed state is preserved
        # as a "snapshot". We must combine positions() + position_snapshots() and
        # filter by is_closed, exactly as NT's own "Total positions" log does.
        everything = all_positions(engine)
        positions = [p for p in everything if p.is_closed]
        still_open = [p for p in everything if not p.is_closed and getattr(p, "is_open", False)]
    except Exception:
        logger.warning("Could not read positions from engine cache", exc_info=True)
        positions, still_open = [], []

    # Positions still open when the engine stopped (with latency the on_stop
    # flatten order is still in flight) used to vanish from trades and
    # return. Mark them at the last execution-bar close instead, charging an
    # estimated taker exit fee.
    open_marks = mark_open_positions(
        engine,
        still_open,
        execution_timeframe=execution_timeframe,
        catalog_path=catalog_path,
        start_ns=window_start_ns,
        end_ns=window_end_ns,
    )

    if not positions and not open_marks:
        result.profit_factor = 0.0
        result.max_drawdown = 0.0
        if has_window:
            result.sharpe_ratio, result.sortino_ratio = 0.0, 0.0
        return

    default_leverage = int(venue_config.default_leverage)
    winning = 0
    losing = 0
    total_fees = 0.0
    total_slippage = 0.0

    fill_cfg = venue_config.fill_config
    impact_k = getattr(fill_cfg, "impact_coefficient", 0.1) if fill_cfg else 0.1
    engine_prob_slippage = (
        float(getattr(fill_cfg, "prob_slippage", 0.0)) if fill_cfg is not None else 0.0
    )
    use_post_fill_spec_slippage = engine_prob_slippage <= 0.0
    slippage_estimator = (
        SlippageEstimator(impact_coefficient=impact_k) if use_post_fill_spec_slippage else None
    )

    if not use_post_fill_spec_slippage:
        logger.info(
            "Skipping post-fill SPEC slippage estimation because engine "
            "prob_slippage=%s is enabled",
            engine_prob_slippage,
        )

    market_stats = estimate_market_stats(engine, primary_timeframe)

    total_funding = 0.0
    funding_fallbacks = 0
    # Realized balance changes for the daily Sharpe series
    cash_events: list[tuple[int, float]] = []
    # Modeled cash flows per instrument for the mark-to-market equity curve
    modeled_cash: dict[str, list[tuple[int, float]]] = {}

    unrealized_at_end = 0.0
    trade_inputs: list[tuple[Any, OpenPositionMark | None]] = [(p, None) for p in positions]
    trade_inputs.extend((mark.position, mark) for mark in open_marks)
    for pos, mark in trade_inputs:
        entry_price = float(pos.avg_px_open)
        # pos.quantity is 0 for closed positions; use peak_qty for trade size
        quantity = float(pos.peak_qty)
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
            unrealized_at_end += mark.unrealized - mark.exit_fee
            modeled_cash.setdefault(str(pos.instrument_id), []).append(
                (mark.ts, -mark.exit_fee)
            )
        total_fees += abs(pos_fees)

        avg_bar_volume, bar_volatility = market_stats.get(
            str(pos.instrument_id),
            (DEFAULT_AVG_BAR_VOLUME, DEFAULT_BAR_VOLATILITY),
        )
        if slippage_estimator is not None:
            # SPEC slippage on BOTH legs (entry and exit fill); it used to be
            # charged on the entry notional only.
            slippage_cost = sum(
                slippage_estimator.estimate_cost(
                    entry_price=leg_price,
                    order_size=quantity,
                    avg_volume=avg_bar_volume,
                    volatility=bar_volatility,
                    spread=0.0001,
                )
                for leg_price in (entry_price, exit_price)
            )
        else:
            slippage_cost = 0.0
        total_slippage += slippage_cost

        entry_time = _ns_to_isoformat(pos.ts_opened)
        exit_time = _ns_to_isoformat(exit_ns) if exit_ns else None

        direction = position_direction(pos)
        instrument_id = str(pos.instrument_id)

        # Post-hoc funding accrual from archived rates (positive = paid)
        if funding_calculator is not None:
            accrual = accrue_position_funding(funding_calculator, pos, exit_ns=exit_ns)
            funding_fees = accrual.total
            funding_fallbacks += accrual.fallback_settlements
            cash_events.extend((ts, -amount) for ts, amount in accrual.payments)
            modeled_cash.setdefault(instrument_id, []).extend(
                (ts, -amount) for ts, amount in accrual.payments
            )
        else:
            funding_fees = 0.0
        total_funding += funding_fees

        # Net PnL: NT realized (fees included) minus modeled slippage/funding
        net_pnl = realized_pnl - slippage_cost - funding_fees
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

        # Split fees 50/50 between entry and exit.
        # NT's MakerTakerFeeModel uses order.liquidity_side per fill:
        # market/stop orders → taker, limit orders → maker. Our strategies
        # use market entry + stop-market SL/TP, so both fills are taker.
        # The Position only exposes total commissions, not per-fill breakdown,
        # so we split evenly (both sides same rate in practice).
        entry_fee = abs(pos_fees) / 2.0
        exit_fee = abs(pos_fees) / 2.0

        # TODO: NT Position does not expose which child order (SL/TP/signal)
        # triggered the close. Detecting exit_reason from price vs SL/TP
        # levels requires correlating with OrderFilled events, which is not
        # readily available from the Position object alone. Closed trades
        # default to "signal"; open-at-end trades are "end_of_data".

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
            gross_pnl=realized_pnl + abs(pos_fees),
            net_pnl=net_pnl,
            roi_percent=roi_pct,
            exit_reason=exit_reason,
        )
        result.trades.append(trade)

    result.total_trades = len(result.trades)
    result.trades.sort(key=lambda t: t.entry_time)
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
    if result.total_trades > 0:
        result.win_rate = winning / result.total_trades
    # Trade PF on net PnL (fees, modeled slippage and funding included) —
    # same definition as screening (bd vibe-quant-e70tl.7).
    result.profit_factor = profit_factor(t.net_pnl for t in result.trades)

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
    mtm_dd = mark_to_market_drawdown(
        engine,
        result.starting_balance,
        execution_timeframe=execution_timeframe,
        catalog_path=catalog_path,
        start_ns=window_start_ns,
        end_ns=window_end_ns,
        extra_cash=modeled_cash,
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
                pass

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
        except (ValueError, TypeError):
            pass

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
