"""Shared performance metrics used across screening and validation pipelines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

#: Finite stand-in for an infinite profit factor (profits but no losing trade).
#: Every computed PF is clamped to this value so ``inf``/``NaN`` never reach the
#: DB, Pareto ranking or the UI. Discovery fitness clamps PF to 5 anyway, so any
#: value >= 5 scores identically there; 100 is chosen to read as "no losses" in
#: the UI rather than as a plausible measured PF.
PROFIT_FACTOR_CAP: float = 100.0


def profit_factor(trade_pnls: Iterable[float]) -> float:
    """Trade profit factor: gross profit / gross loss over closed trades.

    ``trade_pnls`` are per-trade NET PnLs (after fees and, where modeled,
    funding/slippage). This is the single PF definition shared by screening
    and validation (bd vibe-quant-e70tl.7 — previously NT's daily-return PF
    was reported, which is a different statistic).

    Returns:
        - ``0.0`` when there are no trades or no winning trades,
        - :data:`PROFIT_FACTOR_CAP` when there are winners but no losers,
        - otherwise ``gross_profit / gross_loss`` clamped to the cap.
        Non-finite PnLs are ignored.
    """
    gross_profit = 0.0
    gross_loss = 0.0
    for pnl in trade_pnls:
        if not math.isfinite(pnl):
            continue
        if pnl > 0.0:
            gross_profit += pnl
        elif pnl < 0.0:
            gross_loss -= pnl
    if gross_profit <= 0.0:
        return 0.0
    if gross_loss <= 0.0:
        return PROFIT_FACTOR_CAP
    return min(gross_profit / gross_loss, PROFIT_FACTOR_CAP)


_DAY_NS = 86_400 * 1_000_000_000


def daily_balance_returns(
    starting_balance: float,
    cash_events: Iterable[tuple[int, float]],
    start_ns: int,
    end_ns: int,
) -> dict[int, float]:
    """Daily returns of the realized account balance over the whole window.

    Mirrors NT's ``PortfolioAnalyzer`` portfolio returns (UTC daily last
    balance, forward-filled, ``pct_change``) but (a) lets callers add cash
    flows NT never sees (funding, modeled slippage) and (b) spans the full
    backtest window instead of stopping at the last fill, so idle days after
    the last trade count as zero-return days.

    Args:
        starting_balance: Account balance at ``start_ns``.
        cash_events: ``(ts_ns, delta)`` realized balance changes (trade net
            PnL at close, funding payments as negative deltas, ...). Events
            outside the window are clamped to its first/last day.
        start_ns / end_ns: Backtest window (end exclusive).

    Returns:
        ``{day_start_ns: return}`` for every day after the first, the format
        NT's pyo3 ``SharpeRatio.calculate_from_returns`` expects.
    """
    first_day = start_ns // _DAY_NS
    last_day = max(first_day, (end_ns - 1) // _DAY_NS)
    n_days = last_day - first_day + 1
    per_day = [0.0] * n_days
    for ts, delta in cash_events:
        idx = min(max(ts // _DAY_NS - first_day, 0), n_days - 1)
        per_day[idx] += delta
    returns: dict[int, float] = {}
    balance = starting_balance + per_day[0]
    for i in range(1, n_days):
        prev = balance
        balance += per_day[i]
        if prev != 0.0:
            ret = balance / prev - 1.0
            if math.isfinite(ret):
                returns[(first_day + i) * _DAY_NS] = ret
    return returns


# ---------------------------------------------------------------------------
# Mark-to-market equity / max drawdown (bd vibe-quant-e70tl.11)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LedgerEvent:
    """One account event for a single instrument.

    Attributes:
        ts: Event timestamp (ns).
        qty: Signed fill quantity (+buy / -sell); 0 for cash-only events.
        price: Fill price (ignored for cash-only events).
        cash: Realized cash delta besides trading PnL (``-commission``,
            ``-funding``, ``-slippage``).
        resting: True for resting-order fills (limit / triggered stop): they
            happened while the bar was being processed, at the point of the
            bar path where the price reached ``price``. False for market
            fills and cash events, which are located by time: before the bar
            if ``ts`` precedes the bar's ``ts_init``, after its close if equal.
    """

    ts: int
    qty: float = 0.0
    price: float = 0.0
    cash: float = 0.0
    resting: bool = False


@dataclass(frozen=True, slots=True)
class BarSeries:
    """OHLC arrays of the execution bars (sorted by ``ts`` = bar ``ts_init``)."""

    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray

    def __len__(self) -> int:
        return len(self.ts)

    def window(self, start_ns: int | None, end_ns: int | None) -> BarSeries:
        """Bars with ``start_ns <= ts <= end_ns`` (views, no copy)."""
        lo = 0 if start_ns is None else int(np.searchsorted(self.ts, start_ns, side="left"))
        hi = len(self.ts) if end_ns is None else int(np.searchsorted(self.ts, end_ns, side="right"))
        return BarSeries(
            self.ts[lo:hi], self.open[lo:hi], self.high[lo:hi], self.low[lo:hi], self.close[lo:hi]
        )


_QTY_EPS = 1e-12


class _Book:
    """Average-cost netting position + realized cash for one instrument."""

    __slots__ = ("avg", "qty", "realized")

    def __init__(self) -> None:
        self.qty = 0.0
        self.avg = 0.0
        self.realized = 0.0

    def apply(self, ev: LedgerEvent) -> None:
        self.realized += ev.cash
        if ev.qty == 0.0:
            return
        q, fill = self.qty, ev.qty
        if q == 0.0 or (q > 0.0) == (fill > 0.0):
            new_q = q + fill
            self.avg = (abs(q) * self.avg + abs(fill) * ev.price) / abs(new_q)
            self.qty = new_q
            return
        closed = min(abs(fill), abs(q))
        self.realized += closed * (ev.price - self.avg) * (1.0 if q > 0.0 else -1.0)
        new_q = q + fill
        if abs(new_q) <= _QTY_EPS * max(1.0, abs(q)):
            self.qty, self.avg = 0.0, 0.0
        elif (new_q > 0.0) != (q > 0.0):  # flipped through flat
            self.qty, self.avg = new_q, ev.price
        else:
            self.qty = new_q

    def pnl_at(self, price: float) -> float:
        return self.realized + self.qty * (price - self.avg)


def _locate_on_path(points: list[float], start_seg: int, price: float) -> int:
    """First path segment (>= start_seg) whose price range contains ``price``.

    Falls back to the segment nearest to ``price`` (a fill slipped a tick
    beyond the bar extreme, or a latency fill at the previous close).
    """
    best_seg, best_dist = start_seg, math.inf
    for seg in range(start_seg, len(points) - 1):
        lo = min(points[seg], points[seg + 1])
        hi = max(points[seg], points[seg + 1])
        dist = lo - price if price < lo else (price - hi if price > hi else 0.0)
        if dist <= 1e-9 * max(1.0, abs(price)):
            return seg
        if dist < best_dist:
            best_seg, best_dist = seg, dist
    return best_seg


def pnl_path(
    bars: BarSeries,
    events: list[LedgerEvent],
    *,
    adaptive: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar cumulative PnL of one instrument, at close and at its intrabar worst.

    The intrabar path is the one NT's bar execution simulates:
    O -> (H, L) -> C, high first unless ``adaptive`` and the low is nearer the
    open (``bar_adaptive_high_low_ordering``). The position is marked at the
    adverse extreme only while it is actually held: a stop exit stops the
    marking at the stop price, a market fill at the close keeps the whole
    bar, and so on.

    Args:
        bars: Execution bars (the finest timeframe the venue processed).
        events: This instrument's events in occurrence order.
        adaptive: Must match the venue's ``bar_adaptive_high_low_ordering``.

    Returns:
        ``(pnl_close, pnl_min)`` arrays aligned with ``bars``; PnL is
        realized cash + unrealized, relative to the starting balance.
    """
    n = len(bars)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    ts, o, h, lo, c = bars.ts, bars.open, bars.high, bars.low, bars.close
    ev_ts = np.fromiter((e.ts for e in events), dtype=np.int64, count=len(events))
    ev_bar = np.minimum(np.searchsorted(ts, ev_ts, side="left"), n - 1)

    book = _Book()
    event_bars: list[int] = []
    post_r: list[float] = []
    post_q: list[float] = []
    post_avg: list[float] = []
    walk_close: list[float] = []
    walk_min: list[float] = []

    i = 0
    n_events = len(events)
    while i < n_events:
        b = int(ev_bar[i])
        j = i
        while j < n_events and ev_bar[j] == b:
            j += 1
        bar_ts = int(ts[b])
        start_evs = [e for e in events[i:j] if not e.resting and e.ts < bar_ts]
        resting_evs = [e for e in events[i:j] if e.resting]
        end_evs = [e for e in events[i:j] if not e.resting and e.ts >= bar_ts]
        ob, hb, lb, cb = float(o[b]), float(h[b]), float(lo[b]), float(c[b])
        high_first = (not adaptive) or abs(hb - ob) < abs(lb - ob)
        points = [ob, hb, lb, cb] if high_first else [ob, lb, hb, cb]

        worst = math.inf
        for e in start_evs:
            worst = min(worst, book.pnl_at(e.price if e.qty else ob))
            book.apply(e)
            worst = min(worst, book.pnl_at(e.price if e.qty else ob))
        worst = min(worst, book.pnl_at(ob))
        seg = 0
        for e in resting_evs:
            target = _locate_on_path(points, seg, e.price)
            for k in range(seg + 1, target + 1):
                worst = min(worst, book.pnl_at(points[k]))
            worst = min(worst, book.pnl_at(e.price))
            book.apply(e)
            worst = min(worst, book.pnl_at(e.price))
            seg = target
        for k in range(seg + 1, len(points)):
            worst = min(worst, book.pnl_at(points[k]))
        for e in end_evs:
            px = e.price if e.qty else cb
            worst = min(worst, book.pnl_at(px))
            book.apply(e)
            worst = min(worst, book.pnl_at(px))

        event_bars.append(b)
        post_r.append(book.realized)
        post_q.append(book.qty)
        post_avg.append(book.avg)
        walk_close.append(book.pnl_at(cb))
        walk_min.append(min(worst, book.pnl_at(cb)))
        i = j

    if not event_bars:
        return np.zeros(n), np.zeros(n)

    eb = np.asarray(event_bars, dtype=np.int64)
    # State during bar i = state after the last event bar strictly before i
    last_event = np.searchsorted(eb, np.arange(n), side="left") - 1
    has = last_event >= 0
    kk = np.where(has, last_event, 0)
    r_arr = np.where(has, np.asarray(post_r)[kk], 0.0)
    q_arr = np.where(has, np.asarray(post_q)[kk], 0.0)
    avg_arr = np.where(has, np.asarray(post_avg)[kk], 0.0)
    adverse = np.where(q_arr > 0.0, lo, np.where(q_arr < 0.0, h, c))
    pnl_close = r_arr + q_arr * (c - avg_arr)
    pnl_min = r_arr + q_arr * (adverse - avg_arr)
    pnl_close[eb] = walk_close
    pnl_min[eb] = walk_min
    return pnl_close, pnl_min


def mark_to_market_max_drawdown(
    starting_balance: float,
    paths: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> float:
    """Max peak-to-trough decline of mark-to-market equity, as a fraction.

    Peaks are taken at bar closes (and the starting balance); troughs at each
    bar's intrabar worst (adverse extreme while a position is held). Multiple
    instruments are combined on the union of their bar timestamps (forward
    filled); summing per-instrument intrabar worsts is conservative.

    Args:
        starting_balance: Equity before the first bar.
        paths: ``(ts, pnl_close, pnl_min)`` per instrument from :func:`pnl_path`.

    Returns:
        Drawdown in ``[0, 1]`` (1.0 = equity wiped out).
    """
    paths = [p for p in paths if len(p[0])]
    if not paths or starting_balance <= 0:
        return 0.0
    if len(paths) == 1:
        _, total_close, total_min = paths[0]
    else:
        grid = np.unique(np.concatenate([p[0] for p in paths]))
        total_close = np.zeros(len(grid))
        total_min = np.zeros(len(grid))
        for ts, pc, pm in paths:
            idx = np.searchsorted(ts, grid, side="right") - 1
            valid = idx >= 0
            ii = np.where(valid, idx, 0)
            close_vals = np.where(valid, pc[ii], 0.0)
            exact = valid & (ts[ii] == grid)
            total_close += close_vals
            total_min += np.where(exact, pm[ii], close_vals)
    equity_close = starting_balance + total_close
    equity_min = starting_balance + total_min
    peak_prev = np.empty(len(equity_close))
    peak_prev[0] = starting_balance
    if len(equity_close) > 1:
        peak_prev[1:] = np.maximum(
            np.maximum.accumulate(equity_close[:-1]), starting_balance
        )
    drawdown = (peak_prev - equity_min) / peak_prev
    worst = float(np.max(drawdown))
    return min(max(worst, 0.0), 1.0)


def closed_trade_drawdown(
    starting_balance: float, closed_pnls: Iterable[tuple[int, float]]
) -> float:
    """Fallback max drawdown from realized trade PnLs ``(ts_closed, net_pnl)``.

    Ignores open-trade (unrealized) losses — only used when the
    mark-to-market curve cannot be built, and callers log that.
    """
    if starting_balance <= 0:
        return 0.0
    equity = peak = starting_balance
    max_dd = 0.0
    for _, pnl in sorted(closed_pnls, key=lambda x: x[0]):
        equity += pnl
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    return min(max_dd, 1.0)


@dataclass
class PerformanceMetrics:
    """Core performance metrics common to all backtest result types.

    Both :class:`screening.pipeline.BacktestMetrics` and
    :class:`validation.runner.ValidationResult` extend this base so that
    dashboard, overfitting filters, and other consumers can work with a
    single set of canonical field names.

    Field units:
        total_return: Stored as a decimal fraction (0.15 = 15%, -0.05 = -5%).
        profit_factor: Trade PF from :func:`profit_factor` (capped at
            :data:`PROFIT_FACTOR_CAP`, 0.0 with no trades).
    """

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0  # Decimal fraction: 0.15 means 15%
    profit_factor: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    total_fees: float = 0.0
    total_funding: float = 0.0
    execution_time_seconds: float = 0.0
    skewness: float = 0.0  # Return distribution skewness (0 = symmetric)
    kurtosis: float = 3.0  # Return distribution kurtosis (3 = normal)
    trade_returns: tuple[float, ...] = ()  # Per-trade PnL returns (fraction)
    # Funding settlements charged at the fallback rate because the archive
    # had no rate for them (0 = funding fully from archived rates).
    funding_fallback_settlements: int = 0
