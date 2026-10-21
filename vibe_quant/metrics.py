"""Shared performance metrics used across screening and validation pipelines."""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

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


def trade_win_rate(net_pnls: Iterable[float]) -> float:
    """Share of trades whose net PnL is above zero.

    ``net_pnls`` are per-trade NET PnLs -- the same list :func:`profit_factor`
    consumes (after fees and, where modeled, funding/slippage); a trade still
    open at the end of the run counts at its mark. This is the single win-rate
    definition shared by screening and validation (bd vibe-quant-neuvu.4;
    user ruling Q1=A: win rate is net of every cost).

    A trade netting exactly ``0.0`` is not a win. A non-finite PnL counts in
    the denominator and is never a win -- unlike :func:`profit_factor`, which
    skips it: win rate fails closed on purpose.

    Returns:
        ``0.0`` with no trades, else ``wins / len(net_pnls)``.
    """
    total = 0
    wins = 0
    for pnl in net_pnls:
        total += 1
        if math.isfinite(pnl) and pnl > 0.0:
            wins += 1
    return wins / total if total else 0.0


def gross_pnl(
    net_pnl: float,
    fees: float,
    slippage: float = 0.0,
    funding: float = 0.0,
) -> float:
    """Gross PnL in USD: net PnL plus every cost that was paid for it.

    ``net_pnl`` is already net of fees, slippage and funding, so adding the
    costs back recovers the PnL the signal would have made cost-free. Funding
    is signed (positive = paid, negative = received) -- receiving funding
    lowers gross PnL. Mirrors ``frontend/src/lib/metrics.ts`` ``grossPnlUsd``
    (bd vibe-quant-2gyk1.4).
    """
    return net_pnl + fees + slippage + funding


def fee_drag(fees: float, gross: float) -> float | None:
    """Fees as a fraction of gross PnL: how much of the edge the fees ate.

    ``None`` when ``gross <= 0`` or not finite: a strategy that made no money
    (or lost it) has no meaningful "share lost to fees", and dividing by a
    non-positive gross would report a nonsensical ratio (never raises).
    """
    if not math.isfinite(gross) or gross <= 0.0:
        return None
    return fees / gross


def fees_pct_equity_per_year(
    fees: float, starting_balance: float, days: float
) -> float | None:
    """Fees paid in a run, annualized as a fraction of equity.

    ``fees / starting_balance * 365 / days``. The assumption, stated because
    the ledger has no per-day equity curve here, is that equity equals the
    starting balance (no compounding). ``None`` when ``days <= 0``,
    ``starting_balance <= 0``, or either is non-finite -- a run with no
    duration has no annual rate (never raises).
    """
    if not math.isfinite(days) or not math.isfinite(starting_balance):
        return None
    if days <= 0.0 or starting_balance <= 0.0:
        return None
    return fees / starting_balance * 365.0 / days


_DAY_NS = 86_400 * 1_000_000_000


def _daily_cash(
    cash_events: Iterable[tuple[int, float]], start_ns: int, end_ns: int
) -> tuple[int, list[float]]:
    """``(first_day, per_day)``: balance deltas bucketed by UTC day over the window."""
    first_day = start_ns // _DAY_NS
    last_day = max(first_day, (end_ns - 1) // _DAY_NS)
    n_days = last_day - first_day + 1
    per_day = [0.0] * n_days
    for ts, delta in cash_events:
        idx = min(max(ts // _DAY_NS - first_day, 0), n_days - 1)
        per_day[idx] += delta
    return first_day, per_day


@dataclass(frozen=True, slots=True)
class DailyReturns:
    """Dense daily return series: one value per consecutive UTC day.

    ``values[i]`` is the return of day ``first_day + i`` (day index =
    ``ts_ns // 86400e9``). Stored as ``array('d')`` (8 bytes/day, pickles as
    raw bytes) because discovery ships it across the worker pool and keeps
    one per gate-passing genome (bd vibe-quant-yul7u.24).
    """

    first_day: int
    values: array[float]

    def __len__(self) -> int:
        return len(self.values)

    @property
    def end_day(self) -> int:
        """Day index one past the last value."""
        return self.first_day + len(self.values)

    def without_first_day(self) -> DailyReturns:
        """Drop day ``first_day`` (the return vs the starting balance)."""
        return DailyReturns(self.first_day + 1, self.values[1:])


def dense_daily_returns(
    starting_balance: float,
    cash_events: Iterable[tuple[int, float]],
    start_ns: int,
    end_ns: int,
) -> DailyReturns | None:
    """Every day's realized-balance return, INCLUDING the first day.

    Same bucketing and arithmetic as :func:`daily_balance_returns`, so
    ``dense_daily_returns(...).without_first_day()`` holds bit-identical values
    to the dict NT's Sharpe is computed from -- plus day 0's return against
    ``starting_balance`` (needed to concatenate sub-windows without losing the
    boundary day). Returns ``None`` (fail closed) when the balance is ever
    non-positive or a return is non-finite: such a series has no meaningful
    Sharpe, and :func:`daily_balance_returns` would silently drop days.
    """
    if not starting_balance > 0:
        return None
    first_day, per_day = _daily_cash(cash_events, start_ns, end_ns)
    out = array("d")
    prev = starting_balance
    balance = starting_balance
    for i, delta in enumerate(per_day):
        balance = starting_balance + delta if i == 0 else balance + delta
        if not balance > 0:
            return None
        ret = balance / prev - 1.0
        if not math.isfinite(ret):
            return None
        out.append(ret)
        prev = balance
    return DailyReturns(first_day, out)


def concat_daily_returns(parts: Sequence[DailyReturns | None]) -> DailyReturns | None:
    """Chain consecutive sub-window series; ``None`` unless they tile exactly.

    Each part must start on the day after the previous part ends (no gap, no
    duplicated boundary day); any missing part also gives ``None``.
    """
    if not parts or any(p is None for p in parts):
        return None
    chain = cast("list[DailyReturns]", list(parts))
    out = array("d", chain[0].values)
    for prev, nxt in zip(chain, chain[1:], strict=False):
        if nxt.first_day != prev.end_day:
            return None
        out.extend(nxt.values)
    return DailyReturns(chain[0].first_day, out)


def pool_daily_returns(parts: Sequence[DailyReturns | None]) -> DailyReturns | None:
    """Equal-weight mean of per-symbol series, aligned by DATE KEY.

    Fails closed (``None``) on a missing/empty part, a part covering different
    days (never truncated to a common length), or a non-finite value.
    """
    if not parts or any(p is None or len(p) == 0 for p in parts):
        return None
    chain = cast("list[DailyReturns]", list(parts))
    first, n = chain[0].first_day, len(chain[0])
    if any(p.first_day != first or len(p) != n for p in chain):
        return None
    pooled = np.mean(np.vstack([np.frombuffer(p.values, dtype=np.float64) for p in chain]), axis=0)
    if not np.all(np.isfinite(pooled)):
        return None
    return DailyReturns(first, array("d", pooled.tobytes()))


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
    first_day, per_day = _daily_cash(cash_events, start_ns, end_ns)
    n_days = len(per_day)
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
        liquidation: True only on the synthetic close a liquidation adds
            (``liquidation.overlay``): ``price`` is the liquidation close price
            (the bankruptcy-equivalent price when the insurance fund absorbed a
            deficit) and ``-qty`` the liquidated quantity. Changes how
            ``pnl_path`` values that quantity in its execution bar
            (:func:`_liquidation_walk`), never the realized PnL.
        liq_margin: Part of ``cash`` that is margin of a liquidated quantity
            (funding in isolated, all cash in cross), set by the overlay only on
            real events stamped at a liquidation's ``ts``. Valuation only.
        Both default to the plain behaviour, which is unchanged.
    """

    ts: int
    qty: float = 0.0
    price: float = 0.0
    cash: float = 0.0
    resting: bool = False
    liquidation: bool = False
    liq_margin: float = 0.0


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


class LedgerBook:
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


def _liquidation_walk(
    book: LedgerBook, evs: Sequence[LedgerEvent], bar_ts: int, points: list[float]
) -> float:
    """:func:`pnl_path`'s walk of the execution bar that holds a liquidation close.

    Every real event keeps its normal place (before the bar, on the bar path, after
    its close) and its normal valuation. The *instant* is the events stamped at the
    close's ``ts``; the close is replayed right after the last of them (in its own
    normal place when there are none). From the instant on (its first event, or the
    bar open when the close's ts precedes the bar's ``ts_init``) until the close, the
    liquidated quantity ``L = -close.qty`` alone is valued at no worse than its cap::

        cap_s = P + M_s / L

    ``P`` is the close's price (bankruptcy-equivalent when the insurance fund absorbed
    a deficit), ``M_s`` the ``liq_margin`` of the instant's events not yet applied in
    state ``s``: so ``cap_s`` is the bankruptcy price at that state's margin (or the
    venue's liquidation price). A long ``L`` is worth ``max(p, cap_s)``, a short one
    ``min(p, cap_s)``. The rest of the position, and every point before the instant,
    follow the bar path; equity has no floor.

    Worked examples, 1 BTC long @ 40000, 10x, balance 40000:

    * The bar opens at 35000 (past bankruptcy) and funding +100 arrives at the hit:
      margin 4100, ``P`` = 35900, final 36000. Before the funding ``cap`` = 35900 +
      100 = 36000 (equity 36000), after it 35900 (40100 - 4100 = 36000): ``L`` is
      never marked at the gap's 35000, which the insurance fund absorbs.
    * Equity 50000 at 50000; the hit minute opens at 45000, a LIMIT sells 0.9 at 50000
      and funding -200 is paid at the bar's ``ts_init``, leaving 0.1 with LP 38152.61
      (no deficit). The open precedes the instant, so 1 BTC at 45000 (equity 45000) is
      a genuine minimum. After the sale the low 37000 is above ``cap`` = 38152.61 - 2000
      (the funding is still to come), so 0.1 is marked at 37000 (48700); after the
      funding it is held at 38152.61. Final 48600, drawdown 0.1.

    Raises ``ValueError`` when the bar cannot be replayed exactly: more than one close,
    an event after the close's ts, a resting fill before it, or a position left open.
    """
    closes = [e for e in evs if e.liquidation]
    if len(closes) != 1:
        raise ValueError(f"execution bar {bar_ts} holds {len(closes)} liquidation closes")
    close = closes[0]
    t = close.ts
    real = [e for e in evs if e is not close]
    if any(e.ts > t or (e.resting and e.ts < t) for e in real):
        raise ValueError(
            f"execution bar {bar_ts}: an event after the liquidation at {t}, or a resting "
            "fill before it, cannot be replayed exactly"
        )
    ob, cb = points[0], points[-1]
    # pnl_path's replay order: (event, valuation price) or (None, bar path point)
    steps: list[tuple[LedgerEvent | None, float]] = [
        (e, e.price if e.qty else ob) for e in real if not e.resting and e.ts < bar_ts
    ]
    open_at = len(steps)
    steps.append((None, ob))
    seg = 0
    for e in real:
        if e.resting:
            target = _locate_on_path(points, seg, e.price)
            steps.extend((None, points[k]) for k in range(seg + 1, target + 1))
            steps.append((e, e.price))
            seg = target
    steps.extend((None, points[k]) for k in range(seg + 1, len(points)))
    steps.extend((e, e.price if e.qty else cb) for e in real if not e.resting and e.ts >= bar_ts)

    instant = [k for k, (e, _px) in enumerate(steps) if e is not None and e.ts == t]
    # the close follows the instant's last real event, else takes its own normal place
    at = instant[-1] + 1 if instant else (open_at if t < bar_ts else len(steps))
    steps.insert(at, (close, close.price))
    first = instant[0] if instant else at
    activate = min(first, open_at) if t < bar_ts else first

    liq = -close.qty
    pending = {id(e): e.liq_margin for e in real if e.ts == t}

    def value(price: float, active: bool) -> float:
        pnl = book.pnl_at(price)
        q = book.qty
        if not active or q == 0.0 or (q > 0.0) != (liq > 0.0):
            return pnl
        cap = close.price + math.fsum(pending.values()) / liq
        held = min(abs(liq), abs(q))
        if liq > 0.0 and price < cap:
            return pnl + held * (cap - price)
        if liq < 0.0 and price > cap:
            return pnl + held * (price - cap)
        return pnl

    worst = math.inf
    for k, (ev, px) in enumerate(steps):
        active = k >= activate
        worst = min(worst, value(px, active))
        if ev is not None:
            book.apply(ev)
            pending.pop(id(ev), None)
            worst = min(worst, value(px, active))
    if book.qty != 0.0:
        raise ValueError(f"execution bar {bar_ts}: position still open after its liquidation")
    return worst


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

    book = LedgerBook()
    event_bars: list[int] = []
    post_r: list[float] = []
    post_q: list[float] = []
    post_avg: list[float] = []
    walk_close: list[float] = []
    walk_min: list[float] = []
    liq_bars = {int(ev_bar[k]) for k, e in enumerate(events) if e.liquidation}

    i = 0
    n_events = len(events)
    while i < n_events:
        b = int(ev_bar[i])
        j = i
        while j < n_events and ev_bar[j] == b:
            j += 1
        bar_ts = int(ts[b])
        ob, hb, lb, cb = float(o[b]), float(h[b]), float(lo[b]), float(c[b])
        high_first = (not adaptive) or abs(hb - ob) < abs(lb - ob)
        points = [ob, hb, lb, cb] if high_first else [ob, lb, hb, cb]

        if b in liq_bars:
            worst = _liquidation_walk(book, events[i:j], bar_ts, points)
        else:
            start_evs = [e for e in events[i:j] if not e.resting and e.ts < bar_ts]
            resting_evs = [e for e in events[i:j] if e.resting]
            end_evs = [e for e in events[i:j] if not e.resting and e.ts >= bar_ts]

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


def mark_to_market_equity(
    starting_balance: float,
    paths: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    floor_from_ns: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Account equity ``(ts, equity_close, equity_min)`` from per-instrument paths.

    Instruments are combined on the union of their bar timestamps (forward
    filled); summing per-instrument intrabar worsts is conservative.
    ``floor_from_ns`` (a cross liquidation's ``bust_ts``): from the bar holding
    it on, the account cannot be below 0 (the insurance fund absorbs the rest),
    so ``equity_min`` is clamped at 0 on bars with ``ts >= floor_from_ns``. The
    conservative sum can otherwise pair one leg's dip before an intrabar bust
    with another leg's post-bust value. ``None`` = no clamp.
    """
    paths = [p for p in paths if len(p[0])]
    if not paths:
        return np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0)
    if len(paths) == 1:
        grid, total_close, total_min = paths[0]
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
    if floor_from_ns is not None:
        equity_min = np.where(grid >= floor_from_ns, np.maximum(equity_min, 0.0), equity_min)
    return grid, equity_close, equity_min


def mark_to_market_max_drawdown(
    starting_balance: float,
    paths: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    floor_from_ns: int | None = None,
) -> float:
    """Max peak-to-trough decline of mark-to-market equity, as a fraction.

    Peaks are taken at bar closes (and the starting balance); troughs at each
    bar's intrabar worst (adverse extreme while a position is held), on the
    account equity of :func:`mark_to_market_equity` (``floor_from_ns`` there).

    Args:
        starting_balance: Equity before the first bar.
        paths: ``(ts, pnl_close, pnl_min)`` per instrument from :func:`pnl_path`.

    Returns:
        Drawdown in ``[0, 1]`` (1.0 = equity wiped out).
    """
    paths = [p for p in paths if len(p[0])]
    if not paths or starting_balance <= 0:
        return 0.0
    _grid, equity_close, equity_min = mark_to_market_equity(
        starting_balance, paths, floor_from_ns=floor_from_ns
    )
    peak_prev = np.empty(len(equity_close))
    peak_prev[0] = starting_balance
    if len(equity_close) > 1:
        peak_prev[1:] = np.maximum(
            np.maximum.accumulate(equity_close[:-1]), starting_balance
        )
    drawdown = (peak_prev - equity_min) / peak_prev
    worst = float(np.max(drawdown))
    return min(max(worst, 0.0), 1.0)


def mark_to_market_daily_returns(
    starting_balance: float,
    paths: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    start_ns: int,
    end_ns: int,
) -> DailyReturns | None:
    """Daily account returns with open positions marked at each day's close.

    Same UTC day grid as :func:`dense_daily_returns` (``start_ns // DAY`` to
    ``(end_ns - 1) // DAY``, INCLUDING day 0), so the two series are directly
    comparable. Per instrument the ``pnl_close`` of the last execution bar with
    ``ts // DAY <= d`` is carried forward (0 before its first bar) and summed
    across instruments; ``equity_d = starting_balance + total_d`` and
    ``r_d = equity_d / equity_{d-1} - 1`` with ``equity_{-1} = starting_balance``.

    Unlike the realized series, an overnight position contributes its move on
    the day it happens instead of on its exit day. ``None`` (fail closed) when
    the balance is ever non-positive or a return is non-finite, matching
    :func:`dense_daily_returns`.
    """
    if not starting_balance > 0:
        return None
    first_day = start_ns // _DAY_NS
    last_day = max(first_day, (end_ns - 1) // _DAY_NS)
    n_days = last_day - first_day + 1
    days = np.arange(first_day, last_day + 1, dtype=np.int64)
    total = np.zeros(n_days)
    for ts, pnl_close, _pnl_min in paths:
        if len(ts) == 0:
            continue
        # Last bar at or before each day (bars are sorted by ts).
        idx = np.searchsorted(ts // _DAY_NS, days, side="right") - 1
        total += np.where(idx >= 0, pnl_close[np.maximum(idx, 0)], 0.0)
    out = array("d")
    prev = starting_balance
    for i in range(n_days):
        balance = starting_balance + float(total[i])
        if not balance > 0:
            return None
        ret = balance / prev - 1.0
        if not math.isfinite(ret):
            return None
        out.append(ret)
        prev = balance
    return DailyReturns(first_day, out)


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
    # Net PnL plus fees/slippage/funding (see :func:`gross_pnl`); screening sets
    # it so discovery can report fee drag (bd vibe-quant-2gyk1.4).
    gross_pnl: float = 0.0
    execution_time_seconds: float = 0.0
    skewness: float = 0.0  # Return distribution skewness (0 = symmetric)
    kurtosis: float = 3.0  # Return distribution kurtosis (3 = normal)
    trade_returns: tuple[float, ...] = ()  # Per-trade PnL returns (fraction)
    # Funding settlements charged at the fallback rate because the archive
    # had no rate for them (0 = funding fully from archived rates).
    funding_fallback_settlements: int = 0
