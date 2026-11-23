"""Random-timing null: does a strategy's own trade schedule beat itself re-timed at random?

Two kernels share this module:

* The research report kernel (``trade_multiplier``, ``place_trades``, ``null_draws``,
  ``null_percentile``, ``alpha_beta``), moved verbatim from
  ``scripts/research/tv_ports/oos_analysis.py``, which imports them (its report bytes are
  unchanged). It compounds all-in 1x trades.
* The discovery gate kernel (bead vibe-quant-clmmc.14, design arch-gates-2 §2.2, R-g-4):
  ``sim_daily`` replays a trade schedule AS TRADED -- each trade keeps its entry equity
  fraction, the account is marked at every UTC day close exactly like the engine's
  ``metrics.mark_to_market_daily_returns``, and ruin absorbs. ``timing_null_check`` keeps
  every (hold, side, frac) triple, places them at random, chains windows like
  ``NTBacktestFn._chain_window_series``, aggregates symbols with the run's own function and
  ranks the real as-traded compound return among the draws. ``timing_null_decision`` is the
  one verdict rule.

Everything fails closed: bad bars, missing daily closes, an invalid trade or an over-full
window raise :class:`TimingNullError` whose text is the rejection reason; an unreadable record
rejects.
"""

from __future__ import annotations

import hashlib
import math
from array import array
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

from vibe_quant.data.fill_ticks import _BAR_INIT_OFFSET_NS
from vibe_quant.metrics import DailyReturns

if TYPE_CHECKING:
    from vibe_quant.validation.funding import FundingCalculator

DAY_NS = 86_400 * 10**9
#: Null draws per check, pass threshold (share of draws strictly below real) and the
#: exposure above which re-timing is degenerate (buy-and-hold decides instead).
NULL_DRAWS = 500
NULL_PCT = 0.95
EXPOSURE_MAX = 0.9
#: Where the engine stamps a bar's market fill: bar open + 59.999 s (``fill_ticks``). Funding
#: runs between these instants, not between bar closes.
FILL_DELAY_NS = _BAR_INIT_OFFSET_NS

# --------------------------------------------------------------------------- report kernel
# (moved verbatim from scripts/research/tv_ports/oos_analysis.py)


def alpha_beta(y: Sequence[float], x: Sequence[float]) -> tuple[float, float]:
    """OLS ``y = a + b x`` on daily returns: ``(a x 365, b)``; ``(nan, nan)`` if degenerate."""
    ya, xa = np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)
    if len(ya) < 3 or float(np.var(xa)) == 0.0:
        return float("nan"), float("nan")
    b = float(np.cov(xa, ya, ddof=1)[0, 1] / np.var(xa, ddof=1))
    a = float(ya.mean() - b * xa.mean())
    return a * 365.0, b


def trade_multiplier(side: int, p_in: float, p_out: float, fee: float) -> float:
    """Equity multiplier of one all-in 1x trade: ``1 + s (R - 1) - f (1 + R)``, R = out / in.

    The position is the whole equity at entry; the fee is ``fee`` of the entry notional plus
    ``fee`` of the exit notional.
    """
    ratio = p_out / p_in
    return 1.0 + side * (ratio - 1.0) - fee * (1.0 + ratio)


def place_trades(n_bars: int, holds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniformly random non-overlapping starts for trades of ``holds`` bars, in the given order.

    Trades keep their order; the ``k + 1`` gaps (before, between, after) are a uniform random
    composition of the slack ``n_bars - sum(holds)``: sorted draws of ``k`` distinct values from
    ``range(slack + k)`` minus their rank are the cumulative gaps (stars and bars).
    """
    k = len(holds)
    slack = n_bars - int(holds.sum())
    if slack < 0:
        raise ValueError(f"holds need {int(holds.sum())} bars, segment has {n_bars}")
    if k == 0:
        return np.zeros(0, dtype=np.int64)
    picks = np.sort(rng.choice(slack + k, size=k, replace=False))
    gaps = picks - np.arange(k)
    before = np.concatenate(([0], np.cumsum(holds)[:-1]))
    return (gaps + before).astype(np.int64)


def null_draws(
    prices: np.ndarray,
    holds: np.ndarray,
    sides: np.ndarray,
    fee: float,
    draws: int,
    rng: np.random.Generator,
    cell_of_bar: np.ndarray | None = None,
    n_cells: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Random-timing null of one segment.

    ``prices[i]`` is the entry price of bar ``i`` (its open) and ``prices[n]`` the price after
    the last bar, so a trade entering bar ``i`` for ``h`` bars exits at ``prices[i + h]``. Per
    draw the (hold, side) pairs are shuffled together, placed with :func:`place_trades` and
    compounded with :func:`trade_multiplier`. Returns ``(total return per draw, return per draw
    and cell)``; a trade belongs to the cell of its entry bar.
    """
    n_bars = len(prices) - 1
    totals = np.zeros(draws)
    cells = np.zeros((draws, n_cells))
    for j in range(draws):
        order = rng.permutation(len(holds))
        h, s = holds[order], sides[order]
        starts = place_trades(n_bars, h, rng)
        ratio = prices[starts + h] / prices[starts]
        mult = 1.0 + s * (ratio - 1.0) - fee * (1.0 + ratio)
        totals[j] = float(np.prod(mult)) - 1.0
        if cell_of_bar is not None and n_cells:
            for c in range(n_cells):
                cells[j, c] = float(np.prod(mult[cell_of_bar[starts] == c])) - 1.0
    return totals, cells


def null_percentile(real: float, null: np.ndarray) -> float:
    """Share of null values strictly below ``real``."""
    return float(np.mean(null < real)) if len(null) else float("nan")


# --------------------------------------------------------------------------- gate kernel types


class TimingNullError(ValueError):
    """Fail-closed input problem; ``str(err)`` is the rejection reason."""


class TimingNullPolicyError(ValueError):
    """``notes.timing_null_gate`` is present but malformed."""


class NullGateLaunchError(ValueError):
    """A ``--null-gate`` launch the gate cannot judge (operator error, CLI exit 2)."""


@dataclass(frozen=True, slots=True)
class TradeSpan:
    """One trade of the real schedule, in bars of its window's tape.

    ``frac`` = entry notional / account equity at entry (the symbol's own account, flat at
    entry: ``B0 + net PnL of its earlier trades in the window``). The trade fills at the open
    of ``entry_bar`` and exits at the open of ``entry_bar + hold_bars``; one reaching the tape's
    end is still open at the run end (``carried``) and is valued like the engine's run-end mark
    (``mark_open_positions``): the last close minus the estimated taker exit fee on that value,
    no exit fill, no exit slippage, no funding after the last bar (MEASURED on real NT,
    chief R-gb2-1). Known limit: a null copy of a carried trade placed before the end exits at
    a bar open (with any exit slippage) instead of the last close.
    """

    side: int
    frac: float
    entry_bar: int
    hold_bars: int
    carried: bool = False


@dataclass(frozen=True, slots=True, eq=False)
class WindowTape:
    """Execution bars of one eval window ``[start_ns, end_ns)`` of one symbol.

    ``open_ns`` are bar open times (every ``tf_ns``, checked); ``ts`` are the bars' engine
    timestamps (``BarSeries.ts``, the kline close - 1 ms) that place day-close marks, fills
    and cash events exactly as ``metrics.pnl_path`` / ``mark_to_market_daily_returns`` do.

    ``tick`` (the instrument's ``price_increment``) and the unit funding table (``funding_ts``,
    ``funding_unit``: the archived settlements in ``(start_ns, ts[-1]]`` charged to a LONG of
    notional 1, positive = paid) are the real side's cost inputs; the span capture fills them
    and :func:`capture_costs` fails closed without them.
    """

    symbol: str
    timeframe: str
    tf_ns: int
    start_ns: int
    end_ns: int
    open_ns: np.ndarray
    ts: np.ndarray
    open: np.ndarray
    close: np.ndarray
    tick: float | None = None
    funding_ts: np.ndarray | None = None
    funding_unit: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.open)


#: ``(symbol, side, entry_notional, entry_ns, exit_ns) -> [(settlement_ns, amount)]``,
#: positive amount = paid by the trader (``FundingModel.accrue(...).payments``).
FundingHook = Callable[[str, int, float, int, int], Sequence[tuple[int, float]]]
#: ``(symbol, fill side: +1 buy / -1 sell, reference price) -> filled price`` (tick slippage).
FillHook = Callable[[str, int, float], float]
Aggregate = Callable[[Sequence[DailyReturns | None]], DailyReturns | None]


@dataclass(frozen=True, slots=True)
class TradeCosts:
    """The production cost model hooks: taker ``fee`` per leg on the filled notional,
    optional fill-price slippage and signed funding. ``TradeCosts(fee)`` is fee-only."""

    fee: float
    fill_price: FillHook | None = None
    funding: FundingHook | None = None


@dataclass(frozen=True, slots=True)
class TimingNullRecord:
    """Persisted per champion / rejection / capped row (``as_dict``; keys per design §2.2).

    ``fallback`` is ``None`` when the null ran; on degenerate exposure it is the buy-and-hold
    verdict text (``"passed"`` or its reasons) and ``fallback_passed`` decides.
    """

    percentile: float | None
    draws: int
    seeds: dict[str, int]
    exposure: float
    exposure_by_symbol: dict[str, float]
    carried: int
    zero_hold_clipped: int
    real_T: float
    null_median: float | None
    null_p95: float | None
    fallback: str | None
    fallback_passed: bool | None
    alpha_ann: float | None
    beta: float | None
    alpha_note: str | None
    fee: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TimingNullPolicy:
    """``notes.timing_null_gate`` of a run (the verdict's inputs)."""

    enforced: bool
    draws: int
    pct: float
    exposure_max: float
    base_seed: int | None


# --------------------------------------------------------------------------- as-traded sim


def total_return(values: Sequence[float] | np.ndarray | array[float]) -> float:
    """``T(x) = prod(1 + x) - 1`` -- the one compounding of real and null series."""
    return float(np.prod(1.0 + np.asarray(values, dtype=np.float64))) - 1.0


def single_series(parts: Sequence[DailyReturns | None]) -> DailyReturns | None:
    """Aggregate of a one-symbol run: its own series."""
    return parts[0] if len(parts) == 1 else None


def null_seed(base_seed: int, dsl_key: str, symbol: str) -> int:
    """Per (run, genome, symbol) null seed, independent of pool size and evaluation order."""
    digest = hashlib.sha256(f"{base_seed}|{dsl_key}|{symbol}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def funding_window(tape: WindowTape, i: int, xb: int, at_close: bool) -> tuple[int, int]:
    """``(entry_ns, exit_ns)`` a hold entering bar ``i`` is funded between: the engine's fill
    instants (bar open + :data:`FILL_DELAY_NS`), a hold reaching the tape end up to its last
    ``ts`` (the run-end mark). Settlements in ``(entry_ns, exit_ns]`` are charged."""
    entry = int(tape.open_ns[i]) + FILL_DELAY_NS
    exit_ = int(tape.ts[xb]) if at_close else int(tape.open_ns[xb]) + FILL_DELAY_NS
    return entry, exit_


def sim_daily(
    tape: WindowTape, spans: Sequence[TradeSpan], costs: TradeCosts, b0: float
) -> DailyReturns:
    """Daily returns of one window's own account trading ``spans`` as traded.

    Same day grid as the engine (``start_ns // DAY .. (end_ns - 1) // DAY``, INCLUDING day 0)
    and the same arithmetic as ``LedgerBook`` + ``mark_to_market_daily_returns``: quantity
    ``frac x equity / fill price``, fees and funding as realized cash at their timestamps,
    open positions marked at each day's last bar close. Ruin is checked at every exit and
    every day-close mark: equity <= 0 books one -100% day, then 0.0 for the rest of the
    window (no further trades); the next window starts again at ``b0``. Funding is charged
    between the fill instants (:func:`funding_window`). A span reaching
    the tape end stays open: marked at the last close, which also books the mark's estimated
    taker exit fee ``fee x q x last close`` (the engine's modeled run-end cash; no exit
    slippage).
    """
    n = len(tape)
    ts, opens, closes = tape.ts, tape.open, tape.close
    pnl = np.zeros(n)  # realized + unrealized PnL at each bar close, vs b0
    realized = 0.0
    nxt = busy = 0
    for sp in sorted(spans, key=lambda s: s.entry_bar):
        i, s = sp.entry_bar, sp.side
        if i < busy or sp.hold_bars < 1 or i >= n:
            raise ValueError(f"span at bar {i} (hold {sp.hold_bars}) overlaps or leaves the tape")
        busy = i + sp.hold_bars
        pnl[nxt:i] = realized
        nxt = i
        equity = b0 + realized
        if not equity > 0:
            break
        at_close = i + sp.hold_bars >= n
        xb = n - 1 if at_close else i + sp.hold_bars  # bar whose close is flat again
        p_in, p_out = float(opens[i]), float(closes[n - 1] if at_close else opens[xb])
        if costs.fill_price is not None:
            p_in = costs.fill_price(tape.symbol, s, p_in)
            if not at_close:
                p_out = costs.fill_price(tape.symbol, -s, p_out)
        q = sp.frac * equity / p_in
        realized += -(costs.fee * (q * p_in))
        avg = (0.0 + q * p_in) / q  # LedgerBook.apply's average from flat
        held = np.full(xb - i, realized)
        if costs.funding is not None:
            pays = costs.funding(tape.symbol, s, q * p_in, *funding_window(tape, i, xb, at_close))
            pays = sorted(pays, key=lambda p: p[0])
            for pay_ts, amount in pays:
                bar = max(int(np.searchsorted(ts, pay_ts, side="left")), i)
                realized += -amount
                if bar < xb:
                    held[bar - i :] = realized
        pnl[i:xb] = held + (s * q) * (closes[i:xb] - avg)
        if at_close:  # open at the run end: the last close marks it, less the mark's exit fee
            realized += -(costs.fee * (q * p_out))
            pnl[xb] = realized + (s * q) * (p_out - avg)
            nxt = n
            continue
        realized += -(costs.fee * (q * p_out))
        realized += q * (p_out - avg) * (1.0 if s > 0 else -1.0)
        nxt = xb
    pnl[nxt:] = realized

    first_day = tape.start_ns // DAY_NS
    last_day = max(first_day, (tape.end_ns - 1) // DAY_NS)
    days = np.arange(first_day, last_day + 1, dtype=np.int64)
    idx = np.searchsorted(ts // DAY_NS, days, side="right") - 1
    total = np.where(idx >= 0, pnl[np.maximum(idx, 0)], 0.0)
    out = array("d")
    prev = b0
    ruined = False
    for k in range(len(days)):
        balance = b0 + float(total[k])
        if ruined:
            out.append(0.0)
        elif not balance > 0:
            out.append(-1.0)
            ruined = True
        else:
            out.append(balance / prev - 1.0)
            prev = balance
    return DailyReturns(int(first_day), out)


# --------------------------------------------------------------------------- capture


#: ``notes.timing_null_gate`` texts (design arch-gates-2 §2.2; R-gb1-1 carried-trade limit).
NULL_COSTS_NOTE = (
    "the real side's costs at the re-timed holds: taker fee per leg, expected screening tick "
    "slippage (0.5 x one tick per market fill) and signed archived funding per held "
    "settlement from the fill instants"
)
NULL_GATE_LIMITS: tuple[str, ...] = (
    "a trade still open at the window end is valued like the engine's run-end mark: the "
    "last close minus the estimated taker exit fee (no exit slippage, no funding after the "
    "last bar); a null copy of it placed earlier exits at a bar open",
    "each symbol and eval window is its own account starting flat at the starting balance; "
    "a portfolio of 2+ symbols is refused (a shared account couples symbol sizing)",
    "holds are counted in strategy-timeframe bars from the entry fill's bar to the exit "
    "fill's bar: a same-bar trade holds 1 bar (zero_hold_clipped), a hold crossing the "
    "window end is clipped to it (boundary_clipped)",
    "null fills are bar opens with the fill model's expected slippage; the engine fills at "
    "the boundary 1m close with a seeded 0/1-tick draw",
)

_MS = 10**6


def window_tape(
    symbol: str, timeframe: str, tf_ns: int, bars: object, start_ns: int, end_ns: int
) -> WindowTape:
    """The bars of ``bars`` (a ``metrics.BarSeries``, ``ts`` = kline close - 1 ms) opening in
    ``[start_ns, end_ns)`` as a :class:`WindowTape` (copies; gaps are left for
    :func:`validate_tape` to reject)."""
    from vibe_quant.metrics import BarSeries

    assert isinstance(bars, BarSeries)
    w = bars.window(start_ns, end_ns - 1)
    ts = np.array(w.ts, dtype=np.int64)
    return WindowTape(
        symbol=symbol,
        timeframe=timeframe,
        tf_ns=tf_ns,
        start_ns=start_ns,
        end_ns=end_ns,
        open_ns=ts + _MS - tf_ns,
        ts=ts,
        open=np.array(w.open, dtype=np.float64),
        close=np.array(w.close, dtype=np.float64),
    )


def funding_table(
    calc: FundingCalculator, instrument_id: str, start_ns: int, last_ts_ns: int
) -> tuple[np.ndarray, np.ndarray]:
    """The unit funding table of one window: ``FundingCalculator.accrue`` of a LONG of notional
    1 over ``(start_ns, last_ts_ns]`` as ``(settlement ns, signed amount)`` arrays. Archive
    settlements and the fallback grid of holes come out exactly as for any real hold (the
    calculator is the one that charged the real side)."""
    payments = calc.accrue(instrument_id, "LONG", 1.0, start_ns, last_ts_ns).payments
    return (
        np.array([t for t, _ in payments], dtype=np.int64),
        np.array([a for _, a in payments], dtype=np.float64),
    )


def capture_costs(
    fee: float, tapes: Mapping[str, Sequence[WindowTape]], prob_slip: float
) -> TradeCosts:
    """The real side's costs for a null hold, read from the capture's own tapes.

    * ``fill_price``: ``px + side x prob_slip x tick`` of the fill's symbol (the screening fill
      model's mean slip: a buy fills above, a sell below; no random draws).
    * ``funding``: the window's unit table sliced to ``(entry, exit]``, paying ``side x unit x
      notional`` per settlement -- exactly ``FundingCalculator.accrue`` on the same span.
      The window is found by the hold's ``entry_ns`` (windows are disjoint).

    Raises:
        TimingNullError: a tape has no (positive, finite) price increment or no funding table.
    """
    tick: dict[str, float] = {}
    windows: dict[str, tuple[list[int], list[int], list[list[int]], list[list[float]]]] = {}
    for sym, symbol_tapes in tapes.items():
        for tape in symbol_tapes:
            validate_tape(tape)
        starts: list[int] = []
        ends: list[int] = []
        times: list[list[int]] = []
        units: list[list[float]] = []
        for w, tape in enumerate(symbol_tapes, 1):
            if tape.tick is None or not (math.isfinite(tape.tick) and tape.tick > 0.0):
                raise TimingNullError(f"Timing null: no price increment for {sym}")
            if tape.funding_ts is None or tape.funding_unit is None:
                raise TimingNullError(f"Timing null: no funding table for {sym} window {w}")
            tick[sym] = float(tape.tick)
            starts.append(tape.start_ns)
            ends.append(tape.end_ns)
            times.append(tape.funding_ts.tolist())
            units.append(tape.funding_unit.tolist())
        windows[sym] = (starts, ends, times, units)

    def fill_price(symbol: str, side: int, px: float) -> float:
        return px + side * prob_slip * tick[symbol]

    def funding(
        symbol: str, side: int, notional: float, entry_ns: int, exit_ns: int
    ) -> list[tuple[int, float]]:
        starts, ends, times, units = windows[symbol]
        w = bisect_right(starts, entry_ns) - 1
        if w < 0 or entry_ns >= ends[w]:
            raise TimingNullError(f"Timing null: no funding window for {symbol} at {entry_ns}")
        ts = times[w]
        lo, hi = bisect_right(ts, entry_ns), bisect_right(ts, exit_ns)
        return [(ts[k], side * units[w][k] * notional) for k in range(lo, hi)]

    return TradeCosts(fee, fill_price, funding)


def spans_from_ledger(
    costed: Sequence[object], tape: WindowTape, b0: float
) -> tuple[list[TradeSpan], int]:
    """One window's trade schedule from the screening runner's costed ledger (mode off).

    ``costed`` are ``trade_ledger.PositionCosts`` AFTER the run-end marks: a marked item is
    ``carried`` (held to the tape end). ``frac`` = entry notional (``|peak_qty| x
    avg_px_open``, the filled price) / equity at entry, where equity = ``b0`` + net PnL
    (realized incl. commissions, less slippage and funding) of every EARLIER item (event
    order: ``ts_opened``, then input order) that exited at or before the entry ts. So an old
    position closed on the reopen's ts counts (netting: the close fills first), while the
    trade itself (an entry and exit on one ts) and any later-ordered trade never feed its
    own size (no look-ahead). Bars: the tape bar whose open
    is the last one at or before the fill ts. A closed trade exiting at or past
    ``tape.end_ns`` (or entering before the start) is clipped to the half-open window and
    counted in the returned ``boundary_clipped``; a same-bar trade keeps hold 0 (the null
    kernel counts and rounds it). Returned in entry order.
    """
    from vibe_quant.trade_ledger import position_direction

    def net(item: object) -> float:
        accrual = getattr(item, "accrual", None)
        funding = float(accrual.total) if accrual is not None else 0.0
        return float(item.realized_pnl) - float(item.slippage_cost) - funding  # type: ignore[attr-defined]

    n = len(tape)
    clipped = 0
    spans: list[TradeSpan] = []
    ordered = sorted(costed, key=lambda c: int(c.pos.ts_opened))  # type: ignore[attr-defined]
    exits = [(int(c.exit_ns), net(c)) for c in ordered]  # type: ignore[attr-defined]
    for i, item in enumerate(ordered):
        pos = item.pos  # type: ignore[attr-defined]
        entry_ns = int(pos.ts_opened)
        equity = b0 + sum(v for ts, v in exits[:i] if ts <= entry_ns)  # earlier items only
        notional = abs(float(pos.peak_qty)) * float(pos.avg_px_open)
        entry = int(np.searchsorted(tape.open_ns, entry_ns, side="right")) - 1
        if entry < 0:
            entry, clipped = 0, clipped + 1
        carried = item.mark is not None  # type: ignore[attr-defined]
        exit_ns = int(item.exit_ns)  # type: ignore[attr-defined]
        if carried:
            exit_bar = n
        elif exit_ns >= tape.end_ns:
            exit_bar, clipped = n, clipped + 1
        else:
            exit_bar = int(np.searchsorted(tape.open_ns, exit_ns, side="right")) - 1
        side = 1 if position_direction(pos) == "LONG" else -1
        spans.append(TradeSpan(side, notional / equity, entry, exit_bar - entry, carried))
    return spans, clipped


def null_gate_launch_problems(
    *,
    symbols: Sequence[str],
    symbol_agg: str,
    liquidation_on: bool,
    grid: bool,
    nt_engine: bool,
    timeframe: str,
) -> list[str]:
    """Why a ``--null-gate`` launch is refused (empty = it may run). Shared by the CLI run and
    the launch API (B-T3), checked before any pipeline exists. ``timeframe`` must be one the
    span capture can load a bar tape for (``NTScreeningRunner._capture_trade_spans``)."""
    from vibe_quant.data.catalog import INTERVAL_TO_AGGREGATION
    from vibe_quant.validation.extraction import _TIMEFRAME_SECONDS

    problems: list[str] = []
    tapes = [tf for tf in _TIMEFRAME_SECONDS if tf in INTERVAL_TO_AGGREGATION]
    if timeframe not in tapes:
        problems.append(
            f"--null-gate needs a timeframe with a bar tape ({'|'.join(tapes)}), got {timeframe}"
        )
    if len(symbols) > 1 and symbol_agg == "portfolio":
        problems.append(
            "--null-gate needs one symbol or --symbol-agg worst "
            "(a shared account couples symbol sizing)"
        )
    if symbol_agg == "pooled":
        problems.append("--null-gate cannot judge --symbol-agg pooled (no pooled timing null yet)")
    if grid:
        problems.append("--null-gate cannot run a bankroll grid (its cells run liquidation on)")
    elif liquidation_on:
        problems.append(
            "--null-gate needs --liquidation-mode off (liquidated trades have no re-placeable exit)"
        )
    if not nt_engine:
        problems.append("--null-gate needs real NT backtests (no --mock, no --evaluator fast)")
    return problems


# --------------------------------------------------------------------------- the check


def _utc(ns: int) -> str:
    return datetime.fromtimestamp(ns // 10**9, tz=UTC).strftime("%Y-%m-%dT%H:%MZ")


def validate_tape(tape: WindowTape) -> None:
    """Every bar of ``[start_ns, end_ns)`` present, on its grid, with positive finite prices."""
    pre = f"Timing null: no {tape.timeframe} bar {tape.symbol} at window"
    on = tape.open_ns
    if len(on) == 0 or int(on[0]) != tape.start_ns:
        raise TimingNullError(f"{pre} start {_utc(tape.start_ns)}")
    steps = np.diff(on)
    if len(steps) and np.any(steps != tape.tf_ns):
        k = int(np.argmax(steps != tape.tf_ns))
        raise TimingNullError(
            f"Timing null: bar gap {tape.symbol} {tape.timeframe} after {_utc(int(on[k]))}"
        )
    if int(on[-1]) != tape.end_ns - tape.tf_ns:
        raise TimingNullError(f"{pre} end {_utc(tape.end_ns - tape.tf_ns)}")
    with np.errstate(invalid="ignore"):
        bad = ~(
            np.isfinite(tape.open)
            & (tape.open > 0.0)
            & np.isfinite(tape.close)
            & (tape.close > 0.0)
        )
    if np.any(bad):
        k = int(np.argmax(bad))
        o = float(tape.open[k])
        px = o if not (math.isfinite(o) and o > 0.0) else float(tape.close[k])
        raise TimingNullError(f"Timing null: invalid price {tape.symbol} {_utc(int(on[k]))}: {px}")


def _chain(parts: list[DailyReturns]) -> DailyReturns | None:
    from vibe_quant.discovery.backtest_fn import NTBacktestFn

    return NTBacktestFn._chain_window_series([{"mtm_daily_returns": p} for p in parts])  # type: ignore[dict-item]


def _alpha(
    real: DailyReturns, closes: Mapping[str, Mapping[int, float]], symbols: list[str]
) -> tuple[float | None, float | None, str | None]:
    from vibe_quant.discovery.buyhold import buy_hold_daily

    x, err = buy_hold_daily(closes, symbols, real.first_day, len(real))
    if x is None:
        raise TimingNullError(f"Timing null: buy-and-hold record unavailable: {err}")
    a, b = alpha_beta(real.values, x.tolist())
    if not (math.isfinite(a) and math.isfinite(b)):
        return None, None, "degenerate regression"
    return a, b, None


def timing_null_check(
    real: DailyReturns,
    tapes: Mapping[str, Sequence[WindowTape]],
    spans: Mapping[str, Sequence[Sequence[TradeSpan]]],
    *,
    costs: TradeCosts,
    b0: float,
    seeds: Mapping[str, int],
    aggregate: Aggregate,
    closes: Mapping[str, Mapping[int, float]],
    buy_hold: tuple[bool, Sequence[str]] | None = None,
    draws: int = NULL_DRAWS,
    exposure_max: float = EXPOSURE_MAX,
) -> TimingNullRecord:
    """Rank ``real``'s compound return among ``draws`` random re-timings of its own trades.

    ``real`` is the gate's own as-traded MTM series (windows chained, symbols aggregated);
    ``tapes`` / ``spans`` hold each symbol's windows in date order and symbol order is the
    aggregation order. Per symbol one ``default_rng(seeds[sym])`` walks draws, then windows:
    the window's (hold, side, frac) triples are permuted together, placed with
    :func:`place_trades` and replayed by :func:`sim_daily`; windows chain with the engine's
    first-day trim; ``aggregate`` (the run's own: :func:`single_series` or
    ``pool_daily_returns``) joins symbols per draw. Exposure (held bars / window bars, all
    symbols, after clipping zero holds to 1) above ``exposure_max`` skips the draws: the
    buy-and-hold verdict ``buy_hold`` = ``(passed, reasons)`` decides (``None`` fails).

    Raises:
        TimingNullError: no symbols, a missing seed, bad bars, a missing daily close, a trade
            whose side is not +-1 or whose ``frac`` is not finite and > 0, holds that cannot
            fit a window or a null series off the real series' day grid.
    """
    symbols = list(tapes)
    if not symbols:
        raise TimingNullError("Timing null: no symbols")
    for sym in symbols:
        if len(spans.get(sym, ())) != len(tapes[sym]):
            raise TimingNullError(f"Timing null: trade spans missing for {sym}")
        if sym not in seeds:
            raise TimingNullError(f"Timing null: no null seed for {sym}")
        for tape in tapes[sym]:
            validate_tape(tape)
    alpha_ann, beta, alpha_note = _alpha(real, closes, symbols)

    clipped = 0
    triples: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    held_by: dict[str, int] = {}
    bars_by: dict[str, int] = {}
    for sym in symbols:
        triples[sym] = []
        held_by[sym] = bars_by[sym] = 0
        for w, (tape, window_spans) in enumerate(zip(tapes[sym], spans[sym], strict=True), 1):
            for sp in window_spans:  # a NaN size would ruin every draw: never rank on it
                if sp.side not in (-1, 1):
                    bad = f"side {sp.side}"
                elif not (math.isfinite(sp.frac) and sp.frac > 0.0):
                    bad = f"frac {sp.frac}"
                else:
                    continue
                raise TimingNullError(
                    f"Timing null: invalid trade {sym} window {w} bar {sp.entry_bar}: {bad}"
                )
            holds = np.array([sp.hold_bars for sp in window_spans], dtype=np.int64)
            if np.any(holds < 0):
                raise TimingNullError(f"Timing null: negative hold {sym} window {w}")
            clipped += int(np.sum(holds == 0))
            holds = np.maximum(holds, 1)
            if int(holds.sum()) > len(tape):
                raise TimingNullError(
                    f"Timing null: holds need {int(holds.sum())} bars, {sym} window {w} "
                    f"has {len(tape)}"
                )
            sides = np.array([sp.side for sp in window_spans], dtype=np.int64)
            fracs = np.array([sp.frac for sp in window_spans], dtype=np.float64)
            triples[sym].append((holds, sides, fracs))
            held_by[sym] += int(holds.sum())
            bars_by[sym] += len(tape)
    exposure = sum(held_by.values()) / sum(bars_by.values())
    by_symbol = {s: held_by[s] / bars_by[s] for s in symbols}
    carried = sum(sp.carried for sym in symbols for w in spans[sym] for sp in w)
    real_t = total_return(real.values)

    def record(
        pct: float | None,
        n: int,
        med: float | None,
        p95: float | None,
        fallback: str | None,
        fallback_passed: bool | None,
    ) -> TimingNullRecord:
        return TimingNullRecord(
            percentile=pct,
            draws=n,
            seeds={s: int(seeds[s]) for s in symbols},
            exposure=exposure,
            exposure_by_symbol=by_symbol,
            carried=int(carried),
            zero_hold_clipped=clipped,
            real_T=real_t,
            null_median=med,
            null_p95=p95,
            fallback=fallback,
            fallback_passed=fallback_passed,
            alpha_ann=alpha_ann,
            beta=beta,
            alpha_note=alpha_note,
            fee=float(costs.fee),
        )

    if exposure > exposure_max:
        passed = buy_hold is not None and buy_hold[0] is True
        text = "passed" if passed else "; ".join(buy_hold[1]) if buy_hold else ""
        return record(None, 0, None, None, text or "Buy-hold: record unavailable", passed)

    per_symbol: dict[str, list[DailyReturns | None]] = {}
    for sym in symbols:
        rng = np.random.default_rng(seeds[sym])
        series: list[DailyReturns | None] = []
        for _ in range(draws):
            parts = []
            for tape, (holds, sides, fracs) in zip(tapes[sym], triples[sym], strict=True):
                order = rng.permutation(len(holds))
                h, s, f = holds[order], sides[order], fracs[order]
                starts = place_trades(len(tape), h, rng)
                placed = [
                    TradeSpan(int(s[k]), float(f[k]), int(starts[k]), int(h[k]))
                    for k in range(len(h))
                ]
                parts.append(sim_daily(tape, placed, costs, b0))
            series.append(_chain(parts))
        per_symbol[sym] = series
    null = np.empty(draws)
    for j in range(draws):
        agg = aggregate([per_symbol[s][j] for s in symbols])
        if agg is None or agg.first_day != real.first_day or len(agg) != len(real):
            raise TimingNullError("Timing null: null series off the real series' day grid")
        null[j] = total_return(agg.values)
    return record(
        null_percentile(real_t, null),
        draws,
        float(np.median(null)),
        float(np.quantile(null, NULL_PCT)),
        None,
        None,
    )


# --------------------------------------------------------------------------- verdict


def fallback_line(record: Mapping[str, object], exposure_max: float) -> str:
    """The degenerate-exposure line: ``... buy-and-hold decides (<bh reasons | passed>)``."""
    exposure = record.get("exposure")
    shown = f"{exposure:.4f}" if isinstance(exposure, float) else repr(exposure)
    return (
        f"Timing null: exposure {shown} > {exposure_max:.2f}, null degenerate; "
        f"buy-and-hold decides ({record.get('fallback')})"
    )


def timing_null_decision(policy: TimingNullPolicy | None, record: object) -> tuple[bool, list[str]]:
    """The one verdict rule (gate and both promote paths): ``(passed, reasons)``.

    No policy / not enforced -> pass. A missing or unreadable record rejects; a fallback
    record passes iff ``fallback_passed is True``; otherwise ``percentile >= policy.pct``.
    ``reasons`` is non-empty exactly when the verdict is a rejection.
    """
    if policy is None or not policy.enforced:
        return True, []
    if record is None:
        return False, ["Timing null: record unreadable (missing)"]
    if not isinstance(record, Mapping):
        return False, [f"Timing null: record unreadable ({type(record).__name__})"]
    if record.get("fallback") is not None:
        if record.get("fallback_passed") is True:
            return True, []
        return False, [fallback_line(record, policy.exposure_max)]
    pct = record.get("percentile")
    if (
        isinstance(pct, bool)
        or not isinstance(pct, int | float)
        or not math.isfinite(pct)
        or not 0.0 <= pct <= 1.0
    ):
        return False, [f"Timing null: record unreadable (percentile {pct!r})"]
    if pct >= policy.pct:
        return True, []
    return False, [f"Timing null: percentile {pct:.4f} < {policy.pct:.2f}"]


def parse_timing_null_policy(notes: Mapping[str, object] | None) -> TimingNullPolicy | None:
    """Policy from a run's notes; absent key -> ``None``; malformed -> TimingNullPolicyError."""
    if not notes or "timing_null_gate" not in notes:
        return None
    gate = notes["timing_null_gate"]
    if not isinstance(gate, Mapping):
        raise TimingNullPolicyError(f"timing_null_gate is not an object: {gate!r}")
    enforced, pct = gate.get("enforced"), gate.get("pct", NULL_PCT)
    draws, cap = gate.get("draws", NULL_DRAWS), gate.get("exposure_max", EXPOSURE_MAX)
    seed = gate.get("base_seed")
    if not isinstance(enforced, bool):
        raise TimingNullPolicyError(f"timing_null_gate.enforced not a bool: {enforced!r}")
    for name, value in (("pct", pct), ("exposure_max", cap)):
        if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 < value <= 1.0:
            raise TimingNullPolicyError(f"timing_null_gate.{name} out of (0, 1]: {value!r}")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise TimingNullPolicyError(f"timing_null_gate.draws not a positive int: {draws!r}")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TimingNullPolicyError(f"timing_null_gate.base_seed not an int: {seed!r}")
    return TimingNullPolicy(enforced, draws, float(pct), float(cap), seed)
