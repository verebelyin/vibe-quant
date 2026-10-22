"""L1 synthetic trade/return generator with a known true Sharpe.

Job 20261017-power-paper, task T1a (design-power.md §3 "L1 statistical layer").
The generator emits the exact ``bt`` dict a real ``NTBacktestFn`` run would, so the
REAL gate chain (fitness, guardrails, DSR) can be measured on a strategy whose edge
is known by construction.

Units: S is the annualized Sharpe of the dense calendar-day MTM equity series
(``mean/std * sqrt(365)``); the repo's own ``sqrt(252)`` Sharpe equals ``S * 0.8309``.
``mu`` is solved at the REALIZED exposure of every emitted series (train chained
and holdout alike), so each one's population daily Sharpe equals ``S/sqrt(365)``
by construction.

The trades are laid out so the series the gates actually see carries exactly
``n * hold`` active days: the first calendar day of every eval window stays flat
(``_chain_window_series`` drops the first window's day-0 return) and no trade
straddles a window boundary (W-R2b).

Not on the mypy baseline (``files = ["vibe_quant"]``); kept typed and ruff-clean.
"""

from __future__ import annotations

import math
import random
from array import array
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.discovery.fitness import (
    MIN_TRADES,
    FitnessResult,
    _const_backtest,
    _evaluate_single,
)
from vibe_quant.discovery.genome import generate_random_chromosome
from vibe_quant.metrics import DailyReturns, daily_balance_returns, profit_factor
from vibe_quant.trade_ledger import moments_from_returns
from vibe_quant.utils import compute_day_count
from vibe_quant.validation.extraction import sharpe_sortino_from_returns

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from vibe_quant.discovery.operators import StrategyChromosome

#: Nanoseconds per UTC day; the bucketing unit of ``daily_balance_returns``.
DAY_NS: int = 86_400 * 1_000_000_000
#: Sigma of the in-position daily equity return (design §3).
DEFAULT_SIGMA: float = 0.012
#: In-position day count target as a fraction of the chained series length.
HOLD_FRACTION: float = 0.4
#: Student-t degrees of freedom for the in-position return (design §3).
T_DOF: int = 5
#: Scale making a t5 draw unit-variance: ``sqrt(3/5)``.
T5_SCALE: float = math.sqrt(3.0 / T_DOF)
#: Account balance the emitted returns/MTM series start from.
STARTING_BALANCE: float = 1000.0


@dataclass(frozen=True, slots=True)
class Plan:
    """One evaluation window plan (design §6 R3).

    ``chained_days`` is the length of the MTM series the DSR gate sees: the
    per-window series concatenated by ``NTBacktestFn._chain_window_series``, which
    drops the first window's day-0 return (W3: 438 + 438 - 1 = 875).
    """

    name: str
    train_start: str
    train_end: str
    chained_days: int
    eval_windows: tuple[tuple[str, str], ...]
    holdout: tuple[str, str]


PLANS: dict[str, Plan] = {
    "W3": Plan(
        name="W3",
        train_start="2023-06-01",
        train_end="2025-10-24",
        chained_days=875,
        eval_windows=(("2023-06-01", "2024-08-12"), ("2024-08-12", "2025-10-24")),
        holdout=("2025-10-24", "2026-06-01"),
    ),
    "W6": Plan(
        name="W6",
        train_start="2021-01-01",
        train_end="2025-08-18",
        chained_days=1689,
        eval_windows=(("2021-01-01", "2023-04-26"), ("2023-04-26", "2025-08-18")),
        holdout=("2025-08-18", "2026-10-15"),
    ),
}


def day_index(date: str) -> int:
    """Absolute UTC day index (``ts // DAY_NS``) of an ISO date."""
    return int(np.datetime64(date, "D").astype("int64"))


def hold_for(n: int, T: int) -> int:
    """Holding days per trade: ``max(1, round(0.4 * T / n))`` (design §6 R5)."""
    return max(1, round(HOLD_FRACTION * T / n))


def realized_exposure(n: int, hold: int, T: int) -> float:
    """Fraction of the chained series spent in position: ``n * hold / T`` (R5)."""
    return n * hold / T


def zero_inflated_daily_sharpe(
    f_real: float, mu: float, sigma: float = DEFAULT_SIGMA
) -> float:
    """Population daily mean/std of the zero-inflated series (design §6 R5b).

    Each day is in position with probability ``f_real`` (in-position return ``mu``
    mean, ``sigma`` std) and flat otherwise: ``mean = f*mu``,
    ``var = f*sigma^2 + f*mu^2*(1-f)``.
    """
    mean = f_real * mu
    var = f_real * sigma * sigma + f_real * mu * mu * (1.0 - f_real)
    return mean / math.sqrt(var)


def mu_for_sharpe(S: float, f_real: float, sigma: float = DEFAULT_SIGMA) -> float:
    """In-position daily mean for a realized-exposure series daily Sharpe of ``S/sqrt(365)``.

    Inverting :func:`zero_inflated_daily_sharpe`: ``mu = s*sigma / sqrt(f - s^2*(1-f))``
    with ``s = S/sqrt(365)`` (design §6 R5b). ``S == 0`` maps to ``0.0`` exactly.
    """
    if S == 0.0:
        return 0.0
    s = S / math.sqrt(365.0)
    return s * sigma / math.sqrt(f_real - s * s * (1.0 - f_real))


@dataclass(frozen=True, slots=True)
class Trade:
    """One synthetic trade realized inside one eval window."""

    entry_day: int
    exit_day: int
    returns: tuple[float, ...]
    pnl: float
    return_frac: float


@dataclass(frozen=True, slots=True)
class Schedule:
    """The train-span trade schedule and its per-day in-position returns."""

    plan: Plan
    n: int
    S: float
    seed: int
    hold: int
    f_real: float
    mu: float
    first_day: int
    n_days: int
    entries: tuple[int, ...]
    returns: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class WindowGen:
    """One eval window: the bt dict, its trades and its realized cash events."""

    window: tuple[str, str]
    bt: dict[str, Any]
    trades: tuple[Trade, ...]
    cash_events: tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class Generated:
    """A whole plan's synthetic train run: schedule, per-window gens, aggregate bt."""

    schedule: Schedule
    windows: tuple[WindowGen, ...]
    bt: dict[str, Any]


@dataclass(frozen=True, slots=True)
class HoldoutGen:
    """The holdout window's generator parameters and its rendered window.

    ``n`` is the HOLDOUT trade count (derived from the train ``n``), and ``mu``
    is solved at the holdout's own realized exposure, so the emitted series has
    the same known true Sharpe by construction (W-R2b).
    """

    plan: Plan
    n: int
    hold: int
    f_real: float
    mu: float
    seed: int
    window: WindowGen


def slot_bounds(window: tuple[str, str]) -> tuple[int, int]:
    """``[start, end)``: the absolute days a window may host trades on (W-R2b).

    The first calendar day of every window stays flat -- ``_chain_window_series``
    drops the first window's day-0 return, so a trade active there would vanish
    from the chained series the DSR gate reads (round-1 F1: 299 of 300 days).
    ``end`` is the window's last day (``we`` is exclusive).
    """
    ws, we = window
    return day_index(ws) + 1, day_index(we)


def place_entries(
    windows: Sequence[tuple[str, str]], n: int, hold: int
) -> tuple[int, ...]:
    """Absolute entry days of ``n`` whole, non-overlapping ``hold``-day trades.

    The trades are split across the windows in proportion to their schedulable
    length and spread evenly inside each (:func:`slot_bounds`), so no trade
    straddles a window boundary and none sits on a day chaining drops: the
    chained series then carries exactly ``n * hold`` active days.
    """
    bounds = [slot_bounds(window) for window in windows]
    lengths = [end - start for start, end in bounds]
    total = sum(lengths)
    counts: list[int] = []
    left = n
    for i, length in enumerate(lengths):
        count = left if i == len(lengths) - 1 else round(n * length / total)
        counts.append(count)
        left -= count
    entries: list[int] = []
    for (start, _), length, count in zip(bounds, lengths, counts, strict=True):
        if count <= 0 or length < count * hold:
            raise ValueError(f"{n} trades of {hold} days do not fit windows {windows}")
        last = count - 1
        span = length - hold  # an entry at ``span`` ends on the window's last day
        offsets = [0] if last == 0 else [(k * span) // last for k in range(count)]
        entries.extend(start + offset for offset in offsets)
    return tuple(entries)


def holdout_trades(n: int, T: int, holdout_days: int) -> int:
    """Holdout trade count at the train rate: ``max(1, round(n * H / T))`` (W-R2b)."""
    return max(1, round(n * holdout_days / T))


def build_schedule(plan: Plan | str, n: int, S: float, seed: int) -> Schedule:
    """Lay ``n`` non-overlapping trades over the train calendar days and draw them.

    Each trade occupies ``hold`` consecutive days whose returns are
    ``mu + sigma*t5*sqrt(3/5)`` (design §3); flat days stay exactly ``0.0``. The
    entries come from :func:`place_entries`, so every trade sits wholly inside
    one eval window and the chained series carries exactly ``n * hold`` active
    days (W-R2b).
    """
    plan = PLANS[plan] if isinstance(plan, str) else plan
    T = plan.chained_days
    hold = hold_for(n, T)
    f_real = realized_exposure(n, hold, T)
    mu = mu_for_sharpe(S, f_real)
    first_day = day_index(plan.train_start)
    n_days = compute_day_count(plan.train_start, plan.train_end)
    assert n_days is not None
    entries = place_entries(plan.eval_windows, n, hold)
    rng = np.random.default_rng(seed)
    draws = rng.standard_t(T_DOF, size=n * hold)
    x = mu + DEFAULT_SIGMA * T5_SCALE * draws
    returns = np.zeros(n_days, dtype=np.float64)
    k = 0
    for entry in entries:
        offset = entry - first_day
        assert offset >= 0 and offset + hold <= n_days
        for d in range(offset, offset + hold):
            returns[d] = x[k]
            k += 1
    return Schedule(
        plan=plan,
        n=n,
        S=S,
        seed=seed,
        hold=hold,
        f_real=f_real,
        mu=mu,
        first_day=first_day,
        n_days=n_days,
        entries=entries,
        returns=returns,
    )


def window_spans(
    entries: Sequence[int], hold: int, ws_day: int, we_day: int
) -> list[tuple[int, int]]:
    """``(entry_day, exit_day)`` for trades entering in ``[ws_day, we_day)`` (R7).

    A trade entering on ``ws_day`` is included; one entering on ``we_day`` is not.
    A position still open at ``we_day - 1`` is closed there (its exit is ``we_day-1``).
    """
    return [(e, min(e + hold - 1, we_day - 1)) for e in entries if ws_day <= e < we_day]


def window_returns(sched: Schedule, ws_day: int, we_day: int) -> NDArray[np.float64]:
    """Per-day in-position returns over ``[ws_day, we_day)`` (day ``ws_day`` first)."""
    return grid_window_returns(
        sched.returns, sched.first_day, sched.entries, sched.hold, ws_day, we_day
    )


def grid_window_returns(
    returns: NDArray[np.float64],
    first_day: int,
    entries: Sequence[int],
    hold: int,
    ws_day: int,
    we_day: int,
) -> NDArray[np.float64]:
    """Per-day returns a day grid contributes to one window (R7).

    A trade entering on ``ws_day`` is included, one entering on ``we_day`` is not,
    and a position still open at ``we_day - 1`` is closed there.
    """
    out = np.zeros(we_day - ws_day, dtype=np.float64)
    for entry, exit_day in window_spans(entries, hold, ws_day, we_day):
        for d in range(entry, exit_day + 1):
            out[d - ws_day] = returns[d - first_day]
    return out


def window_daily_returns(sched: Schedule, ws: str, we: str) -> DailyReturns:
    """The window's dense MTM series (including day ``ws``), as NT emits it."""
    ws_day = day_index(ws)
    values = window_returns(sched, ws_day, day_index(we))
    return DailyReturns(ws_day, array("d", values.tolist()))


def make_window(sched: Schedule, ws: str, we: str) -> WindowGen:
    """Build one window's bt dict, trades and realized cash events (R8/R9)."""
    return grid_window(
        sched.returns, sched.first_day, sched.entries, sched.hold, ws, we
    )


def grid_window(
    returns: NDArray[np.float64],
    first_day: int,
    entries: Sequence[int],
    hold: int,
    ws: str,
    we: str,
) -> WindowGen:
    """Build one window's bt dict, trades and realized cash events (R8/R9).

    The MTM series compounds to ``total_return``; each trade's PnL is the balance
    delta over its in-window days, realized at ``exit_day`` (mid-day, R9).
    """
    ws_day = day_index(ws)
    we_day = day_index(we)
    values = grid_window_returns(returns, first_day, entries, hold, ws_day, we_day)
    spans = window_spans(entries, hold, ws_day, we_day)

    balance = STARTING_BALANCE
    trades: list[Trade] = []
    cash_events: list[tuple[int, float]] = []
    for entry, exit_day in spans:
        before = balance
        for d in range(entry, exit_day + 1):
            balance *= 1.0 + values[d - ws_day]
        pnl = balance - before
        trades.append(
            Trade(
                entry_day=entry,
                exit_day=exit_day,
                returns=tuple(float(values[d - ws_day]) for d in range(entry, exit_day + 1)),
                pnl=pnl,
                return_frac=pnl / before if before else 0.0,
            )
        )
        cash_events.append((exit_day * DAY_NS + DAY_NS // 2, pnl))

    total_return = balance / STARTING_BALANCE - 1.0
    pnls = [t.pnl for t in trades]
    trade_returns = tuple(t.return_frac for t in trades)
    sharpe, sortino = sharpe_sortino_from_returns(
        daily_balance_returns(
            STARTING_BALANCE, cash_events, ws_day * DAY_NS, we_day * DAY_NS
        )
    )
    equity = np.cumprod(1.0 + values) * STARTING_BALANCE
    peaks = np.maximum.accumulate(
        np.concatenate([np.array([STARTING_BALANCE]), equity])
    )[:-1]
    max_dd = float(np.max((peaks - equity) / peaks))
    skewness, kurtosis, _ = moments_from_returns(list(trade_returns))
    bt: dict[str, Any] = {
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "profit_factor": profit_factor(pnls),
        "total_trades": len(trades),
        "total_return": total_return,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "trade_returns": trade_returns,
        "mtm_daily_returns": DailyReturns(ws_day, array("d", values.tolist())),
        "starting_balance": STARTING_BALANCE,
        "days": we_day - ws_day,
    }
    return WindowGen(
        window=(ws, we), bt=bt, trades=tuple(trades), cash_events=tuple(cash_events)
    )


def train_mtm_values(sched: Schedule) -> NDArray[np.float64]:
    """The chained train MTM series the DSR gate sees, via the real chaining fn."""
    results: list[dict[str, Any]] = [
        {"mtm_daily_returns": window_daily_returns(sched, ws, we)}
        for ws, we in sched.plan.eval_windows
    ]
    chained = NTBacktestFn._chain_window_series(
        cast("list[dict[str, float | int]]", results), key="mtm_daily_returns"
    )
    assert chained is not None
    return np.frombuffer(chained.values, dtype=np.float64)


def aggregate_windows(windows: Sequence[WindowGen]) -> dict[str, Any]:
    """Worst-of-N aggregate over the eval windows + the chained MTM series (R8/R11)."""
    results: list[dict[str, Any]] = [w.bt for w in windows]
    agg: dict[str, Any] = dict(
        NTBacktestFn._aggregate_multi_window(
            cast("list[dict[str, float | int]]", results), min_trades=0
        )
    )
    agg["mtm_daily_returns"] = NTBacktestFn._chain_window_series(
        cast("list[dict[str, float | int]]", results), key="mtm_daily_returns"
    )
    return agg


def generate(plan: Plan | str, n: int, S: float, seed: int) -> Generated:
    """Build the whole synthetic train run for one plan/cell."""
    resolved = PLANS[plan] if isinstance(plan, str) else plan
    sched = build_schedule(resolved, n, S, seed)
    windows = tuple(make_window(sched, ws, we) for ws, we in resolved.eval_windows)
    return Generated(schedule=sched, windows=windows, bt=aggregate_windows(windows))


def holdout(plan: Plan | str, n: int, S: float, seed: int) -> HoldoutGen:
    """Build the holdout window at the train rate and hold (W-R2b, design §3).

    ``n`` is the TRAIN trade count: the holdout gets ``round(n * H / T)`` trades
    (``H`` holdout calendar days, ``T`` chained train days) with the train
    ``hold``. ``mu`` is solved at the holdout's own realized exposure, so the
    emitted series carries the same known true Sharpe ``S`` by construction --
    the same exactness the train series has. Round 1 read the train-only grid on
    the holdout span and emitted 0 trades across its 220 W3 days.
    """
    resolved = PLANS[plan] if isinstance(plan, str) else plan
    T = resolved.chained_days
    hold = hold_for(n, T)
    holdout_days = compute_day_count(*resolved.holdout)
    assert holdout_days is not None
    n_trades = holdout_trades(n, T, holdout_days)
    f_real = realized_exposure(n_trades, hold, holdout_days)
    mu = mu_for_sharpe(S, f_real)
    first_day = day_index(resolved.holdout[0])
    entries = place_entries((resolved.holdout,), n_trades, hold)
    rng = np.random.default_rng([seed, 1])
    draws = rng.standard_t(T_DOF, size=n_trades * hold)
    x = mu + DEFAULT_SIGMA * T5_SCALE * draws
    returns = np.zeros(holdout_days, dtype=np.float64)
    k = 0
    for entry in entries:
        offset = entry - first_day
        assert offset >= 0 and offset + hold <= holdout_days
        for d in range(offset, offset + hold):
            returns[d] = x[k]
            k += 1
    window = grid_window(
        returns, first_day, entries, hold, resolved.holdout[0], resolved.holdout[1]
    )
    return HoldoutGen(
        plan=resolved,
        n=n_trades,
        hold=hold,
        f_real=f_real,
        mu=mu,
        seed=seed,
        window=window,
    )


def evaluate(
    plan: Plan | str,
    n: int,
    S: float,
    seed: int,
    min_trades: int = MIN_TRADES,
    timeframe: str | None = None,
) -> FitnessResult:
    """Fitness of the synthetic train run through the real ``_evaluate_single``."""
    resolved = PLANS[plan] if isinstance(plan, str) else plan
    bt = generate(resolved, n, S, seed).bt
    chrom: StrategyChromosome = generate_random_chromosome(random.Random(seed))
    return _evaluate_single(chrom, _const_backtest(bt), None, min_trades, timeframe)
