"""G-null: seeded zero-edge false-positive check of the timing-null gate (vibe-quant-clmmc.14).

Job 20261122-edge, task B-T4 -- design ``data/swarm/20261122-edge/designs/arch-gates-2.md``
§5 G-null, design review ``handoffs/dreview-b3-2.md`` GATES-9/GATES-10, chief ruling R-gb4-1
(board 8396). Every false-positive rate claim uses seeded synthetic zero-edge controls
(R-c17-2).

What one replicate is
---------------------
1. **Shared tape first.** Per symbol a bar tape at the plan's strategy timeframe (W3 4h,
   W6 1d): log returns multivariate Student-t5 (unit-variance, one shared chi-square per
   bar), drift 0, 60 % annual vol, cross-symbol correlation 0.6, one continuous path over the
   plan's two eval windows; the holdout tape is a separately drawn 365 days from the train end.
   The tape stream never depends on the schedules (spawn key 0; holdout spawn key 1).
2. **Zero-edge schedules**, independent of the tape (spawn key ``(2, i)`` per candidate):
   per window geometric holds (mean 6 bars on W3, 2 on W6 -- R-gb4-1: mean 6 on 1d bars gives
   ~28 train trades < min_trades 50, a dead config) until the held bars would pass ``e x n``,
   sides long with p 0.6, ``frac ~ U(0.05, 0.5)``, placed by the production
   ``timing_null.place_trades``. ``6sync`` = one schedule on all six symbols, ``6async`` =
   independent schedules.
3. **Real side as traded**: the production ``timing_null.sim_daily`` per symbol and window,
   chained like ``NTBacktestFn._chain_window_series``, aggregated by the production function
   (one symbol: its own series; six: ``NTBacktestFn._aggregate_symbols`` = worst mode, pooled
   MTM). Two pre-registered real-side cost cases (GATES-9): ``fee`` = taker fee per leg;
   ``credit`` = taker fee per leg and EVERY trade, long or short, RECEIVES funding 0.0001 per
   8h settlement on its held notional, no slippage (R-gb4-3: the flat-short worst case, real
   side favoured). The null pays the SAME costs as the real side of its case
   (``real_costs(case)``) at its re-timed holds, as the production gate's null does (beads
   .28/.23): a flat short earning the credit ties its null instead of beating a fee-only one.
4. **Production chain via gatechain**: guardrails + DSR (q95 rule) -> [buy-and-hold] ->
   ``timing_null_check`` (500 draws, production ``null_seed``) -> holdout. ``replacement``
   = no buy-and-hold, null; ``combined`` = buy-and-hold, then null.

Outcomes and rules (pre-registered in docs/discovery-journal.md, R-gb4-1)
-------------------------------------------------------------------------
* **Selection null (binding)**: 200 reps x K = 167 candidates, every train-passer through the
  chain at DSR N = 167 (any-pick-of-all, ``layer1.pick_pool(..., "all")``); one line per chain
  x cost case under ``layer1.SELECTION_RULE`` (<= 3/200 PASS, >= 11/200 FAIL, 4..10 -> an
  800-rep top-up on tag 0x7152, PASS iff <= 27/800).
* **Single null (calibration)**: 400 reps, one candidate, the timing-null gate ALONE (percentile
  >= 0.95). A correct 95 % gate sits near 5 % by construction, so the rule is the exact
  one-sided binomial test of rate <= 0.05, Bonferroni over the 24 config x cost lines
  (R-gb4-2): FAIL iff count >= ``CALIBRATION_K`` (34/400). The single-candidate chains
  (DSR N = 1) are information only.

Each cost case has its own null per candidate and buy-and-hold verdict (``_NullCache``, key
(case, verdict)): the production ``timing_null_check`` runs once per case with that case's
costs, its aggregated draws are captured through the ``aggregate`` hook, and the case's real
series is ranked against its own draws only. Joint re-timing (trades of several coins that
overlap move together) and the book seed come from the production check, not a harness copy.

Seeds: ``SeedSequence([20261122, tag, cfg.index, rep])``, binding tag 0x7151, top-up 0x7152,
``--quick`` 0x71A0 (a disjoint stream: no quick run touches a binding replicate; the earlier
0x7141/0x7142 streams are retired, never reused). A run at any other sizes than 400 single /
200 selection reps / K = 167, over a ``--configs`` subset, or with a single ``--outcome`` is
forced onto 0x71A0 too (``run_stream``): only the full 12-config, both-outcome binding command
reads 0x7151. ``null_runs`` counts null COMPUTATIONS (one per case x buy-and-hold verdict), not
candidates: 0 means no pick reached the null. Null seeds: production
``null_seed(base, "cand{i}", book)`` (book = the symbols joined by ``+``) with ``base`` = the
first uint32 of the replicate root (``_rep_seed``).

Limits: the cost cases are synthetic (``credit`` is not an archived funding history, no
slippage on either side). A null draw that reaches the window end is marked at the last close
exactly as the engine's run-end mark: both fee legs (the mark's estimated exit fee included),
no exit slippage, funding to the last bar -- a carried draw therefore differs from a real
trade only by fewer settlements, not by one missing fee leg. Durations set by stop/take-profit
hits are not exercised.

Usage: ``cd <wt> && PYTHONPATH=$PWD .venv/bin/python scripts/research/power/timing_null_power.py
--workers 6`` (the binding run, scheduled by the chief after the PRE-REG). ``--quick`` is a
plumbing check on the quick stream; its exit is 2, never a verdict. The JSON carries no wall
clock, so identical runs write identical files.

Not on the mypy baseline (``files = ["vibe_quant"]``); kept typed and ruff-clean.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import gatechain
import layer1
import numpy as np
import synth
from scipy.stats import binom

from vibe_quant.discovery import timing_null as tn
from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.discovery.fitness import MIN_TRADES, _const_backtest, _evaluate_single
from vibe_quant.discovery.neff import effective_trials
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)
from vibe_quant.metrics import (
    DailyReturns,
    daily_balance_returns,
    pool_daily_returns,
    profit_factor,
)
from vibe_quant.trade_ledger import moments_from_returns
from vibe_quant.validation.extraction import sharpe_sortino_from_returns

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from vibe_quant.discovery.fitness import FitnessResult

SEED_BASE: int = 20261122
TAG: int = 0x7151
TOPUP_TAG: int = 0x7152
QUICK_TAG: int = 0x71A0

WINDOWS: tuple[str, ...] = ("W3", "W6")
EXPOSURES: tuple[float, ...] = (0.10, 0.30)
MODES: tuple[str, ...] = ("1sym", "6sync", "6async")
SYMBOLS6: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "NEARUSDT", "ZECUSDT")
COST_CASES: tuple[str, ...] = ("fee", "credit")
CHAINS: tuple[str, ...] = ("replacement", "combined")

DAY_NS: int = 86_400 * 10**9
_MS: int = 10**6
#: Strategy timeframe per plan window: (name, bar ns, bars per UTC day).
TIMEFRAMES: dict[str, tuple[str, int, int]] = {
    "W3": ("4h", 4 * 3_600 * 10**9, 6),
    "W6": ("1d", DAY_NS, 1),
}
ANNUAL_VOL: float = 0.60
T_DOF: int = 5
COIN_RHO: float = 0.6
P0: float = 100.0
HOLDOUT_DAYS: int = 365
#: Geometric hold mean in bars (R-gb4-1: W6 2 bars, W3 6).
HOLD_MEAN: dict[str, float] = {"W3": 6.0, "W6": 2.0}
LONG_SHARE: float = 0.6
FRAC_RANGE: tuple[float, float] = (0.05, 0.5)
STARTING_BALANCE: float = synth.STARTING_BALANCE

FEE: float = 0.0005
#: R-gb4-3: every real trade RECEIVES this per 8h settlement on its held notional (both sides).
FUNDING_CREDIT: float = 0.0001
SETTLEMENT_NS: int = 8 * 3_600 * 10**9

SELECTION_K: int = layer1.SELECTION_K
SELECTION_N: int = layer1.SELECTION_K
SINGLE_N: int = layer1.SINGLE_NULL_N
SINGLE_REPS: int = 400
SELECTION_REPS: int = 200
QUICK_SINGLE_REPS: int = 10
QUICK_SELECTION_REPS: int = 4
CALIBRATION_ALPHA: float = 0.05
SINGLE_CHUNK: int = 20
SELECTION_CHUNK: int = 5

JOB: str = "20261122-edge"
TASK: str = "gnull-timing-null"
OUT_DIR: str = "data/swarm/20261122-edge/prereg/gnull"
DEFAULT_WORKERS: int = 6


def _critical_count(reps: int, rate: float = 0.05, alpha: float = CALIBRATION_ALPHA) -> int:
    """Smallest ``k`` with ``P(X >= k | reps, rate) <= alpha`` (exact one-sided binomial)."""
    k = 0
    while binom.sf(k - 1, reps, rate) > alpha:
        k += 1
    return k


#: R-gb4-2: Bonferroni over the 24 calibration lines (12 configs x 2 cost cases).
CALIBRATION_LINES: int = len(WINDOWS) * len(EXPOSURES) * len(MODES) * len(COST_CASES)
#: A calibration line FAILs at this many null-alone passes out of ``SINGLE_REPS`` (34/400).
CALIBRATION_K: int = _critical_count(SINGLE_REPS, alpha=CALIBRATION_ALPHA / CALIBRATION_LINES)


@dataclass(frozen=True, slots=True)
class Config:
    """One G-null configuration; ``index`` is its seed-stream coordinate."""

    index: int
    key: str
    window: str
    exposure: float
    mode: str

    @property
    def symbols(self) -> tuple[str, ...]:
        return ("BTCUSDT",) if self.mode == "1sym" else SYMBOLS6


def configs() -> list[Config]:
    """The 12 pre-registered configs: W3/W6 x e {0.10, 0.30} x {1sym, 6sync, 6async}."""
    out: list[Config] = []
    for window in WINDOWS:
        for e in EXPOSURES:
            for mode in MODES:
                out.append(Config(len(out), f"{window}:e={e}:{mode}", window, e, mode))
    return out


def run_stream(
    quick: bool,
    single_reps: int,
    selection_reps: int,
    candidates: int,
    *,
    all_configs: bool = True,
    outcome: str = "both",
) -> tuple[bool, int]:
    """``(quick, tag)``: a run at any non-binding size -- or over a config subset, or one
    outcome alone -- is a quick-stream run (no peek at the binding replicates, no verdict).
    Only the full 12-config, both-outcome binding command reads tag 0x7151."""
    binding = (
        (single_reps, selection_reps, candidates) == (SINGLE_REPS, SELECTION_REPS, SELECTION_K)
        and all_configs
        and outcome == "both"
    )
    forced = quick or not binding
    return forced, stream_tag(quick=forced, topup=False)


def stream_tag(*, quick: bool, topup: bool) -> int:
    """The seed tag of a run: quick runs never touch the binding or top-up streams."""
    if quick:
        return QUICK_TAG
    return TOPUP_TAG if topup else TAG


def replicate_seq(cfg: Config, rep: int, tag: int) -> np.random.SeedSequence:
    """``SeedSequence([20261122, tag, cfg.index, rep])``: the replicate's root."""
    return np.random.SeedSequence([SEED_BASE, tag, cfg.index, rep])


def _child(seq: np.random.SeedSequence, *key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seq.entropy, spawn_key=key))


# --------------------------------------------------------------------------- tape


@dataclass(frozen=True, slots=True)
class Tapes:
    """One replicate's market: train windows and holdout per symbol, plus 1d closes."""

    train: dict[str, list[tn.WindowTape]]
    holdout: dict[str, tn.WindowTape]
    closes: dict[str, dict[int, float]]


def _log_returns(rng: np.random.Generator, n: int, symbols: int, bars_per_day: int) -> np.ndarray:
    """``(symbols, n)`` multivariate-t5 log returns, unit-variance-scaled to ANNUAL_VOL."""
    common = rng.standard_normal(n)
    own = rng.standard_normal((symbols, n))
    z = math.sqrt(COIN_RHO) * common + math.sqrt(1.0 - COIN_RHO) * own
    w = np.sqrt(T_DOF / rng.chisquare(T_DOF, n))
    sigma = ANNUAL_VOL / math.sqrt(365.0 * bars_per_day)
    return z * w * math.sqrt((T_DOF - 2) / T_DOF) * sigma


def _tape(
    symbol: str, timeframe: str, tf_ns: int, start_ns: int, path: np.ndarray
) -> tn.WindowTape:
    n = len(path) - 1
    open_ns = start_ns + np.arange(n, dtype=np.int64) * tf_ns
    return tn.WindowTape(
        symbol=symbol,
        timeframe=timeframe,
        tf_ns=tf_ns,
        start_ns=start_ns,
        end_ns=start_ns + n * tf_ns,
        open_ns=open_ns,
        ts=open_ns + tf_ns - _MS,
        open=path[:-1].copy(),
        close=path[1:].copy(),
    )


def daily_closes(tapes: Mapping[str, Sequence[tn.WindowTape]]) -> dict[str, dict[int, float]]:
    """``{symbol: {utc day: close of the day's last bar}}`` (the buy-and-hold input)."""
    out: dict[str, dict[int, float]] = {}
    for sym, windows in tapes.items():
        days: dict[int, float] = {}
        for tape in windows:
            for ts, close in zip((tape.ts // DAY_NS).tolist(), tape.close.tolist(), strict=True):
                days[int(ts)] = float(close)
        out[sym] = days
    return out


def draw_tapes(cfg: Config, seq: np.random.SeedSequence) -> Tapes:
    """The replicate's shared tape (spawn key 0) and holdout tape (spawn key 1)."""
    timeframe, tf_ns, per_day = TIMEFRAMES[cfg.window]
    plan = synth.PLANS[cfg.window]
    syms = cfg.symbols
    bounds = [(synth.day_index(ws), synth.day_index(we)) for ws, we in plan.eval_windows]
    sizes = [(we - ws) * per_day for ws, we in bounds]
    logs = _log_returns(_child(seq, 0), sum(sizes), len(syms), per_day)
    end_day = bounds[-1][1]
    hold = _log_returns(_child(seq, 1), HOLDOUT_DAYS * per_day, len(syms), per_day)
    train: dict[str, list[tn.WindowTape]] = {}
    holdout: dict[str, tn.WindowTape] = {}
    for k, sym in enumerate(syms):
        path = P0 * np.exp(np.concatenate([[0.0], np.cumsum(logs[k])]))
        windows: list[tn.WindowTape] = []
        offset = 0
        for (ws, _we), size in zip(bounds, sizes, strict=True):
            windows.append(
                _tape(sym, timeframe, tf_ns, ws * DAY_NS, path[offset : offset + size + 1])
            )
            offset += size
        train[sym] = windows
        hpath = path[-1] * np.exp(np.concatenate([[0.0], np.cumsum(hold[k])]))
        holdout[sym] = _tape(sym, timeframe, tf_ns, end_day * DAY_NS, hpath)
    closes = daily_closes({s: [*train[s], holdout[s]] for s in syms})
    return Tapes(train=train, holdout=holdout, closes=closes)


# --------------------------------------------------------------------------- schedule


@dataclass(frozen=True, slots=True)
class Schedule:
    """One zero-edge candidate: spans per symbol and train window, and on the holdout."""

    train: dict[str, list[list[tn.TradeSpan]]]
    holdout: dict[str, list[tn.TradeSpan]]


def _spans(
    rng: np.random.Generator, n: int, exposure: float, mean_hold: float
) -> list[tn.TradeSpan]:
    target = round(exposure * n)
    holds: list[int] = []
    total = 0
    while True:
        h = int(rng.geometric(1.0 / mean_hold))
        if total + h > target:
            break
        holds.append(h)
        total += h
    k = len(holds)
    sides = np.where(rng.random(k) < LONG_SHARE, 1, -1)
    fracs = rng.uniform(FRAC_RANGE[0], FRAC_RANGE[1], k)
    starts = tn.place_trades(n, np.asarray(holds, dtype=np.int64), rng)
    return [
        tn.TradeSpan(
            int(sides[j]),
            float(fracs[j]),
            int(starts[j]),
            holds[j],
            bool(starts[j] + holds[j] >= n),
        )
        for j in range(k)
    ]


def draw_schedule(cfg: Config, seq: np.random.SeedSequence, i: int, tapes: Tapes) -> Schedule:
    """Candidate ``i``'s schedule from spawn key ``(2, i)``: independent of the tape draw."""
    rng = _child(seq, 2, i)
    mean = HOLD_MEAN[cfg.window]
    first = cfg.symbols[0]
    lens = [len(t) for t in tapes.train[first]]
    hlen = len(tapes.holdout[first])

    def one() -> tuple[list[list[tn.TradeSpan]], list[tn.TradeSpan]]:
        return [_spans(rng, n, cfg.exposure, mean) for n in lens], _spans(
            rng, hlen, cfg.exposure, mean
        )

    if cfg.mode == "6async":
        drawn = {sym: one() for sym in cfg.symbols}
    else:
        shared = one()
        drawn = dict.fromkeys(cfg.symbols, shared)
    return Schedule(
        train={s: list(v[0]) for s, v in drawn.items()},
        holdout={s: list(v[1]) for s, v in drawn.items()},
    )


def replicate(cfg: Config, rep: int, tag: int, candidates: int) -> tuple[Tapes, list[Schedule]]:
    """The tape first, then ``candidates`` schedules (each from its own child stream)."""
    seq = replicate_seq(cfg, rep, tag)
    tapes = draw_tapes(cfg, seq)
    return tapes, [draw_schedule(cfg, seq, i, tapes) for i in range(candidates)]


# --------------------------------------------------------------------------- costs and the real side


def funding_credit_hook(rate: float) -> tn.FundingHook:
    """``rate`` x notional RECEIVED per 8h settlement in ``(entry, exit]`` whatever the side
    (a negative amount = received)."""

    def hook(
        _sym: str, _side: int, notional: float, entry_ns: int, exit_ns: int
    ) -> list[tuple[int, float]]:
        first = (entry_ns // SETTLEMENT_NS + 1) * SETTLEMENT_NS
        return [(t, -rate * notional) for t in range(first, exit_ns + 1, SETTLEMENT_NS)]

    return hook


def real_costs(case: str) -> tn.TradeCosts:
    """The real side's pre-registered cost case (GATES-9)."""
    if case == "fee":
        return tn.TradeCosts(fee=FEE)
    if case == "credit":
        return tn.TradeCosts(fee=FEE, funding=funding_credit_hook(FUNDING_CREDIT))
    msg = f"unknown cost case {case!r}"
    raise ValueError(msg)


def trade_pnls(
    tape: tn.WindowTape, spans: Sequence[tn.TradeSpan], costs: tn.TradeCosts, b0: float
) -> list[tuple[float, float, int]]:
    """``(equity at entry, net PnL, exit ts)`` per trade, the cash ``sim_daily`` books.

    Same fills, fees, funding, run-end mark and stop-at-ruin as ``tn.sim_daily``, per trade
    (its daily path only carries the account), so the bt metrics read the real side's trades.
    """
    n = len(tape)
    realized = 0.0
    out: list[tuple[float, float, int]] = []
    for sp in sorted(spans, key=lambda s: s.entry_bar):
        i, s = sp.entry_bar, sp.side
        equity = b0 + realized
        if not equity > 0:
            break
        at_close = i + sp.hold_bars >= n
        xb = n - 1 if at_close else i + sp.hold_bars
        p_in, p_out = float(tape.open[i]), float(tape.close[n - 1] if at_close else tape.open[xb])
        if costs.fill_price is not None:
            p_in = costs.fill_price(tape.symbol, s, p_in)
            if not at_close:
                p_out = costs.fill_price(tape.symbol, -s, p_out)
        q = sp.frac * equity / p_in
        paid = 0.0
        if costs.funding is not None:
            pays = costs.funding(tape.symbol, s, q * p_in, *tn.funding_window(tape, i, xb, at_close))
            paid = sum(amount for _ts, amount in pays)
        pnl = s * q * (p_out - p_in) - costs.fee * q * p_in - costs.fee * q * p_out - paid
        realized += pnl
        out.append((equity, pnl, int(tape.ts[xb])))
    return out


def window_bt(
    tape: tn.WindowTape,
    spans: Sequence[tn.TradeSpan],
    costs: tn.TradeCosts,
    b0: float = STARTING_BALANCE,
) -> dict[str, Any]:
    """One window's bt dict (the NT runner's keys) from the as-traded replay."""
    daily = tn.sim_daily(tape, spans, costs, b0)
    trades = trade_pnls(tape, spans, costs, b0)
    pnls = [pnl for _eq, pnl, _ts in trades]
    trade_returns = tuple(pnl / eq for eq, pnl, _ts in trades)
    sharpe, sortino = sharpe_sortino_from_returns(
        daily_balance_returns(
            b0, [(ts, pnl) for _eq, pnl, ts in trades], tape.start_ns, tape.end_ns
        )
    )
    values = np.frombuffer(daily.values, dtype=np.float64)
    equity = np.cumprod(1.0 + values) * b0
    peaks = np.maximum.accumulate(np.concatenate([[b0], equity]))[:-1]
    skewness, kurtosis, _ = moments_from_returns(list(trade_returns))
    return {
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": float(np.max((peaks - equity) / peaks)) if len(values) else 0.0,
        "profit_factor": profit_factor(pnls),
        "total_trades": len(trades),
        "total_return": tn.total_return(values),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "trade_returns": trade_returns,
        "mtm_daily_returns": daily,
        "starting_balance": b0,
        "days": (tape.end_ns - tape.start_ns) // DAY_NS,
    }


def _symbols_bt(results: list[dict[str, Any]], symbols: Sequence[str]) -> dict[str, Any]:
    if len(symbols) == 1:
        return results[0]
    return dict(
        NTBacktestFn._aggregate_symbols(
            cast("list[dict[str, float | int]]", results), list(symbols)
        )
    )


def candidate_bt(cfg: Config, tapes: Tapes, sched: Schedule, case: str) -> dict[str, Any]:
    """Train bt: windows per symbol (worst-of-N, chained MTM), then symbols (worst mode)."""
    costs = real_costs(case)
    per_symbol = []
    for sym in cfg.symbols:
        results = [
            window_bt(t, sp, costs)
            for t, sp in zip(tapes.train[sym], sched.train[sym], strict=True)
        ]
        rows = cast("list[dict[str, float | int]]", results)
        agg: dict[str, Any] = dict(NTBacktestFn._aggregate_multi_window(rows, min_trades=0))
        agg["mtm_daily_returns"] = NTBacktestFn._chain_window_series(rows, key="mtm_daily_returns")
        per_symbol.append(agg)
    return _symbols_bt(per_symbol, cfg.symbols)


def holdout_bt(cfg: Config, tapes: Tapes, sched: Schedule, case: str) -> dict[str, Any]:
    """The candidate's holdout window metrics (its own schedule on the holdout tape)."""
    costs = real_costs(case)
    results = [window_bt(tapes.holdout[s], sched.holdout[s], costs) for s in cfg.symbols]
    return _symbols_bt(results, cfg.symbols)


def real_series(cfg: Config, tapes: Tapes, sched: Schedule, case: str) -> DailyReturns:
    """The candidate's train MTM series as the gate reads it (chained, aggregated)."""
    series = candidate_bt(cfg, tapes, sched, case)["mtm_daily_returns"]
    if series is None:
        msg = f"{cfg.key}: real series off the day grid"
        raise ValueError(msg)
    return cast("DailyReturns", series)


def _chromosome() -> StrategyChromosome:
    """A fixed two-gene BOTH-direction genome (the buy-and-hold bar is ``|bh|``)."""
    return StrategyChromosome(
        entry_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 30.0, None)],
        exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0, None)],
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        direction=Direction.BOTH,
    )


def candidate_fitness(
    cfg: Config,
    tapes: Tapes,
    sched: Schedule,
    case: str,
    chrom: StrategyChromosome | None = None,
) -> FitnessResult:
    """The real ``_evaluate_single`` on the candidate's as-traded train bt."""
    bt = candidate_bt(cfg, tapes, sched, case)
    timeframe = TIMEFRAMES[cfg.window][0]
    genome = chrom if chrom is not None else _chromosome()
    return _evaluate_single(genome, _const_backtest(bt), None, MIN_TRADES, timeframe)


def chain_settings(cfg: Config, dsr_trials: tuple[int, ...]) -> gatechain.ChainSettings:
    """The window's production settings with this config's symbols, no shifted windows
    (``chain_recent`` never reads them) and the 365-day holdout the tape carries."""
    base = gatechain.settings_for(cfg.window)
    end = date.fromisoformat(synth.PLANS[cfg.window].train_end)
    config = replace(
        base.config,
        symbols=list(cfg.symbols),
        symbol_agg="portfolio" if len(cfg.symbols) == 1 else "worst",
        cross_window_months=[],
        wfa_oos_step_days=0,
        holdout_start_date=end.isoformat(),
        holdout_end_date=(end + timedelta(days=HOLDOUT_DAYS)).isoformat(),
        dsr_rule="q95",
    )
    return replace(base, config=config, symbols=cfg.symbols, dsr_trials=dsr_trials)


def _no_windows(start: str, end: str) -> Callable[[StrategyChromosome], dict[str, float | int]]:
    msg = f"G-null scores no shifted windows ({start}..{end})"
    raise AssertionError(msg)


# --------------------------------------------------------------------------- null


def null_record(
    tapes_train: Mapping[str, Sequence[tn.WindowTape]],
    sched: Schedule,
    real: DailyReturns,
    *,
    costs: tn.TradeCosts,
    seed_base: int,
    key: str,
    closes: Mapping[str, Mapping[int, float]],
    buy_hold: tuple[bool, Sequence[str]] | None,
    draws: int = tn.NULL_DRAWS,
) -> tuple[dict[str, object], np.ndarray]:
    """The production ``timing_null_check`` record and its null draws (captured in draw order
    through the ``aggregate`` hook: the exact values the check ranks ``real`` against). The
    null pays ``costs``: the real side's costs of the case ``real`` was played under."""
    syms = list(tapes_train)
    base = tn.single_series if len(syms) == 1 else pool_daily_returns
    captured: list[float] = []

    def aggregate(parts: Sequence[DailyReturns | None]) -> DailyReturns | None:
        out = base(parts)
        captured.append(tn.total_return(out.values) if out is not None else math.nan)
        return out

    record = tn.timing_null_check(
        real,
        tapes_train,
        {s: sched.train[s] for s in syms},
        costs=costs,
        b0=STARTING_BALANCE,
        seeds={tn.book_name(syms): tn.null_seed(seed_base, key, tn.book_name(syms))},
        aggregate=aggregate,
        closes=closes,
        buy_hold=buy_hold,
        draws=draws,
    )
    return record.as_dict(), np.asarray(captured, dtype=np.float64)


class _NullCache:
    """One null computation per (cost case, buy-and-hold verdict) of a candidate (the record's
    exposure fallback reads the verdict). The null pays its case's real costs and the case's
    real series is ranked on it; nothing is re-ranked across cases."""

    def __init__(self, tapes: Tapes, sched: Schedule, seed: int, key: str):
        self._args = (tapes, sched, seed, key)
        self._done: dict[object, tuple[dict[str, object], np.ndarray]] = {}

    def get(
        self, case: str, real: DailyReturns, buy_hold: tuple[bool, Sequence[str]] | None
    ) -> tuple[dict[str, object], np.ndarray]:
        bh = None if buy_hold is None else (buy_hold[0], tuple(buy_hold[1]))
        if (case, bh) not in self._done:
            tapes, sched, seed, key = self._args
            self._done[case, bh] = null_record(
                tapes.train,
                sched,
                real,
                costs=real_costs(case),
                seed_base=seed,
                key=key,
                closes=tapes.closes,
                buy_hold=buy_hold,
            )
        return self._done[case, bh]

    def callback(
        self, case: str, real: DailyReturns
    ) -> Callable[[tuple[bool, list[str]]], dict[str, object]]:
        def run(buy_hold: tuple[bool, list[str]]) -> dict[str, object]:
            return self.get(case, real, buy_hold)[0]

        return run


def _rep_seed(seq: np.random.SeedSequence) -> int:
    return int(seq.generate_state(1, dtype=np.uint32)[0])


# --------------------------------------------------------------------------- replicates


def single_replicate(cfg: Config, rep: int, tag: int) -> dict[str, dict[str, bool]]:
    """One candidate: the null gate alone (calibration) and the N = 1 chains (info)."""
    seq = replicate_seq(cfg, rep, tag)
    tapes = draw_tapes(cfg, seq)
    sched = draw_schedule(cfg, seq, 0, tapes)
    settings = chain_settings(cfg, (SINGLE_N,))
    reals = {case: real_series(cfg, tapes, sched, case) for case in COST_CASES}
    cache = _NullCache(tapes, sched, _rep_seed(seq), "cand0")
    out: dict[str, dict[str, bool]] = {"null_pass": {}, "chain": {}}
    for case in COST_CASES:
        record, _null = cache.get(
            case, reals[case], (False, ["Buy-hold: calibration reads the null alone"])
        )
        out["null_pass"][case] = tn.timing_null_decision(gatechain.NULL_POLICY, record)[0]
        chrom = _chromosome()
        fit = candidate_fitness(cfg, tapes, sched, case, chrom)
        res = gatechain.evaluate_candidate(
            chrom,
            fit,
            holdout_bt(cfg, tapes, sched, case),
            _no_windows,
            tapes.closes,
            settings,
            timing_null=cache.callback(case, reals[case]),
        )
        for chain in CHAINS:
            out["chain"][f"{chain}:{case}"] = bool(
                cast("dict[int, bool]", res[f"chain_null_{chain}"])[SINGLE_N]
            )
    return out


def selection_replicate(cfg: Config, rep: int, tag: int, candidates: int) -> dict[str, Any]:
    """K candidates on one tape: does ANY train-passer clear each chain at N = 167?"""
    seq = replicate_seq(cfg, rep, tag)
    tapes = draw_tapes(cfg, seq)
    scheds = [draw_schedule(cfg, seq, i, tapes) for i in range(candidates)]
    settings = chain_settings(cfg, (SELECTION_N,))
    seed = _rep_seed(seq)
    caches: dict[int, _NullCache] = {}
    fp = {f"{chain}:{case}": False for chain in CHAINS for case in COST_CASES}
    picks_by_case: dict[str, int] = {}
    for case in COST_CASES:
        entries = []
        for i, sched in enumerate(scheds):
            chrom = _chromosome()
            entries.append((chrom, candidate_fitness(cfg, tapes, sched, case, chrom), i))
        rows = [
            np.frombuffer(fit.mtm_daily_returns.values, dtype=np.float64)
            for _c, fit, _i in entries
            if fit.mtm_daily_returns is not None and len(fit.mtm_daily_returns) >= 2
        ]
        spread = effective_trials(rows, candidates).ev_q95_spread if rows else None
        order = sorted(entries, key=lambda e: e[1].adjusted_score, reverse=True)
        picks = {id(c) for c, _f in layer1.pick_pool([(c, f) for c, f, _i in order], "all")}
        picks_by_case[case] = len(picks)
        for chrom, fit, i in order:
            if id(chrom) not in picks:
                continue
            if fp[f"replacement:{case}"] and fp[f"combined:{case}"]:
                break
            real = cast("DailyReturns", fit.mtm_daily_returns)
            if i not in caches:
                caches[i] = _NullCache(tapes, scheds[i], seed, f"cand{i}")
            res = gatechain.evaluate_candidate(
                chrom,
                fit,
                holdout_bt(cfg, tapes, scheds[i], case),
                _no_windows,
                tapes.closes,
                settings,
                null_spread=spread,
                timing_null=caches[i].callback(case, real),
            )
            for chain in CHAINS:
                if cast("dict[int, bool]", res[f"chain_null_{chain}"])[SELECTION_N]:
                    fp[f"{chain}:{case}"] = True
    return {
        "fp": fp,
        "picks": picks_by_case,
        "null_runs": sum(len(c._done) for c in caches.values()),
    }


# --------------------------------------------------------------------------- grid


def _unit(unit: tuple[str, str, int, int, int, int]) -> list[dict[str, Any]]:
    outcome, key, tag, start, stop, candidates = unit
    cfg = next(c for c in configs() if c.key == key)
    with layer1._quiet_vibe_quant():
        if outcome == "single":
            return [single_replicate(cfg, rep, tag) for rep in range(start, stop)]
        return [selection_replicate(cfg, rep, tag, candidates) for rep in range(start, stop)]


def _units(
    outcome: str, keys: Sequence[str], reps: int, tag: int, candidates: int
) -> list[tuple[str, str, int, int, int, int]]:
    chunk = SINGLE_CHUNK if outcome == "single" else SELECTION_CHUNK
    return [
        (outcome, key, tag, s, min(s + chunk, reps), candidates)
        for key in keys
        for s in range(0, reps, chunk)
    ]


def run_units(
    units: Sequence[tuple[str, str, int, int, int, int]], workers: int
) -> list[list[dict[str, Any]]]:
    """Every unit (inline at ``workers <= 1``); results in unit order for any worker count."""
    if workers <= 1:
        return [_unit(u) for u in units]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_unit, units))


def _collect(
    units: Sequence[tuple[str, str, int, int, int, int]], results: Sequence[list[dict[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for unit, rows in zip(units, results, strict=True):
        out.setdefault(unit[1], []).extend(rows)
    return out


def summarize_single(rows: Sequence[Mapping[str, Mapping[str, bool]]]) -> dict[str, object]:
    return {
        "reps": len(rows),
        "null_pass": {case: sum(r["null_pass"][case] for r in rows) for case in COST_CASES},
        "chain": {k: sum(r["chain"][k] for r in rows) for k in sorted(rows[0]["chain"])}
        if rows
        else {},
    }


def summarize_selection(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    keys = [f"{chain}:{case}" for chain in CHAINS for case in COST_CASES]
    return {
        "reps": len(rows),
        "fp": {k: sum(bool(r["fp"][k]) for r in rows) for k in keys},
        "picks_per_rep": {c: sum(r["picks"][c] for r in rows) / len(rows) for c in COST_CASES}
        if rows
        else {},
        "null_runs": sum(int(r["null_runs"]) for r in rows),
    }


# --------------------------------------------------------------------------- verdicts


def calibration_verdict(counts: Mapping[str, int], reps: int) -> layer1.SequentialVerdict:
    """R-gb4-2: FAIL iff some config's null-alone passes reach ``CALIBRATION_K`` of 400;
    any other sample size is not binding (MARGINAL, exit 2)."""
    failing = (
        tuple(k for k, c in counts.items() if c >= CALIBRATION_K) if reps == SINGLE_REPS else ()
    )
    readings = ", ".join(f"{k}={c}/{reps}" for k, c in counts.items())
    if failing:
        return layer1.SequentialVerdict("FAIL", 1, failing, (), readings)
    if reps != SINGLE_REPS:
        return layer1.SequentialVerdict("MARGINAL", 2, (), (), readings)
    return layer1.SequentialVerdict("PASS", 0, (), (), readings)


def selection_verdict(
    counts: Mapping[str, int], reps: int, topup: Mapping[str, tuple[int, int]] | None = None
) -> layer1.SequentialVerdict:
    """The G-pool two-look table (``layer1.SELECTION_RULE``) on one chain x cost line."""
    return layer1.sequential_verdict(counts, reps, layer1.SELECTION_RULE, topup)


def overall(lines: Mapping[str, layer1.SequentialVerdict], *, quick: bool) -> tuple[str, int]:
    """FAIL if any line FAILs, else MARGINAL if any is, else PASS. A quick-stream run is
    never a verdict (MARGINAL, exit 2), whatever its sample sizes: its lines are information."""
    if quick:
        return "MARGINAL", 2
    codes = {v.exit_code for v in lines.values()}
    if 1 in codes:
        return "FAIL", 1
    if 2 in codes or not lines:
        return "MARGINAL", 2
    return "PASS", 0


def build_report(
    *,
    keys: Sequence[str],
    quick: bool,
    tag: int,
    single_reps: int,
    selection_reps: int,
    candidates: int,
    single: Mapping[str, Mapping[str, object]],
    selection: Mapping[str, Mapping[str, object]],
    topup: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """The JSON report (deterministic: no wall clock) with every verdict line."""
    lines: dict[str, layer1.SequentialVerdict] = {}
    if single:
        for case in COST_CASES:
            counts = {k: int(cast("dict[str, int]", single[k]["null_pass"])[case]) for k in single}
            lines[f"calibration:{case}"] = calibration_verdict(counts, single_reps)
    if selection:
        for chain in CHAINS:
            for case in COST_CASES:
                line = f"{chain}:{case}"
                counts = {
                    k: int(cast("dict[str, int]", selection[k]["fp"])[line]) for k in selection
                }
                first = selection_verdict(counts, selection_reps)
                extra = {
                    k: (
                        int(cast("dict[str, int]", topup[k]["fp"])[line]),
                        int(cast("int", topup[k]["reps"])),
                    )
                    for k in first.marginal
                    if k in topup
                }
                lines[f"selection:{line}"] = selection_verdict(
                    counts, selection_reps, extra or None
                )
    status, code = overall(lines, quick=quick)
    return {
        "job": JOB,
        "task": TASK,
        "seed_base": SEED_BASE,
        "seed_tag": tag,
        "topup_tag": TOPUP_TAG,
        "quick": quick,
        "configs": list(keys),
        "single_reps": single_reps,
        "selection_reps": selection_reps,
        "candidates": candidates,
        "cost_cases": list(COST_CASES),
        "null_costs": "per case (= real side)",
        "retiming": "joint",
        "chains": list(CHAINS),
        "rules": {
            "calibration_k": CALIBRATION_K,
            "calibration_reps": SINGLE_REPS,
            "selection": {
                "reps": layer1.SELECTION_RULE.reps,
                "early_pass": layer1.SELECTION_RULE.early_pass,
                "fail_at": layer1.SELECTION_RULE.fail_at,
                "topup_reps": layer1.SELECTION_RULE.topup_reps,
                "topup_pass": layer1.SELECTION_RULE.topup_pass,
            },
            "selection_n": SELECTION_N,
            "single_chain_n": SINGLE_N,
        },
        "market": {
            "annual_vol": ANNUAL_VOL,
            "t_dof": T_DOF,
            "coin_rho": COIN_RHO,
            "holdout_days": HOLDOUT_DAYS,
            "hold_mean": dict(HOLD_MEAN),
            "long_share": LONG_SHARE,
            "frac_range": list(FRAC_RANGE),
            "fee": FEE,
            "funding_credit_8h": FUNDING_CREDIT,
        },
        "single": dict(single),
        "selection": dict(selection),
        "topup": dict(topup),
        "verdict": {
            "status": status,
            "exit_code": code,
            "lines": {
                name: {
                    "status": v.status,
                    "failing": list(v.failing),
                    "marginal": list(v.marginal),
                    "readings": v.readings,
                }
                for name, v in lines.items()
            },
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: the G-null grid (or a ``--quick`` plumbing check on the quick stream)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--configs", default=None, help="comma-separated config keys (default: all 12)"
    )
    parser.add_argument("--outcome", choices=["both", "single", "selection"], default="both")
    parser.add_argument("--single-reps", type=int, default=None)
    parser.add_argument("--selection-reps", type=int, default=None)
    parser.add_argument("--candidates", type=int, default=SELECTION_K)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    known = [c.key for c in configs()]
    keys = (
        known if args.configs is None else [k.strip() for k in args.configs.split(",") if k.strip()]
    )
    unknown = [k for k in keys if k not in known]
    if unknown:
        parser.error(f"unknown config(s) {unknown}; known: {known}")
    single_reps = (
        args.single_reps
        if args.single_reps is not None
        else (QUICK_SINGLE_REPS if args.quick else SINGLE_REPS)
    )
    selection_reps = (
        args.selection_reps
        if args.selection_reps is not None
        else (QUICK_SELECTION_REPS if args.quick else SELECTION_REPS)
    )
    quick, tag = run_stream(
        args.quick,
        single_reps,
        selection_reps,
        args.candidates,
        all_configs=keys == known,
        outcome=args.outcome,
    )
    out = args.out or Path(OUT_DIR) / ("timing_null.quick.json" if quick else "timing_null.json")
    workers = (
        args.workers
        if args.workers is not None
        else min(DEFAULT_WORKERS, max(1, (os.cpu_count() or 2) - 2))
    )

    start = time.perf_counter()
    plan: list[tuple[str, str, int, int, int, int]] = []
    if args.outcome in ("both", "single"):
        plan += _units("single", keys, single_reps, tag, args.candidates)
    if args.outcome in ("both", "selection"):
        plan += _units("selection", keys, selection_reps, tag, args.candidates)
    results = run_units(plan, workers)
    single_rows = _collect(
        [u for u in plan if u[0] == "single"],
        [r for u, r in zip(plan, results, strict=True) if u[0] == "single"],
    )
    sel_rows = _collect(
        [u for u in plan if u[0] == "selection"],
        [r for u, r in zip(plan, results, strict=True) if u[0] == "selection"],
    )
    single = {k: summarize_single(single_rows[k]) for k in keys if k in single_rows}
    selection = {k: summarize_selection(sel_rows[k]) for k in keys if k in sel_rows}

    topup: dict[str, dict[str, object]] = {}
    if selection and not quick and selection_reps == layer1.SELECTION_RULE.reps:
        marginal: set[str] = set()
        for chain in CHAINS:
            for case in COST_CASES:
                counts = {
                    k: int(cast("dict[str, int]", selection[k]["fp"])[f"{chain}:{case}"])
                    for k in selection
                }
                marginal |= set(selection_verdict(counts, selection_reps).marginal)
        if marginal:
            todo = [k for k in keys if k in marginal]
            tunits = _units(
                "selection", todo, layer1.SELECTION_RULE.topup_reps, TOPUP_TAG, args.candidates
            )
            trows = _collect(tunits, run_units(tunits, workers))
            topup = {k: summarize_selection(trows[k]) for k in todo}

    report = build_report(
        keys=keys,
        quick=quick,
        tag=tag,
        single_reps=single_reps,
        selection_reps=selection_reps,
        candidates=args.candidates,
        single=single,
        selection=selection,
        topup=topup,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    verdict = cast("dict[str, Any]", report["verdict"])
    for name, line in cast("dict[str, dict[str, Any]]", verdict["lines"]).items():
        print(f"G-null {name}: [{line['readings']}] -> {line['status']}")
    print(f"G-null verdict: {verdict['status']} (tag 0x{tag:X}, quick={quick})")
    print(f"G-null wrote {out} in {time.perf_counter() - start:.1f}s")
    return int(verdict["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
