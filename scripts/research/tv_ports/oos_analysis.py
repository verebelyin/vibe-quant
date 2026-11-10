"""oos_analysis.py -- judge every TradingView-port setting on its own out-of-sample record.

Pre-registration: docs/discovery-journal.md, 2026-11-10 "PRE-REGISTRATION TradingView ports --
every finding judged without buy-and-hold". Buy-and-hold is context only, never a bar.

    run        --out DIR [--parallel N] [--only SID,...]
        One continuous screening backtest per (setting, coin) over the coin's whole history
        2023-01-01..catalog end (preset_grid.py's engine config: NTBacktestFn, all-in 1x, default
        fees, funding, isolated). Stores the daily mark-to-market returns and the trade list in
        DIR/results.jsonl. Resumable; a failure is a row with ``error``.
    analyze    --out DIR [--draws 500]
        OOS metrics, the exposure-matched random-timing null, ROBUST / BEATS-NULL / candidate,
        DIR/report.md + DIR/summary.json.
    neighbours --out DIR --sid SID [--parallel N]
        +-1 grid step of every numeric param of one setting, on all coins (DIR/neighbours.jsonl).
    validate   --out DIR --sid SID --coins A,B,.. [--parallel N]
        Validation tier (ValidationRunner, 1m fills, full costs) of one setting on its OOS
        segments for the given coins (DIR/validation.jsonl), on a throwaway copy of the state DB.

Universes (fixed before any result): U1 = the distinct (port, timeframe, values) of
watchlist.json ``candidates`` (round 1, search 2026-01-01..08-01); U2 = every round-2 pick of
presets-grid/report.md (all 47, search 2023-01-01..2026-01-01).

Run from the repository root: ``PYTHONPATH=$PWD .venv/bin/python scripts/research/tv_ports/oos_analysis.py ...``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import preset_grid  # noqa: E402  (sibling script: engine config, presets, ledger helpers)

REPO_ROOT = HERE.parents[2]
SWARM = REPO_ROOT / "data" / "swarm" / "20261108-tv-ports"
CATALOG = REPO_ROOT / "data" / "catalog"
COINS = preset_grid.COINS
RUN_START = "2023-01-01"
EPOCH = date(1970, 1, 1)
DAY_NS = 86_400 * 10**9
TF_NS = {"15m": 900 * 10**9, "1h": 3_600 * 10**9, "4h": 14_400 * 10**9}
TF_BAR = {"15m": "15-MINUTE", "1h": "1-HOUR", "4h": "4-HOUR"}
ANNUAL = math.sqrt(365.0)

# Pre-registered bars
MIN_COIN_SHARE = 0.75  # return > 0 on >= 6 of 8 coins (>= 75% of coins tested)
MIN_CELL_SHARE = 0.60  # (coin, calendar half-year) cells with return > 0
MIN_CELL_DAYS = 30  # a half-year cell needs this many OOS days of the coin
NULL_DRAWS = 500
NULL_PCT = 0.95
BOOT_N = 2_000
BOOT_LEVEL = 0.95
FORWARD_FROM = "2026-10-20"
DSR_TRIALS = 3_280 + 7_520  # every setting searched for these ports (round 1 + round 2)

Row = Mapping[str, Any]


def day_of(iso: str) -> int:
    return (date.fromisoformat(iso) - EPOCH).days


def iso_of(day: int) -> str:
    return date.fromordinal(EPOCH.toordinal() + day).isoformat()


# --------------------------------------------------------------------------- universe


@dataclass(frozen=True)
class Setting:
    sid: str
    port: str
    tf: str
    values: dict[str, Any]
    universes: tuple[str, ...]
    label: str


def setting_id(port: str, tf: str, values: Mapping[str, Any]) -> str:
    raw = json.dumps([port, tf, dict(sorted(values.items()))], sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


def _full_values(port: str, values: Mapping[str, Any]) -> dict[str, Any]:
    from vibe_quant.discovery.port import load_port

    return {**load_port(HERE / port).defaults(), **values}


def report_picks(report_md: Path) -> list[tuple[str, str, str]]:
    """``(port, tf, preset)`` of every pick row in presets-grid/report.md."""
    picks: list[tuple[str, str, str]] = []
    for line in report_md.read_text().splitlines():
        m = re.match(r"\| (\w+) \| (15m|1h|4h) \| (\S+) \| \d", line)
        if m:
            picks.append((m[1], m[2], m[3]))
    return picks


def universe() -> list[Setting]:
    """U1 (round-1 distinct settings) and U2 (round-2 picks), merged by identity."""
    found: dict[str, dict[str, Any]] = {}

    def add(port: str, tf: str, values: Mapping[str, Any], uni: str, label: str) -> None:
        full = _full_values(port, values)
        sid = setting_id(port, tf, full)
        entry = found.setdefault(
            sid, {"port": port, "tf": tf, "values": full, "unis": [], "labels": []}
        )
        if uni not in entry["unis"]:
            entry["unis"].append(uni)
        if label not in entry["labels"]:
            entry["labels"].append(label)

    watch = json.loads((SWARM / "watchlist.json").read_text())
    for cand in watch["candidates"]:
        port = Path(cand["port"]).name
        vals = ",".join(f"{k}={v}" for k, v in sorted(cand["values"].items()))
        add(port, cand["timeframe"], cand["values"], "U1", f"{port}@{cand['timeframe']} {vals}")
    presets = {p.name: {e["name"]: e["values"] for e in preset_grid.load_presets(p)}
               for p in preset_grid.port_dirs("all")}
    for port, tf, preset in report_picks(SWARM / "presets-grid" / "report.md"):
        add(port, tf, presets[port][preset], "U2", f"{port}@{tf} {preset}")
    return [
        Setting(sid, e["port"], e["tf"], e["values"], tuple(sorted(e["unis"])), " | ".join(e["labels"]))
        for sid, e in sorted(found.items(), key=lambda kv: (kv[1]["port"], kv[1]["tf"], kv[0]))
    ]


# --------------------------------------------------------------------------- spans / segments


def coin_spans(coins: Sequence[str]) -> dict[str, tuple[str, str]]:
    """``{coin: (start, end)}``: 2023-01-01 (or first full day + 1) .. last complete day + 1."""
    from vibe_quant.discovery.buyhold import load_daily_closes
    from vibe_quant.validation.extraction import load_catalog_bars

    closes = load_daily_closes(list(coins), CATALOG)
    out: dict[str, tuple[str, str]] = {}
    for coin in coins:
        first = date.fromordinal(EPOCH.toordinal() + min(closes[coin]))
        # isolated liquidation needs MARK bars to the end (BTC LAST runs past its MARK, 2026-11-10)
        mark = load_catalog_bars(CATALOG, f"{coin}-PERP.BINANCE-1-MINUTE-MARK-EXTERNAL")
        mark_end = int(mark.ts[-1] + 1_000_000) // DAY_NS if mark is not None else max(closes[coin]) + 1
        end = iso_of(min(max(closes[coin]) + 1, mark_end))
        start, _ = preset_grid.clip_window(RUN_START, end, first)
        out[coin] = (start.isoformat(), end)
    return out


def oos_segments(uni: str, start: str, end: str) -> list[tuple[int, int]]:
    """OOS day ranges ``[a, b)`` of a universe for a coin run ``[start, end)``."""
    s, e = day_of(start), day_of(end)
    if uni == "U1":
        raw = [(s, day_of("2026-01-01")), (day_of("2026-08-01"), e)]
    elif uni == "U2":
        raw = [(day_of("2026-01-01"), e)]
    elif uni == "FWD":
        raw = [(day_of(FORWARD_FROM), e)]
    else:
        raise ValueError(uni)
    return [(max(a, s), min(b, e)) for a, b in raw if min(b, e) > max(a, s)]


def half_of(day: int) -> str:
    d = date.fromordinal(EPOCH.toordinal() + day)
    return f"{d.year}H{1 if d.month <= 6 else 2}"


# --------------------------------------------------------------------------- pure math


def compound(returns: Iterable[float]) -> float:
    """Total return of a sequence of period returns."""
    total = 1.0
    for r in returns:
        total *= 1.0 + r
    return total - 1.0


def annual_sharpe(daily: Sequence[float] | np.ndarray) -> float:
    """Daily mean / std(ddof=1) x sqrt(365) (the repo's per-day Sharpe, annualized)."""
    from vibe_quant.overfitting.dsr import daily_sharpe_inputs

    return daily_sharpe_inputs(np.asarray(daily, dtype=np.float64))[0] * ANNUAL


def max_drawdown(daily: Sequence[float]) -> float:
    """Largest peak-to-trough fall of the equity built from daily returns (start = 1)."""
    peak = equity = 1.0
    worst = 0.0
    for r in daily:
        equity *= 1.0 + r
        peak = max(peak, equity)
        worst = max(worst, 1.0 - equity / peak)
    return worst


def alpha_beta(y: Sequence[float], x: Sequence[float]) -> tuple[float, float]:
    """OLS ``y = a + b x`` on daily returns: ``(a x 365, b)``; ``(nan, nan)`` if degenerate."""
    ya, xa = np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)
    if len(ya) < 3 or float(np.var(xa)) == 0.0:
        return float("nan"), float("nan")
    b = float(np.cov(xa, ya, ddof=1)[0, 1] / np.var(xa, ddof=1))
    a = float(ya.mean() - b * xa.mean())
    return a * 365.0, b


def pooled_daily(series: Mapping[str, Mapping[int, float]]) -> tuple[list[int], np.ndarray]:
    """Equal-weight mean across coins, per day, over the coins that have that day."""
    days = sorted({d for s in series.values() for d in s})
    pooled = np.array(
        [statistics.fmean(s[d] for s in series.values() if d in s) for d in days], dtype=np.float64
    )
    return days, pooled


def stationary_bootstrap_indices(
    n: int, mean_block: float, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano (1994) stationary bootstrap index matrix ``(n_boot, n)``.

    Each position starts a new block with probability ``p = 1 / mean_block`` (a uniform random
    start); otherwise it continues the previous index + 1, wrapping circularly.
    """
    p = 1.0 / mean_block
    starts = rng.integers(0, n, size=(n_boot, n))
    new_block = rng.random((n_boot, n)) < p
    new_block[:, 0] = True
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def bootstrap_sharpe_ci(
    daily: np.ndarray, *, n_boot: int = BOOT_N, level: float = BOOT_LEVEL, seed: int = 0
) -> tuple[float, float]:
    """Percentile CI of the annualized daily Sharpe under the stationary bootstrap.

    Mean block length ``round(n ** (1/3))`` days (same rate as the trade-level moving-block
    bootstrap, ``vibe_quant.overfitting.bootstrap_sharpe.default_block_length``).
    """
    from vibe_quant.overfitting.bootstrap_sharpe import default_block_length

    x = np.asarray(daily, dtype=np.float64)
    n = len(x)
    if n < 10:
        return float("-inf"), float("inf")
    idx = stationary_bootstrap_indices(n, float(default_block_length(n)), n_boot, np.random.default_rng(seed))
    samples = x[idx]
    std = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpes = np.where(std > 0, samples.mean(axis=1) / std, 0.0) * ANNUAL
    tail = (1.0 - level) / 2.0
    return float(np.quantile(sharpes, tail)), float(np.quantile(sharpes, 1.0 - tail))


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


# --------------------------------------------------------------------------- engine run


_PORTS: dict[str, Any] = {}
_CAPTURED: list[dict[str, Any]] = []  # positions of the last engine run in this process


def _capturing_backtest_fn(coin: str, tf: str, start: str, end: str) -> Any:
    """preset_grid's NTBacktestFn with daily MTM returns on and the positions captured."""
    from vibe_quant.discovery.backtest_fn import NTBacktestFn
    from vibe_quant.launch_settings import LaunchSettings
    from vibe_quant.screening.nt_runner import NTScreeningRunner
    from vibe_quant.trade_ledger import split_positions

    class CapturingRunner(NTScreeningRunner):
        def _extract_metrics(self, params: Any, bt_result: Any, engine: Any, *a: Any, **kw: Any) -> Any:
            closed, still_open = split_positions(engine)
            _CAPTURED[:] = [_position_row(p) for p in [*closed, *still_open]]
            return super()._extract_metrics(params, bt_result, engine, *a, **kw)

    class CapturingFn(NTBacktestFn):
        def _make_runner(self, dsl_dict: Any, start_date: str, end_date: str, symbols: Any = None) -> Any:
            return CapturingRunner(
                dsl_dict=dsl_dict,
                symbols=self.symbols if symbols is None else symbols,
                start_date=start_date,
                end_date=end_date,
                fill_ticks=self.fill_ticks,
                collect_daily_returns=self.collect_daily_returns,
                launch=LaunchSettings(
                    fee_schedule=self.fee_schedule,
                    entry_order=self.entry_order,
                    liquidation=self.liquidation,
                    sizing=self.sizing,
                ),
                pta_precompute=getattr(self, "pta_precompute", True),
            )

    base = preset_grid._backtest_fn(coin, tf, start, end)
    fn = CapturingFn.__new__(CapturingFn)
    fn.__dict__.update(base.__dict__)
    fn.collect_daily_returns = True
    return fn


def _position_row(p: Any) -> dict[str, Any]:
    return {
        "side": 1 if p.entry.name == "BUY" else -1,
        "ts_open": int(p.ts_opened),
        "ts_close": int(p.ts_closed) if p.is_closed else None,
        "px_open": float(p.avg_px_open),
        "px_close": float(p.avg_px_close) if p.is_closed else None,
        "qty": float(p.peak_qty),
        "realized_pnl": float(p.realized_pnl) if p.realized_pnl is not None else None,
    }


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    """One full-history backtest; never raises."""
    row: dict[str, Any] = {
        "kind": task.get("kind", "run"), "key": task["key"], "sid": task["sid"],
        "coin": task["coin"], "tf": task["tf"], "start": task["start"], "end": task["end"],
        "values": task["values"], "port": task["port"], "error": None,
    }
    try:
        os.chdir(REPO_ROOT)
        from vibe_quant.discovery.param_genome import ParamGenome
        from vibe_quant.discovery.port import load_port

        port = _PORTS.get(task["port"]) or _PORTS.setdefault(task["port"], load_port(HERE / task["port"]))
        fn = _capturing_backtest_fn(task["coin"], task["tf"], task["start"], task["end"])
        _CAPTURED.clear()
        res = fn(ParamGenome(port, dict(task["values"])))
        series = res.get("mtm_daily_returns")
        row.update({
            "ret": preset_grid._finite(res.get("total_return", 0.0)),
            "trades": int(res.get("total_trades", 0)),
            "sharpe": preset_grid._finite(res.get("sharpe_ratio", 0.0)),
            "dd": preset_grid._finite(res["max_drawdown"]) if "max_drawdown" in res else None,
            "fees": float(res.get("total_fees", 0.0)),
            "first_day": None if series is None else int(series.first_day),
            "daily": None if series is None else [float(v) for v in series.values],
            "positions": list(_CAPTURED),
            "error": res.get("error"),
        })
        if series is None and not row["error"]:
            row["error"] = "no daily MTM series"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def run_tasks(tasks: list[dict[str, Any]], results: Path, parallel: int) -> None:
    done = errors = 0
    print(f"{len(tasks)} backtests to run", flush=True)
    import multiprocessing

    def record(row: dict[str, Any]) -> None:
        nonlocal done, errors
        preset_grid._append(results, row)
        done += 1
        errors += 1 if row["error"] else 0
        if done % 20 == 0 or done == len(tasks):
            print(f"  {done}/{len(tasks)} done, {errors} errors", flush=True)

    if parallel <= 1:
        for t in tasks:
            record(run_task(t))
        return
    with multiprocessing.get_context("spawn").Pool(
        processes=parallel, maxtasksperchild=preset_grid.WORKER_RECYCLE
    ) as pool:
        for row in pool.imap_unordered(run_task, tasks):
            record(row)


def cmd_run(args: argparse.Namespace) -> int:
    os.chdir(REPO_ROOT)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    present = preset_grid.done_keys(results, retry_errors=args.retry_errors)
    settings = universe()
    if args.only:
        settings = [s for s in settings if s.sid in args.only.split(",")]
    (out / "universe.json").write_text(json.dumps(
        [{"sid": s.sid, "port": s.port, "tf": s.tf, "values": s.values, "universes": s.universes,
          "label": s.label} for s in universe()], indent=1) + "\n")
    spans = coin_spans(COINS)
    tasks = [
        {"key": f"run|{s.sid}|{c}", "sid": s.sid, "port": s.port, "tf": s.tf, "values": s.values,
         "coin": c, "start": spans[c][0], "end": spans[c][1]}
        for s in settings for c in COINS if f"run|{s.sid}|{c}" not in present
    ]
    tasks.sort(key=lambda t: {"15m": 0, "1h": 1, "4h": 2}[str(t["tf"])])
    run_tasks(tasks, results, args.parallel)
    return 0


# --------------------------------------------------------------------------- analysis


_BARS: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def bars(coin: str, tf: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(open_ns, open, close)`` of the coin's catalog bars at ``tf`` (bar ts = close - 1 ms)."""
    if (coin, tf) not in _BARS:
        from vibe_quant.validation.extraction import load_catalog_bars

        b = load_catalog_bars(CATALOG, f"{coin}-PERP.BINANCE-{TF_BAR[tf]}-LAST-EXTERNAL")
        if b is None:
            raise ValueError(f"no {tf} bars for {coin}")
        ts = np.asarray(b.ts, dtype=np.int64) + 1_000_000 - TF_NS[tf]
        _BARS[(coin, tf)] = (ts, np.asarray(b.open, dtype=np.float64), np.asarray(b.close, dtype=np.float64))
    return _BARS[(coin, tf)]


def trade_spans(row: Row, end_ns: int) -> list[tuple[int, int, int, float, float | None]]:
    """``(side, open_ns, close_ns, px_open, px_close)`` per trade; open-at-end closes at ``end_ns``."""
    out = []
    for p in sorted(row["positions"], key=lambda q: q["ts_open"]):
        close = p["ts_close"] if p["ts_close"] is not None else end_ns
        out.append((p["side"], p["ts_open"], min(close, end_ns), p["px_open"], p["px_close"]))
    return out


@dataclass
class CoinOOS:
    coin: str
    ret: float
    trades: int
    sharpe: float
    pf: float
    dd: float
    exposure: float
    alpha: float
    beta: float
    days: int
    cells: dict[str, float]
    null: np.ndarray
    null_cells: dict[str, np.ndarray]
    null_pct: float
    bh: float | None


def coin_oos(
    row: Row, uni: str, coin_daily: Mapping[int, float], closes: Mapping[int, float],
    draws: int, fee: float, seed: int,
) -> CoinOOS | None:
    """Every per-(setting, coin) OOS number and the coin's null draws."""
    from vibe_quant.metrics import profit_factor

    segs = oos_segments(uni, row["start"], row["end"])
    if not segs:
        return None
    first = int(row["first_day"])
    daily = {first + i: v for i, v in enumerate(row["daily"])}
    oos_days = [d for a, b in segs for d in range(a, b) if d in daily]
    vals = [daily[d] for d in oos_days]
    end_ns = day_of(row["end"]) * DAY_NS
    spans = trade_spans(row, end_ns)
    seg_ns = [(a * DAY_NS, b * DAY_NS) for a, b in segs]
    in_oos = [t for t in spans if any(a <= t[1] < b for a, b in seg_ns)]
    exposure_ns = sum(
        max(0, min(t[2], b) - max(t[1], a)) for t in spans for a, b in seg_ns
    )
    total_ns = sum(b - a for a, b in seg_ns)
    pnl_rets = [
        trade_multiplier(t[0], t[3], t[4], fee) - 1.0 for t in in_oos if t[4] is not None
    ]
    x = [coin_daily.get(d) for d in oos_days]
    pairs = [(v, c) for v, c in zip(vals, x, strict=True) if c is not None]
    alpha, beta = alpha_beta([p[0] for p in pairs], [p[1] for p in pairs])
    halves = sorted({half_of(d) for d in oos_days})
    cells = {}
    for h in halves:
        hd = [daily[d] for d in oos_days if half_of(d) == h]
        if len(hd) >= MIN_CELL_DAYS:
            cells[h] = compound(hd)
    # null, per segment
    o_ns, opens, closes_px = bars(row["coin"], row["tf"])
    tf_ns = TF_NS[row["tf"]]
    rng = np.random.default_rng(seed)
    null_total = np.ones(draws)
    cell_names = list(cells)
    null_cells = {h: np.ones(draws) for h in cell_names}
    for (a_ns, b_ns) in seg_ns:
        sel = np.nonzero((o_ns >= a_ns) & (o_ns < b_ns))[0]
        if len(sel) == 0:
            continue
        prices = np.concatenate((opens[sel], [closes_px[sel[-1]]]))
        seg_trades = [t for t in spans if a_ns <= t[1] < b_ns]
        holds = np.array(
            [max(1, round((min(t[2], b_ns) - t[1]) / tf_ns)) for t in seg_trades], dtype=np.int64
        )
        while holds.sum() > len(sel):  # rounding can overfill by a bar or two
            holds[int(np.argmax(holds))] -= 1
        sides = np.array([t[0] for t in seg_trades], dtype=np.float64)
        bar_cell = np.array(
            [cell_names.index(half_of(int(t // DAY_NS))) if half_of(int(t // DAY_NS)) in cells else -1
             for t in o_ns[sel]], dtype=np.int64,
        )
        tot, per_cell = null_draws(prices, holds, sides, fee, draws, rng, bar_cell, len(cell_names))
        null_total *= 1.0 + tot
        for c, h in enumerate(cell_names):
            null_cells[h] *= 1.0 + per_cell[:, c]
    null_total -= 1.0
    for h in cell_names:
        null_cells[h] -= 1.0
    ret = compound(vals)
    bh_parts = []
    for a, b in segs:
        base, last = closes.get(a - 1), closes.get(b - 1)
        bh_parts.append(None if base is None or last is None else last / base - 1.0)
    bh = None if any(p is None for p in bh_parts) else compound([p for p in bh_parts if p is not None])
    return CoinOOS(
        coin=row["coin"], ret=ret, trades=len(in_oos), sharpe=annual_sharpe(vals),
        pf=profit_factor(pnl_rets), dd=max_drawdown(vals),
        exposure=exposure_ns / total_ns if total_ns else 0.0, alpha=alpha, beta=beta,
        days=len(vals), cells=cells, null=null_total, null_cells=null_cells,
        null_pct=null_percentile(ret, null_total), bh=bh,
    )


def judge(coins: Sequence[CoinOOS], pooled_series: Mapping[str, Mapping[int, float]], seed: int) -> dict[str, Any]:
    """ROBUST / BEATS-NULL / candidate of one setting + the chance rate from its null."""
    tested = len(coins)
    need = math.ceil(MIN_COIN_SHARE * tested - 1e-9)
    positive = sum(1 for c in coins if c.ret > 0)
    cells = [v for c in coins for v in c.cells.values()]
    cell_share = sum(1 for v in cells if v > 0) / len(cells) if cells else 0.0
    _, pooled = pooled_daily(pooled_series)
    sharpe = annual_sharpe(pooled)
    lo, hi = bootstrap_sharpe_ci(pooled, seed=seed)
    robust = positive >= need and cell_share >= MIN_CELL_SHARE and lo > 0
    from vibe_quant.overfitting.dsr import (
        DeflatedSharpeRatio,
        daily_sharpe_inputs,
        mtm_dsr_calculate,
    )

    sd, sk, ku = daily_sharpe_inputs(pooled)
    dsr, _ = mtm_dsr_calculate(DeflatedSharpeRatio(), sd, sk, ku, len(pooled), DSR_TRIALS)
    real_port = statistics.fmean(c.ret for c in coins)
    null_port = np.mean(np.vstack([c.null for c in coins]), axis=0)
    port_pct = null_percentile(real_port, null_port)
    beats = port_pct >= NULL_PCT
    # chance: each null draw as a no-skill twin of this setting (Sharpe-CI part not modelled)
    draws = len(null_port)
    null_pos = np.sum(np.vstack([c.null > 0 for c in coins]), axis=0)
    cell_stack = [c.null_cells[h] > 0 for c in coins for h in c.cells]
    null_cell_share = np.mean(np.vstack(cell_stack), axis=0) if cell_stack else np.zeros(draws)
    null_rank = np.array([np.mean(null_port < v) for v in null_port])
    chance_rate = float(np.mean((null_pos >= need) & (null_cell_share >= MIN_CELL_SHARE) & (null_rank >= NULL_PCT)))
    return {
        "tested": tested, "need": need, "positive": positive, "cells": len(cells),
        "cell_share": cell_share, "pooled_sharpe": sharpe, "ci_lo": lo, "ci_hi": hi,
        "robust": robust, "dsr_p": None if dsr is None else dsr.p_value, "port_ret": real_port, "port_null_pct": port_pct,
        "null_port_median": float(np.median(null_port)), "null_port_p95": float(np.quantile(null_port, NULL_PCT)),
        "beats_null": beats, "candidate": robust and beats, "chance_rate": chance_rate,
        "null_robust_parts_rate": float(np.mean((null_pos >= need) & (null_cell_share >= MIN_CELL_SHARE))),
    }


def seed_of(*parts: str) -> int:
    return int(hashlib.sha1("|".join(parts).encode()).hexdigest()[:8], 16)


def analyze_setting(
    setting: Mapping[str, Any], rows: Mapping[str, Row], uni: str,
    coin_daily: Mapping[str, Mapping[int, float]], closes: Mapping[str, Mapping[int, float]],
    draws: int, fee: float,
) -> dict[str, Any]:
    per: list[CoinOOS] = []
    pooled: dict[str, dict[int, float]] = {}
    errors = []
    for coin in COINS:
        row = rows.get(coin)
        if row is None or row.get("error"):
            errors.append(coin)
            continue
        c = coin_oos(row, uni, coin_daily[coin], closes[coin], draws, fee, seed_of(setting["sid"], coin, uni))
        if c is None:
            continue
        per.append(c)
        first = int(row["first_day"])
        segs = oos_segments(uni, row["start"], row["end"])
        pooled[coin] = {
            first + i: v for i, v in enumerate(row["daily"])
            if any(a <= first + i < b for a, b in segs)
        }
    verdict = judge(per, pooled, seed_of(setting["sid"], uni, "boot")) if per else {}
    return {"sid": setting["sid"], "label": setting["label"], "uni": uni, "errors": errors,
            "coins": per, **verdict}


def load_results(path: Path) -> dict[str, dict[str, Row]]:
    by: dict[str, dict[str, Row]] = {}
    for r in preset_grid.read_rows(path):
        by.setdefault(r["sid"], {})[r["coin"]] = r
    return by


def _pct(v: float | None) -> str:
    return "n/a" if v is None or not math.isfinite(v) else f"{v * 100:+.1f}%"


def _f(v: float, nd: int = 2) -> str:
    return "n/a" if not math.isfinite(v) else f"{v:.{nd}f}"


def cmd_analyze(args: argparse.Namespace) -> int:
    from vibe_quant.discovery.buyhold import load_daily_closes
    from vibe_quant.fees import FeeSchedule

    os.chdir(REPO_ROOT)
    out = Path(args.out)
    settings = json.loads((out / "universe.json").read_text())
    rows = load_results(out / "results.jsonl")
    closes = load_daily_closes(list(COINS), CATALOG)
    coin_daily = {c: {d: v / closes[c][d - 1] - 1.0 for d, v in closes[c].items() if d - 1 in closes[c]} for c in COINS}
    fee = float(FeeSchedule().taker_rate)
    results: list[dict[str, Any]] = []
    for uni in ("U1", "U2"):
        for s in settings:
            if uni in s["universes"]:
                results.append(analyze_setting(s, rows.get(s["sid"], {}), uni, coin_daily, closes, args.draws, fee))
                print(f"{uni} {s['label'][:60]} done", flush=True)
    text, summary = build_report(results, settings, rows, fee, args.draws)
    (out / "report.md").write_text(text)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print(text)
    return 0


def build_report(results: Sequence[Mapping[str, Any]], settings: Sequence[Mapping[str, Any]],
                 rows: Mapping[str, Mapping[str, Row]], fee: float, draws: int) -> tuple[str, dict[str, Any]]:
    lines = ["# TradingView ports -- every setting judged out of sample, without buy-and-hold", ""]
    lines.append(
        f"{len(settings)} distinct settings, {sum(len(v) for v in rows.values())} coin runs, "
        f"{sum(1 for v in rows.values() for r in v.values() if r.get('error'))} errors. Null: {draws} "
        f"draws per (setting, coin), fee {fee} per side, no slippage or funding (null favoured)."
    )
    summary: dict[str, Any] = {"universes": {}}
    for uni in ("U1", "U2"):
        res = [r for r in results if r["uni"] == uni]
        cands = [r for r in res if r.get("candidate")]
        expected = sum(r.get("chance_rate", 0.0) for r in res)
        robust = sum(1 for r in res if r.get("robust"))
        beats = sum(1 for r in res if r.get("beats_null"))
        summary["universes"][uni] = {
            "settings": len(res), "robust": robust, "beats_null": beats,
            "candidates": [r["label"] for r in cands], "expected_false_candidates": expected,
            "expected_robust_parts_by_chance": sum(r.get("null_robust_parts_rate", 0.0) for r in res),
        }
        oos = "2023-01-01..2026-01-01 + 2026-08-01..end" if uni == "U1" else "2026-01-01..end"
        lines += [
            "", f"## {uni} (OOS {oos})", "",
            f"- settings {len(res)}; ROBUST {robust}; BEATS-NULL {beats}; candidates {len(cands)}; "
            f"expected false candidates under chance {expected:.2f} (null-estimated, Sharpe-CI part not "
            f"modelled: an upper bound)",
            "",
            "| setting | OOS return per coin | coins>0 | half-year cells>0 | pooled Sharpe [95% CI] | "
            "8-coin mean | null pct | DSR p (N 10,800, info) | ROBUST | BEATS-NULL |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in sorted(res, key=lambda r: -r.get("pooled_sharpe", -9)):
            if not r["coins"]:
                lines.append(f"| {r['label']} | no runs ({r['errors']}) | | | | | | | | |")
                continue
            per = " ".join(f"{c.coin[:-4]} {_pct(c.ret)}" for c in r["coins"])
            lines.append(
                f"| {r['label']} | {per} | {r['positive']}/{r['tested']} | "
                f"{r['cell_share'] * 100:.0f}% of {r['cells']} | {_f(r['pooled_sharpe'])} "
                f"[{_f(r['ci_lo'])}, {_f(r['ci_hi'])}] | {_pct(r['port_ret'])} | "
                f"{r['port_null_pct'] * 100:.1f} | {_f(r['dsr_p'] if r['dsr_p'] is not None else math.nan, 3)} | "
                f"{'YES' if r['robust'] else 'no'} | "
                f"{'YES' if r['beats_null'] else 'no'} |"
            )
        lines += ["", f"### {uni} per-coin detail (return, trades, Sharpe, PF, max DD, exposure, alpha/yr, beta, null pct, B&H context)", ""]
        for r in res:
            if not r["coins"]:
                continue
            lines.append(f"- **{r['label']}**")
            for c in r["coins"]:
                lines.append(
                    f"  - {c.coin}: {_pct(c.ret)}, {c.trades} tr, Sh {_f(c.sharpe)}, PF {_f(c.pf)}, "
                    f"DD {_pct(c.dd)}, exp {c.exposure * 100:.0f}%, alpha {_pct(c.alpha)}, beta {_f(c.beta)}, "
                    f"null pct {c.null_pct * 100:.1f}, B&H {_pct(c.bh)}"
                )
        summary["universes"][uni]["rows"] = [
            {k: v for k, v in r.items() if k != "coins"} | {
                "per_coin": {c.coin: {"ret": c.ret, "trades": c.trades, "sharpe": c.sharpe, "pf": c.pf,
                                      "dd": c.dd, "exposure": c.exposure, "alpha": c.alpha, "beta": c.beta,
                                      "null_pct": c.null_pct, "bh": c.bh, "cells": c.cells}
                             for c in r["coins"]}}
            for r in res
        ]
    return "\n".join(lines) + "\n", summary


# --------------------------------------------------------------------------- candidates: (b) (a)


def neighbours(port_name: str, values: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """``(label, values)`` for +-1 grid step of each free numeric param, in domain only."""
    from vibe_quant.discovery.port import load_port

    port = load_port(HERE / port_name)
    out = []
    for p in port.params:
        if p.kind == "choice" or not p.is_free:
            continue
        for sign in (-1, 1):
            idx = p.index_of(values[p.name]) + sign
            if idx < 0 or idx > p.n_steps:
                continue
            cand = {**values, p.name: p.value_at(idx)}
            if port.satisfies_constraints(cand):
                out.append((f"{p.name}{'+' if sign > 0 else '-'}1", cand))
    return out


def _setting(out: Path, sid: str) -> dict[str, Any]:
    for s in json.loads((out / "universe.json").read_text()):
        if s["sid"] == sid:
            return dict(s)
    raise SystemExit(f"unknown sid {sid}")


def cmd_neighbours(args: argparse.Namespace) -> int:
    os.chdir(REPO_ROOT)
    out = Path(args.out)
    s = _setting(out, args.sid)
    path = out / "neighbours.jsonl"
    present = preset_grid.done_keys(path)
    spans = coin_spans(COINS)
    tasks = []
    for label, vals in neighbours(s["port"], s["values"]):
        nsid = setting_id(s["port"], s["tf"], vals)
        for c in COINS:
            key = f"nb|{args.sid}|{nsid}|{c}"
            if key not in present:
                tasks.append({"kind": "nb", "key": key, "sid": nsid, "port": s["port"], "tf": s["tf"],
                              "values": vals, "coin": c, "start": spans[c][0], "end": spans[c][1],
                              "parent": args.sid, "nb": label})
    run_tasks(tasks, path, args.parallel)
    # summary per neighbour: OOS return per coin for each universe of the parent
    by: dict[str, dict[str, Row]] = {}
    for r in preset_grid.read_rows(path):
        if r["key"].startswith(f"nb|{args.sid}|"):
            by.setdefault(r["sid"], {})[r["coin"]] = r
    for uni in s["universes"]:
        good = 0
        for nsid, coins in by.items():
            rets = []
            for nr in coins.values():
                if nr.get("error"):
                    continue
                first = int(nr["first_day"])
                segs = oos_segments(uni, nr["start"], nr["end"])
                rets.append(compound(v for i, v in enumerate(nr["daily"]) if any(a <= first + i < b for a, b in segs)))
            pos = sum(1 for x in rets if x > 0)
            need = math.ceil(MIN_COIN_SHARE * len(rets) - 1e-9)
            good += pos >= need
            print(f"{uni} neighbour {nsid} {json.dumps(next(iter(coins.values()))['values'])}: "
                  f"coins>0 {pos}/{len(rets)} mean {_pct(statistics.fmean(rets) if rets else None)}")
        print(f"{uni}: {good}/{len(by)} neighbours positive on >= 75% of coins")
    return 0


def _validation_job(task: dict[str, Any]) -> dict[str, Any]:
    import shutil
    import sqlite3
    import tempfile

    os.chdir(REPO_ROOT)
    from vibe_quant.discovery.port import load_port
    from vibe_quant.validation.runner import ValidationRunner

    row: dict[str, Any] = {k: task[k] for k in ("key", "sid", "coin", "start", "end", "uni")}
    tmp = Path(tempfile.mkdtemp(prefix="tvp-oos-val-"))
    try:
        dsl = dict(load_port(HERE / task["port"]).render(task["values"], timeframe=task["tf"]))
        dsl["name"] = f"tvp_oos_{task['sid']}"
        params = {"liquidation_mode": "isolated", "leverage": 1, "max_position_pct": 1.0,
                  "risk_per_trade": 0.5, "max_margin_pct": 1.0}
        db = tmp / "v.db"
        src = sqlite3.connect("file:data/state/vibe_quant.db?mode=ro", uri=True)
        dst = sqlite3.connect(db)
        src.backup(dst)
        dst.close()
        src.close()
        con = sqlite3.connect(db)
        sid = con.execute("INSERT INTO strategies(name, dsl_config) VALUES (?, ?)",
                          (f"{dsl['name']}_{task['coin']}_{task['start']}", json.dumps(dsl))).lastrowid
        rid = con.execute(
            "INSERT INTO backtest_runs(strategy_id, run_mode, symbols, timeframe, start_date, end_date, parameters) "
            "VALUES (?,?,?,?,?,?,?)",
            (sid, "validation", json.dumps([task["coin"]]), task["tf"], task["start"], task["end"], json.dumps(params)),
        ).lastrowid
        con.commit()
        con.close()
        runner = ValidationRunner(db_path=db, logs_path=tmp / "logs")
        assert rid is not None
        try:
            m = runner.run(rid)
        finally:
            runner.close()
        row.update({"ret": m.total_return, "trades": m.total_trades, "sharpe": m.sharpe_ratio,
                    "dd": m.max_drawdown, "error": None})
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:400]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return row


def cmd_validate(args: argparse.Namespace) -> int:
    os.chdir(REPO_ROOT)
    out = Path(args.out)
    s = _setting(out, args.sid)
    path = out / "validation.jsonl"
    present = preset_grid.done_keys(path)
    spans = coin_spans(COINS)
    tasks = []
    for uni in s["universes"]:
        for c in args.coins.split(","):
            for a, b in oos_segments(uni, *spans[c]):
                key = f"val|{s['sid']}|{c}|{iso_of(a)}"
                if key not in present:
                    tasks.append({"key": key, "sid": s["sid"], "port": s["port"], "tf": s["tf"],
                                  "values": s["values"], "coin": c, "start": iso_of(a), "end": iso_of(b), "uni": uni})
    import multiprocessing

    with multiprocessing.get_context("spawn").Pool(processes=max(1, args.parallel)) as pool:
        for row in pool.imap_unordered(_validation_job, tasks):
            preset_grid._append(path, row)
            print(json.dumps(row), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--out", required=True)
    run.add_argument("--parallel", type=int, default=1)
    run.add_argument("--only", default="")
    run.add_argument("--retry-errors", action="store_true")
    run.set_defaults(fn=cmd_run)
    an = sub.add_parser("analyze")
    an.add_argument("--out", required=True)
    an.add_argument("--draws", type=int, default=NULL_DRAWS)
    an.set_defaults(fn=cmd_analyze)
    nb = sub.add_parser("neighbours")
    nb.add_argument("--out", required=True)
    nb.add_argument("--sid", required=True)
    nb.add_argument("--parallel", type=int, default=1)
    nb.set_defaults(fn=cmd_neighbours)
    va = sub.add_parser("validate")
    va.add_argument("--out", required=True)
    va.add_argument("--sid", required=True)
    va.add_argument("--coins", required=True)
    va.add_argument("--parallel", type=int, default=1)
    va.set_defaults(fn=cmd_validate)
    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
