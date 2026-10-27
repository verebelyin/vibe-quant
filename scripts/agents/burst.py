"""Burst discovery: many ~30 s seeded discoveries on a scratch DB to map where signal lives.

One CELL = one tiny discovery (default pop 6 x 1 gen, 1 eval window, no holdout, no gates)
for one indicator pair x direction x symbol x month x leverage x timeframe (1m default;
15m/1h need the NT tier -- the fast tier replays 1m only). ``run`` samples cells
from the full grid (a fixed shuffle, so a bigger ``--cells`` is a superset of a smaller
one), runs them in parallel subprocesses and appends one JSON line per cell to
``<out>/ledger.jsonl`` (resumable: cells already in the ledger are skipped). Every cell
runs against its own scratch DB under ``/tmp`` -- the real state DB is never opened, so
burst trials never enter a real run's DSR trial count. The months run to 2026-10-19
(user, 2026-10-20: tune on the latest data), so a confirm run on the same range has NO
unseen holdout -- paper trading is the out-of-sample test. Record the burst trial total
in the journal.

    PYTHONPATH=$PWD .venv/bin/python scripts/agents/burst.py run \\
        --out data/swarm/burst-20261020 --cells 24 --parallel 2 --budget-s 1800
    PYTHONPATH=$PWD .venv/bin/python scripts/agents/burst.py transfer --out data/swarm/burst-20261020
    PYTHONPATH=$PWD .venv/bin/python scripts/agents/burst.py report --out data/swarm/burst-20261020

``transfer`` re-evaluates each winner's genome on every OTHER symbol of the ledger,
same window, and on the SAME symbol in the NEXT window (clipped to 2026-10-19), so a
win share that is selection noise cannot pass for a cross-symbol edge. It appends one
JSON line per winner to ``<out>/transfer.jsonl`` (resumable by key) and ``report``
prints the transfer share per indicator-pair family.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import date

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PAIRS: tuple[tuple[str, ...], ...] = (
    ("CCI", "STOCH"),
    ("MFI", "WILLR"),
    ("DONCHIAN", "NATR"),
    ("ROC", "PRICE_POSITION"),
    ("RSI", "RAMS"),
    ("KAMA", "NATR"),
    ("BBANDS", "CCI"),
    ("SQUEEZE_RATIO", "ROC"),
)
DIRECTIONS = ("long", "short", "both")
SYMBOLS = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "HYPEUSDT", "NEARUSDT", "SUIUSDT", "ZECUSDT",
)
# [start, end): the engine's end date is exclusive, so each month ends on the next 1st.
MONTHS: tuple[tuple[str, str], ...] = (
    ("2026-03-01", "2026-04-01"),
    ("2026-04-01", "2026-05-01"),
    ("2026-05-01", "2026-06-01"),
    ("2026-06-01", "2026-07-01"),
    ("2026-07-01", "2026-08-01"),
    ("2026-08-01", "2026-09-01"),
    ("2026-09-01", "2026-10-01"),
    ("2026-10-01", "2026-10-20"),  # through 2026-10-19, the last full day in the catalog
)
LEVERAGES = (3, 10, 25)
TIMEFRAMES = ("1m",)  # 1m keeps every pre-timeframe ledger key resumable
MIN_TRADES = 50  # fitness.MIN_TRADES_1M
CELL_TIMEOUT_S = 600
# MEASURED 2026-10-20 (load 14/10 cores): BTCUSDT 1m, one month = ~22 s per backtest.
TRANSFER_CLIP_END = "2026-10-20"  # exclusive: 2026-10-19 is the last full day in the catalog


def default_grid() -> dict[str, Any]:
    """The grid factors (each overridable on the ``run`` command line)."""
    return {
        "pairs": PAIRS, "directions": DIRECTIONS, "symbols": SYMBOLS,
        "windows": MONTHS, "leverages": LEVERAGES, "timeframes": TIMEFRAMES,
        "eval_windows": 1,
    }


def all_cells(seeds: int = 1, grid: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Every grid cell, in factorial order."""
    g = default_grid() if grid is None else grid
    return [
        {
            "pair": list(pair), "direction": d, "symbol": s,
            "start": m[0], "end": m[1], "leverage": lev, "timeframe": tf, "seed": seed,
            "eval_windows": g["eval_windows"],
        }
        for pair, d, s, m, lev, tf, seed in itertools.product(
            g["pairs"], g["directions"], g["symbols"], g["windows"], g["leverages"],
            g["timeframes"], range(1, seeds + 1),
        )
    ]


def cell_key(cell: dict[str, Any]) -> str:
    parts = ["+".join(cell["pair"]), cell["direction"], cell["symbol"], cell["start"],
             f"{cell['leverage']}x", f"s{cell['seed']}"]
    if cell.get("eval_windows", 1) != 1:
        parts.append(f"to{cell['end']}|ew{cell['eval_windows']}")
    if cell.get("tier", "nt") != "nt":
        parts.append(cell["tier"])
    if cell.get("timeframe", "1m") != "1m":
        parts.append(cell["timeframe"])
    return "|".join(parts)


def sample_cells(
    n: int, sample_seed: int, seeds: int = 1, grid: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """The first ``n`` cells of a fixed shuffle of the grid (prefix-stable in ``n``)."""
    cells = all_cells(seeds, grid)
    random.Random(sample_seed).shuffle(cells)
    return cells[:n]


def with_budget(
    cells: list[dict[str, Any]], pop: int, gens: int, tier: str = "nt"
) -> list[dict[str, Any]]:
    """Cells carrying the GA budget (ledger record) and the evaluator tier (record + key)."""
    return [{**c, "pop": pop, "gens": gens, "tier": tier} for c in cells]


def summarize(evals: list[dict[str, Any]]) -> dict[str, Any]:
    """Signal-density summary of one cell's distinct evaluations."""
    ok = [e for e in evals if not e.get("error")]
    active = [e for e in ok if e["trades"] >= MIN_TRADES]
    wins = [e for e in active if e["ret"] > 0]
    best = max(wins, key=lambda e: e["sharpe"]) if wins else None
    return {
        "n": len(evals),
        "n_err": len(evals) - len(ok),
        "n_low": sum(1 for e in ok if e["trades"] < MIN_TRADES),
        "n_active": len(active),
        "n_win": len(wins),
        "median_trades": statistics.median(e["trades"] for e in ok) if ok else None,
        "best": best,
    }


def _discovery_argv(cell: dict[str, Any], run_id: int, db: Path) -> list[str]:
    pop, gens = str(cell.get("pop", 6)), str(cell.get("gens", 1))
    return [
        "discovery",
        "--run-id", str(run_id),
        "--db", str(db),
        "--symbols", cell["symbol"],
        "--timeframe", cell.get("timeframe", "1m"),
        "--start-date", cell["start"],
        "--end-date", cell["end"],
        "--population-size", pop,
        "--max-generations", gens,
        "--convergence-generations", gens,
        "--eval-windows", str(cell.get("eval_windows", 1)),
        "--train-test-split", "0",
        "--direction", cell["direction"],
        "--indicator-pool", ",".join(cell["pair"]),
        "--max-workers", "1",
        "--seed", str(cell["seed"]),
        "--no-bootstrap-ci", "--no-dsr", "--no-buy-hold-gate",
        "--liquidation-mode", "isolated",
        "--leverage", str(cell["leverage"]),
        "--max-margin-pct", "0.5",
    ]


def _missing_data_reason(symbol: str, timeframe: str) -> str | None:
    """Why the catalog cannot serve this cell's bars, or ``None`` when it can."""
    from vibe_quant.data.catalog import CatalogManager

    if CatalogManager(REPO_ROOT / "data" / "catalog").get_bar_date_range(symbol, timeframe) is None:
        return f"no {timeframe} bars for {symbol} in data/catalog"
    return None


def run_cell(cell: dict[str, Any], result_path: Path) -> int:
    """Run one cell in THIS process (VIBE_QUANT_DB must already point at a scratch DB)."""
    db = Path(os.environ["VIBE_QUANT_DB"])
    if db.resolve() == (REPO_ROOT / "data" / "state" / "vibe_quant.db").resolve():
        raise SystemExit("burst cell refuses the real state DB")
    timeframe = cell.get("timeframe", "1m")
    missing = _missing_data_reason(cell["symbol"], timeframe)
    if missing is not None:  # a data gap is a failed cell, not a dead run
        result_path.write_text(json.dumps({"failed": True, "reason": missing}))
        return 0
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery.backtest_fn import NTBacktestFn
    from vibe_quant.discovery.genome import chromosome_to_serializable
    from vibe_quant.fast_tier import fallbacks

    fallbacks.clear()  # a cell reports its own fallbacks, never an earlier cell's
    state = StateManager(db)
    try:
        run_id = state.create_backtest_run(
            strategy_id=None, run_mode="discovery", symbols=[cell["symbol"]],
            timeframe=timeframe, start_date=cell["start"], end_date=cell["end"], parameters={},
        )
    finally:
        state.close()

    evals: dict[str, dict[str, Any]] = {}
    original = NTBacktestFn.__call__

    def _logged(self: Any, chromosome: Any) -> Any:
        res = original(self, chromosome)
        evals[repr(chromosome)] = {
            "trades": int(res.get("total_trades", 0)),
            "ret": float(res.get("total_return", 0.0)),
            "sharpe": float(res.get("sharpe_ratio", 0.0)),
            "pf": float(res.get("profit_factor", 0.0)),
            "dd": float(res.get("max_drawdown", 0.0)),
            "error": res.get("error"),
            "genome": repr(chromosome)[:600],
            # Reloadable twin of ``genome``: the repr cannot be parsed back, so a
            # transfer re-evaluation needs the serialized genome (vibe-quant-8k69u).
            "chromosome": chromosome_to_serializable(chromosome),
        }
        return res

    NTBacktestFn.__call__ = _logged  # type: ignore[method-assign]
    sys.argv = _discovery_argv(cell, run_id, db)
    import vibe_quant.discovery.__main__ as discovery_cli
    import vibe_quant.discovery.run as discovery_run

    if cell.get("tier", "nt") == "fast":
        # the fast tier replaces only the engine call; its parity with NT screening is
        # checked by scripts/agents/fast_tier_parity.py
        from vibe_quant.fast_tier.backtest_fn import FastBacktestFn

        class _BurstFast(FastBacktestFn):
            confirm_on_nt = False  # scratch-DB exploration: never a verdict, no NT re-run

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                # A genome the fast tier cannot replay (BTC_ROC, FUNDING_Z: outside series)
                # runs on NT, as in a real --evaluator fast discovery; without it every such
                # genome counted as an error and the cell read "no winner" (2026-10-26).
                kwargs.setdefault("nt_fallback", True)
                super().__init__(*args, **kwargs)

        setattr(discovery_run, "NTBacktestFn", _BurstFast)  # noqa: B010
    discovery_main = discovery_cli.main

    t0 = time.perf_counter()
    try:
        rc = discovery_main()
    finally:
        NTBacktestFn.__call__ = original  # type: ignore[method-assign]
    out = {
        "rc": rc,
        "wall_s": round(time.perf_counter() - t0, 1),
        "nt_fallbacks": fallbacks.summary(),
        **summarize(list(evals.values())),
    }
    result_path.write_text(json.dumps(out))
    return 0


def _launch(cell: dict[str, Any]) -> tuple[subprocess.Popen[bytes], Path, float]:
    tmp = Path(tempfile.mkdtemp(prefix="burst-", dir="/tmp"))
    env = {
        **os.environ,
        "VIBE_QUANT_DB": str(tmp / "state.db"),
        "VIBE_QUANT_LOG_DIR": str(tmp / "logs"),
        "PYTHONPATH": str(REPO_ROOT),
    }
    with (tmp / "stderr.log").open("wb") as err:
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, __file__, "cell", json.dumps(cell), str(tmp / "result.json")],
            cwd=REPO_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=err,
        )
    return proc, tmp, time.monotonic()


def _finish(cell: dict[str, Any], tmp: Path, rc: int | None, wall: float) -> dict[str, Any]:
    row: dict[str, Any] = {"key": cell_key(cell), "cell": cell}
    result = tmp / "result.json"
    if rc == 0 and result.exists():
        row.update(json.loads(result.read_text()))
    else:
        tail = (tmp / "stderr.log").read_text(errors="replace").splitlines()[-15:]
        row.update({"rc": rc, "wall_s": round(wall, 1), "failed": True, "stderr_tail": tail})
    shutil.rmtree(tmp, ignore_errors=True)
    return row


def load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def grid_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """The default grid with every factor given on the command line replaced."""
    grid = default_grid()
    if args.pairs:
        grid["pairs"] = tuple(tuple(p.split("+")) for p in args.pairs.split(","))
    if args.directions:
        grid["directions"] = tuple(args.directions.split(","))
    if args.symbols:
        grid["symbols"] = tuple(args.symbols.split(","))
    if args.windows:
        grid["windows"] = tuple(tuple(w.split(":")) for w in args.windows.split(","))
    if args.leverages:
        grid["leverages"] = tuple(int(x) for x in args.leverages.split(","))
    if args.timeframes:
        grid["timeframes"] = tuple(args.timeframes.split(","))
    grid["eval_windows"] = args.eval_windows
    return grid


def run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ledger = out / "ledger.jsonl"
    done = {row["key"] for row in load_ledger(ledger)}
    cells = with_budget(
        sample_cells(args.cells, args.sample_seed, args.seeds, grid_from_args(args)),
        args.pop, args.gens, args.tier,
    )
    if args.tier == "fast":
        # the fast tier replays 1m only; refuse instead of silently running NT or 1m
        higher = sorted({c.get("timeframe", "1m") for c in cells} - {"1m"})
        if higher:
            raise SystemExit(
                f"burst: the fast tier is 1m-only; {', '.join(higher)} needs --tier nt"
            )
    todo = [c for c in cells if cell_key(c) not in done]
    deadline = time.monotonic() + args.budget_s
    running: list[tuple[dict[str, Any], subprocess.Popen[bytes], Path, float]] = []
    print(f"burst: {len(todo)} cells to run ({len(done)} already in ledger)", flush=True)
    with ledger.open("a") as fh:
        while todo or running:
            while todo and len(running) < args.parallel and time.monotonic() < deadline:
                cell = todo.pop(0)
                running.append((cell, *_launch(cell)))
            if not running:
                break
            for item in list(running):
                cell, proc, tmp, t0 = item
                wall = time.monotonic() - t0
                if proc.poll() is None and wall < CELL_TIMEOUT_S:
                    continue
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                row = _finish(cell, tmp, proc.returncode, wall)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                running.remove(item)
                best = row.get("best") or {}
                print(
                    f"{row['key']} wall={row.get('wall_s')} n={row.get('n')} win={row.get('n_win')} "
                    f"low={row.get('n_low')} best_sharpe={best.get('sharpe')} failed={row.get('failed', False)}",
                    flush=True,
                )
            time.sleep(0.5)
    print(f"burst: stopped with {len(todo)} cells not started (budget)", flush=True)
    return 0


def _month_bh(closes: dict[str, dict[int, float]], symbol: str, start: str, end: str) -> float | None:
    from datetime import date

    def day(s: str) -> int:
        return (date.fromisoformat(s) - date(1970, 1, 1)).days

    series = closes.get(symbol, {})
    first, last = series.get(day(start) - 1), series.get(day(end) - 1)  # end is exclusive
    return None if first is None or last is None else last / first - 1.0


def _fmt_rets(rets: list[float | None]) -> str:
    return "[" + ",".join("n/a" if x is None else f"{x:+.4f}" for x in rets) + "]"


def _add_months(d: date, months: int) -> date:
    """``d`` moved ``months`` calendar months forward, clamped to the month's last day."""
    import calendar
    from datetime import date

    year, month0 = divmod(d.month - 1 + months, 12)
    year, month = d.year + year, month0 + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def next_window(cell: dict[str, Any]) -> dict[str, str] | None:
    """The window right after the cell's, same length, clipped to the catalog end.

    ``None`` when the cell already ends at the last full day (2026-10-19), so a
    transfer has no next window to test.
    """
    from datetime import date

    start_date, end_date = date.fromisoformat(cell["start"]), date.fromisoformat(cell["end"])
    cap = date.fromisoformat(TRANSFER_CLIP_END)
    if end_date >= cap:
        return None
    if start_date.day == end_date.day:  # whole-month window: step calendar months
        months = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month
        end = _add_months(end_date, max(1, months))
    else:
        end = end_date + (end_date - start_date)
    return {"start": end_date.isoformat(), "end": min(end, cap).isoformat()}


def winner_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rows ``report`` calls a winner: a cell with a best genome."""
    return [r for r in rows if not r.get("failed") and r.get("best")]


def transfer_jobs(
    winners: list[dict[str, Any]], symbols: list[str]
) -> list[dict[str, Any]]:
    """One transfer job per winner: other symbols (same window) + next window."""
    return [
        {
            "key": row["key"],
            "cell": row["cell"],
            "chromosome": row["best"]["chromosome"],
            "symbols": [s for s in symbols if s != row["cell"]["symbol"]],
            "next": next_window(row["cell"]),
        }
        for row in winners
    ]


def _transfer_backtest(cell: dict[str, Any], symbol: str, start: str, end: str) -> Any:
    """The cell's own evaluation settings, aimed at one symbol and one window."""
    from vibe_quant.discovery.backtest_fn import NTBacktestFn
    from vibe_quant.fees import FeeSchedule
    from vibe_quant.liquidation_settings import LiquidationSettings
    from vibe_quant.sizing_settings import SizingSettings
    from vibe_quant.utils import split_into_windows

    n = max(1, int(cell.get("eval_windows", 1)))
    kwargs: dict[str, Any] = {
        "symbols": [symbol],
        "timeframe": cell.get("timeframe", "1m"),
        "start_date": start,
        "end_date": end,
        "windows": split_into_windows(start, end, n) if n > 1 else None,
        "min_trades": MIN_TRADES,
        "symbol_agg": "portfolio",
        "fee_schedule": FeeSchedule(),
        "entry_order": "market",
        "liquidation": LiquidationSettings(mode="isolated", leverage=int(cell["leverage"])),
        "sizing": SizingSettings.from_params({"max_margin_pct": 0.5}),
    }
    if cell.get("tier", "nt") == "fast":
        from vibe_quant.fast_tier.backtest_fn import FastBacktestFn

        fn = FastBacktestFn(**kwargs, nt_fallback=True)
        fn.confirm_on_nt = False  # scratch exploration: never a verdict, no NT re-run
        return fn
    return NTBacktestFn(**kwargs)


def _eval_metrics(res: dict[str, Any]) -> dict[str, Any]:
    return {
        "ret": float(res.get("total_return", 0.0)),
        "trades": int(res.get("total_trades", 0)),
        "sharpe": float(res.get("sharpe_ratio", 0.0)),
        "error": res.get("error"),
    }


def _transfer_eval(fn: Any, chromosome: Any) -> dict[str, Any]:
    try:
        return _eval_metrics(fn(chromosome))
    except Exception as exc:  # a data gap is a failed transfer, not a dead job
        return {"ret": 0.0, "trades": 0, "sharpe": 0.0, "error": f"{type(exc).__name__}: {exc}"}


def transfer_one(spec: dict[str, Any], result_path: Path) -> int:
    """Re-evaluate one winner's genome on the other symbols and the next window."""
    from vibe_quant.discovery.genome import serializable_to_chromosome
    from vibe_quant.fast_tier import fallbacks

    fallbacks.clear()  # a row reports its own evaluations' fallbacks
    chromosome = serializable_to_chromosome(spec["chromosome"])
    cell = spec["cell"]
    row: dict[str, Any] = {"key": spec["key"], "symbols": {}, "next": None}
    for symbol in spec["symbols"]:
        fn = _transfer_backtest(cell, symbol, cell["start"], cell["end"])
        row["symbols"][symbol] = _transfer_eval(fn, chromosome)
    nxt = spec.get("next")
    if nxt:
        fn = _transfer_backtest(cell, cell["symbol"], nxt["start"], nxt["end"])
        row["next"] = {"start": nxt["start"], "end": nxt["end"], **_transfer_eval(fn, chromosome)}
    row["nt_fallbacks"] = fallbacks.summary()
    result_path.write_text(json.dumps(row))
    return 0


def _launch_transfer(spec: dict[str, Any]) -> tuple[subprocess.Popen[bytes], Path, float]:
    tmp = Path(tempfile.mkdtemp(prefix="burst-xfer-", dir="/tmp"))
    env = {
        **os.environ,
        "VIBE_QUANT_DB": str(tmp / "state.db"),
        "VIBE_QUANT_LOG_DIR": str(tmp / "logs"),
        "PYTHONPATH": str(REPO_ROOT),
    }
    with (tmp / "stderr.log").open("wb") as err:
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, __file__, "transfer-one", json.dumps(spec), str(tmp / "result.json")],
            cwd=REPO_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=err,
        )
    return proc, tmp, time.monotonic()


def _finish_transfer(
    spec: dict[str, Any], tmp: Path, rc: int | None, wall: float
) -> dict[str, Any]:
    row: dict[str, Any] = {"key": spec["key"], "symbols": {}, "next": None}
    result = tmp / "result.json"
    if rc == 0 and result.exists():
        row.update(json.loads(result.read_text()))
    else:
        tail = (tmp / "stderr.log").read_text(errors="replace").splitlines()[-15:]
        row.update({"rc": rc, "wall_s": round(wall, 1), "failed": True, "stderr_tail": tail})
    shutil.rmtree(tmp, ignore_errors=True)
    return row


def _wins(m: dict[str, Any]) -> bool:
    """One transfer eval passes by the cell's own winner rule: no error, enough trades, ret > 0."""
    return (
        not m.get("error")
        and (m.get("trades") or 0) >= MIN_TRADES
        and (m.get("ret") or 0) > 0
    )


def genome_transfers(row: dict[str, Any]) -> bool:
    """True when the genome wins (``_wins``) on every OTHER symbol AND the next window.

    A winner with no other-symbol evaluation is not transfer-tested: the metric
    exists to measure cross-symbol robustness, so an empty set proves nothing.
    """
    symbols = row.get("symbols") or {}
    nxt = row.get("next")
    return (
        bool(symbols)
        and nxt is not None
        and all(_wins(m) for m in symbols.values())
        and _wins(nxt)
    )


def _positive_share(evals: list[dict[str, Any]]) -> str:
    if not evals:
        return "n/a"
    return f"{sum(1 for e in evals if _wins(e)) / len(evals):.3f}"


def _rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def family_of(cell: dict[str, Any]) -> str:
    """A family = indicator pair, per timeframe: a 1h RSI+CCI is not the 1m one."""
    pair = "+".join(cell["pair"])
    tf = cell.get("timeframe", "1m")
    return pair if tf == "1m" else f"{pair}@{tf}"


def _binom_upper(wins: int, trials: int, p: Fraction) -> float:
    """One-sided P(X >= wins) for X ~ Binomial(trials, p), in exact integer arithmetic."""
    a, b = p.numerator, p.denominator
    tail = sum(
        math.comb(trials, i) * a**i * (b - a) ** (trials - i)
        for i in range(max(wins, 0), trials + 1)
    )
    return float(Fraction(tail, b**trials))


def null_p(wins: int, trials: int, base: Fraction | None) -> float | None:
    """A family's share against the ledger-wide base rate; ``None`` when there is no base."""
    if base is None or trials <= 0:
        return None
    return _binom_upper(wins, trials, base)


def _p_token(p: float | None, alpha: float = 0.05) -> str:
    """The report cell for a p-value: ``n/a`` without a base, ``*`` when it clears ``alpha``."""
    if p is None:
        return "n/a"
    return f"{p:.4g}" + ("*" if p < alpha else "")


def transfer(args: argparse.Namespace) -> int:
    out = Path(args.out)
    ledger = load_ledger(out / "ledger.jsonl")
    symbols = sorted({r["cell"]["symbol"] for r in ledger if not r.get("failed")})
    winners = winner_rows(ledger)
    with_chromosome = [r for r in winners if (r["best"] or {}).get("chromosome")]
    skipped = len(winners) - len(with_chromosome)
    tpath = out / "transfer.jsonl"
    done = {row["key"] for row in load_ledger(tpath)}
    todo = [j for j in transfer_jobs(with_chromosome, symbols) if j["key"] not in done]
    print(
        f"transfer: {len(with_chromosome)} winners, {len(todo)} to evaluate "
        f"({len(done)} already in transfer.jsonl), skipped {skipped} without a "
        f"chromosome (old ledger)",
        flush=True,
    )
    deadline = time.monotonic() + args.budget_s
    running: list[tuple[dict[str, Any], subprocess.Popen[bytes], Path, float]] = []
    with tpath.open("a") as fh:
        while todo or running:
            while todo and len(running) < args.parallel and time.monotonic() < deadline:
                spec = todo.pop(0)
                running.append((spec, *_launch_transfer(spec)))
            if not running:
                break
            for item in list(running):
                spec, proc, tmp, t0 = item
                wall = time.monotonic() - t0
                if proc.poll() is None and wall < CELL_TIMEOUT_S:
                    continue
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                row = _finish_transfer(spec, tmp, proc.returncode, wall)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                running.remove(item)
                print(
                    f"{row['key']} symbols={sorted(row.get('symbols') or {})} "
                    f"next={row.get('next') is not None} transfers={genome_transfers(row)} "
                    f"failed={row.get('failed', False)}",
                    flush=True,
                )
            time.sleep(0.5)
    print(f"transfer: stopped with {len(todo)} winners not started (budget)", flush=True)
    return 0


def _fallbacks_line(rows: list[dict[str, Any]], label: str = "") -> str:
    """Fast-tier fallbacks over the rows that recorded them; old rows counted apart.

    A row written before the field existed proves nothing about fallbacks, so it is
    counted in ``rows_without_key`` instead of as zero (which would read as "no
    fallback happened" and hide a silent NT run). ``label`` names the row file
    (``cells`` or ``transfer``); no label is the combined total.
    """
    by_reason: dict[str, int] = {}
    with_key = 0
    for r in rows:
        summary = r.get("nt_fallbacks")
        if summary is None:
            continue
        with_key += 1
        for reason, count in summary.get("by_reason", {}).items():
            by_reason[reason] = by_reason.get(reason, 0) + int(count)
    prefix = f"{label} " if label else ""
    return (
        f"{prefix}fallbacks: total={sum(by_reason.values())}"
        f" by_reason={json.dumps(dict(sorted(by_reason.items())))}"
        f" rows_without_key={len(rows) - with_key}"
    )


def _wall_line(rows: list[dict[str, Any]]) -> str:
    """Median cell wall time (the cell result's ``wall_s``), so rounds compare."""
    walls = [float(r["wall_s"]) for r in rows if r.get("wall_s") is not None]
    median = f"{statistics.median(walls):.2f}s" if walls else "n/a"
    return (
        f"wall: median={median} cells={len(walls)}"
        f" rows_without_wall={len(rows) - len(walls)}"
    )


def report(args: argparse.Namespace) -> int:
    ledger = [r for r in load_ledger(Path(args.out) / "ledger.jsonl") if args.filter in r["key"]]
    rows = [r for r in ledger if not r.get("failed")]
    failed = [r for r in ledger if r.get("failed")]
    transfer_rows = [
        r for r in load_ledger(Path(args.out) / "transfer.jsonl") if args.filter in r["key"]
    ]
    from vibe_quant.discovery.buyhold import load_daily_closes
    from vibe_quant.utils import split_into_windows

    closes = load_daily_closes(sorted({r["cell"]["symbol"] for r in rows}), REPO_ROOT / "data" / "catalog")
    for r in rows:
        c = r["cell"]
        r["bh_ret"] = _month_bh(closes, c["symbol"], c["start"], c["end"])
        # worst-of-N cells score each sub-window, so compare with each sub-window's B&H
        n_sub = c.get("eval_windows", 1)
        subs = split_into_windows(c["start"], c["end"], n_sub) if n_sub > 1 else []
        r["bh_sub"] = [_month_bh(closes, c["symbol"], s, e) for s, e in subs]
    evals = sum(r["n"] for r in rows)
    print(f"cells={len(rows)} failed={len(failed)} distinct_evals={evals}")
    print(_fallbacks_line(rows, "cells"))
    print(_fallbacks_line(transfer_rows, "transfer"))
    print(_fallbacks_line(rows + transfer_rows))
    print(_wall_line(rows))
    families: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        families.setdefault(family_of(r["cell"]), []).append(r)
    tmap = {r["key"]: r for r in transfer_rows}
    # The null is the ledger-wide rate measured the same way as the family shares
    # (vibe-quant-rum8f): a family must beat what picking its best genome by luck
    # yields. With a single family there is no rate distinct from itself. Trials are
    # CELLS (cells with a winner / cells), not genomes: a cell's genomes are GA kin of
    # one another, so counting them as independent draws makes p far too small.
    # Several families are tested at once, so the flag is Bonferroni (0.05 / k).
    cells_won = sum(1 for r in rows if r["n_win"])
    win_base = Fraction(cells_won, len(rows)) if len(families) > 1 and rows else None
    alpha = 0.05 / max(len(families), 1)
    winners_all = winner_rows(rows)
    tested_all = [r for r in winners_all if r["key"] in tmap]
    xfer_wins = sum(1 for r in tested_all if genome_transfers(tmap[r["key"]]))
    tested_families = {family_of(r["cell"]) for r in tested_all}
    xfer_base = (
        Fraction(xfer_wins, len(winners_all))
        if len(tested_families) > 1 and winners_all
        else None
    )
    factors: dict[str, Callable[[dict[str, Any]], str]] = {
        "tier": lambda c: c.get("tier", "nt"),
        "pair": family_of,
        "timeframe": lambda c: c.get("timeframe", "1m"),
        "direction": lambda c: c["direction"],
        "symbol": lambda c: c["symbol"],
        "month": lambda c: c["start"][:7],
        "leverage": lambda c: f"{c['leverage']}x",
    }
    for name, fn in factors.items():
        if name == "pair":
            print(
                f"\n## null: one-sided binomial P(X >= wins | trials, p0) against the"
                f" ledger-wide rate; trials = cells (win) / winners (xfer);"
                f" p0 win={_rate(None if win_base is None else float(win_base))}"
                f" xfer={_rate(None if xfer_base is None else float(xfer_base))};"
                f" * = p < {alpha:.4g} (0.05 Bonferroni over {len(families)} families);"
                f" n/a = no base distinct from the family (one family only)"
            )
        print(
            f"\n## {name}: cells | win share (profitable >=50-trade genomes / all) |"
            " cells with a winner | median best sharpe"
            + (" | null p (cells with a winner vs p0)" if name == "pair" else "")
        )
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(fn(r["cell"]), []).append(r)
        for level, rs in sorted(groups.items()):
            n = sum(r["n"] for r in rs)
            win = sum(r["n_win"] for r in rs)
            with_win = [r for r in rs if r["n_win"]]
            med = statistics.median(r["best"]["sharpe"] for r in with_win) if with_win else None
            extra = (
                f" | {_p_token(null_p(len(with_win), len(rs), win_base), alpha)}"
                if name == "pair"
                else ""
            )
            print(f"{level:22s} {len(rs):4d} | {win / n if n else 0:.3f} | {len(with_win)}/{len(rs)} | {med}{extra}")
    print(
        "\n## transfer (family = indicator pair): winners | transfer-tested |"
        " xfer_sym (share of other-symbol evals with return > 0) |"
        " xfer_next (share of next-window evals with return > 0) |"
        " transfer share (return > 0 on ALL other symbols AND the next window / winners) | xfer p"
    )
    for family, rs in sorted(families.items()):
        winners = winner_rows(rs)
        tested = [r for r in winners if r["key"] in tmap]
        sym = [m for r in tested for m in (tmap[r["key"]].get("symbols") or {}).values()]
        nxt = [tmap[r["key"]]["next"] for r in tested if tmap[r["key"]].get("next")]
        transfers = sum(1 for r in tested if genome_transfers(tmap[r["key"]]))
        share = transfers / len(winners) if winners else None
        print(
            f"{family:22s} {len(winners):4d} | {len(tested):4d} | {_positive_share(sym)} |"
            f" {_positive_share(nxt)} | {_rate(share)} |"
            f" {_p_token(null_p(transfers, len(winners), xfer_base), alpha)}"
        )
    print(
        "\n## top cells by best sharpe (ret vs the symbol's buy-and-hold over the same window;"
        " worst-of-N cells: sharpe/ret are the worst sub-window's, bh_sub = each sub-window's B&H)"
    )
    top = sorted((r for r in rows if r["best"]), key=lambda r: r["best"]["sharpe"], reverse=True)[: args.top]
    for r in top:
        b = r["best"]
        bh = r["bh_ret"]
        print(
            f"{r['key']:55s} sharpe={b['sharpe']:.3f} ret={b['ret']:+.4f} trades={b['trades']} "
            f"pf={b['pf']:.3f} dd={b['dd']:.4f} bh={'n/a' if bh is None else f'{bh:+.4f}'}"
            + (f" bh_sub={_fmt_rets(r['bh_sub'])}" if r["bh_sub"] else "")
        )
    for r in failed:
        reason = r.get("reason") or (r.get("stderr_tail") or [""])[-1]
        print(f"FAILED {r['key']} rc={r.get('rc')}: {reason}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--cells", type=int, required=True)
    r.add_argument("--parallel", type=int, default=2)
    r.add_argument("--budget-s", type=int, default=1800, help="no new cell starts after this")
    r.add_argument("--sample-seed", type=int, default=1)
    r.add_argument("--seeds", type=int, default=1, help="GA seeds per grid point")
    r.add_argument("--pop", type=int, default=6)
    r.add_argument("--gens", type=int, default=1)
    r.add_argument("--tier", choices=("nt", "fast"), default="nt", help="fast = vibe_quant.fast_tier")
    r.add_argument("--pairs", default="", help="e.g. KAMA+NATR,ROC+PRICE_POSITION")
    r.add_argument("--directions", default="", help="e.g. both,long")
    r.add_argument("--symbols", default="", help="e.g. ZECUSDT,NEARUSDT")
    r.add_argument("--windows", default="", help="start:end[,start:end] (end exclusive)")
    r.add_argument("--leverages", default="", help="e.g. 10")
    r.add_argument("--timeframes", default="", help="e.g. 15m,1h (default 1m; 15m+ needs --tier nt)")
    r.add_argument("--eval-windows", type=int, default=1, help="worst-of-N sub-windows per cell")
    rep = sub.add_parser("report")
    rep.add_argument("--out", required=True)
    rep.add_argument("--top", type=int, default=15)
    rep.add_argument("--filter", default="", help="only ledger keys containing this, e.g. '|ew3|fast'")
    tr = sub.add_parser("transfer")
    tr.add_argument("--out", required=True, help="the run's ledger directory")
    tr.add_argument("--parallel", type=int, default=2)
    tr.add_argument("--budget-s", type=int, default=1800, help="no new winner starts after this")
    one = sub.add_parser("transfer-one")
    one.add_argument("spec")
    one.add_argument("result")
    c = sub.add_parser("cell")
    c.add_argument("spec")
    c.add_argument("result")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return run(args)
    if args.cmd == "report":
        return report(args)
    if args.cmd == "transfer":
        return transfer(args)
    if args.cmd == "transfer-one":
        return transfer_one(json.loads(args.spec), Path(args.result))
    return run_cell(json.loads(args.spec), Path(args.result))


if __name__ == "__main__":
    raise SystemExit(main())
