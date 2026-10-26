"""Burst discovery: many ~30 s seeded discoveries on a scratch DB to map where 1m signal lives.

One CELL = one tiny discovery (default pop 6 x 1 gen, 1 eval window, no holdout, no gates)
for one indicator pair x direction x symbol x month x leverage. ``run`` samples cells
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
    PYTHONPATH=$PWD .venv/bin/python scripts/agents/burst.py report --out data/swarm/burst-20261020
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

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
MIN_TRADES = 50  # fitness.MIN_TRADES_1M
CELL_TIMEOUT_S = 600
# MEASURED 2026-10-20 (load 14/10 cores): BTCUSDT 1m, one month = ~22 s per backtest.


def default_grid() -> dict[str, Any]:
    """The grid factors (each overridable on the ``run`` command line)."""
    return {
        "pairs": PAIRS, "directions": DIRECTIONS, "symbols": SYMBOLS,
        "windows": MONTHS, "leverages": LEVERAGES, "eval_windows": 1,
    }


def all_cells(seeds: int = 1, grid: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Every grid cell, in factorial order."""
    g = default_grid() if grid is None else grid
    return [
        {
            "pair": list(pair), "direction": d, "symbol": s,
            "start": m[0], "end": m[1], "leverage": lev, "seed": seed,
            "eval_windows": g["eval_windows"],
        }
        for pair, d, s, m, lev, seed in itertools.product(
            g["pairs"], g["directions"], g["symbols"], g["windows"], g["leverages"],
            range(1, seeds + 1),
        )
    ]


def cell_key(cell: dict[str, Any]) -> str:
    parts = ["+".join(cell["pair"]), cell["direction"], cell["symbol"], cell["start"],
             f"{cell['leverage']}x", f"s{cell['seed']}"]
    if cell.get("eval_windows", 1) != 1:
        parts.append(f"to{cell['end']}|ew{cell['eval_windows']}")
    if cell.get("tier", "nt") != "nt":
        parts.append(cell["tier"])
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
        "--timeframe", "1m",
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


def run_cell(cell: dict[str, Any], result_path: Path) -> int:
    """Run one cell in THIS process (VIBE_QUANT_DB must already point at a scratch DB)."""
    db = Path(os.environ["VIBE_QUANT_DB"])
    if db.resolve() == (REPO_ROOT / "data" / "state" / "vibe_quant.db").resolve():
        raise SystemExit("burst cell refuses the real state DB")
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery.backtest_fn import NTBacktestFn

    state = StateManager(db)
    try:
        run_id = state.create_backtest_run(
            strategy_id=None, run_mode="discovery", symbols=[cell["symbol"]],
            timeframe="1m", start_date=cell["start"], end_date=cell["end"], parameters={},
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
    out = {"rc": rc, "wall_s": round(time.perf_counter() - t0, 1), **summarize(list(evals.values()))}
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


def report(args: argparse.Namespace) -> int:
    ledger = [r for r in load_ledger(Path(args.out) / "ledger.jsonl") if args.filter in r["key"]]
    rows = [r for r in ledger if not r.get("failed")]
    failed = [r for r in ledger if r.get("failed")]
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
    factors: dict[str, Callable[[dict[str, Any]], str]] = {
        "tier": lambda c: c.get("tier", "nt"),
        "pair": lambda c: "+".join(c["pair"]),
        "direction": lambda c: c["direction"],
        "symbol": lambda c: c["symbol"],
        "month": lambda c: c["start"][:7],
        "leverage": lambda c: f"{c['leverage']}x",
    }
    for name, fn in factors.items():
        print(f"\n## {name}: cells | win share (profitable >=50-trade genomes / all) | cells with a winner | median best sharpe")
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(fn(r["cell"]), []).append(r)
        for level, rs in sorted(groups.items()):
            n = sum(r["n"] for r in rs)
            win = sum(r["n_win"] for r in rs)
            with_win = [r for r in rs if r["n_win"]]
            med = statistics.median(r["best"]["sharpe"] for r in with_win) if with_win else None
            print(f"{level:22s} {len(rs):4d} | {win / n if n else 0:.3f} | {len(with_win)}/{len(rs)} | {med}")
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
        print(f"FAILED {r['key']} rc={r.get('rc')}: {(r.get('stderr_tail') or [''])[-1]}")
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
    r.add_argument("--eval-windows", type=int, default=1, help="worst-of-N sub-windows per cell")
    rep = sub.add_parser("report")
    rep.add_argument("--out", required=True)
    rep.add_argument("--top", type=int, default=15)
    rep.add_argument("--filter", default="", help="only ledger keys containing this, e.g. '|ew3|fast'")
    c = sub.add_parser("cell")
    c.add_argument("spec")
    c.add_argument("result")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return run(args)
    if args.cmd == "report":
        return report(args)
    return run_cell(json.loads(args.spec), Path(args.result))


if __name__ == "__main__":
    raise SystemExit(main())
