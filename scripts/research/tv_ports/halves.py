"""halves.py -- two TradingView-port settings on 2026 H1 vs H2, every timeframe, literal vs time-matched.

    run     --out DIR [--parallel N] [--tfs 1m,5m,...] [--only-coins A,B] --strategies A,B
        One screening backtest per (strategy, variant, timeframe, coin, half), each on a fresh
        balance, from a warm-up start (half start - max(2 days, 2 x lookback)) to the half's end
        (H2: the coin's data end, oos_analysis.coin_spans), with the daily
        mark-to-market series and positions captured (oos_analysis's capture runner = preset_grid's
        engine config: NTBacktestFn, all-in 1x, default fees, funding, isolated liquidation).
        Appended to DIR/results.jsonl; resumable; a failure is a row with ``error``.
    report  --out DIR
        Slice every run to its half, H1 = [2026-01-01, 2026-07-01) or H2 = [2026-07-01, end), by day and
        write DIR/report.md: Sharpe (oos_analysis.annual_sharpe = daily MTM mean / std(ddof=1)
        x sqrt(365)), compounded return, trades opened in the half, max DD of the daily MTM
        equity, plus buy-and-hold per coin per half.

DIR/settings.json names the settings (kept private). Values outside a port's param domain (e.g. a length above the port's bound) are rendered by plain placeholder
substitution + DSL validation, bypassing only the port's domain/constraint check; each run row
records ``in_domain``. Results are private: write DIR under data/research/ (gitignored).

Run from the repository root: ``PYTHONPATH=$PWD .venv/bin/python scripts/research/tv_ports/halves.py ...``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import oos_analysis as oa  # noqa: E402  (capture runner, slicing math)
import preset_grid  # noqa: E402  (engine config, ledger helpers)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

COINS = preset_grid.COINS
TFS = ("1m", "5m", "15m", "1h", "4h", "1d")
TF_MIN = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
H1_START, H2_START = "2026-01-01", "2026-07-01"
HALVES: dict[str, tuple[str, str | None]] = {"H1": (H1_START, H2_START), "H2": (H2_START, None)}  # None = data end
MIN_WARM_DAYS = 2

# Settings come from DIR/settings.json (private, gitignored): {name: {"port": folder, "values": {...},
# "scaled": [params whose bar count is rescaled to equal clock time on 1h]}}.
STRATEGIES: dict[str, dict[str, Any]] = {}


def time_matched(values: Mapping[str, Any], scaled: Sequence[str], tf: str) -> dict[str, Any]:
    """Each scaled length x 60 / tf-minutes, rounded half-up, minimum 2."""
    out = dict(values)
    for k in scaled:
        out[k] = max(2, math.floor(values[k] * 60 / TF_MIN[tf] + 0.5))
    return out


def lookback_bars(port: str, values: Mapping[str, Any], scaled: Sequence[str]) -> int:
    """Bars until every indicator is finite: HULL ports via tv_hull_ready + offset 2, else the longest SMA + 1."""
    if port.endswith("_hull"):
        from vibe_quant.dsl.plugins._tv_ma import tv_hull_ready

        return tv_hull_ready(values["mode"], int(values["length"])) + 2
    return max(int(values[k]) for k in scaled) + 1


def warm_start(half_start: str, tf: str, bars: int) -> str:
    """``half_start`` minus max(2 days, 2 x the lookback's clock time)."""
    days = max(MIN_WARM_DAYS, math.ceil(2 * bars * TF_MIN[tf] / 1440))
    return (date.fromisoformat(half_start) - timedelta(days=days)).isoformat()


def _genome(port: Any, values: Mapping[str, Any]) -> tuple[Any, bool]:
    """``(genome, in_domain)``; out-of-domain values render with the port's domain check skipped."""
    from vibe_quant.discovery.param_genome import ParamGenome
    from vibe_quant.discovery.port import _substitute_node
    from vibe_quant.dsl.parser import validate_strategy_dict

    if port.try_canonical_values(values) is not None:
        return ParamGenome(port, dict(values)), True

    class FreeGenome(ParamGenome):
        def to_dsl(self, timeframe: str | None = None) -> dict[str, object]:
            dsl: dict[str, object] = dict(_substitute_node(self.port.dsl_tree, self.values))  # type: ignore[call-overload]
            if timeframe is not None:
                dsl["timeframe"] = timeframe
            validate_strategy_dict(dsl)
            dsl["name"] = f"{self.port.name}_{self.uid}"
            return dsl

    return FreeGenome(port, dict(values)), False


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    """One backtest; never raises."""
    row: dict[str, Any] = {k: task[k] for k in ("key", "strategy", "variant", "tf", "coin", "half", "start", "end", "values", "port")}
    row.update({"kind": "run", "error": None})
    t0 = time.perf_counter()
    try:
        os.chdir(oa.REPO_ROOT)
        from vibe_quant.discovery.port import load_port

        port = load_port(HERE / task["port"])
        genome, row["in_domain"] = _genome(port, task["values"])
        fn = oa._capturing_backtest_fn(task["coin"], task["tf"], task["start"], task["end"])
        oa._CAPTURED.clear()
        res = fn(genome)
        series = res.get("mtm_daily_returns")
        row.update({
            "trades_total": int(res.get("total_trades", 0)),
            "fees": float(res.get("total_fees", 0.0)),
            "first_day": None if series is None else int(series.first_day),
            "daily": None if series is None else [float(v) for v in series.values],
            "positions": list(oa._CAPTURED),
            "error": res.get("error"),
        })
        if series is None and not row["error"]:
            row["error"] = "no daily MTM series"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["secs"] = round(time.perf_counter() - t0, 1)
    return row


def plan(tfs: Sequence[str], coins: Sequence[str], strategies: Sequence[str]) -> list[dict[str, Any]]:
    from vibe_quant.discovery.buyhold import load_daily_closes

    spans = oa.coin_spans(coins)
    closes = load_daily_closes(list(coins), oa.CATALOG)
    tasks: list[dict[str, Any]] = []
    for name in strategies:
        spec = STRATEGIES[name]
        for tf in tfs:
            variants = {"literal": dict(spec["values"]), "matched": time_matched(spec["values"], spec["scaled"], tf)}
            if tf == "1h":
                variants = {"literal": variants["literal"]}  # identical by construction; shared
            for variant, values in variants.items():
                for half, (a, b) in HALVES.items():
                    ws = warm_start(a, tf, lookback_bars(spec["port"], values, spec["scaled"]))
                    for coin in coins:
                        first_ok = date.fromordinal(oa.EPOCH.toordinal() + min(closes[coin]) + 1).isoformat()
                        tasks.append({
                            "key": f"run|{name}|{variant}|{tf}|{coin}|{half}", "strategy": name,
                            "variant": variant, "tf": tf, "coin": coin, "half": half,
                            "start": max(ws, first_ok), "end": b or spans[coin][1],
                            "values": values, "port": spec["port"],
                        })
    return tasks


def cmd_run(args: argparse.Namespace) -> int:
    os.chdir(oa.REPO_ROOT)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    present = preset_grid.done_keys(results, retry_errors=True)
    coins = args.only_coins.split(",") if args.only_coins else list(COINS)
    tasks = [t for t in plan(args.tfs.split(","), coins, args.strategies.split(",")) if t["key"] not in present]
    tasks.sort(key=lambda t: TF_MIN[t["tf"]])  # slowest (1m) first
    t0 = time.perf_counter()
    print(f"{len(tasks)} backtests to run", flush=True)
    import multiprocessing

    with multiprocessing.get_context("spawn").Pool(processes=max(1, args.parallel), maxtasksperchild=4) as pool:
        for i, row in enumerate(pool.imap_unordered(run_task, tasks), 1):
            preset_grid._append(results, row)
            print(f"  {i}/{len(tasks)} {row['key']} {row['secs']}s err={row['error']}", flush=True)
    print(f"wall {time.perf_counter() - t0:.0f}s for {len(tasks)} runs", flush=True)
    return 0


# --------------------------------------------------------------------------- report


def half_stats(row: Mapping[str, Any], a: str, b: str) -> dict[str, Any]:
    """Return / Sharpe / max DD of the daily MTM slice [a, b) and positions opened in it."""
    first = int(row["first_day"])
    da, db = oa.day_of(a), oa.day_of(b)
    vals = [v for i, v in enumerate(row["daily"]) if da <= first + i < db]
    lo, hi = da * oa.DAY_NS, db * oa.DAY_NS
    opened = sum(1 for p in row["positions"] if lo <= p["ts_open"] < hi)
    carried = any(p["ts_open"] < lo and (p["ts_close"] is None or p["ts_close"] > lo) for p in row["positions"])
    return {"ret": oa.compound(vals), "sharpe": oa.annual_sharpe(vals) if len(vals) > 1 else float("nan"),
            "dd": oa.max_drawdown(vals), "trades": opened, "days": len(vals), "carried_in": carried}


def _scale(v: float | None, k: float) -> float | None:
    return None if v is None else v * k


def _f(v: float | None, fmt: str) -> str:
    return "n/a" if v is None or not math.isfinite(v) else format(v, fmt)


def cmd_report(args: argparse.Namespace) -> int:
    from vibe_quant.discovery.buyhold import load_daily_closes

    out = Path(args.out)
    rows = [r for r in preset_grid.read_rows(out / "results.jsonl") if r.get("kind") == "run"]
    by = {(r["strategy"], r["variant"], r["tf"], r["coin"], r["half"]): r for r in rows}
    ends = {r["coin"]: r["end"] for r in rows if r["half"] == "H2"}
    end_all = sorted(set(ends.values()))
    halves = HALVES
    stats: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for key, r in by.items():
        if not r.get("error"):
            a, b = halves[key[4]]
            stats[key] = half_stats(r, a, b or r["end"])
    closes = load_daily_closes(list(COINS), oa.CATALOG)
    bh = {(c, h): preset_grid.bh_return(closes[c], date.fromisoformat(a), date.fromisoformat(b or ends.get(c, end_all[-1])))
          for c in COINS for h, (a, b) in halves.items()}
    L = ["# TV ports: 2026 H1 vs H2", "",
         f"H1 = [{H1_START}, {H2_START}), H2 = [{H2_START}, end) with end (exclusive) per coin: "
         + ", ".join(f"{c[:-4]} {e}" for c, e in sorted(ends.items())), "",
         "Sharpe = oos_analysis.annual_sharpe (daily MTM mean / std ddof=1 x sqrt(365)); return = compounded "
         "daily MTM; trades = positions opened in the half; DD = max drawdown of daily MTM equity. "
         "1h literal == 1h time-matched (one run, shown in both).", ""]
    errs = [r for r in rows if r.get("error")]
    reasons: dict[tuple[str, str, str], set[str]] = {}
    for r in errs:
        reasons.setdefault((r["strategy"], r["variant"], r["tf"]), set()).add(r["error"].replace("\n", " "))
    for (s, v, tf), why in sorted(reasons.items()):
        L.append(f"- n/a {s} {v} {tf}(engine refused): {' / '.join(sorted(why))}")
    metrics = [("sharpe", "Sharpe", ".2f", 1.0), ("ret", "Return %", "+.1f", 100.0),
               ("trades", "Trades", "d", 1.0), ("dd", "Max DD %", ".1f", 100.0)]
    for name in STRATEGIES:
        for variant in ("literal", "matched"):
            L += ["", f"## {name} {variant}", ""]
            for m, title, fmt, k in metrics:
                L += [f"### {title}", "",
                      "| tf | params | " + " | ".join(f"{c[:-4]} H1 | {c[:-4]} H2" for c in COINS) + " | mean H1 | mean H2 |",
                      "|---|---|" + "---|" * (2 * len(COINS) + 2)]
                for tf in TFS:
                    v = "literal" if tf == "1h" else variant
                    run = next((r for kk, r in by.items() if kk[:3] == (name, v, tf)), None)
                    params = "n/a" if run is None else ",".join(str(run["values"][p]) for p in STRATEGIES[name]["scaled"])
                    cells: list[str] = []
                    means: dict[str, float | None] = {}
                    for h in halves:
                        col = [stats[(name, v, tf, c, h)][m] for c in COINS if (name, v, tf, c, h) in stats]
                        fin = [x for x in col if isinstance(x, (int, float)) and math.isfinite(x)]
                        means[h] = statistics.fmean(fin) if fin else None
                    for c in COINS:
                        for h in halves:
                            cell = stats.get((name, v, tf, c, h))
                            val: float | None = None if cell is None else cell[m]
                            cells.append(_f(_scale(val, k), fmt) if m != "trades" else ("n/a" if val is None else str(val)))
                    mean_cells = [_f(_scale(means[h], k), ".2f" if m == "sharpe" else ".1f") for h in halves]
                    L.append(f"| {tf} | {params} | " + " | ".join(cells) + " | " + " | ".join(mean_cells) + " |")
                L.append("")
    L += ["## Buy-and-hold % per coin per half (context only)", "",
          "| coin | H1 | H2 |", "|---|---|---|"]
    for c in COINS:
        L.append(f"| {c[:-4]} | {_f(_scale(bh[(c, 'H1')], 100), '+.1f')} | {_f(_scale(bh[(c, 'H2')], 100), '+.1f')} |")
    L += ["", "## Runs", "",
          "| strategy | variant | tf | run start (warm-up from) | max secs per run | sum secs | in_domain |",
          "|---|---|---|---|---|---|---|"]
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["strategy"], r["variant"], r["tf"], r["half"]), []).append(r)
    for (s, v, tf, h), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], TF_MIN[kv[0][2]], kv[0][3])):
        L.append(f"| {s} | {v} | {tf} {h} | {sorted({r['start'] for r in rs})} | {max(r['secs'] for r in rs)} "
                 f"| {sum(r['secs'] for r in rs):.0f} | {sorted({str(r.get('in_domain')) for r in rs})} |")
    text = "\n".join(L) + "\n"
    (out / "report.md").write_text(text)
    (out / "stats.json").write_text(json.dumps(
        [{"strategy": k[0], "variant": k[1], "tf": k[2], "coin": k[3], "half": k[4], **v} for k, v in stats.items()]
        + [{"bh": True, "coin": c, "half": h, "ret": v} for (c, h), v in bh.items()], indent=0) + "\n")
    print(text)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--out", required=True)
    run.add_argument("--parallel", type=int, default=1)
    run.add_argument("--tfs", default=",".join(TFS))
    run.add_argument("--only-coins", default="")
    run.add_argument("--strategies", required=True, help="comma list of settings.json names")
    run.set_defaults(fn=cmd_run)
    rep = sub.add_parser("report")
    rep.add_argument("--out", required=True)
    rep.set_defaults(fn=cmd_report)
    args = p.parse_args(argv)
    STRATEGIES.update(json.loads((Path(args.out) / "settings.json").read_text()))
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
