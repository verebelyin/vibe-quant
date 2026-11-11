"""keepers.py -- collect single-coin TradingView-port keepers into a PRIVATE folder.

The GitHub repo is public: this script embeds no setting or coin. It reads the gitignored
ledgers of job ``data/swarm/20261108-tv-ports`` at run time and writes only to ``--out``
(must be gitignored, e.g. ``data/research/tv-keepers``) and the local state DB (via the API).

    collect  --out DIR [--parallel N]
        Apply the keeper bar per (setting, timeframe, coin) and write DIR/keepers.jsonl,
        DIR/checked.jsonl (every pair judged) and DIR/README.md. Round-2 test windows that can
        pass are re-run with the trade list captured (PF needs it; DIR/reruns.jsonl, resumable).
    render   --out DIR
        DIR/yaml/<name>.yaml per keeper via ``tv.py render``.
    persist  --out DIR [--api URL] [--limit 150] [--parallel 4]
        Per keeper (best unseen-window Sharpe first, up to --limit): register the strategy and
        launch a validation-tier run over the coin's full history through the backend API, add
        the user note, and run the same window at screening tier in-process (DIR/db_runs.jsonl,
        resumable).
    summary  --out DIR
        DIR/summary.md: validation vs screening per persisted keeper.

Keeper bar (chief, 2026-11-10), per (setting, timeframe, coin):
  * search window (the one the setting was chosen or searched on): return > 0, >= 30 trades;
  * unseen window (no part in picking it): return > 0, >= 30 trades, PF >= 1.2, max DD <= 50%.
  Round 2 presets: search 2023-01-01..2026-01-01, unseen 2026-01-01..2026-10-20 (fresh runs).
  Round-1 GA settings: search 2026-01-01..2026-08-01, unseen 2023-01-01..2026-01-01 (slices of
  the oos-analysis continuous run). Forward 2026-10-20.. is recorded when it exists, no bar.

Run from the repository root: ``PYTHONPATH=$PWD .venv/bin/python scripts/research/tv_ports/keepers.py ...``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import oos_analysis as oa  # noqa: E402  (sibling script: window math, engine capture, spans)
import preset_grid  # noqa: E402

SWARM = oa.SWARM
REPO_ROOT = oa.REPO_ROOT
MIN_TRADES = 30
MIN_PF = 1.2
MAX_DD = 0.5
# re-run a round-2 test window only when the ledger leaves it a chance to pass (slack for the
# ledger's engine DD vs the daily-MTM DD the bar uses)
PRE_MIN_TRADES = 25
PRE_MAX_DD = 0.6
R1_SEARCH = ("2026-01-01", "2026-08-01")
R1_UNSEEN = ("2023-01-01", "2026-01-01")
R2_SEARCH = preset_grid.WINDOWS["search"]
R2_UNSEEN = preset_grid.WINDOWS["test"]
SIZING = {"leverage": 1, "max_position_pct": 1.0, "risk_per_trade": 0.5, "max_margin_pct": 1.0}
NOTE = "single-coin keeper, not evidence"

Row = Mapping[str, Any]


# --------------------------------------------------------------------------- pure


def window_metrics(row: Row, start: str, end: str, fee: float) -> dict[str, Any]:
    """Return, entries, Sharpe, PF, max DD of one run's ``[start, end)`` slice.

    Daily MTM returns give return, Sharpe and DD; trades are positions ENTERED in the window,
    and PF is over those that closed (all-in 1x trade multiplier with ``fee`` per side, the
    oos-analysis definition).
    """
    from vibe_quant.metrics import profit_factor

    a, b = oa.day_of(start), oa.day_of(end)
    first = int(row["first_day"])
    vals = [v for i, v in enumerate(row["daily"]) if a <= first + i < b]
    a_ns, b_ns = a * oa.DAY_NS, b * oa.DAY_NS
    entered = [p for p in row["positions"] if a_ns <= p["ts_open"] < b_ns]
    pnl = [
        oa.trade_multiplier(p["side"], p["px_open"], p["px_close"], fee) - 1.0
        for p in entered if p["px_close"] is not None
    ]
    return {
        "window": f"{start}..{end}", "days": len(vals), "ret": oa.compound(vals),
        "trades": len(entered), "sharpe": oa.annual_sharpe(vals) if len(vals) > 2 else 0.0,
        "pf": profit_factor(pnl), "dd": oa.max_drawdown(vals),
    }


def search_ok(m: Row) -> bool:
    return bool(m["ret"] > 0 and m["trades"] >= MIN_TRADES)


def unseen_ok(m: Row) -> bool:
    return bool(m["ret"] > 0 and m["trades"] >= MIN_TRADES and m["pf"] >= MIN_PF and m["dd"] <= MAX_DD)


def _finite(v: float) -> float:
    return v if math.isfinite(v) else 0.0


# --------------------------------------------------------------------------- collect


def _r2_tasks(grid: Sequence[Row]) -> tuple[dict[tuple[str, str, str, str], Row], list[dict[str, Any]]]:
    """Search rows by (port, preset, tf, coin) and the test windows worth re-running."""
    runs = {(r["port"], r["preset"], r["tf"], r["coin"], r["window"]): r for r in grid if r["kind"] == "run"}
    search = {k[:4]: r for k, r in runs.items() if k[4] == "search"}
    presets = {p.name: {e["name"]: e["values"] for e in preset_grid.load_presets(p)}
               for p in preset_grid.port_dirs("all")}
    tasks = []
    for (port, preset, tf, coin, window), r in runs.items():
        if window != "test" or r["error"]:
            continue
        if not (r["ret"] > 0 and r["trades"] >= PRE_MIN_TRADES and (r["dd"] if r["dd"] is not None else 1) <= PRE_MAX_DD):
            continue
        vals = presets[port][preset]
        tasks.append({
            "key": f"rerun|{port}|{preset}|{tf}|{coin}", "sid": oa.setting_id(port, tf, vals),
            "port": port, "tf": tf, "values": vals, "coin": coin,
            "start": r["start"], "end": r["end"], "preset": preset, "ledger_ret": r["ret"],
        })
    return search, tasks


def _fwd(fwd: Mapping[str, Any], sid: str, coin: str) -> dict[str, Any] | None:
    c = fwd.get(sid, {}).get(coin)
    return None if c is None or "ret" not in c else {k: c[k] for k in ("window", "ret", "trades", "dd", "carried")}


def cmd_collect(args: argparse.Namespace) -> int:
    from vibe_quant.fees import FeeSchedule

    os.chdir(REPO_ROOT)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fee = float(FeeSchedule().taker_rate)
    fwd = {t["sid"]: t["per_coin"] for t in json.loads((SWARM / "oos-analysis-fwd" / "forward.json").read_text())}

    # round 2: 7,520 (preset, tf, coin) searches; re-run the test windows that can pass
    grid = preset_grid.read_rows(SWARM / "presets-grid" / "results.jsonl")
    search, tasks = _r2_tasks(grid)
    reruns = out / "reruns.jsonl"
    present = preset_grid.done_keys(reruns)
    todo = [t | {"kind": "rerun"} for t in tasks if t["key"] not in present]
    if todo:
        oa.run_tasks(todo, reruns, args.parallel)
    by_key = {r["key"]: r for r in preset_grid.read_rows(reruns)}
    meta = {t["key"]: t for t in tasks}
    checked: list[dict[str, Any]] = []
    mismatches = 0
    for key, t in meta.items():
        r = by_key.get(key)
        if r is None or r.get("error"):
            continue
        s = search[(t["port"], t["preset"], t["tf"], t["coin"])]
        u = window_metrics(r, t["start"], t["end"], fee)
        mismatches += abs(r["ret"] - t["ledger_ret"]) > 1e-9
        checked.append({
            "round": 2, "port": t["port"], "tf": t["tf"], "coin": t["coin"], "tag": t["preset"],
            "sid": t["sid"], "values": t["values"],
            "search": {"window": f"{s['start']}..{s['end']}", "ret": s["ret"], "trades": s["trades"],
                       "sharpe": s["sharpe"], "dd": s["dd"], "source": "presets-grid ledger"},
            "unseen": u | {"engine_ret": r["ret"], "source": "re-run of the presets-grid test window"},
            "forward": _fwd(fwd, t["sid"], t["coin"]),
        })
    r2_search_pass = sum(1 for s in search.values() if not s["error"] and search_ok(s))
    r2_runs = sum(1 for s in search.values() if not s["error"])

    # round 1: the GA settings of watchlist.json, all 8 coins, from the oos-analysis continuous run
    uni = json.loads((SWARM / "oos-analysis" / "universe.json").read_text())
    rows = oa.load_results(SWARM / "oos-analysis" / "results.jsonl")
    uid_of: dict[str, str] = {}
    for cand in json.loads((SWARM / "watchlist.json").read_text())["candidates"]:
        port = Path(cand["port"]).name
        uid_of.setdefault(oa.setting_id(port, cand["timeframe"], oa._full_values(port, cand["values"])), cand["uid"])
    r1_pairs = r1_search_pass = 0
    r1_unseen_pass_given_fail = 0
    for s in uni:
        if "U1" not in s["universes"]:
            continue
        for coin, cr in rows.get(s["sid"], {}).items():
            if cr.get("error"):
                continue
            sm = window_metrics(cr, max(R1_SEARCH[0], cr["start"]), R1_SEARCH[1], fee)
            um = window_metrics(cr, max(R1_UNSEEN[0], cr["start"]), R1_UNSEEN[1], fee)
            r1_pairs += 1
            r1_search_pass += search_ok(sm)
            r1_unseen_pass_given_fail += (not search_ok(sm)) and unseen_ok(um)
            checked.append({
                "round": 1, "port": s["port"], "tf": s["tf"], "coin": coin, "tag": uid_of[s["sid"]],
                "sid": s["sid"], "values": s["values"],
                "search": sm | {"source": "oos-analysis continuous run, sliced"},
                "unseen": um | {"source": "oos-analysis continuous run, sliced"},
                "forward": _fwd(fwd, s["sid"], coin),
            })

    for c in checked:
        c["keeper"] = search_ok(c["search"]) and unseen_ok(c["unseen"])
        c["name"] = f"tvk-{c['port']}-{c['tf']}-{c['tag']}-{c['coin']}"
    keepers = [c for c in checked if c["keeper"]]
    n_trials = {1: 3_280, 2: r2_runs}
    for k in keepers:
        k["label"] = f"single-coin, selected from {n_trials[k['round']]:,} trials, not evidence"
        k["n_trials"] = n_trials[k["round"]]
    keepers.sort(key=lambda k: -_finite(k["unseen"]["sharpe"]))
    (out / "checked.jsonl").write_text("".join(json.dumps(c) + "\n" for c in checked))
    (out / "keepers.jsonl").write_text("".join(json.dumps(k) + "\n" for k in keepers))

    # selection-free baseline: how often the unseen bar clears when the search bar FAILED
    r2_fail = r2_runs - r2_search_pass
    r2_unseen_given_fail = sum(1 for c in checked if c["round"] == 2 and not search_ok(c["search"]) and unseen_ok(c["unseen"]))
    rate2 = r2_unseen_given_fail / r2_fail if r2_fail else float("nan")
    r1_fail = r1_pairs - r1_search_pass
    rate1 = r1_unseen_pass_given_fail / r1_fail if r1_fail else float("nan")
    k1 = sum(1 for k in keepers if k["round"] == 1)
    k2 = sum(1 for k in keepers if k["round"] == 2)
    readme = f"""# TradingView-port single-coin keepers (PRIVATE -- gitignored, never commit)

Every row is single-coin, selected from N trials, NOT evidence of an edge.

## Bar (chief, 2026-11-10), per (setting, timeframe, coin)
- search window: return > 0, >= {MIN_TRADES} trades
- unseen window: return > 0, >= {MIN_TRADES} trades, PF >= {MIN_PF}, max DD <= {MAX_DD:.0%}
- round 2 presets: search {R2_SEARCH[0]}..{R2_SEARCH[1]}, unseen {R2_UNSEEN[0]}..{R2_UNSEEN[1]} (fresh runs)
- round-1 GA settings: search {R1_SEARCH[0]}..{R1_SEARCH[1]}, unseen {R1_UNSEEN[0]}..{R1_UNSEEN[1]}
  (slices of one continuous run from 2023)
- forward 2026-10-20..: recorded where oos-analysis-fwd has it (52 settings only), no bar

Metric definitions: return/Sharpe/DD from daily mark-to-market returns of the window; trades =
positions entered in the window; PF over those that closed, all-in 1x trade multiplier with
taker fee {fee} per side (oos_analysis.py's definition). Screening tier (NTBacktestFn, all-in 1x,
isolated, funding on).

## Counts
- round 2: {r2_runs:,} (preset, tf, coin) trials; {r2_search_pass:,} pass the search bar; keepers {k2}
- round 1: 3,280 GA trials -> {sum(1 for s in uni if 'U1' in s['universes'])} distinct settings x 8 coins = {r1_pairs} pairs; {r1_search_pass} pass the search bar; keepers {k1}
- keepers total: {len(keepers)} (N = 3,280 + {r2_runs:,} = {3_280 + r2_runs:,} trials)
- round-2 re-runs whose engine return differs from the ledger by > 1e-9: {mismatches}

## Expected keepers by chance
- Random-timing null: unknown. The stored null (oos-analysis) keeps total return per draw only,
  no PF or DD, and the round-2 grid stored no trade lists.
- Selection-free baseline (does the search bar add information?): the unseen bar passes on
  {r2_unseen_given_fail}/{r2_fail:,} = {rate2:.2%} of round-2 trials that FAILED the search bar, so
  {r2_search_pass:,} search-passers would give ~{rate2 * r2_search_pass:.0f} keepers if the search
  bar carried no information (observed {k2}). Round 1: {r1_unseen_pass_given_fail}/{r1_fail} = {rate1:.2%}
  of search-failing pairs pass unseen -> ~{rate1 * r1_search_pass:.1f} expected of {r1_search_pass}
  (observed {k1}).

Files: keepers.jsonl (sorted by unseen Sharpe), checked.jsonl (every pair judged; round 2 only
the re-run test windows), reruns.jsonl (captured re-runs), yaml/, db_runs.jsonl, summary.md.
"""
    (out / "README.md").write_text(readme)
    print(readme)
    return 0


# --------------------------------------------------------------------------- render


def cmd_render(args: argparse.Namespace) -> int:
    os.chdir(REPO_ROOT)
    out = Path(args.out)
    (out / "yaml").mkdir(exist_ok=True)
    for k in preset_grid.read_rows(out / "keepers.jsonl"):
        cmd = [sys.executable, str(HERE / "tv.py"), "render", "--port", str(HERE / k["port"]),
               "--timeframe", k["tf"], *[a for key, v in sorted(k["values"].items()) for a in ("--set", f"{key}={v}")]]
        text = subprocess.run(cmd, check=True, capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": str(REPO_ROOT)}).stdout
        (out / "yaml" / f"{k['name']}.yaml").write_text(text)
    print(f"rendered {len(list((out / 'yaml').glob('*.yaml')))} YAML files")
    return 0


# --------------------------------------------------------------------------- persist


def _api(base: str, method: str, path: str, body: Mapping[str, Any] | None = None) -> Any:
    req = urllib.request.Request(
        base + path, method=method, headers={"Content-Type": "application/json"},
        data=None if body is None else json.dumps(body).encode(),
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{method} {path} -> {exc.code}: {exc.read().decode()[:400]}") from exc
    return json.loads(raw) if raw else None


def _dsl_name(name: str) -> str:
    clean = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
    return clean[:100]


def _screening_job(task: dict[str, Any]) -> dict[str, Any]:
    r = oa.run_task(task | {"kind": "scr"})
    return {"name": task["name"], "error": r["error"], "ret": r.get("ret"), "trades": r.get("trades"),
            "sharpe": r.get("sharpe"), "dd": r.get("dd")}


def cmd_persist(args: argparse.Namespace) -> int:
    os.chdir(REPO_ROOT)
    out = Path(args.out)
    keepers = preset_grid.read_rows(out / "keepers.jsonl")
    seen: set[tuple[str, str]] = set()
    chosen = []
    for k in keepers:  # sorted by unseen Sharpe; one DB run per (setting, coin)
        if (k["sid"], k["coin"]) not in seen:
            seen.add((k["sid"], k["coin"]))
            chosen.append(k)
    chosen = chosen[: args.limit]
    spans = oa.coin_spans(sorted({k["coin"] for k in chosen}))
    log = out / "db_runs.jsonl"
    done = {r["name"]: r for r in preset_grid.read_rows(log)}
    registered = {s["name"]: s["id"] for s in _api(args.api, "GET", "/api/strategies")["strategies"]}
    pending: dict[int, dict[str, Any]] = {}
    queue = [k for k in chosen if k["name"] not in done or done[k["name"]].get("status") != "completed"]
    queue.sort(key=lambda k: k["name"] not in registered)  # adopt in-flight runs before launching new ones
    print(f"{len(chosen)} keepers to persist, {len(queue)} not yet completed", flush=True)

    def launch(k: Row) -> None:
        start, end = spans[k["coin"]]
        dsl = yaml_load(out / "yaml" / f"{k['name']}.yaml")
        dsl["name"] = _dsl_name(k["name"])
        existing = registered.get(k["name"])
        if existing is None:
            existing = _api(args.api, "POST", "/api/strategies", {
                "name": k["name"], "dsl_config": dsl, "description": NOTE})["id"]
        # resume: adopt this window's validation run if an earlier pass launched it
        prior = [r for r in _api(args.api, "GET", f"/api/results/runs?strategy_id={existing}")["runs"]
                 if r["run_mode"] == "validation" and r["start_date"] == start and r["end_date"] == end
                 and r["status"] in ("pending", "running", "completed")]
        run = prior[0] if prior else _api(args.api, "POST", "/api/backtest/validation", {
            "strategy_id": existing, "symbols": [k["coin"]], "timeframe": k["tf"],
            "start_date": start, "end_date": end, "parameters": dict(SIZING), "liquidation_mode": "isolated"})
        pending[run["id"]] = {"name": k["name"], "strategy_id": existing, "run_id": run["id"],
                              "coin": k["coin"], "tf": k["tf"], "start": start, "end": end, "keeper": k}
        print(f"{'adopted' if prior else 'launched'} run {run['id']} {k['name']}", flush=True)

    def finish(rec: dict[str, Any], meta: Row) -> None:
        k = rec.pop("keeper")
        row = {**rec, "status": meta["status"], "error_message": meta.get("error_message")}
        if meta["status"] == "completed":
            _api(args.api, "PUT", f"/api/results/runs/{rec['run_id']}/notes", {"notes": NOTE})
            res = _api(args.api, "GET", f"/api/results/runs/{rec['run_id']}")
            row["validation"] = {f: res.get(f) for f in (
                "total_return", "sharpe_ratio", "max_drawdown", "total_trades", "profit_factor",
                "total_fees", "total_funding", "total_slippage")}
            scr = _screening_job({"key": k["name"], "sid": k["sid"], "port": k["port"], "tf": k["tf"],
                                  "values": k["values"], "coin": k["coin"], "start": rec["start"],
                                  "end": rec["end"], "name": k["name"]})
            row["screening"] = scr
        preset_grid._append(log, row)
        print(f"run {rec['run_id']} {meta['status']} {k['name']}", flush=True)

    while queue or pending:
        while queue and len(pending) < args.parallel:
            try:
                launch(queue.pop(0))
            except RuntimeError as exc:
                preset_grid._append(log, {"name": "?", "status": "launch_error", "error_message": str(exc)})
                print(f"launch error: {exc}", flush=True)
        time.sleep(5)
        for rid in list(pending):
            meta = _api(args.api, "GET", f"/api/results/runs/{rid}/meta")
            if meta["status"] in ("completed", "failed", "cancelled"):
                finish(pending.pop(rid), meta)
    return 0


def yaml_load(path: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: not a mapping")
    return data


# --------------------------------------------------------------------------- summary


def _p(v: float | None) -> str:
    return "n/a" if v is None or not math.isfinite(v) else f"{v * 100:+.1f}%"


def _n(v: float | None, nd: int = 2) -> str:
    return "n/a" if v is None or not math.isfinite(v) else f"{v:.{nd}f}"


def cmd_summary(args: argparse.Namespace) -> int:
    out = Path(args.out)
    latest: dict[str, dict[str, Any]] = {}
    for r in preset_grid.read_rows(out / "db_runs.jsonl"):
        if r["name"] != "?":
            latest[r["name"]] = r
    keep = {k["name"]: k for k in preset_grid.read_rows(out / "keepers.jsonl")}
    lines = [
        "# Keepers: validation tier vs screening (PRIVATE)", "",
        "Same coin, same full-history window, all-in 1x isolated. Single-coin keepers, not evidence.", "",
        "| keeper | run id | window | unseen ret / Sh / PF / DD | screening ret / tr / Sh / DD | "
        "validation ret / tr / Sh / PF / DD | val - scr return |",
        "|---|---|---|---|---|---|---|",
    ]
    gaps: list[tuple[float, float]] = []
    sharpe_gaps: list[float] = []
    for name, r in sorted(latest.items(), key=lambda kv: -_finite(keep[kv[0]]["unseen"]["sharpe"]) if kv[0] in keep else 0):
        k = keep.get(name)
        u = k["unseen"] if k else {}
        if r.get("status") != "completed":
            lines.append(f"| {name} | {r.get('run_id')} | | | | {r.get('status')}: {str(r.get('error_message'))[:80]} | |")
            continue
        v, s = r["validation"], r["screening"]
        gap = (v["total_return"] - s["ret"]) if v["total_return"] is not None and s.get("ret") is not None else None
        if gap is not None:
            gaps.append((v["total_return"], s["ret"]))
            if v["sharpe_ratio"] is not None and s.get("sharpe") is not None:
                sharpe_gaps.append(v["sharpe_ratio"] - s["sharpe"])
        lines.append(
            f"| {name} | {r['run_id']} | {r['start']}..{r['end']} | {_p(u.get('ret'))} / {_n(u.get('sharpe'))} / "
            f"{_n(u.get('pf'))} / {_p(u.get('dd'))} | {_p(s.get('ret'))} / {s.get('trades')} / {_n(s.get('sharpe'))} / "
            f"{_p(s.get('dd'))} | {_p(v['total_return'])} / {v['total_trades']} / {_n(v['sharpe_ratio'])} / "
            f"{_n(v['profit_factor'])} / {_p(v['max_drawdown'])} | {_p(gap)} |"
        )
    if gaps:
        diffs = [a - b for a, b in gaps]
        lines += ["", f"Validation minus screening total return over {len(gaps)} runs: median {_p(statistics.median(diffs))}, "
                  f"mean {_p(statistics.fmean(diffs))}; validation > 0 on {sum(a > 0 for a, _ in gaps)}, "
                  f"screening > 0 on {sum(b > 0 for _, b in gaps)}; validation equity / screening equity "
                  f"median {statistics.median((1 + a) / (1 + b) for a, b in gaps):.3f}; validation below "
                  f"screening on {sum(a < b for a, b in gaps)}; Sharpe (val - scr) median "
                  f"{_n(statistics.median(sharpe_gaps)) if sharpe_gaps else 'n/a'}."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-3:]))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in (("collect", cmd_collect), ("render", cmd_render), ("persist", cmd_persist), ("summary", cmd_summary)):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.set_defaults(fn=fn)
        if name == "collect":
            p.add_argument("--parallel", type=int, default=6)
        if name == "persist":
            p.add_argument("--api", default="http://localhost:8000")
            p.add_argument("--limit", type=int, default=150)
            p.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    ignored = subprocess.run(["git", "check-ignore", "-q", str(out)], cwd=REPO_ROOT).returncode == 0
    if not ignored:
        raise SystemExit(f"{out} is not gitignored; keepers are private")
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
