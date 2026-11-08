"""preset_grid.py -- backtest every hand-picked TradingView preset on a search and an unseen test window.

    run     --out DIR [--parallel N] [--ports all|a,b] [--timeframes 15m,1h,4h]
            [--coins BTCUSDT,...] [--presets N] [--retry-errors]
        One engine backtest per port x preset x timeframe x coin x window, appended to
        DIR/results.jsonl. Resumable: a key already in the file is skipped (a row with an error is
        redone only with --retry-errors). A failing backtest is recorded, never fatal.

    report  --out DIR
        Print and write DIR/report.md: the pre-registered pick per port x timeframe and how it did
        on the unseen test window.

Pre-registered rules (2026-11-08, before any 2026 result was seen):

* SEARCH window 2023-01-01..2026-01-01, TEST window 2026-01-01..2026-10-20 (end exclusive).
  A coin whose data starts later begins at its first full day + 1 (so the buy-and-hold base close
  exists); a coin with fewer than 180 days in the SEARCH window is skipped there and reported.
* PICK per port x timeframe: the preset with the most coins that have a search return > 0 on
  >= 30 trades; ties go to the highest median search Sharpe across coins, then to the preset's
  place in presets.yaml.
* LENIENT bar: the pick's test return > 0 on >= 1 coin. STRICT bar: the pick's test return beats
  buy-and-hold on >= 4 of the coins tested.
* Chance rates: over ALL preset x timeframe x coin test runs, the share with return > 0 and the
  share beating buy-and-hold.

Run from the repository root with ``PYTHONPATH=$PWD .venv/bin/python scripts/research/tv_ports/preset_grid.py ...``
(the engine reads ``data/catalog`` relative to the working directory).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
CATALOG = REPO_ROOT / "data" / "catalog"

COINS = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "HYPEUSDT", "NEARUSDT", "SUIUSDT", "ZECUSDT",
)
TIMEFRAMES = ("15m", "1h", "4h")
WINDOWS: dict[str, tuple[str, str]] = {
    "search": ("2023-01-01", "2026-01-01"),
    "test": ("2026-01-01", "2026-10-20"),
}
MIN_SEARCH_DAYS = 180
MIN_SEARCH_TRADES = 30
STRICT_MIN_BEATS = 4
EPOCH = date(1970, 1, 1)
WORKER_RECYCLE = 40  # backtests per worker process before it is replaced (NT engines leak)

Row = Mapping[str, Any]


# --------------------------------------------------------------------------- keys / windows


def run_key(port: str, preset: str, tf: str, coin: str, window: str) -> str:
    """Identity of one backtest; the resume logic skips keys already in results.jsonl."""
    return f"run|{port}|{preset}|{tf}|{coin}|{window}"


def aux_key(kind: str, coin: str, window: str) -> str:
    """Identity of a buy-and-hold or skip record (one per coin and window)."""
    return f"{kind}|{coin}|{window}"


def done_keys(path: Path, *, retry_errors: bool = False) -> set[str]:
    """Keys already present in ``path``; a torn last line is ignored.

    With ``retry_errors`` a ``run`` row that carries an error does not count as done.
    """
    keys: set[str] = set()
    if not path.exists():
        return keys
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if retry_errors and row.get("kind") == "run" and row.get("error"):
            continue
        keys.add(row["key"])
    return keys


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Every parseable row of ``path`` (a torn last line is ignored)."""
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def clip_window(start: str, end: str, first_close_day: date) -> tuple[date, date]:
    """The window actually run: ``start`` moved to the coin's first full day + 1 when later."""
    s = date.fromisoformat(start)
    first_ok = date.fromordinal(first_close_day.toordinal() + 1)
    return max(s, first_ok), date.fromisoformat(end)


def bh_return(closes: Mapping[int, float], start: date, end: date) -> float | None:
    """Buy-and-hold over ``[start, end)``: last in-window close over the close before ``start``."""
    base = closes.get((start - EPOCH).days - 1)
    last = closes.get((end - EPOCH).days - 1)
    if base is None or last is None or base <= 0:
        return None
    return last / base - 1.0


# --------------------------------------------------------------------------- selection (pure)


def _finite(value: object) -> float:
    number = float(value) if isinstance(value, (int, float)) else 0.0
    return number if math.isfinite(number) else 0.0


def preset_score(rows: Iterable[Row]) -> tuple[int, float]:
    """``(coins positive, median Sharpe)`` of one preset's SEARCH rows (one per coin)."""
    valid = [r for r in rows if not r.get("error")]
    positive = sum(1 for r in valid if r["ret"] > 0 and r["trades"] >= MIN_SEARCH_TRADES)
    median = statistics.median(_finite(r["sharpe"]) for r in valid) if valid else -math.inf
    return positive, median


def select_pick(search_rows: Sequence[Row], order: Sequence[str]) -> str | None:
    """The pre-registered pick among ``order`` (presets.yaml order) from SEARCH rows of one port x tf.

    Most positive coins, then highest median Sharpe, then earliest in ``order``. ``None`` when
    no preset has a valid row.
    """
    best: tuple[int, float] | None = None
    pick: str | None = None
    for name in order:
        rows = [r for r in search_rows if r["preset"] == name]
        if not any(not r.get("error") for r in rows):
            continue
        score = preset_score(rows)
        if best is None or score > best:
            best, pick = score, name
    return pick


def evaluate_pick(
    test_rows: Sequence[Row], bh: Mapping[str, float | None]
) -> dict[str, Any]:
    """How a pick did on the TEST window: per coin return, coins > 0, coins beating buy-and-hold."""
    valid = [r for r in test_rows if not r.get("error")]
    per_coin = {r["coin"]: r["ret"] for r in valid}
    beats = [
        c for c, ret in per_coin.items() if (b := bh.get(c)) is not None and ret > b
    ]
    return {
        "per_coin": per_coin,
        "positive": sum(1 for ret in per_coin.values() if ret > 0),
        "beats_bh": len(beats),
        "tested": len(per_coin),
        "mean": statistics.fmean(per_coin.values()) if per_coin else None,
        "lenient": any(ret > 0 for ret in per_coin.values()),
        "strict": len(beats) >= STRICT_MIN_BEATS,
    }


def chance_rates(
    test_rows: Iterable[Row], bh: Mapping[str, float | None]
) -> dict[str, float | int]:
    """Share of ALL test runs with return > 0 and with return > buy-and-hold."""
    valid = [r for r in test_rows if not r.get("error")]
    with_bh = [r for r in valid if bh.get(r["coin"]) is not None]
    return {
        "runs": len(valid),
        "share_positive": sum(1 for r in valid if r["ret"] > 0) / len(valid) if valid else 0.0,
        "runs_with_bh": len(with_bh),
        "share_beats_bh": (
            sum(1 for r in with_bh if r["ret"] > bh[r["coin"]]) / len(with_bh) if with_bh else 0.0
        ),
    }


# --------------------------------------------------------------------------- presets / ports


def port_dirs(selector: str) -> list[Path]:
    """Port folders (those with a port.yaml) for ``all`` or a comma list of folder names."""
    every = sorted(p.parent for p in HERE.glob("*/port.yaml"))
    if selector == "all":
        return every
    wanted = [s.strip() for s in selector.split(",") if s.strip()]
    by_name = {p.name: p for p in every}
    missing = [w for w in wanted if w not in by_name]
    if missing:
        raise SystemExit(f"unknown port(s): {missing}")
    return [by_name[w] for w in wanted]


def load_presets(port_dir: Path) -> list[dict[str, Any]]:
    """The port's presets in file order, each ``{name, why, values}`` with every param filled."""
    from vibe_quant.discovery.port import load_port

    port = load_port(port_dir)
    raw = yaml.safe_load((port_dir / "presets.yaml").read_text())
    return [{**e, "values": {**port.defaults(), **e["values"]}} for e in raw]


# --------------------------------------------------------------------------- worker


_PORTS: dict[str, Any] = {}


def _backtest_fn(coin: str, tf: str, start: str, end: str) -> Any:
    from vibe_quant.discovery.backtest_fn import NTBacktestFn
    from vibe_quant.fees import FeeSchedule
    from vibe_quant.liquidation_settings import LiquidationSettings
    from vibe_quant.sizing_settings import SizingSettings

    return NTBacktestFn(
        symbols=[coin],
        timeframe=tf,
        start_date=start,
        end_date=end,
        min_trades=0,
        symbol_agg="portfolio",
        fee_schedule=FeeSchedule(),
        entry_order="market",
        liquidation=LiquidationSettings(mode="isolated", leverage=1),
        # burst.sizing_params for --allin at leverage 1
        sizing=SizingSettings.from_params(
            {"max_position_pct": 1.0, "risk_per_trade": 0.5, "max_margin_pct": 1.0}
        ),
    )


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    """One backtest. Never raises: a failure becomes a row with ``error`` set."""
    row: dict[str, Any] = {
        "kind": "run",
        "key": run_key(task["port"], task["preset"], task["tf"], task["coin"], task["window"]),
        "port": task["port"], "preset": task["preset"], "tf": task["tf"],
        "coin": task["coin"], "window": task["window"], "start": task["start"], "end": task["end"],
        "ret": 0.0, "trades": 0, "sharpe": 0.0, "dd": None, "error": None,
    }
    try:
        from vibe_quant.discovery.param_genome import ParamGenome
        from vibe_quant.discovery.port import load_port

        port = _PORTS.get(task["port"]) or _PORTS.setdefault(
            task["port"], load_port(HERE / task["port"])
        )
        fn = _backtest_fn(task["coin"], task["tf"], task["start"], task["end"])
        res = fn(ParamGenome(port, dict(task["values"])))
        row["ret"] = _finite(res.get("total_return", 0.0))
        row["trades"] = int(res.get("total_trades", 0))
        row["sharpe"] = _finite(res.get("sharpe_ratio", 0.0))
        row["dd"] = _finite(res["max_drawdown"]) if "max_drawdown" in res else None
        row["error"] = res.get("error")
    except Exception as exc:  # a data gap is one failed cell, not a dead grid
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


# --------------------------------------------------------------------------- run


def _append(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()


def plan(
    ports: Sequence[Path],
    timeframes: Sequence[str],
    coins: Sequence[str],
    n_presets: int | None,
    present: set[str],
    results: Path,
) -> list[dict[str, Any]]:
    """Tasks still to run; writes the buy-and-hold and skip records that are missing."""
    from vibe_quant.discovery.buyhold import load_daily_closes
    from vibe_quant.discovery.port import load_port

    closes = load_daily_closes(list(coins), CATALOG)
    spans: dict[tuple[str, str], tuple[str, str]] = {}
    for coin in coins:
        first = date.fromordinal(EPOCH.toordinal() + min(closes[coin]))
        for window, (start, end) in WINDOWS.items():
            s, e = clip_window(start, end, first)
            days = (e - s).days
            if window == "search" and days < MIN_SEARCH_DAYS:
                key = aux_key("skip", coin, window)
                if key not in present:
                    _append(results, {
                        "kind": "skip", "key": key, "coin": coin, "window": window,
                        "reason": f"{days} days in the search window (< {MIN_SEARCH_DAYS})",
                    })
                continue
            spans[(coin, window)] = (s.isoformat(), e.isoformat())
            key = aux_key("bh", coin, window)
            if key not in present:
                _append(results, {
                    "kind": "bh", "key": key, "coin": coin, "window": window,
                    "start": s.isoformat(), "end": e.isoformat(),
                    "ret": bh_return(closes[coin], s, e),
                })
    tasks: list[dict[str, Any]] = []
    for port_dir in ports:
        allowed = load_port(port_dir).allowed_timeframes
        tfs = [t for t in timeframes if allowed is None or t in allowed]
        presets = load_presets(port_dir)[:n_presets]
        for tf in tfs:
            for window in WINDOWS:
                for preset in presets:
                    for coin in coins:
                        if (coin, window) not in spans:
                            continue
                        if run_key(port_dir.name, preset["name"], tf, coin, window) in present:
                            continue
                        span_start, span_end = spans[(coin, window)]
                        tasks.append({
                            "port": port_dir.name, "preset": preset["name"], "tf": tf,
                            "coin": coin, "window": window,
                            "start": span_start, "end": span_end,
                            "values": preset["values"],
                        })
    cost = {"15m": 0, "1h": 1, "4h": 2}
    tasks.sort(key=lambda t: (cost.get(t["tf"], 9), t["window"] != "search"))  # slowest first
    return tasks


def cmd_run(args: argparse.Namespace) -> int:
    os.chdir(REPO_ROOT)  # the engine's catalog paths are cwd-relative
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    present = done_keys(results, retry_errors=args.retry_errors)
    coins = [c.strip() for c in args.coins.split(",") if c.strip()]
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]
    tasks = plan(port_dirs(args.ports), timeframes, coins, args.presets, present, results)
    print(f"{len(tasks)} backtests to run ({len(present)} keys already done)", flush=True)
    done = errors = 0

    def record(row: dict[str, Any]) -> None:
        nonlocal done, errors
        _append(results, row)
        done += 1
        errors += 1 if row["error"] else 0
        if done % 25 == 0 or done == len(tasks):
            print(f"  {done}/{len(tasks)} done, {errors} errors", flush=True)

    if args.parallel <= 1:
        for task in tasks:
            record(run_task(task))
        return 0
    import multiprocessing

    # multiprocessing.Pool, not ProcessPoolExecutor(max_tasks_per_child=...): the executor hung
    # for good when all workers reached the recycle count together (9 x 40 = 360 runs, 0 CPU,
    # no workers left, 2026-11-08); Pool's maxtasksperchild replaces workers one by one.
    with multiprocessing.get_context("spawn").Pool(
        processes=args.parallel, maxtasksperchild=WORKER_RECYCLE
    ) as pool:
        for row in pool.imap_unordered(run_task, tasks):
            record(row)
    return 0


# --------------------------------------------------------------------------- report


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.1f}%"


def build_report(rows: Sequence[Row], orders: Mapping[str, Sequence[str]]) -> tuple[str, dict[str, Any]]:
    """The report text and its numbers. ``orders`` is ``{port: preset names in file order}``."""
    runs = [r for r in rows if r.get("kind") == "run"]
    bh: dict[str, dict[str, float | None]] = {"search": {}, "test": {}}
    for r in rows:
        if r.get("kind") == "bh":
            bh[r["window"]][r["coin"]] = r["ret"]
    skips = [r for r in rows if r.get("kind") == "skip"]
    lines = ["# TradingView preset grid", ""]
    lines.append(f"{len(runs)} backtests, {sum(1 for r in runs if r.get('error'))} with an error.")
    for skip in skips:
        lines.append(f"- skipped {skip['coin']} ({skip['window']}): {skip['reason']}")
    lines += [
        "", "## Pre-registered picks on the unseen test window", "",
        "| port | tf | pick | search coins>0 | med search Sharpe | test return per coin | coins>0 | beat B&H | mean | lenient | strict |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    outcomes: list[dict[str, Any]] = []
    for port in sorted(orders):
        for tf in TIMEFRAMES:
            search = [r for r in runs if r["port"] == port and r["tf"] == tf and r["window"] == "search"]
            pick = select_pick(search, orders[port])
            if pick is None:
                continue
            positive, median = preset_score([r for r in search if r["preset"] == pick])
            test = [
                r for r in runs if r["port"] == port and r["tf"] == tf
                and r["window"] == "test" and r["preset"] == pick
            ]
            result = evaluate_pick(test, bh["test"])
            outcomes.append({"port": port, "tf": tf, "pick": pick, **result})
            per_coin = " ".join(
                f"{c[:-4]} {_pct(v)}" for c, v in sorted(result["per_coin"].items())
            )
            lines.append(
                f"| {port} | {tf} | {pick} | {positive} | {median:.2f} | {per_coin or 'n/a'} "
                f"| {result['positive']}/{result['tested']} | {result['beats_bh']}/{result['tested']} "
                f"| {_pct(result['mean'])} | {'PASS' if result['lenient'] else 'fail'} "
                f"| {'PASS' if result['strict'] else 'fail'} |"
            )
    n = len(outcomes)
    lenient = sum(1 for o in outcomes if o["lenient"])
    strict = sum(1 for o in outcomes if o["strict"])
    chance = chance_rates([r for r in runs if r["window"] == "test"], bh["test"])
    lines += [
        "", "## Bars", "",
        f"- LENIENT (test > 0 on >= 1 coin): {lenient}/{n} picks",
        f"- STRICT (test > buy-and-hold on >= {STRICT_MIN_BEATS} coins): {strict}/{n} picks",
        f"- chance, all {chance['runs']} test runs: {chance['share_positive'] * 100:.1f}% > 0, "
        f"{chance['share_beats_bh'] * 100:.1f}% > buy-and-hold ({chance['runs_with_bh']} with a B&H number)",
        "", "## Buy-and-hold per coin (test window)", "",
        " ".join(f"{c[:-4]} {_pct(v)}" for c, v in sorted(bh["test"].items())),
        "",
    ]
    numbers = {"picks": n, "lenient": lenient, "strict": strict, "chance": chance}
    return "\n".join(lines), numbers


def cmd_report(args: argparse.Namespace) -> int:
    out = Path(args.out)
    rows = read_rows(out / "results.jsonl")
    orders = {
        p.name: [e["name"] for e in load_presets(p)] for p in port_dirs("all")
        if any(r.get("port") == p.name for r in rows)
    }
    text, _ = build_report(rows, orders)
    (out / "report.md").write_text(text)
    print(text)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="backtest every preset (resumable)")
    run.add_argument("--out", required=True)
    run.add_argument("--parallel", type=int, default=1)
    run.add_argument("--ports", default="all")
    run.add_argument("--timeframes", default=",".join(TIMEFRAMES))
    run.add_argument("--coins", default=",".join(COINS))
    run.add_argument("--presets", type=int, default=None, help="only the first N presets (smoke)")
    run.add_argument("--retry-errors", action="store_true")
    run.set_defaults(fn=cmd_run)
    report = sub.add_parser("report", help="print the pre-registered summary")
    report.add_argument("--out", required=True)
    report.set_defaults(fn=cmd_report)
    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
