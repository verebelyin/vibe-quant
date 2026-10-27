#!/usr/bin/env python3
"""Rolling-WFA gate null: pass rate on burst winners that failed transfer (vibe-quant-n2pdj).

The population is the burst ledger's winners whose genome lost on at least one OTHER
symbol or in the next window (``burst.genome_transfers`` False): GA in-sample winners an
independent out-of-sample result calls no-edge. Each is scored by the PRODUCTION rolling-WFA
gate (``GateChain.evaluate_wfa_rolling``) over the cell's own train range, so the printed
rate is the rate the gate passes an edge-free winner at. Its windows tile the TRAIN range
the GA fitted: this measures in-sample stability, not out-of-sample skill.

    PYTHONPATH=$PWD .venv/bin/python scripts/agents/wfa_null.py run \\
        --out data/swarm/burst-20261027h --windows 6,4 --parallel 4 \\
        --result data/swarm/20261027-taker/wfa_null.jsonl
    PYTHONPATH=$PWD .venv/bin/python scripts/agents/wfa_null.py report \\
        --result data/swarm/20261027-taker/wfa_null.jsonl

One WORKER per (winner, K), on its own scratch DB under /tmp: it refuses the real state DB
(a copy of ``burst.run_cell``'s guard), so the measurement never enters a real run's DSR
trial count. Rows are resumable by (key, K). ``report`` prints k/n, the Clopper-Pearson 95%
bounds and PASS iff the one-sided 95% upper bound <= 0.05, per ledger and pooled per K,
plus a crosstab against the next window; any row with ``error`` (a data gap) leaves n and
makes the report exit 2.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scipy.stats import beta

if TYPE_CHECKING:
    from collections.abc import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import burst  # noqa: E402  (sibling script: the ledger, transfer and winner readers)

MIN_TRADES = burst.MIN_TRADES
CP_LEVEL = 0.95  # the DSR pre-registration's null false-positive cap
THRESHOLD = 0.05
ROW_TIMEOUT_S = 1800  # 1m cells: ~22 s per month (burst.py), K windows of a few days each


def window_plan(cell: dict[str, Any], k: int) -> tuple[int, list[tuple[str, str]]]:
    """``(step_days, windows)`` for K rolling windows over the cell's train range.

    ``step = train_days // K``, so the windows are back-to-back ``[start, end)`` and the
    remainder is dropped, exactly as the production gate tiles the range.
    """
    from vibe_quant.discovery.config_check import wfa_window_ranges_for

    days = (date.fromisoformat(cell["end"]) - date.fromisoformat(cell["start"])).days
    step = days // k
    return step, wfa_window_ranges_for(step, cell["start"], cell["end"]) if step >= 1 else []


def next_window_win(trow: dict[str, Any]) -> bool | None:
    """Whether the next-window transfer eval wins, ``None`` without one.

    The same win rule as ``burst``'s transfer check (no error, trade floor, return > 0).
    """
    nxt = trow.get("next")
    if not nxt:
        return None
    return (
        not nxt.get("error")
        and int(nxt.get("trades") or 0) >= MIN_TRADES
        and float(nxt.get("ret") or 0) > 0
    )


def population(
    rows: list[dict[str, Any]], tmap: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Winners a complete, failed transfer marks no-edge, plus the exclusion counts.

    A winner needs a transfer row that did not fail, a next window and other symbols;
    those are checked before ``genome_transfers`` (which reads False for both) so an
    untested winner can never pass for a failed one.
    """
    included: list[dict[str, Any]] = []
    excluded = {"transfer_failed": 0, "no_next": 0, "no_symbols": 0, "transferred": 0,
                "no_transfer": 0}
    for row in burst.winner_rows(rows):
        trow = tmap.get(row["key"])
        if trow is None:
            excluded["no_transfer"] += 1
        elif trow.get("failed"):
            excluded["transfer_failed"] += 1
        elif not trow.get("next"):
            excluded["no_next"] += 1
        elif not (trow.get("symbols") or {}):
            excluded["no_symbols"] += 1
        elif burst.genome_transfers(trow):
            excluded["transferred"] += 1
        else:
            included.append(row)
    return included, excluded


def _row(key: str, ledger: str, k: int, step: int | None, next_win: bool | None) -> dict[str, Any]:
    """The result row a measurement starts from (every field the report reads)."""
    return {"key": key, "ledger": ledger, "K": k, "step": step, "windows": 0, "rets": [],
            "trades": [], "profitable": 0, "passed": False, "error": None, "next_win": next_win}


def score(
    base: dict[str, Any], chromosome: Any, factory: Callable[[str, str], Any]
) -> dict[str, Any]:
    """Run the production rolling-WFA gate once and return the result row."""
    from vibe_quant.discovery.gate_chain import GateChain, GateEvidence
    from vibe_quant.discovery.pipeline import DiscoveryConfig
    from vibe_quant.errors import DataUnavailableError

    cell, k = base["cell"], int(base["K"])
    step, _windows = window_plan(cell, k)
    row = _row(base["key"], base["ledger"], k, step, base["next_win"])
    config = DiscoveryConfig(
        symbols=[cell["symbol"]], timeframe=cell.get("timeframe", "1m"),
        start_date=cell["start"], end_date=cell["end"],
        train_test_split=0.0, wfa_oos_step_days=step, min_trades=MIN_TRADES,
    )
    chain = GateChain(config, GateEvidence(None, False, backtest_fn_factory=factory))
    try:
        results, error = chain.evaluate_wfa_rolling([(chromosome, None)], cell["start"], cell["end"])
    except DataUnavailableError as exc:
        row["error"] = f"DataUnavailableError: {exc}"
        return row
    if error is not None:  # the gate's fail-closed verdict (range too short), not a data gap
        return row
    result = results[0]
    row.update(
        windows=result.total_windows,
        rets=[hr.total_return for hr in result.oos_windows],
        trades=[hr.total_trades for hr in result.oos_windows],
        profitable=result.windows_profitable,
        passed=result.passed,
    )
    return row


def _factory(cell: dict[str, Any]) -> Callable[[str, str], Any]:
    """The run's own cell factory: production NT args, no eval windows (min_trades inert)."""
    from vibe_quant.discovery.run import _make_cell_fns
    from vibe_quant.fees import FeeSchedule
    from vibe_quant.liquidation_settings import LiquidationSettings
    from vibe_quant.sizing_settings import SizingSettings

    fns = _make_cell_fns(
        use_mock=False, needs_factory=True, symbols=[cell["symbol"]],
        timeframe=cell.get("timeframe", "1m"), symbol_agg="portfolio", fee_schedule=FeeSchedule(),
        entry_order="market",
        liquidation=LiquidationSettings(mode="isolated", leverage=int(cell["leverage"])),
        sizing=SizingSettings.from_params({"max_margin_pct": 0.5}),
        train_start=cell["start"], train_end=cell["end"], eval_windows=None,
        min_trades=MIN_TRADES, holdout=None,
    )
    return fns.backtest_fn_factory


def run_one(spec: dict[str, Any], result_path: Path) -> int:
    """Measure one (winner, K) in THIS process; VIBE_QUANT_DB must point at a scratch DB."""
    db = Path(os.environ["VIBE_QUANT_DB"])
    if db.resolve() == (REPO_ROOT / "data" / "state" / "vibe_quant.db").resolve():
        raise SystemExit("wfa_null refuses the real state DB")
    from vibe_quant.discovery.genome import serializable_to_chromosome

    base = {field: spec[field] for field in ("key", "ledger", "K", "cell", "next_win")}
    chromosome = serializable_to_chromosome(spec["chromosome"])
    row = score(base, chromosome, _factory(spec["cell"]))
    result_path.write_text(json.dumps(row))
    return 0


def _launch(spec: dict[str, Any]) -> tuple[subprocess.Popen[bytes], Path, float]:
    tmp = Path(tempfile.mkdtemp(prefix="wfa-null-", dir="/tmp"))
    env = {**os.environ, "VIBE_QUANT_DB": str(tmp / "state.db"),
           "VIBE_QUANT_LOG_DIR": str(tmp / "logs"), "PYTHONPATH": str(REPO_ROOT)}
    with (tmp / "stderr.log").open("wb") as err:
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, __file__, "one", json.dumps(spec), str(tmp / "result.json")],
            cwd=REPO_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=err,
        )
    return proc, tmp, time.monotonic()


def _finish(spec: dict[str, Any], tmp: Path, rc: int | None, wall: float) -> dict[str, Any]:
    row = _row(spec["key"], spec["ledger"], spec["K"], None, spec["next_win"])
    result = tmp / "result.json"
    if rc == 0 and result.exists():
        row.update(json.loads(result.read_text()))
    else:  # a dead worker is a failed row, never a skipped one
        tail = (tmp / "stderr.log").read_text(errors="replace").splitlines()[-15:]
        row.update({"error": f"worker rc={rc} wall={wall:.0f}s: {' | '.join(tail)}"})
    shutil.rmtree(tmp, ignore_errors=True)
    return row


def run(args: argparse.Namespace) -> int:
    result = Path(args.result)
    result.parent.mkdir(parents=True, exist_ok=True)
    ks = [int(token) for token in args.windows.split(",") if token.strip()]
    done = {(row["key"], int(row["K"])) for row in burst.load_ledger(result)}
    specs: list[dict[str, Any]] = []
    for out in args.out:
        path = Path(out)
        tmap = {row["key"]: row for row in burst.load_ledger(path / "transfer.jsonl")}
        included, excluded = population(burst.load_ledger(path / "ledger.jsonl"), tmap)
        print(f"{path.name}: {len(included)} no-edge winners, "
              f"excluded={json.dumps(excluded, sort_keys=True)}", flush=True)
        specs.extend(
            {
                "key": row["key"], "ledger": path.name, "K": k, "cell": row["cell"],
                "chromosome": row["best"]["chromosome"],
                "next_win": next_window_win(tmap[row["key"]]),
            }
            for row in included
            for k in ks
        )
    todo = [spec for spec in specs if (spec["key"], spec["K"]) not in done]
    print(f"wfa_null: {len(todo)} rows to run ({len(specs) - len(todo)} already in "
          f"{result.name})", flush=True)
    running: list[tuple[dict[str, Any], subprocess.Popen[bytes], Path, float]] = []
    with result.open("a") as fh:
        while todo or running:
            while todo and len(running) < args.parallel:
                spec = todo.pop(0)
                running.append((spec, *_launch(spec)))
            if not running:
                break
            for item in list(running):
                spec, proc, tmp, t0 = item
                wall = time.monotonic() - t0
                if proc.poll() is None and wall < ROW_TIMEOUT_S:
                    continue
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                row = _finish(spec, tmp, proc.returncode, wall)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                running.remove(item)
                print(f"{row['ledger']} {row['key']} K={row['K']} step={row['step']} "
                      f"{row['profitable']}/{row['windows']} pass={row['passed']} "
                      f"error={row['error']} wall={wall:.0f}s", flush=True)
            time.sleep(0.5)
    return 0


def cp_bounds(k: int, n: int, level: float = CP_LEVEL) -> tuple[float, float]:
    """Two-sided Clopper-Pearson interval for k passes in n trials."""
    alpha = (1.0 - level) / 2.0
    lo = 0.0 if k == 0 else float(beta.ppf(alpha, k, n - k + 1))
    hi = 1.0 if k >= n else float(beta.ppf(1.0 - alpha, k + 1, n - k))
    return lo, hi


def cp_upper(k: int, n: int, level: float = CP_LEVEL) -> float:
    """One-sided Clopper-Pearson upper bound for k passes in n trials."""
    if k >= n:
        return 1.0
    return float(beta.ppf(level, k + 1, n - k))


def verdict(k: int, n: int) -> str:
    """PASS iff the one-sided 95% upper bound <= 0.05 (no evidence is a FAIL)."""
    return "PASS" if n and cp_upper(k, n) <= THRESHOLD else "FAIL"


def report(args: argparse.Namespace) -> int:
    rows = burst.load_ledger(Path(args.result))
    errors = [row for row in rows if row.get("error")]
    ok = [row for row in rows if not row.get("error")]
    print(f"wfa_null: rows={len(rows)} errors={len(errors)}", flush=True)
    for row in errors:
        print(f"ERROR {row['ledger']} {row['key']} K={row['K']}: {row['error']}")
    print(f"\n## pass rate (no-edge winners; CP95 two-sided; one-sided 95% upper;"
          f" PASS iff upper <= {THRESHOLD}); per ledger and pooled per K")
    ledgers = sorted({str(row["ledger"]) for row in rows})
    ks = sorted({int(row["K"]) for row in rows})
    for k in ks:
        for name in (*ledgers, "pooled"):
            group = [r for r in ok if int(r["K"]) == k
                     and (name == "pooled" or r["ledger"] == name)]
            if not group:
                continue
            n, passed = len(group), sum(1 for r in group if r["passed"])
            lo, hi = cp_bounds(passed, n)
            print(f"{name} K={k} step={group[0]['step']} windows={group[0]['windows']}"
                  f" {passed}/{n} rate={passed / n:.3f} CP95=({lo!r},{hi!r})"
                  f" upper={cp_upper(passed, n)!r} {verdict(passed, n)}")
    print("\n## next window (passed, next_win) over rows that have one, pooled per K")
    for k in ks:
        pairs = [(r["passed"], r["next_win"]) for r in ok
                 if int(r["K"]) == k and r.get("next_win") is not None]
        counts = {pair: pairs.count(pair) for pair in dict.fromkeys(pairs)}
        cells = " ".join(
            f"({('T' if p else 'F')},{('T' if w else 'F')})={c}" for (p, w), c in sorted(counts.items())
        )
        print(f"K={k} {cells}")
    return 2 if errors else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", action="append", required=True, help="a burst ledger directory")
    r.add_argument("--windows", required=True, help="K values, e.g. 3,4,6")
    r.add_argument("--parallel", type=int, default=2)
    r.add_argument("--result", required=True, help="the jsonl rows (resumable by key, K)")
    rep = sub.add_parser("report")
    rep.add_argument("--result", required=True)
    one = sub.add_parser("one")
    one.add_argument("spec")
    one.add_argument("result")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return run(args)
    if args.cmd == "report":
        return report(args)
    return run_one(json.loads(args.spec), Path(args.result))


if __name__ == "__main__":
    raise SystemExit(main())
