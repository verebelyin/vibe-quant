#!/usr/bin/env python3
"""Run the paper-node restart chain on several git revs and compare them step by step.

    restart_probe.py --rev main --rev <sha> [--rev <base>...] [--repo DIR] [--json]

Each rev is `git archive`d into a temp dir and a child python (cwd + PYTHONPATH = that tree,
so the rev's own vibe_quant/ and tests/ are imported, checked in the child) drives the REAL
`PaperTradingNode.start()` four times against one state DB, with a fake TradingNode and a
`_run_loop` that raises a marker:

  S1 halted checkpoint (manual halt + risk) whose `positions` column is the JSON `null`
  S2 restart where the S1 failure checkpoint's `orders` column is `null` too
  S3 restart where the newest `node_status` is unreadable text
  S4 plain restart from S3's failure checkpoint

Per step it records the exception, what the run loop saw (state, halt reason, restored
risk), the newest checkpoint's state / halt reason / error message / risk-preserved flag, and
whether the marker leaked into the log or a checkpoint. Output: one line per step and rev,
then `S<n> IDENTICAL|DIFFERS`, then a final `IDENTICAL` or `DIFFERS (first at S<n>)`; exit
0 when every rev matches the first one, 1 when they differ, 2 when a child crashed.
Needs no data/ and no git common-dir (the exactness scripts do: board gotcha rev-carry-perf-1).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

STEPS = ("S1", "S2", "S3", "S4", "LEAKS")

CHILD = r'''
import asyncio, json, logging, sqlite3, sys, tempfile
from pathlib import Path

import pytest

import vibe_quant
root = Path.cwd().resolve()
if root not in Path(vibe_quant.__file__).resolve().parents:
    raise SystemExit(f"imported vibe_quant from {vibe_quant.__file__}, not {root}")
from tests.unit.test_audit_paper_node import DSL, _config, _FakeTradingNode
from vibe_quant.db.state_manager import StateManager
from vibe_quant.paper.node import PaperTradingNode
from vibe_quant.paper.persistence import StateCheckpoint, StatePersistence

MARK = "MARKER-7f3e-SECRET"
RISK = {
    "high_water_mark": "10600", "day": "2026-10-15", "day_start_equity": "10000",
    "consecutive_losses": 3, "last_equity": "10000",
    "drawdown_pct": "0.05660377358490566037735849057", "daily_loss_pct": "0",
}
tmp = Path(tempfile.mkdtemp(prefix="restart-probe-"))
db = tmp / "state.db"


def clean(text):
    return None if text is None else str(text).replace(str(tmp), "<tmp>")


def rows():
    with sqlite3.connect(db) as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(
            "SELECT id, node_status, positions, orders FROM paper_trading_checkpoints ORDER BY id")]


def newest_status():
    raw = rows()[-1]["node_status"]
    try:
        return json.loads(raw)
    except Exception:
        return {"_raw": raw}


def corrupt_newest(column, value):
    with sqlite3.connect(db) as c:
        c.execute(f"UPDATE paper_trading_checkpoints SET {column} = ? "
                  "WHERE id = (SELECT MAX(id) FROM paper_trading_checkpoints)", (value,))


class Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.hits = 0

    def emit(self, record):
        if MARK in record.getMessage() or (record.exc_info and MARK in str(record.exc_info[1])):
            self.hits += 1


async def start_once(mp, seeded):
    node = PaperTradingNode(_config(seeded))
    mp.setattr(node, "_setup_signal_handlers", lambda: None)
    seen = {}

    async def fake_loop():
        seen["state"] = node.status.state.value
        seen["halt"] = node.status.halt_reason.value if node.status.halt_reason else None
        seen["risk_restored"] = (node._restored_risk.to_dict() == RISK) if node._restored_risk else None
        raise RuntimeError(MARK)

    mp.setattr(node, "_run_loop", fake_loop)
    exc = None
    try:
        await node.start()
    except Exception as e:
        exc = e
    st = newest_status()
    return {
        "exc": type(exc).__name__ if exc else None, "exc_msg": clean(exc),
        "loop_saw": seen or None, "state": st.get("state"), "halt_reason": st.get("halt_reason"),
        "risk_kept": st.get("risk") == RISK, "error_message": clean(st.get("error_message")),
    }


async def main():
    import nautilus_trader.live.node as live_node
    sm = StateManager(db)
    sid = sm.create_strategy("test_strategy", DSL)
    rid = sm.create_backtest_run(
        strategy_id=sid, run_mode="validation", symbols=["BTCUSDT"], timeframe="1h",
        start_date="2024-01-01", end_date="2024-12-31",
        parameters={"rsi_14_period": 21, "take_profit_risk_reward": 3.0, "leverage": 5})
    sm.update_backtest_run_status(rid, "completed")
    sm.close()
    seeded = {"db": db, "sid": sid, "rid": rid, "tmp": tmp}
    handler = Capture()
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.DEBUG)
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        _FakeTradingNode.instances = []
        mp.setattr(live_node, "TradingNode", _FakeTradingNode)
        p = StatePersistence(db_path=db, trader_id="PAPER-007")
        p.save_checkpoint(StateCheckpoint(trader_id="PAPER-007", node_status={
            "state": "halted", "halt_reason": "manual", "error_message": "halt via API", "risk": RISK}))
        p.close()
        corrupt_newest("positions", "null")
        out["S1"] = await start_once(mp, seeded)
        corrupt_newest("orders", "null")
        out["S2"] = await start_once(mp, seeded)
        corrupt_newest("node_status", "not json")
        out["S3"] = await start_once(mp, seeded)
        out["S4"] = await start_once(mp, seeded)
    out["LEAKS"] = {
        "log_marker_records": handler.hits,
        "ckpt_marker_rows": [r["id"] for r in rows() if MARK in (r["node_status"] or "")],
        "rows": len(rows()),
    }
    print("PROBE_JSON " + json.dumps(out, sort_keys=True))


asyncio.run(main())
'''


def run_rev(repo: Path, rev: str) -> dict[str, dict[str, object]]:
    """The chain's per-step record on `rev` (git archive of it, imported in a child)."""
    with tempfile.TemporaryDirectory(prefix="restart-probe-tree-") as tree:
        archive = subprocess.run(
            ["git", "-C", str(repo), "archive", rev], capture_output=True, check=True
        ).stdout
        subprocess.run(["tar", "-x", "-C", tree], input=archive, check=True)
        proc = subprocess.run(
            [sys.executable, "-c", CHILD], cwd=tree, capture_output=True, text=True,
            env={"PYTHONPATH": tree, "PATH": "/usr/bin:/bin", "VIBE_QUANT_DB": f"{tree}/none.db"},
        )  # fmt: skip
    for line in proc.stdout.splitlines():
        if line.startswith("PROBE_JSON "):
            record: dict[str, dict[str, object]] = json.loads(line.removeprefix("PROBE_JSON "))
            return record
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-12:])
    raise RuntimeError(f"child for {rev} produced no result (rc={proc.returncode}):\n{tail}")


def compare(results: dict[str, dict[str, dict[str, object]]]) -> list[str]:
    """Steps where some rev's record differs from the first rev's."""
    revs = list(results)
    return [s for s in STEPS if any(results[r][s] != results[revs[0]][s] for r in revs[1:])]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--rev", action="append", required=True, help="git rev; repeat (the first is the reference)")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--json", action="store_true", help="also print the raw records")
    args = parser.parse_args(argv)
    repo = Path(args.repo)
    results: dict[str, dict[str, dict[str, object]]] = {}
    for rev in args.rev:
        try:
            results[rev] = run_rev(repo, rev)
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"{rev}: {exc}", file=sys.stderr)
            return 2
    for step in STEPS:
        for rev, record in results.items():
            print(f"{step} {rev}: {json.dumps(record[step], sort_keys=True)}")
    differing = compare(results)
    for step in STEPS:
        print(f"{step} {'DIFFERS' if step in differing else 'IDENTICAL'}")
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    print(f"DIFFERS (first at {differing[0]})" if differing else "IDENTICAL")
    return 1 if differing else 0


if __name__ == "__main__":
    raise SystemExit(main())
