"""Seeded in-process discovery GA: memo A/B on CPU, evals, RSS and notes.

Runs one ``vibe_quant.discovery`` run in this process against a fresh scratch
DB under ``/tmp`` (``VIBE_QUANT_DB`` and ``VIBE_QUANT_LOG_DIR`` are set before
vibe_quant is imported), so the real state DB is never opened. Two invocations
with the same seed read the same workload, one with the memo off and one on:

    PYTHONPATH=<repo> .venv/bin/python scripts/agents/perf_ga.py \\
        --timeframe 4h --pool full --start 2024-03-01 --end 2024-06-01 \\
        --pop 16 --gens 4 --seed 7 --memo 0 --out /tmp/ga0.txt --notes-out /tmp/ga0.json
    ... --memo 1 --out /tmp/ga1.txt --notes-out /tmp/ga1.json
    cmp /tmp/ga0.txt /tmp/ga1.txt  # identical fitness => the memo changes no number

``--out`` logs one line per PARENT-process ``NTBacktestFn.__call__`` (the GA's
own evaluations plus the gate and full-range backtests), ``repr(chromosome) =>
repr(metrics)``. ``--notes-out`` writes the persisted run notes minus
``ind_memo`` (that key reports the memo and differs by design), so the two
runs' gating records compare byte for byte.

The printed line carries wall/CPU time, evaluations, peak RSS, the scratch DB
path and the memo's ``run_summary()``::

    GA memo=1 workers=1 wall=... cpu=... evals=58 maxrss_MB=... db=/tmp/... ind_memo={"hits": ...}
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SYMBOL = "BTCUSDT"


def _create_run(db: Path, timeframe: str, start: str, end: str) -> int:
    """Create the scratch run row and return its id."""
    from vibe_quant.db.state_manager import StateManager

    state = StateManager(db)
    try:
        return state.create_backtest_run(
            strategy_id=None,
            run_mode="discovery",
            symbols=[SYMBOL],
            timeframe=timeframe,
            start_date=start,
            end_date=end,
            parameters={},
        )
    finally:
        state.close()


def _persisted_notes(db: Path, run_id: int) -> dict[str, Any]:
    """The run's persisted notes without ``ind_memo`` (which differs by design)."""
    from vibe_quant.db.state_manager import StateManager

    state = StateManager(db)
    try:
        row = state.get_backtest_result(run_id) or {}
    finally:
        state.close()
    raw = row.get("notes")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    notes: dict[str, Any] = dict(parsed) if isinstance(parsed, dict) else {}
    notes.pop("ind_memo", None)
    return notes


def _discovery_argv(
    args: argparse.Namespace, run_id: int, db: Path
) -> list[str]:
    """The discovery CLI argv of one seeded run (phase.py's window split + gates)."""
    argv = [
        "discovery",
        "--run-id", str(run_id),
        "--db", str(db),
        "--symbols", SYMBOL,
        "--timeframe", args.timeframe,
        "--start-date", args.start,
        "--end-date", args.end,
        "--population-size", str(args.pop),
        "--max-generations", str(args.gens),
        "--convergence-generations", "10",
        "--eval-windows", "2",
        "--train-test-split", "0.8",
        "--direction", "long",
        "--max-workers", str(args.workers),
        "--seed", str(args.seed),
    ]
    if args.pool != "full":
        argv += ["--indicator-pool", args.pool]
    return argv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timeframe", required=True)
    ap.add_argument("--pool", default="full", help="'full' or a comma list (discovery --indicator-pool)")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--gens", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--memo", choices=("0", "1"), default="1")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", help="per-evaluation log (parent-process NTBacktestFn.__call__)")
    ap.add_argument("--notes-out", help="persisted run notes JSON minus ind_memo")
    args = ap.parse_args(argv)

    tmp = Path(tempfile.mkdtemp(prefix="perf-ga-", dir="/tmp"))
    db = tmp / "state.db"
    os.environ["VIBE_QUANT_DB"] = str(db)
    os.environ["VIBE_QUANT_LOG_DIR"] = str(tmp / "logs")
    os.environ["VIBE_QUANT_IND_MEMO"] = args.memo

    from vibe_quant.discovery.backtest_fn import NTBacktestFn
    from vibe_quant.dsl import indicator_memo

    run_id = _create_run(db, args.timeframe, args.start, args.end)
    log: list[str] = []
    original = NTBacktestFn.__call__

    def _logged(self: Any, chromosome: Any, *call_args: Any, **call_kwargs: Any) -> Any:
        result = original(self, chromosome, *call_args, **call_kwargs)
        log.append(repr(chromosome) + " => " + repr(result))
        return result

    NTBacktestFn.__call__ = _logged
    sys.argv = _discovery_argv(args, run_id, db)
    from vibe_quant.discovery.__main__ import main as discovery_main

    wall0, cpu0 = time.perf_counter(), time.process_time()
    try:
        rc = discovery_main()
    finally:
        NTBacktestFn.__call__ = original
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0

    if args.out:
        Path(args.out).write_text("\n".join(log) + ("\n" if log else ""))
    if args.notes_out:
        Path(args.notes_out).write_text(json.dumps(_persisted_notes(db, run_id), sort_keys=True))
    print(
        f"GA memo={args.memo} workers={args.workers} wall={wall:.1f}s cpu={cpu:.1f}s "
        f"evals={len(log)} maxrss_MB={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.0f} "
        f"db={db} ind_memo={json.dumps(indicator_memo.run_summary(), sort_keys=True)}"
    )
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
