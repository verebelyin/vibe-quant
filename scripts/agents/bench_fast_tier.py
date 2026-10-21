"""Fast-tier speed benchmark (z8vqv perf beads): one seeded in-process 1m discovery.

Defaults are the z8vqv.2 benchmark: BTCUSDT 2026-06-01..07-01, pop 16 x 4 gens, seed 7,
pool KAMA,NATR,ROC,PRICE_POSITION, direction both, fast tier, on a scratch DB in /tmp.
Prints one JSON line: whole-run wall, median compute_signals seconds per genome-month,
and a sha256 digest of every evaluation (genome, trades, return, sharpe, PF, DD) in call
order -- two runs are number-identical iff the digests match.

    PYTHONPATH=$PWD .venv/bin/python scripts/agents/bench_fast_tier.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="KAMA,NATR,ROC,PRICE_POSITION")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", default="2026-06-01")
    ap.add_argument("--end", default="2026-07-01")
    ap.add_argument("--direction", default="both")
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--gens", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dump", default="", help="write every evaluation (JSON) here, to diff two runs")
    args = ap.parse_args(argv)

    tmp = Path(tempfile.mkdtemp(prefix="bench-fast-", dir="/tmp"))
    os.environ["VIBE_QUANT_DB"] = str(tmp / "state.db")
    os.environ["VIBE_QUANT_LOG_DIR"] = str(tmp / "logs")
    import importlib.util

    spec = importlib.util.spec_from_file_location("burst", REPO_ROOT / "scripts" / "agents" / "burst.py")
    assert spec is not None and spec.loader is not None
    burst = importlib.util.module_from_spec(spec)
    sys.modules["burst"] = burst
    spec.loader.exec_module(burst)

    import vibe_quant.fast_tier.signals as fast_signals
    from vibe_quant.discovery.backtest_fn import NTBacktestFn

    signal_s: list[float] = []
    original_signals = fast_signals.compute_signals

    def _timed(*a: Any, **kw: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return original_signals(*a, **kw)
        finally:
            signal_s.append(time.perf_counter() - t0)

    fast_signals.compute_signals = _timed  # imported per call in FastBacktestFn._run_single
    calls: list[list[Any]] = []
    original_call = NTBacktestFn.__call__

    def _logged(self: Any, chromosome: Any) -> Any:
        res = original_call(self, chromosome)
        calls.append([
            repr(chromosome), int(res.get("total_trades", 0)), repr(res.get("total_return")),
            repr(res.get("sharpe_ratio")), repr(res.get("profit_factor")), repr(res.get("max_drawdown")),
        ])
        return res

    cell = {
        "pair": args.pool.split(","), "direction": args.direction, "symbol": args.symbol,
        "start": args.start, "end": args.end, "leverage": 10, "seed": args.seed,
        "eval_windows": 1, "pop": args.pop, "gens": args.gens, "tier": "fast",
    }
    t0 = time.perf_counter()
    NTBacktestFn.__call__ = _logged  # type: ignore[method-assign]
    try:
        burst.run_cell(cell, tmp / "result.json")
    finally:
        NTBacktestFn.__call__ = original_call  # type: ignore[method-assign]
    wall = time.perf_counter() - t0
    result = json.loads((tmp / "result.json").read_text())
    errors = sum(1 for c in calls if c[1] == 0 and c[3] == repr(-1.0))
    if result.get("rc") != 0 or not calls or result.get("n_err"):
        print(f"bench: workload failed (rc={result.get('rc')} evals={len(calls)} "
              f"n_err={result.get('n_err')} sentinel_rows={errors})", file=sys.stderr)
        return 1
    if args.dump:
        Path(args.dump).write_text(json.dumps(calls, indent=1))
    print(json.dumps({
        "wall_s": round(wall, 2),
        "genome_months": len(signal_s),
        "signal_median_s": round(statistics.median(signal_s), 4) if signal_s else None,
        "signal_total_s": round(sum(signal_s), 2),
        "evals": len(calls),
        "digest": hashlib.sha256(json.dumps(calls).encode()).hexdigest()[:16],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
