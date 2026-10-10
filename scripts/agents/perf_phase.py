"""Phase timing / cProfile for one NTScreeningRunner eval. Never writes the state DB.

Times the phases of a single screening eval -- strategy compile, catalog query,
engine data load, the engine run itself and metric extraction -- and prints a
per-phase wall-time table per call (plus a cProfile top-N dump with --profile).

Usage:
    PYTHONPATH=<worktree> .venv/bin/python scripts/agents/perf_phase.py --sid 239 --calls 1 [--profile]

All catalog / data / DB paths are resolved from the repo root
(``Path(__file__).resolve().parents[2]``), so the script can run from any cwd.
The DB is opened read-only (``?mode=ro``); this script never writes the state DB.
"""

from __future__ import annotations

import argparse
import cProfile
import inspect
import io
import json
import os
import pstats
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "data" / "catalog"
ARCHIVE_DB_PATH = REPO_ROOT / "data" / "archive" / "raw_data.db"

TIMES: dict[str, float] = defaultdict(float)


def _resolve_db_path() -> Path:
    """Resolve the state DB read-only path, independent of the current cwd.

    Order: ``VIBE_QUANT_DB`` env override, then the repo-root state DB, then --
    for worktrees, which deliberately do not link ``data/state`` (single-writer
    DB) -- the main checkout's DB via ``git --git-common-dir``.
    """
    override = os.environ.get("VIBE_QUANT_DB")
    if override:
        return Path(override).expanduser().resolve()
    candidate = REPO_ROOT / "data" / "state" / "vibe_quant.db"
    if candidate.exists():
        return candidate
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return candidate
    main_db = Path(common).parent / "data" / "state" / "vibe_quant.db"
    return main_db if main_db.exists() else candidate


def _wrap(obj: object, name: str, label: str) -> None:
    """Time a method/function by monkeypatching it, accumulating into TIMES[label]."""
    orig = getattr(obj, name)

    def w(*args: Any, **kw: Any) -> Any:
        t = time.perf_counter()
        try:
            return orig(*args, **kw)
        finally:
            TIMES[label] += time.perf_counter() - t

    kind = inspect.getattr_static(obj, name)
    setattr(obj, name, staticmethod(w) if isinstance(kind, (classmethod, staticmethod)) else w)


def _load_dsl_config(sid: int | None, dsl_path: str | None, tf: str | None) -> dict[str, Any]:
    from vibe_quant.dsl.parser import validate_strategy_dict
    from vibe_quant.screening.pipeline import _dsl_to_dict

    if sid is not None:
        db_path = _resolve_db_path()
        if not db_path.exists():
            raise SystemExit(f"state DB not found: {db_path} (set VIBE_QUANT_DB to override)")
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT dsl_config FROM strategies WHERE id=?", (sid,)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise SystemExit(f"strategy {sid} not found in {db_path}")
        raw: Any = json.loads(row[0])
    else:
        if dsl_path is None:
            raise SystemExit("either --sid or --dsl is required")
        raw = json.loads(Path(dsl_path).read_text())
    if tf:
        raw["timeframe"] = tf
    return _dsl_to_dict(validate_strategy_dict(raw))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sid", type=int, help="strategy id to load from the state DB")
    ap.add_argument("--dsl", help="path to a DSL dict JSON when --sid is omitted")
    ap.add_argument("--tf", help="override the strategy timeframe")
    ap.add_argument("--sym", default="BTCUSDT")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2026-03-17")
    ap.add_argument("--calls", type=int, default=2)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    if not CATALOG_PATH.is_dir():
        print(f"perf_phase: catalog directory not found: {CATALOG_PATH}", file=sys.stderr)
        return 2
    if not ARCHIVE_DB_PATH.is_file():
        print(
            f"perf_phase: funding archive file not found: {ARCHIVE_DB_PATH} "
            "(expected the raw_data.db file, not the archive directory)",
            file=sys.stderr,
        )
        return 2

    t0 = time.perf_counter()
    from nautilus_trader.backtest.node import BacktestNode

    import vibe_quant.data.catalog as cat
    import vibe_quant.validation.extraction as ex
    from vibe_quant.screening import nt_runner as ntr
    from vibe_quant.screening.nt_runner import NTScreeningRunner

    print(f"imports {time.perf_counter() - t0:.2f}s")

    _wrap(NTScreeningRunner, "_ensure_compiled", "compile(+aux preflight)")
    _wrap(BacktestNode, "build", "node.build")
    _wrap(BacktestNode, "load_data_config", "data: catalog query")
    _wrap(BacktestNode, "_load_engine_data", "data: engine.add_data")
    _wrap(BacktestNode, "_run_oneshot", "node._run_oneshot")
    _wrap(NTScreeningRunner, "_extract_metrics", "extract_metrics")
    _wrap(ex, "mark_to_market_drawdown", "  (mtm_dd)")
    _wrap(BacktestNode, "dispose", "dispose")
    _wrap(cat, "cleanup_epoch_parquet", "cleanup_epoch_parquet")
    _wrap(ntr, "require_bars_in_window", "require_bars")

    dsl_dict = _load_dsl_config(a.sid, a.dsl, a.tf)
    runner = NTScreeningRunner(
        dsl_dict,
        [a.sym],
        a.start,
        a.end,
        catalog_path=str(CATALOG_PATH),
        funding_archive_path=str(ARCHIVE_DB_PATH),
    )

    pr: cProfile.Profile | None = None
    for i in range(a.calls):
        TIMES.clear()
        t = time.perf_counter()
        if a.profile and i == a.calls - 1:
            pr = cProfile.Profile()
            pr.enable()
            m = runner({})
            pr.disable()
        else:
            m = runner({})
        tot = time.perf_counter() - t
        print(f"call{i}: total {tot:.2f}s sharpe={m.sharpe_ratio!r} trades={m.total_trades}")
        for k, v in TIMES.items():
            print(f"   {k:28s} {v:7.3f}s")
        if "node._run_oneshot" in TIMES:
            derived = (
                TIMES["node._run_oneshot"] - TIMES["data: catalog query"] - TIMES["data: engine.add_data"]
            )
            print(f"   {'=> engine sort+run (derived)':28s} {derived:7.3f}s")
        excluded = (
            "node.build",
            "data: catalog query",
            "data: engine.add_data",
            "require_bars",
            "cleanup_epoch_parquet",
        )
        known = sum(v for k, v in TIMES.items() if not k.startswith(" ") and k not in excluded)
        print(f"   {'other (configs, NT glue)':28s} {tot - known:7.3f}s")

    if pr is not None:
        for key in ("tottime", "cumulative"):
            s = io.StringIO()
            pstats.Stats(pr, stream=s).sort_stats(key).print_stats(a.top)
            out = s.getvalue().splitlines()
            i0 = next((i for i, ln in enumerate(out) if "ncalls" in ln), None)
            if i0 is None:
                continue
            print(f"--- top {a.top} by {key}")
            print("\n".join(out[i0 - 1 : i0 + a.top + 1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
