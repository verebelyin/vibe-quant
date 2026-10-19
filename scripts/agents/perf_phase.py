"""Phase timing / cProfile for one NTScreeningRunner eval. Never writes the state DB.

Times the phases of a single screening eval -- strategy compile, catalog query,
engine data load, the engine run itself and metric extraction -- and prints a
per-phase wall-time table per call (plus a cProfile top-N dump with --profile).

Usage:
    PYTHONPATH=<worktree> .venv/bin/python scripts/agents/perf_phase.py --sid 239 --calls 1 [--profile]

Random-genome mode (the perf-6n0n9-1 table, bead 93q2v): evaluate N seeded random
LONG genomes from an indicator pool on one window and print the same per-phase
table as shares + ``s/eval``:

    PYTHONPATH=<worktree> .venv/bin/python scripts/agents/perf_phase.py \
        --random-genomes 4 --pool full --tf 4h --start 2024-01-01 --end 2024-01-05

Each genome is evaluated on the TWO sub-windows of ``[start, end]`` (midpoint
split, ``min_trades=50``), exactly like ``perf-6n0n9/phase.py``: the wrapper labels
are kept byte-identical so a new row compares with that table. The loop is
in-process -- one worker, no pool -- so it can be measured beside a running gate.

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
import random
import resource
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vibe_quant.discovery.operators import StrategyChromosome

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "data" / "catalog"
ARCHIVE_DB_PATH = REPO_ROOT / "data" / "archive" / "raw_data.db"

TIMES: dict[str, float] = defaultdict(float)

#: The GA's per-window trade gate in the perf-6n0n9-1 harness (phase.py), kept identical.
PERF_MIN_TRADES = 50
#: phase.py's RNG seed: the same seed draws the same population, so rows compare.
PERF_SEED = 11
#: phase.py's derived rows, kept byte-identical (see share_lines).
ENGINE_ROW = "engine sort+run(+indicators)"
OTHER_ROW = "other (glue,chromosome->dsl)"
OTHER_PHASES = ("compile", "node.build", "run_oneshot(incl data)", "extract_metrics", "dispose")


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


def split_windows(start: str, end: str) -> list[tuple[str, str]]:
    """The two eval sub-windows of ``[start, end]`` split at the midpoint (phase.py's way)."""
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    mid = first + (last - first) / 2
    return [(start, mid.isoformat()), (mid.isoformat(), end)]


def apply_pool(pool: str, tf: str) -> list[str] | None:
    """Restrict ``operators.INDICATOR_POOL`` exactly as a discovery run's pool filter does.

    ``full`` passes ``None`` (no restriction); otherwise the comma list is the discovery
    ``--indicator-pool`` set. Returns the names for the CFG line (None = full pool).
    """
    from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline

    names = None if pool.strip() == "full" else [p.strip() for p in pool.split(",") if p.strip()]
    DiscoveryPipeline(
        DiscoveryConfig(indicator_pool=names, timeframe=tf), lambda *a, **k: {}
    )._apply_indicator_pool_filter()
    return names


def random_genomes(n: int, pool: str, seed: int, tf: str) -> list[StrategyChromosome]:
    """N seeded random LONG genomes drawn AFTER the pool filter, like phase.py.

    The filter runs first and the seed second: the draw must see only the requested pool,
    and the same (seed, n) must reproduce the same population on any machine.
    """
    from vibe_quant.discovery.operators import Direction, initialize_population

    apply_pool(pool, tf)
    random.seed(seed)
    return initialize_population(n, direction_constraint=Direction.LONG)


def make_backtest_fn(sym: str, tf: str, start: str, end: str) -> Any:
    """The GA's multi-window backtest callable: two sub-windows, min_trades=50, fill ticks."""
    from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH
    from vibe_quant.discovery.backtest_fn import NTBacktestFn
    from vibe_quant.screening.pipeline import resolve_fill_ticks_for

    fn = NTBacktestFn(
        [sym], tf, start, end, windows=split_windows(start, end), min_trades=PERF_MIN_TRADES
    )
    fn.fill_ticks = resolve_fill_ticks_for(tf, {tf}, [sym], start, end, DEFAULT_CATALOG_PATH)
    return fn


def phase_share_lines(times: dict[str, float], total: float, n: int) -> list[str]:
    """phase.py's table rows: per-phase s/eval + share, the derived engine row, the glue row."""
    lines = [
        f"  {name:26s} {value / n:7.3f} s/eval  {100 * value / total:5.1f}%"
        for name, value in times.items()
    ]
    engine = (
        times.get("run_oneshot(incl data)", 0.0)
        - times.get("data:catalog_query", 0.0)
        - times.get("data:add_data", 0.0)
    )
    lines.append(f"  {ENGINE_ROW:26s} {engine / n:7.3f} s/eval  {100 * engine / total:5.1f}%")
    known = sum(times.get(phase, 0.0) for phase in OTHER_PHASES)
    other = total - known
    lines.append(f"  {OTHER_ROW:26s} {other / n:7.3f} s/eval  {100 * other / total:5.1f}%")
    return lines


def _print_profile(pr: cProfile.Profile, top: int) -> None:
    for key in ("tottime", "cumulative"):
        stream = io.StringIO()
        pstats.Stats(pr, stream=stream).sort_stats(key).print_stats(top)
        out = stream.getvalue().splitlines()
        i0 = next((i for i, line in enumerate(out) if "ncalls" in line), None)
        if i0 is None:
            continue
        print(f"--- top {top} by {key}")
        print("\n".join(out[i0 - 1 : i0 + top + 1]))


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


def _install_wrappers() -> None:
    """Single-eval mode: import the stack and time the screening phases."""
    from nautilus_trader.backtest.node import BacktestNode

    import vibe_quant.data.catalog as cat
    import vibe_quant.validation.extraction as ex
    from vibe_quant.screening import nt_runner as ntr
    from vibe_quant.screening.nt_runner import NTScreeningRunner

    _wrap(NTScreeningRunner, "_ensure_compiled", "compile(+aux preflight)")
    _wrap(BacktestNode, "build", "node.build")
    _wrap(BacktestNode, "load_data_config", "data: catalog query")
    _wrap(BacktestNode, "_load_engine_data", "data: engine.add_data")
    _wrap(BacktestNode, "_run_oneshot", "node._run_oneshot")
    _wrap(NTScreeningRunner, "_extract_metrics", "extract_metrics")
    _wrap(ex, "mark_to_market_paths", "  (mtm_paths)")
    _wrap(BacktestNode, "dispose", "dispose")
    _wrap(cat, "cleanup_epoch_parquet", "cleanup_epoch_parquet")
    _wrap(ntr, "require_bars_in_window", "require_bars")


def _install_random_wrappers() -> None:
    """phase.py's wrapper set and labels, so the random-genome table compares row for row."""
    from nautilus_trader.backtest.node import BacktestNode

    from vibe_quant.screening import nt_runner as ntr
    from vibe_quant.screening.nt_runner import NTScreeningRunner

    _wrap(NTScreeningRunner, "_ensure_compiled", "compile")
    _wrap(BacktestNode, "build", "node.build")
    _wrap(BacktestNode, "load_data_config", "data:catalog_query")
    _wrap(BacktestNode, "_load_engine_data", "data:add_data")
    _wrap(BacktestNode, "_run_oneshot", "run_oneshot(incl data)")
    _wrap(NTScreeningRunner, "_extract_metrics", "extract_metrics")
    _wrap(BacktestNode, "dispose", "dispose")
    _wrap(ntr, "require_bars_in_window", "require_bars")


def _single_mode(a: argparse.Namespace) -> int:
    """One named strategy through NTScreeningRunner, `--calls` times."""
    t0 = time.perf_counter()
    _install_wrappers()
    print(f"imports {time.perf_counter() - t0:.2f}s")

    from vibe_quant.screening.nt_runner import NTScreeningRunner

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
        _print_profile(pr, a.top)
    return 0


def _random_mode(a: argparse.Namespace) -> int:
    """N seeded random LONG genomes, two sub-windows each: phase.py's table in one command."""
    t0 = time.perf_counter()
    _install_random_wrappers()
    print(f"imports {time.perf_counter() - t0:.2f}s")

    genomes = random_genomes(a.random_genomes, a.pool, a.seed, a.tf)
    fn = make_backtest_fn(a.sym, a.tf, a.start, a.end)
    fn(genomes[0])  # warm-up: imports, caches (phase.py's own first call)
    TIMES.clear()

    per: list[tuple[float, list[str]]] = []
    pr = cProfile.Profile() if a.profile else None
    cpu0, wall0 = time.process_time(), time.perf_counter()
    if pr is not None:
        pr.enable()
    for chrom in genomes:
        started = time.perf_counter()
        fn(chrom)
        genes = sorted({g.indicator_type for g in chrom.entry_genes + chrom.exit_genes})
        per.append((round(time.perf_counter() - started, 2), genes))
    if pr is not None:
        pr.disable()
    total = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    n = len(genomes)
    print(
        f"CFG tf={a.tf} pool={a.pool} win={a.start}..{a.end} n={n} total={total:.2f}s "
        f"per_eval={total / n:.3f}s cpu_per_eval={cpu / n:.3f}s "
        f"maxrss_MB={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.0f}"
    )
    for line in phase_share_lines(dict(TIMES), total, n):
        print(line)
    print("PER-GENOME " + json.dumps(per))
    if pr is not None:
        _print_profile(pr, a.top)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sid", type=int, help="strategy id to load from the state DB")
    ap.add_argument("--dsl", help="path to a DSL dict JSON when --sid is omitted")
    ap.add_argument("--tf", help="timeframe (override, or required with --random-genomes)")
    ap.add_argument("--sym", default="BTCUSDT")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2026-03-17")
    ap.add_argument("--calls", type=int, default=2)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument(
        "--random-genomes", type=int, default=0,
        help="phase-split mode: evaluate N seeded random LONG genomes from --pool",
    )
    ap.add_argument(
        "--pool", default="full", help="indicator pool of --random-genomes: 'full' or a comma list"
    )
    ap.add_argument("--seed", type=int, default=PERF_SEED, help="RNG seed of --random-genomes")
    a = ap.parse_args(argv)

    if a.random_genomes < 0:
        print("perf_phase: --random-genomes must be >= 1 when given", file=sys.stderr)
        return 2
    if a.random_genomes and not a.tf:
        print("perf_phase: --random-genomes needs --tf (the eval timeframe)", file=sys.stderr)
        return 2
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
    return _random_mode(a) if a.random_genomes else _single_mode(a)


if __name__ == "__main__":
    raise SystemExit(main())
