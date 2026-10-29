"""Real-catalog golden of BOTH tiers over liquidation / sizing / margin / fee / 1d cells.

Guards the trade-ledger / stats / launch-settings refactor (bead vibe-quant-rabmo): every cell
runs the screening runner and ValidationRunner on real data and stores the full results; a later
tree must reproduce them bit-identically. Cells are stored in the capture file together with
the DSL, symbols, window and launch params, so ``--check`` cannot drift from the capture.

    PYTHONPATH=$PWD .venv/bin/python scripts/agents/exactness_tiers.py --capture [PATH]
    PYTHONPATH=$PWD .venv/bin/python scripts/agents/exactness_tiers.py --check [PATH] [--jobs N]

The golden lives OUT of git (user decision): a bare ``--capture``/``--check`` resolves
``<main checkout>/data/swarm/_goldens/tiers.json`` at call time through git-common-dir, so a
linked worktree compares against the main checkout's copy and no checkout path is hard-coded.

``--jobs N`` runs the cells in a spawn process pool (default ``min(4, cpu_count)``; ``1`` is the
serial loop this script used to have). The cells are independent - each validation works in its
own TemporaryDirectory copy of the state DB - and the parent prints and compares them in CELLS
order, so the output is bit-identical either way.

Exit codes: 0 identical, 1 drift (first differing path per cell), 2 capture refused (an intended
count is 0 in either tier) or bad input.

Cell tuning record (strategy 239 DSL, BTCUSDT unless noted, 4h unless noted; screening counts
MEASURED by the architect, validation counts by the first capture, see CELLS):
  C3  isolated lev125 cap100 -> liquidations on 2024-01-01..2024-07-01.
  C4  cross lev50 cap50 risk0.5 -> 1 liquidation (cross lev125 cap100/125 gave 0).
  C5  BTC+ETH cap2 margin0.3 -> 2 refused entries.
  C7/C8  1d timeframe 2023-01-01..2024-07-01; C8 isolated lev125 cap100 (liq 5).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import logging
import multiprocessing
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.screening.pipeline import create_screening_pipeline
from vibe_quant.validation.runner import ValidationRunner

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

# Data paths are absolute from this script's repo root (a relative catalog path silently
# loads 0 events).
REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "data/catalog"
ARCHIVE_DB_PATH = REPO_ROOT / "data/archive/raw_data.db"
# The tier golden is kept OUT of git (user decision): in the MAIN checkout's data/swarm/_goldens/
# (data/swarm/ is gitignored). Resolved at call time, so a linked worktree reads main's copy.
GOLDEN_SUBDIR = ("data", "swarm", "_goldens")
GOLDEN_NAME = "tiers.json"
# Cells to run in parallel: 1 is the serial loop this script shipped with, 4 keeps the pool
# from crowding the host (the gate's --full run shares it with pytest).
JOBS_DEFAULT = min(4, os.cpu_count() or 1)
BASE_STRATEGY_ID = 239

# Runtime-only validation notes keys (wall-clock / tmp paths): none found by diffing two base
# captures (only execution_time_seconds and the position uuids, see _UUID_RE); listed here so a new one is a conscious decision.
VOLATILE_NOTES_KEYS: tuple[str, ...] = ()
# run_id = the source state DB's next id at capture time (the throwaway DB is a copy of it).
VOLATILE_RESULT_KEYS = ("execution_time_seconds", "run_id")

# Liquidation hit keys embed NT's random position uuid ("...Strategy-000-<uuid>@<ts>"); found by
# diffing two base captures (the only nondeterministic content besides execution_time_seconds).
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_REFUSED_RE = re.compile(r"margin limit refused (\d+) entries")


@dataclass(frozen=True)
class Cell:
    name: str
    symbols: tuple[str, ...]
    start: str
    end: str
    params: dict[str, Any] = field(default_factory=dict)
    timeframe: str | None = None  # None = strategy 239's own
    # Intended minimum counts; capture refuses when one is 0 in either tier.
    want: tuple[str, ...] = ()  # subset of liquidations / refusals / trades


_BTC = ("BTCUSDT",)
_PAIR = ("BTCUSDT", "ETHUSDT")
_H1 = ("2024-01-01", "2024-07-01")
_D1 = ("2023-01-01", "2024-07-01")

CELLS: tuple[Cell, ...] = (
    Cell("C1-defaults", _BTC, *_H1, {}, want=("trades",)),
    # Base has only fee tier VIP0 (vibe_quant/fees.py), so VIP3 is not available.
    Cell("C2-fee-bnb", _BTC, *_H1, {"fee_tier": "VIP0", "bnb_discount": True}, want=("trades",)),
    Cell("C3-isolated-125", _BTC, *_H1,
         {"liquidation_mode": "isolated", "leverage": 125, "max_position_pct": 100},
         want=("trades", "liquidations")),
    Cell("C4-cross-50", _BTC, *_H1,
         {"liquidation_mode": "cross", "leverage": 50, "max_position_pct": 50,
          "risk_per_trade": 0.5},
         want=("trades", "liquidations")),
    Cell("C5-margin-refusals", _PAIR, *_H1, {"max_position_pct": 2, "max_margin_pct": 0.3},
         want=("trades", "refusals")),
    Cell("C6-cap2-risk05", _PAIR, *_H1, {"max_position_pct": 2, "risk_per_trade": 0.05},
         want=("trades",)),
    Cell("C7-1d", _BTC, *_D1, {}, timeframe="1d", want=("trades",)),
    Cell("C8-1d-isolated-125", _BTC, *_D1,
         {"liquidation_mode": "isolated", "leverage": 125, "max_position_pct": 100},
         timeframe="1d", want=("trades", "liquidations")),
)


# ---- the one place that knows how the screening runner is built (rename = one line) ----
def _screening_runner(pipeline: Any) -> Any:
    return pipeline._runner


class _RefusalLog(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.counts: list[int] = []

    def emit(self, record: logging.LogRecord) -> None:
        m = _REFUSED_RE.search(record.getMessage())
        if m:
            self.counts.append(int(m.group(1)))


def _main_checkout(root: Path | None = None) -> Path:
    """The main checkout a worktree shares: parent of `git rev-parse --git-common-dir`."""
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True, text=True, check=True, cwd=REPO_ROOT if root is None else root,
    ).stdout.strip()
    return Path(common).parent


def default_golden(root: Path | None = None) -> Path:
    """`<main checkout>/data/swarm/_goldens/tiers.json` (outside git, see the module docstring)."""
    return _main_checkout(root).joinpath(*GOLDEN_SUBDIR, GOLDEN_NAME)


def _find_state_db() -> Path:
    # Worktrees have no data/state (single-writer DB): use the main checkout's, read-only.
    return _main_checkout() / "data/state/vibe_quant.db"


def _base_dsl() -> dict[str, Any]:
    db = _find_state_db()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    row = conn.execute(
        "SELECT dsl_config FROM strategies WHERE id = ?", (BASE_STRATEGY_ID,)
    ).fetchone()
    conn.close()
    if row is None:
        raise SystemExit(f"strategy {BASE_STRATEGY_ID} not found in {db}")
    loaded: dict[str, Any] = json.loads(row[0])
    return loaded


def _jsonable(value: Any) -> Any:
    """Plain-JSON form: dataclasses -> dicts, tuples -> lists; floats stay exact via repr."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, str):
        return _UUID_RE.sub("<UUID>", value)
    return value


def run_screening(cell: dict[str, Any]) -> dict[str, Any]:
    dsl = validate_strategy_dict(copy.deepcopy(cell["dsl"]))
    pipeline = create_screening_pipeline(
        dsl,
        use_mock=False,
        symbols=list(cell["symbols"]),
        start_date=cell["start"],
        end_date=cell["end"],
        catalog_path=str(CATALOG_PATH),
        parameters=cell["params"],
    )
    handler = _RefusalLog()
    nt_logger = logging.getLogger("vibe_quant.screening.nt_runner")
    old_level = nt_logger.level
    nt_logger.setLevel(logging.INFO)
    nt_logger.addHandler(handler)
    try:
        runner = _screening_runner(pipeline)
        metrics = runner({})
    finally:
        nt_logger.removeHandler(handler)
        nt_logger.setLevel(old_level)
    out = _jsonable(metrics)
    for key in VOLATILE_RESULT_KEYS:
        out.pop(key, None)
    return {
        "metrics": out,
        "liquidation_notes": _jsonable(getattr(runner, "last_liquidation_notes", None)),
        "margin_refusals": sum(handler.counts),
    }


def run_validation(cell: dict[str, Any], source_db: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="exactness-tiers-") as td:
        tmpdir = Path(td)
        db = tmpdir / "v.db"
        src = sqlite3.connect(str(source_db))
        dst = sqlite3.connect(str(db))
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
        conn = sqlite3.connect(str(db))
        sid = conn.execute(
            "INSERT INTO strategies(name, description, dsl_config, strategy_type) "
            "VALUES (?,?,?,?)",
            (f"tiers-golden-{cell['name']}", "exactness_tiers", json.dumps(cell["dsl"]), "golden"),
        ).lastrowid
        run_id = conn.execute(
            "INSERT INTO backtest_runs(strategy_id, run_mode, symbols, timeframe, "
            "start_date, end_date, parameters) VALUES (?,?,?,?,?,?,?)",
            (sid, "validation", json.dumps(list(cell["symbols"])), cell["dsl"]["timeframe"],
             cell["start"], cell["end"], json.dumps(cell["params"])),
        ).lastrowid
        conn.commit()
        conn.close()
        assert run_id is not None
        # ValidationRunner resolves catalog/archive paths relative to cwd.
        previous = Path.cwd()
        os.chdir(REPO_ROOT)
        runner = ValidationRunner(db_path=db, logs_path=tmpdir / "logs")
        try:
            result = runner.run(run_id)
        finally:
            runner.close()
            os.chdir(previous)
    out = _jsonable(result)
    for key in VOLATILE_RESULT_KEYS:
        out.pop(key, None)
    notes = out.get("notes")
    if isinstance(notes, dict):
        for key in VOLATILE_NOTES_KEYS:
            notes.pop(key, None)
    return {"result": out}


def _counts(screening: dict[str, Any], validation: dict[str, Any]) -> dict[str, dict[str, int]]:
    sm = screening["metrics"]
    vr = validation["result"]
    vnotes = vr.get("notes", {})
    vliq = vnotes.get("liquidation") or {}
    vliq_n = vliq.get("liquidation_count", len(vr.get("liquidation_events", []) or []))
    return {
        "screening": {
            "trades": int(sm["total_trades"]),
            "liquidations": int(sm.get("liquidation_count") or 0),
            "refusals": int(screening["margin_refusals"]),
        },
        "validation": {
            "trades": int(vr["total_trades"]),
            "liquidations": int(vliq_n or 0),
            "refusals": int((vnotes.get("margin_limit") or {}).get("refused_entries", 0)),
        },
    }


def first_diff(a: Any, b: Any, path: str = "$") -> str | None:
    if type(a) is not type(b):
        return f"{path}: {a!r} != {b!r}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}: missing in {'capture' if k not in a else 'current'}"
            d = first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: len {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            d = first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b or (a != a and b != b) else f"{path}: {a!r} != {b!r}"


def _roundtrip(obj: Any) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(json.dumps(obj)))


def _execute(cell: dict[str, Any], source_db: Path) -> dict[str, Any]:
    screening = run_screening(cell)
    validation = run_validation(cell, source_db)
    return _roundtrip({"screening": screening, "validation": validation})


def _execute_task(task: tuple[dict[str, Any], str]) -> dict[str, Any]:
    """One pool job: module level (picklable), so the spawn context can ship it."""
    spec, db = task
    return _execute(spec, Path(db))


def execute_cells(
    specs: Sequence[dict[str, Any]], source_db: Path, jobs: int = JOBS_DEFAULT
) -> Iterator[dict[str, Any]]:
    """One result per spec, in SPEC ORDER (the CELLS/file order), serial when jobs <= 1.

    The spawn context, not fork: a forked child would inherit this process's NautilusTrader /
    pandas state and the held-open state DB. Each cell is independent (run_validation copies the
    state DB into its own TemporaryDirectory), so the pool cannot move a number; the parent still
    prints and compares in order, which is what keeps the output bit-identical.
    """
    tasks = [(spec, str(source_db)) for spec in specs]
    if jobs <= 1 or len(tasks) <= 1:
        for task in tasks:
            yield _execute_task(task)
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(jobs, len(tasks)), mp_context=context) as pool:
        yield from pool.map(_execute_task, tasks)


def capture(path: Path, jobs: int = JOBS_DEFAULT) -> int:
    base = _base_dsl()
    source_db = _find_state_db()
    specs: list[dict[str, Any]] = []
    for c in CELLS:
        dsl = copy.deepcopy(base)
        if c.timeframe is not None:
            dsl["timeframe"] = c.timeframe
        validate_strategy_dict(copy.deepcopy(dsl))  # fail early on a bad DSL
        specs.append({
            "name": c.name, "dsl": dsl, "symbols": list(c.symbols),
            "start": c.start, "end": c.end, "params": c.params, "want": list(c.want),
        })
    cells: list[dict[str, Any]] = []
    refused = False
    for spec, res in zip(specs, execute_cells(specs, source_db, jobs), strict=True):
        counts = _counts(res["screening"], res["validation"])
        print(f"{spec['name']}: counts={json.dumps(counts)}")
        for tier, tc in counts.items():
            for key in spec["want"]:
                if tc[key] <= 0:
                    print(f"  REFUSED: {key} == 0 in {tier}")
                    refused = True
        cells.append({**spec, "counts": counts, "golden": res})
    if refused:
        print("capture refused: retune the cell(s) above and record why in the script")
        return 2
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cells": cells}, indent=1, sort_keys=True) + "\n")
    print(f"captured {len(cells)} cells -> {path}")
    return 0


def check(path: Path, jobs: int = JOBS_DEFAULT) -> int:
    if not path.exists():
        print(f"no capture file at {path}")
        return 2
    cells = json.loads(path.read_text())["cells"]
    source_db = _find_state_db()
    ok = True
    for spec, res in zip(cells, execute_cells(cells, source_db, jobs), strict=True):
        diff = first_diff(spec["golden"], res)
        print(f"{spec['name']}: {'IDENTICAL' if diff is None else 'DRIFT'}")
        if diff:
            print(f"  first diff {diff}")
            ok = False
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--capture", nargs="?", const="", metavar="PATH",
        help=f"write the golden (default: {GOLDEN_NAME} in the main checkout's data/swarm/_goldens)",
    )
    mode.add_argument(
        "--check", nargs="?", const="", metavar="PATH",
        help=f"compare against the golden (default: {GOLDEN_NAME} in the main checkout's "
             "data/swarm/_goldens)",
    )
    parser.add_argument(
        "--jobs", type=int, default=JOBS_DEFAULT, metavar="N",
        help="cells to run in parallel (1 = serial; default min(4, cpu count))",
    )
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.capture is not None:
        return capture(Path(args.capture) if args.capture else default_golden(), args.jobs)
    return check(Path(args.check) if args.check else default_golden(), args.jobs)


if __name__ == "__main__":
    sys.exit(main())
