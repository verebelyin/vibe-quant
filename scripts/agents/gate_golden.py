"""Golden digest of every champion gate stage (bead vibe-quant-rabmo.6, T5b).

    python scripts/agents/gate_golden.py --out PATH       # write the digest
    python scripts/agents/gate_golden.py --against PATH   # IDENTICAL (0) | diff (1)

Sections run in this fixed order inside one process (in-process pipeline runs
mutate ``operators.INDICATOR_POOL``):

    a0  stock mock GA (all rejected; legacy ``fitness_sharpe`` DSR input)
    a   the T5a fixture, single run (one rejection at every gate stage)
    b   the T5a fixture through the pooled multi-seed re-check (2 seeds)
    c   stored-notes DSR re-check on ``C_RUN_IDS`` (a ``.backup`` copy only)
    d   promote opposing-regime gate on every stored 1m short/both champion

The state DB is never opened read-write: section (c)/(d) run against a ``.backup``
copy in /tmp (opening the real file rw would migrate it).
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

_AGENTS_DIR = Path(__file__).resolve().parent
if str(_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENTS_DIR))

from gate_golden_fixture import (  # noqa: E402
    run_multi_seed,
    run_plain_mock,
    run_single,
    stage_counts,
    tag,
)

if TYPE_CHECKING:
    from vibe_quant.discovery.pipeline import DiscoveryResult

REPO_ROOT = Path(__file__).resolve().parents[2]
C_RUN_IDS = (877, 878, 949, 955, 970, 980, 981)
D_DIRECTIONS = ("short", "both")
D_CAP_SECONDS = 30.0  # if section (d) exceeds this, keep only the first 5 entries
D_CAP_ENTRIES = 5
EXPECTED_A_STAGES = {"cross_window": 1, "guardrails": 1, "holdout": 1, "wfa": 1}
EXPECTED_A_TAGS = [1]


class GateCoverageError(RuntimeError):
    """Section (a) no longer covers every gate stage with the pinned verdict."""


def digest(result: DiscoveryResult) -> dict[str, object]:
    """The refactor fingerprint of one run's gate outcome."""
    return {
        "top": [[chromosome.uid, tag(chromosome)] for chromosome, _ in result.top_strategies],
        "guardrail_rejections": result.guardrail_rejections,
        "dsr_records": result.dsr_records,
        "dsr_trials": result.dsr_trials,
        "holdout_min_trades": result.holdout_min_trades,
        "cross_window_results": [asdict(item) for item in result.cross_window_results],
        "wfa_results": [asdict(item) for item in result.wfa_results],
        "holdout_results": [asdict(item) for item in result.holdout_results],
        "buy_hold_gate": result.buy_hold_gate,
    }


def _jsonable(value: object) -> object:
    """A JSON-stable form: non-finite floats become their repr (NaN != NaN)."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _find_state_db(explicit: Path | None) -> Path | None:
    """The single-writer state DB: ``--db``, else the main checkout's (worktrees share it).

    ``None`` when this tree has neither a git common dir nor ``data/state`` (a
    copied tree): sections (c)/(d) then record that they were skipped.
    """
    if explicit is not None:
        return explicit
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        common = ""
    candidate = (Path(common).parent if common else REPO_ROOT) / "data/state/vibe_quant.db"
    return candidate if candidate.is_file() else None


def _backup(source: Path, target: Path) -> None:
    """``sqlite3 .backup`` semantics; the source is opened READ-ONLY (no migrations)."""
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(target)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


def _section_c(db_copy: Path) -> dict[str, object]:
    """The promote-time DSR re-check over stored notes (fail-closed rows included)."""
    from vibe_quant.overfitting.pipeline import OverfittingPipeline
    from vibe_quant.overfitting.types import FilterConfig

    pipeline = OverfittingPipeline(db_path=db_copy)
    config = FilterConfig(enable_wfa=False, enable_purged_kfold=False)
    out: dict[str, object] = {}
    for run_id in C_RUN_IDS:
        result = pipeline.run(run_id, config)
        out[str(run_id)] = [
            {
                "sweep_result_id": candidate.sweep_result_id,
                "passed_dsr": candidate.passed_dsr,
                "dsr_fail_reason": candidate.dsr_fail_reason,
                "p_value": (
                    None if candidate.dsr_result is None else candidate.dsr_result.p_value
                ),
            }
            for candidate in result.candidates
        ]
    return out


def _section_d(db_copy: Path) -> dict[str, object]:
    """The promote opposing-regime gate: "pass" or the 409 detail, per champion entry."""
    from fastapi import HTTPException

    from vibe_quant.api.routers.discovery import (
        _enforce_short_1m_cross_regime_gate,
        _get_discovery_run_config,
        _get_genome_entry,
    )
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery.record import DiscoveryRecord

    state = StateManager(db_copy)
    try:
        rows = state.conn.execute(
            "select id from backtest_runs"
            " where run_mode='discovery' and timeframe='1m' order by id"
        ).fetchall()
        outcomes: dict[str, str] = {}
        started = time.monotonic()
        for row in rows:
            run_id = int(row[0])
            record = DiscoveryRecord.load(state, run_id)
            discovery_run = _get_discovery_run_config(state, run_id)
            for index, entry in enumerate(record.top_strategies()):
                chromosome = entry.get("chromosome") if isinstance(entry, dict) else None
                direction = chromosome.get("direction") if isinstance(chromosome, dict) else None
                if direction not in D_DIRECTIONS:
                    continue
                try:
                    _enforce_short_1m_cross_regime_gate(
                        record, discovery_run, _get_genome_entry(record, index)
                    )
                except HTTPException as exc:
                    outcomes[f"{run_id}:{index}"] = str(exc.detail)
                else:
                    outcomes[f"{run_id}:{index}"] = "pass"
        if time.monotonic() - started > D_CAP_SECONDS:
            outcomes = dict(list(outcomes.items())[:D_CAP_ENTRIES])
        return outcomes
    finally:
        state.close()


def build(db_path: Path | None = None) -> dict[str, object]:
    """All sections, in the fixed order (one process, one digest)."""
    sections: dict[str, object] = {"a0": digest(run_plain_mock())}

    single = run_single()
    if stage_counts(single) != EXPECTED_A_STAGES or [
        tag(chromosome) for chromosome, _ in single.top_strategies
    ] != EXPECTED_A_TAGS:
        msg = (
            "section (a) lost its gate coverage: "
            f"stages={stage_counts(single)} tags="
            f"{[tag(chromosome) for chromosome, _ in single.top_strategies]}"
        )
        raise GateCoverageError(msg)
    sections["a"] = digest(single)
    sections["b"] = digest(run_multi_seed())

    state_db = _find_state_db(db_path)
    if state_db is None:
        sections["c"] = {"skipped": "no state db"}
        sections["d"] = {"skipped": "no state db"}
    else:
        with tempfile.TemporaryDirectory(prefix="gate_golden_") as tmp:
            db_copy = Path(tmp) / "vibe_quant.db"
            _backup(state_db, db_copy)
            sections["c"] = _section_c(db_copy)
            sections["d"] = _section_d(db_copy)
    return dict(_jsonable(sections))  # type: ignore[arg-type]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Golden digest of every champion gate stage.")
    parser.add_argument("--out", type=Path, default=None, help="write the digest JSON here")
    parser.add_argument("--against", type=Path, default=None, help="compare a digest with this one")
    parser.add_argument("--db", type=Path, default=None, help="state DB (default: main checkout's)")
    args = parser.parse_args(argv)

    try:
        current = build(args.db)
    except (RuntimeError, OSError, sqlite3.Error) as exc:
        print(f"gate_golden: {exc}", file=sys.stderr)
        return 2

    text = json.dumps(current, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    if args.against is None:
        return 0

    stored = json.loads(args.against.read_text())
    if stored == current:
        print("IDENTICAL")
        return 0
    stored_text = json.dumps(stored, indent=2, sort_keys=True) + "\n"
    diff = difflib.unified_diff(
        stored_text.splitlines(),
        text.splitlines(),
        fromfile=str(args.against),
        tofile="current",
        lineterm="",
    )
    print("\n".join(diff))
    return 1


if __name__ == "__main__":
    sys.exit(main())
