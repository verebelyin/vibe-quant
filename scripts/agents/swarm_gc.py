"""Reclaim disk in data/swarm: dry run by default, ``--apply`` deletes.

Deletes only what is throwaway by convention:
- every ``scratch/`` directory inside a job untouched for ``--days`` (scratch catalogs,
  state DBs and runs; 520 MB in one power job), and
- gate logs in ``_gate-logs/`` older than ``--days`` (only same-HEAD logs are counted).

Everything else is evidence (briefs, ledgers, handoffs, buses, burst ledgers, logs the
journal cites, raw downloads) and is only REPORTED: the biggest kept files of idle jobs, so
the user decides. The user had to ask how much was stored (5.7 GB, 3.9 GB removed by hand;
retro 2026-11-05).

    .venv/bin/python scripts/agents/swarm_gc.py [--days 7] [--apply]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SWARM = REPO_ROOT / "data" / "swarm"
REPORT_MIN_BYTES = 10 * 2**20


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _newest_mtime(job: Path) -> float:
    return max((f.stat().st_mtime for f in job.rglob("*") if f.is_file()), default=0.0)


def plan(root: Path, days: float, now: float) -> tuple[list[Path], list[Path]]:
    """(paths to delete, big kept files of idle jobs) under ``root``."""
    cutoff = now - days * 86400
    delete: list[Path] = []
    kept: list[Path] = []
    logs = root / "_gate-logs"
    if logs.is_dir():
        delete += sorted(f for f in logs.iterdir() if f.is_file() and f.stat().st_mtime < cutoff)
    for job in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if _newest_mtime(job) >= cutoff:
            continue  # an active job keeps its scratch
        scratch = sorted(d for d in job.rglob("scratch") if d.is_dir())
        scratch = [d for d in scratch if not any(p in scratch for p in d.parents)]
        delete += scratch
        kept += [
            f for f in job.rglob("*")
            if f.is_file() and f.stat().st_size >= REPORT_MIN_BYTES
            and not any(d in f.parents for d in scratch)
        ]
    return delete, sorted(kept, key=lambda f: -f.stat().st_size)


def _mb(n: int) -> str:
    return f"{n / 2**20:.0f}M"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=float, default=7, help="idle age before a job's scratch goes")
    ap.add_argument("--apply", action="store_true", help="delete (default: list only)")
    ap.add_argument("--root", type=Path, default=SWARM, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if not args.root.is_dir():
        print(f"swarm gc: no {args.root}")
        return 0
    total = _size(args.root)
    delete, kept = plan(args.root, args.days, time.time())
    freed = sum(_size(p) for p in delete)
    logs = sum(1 for p in delete if p.parent.name == "_gate-logs")
    print(
        f"swarm gc: data/swarm {_mb(total)}; {'deleting' if args.apply else 'would delete'}"
        f" {_mb(freed)} ({len(delete) - logs} scratch dirs, {logs} gate logs > {args.days:g} d)"
    )
    for p in delete:
        if p.parent.name != "_gate-logs":
            print(f"  scratch {_mb(_size(p)):>6} {p.relative_to(args.root)}")
    if kept:
        print(f"kept (evidence; delete by hand if done with it), files >= {_mb(REPORT_MIN_BYTES)}:")
        for f in kept[:15]:
            print(f"  {_mb(f.stat().st_size):>6} {f.relative_to(args.root)}")
    if args.apply:
        for p in delete:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
    elif delete:
        print("dry run: rerun with --apply to delete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
