#!/usr/bin/env python3
"""Append one row to a job's ledger.md (the chief's task log), creating the header once.

    scripts/agents/ledger.py --job <job dir> --id <agent id> --status TEXT
        [--runtime t3|claude|chief|-] [--model M] [--task T] [--bead B] [--worktree W]

Rows are append-only: a status change is a new row for the same id, never an edit
(chiefs spent ~180 Bash calls in 3 days hand-writing `printf ... >> ledger.md` rows and
`sed -i` status flips). Pipes in values are escaped so a row never splits a column.
mkbrief.py --route appends the "dispatched" row itself.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

COLUMNS = ("agent", "runtime", "model", "task", "bead", "worktree", "status")


def _cell(value: str | None) -> str:
    return (value or "-").replace("|", "\\|").replace("\n", " ").strip() or "-"


def append_row(job: Path, values: dict[str, str | None]) -> Path:
    """Append one row; the header is written when the file is new or empty."""
    path = job / "ledger.md"
    job.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or not path.read_text(encoding="utf-8").strip()
    with path.open("a", encoding="utf-8") as fh:
        if new:
            fh.write(f"# ledger {job.name}\n| {' | '.join(COLUMNS)} |\n|{'---|' * len(COLUMNS)}\n")
        fh.write("| " + " | ".join(_cell(values.get(c)) for c in COLUMNS) + " |\n")
    return path


def now_hm() -> str:
    return datetime.now(UTC).strftime("%m-%d %H:%MZ")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True, type=Path)
    ap.add_argument("--id", required=True, dest="agent")
    ap.add_argument("--status", required=True)
    for col in ("runtime", "model", "task", "bead", "worktree"):
        ap.add_argument(f"--{col}")
    args = ap.parse_args(argv)
    values = {c: getattr(args, c) for c in COLUMNS}
    values["status"] = f"{now_hm()} {args.status}"
    print(append_row(args.job, values))
    return 0


if __name__ == "__main__":
    sys.exit(main())
