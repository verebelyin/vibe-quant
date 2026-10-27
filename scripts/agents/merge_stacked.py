#!/usr/bin/env python3
"""Merge a stacked task branch into a job branch as ONE commit of its own delta.

    scripts/agents/merge_stacked.py <child branch> <parent branch>   (run inside the job worktree)

A child stacked on a parent that already landed on main (as a new squashed commit)
conflicts add/add on a plain ``git merge`` (the parent's original commits are not
ancestors of main). This applies ``git diff <parent> <child>`` with ``--3way`` and
commits it as ``merge <child> (delta over <parent>)`` (job 20261027-taker hit it twice).
"""

from __future__ import annotations

import subprocess
import sys


def merge_stacked(child: str, parent: str, cwd: str | None = None) -> int:
    diff = subprocess.run(
        ["git", "diff", "--binary", parent, child], cwd=cwd, check=True, capture_output=True
    ).stdout
    if not diff:
        print(f"merge_stacked: {child} adds nothing over {parent}")
        return 0
    applied = subprocess.run(
        ["git", "apply", "--index", "--3way"], cwd=cwd, input=diff, capture_output=True
    )
    if applied.returncode != 0:
        sys.stderr.write(applied.stderr.decode())
        return 1
    subprocess.run(
        ["git", "commit", "-q", "-m", f"merge {child} (delta over {parent})"], cwd=cwd, check=True
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(__doc__)
        return 2
    return merge_stacked(args[0], args[1])


if __name__ == "__main__":
    sys.exit(main())
