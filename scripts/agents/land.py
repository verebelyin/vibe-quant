#!/usr/bin/env python3
"""Land a verified job on main: one commit per task, each from its own fork point.

    scripts/agents/land.py --job <job> --verified <job branch> [--push] \\
        <task branch>=<message file> [<child branch>@<parent branch>=<message file>] ...

Run from the main checkout, on ``main``. In order:

1. Uncommitted board/telemetry mirror files (``docs/orchestration/board/``,
   ``docs/orchestration/telemetry/``) are committed first as ``orchestration: board mirror
   (job <job>)``; any other uncommitted change refuses the land (exit 2).
2. Each task's delta is ``git diff <fork> <task>`` applied with ``git apply --index --3way``
   and committed with ``git commit -F <message file>`` (the commit-msg hook checks it). The
   fork is ``merge-base(<main at start>, <task>)``, or the parent's tip for ``child@parent``
   (a stacked task). A two-dot ``git diff main <task>`` would revert what main gained since
   the fork. A path the 3-way apply leaves conflicted is taken from ``--verified`` when no
   later task in the list touches it; otherwise the land fails as before.
3. The tree must then equal ``--verified`` (board/telemetry paths excluded); otherwise it
   prints the diff stat and exits 1: reset main to the printed start SHA (unpushed) and redo.
4. ``--push``: ``git pull --rebase --autostash`` then ``git push``.

Exit: 0 landed, 1 a step failed (state printed), 2 usage / refused precondition.
Replaces the per-job land scripts (5 ad-hoc ones in one session hit BSD-sed and
dirty-board failures, 2026-10-17).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

MIRROR_PATHS = ("docs/orchestration/board/", "docs/orchestration/telemetry/")
_EXCLUDE_MIRRORS = [f":(exclude){p}" for p in MIRROR_PATHS]


@dataclass(frozen=True)
class Task:
    branch: str
    parent: str | None
    message: Path


class LandError(Exception):
    """A step failed; the message says what state main is in."""


def git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise LandError(f"git {' '.join(args)} failed: {(proc.stderr or proc.stdout).strip()}")
    return proc.stdout


def parse_task(spec: str) -> Task:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"task {spec!r}: expected <branch>[@<parent>]=<message file>"
        )
    ref, msg = spec.split("=", 1)
    branch, _, parent = ref.partition("@")
    if not branch or not Path(msg).is_file():
        raise argparse.ArgumentTypeError(f"task {spec!r}: branch empty or message file missing")
    return Task(branch, parent or None, Path(msg))


def dirty_paths(repo: Path) -> list[str]:
    out = git(repo, "status", "--porcelain", "--untracked-files=all")
    return [line[3:] for line in out.splitlines() if line.strip()]


def commit_mirrors(repo: Path, job: str) -> bool:
    """Commit dirty board/telemetry files; refuse any other uncommitted change."""
    dirty = dirty_paths(repo)
    other = [p for p in dirty if not p.startswith(MIRROR_PATHS)]
    if other:
        shown = ", ".join(other[:5])
        raise LandError(f"uncommitted changes outside the board mirror: {shown}")
    if not dirty:
        return False
    git(repo, "add", "--", *dirty)
    git(repo, "commit", "-q", "-m", f"orchestration: board mirror (job {job})")
    return True


def task_fork(repo: Path, start: str, task: Task) -> str:
    """A stacked task forks at its parent's tip; otherwise at merge-base(start, branch)."""
    if task.parent:
        return git(repo, "rev-parse", task.parent).strip()
    return git(repo, "merge-base", start, task.branch).strip()


def land_task(repo: Path, start: str, task: Task, later: set[str], verified: str) -> str:
    fork = task_fork(repo, start, task)
    diff = git(repo, "diff", "--binary", fork, task.branch, "--", ".", *_EXCLUDE_MIRRORS)
    if not diff.strip():
        raise LandError(f"{task.branch}: empty delta from {fork[:12]}")
    applied = subprocess.run(
        ["git", "-C", str(repo), "apply", "--index", "--3way", "-"],
        input=diff,
        capture_output=True,
        text=True,
    )
    if applied.returncode != 0:
        conflicted = git(repo, "diff", "--name-only", "--diff-filter=U").split()
        if not conflicted or any(path in later for path in conflicted):
            raise LandError(
                f"git apply --index --3way - failed: "
                f"{(applied.stderr or applied.stdout).strip()}"
            )
        for path in conflicted:
            exists = subprocess.run(
                ["git", "-C", str(repo), "cat-file", "-e", f"{verified}:{path}"],
                capture_output=True,
            ).returncode == 0
            if exists:
                git(repo, "checkout", verified, "--", path)
            else:
                git(repo, "rm", "-q", "-f", "--", path)
            print(
                f"land: {task.branch}: {path} conflicted, took {verified} version "
                "(no later task touches it)"
            )
    git(repo, "commit", "-q", "-F", str(task.message))
    return git(repo, "rev-parse", "HEAD").strip()


def tree_mismatch(repo: Path, verified: str) -> str:
    """Diff stat between HEAD and the verified branch (mirror paths excluded); "" = equal."""
    return git(repo, "diff", "--stat", verified, "HEAD", "--", ".", *_EXCLUDE_MIRRORS).strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--job", required=True)
    ap.add_argument("--verified", required=True, help="the job branch the verifier passed")
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--push", action="store_true")
    ap.add_argument("tasks", nargs="+", type=parse_task, metavar="BRANCH[@PARENT]=MSGFILE")
    args = ap.parse_args(argv)
    repo: Path = args.repo
    try:
        if git(repo, "symbolic-ref", "--quiet", "--short", "HEAD", check=False).strip() != "main":
            print("land: run on main in the main checkout", file=sys.stderr)
            return 2
        git(repo, "rev-parse", "--verify", f"{args.verified}^{{commit}}")
        if commit_mirrors(repo, args.job):
            print("land: committed the board mirror first")
        start = git(repo, "rev-parse", "HEAD").strip()
        print(f"land: start {start}")
    except LandError as exc:
        print(f"land: {exc}", file=sys.stderr)
        return 2
    try:
        for i, task in enumerate(args.tasks):
            later = {
                path
                for other in args.tasks[i + 1 :]
                for path in git(
                    repo,
                    "diff",
                    "--name-only",
                    task_fork(repo, start, other),
                    other.branch,
                    "--",
                    ".",
                    *_EXCLUDE_MIRRORS,
                ).split()
            }
            sha = land_task(repo, start, task, later, args.verified)
            print(f"land: {task.branch} -> {sha[:12]}")
        stat = tree_mismatch(repo, args.verified)
        if stat:
            print(f"land: tree differs from {args.verified}:\n{stat}", file=sys.stderr)
            print(f"land: reset main to {start} (unpushed) and redo", file=sys.stderr)
            return 1
        print(f"land: tree equals {args.verified}")
        if args.push:
            git(repo, "pull", "--rebase", "--autostash", "-q")
            git(repo, "push", "-q")
            print(git(repo, "status", "-sb").splitlines()[0])
    except LandError as exc:
        print(f"land: {exc}", file=sys.stderr)
        print(
            f"land: main started at {start}; inspect, then `git reset --hard {start}` to redo",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
