#!/usr/bin/env python3
"""Land a verified job on main: one commit per task, each from its own fork point.

    scripts/agents/land.py --job <job> --verified <job branch> [--push] \\
        <task branch>=<message file> [<child branch>@<parent branch>=<message file>] ...

Run from the main checkout, on ``main``. Every message file must carry the chief's
``Co-Authored-By: Claude <model> <noreply@anthropic.com>`` line and pass the main-branch
commit-msg rules (``scripts/git-hooks/check_commit_msg.py``: Why / What / Verification, no
non-Claude co-author); all files are checked before the first commit and every failing file
is named at once (exit 2, nothing committed). In order:

1. Uncommitted board/telemetry mirror files (``docs/orchestration/board/``,
   ``docs/orchestration/telemetry/``) are committed first as ``orchestration: board mirror
   (job <job>)``; any other uncommitted change refuses the land (exit 2).
2. Each task's delta is ``git diff <fork> <task>`` applied with ``git apply --index --3way``
   and committed with ``git commit -F <message file>`` (the commit-msg hook checks it). The
   fork is ``merge-base(<main at start>, <task>)``, or the parent's tip for ``child@parent``
   (a stacked task). A two-dot ``git diff main <task>`` would revert what main gained since
   the fork. A path the 3-way apply leaves conflicted is first word-merged
   (``merge_words.py``); else it is taken from ``--verified`` with every later task's own
   delta to it undone (``git merge-file``, last task first), so each commit keeps only its
   own change. When an undo conflicts (a later task edits the same lines) or a blob is
   missing, the land fails as before.
3. The tree must then equal ``--verified`` (board/telemetry paths excluded); otherwise it
   prints the diff stat and exits 1: reset main to the printed start SHA (unpushed) and redo.
4. ``--push``: ``git pull --rebase --autostash`` then ``git push``.

Exit: 0 landed, 1 a step failed (state printed), 2 usage / refused precondition.
Replaces the per-job land scripts (5 ad-hoc ones in one session hit BSD-sed and
dirty-board failures, 2026-10-17).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "git-hooks"))
import check_commit_msg  # noqa: E402
import merge_words  # noqa: E402

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


_CLAUDE_TRAILER = re.compile(r"^Co-Authored-By: Claude .*<noreply@anthropic\.com>\s*$", re.MULTILINE)


def parse_task(spec: str) -> Task:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"task {spec!r}: expected <branch>[@<parent>]=<message file>"
        )
    ref, msg = spec.split("=", 1)
    branch, _, parent = ref.partition("@")
    if not branch or not Path(msg).is_file():
        raise argparse.ArgumentTypeError(f"task {spec!r}: branch empty or message file missing")
    if not _CLAUDE_TRAILER.search(Path(msg).read_text(encoding="utf-8", errors="replace")):
        raise argparse.ArgumentTypeError(
            f"task {spec!r}: {msg} lacks the chief's trailer line "
            "`Co-Authored-By: Claude <model> <noreply@anthropic.com>`"
        )
    return Task(branch, parent or None, Path(msg))


def message_problems(tasks: list[Task]) -> list[str]:
    """Every message file the main-branch commit-msg hook would reject, with its reason."""
    problems = []
    for task in tasks:
        text = task.message.read_text(encoding="utf-8", errors="replace")
        foreign = check_commit_msg.foreign_co_authors(text)
        if foreign:
            problems.append(f"{task.message}: non-Claude Co-Authored-By {'; '.join(foreign)}")
        lines = check_commit_msg.message_lines(text)
        subject = next((line.strip() for line in lines if line.strip()), "")
        missing = check_commit_msg.missing_headings(text)
        if missing and not check_commit_msg.EXEMPT_SUBJECT.match(subject):
            problems.append(f"{task.message}: missing {', '.join(missing)}")
    return problems


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


def recorded_base(repo: Path, branch: str) -> str:
    """The ``.vq-base`` of the worktree that has ``branch`` checked out, if it is an ancestor of it."""
    path = ""
    for line in git(repo, "worktree", "list", "--porcelain").splitlines():
        if line.startswith("worktree "):
            path = line[len("worktree ") :]
        elif line.strip() == f"branch refs/heads/{branch}" and path:
            try:
                sha = (Path(path) / ".vq-base").read_text().strip()
            except OSError:
                return ""
            if not sha:
                return ""
            ancestor = subprocess.run(
                ["git", "-C", str(repo), "merge-base", "--is-ancestor", sha, branch],
                capture_output=True,
            )
            return sha if ancestor.returncode == 0 else ""
    return ""


def task_fork(repo: Path, start: str, task: Task) -> str:
    """A stacked task forks at its parent's tip; else at the SHA worktree.sh recorded for its
    worktree (``.vq-base``; merge-base would swallow earlier tasks of a job-branch-based task);
    else at merge-base(start, branch)."""
    if task.parent:
        return git(repo, "rev-parse", task.parent).strip()
    recorded = recorded_base(repo, task.branch)
    if recorded:
        return recorded
    return git(repo, "merge-base", start, task.branch).strip()


def word_merged(repo: Path, path: str) -> bool:
    """Resolve ``path``'s conflict hunks word by word and stage it; False when it can't."""
    try:
        if not merge_words.resolve(repo / path):
            return False
    except (merge_words.MergeError, OSError):
        return False
    git(repo, "add", "--", path)
    return True


def _blob(repo: Path, rev: str, path: str) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"{rev}:{path}"], capture_output=True, text=True
    )
    return proc.stdout if proc.returncode == 0 else None


def verified_minus_later(
    repo: Path, verified: str, path: str, later: list[tuple[str, str]]
) -> str | None:
    """``verified:path`` with each later task's (fork, branch) delta to ``path`` undone, last
    task first; None when a blob is missing or an undo conflicts."""
    text = _blob(repo, verified, path)
    with tempfile.TemporaryDirectory() as tmp:
        for fork, branch in reversed(later):
            tip, base = _blob(repo, branch, path), _blob(repo, fork, path)
            if text is None or tip is None or base is None:
                return None
            files = []
            for name, body in (("current", text), ("tip", tip), ("base", base)):
                f = Path(tmp) / name
                f.write_text(body)
                files.append(str(f))
            # merge-file <current> <base> <other>: undo tip->base on top of current
            proc = subprocess.run(["git", "merge-file", "-p", *files], capture_output=True, text=True)
            if proc.returncode != 0:
                return None
            text = proc.stdout
    return text


def land_task(
    repo: Path, start: str, task: Task, later: dict[str, list[tuple[str, str]]], verified: str
) -> str:
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
        failure = LandError(
            f"git apply --index --3way - failed: {(applied.stderr or applied.stdout).strip()}"
        )
        conflicted = git(repo, "diff", "--name-only", "--diff-filter=U").split()
        if not conflicted:
            raise failure
        for path in conflicted:
            if word_merged(repo, path):
                print(f"land: {task.branch}: {path} conflicted, merged by words")
                continue
            if path in later:
                text = verified_minus_later(repo, verified, path, later[path])
                if text is None:
                    raise failure
                (repo / path).write_text(text)
                git(repo, "add", "--", path)
                print(
                    f"land: {task.branch}: {path} conflicted, took {verified} version "
                    f"minus {len(later[path])} later task(s)"
                )
                continue
            if _blob(repo, verified, path) is not None:
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
        problems = message_problems(args.tasks)
        if problems:
            print("land: message files the commit-msg hook would reject:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
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
            later: dict[str, list[tuple[str, str]]] = {}
            for other in args.tasks[i + 1 :]:
                other_fork = task_fork(repo, start, other)
                names = git(
                    repo, "diff", "--name-only", other_fork, other.branch, "--", ".", *_EXCLUDE_MIRRORS
                ).split()
                for path in names:
                    later.setdefault(path, []).append((other_fork, other.branch))
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
