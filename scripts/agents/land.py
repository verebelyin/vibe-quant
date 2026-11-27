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
2. Main must hold no non-mirror commit the ``--verified`` branch lacks: paths taken from
   ``git diff <merge-base>..main`` are named and the land refuses (exit 2, nothing
   committed) — merge main into ``--verified``, re-verify that delta, then land. (A chief
   commit to main mid-job would otherwise surface only as the step-3 tree mismatch, after
   every task committed: wiki/env-gaps.md 2026-11-05, 2026-11-17.)
3. Each task's fork is printed (``land: <task> fork <sha> (<source>)``) and the whole land is
   rehearsed on a throwaway worktree at the start SHA (step 4, hooks off): any task that fails
   there is named with its fork and the land refuses (exit 2, no task committed). A wrong fork
   (a hand-edited ``.vq-base``) used to fail mid-land after earlier tasks had committed (job
   20261119-deferred5).
4. Each task's delta is ``git diff <fork> <task>`` applied with ``git apply --index --3way``
   and committed with ``git commit -F <message file>`` (the commit-msg hook checks it). The
   fork is ``merge-base(<main at start>, <task>)``, or the parent's tip for ``child@parent``
   (a stacked task). A two-dot ``git diff main <task>`` would revert what main gained since
   the fork. A path the 3-way apply leaves conflicted is first word-merged
   (``merge_words.py``); else it is taken from ``--verified`` with every later task's own
   delta to it undone (``git merge-file``, last task first), so each commit keeps only its
   own change. When an undo conflicts (a later task edits the same lines) or a blob is
   missing, the land fails as before.
5. The tree must then equal ``--verified`` (board/telemetry paths excluded); otherwise it
   prints the diff stat and exits 1: reset main to the printed start SHA (unpushed) and redo.
6. ``--push``: ``git pull --rebase --autostash`` then ``git push``.

Exit: 0 landed, 1 a step failed (state printed), 2 usage / refused precondition.
Replaces the per-job land scripts (5 ad-hoc ones in one session hit BSD-sed and
dirty-board failures, 2026-10-17).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
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


_CLAUDE_TRAILER = re.compile(
    r"^Co-Authored-By: Claude .*<noreply@anthropic\.com>\s*$", re.MULTILINE
)


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
    # absolute: the rehearsal commits from a throwaway worktree, where a relative path is gone
    return Task(branch, parent or None, Path(msg).resolve())


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
    return task_fork_source(repo, start, task)[0]


def task_fork_source(repo: Path, start: str, task: Task) -> tuple[str, str]:
    """(fork sha, where it came from). A stacked task forks at its parent's tip; else at the
    SHA worktree.sh recorded for its worktree (``.vq-base``; merge-base would swallow earlier
    tasks of a job-branch-based task) unless that SHA is an ancestor of a merge-base that is not
    the task's tip (a rebased branch); else at merge-base(start, branch)."""
    if task.parent:
        return git(repo, "rev-parse", task.parent).strip(), f"parent {task.parent}"
    recorded = recorded_base(repo, task.branch)
    base = git(repo, "merge-base", start, task.branch).strip()
    tip = git(repo, "rev-parse", task.branch).strip()
    if recorded and recorded != base and base != tip:
        # a branch rebased onto a newer main keeps its old .vq-base: an ancestor of the
        # merge-base is stale (its delta would re-apply what main already holds). A task
        # already merged into ``start`` has merge-base == its tip: its recorded base stands.
        stale = subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", recorded, base],
            capture_output=True,
        )
        if stale.returncode == 0:
            return base, "merge-base (stale .vq-base)"
    if recorded:
        return recorded, ".vq-base"
    return base, "merge-base"


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
            proc = subprocess.run(
                ["git", "merge-file", "-p", *files], capture_output=True, text=True
            )
            if proc.returncode != 0:
                return None
            text = proc.stdout
    return text


def land_task(
    repo: Path,
    fork: str,
    task: Task,
    later: dict[str, list[tuple[str, str]]],
    verified: str,
    rehearse: bool = False,
) -> str:
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
    if rehearse:  # messages were linted up front; the pre-commit lint runs here by hand
        _rehearse_lint(repo)
        git(
            repo,
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-q",
            "--no-verify",
            "-F",
            str(task.message),
        )
    else:
        git(repo, "commit", "-q", "-F", str(task.message))
    return git(repo, "rev-parse", "HEAD").strip()


def later_deltas(
    repo: Path, tasks: list[Task], forks: list[str]
) -> list[dict[str, list[tuple[str, str]]]]:
    """Per task i: path -> (fork, branch) of every task after it that changes that path."""
    names = [
        git(repo, "diff", "--name-only", fork, task.branch, "--", ".", *_EXCLUDE_MIRRORS).split()
        for task, fork in zip(tasks, forks, strict=True)
    ]
    laters: list[dict[str, list[tuple[str, str]]]] = []
    for i in range(len(tasks)):
        later: dict[str, list[tuple[str, str]]] = {}
        for j in range(i + 1, len(tasks)):
            for path in names[j]:
                later.setdefault(path, []).append((forks[j], tasks[j].branch))
        laters.append(later)
    return laters


_LINT_HOOK = Path(__file__).resolve().parents[1] / "git-hooks" / "pre-commit-lint.sh"
_RUFF = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "ruff"


def _rehearse_lint(wt: Path) -> None:
    """The real commit's pre-commit lint (ruff on the staged Python) on one task's staged
    delta: a task that only lints clean together with a later task (two tasks each adding the
    same def, ruff F811) is refused by the rehearsal, before any task commits on main."""
    env = {**os.environ, "VQ_RUFF": os.environ.get("VQ_RUFF", str(_RUFF))}
    proc = subprocess.run(
        ["bash", str(_LINT_HOOK)], cwd=wt, capture_output=True, text=True, env=env
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-20:])
        raise LandError(f"pre-commit lint failed on this task's own delta:\n{tail}")


def rehearsal_failures(
    repo: Path,
    start: str,
    tasks: list[Task],
    forks: list[tuple[str, str]],
    laters: list[dict[str, list[tuple[str, str]]]],
    verified: str,
) -> list[str]:
    """Land every task, in order, on a throwaway detached worktree at ``start``; a failing
    task is reset away and the rest go on. Returns ``<task> (fork <sha> <source>): <error>``
    per failing task; main is never touched."""
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        wt = Path(tmp) / "rehearsal"
        git(
            repo,
            "-c",
            "core.hooksPath=/dev/null",
            "worktree",
            "add",
            "-q",
            "--detach",
            str(wt),
            start,
        )
        try:
            for task, (fork, source), later in zip(tasks, forks, laters, strict=True):
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        land_task(wt, fork, task, later, verified, rehearse=True)
                except LandError as exc:
                    failures.append(f"{task.branch} (fork {fork[:12]} {source}): {exc}")
                    git(wt, "reset", "-q", "--hard", "HEAD")
        finally:
            git(repo, "worktree", "remove", "--force", str(wt), check=False)
    return failures


def tree_mismatch(repo: Path, verified: str) -> str:
    """Diff stat between HEAD and the verified branch (mirror paths excluded); "" = equal."""
    return git(repo, "diff", "--stat", verified, "HEAD", "--", ".", *_EXCLUDE_MIRRORS).strip()


def main_commits_the_verified_lacks(repo: Path, main: str, verified: str) -> list[str]:
    """Non-mirror paths main changed since ``merge-base(main, verified)``; [] when main holds
    no own commits (or verified already contains them). Landing them would leave the final
    tree ahead of ``--verified`` at the very end, after every task committed."""
    base = git(repo, "merge-base", main, verified).strip()
    return git(repo, "diff", "--name-only", base, main, "--", ".", *_EXCLUDE_MIRRORS).split()


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
        foreign = main_commits_the_verified_lacks(repo, start, args.verified)
        if foreign:
            raise LandError(
                f"main carries commits {args.verified} lacks: {', '.join(foreign)}; "
                f"merge main into {args.verified}, re-verify that delta, then land"
            )
        print(f"land: start {start}")
        forks = [task_fork_source(repo, start, task) for task in args.tasks]
        for task, (fork, source) in zip(args.tasks, forks, strict=True):
            print(f"land: {task.branch} fork {fork[:12]} ({source})")
        laters = later_deltas(repo, args.tasks, [fork for fork, _ in forks])
        failures = rehearsal_failures(repo, start, args.tasks, forks, laters, args.verified)
        if failures:
            print("land: rehearsal failed, no task committed:", file=sys.stderr)
            for failure in failures:
                print(f"  {failure}", file=sys.stderr)
            return 2
    except LandError as exc:
        print(f"land: {exc}", file=sys.stderr)
        return 2
    try:
        for task, (fork, _), later in zip(args.tasks, forks, laters, strict=True):
            sha = land_task(repo, fork, task, later, args.verified)
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
