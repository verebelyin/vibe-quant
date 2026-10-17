#!/usr/bin/env python3
"""Print exactly the review surface of a branch range in one command.

    review_pack.py <worktree> (--range BASE..HEAD | --merge-base PARENT [--head REF])
                   [--max-lines 600] [--full PATH]

Resolves both ends with `git rev-parse` and prints, in order:
    RANGE <base sha>..<head sha>          (always first)
    [WARN <end> is a ref; pinned to <sha>]  (--range only, per end that is not a 40-hex SHA)
    <git log --oneline base..head>
    <git diff --stat base..head>
    SKIPPED <path> (+a -d) generated/lockfile   (one line per excluded changed file)
    <unified diff -U5 base..head, exclusions applied, in review order>

`--merge-base PARENT` takes the base from `git merge-base PARENT HEAD` (HEAD = `--head` or the
worktree HEAD), so a stacked parent that gains commits mid-review cannot move the base out from
under the reviewer — the same reason `--range` warns when an end was typed as a ref instead of
a pinned SHA.

Excluded pathspecs (generated/lockfiles, never worth a reviewer's context): the lockfile/golden
names `*.lock`, `pnpm-lock.yaml`, `uv.lock`, `*.golden` at ANY depth, plus the rooted dirs
`frontend/openapi.json`, `frontend/src/api/generated/**`, `docs/orchestration/board/**`,
`docs/orchestration/telemetry/**`.

The printed diff is capped at `--max-lines` (600). The cut finishes the current file's hunk
header, then prints:
    TRUNCATED: <n> more diff lines in <files>; read one file with:
        git -C <worktree> diff <base>..<head> -- <file>
`--full PATH` additionally writes the whole uncapped diff (same exclusions) to PATH.

Diff sections are ordered: non-test source first, then tests, then docs. Exit 2 with a clear
message on stderr when a ref does not resolve or the range is empty.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from pathlib import Path

_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")

_EXCLUDES: tuple[tuple[str, str], ...] = (
    ("name", "*.lock"),
    ("name", "pnpm-lock.yaml"),
    ("name", "uv.lock"),
    ("name", "*.golden"),
    ("path", "frontend/openapi.json"),
    ("path", "frontend/src/api/generated/**"),
    ("path", "docs/orchestration/board/**"),
    ("path", "docs/orchestration/telemetry/**"),
)

_TEST_SUFFIXES = ("_test.py", ".test.ts", ".test.tsx", ".spec.ts", ".spec.tsx")


def _git(worktree: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", worktree, *args], capture_output=True, text=True, check=check
    )


def _resolve(worktree: str, ref: str) -> str | None:
    proc = _git(worktree, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False)
    sha = proc.stdout.strip()
    return sha if proc.returncode == 0 and sha else None


def _is_excluded(path: str) -> bool:
    """Mirror `_exclude_pathspecs` for a single path: names match at any depth."""
    name = path.rsplit("/", 1)[-1]
    for kind, pattern in _EXCLUDES:
        if kind == "name":
            if fnmatch.fnmatchcase(name, pattern):
                return True
        elif fnmatch.fnmatchcase(path, pattern):
            return True
    return False


def _exclude_pathspecs() -> list[str]:
    """One git pathspec per `_EXCLUDES` entry; name patterns match at any depth."""
    return [
        f":(exclude,glob)**/{pattern}" if kind == "name" else f":(exclude){pattern}"
        for kind, pattern in _EXCLUDES
    ]


def _tier(path: str) -> int:
    """0 = non-test source, 1 = tests, 2 = docs (the review order)."""
    if path.startswith("docs/") or path.endswith(".md"):
        return 2
    name = path.rsplit("/", 1)[-1]
    if (
        "/tests/" in path
        or path.startswith("tests/")
        or name.startswith("test_")
        or name.endswith(_TEST_SUFFIXES)
    ):
        return 1
    return 0


def _section_path(lines: list[str]) -> str:
    for line in lines:
        if line.startswith(("--- ", "+++ ")):
            raw = line[4:].strip()
            if raw == "/dev/null":
                continue
            return raw[2:] if raw.startswith(("a/", "b/")) else raw
    return ""


def _ordered_sections(diff_text: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    for part in re.split(r"(?m)^(?=diff --git )", diff_text):
        if not part.strip():
            continue
        lines = part.splitlines()
        path = _section_path(lines)
        if _is_excluded(path):
            continue
        sections.append((path, lines))
    sections.sort(key=lambda item: (_tier(item[0]), item[0]))
    return sections


def _cap(
    lines: list[str],
    bounds: list[tuple[str, int, int]],
    max_lines: int,
    worktree: str,
    base: str,
    head: str,
) -> tuple[list[str], str]:
    if len(lines) <= max_lines:
        return lines, ""
    cut = max_lines
    end_cut = cut
    current = next((b for b in bounds if b[1] <= cut - 1 < b[2]), None)
    if current is not None:
        _, start, end = current
        hunk = next((i for i in range(start, end) if lines[i].startswith("@@")), None)
        if hunk is not None and hunk >= cut:
            end_cut = hunk + 1
    shown = lines[:end_cut]
    dropped = [path for path, _, end in bounds if end > end_cut]
    first = dropped[0] if dropped else "?"
    message = (
        f"TRUNCATED: {len(lines) - end_cut} more diff lines in {', '.join(dropped)}; "
        f"read one file with: git -C {worktree} diff {base}..{head} -- {first}"
    )
    return shown, message


def _resolve_range(args: argparse.Namespace) -> tuple[str, str, list[tuple[str, str]]]:
    worktree = args.worktree
    warns: list[tuple[str, str]] = []
    if args.range_ is not None:
        typed_base, sep, typed_head = args.range_.partition("..")
        if not sep or not typed_base or not typed_head:
            raise ValueError(f"bad --range {args.range_!r}; expected BASE..HEAD")
        ends = ((typed_base, _resolve(worktree, typed_base)), (typed_head, _resolve(worktree, typed_head)))
        for typed, sha in ends:
            if sha is None:
                raise ValueError(f"{typed!r} does not resolve to a commit in {worktree}")
        for typed, sha in ends:
            if not _SHA_RE.match(typed):
                warns.append((typed, sha or ""))
        base, head = ends[0][1], ends[1][1]
    else:
        head = _resolve(worktree, args.head or "HEAD")
        if head is None:
            raise ValueError(f"{args.head or 'HEAD'!r} does not resolve to a commit in {worktree}")
        merge = _git(worktree, "merge-base", args.merge_base, head, check=False)
        base = merge.stdout.strip()
        if merge.returncode != 0 or not base:
            raise ValueError(f"no merge base between {args.merge_base!r} and {head} in {worktree}")
    assert base is not None and head is not None
    if base == head:
        raise ValueError(f"empty range {base}..{head} in {worktree}")
    return base, head, warns


def _render(args: argparse.Namespace, base: str, head: str, warns: list[tuple[str, str]]) -> int:
    worktree = args.worktree
    span = f"{base}..{head}"

    skipped = []
    for line in _git(worktree, "diff", "--numstat", span).stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and _is_excluded(parts[2]):
            skipped.append(f"SKIPPED {parts[2]} (+{parts[0]} -{parts[1]}) generated/lockfile")
    skipped.sort()

    diff_args = ["diff", "-U5", span, "--", "."]
    diff_args += _exclude_pathspecs()
    sections = _ordered_sections(_git(worktree, *diff_args).stdout)

    lines: list[str] = []
    bounds: list[tuple[str, int, int]] = []
    for path, section in sections:
        start = len(lines)
        lines.extend(section)
        bounds.append((path, start, len(lines)))

    if args.full:
        target = Path(args.full)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + ("\n" if lines else ""))

    out = [f"RANGE {span}"]
    out += [f"WARN {typed} is a ref; pinned to {sha}" for typed, sha in warns]
    for text in (_git(worktree, "log", "--oneline", span).stdout, _git(worktree, "diff", "--stat", span).stdout):
        if text.strip():
            out.append(text.rstrip("\n"))
    out += skipped
    shown, truncated = _cap(lines, bounds, args.max_lines, worktree, base, head)
    if shown:
        out.append("\n".join(shown))
    if truncated:
        out.append(truncated)
    print("\n".join(out))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="review_pack.py", description="Print exactly the review surface of a branch range."
    )
    parser.add_argument("worktree", help="path to the worktree to inspect")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--range", dest="range_", metavar="BASE..HEAD", help="explicit range")
    group.add_argument("--merge-base", metavar="PARENT", help="base = git merge-base PARENT HEAD")
    parser.add_argument("--head", metavar="REF", help="HEAD for --merge-base (default: worktree HEAD)")
    parser.add_argument("--max-lines", type=int, default=600, help="diff line cap (default 600)")
    parser.add_argument("--full", metavar="PATH", help="write the uncapped diff here")
    args = parser.parse_args(argv)

    if args.range_ is not None and args.head is not None:
        print("error: --head only applies to --merge-base", file=sys.stderr)
        return 2
    try:
        base, head, warns = _resolve_range(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        return _render(args, base, head, warns)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip() or f"exit {exc.returncode}"
        print(f"error: git {' '.join(exc.cmd[1:])} failed: {detail}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
