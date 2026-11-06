#!/usr/bin/env python3
"""Prove a test rename/docstring/comment round did not touch test bodies.

    ast_same.py --base <ref> [--repo <dir>] [tests/**/*.py ...]

For each changed test module (or each path given) the base blob and the working-tree
file are parsed and normalised: every docstring (the first statement of a module,
class or function when it is a string constant) is removed, and every ``test_*``
function is renamed to ``test_#<index in file>``. The two normalised trees are then
compared with ``ast.dump(..., include_attributes=False)``.

Comments live in the token stream, not the AST, and docstrings plus names are
normalised away, so only a comment/docstring/name edit can leave the file SAME. A
changed assert value, a deleted test (or file) or any other body edit prints
``DIFF <file>: <first differing node path>`` and exits 1.

Without explicit paths the default set is every ``tests/**/*.py`` that differs from
``--base`` in the working tree, plus untracked test modules (reported ``NEW`` and
skipped: they have no base blob to compare).

Exit codes: 0 all SAME, 1 any DIFF, 2 the base ref or repo is unusable.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

TESTS_ROOT = "tests"
DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


class GitError(Exception):
    """The repo or the base ref could not be read."""


def _git(repo: Path, *args: str) -> str:
    """Run git, returning stdout; raise GitError with stderr on failure."""
    done = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        raise GitError(done.stderr.strip() or f"git {' '.join(args)} failed")
    return done.stdout


def _repo_root(repo: Path | None) -> Path:
    start = repo if repo is not None else Path.cwd()
    return Path(_git(start, "rev-parse", "--show-toplevel").strip())


def _verify_base(repo: Path, base: str) -> None:
    try:
        _git(repo, "rev-parse", "--verify", "--quiet", f"{base}^{{commit}}")
    except GitError as exc:
        raise GitError(f"unknown base ref {base!r}: {exc}") from exc


def _changed_paths(repo: Path, base: str) -> list[str]:
    """Changed and untracked ``tests/**/*.py`` paths, repo-relative and sorted."""
    # --no-renames: a moved file lists its source too, so the source fails as "missing at
    # head" instead of the destination passing as NEW with no comparison.
    tracked = _git(repo, "diff", "--no-renames", "--name-only", base, "--", TESTS_ROOT).splitlines()
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard", "--", TESTS_ROOT)
    paths = {p for p in [*tracked, *untracked.splitlines()] if p.endswith(".py")}
    return sorted(paths)


def _normalise_relative(repo: Path, path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve().relative_to(repo).as_posix()
    return candidate.as_posix()


def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, DOCSTRING_OWNERS) or not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            node.body.pop(0)


def _test_functions(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    ]
    return sorted(found, key=lambda node: (node.lineno, node.col_offset))


def _normalise(tree: ast.Module) -> ast.Module:
    _strip_docstrings(tree)
    for index, node in enumerate(_test_functions(tree)):
        node.name = f"test_#{index}"
    return tree


def _first_diff(base: object, head: object, path: str) -> str | None:
    """The first differing node path (types, fields, list entries and scalars)."""
    if isinstance(base, ast.AST) and isinstance(head, ast.AST):
        if type(base) is not type(head):
            return f"{path} ({type(base).__name__} vs {type(head).__name__})"
        for field in base._fields:
            diff = _first_diff(getattr(base, field), getattr(head, field), f"{path}.{field}")
            if diff:
                return diff
        return None
    if isinstance(base, list) and isinstance(head, list):
        if len(base) != len(head):
            return f"{path} ({len(base)} vs {len(head)} entries)"
        for index, (left, right) in enumerate(zip(base, head, strict=True)):
            diff = _first_diff(left, right, f"{path}[{index}]")
            if diff:
                return diff
        return None
    if base != head:
        return f"{path} ({base!r} != {head!r})"
    return None


def compare_source(
    base_source: str, head_source: str
) -> tuple[bool, int, int, str | None]:
    """Compare one module's base and head source.

    Returns ``(same, base_tests, head_tests, diff)``; ``diff`` is None when same or
    when the test-function count already differs (the count mismatch is the message).
    """
    base_tree = _normalise(ast.parse(base_source))
    head_tree = _normalise(ast.parse(head_source))
    base_tests = len(_test_functions(base_tree))
    head_tests = len(_test_functions(head_tree))
    if base_tests != head_tests:
        return False, base_tests, head_tests, None
    base_dump = ast.dump(base_tree, include_attributes=False)
    head_dump = ast.dump(head_tree, include_attributes=False)
    if base_dump == head_dump:
        return True, base_tests, head_tests, None
    return False, base_tests, head_tests, _first_diff(base_tree, head_tree, "Module")


def check_file(repo: Path, base: str, relpath: str) -> tuple[str, bool]:
    """Return ``(output line, ok)`` for one repo-relative path."""
    head_path = repo / relpath
    try:
        base_source = _git(repo, "show", f"{base}:{relpath}")
    except GitError:
        if not head_path.is_file():
            return f"DIFF {relpath}: missing at head", False
        return f"NEW {relpath}", True
    if not head_path.is_file():
        return f"DIFF {relpath}: missing at head", False
    same, base_tests, head_tests, diff = compare_source(base_source, head_path.read_text())
    if not same:
        if diff is None:
            return f"DIFF {relpath}: {base_tests} test functions at base, {head_tests} at head", False
        return f"DIFF {relpath}: {diff}", False
    return f"SAME {relpath}", True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="base commit/ref to compare against")
    parser.add_argument("--repo", default=None, help="repo to inspect (default: cwd's toplevel)")
    parser.add_argument("paths", nargs="*", help="repo-relative test paths (default: changed)")
    args = parser.parse_args(argv)
    try:
        repo = _repo_root(Path(args.repo) if args.repo else None)
        _verify_base(repo, args.base)
        paths = (
            [_normalise_relative(repo, p) for p in args.paths]
            if args.paths
            else _changed_paths(repo, args.base)
        )
    except (GitError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not paths:
        print("no test files changed", file=sys.stderr)
        return 0
    ok = True
    for relpath in paths:
        line, same = check_file(repo, args.base, relpath)
        print(line)
        ok = ok and same
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
