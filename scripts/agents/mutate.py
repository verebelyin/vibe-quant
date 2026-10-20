#!/usr/bin/env python3
"""One-command mutation check: copy the tree, mutate one file, run the tests, restore.

    scripts/agents/mutate.py [--repo DIR] --file PATH --old TEXT --new TEXT
                             [--name NAME] [--rev SHA] [--timeout SEC] [--quiet-kills] -- <test command...>
    scripts/agents/mutate.py [--repo DIR] --spec mutants.json [--rev SHA] [--timeout SEC] -- <test command...>
    scripts/agents/mutate.py [--repo DIR] --auto-diff BASE [--max N] [--glob G] -- <test command...>

Worked example (the mutant is killed when the tests fail on it):

    scripts/agents/mutate.py --file vibe_quant/x.py --old 'a <= b' --new 'a < b' -- .venv/bin/python -m pytest tests/unit/test_x.py -q -x

`--repo` defaults to the current directory. One scratch copy is built per invocation with
`tempfile.mkdtemp(prefix="mutate-")`: every file git reports with
`git ls-files -z --cached --others --exclude-standard` is copied (so uncommitted edits are
what gets tested), and `<repo>/data/catalog` and `<repo>/data/archive` are symlinked in when
they exist (tests need market data). With `--rev SHA` the copy is instead built from
`git archive SHA` — the pinned commit, so the maker's uncommitted working tree is ignored.
The copy is NOT a git repo and the original repo is never written. `old` must occur exactly
once in the file (plain text, no regex, no partial match); the mutant is applied in the copy,
the test command runs there with cwd = copy, `PYTHONPATH` = copy and
`PYTHONDONTWRITEBYTECODE` = 1, and the file's original bytes are restored and verified
afterwards. (The bytecode flag is required, not cosmetic: CPython invalidates a .pyc by the
source's size and whole-second mtime, so two same-length mutants run within one second would
otherwise both load the FIRST mutant's cached bytecode — a mutant reported as killed or
survived that was never actually executed.) A relative executable in the test command that is
not in the copy (e.g. `.venv/bin/python`, which git never copies) is taken from `--repo`, or
from the main checkout (the parent of `git rev-parse --git-common-dir`) when `--repo` is a
task worktree, whose `.venv` lives there.

The test command runs once unmutated before any mutant; if it fails, the run aborts with
`ERROR baseline: tests fail without mutation` and exit 2. Each mutant prints exactly one
line:

    KILLED <name> (exit N, Xs)   tests failed on the mutant (good)
    SURVIVED <name> (Xs)         tests still passed on the mutant
    ERROR <name>: <why>          the mutant could not be applied (count != 1, missing file)
    INVALID <name>: test command exit N (collection/usage error, no assertion ran)
    TIMEOUT <name>               the test command exceeded --timeout (counts as killed)

When the test command runs pytest (a `pytest`/`py.test` argv element, so `-m pytest` counts)
exit 2, 3, 4 or 5 means the tests could not even be collected or the command was misused: no
assertion ran, so that is an invalid probe rather than a kill, and it is reported as INVALID
and counted as an error (exit 2 overall). Only exit 1 — tests ran and failed — is a kill.
Non-pytest commands are unchanged: any non-zero exit is a kill.

Each KILLED line of a pytest command is followed by `  first failure: <node id>: <first
assertion/error line>` (tells an assertion kill from an import/SQL error); `--quiet-kills` drops it.

`--auto-diff BASE [--max N] [--glob G]` generates the mutants instead: for each line ADDED
in `git diff BASE` of files matching `--glob` (default `vibe_quant/*.py`), the first
applicable of comparison flip (`<=`/`<`, `>=`/`>`, `==`/`!=`), boolean swap (`and`/`or`),
dropped `not`, or integer literal +1. A line none of those hit falls back to swapping its
first `True`/`False` (`flag`) and, when it is a complete single-line call or assignment that
starts at the line's indent (not `try: s.start()` or `else: y = 2`), replacing it with `pass`
at the same indent (`del`); docstrings, comments, blanks, `pass`,
imports, annotations, `...`, `return`/`raise`/`assert`, string assignments and meta lines
yield none. Each is one whole-line replacement (the line must be unique in its file), at most
`--max` (default 8), round-robin over files. Named `<file>:<line>:<op>`.
Makers picked easy mutants: 5 of 9 fix rounds in one job came from mutants a reviewer found.

`<name>` defaults to `<file>:<line of the match>` (just `<file>` when `old` is absent).
`--spec` takes a JSON list of `{"name", "file", "old", "new"}` objects. The last line is
`MUTANTS killed=K survived=S errors=E`; exit 0 when S=0 and E=0, 1 when anything survived,
2 on any error. The scratch copy is removed at the end, including on Ctrl-C or an exception.
Stdlib only.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
# comments, decorators, imports, logging/print calls and docstring lines carry no behaviour
_SKIP_LINE_RE = re.compile(r"^\s*(?:#|@|import |from \S+ import|logger\.|log\.|print\(|\"\"\"|''')")
_STRING_RE = re.compile(r"""(["'])(?:\\.|(?!\1).)*\1""")
_CMP_SWAP = {"<=": "<", "<": "<=", ">=": ">", ">": ">=", "==": "!=", "!=": "=="}
_OPERATORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cmp", re.compile(r"(?<![<>=!-])(<=|>=|==|!=|<|>)(?![<>=])")),
    ("bool", re.compile(r"\b(and|or)\b")),
    ("not", re.compile(r"\bnot\s+")),
    ("int", re.compile(r"(?<![\w.])(\d+)(?![\w.])")),
)
# fallbacks, only for an added line none of _OPERATORS hits
_FLAG_RE = re.compile(r"\b(True|False)\b")

TOOL_TIMEOUT = 300.0  # seconds; --timeout overrides
DATA_LINKS = ("data/catalog", "data/archive")
# pytest: 2 interrupted/collection error, 3 internal error, 4 usage error, 5 no tests collected
PYTEST_NO_ASSERTION = (2, 3, 4, 5)
PYTEST_NAMES = ("pytest", "py.test")


@dataclass(frozen=True)
class Mutant:
    name: str  # "" = derive `<file>:<line of the match>`
    file: str  # repo-relative
    old: str
    new: str


@dataclass(frozen=True)
class RunResult:
    exit_code: int | None  # None when the command could not be started
    seconds: float
    timed_out: bool = False
    error: str = ""
    output: str = ""


if TYPE_CHECKING:
    RunFn = Callable[[Sequence[str], Path, Mapping[str, str], float], RunResult]


def run_command(cmd: Sequence[str], cwd: Path, env: Mapping[str, str], timeout: float) -> RunResult:
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [str(c) for c in cmd],
            cwd=str(cwd),
            env=dict(env),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return RunResult(124, time.monotonic() - start, timed_out=True)
    except OSError as exc:  # missing binary
        return RunResult(None, time.monotonic() - start, error=f"cannot run {cmd[0]}: {exc}")
    return RunResult(proc.returncode, time.monotonic() - start, output=proc.stdout + proc.stderr)


def list_files(repo: Path) -> list[str]:
    """Every file git knows about, including uncommitted/untracked ones git does not ignore."""
    proc = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"git ls-files failed in {repo}: {detail}")
    return [os.fsdecode(name) for name in proc.stdout.split(b"\0") if name]


def build_copy(repo: Path, dest: Path) -> None:
    """Copy the working tree (never the index) into `dest`, then link in the data dirs."""
    for name in list_files(repo):
        src = repo / name
        dst = dest / name
        if not (src.is_file() or src.is_symlink()):  # tracked but deleted in the worktree
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(dst):
            dst.unlink()
        if src.is_symlink():
            dst.symlink_to(os.readlink(src))
        else:
            shutil.copy2(src, dst)
    link_data_dirs(repo, dest)


def link_data_dirs(repo: Path, dest: Path) -> None:
    for rel in DATA_LINKS:
        src = repo / rel
        dst = dest / rel
        if src.exists() and not os.path.lexists(dst):
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src.resolve())


def extract_rev(repo: Path, dest: Path, rev: str) -> None:
    """Write the files of commit `rev` into `dest` (`git archive`), then link in the data dirs."""
    archive = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", rev], capture_output=True
    )
    if archive.returncode != 0:
        detail = archive.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"git archive {rev} failed in {repo}: {detail}")
    untar = subprocess.run(
        ["tar", "-x", "-C", str(dest)], input=archive.stdout, capture_output=True
    )
    if untar.returncode != 0:
        detail = untar.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"tar -x failed in {dest}: {detail}")
    link_data_dirs(repo, dest)


def main_checkout(repo: Path) -> Path | None:
    """The main checkout when `repo` is a linked worktree — where its `.venv` actually lives."""
    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    common = proc.stdout.strip()
    return Path(common).parent if common else None


_ENV_TOKEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def normalize_command(cmd: Sequence[str]) -> tuple[list[str], dict[str, str]]:
    """Accept the shell forms agents type after ``--``: leading ``VAR=value`` tokens become
    env entries (``PYTHONPATH`` is dropped: the copy sets its own), and a bare test file or
    ``pytest`` runs as ``.venv/bin/python -m pytest`` (job 20261019-budget-dsr: both forms
    failed with "No such file" / "Permission denied")."""
    out = [str(c) for c in cmd]
    env: dict[str, str] = {}
    while out and _ENV_TOKEN.match(out[0]):
        key, _, value = out.pop(0).partition("=")
        if key != "PYTHONPATH":
            env[key] = value
    if out and (out[0] in PYTEST_NAMES or (out[0].endswith(".py") and Path(out[0]).name.startswith("test_"))):
        out = [".venv/bin/python", "-m", "pytest", *(out[1:] if out[0] in PYTEST_NAMES else out)]
    return out, env


def resolve_executable(cmd: Sequence[str], repo: Path, workdir: Path) -> list[str]:
    """Resolve a relative `<dir>/<exe>` in the test command against --repo, then the main
    checkout, when the copy does not have it: `.venv/bin/python` is gitignored, so the copy
    never contains it and a task worktree has no `.venv` of its own. The path is kept as
    found, never symlink-resolved: `.venv/bin/python` is a symlink to the base interpreter,
    and resolving it loses the venv's site-packages (pytest is then `No module named`)."""
    out = [str(c) for c in cmd]
    first = out[0]
    if os.sep not in first or (workdir / first).exists():
        return out
    root = repo if (repo / first).exists() else main_checkout(repo)
    if root is not None and (root / first).exists():
        out[0] = str(root / first)
    return out


def is_pytest_command(cmd: Sequence[str]) -> bool:
    """True when argv runs pytest (`-m pytest` puts `pytest` itself in argv, so its basename
    matches too)."""
    return any(Path(str(arg)).name in PYTEST_NAMES for arg in cmd)


_FAILED_RE = re.compile(r"^(?:FAILED|ERROR) (.+)$", re.MULTILINE)
_E_LINE_RE = re.compile(r"^E\s+(\S.*)$", re.MULTILINE)


def _split_node_detail(line: str) -> tuple[str, str | None]:
    """(node id, inline message) split at the first " - " that is OUTSIDE square brackets.

    A pytest id like ``test_value[with - separator]`` contains the " - " separator itself, so a
    plain first-occurrence split truncates the id and reports the id tail as the message.
    """
    depth = 0
    for i, ch in enumerate(line):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        elif ch == "-" and depth == 0 and line[i - 1 : i] == " " and line[i + 1 : i + 2] == " ":
            return line[: i - 1], line[i + 2 :]
    return line, None


def first_failure(output: str) -> str:
    """`  first failure: <node id>: <first assertion/error line>` from pytest output, or ""."""
    node = _FAILED_RE.search(output)
    if node is None:
        return ""
    name, inline = _split_node_detail(node.group(1))
    err = _E_LINE_RE.search(output)
    detail = err.group(1) if err else (inline or "")
    return f"  first failure: {name}: {detail.strip()}"


def check_mutant(
    workdir: Path,
    mutant: Mutant,
    cmd: Sequence[str],
    env: Mapping[str, str],
    timeout: float,
    run: RunFn,
    pytest_like: bool,
    show_failures: bool = True,
) -> tuple[bool, bool, str]:
    """Apply one mutant in the copy and run the tests. Returns (killed, is_error, line)."""
    path = workdir / mutant.file
    if not path.is_file():
        return False, True, f"ERROR {mutant.name or mutant.file}: no such file: {mutant.file}"
    original = path.read_bytes()
    try:
        text = original.decode("utf-8")
    except UnicodeDecodeError:
        return False, True, f"ERROR {mutant.name or mutant.file}: {mutant.file} is not utf-8"

    index = text.find(mutant.old)
    line_no = text.count("\n", 0, index) + 1 if index >= 0 else 0
    name = mutant.name or (f"{mutant.file}:{line_no}" if index >= 0 else mutant.file)

    count = text.count(mutant.old)
    if count != 1:
        return False, True, f"ERROR {name}: old text found {count} times in {mutant.file}"

    path.write_bytes(text.replace(mutant.old, mutant.new, 1).encode("utf-8"))
    try:
        result = run(cmd, workdir, env, timeout)
    finally:
        path.write_bytes(original)
    if path.read_bytes() != original:
        return False, True, f"ERROR {name}: could not restore {mutant.file}"

    if result.error:
        return False, True, f"ERROR {name}: {result.error}"
    if result.timed_out:
        return True, False, f"TIMEOUT {name}"
    seconds = f"{result.seconds:.1f}s"
    if result.exit_code == 0:
        return False, False, f"SURVIVED {name} ({seconds})"
    if pytest_like and result.exit_code in PYTEST_NO_ASSERTION:
        # pytest could not collect/start the tests: no assertion ran, so this is not a kill.
        return (
            False,
            True,
            f"INVALID {name}: test command exit {result.exit_code} "
            "(collection/usage error, no assertion ran)",
        )
    line = f"KILLED {name} (exit {result.exit_code}, {seconds})"
    failure = first_failure(result.output) if show_failures and pytest_like else ""
    return True, False, f"{line}\n{failure}" if failure else line


def load_spec(path: Path) -> list[Mutant]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("spec must be a JSON list of {name, file, old, new}")
    mutants: list[Mutant] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError(f"spec entry is not an object: {entry!r}")
        mutants.append(
            Mutant(
                name=str(entry.get("name") or ""),
                file=str(entry["file"]),
                old=str(entry["old"]),
                new=str(entry["new"]),
            )
        )
    return mutants


def _mutated(op: str, match: re.Match[str]) -> str:
    if op == "cmp":
        return _CMP_SWAP[match.group(1)]
    if op == "bool":
        return "or" if match.group(1) == "and" else "and"
    if op == "not":
        return ""
    if op == "flag":
        return "False" if match.group(1) == "True" else "True"
    return str(int(match.group(1)) + 1)


def added_lines(repo: Path, base: str, glob: str) -> list[tuple[str, int, str]]:
    """(file, line number, text) of every line `git diff base` adds in files matching glob."""
    out = subprocess.run(
        ["git", "-C", str(repo), "diff", "-U0", "--no-color", base, "--", glob],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    lines: list[tuple[str, int, str]] = []
    path, lineno = "", 0
    for raw in out.splitlines():
        if raw.startswith("+++ "):
            path = raw[6:] if raw.startswith("+++ b/") else ""
        elif (m := _HUNK_RE.match(raw)) is not None:
            lineno = int(m.group(1))
        elif raw.startswith("+") and path:
            lines.append((path, lineno, raw[1:]))
            lineno += 1
    untracked = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard", "--", glob],
        capture_output=True, text=True, check=True,
    ).stdout.split()  # fmt: skip
    for rel in untracked:  # a new file not committed yet: every line is added
        text = (repo / rel).read_text(encoding="utf-8", errors="replace")
        lines += [(rel, n, line) for n, line in enumerate(text.splitlines(), start=1)]
    return lines


def string_lines(source: str) -> set[int]:
    """Line numbers inside multi-line string constants (docstrings): prose, not behaviour."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    lines: set[int] = set()
    for node in ast.walk(tree):
        end = getattr(node, "end_lineno", None)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and end and end > node.lineno:
            lines.update(range(node.lineno, end + 1))
    return lines


def _is_str_constant(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _deletable(node: ast.stmt) -> bool:
    """A statement whose replacement by `pass` is a behaviour change: a call expression (not a
    bare constant or `...`), or an assignment that actually stores a non-string value."""
    if isinstance(node, ast.Expr):
        return isinstance(node.value, ast.Call)
    if isinstance(node, ast.Assign):
        return not _is_str_constant(node.value)
    if isinstance(node, ast.AnnAssign):
        return node.value is not None and not _is_str_constant(node.value)
    return isinstance(node, ast.AugAssign)


def deletable_lines(source: str) -> dict[int, str]:
    """{line number: leading indent} of the complete single-line statements a `del` mutant may
    replace with `pass`. Only a simple statement that starts at the line's indentation and ends
    on that same line qualifies: a statement spanning several lines (a multi-line call) has no
    whole-line replacement, two statements on one line (`a = 1; b = 2`) would need both
    deleted, and a simple statement sharing its line with a compound header (`try: s.start()`,
    `else: y = 2`, `if x: f()`, `with s: s.start()`) cannot be replaced without deleting the
    header too."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    lines = source.splitlines()
    whole: dict[int, bool] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue
        lineno = node.lineno
        end = getattr(node, "end_lineno", None)
        good = False
        if end is not None and end == lineno and 0 < lineno <= len(lines):
            line = lines[lineno - 1]
            good = _deletable(node) and node.col_offset == len(line) - len(line.lstrip())
        whole[lineno] = good if lineno not in whole else False
    found: dict[int, str] = {}
    for lineno, deletable in whole.items():
        if deletable:
            line = lines[lineno - 1]
            found[lineno] = line[: len(line) - len(line.lstrip())]
    return found


def diff_mutants(repo: Path, base: str, glob: str, limit: int) -> list[Mutant]:
    """At most ``limit`` whole-line mutants on added code lines, round-robin over files."""
    by_file: dict[str, list[Mutant]] = {}
    sources: dict[str, str | None] = {}
    prose: dict[str, set[int]] = {}
    deletable: dict[str, dict[int, str]] = {}
    for path, lineno, text in added_lines(repo, base, glob):
        if path not in sources:
            try:
                sources[path] = (repo / path).read_text(encoding="utf-8")
            except OSError:
                sources[path] = None
        source = sources[path]
        if source is None:
            continue
        if path.endswith(".py"):
            if path not in prose:
                prose[path] = string_lines(source)
                deletable[path] = deletable_lines(source)
            if lineno in prose[path]:
                continue
        # blank out string-literal contents (same length, so offsets still index `text`)
        code = _STRING_RE.sub(lambda q: q[0][0] + "_" * (len(q[0]) - 2) + q[0][-1], text)
        code = code.split("#", 1)[0]
        if not code.strip() or _SKIP_LINE_RE.match(text):
            continue
        if source.count(text) != 1:
            continue
        found: list[Mutant] = []
        for op, pattern in _OPERATORS:
            m = pattern.search(code)
            if m is not None:
                new = text[: m.start()] + _mutated(op, m) + text[m.end() :]
                found.append(Mutant(f"{path}:{lineno}:{op}", path, text, new))
                break
        if not found:
            flag = _FLAG_RE.search(code)
            if flag is not None:
                new = text[: flag.start()] + _mutated("flag", flag) + text[flag.end() :]
                found.append(Mutant(f"{path}:{lineno}:flag", path, text, new))
            indent = deletable.get(path, {}).get(lineno)
            if indent is not None:
                found.append(Mutant(f"{path}:{lineno}:del", path, text, indent + "pass"))
        by_file.setdefault(path, []).extend(found)
    picked: list[Mutant] = []
    queues = list(by_file.values())
    while len(picked) < limit and any(queues):
        for queue in queues:
            if queue and len(picked) < limit:
                picked.append(queue.pop(0))
    return picked


def main(argv: Sequence[str] | None = None, run: RunFn | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mutate.py", description="One-command mutation check")
    parser.add_argument("--repo", default=".", help="repo to copy (default: the current dir)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", help="repo-relative file to mutate")
    source.add_argument("--spec", help="JSON list of {name, file, old, new}")
    source.add_argument(
        "--auto-diff", metavar="BASE",
        help="generate mutants (flip/swap/drop/delete) on lines added since BASE",
    )
    parser.add_argument("--max", type=int, default=8, help="--auto-diff: mutant cap (default 8)")
    parser.add_argument("--glob", default="vibe_quant/*.py", help="--auto-diff: files to mutate")
    parser.add_argument("--old", help="text to replace; must occur exactly once")
    parser.add_argument("--new", help="replacement text")
    parser.add_argument("--name", default=None, help="mutant name (default <file>:<line>)")
    parser.add_argument(
        "--rev", default=None, help="build the copy from this committed rev (git archive)"
    )
    parser.add_argument("--timeout", type=float, default=TOOL_TIMEOUT, help="seconds per run")
    parser.add_argument(
        "--quiet-kills", action="store_true",
        help="do not print each kill's first failing test and assertion/error line",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="test command, after `--`")
    args = parser.parse_args(argv)

    cmd = list(args.command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("no test command: pass it after `--`")

    if args.file:
        if args.old is None or args.new is None:
            parser.error("--file needs --old and --new")
        mutants = [Mutant(args.name or "", args.file, args.old, args.new)]
    elif args.auto_diff:
        try:
            mutants = diff_mutants(Path(args.repo).resolve(), args.auto_diff, args.glob, args.max)
        except subprocess.CalledProcessError as exc:
            print(f"ERROR auto-diff: {(exc.stderr or '').strip()}")
            return 2
        if not mutants:
            print(f"ERROR auto-diff: no mutable added line in {args.glob} since {args.auto_diff}")
            return 2
    else:
        try:
            mutants = load_spec(Path(args.spec))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"ERROR spec: {exc}")
            return 2

    repo = Path(args.repo).resolve()
    runner: RunFn = run or run_command
    workdir = Path(tempfile.mkdtemp(prefix="mutate-"))
    try:
        try:
            if args.rev:
                extract_rev(repo, workdir, args.rev)
            else:
                build_copy(repo, workdir)
        except (OSError, RuntimeError) as exc:
            print(f"ERROR repo: {exc}")
            return 2

        env = dict(os.environ)
        env["PYTHONPATH"] = str(workdir)
        env["PYTHONDONTWRITEBYTECODE"] = "1"  # else a same-second, same-size mutation reuses a stale .pyc
        cmd, extra_env = normalize_command(cmd)
        env.update(extra_env)
        cmd = resolve_executable(cmd, repo, workdir)
        pytest_like = is_pytest_command(cmd)

        baseline = runner(cmd, workdir, env, args.timeout)
        if baseline.error or baseline.timed_out or baseline.exit_code != 0:
            print("ERROR baseline: tests fail without mutation")
            if baseline.timed_out:
                print(f"baseline timed out after {args.timeout:g}s (raise --timeout)", file=sys.stderr)
            detail = (baseline.error or baseline.output).strip()
            if detail:
                print(detail[-2000:], file=sys.stderr)
            return 2

        killed = survived = errors = 0
        for mutant in mutants:
            was_killed, is_error, line = check_mutant(
                workdir, mutant, cmd, env, args.timeout, runner, pytest_like, not args.quiet_kills
            )
            print(line)
            if is_error:
                errors += 1
            elif was_killed:
                killed += 1
            else:
                survived += 1
        print(f"MUTANTS killed={killed} survived={survived} errors={errors}")
        if errors:
            return 2
        return 1 if survived else 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
