#!/usr/bin/env python3
"""One-command mutation check: copy the tree, mutate one file, run the tests, restore.

    scripts/agents/mutate.py [--repo DIR] --file PATH --old TEXT --new TEXT
                             [--name NAME] [--rev SHA] [--timeout SEC] [--quiet-kills] -- <test command...>
    scripts/agents/mutate.py [--repo DIR] --spec mutants.json [--rev SHA] [--timeout SEC] -- <test command...>
    scripts/agents/mutate.py [--repo DIR] --auto-diff BASE [--max N] [--glob G] -- <test command...>
    scripts/agents/mutate.py [--repo DIR] --compare-rev BASE [--timeout SEC] -- <test command...>

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
`ERROR baseline: tests fail without mutation` and exit 2. A pytest baseline also prints
`BASELINE ran=N skipped=S deselected=D` from its final summary (`N passed`, `S skipped`,
`D deselected`; a missing count is 0), plus `WARN baseline: D tests deselected (-k/-m filter):
a survivor may only mean the filter missed the killing test` when D > 0 and `WARN baseline: S
tests skipped (e.g. needs_git in a --rev copy)` when S > 0: a `-k` filter that ran 2 of 159
tests, or a `--rev` copy whose needs_git tests skip, makes a survivor meaningless. Each
mutant prints exactly one line:

    KILLED <name> (assert, exit N, Xs)          an assertion failed on the mutant (good)
    KILLED <name> (crash <ExcType>, exit N, Xs) an exception escaped: a crash IS a kill
    SURVIVED <name> (Xs)                        tests still passed on the mutant
    ERROR <name>: <why>                         the mutant could not be applied (count != 1,
                                                missing file)
    INVALID <name>: <ExcType> from the mutant text (no behaviour tested)
    TIMEOUT <name>                              the test command exceeded --timeout (killed)

When the test command runs pytest (a `pytest`/`py.test` argv element, so `-m pytest` counts)
exit 2, 3, 4 or 5 means the tests could not even be collected or the command was misused: no
assertion ran, so that is an invalid probe rather than a kill, and it is reported as INVALID
and counted as an error (exit 2 overall). The exception is `-x` under xdist, which exits 2 on
a real failure: exit 2 whose output carries both a `FAILED <node>` line and `Interrupted:
stopping after` is classified exactly like exit 1. Only exit 1 (and that xdist exit 2) —
tests ran and failed — is a kill. Non-pytest commands are unchanged: any non-zero exit is a
kill, reported as `KILLED <name> (exit N, Xs)`.

The copy runs the test command with `PYTEST_ADDOPTS` dropped: the command is the spec, and an
outer harness's addopts (the pre-review gate exports `-q -rfE`, which at `-qq` hides pytest's
summary line, and a `-k`/`-m` there would silently filter the mutant run from the outside)
must not change what runs or what BASELINE reports.

A pytest failure whose first `E` line is `SyntaxError`, `IndentationError`, `ImportError` or
`ModuleNotFoundError`, or a `NameError: name 'X' is not defined` where X occurs as a word in
the mutant's `new` text but not its `old` text, broke the program TEXT and not its behaviour:
no assertion ever exercised it, so it is INVALID and counts as an error (a `--compare-rev`
side with no verdict). A runtime NameError that the mutant did not introduce, a ZeroDivision
and every other exception are kills. Each KILLED line of a pytest command is followed by `  first
failure: <node id>: <first assertion/error line>`; `--quiet-kills` drops it.

`--auto-diff BASE [--max N] [--glob G]` generates the mutants instead: for each line ADDED
in `git diff BASE` of files matching `--glob` (default `vibe_quant/*.py`), `line_mutants`
returns the line's whole-line replacements, the first applicable of comparison flip
(`<=`/`<`, `>=`/`>`, `==`/`!=`), boolean swap (`and`/`or`), dropped `not`, integer literal
+1, `min(`/`max(`, a method swap (`.min`/`.max`, `.idxmin`/`.idxmax`, `floor`/`ceil`,
`any`/`all`, `startswith`/`endswith`, `.head`/`.tail`), the first call keyword `k=v` ->
`k=None`, or a float literal x10 (`0.5` -> `5.0`). A line none of those hit falls back to
swapping its first `True`/`False` (`flag`) and, when it is a complete single-line call or
assignment that starts at the line's indent (not `try: s.start()` or `else: y = 2`),
replacing it with `pass` at the same indent (`del`); docstrings, comments, blanks, `pass`,
imports, annotations, `...`, `return`/`raise`/`assert`, string assignments and meta lines
yield none. Each is one whole-line replacement (the line must be unique in its file) named
`<file>:<line>:<op>`. `line_mutants(path, lineno, text, source, deletable_indent)` is public
and returns the mutants of ONE source line, so a later tool can mutate a covered line the
diff did not add. Candidates are ordered branchy first (comparison, boolean, `not`,
`if`/`elif`/`while`, `min`/`max`, a method swap), at most `--max` (default 8), and each file's
share of that budget is sampled at an even stride over that ordered list (first, last and
evenly between), not its first few lines. Round-robin over files stays.
Makers picked easy mutants: 5 of 9 fix rounds in one job came from mutants a reviewer found.

`--compare-rev BASE` answers "did this round LOSE a kill?" in one command: two copies are
built (one from `git archive BASE`, one from the working tree), the test command's baseline
must pass in BOTH (each phase prints its own `BASELINE`/`WARN` line behind a `base ` or
`head ` prefix), and every mutant runs twice. The line is
`base=<KILLED|SURVIVED> head=<KILLED|SURVIVED> <name>` (a TIMEOUT counts as KILLED, an
ERROR/INVALID makes the run an error), `LOST KILL` marks a mutant the base tree kills and the
working tree does not, the last line is `MUTANTS compared=N lost=K errors=E`, and the exit is
0 when nothing was lost, 1 on a lost kill, 2 on any error. `--rev` and `--compare-rev` are
mutually exclusive (one copy is pinned, or the pair is).

`<name>` defaults to `<file>:<line of the match>` (just `<file>` when `old` is absent).
`--spec` takes a JSON list of `{"name", "file", "old", "new"}` objects. The last line is
`MUTANTS killed=K survived=S errors=E`; exit 0 when S=0 and E=0, 1 when anything survived,
2 on any error. `--auto-diff` on a diff with NO candidate line under `--glob` (a
test-only round under the default `vibe_quant/*.py`) prints `MUTANTS no candidates` and exits
0, with the glob and base on stderr; nothing is copied and no baseline is run.
An xdist `-n N` / `-n auto` in the test command is lowered to the free cores (at least 2)
when the 1-minute load is above half the cores; stderr says so.
The scratch copy is removed at the end, including on Ctrl-C or an exception.
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
from decimal import Decimal
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
    ("int", re.compile(r"(?<![\w.])(?<![\d.][eE][-+])(\d+)(?![\w.])")),  # not 1e-3's exponent
)
# extra operators, tried in this order only when no _OPERATORS entry hits
_MINMAX_RE = re.compile(r"\b(min|max)\(")
_METHOD_SWAPS: tuple[tuple[re.Pattern[str], Mapping[str, str]], ...] = (
    (re.compile(r"\.(min|max)\("), {"min": "max", "max": "min"}),
    (re.compile(r"\.(idxmin|idxmax)\("), {"idxmin": "idxmax", "idxmax": "idxmin"}),
    (re.compile(r"\b(floor|ceil)\b"), {"floor": "ceil", "ceil": "floor"}),
    (re.compile(r"\b(any|all)\("), {"any": "all", "all": "any"}),
    (re.compile(r"\b(startswith|endswith)\("), {"startswith": "endswith", "endswith": "startswith"}),
    (re.compile(r"\.(head|tail)\("), {"head": "tail", "tail": "head"}),
)
_KWARG_RE = re.compile(r"[(,]\s*([A-Za-z_]\w*)\s*=(?!=)\s*([^,()]+?)\s*(?=[,)])")
_FLOAT_RE = re.compile(r"(?<![\w.])(\d+\.\d+(?:[eE][-+]?\d+)?|\d+(?:\.\d+)?[eE][-+]?\d+)(?![\w.])")
# fallbacks, only for an added line none of the operators above hits
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
_WORKER_FLAGS = ("-n", "--numprocesses")


def cap_workers(cmd: Sequence[str], load1: float, cpus: int) -> tuple[list[str], str | None]:
    """Lower an xdist ``-n N`` / ``-n auto`` to the free cores (at least 2) when other jobs
    already load the host past half its cores; returns the argv and a note (None = unchanged).
    Briefs said ``-n 6`` for mutant runs: 9 agents at once drove the load to 154 on 10 cores."""
    out = [str(c) for c in cmd]
    if load1 <= cpus / 2:
        return out, None
    cap = max(2, cpus - round(load1))
    for i, arg in enumerate(out):
        flag, eq, value = arg.partition("=")
        if eq and flag in _WORKER_FLAGS:
            index, prefix = i, f"{flag}="
        elif arg in _WORKER_FLAGS and i + 1 < len(out):
            index, prefix, value = i + 1, "", out[i + 1]
        elif arg.startswith("-n") and arg[2:].isdigit():
            index, prefix, value = i, "-n", arg[2:]
        else:
            continue
        if value.isdigit() and int(value) <= cap:
            return out, None
        out[index] = f"{prefix}{cap}"
        return out, f"mutate: xdist workers {value} -> {cap} (load {load1:.1f} on {cpus} cores)"
    return out, None


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


def first_error_line(output: str) -> str:
    """The text after `E ` on the first error line pytest printed, or ""."""
    err = _E_LINE_RE.search(output)
    return err.group(1).strip() if err else ""


def first_failure(output: str) -> str:
    """`  first failure: <node id>: <first assertion/error line>` from pytest output, or ""."""
    node = _FAILED_RE.search(output)
    if node is None:
        return ""
    name, inline = _split_node_detail(node.group(1))
    detail = first_error_line(output) or (inline or "")
    return f"  first failure: {name}: {detail}"


_BROKEN_TEXT_KINDS = ("SyntaxError", "IndentationError", "ModuleNotFoundError", "ImportError")
_NAME_ERROR_RE = re.compile(r"^NameError: name '([^']+)' is not defined\b")
_ASSERT_PREFIXES = ("AssertionError", "assert ", "Failed: DID NOT RAISE")
_EXC_NAME_RE = re.compile(r"^([A-Za-z_][\w.]*)")
_XDIST_INTERRUPTED = "Interrupted: stopping after"


def broken_text_error(mutant: Mutant, message: str) -> str | None:
    """The exception type when the mutant broke the PROGRAM TEXT, not the behaviour.

    A SyntaxError/IndentationError/ImportError (a module the mutant invented or misspelled), or
    a NameError for a name the mutant's own `new` text introduces: no assertion ever got to
    exercise that code path, so the run is no verdict at all (job 20261028-prune,
    rev-paperstart-1). A runtime NameError the mutant did NOT introduce is a real crash and
    stays a kill."""
    for kind in _BROKEN_TEXT_KINDS:
        if re.match(rf"^{kind}\b", message):
            return kind
    match = _NAME_ERROR_RE.match(message)
    if match is not None:
        introduced = re.compile(rf"\b{re.escape(match.group(1))}\b")
        if introduced.search(mutant.new) and not introduced.search(mutant.old):
            return "NameError"
    return None


def kill_kind(message: str) -> str:
    """`assert`, `crash <ExcType>` or bare `crash` from pytest's first `E` line."""
    if message.startswith(_ASSERT_PREFIXES):
        return "assert"
    match = _EXC_NAME_RE.match(message)
    if match is not None:
        name = match.group(1)
        if message[match.end() : match.end() + 1] == ":" or name.rsplit(".", 1)[-1].endswith(
            ("Error", "Exception", "Interrupt", "Exit", "Failure", "Timeout")
        ):
            return f"crash {name}"
    return "crash"


# pytest's final line: counts then the wall time ("1 skipped in 0.01s" has no "passed")
_SUMMARY_LINE = re.compile(r"(?:\b\d+ (?:passed|failed|skipped|deselected|errors?)\b|no tests ran).* in [\d.]+s\b")
_SUMMARY_COUNTS = (
    ("ran", re.compile(r"(\d+) passed\b")),
    ("skipped", re.compile(r"(\d+) skipped\b")),
    ("deselected", re.compile(r"(\d+) deselected\b")),
)


def baseline_report(output: str, prefix: str = "") -> list[str]:
    """`BASELINE ran/skipped/deselected` and the WARN lines a pytest baseline deserves.

    `ran` is pytest's `N passed`; a missing count is 0. A deselected count means a `-k`/`-m`
    filter hid tests, and a skipped count means the copy lost some (a `--rev` copy has no
    `.git`, so needs_git tests skip): either way a SURVIVED mutant proves far less than the
    bare line suggests (revlite-e4-busenv-1 ran 2 of 159 tests through a `-k` filter)."""
    summary = ""
    for line in reversed(output.splitlines()):
        if _SUMMARY_LINE.search(line):
            summary = line
            break
    counts = {
        key: int(match.group(1)) if (match := pattern.search(summary)) else 0
        for key, pattern in _SUMMARY_COUNTS
    }
    lines = [
        f"{prefix}BASELINE ran={counts['ran']} skipped={counts['skipped']} "
        f"deselected={counts['deselected']}"
    ]
    if counts["deselected"]:
        lines.append(
            f"{prefix}WARN baseline: {counts['deselected']} tests deselected (-k/-m filter): "
            "a survivor may only mean the filter missed the killing test"
        )
    if counts["skipped"]:
        lines.append(
            f"{prefix}WARN baseline: {counts['skipped']} tests skipped "
            "(e.g. needs_git in a --rev copy)"
        )
    return lines


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
    if pytest_like:
        # `-x` under xdist exits 2 on a real failure (a FAILED node and an Interrupted line),
        # so that exit 2 is classified exactly like exit 1.
        interrupted = (
            result.exit_code in PYTEST_NO_ASSERTION
            and _FAILED_RE.search(result.output) is not None
            and _XDIST_INTERRUPTED in result.output
        )
        if result.exit_code != 1 and not interrupted:
            # pytest could not collect/start the tests: no assertion ran, not a kill.
            return (
                False,
                True,
                f"INVALID {name}: test command exit {result.exit_code} "
                "(collection/usage error, no assertion ran)",
            )
        message = first_error_line(result.output)
        broken = broken_text_error(mutant, message)
        if broken is not None:
            return (
                False,
                True,
                f"INVALID {name}: {broken} from the mutant text (no behaviour tested)",
            )
        label = f"{kill_kind(message)}, exit {result.exit_code}"
    else:
        label = f"exit {result.exit_code}"
    line = f"KILLED {name} ({label}, {seconds})"
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


def _code_only(text: str) -> str:
    """`text` with string-literal contents (same length, so offsets still index the line) and
    any trailing comment blanked: prose and log messages are not behaviour."""
    code = _STRING_RE.sub(lambda q: q[0][0] + "_" * (len(q[0]) - 2) + q[0][-1], text)
    return code.split("#", 1)[0]


def _replace(text: str, start: int, end: int, replacement: str) -> str:
    return text[:start] + replacement + text[end:]


def _minmax(code: str) -> tuple[int, int, str] | None:
    """`min(` <-> `max(` as whole words."""
    match = _MINMAX_RE.search(code)
    if match is None:
        return None
    return match.start(1), match.end(1), "max" if match.group(1) == "min" else "min"


def _method(code: str) -> tuple[int, int, str] | None:
    """The first method swap in the table (`.min(`/`.max(`, `.idxmin(`/`.idxmax(`,
    `floor`/`ceil`, `any(`/`all(`, `startswith`/`endswith`, `.head(`/`.tail(`)."""
    for pattern, swaps in _METHOD_SWAPS:
        match = pattern.search(code)
        if match is not None:
            return match.start(1), match.end(1), swaps[match.group(1)]
    return None


def _kwnone(code: str) -> tuple[int, int, str] | None:
    """The first call keyword argument `name=value` -> `name=None`.

    A parameter default (`def f(a=1):`) is not a call keyword and is skipped. A `True`/`False`
    value is left to the `flag` fallback, which already covers it."""
    if re.match(r"^\s*(?:async\s+)?def\b", code):
        return None
    for match in _KWARG_RE.finditer(code):
        if match.group(2).strip() in ("True", "False"):
            continue
        return match.start(2), match.end(2), "None"
    return None


def _x10(literal: str) -> str:
    """`0.5` -> `5.0`, `0.05` -> `0.5`, `1.5e3` -> `15000.0`: exact decimal scaling, written
    back as a float literal (only fractional trailing zeros are trimmed)."""
    out = format(Decimal(literal) * 10, "f")
    if "." not in out:
        return out + ".0"
    out = out.rstrip("0")
    return out + "0" if out.endswith(".") else out


def _scale(code: str) -> tuple[int, int, str] | None:
    """The first nonzero float literal scaled by ten (an int literal is `int`'s job)."""
    for match in _FLOAT_RE.finditer(code):
        if Decimal(match.group(1)) != 0:
            return match.start(1), match.end(1), _x10(match.group(1))
    return None


_EXTRA_OPERATORS: tuple[tuple[str, Callable[[str], tuple[int, int, str] | None]], ...] = (
    ("minmax", _minmax),
    ("method", _method),
    ("kwnone", _kwnone),
    ("scale", _scale),
)


def line_mutants(
    path: str, lineno: int, text: str, source: str, deletable_indent: str | None
) -> list[Mutant]:
    """Whole-line mutants for ONE source line, in priority order (public: a later tool mutates
    covered lines the diff did not add).

    `text` is the line as `git diff` printed it (no newline), `source` the whole file (a mutant
    whose line occurs more than once is ambiguous and yields nothing), `deletable_indent` the
    line's leading whitespace when it is a complete single-line statement (`del` may replace it
    with `pass`), else None. The first operator that hits wins; `flag`/`del` are the last
    fallbacks, so a line with any operator gets exactly one mutant."""
    code = _code_only(text)
    if not code.strip() or _SKIP_LINE_RE.match(text):
        return []
    if source.count(text) != 1:
        return []
    for op, pattern in _OPERATORS:
        match = pattern.search(code)
        if match is not None:
            new = _replace(text, match.start(), match.end(), _mutated(op, match))
            return [Mutant(f"{path}:{lineno}:{op}", path, text, new)]
    for op, find in _EXTRA_OPERATORS:
        hit = find(code)
        if hit is not None:
            start, end, replacement = hit
            new = _replace(text, start, end, replacement)
            return [Mutant(f"{path}:{lineno}:{op}", path, text, new)]
    found: list[Mutant] = []
    flag = _FLAG_RE.search(code)
    if flag is not None:
        new = _replace(text, flag.start(), flag.end(), _mutated("flag", flag))
        found.append(Mutant(f"{path}:{lineno}:flag", path, text, new))
    if deletable_indent is not None:
        found.append(Mutant(f"{path}:{lineno}:del", path, text, deletable_indent + "pass"))
    return found


_BRANCHY_OPS = frozenset({"cmp", "bool", "not", "minmax", "method"})
_BRANCHY_TEXT_RE = re.compile(r"\b(?:if|elif|while)\s")


def _branchy(mutant: Mutant) -> bool:
    """True for a line that guards a branch or runs a comparison: the likeliest to be tested."""
    return mutant.name.rsplit(":", 1)[-1] in _BRANCHY_OPS or (
        _BRANCHY_TEXT_RE.search(_code_only(mutant.old)) is not None
    )


def _budget(by_file: Mapping[str, Sequence[Mutant]], limit: int) -> dict[str, int]:
    """How many of `limit` each file gets, round-robin one at a time: a file with fewer
    candidates than its turn does not waste the budget a busier file can use."""
    counts = dict.fromkeys(by_file, 0)
    remaining = limit
    while remaining > 0:
        live = [path for path, items in by_file.items() if counts[path] < len(items)]
        if not live:
            break
        for path in live:
            if remaining == 0:
                break
            counts[path] += 1
            remaining -= 1
    return counts


def _strided(items: Sequence[Mutant], count: int) -> list[Mutant]:
    """`count` of `items` at an even stride: the first, the last and evenly between, so a file
    with 40 candidates probes the whole file instead of its first 3 lines."""
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[0]]
    last = len(items) - 1
    return [items[round(i * last / (count - 1))] for i in range(count)]


def diff_mutants(repo: Path, base: str, glob: str, limit: int) -> list[Mutant]:
    """At most ``limit`` whole-line mutants on added code lines: each file's candidates are
    ordered branchy first, its share of the budget is sampled at an even stride, and the
    sampled mutants are reported round-robin over files."""
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
        indent: str | None = None
        if path.endswith(".py"):
            if path not in prose:
                prose[path] = string_lines(source)
                deletable[path] = deletable_lines(source)
            if lineno in prose[path]:
                continue
            indent = deletable[path].get(lineno)
        by_file.setdefault(path, []).extend(
            line_mutants(path, lineno, text, source, indent)
        )
    ordered = {
        path: [m for m in items if _branchy(m)] + [m for m in items if not _branchy(m)]
        for path, items in by_file.items()
    }
    budget = _budget(ordered, limit)
    queues = [_strided(items, budget[path]) for path, items in ordered.items()]
    picked: list[Mutant] = []
    while len(picked) < limit and any(queues):
        for queue in queues:
            if queue and len(picked) < limit:
                picked.append(queue.pop(0))
    return picked


_STATE_OF_KIND = {"KILLED": "KILLED", "TIMEOUT": "KILLED", "SURVIVED": "SURVIVED"}


def _mutant_state(line: str) -> tuple[str, str, bool]:
    """(KILLED|SURVIVED|<ERROR|INVALID>, the mutant's name, is_error) from a check line.

    A TIMEOUT counts as KILLED (the mutant never ran to completion, exactly as the single-copy
    report counts it); ERROR/INVALID are no verdict at all and make the run an error."""
    parts = line.split()
    kind = parts[0] if parts else "ERROR"
    name = parts[1].rstrip(":") if len(parts) > 1 else ""
    return _STATE_OF_KIND.get(kind, kind), name, kind not in _STATE_OF_KIND


def _compare_mutants(
    mutants: Sequence[Mutant],
    prepared: Sequence[tuple[Path, Sequence[str], Mapping[str, str], bool]],
    runner: RunFn,
    args: argparse.Namespace,
) -> int:
    """One line per mutant: `base=<state> head=<state> <name>`; exit 1 on a LOST kill.

    A lost kill is a mutant that the base tree's tests kill and the working tree's tests do
    not: the round's added test (or its fixture) stopped discriminating. This is maker step 8
    of a prune job - `mutate.py --compare-rev <base> -- ...` answers it in one command.
    """
    (base_dir, base_cmd, base_env, base_pytest), (head_dir, head_cmd, head_env, head_pytest) = prepared
    lost = errors = 0
    for mutant in mutants:
        states: dict[str, str] = {}
        details: list[str] = []
        name = mutant.name or mutant.file
        for label, dest, cmd, env, pytest_like in (
            ("base", base_dir, base_cmd, base_env, base_pytest),
            ("head", head_dir, head_cmd, head_env, head_pytest),
        ):
            _killed, is_error, line = check_mutant(
                dest, mutant, cmd, env, args.timeout, runner, pytest_like, False
            )
            state, found, state_error = _mutant_state(line)
            states[label] = state
            name = mutant.name or found or name  # the line keeps only a name's first word
            errors += bool(is_error or state_error)
            if state_error:
                details.append(f"  {label}: {line}")
        report = f"base={states['base']} head={states['head']} {name}"
        if states["base"] == "KILLED" and states["head"] == "SURVIVED":
            lost += 1
            report += "  LOST KILL"
        print(report)
        for detail in details:  # why a side has no verdict, right below its line
            print(detail)
    print(f"MUTANTS compared={len(mutants)} lost={lost} errors={errors}")
    if errors:
        return 2
    return 1 if lost else 0


def main(argv: Sequence[str] | None = None, run: RunFn | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mutate.py",
        description="One-command mutation check",
        epilog=(
            "After the unmutated baseline passes, a pytest run prints `BASELINE ran=N "
            "skipped=S deselected=D`\nfrom its summary and a WARN when tests were deselected "
            "(-k/-m) or skipped (a --rev copy\nhas no .git, so needs_git tests skip). Each "
            "mutant prints SURVIVED, KILLED (assert | crash\n<ExcType> | exit N), INVALID, "
            "ERROR or TIMEOUT. A mutant that broke the program TEXT (a\nSyntaxError, an "
            "ImportError, or a NameError for a name its own replacement invents) is\n"
            "INVALID and counts as an error, not a kill; a crash in production code is a "
            "kill.\n--auto-diff takes candidates from the added lines, branchy lines first, "
            "each file's share\nsampled at an even stride: comparison, boolean, `not`, int+1, "
            "min/max, method swaps,\nkeyword -> None, float x10, then the `flag` and `pass` "
            "(delete) fallbacks."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo", default=".", help="repo to copy (default: the current dir)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", help="repo-relative file to mutate")
    source.add_argument("--spec", help="JSON list of {name, file, old, new}")
    source.add_argument(
        "--auto-diff", metavar="BASE",
        help="generate mutants on the lines added since BASE (branchy lines first, stride-"
             "sampled per file; min/max, method swaps, keyword -> None, float x10)",
    )
    parser.add_argument(
        "--max", type=int, default=8,
        help="--auto-diff: mutant cap, round-robin over files and spread across each file "
             "(default 8)",
    )  # fmt: skip
    parser.add_argument("--glob", default="vibe_quant/*.py", help="--auto-diff: files to mutate")
    parser.add_argument("--old", help="text to replace; must occur exactly once")
    parser.add_argument("--new", help="replacement text")
    parser.add_argument("--name", default=None, help="mutant name (default <file>:<line>)")
    copy_source = parser.add_mutually_exclusive_group()
    copy_source.add_argument(
        "--rev", default=None, help="build the copy from this committed rev (git archive)"
    )
    copy_source.add_argument(
        "--compare-rev", default=None, metavar="REV",
        help="run every mutant TWICE - on a copy of REV and on the working tree - and print "
             "`base=<KILLED|SURVIVED> head=<...>`; exits 1 on a mutant KILLED on REV and "
             "SURVIVED on HEAD (a kill the round lost)",
    )  # fmt: skip
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
        if args.max < 1:
            parser.error("--max must be >= 1")
        try:
            mutants = diff_mutants(Path(args.repo).resolve(), args.auto_diff, args.glob, args.max)
        except subprocess.CalledProcessError as exc:
            print(f"ERROR auto-diff: {(exc.stderr or '').strip()}")
            return 2
        if not mutants:
            # A test-only or comment-only diff has no line under --glob: the check ran and
            # found nothing to mutate, so it is a clean exit, not a usage error (bead aq9zy).
            # The glob and base go to stderr so a typo'd glob is still visible in the run.
            print(
                f"auto-diff: no mutable added line in {args.glob} since {args.auto_diff}",
                file=sys.stderr,
            )
            print("MUTANTS no candidates")
            return 0
    else:
        try:
            mutants = load_spec(Path(args.spec))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"ERROR spec: {exc}")
            return 2

    repo = Path(args.repo).resolve()
    runner: RunFn = run or run_command
    cmd, extra_env = normalize_command(cmd)
    cmd, note = cap_workers(cmd, os.getloadavg()[0], os.cpu_count() or 1)
    if note:
        print(note, file=sys.stderr)
    root = Path(tempfile.mkdtemp(prefix="mutate-"))
    try:
        # A phase is (label, copy dir, the rev it is built from): one for a plain run (or a
        # pinned --rev), two for --compare-rev (REV, then the working tree).
        phases: list[tuple[str, Path, str | None]] = []
        if args.compare_rev is not None:
            phases = [("base", root / "base", args.compare_rev), ("head", root / "head", None)]
        else:
            phases = [("", root, args.rev)]
        prepared: list[tuple[Path, list[str], dict[str, str], bool]] = []
        for label, dest, rev in phases:
            dest.mkdir(parents=True, exist_ok=True)
            try:
                if rev:
                    extract_rev(repo, dest, rev)
                else:
                    build_copy(repo, dest)
            except (OSError, RuntimeError) as exc:
                print(f"ERROR repo: {exc}")
                return 2
            env = dict(os.environ)
            env["PYTHONPATH"] = str(dest)
            env["PYTHONDONTWRITEBYTECODE"] = "1"  # else a same-second, same-size mutation reuses a stale .pyc
            env.pop("PYTEST_ADDOPTS", None)  # the command is the spec; a gate's -q -rfE hides the summary
            env.update(extra_env)
            phase_cmd = resolve_executable(cmd, repo, dest)
            baseline = runner(phase_cmd, dest, env, args.timeout)
            if baseline.error or baseline.timed_out or baseline.exit_code != 0:
                print("ERROR baseline: tests fail without mutation")
                if baseline.timed_out:
                    print(f"baseline timed out after {args.timeout:g}s (raise --timeout)", file=sys.stderr)
                detail = (baseline.error or baseline.output).strip()
                if detail:
                    print(detail[-2000:], file=sys.stderr)
                return 2
            pytest_like = is_pytest_command(phase_cmd)
            if pytest_like:  # how many tests the baseline actually ran, before any mutant
                for report in baseline_report(baseline.output, f"{label} " if label else ""):
                    print(report)
            prepared.append((dest, phase_cmd, env, pytest_like))

        if args.compare_rev is not None:
            return _compare_mutants(mutants, prepared, runner, args)

        workdir, phase_cmd, env, pytest_like = prepared[0]
        killed = survived = errors = 0
        for mutant in mutants:
            was_killed, is_error, line = check_mutant(
                workdir, mutant, phase_cmd, env, args.timeout, runner, pytest_like,
                not args.quiet_kills,
            )  # fmt: skip
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
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
