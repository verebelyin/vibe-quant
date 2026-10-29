#!/usr/bin/env python3
"""Did a test-deleting diff LOST a mutant kill? The integration check `parity`.

    scripts/agents/parity_mutants.py --repo DIR --base REV [--head REV] [--max 8]
                                     [--timeout SEC]

`--base` is the revision the round started from; `--head` defaults to the working tree of
`--repo` (as mutate.py), and a `--head` rev builds a `git archive` copy so a past range can be
checked. In job 20261028-prune 13 of 16 reviews found a mutant that the deleted tests killed
on base and the remaining tests no longer kill — makers ran a few hand-picked mutants and
missed them. This finds such losses mechanically:

1. deleted tests: the changed test files are `git diff --name-only --no-renames BASE <head> --
   tests/` (python test files only); their node ids are collected with `pytest --collect-only
   -q` in the base copy and in the head copy, and `deleted = base ids - head ids` (a renamed
   or re-parametrized case counts as deleted: over-approximation is fine). No deleted id
   prints `PARITY no deleted tests` and exits 0 — nothing else runs.
2. candidate lines: the deleted ids run at base under `coverage run --source vibe_quant,scripts`,
   and the head test set (the changed test files that exist at head) at head under coverage.
   A line of a file changed between base and head is mutated only when its exact text occurs
   once in both (a changed file with no such line is named on stderr as skipped). A line that
   no deleted test BODY reached (coverage `test_function` context: imports and conftest
   fixtures only, which every test runs) is dropped unless
   a changed test file imports that module itself; no candidate left prints
   `PARITY compared=0 lost=0 errors=0 (no reachable candidate: skipped <files|none>)` and
   exits 0, so an empty proof reads as one. Priority 1 is a line the deleted tests executed and the head
   set does not (lost coverage), priority 2 the rest of the deleted tests' lines. Mutants come
   from mutate.py's `line_mutants`; the `--max` budget is filled from priority 1 first, each
   tier branchy first and strided across files and lines (never the first N).
3. screen: every mutant runs at base against the changed test files that exist at base and at
   head against the changed test files that exist at head, one baseline per side first (it
   must pass, as mutate.py). Each mutant prints one line:

    base=<KILLED|SURVIVED> head=<...> <name>              plus `  LOST KILL` (mutate.py's
                                                          --compare-rev format) when the
                                                          base side killed it and head did not

4. confirm: a `base=KILLED head=SURVIVED` mutant is re-run at head against the changed test
   files PLUS every test file under tests/ whose text names the mutated module (its dotted
   path, e.g. `vibe_quant.dsl.templates`, or `from vibe_quant.dsl import templates`, one line
   or a parenthesised several: the imports are parsed): a survivor selected by a test NAME is
   not proof of a lost kill (chief rule, job 20261028-prune). That enlarged set must PASS
   unmutated first (once per distinct set, cached): a set that already fails is no
   comparison and counts as an error. Then, still SURVIVED prints

    LOST KILL <name> (killed at base by <first failing node id>)

   and KILLED prints

    covered elsewhere <name> (<first failing node id>)

   as does a mutant that only breaks COLLECTION there (pytest exit 2-5, e.g. a deleted
   definition another test imports):

    covered elsewhere <name> (collection error: <first E line>)

The last line is always `PARITY compared=N lost=K errors=E`; exit 0 when nothing was lost,
1 on a lost kill, 2 on any error. Only the `LOST KILL <name> (` lines start at column 0, so
the gate can quote exactly the CONFIRMED losses. Diagnostics go to stderr: the
`candidate <name> (priority N)` line per picked mutant (what a hand run reports when it finds
nothing) and one `skip <file>: changed between base and head` line per unmutatable file.

An ERROR/INVALID on either side of the screen is NO verdict and makes the run an error, exactly
as in mutate.py: a `del` mutant that removes a module-level definition another test imports
breaks pytest COLLECTION (exit 2, no assertion ran), and a baseline that fails without a
mutant is never a comparison. A run that cannot answer exits 2 rather than reporting a clean 0.

Stdlib only (mutate.py, which is stdlib too, is imported by path); the copies need a python
with pytest and coverage, which is resolved as `<repo>/.venv/bin/python`, then the main
checkout's `.venv/bin/python` (a task worktree has no `.venv`), then this interpreter.
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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

_AGENTS_DIR = Path(__file__).resolve().parent
if str(_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENTS_DIR))

import mutate  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from mutate import Mutant, RunResult

TOOL_TIMEOUT = mutate.TOOL_TIMEOUT  # seconds per pytest/coverage run; --timeout overrides
MAX_MUTANTS = 8
COVERAGE_SOURCE = "vibe_quant,scripts"  # production code only: never tests/
PRODUCTION_PREFIXES = ("vibe_quant/", "scripts/")
# `-o addopts=` drops the repo ini's addopts (`-v --tb=short` turns `--collect-only -q` into a
# TREE, so no node id is ever parsed): the command is the spec, as in mutate.py.
PYTEST_ARGS = ("-o", "addopts=", "-p", "no:randomly", "-q", "-rfE")
FAILED_RE = re.compile(r"^(?:FAILED|ERROR) (\S+)", re.MULTILINE)


class ParityError(RuntimeError):
    """A run that cannot answer (bad rev, failing baseline, broken copy): exit 2."""


def is_test_file(path: str) -> bool:
    parts = path.split("/")
    return (
        len(parts) >= 3
        and parts[0] == "tests"
        and parts[-1].startswith("test_")
        and parts[-1].endswith(".py")
    )


def _tail(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def changed_test_files(repo: Path, base: str, head: str | None) -> list[str]:
    """The python test files the round changed (renames reported as delete + add)."""
    args = ["diff", "--name-only", "--no-renames", base]
    if head:
        args.append(head)
    args += ["--", "tests/"]
    proc = _git(repo, *args)
    if proc.returncode != 0:
        raise ParityError(f"git diff {base}..{head or 'working tree'} failed: {_tail(proc.stderr)}")
    return [line for line in proc.stdout.splitlines() if is_test_file(line)]


def test_python(repo: Path) -> list[str]:
    """The interpreter for the copies: theirs if any, else the main checkout's, else ours."""
    candidates = [repo / ".venv" / "bin" / "python"]
    main = mutate.main_checkout(repo)
    if main is not None:
        candidates.append(main / ".venv" / "bin" / "python")
    for candidate in candidates:
        if candidate.is_file():
            return [str(candidate)]
    return [sys.executable]


def _env(dest: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(dest)
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # a same-second, same-size mutant would reuse a .pyc
    env.pop("PYTEST_ADDOPTS", None)  # the command is the spec, not a gate's -q -rfE
    return env


def _pytest(python: Sequence[str]) -> list[str]:
    return [*python, "-m", "pytest", *PYTEST_ARGS]


def _collect(
    python: Sequence[str], dest: Path, env: Mapping[str, str], files: Sequence[str], timeout: float
) -> list[str]:
    """Every node id pytest collects from `files`, in collection order."""
    if not files:
        return []
    proc = subprocess.run(
        [
            *python,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-p",
            "no:randomly",
            "-q",
            "--collect-only",
            *files,
        ],
        cwd=dest,
        env=dict(env),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode not in (0, 5):  # 5 = no tests collected
        raise ParityError(
            f"collect in {dest.name} failed (exit {proc.returncode}): "
            f"{_tail(proc.stdout + proc.stderr)}"
        )
    return [
        line.strip()
        for line in proc.stdout.splitlines()
        if line.strip().startswith("tests/") and "::" in line
    ]


def coverage_lines(
    python: Sequence[str],
    dest: Path,
    env: Mapping[str, str],
    targets: Sequence[str],
    label: str,
    timeout: float,
    body: dict[str, set[int]] | None = None,
) -> dict[str, set[int]]:
    """{repo-relative production file: executed line numbers} for one test set.

    `targets` are node ids or file paths; they go through `@argsfile` (pytest reads one
    argument per line), so a long deleted-id list never hits the argv limit. A `body` dict is
    filled with the lines a TEST FUNCTION reached (coverage's `test_function` dynamic context),
    as opposed to imports and fixtures, which every test of the directory runs too.
    """
    if not targets:
        return {}
    argsfile = dest / f".parity-{label}-args.txt"
    argsfile.write_text("".join(f"{target}\n" for target in targets), encoding="utf-8")
    data = dest / f".coverage-parity-{label}"
    report = dest / f".parity-{label}-coverage.json"
    rcfile = dest / f".parity-{label}.coveragerc"
    rcfile.write_text(
        "[run]\n" + ("dynamic_context = test_function\n" if body is not None else ""),
        encoding="utf-8",
    )
    run = subprocess.run(
        [
            *python,
            "-m",
            "coverage",
            "run",
            f"--data-file={data}",
            f"--rcfile={rcfile}",
            f"--source={COVERAGE_SOURCE}",
            "-m",
            "pytest",
            *PYTEST_ARGS,
            "@" + str(argsfile),
        ],
        cwd=dest,
        env=dict(env),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if run.returncode != 0:
        raise ParityError(
            f"coverage {label}: pytest exit {run.returncode}: "
            f"{_tail(run.stdout + run.stderr)}"
        )
    dump = subprocess.run(
        [
            *python,
            "-m",
            "coverage",
            "json",
            f"--data-file={data}",
            f"--rcfile={rcfile}",
            *(["--show-contexts"] if body is not None else []),
            "-o",
            str(report),
        ],
        cwd=dest,
        env=dict(env),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if dump.returncode != 0:
        raise ParityError(f"coverage {label}: json exit {dump.returncode}: {_tail(dump.stderr)}")
    payload = json.loads(report.read_text(encoding="utf-8"))
    files = payload.get("files", {})
    if body is not None:
        for name, entry in files.items():
            contexts = entry.get("contexts", {}) if isinstance(entry, dict) else {}
            reached = {int(line) for line, labels in contexts.items() if any(labels)}
            if reached:
                body[Path(name).as_posix()] = reached
    return {
        Path(name).as_posix(): set(entry.get("executed_lines", []))
        for name, entry in files.items()
        if isinstance(entry, dict)
    }


def is_production(path: str) -> bool:
    return path.startswith(PRODUCTION_PREFIXES) and not path.startswith("tests/")


def _identical(base_dir: Path, head_dir: Path, path: str) -> bool:
    base = base_dir / path
    head = head_dir / path
    return base.is_file() and head.is_file() and base.read_bytes() == head.read_bytes()


def candidate_mutants(
    base_dir: Path,
    head_dir: Path,
    base_exec: Mapping[str, set[int]],
    head_exec: Mapping[str, set[int]],
    limit: int,
    body_exec: Mapping[str, set[int]] | None = None,
    named: Callable[[str], bool] = lambda _path: True,
) -> tuple[list[tuple[Mutant, int]], list[str]]:
    """((mutant, priority) for at most `limit` covered lines, priority 1 filled first;
    the changed files that gave no candidate).

    A line of a file changed between base and head is a candidate only when its exact text
    occurs once in both (the mutant is applied by text on each side). A line in
    not in `body_exec` (reached only by imports or fixtures, never by a deleted test's body) is
    dropped unless `named(path)`
    (a changed test file imports that module itself, so its constants are what it tests)."""
    tiers: tuple[dict[str, list[Mutant]], dict[str, list[Mutant]]] = ({}, {})
    skipped: list[str] = []
    for path, lines in sorted(base_exec.items()):
        if not is_production(path) or not lines:
            continue
        head_source: str | None = None
        try:
            source = (base_dir / path).read_text(encoding="utf-8")
            if not _identical(base_dir, head_dir, path):
                if not (head_dir / path).is_file():
                    print(f"skip {path}: deleted at head", file=sys.stderr)
                    continue
                head_source = (head_dir / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            print(f"skip {path}: {exc}", file=sys.stderr)
            continue
        if body_exec is not None and not named(path):
            lines = lines & body_exec.get(path, set())
        prose = mutate.string_lines(source)
        deletable = mutate.deletable_lines(source)
        text_lines = source.splitlines()
        head_lines = head_exec.get(path, set())
        found = False
        for lineno in sorted(lines):
            if lineno in prose or not 1 <= lineno <= len(text_lines):
                continue
            text = text_lines[lineno - 1]
            head_lineno = lineno
            if head_source is not None:
                if head_source.count(text) != 1:
                    continue
                head_lineno = head_source.count("\n", 0, head_source.index(text)) + 1
            tier = tiers[0] if head_lineno not in head_lines else tiers[1]
            for mutant in mutate.line_mutants(
                path, lineno, text, source, deletable.get(lineno)
            ):
                tier.setdefault(path, []).append(mutant)
                found = True
        if head_source is not None and not found:
            print(f"skip {path}: changed between base and head", file=sys.stderr)
            skipped.append(path)
    picked = [(m, 1) for m in _pick(tiers[0], limit)]
    picked += [(m, 2) for m in _pick(tiers[1], limit - len(picked))]
    return picked, skipped


def _pick(groups: Mapping[str, Sequence[Mutant]], limit: int) -> list[Mutant]:
    """`limit` mutants branchy-first, each file's share strided over its lines, round-robin."""
    if limit <= 0 or not groups:
        return []
    ordered = {
        path: [m for m in items if mutate._branchy(m)]
        + [m for m in items if not mutate._branchy(m)]
        for path, items in groups.items()
    }
    budget = mutate._budget(ordered, limit)
    queues = [mutate._strided(ordered[path], budget[path]) for path in ordered]
    picked: list[Mutant] = []
    while len(picked) < limit and any(queues):
        for queue in queues:
            if queue and len(picked) < limit:
                picked.append(queue.pop(0))
    return picked


@dataclass(frozen=True)
class Side:
    """One copy and the test files it can run: base or head."""

    label: str
    dest: Path
    cmd: list[str]  # pytest argv prefix, no test paths
    env: dict[str, str]
    files: list[str]

    def command(self, files: Sequence[str] | None = None) -> list[str]:
        return [*self.cmd, *(self.files if files is None else files)]


class _Recorder:
    """A RunFn that also keeps the last RunResult (the failing node id lives in its output)."""

    def __init__(self, run: Callable[..., RunResult]) -> None:
        self.run = run
        self.last: RunResult | None = None

    def __call__(
        self,
        cmd: Sequence[str],
        cwd: Path,
        env: Mapping[str, str],
        timeout: float,
    ) -> RunResult:
        self.last = self.run(cmd, cwd, env, timeout)
        return self.last


def _verdict(line: str) -> tuple[str, bool]:
    """(KILLED|SURVIVED|ERROR|INVALID|TIMEOUT, is_error) from a check_mutant line."""
    kind = line.split()[0] if line.split() else "ERROR"
    if kind in ("KILLED", "TIMEOUT"):
        return "KILLED", False  # a timeout never completed: exactly how the report counts it
    if kind == "SURVIVED":
        return "SURVIVED", False
    return kind, True


def _first_failed(output: str) -> str:
    found = FAILED_RE.search(output)
    return found.group(1) if found is not None else "unknown"


@dataclass
class Screened:
    mutant: Mutant
    base: str
    head: str
    node: str  # the base side's first failing node id ("" unless base KILLED)
    errors: int


def screen(
    mutants: Sequence[Mutant],
    base: Side,
    head: Side,
    timeout: float,
    run: Callable[..., RunResult],
) -> list[Screened]:
    """Each mutant at base and at head; one baseline per side must pass first."""
    for side in (base, head):
        if not side.files:
            print(
                f"{side.label}: no changed test file left (every mutant survives there)",
                file=sys.stderr,
            )
            continue
        recorder = _Recorder(run)
        result = recorder(side.command(), side.dest, side.env, timeout)
        if result.error or result.timed_out or result.exit_code != 0:
            detail = result.error or _tail(result.output) or f"exit {result.exit_code}"
            raise ParityError(f"{side.label} baseline: tests fail without mutation: {detail}")
        for report in mutate.baseline_report(result.output, f"{side.label} "):
            print(report)
    screened: list[Screened] = []
    for mutant in mutants:
        states: dict[str, str] = {}
        details: list[str] = []
        errors = 0
        node = ""
        for side in (base, head):
            if not side.files:
                states[side.label] = "SURVIVED"
                continue
            recorder = _Recorder(run)
            _killed, is_error, line = mutate.check_mutant(
                side.dest, mutant, side.command(), side.env, timeout, recorder, True, False
            )
            state, state_error = _verdict(line)
            states[side.label] = state
            errors += int(is_error or state_error)
            if state_error:
                details.append(f"  {side.label}: {line}")
            elif side.label == "base" and state == "KILLED" and recorder.last is not None:
                node = _first_failed(recorder.last.output)
        report = f"base={states['base']} head={states['head']} {mutant.name or mutant.file}"
        if states["base"] == "KILLED" and states["head"] == "SURVIVED":
            report += "  LOST KILL"  # mutate.py --compare-rev's format; confirmed below
        print(report)
        for detail in details:
            print(detail)
        screened.append(Screened(mutant, states["base"], states["head"], node, errors))
    return screened


def module_needles(path: str) -> tuple[str, str | None]:
    """(dotted module path, "package|name" for a `from package import name`) of a file."""
    stem = path[: -len(".py")] if path.endswith(".py") else path
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    dotted = stem.replace("/", ".")
    package, _, name = dotted.rpartition(".")
    return dotted, (f"{package}|{name}" if package else None)


def _imports_from(text: str, package: str, name: str, fallback: re.Pattern[str]) -> bool:
    """Does `text` hold `from <package> import ... <name> ...`, on one line or several?

    The import statements are parsed, so a parenthesised multi-line form counts and a comment
    does not; a file that does not parse is read by the single-line `fallback` regex."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return fallback.search(text) is not None
    return any(
        isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module == package
        and any(alias.name == name for alias in node.names)
        for node in ast.walk(tree)
    )


def _namer(path: str) -> Callable[[str], bool]:
    """A test-file text -> "does it name the module at `path`" predicate."""
    dotted, import_form = module_needles(path)
    package, name = import_form.split("|") if import_form else ("", "")
    fallback = re.compile(
        rf"\bfrom\s+{re.escape(package)}\s+import\s+[^\n]*\b{re.escape(name)}\b"
    )
    return lambda text: dotted in text or bool(
        package and _imports_from(text, package, name, fallback)
    )


def test_files_naming(head_dir: Path, path: str) -> list[str]:
    """Every tests/ file whose text names the mutated module."""
    names = _namer(path)
    return [
        test_file.relative_to(head_dir).as_posix()
        for test_file in sorted((head_dir / "tests").rglob("test_*.py"))
        if names(test_file.read_text(encoding="utf-8", errors="replace"))
    ]


def _importer_baseline(
    head: Side, files: Sequence[str], timeout: float, run: Callable[..., RunResult]
) -> str:
    """"" when `files` pass at head with no mutant applied, else why they do not."""
    result = run(head.command(files), head.dest, head.env, timeout)
    if result.error:
        return result.error
    if result.timed_out:
        return "timed out"
    if result.exit_code == 0:
        return ""
    failed = _first_failed(result.output)
    return failed if failed != "unknown" else _tail(result.output) or f"exit {result.exit_code}"


def confirm(
    screened: Sequence[Screened],
    head: Side,
    timeout: float,
    run: Callable[..., RunResult],
) -> tuple[int, int]:
    """(lost, errors) after re-running every candidate against every test naming its module.

    The enlarged test set must pass UNMUTATED first (once per distinct set): a test that
    already fails says nothing about the mutant, so a failing set is an error, never "covered
    elsewhere". After a passing baseline, a mutant that only breaks COLLECTION (pytest exit 2-5,
    e.g. a deleted definition another test imports) is "covered elsewhere" with that error."""
    lost = errors = 0
    baselines: dict[tuple[str, ...], str] = {}
    for item in screened:
        if item.base != "KILLED" or item.head != "SURVIVED":
            continue
        files = tuple(sorted(set(head.files) | set(test_files_naming(head.dest, item.mutant.file))))
        name = item.mutant.name or item.mutant.file
        if not files:
            lost += 1
            print(f"LOST KILL {name} (killed at base by {item.node})")
            continue
        if files not in baselines:
            baselines[files] = _importer_baseline(head, files, timeout, run)
        if baselines[files]:
            errors += 1
            print(f"  confirm: {name}: tests fail without the mutant: {baselines[files]}")
            continue
        recorder = _Recorder(run)
        _killed, is_error, line = mutate.check_mutant(
            head.dest,
            item.mutant,
            head.command(files),
            head.env,
            timeout,
            recorder,
            True,
            False,
        )
        state, state_error = _verdict(line)
        last = recorder.last
        if (
            state == "INVALID"
            and last is not None
            and last.exit_code in mutate.PYTEST_NO_ASSERTION
            and "collection/usage error" in line
        ):
            reason = mutate.first_error_line(last.output) or f"exit {last.exit_code}"
            print(f"covered elsewhere {name} (collection error: {reason})")
            continue
        errors += int(is_error or state_error)
        node = _first_failed(last.output) if last is not None else "unknown"
        if state == "SURVIVED":
            lost += 1
            print(f"LOST KILL {name} (killed at base by {item.node})")
        elif state == "KILLED":
            print(f"covered elsewhere {name} ({node})")
        else:
            print(f"  confirm: {line}")
    return lost, errors


def _require(python: Sequence[str]) -> None:
    probe = subprocess.run(
        [*python, "-c", "import coverage, pytest"], capture_output=True, text=True
    )
    if probe.returncode != 0:
        raise ParityError(
            f"{python[0]}: needs coverage and pytest ({_tail(probe.stderr + probe.stdout)})"
        )


def check_base(repo: Path, base: str) -> None:
    proc = _git(repo, "rev-parse", "--verify", "--quiet", f"{base}^{{commit}}")
    if proc.returncode != 0:
        raise ParityError(f"unknown base rev: {base}")


def run(args: argparse.Namespace, runner: Callable[..., RunResult]) -> int:
    repo = Path(args.repo).resolve()
    check_base(repo, args.base)
    python = test_python(repo)
    _require(python)
    root = Path(tempfile.mkdtemp(prefix="parity-"))
    try:
        base_dir = root / "base"
        base_dir.mkdir()
        mutate.extract_rev(repo, base_dir, args.base)
        head_dir = root / "head"
        head_dir.mkdir()
        if args.head:
            mutate.extract_rev(repo, head_dir, args.head)
        else:
            mutate.build_copy(repo, head_dir)

        changed = changed_test_files(repo, args.base, args.head)
        base_files = [path for path in changed if (base_dir / path).is_file()]
        head_files = [path for path in changed if (head_dir / path).is_file()]
        base_env, head_env = _env(base_dir), _env(head_dir)
        base_ids = _collect(python, base_dir, base_env, base_files, args.timeout)
        head_ids = _collect(python, head_dir, head_env, head_files, args.timeout)
        deleted = [node for node in base_ids if node not in set(head_ids)]
        if not deleted:
            print("PARITY no deleted tests")
            return 0

        body_exec: dict[str, set[int]] = {}
        base_exec = coverage_lines(
            python, base_dir, base_env, deleted, "base", args.timeout, body_exec
        )
        test_texts = [
            (base_dir / path).read_text(encoding="utf-8", errors="replace")
            for path in base_files
        ]
        head_exec = (
            coverage_lines(python, head_dir, head_env, head_files, "head", args.timeout)
            if head_files
            else {}
        )
        picks, skipped = candidate_mutants(
            base_dir,
            head_dir,
            base_exec,
            head_exec,
            args.max,
            body_exec,
            lambda path: any(map(_namer(path), test_texts)),
        )
        for mutant, tier in picks:
            print(f"candidate {mutant.name} (priority {tier})", file=sys.stderr)
        if not picks:
            print(
                "PARITY compared=0 lost=0 errors=0 (no reachable candidate: skipped "
                f"{', '.join(skipped) or 'none'})"
            )
            return 0

        base = Side("base", base_dir, _pytest(python), base_env, base_files)
        head = Side("head", head_dir, _pytest(python), head_env, head_files)
        screened = screen([mutant for mutant, _tier in picks], base, head, args.timeout, runner)
        lost, errors = confirm(screened, head, args.timeout, runner)
        errors += sum(item.errors for item in screened)
        print(f"PARITY compared={len(screened)} lost={lost} errors={errors}")
        if errors:
            return 2
        return 1 if lost else 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main(argv: Sequence[str] | None = None, run_fn: Callable[..., RunResult] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="parity_mutants.py",
        description="Did a test-deleting diff lose a mutant kill?",
        epilog=(
            "Prints `base=<KILLED|SURVIVED> head=<...> <name>` per mutant, then `LOST KILL "
            "<name>\n(killed at base by <node id>)` for a loss confirmed against every test "
            "file that names\nthe module, or `covered elsewhere <name> (<node id>)`. The last "
            "line is always\n`PARITY compared=N lost=K errors=E`; exit 0 nothing lost, 1 lost "
            "kill, 2 error."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo", default=".", help="repo to copy (default: the current dir)")
    parser.add_argument("--base", required=True, help="the revision the round started from")
    parser.add_argument(
        "--head", default=None, help="the revision to compare with (default: the working tree)"
    )
    parser.add_argument(
        "--max",
        type=int,
        default=MAX_MUTANTS,
        help=f"mutant cap, priority-1 lines first (default {MAX_MUTANTS})",
    )
    parser.add_argument(
        "--timeout", type=float, default=TOOL_TIMEOUT, help="seconds per pytest/coverage run"
    )
    args = parser.parse_args(argv)
    if args.max < 1:
        parser.error("--max must be >= 1")
    try:
        return run(args, run_fn or mutate.run_command)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(f"PARITY error: {' '.join(str(exc).split())[:200]}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
