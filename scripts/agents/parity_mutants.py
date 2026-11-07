#!/usr/bin/env python3
"""Did a test-deleting diff LOST a mutant kill? The integration check `parity`.

    scripts/agents/parity_mutants.py --repo DIR --base REV [--head REV] [--max 8]
                                     [--timeout SEC] [--workers N] [--budget SEC]

`--base` is the revision the round started from; `--head` defaults to the working tree of
`--repo` (as mutate.py), and a `--head` rev builds a `git archive` copy so a past range can be
checked. In job 20261028-prune 13 of 16 reviews found a mutant that the deleted tests killed
on base and the remaining tests no longer kill — makers ran a few hand-picked mutants and
missed them. This finds such losses mechanically:

1. deleted tests: the changed test files are `git diff --name-only --no-renames BASE <head> --
   tests/` (python test files only); their node ids are collected with `pytest --collect-only
   -q` in the base copy and in the head copy (both at once, each in slices), and `deleted = base
   ids - head ids`, minus the RENAMED ones. A missing base id counts as renamed only while (a)
   the diff changes no file a test can import or load (any .py that is not a test file, anything
   under vibe_quant/ or scripts/, anything under tests/ but a test file: a kept test's text proves
   nothing once its production code or helpers moved) and (b) its test FILE as a whole is
   ast_same-identical (module-level code included; names, docstrings and comments ignored) and
   names no other changed test file; then a new id of the same file holding an AST-identical test
   with the same `[param]` answers it (each new head id answers one missing id, so a deleted
   duplicate stays deleted). In every other case, and for a test moved to another file or a
   re-parametrized case, the id counts as deleted (over-approximation is fine). No deleted id
   prints `PARITY no deleted tests` and exits 0 - nothing else runs.
2. candidate lines: the deleted ids run at base
   under `coverage run --source vibe_quant,scripts`, and the head test files at head under
   coverage, split in slices that run in separate coverage processes at once (`--include` only the files the
   deleted tests executed; `dynamic_context = test_function`). A line of a file changed between
   base and head is mutated only when its exact text occurs once in both (a changed file with no
   such line is named on stderr as skipped). A line that no deleted test BODY reached (coverage
   `test_function` context: imports and conftest fixtures only, which every test runs) is
   dropped unless a changed test file imports that module itself; no candidate left prints
   `PARITY compared=0 lost=0 errors=0 (no reachable candidate: skipped <files|none>)` and
   exits 0, so an empty proof reads as one. Priority 1 is a line the deleted tests executed and the head
   set does not (lost coverage), priority 2 the rest of the deleted tests' lines, then the code of strategy template strings
   (templates.py `_lines(`: no body-reach proof, spare budget only). Mutants come
   from mutate.py's `line_mutants`; the `--max` budget is filled from priority 1 first, each
   tier branchy first and strided across files and lines (never the first N).
3. screen: each mutant runs at base against every changed test file except those provably
   unchanged (rules (a) and (b) above, and no id of the file deleted). The base side is never
   narrowed by coverage: a context cannot prove which base test kills a mutant (a fixture, a
   subprocess or changed production code hides one). When base killed it, it runs at head against
   the head tests whose coverage context executed the line (no context reached it, e.g. an
   import-time constant: the changed head test files naming the module, else all of them). A
   mutant no base test kills cannot lose a kill, so the head side is skipped there
   (`head=SKIPPED`). A KILLED verdict
   stands only if the tests that failed under the mutant pass unmutated (they re-run on the
   restored copy: a base or head that is red without any mutant is an error, as mutate.py's
   baseline; this replaces the old whole-suite baseline run). Each mutant prints one line:

    base=<KILLED|SURVIVED|...> head=<...|SKIPPED> <name>  plus `  LOST KILL` (mutate.py's
                                                          --compare-rev format) when the
                                                          base side killed it and head did not

   and a `parity: select <name> head=<n|all>` line on stderr (how many head test targets).

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

Parallel (`--workers`, default and cap `os.cpu_count()`; 0 or 1 = serial): the collection runs, the base
coverage runs, the head coverage slices and the context queries run at the same time. The screen
runs one mutant per SLOT (min(workers, mutants) slots, each its own clone of the base and head
copy, workers // slots pytest workers per run); lines print in candidate order. The
confirm baseline and the confirm runs carry `-n <workers> --dist worksteal`. Progress goes to
stderr, one line at each phase start (`parity: collect|coverage|screen|confirm ...`), one per
shard (`parity: shard head3 ...s`) and one per mutant (`parity: screen i/N <name> <elapsed>s`),
so a long run is never silent.

`--budget SEC` (default 0 = no cap) is a deadline for the whole run: before each pytest or
coverage run the script stops once it has passed, else that run's timeout is
min(`--timeout`, remaining). A run cut short by that clamp is NOT a verdict — a TIMEOUT counts
KILLED otherwise, but a budget-clamped one proves nothing — so it stops the run. The last line
is then `PARITY INCOMPLETE compared=i of N lost=K errors=E (budget Bs hit in <phase>)` after
any `LOST KILL` lines already confirmed; exit 2 when an error was counted (fail closed: an
error seen on either side of a mutant is kept even when the cap then interrupts the other side),
else 1 when lost > 0, else 3. The gate shows exit 3 as a WARN (checked i of N) and the landing
proceeds; an error or a confirmed lost kill before the cap is still a FAIL.

An ERROR/INVALID on either side of the screen is NO verdict and makes the run an error, exactly
as in mutate.py, with one exception: a mutant INVALID at BOTH base and head can neither keep nor
lose a kill, so it is dropped (`dropped <name>: INVALID at base and head`) and the next candidate
from a wider pick takes its slot; a run left with no valid comparison is an error (a job gate
once counted 3 such mutants as 6 errors with lost=0). a `del` mutant that removes a module-level definition another test imports
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
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING

_AGENTS_DIR = Path(__file__).resolve().parent
if str(_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENTS_DIR))

import ast_same  # noqa: E402
import mutate  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

    from mutate import Mutant, RunResult

TOOL_TIMEOUT = mutate.TOOL_TIMEOUT  # seconds per pytest/coverage run; --timeout overrides
MAX_MUTANTS = 8
SHARDS_PER_WORKER = 2  # more bins than coverage processes at once: a free process takes the next
COVERAGE_SOURCE = "vibe_quant,scripts"  # production code only: never tests/
PRODUCTION_PREFIXES = ("vibe_quant/", "scripts/")
# `-o addopts=` drops the repo ini's addopts (`-v --tb=short` turns `--collect-only -q` into a
# TREE, so no node id is ever parsed): the command is the spec, as in mutate.py.
PYTEST_ARGS = ("-o", "addopts=", "-p", "no:randomly", "-q", "-rfE", "-m", "not timing")  # timing tests lie under coverage
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


def _importable(path: str) -> bool:
    """Can a test import or load this path? Assume so for every file but a test file: tests read
    config (pyproject.toml), docs and data too, and a wrong "no" would fail open."""
    return not is_test_file(path)


def changed_non_test_paths(repo: Path, base: str, head: str | None) -> list[str]:
    """The files changed between `base` and `head` that tests can import or load (sorted).

    A working-tree head also counts untracked files. While this list is not empty a test that is
    textually unchanged may behave differently, so nothing about the tests may be taken as
    unchanged from their text."""
    args = ["diff", "--name-only", "--no-renames", base]
    if head:
        args.append(head)
    proc = _git(repo, *args)
    if proc.returncode != 0:
        raise ParityError(f"git diff {base}..{head or 'working tree'} failed: {_tail(proc.stderr)}")
    paths = proc.stdout.splitlines()
    if not head:
        untracked = _git(repo, "ls-files", "--others", "--exclude-standard")
        if untracked.returncode != 0:
            raise ParityError(f"git ls-files failed: {_tail(untracked.stderr)}")
        paths += untracked.stdout.splitlines()
    return sorted({path for path in paths if _importable(path)})


def unchanged_test_files(base_dir: Path, head_dir: Path, files: Sequence[str]) -> set[str]:
    """The test files of `files` that behave the same at base and head as far as their text goes.

    A file counts when it exists on both sides and its WHOLE module is ast_same-identical
    (module-level code included, not just the test bodies; test names, docstrings and comments
    ignored), and its text names no other changed test file (it could import that module's
    helpers). A file that cannot be read or parsed does not count (fail closed). This is the
    precondition for treating a renamed test as the same test and for leaving the file out of
    the base screen; it also needs `changed_non_test_paths` to be empty, which the caller checks."""
    same: set[str] = set()
    for path in files:
        try:
            if ast_same.compare_source(
                (base_dir / path).read_text(encoding="utf-8"),
                (head_dir / path).read_text(encoding="utf-8"),
            )[0]:
                same.add(path)
        except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
            continue
    others = [Path(path).stem for path in files if path not in same]
    return {
        path
        for path in same
        if not any(stem in (head_dir / path).read_text(encoding="utf-8") for stem in others)
    }


def cpu_budget() -> int:
    """The host's cores: the most pytest/coverage processes the run may keep busy at once."""
    return max(os.cpu_count() or 1, 1)


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


def _pytest(python: Sequence[str], workers: int = 1) -> list[str]:
    """The pytest argv prefix for the copies; `-n W --dist worksteal` when `workers` >= 2.

    Only the runs that need no coverage contexts use it (the screen baselines, the mutant runs
    of screen()/confirm() and the confirm baseline). Collection stays serial, and the coverage
    runs build their own argv: `dynamic_context=test_function` needs pytest-cov under xdist."""
    parallel = ["-n", str(workers), "--dist", "worksteal"] if workers >= 2 else []
    return [*python, "-m", "pytest", *PYTEST_ARGS, *parallel]


def _collect(
    python: Sequence[str],
    dest: Path,
    env: Mapping[str, str],
    files: Sequence[str],
    timeout: float,
    run: Callable[..., RunResult],
) -> list[str]:
    """Every node id pytest collects from `files`, in collection order."""
    if not files:
        return []
    result = run(
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
        dest,
        dict(env),
        timeout,
    )
    if result.error:
        raise ParityError(f"collect in {dest.name} failed: {result.error}")
    if result.timed_out or result.exit_code not in (0, 5):  # 5 = no tests collected
        detail = "timed out" if result.timed_out else f"exit {result.exit_code}"
        raise ParityError(f"collect in {dest.name} failed ({detail}): {_tail(result.output)}")
    return [
        line.strip()
        for line in result.output.splitlines()
        if line.strip().startswith("tests/") and "::" in line
    ]


def coverage_data(dest: Path, label: str) -> Path:
    """The coverage data file of one run: beside the copy, so cloning the copy never carries it."""
    return dest.parent / f".coverage-parity-{dest.name}-{label}"


def split_node_id(node_id: str) -> tuple[str, str, str]:
    """(file, ``Class::test_x``, ``[param]`` or "") of a pytest node id."""
    file, _, rest = node_id.partition("::")
    name, bracket, param = rest.partition("[")
    return file, name, bracket + param


class _FileFingerprints:
    """Per-directory cache of a test file's ast_same fingerprints (None: unreadable/unparsable)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache: dict[
            str, tuple[dict[str, ast_same.TestFingerprint], dict[str, str]] | None
        ] = {}

    def get(
        self, path: str
    ) -> tuple[dict[str, ast_same.TestFingerprint], dict[str, str]] | None:
        if path not in self._cache:
            try:
                source = (self.root / path).read_text(encoding="utf-8")
                self._cache[path] = (
                    ast_same.fingerprint_tests(source),
                    ast_same.env_fingerprints(source),
                )
            except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
                self._cache[path] = None
        return self._cache[path]


def renamed_ids(
    base_dir: Path,
    head_dir: Path,
    deleted: Sequence[str],
    base_ids: Sequence[str],
    head_ids: Sequence[str],
    unchanged: Collection[str],
) -> set[str]:
    """The `deleted` ids that were only RENAMED: in a file of `unchanged` (see
    `unchanged_test_files`; any other file's ids stay deleted), a head id of the same file that is
    new (not a base id) holds an AST-identical test (decorators, arguments, body; name, docstring
    and comments ignored, as ast_same.py) with the same `[param]`, and every module-level name the
    test refers to is unchanged there. Each new head id answers ONE deleted id, so a deleted
    duplicate of a test that stays is still deleted. Anything that cannot be parsed stays deleted
    (fail closed). A test moved to another file is a deletion."""
    base_set = set(base_ids)
    base_fp, head_fp = _FileFingerprints(base_dir), _FileFingerprints(head_dir)
    pool: dict[tuple[str, str, str], list[tuple[str, dict[str, str]]]] = {}
    for node_id in head_ids:
        if node_id in base_set:
            continue
        file, qualname, param = split_node_id(node_id)
        parsed = head_fp.get(file)
        found = parsed[0].get(qualname) if parsed is not None else None
        if parsed is not None and found is not None:
            pool.setdefault((found.body, found.header, param), []).append((file, parsed[1]))
    renamed: set[str] = set()
    for node_id in deleted:
        file, qualname, param = split_node_id(node_id)
        if file not in unchanged:
            continue
        parsed = base_fp.get(file)
        found = parsed[0].get(qualname) if parsed is not None else None
        if parsed is None or found is None:
            continue
        candidates = pool.get((found.body, found.header, param), [])
        for index, (head_file, head_env) in enumerate(candidates):
            if head_file != file:
                continue
            if all(parsed[1].get(name) == head_env.get(name) for name in found.names):
                del candidates[index]
                renamed.add(node_id)
                break
    return renamed


def collect_parallel(
    python: Sequence[str],
    dest: Path,
    env: Mapping[str, str],
    files: Sequence[str],
    timeout: float,
    run: Callable[..., RunResult],
    chunks: int,
) -> list[str]:
    """`_collect` over `chunks` contiguous slices of `files` at the same time: the same ids in
    the same order as one serial collection."""
    size = max(-(-len(files) // max(chunks, 1)), 1)
    parts = [files[i : i + size] for i in range(0, len(files), size)]
    if len(parts) <= 1:
        return _collect(python, dest, env, files, timeout, run)
    with ThreadPoolExecutor(max_workers=len(parts)) as executor:
        futures = [executor.submit(_collect, python, dest, env, part, timeout, run) for part in parts]
        return [node_id for future in futures for node_id in future.result()]


def _shards(weights: Mapping[str, int], files: Sequence[str], count: int) -> list[list[str]]:
    """`files` in `count` bins of about equal weight (largest first into the lightest bin)."""
    bins: list[list[str]] = [[] for _ in range(max(min(count, len(files)), 1))]
    load = [0] * len(bins)
    for path in sorted(files, key=lambda f: (-weights.get(f, 0), f)):
        lightest = load.index(min(load))
        bins[lightest].append(path)
        load[lightest] += weights.get(path, 0) + 1
    order = sorted(range(len(bins)), key=lambda i: -load[i])
    return [bins[i] for i in order if bins[i]]


def coverage_sharded(
    python: Sequence[str],
    dest: Path,
    env: Mapping[str, str],
    files: Sequence[str],
    label: str,
    timeout: float,
    run: Callable[..., RunResult],
    shards: int,
    include: Sequence[str],
    weights: Mapping[str, int] | None = None,
) -> tuple[dict[str, set[int]], list[Path]]:
    """Executed lines (union) and the dynamic-context data files of `files` run in `shards`
    independent coverage processes (xdist cannot carry contexts, separate processes can).
    `weights` (tests per file) balance the bins; each shard prints its wall time on stderr."""
    bins = _shards(weights or {}, files, shards * SHARDS_PER_WORKER if shards > 1 else 1)
    labels = [label] if len(bins) == 1 else [f"{label}{k}" for k in range(len(bins))]

    def one(part: list[str], name: str) -> dict[str, set[int]]:
        started = time.monotonic()
        lines = coverage_lines(
            python, dest, env, part, name, timeout, run, None, include, True
        )
        print(
            f"parity: shard {name} {len(part)} file(s) {time.monotonic() - started:.0f}s",
            file=sys.stderr,
            flush=True,
        )
        return lines

    with ThreadPoolExecutor(max_workers=max(min(shards, len(bins)), 1)) as executor:
        futures = [
            executor.submit(one, part, name) for part, name in zip(bins, labels, strict=True)
        ]
        merged: dict[str, set[int]] = {}
        for future in futures:
            for path, lines in future.result().items():
                merged.setdefault(path, set()).update(lines)
    return merged, [coverage_data(dest, name) for name in labels]


def coverage_lines(
    python: Sequence[str],
    dest: Path,
    env: Mapping[str, str],
    targets: Sequence[str],
    label: str,
    timeout: float,
    run: Callable[..., RunResult],
    body: dict[str, set[int]] | None = None,
    include: Sequence[str] | None = None,
    dynamic: bool = False,
) -> dict[str, set[int]]:
    """{repo-relative production file: executed line numbers} for one test set.

    `targets` are node ids or file paths; they go through `@argsfile` (pytest reads one
    argument per line), so a long deleted-id list never hits the argv limit. A `body` dict is
    filled with the lines a TEST FUNCTION reached (coverage's `test_function` dynamic context),
    as opposed to imports and fixtures, which every test of the directory runs too.
    `include` (repo-relative files) traces only those files instead of the whole source set, and
    `dynamic` records the dynamic contexts without reading them back here: the data file
    (`coverage_data(dest, label)`) is queried per line later (`Selector`).
    """
    if not targets:
        return {}
    argsfile = dest / f".parity-{label}-args.txt"
    argsfile.write_text("".join(f"{target}\n" for target in targets), encoding="utf-8")
    data = coverage_data(dest, label)
    report = dest / f".parity-{label}-coverage.json"
    rcfile = dest / f".parity-{label}.coveragerc"
    rcfile.write_text(
        "[run]\n"
        + ("dynamic_context = test_function\n" if body is not None or dynamic else ""),
        encoding="utf-8",
    )
    scope = f"--include={','.join(include)}" if include is not None else f"--source={COVERAGE_SOURCE}"
    measured = run(
        [
            *python,
            "-m",
            "coverage",
            "run",
            f"--data-file={data}",
            f"--rcfile={rcfile}",
            scope,
            "-m",
            "pytest",
            *PYTEST_ARGS,
            "@" + str(argsfile),
        ],
        dest,
        dict(env),
        timeout,
    )
    if measured.error or measured.timed_out or measured.exit_code != 0:
        detail = measured.error or (
            "timed out" if measured.timed_out else f"pytest exit {measured.exit_code}"
        )
        raise ParityError(f"coverage {label}: {detail}: {_tail(measured.output)}")
    dump = run(
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
        dest,
        dict(env),
        timeout,
    )
    if dump.error or dump.timed_out or dump.exit_code != 0:
        detail = dump.error or ("timed out" if dump.timed_out else f"json exit {dump.exit_code}")
        raise ParityError(f"coverage {label}: {detail}: {_tail(dump.output)}")
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
    """((mutant, priority) for at most `limit` covered lines, priority 1 filled first, template-string
    code last (priority 2: it has no body-reach proof);
    the changed files that gave no candidate).

    A line of a file changed between base and head is a candidate only when its exact text
    occurs once in both (the mutant is applied by text on each side). A line in
    not in `body_exec` (reached only by imports or fixtures, never by a deleted test's body) is
    dropped unless `named(path)`
    (a changed test file imports that module itself, so its constants are what it tests)."""
    tiers: tuple[dict[str, list[Mutant]], ...] = ({}, {}, {})
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
        # a template body is exec'd under a synthetic module name, so coverage cannot see its
        # lines, only the `_lines(` statement: its code is a candidate when that ran, but with
        # no body-reach proof it fills spare budget only (third bucket, after measured lines)
        templated = {
            n
            for start, code in mutate.template_code_lines(source).items()
            if start in lines
            for n in code
        }
        if body_exec is not None and not named(path):
            lines = lines & body_exec.get(path, set())
        lines = lines | templated
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
            if lineno in templated:
                tier = tiers[2]
            else:
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
    picked += [(m, 2) for m in _pick(tiers[2], limit - len(picked))]
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

    def selected(self, ids: Sequence[str] | None, tag: str) -> list[str]:
        """The command for `ids` (a pytest `@argsfile`: a hot line can name thousands), or for
        every changed test file of the side when `ids` is None."""
        if ids is None:
            return self.command()
        argsfile = self.dest / f".parity-sel-{tag}.txt"
        argsfile.write_text("".join(f"{node_id}\n" for node_id in ids), encoding="utf-8")
        return [*self.cmd, "@" + str(argsfile)]


@dataclass(frozen=True)
class Slot:
    """One mutant at a time runs in a slot: its own base and head copy (a mutant edits the copy)."""

    base: Side
    head: Side


def clone_tree(src: Path, dst: Path) -> None:
    """Copy-on-write clone where the filesystem has it (APFS `cp -c`, reflink), else a plain copy.

    The copy is independent of `src` either way: a mutant edits files in place in a slot's copy."""
    flags = ["-cR"] if sys.platform == "darwin" else ["-a", "--reflink=auto"]
    proc = subprocess.run(["cp", *flags, str(src), str(dst)], capture_output=True, text=True)
    if proc.returncode != 0:
        shutil.rmtree(dst, ignore_errors=True)
        # a plain copy of the data dirs is 11 GB: leave them out, as mutate.clone_data_dirs
        # does without copy-on-write
        shutil.copytree(
            src,
            dst,
            symlinks=True,
            ignore=lambda folder, names: [
                n for n in names if os.path.relpath(os.path.join(folder, n), src) in mutate.DATA_LINKS
            ],
        )
    # never a link to the real data, even from a tree an older mutate.py built
    for rel in mutate.DATA_LINKS:
        if (dst / rel).is_symlink():
            (dst / rel).unlink()
        elif (dst / rel).is_dir() and mutate.holds_symlink(dst / rel):
            shutil.rmtree(dst / rel)  # a link inside the copy still reaches the real file


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


def both_invalid(item: Screened) -> bool:
    return item.base == "INVALID" and item.head == "INVALID"


class BudgetStop(RuntimeError):
    """The --budget wall cap was hit in `phase`: no verdict (exit 2 on an error, 1 on a loss, else 3)."""

    def __init__(self, phase: str) -> None:
        super().__init__(phase)
        self.phase = phase


class Budget:
    """The run's wall-clock cap (`--budget SEC`): a deadline every run is sized against.

    `limit` 0 means no cap. `remaining()` is the seconds left or None; each run's timeout is
    min(`--timeout`, remaining). A run cut short by that clamp is no verdict at all: a TIMEOUT
    counts KILLED otherwise, so `BudgetedRun` turns the clamped timeout into a BudgetStop."""

    def __init__(self, limit: float, now: Callable[[], float]) -> None:
        self.limit = limit
        self.now = now
        self.start = now()
        self.phase = "collect"

    def remaining(self) -> float | None:
        if self.limit <= 0:
            return None
        return self.start + self.limit - self.now()

    def check(self) -> None:
        left = self.remaining()
        if left is not None and left <= 0:
            raise BudgetStop(self.phase)

    def clamp(self, timeout: float) -> tuple[float, bool]:
        """(the timeout to use, was it cut short by the remaining budget)."""
        left = self.remaining()
        if left is None or left >= timeout:
            return timeout, False
        return left, True


class BudgetedRun:
    """The run callable every phase gets: enforces the deadline and sizes each subprocess."""

    def __init__(self, run: Callable[..., RunResult], budget: Budget) -> None:
        self.run = run
        self.budget = budget

    def __call__(
        self,
        cmd: Sequence[str],
        cwd: Path,
        env: Mapping[str, str],
        timeout: float,
    ) -> RunResult:
        self.budget.check()
        capped, clamped = self.budget.clamp(timeout)
        result = self.run(cmd, cwd, env, capped)
        if clamped and result.timed_out:
            raise BudgetStop(self.budget.phase)
        return result


class Progress:
    """The `parity: <phase> ...` stderr lines and the counts a BudgetStop reports.

    `checked` counts mutants whose comparison finished (a dropped INVALID-at-both is taken back
    out), `total` how many the run meant to compare, and `lost`/`errors` accumulate as they are
    confirmed, so a run cut short still reports what it proved."""

    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        self.total = 0
        self.checked = 0
        self.lost = 0
        self.errors = 0
        self.index = 0
        self.lock = threading.Lock()  # screens run in parallel slots

    def start(self, phase: str, detail: str) -> None:
        self.budget.phase = phase
        self._say(f"{phase} {detail}")

    def mutant(self, phase: str, index: int, total: int, name: str) -> None:
        elapsed = self.budget.now() - self.budget.start
        self._say(f"{phase} {index}/{total} {name} {elapsed:.1f}s")

    def error(self) -> None:
        """One side's error, counted the moment it is seen: a BudgetStop on the mutant's other
        side must not lose it (fail closed)."""
        with self.lock:
            self.errors += 1

    def screened(self, item: Screened) -> None:
        with self.lock:
            self.checked += 1  # its errors were counted by error() as each side finished

    def dropped(self, item: Screened) -> None:
        self.checked -= 1
        self.errors -= item.errors

    def _say(self, text: str) -> None:
        print(f"parity: {text}", file=sys.stderr, flush=True)


def screen_replacing(
    first: Sequence[Mutant],
    more: Callable[[], Sequence[Mutant]],
    screen_some: Callable[[Sequence[Mutant]], list[Screened]],
    progress: Progress | None = None,
) -> tuple[list[Screened], int]:
    """(screened mutants with a verdict, how many INVALID-at-both were dropped).

    Each mutant INVALID at base AND head is replaced by the next unseen one from `more()`
    (asked once, only when needed) until the slots are full or the candidates run out."""
    kept: list[Screened] = []
    dropped = 0
    queue: list[Mutant] = []
    seen = {m.name or m.file for m in first}
    asked = False
    batch = list(first)
    slots = len(batch)
    while batch:
        for item in screen_some(batch):
            if both_invalid(item):
                dropped += 1
                if progress is not None:
                    progress.dropped(item)
                print(f"dropped {item.mutant.name or item.mutant.file}: INVALID at base and head")
            else:
                kept.append(item)
        need = slots - len(kept)
        if need <= 0:
            break
        if not asked:
            asked = True
            queue = [m for m in more() if (m.name or m.file) not in seen]
        batch, queue = queue[:need], queue[need:]
        seen.update(m.name or m.file for m in batch)
    return kept, dropped


@dataclass(frozen=True)
class Selection:
    """The head test ids one mutant runs; None = every changed test file of the head copy.

    The base side has no selection: it always runs every changed test file that is not provably
    unchanged (see `unchanged_test_files`)."""

    head: tuple[str, ...] | None = None


# Reads the coverage data files of a run and prints {file: {line: [dynamic contexts]}} for the
# wanted lines only (the JSON report with contexts for the whole suite would be gigabytes).
_CONTEXT_SCRIPT = """
import json, sys
from pathlib import Path
from coverage import CoverageData
root = Path(sys.argv[1])
wanted = json.loads(Path(sys.argv[2]).read_text())
lines = {path: {int(n) for n in numbers} for path, numbers in wanted.items()}
names = {str((root / path).resolve()): path for path in lines}
found = {path: {} for path in lines}
for data_file in sys.argv[3:]:
    data = CoverageData(basename=data_file)
    data.read()
    for name in data.measured_files():
        path = names.get(str(Path(name).resolve()))
        if path is None:
            continue
        for number, labels in data.contexts_by_lineno(name).items():
            if number in lines[path]:
                found[path].setdefault(str(number), set()).update(l for l in labels if l)
print(json.dumps({p: {n: sorted(v) for n, v in d.items()} for p, d in found.items()}))
"""


def context_labels(
    python: Sequence[str],
    dest: Path,
    env: Mapping[str, str],
    data_files: Sequence[Path],
    wanted: Mapping[str, Sequence[int]],
    timeout: float,
    run: Callable[..., RunResult],
    tag: str,
) -> dict[str, dict[int, set[str]]]:
    """{file: {line: dynamic-context labels that executed it}} for the `wanted` lines."""
    if not wanted or not data_files:
        return {}
    wanted_file = dest / f".parity-wanted-{tag}.json"
    wanted_file.write_text(json.dumps({k: list(v) for k, v in wanted.items()}), encoding="utf-8")
    result = run(
        [*python, "-c", _CONTEXT_SCRIPT, str(dest), str(wanted_file), *map(str, data_files)],
        dest,
        dict(env),
        timeout,
    )
    payload = next(
        (ln for ln in reversed(result.output.splitlines()) if ln.startswith("{")), None
    )
    if result.error or result.timed_out or result.exit_code != 0 or payload is None:
        detail = result.error or ("timed out" if result.timed_out else _tail(result.output))
        raise ParityError(f"coverage contexts ({tag}): {detail}")
    return {
        path: {int(line): set(labels) for line, labels in lines.items()}
        for path, lines in json.loads(payload).items()
    }


def context_names(node_id: str) -> set[str]:
    """The dynamic-context labels coverage can give the test of a node id.

    `dynamic_context = test_function` names a context `<module>.<Class>.<test>` with the module
    as pytest imported it (`test_x` with rootdir-relative imports, `tests.unit.test_x` with
    packages), so every dotted suffix of the file path is a candidate."""
    file, qualname, _param = split_node_id(node_id)
    parts = file[: -len(".py")].split("/") if file.endswith(".py") else file.split("/")
    tail = qualname.replace("::", ".")
    return {".".join(parts[i:]) + "." + tail for i in range(len(parts))}


def _mutant_lines(root: Path, mutant: Mutant) -> list[int] | None:
    """The line numbers `mutant.old` covers in `root`'s copy of the file; None if not unique."""
    try:
        text = (root / mutant.file).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if text.count(mutant.old) != 1:
        return None
    first = text.count("\n", 0, text.index(mutant.old)) + 1
    return list(range(first, first + mutant.old.count("\n") + 1))


class Selector:
    """Which head tests to run for a mutant.

    The tests whose coverage context executed the mutant's line; when no context reached it (an
    import-time constant) or a label names no collected test, the changed head test files that
    name the mutated module, else every changed head test file. A test that reaches the line only
    through a fixture is skipped, so the head side may report a survivor too much; confirm()
    re-runs every such candidate against all test files naming the module, which is the safety
    net. The base side is never narrowed: a coverage context cannot prove which base tests kill a
    mutant (fixtures, subprocesses and changed production code all hide a killer)."""

    def __init__(
        self,
        python: Sequence[str],
        head: Side,
        data: Sequence[Path],
        ids: Sequence[str],
        naming: Callable[[str], list[str]],
        timeout: float,
        run: Callable[..., RunResult],
    ) -> None:
        self.python = python
        self.head = head
        self.data = data
        self.naming = naming
        self.timeout = timeout
        self.run = run
        self.label_ids: dict[str, list[str]] = {}
        for node_id in ids:
            for label in context_names(node_id):
                self.label_ids.setdefault(label, []).append(node_id)
        self._lines: dict[str, list[int] | None] = {}
        self._labels: dict[str, dict[int, set[str]]] = {}

    def prepare(self, mutants: Sequence[Mutant]) -> None:
        """Query the contexts of every line the `mutants` touch that is not known yet."""
        wanted: dict[str, list[int]] = {}
        for mutant in mutants:
            key = mutant.name or mutant.file
            if key in self._lines:
                continue
            self._lines[key] = _mutant_lines(self.head.dest, mutant)
            lines = self._lines[key]
            if lines is not None:
                wanted.setdefault(mutant.file, []).extend(lines)
        for path, by_line in context_labels(
            self.python,
            self.head.dest,
            self.head.env,
            self.data,
            wanted,
            self.timeout,
            self.run,
            f"head-{len(self._lines)}",
        ).items():
            self._labels.setdefault(path, {}).update(by_line)

    def _reached(self, mutant: Mutant) -> tuple[str, ...] | None:
        """The head test ids whose context executed the mutant's line; None if unproven."""
        lines = self._lines.get(mutant.name or mutant.file)
        known = self._labels.get(mutant.file, {})
        labels = set().union(*(known.get(n, set()) for n in lines)) if lines else set()
        found: set[str] = set()
        for label in labels:
            mapped = self.label_ids.get(label)
            if mapped is None:
                return None
            found.update(mapped)
        return tuple(sorted(found)) if found else None

    def get(self, mutant: Mutant) -> Selection:
        head = self._reached(mutant)
        if head is None:
            head = tuple(self.naming(mutant.file)) or None
        return Selection(head)


def _count(ids: Sequence[str] | None) -> str:
    return "all" if ids is None else str(len(ids))


def _failed_ids(output: str) -> list[str]:
    return list(dict.fromkeys(FAILED_RE.findall(output)))


def _verified_kill(
    side: Side, output: str, timeout: float, run: Callable[..., RunResult], tag: str
) -> None:
    """A KILLED verdict stands only if the tests that failed under the mutant pass unmutated.

    Replaces the old whole-suite baseline: the failing node ids run again on the restored copy,
    and a kill whose every failure also fails there is no kill but a broken baseline
    (ParityError). A re-run that cannot answer (collection or usage error) leaves the kill."""
    failed = _failed_ids(output)
    if not failed:
        return
    result = run(side.selected(failed, f"{tag}v"), side.dest, side.env, timeout)
    if result.error or result.timed_out:
        raise ParityError(f"{side.label} baseline: {result.error or 'timed out'}")
    if result.exit_code != 1:
        return  # 0: they pass unmutated; 2-5: nothing ran to compare
    still = set(_failed_ids(result.output))
    if not [node for node in failed if node not in still]:
        detail = _first_failed(result.output)
        raise ParityError(f"{side.label} baseline: tests fail without mutation: {detail}")


def _screen_one(
    mutant: Mutant,
    slot: Slot,
    selection: Selection,
    timeout: float,
    run: Callable[..., RunResult],
    progress: Progress | None,
    tag: str,
) -> tuple[Screened, list[str]]:
    """One mutant at base and, unless base survived, at head in `slot`; (verdicts, printed lines).

    A mutant no base test kills cannot lose a kill, so the head side is skipped there."""
    states: dict[str, str] = {}
    details: list[str] = []
    errors = 0
    node = ""
    for side, ids in ((slot.base, None), (slot.head, selection.head)):
        if not side.files:
            states[side.label] = "SURVIVED"
            continue
        if side.label == "head" and states.get("base") == "SURVIVED":
            states["head"] = "SKIPPED"
            continue
        recorder = _Recorder(run)
        _killed, is_error, line = mutate.check_mutant(
            side.dest, mutant, side.selected(ids, tag), side.env, timeout, recorder, True, False
        )
        state, state_error = _verdict(line)
        if state == "KILLED" and recorder.last is not None and not line.startswith("TIMEOUT"):
            _verified_kill(side, recorder.last.output, timeout, run, f"{tag}{side.label}")
        states[side.label] = state
        side_error = int(is_error or state_error)
        errors += side_error
        if side_error and progress is not None:
            progress.error()
        if state_error:
            details.append(f"  {side.label}: {line}")
        elif side.label == "base" and state == "KILLED" and recorder.last is not None:
            node = _first_failed(recorder.last.output)
    report = f"base={states['base']} head={states['head']} {mutant.name or mutant.file}"
    if states["base"] == "KILLED" and states["head"] == "SURVIVED":
        report += "  LOST KILL"  # mutate.py --compare-rev's format; confirmed below
    item = Screened(mutant, states["base"], states["head"], node, errors)
    if progress is not None:
        progress.screened(item)
    return item, [report, *details]


def screen(
    mutants: Sequence[Mutant],
    base: Side,
    head: Side,
    timeout: float,
    run: Callable[..., RunResult],
    progress: Progress | None = None,
    selector: Selector | None = None,
    slots: Sequence[Slot] | None = None,
) -> list[Screened]:
    """Each mutant at base and (unless base survived) at head.

    `selector` picks the tests each mutant runs (see Selector; None = every changed test file).
    `slots` are independent copies: one mutant runs in each at the same time, and the lines print
    in candidate order. Each KILLED verdict is checked against the unmutated copy."""
    if selector is not None:
        selector.prepare(mutants)
    selections = [selector.get(m) if selector is not None else Selection() for m in mutants]
    for mutant, selection in zip(mutants, selections, strict=True):
        print(
            f"parity: select {mutant.name or mutant.file} head={_count(selection.head)}",
            file=sys.stderr,
        )
    pool = list(slots) if slots else [Slot(base, head)]
    free: Queue[Slot] = Queue()
    for slot in pool:
        free.put(slot)
    results: dict[int, tuple[Screened, list[str]]] = {}
    printed = 0

    def task(index: int) -> tuple[Screened, list[str]]:
        mutant = mutants[index]
        if progress is not None:
            with progress.lock:
                progress.index += 1
                progress.total = max(progress.total, progress.index)
                position, total = progress.index, progress.total
            progress.mutant("screen", position, total, mutant.name or mutant.file)
        slot = free.get()
        try:
            return _screen_one(
                mutant, slot, selections[index], timeout, run, progress, f"m{index}"
            )
        finally:
            free.put(slot)

    def flush() -> None:
        nonlocal printed
        while printed in results:
            for line in results[printed][1]:
                print(line)
            printed += 1

    if len(pool) == 1:
        for index in range(len(mutants)):
            results[index] = task(index)
            flush()
    else:
        with ThreadPoolExecutor(max_workers=len(pool)) as executor:
            pending = {executor.submit(task, index): index for index in range(len(mutants))}
            try:
                while pending:
                    finished, _ = wait(set(pending), return_when=FIRST_COMPLETED)
                    for future in finished:
                        results[pending.pop(future)] = future.result()
                    flush()
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
    return [results[index][0] for index in range(len(mutants))]


def module_needles(path: str) -> tuple[str, str | None]:
    """(dotted module path, "package|name" for a `from package import name`) of a file."""
    stem = path[: -len(".py")] if path.endswith(".py") else path
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    dotted = stem.replace("/", ".")
    package, _, name = dotted.rpartition(".")
    return dotted, (f"{package}|{name}" if package else None)


@lru_cache(maxsize=4096)
def _import_pairs(text: str) -> frozenset[tuple[str, str]] | None:
    """Every absolute `from <module> import <name>` pair of `text` (None if it does not parse).

    Cached per text: _namer asks the same test file about every mutated module."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    return frozenset(
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None
        for alias in node.names
    )


def _imports_from(text: str, package: str, name: str, fallback: re.Pattern[str]) -> bool:
    """Does `text` hold `from <package> import ... <name> ...`, on one line or several?

    The import statements are parsed, so a parenthesised multi-line form counts and a comment
    does not; a file that does not parse is read by the single-line `fallback` regex."""
    pairs = _import_pairs(text)
    if pairs is None:
        return fallback.search(text) is not None
    return (package, name) in pairs


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


def confirm_candidates(screened: Sequence[Screened]) -> list[Screened]:
    """The `base=KILLED head=SURVIVED` screens: the only ones confirm() has to re-run."""
    return [item for item in screened if item.base == "KILLED" and item.head == "SURVIVED"]


def confirm(
    screened: Sequence[Screened],
    head: Side,
    timeout: float,
    run: Callable[..., RunResult],
    progress: Progress | None = None,
) -> tuple[int, int]:
    """(lost, errors) after re-running every candidate against every test naming its module.

    The enlarged test set must pass UNMUTATED first (once per distinct set): a test that
    already fails says nothing about the mutant, so a failing set is an error, never "covered
    elsewhere". After a passing baseline, a mutant that only breaks COLLECTION (pytest exit 2-5,
    e.g. a deleted definition another test imports) is "covered elsewhere" with that error."""
    lost = errors = 0
    baselines: dict[tuple[str, ...], str] = {}
    candidates = confirm_candidates(screened)
    for index, item in enumerate(candidates, 1):
        name = item.mutant.name or item.mutant.file
        if progress is not None:
            progress.mutant("confirm", index, len(candidates), name)
        item_lost = item_errors = 0
        files = tuple(sorted(set(head.files) | set(test_files_naming(head.dest, item.mutant.file))))
        if not files:
            item_lost = 1
            print(f"LOST KILL {name} (killed at base by {item.node})")
        else:
            if files not in baselines:
                baselines[files] = _importer_baseline(head, files, timeout, run)
            if baselines[files]:
                item_errors = 1
                print(f"  confirm: {name}: tests fail without the mutant: {baselines[files]}")
            else:
                item_lost, item_errors = _confirm_one(item, head, files, timeout, run)
        if item_lost:
            lost += 1
        errors += item_errors
        if progress is not None:
            progress.lost += item_lost
            progress.errors += item_errors
    return lost, errors


def _confirm_one(
    item: Screened, head: Side, files: Sequence[str], timeout: float, run: Callable[..., RunResult]
) -> tuple[int, int]:
    """(lost, errors) for one candidate against the enlarged set, which already passed."""
    name = item.mutant.name or item.mutant.file
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
        return 0, 0
    if state == "SURVIVED":
        print(f"LOST KILL {name} (killed at base by {item.node})")
        return 1, 0
    if state == "KILLED":
        node = _first_failed(last.output) if last is not None else "unknown"
        print(f"covered elsewhere {name} ({node})")
        return 0, 0
    print(f"  confirm: {line}")
    return 0, int(is_error or state_error)


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


def run(
    args: argparse.Namespace,
    runner: Callable[..., RunResult],
    now: Callable[[], float] = time.monotonic,
) -> int:
    repo = Path(args.repo).resolve()
    check_base(repo, args.base)
    python = test_python(repo)
    _require(python)
    budget = Budget(args.budget, now)
    metered = BudgetedRun(runner, budget)
    progress = Progress(budget)
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
        workers = max(min(args.workers, cpu_budget()), 1)  # never more processes than cores
        progress.start(
            "collect", f"{len(base_files)} base file(s), {len(head_files)} head file(s)"
        )
        chunks = max(workers // 2, 1)
        with ThreadPoolExecutor(max_workers=2) as executor:
            base_job = executor.submit(
                collect_parallel, python, base_dir, base_env, base_files, args.timeout, metered, chunks
            )
            head_job = executor.submit(
                collect_parallel, python, head_dir, head_env, head_files, args.timeout, metered, chunks
            )
            base_ids, head_ids = base_job.result(), head_job.result()
        head_set = set(head_ids)
        deleted = [node for node in base_ids if node not in head_set]
        # a text-identical test is the same test only while nothing a test can import changed
        # and its whole file is unchanged; otherwise every missing id is deleted and the base
        # side runs every changed test file (the original behaviour)
        unchanged: set[str] = set()
        if deleted:
            imported = changed_non_test_paths(repo, args.base, args.head)
            if imported:
                print(
                    f"parity: {len(imported)} non-test file(s) changed ({imported[0]}...): "
                    "renames and the base screen are not narrowed",
                    file=sys.stderr,
                )
            else:
                unchanged = unchanged_test_files(base_dir, head_dir, changed)
        renamed = renamed_ids(base_dir, head_dir, deleted, base_ids, head_ids, unchanged)
        if renamed:
            print(
                f"parity: {len(renamed)} of {len(deleted)} missing test id(s) are AST-identical "
                "renames, not deleted",
                file=sys.stderr,
            )
            deleted = [node for node in deleted if node not in renamed]
        if not deleted:
            print("PARITY no deleted tests")
            return 0

        # the base screen skips only files that cannot have changed behaviour
        screen_files = [path for path in base_files if path not in unchanged]
        progress.start("coverage", f"{len(deleted)} deleted test(s)")
        body_exec: dict[str, set[int]] = {}
        base_exec = coverage_lines(
            python, base_dir, base_env, deleted, "base", args.timeout, metered, body_exec
        )
        test_texts = [
            (base_dir / path).read_text(encoding="utf-8", errors="replace")
            for path in base_files
        ]
        head_texts = {
            path: (head_dir / path).read_text(encoding="utf-8", errors="replace")
            for path in head_files
        }
        traced = sorted(path for path, lines in base_exec.items() if is_production(path) and lines)
        head_data: list[Path] = []
        head_exec: dict[str, set[int]] = {}
        if head_files and traced:
            per_file: dict[str, int] = {}
            for node_id in head_ids:
                per_file[split_node_id(node_id)[0]] = per_file.get(split_node_id(node_id)[0], 0) + 1
            head_exec, head_data = coverage_sharded(
                python, head_dir, head_env, head_files, "head", args.timeout, metered, workers,
                traced, per_file,
            )
        def naming(path: str) -> list[str]:
            names = _namer(path)
            return [f for f in head_files if names(head_texts[f])]

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

        base = Side("base", base_dir, _pytest(python, workers), base_env, screen_files)
        head = Side("head", head_dir, _pytest(python, workers), head_env, head_files)
        # one mutant per slot, each slot its own pair of copies; a slot's pytest runs share the
        # worker budget, so the pool never holds more processes than --workers
        slot_count = max(min(workers, len(picks)), 1)
        per_run = max(workers // slot_count, 1)
        slots = [
            Slot(
                replace(base, cmd=_pytest(python, per_run)),
                replace(head, cmd=_pytest(python, per_run)),
            )
        ]
        for k in range(1, slot_count):
            sides: list[Side] = []
            for side in (base, head):
                (root / f"slot{k}").mkdir(exist_ok=True)
                copy = root / f"slot{k}" / side.label
                clone_tree(side.dest, copy)
                sides.append(
                    replace(side, dest=copy, env=_env(copy), cmd=_pytest(python, per_run))
                )
            slots.append(Slot(sides[0], sides[1]))
        selector = Selector(python, head, head_data, head_ids, naming, args.timeout, metered)

        def screen_some(batch: Sequence[Mutant]) -> list[Screened]:
            return screen(
                batch,
                base,
                head,
                args.timeout,
                metered,
                progress=progress,
                selector=selector,
                slots=slots,
            )

        def wider() -> list[Mutant]:
            more, _skipped = candidate_mutants(
                base_dir, head_dir, base_exec, head_exec, args.max * 4, body_exec,
                lambda path: any(map(_namer(path), test_texts)),
            )  # fmt: skip
            return [mutant for mutant, _tier in more]

        progress.start("screen", f"{len(picks)} mutant(s)")
        progress.total = len(picks)
        screened, dropped = screen_replacing(
            [m for m, _tier in picks], wider, screen_some, progress
        )
        progress.start("confirm", f"{len(confirm_candidates(screened))} candidate(s)")
        lost, errors = confirm(screened, head, args.timeout, metered, progress)
        errors += sum(item.errors for item in screened)
        if not screened and dropped:
            print(f"  no valid comparison: all {dropped} candidates INVALID at base and head")
            errors += 1
        print(f"PARITY compared={len(screened)} lost={lost} errors={errors}")
        if errors:
            return 2
        return 1 if lost else 0
    except BudgetStop as stop:
        print(
            f"PARITY INCOMPLETE compared={progress.checked} of {progress.total} "
            f"lost={progress.lost} errors={progress.errors} "
            f"(budget {budget.limit:g}s hit in {stop.phase})"
        )
        if progress.errors:
            return 2
        return 1 if progress.lost else 3
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main(
    argv: Sequence[str] | None = None,
    run_fn: Callable[..., RunResult] | None = None,
    now: Callable[[], float] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="parity_mutants.py",
        description="Did a test-deleting diff lose a mutant kill?",
        epilog=(
            "Prints `base=<KILLED|SURVIVED> head=<...> <name>` per mutant, then `LOST KILL "
            "<name>\n(killed at base by <node id>)` for a loss confirmed against every test "
            "file that names\nthe module, or `covered elsewhere <name> (<node id>)`. The last "
            "line is always\n`PARITY compared=N lost=K errors=E`; exit 0 nothing lost, 1 lost "
            "kill, 2 error. A `--budget` cap that runs out prints `PARITY INCOMPLETE "
            "compared=i of N\nlost=K errors=E (budget Bs hit in <phase>)`: exit 2 on an "
            "error, else 1 on a\nconfirmed loss, else 3 (the gate shows 3 as a WARN)."
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
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="pytest-xdist workers for the non-coverage runs (default and cap: the "
        "host's cores; 0 or 1 = serial)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=0.0,
        help="wall-clock cap for the whole run, seconds (default 0 = no cap); an exhausted "
        "cap prints PARITY INCOMPLETE and exits 3",
    )
    args = parser.parse_args(argv)
    if args.max < 1:
        parser.error("--max must be >= 1")
    if args.workers is None:
        args.workers = cpu_budget()
    if args.workers < 0:
        parser.error("--workers must be >= 0")
    if args.budget < 0:
        parser.error("--budget must be >= 0")
    try:
        return run(args, run_fn or mutate.run_command, now or time.monotonic)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(f"PARITY error: {' '.join(str(exc).split())[:200]}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
