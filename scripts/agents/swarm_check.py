#!/usr/bin/env python3
"""Pre-review gate for a worker's worktree — run by the orchestrator before any review.

    swarm-check.sh <worktree> [--base REF] [--scope glob,glob,...]
                   [--exactness auto|always|never] [--frontend auto|always|never]
                   [--json out.json] [--no-cache] [--bus DIR] [--bus-agent NAME]

Checks (each prints "PASS|FAIL|SKIP <name>: <detail>"), then "GATE PASS" /
"GATE FAIL (n failed)" as the last line; exit 0/1. Default base: main;
default exactness/frontend: auto (run only when relevant paths changed).

bus-inbox: with --bus (or $SWARM_BUS) and --bus-agent (or $SWARM_AGENT) both known the gate
peeks that agent's swarm inbox and FAILs while any message from "chief" is unread (each id +
full body listed, ending "run: bus inbox"); it never advances a cursor. It is evaluated
BEFORE any tool runs (a DM already waiting refuses the hand-back without paying for the suite)
AND again after every tool check, right before the verdict, so a DM that lands mid-run still
blocks. SKIP only when neither is configured (or only a bus, no agent); an agent without a
bus, a bus dir that does not exist, or an unreadable/garbled cursor is a FAIL with the reason.
The gate result is posted to the resolved bus (flag or env) and never creates a bus dir: a
missing one is skipped with a note on stderr. Time-dependent, so it runs even when the tool
checks are served from a cached PASS and is never written into the stamp.

PASS-stamp cache: a GATE PASS on a clean worktree writes
`<git-dir>/swarm-check/pass-<key[:16]>.json` in that worktree's git dir (per-worktree,
never committed, invisible to `git status`). The key is a sha256 over HEAD, the base
commit, the sorted scope globs, exactness/frontend modes, this script's own bytes and the
sha256 of the MAIN checkout's scripts/agents/exactness_239.py,
scripts/agents/exactness_carry.py, uv.lock and frontend/pnpm-lock.yaml (a missing file
hashes as a fixed token), so any change to the
diff or to the checks invalidates it. A later run whose tree is clean and whose key
matches prints the stored check lines as `(cached) ...`, then
`PASS cache: reused stamp from <ts> for <head[:12]>` and `GATE PASS` (and posts
"GATE PASS (cached <head12>)" to the bus), without running any tool.
The stamp is written only when the key computed BEFORE the run is non-empty (clean start),
AND equals the key recomputed after it (nothing moved while the tools ran: a commit made
mid-run would otherwise stamp a HEAD that was never gated), AND the HEAD reflog length is
unchanged (`git rev-list --walk-reflogs --count HEAD`). The reflog guard catches a
commit + `reset --hard` made mid-run: it leaves the key identical but adds reflog entries.
Writes are atomic (temp file + os.replace). A tool FAIL, a WARN, a dirty tree or a stub
blocks the stamp, but a bus-inbox-only FAIL does not: the stored checks exclude the bus line
(time-dependent), so the next run after `bus inbox` reuses the tool PASS instead of re-running
the suite, and the bus check itself re-runs on every invocation. load_stamp accepts only
PASS/SKIP lines, and any problem reading a stamp falls back to the full gate (fail closed:
never a false PASS). `--no-cache` disables both reuse and writing.
Known limit (accepted): the reflog guard cannot see an edit + revert of the working tree
with no commit - the tree and HEAD return to their start state and no reflog entry is added,
so such a run is still stamped.
Known limit (accepted): with core.logAllRefUpdates=false and no HEAD reflog file, git writes
no reflog, so head_reflog_count is 0 before and after (0 == 0) and the reflog guard is blind:
a commit + reset --hard made mid-run is stamped (the pre-N1 behaviour). Measured on a fresh
repo with the setting off: count stays 0 across commit, commit, reset --hard. Git's default
(true) keeps the guard working.
NOT in the key (known gaps; --no-cache to force): the main checkout's .venv (tool
versions), data/catalog (exactness inputs), the frontend lint baseline in main's
working tree and any uncommitted edit in main other than the keyed files; and any
.gitignore'd input, which is part of neither the clean check nor the key.

The gate FAILS CLOSED: a bad --base, any git error, a timed-out or missing tool is a
FAIL, never a PASS. Exactness always runs the MAIN checkout's exactness_239.py (the
worktree copy must be identical or the check FAILs). exactness-carry does the same with
the MAIN checkout's scripts/agents/exactness_carry.py, except that a main without that
script is a SKIP, not a FAIL (it lands with the carry engine; the trigger set is the
funding-carry paths listed in EXACTNESS_CARRY_TRIGGERS).

Tool commands can be stubbed via env (full command, shell-quoted):
SWARM_CHECK_RUFF / SWARM_CHECK_MYPY / SWARM_CHECK_PYTEST (the pytest stub replaces
the "<python> -m pytest" prefix and keeps the arguments). Stubs are refused (FAIL,
real tools used) unless SWARM_CHECK_ALLOW_STUBS=1; active stubs are reported.
Stdlib only; runs with any python3.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import fnmatch
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from collections.abc import Set as AbstractSet
    from typing import Any

    RunResult = tuple[int, str]
    RunFn = Callable[..., RunResult]  # (cmd, cwd, env=None, timeout=TOOL_TIMEOUT)

# Exactness triggers on ANY change under vibe_quant/ except these packages. Allowlist =
# packages outside the transitive import closure of screening/nt_runner, discovery/pipeline
# and validation/runner (checked with an import walk 2026-10-10), so they cannot move
# backtest numbers. Re-verify before adding to it.
EXACTNESS_SAFE_DIRS = (
    "vibe_quant/api/",
    "vibe_quant/paper/",
    "vibe_quant/research/",
    "vibe_quant/alerts/",
    "vibe_quant/ethereal/",
    "vibe_quant/jobs/",
    "vibe_quant/reconciliation.py",
    "vibe_quant/risk/",
    "vibe_quant/strategies/",
)
EXACTNESS_FILES = (
    "pyproject.toml",
    "uv.lock",
    "scripts/agents/exactness_239.py",
)
EXACTNESS_SCRIPT = "scripts/agents/exactness_239.py"
# exactness-carry: a SECOND, independent exactness proof for the funding-carry engine
# (vibe_quant/portfolio/ + the funding settlement code it pins). A trailing "/" in a
# trigger is a path prefix, anything else is an exact path. Kept separate from the
# exactness_239 trigger above: the carry engine's own paths are the only ones that move
# the carry numbers, and the script arrives with that engine (main may not have it yet,
# which is a SKIP, never a FAIL).
EXACTNESS_CARRY_SCRIPT = "scripts/agents/exactness_carry.py"
EXACTNESS_CARRY_TRIGGERS = (
    "vibe_quant/portfolio/",
    "vibe_quant/validation/funding.py",
    EXACTNESS_CARRY_SCRIPT,
    "pyproject.toml",
    "uv.lock",
)
CHECK_ORDER = (
    "scope", "clean", "bus-inbox", "ruff", "mypy", "tests", "exactness", "exactness-carry",
    "frontend", "stubs", "internal",
)
UNTRACKED_OK = ("cmd-report.md",)
LINT_ERROR_RE = re.compile(r"^Found (\d+) errors?\.", re.MULTILINE)
SUMMARY_RE = re.compile(r"passed|failed|error|errors|no tests ran")
SKIP_STEMS = frozenset({"__init__", "conftest", "__main__"})
STUB_VARS = ("SWARM_CHECK_RUFF", "SWARM_CHECK_MYPY", "SWARM_CHECK_PYTEST")
TEST_TIMEOUT = 1800  # seconds: pytest, exactness
TOOL_TIMEOUT = 600  # seconds: everything else
PYTEST_NO_TESTS_RC = 5
# Big pytest selections (full suite, -k group) run under xdist: the wall time drops
# ~3x (183s -> ~60s on 10 cores, job 20261011-speed). `worksteal` beat the default
# `load` scheduler (56s vs 66s) once relative-speed tests were pulled out. A handful
# of targeted files stays serial: worker startup costs more than it saves there.
# Tests marked `timing` assert relative wall-clock speed and flake under worker
# contention, so they are deselected from the parallel pass and run serially after.
PYTEST_PARALLEL_ARGS = ("-n", "auto", "--dist", "worksteal")
TIMING_MARK = "timing"
# Cap on the importer-file selection (the reverse import closure of the changed modules).
# Past this many files a per-file targeted run no longer costs less than the same work as
# one whole-suite run under xdist, so the gate runs full tests/unit instead: the closure is
# a way to run LESS work, never a way to skip more of it. MEASURED on the 2026-10-16 tree
# (463 files parsed in 1.2s): 135 importer test files for vibe_quant/metrics.py, 127 for
# errors.py, 50 for discovery/pipeline.py, 15 for paper/node.py - a hub module reaches half
# of tests/unit, and only a full run is cheap enough there. A conftest/__init__ change
# already forces the full suite via needs_full_suite, before any graph is built.
MAX_IMPORTER_FILES = 150
CACHE_DIR = "swarm-check"
CACHE_VERSION = 2
CHECK_STATUSES = ("PASS", "FAIL", "SKIP", "WARN")
STAMP_STATUSES = ("PASS", "SKIP")  # the only line statuses a reusable stamp may contain
MAIN_KEY_FILES = (
    "scripts/agents/exactness_239.py",
    EXACTNESS_CARRY_SCRIPT,
    "uv.lock",
    "frontend/pnpm-lock.yaml",
)
MISSING_FILE_TOKEN = "missing"


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # PASS | FAIL | SKIP | WARN (only FAIL fails the gate)
    detail: str


@dataclass(frozen=True)
class Config:
    worktree: Path
    main_root: Path
    base: str = "main"
    scope: tuple[str, ...] = field(default_factory=tuple)
    exactness: str = "auto"
    frontend: str = "auto"
    json_out: Path | None = None
    bus: str = ""  # job bus dir (--bus / SWARM_BUS) for the bus-inbox check
    bus_agent: str = ""  # whose inbox to peek (--bus-agent / SWARM_AGENT)


@dataclass(frozen=True)
class GitState:
    changed: list[str]  # committed diff + uncommitted incl. untracked (cmd-report.md excluded)
    untracked: list[str]  # untracked files, cmd-report.md excluded
    error: str = ""  # non-empty: the diff is UNKNOWN (bad base / git failure) -> gate FAIL
    uncommitted: list[str] = field(default_factory=list)  # tracked/index changes (not committed)


@dataclass(frozen=True)
class Stamp:
    head: str
    ts: str
    checks: list[Check]


def real_run(
    cmd: Sequence[str],
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout: int = TOOL_TIMEOUT,
) -> RunResult:
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    try:
        proc = subprocess.run(
            [str(c) for c in cmd],
            cwd=str(cwd),
            env=full_env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s: {cmd[0]}"
    except OSError as exc:  # missing binary or missing cwd
        return 127, f"cannot run {cmd[0]}: {exc}"
    return proc.returncode, proc.stdout + proc.stderr


def parse_porcelain(text: str) -> tuple[list[str], list[str], list[str]]:
    """Split `git status --porcelain=v1 -z` into (all changed, untracked, uncommitted).

    NUL-separated, so no quoting. A rename/copy entry is "XY new\\0old\\0": both sides
    are reported so an out-of-scope old path cannot hide behind a rename. `uncommitted`
    is every status entry that is not untracked (a dirty working tree/index) and is used
    by the PASS-stamp cache, which never caches a tree with uncommitted edits.
    """
    changed: list[str] = []
    untracked: list[str] = []
    uncommitted: list[str] = []
    tokens = text.split("\0")
    i = 0
    while i < len(tokens):
        entry = tokens[i]
        i += 1
        if len(entry) < 4:
            continue
        xy, path = entry[:2], entry[3:]
        paths = [path]
        if ("R" in xy or "C" in xy) and i < len(tokens):
            paths.append(tokens[i])
            i += 1
        if xy == "??":
            if path in UNTRACKED_OK:
                continue
            untracked.append(path)
        else:
            uncommitted.extend(p for p in paths if p)
        changed.extend(p for p in paths if p)
    return changed, untracked, uncommitted


def git_state(wt: Path, base: str, run: RunFn) -> GitState:
    """Committed + uncommitted + untracked changes. Fails closed: any git error -> .error."""

    def git(*args: str) -> RunResult:
        return run(["git", "-C", str(wt), *args], wt, None, TOOL_TIMEOUT)

    rc, out = git("rev-parse", "--verify", "--quiet", base + "^{commit}")
    if rc != 0:
        return GitState([], [], f"bad base ref: {base}")
    rc, out = git("merge-base", base, "HEAD")
    if rc != 0:
        return GitState([], [], f"no merge-base between {base} and HEAD: {tail_line(out)}")
    rc, diff_out = git("diff", "--name-only", "--no-renames", "-z", base + "...HEAD")
    if rc != 0:
        return GitState([], [], f"git diff failed: {tail_line(diff_out)}")
    rc, status_out = git("status", "--porcelain=v1", "-z", "--untracked-files=all")
    if rc != 0:
        return GitState([], [], f"git status failed: {tail_line(status_out)}")
    committed = [p for p in diff_out.split("\0") if p]
    changed, untracked, uncommitted = parse_porcelain(status_out)
    seen: dict[str, None] = {}
    for path in committed + changed:
        seen.setdefault(path, None)
    return GitState(changed=list(seen), untracked=untracked, uncommitted=uncommitted)


def scope_offenders(paths: Sequence[str], globs: Sequence[str]) -> list[str]:
    """Paths matching no glob. NB fnmatch: '*' also crosses '/', so 'vibe_quant/*' covers
    vibe_quant/a/b.py."""
    return [p for p in paths if not any(fnmatch.fnmatch(p, g) for g in globs)]


def is_changed_test(path: str) -> bool:
    parts = path.split("/")
    return (
        len(parts) >= 3
        and parts[0] == "tests"
        and parts[-1].startswith("test_")
        and parts[-1].endswith(".py")
    )


def changed_test_files(paths: Sequence[str]) -> list[str]:
    return [p for p in paths if is_changed_test(p)]


def k_expr(paths: Sequence[str]) -> str:
    stems = sorted(
        {
            Path(p).stem
            for p in paths
            if p.endswith(".py") and not is_changed_test(p)
        }
        - SKIP_STEMS
    )
    stems = [st for st in stems if st.isidentifier()]
    return " or ".join(stems)


def needs_full_suite(paths: Sequence[str]) -> bool:
    """True if a changed .py has a stem k_expr drops (conftest/__init__/__main__): no -k
    selection can represent it, so the whole tests/unit must run."""
    return any(p.endswith(".py") and Path(p).stem in SKIP_STEMS for p in paths)


def _dotted(path: str) -> str:
    """Dotted name of a repo-relative .py path (`__init__.py` names its own package)."""
    parts = path[:-3].split("/") if path.endswith(".py") else [path]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def module_name(path: str) -> str:
    """The vibe_quant module a repo-relative path defines, "" when it defines none.

    Path-derived and never read from the tree, so a module DELETED by the diff still seeds
    the reverse closure by name.
    """
    if path.startswith("vibe_quant/") and path.endswith(".py"):
        return _dotted(path)
    return ""


def _is_vibe_quant(name: str) -> bool:
    return name == "vibe_quant" or name.startswith("vibe_quant.")


def _from_base(node: ast.ImportFrom, package: str) -> str | None:
    """Module named as the base of a `from ... import`, None when it is not ours/unknown.

    Relative levels count from the importing file's package: level 1 is that package
    itself, level 2 its parent. A level climbing above the repo root resolves to nothing.
    """
    if not node.level:
        return node.module or ""
    parts = package.split(".") if package else []
    if node.level > len(parts):
        return None
    base = parts[: len(parts) - (node.level - 1)]
    if node.module:
        base += node.module.split(".")
    return ".".join(base)


def parse_imports(
    source: str, module: str, known: AbstractSet[str], *, is_package: bool = False
) -> set[str] | None:
    """The vibe_quant modules `source` imports, or None when it does not parse.

    ast.walk sees imports in ANY scope, so a function-level (lazy) `from ... import f` -
    common in this repo - counts, as does one under `if TYPE_CHECKING:`. Recognised:
    `import a.b`, `import a.b as x`, `from a.b import y`, `from a import b` (b a submodule
    recorded in `known`) and the relative forms resolved against this file's package.
    NO implicit parent edge: `import vibe_quant.a.c` imports vibe_quant.a.c ONLY; adding
    vibe_quant.a too would select every test of that package for any submodule change.
    A `from vibe_quant.a import X` whose X is no module keeps its base (the package
    __init__). Imports of anything outside vibe_quant are dropped.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):  # ValueError: NUL bytes / a decode that slipped through
        return None
    package = module if is_package else module.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names if _is_vibe_quant(alias.name))
        elif isinstance(node, ast.ImportFrom):
            base = _from_base(node, package)
            if base is None or not _is_vibe_quant(base):
                continue
            found.add(base)
            for alias in node.names:
                candidate = f"{base}.{alias.name}"
                if alias.name != "*" and candidate in known:
                    found.add(candidate)
    return found


def build_import_graph(
    worktree: Path, extra_modules: Sequence[str] = ()
) -> dict[str, set[str]]:
    """repo-relative path -> the vibe_quant modules that file imports.

    Nodes: every `vibe_quant/**/*.py` and every `tests/unit/**/test_*.py` of the WORKTREE.
    A file that cannot be read or parsed contributes no edges and never raises: one
    SyntaxError must not crash the gate (ruff, mypy and pytest report it anyway). Only
    vibe_quant modules carry the walk, so a chain through a tests/ helper is out of scope.
    `extra_modules` (the changed files) join the known names: a DELETED module has no file on
    disk, yet `from vibe_quant.pkg import gone` must still resolve to the submodule edge.
    """
    roots = ((worktree / "vibe_quant", "*.py"), (worktree / "tests" / "unit", "test_*.py"))
    rels = sorted(
        p.relative_to(worktree).as_posix()
        for root, pattern in roots
        if root.is_dir()
        for p in root.rglob(pattern)
    )
    known = {_dotted(rel) for rel in rels if module_name(rel)}
    known.update(_dotted(p) for p in extra_modules if module_name(p))
    graph: dict[str, set[str]] = {}
    for rel in rels:
        try:
            source = (worktree / rel).read_text(encoding="utf-8")
        except (OSError, ValueError):
            graph[rel] = set()
            continue
        graph[rel] = parse_imports(
            source, _dotted(rel), known, is_package=rel.endswith("__init__.py")
        ) or set()
    return graph


def importer_test_files(changed: Sequence[str], graph: Mapping[str, set[str]]) -> list[str]:
    """tests/unit test files that reach a changed vibe_quant module (reverse closure).

    Selection by test NAME (`-k <stem>`) misses a caller test whose names never mention the
    module: tests/unit/test_symbol_agg.py::test_topk_ranks_gate_passers_first stayed red
    through DiscoveryPipeline.run while the gate PASSed (20261015-funding-carry pqon2 r1).
    The seeds are the modules the diff names; the walk follows only vibe_quant modules, so
    a test importing another test cannot pull it in. Sorted: the pytest command line must
    not depend on filesystem order.
    """
    seeds = {mod for path in changed if (mod := module_name(path))}
    if not seeds:
        return []
    importers: dict[str, set[str]] = {}
    for path, imports in graph.items():
        if mod := module_name(path):
            for target in imports:
                importers.setdefault(target, set()).add(mod)
    reached = set(seeds)
    frontier = list(seeds)
    while frontier:
        for mod in importers.get(frontier.pop(), ()):
            if mod not in reached:
                reached.add(mod)
                frontier.append(mod)
    return sorted(
        path for path, imports in graph.items() if is_changed_test(path) and imports & reached
    )


def should_run_exactness(paths: Sequence[str]) -> bool:
    return any(
        (p.startswith("vibe_quant/") and not p.startswith(EXACTNESS_SAFE_DIRS))
        or p in EXACTNESS_FILES
        for p in paths
    )


def should_run_exactness_carry(paths: Sequence[str]) -> bool:
    """True if a change can move a funding-carry number: the engine's own paths, its
    pinning script, or a lockfile. Narrower than the exactness_239 trigger on purpose."""
    return any(
        p.startswith(t) if t.endswith("/") else p == t
        for p in paths
        for t in EXACTNESS_CARRY_TRIGGERS
    )


def should_run_frontend(paths: Sequence[str]) -> bool:
    return any(p.startswith("frontend/") for p in paths)


def parse_lint_errors(output: str) -> int:
    return sum(int(n) for n in LINT_ERROR_RE.findall(output))


def has_lint_count(output: str) -> bool:
    return LINT_ERROR_RE.search(output) is not None


def lint_regressed(wt_output: str, main_output: str) -> bool:
    return parse_lint_errors(wt_output) > parse_lint_errors(main_output)


def pytest_summary(output: str) -> str:
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    for line in reversed(lines):
        if SUMMARY_RE.search(line):
            return line
    return lines[-1] if lines else "no output"


def tail_line(output: str) -> str:
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    return lines[-1] if lines else "ok"


def format_line(check: Check) -> str:
    return f"{check.status} {check.name}: {check.detail}"


def gate_summary(checks: Sequence[Check]) -> tuple[str, int]:
    failed = [c for c in checks if c.status == "FAIL"]
    if failed:
        return f"GATE FAIL ({len(failed)} failed)", len(failed)
    return "GATE PASS", 0


def to_json_dict(checks: Sequence[Check]) -> dict[str, Any]:
    data: dict[str, Any] = {
        c.name: {"status": c.status, "detail": c.detail} for c in checks
    }
    _, failed = gate_summary(checks)
    data["gate"] = "FAIL" if failed else "PASS"
    return data


def is_clean(state: GitState) -> bool:
    """A reuse/write-safe tree: known diff, no untracked files, no uncommitted edits."""
    return not state.error and not state.untracked and not state.uncommitted


def tool_hash() -> str:
    """sha256 of this script's own source bytes: any gate change invalidates the cache."""
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def main_files_hash(main_root: Path) -> dict[str, str]:
    """sha256 per main-checkout input of the gate (exactness script, lockfiles)."""
    out: dict[str, str] = {}
    for rel in MAIN_KEY_FILES:
        try:
            out[rel] = hashlib.sha256((main_root / rel).read_bytes()).hexdigest()
        except OSError:
            out[rel] = MISSING_FILE_TOKEN
    return out


def cache_key(head: str, base_sha: str, cfg: Config, thash: str) -> str:
    payload = json.dumps(
        {
            "head": head,
            "base": base_sha,
            "scope": sorted(cfg.scope),
            "exactness": cfg.exactness,
            "frontend": cfg.frontend,
            "tool": thash,
            "main": main_files_hash(cfg.main_root),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def parse_check_line(line: str) -> Check | None:
    """Inverse of format_line (None when the line is not a well-formed check)."""
    status, sep, rest = line.partition(" ")
    name, sep2, detail = rest.partition(": ")
    if not sep or not sep2 or status not in CHECK_STATUSES or not name:
        return None
    return Check(name, status, detail)


def _git_out(wt: Path, run: RunFn, *args: str) -> str:
    rc, out = run(["git", "-C", str(wt), *args], wt, None, TOOL_TIMEOUT)
    return out.strip() if rc == 0 else ""


def git_dir(wt: Path, run: RunFn) -> Path | None:
    """This worktree's own git dir (per-worktree for linked worktrees; never committed)."""
    out = _git_out(wt, run, "rev-parse", "--absolute-git-dir")
    return Path(out) if out else None


def stamp_path(wt: Path, run: RunFn, key: str) -> Path | None:
    gd = git_dir(wt, run)
    return gd / CACHE_DIR / f"pass-{key[:16]}.json" if gd is not None else None


def head_reflog_count(wt: Path, run: RunFn) -> int | None:
    """HEAD reflog length, or None when git cannot report it (the caller fails closed)."""
    out = _git_out(wt, run, "rev-list", "--walk-reflogs", "--count", "HEAD")
    return int(out) if out.isdigit() else None


def _cache_probe(cfg: Config, run: RunFn) -> tuple[str, GitState | None]:
    """(cache key, git state): the key is "" when a stamp must not be reused or written."""
    try:
        state = git_state(cfg.worktree, cfg.base, run)
        if not is_clean(state):
            return "", state
        head = _git_out(cfg.worktree, run, "rev-parse", "HEAD")
        base_sha = _git_out(cfg.worktree, run, "rev-parse", cfg.base + "^{commit}")
        if not head or not base_sha:
            return "", state
        return cache_key(head, base_sha, cfg, tool_hash()), state
    except Exception:  # fail closed: any doubt -> run the full gate
        return "", None


def load_stamp(wt: Path, run: RunFn, key: str) -> Stamp | None:
    """A valid stored PASS for `key`, or None (any corruption/mismatch -> full gate)."""
    if not key:
        return None
    try:
        path = stamp_path(wt, run, key)
        if path is None or not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        if data.get("version") != CACHE_VERSION or data.get("key") != key:
            return None
        if data.get("gate") != "PASS":
            return None
        head, ts, lines = data.get("head"), data.get("ts"), data.get("checks")
        if not isinstance(head, str) or not isinstance(ts, str):
            return None
        if not isinstance(lines, list) or not lines:
            return None
        checks: list[Check] = []
        for line in lines:
            parsed = parse_check_line(line) if isinstance(line, str) else None
            if parsed is None:
                return None
            checks.append(parsed)
        if any(c.status not in STAMP_STATUSES for c in checks):
            return None  # only PASS/SKIP lines: a stored FAIL or WARN is never a PASS
        return Stamp(head=head, ts=ts, checks=checks)
    except Exception:
        return None


def write_stamp(wt: Path, run: RunFn, cfg: Config, key: str, checks: Sequence[Check]) -> None:
    if not key:
        return
    path = stamp_path(wt, run, key)
    if path is None:
        return
    data = {
        "version": CACHE_VERSION,
        "key": key,
        "head": _git_out(wt, run, "rev-parse", "HEAD"),
        "base": cfg.base,
        "base_sha": _git_out(wt, run, "rev-parse", cfg.base + "^{commit}"),
        "scope": sorted(cfg.scope),
        "exactness": cfg.exactness,
        "frontend": cfg.frontend,
        "tool": tool_hash(),
        "ts": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "gate": "PASS",
        "checks": [format_line(c) for c in checks],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-pass-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, indent=2) + "\n")
        os.replace(tmp_name, path)  # atomic: a reader sees the whole stamp or none
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def stubs_allowed() -> bool:
    return os.environ.get("SWARM_CHECK_ALLOW_STUBS") == "1"


def active_stubs() -> dict[str, str]:
    return {v: os.environ[v] for v in STUB_VARS if os.environ.get(v)}


def _cmd(env_var: str, default: list[str]) -> list[str]:
    override = os.environ.get(env_var) if stubs_allowed() else None
    return shlex.split(override) if override else default


def _tool_commands(cfg: Config) -> dict[str, list[str]]:
    venv = cfg.main_root / ".venv" / "bin"
    return {
        "ruff": _cmd("SWARM_CHECK_RUFF", [str(venv / "ruff"), "check"]),
        "mypy": _cmd("SWARM_CHECK_MYPY", [str(venv / "mypy")]),
        "pytest": _cmd("SWARM_CHECK_PYTEST", [str(venv / "python"), "-m", "pytest"]),
        "python": [str(venv / "python")],
    }


def _run_tests(
    cfg: Config, state: GitState, tools: dict[str, list[str]], run: RunFn, wt_env: dict[str, str]
) -> Check:
    python_changed = [p for p in state.changed if p.endswith(".py")]
    if not python_changed:
        return Check("tests", "SKIP", "no python changed")
    targets = [p for p in changed_test_files(state.changed) if (cfg.worktree / p).exists()]
    expr = k_expr(state.changed)
    notes: list[str] = []
    # (args, rc5_is_skip, rc5_note)
    commands: list[tuple[list[str], bool, str]] = []
    # A handful of targeted files runs serially (worker startup costs more than it
    # saves); timing-marked tests in them therefore run uncontended.
    if targets:
        commands.append((targets, False, ""))
    full = needs_full_suite(state.changed)
    # A changed vibe_quant module also runs every test file that imports it, directly or
    # through other vibe_quant modules (reverse import closure): a caller test whose names
    # never mention the module is invisible to `-k <stem>` (see importer_test_files).
    if not full and any(module_name(p) for p in state.changed):
        seen = set(targets)
        importers = [
            p
            for p in importer_test_files(state.changed, build_import_graph(cfg.worktree, state.changed))
            if p not in seen
        ]
        if len(importers) > MAX_IMPORTER_FILES:
            full = True  # a low-level module reaches most of tests/unit: one whole-suite run
            notes.append(
                f"{len(importers)} importer files > cap {MAX_IMPORTER_FILES}: full tests/unit"
            )
        elif importers:
            notes.append(f"{len(importers)} importer files")
            # ONE parallel call with the timing-marked tests deselected, then those run
            # serially, exactly like the -k group below.
            commands.append(
                ([*importers, *PYTEST_PARALLEL_ARGS, "-m", f"not {TIMING_MARK}"], True,
                 "no non-timing tests in the importer files")
            )
            commands.append(
                ([*importers, "-m", TIMING_MARK], True, "no timing tests in the importer files")
            )
    # A big selection (full suite, -k group) runs under xdist with the timing-marked
    # relative-speed tests deselected, then those run serially (they flake under load).
    group = ["tests/unit"] if full else (["tests/unit", "-k", expr] if expr else [])
    if group:
        # rc5 on a full-suite run is a real failure; on a -k group it means "no match".
        commands.append(
            (group + list(PYTEST_PARALLEL_ARGS) + ["-m", f"not {TIMING_MARK}"], not full,
             f"no tests matched -k {expr}" if expr else "")
        )
        commands.append(
            (group + ["-m", TIMING_MARK], True,
             f"no timing tests matched -k {expr}" if expr else "no timing tests matched")
        )
    if not commands:
        return Check("tests", "SKIP", "nothing to run for changed python")
    summaries: list[str] = []
    failed = False
    passed = 0
    for args, rc5_skip, rc5_note in commands:
        rc, out = run(tools["pytest"] + args, cfg.worktree, wt_env, TEST_TIMEOUT)
        if rc == PYTEST_NO_TESTS_RC and rc5_skip:
            summaries.append(rc5_note)
            continue
        summaries.append(pytest_summary(out))
        if rc != 0:
            failed = True
        else:
            passed += 1
    summaries.extend(notes)  # how many importer files ran, or why the full suite ran instead
    if failed:
        return Check("tests", "FAIL", " | ".join(summaries))
    if passed == 0:
        return Check("tests", "SKIP", " | ".join(summaries))
    return Check("tests", "PASS", " | ".join(summaries))


def _run_exactness(cfg: Config, tools: dict[str, list[str]], run: RunFn) -> Check:
    main_script = cfg.main_root / EXACTNESS_SCRIPT
    wt_script = cfg.worktree / EXACTNESS_SCRIPT
    if not main_script.is_file():
        return Check("exactness", "FAIL", f"main exactness script missing: {main_script}")
    if wt_script.is_file() and wt_script.read_bytes() != main_script.read_bytes():
        return Check("exactness", "FAIL", "exactness script modified in worktree")
    rc, out = run(
        tools["python"] + [str(main_script)],
        cfg.worktree,
        {"PYTHONPATH": str(cfg.worktree)},
        TEST_TIMEOUT,
    )
    if rc == 0 and "IDENTICAL" in out:
        return Check("exactness", "PASS", "IDENTICAL")
    return Check("exactness", "FAIL", f"drift (rc={rc}): " + tail_line(out))


def _run_exactness_carry(cfg: Config, tools: dict[str, list[str]], run: RunFn) -> Check:
    """The carry proof. Mirrors _run_exactness except that a missing MAIN script is a SKIP:
    scripts/agents/exactness_carry.py lands with the carry engine and a worktree cannot be
    expected to gate against a proof main does not have yet."""
    main_script = cfg.main_root / EXACTNESS_CARRY_SCRIPT
    wt_script = cfg.worktree / EXACTNESS_CARRY_SCRIPT
    if not main_script.is_file():
        return Check("exactness-carry", "SKIP", "main has no exactness_carry.py yet")
    if wt_script.is_file() and wt_script.read_bytes() != main_script.read_bytes():
        return Check("exactness-carry", "FAIL", "exactness_carry script modified in worktree")
    rc, out = run(
        tools["python"] + [str(main_script)],
        cfg.worktree,
        {"PYTHONPATH": str(cfg.worktree)},
        TEST_TIMEOUT,
    )
    if rc == 0 and "IDENTICAL" in out:
        return Check("exactness-carry", "PASS", "IDENTICAL")
    return Check("exactness-carry", "FAIL", f"drift (rc={rc}): " + tail_line(out))


def _lint_problem(label: str, rc: int, out: str) -> str:
    if rc != 0 and not has_lint_count(out):
        return f"{label} lint failed without an error count: " + tail_line(out)
    return ""


def _run_frontend(cfg: Config, run: RunFn) -> Check:
    wt_fe = cfg.worktree / "frontend"
    tsc_rc, tsc_out = run(["npx", "tsc", "-b"], wt_fe, None, TOOL_TIMEOUT)
    build_rc, build_out = run(["pnpm", "build"], wt_fe, None, TOOL_TIMEOUT)
    wt_rc, lint_wt = run(["pnpm", "lint"], wt_fe, None, TOOL_TIMEOUT)
    main_rc, lint_main = run(["pnpm", "lint"], cfg.main_root / "frontend", None, TOOL_TIMEOUT)
    wt_errs, main_errs = parse_lint_errors(lint_wt), parse_lint_errors(lint_main)
    problems = []
    if tsc_rc != 0:
        problems.append("tsc: " + tail_line(tsc_out))
    if build_rc != 0:
        problems.append("build: " + tail_line(build_out))
    for label, rc, out in (("worktree", wt_rc, lint_wt), ("main", main_rc, lint_main)):
        if msg := _lint_problem(label, rc, out):
            problems.append(msg)
    if wt_errs > 0:
        problems.append(f"lint errors {wt_errs} (must be 0; main {main_errs})")
    counts = f"lint errors {wt_errs} (main {main_errs})"
    if problems:
        return Check("frontend", "FAIL", "; ".join(problems) + "; " + counts)
    return Check("frontend", "PASS", "tsc ok, build ok, " + counts)


def run_checks(cfg: Config, run: RunFn = real_run) -> list[Check]:
    state = git_state(cfg.worktree, cfg.base, run)
    tools = _tool_commands(cfg)
    wt_env = {"PYTHONPATH": str(cfg.worktree)}
    checks: list[Check] = []

    # 0. stubs: refused unless explicitly allowed; always reported
    stubs = active_stubs()
    if stubs:
        listed = ", ".join(f"{k}={v}" for k, v in stubs.items())
        if stubs_allowed():
            checks.append(Check("stubs", "WARN", f"NOTE stub active: {listed}"))
        else:
            checks.append(
                Check("stubs", "FAIL", f"stubs refused (set SWARM_CHECK_ALLOW_STUBS=1): {listed}")
            )

    # 1. scope
    if state.error:
        checks.append(Check("scope", "FAIL", state.error))
    elif not state.changed:
        checks.append(Check("scope", "FAIL", f"empty diff vs {cfg.base} (nothing to review)"))
    elif cfg.scope:
        offenders = scope_offenders(state.changed, cfg.scope)
        if offenders:
            shown = ", ".join(offenders[:5])
            more = f" (+{len(offenders) - 5} more)" if len(offenders) > 5 else ""
            detail = f"{len(offenders)}/{len(state.changed)} paths out of scope: {shown}{more}"
            checks.append(Check("scope", "FAIL", detail))
        else:
            checks.append(Check("scope", "PASS", f"{len(state.changed)} paths in scope"))
    else:
        checks.append(Check("scope", "SKIP", "no --scope given"))

    # 2. clean
    if state.untracked:
        shown = ", ".join(state.untracked[:5])
        more = f" (+{len(state.untracked) - 5} more)" if len(state.untracked) > 5 else ""
        checks.append(Check("clean", "WARN", f"{len(state.untracked)} untracked: {shown}{more}"))
    else:
        checks.append(Check("clean", "PASS", "no untracked files"))

    # 3. ruff / 4. mypy
    for name in ("ruff", "mypy"):
        rc, out = run(tools[name], cfg.worktree, None, TOOL_TIMEOUT)
        checks.append(Check(name, "PASS" if rc == 0 else "FAIL", tail_line(out)))

    unknown = "git diff unknown (see scope)"
    # 5. tests
    if state.error:
        checks.append(Check("tests", "SKIP", unknown))
    else:
        checks.append(_run_tests(cfg, state, tools, run, wt_env))

    # 6. exactness
    if cfg.exactness == "never":
        checks.append(Check("exactness", "SKIP", "disabled"))
    elif cfg.exactness == "always" or (not state.error and should_run_exactness(state.changed)):
        checks.append(_run_exactness(cfg, tools, run))
    elif state.error:
        checks.append(Check("exactness", "SKIP", unknown))
    else:
        checks.append(
            Check(
                "exactness",
                "SKIP",
                "auto: nothing changed under vibe_quant/ (except "
                + ", ".join(d.rstrip("/").removeprefix("vibe_quant/") for d in EXACTNESS_SAFE_DIRS)
                + ") nor in " + ", ".join(EXACTNESS_FILES),
            )
        )

    # 6b. exactness-carry: the same flag governs both proofs, and a main without the
    # script is a SKIP so this check can land before the carry engine does
    if cfg.exactness == "never":
        checks.append(Check("exactness-carry", "SKIP", "disabled"))
    elif cfg.exactness == "always" or (
        not state.error and should_run_exactness_carry(state.changed)
    ):
        checks.append(_run_exactness_carry(cfg, tools, run))
    elif state.error:
        checks.append(Check("exactness-carry", "SKIP", unknown))
    else:
        checks.append(
            Check(
                "exactness-carry",
                "SKIP",
                "auto: nothing changed in " + ", ".join(EXACTNESS_CARRY_TRIGGERS),
            )
        )

    # 7. frontend
    if cfg.frontend == "never":
        checks.append(Check("frontend", "SKIP", "disabled"))
    elif cfg.frontend == "always" or (not state.error and should_run_frontend(state.changed)):
        checks.append(_run_frontend(cfg, run))
    elif state.error:
        checks.append(Check("frontend", "SKIP", unknown))
    else:
        checks.append(Check("frontend", "SKIP", "auto: nothing under frontend/ changed"))

    # bus-inbox LAST of the evaluations: a chief DM that lands while the tools run must still
    # block the hand-back (time-dependent, never cached)
    checks.append(bus_inbox_check(cfg))

    order = {name: i for i, name in enumerate(CHECK_ORDER)}
    checks.sort(key=lambda c: order.get(c.name, len(order)))
    return checks


def _derive_main_root(wt: Path) -> Path:
    proc = subprocess.run(
        ["git", "-C", str(wt), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return Path("/Users/verebelyin/projects/vibe-quant")
    return Path(proc.stdout.strip()).parent


def _bus_module() -> Any:
    """The bus.py next to this gate: the same file the worktree's workers use."""
    path = Path(__file__).resolve().parent / "bus.py"
    spec = importlib.util.spec_from_file_location("swarm_gate_bus", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bus_inbox_check(cfg: Config) -> Check:
    """FAIL when the agent still has an unread message from the chief; never advances cursors.

    A worker must not hand back while a chief DM is unread, so this runs on EVERY invocation,
    including a cached PASS: it is time-dependent (a DM can arrive at any moment), so it is
    NOT part of the PASS-stamp cache key. It peeks (never consumes) the agent's inbox.
    """
    if not cfg.bus_agent and not cfg.bus:
        return Check("bus-inbox", "SKIP", "no --bus-agent/SWARM_AGENT or --bus/SWARM_BUS")
    if not cfg.bus_agent:
        return Check("bus-inbox", "SKIP", f"bus {cfg.bus} given but no --bus-agent/SWARM_AGENT to peek")
    if not cfg.bus:
        return Check("bus-inbox", "FAIL", f"agent {cfg.bus_agent} given but no --bus/SWARM_BUS to peek")
    if not Path(cfg.bus).is_dir():
        return Check("bus-inbox", "FAIL", f"configured bus {cfg.bus} is not a directory")
    try:
        msgs = _bus_module().inbox(Path(cfg.bus), cfg.bus_agent, peek=True)
    except (Exception, SystemExit) as exc:  # fail closed (bus.py exits on a garbled cursor)
        reason = str(exc) if isinstance(exc, SystemExit) else f"{type(exc).__name__}: {exc}"
        return Check("bus-inbox", "FAIL", f"cannot read bus {cfg.bus}: {reason}")
    unread = [m for m in msgs if m.get("from") == "chief" and m.get("to") in (cfg.bus_agent, "all")]
    if not unread:
        return Check("bus-inbox", "PASS", f"no unread message from chief to {cfg.bus_agent}")
    listed = "; ".join(f"{m['id']} {str(m['body'])}" for m in unread)
    return Check("bus-inbox", "FAIL", f"{len(unread)} unread from chief: {listed} — run: bus inbox")


def bus_post(cfg: Config, checks: Sequence[Check], cached_head: str = "") -> None:
    bus = cfg.bus  # the resolved --bus / $SWARM_BUS, so a flag-only invocation reports too
    if not bus:
        return
    if not Path(bus).is_dir():  # bus.py would create it, turning a typo into a configured bus
        print(f"swarm-check: bus {bus} does not exist; not posting the gate result", file=sys.stderr)
        return
    gate, _ = gate_summary(checks)
    failed = [c.name for c in checks if c.status == "FAIL"]
    body = f"{cfg.worktree.name}: {gate}"
    if cached_head:
        body += f" (cached {cached_head[:12]})"
    if failed:
        body += " (" + ", ".join(failed) + ")"
    with contextlib.suppress(Exception):
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "bus.py"),
                "--bus", bus, "--as", "swarm-check",
                "post", "--to", "chief", "--kind", "info", "--body", body,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )


def main(argv: Sequence[str] | None = None, run: RunFn | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-review gate for a worker's worktree")
    parser.add_argument("worktree")
    parser.add_argument("--base", default="main")
    parser.add_argument("--scope", default="")
    parser.add_argument("--exactness", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--frontend", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--json", dest="json_out", default=None)
    parser.add_argument("--no-cache", dest="no_cache", action="store_true")
    parser.add_argument("--bus", default=None, help="job bus dir for the bus-inbox check (else $SWARM_BUS)")
    parser.add_argument("--bus-agent", dest="bus_agent", default=None, help="whose inbox to peek (else $SWARM_AGENT)")
    args = parser.parse_args(argv)

    wt = Path(args.worktree).resolve()
    if not (wt / ".git").exists() and subprocess.run(
        ["git", "-C", str(wt), "rev-parse", "--git-dir"], capture_output=True
    ).returncode != 0:
        print(f"not a git worktree: {wt}")
        return 2

    cfg = Config(
        worktree=wt,
        main_root=_derive_main_root(wt),
        base=args.base,
        scope=tuple(g for g in (s.strip() for s in args.scope.split(",")) if g),
        exactness=args.exactness,
        frontend=args.frontend,
        json_out=Path(args.json_out) if args.json_out else None,
        bus=args.bus or os.environ.get("SWARM_BUS") or "",
        bus_agent=args.bus_agent or os.environ.get("SWARM_AGENT") or "",
    )

    runner: RunFn = run or real_run

    # B1: refuse the hand-back BEFORE spending a suite run (or even reading the cache) when the
    # chief's DM is already waiting. A DM that lands mid-run is still caught by the end-of-run
    # bus-inbox check in run_checks, so this only short-circuits the already-unread case.
    pre_bus = bus_inbox_check(cfg)
    if pre_bus.status == "FAIL":
        print(format_line(pre_bus))
        print(gate_summary([pre_bus])[0])
        bus_post(cfg, [pre_bus])
        return 1

    key, _ = _cache_probe(cfg, runner) if not args.no_cache else ("", None)  # PRE-run key
    stamp = load_stamp(wt, runner, key)

    # bus-inbox is time-dependent, so it is re-evaluated even when the tool checks are cached:
    # a cached PASS with an unread chief DM is still a GATE FAIL (run_checks adds it otherwise).
    if stamp is not None:
        for check in stamp.checks:
            print(f"(cached) {format_line(check)}")
        print(f"PASS cache: reused stamp from {stamp.ts} for {stamp.head[:12]}")
        bus_check = bus_inbox_check(cfg)
        checks = [*stamp.checks, bus_check]
        print(format_line(bus_check))
        gate, failed = gate_summary(checks)
        print(gate)
        if cfg.json_out:
            cached = to_json_dict(checks)
            cached["cached"] = True
            with open(cfg.json_out, "w", encoding="utf-8") as fh:
                json.dump(cached, fh, indent=2)
                fh.write("\n")
        bus_post(cfg, checks, cached_head=stamp.head)
        return 1 if failed else 0

    # A commit+reset --hard made while the tools run puts HEAD and the tree back where they
    # started, so the pre/post key compare cannot see it; the reflog length can (see N1 doc).
    reflog_before = head_reflog_count(wt, runner) if key else None

    try:
        checks = run_checks(cfg) if run is None else run_checks(cfg, runner)
    except Exception as exc:  # fail closed: a crashed gate is a failed gate
        checks = [Check("internal", "FAIL", f"gate crashed: {type(exc).__name__}: {exc}")]
    for check in checks:
        print(format_line(check))
    gate, failed = gate_summary(checks)
    print(gate)

    if cfg.json_out:
        with open(cfg.json_out, "w", encoding="utf-8") as fh:
            json.dump(to_json_dict(checks), fh, indent=2)
            fh.write("\n")

    # B3: a bus-inbox-only FAIL is time-dependent (the DM may simply be read next run), so the
    # tool PASS is still stamped; only a real tool FAIL blocks the stamp.
    tool_failed = [c for c in checks if c.status == "FAIL" and c.name != "bus-inbox"]
    if not args.no_cache and not tool_failed and not active_stubs() and key:
        # key is the PRE-run key (non-empty = clean start). Stamp only if nothing moved while
        # the tools ran; otherwise the verdict is for a HEAD/tree that was never gated.
        post_key, _ = _cache_probe(cfg, runner)
        reflog_after = head_reflog_count(wt, runner)
        if post_key == key and reflog_before is not None and reflog_after == reflog_before:
            # bus-inbox is time-dependent and not part of the key: never store it as a cached PASS/SKIP
            stamped = [c for c in checks if c.name != "bus-inbox"]
            with contextlib.suppress(Exception):
                write_stamp(wt, runner, cfg, key, stamped)

    bus_post(cfg, checks)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
