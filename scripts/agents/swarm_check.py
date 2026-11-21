#!/usr/bin/env python3
"""Pre-review gate for a worker's worktree — run by the orchestrator before any review.

    swarm-check.sh <worktree> [--base REF] [--scope glob,glob,...]
                   [--exactness auto|always|never] [--frontend auto|always|never]
                   [--json out.json] [--no-cache] [--bus DIR] [--bus-agent NAME]
                   [--full] [--list-tests]

Checks (each prints "PASS|FAIL|SKIP <name>: <detail>"), then "GATE PASS" /
"GATE FAIL (n failed)" as the last line; exit 0/1. Default base: main;
default exactness/frontend: auto (run only when relevant paths changed).
A NEW check is registered in WARN_MODE_UNTIL with a date: until then its FAIL prints as
"WARN <name>: warn-mode until <date>: <detail>" and does not fail the gate, so its first job
surfaces the backlog without a gate rerun.

Two modes (bead q2ijs). PER ROUND (default, a maker's gate): ONE parallel pytest call over
the changed test files + POLICY_TESTS + the reverse import closure of the changed vibe_quant
modules (no cap) + the tests that reach a changed file by path or name (path-mentions,
script consumers) + the files a `-k <stems>` can match (filtered by -k, so the explicit files
still run whole), `timing` tests deselected; then a serial `-m timing` call over the changed
test files only.
INTEGRATION (`--full`, the verifier before land): the whole tests/unit in parallel, then
every `timing` test serially. A conftest/__init__ .py, pyproject.toml, uv.lock, a
`__main__.py` outside vibe_quant/, a non-Python file under vibe_quant/ or tests/ (readers not
modelled) or any file the selection cannot read or parse makes a per-round run full too:
when the selection is uncertain it GROWS, never shrinks (review rev-gate-speed-1). The import
graph spans vibe_quant/ AND tests/ (helpers, test files importing test files), adds production
string references ("vibe_quant.x", "-m pkg") as edges, and seeds DIRECTORY_LOADERS for a
changed plugin; a non-Python file elsewhere selects the tests naming it or its top-level dir.
ruff, mypy, exactness, exactness-carry, exactness-tiers, gate-golden and frontend run
on threads while pytest runs. exactness-tiers and gate-golden are the GOLDEN PROOFS nothing ran until this check
existed (98cc4b86 declared a summation-order change and main drifted on 2 PF last digits):
they compare the WORKTREE's `exactness_tiers.py --check` / `gate_golden.py --against` against
goldens kept OUT of git in the main checkout's `data/swarm/_goldens/`, and they run only in
`--full` (or `--exactness always`); per round they SKIP `per-round: runs in --full`, and
`--exactness never` SKIPs them as `disabled`. A missing golden is a FAIL (fail closed). A
drift FAIL re-runs the SAME script from the MAIN checkout and appends whether main is
identical (this diff caused it) or already drifted (pre-existing on main).
`--list-tests` prints the selection (one test file per line, the calls on stderr) and runs
no tool. The LOG records each tool's wall time.

bus-inbox: with --bus (or $SWARM_BUS) and --bus-agent (or $SWARM_AGENT) both known the gate
reads that agent's task thread on the board: the root in `job/<job>` labelled
`agent=<bus-agent>`; unread = chief messages after the agent's newest message there (the chief's
root counts, so a worker posts at least once; a peer's note is ignored), listed as id + full text
ending "run: vb read <thread>, then reply"; FAIL while any is unread. It is a WARN instead when
EVERY unread chief message is older than the worktree's last commit by the board's `createdAt`
(the maker already acted on it); a bad timestamp or a git error stays a FAIL. It is evaluated
BEFORE any tool runs (a message already waiting refuses the hand-back without paying for the
suite) AND again after every tool check, right before the verdict, so one that lands mid-run
still blocks. SKIP only when neither is configured (or only a bus, no agent); an agent without a
bus, a bus dir that does not exist, no thread, no message by the agent, or a board that cannot be
read (named fix: scripts/agents/board-up.sh) is a FAIL, never a SKIP. The file store
(`SWARM_STORE=jsonl`) has no worker inbox (job 20261120-board R27): with an agent it is a FAIL
"the file store has no worker inbox: unset SWARM_STORE". In both stores a
`done` message by the agent whose `<job>/handoffs/<agent>.md` is missing or fails handoff_lint is a
FAIL ("write and lint ..."): a raw `vb reply '[done] ...'` skips bus.py's own refusal.
The gate posts nothing to the bus (its posts were DMs nobody read): the verdict is the LOG's
last lines, which handoff_lint checks. Time-dependent, so it runs even when the tool
checks are served from a cached PASS and is never written into the stamp.

PASS-stamp cache: a GATE PASS on a clean worktree writes
`<git-dir>/swarm-check/pass-<key[:16]>.json` in that worktree's git dir (per-worktree,
never committed, invisible to `git status`). The key is a sha256 over HEAD^{tree} (not the
commit: a message-only amend reuses the stamp), the base commit, the merge-base, the sorted
scope globs, exactness/frontend/full modes, this script's own bytes and the
sha256 of the MAIN checkout's scripts/agents/exactness_239.py,
scripts/agents/exactness_carry.py, uv.lock and frontend/pnpm-lock.yaml (a missing file
hashes as a fixed token), so any change to the
diff or to the checks invalidates it. A later run whose tree is clean and whose key
matches prints the stored check lines as `(cached) ...`, then
`PASS cache: reused stamp from <ts> for <head[:12]>` and `GATE PASS`, without running any tool.
The stamp is written only when the key computed BEFORE the run is non-empty (clean start),
AND equals the key recomputed after it (nothing moved while the tools ran: a commit made
mid-run would otherwise stamp a HEAD that was never gated), AND the HEAD reflog length is
unchanged (`git rev-list --walk-reflogs --count HEAD`). The reflog guard catches a
commit + `reset --hard` made mid-run: it leaves the key identical but adds reflog entries.
Writes are atomic (temp file + os.replace). A tool FAIL, a WARN, a dirty tree or a stub
blocks the stamp, but a bus-inbox-only FAIL does not: the stored checks exclude the bus line
(time-dependent), so the next run after the agent's thread reply reuses the tool PASS instead of re-running
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

Base selection: an explicit --base is used verbatim. With NO --base off branch main, a
stacked worktree (its `.vq-base` start SHA not on main, main not merged since) gates vs that
SHA (INFO line); otherwise the gate keeps `main`, except when the worktree's current branch is `main` and `origin/main` exists: it then gates
`origin/main` and prints `INFO base: on main, using origin/main (pass --base to override)`
(`--base main` is never rewritten). An empty diff against that auto-selected base FAILs with
`empty diff vs origin/main: nothing unpushed to gate; pass --base <sha> to gate pushed commits`
instead of the generic "empty diff" (failing closed, but with the fix in the message).
Paths under docs/orchestration/board/ and docs/orchestration/telemetry/ are the bus's own
mirror files (messages.jsonl, export.jsonl, tasks.jsonl): they are never scope offenders and
never make a diff look non-empty.

Gate log + run counter: EVERY run writes the full transcript for each tool it ran (command
line, combined stdout+stderr, exit code) to
`<main checkout>/data/swarm/_gate-logs/<worktree dir name>-<HEAD[:12]>-<UTC %Y%m%dT%H%M%SZ>.log`
(the dir is created; the main checkout root is `_derive_main_root`), prints `LOG <abs path>`
immediately before the final GATE line, and appends ` (full output: LOG above)` to every FAIL
line of a tool check. A scope FAIL skips every tool, so its log holds just the scope line,
which names EVERY changed path outside --scope (no first-five cut). A cached run writes a
one-line log (no tool ran). Every log ends with `HEAD <sha> TREE <tree>` (TREE is `dirty` when
tracked files had uncommitted edits, `moved` when HEAD/tree/edits changed while the tools ran:
no commit holds what was gated) and the GATE line;
handoff_lint.py matches a maker's cited commit against them. A run that finds an
existing log for the same worktree name + HEAD also prints
`INFO gate run #<n+1> on this HEAD - ...` first (advisory only, never fails): the gate is
meant to run once per round, so read the previous LOG instead of re-running.

gate-runs cap: that rule is now enforced for makers. With --bus and a --bus-agent starting
`cc-` or `impl-` (not `--full`), an UNCACHED run counts itself in the job's
`<bus dir>/../gate-runs.jsonl` (ts, agent, worktree, log) and a maker round gets 3 runs
(`$SWARM_GATE_CAP` overrides). The 4th prints `FAIL gate-runs: <agent> ran the gate <n> times
(cap <cap>); read the last LOG <log>, fix, then ask the chief in your thread ([question]) for
another run`, then the GATE FAIL line, and exits 1 before any tool: no LOG is written and the ledger is
not appended. A cached PASS is never refused nor counted; `ver-*`, reviewers, `--full` and a
run with no bus stay outside it.

stale-reads: a T3 child starts its shell in the MAIN checkout, and 59 of 342 children in
Oct 14-28 ran commands that read or tested main's STALE code instead of their worktree
(scripts/agents/stale_reads.py). For a maker (--bus-agent starting `cc-`/`impl-`, not
--full) the gate reads the T3 history DB (`$SWARM_T3_DB`, default
`~/.t3/userdata/statev2.sqlite`) read-only and FAILs while any of that agent's
command_execution rows read a repo-relative code path (scripts/, vibe_quant/, tests/,
frontend/src) with no `cd <worktree>` / absolute worktree path, or run pytest/python/mypy/
ruff with `PYTHONPATH=$PWD` from elsewhere (a plain sed/cat/rg/head read of only
scripts/agents/ files is exempt while the branch leaves scripts/agents/ unchanged); the
DETAIL names the first command (`re-read from <worktree>`). A missing DB is a SKIP with its path. Time-dependent like bus-inbox (the
log grows mid-round): re-evaluated on a cached PASS, never stored in the stamp, and a
stale-reads-only FAIL still writes the tool PASS stamp.

pinned-text: when the diff touches vibe_quant/**/*.py the gate collects the string literals
(>= 12 chars after strip) that existed in the BASE version of each changed file and no longer
appear anywhere in the worktree's vibe_quant/ (plain substring search over all .py files),
then FAILs for each tests/**/test_*.py that still pins one of them UNLESS the gate already
runs that file (it is a changed test file, an importer file, or the whole tests/unit runs).
A literal the changed file still holds by ast (implicitly concatenated f-string pieces merge
across lines, so no source text has it) is not removed.
That catches a pin OUTSIDE the targeted set that stays green here and goes red in the full
suite later.

policy-tests: POLICY_TESTS are repo-wide policy tests that SCAN the tree or the docs
(silent-exception ratchet, steering word budgets, other tree scanners) instead of importing
the module under change, so neither `-k` nor the reverse import closure can select them. Each
runs when the diff touches the tree it scans (POLICY_TRIGGERS, read off the test and pinned
by test_swarm_check_policy_triggers.py), its own file or this script - including a diff with
NO .py file, where a docs edit can turn the word-budget test red. They lead the one parallel
call. A full tests/unit run covers them all. Keeps a gate PASS from leaving main red. The
serial `-m timing` call runs only over changed test files that carry a timing mark.

path-mentions: a test that READS a changed source file by path instead of importing it is
invisible to both `-k <stem>` and the reverse import closure. tests/unit/
test_state_manager_locking.py reads vibe_quant/screening/replay_drift.py and
vibe_quant/validation/consistency.py through `read_text()`, so a change to either module left
the gate green (bead olizd, found by rev-si-gate-global-1). For every changed vibe_quant
module the selection also scans tests/unit for the repo-relative path or the module FILE name
(e.g. "replay_drift.py"); a bare stem is deliberately NOT a needle (`pipeline`, `runner` would
select half of tests/unit, and `-k` already matches the test files named after them). Those
files join the importer selection. A generic file name (`__main__.py`, `__init__.py`, one
without a suffix) is matched only as "<parent dir>/<name>". A changed scripts/** file selects
the tests naming it or any script that names it (script_consumer_test_files).

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
import functools
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
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
# The two golden proofs no gate ran before (the 98cc4b86 drift): exactness_tiers reproduces 8
# real-catalog cells and gate_golden every champion gate stage. Both goldens are kept OUT of git
# (user decision) in the MAIN checkout's data/swarm/_goldens/ (data/swarm/ is gitignored), and
# both scripts resolve a bare default there at call time; the gate passes the path explicitly.
EXACTNESS_TIERS_SCRIPT = "scripts/agents/exactness_tiers.py"
GATE_GOLDEN_SCRIPT = "scripts/agents/gate_golden.py"
GOLDEN_SUBDIR = ("data", "swarm", "_goldens")
GOLDEN_TIERS_NAME = "tiers.json"
GOLDEN_GATE_NAME = "gate_golden.json"
# Why the golden proofs are skipped ("" when they run): --full or --exactness always. A
# per-round gate must not pay ~90 s of real-catalog cells for every fix round.
GOLDEN_SKIP_PER_ROUND = "per-round: runs in --full"
CHECK_ORDER = (
    "scope", "clean", "bus-inbox", "stale-reads", "ruff", "mypy", "tests", "slow-tests", "simple-tests",
    "pinned-text",
    "exactness", "exactness-carry", "exactness-tiers", "gate-golden", "frontend",
    "stubs", "internal",
)
# Warn mode for a NEW check: until the ISO date (exclusive) its FAIL is reported as a WARN, so
# the first job that runs it sees the backlog without failing gates (retro 2026-11-05 #5).
# Add the check here with a date about one job ahead; remove the entry once it has passed.
# History: slow-tests enforced from its first job (20261105-prune2) and first fired at
# integration: 7 FAILs, 3 gate reruns (retro 2026-11-06).
WARN_MODE_UNTIL: dict[str, str] = {}
# Checks whose verdict can change without the diff changing: re-evaluated on a cached PASS,
# never written into the PASS stamp and never blocking the stamp on their own.
TIME_DEPENDENT_CHECKS = frozenset({"bus-inbox", "stale-reads"})
UNTRACKED_OK = ("cmd-report.md",)
# The bus writes these mirror files into the worktree on every job; they are never the
# worker's change, so they must not fail the scope check nor make an empty diff look real.
BOARD_PATH_PREFIXES = (
    "docs/orchestration/board/",
    "docs/orchestration/telemetry/",
)
# The per-run tool transcript (see the module docstring): <main checkout>/data/swarm/_gate-logs/.
GATE_LOG_SUBDIR = ("data", "swarm", "_gate-logs")
# The gate runs once per round (docs/claude/agent-speed.md), now enforced: 3 UNCACHED runs per
# maker round, counted in the job's own ledger (`<bus dir>/../gate-runs.jsonl`). A cached PASS
# costs nothing and is never refused nor counted; verifiers, reviewers, --full and a run with
# no bus stay outside it. Agents whose id starts with one of these prefixes are makers.
GATE_RUNS_FILE = "gate-runs.jsonl"
GATE_RUNS_CAP = 3
CAPPED_AGENT_PREFIXES = ("cc-", "impl-")
# stale-reads: the T3 history DB of the child's own commands (`$SWARM_T3_DB` overrides the
# local T3 state) and how much of the first stale command the FAIL quotes.
T3_DB_ENV = "SWARM_T3_DB"
T3_DB_DEFAULT = Path.home() / ".t3" / "userdata" / "statev2.sqlite"
STALE_READS_SHOWN_CHARS = 120
PINNED_TEXT_MIN_CHARS = 12  # a literal shorter than this is too generic to be a pinned string
PINNED_TEXT_SHOWN_CHARS = 60  # how much of the removed literal a pinned-text FAIL quotes
# Checks that shell out to a real tool: their FAIL lines carry the "(full output: LOG above)"
# pointer. scope/clean/bus-inbox/stubs/internal are git/config facts with no tool transcript.
TOOL_CHECK_NAMES = frozenset(
    {"ruff", "mypy", "tests", "slow-tests", "simple-tests", "exactness", "exactness-carry", "exactness-tiers",
     "gate-golden", "frontend"}
)
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
# Through PYTEST_ADDOPTS so the argv stays as is: -q cancels the ini's -v (a full-suite log was
# 1.7 MB, 8105 PASSED lines, which reviewers then grep), -rfE keeps one FAILED/ERROR line per test.
PYTEST_QUIET_ADDOPTS = "-q -rfE"
FAILED_TEST_RE = re.compile(r"^(?:FAILED|ERROR) (\S+)", re.MULTILINE)
TIMING_MARK = "timing"
# The 2 s rule (job 20261105-prune2 lane G, user decision 2026-11-05): a NEW or CHANGED unit
# test that takes longer than this ALONE fails the gate unless it carries
# `@pytest.mark.slow(reason=...)`. tests/unit grew +395 tests in a week after a prune (the
# serial suite is 754 s against a ~120 s goal), so a slow test is a decision the diff states.
SLOW_TEST_SECONDS = 2.0
SLOW_TEST_MARK = "slow"
SLOW_TEST_HINT = '@pytest.mark.slow(reason="...")'
# The simple-target rule (CLAUDE.md § Tests prove outcomes, user decision 2026-11-20): a NEW
# test whose run touches only functions with cognitive complexity <= SIMPLE_MAX_COGNITIVE
# (cccc) that are not kill-zone/money code and hold no regex/SQL FAILs the gate unless it
# carries `@pytest.mark.simple_target(reason=...)`. The exemptions err wide: a FAIL here
# deletes a test.
SIMPLE_MAX_COGNITIVE = 5
SIMPLE_TEST_MARK = "simple_target"
SIMPLE_TEST_HINT = '@pytest.mark.simple_target(reason="...")'
CCCC_SCRIPT = "scripts/agents/cccc.sh"
SIMPLE_EXEMPT_PATH = re.compile(
    r"overfitting/|metrics|fitness|sizing|liquidat|fill|fees|funding|validation/|screening/|"
    r"vibe_quant/dsl/|vibe_quant/data/|balance|indicator|portfolio/|paper/|ethereal/|risk/|"
    r"vibe_quant/db/|bankroll|state_manager|archive|lookahead|exactness|golden"
)
SIMPLE_EXEMPT_NAME = re.compile(
    r"sharpe|sortino|calmar|drawdown|pnl|fee|funding|liquidat|margin|siz(e|ing)|fill|slippage|"
    r"leverage|order|profit|return|equity|dsr|bootstrap|kfold|wfa|lookahead|warmup|shift",
    re.I,
)
SIMPLE_EXEMPT_SQL = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE FROM|CREATE TABLE|ALTER TABLE|PRAGMA)\b")
SIMPLE_EXEMPT_REGEX = re.compile(
    r"\bre\.(compile|match|search|sub|fullmatch|findall|finditer|split)\b|\b[A-Z_]+_RE\b"
)
# The junit xml each pytest call writes: it rides in that call's PYTEST_ADDOPTS, so the argv
# the plan built (and --list-tests prints) stays exactly what the tests pin.
JUNIT_OPT = "--junitxml="
# The re-run that separates a real slow test from worker contention: one test, serially. At
# most SLOW_RERUN_CAP per gate (slowest first), each capped at SLOW_RERUN_TIMEOUT seconds.
SLOW_ALONE_ARGS = ("-p", "no:randomly", "-n0")
SLOW_RERUN_CAP = 5
SLOW_RERUN_TIMEOUT = 60


def xdist_auto_workers(env: dict[str, str], load1: float | None, cpus: int, running: int = 1) -> str | None:
    """What ``-n auto`` should mean: SWARM_CHECK_WORKERS wins; else the smaller of (a) the
    free cores, but at least half of them, when other jobs already load the host past half
    its cores, and (b) an even split of the cores among the ``running`` gates (this one
    included; gate_slots.xdist_share); else None (xdist decides). Applied by ``_run_tests``
    through xdist's PYTEST_XDIST_AUTO_NUM_WORKERS, so the gate's commands and test selection
    never depend on the host. Job 20261019-budget-dsr: a ``-n auto`` gate on top of two
    discoveries hit load 29.5 on 10 cores; job 20261021-rabmo: a floor of 2 workers made 4
    concurrent gates crawl (8-10 min each); job 20261105-prune2: 5 gates at ``-n auto``
    measured the same suite at 147-231 s purely from contention (load sizing lags, so gates
    that start together all saw a quiet host). ``load1`` None (no load reading: sample_load)
    drops cap (a), keeping the slot share."""
    if env.get("SWARM_CHECK_WORKERS"):
        return env["SWARM_CHECK_WORKERS"]
    caps: list[int] = []
    if load1 is not None and load1 > cpus / 2:
        caps.append(max(cpus // 2, 2, cpus - round(load1)))
    share = _gate_slots_module().xdist_share(cpus, running)
    if share is not None:
        caps.append(share)
    return str(min(caps)) if caps else None


def _gate_slots_module() -> Any:
    """The gate_slots.py next to this gate (stdlib only)."""
    cached = sys.modules.get("swarm_gate_slots")
    if cached is not None:
        return cached
    path = Path(__file__).resolve().parent / "gate_slots.py"
    spec = importlib.util.spec_from_file_location("swarm_gate_slots", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["swarm_gate_slots"] = mod
    spec.loader.exec_module(mod)
    return mod





# No cap on the importer-file selection (the reverse import closure of the changed modules):
# a hub module (metrics.py, errors.py, dsl/compiler.py) reaches 200-250 of ~380 test files,
# and that closure still costs less than the whole suite, which also pays for the heaviest
# indicator-port files. The full suite runs once per integration (`--full`, the verifier)
# and whenever full_suite_reason says the selection cannot be trusted (a conftest/__init__,
# pyproject.toml or uv.lock change). Bead q2ijs: the old 150-file cap sent every
# discovery/__main__.py round to the full suite (7-9 min under 4 parallel makers).
# Diff paths that force the whole tests/unit even per round: they change what every test
# imports or how pytest/collection behaves, so no per-module selection can represent them.
FULL_SUITE_FILES = ("pyproject.toml", "uv.lock")
FULL_SUITE_STEMS = frozenset({"__init__", "conftest"})
# A file name this generic is no path-mention needle on its own (every package has one): it
# is matched as "<parent dir>/<name>" instead (test_swarm_check.py mentions "__main__.py" as
# fixture text and was selected for every __main__ change).
GENERIC_FILE_NAMES = frozenset({"__init__.py", "__main__.py", "conftest.py"})
# Repo-wide policy tests: each SCANS the tree (or the docs) instead of importing the module
# it checks, so the reverse import closure can never select one and `-k` never names it. A
# change anywhere in the scanned tree can turn one red while a targeted gate run stays green -
# 2026-10-18: a new silent handler in vibe_quant/discovery/neff.py went red in
# test_exception_hygiene.py although every gate PASSed. Every non-full selection appends the
# ones present in the worktree (a full tests/unit run already covers them). Add a file only
# when its verdict depends on the TREE, not on the changed module: a test that imports what
# it checks is already in the importer closure.
POLICY_TESTS = (
    "tests/unit/test_exception_hygiene.py",  # silent-swallow ratchet over the money packages
    "tests/unit/test_steering_budget.py",  # word budgets over the steering docs and agents
    "tests/unit/test_api_routers_conn_locking.py",  # tree scan: routers/discovery conn use
    "tests/unit/test_discovery_liquidation.py",  # tree scan: every NTBacktestFn passes liquidation=
    "tests/unit/test_script_exec_bits.py",  # git index scan: shebang scripts are 100755
    "tests/unit/test_swarm_check_loaders.py",  # tree scan: every dynamic import is modelled
    "tests/unit/test_scripts_import_smoke.py",  # tree scan: every script importing vibe_quant loads
)
# What each policy test SCANS (fnmatch globs; `*` crosses `/`), read off its source - pinned
# by tests/unit/test_swarm_check_policy_triggers.py. A policy test runs per round only when
# the diff touches one of its globs, its own file or this script (POLICY_ALWAYS); its other
# tests import what they check, so the import closure selects them. --full runs all.
POLICY_TRIGGERS: dict[str, tuple[str, ...]] = {
    # SCAN_PACKAGES: rglob("*.py") under each money package; BASELINE is in the test file
    "tests/unit/test_exception_hygiene.py": tuple(
        f"vibe_quant/{pkg}/*.py"
        for pkg in (
            "screening", "validation", "discovery", "overfitting", "dsl", "risk", "paper",
            "data", "portfolio",
        )
    ),
    # _steering_files() + BUDGETS (the out-of-repo MEMORY.md profile cannot be in a diff)
    "tests/unit/test_steering_budget.py": (
        "CLAUDE.md",
        ".claude/agents/*.md",
        ".claude/skills/*/SKILL.md",
        "docs/orchestration/*.md",
        "docs/claude/*.md",
    ),
    # rglob scripts/**/*.py; a vibe_quant rename that breaks a script is left to --full
    # (a vibe_quant/* trigger would add it to every round)
    "tests/unit/test_scripts_import_smoke.py": ("scripts/*.py",),
    # _SCANNED = api/routers and discovery
    "tests/unit/test_api_routers_conn_locking.py": (
        "vibe_quant/api/routers/*.py",
        "vibe_quant/discovery/*.py",
    ),
    # test_every_construction_passes_liquidation: rglob("*.py") over all of vibe_quant
    "tests/unit/test_discovery_liquidation.py": ("vibe_quant/*.py",),
    # `git ls-files -s scripts`: content AND mode of every tracked file under scripts/
    "tests/unit/test_script_exec_bits.py": ("scripts/*",),
    # loader_call_sites: rglob("*.py") over all of vibe_quant
    "tests/unit/test_swarm_check_loaders.py": ("vibe_quant/*.py",),
}
POLICY_ALWAYS = ("scripts/agents/swarm_check.py",)
CACHE_DIR = "swarm-check"
CACHE_VERSION = 2
CHECK_STATUSES = ("PASS", "FAIL", "SKIP", "WARN")
STAMP_STATUSES = ("PASS", "SKIP")  # the only line statuses a reusable stamp may contain
MAIN_KEY_FILES = (
    "scripts/agents/exactness_239.py",
    EXACTNESS_CARRY_SCRIPT,
    "uv.lock",
    "frontend/pnpm-lock.yaml",
    # The golden proofs compare against these, so a recapture on main invalidates a stamp too.
    "data/swarm/_goldens/tiers.json",
    "data/swarm/_goldens/gate_golden.json",
)
MISSING_FILE_TOKEN = "missing"
# The per-test-file PASS cache (bead sfghw, chief ruling in thread 7760): one entry per test
# file holds the tree it last PASSed at; shared by every worktree (one object DB).
FILE_CACHE_SUBDIR = ("data", "swarm", "_gate-cache", "files")
FILE_CACHE_VERSION = 1
# Main-checkout inputs of every test run: the venv's interpreter and its locked deps (plus the
# installed *.dist-info names and the gate python's sys.version: file_cache_env).
FILE_CACHE_ENV_FILES = ("uv.lock", ".venv/pyvenv.cfg")
# A test whose source names an external data input (ignored catalog/archive/state files, the
# state DB) can turn red with no tree change: its PASS is never stored (review rev-gate-1 B1).
EXTERNAL_DATA_RE = re.compile(r"data/|catalog|archive|VIBE_QUANT_DB|state/")
POLICY_STOP = "policy tests failed: selection skipped"
TREE_SHA_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")


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
    # overrides the scope FAIL detail when the diff is empty; set only for the auto-selected
    # origin/main base, where "empty diff" means "nothing unpushed to gate".
    empty_diff_detail: str = ""
    full: bool = False  # --full: integration mode, the whole tests/unit + every timing test
    # --expect-pins / SWARM_EXPECT_PINS: a declared comparability break's new exactness pins
    # (exactness_239.py --expect); empty = the script's own constants.
    expect_pins: str = ""
    # Shared dir of running gates' pid-named slots (gate_slots.py); None = no sharing (tests,
    # injected runners). The test step splits the cores among the slots live when it starts.
    slots_dir: Path | None = None
    # Shared dir of per-test-file PASS entries (FILE_CACHE_SUBDIR); None = off (tests, injected
    # runners, --no-cache, --full).
    file_cache_dir: Path | None = None


@dataclass(frozen=True)
class TestRun:
    """The tests check result plus the pytest file set it ran, for the pinned-text check.

    `explicit` is every test file the gate passed to pytest by path; `full_unit` says the
    whole tests/unit ran (a conftest/__init__ change or more importer files than the cap).
    `slow` is the 2 s rule's own check line (None only when the diff itself is unknown).
    """

    check: Check
    explicit: frozenset[str] = frozenset()
    full_unit: bool = False
    slow: Check | None = None
    stopped: bool = False  # a red fail-fast (policy) call ended the step


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


class ToolLog:
    """RunFn wrapper that records every tool call for the gate log (see the module docstring).

    `git` calls are delegated but NOT recorded: the transcript is the tool output a FAIL would
    otherwise lose (ruff/mypy/pytest/exactness/frontend), not the gate's own plumbing. Entries
    are (command line, exit code, combined stdout+stderr + a trailing wall-time line), so the
    LOG shows where a slow gate spent its time. Tool checks run on worker threads
    (run_checks), hence the lock.
    """

    def __init__(self, run: RunFn) -> None:
        self._run = run
        self._lock = threading.Lock()
        self.entries: list[tuple[str, int, str] | str] = []

    def __call__(
        self,
        cmd: Sequence[str],
        cwd: Path,
        env: Mapping[str, str] | None = None,
        timeout: int = TOOL_TIMEOUT,
    ) -> RunResult:
        start = time.monotonic()
        rc, out = self._run(cmd, cwd, env, timeout)
        if str(cmd[0]) != "git":
            took = f"[wall {time.monotonic() - start:.1f}s]"
            with self._lock:
                self.entries.append((" ".join(str(c) for c in cmd), rc, f"{out.rstrip()}\n{took}"))
        return rc, out

    def note(self, line: str) -> None:
        """A context line the LOG states verbatim, in order with the tool entries."""
        with self._lock:
            self.entries.append(line)


def is_board_path(path: str) -> bool:
    """True for the bus's own mirror files (board/telemetry), which are never the worker's diff."""
    return path.startswith(BOARD_PATH_PREFIXES)


def reviewable_paths(paths: Sequence[str]) -> list[str]:
    return [p for p in paths if not is_board_path(p)]


def resolve_base(wt: Path, base: str | None, run: RunFn) -> tuple[str, str]:
    """(base ref, INFO line). With no explicit --base, gate origin/main when on main.

    An explicit --base (including `--base main`) is returned verbatim so today's behaviour is
    untouched. Only the no-flag, on-branch-`main`, origin/main-exists case is rewritten; the
    INFO line is printed by main() and is "" when nothing changed.
    """
    if base is not None:
        return base, ""
    branch = _git_out(wt, run, "symbolic-ref", "--short", "HEAD")
    if branch != "main":
        return stacked_base(wt, run)
    if not _git_out(wt, run, "rev-parse", "--verify", "--quiet", "origin/main^{commit}"):
        return "main", ""
    return "origin/main", "INFO base: on main, using origin/main (pass --base to override)"


def stacked_base(wt: Path, run: RunFn) -> tuple[str, str]:
    """A stacked worktree (worktree.sh recorded a start SHA in <wt>/.vq-base that is not on
    main) gates its own commits: base = that SHA. Anything else keeps main: no or unreadable
    .vq-base, a start SHA on main, or a branch that merged main after its start (main's
    merge-base with HEAD is no longer inside the start SHA's history)."""
    try:
        sha = (wt / ".vq-base").read_text(encoding="utf-8").strip()
    except OSError:
        return "main", ""
    if not sha or not _git_out(wt, run, "rev-parse", "--verify", "--quiet", sha + "^{commit}"):
        return "main", ""
    if _git_rc(wt, run, "merge-base", "--is-ancestor", sha, "main") == 0:
        return "main", ""
    joint = _git_out(wt, run, "merge-base", "main", "HEAD")
    if not joint:
        return "main", ""
    if _git_rc(wt, run, "merge-base", "--is-ancestor", joint, sha) != 0:
        return "main", "INFO base: stacked worktree merged main after its start; gating vs main"
    return sha, f"INFO base: stacked worktree, using its start {sha[:12]} from .vq-base"


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


class GitFacts:
    """One PHASE's memo of the git questions whose answer cannot change while a tree is read.

    A run asks `rev-parse <base>^{commit}` and `merge-base <base> HEAD` for the diff
    (`git_state`), for the PASS-stamp key (`_cache_probe`) and for the stamp it records
    (`write_stamp`): four spawns for two facts. Each answer is memoised here instead, so the
    pre-run phase and the post-run phase ask each question ONCE. The post-run freshness check
    gets a NEW memo on purpose: that is what sees a commit or a moved base ref made mid-run.
    """

    def __init__(self) -> None:
        self._answers: dict[tuple[str, tuple[str, ...]], tuple[int, str]] = {}

    def call(self, wt: Path, run: RunFn, *args: str) -> tuple[int, str]:
        key = (str(wt), args)
        if key not in self._answers:
            self._answers[key] = run(["git", "-C", str(wt), *args], wt, None, TOOL_TIMEOUT)
        return self._answers[key]

    def out(self, wt: Path, run: RunFn, *args: str) -> str:
        rc, out = self.call(wt, run, *args)
        return out.strip() if rc == 0 else ""


def git_state(wt: Path, base: str, run: RunFn, facts: GitFacts | None = None) -> GitState:
    """Committed + uncommitted + untracked changes. Fails closed: any git error -> .error."""
    facts = GitFacts() if facts is None else facts

    rc, out = facts.call(wt, run, "rev-parse", "--verify", "--quiet", base + "^{commit}")
    if rc != 0:
        return GitState([], [], f"bad base ref: {base}")
    rc, out = facts.call(wt, run, "merge-base", base, "HEAD")
    if rc != 0:
        return GitState([], [], f"no merge-base between {base} and HEAD: {tail_line(out)}")
    rc, diff_out = facts.call(wt, run, "diff", "--name-only", "--no-renames", "-z", base + "...HEAD")
    if rc != 0:
        return GitState([], [], f"git diff failed: {tail_line(diff_out)}")
    rc, status_out = facts.call(wt, run, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if rc != 0:
        return GitState([], [], f"git status failed: {tail_line(status_out)}")
    committed = [p for p in diff_out.split("\0") if p]
    changed, untracked, uncommitted = parse_porcelain(status_out)
    seen: dict[str, None] = {}
    for path in committed + changed:
        seen.setdefault(path, None)
    return GitState(changed=list(seen), untracked=untracked, uncommitted=uncommitted)


def parse_scope(raw: str) -> tuple[str, ...]:
    """Comma-separated SCOPE globs; a leading `+` (a brief's new-file marker) is dropped."""
    return tuple(g for g in (s.strip().lstrip("+") for s in raw.split(",")) if g)


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


def full_suite_reason(paths: Sequence[str]) -> str:
    """Why the per-round selection must be the whole tests/unit ("" when it need not be).

    A conftest/__init__ change, a pyproject.toml/uv.lock change (pytest config, deps) or a
    `__main__.py` OUTSIDE vibe_quant/ (no module name, so no import closure; `-k` drops the
    stem) cannot be represented by a selection: fail closed. A vibe_quant `__main__.py` is a
    module like any other and seeds the reverse import closure.
    """
    for p in paths:
        name = Path(p).name
        if p in FULL_SUITE_FILES:
            return f"{p} changed"
        if p.endswith(".py") and Path(p).stem in FULL_SUITE_STEMS:
            return f"{p} changed"
        if name == "__main__.py" and not module_name(p):
            return f"{p} changed (no module name: no import closure)"
        if not p.endswith(".py") and not is_modelled_resource(p) and not is_fixture_resource(p):
            # a resource (json/yaml/sql/...) is read by path or importlib.resources by modules
            # the graph cannot see (review rev-gate-speed-1 F3 binance_um_brackets.json,
            # rev-gate-speed-2 F2 a root settings.json read by vibe_quant/reader.py)
            return f"{p} changed (non-Python resource: readers not modelled)"
    return ""


FIXTURE_ROOT = "tests/fixtures/"
# A string literal that names the fixtures ROOT, not an entry in it (`HERE / "fixtures"`,
# "tests/fixtures", "tests/fixtures/" + x, f"tests/fixtures/{x}", "tests/fixtures/*"): such
# a file may walk every fixture dir without naming one. A literal that continues with a name
# ("tests/fixtures/x.json") names that entry, which the stem check sees (chief ruling R3b:
# catalog.py's comment naming one fixture file made every fixture diff select 385 files).
# Accepted residual risk (review rev-gate-1 B5, ruling R3): a root built from parts no
# literal spells (`"fix" + "tures"`) is invisible; --full runs that reader before land.
FIXTURE_WALK_RE = re.compile(r"""["'/]fixtures(?:["']|/(?=["'*?{\[]))""")


def is_fixture_resource(path: str) -> bool:
    """A non-Python file under tests/fixtures/: fixture_reader_files models its readers."""
    return path.startswith(FIXTURE_ROOT) and not path.endswith(".py")


def fixture_reader_files(changed: Sequence[str], worktree: Path) -> list[str]:
    """Every file that may read a changed tests/fixtures resource: the .py files under tests/
    and vibe_quant/ and every file under scripts/ that spell the stem of its top entry under
    tests/fixtures/ (`board_legacy` for tests/fixtures/board_legacy/x/y.jsonl,
    `fee_drag_vectors` for tests/fixtures/fee_drag_vectors.json), plus the tests/ and scripts/
    files that name the fixtures root itself (FIXTURE_WALK_RE: a glob over every dir). Job
    20261120-board: a new fixture dir ran the whole suite (245 s) for one reader test. The
    caller seeds the selection with the readers (a helper's importers, a script's consumers)
    and fails closed on a conftest/__init__ reader. A non-Python file under scripts/ or
    vibe_quant/ that mentions `fixtures` (a shell/yaml reader: `cat tests/fixtures/g*/a.json`)
    is not modelled: FullSuite. A read error raises (plan_tests -> full).
    """
    stems = {
        Path(p.removeprefix(FIXTURE_ROOT).split("/")[0]).stem
        for p in changed
        if is_fixture_resource(p)
    }
    if not stems:
        return []
    out: list[str] = []
    for top in ("tests", "scripts", "vibe_quant"):
        root = worktree / top
        for path in sorted(root.rglob("*")) if root.is_dir() else ():
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if top == "tests" and path.suffix != ".py":
                continue
            text = read_cached(path, "replace")  # OSError: plan_tests -> full
            rel = path.relative_to(worktree).as_posix()
            if path.suffix != ".py":
                if "fixtures" in text:
                    raise FullSuite(f"{rel} names fixtures (a non-Python reader: not modelled)")
                if top == "vibe_quant":
                    continue  # no consumer walk for a vibe_quant resource: its stem is noise
            if FIXTURE_WALK_RE.search(text) or any(stem in text for stem in stems):
                out.append(rel)
    return out


# Non-Python paths the per-round selection models; any other one runs the full suite.
#   frontend/**      the frontend check (tsc, build, lint) covers it; python tests read only
#                    frontend `__fixtures__/` (rg 2026-10-22), so those stay full
#   .beads/**        the issue tracker's own data, read by no test
#   data/swarm/**    job ledgers and gate logs (gitignored in the main checkout)
#   scripts/**       the script-consumer walk (scripts load each other by file name)
#   *.md outside vibe_quant/   md_reader_test_files + POLICY_TESTS
INERT_PREFIXES = (".beads/", "data/swarm/", "frontend/")


def is_modelled_resource(path: str) -> bool:
    if is_board_path(path) or path.startswith("scripts/"):
        return True
    if path.endswith(".md") and not path.startswith("vibe_quant/"):
        return True
    return path.startswith(INERT_PREFIXES) and "/__fixtures__/" not in path


def needs_full_suite(paths: Sequence[str]) -> bool:
    return bool(full_suite_reason(paths))


def base_literals(source: str) -> set[str]:
    """String literals of `source` worth pinning: len >= PINNED_TEXT_MIN_CHARS after stripping.

    ast walk, so f-string literal parts and nested literals count; a file that does not parse
    contributes nothing (ruff/mypy report it anyway) and never raises.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):  # ValueError: NUL bytes / a decode that slipped through
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            stripped = node.value.strip()
            if len(stripped) >= PINNED_TEXT_MIN_CHARS:
                out.add(stripped)
    return out


def _git_show(wt: Path, base: str, path: str, run: RunFn) -> str:
    """The BASE version of `path` ("" when it did not exist at base or git failed)."""
    rc, out = run(["git", "-C", str(wt), "show", f"{base}:{path}"], wt, None, TOOL_TIMEOUT)
    return out if rc == 0 else ""


def worktree_vibe_quant_text(wt: Path) -> str:
    """Every vibe_quant/**/*.py of the worktree, concatenated for substring searches."""
    root = wt / "vibe_quant"
    if not root.is_dir():
        return ""
    chunks: list[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(chunks)


def removed_base_literals(cfg: Config, changed_py: Sequence[str], run: RunFn) -> list[str]:
    """Sorted base literals that survived in no worktree vibe_quant/ .py (plain substring)."""
    literals: set[str] = set()
    for path in changed_py:
        literals |= base_literals(_git_show(cfg.worktree, cfg.base, path, run))
    if not literals:
        return []
    wt_text = worktree_vibe_quant_text(cfg.worktree)
    # ast merges implicitly concatenated pieces across lines (`"a, " f"b={x}"` holds the
    # constant "a, b="), which no source text contains: compare against the same files' ast
    # literals too, or an unchanged f-string reads as removed (a discovery/__main__.py diff
    # that only added a comment FAILed on ", candidates=").
    for path in changed_py:
        try:
            literals -= base_literals((cfg.worktree / path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return sorted(lit for lit in literals if lit not in wt_text)


def pinned_test_pins(wt: Path, literals: Sequence[str]) -> list[tuple[str, int, str]]:
    """(test file, line, literal) for every tests/**/test_*.py that still contains a literal.

    Only test files are scanned (a conftest/module pin is not what this check is for). The
    line is the first occurrence, so the FAIL points at the pinning assertion.
    """
    if not literals:
        return []
    root = wt / "tests"
    if not root.is_dir():
        return []
    out: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("test_*.py")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lit in literals:
            idx = text.find(lit)
            if idx >= 0:
                out.append((path.relative_to(wt).as_posix(), text.count("\n", 0, idx) + 1, lit))
    return out


def test_already_run(path: str, test_run: TestRun) -> bool:
    """True if the gate's own pytest selection already covers `path` (so it will go red here)."""
    if path in test_run.explicit:
        return True
    return test_run.full_unit and path.startswith("tests/unit/")


def pinned_text_check(cfg: Config, state: GitState, run: RunFn, test_run: TestRun) -> Check:
    """FAIL when a test the gate does NOT run pins a string the diff removed from production."""
    if state.error:
        return Check("pinned-text", "SKIP", "git diff unknown (see scope)")
    changed_py = [
        p for p in state.changed if p.startswith("vibe_quant/") and p.endswith(".py")
    ]
    if not changed_py:
        return Check("pinned-text", "SKIP", "no vibe_quant/**/*.py changed")
    removed = removed_base_literals(cfg, changed_py, run)
    offenders = [
        f'{path}:{line} pins removed text "{lit[:PINNED_TEXT_SHOWN_CHARS]}"'
        for path, line, lit in pinned_test_pins(cfg.worktree, removed)
        if not test_already_run(path, test_run)
    ]
    if offenders:
        return Check("pinned-text", "FAIL", "; ".join(offenders))
    return Check("pinned-text", "PASS", f"{len(removed)} removed literals, no unrun test pins them")


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


def graph_module_name(path: str) -> str:
    """The module a repo-relative .py under vibe_quant/ OR tests/ defines ("" otherwise).

    tests/ modules are graph nodes too: a helper (tests/unit/pta_stream_harness.py) or a test
    file another test imports (test_paper_watch_inherit.py imports test_audit_paper_node.py)
    reaches its importers only through this walk (review rev-gate-speed-1 F2).
    """
    if path.endswith(".py") and path.startswith(("vibe_quant/", "tests/")):
        return _dotted(path)
    return ""


def _is_local(name: str) -> bool:
    return name.split(".")[0] in ("vibe_quant", "tests")


class GraphError(Exception):
    """A file the import graph needs could not be read or parsed: the selection is unknown."""


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


# A module named in a string: "vibe_quant.x" or a "vibe_quant.x.attr" patch target.
QUOTED_MODULE_RE = re.compile(r"""["'](vibe_quant(?:\.\w+)*)["'.]""")
# A package run as a program: "-m", "vibe_quant.discovery" (its __main__ is what runs). A bare
# quoted package name is no needle: `root / "vibe_quant"` is in a dozen tree-scanning tests.
RUN_MODULE_RE = re.compile(r"""["']-m["']\s*,\s*["'](vibe_quant(?:\.\w+)*)["']""")


def string_module_refs(source: str, known: AbstractSet[str]) -> set[str]:
    """Known vibe_quant modules a PRODUCTION file names in a string: the longest known prefix
    of a quoted dotted name (`importlib.import_module("vibe_quant.x")`, a computed loader
    name) and the `__main__` of a `"-m", "<pkg>"` subprocess (api routers and
    discovery/campaign.py launch `-m vibe_quant.discovery` / `vibe_quant.validation`)."""
    out: set[str] = set()
    for match in QUOTED_MODULE_RE.finditer(source):
        parts = match.group(1).split(".")
        for i in range(len(parts), 1, -1):
            if (name := ".".join(parts[:i])) in known:
                out.add(name)
                break
    for match in RUN_MODULE_RE.finditer(source):
        if (main := match.group(1) + ".__main__") in known:
            out.add(main)
    return out


@functools.lru_cache(maxsize=8)
def ast_cached(source: str) -> ast.Module:
    """`ast.parse(source)`, one parse per text: the graph walk, the pytest_plugins fold,
    `_conftest_parts` and `_requests` all analyse file bytes another scan saw too (a cold plan
    made 1235 parses, 1.48 s of 4.09 s; job 20261029-gaps-speed, handoffs/cc-toolcache-1.md).

    Keyed by the TEXT, not the path: content is the only thing the analyses read, so a file
    edited between two plans in one process simply has different bytes and gets a fresh tree
    (no re-key bookkeeping to get wrong), and an `ast.unparse` fragment from `_conftest_parts`
    shares whatever text it came from. Callers treat the tree as READ-ONLY: `parse_imports`
    skips TYPE_CHECKING bodies in its walk instead of emptying them in place, which would hide
    a `pytest_plugins` registration from the fold that reads the same tree. A parse that fails
    is NOT cached (`lru_cache` drops the raise): callers report it as they did.

    DELIBERATELY SMALL. A tree is ~650 KB of objects and the tree-wide scan touches ~700
    files: holding them all measured 509 MB RSS and a 2x cold plan (5 M live objects make
    every cyclic GC pass and the interpreter's exit scan walk them - 3.7 s -> 8.0 s). MEASURED
    in this window: 8 entries 2.63 s, 32 entries 2.83 s, all 700 8.0 s. A miss is one parse -
    exactly the parse the caller made before this cache existed - so the window only has to
    cover the repeats that are close together: the fold re-reading the bytes `parse_imports`
    just parsed, `_conftest_parts`' `ast.unparse` fragments right after their conftest, and
    `_requests` on the test files a plan asks about."""
    return ast.parse(source)


def _runtime_nodes(tree: ast.AST) -> Iterator[ast.AST]:
    """`ast.walk(tree)` minus the bodies of `if TYPE_CHECKING:` blocks, at any depth.

    Same nodes the pre-cache code analysed by emptying `node.body` in place - which is not an
    option on a tree another scan reads after us (R3). `orelse` still counts: the block runs."""
    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        yield node
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            stack.append(node.test)
            stack.extend(node.orelse)
        else:
            stack.extend(ast.iter_child_nodes(node))


_STATEMENT_LISTS = ("body", "orelse", "finalbody", "handlers", "cases")


def _import_nodes(tree: ast.AST) -> Iterator[ast.Import | ast.ImportFrom]:
    """The `import` / `from ... import` nodes of ``tree``: the ones `ast.walk` finds, without
    visiting an expression.

    An import is a statement and no expression holds one, so only the statement lists need a
    visit: a body, an `else`, a `finally`, a `try`'s handlers, a `match`'s cases. `ast.walk`
    of ~770 files touched 1.6 M nodes (over half of a cold import-graph build, MEASURED
    1.2 s of 2.3 s); statements are about a tenth of them. A list of those fields on an
    expression (`IfExp.body`, `Lambda.body`) is never reached: only statements are pushed."""
    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
            continue
        for name in _STATEMENT_LISTS:
            children = getattr(node, name, None)
            if isinstance(children, list):
                stack.extend(children)


def parse_imports(
    source: str,
    module: str,
    known: AbstractSet[str],
    *,
    is_package: bool = False,
    bare: Mapping[str, str] | None = None,
    runtime_only: bool = False,
) -> set[str] | None:
    """The vibe_quant/tests modules `source` imports, or None when it does not parse.

    ast.walk sees imports in ANY scope, so a function-level (lazy) `from ... import f` -
    common in this repo - counts, as does one under `if TYPE_CHECKING:`. Recognised:
    `import a.b`, `import a.b as x`, `from a.b import y`, `from a import b` (b a submodule
    recorded in `known`) and the relative forms resolved against this file's package.
    NO implicit parent edge: `import vibe_quant.a.c` imports vibe_quant.a.c ONLY; adding
    vibe_quant.a too would select every test of that package for any submodule change.
    A `from vibe_quant.a import X` whose X is no module keeps its base (the package
    __init__). `bare` maps a top-level name to a tests module for the sys.path-insert form
    (`from pta_stream_harness import ...` in tests/unit/test_plugins/). Anything else outside
    vibe_quant/tests is dropped. `runtime_only` skips `if TYPE_CHECKING:` bodies (they never
    execute), used for conftest files whose reach selects a whole subtree.
    """
    try:
        tree = ast_cached(source)
    except (SyntaxError, ValueError):  # ValueError: NUL bytes / a decode that slipped through
        return None
    bare = bare or {}
    package = module if is_package else module.rpartition(".")[0]
    found: set[str] = set()

    def local(name: str) -> str | None:
        if _is_local(name):
            return name
        return bare.get(name)

    for node in _runtime_nodes(tree) if runtime_only else _import_nodes(tree):
        if isinstance(node, ast.Import):
            found.update(n for alias in node.names if (n := local(alias.name)))
        elif isinstance(node, ast.ImportFrom):
            raw = _from_base(node, package)
            base = local(raw) if raw is not None else None
            if base is None:
                continue
            found.add(base)
            for alias in node.names:
                candidate = f"{base}.{alias.name}"
                if alias.name != "*" and candidate in known:
                    found.add(candidate)
    return found


_TREE_CACHE: dict[tuple[object, ...], Any] = {}


def tree_fingerprint(worktree: Path, *subdirs: str) -> tuple[tuple[str, int, int], ...]:
    """(path, mtime_ns, size) of every .py under the worktree's `subdirs`: the cache key of
    the tree-wide scans below. Stat only, so a plan that re-reads nothing stays milliseconds;
    any edit, add or delete changes it (the cache never serves a stale scan).

    `os.scandir` rather than `os.walk` + `os.stat`: the directory entry carries the stat
    fields, so one syscall per file instead of two (this walk is on every plan's path)."""
    out: list[tuple[str, int, int]] = []
    pending = [
        str(worktree / sub) for sub in subdirs if os.path.isdir(str(worktree / sub))
    ]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    if entry.name != "__pycache__":
                        pending.append(entry.path)
                elif entry.name.endswith(".py"):
                    st = entry.stat()
                    out.append((entry.path, st.st_mtime_ns, st.st_size))
    out.sort()
    return tuple(out)


def plan_snapshot(worktree: Path) -> tuple[tuple[str, int, int], ...]:
    """The ONE tree walk a plan pays for: every tree-wide scan below keys its cache on it.

    A missing subdir must not raise (`os.scandir` on it would): the scans treat an absent
    vibe_quant/ or tests/ as empty and the plan then selects nothing, exactly as before."""
    return tree_fingerprint(
        worktree, *(sub for sub in ("vibe_quant", "tests") if (worktree / sub).is_dir())
    )


def _snapshot_or_walk(
    worktree: Path, snapshot: tuple[tuple[str, int, int], ...] | None
) -> tuple[tuple[str, int, int], ...]:
    return plan_snapshot(worktree) if snapshot is None else snapshot


def read_cached(path: Path, errors: str = "strict") -> str:
    """`path.read_text(encoding="utf-8", errors=...)`, cached per process by (path, mtime, size):
    the tree-wide scans re-read the same ~400 test files for every plan otherwise."""
    st = path.stat()
    key = ("text", path, errors, st.st_mtime_ns, st.st_size)
    if key not in _TREE_CACHE:
        _TREE_CACHE[key] = path.read_text(encoding="utf-8", errors=errors)
    return str(_TREE_CACHE[key])


def import_graph_pair(
    worktree: Path,
    extra_modules: Sequence[str] = (),
    snapshot: tuple[tuple[str, int, int], ...] | None = None,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """(graph, reverse imports) for a PLAN: both READ-ONLY, never copied and never mutated.

    `build_import_graph` hands every caller its own copy (a mutation must not reach the
    cache) - 705 sets and a dict per call. A plan only reads, and that copy was most of the
    garbage a plan produced: with it in place the 241-plan cap test spent 34.5 ms per plan,
    of which GC pauses were 22 (same loop, gc.disable() -> 12.6 ms)."""
    gone = tuple(sorted(
        p for p in extra_modules if graph_module_name(p) and not (worktree / p).exists()
    ))
    walked = _snapshot_or_walk(worktree, snapshot)
    key = ("graph", worktree.resolve(), walked, gone)
    if key not in _TREE_CACHE:
        graph = _build_import_graph(worktree, extra_modules, walked)
        importers: dict[str, set[str]] = {}
        for path, imports in graph.items():
            if mod := graph_module_name(path):
                for target in imports:
                    importers.setdefault(target, set()).add(mod)
        _TREE_CACHE[key] = (graph, importers)
    pair: tuple[dict[str, set[str]], dict[str, set[str]]] = _TREE_CACHE[key]
    return pair


def build_import_graph(
    worktree: Path,
    extra_modules: Sequence[str] = (),
    snapshot: tuple[tuple[str, int, int], ...] | None = None,
) -> dict[str, set[str]]:
    """`import_graph_pair`'s graph, COPIED: callers may mutate what they get."""
    graph, _importers = import_graph_pair(worktree, extra_modules, snapshot)
    return {rel: set(mods) for rel, mods in graph.items()}


def _build_import_graph(
    worktree: Path,
    extra_modules: Sequence[str] = (),
    snapshot: tuple[tuple[str, int, int], ...] | None = None,
) -> dict[str, set[str]]:
    """repo-relative path -> the vibe_quant/tests modules that file imports.

    Nodes: every `vibe_quant/**/*.py` and every `tests/**/*.py` (helpers, conftests and test
    files) of the WORKTREE. Production files also get string_module_refs edges. A file that
    cannot be read or parsed RAISES GraphError: a missing edge would silently shrink the
    selection, so plan_tests runs the full suite instead (review rev-gate-speed-1 F4).
    `extra_modules` (the changed files) join the known names: a DELETED module has no file on
    disk, yet `from vibe_quant.pkg import gone` must still resolve to the submodule edge.

    The `pytest_plugins` scan is FOLDED IN: this walk visits every tests/**/*.py anyway, so it
    records the offenders (`check_pytest_plugins` raises from that, after this call, as
    before) instead of `_check_pytest_plugins` re-reading and re-parsing all 446 test files
    (~1.3 s of a cold plan, job 20261029-gaps-speed).
    """
    roots = (worktree / "vibe_quant", worktree / "tests")
    rels = sorted(
        p.relative_to(worktree).as_posix()
        for root in roots
        if root.is_dir()
        for p in root.rglob("*.py")
        if "__pycache__" not in p.parts
    )
    known = {_dotted(rel) for rel in rels}
    known.update(_dotted(p) for p in extra_modules if graph_module_name(p))
    stems: dict[str, list[str]] = {}
    for rel in rels:
        if rel.startswith("tests/") and not rel.endswith("__init__.py"):
            stems.setdefault(Path(rel).stem, []).append(_dotted(rel))
    bare = {stem: mods[0] for stem, mods in stems.items() if len(mods) == 1}
    graph: dict[str, set[str]] = {}
    plugins: list[str] = []
    for rel in rels:
        try:
            source = read_cached(worktree / rel)
        except (OSError, ValueError) as exc:
            raise GraphError(f"cannot read {rel}: {exc}") from exc
        imports = parse_imports(
            source, _dotted(rel), known, is_package=rel.endswith("__init__.py"), bare=bare,
            runtime_only=rel.endswith("/conftest.py"),
        )
        if imports is None:
            raise GraphError(f"cannot parse {rel}")
        # the folded pytest_plugins scan: same bytes, same tree (a cache hit just parsed here).
        # Python NFKC-normalises identifiers, so fullwidth `ｐｙｔｅｓｔ_plugins` IS the name:
        # a non-ASCII file is searched in its NFKC form (review rev-parse-1 F1). A hit in a
        # comment or string only costs a walk; the walk decides.
        maybe = "pytest_plugins" in source or (
            not source.isascii() and "pytest_plugins" in unicodedata.normalize("NFKC", source)
        )
        if rel.startswith("tests/") and maybe and _names_pytest_plugins(ast_cached(source)):
            plugins.append(rel)
        if rel.startswith("vibe_quant/"):
            imports |= string_module_refs(source, known)
        graph[rel] = imports
    if snapshot is not None:
        # the same entry `check_pytest_plugins` reads: the plan asks it right after this call
        # and must raise from there, so a FullSuite lands where it landed before (R4).
        _TREE_CACHE[_plugins_cache_key(worktree, snapshot)] = _plugins_reason(plugins)
    return graph


# Production modules loaded by DIRECTORY SCAN, not by name: dsl/plugin_loader.py walks
# dsl/plugins/ with pkgutil.iter_modules + importlib, dsl/compiler.py hashes plugins/*.py
# and dsl/indicators.py triggers the load. Any change under the prefix (edit, add, delete,
# rename) seeds these modules' reverse closure (review rev-gate-speed-1 F1: a rams.py diff
# omitted tests/unit/test_indicator_compute.py). Re-check with
# `rg 'importlib|import_module|iter_modules|spec_from_file|glob' vibe_quant` when adding one.
DIRECTORY_LOADERS = {
    "vibe_quant/dsl/plugins/": (
        "vibe_quant/dsl/plugin_loader.py",
        "vibe_quant/dsl/indicators.py",
        "vibe_quant/dsl/compiler.py",
    ),
    # register_source() registry: load_sources() walks the package with iter_modules and
    # import_module/reload (review rev-gate-speed-2 F3)
    "vibe_quant/research/sources/": ("vibe_quant/research/sources/__init__.py",),
}
# Dynamic-import call sites that load NO repo module by scan or computed name. Every other
# vibe_quant file with such a call must be a DIRECTORY_LOADERS loader, or the gate runs the
# full suite (unmodelled_loaders); tests/unit/test_swarm_check_loaders.py keeps this honest.
LOADER_EXEMPTIONS = {
    "vibe_quant/dsl/indicators.py": "_get_nt_class imports third-party NautilusTrader classes",
    "vibe_quant/dsl/compiler.py": "spec_from_loader builds an in-memory module from generated source",
    "vibe_quant/utils.py": "__import__('datetime') is a stdlib import written inline",
}
LOADER_CALL_RE = re.compile(
    r"\b(?:iter_modules|import_module|__import__|spec_from_file_location|spec_from_loader)"
    r"\s*\(|\bimportlib\.reload\s*\("
)


def loader_call_sites(
    worktree: Path, snapshot: tuple[tuple[str, int, int], ...] | None = None
) -> list[str]:
    """Per-process cached `_loader_call_sites` (keyed by the plan's tree snapshot)."""
    key = ("loaders", worktree.resolve(), _snapshot_or_walk(worktree, snapshot))
    if key not in _TREE_CACHE:
        _TREE_CACHE[key] = _loader_call_sites(worktree)
    return list(_TREE_CACHE[key])


def _loader_call_sites(worktree: Path) -> list[str]:
    """Sorted vibe_quant/**/*.py files with a dynamic-import CALL (strings do not count)."""
    root = worktree / "vibe_quant"
    if not root.is_dir():
        return []
    out: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        code = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
        if LOADER_CALL_RE.search(re.sub(r"(['\"])(?:(?!\1).)*\1", '""', code)):
            out.append(path.relative_to(worktree).as_posix())
    return out


def unmodelled_loaders(
    worktree: Path, snapshot: tuple[tuple[str, int, int], ...] | None = None
) -> list[str]:
    """loader_call_sites that are neither a DIRECTORY_LOADERS loader nor exempted."""
    covered = {loader for loaders in DIRECTORY_LOADERS.values() for loader in loaders}
    covered |= set(LOADER_EXEMPTIONS)
    return [p for p in loader_call_sites(worktree, snapshot) if p not in covered]


def selection_seeds(changed: Sequence[str]) -> list[str]:
    """The changed paths plus the directory loaders of any changed path they scan."""
    seeds = list(changed)
    for prefix, loaders in DIRECTORY_LOADERS.items():
        if any(p.startswith(prefix) for p in changed):
            seeds.extend(loader for loader in loaders if loader not in seeds)
    return seeds


def reached_modules(
    changed: Sequence[str],
    graph: Mapping[str, set[str]],
    importers: Mapping[str, set[str]] | None = None,
) -> set[str]:
    """The changed vibe_quant/tests modules plus every module importing one of them,
    directly or transitively (the reverse closure over the graph's modules).

    `importers` is the prebuilt reverse index (`import_graph_pair`), which a plan reuses: it
    is 705 sets to rebuild, one of the two allocations that dominated a plan's garbage."""
    seeds = {mod for path in changed if (mod := graph_module_name(path))}
    if importers is None:
        built: dict[str, set[str]] = {}
        for path, imports in graph.items():
            if mod := graph_module_name(path):
                for target in imports:
                    built.setdefault(target, set()).add(mod)
        importers = built
    reached = set(seeds)
    frontier = list(seeds)
    while frontier:
        for mod in importers.get(frontier.pop(), ()):
            if mod not in reached:
                reached.add(mod)
                frontier.append(mod)
    return reached


@dataclass(frozen=True)
class TestScanIndex:
    """tests/unit/test_*.py indexed once per tree: the file list plus, for every dotted module
    name a test spells in a string, every dotted prefix (from 2 parts up) of that name, and
    every `"-m", "<pkg>"` package. A plan then asks a dict instead of parsing 446 files."""

    files: tuple[str, ...]
    prefix_files: dict[str, tuple[str, ...]]
    run_files: dict[str, tuple[str, ...]]


def test_scan_index(
    worktree: Path, snapshot: tuple[tuple[str, int, int], ...] | None = None
) -> TestScanIndex:
    """Per-process cached `_build_test_scan_index`, keyed by the plan's tree snapshot (an
    edit, add or delete under tests/ rebuilds it: it never serves a stale scan)."""
    key = ("testscan", worktree.resolve(), _snapshot_or_walk(worktree, snapshot))
    if key not in _TREE_CACHE:
        _TREE_CACHE[key] = _build_test_scan_index(worktree)
    index: TestScanIndex = _TREE_CACHE[key]
    return index


def _build_test_scan_index(worktree: Path) -> TestScanIndex:
    """Read every tests/unit test file ONCE and invert its string references.

    A read error raises (plan_tests then runs the full suite): a missing entry would shrink
    the selection."""
    root = worktree / "tests" / "unit"
    files: list[str] = []
    prefixes: dict[str, list[str]] = {}
    runs: dict[str, list[str]] = {}
    for path in sorted(root.rglob("test_*.py")) if root.is_dir() else ():
        rel = path.relative_to(worktree).as_posix()
        files.append(rel)
        text = read_cached(path, "replace")  # OSError: plan_tests -> full
        for name in sorted({m.group(1) for m in QUOTED_MODULE_RE.finditer(text)}):
            parts = name.split(".")
            for i in range(2, len(parts) + 1):
                bucket = prefixes.setdefault(".".join(parts[:i]), [])
                if not bucket or bucket[-1] != rel:  # files are visited in sorted order
                    bucket.append(rel)
        for package in sorted({m.group(1) for m in RUN_MODULE_RE.finditer(text)}):
            bucket = runs.setdefault(package, [])
            if not bucket or bucket[-1] != rel:
                bucket.append(rel)
    return TestScanIndex(
        tuple(files),
        {name: tuple(rels) for name, rels in prefixes.items()},
        {package: tuple(rels) for package, rels in runs.items()},
    )


def dynamic_reference_test_files(
    reached: AbstractSet[str],
    worktree: Path,
    snapshot: tuple[tuple[str, int, int], ...] | None = None,
) -> list[str]:
    """tests/unit test files naming a reached module in a STRING: a subprocess
    `[sys.executable, "-m", "vibe_quant.discovery"]` (the package of a reached __main__), an
    `importlib.import_module("vibe_quant.x")` or a `monkeypatch.setattr("vibe_quant.x.f", ...)`
    target. None of them is an import edge. A package counts only in the `"-m", "<pkg>"`
    form. Replaces the `-k <stem>` group for vibe_quant modules, which matched module NAMES
    (an api/routers/discovery.py change ran every test_discovery_*.py) instead of references.

    The index holds, per file, every dotted prefix of every name it spells, so "reached"
    answers with a dict lookup and the result is the old per-file prefix intersection exactly.
    A read error raises (plan_tests then runs the full suite).
    """
    if not reached:
        return []
    index = test_scan_index(worktree, snapshot)
    packages = {m.removesuffix(".__main__") for m in reached if m.endswith(".__main__")}
    out: set[str] = set()
    for module in reached:
        out.update(index.prefix_files.get(module, ()))
    for package in packages:
        out.update(index.run_files.get(package, ()))
    return sorted(out)


def importer_test_files(changed: Sequence[str], graph: Mapping[str, set[str]]) -> list[str]:
    """tests/unit test files that reach a changed vibe_quant/tests module (reverse closure).

    Selection by test NAME (`-k <stem>`) misses a caller test whose names never mention the
    module: tests/unit/test_symbol_agg.py::test_topk_ranks_gate_passers_first stayed red
    through DiscoveryPipeline.run while the gate PASSed (20261015-funding-carry pqon2 r1).
    The seeds are the modules the diff names; the walk follows vibe_quant AND tests modules
    (a helper, or a test file another test imports). Sorted: the pytest command line must
    not depend on filesystem order.
    """
    reached = reached_modules(changed, graph)
    if not reached:
        return []
    return sorted(
        path
        for path, imports in graph.items()
        if path.startswith("tests/unit/") and is_changed_test(path) and imports & reached
    )


def path_mention_test_files(changed: Sequence[str], worktree: Path) -> list[str]:
    """tests/unit test files that MENTION a changed vibe_quant module's path or file name.

    The reverse import closure only sees ``import`` edges. A test that reads the source file
    BY PATH - tests/unit/test_state_manager_locking.py's ``_EXTERNAL_MODULES`` tuple and
    ``(root / rel).read_text()`` - is invisible to it and to `-k <stem>` when the module's stem
    is not in the test's own name (bead olizd). Needles are the repo-relative path and the
    module FILE name: a bare stem would select every test that merely mentions "pipeline" or
    "runner". Only vibe_quant/**/*.py paths are needles; a changed test file selects nothing.
    Sorted: the pytest command line must not depend on filesystem order.
    """
    needles: set[str] = set()
    for path in changed:
        if graph_module_name(path):
            needles.update(file_needles(path))
    return [t for t in tests_mentioning(worktree, needles) if not is_gate_meta_test(t)]


def is_gate_meta_test(test: str) -> bool:
    """tests/unit/test_swarm_check*.py test this gate on synthetic repos and quote vibe_quant
    paths only as fixture text, so a mention proves no dependency. Left in, every vibe_quant
    diff paid ~140 s of them on the xdist critical path (bead c5pfj). They still run when
    they or scripts/agents/swarm_check.py change (import closure), and the loaders policy
    test is in POLICY_TESTS."""
    return Path(test).name.startswith("test_swarm_check")


def md_reader_test_files(changed: Sequence[str], worktree: Path) -> list[str]:
    """For a changed `*.md` outside vibe_quant/: every tests/unit file whose source spells
    the `.md` suffix (a reader names it: Path("SKILL.md"), glob("*.md"), prompts/x.md). A
    test that walks a directory without naming the suffix is a POLICY_TESTS scanner, which
    always runs. test_mkbrief.py renders docs/orchestration/prompts/*.md, so .md is not
    inert."""
    if not any(p.endswith(".md") and not p.startswith("vibe_quant/") for p in changed):
        return []
    return tests_mentioning(worktree, {".md"})


def file_needles(path: str) -> set[str]:
    """The strings a test or script that reaches `path` BY PATH must contain: the
    repo-relative path and the file name, or "<parent>/<name>" for a generic name or one
    without a suffix (the `scripts/agents/bus` shim: a bare "bus" is in every bus test)."""
    p = Path(path)
    generic = p.name in GENERIC_FILE_NAMES or not p.suffix
    return {path, f"{p.parent.name}/{p.name}" if generic else p.name}


def tests_mentioning(worktree: Path, needles: AbstractSet[str]) -> list[str]:
    """Sorted tests/unit test files whose text contains any needle (plain substring).

    A needle that CONTAINS another needle adds no file: a file holding the longer string holds
    the shorter one. `file_needles` returns the repo path AND the file name, so this leaves one
    scan per changed file instead of two - an exact shortcut, not a narrowing (pinned by
    tests/unit/test_swarm_check_scan_index.py against this same scan without it)."""
    if not needles:
        return []
    root = worktree / "tests" / "unit"
    if not root.is_dir():
        return []
    minimal = [n for n in needles if not any(o != n and o in n for o in needles)]
    out: list[str] = []
    for path in sorted(root.rglob("test_*.py")):
        text = read_cached(path, "replace")  # OSError: plan_tests -> full
        if any(needle in text for needle in minimal):
            out.append(path.relative_to(worktree).as_posix())
    return out


def script_consumer_test_files(changed: Sequence[str], worktree: Path) -> list[str]:
    """tests/unit test files that reach a changed scripts/** file by NAME, directly or through
    other scripts.

    Scripts load each other by path (`spec_from_file_location(..., with_name("x.py"))`, a
    subprocess of `scripts/agents/x.py`), so there is no import edge to walk. 2026-10-22:
    commit 8805dea8 changed scripts/agents/handoff_lint.py, the gate PASSed, and
    tests/unit/test_swarm_bus.py failed - bus.py loads handoff_lint.py by file name and the
    test loads bus.py. The walk: every scripts/** file whose text names a reached file joins
    the reached set (fixpoint), then every test file naming any reached file is selected.
    Needles are file_needles (path + file name), never a bare stem. A reached
    scripts/agents/<mod>.py is also reached through a Python import by module name
    (script_import_re): review rev-gate-1 B6, such an importer was only a `-k` candidate.
    """
    seeds = [p for p in changed if p.startswith("scripts/") and not is_board_path(p)]
    if not seeds:
        return []
    root = worktree / "scripts"
    texts: dict[str, str] = {}
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            texts[path.relative_to(worktree).as_posix()] = path.read_text(
                encoding="utf-8", errors="replace"
            )  # OSError: plan_tests runs the full suite
    reached = set(seeds)
    needles: set[str] = set()
    for seed in seeds:
        needles |= file_needles(seed)
    imports = script_import_re(reached)
    grew = True
    while grew:
        grew = False
        for rel, text in texts.items():
            if rel in reached:
                continue
            if any(needle in text for needle in needles) or (imports and imports.search(text)):
                reached.add(rel)
                needles |= file_needles(rel)
                imports = script_import_re(reached)
                grew = True
    out = set(tests_mentioning(worktree, needles))
    unit = worktree / "tests" / "unit"
    if imports is not None and unit.is_dir():
        for path in sorted(unit.rglob("test_*.py")):
            if imports.search(read_cached(path, "replace")):  # OSError: plan_tests -> full
                out.add(path.relative_to(worktree).as_posix())
    return sorted(out)


def script_import_re(paths: Iterable[str]) -> re.Pattern[str] | None:
    """Matches a Python import of any scripts/agents/<mod>.py in `paths` by module name:
    `scripts.agents.<mod>`, `from scripts.agents import (..., <mod>)`, and `import <mod>` /
    `from <mod> import` (after a sys.path insert of scripts/agents). None: no such module."""
    mods = sorted({Path(p).stem for p in paths if re.fullmatch(r"scripts/agents/\w+\.py", p)})
    if not mods:
        return None
    alt = "|".join(mods)  # \w+ stems: nothing to escape
    return re.compile(
        rf"\bscripts\.agents\.(?:{alt})\b"
        rf"|^[ \t]*from[ \t]+scripts\.agents[ \t]+import[ \t]+(?:\([^)]*|[^\n]*)\b(?:{alt})\b"
        rf"|^[ \t]*import[ \t][^\n]*\b(?:{alt})\b|^[ \t]*from[ \t]+(?:{alt})\b",
        re.MULTILINE,
    )


def policy_test_files(worktree: Path, changed: Sequence[str] | None = None) -> list[str]:
    """The POLICY_TESTS present in `worktree` whose scanned tree the diff touches, in constant
    order (all of them when `changed` is None).

    A tree scan can only change verdict when the tree it scans changes: POLICY_TRIGGERS, its
    own file and POLICY_ALWAYS. Filtered by existence so a tree that predates a file (or a
    throwaway test repo) never hands pytest a path that does not exist.
    """

    def hit(test: str) -> bool:
        if changed is None:
            return True
        globs = (*POLICY_TRIGGERS.get(test, ("*",)), test, *POLICY_ALWAYS)
        return any(fnmatch.fnmatch(p, g) for p in changed for g in globs)

    return [p for p in POLICY_TESTS if (worktree / p).is_file() and hit(p)]


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


def failed_tests(output: str, limit: int = 3) -> list[str]:
    """The first failing node ids, so the FAIL line names what to re-run alone."""
    ids = FAILED_TEST_RE.findall(output)
    if not ids:
        return []
    more = f" (+{len(ids) - limit} more)" if len(ids) > limit else ""
    return ["FAILED " + ", ".join(ids[:limit]) + more]


def tail_line(output: str) -> str:
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    return lines[-1] if lines else "ok"


def format_line(check: Check) -> str:
    return f"{check.status} {check.name}: {check.detail}"


def apply_warn_mode(
    checks: Sequence[Check], today: str | None = None, until: dict[str, str] | None = None
) -> list[Check]:
    """A FAIL of a check still in warn mode (``WARN_MODE_UNTIL``) becomes a WARN."""
    until = WARN_MODE_UNTIL if until is None else until
    today = today or time.strftime("%Y-%m-%d")
    return [
        Check(c.name, "WARN", f"warn-mode until {until[c.name]}: {c.detail}")
        if c.status == "FAIL" and today < until.get(c.name, "")
        else c
        for c in checks
    ]


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
    """sha256 per main-checkout input of the gate (exactness scripts, lockfiles, goldens)."""
    out: dict[str, str] = {}
    for rel in MAIN_KEY_FILES:
        try:
            out[rel] = hashlib.sha256((main_root / rel).read_bytes()).hexdigest()
        except OSError:
            out[rel] = MISSING_FILE_TOKEN
    return out


def _pins_identity(path: str) -> str:
    """Path + content hash of the declared pins: a PASS against other pins is not reused."""
    if not path:
        return ""
    try:
        return f"{path}:{hashlib.sha256(Path(path).read_bytes()).hexdigest()}"
    except OSError:
        return f"{path}:missing"


def cache_key(tree: str, base_sha: str, cfg: Config, thash: str, merge_base: str = "") -> str:
    """`tree` is HEAD^{tree}, not the commit: a message-only amend after a PASS keeps the
    tree, the base and the merge-base, so it reuses the stamp (HANDOFF R3: 17 makers re-ran
    a 9-minute gate after rewording the commit)."""
    payload = json.dumps(
        {
            "tree": tree,
            "base": base_sha,
            "merge_base": merge_base,
            "full": cfg.full,
            "scope": sorted(cfg.scope),
            "exactness": cfg.exactness,
            "frontend": cfg.frontend,
            "tool": thash,
            "main": main_files_hash(cfg.main_root),
            "expect_pins": _pins_identity(cfg.expect_pins),
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


def _git_rc(wt: Path, run: RunFn, *args: str) -> int:
    return run(["git", "-C", str(wt), *args], wt, None, TOOL_TIMEOUT)[0]


def _git_out(wt: Path, run: RunFn, *args: str) -> str:
    rc, out = run(["git", "-C", str(wt), *args], wt, None, TOOL_TIMEOUT)
    return out.strip() if rc == 0 else ""


def git_dir(wt: Path, run: RunFn, facts: GitFacts | None = None) -> Path | None:
    """This worktree's own git dir (per-worktree for linked worktrees; never committed)."""
    out = (
        facts.out(wt, run, "rev-parse", "--absolute-git-dir")
        if facts is not None
        else _git_out(wt, run, "rev-parse", "--absolute-git-dir")
    )
    return Path(out) if out else None


def stamp_path(wt: Path, run: RunFn, key: str, facts: GitFacts | None = None) -> Path | None:
    gd = git_dir(wt, run, facts)
    return gd / CACHE_DIR / f"pass-{key[:16]}.json" if gd is not None else None


def head_reflog_count(wt: Path, run: RunFn) -> int | None:
    """HEAD reflog length, or None when git cannot report it (the caller fails closed)."""
    out = _git_out(wt, run, "rev-list", "--walk-reflogs", "--count", "HEAD")
    return int(out) if out.isdigit() else None


def gate_log_target(cfg: Config, run: RunFn, facts: GitFacts | None = None) -> tuple[Path, int]:
    """(this run's log path, number of prior logs for this worktree name + HEAD).

    The name is `<worktree dir name>-<HEAD[:12]>-<UTC %Y%m%dT%H%M%SZ>.log` under the MAIN
    checkout's data/swarm/_gate-logs (gitignored there). The prior count drives the
    informational "gate run #<n+1>" line; a second run in the same second overwrites the
    first log rather than counting it twice.
    """
    head = (
        facts.out(cfg.worktree, run, "rev-parse", "HEAD")
        if facts is not None
        else _git_out(cfg.worktree, run, "rev-parse", "HEAD")
    ) or "unknown"
    head12 = head[:12]
    log_dir = cfg.main_root.joinpath(*GATE_LOG_SUBDIR)
    pattern = f"{cfg.worktree.name}-{head12}-*.log"
    prior = list(log_dir.glob(pattern)) if log_dir.is_dir() else []
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return log_dir / f"{cfg.worktree.name}-{head12}-{ts}.log", len(prior)


def _ledger_rows(path: Path) -> list[dict[str, Any]]:
    """Every well-formed JSON object line of a `.jsonl` ledger (a garbled line is ignored)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def gate_runs_cap(cfg: Config, log_path: Path) -> Check | None:
    """None when this uncached run may proceed (and is counted), else the FAIL check.

    A steering cap, enforced in code: the gate runs once per round, and 47 maker rounds in one
    job ran it more than once (one 9 times). A maker round gets GATE_RUNS_CAP uncached runs,
    counted in the JOB's ledger (`<bus dir>/../gate-runs.jsonl`, one line per run with the LOG
    it wrote); `$SWARM_GATE_CAP` overrides the cap: a value that is not an integer keeps the
    default rather than lifting the guard, and a negative one counts as 0 (every run refused). A cached PASS is never refused
    nor counted (this runs after the cached-stamp branch); `ver-*`, reviewers, `--full` and a
    run with no bus are outside it.
    """
    if cfg.full or not cfg.bus or not cfg.bus_agent.startswith(CAPPED_AGENT_PREFIXES):
        return None
    cap = GATE_RUNS_CAP
    raw = os.environ.get("SWARM_GATE_CAP")
    if raw is not None:
        with contextlib.suppress(ValueError):
            cap = max(int(raw), 0)
    ledger = Path(cfg.bus).parent / GATE_RUNS_FILE
    mine = [row for row in _ledger_rows(ledger) if row.get("agent") == cfg.bus_agent]
    if len(mine) >= cap:
        last = str(mine[-1].get("log") or "") if mine else "(none: no earlier run)"
        return Check(
            "gate-runs",
            "FAIL",
            f"{cfg.bus_agent} ran the gate {len(mine)} times (cap {cap}); read the last LOG "
            f"{last}, fix, then ask the chief in your thread ([question]) for another run",
        )
    ledger.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "agent": cfg.bus_agent,
        "worktree": cfg.worktree.name,
        "log": str(log_path),
    }
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return None


def write_gate_log(
    path: Path,
    entries: Sequence[tuple[str, int, str] | str],
    header: str = "",
    trailer: Sequence[str] = (),
) -> None:
    """Write the tool transcript: command line, combined output, exit code per tool call; a
    str entry is a context line (ToolLog.note), written as is."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        if header:
            fh.write(header + "\n")
        for entry in entries:
            if isinstance(entry, str):
                fh.write(entry + "\n")
                continue
            cmd, rc, out = entry
            fh.write(f"$ {cmd}\n")
            fh.write(out if out.endswith("\n") else out + "\n")
            fh.write(f"[exit {rc}]\n")
        for line in trailer:
            fh.write(line + "\n")


def _gated_tree(cfg: Config, run: RunFn, facts: GitFacts) -> tuple[str, str]:
    """(HEAD, HEAD^{tree}) as `facts` sees them; the tree is `dirty` with uncommitted tracked
    edits (the tools read content no commit holds) and `unknown` when git cannot say."""
    head = facts.out(cfg.worktree, run, "rev-parse", "HEAD") or "unknown"
    tree = facts.out(cfg.worktree, run, "rev-parse", "HEAD^{tree}") or "unknown"
    rc, status = facts.call(
        cfg.worktree, run, "status", "--porcelain=v1", "-z", "--untracked-files=all"
    )
    if rc != 0:
        tree = "unknown"
    elif parse_porcelain(status)[2]:
        tree = "dirty"
    return head, tree


def gate_log_trailer(
    cfg: Config, run: RunFn, pre: GitFacts, post: GitFacts, gate: str
) -> list[str]:
    """The log's last lines: what was gated (`HEAD <sha> TREE <tree>`), then the verdict.

    `pre` is the pre-run memo, `post` a NEW one read after the tools (shared with the stamp's
    freshness probe): an edit or commit made while the tools ran makes TREE `moved` (no single
    tree was gated). A message-only amend keeps the tree, so handoff_lint still matches it."""
    head, tree = _gated_tree(cfg, run, pre)
    if _gated_tree(cfg, run, post) != (head, tree):
        tree = "moved"
    return [f"HEAD {head} TREE {tree}", gate]


def annotate_tool_fails(checks: Sequence[Check]) -> list[Check]:
    """Append the log pointer to every FAIL line for a check that ran an external tool."""
    return [
        Check(c.name, c.status, c.detail + " (full output: LOG above)")
        if c.status == "FAIL" and c.name in TOOL_CHECK_NAMES
        else c
        for c in checks
    ]


def _cache_probe(
    cfg: Config,
    run: RunFn,
    state: GitState | None = None,
    facts: GitFacts | None = None,
) -> tuple[str, GitState | None]:
    """(cache key, git state): the key is "" when a stamp must not be reused or written.

    `state`/`facts` let the pre-run caller hand over what it already asked: the probed state
    is the same one `run_checks` gates, and re-reading it here was a diff+status spawn pair.
    The post-run probe passes neither: it must be FRESH (the whole point of re-probing)."""
    try:
        facts = GitFacts() if facts is None else facts
        state = git_state(cfg.worktree, cfg.base, run, facts) if state is None else state
        if not is_clean(state):
            return "", state
        tree = facts.out(cfg.worktree, run, "rev-parse", "HEAD^{tree}")
        base_sha = facts.out(
            cfg.worktree, run, "rev-parse", "--verify", "--quiet", cfg.base + "^{commit}"
        )
        merge_base = facts.out(cfg.worktree, run, "merge-base", cfg.base, "HEAD")
        if not tree or not base_sha or not merge_base:
            return "", state
        return cache_key(tree, base_sha, cfg, tool_hash(), merge_base), state
    except Exception:  # fail closed: any doubt -> run the full gate
        return "", None


def load_stamp(wt: Path, run: RunFn, key: str, facts: GitFacts | None = None) -> Stamp | None:
    """A valid stored PASS for `key`, or None (any corruption/mismatch -> full gate)."""
    if not key:
        return None
    try:
        path = stamp_path(wt, run, key, facts)
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


def write_stamp(
    wt: Path,
    run: RunFn,
    cfg: Config,
    key: str,
    checks: Sequence[Check],
    facts: GitFacts | None = None,
) -> None:
    if not key:
        return
    path = stamp_path(wt, run, key, facts)
    if path is None:
        return
    # `facts` is the POST-run memo: its HEAD/base answers are the ones the key was checked
    # against, so the stamp records exactly what was gated (and asks git nothing new).
    def out(*args: str) -> str:
        return (
            facts.out(wt, run, *args)
            if facts is not None
            else _git_out(wt, run, *args)
        )

    data = {
        "version": CACHE_VERSION,
        "key": key,
        "head": out("rev-parse", "HEAD"),
        "base": cfg.base,
        "base_sha": out("rev-parse", "--verify", "--quiet", cfg.base + "^{commit}"),
        "scope": sorted(cfg.scope),
        "exactness": cfg.exactness,
        "frontend": cfg.frontend,
        "full": cfg.full,
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


def file_cache_env(main_root: Path) -> str | None:
    """sha256 of the cache version, the gate python's sys.version, the main checkout's
    FILE_CACHE_ENV_FILES and the sorted *.dist-info names of its venv (a package changed with
    no lockfile change): a PASS under another interpreter or other deps is never reused. None
    when any input is unreadable or no site-packages exists: this run neither reuses nor
    stores (review rev-gate-1 B1)."""
    digest = hashlib.sha256(f"v{FILE_CACHE_VERSION}\0{sys.version}".encode())
    try:
        for rel in FILE_CACHE_ENV_FILES:
            data = (main_root / rel).read_bytes()
            digest.update(rel.encode() + b"\0" + hashlib.sha256(data).digest())
        sites = sorted((main_root / ".venv" / "lib").glob("python*/site-packages"))
        if not sites:
            return None
        for site in sites:
            names = sorted(p.name for p in site.iterdir() if p.name.endswith(".dist-info"))
            digest.update(site.as_posix().encode() + b"\0" + "\0".join(names).encode())
    except OSError:
        return None
    return digest.hexdigest()


def _file_entry(cache_dir: Path, env: str, test: str) -> Path:
    name = hashlib.sha256("\0".join((env, test)).encode()).hexdigest()[:32]
    return cache_dir / f"{name}.json"


def load_file_pass(cache_dir: Path, env: str, test: str) -> str | None:
    """The tree `test` last PASSed at, or None (missing, unreadable or foreign entry: a miss)."""
    try:
        data = json.loads(_file_entry(cache_dir, env, test).read_text(encoding="utf-8"))
        tree = data.get("tree")
        if (
            data.get("version") == FILE_CACHE_VERSION
            and data.get("env") == env
            and data.get("path") == test
            and isinstance(tree, str)
            and TREE_SHA_RE.fullmatch(tree)
        ):
            return tree
    except Exception:
        pass
    return None


def write_file_passes(cache_dir: Path, env: str, tree: str, tests: Sequence[str]) -> None:
    """Store `tree` as the PASS of each test file; atomic per entry (worktrees gate at once)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    for test in tests:
        data = {"version": FILE_CACHE_VERSION, "env": env, "path": test, "tree": tree, "ts": ts}
        fd, tmp_name = tempfile.mkstemp(dir=cache_dir, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(data) + "\n")
            os.replace(tmp_name, _file_entry(cache_dir, env, test))
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise


def diff_reaches(worktree: Path, paths: Sequence[str]) -> set[str] | None:
    """The tests/unit files this gate's per-round selection runs for a diff of `paths`: the
    consumers, the `-k` candidates, the changed tests and the triggered policy tests. None
    when it would run the whole suite, a conftest/pyproject/uv.lock changed, or the
    selection is unknown (any error): the caller then reuses nothing."""
    if full_suite_reason(paths) or any(
        Path(p).name == "conftest.py" or p in FULL_SUITE_FILES for p in paths
    ):
        return None
    try:
        reached = _selected_consumers(worktree, paths)
        if expr := k_expr([p for p in paths if not graph_module_name(p)]):
            reached.update(k_candidate_files(worktree, expr, set()))
        reached.update(policy_test_files(worktree, paths))
    except Exception:
        return None
    return reached | set(changed_test_files(paths))


def reusable_test_files(
    cfg: Config, run: RunFn, candidates: Sequence[str], head_tree: str, env: str
) -> dict[str, str]:
    """{test file: tree it PASSed at} for the candidates whose stored PASS still holds: the
    gate's own selection (diff_reaches) for `git diff <that tree> <HEAD tree>` does not reach
    them. A tree missing from the object DB or a failed git call is a miss (fail closed)."""
    if cfg.file_cache_dir is None:
        return {}
    by_tree: dict[str, list[str]] = {}
    for test in candidates:
        if (tree := load_file_pass(cfg.file_cache_dir, env, test)) is not None:
            by_tree.setdefault(tree, []).append(test)
    wt = cfg.worktree
    out: dict[str, str] = {}
    for tree, tests in by_tree.items():
        reached: set[str] | None = set()
        if tree != head_tree:
            if run(["git", "-C", str(wt), "cat-file", "-e", f"{tree}^{{tree}}"], wt, None, TOOL_TIMEOUT)[0]:
                continue
            rc, diff = run(
                ["git", "-C", str(wt), "diff", "--name-only", "--no-renames", "-z", tree, head_tree],
                wt, None, TOOL_TIMEOUT,
            )
            reached = diff_reaches(wt, [p for p in diff.split("\0") if p]) if rc == 0 else None
        if reached is not None:
            out.update((test, tree) for test in tests if test not in reached)
    return out


# Review rev-gate-1 B2 (chief ruling R3c): under xdist a collection hook's deselection runs in
# the workers and no summary line counts it, so a call holding cacheable files loads this
# record-only plugin (-p, from the gate's junit tmp dir): each process appends the nodeids it
# deselected to its own file (worker id), and a file with a deselected item is never stored.
DESELECT_PLUGIN = "swarm_check_deselect_record"
DESELECT_ENV = "SWARM_CHECK_DESELECT_DIR"
DESELECT_PLUGIN_SOURCE = f"""\
import os
from pathlib import Path


def _record():
    return Path(os.environ["{DESELECT_ENV}"]) / (os.environ.get("PYTEST_XDIST_WORKER", "main") + ".txt")


def pytest_configure(config):
    _record().touch()  # this process loaded the plugin


def pytest_deselected(items):
    with _record().open("a", encoding="utf-8") as fh:
        fh.write("".join(item.nodeid + "\\n" for item in items))
"""


def deselected_files(records: Sequence[Path]) -> set[str] | None:
    """The test files with an item a call deselected, from the DESELECT_PLUGIN record dirs.
    None (store nothing) when a dir holds no record (the plugin did not load), a record ends
    mid-line, or one cannot be read."""
    out: set[str] = set()
    try:
        for record in records:
            files = sorted(record.glob("*.txt"))
            if not files:
                return None
            for path in files:
                text = path.read_text(encoding="utf-8")
                if text and not text.endswith("\n"):
                    return None
                out.update(line.split("::")[0] for line in text.splitlines())
    except (OSError, UnicodeDecodeError):
        return None
    return out


def junit_passed_files(junits: Sequence[Path], files: Sequence[str]) -> list[str]:
    """The `files` with at least one testcase in the junit reports and no failure, error or
    skipped row (skip, xfail and xpass: the run did not prove the file green; review
    rev-gate-1 B2). A testcase's classname is its file's dotted module, then the class. Any
    unreadable report: [] (nothing is stored)."""
    modules = {f[:-3].replace("/", "."): f for f in files if f.endswith(".py")}
    seen: set[str] = set()
    bad: set[str] = set()
    try:
        for path in junits:
            for case in ET.parse(path).getroot().iter("testcase"):
                cls = case.get("classname", "")
                owner = next(
                    (f for m, f in modules.items() if cls == m or cls.startswith(m + ".")), None
                )
                if owner is None:
                    continue
                seen.add(owner)
                if any(case.find(tag) is not None for tag in ("failure", "error", "skipped")):
                    bad.add(owner)
    except (ET.ParseError, OSError):
        return []
    return sorted(seen - bad)


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


@dataclass(frozen=True)
class TestPlan:
    """The pytest calls a gate run makes: (args, rc5_is_skip, rc5_note) per call.

    `explicit` is every test file handed to pytest by path WITHOUT a -k filter (the
    pinned-text check counts those as run); `full` says the whole tests/unit runs; `notes`
    go into the tests detail; `files` is every test file the plan touches (for --list-tests).
    """

    commands: list[tuple[list[str], bool, str]]
    explicit: frozenset[str]
    full: bool
    notes: list[str]
    files: list[str]
    reasons: dict[str, list[str]] = field(default_factory=dict)  # selection kind -> files (--list-tests)
    # How many leading commands fail fast: a red one ends the tests step (the policy call).
    fail_fast: int = 0


def k_candidate_files(worktree: Path, expr: str, exclude: AbstractSet[str]) -> list[str]:
    """tests/unit test files a `-k <stems>` run could select anything from, minus `exclude`.

    pytest -k matches names case-insensitively: the module's path, class, function and
    parametrize ids, markers. Every one of those is in the file's path or text, so a file
    mentioning no stem (case-insensitive substring) cannot match and is not collected - a
    `tests/unit -k x` run paid a whole-suite collection (9-13 s per xdist worker) to keep a
    handful of files. Known limit: a test method inherited from a class defined in another
    file, or an id built at runtime, matches only in the --full run.
    """
    stems = [st.lower() for st in expr.split(" or ") if st]
    root = worktree / "tests" / "unit"
    if not stems or not root.is_dir():
        return []
    out: list[str] = []
    for path in sorted(root.rglob("test_*.py")):
        rel = path.relative_to(worktree).as_posix()
        if rel in exclude:
            continue
        text = (rel + "\n" + path.read_text(encoding="utf-8", errors="replace")).lower()
        if any(st in text for st in stems):
            out.append(rel)
    return out


class FullSuite(Exception):
    """The selection cannot represent the change: run the whole tests/unit (str = reason)."""


def _selected_consumers(
    worktree: Path, changed: Sequence[str], why: dict[str, list[str]] | None = None
) -> set[str]:
    """Every tests/unit file reaching a changed path: the reverse import closure over
    vibe_quant AND tests modules (seeded with the directory loaders of a changed plugin), the
    tests naming a changed module's path or a reached module in a string, the consumers of a
    changed script, and the tests that may read a changed non-Python file. Raises GraphError
    or OSError when a file cannot be read, FullSuite when the change reaches something the
    selection does not model: the caller then runs the full suite. `why` collects the files
    each rule selected (kind -> files). A changed tests/fixtures resource adds its readers
    (fixture_reader_files) as if they had changed: a reader test runs, a reader helper
    seeds its importers, a reader script its consumers."""
    selected: set[str] = set()

    def add(kind: str, files: Sequence[str]) -> None:
        selected.update(files)
        if why is not None and files:
            why[kind] = sorted(files)

    readers = fixture_reader_files(changed, worktree)
    if implicit := [r for r in readers if Path(r).stem in FULL_SUITE_STEMS]:
        raise FullSuite(f"{implicit[0]} may read a changed fixture (loaded by pytest itself)")
    add("fixture", [
        r for r in readers
        if r.startswith("tests/unit/") and is_changed_test(r) and not is_gate_meta_test(r)
    ])
    changed = [*changed, *(r for r in readers if r not in changed)]
    seeds = selection_seeds(changed)
    snapshot = plan_snapshot(worktree)  # ONE walk, shared by every tree-wide scan below

    if any(graph_module_name(p) for p in seeds):
        if loose := unmodelled_loaders(worktree, snapshot):
            raise FullSuite(f"unmodelled dynamic loader {', '.join(loose)}")
        graph, importers = import_graph_pair(worktree, seeds, snapshot)
        reached = reached_modules(seeds, graph, importers)
        add("importer", importer_test_files(seeds, graph))
        add("mention", path_mention_test_files(seeds, worktree))
        add("dynamic", dynamic_reference_test_files(reached, worktree, snapshot))
        check_pytest_plugins(worktree, snapshot)
        add("conftest", conftest_fixture_test_files(reached, graph, worktree, snapshot))
    add("script", script_consumer_test_files(changed, worktree))
    add("md", md_reader_test_files(changed, worktree))
    return selected


# `sys.modules.get("vibe_quant.x")` acts on the module only when the test already loaded it -
# and a test that loads it imports it, so the import closure selects that test already. Such
# a lookup is no fixture dependency (tests/unit/conftest.py's _restore_aux_data_paths and
# _restore_ga_indicator_pool). `sys.modules["x"]` (raises when absent) stays a dependency.
LOADED_ONLY_RE = re.compile(r"""sys\.modules\.get\(\s*(['"])[\w.]+\1\s*\)""")


def _names_pytest_plugins(tree: ast.AST) -> bool:
    """True when `tree` names `pytest_plugins` (a Name, Attribute or assignment target at any
    nesting): pytest reads the attribute after the module ran, so a conditional, nested or
    computed registration counts. Blunt on purpose, no modelling (review rev-gate-speed-3/4
    F2). A string is no hit. Walks the WHOLE tree, `if TYPE_CHECKING:` bodies included: their
    registration is what pytest reads."""
    for sub in ast.walk(tree):
        name = (
            sub.id if isinstance(sub, ast.Name)
            else sub.attr if isinstance(sub, ast.Attribute)
            else None
        )
        if name == "pytest_plugins":
            return True
    return False


def _plugins_message(rel: str) -> str:
    """The FullSuite detail naming one offending file."""
    return f"pytest_plugins in {rel} (plugin fixtures are not modelled)"


def _plugins_reason(offenders: Sequence[str]) -> str | None:
    """The FullSuite message for the first offender, None when tests/ registers no plugin.

    `sorted(Path)` order, not string order: `tests/a/b.py` precedes `tests/a.py` (Path
    compares parts), which is the order the uncached scan reported the first offender in."""
    if not offenders:
        return None
    return _plugins_message(min(offenders, key=lambda rel: Path(rel).parts))


def _plugins_cache_key(
    worktree: Path, snapshot: tuple[tuple[str, int, int], ...]
) -> tuple[object, ...]:
    """`_TREE_CACHE` key of the pytest_plugins verdict: one per (worktree, tree snapshot).
    The graph walk FILLS it (`_build_import_graph`); `check_pytest_plugins` reads it."""
    return ("plugins", worktree.resolve(), snapshot)


def check_pytest_plugins(
    worktree: Path, snapshot: tuple[tuple[str, int, int], ...] | None = None
) -> None:
    """FullSuite when any tests/**/*.py names `pytest_plugins`, cached per process by the
    plan's tree snapshot (a FullSuite is cached and re-raised too).

    A plan gets the verdict from the import-graph walk's fold, which visits every
    tests/**/*.py already; `_check_pytest_plugins` is the fallback for a caller that asks
    without a graph, and the entry it fills keeps the next ask free."""
    key = _plugins_cache_key(worktree, _snapshot_or_walk(worktree, snapshot))
    if key not in _TREE_CACHE:
        try:
            _check_pytest_plugins(worktree)
            _TREE_CACHE[key] = None
        except FullSuite as exc:
            _TREE_CACHE[key] = str(exc)
    reason = _TREE_CACHE[key]
    if reason is not None:
        raise FullSuite(str(reason))


def _check_pytest_plugins(worktree: Path) -> None:
    """FullSuite on the first tests/**/*.py (sorted path order) that names `pytest_plugins`.

    Only `check_pytest_plugins`' fallback calls this: a plan's graph walk has already folded
    the same check in, without the second read and parse of every test file."""
    root = worktree / "tests"
    files = sorted(root.rglob("*.py")) if root.is_dir() else []
    for path in (f for f in files if "__pycache__" not in f.parts):
        tree = ast_cached(read_cached(path))  # SyntaxError: GraphError raised earlier
        if _names_pytest_plugins(tree):
            raise FullSuite(_plugins_message(path.relative_to(worktree).as_posix()))


PATCH_METHODS = frozenset({"setattr", "setitem", "delattr", "delitem", "setenv", "delenv"})


@dataclass
class _Fixture:
    name: str
    autouse: bool
    params: set[str]
    deps: set[str]
    patch_only: bool
    dynamic_request: bool = False  # a getfixturevalue(<non-constant>) in its body or helpers


REQUEST_CALLS = frozenset({"getfixturevalue", "usefixtures"})


def _request_call(node: ast.AST) -> tuple[set[str], bool] | None:
    """(fixture names, dynamic) for a `<x>.getfixturevalue(...)` / `<x>.usefixtures(...)`
    call, None for any other node. Constant positional strings and a constant
    `argname="x"` are names; anything else (a variable, `*args`, `**kw`, another keyword)
    is dynamic. Shared by fixture bodies and test files (review rev-gate-speed-4 F1)."""
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in REQUEST_CALLS
    ):
        return None
    names: set[str] = set()
    dynamic = False
    values = list(node.args)
    for kw in node.keywords:
        if kw.arg == "argname" and node.func.attr == "getfixturevalue":
            values.append(kw.value)
        else:
            dynamic = True  # **kw (arg None) or an unknown keyword
    for value in values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            names.add(value.value)
        else:
            dynamic = True
    return names, dynamic


def _runtime_requests(nodes: Sequence[ast.AST]) -> tuple[set[str], bool]:
    """(constant fixture names requested at runtime, True when a request is dynamic) in
    `nodes`. Review rev-gate-speed-3/4 F1: a fixture requesting another is an edge."""
    names: set[str] = set()
    dynamic = False
    for node in nodes:
        for sub in ast.walk(node):
            if (found := _request_call(sub)) is not None:
                names |= found[0]
                dynamic |= found[1]
    return names, dynamic


def _fixture_decorator(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, bool] | None:
    """(fixture name, autouse) when `fn` is decorated with pytest.fixture / fixture."""
    for dec in fn.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        is_fixture = (isinstance(target, ast.Attribute) and target.attr == "fixture") or (
            isinstance(target, ast.Name) and target.id == "fixture"
        )
        if not is_fixture:
            continue
        name, autouse = fn.name, False
        for kw in dec.keywords if isinstance(dec, ast.Call) else ():
            if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                name = str(kw.value.value)
            if kw.arg == "autouse":
                autouse = not (isinstance(kw.value, ast.Constant) and kw.value.value is False)
        return name, autouse
    return None


def _bound_import_names(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, (ast.Import, ast.ImportFrom)):
            for alias in sub.names:
                out.add(alias.asname or alias.name.split(".")[0])
    return out


def _only_patched(fn: ast.AST, names: AbstractSet[str]) -> bool:
    """True when every load of `names` in `fn` is the target of a statement-level
    `<x>.setattr(name, "CONST", value)` (setitem/delattr/setenv...): the fixture only patches
    attributes on the module, never calls into it nor uses a return value."""
    allowed: set[int] = set()
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Expr) and isinstance(sub.value, ast.Call):
            call = sub.value
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr in PATCH_METHODS
                and len(call.args) >= 2
                and isinstance(call.args[0], ast.Name)
                and isinstance(call.args[1], ast.Constant)
                and isinstance(call.args[1].value, str)
            ):
                allowed.add(id(call.args[0]))
    return all(
        id(sub) in allowed
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Name) and sub.id in names and isinstance(sub.ctx, ast.Load)
    )


def _conftest_parts(
    source: str, module: str, known: AbstractSet[str], bare: Mapping[str, str]
) -> tuple[set[str], list[_Fixture]]:
    """(deps every fixture and hook of this conftest shares, its fixtures with their deps).

    Shared = module-level runtime imports and string refs, plus every pytest_* hook's.
    A fixture's deps = its body's imports and quoted "vibe_quant.x" names (sys.modules.get)
    plus those of each conftest-local helper function it references (transitively)."""
    tree = ast_cached(source)

    def deps_of(node: ast.AST) -> set[str]:
        text = LOADED_ONLY_RE.sub("None", ast.unparse(node))
        found = parse_imports(text, module, known, bare=dict(bare), runtime_only=True) or set()
        return found | string_module_refs(text, known)

    helpers: dict[str, ast.AST] = {}
    fixtures: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, str, bool]] = []
    shared: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            info = _fixture_decorator(stmt)
            if info is not None:
                fixtures.append((stmt, *info))
            elif stmt.name.startswith("pytest_"):
                shared |= deps_of(stmt)
            else:
                helpers[stmt.name] = stmt
        else:
            shared |= deps_of(stmt)

    def with_helpers(node: ast.AST) -> tuple[set[str], bool, list[ast.AST]]:
        """(deps incl. referenced helpers, True when a helper with deps is referenced, the
        node plus every helper it reaches)."""
        deps, seen, frontier, helper_deps = deps_of(node), set(), [node], False
        nodes: list[ast.AST] = [node]
        while frontier:
            for sub in ast.walk(frontier.pop()):
                if isinstance(sub, ast.Name) and sub.id in helpers and sub.id not in seen:
                    seen.add(sub.id)
                    extra = deps_of(helpers[sub.id])
                    helper_deps |= bool(extra)
                    deps |= extra
                    frontier.append(helpers[sub.id])
                    nodes.append(helpers[sub.id])
        return deps, helper_deps, nodes

    out: list[_Fixture] = []
    for fn, name, autouse in fixtures:
        deps, helper_deps, nodes = with_helpers(fn)
        params = {a.arg for a in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)}
        runtime, dynamic = _runtime_requests(nodes)
        params |= runtime
        own_strings = string_module_refs(LOADED_ONLY_RE.sub("None", ast.unparse(fn)), known)
        patch_only = (
            not helper_deps
            and not own_strings
            and _only_patched(fn, _bound_import_names(fn))
        )
        out.append(_Fixture(name, autouse, params, deps, patch_only, dynamic))
    return shared, out


@functools.lru_cache(maxsize=2048)
def _requests(text: str) -> tuple[set[str], bool]:
    """(fixture names a test file can request, True when a request is not static): every
    function parameter name plus every string constant split on commas/whitespace
    (usefixtures("x"), getfixturevalue("x"), parametrize(..., indirect=True) argnames)."""
    tree = ast_cached(text)
    names: set[str] = set()
    dynamic = False
    for sub in ast.walk(tree):
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = sub.args
            names.update(x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs))
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            names.update(re.split(r"[,\s]+", sub.value))
        elif (found := _request_call(sub)) is not None:
            names |= found[0]
            dynamic |= found[1]
    return names, dynamic


@functools.lru_cache(maxsize=64)
def _conftest_parts_cached(
    source: str, module: str, known: tuple[str, ...], bare: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, ...], tuple[_Fixture, ...]]:
    shared, fixtures = _conftest_parts(source, module, set(known), dict(bare))
    return tuple(sorted(shared)), tuple(fixtures)


def conftest_parts(
    source: str, module: str, known: AbstractSet[str], bare: Mapping[str, str]
) -> tuple[set[str], list[_Fixture]]:
    """`_conftest_parts`, memoised per process: it ast.unparses and re-parses every fixture,
    helper and hook of the conftest, and one plan calls it once per conftest (the 241-plan
    full-count test re-parsed the same two files 482 times). Keyed by the arguments, so an
    edited conftest (its source is in the key) is re-analysed. The returned fixtures are
    READ-ONLY: callers take copies of the containers, never mutate a fixture."""
    known_key = tuple(sorted(known))
    bare_key = tuple(sorted(bare.items()))
    shared, fixtures = _conftest_parts_cached(source, module, known_key, bare_key)
    return set(shared), list(fixtures)


@dataclass(frozen=True)
class ConftestIndex:
    """The per-TREE half of conftest_fixture_test_files: both are pure functions of the tree
    (the graph's path set, each conftest's bytes), so one plan need not rebuild them."""

    tests_in: tuple[str, ...]
    # (scope dir, the conftest's module-level/hook deps, its fixtures) per conftest
    conftests: tuple[tuple[str, tuple[str, ...], tuple[_Fixture, ...]], ...]


def conftest_index(
    worktree: Path,
    graph: Mapping[str, set[str]],
    snapshot: tuple[tuple[str, int, int], ...] | None = None,
) -> ConftestIndex:
    """Per-process cached `_build_conftest_index`, keyed by the plan's tree snapshot."""
    key = ("conftest", worktree.resolve(), _snapshot_or_walk(worktree, snapshot))
    if key not in _TREE_CACHE:
        _TREE_CACHE[key] = _build_conftest_index(worktree, graph)
    index: ConftestIndex = _TREE_CACHE[key]
    return index


def _build_conftest_index(worktree: Path, graph: Mapping[str, set[str]]) -> ConftestIndex:
    known = {_dotted(p) for p in graph}
    stems: dict[str, list[str]] = {}
    for rel in graph:
        if rel.startswith("tests/") and not rel.endswith("__init__.py"):
            stems.setdefault(Path(rel).stem, []).append(_dotted(rel))
    bare = {stem: mods[0] for stem, mods in stems.items() if len(mods) == 1}
    tests_in = tuple(sorted(p for p in graph if p.startswith("tests/unit/") and is_changed_test(p)))
    conftests: list[tuple[str, tuple[str, ...], tuple[_Fixture, ...]]] = []
    for rel in sorted(p for p in graph if p.startswith("tests/") and p.endswith("/conftest.py")):
        source = read_cached(worktree / rel)
        shared, fixtures = conftest_parts(source, _dotted(rel), known, bare)
        conftests.append((rel.removesuffix("conftest.py"), tuple(sorted(shared)), tuple(fixtures)))
    return ConftestIndex(tests_in, tuple(conftests))


def conftest_fixture_test_files(
    reached: AbstractSet[str],
    graph: Mapping[str, set[str]],
    worktree: Path,
    snapshot: tuple[tuple[str, int, int], ...] | None = None,
) -> list[str]:
    """Tests that reach the change only through a conftest FIXTURE (pytest loads conftests
    implicitly, so no test imports them). Review rev-gate-speed-2 F1 + chief round-3 model:

    - a reached tests/ helper the conftest imports, a reached module-level import or a
      reached pytest_* hook dependency: the conftest's whole subtree (FullSuite for
      tests/ or tests/unit/);
    - a reached non-autouse fixture (or one requesting it, transitively): the subtree's test
      files that request it by name; a non-static request in such a file: the subtree;
    - a module reached only through `sys.modules.get("...")` is no dependency (the fixture
      acts on it only in tests that loaded it, which import it: LOADED_ONLY_RE);
    - a reached autouse fixture that only patches attributes on the module (setattr with a
      constant name): no dependency - every selected test runs it, so a renamed or removed
      attribute fails them all (one test file of the subtree is kept selected for that);
    - any other reached autouse fixture: the subtree.

    The tree-derived half (the conftests' parsed fixtures, the per-tree `known`/`bare`
    tables, tests_in) comes from `conftest_index`, cached per snapshot: building it costs an
    ast.unparse per fixture/helper, and a plan asks the same two conftests every time."""
    index = conftest_index(worktree, graph, snapshot)
    if not index.conftests:
        return []
    tests_in = index.tests_in
    out: set[str] = set()

    def subtree(rel_dir: str, why: str) -> None:
        if rel_dir in ("tests/", "tests/unit/"):
            raise FullSuite(f"{rel_dir}conftest.py: {why}")
        out.update(p for p in tests_in if p.startswith(rel_dir))

    parts: list[tuple[str, _Fixture]] = []
    for rel_dir, shared, fixtures in index.conftests:
        all_deps = set(shared).union(*(f.deps for f in fixtures))
        if helper := sorted(m for m in all_deps & reached if m.startswith("tests.")):
            subtree(rel_dir, f"imports the changed helper {helper[0]}")
        if set(shared) & reached:
            subtree(rel_dir, f"module-level/hook import of {sorted(set(shared) & reached)[0]}")
        parts.extend((rel_dir, f) for f in fixtures)
    dependent: set[tuple[str, str]] = set()  # (scope dir, fixture name)
    keep_one: set[str] = set()
    grew = True
    while grew:
        grew = False
        for rel_dir, fx in parts:
            if (rel_dir, fx.name) in dependent:
                continue
            requests_dependent = any(
                name in fx.params and rel_dir.startswith(scope) for scope, name in dependent
            )
            if not (fx.deps & reached) and not requests_dependent:
                continue
            if fx.autouse:
                if fx.patch_only and not requests_dependent:
                    keep_one.add(rel_dir)
                    continue
                subtree(rel_dir, f"autouse fixture {fx.name} depends on the change")
            dependent.add((rel_dir, fx.name))
            grew = True
    for rel_dir, fx in parts:
        if fx.dynamic_request and any(rel_dir.startswith(scope) for scope, _ in dependent):
            subtree(rel_dir, f"fixture {fx.name} calls getfixturevalue(<non-constant>)")
    for path in tests_in:
        scopes = [(scope, name) for scope, name in dependent if path.startswith(scope)]
        if not scopes:
            continue
        names, dynamic = _requests(read_cached(worktree / path))
        if dynamic:
            subtree(scopes[0][0], f"{path} requests fixtures dynamically")
        if any(name in names for _, name in scopes):
            out.add(path)
    for rel_dir in keep_one:  # a patch-only autouse fixture still runs in one test
        if not any(p.startswith(rel_dir) for p in out):
            first = next((p for p in tests_in if p.startswith(rel_dir)), None)
            if first:
                out.add(first)
    return sorted(out)


TIMING_MARK_RE = re.compile(r"\bmark\.timing\b|[\"']timing[\"']")


def has_timing_mark(path: Path) -> bool:
    """True when a test file can carry a `timing` mark (`pytest.mark.timing` or the name in a
    string, e.g. `pytestmark = getattr(pytest.mark, "timing")`): grows, never shrinks."""
    return bool(TIMING_MARK_RE.search(path.read_text(encoding="utf-8", errors="replace")))


def _is_literal_mark(node: ast.expr) -> bool:
    """`mark.<name>` / `pytest.mark.<name>`, called or not, naming a mark other than timing."""
    if isinstance(node, ast.Call):
        node = node.func
    return (
        isinstance(node, ast.Attribute)
        and node.attr != TIMING_MARK
        and ast.unparse(node.value) in ("mark", "pytest.mark")
    )


def adds_marks_dynamically(source: str) -> bool:
    """True when conftest code may add a mark whose name only runtime knows: a getattr on a
    mark namespace, or add_marker(<arg>) with an arg that is neither a literal non-timing mark
    nor a name bound only to such marks (the NT preflight's `marker = pytest.mark.skip(...)`).
    Unparseable: True. Review rev-gate-1 B8."""
    if re.search(r"getattr\(\s*(?:pytest\.)?mark\b", source):
        return True
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return True
    bound: dict[str, list[ast.expr]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bound.setdefault(target.id, []).append(node.value)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_marker"):  # fmt: skip
            continue
        arg = node.args[0] if node.args else None
        named = bound.get(arg.id, []) if isinstance(arg, ast.Name) else []
        if arg is None or not (_is_literal_mark(arg) or (named and all(map(_is_literal_mark, named)))):
            return True
    return False


def timing_test_files(worktree: Path) -> list[str] | None:
    """Every tests/unit file has_timing_mark finds, for the full run's serial timing step (job
    20261120-board: `tests/unit -m timing` collected 11962 tests to run 10, 44.70 s). None
    (collect tests/unit) when a conftest under tests/ or at the root names the mark or adds
    marks dynamically (a collection hook could put it on any test), or a tests/unit file
    follows pytest's other `*_test.py` pattern, which this scan does not read. A read error
    raises (the caller collects)."""
    tests = worktree / "tests"
    if not tests.is_dir():
        return []
    conftests = [*sorted(tests.rglob("conftest.py")), worktree / "conftest.py"]
    for conftest in (c for c in conftests if c.is_file()):
        if has_timing_mark(conftest) or adds_marks_dynamically(read_cached(conftest, "replace")):
            return None
    root = tests / "unit"
    if not root.is_dir():
        return []
    if any(not p.name.startswith("test_") for p in root.rglob("*_test.py")):
        return None
    return [
        p.relative_to(worktree).as_posix() for p in sorted(root.rglob("test_*.py")) if has_timing_mark(p)
    ]


def sample_load() -> float | None:
    """The host's 1-minute load average; None when os.getloadavg() cannot read it (OSError)."""
    try:
        return os.getloadavg()[0]
    except OSError:
        return None


def load_line(load1: float | None) -> str:
    """The LOG's context line for the test step: the host load a slow run is judged against."""
    load = "load unknown" if load1 is None else f"load {load1:.2f} on {os.cpu_count() or 1} cpus"
    return f"{load} at test start"


def plan_tests(cfg: Config, state: GitState) -> TestPlan:
    """What the tests check runs. First, serially, the triggered POLICY_TESTS (fail fast: a red
    one skips the rest). Per round (default): ONE parallel call over the changed test
    files + the reverse import closure + the tests that reach a changed file
    by path/name, plus the files a `-k <stems>` can match (filtered by -k), with `timing`
    tests deselected; then a serial `-m timing` call over the changed test files only.
    --full (integration) or a full_suite_reason runs the whole tests/unit in parallel, then
    every `timing` test serially."""
    changed = reviewable_paths(state.changed)  # the bus's board/telemetry mirror files
    reason = "--full (integration mode)" if cfg.full else full_suite_reason(changed)
    selected: set[str] = set()
    reasons: dict[str, list[str]] = {}
    if not reason:
        try:
            selected = _selected_consumers(cfg.worktree, changed, reasons)
        except FullSuite as exc:
            reason = str(exc)
        except (GraphError, OSError) as exc:
            # an unreadable/unparseable file hides edges: grow to the full suite, never shrink
            reason = f"selection unknown ({exc})"
    if reason:
        # the cheap tree-scanning policy files first (fail fast: job 20261120-board's missing
        # exec bit surfaced after 4 min of suite), then the rest of tests/unit
        policy = policy_test_files(cfg.worktree)
        ignore = [f"--ignore={p}" for p in policy]
        try:
            timing = timing_test_files(cfg.worktree)
        except OSError:
            timing = None  # unknown: collect tests/unit
        timing_args = ["tests/unit"] if timing is None else timing  # [] = no timing test
        return TestPlan(
            [
                *([([*policy, "-m", f"not {TIMING_MARK}"], True, "no policy tests matched")] if policy else []),
                (["tests/unit", *ignore, *PYTEST_PARALLEL_ARGS, "-m", f"not {TIMING_MARK}"], False, ""),
                *([([*timing_args, "-m", TIMING_MARK], True, "no timing tests matched")] if timing_args else []),
            ],
            frozenset(), True, [f"full tests/unit: {reason}"], ["tests/unit"], fail_fast=int(bool(policy)),
        )
    targets = [p for p in changed_test_files(changed) if (cfg.worktree / p).exists()]
    # Repo-wide policy tests SCAN the tree instead of importing the changed module, so they
    # run on every per-round selection - even with NO python change (a docs/agent edit can
    # turn the word-budget test red): a gate PASS must never leave main red (POLICY_TESTS).
    # Every triggered one runs in the fail-fast first call only, also when it is a target or
    # an importer (review rev-gate-1 B7); its junit report still feeds slow-tests.
    policy = policy_test_files(cfg.worktree, changed)
    notes: list[str] = []
    importers = sorted(selected - set(targets) - set(policy))
    if importers:
        notes.append(f"{len(importers)} importer files")
    if policy:
        notes.append(f"{len(policy)} policy tests")
    explicit = [*targets, *(p for p in policy if p not in targets), *importers]
    selection = [*(t for t in targets if t not in policy), *importers]
    if policy:
        reasons["policy"] = policy
    # `-k <stem>` only for changed files with no module name (scripts/, tests/ helpers): a
    # vibe_quant module is covered by its import closure + path and string references, and
    # its stem mostly matched namesakes (`-k pipeline` ran test_screening_pipeline.py and
    # test_overfitting_pipeline.py for a discovery/pipeline.py change).
    expr = k_expr([p for p in changed if not graph_module_name(p)])
    k_files = k_candidate_files(cfg.worktree, expr, set(explicit)) if expr else []
    # ONE xdist call (one worker start-up, one collection) for both groups: `-k` also names
    # every selected file's module, which pytest matches against each of its tests, so the
    # selected files still run whole. A name -k cannot parse keeps the -k group separate.
    file_stems = [Path(p).stem for p in selection]
    merged = bool(k_files) and all(st.isidentifier() for st in file_stems)
    commands: list[tuple[list[str], bool, str]] = []
    if policy:  # serial: 7 files cost less than an xdist start-up
        commands.append(([*policy, "-m", f"not {TIMING_MARK}"], True, "no policy tests matched"))
    if selection or merged:
        args = [*selection, *k_files, "-k", " or ".join([expr, *file_stems])] if merged else selection
        commands.append(
            # rc5 is a real failure when a changed test file is in the call (it collected
            # nothing), a SKIP note otherwise
            ([*args, *PYTEST_PARALLEL_ARGS, "-m", f"not {TIMING_MARK}"],
             not any(t in selection for t in targets),
             f"no tests matched -k {expr} in the selected files" if merged
             else "no non-timing tests in the selected files")
        )
    if k_files and not merged:
        commands.append(
            ([*k_files, "-k", expr, *PYTEST_PARALLEL_ARGS, "-m", f"not {TIMING_MARK}"], True,
             f"no tests matched -k {expr}")
        )
    if timing_targets := [t for t in targets if has_timing_mark(cfg.worktree / t)]:
        # a changed test file runs whole: its relative-speed tests too, uncontended (only
        # when it has one: the call otherwise collects for ~1 s and exits 5)
        commands.append(
            ([*timing_targets, "-m", TIMING_MARK], True,
             "no timing tests in the changed test files")
        )
    return TestPlan(
        commands, frozenset(explicit), False, notes, [*explicit, *k_files], reasons,
        fail_fast=int(bool(policy)),
    )


@dataclass(frozen=True)
class ChangedTest:
    """One test function the diff added or changed, with the slow mark it carries.

    `cls` is the enclosing class ("" at module level), `name` the function name without a
    parametrization id, `reason` the `@pytest.mark.slow(reason=...)` value ("" when the mark
    has no usable reason, None when the test carries no slow mark at all).
    """

    file: str
    cls: str
    name: str
    reason: str | None


HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _slow_mark(node: ast.expr, mark: str = SLOW_TEST_MARK) -> str | None:
    """The `reason=` of a `pytest.mark.<mark>` decorator: None when it is not that mark, "" when
    the mark carries no reason (or a non-literal one, which is no reason at all)."""
    call = node if isinstance(node, ast.Call) else None
    target = call.func if call else node
    if not (
        isinstance(target, ast.Attribute)
        and target.attr == mark
        and isinstance(target.value, ast.Attribute)
        and target.value.attr == "mark"
    ):
        return None
    for kw in call.keywords if call else ():
        if kw.arg == "reason" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return ""


def _slow_reason(marks: Sequence[str | None]) -> str | None:
    """One slow mark out of a test's own and its class's: a non-empty reason wins over a bare
    mark, None means unmarked."""
    return max((m for m in marks if m is not None), key=bool, default=None)


def _test_functions(
    tree: ast.Module,
) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, str | None]]:
    """(enclosing class, node, slow reason) for every module-level and `Test*` class test."""
    found: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, str | None]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            found.append(("", node, _slow_reason([_slow_mark(d) for d in node.decorator_list])))
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            class_marks = [_slow_mark(d) for d in node.decorator_list]
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test_"):
                    marks = [_slow_mark(d) for d in child.decorator_list]
                    found.append((node.name, child, _slow_reason([*marks, *class_marks])))
    return found


_Def = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def _header_start(node: _Def) -> int:
    """First line of a def/class header: its first decorator, else the `def`/`class` line."""
    return min([node.lineno, *(d.lineno for d in node.decorator_list)])


def _parse(source: str) -> ast.Module | None:
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _lost_a_decorator(base_tree: ast.Module, hunk_removed: AbstractSet[int], all_removed: AbstractSet[int]) -> bool:
    """True when a BASE line this hunk removed sat in a decorator of a def/class whose own
    `def`/`class` line was NOT removed (anywhere in the file's diff): a decorator (e.g.
    `@pytest.mark.slow`) came off a test that still exists. A removed whole test takes its
    decorators and its `def` together and is not that."""
    for node in ast.walk(base_tree):
        if isinstance(node, _Def) and node.lineno not in all_removed:
            for dec in node.decorator_list:
                if any(n in hunk_removed for n in range(dec.lineno, (dec.end_lineno or dec.lineno) + 1)):
                    return True
    return False


def _surviving_header(head_lines: Sequence[str], head_tree: ast.Module, pos: int) -> int | None:
    """Header start of the first def/class whose `def`/`class` line is at or after head line
    `pos`, when only blank and comment lines lie between `pos` and that header; None otherwise."""
    nodes = sorted(
        (n for n in ast.walk(head_tree) if isinstance(n, _Def) and n.lineno >= pos),
        key=_header_start,
    )
    if not nodes:
        return None
    start = _header_start(nodes[0])
    trivia = head_lines[pos - 1 : max(start - 1, pos - 1)]
    if all(not t.strip() or t.lstrip().startswith("#") for t in trivia):
        return start
    return None


def _changed_test_lines(worktree: Path, base: str, run: RunFn) -> dict[str, set[int]]:
    """{test file: head line numbers the diff added or changed} against the MERGE BASE, so a
    base-side move of a line is not mistaken for a change of ours. Uncommitted edits count:
    the gate gates the worktree, not HEAD.

    A hunk that removed a decorator of a def/class that still exists also marks that survivor's
    header: the first def/class (decorators included) at or after the hunk's head position,
    across blank and comment lines. The hunk itself sits above the decorator span, which no
    test covers, so without this the test that lost its mark is never judged."""
    merge_base = _git_out(worktree, run, "merge-base", base, "HEAD") or base
    out = _git_out(worktree, run, "diff", "-U0", "--no-renames", merge_base, "--", "tests/")
    lines: dict[str, set[int]] = {}
    hunks: dict[str, list[tuple[int, int, int, int]]] = {}  # (old start, old count, head start, head count)
    current = ""
    for line in out.splitlines():
        if line.startswith(("diff --git ", "+++ b/")):
            if line.startswith("+++ b/"):
                current = line[len("+++ b/"):]
                lines.setdefault(current, set())
            continue
        match = HUNK_RE.match(line)
        if match and current:
            old_start = int(match.group(1))
            old_count = int(match.group(2) or 1)
            start = int(match.group(3))
            count = int(match.group(4) or 1)
            # a deletion-only hunk (count 0) sits right after head line `start`: that line's
            # test lost lines, so it is changed
            lines[current].update(range(start, start + count) if count else [start])
            hunks.setdefault(current, []).append((old_start, old_count, start, count))
    for path, spans in hunks.items():
        removed = [set(range(o, o + oc)) for o, oc, _, _ in spans]
        all_removed = set().union(*removed)
        if not all_removed:
            continue
        base_tree = _parse(_git_out(worktree, run, "show", f"{merge_base}:{path}"))
        source = worktree / path
        head_text = source.read_text(encoding="utf-8", errors="replace") if source.is_file() else ""
        head_tree = _parse(head_text)
        if base_tree is None or head_tree is None:
            continue
        head_lines = head_text.splitlines()
        for (_, _, start, count), hunk_removed in zip(spans, removed, strict=True):
            if not hunk_removed or not _lost_a_decorator(base_tree, hunk_removed, all_removed):
                continue
            # removed lines sit at head position `start` (before this hunk's added lines) or,
            # for a deletion-only hunk, right after head line `start`
            for pos in {start + 1} if not count else {start, start + count}:
                if (header := _surviving_header(head_lines, head_tree, pos)) is not None:
                    lines[path].add(header)
    return lines


def _spans(lines: AbstractSet[int], node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> bool:
    """True when a diff line falls inside the test (its decorators included)."""
    if not lines:
        return False
    decorators = getattr(node, "decorator_list", ())
    start = min([node.lineno, *(d.lineno for d in decorators)])
    end = getattr(node, "end_lineno", None) or node.lineno
    return any(start <= line <= end for line in lines)


def _class_headers(tree: ast.Module) -> dict[str, tuple[int, int]]:
    """{`Test*` class name: (first decorator line, `class` line)}: a mark on the class is a mark
    on every method, so touching it changes them all."""
    return {
        node.name: (min([node.lineno, *(d.lineno for d in node.decorator_list)]), node.lineno)
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test")
    }


def changed_test_functions(cfg: Config, state: GitState, run: RunFn) -> list[ChangedTest]:
    """The test functions the diff added or changed, with the slow mark each carries.

    A new or untracked test file contributes ALL its tests (there is no base body to compare).
    An unchanged slow test is existing debt and is never returned: only the function whose
    body, decorators or line range the diff touched is judged.
    """
    paths = changed_test_files(reviewable_paths(state.changed))
    if not paths:
        return []
    hunks = _changed_test_lines(cfg.worktree, cfg.base, run)
    found: list[ChangedTest] = []
    for path in paths:
        source = cfg.worktree / path
        if not source.is_file():
            continue
        touched = hunks.get(path)
        whole = path in state.untracked or touched is None
        if not whole and not touched:
            continue  # e.g. a mode-only change: no test body moved
        try:
            tree = ast.parse(source.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue  # a file that does not parse already fails the tests check
        headers = _class_headers(tree)
        for cls, node, reason in _test_functions(tree):
            header = headers.get(cls)
            header_hit = header is not None and any(header[0] <= n <= header[1] for n in touched or ())
            if whole or header_hit or _spans(touched or set(), node):
                found.append(ChangedTest(path, cls, node.name, reason))
    return found


def _junit_cases(paths: Sequence[Path]) -> list[tuple[str, str, float]]:
    """(classname, test name, seconds) for every testcase in these junit files. A missing or
    unparseable file, or a testcase without a finite non-negative `time`, raises ValueError:
    timing evidence that cannot be read is never read as "fast"."""
    found: list[tuple[str, str, float]] = []
    for path in paths:
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError) as exc:
            raise ValueError(f"no usable junit report ({path.name}: {type(exc).__name__})") from exc
        for case in root.iter("testcase"):
            try:
                seconds = float(case.get("time", ""))
            except ValueError:
                seconds = math.nan
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError(f"junit testcase {case.get('name')} has no valid time")
            found.append((case.get("classname") or "", case.get("name") or "", seconds))
    return found


def _module_of(path: str) -> str:
    return path[: -len(".py")].replace("/", ".") if path.endswith(".py") else path


def _match_changed_test(modules: Mapping[str, str], classname: str) -> tuple[str, str] | None:
    """(changed file, class path) a junit testcase belongs to, or None for an unjudged test:
    its classname is matched against the CHANGED test files' dotted modules (pytest's default
    xunit2 report carries no `file` attribute)."""
    for module, path in modules.items():
        if classname == module:
            return path, ""
        if classname.startswith(f"{module}."):
            return path, classname[len(module) + 1:]
    return None


def _alone_seconds(
    cfg: Config,
    tools: dict[str, list[str]],
    run: RunFn,
    test_env: dict[str, str],
    node: str,
    xml: Path,
) -> tuple[float | None, str]:
    """(seconds, "") the node id took when re-run alone and serially (`junit time`, so pytest's
    start-up is not counted); (None, why) when that re-run failed or did not report the test."""
    env = dict(test_env)
    env["PYTEST_ADDOPTS"] = f"{env.get('PYTEST_ADDOPTS', '')} {JUNIT_OPT}{xml}".strip()
    rc, out = run([*tools["pytest"], node, *SLOW_ALONE_ARGS], cfg.worktree, env, SLOW_RERUN_TIMEOUT)
    if rc != 0:
        return None, f"the serial re-run exited {rc}: {(out.strip().splitlines() or [''])[-1][:120]}"
    try:
        cases = _junit_cases([xml])
    except ValueError as exc:
        return None, f"the serial re-run left {exc}"
    times = [seconds for _, name, seconds in cases if name == node.rsplit("::", 1)[-1]]
    return (max(times), "") if times else (None, "the serial re-run's junit lacks the test")


def _id_list(ids: Sequence[str]) -> str:
    return ", ".join(ids[:5]) + (f" (+{len(ids) - 5} more)" if len(ids) > 5 else "")


def slow_tests_check(
    cfg: Config,
    state: GitState,
    tools: dict[str, list[str]],
    run: RunFn,
    test_env: dict[str, str],
    junits: Sequence[Path],
) -> Check:
    """FAIL a NEW or CHANGED unit test that takes > SLOW_TEST_SECONDS alone, unless it carries
    `@pytest.mark.slow(reason=...)`.

    The gate run's junit times say which changed tests are slow under xdist; the slowest are
    then re-run ALONE and serially (at most SLOW_RERUN_CAP, stopping at the first failure),
    because a test that only crosses the line under worker contention is load noise. Fails
    closed: a changed test with no timing (no/unreadable junit, never mapped) or a re-run that
    did not pass is a FAIL. Unchanged slow tests are existing debt, ignored by construction
    (changed_test_functions).
    """
    changed = changed_test_functions(cfg, state, run)
    if not changed:
        return Check("slow-tests", "SKIP", "no python test changed")
    try:
        cases = _junit_cases(junits)
    except ValueError as exc:
        return Check("slow-tests", "FAIL", f"{len(changed)} changed tests have no timing: {exc}")
    wanted = {(test.file, test.cls, test.name): test for test in changed}
    modules = {_module_of(test.file): test.file for test in changed}
    seen: set[tuple[str, str, str]] = set()
    over: dict[str, tuple[ChangedTest, float]] = {}  # node id -> test, its slowest gate time
    for classname, name, seconds in cases:
        match = _match_changed_test(modules, classname)
        if match is None:
            continue
        key = (match[0], match[1].split(".")[-1] if match[1] else "", name.split("[")[0])
        if (test := wanted.get(key)) is None:
            continue
        seen.add(key)
        if seconds > SLOW_TEST_SECONDS:
            node = f"{test.file}::{test.cls + '::' if test.cls else ''}{name}"
            over[node] = (test, max(seconds, over[node][1] if node in over else 0.0))
    unseen = [f"{test.file}::{test.name}" for key, test in wanted.items() if key not in seen]
    if unseen:
        return Check(
            "slow-tests", "FAIL", f"no timing for changed tests (not run or not mapped): {_id_list(unseen)}"
        )
    ordered = sorted(over.items(), key=lambda item: -item[1][1])  # slowest first
    marked = [f"{node} ({secs:g}s)" for node, (test, secs) in ordered if test.reason]
    todo = [(node, test, secs) for node, (test, secs) in ordered if not test.reason]
    noise: list[str] = []
    failure = ""
    done = 0
    with tempfile.TemporaryDirectory(prefix="swarm-check-slow-") as tmp:
        for node, test, secs in todo[:SLOW_RERUN_CAP]:
            done += 1
            alone, why = _alone_seconds(cfg, tools, run, test_env, node, Path(tmp) / f"alone-{done}.xml")
            if alone is None:
                failure = f"{node} {secs:g}s in the gate; {why}"
            elif alone > SLOW_TEST_SECONDS:
                fix = (
                    f"mark it {SLOW_TEST_HINT} or make it faster"
                    if test.reason is None
                    else 'the slow mark needs a reason: @pytest.mark.slow(reason="why it is slow")'
                )
                failure = f"{node} {secs:g}s in the gate, {alone:g}s alone; {fix}"
            else:
                noise.append(f"{node} {secs:g}s in the gate, {alone:g}s alone")
            if failure:
                break
    unverified = [node for node, _, _ in todo[done:]]
    if failure or unverified:
        parts = [failure] if failure else []
        if unverified:
            parts.append(
                f"{_id_list(unverified)} > {SLOW_TEST_SECONDS:g} s in gate run, not re-run (cap {SLOW_RERUN_CAP})"
            )
        return Check("slow-tests", "FAIL", " | ".join(parts))
    if not marked and not noise:
        return Check("slow-tests", "PASS", f"{len(changed)} changed tests, none over {SLOW_TEST_SECONDS:g}s")
    parts = [f"slow-marked: {', '.join(marked)}"] if marked else []
    if noise:
        parts.append("load noise, not slow: " + ", ".join(noise))
    return Check("slow-tests", "PASS", " | ".join(parts))


def _new_unmarked_tests(cfg: Config, state: GitState, run: RunFn) -> list[str]:
    """Node ids of the test functions the diff ADDED (absent from the merge base's file) that
    carry no `simple_target` mark with a reason (on the test or its class)."""
    merge_base = _git_out(cfg.worktree, run, "merge-base", cfg.base, "HEAD") or cfg.base
    by_file: dict[str, list[ChangedTest]] = {}
    for test in changed_test_functions(cfg, state, run):
        by_file.setdefault(test.file, []).append(test)
    nodes: list[str] = []
    for path, tests in by_file.items():
        base_tree = _parse(_git_out(cfg.worktree, run, "show", f"{merge_base}:{path}"))
        old = {(cls, node.name) for cls, node, _ in _test_functions(base_tree)} if base_tree else set()
        head = _parse((cfg.worktree / path).read_text(encoding="utf-8", errors="replace"))
        marks: dict[tuple[str, str], bool] = {}
        for node in head.body if head else ():
            funcs = node.body if isinstance(node, ast.ClassDef) else [node]
            cls_marks = node.decorator_list if isinstance(node, ast.ClassDef) else []
            for func in funcs:
                if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    decorators = [*func.decorator_list, *cls_marks]
                    key = (node.name if isinstance(node, ast.ClassDef) else "", func.name)
                    marks[key] = any(_slow_mark(d, SIMPLE_TEST_MARK) for d in decorators)
        for test in tests:
            if (test.cls, test.name) not in old and not marks.get((test.cls, test.name)):
                nodes.append(f"{path}::{test.cls + '::' if test.cls else ''}{test.name}")
    return nodes


def _function_spans(path: Path) -> list[tuple[int, int, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """(first body line, last line, node) for every function in the file, nested ones too."""
    tree = _parse(path.read_text(encoding="utf-8", errors="replace"))
    spans: list[tuple[int, int, ast.FunctionDef | ast.AsyncFunctionDef]] = []
    for node in ast.walk(tree) if tree else ():
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.append((node.body[0].lineno, node.end_lineno or node.lineno, node))
    return spans


def _covered_by_test(db_path: Path, roots: Sequence[Path]) -> dict[str, dict[str, set[int]]]:
    """{test node id (no parametrization): {repo path: lines its run phase executed}}."""
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        files: dict[int, str] = {}
        for fid, raw in db.execute("select id, path from file"):
            resolved = Path(raw).resolve()
            for root in roots:
                with contextlib.suppress(ValueError):
                    files[fid] = resolved.relative_to(root.resolve()).as_posix()
                    break
        contexts = dict(db.execute("select id, context from context"))
        tables = {name for (name,) in db.execute("select name from sqlite_master where type='table'")}
        rows: list[tuple[int, int, int]] = []
        if "arc" in tables:
            for fid, cid, a, b in db.execute("select file_id, context_id, fromno, tono from arc"):
                rows.extend((fid, cid, n) for n in (a, b) if n > 0)
        if "line_bits" in tables:
            # coverage.numbits_to_nums, inlined: the gate runs under the system python3,
            # which has no `coverage` (bit i of byte j = line 8*j + i)
            for fid, cid, bits in db.execute("select file_id, context_id, numbits from line_bits"):
                rows.extend(
                    (fid, cid, 8 * j + i) for j, byte in enumerate(bits) for i in range(8) if byte >> i & 1
                )
    finally:
        db.close()
    out: dict[str, dict[str, set[int]]] = {}
    for fid, cid, line in rows:
        ctx = contexts.get(cid, "")
        if not ctx.endswith("|run") or fid not in files:
            continue
        node = ctx[: -len("|run")].split("[", 1)[0]
        out.setdefault(node, {}).setdefault(files[fid], set()).add(line)
    return out


def simple_tests_check(
    cfg: Config, state: GitState, tools: dict[str, list[str]], run: RunFn, test_env: dict[str, str]
) -> Check:
    """FAIL a NEW test whose run touches only simple, non-exempt functions (see
    SIMPLE_MAX_COGNITIVE) unless it carries `@pytest.mark.simple_target(reason=...)`.

    The new tests are run once more, alone, under per-test coverage; the functions their run
    phase executed are scored with cccc. A test that executes no vibe_quant/ or scripts/
    function is not judged. Fails closed: a failed run, missing coverage data or a cccc
    failure is a FAIL, never a PASS."""
    if state.error:
        return Check("simple-tests", "SKIP", "git diff unknown (see scope)")
    nodes = _new_unmarked_tests(cfg, state, run)
    if not nodes:
        return Check("simple-tests", "SKIP", "no new unmarked test")
    with tempfile.TemporaryDirectory(prefix="swarm-check-simple-") as tmp:
        env = dict(test_env)
        env["COVERAGE_FILE"] = str(Path(tmp) / ".coverage")
        args = ["-p", "no:randomly", "--cov=vibe_quant", "--cov=scripts", "--cov-context=test",
                "--cov-report=", "-n", "0"]  # fmt: skip
        rc, out = run([*tools["pytest"], *nodes, *args], cfg.worktree, env, TEST_TIMEOUT)
        if rc != 0:
            return Check("simple-tests", "FAIL", f"the coverage run of {len(nodes)} new tests exited {rc}: {tail_line(out)}")
        try:
            covered = _covered_by_test(Path(tmp) / ".coverage", [cfg.worktree, cfg.main_root])
        except sqlite3.Error as exc:
            return Check("simple-tests", "FAIL", f"no coverage data for the new tests: {exc}")
    files = sorted({f for per in covered.values() for f in per if f.endswith(".py")})
    if not files:
        return Check("simple-tests", "PASS", f"{len(nodes)} new tests, none executes product code")
    rc, out = run([str(cfg.worktree / CCCC_SCRIPT), *files], cfg.worktree, None, TOOL_TIMEOUT)
    try:
        scored = {
            (f["path"], g["name"], g["line"]): g["cognitive"]
            for f in json.loads(out if rc == 0 else "")["files"]
            for g in f["functions"]
        }
    except (ValueError, KeyError, TypeError):
        return Check("simple-tests", "FAIL", f"cccc failed (exit {rc}): {tail_line(out)}")
    simple: dict[tuple[str, int], str] = {}  # (file, body line) -> "file:name cog=N", "" when not simple
    for rel in files:
        source = cfg.worktree / rel
        if not source.is_file():
            continue
        text_lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
        for start, end, node in _function_spans(source):
            body = "\n".join(text_lines[node.lineno - 1 : end])
            cog = next((c for (p, n, ln), c in scored.items() if p == rel and n == node.name
                        and abs(ln - node.lineno) <= len(node.decorator_list) + 1), None)  # fmt: skip
            exempt = (
                cog is None
                or cog > SIMPLE_MAX_COGNITIVE
                or SIMPLE_EXEMPT_PATH.search(rel)
                or SIMPLE_EXEMPT_NAME.search(node.name)
                or SIMPLE_EXEMPT_SQL.search(body)
                or SIMPLE_EXEMPT_REGEX.search(body)
            )
            for line in range(start, end + 1):
                simple[(rel, line)] = "" if exempt else f"{rel}:{node.name} cog={cog}"
    flagged: list[str] = []
    for node_id in nodes:
        hit = [simple.get((f, n)) for f, lines in covered.get(node_id, {}).items() for n in lines]
        funcs = [h for h in hit if h is not None]  # None: a module-level line, not judged
        if funcs and all(funcs):
            flagged.append(f"{node_id} ({', '.join(sorted(set(funcs))[:3])})")
    if flagged:
        return Check(
            "simple-tests",
            "FAIL",
            f"{_id_list(flagged)} only touch simple code: delete and cover through the caller, "
            f"or mark {SIMPLE_TEST_HINT}",
        )
    return Check("simple-tests", "PASS", f"{len(nodes)} new tests, none only on simple code")


def _run_tests(
    cfg: Config, state: GitState, tools: dict[str, list[str]], run: RunFn, wt_env: dict[str, str]
) -> TestRun:
    plan = plan_tests(cfg, state)
    test_env = dict(wt_env)
    test_env["PYTEST_ADDOPTS"] = f"{wt_env.get('PYTEST_ADDOPTS', '')} {PYTEST_QUIET_ADDOPTS}".strip()
    if not plan.commands:
        python_changed = any(p.endswith(".py") for p in state.changed)
        why = "nothing to run for changed python" if python_changed else "no python changed"
        slow = slow_tests_check(cfg, state, tools, run, test_env, [])
        return TestRun(Check("tests", "SKIP", why), plan.explicit, plan.full, slow)
    summaries: list[str] = []
    failed = False
    stopped = False  # a red fail-fast (policy) call ended the step: the selection never ran
    passed = 0
    cache = _file_cache_plan(cfg, state, plan, run)
    reused = cache.reused
    if isinstance(run, ToolLog):
        for test, tree in sorted(reused.items()):
            run.note(f"reused {test} (passed at tree {tree[:12]})")
    commands: list[tuple[list[str], bool, str, bool]] = []
    for index, (args, rc5_skip, rc5_note) in enumerate(plan.commands):
        kept = [a for a in args if a not in reused]
        if reused and not any(a.startswith("tests/") for a in kept):
            continue  # every file of this call PASSed at a tree the diff since cannot reach
        commands.append((kept, rc5_skip, rc5_note, index < plan.fail_fast))
    with contextlib.ExitStack() as slot, tempfile.TemporaryDirectory(prefix="swarm-check-junit-") as tmp:
        load1 = sample_load()  # ONE reading: the LOG note states what the sizing used
        if cfg.slots_dir is not None:
            # Decided now, not at gate start: other gates may have started (or ended) since.
            # An explicit PYTEST_XDIST_AUTO_NUM_WORKERS keeps its value but still holds a slot,
            # so peers count this gate.
            running = slot.enter_context(_gate_slots_module().gate_slot(cfg.slots_dir, TEST_TIMEOUT))
            workers = xdist_auto_workers(dict(os.environ), load1, os.cpu_count() or 1, running)
            if workers is not None and "PYTEST_XDIST_AUTO_NUM_WORKERS" not in os.environ:
                test_env["PYTEST_XDIST_AUTO_NUM_WORKERS"] = workers
                print(f"gate: {running} gate(s) running, xdist auto = {workers} workers", file=sys.stderr)
        if isinstance(run, ToolLog):  # the LOG says what load a slow run ran under
            run.note(load_line(load1))
        junits: list[Path] = []
        records: list[Path] = []  # per call holding cacheable files: its deselection record
        if cache.cacheable:
            (Path(tmp) / f"{DESELECT_PLUGIN}.py").write_text(DESELECT_PLUGIN_SOURCE, encoding="utf-8")
        for index, (args, rc5_skip, rc5_note, fail_fast) in enumerate(commands):
            xml = Path(tmp) / f"run-{index}.xml"
            junits.append(xml)
            env = dict(test_env)
            env["PYTEST_ADDOPTS"] = f"{test_env['PYTEST_ADDOPTS']} {JUNIT_OPT}{xml}".strip()
            plugin: list[str] = []
            if any(a in cache.cacheable for a in args):
                records.append(record := Path(tmp) / f"deselected-{index}")
                record.mkdir()
                # argv, not PYTEST_ADDOPTS: a pytest that a test spawns inherits the env and
                # exited 3 on the plugin it could not import (test_slow_test_guard.py)
                plugin = ["-p", DESELECT_PLUGIN]
                env["PYTHONPATH"] = os.pathsep.join(filter(None, [env.get("PYTHONPATH"), tmp]))
                env[DESELECT_ENV] = str(record)
            rc, out = run([*tools["pytest"], *plugin, *args], cfg.worktree, env, TEST_TIMEOUT)
            if rc == PYTEST_NO_TESTS_RC and rc5_skip:
                summaries.append(rc5_note)
                continue
            summaries.append(pytest_summary(out))
            if rc != 0:
                failed = True
                summaries.extend(failed_tests(out))
                if fail_fast:
                    summaries.append(POLICY_STOP)
                    stopped = True
                    break
            else:
                passed += 1
        slot.close()  # the slow-test re-runs are serial (-n0): they hold no share of the cores
        slow = (
            Check("slow-tests", "SKIP", "policy tests failed: changed tests not run")
            if stopped
            else slow_tests_check(cfg, state, tools, run, test_env, junits)
        )
        if not failed and cache.cacheable and (dropped := deselected_files(records)) is not None:
            passed_files = junit_passed_files(junits, cache.cacheable)
            _store_file_passes(cfg, run, cache, [f for f in passed_files if f not in dropped])
    summaries.extend(plan.notes)  # what the selection was made of, or why the full suite ran
    if reused:
        summaries.append(f"cached: {len(reused)} files skipped")
        if isinstance(run, ToolLog):
            run.note(f"cached: {len(reused)} files skipped")
        passed += not failed
    if failed:
        check = Check("tests", "FAIL", " | ".join(summaries))
    elif passed == 0:
        check = Check("tests", "SKIP", " | ".join(summaries))
    else:
        check = Check("tests", "PASS", " | ".join(summaries))
    return TestRun(check, plan.explicit, plan.full, slow, stopped)


@dataclass(frozen=True)
class FileCacheRun:
    """The per-file cache's pre-run decision: the files whose PASS may be stored, the files
    reused ({file: tree}), and the identity a store must still match after the tests: env
    hash and (HEAD sha, HEAD^{tree}, HEAD reflog length) (review rev-gate-1 B1/B3)."""

    cacheable: list[str] = field(default_factory=list)
    reused: dict[str, str] = field(default_factory=dict)
    env: str = ""
    head: tuple[str, str, int] | None = None


def _head_identity(cfg: Config, run: RunFn) -> tuple[str, str, int] | None:
    """(HEAD sha, HEAD^{tree}, HEAD reflog length); None when git cannot say. The reflog
    sees a switch away and back while the tests ran, as for the stamp (head_reflog_count)."""
    sha = _git_out(cfg.worktree, run, "rev-parse", "HEAD")
    tree = _git_out(cfg.worktree, run, "rev-parse", "HEAD^{tree}")
    reflog = head_reflog_count(cfg.worktree, run)
    if reflog is None or not TREE_SHA_RE.fullmatch(sha) or not TREE_SHA_RE.fullmatch(tree):
        return None
    return sha, tree, reflog


def _file_cache_plan(cfg: Config, state: GitState, plan: TestPlan, run: RunFn) -> FileCacheRun:
    """The per-file cache's pre-run decision (FileCacheRun).

    Off for --full/a whole-suite plan, a dirty tree, active stubs, no cache dir, an unknown env
    or HEAD. Never cached: a test file in the diff, a file with a timing mark (its main run is
    partial), a file naming external data (EXTERNAL_DATA_RE), a file pinning a literal the diff
    removed (pinned-text must see it run), a -k candidate."""
    if cfg.file_cache_dir is None or plan.full or not is_clean(state) or active_stubs():
        return FileCacheRun()
    try:
        env = file_cache_env(cfg.main_root)
        head = _head_identity(cfg, run)
        if env is None or head is None:
            return FileCacheRun()
        changed_py = [p for p in state.changed if p.startswith("vibe_quant/") and p.endswith(".py")]
        removed = removed_base_literals(cfg, changed_py, run) if changed_py else []
        pinned = {path for path, _, _ in pinned_test_pins(cfg.worktree, removed)}
        in_diff = set(changed_test_files(state.changed))
        cacheable = []
        for f in sorted(plan.explicit):
            text = (cfg.worktree / f).read_text(encoding="utf-8", errors="replace")
            if f in in_diff or f in pinned or TIMING_MARK_RE.search(text) or EXTERNAL_DATA_RE.search(text):
                continue
            cacheable.append(f)
        return FileCacheRun(cacheable, reusable_test_files(cfg, run, cacheable, head[1], env), env, head)
    except Exception:  # fail closed: no reuse, no store
        return FileCacheRun()


def _store_file_passes(cfg: Config, run: RunFn, cache: FileCacheRun, passed: Sequence[str]) -> None:
    """Store the PASSes only if the env, HEAD, tree and HEAD reflog are what they were before
    the tests and the tree is clean (nothing moved while they ran); a write error never fails
    the gate (it only costs a later re-run)."""
    if cfg.file_cache_dir is None or cache.head is None or not passed:
        return
    with contextlib.suppress(Exception):
        if (
            _head_identity(cfg, run) == cache.head
            and file_cache_env(cfg.main_root) == cache.env
            and is_clean(git_state(cfg.worktree, cfg.base, run))
        ):
            write_file_passes(cfg.file_cache_dir, cache.env, cache.head[1], passed)


def _run_exactness(cfg: Config, tools: dict[str, list[str]], run: RunFn) -> Check:
    main_script = cfg.main_root / EXACTNESS_SCRIPT
    wt_script = cfg.worktree / EXACTNESS_SCRIPT
    if not main_script.is_file():
        return Check("exactness", "FAIL", f"main exactness script missing: {main_script}")
    if wt_script.is_file() and wt_script.read_bytes() != main_script.read_bytes():
        return Check("exactness", "FAIL", "exactness script modified in worktree")
    expect: list[str] = []
    if cfg.expect_pins:
        if not Path(cfg.expect_pins).is_file():
            return Check("exactness", "FAIL", f"declared pins file missing: {cfg.expect_pins}")
        expect = ["--expect", cfg.expect_pins]
    rc, out = run(
        tools["python"] + [str(main_script), *expect],
        cfg.worktree,
        {"PYTHONPATH": str(cfg.worktree)},
        TEST_TIMEOUT,
    )
    if rc == 0 and "IDENTICAL" in out:
        return Check("exactness", "PASS", f"IDENTICAL to declared pins {cfg.expect_pins}" if expect else "IDENTICAL")
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


def _exactness_check(
    cfg: Config, state: GitState, tools: dict[str, list[str]], run: RunFn
) -> Check:
    if cfg.exactness == "never":
        return Check("exactness", "SKIP", "disabled")
    if cfg.exactness == "always" or (not state.error and should_run_exactness(state.changed)):
        return _run_exactness(cfg, tools, run)
    if state.error:
        return Check("exactness", "SKIP", "git diff unknown (see scope)")
    return Check(
        "exactness",
        "SKIP",
        "auto: nothing changed under vibe_quant/ (except "
        + ", ".join(d.rstrip("/").removeprefix("vibe_quant/") for d in EXACTNESS_SAFE_DIRS)
        + ") nor in " + ", ".join(EXACTNESS_FILES),
    )


def _exactness_carry_check(
    cfg: Config, state: GitState, tools: dict[str, list[str]], run: RunFn
) -> Check:
    """The same flag governs both proofs; a main without the script is a SKIP so this check
    can land before the carry engine does."""
    if cfg.exactness == "never":
        return Check("exactness-carry", "SKIP", "disabled")
    if cfg.exactness == "always" or (
        not state.error and should_run_exactness_carry(state.changed)
    ):
        return _run_exactness_carry(cfg, tools, run)
    if state.error:
        return Check("exactness-carry", "SKIP", "git diff unknown (see scope)")
    return Check(
        "exactness-carry", "SKIP", "auto: nothing changed in " + ", ".join(EXACTNESS_CARRY_TRIGGERS)
    )


def _golden_skip(cfg: Config) -> str:
    """Why the golden proofs are skipped, "" when they run: --full, or --exactness always."""
    if cfg.exactness == "never":
        return "disabled"
    if cfg.full or cfg.exactness == "always":
        return ""
    return GOLDEN_SKIP_PER_ROUND


def _golden_path(cfg: Config, name: str) -> Path:
    """The MAIN checkout's golden: `<main root>/data/swarm/_goldens/<name>` (outside git)."""
    return cfg.main_root.joinpath(*GOLDEN_SUBDIR, name)


def _script_changed_in_diff(cfg: Config, script: str) -> bool:
    """True when the worktree's copy exists and main has a different one (new or edited here).

    Informational only (a reviewer reads it), never a FAIL: these two checks run the WORKTREE's
    script on purpose - it is the code under test - unlike exactness_239/carry, where a worktree
    copy may never differ from main's.
    """
    wt_script = cfg.worktree / script
    main_script = cfg.main_root / script
    if not wt_script.is_file():
        return False
    try:
        return not main_script.is_file() or wt_script.read_bytes() != main_script.read_bytes()
    except OSError:
        return False


def _tiers_evidence(output: str) -> str:
    """The first DRIFT / first-diff line of an exactness_tiers --check run ("" when none)."""
    for line in output.splitlines():
        stripped = line.strip()
        if "DRIFT" in stripped or stripped.startswith("first diff"):
            return stripped
    return ""


def _golden_diff_line(output: str) -> str:
    """The first changed line of gate_golden's unified diff (skips the ---/+++/@@ headers)."""
    for line in output.splitlines():
        if line.startswith(("---", "+++", "@@")):
            continue
        if line.startswith(("-", "+")):
            return line.strip()
    return ""


def _run_golden_proof(
    cfg: Config,
    tools: dict[str, list[str]],
    run: RunFn,
    *,
    name: str,
    script: str,
    golden_name: str,
    compare_flag: str,
    capture_flag: str,
    identical: Callable[[str], bool],
    evidence: Callable[[str], str],
) -> Check:
    """One golden proof, run from the WORKTREE with the MAIN checkout's golden.

    Fails closed: a missing main golden (the drift 98cc4b86 shipped unnoticed), a missing
    worktree script, a nonzero rc or no IDENTICAL evidence are all FAILs. The FAIL names the
    first DRIFT/diff line and the exact recapture command, because a DECLARED comparability
    break (summation order and friends) is legitimate - it just has to recapture the golden in
    the same commit. A drift FAIL then runs the SAME script from the MAIN checkout, so the
    detail says whether main is identical (this diff caused it) or already drifted (a
    pre-existing on main a verifier would otherwise spend several gate runs proving). The
    status stays FAIL either way.
    """
    golden = _golden_path(cfg, golden_name)
    if not golden.is_file():
        return Check(name, "FAIL", f"main golden missing: {golden}")
    wt_script = cfg.worktree / script
    if not wt_script.is_file():
        return Check(name, "FAIL", f"worktree script missing: {wt_script}")
    rc, out = run(
        tools["python"] + [str(wt_script), compare_flag, str(golden)],
        cfg.worktree,
        {"PYTHONPATH": str(cfg.worktree)},
        TEST_TIMEOUT,
    )
    suffix = " (script changed in this diff)" if _script_changed_in_diff(cfg, script) else ""
    if rc == 0 and identical(out):
        return Check(name, "PASS", "IDENTICAL" + suffix)
    capture = (
        f"PYTHONPATH={cfg.worktree} {' '.join(tools['python'])} {wt_script} "
        f"{capture_flag} {golden}"
    )
    detail = (
        f"drift (rc={rc}): {evidence(out) or tail_line(out)}"
        f"; a declared comparability break recaptures: {capture}{suffix}"
    )
    main_script = cfg.main_root / script
    if not main_script.is_file():
        return Check(name, "FAIL", detail + f"; main has no {script}")
    main_rc, main_out = run(
        tools["python"] + [str(main_script), compare_flag, str(golden)],
        cfg.main_root,
        {"PYTHONPATH": str(cfg.main_root)},
        TEST_TIMEOUT,
    )
    if main_rc == 0 and identical(main_out):
        return Check(name, "FAIL", detail + "; main IDENTICAL (this diff caused it)")
    main_evidence = evidence(main_out) or tail_line(main_out)
    return Check(name, "FAIL", detail + f"; pre-existing on main (main: {main_evidence})")


def _exactness_tiers_check(
    cfg: Config, tools: dict[str, list[str]], run: RunFn
) -> Check:
    """The 8-cell real-catalog golden: the proof nobody ran when 98cc4b86 moved the numbers."""
    skipped = _golden_skip(cfg)
    if skipped:
        return Check("exactness-tiers", "SKIP", skipped)
    return _run_golden_proof(
        cfg, tools, run,
        name="exactness-tiers",
        script=EXACTNESS_TIERS_SCRIPT,
        golden_name=GOLDEN_TIERS_NAME,
        compare_flag="--check",
        capture_flag="--capture",
        identical=lambda out: "DRIFT" not in out,
        evidence=_tiers_evidence,
    )


def _gate_golden_check(cfg: Config, tools: dict[str, list[str]], run: RunFn) -> Check:
    """Every champion gate stage: the digest must match main's golden byte for byte."""
    skipped = _golden_skip(cfg)
    if skipped:
        return Check("gate-golden", "SKIP", skipped)
    return _run_golden_proof(
        cfg, tools, run,
        name="gate-golden",
        script=GATE_GOLDEN_SCRIPT,
        golden_name=GOLDEN_GATE_NAME,
        compare_flag="--against",
        capture_flag="--out",
        identical=lambda out: any(line.strip() == "IDENTICAL" for line in out.splitlines()),
        evidence=_golden_diff_line,
    )


def _frontend_check(cfg: Config, state: GitState, run: RunFn) -> Check:
    if cfg.frontend == "never":
        return Check("frontend", "SKIP", "disabled")
    if cfg.frontend == "always" or (not state.error and should_run_frontend(state.changed)):
        return _run_frontend(cfg, run)
    if state.error:
        return Check("frontend", "SKIP", "git diff unknown (see scope)")
    return Check("frontend", "SKIP", "auto: nothing under frontend/ changed")


def run_checks(
    cfg: Config, run: RunFn = real_run, facts: GitFacts | None = None
) -> list[Check]:
    state = git_state(cfg.worktree, cfg.base, run, facts)
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

    # 1. scope (the bus's board/telemetry mirror files are never the worker's diff)
    reviewable = reviewable_paths(state.changed)
    if state.error:
        checks.append(Check("scope", "FAIL", state.error))
    elif not reviewable:
        detail = cfg.empty_diff_detail or f"empty diff vs {cfg.base} (nothing to review)"
        checks.append(Check("scope", "FAIL", detail))
    elif cfg.scope:
        offenders = scope_offenders(reviewable, cfg.scope)
        if offenders:
            # EVERY offender: the maker fixes the scope from this line, and the tools are all
            # skipped on a scope FAIL, so a first-five cut left the rest to a second command.
            shown = ", ".join(offenders)
            detail = f"{len(offenders)}/{len(reviewable)} paths out of scope: {shown}"
            checks.append(Check("scope", "FAIL", detail))
        else:
            checks.append(Check("scope", "PASS", f"{len(reviewable)} paths in scope"))
    else:
        checks.append(Check("scope", "SKIP", "no --scope given"))
    if checks[-1].name == "scope" and "out of scope" in checks[-1].detail:
        # Nothing else is worth minutes: twice a scope-only FAIL surfaced after a 2-3 min test
        # run. The tools run once the diff is in scope (bus-inbox still runs: time-dependent).
        checks += [Check(n, "SKIP", "scope FAIL: fix scope first") for n in TOOL_CHECK_NAMES]
        checks.append(bus_inbox_check(cfg))
        order = {name: i for i, name in enumerate(CHECK_ORDER)}
        return sorted(checks, key=lambda c: order.get(c.name, len(order)))

    # 2. clean
    if state.untracked:
        shown = ", ".join(state.untracked[:5])
        more = f" (+{len(state.untracked) - 5} more)" if len(state.untracked) > 5 else ""
        checks.append(Check("clean", "WARN", f"{len(state.untracked)} untracked: {shown}{more}"))
    else:
        checks.append(Check("clean", "PASS", "no untracked files"))

    # 3. ruff / 4. mypy / 6. exactness / 6b. exactness-carry / 6c. exactness-tiers /
    # 6d. gate-golden / 7. frontend run on worker threads WHILE pytest runs: each is one mostly
    # single-threaded process (mypy ~5 s, exactness ~3-6 s), and the test run leaves cores idle
    # in its tail, so the gate's wall time is the test run's instead of the sum (bead q2ijs).
    # The two golden proofs only start a process in --full / --exactness always (the 8-cell
    # tiers check keeps its own 4-process pool busy for ~1.5 min there); per round they SKIP.
    unknown = "git diff unknown (see scope)"

    def lint(name: str) -> Check:
        rc, out = run(tools[name], cfg.worktree, None, TOOL_TIMEOUT)
        return Check(name, "PASS" if rc == 0 else "FAIL", tail_line(out))

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="swarm-check") as pool:
        side = [
            # first in the queue: a ~4 s history-DB read must not wait behind the tools
            pool.submit(stale_reads_check, cfg),
            pool.submit(lint, "ruff"),
            pool.submit(lint, "mypy"),
            pool.submit(_exactness_check, cfg, state, tools, run),
            pool.submit(_exactness_carry_check, cfg, state, tools, run),
            pool.submit(_exactness_tiers_check, cfg, tools, run),
            pool.submit(_gate_golden_check, cfg, tools, run),
            pool.submit(_frontend_check, cfg, state, run),
        ]
        # 5. tests (+ the file set it ran, which the pinned-text check compares against)
        if state.error:
            test_run = TestRun(Check("tests", "SKIP", unknown))
        else:
            test_run = _run_tests(cfg, state, tools, run, wt_env)
        checks.append(test_run.check)
        # 5a. the 2 s rule: a NEW or CHANGED test over 2 s alone needs a reasoned slow mark
        if test_run.slow is not None:
            checks.append(test_run.slow)
        # 5a'. simple-target: a NEW test that only touches simple, non-exempt code (not after
        # a red policy call: the gate already FAILs, a coverage re-run only costs time)
        checks.append(
            Check("simple-tests", "SKIP", f"{POLICY_STOP}: new tests not re-run")
            if test_run.stopped
            else simple_tests_check(cfg, state, tools, run, wt_env)
        )
        # 5b. pinned-text: a test the gate does NOT run that pins a string the diff removed
        checks.append(pinned_text_check(cfg, state, run, test_run))
        checks.extend(f.result() for f in side)  # re-raises a crash: main() fails closed

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


def _board_store_module() -> Any:
    """The board_store.py next to this gate, as bus.py imports it (one module, one read cache)."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    return importlib.import_module("board_store")


def _handoff_lint_module() -> Any:
    """The handoff_lint.py next to this gate (done-needs-handoff)."""
    path = Path(__file__).resolve().parent / "handoff_lint.py"
    spec = importlib.util.spec_from_file_location("swarm_gate_handoff_lint", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stale_reads_module() -> Any:
    """The stale_reads.py next to this gate (stdlib only; reads the T3 history DB)."""
    path = Path(__file__).resolve().parent / "stale_reads.py"
    spec = importlib.util.spec_from_file_location("swarm_gate_stale_reads", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _t3_db_path() -> Path:
    """The T3 history DB ($SWARM_T3_DB overrides the local T3 state; a test patches the default)."""
    raw = os.environ.get(T3_DB_ENV)
    return Path(raw).expanduser() if raw else T3_DB_DEFAULT


def stale_reads_check(cfg: Config) -> Check:
    """FAIL while the maker's own commands read or ran a tree other than its worktree.

    Time-dependent (the child's command log grows until it hands back), so this runs on every
    invocation, including a cached PASS, and is never part of the stamp - like bus-inbox. It
    applies to the same makers as the gate-runs cap (cc-/impl-), never to --full, verifiers or
    reviewers, whose commands are not the change under review.
    """
    if not cfg.bus_agent:
        return Check("stale-reads", "SKIP", "no --bus-agent/SWARM_AGENT to look up")
    if not cfg.bus_agent.startswith(CAPPED_AGENT_PREFIXES):
        return Check(
            "stale-reads", "SKIP", f"{cfg.bus_agent} is not a maker (cc-/impl-): no per-agent check"
        )
    if cfg.full:
        return Check("stale-reads", "SKIP", "integration run (--full): the maker gate only")
    db = _t3_db_path()
    if not db.is_file():
        return Check("stale-reads", "SKIP", f"no T3 history db at {db}")
    try:
        mod = _stale_reads_module()
        stale, total = mod.stale_report(
            db,
            cfg.bus_agent,
            cfg.worktree,
            cfg.main_root,
            unchanged=mod.unchanged_paths(cfg.worktree, cfg.main_root, base=cfg.base),
        )
    except Exception as exc:  # fail closed: a history DB that cannot be read is not a PASS
        reason = f"{type(exc).__name__}: {exc}"
        return Check("stale-reads", "FAIL", f"cannot read T3 history db {db}: {reason}")
    if not stale:
        return Check("stale-reads", "PASS", f"0 of {total} commands")
    first = " ".join(stale[0].split())[:STALE_READS_SHOWN_CHARS]
    return Check(
        "stale-reads",
        "FAIL",
        f"{len(stale)} stale command(s), e.g. {first}; re-read from {cfg.worktree}",
    )


def _last_commit(cfg: Config) -> tuple[int, str] | None:
    """(epoch, short sha) of the worktree's last commit, None on any git error (fail closed).

    `git -C <worktree> log -1 --format=%ct %h HEAD`: the epoch the unread-message rule compares
    against, plus the short sha its WARN detail names.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(cfg.worktree), "log", "-1", "--format=%ct %h", "HEAD"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    parts = proc.stdout.split()
    if proc.returncode != 0 or len(parts) != 2 or not parts[0].isdigit():
        return None
    return int(parts[0]), parts[1]


_NO_FILE_INBOX = "the file store has no worker inbox: unset SWARM_STORE"


def bus_inbox_check(cfg: Config) -> Check:
    """WARN when every unread chief message predates the worktree's last commit; else FAIL.

    A worker must not hand back while a chief DM it has NOT acted on is unread, so this runs on
    EVERY invocation, including a cached PASS: it is time-dependent (a DM can arrive at any
    moment), so it is NOT part of the PASS-stamp cache key. It peeks (never consumes) the
    agent's inbox.

    The retro case (bead vibe-quant-hf9d3, item C10): the chief posted a ruling just before the
    maker's gate and the maker had already acted on it in its last commit (read in a DM thread),
    yet the peek cursor still showed it unread and the gate FAILed a round. A message older than
    the last commit cannot undo what that commit already ships, so it is a WARN (a reminder).
    Any message newer than OR EQUAL TO the last commit may still be unread work, and a bad
    timestamp or a git error is not evidence of anything: those stay FAIL (fail closed).

    The board store reads `_board_inbox`; the file store has no worker inbox (R27): FAIL. A done
    message by the agent without a linted handoff FAILs in both (handoff_lint.done_without_handoff
    over bus._all, so store-independent).
    """
    if not cfg.bus_agent and not cfg.bus:
        return Check("bus-inbox", "SKIP", "no --bus-agent/SWARM_AGENT or --bus/SWARM_BUS")
    if not cfg.bus_agent:
        return Check("bus-inbox", "SKIP", f"bus {cfg.bus} given but no --bus-agent/SWARM_AGENT to peek")
    if not cfg.bus:
        return Check("bus-inbox", "FAIL", f"agent {cfg.bus_agent} given but no --bus/SWARM_BUS to peek")
    if not Path(cfg.bus).is_dir():
        return Check("bus-inbox", "FAIL", f"configured bus {cfg.bus} is not a directory")
    board = False
    try:
        mod = _bus_module()
        board = mod.store_kind() == "board"
        job = Path(cfg.bus).resolve().parent
        done = [
            hit for hit in _handoff_lint_module().done_without_handoff(mod._all(Path(cfg.bus)), job)
            if hit[0] == cfg.bus_agent
        ]
        inbox = _board_inbox(cfg) if board else Check("bus-inbox", "FAIL", _NO_FILE_INBOX)
    except (Exception, SystemExit) as exc:  # fail closed (bus.py exits on a board error)
        reason = str(exc) if isinstance(exc, SystemExit) else f"{type(exc).__name__}: {exc}"
        if board:
            return Check(
                "bus-inbox", "FAIL",
                f"cannot read the board for {cfg.bus_agent}: {reason} — start it: scripts/agents/board-up.sh",
            )
        return Check("bus-inbox", "FAIL", f"cannot read bus {cfg.bus}: {reason}")
    if not done:
        return inbox
    path = job / "handoffs" / f"{cfg.bus_agent}.md"
    details = [f"{a} posted done ({mid}) without a handoff: {why} — write and lint {path}" for a, mid, why in done]
    if inbox.status != "PASS":
        details.append(inbox.detail)
    return Check("bus-inbox", "FAIL", "; ".join(details))


def _created_epoch(msg: Mapping[str, object]) -> float | None:
    """A board message's `createdAt` (ISO, UTC `Z`) as epoch seconds; None when it cannot be
    parsed or carries no zone (fail closed: no guess about which clock it was)."""
    try:
        stamp = datetime.fromisoformat(str(msg.get("createdAt")).replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp.timestamp() if stamp.tzinfo is not None else None


def _board_inbox(cfg: Config) -> Check:
    """The board store: the agent's task thread (root in job/<job> labelled agent=<id>, R18).

    Unread = chief messages after the agent's newest message in it, by board order (ids are
    opaque). The author is the `from` label, else `by.profile` (R15), never the old-shape `to`:
    that follows the root's label and would hide a chief note written before a round change.
    """
    bs = _board_store_module()
    agent = cfg.bus_agent
    channel = f"job/{bs.job_slug(Path(cfg.bus).resolve().parent.name)}"

    def labels(m: Mapping[str, object]) -> Mapping[str, object]:
        raw = m.get("labels")
        return raw if isinstance(raw, dict) else {}

    def author(m: Mapping[str, object]) -> str:
        by = m.get("by")
        return str(labels(m).get("from") or (by.get("profile") if isinstance(by, dict) else "") or "")

    roots = [
        m for m in bs.read_channel(bs.ns() + channel)
        if not m.get("re") and not m.get("deleted") and labels(m).get("agent") == agent
    ]
    if not roots:
        return Check("bus-inbox", "FAIL", f"no task thread for {agent} in {channel}")
    tid = str(roots[-1]["id"])
    me = {agent, bs.profile_of(agent)}
    talk = [m for m in bs.read_thread(tid) if not m.get("deleted") and (author(m) in me or author(m) == "chief")]
    last_mine = max((i for i, m in enumerate(talk) if author(m) in me), default=-1)
    unread = [m for m in talk[last_mine + 1:] if author(m) == "chief"]
    if last_mine < 0:
        quote = f"; newest chief message {unread[-1]['id']} {unread[-1].get('text')}" if unread else ""
        return Check("bus-inbox", "FAIL", f"no reply by {agent} in thread {tid}{quote} — reply in thread {tid}")
    if not unread:
        return Check("bus-inbox", "PASS", f"no unread chief message in thread {tid} for {agent}")
    listed = "; ".join(f"{m['id']} {m.get('text')}" for m in unread)
    tail = f" — run: vb read {tid}, then reply"
    epochs = [_created_epoch(m) for m in unread]
    last = _last_commit(cfg) if all(e is not None for e in epochs) else None
    if last is not None and all(e < last[0] for e in epochs if e is not None):
        return Check(
            "bus-inbox",
            "WARN",
            f"{len(unread)} unread from chief in thread {tid}, all older than last commit {last[1]}: {listed}{tail}",
        )
    return Check("bus-inbox", "FAIL", f"{len(unread)} unread from chief in thread {tid}: {listed}{tail}")


def list_tests(cfg: Config, run: RunFn) -> int:
    """--list-tests: the selection, one test file per line on stdout (pipe it to pytest), and
    one `#` line per pytest call on stderr. Runs no tool; exit 1 when the diff is unknown."""
    state = git_state(cfg.worktree, cfg.base, run)
    if state.error:
        print(f"swarm-check: {state.error}", file=sys.stderr)
        return 1
    plan = plan_tests(cfg, state)
    for path in plan.files:
        print(path)
    for args, _, _ in plan.commands:
        shown = [a for a in args if not a.startswith("tests/")]
        files = len(args) - len(shown)
        print(f"# pytest <{files} files> {' '.join(shown)}".rstrip(), file=sys.stderr)
    for note in plan.notes:
        print(f"# {note}", file=sys.stderr)
    for kind, paths in plan.reasons.items():  # why each file is in the list (a file may repeat)
        print(f"# selected by {kind} ({len(paths)}): {', '.join(Path(f).stem for f in paths)}", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None, run: RunFn | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-review gate for a worker's worktree")
    parser.add_argument("worktree")
    parser.add_argument(
        "--base", default=None, help="base ref; default main, or origin/main on branch main"
    )
    parser.add_argument("--scope", default="")
    parser.add_argument("--exactness", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--frontend", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--json", dest="json_out", default=None)
    parser.add_argument("--no-cache", dest="no_cache", action="store_true")
    parser.add_argument(
        "--full", action="store_true",
        help="integration mode: whole tests/unit + every timing test (the verifier, before land)",
    )
    parser.add_argument(
        "--list-tests", dest="list_tests", action="store_true",
        help="print the test files this gate would run (one per line) and exit; runs no tool",
    )
    parser.add_argument("--bus", default=None, help="job bus dir for the bus-inbox check (else $SWARM_BUS)")
    parser.add_argument("--bus-agent", dest="bus_agent", default=None, help="whose inbox to peek (else $SWARM_AGENT)")
    parser.add_argument(
        "--expect-pins", dest="expect_pins", default=None,
        help="declared comparability-break pins JSON for exactness_239 --expect (else $SWARM_EXPECT_PINS)",
    )
    args = parser.parse_args(argv)

    wt = Path(args.worktree).resolve()
    if not (wt / ".git").exists() and subprocess.run(
        ["git", "-C", str(wt), "rev-parse", "--git-dir"], capture_output=True
    ).returncode != 0:
        print(f"not a git worktree: {wt}")
        return 2

    runner: RunFn = run or real_run
    # No --base passed: on branch `main` with an origin/main, gate that instead of the local
    # main the diff is measured against (which is HEAD there, hence always empty).
    base, base_info = resolve_base(wt, args.base, runner)
    if base_info:
        print(base_info)

    main_root = _derive_main_root(wt)
    cfg = Config(
        worktree=wt,
        main_root=main_root,
        base=base,
        scope=parse_scope(args.scope),
        exactness=args.exactness,
        frontend=args.frontend,
        full=args.full,
        json_out=Path(args.json_out) if args.json_out else None,
        slots_dir=main_root / _gate_slots_module().SLOTS_SUBDIR if run is None else None,
        file_cache_dir=(
            main_root.joinpath(*FILE_CACHE_SUBDIR)
            if run is None and not args.no_cache and not args.full
            else None
        ),
        bus=args.bus or os.environ.get("SWARM_BUS") or "",
        bus_agent=args.bus_agent or os.environ.get("SWARM_AGENT") or "",
        expect_pins=str(Path(pins).resolve()) if (pins := args.expect_pins or os.environ.get("SWARM_EXPECT_PINS")) else "",
        empty_diff_detail=(
            "empty diff vs origin/main: nothing unpushed to gate; "
            "pass --base <sha> to gate pushed commits"
            if base_info
            else ""
        ),
    )

    if args.list_tests:
        return list_tests(cfg, runner)

    # B1: refuse the hand-back BEFORE spending a suite run (or even reading the cache) when the
    # chief's DM is already waiting. A DM that lands mid-run is still caught by the end-of-run
    # bus-inbox check in run_checks, so this only short-circuits the already-unread case.
    pre_bus = bus_inbox_check(cfg)
    if pre_bus.status == "FAIL":
        print(format_line(pre_bus))
        print(gate_summary([pre_bus])[0])
        return 1

    # The log target is fixed before the tools run so its prior count is the run counter.
    facts = GitFacts()  # the PRE-run phase's git answers: asked once, shared by probe+checks
    log_path, prior_runs = gate_log_target(cfg, runner, facts)
    if prior_runs >= 1:
        print(
            f"INFO gate run #{prior_runs + 1} on this HEAD - the gate runs once per round; "
            "read the LOG of the last run instead of re-running"
        )

    # --full (the verifier, before land) backstops every per-round shortcut: it never reads
    # or writes a stamp (review rev-gate-1 B4). An empty key disables both.
    key, _ = (
        _cache_probe(cfg, runner, facts=facts) if not args.no_cache and not args.full else ("", None)
    )  # PRE-run key
    stamp = load_stamp(wt, runner, key, facts) if key else None

    # bus-inbox and stale-reads are time-dependent, so they are re-evaluated even when the
    # tool checks are cached: a cached PASS with an unread chief DM, or with a command the
    # maker ran against main, is still a GATE FAIL (run_checks adds them otherwise).
    if stamp is not None:
        for check in stamp.checks:
            print(f"(cached) {format_line(check)}")
        print(f"PASS cache: reused stamp from {stamp.ts} for {stamp.head[:12]}")
        live = apply_warn_mode([stale_reads_check(cfg), bus_inbox_check(cfg)])
        checks = [*stamp.checks, *live]
        for check in live:
            print(format_line(check))
        gate, failed = gate_summary(checks)
        write_gate_log(
            log_path,
            [],
            header=f"# cached run at {stamp.ts}: no tool ran",
            trailer=gate_log_trailer(cfg, runner, facts, GitFacts(), gate),
        )
        print(f"LOG {log_path}")
        print(gate)
        if cfg.json_out:
            cached = to_json_dict(checks)
            cached["cached"] = True
            with open(cfg.json_out, "w", encoding="utf-8") as fh:
                json.dump(cached, fh, indent=2)
                fh.write("\n")
        return 1 if failed else 0

    # The gate runs once per round: refuse a maker's 4th UNCACHED run before any tool. A cached
    # PASS returned above, so it is neither refused nor counted (it costs nothing).
    runs_refusal = gate_runs_cap(cfg, log_path)
    if runs_refusal is not None:
        print(format_line(runs_refusal))
        print(gate_summary([runs_refusal])[0])
        return 1

    # A commit+reset --hard made while the tools run puts HEAD and the tree back where they
    # started, so the pre/post key compare cannot see it; the reflog length can (see N1 doc).
    reflog_before = head_reflog_count(wt, runner) if key else None

    recorder = ToolLog(runner)
    try:
        checks = run_checks(cfg, recorder, facts)
    except Exception as exc:  # fail closed: a crashed gate is a failed gate
        checks = [Check("internal", "FAIL", f"gate crashed: {type(exc).__name__}: {exc}")]
    checks = apply_warn_mode(annotate_tool_fails(checks))
    for check in checks:
        print(format_line(check))
    # A scope FAIL skips every tool, so the transcript would be empty: the scope line (which
    # lists every out-of-scope path) becomes the log header, so the LOG names what to fix.
    header = next(
        (format_line(c) for c in checks if c.name == "scope" and c.status == "FAIL"), ""
    )
    gate, failed = gate_summary(checks)
    post_facts = GitFacts()  # a NEW phase: a mid-run commit/edit must be seen (trailer + stamp)
    write_gate_log(
        log_path,
        recorder.entries,
        header=header,
        trailer=gate_log_trailer(cfg, runner, facts, post_facts, gate),
    )
    print(f"LOG {log_path}")
    print(gate)

    if cfg.json_out:
        with open(cfg.json_out, "w", encoding="utf-8") as fh:
            json.dump(to_json_dict(checks), fh, indent=2)
            fh.write("\n")

    # B3: a bus-inbox-only FAIL is time-dependent (the DM may simply be read next run), and so
    # is a stale-reads-only FAIL (the maker can re-read from its worktree and re-run), so the
    # tool PASS is still stamped; only a real tool FAIL blocks the stamp.
    tool_failed = [
        c for c in checks if c.status == "FAIL" and c.name not in TIME_DEPENDENT_CHECKS
    ]
    if not args.no_cache and not tool_failed and not active_stubs() and key:
        # key is the PRE-run key (non-empty = clean start). Stamp only if nothing moved while
        # the tools ran; otherwise the verdict is for a HEAD/tree that was never gated.
        post_key, _ = _cache_probe(cfg, runner, facts=post_facts)
        reflog_after = head_reflog_count(wt, runner)
        if post_key == key and reflog_before is not None and reflog_after == reflog_before:
            # time-dependent checks are not part of the key: never store them as a cached PASS/SKIP
            stamped = [c for c in checks if c.name not in TIME_DEPENDENT_CHECKS]
            with contextlib.suppress(Exception):
                write_stamp(wt, runner, cfg, key, stamped, post_facts)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
