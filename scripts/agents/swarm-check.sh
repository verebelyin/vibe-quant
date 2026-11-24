#!/usr/bin/env bash
# Pre-review gate: mechanical checks on a worker's worktree before any reviewer sees the diff.
#
#   scripts/agents/swarm-check.sh <worktree> [--base REF] [--scope glob,...]
#       [--exactness auto|always|never] [--frontend auto|always|never]
#       [--json out.json] [--no-cache] [--bus DIR] [--bus-agent NAME] [--full] [--list-tests]
#
# Per round (default): the triggered policy tests first (a red one ends the step), then the
# changed tests + import closure + by-path/name consumers, timing tests deselected; a test
# file whose PASS (data/swarm/_gate-cache/files, shared by all worktrees) no later diff
# reaches is skipped. --full (verifier, before land): whole tests/unit + the timing files.
# --list-tests prints the selection and runs nothing.
# One "PASS|FAIL|SKIP <name>: <detail>" line per check, then "GATE PASS" /
# "GATE FAIL (n failed)"; exit 0/1. Fails closed (bad base, empty diff, git errors).
# Dependency changes need no workaround: a uv.lock/pyproject.toml differing from main's runs
# every python tool (and exactness) from <worktree>/.venv, uv-synced to the worktree lock
# ("PASS python-env: worktree .venv (lock differs)"; a failed sync FAILs); a changed
# frontend/package.json or pnpm-lock.yaml swaps the shared node_modules symlink for a private
# install. Frontend runs tsc, build, vitest and lint. SWARM_CHECK_RUFF / _MYPY / _PYTEST
# stubs (with SWARM_CHECK_ALLOW_STUBS=1, else FAIL) exist for the gate's own tests.
#
# With no --base the gate keeps main, except on branch `main` with an origin/main: it then
# gates origin/main and prints "INFO base: ..." (an explicit --base main is never rewritten).
# Paths under docs/orchestration/board|telemetry/ are the bus's mirror files: never scope
# offenders, never a non-empty diff. Every run writes the tool transcript to
# <main checkout>/data/swarm/_gate-logs/<worktree>-<HEAD[:12]>-<UTC ts>.log, prints
# "LOG <path>" right before the final GATE line and marks tool FAIL lines with
# "(full output: LOG above)"; a second run on the same HEAD prints "INFO gate run #n".
# `pinned-text` FAILs when a test the gate does NOT run pins a string the diff removed.
# A GATE PASS on a clean start (and an unmoved tree at the end) writes a stamp in the
# worktree's git dir; a later run with the same tree/base/merge-base/scope/modes, unchanged
# swarm_check.py, python env and unchanged main exactness files/lockfiles reuses it and
# prints "(cached) ..." + "PASS cache: ..." (--no-cache and --full disable both caches).
# Stdlib-only helper.
set -euo pipefail
exec python3 "$(cd "$(dirname "$0")" && pwd)/swarm_check.py" "$@"
