#!/usr/bin/env bash
# Pre-review gate: mechanical checks on a worker's worktree before any reviewer sees the diff.
#
#   scripts/agents/swarm-check.sh <worktree> [--base REF] [--scope glob,...]
#       [--exactness auto|always|never] [--frontend auto|always|never]
#       [--json out.json] [--no-cache] [--bus DIR] [--bus-agent NAME]
#
# One "PASS|FAIL|SKIP <name>: <detail>" line per check, then "GATE PASS" /
# "GATE FAIL (n failed)"; exit 0/1. Tool commands overridable via
# SWARM_CHECK_RUFF / SWARM_CHECK_MYPY / SWARM_CHECK_PYTEST, but only honoured (else FAIL)
# with SWARM_CHECK_ALLOW_STUBS=1. Fails closed (bad base, empty diff, git errors).
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
# worktree's git dir; a later run with the same HEAD/base/scope/modes, unchanged
# swarm_check.py and unchanged main exactness files/lockfiles reuses it and
# prints "(cached) ..." + "PASS cache: ..." (--no-cache disables reuse and writing).
# Stdlib-only helper.
set -euo pipefail
exec python3 "$(cd "$(dirname "$0")" && pwd)/swarm_check.py" "$@"
