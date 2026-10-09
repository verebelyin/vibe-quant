#!/usr/bin/env bash
# Pre-review gate: mechanical checks on a worker's worktree before any reviewer sees the diff.
#
#   scripts/agents/swarm-check.sh <worktree> [--base REF] [--scope glob,...]
#       [--exactness auto|always|never] [--frontend auto|always|never] [--json out.json]
#
# One "PASS|FAIL|SKIP <name>: <detail>" line per check, then "GATE PASS" /
# "GATE FAIL (n failed)"; exit 0/1. Tool commands overridable via
# SWARM_CHECK_RUFF / SWARM_CHECK_MYPY / SWARM_CHECK_PYTEST, but only honoured (else FAIL)
# with SWARM_CHECK_ALLOW_STUBS=1. Fails closed (bad base, empty diff, git errors).
# Stdlib-only helper.
set -euo pipefail
exec python3 "$(cd "$(dirname "$0")" && pwd)/swarm_check.py" "$@"
