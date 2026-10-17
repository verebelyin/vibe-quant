#!/usr/bin/env bash
# Pre-commit lint: ruff on the Python files staged for this commit, nothing else.
#
#   scripts/git-hooks/pre-commit-lint.sh
#
# Wired into `<git-common-dir>/hooks/pre-commit` by scripts/agents/install-hooks.sh.
# Exits 0 when no Python file is staged, or with SKIP_LINT=1. The interpreter's venv is
# resolved from the shared git dir (`git rev-parse --git-common-dir`) so a linked
# worktree uses the main checkout's toolchain — a worktree usually has no `.venv` of its
# own. Override the binary with VQ_RUFF (the hook tests do).
set -euo pipefail

if [ "${SKIP_LINT:-0}" = "1" ]; then
  exit 0
fi

common_dir="$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" || exit 0
ruff_bin="${VQ_RUFF:-$(dirname "$common_dir")/.venv/bin/ruff}"

files=()
while IFS= read -r staged; do
  if [ -n "$staged" ]; then
    files+=("$staged")
  fi
done < <(git diff --cached --name-only --diff-filter=ACM -- '*.py')

if [ "${#files[@]}" -eq 0 ]; then
  exit 0
fi

if [ ! -x "$ruff_bin" ]; then
  printf 'pre-commit-lint: ruff not found at %s (set VQ_RUFF)\n' "$ruff_bin" >&2
  exit 1
fi

"$ruff_bin" check --force-exclude "${files[@]}"
