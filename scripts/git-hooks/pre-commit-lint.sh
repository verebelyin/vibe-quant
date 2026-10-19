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
#
# On main, staged Python also runs the gate's POLICY_TESTS (tree scanners, ~3 s): the chief
# commits on main directly, and main went red once because only swarm-check ran them.
# Override with VQ_POLICY_CMD (the hook tests do).
#
# Staged steering files (CLAUDE.md, .claude/, docs/orchestration/, docs/claude/) also run
# tests/unit/test_steering_budget.py: word budgets, no changelogs, no job-id narrative
# (a self-improve commit once pushed two files over budget). Override with VQ_BUDGET_CMD.
set -euo pipefail

if [ "${SKIP_LINT:-0}" = "1" ]; then
  exit 0
fi

common_dir="$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" || exit 0
ruff_bin="${VQ_RUFF:-$(dirname "$common_dir")/.venv/bin/ruff}"

toplevel="$(git rev-parse --show-toplevel)"
steering="$(git diff --cached --name-only --diff-filter=ACMR -- CLAUDE.md '.claude/*' 'docs/orchestration/*.md' 'docs/orchestration/prompts/*' 'docs/claude/*.md')"
if [ -n "$steering" ]; then
  budget_cmd="${VQ_BUDGET_CMD:-$(dirname "$common_dir")/.venv/bin/python -m pytest tests/unit/test_steering_budget.py -q -p no:cacheprovider}"
  if ! (cd "$toplevel" && $budget_cmd); then
    printf 'pre-commit-lint: steering-file budget check failed (merge or cut a line; raising a budget needs the user)\n' >&2
    exit 1
  fi
fi

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

branch="$(git symbolic-ref --quiet --short HEAD 2>/dev/null || true)"
if [ "$branch" = "main" ]; then
  py="$(dirname "$common_dir")/.venv/bin/python"
  if [ -n "${VQ_POLICY_CMD:-}" ]; then
    policy_cmd="$VQ_POLICY_CMD"
  else
    policy=""
    if [ -f "$toplevel/scripts/agents/swarm_check.py" ] && [ -x "$py" ]; then
      # read the literal, never import the gate (its dataclasses need a registered module)
      policy="$(cd "$toplevel" && "$py" -c 'import ast, os
tree = ast.parse(open("scripts/agents/swarm_check.py", encoding="utf-8").read())
value = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
             and any(getattr(t, "id", "") == "POLICY_TESTS" for t in n.targets))
print(" ".join(p for p in value if os.path.exists(p)))')"
    fi
    policy_cmd="${policy:+$py -m pytest $policy -q -p no:cacheprovider}"
  fi
  if [ -n "$policy_cmd" ] && ! (cd "$toplevel" && $policy_cmd >/tmp/vq-policy-$$.log 2>&1); then
    tail -20 "/tmp/vq-policy-$$.log" >&2
    printf 'pre-commit-lint: policy tests failed on main (log: /tmp/vq-policy-%s.log)\n' "$$" >&2
    exit 1
  fi
fi
