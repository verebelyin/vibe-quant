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

# ONE `git rev-parse` for the two paths the hook needs (a spawn costs 25-40 ms on a loaded
# host and this hook runs on every commit). The branch stays a separate question: it must not
# be asked before the staged list on an unborn HEAD (`--abbrev-ref HEAD` fails there).
repo_facts="$(git rev-parse --path-format=absolute --git-common-dir --show-toplevel 2>/dev/null)" || exit 0
common_dir="${repo_facts%%$'\n'*}"
toplevel="${repo_facts#*$'\n'}"
ruff_bin="${VQ_RUFF:-$(dirname "$common_dir")/.venv/bin/ruff}"

# ONE `git diff --cached` for BOTH lists below, filtered here. `--name-status -z` keeps each
# entry's status, because a .py that is only RENAMED must stay out of ruff (today's
# `--diff-filter=ACM`) while a renamed steering file still runs the budget check (`ACMR`);
# `-z` also keeps a path with a space intact. The globs are matched with `case`, whose `*`
# crosses `/` exactly like a plain git pathspec (`*.py` matches deep/pkg/mod.py, and git
# matches a rename's DESTINATION only).
files=()
steering=""
while IFS= read -r -d '' status; do
  case "$status" in
    R* | C*) IFS= read -r -d '' source || break ;;  # R100/C75 carry their source path
  esac
  IFS= read -r -d '' path || break
  case "$path" in
    CLAUDE.md | .claude/* | docs/orchestration/*.md | docs/orchestration/prompts/* | docs/claude/*.md)
      steering="$path" ;;
  esac
  case "$status" in
    R*)
      # A rename is skipped only when its source is a .py too: the old `-- '*.py'` pathspec
      # saw code.txt -> code.py as an ADDED code.py (the source is outside the pathspec), so
      # it was linted (rev-toolcache-1 F1).
      case "$path" in
        *.py)
          case "$source" in
            *.py) ;;
            *) files+=("$path") ;;
          esac
          ;;
      esac
      ;;
    *)
      case "$path" in
        *.py) files+=("$path") ;;
      esac
      ;;
  esac
done < <(git diff --cached --name-status -z --diff-filter=ACMR)

if [ -n "$steering" ]; then
  budget_cmd="${VQ_BUDGET_CMD:-$(dirname "$common_dir")/.venv/bin/python -m pytest tests/unit/test_steering_budget.py -q -p no:cacheprovider}"
  if ! (cd "$toplevel" && $budget_cmd); then
    printf 'pre-commit-lint: steering-file budget check failed (merge or cut a line; raising a budget needs the user)\n' >&2
    exit 1
  fi
fi

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
