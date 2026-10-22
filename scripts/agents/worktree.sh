#!/usr/bin/env bash
# Create a swarm worktree that can run the repo's Python, tests and exactness proofs.
#
#   scripts/agents/worktree.sh <slug> [base-ref] [--private-node-modules]     # base-ref defaults to HEAD of the main checkout
#
# Creates ../vq-<slug> on branch swarm/<slug> from base-ref (pass a sibling task's
# branch to stack work), symlinks the main checkout's read-only market data
# (data/catalog, data/archive) so catalog-backed tests and exactness_239.py run
# there, links the main checkout's .venv (so `.venv/bin/ruff` and friends resolve in the
# worktree; a missing .venv once made `x | tail; echo EXIT=$?` report EXIT=0 for checks
# that never ran), records the start SHA in <wt>/.vq-base, and prints the env line every brief must paste. data/state is NOT linked:
# the state DB stays single-writer (exactness_239.py opens main's DB read-only).
#
# frontend/node_modules is symlinked to the main checkout's by default, so `pnpm build`
# works without a reinstall -- but a `pnpm install` in this worktree then writes through
# the symlink and changes the MAIN checkout's deps for every worktree. Pass
# --private-node-modules (any position) to skip the symlink and install a private copy
# instead: cd <worktree>/frontend && pnpm install --frozen-lockfile. Use it whenever
# package.json / pnpm-lock.yaml will change.
set -euo pipefail

private_node_modules=0
positional=()
for arg in "$@"; do
  case "$arg" in
    --private-node-modules) private_node_modules=1 ;;
    --*) echo "worktree.sh: unknown option '$arg'" >&2; exit 2 ;;
    *) positional+=("$arg") ;;
  esac
done
[[ ${#positional[@]} -ge 1 && ${#positional[@]} -le 2 ]] || { sed -n '4,4p' "$0" >&2; exit 2; }
slug="${positional[0]}"; base="${positional[1]:-HEAD}"
main="$(cd "$(dirname "$0")/../.." && pwd)"
wt="$(dirname "$main")/vq-$slug"

git -C "$main" worktree add -q "$wt" -b "swarm/$slug" "$base"
# Start SHA: mkbrief --review and land.py fork the task's own commits here, not at a moving
# merge-base(main|job branch, task). Excluded locally so it never shows up as untracked.
git -C "$wt" rev-parse HEAD > "$wt/.vq-base"
exclude="$(git -C "$wt" rev-parse --path-format=absolute --git-path info/exclude)"
grep -qxF '.vq-base' "$exclude" 2>/dev/null || echo '.vq-base' >> "$exclude"
if [[ -e "$main/.venv" ]]; then ln -s "$main/.venv" "$wt/.venv"; fi
mkdir -p "$wt/data"
for d in catalog archive; do
  [[ -e "$main/data/$d" ]] && ln -s "$main/data/$d" "$wt/data/$d"
done
# Shared deps so `pnpm build` works without a reinstall (re-install in the worktree if package.json changes).
if [[ $private_node_modules -eq 1 ]]; then
  echo "node_modules: private — run: cd $wt/frontend && pnpm install --frozen-lockfile"
elif [[ -d "$main/frontend/node_modules" ]]; then
  ln -s "$main/frontend/node_modules" "$wt/frontend/node_modules"
  echo "WARNING: $wt/frontend/node_modules is a symlink to the main checkout's — a 'pnpm install' here modifies the MAIN checkout's node_modules for every worktree; use --private-node-modules when package.json/pnpm-lock.yaml will change." >&2
fi

cat <<EOF
worktree: $wt
branch:   swarm/$slug (from $base)
run as:   cd $wt && PYTHONPATH=\$PWD $main/.venv/bin/python -m pytest ...
gates:    $main/.venv/bin/ruff check && $main/.venv/bin/mypy
remove:   git -C $main worktree remove $wt && git -C $main branch -D swarm/$slug
EOF
