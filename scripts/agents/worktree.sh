#!/usr/bin/env bash
# Create a swarm worktree that can run the repo's Python, tests and exactness proofs.
#
#   scripts/agents/worktree.sh <slug> [base-ref]     # base-ref defaults to HEAD of the main checkout
#
# Creates ../vq-<slug> on branch swarm/<slug> from base-ref (pass a sibling task's
# branch to stack work), symlinks the main checkout's read-only market data
# (data/catalog, data/archive) so catalog-backed tests and exactness_239.py run
# there, and prints the env line every brief must paste. data/state is NOT linked:
# the state DB stays single-writer (exactness_239.py opens main's DB read-only).
set -euo pipefail

[[ $# -ge 1 && $# -le 2 ]] || { sed -n '4,4p' "$0" >&2; exit 2; }
slug="$1"; base="${2:-HEAD}"
main="$(cd "$(dirname "$0")/../.." && pwd)"
wt="$(dirname "$main")/vq-$slug"

git -C "$main" worktree add -q "$wt" -b "swarm/$slug" "$base"
mkdir -p "$wt/data"
for d in catalog archive; do
  [[ -e "$main/data/$d" ]] && ln -s "$main/data/$d" "$wt/data/$d"
done

cat <<EOF
worktree: $wt
branch:   swarm/$slug (from $base)
run as:   cd $wt && PYTHONPATH=\$PWD $main/.venv/bin/python -m pytest ...
gates:    $main/.venv/bin/ruff check && $main/.venv/bin/mypy
remove:   git -C $main worktree remove $wt && git -C $main branch -D swarm/$slug
EOF
