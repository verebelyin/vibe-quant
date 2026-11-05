#!/usr/bin/env bash
# Throwaway UI stack for checking a worktree's code in a browser, next to the user's own servers.
#
#   scripts/agents/ui-check.sh up   <worktree>   # backend :8001 + Vite :5188 on a COPY of the state DB
#   scripts/agents/ui-check.sh down <worktree>   # stop both, delete the DB copy
#
# `up` exits 3 when :8001/:5188 is already busy (a live stack), so callers can tell
# "someone else owns the port" apart from "the stack failed to start" and not kill it.
#
# The DB copy lives at /tmp/vq-ui-<name>.db; seed test rows there (never in data/state).
# Never touches :8000/:5173. Drive the page with agent-browser (Bash tool needs
# dangerouslyDisableSandbox: true) and ALWAYS run `down` afterwards, pass or fail.
set -euo pipefail

[[ $# -eq 2 && ( "$1" == up || "$1" == down ) ]] || { sed -n '4,5p' "$0" >&2; exit 2; }
cmd="$1"; wt="$(cd "$2" && pwd)"
# Main checkout (holds data/state + .venv) even when this script runs from a worktree copy.
main="$(cd "$(git -C "$(dirname "$0")" rev-parse --path-format=absolute --git-common-dir)/.." && pwd)"
name="$(basename "$wt")"
db="/tmp/vq-ui-$name.db"; pids="/tmp/vq-ui-$name.pids"
api=8001; web=5188

# Stop this stack: the recorded pids, then whatever still listens on its ports (pnpm forks
# vite, so the recorded pnpm pid alone can leave vite running), then the DB copy, then the
# browsers the checks drove: the agent-browser session `up` names and orphaned automation
# Chromium (a stray one held a core through two research runs, retro 2026-11-05).
stop_stack() {
  [[ -f "$pids" ]] && xargs kill < "$pids" 2>/dev/null || true
  for p in $api $web; do lsof -ti "tcp:$p" -sTCP:LISTEN 2>/dev/null | xargs kill 2>/dev/null || true; done
  rm -f "$pids" "$db" "$db-wal" "$db-shm"
  ! command -v agent-browser >/dev/null || agent-browser --session "vq-ui-$name" close >/dev/null 2>&1 || true
  "$(dirname "$0")/kill-stray-browsers.sh" || true
}

if [[ "$cmd" == down ]]; then
  stop_stack
  echo "ui-check down ($name)"; exit 0
fi

for p in $api $web; do
  if lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then
    # Exit 3 specifically: another stack is live on this port. Callers must NOT
    # treat this like "stack failed to start" and tear the live stack down.
    echo "port $p busy — run '$0 down <worktree>' for the stale stack first" >&2; exit 3
  fi
done
# Ctrl-C/TERM during `up` must not orphan the servers: bash async children ignore SIGINT
# without job control, so only this trap can stop them. Installed after the busy check (the
# ports are ours to stop now); cleared once `up` succeeds. A stale pids file could name
# recycled pids, so drop it before the new stack writes its own.
trap 'trap - INT TERM; stop_stack; echo "ui-check up interrupted ($name): stack stopped" >&2; exit 130' INT TERM
rm -f "$pids"
[[ -e "$wt/frontend/node_modules" ]] || ln -s "$main/frontend/node_modules" "$wt/frontend/node_modules"

# Online backup = consistent copy even while the real backend writes.
python3 -c "import sqlite3,sys; s=sqlite3.connect(f'file:{sys.argv[1]}?mode=ro',uri=True); d=sqlite3.connect(sys.argv[2]); s.backup(d)" \
  "$main/data/state/vibe_quant.db" "$db"

# Background each server with BOTH streams redirected and stdin from /dev/null.
# (The old `(cd … && nohup … & echo $!)` backgrounded the whole `cd && nohup` list,
# so a subshell that inherited the caller's stdout stayed alive as pnpm's parent —
# `… | tail` never returned — and recorded the subshell pid, not the server's.) The
# uvicorn line is exec'd, so its recorded pid IS the server pid; pnpm forks vite, so
# the Vite pid recorded here is the pnpm wrapper (down also kills by port to catch
# the forked child).
(cd "$wt" && exec env VIBE_QUANT_DB="$db" PYTHONPATH="$wt" nohup "$main/.venv/bin/uvicorn" \
  "vibe_quant.api.app:create_app" --factory --port $api) \
  > "/tmp/vq-ui-$name-api.log" 2>&1 </dev/null &
echo $! >> "$pids"
(cd "$wt/frontend" && exec env VQ_API_PORT=$api nohup pnpm exec vite --port $web --strictPort) \
  > "/tmp/vq-ui-$name-web.log" 2>&1 </dev/null &
echo $! >> "$pids"

for _ in $(seq 60); do
  curl -sf "localhost:$api/api/discovery/jobs" >/dev/null 2>&1 && curl -sf "localhost:$web/" >/dev/null 2>&1 && break
  sleep 1
done
curl -sf "localhost:$web/" >/dev/null || { echo "stack failed to start; logs: /tmp/vq-ui-$name-{api,web}.log" >&2; exit 1; }
trap - INT TERM

cat <<EOF
ui-check up ($name)
web: http://localhost:$web   (title must be "vibe-quant")
api: http://localhost:$api   db copy: $db
logs: /tmp/vq-ui-$name-{api,web}.log
browse: agent-browser --session vq-ui-$name open http://localhost:$web   (down closes it)
stop: $0 down $wt
EOF
