#!/usr/bin/env bash
# Throwaway UI stack for checking a worktree's code in a browser, next to the user's own servers.
#
#   scripts/agents/ui-check.sh up   <worktree>   # backend :8001 + Vite :5188 on a COPY of the state DB
#   scripts/agents/ui-check.sh down <worktree>   # stop both, delete the DB copy
#
# The DB copy lives at /tmp/vq-ui-<name>.db; seed test rows there (never in data/state).
# Never touches :8000/:5173. Drive the page with agent-browser (Bash tool needs
# dangerouslyDisableSandbox: true) and ALWAYS run `down` afterwards, pass or fail.
set -euo pipefail

[[ $# -eq 2 && ( "$1" == up || "$1" == down ) ]] || { sed -n '4,5p' "$0" >&2; exit 2; }
cmd="$1"; wt="$(cd "$2" && pwd)"
main="$(cd "$(dirname "$0")/../.." && pwd)"
name="$(basename "$wt")"
db="/tmp/vq-ui-$name.db"; pids="/tmp/vq-ui-$name.pids"
api=8001; web=5188

if [[ "$cmd" == down ]]; then
  [[ -f "$pids" ]] && xargs kill < "$pids" 2>/dev/null || true
  # pnpm forks vite; also stop whatever still listens on the stack's own ports.
  for p in $api $web; do lsof -ti "tcp:$p" -sTCP:LISTEN 2>/dev/null | xargs kill 2>/dev/null || true; done
  rm -f "$pids" "$db" "$db-wal" "$db-shm"
  echo "ui-check down ($name)"; exit 0
fi

for p in $api $web; do
  if lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "port $p busy — run '$0 down <worktree>' for the stale stack first" >&2; exit 1
  fi
done
[[ -e "$wt/frontend/node_modules" ]] || ln -s "$main/frontend/node_modules" "$wt/frontend/node_modules"

# Online backup = consistent copy even while the real backend writes.
python3 -c "import sqlite3,sys; s=sqlite3.connect(f'file:{sys.argv[1]}?mode=ro',uri=True); d=sqlite3.connect(sys.argv[2]); s.backup(d)" \
  "$main/data/state/vibe_quant.db" "$db"

(cd "$wt" && VIBE_QUANT_DB="$db" PYTHONPATH="$wt" nohup "$main/.venv/bin/uvicorn" \
  "vibe_quant.api.app:create_app" --factory --port $api > "/tmp/vq-ui-$name-api.log" 2>&1 & echo $! >> "$pids")
(cd "$wt/frontend" && VQ_API_PORT=$api nohup pnpm exec vite --port $web --strictPort \
  > "/tmp/vq-ui-$name-web.log" 2>&1 & echo $! >> "$pids")

for _ in $(seq 60); do
  curl -sf "localhost:$api/api/discovery/jobs" >/dev/null 2>&1 && curl -sf "localhost:$web/" >/dev/null 2>&1 && break
  sleep 1
done
curl -sf "localhost:$web/" >/dev/null || { echo "stack failed to start; logs: /tmp/vq-ui-$name-{api,web}.log" >&2; exit 1; }

cat <<EOF
ui-check up ($name)
web: http://localhost:$web   (title must be "vibe-quant")
api: http://localhost:$api   db copy: $db
logs: /tmp/vq-ui-$name-{api,web}.log
stop: $0 down $wt
EOF
