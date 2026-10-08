#!/usr/bin/env bash
# Run one headless Command Code (`cmd`) task from a brief file.
# Prints the agent's final text on stdout; full NDJSON transcript goes to the log dir.
# Exit code: cmd's own (0 ok, 8 = hit --max-turns), 2 = usage error.
#
#   scripts/agents/cmd-task.sh [--tier fast|code|pro|long | --model ID] [--max-turns N]
#                              [--cwd DIR] [--log-dir DIR] [--effort LEVEL] BRIEF.md
#
# BRIEF.md may be "-" to read the brief from stdin.
# --yolo is always on (no permission prompts): point --cwd at a disposable worktree
# for any task that writes files. See docs/orchestration/cheap-agents.md.
set -uo pipefail

tier="fast"; model=""; turns=30; cwd="$PWD"; log_dir=""; effort=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier) tier="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --max-turns) turns="$2"; shift 2 ;;
    --cwd) cwd="$2"; shift 2 ;;
    --log-dir) log_dir="$2"; shift 2 ;;
    --effort) effort="$2"; shift 2 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    -) break ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) break ;;
  esac
done
[[ $# -eq 1 ]] || { echo "usage: $0 [options] BRIEF.md|-" >&2; exit 2; }

if [[ -z "$model" ]]; then
  case "$tier" in
    fast) model="deepseek/deepseek-v4.1-flash-fast" ;;
    code) model="deepseek/deepseek-v4.1-flash"; effort="${effort:-max}" ;;
    pro) model="xiaomi/mimo-v2.6-pro" ;;
    long) model="moonshotai/kimi-k3" ;;
    *) echo "unknown tier: $tier (fast|code|pro|long)" >&2; exit 2 ;;
  esac
fi

if [[ "$1" == "-" ]]; then
  brief="$(cat)"; name="stdin"
else
  [[ -f "$1" ]] || { echo "brief not found: $1" >&2; exit 2; }
  brief="$(cat "$1")"; name="$(basename "$1" .md)"
fi

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
log_dir="${log_dir:-$repo_root/data/swarm/cmd-logs}"
mkdir -p "$log_dir"
log="$log_dir/$(date +%Y%m%d-%H%M%S)-$name-${model//\//_}.ndjson"

args=(-p "$brief" -m "$model" --yolo -t --skip-onboarding --no-session --no-auto-update
      --max-turns "$turns" --output-format json)
[[ -n "$effort" ]] && args+=(--effort "$effort")

(cd "$cwd" && cmd "${args[@]}") > "$log" 2>&1
rc=$?

python3 - "$log" <<'PY'
import json, sys
result = None
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    try:
        ev = json.loads(line)
    except ValueError:
        continue
    if ev.get("type") == "result":
        result = ev
if result is None:
    print("[cmd-task] no result event; see log", file=sys.stderr)
else:
    u = result.get("usage") or {}
    print(result.get("finalText", ""))
    print(f"[cmd-task] {result.get('subtype')} stop={result.get('stopReason')} "
          f"ms={result.get('durationMs')} in={u.get('inputTokens')} out={u.get('outputTokens')}",
          file=sys.stderr)
PY
echo "[cmd-task] model=$model rc=$rc log=$log" >&2
exit $rc
