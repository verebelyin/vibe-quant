#!/usr/bin/env bash
# Run one headless Command Code (`cmd`) task from a brief file.
# LEGACY (2026-10-10): swarm dispatch uses T3 delegate_task subagents instead — see
# docs/orchestration/cheap-agents.md § Dispatch protocol. Keep this for manual one-offs only.
# Prints the agent's final text on stdout; full NDJSON transcript goes to the log dir.
# Exit code: cmd's own (0 ok, 8 = hit --max-turns), 2 = usage error.
#
#   scripts/agents/cmd-task.sh [--tier fast|code|pro|long | --model ID] [--max-turns N]
#                              [--cwd DIR] [--log-dir DIR] [--effort LEVEL] [--agent NAME] BRIEF.md
#
# Every worker joins the swarm board as --agent NAME (default: the brief name): the job bus if
# SWARM_BUS=<job>/bus is set, else the persistent lobby. The protocol in
# docs/orchestration/prompts/bus-protocol.md is prepended to the brief (see scripts/agents/bus.py).
# The worker always runs in its own session (signals to the wrapper are forwarded). With a job bus
# it is also registered with stall_watch.py and ingested into telemetry.py on exit (SWARM_HOOKS=0 off).
#
# BRIEF.md may be "-" to read the brief from stdin.
# --yolo is always on (no permission prompts): point --cwd at a disposable worktree
# for any task that writes files. See docs/orchestration/cheap-agents.md.
set -uo pipefail

tier="fast"; model=""; turns=30; cwd="$PWD"; log_dir=""; effort=""; agent=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier) tier="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --max-turns) turns="$2"; shift 2 ;;
    --cwd) cwd="$2"; shift 2 ;;
    --log-dir) log_dir="$2"; shift 2 ;;
    --effort) effort="$2"; shift 2 ;;
    --agent) agent="$2"; shift 2 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    -) break ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) break ;;
  esac
done
[[ $# -eq 1 ]] || { echo "usage: $0 [options] BRIEF.md|-" >&2; exit 2; }

if [[ -z "$model" ]]; then
  case "$tier" in
    fast) model="deepseek/deepseek-v4.1-flash-fast" ;;
    code) model="deepseek/deepseek-v4.1-flash"; effort="${effort:-high}" ;;  # supports off/low/high/max; high since 2026-10-10 (max read-looped)
    pro) model="xiaomi/mimo-v2.6-pro" ;;  # no adjustable effort: --effort makes cmd fail
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
# Every worker joins the swarm board: the job bus if SWARM_BUS is set, else the persistent lobby.
agent="${agent:-$name}"
bus_dir="${SWARM_BUS:-$repo_root/docs/orchestration/board}"
proto="$(sed -e "s|{AGENT}|$agent|g" -e "s|{BUS}|$bus_dir|g" -e "s|{REPO}|$repo_root|g" "$repo_root/docs/orchestration/prompts/bus-protocol.md")"
brief="$proto"$'\n\n'"$brief"
export SWARM_AGENT="$agent" SWARM_BUS="$bus_dir"
if [[ -n "${SWARM_BUS_ANNOUNCE:-1}" && "$bus_dir" != "$repo_root/docs/orchestration/board" ]]; then
  python3 "$repo_root/scripts/agents/bus.py" --bus "$bus_dir" --as "$agent" post --to chief --kind info --body "joined (brief: $name)" >/dev/null || true
fi
log_dir="${log_dir:-$repo_root/data/swarm/cmd-logs}"
mkdir -p "$log_dir"
log="$log_dir/$(date +%Y%m%d-%H%M%S)-$name-${model//\//_}.ndjson"

args=(-p "$brief" -m "$model" --yolo -t --skip-onboarding --no-session --no-auto-update
      --max-turns "$turns" --output-format json)
[[ -n "$effort" ]] && args+=(--effort "$effort")

# Own session per worker (macOS has no setsid(1)) so stall_watch --kill can signal its whole group
# without touching sibling workers launched from the same shell. Background jobs inherit SIG_IGN for
# INT/QUIT, so restore the defaults; exec keeps the pid (registered pid == cmd's pid).
launch='import os, signal, sys
try:
    os.setsid()
except OSError:  # already a group leader (job control on): stall_watch falls back to os.kill
    pass
signal.signal(signal.SIGINT, signal.SIG_DFL)
signal.signal(signal.SIGQUIT, signal.SIG_DFL)
try:
    os.execvp(sys.argv[1], sys.argv[1:])
except FileNotFoundError:
    sys.exit(127)'
# The worker is outside our process group: forward Ctrl-C / timeouts / group kills so it never orphans.
# Armed before the fork; a signal landing before $pid is set just exits (no worker yet... or a
# same-instant one, which the registry still shows to stall_watch).
pid=""
fwd() { [[ -n "$pid" ]] || exit 143; kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null; }
trap fwd TERM INT HUP
(cd "$cwd" && exec python3 -c "$launch" cmd "${args[@]}") > "$log" 2>&1 &
pid=$!
# Supervision + telemetry hooks (best effort; SWARM_HOOKS=0 disables). Job = <job> for SWARM_BUS=<job>/bus.
job_dir=""; bus_dir="${bus_dir%/}"
[[ "${SWARM_HOOKS:-1}" != 0 && "$bus_dir" == */bus && "$bus_dir" != "$repo_root/docs/orchestration/board" ]] \
  && job_dir="$(dirname "$bus_dir")"
sw="$repo_root/scripts/agents/stall_watch.py"; tm="$repo_root/scripts/agents/telemetry.py"
task="$name"; [[ "$name" == stdin ]] && task="$agent"
hook() { "$@" >/dev/null 2>&1 || echo "[cmd-task] WARN hook failed: ${*:2:2}" >&2; }
[[ -n "$job_dir" && -f "$sw" ]] && hook python3 "$sw" register-start --job "$job_dir" --agent "$agent" \
  --pid "$pid" --log "$log" --cwd "$(cd "$cwd" && pwd)"
wait "$pid"; rc=$?
while kill -0 "$pid" 2>/dev/null; do wait "$pid"; rc=$?; done  # wait returns early on a trapped signal
trap - TERM INT HUP
[[ -n "$job_dir" && -f "$sw" ]] && hook python3 "$sw" register-end --job "$job_dir" --agent "$agent" \
  --pid "$pid" --rc "$rc"
# --force: the log is complete (cmd exited), the < 120 s freshness guard is for manual ingests.
[[ -n "$job_dir" && -f "$tm" ]] && hook python3 "$tm" ingest-cmd-log --job "$(basename "$job_dir")" \
  --task "$task" --agent "$agent" --model "$model" --auto-round --force "$log"

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
