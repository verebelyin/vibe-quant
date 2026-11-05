#!/usr/bin/env bash
# Kill automation Chromium whose parent died (ppid 1): a Playwright run killed by a shell
# timeout or a crashed agent-browser leaves one holding a core, and nobody can close it.
# Matches only the Playwright and agent-browser builds (~/Library/Caches/ms-playwright,
# ~/.agent-browser/browsers), never the user's Chrome. crashpad handlers are skipped: a
# live browser's handler has ppid 1 too. ui-check.sh down runs this (e2e.sh through it).
set -uo pipefail
strays="$(ps -axo pid=,ppid=,command= | awk '
  $2 == 1 && ($0 ~ /\/ms-playwright\// || $0 ~ /\/\.agent-browser\/browsers\//) &&
  $0 !~ /chrome_crashpad_handler/ { print $1 }')"
[[ -z "$strays" ]] && exit 0
echo "$strays" | xargs kill 2>/dev/null || true
echo "killed stray automation browser pid(s): $(echo $strays)"
