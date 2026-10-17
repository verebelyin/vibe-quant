#!/usr/bin/env bash
# Install the vibe-quant git hooks, idempotently, into the shared hooks directory.
#
#   scripts/agents/install-hooks.sh
#
# Hooks live in `<git-common-dir>/hooks`, which the main checkout and every linked
# worktree share, so one run covers both. Two marked blocks are installed:
#
#   pre-commit  ->  scripts/git-hooks/pre-commit-lint.sh   (ruff on staged .py files)
#   commit-msg  ->  scripts/git-hooks/check_commit_msg.py  (message format on `main`)
#
# An existing pre-commit hook is preserved (the beads integration section keeps
# working); the vibe-quant block is appended, or replaced in place when it is already
# present, so running twice changes nothing. Script paths are resolved from the main
# checkout root (the common dir's parent).
set -euo pipefail

common_dir="$(git rev-parse --path-format=absolute --git-common-dir)"
main_root="$(dirname "$common_dir")"
hooks_dir="$common_dir/hooks"
mkdir -p "$hooks_dir"

begin='# --- BEGIN VIBE-QUANT LINT ---'
end='# --- END VIBE-QUANT LINT ---'
lint_script="$main_root/scripts/git-hooks/pre-commit-lint.sh"
msg_script="$main_root/scripts/git-hooks/check_commit_msg.py"

pre_commit_block="$begin
\"$lint_script\" || exit 1
$end"

commit_msg_block="$begin
vq_python=\"$main_root/.venv/bin/python\"
[ -x \"\$vq_python\" ] || vq_python=python3
\"\$vq_python\" \"$msg_script\" \"\$1\" || exit 1
$end"

install_block() {
  local file="$1"
  local block="$2"
  [ -e "$file" ] || : > "$file"
  local merged
  merged="$(mktemp)"
  # git execs the hook directly, so a missing shebang breaks it: prepend one.
  if [ -s "$file" ] && head -n 1 "$file" | grep -q '^#!'; then
    : > "$merged"
  else
    printf '#!/usr/bin/env sh\n' > "$merged"
  fi
  awk -v begin="$begin" -v end="$end" '
    $0 == begin { skip = 1 }
    skip && $0 == end { skip = 0; next }
    !skip { print }
  ' "$file" >> "$merged"
  if [ -s "$merged" ] && [ -n "$(tail -c 1 "$merged")" ]; then
    printf '\n' >> "$merged"
  fi
  printf '%s\n' "$block" >> "$merged"
  if ! cmp -s "$merged" "$file"; then
    cat "$merged" > "$file"
  fi
  rm -f "$merged"
  chmod +x "$file"
}

install_block "$hooks_dir/pre-commit" "$pre_commit_block"
install_block "$hooks_dir/commit-msg" "$commit_msg_block"
