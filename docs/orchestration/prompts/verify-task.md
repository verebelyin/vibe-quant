<!--
Render:
  mkbrief.py --job <job> --id ver-<slug>-<n> --template docs/orchestration/prompts/verify-task.md \
    --worktree <abs merged job branch> --scope '<globs>' \
    --set-file EXTRA_CHECKS=<exactness/UI lines, or a file holding "none"> \
    --append docs/orchestration/prompts/board-protocol.md
-->
ROLE: verifier "{ID}", job {JOB}
{TELEMETRY}

WORKTREE: `{WT}` (the merged job branch). Your session starts in the main checkout: edit nothing there. Prefix EVERY shell command with `cd {WT} && `. Run Python as `{RUN_AS} ...`.

Your role definition is /Users/verebelyin/projects/vibe-quant/.claude/agents/verifier.md: read it first and follow it.

GOAL: run every gate this brief lists and report only what the commands printed. Fix nothing.

SCOPE (the only files the diff may touch): {SCOPE}

GATE (may exceed the 600 s foreground cap: log it in the background and wait per CLAUDE.md § Shell): `{GATE}`
NEVER end your turn while a command you started is still running: no "waiting" handoff, no verdict before its last line.

EXTRA CHECKS (run each that applies, waited for the same way):
{EXTRA_CHECKS}

REPORT: per gate the verbatim result line (first failure plus summary) and PASS/FAIL/UNVERIFIED; every claim in the maker handoff re-checked as VERIFIED or REFUTED with output.

RETURN ≤ 80 words: status, GATE line, handoff path. Handoff (`handoff_lint.py --template reviewer`): `status: done` only when every applicable gate passed; `role: verifier ({ID})`; `verdict: PASS | SHOULD-FIX | BLOCK`; lint it with `handoff_lint.py --role reviewer <file>`.
