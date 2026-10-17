<!--
Fix-round brief for an implementer (round >= 2). Render with scripts/agents/mkbrief.py:
  mkbrief.py --job <job> --id impl-<slug>-<n> --template docs/orchestration/prompts/fix-task.md \
    --worktree <abs wt> --scope 'a.py,tests/unit/test_a.py' --set-file PRIOR_FINDINGS=<file> \
    --set-file RULINGS=<file> --set PEERS=... --append docs/orchestration/prompts/bus-protocol.md
ROUND comes from the id. Use "none" for an empty RULINGS. The child sees only this text.
-->
ROLE: fixer "{ID}", job {JOB}, round {ROUND}
{TELEMETRY}

WORKTREE: `{WT}` (the branch the reviewer judged). Your session starts in the main checkout: edit nothing there. Prefix EVERY shell command with `cd {WT} && `. Run Python as `{RUN_AS} ...`.

GOAL: fix every finding below; they are binding. Keep what was right.

PRIOR FINDINGS (reviewer, verbatim):
{PRIOR_FINDINGS}

RULINGS (chief decisions, binding):
{RULINGS}

SCOPE (the only files you may edit or create): {SCOPE}

RULES: follow the RULES of docs/orchestration/prompts/cmd-task.md. Per finding: a test that is red on the current code, then green after the fix; keep the red run for the handoff.

GATE, LAST command, once per round (must print GATE PASS): `{GATE}`

RETURN the handoff contract (docs/orchestration/README.md#handoff-contract): one `evidence:` item per finding (red run, green run), plus the GATE line.
