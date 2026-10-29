<!--
Render:
  scripts/agents/mkbrief.py --job <job> --id impl-<slug>-<n> --template docs/orchestration/prompts/fix-task.md --worktree <abs wt> --scope 'a.py,tests/unit/test_a.py' --prior-from <rev handoff> --set-file RULINGS=<file> --set PEERS=... --append docs/orchestration/prompts/bus-protocol.md
ROUND from the id; RULINGS=none when empty. The child sees only this text.
-->
ROLE: fixer "{ID}", job {JOB}, round {ROUND}
{TELEMETRY}

WORKTREE: `{WT}` (the reviewed branch). Start in the main checkout: edit nothing there. Prefix EVERY shell command with `cd {WT} && `. Run Python as `{RUN_AS} ...`.

GOAL: fix every finding below (binding) unless you can DISPROVE one (file:line, test output): then `ask --to chief` with that evidence and await a ruling; never "fix" a finding just to end the round. The chief re-runs the reviewer's repros; no re-review.

PRIOR FINDINGS (reviewer, verbatim):
{PRIOR_FINDINGS}

RULINGS (chief decisions, binding):
{RULINGS}

SCOPE (the only files you may edit or create): {SCOPE}

RULES: follow docs/orchestration/prompts/cmd-task.md. Per finding: a test red on the current code, green after the fix.

GATE, LAST command, once per round (must print GATE PASS): `{GATE}`

RETURN ≤ 80 words: status, GATE line, handoff path. Handoff (docs/orchestration/README.md#handoff-contract): per finding `fixed: <id> <evidence>` (red, green) or `disputed: <id> <evidence>`; per fixed id also `sibling-scan: <id> <rg command> -> <N> sites (<fixed|why not>)`; plus the GATE line.
