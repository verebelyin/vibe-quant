<!--
Brief template for a Command Code T3 subagent (delegate_task `task` field; see docs/orchestration/cheap-agents.md).
Save the filled brief to data/swarm/<job>/prompts/<slug>.md for the ledger, then pass its full text as `task`.
Fill every <...>, delete these comments. Keep it self-contained: the child sees ONLY this text — none of the chief's conversation.
-->
ROLE: Command Code worker (agent id "<slug>", bead <bead-id>)

WORKSPACE (mandatory): work only inside the git worktree `<absolute worktree path>` (branch `<branch>`).
Your session starts in a different directory (the main checkout) — do not edit anything there.
Prefix EVERY shell command with `cd <absolute worktree path> && `. Run Python as
`cd <absolute worktree path> && PYTHONPATH=$PWD /Users/verebelyin/projects/vibe-quant/.venv/bin/python ...`.

GOAL: <one sentence: the exact change or extraction>

FILES YOU MAY EDIT OR CREATE (inside the worktree): <exact paths>. Leave every other file unchanged.
FILES TO READ FIRST: <exact paths, the fewest that suffice>

FACTS YOU NEED:
- <pasted schema / signature / convention>
- <the 1–3 gotchas that apply>

EXAMPLE OF A CORRECT RESULT:
<a short worked example of the output or the edit>

STEPS:
1. <step>
2. <step>
3. Run this check: `cd <absolute worktree path> && <command>`. Expected: <result>. If it fails, fix and re-run, at most 3 times.
4. Lint: `cd <absolute worktree path> && /Users/verebelyin/projects/vibe-quant/.venv/bin/ruff check <files> && /Users/verebelyin/projects/vibe-quant/.venv/bin/mypy <files>` → 0 errors.

DONE WHEN: <checkable criterion: command + expected output>.

GIT: <either "Do not commit." or "Commit in the worktree: `git add <paths> && git commit -m '<msg> (<bead-id>)'`">. Never push, never touch main.

<bus-protocol block from docs/orchestration/prompts/bus-protocol.md, placeholders filled: {AGENT}=<slug>, {BUS}=<abs job bus dir>, {REPO}=/Users/verebelyin/projects/vibe-quant, {PEERS}=<parallel agents: id: scope; ...>>

FINAL REPLY: exactly one line `DONE <output path or commit sha>` or `FAILED <one-line reason>` — OR, for implementation tasks,
the handoff contract (status / artifacts / evidence with commands + verbatim output / claims / open / comms).
