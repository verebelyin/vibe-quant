<!--
Brief template for a Command Code T3 subagent (delegate_task `task` field; see docs/orchestration/cheap-agents.md).
Save the filled brief to data/swarm/<job>/prompts/<slug>.md for the ledger, then pass its full text as `task`.
Fill every <...>, delete these comments. Keep it self-contained: the child sees ONLY this text — none of the chief's conversation.
-->
ROLE: Command Code worker (agent id "<slug>", bead <bead-id>), job <job id>
TELEMETRY: task=<task> round=<n>

WORKSPACE (mandatory): work only inside the git worktree `<absolute worktree path>` (branch `<branch>`).
Your session starts in a different directory (the main checkout) — do not edit anything there.
Prefix EVERY shell command with `cd <absolute worktree path> && `. Run Python as
`cd <absolute worktree path> && PYTHONPATH=$PWD /Users/verebelyin/projects/vibe-quant/.venv/bin/python ...`.

GOAL: <one sentence: the exact change or extraction>

FILES YOU MAY EDIT OR CREATE (inside the worktree): <exact paths>. Leave every other file unchanged.
FILES TO READ FIRST: <exact paths + line ranges, the fewest that suffice — read them all in ONE turn>

SPEED RULES (measured: one model turn costs 4–10 s; the full test suite costs ~3 min serial, ~1 min under `-n auto --dist worksteal`):
- BATCH: your first turn issues every read in FILES TO READ FIRST as parallel tool calls. After that, before each turn list what else can run now and send it in the same turn (independent reads, greps, test files). Target ≥ 2 tool calls per turn; measured jobs averaged 1.1, with ~90% single-tool turns (jobs 20261010-backlog, 20261011-testspeed, 20261011-grind). Read whole functions or the given line ranges, not 70-line slices. No todo lists.
- RED RUN: for a new module or symbol the first red run may be a collection `ImportError`; then ALSO paste a one-command probe of the pre-fix behaviour (the real gap). In a test-only fix round, mutate the guard line the new test pins (it is unchanged by the round) and paste that red run. In a test-DELETION round, mutate the fixture of the surviving reference test that produces the failure, and say so in the handoff (cc-fu-r42). When the bead's own failing test is already green on today's tree, the pre-fix probe of the gap IS the red evidence: say so and do not hunt for a live failure (cc-sf-gatesel).
- ORDER: make the final verbose commit, then run swarm-check exactly once as the last command (the gate PASS line goes in the handoff, not in the commit body). Scope "before/after" greps to the files you may edit, or name known out-of-scope hits.
- GATE ONCE: never run `pytest tests/unit` yourself. Run swarm-check exactly once per round, as the last command before commit/handoff, and only after targeted tests + ruff + mypy are green. If it fails, fix and re-run only the failing targeted test, then run swarm-check once more. Do not re-run it after a rebase or to double-check a cached PASS. (Full-suite/gate time stayed ~21 min per job across three jobs; one maker ran it 7 times.)
- `--scope` takes ONE comma-separated argument: `--scope 'vibe_quant/dsl/**,tests/unit/dsl/**'`.
- The gate is main's `scripts/agents/swarm-check.sh <worktree>`. Run it with the job bus so it can refuse a hand-back with an unread chief DM: `swarm-check.sh <worktree> --scope '...' --bus <job bus dir> --bus-agent <agent id>` (`--bus`/`--bus-agent` also read `$SWARM_BUS`/`$SWARM_AGENT`; the check FAILs on a bus dir that does not exist, on an agent without a bus, or on a corrupt cursor; PASSes on an existing empty bus; SKIPs only when nothing is configured or when only a bus is given; the gate never creates a bus dir). Only when your task changes the gate itself (`scripts/agents/swarm_check.py`), prove it with the worktree's copy: `<worktree>/scripts/agents/swarm-check.sh <worktree>`.
- Chain checks in one shell call: `ruff check && mypy && <targeted pytest>`.
- Run `bus inbox` immediately before `git commit` as well as before each major step: the gate refuses a hand-back while a chief DM is unread (cc-fc-t1: the R42 DM arrived after the red run and cost an edit+test cycle).
- Before the gate, run EVERY test file that calls what you changed: `rg -l '<function_or_class>\(' tests/unit` and run them all, not "the most relevant" (cc-fc-topk-dedupe r1: gate PASS while tests/unit/test_symbol_agg.py was red through DiscoveryPipeline.run).
- Mutants: apply with a python `str.replace` helper, assert the file changed, restore it; never `sed -i` (macOS BSD sed fails silently). Anchor on a unique token when the target line appears twice (cc-fc-t9). Restore by replacing the MUTATED text back with the original, asserting that count is 1 BEFORE writing; never anchor the restore on a short token like `False` (cc-sf-migrate: the restore matched 3 sites, aborted, and left the mutant in the worktree). A mutant on a redundant guard behind a primary one survives by construction: mutate the primary, name the redundant one in the handoff (cc-fc-t6a). After a mutant run that touches path resolution, delete gitignored artifacts it wrote (cc-fc-t6a: data/portfolio/ in the worktree).
- A sibling task's API that is not on your base yet: call it as designed through ONE documented `# type: ignore[attr-defined]`, stub it in your tests, and do not rebase yourself; the chief merges siblings (cc-fc-t2b).
- E2E: iterate with `playwright test <spec> -g "<scenario>"`; run the full spec once at the end.
- Tests of anything launched asynchronously (API returns 201/running, work happens in a subprocess) must poll the run's FINAL status and fail on `failed` — a 201 proves nothing.
- Mutation self-check: at most 3 mutants, only on the lines you changed, each run against the targeted test file only.
- Every comparison you add in money code (`<`, `<=`, `a <= x < b`, strict ordering) gets a test AT each equality edge, so flipping `<`↔`<=` fails a test. Job 20261015-honest-safe: V1a and V1b each needed an extra review round only because a `<`↔`<=` mutant survived.
- New test files must satisfy ruff flake8-type-checking (TCH001/2/3): type-only imports get `# noqa: TCH00x`, as `tests/unit/api/test_liquidation_launch.py` does (cc-hl-lev-t2: 3 TC errors).
- Tests that patch a collaborator teardown also uses (event writer, client): restore it before calling shutdown/teardown, or the teardown fails for an unrelated reason (cc-sb-alertwrite: 3 of 4 new tests went red at `_shutdown`).
- Secret-leak tests: use leak markers that contain none of the redacted names (`ZZLEAK9`, not `BEARERSECRET9`), or a mutant that drops a name survives (impl-sb-watchlog-2).
- Your file-read tool may be scoped to the main checkout: read WORKTREE files with the shell (`cat`/`sed -n`/`rg`), never the main checkout's copy (cc-sz-settings: a refused read cost a turn).
- Frontend tests: vitest includes only `src/**/*.test.ts` in a node env (no jsdom): write pure-function `.test.ts` tests, never `.test.tsx` render tests (cc-sz-sweepui: a `.test.tsx` was never collected).
- Frontend edits: format your NEW/edited files before committing (`./node_modules/.bin/biome check --write <your files>`, never on files you did not touch); biome counts format diffs as lint errors (cc-sb-exitui).
- Frontend edits: the gate fails only on NEW lint diagnostics (worktree errors > main errors; main baseline 77 errors on 2026-10-14, bead vibe-quant-mi71m). Target no new diagnostics (cc-hl-lev-t3).

FACTS YOU NEED (exact code excerpts / line ranges, so you don't have to search):
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

GIT: <either "Do not commit." or "Commit WIP after each green step (`git commit -m 'wip(<slug>): <step>'`; the chief squashes at Land, and a quota kill then loses nothing), and finish by committing in the worktree with a verbose message in the format of docs/orchestration/prompts/commit-message.md (Why / What / How / Decisions / Verification): `git add <paths> && git commit -F <msgfile>`">. Never push, never touch main.

<bus-protocol block from docs/orchestration/prompts/bus-protocol.md, placeholders filled: {AGENT}=<slug>, {BUS}=<abs job bus dir>, {SCOPE_PATHS}=<the task's SCOPE files/globs, space-separated, globs quoted>, {REPO}=/Users/verebelyin/projects/vibe-quant, {PEERS}=<parallel agents: id: scope; ...>>

FINAL REPLY: exactly one line `DONE <output path or commit sha>` or `FAILED <one-line reason>` — OR, for implementation tasks,
the handoff contract (status / artifacts / evidence with commands + verbatim output / claims / open / comms).
