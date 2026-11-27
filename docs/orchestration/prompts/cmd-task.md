<!--
Render:
  mkbrief.py --job <job> --id cc-<slug>-<n> --template docs/orchestration/prompts/cmd-task.md \
    --worktree <abs wt> --scope 'a.py,tests/unit/test_a.py' \
    --set GOAL=... --set-file FACTS=<file> --set-file STEPS=<file> --set DONE_WHEN=... --set PEERS=... \
    --new NAME[,NAME] --facts-from <bead> --append docs/orchestration/prompts/board-protocol.md
-->
ROLE: maker "{ID}", job {JOB}
{TELEMETRY}

WORKTREE: `{WT}`. Your session starts in the main checkout: edit nothing there. Prefix EVERY shell command with `cd {WT} && `. Run Python as `{RUN_AS} ...`.

GOAL: {GOAL}

SCOPE (the only files you may edit or create): {SCOPE}

FACTS (so you need not search):
{FACTS}

STEPS:
{STEPS}

DONE WHEN: {DONE_WHEN}

RULES
- Test-first. Keep the red run for the handoff. Existing file: `git diff -- <file> > /tmp/{ID}/fix.patch && git checkout -- <file>`, new tests red (passing on base needs a reason), `git apply /tmp/{ID}/fix.patch` (no `git stash`: shared); new module: also one probe of the old behaviour; docs-only, rename or test-only: an `rg` probe before/after.
- Tests with teeth: every comparison you add in money code gets a test at each equality edge; a value your code forwards (series, key, class name) is tested through the public entry point and over two unrelated inputs with literal expected values; anything asynchronous is tested by polling its final status; a new log call in a money-path `except` gets a test that an exception-text marker is absent from `caplog.text` (no `exc_info`); a volume-threshold gate gets full-volume fixture rows that cross it, then a rerun. Keep every step that can raise a skip error inside the same try as its siblings.
- Before the gate, run every test file using your changes (`rg -l '<symbol>' tests/unit`, `frontend/src --glob '*.test.ts'`) serially (`-p no:randomly`)
- No test for a function with cognitive complexity ≤ 5 unless CLAUDE.md § Tests prove outcomes exempts it. Mutants: optional, money code only, ≤ 2: `SWARM_AGENT={ID} SWARM_BUS={BUS} scripts/agents/mutate.py --spec '<JSON [{name,file,old,new}]>' -- <targeted pytest>`.
- Shell: begin with `set -o pipefail;`; tool timeout 600000; the gate: background log + wait (CLAUDE.md § Shell); main-tree tests and reads of files you changed FAIL `stale-reads`; scratch copies: a literal `/tmp/<name>`, never `cd "$VAR"`; logs: `/tmp/{ID}/` (no `rm` outside `/tmp`); `ruff format` only your files; read each range once (`sed -n a,bp`).
- E2E: iterate with `-g`, full spec once. <!-- if frontend -->
- Sibling API missing on your base: one documented `# type: ignore[attr-defined]`, stubbed in tests.
- Frontend: vitest runs `src/**/*.test.ts` in node (no jsdom): test pure helpers; `./node_modules/.bin/biome check --write <files>`; `pnpm lint` prints no `Found N errors`. <!-- if frontend -->
- Never `git add -f`, never push. Crash probes: SIGKILL only (macOS crash dialogs).
- Commit WIP after each green step (`wip({ID}): <step>`); the final commit follows docs/orchestration/prompts/commit-message.md, no `Co-authored-by` line.
- LAST command, once per round: `{GATE}` → GATE PASS. On FAIL read its LOG, fix, re-run only the failing test, then the gate again. Unclear FAIL row: ask the chief.

RETURN ≤ 80 words: status, GATE line, handoff path, open questions; start from `handoff_lint.py --template maker` (red run + GATE + LOG in `evidence:`).
