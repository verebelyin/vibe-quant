<!--
Render:
  mkbrief.py --job <job> --id cc-<slug>-<n> --template docs/orchestration/prompts/cmd-task.md \
    --worktree <abs wt> --scope 'a.py,tests/unit/test_a.py' \
    --set GOAL=... --set-file FACTS=<file> --set-file STEPS=<file> --set DONE_WHEN=... --set PEERS=... \
    --new NAME[,NAME] --facts-from <bead> --append docs/orchestration/prompts/bus-protocol.md
-->
ROLE: maker "{ID}", job {JOB}
{TELEMETRY}

WORKTREE: `{WT}`. Your session starts in the main checkout: edit nothing there. Prefix EVERY shell command with `cd {WT} && `. Run Python as `{RUN_AS} ...`.

GOAL: {GOAL}

SCOPE (the only files you may edit or create): {SCOPE}

FACTS (what you need, so you need not search):
{FACTS}

STEPS:
{STEPS}

DONE WHEN: {DONE_WHEN}

RULES
- Test-first. Keep the red run for the handoff. Existing file: `git stash push -- <fixed file>`, new tests red (one that passes on base needs a reason), `git stash pop`; new module: also one probe of the old behaviour; docs-only or pure move/rename: an `rg` probe before/after; test-only: a mutant of the pinned production line as the red run.
- Tests with teeth: every comparison you add in money code gets a test at each equality edge; a value your code forwards (series, key, class name) is tested through the public entry point and over two unrelated inputs with literal expected values; anything asynchronous is tested by polling its final status; a new log call in a money-path `except` gets a test that an exception-text marker is absent from `caplog.text` (no `exc_info`); a volume-threshold gate gets full-volume fixture rows that cross it, then a rerun. Keep every step that can raise a skip error inside the same try as its siblings.
- Before the gate, run every test file using what you changed: `rg -l '<symbol>' tests/unit` (and `frontend/src --glob '*.test.ts'`). Run them in one serial call (`-p no:randomly`).
- Self-check (cap 3 mutants/round): STEPS-named mutants: `SWARM_AGENT={ID} SWARM_BUS={BUS} scripts/agents/mutate.py --spec '<JSON [{name,file,old,new}]>' -- <targeted pytest>`; else `--auto-diff <base> --max 3`. Kill or justify each SURVIVED.
- Shell: begin with `set -o pipefail;`; pytest, exactness, mutate and the gate need a tool timeout ≥ 600000 ms; Main-tree tests, and reads of files you changed, FAIL `stale-reads`.
- E2E: iterate with `-g`, full spec once. <!-- if frontend -->
- A sibling's API missing on your base: call it via one documented `# type: ignore[attr-defined]`, stubbed in tests.
- Frontend: vitest runs `src/**/*.test.ts` in node (no jsdom): test pure helpers; `./node_modules/.bin/biome check --write <files>`; `pnpm lint` prints no `Found N errors`. <!-- if frontend -->
- Never `git add -f`; never push.
- Commit WIP after each green step (`wip({ID}): <step>`); finish with one commit whose message follows docs/orchestration/prompts/commit-message.md, with no `Co-authored-by` line.
- LAST command, once per round: `{GATE}`. It must print GATE PASS. On FAIL read the LOG it prints, fix, re-run only the failing targeted test, then the gate once more. Unclear FAIL row: `ask --to chief` with it; the gate's source is not your task.

RETURN ≤ 80 words: status, GATE line, handoff path, open questions; start from `handoff_lint.py --template maker` (red run + GATE + LOG in `evidence:`).
