<!--
Maker brief template (DeepSeek via T3, or a Claude implementer). Render with scripts/agents/mkbrief.py:
  mkbrief.py --job <job> --id cc-<slug>-<n> --template docs/orchestration/prompts/cmd-task.md \
    --worktree <abs wt> --scope 'a.py,tests/unit/test_a.py' \
    --set GOAL=... --set-file FACTS=<file> --set-file STEPS=<file> --set DONE_WHEN=... --set PEERS=... \
    --append docs/orchestration/prompts/bus-protocol.md
It writes prompts/<id>.md and prints the dispatch text (pointer + TELEMETRY line). The child sees only this text.
-->
ROLE: maker "{ID}", job {JOB}
{TELEMETRY}

WORKTREE: `{WT}`. Your session starts in the main checkout: edit nothing there. Prefix EVERY shell command with `cd {WT} && `. Run Python as `{RUN_AS} ...`.

GOAL: {GOAL}

SCOPE (the only files you may edit or create): {SCOPE}

FACTS (excerpts and conventions you need, so you don't have to search):
{FACTS}

STEPS:
{STEPS}

DONE WHEN: {DONE_WHEN}

RULES
- Test-first. Keep the red run for the handoff. For a fix to an existing file: `git stash push -- <fixed file>`, run the new tests (red), `git stash pop`. A new module's ImportError red run also needs a one-command probe of the old behaviour. Docs-only work: an `rg` probe before/after instead of a test.
- Tests with teeth: every comparison you add in money code gets a test at each equality edge; a value your code forwards (series, key, class name) is tested through the public entry point and over two unrelated inputs with literal expected values; anything launched asynchronously is tested by polling its final status; a new log call in a money-path `except` gets a test that an exception-text marker is absent from `caplog.text` (no `exc_info`); a volume-threshold gate gets full-volume fixture rows that cross it, then a rerun. Keep every step that can raise a skip error inside the same try as its siblings.
- Before the gate, run every test file that uses what you changed: `rg -l '<symbol>' tests/unit` (and `frontend/src --glob '*.test.ts'`). Run them in ONE serial pytest call (`-p no:randomly`) to catch leaked module state.
- Self-check with ≤ 3 mutants on lines you changed: `scripts/agents/mutate.py --file F --old '...' --new '...' -- <targeted pytest>`, in a turn of its own.
- Chain checks: `ruff check && mypy && <targeted pytest>`. E2E: iterate with `-g "<scenario>"`, full spec once.
- A sibling task's API not on your base yet: call it through one documented `# type: ignore[attr-defined]` and stub it in tests; the chief merges siblings.
- Frontend: vitest collects only `src/**/*.test.ts` (node env, no jsdom), so test pure helpers; run `./node_modules/.bin/biome check --write <your files>`; `pnpm lint` must print no `Found N errors`; a new file name must not differ from a sibling only by case.
- Never `git add -f`; never push; never touch main.
- Commit WIP after each green step (`wip({ID}): <step>`); finish with one commit whose message follows docs/orchestration/prompts/commit-message.md.
- LAST command, once per round: `{GATE}`. It must print GATE PASS. On FAIL read the LOG it prints, fix, re-run only the failing targeted test, then the gate once more.

RETURN the handoff contract (docs/orchestration/README.md#handoff-contract), red run and GATE line in `evidence:`.
