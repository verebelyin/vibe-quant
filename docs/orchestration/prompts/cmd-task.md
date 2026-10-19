<!--
Maker brief template (DeepSeek or Claude). Render with scripts/agents/mkbrief.py:
  mkbrief.py --job <job> --id cc-<slug>-<n> --template docs/orchestration/prompts/cmd-task.md \
    --worktree <abs wt> --scope 'a.py,tests/unit/test_a.py' \
    --set GOAL=... --set-file FACTS=<file> --set-file STEPS=<file> --set DONE_WHEN=... --set PEERS=... \
    --append docs/orchestration/prompts/bus-protocol.md
Writes prompts/<id>.md, prints the dispatch text; the child sees only it.
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
- Test-first. Keep the red run for the handoff. For a fix to an existing file: `git stash push -- <fixed file>`, run the new tests (red), `git stash pop`. A new module's ImportError red run also needs a one-command probe of the old behaviour. Docs-only work: an `rg` probe before/after instead of a test.
- Tests with teeth: every comparison you add in money code gets a test at each equality edge; a value your code forwards (series, key, class name) is tested through the public entry point and over two unrelated inputs with literal expected values; anything launched asynchronously is tested by polling its final status; a new log call in a money-path `except` gets a test that an exception-text marker is absent from `caplog.text` (no `exc_info`); a volume-threshold gate gets full-volume fixture rows that cross it, then a rerun. Keep every step that can raise a skip error inside the same try as its siblings.
- Before the gate, run every test file that uses what you changed: `rg -l '<symbol>' tests/unit` (and `frontend/src --glob '*.test.ts'`). Run them in one serial call (`-p no:randomly`).
- Self-check with ≤ 3 mutants on lines you changed: `scripts/agents/mutate.py --file F --old '...' --new '...' -- <targeted pytest>`, in a turn of its own.
- Shell: begin with `set -o pipefail;` (`x | tail` hides x's exit code); pytest, exactness, mutate and the gate need a tool timeout ≥ 600000 ms (30 s default kills them); macOS has no `timeout`. Chain checks: `.venv/bin/ruff check && .venv/bin/mypy && <targeted pytest>`. E2E: iterate with `-g`, full spec once.
- A sibling's API missing on your base: call it via one documented `# type: ignore[attr-defined]`, stubbed in tests.
- Frontend: vitest runs `src/**/*.test.ts` in node (no jsdom): test pure helpers; `./node_modules/.bin/biome check --write <files>`; `pnpm lint` prints no `Found N errors`.
- Never `git add -f`; never push; never touch main.
- Commit WIP after each green step (`wip({ID}): <step>`); finish with one commit whose message follows docs/orchestration/prompts/commit-message.md.
- LAST command, once per round: `{GATE}`. It must print GATE PASS. On FAIL read the LOG it prints, fix, re-run only the failing targeted test, then the gate once more.

RETURN ≤ 80 words: status, GATE line, handoff path, open questions; the full contract (docs/orchestration/README.md#handoff-contract, red run + GATE in `evidence:`) lives only in that file.
