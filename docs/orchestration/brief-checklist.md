# Brief checklist (chief, before every dispatch)

`scripts/agents/mkbrief.py` already guarantees: the brief file exists before dispatch, the dispatched text carries the TELEMETRY line, every SCOPE glob matches a file (`+path` for new files), the gate command is literal, and reviewer ranges come from `git merge-base`. This list covers what a script can't check; each line cost a round when missed.

## Scope: find every file the task must touch
- A helper module with a single consumer goes in the same task as its consumer.
- Tests that build what changes, including monkeypatched doubles and lambdas of any function whose signature changes: `rg -l '<symbol>' tests/`.
- A rename or move: also `rg` the old dotted path as a string (`setattr("pkg.mod._old"`) and every double of a method whose kwargs change.
- Changed message, help or error text: `rg` the old text in `tests/` and docs; put every hit in SCOPE (the gate's `pinned-text` check catches what you miss, one round late).
- A removal: list readers with a word-boundary search, `rg -nP '(?<![A-Za-z0-9_])name\b'`, never `rg name | rg -v kept`.
- A goal that says "every" or "all": `rg` the whole class and list every site in the brief.
- A shared helper whose failure behaviour changes (swallow → raise, new early return): every caller gets a contract row saying what it caches, closes and retries.
- A field the API must carry to another process: the hand-off file goes in SCOPE (`rg -n 'load_config_from_json|argv' vibe_quant`).
- Generated-strategy source changes: `tests/fixtures/compiled_source/*.golden`, `test_compiler_golden_source.py`, `test_audit_dsl_runtime.py`.
- A new Pydantic response model or `BacktestMetrics` field: `tests/unit/api/test_openapi.py` EXPECTED_SCHEMA_COUNT and `equal_metrics` in `tests/unit/test_screening_liquidation.py`.
- A new archive table: SPEC § Raw Data Archive schema block joins the docs task.
- LIVE: each SCOPE path is imported or called from a live entry point (`rg`), so nobody builds dead code.
- A launch precheck or anything that mirrors a subprocess: list the subprocess's input precedence (overrides such as `parameters.dsl_override`, which loader runs on which input) and every error class each step raises, as contract rows, and every call site of the mirrored resolver.
- Splitting one change into tasks: assign every string, doc and claim the change invalidates to exactly one task.

## Facts the maker needs
- Paste the excerpts it must change or call, with the file of every type it names. Point at design sections by heading and code by symbol, never line ranges.
- Name the mechanism a fix must call (found with `rg` at an existing call site) and the fixture builders that already exist (file:symbol, and whether they cover the case).
- Changes to an input domain (parser, validator, translator): a table of input classes up front — numbers, integral floats, numeric strings, bools, None, non-numeric strings, inf/nan, huge ints — with the intended outcome of each, plus the frontend mirror.
- Test the real thing: NT callbacks get real NT events (`TestEventStubs`); paper alerts a real TelegramBot (mocked HTTP) after a recent ERROR, a 30-cycle flap plus a HALT through a 20/min 429 mock, ending on the current state.
- Name existing tests that already pin unchanged contract rows, so the maker re-runs them.
- Diagnosis briefs list hypotheses per layer crossed.
- When a fact depends on a sibling task landing, say so. When design and FACTS disagree, say which wins.
- Two beads in one brief: name the commit order and which files go in which commit.

## Contracts and rulings
- Anything touching fills, funding, fees, liquidation, metrics, fitness, gates or time windows gets the architect's semantics-contract table (boundary rows with expected numbers) before dispatch, pasted into every maker and reviewer brief.
- A mid-task ruling becomes the next numbered row (R7, R7b…) under that task's heading in `brief.md`, with its expected numbers and the round that raised it. More than 2 mid-flight rows on one task means the contract missed a question class: tell the architect.
- A ruling that adds an error names the exception class, payload fields and raise site; one that widens a match predicate (retry on text, ignore on substring) names what must never match.
- Process-control changes (spawning a child, writing run/job status) get a contract row for every parent/child interleaving and every exception boundary between spawn and registration.
- Parity demands name the exact rows that must match.

## Fix rounds
- Quote each finding's bullet and the reviewer's FIX text verbatim; reference code by symbol.
- A fix that adds an `except` returning a fallback under `vibe_quant/` must log (no `exc_info` on money paths): `tests/unit/test_exception_hygiene.py` ratchets silent handlers.
- The gate scope is the whole task scope (round 1 plus this round). Stacked task: `--base <parent branch>`; two or more parents: merge them in the new worktree and use `--base <merge sha>`.
- Re-review briefs carry prior findings plus the fix delta, not a fresh hunt.
- After 2 failed rounds on one task, stop and do a design pass: fold every ruling and finding into contract rows, then ask whether the remaining findings come from a missing row, duplicated logic or an unreachable edge, and re-scope. After 3, also change the model or ask the user.
