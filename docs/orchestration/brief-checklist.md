# Brief checklist (chief, before every dispatch)

`mkbrief.py` checks the brief file, TELEMETRY, SCOPE globs, the gate command and reviewer ranges. Below: the rest; each line cost a round when missed.

## Scope: find every file the task must touch
- A helper module with a single consumer goes in the same task as its consumer.
- Tests that build what changes, including monkeypatched doubles and lambdas of any function whose signature changes: `rg -l '<symbol>' tests/`.
- A rename or move: also `rg` the old dotted path as a string (`setattr("pkg.mod._old"`).
- Changed message, help or error text: `rg` the old text in `tests/` and docs; put every hit in SCOPE (the gate's `pinned-text` catches misses a round late).
- A removal: list readers with a word-boundary search, `rg -nP '(?<![A-Za-z0-9_])name\b'`, never `rg name | rg -v kept`.
- A goal that says "every" or "all": `rg` the whole class and list every site in the brief.
- A shared helper whose failure behaviour changes (swallow → raise, new early return): every caller gets a contract row saying what it caches, closes and retries.
- A field the API must carry to another process: the hand-off file goes in SCOPE (`rg -n 'load_config_from_json|argv' vibe_quant`).
- Generated-strategy source changes: `tests/fixtures/compiled_source/*.golden`, `test_compiler_golden_source.py`, `test_audit_dsl_runtime.py`.
- A run flag that changes the run but not its spec hash (lag, placebo, null): list every ledger it writes and its readers; a named test keeps each ledger byte-identical.
- A new archive table: SPEC § Raw Data Archive schema block joins the docs task.
- LIVE: each SCOPE path is imported or called from a live entry point (`rg`), so nobody builds dead code.
- A launch precheck mirroring a subprocess: contract rows for the subprocess's input precedence, each step's error classes and every call site of the mirrored resolver.
- Splitting one change into tasks: assign every string, doc and claim the change invalidates to exactly one task. A new DSL key lands with its codegen, or the compiler raises on it: validated but never emitted is a silent no-op.

## Facts the maker needs
- Paste the excerpts it must change or call, with the file of every type it names. Point at design sections by heading and code by symbol, never line ranges.
- Name the mechanism a fix must call (found with `rg` at an existing call site) and the fixture builders that already exist (file:symbol, and whether they cover the case).
- Input-domain changes (parser, validator, translator): a table of input classes (numbers, integral floats, numeric strings, bools, None, junk strings, inf/nan, huge ints) with each outcome, plus the frontend mirror.
- Test the real thing with fixtures that can fail: real NT events (`TestEventStubs`), a real TelegramBot over mocked HTTP, the real engine for an engine row; every mode × case cell a row claims, the event inside the gap, more items than the cache holds.
- Name existing tests that pin unchanged contract rows. DONE WHEN reuses the gate's commands (bare `.venv/bin/mypy`).
- Diagnosis briefs list hypotheses per layer crossed.
- When a fact depends on a sibling task landing, say so. Quote design rows and rulings verbatim; when design and FACTS disagree, say which wins.
- Any new `except` returning a fallback under `vibe_quant/` logs above DEBUG (no `exc_info` on money paths): `test_exception_hygiene.py` ratchets silent handlers.
- Two beads in one brief: name the commit order and which files go in which commit.

## Contracts and rulings
- Anything touching fills, funding, fees, liquidation, metrics, fitness, gates or time windows gets the architect's semantics-contract table (boundary rows with expected numbers) before dispatch, pasted into every maker and reviewer brief.
- A mid-task ruling becomes the next numbered row (R7, R7b…) under that task's heading in `brief.md`, with its expected numbers and the round that raised it. More than 2 mid-flight rows on one task means the contract missed a question class: tell the architect.
- A ruling that adds an error names the exception class, payload fields and raise site; one that widens a match predicate (retry on text, ignore on substring) names what must never match.
- Process control (spawning a child, run/job status writes): a contract row per parent/child interleaving and per exception boundary between spawn and registration.
- Parity demands name the exact rows that must match. A row saying note, label or check names the persisted field (`notes.consistency.notes`) and the enforcing layer (API precheck or run).
- Operator alerts/status: venue facts (stop resting, size) only from venue-confirmed data, else "unconfirmed"; status from the current process.

## Fix rounds
- Quote each finding's bullet and the reviewer's FIX text verbatim; reference code by symbol.
- The gate scope is the whole task scope (round 1 plus this round). Stacked task: `--base <parent branch>`; two or more parents: merge them in the new worktree and use `--base <merge sha>`.
- Re-review briefs carry prior findings plus the fix delta, not a fresh hunt.
- After 2 failed rounds: design pass (fold rulings and findings into contract rows; missing row, duplicated logic or unreachable edge?), re-scope. After 3, also change the model or ask the user.
