# Cheap agents: Command Code as T3 subagents

[Command Code](https://commandcode.ai) serves open models (only DeepSeek V4.1 Flash at effort high is allowed here) for a fraction of Claude's price. Its models are **cheap and literal**: fast at well-specified mechanical work, weak at judgement, at keeping scope, and at noticing they're wrong. Brief them like a new contractor. Every Command Code output is a maker artifact and needs a checker.

**Since 2026-10-10 (user): Command Code tasks are dispatched as T3 subagents through the `t3-code` MCP server (`delegate_task`), never through the built-in Agent/Task tool and not through `scripts/agents/cmd-task.sh`.** They then show up as subagents of the chief's thread.

## Dispatch protocol (follow exactly)

1. **Find the provider.** Call `orchestrator_capabilities` (may appear as `mcp__t3_code__orchestrator_capabilities` / `mcp__t3-code__orchestrator_capabilities`). Pick the provider instance whose `driverKind` is `commandcode` (2026-10-10: `providerInstanceId` = `commandcode_command`) and a model slug **from that instance's `models` list** (see § Tiers).
   - Not listed, `canRunChildTask: false`, or a constraint says unavailable / not signed in → **stop and tell the user. Do not fall back to another provider** (not Claude, not Codex, not `cmd-task.sh`).
2. **Delegate each subtask** with `delegate_task`:
   - `task`: the full brief from [`prompts/cmd-task.md`](prompts/cmd-task.md). The child sees nothing of the chief's conversation: goal, files, constraints, worktree, verification command, what "done" means.
   - `target`: `{ "providerInstanceId": "commandcode_command", "model": "<slug>" }`. Add `options` only when the catalog advertises them for that model (e.g. `deepseek/deepseek-v4.1-flash-fast` has `effort` off/low/high/max).
   - `mode`: `"async"`.
   - `clientRequestId`: unique and stable per subtask round, `cc-<slug>-<round>` (e.g. `cc-d4-logdir-1`). Reuse the same id **only** to retry that same call.
   - Optional: `title` (short, shown in the UI) and `role` (`implementation` / `test` / `research` / …).
3. **Track.** Record every returned `taskId` (and `childThreadId`) in the job ledger row. A finished async child wakes the chief's thread with a notification — end the turn instead of polling. Use `task_status` only when a result is needed mid-turn, never in a tight loop. `task_cancel` **only when the user asks**.
4. **Follow-up / fix rounds:** call `delegate_task` again with the original brief + the checker's findings verbatim + anything unresolved, under a new `clientRequestId` (`cc-<slug>-2`, …). Never `t3_thread_send` to the `childThreadId` (it is backing storage, not a conversation target). Never widen a running round's scope by bus DM: the child may not read it (job 20261012-nextsteps: cc-d3-4a round 3 finished 'test-only' and missed a DM'd code fix) — dispatch a new round with the change in the brief.
5. **Never** use `t3_thread_launch` or `create_threads` for these: those create separate top-level threads that don't show as subagents.
6. **Errors:** if `delegate_task` returns an error, show the user the error code and message **word for word**. Don't work around it (no retry on another provider, no `cmd-task.sh`).
7. **When all tasks finish:** summarise what each Command Code subagent did, then check its work yourself (diff scope, re-run its verification command, then the normal checker per § Checking) before reporting to the user.

### Workspace — the child has no cwd parameter

`delegate_task` children run in the chief's thread workspace (normally the **main checkout**) with the inherited runtime mode (full access). So every brief must:

- name the absolute worktree from `scripts/agents/worktree.sh <slug> [base]` and say: *work only inside `<abs worktree>`; prefix every shell command with `cd <abs worktree> &&`*;
- give the worktree run line (`PYTHONPATH=$PWD /Users/verebelyin/projects/vibe-quant/.venv/bin/python …` from the worktree cwd — `python -m` from the main checkout ignores `PYTHONPATH`, CLAUDE.md #7);
- forbid edits outside the worktree. After the task, check `git -C /Users/verebelyin/projects/vibe-quant status --short` is unchanged as part of the review.
- note: the Command Code runtime itself writes `.commandcode/taste/taste.md` (learned preferences) into the workspace it starts in — the main checkout. It is gitignored; it is not an edit by the model (smoke test `cc-t3-smoke-1`, 2026-10-10).

### Swarm bus, stall watch, telemetry

`cmd-task.sh` used to do these automatically; with T3 the chief does them:

- **Bus:** paste the block from [`prompts/bus-protocol.md`](prompts/bus-protocol.md) (placeholders filled, agent id = the `clientRequestId` slug) into the brief, same as for Claude subagents.
- **Stalls:** `scripts/agents/stall_watch.py` only tracks `cmd-task.sh` processes, so it doesn't see T3 children. Supervise through `task_status` (`workState`) when a child's notification is overdue (> ~45 min for a code-tier task) and through `git -C <worktree> status` (edits appearing?). A child with no edits after ~15–20 min is the read-loop stall: ask the user before cancelling (rule 3), then re-delegate the round to a Claude `implementer`.
- **Telemetry:** don't hand-record rounds. `scripts/agents/telemetry.py ingest-runs --job <job>` reads every finished T3 child (Command Code tokens come from its session file, Codex from T3 usage) and every Claude subagent transcript of the job: tokens, wall time, outcome, and the handoff `comms:` line. It is idempotent, so run it before each review verdict and at Land, then `telemetry.py review --job <job> --task <task> --round N --review PASS|SHOULD-FIX|BLOCK`. Put `TELEMETRY: task=<task> round=<n>` in every brief so the run maps to the right task and round. Without it the name heuristic (`cc-<slug>-<n>`, `-rN`, `-fixN`, `-opusN`) applies.

### Legacy: `scripts/agents/cmd-task.sh`

The headless CLI wrapper (`cmd -p … --yolo`, NDJSON logs, stall_watch registration) still exists for manual one-offs outside a swarm. Don't use it for swarm dispatch, and never as a fallback when `delegate_task` fails.

Each Command Code call carries ~17–20k input tokens of system-prompt overhead even for a one-word answer, so batch tiny items into one brief (e.g. 20 vault files per brief, not 1).

## Tiers

Model slugs as listed by `orchestrator_capabilities` for `commandcode_command` on 2026-10-10 (case matters; re-check the catalog when a slug fails).

**Allowed model (user 2026-10-16, supersedes the 2026-10-15 MiMo allowance):** only `deepseek/deepseek-v4.1-flash` at effort `high`, for every Command Code task. MiMo V2.6 Pro is disabled (expensive and ~5x slower; see Changelog 2026-10-16). Difficult tasks (NT lifecycle, money-path asyncio, semantics fixes) go to DeepSeek with a Sonnet `implementer` escalation after one failed round, or straight to a Sonnet `implementer` when the chief judges DeepSeek too risky (write why in the ledger). Never MiMo, Kimi, GLM, Qwen, MiniMax or any other Command Code model. Fix rounds with `fix-route: sonnet` still go to a Claude Sonnet 5.5 `implementer`.

| Tier | Model slug | Use for |
|---|---|---|
| `code` (default) | `deepseek/deepseek-v4.1-flash` (`options: {"effort": "high"}`; user 2026-10-10: "cheaper and better than MiMo", high not max — max may cause the read-loop stall) | every well-specified task: code, tests, mechanical edits, bulk extraction, read-heavy audits |

## Routing

**DeepSeek is the workhorse (user, 2026-10-10: "almost free").** Every coding task with clear acceptance criteria goes to Command Code `code` tier (`deepseek/deepseek-v4.1-flash`) via T3 — including semantics-heavy ones; the checker carries the quality load:

| Step | Who |
|---|---|
| Write the diff (test-first) | DeepSeek via T3 `delegate_task` |
| Pre-review gate | `scripts/agents/swarm-check.sh` (GATE FAIL → straight back to DeepSeek) |
| Review | `reviewer` on **Sonnet** (`reviewer-lite`/Haiku for mechanical diffs; Opus only in design mode) |
| Fix, small findings (`fix-route: deepseek`) | DeepSeek again: new `delegate_task` = original brief + findings verbatim |
| Fix, too many problems (`fix-route: sonnet`: any blocking finding, ≥ 3 should-fix, 2nd failed round, or the maker misunderstood the task) | Sonnet `implementer` with the findings verbatim — not another DeepSeek round |
| Re-review a fix | the same reviewer for DeepSeek fixes; a **fresh** reviewer for a Sonnet fix |

Claude stays the maker only for: open-ended design (`architect`, Opus), research roles, verdicts on money (`overfit-auditor`, `risk-officer`, Opus), and tasks the chief explicitly judges too risky for DeepSeek (say why in the ledger).

**Stalls.** DeepSeek and MiMo share the read-loop stall: re-reading the same few files with 0 edits (MiMo T4 37 min, 2026-10-08; MiMo w2-fix + DeepSeek w4-fix ~2 h each, 2026-10-09). A child with no worktree edits after ~15–20 min → tell the user (cancel only with their OK) and route the task to a Sonnet `implementer`.

A Command Code brief still needs:

1. The exact files to touch or read and the exact output format.
2. Done checkable by a command (test, parser, `jq` schema, `rg` count) or by a quick diff read.
3. A wrong answer caught by the checker before it reaches `main`, the DB or a decision.

Good fits for DeepSeek even when tiny: Pine/FMZ spec triage to JSON, first-draft DSL YAML from a spec (the parser checks it), docstrings, renames across known files, fixture/table generation, journal entry formatting from a handoff, summarising run logs.

**Escalation:** A DeepSeek `code` task gets at most 2 rounds; then the Sonnet `implementer` takes it with all findings and failure output. (Re-routing failed work to Sonnet is the defined escalation, not a provider fallback for a `delegate_task` error — those go to the user verbatim.)

## Writing the brief

Start from [`prompts/cmd-task.md`](prompts/cmd-task.md). Cheap models follow a brief literally, so:

- **Paste the code, don't point at it.** Each DeepSeek turn costs 4–10 s and 69% of its turns used a single tool (job 20261010-backlog). Paste the exact line-numbered excerpts it must change or call, give read lists with line ranges, and tell it to batch reads in one turn and skip todo lists.
- **Self-contained.** Paste the facts the model needs: file paths, the schema, 1–2 worked examples, the 2–3 gotchas that apply. Keep CLAUDE.md out of its reading list; at ~19KB it's mostly noise for a narrow task.
- **Worktree first.** Absolute worktree path and the `cd <worktree> &&` rule (see § Workspace).
- **Positive scope.** "Edit only `a.py` and `tests/test_a.py`" rather than a list of forbidden things.
- **Exact verification command** the agent must run before it answers, plus the expected result.
- **Keep proofs fast.** No long discovery/backtest reruns as per-task proofs (a seeded 878 rerun took ~30 min, user 2026-10-10). Use unit tests, mutants and single fixed-strategy evals (`exactness_239.py`, ~6 s); if a live discovery check is really needed, give tiny settings (1–2 symbols, population 4–6, 1–2 generations, 2–3 month window, ~1–2 min). Full-size reruns happen at most once, in the final verifier pass, and only when necessary.
- **Exact output.** Name the output file and give its format (JSON schema or a filled example). The final reply is one line: `DONE <path>` or `FAILED <reason>`, or the handoff contract when the task is an implementation.
- **Commits:** the child may commit on its own worktree branch when the brief says so (message ends with the bead id). Never push, never touch `main`, no secrets. The orchestrator owns merges.

## Checking Command Code output

The `verifier` (or the orchestrator for tiny tasks) must:

1. `git -C <worktree> diff --stat` (vs the base): only the files in scope changed; `git -C /Users/verebelyin/projects/vibe-quant status --short` unchanged (the child ran with the main checkout as its workspace).
2. Re-run the brief's verification command and check the result.
3. `scripts/agents/swarm-check.sh <worktree> --scope "<glob,glob>"` must say GATE PASS before any reviewer sees it.
4. For bulk JSON: validate every file against the schema (`python3 -c` + `json.load`), and spot-check ~5% by hand against the source. Cheap models fabricate fields when a source is ambiguous.

## Changelog

- 2026-10-10: dispatch moved from `cmd-task.sh` to T3 `delegate_task` subagents (`commandcode_command`), with the user's protocol (async, unique `clientRequestId`, keep `taskId`, no tight polling, no `t3_thread_send` / `t3_thread_launch` / `create_threads`, errors verbatim, no provider fallback, verify before reporting). Tier slugs updated to the T3 catalog — user request.
- 2026-10-10: DeepSeek is the default maker for all coding tasks; Sonnet reviews and fixes on `fix-route: sonnet` (user: "the workhorse should be deepseek because it is almost free").

- 2026-10-11: allowed Command Code models reduced to `deepseek/deepseek-v4.1-flash` + `xiaomi/mimo-v2.6-pro`; `fast`/`long` tiers and all other alternates removed — user: "DONT run Kimi K3" (job 20261011-testspeed).
- 2026-10-11 17:16: Command Code restricted to `deepseek/deepseek-v4.1-flash` effort high (MiMo `pro` tier removed); reviews on GPT-6.1-Sol via Codex, Sonnet/Haiku for implementation and fixes, Opus medium for planning — user rule (cost).
- 2026-10-12: no mid-round scope changes by DM; new round instead — evidence: job 20261012-nextsteps cc-d3-4a-feedrag-3 missed a scope DM (chief).
- 2026-10-15: MiMo V2.6 Pro allowed again for difficult tasks; DeepSeek V4.1 Flash high stays the default; reviews GPT-6.1-Sol high — user (job 20261015-small-beads).
- 2026-10-15 (job 20261015-small-beads): first MiMo V2.6 Pro task after re-allowing it (T5, real-NT test) had 0 worktree edits after ~90 min when a T3 restart cancelled it, so the chief escalated to a Sonnet implementer (done in ~7 min). Correction: the MiMo run then RESUMED by itself, worked in the same worktree in parallel, and reached the same verified result (probes, red run, 26 passed); its 0-edit period was probing in /tmp, not a stall. Check probe output, not only worktree edits, before calling a MiMo stall.
- 2026-10-16 (job 20261016-safety-bugs): DeepSeek V4.1 Flash made 4 tasks (T1 paper, T3 gate, T4 migrations, T5 discovery API) at 1-2 rounds each; 2 needed a Sonnet fix after Sol found a blocking miss (T3 deleted-module edge, T4 cached half-migrated connection). MiMo V2.6 Pro did the real-NT lifecycle test (T2) in ~45 min (first worktree edit ~20 min in), passed Sol first round.
- 2026-10-16 (job 20261016-sizing): MiMo V2.6 Pro disabled — user, after reviewing its record: clean first rounds (safety-bugs T2 Sol PASS; sizing K1 no code defect, 2 test gaps, risk PASS) but ~31-34 min and 1.9-5.2M input tokens per task vs DeepSeek ~7 min, plus 37 min / 2 h / 90 min stalls (10-08, 10-09, 10-15). DeepSeek failures (S2, S4 blocking) were recovered by a ~4 min Sonnet fix. Hard tasks: DeepSeek, Sonnet implementer on escalation.
