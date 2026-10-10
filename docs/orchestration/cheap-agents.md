# Cheap agents: Command Code as T3 subagents

[Command Code](https://commandcode.ai) serves open models (DeepSeek, Qwen, GLM, Kimi, MiniMax, MiMo, …) for a fraction of Claude's price. Its models are **cheap and literal**: fast at well-specified mechanical work, weak at judgement, at keeping scope, and at noticing they're wrong. Brief them like a new contractor. Every Command Code output is a maker artifact and needs a checker.

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
4. **Follow-up / fix rounds:** call `delegate_task` again with the original brief + the checker's findings verbatim + anything unresolved, under a new `clientRequestId` (`cc-<slug>-2`, …). Never `t3_thread_send` to the `childThreadId` (it is backing storage, not a conversation target).
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
- **Telemetry:** record each finished round yourself: `scripts/agents/telemetry.py record --job <job> --task <slug> --agent <slug> --runtime cmd --model <slug> --round N --outcome done|failed|stalled|error [--review PASS|SHOULD-FIX|BLOCK] [--wall-s S] [--bead <id>]`.

### Legacy: `scripts/agents/cmd-task.sh`

The headless CLI wrapper (`cmd -p … --yolo`, NDJSON logs, stall_watch registration) still exists for manual one-offs outside a swarm. Don't use it for swarm dispatch, and never as a fallback when `delegate_task` fails.

Each Command Code call carries ~17–20k input tokens of system-prompt overhead even for a one-word answer, so batch tiny items into one brief (e.g. 20 vault files per brief, not 1).

## Tiers

Model slugs as listed by `orchestrator_capabilities` for `commandcode_command` on 2026-10-10 (case matters; re-check the catalog when a slug fails).

| Tier | Default model slug | Alternates | Use for |
|---|---|---|---|
| `fast` | `deepseek/deepseek-v4.1-flash-fast` (`options: {"effort": "high"}`) | `Qwen/Qwen3.8-Flash`, `z-ai/glm-5.3-flash` | single-file mechanical edits, bulk classification/extraction to JSON, log summarisation |
| `code` | `deepseek/deepseek-v4.1-flash` (no options advertised in the T3 catalog; user 2026-10-10: "cheaper and better than MiMo", high not max — max may cause the read-loop stall) | `zai-org/GLM-5.3`, `Qwen/Qwen3.8-Max-0902`, `MiniMaxAI/MiniMax-M3` | multi-file but fully specified changes, tests from a written spec, boilerplate |
| `pro` | `xiaomi/mimo-v2.6-pro` (fixed reasoning effort; user 2026-10-08: "better than DeepSeek, ~Sonnet 5.5 level") | `xiaomi/mimo-v2.6-pro-ultraspeed` | harder multi-file / judgement-light coding a Claude implementer would otherwise do; fix rounds that need care |
| `long` | `moonshotai/Kimi-K3` (1M ctx) | `z-ai/glm-5.3-flash`, `xiaomi/mimo-v2.6-pro` | reading a lot to produce a short structured output |

Never `glm-5.3` as the implementer default (user). Qwen 3.8 Max 0902 A/B vs MiMo (2026-10-08, same brief): equivalent correct diff, a peer of MiMo, not the default.

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

Good fits for the cheaper `fast` tier: Pine/FMZ spec triage to JSON, first-draft DSL YAML from a spec (the parser checks it), docstrings, renames across known files, fixture/table generation, journal entry formatting from a handoff, summarising run logs.

**Escalation:** a `fast` task that fails its check twice moves to `code`. A DeepSeek `code` task gets at most 2 rounds; then the Sonnet `implementer` takes it with all findings and failure output. (Re-routing failed work to Sonnet is the defined escalation, not a provider fallback for a `delegate_task` error — those go to the user verbatim.)

## Writing the brief

Start from [`prompts/cmd-task.md`](prompts/cmd-task.md). Cheap models follow a brief literally, so:

- **Self-contained.** Paste the facts the model needs: file paths, the schema, 1–2 worked examples, the 2–3 gotchas that apply. Keep CLAUDE.md out of its reading list; at ~19KB it's mostly noise for a narrow task.
- **Worktree first.** Absolute worktree path and the `cd <worktree> &&` rule (see § Workspace).
- **Positive scope.** "Edit only `a.py` and `tests/test_a.py`" rather than a list of forbidden things.
- **Exact verification command** the agent must run before it answers, plus the expected result.
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
