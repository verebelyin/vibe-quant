# Multi-agent orchestration

How one **orchestrator** (a Claude Code session running the [`orchestrate`](../../.claude/skills/orchestrate/SKILL.md) skill) splits large work across specialist agents. This file is the single source for the roster, the brief format, the handoff contract and the workspace layout. Other files cover the rest:

- [`cheap-agents.md`](cheap-agents.md): delegating mechanical tasks to the `cmd` CLI (DeepSeek / Qwen / GLM / Kimi).
- [`research-swarm.md`](research-swarm.md): the strategy-factory lane, plus what we took (and rejected) from the 2026 "AI hedge fund" agent-swarm posts.
- [`prompts/`](prompts/): brief templates for `cmd` tasks.
- Agent definitions: [`.claude/agents/*.md`](../../.claude/agents/). Each file's body is the agent's system prompt.

## Principles

1. **Maker-checker.** Agents never grade their own output. Each artifact has a different agent (ideally on a different model) that can block it. Self-evaluation consistently overrates mediocre work, and this repo has seen it many times: every champion that was forced past the bootstrap-CI gate collapsed in validation.
2. **One voice.** Only the orchestrator talks to the user, commits to `main`, pushes, closes beads and launches paper trading. Subagents return handoffs and nothing else.
3. **Bounded briefs.** Every dispatch is one task with a checkable **done-when** criterion. "Investigate X" is not a brief; "list every caller of X with file:line" is.
4. **Structured handoffs.** Every agent ends with the [handoff contract](#handoff-contract). The orchestrator reads handoffs, not transcripts.
5. **Evidence over claims.** A claim without the command and output that prove it is UNVERIFIED. Before the orchestrator relays a result to the user, it re-runs the decisive command or has a checker do it.
6. **Cheapest capable tier.** Mechanical work goes to `cmd` (cents). Judgement goes to Claude. Final verdicts on money or correctness go to Opus.
7. **Gates block.** A failed gate means the work goes back to its maker, gets escalated, or gets dropped. Lowering a threshold to get something through is never an option (see CLAUDE.md § Verification Rules).

## Roster

| Agent | Lane | Model | Writes | Job | Checked by |
|---|---|---|---|---|---|
| orchestrator | both | session model (Opus) | ledger, merges, beads | decompose, dispatch, integrate, talk to user | user gates |
| `architect` | eng | opus | design doc only | spec → design → task plan with acceptance criteria | user gate |
| `implementer` | eng | sonnet | code + tests, in a worktree | one task, test-first | `reviewer` + `verifier` |
| `reviewer` | eng | opus | nothing | two verdicts: matches spec? well built? | orchestrator |
| `verifier` | eng | sonnet | nothing | runs quality gates + exactness proofs | orchestrator |
| `alpha-scout` | research | sonnet | hypothesis tickets | external sources → falsifiable hypotheses | `strategy-author` (feasibility), `overfit-auditor` |
| `strategy-author` | research | sonnet | DSL YAML, indicator plugins | hypothesis → parser-valid strategy | `overfit-auditor` |
| `backtest-operator` | research | sonnet | DB rows via API/CLI only | runs screening / discovery / validation, reports raw numbers | `overfit-auditor` |
| `overfit-auditor` | research | opus | nothing | skeptic: PASS / REJECT against the repo's gates | `risk-officer` |
| `risk-officer` | research | opus | nothing | veto before any paper/live step, no negotiation | user |
| `cmd` worker | both | DeepSeek / Qwen / GLM / Kimi | scoped files, in a worktree | mechanical bulk work | `verifier` or a deterministic check |

Model choice follows maker-checker: the agents that write run on sonnet or cheap models, and the checkers run on opus, so writer and checker share fewer blind spots. Override per dispatch with the Agent tool's `model` parameter when a task is harder or easier than usual.

## Dispatch

| Runtime | When | How |
|---|---|---|
| Claude subagent | default for judgement work | Agent tool, `subagent_type: "<name>"`. Writers work in a worktree from `scripts/agents/worktree.sh` (not `isolation: "worktree"`, which can only branch from `main` and lacks market data). |
| `cmd` worker | mechanical / bulk / well-specified | `scripts/agents/cmd-task.sh` (see [`cheap-agents.md`](cheap-agents.md)) |
| T3 `delegate_task` | other providers (Codex, …) or a long-running background child | `orchestrator_capabilities` → `delegate_task`; paste the agent file's body as the task prompt |

- **Worktrees:** `scripts/agents/worktree.sh <slug> [base-ref]` creates `../vq-<slug>` on `swarm/<slug>`, links the main checkout's `data/catalog` + `data/archive` (so catalog-backed tests and `exactness_239.py` run there) and prints the `run as` line. Paste that line into every worktree brief: the editable install points at the main checkout, so Python run without `PYTHONPATH=$PWD` tests main's code (CLAUDE.md #7).
- **Shared state is single-writer.** `data/state/vibe_quant.db`, the backend on :8000 and the beads Dolt store each have one owner at a time. Only `backtest-operator` launches jobs, only the orchestrator mutates beads, and only one agent at a time restarts servers.
- **WIP limit:** at most 3–5 parallel writers, which is about as many diffs as can be reviewed properly. Readers (scouts, reviewers) can fan out wider.
- New or edited files in `.claude/agents/` load when a session starts. After changing them, restart the session (or use `/agents`) before dispatching.

## Profiles

| Profile | Use when | Design | Build + check |
|---|---|---|---|
| **lean** (default) | ≤ ~500 changed lines, one package, no NT engine lifecycle, fill/metric semantics or DB schema change | orchestrator writes the plan from the bead AC | implementer(s) → `reviewer` → `verifier` |
| **full** | anything bigger, cross-package, or touching the areas above | `architect` → `reviewer` (design mode) → user gate | same as lean |

Measured on job `20261008-discovery-robustness` (2 beads, +~330 lines, full profile): ~335k Claude subagent tokens + ~604k cmd tokens, ~1 h wall. The Opus review found 3 real defects; the architect step added little the bead AC didn't already say, and one design choice (seeded uids) caused a defect only a design review would have caught cheaply.

## Brief format

Every dispatch prompt, for any runtime, has these fields:

```
ROLE: <agent name>
GOAL: <one sentence>
CONTEXT: <bead id, files, run ids, prior handoff paths; paste the decisive facts instead of saying "see above">
SCOPE: may touch <paths>; leave everything else unchanged
DONE WHEN: <checkable criterion: command + expected result, or an exhaustive list to produce>
BUDGET: <max turns / time; what to do when exhausted>
RETURN: handoff contract (docs/orchestration/README.md#handoff-contract)
```

The subagent can't see the orchestrator's conversation. Anything it needs has to be in the brief.

## Handoff contract

The last message of every agent:

```
## Handoff
status: done | blocked | failed | needs-decision
role: <agent>
task: <one line>
artifacts:
  - <path | run id | bead id | commit sha | worktree path+branch>
evidence:
  - `<command>` → <decisive output line, verbatim numbers>
claims:
  - <claim> — VERIFIED (<how>) | UNVERIFIED (<why not>)
open:
  - <follow-up, suspected bug, decision needed; one line each>
```

- `blocked` means a missing input or a failed precondition. Name it.
- `needs-decision` means the agent found a fork that is the user's call. Give options and a recommendation.
- Numbers are copied verbatim from tool output, never paraphrased or rounded.

## Workspace

Each orchestrated job gets `data/swarm/<job-id>/` (gitignored), where `<job-id>` = `YYYYMMDD-<slug>`:

```
brief.md          user goal, constraints, gate decisions (append-only)
ledger.md         task table: id | agent | status | worktree | handoff file | bead
handoffs/NN-<agent>-<slug>.md   each handoff, verbatim
prompts/          briefs sent to cmd workers
cmd-logs/         NDJSON transcripts from cmd-task.sh
hypotheses/       research lane: H-*.json tickets
```

The ledger is the orchestrator's memory. If the context gets compacted, re-read `brief.md` + `ledger.md` + the latest handoffs and continue. Durable outcomes go to their permanent homes: beads (tasks, follow-ups), `bd remember` (project facts), `docs/discovery-journal.md` (research results), git (code).

## Lanes

### Engineering lane

Six gated phases:

1. **Brief.** The orchestrator restates the goal and constraints in `brief.md`. Ambiguity goes to the user now, not mid-build.
2. **Design.** Lean profile: the orchestrator writes the task plan. Full profile: `architect` reads the code and returns a design + a task plan (tasks with acceptance criteria, file paths, dependencies, parallel-safe groups), then `reviewer` checks it in design mode. **User gate:** approve before any code.
3. **Plan → beads.** The orchestrator files one bead per task (`bd create`) with the acceptance criteria as user-testable checks.
4. **Build.** One `implementer` per bead, each in its own worktree, test-first. Parallelise only tasks the plan marks as disjoint. Mechanical sub-steps (renames, fixture tables, boilerplate tests) go to `cmd` workers.
5. **Check.** `reviewer` (spec + quality verdicts) and `verifier` (gates + exactness proofs) run on each diff. A blocking finding sends the task back to an implementer with the finding pasted into the brief. After 3 rounds on the same task, escalate: switch model, re-scope, or ask the user.
6. **Land.** The orchestrator merges worktrees one at a time onto the job branch, runs the full gate set on the merged result, squash-merges it to `main` (one commit per job), closes beads and pushes (CLAUDE.md § Session Completion).

### Research lane

Strategy factory: scout → author → screen/discover → audit → validate → audit → risk → journal. Full description, gates and ticket schema are in [`research-swarm.md`](research-swarm.md).

## Failure handling

| Symptom | Response |
|---|---|
| Same error 3 times | stop the loop; re-brief with the error and a different approach, or escalate the tier |
| Agent drifts out of scope | discard the out-of-scope hunks; file a bead for anything worth keeping |
| Handoff missing evidence | treat every claim as UNVERIFIED; dispatch `verifier` |
| Two agents need the same file | serialize them, or have `architect` re-cut the task boundary |
| Process dies with no traceback | NT Rust abort (CLAUDE.md § NautilusTrader Gotchas) and not an agent fault; check `retain_log_guard` / TraderId |
| Results look too good | `overfit-auditor` before anyone celebrates |
