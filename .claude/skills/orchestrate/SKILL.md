---
name: orchestrate
description: Run a large job as orchestrator over specialist subagents (architect, implementer, reviewer, verifier, alpha-scout, strategy-author, backtest-operator, overfit-auditor, risk-officer) and cheap Command Code workers (T3 `delegate_task` subagents). Use when the user says /orchestrate, "agent swarm", "multi-agent", "fan out", or hands over work spanning many beads or a strategy-research campaign.
---

# Orchestrate

You are the **chief**: you decompose, dispatch, check and integrate. You write code only as glue between handoffs. Roster, brief format, handoff contract, bus and workspace: [`docs/orchestration/README.md`](../../../docs/orchestration/README.md). T3 dispatch mechanics: [`cheap-agents.md`](../../../docs/orchestration/cheap-agents.md). Before writing any brief: [`brief-checklist.md`](../../../docs/orchestration/brief-checklist.md).

**Who runs what:** `scripts/agents/models.py route <role>` (roles: maker, operator, fixer-small, fixer, reviewer, reviewer-lite, verifier, architect, design-review, money-verdict, memory). It reads `docs/orchestration/models.json`, the only routing source. When the user enables, disables or pauses a model, run `models.py set <key> --off --until <ISO time> --note '<user words>'` at once, and state the route in your next dispatch message. A task already running on a model the user just disabled finishes unless they say otherwise; its next rounds take the new route.

## Steps

1. **Open the job.** `data/swarm/<YYYYMMDD-slug>/` with `bus/` (`export SWARM_BUS=<abs>/bus`), `brief.md` (goal, constraints, profile, a finish condition the user could check) and `ledger.md`. Lane: engineering, research ([`research-swarm.md`](../../../docs/orchestration/research-swarm.md)) or both. Profile: **lean** by default, **full** when README § Profiles says so.
2. **Clarify once.** Put every open question to the user in one batch (§ Asking the user format) before any dispatch. Record answers in `brief.md`.
3. **Design.** Lean: write the task plan into `brief.md` from the beads' acceptance criteria. Full: `architect`, then `reviewer` in design mode. Research: `alpha-scout`. Money semantics get the architect's contract table in every profile. **User gate:** show the plan and wait for approval. Done when every task has a done-when criterion and the user approved.
4. **Ledger.** One row per task: id, runtime, model, dependencies, parallel group, bead, worktree, `clientRequestId` + `taskId` for T3 rows. File the beads.
5. **Dispatch loop** until every row is `done` or `dropped`:
   - Each writer gets `scripts/agents/worktree.sh <slug> [base]` (stacked task: the parent's branch as base). WIP ≤ 5 writers.
   - Render every brief with `scripts/agents/mkbrief.py --route <role> --bead <id>` from `prompts/cmd-task.md` (makers) or `prompts/review-task.md` (reviewers), appending `prompts/bus-protocol.md`. It appends the ledger row and prints the dispatch: a `DELEGATE_TASK {json}` line to pass to `delegate_task` as is, or the Agent-tool route for Claude roles. Later status rows: `scripts/agents/ledger.py`.
   - Pre-review gate: `swarm-check.sh <wt> --scope '<SCOPE>'` must say GATE PASS. A FAIL goes straight back to the maker.
   - Send every handoff through `scripts/agents/handoff_lint.py`, save it verbatim to `handoffs/`, and update the ledger. Return a maker handoff without a red run.
   - Route each diff to its checker (README § Tiered review) and follow the reviewer's `fix-route:`: `deepseek` → new `delegate_task` round with the original brief; `sonnet` → Sonnet `implementer`; both get the findings verbatim. A blocking finding caused by your own ruling, fully specified as a contract row, may go back to DeepSeek. One review round per task (README): you verify the fix (reviewer's repros, targeted tests, gate); no second review; only the user can authorise one. Before routing, re-grade each finding's severity yourself (another path, a rare case or a small cost is a nit whatever its label). A timeout, error or missing reply is never approval. A finding still disputed after one chief ruling goes to the user as both sides in 2-3 lines.
   - Answer `ask`s within minutes with `bus.py --as chief reply --ref <id>`. Rulings go into the contract (brief-checklist § Contracts). Relay streamed findings to an idle maker as a new round (T3) or `SendMessage` (Claude).
   - Mid-round scope changes go into a new round, never a DM: T3 children read the bus only when they run `bus inbox`.
6. **Integrate.** `worktree.sh job-<slug>`; merge every task's final tip one at a time (a child whose parent landed: `merge_stacked.py <child> <parent>`). Dispatch `verifier` on it with the union of the merged tasks' SCOPE (and its UI check when the frontend or a UI-visible API changed). Any commit merged after the verifier ran gets a fresh verifier pass on that delta. Done when the latest verifier handoff covers HEAD: full suite green, `ruff check` + bare `mypy` at zero, UI criteria seen on screen, required exactness proofs IDENTICAL.
7. **Land.**
   - Write one message file per task ([`prompts/commit-message.md`](../../../docs/orchestration/prompts/commit-message.md)), then `scripts/agents/land.py --job <job> --verified <job branch> <task>=<msg> [<child>@<parent>=<msg>] ...` in dependency order (docs after the code they describe). It commits the board mirror, applies each task from its own fork point and checks the tree equals the verified branch; on exit 1, reset main to the start SHA it prints and redo.
   - `scripts/agents/telemetry.py ingest-runs --job <job>`, then `telemetry.py review` per verdict; `bus.py board render --global --out docs/orchestration/board/BOARD.md`; commit the board.
   - Memory: `consolidate_memory.py propose --job <job>`; read and prune `memory-ops.json`, then `apply --ops <that file>`.
   - Close beads (UI beads only after their on-screen check), file follow-ups from every `open:` list, record research results in the journal, push, confirm `git status` is up to date, remove the job's worktrees and `swarm/*` branches.
   - Run the § Self-improvement sweep.
8. **Report.** What shipped, what was dropped and why, every number verbatim from tool output, any skipped gate. Then recommend a fresh session for the next job: every turn re-reads the whole context.

## Waiting well

- Keep one background Bash call (`run_in_background`, timeout 7200000) on `bus.py --as chief tail --to chief --durable --kinds question,blocker --exit-on-first`. It ends on the first ask: answer, then restart it.
- A notification with nothing new (a task already ledgered, a duplicate completion) gets no prose turn. Read `task_status` only when you need a result mid-turn and the handoff is not on the bus.
- A T3 child overdue (> ~45 min) or with no worktree edits after ~20 min: tell the user; cancel only when they say so. After a T3 server restart, `task_status` every child before re-dispatching: a cancelled child can resume on its own.
- Every status message to the user ends with three lines: **Running** and **Startable** (from `bus.py --as chief status`), and **Waiting on you** (open needs-user plus your unasked questions, asked right there).
- Between dispatch waves, run `bus.py --as chief digest` and act on it: answer questions, correct wrong claims with `reply --ref`, file beads, promote durable facts to `bd remember`.
- A Claude subagent stopped by the account session limit keeps its work: `SendMessage` it after the reset instead of re-dispatching.

## Ground rules for the job

- Keep proofs short: unit tests, mutants (`scripts/agents/mutate.py`) and `exactness_239.py`. A live discovery check uses pop 4–6, 1–2 gens, a 2–3 month window. Full-size reruns at most once, in the final verifier pass.
- Writes to shared real data (archive backfill, catalog rebuild) wait until no verifier or gate is reading the catalog. Ingesting new symbols or date ranges into `data/archive`/`data/catalog` needs the user's OK first: worktrees share them.
- Never edit a script that running workers execute; write a new file and `mv` it over.
- If main is force-pushed mid-job, running worktrees gate with `--base <old main tip>` until Land, and land by cherry-pick.
- Write brief files with `mkbrief.py`, the Write tool or a quoted heredoc (`<<'EOF'`); unquoted heredocs run backtick spans.
- `bd update --notes` replaces notes; append decisions with `--append-notes`.
- Anything published outside the repo (an issue, PR or third-party message) is drafted for the user and never offered as the recommended option.

## Asking the user

Ask in plain words, as to a friend who is not reading the code. Per question, short lines:

1. **The question**, one friendly sentence about what happens in the user's world (bead id in brackets at the end).
2. **What happens today**, as one real example with a coin, a time and numbers ("you hold BTC and ETH; ETH fails to save; the screen shows no positions").
3. **The options**, each with its everyday consequence in one line, and your recommendation marked.

Tag every number in an option `MEASURED (<command>)` or `Likely:`; an option that loosens a gate states its measured no-edge pass rate. No internal names, file paths or contract-row ids in the question; put evidence and tables after the questions. End with a one-line summary of what you need ("1 = A or B? 2 = OK?"). Put it all in the visible message, never only in AskUserQuestion previews.

## Self-improvement

You own the swarm's environment: scripts in `scripts/agents/`, this skill, `.claude/agents/*.md`, `docs/orchestration/`. Improve it at every Land step and whenever a task reaches its 3rd round, without asking, in a separate `self-improve:` commit, and report what changed.

1. Collect every handoff's `improve:` line and the job's `#self-improvement` posts.
2. A **mechanical** failure (wrong command, path, flag, format, forgotten step) gets a script, test or hook change so it cannot recur. No prose.
3. A **judgement** failure seen at least twice gets one positively phrased line in the file that owns it (an agent definition, `brief-checklist.md`, this skill). The evidence goes in the commit message, never in the file.
4. A rule that was broken again after it was written is a no-op: replace it with a check, or delete it.
5. Every steering file has a word budget enforced by `tests/unit/test_steering_budget.py`. To add a line, merge or delete another. Raising a budget needs the user.
6. Never weaken gates, maker-checker, verification rules or the hard rules below on your own: propose and wait.

Agent definitions load at session start: say when a change needs a restart (or `/agents`).

## Hard rules

- Only you commit to `main`, push, close beads, start paper trading, or tell the user something is verified.
- No paper or live step without a `risk-officer` PASS and the user's explicit approval in this conversation.
- Copy gate thresholds from the repo (CLAUDE.md, `vibe_quant/discovery/guardrails.py`, `validation/consistency.py`). When a gate blocks, the user decides whether it changes.
- A `delegate_task` error goes to the user verbatim; no provider fallback.
