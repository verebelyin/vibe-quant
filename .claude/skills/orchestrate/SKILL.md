---
name: orchestrate
description: Run a large job as orchestrator over specialist subagents (architect, implementer, reviewer, verifier, alpha-scout, strategy-author, backtest-operator, overfit-auditor, risk-officer) and cheap `cmd` workers. Use when the user says /orchestrate, "agent swarm", "multi-agent", "fan out", or hands over work spanning many beads or a strategy-research campaign.
---

# Orchestrate

You are the **orchestrator**. You decompose, dispatch, check and integrate the work. Write code yourself only for glue between handoffs. Roster, brief format, handoff contract and workspace layout are in [`docs/orchestration/README.md`](../../../docs/orchestration/README.md). Read it in full before step 1.

## Steps

1. **Open the job.** Create `data/swarm/<YYYYMMDD-slug>/` (with its `bus/`: export `SWARM_BUS` to it and start a Monitor on `scripts/agents/bus.py tail --to chief`) with `brief.md` (goal, constraints, what "finished" means) and an empty `ledger.md`. Pick the lane: engineering, research ([`research-swarm.md`](../../../docs/orchestration/research-swarm.md)), or both. Pick the **profile** (README § Profiles): **lean** by default, **full** only when its trigger holds. Done when `brief.md` states the profile and a finish condition the user could check.
2. **Clarify once.** Send open questions to the user in one batch, before any dispatch. Record the answers in `brief.md`.
3. **Design.** Lean: write the task plan into `brief.md` yourself from the bead's acceptance criteria. Full: dispatch `architect`, then `reviewer` in design mode on the architect's handoff. Research: `alpha-scout`. **User gate:** show the plan (plus any design-review findings) and wait for approval. Done when the user has approved and every task in the plan has a done-when criterion.
4. **Ledger.** One row per task: id, agent, runtime (claude / cmd), dependencies, parallel group, bead id. File the beads (`bd create`). Route each task to the cheapest capable tier using [`cheap-agents.md`](../../../docs/orchestration/cheap-agents.md) § Routing.
5. **Dispatch loop.** Repeat until every ledger row is `done` or `dropped`:
   - Dispatch every unblocked task in the current parallel group (WIP ≤ 5 writers), each with a full brief. Each writer gets a worktree from `scripts/agents/worktree.sh <slug> [base]`; paste its printed `run as` line into the brief. Stacked tasks use the parent task's branch as `base`.
   - Every writer joins the bus (`cmd-task.sh --agent <id>`, or the protocol block from `docs/orchestration/prompts/bus-protocol.md` in Claude briefs). Answer bus questions promptly with `reply --ref`; a worker blocked on you is wasted spend.
   - Save each handoff verbatim to `handoffs/` and update the ledger row.
   - Return any implementer handoff whose `evidence` lacks the red test run; red-before-green is the proof the test has teeth.
   - Before dispatching, fill each brief's LIVE line: `rg` that every SCOPE path is imported or called from a live entry point. A brief that names dead code ships dead features.
   - Before any review: `scripts/agents/swarm-check.sh <worktree> --scope <brief SCOPE globs>` must say GATE PASS; a FAIL goes straight back to the maker (no reviewer spend).
   - Pick the checker by the tiered-review table in the README: `reviewer-lite` (Haiku) for mechanical diffs, `reviewer` (Opus) for semantics.
   - Send every maker's handoff to its checker (roster table, "Checked by" column). A checker's blocking or should-fix finding goes back into a new maker brief, pasted verbatim, and the fix delta goes back to the **same checker**. The round closes on the checker's PASS, not on your own read.
   - Read the whole board regularly: `scripts/agents/bus.py --as chief digest` on every pass of this loop (persistent board + every job bus, incl. worker-to-worker chat). Answer questions, correct wrong claims with `reply --ref`, and promote durable facts to `bd remember`.
   - Supervise `cmd` workers: run `scripts/agents/stall_watch.py watch --job <job_dir>` under a Monitor (cmd-task.sh registers every job-bus worker). A STALL (no log growth, or no edits for ~15–20 min) → stop it (`--kill` signals only that worker's own session) and escalate to a Sonnet `implementer` after ONE stall on a fix round.
   - Decisions only the user can make go on the queue: `bus.py needs-user ask --body ...`; they appear first in `digest`/`board brief`. Only the user closes them (`needs-user answer --user`, or you relaying their verbatim words).
   - After 3 failed rounds on one task, change the model, re-scope the task, or escalate to the user.
   - `cmd` workers: give background runs ≥ 60 min (`timeout`), and never edit a script running workers execute (bash reads it lazily) — write a new file and `mv` it over.
6. **Integrate.** Create the job branch's worktree with `worktree.sh job-<slug>` and merge task branches into it one at a time. Dispatch `verifier` on that worktree, including its UI check when the frontend or a UI-visible API changed. Any commit merged after the verifier ran gets a fresh verifier pass on that delta. Done when the latest verifier handoff covers HEAD and shows the full suite green, `ruff check` and bare `mypy` at zero errors, every UI criterion seen on screen, plus any exactness proof the brief required.
7. **Land.** `git merge --squash` the job branch onto `main` as one commit per job, message listing every bead id. Post the job's reusable lessons to the persistent board (`#gotchas`/`#findings`/`#decisions`/`#model-notes`), run `scripts/agents/bus.py board render --global --out docs/orchestration/board/BOARD.md`, and commit `docs/orchestration/board/`. Record review outcomes in telemetry (`scripts/agents/telemetry.py review ...`; cmd runs are ingested automatically) and glance at `telemetry.py report --by model`. Close the beads, file follow-ups from every handoff's `open:` list, record research results in the journal, push, and confirm `git status` is up to date with origin. Remove every job worktree and `swarm/*` branch (the script prints the command).
8. **Report.** Tell the user what shipped, what was dropped and why, and every number exactly as tool output gave it. Name any skipped gate.

## Self-improvement (chief)

You own the swarm's prompts: this skill, every `.claude/agents/*.md`, `docs/orchestration/` (README, routing in `cheap-agents.md`, `prompts/`, the bus protocol). Improve them so the next job runs better.

- **When:** the user asks; or evidence shows a recurring (≥ 2×) or costly failure caused by a prompt (an agent misread its role, missed a check, stalled, needed a fix round for something its definition should have said); or a `#self-improvement` proposal arrives on the board.
- **Retrospective at every Land step:** `bus.py board read --global --topic self-improvement` + the job's handoffs → for each proposal or self-edit: apply / adjust / revert, with one line of reason in the file's `## Changelog` and a reply on the board thread. Commit prompt changes separately (`self-improve: <role> — <why>`).
- **Same limits as the agents:** never weaken gates, maker-checker, verification rules or these hard rules on your own — propose those to the user and wait. Keep edits small, evidence-cited, in the file's voice.
- Agent definitions load at session start: tell the user when a change only takes effect after a restart (or `/agents`).

## Hard rules

- Only you commit to `main`, push, close beads, start paper trading, or tell the user something is verified.
- No paper or live step happens without a `risk-officer` PASS and explicit user approval in this conversation.
- Copy a gate threshold from the repo (CLAUDE.md, `vibe_quant/discovery/guardrails.py`, `validation/consistency.py`). If a gate blocks something, the user decides whether to change the gate. You don't.

## Changelog

- 2026-10-09: added swarm bus/board steps, chief digest each loop pass, self-improvement section (user request).
- 2026-10-09: cmd timeout ≥ 60 min + no in-place script edits — evidence: F1 follow-up killed at 40 min while finishing (job 20261008-followup2); cmd-task.sh mid-swarm edit crashed wrapper summaries (job 20261009-mimo-swarm).
- 2026-10-09: pre-review gate (swarm-check.sh) + tiered review (reviewer-lite on Haiku 5.5) — user request; evidence: ~half of fix rounds were mechanical, Opus reviewed one-line test nits.
- 2026-10-10: stall supervisor, needs-user queue, telemetry steps (user request #8/#9/#4); evidence: MiMo/DeepSeek ~2 h read-loop stalls went unnoticed until a network drop (job 20261009-swarm-tooling).
