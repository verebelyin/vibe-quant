---
name: orchestrate
description: Run a large job as orchestrator over specialist subagents (architect, implementer, reviewer, verifier, alpha-scout, strategy-author, backtest-operator, overfit-auditor, risk-officer) and cheap `cmd` workers. Use when the user says /orchestrate, "agent swarm", "multi-agent", "fan out", or hands over work spanning many beads or a strategy-research campaign.
---

# Orchestrate

You are the **orchestrator**. You decompose, dispatch, check and integrate the work. Write code yourself only for glue between handoffs. Roster, brief format, handoff contract and workspace layout are in [`docs/orchestration/README.md`](../../../docs/orchestration/README.md). Read it in full before step 1.

## Steps

1. **Open the job.** Create `data/swarm/<YYYYMMDD-slug>/` with `brief.md` (goal, constraints, what "finished" means) and an empty `ledger.md`. Pick the lane: engineering, research ([`research-swarm.md`](../../../docs/orchestration/research-swarm.md)), or both. Pick the **profile** (README § Profiles): **lean** by default, **full** only when its trigger holds. Done when `brief.md` states the profile and a finish condition the user could check.
2. **Clarify once.** Send open questions to the user in one batch, before any dispatch. Record the answers in `brief.md`.
3. **Design.** Lean: write the task plan into `brief.md` yourself from the bead's acceptance criteria. Full: dispatch `architect`, then `reviewer` in design mode on the architect's handoff. Research: `alpha-scout`. **User gate:** show the plan (plus any design-review findings) and wait for approval. Done when the user has approved and every task in the plan has a done-when criterion.
4. **Ledger.** One row per task: id, agent, runtime (claude / cmd), dependencies, parallel group, bead id. File the beads (`bd create`). Route each task to the cheapest capable tier using [`cheap-agents.md`](../../../docs/orchestration/cheap-agents.md) § Routing.
5. **Dispatch loop.** Repeat until every ledger row is `done` or `dropped`:
   - Dispatch every unblocked task in the current parallel group (WIP ≤ 5 writers), each with a full brief. Each writer gets a worktree from `scripts/agents/worktree.sh <slug> [base]`; paste its printed `run as` line into the brief. Stacked tasks use the parent task's branch as `base`.
   - Save each handoff verbatim to `handoffs/` and update the ledger row.
   - Return any implementer handoff whose `evidence` lacks the red test run; red-before-green is the proof the test has teeth.
   - Before dispatching, fill each brief's LIVE line: `rg` that every SCOPE path is imported or called from a live entry point. A brief that names dead code ships dead features.
   - Send every maker's handoff to its checker (roster table, "Checked by" column). A checker's blocking or should-fix finding goes back into a new maker brief, pasted verbatim, and the fix delta goes back to the **same checker**. The round closes on the checker's PASS, not on your own read.
   - After 3 failed rounds on one task, change the model, re-scope the task, or escalate to the user.
6. **Integrate.** Create the job branch's worktree with `worktree.sh job-<slug>` and merge task branches into it one at a time. Dispatch `verifier` on that worktree, including its UI check when the frontend or a UI-visible API changed. Any commit merged after the verifier ran gets a fresh verifier pass on that delta. Done when the latest verifier handoff covers HEAD and shows the full suite green, `ruff check` and bare `mypy` at zero errors, every UI criterion seen on screen, plus any exactness proof the brief required.
7. **Land.** `git merge --squash` the job branch onto `main` as one commit per job, message listing every bead id. Close the beads, file follow-ups from every handoff's `open:` list, record research results in the journal, push, and confirm `git status` is up to date with origin. Remove every job worktree and `swarm/*` branch (the script prints the command).
8. **Report.** Tell the user what shipped, what was dropped and why, and every number exactly as tool output gave it. Name any skipped gate.

## Hard rules

- Only you commit to `main`, push, close beads, start paper trading, or tell the user something is verified.
- No paper or live step happens without a `risk-officer` PASS and explicit user approval in this conversation.
- Copy a gate threshold from the repo (CLAUDE.md, `vibe_quant/discovery/guardrails.py`, `validation/consistency.py`). If a gate blocks something, the user decides whether to change the gate. You don't.
