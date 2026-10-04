---
name: orchestrate
description: Run a large job as orchestrator over specialist subagents (architect, implementer, reviewer, verifier, alpha-scout, strategy-author, backtest-operator, overfit-auditor, risk-officer) and cheap `cmd` workers. Use when the user says /orchestrate, "agent swarm", "multi-agent", "fan out", or hands over work spanning many beads or a strategy-research campaign.
---

# Orchestrate

You are the **orchestrator**. You decompose, dispatch, check and integrate the work. Write code yourself only for glue between handoffs. Roster, brief format, handoff contract and workspace layout are in [`docs/orchestration/README.md`](../../../docs/orchestration/README.md). Read it in full before step 1.

## Steps

1. **Open the job.** Create `data/swarm/<YYYYMMDD-slug>/` with `brief.md` (goal, constraints, what "finished" means) and an empty `ledger.md`. Pick the lane: engineering, research ([`research-swarm.md`](../../../docs/orchestration/research-swarm.md)), or both. Done when `brief.md` states a finish condition the user could check.
2. **Clarify once.** Send open questions to the user in one batch, before any dispatch. Record the answers in `brief.md`.
3. **Design.** Dispatch `architect` (engineering) or `alpha-scout` (research). **User gate:** show the design or hypothesis shortlist and wait for approval. Done when the user has approved and every task in the plan has a done-when criterion.
4. **Ledger.** One row per task: id, agent, runtime (claude / cmd), dependencies, parallel group, bead id. File the beads (`bd create`). Route each task to the cheapest capable tier using [`cheap-agents.md`](../../../docs/orchestration/cheap-agents.md) § Routing.
5. **Dispatch loop.** Repeat until every ledger row is `done` or `dropped`:
   - Dispatch every unblocked task in the current parallel group (WIP ≤ 5 writers), each with a full brief. Writers get their own worktree.
   - Save each handoff verbatim to `handoffs/` and update the ledger row.
   - Send every maker's handoff to its checker (roster table, "Checked by" column). A checker's blocking finding goes back into a new maker brief, pasted verbatim.
   - After 3 failed rounds on one task, change the model, re-scope the task, or escalate to the user.
6. **Integrate.** Merge worktrees one at a time onto the job branch. After all merges, dispatch `verifier` on the merged tree. Done when the verifier's handoff shows tests, `ruff check` and `mypy` at zero errors, plus any exactness proof the brief required.
7. **Land.** Close the beads, file follow-ups from every handoff's `open:` list, record research results in the journal, then commit, push, and confirm `git status` is up to date with origin.
8. **Report.** Tell the user what shipped, what was dropped and why, and every number exactly as tool output gave it. Name any skipped gate.

## Hard rules

- Only you commit to `main`, push, close beads, start paper trading, or tell the user something is verified.
- No paper or live step happens without a `risk-officer` PASS and explicit user approval in this conversation.
- Copy a gate threshold from the repo (CLAUDE.md, `vibe_quant/discovery/guardrails.py`, `validation/consistency.py`). If a gate blocks something, the user decides whether to change the gate. You don't.
