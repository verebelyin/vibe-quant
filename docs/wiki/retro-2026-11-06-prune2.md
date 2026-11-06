# Retro 2026-11-06: job 20261105-prune2 vs retro 2026-11-05

Sources: this chief session (22 children: 6 DeepSeek makers, 8 Sonnet fixers/makers, 5 GPT reviewers, 2 Haiku reviewer-lite, 1 Sonnet verifier), data/swarm/20261105-prune2/{brief.md,ledger.md,handoffs/}, docs/orchestration/telemetry/tasks.jsonl, retro list data/swarm/20261105-retro/candidates.md.

## Held (retro 2026-11-05 fixes that worked here)
- C1 timeout shim, C5 BSD-sed guard, C6 review worktrees (`worktree.sh`), #7 gate runs worktree scripts: no incident.
- C2 handoff contract: 20 of 21 handoffs lint OK first time (slots: missing `absorbed:` cite).
- D20 `bus tail --stop` landed the same day (another session); chief still orphaned tails twice with `&` before it.

## Recurred (retro items not done, cost measured here)
- #11 parity progress + time cap (leftover, bead vibe-quant-hf9d3): gate parity FAILed by timeout 3x (300 s TOOL_TIMEOUT); one gate looked hung 30 min with no output (parity is silent); standalone run killed at 90 min, no verdict; job landed without parity (user option B). Bead vibe-quant-50gha.
- #14 chief.py init / verify template (leftover): chief hand-wrote the ledger header -> mkbrief refused (same failure the retro named); verifier ver-job-1 ended its turn "waiting" for its own background gate -> no verdict, chief re-read the gate log.
- #5 "new gate = warn-mode dry run on a past job first": not applied to the slow-tests gate -> 7 FAILs surfaced only at integration, 3 gate reruns.
- #10 telemetry: every reviewed row appears twice (once review=null); Claude T3 children log tokens 0.
- C7 asks unanswered: rev-spdeng-1 asked on the lobby (no --bus); chief tail watched only the job bus; 14 min until it re-asked.

## New
- stale-reads: 5 of 6 FAILs false positives (bead vibe-quant-9hygx).
- Gate oversubscription (load 40-53) -> SLOTS landed a6558ba4.
- /usr/bin/git shim 20.2 ms vs 10.7 ms wrapper -> machine fix ([machine-setup.md](machine-setup.md)).
- Model outcomes: DeepSeek money/gate lanes 3/3 BLOCK (lost kills, fail-open); Sonnet gate lanes 2/2 BLOCK too (lifecycle edge cases, a behaviour change) — see [routing-observations.md](routing-observations.md).
- land.py: a stacked series needed 4 manual steps (conflict not taken from --verified; commit-msg sections missing found one commit at a time; an intermediate commit over a word budget).
- DeepSeek token use: 6 maker rounds = 125.5 M tokens in (telemetry), cc-spdeng-1 36.3 M alone.

## Follow-ups filed 2026-11-06 (user: beads for a fresh session)
- P1: vibe-quant-hf9d3 (parity cap #11, chief.py init + verify template #14, lobby asks C7), vibe-quant-50gha (gate parity timeout), vibe-quant-9hygx (stale-reads false positives), vibe-quant-5ez98 (land.py pre-flight).
- P3: vibe-quant-7tueg (new gates start in --warn), vibe-quant-ahn8n (telemetry dupes, Claude tokens).

## Status 2026-11-06 (after job 20261106-p1tools)
- Fixed: #11 parity (ac3dec2f; verifier standalone `--budget 1500` gave a verdict, compared=8 lost=0, 1178 s), #14 chief.py init + verify-task.md (11ea59f6; template forbids ending a turn on a background gate), C7 lobby asks reach the job tail (598b145f), stale-reads false positives (7e7ed0cd; p1tools needed 1 absorbed ruling vs 6 FAILs in prune2).
- Open: #5 warn-mode for new gates (vibe-quant-7tueg), #10 telemetry (vibe-quant-ahn8n), land.py pre-flight (vibe-quant-5ez98).
