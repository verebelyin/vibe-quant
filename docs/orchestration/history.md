# Orchestration history (archive)

Changelogs moved out of the agent definitions, the orchestrate skill and cheap-agents.md on 2026-10-17 (job 20261017-retro-distill), verbatim. Nothing loads this file; it is evidence for future retros. New history lives in commit messages.

## .claude/agents/alpha-scout.md

- 2026-10-09: self-improvement + changelog sections added (user request: agents may improve their own prompts; chief reviews).
- 2026-10-10: added messaging triggers (ask/DM/reply/inbox) + handoff `comms:` line — evidence: job 20261010-backlog had 0 asks, 0 replies, 0 peer DMs across 15 agents (user request).
- 2026-10-13: durable-bus usage (inbox before gate, gate --bus/--bus-agent, --client-id, pending) - evidence: job 20261012-p1-honest (workers missed chief DMs; bus hardened in f40fd209) - user request.
- 2026-10-13: self-improvement made a priority duty; every handoff ends with an `improve:` line, which the chief applies automatically. User request after job 20261012-p1-honest (84 handoffs, no prompt edits during an 8-round loop).
- 2026-10-14: board Start line now searches by SCOPE path; posting rules (RULE/WHEN/EVIDENCE, search before posting, verdicts and status off the global board). Evidence: orchestration analysis 2026-10-14: 93% of the posts agents relied on came from their own job; 49% of global posts were task-local; 15 duplicates.
- 2026-10-14: per-idea: DSR-required Sharpe at the test's N, fixed-DSL default, controls/nulls fully specified — evidence: job 20261014-honest-lev: scout-hl DSR bisection, H-5 degenerate control, H-4 null method.
- 2026-10-15: Reddit/X blocked note under Bulk sources. Evidence: job 20261015-funding-carry research agents (res-practitioner: domain-blocked for fetch and search).

## .claude/agents/architect.md

- 2026-10-09: self-improvement + changelog sections added (user request: agents may improve their own prompts; chief reviews).
- 2026-10-10: verify a perf bead's premise against run history before designing — evidence: arch-jobs round 2 (f2z5p dropped after F0 measurement).
- 2026-10-10: added messaging triggers (ask/DM/reply/inbox) + handoff `comms:` line — evidence: job 20261010-backlog had 0 asks, 0 replies, 0 peer DMs across 15 agents (user request).
- 2026-10-13: durable-bus usage (inbox before gate, gate --bus/--bus-agent, --client-id, pending) - evidence: job 20261012-p1-honest (workers missed chief DMs; bus hardened in f40fd209) - user request.
- 2026-10-13: semantics contract table (6) + one shared implementation per money rule (7). Evidence: job 20261012-p1-honest. Liquidation T3 took 8 review rounds and T4 at least 5, because boundary rules (end_ns inclusive or exclusive, funding horizon, end-of-run fee, raw-fill timestamps) were decided as ~10 mid-review chief rulings (R1–R6b). Screening and validation each re-implemented them, so every bug was found and fixed twice. User approved.
- 2026-10-13: self-improvement made a priority duty; every handoff ends with an `improve:` line, which the chief applies automatically. User request after job 20261012-p1-honest (84 handoffs, no prompt edits during an 8-round loop).
- 2026-10-13: list the broad excepts on error-propagation paths. Evidence: arch-liq-t5 found that overfitting/pipeline.py:305 `except ValueError` swallows DataUnavailableError, which the bead notes had missed (they named only the screening pipeline and backtest_fn). Applied automatically.
- 2026-10-13: the contract lists existing assertions it breaks. Evidence: cc-liq-a1 `improve:` line; A1 had to rediscover 5 changed assertions (leverage message, 2 MarkDataMissing → LiquidationIntegrityError, 2 import-isolation prefixes), costing a test-gate cycle. Applied automatically.
- 2026-10-14: board Start line now searches by SCOPE path; posting rules (RULE/WHEN/EVIDENCE, search before posting, verdicts and status off the global board). Evidence: orchestration analysis 2026-10-14: 93% of the posts agents relied on came from their own job; 49% of global posts were task-local; 15 duplicates.
- 2026-10-14: semantics contract rows numbered R1..Rn, with five more question classes and a reviewer-probe pass before hand-off. Evidence: 24 of 54 questions and blockers in 10-08..10-13 were mid-flight liquidation rulings (p1-honest R1–R6b). 18 fall in the five new classes (ordering, signs/floors, cost buckets, aggregation, input shapes); 5 were in classes already listed but left unfilled (window edge, horizon, mode-off); 1 was a mutant-severity call.
- 2026-10-14: before-designing checks: NT numeric ranges, contract values must be shipped values, batch-writer key uniformity — evidence: job 20261014-honest-lev: arch-hl-lev, cc-hl-fees, cc-hl-liqnull.
- 2026-10-15: seven before-designing checks (premise for fix beads, one-consumer overrides, default-meaning config fields, entity identity, exclude-vs-reject precedence, cross-task signatures, hand-rolled schema fixtures). Evidence: job 20261015-honest-safe: both design reviews found blockers in these classes (GPT B1, Opus B1) and makers needed ~15 mid-flight rulings (R17d-h, R28b, R44b/c, R47c-e, R50c, R52d-f). Proposals 18dea6802ce847a8-7400, 18dea721c51bf438-9199, 18dea76302e91250-4743, 18dea8dd34cca630-9377, cc-hs-v2-1, cc-hs-p3a2-1, cc-hs-v1c-1, cc-hs-z2-1.
- 2026-10-15: NT lifecycle probe for new engine designs; contract classes for error payload + raise site and integer-step quantities; owner-test column. Evidence: job 20261015-funding-carry (arch-fc probe, T1 R44 rounds 2-3, T2a B1, design r1 SF1/SF6). Applied automatically.
- 2026-10-15 (Land): parity comparators walk dataclasses.fields(); perf estimates time the full loop. Evidence: rev-fc-t6b SF1 (13 skipped fields), cc-fc-t8 R55 timing.
- 2026-10-15: credential-free on_start gate probe; needs-decision instead of blocking asks for user-visible new values; check existing halt semantics before reusing them. Evidence: job 20261015-small-beads design REJECT r1 (rev-sb-design-1 B1, probe startup_filled_entries: 0) and arch-sb improve line. Applied automatically at T3's 3rd round.
- 2026-10-15 (Land, job 20261015-small-beads): shared status-field writers rule. Evidence: arch-sb-2 design r2 rows R24/R24b. Applied automatically.
- 2026-10-16 (Land, job 20261016-followups2): run-level settings list runner constructors, child-run creators and parameter comparators. Evidence: arch-fu2 proposal 18def38cc20b8d00-8327 + rev-fu2-design SF2/SF3. Applied automatically.
- 2026-10-16 (Land, job 20261016-sizing): recorded-JSON absent/null/type/invalid rows + writer entry point; fail-closed sibling steps; harness numbers are not contract numbers — evidence S29d, S23b B1, S23e, arch-sz proposal 18df043d81716438-5113, H25. Applied by the chief.
- 2026-10-16 (job 20261016-safety-bankroll, design round 3): stall-term + worker-owned-close rows for fail-closed timing bounds; confirmed leverage and order-set union for margin math. Evidence: rev-sb-design-1 B1/B2/B4, rev-sb-design-2 B2 residual; proposals 18df18789d25e010-7418 and arch-sb r2 improve line.
- 2026-10-16 (job 20261016-safety-bankroll, T7 round 3): normalization rules at the shared consumer. Evidence: rev-sb-t7-2 B1 residual, ruling T7-R4.
- 2026-10-17 (job 20261016-safety-bankroll): live vs backtest engine rows for order-flow contracts. Evidence: risk-sb-t4-1 F1 (R30d added mid-job).
- 2026-10-17: needs-decision items carry plain terms + example flow + numbers for the user format. Evidence: job 20261017-grind user feedback on the 6osqd/lcf12 explanations.
- 2026-10-17 (Land 2, job 20261017-grind): re-run prior probes through production objects; rg stubs of a dataclass's source when adding a field. Evidence: arch-gr-1 5byet IOC finding, rev-gr-design-1 SF1.
- 2026-10-17 (job 20261017-easy4): venue-symbol history scan; off-loop moves list collaborator-patching tests. Evidence: arch-e4-coins-1, arch-e4-events-1 improve lines.

## .claude/agents/backtest-operator.md

- 2026-10-09: self-improvement + changelog sections added (user request: agents may improve their own prompts; chief reviews).
- 2026-10-10: added messaging triggers (ask/DM/reply/inbox) + handoff `comms:` line — evidence: job 20261010-backlog had 0 asks, 0 replies, 0 peer DMs across 15 agents (user request).
- 2026-10-13: durable-bus usage (inbox before gate, gate --bus/--bus-agent, --client-id, pending) - evidence: job 20261012-p1-honest (workers missed chief DMs; bus hardened in f40fd209) - user request.
- 2026-10-13: self-improvement made a priority duty; every handoff ends with an `improve:` line, which the chief applies automatically. User request after job 20261012-p1-honest (84 handoffs, no prompt edits during an 8-round loop).
- 2026-10-13: skip API pre-checks when runners are called directly. Evidence: op-liq-parity `improve:` line (the status curl returned an HTML error page; the runs were unaffected). Applied automatically per the user's 2026-10-13 rule.
- 2026-10-14: board Start line now searches by SCOPE path; posting rules (RULE/WHEN/EVIDENCE, search before posting, verdicts and status off the global board). Evidence: orchestration analysis 2026-10-14: 93% of the posts agents relied on came from their own job; 49% of global posts were task-local; 15 duplicates.
- 2026-10-14: discovery needs a pre-created run row (StateManager, disclosed); check MARK coverage before liquidation runs — evidence: job 20261014-honest-lev: op-hl run 919, ver-hl-lev runs 923/924.

## .claude/agents/implementer.md

- 2026-10-09: self-improvement + changelog sections added (user request: agents may improve their own prompts; chief reviews).
- 2026-10-09: added step 5 (mutation-check your own tests) — evidence: tautological tests in T2 (vibe-quant-91g20), F1 follow-up (ox73t) and the SB identity test (t4aey, reviewer B1).
- 2026-10-10: role is now Sonnet fixer/escalation; DeepSeek via T3 is the default maker — user request.
- 2026-10-10: added messaging triggers (ask/DM/reply/inbox) + handoff `comms:` line — evidence: job 20261010-backlog had 0 asks, 0 replies, 0 peer DMs across 15 agents (user request).
- 2026-10-11: step 5 full suite once via swarm-check, targeted tests while iterating, E2E -g, batched reads; mutation self-check budget ≤5 — evidence: job 20261010-backlog time analysis (300 agent-min: ~65 min in repeated full suites, 30 min mutation, 58% of DeepSeek time in 1-tool turns) — user request.
- 2026-10-11: async-launch tests poll final status; no todo lists — user decision 2026-10-11 (speed job 20261011-speed).
- 2026-10-11: step 5 made concrete — gate exactly once per round as last command, no gate after rebase, comma-separated --scope; batch every turn with a ≥2 calls/turn target — evidence: time analysis of jobs 20261010-backlog / 20261011-testspeed / 20261011-grind (1.11 / 1.17 / 1.10 calls per turn; full-suite+gate ~21-23 min flat; impl-sweep ran swarm-check 7×) — user approved.
- 2026-10-13: durable-bus usage (inbox before gate, gate --bus/--bus-agent, --client-id, pending) - evidence: job 20261012-p1-honest (workers missed chief DMs; bus hardened in f40fd209) - user request.
- 2026-10-13: semantics-contract section: one test per row, ask the chief on undecided cases, put shared rules in the shared module. Evidence: job 20261012-p1-honest. Makers picked boundary rules on their own: funding to end_ns vs to the mark ts, and a 2-pass rescan. Each pick became the next round's blocker (T3 8 rounds). User approved.
- 2026-10-13: self-improvement made a priority duty; every handoff ends with an `improve:` line, which the chief applies automatically. User request after job 20261012-p1-honest (84 handoffs, no prompt edits during an 8-round loop).
- 2026-10-14: board Start line now searches by SCOPE path; posting rules (RULE/WHEN/EVIDENCE, search before posting, verdicts and status off the global board). Evidence: orchestration analysis 2026-10-14: 93% of the posts agents relied on came from their own job; 49% of global posts were task-local; 15 duplicates.
- 2026-10-14: commit WIP after each green step. Evidence: quota waves killed 12 runs mid-task (6 Claude 429, 3 Command Code, 3 Codex), and the Claude kills left work uncommitted in worktrees.
- 2026-10-14: Bash: no trailing cd outside the worktree; rm scratch alone; long gates in background — evidence: job 20261014-honest-lev: impl-hl-data-gap lost ~4 turns to "no active turn" x3.
- 2026-10-15: macOS mutants with `sed -i ''` or Python; equality-edge tests for every money comparison — evidence: job 20261015-honest-safe impl-hs-p3b-2 (two mutants silently unapplied), V1a/V1b extra rounds.
- 2026-10-16 (Land, job 20261016-safety-bugs): fix rounds may narrow the board search but must post done. Evidence: both Sonnet fixers in this job skipped the search and the done post. Applied automatically.
- 2026-10-17 (job 20261016-safety-bankroll): live TradingNode for NT order-set semantics — evidence: risk-sb-t4-1 F1, impl-sb-t4-1 improve line.
- 2026-10-17 (job 20261017-easy4): wall-clock waits after moving work to a thread. Evidence: impl-e4-evnode-1.

## .claude/agents/overfit-auditor.md

- 2026-10-09: self-improvement + changelog sections added (user request: agents may improve their own prompts; chief reviews).
- 2026-10-10: added messaging triggers (ask/DM/reply/inbox) + handoff `comms:` line — evidence: job 20261010-backlog had 0 asks, 0 replies, 0 peer DMs across 15 agents (user request).
- 2026-10-13: durable-bus usage (inbox before gate, gate --bus/--bus-agent, --client-id, pending) - evidence: job 20261012-p1-honest (workers missed chief DMs; bus hardened in f40fd209) - user request.
- 2026-10-13: self-improvement made a priority duty; every handoff ends with an `improve:` line, which the chief applies automatically. User request after job 20261012-p1-honest (84 handoffs, no prompt edits during an 8-round loop).
- 2026-10-14: board Start line now searches by SCOPE path; posting rules (RULE/WHEN/EVIDENCE, search before posting, verdicts and status off the global board). Evidence: orchestration analysis 2026-10-14: 93% of the posts agents relied on came from their own job; 49% of global posts were task-local; 15 duplicates.
- 2026-10-14: holdout diff for gated arms (identical = NOT INFORMATIVE); rule falsifier clauses before repo gates — evidence: job 20261014-honest-lev: audit-hl H-1 holdout identity and corrected premise.

## .claude/agents/reviewer-lite.md

- 2026-10-09: created (user request: tiered review — Haiku 5.5 first pass for mechanical diffs, Opus `reviewer` for semantics).
- 2026-10-09: effort pinned to high (user rule: Haiku only at high or xhigh — never lower, never max).
- 2026-10-09: line-count escalation excludes generated files (lockfiles, api/generated) — evidence: SC orval-pin review had to ask the chief whether an 889-line pnpm-lock.yaml counted (job 20261009-mimo-swarm).
- 2026-10-10: added messaging triggers (ask/DM/reply/inbox) + handoff `comms:` line — evidence: job 20261010-backlog had 0 asks, 0 replies, 0 peer DMs across 15 agents (user request).
- 2026-10-11: scratch copies via git archive (rev-e-tools #self-improvement: cp -r staged main's files into the worktree index); mutation budget ≤5 — evidence: job 20261010-backlog time analysis (300 agent-min: ~65 min in repeated full suites, 30 min mutation, 58% of DeepSeek time in 1-tool turns) — user request.
- 2026-10-11: non-money surviving mutants are nits; mutation cap ~3 min; batched reads — user decision 2026-10-11 (speed job 20261011-speed).
- 2026-10-11: turn-batching section (≥2 tool calls per turn) — evidence: time analysis of jobs 20261010-backlog / 20261011-testspeed / 20261011-grind, 1.1 calls per turn — user approved.
- 2026-10-13: durable-bus usage (inbox before gate, gate --bus/--bus-agent, --client-id, pending); reviewers stream confirmed blockers and pin the reviewed commit - evidence: job 20261012-p1-honest (workers missed chief DMs; bus hardened in f40fd209) - user request.
- 2026-10-13: severity bar aligned with reviewer.md: blocking means a contract break or a reachable wrong value; a missing test means should-fix. Evidence: job 20261012-p1-honest review loop. User approved.
- 2026-10-13: self-improvement made a priority duty; every handoff ends with an `improve:` line, which the chief applies automatically. User request after job 20261012-p1-honest (84 handoffs, no prompt edits during an 8-round loop).
- 2026-10-13: scratch hygiene (no rm combined with other commands; no `timeout` binary). Evidence: rev-liq-c1 #self-improvement post, plus revlite-liq-b3-r2/a1-r2 cleanup failures. Applied automatically.
- 2026-10-13: docs-only checklist item: verify UI claims against frontend/src and check every tier's code path for leverage/default claims. Evidence: revlite-liq-e3 `improve:`. The E3 docs were wrong about validation leverage even though the commit messages read as correct. Applied automatically.
- 2026-10-14: the docs checklist also checks the off branch. Evidence: revlite-liq-e3-r2 `improve:` (the screening leverage claim was false with the mode off). Applied automatically.
- 2026-10-14: board Start line now searches by SCOPE path; posting rules (RULE/WHEN/EVIDENCE, search before posting, verdicts and status off the global board). Evidence: orchestration analysis 2026-10-14: 93% of the posts agents relied on came from their own job; 49% of global posts were task-local; 15 duplicates.
- 2026-10-14: docs checklist: list every key a "sends nothing" claim still sends; doc rg in the worktree — evidence: job 20261014-honest-lev: revlite-hl-lev-t3 S1/S2.
- 2026-10-15: docs checklist: tier scope per bullet, inherited labels contradicted by new facts, before-greps on rendered markup — evidence: job 20261015-honest-safe revlite-hs-doc-1 S1/S2.
- 2026-10-15: scratch-copy data symlinks + python mutant helper. Evidence: revlite-fc-topk-2 (baseline false red MissingBarDataError in scratch, 94 passed in worktree), rev-fc-t2b (silent sed). Applied automatically.
- 2026-10-15 (Land): per-assert kill attribution, no -I for repo CLIs, byte-level quote checks. Evidence: revlite-fc-t4-3, revlite-fc-t7 r1/r2.
- 2026-10-15 (Land, job 20261015-followups): test-deletion check. Evidence: revl-fu-r42-1 should-fix. Applied automatically.
- 2026-10-16 (Land, job 20261016-followups2): no `-p no:logging` in mutant runs. Evidence: revl-fu2-markp1-2 improve line. Applied automatically.
- 2026-10-16 (Land, job 20261016-sizing): docs check names the code outcome of each behaviour verb — evidence revl-sz-docs-1 (min-qty skip vs clamp-up). Applied by the chief.
- 2026-10-16 (Land, job 20261016-followups3): -W overrides ini filterwarnings; probe without it — evidence revl-fu3-swarmtest-1. Applied by the chief.
- 2026-10-17 (job 20261017-grind): never checkout/switch/reset in a maker worktree; read with git show, test in a git archive copy. Evidence: vq-gr-hn HEAD detached to the reviewer's pinned sha.
- 2026-10-17 (Land 4, job 20261017-grind): docs citing unlanded code -> check on the branch, flag land order. Evidence: revlite-gr-t5-docs-1.
- 2026-10-17 (job 20261017-easy3): scratch copies in the session scratchpad when named, deleted before handoff. Evidence: revlite-e3-sources-1 improve line; rev-e3 left /tmp/rev-e3.
- 2026-10-17 (job 20261017-easy4): child-pytest -k --collect-only check. Evidence: revlite-e4-busenv-1.

## .claude/agents/reviewer.md

- 2026-10-09: self-improvement + changelog sections added (user request: agents may improve their own prompts; chief reviews).
- 2026-10-09: tests bullet now requires a mutation check on test-heavy diffs — evidence: SB review B1 (vibe-quant-t4aey) was found only by mutation.
- 2026-10-10: model opus → sonnet; added fix-route line (DeepSeek is the default maker; Sonnet fixes when there are too many problems) — user request.
- 2026-10-10: applied 5 #self-improvement proposals (2-symbol fill probe, gate-alternative simulation, importer test files, real ProcessPool probe, memo eviction worst case) — evidence: board posts from rev-fill-design, rev-dsr-design, rev-z1, rev-p6, rev-p1 (job 20261010-edge-engine).
- 2026-10-10: added messaging triggers (ask/DM/reply/inbox) + handoff `comms:` line — evidence: job 20261010-backlog had 0 asks, 0 replies, 0 peer DMs across 15 agents (user request).
- 2026-10-11: mutation budget (≤5, changed logic, targeted file, git archive), stress loops ≤10, no full-suite re-run — evidence: job 20261010-backlog time analysis (300 agent-min: ~65 min in repeated full suites, 30 min mutation, 58% of DeepSeek time in 1-tool turns) — user request.
- 2026-10-11: surviving mutants outside money code are nits (follow-up bead, no fix round); mutation cap ~3 min; nits alone → fix-route none; async-launch final-status check; batched reads — user decision 2026-10-11 (speed job 20261011-speed).
- 2026-10-11: cache/stamp race check (pre-run key == post-run key, probe mid-run change) — evidence: rev-gate-cache found a false GATE PASS in the swarm-check stamp cache (job 20261011-speed).
- 2026-10-11: turn-batching section (≥2 tool calls per turn) — evidence: time analysis of jobs 20261010-backlog / 20261011-testspeed / 20261011-grind, 1.1 calls per turn — user approved.
- 2026-10-13: durable-bus usage (inbox before gate, gate --bus/--bus-agent, --client-id, pending); reviewers stream confirmed blockers and pin the reviewed commit - evidence: job 20261012-p1-honest (workers missed chief DMs; bus hardened in f40fd209) - user request.
- 2026-10-13: added a severity bar and scoped re-reviews.
  - Severity bar: blocking = current code breaks the contract, or gives a reachable wrong value. Correct code with a missing test, or an unreachable edge, is should-fix. An undecided case means ask the chief.
  - Re-reviews (round ≥2) cover the prior findings plus the delta only.
  - Evidence: job 20261012-p1-honest. Liquidation T3 needed 8 rounds and T4 ≥5. Each fresh reviewer re-hunted the whole surface and blocked on new edges, some unreachable on real data (a fill at exactly end_ns). T4 r5's only blocker was a test gap where production was already correct. User approved.
- 2026-10-13: self-improvement made a priority duty; every handoff ends with an `improve:` line, which the chief applies automatically. User request after job 20261012-p1-honest (84 handoffs, no prompt edits during an 8-round loop).
- 2026-10-13: scratch hygiene (no rm combined with other commands; no `timeout` binary). Evidence: rev-liq-c1 #self-improvement post, plus revlite-liq-b3-r2/a1-r2 cleanup failures. Applied automatically.
- 2026-10-14: board Start line now searches by SCOPE path; posting rules (RULE/WHEN/EVIDENCE, search before posting, verdicts and status off the global board). Evidence: orchestration analysis 2026-10-14: 93% of the posts agents relied on came from their own job; 49% of global posts were task-local; 15 duplicates.
- 2026-10-14: numeric-domain/propagation probes (full-message assertions, 10**400 settings, null-vs-absent inheritance, staged-repair state) and bus kind list — evidence: job 20261014-honest-lev: rev-hl-wfa-err (truncation mutant survived 13/13), rev-hl-lev-t1 B1, rev-hl-lev-t2 B1, rev-hl-data-gap B2.
- 2026-10-15: boundary/plumbing/failure probes (equality edges, factory pass-through, per-branch mutants, non-domain exceptions up to the callback, pre-component checkpoints, failing item first, overridden derivations). Evidence: job 20261015-honest-safe: 7 of 15 tasks needed an extra round only for a surviving mutant of this kind (V1a, V1b, Z1, Z2, P2a, P3a2) and P2b B1 / P3b SF1 were found only by these probes. Proposals from rev-hs-v1b-1, rev-hs-z1-1, rev-hs-z2-1, rev-hs-p2b-1/2, rev-hs-p3b-1, rev-hs-v1a-2, cc-hs-p3a2-3, cc-hs-p2a-2.
- 2026-10-15: design-mode probes for cross-boundary decomposition buckets and injectable research ledgers; scratch-copy data symlinks; python mutant helper (no `sed -i`). Evidence: job 20261015-funding-carry rev-fc-design B1, rev-fc-design-2 SF-a, revlite-fc-topk-2 false red, rev-fc-t2b silent sed. Applied automatically.
- 2026-10-16 (Land, job 20261016-safety-bugs): invalid-probe rule for mutants, cached-caller retry probe for helpers that start raising, on-disk deletion probe for graph changes. Evidence: rev-sf-paperckpt-1, rev-sf-migrate-1 B1, rev-sf-gatesel-1 B1 proposals (18def02a724b9220-1874, 18def075263dc3d0-6436, 18def08282ddf8a0-0851). Applied automatically.
- 2026-10-15 (Land): unique-token mutant anchors, executor-parity field diff first, shared-ts/after-last-decision fixtures, redundant-guard rule. Evidence: cc-fc-t9 gotcha, rev-fc-t3 B1/B2, rev-fc-t5 M1/M2, cc-fc-t6a r2.
- 2026-10-15: ordering/layering/leak probes (mixed-invalid before-validators, one ordered side-effect trace, formatted-log secret checks with chained exceptions). Evidence: job 20261015-small-beads rev-sb-nan-1 B1, rev-sb-alertwrite-1 SF1, rev-sb-watchlog-1 B1/B2; proposals 18deca58a448a468-4631, 18deca71b0606a60-1522, 18deca98c4fa16c0-0853. Applied automatically at T3's 3rd round.
- 2026-10-15 (Land, job 20261015-small-beads): legacy-row migration probe. Evidence: rev-sb-exittiming-1 M2 (proposal 18decbdf5d01a310-1594). Applied automatically.
- 2026-10-15 (Land, job 20261015-followups): helper-conversion probes now cover the full event tuple and the trailing persistence call. Evidence: rev-fu-paperalert-sol-1 SF1/SF2 (proposal 18dece4dee119b38-5493). Applied automatically.
- 2026-10-16 (Land, job 20261016-followups2): admission-check race probe for new awaits, held-worker loop-blocking test, design-mode child-run creators and parameter comparators. Evidence: rev-fu2-markp2-1 B1/SF1 (proposal 18def5c5937c4368-0175), rev-fu2-design SF2/SF3. Applied automatically.
- 2026-10-16 (Land, job 20261016-sizing): settings/latch/restore probes (read-back types, every threaded field, normalized-name guards, re-armed latches, restore lift paths + public start(), localized inputs) — evidence rev-sz-api-1, rev-sz-screen-1, rev-sz-kill-1, rev-sz-design, risk-sz-halt-1, rev-sz-forms-1. Applied by the chief.
- 2026-10-16 (job 20261016-easy-beads-2): list-key uniqueness probe — evidence: rev-nb-felint-1 proposal 18df148eb746d960-9222, B1 CheckpointsList collision.
- 2026-10-16 (job 20261016-easy-beads-3): numeric reachability traces upstream normalization; collection-error reds are not defect evidence — evidence: rev-ec-nansharpe-1 proposal 18df15b6c6236f88-6333, rev-ec-candles-1 proposal 18df15f4ef0528b8-7552.
- 2026-10-16 (job 20261016-safety-bankroll): design-mode held-worker probes through resume/stop/teardown — evidence: rev-sb-design-1 proposal 18df18c88e9afad0-4369, rev-sb-design-2 proposal 18df198cf597ec88-8009.
- 2026-10-16 (job 20261016-safety-bankroll, T7 round 3 sweep): singleton recovery via cached+fresh paths, sub-precision excess probes, persisted-null consumer probes. Evidence: rev-sb-t1-1, rev-sb-t3-1 (18df1ab8bfdb6570-1376), rev-sb-t7-1 (18df1b06fad710b8-5445), rev-sb-t7-2 (18df1b707e9a5388-3925).
- 2026-10-17 (job 20261016-safety-bankroll): settings-grid pooling probes. Evidence: rev-sb-t8-1 proposal 18df45bd43d2b688-8506.
- 2026-10-17 (job 20261017-grind): forwarded-text + precedence-chain probes. Evidence: rev-gr-guard-1 SF1, rev-gr-api-1 N1.
- 2026-10-17 (job 20261017-grind): never checkout/switch/reset in a maker worktree; read with git show, test in a git archive copy. Evidence: vq-gr-hn HEAD detached to the reviewer's pinned sha.
- 2026-10-17 (job 20261017-easy3): scratch copies in the session scratchpad when named, deleted before handoff. Evidence: revlite-e3-sources-1 improve line; rev-e3 left /tmp/rev-e3.
- 2026-10-17 (job 20261017-easy4): design mode covers streams verify skips. Evidence: rev-e4-coins-design-1 SF1.

## .claude/agents/risk-officer.md

- 2026-10-09: self-improvement + changelog sections added (user request: agents may improve their own prompts; chief reviews).
- 2026-10-10: added messaging triggers (ask/DM/reply/inbox) + handoff `comms:` line — evidence: job 20261010-backlog had 0 asks, 0 replies, 0 peer DMs across 15 agents (user request).
- 2026-10-12: code-change review mode — evidence: risk-d3-7 (job 20261012-nextsteps) was asked for a diff review the launch-only template did not cover (proposal 18ddda9fb7f7ad78-6192, applied by chief).
- 2026-10-13: durable-bus usage (inbox before gate, gate --bus/--bus-agent, --client-id, pending) - evidence: job 20261012-p1-honest (workers missed chief DMs; bus hardened in f40fd209) - user request.
- 2026-10-13: self-improvement made a priority duty; every handoff ends with an `improve:` line, which the chief applies automatically. User request after job 20261012-p1-honest (84 handoffs, no prompt edits during an 8-round loop).
- 2026-10-14: board Start line now searches by SCOPE path; posting rules (RULE/WHEN/EVIDENCE, search before posting, verdicts and status off the global board). Evidence: orchestration analysis 2026-10-14: 93% of the posts agents relied on came from their own job; 49% of global posts were task-local; 15 duplicates.
- 2026-10-15: new-endpoint probe (unauthenticated 401/404 + the one signed call for the operator); liquidation overlay is opt-in, read notes when on — evidence: job 20261015-honest-safe risk-hs-1 (proposal 18dead0d8fc58de8-7685); the 'screening doesn't model liquidation' line was stale since 2026-10-13.
- 2026-10-15 (Land, job 20261015-small-beads): chained-exception repro on the merged job branch. Evidence: risk-sb VETO of T4 (repro LEAK True), single-branch rerun inconclusive. Applied automatically.
- 2026-10-15 (Land, job 20261015-followups): shutdown-order check against stop_async. Evidence: risk-fu-paperalert-1 improve line. Applied automatically.
- 2026-10-16 (Land, job 20261016-safety-bugs): callee-moved-into-finally chained-exception repro. Evidence: risk-sf-paperckpt-1 improve line. Applied automatically.
- 2026-10-16 (Land, job 20261016-followups2): repro on the base commit too. Evidence: risk-fu2-paper-1 improve line. Applied automatically.
- 2026-10-16 (Land, job 20261016-sizing): code-change review checks (public start() repro, halt-reason replacement, portfolio total of per-symbol caps) — evidence risk-sz-halt-1, risk-sz-kill-1, risk-sz-refuse-1. Applied by the chief.
- 2026-10-16 (Land, job 20261016-followups3): fresh-connection repro for seeded rows — evidence risk-fu3-sysstate-1. Applied by the chief.
- 2026-10-16 (job 20261016-safety-bankroll): emergency action reachable in UI error states; carve-out probes. Evidence: risk-sb-t1-1 F1 (proposal 18df1a81e03c66e8-8873), risk-sb-t1-2.
- 2026-10-17 (job 20261016-safety-bankroll): base ancestry check, same-step live-node repro for pending sums, interrupted-command status check. Evidence: risk-sb-t2b-1, risk-sb-t4-1 F1, risk-sb-t2b-2 nit (proposals 18df1cbd9b5008a8-2068, 18df4584e8f34378-1248).
- 2026-10-17 (job 20261017-grind): CancelledError per-item flag probe; real-engine base-vs-head repro for gate fail-safes. Evidence: risk-gr-paperops-1 M1, risk-gr-guard-1.
- 2026-10-17 (Land 2, job 20261017-grind): alert-latch repro with the real _write_then_alert. Evidence: risk-gr-alert-1 M3.

## .claude/agents/strategy-author.md

- 2026-10-09: self-improvement + changelog sections added (user request: agents may improve their own prompts; chief reviews).
- 2026-10-10: added messaging triggers (ask/DM/reply/inbox) + handoff `comms:` line — evidence: job 20261010-backlog had 0 asks, 0 replies, 0 peer DMs across 15 agents (user request).
- 2026-10-13: durable-bus usage (inbox before gate, gate --bus/--bus-agent, --client-id, pending) - evidence: job 20261012-p1-honest (workers missed chief DMs; bus hardened in f40fd209) - user request.
- 2026-10-13: self-improvement made a priority duty; every handoff ends with an `improve:` line, which the chief applies automatically. User request after job 20261012-p1-honest (84 handoffs, no prompt edits during an 8-round loop).
- 2026-10-14: board Start line now searches by SCOPE path; posting rules (RULE/WHEN/EVIDENCE, search before posting, verdicts and status off the global board). Evidence: orchestration analysis 2026-10-14: 93% of the posts agents relied on came from their own job; 49% of global posts were task-local; 15 duplicates.
- 2026-10-14: ask the chief on thesis vs engine_notes conflicts; no unregistered sensitivity arms — evidence: job 20261014-honest-lev: author-hl H-2, ruling R8.

## .claude/agents/verifier.md

- 2026-10-09: self-improvement + changelog sections added (user request: agents may improve their own prompts; chief reviews).
- 2026-10-10: added messaging triggers (ask/DM/reply/inbox) + handoff `comms:` line — evidence: job 20261010-backlog had 0 asks, 0 replies, 0 peer DMs across 15 agents (user request).
- 2026-10-11: turn-batching section (≥2 tool calls per turn) — evidence: time analysis of jobs 20261010-backlog / 20261011-testspeed / 20261011-grind, 1.1 calls per turn — user approved.
- 2026-10-13: durable-bus usage (inbox before gate, gate --bus/--bus-agent, --client-id, pending) - evidence: job 20261012-p1-honest (workers missed chief DMs; bus hardened in f40fd209) - user request.
- 2026-10-13: self-improvement made a priority duty; every handoff ends with an `improve:` line, which the chief applies automatically. User request after job 20261012-p1-honest (84 handoffs, no prompt edits during an 8-round loop).
- 2026-10-14: UI-check hints (Radix date pickers, cap end_date at catalog coverage). Evidence: ver-liq-ui `improve:`; runs 919/920 failed with MissingBarDataError. Applied automatically.
- 2026-10-14: board Start line now searches by SCOPE path; posting rules (RULE/WHEN/EVIDENCE, search before posting, verdicts and status off the global board). Evidence: orchestration analysis 2026-10-14: 93% of the posts agents relied on came from their own job; 49% of global posts were task-local; 15 duplicates.
- 2026-10-14: UI check: cap liquidation windows at MARK coverage; re-snapshot refs, Radix options via eval — evidence: job 20261014-honest-lev: ver-hl-lev runs 923/924.
- 2026-10-16 (Land, job 20261016-safety-bugs): real-DB smoke recipe (Path, backup copy, main-vs-branch baseline). Evidence: ver-sf-job improve line + unresolved migration finding. Applied automatically.
- 2026-10-16 (Land, job 20261016-sizing): choose live-AC windows from the first trade date — evidence ver-sz-job-1 runs 920/921. Applied by the chief.
- 2026-10-16 (job 20261016-easy-beads-3): seeded render-only fallback for rare-row UI criteria — evidence: nq6vg runs 920/921 produced no liquidated rows.
- 2026-10-17 (job 20261016-safety-bankroll): clearing inputs in agent-browser. Evidence: ver-sb-1 improve line.
- 2026-10-17 (job 20261017-grind): number-input leading-0 note. Evidence: ver-gr-2.
- 2026-10-17 (Land 3, job 20261017-grind): lockfile check via temp dir (uv.lock gitignored). Evidence: ver-gr-3.
- 2026-10-17 (job 20261017-easy5): open cited screenshots; tiny discovery runs give 0 rows. Evidence: ver-e5-ingest-ui-1, ver-e5-badge-1.
- 2026-10-17 (job 20261017-easy6): archive SQL via Python with an absolute ro URI. Evidence: ver-coins-t4-1.

## .claude/skills/orchestrate/SKILL.md

- 2026-10-09: added swarm bus/board steps, chief digest each loop pass, self-improvement section (user request).
- 2026-10-09: cmd timeout ≥ 60 min + no in-place script edits — evidence: F1 follow-up killed at 40 min while finishing (job 20261008-followup2); cmd-task.sh mid-swarm edit crashed wrapper summaries (job 20261009-mimo-swarm).
- 2026-10-09: pre-review gate (swarm-check.sh) + tiered review (reviewer-lite on Haiku 5.5) — user request; evidence: ~half of fix rounds were mechanical, Opus reviewed one-line test nits.
- 2026-10-10: stall supervisor, needs-user queue, telemetry steps (user request #8/#9/#4); evidence: MiMo/DeepSeek ~2 h read-loop stalls went unnoticed until a network drop (job 20261009-swarm-tooling).
- 2026-10-10: Command Code dispatch moved to T3 `delegate_task` subagents (async, unique clientRequestId, taskId in ledger, no polling/t3_thread_send/thread launch, errors verbatim, no fallback, verify before reporting); stall_watch/cmd-task.sh no longer used for swarm workers — user request.
- 2026-10-10: DeepSeek = default maker, Sonnet = reviewer + fixer on fix-route sonnet; chief board watching made explicit (tail + digest each pass, act on interesting posts) — user request.
- 2026-10-10: keep proofs fast — no long discovery reruns in briefs; tiny settings otherwise — user request (cc-dsr-finish 878 rerun took ~30 min).
- 2026-10-10: messaging triggers (ask/DM/reply/inbox), `{PEERS}` roster in briefs, handoff `comms:` line — user request; evidence: job 20261010-backlog 0 asks/0 replies/0 peer DMs across 15 agents.
- 2026-10-11: speed-first briefs (code excerpts, batched reads, targeted tests + one gate, E2E -g, async final status), accept swarm-check PASS cache, non-money nits → beads — user decision 2026-10-11 (speed job 20261011-speed).
- 2026-10-11: Land = one verbose commit per task (commit-message.md), not one squash per job — user request (commits are history/context for future agents).
- 2026-10-11: briefs paste the cmd-task SPEED RULES block verbatim for every maker/fixer (batch ≥2 calls/turn, gate once per round as last command, comma-separated --scope) — evidence: speed analysis of jobs 20261010-backlog/20261011-testspeed/20261011-grind (1.1 calls per turn, full-suite+gate ~21 min flat) — user approved.
- 2026-10-11: T3/Codex reviewers get the bus block with required inbox/done post + stacked pairs in PEERS; Codex usage-limit failure → immediate Sonnet re-dispatch; input-domain table in briefs for parser/translator changes — evidence: job 20261011-grind2 (round-1 reviewers had optional bus; yubhb 4 rounds; Codex limit killed rev-g2-translator-r2 with no output).
- 2026-10-13: stream confirmed review findings to the maker mid-review (chief relays to idle makers; reviewers pin the commit); gate --bus/--bus-agent; `pending` - user request + evidence: job 20261012-p1-honest (an idle implementer never read a bus DM; streamed findings cut a round each on T2b and bus hardening).
- 2026-10-13: four changes to stop the review loop, plus automatic self-improvement.
  - Money tasks get a semantics contract first, in every profile.
  - Rulings are appended to the contract.
  - Re-reviews are scoped to the prior findings plus the delta.
  - Stop after 2 failed rounds and do a design pass.
  - Self-improvement is automatic: the chief edits agent definitions without asking, and reports the change.

  Evidence: job 20261012-p1-honest, 84 handoffs. T3 needed 8 rounds, T4 5+, T2b 6 and bus hardening 6, with about 10 mid-review rulings (R1–R6b). The chief made no prompt edits during the loop. User request.
- 2026-10-13: self-improvement raised to a top-priority duty: a header line, a mandatory per-agent sweep at Land and at round 3, a hard rule, and a mandatory `improve:` handoff line. User request.
- 2026-10-13: write the brief file before dispatching a pointer-prompt. Evidence: the cc-liq-a1 dispatch raced its brief file. Applied automatically.
- 2026-10-13: fix-round gate scope = the whole task scope. Evidence: false scope FAILs on A1 r2 and T4 r2. Applied automatically.
- 2026-10-13: stacked-task gate command needs --base <parent>. Evidence: B1/B2/B3 scope FAILs; cc-liq-b2 found the fix. Applied automatically.
- 2026-10-13: brief scope includes the existing test files the task must edit. Evidence: cc-liq-b3 `improve:` line; B1/B3 scope FAILs on fixture edits. Applied automatically.
- 2026-10-13: landing a stacked child applies only its own delta (`git diff <parent tip> <child tip> | git apply`). Evidence: merging B2 with `--squash` conflicted against the A1 squash. Applied automatically.
- 2026-10-14: close UI beads only after the on-screen check. Evidence: 2gyk1.1.10 was closed before ver-liq-ui reported (it passed, but the order was wrong). Found in the land-step sweep.
- 2026-10-14: orchestration analysis of 10-08..10-13 (516 runs, 1,692 messages, 381 graded board posts). Five changes:
  - The start-up `board search` by SCOPE replaces `--recent 30`. 93% of the posts agents relied on came from their own job, and a post left the window after a median 5.5 h.
  - Board hygiene rules.
  - Rulings become numbered contract rows that record their round. 24 of 54 questions and blockers were mid-flight liquidation rulings.
  - Every brief gets the bus block and a TELEMETRY line, checked with `rg --files-without-match` (originally written as `rg -L`, which is `--follow` in ripgrep; fixed 2026-10-15).
  - `telemetry.py ingest-runs` replaces hand-recording.

  The check is cheap insurance. The analysis's first claim that Codex re-reviews lacked the bus block was a parsing artifact: 77 of 80 Codex reviewers used the bus, median 9 calls. User request.
- 2026-10-14: dispatch-loop bullets: stacked child may start during parent review, rg invalidated strings when splitting tasks, hold shared-data writes while readers run, reviewers use --kind info — evidence: job 20261014-honest-lev (T2 during T1 review; cc-hl-lev-t1 help text; SOL backfill vs ver-hl-lev; rev-hl-wfa-err bus kind rejection).
- 2026-10-15: `rg -L` -> `rg --files-without-match` (ripgrep -L is --follow); fix-round briefs quote bullet text; reviewer fallback when Sol is unavailable; multi-parent stacked bases via merge + `--base <merge sha>` — evidence: job 20261015-honest-safe (brief check printed matches; cc-hs-doc-2 line drift; user stopped Sol mid-job; V2 and P3b bases).
- 2026-10-15: TELEMETRY line inside pointer prompts — evidence: job 20261015-honest-safe telemetry rows were keyed by the name heuristic only.
- 2026-10-15: rulings name exception payload + raise site; resume limit-stopped Claude agents via SendMessage; maker template: inbox before commit, run every caller test file, python mutants, sibling-API stub. Evidence: job 20261015-funding-carry (T1 rounds 2-3, pqon2 r1 gate gap, rev-fc-t2b silent sed, cc-fc-t2b sibling API). Applied automatically at T1's 3rd review round.
- 2026-10-15 (Land, job 20261015-funding-carry): land stacked tasks from each fork point + land script + reset on non-empty tree check; fix briefs quote reviewer fix text verbatim, name fixture builders, pin artifact schemas, state sibling-task dependencies; single-quoted bus bodies; probes in the job dir. Evidence: T1 revert caught by tree check; T6a r2 regression; cc-fc-topk/t2a/t6a/t7/t3 improve lines.
- 2026-10-15 (T3 round 3, job 20261015-small-beads): absolute paths when appending rulings. Evidence: R1c landed in the repo root and rev-sb-watchlog-1 asked for it. Applied automatically.
- 2026-10-15 (Land, job 20261015-small-beads): collateral test pins for new API models/metrics fields; design refs by heading; restart recovery checks task_status before re-dispatch. Evidence: cc-sb-exittiming T7a-R1/R2 rulings, rev-sb-exitui-1 proposal 18decc452a816d38-4486, duplicated T7a run after the restart. Applied automatically.
- 2026-10-15 (job 20261015-followups): Sol is the default reviewer and is available; fall back only on a live capability check or a failed dispatch, never on an old note. Evidence: the chief sent T1/T2 to Sonnet on the stale 'no GPT-6.1-Sol' line; user correction.
- 2026-10-16 (Land, job 20261016-followups2): board backup before reset --hard, quoted heredocs for commit messages, merge-base for reviewer SHAs. Evidence: lost board posts and a vanished verification line in this job's land; rev-fu2-paper-1 base correction. Applied automatically.
- 2026-10-16 (Land, job 20261016-safety-bugs, 2nd pass): reviewer ranges pinned as SHA..SHA; rulings appended under the task heading. Evidence: rev-sf-refusal-1 proposal 18def19d7a644438-0623 and its M7/M2b question.
- 2026-10-16 (Land, job 20261016-safety-bugs): every caller of a helper whose failure behaviour changes gets a contract row; `ask-user` replaces the non-existent `needs-user ask`. Evidence: T4 M7/M6 mid-flight rulings and a 3-round task; chief's `needs-user ask` call failed with 'invalid choice'.
- 2026-10-15 (Land, job 20261015-followups): rg every brief path/glob before dispatch; enumerate all sites for an 'every' goal; symbol anchors in fix rounds; name existing tests that pin unchanged rows. Evidence: cc-fu-ledgerbook, cc-fu-paperalert r1/r2 and cc-fu-busreply improve lines.
- 2026-10-16 (Land, job 20261016-sizing): stacked-task reviewer base = merge-base(parent, child), not the parent tip; brief generators slice by full heading; mid-job model switches. Evidence: rev-sz-halt-1/risk-sz-halt-1 got 7e793f70 (K1 r2 tip) instead of 4768605b and saw a phantom -70-line test deletion; S1 brief carried the MARK/thin-tier contract; MiMo disabled + Sol token outage mid-job.
- 2026-10-16 (Land, job 20261016-followups3): scope includes monkeypatched doubles of changed signatures; chief perf figures measured serially (CLAUDE.md speed rule). Evidence: F2-R1/R1b asks; F4-R1 and the NATR no-gain result.
- 2026-10-16 (Land, job 20261016-easy-beads): old-message-text tests in SCOPE, case-safe helper names, bounded parity notes to reviewers. Evidence: cc-eb-papercmd-1 ruling R1, cc-eb-feui-1 TS1261, rev-eb-sweep-1 B1 ruled nit.
- 2026-10-16 (Land, job 20261016-easy-beads-2): rename briefs rg old dotted paths + method doubles; lint counts from the tool's severity output; old-base rule after a mid-job force-push. Evidence: cc-nb-dsrobs-1 R2/R3, cc-nb-felint-1 FACTS error, nb-multiseed gate scope FAIL.
- 2026-10-16 (Land, job 20261016-easy-beads-3): word-boundary reader searches in removal briefs; multi-layer hypotheses in diagnosis briefs. Evidence: cc-ec-dropdaily-1 and cc-ec-candles-1 self-improvement posts.
- 2026-10-17 (Land, job 20261016-safety-bankroll): merge every task tip incl. stacked parents' test-only rounds; write brief files before spawning; API-field hand-off files and compiled-source goldens in SCOPE. Evidence: land tree check (wiring test), rev-sb-t6-1 missing brief, T4-R1, T5-R1.
- 2026-10-17 (job 20261017-grind): § Asking the user — decisions use simple terms + example flow + numbers table + what breaks + recommendation. User request ("this was very good").
- 2026-10-17 (Land, job 20261017-grind): two-bead briefs name commit order; verifier briefs carry SCOPE; T3 bus is pull-only; bd --append-notes. Evidence: cc-gr-api-1 improve line, ver-gr-1 UNVERIFIED scope, hermetic/paperops-2 late-DM gate fails, 6osqd notes wipe.
- 2026-10-17 (Land 2, job 20261017-grind): scope glob for top-level tests; predicate rulings name what must never match; ruling-caused blockers may go to DeepSeek. Evidence: cc-gr-maxlev-1 scope FAIL, rev-gr-arctic-1 B1, user question.
- 2026-10-17 (Land 4, job 20261017-grind): docs-only briefs use red probes, docs land after code; new archive tables go into SPEC's schema block. Evidence: cc-gr-t5-docs-1 posts, revlite-gr-t5-docs-1.
- 2026-10-17 (Land, job 20261017-easy3): re-arm the chief Monitor on expiry + `pending`. Evidence: cc-e3-sources-1 ask timed out unanswered. cmd CLI launch rule moved to cheap-agents.md.
- 2026-10-17 (Land, job 20261017-easy4): round-N briefs paste the literal gate command. Evidence: cc-e4-carry-2 improve line.
- 2026-10-17 (Land, job 20261017-easy5): fix briefs name the mechanism to call and each type's file. Evidence: cc-e5-chart-1/2 self-improvement posts.

## docs/orchestration/cheap-agents.md

- 2026-10-10: dispatch moved from `cmd-task.sh` to T3 `delegate_task` subagents (`commandcode_command`), with the user's protocol (async, unique `clientRequestId`, keep `taskId`, no tight polling, no `t3_thread_send` / `t3_thread_launch` / `create_threads`, errors verbatim, no provider fallback, verify before reporting). Tier slugs updated to the T3 catalog — user request.
- 2026-10-10: DeepSeek is the default maker for all coding tasks; Sonnet reviews and fixes on `fix-route: sonnet` (user: "the workhorse should be deepseek because it is almost free").

- 2026-10-11: allowed Command Code models reduced to `deepseek/deepseek-v4.1-flash` + `xiaomi/mimo-v2.6-pro`; `fast`/`long` tiers and all other alternates removed — user: "DONT run Kimi K3" (job 20261011-testspeed).
- 2026-10-11 17:16: Command Code restricted to `deepseek/deepseek-v4.1-flash` effort high (MiMo `pro` tier removed); reviews on GPT-6.1-Sol via Codex, Sonnet/Haiku for implementation and fixes, Opus medium for planning — user rule (cost).
- 2026-10-12: no mid-round scope changes by DM; new round instead — evidence: job 20261012-nextsteps cc-d3-4a-feedrag-3 missed a scope DM (chief).
- 2026-10-15: MiMo V2.6 Pro allowed again for difficult tasks; DeepSeek V4.1 Flash high stays the default; reviews GPT-6.1-Sol high — user (job 20261015-small-beads).
- 2026-10-15 (job 20261015-small-beads): first MiMo V2.6 Pro task after re-allowing it (T5, real-NT test) had 0 worktree edits after ~90 min when a T3 restart cancelled it, so the chief escalated to a Sonnet implementer (done in ~7 min). Correction: the MiMo run then RESUMED by itself, worked in the same worktree in parallel, and reached the same verified result (probes, red run, 26 passed); its 0-edit period was probing in /tmp, not a stall. Check probe output, not only worktree edits, before calling a MiMo stall.
- 2026-10-16 (job 20261016-safety-bugs): DeepSeek V4.1 Flash made 4 tasks (T1 paper, T3 gate, T4 migrations, T5 discovery API) at 1-2 rounds each; 2 needed a Sonnet fix after Sol found a blocking miss (T3 deleted-module edge, T4 cached half-migrated connection). MiMo V2.6 Pro did the real-NT lifecycle test (T2) in ~45 min (first worktree edit ~20 min in), passed Sol first round.
- 2026-10-16 (job 20261016-sizing): MiMo V2.6 Pro disabled — user, after reviewing its record: clean first rounds (safety-bugs T2 Sol PASS; sizing K1 no code defect, 2 test gaps, risk PASS) but ~31-34 min and 1.9-5.2M input tokens per task vs DeepSeek ~7 min, plus 37 min / 2 h / 90 min stalls (10-08, 10-09, 10-15). DeepSeek failures (S2, S4 blocking) were recovered by a ~4 min Sonnet fix. Hard tasks: DeepSeek, Sonnet implementer on escalation.
- 2026-10-17 (job 20261017-grind): DeepSeek V4.1 Flash made 5 small tasks (8 beads) + a read-only audit (5byet) in parallel; all gate PASS first try; 2 needed a test-only round 2 (guard SF1 from Sol, paperops J5 from Sonnet), 0 production defects found by any reviewer. User then said "dont use sol for review, no limit": new reviews went to the Claude Sonnet `reviewer` (reviewer-lite for test-only deltas); one running Sol review was kept at the user's request.
- 2026-10-16 (job 20261016-easy-beads): DeepSeek V4.1 Flash made 5 small tasks (9 beads) in parallel, all gate PASS first try, ~10-25 min each; GPT-6.1-Sol at reasoningEffort MEDIUM (user override) passed all 5 in round 1. Its one blocking claim came from an over-broad chief parity note and was ruled a nit. Medium looks enough for small, clearly specified diffs; keep high for money-path semantics.
- 2026-10-17 (job 20261017-easy3): cmd CLI launch rule when T3 is missing (run_in_background per worker; liveness by log mtime/bus, not ps). Evidence: duplicate makers in vq-easy3-sources/intervals.


## .claude/skills/grind/SKILL.md

- 2026-10-11: subagents allowed when run under /orchestrate; commit format now points at commit-message.md (CLAUDE.md verbose per-task commits) — user: "the grind skill is just high-level stuff ... you can still use sub-agents" (job 20261011-testspeed).
