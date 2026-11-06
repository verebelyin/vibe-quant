# Model routing observations (dated)

Routing source: docs/orchestration/models.json (`models.py route <role>`). This page records outcomes that may justify a routing change; the change itself needs evidence from >= 2 jobs.

- 2026-11-05, job 20261105-prune2: DeepSeek V4.1 Flash makers, first 2 reviewed lanes both BLOCKed by the GPT full reviewer:
  - lane C (paper tests): speed fixture copied a production helper (`wait_future`) -> a timeout mutant went base=KILLED head=SURVIVED (handoff rev-spdpaper-1.md).
  - lane G (slow-test guard, new gate feature): 5 blockers, 4 of them fail-open paths (missing junit, failed rerun, deletion-only hunk, unbounded reruns) + its own 11.92 s test (handoff rev-spdguard-1.md).
  - Both went to Sonnet fixers (impl-spdpaper-2, impl-spdguard-2).
  - lane B (engine/indicator, money): BLOCK too — a fixture with constant taker fraction lost a look-ahead mutant kill, and a funding e2e row was deleted after the chief's "no spec/e2e deletions" ruling (handoff rev-spdeng-1.md).
  - Lanes A1, A2 (tooling tests, reviewer-lite): SHOULD-FIX only. Score 2026-11-05: DeepSeek 3 of 3 money/gate lanes BLOCKed; 0 of 2 tooling-test lanes BLOCKed.
  - Hypothesis (unconfirmed): new gate/check features (fail-closed logic) should route to Sonnet `implementer` directly; DeepSeek stays on bulk test edits with clear mechanical targets.

Related: [env-gaps.md](env-gaps.md), [test-suite-speed.md](test-suite-speed.md).
- 2026-11-05: user decided (3=A) after the score above: new route `maker-money` -> Sonnet for money, metric and gate code (job branch c42d2399, lands with job 20261105-prune2). DeepSeek stays `maker` for tooling and bulk test edits. Supersedes the "hypothesis (unconfirmed)" line above.
- 2026-11-06 (retro): Sonnet-built gate lanes in the same job were BLOCKed too: SLOTS 4 findings (SIGTERM cleanup, pid reuse, explicit gates invisible, race), CACHE 2 (unrelated-history behaviour change, 1 s test). Severity differs: DeepSeek blocks were lost mutant kills / fail-open checks; Sonnet blocks were lifecycle edge cases and a rare behaviour change. Reviewer GPT-6.1 sol caught all of them; review stays mandatory whatever the maker.
- 2026-11-06, job 20261106-backlog (7 lanes, main 223147f4): GPT-6.1 sol reviewer verdicts: BLOCK 5 (slowguard Sonnet, emitfstr DeepSeek, atrform DeepSeek, ruletest Sonnet, t3hist DeepSeek), SHOULD-FIX 1 (chiefux DeepSeek); Haiku reviewer-lite PASS 1 (papertests Sonnet, test-only). Every BLOCK was a real bug with a runnable repro (unicode byte offsets mutating inside f-string fields; ATR sizing dropped on sweep->validation; failed NT run exiting 0 with zero metrics; ~/.t3 overwrite; slow-guard escapes across trivia). Block rate by maker: DeepSeek 3/4, Sonnet 2/3 — no routing change indicated; review stays mandatory.
- 2026-11-06: GPT reviewer filed findings under `claims:` instead of `open:` twice (rev-chiefux-1, rev-ruletest-1); mkbrief --prior-from copied only open: -> fixer missed SF2. Fixed on job branch cffad7a8 (graded claims carried).
