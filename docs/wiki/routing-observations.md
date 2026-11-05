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
