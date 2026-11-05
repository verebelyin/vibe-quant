# tests/unit speed

Related: [user-preferences.md](user-preferences.md), [decision-2026-11-05-prune2-scope.md](decision-2026-11-05-prune2-scope.md).

## Goals
- 2026-10-28, user: "i want good performance here, that is about 120 s unit test, swarm check 60s" (source: data/swarm/20261028-prune/brief.md:33). Chief reading then: full suite under `-n auto` incl. timing set ~120 s; `swarm-check --full` ~60 s.
- 2026-11-05, user: "we need quicker test runs!!! prio".

## Measurements (verbatim from sources)
- 2026-10-22, main 5acb51a7: 10113 tests collected (bead vibe-quant-qlx46).
- 2026-10-28, job 20261028-prune: 10781 → 9108 tests; serial wall 915.80 s → 593.48 s; parallel 97.58 s + 33.27 s timing set (load 3.96) (bead vibe-quant-tqy5r).
- 2026-11-03, main 9f0b34b0: 9503 collected (+395 since qlx46) (bead vibe-quant-tqy5r notes).
- 2026-11-05, main 88a31720 (data/swarm/20261105-prune2/evidence/baseline_meta.txt, 10 cores): 9666 collected; parallel not-timing 136 s (load 4.48 start, 17.63 end); timing set 38 s; serial not-timing 754 s (started at load 10.84 — script bug, comma decimal; per-test times slightly inflated). Summed per-test 742.1 s.
- 2026-11-05 finding: the slowest files are agent-tooling tests (parity_mutants 54.4 s, swarm_check_scan_index 40.7 s, mkbrief 36.2 s, swarm_check 24.8 s, power_layer2 18.3 s, mutate 16.7 s ...), ~35% of summed seconds. The bead tqy5r scope files (test_plugins/, math_precision, genome, ...) are not in the top 40 by seconds.

## What we learned
- 2026-10-28 (job 20261028-prune brief): parallel wall is bounded by the slowest single tests (xdist critical path) and the serial timing set, so slow-test work beats count work for wall time.
- 2026-11-03: the suite grew +395 tests in 6 days after a prune. Pruning alone does not hold the line; see [env-gaps.md](env-gaps.md) (growth guard).
- 2026-11-05 (lane D, handoff data/swarm/20261105-prune2/handoffs/cc-spdpar-1.md): the parallel pass is CPU-bound: `time -p` 837 s user+sys (742 test bodies, 55 collection = 11 x 4.9 s, 20 fixtures) => 84 s floor on 10 cores; the 136 s baseline got ~6 cores. Rejected by measurement: `-v`, autouse fixtures (<= 1.8 ms/test), `-n 8`. sys time ~192 s (~23%), Likely: subprocess-heavy tooling tests. Timing set: 17.6 s is ONE test in test_discovery_fast_evaluator.py. Lever = cut summed CPU seconds, not xdist config.
- 2026-11-05 ~21:45, job branch swarm/job-prune2 e2c37958 (all 6 lanes merged): `pytest tests/unit -n auto --dist worksteal -m "not timing"` 9665 passed, 1 failed (pre-existing wall-clock flake, bead vibe-quant-n88un), 117.65 s at load ~10-24 (data/swarm/20261105-prune2/evidence/job_parallel.log). Quiet-host after-measurement pending.
- 2026-11-05 ~22:30, AFTER job 20261105-prune2, job branch ced1e121 (evidence/after_meta.txt, after_perfile.txt): 9683 collected; parallel not-timing **104 s** (load 3.84 start, 14.43 end); timing set **36 s**; serial not-timing **553 s** (load 4.87 start); summed per-test **540.9 s**. vs baseline 88a31720: parallel 136 -> 104 s, timing 38 -> 36 s, total 174 -> 140 s (goal ~120 s), serial 754 -> 553 s, summed 742.1 -> 540.9 s (-27%). Count +17 (new guard tests); count target (<= 8085) not pursued per [decision-2026-11-05-prune2-scope.md](decision-2026-11-05-prune2-scope.md).
- 2026-11-05 (impl-cache-1 finding, chief re-measured): `/usr/bin/git` is the macOS xcrun shim: 20.2 ms per spawn vs 6.4 ms for /Applications/Xcode.app/Contents/Developer/usr/bin/git (30 x `git --version`). impl-cache-1: test_mkbrief+test_land+test_swarm_check serial 46.7 s -> 27.7 s with the direct git first on PATH (load ~14). ~1700 git spawns in 8 tooling test files. Chief ruled no PATH rewiring in repo code; machine-level fix proposed to the user.
- 2026-11-05: user approved (A) a machine-level fix: `~/.local/bin/git` -> /Applications/Xcode.app/Contents/Developer/usr/bin/git (~/.local/bin precedes /usr/bin on PATH). `git` spawn 20.2 ms -> 7.0 ms (30 x --version). Undo: delete the symlink; breaks if Xcode is removed. Benchmarks after this date run with the fast git; compare A/B only within one PATH.
