---
name: perf-profiling
description: Performance profiling playbook for vibe-quant backtests — how the 300× and 4.4× wins were found. Use when investigating slow discovery/screening/validation runs, slow indicators, or any "the engine is slow" report.
---

# Performance Profiling Playbook (how the 300× and 4.4× wins were found)

1. **Read the engine's own numbers first.** Run one eval with
   `VIBE_QUANT_NT_LOG_LEVEL_SCREENING=INFO` (validation logs INFO by default) and read:
   `Added N ... Bar elements` (data volume ground truth), `Read N events from parquet in Xs`,
   `Engine load took Xs`, `Iterations: N`. Iterations ≫ expected bars ⇒ data bug, not slow compute.
2. **Time phases coarsely before profiling**: compile (`_ensure_compiled`), data load, engine
   run. Most "slow engine" reports are actually one phase.
3. **cProfile only on SHORT windows** (1–2 months). Full-window + profiler overhead exceeds
   command timeouts. `py-spy` needs root on macOS. Sort by `tottime` to find pandas
   internals; sort by `cumulative` to attribute them to indicator functions.
4. **Discovery `s/chromosome` is wall time across ~8 workers** — multiply by worker count for
   per-eval CPU. A "1.6s/chromosome" gen can hide 100s-CPU genomes.
5. **pandas-ta functions with per-element `.iloc` loops are pathological** (KAMA was 62% of
   eval time at ~220 pandas calls/bar). Fix = exact numpy port of the recurrence keeping the
   library's vectorized prep, guarded by **zero-tolerance equality tests**
   (`tests/unit/test_plugins/test_kama_exactness.py` is the template). `ta.adx` is the next
   known target (bead vibe-quant-3f0er).
6. **Every perf change ships with an exactness proof** — see CLAUDE.md § Verification Rules.
