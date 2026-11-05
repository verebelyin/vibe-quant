# 2026-11-05 — prune job 20261105-prune2 scope

User answered "all rec" on 2026-11-05 to three questions (job brief: data/swarm/20261105-prune2/brief.md):
1. Prune only (bead vibe-quant-tqy5r); id-stripping rename (vibe-quant-t8hme) is a later, separate job. Supersedes the 2026-11-03 plan (tqy5r notes) to run both in one job.
2. Target wall time first: measure per-file and per-test durations, prune by seconds saved; the count target (<= 8085) is secondary.
3. Chief wiki lives at docs/wiki/ (this folder).

Related: [test-suite-speed.md](test-suite-speed.md).

## Plan approved 2026-11-05 ("all rec")
- Lanes by seconds saved (baseline at main 88a31720): A1 mutant/scan-index tooling tests (tqy5r.1), A2 swarm tooling tests (tqy5r.2), B engine/indicator/parity (tqy5r.3, money: full reviewer + mutate --compare-rev 88a31720), C paper (tqy5r.4, money: full reviewer + risk-officer reads delete list), D xdist efficiency + timing set (tqy5r.5, after a writer frees), E count-only (only if count > 8085 after A–D).
- Slow-test guard (bead vibe-quant-ektwt): gate fails on a new/changed test > 2 s alone unless `@pytest.mark.slow(reason=...)`. User chose A over "no guard".
- 2026-11-05 19:07Z: A1, A2, B, C, G dispatched to DeepSeek V4.1 Flash makers (T3), briefs in data/swarm/20261105-prune2/prompts/.

## Round-2 answers 2026-11-05 ("1 b, 2 a, 3 a, 4 a")
1. B: halve kill-switch timing SCALE 0.25 -> 0.125 (user accepts flake risk under load) — job branch dddb9e84.
2. A: production caching — user 2026-11-05 later: "fix this now" (same job, lane CACHE); was: next job = in mkbrief/swarm_check (bead vibe-quant-3phet) to close the ~20 s gap to ~120 s.
3. A: money/gate makers -> Sonnet (`maker-money` route) — see [routing-observations.md](routing-observations.md).
4. A: gate shares CPU across concurrent gates (bead vibe-quant-vdf7c) — user: do it now (lane SLOTS).
