# Jobs 20261120-testspeed and the test prune (2026-11-20 .. 11-21)

Source: data/swarm/20261120-testspeed/ (brief.md, ledger.md, handoffs/); prune thread "Test Complexity and Mutation Testing Policy" (bead vibe-quant-lm77z).

- User 2026-11-20: no dedicated test for code with cognitive complexity <= 5 (cccc) except regexes, money/metrics/fills/gates, SQL and DSL codegen; mutants optional and only for money code; parity check removed ("it does not give any value"). Rule now in CLAUDE.md § Tests prove outcomes.
- Prune: 1,274 tooling tests and 127 product tests deleted (f2628337, a50f4eea, eebb8023, 40a6d8ac, 3e234a54); 2 kill-zone tests restored (d93883b8); `simple-tests` gate check added (3fd5f330); parity removal landed (5e48dffe).
- testspeed beads: fast gate with fixture readers and a per-file PASS cache (468e0239, sfghw), network guard + synthetic warm-up for every test (9146a256, q8uv6), land.py keeps .vq-base of a merged task (8872f9ba, btg4k).
