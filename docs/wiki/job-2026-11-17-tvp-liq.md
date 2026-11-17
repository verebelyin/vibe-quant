# Job 20261117-tvp-liq (2026-11-17)

Source: data/swarm/20261117-tvp-liq/ (brief.md, design.md, ledger.md, handoffs/). Landed on main as a55d94dc, 7e7d8a96, 50569b4b (pushed 2026-11-17).

- 2026-11-17, user: "work on vibe-quant-20fx4 ... you can work on nxmxd.4 as well"; after the architect design, "all rec" (Q1 close 20fx4.15 won't-fix, Q2 screening's liquidation formula wins, Q3 no fast-tier fix). See [user-preferences.md](user-preferences.md).
- 20fx4.15 closed won't-fix: MEASURED per recompute SUPERTREND 1.31 / 12.59 / 70.92 / 344.87 ms at 2k / 20,160 / 100k / 525,600 bars vs 900 s per 15m bar (architect, design.md §1). Side finding filed as vibe-quant-3g81l (paper starts recursive indicators cold).
- nxmxd.4 (as of vibe-quant@7e7d8a96): one `liquidated_close` and one `valuation_horizon` (>=) for both tiers. Declared comparability break: exactness_tiers C3-isolated-125 validation trade[4] gross/net/roi + profit_factor at 1-2 ulp; tiers.json golden re-captured 2026-11-17 (old copy data/swarm/_goldens/tiers.json.pre-20261117); exactness_239 + --validation-multi IDENTICAL.
- 20fx4.14 grew in round 2 by chief ruling: a YAML-tab parse error now blocks Save/Validate (pre-existing silent drop found by reviewer-lite). Follow-up 20fx4.23 (leg value checks at parse).
- 20fx4.20 scheduled via T3 schedule_task for 2026-12-02 08:00Z; asks the user before downloading data.
- Review outcomes: GPT-6.1 sol PASS on both Sonnet money/fill tasks (T2, T3), Haiku reviewer-lite SHOULD-FIX on the DeepSeek UI task (T1). See [routing-observations.md](routing-observations.md).
- Land incident: chief committed the wiki to main mid-job, land.py failed the final tree check after committing all tasks; fixed by merging main into the job branch. Self-improve task cc-tvl-land-1 adds a pre-check. See [env-gaps.md](env-gaps.md).
