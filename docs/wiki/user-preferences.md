# User preferences (dated)

- 2026-10-22 (bead vibe-quant-qlx46, verbatim): "test outcomes at a module's edges, one test per real risk. Don't pin internal call order or config field order. Planted-bug tests stay mandatory only for money, metrics, fills and gates".
- 2026-10-28: test-speed goal ~120 s unit suite, swarm-check ~60 s. See [test-suite-speed.md](test-suite-speed.md).
- 2026-11-05: quicker test runs are top priority ("prio"). Accepted all chief recommendations in [decision-2026-11-05-prune2-scope.md](decision-2026-11-05-prune2-scope.md).
- 2026-11-17: "work on vibe-quant-20fx4 ... you can work on nxmxd.4 as well". nxmxd.4 had been deferred 2026-10-27 with the user's OK (liquidation is opt-in); this lifts that deferral. Job: data/swarm/20261117-tvp-liq. The other deferred beads (2gyk1.3, hpb8u, 68m53, fnzg6) stay deferred per the user's 2026-10-17 rule "wait until the first champion passes validation + paper"; x0axf waits on the H-11F4 forward test (2027-04-13).
- 2026-11-17: for paper vs validation parity (bead 3g81l), the user chose "equal within 1e-9 + identical signals" over bit-exact (job 20261117-paper-warmup brief.md, Q3=A), on condition that the warm-up is not a performance issue (start-up <= 10 s per symbol x timeframe, per-bar cost within 10 ms of cold). Bit-exact stays the bar for backtest tiers (exactness_239).
- 2026-11-17: user approved a monthly wiki lint (T3 schedule "vq monthly wiki lint", first run 2026-12-17).
- 2026-11-18: user closed the TradingView-ports epic vibe-quant-20fx4 and declined monthly forward re-checks ("i dont want montly rechecks"): 20fx4.20 closed won't-do, the 2026-12-02 scheduled re-score deleted. Research verdict at close: no edge (journal 2026-11-08..11-11).
