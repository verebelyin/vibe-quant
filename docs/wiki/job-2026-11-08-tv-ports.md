# Job 20261108-tv-ports (2026-11-08 .. 11-11)

Source: data/swarm/20261108-tv-ports/ (brief.md, design.md, design-v2.md, ledger.md, handoffs/, retro.md). Epic vibe-quant-20fx4.

- Goal (user 2026-11-08): port TradingView Pine strategies as pluggable "ports" (scripts/research/tv_ports/) runnable as a backtest, a burst cell and a GA over their own inputs. 16 ports landed (1c2ed6b3, 1bc2e466) behind opt-in features; exactness_239 (+multi), carry and fast-tier parity never drifted.
- Results (journal): 2026 search -> unseen test, 23/47 pass the lenient bar, 0/53 beat buy-and-hold (15b9630c); 20 hand-picked presets, 16/47 strict picks vs ~11 expected by chance (953cf734). Single-coin keepers stay in gitignored data/research/ (user 2026-11-10: not public).
- Job retro (retro.md, 2026-11-08): 16 real blocking review findings; 11 of 34 gate runs failed, 8 not code defects; a network outage killed 3 children. User OK'd one same-provider retry on network errors (257341f6).
- Closed 2026-11-18 by the user ("close epic, i dont want montly rechecks"); verdict no edge. See [user-preferences.md](user-preferences.md).
