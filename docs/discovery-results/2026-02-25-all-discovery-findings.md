# Discovery Run Findings — 2026-02-24/25

All strategies found from GA runs on BTCUSDT. Runs 3/4/6 killed before completion; run 2 completed fully.

## Top Strategies

### #1: Genome136e01a48d3f (Run 4) — +44.9%
- **See**: [detailed report](2026-02-25-best-strategy-run4.md)
- BTCUSDT 4h, Jan 2024 → Dec 2025
- DonchianChannel(43) + ROC(17) + WMA(79) + RSI(39)
- Sharpe 1.97, Sortino 4.93, PF 1.40, WR 17.5%, R:R 5.9:1
- 137 trades, +328 USDT/trade expectancy

### #2: Genome6b350a772061 (Run 6) — +22.9%
- BTCUSDT 1h, Jan 2025 → Feb 2026
- BollingerBands(30, 1.83) + ATR(27)
- Sharpe 2.28, Sortino 4.49, PF 1.34, WR 35.8%, R:R 2.3:1
- ~67 trades, +342 USDT/trade expectancy
- Higher win rate, lower reward:risk vs #1

### #3: Run 6 #2 — +12.7%
- BTCUSDT 1h, Jan 2025 → Feb 2026
- Sharpe 1.53, Sortino 3.29, PF 1.24, WR 27.6%, R:R 3.1:1
- +167 USDT/trade expectancy

### #4: Genome4f01e7676340 (Run 3) — +12.1%
- BTCUSDT 1h, Jun 2024 → Dec 2025
- RSI(43) + SMA(128) + WMA(25)
- Sharpe 1.05, Sortino 2.18, PF 1.18, WR 18.7%, R:R 4.8:1
- +113 USDT/trade expectancy

### #5: Run 2 (completed, 10 gens) — -46.8% (FAILURE)
- BTCUSDT 1h, pop=30, 10 generations fully converged (fitness 0.5846)
- 1,872 positions, -54K in commissions on 100K account
- **Lesson**: fitness function rewards Sharpe on daily returns but doesn't penalize overtrading/commissions enough. GA converged to high-frequency losers.

## Key Observations

1. **Best strategies found in gen 0** — initial random population, no evolution needed
2. **60-78% of genomes produce 0 trades** — conditions too restrictive, wasted compute
3. **Fitness ≠ profitability** — Run 2 fitness 0.5846 = -46.8%, Run 4 fitness 0.5922 = +44.9%
4. **All profitable strategies are trend-following** with low win rate + high reward:risk
5. **4h timeframe outperformed 1h** — less noise, bigger moves, fewer commissions
6. **Commission awareness critical** — Run 2 shows GA exploiting Sharpe without accounting for fees

## Action Items

- [ ] Validate #1 and #2 out-of-sample (2026+ data)
- [ ] Fix fitness function to penalize high trade count / commission drag
- [ ] Reduce wasted evals (0-trade genomes) — smarter initialization or constraint
- [ ] Run walk-forward analysis on top strategies
- [ ] Consider 4h as default timeframe for discovery (better signal-to-noise)
