# Best Discovery Strategy — Run 4, Genome136e01a48d3f

**Found**: 2026-02-24, Discovery Run 4 (GA gen 0, first random population)
**Status**: Unvalidated — needs out-of-sample testing and overfitting filters

## Performance (Jan 2024 → Dec 2025, BTCUSDT 4h)

| Metric | Value |
|--------|-------|
| **Total Return** | **+44.94%** (100K → 144,935 USDT) |
| Sharpe Ratio | 1.97 |
| Sortino Ratio | 4.93 |
| Profit Factor | 1.40 |
| Win Rate | 17.5% |
| Avg Winner | 9,147 USDT |
| Avg Loser | -1,545 USDT |
| Reward:Risk | 5.9:1 |
| Expectancy | +328 USDT/trade |
| Total Positions | ~137 |

## Strategy Configuration

- **Symbol**: BTCUSDT perpetual futures
- **Timeframe**: 4 hours
- **Direction**: Long-only trend following

### Indicators
| Indicator | Period |
|-----------|--------|
| DonchianChannel | 43 |
| RateOfChange | 17 |
| WeightedMovingAverage | 79 |
| RelativeStrengthIndex (EMA) | 39 |

### Risk Parameters (from first trade)
- **Stop Loss**: ~2.2% (StopMarket order)
- **Take Profit**: ~14.9% (Limit order)
- **Position sizing**: ~50-70% of equity per trade (variable)

## GA Discovery Parameters

| Parameter | Value |
|-----------|-------|
| Population | 60 |
| Max Generations | 80 |
| Mutation Rate | 0.12 |
| Crossover Rate | 0.80 |
| Elite Count | 2 |
| Tournament Size | 3 |
| Convergence Gens | 15 |

## Genome UID
`136e01a48d3f` — appeared 4 times in evaluations with consistent ~44.9% return

## Notes

- Found in the **initial random population** (gen 0), before any evolution
- Low win rate (17.5%) offset by exceptional reward:risk ratio (5.9:1)
- Uses Donchian Channel breakout with ROC momentum confirmation
- RSI(39) likely filters overbought entries, WMA(79) for trend direction
- Needs validation against:
  - Out-of-sample period (2026+)
  - Walk-forward analysis
  - Monte Carlo simulation
  - Different market regimes (bear market, sideways)
- Commission drag is moderate at 137 trades over 2 years
