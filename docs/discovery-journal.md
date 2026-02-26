# Discovery Journal

Research diary tracking GA strategy discovery experiments, screening verification, and validation results.

---

## 2026-02-26: Post-Bug-Fix Verification + 1m Strategy Search

### Context

After fixing the `pos.entry→pos.side` cross-enum comparison bug (commit `2944ad3`), discovery→screening trade counts now match exactly. This session verified the fix and launched new 1m strategy discovery runs.

### Bug Fix Verification (Runs 33-39)

Ran two complete pipelines to confirm discovery↔screening consistency.

**Pipeline A: BTCUSDT/1h (2025-06-01 to 2025-12-31)**
- Discovery config: pop=4, gen=2, 8 evaluations
- Best genome: CCI(45) < 60.66 + WILLR(23) >= -32.56 → RSI(8) <= 51.17 exit, SL=7.2%, TP=19.45%

| Step | Run | Trades | Return | Sharpe | PF | Fees |
|------|-----|--------|--------|--------|-----|------|
| Discovery | 33 | 283 | -5.52% | 0.5275 | 1.066 | — |
| Screening | 36 | **283** | -5.52% | 0.5273 | 1.066 | $21 |
| Validation | 37 | **283** | -5.69% | 0.4009 | 1.048 | $6,117 |

**Pipeline B: BTCUSDT/4h (2024-06-01 to 2025-12-31)**
- Discovery config: pop=6, gen=3, 18 evaluations
- Best genome: MACD(20,29,8) <= -0.0019 → ROC(16) crosses_below 38.43 exit, SL=2.73%, TP=15.76%

| Step | Run | Trades | Return | Sharpe | PF | Fees |
|------|-----|--------|--------|--------|-----|------|
| Discovery | 35 | 53 | -8.15% | -0.419 | 0.937 | — |
| Screening | 38 | **53** | -8.15% | -0.419 | 0.937 | $35 |
| Validation | 39 | **53** | -11.47% | -0.816 | 0.878 | $1,958 |

**Findings:**
1. Discovery ↔ Screening: **exact match** on all metrics (trade count, return, Sharpe, PF)
2. Validation degrades gracefully: same trade count but worse Sharpe/return due to custom fill model (VolumeSlippageFillModel with sqrt market impact), 200ms retail latency, and realistic fee modeling
3. Validation fees are ~100-300x higher than screening fees — screening uses simplified probabilistic fill; validation models real market impact
4. Max drawdown only appears in validation (NT 1.222 limitation: `MaxDrawdown` stat not registered in screening mode)

### 1m Strategy Discovery (Runs 40-44)

Launched 5 parallel discovery runs on **BTCUSDT/1m, 2025-09-01 to 2026-02-24** (~260K bars) to find high-frequency strategies.

| Run | Archetype | Indicators | Pop×Gen | Mutation | Status |
|-----|-----------|------------|---------|----------|--------|
| 40 | Momentum | RSI, MACD, ROC, WILLR | 12×10 | 20% | Running (very slow) |
| 41 | Mean Reversion | RSI, CCI, STOCH, BBANDS | 12×10 | 20% | Running (very slow) |
| **42** | **Trend** | **SMA, EMA, ADX, ATR** | **12×10** | **20%** | **Completed** |
| 43 | Exotic | CCI, WILLR, STOCH, ROC | 12×10 | 25% | Running (very slow) |
| 44 | Full Pool | All 10 indicators | 16×12 | 30% | Running (very slow) |

**Timing insight:** Run 42 (Trend) completed in ~15 min because SMA/EMA/ADX/ATR are Rust-native NT indicators. Runs using MACD/STOCH/BBANDS/CCI fall back to pandas-ta (Python) which is orders of magnitude slower on 260K 1m bars. Each evaluation takes 3-5+ minutes vs ~30s for Rust-native indicators.

### Run 42 Results: ATR Volatility Strategy

**Best genome:** `genome_b8e894ad202a` — pure ATR(22) strategy
- Entry (short): `atr > 0.0026` (enter short when volatility exceeds threshold)
- Exit (short): `atr crosses_above 0.0092` (exit when volatility spikes further)
- SL: 3.25%, TP: 1.64%
- All top-5 genomes converged to similar ATR patterns (small indicator pool = fast convergence)

| Step | Run | Trades | Return | Sharpe | PF | Max DD | Fees |
|------|-----|--------|--------|--------|-----|--------|------|
| Discovery | 42 | 202 | +22.38% | 2.523 | 1.308 | 0.0% | — |
| Screening | 45 | **202** | +22.38% | 2.523 | 1.308 | 0.0% | $49 |
| Validation | 46 | **2** | -1.65% | -1.325 | 0.789 | 2.54% | $149 |

**Critical finding:** Validation collapsed from 202→2 trades. Possible causes:
1. **Latency model kills 1m strategies**: 200ms retail latency on 1-minute bars means orders arrive ~0.33 bars late. The ATR-based entry (`atr > 0.0026`) fires frequently but with latency, the fill price has moved significantly.
2. **VolumeSlippageFillModel**: The custom fill model with sqrt market-impact slippage may reject or significantly worsen fills on rapid 1m entries.
3. **This is actually a useful signal**: A strategy that only works with zero-latency fills is not executable in real trading. The validation model correctly identified this.

**Implication for 1m strategies:** Need strategies robust to 200ms latency. Consider:
- Using `latency_preset="co_located"` (1ms) instead of "retail" (200ms) if deploying on co-located infrastructure
- Filtering for strategies that maintain trade count in validation
- Using higher timeframes (5m, 15m) where latency is proportionally smaller

### Observations on GA Behavior

1. **Population diversity matters**: pop=4 gen=2 (runs 33-34) barely explored the space. Run 34 failed with "no strategies" because no genome passed the 50-trade hard gate on 4h data.
2. **Indicator pool affects convergence speed**: Rust-native indicators (SMA, EMA, ADX, ATR) evaluate ~10x faster than pandas-ta fallbacks (MACD signal/histogram, STOCH, BBANDS).
3. **1m data is expensive**: ~260K bars per eval. Budget ~3-5 min per evaluation for complex indicators. A full run (pop=12, gen=10 = ~120 evals) with mixed indicators could take 6+ hours.
4. **Convergence is fast with small pools**: Run 42 found ATR as dominant pattern within 3-4 generations, then spent remaining gens refining thresholds. Top-5 genomes were nearly identical.

### Known Issues

- **Max drawdown always 0 in screening**: NT 1.222's `MaxDrawdown` statistic lacks `calculate_from_realized_pnls`. Only validation reconstructs it from trades.
- **Validation runs 37, 39 show "Job stale" error**: The heartbeat timeout (120s) is too short for long backtests. Jobs completed successfully but were marked stale by the job manager before results were written. Results are still present in DB.
- **Direction field ambiguity**: Run 42's genomes show `direction=long` but conditions are in `entry_short`/`exit_short`. Need to investigate if this is a display bug or a genome→DSL translation issue.

### Switch to 5m Timeframe (Runs 51-54)

Killed 1m runs 40-44 (too slow with pandas-ta on 260K bars). Relaunched on **BTCUSDT/5m, 2025-09-01 to 2026-02-24** (~52K bars).

Also fixed crossover regex bug (commit `6282d41`): `_CROSS_PATTERN` in `conditions.py` didn't support negative thresholds like `-1.7083`, silently discarding genomes with ROC/MACD crossover conditions.

| Run | Archetype | Indicators | Pop×Gen | Status | Duration | Best Fitness |
|-----|-----------|------------|---------|--------|----------|-------------|
| 51 | Momentum | RSI, MACD, ROC, WILLR | 12×10 | **Killed** (too slow) | ~2h, gen 6/10 | 0.33 |
| **52** | **Mean Reversion** | **RSI, CCI, STOCH, BBANDS** | **12×10** | **Completed** | ~38 min | **0.8412** |
| 53 | Exotic | CCI, WILLR, STOCH, ROC | 16×10 | **Died at gen 9/10** | ~3h | 0.6978 |
| 54 | Full Pool | All 10 indicators | 16×12 | **Killed** (too slow) | ~38 min, gen ~3 | — |

### Run 52 Results: CCI + RSI Mean Reversion (WINNER)

**Best genome:** CCI(entry) >= 7.2 (short entry) + RSI(exit) crosses_below 70.5, SL=5.14%, TP=4.17%

| Step | Run | Trades | Return | Sharpe | PF | Max DD | Fees |
|------|-----|--------|--------|--------|-----|--------|------|
| Discovery | 52 | 54 | +25.02% | 4.425 | 1.722 | 0.0% | — |
| Screening | 55 | **54** | +25.02% | 4.425 | 1.722 | — | — |
| Validation | 56 | **46** | +24.61% | 4.643 | 1.854 | 7.28% | — |

**Key finding:** This strategy **survived validation** with only 8 lost trades (54→46), maintained return (+25%→+24.6%), and Sharpe actually *improved* (4.43→4.64). This is exceptional — most strategies degrade significantly in validation.

### Run 53: Exotic CCI/WILLR/STOCH/ROC (Unfinished)

Ran 3+ hours, reached gen 9/10 with fitness 0.6978 (101 trades, +17.1% return) but process died or timed out before gen 10 completed. API returned empty strategies. Best genome at gen 9 had mean fitness 0.6749, showing good convergence.

**Lesson:** STOCH (pandas-ta) is extremely slow on 5m data. Gen 9 alone took 3079s (~51 min). Mixed pandas-ta indicator pools on 5m should use smaller populations or fewer generations.

---

## Historical Discovery Runs (Pre-Bug-Fix)

These runs from earlier sessions produced inflated discovery metrics due to the `pos.entry→pos.side` bug. The bug caused SL/TP orders to submit with wrong side, preventing exits and reducing screening replay trades to 1.

### Affected Runs (2026-02-25)

| Discovery Run | Strategy | Discovery Trades | Screening Trades | Note |
|---------------|----------|-----------------|-----------------|------|
| 16 | ga_willr_roc_macd_short | 155 | 1 | Bug: wrong SL/TP side |
| 15 | ga_macd_willr_bidir | 55 | 1 | Bug + 0.5% SL → -17,071% loss |
| 14 | ga_cci_rsi_long | 235 | 1 | Bug: wrong SL/TP side |

**Root cause:** `pos.entry` returns `OrderSide` (BUY=1, SELL=2) but was compared against `PositionSide` (LONG=2, SHORT=3). `OrderSide.SELL` (2) accidentally matched `PositionSide.LONG` (2) numerically, causing all SL/TP to submit with wrong side.

**Fix:** Changed `pos.entry` → `pos.side` in `templates.py` (commit `2944ad3`).

**These discovery results should be discarded.** Re-run with current compiler version `63ca6bfb8b9e` for valid results.

---

## Appendix: Configuration Reference

### Discovery GA Parameters

| Parameter | Description | Recommended Range |
|-----------|-------------|-------------------|
| population | Genomes per generation | 10-30 (higher = more exploration) |
| generations | Evolution rounds | 8-20 (higher = more refinement) |
| mutation_rate | Chance of random gene change | 0.1-0.3 (higher = more diversity) |
| crossover_rate | Chance of parent gene swap | 0.7-0.9 |
| elite_count | Best genomes carried forward unchanged | 1-3 |
| tournament_size | Selection pressure (2=low, 4=high) | 2-4 |
| convergence_generations | Stop if no improvement for N gens | 5-10 |

### Fitness Function

```
Score = 0.35×Sharpe + 0.25×(1-MaxDD) + 0.20×PF + 0.20×Return
```
- Hard gate: score=0 if trades < 50
- Complexity penalty: -0.02 per gene beyond 2 (capped at -0.10)
- Overtrade penalty: -0.05 per 100 trades beyond 300

### Screening vs Validation Differences

| Aspect | Screening | Validation |
|--------|-----------|------------|
| Fill model | NT built-in FillModel | Custom VolumeSlippageFillModel |
| Slippage | 50% chance of 1-tick | Sqrt market-impact formula |
| Latency | None (instant) | 200ms retail (configurable) |
| Fees | ~$20-50 per run | ~$150-6,000 per run |
| Max drawdown | Always 0 (stat not registered) | Computed from trades |
| Purpose | Fast screening of many strategies | Realistic execution simulation |
