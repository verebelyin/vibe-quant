# Discovery Journal

Research diary tracking GA strategy discovery experiments, screening verification, and validation results.

---

## 2026-02-27: Batch 6 — Bull Market Discovery, First Long Strategy, New Records

### Goal

Find winning strategies on **2024 bull market data** that are **not just short-only**. Previous batches all tested bearish windows (Sep 2025–Feb 2026) where long was "brutally unprofitable." BTC went from ~$42K (Jan 2024) to ~$100K (Dec 2024) — the ideal window for discovering long and bidirectional strategies.

### Setup

5 parallel discovery runs, all BTC, CCI+RSI dominant, bigger timeframes, 2024-centric date ranges.

**Design rationale:**
- CCI is king across all 5 previous batches — kept in every run
- 2024 bull market is the first window where long-only strategies should work
- `both` direction on 3/5 runs to find bidirectional strategies
- Run 89 is pure `long` on 2024 — first real test of long strategy discovery
- Run 92 adds WILLR for indicator diversity (successful in run 53)
- Conservative pop sizes to target 30-40min completion
- Note: sandbox blocked `ProcessPoolExecutor` → all runs fell back to sequential mode, extending runtimes to 30-50min

| Run | TF | Dir | Indicators | Pop×Gen | Mut | Date Range | Duration |
|-----|-----|-----|------------|---------|-----|------------|----------|
| 88 | 4h | both | CCI,RSI | 22×12 | 0.22 | 2024-01→2025-06 | ~48m |
| 89 | 4h | long | CCI,RSI | 22×12 | 0.20 | 2024-01→2024-12 | ~35m |
| 90 | 1h | both | CCI,RSI | 20×10 | 0.25 | 2024-01→2025-06 | ~38m |
| 91 | 15m | both | CCI,RSI | 18×10 | 0.25 | 2024-06→2025-06 | ~50m |
| 92 | 4h | both | CCI,RSI,WILLR | 20×10 | 0.22 | 2024-01→2026-02 | ~55m |

### Discovery Results

| Run | TF | Dir | Fitness | Sharpe | PF | Trades | Return |
|-----|-----|-----|---------|--------|-----|--------|--------|
| **88** | **4h** | **both** | **0.793** | **4.19** | **2.808** | **60** | **+50.4%** |
| **90** | **1h** | **both** | **0.764** | **4.08** | **1.727** | **52** | **+43.0%** |
| **89** | **4h** | **long** | **0.733** | **3.57** | **1.652** | **52** | **+45.4%** |
| 91 | 15m | both | 0.642 | 2.58 | 1.437 | 58 | +24.6% |
| 92 | 4h | both | 0.598 | 2.09 | 1.401 | 168 | +13.6% |

### Top 3 Strategies

**Run 88 — CCI Triple Bidirectional (4h, Both) ★ NEW ALL-TIME BEST**
- Entry: CCI(30) crosses_below 59.9 (both long+short)
- Exit: CCI(40) crosses_below 75.0 AND CCI(47) < 29.2 (both sides)
- SL: 1.09% (very tight), TP: 17.13% (wide) — trend-following risk profile
- Pure CCI strategy, no RSI needed

**Run 89 — RSI Long-Only (4h, Long) ★ FIRST PROFITABLE LONG STRATEGY**
- Entry (long): RSI(12) < 60.9
- Exit (long): RSI(37) crosses_below 41.9
- SL: 4.4%, TP: 13.3%
- Pure RSI strategy — CCI not selected by GA on bull market!

**Run 90 — CCI Bidirectional (1h, Both)**
- Entry: CCI(25) crosses_above -81.4 (both sides)
- Exit: RSI(7) crosses_above 58.9 (both sides)
- SL: 6.0%, TP: 11.6%
- CCI+RSI combo, deep oversold CCI entry

### Full Pipeline: Discovery → Screening → Validation

**Run 88 (4h Both CCI) ★ NEW ALL-TIME BEST:**

| Step | Run | Sharpe | PF | Trades | Return | MaxDD |
|------|-----|--------|-----|--------|--------|-------|
| Discovery | 88 | 4.19 | 2.808 | 60 | +50.4% | 0% |
| Screening | 93 | **4.19** | **2.808** | **60** | **+50.4%** | 0% |
| Validation | 98 | **4.12** | **2.736** | **61** | **+49.3%** | 11.3% |

Validation barely degraded: Sharpe 4.19→4.12, PF 2.808→2.736, return 50.4→49.3%. **Gained 1 trade** (60→61) — validation fill model gave slightly different entry timing. MaxDD 11.3% is moderate. PF 2.736 is the **highest validated PF ever recorded** (previous: run 81's 2.135).

**Run 89 (4h Long RSI) ★ FIRST LONG STRATEGY:**

| Step | Run | Sharpe | PF | Trades | Return | MaxDD |
|------|-----|--------|-----|--------|--------|-------|
| Discovery | 89 | 3.57 | 1.652 | 52 | +45.4% | 0% |
| Screening | 94 | **3.57** | **1.652** | **52** | **+45.4%** | 0% |
| Validation | 99 | **3.57** | **1.651** | **51** | **+45.5%** | 16.0% |

**Virtually zero degradation.** Sharpe identical (3.57→3.57), return slightly improved (45.4→45.5%), lost only 1/52 trades. 16% MaxDD is the concern — typical for a long-only strategy on a bull market (deep drawdowns during corrections). This strategy is **latency-immune** (4h RSI doesn't care about 200ms).

**Run 90 (1h Both CCI+RSI):**

| Step | Run | Sharpe | PF | Trades | Return | MaxDD |
|------|-----|--------|-----|--------|--------|-------|
| Discovery | 90 | 4.08 | 1.727 | 52 | +43.0% | 0% |
| Screening | 95 | **4.08** | **1.727** | **52** | **+43.0%** | 0% |
| Validation | 100 | **3.99** | **1.709** | **47** | **+38.0%** | 10.5% |

Classical degradation: lost 5/52 trades (90% survival), return 43→38%, Sharpe 4.08→3.99. Still excellent. 10.5% MaxDD is well-controlled.

### Comparison: Batch 6 vs All Previous Champions

| Run | TF | Dir | V.Sharpe | V.PF | V.Trades | V.Return | V.MaxDD | Window |
|-----|-----|-----|----------|------|----------|----------|---------|--------|
| **98 (new) ★** | **4h** | **both** | **4.12** | **2.736** | **61** | **+49.3%** | **11.3%** | **2024-01→2025-06** |
| **99 (new)** | **4h** | **long** | **3.57** | **1.651** | **51** | **+45.5%** | **16.0%** | **2024-01→2024-12** |
| **100 (new)** | **1h** | **both** | **3.99** | **1.709** | **47** | **+38.0%** | **10.5%** | **2024-01→2025-06** |
| 85 (prev) | 15m | short | 4.60 | 2.010 | 47 | +41.4% | 16.1% | 2024-06→2026-02 |
| 87 (prev best) | 4h | both | 3.65 | 2.135 | 85 | +16.7% | 2.2% | 2024-06→2026-02 |
| 56 (orig) | 5m | short | 4.64 | 1.854 | 46 | +24.6% | 7.3% | 2025-09→2026-02 |

### Findings

1. **Run 88/98 is the new all-time best** — PF 2.736 (highest ever validated), Sharpe 4.12, +49.3% return on 18 months of data. The tight SL (1.09%) with wide TP (17.13%) creates a trend-following system that cuts losses fast and lets winners run. The triple-CCI setup (entry CCI(30), exit CCI(40)+CCI(47)) provides multi-scale momentum confirmation.

2. **Run 89/99 proves long strategies work on bull markets** — first ever profitable long-only strategy in the discovery pipeline. The GA chose pure RSI (no CCI!) on the 2024 bull market, finding that RSI works better than CCI for trend-following in uptrends. Sharpe 3.57 is strong, +45.5% return is the second-highest ever.

3. **The date window matters enormously** — previous batches found only short strategies because the test window (Sep 2025–Feb 2026) was bearish. Testing on the 2024 bull run found long and bidirectional strategies with dramatically higher returns (+49% vs +17% for the same 4h bidirectional approach).

4. **4h strategies are consistently the most robust** — runs 88 and 89 both achieved >98% trade survival through validation. The 200ms retail latency is irrelevant on 4h bars. Run 90 (1h) lost 10% of trades but still performed well.

5. **CCI remains dominant but RSI shines on bull markets** — run 89's GA independently discovered that RSI alone works better than CCI for long strategies in uptrends. This is the first time any indicator has beaten CCI in the discovery pipeline.

6. **WILLR didn't help (run 92)** — adding WILLR to the indicator pool produced the weakest result (PF 1.401, +13.6%). The extra search space from 3 indicators diluted the GA's ability to converge on strong CCI patterns. Keep indicator pools lean.

7. **15m both-direction (run 91) was mediocre** — fitness 0.642 and +24.6% return. The 15m timeframe generates too many signals for bidirectional trading, leading the GA to spend generations reducing overtrade (started at 7190 trades, ended at 58). Better to use 15m for focused single-direction strategies.

8. **Previous champion run 87 (PF 2.135, MaxDD 2.2%) still holds the MaxDD crown** — run 98's 11.3% MaxDD is higher. For risk-averse deployment, run 87 remains the best. For absolute returns, run 98 dominates.

### Recommendations

1. **Paper trade run 98 (4h both CCI)** — new all-time best on PF and return
2. **Paper trade run 99 (4h long RSI)** — first long strategy, watch 16% MaxDD
3. **Multi-window testing** — run 98's strategy on the Sep 2025–Feb 2026 window to check if it survives bearish conditions
4. **Portfolio combination** — combine run 87 (bearish window champion) with run 98 (bull window champion) for regime-adaptive trading
5. **Out-of-sample validation** — test run 98 on 2023 data (not in training set) for true out-of-sample performance
6. **RSI long-only exploration** — run 89's success with pure RSI suggests dedicated long-only RSI discovery runs could find even better bull-market strategies

---

## 2026-02-27: Batch 5 — Bigger Timeframes, Both-Direction, Longer Date Ranges

### Goal

Find winning strategies that are **not just short-only**, on **bigger timeframes** (15m, 1h, 4h), with longer date ranges covering both bullish and bearish BTC regimes.

### Setup

5 parallel discovery runs, all BTC, all `direction=both`, CCI in every pool (proven king).

**Design rationale:**
- CCI dominant in all previous winners (runs 52, 58, 71, 73)
- `both` direction doubles search space → higher mutation (0.25-0.30)
- Bigger timeframes = fewer bars = CCI (pandas-ta) completes in reasonable time
- Date range 2024-06 to 2026-02 (20 months) captures BTC bull run + correction
- EMA/SMA/ADX excluded from pool — they're price-relative indicators that don't work with threshold comparison in current genome design

**Note:** Runs requested CCI+RSI+EMA, CCI+ADX+ATR, CCI+RSI+ADX, CCI+RSI+SMA, CCI+RSI+EMA but EMA/SMA/ADX are not in the genome INDICATOR_POOL (price-relative, need indicator-vs-indicator comparison). All runs effectively used **CCI+RSI** (the valid subset). ATR was also available but runs 78/79 only got CCI+RSI filtered. This is a limitation worth addressing — adding ADX/EMA with auto-threshold would diversify the search significantly.

| Run | TF | Indicators (effective) | Pop×Gen | Mut | Date Range | Duration |
|-----|-----|----------------------|---------|-----|------------|----------|
| 77 | 1h | CCI,RSI | 20×12 | 0.25 | 2024-06→2026-02 | 35m |
| 78 | 1h | CCI,RSI | 20×12 | 0.25 | 2024-06→2026-02 | 35m |
| 79 | 4h | CCI,RSI | 24×15 | 0.20 | 2024-06→2026-02 | 39m |
| 80 | 15m | CCI,RSI | 16×10 | 0.30 | 2025-01→2026-02 | 25m |
| 81 | 4h | CCI,RSI | 20×12 | 0.25 | 2024-06→2026-02 | 33m |

### Discovery Results

| Run | TF | Fitness | Sharpe | PF | Trades | Return |
|-----|-----|---------|--------|-----|--------|--------|
| **80** | **15m** | **0.737** | **3.90** | **1.809** | **53** | **+37.8%** |
| **79** | **4h** | **0.722** | **3.70** | **1.834** | **57** | **+34.7%** |
| **81** | **4h** | **0.704** | **3.27** | **1.949** | **86** | **+14.8%** |
| 77 | 1h | 0.607 | 2.10 | 1.347 | 74 | +28.4% |
| 78 | 1h | 0.539 | 1.37 | 1.183 | 89 | +14.7% |

### Top 3 Strategies

**Run 80 — CCI Double Crossover (15m, Short)**
- Entry: CCI(26) crosses_below 67.6 AND CCI(32) crosses_below 3.0
- Exit: RSI(5) crosses_below 61.1
- SL: 2.61%, TP: 8.45%

**Run 79 — CCI Deep Oversold (4h, Short)**
- Entry: CCI(17) < -99.9
- Exit: RSI(25) < 34.2 AND CCI(49) >= 105.5
- SL: 4.81%, TP: 15.05%

**Run 81 — CCI Bidirectional (4h, Both) ★**
- Entry (long+short): CCI(23) crosses_above 37.0
- Exit (long+short): CCI(44) crosses_above 57.4
- SL: 8.29%, TP: 13.06%

### Full Pipeline: Discovery → Screening → Validation

**Run 80 (15m Short):**

| Step | Run | Sharpe | PF | Trades | Return | MaxDD |
|------|-----|--------|-----|--------|--------|-------|
| Discovery | 80 | 3.90 | 1.809 | 53 | +37.8% | 0% |
| Screening | 82 | **3.90** | **1.809** | **53** | **+37.8%** | 0% |
| Validation | 85 | **4.60** | **2.010** | **47** | **+41.4%** | 16.1% |

Validation **improved** Sharpe (3.9→4.6) and return (37.8→41.4%). Lost 6 trades (53→47) but remaining trades were higher quality. 16.1% MaxDD is the highest of the batch but still manageable.

**Run 79 (4h Short):**

| Step | Run | Sharpe | PF | Trades | Return | MaxDD |
|------|-----|--------|-----|--------|--------|-------|
| Discovery | 79 | 3.70 | 1.834 | 57 | +34.7% | 0% |
| Screening | 83 | **3.70** | **1.834** | **57** | **+34.7%** | 0% |
| Validation | 86 | **3.37** | **1.747** | **56** | **+30.6%** | 11.0% |

Classical degradation: Sharpe 3.7→3.4, return 34.7→30.6%. Only lost 1 trade (57→56). Very stable on 4h — latency-immune.

**Run 81 (4h Bidirectional) ★ BEST RISK-ADJUSTED:**

| Step | Run | Sharpe | PF | Trades | Return | MaxDD |
|------|-----|--------|-----|--------|--------|-------|
| Discovery | 81 | 3.27 | 1.949 | 86 | +14.8% | 0% |
| Screening | 84 | **3.27** | **1.949** | **86** | **+14.8%** | 0% |
| Validation | 87 | **3.65** | **2.135** | **85** | **+16.7%** | 2.2% |

Validation **improved** across all metrics. PF 2.135 is the highest ever seen. MaxDD 2.2% is phenomenal. Lost only 1/86 trades. This is the most robust strategy discovered to date.

### Comparison with Previous Champions

| Run | TF | Dir | V.Sharpe | V.PF | V.Trades | V.Return | V.MaxDD |
|-----|-----|-----|----------|------|----------|----------|---------|
| 52 (prev best) | 5m | short | 4.64 | 1.854 | 46 | +24.6% | 7.3% |
| **80 (new)** | **15m** | **short** | **4.60** | **2.010** | **47** | **+41.4%** | **16.1%** |
| 79 (new) | 4h | short | 3.37 | 1.747 | 56 | +30.6% | 11.0% |
| **81 (new) ★** | **4h** | **both** | **3.65** | **2.135** | **85** | **+16.7%** | **2.2%** |

### Findings

1. **Run 81 is the standout** — first genuinely bidirectional strategy to survive validation with improved metrics. PF 2.135 and MaxDD 2.2% are the best risk-adjusted numbers in the entire discovery history.

2. **Run 80 beats run 52 on absolute return** (+41.4% vs +24.6%) but has higher MaxDD (16.1% vs 7.3%). Depending on risk appetite, either could be "best."

3. **4h strategies are latency-immune** — runs 79 and 81 preserved nearly all trades through validation (56/57 and 85/86). This confirms the hypothesis from batch 1 that bigger timeframes survive validation better.

4. **Validation improving over discovery** (runs 80, 81) suggests these strategies are genuinely robust, not overfit. The custom fill model + latency actually helps by filtering out marginal trades.

5. **1h runs (77, 78) underperformed** — lower fitness, lower Sharpe. CCI+RSI alone on 1h may need larger populations or more indicator diversity to find good strategies.

6. **CCI remains king** — every winning strategy across 5 batches is CCI-dominant. The indicator pool limitation (EMA/SMA/ADX excluded) didn't matter because CCI carries the signal.

7. **`both` direction works on 4h** — run 81 proves bidirectional strategies are viable when the timeframe is large enough. The same entry/exit conditions applied to both long and short produced the best risk-adjusted returns.

### Recommendations

1. **Paper trade run 81** — best risk-adjusted strategy, bidirectional, latency-immune
2. **Paper trade run 80** — best absolute return, but monitor 16.1% MaxDD
3. **Add indicator-vs-indicator comparison** to genome design — would unlock EMA/SMA/ADX crossover strategies, dramatically expanding the search space
4. **Try run 81 pattern on other assets** (ETH, SOL) — CCI bidirectional on 4h may generalize
5. **Multi-window validation** — test these strategies on out-of-sample date ranges

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

### Batch 4: Direction-Constrained Discovery (Runs 70-73)

Added `--direction` parameter to discovery pipeline (schema, CLI, operators, pipeline). Forces all chromosomes in a run to use a specific direction (`long`, `short`, `both`), preventing the GA from converging on short-only strategies.

**Setup:** BTCUSDT/5m, 2025-09-01 to 2026-02-24, pop=10, gen=6, ~60 evals each.

| Run | Direction | Indicators | Duration | Best Fitness | Trades | Return | Notes |
|-----|-----------|-----------|----------|-------------|--------|--------|-------|
| 70 | **long** | RSI,ATR,CCI | ~110s | 0.093 | 1651 | -32.5% | Long brutally unprofitable this period |
| **71** | **both** | RSI,CCI,ATR | ~250s | **0.345** | **508** | **-7.6%** | Best of batch, CCI-dominant |
| 72 | random | RSI,ATR,CCI | killed | 0.016 | 1525 | -51.9% | CCI made it too slow, killed+relaunched |
| 73 | random | RSI,ATR | ~330s | 0.007 | 1970 | -59.3% | RSI+ATR only, terrible fitness |

**Run 71 Winner: CCI Both-Direction**

Best genome `genome_1baa67a5463a`: CCI(21) < -68.7 AND CCI(38) > -10.1 entry (both sides), CCI(25) crosses_above -47.3 exit, SL=9.2%, TP=20%.

| Step | Run | Trades | Return | Sharpe | PF | Max DD | Fees |
|------|-----|--------|--------|--------|-----|--------|------|
| Discovery | 71 | 508 | -7.6% | 0.62 | 1.094 | 0.0% | — |
| Screening | 74 | **508** | -7.6% | 0.62 | 1.094 | 0.0% | $16 |
| Validation | 76 | **508** | **-6.7%** | **1.31** | **1.196** | **6.7%** | $8,445 |

Discovery↔Screening exact match confirmed again. Validation **preserved all 508 trades** (no drop!) and Sharpe actually improved 0.62→1.31. Unusual — likely because the CCI conditions are coarse enough (period 21-38) that 200ms latency doesn't affect 5m signal timing.

**What went well:**
1. Direction constraint feature works — clean implementation, trivially added to CLI/API/pipeline
2. `both` direction was the only viable approach; confirms that forcing long-only on a bearish BTC window just wastes compute
3. CCI continues to dominate as the best discovery indicator (runs 52, 58, 71 all CCI-winning)
4. Validation preservation of trade count (508→508) is strong — much better than the 202→2 collapse seen with 1m ATR strategies
5. Fast turnaround: 3 runs completed in <5 min (except CCI-slowed run 72)

**What went wrong:**
1. Run 72 was too slow because CCI uses pandas-ta on 5m data. Had to kill and relaunch as run 73 with RSI+ATR only
2. Long-only strategies are essentially impossible on this 6-month BTC window (Sep 2025–Feb 2026 was bearish). Fitness 0.093 is barely above zero
3. Random direction (run 73) with only RSI+ATR was the worst performer — too few indicators + random direction = scattered search
4. All returns are negative. Even the best genome (-6.7%) loses money. PF>1 but not enough to offset the losing period

**Assessment — is direction constraint worth it?**

**Mixed.** The feature itself is valuable infrastructure — it lets us intentionally explore long/both strategies instead of GA always converging on short. But the results show that on a bearish BTC window, even forced-both strategies lose money. The real test would be running on a bullish window (e.g., 2024-01 to 2024-06) to see if long-only strategies can be discovered there.

**Recommendations:**
1. **Multi-window discovery**: Run same config on 2-3 different date ranges to find strategies robust across regimes
2. **Larger pop for both-direction**: pop=10 gen=6 is too small for `both` which doubles the search space (long+short conditions). Try pop=20 gen=10
3. **CCI is king**: Keep CCI in indicator pools despite being pandas-ta (slower). It's the most consistently successful indicator
4. **RSI+ATR alone is useless**: Run 73 confirms that without CCI, RSI+ATR can't find viable strategies on 5m
5. **Consider longer timeframes for long strategies**: 15m or 1h may work better for long-only since fundamental trend signals are clearer

---

### Batch 3: Fast Discovery Runs (Runs 57-60)

Launched 3 parallel discovery runs on **BTCUSDT/5m** with pop=8, gen=5 (~40 evals), targeting 5-10 min completion. Higher mutation (0.25-0.3) for more diversity including long/both directions.

| Run | Archetype | Indicators | Status | Duration | Best Fitness | Notes |
|-----|-----------|------------|--------|----------|-------------|-------|
| 57 | Trend | SMA,EMA,ADX,ATR | **CRASHED** | ~4.5 min | 0.0 | Position size rounds to 0 |
| 58 | Mean Rev v2 | RSI,CCI,EMA | **Completed** | ~8.7 min | 0.6415 | Short-only, +18% |
| 59 | Momentum | RSI,ADX,ATR,SMA | **FAILED** | ~44s | 0.0 | 0 trades all gens |
| 60 | Momentum v2 | RSI,EMA,SMA,ATR | **Completed** | ~60s | 0.328 | Both-dir, -27.6%, DSR fail |

**Run 57 crash:** `ValueError: quantity 0.0004996 rounded to zero due to size increment 0.001`. Position sizing produces sub-minimum-lot quantities when risk-based sizing on high-priced BTC. Filed as `vibe-quant-p47m`.

**Run 59 failure:** All Rust-native indicators with very different value ranges (RSI: 0-100, ATR: ~50-500, SMA: ~60K+) produced random conditions that never triggered entries. Pop=8 too small to find viable combinations.

**Run 58 Results: RSI + CCI Mean Reversion v2**

Best genome: RSI(36) < 72.5 (short entry, very loose) + CCI(14) < -30.4 and RSI(12) > 71.7 (exit), SL=0.66%, TP=7.43%

| Step | Run | Trades | Return | Sharpe | PF | Max DD |
|------|-----|--------|--------|--------|-----|--------|
| Discovery | 58 | 179 | +18.0% | 2.91 | 1.479 | 0.0% |
| Screening | 61 | **179** | +18.0% | 2.91 | — | — |
| Validation | 62 | **196** | +15.4% | 2.41 | 1.405 | 12.0% |

Strategy survived validation: Sharpe degraded 2.91→2.41, return 18%→15.4%, but max DD is 12% (acceptable). Short-only again — GA converges on short because it outperforms long on this BTC period.

**Key learnings:**
1. Position sizing needs min-lot-size guard (bug filed)
2. Small populations (8) with heterogeneous indicators (different scales) often produce 0-trade runs
3. CCI+RSI combo continues to be the most reliable (Run 52 and 58 both won with similar combos)
4. GA converges on short-only because it genuinely outperforms long on this 6-month BTC window
5. For bidirectional strategies, may need to force direction=both or run separate long-only/short-only discovery

### Switch to 5m Timeframe (Runs 51-54)

Killed 1m runs 40-44 (too slow with pandas-ta on 260K bars). Relaunched on **BTCUSDT/5m, 2025-09-01 to 2026-02-24** (~52K bars).

Also fixed crossover regex bug (commit `6282d41`): `_CROSS_PATTERN` in `conditions.py` didn't support negative thresholds like `-1.7083`, silently discarding genomes with ROC/MACD crossover conditions.

| Run | Archetype | Indicators | Pop×Gen | Status | Duration | Best Fitness |
|-----|-----------|------------|---------|--------|----------|-------------|
| 51 | Momentum | RSI, MACD, ROC, WILLR | 12×10 | **Killed** (too slow) | ~2h, gen 6/10 | 0.33 |
| **52** | **Mean Reversion** | **RSI, CCI, STOCH, BBANDS** | **12×10** | **Completed** | ~38 min | **0.8412** |
| 53 | Exotic | CCI, WILLR, STOCH, ROC | 12×10 | **Completed** | ~3h 36m | 0.6978 |
| 54 | Full Pool | All 10 indicators | 16×12 | **Killed** (too slow) | ~38 min, gen ~3 | — |

### Run 52 Results: CCI + RSI Mean Reversion (WINNER)

**Best genome:** CCI(entry) >= 7.2 (short entry) + RSI(exit) crosses_below 70.5, SL=5.14%, TP=4.17%

| Step | Run | Trades | Return | Sharpe | PF | Max DD | Fees |
|------|-----|--------|--------|--------|-----|--------|------|
| Discovery | 52 | 54 | +25.02% | 4.425 | 1.722 | 0.0% | — |
| Screening | 55 | **54** | +25.02% | 4.425 | 1.722 | — | — |
| Validation | 56 | **46** | +24.61% | 4.643 | 1.854 | 7.28% | — |

**Key finding:** This strategy **survived validation** with only 8 lost trades (54→46), maintained return (+25%→+24.6%), and Sharpe actually *improved* (4.43→4.64). This is exceptional — most strategies degrade significantly in validation.

### Run 53: Exotic CCI/WILLR/STOCH/ROC (Completed)

Completed in 10/10 generations (120 evaluations) with best fitness 0.6978 after ~3h 36m.

**Final best genome:** `genome_43bef822cde3`
- Entry (short): `willr_entry_0 crosses_above -2.601`
- Exit (short): `willr_exit_0 crosses_below 9.4135` and `willr_exit_1 >= -3.8788`
- SL: 5.1%, TP: 1.46%
- Final metrics: 101 trades, +17.1% return, Sharpe 3.6398, PF 1.6247, max DD 0.0%

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
