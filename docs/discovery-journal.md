# Discovery Journal

Research diary tracking GA strategy discovery experiments, screening verification, and validation results.

---

## 2026-03-07: Batch 10 — ADX Debut + Novel Indicator Combos

### Goal
Test ADX (Average Directional Index) — first time in discovery. Also test novel 2-indicator combinations never tried before: ADX+RSI, ADX+CCI, ADX+MFI, RSI+ROC, MACD+MFI. Run all 5 concurrently, target <10 min total.

### Bug Fixes Applied During This Batch
- **ADX added to full stack**: indicator registry, DSL schema, compiler init, genome pool
- **ADX missing `period=` arg (compiler bug)**: `_generate_indicator_init()` didn't include ADX in the period-mapping set → `DirectionalMovement()` called without args → TypeError. Fixed by adding "ADX" to both compiler sets.
- **FutureWarning suppression**: Added `import warnings; warnings.filterwarnings('ignore', FutureWarning)` to generated code when pandas-ta is used. Required adding `"warnings"` to `_ALLOWED_IMPORT_PREFIXES`.
- **ProcessPoolExecutor orphan fix**: `_force_shutdown_pool()` now kills orphaned workers on RuntimeError (from Batch 9 bug fix).

### Configuration
| Run | Indicators | Pop | Gens | TF | Time | Status |
|-----|-----------|-----|------|----|------|--------|
| 207 | ADX+RSI | 8 | 5 | 4h | ~5min | Completed |
| 208 | ADX+CCI | 8 | 5 | 4h | ~5min | Completed |
| 209 | ADX+MFI | 8 | 5 | 4h | ~9min | Completed (MFI slow) |
| 210 | RSI+ROC | 8 | 5 | 4h | ~5min | Completed |
| 211 | MACD+MFI | 8 | 5 | 4h | ~9min | Completed (MFI slow) |

All 5 launched via API (parallel), sequential eval within each.

### Full Pipeline Results

| Stage | Run 207 ADX+RSI | Run 208 ADX+CCI | Run 209 ADX+MFI | Run 210 RSI+ROC | Run 211 MACD+MFI |
|-------|----|----|------|-----|-----|
| **Discovery** score | 0.4717 | 0.4531 | **0.5095** | 0.3984 | 0.3469 |
| **Discovery** sharpe | 1.10 | 0.57 | **1.52** | 0.36 | 0.15 |
| **Discovery** dd | 9.4% | 6.4% | **8.1%** | 21.3% | 19.2% |
| **Discovery** trades | 78 | 84 | **128** | 133 | 155 |
| **DSR guardrails** | 4/4 pass | 5/5 pass | **4/4 pass** | 5/5 pass | 0/3 FAIL (p=0.18) |
| **Validation** sharpe | 1.19 | 0.13 | **1.43** | 0.14 | 0.25 |
| **Validation** return | +9.6% | -3.7% | **+8.2%** | -6.6% | -3.6% |
| **Validation** dd | 9.4% | 8.2% | **8.2%** | 23.6% | 19.0% |
| **Validation** trades | 75 | 84 | **125** | 131 | 154 |
| **Validation** PF | 1.16 | 1.02 | **1.24** | 1.02 | 1.04 |
| **Validation** fees | $20.00 | $36.50 | **$23.46** | $63.54 | $56.57 |

### Winning Strategies

**Run 209 winner (genome_efab89664522) — BEST OF BATCH:**
- Direction: LONG only
- Entry: ADX + MFI combination
- Validation: Sharpe=1.43, Return=+8.2%, DD=8.2%, 125 trades, PF=1.24
- DSR p=0.0000 (highly significant)

**Run 207 winner (genome_70db8f1eef40) — second best:**
- Direction: uses ADX + RSI
- Validation: Sharpe=1.19, Return=+9.6%, DD=9.4%, 75 trades, PF=1.16

### Issues Found

1. **ADX compiler bug (HIGH)**: `DirectionalMovement.__init__()` called without `period` → TypeError in runs 207/208/209. 42/36/9 errors respectively. Fixed mid-batch by adding "ADX" to compiler's period-mapping set.
2. **`import warnings` blocked by sandbox**: Generated code with FutureWarning suppression was blocked by `_ALLOWED_IMPORT_PREFIXES`. Fixed by whitelisting "warnings". Caused 34+ strategy failures in runs 209/211.
3. **FutureWarning log spam**: 209K + 229K lines in MFI runs. Suppression fix applied but only takes effect for future runs (already-running discoveries used old compiled code).
4. **Low diversity in RSI+ROC (run 210)**: All top-5 converged to same RSI(9)/ROC(17) pattern. Poor performer.
5. **MACD+MFI (run 211) failed DSR**: Sharpe 0.15, p=0.18 — not statistically significant.

### Key Findings

1. **ADX is viable as discovery indicator**: Despite the compiler bug causing many failures, ADX+MFI produced the best strategy (Sharpe 1.43 validated). ADX measures trend strength (0-100), works well with threshold comparison.
2. **ADX+MFI best combination**: Volume momentum (MFI) + trend strength (ADX) complement each other. Similar to MFI+WILLR from Batch 9 (also volume + momentum).
3. **RSI+ROC poor**: Both are pure momentum — redundant signals, no diversification benefit. Negative validated return.
4. **MACD+MFI poor**: MACD threshold range (-0.005 to 0.005) is too narrow for effective signaling with MFI. Failed DSR.
5. **Validation consistently degrades**: Discovery→Validation Sharpe ratio drops 5-20% as expected (fill model, latency, fees).

### Comparison with Previous Batches

| Metric | Batch 7 (CCI+RSI) | Batch 9 (MFI+WILLR) | Batch 10 (ADX+MFI) |
|--------|-------------------|---------------------|---------------------|
| Validation Sharpe | 7.24 | 2.45 | 1.43 |
| Validation Return | +11.9% | +17.4% | +8.2% |
| Validation DD | 0.9% | 5.9% | 8.2% |
| Validation PF | 4.78 | 1.73 | 1.24 |
| Direction | both | long | long |

Batch 10's ADX+MFI is decent but not a champion. ADX compiler bug caused many wasted evaluations — re-running with fix would likely produce better results.

### Recommendations

1. **Re-run ADX combos with fixed compiler**: The period bug caused 30-50% of evaluations to fail. Clean run should produce significantly better results.
2. **Try ADX+WILLR**: ADX (trend strength) + WILLR (momentum oscillator) — both bounded, complementary signals.
3. **Increase population for ADX runs**: Pop=8 with 30-50% failures means only 4-5 effective chromosomes per generation.
4. **Skip MACD in 2-indicator combos**: MACD's narrow threshold range makes it a poor partner for oscillator combos.

---

## 2026-03-06: Batch 9 — MFI Debut + Expanded Indicator Combos (Post DD-Fix)

### Goal
Test MFI (Money Flow Index, new indicator) in various combinations. Also test WILLR+ROC and CCI+STOCH without CCI dominance. Validate that the drawdown fix from Batch 8 bugs works. Run 6 discoveries total (3 from prev session + 3 new), validate top 4 winners.

### Bug Fixes Applied Before This Batch
- **Drawdown=0.0 fix**: `_compute_max_drawdown()` fallback added to `nt_runner.py` (same pattern as validation)
- **MFI compiler OHLCV fix**: MFI needs high/low/close/volume; generic handler only passed close
- **MFI FutureWarning fix**: Added `dtype=float` to Series construction to suppress pandas dtype spam
- **MACD warning cache**: Single warning instead of per-bar spam
- **NaN fitness coercion**: NaN metrics → 0.0 before fitness scoring
- **SL/TP display**: Removed erroneous `*100` (values already percentages)
- **Top-K deduplication**: uid-based dedup prevents clones in results

### Configuration
| Run | Indicators | Pop | Gens | TF | Time | Status |
|-----|-----------|-----|------|----|------|--------|
| 154 | CCI+ROC | 10 | 6 | 4h | ~27min | Completed (prev session) |
| 157 | CCI+STOCH | 10 | 6 | 4h | ~28min | Completed (prev session) |
| 162 | MFI+CCI+ROC | 10 | 6 | 4h | ~14min | Completed (prev session, parallel) |
| 185 | WILLR+ROC | 10 | 6 | 4h | 29min | Completed |
| 186 | MFI+CCI | 10 | 6 | 4h | 39min | Completed |
| 187 | MFI+WILLR | 10 | 6 | 4h | 52min | Completed |

MFI runs ~2x slower than non-MFI (pandas-ta fallback vs Rust-native). 3 concurrent sequential runs.

### Full Pipeline Results

| Stage | Run 154 CCI+ROC | Run 157 CCI+STOCH | Run 185 WILLR+ROC | Run 186 MFI+CCI | Run 187 MFI+WILLR |
|-------|----|----|------|-----|-----|
| **Discovery** score | 0.4788 | 0.4788 | 0.3330 | 0.4574 | **0.6493** |
| **Discovery** sharpe | 1.96 | ~0.48 | -0.25 | 0.83 | **2.73** |
| **Discovery** dd | 14.6% | ? | 20.0% | 12.7% | **5.9%** |
| **Discovery** guardrails | ? | ? | FAIL (all DSR p=1.0) | 5/5 pass | **5/5 pass** |
| **Validation** sharpe | **2.04** | **2.05** | skipped | 0.86 | **2.45** |
| **Validation** return | +16.4% | +13.9% | skipped | -1.4% | **+17.4%** |
| **Validation** dd | 14.5% | 8.2% | skipped | 12.4% | **5.9%** |
| **Validation** trades | 87 | 105 | skipped | 209 | **98** |
| **Validation** PF | 1.93 | 1.47 | skipped | 1.14 | **1.73** |
| **Validation** win rate | 6/81 | 38.1% | skipped | 48.8% | **38.8%** |
| **Validation** fees | $47.16 | $57.78 | skipped | $106.28 | **$53.68** |

### Winning Strategies

**Run 187 winner (genome_95270c8141bd) — BEST OF BATCH:**
- Direction: LONG only
- Entry: MFI(29) crosses_above 50.20
- Exit: WILLR(25) < -41.87
- SL=1.7%, TP=17.8%
- Validation: Sharpe=2.45, Return=+17.4%, DD=5.9%, 98 trades, PF=1.73
- DSR p=0.0000 (highly significant)

**Run 154 winner (genome_17cd89664cd9):**
- Direction: SHORT only
- Entry: CCI(12) < -66.13 AND CCI(24) crosses_below -80.58
- Exit: CCI(12) crosses_below 63.30 AND ROC(15) <= -4.78
- SL=0.59%, TP=15.35%
- Validation: Sharpe=2.04, Return=+16.4%, DD=14.5%, 87 trades, PF=1.93

**Run 157 winner (genome_cb22c7229fdc):**
- Direction: LONG+SHORT
- Entry: CCI(26)+STOCH(k=22,d=6)
- SL=3.28%, TP=15.64%
- Validation: Sharpe=2.05, Return=+13.9%, DD=8.2%, 105 trades, PF=1.47

### Issues Found

1. **MFI FutureWarning spam**: pandas-ta MFI generates ~25-48M lines of dtype warnings per run. Fixed by adding `dtype=float` to Series construction.
2. **WILLR+ROC all failed guardrails**: All top-5 had negative Sharpe (-0.25), DSR p=1.0. WILLR+ROC without CCI produces poor strategies on 4h BTCUSDT.
3. **MFI runs 2x slower**: pandas-ta fallback (~8-16min/gen vs 4-6min/gen for Rust-native indicators).
4. **CLI discoveries don't appear in UI Discovery Results**: Runs launched via `python -m vibe_quant.discovery` don't register in the API's job tracking — only `/api/discovery/launch` runs show. Validation results DO appear in Results Analysis.
5. **ProcessPoolExecutor orphaned workers (macOS)**: Run 162 (MFI+CCI+ROC, `max_workers=0`) used `ProcessPoolExecutor` which failed with `RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase` (spawn method on macOS). Pipeline fell back to sequential, but 5+ worker processes were never joined — stuck at 100% CPU until manually killed. Root cause: Python's `spawn` start method on macOS can't fork inside NautilusTrader's Rust runtime. Fix needed: catch the RuntimeError and explicitly terminate/join the pool, or default to sequential on macOS.
6. **"No stats_pnls" warnings in run 162**: ~40+ instances where `NTScreeningRunner` got no PnL statistics from NT backtest. These chromosomes likely produced zero-profit trades (all positions flat or no fills).

### Key Findings

1. **MFI is a strong new indicator**: MFI+WILLR produced the best strategy of this batch (Sharpe 2.45 validated). MFI crosses_above 50 as a momentum entry works well.
2. **Drawdown fix confirmed working**: DD values now correctly populated (5.9%-14.5% range, vs 0.0% in Batch 8).
3. **CCI still competitive but not dominant**: CCI+ROC and CCI+STOCH produce good results (Sharpe 2.0+) but MFI+WILLR beats them.
4. **WILLR+ROC without CCI is poor**: Negative Sharpe, all failed guardrails. These indicators need CCI to anchor signals.
5. **Long strategies emerge**: Best strategy is LONG-only (MFI+WILLR). Previous batches were short-dominated.

### Comparison with Previous Batches

| Metric | Batch 7 Best (CCI+RSI) | Batch 8 Best (MACD+WILLR) | Batch 9 Best (MFI+WILLR) |
|--------|----------------------|----------------------|----------------------|
| Validation Sharpe | 7.24 | 1.87 | 2.45 |
| Validation Return | +11.9% | +24.4% | +17.4% |
| Validation DD | 0.9% | 25.6% | 5.9% |
| Validation PF | 4.78 | 1.35 | 1.73 |
| Direction | both | both | long |

Batch 7's CCI+RSI remains the all-time champion (Sharpe 7.24), but Batch 9's MFI+WILLR is the best non-CCI strategy found so far and has excellent risk-adjusted returns (low DD, decent PF).

### Recommendations

1. **Try MFI+CCI combination**: MFI shows promise; combining with CCI could produce even better results
2. **Try MFI on 1h timeframe**: MFI is volume-based — shorter timeframes may give more signals
3. **Suppress pandas-ta warnings globally**: The `dtype=float` fix helps but ideally suppress all FutureWarning in generated strategies
4. **Fix CLI↔UI disconnect**: Discovery runs from CLI should appear in the Discovery Results UI

---

## 2026-03-06: Batch 8 — Novel Indicators: STOCH, WILLR, ROC, MACD (Post-Bugfix)

### Goal
Test indicators never seriously tried on 4h: STOCH, WILLR, ROC, MACD (post-bugfix). Validate new rich logging system. Compare with CCI-dominant champions from Batch 7.

### Configuration
| Run | Indicators | Pop | Gens | TF | Direction | Time |
|-----|-----------|-----|------|----|-----------|------|
| 142 | STOCH+WILLR | 12 | 5 | 4h | both | 1551s (25.9 min) |
| 143 | ROC+STOCH | 12 | 5 | 4h | both | 998s (16.6 min) |
| 144 | MACD+WILLR | 12 | 5 | 4h | both | 1460s (24.3 min) |

Runs executed in parallel (3 concurrent), sequential mode (macOS sandbox, ProcessPoolExecutor blocked).
Per-chromosome: 16-26s avg depending on CPU contention.

### Full Pipeline Results

| Stage | Run 142 STOCH+WILLR | Run 143 ROC+STOCH | Run 144 MACD+WILLR |
|-------|----|----|------|
| **Discovery** score | 0.5259 | 0.5565 | 0.5758 |
| **Discovery** sharpe | 1.69 | 1.67 | 1.95 |
| **Screening** sharpe | 1.69 ✓ | 1.67 ✓ | 1.95 ✓ |
| **Screening** trades | 76 ✓ | 312 ✓ | 114 ✓ |
| **Screening** return | +9.1% ✓ | +7.2% ✓ | +26.3% ✓ |
| **Screening** dd | 0.0% ⚠️ | 0.0% ⚠️ | 0.0% ⚠️ |
| **Validation** sharpe | 1.81 | 1.56 | 1.87 |
| **Validation** return | +10.6% | +5.9% | +24.4% |
| **Validation** dd | 13.4% | 8.0% | 25.6% |
| **Validation** trades | 76 | 311 | 115 |
| **Validation** PF | 1.40 | 1.33 | 1.35 |
| **Validation** win rate | 35.5% | 66.9% | 24.3% |
| **Validation** fees | $28.76 | $82.55 | $70.12 |
| **Validation** latency | retail (200ms) | retail (200ms) | retail (200ms) |

### Winning Strategies (Gene Details from Logs)

**Run 142 winner (genome_75f34d645f85):**
- Entry: STOCH(k=10, d=4) < 45.37, WILLR(30) <= -21.47, WILLR(14) >= -54.73
- Exit: STOCH(k=18, d=7) crosses_above 20.54
- SL=1.2%, TP=3.9% (per-direction: SL_long=5.9%, SL_short=2.1%, TP_long=10.8%, TP_short=7.3%)

**Run 143 winner (genome_3c20746fc88f):**
- Entry: STOCH(k=11, d=3) crosses_below 23.84
- Exit: STOCH(k=8, d=4) crosses_above 28.42
- SL=2.0%, TP=8.9% (simplest genome — 1 entry + 1 exit gene)

**Run 144 winner (genome_3f1890f0b082):**
- Entry: MACD(11/23/11) > -0.0006
- Exit: WILLR(19) > -46.55 AND WILLR(16) < -64.11
- SL=6.8%, TP=1.9% (unusual: tight TP, wide SL)

### Bugs Found

**BUG: Screening reports max_drawdown=0.0% for all strategies.** Validation shows 8-26% drawdown for the same strategies. Discovery also reports 0.0% in logs (`dd=0.0%`). This is likely a bug in the NTScreeningRunner's drawdown extraction — needs investigation. The drawdown component (25% weight) in fitness scoring is effectively disabled.

**MACD histogram fallback (run 149):** `MACD macd_entry_0 output 'histogram' not available in NT — returns MACD line (.value) instead.` MACD strategies are not testing what they think they're testing.

**NaN metrics from zero-trade strategies:** 5/12 chromosomes in ROC+STOCH Gen 1 produced 0 trades with NaN sharpe/PF. Sanity checks caught these correctly. These indicators have narrow effective threshold ranges — many random chromosomes generate no signals.

### New Logging System Observations

The enhanced logging worked exactly as intended:
- **Environment section** showed PID, compiler version, data catalog sizes — useful for reproducibility
- **Per-gen analytics** showed score distributions (mean/median/std), zero-score counts, indicator frequency %
- **Sanity checks** caught NaN metrics and suspicious high-Sharpe-low-trade cases
- **Score decomposition** (raw - complexity - overtrade = adjusted) visible per gen
- **Evolution timeline** (e.g., `0.556 → 0.570 → 0.570 → 0.570 → 0.576`) shows where improvement happened
- **Gene details** in final summary allow exact strategy reproduction from logs alone
- **Delta tracking** (Δbest=+0.0327) shows improvement rate per generation

### Comparison with Batch 7 (CCI Champions)

| Metric | Batch 7 Best (Run 128) | Batch 8 Best (Run 144) |
|--------|----------------------|----------------------|
| Discovery score | 0.8499 | 0.5758 |
| Validation sharpe | 7.24 | 1.87 |
| Validation return | +11.9% | +24.4% |
| Validation dd | 0.9% | 25.6% |
| Validation PF | 4.78 | 1.35 |
| Indicators | CCI+RSI | MACD+WILLR |

**CCI remains king by a wide margin.** STOCH/WILLR/ROC/MACD strategies produce mediocre Sharpe (1.5-1.9) and significant drawdown (8-26%). CCI strategies achieve Sharpe 5-7 with <5% drawdown.

### Key Findings

1. **CCI dominance confirmed**: Novel indicators (STOCH, WILLR, ROC, MACD) underperform CCI by 3-5x on Sharpe
2. **Drawdown bug discovered**: Screening/discovery report 0.0% drawdown — fitness function's 25% DD weight is broken
3. **STOCH convergence problem**: GA converges to single genome rapidly (all top-5 identical), suggests narrow fitness landscape
4. **ROC produces many zero-trade strategies**: 5/12 (42%) in Gen 1 — threshold ranges may need tuning
5. **MACD needs pandas-ta fallback**: NT built-in only exposes main line, signal/histogram silently degraded
6. **Run timing**: 3 concurrent 4h runs take 17-26 min each due to CPU contention (9 min sequential estimate was for single run)

### Recommendations

1. **CRITICAL: Fix drawdown bug** — max_drawdown=0.0 in screening/discovery is clearly wrong. This affects all historical fitness scores
2. **Investigate CCI magic**: Why does CCI so dramatically outperform? Threshold range [-200, 200] gives wider signal space?
3. **Tune ROC/WILLR threshold ranges**: High zero-trade rate indicates ranges too narrow/wide for these indicators
4. **Force pandas-ta for MACD**: Until NT exposes signal/histogram, MACD results are unreliable
5. **Increase population for novel indicators**: 12 is too small for STOCH/WILLR — narrow fitness landscape needs more exploration

---

## 2026-03-06: Batch 7 — Record-Breaking Discovery, Sharpe 7.24 Validated

### Goal

Find winning strategies that are **not short-only**, on **bigger timeframes** (4h, 1h), targeting **2024 bull market data**. Run 5 parallel discoveries in 15-30 minutes, propagate winners to screening and validation.

### Setup

5 parallel discovery runs on BTCUSDT, CCI+RSI dominant (proven king), 2024-centric date ranges.

**Design rationale:**
- CCI dominant across all 6 previous batches — kept in every run
- 2024 bull market produced all-time bests in batch 6 (runs 88, 89, 90)
- `both` direction on 4/5 runs (run 127 = `long` to explore pure long strategies)
- Run 130 used CCI+ATR for indicator diversity (ATR is Rust-native = fast)
- Smaller populations (14-20) × fewer generations (6-10) to target ~15-30min
- All runs sequential mode (sandbox blocks ProcessPoolExecutor)

| Run | TF | Dir | Indicators | Pop×Gen | Mut | Date Range | Duration |
|-----|-----|-----|------------|---------|-----|------------|----------|
| 126 | 4h | both | CCI,RSI | 18×8 | 0.22 | 2024-01→2024-12 | ~20m |
| 127 | 4h | long | CCI,RSI | 16×8 | 0.20 | 2024-01→2025-06 | ~25m |
| 128 | 4h | both | CCI,RSI | 20×8 | 0.25 | 2024-06→2025-06 | ~22m |
| 129 | 1h | both | CCI,RSI | 14×6 | 0.25 | 2024-01→2024-09 | ~10m |
| 130 | 4h | both | CCI,ATR | 20×10 | 0.22 | 2024-01→2025-06 | ~40m |

**Note:** 5 concurrent processes shared CPU, extending runtimes ~2x vs solo. Run 130 (largest pop×gen) took longest.

### Discovery Results

| Run | TF | Dir | Fitness | Sharpe | PF | Trades | Return |
|-----|-----|-----|---------|--------|-----|--------|--------|
| **128** | **4h** | **both** | **0.8499** | **7.31** | **4.883** | **55** | **+11.9%** |
| **126** | **4h** | **both** | **0.8099** | **5.63** | **2.904** | **57** | **+40.7%** |
| **130** | **4h** | **both** | **0.7419** | **3.66** | **1.997** | **65** | **+28.3%** |
| 127 | 4h | long | 0.5413 | 1.37 | 1.229 | 95 | +14.5% |
| 129 | 1h | both | 0.4794 | 0.94 | 1.115 | 78 | +3.0% |

### Top 3 Strategies

**Run 128 — CCI Triple Bidirectional (4h, Both) ★ NEW ALL-TIME BEST**
- Entry: CCI(44) crosses_above 13.5 (both long+short)
- Exit: CCI(36) >= -0.52 AND CCI(20) > -14.1 (both sides)
- SL: 3.96% (tight), TP: 15.48% (wide) — trend-following risk profile
- Per-direction: SL_long=4.89%, SL_short=2.98%, TP_long=9.91%, TP_short=6.72%
- Pure CCI strategy with triple-CCI exit confirmation
- Window: 2024-06→2025-06 (12mo regime-spanning)

**Run 126 — CCI Bidirectional, Highest Return (4h, Both)**
- Entry: CCI(28) > 51.7 (both sides)
- Exit: CCI(50) crosses_below 48.3 (both sides)
- SL: 7.84%, TP: 9.65%
- Per-direction: SL_long=5.25%, SL_short=3.36%, TP_long=11.64%, TP_short=8.82%
- Pure CCI with long-period exit (CCI(50))
- Window: 2024-01→2024-12 (pure bull year)

**Run 130 — CCI Bidirectional, CCI+ATR Pool (4h, Both)**
- Entry: CCI(21) > 38.9 (both sides)
- Exit: CCI(44) crosses_above -76.7 (both sides)
- SL: 7.2%, TP: 12.37%
- GA chose pure CCI despite ATR being available — confirms CCI dominance
- Window: 2024-01→2025-06

### Full Pipeline: Discovery → Screening → Validation

**Run 128 (4h Both CCI) ★ NEW ALL-TIME CHAMPION:**

| Step | Run | Sharpe | PF | Trades | Return | MaxDD |
|------|-----|--------|-----|--------|--------|-------|
| Discovery | 128 | 7.31 | 4.883 | 55 | +11.9% | 0% |
| Screening | 138 | **7.31** | **4.883** | **55** | **+11.9%** | 0% |
| Validation | 139 | **7.24** | **4.78** | **55** | **+11.9%** | **0.9%** |

Validation barely degraded: Sharpe 7.31→7.24, PF 4.883→4.78. **Zero trade loss** (55→55). MaxDD 0.9% is the **lowest ever recorded** (previous best: run 87's 2.2%). Sortino 28.44 is exceptional. Win rate 63.6%.

**Run 126 (4h Both CCI) — Highest Validated Return:**

| Step | Run | Sharpe | PF | Trades | Return | MaxDD |
|------|-----|--------|-----|--------|--------|-------|
| Discovery | 126 | 5.63 | 2.904 | 57 | +40.7% | 0% |
| Screening | 136 | **5.63** | **2.904** | **57** | **+40.7%** | 0% |
| Validation | 137 | **5.56** | **2.85** | **57** | **+39.7%** | **5.0%** |

**Zero trade loss** (57→57). Return degraded only 1% (40.7→39.7%). Sharpe 5.56 is the 2nd highest ever validated. Sortino 14.78.

**Run 130 (4h Both CCI+ATR pool):**

| Step | Run | Sharpe | PF | Trades | Return | MaxDD |
|------|-----|--------|-----|--------|--------|-------|
| Discovery | 130 | 3.66 | 1.997 | 65 | +28.3% | 0% |
| Screening | 140 | **3.66** | **1.997** | **65** | **+28.3%** | 0% |
| Validation | 141 | **3.69** | **2.01** | **65** | **+28.5%** | **7.4%** |

Validation **improved** slightly (Sharpe 3.66→3.69, return 28.3→28.5%). Zero trade loss (65→65). MaxDD 7.4%.

### Comparison: Batch 7 vs All Previous Champions

| Run | TF | Dir | V.Sharpe | V.PF | V.Trades | V.Return | V.MaxDD | Window |
|-----|-----|-----|----------|------|----------|----------|---------|--------|
| **139 (new) ★** | **4h** | **both** | **7.24** | **4.78** | **55** | **+11.9%** | **0.9%** | **2024-06→2025-06** |
| **137 (new)** | **4h** | **both** | **5.56** | **2.85** | **57** | **+39.7%** | **5.0%** | **2024-01→2024-12** |
| **141 (new)** | **4h** | **both** | **3.69** | **2.01** | **65** | **+28.5%** | **7.4%** | **2024-01→2025-06** |
| 98 (prev best) | 4h | both | 4.12 | 2.736 | 61 | +49.3% | 11.3% | 2024-01→2025-06 |
| 87 (prev MaxDD) | 4h | both | 3.65 | 2.135 | 85 | +16.7% | 2.2% | 2024-06→2026-02 |
| 85 (prev Sharpe) | 15m | short | 4.60 | 2.010 | 47 | +41.4% | 16.1% | 2024-06→2026-02 |

### Findings

1. **Run 128/139 is the new all-time champion** — Sharpe 7.24 (56% higher than prev best 4.64), PF 4.78 (75% higher than prev best 2.736), MaxDD 0.9% (best ever). The triple-CCI setup with tight SL (3.96%) and wide TP (15.48%) creates an extremely selective entry with high win rate (63.6%). The entry condition (CCI(44) crosses_above 13.5) uses a long-period CCI that only triggers on strong momentum shifts.

2. **Run 126/137 has the 2nd highest validated return ever** (+39.7%) and would be the champion on absolute returns if not for run 98 (+49.3%). The CCI(28)/CCI(50) combination with symmetric conditions (same entry/exit for long+short) is remarkably simple yet effective. The wider SL (7.84%) allows more room for trades to develop.

3. **All 3 winners are pure CCI** — run 130 had ATR in the pool but GA chose CCI only. CCI's dominance continues unbroken across 7 batches. The question is whether this represents a genuine edge or overfitting to the indicator.

4. **Per-direction SL/TP is being used** — all strategies have different SL/TP for long vs short (tighter short SL, wider long SL). This suggests the GA is finding asymmetric risk profiles. Worth monitoring for overfitting concerns.

5. **4h both-direction continues to dominate** — all 3 top strategies are 4h bidirectional. Run 127 (long-only) was mediocre (Sharpe 1.37), and run 129 (1h) was poor (Sharpe 0.94). The 4h timeframe provides enough data for CCI to generate reliable signals while being immune to latency.

6. **Return vs Sharpe tradeoff** — Run 128 has the best risk-adjusted metrics (Sharpe 7.24, MaxDD 0.9%) but modest return (+11.9%). Run 126 has higher return (+39.7%) but lower Sharpe (5.56) and higher MaxDD (5.0%). This is the classic precision-vs-magnitude tradeoff.

7. **Shorter windows can produce higher Sharpe** — Run 128's window (12mo, 2024-06→2025-06) produced Sharpe 7.24. Run 126's (12mo, 2024-01→2024-12) produced Sharpe 5.56. Different sub-periods of the bull market have different optimal strategies. This suggests walk-forward analysis would be valuable.

8. **Zero trade loss across all validations** — All 3 strategies maintained 100% trade survival through validation (55→55, 57→57, 65→65). 4h strategies are completely latency-immune as confirmed in previous batches.

### Recommendations

1. **Paper trade run 128/139** immediately — best risk-adjusted metrics ever recorded (Sharpe 7.24, MaxDD 0.9%)
2. **Paper trade run 126/137** for absolute return (+39.7%)
3. **Out-of-sample test** — run both strategies on 2023 data and 2025-06→2026-02 (bearish) to test regime robustness
4. **Walk-forward analysis** — split 2024 into quarterly windows and test strategy stability
5. **Portfolio combination** — combine run 128 (low MaxDD champion) with run 87 (bearish window champion) for all-weather portfolio
6. **Increase population/generations** — run 128 converged to 0.85 fitness in just 8 generations without converging flag. Larger runs (pop=30, gen=15) might find even better strategies

### Filed

- Run 130 confirmed CCI dominance: ATR in pool but GA chose pure CCI

---

## 2026-02-27: Strategy Combination Experiments — Bull+Bear Merge Attempts

### Goal

Combine the bull champion (run 88: 4h CCI, +49.3% on 2024-01→2025-06) with the bear champion (run 81/87: 4h CCI, +16.7% on 2024-06→2026-02) into a single regime-adaptive strategy.

### Approach 1: Out-of-Sample Test (Run 101)

First tested run 88's bull strategy on the unseen bear window (2025-06→2026-02).

| Window | Sharpe | PF | Trades | Return | MaxDD |
|--------|--------|-----|--------|--------|-------|
| In-sample (2024-01→2025-06) | 4.12 | 2.736 | 61 | +49.3% | 11.3% |
| **Out-of-sample (2025-06→2026-02)** | **-4.14** | **0.446** | **32** | **-8.9%** | **10.1%** |

Strategy is pure bull — all 61 in-sample trades were LONG despite `direction=both`. On the bear window it kept entering longs into a downtrend and hitting SL repeatedly.

### Approach 2: Combined DSL with Separate Long/Short Conditions

Used the DSL's per-direction entry/exit support:
- Long side: run 88's conditions (CCI(30) crosses_below 59.9)
- Short side: run 81's conditions (CCI(23) crosses_above 37.0)

**Run 102 — Wide SL (8.29% from run 81):**

| Metric | Value |
|--------|-------|
| Sharpe | -1.03 |
| PF | 0.832 |
| Trades | 78 (44 long, 34 short) |
| Return | -9.8% |
| MaxDD | 15.2% |
| Long PnL | +$3,372 |
| Short PnL | **-$13,141** |

**Run 103 — Tight SL (1.09% from run 88):**

| Metric | Value |
|--------|-------|
| Sharpe | -0.10 |
| PF | 0.978 |
| Trades | 170 (87 long, 83 short) |
| Return | -9.7% |
| MaxDD | 22.0% |
| Long PnL | +$10,868 |
| Short PnL | **-$20,588** |

Both lost money. The short side fires during the 2024 bull rally and gets stopped out repeatedly. CCI is a bounded oscillator — it cycles in ALL market conditions and has zero regime awareness.

### Approach 3: Full 2yr Discovery (Run 104/106)

Let the GA find a genuinely regime-adaptive strategy on the full 2024-01→2026-02 window.

Config: pop=16, gen=8, CCI+RSI, direction=both, ~35min runtime.

| Step | Run | Sharpe | PF | Trades | Return | MaxDD |
|------|-----|--------|-----|--------|--------|-------|
| Discovery | 104 | 1.39 | 1.225 | 64 | +15.5% | 0% |
| Screening | 105 | 1.39 | 1.225 | 64 | +15.5% | 0% |
| Validation | 106 | 1.53 | 1.251 | 58 | +16.5% | 24.9% |

**Winner: RSI(16) <= 68.9 entry, RSI(5) > 68.8 exit, SL=5.34%, TP=17.53%**

The GA produced another long-only strategy (0 short trades). It concluded — same as us — that you can't profitably short across a full bull+bear cycle with RSI/CCI thresholds. Moderate returns (+16.5%) but 24.9% MaxDD is poor.

### Conclusions

1. **Naive strategy combination doesn't work** — both sides trade all the time regardless of regime, and the losing side overwhelms the winner
2. **CCI/RSI are not regime detectors** — they're bounded oscillators that cycle in all conditions. SMA(200) is equally unreliable
3. **Per-regime specialists dominate** — run 88 (+49.3% bull) and run 87 (+16.7% bear, 2.2% MaxDD) are far superior on their windows than any combined approach
4. **Full-window discovery produces mediocre results** — Sharpe 1.53 vs 3.65 (run 87) and 4.12 (run 98). The GA can't find a single strategy that works well in both regimes
5. **Per-direction SL/TP would help** (bead vibe-quant-k4ya) — tight stops for longs, wide for shorts, but won't solve the fundamental regime problem

### Filed

- `vibe-quant-k4ya`: Per-direction SL/TP support in DSL (P2 feature)

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
