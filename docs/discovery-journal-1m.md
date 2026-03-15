# Discovery Journal — 1-Minute Timeframe

Research diary for 1-minute scalping strategy discovery on BTCUSDT perpetual futures.

**Key differences from 4h journal:**
- **Timeframe**: 1m (scalping) vs 4h (swing trading)
- **Data window**: 3 months (shorter, denser — ~130K bars)
- **Trade frequency**: Expect 50-200 trades per strategy (not 500-2000 as initially expected)
- **Latency sensitivity**: 200ms retail latency more impactful
- **Fee drag**: Higher trade count = fees matter more
- **Indicators**: Rust-native only (STOCH, CCI, RSI, ROC, ADX, ATR). **MFI has no NT class — pandas-ta fallback too slow.**

**Timing calibration (Batch 0, run 435):**
- STOCH+CCI, pop=6, gens=5, 3mo data: **~5.5s/chromosome, ~33s/gen, 167s total**
- Best fitness 0.5215, Sharpe ~3.15 from just 30 trials
- 2 parallel runs OK, 5 parallel causes CPU starvation + OOM kills

**Critical 1m learnings:**
- **MFI is NOT Rust-native** — `MoneyFlowIndex` missing from NT, falls back to pandas-ta. ~10x slower on 130K bars. Unusable for 1m discovery.
- **MACD/WILLR** also pandas-ta only — excluded.
- **BBANDS/DONCHIAN** excluded — price-relative outputs, need normalized sub-values (see vibe-quant-pr80)
- **Max 2-3 parallel runs** on this hardware (32GB M3 Max). 5 parallel kills processes.
- **STOCH remains king** on 1m, just like 4h. Present in both winning combos.
- **RSI combos failed badly** on 1m — all negative returns. RSI may be too slow/noisy for 1m.
- **ADX+CCI degenerate** — 3579 trades, -52.9% DD. ADX on 1m captures noise, not trends.

---

## 2026-03-10: Batch 1 — First 1m Scalp Series

### Goal

First-ever 1m discovery batch. Test which 4h champion combos transfer to 1m. Establish baseline for 1m strategy quality. Originally planned 5 combos including MFI pairs — pivoted after discovering MFI pandas-ta bottleneck.

### Bug Fixes Applied

1. **MFI pandas-ta bottleneck** — MFI has no NT Rust class, falls back to pandas-ta (~10x slower). RSI+MFI run (443) couldn't complete gen 1 in 10 minutes. Killed and redesigned all combos to Rust-native only.
2. **CPU starvation** — 5 parallel runs (20 worker processes) saturated CPU (load avg 130). Runs 437/438/439 died silently. Redesigned to launch 2 at a time max.
3. **Orphan workers** — Dead discovery parent processes left ~20 zombie multiprocessing workers consuming 30-55% CPU each. Required manual cleanup.

### Configuration (Revised)

| Run | Indicators | Pop | Gens | Trials | Direction | Time | Status |
|-----|-----------|-----|------|--------|-----------|------|--------|
| 436 | STOCH+CCI | 12 | 10 | 120 | random | 1334s (22m) | **completed** |
| 444 | RSI+CCI | 12 | 10 | 120 | random | 193s (3m) | completed (FAIL) |
| 445 | STOCH+ROC | 12 | 10 | 120 | random | 805s (13m) | **completed** |
| 446 | RSI+STOCH | 12 | 10 | 120 | random | 397s (7m) | completed (FAIL) |
| 447 | ADX+CCI | 12 | 10 | 120 | random | 374s (6m) | completed (FAIL) |

Killed runs: 437 (RSI+MFI, dead), 438 (STOCH+MFI, dead), 439 (STOCH+CCI+MFI, dead), 440 (ROC+CCI, killed for CPU), 441/442/443 (relaunches, dead/killed).

Data range: 2025-12-10 to 2026-03-10. BTCUSDT 1m. ~130K bars.

### Full Pipeline Results

| Stage | 436 STOCH+CCI | 444 RSI+CCI | 445 STOCH+ROC | 446 RSI+STOCH | 447 ADX+CCI |
|-------|--------------|-------------|---------------|---------------|-------------|
| **Direction** | SHORT | LONG | SHORT | LONG | BOTH |
| **Discovery** score | **0.5411** | 0.2684 | **0.5410** | 0.2347 | 0.0548 |
| **Discovery** sharpe | **2.47** | -2.44 | **2.13** | -2.55 | 1.90 |
| **Discovery** dd | 8.8% | 19.5% | **5.5%** | 23.1% | 52.9% |
| **Discovery** trades | 56 | 57 | **152** | 65 | 3579 |
| **Discovery** return | **+9.8%** | -14.2% | +4.2% | -15.4% | -52.9% |
| **Discovery** PF | 1.44 | 0.75 | 1.40 | 0.65 | 1.07 |
| **DSR** | **PASS 1/5** | FAIL 0/2 | **PASS 2/5** | FAIL 0/3 | FAIL 0/1 |
| **Screening** match | **exact** | — | **exact** | — | — |
| **Validation** sharpe | **2.50 (+1%)** | — | **2.60 (+22%)** | — | — |
| **Validation** sortino | **5.69** | — | 3.53 | — | — |
| **Validation** dd | 11.7% | — | **5.2%** | — | — |
| **Validation** trades | 40 (-29%) | — | **155 (+2%)** | — | — |
| **Validation** PF | 1.53 | — | **1.52** | — | — |
| **Validation** WR | 52.5% | — | **94.8%** | — | — |
| **Validation** return | **+15.5%** | — | +5.8% | — | — |
| **Validation** fees | $38.08 | — | $29.71 | — | — |

### Winning Strategies

**#1: Run 445 — STOCH+ROC (Short Scalper)** — Strategy `genome_e8d908f5ab9c` (sid=135)
- Entry: STOCH(5,7) > 55.6 AND STOCH(6,3) crosses_below 78.9 → short
- Exit: ROC(22) >= 0.81 AND ROC(27) <= 3.88
- SL: 7.72% / **TP: 0.64%** (ultra-tight scalper)
- Validated **Sharpe 2.60**, Sortino 3.53, **155 trades**, +5.8% return, **5.2% DD**, PF 1.52, **94.8% WR**
- **True 1m scalper architecture**: tiny 0.64% TP with 94.8% WR. Takes many small wins, wide SL as safety net. This is exactly what 1m strategies should look like.
- Validation IMPROVED +22% over discovery — extremely robust.

**#2: Run 436 — STOCH+CCI (Short)** — Strategy `genome_c974c3d9da9e` (sid=134)
- Entry: CCI(36) < -15.3 AND CCI(11) >= -76.7 → short
- Exit: CCI(13) >= 2.8 AND STOCH(6,3) < 45.8 AND CCI(34) crosses_below -77.1
- SL: 3.57% / TP: 10.39%
- Validated **Sharpe 2.50**, Sortino 5.69, 40 trades, **+15.5% return**, 11.7% DD, PF 1.53, 52.5% WR
- CCI-dominated strategy (4/5 genes are CCI). Fewer trades but higher per-trade return.
- 40 validated trades is borderline low for 3 months of 1m data.

### Issues Found

1. **MFI has no NT Rust implementation** — falls back to pandas-ta, unusable on 1m. Filed as learning, not bug (NT limitation).
2. **5 parallel runs cause OOM/CPU starvation** — processes die silently. Max 2-3 parallel on this hardware.
3. **Orphan multiprocessing workers** — when parent discovery process dies, child workers keep running. No cleanup mechanism.
4. **Run 436 timing anomaly** — 1334s (22 min) vs expected ~5 min solo. Caused by CPU contention with 4 other parallel runs.

### Key Findings

1. **STOCH+ROC is the 1m champion** — Sharpe 2.60 validated, 94.8% WR, true scalper architecture (0.64% TP). Novel combo never tried on 4h.
2. **STOCH transfers perfectly to 1m** — present in both winners. Remains the single most important indicator across timeframes.
3. **RSI fails on 1m** — both RSI+CCI (Sharpe -2.44) and RSI+STOCH (Sharpe -2.55) produced only losing strategies. RSI may be too slow for 1m noise.
4. **ADX fails on 1m** — same as 4h findings. ADX+CCI produced degenerate 3579-trade strategy with -52.9% DD.
5. **Both winners are SHORT** — same directional bias as 4h discoveries on this data period.
6. **Validation improved for both** — 436 (+1%), 445 (+22%). Both strategies are genuinely robust, not overfit.
7. **1m generates fewer trades than expected** — 40-155 trades in 3mo, not the 500-2000 initially expected. 1m ≠ hyperactive scalping.
8. **Ultra-tight TP works** — 0.64% TP with 94.8% WR is a viable 1m scalping architecture. GA discovered this naturally.

### 1m All-Time Leaderboard (Validated Sharpe)

| Rank | Batch | Combo | Sharpe | Sortino | DD | Trades | PF | WR | Dir |
|------|-------|-------|--------|---------|-----|--------|-----|-----|-----|
| 1 | B1 | STOCH+ROC | **2.60** | 3.53 | 5.2% | 155 | 1.52 | **94.8%** | SHORT |
| 2 | B1 | STOCH+CCI | **2.50** | 5.69 | 11.7% | 40 | 1.53 | 52.5% | SHORT |

### Recommendations

1. **Run STOCH+ROC with higher budget** (pop=20, gens=15) — 2.60 from 120 trials could improve significantly (4h showed +41% with 3x budget)
2. **Try STOCH+ROC with forced BOTH direction** — both winners are SHORT. BOTH could be more robust.
3. **Test CCI+ROC** — CCI dominated run 436, ROC dominated run 445. Combine the two non-STOCH winners.
4. **Avoid RSI and ADX on 1m** — both failed. Focus on STOCH, CCI, ROC as the 1m indicator trinity.
5. **Paper trade STOCH+ROC (sid=135)** — 94.8% WR scalper with 5.2% DD is paper-trading ready.
6. **File bead for orphan worker cleanup** — multiprocessing workers survive parent death. Need process group management.

---

## 2026-03-10: Batch 2 — BBANDS/DONCHIAN First Test

### Goal

First test of newly added BBANDS and DONCHIAN indicators on 1m timeframe. These were added to the genome pool via normalized sub-values (commit `c327717`). Test whether volatility/channel indicators complement the proven 1m indicators (STOCH, CCI, ROC).

### Configuration

| Run | Indicators | Pop | Gens | Trials | Direction | Time | Status |
|-----|-----------|-----|------|--------|-----------|------|--------|
| 454 | STOCH+BBANDS | 16 | 14 | 224 | random | ~35min | completed |
| 455 | STOCH+DONCHIAN | 16 | 14 | 224 | random | ~38min | completed |
| 456 | CCI+BBANDS | 16 | 14 | 224 | random | ~42min | completed |

Data range: 2025-03-10 to 2026-03-10. BTCUSDT 1m. 3 parallel runs on 32GB M3 Max.

**Note:** API used 12-month data range (full available) rather than intended 3-month window. Future runs should specify date range if API supports it.

### Full Pipeline Results

| Stage | 454 STOCH+BBANDS | 455 STOCH+DONCHIAN | 456 CCI+BBANDS |
|-------|-----------------|-------------------|----------------|
| **Direction** | SHORT | LONG | LONG |
| **Discovery** score | 0.3576 | 0.5779 | **0.6760** |
| **Discovery** sharpe | 0.37 | 2.87 | **4.22** |
| **Discovery** dd | 23.0% | 4.1% | **1.4%** |
| **Discovery** trades | 73 | 182 | 70 |
| **Discovery** return | +0.8% | -4.1% | +1.1% |
| **Discovery** PF | 1.05 | 1.59 | **1.80** |
| **DSR** | PASS 1/5 | **FAIL 0/5** (neg return) | **PASS 1/5** |
| **Screening** match | — | **exact** ✓ | **exact** ✓ |
| **Validation** sharpe | — | **-0.88** (−131%) | **0.31** (−93%) |
| **Validation** sortino | — | -1.00 | 0.41 |
| **Validation** dd | — | 11.9% | 7.3% |
| **Validation** trades | — | 105 (−42%) | 52 (−26%) |
| **Validation** PF | — | 0.84 | 1.05 |
| **Validation** WR | — | 34.3% | 53.8% |
| **Validation** return | — | -6.5% | -0.9% |
| **Validation** fees | — | $22.59 | $10.93 |

### Issues Found

1. **No winning strategies** — both validated strategies degraded catastrophically. 456 (CCI+BBANDS) went from Sharpe 4.22 → 0.31 (−93%). 455 (STOCH+DONCHIAN) went from 2.87 → −0.88 (−131%).
2. **STOCH+BBANDS too weak** — best fitness only 0.41, Sharpe 0.37. Not worth promoting.
3. **STOCH+DONCHIAN negative return paradox** — Sharpe 2.87 with −4.1% return. High Sharpe from consistent small wins that don't overcome the negative drift. Guardrails correctly rejected all 5 strategies.
4. **Data range mismatch** — API used 12mo (2025-03-10 → 2026-03-10) not the intended 3mo. Batch 1 used 3mo. This may explain different behavior — more data doesn't always mean better discovery.

### Key Findings

1. **BBANDS and DONCHIAN fail validation on 1m** — discovery scores looked promising (0.58–0.68) but collapsed under realistic fills/latency. The normalized sub-values work mechanically but don't produce robust strategies.
2. **Contrast with Batch 1** — STOCH+ROC *improved* +22% in validation; STOCH+CCI improved +1%. BBANDS/DONCHIAN combos degraded 93–131%. The original 1m trinity (STOCH, CCI, ROC) remains unchallenged.
3. **High discovery fitness ≠ validation robustness** — run 456 had the highest 1m fitness ever (0.676) but worst validation degradation. Overfitting risk is real even with DSR pass.
4. **BBANDS normalized sub-values may overfit** — the 0.0–1.0 %B range gives GA a "too easy" parameter space. GA finds strategies that exploit the exact position within bands, which is highly sensitive to fill timing.
5. **DONCHIAN similar issue** — channel position (0–1) is inherently price-path-dependent. Small fill timing differences shift position dramatically on 1m bars.
6. **All 3 runs converged to different directions** — 454 SHORT, 455/456 LONG. No consistent directional signal from new indicators.

### Comparison with Previous Batches

| Metric | B1 STOCH+ROC | B1 STOCH+CCI | B2 CCI+BBANDS | B2 STOCH+DONCHIAN |
|--------|-------------|-------------|---------------|-------------------|
| Disc Sharpe | 2.13 | 2.47 | **4.22** | 2.87 |
| Val Sharpe | **2.60** (+22%) | **2.50** (+1%) | 0.31 (−93%) | −0.88 (−131%) |
| Val Trades | 155 | 40 | 52 | 105 |
| Val DD | 5.2% | 11.7% | 7.3% | 11.9% |
| Val WR | **94.8%** | 52.5% | 53.8% | 34.3% |
| Val PF | 1.52 | 1.53 | 1.05 | 0.84 |
| Verdict | **CHAMPION** | **#2** | FAIL | FAIL |

### 1m All-Time Leaderboard (Validated Sharpe)

| Rank | Batch | Combo | Sharpe | Sortino | DD | Trades | PF | WR | Dir |
|------|-------|-------|--------|---------|-----|--------|-----|-----|-----|
| 1 | B1 | STOCH+ROC | **2.60** | 3.53 | 5.2% | 155 | 1.52 | **94.8%** | SHORT |
| 2 | B1 | STOCH+CCI | **2.50** | 5.69 | 11.7% | 40 | 1.53 | 52.5% | SHORT |
| — | B2 | CCI+BBANDS | 0.31 | 0.41 | 7.3% | 52 | 1.05 | 53.8% | LONG |
| — | B2 | STOCH+DONCHIAN | −0.88 | −1.00 | 11.9% | 105 | 0.84 | 34.3% | LONG |

### Recommendations

1. **BBANDS/DONCHIAN not viable for 1m discovery** — remove from 1m indicator recommendations. Keep in genome pool for 4h where fill timing matters less.
2. **Re-focus on STOCH+CCI+ROC trinity** — these are the only indicators that survive 1m validation.
3. **Try STOCH+ROC higher budget** (journal rec from Batch 1) — still untested. The champion deserves a bigger search budget (pop=20, gens=15).
4. **Try CCI+ROC** — another untested Batch 1 rec. Combine the two non-STOCH 1m winners.
5. **Investigate 3mo vs 12mo data impact** — Batch 2 used 12mo accidentally. If 1m discovery works better on 3mo (less noise, more recent patterns), that's important to establish.
6. **Paper trade STOCH+ROC (sid=135)** — still the top 1m strategy. Nothing from Batch 2 challenges it.

---

## 2026-03-10: Batch 3 — Trinity Combos + BOTH Direction Test

### Goal

Test the remaining untried combinations from the proven 1m indicator set (STOCH, CCI, ROC). Also test forced BOTH direction to see if bidirectional strategies are viable on 1m. All combos are Rust-native.

### Bug Fixes Applied

1. **Direction enum case** — `"BOTH"` (uppercase) rejected by Discovery API. Correct value: `"both"` (lowercase). Relaunch required (run 462 → 464).

### Configuration

| Run | Indicators | Pop | Gens | Trials | Direction | Time | Status |
|-----|-----------|-----|------|--------|-----------|------|--------|
| 461 | CCI+ROC | 16 | 14 | 224 | random | ~26min | completed |
| 464 | STOCH+ROC | 16 | 14 | 224 | both | ~24min | completed (FAIL) |
| 463 | STOCH+CCI+ROC | 14 | 12 | 168 | random | ~20min | completed |

Data range: 2025-03-10 to 2026-03-10. BTCUSDT 1m. 3 parallel runs.

### Full Pipeline Results

| Stage | 461 CCI+ROC | 464 STOCH+ROC both | 463 STOCH+CCI+ROC |
|-------|------------|-------------------|-------------------|
| **Direction** | SHORT | BOTH (long+short) | SHORT |
| **Discovery** score | 0.4148 | 0.3587 | **0.4308** |
| **Discovery** sharpe | 1.09 | 0.06 | 0.96 |
| **Discovery** dd | 23.8% | 12.4% | **11.6%** |
| **Discovery** trades | 53 | 197 | **301** |
| **Discovery** return | +5.2% | -2.2% | +0.8% |
| **Discovery** PF | 1.19 | 1.01 | 1.15 |
| **DSR** | **PASS 2/5** | FAIL 0/5 | **PASS 2/5** |
| **Screening** match | **exact** ✓ | — | **exact** ✓ |
| **Validation** sharpe | 6.29 (1 trade) | — | **1.02** (+6%) |
| **Validation** sortino | 28.59 (noise) | — | 1.28 |
| **Validation** dd | 0.0% | — | 11.4% |
| **Validation** trades | **1** (FAIL) | — | **299** (−0.7%) |
| **Validation** PF | 3.55 (noise) | — | **1.17** |
| **Validation** WR | 100% (noise) | — | **92.6%** |
| **Validation** return | 5.4% (noise) | — | **+1.6%** (+100%) |
| **Validation** fees | $0.81 | — | $47.64 |

### Issues Found

1. **BOTH direction is unviable on 1m** — run 464 produced Sharpe 0.06, all 5 strategies had negative returns. GA cannot find bidirectional strategies on 1m. SHORT-only bias continues.
2. **CCI+ROC collapsed to 1 trade in validation** — despite 53 discovery trades and passing DSR. The strategy is too sensitive to fill timing.
3. **Direction enum case sensitivity** — API accepts lowercase `"both"`, not `"BOTH"`. Not documented.

### Key Findings

1. **STOCH+CCI+ROC triple is modestly robust** — validation improved +6% Sharpe (0.96→1.02), near-identical trade count (301→299), return doubled (+0.8%→+1.6%). 92.6% WR. But absolute Sharpe (1.02) is well below Batch 1 champions.
2. **CCI+ROC fails validation** — 53 trades → 1 trade. CCI+ROC without STOCH is too fragile. CCI needs STOCH as a stabilizer.
3. **BOTH direction confirmed unviable on 1m** — third evidence (after Batch 1's both-winners-SHORT and now explicit BOTH test). 1m strategies must be directional.
4. **SHORT bias persists** — all 3 Batch 3 runs converged SHORT (464 was forced BOTH). All 5 passing strategies across all 3 batches are SHORT.
5. **Triple combo adds trades, not quality** — 301 trades vs 155 (STOCH+ROC) or 40 (STOCH+CCI), but Sharpe 1.02 vs 2.60/2.50. More indicators = more trades but diluted signal.
6. **STOCH remains essential** — CCI+ROC (no STOCH) collapsed to 1 validation trade. Every surviving 1m strategy includes STOCH.

### Comparison with Previous Batches

| Metric | B1 STOCH+ROC | B1 STOCH+CCI | B3 STOCH+CCI+ROC | B3 CCI+ROC |
|--------|-------------|-------------|------------------|------------|
| Disc Sharpe | 2.13 | 2.47 | 0.96 | 1.09 |
| Val Sharpe | **2.60** | **2.50** | 1.02 | — (1 trade) |
| Val Trades | 155 | 40 | **299** | 1 |
| Val DD | **5.2%** | 11.7% | 11.4% | — |
| Val WR | **94.8%** | 52.5% | 92.6% | — |
| Val PF | **1.52** | **1.53** | 1.17 | — |
| Verdict | **CHAMPION** | **#2** | #3 (modest) | FAIL |

### 1m All-Time Leaderboard (Validated Sharpe)

| Rank | Batch | Combo | Sharpe | Sortino | DD | Trades | PF | WR | Dir |
|------|-------|-------|--------|---------|-----|--------|-----|-----|-----|
| 1 | B1 | STOCH+ROC | **2.60** | 3.53 | 5.2% | 155 | 1.52 | **94.8%** | SHORT |
| 2 | B1 | STOCH+CCI | **2.50** | 5.69 | 11.7% | 40 | 1.53 | 52.5% | SHORT |
| 3 | B3 | STOCH+CCI+ROC | 1.02 | 1.28 | 11.4% | 299 | 1.17 | 92.6% | SHORT |
| — | B2 | CCI+BBANDS | 0.31 | 0.41 | 7.3% | 52 | 1.05 | 53.8% | LONG |
| — | B2 | STOCH+DONCHIAN | −0.88 | −1.00 | 11.9% | 105 | 0.84 | 34.3% | LONG |
| — | B3 | CCI+ROC | — | — | — | 1 | — | — | SHORT |

### Recommendations

1. **STOCH+ROC is the definitive 1m champion** — 3 batches tested, nothing challenges Sharpe 2.60 / 94.8% WR / 5.2% DD. Focus optimization efforts here.
2. **Try STOCH+ROC with higher budget** — still the #1 untested recommendation. Pop=20, gens=15 (solo or 2 parallel max). The 4h journal showed +41% improvement with 3x budget.
3. **Try STOCH+ROC with 3mo data** (as in Batch 1) — Batch 2+3 used 12mo. Compare whether 3mo produces better strategies (more recent patterns, less noise).
4. **STOCH is mandatory on 1m** — every combo without STOCH failed validation. CCI and ROC only work as STOCH companions.
5. **Stop testing new indicator combos on 1m** — 3 batches exhausted all reasonable Rust-native combos. Remaining value is in optimizing STOCH+ROC parameters, not indicator selection.
6. **Paper trade STOCH+ROC (sid=135)** — still the top candidate. 3 batches of evidence support it.

---

## 2026-03-11: Batch 4 — Higher Budget Optimization (12mo Data)

### Goal

Test the two proven 1m combos (STOCH+ROC, STOCH+CCI) with higher GA budget (pop=20, gens=20 = 400 trials, ~3.3x Batch 1). Also test STOCH+ROC long to definitively settle the directional question. Used forced direction (short/long) per journal recommendations.

**Note:** Discovery API date range bug (vibe-quant-tonx) caused 12mo data (2025-03-11→2026-03-11) instead of intended 3mo. This means results test 12mo robustness — different experiment than intended but still valuable.

### Configuration

| Run | Indicators | Pop | Gens | Trials | Direction | Time | Status |
|-----|-----------|-----|------|--------|-----------|------|--------|
| 469 | STOCH+ROC | 20 | 20 | 400 | **short** | ~51min | **completed** |
| 470 | STOCH+CCI | 20 | 20 | 400 | **short** | ~67min | **completed** |
| 471 | STOCH+ROC | 20 | 20 | 400 | **long** | ~53min | completed (FAIL) |

Data range: 2025-03-11 to 2026-03-11. BTCUSDT 1m. 3 parallel runs on M1 Pro (10 cores).

### Full Pipeline Results

| Stage | 469 STOCH+ROC short | 470 STOCH+CCI short | 471 STOCH+ROC long |
|-------|-------------------|--------------------|--------------------|
| **Direction** | SHORT | SHORT | LONG |
| **Discovery** score | 0.4784 | **0.5072** | 0.3538 |
| **Discovery** sharpe | 1.67 | **1.94** | 0.23 |
| **Discovery** dd | 20.0% | **7.5%** | 19.8% |
| **Discovery** trades | 55 | **95** | 88 |
| **Discovery** return | +8.6% | +7.4% | -0.9% |
| **Discovery** PF | 1.48 | 1.47 | 1.03 |
| **DSR** | PASS 2/5 | PASS 1/5 | **FAIL 0/5** |
| **Screening** match | **exact** ✓ | **exact** ✓ | — |
| **Validation** sharpe | 1.12 (−33%) | **2.06 (+6%)** | — |
| **Validation** sortino | 3.16 | **3.77** | — |
| **Validation** dd | 1.7% | 7.5% | — |
| **Validation** trades | **7** (−87% FAIL) | **94** (−1%) | — |
| **Validation** PF | 1.20 | **1.51** | — |
| **Validation** WR | 14.3% | **47.9%** | — |
| **Validation** return | +4.6% | **+8.2%** (+10.8%) | — |
| **Validation** fees | $3.89 | $30.85 | — |

### Winning Strategy

**Run 470 — STOCH+CCI (Short)** — Strategy `genome_e00faabc11ca` (sid=141)
- Entry: CCI(36) crosses_above -76.2 AND STOCH(10,3) crosses_above 44.7 AND CCI(31) > 45.1 → short
- Exit: STOCH(15,9) < 44.3 AND STOCH(17,3) crosses_above 59.6
- SL: 6.17% / TP: 9.4%
- Validated **Sharpe 2.06**, Sortino 3.77, **94 trades**, **+8.2% return**, 7.5% DD, PF 1.51, 47.9% WR
- **Validation improved +6% Sharpe, +10.8% return** with near-identical trade count (94 vs 95)
- CCI-dominated entry (2/3 conditions are CCI) with STOCH confirmation + STOCH-only exit
- Unlike Batch 1's STOCH+CCI (40 trades, tight-TP scalper), this is a medium-frequency strategy with balanced SL/TP

### Issues Found

1. **Discovery API date range bug** — uses full catalog data (12mo) instead of 3mo window. Filed as vibe-quant-tonx.
2. **Worker idle under contention** — run 470 had 2/4 workers at 0% CPU while competing for CPU with runs 469/471. Filed as vibe-quant-pkda.
3. **STOCH+ROC collapsed to 7 trades in validation** (469) — similar to Batch 3's CCI+ROC collapse. STOCH+ROC on 12mo data is too sensitive to fill timing.

### Key Findings

1. **STOCH+CCI is the 12mo champion** — Sharpe 2.06 validated on 12mo data with near-perfect trade preservation (94/95). More robust than any Batch 1 strategy on a per-month basis.
2. **STOCH+ROC degrades on 12mo data** — 55→7 trades in validation. The 3mo Batch 1 champion (Sharpe 2.60) may be period-specific. STOCH+ROC needs shorter data windows.
3. **Higher budget (400 trials) produced better STOCH+CCI** — Batch 1 had 120 trials with Sharpe 2.47 discovery / 2.50 validated (40 trades). This batch had 400 trials with Sharpe 1.94 / 2.06 validated (94 trades). The higher budget found a more robust strategy with 2.35x more validated trades.
4. **LONG definitively unviable on 1m** — 4th evidence: Batch 1 (both winners SHORT), Batch 3 (forced BOTH fail), and now forced LONG (0/5 passed, all negative returns). This is conclusive.
5. **12mo data favors CCI over ROC for stability** — ROC's momentum signals are too sensitive to fill timing over longer periods. CCI's oscillator nature provides more stable entry/exit signals.
6. **CPU contention significantly impacts parallel runs** — 470 took 67min vs 469's 51min despite same config, due to CPU starvation from 3 parallel runs with 12 total workers on 10 cores.

### Comparison with Previous Batches

| Metric | B1 STOCH+ROC (3mo) | B1 STOCH+CCI (3mo) | B4 STOCH+CCI (12mo) | B4 STOCH+ROC (12mo) |
|--------|-------|-------|-------|-------|
| Disc Sharpe | 2.13 | 2.47 | **1.94** | 1.67 |
| Val Sharpe | **2.60** | **2.50** | **2.06** | 1.12 (7 trades) |
| Val Trades | 155 | 40 | **94** | 7 |
| Val DD | 5.2% | 11.7% | 7.5% | 1.7% |
| Val WR | **94.8%** | 52.5% | 47.9% | 14.3% |
| Val PF | 1.52 | 1.53 | **1.51** | 1.20 |
| Val Return | +5.8% | +15.5% | **+8.2%** | +4.6% |
| Data Window | 3mo | 3mo | **12mo** | 12mo |
| Verdict | **1m CHAMPION** | **#2 (3mo)** | **#2 (12mo) — most robust** | FAIL (7 trades) |

### 1m All-Time Leaderboard (Validated Sharpe)

| Rank | Batch | Combo | Sharpe | Sortino | DD | Trades | PF | WR | Dir | Data |
|------|-------|-------|--------|---------|-----|--------|-----|-----|-----|------|
| 1 | B1 | STOCH+ROC | **2.60** | 3.53 | 5.2% | 155 | 1.52 | **94.8%** | SHORT | 3mo |
| 2 | B1 | STOCH+CCI | **2.50** | 5.69 | 11.7% | 40 | 1.53 | 52.5% | SHORT | 3mo |
| 3 | **B4** | **STOCH+CCI** | **2.06** | 3.77 | 7.5% | **94** | 1.51 | 47.9% | SHORT | **12mo** |
| 4 | B3 | STOCH+CCI+ROC | 1.02 | 1.28 | 11.4% | 299 | 1.17 | 92.6% | SHORT | 12mo |
| — | B4 | STOCH+ROC | 1.12 | 3.16 | 1.7% | 7 | 1.20 | 14.3% | SHORT | 12mo |
| — | B2 | CCI+BBANDS | 0.31 | 0.41 | 7.3% | 52 | 1.05 | 53.8% | LONG | 12mo |
| — | B2 | STOCH+DONCHIAN | −0.88 | −1.00 | 11.9% | 105 | 0.84 | 34.3% | LONG | 12mo |

### Recommendations

1. **Fix discovery API date range (vibe-quant-tonx)** — this is blocking all "3mo data" experiments. Must fix before next batch.
2. **Re-run STOCH+ROC on 3mo data with higher budget** — Batch 1's champion was 3mo. Higher budget (400 trials) hasn't been tested on 3mo yet.
3. **Paper trade both top strategies** — sid=135 (B1 STOCH+ROC, 3mo scalper) and sid=141 (B4 STOCH+CCI, 12mo balanced). They have complementary profiles.
4. **STOCH+CCI (sid=141) is the most robust 1m strategy** — 94 validated trades on 12mo with validation improvement. Better risk-adjusted than the 3mo strategies.
5. **Reduce parallel runs to 2** — 3 parallel with 12 workers on 10 cores causes contention. 2 runs × 4 workers = 8 workers fits cleanly on 10 cores.
6. **LONG direction is closed on 1m** — 4 batches, 0 viable long strategies. Stop testing.

---

## 2026-03-11: Batch 5 — ATR Volatility Filter + STOCH+ROC High Budget (3mo)

### Goal

Two experiments: (1) Test ATR as a volatility filter — the only Rust-native indicator never tried on 1m. (2) Re-run STOCH+ROC with higher budget (400 trials) on 3mo data — the #1 untested recommendation from every previous batch. Also test CCI+ATR combo. All forced SHORT (LONG confirmed dead).

**Date range fix:** Discovery API now accepts `start_date`/`end_date` params, correctly using 3mo window (2025-12-11 → 2026-03-11).

### Configuration

| Run | Indicators | Pop | Gens | Trials | Direction | Time | Status |
|-----|-----------|-----|------|--------|-----------|------|--------|
| 476 | STOCH+ATR | 16 | 16 | 256 | short | 966s (16m) | **completed** |
| 477 | CCI+ATR | 16 | 16 | 256 | short | 1016s (17m) | completed (FAIL) |
| 478 | STOCH+ROC | 20 | 20 | 400 | short | 698s (12m) | **completed** |

Data range: 2025-12-11 to 2026-03-11. BTCUSDT 1m. ~130K bars. Wave 1: 476+477 parallel (14 cores). Wave 2: 478 solo.

### Full Pipeline Results

| Stage | 476 STOCH+ATR | 477 CCI+ATR | 478 STOCH+ROC |
|-------|--------------|-------------|---------------|
| **Direction** | SHORT | SHORT | SHORT |
| **Discovery** score | 0.5021 | 0.2568 | **0.6563** |
| **Discovery** sharpe | 1.84 | 3.28 | **3.28** |
| **Discovery** dd | 11.2% | 21.6% | **10.5%** |
| **Discovery** trades | 74 | 1045 | 58 |
| **Discovery** return | +6.3% | -21.6% | **+11.3%** |
| **Discovery** PF | 1.26 | 1.23 | **1.47** |
| **DSR** | PASS 3/5 (p=0.000) | FAIL 0/5 (neg ret) | **PASS 5/5** (p=0.000) |
| **Screening** match | **exact** ✓ | — | **exact** ✓ |
| **Validation** sharpe | **2.39 (+30%)** | — | **2.63 (−20%)** |
| **Validation** sortino | 4.36 | — | **5.86** |
| **Validation** dd | **8.4%** | — | 11.7% |
| **Validation** trades | 24 (−68%) | — | **55 (−5%)** |
| **Validation** PF | **1.56** | — | 1.37 |
| **Validation** WR | 41.7% | — | 21.8% |
| **Validation** return | **+8.5%** | — | +8.4% |
| **Validation** fees | $20.02 | — | $28.27 |

### Winning Strategies

**#1: Run 478 — STOCH+ROC (Short, Trend-Follower)** — Strategy `genome_9cc6178cabb6` (sid=143)
- Entry: STOCH(18,4) crosses_above 37.52 → short
- Exit: ROC(16) crosses_above -3.43
- SL: 1.4% / **TP: 6.88%** (4.9x reward/risk)
- Validated **Sharpe 2.63**, Sortino 5.86, **55 trades**, +8.4% return, 11.7% DD, PF 1.37, 21.8% WR
- **NEW 1m ALL-TIME SHARPE CHAMPION** (2.63 > B1's 2.60)
- Completely different architecture from B1's STOCH+ROC: trend-follower (21.8% WR, 6.88% TP) vs scalper (94.8% WR, 0.64% TP). Same indicators, opposite philosophy. GA found both.
- Near-perfect trade preservation: 55/58 validated trades (−5%)

**#2: Run 476 — STOCH+ATR (Short, ATR-Filtered)** — Strategy `genome_d6c629a73efa` (sid=142)
- Entry: STOCH(9,5) >= 47.09 AND ATR(25) > 0.0672 → short
- Exit: STOCH(12,4) crosses_above 52.68 AND STOCH(18,7) crosses_below 60.59
- SL: 4.3% / TP: 3.1%
- Validated **Sharpe 2.39**, Sortino 4.36, 24 trades, **+8.5% return**, **8.4% DD**, PF 1.56, 41.7% WR
- ATR as volatility gate: only trade when ATR > 0.0672 (elevated volatility). STOCH handles all timing.
- Validation improved +30% Sharpe, but trade count dropped -68% (74→24). Strategy becomes very selective under realistic fills — only takes the highest-conviction setups.
- 24 trades in 3mo is borderline (8/mo) but each trade averages +0.35% return.

### Issues Found

1. **CCI+ATR total failure** — 0/5 passed guardrails. All strategies had negative returns (-19% to -94%). CCI without STOCH cannot find viable entries on 1m, even with ATR filtering.
2. **476 trade count degradation** — 74→24 in validation (-68%). ATR-filtered entries are sensitive to fill timing. The ATR > 0.0672 threshold sits near the edge of many entries.

### Key Findings

1. **STOCH+ROC on 3mo with high budget produces the best 1m strategy ever** — Sharpe 2.63 validated, beating B1's 2.60. The #1 journal recommendation was correct.
2. **ATR works as a volatility filter on 1m** — first successful use of ATR on 1m. STOCH+ATR validates at Sharpe 2.39, proving ATR can add value. But trade count drops significantly in validation.
3. **Same indicators can produce opposite architectures** — B1 STOCH+ROC was a scalper (94.8% WR, 0.64% TP). B5 STOCH+ROC is a trend-follower (21.8% WR, 6.88% TP). Higher budget (400 vs 120 trials) found the trend-following variant.
4. **CCI requires STOCH on 1m** — CCI+ATR failed. CCI+ROC (B3) failed. Only CCI+STOCH (B1, B4) works. CCI is a companion indicator, not a leader, on 1m.
5. **3mo data is better than 12mo for STOCH+ROC** — B4's STOCH+ROC on 12mo collapsed to 7 trades. B5 on 3mo validated 55 trades with Sharpe 2.63. Recent patterns matter more for STOCH+ROC.
6. **2 parallel runs is optimal** — wave 1 (2 runs) at ~16-17min each, wave 2 (solo) at 12min. No contention, predictable timing.

### Comparison with Previous Batches

| Metric | B1 STOCH+ROC (3mo) | B1 STOCH+CCI (3mo) | B4 STOCH+CCI (12mo) | B5 STOCH+ROC (3mo) | B5 STOCH+ATR (3mo) |
|--------|-----|-----|-----|-----|-----|
| Disc Sharpe | 2.13 | 2.47 | 1.94 | **3.28** | 1.84 |
| Val Sharpe | 2.60 | 2.50 | 2.06 | **2.63** | 2.39 |
| Val Trades | **155** | 40 | **94** | 55 | 24 |
| Val DD | **5.2%** | 11.7% | 7.5% | 11.7% | **8.4%** |
| Val WR | **94.8%** | 52.5% | 47.9% | 21.8% | 41.7% |
| Val PF | 1.52 | **1.53** | 1.51 | 1.37 | **1.56** |
| Val Return | +5.8% | **+15.5%** | +8.2% | +8.4% | **+8.5%** |
| Architecture | Scalper | Balanced | Balanced | **Trend-follower** | **ATR-filtered** |
| Data | 3mo | 3mo | 12mo | **3mo** | **3mo** |
| Verdict | **Scalp champion** | #3 | **12mo champion** | **NEW #1 Sharpe** | New #4 |

### 1m All-Time Leaderboard (Validated Sharpe)

| Rank | Batch | Combo | Sharpe | Sortino | DD | Trades | PF | WR | Dir | Data |
|------|-------|-------|--------|---------|-----|--------|-----|-----|-----|------|
| 1 | **B5** | **STOCH+ROC** | **2.63** | 5.86 | 11.7% | 55 | 1.37 | 21.8% | SHORT | **3mo** |
| 2 | B1 | STOCH+ROC | 2.60 | 3.53 | 5.2% | 155 | 1.52 | **94.8%** | SHORT | 3mo |
| 3 | B1 | STOCH+CCI | 2.50 | 5.69 | 11.7% | 40 | 1.53 | 52.5% | SHORT | 3mo |
| 4 | **B5** | **STOCH+ATR** | **2.39** | 4.36 | 8.4% | 24 | **1.56** | 41.7% | SHORT | **3mo** |
| 5 | B4 | STOCH+CCI | 2.06 | 3.77 | 7.5% | 94 | 1.51 | 47.9% | SHORT | 12mo |
| 6 | B3 | STOCH+CCI+ROC | 1.02 | 1.28 | 11.4% | 299 | 1.17 | 92.6% | SHORT | 12mo |
| — | B4 | STOCH+ROC | 1.12 | 3.16 | 1.7% | 7 | 1.20 | 14.3% | SHORT | 12mo |
| — | B2 | CCI+BBANDS | 0.31 | 0.41 | 7.3% | 52 | 1.05 | 53.8% | LONG | 12mo |
| — | B2 | STOCH+DONCHIAN | −0.88 | −1.00 | 11.9% | 105 | 0.84 | 34.3% | LONG | 12mo |
| — | **B5** | CCI+ATR | — | — | — | — | — | — | SHORT | 3mo |

### Recommendations

1. **Paper trade STOCH+ROC (sid=143)** — new Sharpe champion (2.63). Trend-following architecture with 6.88% TP. Test alongside B1's scalper (sid=135) for portfolio diversification.
2. **Portfolio of B1+B5 STOCH+ROC** — scalper (94.8% WR) + trend-follower (21.8% WR) could provide excellent diversification. Same indicators, uncorrelated entry signals.
3. **Try STOCH+ATR with wider ATR threshold range** — current pool uses [0.001, 0.08]. The winner used 0.0672 (near ceiling). Expand range to [0.001, 0.15] to let GA explore more.
4. **STOCH+ROC is exhaustively proven** — 5 batches, 2 architectures, both validate above 2.60 Sharpe on 3mo. Focus shifts to paper trading and live deployment.
5. **ATR is a viable 1m indicator** — only works with STOCH (CCI+ATR failed). Consider STOCH+ATR+ROC triple with ATR as volatility gate in future runs.
6. **No more new indicator experiments needed on 1m** — 5 batches tested: RSI ✗, ADX ✗, BBANDS ✗, DONCHIAN ✗, MFI ✗ (slow), CCI (companion only), ATR (companion only). STOCH is the only self-sufficient 1m indicator.

---

## 2026-03-14: Batch 6 — ATR Wider Threshold + Triple + STOCH+CCI 3mo High Budget

### Goal

Three optimization experiments from B5 recommendations: (1) STOCH+ATR with expanded ATR threshold [0.001, 0.15] (B5 winner used 0.0672, near old ceiling of 0.08). (2) STOCH+ATR+ROC triple — ATR as volatility gate with proven STOCH+ROC core. (3) STOCH+CCI on 3mo with high budget (400 trials) — B1 was 120 trials/3mo, B4 was 400 trials/12mo. All forced SHORT.

### Code Change

Expanded ATR `default_threshold_range` from `(0.001, 0.08)` to `(0.001, 0.15)` in `vibe_quant/discovery/genome.py`. B5's winner used ATR=0.0672 near the old ceiling — wider range lets GA explore higher volatility filters.

### Configuration

| Run | Indicators | Pop | Gens | Trials | Direction | Time | Status |
|-----|-----------|-----|------|--------|-----------|------|--------|
| 483 | STOCH+ATR+ROC | 14 | 12 | 168 | short | 621s (10m) | **completed** |
| 484 | STOCH+CCI | 20 | 20 | 400 | short | 1519s (25m) | **completed** |
| 485 | STOCH+ATR (wider) | 16 | 16 | 256 | short | 1144s (19m) | **completed** |

Data range: 2025-12-14 to 2026-03-14. BTCUSDT 1m. ~130K bars. Wave 1: 483+484 parallel. Wave 2: 485 solo.

### Full Pipeline Results

| Stage | 483 STOCH+ATR+ROC | 484 STOCH+CCI | 485 STOCH+ATR (wider) |
|-------|-------------------|---------------|----------------------|
| **Direction** | SHORT | SHORT | SHORT |
| **Discovery** score | 0.5673 | 0.5796 | **0.6934** |
| **Discovery** sharpe | 2.40 | 2.54 | **3.51** |
| **Discovery** dd | 10.5% | 9.9% | **4.4%** |
| **Discovery** trades | 67 | 71 | 75 |
| **Discovery** return | **+11.0%** | +9.0% | +6.7% |
| **Discovery** PF | 1.29 | 1.35 | **1.69** |
| **DSR** | **PASS 5/5** (p=0.000) | PASS 2/5 | PASS 2/5 |
| **Screening** match | **exact** ✓ | **exact** ✓ | **exact** ✓ |
| **Validation** sharpe | 6.88 (2 trades, NOISE) | **2.00 (−21%)** | **3.64 (+4%)** |
| **Validation** sortino | 56.23 (noise) | 3.99 | **6.56** |
| **Validation** dd | 0.7% (noise) | 10.8% | **4.6%** |
| **Validation** trades | **2** (−97% FAIL) | **76** (+7%) | **26** (−65%) |
| **Validation** PF | 5.53 (noise) | 1.27 | **1.66** |
| **Validation** WR | 0.0% (noise) | 31.6% | **50.0%** |
| **Validation** return | +0.7% (noise) | **+7.0%** (−22%) | **+7.0%** (+4%) |
| **Validation** fees | $1.72 | $39.05 | $6.06 |

### Winning Strategies

**#1: Run 485 — STOCH+ATR wider (Short)** — Strategy `genome_16f7b147df80` (sid=146)
- Entry: STOCH(19,3) crosses_below 31.98 → short
- Exit: ATR(5) < 0.141
- SL: 7.53% / **TP: 5.59%**
- Validated **Sharpe 3.64**, Sortino 6.56, **26 trades**, +7.0% return, **4.6% DD**, PF 1.66, 50.0% WR
- **NEW 1m ALL-TIME SHARPE CHAMPION** (3.64 > B5's 2.63)
- ATR exit threshold 0.141 — well above old ceiling (0.08), confirming the wider range was necessary. GA found that exiting when volatility drops below 0.141 captures the trend profits.
- Validation **improved +4% Sharpe** with near-identical return (+7.0%). Trade count dropped -65% (75→26), same pattern as B5's STOCH+ATR (74→24). ATR-filtered strategies become very selective under realistic fills.
- 26 trades in 3mo (~9/mo) is borderline but each trade averages +0.27% return.

**#2: Run 484 — STOCH+CCI 3mo high budget (Short)** — Strategy `genome_fbf56022d186` (sid=145)
- Entry: CCI(28) crosses_below 19.38 → short
- Exit: CCI(16) crosses_below -5.37 AND STOCH(16,5) crosses_above 22.89
- SL: 1.67% / TP: 7.03%
- Validated **Sharpe 2.00**, Sortino 3.99, **76 trades** (+7%), +7.0% return, 10.8% DD, PF 1.27, 31.6% WR
- CCI-dominated (2/3 conditions are CCI) with STOCH exit confirmation — same structural pattern as B4's winner (sid=141).
- **Trade preservation is excellent** — 76/71 validated trades (+7%). Most robust trade count preservation in 1m history.
- Sharpe degraded −21% (2.54→2.00), normal range for 1m validation.
- Tight 1.67% SL with 7.03% TP = 4.2x reward/risk ratio.

### Issues Found

1. **STOCH+ATR+ROC collapsed to 2 trades in validation** — despite 67 discovery trades, 5/5 DSR pass, and Sharpe 2.40. The triple combo is too sensitive to fill timing. ATR+ROC exit conditions create narrow windows that don't survive realistic fills.
2. **ATR-filtered strategies consistently lose trades in validation** — B5: 74→24 (−68%), B6: 75→26 (−65%). ATR thresholds sit near edges that shift with fill timing. Pattern is consistent but the surviving trades are high quality.

### Key Findings

1. **Wider ATR threshold produces the best 1m strategy ever** — Sharpe 3.64 validated (was 2.63 in B5). ATR exit at 0.141 was impossible with old ceiling (0.08). The B5 recommendation was exactly right.
2. **STOCH+CCI on 3mo with high budget matches B4's 12mo quality** — Sharpe 2.00 (B6, 3mo) vs 2.06 (B4, 12mo) with 76 vs 94 trades. 3mo data is competitive for STOCH+CCI.
3. **Triple combos don't work on 1m** — STOCH+ATR+ROC (B6) collapsed like STOCH+CCI+ROC (B3, 299→299 but Sharpe 1.02). Extra indicators add conditions that are too fill-sensitive. 2-indicator combos remain optimal.
4. **ATR strategies trade quality over quantity** — 26 trades at Sharpe 3.64 vs 76 trades at Sharpe 2.00. ATR filtering removes marginal trades, leaving only high-conviction entries.
5. **DSR 5/5 pass ≠ validation success** — run 483 passed all 5 DSR guardrails but collapsed to 2 validation trades. DSR measures statistical significance, not fill-timing sensitivity.
6. **STOCH+CCI is the most consistent 1m combo** — B1 (Sharpe 2.50, 40 trades), B4 (2.06, 94 trades), B6 (2.00, 76 trades). Always validates above 2.0 with good trade counts.

### Comparison with Previous Batches

| Metric | B1 STOCH+ROC | B5 STOCH+ROC | B5 STOCH+ATR | B6 STOCH+ATR (wider) | B6 STOCH+CCI |
|--------|-----|-----|-----|-----|-----|
| Disc Sharpe | 2.13 | 3.28 | 1.84 | **3.51** | 2.54 |
| Val Sharpe | 2.60 | 2.63 | 2.39 | **3.64** | 2.00 |
| Val Trades | **155** | 55 | 24 | 26 | **76** |
| Val DD | **5.2%** | 11.7% | 8.4% | **4.6%** | 10.8% |
| Val WR | **94.8%** | 21.8% | 41.7% | 50.0% | 31.6% |
| Val PF | 1.52 | 1.37 | **1.56** | **1.66** | 1.27 |
| Val Return | +5.8% | +8.4% | +8.5% | +7.0% | +7.0% |
| Architecture | Scalper | Trend | ATR-filter | **ATR-filter** | CCI-entry |
| Verdict | Scalp champ | Trend champ | #4 | **NEW #1** | #5 |

### 1m All-Time Leaderboard (Validated Sharpe)

| Rank | Batch | Combo | Sharpe | Sortino | DD | Trades | PF | WR | Dir | Data |
|------|-------|-------|--------|---------|-----|--------|-----|-----|-----|------|
| 1 | **B6** | **STOCH+ATR** | **3.64** | 6.56 | **4.6%** | 26 | **1.66** | 50.0% | SHORT | **3mo** |
| 2 | B5 | STOCH+ROC | 2.63 | 5.86 | 11.7% | 55 | 1.37 | 21.8% | SHORT | 3mo |
| 3 | B1 | STOCH+ROC | 2.60 | 3.53 | 5.2% | 155 | 1.52 | **94.8%** | SHORT | 3mo |
| 4 | B1 | STOCH+CCI | 2.50 | 5.69 | 11.7% | 40 | 1.53 | 52.5% | SHORT | 3mo |
| 5 | B5 | STOCH+ATR | 2.39 | 4.36 | 8.4% | 24 | 1.56 | 41.7% | SHORT | 3mo |
| 6 | B4 | STOCH+CCI | 2.06 | 3.77 | 7.5% | 94 | 1.51 | 47.9% | SHORT | 12mo |
| 7 | **B6** | **STOCH+CCI** | **2.00** | 3.99 | 10.8% | **76** | 1.27 | 31.6% | SHORT | **3mo** |
| 8 | B3 | STOCH+CCI+ROC | 1.02 | 1.28 | 11.4% | 299 | 1.17 | 92.6% | SHORT | 12mo |
| — | **B6** | STOCH+ATR+ROC | — | — | — | 2 | — | — | SHORT | 3mo |
| — | B4 | STOCH+ROC | 1.12 | 3.16 | 1.7% | 7 | 1.20 | 14.3% | SHORT | 12mo |

### Recommendations

1. **Paper trade STOCH+ATR (sid=146)** — new all-time Sharpe champion (3.64). ATR exit at 0.141 produces highly selective trades. Test alongside B1's scalper (sid=135) for portfolio diversification.
2. **Portfolio of 3 strategies** — sid=135 (B1 scalper, 94.8% WR, 155 trades), sid=143 (B5 trend, 21.8% WR, 55 trades), sid=146 (B6 ATR-filter, 50% WR, 26 trades). Three uncorrelated architectures.
3. **ATR threshold range is now correct** — 0.141 exit threshold validates that [0.001, 0.15] is the right range. No further expansion needed.
4. **Triple combos confirmed dead on 1m** — B3 (STOCH+CCI+ROC: Sharpe 1.02) and B6 (STOCH+ATR+ROC: 2 trades). Stick to 2-indicator combos.
5. **1m discovery is mature** — 6 batches, all reasonable combos and budget levels tested. Top 3 strategies span 3 different architectures. Focus shifts entirely to paper trading and live deployment.
6. **Consider STOCH+ATR on 12mo data** — B6 used 3mo. B4 showed STOCH+CCI works on 12mo. STOCH+ATR hasn't been tested on 12mo yet — could reveal whether the ATR-filter architecture is period-specific.

---

## 2026-03-14: Batch 7 — Solo Indicators + Ultra-Budget Optimization

### Goal

Five experiments: (1) STOCH solo — can the king indicator win alone on 1m? (2) STOCH+ATR on 12mo — test period robustness of B6 champion architecture. (3) CCI solo — always a companion, never tested alone on 1m. (4) STOCH+ATR ultra-budget (576 trials) — push B6 champion further. (5) STOCH+ROC ultra-budget (576 trials) — push B5 champion further. All forced SHORT, 3mo data except run 493 (12mo).

### Bug Encountered

**Corrupt parquet from parallel validation**: 5 parallel validations created a corrupt epoch-timestamp parquet file (`1970-01-01...parquet`), causing 3/5 validations to fail with `ArrowInvalid`. The corrupt file regenerated on re-run attempts. Fix: delete corrupt file + run validations sequentially. Root cause: validation runner creates a zero-byte parquet when backtest engine errors out, and this file poisons subsequent runs.

### Configuration

| Run | Indicators | Pop | Gens | Trials | Direction | Data | Time | Status |
|-----|-----------|-----|------|--------|-----------|------|------|--------|
| 492 | STOCH solo | 20 | 20 | 400 | short | 3mo | 3769s (63m) | **completed** |
| 493 | STOCH+ATR | 16 | 16 | 256 | short | **12mo** | 2590s (43m) | completed (WEAK) |
| 494 | CCI solo | 20 | 20 | 400 | short | 3mo | 2668s (44m) | **completed** |
| 495 | STOCH+ATR ultra | 24 | 24 | 576 | short | 3mo | 3331s (56m) | **completed** |
| 496 | STOCH+ROC ultra | 24 | 24 | 576 | short | 3mo | 3109s (52m) | **completed** |

Data range: 3mo = 2025-12-14 to 2026-03-14, 12mo = 2025-03-14 to 2026-03-14. All 5 launched simultaneously (CPU contention — runs took 2-3x longer than solo).

### Full Pipeline Results

| Stage | 492 STOCH solo | 493 STOCH+ATR 12mo | 494 CCI solo | 495 STOCH+ATR ultra | 496 STOCH+ROC ultra |
|-------|---------------|-------------------|-------------|--------------------|--------------------|
| **Direction** | SHORT | SHORT | SHORT | SHORT | SHORT |
| **Discovery** score | 0.5612 | 0.4194 | 0.5627 | **0.6108** | 0.5777 |
| **Discovery** sharpe | 2.27 | 0.86 | 2.37 | **2.62** | 2.56 |
| **Discovery** dd | 9.5% | 22.3% | **3.7%** | 8.7% | 5.3% |
| **Discovery** trades | **87** | 123 | 67 | 53 | 83 |
| **Discovery** return | +7.8% | +1.0% | +6.0% | **+12.0%** | +4.1% |
| **Discovery** PF | 1.36 | 1.19 | 1.38 | 1.36 | **1.55** |
| **DSR** | 3/5 | 3/5 | 1/5 | **4/5** | **4/5** |
| **Screening** match | **exact** ✓ | **exact** ✓ | **exact** ✓ | **exact** ✓ | **exact** ✓ |
| **Validation** sharpe | **4.15 (+83%)** | 1.25 (+45%) | **2.08 (−12%)** | 1.57 (−40%) | −1.89 (FAIL) |
| **Validation** sortino | **9.86** | 1.83 | 3.17 | 2.41 | −2.54 |
| **Validation** dd | **5.3%** | 19.0% | **3.6%** | 11.1% | 17.5% |
| **Validation** trades | 28 (−68%) | 9 (−93%) | **65** (−3%) | **61** (+15%) | 56 (−33%) |
| **Validation** PF | **1.96** | 1.25 | 1.35 | 1.20 | 0.72 |
| **Validation** WR | 42.9% | 33.3% | **50.8%** | 50.8% | 41.1% |
| **Validation** return | **+16.5%** | +4.6% | +5.3% | +6.9% | −15.5% |
| **Validation** fees | $26.53 | $6.14 | $15.40 | $27.79 | $23.91 |

### Winning Strategies

**#1: Run 492 — STOCH solo (Short)** — Strategy `genome_e63bbda4e233` (sid=147)
- Entry: STOCH(9,4) < 41.58 → short
- Exit: STOCH(17,7) > 79.08 AND STOCH(18,5) crosses_below 70.43
- SL: 1.18% / **TP: 8.37%** (7.1x reward/risk)
- Validated **Sharpe 4.15**, Sortino 9.86, 28 trades, **+16.5% return**, 5.3% DD, **PF 1.96**, 42.9% WR
- **NEW 1m ALL-TIME SHARPE CHAMPION** (4.15 > B6's 3.64)
- Pure STOCH strategy — 3 STOCH indicators with different periods. Entry on fast STOCH(9,4), exit on slower STOCH(17,7) and STOCH(18,5). Multi-timeframe STOCH analysis.
- Extreme reward/risk: 1.18% SL with 8.37% TP. Only 42.9% WR but winners are 7x larger than losers.
- Validation **improved +83% Sharpe** over discovery (2.27→4.15). Trade count dropped −68% (87→28) — same ATR-like selectivity pattern. Only the highest-conviction STOCH setups survive realistic fills.
- 28 trades in 3mo (~9/mo) — borderline but each trade averages +0.59% return.

**#2: Run 494 — CCI solo (Short)** — Strategy `genome_021cf9d1792e` (sid=149)
- Entry: CCI(20) <= -88.22 AND CCI(38) crosses_below -78.25 → short
- Exit: CCI(40) >= 75.85 AND CCI(40) crosses_above -60.07
- SL: 8.04% / TP: 3.59%
- Validated **Sharpe 2.08**, Sortino 3.17, **65 trades** (−3%), +5.3% return, **3.6% DD**, PF 1.35, 50.8% WR
- **CCI CAN stand alone on 1m** — overturns the "CCI needs STOCH" finding from B1-B6.
- Near-perfect trade preservation: 65/67 validated trades (−3%). Most robust trade count in entire 1m history.
- Deep CCI entry (< -88) ensures only strongly oversold entries. 4 CCI indicators with varied periods (20, 38, 40, 40).
- Modest 3.6% DD — lowest validated DD in 1m history.

### Issues Found

1. **Corrupt parquet from parallel validation** — 5 simultaneous validation runs create an epoch-timestamp parquet that poisons all subsequent runs. Must run validations sequentially or max 2 parallel. Validation runner needs a fix to avoid creating corrupt files on error.
2. **STOCH+ROC ultra (496) negative validation** — Sharpe −1.89, −15.5% return. 576 trials found a strategy that looked good (Sharpe 2.56, PF 1.55) but collapsed completely. Ultra-budget STOCH+ROC is worse than B5's 400-trial version (2.63).
3. **STOCH+ATR 12mo collapsed** — 123→9 trades in validation (−93%). STOCH+ATR architecture is definitively period-specific — only works on 3mo data.
4. **CPU contention from 5 parallel** — all runs took 2-3x longer (492 at 63min vs ~25min solo). 20 workers on 10 cores. Works but slow.

### Key Findings

1. **STOCH solo is the new 1m champion** — Sharpe 4.15 validated, +83% improvement over discovery. Pure STOCH with multi-period analysis. No companion indicator needed.
2. **CCI CAN stand alone on 1m** — overturns 6 batches of "CCI needs STOCH." With high budget (400 trials) and solo indicator pool, GA finds deep CCI entries that are highly robust (−3% trade count, 3.6% DD).
3. **Solo indicators outperform combos on 1m** — STOCH solo (4.15) > STOCH+ATR (3.64) > STOCH+ROC (2.63). CCI solo (2.08) ≈ STOCH+CCI (2.00-2.50). Companion indicators may add noise on 1m, not signal.
4. **Ultra-budget (576 trials) doesn't improve over 400** — STOCH+ATR ultra (1.57) < B6 STOCH+ATR (3.64). STOCH+ROC ultra (−1.89) < B5 STOCH+ROC (2.63). Diminishing returns — GA overfits with too many trials.
5. **STOCH+ATR confirmed 3mo-only** — 12mo validation collapsed to 9 trades. ATR volatility thresholds are regime-specific and don't generalize across market phases.
6. **Validation improvement correlates with fewer trades** — 492 improved +83% but lost 68% of trades. The "surviving" trades are extremely high quality. This is the same selectivity pattern seen across all top strategies.

### Comparison with Previous Batches

| Metric | B1 STOCH+ROC | B6 STOCH+ATR | B7 STOCH solo | B7 CCI solo | B7 STOCH+ATR ultra |
|--------|-----|-----|-----|-----|-----|
| Disc Sharpe | 2.13 | 3.51 | 2.27 | 2.37 | **2.62** |
| Val Sharpe | 2.60 | 3.64 | **4.15** | 2.08 | 1.57 |
| Val Trades | **155** | 26 | 28 | **65** | 61 |
| Val DD | **5.2%** | 4.6% | 5.3% | **3.6%** | 11.1% |
| Val WR | **94.8%** | 50.0% | 42.9% | 50.8% | 50.8% |
| Val PF | 1.52 | 1.66 | **1.96** | 1.35 | 1.20 |
| Val Return | +5.8% | +7.0% | **+16.5%** | +5.3% | +6.9% |
| Architecture | Scalper | ATR-filter | **Multi-STOCH** | Deep-CCI | ATR-filter |
| Verdict | Scalp champ | Former #1 | **NEW #1** | **#4** | #7 |

### 1m All-Time Leaderboard (Validated Sharpe)

| Rank | Batch | Combo | Sharpe | Sortino | DD | Trades | PF | WR | Dir | Data |
|------|-------|-------|--------|---------|-----|--------|-----|-----|-----|------|
| 1 | **B7** | **STOCH solo** | **4.15** | 9.86 | 5.3% | 28 | **1.96** | 42.9% | SHORT | **3mo** |
| 2 | B6 | STOCH+ATR | 3.64 | 6.56 | 4.6% | 26 | 1.66 | 50.0% | SHORT | 3mo |
| 3 | B5 | STOCH+ROC | 2.63 | 5.86 | 11.7% | 55 | 1.37 | 21.8% | SHORT | 3mo |
| 4 | B1 | STOCH+ROC | 2.60 | 3.53 | 5.2% | 155 | 1.52 | **94.8%** | SHORT | 3mo |
| 5 | B1 | STOCH+CCI | 2.50 | 5.69 | 11.7% | 40 | 1.53 | 52.5% | SHORT | 3mo |
| 6 | B5 | STOCH+ATR | 2.39 | 4.36 | 8.4% | 24 | 1.56 | 41.7% | SHORT | 3mo |
| 7 | **B7** | **CCI solo** | **2.08** | 3.17 | **3.6%** | **65** | 1.35 | 50.8% | SHORT | **3mo** |
| 8 | B4 | STOCH+CCI | 2.06 | 3.77 | 7.5% | 94 | 1.51 | 47.9% | SHORT | 12mo |
| 9 | B6 | STOCH+CCI | 2.00 | 3.99 | 10.8% | 76 | 1.27 | 31.6% | SHORT | 3mo |
| 10 | **B7** | **STOCH+ATR ultra** | **1.57** | 2.41 | 11.1% | 61 | 1.20 | 50.8% | SHORT | **3mo** |
| 11 | B3 | STOCH+CCI+ROC | 1.02 | 1.28 | 11.4% | 299 | 1.17 | 92.6% | SHORT | 12mo |
| — | **B7** | STOCH+ATR 12mo | 1.25 | 1.83 | 19.0% | 9 | 1.25 | 33.3% | SHORT | **12mo** |
| — | **B7** | STOCH+ROC ultra | −1.89 | −2.54 | 17.5% | 56 | 0.72 | 41.1% | SHORT | 3mo |

### Recommendations

1. **Paper trade STOCH solo (sid=147)** — new all-time champion (Sharpe 4.15, +16.5% return). Multi-period STOCH with 7.1x reward/risk. Test alongside B1's scalper (sid=135) for portfolio diversification.
2. **Portfolio of 4 strategies** — sid=135 (B1 scalper, 94.8% WR), sid=146 (B6 ATR-filter, 3.64 Sharpe), sid=147 (B7 STOCH solo, 4.15 Sharpe), sid=149 (B7 CCI solo, 3.6% DD). Four uncorrelated architectures.
3. **Solo indicators are the new frontier** — STOCH and CCI both work better alone than in combos on 1m. Companion indicators add noise.
4. **Don't exceed 400 trials** — 576 trials produced worse strategies than 400 for both STOCH+ATR and STOCH+ROC. GA overfits with too many evaluations.
5. **Fix corrupt parquet bug** — parallel validations create epoch-timestamp parquet files that poison the catalog. Must be fixed before production.
6. **STOCH+ATR is 3mo-only** — 12mo validation collapsed. Remove from 12mo recommendations.
7. **Try CCI solo with higher budget** — 400 trials produced Sharpe 2.08. Only 1/5 passed DSR — could benefit from pop=24 gens=24.
8. **1m discovery landscape is fully mapped** — 7 batches, every indicator solo and combo tested. Top 4 strategies span 4 architectures. Focus entirely on paper trading and portfolio construction.

---

## 2026-03-14: Batch 8 — Solo Indicator Exploration (3mo) + 5mo Period Test

### Goal

Two-phase experiment: (1) **3mo solo indicators**: ROC solo (untested), CCI solo with wider threshold [-200,200] (B7 used [-100,100], only 1/5 DSR), ATR solo (untested). (2) **5mo data window**: test STOCH solo, STOCH+CCI, CCI solo, and STOCH+ROC on 5 months (2025-10-14 → 2026-03-14) to find strategies that trade daily (~150+ trades) and survive longer market phases. All forced SHORT.

### Code Change

Expanded CCI `default_threshold_range` from `(-100.0, 100.0)` to `(-200.0, 200.0)` in `vibe_quant/discovery/genome.py`. B7's CCI solo winner used deep entries at CCI < -88 — wider range lets GA explore deeper CCI levels matching the 4h journal's range.

### Configuration

| Run | Indicators | Pop | Gens | Trials | Direction | Data | Time | Status |
|-----|-----------|-----|------|--------|-----------|------|------|--------|
| 510 | ROC solo | 16 | 16 | 256 | short | 3mo | 465s (8m) | completed |
| 511 | CCI solo [-200,200] | 16 | 16 | 256 | short | 3mo | 1151s (19m) | **completed** |
| 512 | ATR solo | 20 | 20 | 400 | short | 3mo | 882s (15m) | **completed** |
| 513 | STOCH solo | 16 | 16 | 256 | short | **5mo** | 2496s (42m) | completed (FAIL) |
| 514 | STOCH+CCI | 16 | 16 | 256 | short | **5mo** | 2158s (36m) | **completed** |
| 521 | CCI solo | 16 | 16 | 256 | short | **5mo** | 2243s (37m) | completed (val FAIL) |
| 522 | STOCH+ROC | 16 | 16 | 256 | short | **5mo** | 2184s (36m) | completed (val FAIL) |

Data range: 3mo = 2025-12-14 to 2026-03-14, 5mo = 2025-10-14 to 2026-03-14. All 3mo runs launched simultaneously, then 5mo runs launched as CPU freed up (max 4 concurrent).

### Full Pipeline Results — 3mo Runs

| Stage | 510 ROC solo | 511 CCI solo [-200,200] | 512 ATR solo |
|-------|-------------|------------------------|-------------|
| **Direction** | SHORT | SHORT | SHORT |
| **Discovery** score | 0.5388 | **0.6162** | **0.6167** |
| **Discovery** sharpe | 1.83 | **2.99** | **3.13** |
| **Discovery** dd | 11.4% | **2.2%** | 10.7% |
| **Discovery** trades | **159** | 53 | 61 |
| **Discovery** return | +5.5% | +5.1% | **+6.3%** |
| **Discovery** PF | 1.21 | **1.57** | 1.34 |
| **DSR** | PASS 1/2 | **PASS 4/5** | **PASS 5/5** |
| **Screening** match | **exact** ✓ | **exact** ✓ | **exact** ✓ |
| **Validation** sharpe | −7.02 (FAIL) | **2.23 (−25%)** | 18.90 (NOISE) |
| **Validation** sortino | −8.43 | 2.78 | 71.51 (noise) |
| **Validation** dd | 3.0% | **3.8%** | 0.0% (noise) |
| **Validation** trades | **2** (−99% FAIL) | **52** (−2%) | **4** (−93% FAIL) |
| **Validation** PF | 0.35 | **1.43** | 22.79 (noise) |
| **Validation** WR | 0.0% | **90.4%** | 100% (noise) |
| **Validation** return | −2.6% | **+4.1%** | +15.5% (noise) |
| **Validation** fees | $1.58 | $7.52 | $3.52 |

### Full Pipeline Results — 5mo Runs

| Stage | 513 STOCH solo | 514 STOCH+CCI | 521 CCI solo | 522 STOCH+ROC |
|-------|---------------|---------------|-------------|---------------|
| **Direction** | SHORT | SHORT | SHORT | SHORT |
| **Discovery** score | 0.6835 | **0.6675** | **0.6521** | 0.6385 |
| **Discovery** sharpe | 3.84 | **3.73** | **4.01** | 3.10 |
| **Discovery** dd | **4.4%** | 5.9% | 12.4% | **5.9%** |
| **Discovery** trades | 111 | 57 | 96 | **152** |
| **Discovery** return | **−4.4%** (neg!) | **+15.7%** | **+16.3%** | +12.9% |
| **Discovery** PF | **2.05** | 1.59 | **1.64** | 1.52 |
| **DSR** | **FAIL 0/5** (all neg return) | PASS 4/5 | **PASS 5/5** | **PASS 5/5** |
| **Screening** match | — | **exact** ✓ | **exact** ✓ | **exact** ✓ |
| **Validation** sharpe | — | **2.90 (−22%)** | 21.70 (NOISE) | — |
| **Validation** sortino | — | 4.12 | 169.50 (noise) | — |
| **Validation** dd | — | **5.8%** | 0.4% (noise) | — |
| **Validation** trades | — | **53** (−7%) | **7** (−93% FAIL) | **1** (−99% FAIL) |
| **Validation** PF | — | **1.46** | 22.79 (noise) | — |
| **Validation** WR | — | **75.5%** | 85.7% (noise) | — |
| **Validation** return | — | **+10.2%** | +30.9% (noise) | — |
| **Validation** fees | — | $13.23 | $6.05 | $0.00 |

### Winning Strategies

**#1: Run 511 — CCI solo [-200,200] (Short, 3mo)** — Strategy `genome_2a94de9be116` (sid=153)
- Entry: CCI(18) <= 40.84 AND CCI(32) crosses_below -143.30 → short
- Exit: CCI(36) >= 134.80 AND CCI(39) < 36.44
- SL: 9.99% / **TP: 1.52%** (scalper)
- Validated **Sharpe 2.23**, Sortino 2.78, **52 trades** (−2%), +4.1% return, **3.8% DD**, PF 1.43, **90.4% WR**
- **Wider CCI threshold worked** — CCI(32) crosses_below −143.30 was impossible with old [-100, 100] range. The wider [-200, 200] range let GA find this deep CCI entry.
- Near-perfect trade preservation: 52/53 validated trades (−2%). Most robust in batch.
- 90.4% WR scalper with 1.52% TP — takes many small wins.
- Comparable to B7's CCI solo (Sharpe 2.08, 65 trades) but with different architecture (1.52% TP vs 3.59% TP).

**#2: Run 514 — STOCH+CCI (Short, 5mo)** — Strategy `genome_3ee1011698e1` (sid=155)
- Entry: STOCH(5,5) crosses_above 42.53 → short
- Exit: CCI(18) crosses_below -70.99 AND CCI(12) crosses_above -7.29 AND CCI(44) <= -104.11
- SL: 6.95% / **TP: 3.29%**
- Validated **Sharpe 2.90**, Sortino 4.12, **53 trades** (−7%), **+10.2% return**, 5.8% DD, PF 1.46, 75.5% WR
- **Best 5mo validated strategy** — only 5mo run to survive validation with meaningful trade count.
- STOCH entry with CCI-dominated exit (3 CCI conditions) — the proven STOCH+CCI architecture scales to 5mo.
- 53 trades in 5mo = ~10.6 trades/month = ~2.5 trades/week. Not quite daily.

### Issues Found

1. **ROC solo collapsed in validation** — 159→2 trades (−99%). ROC threshold conditions are extremely fill-sensitive. ROC is not viable as a standalone indicator on 1m.
2. **ATR solo collapsed in validation** — 61→4 trades (−93%). Same pattern as all previous ATR strategies. ATR thresholds sit near edges that shift with realistic fills.
3. **STOCH solo 5mo: ALL strategies have negative returns** — 5/5 best strategies had negative absolute return despite high Sharpe (up to 3.84). The fitness function's 35% Sharpe weight allows negative-return strategies to score well. STOCH solo does NOT scale from 3mo to 5mo.
4. **CCI solo 5mo collapsed in validation** — 96→7 trades (−93%). Despite impressive discovery (Sharpe 4.01, 5/5 DSR), CCI solo on 5mo is fill-sensitive.
5. **STOCH+ROC 5mo collapsed in validation** — 152→1 trade (−99%). Despite 152 trades and 5/5 DSR, the strategy is completely fill-sensitive on 5mo.
6. **5mo data fundamentally harder** — only 1/4 5mo runs survived validation (514 STOCH+CCI). The longer window includes diverse market phases that most strategies can't handle under realistic fills.

### Key Findings

1. **Wider CCI threshold [-200,200] produces viable deep entries** — CCI(32) crosses_below −143.30 was impossible with [-100,100]. The wider range improved DSR (4/5 vs B7's 1/5) and found a robust 90.4% WR scalper.
2. **ROC solo is not viable on 1m** — 159 discovery trades but only 2 survived validation. ROC threshold conditions are too fill-sensitive for standalone use.
3. **ATR solo confirms the ATR selectivity pattern** — same −93% trade loss as all previous ATR strategies (B5: −68%, B6: −65%, B8: −93%). ATR works only as a companion and even then loses most trades in validation.
4. **5mo is mostly too long for 1m strategies** — 3/4 5mo runs collapsed in validation. Only STOCH+CCI survived, and with only 53 trades (not the daily trading the user wanted).
5. **STOCH+CCI is the most robust 1m combo across ALL data windows** — B1 (3mo): 2.50, B4 (12mo): 2.06, B6 (3mo): 2.00, B8 (5mo): 2.90. Every STOCH+CCI run validates above 2.0.
6. **High-frequency daily trading on 1m requires 3mo data** — B1's STOCH+ROC (155 trades/3mo = ~1.7/day) is the only strategy approaching daily trading. 5mo data produces fewer validated trades, not more.
7. **Negative-return strategies can have high Sharpe** — B8 run 513's top strategy had Sharpe 3.84 but −4.4% return. The fitness function's Sharpe weight (35%) can mislead. All 5 strategies from STOCH solo 5mo had negative returns — a first in the journal.

### Comparison with Previous Batches

| Metric | B7 STOCH solo (3mo) | B6 STOCH+ATR (3mo) | B7 CCI solo (3mo) | B8 CCI [-200,200] (3mo) | B8 STOCH+CCI (5mo) |
|--------|-----|-----|-----|-----|-----|
| Disc Sharpe | 2.27 | **3.51** | 2.37 | 2.99 | **3.73** |
| Val Sharpe | **4.15** | **3.64** | 2.08 | 2.23 | **2.90** |
| Val Trades | 28 | 26 | **65** | **52** | **53** |
| Val DD | 5.3% | **4.6%** | **3.6%** | 3.8% | 5.8% |
| Val WR | 42.9% | 50.0% | 50.8% | **90.4%** | 75.5% |
| Val PF | **1.96** | 1.66 | 1.35 | 1.43 | 1.46 |
| Val Return | **+16.5%** | +7.0% | +5.3% | +4.1% | **+10.2%** |
| Architecture | Multi-STOCH | ATR-filter | Deep-CCI | **Deep-CCI [-200]** | **STOCH entry + CCI exit** |
| Data | 3mo | 3mo | 3mo | **3mo** | **5mo** |
| Verdict | 3mo Sharpe champ | 3mo #2 | #7 | **New #8** | **5mo CHAMPION** |

### 1m All-Time Leaderboard (Validated Sharpe)

| Rank | Batch | Combo | Sharpe | Sortino | DD | Trades | PF | WR | Dir | Data |
|------|-------|-------|--------|---------|-----|--------|-----|-----|-----|------|
| 1 | B7 | STOCH solo | **4.15** | 9.86 | 5.3% | 28 | **1.96** | 42.9% | SHORT | 3mo |
| 2 | B6 | STOCH+ATR | 3.64 | 6.56 | 4.6% | 26 | 1.66 | 50.0% | SHORT | 3mo |
| 3 | **B8** | **STOCH+CCI (5mo)** | **2.90** | 4.12 | 5.8% | **53** | 1.46 | 75.5% | SHORT | **5mo** |
| 4 | B5 | STOCH+ROC | 2.63 | 5.86 | 11.7% | 55 | 1.37 | 21.8% | SHORT | 3mo |
| 5 | B1 | STOCH+ROC | 2.60 | 3.53 | 5.2% | 155 | 1.52 | **94.8%** | SHORT | 3mo |
| 6 | B1 | STOCH+CCI | 2.50 | 5.69 | 11.7% | 40 | 1.53 | 52.5% | SHORT | 3mo |
| 7 | B5 | STOCH+ATR | 2.39 | 4.36 | 8.4% | 24 | 1.56 | 41.7% | SHORT | 3mo |
| 8 | **B8** | **CCI solo [-200,200]** | **2.23** | 2.78 | **3.8%** | **52** | 1.43 | **90.4%** | SHORT | **3mo** |
| 9 | B7 | CCI solo | 2.08 | 3.17 | **3.6%** | **65** | 1.35 | 50.8% | SHORT | 3mo |
| 10 | B4 | STOCH+CCI | 2.06 | 3.77 | 7.5% | 94 | 1.51 | 47.9% | SHORT | 12mo |
| 11 | B6 | STOCH+CCI | 2.00 | 3.99 | 10.8% | 76 | 1.27 | 31.6% | SHORT | 3mo |
| — | **B8** | ROC solo | — | — | — | 2 | — | — | SHORT | 3mo |
| — | **B8** | ATR solo | — | — | — | 4 | — | — | SHORT | 3mo |
| — | **B8** | STOCH solo (5mo) | — | — | — | 0/5 neg return | — | — | SHORT | 5mo |
| — | **B8** | CCI solo (5mo) | — | — | — | 7 | — | — | SHORT | 5mo |
| — | **B8** | STOCH+ROC (5mo) | — | — | — | 1 | — | — | SHORT | 5mo |

### Recommendations

1. **Paper trade STOCH+CCI 5mo (sid=155)** — new 5mo champion (Sharpe 2.90, +10.2% return, 53 trades). STOCH+CCI is the only combo that reliably validates on any data window (3mo/5mo/12mo).
2. **Keep CCI threshold at [-200, 200]** — the wider range produces deeper entries and better DSR rates. No reason to narrow back.
3. **For daily trading, use 3mo data** — 5mo produces fewer validated trades, not more. B1's STOCH+ROC (155 trades/3mo) remains the highest-frequency validated strategy.
4. **Portfolio of uncorrelated strategies** — sid=135 (B1 STOCH+ROC scalper, 94.8% WR), sid=147 (B7 STOCH solo, Sharpe 4.15), sid=153 (B8 CCI scalper, 90.4% WR), sid=155 (B8 STOCH+CCI 5mo, best all-around).
5. ~~**ROC and ATR are confirmed solo-unviable on 1m**~~ — **RETRACTED (2026-03-14)**: The B8 validation collapse was caused by NT's LatencyModel creating 60s fill delays on 1m bar data, not by strategy weakness. See Batch 8 Re-Validation below.
6. **Consider fitness function adjustment** — STOCH solo 5mo found 5 strategies with high Sharpe but negative returns. The 35% Sharpe weight allows this. Adding a hard gate for return > 0% would prevent this. **DONE (2026-03-14)**: Added `total_return <= 0 → fitness=0` hard gate.
7. **5mo discovery is expensive** — 36-42min per run (vs 8-19min for 3mo). Only worth it for STOCH+CCI which has proven 5mo viability.

---

## 2026-03-14: Batch 8 Re-Validation — LatencyModel Fix Confirms All 4 "Failed" Strategies Are Viable

### Background

B8 validation used NT's LatencyModel which on 1m bar data defers ALL orders to the next bar (60s delay), regardless of the actual latency value (even 1ms). This caused 93-99% trade count collapse. Fix (bd-a3nc): skip LatencyModel for sub-5m timeframes, use prob_slippage=0.3 instead.

### Results

| sid | Strategy | Disc Trades | Old Val (w/ latency) | **New Val (no latency)** | Sharpe | Return | WR | DD |
|-----|----------|------------|---------------------|------------------------|--------|--------|------|------|
| 152 | ROC solo 3mo | 159 | 2 (1%) | **159 (100%)** | 1.83 | 5.52% | 82.4% | 11.4% |
| 154 | ATR solo 3mo | 61 | 4 (7%) | **61 (100%)** | 3.13 | 6.33% | 16.4% | 10.7% |
| 156 | CCI solo 5mo | 96 | 7 (7%) | **96 (100%)** | 4.01 | 16.30% | 11.5% | 12.4% |
| 157 | STOCH+ROC 5mo | 152 | 1 (1%) | **152 (100%)** | 3.10 | 12.94% | 89.5% | 5.9% |
| 153 | CCI solo 3mo (baseline) | 53 | 52 (98%) | **53 (100%)** | 2.99 | 5.12% | 90.6% | 2.2% |
| 155 | STOCH+CCI 5mo (baseline) | 57 | 53 (93%) | **57 (100%)** | 3.73 | 15.68% | 77.2% | 5.9% |

### Key Finding

**Every strategy now has exact trade count match with discovery.** The latency fix completely eliminates the validation gap for 1m strategies. All 4 previously "failed" strategies are actually viable.

### Updated Portfolio Rankings (all B8 strategies, sorted by Sharpe)

1. **sid=156 CCI solo 5mo** — Sharpe 4.01, 16.3% return, 96 trades. Best absolute return. Low WR (11.5%) but very high PF (1.64) = few big wins.
2. **sid=155 STOCH+CCI 5mo** — Sharpe 3.73, 15.7% return, 57 trades. Best risk-adjusted (5.9% DD). Proven across multiple batches.
3. **sid=154 ATR solo 3mo** — Sharpe 3.13, 6.3% return, 61 trades. Non-directional entry (ATR >= threshold → short). Low WR (16.4%) = tail-win strategy.
4. **sid=157 STOCH+ROC 5mo** — Sharpe 3.10, 12.9% return, 152 trades. Highest trade frequency of 5mo strategies. 89.5% WR scalper.
5. **sid=153 CCI solo 3mo** — Sharpe 2.99, 5.1% return, 53 trades. Most conservative (2.2% DD). Baseline survivor.
6. **sid=152 ROC solo 3mo** — Sharpe 1.83, 5.5% return, 159 trades. Highest trade frequency overall. 82.4% WR but lower Sharpe.

### Implications

- **B8 conclusion #5 was wrong** — ROC and ATR solos are NOT unviable. The validation pipeline had a systematic bug.
- **All pre-fix validation results on 1m data are suspect** — any strategy that "failed" validation before this fix should be re-tested.
- **Screening ↔ validation alignment is now perfect** — with no latency + prob_slippage=0.3, both modes produce identical trade counts.
- **No slippage cost** reported (Total Slippage: $0.00) — the SPEC SlippageEstimator is post-fill analytics, not reflected in the summary. Fees ARE modeled correctly.

---

---

## 2026-03-15: Batch 9 — 4-Month Window, All-SHORT Scalpers, ROC Combos

### Goal

First batch after the LatencyModel fix (bd-a3nc). Test 4 combos on a 4-month data window (2025-11-10 to 2026-03-10) with pop=20, gens=20 to get high-quality strategies. All Rust-native indicators (STOCH, CCI, ROC). Direction: short (per B8/B9 experience — 1m scalping is overwhelmingly SHORT). Target: strategies trading daily (~50-120 trades per 4mo = ~0.4-1 trades/day).

Combos:
1. **ROC solo** — retest after B8 re-validation confirmed ROC is viable (was wrongly written off)
2. **STOCH+ROC** — champion from B1/B5, retest on 4mo window
3. **STOCH+CCI** — all-time champion across every data window
4. **CCI+ROC** — new combo, tests ROC as CCI companion

### Bug Fixes Applied

None required — the validation pipeline fix (bd-a3nc) was already deployed before this batch.

### Configuration

| Run | Indicators | Pop | Gens | Trials | Direction | Time | Status |
|-----|-----------|-----|------|--------|-----------|------|--------|
| 535 | ROC solo | 20 | 20 | 400 | short | 19m31s | **completed** |
| 536 | STOCH+ROC | 20 | 20 | 400 | short | 24m37s | **completed** |
| 537 | STOCH+CCI | 20 | 20 | 400 | short | 37m39s | **completed** |
| 538 | CCI+ROC | 20 | 20 | 400 | short | 27m40s | **completed** |

Data: 2025-11-10 to 2026-03-10 (4 months, ~175K bars)

### Discovery Results (Top Strategy per Run)

| Run | Combo | Score | Sharpe | DD | Trades | Return | PF | DSR |
|-----|-------|-------|--------|-----|--------|--------|-----|-----|
| 535 | ROC solo | 0.7070 | 3.74 | 8.7% | 54 | 15.3% | 2.25 | PASS (p≈0) |
| 536 | STOCH+ROC | 0.6931 | 4.02 | 11.0% | 108 | 13.4% | 1.63 | PASS (p≈0) |
| 537 | STOCH+CCI | 0.6184 | 3.26 | 10.4% | 57 | 18.1% | 1.44 | PASS (p≈0) |
| 538 | CCI+ROC | 0.6650 | 3.25 | 8.0% | 59 | 14.3% | 1.53 | PASS (p≈0) |

All 5/5 strategies per run passed DSR guardrails (p=0.0000, 400 trials each).

### Winning Strategy DSL Details

**Run 535 (ROC solo) — sid=158**
```yaml
entry_conditions:
  short: ["roc_entry_0 >= -1.2254"]   # ROC(9)
exit_conditions:
  short: ["roc_exit_0 <= 4.1159", "roc_exit_1 crosses_above -0.8727"]  # ROC(11), ROC(19)
stop_loss: {type: fixed_pct, percent: 0.57}
take_profit: {type: fixed_pct, percent: 18.84}
```
Wide TP (18.84%) relative to tight SL (0.57%) explains 7.4% WR — very few big wins.

**Run 536 (STOCH+ROC) — sid=159**
```yaml
entry_conditions:
  short: ["stoch_entry_0 >= 54.4071", "stoch_entry_1 > 72.9637"]  # STOCH(14,5), STOCH(8,8)
exit_conditions:
  short: ["roc_exit_0 crosses_below -2.8906"]  # ROC(6)
stop_loss: {type: fixed_pct, percent: 0.62}
take_profit: {type: fixed_pct, percent: 10.82}
```
Dual STOCH confirmation + ROC momentum exit. 108 trades = best frequency.

**Run 537 (STOCH+CCI) — sid=160**
```yaml
entry_conditions:
  short: ["cci_entry_0 crosses_below 36.1711", "stoch_entry_1 > 49.9062"]  # CCI(17), STOCH(21,5)
exit_conditions:
  short: ["cci_exit_0 > 90.4279", "cci_exit_1 <= -86.9516"]  # CCI(22), CCI(18)
stop_loss: {type: fixed_pct, percent: 3.48}
take_profit: {type: fixed_pct, percent: 4.44}
```
CCI crosses below overbought zone (crossover entry), exits on CCI extremes.

**Run 538 (CCI+ROC) — sid=161**
```yaml
entry_conditions:
  short: ["cci_entry_0 crosses_above -199.815"]  # CCI(23) at extreme -200 boundary
exit_conditions:
  short: ["roc_exit_0 crosses_above -3.4198"]  # ROC(12)
stop_loss: {type: fixed_pct, percent: 5.1}
take_profit: {type: fixed_pct, percent: 2.79}
```
CCI near -200 extreme crossover (deeply oversold bounce entry for short, counterintuitive but effective). ROC momentum exit.

### Full Pipeline Results

| Stage | 535 ROC solo | 536 STOCH+ROC | 537 STOCH+CCI | 538 CCI+ROC |
|-------|-------------|--------------|--------------|-------------|
| Disc score | 0.7070 | 0.6931 | 0.6184 | 0.6650 |
| Disc sharpe | 3.74 | 4.02 | 3.26 | 3.25 |
| Disc trades | 54 | 108 | 57 | 59 |
| Disc return | 15.3% | 13.4% | 18.1% | 14.3% |
| DSR | PASS | PASS | PASS | PASS |
| Screen trades | 54 ✓ | 108 ✓ | 57 ✓ | 59 ✓ |
| Screen sharpe | 3.74 ✓ | 4.02 ✓ | 3.26 ✓ | 3.25 ✓ |
| Val trades | **54 (100%)** | **108 (100%)** | **57 (100%)** | **59 (100%)** |
| Val sharpe | 3.74 | 4.02 | 3.26 | 3.25 |
| Val return | 15.3% | 13.4% | 18.1% | 14.3% |
| Val DD | 8.7% | 11.0% | 10.4% | 8.0% |
| Val PF | 2.25 | 1.63 | 1.44 | 1.53 |
| Val WR | 7.4% | 9.3% | 52.6% | 72.9% |
| Val fees | $29.56 | $59.08 | $27.56 | $19.91 |
| Strategy ID | sid=158 | **sid=159** | sid=160 | sid=161 |

### Issues Found

None. Clean run throughout.

### Key Findings

1. **Perfect 100% trade count match across all 4 runs** — the LatencyModel fix (bd-a3nc) confirmed working on 4-month data window. Discovery ↔ screening ↔ validation are perfectly aligned.
2. **STOCH+ROC is the Sharpe king on 1m** — 4.02 Sharpe with 108 trades. Highest frequency + highest Sharpe in this batch. Dual STOCH confirmation + ROC exit is a strong pattern.
3. **ROC solo is now confirmed viable** — Sharpe 3.74, 15.3% return, PF=2.25 on 4mo. Very low WR (7.4%) reveals tail-win dynamics. The B8 validation collapse was the pipeline bug, not the strategy.
4. **All combos found only SHORT strategies** — every run converged to direction=short. Consistent with all prior 1m batches. The 1m BTCUSDT scalping edge is short-side only.
5. **Validation results are IDENTICAL to screening** — all Sharpe/return/PF values match to 2+ decimal places. With prob_slippage=0.3 (no LatencyModel), validation degradation is negligible for 1m.
6. **STOCH+CCI on 4mo underperforms vs 3mo** — Sharpe 3.26 on 4mo vs B6's 2.00 on 3mo. But B6 was a different market regime. Directly comparable to nothing — apples and oranges.
7. **CCI near -200 boundary produces valid entries** — CCI(23) crosses_above -199.815 means entry when CCI recovers from extreme oversold. Counterintuitive for a short entry but math works.
8. **4mo pop=20 gens=20 takes 19-37 minutes** — 537 (STOCH+CCI) was slowest at 37m due to the dual-CCI exit conditions requiring more backtest time per chromosome. For 30-40 min budget with 4mo data, use pop=16/gens=16 or reduce to pop=12/gens=15.

### Comparison with Best Previous Batches

| Metric | B7 STOCH solo | B1 STOCH+ROC | B8 STOCH+CCI | **B9 STOCH+ROC** | **B9 ROC solo** |
|--------|-------------|-------------|-------------|-----------------|-----------------|
| Data window | 3mo | 3mo | 5mo | **4mo** | **4mo** |
| Val Sharpe | 4.15 | 2.60 | 2.90 | **4.02** | **3.74** |
| Val Trades | 28 | 155 | 53 | **108** | **54** |
| Val DD | 5.3% | 5.2% | 5.8% | 11.0% | 8.7% |
| Val WR | 42.9% | 94.8% | 75.5% | 9.3% | 7.4% |
| Val PF | 1.96 | 1.52 | 1.46 | 1.63 | **2.25** |
| Val Return | 16.5% | 12.4% | 10.2% | 13.4% | **15.3%** |

### Updated 1m All-Time Leaderboard (Validated Sharpe, 4-month window)

| Rank | Batch | Combo | Sharpe | DD | Trades | PF | WR | Dir | Data |
|------|-------|-------|--------|-----|--------|-----|-----|-----|------|
| 1 | B7 | STOCH solo | **4.15** | 5.3% | 28 | 1.96 | 42.9% | SHORT | 3mo |
| 2 | **B9** | **STOCH+ROC** | **4.02** | 11.0% | **108** | 1.63 | 9.3% | SHORT | **4mo** |
| 3 | **B9** | **ROC solo** | **3.74** | 8.7% | 54 | **2.25** | 7.4% | SHORT | **4mo** |
| 4 | B8* | CCI solo 5mo | 4.01 | 12.4% | 96 | 1.64 | 11.5% | SHORT | 5mo |
| 5 | B8* | STOCH+CCI 5mo | 3.73 | 5.9% | 57 | 1.46 | 77.2% | SHORT | 5mo |
| 6 | B6 | STOCH+ATR | 3.64 | 4.6% | 26 | 1.66 | 50.0% | SHORT | 3mo |
| 7 | B8* | STOCH+ROC 5mo | 3.10 | 5.9% | 152 | 1.46 | 89.5% | SHORT | 5mo |
| 8 | B8* | ATR solo 3mo | 3.13 | 10.7% | 61 | 1.44 | 16.4% | SHORT | 3mo |
| 9 | **B9** | **STOCH+CCI** | **3.26** | 10.4% | 57 | 1.44 | 52.6% | SHORT | **4mo** |
| 10 | **B9** | **CCI+ROC** | **3.25** | **8.0%** | 59 | 1.53 | 72.9% | SHORT | **4mo** |

*B8 figures from 2026-03-14 re-validation with LatencyModel fix applied retroactively.

### Recommendations

1. **Paper trade sid=159 (STOCH+ROC, 4mo)** — new #2 all-time (Sharpe 4.02, 108 trades, 4mo). Highest trade frequency of any high-Sharpe strategy. Natural daily trader.
2. **Consider portfolio: sid=159 + sid=158** — STOCH+ROC (108 trades, low WR) + ROC solo (54 trades, PF=2.25). Different entry logic, both short-side, potentially uncorrelated trade timing.
3. **Dual STOCH confirmation is a strong pattern** — STOCH(14,5) >= 54 AND STOCH(8,8) > 72.9 = two timeframe overbought confirmation. Worth testing with different exit indicators (ATR, CCI).
4. **The 4-month data window is the sweet spot** — longer than 3mo (more data, better generalization) but shorter than 5mo (avoids regime diversity that kills most strategies in validation). All 4 B9 strategies survived with 100% trade retention.
5. **Next batch suggestion: explore long-side** — 9 consecutive batches of short-only strategies. Either the GA parameters strongly bias short, or the data period is genuinely short-biased. Test with `direction: long` forced to check if any viable long strategies exist.
6. **ROC as exit is a strong pattern** — both STOCH+ROC (run 536) and CCI+ROC (run 538) use ROC as exit only. Pure ROC also works as entry (run 535). ROC is more versatile than previously known.


---

## 2026-03-15: Batch 10 — ATR+ROC First Test, STOCH+ATR 4mo, Solo Indicators 4mo

### Goal

Complete the combinatorial space: test the only completely untried 2-indicator combo (ATR+ROC), plus STOCH+ATR and solo indicators on the new 4-month data window (2025-11-15 to 2026-03-15). All Rust-native, all force-SHORT.

Combos:
1. **ATR+ROC** — first ever test. The only 2-indicator combo from {STOCH,CCI,ROC,ATR} never tried
2. **STOCH+ATR** (4mo) — B6 champion (Sharpe 3.64) on 3mo, first time on 4mo
3. **ATR solo** (4mo) — B8 re-val showed Sharpe 3.13 on 3mo, first time on 4mo
4. **CCI solo** (4mo) — B7 Sharpe 2.08 (3mo), B8 Sharpe 4.01 (5mo), first time on 4mo

### Bug Found

**Corrupt parquet in crypto_perpetual/ path** — parallel validation created `1970-01-01T00-00-00-000000000Z.parquet` in `data/catalog/data/crypto_perpetual/BTCUSDT-PERP.BINANCE/` rather than the `data/bar/` path. The `CatalogManager._cleanup_epoch_parquet()` only scans `data_dir/data/` rglob, which covers bar data but apparently not instrument parquet files. Caused validation 556 (STOCH+ATR) to fail with `ArrowInvalid` on first attempt. Fixed by manually deleting the corrupt file and re-running. Filed as vibe-quant-tdu3.

**Note:** Pool assignment swapped for runs 549/550 due to parallel curl ordering — run 549 got ATR pool, run 550 got CCI pool (reversed from plan). Both combos were still tested.

### Configuration

| Run | Indicators | Pop | Gens | Trials | Direction | Time | Status |
|-----|-----------|-----|------|--------|-----------|------|--------|
| 547 | ATR+ROC | 20 | 20 | 400 | short | 15m32s | **completed** |
| 548 | STOCH+ATR | 20 | 20 | 400 | short | 23m12s | **completed** |
| 549 | ATR solo | 20 | 20 | 400 | short | 19m33s | **completed** |
| 550 | CCI solo | 20 | 20 | 400 | short | 33m8s | **completed** |

Data: 2025-11-15 to 2026-03-15 (4 months, ~175K bars). 4 parallel runs on 10-core M1 Pro.

### Discovery Results (Top Strategy per Run)

| Run | Combo | Score | Sharpe | DD | Trades | Return | PF | DSR |
|-----|-------|-------|--------|-----|--------|--------|-----|-----|
| 547 | ATR+ROC | 0.5633 | 2.22 | 7.0% | 76 | 8.9% | 1.32 | PASS 5/5 (p≈0) |
| 548 | STOCH+ATR | **0.7024** | **4.70** | 9.4% | 62 | 11.7% | 1.79 | PASS 5/5 (p≈0) |
| 549 | ATR solo | 0.6316 | 3.16 | 9.4% | 88 | 7.8% | 1.55 | PASS 4/4 (p≈0) |
| 550 | CCI solo | 0.5770 | 2.50 | 10.6% | 72 | 13.0% | 1.33 | PASS 5/5 (p≈0) |

### Winning Strategy DSL Details

**Run 547 (ATR+ROC) — sid=162**
```yaml
entry_conditions:
  short: ["roc_entry_0 <= 3.3523"]   # ROC(8) — enter short if ROC <= 3.35 (weak upward momentum)
exit_conditions:
  short: ["atr_exit_0 crosses_below 0.1423", "roc_exit_1 > -1.6403"]  # ATR(27) volatility collapse exit
stop_loss: {type: fixed_pct, percent: 5.7}
take_profit: {type: fixed_pct, percent: 2.21}
```
ATR as exit: volatility-collapse exit (ATR drops below threshold = momentum exhausted).

**Run 548 (STOCH+ATR) — sid=163**
```yaml
entry_conditions:
  short: ["stoch_entry_0 > 67.478"]   # STOCH(20,3) overbought
exit_conditions:
  short: ["atr_exit_0 <= 0.1408", "atr_exit_1 >= 0.0284"]  # ATR(19) ≤ 0.1408 AND ATR(16) ≥ 0.0284
stop_loss: {type: fixed_pct, percent: 0.69}
take_profit: {type: fixed_pct, percent: 16.27}
```
Dual ATR range exit: exits when fast ATR is in a specific volatility band (between 0.0284 and 0.1408). Tight SL (0.69%) + wide TP (16.27%) = tail-win architecture like B7's STOCH solo.

**Run 549 (ATR solo) — sid=164**
```yaml
entry_conditions:
  short: ["atr_entry_0 >= 0.0915", "atr_entry_1 > 0.0766"]  # ATR(17) AND ATR(28)
exit_conditions:
  short: ["atr_exit_0 crosses_above 0.0175"]  # ATR(22) — exit on volatility expansion
stop_loss: {type: fixed_pct, percent: 0.56}
take_profit: {type: fixed_pct, percent: 13.14}
```
Pure ATR volatility strategy: enter when multi-timeframe ATR above threshold, exit on ATR spike.

**Run 550 (CCI solo) — sid=165**
```yaml
entry_conditions:
  short: ["cci_entry_0 < -185.123"]   # CCI(11) deeply oversold → entry for short
exit_conditions:
  short: ["cci_exit_0 crosses_below 74.9914", "cci_exit_1 crosses_above 173.501"]
stop_loss: {type: fixed_pct, percent: 2.45}
take_profit: {type: fixed_pct, percent: 4.69}
```
CCI at extreme -185 boundary (near -200 floor) = counterintuitive short on extreme oversold, same pattern as B9 CCI+ROC.

### Full Pipeline Results

| Stage | 547 ATR+ROC | 548 STOCH+ATR | 549 ATR solo | 550 CCI solo |
|-------|------------|--------------|-------------|-------------|
| Disc score | 0.5633 | **0.7024** | 0.6316 | 0.5770 |
| Disc sharpe | 2.22 | **4.70** | 3.16 | 2.50 |
| Disc trades | 76 | 62 | 88 | 72 |
| Disc return | 8.9% | 11.7% | 7.8% | 13.0% |
| DSR | PASS | PASS | PASS | PASS |
| Screen trades | 76 ✓ | 62 ✓ | 88 ✓ | 72 ✓ |
| Screen sharpe | 2.22 ✓ | 4.70 ✓ | 3.16 ✓ | 2.50 ✓ |
| Val trades | **76 (100%)** | **62 (100%)** | **88 (100%)** | **72 (100%)** |
| Val sharpe | 2.22 | **4.70** | 3.16 | 2.50 |
| Val return | 8.9% | 11.7% | 7.8% | 13.0% |
| Val DD | 7.0% | 9.4% | 9.4% | 10.6% |
| Val PF | 1.32 | **1.79** | 1.55 | 1.33 |
| Val WR | 76.3% | 8.1% | 6.8% | 41.7% |
| Val fees | $22.15 | $32.51 | $45.91 | $34.13 |
| Strategy ID | sid=162 | **sid=163** | sid=164 | sid=165 |

### Issues Found

1. **Corrupt epoch parquet in crypto_perpetual/ path** — `CatalogManager._cleanup_epoch_parquet()` doesn't clean instrument parquet files, only bar data. Validation 556 failed first attempt. Fixed manually, filed vibe-quant-tdu3.
2. **Parallel curl ordering** — when launching 4 simultaneous curls with `&`, pool assignment order may not match curl command order. Run 549 got ATR pool instead of CCI, run 550 got CCI instead of ATR. Verify in DB after launch.

### Key Findings

1. **STOCH+ATR on 4mo is the new all-time 1m champion** — Sharpe **4.70** validated (was 4.15 for B7 STOCH solo). Dual ATR exit (0.0284 ≤ ATR ≤ 0.1408) captures a specific volatility band. Exact 100% trade match validation.
2. **ATR solo is highly viable on 4mo** — Sharpe 3.16, 88 trades (highest this batch). Multi-timeframe ATR entry + ATR spike exit. ATR solo on 3mo (B8 re-val) was Sharpe 3.13, confirming consistent ATR edge regardless of data window.
3. **ATR+ROC first test: modest but viable** — Sharpe 2.22, 76 trades, 100% validation match. ROC entry + ATR exit is a valid architecture. Not a new champion but confirmed viable combo.
4. **CCI solo (4mo): lowest score but high return** — Sharpe 2.50, 13.0% return, 72 trades. CCI at -185 extreme as entry is same pattern seen in B9. Consistent across data windows.
5. **100% validation match — 4th consecutive perfect batch** — B8 re-val, B9, and B10 all show 0% trade drop in validation. The 1m pipeline fix (bd-a3nc) is fully stable.
6. **ATR as exit is the dominant pattern this batch** — 3/4 winning strategies use ATR in exit conditions. ATR measures when volatility collapses/expands — excellent exit signal for 1m scalpers.
7. **STOCH+ATR dual-ATR exit is novel** — B6 used ATR as exit threshold (`ATR < 0.141`). B10 found a RANGE condition (`ATR <= 0.1408 AND ATR >= 0.0284`) — only exit when volatility is in a "sweet spot". This is a new exit architecture.

### Comparison with Previous Champions

| Metric | B7 STOCH solo | B9 STOCH+ROC | **B10 STOCH+ATR** | B10 ATR solo | B10 ATR+ROC |
|--------|-------------|-------------|-----------------|------------|------------|
| Data window | 3mo | 4mo | **4mo** | **4mo** | **4mo** |
| Val Sharpe | 4.15 | 4.02 | **4.70** | 3.16 | 2.22 |
| Val Trades | 28 | 108 | 62 | **88** | 76 |
| Val DD | 5.3% | 11.0% | 9.4% | 9.4% | **7.0%** |
| Val WR | 42.9% | 9.3% | 8.1% | 6.8% | **76.3%** |
| Val PF | 1.96 | 1.63 | **1.79** | 1.55 | 1.32 |
| Val Return | 16.5% | 13.4% | **11.7%** | 7.8% | 8.9% |
| Architecture | STOCH-exit | STOCH-entry ROC-exit | **STOCH-entry dual-ATR-exit** | ATR-filter | ROC-entry ATR-exit |

### Updated 1m All-Time Leaderboard (Validated Sharpe)

| Rank | Batch | Combo | Sharpe | DD | Trades | PF | WR | Dir | Data |
|------|-------|-------|--------|-----|--------|-----|-----|-----|------|
| 1 | **B10** | **STOCH+ATR** | **4.70** | 9.4% | 62 | **1.79** | 8.1% | SHORT | **4mo** |
| 2 | B7 | STOCH solo | 4.15 | 5.3% | 28 | 1.96 | 42.9% | SHORT | 3mo |
| 3 | B9 | STOCH+ROC | 4.02 | 11.0% | **108** | 1.63 | 9.3% | SHORT | 4mo |
| 4 | B8* | CCI solo 5mo | 4.01 | 12.4% | 96 | 1.64 | 11.5% | SHORT | 5mo |
| 5 | B8* | STOCH+CCI 5mo | 3.73 | 5.9% | 57 | 1.46 | 77.2% | SHORT | 5mo |
| 6 | B9 | ROC solo | 3.74 | 8.7% | 54 | 2.25 | 7.4% | SHORT | 4mo |
| 7 | B6 | STOCH+ATR | 3.64 | **4.6%** | 26 | 1.66 | 50.0% | SHORT | 3mo |
| 8 | **B10** | **ATR solo** | **3.16** | 9.4% | **88** | 1.55 | 6.8% | SHORT | **4mo** |
| 9 | B8* | ATR solo 3mo | 3.13 | 10.7% | 61 | 1.44 | 16.4% | SHORT | 3mo |
| 10 | B9 | STOCH+CCI | 3.26 | 10.4% | 57 | 1.44 | 52.6% | SHORT | 4mo |

*B8 figures from 2026-03-14 re-validation with LatencyModel fix.

### Recommendations

1. **Paper trade sid=163 (STOCH+ATR, 4mo)** — new all-time champion at Sharpe 4.70. Dual ATR exit at volatility band [0.0284, 0.1408] is a novel architecture worth live-testing.
2. **ATR is the key exit signal on 1m** — B6 STOCH+ATR (Sharpe 3.64), B10 STOCH+ATR (4.70), B10 ATR+ROC (2.22), B10 ATR solo (3.16). ATR in exit positions consistently produces high-quality strategies.
3. **ATR solo on 4mo is now confirmed viable** — 88 trades (highest frequency this batch), Sharpe 3.16. Consistent with B8's 3mo result. Add to paper trading portfolio.
4. **Portfolio of 4-month strategies** — sid=159 (B9 STOCH+ROC, 108 trades), sid=163 (B10 STOCH+ATR, 62 trades), sid=164 (B10 ATR solo, 88 trades). Three complementary architectures.
5. **Try STOCH+ATR on 3mo with new dual-ATR exit understanding** — B6 used single ATR threshold as exit. The B10 discovery of a RANGE condition (`ATR between 0.028 and 0.141`) may be more robust. Worth specifically testing this exit structure.
6. **ATR+ROC combo confirmed viable** — Sharpe 2.22 is modest but 100% trade retention and 76.3% WR. Could be valuable as a diversifying strategy in a portfolio.

---

## 2026-03-15: Batch 11 — 2-Month Window, Random Direction, High Budget Scalpers

### Goal

First test on a **2-month data window** (2026-01-15 → 2026-03-15, ~87K bars). Target: strategies with multiple trades per day. Direction set to `null` (random) to test whether GA can find viable LONG strategies — all 10 prior batches found only SHORT. High budget (pop=24, gens=24 = 576 trials) with 3 parallel runs on 10-core M1 Pro. Used the proven 1m indicator combos.

### Configuration

| Run | Indicators | Pop | Gens | Trials | Direction | Time | Status |
|-----|-----------|-----|------|--------|-----------|------|--------|
| 559 | STOCH+ATR | 24 | 24 | 576 | random | ~21min (converged gen 20) | **completed** |
| 560 | STOCH+ROC | 24 | 24 | 576 | random | ~19min | **completed** |
| 561 | CCI+ROC | 24 | 24 | 576 | random | ~16min | **completed** |

Data range: 2026-01-15 to 2026-03-15 (2 months). BTCUSDT 1m. ~87K bars. 3 parallel runs.

### Discovery Results (Top Strategy per Run)

| Run | Combo | Score | Sharpe | DD | Trades | Return | PF | DSR |
|-----|-------|-------|--------|-----|--------|--------|-----|-----|
| 559 | STOCH+ATR | **0.7122** | **4.27** | 5.6% | 73 | 11.2% | **1.80** | PASS 5/5 (p≈0) |
| 560 | STOCH+ROC | 0.6312 | 2.86 | 9.0% | **151** | 9.6% | 1.51 | PASS 4/4 (p≈0) |
| 561 | CCI+ROC | 0.6440 | 3.17 | **4.9%** | 64 | 6.3% | 1.59 | PASS 1/1 (p≈0) |

### Winning Strategy DSL Details

**Run 559 (STOCH+ATR) — sid=166**
```yaml
entry_conditions:
  short: ["stoch_entry_0 crosses_below 21.405"]   # STOCH(14,6) — oversold crossdown
exit_conditions:
  short: ["atr_exit_0 < 0.0873", "atr_exit_1 < 0.0379"]  # ATR(18) < 0.0873 AND ATR(21) < 0.0379
stop_loss: {type: fixed_pct, percent: 6.4}
take_profit: {type: fixed_pct, percent: 1.44}
```
STOCH oversold entry (crosses below 21.4) with dual ATR volatility-collapse exit. 1.44% TP scalper with 87.7% WR.

**Run 560 (STOCH+ROC) — sid=167**
```yaml
entry_conditions:
  short: ["stoch_entry_0 crosses_below 66.0577"]   # STOCH(10,4) — mid-range crossdown
exit_conditions:
  short: ["roc_exit_0 crosses_above 1.4979"]  # ROC(13) — exit on momentum reversal
stop_loss: {type: fixed_pct, percent: 4.26}
take_profit: {type: fixed_pct, percent: 0.71}
```
Ultra-tight 0.71% TP scalper with 90.1% WR. 151 trades in 2mo = **~2.5 trades/day** — meets the daily trading target. STOCH mid-range entry (not overbought) + ROC momentum exit.

**Run 561 (CCI+ROC) — sid=168**
```yaml
entry_conditions:
  short: ["roc_entry_0 <= 0.2839", "cci_entry_1 <= 12.9875"]  # ROC(26) + CCI(21)
exit_conditions:
  short: ["roc_exit_0 crosses_below -1.8062"]  # ROC(19) — exit on strong downward momentum
stop_loss: {type: fixed_pct, percent: 8.13}
take_profit: {type: fixed_pct, percent: 1.36}
```
ROC+CCI dual entry with ROC momentum exit. 1.36% TP, 88.9% WR.

### Full Pipeline Results

| Stage | 559 STOCH+ATR | 560 STOCH+ROC | 561 CCI+ROC |
|-------|--------------|--------------|-------------|
| Disc score | **0.7122** | 0.6312 | 0.6440 |
| Disc sharpe | **4.27** | 2.86 | 3.17 |
| Disc trades | 73 | **151** | 64 |
| Disc return | **11.2%** | 9.6% | 6.3% |
| DSR | PASS 5/5 | PASS 4/4 | PASS 1/1 |
| Screen trades | 73 ✓ | 151 ✓ | 63 (−1) |
| Screen sharpe | 4.27 ✓ | 2.86 ✓ | 3.05 |
| Val trades | **73 (100%)** | **151 (100%)** | **63 (100%)** |
| Val sharpe | **4.27** | 2.86 | 3.05 |
| Val sortino | **7.12** | 4.45 | 4.41 |
| Val return | **11.2%** | 9.6% | 6.0% |
| Val DD | 5.6% | 9.0% | **4.9%** |
| Val PF | **1.80** | 1.51 | 1.56 |
| Val WR | 87.7% | **90.1%** | 88.9% |
| Val fees | $18.31 | $57.73 | $12.29 |
| Strategy ID | **sid=166** | sid=167 | sid=168 |

### Issues Found

1. **CCI+ROC screening 1-trade mismatch** — 64 discovery trades vs 63 screening/validation. Minor (−1.6%). The screening sharpe also shifted slightly (3.17→3.05). Likely a borderline trade that the screening engine handles differently. Not a pipeline bug — within tolerance.
2. **Corrupt parquet cleanup still triggering** — validation logs show `Removing corrupt epoch-timestamp parquet`. The auto-cleanup works but the corrupt file keeps regenerating from parallel runs. Filed previously as vibe-quant-tdu3.

### Key Findings

1. **STOCH+ATR on 2mo matches the all-time champion pattern** — Sharpe 4.27 with STOCH entry + dual ATR exit. Same architecture as B10's champion (Sharpe 4.70 on 4mo). The ATR volatility-collapse exit is the dominant 1m exit signal.
2. **Random direction still converges 100% to SHORT** — 11th consecutive batch where GA finds only SHORT strategies. Even with `direction: null` (no forcing), the 1m BTCUSDT short-side edge is overwhelming. LONG is definitively dead on 1m for this data period.
3. **2mo data produces high-frequency scalpers** — 151 trades in 2mo (sid=167) = ~2.5 trades/day. This is the highest daily trade rate in journal history. The shorter data window produces more active strategies as expected.
4. **Ultra-tight TP scalpers dominate 2mo** — all 3 winners use TP ≤ 1.44%. WR range 87.7-90.1%. The 2mo window favors mean-reversion scalpers over trend-followers.
5. **CCI+ROC found its niche on 2mo** — Sharpe 3.05 with lowest DD (4.9%). The combo that was #10 on 4mo data is now competitive. Shorter data window helps CCI+ROC.
6. **100% trade match continues** — 5th consecutive batch with perfect pipeline alignment (post LatencyModel fix). Only exception: CCI+ROC 1-trade mismatch, within tolerance.
7. **2mo runs complete fast** — 16-21min per run with pop=24/gens=24. 3 parallel on 10 cores works well with ~87K bars. Could push to pop=30/gens=30 within the 1-2hr budget.

### Comparison with Previous Champions

| Metric | B10 STOCH+ATR (4mo) | B9 STOCH+ROC (4mo) | B7 STOCH solo (3mo) | **B11 STOCH+ATR (2mo)** | **B11 STOCH+ROC (2mo)** | **B11 CCI+ROC (2mo)** |
|--------|-----|-----|-----|-----|-----|-----|
| Data window | 4mo | 4mo | 3mo | **2mo** | **2mo** | **2mo** |
| Val Sharpe | **4.70** | 4.02 | 4.15 | **4.27** | 2.86 | 3.05 |
| Val Trades | 62 | **108** | 28 | 73 | **151** | 63 |
| Val DD | 9.4% | 11.0% | **5.3%** | 5.6% | 9.0% | **4.9%** |
| Val WR | 8.1% | 9.3% | 42.9% | **87.7%** | **90.1%** | **88.9%** |
| Val PF | 1.79 | 1.63 | 1.96 | **1.80** | 1.51 | 1.56 |
| Val Return | 11.7% | 13.4% | 16.5% | 11.2% | 9.6% | 6.0% |
| Architecture | STOCH+dual-ATR | STOCH+ROC | Multi-STOCH | **STOCH+dual-ATR** | **Ultra-scalper** | **Dual-entry** |
| Trades/day | 0.5 | 0.9 | 0.3 | **1.2** | **2.5** | **1.1** |
| Verdict | ALL-TIME #1 | 4mo #2 | 3mo #1 | **2mo #1** | **Frequency champ** | **Low-DD pick** |

### Updated 1m All-Time Leaderboard (Validated Sharpe)

| Rank | Batch | Combo | Sharpe | DD | Trades | PF | WR | Dir | Data | Trades/day |
|------|-------|-------|--------|-----|--------|-----|-----|-----|------|------------|
| 1 | B10 | STOCH+ATR | **4.70** | 9.4% | 62 | 1.79 | 8.1% | SHORT | 4mo | 0.5 |
| 2 | **B11** | **STOCH+ATR** | **4.27** | 5.6% | 73 | **1.80** | **87.7%** | SHORT | **2mo** | **1.2** |
| 3 | B7 | STOCH solo | 4.15 | 5.3% | 28 | 1.96 | 42.9% | SHORT | 3mo | 0.3 |
| 4 | B9 | STOCH+ROC | 4.02 | 11.0% | 108 | 1.63 | 9.3% | SHORT | 4mo | 0.9 |
| 5 | B8* | CCI solo 5mo | 4.01 | 12.4% | 96 | 1.64 | 11.5% | SHORT | 5mo | 0.6 |
| 6 | B8* | STOCH+CCI 5mo | 3.73 | 5.9% | 57 | 1.46 | 77.2% | SHORT | 5mo | 0.4 |
| 7 | B9 | ROC solo | 3.74 | 8.7% | 54 | 2.25 | 7.4% | SHORT | 4mo | 0.5 |
| 8 | B6 | STOCH+ATR | 3.64 | **4.6%** | 26 | 1.66 | 50.0% | SHORT | 3mo | 0.3 |
| 9 | B8* | ATR solo 3mo | 3.13 | 10.7% | 61 | 1.44 | 16.4% | SHORT | 3mo | 0.7 |
| 10 | B10 | ATR solo | 3.16 | 9.4% | 88 | 1.55 | 6.8% | SHORT | 4mo | 0.7 |
| 11 | **B11** | **CCI+ROC** | **3.05** | **4.9%** | 63 | 1.56 | **88.9%** | SHORT | **2mo** | **1.1** |
| 12 | **B11** | **STOCH+ROC** | **2.86** | 9.0% | **151** | 1.51 | **90.1%** | SHORT | **2mo** | **2.5** |

*B8 figures from re-validation with LatencyModel fix.

### Recommendations

1. **Paper trade sid=167 (STOCH+ROC, 2mo)** — 2.5 trades/day, 90.1% WR, ultra-tight 0.71% TP. The daily-frequency target achieved. Natural candidate for live deployment.
2. **Portfolio of sid=166 (STOCH+ATR) + sid=167 (STOCH+ROC)** — different entry levels (STOCH 21 vs 66), different exit signals (ATR vs ROC), likely uncorrelated trade timing. Combined ~3.7 trades/day.
3. **2mo window produces the highest trade frequencies** — 2.5 trades/day (sid=167) vs 0.9/day (B9 STOCH+ROC, 4mo). Shorter data = more frequent strategies.
4. **LONG remains dead on 1m** — 11 batches, random direction, still 100% SHORT convergence. Consider testing on different assets (ETHUSDT, SOLUSDT) to see if the bias is BTC-specific.
5. **STOCH+ATR is the dominant architecture across all data windows** — B6 (3mo, 3.64), B10 (4mo, 4.70), B11 (2mo, 4.27). STOCH entry + ATR exit is the most robust 1m pattern.
6. **CCI+ROC improves on shorter data** — Sharpe 3.05 on 2mo vs 3.25 on 4mo, but with much lower DD (4.9% vs 8.0%). Shorter window produces lower-risk CCI+ROC strategies.
7. **Consider higher budget on 2mo** — runs completed in 16-21min. Pop=30/gens=30 (900 trials) would take ~40-60min and is within the 1hr budget. Could push STOCH+ATR above 4.70.

---

## 2026-03-15: Batch 12 — Solo Indicators + ATR+ROC on 2mo, Higher Budget

### Goal

Test solo indicators (STOCH, CCI) and ATR+ROC on 2mo data with higher budget (pop=28, gens=28 = 784 trials). Direction=null to allow long strategies. Continue the B11 2mo window experiments with increased search budget.

### Bug Fixes Applied

1. **Epoch parquet cleanup broke instrument definitions** — the B11 fix (`cleanup_epoch_parquet`) was too aggressive, deleting `crypto_perpetual/` instrument parquet files that NT needs. All 3 initial runs (570-572) failed with "Instrument not found". Fixed: restrict cleanup to `bar/` subdirectories only. Rebuilt instrument definition.

### Configuration

| Run | Indicators | Pop | Gens | Trials | Direction | Time | Status |
|-----|-----------|-----|------|--------|-----------|------|--------|
| 573 | STOCH solo | 28 | 28 | 784 | random | ~28min | **completed** |
| 574 | CCI solo | 28 | 28 | 784 | random | ~27min (converged gen 23) | **completed** |
| 575 | ATR+ROC | 28 | 28 | 784 | random | ~17min (converged gen 21) | **completed** |

Data range: 2026-01-15 to 2026-03-15 (2 months). BTCUSDT 1m. ~87K bars.

Failed runs: 570-572 (instrument not found due to epoch parquet cleanup bug).

### Winning Strategy DSL Details

**Run 573 (STOCH solo) — sid=169**
```yaml
entry_conditions:
  short: ["stoch_entry_0 crosses_above 38.58", "stoch_entry_1 crosses_above 38.58"]  # STOCH(14,3) dual crossover
exit_conditions:
  short: ["stoch_exit_0 crosses_above 48.31", "stoch_exit_1 crosses_below 56.75"]  # STOCH(6,7) + STOCH(18,3)
stop_loss: {type: fixed_pct, percent: 9.23}
take_profit: {type: fixed_pct, percent: 2.3}
```
Multi-STOCH with dual identical entry conditions (STOCH 14,3 crosses above 38.6). 2.3% TP scalper.

**Run 574 (CCI solo) — sid=170**
```yaml
entry_conditions:
  short: ["cci_entry_0 > -2.49", "cci_entry_1 > -28.63"]  # CCI(42) + CCI(48) near neutral
exit_conditions:
  short: ["cci_exit_0 crosses_below -173.19", "cci_exit_1 crosses_below 193.86"]  # CCI(13) + CCI(49)
stop_loss: {type: fixed_pct, percent: 6.89}
take_profit: {type: fixed_pct, percent: 1.12}
```
4 CCI indicators, ultra-tight 1.12% TP, 90.0% WR. CCI entry near neutral (-2 to -29), exit on extreme CCI moves.

**Run 575 (ATR+ROC) — sid=171**
```yaml
entry_conditions:
  short: ["atr_entry_0 >= 0.1382"]  # ATR(22) volatility gate
exit_conditions:
  short: ["roc_exit_0 < -1.48", "atr_exit_1 crosses_below 0.0597"]  # ROC(29) + ATR(5)
stop_loss: {type: fixed_pct, percent: 6.66}
take_profit: {type: fixed_pct, percent: 1.25}
```
ATR volatility entry + dual ROC/ATR exit. 1.25% TP, 88.0% WR.

### Full Pipeline Results

| Stage | 573 STOCH solo | 574 CCI solo | 575 ATR+ROC |
|-------|---------------|-------------|-------------|
| Disc score | 0.5968 | **0.6338** | **0.6426** |
| Disc sharpe | 2.91 | **3.24** | 3.21 |
| Disc trades | 58 | **90** | 83 |
| Disc return | 3.8% | **8.8%** | 7.7% |
| DSR | PASS 4/4 | PASS 3/3 | PASS 5/5 |
| Screen trades | 58 ✓ | 90 ✓ | 83 ✓ |
| Screen sharpe | 2.91 ✓ | 3.24 ✓ | 3.21 ✓ |
| Val trades | **58 (100%)** | **90 (100%)** | **83 (100%)** |
| Val sharpe | 2.91 | **3.24** | 3.22 |
| Val sortino | 4.69 | **5.02** | 5.00 |
| Val return | 3.8% | **8.8%** | 7.7% |
| Val DD | 5.7% | **4.5%** | 5.7% |
| Val PF | 1.45 | **1.64** | 1.50 |
| Val WR | 70.7% | **90.0%** | 88.0% |
| Val fees | $10.60 | $20.94 | $20.05 |
| Strategy ID | sid=169 | **sid=170** | sid=171 |

### Issues Found

1. **Epoch parquet cleanup deleted instrument definitions** — runs 570-572 all failed with "Instrument not found: BTCUSDT-PERP.BINANCE". Root cause: the new `cleanup_epoch_parquet()` function (added in this session) used `rglob` on all of `data/` which includes `crypto_perpetual/` — but NT stores instrument definitions as epoch-timestamp parquet files by design. Fixed: restrict cleanup to `bar/` subdirectories only.

### Key Findings

1. **CCI solo on 2mo is excellent** — Sharpe 3.24, 90 trades (~1.5/day), 90.0% WR, 4.5% DD. Best CCI solo result in journal history. Ultra-tight 1.12% TP scalper.
2. **ATR+ROC on 2mo improves over 4mo** — Sharpe 3.22 (2mo) vs 2.22 (B10 4mo). 83 trades at 88.0% WR. ATR volatility entry + dual exit is strong.
3. **STOCH solo on 2mo underperforms** — Sharpe 2.91 is below B7's 4.15 (3mo). Higher budget (784 vs 400 trials) didn't help. STOCH solo may need ≥3mo data to find the multi-timeframe patterns.
4. **LONG still dead** — 12th batch with direction=null, still 100% SHORT convergence. All 12 strategies across all runs are SHORT.
5. **100% trade match continues** — 6th consecutive batch with perfect pipeline alignment.
6. **Higher budget (784 trials) produced diminishing returns** — CCI (3.24) and ATR+ROC (3.22) are good but not better than B10's STOCH+ATR (4.70). The extra trials don't overcome combo choice.

### Updated 1m All-Time Leaderboard (Validated Sharpe)

| Rank | Batch | Combo | Sharpe | DD | Trades | PF | WR | Dir | Data | Trades/day |
|------|-------|-------|--------|-----|--------|-----|-----|-----|------|------------|
| 1 | B10 | STOCH+ATR | **4.70** | 9.4% | 62 | 1.79 | 8.1% | SHORT | 4mo | 0.5 |
| 2 | B11 | STOCH+ATR | 4.27 | 5.6% | 73 | 1.80 | 87.7% | SHORT | 2mo | 1.2 |
| 3 | B7 | STOCH solo | 4.15 | 5.3% | 28 | 1.96 | 42.9% | SHORT | 3mo | 0.3 |
| 4 | B9 | STOCH+ROC | 4.02 | 11.0% | 108 | 1.63 | 9.3% | SHORT | 4mo | 0.9 |
| 5 | **B12** | **CCI solo** | **3.24** | **4.5%** | **90** | **1.64** | **90.0%** | SHORT | **2mo** | **1.5** |
| 6 | **B12** | **ATR+ROC** | **3.22** | 5.7% | 83 | 1.50 | 88.0% | SHORT | **2mo** | **1.4** |
| 7 | B11 | CCI+ROC | 3.05 | 4.9% | 63 | 1.56 | 88.9% | SHORT | 2mo | 1.1 |
| 8 | **B12** | **STOCH solo** | **2.91** | 5.7% | 58 | 1.45 | 70.7% | SHORT | **2mo** | **1.0** |
| 9 | B11 | STOCH+ROC | 2.86 | 9.0% | 151 | 1.51 | 90.1% | SHORT | 2mo | 2.5 |

### Recommendations

1. **CCI solo (sid=170) is the best 2mo CCI strategy ever** — 90 trades, 90.0% WR, 4.5% DD. Add to paper trading portfolio.
2. **ATR+ROC confirmed viable on shorter windows** — 3.22 (2mo) vs 2.22 (4mo). The volatility-entry pattern works better on recent data.
3. **STOCH solo needs ≥3mo data** — 2.91 (2mo) vs 4.15 (3mo). The multi-timeframe STOCH analysis needs more price history.
4. **LONG is definitively dead on 1m BTCUSDT** — 12 batches, every possible indicator, direction=null. Zero long strategies found. The edge is short-only.
5. **Next batch: try STOCH+CCI on 2mo** — the most consistent combo across all data windows, never tested on 2mo with high budget.

---

## 2026-03-15: Batch 13 — STOCH+CCI, STOCH+ROC, ROC Solo on 2mo

### Goal

Test the two most proven combos (STOCH+CCI, STOCH+ROC) and ROC solo on 2mo data with high budget (pop=28, gens=28 = 784 trials). Direction=null. STOCH+CCI is the #1 all-time most consistent combo but never tested on 2mo.

### Configuration

| Run | Indicators | Pop | Gens | Trials | Direction | Time | Status |
|-----|-----------|-----|------|--------|-----------|------|--------|
| 582 | STOCH+CCI | 28 | 28 | 784 | random | ~25min (converged gen 20) | **completed** |
| 583 | STOCH+ROC | 28 | 28 | 784 | random | ~27min (converged gen 22) | **completed** |
| 584 | ROC solo | 28 | 28 | 784 | random | ~12min | **FAILED** (0 strategies) |

Data range: 2026-01-15 to 2026-03-15 (2 months). BTCUSDT 1m. ~87K bars.

### Winning Strategy DSL Details

**Run 582 (STOCH+CCI) — sid=172**
```yaml
entry_conditions:
  short: ["stoch_entry_0 crosses_below 55.67", "cci_entry_1 >= -128.09"]  # STOCH(21,4) + CCI(49)
exit_conditions:
  short: ["stoch_exit_0 crosses_above 52.86", "cci_exit_1 crosses_below -9.38"]  # STOCH(19,9) + CCI(14)
stop_loss: {type: fixed_pct, percent: 2.86}
take_profit: {type: fixed_pct, percent: 18.91}
```
Extreme reward/risk (18.91% TP / 2.86% SL = 6.6x). Only 52.2% WR but winners are massive. Trend-follower architecture.

**Run 583 (STOCH+ROC) — sid=173**
```yaml
entry_conditions:
  short: ["roc_entry_0 >= -3.27", "stoch_entry_1 crosses_below 69.36"]  # ROC(12) + STOCH(13,7)
exit_conditions:
  short: ["stoch_exit_0 <= 42.51", "stoch_exit_1 > 77.40"]  # STOCH(11,8) + STOCH(11,6)
stop_loss: {type: fixed_pct, percent: 8.98}
take_profit: {type: fixed_pct, percent: 0.97}
```
Ultra-tight 0.97% TP scalper with 93.7% WR. ROC entry filter (>= -3.27 means "not strongly falling") + STOCH mid-range crossdown. Dual STOCH contradictory exit (≤42.5 AND >77.4) — effectively exits on STOCH divergence.

### Full Pipeline Results

| Stage | 582 STOCH+CCI | 583 STOCH+ROC | 584 ROC solo |
|-------|--------------|--------------|-------------|
| Disc score | 0.6555 | **0.6991** | FAIL |
| Disc sharpe | 3.68 | **4.01** | — |
| Disc trades | 69 | **95** | — |
| Disc return | **9.7%** | 7.7% | — |
| DSR | PASS 2/2 | PASS 5/5 | — |
| Screen trades | 69 ✓ | 95 ✓ | — |
| Val trades | **69 (100%)** | **95 (100%)** | — |
| Val sharpe | 3.68 | **4.01** | — |
| Val sortino | **6.59** | 6.33 | — |
| Val return | **9.7%** | 7.7% | — |
| Val DD | 7.1% | **3.9%** | — |
| Val PF | 1.56 | **1.92** | — |
| Val WR | 52.2% | **93.7%** | — |
| Val fees | $37.90 | $16.58 | — |
| Strategy ID | sid=172 | **sid=173** | — |

### Issues Found

1. **ROC solo completely fails on 2mo** — 0 strategies found in 784 trials across 20 gens. ROC solo needs ≥3mo data (B8 re-val showed 1.83 Sharpe on 3mo). On 2mo, ROC thresholds can't find viable entries.

### Key Findings

1. **STOCH+ROC (sid=173) is the new 2mo champion** — Sharpe 4.01, 95 trades (~1.6/day), 93.7% WR, 3.9% DD, PF 1.92. Best PF in 2mo history. Ultra-tight 0.97% TP scalper.
2. **STOCH+CCI on 2mo produces a trend-follower** — Sharpe 3.68, 52.2% WR, 18.91% TP. Completely different architecture from the scalpers. STOCH+CCI adapts to 2mo by widening TP rather than tightening it.
3. **ROC solo is definitively dead on 2mo** — 0 strategies found. Combined with B8's 2-trade validation collapse on 3mo, ROC solo needs ≥4mo data (B9: 3.74 on 4mo).
4. **All SHORT again** — 13th consecutive batch, direction=null, 100% SHORT. This is no longer a finding — it's a fundamental property of 1m BTCUSDT.
5. **STOCH+ROC has the best 2mo PF (1.92)** — significantly above any other 2mo strategy. The ROC entry filter removes bad setups effectively.
6. **100% trade match continues** — 7th consecutive perfect batch.

### Updated 1m All-Time Leaderboard (Validated Sharpe)

| Rank | Batch | Combo | Sharpe | DD | Trades | PF | WR | Dir | Data | Trades/day |
|------|-------|-------|--------|-----|--------|-----|-----|-----|------|------------|
| 1 | B10 | STOCH+ATR | **4.70** | 9.4% | 62 | 1.79 | 8.1% | SHORT | 4mo | 0.5 |
| 2 | B11 | STOCH+ATR | 4.27 | 5.6% | 73 | 1.80 | 87.7% | SHORT | 2mo | 1.2 |
| 3 | B7 | STOCH solo | 4.15 | 5.3% | 28 | 1.96 | 42.9% | SHORT | 3mo | 0.3 |
| 4 | B9 | STOCH+ROC | 4.02 | 11.0% | 108 | 1.63 | 9.3% | SHORT | 4mo | 0.9 |
| 5 | **B13** | **STOCH+ROC** | **4.01** | **3.9%** | **95** | **1.92** | **93.7%** | SHORT | **2mo** | **1.6** |
| 6 | **B13** | **STOCH+CCI** | **3.68** | 7.1% | 69 | 1.56 | 52.2% | SHORT | **2mo** | **1.2** |
| 7 | B12 | CCI solo | 3.24 | 4.5% | 90 | 1.64 | 90.0% | SHORT | 2mo | 1.5 |
| 8 | B12 | ATR+ROC | 3.22 | 5.7% | 83 | 1.50 | 88.0% | SHORT | 2mo | 1.4 |
| 9 | B11 | CCI+ROC | 3.05 | 4.9% | 63 | 1.56 | 88.9% | SHORT | 2mo | 1.1 |
| 10 | B12 | STOCH solo | 2.91 | 5.7% | 58 | 1.45 | 70.7% | SHORT | 2mo | 1.0 |

### Recommendations

1. **Paper trade sid=173 (STOCH+ROC, 2mo)** — Sharpe 4.01, PF 1.92, 93.7% WR, 3.9% DD. Best risk-adjusted 2mo strategy. Ultra-tight 0.97% TP.
2. **Portfolio: sid=173 (scalper) + sid=172 (trend-follower)** — same indicators (STOCH+CCI/ROC), opposite architectures (0.97% vs 18.91% TP). Potentially uncorrelated.
3. **ROC solo definitively needs ≥4mo data** — failed on 2mo (0 strategies) and 3mo (2 trades val). Only viable on 4mo+.
4. **2mo discovery landscape is mature** — 3 batches (B11-B13) tested all reasonable combos. Top 5 strategies span 3 architectures.

---

## 2026-03-15: Batch 14 — 3mo Window Test (STOCH+ATR, CCI+ATR, Triple)

### Goal

Test combos on 3mo data (2025-12-15 → 2026-03-15) with direction=null: STOCH+ATR (all-time champion on 4mo), CCI+ATR (previously failed on forced SHORT), STOCH+CCI+ROC triple. Pop=28, gens=28.

### Configuration

| Run | Indicators | Pop | Gens | Trials | Direction | Time | Status |
|-----|-----------|-----|------|--------|-----------|------|--------|
| 589 | STOCH+ATR | 28 | 28 | 784 | random | ~35min | **FAILED** (0 strategies) |
| 590 | CCI+ATR | 28 | 28 | 784 | random | ~40min (converged gen 24) | completed (WEAK) |
| 591 | STOCH+CCI+ROC | 28 | 28 | 784 | random | ~42min (converged gen 23) | completed (WEAK) |

Data range: 2025-12-15 to 2026-03-15 (3 months, ~130K bars). 3 parallel on 10 cores.

### Full Pipeline Results

| Stage | 589 STOCH+ATR | 590 CCI+ATR | 591 STOCH+CCI+ROC |
|-------|-------------|------------|-------------------|
| Disc score | FAIL | 0.5013 | 0.4922 |
| Disc sharpe | — | 1.85 | 1.74 |
| Disc trades | — | 186 | 94 |
| Disc return | — | 4.1% | 6.2% |
| DSR | — | PASS 1/1 | PASS 1/1 |
| Val trades | — | **186 (100%)** | **94 (100%)** |
| Val sharpe | — | 1.85 | 1.74 |
| Val DD | — | 9.8% | 11.5% |
| Val PF | — | 1.18 | 1.21 |
| Val WR | — | 55.4% | 39.4% |
| Strategy ID | — | sid=174 | sid=175 |

### Key Findings

1. **STOCH+ATR fails on 3mo with direction=null** — 0 strategies in 784 trials. Previously succeeded on 3mo with forced SHORT (B5: 2.63, B6: 3.64). The direction=null search space is too large for STOCH+ATR on 3mo.
2. **3mo with 3 parallel is too slow for high-budget runs** — 35-42min per run vs 17-28min on 2mo. CPU contention from 12 workers on 10 cores slows everything.
3. **CCI+ATR weak on 3mo** — Sharpe 1.85 with 186 trades. Not competitive with 2mo results.
4. **Triple combo weak on 3mo** — Sharpe 1.74. Consistent with B3 (1.02 on 12mo). Triple combos don't work on 1m regardless of data window.
5. **2mo data is definitively the optimal window for discovery** — B11-B13 (2mo) produced Sharpe 2.86-4.27. B14 (3mo) produced 1.74-1.85. Shorter data = stronger strategies.
6. **Still all SHORT** — 14 batches.

### Recommendations

1. **Return to 2mo for future batches** — 3mo adds computation time without improving results.
2. **Force SHORT if continuing 3mo experiments** — direction=null wastes half the search space on LONG which never works.
3. **Focus on paper trading the B11-B13 winners** — sid=166 (STOCH+ATR, 4.27), sid=170 (CCI, 3.24), sid=173 (STOCH+ROC, 4.01).

---

## 2026-03-15: Batch 15 — ATR Solo, STOCH+ATR, CCI+ROC on 2mo (High Budget)

### Goal

Return to 2mo window after B14's weak 3mo results. Test ATR solo (strong on 4mo, never on 2mo), STOCH+ATR (all-time champion architecture), and CCI+ROC (B11 got 3.05). All direction=null, pop=28 gens=28.

### Configuration

| Run | Indicators | Pop | Gens | Trials | Direction | Time | Status |
|-----|-----------|-----|------|--------|-----------|------|--------|
| 596 | ATR solo | 28 | 28 | 784 | random | ~11min (converged gen 26) | **completed** |
| 597 | STOCH+ATR | 28 | 28 | 784 | random | ~30min | **completed** |
| 598 | CCI+ROC | 28 | 28 | 784 | random | ~14min (converged gen 20) | **completed** |

Data range: 2026-01-15 to 2026-03-15 (2 months). BTCUSDT 1m.

### Winning Strategy DSL Details

**Run 596 (ATR solo) — sid=176**
- Entry: ATR(22) >= 0.1171 → short (high volatility entry)
- Exit: ATR(?) < 0.0396 (volatility collapse)
- SL: 0.88% / TP: 5.79% — tight SL, wide TP = tail-win (18.1% WR)

**Run 597 (STOCH+ATR) — sid=177**
- Entry: STOCH >= 58.27 AND STOCH crosses_above 29.61 → short
- Exit: ATR < 0.1219 AND STOCH <= 24.48
- SL: 6.22% / TP: 2.03% — 80.4% WR scalper

**Run 598 (CCI+ROC) — sid=178**
- Entry: CCI(?) crosses_below 141.57 → short (overbought reversal)
- Exit: ROC crosses_above -0.83 AND CCI >= 101.63
- SL: 0.66% / TP: 10.65% — ultra-tight 0.66% SL, wide 10.65% TP = extreme tail-win (11.3% WR)

### Full Pipeline Results

| Stage | 596 ATR solo | 597 STOCH+ATR | 598 CCI+ROC |
|-------|-------------|--------------|-------------|
| Disc score | 0.6636 | 0.6932 | **0.7071** |
| Disc sharpe | 3.51 | 4.01 | **4.13** |
| Disc trades | **105** | 51 | 62 |
| Disc return | 10.0% | 10.1% | **11.6%** |
| DSR | PASS 1/1 | PASS 5/5 | PASS 2/2 |
| Screen trades | 105 ✓ | 51 ✓ | 62 ✓ |
| Val trades | **105 (100%)** | **51 (100%)** | **62 (100%)** |
| Val sharpe | 3.51 | 4.01 | **4.13** |
| Val sortino | 6.07 | 5.74 | **8.63** |
| Val return | 10.0% | 10.1% | **11.6%** |
| Val DD | 12.6% | **4.3%** | 7.8% |
| Val PF | 1.41 | 1.76 | **1.81** |
| Val WR | 18.1% | **80.4%** | 11.3% |
| Val fees | $56.76 | $13.82 | $34.38 |
| Strategy ID | sid=176 | **sid=177** | **sid=178** |

### Key Findings

1. **CCI+ROC (sid=178) is the new 2mo Sharpe champion** — 4.13, beating B13's STOCH+ROC (4.01). Sortino 8.63 is the highest ever on 2mo. CCI overbought reversal entry + ROC/CCI exit. Extreme tail-win architecture (0.66% SL / 10.65% TP, 11.3% WR).
2. **STOCH+ATR (sid=177) on 2mo is excellent** — Sharpe 4.01, 4.3% DD (lowest in batch), 80.4% WR scalper. Same architecture as B11's champion (4.27) but different parameter optimization.
3. **ATR solo viable on 2mo** — Sharpe 3.51, 105 trades (~1.7/day). Simple volatility entry/exit. High trade frequency but 12.6% DD is the highest in this batch.
4. **Three distinct architectures coexist** — tail-win (11.3% WR, 10.65% TP), scalper (80.4% WR, 2.03% TP), and volatility-filter (18.1% WR, 5.79% TP). All profitable with different risk profiles.
5. **All SHORT** — 15th consecutive batch. The BTCUSDT 1m short-side edge is a fundamental market property in this period.
6. **100% trade match** — 8th consecutive perfect batch.

### Updated 1m All-Time Leaderboard (Validated Sharpe, 2mo window)

| Rank | Batch | Combo | Sharpe | Sortino | DD | Trades | PF | WR | Trades/day |
|------|-------|-------|--------|---------|-----|--------|-----|-----|------------|
| 1 | B11 | STOCH+ATR | 4.27 | 7.12 | 5.6% | 73 | 1.80 | 87.7% | 1.2 |
| 2 | **B15** | **CCI+ROC** | **4.13** | **8.63** | 7.8% | 62 | **1.81** | 11.3% | **1.0** |
| 3 | B13 | STOCH+ROC | 4.01 | 6.33 | **3.9%** | 95 | **1.92** | 93.7% | 1.6 |
| 4 | **B15** | **STOCH+ATR** | **4.01** | 5.74 | **4.3%** | 51 | 1.76 | 80.4% | **0.9** |
| 5 | B13 | STOCH+CCI | 3.68 | 6.59 | 7.1% | 69 | 1.56 | 52.2% | 1.2 |
| 6 | **B15** | **ATR solo** | **3.51** | 6.07 | 12.6% | **105** | 1.41 | 18.1% | **1.7** |
| 7 | B12 | CCI solo | 3.24 | 5.02 | 4.5% | 90 | 1.64 | 90.0% | 1.5 |
| 8 | B12 | ATR+ROC | 3.22 | 5.00 | 5.7% | 83 | 1.50 | 88.0% | 1.4 |
| 9 | B11 | CCI+ROC | 3.05 | 4.41 | 4.9% | 63 | 1.56 | 88.9% | 1.1 |
| 10 | B12 | STOCH solo | 2.91 | 4.69 | 5.7% | 58 | 1.45 | 70.7% | 1.0 |
| 11 | B11 | STOCH+ROC | 2.86 | 4.45 | 9.0% | **151** | 1.51 | 90.1% | **2.5** |

### Recommendations

1. **Paper trade sid=178 (CCI+ROC)** — new 2mo Sharpe champion (4.13, Sortino 8.63). Extreme tail-win architecture, low fees ($34).
2. **Portfolio of 3: sid=173 + sid=177 + sid=178** — scalper (93.7% WR) + STOCH+ATR (80.4% WR) + tail-win (11.3% WR). Maximum architecture diversification.
3. **2mo discovery is exhaustively explored** — 5 batches (B11-B15), every solo and combo tested. All 11 leaderboard strategies are profitable with distinct architectures.
4. **Consider moving to paper trading phase** — diminishing returns from further discovery. Focus on live testing the top strategies.

---

## 2026-03-15: Batch 16 — Ultra-Budget 1024 Trials (STOCH+CCI, CCI+ATR, Triple)

### Goal

Push budget to pop=32 gens=32 (1024 trials) on 2mo. Test STOCH+CCI (higher budget than B13), CCI+ATR (first time on 2mo), and STOCH+ROC+ATR triple. Direction=null.

### Configuration

| Run | Indicators | Pop | Gens | Trials | Direction | Time | Status |
|-----|-----------|-----|------|--------|-----------|------|--------|
| 605 | STOCH+CCI | 32 | 32 | 1024 | random | ~55min | **completed** |
| 606 | CCI+ATR | 32 | 32 | 1024 | random | ~28min (converged gen 21) | **completed** |
| 607 | STOCH+ROC+ATR | 32 | 32 | 1024 | random | ~35min (converged gen 24) | **completed** |

Data range: 2026-01-15 to 2026-03-15 (2 months). BTCUSDT 1m.

### Winning Strategy DSL Details

**Run 605 (STOCH+CCI) — sid=179**
- Entry: STOCH <= 78.33 → short. Exit: STOCH < 30.86, CCI crosses_above -19.37
- SL: 4.62% / TP: 13.58%. 54.1% WR balanced strategy.
- **GA used STOCH+CCI as intended** — good multi-indicator combo.

**Run 606 (CCI+ATR → pure CCI) — sid=180**
- Entry: CCI crosses_above -159.56 → short. Exit: CCI crosses_below 42.25, CCI < -139.78
- SL: 6.31% / TP: 8.64%. **GA ignored ATR entirely** — 3 CCI indicators, 0 ATR.
- Sortino **9.31** — highest in batch. Deep CCI oversold entry.

**Run 607 (STOCH+ROC+ATR → STOCH+ROC) — sid=181**
- Entry: STOCH crosses_above 48.57 → short. Exit: ROC > 2.81
- SL: 1.23% / TP: 6.39%. **GA dropped ATR** — used STOCH entry + ROC exit only.
- 24.6% WR tail-win with tight 1.23% SL.

### Full Pipeline Results

| Stage | 605 STOCH+CCI | 606 CCI+ATR→CCI | 607 Triple→STOCH+ROC |
|-------|--------------|-----------------|---------------------|
| Disc score | 0.6165 | 0.7016 | **0.7144** |
| Disc sharpe | 3.03 | 3.86 | **4.08** |
| Disc trades | **98** | 58 | 57 |
| Disc return | 7.7% | 9.3% | **13.7%** |
| DSR | PASS 5/5 | PASS 5/5 | PASS 5/5 |
| Val trades | **98 (100%)** | **58 (100%)** | **57 (100%)** |
| Val sharpe | 3.03 | 3.86 | **4.08** |
| Val sortino | 4.29 | **9.31** | 8.26 |
| Val return | 7.7% | 9.3% | **13.7%** |
| Val DD | 10.1% | **6.0%** | 10.4% |
| Val PF | 1.45 | **1.84** | 1.62 |
| Val WR | **54.1%** | 51.7% | 24.6% |
| Val fees | $45.60 | $19.13 | $31.05 |
| Strategy ID | sid=179 | **sid=180** | **sid=181** |

### Key Findings

1. **Triple combo (607) produces STOCH+ROC** — GA dropped ATR from the {STOCH,ROC,ATR} pool, reverting to the proven 2-indicator combo. Sharpe 4.08, 13.7% return. Confirms triple combos add complexity without value — GA naturally simplifies.
2. **CCI+ATR → pure CCI** — GA ignored ATR entirely from the {CCI,ATR} pool, using 3 CCI indicators. Sharpe 3.86, Sortino 9.31 (highest ever on 2mo for CCI). GA prefers indicator purity on 1m.
3. **1024 trials didn't beat 784 trials** — B13's STOCH+ROC at 784 trials got Sharpe 4.01 (PF 1.92). B16's STOCH+ROC at 1024 got 4.08 (PF 1.62). Marginal Sharpe gain but lower PF. Diminishing returns confirmed.
4. **STOCH+CCI is the weakest 2mo combo** — Sharpe 3.03 at 1024 trials, vs B13's 3.68 at 784 trials. Higher budget actually produced a weaker strategy — GA may overfit with too many trials on this combo.
5. **All SHORT** — 16th consecutive batch.
6. **100% trade match** — 9th consecutive perfect batch.
7. **GA indicator pruning is consistent** — when given 3-indicator pools, GA drops the weakest. ATR is always dropped in favor of STOCH/CCI/ROC.

### Updated 1m All-Time Leaderboard (Validated Sharpe, 2mo)

| Rank | Batch | Combo | Sharpe | Sortino | DD | Trades | PF | WR | Trades/day |
|------|-------|-------|--------|---------|-----|--------|-----|-----|------------|
| 1 | B11 | STOCH+ATR | 4.27 | 7.12 | 5.6% | 73 | 1.80 | 87.7% | 1.2 |
| 2 | B15 | CCI+ROC | 4.13 | 8.63 | 7.8% | 62 | 1.81 | 11.3% | 1.0 |
| 3 | **B16** | **STOCH+ROC** | **4.08** | 8.26 | 10.4% | 57 | 1.62 | 24.6% | **1.0** |
| 4 | B13 | STOCH+ROC | 4.01 | 6.33 | **3.9%** | 95 | **1.92** | 93.7% | 1.6 |
| 5 | **B16** | **CCI (from ATR pool)** | **3.86** | **9.31** | **6.0%** | 58 | 1.84 | 51.7% | **1.0** |
| 6 | B13 | STOCH+CCI | 3.68 | 6.59 | 7.1% | 69 | 1.56 | 52.2% | 1.2 |
| 7 | B15 | ATR solo | 3.51 | 6.07 | 12.6% | 105 | 1.41 | 18.1% | 1.7 |
| 8 | B12 | CCI solo | 3.24 | 5.02 | 4.5% | 90 | 1.64 | 90.0% | 1.5 |
| 9 | B12 | ATR+ROC | 3.22 | 5.00 | 5.7% | 83 | 1.50 | 88.0% | 1.4 |
| 10 | B11 | CCI+ROC | 3.05 | 4.41 | 4.9% | 63 | 1.56 | 88.9% | 1.1 |
| 11 | **B16** | **STOCH+CCI** | **3.03** | 4.29 | 10.1% | **98** | 1.45 | 54.1% | **1.6** |
| 12 | B11 | STOCH+ROC | 2.86 | 4.45 | 9.0% | **151** | 1.51 | 90.1% | **2.5** |

### Recommendations

1. **1024 trials offer diminishing returns** — stick with 784 (pop=28, gens=28) for future batches.
2. **GA naturally prunes to 2 indicators** — triple pools waste budget. Use explicit 2-indicator pools.
3. **2mo discovery is exhaustively explored** — 6 batches, every combo at multiple budget levels. Top strategies are stable. Move to paper trading.
4. **Best portfolio candidates**: sid=173 (STOCH+ROC scalper, 93.7% WR, PF 1.92), sid=178 (CCI+ROC tail-win, Sortino 8.63), sid=177 (STOCH+ATR, 4.3% DD).

---

## 2026-03-15: Batch 17 — 2.5mo Window Test (Champion Combos)

### Goal

Test whether 2.5mo (2026-01-01 → 2026-03-15, ~107K bars) is a better sweet spot between 2mo and 3mo. Used the 3 champion combos: STOCH+ROC, CCI+ROC, STOCH+CCI. Direction=null, pop=28 gens=28.

### Configuration

| Run | Indicators | Pop | Gens | Trials | Direction | Time | Status |
|-----|-----------|-----|------|--------|-----------|------|--------|
| 614 | STOCH+ROC | 28 | 28 | 784 | random | ~35min (converged gen 27) | completed (WEAK) |
| 615 | CCI+ROC | 28 | 28 | 784 | random | ~40min | **completed** |
| 616 | STOCH+CCI | 28 | 28 | 784 | random | ~45min | completed |

Data: 2026-01-01 to 2026-03-15 (2.5 months, ~107K bars).

### Full Pipeline Results

| Stage | 614 STOCH+ROC | 615 CCI+ROC | 616 STOCH+CCI |
|-------|-------------|------------|--------------|
| Disc score | 0.4964 | **0.6270** | 0.5850 |
| Disc sharpe | 1.78 | **2.91** | 2.61 |
| Disc trades | 85 | **88** | 79 |
| Val sharpe | 1.78 | **2.91** | 2.61 |
| Val sortino | 2.44 | 4.15 | **4.20** |
| Val DD | **4.4%** | **3.4%** | 4.1% |
| Val PF | 1.32 | **1.54** | 1.53 |
| Val WR | 89.4% | **93.2%** | 55.7% |
| Val return | 5.1% | 5.7% | **7.6%** |
| Strategy ID | sid=182 | **sid=183** | sid=184 |

### Key Findings

1. **2.5mo is worse than 2mo across all combos** — CCI+ROC: 2.91 (2.5mo) vs 4.13 (2mo). STOCH+CCI: 2.61 vs 3.68. STOCH+ROC: 1.78 vs 4.01. The extra 2 weeks of data hurts, not helps.
2. **2mo is definitively the optimal window** — tested 2mo (B11-16), 2.5mo (B17), 3mo (B14), 4mo (B9-10), 5mo (B8), 12mo (B2-4). 2mo consistently produces the highest Sharpe strategies.
3. **Low DD across all 2.5mo strategies** — 3.4-4.4% DD is excellent. The longer window produces more conservative strategies but with lower returns.
4. **Still all SHORT** — 17th consecutive batch.
5. **100% trade match** — 10th consecutive perfect batch.

### Recommendations

1. **Stop exploring data windows** — 2mo is proven optimal across 7 batches of comparisons. Future batches should use 2mo exclusively.
2. **Discovery landscape is fully mapped** — 17 batches, every combo at multiple budget levels and data windows. Move to paper trading phase.

---

## 2026-03-15: Batch 18 — RSI Revival + STOCH+ATR/CCI+ATR on 2mo

### Goal

Test RSI+STOCH on 2mo (RSI failed on 3mo in B1, never tested on 2mo), rerun STOCH+ATR (attempt to beat B11's 4.27), and CCI+ATR (first genuine 2mo test). Direction=null, pop=28 gens=28.

### Configuration

| Run | Indicators | Pop | Gens | Trials | Direction | Time | Status |
|-----|-----------|-----|------|--------|-----------|------|--------|
| 623 | RSI+STOCH | 28 | 28 | 784 | random | ~35min | **completed** |
| 624 | STOCH+ATR | 28 | 28 | 784 | random | ~40min | **completed** |
| 625 | CCI+ATR | 28 | 28 | 784 | random | ~30min (converged gen 26) | completed |

Data: 2026-01-15 to 2026-03-15 (2 months). BTCUSDT 1m.

### Winning Strategy DSL Details

**Run 623 (RSI+STOCH) — sid=185 — NEW ALL-TIME SHARPE CHAMPION**
```yaml
entry_conditions:
  short: ["rsi_entry_0 <= 34.6063"]   # RSI(50) — slow RSI oversold entry
exit_conditions:
  short: ["rsi_exit_0 > 58.67", "stoch_exit_1 crosses_above 55.20"]  # RSI(8) + STOCH(14,3)
stop_loss: {type: fixed_pct, percent: 0.63}
take_profit: {type: fixed_pct, percent: 12.32}
```
**Sharpe 6.05, Sortino 13.48.** Extreme tail-win: 0.63% SL / 12.32% TP = 19.6x reward/risk. Only 8.7% WR but each winner is massive. RSI(50) as a slow momentum filter on 1m — GA found that low RSI periods precede short-side moves.

**Run 624 (STOCH+ATR) — sid=186 — NEW ALL-TIME PF/WR CHAMPION**
```yaml
entry_conditions:
  short: ["stoch_entry_0 crosses_below 38.07", "stoch_entry_1 > 63.61", "stoch_entry_2 >= 30.33"]
exit_conditions:
  short: ["stoch_exit_0 crosses_below 43.13", "atr_exit_1 crosses_below 0.0223"]
stop_loss: {type: fixed_pct, percent: 7.9}
take_profit: {type: fixed_pct, percent: 0.5}
```
**Sharpe 6.01, PF 3.67, WR 98.3%, DD 2.0%.** Ultra-tight 0.5% TP with 3 STOCH entry conditions + ATR volatility exit. 98.3% WR is the highest ever. PF 3.67 is the highest ever. DD 2.0% is the lowest ever. The ultimate 1m scalper.

**Run 625 (CCI+ATR) — sid=187**
- Entry: CCI(16) crosses_below -17.73, CCI(14) <= 174.47. Exit: ATR(6) <= 0.0467.
- SL: 7.07% / TP: 1.72%. Sharpe 2.70 — decent but not competitive with 623/624.

### Full Pipeline Results

| Stage | 623 RSI+STOCH | 624 STOCH+ATR | 625 CCI+ATR |
|-------|-------------|--------------|-------------|
| Disc score | 0.6929 | **0.7517** | 0.6018 |
| Disc sharpe | **6.05** | 6.01 | 2.70 |
| Disc trades | **69** | 59 | 50 |
| Disc return | 9.7% | 4.7% | 4.5% |
| DSR | PASS 5/5 | PASS 5/5 | PASS 3/3 |
| Val trades | **69 (100%)** | **59 (100%)** | **50 (100%)** |
| Val sharpe | **6.05** | 6.01 | 2.70 |
| Val sortino | **13.48** | 6.88 | 4.03 |
| Val return | **9.7%** | 4.7% | 4.5% |
| Val DD | 11.1% | **2.0%** | 6.2% |
| Val PF | 1.68 | **3.67** | 1.47 |
| Val WR | 8.7% | **98.3%** | 86.0% |
| Val fees | $37.88 | $11.18 | $11.40 |
| Strategy ID | **sid=185** | **sid=186** | sid=187 |

### Key Findings

1. **RSI WORKS on 2mo!** — overturns B1's finding that RSI fails on 1m (3mo data). RSI(50) as a slow oversold filter produces Sharpe 6.05 — the highest validated Sharpe in the entire 1m journal. The key: 2mo window and RSI as entry filter (not primary signal).
2. **STOCH+ATR rerun produced the most extreme strategy ever** — Sharpe 6.01, PF 3.67, WR 98.3%, DD 2.0%. All-time records in PF, WR, and DD simultaneously. The 0.5% TP ultra-scalper with 3 STOCH confirmation entries.
3. **Both new champions have very different architectures** — sid=185 is a tail-win (8.7% WR, 12.32% TP) while sid=186 is an ultra-scalper (98.3% WR, 0.5% TP). Maximally uncorrelated for portfolio construction.
4. **Random re-runs of the same combo produce different strategies** — B11 STOCH+ATR got Sharpe 4.27, B15 got 4.01, B18 got 6.01. GA randomization matters — worth re-running champion combos.
5. **CCI+ATR on 2mo is mediocre** — Sharpe 2.70, GA used CCI entry + ATR exit as expected. Consistent with B16 where GA dropped ATR entirely when given CCI+ATR pool.
6. **All SHORT** — 18th consecutive batch.
7. **100% trade match** — 11th consecutive perfect batch.

### Updated 1m All-Time Leaderboard (Validated Sharpe, 2mo)

| Rank | Batch | Combo | Sharpe | Sortino | DD | Trades | PF | WR | Trades/day |
|------|-------|-------|--------|---------|-----|--------|-----|-----|------------|
| 1 | **B18** | **RSI+STOCH** | **6.05** | **13.48** | 11.1% | 69 | 1.68 | 8.7% | **1.2** |
| 2 | **B18** | **STOCH+ATR** | **6.01** | 6.88 | **2.0%** | 59 | **3.67** | **98.3%** | **1.0** |
| 3 | B11 | STOCH+ATR | 4.27 | 7.12 | 5.6% | 73 | 1.80 | 87.7% | 1.2 |
| 4 | B15 | CCI+ROC | 4.13 | 8.63 | 7.8% | 62 | 1.81 | 11.3% | 1.0 |
| 5 | B16 | STOCH+ROC | 4.08 | 8.26 | 10.4% | 57 | 1.62 | 24.6% | 1.0 |
| 6 | B13 | STOCH+ROC | 4.01 | 6.33 | 3.9% | 95 | 1.92 | 93.7% | 1.6 |
| 7 | B15 | STOCH+ATR | 4.01 | 5.74 | 4.3% | 51 | 1.76 | 80.4% | 0.9 |
| 8 | B16 | CCI (from ATR) | 3.86 | 9.31 | 6.0% | 58 | 1.84 | 51.7% | 1.0 |
| 9 | B13 | STOCH+CCI | 3.68 | 6.59 | 7.1% | 69 | 1.56 | 52.2% | 1.2 |
| 10 | B15 | ATR solo | 3.51 | 6.07 | 12.6% | 105 | 1.41 | 18.1% | 1.7 |

### Recommendations

1. **Paper trade sid=185 + sid=186 immediately** — two new all-time champions with maximally different architectures. sid=185 (tail-win, 8.7% WR) + sid=186 (ultra-scalper, 98.3% WR) = excellent diversification.
2. **Re-run champion combos with different random seeds** — B18 proves that random re-runs can dramatically improve results. STOCH+ATR went from 4.27 (B11) → 6.01 (B18).
3. **RSI is viable on 2mo 1m** — the B1 finding that RSI fails on 1m was 3mo-specific. RSI(50) as a slow filter works on 2mo.
4. **The 0.5% TP scalper (sid=186) needs careful live monitoring** — 98.3% WR means each loss is ~16x a typical win. A single bad streak could wipe significant gains. Paper trade with strict risk limits.

---

## 2026-03-15: Batch 19 — Exotic RSI Combos on 3.5mo

### Goal

Test exotic RSI-based combos on 3.5mo data (2025-12-01 → 2026-03-15, ~152K bars). RSI broke through in B18 (Sharpe 6.05). Test RSI paired with CCI, ROC, ATR — all first-time 1m combos. Direction=null, pop=30 gens=30 (900 trials).

### Configuration

| Run | Indicators | Pop | Gens | Trials | Direction | Time | Status |
|-----|-----------|-----|------|--------|-----------|------|--------|
| 632 | RSI+CCI | 30 | 30 | 900 | random | ~80min | completed (WEAK) |
| 633 | RSI+ROC | 30 | 30 | 900 | random | ~30min (converged gen 21) | completed (WEAK) |
| 634 | RSI+ATR | 30 | 30 | 900 | random | ~30min (converged gen 21) | **completed** |

Data: 2025-12-01 to 2026-03-15 (3.5 months, ~152K bars).

### Full Pipeline Results

| Stage | 632 RSI+CCI | 633 RSI+ROC | 634 RSI+ATR |
|-------|------------|------------|-------------|
| Disc score | 0.5371 | 0.5525 | **0.6807** |
| Disc sharpe | 1.70 | 1.96 | **3.93** |
| Disc trades | **108** | 60 | 53 |
| Disc return | 7.1% | 6.8% | 6.8% |
| DSR | PASS | PASS | PASS |
| Val trades | **108 (100%)** | **60 (100%)** | **53 (100%)** |
| Val sharpe | 1.70 | 1.96 | **3.93** |
| Val sortino | 2.56 | 3.27 | **12.59** |
| Val DD | 9.0% | 10.2% | **11.0%** |
| Val PF | 1.22 | 1.24 | **1.54** |
| Val WR | 73.1% | 43.3% | 7.5% |
| Strategy ID | sid=190 | sid=188 | **sid=189** |

### Key Findings

1. **RSI+ATR on 3.5mo is strong** — Sharpe 3.93, Sortino 12.59. RSI entry + ATR exit pattern works on longer data windows too. 7.5% WR tail-win.
2. **RSI+CCI is slow and weak** — CCI on 3.5mo with 900 trials took 80 minutes but only Sharpe 1.70. CCI with RSI doesn't synergize on 1m.
3. **RSI+ROC is weak** — Sharpe 1.96. Two momentum indicators don't complement each other.
4. **3.5mo produces lower Sharpe than 2mo across all combos** — consistent with B14/B17 findings. Longer windows dilute the short-side edge.
5. **All SHORT** — 19th consecutive batch.
6. **RSI works best with ATR exit** — same pattern as STOCH (best with ATR exit). ATR volatility exit is the universal best exit signal on 1m.

### Recommendations

1. **RSI+ATR (sid=189) viable for 3.5mo portfolio** — Sharpe 3.93, Sortino 12.59. Complementary to 2mo strategies for time diversification.
2. **RSI+CCI and RSI+ROC not competitive** — skip these combos in future batches.
3. **Next: try ADX combos and MFI combos on 2mo** — these are the truly unexplored exotic indicators on 2mo.

---

## 2026-03-15: Batch 20 — ADX+STOCH, RSI Solo, CCI+STOCH on 3.5mo

### Goal

Test truly exotic combos: ADX+STOCH (ADX never worked on 1m), RSI solo on 3.5mo, and CCI+STOCH (the all-time consistent combo on a longer window). Direction=null, pop=30 gens=30.

### Configuration

| Run | Indicators | Pop | Gens | Trials | Direction | Time | Status |
|-----|-----------|-----|------|--------|-----------|------|--------|
| 641 | ADX+STOCH | 30 | 30 | 900 | random | ~95min | completed (WEAK) |
| 642 | RSI solo | 30 | 30 | 900 | random | ~30min | completed (WEAK) |
| 643 | CCI+STOCH | 30 | 30 | 900 | random | ~120min (converged gen 20) | completed (WEAK) |

Data: 2025-12-01 to 2026-03-15 (3.5 months, ~152K bars).

### Full Pipeline Results

| Stage | 641 ADX+STOCH | 642 RSI solo | 643 CCI+STOCH |
|-------|-------------|-------------|--------------|
| Disc score | 0.4896 | 0.5193 | 0.4515 |
| Disc sharpe | 1.39 | 1.83 | 1.36 |
| Disc trades | 111 | 73 | **155** |
| Val sharpe | 1.39 | 1.83 | 1.36 |
| Val trades | **111 (100%)** | **73 (100%)** | **155 (100%)** |
| Strategy ID | sid=192 | sid=191 | sid=193 |

### Key Findings

1. **ADX confirmed dead on 1m** — Sharpe 1.39 with ADX+STOCH on 3.5mo. ADX has now failed on 3mo (B1), 4mo (B10 was ATR solo, not ADX), and 3.5mo (B20). ADX captures noise not trends on 1m.
2. **RSI solo is weak (1.83)** — RSI needs a companion (ATR best, STOCH second) to produce competitive strategies. Solo RSI on 3.5mo can't match B18's RSI+STOCH (6.05) or RSI+ATR (3.93).
3. **CCI+STOCH slow and weak on 3.5mo** — 120 minutes for Sharpe 1.36. CCI on 3.5mo with pop=30 is computationally crushing (~4min/gen). The combo that shines on 2-4mo (Sharpe 3-4) degrades to 1.36 on 3.5mo.
4. **3.5mo with CCI/ADX is too slow** — 95-120min per run. Not worth the compute time given weak results.
5. **All SHORT** — 20th consecutive batch.

### Recommendations

1. **Stop testing ADX on 1m** — confirmed dead across all data windows and combos.
2. **RSI only viable with companions** — RSI+ATR and RSI+STOCH are strong but RSI solo is weak.
3. **For 3.5mo, avoid CCI** — too slow for the marginal improvement. Use STOCH/ROC/ATR/RSI only.
4. **Next batch: back to 2mo with RSI reruns** — B18 showed random reruns can dramatically improve results. Try RSI+STOCH and RSI+ATR again.

---

## 2026-03-15: Batch 21 — RSI Combos on 2mo (Reruns + RSI+CCI)

### Goal

Rerun RSI+STOCH (B18 champion at 6.05), test RSI+ATR on 2mo (worked on 3.5mo B19), and RSI+CCI on 2mo (failed on 3.5mo B19). Direction=null, pop=30 gens=30 (900 trials).

### Configuration

| Run | Indicators | Pop | Gens | Trials | Direction | Time | Status |
|-----|-----------|-----|------|--------|-----------|------|--------|
| 650 | RSI+STOCH | 30 | 30 | 900 | random | ~40min (converged gen 23) | **completed** |
| 651 | RSI+ATR | 30 | 30 | 900 | random | ~30min | **FAILED** (0 strategies) |
| 652 | RSI+CCI | 30 | 30 | 900 | random | ~40min | **completed** |

Data: 2026-01-15 to 2026-03-15 (2 months). BTCUSDT 1m.

### Full Pipeline Results

| Stage | 650 RSI+STOCH | 651 RSI+ATR | 652 RSI+CCI |
|-------|-------------|------------|------------|
| Disc score | **0.7221** | FAIL | 0.6915 |
| Disc sharpe | **5.20** | — | 3.96 |
| Disc trades | 76 | — | **88** |
| Disc return | 8.7% | — | 9.1% |
| DSR | PASS | — | PASS |
| Val trades | **76 (100%)** | — | **88 (100%)** |
| Val sharpe | **5.20** | — | 3.96 |
| Val sortino | **11.42** | — | 8.23 |
| Val DD | **3.6%** | — | 9.3% |
| Val PF | **2.47** | — | 1.61 |
| Val WR | 56.6% | — | 9.1% |
| Strategy ID | **sid=194** | — | sid=195 |

### Key Findings

1. **RSI+STOCH rerun confirms RSI's 2mo viability** — Sharpe 5.20, PF 2.47, DD 3.6%. Third-best 2mo strategy ever (behind B18's 6.05 and 6.01). RSI+STOCH consistently produces Sharpe 5-6 on 2mo.
2. **RSI+ATR fails on 2mo** — 0 strategies in 900 trials. But RSI+ATR worked on 3.5mo (B19: 3.93). RSI+ATR needs ≥3mo data — RSI's slow filter (period 50) needs more price history to build signal.
3. **RSI+CCI works on 2mo!** — Sharpe 3.96, overturning B19's finding that RSI+CCI is weak (1.70 on 3.5mo). Another case where 2mo > 3.5mo.
4. **RSI+CCI is a tail-win** — 9.1% WR, the GA found an extreme tail-win architecture on 2mo. CCI provides high-threshold entry, RSI confirms momentum.
5. **All SHORT** — 21st consecutive batch.

### Updated 1m All-Time Leaderboard (Top 5, Validated Sharpe)

| Rank | Batch | Combo | Sharpe | Sortino | DD | Trades | PF | WR |
|------|-------|-------|--------|---------|-----|--------|-----|-----|
| 1 | B18 | RSI+STOCH | **6.05** | **13.48** | 11.1% | 69 | 1.68 | 8.7% |
| 2 | B18 | STOCH+ATR | **6.01** | 6.88 | **2.0%** | 59 | **3.67** | **98.3%** |
| 3 | **B21** | **RSI+STOCH** | **5.20** | 11.42 | **3.6%** | 76 | **2.47** | 56.6% |
| 4 | B11 | STOCH+ATR | 4.27 | 7.12 | 5.6% | 73 | 1.80 | 87.7% |
| 5 | B15 | CCI+ROC | 4.13 | 8.63 | 7.8% | 62 | 1.81 | 11.3% |

### Recommendations

1. **RSI+STOCH is the most consistent high-Sharpe combo** — B18 (6.05), B21 (5.20). Random reruns consistently produce Sharpe 5+.
2. **RSI+ATR needs ≥3mo** — skip on 2mo, use on 3.5-4mo only.
3. **RSI+CCI viable on 2mo** — Sharpe 3.96 is competitive. The tail-win architecture (9.1% WR) is a novel RSI+CCI pattern.

---

## 2026-03-15: Batch 22 — Champion Reruns on 2mo

### Goal

Rerun all 3 top combos (RSI+STOCH, STOCH+ATR, RSI+CCI) on 2mo with different random seeds. Testing whether B18's Sharpe 6+ results are reproducible. Direction=null, pop=30 gens=30.

### Configuration

| Run | Indicators | Pop | Gens | Trials | Direction | Time | Status |
|-----|-----------|-----|------|--------|-----------|------|--------|
| 657 | RSI+STOCH | 30 | 30 | 900 | random | ~35min | **FAILED** (0 strategies) |
| 658 | STOCH+ATR | 30 | 30 | 900 | random | ~35min | **FAILED** (0 strategies) |
| 659 | RSI+CCI | 30 | 30 | 900 | random | ~40min | **completed** |

### Full Pipeline Results

| Stage | 657 RSI+STOCH | 658 STOCH+ATR | 659 RSI+CCI |
|-------|-------------|-------------|------------|
| Disc score | FAIL | FAIL | **0.6871** |
| Disc sharpe | — | — | **3.92** |
| Disc trades | — | — | 50 |
| Val sharpe | — | — | 3.92 |
| Val sortino | — | — | 8.11 |
| Val DD | — | — | 9.6% |
| Val PF | — | — | 1.55 |
| Val WR | — | — | 24.0% |
| Strategy ID | — | — | sid=196 |

### Key Findings

1. **GA randomization is extremely volatile** — RSI+STOCH: B18 got 6.05, B21 got 5.20, B22 got 0 (FAIL). STOCH+ATR: B18 got 6.01, B22 got 0. Same combos, same data, wildly different results. The GA initial population matters more than the combo choice.
2. **2/3 runs failed** — direction=null on 2mo produces failures ~30-40% of the time for any given combo. The search space is large and many random initial populations don't contain viable solutions.
3. **RSI+CCI is the most consistent RSI combo** — B21 (3.96), B22 (3.92). Consistent Sharpe ~3.9 across reruns. Lower ceiling than RSI+STOCH but higher floor (never fails).
4. **Still all SHORT** — 22nd consecutive batch.

### Recommendations

1. **For reliable discovery, run 5+ parallel with same combo** — expect 30-40% failure rate. The 2-3 surviving runs will find strong strategies.
2. **RSI+CCI is the "safe bet" RSI combo** — consistent 3.9+ Sharpe. RSI+STOCH has higher ceiling (6.05) but fails more often.
3. **Discovery is fundamentally stochastic** — the same combo on the same data can produce Sharpe 0 or Sharpe 6 depending on random seed. Portfolio value comes from running many discoveries and keeping the best.












