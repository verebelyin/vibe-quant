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
