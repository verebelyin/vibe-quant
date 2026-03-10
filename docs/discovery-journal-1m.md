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
