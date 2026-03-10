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
