# Code Review: `vibe_quant/overfitting` Module

**Reviewer:** Claude Opus 4.6
**Date:** 2026-02-12
**Scope:** Statistical overfitting filters -- DSR, WFA, Purged K-Fold CV, pipeline

---

## Module Overview

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 36 | Public API exports |
| `pipeline.py` | 267 | Filter chain orchestration |
| `dsr.py` | 405 | Deflated Sharpe Ratio |
| `wfa.py` | 447 | Walk-Forward Analysis |
| `purged_kfold.py` | 418 | Purged K-Fold Cross-Validation |
| `cscv.py` | 310 | Combinatorial Symmetric CV |
| `__main__.py` | 220 | CLI entrypoint |
| **Total** | **~2103** | |
| **Tests** | ~2100 | 5 test files |

---

## Findings

### HIGH (3)

#### H-1: DSR uses hardcoded normal distribution assumption
**File:** `dsr.py` (moments lookup)

Deflated Sharpe Ratio requires skewness and kurtosis of returns. Current implementation uses `skewness=0.0` and `kurtosis=3.0` (normal distribution defaults) because `sweep_results` table doesn't store return moments.

**Impact:** For non-normal distributions (common in crypto), DSR may underestimate deflation. Overconfident pass decisions.

**Fix:** Document limitation or implement moment storage in sweep results.

#### H-2: WFA efficiency returns infinity when mean IS return is zero
**File:** `wfa.py:432-437`

When `mean_is_return == 0`, returns `float("inf")` for non-zero OOS return. This passes `is_robust` check (efficiency >= min_efficiency) even though zero IS performance means no predictive value.

**Impact:** Strategy with zero in-sample edge incorrectly passes robustness check.

**Fix:** Return `0.0` efficiency when `mean_is_return == 0`.

#### H-3: Purged K-Fold ignores indicator_lookback_bars from CVConfig
**File:** `purged_kfold.py:299-305`

`PurgedKFoldCV.__post_init__` doesn't pass `indicator_lookback_bars` to internal `PurgedKFold` instance. SPEC Section 8 requires purge period equal to max indicator lookback.

**Fix:** Pass `indicator_lookback_bars=self.config.indicator_lookback_bars` to `PurgedKFold`.

### MEDIUM (3)

**M-1:** Pipeline logs mock runner warning on every candidate in CV loop (`pipeline.py:222-227`). Reference comparison `if candidate == candidates[0]` may fail if list is modified.

**M-2:** CLI `parse_filters` doesn't validate unknown filter names (`__main__.py:23-46`). Typo "drs" instead of "dsr" silently produces WFA-only config.

**M-3:** WFA `generate_windows` uses `timedelta(days=N)` without accounting for market calendar. Acceptable for 24/7 crypto, but should be documented.

### LOW (2)

**L-1:** CLI output labels returns as percentages while upstream stores decimal-fraction values (`__main__.py:142,209`).

**L-2:** Strict typing/export hygiene has unresolved attr-defined warnings.

### INFO (3)

**I-1:** Core statistical methods (DSR, WFA, Purged K-Fold) correctly follow Lopez de Prado's methods. Tests are thorough.

**I-2:** Filter-chain design (toggleable DSR/WFA/CV) matches spec intent. Clean separation of concerns.

**I-3:** Test coverage is strong (2100 test lines for 2100 source lines). Edge cases well-covered.

---

## Summary

| Severity | Count |
|----------|-------|
| HIGH | 3 |
| MEDIUM | 3 |
| LOW | 2 |
| INFO | 3 |
| **Total** | **11** |

## Recommendations

**Priority 1 (Must Fix):** Document DSR normal distribution assumption or implement moment storage. Fix WFA efficiency for zero IS returns. Pass indicator_lookback_bars through CVConfig.

**Priority 2:** Validate filter names in CLI. Use flag for mock warning. Document calendar vs trading days.

**Priority 3:** Normalize return units in reports. Clean type/export boundaries.
