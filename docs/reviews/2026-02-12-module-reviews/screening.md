# Code Review: `vibe_quant/screening` Module

**Reviewer:** Claude Opus 4.6
**Date:** 2026-02-12
**Scope:** Parallel parameter sweep, NT screening mode, result aggregation

---

## Module Overview

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 41 | Public API exports |
| `types.py` | 68 | Dataclasses for metrics, filters, results |
| `grid.py` | 149 | Parameter grid utilities, Pareto front |
| `pipeline.py` | 357 | Screening orchestration, parallel execution |
| `nt_runner.py` | 323 | NT BacktestNode runner for screening |
| `consistency.py` | 408 | Screening vs validation comparison |
| `__main__.py` | 66 | CLI entrypoint |
| **Total** | **~1412** | |

---

## Findings

### HIGH (3)

#### H-1: CLI path allows mock runner unintentionally
**File:** `__main__.py:41`, `pipeline.py:287`

CLI builds screening pipeline without forcing real runner, so mock metrics can be used unintentionally. Synthetic defaults can invalidate optimization conclusions in production contexts.

**Fix:** Make non-mock mode the CLI default.

#### H-2: ProcessPoolExecutor serialization failure risk
**File:** `pipeline.py:207-212`

`ProcessPoolExecutor` passes `self._runner` as closure. If `NTScreeningRunner` has cached state (`_compiled`, `_catalog_path` Path object), pickling may fail. "Can't pickle X" errors would silently fail all backtests.

**Fix:** Convert Path to string in `__init__`. Ensure all runner state is picklable.

#### H-3: Pareto front O(n^2) performance
**File:** `grid.py:116-148`

Nested loop with dominance check. For 10,000 results (common in parameter sweeps), this is 100M comparisons taking 1-100 seconds.

**Fix:** Pre-filter to top 1000 by Sharpe before Pareto, or use skyline algorithm (O(n log n)).

### MEDIUM (3)

**M-1:** NT runner type coupling imports `BacktestMetrics` from pipeline module (`nt_runner.py:16,151`), generating mypy warnings and cyclic coupling.

**M-2:** Error fallback stores only `sharpe=-inf` and loses diagnostic detail (`pipeline.py:228`). These values then enter Pareto computation.

**Fix:** Filter out `-inf` Sharpe before ranking. Persist richer failure diagnostics.

**M-3:** No timeout for individual backtests (`pipeline.py:215-227`). `future.result()` blocks indefinitely if a single backtest hangs. One pathological parameter combo stalls entire sweep.

**Fix:** `future.result(timeout=300)` and handle `TimeoutError`.

### LOW (2)

**L-1:** Unit conventions for returns/drawdown not explicit at screening output boundary. Downstream formatting drift risk.

**L-2:** Mock backtest uses MD5 for deterministic seeding (`pipeline.py:60-61`). Fine for mock but should be documented.

### INFO (2)

**I-1:** Pareto/ranking/filter logic is heavily tested. Parallel pipeline well-structured and resilient to partial worker errors.

**I-2:** Screening vs validation consistency checker is a strong quality gate concept.

---

## Summary

| Severity | Count |
|----------|-------|
| HIGH | 3 |
| MEDIUM | 3 |
| LOW | 2 |
| INFO | 2 |
| **Total** | **10** |

## Recommendations

**Priority 1 (Scalability):** Fix Path pickling. Optimize Pareto front. Add per-backtest timeout. Make non-mock mode CLI default.

**Priority 2 (Robustness):** Exclude `-inf` Sharpe before Pareto. Persist richer failure diagnostics. Refactor type dependencies.

**Priority 3:** Document unit conventions. Add integration test verifying ProcessPoolExecutor serialization.
