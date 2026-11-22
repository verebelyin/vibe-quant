# Code Review: `vibe_quant/validation` Module

**Reviewer:** Claude Opus 4.6
**Date:** 2026-02-12
**Scope:** Full-fidelity backtest, FillModel/LatencyModel, cost modeling, SPEC alignment

---

## Module Overview

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 81 | Public API exports |
| `runner.py` | 682 | ValidationRunner orchestration |
| `fill_model.py` | 227 | VolumeSlippageFillModel, SlippageEstimator |
| `latency.py` | 131 | LatencyModelConfig presets |
| `venue.py` | 283 | VenueConfig for backtest venues |
| `results.py` | 134 | ValidationResult, TradeRecord dataclasses |
| `extraction.py` | 391 | Extract metrics/trades from NT output |
| `__main__.py` | 36 | CLI entrypoint |
| **Total** | **~1965** | |

---

## Findings

### CRITICAL (1)

#### C-1: Validation blocked by NT binary compatibility error
**File:** Runtime environment

NautilusTrader binary compatibility error (`Quantity size changed...`) blocks all validation execution. 8 test failures cluster here.

**Fix:** Resolve environment/runtime compatibility (NautilusTrader binary/Python version match).

### HIGH (3)

#### H-1: Slippage formula mismatch with NT engine
**File:** `fill_model.py:35-48`

`VolumeSlippageFillModel` stores `impact_coefficient` but NT's FillModel applies 1-tick slippage with probability `prob_slippage`. Post-fill, `SlippageEstimator` recalculates using SPEC Section 7 formula (`slippage = spread/2 + k * volatility * sqrt(order_size / avg_volume)`). Two different slippage values.

**Impact:** `TradeRecord.slippage_cost` doesn't match actual backtest fill behavior.

**Fix:** Document limitation. Consider subclassing `ExecutionModel` instead for SPEC-accurate slippage.

#### H-2: Walk-forward analysis not implemented
**File:** `runner.py` (entire file)

SPEC Section 7 explicitly requires walk-forward analysis capability. No implementation exists. `ValidationRunner.run()` runs a single backtest for entire date range.

**Impact:** Users cannot perform walk-forward validation. Increases overfitting risk.

**Fix:** Add `run_walk_forward()` with rolling train/test windows.

#### H-3: Runner assumes node.get_engine() always returns engine
**File:** `runner.py:399`

On failing paths, engine is `None` and raises secondary `AttributeError`, masking primary build/run errors and reducing debuggability.

**Fix:** Add explicit engine-null guard and preserve root exception details.

### MEDIUM (4)

**M-1:** `_create_venue_config` passes `float` to API typed as `int` (`runner.py:234`, `venue.py:106`).

**M-2:** `_parse_symbols` uses raw `json.loads` and can raise on malformed DB values (`runner.py:495`).

**M-3:** Metric unit conventions mixed across extraction, CLI, and tests (decimal fractions vs percentage points).

**M-4:** CAGR edge case: `total_return_frac == -1.0` (100% loss) leaves `cagr = 0.0` instead of `-1.0` or `-inf` (`extraction.py:361-375`).

### LOW (3)

**L-1:** `TradeRecord.exit_reason` always `"signal"` (`extraction.py:236`). Doesn't distinguish SL, TP, liquidation.

**L-2:** `TradeRecord.funding_fees` always `0.0`. NT models funding but extraction doesn't populate it.

**L-3:** Volatility uses trade-level variance, not daily equity returns (`extraction.py:377-387`). Not comparable to market benchmarks.

### INFO (2)

**I-1:** Extraction computes broad metric set (win/loss streaks, duration, CAGR, volatility, Calmar). Slippage model and fee extraction are explicit and test-backed.

**I-2:** Latency presets correctly match SPEC Section 7 (co-located 1ms, domestic 20ms, international 100ms, retail 200ms).

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 1 |
| HIGH | 3 |
| MEDIUM | 4 |
| LOW | 3 |
| INFO | 2 |
| **Total** | **13** |

## Recommendations

**Priority 1 (SPEC Compliance):** Resolve NT binary compatibility. Document slippage limitation. Implement walk-forward analysis.

**Priority 2 (Correctness):** Add engine-null guard. Fix CAGR edge case. Harden symbol parsing. Extract funding fees.

**Priority 3:** Define one metric-unit policy. Parse actual exit reasons. Add integration test with walk-forward config in DB schema.
