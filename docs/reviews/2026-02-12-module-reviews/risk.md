# Code Review: `vibe_quant/risk` Module

**Reviewer:** Claude Opus 4.6
**Date:** 2026-02-12
**Scope:** Position sizing math, actor-based risk controls, circuit breakers

---

## Module Overview

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 84 | Lazy imports for NT-dependent classes |
| `config.py` | 116 | Risk config dataclasses with validation |
| `types.py` | 60 | Shared enums and event types |
| `sizing.py` | 439 | Position sizers (FixedFractional, Kelly, ATR) |
| `actors.py` | 32 | Re-export wrapper for actor modules |
| `strategy_actor.py` | 303 | Strategy-level risk monitoring Actor |
| `portfolio_actor.py` | 264 | Portfolio-level risk monitoring Actor |
| **Total** | **~1298** | |

---

## Findings

### HIGH (2)

#### H-1: Rounding mode violates conservative sizing contract
**File:** `sizing.py:186-188`

`_apply_limits` uses default `ROUND_HALF_EVEN` instead of `ROUND_DOWN`:

```python
quantize_str = "1." + "0" * size_precision if size_precision > 0 else "1"
return final_size.quantize(Decimal(quantize_str))
```

With `size_precision=2`, size `1.005` rounds to `1.01` (exceeds limit) instead of `1.00`.

**Impact:** Position sizes can round UP, breaching conservative sizing assumptions. SPEC Section 9 requires conservative sizing.

**Fix:** `return final_size.quantize(Decimal(quantize_str), rounding=ROUND_DOWN)`

#### H-2: Kelly formula silently clips negative f* without logging
**File:** `sizing.py:291-310`

`kelly_f` clips to `[0, 1]` silently. Negative Kelly fraction (unfavorable edge) returns zero with no user feedback. Strategy with negative expected value produces zero positions and user doesn't know why.

**Fix:** Add `logger.warning("Kelly f* negative (%.4f) - unfavorable edge", f_adjusted)` when clipping.

### MEDIUM (3)

**M-1:** `_halted_strategies` declared but never used (`portfolio_actor.py:85`). Portfolio actor halts ALL trading, doesn't track per-strategy halts.

**M-2:** Daily loss check skipped if `daily_start_equity <= 0` (`strategy_actor.py:176-180`). Edge case: actor starts with zero balance after liquidation.

**M-3:** Position count breach sets `WARNING` not `HALTED` (`strategy_actor.py:188-190`). Strategy can still open positions. Race condition allows exceeding limit by 1.

### LOW (3)

**L-1:** Exception logging inconsistency. `_update_position_count` catches exceptions silently, `_update_exposures` logs errors (`strategy_actor.py:264`, `portfolio_actor.py:239`).

**L-2:** `Quantity(float(final_size), ...)` in `sizing.py:265`. Decimal -> float can lose precision for very small sizes. Negligible for typical crypto (0-8 decimals).

**L-3:** Rounding/precision handling should be centralized and shared across any module emitting tradable quantities.

### INFO (2)

**I-1:** Fixed fractional, Kelly, and ATR formulas are explicit and thoroughly tested.

**I-2:** Actor-side portfolio/strategy circuit breakers are clearly implemented with clean state transitions.

---

## Summary

| Severity | Count |
|----------|-------|
| HIGH | 2 |
| MEDIUM | 3 |
| LOW | 3 |
| INFO | 2 |
| **Total** | **10** |

## Recommendations

**Priority 1 (Production Blocking):** Fix `ROUND_DOWN` in `_apply_limits` with boundary test coverage. Add Kelly logging for negative f*.

**Priority 2:** Remove `_halted_strategies` or implement per-strategy tracking. Add logging when daily loss check skipped. Document position count WARNING vs HALTED behavior.

**Priority 3:** Consistent exception logging across actors. Integration test: risk actor + sizing + NautilusTrader order flow. Centralize rounding/precision utility.
