# Code Review: `vibe_quant/alerts` Module

**Reviewer:** Claude Opus 4.6
**Date:** 2026-02-12
**Scope:** Telegram bot integration for paper trading alerts

---

## Module Overview

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 22 | Public API exports |
| `telegram.py` | 347 | Telegram bot with rate limiting |
| **Total** | **~369** | |
| **Tests** | 451 | `tests/unit/test_telegram_alerts.py` |

---

## Findings

### HIGH (2)

#### H-1: HTTP client resource leak on exceptions
**File:** `telegram.py:246-256`

`send_alert` catches `httpx.HTTPStatusError` and `httpx.RequestError` but doesn't ensure client cleanup. If exception occurs during first use, client may be left in inconsistent state.

**Fix:** Document that `await bot.close()` must be called in finally block, or add async context manager support (`__aenter__`/`__aexit__`).

#### H-2: Rate limit uses time.monotonic() -- incorrect after system sleep
**File:** `telegram.py:181-195`

`time.monotonic()` behavior varies across platforms during sleep. On wake, rate limits may immediately expire, causing alert storms.

**Fix:** Use `time.time()` (wall clock) for rate limiting since real-world time intervals matter.

### MEDIUM (3)

**M-1:** No retry logic for transient HTTP failures (`telegram.py:246-256`). 1-second network blip causes `send_alert` to return `False` and alert is permanently lost. Critical for error notifications per SPEC Section 12.

**Fix:** Add configurable retry with exponential backoff for critical alert types.

**M-2:** `DailySummary.format_message` uses `{self.total_pnl:.2f}` on Decimal values (`telegram.py:119-126`). May produce unexpected output for high-precision Decimals.

**Fix:** Explicitly `quantize()` before formatting.

**M-3:** `send_trade` doesn't validate price/quantity > 0 (`telegram.py:295-319`). Bug in caller could send confusing zero/negative price alerts.

### LOW (3)

**L-1:** Alert sending swallows HTTP/network failures into boolean `False` without structured error surface (`telegram.py:252,254`). No observability.

**L-2:** `_get_client` silently recreates client if previous was closed (`telegram.py:162-170`). Should log at debug/warning level.

**L-3:** `send_circuit_breaker` concatenates reason + details without explicit separator (`telegram.py:288-290`).

### INFO (3)

**I-1:** Clean async/await pattern with httpx. Rate limiting per alert type is well-designed.

**I-2:** Test coverage is excellent (451 lines for 369 source lines). 100% of public methods tested with mocks.

**I-3:** Environment-based configuration pattern is clean and secure.

---

## Summary

| Severity | Count |
|----------|-------|
| HIGH | 2 |
| MEDIUM | 3 |
| LOW | 3 |
| INFO | 3 |
| **Total** | **11** |

## Recommendations

**Priority 1 (Must Fix):** Add retry logic for critical alerts (errors, circuit breakers). Use `time.time()` for rate limiting. Document client lifecycle.

**Priority 2:** Quantize Decimal before formatting. Validate trade parameters. Return structured failure metadata.

**Priority 3:** Add debug logging for client recreation. Make rate limits configurable per alert type.
