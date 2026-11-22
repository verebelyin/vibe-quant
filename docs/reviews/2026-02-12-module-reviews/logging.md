# Code Review: `vibe_quant/logging` Module

**Reviewer:** Claude Opus 4.6
**Date:** 2026-02-12
**Scope:** Event logging system -- DuckDB writer, query, typed events

---

## Module Overview

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | ~15 | Module exports |
| `writer.py` | ~150 | Thread-safe DuckDB event writer |
| `query.py` | ~120 | DuckDB query builders |
| `events.py` | ~100 | Typed event dataclasses |
| **Total** | **~385** | |
| **Tests** | ~220 | `test_event_logging.py` |

---

## Findings

### MEDIUM (1)

**M-1:** DuckDB query builders use string interpolation for paths and filters (`query.py:60,63`). Low risk in local usage but parameterization would prevent issues if query inputs ever come from external sources.

### LOW (2)

**L-1:** `EventWriter.write_many` appears currently unused (`writer.py:86`). Dead code.

**L-2:** No schema version marker for event log evolution. Future format changes would break old log readers.

### INFO (2)

**I-1:** Typed event model is consistent and easy to extend.

**I-2:** Writer is thread-safe and supports context-managed cleanup. Clean design.

---

## Summary

| Severity | Count |
|----------|-------|
| MEDIUM | 1 |
| LOW | 2 |
| INFO | 2 |
| **Total** | **5** |

## Recommendations

**Priority 1:** Harden query construction for untrusted inputs.

**Priority 2:** Remove unused `write_many` or wire it into batch emitters.

**Priority 3:** Add schema version marker for event log evolution.
