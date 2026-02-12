# Code Review: Root Package and CLI

**Reviewer:** Claude Opus 4.6
**Date:** 2026-02-12
**Scope:** `vibe_quant/__init__.py`, `__main__.py`, `metrics.py`, CLI entry behavior

---

## Module Overview

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | ~20 | Package initialization |
| `__main__.py` | ~130 | CLI entry point with subcommands |
| `metrics.py` | 28 | Shared PerformanceMetrics base class |
| **Total** | **~178** | |

---

## Findings

### MEDIUM (2)

**M-1:** `ValidationRunner` not closed on all exception paths in CLI (`__main__.py:27,30`). Resource leak on failed validation runs.

**Fix:** Wrap runner lifecycle with `try/finally`.

**M-2:** CLI prints percentage symbols for values stored as decimal fractions (`__main__.py:34,37,39,86`). Misleading output if values aren't pre-multiplied.

### LOW (1)

**L-1:** Command forwarding mutates global `sys.argv` for data/screening commands (`__main__.py:103,120`). Fragile in embedded/in-process contexts.

**Fix:** Replace `sys.argv` mutation with direct argument passing.

### INFO (1)

**I-1:** `PerformanceMetrics` is a clean shared abstraction reused across screening/validation. Good DRY design.

---

## Summary

| Severity | Count |
|----------|-------|
| MEDIUM | 2 |
| LOW | 1 |
| INFO | 1 |
| **Total** | **4** |

## Recommendations

**Priority 1:** Wrap CLI runner lifecycle with `try/finally`. Define one canonical percent policy.

**Priority 2:** Replace `sys.argv` mutation with direct argument passing.
