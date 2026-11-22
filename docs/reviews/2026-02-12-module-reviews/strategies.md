# Code Review: `vibe_quant/strategies` Module

**Reviewer:** Claude Opus 4.6
**Date:** 2026-02-12
**Scope:** Strategy templates, metadata, and module structure

---

## Module Overview

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 6 | Empty module exports |
| `examples/__init__.py` | 6 | Empty module exports |
| `templates/__init__.py` | 2 | Template package marker |
| `templates/_metadata.py` | 199 | Template registry with 12 strategy templates |
| **Total** | **~213** | |
| **Tests** | 287 | Dashboard integration tests |

---

## Findings

### HIGH (2)

#### H-1: Template registry references 12 non-existent YAML files
**File:** `_metadata.py:43-178`

All 12 `TemplateMeta` entries reference YAML files (e.g., `rsi_mean_reversion.yaml`, `macd_crossover.yaml`) that don't exist in the templates directory. Dashboard "Load Template" feature crashes with `FileNotFoundError`.

**Impact:** Template functionality is completely non-functional.

**Fix:** Create the 12 YAML files, OR filter registry to only available templates with `get_available_templates()` that checks `path.exists()`.

#### H-2: TemplateMeta.load_dict() has no error handling
**File:** `_metadata.py:34-36`

```python
def load_dict(self) -> dict:
    return yaml.safe_load(self.path.read_text())  # No error handling
```

**Impact:** Crashes with unhandled `FileNotFoundError` or `yaml.YAMLError`.

**Fix:** Add error handling, validate returned type is dict.

### MEDIUM (3)

**M-1:** Template metadata typing is incomplete. `load_dict` returns untyped `dict`, causing strict mypy failures.

**M-2:** `get_template_by_filename` uses case-sensitive search (`_metadata.py:193-198`). Fails on "RSI_Mean_Reversion.yaml" vs "rsi_mean_reversion.yaml".

**M-3:** Template `instruments` field is free-text with inconsistent format ("BTC, ETH, SOL" vs "All major perpetuals"). No structured field for programmatic use.

### LOW (2)

**L-1:** Category/difficulty are free-form strings. Enum-backed validation would prevent taxonomy drift.

**L-2:** Template validity relies on downstream parser at use time. Pre-validation step missing.

### INFO (2)

**I-1:** Good breadth of template coverage (Momentum, Trend, Volatility, Multi-Timeframe, Volume). Rich metadata with descriptions and market conditions.

**I-2:** Module structure suggests future strategy base classes not yet implemented. Empty `__all__` lists consistent with phased implementation.

---

## Summary

| Severity | Count |
|----------|-------|
| HIGH | 2 |
| MEDIUM | 3 |
| LOW | 2 |
| INFO | 2 |
| **Total** | **9** |

## Recommendations

**Priority 1 (Critical):** Create 12 template YAML files OR filter registry to available templates only. Add error handling to `load_dict()`.

**Priority 2:** Add precise typing. Make filename lookup case-insensitive. Add structured instruments field.

**Priority 3:** Use enums for category/difficulty. Validate all templates in CI by parsing with DSL parser.
