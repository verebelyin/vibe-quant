# Code Review: `vibe_quant/dsl` Module

**Reviewer:** Claude Opus 4.6
**Date:** 2026-02-12
**Scope:** All files in vibe_quant/dsl/ and related tests

---

## Module Overview

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 127 | Module exports |
| `schema.py` | 522 | Pydantic models for DSL |
| `parser.py` | 464 | YAML parsing and validation |
| `compiler.py` | 1062 | DSL to NautilusTrader codegen |
| `indicators.py` | 500 | Indicator registry |
| `conditions.py` | ~200 | Condition parsing |
| `indicator_metadata.py` | ~250 | Indicator catalog metadata |
| `templates.py` | ~150 | Code templates |
| **Total** | **~3275** | |
| **Tests** | ~830 | `test_dsl_parser.py` + `test_compiler.py` |

---

## Findings

### HIGH (1)

#### H-1: Unsafe exec() without sandboxing
**File:** `compiler.py:130`

`exec(source, module.__dict__)` executes dynamically generated Python code without sandboxing. Schema validation (`StrategyDSL.name` validates `^[a-z][a-z0-9_]*$`) provides some protection, but indicator parameters and condition values also flow into generated code. The `# noqa: S102` suppresses bandit without documenting mitigation.

**Impact:** If schema validation is bypassed or indicators contain crafted parameters, arbitrary execution possible.

**Fix:** Add AST validation before exec. Consider restricting `__builtins__` in exec namespace.

### MEDIUM (3)

**M-1:** Unsupported indicators without NT class emit placeholder `None` objects in generated code (`compiler.py:531,535`) instead of a strict compile error. Strategy runs with silent null indicators.

**M-2:** Sweep parameter validation (`parser.py:272-343`) checks indicator existence but doesn't validate parameter paths against indicator TYPE. `rsi.fast_period` would pass even though RSI has no `fast_period`.

**M-3:** Condition threshold naming collision in `compiler.py:344-356`. Two conditions with same indicator/value but different operators: first gets short name, second gets long name. More than 2 conditions can break.

### LOW (3)

**L-1:** Condition reference extraction swallows parse errors (`conditions.py:350`). Practical but invisible unless downstream validation catches it.

**L-2:** Compiler TODO for pandas-ta fallback remains open (`compiler.py:531`). Can mislead feature assumptions.

**L-3:** Missing multi-output indicator accessor validation. MACD outputs `macd`, `signal`, `histogram` but conditions reference `macd_ind` without dot-notation for specific output.

### INFO (2)

**I-1:** Pydantic schema validation is thorough with regex patterns, range validators, and cross-field validation. Strong guardrails for invalid strategies.

**I-2:** Test coverage is strong (830 lines). Parser edge cases well-covered.

---

## Summary

| Severity | Count |
|----------|-------|
| HIGH | 1 |
| MEDIUM | 3 |
| LOW | 3 |
| INFO | 2 |
| **Total** | **9** |

## Recommendations

**Priority 1:** Add AST validation before exec(). Convert placeholder generation into explicit compile-time failure.

**Priority 2:** Validate sweep parameters against indicator schema. Fix threshold naming.

**Priority 3:** Emit structured warnings for condition parse skips. Add multi-output accessor support. Implement or remove pandas-ta fallback signaling.
