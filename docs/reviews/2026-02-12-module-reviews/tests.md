# Code Review: Test Suite

**Reviewer:** Claude Opus 4.6
**Date:** 2026-02-12
**Scope:** All test files in `tests/`

---

## Test Health

| Metric | Value |
|--------|-------|
| Test files | 46 |
| Test classes | 287 |
| Test functions | 1238 |
| Collected | 1305 |
| Passing | 1283 |
| Failing | 18 |
| Skipped | 4 |
| **Pass rate** | **98.3%** |

---

## Failure Clusters

### Cluster 1: Dashboard settings page (4 failures)
Import-time crash due to missing `DEFAULT_DB_PATH`. Fix tracked in dashboard review H-1.

### Cluster 2: Validation runner (8 failures)
NT binary compatibility error (`Quantity size changed`). Fix tracked in validation review C-1 and project_config review H-2.

### Cluster 3: Discovery dashboard (6 failures)
Missing imports and FK constraint violations. Fix tracked in discovery review H-2 and db review C-2.

---

## Coverage Analysis

### Strong Coverage (>80%)
- DSL parser and compiler (62 test functions, 794 test lines)
- Risk sizing (40 test functions)
- Risk actors (55 test functions)
- Overfitting pipeline/DSR/WFA/Purged K-Fold (139 test functions across 5 files)
- Ethereal clients (80 test functions across 3 files)
- Paper trading (27+18+10+34 test functions across 4 files)
- Math precision (40 test functions with known-result cross-validation)

### Coverage Gaps
- No end-to-end dashboard smoke test that imports each page and performs critical actions
- No NT runtime compatibility preflight test
- No integration tests spanning multiple modules (e.g., screening -> overfitting -> validation pipeline)
- Template metadata functions have zero test coverage
- No tests for CLI entry points (`__main__.py` across modules)

---

## Findings

### HIGH (1)

#### H-1: No dashboard smoke test
All dashboard test failures are import-time crashes that a simple "import each page" smoke test would catch. Current gap allows regressions to ship.

**Fix:** Add smoke suite: import every page, exercise settings render, attempt discovery job launch.

### MEDIUM (2)

**M-1:** No NT compatibility preflight fixture. Expensive validation tests run and fail instead of skipping with clear reason.

**Fix:** Assert NT import compatibility in conftest, skip with explicit message if incompatible.

**M-2:** Metric unit contract tests missing. No enforcement that screening/validation/CLI use consistent fraction vs percentage semantics.

### LOW (1)

**L-1:** Lint coverage diluted by benchmark/test style rules mixed with production issues.

### INFO (3)

**I-1:** Very strong mathematical precision and known-result regression coverage. `test_math_precision.py` cross-validates optimized vs reference formulas.

**I-2:** Broad unit coverage across all major modules. 1238 test functions is substantial.

**I-3:** Good fixture patterns (temp DB, sample configs, mocked clients). Test isolation is generally clean.

---

## Summary

| Severity | Count |
|----------|-------|
| HIGH | 1 |
| MEDIUM | 2 |
| LOW | 1 |
| INFO | 3 |
| **Total** | **7** |

## Recommended Test Additions

1. **Dashboard smoke suite:** Import every page, exercise settings render, attempt discovery job launch with valid/invalid IDs.
2. **Validation preflight fixture:** Assert NT import compatibility, skip with reason if incompatible.
3. **Metric unit contract tests:** Enforce decimal-fraction semantics across screening, validation, CLI.
4. **Template metadata tests:** Validate all templates parse with DSL parser.
5. **Job FK behavior tests:** Explicit test that discovery cannot use orphan run IDs.
