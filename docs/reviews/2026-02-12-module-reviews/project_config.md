# Code Review: Project Configuration

**Reviewer:** Claude Opus 4.6
**Date:** 2026-02-12
**Scope:** `pyproject.toml`, quality gate configuration, dependency posture

---

## Overview

- **Package manager:** uv
- **Python:** 3.13
- **Toolchain:** ruff (linting), mypy (strict), pytest (testing)
- **Key dependency:** NautilusTrader (LGPL-3.0, narrowly pinned)
- **Tests:** 46 test files, 1238 test functions, 287 test classes

---

## Findings

### HIGH (2)

#### H-1: Quality gates not currently passing
**File:** `pyproject.toml` toolchain config

Configured strict quality bar is not met: `ruff` has ~30 errors, `mypy` has ~42 errors. CI enforcement would block all PRs.

**Fix:** Introduce two CI tracks: `blocking-runtime` (errors that crash at runtime) and `style-debt` (type annotations, unused imports).

#### H-2: NT binary compatibility not tested
**File:** Runtime environment

Validation runtime blocked by NautilusTrader binary compatibility error. Dependency pinning alone doesn't guarantee binary consistency across platforms/Python builds.

**Fix:** Add environment compatibility preflight check before expensive validation tests.

### MEDIUM (1)

**M-1:** Ruff includes benchmark/test-only style issues mixed with production runtime failures, which slows triage.

### LOW (1)

**L-1:** Mypy strict mode error backlog dilutes signal. Should be reduced in prioritized batches by module.

### INFO (2)

**I-1:** Clear toolchain with modern Python target. Good project structure.

**I-2:** Narrow NautilusTrader version pin range reduces accidental major drift.

---

## Summary

| Severity | Count |
|----------|-------|
| HIGH | 2 |
| MEDIUM | 1 |
| LOW | 1 |
| INFO | 2 |
| **Total** | **6** |

## Recommendations

**Priority 1:** Split CI into blocking-runtime and style-debt tracks. Add NT compatibility preflight.

**Priority 2:** Burn down mypy errors by module ownership (dashboard -> validation -> screening).

**Priority 3:** Separate ruff rules for test vs production code.
