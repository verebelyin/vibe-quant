# Code Review: `vibe_quant/dashboard` Module

**Reviewer:** Claude Opus 4.6
**Date:** 2026-02-12
**Scope:** Complete dashboard module -- app.py, all pages, all components, charts.py, data_builders.py, utils.py

---

## Module Overview

| File | Lines | Purpose |
|------|-------|---------|
| `app.py` | 77 | Main entry point with st.navigation |
| `charts.py` | 653 | Plotly figure builders (pure functions) |
| `data_builders.py` | 404 | Data loading and metric computation (pure) |
| `utils.py` | 84 | Session state singletons and formatters |
| `pages/strategy_management.py` | 618 | Strategy CRUD with visual/YAML editor |
| `pages/discovery.py` | 535 | Genetic strategy discovery UI |
| `pages/backtest_launch.py` | 259 | Backtest configuration and job launch |
| `pages/results_analysis.py` | 722 | Results visualization and comparison |
| `pages/paper_trading.py` | 610 | Live trading monitoring and control |
| `pages/data_management.py` | 694 | Data ingestion and browser |
| `pages/settings.py` | 508 | Configuration management |
| `components/backtest_config.py` | 170 | Backtest selectors (strategy, date, latency) |
| `components/condition_builder.py` | 398 | Visual condition editor |
| `components/form_state.py` | 198 | Form state management helpers |
| `components/indicator_catalog.py` | ~200 | Indicator selector catalog |
| + 9 more component files | ~1200 | Strategy cards, risk panels, wizards, etc. |
| **Total** | **~6300** | |

---

## Findings

### CRITICAL (0)

No critical issues found.

### HIGH (3)

#### H-1: Missing `DEFAULT_DB_PATH` import causes NameError in discovery.py
**File:** `pages/discovery.py:296`

Line 296 references `DEFAULT_DB_PATH` which is never imported. Clicking "Start Discovery" crashes with `NameError`.

```python
296:    db_path = st.session_state.get("db_path", str(DEFAULT_DB_PATH))
```

**Impact:** Discovery job launch crashes 100% of the time.

**Fix:** Add `from vibe_quant.db.connection import DEFAULT_DB_PATH` at top of file.

#### H-2: Excessive `Any` usage violates project conventions (73 occurrences)
**Files:** 11 files use `: Any` type annotations

Project guidelines explicitly state "Never use the `any` type." Yet the dashboard module uses `Any` extensively across `charts.py`, `data_builders.py`, `form_state.py`, and multiple page/component files.

**Impact:** Type safety eliminated, IDE autocomplete degraded, refactoring unsafe.

**Fix:** Replace `Any` with specific types or `JsonDict` alias from `state_manager`.

#### H-3: Import-time execution in all page modules
**Files:** All 7 page files end with top-level render calls

All page modules call their render function at module bottom (e.g., `render_strategy_management_tab()` at line 617). With `st.navigation` API, Streamlit imports ALL pages at startup. Import-time execution means pages render before `pg.run()` is called.

**Impact:** Unnecessary computation, potential widget errors.

**Fix:** Remove all top-level render calls. Let `st.navigation` handle execution.

### MEDIUM (8)

**M-1:** Inconsistent error handling in `data_management.py:337-396`. Three subprocess runners catch bare `Exception` without logging. Error context lost on dashboard crash.

**M-2:** f-string SQL in `settings.py:432`: `cursor = manager.conn.execute(f"SELECT COUNT(*) FROM {table}")`. While `table` is from hardcoded list, pattern sets bad precedent. Flagged by bandit S608.

**M-3:** Unused variables in `condition_builder.py:139-143`. Variables `right`, `low`, `high`, `use_number` assigned defaults but may be unused depending on code path.

**M-4:** Missing type validation in `paper_trading.py:211-212`. `symbols_json` parsed without validating it's a string or list.

**M-5:** Overly broad `except Exception` in multiple locations: `data_management.py:93,108,356,395,433`, `results_analysis.py:377`, `settings.py:435`.

**M-6:** Hardcoded credentials in `paper_trading.py:196-207`. Form pre-fills API credentials from `os.environ` and stores in `st.text_input` value, appearing in Streamlit internal state.

**M-7:** Race condition in session state updates (`strategy_management.py:491,502,505`). Direct mutation of `st.session_state` dicts then `st.rerun()` is not atomic.

**M-8:** Missing null checks in chart builders (`charts.py:616-651`). `build_funding_pie` uses `.get()` with defaults but doesn't validate numeric operations succeed if values are unexpectedly `None`.

### LOW (12)

**L-1:** `charts.py:10` -- Import `Any` only used in function signatures.

**L-2:** `data_builders.py:388` -- Variable `total_days` computed but unused (F841).

**L-3:** `results_analysis.py:404` -- Complex rgba string generation in f-string. Extract to helper.

**L-4:** `backtest_launch.py:20` -- Unused import `LATENCY_OPTIONS`.

**L-5:** `strategy_management.py:48-50` -- Backward-compatible aliases `_validate_dsl` and `_validate_dsl_dict` unused.

**L-6:** `discovery.py:531` -- Alias `render = render_discovery_tab` unused.

**L-7:** `paper_trading.py:606` -- Alias `render = render_paper_trading_tab` unused.

**L-8:** `data_management.py:653` -- `render()` has different signature than other pages (no params).

**L-9:** `settings.py:452` -- Duplicate `import sys` at module level AND inside function.

**L-10:** `utils.py:22-41` -- `get_state_manager()` and `get_job_manager()` duplicate db_path retrieval logic.

**L-11:** `charts.py:565-612` -- `build_degradation_scatter()` matches by list index, not strategy ID.

**L-12:** `condition_builder.py:263` -- Missing "and" in between operator: `"{left} between {low} {high}"` should be `"{left} between {low} and {high}"`.

### INFO (5)

**I-1:** Excellent separation of concerns. `charts.py` and `data_builders.py` are pure functions with no Streamlit imports. Unit-testable and reusable.

**I-2:** Comprehensive error messages in form validation (`form_state.py:155-197`). Detailed per-field errors using Pydantic.

**I-3:** Proper `@st.fragment(run_every=5)` for auto-refresh in `paper_trading.py:343` and `discovery.py:444`.

**I-4:** WAL mode correctly enabled via `get_connection()` per project conventions.

**I-5:** All imports use `from __future__ import annotations` (PEP 563). Good practice.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 0 |
| HIGH | 3 |
| MEDIUM | 8 |
| LOW | 12 |
| INFO | 5 |
| **Total** | **28** |

## Recommendations

### Immediate (Before Next Deploy)
1. **Fix H-1**: Add `DEFAULT_DB_PATH` import to `discovery.py` (1-line fix, prevents crash)
2. **Fix H-3**: Remove all top-level render calls in page modules (7 lines deleted)

### Short-Term
3. **Address H-2**: Replace `Any` with `JsonDict` or specific types across all 11 files
4. **Fix M-1**: Add structured logging to subprocess error handlers
5. **Fix M-2**: Replace f-string SQL with safe concatenation
6. **Fix M-6**: Move credential defaults out of `st.text_input` value param

### Long-Term
7. **Address M-3 through M-8**: Refactor error handling, add null checks
8. **Clean up L-1 through L-12**: Remove unused imports, fix minor inconsistencies
9. **Increase test coverage**: `charts.py` and `data_builders.py` are pure and highly testable

### Architecture Notes
- **Overall quality**: HIGH. Well-organized, follows Streamlit best practices.
- **Main weakness**: Type safety (excessive `Any`) and import-time execution anti-pattern.
- **Strengths**: Pure data builders, comprehensive error messages, WAL mode, structured form state.
