# Code Review: `vibe_quant/db` Module

**Reviewer:** Claude Opus 4.6
**Date:** 2026-02-12
**Scope:** All files in vibe_quant/db/ and related tests

---

## Module Overview

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 7 | Module exports |
| `connection.py` | 35 | SQLite connection factory with WAL |
| `schema.py` | 174 | Database schema definitions |
| `state_manager.py` | 650 | CRUD operations for state DB |
| **Total** | **~866** | |
| **Tests** | 379 | `tests/unit/test_db.py` |

---

## Findings

### CRITICAL (2)

#### C-1: SQL injection via dynamic column insertion in state_manager
**File:** `state_manager.py:414-419`, `state_manager.py:458-463`

`save_backtest_result()` and `save_trade()` construct INSERT statements using string interpolation of dict keys without sanitization:

```python
columns = ["run_id"] + list(metrics.keys())  # User-controlled keys
cursor = self.conn.execute(
    f"INSERT INTO backtest_results ({', '.join(columns)}) VALUES ({placeholders})",
    values,
)
```

**Impact:** Attacker controlling dict keys could inject SQL. Same pattern in `save_trade()`, `save_sweep_result()`, `save_trades_batch()`, `save_sweep_results_batch()`.

**Fix:** Whitelist allowed columns and validate against schema before insertion.

#### C-2: Discovery job start with `run_id=0` violates FK constraint
**File:** Dashboard discovery flow -> `state_manager.py`

Discovery page uses placeholder `run_id=0` which violates `background_jobs.run_id -> backtest_runs.id` foreign key. Results in `IntegrityError` on launch.

**Fix:** Allocate real run IDs for discovery jobs or separate discovery jobs from backtest-run FK model.

### HIGH (1)

#### H-1: Missing index on backtest_runs.status
**File:** `schema.py:156-163`

Queries filtering by status (`list_backtest_runs(status=...)`) do full table scans.

**Fix:** `CREATE INDEX IF NOT EXISTS idx_backtest_runs_status ON backtest_runs(status);`

### MEDIUM (3)

**M-1:** Background job operations duplicated between `StateManager` (`state_manager.py:581+`) and `BacktestJobManager` (`jobs/manager.py`). Behavioral drift risk.

**M-2:** Status fields have no schema-level constraints (`schema.py:55,148`), allowing invalid values.

**M-3:** `connection.py` correctly enables WAL/busy_timeout/foreign_keys but docstring doesn't explain WHY WAL is critical for concurrent dashboard+background-job access.

### LOW (1)

**L-1:** No explicit migration/version scheme. Schema evolution risk grows with feature expansion.

### INFO (2)

**I-1:** WAL mode, foreign keys, and busy timeout correctly configured on every connection per CLAUDE.md conventions.

**I-2:** Result/trade/sweep persistence paths are broadly tested (379 test lines for 866 source lines).

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 2 |
| HIGH | 1 |
| MEDIUM | 3 |
| LOW | 1 |
| INFO | 2 |
| **Total** | **9** |

## Recommendations

**Priority 1 (URGENT):** Fix SQL injection by whitelisting column names. Audit all f-string SQL.

**Priority 2:** Introduce valid run identity model for discovery jobs. Add status index.

**Priority 3:** Unify job-table operations under one owner. Add status constraints and migration metadata.
