# Code Review: `vibe_quant/jobs` Module

**Reviewer:** Claude Opus 4.6
**Date:** 2026-02-12
**Scope:** Background job management, heartbeat, process lifecycle

---

## Module Overview

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | ~10 | Module exports |
| `manager.py` | ~490 | BacktestJobManager with heartbeat/lifecycle |
| **Total** | **~500** | |
| **Tests** | ~220 | `test_backtest_job_manager.py` |

---

## Findings

### HIGH (1)

#### H-1: Discovery page FK constraint conflict
**File:** Dashboard discovery flow -> `manager.py`

Jobs require valid `backtest_runs.id` FK reference. Discovery page uses placeholder `run_id=0`, causing `IntegrityError` on job creation. Cross-cutting issue shared with db module (C-2).

**Fix:** Allocate real run IDs for discovery jobs or separate from backtest-run FK model.

### MEDIUM (2)

**M-1:** `run_with_heartbeat` starts infinite daemon thread with no explicit stop mechanism (`manager.py:487`). Lifecycle is process-bound but not explicitly terminated in long-lived hosts.

**M-2:** Job status strings are application-enforced but not schema-constrained. Invalid status values silently accepted.

### LOW (1)

**L-1:** Heartbeat thread return value discarded by callers (paper CLI, dashboard).

### INFO (2)

**I-1:** Heartbeat, stale detection, and kill/sync logic are clear and well-testable.

**I-2:** Process liveness checks handle macOS non-child process behavior gracefully.

---

## Summary

| Severity | Count |
|----------|-------|
| HIGH | 1 |
| MEDIUM | 2 |
| LOW | 1 |
| INFO | 2 |
| **Total** | **6** |

## Recommendations

**Priority 1:** Separate discovery jobs from backtest-run FK model or allocate real run IDs.

**Priority 2:** Add explicit stop signal for heartbeat thread. Add schema constraints for job status.

**Priority 3:** Clean up return value handling for heartbeat thread.
