# Comprehensive Code Review (2026-02-05)

## Review Basis

Primary specification and conventions used for this review:

- `SPEC.md` (authoritative architecture + phase acceptance criteria)
- `README.md`
- `CLAUDE.md`
- `docs/claude/conventions.md`

Key expectation anchors used in findings:

- Single-engine, two-tier fidelity and realistic validation execution (`SPEC.md:89`, `SPEC.md:814`, `SPEC.md:1527-1567`, `README.md:16-30`)
- Dashboard background subprocess lifecycle (`SPEC.md:1168-1175`, `SPEC.md:1208-1241`)
- Data/state layout and canonical DB path (`SPEC.md:224-225`)
- Security rule: secrets in env vars only (`docs/claude/conventions.md:78-80`, `CLAUDE.md:40`)
- Quality gates (`CLAUDE.md:10-12`)

## Scope

Reviewed modules and cross-module behavior:

- `vibe_quant/dsl/*`
- `vibe_quant/screening/*`
- `vibe_quant/validation/*`
- `vibe_quant/overfitting/*`
- `vibe_quant/paper/*`
- `vibe_quant/dashboard/pages/*`
- `vibe_quant/jobs/*`
- `vibe_quant/db/*`
- `vibe_quant/data/*`
- `vibe_quant/ethereal/*`

## Quality Gate Baseline (Current)

Commands run on this codebase:

- `uv run pytest` -> `975 passed, 4 skipped`
- `uv run pytest --cov=vibe_quant --cov-report=term-missing:skip-covered` -> total coverage `59%`
- `uv run ruff check .` -> fails with `27` issues
- `uv run mypy vibe_quant tests` -> fails with `111` errors
- `.venv/bin/python -m vibe_quant.screening` -> fails (`No module named vibe_quant.screening.__main__`)
- `.venv/bin/python -m vibe_quant.validation` -> fails (`No module named vibe_quant.validation.__main__`)

## Executive Summary

The codebase has strong test breadth and many well-structured modules, but there are several high-impact runtime and architecture gaps against `SPEC.md` goals. The largest issues are:

1. Validation currently executes a mock path instead of full-fidelity backtesting.
2. Dashboard job launch command wiring is broken for screening/validation modules.
3. DSL compiler-generated condition methods can fail at runtime for price-based conditions.

These are blocking issues for trustworthy end-to-end behavior and for the “screening -> validation -> paper” workflow described in the spec.

---

## Findings (Ordered by Severity)

### [P0] Validation runner is still mock-driven, not full-fidelity execution

**Evidence**

- Main run path calls mock runner directly: `vibe_quant/validation/runner.py:241-249`
- Mock function is explicitly a placeholder: `vibe_quant/validation/runner.py:367-380`
- Spec requires full-fidelity validation with realistic fills/latency/costs: `SPEC.md:814`, `SPEC.md:1527-1567`

**Impact**

- Validation metrics are synthetic and cannot satisfy the spec’s realism and promotion intent.
- Screening-vs-validation comparisons are not trustworthy.

**Recommendation**

- Replace mock execution in `ValidationRunner.run()` with actual NautilusTrader engine integration.
- Keep mock path only behind an explicit test/dev flag.

---

### [P0] Dashboard launches backtest jobs using non-executable module targets

**Evidence**

- Dashboard command builder uses `python -m vibe_quant.<screening|validation>`: `vibe_quant/dashboard/pages/backtest_launch.py:411-419`
- No package entrypoints exist:
  - `vibe_quant/screening/` has no `__main__.py`
  - `vibe_quant/validation/` has no `__main__.py`
- Runtime verification failed:
  - `.venv/bin/python -m vibe_quant.screening` -> no `__main__`
  - `.venv/bin/python -m vibe_quant.validation` -> no `__main__`

**Impact**

- Backtest jobs launched from UI fail immediately.
- Violates spec expectation of dashboard-triggered background subprocess execution (`SPEC.md:1168-1175`, `SPEC.md:1208-1241`).

**Recommendation**

- Launch supported CLI paths (for example top-level `vibe_quant` subcommands), or add explicit `__main__.py` entrypoints for `screening` and `validation` packages.

---

### [P0] DSL compiler emits price-dependent condition code without `bar` in method scope

**Evidence**

- Entry/exit checks call helper methods without `bar`: `vibe_quant/dsl/compiler.py:568`, `vibe_quant/dsl/compiler.py:571`, `vibe_quant/dsl/compiler.py:581`, `vibe_quant/dsl/compiler.py:588`
- Generated condition methods are defined as `def ... (self) -> bool`: `vibe_quant/dsl/compiler.py:885`
- Price operands compile to `bar.<field>` access: `vibe_quant/dsl/compiler.py:954-957`
- DSL parser explicitly supports price operands (`open`, `high`, `low`, `close`, `volume`): `vibe_quant/dsl/conditions.py:79-83`

**Impact**

- Strategies using price in conditions can raise runtime `NameError`/scope errors.
- Breaks core DSL acceptance criterion (“DSL compiles to valid strategy that runs without errors”): `SPEC.md:1565`.

**Recommendation**

- Pass `bar` into condition helper methods, or inline price-dependent checks inside `on_bar` where `bar` is in scope.
- Add compiler regression tests with price-based entry and exit conditions.

---

### [P1] Compiled order sizing path is placeholder and exit sizing may be incorrect

**Evidence**

- Position sizing hardcoded to fixed `1.0` quantity: `vibe_quant/dsl/compiler.py:1037-1040`
- Exit order uses fresh calculated quantity with TODO note instead of tracked open size: `vibe_quant/dsl/compiler.py:1030`

**Impact**

- Does not satisfy spec requirement for real sizing/risk enforcement (`SPEC.md:1538-1543`, `SPEC.md:1567`).
- Exit quantity mismatch can lead to partial/unbalanced position handling.

**Recommendation**

- Integrate pluggable sizing module output into compiled strategy runtime.
- Track filled position quantity and use that exact value on exit.

---

### [P1] Screening consistency checker queries columns not present in production schema

**Evidence**

- Checker selects `strategy_name` from `sweep_results` and `backtest_results`: `vibe_quant/screening/consistency.py:126`, `vibe_quant/screening/consistency.py:146`
- Production schema does not include those columns:
  - `vibe_quant/db/schema.py:65-92` (`backtest_results`)
  - `vibe_quant/db/schema.py:121-137` (`sweep_results`)
- Tests mask mismatch by creating custom tables that do include `strategy_name`: `tests/unit/test_consistency_checker.py:27-44`

**Impact**

- Consistency checks can fail in real DB while tests pass.
- Directly conflicts with spec deliverable for screening-to-validation consistency checks (`SPEC.md:1557-1561`).

**Recommendation**

- Join through `backtest_runs` and `strategies` to resolve strategy name.
- Update tests to use production schema initialization path.

---

### [P1] Paper promotion query expects overfitting metrics that validation runner does not persist

**Evidence**

- Paper strategy selection requires non-null `deflated_sharpe`, `walk_forward_efficiency`, `purged_kfold_mean_sharpe`: `vibe_quant/dashboard/pages/paper_trading.py:83-85`
- Validation result persistence writes only `ValidationResult.to_metrics_dict()`: `vibe_quant/validation/runner.py:138-154`, `vibe_quant/validation/runner.py:532-541`
- Those overfitting fields are not included in `to_metrics_dict()`: `vibe_quant/validation/runner.py:138-154`

**Impact**

- “Start New Session” can show no eligible strategies even after successful validation.
- Breaks manual promotion workflow usability (`SPEC.md:1725-1729`).

**Recommendation**

- Unify metric model: either persist filter outputs into `backtest_results`, or drive promotion from filter result tables with explicit join keys.

---

### [P1] Secrets are written to plaintext JSON in `/tmp`

**Evidence**

- API key/secret serialized into config payload: `vibe_quant/dashboard/pages/paper_trading.py:127-128`
- File is written to `/tmp/paper_<trader_id>.json`: `vibe_quant/dashboard/pages/paper_trading.py:149-151`
- Project convention requires env-var-only secret handling: `docs/claude/conventions.md:78-80`, `CLAUDE.md:40`

**Impact**

- Credential exposure risk via filesystem artifacts.

**Recommendation**

- Pass credentials to subprocess via environment or secure IPC.
- Avoid plaintext secret material on disk.

---

### [P1] Job table upsert logic is inconsistent with schema guarantees

**Evidence**

- `background_jobs` has no `UNIQUE(run_id)`: `vibe_quant/db/schema.py:141-151`
- Manager uses `INSERT OR REPLACE` on `run_id`: `vibe_quant/jobs/manager.py:151-154`
- Lookup is `SELECT ... WHERE run_id = ?` then `fetchone()` with no deterministic ordering: `vibe_quant/jobs/manager.py:377-381`

**Impact**

- Multiple rows for same run are possible.
- Status reads may return stale/non-deterministic rows.

**Recommendation**

- Add `UNIQUE(run_id)` and migrate existing data.
- Keep explicit upsert semantics and deterministic read ordering.

---

### [P1] Paper trading runtime and dashboard controls are mostly placeholders/no-ops

**Evidence**

- Paper node uses placeholder dict instead of real TradingNode runtime: `vibe_quant/paper/node.py:302-308`
- Run loop explicitly marked placeholder: `vibe_quant/paper/node.py:313-315`
- Dashboard HALT/RESUME/CLOSE ALL handlers only display informational messages: `vibe_quant/dashboard/pages/paper_trading.py:529-539`

**Impact**

- Spec’s paper-trading operational goals and manual controls are not actually implemented (`SPEC.md:1693-1737`).

**Recommendation**

- Implement real node control signaling and action handlers wired to running process/state.

---

### [P1] `StateManager.update_job_status` ignores provided error message

**Evidence**

- `error` parameter exists: `vibe_quant/db/state_manager.py:605-607`
- Both `if error` and `else` branches execute identical SQL, never persisting error text: `vibe_quant/db/state_manager.py:616-629`

**Impact**

- Lost failure context in job-tracking layer.
- Harder operational debugging and UI diagnostics.

**Recommendation**

- Add error column update when `error` is provided.

---

### [P2] Default DB path diverges from canonical state DB layout

**Evidence**

- Canonical default: `data/state/vibe_quant.db` in connection factory: `vibe_quant/db/connection.py:7`
- Other modules default to `data/state.db`:
  - `vibe_quant/overfitting/pipeline.py:207`
  - `vibe_quant/screening/consistency.py:64`
- Spec storage layout defines `data/state/vibe_quant.db`: `SPEC.md:224-225`

**Impact**

- Different modules can silently operate on different DB files.

**Recommendation**

- Centralize all defaults on `DEFAULT_DB_PATH` from `vibe_quant/db/connection.py`.

---

### [P2] Ingestion insert methods over-report inserted row counts

**Evidence**

- `INSERT OR IGNORE` used, but returned value is `len(rows)` (attempted inserts):
  - `vibe_quant/data/archive.py:126-135`, `vibe_quant/data/archive.py:157-164`
  - `vibe_quant/ethereal/ingestion.py:171-179`, `vibe_quant/ethereal/ingestion.py:201-208`

**Impact**

- Metrics/logs can report inflated insert counts, misleading ingestion health and monitoring.

**Recommendation**

- Report actual inserted count via `SELECT changes()`/cursor deltas, or compare pre/post row counts.

---

### [P2] Broad exception swallowing in download/ingest paths hides data-quality failures

**Evidence**

- Binance downloader catches generic exceptions and returns `None`: `vibe_quant/data/downloader.py:83-84`
- Ethereal ingestion loops swallow generic exceptions and continue: `vibe_quant/ethereal/ingestion.py:410-411`, `vibe_quant/ethereal/ingestion.py:474-475`

**Impact**

- Parse/corruption/system faults can look like “no data available.”
- Weakens reproducibility and root-cause diagnosis.

**Recommendation**

- Log structured context (symbol/timeframe/month + error type).
- Distinguish expected `404` from unexpected failures.

---

### [P2] HTTP client lifecycle is inefficient in month-loop downloads

**Evidence**

- New `httpx.Client` created inside each month iteration for Ethereal kline/funding download loops: `vibe_quant/ethereal/ingestion.py:375-377`, `vibe_quant/ethereal/ingestion.py:446-448`
- Binance monthly downloader creates new client per month call: `vibe_quant/data/downloader.py:49`, invoked in loop at `vibe_quant/data/ingest.py:201-203`

**Impact**

- Extra connection setup overhead and reduced throughput on large ranges.

**Recommendation**

- Reuse one client per ingestion session and perform per-request calls inside the same session.

---

### [P2] Leap-day edge case in default date handling

**Evidence**

- Default start date derived via `replace(year=end_date.year - 2)`: `vibe_quant/ethereal/ingestion.py:708`

**Impact**

- Can raise `ValueError` on Feb 29 when target year is non-leap.

**Recommendation**

- Use timedelta-based fallback or guarded calendar logic.

---

### [P2] Top-level CLI still exposes placeholder `data` and `screening` commands

**Evidence**

- `cmd_data` and `cmd_screening` only print “not yet implemented”: `vibe_quant/__main__.py:93-122`

**Impact**

- Inconsistent with project’s “full lifecycle management” messaging and expected operational CLI paths (`README.md:121`, `SPEC.md` phase deliverables).

**Recommendation**

- Either wire real implementations or hide unfinished commands behind feature flags/internal commands.

---

### [P2] Instrument base-currency semantics appear inconsistent with configured symbols

**Evidence**

- Binance instrument configs include explicit base assets (`BTC`, `ETH`, `SOL`): `vibe_quant/data/catalog.py:29-68`
- Constructed instrument sets `base_currency=USDT` for all Binance symbols: `vibe_quant/data/catalog.py:95`
- Ethereal instruments set `base_currency=USDE` for all symbols: `vibe_quant/ethereal/instruments.py:108`

**Impact**

- Potential semantic mismatch in instrument metadata, with downstream effects on analytics/risk assumptions.

**Recommendation**

- Verify NautilusTrader `CryptoPerpetual` base/quote/settlement conventions and align to symbol semantics.

---

### [P2] Static quality gates are currently red and core/runtime coverage is below target

**Evidence**

- `ruff check .` fails with 27 issues.
- `mypy vibe_quant tests` fails with 111 errors (includes `vibe_quant/__main__.py:82`, `vibe_quant/__main__.py:223` plus extensive test typing debt).
- Coverage is `59%`, below target called out in project docs (`CLAUDE.md:10`).
- Low-coverage runtime-facing modules include:
  - `vibe_quant/dashboard/pages/backtest_launch.py` (8%)
  - `vibe_quant/dashboard/pages/paper_trading.py` (13%)
  - `vibe_quant/data/ingest.py` (4%)

**Impact**

- Weak confidence in orchestration and operational paths despite broad test count.

**Recommendation**

- Prioritize type/lint cleanup in shared runtime paths.
- Add integration tests for launch/promotion/paper control flows.

---

### [P3] Test run reports unclosed SQLite connections (resource warnings)

**Evidence**

- Coverage run emitted `ResourceWarning: unclosed database in <sqlite3.Connection ...>` during `tests/unit/test_paper_trading.py`.

**Impact**

- Usually a test hygiene issue, but can hide lifecycle bugs and make CI noise-prone.

**Recommendation**

- Ensure fixture teardown closes all DB connections consistently.

---

## Positive Observations

- Unit test suite breadth is strong (979 collected; 975 passing).
- Many quantitative modules are well-covered and defensively implemented (DSR/WFA/Purged K-Fold/sizing).
- Project package lint quality appears materially better than test-side quality debt (most Ruff failures concentrated in tests).
- WAL + busy-timeout usage is broadly consistent in core connection paths.

## Performance Review

- Current test runtime is good (~16-19s full unit suite).
- Throughput claims for validation cannot be trusted yet because validation execution is mock-driven.
- Most actionable performance wins today are:
  - client/session reuse in ingestion loops,
  - reduction of repeated setup work in monthly download paths,
  - increased coverage on orchestration modules where regressions are likely.

## Suggested Remediation Order

1. Fix validation execution path and dashboard subprocess entrypoints (P0).
2. Fix DSL compiler price-condition scope issue (P0).
3. Secure secret handling and repair job-tracking invariants (P1).
4. Align consistency/promotion data model across screening/validation/overfitting tables (P1).
5. Implement real paper-node controls and persist operational errors cleanly (P1).
6. Standardize DB path defaults, ingestion error/reporting correctness, and client reuse (P2).
7. Raise quality gates (mypy/ruff) and runtime coverage in dashboard/data orchestration modules (P2).
