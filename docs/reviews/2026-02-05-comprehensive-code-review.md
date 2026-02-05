# Comprehensive Code Review (2026-02-05, Revalidated)

## Review Basis

Primary sources used as acceptance baseline:

- `SPEC.md` (authoritative, per `CLAUDE.md:29`)
- `README.md`
- `CLAUDE.md`
- `docs/claude/conventions.md`

Key requirement anchors used in findings:

- Two-tier architecture and realistic execution expectations: `SPEC.md:679-715`, `SPEC.md:810-896`, `README.md:27-43`, `CLAUDE.md:21-24`
- Dashboard subprocess job flow: `SPEC.md:1162-1175`, `SPEC.md:1206-1270`
- Validation/screening CLI acceptance: `SPEC.md:1517`, `SPEC.md:1550`
- Paper trading architecture and controls: `SPEC.md:1276-1331`, `SPEC.md:1718-1729`
- SQLite/WAL and data-path conventions: `SPEC.md:224-225`, `SPEC.md:324-329`, `docs/claude/conventions.md:35-41`
- Secrets policy: `docs/claude/conventions.md:76-80`, `CLAUDE.md:40`
- Testing target: `SPEC.md:1410`, `CLAUDE.md:10`

## Scope

Deep audit pass covered:

- Runtime/orchestration modules: `vibe_quant/screening/*`, `vibe_quant/validation/*`, `vibe_quant/overfitting/*`, `vibe_quant/jobs/*`, `vibe_quant/dashboard/pages/*`, `vibe_quant/paper/*`
- Core strategy pipeline: `vibe_quant/dsl/*`, `vibe_quant/risk/*`
- Data/db integration: `vibe_quant/data/*`, `vibe_quant/db/*`, `vibe_quant/ethereal/*`
- Relevant tests under `tests/unit/*` for masking and coverage gaps

## Revalidation Commands

Commands run against current working tree:

- `uv run pytest` -> `1149 passed, 4 skipped`
- `uv run pytest --cov=vibe_quant --cov-report=term-missing:skip-covered` -> total coverage `63%`
- `.venv/bin/coverage report` -> confirms `63%` total
- `uv run ruff check .` -> fails with `27` issues (test files)
- `uv run ruff check vibe_quant` -> passes
- `uv run mypy vibe_quant tests` -> fails with `109` errors (test files)
- `uv run mypy vibe_quant` -> passes
- `.venv/bin/python -m vibe_quant.screening` -> fails (`No module named vibe_quant.screening.__main__`)
- `.venv/bin/python -m vibe_quant.validation` -> fails (`No module named vibe_quant.validation.__main__`)

## Executive Summary

The previous review was directionally right on major runtime blockers, but it is no longer fully accurate:

- Quality-gate numbers were stale (tests/coverage/type/lint counts changed materially).
- One major gap was missing: screening is also mock-driven by default (not just validation).
- Additional architecture mismatches remain around overfitting execution realism, CLI contracts, and metric-unit consistency.

The critical path from spec perspective is still blocked: screening -> validation -> dashboard launch -> paper promotion is not yet a trustworthy end-to-end real-execution workflow.

---

## Findings (Ordered by Severity)

### [P0] Screening pipeline is mock-driven; factory path cannot disable mock

**Evidence**

- Mock backtest implementation is the core runner: `vibe_quant/screening/pipeline.py:240-279`
- Pipeline defaults to mock when no runner provided: `vibe_quant/screening/pipeline.py:316`
- Factory `use_mock=False` still passes `None`, which falls back to mock runner: `vibe_quant/screening/pipeline.py:480-494`, `vibe_quant/screening/pipeline.py:316`

**Impact**

- Spec requirement for NautilusTrader screening execution is not met (`SPEC.md:700-708`, `SPEC.md:1517-1520`).
- All screening metrics can be synthetic, so downstream ranking/filtering/promotion decisions are not reliable.

**Recommendation**

- Add a real screening runner integration (BacktestNode/BacktestRunConfig) and make it default runtime behavior.
- Make `create_screening_pipeline(use_mock=False)` enforce non-mock runner wiring or fail fast.

---

### [P0] Validation runner still executes mocked backtests instead of full-fidelity validation

**Evidence**

- Main run path calls mocked implementation: `vibe_quant/validation/runner.py:241-249`
- Mock function explicitly marked placeholder: `vibe_quant/validation/runner.py:376-380`

**Impact**

- Violates Phase 3 validation goals (`SPEC.md:1527-1555`, `SPEC.md:1565-1569`).
- Validation metrics are not suitable for comparing latency/slippage/funding realism.

**Recommendation**

- Replace `_run_backtest_mock` in runtime path with NautilusTrader full-fidelity execution integration.
- Keep mock behind explicit test-only/dev flag.

---

### [P0] Dashboard backtest launch command targets are non-executable packages

**Evidence**

- Dashboard launches `python -m vibe_quant.screening` / `python -m vibe_quant.validation`: `vibe_quant/dashboard/pages/backtest_launch.py:411-419`
- Both packages lack `__main__.py` entrypoints (runtime confirmed by module execution failures).

**Impact**

- Backtests launched from UI fail immediately, breaking required background subprocess flow (`SPEC.md:1168-1175`, `SPEC.md:1208-1241`).

**Recommendation**

- Either add package entrypoints (`vibe_quant/screening/__main__.py`, `vibe_quant/validation/__main__.py`) or switch to a stable CLI command contract and update dashboard command builder accordingly.

---

### [P0] DSL compiler emits price-dependent conditions with `bar` out of scope

**Evidence**

- `on_bar` invokes condition helpers without `bar`: `vibe_quant/dsl/compiler.py:568-588`
- Generated helper signatures are `def ... (self) -> bool`: `vibe_quant/dsl/compiler.py:885`
- Price operands compile to `bar.<field>` expressions: `vibe_quant/dsl/compiler.py:954-957`

**Impact**

- Price-based DSL conditions can fail at runtime due missing `bar` in helper scope.
- Breaks acceptance criterion that compiled DSL runs without errors (`SPEC.md:1565`).

**Recommendation**

- Pass `bar` into condition helpers or inline bar-dependent expressions in `on_bar`.
- Add runtime execution tests for price-based entry/exit conditions.

---

### [P1] Overfitting pipeline defaults to mock WFA/CV runners in CLI/runtime path

**Evidence**

- CLI constructs `OverfittingPipeline` without real runners: `vibe_quant/overfitting/__main__.py:70-72`
- Pipeline defaults explicitly to mock runners if none provided: `vibe_quant/overfitting/pipeline.py:203-205`
- WFA uses `MockBacktestRunner` by default: `vibe_quant/overfitting/pipeline.py:285`
- Purged K-Fold path also defaults to `MockBacktestRunner`: `vibe_quant/overfitting/pipeline.py:344-348`

**Impact**

- Filter outputs can be synthetic, violating spec intent that WFA/CV are based on backtest behavior (`SPEC.md:1588-1599`).
- Promotion confidence from overfitting filters is overstated.

**Recommendation**

- Wire real backtest runners (screening-mode NT) into CLI default path.
- Fail fast if real runners are unavailable in non-test execution.

---

### [P1] Paper promotion query requires overfitting metrics that validation runner never writes

**Evidence**

- Paper strategy selection requires non-null `deflated_sharpe`, `walk_forward_efficiency`, `purged_kfold_mean_sharpe`: `vibe_quant/dashboard/pages/paper_trading.py:83-85`
- Validation persistence writes only `to_metrics_dict()` fields: `vibe_quant/validation/runner.py:138-154`, `vibe_quant/validation/runner.py:258`
- `to_metrics_dict()` omits those overfitting fields: `vibe_quant/validation/runner.py:138-154`

**Impact**

- “Start New Session” may show no eligible strategies even when validation completes.
- Blocks manual promotion workflow (`SPEC.md:1725-1729`).

**Recommendation**

- Persist overfitting outputs into `backtest_results` (or join to filter result tables) before promotion query.

---

### [P1] Secrets are serialized to plaintext JSON under `/tmp`

**Evidence**

- API credentials written into config payload: `vibe_quant/dashboard/pages/paper_trading.py:127-128`
- File written to `/tmp/paper_<trader_id>.json`: `vibe_quant/dashboard/pages/paper_trading.py:149-151`

**Impact**

- Violates env-var-only secret handling convention (`docs/claude/conventions.md:78-80`, `CLAUDE.md:40`).
- Increases credential exposure risk via local filesystem artifacts.

**Recommendation**

- Pass credentials via environment or secure IPC; avoid writing plaintext secrets to disk.

---

### [P1] Consistency checker queries schema columns that do not exist in production tables

**Evidence**

- Queries `strategy_name` directly from `sweep_results` and `backtest_results`: `vibe_quant/screening/consistency.py:126-127`, `vibe_quant/screening/consistency.py:146-147`
- Production schema lacks `strategy_name` in those tables: `vibe_quant/db/schema.py:65-96`, `vibe_quant/db/schema.py:121-138`
- Tests mask this by creating custom tables with `strategy_name`: `tests/unit/test_consistency_checker.py:27-45`

**Impact**

- Runtime checker can fail against actual DB schema while unit tests pass.
- Undermines Phase 3 consistency check deliverable (`SPEC.md:1557-1561`).

**Recommendation**

- Join through `backtest_runs` and `strategies` to resolve names.
- Rework tests to use real schema initialization.

---

### [P1] Background job upsert semantics are inconsistent with schema constraints

**Evidence**

- `background_jobs` has no `UNIQUE(run_id)`: `vibe_quant/db/schema.py:141-151`
- Manager uses `INSERT OR REPLACE` on `run_id`: `vibe_quant/jobs/manager.py:151-154`
- Fetch path assumes single row and returns first match: `vibe_quant/jobs/manager.py:377-381`

**Impact**

- Duplicate rows per run are possible; status reads can become non-deterministic/stale.

**Recommendation**

- Add `UNIQUE(run_id)` and migrate data.
- Keep deterministic reads with explicit ordering as safety net.

---

### [P1] Validation run failures can leave runs stuck in `running`

**Evidence**

- Status set to running before execution: `vibe_quant/validation/runner.py:233-234`
- Success path sets completed at end: `vibe_quant/validation/runner.py:260-261`
- No enclosing error handler to update failed status for exceptions after state transitions: `vibe_quant/validation/runner.py:236-263`

**Impact**

- Failed runs can remain `running` until stale cleanup logic intervenes.
- Dashboard status accuracy degrades; operational triage becomes harder.

**Recommendation**

- Wrap run body with `try/except/finally` and set `failed` plus error message on exceptions.

---

### [P1] Paper trading runtime/control path is still largely placeholder

**Evidence**

- Trading node is placeholder dict, not actual NT node runtime: `vibe_quant/paper/node.py:302-308`
- Main loop explicitly placeholder: `vibe_quant/paper/node.py:313-315`
- Dashboard HALT/RESUME/CLOSE ALL are informational no-ops: `vibe_quant/dashboard/pages/paper_trading.py:529-539`

**Impact**

- Manual control and runtime behavior do not match Phase 6 requirements (`SPEC.md:1697-1724`).

**Recommendation**

- Implement real node control signaling and action handlers wired to live process state.

---

### [P1] Metric units are inconsistent across screening, validation, and dashboard rendering

**Evidence**

- Screening mock produces decimal-style rates (`total_return` around -0.5..1.5, `win_rate` 0.3..0.9): `vibe_quant/screening/pipeline.py:262-265`
- Validation model/tests treat values as percentage points (`total_return=5.2`, `max_drawdown=8.5`, `win_rate=59.5`): `vibe_quant/validation/runner.py:105-113`, `vibe_quant/validation/runner.py:396-404`
- Dashboard formatter multiplies by 100 (`val * 100`): `vibe_quant/dashboard/pages/results_analysis.py:41`
- Paper tab formats validation values with percent formatter (`:.1%`): `vibe_quant/dashboard/pages/paper_trading.py:200-202`

**Impact**

- UI can display materially wrong values (e.g., `5.2` interpreted as `520%`).
- Cross-stage comparisons (screening vs validation) can be distorted by unit mismatch.

**Recommendation**

- Standardize all percentage-like metrics to one canonical unit (decimal fraction or percentage points) and enforce at persistence boundary with conversion tests.

---

### [P2] `StateManager.update_job_status` ignores provided error text

**Evidence**

- `error` argument exists: `vibe_quant/db/state_manager.py:605-607`
- Both branches execute identical SQL without persisting `error`: `vibe_quant/db/state_manager.py:616-629`

**Impact**

- Job failure context is dropped from this code path.

**Recommendation**

- Persist `error` into an error column (or remove unused parameter to avoid false expectations).

---

### [P2] Default DB path diverges from canonical state DB location

**Evidence**

- Canonical default path is `data/state/vibe_quant.db`: `vibe_quant/db/connection.py:7`
- Other modules default to `data/state.db`: `vibe_quant/screening/consistency.py:64`, `vibe_quant/overfitting/pipeline.py:207`

**Impact**

- Different workflows can silently read/write different DB files.

**Recommendation**

- Centralize defaults on `DEFAULT_DB_PATH` from `vibe_quant/db/connection.py`.

---

### [P2] Ingestion insert functions over-report inserted row counts

**Evidence**

- Uses `INSERT OR IGNORE` but returns attempted row count `len(rows)`:
  - `vibe_quant/data/archive.py:171-179`, `vibe_quant/data/archive.py:201-208`
  - `vibe_quant/ethereal/ingestion.py:171-179`, `vibe_quant/ethereal/ingestion.py:201-208`

**Impact**

- Ingestion metrics/logs can be inflated when duplicates are ignored.

**Recommendation**

- Return actual inserted rows via `SELECT changes()` or pre/post counts.

---

### [P2] Download/ingestion paths broadly swallow exceptions

**Evidence**

- Binance downloader catches all exceptions and returns `None`: `vibe_quant/data/downloader.py:83-84`
- Ethereal monthly loops swallow all exceptions and continue: `vibe_quant/ethereal/ingestion.py:410-411`, `vibe_quant/ethereal/ingestion.py:474-475`

**Impact**

- Corruption/parse/network issues can be misclassified as “no data.”

**Recommendation**

- Log structured error context and separate expected `404` from unexpected failures.

---

### [P2] HTTP clients are recreated inside per-month loops

**Evidence**

- Ethereal downloads create `httpx.Client` each month: `vibe_quant/ethereal/ingestion.py:376`, `vibe_quant/ethereal/ingestion.py:447`
- Binance downloader creates client per monthly fetch call: `vibe_quant/data/downloader.py:49` (called in loop at `vibe_quant/data/ingest.py:201-203`)

**Impact**

- Unnecessary connection setup overhead and lower ingestion throughput.

**Recommendation**

- Reuse one client per ingestion session.

---

### [P2] Leap-day edge case in default Ethereal start date

**Evidence**

- Default start date uses `end_date.replace(year=end_date.year - 2)`: `vibe_quant/ethereal/ingestion.py:708`

**Impact**

- Can raise `ValueError` on Feb 29 when target year is non-leap.

**Recommendation**

- Use safe date arithmetic with leap-day fallback.

---

### [P2] Top-level CLI exposes placeholder `data`/`screening` commands while module CLIs exist

**Evidence**

- Placeholder handlers: `vibe_quant/__main__.py:94-123`
- Real data CLI exists at `python -m vibe_quant.data`: `vibe_quant/data/__main__.py:3`, `vibe_quant/data/ingest.py:490-568`

**Impact**

- User-facing CLI contract is confusing and diverges from phase acceptance commands (`SPEC.md:1464-1467`, `SPEC.md:1517`).

**Recommendation**

- Either forward top-level commands to module CLIs or hide unfinished top-level commands.

---

### [P2] Dashboard latency selector omits required `custom` option

**Evidence**

- Spec includes `custom` in latency selector: `SPEC.md:1645`
- UI options include only `None` + enum presets: `vibe_quant/dashboard/pages/backtest_launch.py:319-339`

**Impact**

- Spec-promised custom latency configuration cannot be selected in launch flow.

**Recommendation**

- Add `custom` option and corresponding value inputs persisted into run config.

---

### [P2] Testing strategy gaps: strong unit suite, but integration coverage remains limited

**Evidence**

- Test tree contains only `tests/unit` (no integration suite folder)
- Overall coverage is `63%`, below target (`SPEC.md:1410`, `CLAUDE.md:10`)
- Very low-coverage orchestration modules include:
  - `vibe_quant/dashboard/pages/backtest_launch.py` (`8%`)
  - `vibe_quant/dashboard/pages/paper_trading.py` (`13%`)
  - `vibe_quant/data/ingest.py` (`4%`)

**Impact**

- Core pipeline glue paths remain under-tested despite broad unit count.

**Recommendation**

- Add integration tests for subprocess launch, CLI contracts, and screening->validation->promotion flow.

---

### [P3] Coverage run still emits unclosed SQLite connection warnings

**Evidence**

- `pytest --cov` emitted `ResourceWarning: unclosed database in <sqlite3.Connection ...>` (observed during `tests/unit/test_purged_kfold.py` execution context).

**Impact**

- Primarily test hygiene noise, but can hide lifecycle problems.

**Recommendation**

- Ensure fixture/teardown closes all temporary connections consistently.

---

## Positive Observations

- Source package static quality is solid: `ruff` and `mypy` pass for `vibe_quant`.
- Unit test breadth is substantial and currently green (`1149 passed, 4 skipped`).
- Many computational modules (DSR/WFA/PurgedKFold/sizing/discovery operators) are strongly covered and stable.
- WAL + busy-timeout configuration is consistently applied in core DB connection paths.

## Suggested Remediation Order

1. Replace mock runtime paths in screening + validation (P0).
2. Fix dashboard subprocess command contract and package entrypoints (P0).
3. Fix DSL price-condition scope bug (P0).
4. Replace mock defaults in overfitting runtime path (P1).
5. Resolve promotion pipeline blockers (overfitting metrics persistence, metric units) (P1).
6. Secure paper config secret handling and implement real paper controls (P1).
7. Repair DB/data integrity issues (consistency schema mismatch, job uniqueness/status flow, DB-path alignment) (P1/P2).
8. Expand integration testing for orchestration modules and close resource warnings (P2/P3).
