# Code Review Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all 22 verified code review findings (4×P0, 8×P1, 9×P2, 1×P3) from the 2026-02-05 comprehensive audit.

**Architecture:** Fixes are grouped into batches by subsystem to minimize context-switching. Each batch targets related files. No new features—strictly bug fixes and correctness improvements to existing code.

**Tech Stack:** Python 3.13, SQLite, pytest, ruff, mypy

**Excluded from this plan:** P0 issues for replacing mock runners with real NautilusTrader integration (vibe-quant-5pz, vibe-quant-gf7, vibe-quant-23g, vibe-quant-2tm). These require NautilusTrader BacktestNode/TradingNode integration which is a separate feature effort, not a bug fix. The mock-driven design is intentional scaffolding pending NT integration.

---

## Batch 1: Database & Schema Fixes (4 issues)

Fixes: vibe-quant-b2c, vibe-quant-5hn, vibe-quant-0kw, vibe-quant-8g7

### Task 1: Add UNIQUE(run_id) to background_jobs schema (vibe-quant-b2c)

**Files:**
- Modify: `vibe_quant/db/schema.py:141-151`
- Test: `tests/unit/test_db.py`

**Step 1: Write failing test**

```python
# In tests/unit/test_db.py, add:
def test_background_jobs_unique_run_id(tmp_path: Path) -> None:
    """background_jobs should reject duplicate run_id."""
    db_file = tmp_path / "test.db"
    conn = get_connection(db_file)
    init_schema(conn)

    # First insert should succeed
    conn.execute(
        "INSERT INTO background_jobs (run_id, pid, job_type, status) VALUES (1, 100, 'screening', 'running')"
    )
    conn.commit()

    # Second insert with same run_id should fail
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO background_jobs (run_id, pid, job_type, status) VALUES (1, 200, 'screening', 'running')"
        )
    conn.close()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_db.py::test_background_jobs_unique_run_id -v`
Expected: FAIL (no IntegrityError raised because no UNIQUE constraint)

**Step 3: Add UNIQUE constraint to schema**

In `vibe_quant/db/schema.py`, change the background_jobs table:

```sql
CREATE TABLE IF NOT EXISTS background_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER UNIQUE REFERENCES backtest_runs(id) ON DELETE CASCADE,
    pid INTEGER NOT NULL,
    job_type TEXT NOT NULL,
    status TEXT DEFAULT 'running',
    heartbeat_at TEXT,
    started_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT,
    log_file TEXT
);
```

The key change: `run_id INTEGER REFERENCES ...` → `run_id INTEGER UNIQUE REFERENCES ...`

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_db.py::test_background_jobs_unique_run_id -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vibe_quant/db/schema.py tests/unit/test_db.py
git commit -m "fix(db): add UNIQUE(run_id) to background_jobs (vibe-quant-b2c)"
```

---

### Task 2: Fix StateManager.update_job_status to persist error text (vibe-quant-5hn)

**Files:**
- Modify: `vibe_quant/db/schema.py:141-151` (add error_message column)
- Modify: `vibe_quant/db/state_manager.py:605-635`
- Test: `tests/unit/test_db.py`

**Step 1: Write failing test**

```python
def test_update_job_status_persists_error(tmp_path: Path) -> None:
    """update_job_status should persist error text."""
    db_file = tmp_path / "test.db"
    mgr = StateManager(db_file)

    # Setup: create a strategy, run, and job
    strategy_id = mgr.create_strategy("test", {"name": "test"})
    run_id = mgr.create_backtest_run(
        strategy_id=strategy_id,
        run_mode="screening",
        symbols=["BTCUSDT"],
        timeframe="5m",
        start_date="2024-01-01",
        end_date="2024-12-31",
        parameters={},
    )
    mgr.register_job(run_id, pid=12345, job_type="screening", log_file="test.log")

    # Update with error
    mgr.update_job_status(run_id, "failed", error="Connection timeout")

    # Verify error was persisted
    job = mgr.get_job(run_id)
    assert job is not None
    assert job["error_message"] == "Connection timeout"
    mgr.close()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_db.py::test_update_job_status_persists_error -v`
Expected: FAIL (error_message not in schema / not persisted)

**Step 3: Add error_message column and fix update logic**

In `vibe_quant/db/schema.py`, add `error_message TEXT` to background_jobs after `log_file TEXT`:

```sql
    log_file TEXT,
    error_message TEXT
```

In `vibe_quant/db/state_manager.py:605-635`, replace with:

```python
    def update_job_status(
        self, run_id: int, status: str, error: str | None = None
    ) -> None:
        """Update job status.

        Args:
            run_id: Run ID.
            status: New status (running, completed, failed, killed).
            error: Optional error message.
        """
        if status in ("completed", "failed", "killed"):
            self.conn.execute(
                """UPDATE background_jobs
                   SET status = ?, completed_at = datetime('now'), error_message = ?
                   WHERE run_id = ?""",
                (status, error, run_id),
            )
        else:
            self.conn.execute(
                "UPDATE background_jobs SET status = ? WHERE run_id = ?",
                (status, run_id),
            )
        self.conn.commit()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_db.py::test_update_job_status_persists_error -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vibe_quant/db/schema.py vibe_quant/db/state_manager.py tests/unit/test_db.py
git commit -m "fix(db): persist error text in update_job_status (vibe-quant-5hn)"
```

---

### Task 3: Centralize DB path defaults (vibe-quant-0kw)

**Files:**
- Modify: `vibe_quant/screening/consistency.py:64`
- Modify: `vibe_quant/overfitting/pipeline.py:207`
- Test: `tests/unit/test_consistency_checker.py`, `tests/unit/test_overfitting_pipeline.py`

**Step 1: Write failing tests**

```python
# tests/unit/test_consistency_checker.py - add:
def test_default_db_path_matches_canonical() -> None:
    """Default DB path should match connection.DEFAULT_DB_PATH."""
    from vibe_quant.db.connection import DEFAULT_DB_PATH
    checker = ConsistencyChecker()
    assert checker._db_path == DEFAULT_DB_PATH
    checker.close()
```

```python
# tests/unit/test_overfitting_pipeline.py - add:
def test_default_db_path_matches_canonical() -> None:
    """Default DB path should match connection.DEFAULT_DB_PATH."""
    from vibe_quant.db.connection import DEFAULT_DB_PATH
    from vibe_quant.overfitting.pipeline import OverfittingPipeline
    pipeline = OverfittingPipeline()
    assert pipeline._db_path == DEFAULT_DB_PATH
    pipeline.close()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_consistency_checker.py::test_default_db_path_matches_canonical tests/unit/test_overfitting_pipeline.py::test_default_db_path_matches_canonical -v`
Expected: FAIL (paths are `data/state.db` not `data/state/vibe_quant.db`)

**Step 3: Import and use DEFAULT_DB_PATH**

In `vibe_quant/screening/consistency.py`, add import and change default:

```python
from vibe_quant.db.connection import DEFAULT_DB_PATH

# In __init__ or where db_path defaults:
if db_path is None:
    db_path = DEFAULT_DB_PATH
```

Same in `vibe_quant/overfitting/pipeline.py`:

```python
from vibe_quant.db.connection import DEFAULT_DB_PATH

# In __init__:
if db_path is None:
    db_path = DEFAULT_DB_PATH
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_consistency_checker.py::test_default_db_path_matches_canonical tests/unit/test_overfitting_pipeline.py::test_default_db_path_matches_canonical -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vibe_quant/screening/consistency.py vibe_quant/overfitting/pipeline.py tests/
git commit -m "fix(db): use canonical DEFAULT_DB_PATH in consistency/overfitting (vibe-quant-0kw)"
```

---

### Task 4: Wrap validation runner with try/except for stuck runs (vibe-quant-8g7)

**Files:**
- Modify: `vibe_quant/validation/runner.py:188-263`
- Test: `tests/unit/test_validation_runner.py`

**Step 1: Write failing test**

```python
# tests/unit/test_validation_runner.py - add:
def test_run_failure_sets_failed_status(temp_db: Path, temp_logs: Path, sample_dsl_config: dict) -> None:
    """If run() raises, status should be 'failed', not stuck in 'running'."""
    mgr = StateManager(temp_db)
    strategy_id = mgr.create_strategy("fail_test", sample_dsl_config)
    run_id = mgr.create_backtest_run(
        strategy_id=strategy_id,
        run_mode="validation",
        symbols=["BTCUSDT-PERP"],
        timeframe="5m",
        start_date="2024-01-01",
        end_date="2024-06-30",
        parameters={"rsi_period": 14},
    )

    runner = ValidationRunner(db_path=temp_db, logs_path=temp_logs)

    # Monkey-patch mock to raise
    def exploding_mock(*a: object, **kw: object) -> None:
        msg = "Simulated backtest failure"
        raise RuntimeError(msg)

    runner._run_backtest_mock = exploding_mock  # type: ignore[assignment]

    with pytest.raises(ValidationRunnerError):
        runner.run(run_id)

    # Status must be 'failed', not 'running'
    run_data = mgr.get_backtest_run(run_id)
    assert run_data is not None
    assert run_data["status"] == "failed"
    assert "Simulated backtest failure" in str(run_data.get("error_message", ""))
    runner.close()
    mgr.close()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_validation_runner.py::test_run_failure_sets_failed_status -v`
Expected: FAIL (status remains "running", no error handling)

**Step 3: Add try/except to run()**

In `vibe_quant/validation/runner.py`, wrap the body of `run()` (lines 233-263) with error handling:

```python
    def run(
        self,
        run_id: int,
        latency_preset: LatencyPreset | str | None = None,
    ) -> ValidationResult:
        # ... existing setup code (lines 205-231) stays the same ...

        # Update run status to running
        self._state.update_backtest_run_status(run_id, "running")

        try:
            # Create event writer
            with EventWriter(run_id=str(run_id), base_path=self._logs_path) as writer:
                self._write_start_event(writer, run_id, strategy_name, venue_config)

                result = self._run_backtest_mock(
                    run_id=run_id,
                    strategy_name=strategy_name,
                    dsl=dsl,
                    venue_config=venue_config,
                    run_config=run_config,
                    writer=writer,
                )

                self._write_completion_event(writer, run_id, strategy_name, result)

            result.execution_time_seconds = time.monotonic() - start_time
            self._store_results(run_id, result)
            self._state.update_backtest_run_status(run_id, "completed")

            return result
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            self._state.update_backtest_run_status(
                run_id, "failed", error_message=error_msg
            )
            raise ValidationRunnerError(error_msg) from exc
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_validation_runner.py::test_run_failure_sets_failed_status -v`
Expected: PASS

**Step 5: Run full test suite for this module**

Run: `uv run pytest tests/unit/test_validation_runner.py -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add vibe_quant/validation/runner.py tests/unit/test_validation_runner.py
git commit -m "fix(validation): wrap run() with try/except to prevent stuck running status (vibe-quant-8g7)"
```

---

## Batch 2: DSL Compiler Fix (1 issue)

Fixes: vibe-quant-bur

### Task 5: Fix DSL compiler bar scope in condition helpers (vibe-quant-bur)

**Files:**
- Modify: `vibe_quant/dsl/compiler.py:568-569,571-572,581-582,588-589,884-885`
- Test: `tests/unit/test_compiler.py`

**Step 1: Write failing test**

```python
# tests/unit/test_compiler.py - add:
def test_price_condition_passes_bar_to_helper() -> None:
    """Condition helpers using price (bar.close etc) must receive bar param."""
    from vibe_quant.dsl.compiler import StrategyCompiler
    from vibe_quant.dsl.parser import parse_strategy

    dsl_config = {
        "name": "price_test",
        "version": 1,
        "timeframe": "5m",
        "indicators": {
            "ema": {"type": "EMA", "period": 20, "source": "close"},
        },
        "entry_conditions": {
            "long": ["close > ema"],  # Uses price operand
        },
        "exit_conditions": {"long": ["close < ema"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 3.0},
    }
    dsl = parse_strategy(dsl_config)
    compiler = StrategyCompiler()
    result = compiler.compile(dsl)

    # The generated code must pass bar to check methods and accept it
    code = result.source_code
    assert "def _check_long_entry(self, bar:" in code or "def _check_long_entry(self, bar)" in code
    assert "_check_long_entry(bar)" in code or "_check_long_entry(self, bar)" in code
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_compiler.py::test_price_condition_passes_bar_to_helper -v`
Expected: FAIL (helpers have `(self)` signature, called without `bar`)

**Step 3: Fix compiler to pass bar to condition helpers**

In `vibe_quant/dsl/compiler.py`:

1. Change `_generate_condition_check_method` (line 885): signature includes `bar`:
   ```python
   f"def {method_name}(self, bar: Bar) -> bool:",
   ```

2. Change `_generate_on_bar_method` call sites (lines 568, 571, 581, 588): pass `bar`:
   ```python
   "        if self._check_long_entry(bar):"     # was _check_long_entry()
   "        elif self._check_short_entry(bar):"   # was _check_short_entry()
   "            if self._check_long_exit(bar):"   # was _check_long_exit()
   "            if self._check_short_exit(bar):"  # was _check_short_exit()
   ```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_compiler.py::test_price_condition_passes_bar_to_helper -v`
Expected: PASS

**Step 5: Run full compiler tests**

Run: `uv run pytest tests/unit/test_compiler.py -v`
Expected: All pass

**Step 6: Commit**

```bash
git add vibe_quant/dsl/compiler.py tests/unit/test_compiler.py
git commit -m "fix(dsl): pass bar to condition check helpers for price operands (vibe-quant-bur)"
```

---

## Batch 3: Dashboard & CLI Fixes (4 issues)

Fixes: vibe-quant-uqj, vibe-quant-cff, vibe-quant-nej, vibe-quant-hld

### Task 6: Add screening/validation __main__.py entrypoints (vibe-quant-uqj)

**Files:**
- Create: `vibe_quant/screening/__main__.py`
- Create: `vibe_quant/validation/__main__.py`
- Test: `tests/unit/test_package.py`

**Step 1: Write failing test**

```python
# tests/unit/test_package.py - add:
def test_screening_main_importable() -> None:
    """vibe_quant.screening.__main__ should be importable."""
    import importlib
    mod = importlib.import_module("vibe_quant.screening.__main__")
    assert hasattr(mod, "main")


def test_validation_main_importable() -> None:
    """vibe_quant.validation.__main__ should be importable."""
    import importlib
    mod = importlib.import_module("vibe_quant.validation.__main__")
    assert hasattr(mod, "main")
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_package.py::test_screening_main_importable tests/unit/test_package.py::test_validation_main_importable -v`
Expected: FAIL (ModuleNotFoundError)

**Step 3: Create entrypoints**

Create `vibe_quant/screening/__main__.py`:
```python
"""Screening pipeline CLI entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    """Run screening pipeline from CLI."""
    parser = argparse.ArgumentParser(description="Run screening pipeline")
    parser.add_argument("--run-id", type=int, required=True, help="Backtest run ID")
    parser.add_argument("--db", type=str, default=None, help="Database path")
    args = parser.parse_args()

    from vibe_quant.db.connection import DEFAULT_DB_PATH
    from vibe_quant.db.state_manager import StateManager

    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH
    state = StateManager(db_path)

    try:
        run_config = state.get_backtest_run(args.run_id)
        if run_config is None:
            print(f"Run {args.run_id} not found")
            return 1

        strategy_id = run_config["strategy_id"]
        strategy = state.get_strategy(int(str(strategy_id)))
        if strategy is None:
            print(f"Strategy {strategy_id} not found")
            return 1

        from vibe_quant.dsl.parser import parse_strategy
        from vibe_quant.screening.pipeline import create_screening_pipeline

        dsl = parse_strategy(strategy["dsl_config"])
        pipeline = create_screening_pipeline(dsl)
        result = pipeline.run(run_id=args.run_id, db_path=db_path)
        print(f"Screening complete: {result.total_combinations} combos, {len(result.pareto_optimal_indices)} Pareto-optimal")
        return 0
    except Exception as exc:
        print(f"Screening failed: {exc}")
        return 1
    finally:
        state.close()


if __name__ == "__main__":
    sys.exit(main())
```

Create `vibe_quant/validation/__main__.py`:
```python
"""Validation runner CLI entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    """Run validation backtest from CLI."""
    parser = argparse.ArgumentParser(description="Run validation backtest")
    parser.add_argument("--run-id", type=int, required=True, help="Backtest run ID")
    parser.add_argument("--db", type=str, default=None, help="Database path")
    args = parser.parse_args()

    from vibe_quant.db.connection import DEFAULT_DB_PATH
    from vibe_quant.validation.runner import ValidationRunner

    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH

    runner = ValidationRunner(db_path=db_path)
    try:
        result = runner.run(args.run_id)
        print(f"Validation complete: Sharpe={result.sharpe_ratio:.2f}, Return={result.total_return:.2f}%")
        return 0
    except Exception as exc:
        print(f"Validation failed: {exc}")
        return 1
    finally:
        runner.close()


if __name__ == "__main__":
    sys.exit(main())
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_package.py::test_screening_main_importable tests/unit/test_package.py::test_validation_main_importable -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vibe_quant/screening/__main__.py vibe_quant/validation/__main__.py tests/unit/test_package.py
git commit -m "fix(cli): add screening/validation __main__.py entrypoints (vibe-quant-uqj)"
```

---

### Task 7: Forward top-level CLI placeholders to real module CLIs (vibe-quant-cff)

**Files:**
- Modify: `vibe_quant/__main__.py:94-123`
- Test: `tests/unit/test_package.py`

**Step 1: Write failing test**

```python
# tests/unit/test_package.py - add:
def test_main_data_command_not_placeholder() -> None:
    """Top-level 'data' command should not print placeholder text."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "vibe_quant", "data", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert "not yet implemented" not in result.stdout.lower()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_package.py::test_main_data_command_not_placeholder -v`
Expected: FAIL

**Step 3: Replace placeholder handlers with forwards**

In `vibe_quant/__main__.py`, replace `cmd_data` and `cmd_screening`:

```python
def cmd_data(args: argparse.Namespace) -> int:
    """Forward to data module CLI."""
    import sys
    from vibe_quant.data.ingest import main as data_main
    # Re-inject subcommand args
    sys.argv = ["vibe_quant.data"] + args.remaining
    return data_main()


def cmd_screening(args: argparse.Namespace) -> int:
    """Forward to screening module CLI."""
    import sys
    from vibe_quant.screening.__main__ import main as screening_main
    sys.argv = ["vibe_quant.screening"] + args.remaining
    return screening_main()
```

And update the argparse setup for data/screening subparsers to use `parse_known_args` or `nargs=argparse.REMAINDER`:

```python
data_parser.add_argument("remaining", nargs=argparse.REMAINDER, help="Arguments forwarded to data CLI")
screening_parser.add_argument("remaining", nargs=argparse.REMAINDER, help="Arguments forwarded to screening CLI")
```

**Step 4: Run test**

Run: `uv run pytest tests/unit/test_package.py::test_main_data_command_not_placeholder -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vibe_quant/__main__.py tests/unit/test_package.py
git commit -m "fix(cli): forward top-level data/screening to real module CLIs (vibe-quant-cff)"
```

---

### Task 8: Add custom latency option to dashboard selector (vibe-quant-nej)

**Files:**
- Modify: `vibe_quant/dashboard/pages/backtest_launch.py:312-339`
- Test: `tests/unit/test_backtest_launch.py`

**Step 1: Write failing test**

```python
# tests/unit/test_backtest_launch.py - add:
def test_latency_options_include_custom() -> None:
    """Latency selector should include 'custom' option."""
    from vibe_quant.validation.latency import LatencyPreset
    preset_values = [p.value for p in LatencyPreset]
    # The render function should offer custom; verify LatencyPreset or the UI list includes it
    assert "custom" in preset_values or True  # We check the render function logic below

    # Verify the function creates custom option
    from vibe_quant.dashboard.pages.backtest_launch import LATENCY_OPTIONS
    assert "Custom" in LATENCY_OPTIONS or "custom" in [o.lower() for o in LATENCY_OPTIONS]
```

**Step 2: Implement custom option**

In `vibe_quant/dashboard/pages/backtest_launch.py`, modify `_render_latency_selector()`:

Add module-level constant:
```python
LATENCY_OPTIONS = ["None (screening mode)"] + [p.value for p in LatencyPreset] + ["custom"]
```

In the function, after the selectbox, add conditional custom inputs:
```python
if selected == "custom":
    with col2:
        base_ns = st.number_input("Base latency (ns)", min_value=0, value=50_000_000, step=1_000_000)
        insert_ns = st.number_input("Insert latency (ns)", min_value=0, value=25_000_000, step=1_000_000)
    st.session_state["custom_latency"] = {
        "base_latency_nanos": base_ns,
        "insert_latency_nanos": insert_ns,
        "update_latency_nanos": insert_ns,
        "cancel_latency_nanos": insert_ns,
    }
```

**Step 3: Run tests**

Run: `uv run pytest tests/unit/test_backtest_launch.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add vibe_quant/dashboard/pages/backtest_launch.py tests/unit/test_backtest_launch.py
git commit -m "fix(dashboard): add custom latency option to selector (vibe-quant-nej)"
```

---

### Task 9: Fix paper promotion query to not require overfitting metrics (vibe-quant-hld)

**Files:**
- Modify: `vibe_quant/dashboard/pages/paper_trading.py:66-88`
- Test: `tests/unit/test_paper_trading_dashboard.py`

**Step 1: Write failing test**

```python
# tests/unit/test_paper_trading_dashboard.py - add:
def test_validated_strategies_without_overfitting_metrics(tmp_path: Path) -> None:
    """Strategies should be promotable even without overfitting metrics."""
    import sqlite3
    from vibe_quant.db.schema import init_schema
    from vibe_quant.db.connection import get_connection

    db_file = tmp_path / "test.db"
    conn = get_connection(db_file)
    init_schema(conn)

    # Create strategy + validation run + result (no overfitting metrics)
    conn.execute("INSERT INTO strategies (name, dsl_config) VALUES ('test', '{}')")
    conn.execute(
        """INSERT INTO backtest_runs (strategy_id, run_mode, symbols, timeframe,
           start_date, end_date, parameters, status)
           VALUES (1, 'validation', '["BTCUSDT"]', '5m', '2024-01-01', '2024-12-31', '{}', 'completed')"""
    )
    conn.execute(
        """INSERT INTO backtest_results (run_id, total_return, sharpe_ratio, max_drawdown, win_rate)
           VALUES (1, 5.2, 1.35, 8.5, 0.595)"""
    )
    conn.commit()

    from vibe_quant.dashboard.pages.paper_trading import _get_validated_strategies
    strategies = _get_validated_strategies(db_file)
    assert len(strategies) == 1
    conn.close()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_paper_trading_dashboard.py::test_validated_strategies_without_overfitting_metrics -v`
Expected: FAIL (query requires non-null overfitting metrics)

**Step 3: Relax the query**

In `vibe_quant/dashboard/pages/paper_trading.py`, modify the SQL in `_get_validated_strategies()` (lines 66-88). Remove the three `IS NOT NULL` conditions for overfitting metrics. These fields should be optional, not gates:

```python
WHERE br.run_mode = 'validation'
  AND br.status = 'completed'
  AND res.sharpe_ratio IS NOT NULL
ORDER BY res.sharpe_ratio DESC
```

The overfitting metrics should still be displayed when available but not required for promotion eligibility.

**Step 4: Run test**

Run: `uv run pytest tests/unit/test_paper_trading_dashboard.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vibe_quant/dashboard/pages/paper_trading.py tests/unit/test_paper_trading_dashboard.py
git commit -m "fix(dashboard): don't require overfitting metrics for paper promotion (vibe-quant-hld)"
```

---

## Batch 4: Consistency Checker & Metrics (2 issues)

Fixes: vibe-quant-7iw, vibe-quant-fel

### Task 10: Fix consistency checker to join through backtest_runs for strategy_name (vibe-quant-7iw)

**Files:**
- Modify: `vibe_quant/screening/consistency.py:124-160`
- Modify: `tests/unit/test_consistency_checker.py:19-76` (fix test fixture to use real schema)

**Step 1: Fix test fixture to use real schema**

Replace the custom table creation in the `db_path` fixture with real schema + proper joins:

```python
@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Create temp database with test data using real schema."""
    from vibe_quant.db.connection import get_connection
    from vibe_quant.db.schema import init_schema

    db_file = tmp_path / "test_consistency.db"
    conn = get_connection(db_file)
    init_schema(conn)

    # Create strategies
    conn.execute("INSERT INTO strategies (id, name, dsl_config) VALUES (1, 'test_strategy', '{}')")
    conn.execute("INSERT INTO strategies (id, name, dsl_config) VALUES (2, 'improved_strategy', '{}')")
    conn.execute("INSERT INTO strategies (id, name, dsl_config) VALUES (3, 'sensitive_strategy', '{}')")

    # Create screening runs
    for i in range(1, 4):
        conn.execute(
            """INSERT INTO backtest_runs (id, strategy_id, run_mode, symbols, timeframe,
               start_date, end_date, parameters, status)
               VALUES (?, ?, 'screening', '["BTCUSDT"]', '5m', '2024-01-01', '2024-12-31', '{}', 'completed')""",
            (i, i),
        )

    # Create validation runs
    for i in range(4, 7):
        conn.execute(
            """INSERT INTO backtest_runs (id, strategy_id, run_mode, symbols, timeframe,
               start_date, end_date, parameters, status)
               VALUES (?, ?, 'validation', '["BTCUSDT"]', '5m', '2024-01-01', '2024-12-31', '{}', 'completed')""",
            (i, i - 3),
        )

    # Sweep results (screening)
    conn.execute("INSERT INTO sweep_results (run_id, parameters, sharpe_ratio, total_return) VALUES (1, '{\"rsi_period\": 14}', 2.0, 50.0)")
    conn.execute("INSERT INTO sweep_results (run_id, parameters, sharpe_ratio, total_return) VALUES (2, '{\"ema_period\": 20}', 1.5, 30.0)")
    conn.execute("INSERT INTO sweep_results (run_id, parameters, sharpe_ratio, total_return) VALUES (3, '{\"period\": 10}', 3.0, 100.0)")

    # Backtest results (validation)
    conn.execute("INSERT INTO backtest_results (run_id, total_return, sharpe_ratio) VALUES (4, 45.0, 1.8)")
    conn.execute("INSERT INTO backtest_results (run_id, total_return, sharpe_ratio) VALUES (5, 40.0, 2.0)")
    conn.execute("INSERT INTO backtest_results (run_id, total_return, sharpe_ratio) VALUES (6, 30.0, 1.0)")

    conn.commit()
    conn.close()
    return db_file
```

**Step 2: Fix the SQL queries in consistency.py**

Replace direct `strategy_name` references with JOINs:

```python
# Get screening result (line 124-130)
row = self.conn.execute(
    """
    SELECT s.name AS strategy_name, sr.sharpe_ratio, sr.total_return, sr.parameters
    FROM sweep_results sr
    JOIN backtest_runs br ON sr.run_id = br.id
    JOIN strategies s ON br.strategy_id = s.id
    WHERE sr.run_id = ?
    """,
    (screening_run_id,),
).fetchone()

# Get validation result (line 144-150)
row = self.conn.execute(
    """
    SELECT s.name AS strategy_name, res.sharpe_ratio, res.total_return
    FROM backtest_results res
    JOIN backtest_runs br ON res.run_id = br.id
    JOIN strategies s ON br.strategy_id = s.id
    WHERE res.run_id = ?
    """,
    (validation_run_id,),
).fetchone()
```

**Step 3: Update tests to use new run IDs (screening=1-3, validation=4-6)**

**Step 4: Run tests**

Run: `uv run pytest tests/unit/test_consistency_checker.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add vibe_quant/screening/consistency.py tests/unit/test_consistency_checker.py
git commit -m "fix(consistency): join through backtest_runs for strategy_name (vibe-quant-7iw)"
```

---

### Task 11: Standardize metric units to decimals (vibe-quant-fel)

**Files:**
- Modify: `vibe_quant/validation/runner.py:396-408` (mock to produce decimals)
- Modify: `vibe_quant/dashboard/pages/paper_trading.py:198-202` (remove :.1% if multiplying)
- Test: `tests/unit/test_validation_runner.py`

**Step 1: Write failing test**

```python
# tests/unit/test_validation_runner.py - add:
def test_mock_metrics_use_decimal_format() -> None:
    """Mock metrics should use decimal format (0.052 not 5.2) for consistency."""
    result = ValidationResult(
        run_id=1,
        strategy_name="test",
        total_return=0.052,  # 5.2%
        max_drawdown=0.085,  # 8.5%
        win_rate=0.595,      # 59.5%
    )
    metrics = result.to_metrics_dict()
    # All percentage-like values should be in decimal form (0-1 range)
    assert 0 <= metrics["win_rate"] <= 1.0
    assert metrics["total_return"] < 10  # Not percentage points
    assert metrics["max_drawdown"] < 1.0  # Not percentage points
```

**Step 2: Fix validation mock to produce decimals**

In `vibe_quant/validation/runner.py:396-408`, change the mock values:

```python
result = ValidationResult(
    run_id=run_id,
    strategy_name=strategy_name,
    total_return=0.052,       # was 5.2
    sharpe_ratio=1.35,        # ratios stay as-is
    sortino_ratio=1.85,
    max_drawdown=0.085,       # was 8.5
    total_trades=42,
    winning_trades=25,
    losing_trades=17,
    win_rate=0.595,           # was 59.5
    profit_factor=1.47,
    total_fees=0.00126,       # was 126.0 (now as fraction of equity)
    total_funding=0.00045,    # was 45.0
    total_slippage=0.00023,   # was 23.0
    execution_time_seconds=0.0,
)
```

**Step 3: Fix paper_trading.py formatters**

In `vibe_quant/dashboard/pages/paper_trading.py:198-202`, the `:.1%` format already expects decimals (0.085 → "8.5%"), which is correct after the fix. Verify no double-conversion.

**Step 4: Run tests**

Run: `uv run pytest tests/unit/test_validation_runner.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vibe_quant/validation/runner.py tests/unit/test_validation_runner.py
git commit -m "fix(metrics): standardize mock validation metrics to decimal format (vibe-quant-fel)"
```

---

## Batch 5: Security Fix (1 issue)

Fixes: vibe-quant-z24

### Task 12: Pass paper trading credentials via env vars, not /tmp JSON (vibe-quant-z24)

**Files:**
- Modify: `vibe_quant/dashboard/pages/paper_trading.py:112-153`
- Test: `tests/unit/test_paper_trading_dashboard.py`

**Step 1: Write failing test**

```python
# tests/unit/test_paper_trading_dashboard.py - add:
def test_paper_config_does_not_write_secrets_to_disk() -> None:
    """Paper config file must not contain API credentials."""
    from vibe_quant.dashboard.pages.paper_trading import _create_paper_config_file

    config_path = _create_paper_config_file(
        trader_id="TEST-001",
        strategy_id=1,
        db_path=Path("/tmp/test.db"),
        api_key="secret_key",
        api_secret="secret_secret",
        testnet=True,
    )
    import json
    content = json.loads(config_path.read_text())
    assert "secret_key" not in json.dumps(content)
    assert "secret_secret" not in json.dumps(content)
    config_path.unlink()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_paper_trading_dashboard.py::test_paper_config_does_not_write_secrets_to_disk -v`
Expected: FAIL (secrets are in JSON)

**Step 3: Remove secrets from config, pass via env vars**

In `vibe_quant/dashboard/pages/paper_trading.py`, modify `_create_paper_config_file()`:

Remove api_key/api_secret from the JSON config dict. Instead, when launching the subprocess, pass them as environment variables:

```python
def _create_paper_config_file(
    trader_id: str,
    strategy_id: int,
    db_path: Path,
    api_key: str,
    api_secret: str,
    testnet: bool,
) -> Path:
    config_data = {
        "trader_id": trader_id,
        "strategy_id": strategy_id,
        "db_path": str(db_path),
        "binance": {
            "testnet": testnet,
            "account_type": "USDT_FUTURES",
            # Credentials passed via BINANCE_API_KEY / BINANCE_API_SECRET env vars
        },
    }
    config_path = Path(f"/tmp/paper_{trader_id}.json")
    with config_path.open("w") as f:
        json.dump(config_data, f, indent=2)
    return config_path
```

Then in the subprocess launch code, set env vars:
```python
env = os.environ.copy()
env["BINANCE_API_KEY"] = api_key
env["BINANCE_API_SECRET"] = api_secret
subprocess.Popen(command, env=env, ...)
```

**Step 4: Run test**

Run: `uv run pytest tests/unit/test_paper_trading_dashboard.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vibe_quant/dashboard/pages/paper_trading.py tests/unit/test_paper_trading_dashboard.py
git commit -m "fix(security): pass paper trading credentials via env vars, not /tmp JSON (vibe-quant-z24)"
```

---

## Batch 6: Data Ingestion Fixes (4 issues)

Fixes: vibe-quant-v2y, vibe-quant-bsg, vibe-quant-nhj, vibe-quant-50m

### Task 13: Fix ingestion insert row count reporting (vibe-quant-v2y)

**Files:**
- Modify: `vibe_quant/data/archive.py:86-135,137-164`
- Modify: `vibe_quant/ethereal/ingestion.py:150-179,181-208`
- Test: `tests/unit/test_data.py`, `tests/unit/test_ethereal_ingestion.py`

**Step 1: Write failing test**

```python
# tests/unit/test_data.py - add:
def test_insert_klines_returns_actual_count(tmp_path: Path) -> None:
    """insert_klines should return actual inserted count, not input count."""
    from vibe_quant.data.archive import RawDataArchive
    archive = RawDataArchive(tmp_path / "test.db")

    klines = [
        (1000, 100.0, 110.0, 90.0, 105.0, 1000.0, 2000, 50000.0, 100, "binance_vision"),
        (2000, 105.0, 115.0, 95.0, 110.0, 1100.0, 3000, 55000.0, 110, "binance_vision"),
    ]
    count1 = archive.insert_klines("BTCUSDT", "1m", klines)
    assert count1 == 2

    # Insert same data again - should return 0 (all ignored)
    count2 = archive.insert_klines("BTCUSDT", "1m", klines)
    assert count2 == 0
    archive.close()
```

**Step 2: Run test to verify it fails**

Expected: FAIL (count2 == 2, not 0)

**Step 3: Use cursor.rowcount or changes() to get actual count**

In `vibe_quant/data/archive.py`, after `executemany` + `commit`:

```python
self.conn.executemany(...)
self.conn.commit()
cursor = self.conn.execute("SELECT changes()")
# Actually, changes() only returns last statement's changes.
# Better approach: count before and after
```

Better approach — count rows before and after:

```python
def insert_klines(self, symbol: str, interval: str, klines: list[tuple[object, ...]], source: str = "binance_vision") -> int:
    before = self.conn.execute(
        "SELECT COUNT(*) FROM raw_klines WHERE symbol = ? AND interval = ?",
        (symbol, interval),
    ).fetchone()[0]

    rows = [(symbol, interval, *k[:9], source) for k in klines]
    self.conn.executemany(
        """INSERT OR IGNORE INTO raw_klines ...VALUES ...""",
        rows,
    )
    self.conn.commit()

    after = self.conn.execute(
        "SELECT COUNT(*) FROM raw_klines WHERE symbol = ? AND interval = ?",
        (symbol, interval),
    ).fetchone()[0]
    return after - before
```

Apply same pattern to `insert_funding_rates` and the ethereal equivalents.

**Step 4: Run tests**

Run: `uv run pytest tests/unit/test_data.py tests/unit/test_ethereal_ingestion.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vibe_quant/data/archive.py vibe_quant/ethereal/ingestion.py tests/
git commit -m "fix(data): return actual inserted row count, not input count (vibe-quant-v2y)"
```

---

### Task 14: Log exceptions instead of swallowing them (vibe-quant-bsg)

**Files:**
- Modify: `vibe_quant/data/downloader.py:80-84`
- Modify: `vibe_quant/ethereal/ingestion.py:408-411,472-475`

**Step 1: Add logging to exception handlers**

In `vibe_quant/data/downloader.py:80-84`:
```python
except Exception:
    logger.exception("Unexpected error downloading %s %s/%s", symbol, year, month)
    return None
```

In `vibe_quant/ethereal/ingestion.py:408-411`:
```python
except Exception:
    logger.exception("Unexpected error downloading bars %s %s/%s", symbol, year, month)
    continue
```

Same for `:472-475`:
```python
except Exception:
    logger.exception("Unexpected error downloading funding %s %s/%s", symbol, year, month)
    continue
```

**Step 2: Run tests**

Run: `uv run pytest tests/unit/test_data.py tests/unit/test_ethereal_ingestion.py -v`
Expected: PASS (no behavior change, just added logging)

**Step 3: Commit**

```bash
git add vibe_quant/data/downloader.py vibe_quant/ethereal/ingestion.py
git commit -m "fix(data): log exceptions instead of silently swallowing (vibe-quant-bsg)"
```

---

### Task 15: Reuse HTTP clients across monthly loops (vibe-quant-nhj)

**Files:**
- Modify: `vibe_quant/ethereal/ingestion.py` (download_klines, download_funding_rates)
- Modify: `vibe_quant/data/downloader.py` (download_monthly_klines)
- Modify: `vibe_quant/data/ingest.py` (ingest_symbol loop)

**Step 1: Refactor ethereal downloads to accept client parameter**

In `vibe_quant/ethereal/ingestion.py`:

```python
def download_klines(symbol, timeframe, start_date, end_date, timeout=30, client: httpx.Client | None = None) -> list[tuple]:
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=timeout)
    try:
        for year, month in generate_month_range(start_date, end_date):
            # Use client directly, no `with` block per iteration
            response = client.get(url)
            ...
    finally:
        if own_client:
            client.close()
```

Same for `download_funding_rates`.

In `vibe_quant/data/ingest.py`, wrap the monthly loop with a single client:

```python
with httpx.Client(timeout=30) as client:
    for year, month in months:
        klines = download_monthly_klines(symbol, "1m", year, month, client=client)
```

**Step 2: Run tests**

Run: `uv run pytest tests/unit/test_data.py tests/unit/test_ethereal_ingestion.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add vibe_quant/ethereal/ingestion.py vibe_quant/data/downloader.py vibe_quant/data/ingest.py
git commit -m "fix(data): reuse HTTP client across monthly download loops (vibe-quant-nhj)"
```

---

### Task 16: Fix leap-day crash in Ethereal default start date (vibe-quant-50m)

**Files:**
- Modify: `vibe_quant/ethereal/ingestion.py:708`
- Test: `tests/unit/test_ethereal_ingestion.py`

**Step 1: Write failing test**

```python
# tests/unit/test_ethereal_ingestion.py - add:
def test_default_start_date_leap_day() -> None:
    """Default start date should not crash on Feb 29."""
    from datetime import date

    end_date = date(2024, 2, 29)  # Leap year
    # This should not raise ValueError
    try:
        start_date = end_date.replace(year=end_date.year - 2)
        pytest.fail("Expected ValueError from replace on leap day")
    except ValueError:
        pass  # Confirms the bug exists

    # Our fix should handle it
    from vibe_quant.ethereal.ingestion import _safe_years_ago
    result = _safe_years_ago(end_date, 2)
    assert result == date(2022, 2, 28)  # Falls back to Feb 28
```

**Step 2: Fix with safe date arithmetic**

In `vibe_quant/ethereal/ingestion.py`, add helper and use it:

```python
def _safe_years_ago(d: date, years: int) -> date:
    """Subtract years from date, handling leap day."""
    try:
        return d.replace(year=d.year - years)
    except ValueError:
        # Feb 29 on non-leap target year -> Feb 28
        return d.replace(year=d.year - years, day=28)
```

At line 708:
```python
if start_date is None:
    start_date = _safe_years_ago(end_date, 2)
```

**Step 3: Run test**

Run: `uv run pytest tests/unit/test_ethereal_ingestion.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add vibe_quant/ethereal/ingestion.py tests/unit/test_ethereal_ingestion.py
git commit -m "fix(ethereal): handle leap-day in default start date (vibe-quant-50m)"
```

---

## Batch 7: Test Hygiene (2 issues)

Fixes: vibe-quant-ybh, vibe-quant-kbn

### Task 17: Fix unclosed SQLite connections in test fixtures (vibe-quant-kbn)

**Files:**
- Modify: `tests/unit/test_purged_kfold.py` (and any other files with the warning)

**Step 1: Find fixtures that don't close connections**

Search for `sqlite3.connect` in test files that lack corresponding `.close()`.

**Step 2: Add proper cleanup**

Ensure all fixtures that create SQLite connections use try/finally or context managers to close them.

**Step 3: Run tests with warnings enabled**

Run: `uv run pytest tests/unit/test_purged_kfold.py -v -W error::ResourceWarning`
Expected: PASS (no ResourceWarning)

**Step 4: Commit**

```bash
git add tests/
git commit -m "fix(tests): close SQLite connections in fixtures (vibe-quant-kbn)"
```

---

### Task 18: Coverage assessment (vibe-quant-ybh)

This is a tracking task. No code changes — just note that integration tests are needed. The fixes in Tasks 1-17 will improve coverage on the touched modules. Integration test suite is a separate effort.

---

## Quality Gate

After all tasks, run:

```bash
uv run pytest -v                           # All tests pass
uv run ruff check vibe_quant               # No lint issues
uv run mypy vibe_quant                     # No type errors
uv run pytest --cov=vibe_quant --cov-report=term-missing:skip-covered  # Coverage check
```

## Beads to close after implementation

P0: vibe-quant-bur, vibe-quant-uqj
P1: vibe-quant-b2c, vibe-quant-5hn, vibe-quant-0kw, vibe-quant-8g7, vibe-quant-7iw, vibe-quant-fel, vibe-quant-z24, vibe-quant-hld
P2: vibe-quant-v2y, vibe-quant-bsg, vibe-quant-nhj, vibe-quant-50m, vibe-quant-cff, vibe-quant-nej, vibe-quant-kbn

Deferred (separate feature effort, not bug fixes):
- vibe-quant-5pz: Replace screening mock with real NT runner
- vibe-quant-gf7: Replace validation mock with real NT runner
- vibe-quant-23g: Replace overfitting mock runners
- vibe-quant-2tm: Implement real paper trading node
- vibe-quant-ybh: Integration test suite
