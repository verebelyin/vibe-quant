# Gen-1 Chromosome Viability Improvement Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce gen-1 zero-fitness rate from ~60% to ~20% by fixing degenerate threshold ranges, `_perturb()` boundary bug, and adding threshold validation to `is_valid_chromosome()`.

**Architecture:** Three targeted fixes in `operators.py` and `genome.py` — widen MACD/ATR threshold ranges, fix `_perturb()` to use range-relative perturbation for zero values, and enforce threshold validation in `is_valid_chromosome()`. No new modules. All changes are backward-compatible.

**Tech Stack:** Python 3.13, pytest

---

### Task 1: Widen MACD Threshold Range

MACD's threshold range `(-0.005, 0.005)` is so narrow that random chromosomes almost never produce viable entry/exit signals. Journal evidence shows MACD is systematically eliminated by the GA. Widening 10x makes it viable.

**Files:**
- Modify: `vibe_quant/discovery/genome.py:97`
- Test: `tests/unit/test_discovery_operators.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_discovery_operators.py`:

```python
class TestThresholdRanges:
    def test_macd_threshold_range_wide_enough(self) -> None:
        """MACD threshold range must span at least 0.05 to produce viable signals."""
        from vibe_quant.discovery.operators import THRESHOLD_RANGES, _ensure_pool
        _ensure_pool()
        lo, hi = THRESHOLD_RANGES["MACD"]
        assert hi - lo >= 0.05, f"MACD range too narrow: ({lo}, {hi})"

    def test_atr_threshold_range_wide_enough(self) -> None:
        """ATR threshold range must span at least 0.05 to produce viable signals."""
        from vibe_quant.discovery.operators import THRESHOLD_RANGES, _ensure_pool
        _ensure_pool()
        lo, hi = THRESHOLD_RANGES["ATR"]
        assert hi - lo >= 0.05, f"ATR range too narrow: ({lo}, {hi})"
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_discovery_operators.py::TestThresholdRanges -v`
Expected: FAIL — MACD range is 0.01, ATR range is 0.029

**Step 3: Widen the ranges**

In `vibe_quant/discovery/genome.py:97`, change MACD's `default_threshold_range`:

```python
# Before:
default_threshold_range=(-0.005, 0.005),
# After:
default_threshold_range=(-0.05, 0.05),
```

In `vibe_quant/discovery/genome.py:104`, change ATR's `default_threshold_range`:

```python
# Before:
default_threshold_range=(0.001, 0.03),
# After:
default_threshold_range=(0.001, 0.08),
```

**IMPORTANT:** `THRESHOLD_RANGES` in `operators.py` is derived from `genome.py` via `_build_threshold_ranges()` at line 208-220. The genome MACD/ATR ranges flow through automatically. BUT `_build_threshold_ranges()` also has manual overrides for CCI/WILLR/MFI/ROC at lines 214-219 — these override the genome values. MACD and ATR are NOT overridden there, so the genome change is sufficient.

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_discovery_operators.py::TestThresholdRanges -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vibe_quant/discovery/genome.py tests/unit/test_discovery_operators.py
git commit -m "fix: widen MACD/ATR threshold ranges to reduce zero-fitness chromosomes (bd-qv9n)"
```

---

### Task 2: Fix `_perturb()` for Zero-Value Thresholds

When `value == 0.0`, `_perturb()` uses absolute `[-frac, frac]` = `[-0.2, 0.2]` as the perturbation range. For indicators with tiny valid ranges (MACD: -0.05 to 0.05), this pushes the threshold way out of bounds, and clamping pins it at the boundary forever. Fix: use `frac * (hi - lo)` as the perturbation scale when bounds are known.

**Files:**
- Modify: `vibe_quant/discovery/operators.py:273-294`
- Test: `tests/unit/test_discovery_operators.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_discovery_operators.py`:

```python
class TestPerturb:
    def test_perturb_zero_with_narrow_bounds(self) -> None:
        """_perturb(0.0) with narrow bounds should stay within bounds, not use absolute frac."""
        import random
        from vibe_quant.discovery.operators import _perturb
        random.seed(42)
        lo, hi = -0.05, 0.05
        results = [_perturb(0.0, 0.2, lo, hi) for _ in range(200)]
        # Should have variety (not all pinned at boundaries)
        unique = len(set(results))
        assert unique > 10, f"Only {unique} unique values — stuck at boundary"
        # All within bounds
        for r in results:
            assert lo <= r <= hi, f"Result {r} outside [{lo}, {hi}]"

    def test_perturb_zero_without_bounds_uses_absolute(self) -> None:
        """_perturb(0.0) without bounds should still use frac as absolute range."""
        import random
        from vibe_quant.discovery.operators import _perturb
        random.seed(42)
        results = [_perturb(0.0, 0.2) for _ in range(100)]
        # Should produce values in [-0.2, 0.2]
        assert any(r < -0.05 for r in results), "Should produce values below -0.05 when unbounded"
        assert any(r > 0.05 for r in results), "Should produce values above 0.05 when unbounded"
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_discovery_operators.py::TestPerturb -v`
Expected: `test_perturb_zero_with_narrow_bounds` FAILS — only 2-3 unique values because results get pinned at -0.05 or 0.05

**Step 3: Fix `_perturb()`**

Replace `vibe_quant/discovery/operators.py` lines 273-294:

```python
def _perturb(
    value: float, frac: float = 0.2, lo: float | None = None, hi: float | None = None
) -> float:
    """Perturb a value by +/- frac fraction. Optionally clamp to [lo, hi].

    When value is exactly 0.0 and bounds are provided, uses ``frac * (hi - lo)``
    as the perturbation scale so the result stays proportional to the valid range.
    When value is 0.0 without bounds, uses ``frac`` as an absolute range so
    genes can mutate away from zero.
    """
    if value == 0.0:
        if lo is not None and hi is not None:
            # Use fraction of the valid range, not absolute frac
            scale = (hi - lo) * frac
            result = random.uniform(-scale, scale)
        else:
            result = random.uniform(-frac, frac)
    else:
        delta = value * frac
        result = value + random.uniform(-delta, delta)
    if lo is not None and result < lo:
        result = lo
    if hi is not None and result > hi:
        result = hi
    return round(result, 4)
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_discovery_operators.py::TestPerturb -v`
Expected: PASS

**Step 5: Run all existing operator tests to check for regressions**

Run: `.venv/bin/pytest tests/unit/test_discovery_operators.py -v`
Expected: All pass

**Step 6: Commit**

```bash
git add vibe_quant/discovery/operators.py tests/unit/test_discovery_operators.py
git commit -m "fix: _perturb() uses range-relative scale for zero values with bounds (bd-qv9n)"
```

---

### Task 3: Add Threshold Validation to `is_valid_chromosome()`

`is_valid_chromosome()` already validates threshold ranges (lines 337-340). But `_random_chromosome()` and mutation can sometimes produce chromosomes that pass this check with extreme-but-valid thresholds. Add a repair step: when a mutated chromosome has out-of-range thresholds, clamp them instead of rejecting. This is done in the existing `_mutate_single_gene` indicator swap path — but crossover can also produce invalid combos.

Actually, re-reading the code — `is_valid_chromosome()` at line 337-340 already checks thresholds and returns False. And `mutate()` already resets thresholds on indicator swap (line 540-543). The real gap is: **crossover can combine parent A's threshold with parent B's indicator type**, creating out-of-range thresholds.

**Files:**
- Modify: `vibe_quant/discovery/operators.py` — add `_repair_chromosome()` function
- Test: `tests/unit/test_discovery_operators.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_discovery_operators.py`:

```python
class TestRepairChromosome:
    def test_repair_fixes_out_of_range_threshold(self) -> None:
        """Chromosomes with out-of-range thresholds should be repaired, not rejected."""
        from vibe_quant.discovery.operators import (
            StrategyGene, StrategyChromosome, ConditionType, Direction,
            is_valid_chromosome, _repair_chromosome, _ensure_pool, THRESHOLD_RANGES,
        )
        _ensure_pool()
        # ATR with RSI-scale threshold (impossible: ATR range is 0.001-0.08)
        bad_gene = StrategyGene(
            indicator_type="ATR", parameters={"period": 14.0},
            condition=ConditionType.GT, threshold=72.0,
        )
        chrom = StrategyChromosome(
            entry_genes=[bad_gene],
            exit_genes=[StrategyGene(
                indicator_type="RSI", parameters={"period": 14.0},
                condition=ConditionType.LT, threshold=50.0,
            )],
            stop_loss_pct=2.0, take_profit_pct=5.0, direction=Direction.LONG,
        )
        assert not is_valid_chromosome(chrom)
        repaired = _repair_chromosome(chrom)
        assert is_valid_chromosome(repaired)
        lo, hi = THRESHOLD_RANGES["ATR"]
        assert lo <= repaired.entry_genes[0].threshold <= hi

    def test_repair_preserves_valid_chromosome(self) -> None:
        """Valid chromosomes should pass through repair unchanged."""
        from vibe_quant.discovery.operators import (
            StrategyGene, StrategyChromosome, ConditionType, Direction,
            is_valid_chromosome, _repair_chromosome,
        )
        gene = StrategyGene(
            indicator_type="RSI", parameters={"period": 14.0},
            condition=ConditionType.GT, threshold=50.0,
        )
        chrom = StrategyChromosome(
            entry_genes=[gene],
            exit_genes=[gene.clone()],
            stop_loss_pct=2.0, take_profit_pct=5.0, direction=Direction.LONG,
        )
        assert is_valid_chromosome(chrom)
        repaired = _repair_chromosome(chrom)
        assert repaired.entry_genes[0].threshold == 50.0
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_discovery_operators.py::TestRepairChromosome -v`
Expected: FAIL — `_repair_chromosome` not found

**Step 3: Implement `_repair_chromosome()`**

Add after `is_valid_chromosome()` (around line 342) in `vibe_quant/discovery/operators.py`:

```python
def _repair_chromosome(chrom: StrategyChromosome) -> StrategyChromosome:
    """Repair a chromosome by clamping out-of-range thresholds and params.

    Instead of rejecting invalid chromosomes (wasting compute), this clamps
    thresholds to valid ranges and fixes parameter constraints. Called after
    crossover to handle cases where parent A's threshold gets paired with
    parent B's indicator type.
    """
    _ensure_pool()
    for gene in chrom.entry_genes + chrom.exit_genes:
        # Clamp threshold to valid range
        if gene.indicator_type in THRESHOLD_RANGES:
            tlo, thi = THRESHOLD_RANGES[gene.indicator_type]
            if gene.threshold < tlo or gene.threshold > thi:
                gene.threshold = round(random.uniform(tlo, thi), 4)
        # Fix MACD fast >= slow constraint
        if gene.indicator_type == "MACD":
            _enforce_param_constraints(gene.indicator_type, gene.parameters)
    # Clamp SL/TP
    chrom.stop_loss_pct = max(SL_RANGE[0], min(SL_RANGE[1], chrom.stop_loss_pct))
    chrom.take_profit_pct = max(TP_RANGE[0], min(TP_RANGE[1], chrom.take_profit_pct))
    return chrom
```

**Step 4: Wire `_repair_chromosome()` into crossover output**

In `vibe_quant/discovery/operators.py`, find the `crossover()` function (line 349). It returns two children at the end. Before returning, apply repair:

Find the return statement in `crossover()` (should be around line 430-440, returning `(child_a, child_b)`). Wrap both children:

```python
    return _repair_chromosome(child_a), _repair_chromosome(child_b)
```

**Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_discovery_operators.py::TestRepairChromosome -v`
Expected: PASS

**Step 6: Run all operator + pipeline tests**

Run: `.venv/bin/pytest tests/unit/test_discovery_operators.py tests/unit/test_discovery_pipeline.py -v`
Expected: All pass

**Step 7: Commit**

```bash
git add vibe_quant/discovery/operators.py tests/unit/test_discovery_operators.py
git commit -m "feat: repair chromosomes after crossover instead of rejecting (bd-qv9n)"
```

---

### Task 4: Verify Gen-1 Improvement with Mock Backtest

Run a before/after comparison using mock backtest to verify the zero-fitness rate dropped.

**Files:**
- Test: `tests/unit/test_discovery_operators.py`

**Step 1: Write the viability test**

Add to `tests/unit/test_discovery_operators.py`:

```python
class TestGen1Viability:
    def test_random_chromosomes_mostly_valid_thresholds(self) -> None:
        """At least 80% of random chromosomes should have all thresholds in valid ranges."""
        import random
        from vibe_quant.discovery.operators import (
            initialize_population, is_valid_chromosome,
        )
        random.seed(42)
        pop = initialize_population(100)
        valid_count = sum(1 for c in pop if is_valid_chromosome(c))
        assert valid_count == 100, f"Only {valid_count}/100 chromosomes valid"

    def test_crossover_produces_valid_offspring(self) -> None:
        """Crossover + repair should always produce valid chromosomes."""
        import random
        from vibe_quant.discovery.operators import (
            initialize_population, crossover, is_valid_chromosome,
        )
        random.seed(42)
        pop = initialize_population(20)
        invalid_count = 0
        for i in range(0, len(pop) - 1, 2):
            c1, c2 = crossover(pop[i], pop[i + 1])
            if not is_valid_chromosome(c1):
                invalid_count += 1
            if not is_valid_chromosome(c2):
                invalid_count += 1
        assert invalid_count == 0, f"{invalid_count} invalid offspring from crossover"
```

**Step 2: Run test**

Run: `.venv/bin/pytest tests/unit/test_discovery_operators.py::TestGen1Viability -v`
Expected: PASS

**Step 3: Run full test suite**

Run: `.venv/bin/pytest tests/unit/ -v --tb=short 2>&1 | tail -20`
Expected: All pass, no regressions

**Step 4: Commit**

```bash
git add tests/unit/test_discovery_operators.py
git commit -m "test: gen-1 viability assertions for random chromosomes and crossover (bd-qv9n)"
```

---

### Task 5: Lint + Close Bead

**Step 1: Run ruff**

Run: `.venv/bin/ruff check vibe_quant/discovery/operators.py vibe_quant/discovery/genome.py`
Expected: No errors. If errors, fix with `.venv/bin/ruff check --fix`

**Step 2: Run mypy**

Run: `.venv/bin/mypy vibe_quant/discovery/operators.py vibe_quant/discovery/genome.py`
Expected: No errors

**Step 3: Close bead**

```bash
bd close vibe-quant-qv9n
bd sync
```

**Step 4: Push**

```bash
git push
```

---

## Unresolved Questions

- Should MACD be removed from the indicator pool entirely? Journal shows it's always eliminated. Keeping for now with wider range.
- Should ROC range be widened too? Current `(-5.0, 5.0)` in `_build_threshold_ranges()` overrides genome's `(-10.0, 10.0)`. The operators override is narrower — intentional?
- Real NT backtest needed to measure actual zero-fitness rate improvement. Mock test only validates structural validity.
