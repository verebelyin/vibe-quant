# Code Review: `vibe_quant/discovery` Module

**Reviewer:** Claude Opus 4.6
**Date:** 2026-02-11
**Module:** `vibe_quant/discovery/`
**Scope:** Genetic algorithm-based strategy discovery -- 5 source files + 6 test files

---

## Module Overview

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 28 | Public API exports |
| `fitness.py` | 341 | Multi-objective fitness scoring, Pareto ranking |
| `genome.py` | 460 | Chromosome/gene types, random generation, DSL conversion |
| `guardrails.py` | 270 | Overfitting filter integration |
| `operators.py` | 420 | Selection, crossover, mutation operators |
| `pipeline.py` | 534 | Discovery pipeline orchestration |
| **Total** | **~2053** | |

---

## Findings

### CRITICAL (4)

#### C-1: Dual type system -- `genome.py` and `operators.py` define incompatible types
**Files:** `genome.py` lines 35-55, `operators.py` lines 43-140

Two separate `StrategyGene` and `StrategyChromosome` classes exist with incompatible field names, mutability, and structure. `genome.py` types use frozen dataclasses with string conditions; `operators.py` types use mutable dataclasses with enum conditions. The pipeline uses `operators.py` types, but `genome.py` types appear in `fitness.py` type hints via `# type: ignore[arg-type]` comments.

**Impact:** Type safety completely broken across the module. 5+ `# type: ignore` comments suppress real errors.

**Fix:** Consolidate to a single canonical type system. Keep `operators.py` types (richer, mutable for mutation). Remove or re-export from `genome.py`.

#### C-2: STOCH parameter names inconsistent between genome.py and operators.py
**Files:** `genome.py` lines 92-95, `operators.py` lines 38-39

`genome.py`: `k_period`, `d_period`
`operators.py`: `period_k`, `period_d`

When the pipeline creates a chromosome via `operators.py` with `period_k: 14` and passes it through `genome.py`'s `chromosome_to_dsl()`, `gene.parameters.get("k_period", 14)` always gets the default, ignoring the evolved value.

**Impact:** STOCH indicators always use default period=14 regardless of evolution.

**Fix:** Standardize on one naming convention.

#### C-3: MACD parameter ranges in operators.py allow fast_period > slow_period
**Files:** `genome.py` lines 73-76, `operators.py` line 30

`operators.py`: `fast_period: (5, 30)`, `slow_period: (15, 60)`. Allows fast=30, slow=15 which is mathematically invalid for MACD.

**Impact:** Can generate invalid MACD configurations that produce zero/erratic output.

**Fix:** Add post-generation constraint ensuring `fast_period < slow_period`.

#### C-4: SL/TP scale inconsistency between genome.py and operators.py
**Files:** `genome.py` lines 217-218, `operators.py` lines 108-109

`genome.py`: SL range `0.01-0.15` (fraction), `operators.py`: SL range `0.5-10.0` (percentage). If an `operators.py` chromosome passes through `genome.py`'s conversion (`sl_pct * 100`), a 5.0% SL becomes 500% -- nonsensical.

**Impact:** Cross-module conversion produces 100x inflated SL/TP values.

**Fix:** Unify to one representation (fraction or percentage) everywhere.

### HIGH (6)

#### H-1: `_export_top_strategies` is dead code
**File:** `pipeline.py`, lines 307-329. Defined but never called by `run()`. SPEC says output should be DSL YAML but `run()` returns raw chromosomes.

#### H-2: Missing imports in dashboard discovery page cause NameError
**File:** `dashboard/pages/discovery.py`, lines 278, 296. `BacktestJobManager` and `DEFAULT_DB_PATH` used but never imported. Clicking "Start Discovery" crashes.

#### H-3: Convergence check may trigger prematurely
**File:** `pipeline.py`, lines 287-305. With exactly n+1 generations, compares last n against only generation 0. Unusually good random init triggers false convergence.

#### H-4: `_mutate_single_gene` mutates in place -- incompatible with frozen genes
**File:** `operators.py`, lines 374-401. Works with `operators.py`'s mutable types but would crash with `genome.py`'s frozen types.

#### H-5: Invalid chromosomes added to population after retry exhaustion
**File:** `pipeline.py`, lines 277-283. After 10 failed mutation retries, invalid child is added anyway.

**Fix:** Generate fresh random chromosome or skip.

#### H-6: `evaluate_population` does not handle backtest exceptions
**File:** `fitness.py`, lines 304-341. Single bad chromosome aborts entire generation evaluation.

**Fix:** Wrap each `backtest_fn` call in try/except. Assign zero fitness on failure.

### MEDIUM (11)

**M-1:** Duplicate indicator pools in `genome.py` (6 indicators) and `operators.py` (16 indicators). Dashboard shows 6, algorithm uses 16.

**M-2:** DSL conversion logic duplicated 3 times (`genome.py`, `pipeline.py`, `dashboard/pages/discovery.py`) with subtle differences.

**M-3:** `_normalize` and `_clamp` functions are dead code (`fitness.py` lines 78-91).

**M-4:** `_objectives` function is dead code (`fitness.py` lines 183-185).

**M-5:** `ObjectivesTuple` type alias unused (`fitness.py` line 180).

**M-6:** `type Chromosome = StrategyChromosome` alias adds confusion (`operators.py` line 141).

**M-7:** `Any` type used 16 times across module (violates CLAUDE.md).

**M-8:** No parallelization of fitness evaluation despite SPEC requiring parallel backtest. Sequential loop is the dominant bottleneck.

**M-9:** `_random_param_value` in `genome.py` is dead code (lines 162-166).

**M-10:** `pareto_rank` has O(n^3) worst-case complexity (`fitness.py` lines 219-280). Acceptable for current sizes but won't scale.

**M-11:** Dashboard discovery page has unconditional top-level `render_discovery_tab()` call (line 534). Runs on import.

### LOW (8)

**L-1:** `genome.py` `generate_random_chromosome` uses inconsistent RNG sources when `rng=None`.

**L-2:** BBANDS `std_dev` range differs between `genome.py` (1.0-3.0) and `operators.py` (1.0-4.0).

**L-3:** `_perturb` with `value=0.0` uses fixed absolute range 0.2, inappropriate for all thresholds.

**L-4:** Fragile threshold formatting with float equality check in `genome.py`.

**L-5:** `discovered_{id(chrom)}` produces non-deterministic names (memory address).

**L-6:** Dashboard missing `tournament_size` and `convergence_generations` config controls.

**L-7:** Dead code `_random_param_value` has no test coverage.

**L-8:** `evaluate_population` signature doesn't indicate parallelization safety.

### INFO (5)

**I-1:** Well-structured fitness score design. Multi-objective with reasonable weights (0.4 Sharpe, 0.3 MaxDD, 0.3 PF).

**I-2:** Thorough test coverage for core math. `test_math_precision.py` cross-validates optimized vs reference formulas.

**I-3:** Good separation in guardrails module. Configurable toggles with full transparency.

**I-4:** Pareto dominance implementation is mathematically correct. Tests cover corner cases.

**I-5:** Pre-computed inverse ranges in fitness avoid repeated division. Validated to 15 decimal places.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 4 |
| HIGH | 6 |
| MEDIUM | 11 |
| LOW | 8 |
| INFO | 5 |
| **Total** | **34** |

## Recommendations

### Priority 1: Consolidate Type Systems (C-1, C-2, C-3, C-4, H-4, M-1, M-2, L-2, L-5)
Single most impactful change. Root cause of 9+ findings. Keep `operators.py` as canonical, add `uid` field, unify param names, ranges, and SL/TP representation. Create single `chromosome_to_dsl` function.

### Priority 2: Fix Runtime Bugs (H-2, H-5, H-6)
Add missing dashboard imports. Handle invalid chromosomes after retry. Add exception handling in `evaluate_population`.

### Priority 3: Remove Dead Code (M-3, M-4, M-5, M-9)
Remove `_clamp`, `_normalize`, `_objectives`, `ObjectivesTuple`, `_random_param_value`.

### Priority 4: Performance (M-8)
Add multiprocessing to `evaluate_population` per SPEC requirements.

### Unresolved Questions
- Should `genome.py` be kept at all, or fully replaced by `operators.py`?
- Should convergence detection use running-best approach?
- What's target population size -- does O(n^3) Pareto ranking matter?
