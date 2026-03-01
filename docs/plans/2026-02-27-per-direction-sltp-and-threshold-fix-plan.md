# Per-Direction SL/TP + Mixed-Scale Threshold Fix — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add per-direction SL/TP overrides across the full stack (DSL→compiler→templates→genome→operators) and fix the mixed-scale threshold bug where mutation swaps indicator type without resetting threshold.

**Architecture:** Override-field approach — keep existing `stop_loss`/`take_profit` as defaults, add optional `stop_loss_long/short`, `take_profit_long/short`. Templates select config based on `is_long`. Genome gains per-direction SL/TP genes for `direction==BOTH` chromosomes.

**Tech Stack:** Python 3.13, Pydantic v2, NautilusTrader, pytest

---

### Task 1: Mixed-Scale Threshold Fix — Mutation Reset

**Files:**
- Modify: `vibe_quant/discovery/operators.py:467-495` (`_mutate_single_gene`)
- Test: `tests/unit/test_genetic_operators.py`

**Step 1: Write failing test**

Add to `tests/unit/test_genetic_operators.py`:

```python
class TestMutateThresholdReset:
    def test_indicator_swap_resets_threshold_to_valid_range(self) -> None:
        """When mutation swaps indicator type, threshold must be in new indicator's range."""
        from vibe_quant.discovery.operators import _THRESHOLD_RANGES

        random.seed(42)
        # Create gene with RSI threshold (25-75 range)
        gene = _make_gene("RSI", threshold=72.0, condition=ConditionType.GT)
        chrom = StrategyChromosome(
            entry_genes=[gene],
            exit_genes=[_make_gene()],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
        )

        # Mutate many times — after indicator swap, threshold should be in new range
        for _ in range(500):
            mutated = mutate(chrom, mutation_rate=1.0)
            for g in mutated.entry_genes:
                if g.indicator_type in _THRESHOLD_RANGES:
                    lo, hi = _THRESHOLD_RANGES[g.indicator_type]
                    assert lo <= g.threshold <= hi, (
                        f"{g.indicator_type} threshold {g.threshold} outside [{lo}, {hi}]"
                    )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_genetic_operators.py::TestMutateThresholdReset -v`
Expected: FAIL — current mutation doesn't reset threshold on indicator swap

**Step 3: Fix `_mutate_single_gene` in operators.py**

In `vibe_quant/discovery/operators.py`, modify `_mutate_single_gene` (line 467-495). In the `mutation_type == 0` branch (lines 471-475), after swapping indicator and params, also reset threshold:

```python
    if mutation_type == 0:
        # Swap indicator type
        new_ind = random.choice(_INDICATOR_NAMES)
        gene.indicator_type = new_ind
        gene.parameters = _random_params(new_ind)
        # Reset threshold to valid range for new indicator
        if new_ind in _THRESHOLD_RANGES:
            tlo, thi = _THRESHOLD_RANGES[new_ind]
            gene.threshold = round(random.uniform(tlo, thi), 4)
        elif new_ind in _PRICE_RELATIVE_INDICATORS:
            gene.threshold = 0.0
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_genetic_operators.py::TestMutateThresholdReset -v`
Expected: PASS

**Step 5: Run full test suite for operators**

Run: `pytest tests/unit/test_genetic_operators.py -v`
Expected: All pass

**Step 6: Commit**

```bash
git add vibe_quant/discovery/operators.py tests/unit/test_genetic_operators.py
git commit -m "fix: reset threshold on indicator swap in mutation (bd-b7of)"
```

---

### Task 2: Mixed-Scale Threshold Fix — Validation Guard

**Files:**
- Modify: `vibe_quant/discovery/operators.py` (export `_THRESHOLD_RANGES`)
- Modify: `vibe_quant/discovery/genome.py:222-294` (`validate_chromosome`)
- Test: `tests/unit/test_genome.py`

**Step 1: Write failing test**

Add to `tests/unit/test_genome.py`:

```python
class TestThresholdValidation:
    def test_threshold_out_of_range_detected(self) -> None:
        """ATR with threshold=72 (RSI range) should fail validation."""
        gene = StrategyGene("ATR", {"period": 14}, "greater_than", 72.0)
        chrom = StrategyChromosome(
            entry_genes=[gene],
            exit_genes=[StrategyGene("RSI", {"period": 14}, "greater_than", 70.0)],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
        )
        errors = validate_chromosome(chrom)
        assert any("threshold" in e.lower() for e in errors)

    def test_threshold_in_range_passes(self) -> None:
        """ATR with valid threshold should pass."""
        gene = StrategyGene("ATR", {"period": 14}, "greater_than", 0.015)
        chrom = StrategyChromosome(
            entry_genes=[gene],
            exit_genes=[StrategyGene("RSI", {"period": 14}, "greater_than", 70.0)],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
        )
        errors = validate_chromosome(chrom)
        assert errors == []

    def test_rsi_threshold_out_of_range(self) -> None:
        """RSI with threshold=150 (>100) should fail."""
        gene = StrategyGene("RSI", {"period": 14}, "greater_than", 150.0)
        chrom = StrategyChromosome(
            entry_genes=[gene],
            exit_genes=[StrategyGene("RSI", {"period": 14}, "greater_than", 70.0)],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
        )
        errors = validate_chromosome(chrom)
        assert any("threshold" in e.lower() for e in errors)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_genome.py::TestThresholdValidation -v`
Expected: FAIL — current validation doesn't check threshold ranges

**Step 3: Export threshold ranges and add validation**

In `vibe_quant/discovery/operators.py`, make `_THRESHOLD_RANGES` public by renaming to `THRESHOLD_RANGES` (line 196). Also export it in the module's public API. Update all internal references.

In `vibe_quant/discovery/genome.py`, import `THRESHOLD_RANGES` from operators and add to `validate_chromosome()` after the parameter validation loop (around line 283, before SL/TP checks):

```python
            # Validate threshold is within indicator's expected range
            if gene.indicator_type in THRESHOLD_RANGES:
                tlo, thi = THRESHOLD_RANGES[gene.indicator_type]
                if not (tlo <= gene.threshold <= thi):
                    errors.append(
                        f"{prefix}: threshold {gene.threshold} outside "
                        f"[{tlo}, {thi}] for {gene.indicator_type}"
                    )
```

Also update `is_valid_chromosome` in `operators.py` to check threshold ranges:

```python
    for gene in chrom.entry_genes + chrom.exit_genes:
        # ... existing checks ...
        if gene.indicator_type in THRESHOLD_RANGES:
            tlo, thi = THRESHOLD_RANGES[gene.indicator_type]
            if not (tlo <= gene.threshold <= thi):
                return False
```

**Step 4: Run tests**

Run: `pytest tests/unit/test_genome.py tests/unit/test_genetic_operators.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add vibe_quant/discovery/operators.py vibe_quant/discovery/genome.py tests/unit/test_genome.py
git commit -m "fix: validate threshold ranges in chromosome validation (bd-b7of)"
```

---

### Task 3: Per-Direction SL/TP — Schema Layer

**Files:**
- Modify: `vibe_quant/dsl/schema.py:408-548` (`StrategyDSL`)
- Test: `tests/unit/test_schema_per_direction.py` (new)

**Step 1: Write failing tests**

Create `tests/unit/test_schema_per_direction.py`:

```python
"""Tests for per-direction SL/TP in DSL schema."""
from __future__ import annotations

import pytest
from vibe_quant.dsl.schema import StrategyDSL


def _base_dsl(**overrides) -> dict:
    """Minimal valid DSL dict."""
    d = {
        "name": "test_strategy",
        "timeframe": "5m",
        "indicators": {"rsi": {"type": "RSI", "period": 14}},
        "entry_conditions": {"long": ["rsi > 50"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
    }
    d.update(overrides)
    return d


class TestPerDirectionSLTP:
    def test_backward_compatible_no_per_direction(self) -> None:
        """Existing strategies without per-direction fields still work."""
        strategy = StrategyDSL(**_base_dsl())
        assert strategy.stop_loss.percent == 2.0
        assert strategy.stop_loss_long is None
        assert strategy.stop_loss_short is None
        assert strategy.take_profit_long is None
        assert strategy.take_profit_short is None

    def test_per_direction_stop_loss(self) -> None:
        strategy = StrategyDSL(**_base_dsl(
            stop_loss_long={"type": "fixed_pct", "percent": 1.09},
            stop_loss_short={"type": "fixed_pct", "percent": 8.29},
        ))
        assert strategy.stop_loss_long.percent == 1.09
        assert strategy.stop_loss_short.percent == 8.29
        assert strategy.stop_loss.percent == 2.0  # base still there

    def test_per_direction_take_profit(self) -> None:
        strategy = StrategyDSL(**_base_dsl(
            take_profit_long={"type": "fixed_pct", "percent": 17.13},
            take_profit_short={"type": "fixed_pct", "percent": 13.06},
        ))
        assert strategy.take_profit_long.percent == 17.13
        assert strategy.take_profit_short.percent == 13.06

    def test_per_direction_atr_validates_indicator(self) -> None:
        """Per-direction ATR SL must reference existing indicator."""
        with pytest.raises(Exception):
            StrategyDSL(**_base_dsl(
                stop_loss_long={"type": "atr_fixed", "atr_multiplier": 1.5, "indicator": "nonexistent"},
            ))

    def test_per_direction_atr_valid_indicator(self) -> None:
        strategy = StrategyDSL(**_base_dsl(
            indicators={"rsi": {"type": "RSI"}, "atr_main": {"type": "ATR"}},
            stop_loss_long={"type": "atr_fixed", "atr_multiplier": 1.5, "indicator": "atr_main"},
        ))
        assert strategy.stop_loss_long.atr_multiplier == 1.5

    def test_only_long_override(self) -> None:
        """Can override just one direction."""
        strategy = StrategyDSL(**_base_dsl(
            stop_loss_long={"type": "fixed_pct", "percent": 1.0},
        ))
        assert strategy.stop_loss_long.percent == 1.0
        assert strategy.stop_loss_short is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_schema_per_direction.py -v`
Expected: FAIL — fields don't exist on StrategyDSL

**Step 3: Add per-direction fields to schema.py**

In `vibe_quant/dsl/schema.py`, add 4 optional fields to `StrategyDSL` (after `take_profit` on line 455):

```python
    stop_loss_long: StopLossConfig | None = Field(default=None, description="Stop loss override for longs")
    stop_loss_short: StopLossConfig | None = Field(default=None, description="Stop loss override for shorts")
    take_profit_long: TakeProfitConfig | None = Field(default=None, description="Take profit override for longs")
    take_profit_short: TakeProfitConfig | None = Field(default=None, description="Take profit override for shorts")
```

Add model validators to check ATR indicator references for per-direction configs (after existing `validate_take_profit_indicator`):

```python
    @model_validator(mode="after")
    def validate_per_direction_indicators(self) -> StrategyDSL:
        """Validate per-direction SL/TP indicator references exist."""
        for field_name in ("stop_loss_long", "stop_loss_short"):
            cfg = getattr(self, field_name)
            if cfg is not None and cfg.type in {"atr_fixed", "atr_trailing"}:
                if cfg.indicator and cfg.indicator not in self.indicators:
                    msg = f"{field_name} references indicator '{cfg.indicator}' which is not defined"
                    raise ValueError(msg)
        for field_name in ("take_profit_long", "take_profit_short"):
            cfg = getattr(self, field_name)
            if cfg is not None and cfg.type == "atr_fixed":
                if cfg.indicator and cfg.indicator not in self.indicators:
                    msg = f"{field_name} references indicator '{cfg.indicator}' which is not defined"
                    raise ValueError(msg)
        return self
```

**Step 4: Run tests**

Run: `pytest tests/unit/test_schema_per_direction.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add vibe_quant/dsl/schema.py tests/unit/test_schema_per_direction.py
git commit -m "feat: add per-direction SL/TP fields to DSL schema (bd-k4ya)"
```

---

### Task 4: Per-Direction SL/TP — Compiler Layer

**Files:**
- Modify: `vibe_quant/dsl/compiler.py:439-465` (`_generate_config_class`)
- Test: `tests/unit/test_compiler.py`

**Step 1: Write failing test**

Add to `tests/unit/test_compiler.py`:

```python
class TestPerDirectionSLTPCompiler:
    def test_per_direction_sl_emits_config_attrs(self) -> None:
        """Compiler should emit stop_loss_long_* attrs when per-direction SL is set."""
        from vibe_quant.dsl.compiler import StrategyCompiler
        from vibe_quant.dsl.schema import StrategyDSL

        dsl = StrategyDSL(
            name="test_per_dir",
            timeframe="5m",
            indicators={"rsi": {"type": "RSI", "period": 14}},
            entry_conditions={"long": ["rsi > 50"]},
            stop_loss={"type": "fixed_pct", "percent": 2.0},
            take_profit={"type": "fixed_pct", "percent": 4.0},
            stop_loss_long={"type": "fixed_pct", "percent": 1.09},
            stop_loss_short={"type": "fixed_pct", "percent": 8.29},
        )
        compiler = StrategyCompiler()
        source = compiler.compile(dsl)
        assert "stop_loss_long_type" in source
        assert "stop_loss_long_percent" in source
        assert "1.09" in source
        assert "stop_loss_short_type" in source
        assert "8.29" in source

    def test_no_per_direction_no_extra_attrs(self) -> None:
        """Without per-direction, should not emit *_long_* or *_short_* attrs."""
        from vibe_quant.dsl.compiler import StrategyCompiler
        from vibe_quant.dsl.schema import StrategyDSL

        dsl = StrategyDSL(
            name="test_unified",
            timeframe="5m",
            indicators={"rsi": {"type": "RSI", "period": 14}},
            entry_conditions={"long": ["rsi > 50"]},
            stop_loss={"type": "fixed_pct", "percent": 2.0},
            take_profit={"type": "fixed_pct", "percent": 4.0},
        )
        compiler = StrategyCompiler()
        source = compiler.compile(dsl)
        assert "stop_loss_long_type" not in source
        assert "stop_loss_short_type" not in source
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_compiler.py::TestPerDirectionSLTPCompiler -v`
Expected: FAIL

**Step 3: Add per-direction config emission to compiler**

In `vibe_quant/dsl/compiler.py`, in `_generate_config_class()`, after the existing take profit block (around line 465), add:

```python
        # Add per-direction stop loss parameters (if present)
        for direction in ("long", "short"):
            sl_cfg = getattr(dsl, f"stop_loss_{direction}", None)
            if sl_cfg is not None:
                lines.append("")
                lines.append(f"    # Stop loss ({direction}) parameters")
                lines.append(f'    stop_loss_{direction}_type: str = "{sl_cfg.type}"')
                if sl_cfg.percent is not None:
                    lines.append(f"    stop_loss_{direction}_percent: float = {sl_cfg.percent}")
                if sl_cfg.atr_multiplier is not None:
                    lines.append(f"    stop_loss_{direction}_atr_multiplier: float = {sl_cfg.atr_multiplier}")
                if sl_cfg.indicator is not None:
                    lines.append(f'    stop_loss_{direction}_indicator: str = "{sl_cfg.indicator}"')

        # Add per-direction take profit parameters (if present)
        for direction in ("long", "short"):
            tp_cfg = getattr(dsl, f"take_profit_{direction}", None)
            if tp_cfg is not None:
                lines.append("")
                lines.append(f"    # Take profit ({direction}) parameters")
                lines.append(f'    take_profit_{direction}_type: str = "{tp_cfg.type}"')
                if tp_cfg.percent is not None:
                    lines.append(f"    take_profit_{direction}_percent: float = {tp_cfg.percent}")
                if tp_cfg.atr_multiplier is not None:
                    lines.append(f"    take_profit_{direction}_atr_multiplier: float = {tp_cfg.atr_multiplier}")
                if tp_cfg.risk_reward_ratio is not None:
                    lines.append(f"    take_profit_{direction}_risk_reward: float = {tp_cfg.risk_reward_ratio}")
                if tp_cfg.indicator is not None:
                    lines.append(f'    take_profit_{direction}_indicator: str = "{tp_cfg.indicator}"')
```

**Step 4: Run tests**

Run: `pytest tests/unit/test_compiler.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add vibe_quant/dsl/compiler.py tests/unit/test_compiler.py
git commit -m "feat: compiler emits per-direction SL/TP config attrs (bd-k4ya)"
```

---

### Task 5: Per-Direction SL/TP — Templates Layer

**Files:**
- Modify: `vibe_quant/dsl/templates.py:152-249` (`_calculate_sl_price`, `_calculate_tp_price`)
- Test: integration via compiler test

**Step 1: Write failing test**

Add to `tests/unit/test_compiler.py`:

```python
class TestPerDirectionSLTPRuntime:
    def _compile_and_get_class(self, dsl_dict: dict):
        from vibe_quant.dsl.compiler import StrategyCompiler
        from vibe_quant.dsl.schema import StrategyDSL

        dsl = StrategyDSL(**dsl_dict)
        compiler = StrategyCompiler()
        return compiler.compile_to_class(dsl)

    def test_per_direction_sl_in_source(self) -> None:
        """Templates should check for per-direction SL before unified."""
        from vibe_quant.dsl.compiler import StrategyCompiler
        from vibe_quant.dsl.schema import StrategyDSL

        dsl = StrategyDSL(
            name="test_per_dir_tpl",
            timeframe="5m",
            indicators={"rsi": {"type": "RSI", "period": 14}},
            entry_conditions={"long": ["rsi > 50"]},
            stop_loss={"type": "fixed_pct", "percent": 2.0},
            take_profit={"type": "fixed_pct", "percent": 4.0},
            stop_loss_long={"type": "fixed_pct", "percent": 1.09},
            stop_loss_short={"type": "fixed_pct", "percent": 8.29},
        )
        compiler = StrategyCompiler()
        source = compiler.compile(dsl)
        # Template should contain per-direction config lookup
        assert "stop_loss_long_type" in source
        assert "stop_loss_short_type" in source
```

**Step 2: Modify templates.py**

Replace the `_calculate_sl_price` method lines (153-172) to check per-direction config first:

```python
    "def _calculate_sl_price(self, entry_price: float, is_long: bool) -> float | None:",
    '    """Calculate stop-loss price based on config type (per-direction aware)."""',
    "    # Check for per-direction override first",
    "    dir_suffix = 'long' if is_long else 'short'",
    "    sl_type = getattr(self.config, f'stop_loss_{dir_suffix}_type', None)",
    "    if sl_type is None:",
    "        sl_type = self.config.stop_loss_type",
    "        prefix = 'stop_loss'",
    "    else:",
    "        prefix = f'stop_loss_{dir_suffix}'",
    '    if sl_type == "fixed_pct":',
    "        pct = getattr(self.config, f'{prefix}_percent')",
    "        if is_long:",
    "            return entry_price * (1 - pct / 100)",
    "        else:",
    "            return entry_price * (1 + pct / 100)",
    '    elif sl_type in ("atr_fixed", "atr_trailing"):',
    "        sl_ind = getattr(self.config, f'{prefix}_indicator', None)",
    "        if sl_ind is None:",
    "            return None",
    "        atr_value = self._get_indicator_value(sl_ind)",
    "        multiplier = getattr(self.config, f'{prefix}_atr_multiplier')",
    "        if is_long:",
    "            return entry_price - atr_value * multiplier",
    "        else:",
    "            return entry_price + atr_value * multiplier",
    "    return None",
```

Similarly replace `_calculate_tp_price` (lines 221-249):

```python
    "def _calculate_tp_price(self, entry_price: float, is_long: bool) -> float | None:",
    '    """Calculate take-profit price based on config type (per-direction aware)."""',
    "    dir_suffix = 'long' if is_long else 'short'",
    "    tp_type = getattr(self.config, f'take_profit_{dir_suffix}_type', None)",
    "    if tp_type is None:",
    "        tp_type = self.config.take_profit_type",
    "        prefix = 'take_profit'",
    "    else:",
    "        prefix = f'take_profit_{dir_suffix}'",
    '    if tp_type == "fixed_pct":',
    "        pct = getattr(self.config, f'{prefix}_percent')",
    "        if is_long:",
    "            return entry_price * (1 + pct / 100)",
    "        else:",
    "            return entry_price * (1 - pct / 100)",
    '    elif tp_type == "atr_fixed":',
    "        tp_ind = getattr(self.config, f'{prefix}_indicator', None)",
    "        if tp_ind is None:",
    "            return None",
    "        atr_value = self._get_indicator_value(tp_ind)",
    "        multiplier = getattr(self.config, f'{prefix}_atr_multiplier')",
    "        if is_long:",
    "            return entry_price + atr_value * multiplier",
    "        else:",
    "            return entry_price - atr_value * multiplier",
    '    elif tp_type == "risk_reward":',
    "        sl_price = self._calculate_sl_price(entry_price, is_long)",
    "        if sl_price is not None:",
    "            sl_distance = abs(entry_price - sl_price)",
    "            ratio = getattr(self.config, f'{prefix}_risk_reward')",
    "            if is_long:",
    "                return entry_price + sl_distance * ratio",
    "            else:",
    "                return entry_price - sl_distance * ratio",
    "    return None",
```

Also update `_update_trailing_stop` (lines 174-218) to use per-direction prefix:

```python
    "def _update_trailing_stop(self, bar: Bar) -> None:",
    '    """Update trailing stop loss as price moves favorably."""',
    "    if not self._position_open:",
    "        return",
    "    is_long = self._position_side == OrderSide.BUY",
    "    dir_suffix = 'long' if is_long else 'short'",
    "    sl_type = getattr(self.config, f'stop_loss_{dir_suffix}_type', None)",
    "    if sl_type is None:",
    "        sl_type = self.config.stop_loss_type",
    "        prefix = 'stop_loss'",
    "    else:",
    "        prefix = f'stop_loss_{dir_suffix}'",
    '    if sl_type != "atr_trailing":',
    "        return",
    "    sl_ind = getattr(self.config, f'{prefix}_indicator', None)",
    "    if sl_ind is None:",
    "        return",
    ...  # rest of trailing logic unchanged, just uses `prefix` for atr_multiplier
```

**Step 3: Run tests**

Run: `pytest tests/unit/test_compiler.py -v`
Expected: All pass

**Step 4: Commit**

```bash
git add vibe_quant/dsl/templates.py tests/unit/test_compiler.py
git commit -m "feat: templates select per-direction SL/TP config (bd-k4ya)"
```

---

### Task 6: Per-Direction SL/TP — Genome Layer

**Files:**
- Modify: `vibe_quant/discovery/operators.py:131-161` (`StrategyChromosome`)
- Modify: `vibe_quant/discovery/operators.py:314-366` (`crossover`)
- Modify: `vibe_quant/discovery/operators.py:403-443` (`mutate`)
- Modify: `vibe_quant/discovery/operators.py:580-601` (`_random_chromosome`)
- Modify: `vibe_quant/discovery/genome.py:348-422` (`chromosome_to_dsl`)
- Modify: `vibe_quant/discovery/genome.py:222-294` (`validate_chromosome`)
- Test: `tests/unit/test_genome.py`, `tests/unit/test_genetic_operators.py`

**Step 1: Write failing tests**

Add to `tests/unit/test_genome.py`:

```python
class TestPerDirectionSLTPGenome:
    def test_chromosome_to_dsl_emits_per_direction(self) -> None:
        """BOTH-direction chromosome with per-direction SL/TP emits correct DSL."""
        g = StrategyGene("RSI", {"period": 14}, "less_than", 30.0)
        chrom = StrategyChromosome(
            entry_genes=[g],
            exit_genes=[g],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            direction="both",
            stop_loss_long_pct=1.09,
            stop_loss_short_pct=8.29,
            take_profit_long_pct=17.13,
            take_profit_short_pct=13.06,
        )
        dsl_dict = chromosome_to_dsl(chrom)
        assert dsl_dict["stop_loss_long"]["percent"] == 1.09
        assert dsl_dict["stop_loss_short"]["percent"] == 8.29
        assert dsl_dict["take_profit_long"]["percent"] == 17.13
        assert dsl_dict["take_profit_short"]["percent"] == 13.06
        # Base SL/TP still present
        assert dsl_dict["stop_loss"]["percent"] == 2.0

    def test_chromosome_to_dsl_no_per_direction_for_single_direction(self) -> None:
        """LONG-only chromosome should not emit per-direction fields."""
        g = StrategyGene("RSI", {"period": 14}, "less_than", 30.0)
        chrom = StrategyChromosome(
            entry_genes=[g],
            exit_genes=[g],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            direction="long",
        )
        dsl_dict = chromosome_to_dsl(chrom)
        assert "stop_loss_long" not in dsl_dict
        assert "stop_loss_short" not in dsl_dict

    def test_per_direction_dsl_validates(self) -> None:
        """Per-direction DSL dict should parse as valid StrategyDSL."""
        g = StrategyGene("RSI", {"period": 14}, "less_than", 30.0)
        chrom = StrategyChromosome(
            entry_genes=[g],
            exit_genes=[g],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            direction="both",
            stop_loss_long_pct=1.09,
            stop_loss_short_pct=8.29,
            take_profit_long_pct=17.13,
            take_profit_short_pct=13.06,
        )
        dsl_dict = chromosome_to_dsl(chrom)
        strategy = StrategyDSL(**dsl_dict)
        assert strategy.stop_loss_long.percent == 1.09

    def test_validate_per_direction_sl_range(self) -> None:
        """Per-direction SL/TP must be in valid range."""
        g = StrategyGene("RSI", {"period": 14}, "less_than", 30.0)
        chrom = StrategyChromosome(
            entry_genes=[g],
            exit_genes=[g],
            stop_loss_pct=2.0,
            take_profit_pct=4.0,
            direction="both",
            stop_loss_long_pct=15.0,  # out of range
        )
        errors = validate_chromosome(chrom)
        assert any("stop_loss_long" in e for e in errors)
```

Add to `tests/unit/test_genetic_operators.py`:

```python
class TestPerDirectionSLTPOperators:
    def test_crossover_preserves_per_direction(self) -> None:
        """Crossover should swap per-direction SL/TP independently."""
        random.seed(42)
        a = _make_chromosome(direction=Direction.BOTH)
        a.stop_loss_long_pct = 1.0
        a.stop_loss_short_pct = 5.0
        b = _make_chromosome(direction=Direction.BOTH)
        b.stop_loss_long_pct = 2.0
        b.stop_loss_short_pct = 8.0

        sl_long_vals = set()
        for _ in range(100):
            c1, _ = crossover(a, b)
            if c1.stop_loss_long_pct is not None:
                sl_long_vals.add(c1.stop_loss_long_pct)
        assert sl_long_vals <= {1.0, 2.0}

    def test_mutation_perturbs_per_direction_independently(self) -> None:
        """Per-direction SL/TP should mutate independently."""
        random.seed(42)
        chrom = _make_chromosome(direction=Direction.BOTH)
        chrom.stop_loss_long_pct = 2.0
        chrom.stop_loss_short_pct = 5.0
        changed_long = False
        changed_short = False
        for _ in range(100):
            mutated = mutate(chrom, mutation_rate=1.0)
            if mutated.stop_loss_long_pct != 2.0:
                changed_long = True
            if mutated.stop_loss_short_pct != 5.0:
                changed_short = True
        assert changed_long
        assert changed_short

    def test_random_both_chromosome_has_per_direction(self) -> None:
        """BOTH-direction random chromosomes should have per-direction SL/TP."""
        random.seed(42)
        pop = initialize_population(size=50, direction_constraint=Direction.BOTH)
        has_per_dir = sum(1 for c in pop if c.stop_loss_long_pct is not None)
        assert has_per_dir == 50  # all BOTH chromosomes get per-direction
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_genome.py::TestPerDirectionSLTPGenome tests/unit/test_genetic_operators.py::TestPerDirectionSLTPOperators -v`
Expected: FAIL

**Step 3: Implement genome changes**

**operators.py — StrategyChromosome (lines 131-161):**

Add 4 optional fields:

```python
    stop_loss_long_pct: float | None = field(default=None)
    stop_loss_short_pct: float | None = field(default=None)
    take_profit_long_pct: float | None = field(default=None)
    take_profit_short_pct: float | None = field(default=None)
```

Update `clone()` to copy them.

**operators.py — `crossover` (lines 343-347):**

After existing SL/TP crossover, add:

```python
    # Per-direction SL/TP: random pick per child
    sl_long_a = parent_a.stop_loss_long_pct if random.random() < 0.5 else parent_b.stop_loss_long_pct
    sl_short_a = parent_a.stop_loss_short_pct if random.random() < 0.5 else parent_b.stop_loss_short_pct
    tp_long_a = parent_a.take_profit_long_pct if random.random() < 0.5 else parent_b.take_profit_long_pct
    tp_short_a = parent_a.take_profit_short_pct if random.random() < 0.5 else parent_b.take_profit_short_pct
    # ... same for child_b
```

Pass these into child construction.

**operators.py — `mutate` (lines 437-441):**

After existing SL/TP mutation, add:

```python
    # Mutate per-direction SL/TP
    if chrom.stop_loss_long_pct is not None and random.random() < mutation_rate * 0.5:
        chrom.stop_loss_long_pct = _perturb(chrom.stop_loss_long_pct, 0.2, SL_RANGE[0], SL_RANGE[1])
    if chrom.stop_loss_short_pct is not None and random.random() < mutation_rate * 0.5:
        chrom.stop_loss_short_pct = _perturb(chrom.stop_loss_short_pct, 0.2, SL_RANGE[0], SL_RANGE[1])
    if chrom.take_profit_long_pct is not None and random.random() < mutation_rate * 0.5:
        chrom.take_profit_long_pct = _perturb(chrom.take_profit_long_pct, 0.2, TP_RANGE[0], TP_RANGE[1])
    if chrom.take_profit_short_pct is not None and random.random() < mutation_rate * 0.5:
        chrom.take_profit_short_pct = _perturb(chrom.take_profit_short_pct, 0.2, TP_RANGE[0], TP_RANGE[1])
```

**operators.py — `_random_chromosome` (lines 580-601):**

After creating chromosome, if direction is BOTH, populate per-direction:

```python
    chrom = StrategyChromosome(...)
    if chrom.direction == Direction.BOTH:
        chrom.stop_loss_long_pct = round(random.uniform(*SL_RANGE), 4)
        chrom.stop_loss_short_pct = round(random.uniform(*SL_RANGE), 4)
        chrom.take_profit_long_pct = round(random.uniform(*TP_RANGE), 4)
        chrom.take_profit_short_pct = round(random.uniform(*TP_RANGE), 4)
    return chrom
```

**operators.py — `is_valid_chromosome`:**

Add per-direction range checks:

```python
    for attr, valid_range in [
        ("stop_loss_long_pct", SL_RANGE),
        ("stop_loss_short_pct", SL_RANGE),
        ("take_profit_long_pct", TP_RANGE),
        ("take_profit_short_pct", TP_RANGE),
    ]:
        val = getattr(chrom, attr)
        if val is not None and not (valid_range[0] <= val <= valid_range[1]):
            return False
```

**genome.py — `validate_chromosome` (after line 288):**

```python
    # Per-direction SL/TP ranges
    for attr, label, valid_range in [
        ("stop_loss_long_pct", "stop_loss_long_pct", (0.5, 10.0)),
        ("stop_loss_short_pct", "stop_loss_short_pct", (0.5, 10.0)),
        ("take_profit_long_pct", "take_profit_long_pct", (0.5, 20.0)),
        ("take_profit_short_pct", "take_profit_short_pct", (0.5, 20.0)),
    ]:
        val = getattr(chrom, attr, None)
        if val is not None and not (valid_range[0] <= val <= valid_range[1]):
            errors.append(f"{label}={val} out of range [{valid_range[0]}, {valid_range[1]}]")
```

**genome.py — `chromosome_to_dsl` (lines 404-416):**

After building base SL/TP dict, add per-direction:

```python
    # Per-direction SL/TP overrides (only for BOTH direction)
    if direction == Direction.BOTH:
        if chrom.stop_loss_long_pct is not None:
            dsl["stop_loss_long"] = {"type": "fixed_pct", "percent": round(chrom.stop_loss_long_pct, 2)}
        if chrom.stop_loss_short_pct is not None:
            dsl["stop_loss_short"] = {"type": "fixed_pct", "percent": round(chrom.stop_loss_short_pct, 2)}
        if chrom.take_profit_long_pct is not None:
            dsl["take_profit_long"] = {"type": "fixed_pct", "percent": round(chrom.take_profit_long_pct, 2)}
        if chrom.take_profit_short_pct is not None:
            dsl["take_profit_short"] = {"type": "fixed_pct", "percent": round(chrom.take_profit_short_pct, 2)}
```

**Step 4: Run all tests**

Run: `pytest tests/unit/test_genome.py tests/unit/test_genetic_operators.py -v`
Expected: All pass

**Step 5: Run full test suite**

Run: `pytest tests/ -v --timeout=60`
Expected: All pass

**Step 6: Commit**

```bash
git add vibe_quant/discovery/operators.py vibe_quant/discovery/genome.py tests/unit/test_genome.py tests/unit/test_genetic_operators.py
git commit -m "feat: per-direction SL/TP in genome, operators, crossover, mutation (bd-k4ya)"
```

---

### Task 7: Integration Test + Lint + Final Verification

**Files:**
- All modified files

**Step 1: Run full test suite**

Run: `pytest tests/ -v --timeout=60`
Expected: All pass

**Step 2: Run linter**

Run: `ruff check vibe_quant/dsl/ vibe_quant/discovery/ tests/unit/`
Expected: No errors

**Step 3: Run type checker**

Run: `mypy vibe_quant/dsl/schema.py vibe_quant/dsl/compiler.py vibe_quant/dsl/templates.py vibe_quant/discovery/genome.py vibe_quant/discovery/operators.py`
Expected: No errors (or pre-existing only)

**Step 4: Close beads issues**

```bash
bd close vibe-quant-b7of vibe-quant-k4ya
bd sync
```

**Step 5: Final commit and push**

```bash
git push
```
