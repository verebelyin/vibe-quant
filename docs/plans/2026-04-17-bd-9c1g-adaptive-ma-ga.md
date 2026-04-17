# bd-9c1g design — Adaptive MAs (KAMA/VIDYA/FRAMA) in GA discovery

## Problem

Plugins for KAMA, VIDYA, FRAMA register cleanly and compile for hand-written strategies (`close > kama`). But they're excluded from GA discovery: `_build_indicator_pool_from_registry` at `vibe_quant/discovery/genome.py:110` skips any `IndicatorSpec` with `threshold_range=None`, and all three MAs set it to None deliberately (a threshold of e.g. "12500" against a price series has no meaningful GA search range — it changes with market level).

Current GA gene structure (`StrategyGene`): `indicator vs scalar threshold`. MAs don't fit because the natural condition for an MA is `price vs ma` or `ma vs ma` — relative, not scalar.

## Two options

### (a) Gene variant: price-vs-MA / ma-vs-ma conditions

Add a new gene shape to `StrategyGene` (or a sibling dataclass):
- `MAConditionGene{ indicator_a: "KAMA", params_a: {...}, op: GT|LT|crosses, indicator_b: "close"|"KAMA", params_b: {...} }`

Compiler changes:
- Current `StrategyCompiler` emits `indicator_name COND threshold_value`.
- Needs a new path: `indicator_a COND indicator_b` where `indicator_b` may be the raw close series.
- Works because `StrategyCompiler` already computes indicator series — just reference two instead of one.

Operators changes:
- `mutate_gene`, `initialize_gene` need to handle the new shape.
- Tournament/crossover need compatibility — probably simpler if `MAConditionGene` is a separate type and only MA-enrolled indicators produce it.

Discovery enrollment:
- Remove the "skip if threshold_range=None" gate for MAs specifically (or add a `ma_like=True` field on IndicatorSpec).
- Enroll MAs into a separate `MA_POOL` distinct from `INDICATOR_POOL`; `operators.py` picks from MA_POOL only for MAConditionGenes.

**Cost:** ~300 lines. Touches genome.py, operators.py, compiler.py, dsl/schema.py, and all their tests. Probably 1–2 days.

**Benefit:** Clean, matches how traders actually use MAs. Opens the door for ribbon strategies (ma_fast > ma_slow), golden-cross patterns.

### (b) Synthesize a scalar threshold from an MA property

Add a transform that reduces an MA to a unitless scalar per bar:
- MA-slope (`(ma[t] - ma[t-N]) / ma[t-N]` → fraction) — thresholds in e.g. (-0.01, 0.01)
- MA-vs-price (`(close - ma) / ma`) — thresholds in e.g. (-0.05, 0.05)

Set `threshold_range` on each MA plugin to a sensible interval, and `compute_fn` returns the transformed series instead of the raw MA.

**Cost:** ~30 lines per plugin (3 plugins × 1 transform + test). Half a day. No core engine change.

**Benefit:** Cheap, works today. KAMA/VIDYA/FRAMA become usable in GA.

**Downsides:**
- Lies to the user — a plugin labeled KAMA now outputs KAMA-slope, which is confusing when eyeballing DSL JSON.
- Hand-written strategies using `kama()` still return raw MA — plugin now has two personalities depending on context.
- Doesn't enable ribbon/cross patterns (`ma_fast > ma_slow`).

## Recommendation: (a), but phased

Option (b) is a hack that leaks the transform semantic into the plugin name. That's the kind of shortcut that comes back six months later as a gotcha memory.

Option (a) is the right architecture but large. Split into two phases:

**Phase 1 — `close vs MA` only:**
- New gene: `PriceVsMAConditionGene{ ma_name, params, op, }` — always compares raw close to the MA.
- Enroll MAs in a new `MA_POOL`, keep `INDICATOR_POOL` untouched.
- Compiler: new condition emitter `close COND ma_series`.
- Mutation: one MA swap, one params mutation, one op flip.
- ~150 lines + tests. Unlocks KAMA/VIDYA/FRAMA for pullback strategies.

**Phase 2 — `ma_fast vs ma_slow`:**
- Extend to two MAs of the same type with different params (e.g. KAMA-10 vs KAMA-30).
- Needs crossover logic in the genome to keep `period_fast < period_slow` invariant.
- Unlocks ribbon/cross strategies.

Phase 1 alone closes bd-9c1g usefully; phase 2 is a follow-up.

## Open questions

1. `PriceVsMAConditionGene` lives in `StrategyGene` as a variant, or as a separate chromosome field? Variant keeps mutation logic uniform; separate field makes typing cleaner.
2. Entry/exit parity — a `close < kama` entry naturally pairs with `close > kama` exit. Should chromosomes enforce this, or let GA discover independently?
3. How many MA-genes per chromosome? Cap at 1 entry + 1 exit initially to avoid blowing up the search space?
