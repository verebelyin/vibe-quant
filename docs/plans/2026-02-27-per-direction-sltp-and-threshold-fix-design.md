# Per-Direction SL/TP + Mixed-Scale Threshold Fix

Date: 2026-02-27
Issues: vibe-quant-k4ya (P2 feature), vibe-quant-b7of (P4 chore)

## 1. Per-Direction SL/TP (k4ya)

### Problem
Single SL/TP for whole strategy. Combined (BOTH) strategies need different risk per direction — bull champion wants 1.09%/17.13% SL/TP, bear wants 8.29%/13.06%.

### Design: Override Fields
Keep `stop_loss`/`take_profit` as defaults; add optional `stop_loss_long`, `stop_loss_short`, `take_profit_long`, `take_profit_short` that override per direction.

```yaml
risk:
  stop_loss:
    type: fixed_pct
    percent: 2.0          # default
  stop_loss_long:
    type: fixed_pct
    percent: 1.09         # override for longs
  stop_loss_short:
    type: fixed_pct
    percent: 8.29         # override for shorts
```

### Changes

**schema.py:** Add 4 optional fields (same types as existing SL/TP configs). Validation: per-direction must be valid config if present.

**compiler.py:** Emit per-direction config attrs (`stop_loss_long_type`, `stop_loss_long_percent`, etc.) when present. Fall back to unified.

**templates.py:** `_calculate_sl_price()`/`_calculate_tp_price()` check `getattr(self, f"stop_loss_{side}_type", None)` first, fall back to unified. Math unchanged.

**genome.py:** `StrategyChromosome` gains optional `stop_loss_long_pct`, `stop_loss_short_pct`, `take_profit_long_pct`, `take_profit_short_pct`. Only populated when `direction == BOTH`. `chromosome_to_dsl()` emits per-direction when present.

**operators.py:** Mutation independently mutates each direction's SL/TP. Crossover swaps per-direction values as separate gene pairs. Population init: randomize per-direction for BOTH-direction chromosomes.

### Overfitting Warning
4 extra free parameters in GA. Per-direction optimal values may be curve-fitted. **Monitor**: do BOTH-direction strategies with per-direction SL/TP survive WFA/purged-kfold? If not → rollback GA integration, keep DSL/compiler support for manual strategies only.

## 2. Mixed-Scale Threshold Fix (b7of)

### Problem
Mutation swaps indicator type without resetting threshold. E.g. RSI(14)>72 mutates indicator to ATR → ATR(14)>72 (impossible, ATR ~0.001-0.03). Produces 0 trades. Small populations amplify this — 60-78% zero-trade rate observed.

### Fix

**operators.py `_mutate_single_gene()`:** When mutation_type==0 (indicator swap), re-sample threshold from `_THRESHOLD_RANGES[new_indicator]`.

**genome.py `validate_chromosome()`:** Add threshold range check — if indicator has defined range in `_THRESHOLD_RANGES`, threshold must be within that range.

**Shared ranges:** Export `_THRESHOLD_RANGES` from operators.py so validation references same source of truth.

## Alternatives Considered

- **Normalized 0-1 thresholds**: Fundamental but breaking change. Deferred.
- **DSL-only per-direction SL/TP (no GA)**: Simpler, but user wants full stack. Can rollback GA part if overfitting detected.
- **Wider threshold ranges + small-pop bias**: Addresses symptoms not root cause. Threshold reset fixes root cause.
