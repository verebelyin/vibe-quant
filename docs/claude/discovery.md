# Discovery pipeline

Read before running, interpreting or changing discovery, fitness or gates.

## Current behaviour
- Discovery and screening share one path (`NTScreeningRunner` → `StrategyCompiler`). A champion replays exactly within a run when `eval_windows=1`.
- **Liquidation is an opt-in overlay, OFF by default.** Settings/flags: [`engine-modes.md`](engine-modes.md). No penalty or gate: mode on scores liquidation-adjusted metrics (same formula, truncated ledgers). Rows carry `notes.liquidation = {mode, leverage}` plus `liquidation_count` / `liquidation_forfeited`; mode off writes no `liquidation*` key. Peak RSS ≈ 1.15 GB/worker (size `max_workers`).
- **Fail closed on liquidation data:** a missing or incomplete MARK 1m series, an unpriceable open survivor, or a fill/funding/cash flow at run end fails the run with `LiquidationIntegrityError: liquidation_mode='isolated': ...` (never a `-999`/`-inf` row), at every stage.
- **`path_divergence`** (any hit): NT's later positions were sized on pre-liquidation equity and missed re-entries cannot be rebuilt.
- **Liquidation exit timing (screening/discovery, mode on, timeframe above 1m):** a resting STOP_MARKET/LIMIT exit is timed at the first 1m LAST bar inside the strategy bar (after it went live) that reached its level (stop: touch; limit: trade through). MEASURED, 239 BTCUSDT 4h: 0 vs 0 hits at 50x, 60 vs 60 at 125x.
- **Position sizing is a run-level setting, never a gene.** CLI `--max-position-pct` / `--risk-per-trade` (API top-level `max_position_pct` / `risk_per_trade`), defaults 0.5 / 0.02, stored only when non-default. Recorded in `notes.sizing`; promote/replay inherit it (unreadable: 409). A sizing-field sweep key (dotted spellings too) is rejected.
- **Margin limit and bankroll grid are opt-in, OFF by default.**
  - Margin limit: `--max-margin-pct` (API `max_margin_pct`, in (0, 1]) caps margin in use (open + pending + candidate entry) via `vibe_quant/margin_limit.py`; in `notes.sizing`, inherited by promote/replay.
  - Bankroll grid: CLI `--bankroll-modes` (`isolated`/`cross`) + `--bankroll-leverages` (whole 1–125) + `--bankroll-max-margin-pcts` (in (0, 1]), all three, or API `bankroll_grid = {modes, leverages, max_margin_pcts}`. Cells = product(modes in `(isolated, cross)` order, sorted leverages, sorted cushions), deduplicated, ≤ 12. Exclusive with run-level `--liquidation-mode` / `--leverage` / `--max-margin-pct`.
  - Discovery picks strategy and cell together; every cell × seed evaluation is a DSR trial (**N = Σ over cells × seeds**).
  - A champion records `top_strategies[i].bankroll = {liquidation_mode, leverage, max_margin_pct}`. Promote/replay inherit the cell (missing or invalid: 422).
- **`--exit-genes extended` (opt-in):** GA draws ATR/trailing/safety stops, ATR/R:R targets, time exits. Needs `--sizing-mode atr`; `--leverage 1` if liquidation off.
- **`--indicator-pool` also bounds MA genes:** an MA name in an explicit pool enables it, else none (`None` = all).
- **`TAKER_IMBALANCE` needs an explicit `--indicator-pool`** (context indicator); no NaN hold over a gap, paper refuses it, the fast tier replays it as-of.
- **Budget and power:** CLI/API default 50×20 (was 20×15). `notes.power_preflight` (form too): DSR Sharpe bars (√252) at N = prior+1, prior+pop×gen; reporting only, uncomputable = `status: unavailable`.
- **Tiny runs produce no rows:** fitness is 0 below 50 trades or at return <= 0 (`fitness.py:573`), so a pop 8–12, 2-generation, 2–3 month run ends `no_viable_strategies`. Prove row-level behaviour with a pipeline test.
- **Fitness** = 35% Sharpe + 25% (1 − MaxDD) + 20% PF + 20% Return.
- **`--fitness v2` (opt-in):** shrunk MTM t-stat × profitable-window factor, averaged with its one-step neighbours' median. Refused with `--evaluator fast`, `--null-gate`, `--port`, `--mock`.
- **`eval_windows` (default 3) stores WORST-of-N sub-window metrics** (min Sharpe/return/PF, max DD; each window needs ≥ max(1, min_trades // (2N)) trades). A full-window replay legitimately differs.

## Gates
- **Fail closed:** 0 champions + `guardrail_rejections`. Every run keeps a default 20% holdout, used once as the final gate.
- **Bootstrap-CI floor** (statistic = t-stat mean/std·√n): 1m 0.5, 4h/1d 0.0 (1.0 is unpassable at ~50–180 trades/yr). A champion flagged `no_bootstrap_ci=true` is overfit (forced-past: 5.40 → −2.78 in validation).
- **DSR N = effective count:** this run's EV-matched effective trials (`notes.dsr_trials.run_n_eff`; raw when < 2 retained series) + earlier same-timeframe overlapping-train runs' summed `run_n_eff` under `n_rule` `cumulative_effective_same_tf_overlapping_train` (`gate_n/prior_n/prior_run_ids`; `prior_legacy_*` tallied, not charged). DB error -> run-only N plus `prior_error`; `--no-dsr` records `run_n_eff` = raw. DSR passes at the legacy PSR statistic >= the run's q95 MC spread (default; `--dsr-rule emax` = 1.6449) and tests the champion's **mark-to-market** series; missing fails DSR closed.
- **Buy-and-hold gate (fail closed):** champions must beat the directional equal-weight, daily-rebalanced buy-and-hold Sharpe of their symbols over the train window, on the MTM daily series (`notes.buy_hold_gate`). `symbol_agg=worst` also checks each symbol against its own benchmark. Skips only for a non-NT backtest fn; `--no-buy-hold-gate` disables.
- **Consistency:** `validation/consistency.py` flags screening→validation collapse (holdout champions compare on holdout metrics).
- **Seed:** `--seed N` (auto-drawn otherwise, stored in `notes.seed`) replays a run exactly (same config, code).
- **Fails loudly:** an all-errored eval batch raises `DiscoveryEvaluationError` (run `failed`, not "0 champions").

## Data and speed facts
- `compiler_version_hash()` changes with indicator/codegen changes: `bd recall discovery:compiler-hash`. Latest results and verdicts: the `## ` headlines of `docs/discovery-journal.md`.
- **HYPE history is short:** ~504 d (2025-05-30 10:30 → 2026-10-16; `OWN_TREND(200)` NaN for ~200 daily bars). Multi-symbol runs start >= 2025-05-31 (else fill-tick coverage fails closed); HYPE-only champions are low-confidence. HYPE funds every 4h; one archived skip (2026-06-24 04:00) is a flagged fallback (`0.00005 x notional`) in `funding_fallback_settlements`.
- **1m is slow:** Rust-native indicators (SMA/EMA/CCI/STOCH/ATR) beat pandas-path ones (ADX/MACD/BBANDS/KAMA) by ~10×.

## Comparability breaks

Replay-changing breaks also bump `ENGINE_EPOCH` (`vibe_quant/shadow/tracker.py`) in its commit; shadow rows that stop reproducing retire (SPEC § Shadow Tracker).
Results from before each date are NOT comparable with later ones:
- **2026-11-24 NOT a break: Sharpe shown in √365** (vibe-quant-clmmc.15): texts, logs and typed limits use the display Sharpe (stored × 1.2035; GLOSSARY Display Sharpe); stored numbers and gates unchanged; notes goldens, gate_golden and tiers C1/C2 re-pinned text-only (masked diff IDENTICAL).
- **2026-11-23 stop cap** (vibe-quant-clmmc.29): with liquidation off the GA draws stops only up to 95 % of the isolated liquidation distance (10x: BTC/ETH 9.08, SOL/BNB 8.98, others 8.46); champion sets not comparable; exactness_239 unaffected, no ENGINE_EPOCH bump.
- **2026-11-23 `--gate-pool all` default** (vibe-quant-clmmc.2): default runs gate every train-passer (survivors capped at top-K): champion sets not comparable with earlier ones (`--gate-pool top_k` reproduces them); exactness_239 unaffected, no ENGINE_EPOCH bump (fixed-strategy replay unchanged).
- **2026-11-21 ADAPTIVE_RSI alpha 0.7-1.0** (vibe-quant-b9uet): alpha < 0.7 refused; default 0.5->0.7; GA draws 0.7-0.9; older seeds dropped; gate_golden sections a, b re-captured. Legacy RNG golden re-pinned.
- **2026-11-20 catalog origin for start-dependent indicators** (vibe-quant-d42m5): ADX (period > 16), VIDYA and ADAPTIVE_RSI compute from the coin's first stored bar in screening, discovery and validation, so a value at bar T no longer depends on the run's start date; ADX period <= 16 keeps the old path (239 ADX(12): exactness_239 and `--validation-multi` IDENTICAL). Such runs move in both tiers; paper refuses them. 1m runs slower (MEASURED ADAPTIVE_RSI screening 1.76 s -> 9.9 s; vibe-quant-mjw5l).
- **2026-11-17 liquidated close, one formula** (vibe-quant-nxmxd.4): validation books a liquidated trade's PnL as `book.realized - hit.loss` (`trade_ledger.liquidated_close`), same value in another float order: VALIDATION liquidation-on numbers move 1-2 ulp (screening, discovery, liquidation-off bit-identical); `exactness_tiers.py` C3-isolated-125 moved (commit 7e7d8a96).
- **2026-11-07 indicator definitions** (vibe-quant-0qkus): RSI, ATR, NATR, ROC (ROC(1) now valid), STOCH (%K/%D, ready at k+d-2, +2 rows), BBANDS and KC now use the standard definitions (Wilder RSI/ATR, n-bar ROC, SMA-%D, close bands), sizing ATR included; every strategy/champion using them moves in both tiers, and a pre-break warm-start seed returns the hash-mismatch path; exactness_239/tiers, gate_golden, exactness_carry IDENTICAL.
- **2026-11-03 costed sizing** (q3eq5.1/.2): backtests size on the costed balance; validation slippage from bars before each fill. 239 screening `0.7974716442710239` -> `0.8004025908124951`, validation `0.780452281957465` -> `0.7847999777719458`, BTC+ETH `0.4956474374166028` -> `0.4936201617334757`.
- **2026-10-27 funding exit per coin interval, ledger order** (vibe-quant-nxmxd.3): screening/discovery, mode off and on, fund a boundary SL/TP exit to its 1m instant whenever the execution bar is longer than the coin's funding interval (8h if unarchived); a refined coin lacking 1m LAST fails closed (`MissingExitBars`). SOLUSDT 239@4h 2022-06..2023-01 funding `-0.7586179717327` -> `0.008269317554700006`, Sharpe `-0.5764731164989252` -> `-0.5938515057405221`, PF `0.6901515415594952`. BTC 4h/1d unchanged. Screening open-mark/PF sums follow validation's order.
- **2026-10-20 complexity cap 10** (vibe-quant-dktr4): `max_complexity` 8 -> 10 scalar genes, both sides, GA-enforced by a no-RNG trim; long/short runs bit-identical, both/mixed differ, earlier both champion counts stricter.
- **2026-10-19 explicit pool restricts MA genes** (vibe-quant-nn7i7): explicit `--indicator-pool` without KAMA/VIDYA/FRAMA draws no MA genes; `None` unchanged.
- **2026-10-19 DSR q95 critical value** (vibe-quant-rf0y7): DSR passes iff the legacy PSR statistic >= the run's MC spread (`q95-mean` of the correlated max) instead of 1.6449; null FPR <= 0.0200 (G2); earlier champion counts stricter; `--dsr-rule emax` reproduces old runs.
- **2026-10-18 direction=both own short genes** (vibe-quant-inydc): both genomes get their own short genes (before, short copied long and never fired). Long/short-constrained runs bit-identical; stored both champions replay long-only.
- **2026-10-18 DSR effective N** (vibe-quant-megq1): DSR N = the effective count (see Gates); re-check `notes.dsr_trials.gate_n`, earlier verdicts used raw cumulative N (4h 22121, 1d 167).
- **2026-10-17 1d funding exit timing** (vibe-quant-6osqd): a 1d boundary SL/TP exit (also WFA/CV) is funded to the 1m bar that hit the level, not the 1d close. Strategy 239@1d: funding `0.884065761488` -> `0.6492302331569999`, Sharpe `0.44582502161732274` -> `0.4484290923485963`, return `0.03747980496684309`, PF `1.902714157961589`, max DD `0.017959082659545107`.
- **2026-10-16 NOT a break:** opt-in margin limit and bankroll grid; a grid run compares only with its own grid (DSR N).
- Older breaks (2026-10-15 and before): [`comparability-breaks-older.md`](comparability-breaks-older.md).
