# Deep audit — 2026-10-02

Scope: all of `vibe_quant/` (+ result-display bits of `frontend/`). 8 parallel subsystem audits
(DSL/compiler, screening+metrics, validation, overfitting, discovery, risk/paper/live, data,
API/jobs/DB), each with repro scripts; top findings re-verified by hand.
Baseline at audit time: `pytest` 2349 passed / 4 skipped, ruff+mypy zero — **none of the
below is caught by the suite.**

Beads: epic `vibe-quant-e70tl` — C1–C6 = `.1`–`.6`, H1–H16 = `.7`–`.22`, medium/low triage = `.23`.
Repro scripts (ephemeral, not committed): `/tmp/vq_audit/`, `/tmp/vqaudit/`, `/tmp/a*.py`.

Legend: **[V]** re-verified by hand · **[R]** agent repro script / real DB evidence · **[C]** code read.

## Status — 2026-10-03

All CRITICAL (C1–C6) and HIGH (H1–H16) findings fixed on branch `audit-fixes` (5 parallel agents,
merged + integration-tested), plus most MEDIUM items in the touched files. Gates: pytest 2700+ pass,
ruff/mypy zero, frontend build + vitest green.

Verified end-to-end on real data (DB copies): strategy 239 screening `1.3165049716553048`/68 trades
(bit-identical ×2); validation `0.7802305953007851`/67 trades (bit-identical ×2); v16 migration on a
real-DB copy (runs 337/556 → 66/62 trades, 459 → 105 trade rows, 368 → failed); discovery with
default holdout → gates fail closed with recorded rejections; promote → holdout validation via API.

Integration fixes found while merging: paper `/stop` PID-identity check, warm-start re-injecting
legacy ATR genes, consistency check comparing different windows, tests able to migrate the real DB.

**Semantics break:** all screening/discovery/validation scores before 2026-10-03 are not comparable.
Remaining MEDIUM/LOW items (mostly data layer: Vision filler candles, `update` skips funding,
partial candles, silent ingest holes, instrument spec drift): bead `vibe-quant-e70tl.23`.

## Bottom line

Backtest numbers are currently not trustworthy enough to deploy capital on:

- discovery can score one genome with another genome's code (C1),
- validation fills every TP/SL at the bar extreme (C2),
- profit factor and max DD (45% of fitness) are computed on the wrong basis (H1, H5),
- "worst-of-N" eval windows are actually a mean (C6),
- every overfitting gate fails *open* when all candidates fail (C5).

The paper/live path has no risk enforcement at all, and its halt/kill/close-all don't work (C4).
Existing champions + `docs/discovery-journal.md` conclusions should be considered suspect until
C1/C2/C5/C6/H1/H5 are fixed and champions re-validated.

---

## CRITICAL

### C1. Genome name collision → wrong strategy code evaluated / promoted [V][R]
- `discovery/operators.py:235` `clone()` keeps `uid`; `mutate()` clones → elite and mutant share
  `genome_{uid}` (`genome.py:608`).
- `screening/nt_runner.py:115-121` `_COMPILE_CACHE` keyed by DSL JSON but returns
  module path `vibe_quant.dsl.generated.{name}`; `compiler.py:229-238` overwrites `sys.modules[name]`.
  Re-evaluating elite A after mutant B (same worker) runs **B's code**. Repro: A 23 trades/0.737 →
  B 25/−0.182 → A again 25/−0.182.
- Promote/export (`api/routers/discovery.py:616,714-729`) reuse strategy row by **name** → validates
  a different genome's DSL. 13 runs in DB (e.g. 607, 615) have same-name/different-DSL champions.
- Fix: fresh uid on mutate/crossover; module name = content hash; promote lookup by DSL equality.

### C2. Validation fill model fills TP limits / SL stops at bar extreme [V]
- `validation/fill_model.py:153-206` always returns a synthetic book; NT uses it for resting limits &
  triggered stops, bypassing the clamp to limit/trigger price.
- Repro: long TP limit 10100, bar high 10300 → filled **10300** (NT default 10100); SL stop 9900,
  bar low 9700 → **9700**. Real BTC 1m 2025: TP +6.7–11.4 bps/hit, SL −6–10.6 bps/hit.
- Tight-TP/wide-SL strategies look too good; tight-SL ones too bad.

### C3. Price-vs-indicator crossovers broken [V][R]
- `dsl/compiler.py:1548-1557,1648-1666`: prices have no prev value, so `close crosses_above ema`
  → `close > ema and close <= prev_ema`. For EMA/KAMA (MA moves toward price) this is **impossible**
  (0 / ~1411 true crosses fire); SMA fires 152× with only 89 real; DEMA 307× / 164 real.
- Hits discovery MA genes (`close crosses_* kama/vidya/frama`) and `bollinger_squeeze.yaml` exits.

### C4. Paper/live has no risk control; halt/kill/close-all broken [V][R]
- `paper/node.py:339-343`: sizing/risk configs validated then discarded; no risk actor wired
  (`StrategyRiskActor`/`PortfolioRiskActor` have zero non-test callers). Max DD / daily loss never enforced.
- Halt/kill (`node.py:449-453,780-819`, `api/routers/system.py`): 1st SIGUSR1 only sets PAUSED label;
  2nd cancels **all orders incl. reduce-only SL/TP**; strategies keep running and keep entering.
- Close-all (`node.py:598-601`) calls `close_all_positions()` without instrument → TypeError swallowed,
  API still says `closing_positions`.
- Sizing (`dsl/templates.py:286-295`) falls back to `make_qty(1.0)` (1 BTC!) when account missing /
  equity ≤ 0 / price ≤ 0; equity read from `account.currencies()[0]` (may be BNB, not USDT).
- No pending-order guard + exit not reduce-only (`templates.py:57-120`): repro with latency → double
  entry (2.665 vs 1.333 intended), long-only strategy ends **short**. SL/TP sized on first fill only
  (`PositionChanged` unhandled) → partial-fill remainder unprotected.

### C5. Overfitting gates fail open [V][R]
- Discovery DSR: `discovery/pipeline.py:1513-1514` returns `None` when **all** top-K fail soft
  guardrails → `:698-700` keeps all unfiltered. Last 5 discovery runs (854–867): every champion
  failed DSR (p=0.64–0.98), all persisted.
- Cross-window / WFA / holdout all-fail → "keeping originals" (`pipeline.py:842-844,1061-1066,1196-1198`).
  Run 799's sole champion failed WFA 0/1, persisted.
- Cross-window counts in-sample window 0 as a pass, `min_pass=2` (`:981-990,1028`).
- Holdout leaks: shifted cross-windows extend into holdout (`:954-962`); WFA windows sit inside
  holdout yet filter champions; holdout never gates; API defaults `train_test_split=0`,
  `wfa_oos_step_days=0` → default discovery has no OOS gate. Promote validates over the discovery's
  full (in-sample) range (`api/routers/discovery.py:748`).

### C6. `eval_windows` is mean-of-N, not worst-of-N [V]
- `discovery/backtest_fn.py:114-127` averages Sharpe/return, sums trades; only per-window check is
  ≥1 trade. Class docstring, CLI help, replay note **and CLAUDE.md** claim worst-of-N.
- Windows Sharpe [6,−1,−1], trades [48,1,1] → mean 1.33, 50 trades, fitness 0.587 (worst-of-N: 0).

---

## HIGH

### H1. Profit factor = daily-return PF, not trade PF [V][R]
`screening/nt_runner.py:436`, `validation/extraction.py:230-231`. NT's realized-PnL PF returns None →
falls back to sum(+daily)/sum(−daily). Synthetic trade PF 1.345 reported 3.335; DB run 843 0.684 vs
0.485 trade PF. Feeds 20% of fitness + `min_profit_factor` + Pareto.

### H2. Pandas-path indicator params unsweepable; ignore `timeframe` [R]
- `compiler.py:1253-1260,1317-1322` bake literals (`compute_adx(_df, {"period": 14})`) → sweeps/WFA
  over ADX, MACD, KAMA, MFI, WILLR, TEMA, ICHIMOKU, VOLSMA, all plugins are no-ops (strategy 239:
  ADX 5/14/40 bit-identical). DSR trial count inflated.
- `nt_runner.py:198-202` silently drops unknown override keys (e.g. `take_profit.risk_reward_ratio`
  → field is `take_profit_risk_reward`; STOCH `period_k/period_d`).
- `compiler.py:846-873`: pandas buffer only gets primary-TF bars → "4h ADX" on 1h strategy is 1h ADX.

### H3. DSR trial count / NaN handling [R]
- `overfitting/pipeline.py:149` `num_trials=len(candidates)`; CLI never passes `total_trials`;
  `--observations` default 252 regardless of window → GA champion with 6000 evals: p=0.014 PASS vs
  correct 0.42 FAIL. DB sweep rows 133/134/135/279 stamped `passed_deflated_sharpe=1` with N=1.
- `screening/pipeline.py:215` skips DSR when exactly one combo survives.
- `screening/pipeline.py:224-231` variance over all results incl. NaN/−inf → one NaN rejects all
  (19 NULL + 9 −inf Sharpe rows already in `sweep_results`).

### H4. WFA / CV correctness [R]
- `overfitting/wfa.py:442-448` efficiency = total OOS / total IS (not length-normalized) → stationary
  edge gets 0.33, fails; mean IS < 0 & OOS > 0 → efficiency 5.0 → a strategy losing 18.6% passes.
- `overfitting/pipeline.py:227-228` WFA dates hardcoded 2024-01-01..2025-12-31; `:241` single-combo
  grid → never re-optimizes per window.
- `overfitting/nt_cv_runner.py:197` `runner({})` → every candidate gets identical CV verdict.

### H5. Max drawdown ignores open/intraday losses [R]
Validation takes NT daily-cash-balance DD (`extraction.py:144-233`); screening uses closed-trade
fallback (`nt_runner.py:282-283,475-480`). Repro: 40% adverse excursion recovered to entry → DD 0.00025
vs true 0.20. 25% of fitness.

### H6. Same-bar SL/TP ambiguity always resolved O→H→L→C [R]
`validation/venue.py:208-224` never sets `bar_adaptive_high_low_ordering` → longs book TP, shorts book
SL when both inside a bar. Screening (strategy-TF bars only) flatters longs, penalizes shorts.

### H7. Validation silently degrades to strategy-TF fills [R]
`validation/runner.py:1146-1169` + `:677-681`: if window extends past 1m data by 1 day (e.g. end=today,
data ends 2026-03-17), 1m detail dropped with INFO log, latency kept → fills at **next 4h bar close**
(repro: signal 10000 → fill 10800).

### H8. ATR threshold range in wrong units [R]
`dsl/indicators.py:869` range (0.001, 0.15) but ATR is absolute price (BTC 1m min 0.25, median ~40) →
ATR genes are constant true/false. 208 persisted BTC champions contain ATR genes; run 607 #1 is
effectively "always short".

### H9. Result persistence / status [V][R]
- Rerun appends duplicate `backtest_results`/`trades` rows; `get_backtest_result` `fetchone()` no
  ORDER BY (`state_manager.py:622-649,734-754`). Runs 337/556 show stale 0-trade row (real: 66/62 trades,
  Sharpe 2.32/4.70); run 459 has 315 trade rows for 105 trades → equity curve −19.4% vs real −6.5%.
- 0-trade validation marked `failed` then overwritten to `completed` by `mark_completed()`
  (`vibe_quant/__main__.py:53-56`, `validation/__main__.py:34`). Runs 337/368/556.
- Validation↔screening collapse check picks arbitrary `LIMIT 1` sweep row (`validation/consistency.py:104-114`)
  → strategies 70/71/72 compared against (sharpe=−inf, trades=0) → collapse flag can never fire.

### H10. Notes editor destroys discovery results [R]
`api/routers/results.py:356-371` + `NotesPanel.tsx`: `backtest_results.notes` holds discovery JSON;
UI notes box saves free text over it → `strategies: []`, promote 404.

### H11. Job manager kills wrong processes [R]
`jobs/manager.py:383-442`: data jobs never heartbeat → marked stale after 120 s → Backtest page's
auto `/jobs/cleanup-stale` SIGKILLs the download (no data job has ever `completed`). No PID identity
check → PID reuse kills unrelated processes (kill switch SIGUSR1s stored paper PIDs too).
Bodyless POSTs (`cleanup-stale`, `indicators/reload`, heartbeat) are CSRF-triggerable.

### H12. Settings DB switch splits state [R]
`api/routers/settings.py:262-292` swaps only `app.state.state_manager`; job manager keeps old DB,
subprocesses get no `--db` → new run in new DB, job row + subprocess hit old DB's run with same id.

### H13. Paper ≠ validated config; no restart recovery [C]
- `node.py:342` uses compiled DSL defaults, not the validated run's sweep overrides
  (`validation/runner.py:1016-1040`); leverage never set on `BinanceExecClientConfig`.
- No checkpoint load, no `external_order_claims`, `on_start` never syncs position → after restart
  strategy thinks flat, opens a second position.
- API paper path: `/start` writes `symbols: []` (always fails validation); default trader_id
  `paper_{id}` aborts NT (needs `-`); `testnet` defaults **False → LIVE**; UI API keys dropped;
  CLI reads `BINANCE_TESTNET_*` env even for live. `/start` allows concurrent sessions.

### H14. Funding not modeled in screening/discovery [R]
`screening/nt_runner.py:472-473` `total_funding = 0.0`. SPEC + CLAUDE.md say screening models
funding. GA ranks with no carry cost (strategy 239: −0.8% equity omitted).

### H15. Secrets / monitoring [R]
- Telegram bot token logged via httpx INFO + failure warning (`alerts/telegram.py:242,264-275`).
- `ErrorHandler.handle_error` has no callers → no alerts ever fire.
- Reconciliation `parity_rate()` returns 1.0 with zero trades; validation position events have
  `position_id=""` → all dropped (`reconciliation.py:109-114,214-222`). Reports perfect parity on no data.

### H16. Frontend shows wrong money numbers [V]
`CostBreakdown.tsx:46`, `FuturesAnalytics.tsx:43` do `total_return/100*balance` but `total_return` is a
fraction → run 870 Net PnL $0.52 instead of $51.72. `ExportPanel.tsx:25-34,63-73` prints fractions as %
(5.17% → "0.05%"). `DrawdownChart.tsx:68` domain expects negatives, API sends positives.

---

## MEDIUM

Validation / execution realism
- 1m/5m validation: no 5s data → latency dropped, only 30% deferral → 70% fill at signal-bar close
  (same-bar look-ahead per `test_fill_timing.py`); 5m asks for 5s detail instead of 1m (`runner.py:1056-1057,1113-1117`).
- Open position at end dropped from trades+return when latency on (`extraction.py:272`); `total_trades=1` w/ empty trade list.
- Random baseline "Sharpe" is a t-stat `mean/std*sqrt(n)` (~2× too low vs annualized) → biased toward
  "genuine alpha" verdict; no p-value (`random_baseline.py:266-276`).
- Liquidation never enabled (`venue.py:208-224`); `reject_stop_orders=False` (Binance rejects −2021, live position unprotected).
- Post-fill slippage charged on entry leg only (`extraction.py:322-328`).
- Screening ignores launch `initial_balance`/`leverage` (always 1000 USDT/10x, `nt_runner.py:247`); validation honors them.
- Instrument specs drifted: SOL tick 0.001 (real 0.01), SOL size_increment 1 (real 0.01 since 2025-04-09),
  min_notional 5 (real BTC 50/100?, ETH 20) (`data/catalog.py:58-70,112`).

DSL / indicators
- Trailing stop loosens on first update (`templates.py:187-221`): SL 96 → 94.
- Time filters / funding avoidance use `bar.ts_event` = **open** in backtest but **close** live (NT Binance
  adapter) → inverted blocking on ≥15m; filters also `return` before exits/trailing updates (`compiler.py:898-911`).
  Latent: 0/241 strategies use time filters.
- NT indicators ≠ standard: WMA==SMA, ROC(n)==pandas ROC(n−1), RSI EMA-smoothed not Wilder (14 pts),
  STOCH k unsmoothed, BBANDS/KC on typical price, ATR SMA-smoothed (~8%).
- ICHIMOKU outputs mislabeled (`compute_builtins.py:271-276`).
- ADAPTIVE_RSI depends on buffer-trim point (`plugins/example_adaptive_rsi.py:92-96`).
- Translator drops exit conditions, `timeframe_override`, `trailing_stop_pct`, ignores `logic: or` (`translator.py:122-138,229-239`).

Discovery GA
- Immigrant injection uses previous gen's scores on new population → elite killed in 345/500 sims;
  prod run 512 gen 18 best 0.51 → gen 19 0.00 (`pipeline.py:652-658`). `immigrant_fraction=0` still injects 1.
- Deterministic crowding passes parents' scores as offspring fitness → ~no selection pressure (`pipeline.py:1357-1362`).
- Dead genes: ADAPTIVE_RSI `alpha`, STOCH `period_d`, float periods truncated → fake diversity, duplicate trials (`genome.py:448-450`).
- SL/TP ratio penalty reads generic fields; direction=BOTH uses per-direction (`fitness.py:430`).
- Persisted WFA efficiency ≈ Σ OOS / mean-per-window IS → inflated by N_oos×N_eval (`discovery/__main__.py:956-961`).
- Multi-seed DSR trials undercounted (per-seed N, pooled pick) (`discovery/__main__.py:230`).
- 1m-short regime gate compares mismatched windows; neutral base regime passes (`api/routers/discovery.py:480-507`).
- Discovery silently falls back to mock backtests when catalog data missing (`discovery/__main__.py:691`).

Overfitting stats
- CV robustness std<1.0 rejects true-Sharpe-4 strategy 88% of time on 1y (`purged_kfold.py:461`).
- Purge/embargo only on adjacent fold (`purged_kfold.py:301-309`).
- Bootstrap i.i.d. → false-pass 13% at ρ=0.5 (`bootstrap_sharpe.py:118`).
- DSR mixes daily Sharpe w/ per-trade skew/kurt (`nt_runner.py:486-542`).
- Cross-window API rows mislabeled (window 0 shown "+3mo") (`api/routers/results.py:74`).
- OverfittingBadges thresholds ≠ backend (DSR z>0, PKFold >0 vs 0.5, bootstrap 1.0 vs 0.0; "All Passed" with no tests).
- `--allow-mock` writes `passed_*` with no provenance.

Data
- Binance Vision flat zero-volume filler candles where exchange traded (BTC 2024-10-28 20:00–21:14,
  2025-01-14, 2025-01-29) — ingested and passed by `verify.py`.
- `update` never fetches funding → holes (BTC 2026-02-23→03-10); validation charges 0 funding inside hole (`funding.py:109`).
- 15 still-forming candles archived pre-`fc6785c`, unfixable via `INSERT OR IGNORE` (`archive.py:147-153`).
- Silent ingest holes: downloader errors → None → "no data"; failed month never backfilled; 1-month range makes 0 calls (`ingest.py:385-423`).

API / jobs
- WS broadcast iterates live set while awaiting → `Set changed size` → launch endpoints 500 after job started (dup-launch risk), ping task dies (`api/ws/manager.py`).
- `sizing_config_id` / `risk_config_id` / `overfitting_filters` accepted at launch, never applied (`api/routers/backtest.py:92-103`).
- UI sweeps broken: SweepBuilder payload fails DSL validation; keys don't match config field names (`SweepBuilder.tsx:148,195`).
- Paper: SIGWINCH used for close-all (terminal resize flattens); paper audit events lose payload, halt stored `passed: True`.
- Ethereal ingestion rounds bars to first bar's precision (`ethereal/ingestion.py:272-280`).

## LOW (abridged)
- Last aggregated bar can be partial (1 min of data in a 4h bar) (`catalog.py:229-231`).
- `verify.py` misses gaps ≤4 min; data-quality endpoint always errors (`bar.get()` on NT Bar).
- NaN passes hard filters, always on Pareto front, breaks `rank_by_sharpe` sort (`screening/grid.py`).
- Mixed int/float sweep lists → every combo −inf; screening lacks `raise_exception=True` (also discovery).
- `replay_drift.py:113-116` one-sided, sign-blind.
- Sharpe series ends at last fill; √252 on 365-day data (consistent, conservative).
- WFA: cold indicators per window; max DD = max of windows.
- `source:` field ignored by compiler; Donchian upper includes current bar (breakout impossible).
- Pandas-path warmup returns 0 not NaN (KAMA p<10 → `close > kama` true in warmup).
- `compiler_version_hash` excludes `compute_builtins.py`, `derived.py`, `plugins/*`.
- Risk actors inert even if wired (`portfolio.account(venue=None)` raises; `Decimal(str(Money))` raises; no position hooks).
- `risk/sizing.py` ignores size_increment / min qty / min notional.
- Ethereal exec: unknown status raises after venue accepted → retry duplicates; zero-address verifying contract.
- Reconciliation slippage sign inverted for shorts.
- Paper UI positions/orders mapping wrong (symbol '', entry 0).
- Log file handle leak per job; blocking `time.sleep` in async kill endpoint (up to 10 s server stall).
- Input validation: reversed/garbage dates and empty symbols → 201; leverage ≤0 silently → 10x.
- Screening early `return 1` leaves runs `running` forever (runs 190–195).
- Top-K diversity distance positional → near-duplicates both kept.
- PF NaN (no losing days) scored 0 instead of cap.

## Docs that are wrong
- CLAUDE.md: "`eval_windows` stores worst-of-N" — code is mean (C6).
- CLAUDE.md/SPEC: "screening still models leverage/funding/liquidation" — no funding, no liquidation (H14).

## Verified NOT bugs (don't re-investigate)
- No look-ahead in data layer: catalog `ts_init` = Binance close_time for 100% of bars; NT orders by `ts_init`; MTF indicators update on HTF close.
- Aggregation 1m→5m/15m/1h/4h exact; no dup/missing minutes; UTC everywhere.
- DSR/PSR formula matches paper to 3e-15 (non-excess kurtosis, √(T−1), E[maxZ]).
- Funding sign/units/settlement window correct (where data exists).
- Fees charged (maker 2 bps TP limits, taker 5 bps market/stop); MARGIN/NETTING/USDT correct.
- Market-order slippage direction correct; gap-through stops fill at open; latency ms→ns correct.
- Literal-threshold crossovers correct; SL/TP signs + per-direction overrides correct.
- Purged k-fold splits leak-free; `indicator_lookback_bars` propagation fixed.
- Overfitting CLI no longer silently uses mock runners.
- Parallel results map to correct genome/params; no fitness cache staleness (apart from C1).
- SQL uses placeholders everywhere; WAL+busy_timeout on all connections; no path traversal.
- Binance keys env-only; not in DB/API responses.
