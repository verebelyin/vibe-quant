---
name: discovery-run
description: Run a full discovery cycle — parallel GA discoveries sized to the CPU, monitored with disc_watch.py, then overfitting checks, validation, journal entry and error triage. Use when the user says /discovery-run, "run discoveries", "find new strategies" or "discovery batch".
---

# Discovery Run

Cycle: journal review, N parallel discoveries, monitoring, gate check, promote/validate, log audit, summary, journal entry. A strategy that looks great is a bug until gates, validation and paper agree; report 0 champions as 0 champions.

## Shell and sandbox rules

- Search with `rg`, never `grep`. Never name a shell variable `status` (read-only in zsh); use `run_status`.
- Run every `curl` to localhost and every `python3 -c` that opens sqlite with `dangerouslyDisableSandbox: true` (the sandbox returns empty bodies, which look like JSON errors).
- Keep `$rid` in the shell layer; pass values into python via `sys.argv` or `?` placeholders. Format DB values only after a `None` check.
- Wait with Monitor or a `run_in_background` `until` loop; foreground `sleep N &&` is blocked.
- State DB: `data/state/vibe_quant.db`. Discovery runs are `backtest_runs` rows with `run_mode='discovery'` (no `discovery_runs` table). Key tables: `backtest_runs`, `strategies`, `backtest_results` (one row per run), `sweep_results` (screening metrics), `background_jobs`.

## Setup (once per batch)

1. **Choose `PORT`.** Port 8000 may hold a foreign server. Probe `curl -s localhost:$PORT/health`; `{"status":"ok"}` means our backend. A port that answers anything else, or nothing but is held (`lsof -i :$PORT`), is not ours: take another (e.g. 8010).
2. **Start the backend detached** with the Bash tool's `run_in_background: true` and a long `timeout`, using `PORT=8010; .venv/bin/uvicorn "vibe_quant.api.app:create_app" --factory --port $PORT`. Shell state does not persist between Bash calls, so start every later command with `PORT=8010;` too (`PORT=8010 cmd --port $PORT` expands `$PORT` before the assignment). Never use `&` plus `sleep`: a backend killed at the tool time limit takes its discovery runs with it and leaves them `failed`.
3. **Wait for readiness:** `until curl -s localhost:$PORT/health >/dev/null; do sleep 1; done`. Use `http://localhost:$PORT` in every call below.
4. **Cores and workers.** `sysctl -n hw.ncpu`. Launch with explicit `max_workers` so runs × workers ≤ cores (API default 4 per run; 4 runs on 10 cores = 16 processes, load ≈ 29, backtests slowed from 1.5–3 s to 20–60 s). On 10 cores: 2 runs × 4, or 4 runs × 2. More runs than fit go in waves.
5. **Data coverage.** Every symbol/timeframe must cover the window, including the funding history. Check the catalog the engine reads: `curl -s -X POST localhost:$PORT/api/backtest/validate-coverage -H 'Content-Type: application/json' -d '{"symbols":["BTCUSDT"],"timeframe":"4h","start_date":"2022-01-01","end_date":"2026-09-01"}'` and require `has_data` true with the start/end dates you need (launch-time 422 pending, bead vibe-quant-b5xfj). Archive overview: `GET /api/data/coverage`.
6. **New symbols or date ranges need the user's OK before ingesting** into `data/archive` or `data/catalog` (worktrees share them). Command once approved: `.venv/bin/python -m vibe_quant.data ingest --help`.
7. **Stale progress files.** Run ids collide across DB copies sharing `logs/`. Before launch: `ls logs/discovery_<next ids>_progress.json` and delete leftovers.
8. Print the live genome pool: `.venv/bin/python -c "from vibe_quant.discovery.genome import INDICATOR_POOL as P; [print(k, v.param_ranges) for k, v in P.items()]" 2>/dev/null`.

## Phase 1: journal review and combos

Read `docs/discovery-journal.md` with offset/limit (it is huge). List tried combos and results. Pick the planned number of NEW combos of 2+ complementary indicators (trend+momentum, volume+oscillator), and present them to the user before launching. Read the journal's latest day summary first: price momentum decayed after mid-2022 (hedged alt-vs-BTC and cross-sectional momentum: Sharpe >1.2 in 2020-10..2022-05, about 0 afterwards), so new information (OI, liquidations, order flow) is the open lever; more GA budget on the same OHLCV families is not.

Heuristics from the journal:
- CCI, STOCH, MFI+WILLR are the best past pieces; ADX is ignored by the GA; MACD's threshold range is narrow; pure momentum pairs (RSI+ROC, MFI+RSI) are redundant.
- The GA tends to converge to single-indicator strategies; multi-indicator ones lose more in validation.
- Set `direction` when the hypothesis has one: with most genomes at score 0, selection is random and populations drift (14/18 shorts in an uptrend).
- Context indicators (FUNDING, FUNDING_Z, BTC_TREND, BTC_ROC, OWN_TREND) work in discovery only when named in `indicator_pool`; they are slow (1h 2-sym 4.3 y ≈ 8 min/gen at pop 20); screening/validation only, paper/live refuse them.
- To add an indicator to the pool: `INDICATOR_POOL` in `vibe_quant/discovery/genome.py` (`params` name→[min,max], `threshold_range`), plus `vibe_quant/dsl/compiler.py` and `vibe_quant/dsl/indicators.py`. Rust-native indicators are about 10× faster than pandas-ta ones (TEMA, WILLR, ICHIMOKU, VOLSMA are pandas-ta).
- Pre-register a falsifier in the journal before a follow-up run (e.g. best pooled ann Sharpe < 1.0 or no DSR p < 0.5 → reject). For long-only crypto, benchmark against equal-weight buy-and-hold.

## Phase 2: launch

`POST /api/discovery/launch` (one call per combo, as parallel Bash calls). Body fields: `population`, `generations`, `symbols`, `timeframes`, `indicator_pool`, `direction` (`long|short|both|null`), `max_workers`, `symbol_agg` (`portfolio|worst`), `eval_windows` (default 3), `seed`, `cross_window_months`, `wfa_oos_step_days`, `train_test_split` (0.8; last 20% is the holdout), `start_date`/`end_date`, `convergence_generations` (the effective value is capped at generations // 2; the log's `Config: convergence_gens=N` shows it).

```bash
curl -s -X POST localhost:$PORT/api/discovery/launch -H 'Content-Type: application/json' -d '{"population":12,"generations":8,"symbols":["BTCUSDT"],"timeframes":["4h"],"indicator_pool":["A","B"],"direction":null,"max_workers":4}'
```

Record every `run_id` and the launch timestamp (journal timing column). Default budget 20 minutes wall-clock per batch: shrink population/generations rather than exceed it, unless the user sets a bigger budget.

Sizing (MEASURED, 4h, ~2.75 y): single-symbol backtest 1.5–3 s idle, 20–30 s under 4-way contention; TREND_ENSEMBLE 20–65 s; `symbol_agg=worst` on 3 symbols = 3 backtests per genome; 15m ROC/DONCHIAN/RSI/CCI/NATR ≈ 150 s/gen at pop 20–24, BBANDS/MFI 80–170 s per backtest (avoid). Pop 10–12 × 4 gens takes 1–6 min. Reduce pop/gens with pandas-ta indicators (1: pop 10 × 6; 2: pop 8 × 5). Higher-budget runs (30–40 min): pop 20 × 20, Rust-native only.

Recipe that gives the GA a gradient: at most 2 parallel runs, multi-symbol `portfolio`, window ≥ 3.75 y, `eval_windows` 2, pop 16–20 × 6–8 (3-sym 4h 4.3 y with TREND_ENSEMBLE 176 s/gen at pop 16; 2-sym 1h 3.75 y 189 s/gen at pop 20).

Hard facts about scoring:
- Zero score is a hard gate: worst-window `trades < min_trades` (50) or `total_return <= 0` → 0 (`fitness.py`); the overtrade penalty can also zero >300-trade genomes. 4h single-symbol on ~2.2 y trains gives 17–48 trades, so every genome scores 0; use ≥4 y windows on 4h or a multi-symbol portfolio. 1h BTC 1.7 y with `eval_windows` 3 also scored all 0.
- Never pass `no_bootstrap_ci` or `--no-dsr` to get champions (forced champions collapsed in validation). Move the search instead: least-mined timeframe (15m, ~0.3k prior trials) and more trades. Bootstrap floor: 1m 0.5, 4h/1d 0.0, 5m/15m/1h 1.0.
- DSR trial count N is cumulative across ALL prior same-timeframe runs with an overlapping train window, any symbol (4h ≈ 22k, 1m 63k, 1h 3.6k, 5m 0.8k, 15m 0.3k at 2026-10-11); read `notes.dsr_trials` (`gate_n`, `prior_n`, `prior_run_ids`). N is rarely binding: ann Sharpe 0.86 over 2.2 y fails at N=1; p<0.05 needs ann Sharpe ≈ 3.5 at N=600 and >4 at N=21 500. Check a rejected candidate with `calculate_dsr(deannualize_sharpe(sr), n, T)` before blaming N.
- Gate-rejected candidates keep no DSL (`guardrail_rejections` holds uid + reasons). With 0 champions there is nothing to promote or validate; say so.
- `OrderRejected ... REDUCE_ONLY ... would have increased position` and `Canceling REDUCE_ONLY` in logs are by design.

## Phase 3: monitor

Run `scripts/agents/disc_watch.py RUN_ID [RUN_ID ...]` under Monitor (`.venv/bin/python scripts/agents/disc_watch.py 901 902`). It prints improving GEN lines, COMPLETE, champion and error lines, and exits when every run is terminal. Monitor only through it: ad-hoc `tail | rg | cut` pipes buffer and went silent for 30 minutes while runs completed. Also schedule a fallback wakeup (ScheduleWakeup) at ≤ half of any user deadline that checks run status directly. Post a status table (run, status, gen, best, mean, ETA) about every 3 minutes of wall-clock; the API view is `GET /api/discovery/jobs/$rid/progress`, but trust `=== GEN n/m ===` lines in `logs/discovery_<id>_<ts>.log` over it.

## Phase 4: results and gates

`GET /api/discovery/results/$rid` → `strategies[]` with `dsl, score, sharpe, max_dd, pf, trades, return_pct` (top 3 are enough). DSR and guardrail detail is in the log: `rg "DSR|guardrail|deflated|Gate FAIL|No champion" logs/discovery_<id>_*.log -A5`. A candidate failing DSR (p > 0.05), bootstrap CI, cross-window, WFA or holdout is eliminated; note which gate. A run's champion list comes from the gates, never from the raw top-score list.

## Phase 5: screening replay and validation

For each champion: `POST /api/discovery/results/$rid/promote/0?mode=screening` returns `{strategy_id, run_id, name, mode}`; promote auto-runs screening (launch persists `start_date`/`end_date`, so no date fixing). Check `GET /api/results/runs/<screening_run_id>`: trades, Sharpe, return, PF must match discovery exactly when the run used `eval_windows=1`; otherwise investigate and file a bead. If a screening needs a re-run: `.venv/bin/python -m vibe_quant screening run --run-id N`. Screening metrics live in `sweep_results`.

Validation: `POST .../promote/0?mode=validation` (holdout range by default; `validation_range=full` for the whole range), or re-run an existing row with `.venv/bin/python -m vibe_quant validation run --run-id N`. Start all validation scopes at once (portfolio, per-symbol, windows; 6 parallel 4h validations took ~6 min). Expect the same trades with a 5–20% Sharpe drop (fill model, 60 ms cloud latency, fees); flag a >30% drop or a trade-count divergence >10%. Results: `GET /api/results/runs/<id>` (`sharpe_ratio, total_trades, total_return, profit_factor, max_drawdown, win_rate, total_fees`; these names differ from the discovery ones).

## Phase 6: log audit and beads

For every run: `rg -c "ERROR|Exception|Traceback" logs/{discovery,screening,validation}_<id>*.log`. File a bead for each distinct problem the moment you see it (log error, screening mismatch, crash, corrupt file): `bd create --title="[discovery-run] <desc>" --description="run ids + error" --type=bug --priority=2`.

## Phase 7: summary and journal

Summary table, one column per run: discovery score/Sharpe/trades, gate results, screening match, validation Sharpe/return/DD/PF. Then append a batch entry to `docs/discovery-journal.md` in the format of the previous entries: date and batch number, goal, fixes applied, configuration table (run ids, indicators, pop, gens, timeframe, wall-clock per run, status), full-pipeline table, winners, issues, key findings, comparison with prior batches, recommendations. Commit the journal and push (`git add docs/discovery-journal.md && git commit && git pull --rebase && git push`); the work is not done until the push succeeds.

## Key rules

- Review the journal before choosing combos.
- Honest numbers: gates fail closed; report failures and 0-champion runs plainly.
- Audit every log; bead every problem.
- Backend detached, one `PORT`, runs × `max_workers` ≤ cores, monitoring through `disc_watch.py`.
