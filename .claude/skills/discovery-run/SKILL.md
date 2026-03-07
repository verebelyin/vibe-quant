---
name: discovery-run
description: Run a full discovery cycle — 5 parallel GA discoveries (20min max), 3min status reports, overfitting validation, backtesting, journal entry, and error triage. Use when user says /discovery-run, "run discoveries", "find new strategies", or "discovery batch". Triggers the complete pipeline from discovery through validation with automated monitoring and journaling.
---

# Discovery Run

Full-cycle automated discovery: journal review, 5 parallel discoveries, monitoring, overfitting checks, backtesting, summary, and journal entry.

## Shell Gotchas (CRITICAL — read before writing any bash)

1. **`status` is read-only in zsh** — never use `status` as a variable name. Use `st`, `stat`, or `run_status` instead.
2. **Always use `rg` (ripgrep) instead of `grep`** — faster, no `|` escaping. `rg "ERROR|Exception" file` not `grep "ERROR\|Exception" file`.
3. **Don't embed `$rid` inside python3 heredocs** — shell expands it before python sees it. Use the python3 `-c` approach where `$rid` stays in the shell layer, not inside python strings.
4. **USE `sleep N &&` for polling waits** — combine `sleep 180 && <poll_command>` in a single Bash tool call with `timeout: 300000`. This prevents spamming the user with rapid tool calls. Each poll should wait 3 minutes via sleep before checking.
5. **Parallel background commands**: use `cmd1 & cmd2 & wait` pattern. Check exit codes after `wait`.

**Working polling script (copy-paste safe):**

```bash
for rid in 232 233 234 235 236; do
  echo -n "Run $rid: "
  curl -s "http://localhost:8000/api/discovery/jobs/$rid/progress" | python3 -c "
import json,sys
d=json.load(sys.stdin)
g=d.get('progress') or {}
gen=g.get('generation','?')
mx=g.get('max_generations','?')
best=g.get('best_fitness',0)
mean=g.get('mean_fitness',0)
st=d.get('status','?')
e=g.get('eta_seconds')
eta=f'{int(e)//60}m{int(e)%60}s' if e else '-'
bf=f'{best:.4f}' if isinstance(best,(int,float)) else str(best)
mf=f'{mean:.4f}' if isinstance(mean,(int,float)) else str(mean)
print(f'{st} | Gen {gen}/{mx} | Best={bf} | Mean={mf} | ETA={eta}')
"
done
```

## Indicator Catalog

### Currently in Discovery Genome Pool

These can be used directly in discovery (`vibe_quant/discovery/genome.py`):

| Indicator | Category   | Speed              | Param Ranges                                | Threshold Range |
| --------- | ---------- | ------------------ | ------------------------------------------- | --------------- |
| RSI       | Momentum   | Rust-native (fast) | period: [5, 50]                             | [25.0, 75.0]    |
| MACD      | Momentum   | pandas-ta (slow)   | fast: [8,21], slow: [21,50], signal: [5,13] | [-0.005, 0.005] |
| ATR       | Volatility | Rust-native (fast) | period: [5, 30]                             | [0.001, 0.03]   |
| STOCH     | Momentum   | Rust-native (fast) | k_period: [5,21], d_period: [3,9]           | [20.0, 80.0]    |
| MFI       | Volume     | Rust-native (fast) | period: [5, 30]                             | [20.0, 80.0]    |
| ADX       | Trend      | Rust-native (fast) | period: [7, 30]                             | [15.0, 60.0]    |
| CCI       | Momentum   | Rust-native (fast) | period: [10, 50]                            | [-200.0, 200.0] |
| WILLR     | Momentum   | pandas-ta (slow)   | period: [5, 30]                             | [-100.0, 0.0]   |
| ROC       | Momentum   | Rust-native (fast) | period: [5, 30]                             | [-10.0, 10.0]   |

**IMPORTANT:** Always verify the genome pool is current before launching. Run:

```bash
rg "INDICATOR_POOL" vibe_quant/discovery/genome.py -A 3 | head -40
```

If a needed indicator is missing, add it before discovery.

### Full DSL Catalog (available for backtesting, need genome pool expansion for discovery)

**Trend:** EMA, SMA, WMA, DEMA, TEMA, ICHIMOKU
**Momentum:** RSI, MACD, STOCH, CCI, WILLR, ROC, ADX
**Volatility:** ATR, BBANDS, KC, DONCHIAN
**Volume:** OBV, VWAP, MFI, VOLSMA

**Rust-native (fast, ~10x):** EMA, SMA, WMA, DEMA, RSI, MACD, STOCH, CCI, ROC, ADX, ATR, BBANDS, KC, DONCHIAN, OBV, VWAP, MFI
**pandas-ta only (slow):** TEMA, WILLR, ICHIMOKU, VOLSMA

**To add an indicator to genome pool**: Edit `INDICATOR_POOL` dict in `vibe_quant/discovery/genome.py` — add entry with `params` (name→[min,max] ranges) and `threshold_range` ([min, max]). Also ensure it's in the compiler (`vibe_quant/dsl/compiler.py`) and indicator registry (`vibe_quant/dsl/indicators.py`).

**NOTE:** If the journal review suggests trying an indicator NOT in the genome pool, add it to the pool first (following the ADX addition pattern from Batch 10), then proceed with discovery.

## API Reference (Critical — avoid guessing endpoints)

**Backend startup:**

```bash
# ALWAYS check if backend is already running before starting
lsof -i :8000 2>/dev/null | head -3
# Only start if nothing is listening:
.venv/bin/uvicorn "vibe_quant.api.app:create_app" --factory --port 8000 &
# Wait and verify:
sleep 3 && curl -s http://localhost:8000/api/data/coverage | python3 -c "import json,sys; print('Backend OK' if json.load(sys.stdin) else 'FAIL')"
```

Note: uses `create_app` factory, NOT `app` directly. Module is `vibe_quant.api.app`, NOT `vibe_quant.api.main`.

**Discovery endpoints:**

```
POST /api/discovery/launch                              → {run_id, status, ...}
GET  /api/discovery/jobs/{run_id}/progress              → {run_id, status, progress: {generation, best_fitness, ...}}
GET  /api/discovery/results/{run_id}                    → {strategies: [{dsl, score, sharpe, max_dd, pf, trades, return_pct}, ...]}
POST /api/discovery/results/{run_id}/promote/{idx}?mode=screening  → {strategy_id, run_id, name, mode}
```

**Results endpoints:**

```
GET  /api/results/runs/{run_id}                         → {sharpe_ratio, total_trades, total_return, profit_factor, max_drawdown, win_rate, total_fees, sortino_ratio, ...}
GET  /api/results/runs/{run_id}/trades                  → trade list
GET  /api/results/runs/summary                          → all runs summary
```

**Strategy endpoints:**

```
GET  /api/strategies/{id}                               → {id, name, dsl_config, ...}
```

**Discovery result field names** (NOT the same as results/runs):

- `score`, `sharpe`, `max_dd`, `pf`, `trades`, `return_pct`
- DSR info is in the discovery **log file**, not the API response

**Results/runs field names:**

- `sharpe_ratio`, `total_trades`, `total_return`, `profit_factor`, `max_drawdown`, `win_rate`, `total_fees`

**Known bug — discovery launch missing dates:**
Discovery launch API doesn't save `start_date`/`end_date` to `backtest_runs`. This causes screening to fail with 0 trades.

**CRITICAL ORDERING:** You MUST fix dates BEFORE promoting. The promote endpoint auto-runs screening immediately. If dates are empty when you promote, screening produces 0 trades and you must re-run.

**Correct workflow:**

```bash
# 1. Get the data range from ANY discovery log
rg "Data range" logs/discovery_{FIRST_RUN_ID}.log | head -1
# Output: "Data range: 2025-03-07 to 2026-03-07 ..."

# 2. Fix dates on ALL discovery runs FIRST (before promote)
python3 -c "
import sqlite3
conn = sqlite3.connect('data/state/vibe_quant.db')
for rid in [RUN_ID1, RUN_ID2, ...]:
    conn.execute('UPDATE backtest_runs SET start_date=?, end_date=? WHERE id=?', ('2025-03-07', '2026-03-07', rid))
conn.commit()
print('Discovery run dates fixed')
"

# 3. NOW promote (screening will auto-run with correct dates)
curl -s -X POST "http://localhost:8000/api/discovery/results/{run_id}/promote/0?mode=screening"
# Returns: {strategy_id, run_id (screening_run_id), name, mode}

# 4. Fix dates on the NEW screening runs too (promote creates them with empty dates)
python3 -c "
import sqlite3
conn = sqlite3.connect('data/state/vibe_quant.db')
for rid in [SCREENING_RUN_ID1, SCREENING_RUN_ID2, ...]:
    conn.execute('UPDATE backtest_runs SET start_date=?, end_date=? WHERE id=?', ('2025-03-07', '2026-03-07', rid))
conn.commit()
print('Screening run dates fixed')
"

# 5. Check if screening already ran successfully (promote auto-runs it)
curl -s "http://localhost:8000/api/results/runs/{screening_run_id}" | python3 -c "
import json,sys; d=json.load(sys.stdin)
tr=d.get('total_trades',0)
print(f'trades={tr}' + (' — OK' if tr > 0 else ' — NEED RE-RUN'))
"

# 6. If 0 trades, re-run screening manually:
.venv/bin/python -m vibe_quant screening run --run-id {SCREENING_RUN_ID}
```

**Validation runs — create manually** (promote endpoint may fail for re-promotes):

```python
python3 -c "
import sqlite3, json
conn = sqlite3.connect('data/state/vibe_quant.db')
for sid in [STRATEGY_IDS]:
    cursor = conn.execute(
        'INSERT INTO backtest_runs (strategy_id, run_mode, symbols, timeframe, start_date, end_date, parameters, status) VALUES (?,?,?,?,?,?,?,?)',
        (sid, 'validation', json.dumps(['BTCUSDT']), '4h', 'START_DATE', 'END_DATE', json.dumps({'latency_preset': 'retail'}), 'pending'))
    print(f'Created validation run {cursor.lastrowid} for strategy {sid}')
conn.commit()
"
```

Then launch: `.venv/bin/python -m vibe_quant validation run --run-id {run_id}`

**Screening re-runs:**

```bash
.venv/bin/python -m vibe_quant screening run --run-id {run_id}
```

## SQLite Queries (state DB)

DB path: `data/state/vibe_quant.db`

**Always use `python3 -c` with `?` placeholders** — NEVER use f-strings or `.format()` for values:

```python
python3 -c "
import sqlite3, json
conn = sqlite3.connect('data/state/vibe_quant.db')
conn.row_factory = sqlite3.Row
row = conn.execute('SELECT * FROM backtest_runs WHERE id=?', (RUN_ID,)).fetchone()
if row: print(dict(row))
"
```

**Handle None before formatting** — DB values can be `None`, `str`, or numeric:

```python
# WRONG: print(f'{val:.2f}')  — crashes if val is None
# RIGHT: print(f'{val:.2f}' if val is not None else 'N/A')
```

**Key tables:**

- `backtest_runs` — all runs (discovery, screening, validation). Columns: id, strategy_id, run_mode, symbols, timeframe, start_date, end_date, parameters, status, pid, started_at, completed_at
- `strategies` — saved strategies. Columns: id, name, dsl_config, strategy_type
- `backtest_results` — stored results per run
- `background_jobs` — job tracking. Columns: id, run_id, pid, job_type, status, log_file

**There is NO `discovery_runs` table** — discovery runs are stored in `backtest_runs` with `run_mode='discovery'`.

## Workflow

### Phase 1: Journal Review & Combo Selection

1. Read `docs/discovery-journal.md` (use offset/limit — file is large, >25k tokens)
2. Identify all previously tried indicator combinations and their results
3. Select 5 NEW indicator combinations not yet tried (or re-run promising ones with better params)
4. Each combo MUST use 2+ indicators from the available pool
5. Prefer complementary signal types (trend+momentum, volume+oscillator) — avoid redundant pairs
6. Present the 5 combos to user for approval before launching

**Combo selection heuristics (from journal learnings):**

- Volume + momentum combos perform well (MFI+WILLR best so far)
- CCI+RSI is all-time champion — consider variations
- STOCH is the #2 indicator after CCI (Batch 12: Sharpe 2.55 validated standalone)
- ADX is weak for threshold-based discovery — GA consistently ignores it (Batch 10+11+12). Don't prioritize ADX combos.
- MACD has narrow threshold range — poor in 2-indicator combos
- Pure momentum pairs (RSI+ROC, MFI+RSI) produce poor, redundant signals
- GA overwhelmingly converges to single-indicator strategies (3/4 runs in Batch 12). Multi-indicator strategies degrade more in validation (35% Sharpe drop on MFI+ROC Batch 12).
- CCI's wide threshold [-200,200] dominates — GA frequently picks pure CCI
- Indicators NOT yet in genome pool are fair game — add them first, then discover

### Phase 2: Launch 5 Parallel Discoveries

**Pre-launch checklist:**

1. Verify backend is running: `lsof -i :8000`
2. Verify genome pool: `rg "INDICATOR_POOL" vibe_quant/discovery/genome.py -A 3 | head -40`

Launch all 5 via API concurrently. Use these defaults (adjust based on 20min budget):

```
Population: 12
Generations: 8
Timeframe: 4h
Symbols: ["BTCUSDT"]
Direction: null (random)
Convergence generations: 5
```

**Time budget**: 20 minutes total. If combos include slow indicators (pandas-ta), reduce pop/gens:

- All Rust-native: pop=12, gens=8
- 1 pandas-ta indicator: pop=10, gens=6
- 2 pandas-ta indicators: pop=8, gens=5

Launch command (repeat for each combo):

```bash
curl -s -X POST http://localhost:8000/api/discovery/launch \
  -H "Content-Type: application/json" \
  -d '{
    "population": 12,
    "generations": 8,
    "symbols": ["BTCUSDT"],
    "timeframes": ["4h"],
    "indicator_pool": ["INDICATOR1", "INDICATOR2"],
    "direction": null
  }'
```

Record all `run_id` values. Launch all 5 as parallel Bash tool calls.

### Phase 3: Monitor Every 3 Minutes

Poll all runs every 3 minutes. **USE `sleep 180 &&` before the poll command** in a single Bash tool call with `timeout: 300000`. This avoids spamming the user with rapid tool calls.

```bash
sleep 180 && for rid in RUN_IDS; do
  echo -n "Run $rid: "
  curl -s "http://localhost:8000/api/discovery/jobs/$rid/progress" | python3 -c "
import json,sys
d=json.load(sys.stdin)
g=d.get('progress') or {}
gen=g.get('generation','?')
mx=g.get('max_generations','?')
best=g.get('best_fitness',0)
mean=g.get('mean_fitness',0)
st=d.get('status','?')
e=g.get('eta_seconds')
eta=f'{int(e)//60}m{int(e)%60}s' if e else '-'
bf=f'{best:.4f}' if isinstance(best,(int,float)) else str(best)
mf=f'{mean:.4f}' if isinstance(mean,(int,float)) else str(mean)
print(f'{st} | Gen {gen}/{mx} | Best={bf} | Mean={mf} | ETA={eta}')
"
done
```

Report results in a markdown table. Repeat until all runs show `completed`.

### Phase 4: Collect Results

Once all complete, fetch results. **Use the correct field names** (discovery results differ from backtest results):

```bash
# Discovery results — fields: score, sharpe, max_dd, pf, trades, return_pct
for rid in 232 233 234 235 236; do
  echo "=== Run $rid ==="
  curl -s "http://localhost:8000/api/discovery/results/$rid" | python3 -c "
import json,sys
d=json.load(sys.stdin)
for i,s in enumerate(d.get('strategies',[])[:3]):
    ec=s.get('dsl',{}).get('entry_conditions',{})
    dr=','.join(ec.keys()) if ec else '?'
    print(f'  #{i}: score={s[\"score\"]:.4f} sharpe={s[\"sharpe\"]:.2f} dd={s[\"max_dd\"]*100:.1f}% trades={s[\"trades\"]} ret={s[\"return_pct\"]*100:.1f}% pf={s[\"pf\"]:.2f} dir={dr}')
"
done
```

```bash
# Backtest/validation results — fields: sharpe_ratio, total_trades, total_return, profit_factor, max_drawdown, win_rate, total_fees
for rid in 237 238 239 240; do
  echo -n "Run $rid: "
  curl -s "http://localhost:8000/api/results/runs/$rid" | python3 -c "
import json,sys
d=json.load(sys.stdin)
sr=d.get('sharpe_ratio'); tr=d.get('total_trades',0); ret=d.get('total_return',0)
pf=d.get('profit_factor',0); dd=d.get('max_drawdown',0); fees=d.get('total_fees',0)
if sr is not None: print(f'sharpe={sr:.2f} trades={tr} ret={ret*100:.1f}% pf={pf:.2f} dd={dd*100:.1f}% fees=\${fees:.2f}')
else: print(f'FAILED (sharpe=null, trades={tr})')
"
done
```

### Phase 5: Overfitting Validation (DSR)

For each winning strategy (top-1 per run), check DSR guardrails.

```bash
# Check DSR results in discovery log — use rg not grep
rg "DSR|guardrail|deflated" logs/discovery_{run_id}.log -A5
```

Strategies that FAIL DSR (p > 0.05) are eliminated. Note failures in summary.

### Phase 6: Screening Replay

**CRITICAL ORDERING — dates must be fixed BEFORE promote auto-runs screening:**

```bash
# 1. Get data range from discovery log
rg "Data range" logs/discovery_{FIRST_RUN_ID}.log | head -1

# 2. Fix dates on discovery runs
python3 -c "
import sqlite3
conn = sqlite3.connect('data/state/vibe_quant.db')
for rid in [DISCOVERY_RUN_IDS]:
    conn.execute('UPDATE backtest_runs SET start_date=?, end_date=? WHERE id=?', ('START', 'END', rid))
conn.commit()
"

# 3. Promote (auto-runs screening)
for rid in DISCOVERY_RUN_IDS; do
  echo "=== Promoting run $rid ==="
  curl -s -X POST "http://localhost:8000/api/discovery/results/$rid/promote/0?mode=screening"
  echo ""
done
# Save the returned strategy_id and screening run_id values

# 4. Fix dates on screening runs too
python3 -c "
import sqlite3
conn = sqlite3.connect('data/state/vibe_quant.db')
for rid in [SCREENING_RUN_IDS]:
    conn.execute('UPDATE backtest_runs SET start_date=?, end_date=? WHERE id=?', ('START', 'END', rid))
conn.commit()
"

# 5. Check if screening auto-ran successfully
for rid in SCREENING_RUN_IDS; do
  echo -n "Screening $rid: "
  curl -s "http://localhost:8000/api/results/runs/$rid" | python3 -c "
import json,sys; d=json.load(sys.stdin); tr=d.get('total_trades',0)
print(f'trades={tr}' + (' OK' if tr > 0 else ' NEED RE-RUN'))
"
done

# 6. Re-run any screening that got 0 trades
for rid in SCREENING_RUN_IDS_NEEDING_RERUN; do
  .venv/bin/python -m vibe_quant screening run --run-id $rid &
done
wait
```

Verify: discovery metrics == screening metrics (trade count, Sharpe, return, PF must match exactly).

### Phase 7: Validation Backtest

For strategies that passed screening, create validation runs manually:

```bash
python3 -c "
import sqlite3, json
conn = sqlite3.connect('data/state/vibe_quant.db')
for sid in [STRATEGY_IDS]:
    cursor = conn.execute(
        'INSERT INTO backtest_runs (strategy_id, run_mode, symbols, timeframe, start_date, end_date, parameters, status) VALUES (?,?,?,?,?,?,?,?)',
        (sid, 'validation', json.dumps(['BTCUSDT']), '4h', 'START_DATE', 'END_DATE', json.dumps({'latency_preset': 'retail'}), 'pending'))
    print(f'Created validation run {cursor.lastrowid} for strategy {sid}')
conn.commit()
"

# Launch validation (parallel, with log capture)
for rid in VALIDATION_RUN_IDS; do
  .venv/bin/python -m vibe_quant validation run --run-id $rid 2>&1 | tee logs/validation_${rid}.log &
done
wait
```

Expected: same trades, degraded metrics (5-20% Sharpe drop normal due to fill model + 200ms latency + fees).
Fetch results with `/api/results/runs/{validation_run_id}`.

### Phase 8: Log File Audit

For EVERY run (discovery, screening, validation), check logs for errors:

```bash
# Use rg (ripgrep) — NOT grep. Use specific run IDs, not wildcards.
for rid in RUN_ID1 RUN_ID2; do
  for prefix in discovery screening validation; do
    f="logs/${prefix}_${rid}.log"
    [ -f "$f" ] || continue
    errors=$(rg -c "ERROR|Exception|Traceback" "$f" 2>/dev/null || echo 0)
    warnings=$(rg -c "WARNING|FutureWarning" "$f" 2>/dev/null || echo 0)
    echo "$f: $errors errors, $warnings warnings"
  done
done
```

**If errors found**: Create a bead for each distinct issue:

```bash
bd create --title="[discovery-run] <error description>" --type=bug --priority=2
bd sync && git push
```

### Phase 9: Summary Table

Create a comparison table across all stages:

```
| Stage | Run A (X+Y) | Run B (X+Z) | ... |
|-------|-------------|-------------|-----|
| Discovery score | 0.52 | 0.48 | ... |
| Discovery sharpe | 1.2 | 0.9 | ... |
| Discovery trades | 95 | 120 | ... |
| DSR guardrails | PASS | FAIL | ... |
| Screening match | exact | n/a | ... |
| Validation sharpe | 1.1 | n/a | ... |
| Validation return | +8% | n/a | ... |
| Validation DD | 7% | n/a | ... |
| Validation PF | 1.3 | n/a | ... |
```

### Phase 10: Journal Entry

Append a new batch entry to `docs/discovery-journal.md` following the exact format of previous entries. Include:

1. **Date and batch number** (increment from last batch)
2. **Goal** — what combos were tested and why
3. **Bug Fixes Applied** — any fixes made during the run
4. **Configuration table** — run IDs, indicators, pop, gens, TF, time, status
5. **Full Pipeline Results table** — all stages for all runs
6. **Winning Strategies** — detailed breakdown of top performers
7. **Issues Found** — numbered list of any problems
8. **Key Findings** — what worked, what didn't, patterns observed
9. **Comparison with Previous Batches** — table comparing with best of prior batches
10. **Recommendations** — what to try next

### Phase 11: Commit & Push

```bash
git add docs/discovery-journal.md
bd sync
git commit -m "feat: Batch N discovery journal — <summary>"
bd sync
git push
```

## Key Rules

- **NEVER skip the journal review** — it prevents re-running the same combos
- **NEVER skip log audits** — errors must be caught and filed as beads
- **20 minute budget is hard** — reduce pop/gens rather than exceed it
- **Exact screening match required** — if screening doesn't match discovery, something is broken (file a bead)
- **Validation degradation is normal** — 5-20% Sharpe drop expected. Flag if >30% drop or trade count diverges >10%
- **Always push at the end** — work is not done until git push succeeds
- **Expand genome pool as needed** — if a promising indicator isn't in the pool, add it before discovery
- **Fix dates BEFORE promote** — promote auto-runs screening; empty dates = 0 trades = wasted time
- **Always use `rg` not `grep`** — faster, simpler regex, no escaping needed
- **Never use `status` as a shell variable** — it's read-only in zsh
