# E2E Pipeline Test Plan: Discovery -> Screening -> Validation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Verify full pipeline works end-to-end: discovery finds strategies, export to screening works, validation with latency/risk settings works, UI displays all results correctly, logs are clean.

**Architecture:** API-driven tests via curl + UI verification via agent-browser. Uses existing BTCUSDT 4h data (2024-01-01 to 2024-02-01) to keep runs under 2 minutes each.

**Tech Stack:** Backend (uvicorn), Frontend (Vite), agent-browser for UI screenshots, curl for API calls, sqlite3 for DB verification.

---

### Task 0: Start Services & Verify Health

**Step 1: Start backend**
```bash
cd /Users/verebelyin/projects/vibe-quant
.venv/bin/uvicorn vibe_quant.api.main:app --port 8000 &
sleep 3
curl -s http://localhost:8000/health | jq .
```
Expected: `{"status": "ok"}`

**Step 2: Start frontend**
```bash
cd /Users/verebelyin/projects/vibe-quant/frontend
pnpm dev --port 5173 &
sleep 3
```

**Step 3: Verify data coverage**
```bash
curl -s http://localhost:8000/api/data/coverage | jq '.symbols[] | {symbol, kline_1m_count, min_date, max_date}'
```
Expected: BTCUSDT with 1M+ bars, dates covering 2024-01-01 to 2024-02-01

**Step 4: Screenshot home page**
```bash
agent-browser open http://localhost:5173 && agent-browser screenshot /tmp/claude/00-home.png
```
Verify: Dashboard loads, sidebar visible

---

### Task 1: Configure Risk Management Settings

**Step 1: Create risk config via API**
```bash
curl -s -X POST http://localhost:8000/api/settings/risk \
  -H "Content-Type: application/json" \
  -d '{
    "name": "E2E Test Risk",
    "strategy_level": {
      "max_position_size_pct": 0.05,
      "max_drawdown_pct": 0.15,
      "max_daily_loss_pct": 0.03,
      "cooldown_after_halt_hours": 1
    },
    "portfolio_level": {
      "max_portfolio_drawdown_pct": 0.20,
      "max_total_exposure_pct": 0.50,
      "max_single_instrument_pct": 0.30,
      "cooldown_after_halt_hours": 24
    }
  }' | jq .
```
Expected: 201 with config_id

**Step 2: Create sizing config**
```bash
curl -s -X POST http://localhost:8000/api/settings/sizing \
  -H "Content-Type: application/json" \
  -d '{
    "name": "E2E Test Sizing",
    "method": "fixed_fractional",
    "config": {"risk_fraction": 0.02, "max_leverage": 3.0}
  }' | jq .
```
Expected: 201 with config_id

**Step 3: Verify settings in UI**
```bash
agent-browser open http://localhost:5173/settings && agent-browser screenshot /tmp/claude/01-settings.png
```
Verify: Risk and sizing configs visible in settings page

**Step 4: Check latency presets**
```bash
curl -s http://localhost:8000/api/settings/latency-presets | jq .
```
Expected: 4 presets (co_located 1ms, domestic 20ms, international 100ms, retail 200ms)

---

### Task 2: Run Discovery (Small, Fast)

**Step 1: Launch discovery via API**
```bash
curl -s -X POST http://localhost:8000/api/discovery/launch \
  -H "Content-Type: application/json" \
  -d '{
    "population": 6,
    "generations": 3,
    "mutation_rate": 0.15,
    "crossover_rate": 0.7,
    "elite_count": 1,
    "tournament_size": 2,
    "convergence_generations": 10,
    "symbols": ["BTCUSDT"],
    "timeframes": ["4h"],
    "start_date": "2024-01-01",
    "end_date": "2024-02-01"
  }' | jq .
```
Expected: 201 with run_id. Record the run_id as DISCOVERY_RUN_ID.

**Step 2: Screenshot discovery page immediately**
```bash
agent-browser open http://localhost:5173/discovery && agent-browser screenshot /tmp/claude/02-discovery-running.png
```
Verify: Discovery job appears with "running" status, progress bar visible

**Step 3: Poll for completion (max 3 min)**
```bash
DISCOVERY_RUN_ID=<from step 1>
for i in $(seq 1 36); do
  STATUS=$(curl -s http://localhost:8000/api/discovery/jobs | jq -r ".[0].status")
  echo "[$i] Status: $STATUS"
  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then break; fi
  sleep 5
done
```

**Step 4: Check discovery results**
```bash
curl -s "http://localhost:8000/api/discovery/results/$DISCOVERY_RUN_ID" | jq '{strategies_count: (.strategies | length), top_fitness: .strategies[0].fitness, top_sharpe: .strategies[0].sharpe_ratio, top_trades: .strategies[0].total_trades}'
```
Expected: At least 1 strategy, non-null fitness

**Step 5: Check discovery log for errors/warnings**
```bash
grep -iE "error|warning|exception|traceback" logs/discovery_${DISCOVERY_RUN_ID}.log | head -20
```
Expected: No errors. Warnings about "No trades" or "Low trade count" acceptable.

**Step 6: Screenshot discovery results in UI**
```bash
agent-browser open http://localhost:5173/discovery && sleep 2 && agent-browser screenshot /tmp/claude/03-discovery-results.png
```
Verify: Results table shows strategies with fitness, Sharpe, return, trades columns

---

### Task 3: Export Best Strategy & Run Screening

**Step 1: Export best strategy from discovery**
```bash
curl -s -X POST "http://localhost:8000/api/discovery/results/$DISCOVERY_RUN_ID/export/0" | jq .
```
Expected: strategy_id returned. Record as STRATEGY_ID.

**Step 2: Verify strategy in strategies list**
```bash
curl -s http://localhost:8000/api/strategies | jq '.[] | select(.name | test("discovered")) | {id, name, symbols}'
```

**Step 3: Screenshot strategy in UI**
```bash
agent-browser open http://localhost:5173/strategies && sleep 1 && agent-browser screenshot /tmp/claude/04-strategies.png
```
Verify: Discovered strategy visible in list

**Step 4: Launch screening for exported strategy**
```bash
curl -s -X POST http://localhost:8000/api/backtest/screening \
  -H "Content-Type: application/json" \
  -d "{
    \"strategy_id\": $STRATEGY_ID,
    \"symbols\": [\"BTCUSDT\"],
    \"timeframe\": \"4h\",
    \"start_date\": \"2024-01-01\",
    \"end_date\": \"2024-02-01\",
    \"parameters\": {}
  }" | jq .
```
Expected: 201 with run_id. Record as SCREENING_RUN_ID.

**Step 5: Poll for screening completion (max 2 min)**
```bash
for i in $(seq 1 24); do
  STATUS=$(curl -s "http://localhost:8000/api/backtest/jobs/$SCREENING_RUN_ID" | jq -r ".status")
  echo "[$i] Status: $STATUS"
  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then break; fi
  sleep 5
done
```

**Step 6: Check screening results**
```bash
curl -s "http://localhost:8000/api/results/runs/$SCREENING_RUN_ID" | jq '{status, sharpe_ratio, total_return_pct, max_drawdown_pct, total_trades, profit_factor}'
```
Expected: Non-null metrics, total_trades > 0

**Step 7: Check screening log for errors**
```bash
grep -iE "error|warning|exception|traceback" logs/screening_${SCREENING_RUN_ID}.log | head -20
```

**Step 8: Screenshot backtest results in UI**
```bash
agent-browser open http://localhost:5173/results && sleep 2 && agent-browser screenshot /tmp/claude/05-screening-results.png
```
Verify: Metrics panel, equity curve, trade log visible

---

### Task 4: Run Validation with Latency

**Step 1: Launch validation with retail latency**
```bash
curl -s -X POST http://localhost:8000/api/backtest/validation \
  -H "Content-Type: application/json" \
  -d "{
    \"strategy_id\": $STRATEGY_ID,
    \"symbols\": [\"BTCUSDT\"],
    \"timeframe\": \"4h\",
    \"start_date\": \"2024-01-01\",
    \"end_date\": \"2024-02-01\",
    \"parameters\": {},
    \"latency_preset\": \"retail\"
  }" | jq .
```
Expected: 201 with run_id. Record as VALIDATION_RUN_ID.

**Step 2: Poll for completion (max 2 min)**
```bash
for i in $(seq 1 24); do
  STATUS=$(curl -s "http://localhost:8000/api/backtest/jobs/$VALIDATION_RUN_ID" | jq -r ".status")
  echo "[$i] Status: $STATUS"
  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then break; fi
  sleep 5
done
```

**Step 3: Check validation results and compare with screening**
```bash
echo "=== SCREENING ==="
curl -s "http://localhost:8000/api/results/runs/$SCREENING_RUN_ID" | jq '{sharpe_ratio, total_return_pct, max_drawdown_pct, total_trades}'
echo "=== VALIDATION (retail latency) ==="
curl -s "http://localhost:8000/api/results/runs/$VALIDATION_RUN_ID" | jq '{sharpe_ratio, total_return_pct, max_drawdown_pct, total_trades}'
```
Expected: Validation has same or fewer trades, possibly worse metrics (latency + realistic fills)

**Step 4: Check validation log for latency model**
```bash
grep -iE "latency|fill.model|error|warning|exception" logs/validation_${VALIDATION_RUN_ID}.log | head -20
```
Expected: Latency model mentioned in startup, no errors

**Step 5: Screenshot validation results in UI**
```bash
agent-browser open "http://localhost:5173/results" && sleep 2 && agent-browser screenshot /tmp/claude/06-validation-results.png
```

---

### Task 5: Verify Risk Management Config Integration

**Step 1: Launch validation with risk config**
```bash
RISK_CONFIG_ID=<from Task 1>
SIZING_CONFIG_ID=<from Task 1>
curl -s -X POST http://localhost:8000/api/backtest/validation \
  -H "Content-Type: application/json" \
  -d "{
    \"strategy_id\": $STRATEGY_ID,
    \"symbols\": [\"BTCUSDT\"],
    \"timeframe\": \"4h\",
    \"start_date\": \"2024-01-01\",
    \"end_date\": \"2024-02-01\",
    \"parameters\": {},
    \"latency_preset\": \"domestic\",
    \"risk_config_id\": $RISK_CONFIG_ID,
    \"sizing_config_id\": $SIZING_CONFIG_ID
  }" | jq .
```
Expected: 201 with run_id. Record as RISK_RUN_ID.

**Step 2: Poll + check results**
Same polling loop. Compare metrics with previous runs.

**Step 3: Check risk config was applied in log**
```bash
grep -iE "risk|sizing|position.size|drawdown|halt" logs/validation_${RISK_RUN_ID}.log | head -20
```

---

### Task 6: UI Verification Tour

**Step 1: Data Management page**
```bash
agent-browser open http://localhost:5173/data && agent-browser screenshot /tmp/claude/07-data.png
```
Verify: Coverage table shows BTCUSDT with dates/bar counts

**Step 2: Discovery page with completed run**
```bash
agent-browser open http://localhost:5173/discovery && agent-browser screenshot /tmp/claude/08-discovery-final.png
```
Verify: Completed discovery run, strategies table, export buttons

**Step 3: Backtest Launch page**
```bash
agent-browser open http://localhost:5173/backtest && agent-browser screenshot /tmp/claude/09-backtest.png
```
Verify: Strategy selector, mode toggle, symbol/timeframe selectors

**Step 4: Results page with multiple runs**
```bash
agent-browser open http://localhost:5173/results && agent-browser screenshot /tmp/claude/10-results.png
```
Verify: Multiple runs listed (screening + 2 validation), metrics visible

**Step 5: Settings page**
```bash
agent-browser open http://localhost:5173/settings && agent-browser screenshot /tmp/claude/11-settings-final.png
```
Verify: Risk config and sizing config from Task 1 visible

---

### Task 7: Log Audit & Cleanup

**Step 1: Comprehensive log check**
```bash
for f in logs/discovery_*.log logs/screening_*.log logs/validation_*.log; do
  echo "=== $f ==="
  grep -ciE "error|exception|traceback" "$f" || echo "0 errors"
  grep -iE "error|exception|traceback" "$f" | head -5
  echo ""
done
```
Expected: Zero unexpected errors across all logs

**Step 2: DB integrity check**
```bash
sqlite3 data/state/vibe_quant.db "
  SELECT run_mode, status, COUNT(*) FROM backtest_runs GROUP BY run_mode, status;
  SELECT COUNT(*) as total_trades FROM trades;
  SELECT COUNT(*) as total_strategies FROM strategies;
"
```
Expected: discovery/screening/validation runs all "completed", trades > 0

**Step 3: Stop services**
```bash
kill %1 %2  # backend + frontend
```
