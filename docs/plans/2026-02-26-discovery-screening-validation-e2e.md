# Discovery → Screening → Validation E2E Verification

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Verify that discovery, screening, and validation produce consistent trade counts/metrics after the `pos.entry→pos.side` bug fix.

**Architecture:** Launch small discovery via API (pop=4, gen=2, ~8 evaluations), promote best genome to screening, then validation. Compare metrics at each step. Use agent-browser to verify UI reflects results. Run 2 parallel pipelines with different timeframes.

**Tech Stack:** FastAPI backend, agent-browser for UI verification, curl for API calls, sqlite3 for result queries.

---

### Task 1: Start backend + frontend

**Step 1: Kill any existing processes**

```bash
pkill -f "uvicorn vibe_quant" || true
pkill -f "pnpm dev" || true
```

**Step 2: Start backend**

```bash
cd /Users/verebelyin/projects/vibe-quant
.venv/bin/uvicorn vibe_quant.api.main:app --port 8000 &
```

**Step 3: Start frontend**

```bash
cd /Users/verebelyin/projects/vibe-quant/frontend && pnpm dev --port 5173 &
```

**Step 4: Verify both are running**

```bash
curl -s http://localhost:8000/api/backtest/jobs | head -1
curl -s http://localhost:5173 | head -1
```

---

### Task 2: Launch 2 discovery runs via API

Fast configs for ~2 min total:
- **Run A:** BTCUSDT/1h, 2025-06-01 to 2025-12-31, pop=4, gen=2
- **Run B:** BTCUSDT/4h, 2025-01-01 to 2025-12-31, pop=4, gen=2

**Step 1: Launch both in parallel**

```bash
# Run A: 1h timeframe
curl -s -X POST http://localhost:8000/api/discovery/launch \
  -H "Content-Type: application/json" \
  -d '{"population":4,"generations":2,"mutation_rate":0.15,"crossover_rate":0.8,"elite_count":1,"tournament_size":2,"convergence_generations":2,"symbols":["BTCUSDT"],"timeframes":["1h"],"start_date":"2025-06-01","end_date":"2025-12-31"}' | python3 -m json.tool

# Run B: 4h timeframe
curl -s -X POST http://localhost:8000/api/discovery/launch \
  -H "Content-Type: application/json" \
  -d '{"population":4,"generations":2,"mutation_rate":0.15,"crossover_rate":0.8,"elite_count":1,"tournament_size":2,"convergence_generations":2,"symbols":["BTCUSDT"],"timeframes":["4h"],"start_date":"2025-01-01","end_date":"2025-12-31"}' | python3 -m json.tool
```

Record `run_id` from each response.

**Step 2: Poll until complete (timeout 3 min)**

```bash
# Poll both jobs
curl -s http://localhost:8000/api/discovery/jobs | python3 -m json.tool
```

Repeat every 15s until both show `"status": "completed"` or 3 min elapsed.

**Step 3: Get discovery results**

```bash
curl -s http://localhost:8000/api/discovery/results/{RUN_A_ID} | python3 -m json.tool
curl -s http://localhost:8000/api/discovery/results/{RUN_B_ID} | python3 -m json.tool
```

Record: trades, sharpe, return for the best genome (index 0) from each.

---

### Task 3: Promote genomes → screening

**Step 1: Promote best genome from each run**

```bash
# Promote Run A genome[0] as screening
curl -s -X POST "http://localhost:8000/api/discovery/results/{RUN_A_ID}/promote/0?mode=screening" | python3 -m json.tool

# Promote Run B genome[0] as screening
curl -s -X POST "http://localhost:8000/api/discovery/results/{RUN_B_ID}/promote/0?mode=screening" | python3 -m json.tool
```

Record `strategy_id` and screening `run_id` from each.

**Step 2: Poll until screening completes (timeout 1 min)**

```bash
curl -s http://localhost:8000/api/backtest/jobs | python3 -m json.tool
```

**Step 3: Query screening results from DB**

```bash
sqlite3 data/state/vibe_quant.db "SELECT run_id, total_trades, total_return, sharpe_ratio, profit_factor, notes FROM sweep_results WHERE run_id IN ({SCREENING_A_ID},{SCREENING_B_ID});"
```

Compare with discovery metrics.

---

### Task 4: Launch validation for both strategies

**Step 1: Launch validation runs**

```bash
# Validation for strategy A
curl -s -X POST http://localhost:8000/api/backtest/validation \
  -H "Content-Type: application/json" \
  -d '{"strategy_id":{STRAT_A_ID},"symbols":["BTCUSDT"],"timeframe":"1h","start_date":"2025-06-01","end_date":"2025-12-31","parameters":{}}' | python3 -m json.tool

# Validation for strategy B
curl -s -X POST http://localhost:8000/api/backtest/validation \
  -H "Content-Type: application/json" \
  -d '{"strategy_id":{STRAT_B_ID},"symbols":["BTCUSDT"],"timeframe":"4h","start_date":"2025-01-01","end_date":"2025-12-31","parameters":{}}' | python3 -m json.tool
```

**Step 2: Poll until complete (timeout 2 min)**

**Step 3: Query validation results**

```bash
sqlite3 data/state/vibe_quant.db "SELECT run_id, total_trades, total_return, sharpe_ratio, profit_factor FROM sweep_results WHERE run_id IN ({VAL_A_ID},{VAL_B_ID});"
```

---

### Task 5: Compare all results + UI verification

**Step 1: Build comparison table**

For each pipeline (A and B), compare:

| Metric | Discovery | Screening | Validation |
|--------|-----------|-----------|------------|
| Trades | | | |
| Return | | | |
| Sharpe | | | |
| PF | | | |

Discovery and screening should match closely (same engine, same fill model).
Validation may differ due to custom fill model + latency.

**Step 2: Agent-browser UI verification**

```bash
agent-browser open http://localhost:5173/discovery && agent-browser screenshot /tmp/claude/discovery.png
agent-browser open http://localhost:5173/results && agent-browser screenshot /tmp/claude/results.png
```

Verify:
- Discovery page shows the completed runs with genome results
- Results page shows screening and validation runs
- Trade counts in UI match DB queries

---

### Success Criteria

1. **Discovery → Screening trade count ratio < 1.5x** (was 155:1 before fix)
2. **Discovery → Screening Sharpe within ±0.5**
3. **Validation produces trades** (not 0 or 1)
4. **UI displays results correctly**

### Timing Budget

| Step | Expected |
|------|----------|
| Discovery (2 parallel, pop=4 gen=2) | ~2 min |
| Screening (2 parallel) | ~15s |
| Validation (2 parallel) | ~30s |
| UI checks | ~30s |
| **Total** | **~3-4 min** |
