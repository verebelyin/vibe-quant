# CLAUDE.md

Algorithmic trading engine for crypto perpetual futures using NautilusTrader (Rust core) with two-tier backtesting (screening + validation), strategy DSL, overfitting prevention, and paper/live execution.

## This file

The rule of this file is to describe common mistakes and confusion points that agents might encounter as they work in this project. If you ever encounter something in this project that surprises you, or you failed to do after multiple attempts, please alert the developer working with you and indicate and describe that in the @AGENTS.md file to help prevent future agents from having the same issue.

## Quick Reference

- **Package manager:** `uv` (not pip/poetry)
- **Python:** 3.13
- **Install:** `uv pip install -e .`
- **Tests:** `pytest` (target 80% coverage on core modules)
- **Lint:** `ruff check` — **baseline is ZERO** (since 2026-07-09). Any error you see is from current work; fix it, don't assume pre-existing.
- **Type check:** `mypy` — also zero-baseline. Same rule.
- **Backend:** `.venv/bin/uvicorn "vibe_quant.api.app:create_app" --factory --port 8000`
- **Frontend:** `cd frontend && pnpm dev` (Vite on port 5173 — **but see Dev Server Gotchas below: 5173 may belong to a different project**)
- **Frontend build:** `cd frontend && pnpm build`
- **Extraction worker:** `.venv/bin/vibe-quant extraction-worker` (drains `/api/research/items/{id}/extract` queue; run alongside backend so manual re-extractions actually progress)

## Dev Server Gotchas (stepped into repeatedly — read before starting servers)

1. **Port 5173 may be a DIFFERENT project.** The user runs other Vite apps; vibe-quant's
   frontend can land on 5174/5175 when 5173 is taken. **Always screenshot or check the
   page title before UI testing** — one session spent time driving a Minecraft clone.
2. **Never `lsof -ti :8000 | xargs kill`.** Vite holds proxy connections to :8000, so
   this kills the frontend too. Use `pgrep -f "uvicorn vibe_quant" | xargs kill`.
3. **The backend caches code at startup.** Plugin registrations and endpoint changes
   only show in API responses after a backend restart. But discovery/validation/
   screening run as **subprocesses** — they pick up new code from disk without a
   restart. Know which one you're testing.
4. **agent-browser clicks silently no-op on elements inside scrolled-out overflow
   containers.** `scrollIntoView({block:'center'})` via `eval` first, then click;
   re-snapshot after every DOM change (refs go stale).
5. Long test suites / servers: use `run_in_background`, never `sleep`-polling.

## UI Testing (agent-browser)

Start backend + frontend then test with `agent-browser`. **Always use `dangerouslyDisableSandbox: true`** for agent-browser commands (it needs `~/.agent-browser` socket dir).

**Quick start:**

```bash
# Start backend + frontend (background)
.venv/bin/uvicorn "vibe_quant.api.app:create_app" --factory --port 8000 &
cd frontend && pnpm dev --port 5173 &

# Open and take initial screenshot
agent-browser open http://localhost:5173 && agent-browser screenshot /tmp/claude/page.png
```

**Chain commands with `&&`** to reduce round-trips:

```bash
# Navigate + snapshot in one call
agent-browser open http://localhost:5173 && agent-browser snapshot -i

# Click + wait + screenshot in one call
agent-browser click @e5 && agent-browser wait 2000 && agent-browser screenshot /tmp/claude/result.png

# Fill form + click save in one call
agent-browser fill @e1 "test_strategy" && agent-browser fill @e2 "description" && agent-browser click @e3
```

**Key workflow:**

1. `agent-browser open <url>` — navigate
2. `agent-browser snapshot -i` — get interactive elements with refs (`@e1`, `@e2`...)
3. `agent-browser click @e1` / `agent-browser fill @e2 "text"` — interact using refs
4. `agent-browser screenshot /tmp/claude/name.png` — capture state
5. Re-snapshot after any navigation or DOM change (refs get invalidated)

**Navigation pages** (sidebar links): Strategy Management, Discovery, Backtest Launch, Results Analysis, Paper Trading, Data Management, Settings

**Test flow:** Data Management (download data) → Strategy Management (create strategy) → Backtest Launch (run screening) → Results Analysis (verify results)

## Shell Preferences

- **Always use `rg` (ripgrep) instead of `grep`** — faster, simpler regex syntax (no escaping `|`), better defaults. Use `rg` in Bash tool calls, skills, and scripts. This applies to ALL search operations in the terminal.
  ```bash
  rg "ERROR|Exception" logs/          # NOT: grep "ERROR\|Exception" logs/
  rg -c "pattern" file                # count matches
  rg "pattern" -A5 file               # context after
  rg "pattern" -l                     # list files only
  rg "pattern" -B2 -A2 file           # context before and after
  ```
- **`status` is read-only in zsh** — never use it as a variable name in shell scripts. Use `st`, `stat`, or `run_status` instead.
- **Don't use `sleep N` in Bash tool calls for polling** — make separate tool calls when ready instead. `sleep` blocks the tool and wastes time.
- **Check ports before starting servers**: `lsof -i :8000` before launching uvicorn. Avoids "address already in use" errors.

## SQLite Queries (state DB)

DB path: `data/state/vibe_quant.db`. Always use WAL mode.

**Common mistakes to avoid:**
1. **Don't use `.format()` or f-strings with values** — use `?` placeholders for ALL query values
2. **Values can be `None`/`str`/numeric** — always handle `None` before formatting with `:.2f`
3. **No `discovery_runs` table** — discovery runs are in `backtest_runs` with `run_mode='discovery'`
4. **`row_factory = sqlite3.Row`** enables dict-style access
5. **Always `conn.commit()` after INSERT/UPDATE** — SQLite doesn't auto-commit

**Canonical pattern (query + mutate):**
```python
python3 -c "
import sqlite3
conn = sqlite3.connect('data/state/vibe_quant.db')
conn.row_factory = sqlite3.Row
for r in conn.execute('SELECT * FROM backtest_runs WHERE run_mode=? ORDER BY id DESC LIMIT 5', ('discovery',)):
    print(dict(r))
conn.execute('UPDATE backtest_runs SET status=? WHERE id=?', ('failed', 999))
conn.commit()  # DON'T FORGET
"
```

**Key tables:** `backtest_runs` (all run modes), `strategies`, `backtest_results` (validation metrics + discovery notes JSON), `sweep_results` (screening/replay metrics), `background_jobs`, `trades`, `research_extractions`

## Architecture

```
Strategy DSL (YAML) → Screening (NT simplified, parallel) → Overfitting Filters → Validation (NT full fidelity) → Paper → Live
```

Single engine (NautilusTrader) with two modes:

- **Screening mode**: simplified fills, no latency, multiprocessing parallelism -- still models leverage/funding/liquidation
- **Validation mode**: custom FillModel, LatencyModel (co-located 1ms → retail 200ms), full cost modeling

## Key Specifications

| Detail                   | Reference                                                                                                        |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| Full implementation spec | [`SPEC.md`](SPEC.md) -- **the authoritative source** for architecture, DSL, pipelines, data, schemas, and phases |
| Sections 1-5             | Architecture, tech stack, decisions, data layout, strategy DSL                                                   |
| Sections 6-7             | Screening pipeline, validation backtesting                                                                       |
| Sections 8-13            | Overfitting, risk, dashboard, paper trading, observability, testing                                              |
| Phases 1-8               | Implementation roadmap with deliverables and acceptance criteria                                                 |

## Conventions

See [docs/claude/conventions.md](docs/claude/conventions.md) for full details. Critical rules:

- **License:** MIT project. NautilusTrader (LGPL-3.0) used as unmodified library dependency -- this is acceptable. Never modify NT source. Avoid AGPL dependencies.
- **Secrets:** API keys in env vars only, never in code.
- **SQLite:** Always enable WAL mode (`PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;`) on every connection.
- **Indicators:** Prefer NautilusTrader built-in (Rust) indicators. Fall back to `pandas-ta-classic` for exotic ones. Never use the original `pandas-ta` (compromised maintainership). Custom indicators: drop a `.py` file in `vibe_quant/dsl/plugins/` — see [`plugins/README.md`](vibe_quant/dsl/plugins/README.md) for the full extension API.
- **Data:** Raw downloaded data archived in SQLite before processing to ParquetDataCatalog. Catalog is rebuildable from archive.

## Issue Tracking (Beads)

**IMPORTANT: Use `bd` (beads) for ALL task/issue tracking. NEVER use TodoWrite, TaskCreate, or markdown files for tasks.** Full reference (install, Dolt architecture, memory taxonomy, performance): [docs/claude/beads.md](docs/claude/beads.md).

```bash
bd ready                          # Find available work
bd show <id>                      # View issue details
bd update <id> --claim            # Claim work
bd close <id>                     # Complete work (multiple ids OK)
bd create --title="..." --description="..." --type=bug --priority=2   # priority 0-4
bd remember "fact" --key category:specific-item   # Persistent project memory
bd recall <key> / bd memories <keyword>           # Read memories
```

- **Workflow:** `bd ready` → `bd update <id> --claim` → implement → `bd close <id>` → `git push`.
- **NEVER use `bd edit`** — it opens `$EDITOR` which blocks agents.
- `bd prime` (auto-run by hooks) injects all memories each session. When you learn a surprising non-obvious project fact, `bd remember` it (key taxonomy in [docs/claude/beads.md](docs/claude/beads.md)). If CLAUDE.md contradicts a memory, CLAUDE.md wins — update or `bd forget` the memory.

### Session Completion

**YOU MUST push before calling work done.** File beads for follow-ups → run quality gates → close/update beads → `git pull --rebase && git push` → `git status` must show "up to date with origin". Optional: `bd dolt push` backs memories up off-machine.

## Directory Structure

```
vibe_quant/          # Backend Python package
  api/               # FastAPI: app.py factory, routers/, schemas/, sse/, ws/
  data/              # Downloader, SQLite archive, ParquetDataCatalog
  db/                # SQLite connection (WAL), schema.py, state_manager
  discovery/         # GA: genome, operators, fitness, pipeline, guardrails
  dsl/               # Strategy DSL: parser, compiler, schema, indicators; plugins/ (drop-in indicators)
  overfitting/       # WFA, purged k-fold, DSR, bootstrap CI
  paper/             # Paper trading: NT TradingNode, persistence, CLI
  screening/         # nt_runner.py (shared by discovery/screening/WFA), grid sweep
  validation/        # runner.py, venue/fill/latency models, extraction, consistency.py
  nt_compat.py       # NT compatibility helpers (log-guard retention)
frontend/src/        # React SPA (Vite + Tailwind 4 + shadcn + TanStack Router)
  api/generated/     # orval-generated hooks/models — DO NOT EDIT; regenerate: dump openapi.json + pnpm generate-api
  components/        # by domain: backtest/ charts/ data/ discovery/ paper/ results/ settings/ strategies/ ui/
  routes/            # file-based routes (strategies, backtest, discovery, results, paper-trading, data, settings)
tests/               # pytest; fixtures/known_results = golden metrics
data/                # Runtime (gitignored): archive/, catalog/, state/vibe_quant.db
logs/                # Per-run logs: {discovery|validation|screening}_<run_id>_*.log + events/*.jsonl
docs/                # claude/conventions.md (coding rules), claude/beads.md, discovery-journal.md, plans/
SPEC.md              # Authoritative implementation spec
```

**Key paths:** DB schema `vibe_quant/db/schema.py` · DSL types `vibe_quant/dsl/schema.py` (frontend mirror `frontend/src/components/strategies/editor/types.ts`) · theme `frontend/src/index.css`

## Discovery Pipeline Notes

- **Research diary:** `docs/discovery-journal.md` — experiment log with GA configs, metrics, and findings
- Discovery and screening use **identical** code path (`NTScreeningRunner` → `StrategyCompiler`). Results match exactly *within one run* (champion → replay).
- Validation uses custom fill model + latency → fewer trades and worse metrics (expected)
- **Bug fix `2944ad3`:** `pos.entry→pos.side` enum mismatch caused 155:1 trade ratio. All runs before this fix are invalid.
- **Semantics break `11c5f00` (2026-07-09):** screening now feeds ONLY the strategy timeframe (an NT data-loading bug previously fed ALL timeframes incl. 1m, giving screening accidental intrabar fills). Discovery scores ≤ run 854 are not comparable with newer runs. **Validation is unchanged** — it loads 1m detail explicitly and reproduces historical results bit-for-bit.
- **Compiler version hash:** stored in discovery notes for staleness detection. Changes when the indicator registry changes — check `bd recall discovery:compiler-hash` for the current value; recompute with `compiler_version_hash()`.
- Champion rankings live in `bd recall discovery:champions` (journal has full history). Batch-13 STOCH+CCI headline numbers are historical only (pre-`11c5f00`).
- **1m data is slow:** Rust-native indicators (SMA/EMA/CCI/STOCH/ATR) ~10x faster than pandas-path ones (ADX/MACD/BBANDS/KAMA). Budget accordingly.
- **Fitness function:** 35% Sharpe + 25% (1-MaxDD) + 20% PF + 20% Return. Hard gate: 0 if <50 trades.
- **`eval_windows` (default 3) stores worst-of-N sub-window fitness** — a full-window replay legitimately shows different Sharpe/return (`ReplayResponse.metrics_note` explains this). Not a bug.
- **Single-seed discovery is unseeded** — identical configs produce different populations across runs. Never compare two discovery runs to validate a code change (see Verification Rules below).
- **Bootstrap-CI gate keeps being vindicated:** every champion forced past it with `no_bootstrap_ci=true` and then validated has collapsed (Batch 41: 5.40→−2.78; Batch 43 RAMS: 0.59→−0.36). The validation runner auto-flags collapses (`validation/consistency.py`); treat a flagged strategy as overfit, not as a validation bug.
- 4h/1d discovery uses bootstrap floor 0.0 by default (1.0 is structurally unpassable at ~50-180 trades/yr); 1m uses 0.5.

## NautilusTrader Gotchas (each of these cost real debugging time)

- **IMPORTANT: `BacktestDataConfig.data_cls` MUST be the CLASS object** (`from nautilus_trader.model.data import Bar`), never the import string. `config.query` compares `data_cls is Bar`; a string silently disables `bar_types`/`bar_spec` narrowing and loads the ENTIRE catalog (~300× data, and fills change because the venue processes stray finer-granularity bars). Upstream issue draft: `docs/nt-upstream-issue-data-cls.md`.
- **IMPORTANT: creating a `BacktestEngine`/`BacktestNode` after disposing a previous one in the same process hard-aborts** (Rust logger can only init once; the process just dies with no Python traceback). YOU MUST call `vibe_quant.nt_compat.retain_log_guard(engine)` before dispose in any new engine-lifecycle code; both runners and `test_fill_timing.py` already do.
- **NT 1.226+ config decoding rejects unknown fields** (fast-fail). Forward only params the generated `StrategyConfig` declares (`__struct_fields__` filter in both runners). Run-level knobs like `initial_balance`/`leverage` belong on the venue, not the strategy config.
- **`node.build()` swallows engine-build exceptions** — `get_engine()` returns `None` and you see only "engine not found". Validation sets `BacktestRunConfig(raise_exception=True)`; keep it that way, and set it when writing new runner code.
- **NT `BacktestResult.elapsed_time` is the simulated window in seconds**, not wall time. `Iterations` ≈ bars processed — if it's far above the expected bar count, you have a data-loading bug (see Performance Playbook).
- ADX stays on the pandas path deliberately (NT has no true ADX — `DirectionalMovement.value` is always 0). Don't "optimize" it to `nt_class` without checking values.

## Performance Profiling Playbook (how the 300× and 4.4× wins were found)

1. **Read the engine's own numbers first.** Run one eval with
   `VIBE_QUANT_NT_LOG_LEVEL_SCREENING=INFO` (validation logs INFO by default) and read:
   `Added N ... Bar elements` (data volume ground truth), `Read N events from parquet in Xs`,
   `Engine load took Xs`, `Iterations: N`. Iterations ≫ expected bars ⇒ data bug, not slow compute.
2. **Time phases coarsely before profiling**: compile (`_ensure_compiled`), data load, engine
   run. Most "slow engine" reports are actually one phase.
3. **cProfile only on SHORT windows** (1–2 months). Full-window + profiler overhead exceeds
   command timeouts. `py-spy` needs root on macOS. Sort by `tottime` to find pandas
   internals; sort by `cumulative` to attribute them to indicator functions.
4. **Discovery `s/chromosome` is wall time across ~8 workers** — multiply by worker count for
   per-eval CPU. A "1.6s/chromosome" gen can hide 100s-CPU genomes.
5. **pandas-ta functions with per-element `.iloc` loops are pathological** (KAMA was 62% of
   eval time at ~220 pandas calls/bar). Fix = exact numpy port of the recurrence keeping the
   library's vectorized prep, guarded by **zero-tolerance equality tests**
   (`tests/unit/test_plugins/test_kama_exactness.py` is the template). `ta.adx` is the next
   known target (bead vibe-quant-3f0er).
6. **Every perf change ships with an exactness proof** — see Verification Rules.

## Verification Rules (what counts as "results are the same")

Valid proofs that a change preserved correctness:
- **Fixed-strategy eval before/after**: one `NTScreeningRunner` call on a saved strategy must
  return bit-identical metrics (e.g. strategy 239: sharpe `1.3420101065837169`, 68 trades).
- **Validation repeatability**: the same validation run twice is bit-identical (proven:
  runs 868 == 870 to full float precision). Any drift = regression.
- **Within-run replay**: discovery champion → `/replay` matches exactly when the run used
  `eval_windows=1`.
- Zero-tolerance unit tests against the reference implementation for ported math.

INVALID proofs (these wasted time):
- Comparing two discovery runs — single-seed runs are unseeded and nondeterministic
  (vibe-quant-8t7nv), and ANY code change shifts the RNG draw sequence.
- Comparing an `eval_windows>1` champion's stored fitness to a full-window replay
  (worst-of-N vs full window — differs by design).
- Comparing screening metrics across the `11c5f00` semantics break.

## Historical Documentation

`docs/opus-prd.md`, `docs/opus-spec.md`, `docs/opus-research.md`, `docs/gpt-research.md`, and `docs/crypto-trading-bot-specification.md` predate SPEC.md and describe **abandoned architectures** (FreqTrade, VectorBT, PostgreSQL, Redis). Never follow them; when any `docs/*.md` contradicts SPEC.md, **SPEC.md wins**.
