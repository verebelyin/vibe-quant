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

**Pattern for safe queries:**
```python
python3 -c "
import sqlite3, json
conn = sqlite3.connect('data/state/vibe_quant.db')
conn.row_factory = sqlite3.Row
rows = conn.execute('SELECT * FROM backtest_runs WHERE run_mode=? ORDER BY id DESC LIMIT 5', ('discovery',)).fetchall()
for r in rows: print(dict(r))
"
```

**Pattern for safe formatting (handle None):**
```python
python3 -c "
import sqlite3
conn = sqlite3.connect('data/state/vibe_quant.db')
conn.row_factory = sqlite3.Row
row = conn.execute('SELECT * FROM backtest_runs WHERE id=?', (RUN_ID,)).fetchone()
if row:
    d = dict(row)
    # WRONG: print(f'{d[\"start_date\"]:.2f}')  — crashes if None or str
    # RIGHT: guard with 'if val is not None'
    for k, v in d.items():
        print(f'  {k}: {v}')
"
```

**Pattern for batch updates:**
```python
python3 -c "
import sqlite3
conn = sqlite3.connect('data/state/vibe_quant.db')
for rid in [237, 238, 239]:
    conn.execute('UPDATE backtest_runs SET start_date=?, end_date=? WHERE id=?', ('2025-03-07', '2026-03-07', rid))
conn.commit()  # DON'T FORGET THIS
print('Updated')
"
```

**Key tables:** `backtest_runs` (all run types), `strategies`, `backtest_results`, `background_jobs`, `trades`, `sweep_results`

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

**CRITICAL:** Use `bd` (beads) for ALL task/issue tracking. **NEVER use TodoWrite, TaskCreate, or markdown files for task tracking.** Beads is the single source of truth.

**Current version:** `bd 1.0.0` — embedded Dolt backend (no daemon, no SQLite, no JSONL sync layer).

### Installing `bd` (v1.0.0+)

**macOS (this machine):** Homebrew or direct binary download.

```bash
# Option A — Homebrew
brew install beads

# Option B — Direct binary (darwin_arm64 example)
VERSION=1.0.0
curl -sL -o /tmp/beads.tar.gz \
  "https://github.com/gastownhall/beads/releases/download/v${VERSION}/beads_${VERSION}_darwin_arm64.tar.gz"
tar xzf /tmp/beads.tar.gz -C /tmp
cp /tmp/beads_${VERSION}_darwin_arm64/bd ~/.local/bin/bd
chmod +x ~/.local/bin/bd
```

**Init in a new repo:**
```bash
bd init --non-interactive --role maintainer    # Fresh init
bd init --from-jsonl --non-interactive --role maintainer   # Import from existing .beads/issues.jsonl
chmod 700 .beads                                # bd warns if not 0700
```

### v1.0.0 architecture

- **Embedded Dolt** — no separate server, no daemon process. Each `bd` command opens the DB, runs, exits. Exclusive file lock means **one writer at a time**.
- **No `bd sync`, no `bd daemon`** — both removed. Beads auto-commits to Dolt on every mutation.
- **No SQLite fallback** — SQLite backend was deleted in v0.58. `--backend sqlite` only prints migration advice.
- **JSONL still exists** (`.beads/issues.jsonl`) as a plain-text export for git diffs, but is no longer the primary store.
- **`bd doctor` is not supported in embedded mode** — use `ls -la .beads/embeddeddolt/` and `bd info` instead.
- **Valid issue types:** `bug`, `feature`, `task`, `epic`, `chore`, `spike`, `story`, `milestone`, `merge-request`, `molecule`, `gate`, `agent`, `role`, `rig`, `convoy`, `event`. Use `feature` not `enhancement`.

### Usage

```bash
bd ready                            # Find available work
bd show <id>                        # View issue details
bd update <id> --claim              # Claim work (replaces --status in_progress)
bd close <id>                       # Complete work
bd close <id1> <id2>                # Close multiple at once
bd search "keyword"                 # Full-text search
bd dolt push / bd dolt pull         # Push/pull beads via Dolt remote
```

**Workflow:** `bd ready` → `bd update <id> --claim` → implement → `bd close <id>` → `git push`.

**NEVER use `bd edit`** — it opens `$EDITOR` which blocks agents.

### Persistent memory (`bd remember` / `bd memories` / `bd recall` / `bd forget`)

Project-specific facts live in the **beads memory store** — small keyed notes that auto-inject into every session via `bd prime` (installed as a `SessionStart` + `PreCompact` hook). They're an alternative to fragmented `MEMORY.md` files for repo-level knowledge: one canonical store per project, shared across agents on the same machine. Note: memories live in gitignored `.beads/embeddeddolt/`, so they're machine-local by default — see Session Completion below if you want them durable off-machine.

```bash
bd remember "insight" --key category:specific-item    # Save or update
bd memories                                           # List all
bd memories <keyword>                                 # Full-text search
bd recall <key>                                       # Fetch one by key
bd forget <key>                                       # Delete
```

**Key naming convention (from beads team):** `category:specific-item` with a colon separator. Current categories in use:

- `architecture:` — system layout decisions (`architecture:backtest-tiers`, `architecture:data-archive`)
- `command:` — commonly-used dev commands (`command:backend-start`, `command:dev-toolchain`)
- `design:` — contentious design decisions + their risks (`design:per-direction-sltp`)
- `discovery:` — discovery pipeline facts (`discovery:champions`, `discovery:fitness-formula`, `discovery:compiler-hash`)
- `gotcha:` — non-obvious pitfalls (`gotcha:sqlite-state-db`, `gotcha:bbands-normalized`)
- `indicator:` — indicator-specific facts (`indicator:rust-native-nt`)
- `ops:` — operational / rollback playbooks (`ops:bd-v1-rollback`)
- `pipeline:` — cross-component invariants (`pipeline:discovery-vs-validation`)
- `policy:` — hard rules (`policy:licenses-secrets`, `policy:indicators`)
- `reference:` — pointers to canonical docs (`reference:spec`, `reference:research-diaries`)

**When to use `bd remember` vs alternatives:**

| Fact type | Put it in |
|---|---|
| Project-level facts, repo-specific gotchas, invariants | `bd remember` (shared across all agents on any machine) |
| Trackable work with status / priority / dependencies | `bd create` (issues, not memories) |
| User-level prefs ("I like concise commits", "use rg not grep") | Claude Code auto-memory (`~/.claude/projects/.../memory/`) |
| Architecture that's already documented in SPEC.md | Nothing — just reference SPEC.md |
| Conversation-local state ("we're halfway through refactor X") | Plans / tasks, not memory |

**Best practices (from beads team + project conventions):**

1. **Lead with the fact, not the metadata.** Good: `"BacktestDataConfig data_cls must be the class object — a string disables bar-type narrowing"`. Bad: `"Remember that in the July session we found..."`.
2. **Update in place.** Passing `--key` to an existing key overwrites — don't create `discovery:champions-v2`.
3. **Keep memories evergreen.** If a fact is a dated snapshot ("as of 2026-04-11 we had 894 beads"), prefer a commit or a bead over a memory.
4. **Delete when stale.** `bd forget <key>` on facts that turn out wrong or get superseded.
5. **Don't duplicate CLAUDE.md / SPEC.md.** Those are already loaded. Memories are for facts that are either more specific, or that would otherwise be lost.
6. **When a session reveals a surprising non-obvious fact, save it** — future sessions get it for free via `bd prime`.

**Instructions for future sessions:** On session start, `bd prime` output will include all current memories. When you learn a new non-obvious project fact, `bd remember "fact" --key category:specific-item`. When CLAUDE.md contradicts a memory, prefer CLAUDE.md and delete or update the memory.

### Performance notes

- **Cold start (first call in a while):** ~1–3s (embedded Dolt schema load). Occasionally seen at ~20s on fully cold caches.
- **Warm queries:** 0.3–1.0s per command (`bd stats`, `bd list`, `bd ready`). ~5× slower than the old SQLite backend but still sub-second.
- **Disk usage:** `.beads/embeddeddolt/` ~34 MB vs old `.beads/beads.db` ~10 MB (Dolt stores full commit history).
- **Parallel agents:** the embedded backend holds an exclusive lock. If you need concurrent writers (multi-agent work), switch to server mode via `bd init --server`.

### Session Completion

Work is NOT complete until `git push` succeeds.

1. File issues for remaining work
2. Run quality gates (tests, linters) if code changed
3. Update issue status — close finished, update in-progress
4. Push:
   ```bash
   git pull --rebase && git push
   git status  # MUST show "up to date with origin"
   ```
5. If push fails, resolve and retry

**Caveat about `bd remember` durability:** `.beads/embeddeddolt/` is gitignored, so `git push` publishes code + `.beads/issues.jsonl` (issues survive) but NOT the Dolt store (memories + Dolt history are machine-local). If you want memories backed up off-machine, run `bd dolt push` — it publishes to `refs/dolt/data` on the same GitHub remote. This is optional and not part of the mandatory session-close flow (matches the beads team's own AGENTS.md recommendation).

## Directory Structure

```
vibe-quant/
├── vibe_quant/                    # Backend Python package
│   ├── api/
│   │   ├── app.py                 # FastAPI app factory
│   │   ├── deps.py                # DI dependencies (DB session etc.)
│   │   ├── routers/               # Route handlers: strategies, backtest, data, discovery, paper_trading, results, settings
│   │   ├── schemas/               # Pydantic request/response models (same domains as routers)
│   │   ├── sse/progress.py        # Server-sent events for progress
│   │   └── ws/                    # WebSocket handlers: discovery, jobs, trading
│   ├── data/                      # Data pipeline: downloader, archive (SQLite), catalog (Parquet)
│   ├── db/                        # SQLite: connection (WAL), schema, state_manager
│   ├── discovery/                 # Genetic algo strategy discovery: genome, operators, fitness, pipeline
│   ├── dsl/                       # Strategy DSL: parser, compiler, schema, indicators, conditions, translator
│   ├── overfitting/               # Overfitting filters: WFA, purged k-fold, DSR
│   ├── paper/                     # Paper trading: node (NT engine), persistence, config, CLI
│   ├── risk/                      # Risk actors, portfolio/strategy actors, sizing
│   ├── screening/                 # Screening pipeline: NT runner, grid sweep, consistency checks
│   ├── strategies/                # Strategy templates + examples
│   ├── validation/                # Validation pipeline: NT runner, fill/latency models, results extraction
│   ├── alerts/telegram.py         # Telegram notifications
│   └── logging/                   # Structured logging
│
├── frontend/                      # React/TypeScript SPA (Vite + TailwindCSS 4 + shadcn)
│   ├── src/
│   │   ├── api/                   # API client + generated hooks (orval from OpenAPI)
│   │   │   ├── client.ts          # Axios instance with /api proxy
│   │   │   └── generated/         # Auto-generated: models + per-router hooks (DO NOT EDIT)
│   │   ├── components/
│   │   │   ├── backtest/          # BacktestLaunchForm, SweepBuilder, ActiveJobsPanel, PreflightStatus
│   │   │   ├── charts/            # Recharts/Plotly charts: CandlestickChart, EquityCurve, Drawdown, Heatmap, etc.
│   │   │   ├── data/              # DataBrowser, IngestForm, CoverageTable, DownloadProgress
│   │   │   ├── discovery/         # DiscoveryConfig, DiscoveryJobList, DiscoveryResults, DiscoveryProgress
│   │   │   ├── layout/            # Header, Sidebar, PageLayout
│   │   │   ├── paper/             # LiveDashboard, SessionControl, PositionsTable, CheckpointsList
│   │   │   ├── results/           # MetricsPanel, TradeLog, ChartsPanel, SweepAnalysis, WinLossPanel, etc.
│   │   │   ├── settings/          # DatabaseTab, LatencyTab, RiskTab, SizingTab, SystemTab
│   │   │   ├── strategies/        # StrategyList, StrategyCreateDialog, StrategyWizard, StrategyEditor, StrategyDeleteDialog
│   │   │   │   └── editor/        # GeneralTab, IndicatorsTab, ConditionsTab, RiskTab, TimeTab, YamlEditor, types.ts
│   │   │   └── ui/                # shadcn primitives + StrategyCard, etc.
│   │   ├── hooks/                 # Custom React hooks
│   │   ├── lib/utils.ts           # cn() and other helpers
│   │   ├── routes/                # TanStack Router file-based routes
│   │   │   ├── __root.tsx         # Root layout
│   │   │   ├── index.tsx          # Dashboard/home
│   │   │   ├── strategies.tsx     # Strategy management list page
│   │   │   ├── strategies.$strategyId.tsx  # Strategy editor page
│   │   │   ├── backtest.tsx, discovery.tsx, results.tsx
│   │   │   ├── paper-trading.tsx, data.tsx, settings.tsx
│   │   ├── stores/                # Zustand global stores
│   │   ├── app.tsx                # Router setup + providers
│   │   └── index.css              # TailwindCSS 4 theme (dark-only, OKLch colors, Geist/JetBrains fonts)
│   ├── e2e/                       # Playwright E2E tests
│   └── orval.config.ts            # API codegen config
│
├── tests/                         # Python unit tests (pytest)
│   ├── unit/api/                  # API route tests
│   └── fixtures/                  # Test data + known_results
├── data/                          # Runtime data (gitignored)
│   ├── archive/                   # SQLite raw data archive
│   ├── catalog/                   # ParquetDataCatalog
│   └── state/                     # Paper trading state
├── docs/
│   ├── claude/conventions.md      # Coding conventions (authoritative)
│   ├── plans/                     # Implementation plans
│   └── reviews/                   # Code review notes
├── scripts/                       # Utility scripts
├── SPEC.md                        # Authoritative implementation spec
└── CLAUDE.md                      # This file
```

**Key paths to remember:**

- Strategy card: `frontend/src/components/ui/StrategyCard.tsx`
- Strategy list/page: `frontend/src/components/strategies/StrategyList.tsx`, `frontend/src/routes/strategies.tsx`
- Theme/CSS vars: `frontend/src/index.css`
- API hooks: `frontend/src/api/generated/` (auto-generated, don't edit)
- Backend entry: `vibe_quant/api/app.py` → `main.py` entrypoint via uvicorn
- DB schema: `vibe_quant/db/schema.py`
- DSL types: `vibe_quant/dsl/schema.py`, frontend mirror: `frontend/src/components/strategies/editor/types.ts`

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

- **`BacktestDataConfig.data_cls` must be the CLASS object** (`from nautilus_trader.model.data import Bar`), never the import string. `config.query` compares `data_cls is Bar`; a string silently disables `bar_types`/`bar_spec` narrowing and loads the ENTIRE catalog (~300× data, and fills change because the venue processes stray finer-granularity bars). Upstream issue draft: `docs/nt-upstream-issue-data-cls.md`.
- **Creating a `BacktestEngine`/`BacktestNode` after disposing a previous one in the same process hard-aborts** (Rust logger can only init once; the abort message only appears with tracebacks — the process just dies). Call `vibe_quant.nt_compat.retain_log_guard(engine)` before dispose; both runners and `test_fill_timing.py` already do.
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

The files in `docs/` predate `SPEC.md` and contain **outdated architectural decisions** (FreqTrade, VectorBT, PostgreSQL, TimescaleDB, Redis, 5-year data). They are retained as research context only. When any `docs/*.md` file contradicts `SPEC.md`, **SPEC.md wins**.

| File                                       | Status                | Contents                                                    |
| ------------------------------------------ | --------------------- | ----------------------------------------------------------- |
| `docs/opus-prd.md`                         | Superseded by SPEC.md | Original PRD (recommends FreqTrade -- no longer applicable) |
| `docs/opus-spec.md`                        | Superseded by SPEC.md | Older technical spec (hybrid VectorBT approach -- removed)  |
| `docs/opus-research.md`                    | Historical reference  | Framework comparison research                               |
| `docs/gpt-research.md`                     | Historical reference  | Extended framework evaluation                               |
| `docs/crypto-trading-bot-specification.md` | Historical reference  | Original bot specification                                  |
