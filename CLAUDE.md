# CLAUDE.md

Algorithmic trading engine for crypto perpetual futures using NautilusTrader (Rust core) with two-tier backtesting (screening + validation), strategy DSL, overfitting prevention, and paper/live execution.

## This file

The rule of this file is to describe common mistakes and confusion points that agents might encounter as they work in this project. If you ever encounter something in this project that surprises you, or you failed to do after multiple attempts, please alert the developer working with you and indicate and describe that in the @AGENTS.md file to help prevent future agents from having the same issue.

## Quick Reference

- **Package manager:** `uv` (not pip/poetry)
- **Python:** 3.13
- **Install:** `uv pip install -e .`
- **Tests:** `pytest` (target 80% coverage on core modules)
- **Lint:** `ruff check`
- **Type check:** `mypy`
- **Backend:** `.venv/bin/uvicorn vibe_quant.api.main:app --port 8000`
- **Frontend:** `cd frontend && pnpm dev` (Vite on port 5173)
- **Frontend build:** `cd frontend && pnpm build`

## UI Testing (agent-browser)

Start backend + frontend then test with `agent-browser`. **Always use `dangerouslyDisableSandbox: true`** for agent-browser commands (it needs `~/.agent-browser` socket dir).

**Quick start:**

```bash
# Start backend + frontend (background)
.venv/bin/uvicorn vibe_quant.api.main:app --port 8000 &
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
- **Indicators:** Prefer NautilusTrader built-in (Rust) indicators. Fall back to `pandas-ta-classic` for exotic ones. Never use the original `pandas-ta` (compromised maintainership).
- **Data:** Raw downloaded data archived in SQLite before processing to ParquetDataCatalog. Catalog is rebuildable from archive.

## Issue Tracking (Beads)

**CRITICAL:** Use `bd` (beads) for ALL task/issue tracking. **NEVER use TodoWrite, TaskCreate, or markdown files for task tracking.** Beads is the single source of truth.

### Installing `bd`

**This environment** (Claude Code remote container): DNS doesn't resolve for npm/go, but `curl` works through the envoy proxy. Use this two-step install:

```bash
# 1. Install npm wrapper (skip postinstall since it can't reach GitHub directly)
npm install -g @beads/bd --ignore-scripts

# 2. Download the binary via curl and place it where the npm wrapper expects it
VERSION=0.49.4
curl -sL -o /tmp/beads.tar.gz \
  "https://github.com/steveyegge/beads/releases/download/v${VERSION}/beads_${VERSION}_linux_amd64.tar.gz"
tar xzf /tmp/beads.tar.gz -C /tmp
cp /tmp/bd "$(npm root -g)/@beads/bd/bin/bd"
chmod +x "$(npm root -g)/@beads/bd/bin/bd"

# 3. Initialize and start daemon
bd init                # Creates .beads/beads.db, imports issues from JSONL
bd hooks install       # Installs pre-commit/pre-push shims
bd daemon start        # Background daemon for auto-export to JSONL
```

**Other environments** (normal network access):

```bash
npm install -g @beads/bd          # npm (auto-downloads binary)
bun install -g --trust @beads/bd  # bun alternative
go install github.com/steveyegge/beads/cmd/bd@latest  # go (needs go 1.25+)
```

### Daemon notes

- The daemon auto-exports to JSONL after every mutation — no manual `bd sync` needed for daily work
- CRUD commands (`bd ready/show/create/update/close`) go through daemon RPC and work while daemon runs
- `bd sync` and `bd doctor` need direct DB access — stop the daemon first: `bd daemon stop <PID>`
- If daemon crashes (repo fingerprint mismatch after push): `rm -f .beads/beads.db .beads/beads.db-wal .beads/beads.db-shm && bd init`
- Valid issue types: `bug`, `feature`, `task`, `epic`, `chore`, `merge-request`, `molecule`, `gate`, `agent`, `role`, `rig`, `convoy`, `event`. Use `feature` not `enhancement` (alias fails import validation)

**Fallback (no `bd`):** Edit `.beads/issues.jsonl` directly. Each issue is one JSON line:

```json
{
  "id": "vibe-quant-xxx",
  "title": "...",
  "description": "...",
  "status": "open",
  "priority": 1,
  "issue_type": "bug",
  "owner": "verebelyin@gmail.com",
  "created_at": "2026-02-07T14:00:00.000000+01:00",
  "created_by": "Claude"
}
```

To list open issues without `bd`: `python3 -c "import json; [print(f'{j[\"id\"]}: {j[\"title\"]}') for line in open('.beads/issues.jsonl') if (j:=json.loads(line.strip())) and j.get('status')=='open']"`

### Usage

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work BEFORE starting
bd close <id>         # Complete work
bd close <id1> <id2>  # Close multiple at once
bd sync               # Sync with git
```

**Workflow:** `bd ready` → `bd update <id> --status in_progress` → implement → `bd close <id>` → `bd sync`

**NEVER use `bd edit`** -- it opens $EDITOR which blocks agents.

### Session Completion

Work is NOT complete until `git push` succeeds.

1. File issues for remaining work
2. Run quality gates (tests, linters) if code changed
3. Update issue status - close finished, update in-progress
4. Push:
   ```bash
   git pull --rebase && bd sync && git push
   git status  # MUST show "up to date with origin"
   ```
5. If push fails, resolve and retry

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
- Discovery and screening use **identical** code path (`NTScreeningRunner` → `StrategyCompiler`). Results match exactly.
- Validation uses custom fill model + latency → fewer trades and worse metrics (expected)
- **Bug fix `2944ad3`:** `pos.entry→pos.side` enum mismatch caused 155:1 trade ratio. All runs before this fix are invalid.
- **Compiler version hash:** stored in discovery notes for staleness detection. Current valid hash: `8ae876464003`
- **1m data is slow:** Rust-native indicators (SMA/EMA/ADX/ATR) ~10x faster than pandas-ta (MACD/STOCH/BBANDS). Budget accordingly.
- **Fitness function:** 35% Sharpe + 25% (1-MaxDD) + 20% PF + 20% Return. Hard gate: 0 if <50 trades.

## Historical Documentation

The files in `docs/` predate `SPEC.md` and contain **outdated architectural decisions** (FreqTrade, VectorBT, PostgreSQL, TimescaleDB, Redis, 5-year data). They are retained as research context only. When any `docs/*.md` file contradicts `SPEC.md`, **SPEC.md wins**.

| File                                       | Status                | Contents                                                    |
| ------------------------------------------ | --------------------- | ----------------------------------------------------------- |
| `docs/opus-prd.md`                         | Superseded by SPEC.md | Original PRD (recommends FreqTrade -- no longer applicable) |
| `docs/opus-spec.md`                        | Superseded by SPEC.md | Older technical spec (hybrid VectorBT approach -- removed)  |
| `docs/opus-research.md`                    | Historical reference  | Framework comparison research                               |
| `docs/gpt-research.md`                     | Historical reference  | Extended framework evaluation                               |
| `docs/crypto-trading-bot-specification.md` | Historical reference  | Original bot specification                                  |
