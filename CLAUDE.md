# CLAUDE.md

Algorithmic trading engine for crypto perpetual futures using NautilusTrader (Rust core) with two-tier backtesting (screening + validation), strategy DSL, overfitting prevention, and paper/live execution.

## Quick Reference

- **Package manager:** `uv` (not pip/poetry)
- **Python:** 3.13
- **Install:** `uv pip install -e .`
- **Tests:** `pytest` (target 80% coverage on core modules)
- **Lint:** `ruff check`
- **Type check:** `mypy`
- **Dashboard:** `.venv/bin/streamlit run vibe_quant/dashboard/app.py --server.port 8501 --server.headless true`

## UI Testing (agent-browser)

Start the dashboard then test with `agent-browser`. **Always use `dangerouslyDisableSandbox: true`** for agent-browser commands (it needs `~/.agent-browser` socket dir).

**Quick start:**
```bash
# Start app (background)
.venv/bin/streamlit run vibe_quant/dashboard/app.py --server.port 8501 --server.headless true &

# Open and take initial screenshot
agent-browser open http://localhost:8501 && agent-browser screenshot /tmp/claude/page.png
```

**Chain commands with `&&`** to reduce round-trips:
```bash
# Navigate + snapshot in one call
agent-browser open http://localhost:8501 && agent-browser snapshot -i

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

| Detail | Reference |
|--------|-----------|
| Full implementation spec | [`SPEC.md`](SPEC.md) -- **the authoritative source** for architecture, DSL, pipelines, data, schemas, and phases |
| Sections 1-5 | Architecture, tech stack, decisions, data layout, strategy DSL |
| Sections 6-7 | Screening pipeline, validation backtesting |
| Sections 8-13 | Overfitting, risk, dashboard, paper trading, observability, testing |
| Phases 1-8 | Implementation roadmap with deliverables and acceptance criteria |

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
{"id":"vibe-quant-xxx","title":"...","description":"...","status":"open","priority":1,"issue_type":"bug","owner":"verebelyin@gmail.com","created_at":"2026-02-07T14:00:00.000000+01:00","created_by":"Claude"}
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

## Historical Documentation

The files in `docs/` predate `SPEC.md` and contain **outdated architectural decisions** (FreqTrade, VectorBT, PostgreSQL, TimescaleDB, Redis, 5-year data). They are retained as research context only. When any `docs/*.md` file contradicts `SPEC.md`, **SPEC.md wins**.

| File | Status | Contents |
|------|--------|----------|
| `docs/opus-prd.md` | Superseded by SPEC.md | Original PRD (recommends FreqTrade -- no longer applicable) |
| `docs/opus-spec.md` | Superseded by SPEC.md | Older technical spec (hybrid VectorBT approach -- removed) |
| `docs/opus-research.md` | Historical reference | Framework comparison research |
| `docs/gpt-research.md` | Historical reference | Extended framework evaluation |
| `docs/crypto-trading-bot-specification.md` | Historical reference | Original bot specification |
