# CLAUDE.md

Algorithmic trading engine for crypto perpetual futures using NautilusTrader (Rust core) with two-tier backtesting (screening + validation), strategy DSL, overfitting prevention, and paper/live execution.

## Quick Reference

- **Package manager:** `uv` (not pip/poetry)
- **Python:** 3.13
- **Install:** `uv pip install -e .`
- **Tests:** `pytest` (target 80% coverage on core modules)
- **Lint:** `ruff check`
- **Type check:** `mypy`
- **Dashboard:** `streamlit run vibe_quant/dashboard/app.py`

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

Install via npm (preferred in this env), or any method from [beads repo](https://github.com/steveyegge/beads):

```bash
npm install -g @beads/bd          # npm (needs network to github for binary)
bun install -g --trust @beads/bd  # bun alternative
go install github.com/steveyegge/beads/cmd/bd@latest  # go (needs go 1.25+)
# Or: curl -fsSL https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh | bash
```

**Fallback (no network):** If `bd` cannot be installed (e.g., no external network), edit `.beads/issues.jsonl` directly. Each issue is one JSON line:
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
