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

## Historical Documentation

The files in `docs/` predate `SPEC.md` and contain **outdated architectural decisions** (FreqTrade, VectorBT, PostgreSQL, TimescaleDB, Redis, 5-year data). They are retained as research context only. When any `docs/*.md` file contradicts `SPEC.md`, **SPEC.md wins**.

| File | Status | Contents |
|------|--------|----------|
| `docs/opus-prd.md` | Superseded by SPEC.md | Original PRD (recommends FreqTrade -- no longer applicable) |
| `docs/opus-spec.md` | Superseded by SPEC.md | Older technical spec (hybrid VectorBT approach -- removed) |
| `docs/opus-research.md` | Historical reference | Framework comparison research |
| `docs/gpt-research.md` | Historical reference | Extended framework evaluation |
| `docs/crypto-trading-bot-specification.md` | Historical reference | Original bot specification |
