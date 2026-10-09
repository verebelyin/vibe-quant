# CLAUDE.md

Algorithmic trading engine for crypto perpetual futures using NautilusTrader (Rust core) with two-tier backtesting (screening + validation), strategy DSL, overfitting prevention, and paper/live execution.

## This file

The rule of this file is to describe common mistakes and confusion points that agents might encounter as they work in this project. If you ever encounter something in this project that surprises you, or you failed to do after multiple attempts, please alert the developer working with you and describe it in this file to help prevent future agents from having the same issue. (AGENTS.md is a pointer to this file for non-Claude tools — never duplicate content there.)

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
6. **UI testing:** to check a worktree/branch next to the user's servers, `scripts/agents/ui-check.sh up|down <worktree>` (backend :8001 + Vite :5188 on a DB copy; Vite reads `VQ_API_PORT`). Use the `agent-browser` skill — **always with `dangerouslyDisableSandbox: true`**
   (it needs the `~/.agent-browser` socket dir). Sidebar pages: Strategy Management, Discovery,
   Backtest Launch, Results Analysis, Paper Trading, Data Management, Settings. E2E flow:
   Data Management (download) → Strategy Management (create) → Backtest Launch (screen) →
   Results Analysis (verify).
7. **Git worktrees import the MAIN repo's code.** The editable install (`.pth`) points at
   `/Users/verebelyin/projects/vibe-quant`, so `python script.py` inside a worktree runs main's
   `vibe_quant`. Use `PYTHONPATH=$PWD .venv/bin/python -m ...` from the worktree.
   Bare `git worktree add` also lacks `data/catalog` (8 research tests fail); create worktrees with
   `scripts/agents/worktree.sh <slug> [base]`, which links the market data and prints the run line.

## Shell Preferences

- **Always use `rg` (ripgrep) instead of `grep`** — faster, simpler regex syntax (no escaping `|`), better defaults. Use `rg` in Bash tool calls, skills, and scripts. This applies to ALL search operations in the terminal.
- **`status` is read-only in zsh** — never use it as a variable name in shell scripts. Use `st`, `stat`, or `run_status` instead.
- **Don't use `sleep N` in Bash tool calls for polling** — make separate tool calls when ready instead. `sleep` blocks the tool and wastes time.
- **Check ports before starting servers**: `lsof -i :8000` before launching uvicorn. Avoids "address already in use" errors.

## SQLite Queries (state DB)

DB path: `data/state/vibe_quant.db` (override with `VIBE_QUANT_DB` at backend start — inherited by every job subprocess; there is no runtime DB switch). Always use WAL mode.

**Opening the DB runs schema migrations.** `tests/conftest.py` pins `VIBE_QUANT_DB` to a temp file so a test that forgets its tmp DB can't migrate the real one — keep that guard; scratch scripts against the real DB should use `?mode=ro` URIs.

`backtest_results` has ONE row per run (re-runs replace result + trades). `notes` is machine JSON (discovery payload, data_window, funding, consistency); user text lives in `user_notes`.

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

- **Screening mode**: simplified fills, no latency, multiprocessing parallelism -- models leverage + funding (from the archive; holes charged a flagged fallback rate), NOT liquidation. Fills land at the NEXT strategy-timeframe bar close (validation: ~1 min after signal via 1m detail) — fast-cycling strategies trade noticeably less in screening.
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

- **Audit 2026-10-02 (`docs/reviews/2026-10-02-deep-audit.md`, epic `vibe-quant-e70tl`) — critical/high fixed 2026-10-03 = SEMANTICS BREAK.** Screening/validation metrics changed (trade-based PF capped at 100, mark-to-market max DD, funding in screening, adaptive same-bar ordering, TP/SL fills at limit/trigger, sizing rounds down), discovery changed (worst-of-N eval windows, default 20% holdout used once as final gate, every gate fails closed → 0 champions + `guardrail_rejections`). Champions/journal scores before this are NOT comparable. Medium/low leftovers: bead `vibe-quant-e70tl.23`.

- **Research diary:** `docs/discovery-journal.md` — experiment log with GA configs, metrics, and findings
- Discovery and screening use **identical** code path (`NTScreeningRunner` → `StrategyCompiler`). Results match exactly *within one run* (champion → replay).
- Validation uses custom fill model + latency + 1m detail. It clamps the run window to 1m coverage (recorded in `notes.data_window`) and FAILS without 1m data. Its consistency check flags screening→validation collapse/trade divergence against the same window (holdout-validated champions use the champion's holdout metrics).
- **Bug fix `2944ad3`:** `pos.entry→pos.side` enum mismatch caused 155:1 trade ratio. All runs before this fix are invalid.
- **Semantics break `11c5f00` (2026-07-09):** screening now feeds ONLY the strategy timeframe (an NT data-loading bug previously fed ALL timeframes incl. 1m, giving screening accidental intrabar fills). Discovery scores ≤ run 854 are not comparable with newer runs. **Validation is unchanged** — it loads 1m detail explicitly and reproduces historical results bit-for-bit.
- **Compiler version hash:** stored in discovery notes for staleness detection. Changes when the indicator registry changes — check `bd recall discovery:compiler-hash` for the current value; recompute with `compiler_version_hash()`.
- Champion rankings live in `bd recall discovery:champions` (journal has full history). Batch-13 STOCH+CCI headline numbers are historical only (pre-`11c5f00`).
- **1m data is slow:** Rust-native indicators (SMA/EMA/CCI/STOCH/ATR) ~10x faster than pandas-path ones (ADX/MACD/BBANDS/KAMA). Budget accordingly.
- **Fitness function:** 35% Sharpe + 25% (1-MaxDD) + 20% PF + 20% Return. Hard gate: 0 if <50 trades.
- **`eval_windows` (default 3) stores WORST-of-N sub-window metrics** (min Sharpe/return/PF, max DD; each window needs ≥ max(1, min_trades // (2N)) trades) — a full-window replay legitimately shows different Sharpe/return (`ReplayResponse.metrics_note` explains this). Not a bug.
- **Discovery seed:** `--seed N` (CLI only; auto-drawn when absent, always persisted in `notes.seed`) replays a run exactly — same seed + same config + same code. Runs without a recorded seed (pre-2026-10-08) can't be replayed. Still never compare discovery runs ACROSS code changes (any change shifts the RNG draw sequence).
- **Discovery fails loudly:** an all-errored eval batch (before any success, or ≥ max(2, pop//4) after) raises `DiscoveryEvaluationError` → run `failed`, not "0 champions".
- **Bootstrap-CI gate keeps being vindicated:** every champion forced past it with `no_bootstrap_ci=true` and then validated has collapsed (Batch 41: 5.40→−2.78; Batch 43 RAMS: 0.59→−0.36). The validation runner auto-flags collapses (`validation/consistency.py`); treat a flagged strategy as overfit, not as a validation bug.
- 4h/1d discovery uses bootstrap floor 0.0 by default (1.0 is structurally unpassable at ~50-180 trades/yr); 1m uses 0.5.

## NautilusTrader Gotchas (each of these cost real debugging time)

- **IMPORTANT: `BacktestDataConfig.data_cls` MUST be the CLASS object** (`from nautilus_trader.model.data import Bar`), never the import string. `config.query` compares `data_cls is Bar`; a string silently disables `bar_types`/`bar_spec` narrowing and loads the ENTIRE catalog (~300× data, and fills change because the venue processes stray finer-granularity bars). Upstream issue draft: `docs/nt-upstream-issue-data-cls.md`.
- **IMPORTANT: creating a `BacktestEngine`/`BacktestNode` after disposing a previous one in the same process hard-aborts** (Rust logger can only init once; the process just dies with no Python traceback). YOU MUST call `vibe_quant.nt_compat.retain_log_guard(engine)` before dispose in any new engine-lifecycle code; both runners and `test_fill_timing.py` already do.
- **NT 1.226+ config decoding rejects unknown fields** (fast-fail). Forward only params the generated `StrategyConfig` declares (`__struct_fields__` filter in both runners). Run-level knobs like `initial_balance`/`leverage` belong on the venue, not the strategy config.
- **`node.build()` swallows engine-build exceptions** — `get_engine()` returns `None` and you see only "engine not found". Validation sets `BacktestRunConfig(raise_exception=True)`; keep it that way, and set it when writing new runner code.
- **NT `BacktestResult.elapsed_time` is the simulated window in seconds**, not wall time. `Iterations` ≈ bars processed — if it's far above the expected bar count, you have a data-loading bug (see Performance Playbook).
- **Invalid `TraderId` (e.g. `paper_1`, no `-`) aborts the whole process from Rust** — no Python exception. Validate with `paper/config.py`'s regex first.
- **`TradingState.REDUCING` does NOT stop a flat account from opening a position** — paper uses its own `TradingGuard` order gate (`paper/guard.py`).
- `BacktestEngine.run()` stops all strategies when it ends — assert mid-run strategy state from a scheduled actor, not after `run()`.
- Generated strategy modules are content-addressed (`{name}_{hash}`) — always use `module.__name__`, never rebuild the path from `dsl.name`.
- ADX stays on the pandas path deliberately (NT has no true ADX — `DirectionalMovement.value` is always 0). Don't "optimize" it to `nt_class` without checking values.

## Swarm board (every agent: read it, write to it)

A persistent message board shared by every agent and session in this repo — main sessions, subagents, `cmd` workers, future swarms. The SessionStart hook prints the latest posts; full view `docs/orchestration/board/BOARD.md`.
- Catch up: `python3 scripts/agents/bus.py --as <you> board read --global --recent 30`
- Record: `... board post --topic findings|gotchas|decisions|model-notes|thoughts --body "..."` — anything a future agent should know (persisted in git). Chat: `--topic chat`, `post --to <agent>`, `reply --ref <id>`.
- Orchestrator: `... --as chief digest` reads everything new on the persistent board + every job bus. Details: [docs/orchestration/README.md § Swarm bus](docs/orchestration/README.md).

## Performance Profiling

Use the `perf-profiling` skill (`.claude/skills/perf-profiling/SKILL.md`) when investigating
slow runs or indicators — it documents how the 300× and 4.4× wins were found. Every perf
change ships with an exactness proof (see Verification Rules).

## Multi-Agent Orchestration

Large jobs (many beads, research campaigns): `orchestrate` skill + [`docs/orchestration/`](docs/orchestration/README.md)
(roster in `.claude/agents/`, handoff contract, research lane). Mechanical bulk work goes to the cheap
`cmd` CLI (DeepSeek/Qwen/GLM/Kimi) via `scripts/agents/cmd-task.sh` — read `docs/orchestration/cheap-agents.md`
before delegating; `cmd` runs `--yolo`, so writes go in a disposable worktree.

## Verification Rules (what counts as "results are the same")

Valid proofs that a change preserved correctness:
- **Fixed-strategy eval before/after** (`scripts/agents/exactness_239.py`): one `NTScreeningRunner` call on a saved strategy must
  return bit-identical metrics (strategy 239, BTCUSDT 2024-01-01..2026-03-17: sharpe
  `1.3169239785208688`, 68 trades since the 2026-10-08 data refresh filled a BTC funding hole —
  bars unchanged, funding only; `1.3165049716553048` after the 2026-10-03 audit fixes).
- **Validation repeatability**: the same validation run twice is bit-identical (strategy 239
  since the 2026-10-08 data refresh: sharpe `0.780452281957465`, 67 trades; `0.7802305953007851` after
  the 2026-10-03 audit fixes; runs 868 == 870 before). Any drift = regression.
- **Within-run replay**: discovery champion → `/replay` matches exactly when the run used
  `eval_windows=1`.
- Zero-tolerance unit tests against the reference implementation for ported math.

INVALID proofs (these wasted time):
- Comparing two discovery runs — single-seed runs are unseeded and nondeterministic
  unless both used the same `--seed`, and ANY code change shifts the RNG draw sequence.
- Comparing an `eval_windows>1` champion's stored fitness to a full-window replay
  (mean-of-N sub-windows vs full window — differs by design).
- Comparing screening metrics across the `11c5f00` or 2026-10-03 audit-fix semantics breaks.

## Historical Documentation

`docs/opus-prd.md`, `docs/opus-spec.md`, `docs/opus-research.md`, `docs/gpt-research.md`, and `docs/crypto-trading-bot-specification.md` predate SPEC.md and describe **abandoned architectures** (FreqTrade, VectorBT, PostgreSQL, Redis). Never follow them; when any `docs/*.md` contradicts SPEC.md, **SPEC.md wins**.
