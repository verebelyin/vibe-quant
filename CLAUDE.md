# CLAUDE.md

Algorithmic trading engine for crypto perpetual futures (Binance USDT-M; Ethereal DEX later) on NautilusTrader (Rust core): strategy DSL → two-tier backtesting (screening + validation) → overfitting gates → paper → live. FastAPI backend (`vibe_quant/`) + React SPA (`frontend/`). [`SPEC.md`](SPEC.md) is the authoritative spec; when another doc disagrees, SPEC.md wins.

## Goal and priorities

**Goal:** find crypto-perp strategies with a real, repeatable edge and trade them safely — paper first, live only with the user's explicit approval. The product is trustworthy evidence: a strategy that looks great is a bug until the gates, validation and paper agree.

**State (2026-10-11):** SPEC phases 1–8 are built (data layer, DSL, screening/validation, overfitting gates, React dashboard replacing the Streamlit plan, paper trading with alerts, Ethereal client, GA discovery) plus a Reddit research pipeline. Under current semantics recent discovery runs end with 0 champions (runs 877/878); older champion numbers (e.g. Batch 13) predate comparability breaks. Current work: engine fidelity, research-loop speed, honest gates.

**When goals conflict, earlier wins:**
1. **Honest numbers** — no look-ahead; screening, validation and paper fill the same way; gates fail closed; exactness proofs stay bit-identical unless a comparability break is declared.
2. **Money safety** — paper/live changes need a risk check and the user's explicit approval.
3. **Research-loop speed** — faster backtests, discovery and agent rounds compound.
4. Features and UI polish.

**Operator:** one person (the user); agents write most of the code. The user values fast iteration, bit-identical proofs, honest failure reports and a git history that explains decisions.

## Vocabulary

- **screening** — fast NT backtest with simplified fills; discovery uses the same code path.
- **validation** — full-fidelity NT backtest on 1m detail bars; the reference for fills.
- **discovery** — GA search over DSL genomes; a **champion** is a genome that passes every **gate** (bootstrap CI, DSR, cross-window, WFA, **holdout**).
- **eval windows** — discovery scores the worst of N sub-windows.
- **exactness proof** — `exactness_239.py` must reproduce the pinned strategy-239 metrics bit for bit.
- **comparability break** — a dated change after which older metrics can't be compared with newer ones.
- **fill tick / outbox** — orders queue per instrument and release on that instrument's next price update.
- **archive → catalog** — raw downloads in SQLite, rebuilt into the Parquet catalog the engine reads.
- **plugin** — a drop-in indicator file in `vibe_quant/dsl/plugins/`.
- **chief / maker / checker** — orchestrator / coding agent / reviewer; **bead** — an issue in `bd`.

## This file

Always-loaded rules, commands and gotchas for every agent here. When something surprises you, or you fail at it twice, tell the developer and add one line (with the check that catches it) — here if every task needs it, otherwise in the matching `docs/claude/` file. `AGENTS.md` only points here.

## Read before you touch

| Before you… | Read |
|---|---|
| write or change code that builds/runs a NautilusTrader engine or node (runners, paper node, fill/latency models, data configs, generated strategies) | [`docs/claude/nautilus-gotchas.md`](docs/claude/nautilus-gotchas.md) |
| run discovery, interpret or compare its results, or change discovery, fitness or gates | [`docs/claude/discovery.md`](docs/claude/discovery.md) |
| start servers, check UI, or run the Playwright spec | [`docs/claude/ui-testing.md`](docs/claude/ui-testing.md) |
| add an indicator | [`vibe_quant/dsl/plugins/README.md`](vibe_quant/dsl/plugins/README.md) |
| run a multi-agent job | `orchestrate` skill + [`docs/orchestration/README.md`](docs/orchestration/README.md) |
| investigate slow runs or indicators | `perf-profiling` skill |
| use beads beyond the basics below | [`docs/claude/beads.md`](docs/claude/beads.md) |

## Working principles

Bias toward caution over speed on money paths; use judgment on trivial tasks.

1. **Think before coding.** State assumptions; when uncertain, ask. When a request has several readings, present them instead of picking one silently; when a simpler approach exists, say so and push back when warranted.
2. **Simplicity first.** Write the minimum code that solves the asked problem, following existing patterns (plugins, `StateManager`, runner seams). No speculative options, no single-use abstractions, no handling for impossible cases. Test: would a senior engineer call this overcomplicated?
3. **Surgical changes.** Every changed line traces to the request. Match the surrounding style; remove what your change orphaned; mention (don't delete) pre-existing dead code.
4. **Goal-driven execution.** Turn the task into a check that goes red first: a failing test, an exactness drift, a reproduced bug. For multi-step work, write a short plan with a check per step, and persist intermediate results (bead notes, `data/swarm/<job>/` ledger, files) so an interrupted session can resume.
5. **Verify the kill-zone hardest.** Mistakes that stay invisible for weeks get an independent check: fill timing and look-ahead, metric math (Sharpe/PF/drawdown/fitness), overfitting gates (they fail closed), DSL codegen, DB writes, anything that places orders.
6. **Mark uncertainty inline.** State verified claims plainly; prefix the rest with `Likely:` (+ evidence) or `Assuming:` (+ what breaks if wrong). A guess never reads as fact.
7. **Verify library claims at the source.** For NautilusTrader, pandas-ta-classic, FastAPI or frontend library behaviour, versions and best practices, check the installed source (`.venv/lib/python3.13/site-packages/...`, `node_modules/`) or official docs/web search before asserting — NT changes between minors.
8. **Label performance and money numbers.** Perf claims are MEASURED (command + output) or EYEBALLED; predict the expected number before measuring. Metrics are copied verbatim from tool output, never rounded or recalled.

## Implementation guardrails

- **Documentation:** create `.md` files for a task only when asked. Gotchas go into this file or the matching `docs/claude/` file; research results into `docs/discovery-journal.md`; project facts into `bd remember`.
- **Env & secrets:** a new env var goes into `.env.example` in the same change; secrets only from env (Binance, Ethereal, Telegram).
- **Tests prove outcomes:** assert the product result (a trade filled at the right price, a run ending `completed`, a gate rejecting), not just a status code or "no exception". Name tests and test files by feature; dates, bead ids and review-finding ids belong in commit messages.
- **Flake or regression:** rerun the failing test alone (`pytest path::test_id`, then its file) before calling it flaky; a failure that reproduces alone is a regression.

## Finishing a task

1. Targeted tests for the changed modules plus every module that imports a changed symbol (`rg` the import).
2. `ruff check && mypy` → 0 errors.
3. Changed anything under `vibe_quant/` that can move numbers (screening, validation, discovery, dsl, overfitting, data) → `exactness_239.py` IDENTICAL (see Verification rules).
4. UI or UI-visible API change → check it on screen ([`docs/claude/ui-testing.md`](docs/claude/ui-testing.md)) and regenerate the orval client if the OpenAPI schema changed.
5. Full suite once, at the end (`swarm-check.sh` runs it).
6. Commit per [Commit messages](#commit-messages); push as part of [Session completion](#issue-tracking-beads).

Coverage axes — check which ones the change crosses and verify each side:
- **Engine tier:** screening vs validation (different fill paths; discovery = screening code path).
- **Symbols:** single vs multi-symbol (fill release is per instrument).
- **Timeframe:** 4h/1d vs 1m (different defaults, data volume, bootstrap floors).
- **Discovery windows:** `eval_windows` 1 vs >1 (worst-of-N metrics).
- **Frontend ↔ API:** generated client in `frontend/src/api/generated/` matches the backend schema.

## Commands

`uv` + Python 3.13 (`uv pip install -e .`); `pnpm` for the frontend. Lint and types are at a **zero baseline** (since 2026-07-09): any `ruff`/`mypy` error you see comes from current work.

| Task | Command |
|---|---|
| Tests (targeted / full ≈ 1 min) | `.venv/bin/python -m pytest tests/unit/test_x.py -q` / `... tests/unit -q -n auto --dist worksteal -m "not timing"` then `... tests/unit -m timing` (serial; 183 s serial without `-n`) |
| Lint / types | `.venv/bin/ruff check` / `.venv/bin/mypy` |
| Exactness proof (≈ 6 s) | `.venv/bin/python scripts/agents/exactness_239.py [--validation-multi]` |
| Carry backtest / research report | `.venv/bin/python -m vibe_quant.portfolio run --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT --start 2022-06-01 --end 2026-09-28 [--engine nt\|reference]` / `... evaluate --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT --start 2022-06-01 --end 2026-09-28 [--draws 200] [--seed 0] [--trials PATH]` (`evaluate` has no `--engine`; ≈ 7.5 min at `--draws 200`, ≈ 1.1 s/draw, dominated by metrics) |
| Carry exactness proof (≈ 4.5 s) | `.venv/bin/python scripts/agents/exactness_carry.py` |
| Pre-review gate (PASS cached per HEAD; runs the unit suite in parallel) | `scripts/agents/swarm-check.sh <worktree> --scope '<glob>,<glob>' [--no-cache]` (one comma-separated arg) |
| Backend | `.venv/bin/uvicorn "vibe_quant.api.app:create_app" --factory --port 8000` |
| Frontend dev / build / unit | `cd frontend && pnpm dev` / `pnpm build` / `pnpm vitest run` |
| E2E (against ui-check stack) | `scripts/agents/e2e.sh <worktree>` (boots the stack, runs all specs, always tears down; `... <worktree> e2e/<file>` to filter) |
| Extraction worker | `.venv/bin/vibe-quant extraction-worker` (run beside the backend so re-extractions progress) |
| Throwaway UI stack | `scripts/agents/ui-check.sh up\|down <worktree>` (backend :8001 + Vite :5188 on a DB copy) |
| Worktree for a task | `scripts/agents/worktree.sh <slug> [base] [--private-node-modules]` |
| Rebuild catalog | `.venv/bin/python -m vibe_quant.data rebuild --from-archive` |

**Worktrees import the main checkout's code** (the editable install points at it): run worktree code as `PYTHONPATH=$PWD .venv/bin/python -m ...`. Create worktrees with `scripts/agents/worktree.sh` — it links `data/catalog` + `data/archive` (a bare `git worktree add` lacks them and 8 research tests fail). `pnpm install` in a worktree writes through the shared `node_modules` symlink; pass `--private-node-modules` when deps change.

## Agent speed rules

Measured costs: full `tests/unit` ≈ 3 min serial, ≈ 1 min under `-n auto --dist worksteal` (the gate uses this: `-m "not timing"` in parallel, `-m timing` serially) · `swarm-check.sh` ≈ 1 min when it runs the full suite, seconds on a cached PASS · `exactness_239.py` ≈ 6 s · Playwright spec ≈ 1.5 min + `ui-check.sh` up/down · one model turn ≈ 4–10 s.
- **Gate once per round:** never run `pytest tests/unit` yourself; `swarm-check.sh` runs it as your last command, after targeted tests + ruff + mypy are green. On FAIL re-run only the failing test, then the gate once more; no gate after a rebase. Full-suite/gate time stayed ~21 min per job across three jobs (one maker ran it 7×).
- **Fewer, fatter turns:** first turn reads every needed file in parallel; afterwards send all independent reads/greps/test runs in one turn, read whole functions, chain shell checks with `&&`, track work in beads instead of todo lists. Target ≥ 2 tool calls per turn — jobs 20261010-backlog / 20261011-testspeed / 20261011-grind averaged 1.11 / 1.17 / 1.10 (~90% single-tool turns).
- **Mutation checks are capped:** ≤ 5 mutants (~3 min) per review, ≤ 3 per maker self-check, only on changed logic, each against the targeted test file in a `git archive` scratch copy (`cp -r` shares the worktree's git index). Stress loops ≤ 10 runs unless flakiness is the finding. A surviving mutant outside money code (screening/validation/discovery/overfitting/dsl/risk/paper/data, fills, metrics, DB writes, concurrency) becomes a follow-up bead, not a fix round.
- **Gate cache:** `swarm-check.sh` caches a PASS per HEAD + base + scope + tool hash (clean tree only); re-running it on an unchanged worktree is instant, `--no-cache` forces a full run.
- **E2E:** iterate with `-g <scenario>`, run the full spec once at the end.
- **Async launches:** a test of anything that returns 201 and then runs in a subprocess polls the run's final status.
- **Measure per-test time serially:** `--durations` under `-n auto` is inflated by CPU contention (job 20261016-followups3: 58.7 s and 7.8 s parallel-summed figures were 23.3 s and 2.98 s serially; two speed beads chased nothing).
- **Keep proofs short:** never put long discovery/backtest reruns in a task brief; use unit tests, mutants and `exactness_239.py`, and tiny live settings (pop 4–6, 1–2 gens, 2–3 month window) when a live check is unavoidable.

## Shell

- Search with `rg` (ripgrep); `rg -g '*.tsx'` (there is no `--type tsx`).
- Quote globs and paren paths (`--include="*.ts"`, `'frontend/src/routes/(app)/…'`) — unquoted, zsh aborts with "no matches found".
- Keep exit codes visible: `cmd >/tmp/log 2>&1; echo "EXIT=$?"` (zsh uses lowercase `$pipestatus`). Trust the `ruff`/`mypy`/`pytest` exit code over editor diagnostics mid-refactor.
- zsh: `status` is read-only (use `st`/`run_status`); unquoted variables don't word-split (`$CMD args` fails — use `bash -c` or a function for loops over `set -- $row`).
- Wait on conditions (Monitor, `run_in_background`, `until` loops), not fixed `sleep`.

## SQLite (state DB)

`data/state/vibe_quant.db`; override with `VIBE_QUANT_DB` at backend start (every job subprocess inherits it; there is no runtime DB switch). Every connection sets `PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;` (`StateManager` does this).

- **Opening the DB runs schema migrations.** `tests/conftest.py` pins `VIBE_QUANT_DB` to a temp file; keep that guard. Scratch scripts against the real DB use `sqlite3.connect('file:data/state/vibe_quant.db?mode=ro', uri=True)`.
- Use `?` placeholders for every value; handle `None` before formatting; `conn.commit()` after writes; `row_factory = sqlite3.Row` for dict access.
- Discovery runs live in `backtest_runs` with `run_mode='discovery'` (there is no `discovery_runs` table). `backtest_results` has ONE row per run (re-runs replace result + trades); its `notes` column is machine JSON (discovery payload, `data_window`, funding, consistency, `dsr_trials`) and user text lives in `user_notes`.
- Key tables: `backtest_runs` (all run modes), `strategies`, `backtest_results`, `sweep_results` (screening/replay metrics), `background_jobs`, `trades`, `research_extraction_jobs`, `download_sessions` (status `completed|partial|failed`).
- `StateManager` shares one connection across threads: reads used from worker/API threads take `_write_lock` (an unlocked read caused `sqlite3.InterfaceError`; bead `vibe-quant-jfxsr` covers the rest).

## Architecture

```
Strategy DSL (YAML) → Screening (NT simplified, parallel) → Overfitting gates → Validation (NT full fidelity) → Paper → Live
```

One engine (NautilusTrader), two modes:
- **Screening** (also used by discovery): simplified fills, multiprocessing; models leverage + funding (archive; holes charged a flagged fallback rate); starting balance = `parameters.initial_balance` (default 1000) on screening launches (discovery stays 1000). Liquidation is an opt-in per-run overlay (OFF by default; `off|isolated|cross`, leverage a whole number in [1,125], default 10); the screening venue runs at `parameters.leverage` in every mode and, when the overlay is on, every metric is recomputed from the truncated ledgers. Each strategy queues orders in a per-instrument outbox released on its own two synthetic 1m fill ticks per bar; open positions are marked at run end; drawdown is marked on strategy-timeframe bars. Position size per trade = min(risk_per_trade / stop, max_position_pct cap per symbol) of equity; both run-level, defaults 0.02 / 0.5 ([SPEC.md § Position sizing](SPEC.md#position-sizing)).
- **Validation**: custom FillModel; orders release on the instrument's own next 1m detail bar (the latency preset only selects the fill model on bar data); full cost modeling. Clamps the window to 1m coverage (`notes.data_window`) and fails without 1m data.
- **Liquidation** (opt-in, OFF by default): a per-run overlay for screening, validation and discovery; `liquidation_mode` `off|isolated|cross` with leverage a whole number in [1,125] (default 10; shared rule `vibe_quant/liquidation_settings.py`). Screening/validation set the top-level `liquidation_mode` on the launch (API/UI; the launch form has a **Liquidation** Off/Isolated/Cross select that reuses the existing **Leverage** input; leverage = `parameters.leverage`, the screening venue's leverage in every mode); discovery uses `--liquidation-mode isolated --leverage 20` (deprecated `--liquidation-leverage` alias), or API `liquidation_mode`+`leverage` (deprecated `liquidation_leverage` alias; a conflict is a 422; its launch form has an always-visible **Leverage (1-125)** input); promote/replay inherit the parent; a mode-on run's results show a liquidation chip and counts. Needs MARK 1m klines (`python -m vibe_quant.data download-mark --start YYYY-MM-DD`). Details: [SPEC.md § How to turn liquidation on](SPEC.md#how-to-turn-liquidation-on).

Key paths: DB schema `vibe_quant/db/schema.py` · DSL types `vibe_quant/dsl/schema.py` (frontend mirror `frontend/src/components/strategies/editor/types.ts`) · theme `frontend/src/index.css` · runtime data `data/{archive,catalog,state}` (gitignored) · per-run logs `logs/{discovery|validation|screening}_<run_id>_*.log`.

## Conventions

- **License:** MIT project. NautilusTrader (LGPL-3.0) is an unmodified library dependency — never modify, vendor or copy NT source; keep custom code in `vibe_quant/`. Avoid AGPL dependencies. NT is pinned to one minor (`pyproject.toml`); test before bumping.
- **Indicators:** prefer NautilusTrader built-ins (Rust); fall back to `pandas-ta-classic` — never the original `pandas-ta` (compromised maintainership). Custom indicators are plugins in `vibe_quant/dsl/plugins/`.
- **Data:** raw downloads are archived to SQLite (`data/archive/`, append-only) before processing into the ParquetDataCatalog; the catalog is rebuildable from the archive.
- **Frontend API hooks are generated by orval** (`frontend/src/api/generated/`): regenerate by dumping `openapi.json` (`PYTHONPATH=$PWD .venv/bin/python -c "import json; from vibe_quant.api.app import create_app; f=open('frontend/openapi.json','w'); json.dump(create_app().openapi(), f, indent=2); f.write('\n')"`; the trailing newline matters) + `cd frontend && pnpm generate-api`, never hand-edit.

## Verification rules

Valid proofs that a change preserved correctness:
- **Fixed-strategy eval** (`scripts/agents/exactness_239.py`): strategy 239, BTCUSDT 2024-01-01..2026-03-17, screening must return sharpe `0.7974716442710239`, 67 trades, return `0.159784359286041`, PF `1.6359612484989312`, max DD `0.05716872229767501` (all five asserted).
- **Validation repeatability:** single-symbol 239 = sharpe `0.780452281957465`, 67 trades. Two-symbol (BTCUSDT+ETHUSDT, `--validation-multi`) = sharpe `0.4956474374166028`, 164 trades, return `0.13753127107369148`, PF `1.1990127778814956`, max DD `0.11707174708945581`, symbol-order-invariant. Any drift is a regression unless the change is a declared comparability break.
- **Carry exactness** (`scripts/agents/exactness_carry.py`, ≈ 4.5 s): the NT funding-carry engine must equal the no-NT reference **exactly** (`compare_runs` over every dataclass field) on BTCUSDT/ETHUSDT/SOLUSDT/BNBUSDT 2022-06-01..2026-09-28; the pinned headline numbers live in the script. Change the shared rule modules (`vibe_quant/portfolio/` signal/sizing/ledger/ticks), never one executor. swarm-check runs main's copy when `vibe_quant/portfolio/`, `vibe_quant/validation/funding.py`, `exactness_carry.py`, `pyproject.toml` or `uv.lock` change (SKIP while main has no copy).
- **Within-run replay:** discovery champion → `/replay` matches exactly when the run used `eval_windows=1`.
- Zero-tolerance unit tests against the reference implementation for ported math.

Not proofs (they wasted time): comparing two discovery runs (nondeterministic unless seeded, and any code change shifts the RNG sequence); comparing an `eval_windows>1` champion's stored fitness to a full-window replay; comparing metrics across a comparability break (dated list in [`docs/claude/discovery.md`](docs/claude/discovery.md#comparability-breaks)).

## Issue tracking (beads)

Use `bd` for all task tracking (not TodoWrite, TaskCreate or markdown task lists). Reference: [`docs/claude/beads.md`](docs/claude/beads.md).

```bash
bd ready                                    # available work
bd show <id> / bd update <id> --claim / bd close <id> [<id>...]
bd create --title="..." --description="..." --type=bug --priority=2   # priority 0-4
bd remember "fact" --key category:item       # persistent project memory (bd recall / bd memories <kw>)
```

- `bd edit` opens `$EDITOR` and blocks agents; use `bd update` flags.
- `bd prime` (auto-run by hooks) injects all memories each session. If a memory contradicts this file, this file wins — update or `bd forget` the memory.
- **Session completion:** file beads for follow-ups → quality gates → close beads → `git pull --rebase && git push` → `git status` shows "up to date with origin". Work isn't done until the push succeeds.

## Commit messages

One commit per task (bead or tightly coupled pair), never one squash commit per job, with a verbose body — **Why / What / How / Decisions / Verification / Review** — in the format of [`docs/orchestration/prompts/commit-message.md`](docs/orchestration/prompts/commit-message.md). `git log` is the project history and a context source for future agents, so it records why and who decided, not just bead ids. This repo overrides any global "concise commit message" preference.

## Multi-agent orchestration

Large jobs use the `orchestrate` skill + [`docs/orchestration/`](docs/orchestration/README.md). Model routing (user 2026-10-16): DeepSeek V4.1 Flash (effort high) is the only Command Code coder (MiMo disabled: expensive, slow); difficult tasks escalate to a Sonnet implementer; GPT-6.1-Sol (Codex, via T3 `delegate_task`, effort high) reviews; Sonnet 5.5 / Haiku 5.5 implement and fix; Opus at effort medium plans and architects. Command Code workers (`deepseek/deepseek-v4.1-flash` effort high only; never MiMo/Kimi/GLM/Qwen) run only as T3 subagents via the `t3-code` MCP `delegate_task` (instance `driverKind: commandcode`, async, unique `clientRequestId`) — see `docs/orchestration/cheap-agents.md` § Dispatch protocol. Children start in the main checkout with no cwd: every brief names the absolute worktree and requires `cd <worktree> &&` on each command.

**Self-improvement is top priority (user 2026-10-13):** every agent handoff ends with `improve: <change + evidence> | none`. At every Land step, and whenever a task reaches its 3rd review round, the chief sweeps every `.claude/agents/*.md`, the orchestrate skill and `docs/orchestration/prompts/`. It applies evidence-backed improvements without asking, commits them separately (`self-improve: ...`), and reports them. Gates, verification rules and hard rules still need the user to weaken them.

**Swarm board** — shared memory and chat for every agent (`scripts/agents/bus.py`; human view `docs/orchestration/board/BOARD.md`): catch up with `python3 scripts/agents/bus.py --as <you> board search --global --path <files you will touch> --topic gotchas findings decisions` (plus `board read --global --topic gotchas --recent 15`); record with `board post --topic findings|gotchas|decisions|model-notes|thoughts --body "..."`; ask the chief with `ask --to chief`, message peers with `post --to <agent>`, answer with `reply --ref <id>`.

## Historical documentation

`docs/opus-prd.md`, `docs/opus-spec.md`, `docs/opus-research.md`, `docs/gpt-research.md` and `docs/crypto-trading-bot-specification.md` predate SPEC.md and describe abandoned architectures (FreqTrade, VectorBT, PostgreSQL, Redis). Read them only for history; follow SPEC.md.
