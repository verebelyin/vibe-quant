# Task: Credibility score end-to-end (beads ticket vibe-quant-7go7m.4)

Read AGENTS.md first — all project conventions apply (uv, zero-baseline ruff+mypy, pytest, SQLite WAL + `?` placeholders, never edit `frontend/src/api/generated/`).

## What to build

The LLM research extractor must emit a **credibility score** for every finding — parsed AND skipped — persisted to the DB, exposed via the API, and sortable/visible in the research UI. Demoable on the existing 50-item corpus (no new scraping needed).

Two new fields per extraction:

- `evidence_level`: enum `live_traded | backtested | idea_only` — did the post's author actually trade it (PnL screenshots, broker statements, "ran this for 6 months"), only backtest it, or just describe an idea?
- `completeness`: float 0..1 — how implementable are the rules as stated (exact entries/exits/params = high; vague vibes = low)?

## Why

User wants "real use cases" privileged: evidence-backed, complete posts jump the triage queue. This score will rank a soon-to-be-backfilled corpus of hundreds of items.

## Touch points (explore before editing)

1. `vibe_quant/research/schema.py` — `ExtractionResult` dataclass: add both fields.
2. `vibe_quant/research/extractor.py` — extend `_build_system_prompt()` JSON contract to require both fields on every finding (parsed and skipped); parse them in `_finding_to_result` / `_single` with defensive coercion (missing/invalid → None, never crash).
3. `vibe_quant/db/schema.py` — add columns to `research_extractions` via `_migrate_add_columns` list; bump `SCHEMA_VERSION`. Follow the existing idempotent ALTER pattern.
4. Persistence layer (find where `research_extractions` rows are INSERTed — likely `vibe_quant/db/state_manager.py`) — write both fields.
5. `vibe_quant/api/schemas/research.py` + `vibe_quant/api/routers/research.py` — expose both fields in extraction responses; add a sort option so items can be ordered by credibility (evidence_level rank desc, then completeness desc). Follow existing sort/filter param patterns in that router.
6. Frontend: regenerate the orval client (start backend or use the documented flow: dump openapi.json + `cd frontend && pnpm generate-api`), then show both fields in `frontend/src/components/research/ExtractionCard.tsx` and make the research list sortable by credibility (see `frontend/src/routes/research.tsx`). TypeScript: never use `any`; prefer `unknown` if a top type is unavoidable.

## Acceptance criteria (from the ticket)

- [ ] Schema migration adds both fields (idempotent, SCHEMA_VERSION bumped)
- [ ] Extractor prompt + JSON contract emits them on every parsed AND skipped finding
- [ ] API returns fields; results sortable by credibility
- [ ] UI displays them; list sortable
- [ ] Tests green

## Approach

- TDD at the seams: write failing unit tests first for (a) extractor response parsing with/without the new fields (defensive: absent, wrong type, out-of-range clamped or None), (b) migration idempotency (run init_schema twice on a temp DB), (c) API response includes fields + sort order correct. Look at existing tests under `tests/` for research/extractor patterns and mirror them.
- Run single test files as you go; full `pytest` + `ruff check` + `mypy` + `cd frontend && pnpm build` at the end — all must be clean (zero-baseline).
- Do NOT commit — leave the working tree dirty for review.
- Do NOT run live LLM calls or scrapes; unit tests mock the LLM boundary like existing tests do.
- Do NOT modify `frontend/src/api/generated/` by hand.

## Deliverable

Working tree with all changes + a short summary (files touched, test results, anything you couldn't finish) written to `codex-report.md` in the repo root.
