# Credibility score end-to-end

Implemented bead `vibe-quant-7go7m.4` without committing.

## Changes

- Added `evidence_level` and `completeness` to extraction domain results, the LLM JSON contract, defensive response parsing, persistence, API responses, and the generated Orval model.
- Added schema v14 migration columns with enum/range checks. The migration test starts from a legacy table, initializes twice, and verifies invalid values are rejected.
- Added `sort=credibility` for research items. It ranks each item's best finding by live-traded, backtested, then idea-only evidence, with completeness descending inside each evidence tier and unscored items last.
- Added credibility labels and completeness percentages to `ExtractionCard`, plus a Highest credibility option in the research list.
- Added extractor, migration/persistence, API response, and API ordering coverage.

Primary files touched: `vibe_quant/research/{schema,extractor,pipeline}.py`, `vibe_quant/db/{schema,state_manager}.py`, `vibe_quant/api/{schemas,routers}/research.py`, research UI/store files, `frontend/openapi.json`, generated API models, and the corresponding research tests.

## Verification

- `uv run pytest -q`: 2315 passed, 4 skipped, 13 warnings
- `uv run ruff check`: clean
- `uv run mypy vibe_quant`: clean (160 source files)
- Research UI Biome check: clean for all three touched UI files
- `cd frontend && pnpm typecheck`: clean
- `cd frontend && pnpm build`: successful

The repository-wide frontend Biome check remains blocked by unrelated pre-existing findings outside this ticket; no unrelated files were changed to address them.

## Not run

The existing 50-item corpus was not re-extracted because the task explicitly prohibited live LLM calls. No scraping was run. Existing items can be re-extracted with the extraction worker after review to populate the new scores without fetching new posts.

Orval regeneration also synchronized the already-present backend `ReplayResponse.metrics_note` field into its generated model; that incidental generated change is unrelated to credibility scoring.
