# Task: Image pipeline end-to-end (beads ticket vibe-quant-7go7m.5)

Read AGENTS.md first — all conventions apply (uv, zero-baseline ruff+mypy, pytest, SQLite WAL + `?` placeholders, data archive is rebuildable, never edit generated frontend code). Do NOT commit — leave the tree dirty for review. Write a short `codex-report.md` at the end.

## What to build

Reddit strategy posts often carry the real content in screenshots (rule tables, code, TradingView setups, equity curves). Today those arrive with an empty/thin body and get skipped. Make the scraper **download + archive** post/gallery/preview images, and make the LLM extractor **read them** so an image-only post yields a real extraction (or an image-informed skip). Vision also feeds the existing credibility score (equity-curve = evidence).

## CRITICAL pre-proven fact (do not re-litigate)

The extractor shells out to the `claude` CLI (`vibe_quant/research/extractor.py` `_run_claude`: `claude -p --output-format json [--model X] <prompt>`). I already spiked that **`claude -p` reads local images when you (a) reference the absolute image path in the prompt and (b) pass `--allowedTools Read`**. Verified: it OCR'd a strategy screenshot exactly (`Entry: RSI < 30 AND EMA(9) > EMA(21)` ...). So the vision path is: archive images to disk → put their absolute paths in the prompt → add `--allowedTools Read` to argv. NO new SDK, NO tesseract, NO API keys.

## Implementation

1. **Reddit source captures image URLs** (`vibe_quant/research/sources/reddit.py`, `_to_raw_item`): pull image URLs from the post JSON into `extras["image_urls"]` (a list). Sources to cover: `data.url` when it ends in .jpg/.jpeg/.png/.webp or host is i.redd.it; `data.preview.images[].source.url` (HTML-unescape `&amp;`); gallery posts via `data.media_metadata` (each `.s.u` or `.s.gif`). Cap at a small constant (e.g. `MAX_IMAGES_PER_POST = 4`). Top-level post only. Leave `image_urls` absent/empty when none.

2. **Image archiver** (new helper, e.g. `vibe_quant/research/image_archive.py`): given an item's external_id + `image_urls`, download each to `data/archive/research_images/<source>/<external_id>/<idx>.<ext>` with: per-file size cap (e.g. 5 MB, stream + abort if exceeded), total count cap, timeout, and skip-on-failure (one bad URL must not fail the item). Return the list of local absolute paths. Reuse httpx; if a redd.it CDN 403s on plain httpx, retry with a browser User-Agent (mirror the challenge client's UA constant). Store the resulting paths in `extras["image_paths"]`. Downloads happen at scrape time so the archive stays self-contained/rebuildable — wire it into the scrape/archive path (`pipeline.py archive_item` or the source), your call, but persisted `extras_json` must end up with `image_paths`.

3. **Extractor reads images** (`extractor.py`):
   - `_format_user_content` / `_build_prompt`: when `item.extras["image_paths"]` is non-empty, add a section listing the absolute paths and instruct the model to Read each and incorporate rule tables / code / equity-curve evidence.
   - `_build_system_prompt`: tell the model images may contain the actual strategy rules and that an equity-curve/PnL screenshot counts toward `evidence_level`/`completeness` (ties into the credibility fields already shipped).
   - `_run_claude`: when the item has image paths, append `--allowedTools Read` to argv (only then — keep the no-image path unchanged so its behavior/cost is identical). You'll need `_run_claude` (or the builder) to know whether images are present; thread that through cleanly.
   - Bound total images sent (reuse the cap) so a gallery can't blow the timeout.

4. **Tests** (mirror `tests/unit/research/test_reddit_source.py` and `test_extractor.py` style — fully mocked, no network, no real `claude`):
   - reddit source extracts image_urls from preview/gallery/direct-link post JSON, and caps count.
   - image_archive: downloads via a mocked httpx transport, enforces size + count caps, skips a failing URL, returns paths; (use tmp_path).
   - extractor: with image_paths present, the built prompt references the paths AND argv includes `--allowedTools Read`; without images, argv is unchanged (no Read tool).

## Acceptance criteria (ticket)

- [ ] Images archived with size+count caps; rebuildable per data conventions
- [ ] Extractor references image paths + enables Read only when images present
- [ ] >=1 image-bearing post demonstrably extracted using image content (leave a note in codex-report.md on how to demo; do NOT make live claude/network calls in tests)
- [ ] pytest + ruff + mypy green (zero-baseline)

## Constraints

- Do NOT call the real `claude` CLI or hit the network in tests.
- Keep the no-image extraction path byte-for-byte unchanged (same argv, same cost) — this is a regression risk; the credibility-score tests must still pass.
- `data/archive/research_images/` is runtime data (gitignored) — do not commit fixtures there.
