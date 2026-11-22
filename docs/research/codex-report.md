# Image pipeline — implementation report (vibe-quant-7go7m.5)

End-to-end image pipeline for Reddit research: scrape captures post/gallery/preview
image URLs, the scraper archives them to disk, and the `claude -p` extractor reads
them via `--allowedTools Read` so image-only posts yield real extractions.

## Files touched

- `vibe_quant/research/sources/reddit.py` — `_extract_image_urls()` pulls image URLs
  from three shapes into `extras["image_urls"]` (top-level post only, HTML-unescaped,
  order-preserving, de-duplicated, capped at `MAX_IMAGES_PER_POST=4`):
  - direct `data.url` when it ends in `.jpg/.jpeg/.png/.webp` or the host is `i.redd.it`
  - `data.preview.images[].source.url`
  - gallery `data.media_metadata[*].s.u` / `.s.gif`
  Key absent/empty when no images (snapshot-parity test unaffected).
- `vibe_quant/research/image_archive.py` (NEW) — downloader:
  - `download_images(...)` streams each URL to
    `data/archive/research_images/<source>/<external_id>/<idx>.<ext>`
  - per-file 5 MB cap (`MAX_IMAGE_BYTES`, aborts mid-stream, drops the partial),
    count cap (`MAX_IMAGES=4`), 20 s timeout, skip-on-failure (one bad URL never
    sinks the item), contiguous indices across skips.
  - browser-UA retry (`BROWSER_USER_AGENT` from the challenge client) on a redd.it
    CDN 403; non-redd.it 403s are not retried.
  - extension resolved from URL suffix, else `Content-Type`, else `.jpg`.
  - `archive_item_images(item, ...)` mutates `item.extras["image_paths"]` in place.
- `vibe_quant/research/pipeline.py` — `run_scrape` calls `archive_item_images(item)`
  at scrape time, before `archive_item(...)`, so the persisted `extras_json` carries
  local `image_paths`. Best-effort (exception-guarded + logged).
- `vibe_quant/research/extractor.py`:
  - `_image_paths(item)` — validated, capped (`MAX_PROMPT_IMAGES=4`) local paths.
  - `_build_prompt` appends `_build_image_section(...)` **only when images present**,
    as a trusted instruction block AFTER the `<<<END>>>` delimiter (image "Read
    these" directive is an instruction, image text stays untrusted data).
  - `_is_empty_input` returns False when image paths exist, so an empty-title/body
    image-only post is no longer short-circuited to `skipped`.
  - `_run_claude(prompt, *, has_images=False)` appends `--allowedTools Read` **only**
    when `has_images`. The no-image argv/prompt/cost is byte-for-byte unchanged.

## Test results (all green, zero-baseline)

- `uv run pytest tests/unit/research/ -q` → **242 passed**
- `uv run pytest -q` → **2343 passed, 4 skipped** (warnings are pre-existing Pandas4)
- `uv run ruff check` → **All checks passed!**
- `uv run mypy vibe_quant` → **Success: no issues found in 162 source files**

New tests:
- `test_reddit_source.py`: direct/preview/gallery URL capture, HTML-unescape, dedupe,
  count cap, non-image link ignored, no-key-when-none.
- `test_image_archive.py` (NEW): download to per-item dir, content-type extension
  fallback, size cap aborts + drops file, count cap, skip-failure with contiguous
  indices, redd.it 403 → browser-UA retry, non-reddit 403 not retried, extras mutation.
- `test_extractor.py`: prompt references image paths; **no-image prompt byte-identical**;
  argv includes `--allowedTools Read` only with images (no-image argv asserted exact);
  image-only post not short-circuited as empty.

## How to demo an image-bearing extraction manually (no live calls in tests)

1. Ensure `claude` CLI is on PATH.
2. Archive an image so it lands under the archive dir, e.g. from a Python shell:
   ```python
   from vibe_quant.research.image_archive import download_images
   paths = download_images(source="reddit", external_id="demo",
                           image_urls=["<url-to-a-strategy-screenshot>"])
   print(paths)  # e.g. ['/…/data/archive/research_images/reddit/demo/0.png']
   ```
3. Build a `RawItem` with `extras={"image_paths": paths, "comments": []}` (title/body
   may be empty) and run the extractor:
   ```python
   from vibe_quant.research.extractor import get_default_extractor
   from vibe_quant.research.schema import RawItem
   item = RawItem(source="reddit", external_id="demo", url="…", title="", body="",
                  author=None, posted_at=None, score=None,
                  extras={"image_paths": paths, "comments": []})
   print(get_default_extractor().extract_all(item).results)
   ```
   The extractor argv gains `--allowedTools Read` and the prompt lists the absolute
   image path; `claude -p` reads the screenshot and OCRs the rules into a finding.
4. Full loop: run a scrape against a subreddit with image posts — images archive at
   scrape time and re-extraction reloads `image_paths` from `extras_json` automatically
   (via `row_to_raw_item`).

## Notes / unfinished

- No live `claude` or network calls in any test (httpx `MockTransport` + `tmp_path`;
  `subprocess.run` patched). Tree left dirty, uncommitted, per instructions.
- `data/archive/research_images/` is gitignored runtime data — nothing committed there.
- Frontend was not touched (no UI surfacing of archived images in this scope).
