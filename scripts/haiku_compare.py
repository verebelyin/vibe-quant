"""One-off: re-extract run-12 items with Haiku 4.5 and write parallel logs.

A/B test against the existing Opus 4.7 logs at data/research/logs/12/.
Does NOT touch research_extractions / research_items — strictly logs.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

from vibe_quant.research import extraction_log
from vibe_quant.research.archive import row_to_raw_item
from vibe_quant.research.extraction_log import write_extraction_log
from vibe_quant.research.extractor import ClaudePExtractor, extractor_version

HAIKU_MODEL = "claude-haiku-4-5-20251001"
RUN_ID = 12
LOG_ROOT = Path("data/research/logs")


def main() -> int:
    conn = sqlite3.connect("data/state/vibe_quant.db")
    conn.row_factory = sqlite3.Row

    items = conn.execute(
        """
        SELECT i.* FROM research_items i
        JOIN research_extractions e ON e.research_item_id = i.id
        WHERE e.id IN (
            SELECT MIN(id) FROM research_extractions
            WHERE extracted_at >= (SELECT started_at FROM research_scrape_runs WHERE id = ?)
            GROUP BY research_item_id
        )
        ORDER BY i.id
        """,
        (RUN_ID,),
    ).fetchall()
    print(f"items in run {RUN_ID}: {len(items)}")

    log_dir = LOG_ROOT / f"{RUN_ID}-haiku"
    log_dir.mkdir(parents=True, exist_ok=True)

    extractor = ClaudePExtractor(model=HAIKU_MODEL)
    version = f"{extractor_version()}#{HAIKU_MODEL}"

    t0 = time.time()
    parsed = skipped = failed = 0
    for n, row in enumerate(items, 1):
        rid = int(row["id"])
        raw_item = row_to_raw_item(dict(row))
        item_t0 = time.time()
        batch = extractor.extract_all(raw_item)
        elapsed = time.time() - item_t0

        write_extraction_log(
            log_dir=log_dir,
            item_id=rid,
            batch=batch,
            extractor_version=version,
            scrape_run_id=RUN_ID,
        )
        statuses = [r.status for r in batch.results]
        for s in statuses:
            if s == "parsed":
                parsed += 1
            elif s == "failed":
                failed += 1
            else:
                skipped += 1
        print(
            f"  [{n:>2}/{len(items)}] item#{rid} {elapsed:>5.1f}s findings={len(statuses)} "
            f"statuses={','.join(statuses)}"
        )
        sys.stdout.flush()

    total = time.time() - t0
    print()
    print(f"=== haiku run done in {total:.1f}s ===")
    print(f"  parsed:  {parsed}")
    print(f"  skipped: {skipped}")
    print(f"  failed:  {failed}")
    print(f"  logs:    {log_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
