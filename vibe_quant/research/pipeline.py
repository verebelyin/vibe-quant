"""Orchestrator: source.fetch → archive → (extract).

Designed to be invoked from the CLI as a long-lived subprocess. Maintains a
heartbeat row in `research_scrape_runs` so the API can detect crashed
runners.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vibe_quant.research.archive import archive_item
from vibe_quant.research.sources import get_source, load_builtin_sources

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.research.schema import ExtractionResult, RawItem

    ExtractFn = Callable[[RawItem, int], ExtractionResult]

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_SECONDS = 30


@dataclass
class ScrapeSummary:
    scrape_run_id: int
    items_fetched: int
    items_new: int
    items_extracted: int
    items_failed: int
    items_skipped: int
    status: str
    error_message: str | None


def _build_source(source_name: str) -> object:
    load_builtin_sources()
    cls = get_source(source_name)
    return cls()  # each source supplies its own env-driven defaults


def run_scrape(
    *,
    sm: StateManager,
    source_name: str,
    limit: int,
    extract_fn: ExtractFn | None = None,
    db_path: Path | None = None,  # noqa: ARG001  (kept for symmetry with workers)
) -> ScrapeSummary:
    """Run one scrape pass and return a summary.

    Creates the `research_scrape_runs` row, drives the source, archives every
    `RawItem`, optionally invokes `extract_fn` per archived item, and finalizes
    the run row.

    Args:
        sm: A `StateManager` already pointed at the right DB.
        source_name: Registered source name (e.g., `"reddit"`).
        limit: Per-source cap on items to scrape.
        extract_fn: Optional callable `(item: RawItem, item_id: int) -> ExtractionResult`.
            If `None`, items land at `extraction_status='pending'`.
    """
    # Build the source up-front so unknown name / missing creds / config
    # errors don't leave an orphan scrape_run row in the DB.
    source = _build_source(source_name)

    pid = os.getpid()
    scrape_run_id = sm.create_scrape_run(
        source=source_name,
        pid=pid,
        config={"source": source_name, "limit": limit, "extract": extract_fn is not None},
    )

    stop_event = threading.Event()

    def heartbeat_loop() -> None:
        while not stop_event.is_set():
            with contextlib.suppress(Exception):
                sm.update_scrape_run_heartbeat(scrape_run_id)
            stop_event.wait(HEARTBEAT_INTERVAL_SECONDS)

    hb_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    hb_thread.start()

    # SIGTERM → finalize as killed and let the iterator unwind
    kill_flag = threading.Event()
    prev_handlers: dict[signal.Signals, object] = {}

    def _signal_handler(signum: int, _frame: object) -> None:  # noqa: ARG001
        logger.warning("scrape_run %s received signal %s; finalizing as killed", scrape_run_id, signum)
        kill_flag.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(ValueError):
            prev_handlers[sig] = signal.signal(sig, _signal_handler)

    fetched = new = extracted = failed = skipped = 0
    error_message: str | None = None
    final_status = "completed"

    try:
        for item in source.fetch(since=None, limit=limit):  # type: ignore[attr-defined]
            if kill_flag.is_set():
                final_status = "killed"
                break
            fetched += 1
            try:
                was_new, item_id = archive_item(sm, item)
            except Exception:  # noqa: BLE001
                logger.exception("archive failed for %s/%s", item.source, item.external_id)
                failed += 1
                sm.increment_scrape_run_counters(scrape_run_id, fetched=1, failed=1)
                continue
            sm.increment_scrape_run_counters(
                scrape_run_id, fetched=1, new=1 if was_new else 0
            )
            if was_new:
                new += 1
                if extract_fn is not None and item_id is not None:
                    try:
                        result: ExtractionResult = extract_fn(item, item_id)
                    except Exception:  # noqa: BLE001
                        logger.exception("extractor failed for item_id=%s", item_id)
                        failed += 1
                        sm.update_research_item_status(item_id, "failed")
                        sm.increment_scrape_run_counters(scrape_run_id, failed=1)
                        continue
                    _persist_extraction(sm, item_id, result)
                    if result.status == "parsed":
                        extracted += 1
                        sm.increment_scrape_run_counters(scrape_run_id, extracted=1)
                    elif result.status == "skipped":
                        skipped += 1
                    else:  # failed
                        failed += 1
                        sm.increment_scrape_run_counters(scrape_run_id, failed=1)
    except KeyboardInterrupt:
        final_status = "killed"
    except Exception as e:  # noqa: BLE001
        logger.exception("scrape_run %s failed", scrape_run_id)
        final_status = "failed"
        error_message = f"{type(e).__name__}: {e}"
    finally:
        stop_event.set()
        hb_thread.join(timeout=HEARTBEAT_INTERVAL_SECONDS + 1)
        # restore signal handlers
        for sig, prev in prev_handlers.items():
            with contextlib.suppress(ValueError, TypeError):
                signal.signal(sig, prev)  # type: ignore[arg-type]
        sm.complete_scrape_run(
            scrape_run_id,
            status=final_status,
            error_message=error_message,
        )

    return ScrapeSummary(
        scrape_run_id=scrape_run_id,
        items_fetched=fetched,
        items_new=new,
        items_extracted=extracted,
        items_failed=failed,
        items_skipped=skipped,
        status=final_status,
        error_message=error_message,
    )


def _persist_extraction(sm: StateManager, item_id: int, result: ExtractionResult) -> None:
    """Write an extraction row + mirror status onto the research_item."""
    sm.create_extraction(
        research_item_id=item_id,
        status=result.status,
        llm_model=result.llm_model,
        confidence=result.confidence,
        rationale=result.rationale,
        raw_response=result.raw_response,
        dsl_yaml=result.dsl_yaml,
        parsed_dsl_json=result.parsed_dsl_json,
        parse_error=result.parse_error,
    )
    item_status_map = {
        "parsed": "extracted",
        "failed": "failed",
        "skipped": "skipped",
    }
    sm.update_research_item_status(item_id, item_status_map.get(result.status, "failed"))
