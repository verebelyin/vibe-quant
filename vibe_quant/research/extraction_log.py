"""Per-extraction log files: prompt + raw response + finding summaries.

Each LLM call writes one JSON file so future prompt-engineering work can
replay or analyze the corpus verbatim, even after the prompt template
changes. Files are atomic (write-tmp + rename) so a crashed extractor
never leaves a half-written log.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibe_quant.research.schema import ExtractionBatch

logger = logging.getLogger(__name__)

DEFAULT_LOG_ROOT = Path("data/research/logs")


def _resolve_root(root: Path | None) -> Path:
    # Resolved per-call so tests (and any future config) can monkeypatch
    # ``DEFAULT_LOG_ROOT`` without re-importing this module.
    return root if root is not None else DEFAULT_LOG_ROOT


def log_dir_for_scrape(run_id: int, root: Path | None = None) -> Path:
    return _resolve_root(root) / str(run_id)


def log_dir_for_manual(root: Path | None = None) -> Path:
    return _resolve_root(root) / "manual"


def write_extraction_log(
    *,
    log_dir: Path,
    item_id: int,
    batch: ExtractionBatch,
    extractor_version: str,
    scrape_run_id: int | None = None,
) -> Path | None:
    """Atomically write a JSON log of one LLM call to ``log_dir``.

    Manual re-extracts (no scrape_run_id) get a timestamp suffix so repeated
    re-extracts of the same item don't clobber prior runs. Scrape-driven
    calls don't suffix because each (run_id, item_id) is unique by
    construction.

    Returns the written path on success, None on IO failure (logged but
    swallowed — log writes must never abort an extraction).
    """
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("could not create log dir %s", log_dir)
        return None

    if scrape_run_id is None:
        suffix = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        filename = f"{item_id}_{suffix}.json"
    else:
        filename = f"{item_id}.json"
    target = log_dir / filename

    payload = {
        "extractor_version": extractor_version,
        "timestamp": datetime.now(UTC).isoformat(),
        "scrape_run_id": scrape_run_id,
        "item_id": item_id,
        "prompt": batch.prompt,
        "raw_response": batch.raw_response,
        "findings": [
            {
                "status": r.status,
                "confidence": r.confidence,
                "rationale": r.rationale,
                "parse_error": r.parse_error,
                "llm_model": r.llm_model,
                "has_dsl_yaml": r.dsl_yaml is not None,
                "has_parsed_dsl_json": r.parsed_dsl_json is not None,
                "has_proposed_indicators": r.proposed_indicators_json is not None,
                "has_risk_management": r.risk_management_json is not None,
                "has_notable_parameters": r.notable_parameters_json is not None,
            }
            for r in batch.results
        ],
    }

    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, target)
    except OSError:
        logger.exception("could not write extraction log %s", target)
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        return None
    return target
