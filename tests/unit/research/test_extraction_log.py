"""Tests for the on-disk extraction log writer."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from vibe_quant.research import extraction_log
from vibe_quant.research.extraction_log import (
    log_dir_for_manual,
    log_dir_for_scrape,
    write_extraction_log,
)
from vibe_quant.research.schema import ExtractionBatch, ExtractionResult

if TYPE_CHECKING:
    import pytest


def _result(status: str = "parsed") -> ExtractionResult:
    return ExtractionResult(
        status=status,
        confidence=0.7,
        rationale="r",
        raw_response='{"x":1}',
        dsl_yaml="name: x\n",
        parsed_dsl_json='{"name":"x"}',
        parse_error=None,
        llm_model="claude-p",
        proposed_indicators_json='[{"name":"adaptive_x"}]',
    )


def _batch() -> ExtractionBatch:
    return ExtractionBatch(
        prompt="SYSTEM\n<<<USER_CONTENT>>>\nbody\n<<<END>>>\n",
        raw_response='{"result":"[]"}',
        results=[_result("parsed"), _result("skipped")],
    )


def test_write_extraction_log_creates_file_per_scrape_item(tmp_path: Path) -> None:
    log_dir = log_dir_for_scrape(42, root=tmp_path)
    target = write_extraction_log(
        log_dir=log_dir,
        item_id=7,
        batch=_batch(),
        extractor_version="claude-p:abc123",
        scrape_run_id=42,
    )
    assert target == tmp_path / "42" / "7.json"
    assert target.exists()
    payload = json.loads(target.read_text())
    assert payload["item_id"] == 7
    assert payload["scrape_run_id"] == 42
    assert payload["extractor_version"] == "claude-p:abc123"
    assert payload["prompt"].startswith("SYSTEM")
    assert payload["raw_response"] == '{"result":"[]"}'
    assert len(payload["findings"]) == 2
    assert payload["findings"][0]["status"] == "parsed"
    # Don't leak heavy fields — only summarized booleans
    assert payload["findings"][0]["has_dsl_yaml"] is True
    assert payload["findings"][0]["has_parsed_dsl_json"] is True
    assert payload["findings"][0]["has_proposed_indicators"] is True


def test_write_extraction_log_manual_reextract_uses_timestamp_suffix(tmp_path: Path) -> None:
    log_dir = log_dir_for_manual(root=tmp_path)
    first = write_extraction_log(
        log_dir=log_dir,
        item_id=11,
        batch=_batch(),
        extractor_version="v1",
        scrape_run_id=None,
    )
    # Sleep just enough to advance the second-resolution suffix.
    time.sleep(1.1)
    second = write_extraction_log(
        log_dir=log_dir,
        item_id=11,
        batch=_batch(),
        extractor_version="v1",
        scrape_run_id=None,
    )
    assert first is not None and second is not None
    assert first.name != second.name
    assert first.exists() and second.exists()
    assert first.parent == tmp_path / "manual"


def test_write_extraction_log_atomic_no_tmp_left_behind(tmp_path: Path) -> None:
    log_dir = log_dir_for_scrape(99, root=tmp_path)
    target = write_extraction_log(
        log_dir=log_dir,
        item_id=1,
        batch=_batch(),
        extractor_version="v",
        scrape_run_id=99,
    )
    assert target is not None and target.exists()
    leftovers = list(target.parent.glob("*.tmp"))
    assert leftovers == []


def test_write_extraction_log_returns_none_on_io_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken filesystem must not abort extraction — the writer returns None."""
    log_dir = tmp_path / "scrape-1"

    def boom(self: Path, *args: object, **kwargs: object) -> None:  # noqa: ARG001
        raise OSError("disk full")

    monkeypatch.setattr(Path, "mkdir", boom)
    target = write_extraction_log(
        log_dir=log_dir,
        item_id=5,
        batch=_batch(),
        extractor_version="v",
        scrape_run_id=1,
    )
    assert target is None


def test_log_dir_helpers_pick_up_default_root_monkeypatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``DEFAULT_LOG_ROOT`` is resolved at call time, not at import time."""
    monkeypatch.setattr(extraction_log, "DEFAULT_LOG_ROOT", tmp_path)
    assert log_dir_for_scrape(1) == tmp_path / "1"
    assert log_dir_for_manual() == tmp_path / "manual"


def test_write_extraction_log_unicode_round_trip(tmp_path: Path) -> None:
    """Reddit posts can contain emoji + non-ASCII; persistence must keep them."""
    batch = ExtractionBatch(
        prompt="prompt 🟢",
        raw_response='{"r":"日本語"}',
        results=[_result()],
    )
    target = write_extraction_log(
        log_dir=tmp_path,
        item_id=1,
        batch=batch,
        extractor_version="v",
        scrape_run_id=1,
    )
    assert target is not None
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["prompt"] == "prompt 🟢"
    assert payload["raw_response"] == '{"r":"日本語"}'


def test_write_extraction_log_overwrites_same_item_in_same_scrape(tmp_path: Path) -> None:
    """One scrape_run can re-call only on retry/replay; same path is OK to overwrite
    because the (run_id, item_id) tuple is unique by construction."""
    log_dir = log_dir_for_scrape(5, root=tmp_path)
    a = write_extraction_log(
        log_dir=log_dir,
        item_id=2,
        batch=ExtractionBatch(prompt="p1", raw_response="r1", results=[_result()]),
        extractor_version="v",
        scrape_run_id=5,
    )
    b = write_extraction_log(
        log_dir=log_dir,
        item_id=2,
        batch=ExtractionBatch(prompt="p2", raw_response="r2", results=[_result()]),
        extractor_version="v",
        scrape_run_id=5,
    )
    assert a == b == log_dir / "2.json"
    payload = json.loads(b.read_text())
    assert payload["prompt"] == "p2"
    assert payload["raw_response"] == "r2"
    # Make sure we replaced cleanly — no tmp file left behind
    assert list(log_dir.glob("*.tmp")) == []
    # And no stale data lingered
    assert os.path.exists(b)
