"""Pipeline orchestrator tests with fake sources (no network)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.research.pipeline import run_scrape
from vibe_quant.research.schema import ExtractionResult, RawItem
from vibe_quant.research.sources import _reset_for_tests, register_source

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolate_registry() -> Generator[None]:
    _reset_for_tests()
    yield
    _reset_for_tests()


@pytest.fixture
def sm(tmp_path: Path) -> Generator[StateManager]:
    s = StateManager(tmp_path / "p.db")
    yield s
    s.close()


def _register_fake_source(items: list[RawItem]) -> None:
    @register_source("fake")
    class FakeSource:
        name = "fake"

        def fetch(self, since, limit):  # noqa: ARG002
            yield from items[:limit]


def _item(i: int) -> RawItem:
    return RawItem(
        source="fake",
        external_id=f"e{i}",
        url=f"http://example.com/{i}",
        title=f"t{i}",
        body=f"b{i}",
        author="u/x",
        posted_at=datetime(2026, 1, 1, tzinfo=UTC),
        score=i,
        extras={"k": i},
    )


def test_scrape_archives_items_and_dedups(sm: StateManager) -> None:
    items = [_item(i) for i in range(3)]
    _register_fake_source(items)

    # Bypass the package-loader scan — we register directly via the decorator
    # in this test, but run_scrape calls load_builtin_sources() which is a
    # no-op when nothing is in the package directory.

    summary = run_scrape(sm=sm, source_name="fake", limit=10)
    assert summary.items_fetched == 3
    assert summary.items_new == 3
    assert summary.status == "completed"

    # Re-run: dedup
    summary2 = run_scrape(sm=sm, source_name="fake", limit=10)
    assert summary2.items_fetched == 3
    assert summary2.items_new == 0
    assert summary2.status == "completed"

    rows = sm.list_research_items(source="fake")
    assert len(rows) == 3


def test_scrape_extract_fn_invoked_per_new_item(sm: StateManager) -> None:
    _register_fake_source([_item(i) for i in range(3)])

    calls: list[int] = []

    def fake_extract(item: RawItem, item_id: int) -> list[ExtractionResult]:  # noqa: ARG001
        calls.append(item_id)
        return [
            ExtractionResult(
                status="parsed",
                confidence=0.8,
                rationale="ok",
                raw_response="{}",
                dsl_yaml="name: x\n",
                parsed_dsl_json='{"name": "x"}',
                parse_error=None,
                llm_model="test",
            )
        ]

    summary = run_scrape(sm=sm, source_name="fake", limit=10, extract_fn=fake_extract)
    assert summary.items_extracted == 3
    assert len(calls) == 3
    # extraction rows persisted
    items = sm.list_research_items(source="fake")
    for it in items:
        ex = sm.list_extractions_for_item(it["id"])
        assert len(ex) == 1
        assert ex[0]["status"] == "parsed"
        assert it["extraction_status"] == "extracted"


def test_extractor_failure_does_not_abort_run(sm: StateManager) -> None:
    _register_fake_source([_item(i) for i in range(5)])

    def flaky_extract(item: RawItem, item_id: int) -> list[ExtractionResult]:  # noqa: ARG001
        if "e2" in item.external_id:
            raise RuntimeError("boom")
        return [
            ExtractionResult(
                status="parsed",
                confidence=0.5,
                rationale=None,
                raw_response="",
                dsl_yaml=None,
                parsed_dsl_json=None,
                parse_error=None,
                llm_model="t",
            )
        ]

    summary = run_scrape(sm=sm, source_name="fake", limit=10, extract_fn=flaky_extract)
    assert summary.status == "completed"
    assert summary.items_failed == 1
    assert summary.items_extracted == 4

    failed_item = next(
        it for it in sm.list_research_items(source="fake") if it["external_id"] == "e2"
    )
    assert failed_item["extraction_status"] == "failed"


def test_no_extract_leaves_items_pending(sm: StateManager) -> None:
    _register_fake_source([_item(i) for i in range(2)])
    summary = run_scrape(sm=sm, source_name="fake", limit=10, extract_fn=None)
    assert summary.items_new == 2
    assert summary.items_extracted == 0
    rows = sm.list_research_items(source="fake")
    assert all(r["extraction_status"] == "pending" for r in rows)


def test_extracted_false_recorded_as_skipped(sm: StateManager) -> None:
    """status=skipped must NOT increment items_failed and should mark items as skipped."""
    _register_fake_source([_item(i) for i in range(3)])

    def skip_extract(item: RawItem, item_id: int) -> list[ExtractionResult]:  # noqa: ARG001
        return [
            ExtractionResult(
                status="skipped",
                confidence=0.0,
                rationale="not a strategy",
                raw_response="{}",
                dsl_yaml=None,
                parsed_dsl_json=None,
                parse_error=None,
                llm_model="t",
            )
        ]

    summary = run_scrape(sm=sm, source_name="fake", limit=10, extract_fn=skip_extract)
    assert summary.items_new == 3
    assert summary.items_skipped == 3
    assert summary.items_failed == 0
    assert summary.items_extracted == 0

    rows = sm.list_research_items(source="fake")
    assert all(r["extraction_status"] == "skipped" for r in rows)
    # extraction rows still persisted for triage
    for r in rows:
        ex = sm.list_extractions_for_item(r["id"])
        assert len(ex) == 1
        assert ex[0]["status"] == "skipped"


def test_unknown_source_raises_keyerror_with_no_run_row(sm: StateManager) -> None:
    with pytest.raises(KeyError, match="nonexistent"):
        run_scrape(sm=sm, source_name="nonexistent", limit=1)
    # No orphan scrape_run row created
    assert sm.latest_scrape_run("nonexistent") is None


def test_source_construction_failure_creates_no_run_row(sm: StateManager) -> None:
    @register_source("badcfg")
    class BadCfgSource:
        name = "badcfg"

        def __init__(self) -> None:
            from vibe_quant.alerts.telegram import ConfigurationError

            raise ConfigurationError("Missing FOO env var")

        def fetch(self, since, limit):  # noqa: ARG002
            yield from ()

    from vibe_quant.alerts.telegram import ConfigurationError

    with pytest.raises(ConfigurationError, match="FOO"):
        run_scrape(sm=sm, source_name="badcfg", limit=1)
    assert sm.latest_scrape_run("badcfg") is None


def test_source_exception_marks_run_failed(sm: StateManager) -> None:
    @register_source("bad")
    class BadSource:
        name = "bad"

        def fetch(self, since, limit):  # noqa: ARG002
            raise RuntimeError("source crashed")
            yield  # pragma: no cover  (make it a generator)

    summary = run_scrape(sm=sm, source_name="bad", limit=10)
    assert summary.status == "failed"
    assert summary.error_message is not None
    assert "source crashed" in summary.error_message
    run_row = sm.get_scrape_run(summary.scrape_run_id)
    assert run_row is not None
    assert run_row["status"] == "failed"


def test_scrape_run_row_finalized_on_success(sm: StateManager) -> None:
    _register_fake_source([_item(i) for i in range(2)])
    summary = run_scrape(sm=sm, source_name="fake", limit=10)
    row = sm.get_scrape_run(summary.scrape_run_id)
    assert row is not None
    assert row["status"] == "completed"
    assert row["completed_at"] is not None
    assert row["items_new"] == 2
    assert row["pid"] is not None


def test_sigterm_during_iteration_finalizes_as_killed(sm: StateManager) -> None:
    """Source self-signals SIGTERM mid-fetch → run finalizes as killed."""
    import os
    import signal as _signal

    items = [_item(i) for i in range(20)]

    @register_source("self_killer")
    class SelfKillerSource:
        name = "self_killer"

        def fetch(self, since, limit):  # noqa: ARG002
            yield items[0]
            # Signal ourselves; handler sets kill_flag, next iteration breaks
            os.kill(os.getpid(), _signal.SIGTERM)
            yield items[1]
            yield items[2]

    summary = run_scrape(sm=sm, source_name="self_killer", limit=10)
    assert summary.status == "killed"
    row = sm.get_scrape_run(summary.scrape_run_id)
    assert row is not None
    assert row["status"] == "killed"
    assert row["completed_at"] is not None
    # At least one item was archived before the kill
    assert summary.items_fetched >= 1
