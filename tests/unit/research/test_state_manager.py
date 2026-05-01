"""StateManager helpers for the research pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vibe_quant.db.state_manager import DuplicateResearchItem, StateManager

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@pytest.fixture
def sm(tmp_path: Path) -> Generator[StateManager]:
    mgr = StateManager(tmp_path / "research.db")
    yield mgr
    mgr.close()


def test_schema_creates_all_three_tables(sm: StateManager) -> None:
    rows = sm.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'research_%'"
    ).fetchall()
    names = {r["name"] for r in rows}
    assert names == {"research_items", "research_extractions", "research_scrape_runs"}


def test_init_schema_idempotent(tmp_path: Path) -> None:
    sm1 = StateManager(tmp_path / "x.db")
    _ = sm1.conn
    sm1.close()
    sm2 = StateManager(tmp_path / "x.db")
    _ = sm2.conn
    sm2.close()


def test_wal_mode_enabled(sm: StateManager) -> None:
    mode = sm.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_create_and_get_research_item_round_trip(sm: StateManager) -> None:
    item_id = sm.create_research_item(
        source="reddit",
        external_id="abc123",
        url="https://reddit.com/r/x/comments/abc123",
        title="hello",
        body="body text",
        author="u/foo",
        posted_at="2026-05-01T12:00:00Z",
        score=42,
        extras={"comments": [{"author": "u/bar", "body": "+1", "score": 3}]},
    )
    got = sm.get_research_item(item_id)
    assert got is not None
    assert got["source"] == "reddit"
    assert got["external_id"] == "abc123"
    assert got["title"] == "hello"
    assert got["score"] == 42
    assert got["extraction_status"] == "pending"
    assert got["extras"]["comments"][0]["author"] == "u/bar"


def test_duplicate_research_item_raises_typed_error(sm: StateManager) -> None:
    sm.create_research_item(
        source="reddit",
        external_id="dup1",
        url="u",
        title=None,
        body=None,
        author=None,
        posted_at=None,
        score=None,
    )
    with pytest.raises(DuplicateResearchItem) as exc:
        sm.create_research_item(
            source="reddit",
            external_id="dup1",
            url="u2",
            title=None,
            body=None,
            author=None,
            posted_at=None,
            score=None,
        )
    assert "dup1" in str(exc.value)


def test_research_item_with_all_nullable_fields_succeeds(sm: StateManager) -> None:
    item_id = sm.create_research_item(
        source="reddit",
        external_id="link-only",
        url="https://reddit.com/x",
        title=None,
        body=None,
        author=None,
        posted_at=None,
        score=None,
        extras=None,
    )
    got = sm.get_research_item(item_id)
    assert got is not None
    assert got["body"] is None
    assert got["author"] is None
    assert got["score"] is None
    assert got["extras"] == {}


def test_list_research_items_filters_by_source_and_status(sm: StateManager) -> None:
    for i in range(3):
        sm.create_research_item(
            source="reddit",
            external_id=f"r{i}",
            url="u",
            title=f"t{i}",
            body=None,
            author=None,
            posted_at=f"2026-05-0{i+1}",
            score=None,
        )
    sm.create_research_item(
        source="arxiv",
        external_id="a1",
        url="u",
        title="ax",
        body=None,
        author=None,
        posted_at=None,
        score=None,
    )

    reddit_items = sm.list_research_items(source="reddit")
    assert len(reddit_items) == 3
    arxiv_items = sm.list_research_items(source="arxiv")
    assert len(arxiv_items) == 1

    # filter by status
    sm.update_research_item_status(reddit_items[0]["id"], "parsed")
    parsed = sm.list_research_items(source="reddit", status="parsed")
    assert len(parsed) == 1


def test_extraction_round_trip(sm: StateManager) -> None:
    item_id = sm.create_research_item(
        source="reddit",
        external_id="ext1",
        url="u",
        title="t",
        body="b",
        author=None,
        posted_at=None,
        score=None,
    )
    ex_id = sm.create_extraction(
        research_item_id=item_id,
        status="parsed",
        llm_model="claude-opus-4-7",
        confidence=0.87,
        rationale="looks like RSI mean reversion",
        raw_response='{"extracted": true}',
        dsl_yaml="name: foo\n",
        parsed_dsl_json='{"name":"foo"}',
        parse_error=None,
    )
    got = sm.get_extraction(ex_id)
    assert got is not None
    assert got["status"] == "parsed"
    assert got["confidence"] == 0.87
    assert got["strategy_id"] is None

    items = sm.list_extractions_for_item(item_id)
    assert len(items) == 1
    assert items[0]["id"] == ex_id


def test_extraction_fk_violation_raises(sm: StateManager) -> None:
    import sqlite3

    sm.conn.execute("PRAGMA foreign_keys = ON")
    with pytest.raises(sqlite3.IntegrityError):
        sm.create_extraction(
            research_item_id=999999,
            status="parsed",
            llm_model="m",
            confidence=0.5,
            rationale=None,
            raw_response="",
            dsl_yaml=None,
            parsed_dsl_json=None,
            parse_error=None,
        )


def test_update_extraction_status_with_strategy_id(sm: StateManager) -> None:
    item_id = sm.create_research_item(
        source="reddit",
        external_id="p1",
        url="u",
        title="t",
        body="b",
        author=None,
        posted_at=None,
        score=None,
    )
    ex_id = sm.create_extraction(
        research_item_id=item_id,
        status="parsed",
        llm_model=None,
        confidence=None,
        rationale=None,
        raw_response="",
        dsl_yaml=None,
        parsed_dsl_json=None,
        parse_error=None,
    )
    strategy_id = sm.create_strategy(name="reddit_p1", dsl_config={})
    ok = sm.update_extraction_status(ex_id, "promoted", strategy_id=strategy_id)
    assert ok
    got = sm.get_extraction(ex_id)
    assert got is not None
    assert got["status"] == "promoted"
    assert got["strategy_id"] == strategy_id


def test_scrape_run_lifecycle(sm: StateManager) -> None:
    run_id = sm.create_scrape_run(source="reddit", pid=12345, config={"limit": 50})
    assert sm.get_scrape_run(run_id) is not None
    assert sm.latest_scrape_run("reddit")["id"] == run_id

    sm.increment_scrape_run_counters(run_id, fetched=10, new=8, extracted=6, failed=2)
    got = sm.get_scrape_run(run_id)
    assert got["items_fetched"] == 10
    assert got["items_new"] == 8
    assert got["items_extracted"] == 6
    assert got["items_failed"] == 2

    # heartbeat on existing run
    assert sm.update_scrape_run_heartbeat(run_id) is True

    # heartbeat on non-existent run returns False
    assert sm.update_scrape_run_heartbeat(99999) is False

    sm.complete_scrape_run(run_id, status="completed")
    got = sm.get_scrape_run(run_id)
    assert got["status"] == "completed"
    assert got["completed_at"] is not None


def test_complete_nonexistent_scrape_run_returns_false(sm: StateManager) -> None:
    assert sm.complete_scrape_run(99999, status="failed", error_message="x") is False
