"""StateManager helpers for the research pipeline."""

from __future__ import annotations

import sqlite3
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


def test_schema_creates_all_research_tables(sm: StateManager) -> None:
    rows = sm.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'research_%'"
    ).fetchall()
    names = {r["name"] for r in rows}
    assert names == {
        "research_items",
        "research_extractions",
        "research_extraction_jobs",
        "research_indicator_scaffolds",
        "research_scrape_runs",
        "research_settings",
    }


def test_init_schema_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "x.db"
    legacy = sqlite3.connect(db_path)
    legacy.execute(
        """CREATE TABLE research_extractions (
               id INTEGER PRIMARY KEY,
               research_item_id INTEGER,
               status TEXT
           )"""
    )
    legacy.commit()
    legacy.close()

    sm1 = StateManager(db_path)
    _ = sm1.conn
    sm1.close()
    sm2 = StateManager(db_path)
    columns = {
        row["name"]
        for row in sm2.conn.execute("PRAGMA table_info(research_extractions)").fetchall()
    }
    assert {"evidence_level", "completeness"} <= columns
    with pytest.raises(sqlite3.IntegrityError):
        sm2.conn.execute(
            "INSERT INTO research_extractions (evidence_level) VALUES (?)",
            ("paper_traded",),
        )
    with pytest.raises(sqlite3.IntegrityError):
        sm2.conn.execute(
            "INSERT INTO research_extractions (completeness) VALUES (?)",
            (1.01,),
        )
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
    sm.update_research_item_status(reddit_items[0]["id"], "extracted")
    extracted = sm.list_research_items(source="reddit", status="extracted")
    assert len(extracted) == 1


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
        evidence_level="backtested",
        completeness=0.84,
    )
    got = sm.get_extraction(ex_id)
    assert got is not None
    assert got["status"] == "parsed"
    assert got["confidence"] == 0.87
    assert got["evidence_level"] == "backtested"
    assert got["completeness"] == 0.84
    assert got["strategy_id"] is None

    items = sm.list_extractions_for_item(item_id)
    assert len(items) == 1
    assert items[0]["id"] == ex_id


def test_extraction_fk_violation_raises(sm: StateManager) -> None:
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


def test_research_subreddits_unset_returns_none(sm: StateManager) -> None:
    assert sm.get_research_subreddits("reddit") is None


def test_research_subreddits_set_then_get(sm: StateManager) -> None:
    sm.set_research_subreddits("reddit", ["algotrading", "quant"])
    assert sm.get_research_subreddits("reddit") == ["algotrading", "quant"]


def test_research_subreddits_upsert_replaces(sm: StateManager) -> None:
    sm.set_research_subreddits("reddit", ["a"])
    sm.set_research_subreddits("reddit", ["b", "c"])
    assert sm.get_research_subreddits("reddit") == ["b", "c"]


def test_research_subreddits_clear(sm: StateManager) -> None:
    sm.set_research_subreddits("reddit", ["a"])
    sm.clear_research_subreddits("reddit")
    assert sm.get_research_subreddits("reddit") is None


def _make_item_with_screen(
    sm: StateManager,
    *,
    external_id: str,
    sharpe: float | None,
    trades: int | None,
) -> int:
    """Helper: create item + one extraction stamped with screen metrics."""
    item_id = sm.create_research_item(
        source="reddit",
        external_id=external_id,
        url="u",
        title=external_id,
        body=None,
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
        parsed_dsl_json='{"name":"x"}',
        parse_error=None,
    )
    sm.update_extraction_screen_results(
        ex_id,
        screen_sharpe=sharpe,
        screen_status="done" if sharpe is not None else "failed",
        screen_run_id=None,
        screen_trades=trades,
        screen_completed_at="2026-05-20T00:00:00+00:00",
    )
    return item_id


def test_list_research_items_sort_by_screen_sharpe(sm: StateManager) -> None:
    low = _make_item_with_screen(sm, external_id="low", sharpe=0.5, trades=80)
    high = _make_item_with_screen(sm, external_id="high", sharpe=2.5, trades=80)
    mid = _make_item_with_screen(sm, external_id="mid", sharpe=1.2, trades=80)
    none = _make_item_with_screen(sm, external_id="none", sharpe=None, trades=None)

    items = sm.list_research_items(sort="screen_sharpe")
    ordered_ids = [r["id"] for r in items]
    # high, mid, low ranked by sharpe; none (NULL) sorts last.
    assert ordered_ids[0] == high
    assert ordered_ids[1] == mid
    assert ordered_ids[2] == low
    assert ordered_ids[3] == none


def test_list_research_items_hide_low_trade_excludes_only_all_low(
    sm: StateManager,
) -> None:
    # Item with single low-trade extraction → hidden.
    hidden = _make_item_with_screen(sm, external_id="hidden", sharpe=3.0, trades=10)
    # Item with single high-trade extraction → shown.
    shown_high = _make_item_with_screen(sm, external_id="shown_h", sharpe=1.0, trades=100)
    # Item with no extractions → shown (no trade-count data).
    shown_no_ex = sm.create_research_item(
        source="reddit",
        external_id="shown_noex",
        url="u",
        title="t",
        body=None,
        author=None,
        posted_at=None,
        score=None,
    )
    # Item where one extraction is low, another is high → shown (mixed).
    mixed = sm.create_research_item(
        source="reddit",
        external_id="mixed",
        url="u",
        title="t",
        body=None,
        author=None,
        posted_at=None,
        score=None,
    )
    for s, t in [(0.2, 5), (1.8, 200)]:
        ex_id = sm.create_extraction(
            research_item_id=mixed,
            status="parsed",
            llm_model=None,
            confidence=None,
            rationale=None,
            raw_response="",
            dsl_yaml=None,
            parsed_dsl_json='{"name":"x"}',
            parse_error=None,
        )
        sm.update_extraction_screen_results(
            ex_id,
            screen_sharpe=s,
            screen_status="done",
            screen_run_id=None,
            screen_trades=t,
            screen_completed_at="2026-05-20T00:00:00+00:00",
        )

    items = sm.list_research_items(hide_low_trade=True)
    visible_ids = {r["id"] for r in items}
    assert hidden not in visible_ids
    assert shown_high in visible_ids
    assert shown_no_ex in visible_ids
    assert mixed in visible_ids

    # count_research_items must match the filtered list length.
    assert sm.count_research_items(hide_low_trade=True) == len(visible_ids)


# --- _migrate_research_items_allow_queued regression tests (bd-mrpl) ---


def _downgrade_research_items_to_pre_queued(db_path: Path) -> None:
    """Take a fully-migrated DB and rewrite research_items to the pre-queued
    shape (no CHECK constraint on extraction_status). Done with FK off so we
    can DROP TABLE without the test itself hitting the bug under repair."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.executescript(
        """
        BEGIN;
        CREATE TABLE research_items_legacy (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            external_id TEXT NOT NULL,
            url TEXT NOT NULL,
            title TEXT,
            body TEXT,
            author TEXT,
            posted_at TEXT,
            score INTEGER,
            extras_json TEXT,
            fetched_at TEXT DEFAULT (datetime('now')),
            extraction_status TEXT DEFAULT 'pending',
            UNIQUE(source, external_id)
        );
        INSERT INTO research_items_legacy SELECT * FROM research_items;
        DROP TABLE research_items;
        ALTER TABLE research_items_legacy RENAME TO research_items;
        COMMIT;
        """
    )
    conn.commit()
    conn.close()


def _seed_legacy_research_items_db(db_path: Path) -> None:
    """Build a DB pinned to the pre-queued shape, with one FK-bearing
    research_extractions row to reproduce the original bug condition."""
    # Bootstrap the modern schema first so all tables (and FKs) exist.
    sm = StateManager(db_path)
    sm.create_research_item(
        source="reddit",
        external_id="abc",
        url="https://x/abc",
        title=None,
        body=None,
        author=None,
        posted_at=None,
        score=None,
    )
    sm.create_research_item(
        source="reddit",
        external_id="def",
        url="https://x/def",
        title=None,
        body=None,
        author=None,
        posted_at=None,
        score=None,
    )
    sm.conn.execute(
        "INSERT INTO research_extractions (research_item_id) VALUES (1)"
    )
    sm.conn.commit()
    sm.close()
    # Then rewrite research_items into the pre-queued (no-CHECK) shape.
    _downgrade_research_items_to_pre_queued(db_path)


def test_migrate_allow_queued_succeeds_with_fk_referencing_items(tmp_path: Path) -> None:
    """Migration must rebuild research_items even when other tables hold
    foreign keys to it (regression for FOREIGN KEY constraint failed bug)."""
    db_path = tmp_path / "legacy.db"
    _seed_legacy_research_items_db(db_path)

    sm = StateManager(db_path)
    _ = sm.conn  # triggers init_schema → migration

    # CHECK constraint with 'queued' should now be present.
    sql = sm.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='research_items'"
    ).fetchone()[0]
    assert "'queued'" in sql

    # No orphan rebuild table left behind.
    orphan = sm.conn.execute(
        "SELECT name FROM sqlite_master WHERE name='research_items_new'"
    ).fetchone()
    assert orphan is None

    # Data preserved.
    rows = sm.conn.execute("SELECT id, source FROM research_items ORDER BY id").fetchall()
    assert [(r["id"], r["source"]) for r in rows] == [(1, "reddit"), (2, "reddit")]

    # FK enforcement is restored.
    fk = sm.conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert fk == 1
    sm.close()


def test_migrate_allow_queued_cleans_up_orphan_rebuild_table(tmp_path: Path) -> None:
    """A prior crashed migration may have left research_items_new in place.
    The migration must drop it and complete cleanly."""
    import sqlite3

    db_path = tmp_path / "orphan.db"
    _seed_legacy_research_items_db(db_path)
    # Simulate a half-finished prior migration: rebuild table already exists.
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE research_items_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            external_id TEXT NOT NULL,
            url TEXT NOT NULL,
            title TEXT,
            body TEXT,
            author TEXT,
            posted_at TEXT,
            score INTEGER,
            extras_json TEXT,
            fetched_at TEXT DEFAULT (datetime('now')),
            extraction_status TEXT DEFAULT 'pending'
                CHECK (extraction_status IN
                    ('pending', 'queued', 'running', 'extracted', 'failed', 'skipped')),
            UNIQUE(source, external_id)
        );
        """
    )
    conn.commit()
    conn.close()

    sm = StateManager(db_path)
    _ = sm.conn

    # Orphan must be gone, live table must have CHECK.
    orphan = sm.conn.execute(
        "SELECT name FROM sqlite_master WHERE name='research_items_new'"
    ).fetchone()
    assert orphan is None
    sql = sm.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='research_items'"
    ).fetchone()[0]
    assert "'queued'" in sql
    sm.close()


def test_migrate_allow_queued_drops_orphan_on_already_migrated_db(tmp_path: Path) -> None:
    """When research_items already has the CHECK (no rebuild needed), the
    migration should still clean up any stale research_items_new."""
    import sqlite3

    # First, build a fully-migrated DB.
    db_path = tmp_path / "fresh.db"
    sm = StateManager(db_path)
    _ = sm.conn
    sm.close()

    # Now plant an orphan rebuild table.
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE research_items_new (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    sm2 = StateManager(db_path)
    _ = sm2.conn  # init_schema runs again
    orphan = sm2.conn.execute(
        "SELECT name FROM sqlite_master WHERE name='research_items_new'"
    ).fetchone()
    assert orphan is None
    sm2.close()
