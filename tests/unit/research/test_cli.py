"""CLI behavior — graceful degradation, exit codes, and argument routing."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from vibe_quant.research.cli import main
from vibe_quant.research.schema import RawItem
from vibe_quant.research.sources import _reset_for_tests, register_source

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolate_registry() -> Generator[None]:
    _reset_for_tests()
    yield
    _reset_for_tests()


def _register_dummy_source() -> None:
    @register_source("dummy_cli")
    class DummyCliSource:
        name = "dummy_cli"

        def fetch(self, since, limit):  # noqa: ARG002
            for i in range(2):
                yield RawItem(
                    source="dummy_cli",
                    external_id=f"d{i}",
                    url=f"http://x/{i}",
                    title=f"t{i}",
                    body=f"b{i}",
                    author="u/x",
                    posted_at=datetime(2026, 1, 1, tzinfo=UTC),
                    score=i,
                    extras={},
                )


def test_unknown_source_exits_2(tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    rc = main(["scrape", "--source", "nonexistent", "--limit", "1", "--db-path", str(db)])
    assert rc == 2


def test_no_extract_leaves_items_pending(tmp_path: Path) -> None:
    _register_dummy_source()
    db = tmp_path / "x.db"
    rc = main(["scrape", "--source", "dummy_cli", "--limit", "5", "--no-extract", "--db-path", str(db)])
    assert rc == 0

    from vibe_quant.db.state_manager import StateManager
    sm = StateManager(db)
    try:
        rows = sm.list_research_items(source="dummy_cli")
        assert len(rows) == 2
        assert all(r["extraction_status"] == "pending" for r in rows)
    finally:
        sm.close()


def test_claude_unavailable_falls_back_to_pending(tmp_path: Path) -> None:
    """When claude CLI is missing, CLI should warn + run with no-extract semantics."""
    _register_dummy_source()
    db = tmp_path / "x.db"

    # Patch the lazy-imported get_default_extractor to raise FileNotFoundError
    with patch(
        "vibe_quant.research.extractor.get_default_extractor",
        side_effect=FileNotFoundError("'claude' CLI not on PATH"),
    ):
        rc = main(["scrape", "--source", "dummy_cli", "--limit", "5", "--db-path", str(db)])

    assert rc == 0  # scrape still completes

    from vibe_quant.db.state_manager import StateManager
    sm = StateManager(db)
    try:
        rows = sm.list_research_items(source="dummy_cli")
        assert len(rows) == 2
        assert all(r["extraction_status"] == "pending" for r in rows)
        # scrape_run row exists and is completed
        run = sm.latest_scrape_run("dummy_cli")
        assert run is not None
        assert run["status"] == "completed"
    finally:
        sm.close()


def test_sources_subcommand_lists_registered(capsys: pytest.CaptureFixture[str]) -> None:
    _register_dummy_source()
    rc = main(["sources"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "dummy_cli" in captured.out
