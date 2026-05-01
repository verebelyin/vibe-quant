"""Persist `RawItem` objects into the `research_items` table.

Thin wrapper over `StateManager` that catches the unique-constraint violation
on `(source, external_id)` so duplicates are silently skipped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vibe_quant.db.state_manager import DuplicateResearchItem, StateManager

if TYPE_CHECKING:
    from pathlib import Path

    from vibe_quant.research.schema import RawItem


def archive_item(
    sm: StateManager,
    item: RawItem,
) -> tuple[bool, int | None]:
    """Insert a RawItem; return (was_new, item_id_or_None_on_dup).

    If `(source, external_id)` already exists, returns `(False, None)` rather
    than raising — pipelines can use the boolean to decide whether to spawn
    extraction.
    """
    posted_at_str = item.posted_at.isoformat() if item.posted_at is not None else None
    try:
        item_id = sm.create_research_item(
            source=item.source,
            external_id=item.external_id,
            url=item.url,
            title=item.title,
            body=item.body,
            author=item.author,
            posted_at=posted_at_str,
            score=item.score,
            extras=item.extras or None,
        )
        return True, item_id
    except DuplicateResearchItem:
        return False, None


def open_state_manager(db_path: Path | None = None) -> StateManager:
    """Convenience constructor for non-test callers."""
    return StateManager(db_path)
