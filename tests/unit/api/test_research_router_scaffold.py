"""Endpoint tests for POST /api/research/extractions/{id}/indicators/{idx}/scaffold.

Slice 1 of bd-3p1k.1: only the validation/cache layer is wired up. The
success-path test asserts the endpoint stops at ``not_implemented``
until slices 2 + 3 add LLM codegen / auto-commit.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from vibe_quant.api.app import create_app
from vibe_quant.api.deps import get_state_manager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.research import extraction_log
from vibe_quant.research.sources import _reset_for_tests, register_source

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@pytest.fixture
def sm(tmp_path: Path) -> Generator[StateManager]:
    s = StateManager(tmp_path / "scaffold.db")
    yield s
    s.close()


@pytest.fixture(autouse=True)
def _isolate_log_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[Path]:
    root = tmp_path / "research-logs"
    monkeypatch.setattr(extraction_log, "DEFAULT_LOG_ROOT", root)
    yield root


@pytest.fixture(autouse=True)
def _isolate_registry() -> Generator[None]:
    _reset_for_tests()

    @register_source("reddit")
    class FakeReddit:
        name = "reddit"

        def fetch(self, since, limit):  # noqa: ARG002
            yield from ()

    yield
    _reset_for_tests()


@pytest.fixture
def client(sm: StateManager) -> Generator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_state_manager] = lambda: sm
    with TestClient(app) as c:
        yield c


def _seed_extraction_with_proposals(
    sm: StateManager, proposals: list[dict] | None
) -> int:
    item_id = sm.create_research_item(
        source="reddit",
        external_id="abc",
        url="https://reddit.com/abc",
        title="t",
        body="b",
        author="u/x",
        posted_at=datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
        score=10,
        extras={"comments": []},
    )
    return sm.create_extraction(
        research_item_id=item_id,
        status="parsed",
        llm_model="t",
        confidence=0.8,
        rationale="ok",
        raw_response="{}",
        dsl_yaml="name: x\n",
        parsed_dsl_json=None,
        parse_error=None,
        proposed_indicators_json=(
            json.dumps(proposals) if proposals is not None else None
        ),
    )


def _scaffold(client: TestClient, ext_id: int, idx: int, force: bool = False) -> dict:
    params = {"force": "true"} if force else None
    resp = client.post(
        f"/api/research/extractions/{ext_id}/indicators/{idx}/scaffold",
        params=params,
    )
    return {"status_code": resp.status_code, "body": resp.json()}


# ---------- 404 paths ----------


def test_scaffold_unknown_extraction_404(client: TestClient) -> None:
    out = _scaffold(client, 9999, 0)
    assert out["status_code"] == 404


def test_scaffold_extraction_without_proposals_404(
    client: TestClient, sm: StateManager
) -> None:
    ext_id = _seed_extraction_with_proposals(sm, proposals=None)
    out = _scaffold(client, ext_id, 0)
    assert out["status_code"] == 404


def test_scaffold_idx_out_of_range_404(client: TestClient, sm: StateManager) -> None:
    ext_id = _seed_extraction_with_proposals(
        sm, proposals=[{"name": "x", "formula": "f"}]
    )
    out = _scaffold(client, ext_id, 5)
    assert out["status_code"] == 404


# ---------- status=invalid_input ----------


def test_scaffold_missing_formula_invalid_input(
    client: TestClient, sm: StateManager
) -> None:
    ext_id = _seed_extraction_with_proposals(
        sm, proposals=[{"name": "x"}]  # no formula
    )
    out = _scaffold(client, ext_id, 0)
    assert out["status_code"] == 200
    assert out["body"]["status"] == "invalid_input"
    assert "formula" in (out["body"].get("error") or "").lower()


# ---------- status=name_collision ----------


def test_scaffold_name_collision_returns_suggested(
    client: TestClient, sm: StateManager
) -> None:
    # RSI is a built-in spec.
    ext_id = _seed_extraction_with_proposals(
        sm, proposals=[{"name": "rsi", "formula": "100 - 100/(1+RS)"}]
    )
    out = _scaffold(client, ext_id, 0)
    assert out["status_code"] == 200
    assert out["body"]["status"] == "name_collision"
    assert out["body"]["name"] == "RSI"
    assert out["body"]["suggested_name"] == "RSI_V2"


# ---------- status=not_implemented (happy-path stub) ----------


def test_scaffold_happy_path_returns_not_implemented(
    client: TestClient, sm: StateManager
) -> None:
    ext_id = _seed_extraction_with_proposals(
        sm,
        proposals=[
            {
                "name": "my_novel",
                "formula": "ema(close, period) / atr(period)",
                "parameters": {"period": {"default": 14, "range": [5, 30]}},
                "output_range": "unbounded",
            }
        ],
    )
    out = _scaffold(client, ext_id, 0)
    assert out["status_code"] == 200
    assert out["body"]["status"] == "not_implemented"
    assert out["body"]["name"] == "MY_NOVEL"


# ---------- status=already_scaffolded + force ----------


def test_scaffold_already_scaffolded_returns_cached_row(
    client: TestClient, sm: StateManager
) -> None:
    ext_id = _seed_extraction_with_proposals(
        sm,
        proposals=[{"name": "novel_a", "formula": "ema(close, period)"}],
    )
    # Simulate slices 2+3 having already succeeded by seeding the scaffold row.
    sm.upsert_indicator_scaffold(
        extraction_id=ext_id,
        idx=0,
        status="ok",
        plugin_path="vibe_quant/dsl/plugins/proposed_novel_a.py",
        test_path="tests/unit/test_plugins/test_proposed_novel_a.py",
        commit_sha="abc123",
    )
    out = _scaffold(client, ext_id, 0)
    assert out["body"]["status"] == "already_scaffolded"
    assert out["body"]["plugin_path"] == "vibe_quant/dsl/plugins/proposed_novel_a.py"
    assert out["body"]["commit_sha"] == "abc123"


def test_scaffold_force_clears_cache_and_falls_through(
    client: TestClient, sm: StateManager
) -> None:
    ext_id = _seed_extraction_with_proposals(
        sm,
        proposals=[{"name": "novel_b", "formula": "ema(close, period)"}],
    )
    sm.upsert_indicator_scaffold(
        extraction_id=ext_id,
        idx=0,
        status="ok",
        plugin_path="vibe_quant/dsl/plugins/proposed_novel_b.py",
        commit_sha="def456",
    )
    out = _scaffold(client, ext_id, 0, force=True)
    # Force=1 nukes the cache and the slice-1 stub returns not_implemented.
    assert out["body"]["status"] == "not_implemented"
    assert sm.get_indicator_scaffold(ext_id, 0) is None


def test_scaffold_failed_cache_does_not_short_circuit(
    client: TestClient, sm: StateManager
) -> None:
    # A prior codegen_failed row should NOT count as "already scaffolded";
    # the user clicking again is a retry intent.
    ext_id = _seed_extraction_with_proposals(
        sm,
        proposals=[{"name": "novel_c", "formula": "f"}],
    )
    sm.upsert_indicator_scaffold(
        extraction_id=ext_id,
        idx=0,
        status="codegen_failed",
        error="banned_import:os",
    )
    out = _scaffold(client, ext_id, 0)
    assert out["body"]["status"] == "not_implemented"


# ---------- state manager round-trip ----------


def test_state_manager_scaffold_round_trip(sm: StateManager) -> None:
    ext_id = _seed_extraction_with_proposals(
        sm, proposals=[{"name": "x", "formula": "f"}]
    )
    assert sm.get_indicator_scaffold(ext_id, 0) is None

    row = sm.upsert_indicator_scaffold(
        extraction_id=ext_id,
        idx=0,
        status="ok",
        plugin_path="p",
        test_path="t",
        commit_sha="abc",
    )
    assert row["status"] == "ok"
    assert row["plugin_path"] == "p"

    # Update flips status without losing key identity.
    row2 = sm.upsert_indicator_scaffold(
        extraction_id=ext_id,
        idx=0,
        status="codegen_failed",
        error="boom",
    )
    assert row2["status"] == "codegen_failed"
    assert row2["error"] == "boom"
    assert row2["plugin_path"] is None  # cleared by upsert

    assert sm.delete_indicator_scaffold(ext_id, 0) is True
    assert sm.get_indicator_scaffold(ext_id, 0) is None
    assert sm.delete_indicator_scaffold(ext_id, 0) is False
