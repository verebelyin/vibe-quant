"""HTTP contract tests for /api/research/* (no real subprocess, no claude)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from vibe_quant.api.app import create_app
from vibe_quant.api.deps import get_state_manager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.research.sources import _reset_for_tests, register_source

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@pytest.fixture
def sm(tmp_path: Path) -> Generator[StateManager]:
    s = StateManager(tmp_path / "research.db")
    yield s
    s.close()


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


def _seed_item(sm: StateManager, *, source: str = "reddit", external_id: str = "abc") -> int:
    return sm.create_research_item(
        source=source,
        external_id=external_id,
        url=f"https://reddit.com/{external_id}",
        title="t",
        body="b",
        author="u/x",
        posted_at=datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
        score=10,
        extras={"comments": []},
    )


def _seed_extraction(
    sm: StateManager,
    item_id: int,
    *,
    status: str = "parsed",
    parsed_dsl_json: str | None = None,
) -> int:
    if parsed_dsl_json is None and status == "parsed":
        parsed_dsl_json = (
            '{"name":"x","timeframe":"1h",'
            '"indicators":{"rsi":{"type":"RSI","period":14}},'
            '"entry_conditions":{"long":["rsi < 30"],"short":[]},'
            '"exit_conditions":{"long":["rsi > 70"],"short":[]},'
            '"stop_loss":{"type":"fixed_pct","percent":2.0},'
            '"take_profit":{"type":"fixed_pct","percent":5.0}}'
        )
    return sm.create_extraction(
        research_item_id=item_id,
        status=status,
        llm_model="t",
        confidence=0.8,
        rationale="ok",
        raw_response="{}",
        dsl_yaml="name: x\n",
        parsed_dsl_json=parsed_dsl_json,
        parse_error=None,
    )


# ---------- /sources ----------


def test_get_sources_lists_registered(client: TestClient) -> None:
    resp = client.get("/api/research/sources")
    assert resp.status_code == 200
    assert "reddit" in resp.json()["sources"]


# ---------- /credentials/status ----------


def test_credentials_status_user_agent_set(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REDDIT_USER_AGENT", "vibe-quant-research:0.1 (by /u/me)")
    resp = client.get("/api/research/credentials/status")
    assert resp.status_code == 200
    assert resp.json() == {
        "source": "reddit",
        "user_agent_set": True,
        "user_agent_value": "vibe-quant-research:0.1 (by /u/me)",
        "using_default": False,
    }


def test_credentials_status_user_agent_unset_returns_default(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("REDDIT_USER_AGENT", raising=False)
    resp = client.get("/api/research/credentials/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "reddit"
    assert body["user_agent_set"] is False
    assert body["using_default"] is True
    assert body["user_agent_value"]  # always non-null
    assert "vibe-quant-research" in body["user_agent_value"]


def test_credentials_status_unknown_source_422(client: TestClient) -> None:
    resp = client.get("/api/research/credentials/status", params={"source": "twitter"})
    assert resp.status_code == 422


def test_credentials_status_response_omits_deprecated_fields(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REDDIT_USER_AGENT", "x")
    body = client.get("/api/research/credentials/status").json()
    for legacy in ("configured", "missing", "set_vars"):
        assert legacy not in body


# ---------- /scrape (POST) ----------


def test_post_scrape_creates_run_and_spawns_subprocess(client: TestClient, sm: StateManager) -> None:
    with patch("vibe_quant.api.routers.research.subprocess.Popen") as popen:
        resp = client.post("/api/research/scrape", json={"source": "reddit", "limit": 10})

    assert resp.status_code == 201
    body = resp.json()
    assert body["source"] == "reddit"
    assert body["status"] == "running"
    assert isinstance(body["id"], int)
    popen.assert_called_once()
    cmd = popen.call_args[0][0]
    assert "vibe_quant.research" in cmd
    assert "--scrape-run-id" in cmd
    assert str(body["id"]) in cmd


def test_post_scrape_unknown_source_422(client: TestClient) -> None:
    resp = client.post("/api/research/scrape", json={"source": "nonexistent", "limit": 10})
    assert resp.status_code == 422


def test_post_scrape_invalid_source_pattern_422(client: TestClient) -> None:
    resp = client.post("/api/research/scrape", json={"source": "Bad-Name!", "limit": 10})
    assert resp.status_code == 422


def test_post_scrape_conflict_when_already_running(client: TestClient, sm: StateManager) -> None:
    # Seed a running scrape directly
    sm.create_scrape_run(source="reddit", pid=99999, config={"x": 1})

    with patch("vibe_quant.api.routers.research.subprocess.Popen"):
        resp = client.post("/api/research/scrape", json={"source": "reddit", "limit": 10})
    assert resp.status_code == 409


# ---------- /scrape/{id} ----------


def test_get_scrape_returns_row(client: TestClient, sm: StateManager) -> None:
    rid = sm.create_scrape_run(source="reddit", pid=None, config={"x": 1})
    resp = client.get(f"/api/research/scrape/{rid}")
    assert resp.status_code == 200
    assert resp.json()["id"] == rid


def test_get_scrape_404(client: TestClient) -> None:
    resp = client.get("/api/research/scrape/9999999")
    assert resp.status_code == 404


def test_get_scrape_latest_returns_most_recent(client: TestClient, sm: StateManager) -> None:
    sm.create_scrape_run(source="reddit", pid=None)
    rid2 = sm.create_scrape_run(source="reddit", pid=None)
    resp = client.get("/api/research/scrape/latest", params={"source": "reddit"})
    assert resp.status_code == 200
    assert resp.json()["id"] == rid2


def test_get_scrape_latest_returns_null_when_no_runs(client: TestClient) -> None:
    resp = client.get("/api/research/scrape/latest", params={"source": "reddit"})
    assert resp.status_code == 200
    assert resp.json() is None


# ---------- DELETE /scrape/{id} ----------


def test_delete_scrape_running_signals_pid(client: TestClient, sm: StateManager) -> None:
    rid = sm.create_scrape_run(source="reddit", pid=12345)
    with patch("vibe_quant.api.routers.research.os.kill") as kill:
        resp = client.delete(f"/api/research/scrape/{rid}")
    assert resp.status_code == 200
    kill.assert_called_once()


def test_delete_scrape_already_completed_400(client: TestClient, sm: StateManager) -> None:
    rid = sm.create_scrape_run(source="reddit", pid=None)
    sm.complete_scrape_run(rid, status="completed")
    resp = client.delete(f"/api/research/scrape/{rid}")
    assert resp.status_code == 400


def test_delete_scrape_no_pid_finalizes_directly(client: TestClient, sm: StateManager) -> None:
    rid = sm.create_scrape_run(source="reddit", pid=None)
    resp = client.delete(f"/api/research/scrape/{rid}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "killed"


def test_delete_scrape_dead_pid_finalizes_as_killed(client: TestClient, sm: StateManager) -> None:
    rid = sm.create_scrape_run(source="reddit", pid=999999)
    with patch("vibe_quant.api.routers.research.os.kill", side_effect=ProcessLookupError):
        resp = client.delete(f"/api/research/scrape/{rid}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "killed"


# ---------- /items ----------


def test_list_items_with_filters_and_pagination(client: TestClient, sm: StateManager) -> None:
    for i in range(5):
        _seed_item(sm, external_id=f"e{i}")
    resp = client.get("/api/research/items", params={"source": "reddit", "limit": 2, "offset": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 5
    assert len(body["items"]) == 2
    assert body["limit"] == 2
    assert body["offset"] == 1


def test_get_item_404(client: TestClient) -> None:
    resp = client.get("/api/research/items/9999999")
    assert resp.status_code == 404


def test_get_item_returns_item_plus_extractions(client: TestClient, sm: StateManager) -> None:
    iid = _seed_item(sm)
    eid = _seed_extraction(sm, iid)
    resp = client.get(f"/api/research/items/{iid}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == iid
    assert len(body["extractions"]) == 1
    assert body["extractions"][0]["id"] == eid


def test_extract_item_creates_new_extraction_row(client: TestClient, sm: StateManager) -> None:
    iid = _seed_item(sm)
    _seed_extraction(sm, iid)  # existing one — must not be overwritten

    from vibe_quant.research.schema import ExtractionResult

    fake_result = ExtractionResult(
        status="skipped",
        confidence=0.1,
        rationale="re-run says no",
        raw_response="{}",
        dsl_yaml=None,
        parsed_dsl_json=None,
        parse_error=None,
        llm_model="t",
    )

    class _FakeExt:
        def extract(self, _item: Any) -> ExtractionResult:
            return fake_result

    with patch("vibe_quant.research.extractor.get_default_extractor", return_value=_FakeExt()):
        resp = client.post(f"/api/research/items/{iid}/extract")

    assert resp.status_code == 201
    rows = sm.list_extractions_for_item(iid)
    assert len(rows) == 2  # history preserved


def test_extract_item_404(client: TestClient) -> None:
    resp = client.post("/api/research/items/9999999/extract")
    assert resp.status_code == 404


# ---------- /extractions/{id}/promote ----------


def test_promote_creates_strategy_and_marks_promoted(client: TestClient, sm: StateManager) -> None:
    iid = _seed_item(sm)
    eid = _seed_extraction(sm, iid)

    resp = client.post(f"/api/research/extractions/{eid}/promote")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["strategy_id"], int)

    after = sm.get_extraction(eid)
    assert after is not None
    assert after["status"] == "promoted"
    assert after["strategy_id"] == body["strategy_id"]
    strategy = sm.get_strategy(body["strategy_id"])
    assert strategy is not None


def test_promote_failed_extraction_400(client: TestClient, sm: StateManager) -> None:
    iid = _seed_item(sm)
    eid = _seed_extraction(sm, iid, status="failed", parsed_dsl_json=None)
    resp = client.post(f"/api/research/extractions/{eid}/promote")
    assert resp.status_code == 400
    assert "unparsed" in resp.json()["detail"].lower()


def test_promote_already_promoted_400(client: TestClient, sm: StateManager) -> None:
    iid = _seed_item(sm)
    eid = _seed_extraction(sm, iid)
    # First promote
    resp1 = client.post(f"/api/research/extractions/{eid}/promote")
    assert resp1.status_code == 200
    sid = resp1.json()["strategy_id"]
    # Second promote → 400 with reference to existing strategy
    resp2 = client.post(f"/api/research/extractions/{eid}/promote")
    assert resp2.status_code == 400
    assert str(sid) in resp2.json()["detail"]


def test_promote_rejected_400(client: TestClient, sm: StateManager) -> None:
    iid = _seed_item(sm)
    eid = _seed_extraction(sm, iid)
    sm.update_extraction_status(eid, status="rejected")
    resp = client.post(f"/api/research/extractions/{eid}/promote")
    assert resp.status_code == 400


def test_promote_nonexistent_404(client: TestClient) -> None:
    resp = client.post("/api/research/extractions/9999999/promote")
    assert resp.status_code == 404


# ---------- /extractions/{id}/reject ----------


def test_reject_marks_rejected(client: TestClient, sm: StateManager) -> None:
    iid = _seed_item(sm)
    eid = _seed_extraction(sm, iid)
    resp = client.post(f"/api/research/extractions/{eid}/reject")
    assert resp.status_code == 200
    assert resp.json()["status"] == "rejected"


def test_reject_already_promoted_400(client: TestClient, sm: StateManager) -> None:
    iid = _seed_item(sm)
    eid = _seed_extraction(sm, iid)
    # Promote first via the router so a real strategy row exists
    promote = client.post(f"/api/research/extractions/{eid}/promote")
    assert promote.status_code == 200
    resp = client.post(f"/api/research/extractions/{eid}/reject")
    assert resp.status_code == 400


# ---------- /docs ----------


def test_openapi_includes_research_tag(client: TestClient) -> None:
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    spec = resp.json()
    paths = spec.get("paths", {})
    research_paths = [p for p in paths if p.startswith("/api/research/")]
    assert len(research_paths) >= 8  # we registered ~10
