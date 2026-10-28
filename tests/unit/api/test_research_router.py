"""HTTP wiring tests for /api/research/* (no real subprocess, no claude).

One test per route: the status code plus the DB row/outcome it must leave.
Refusal rows (404/409/422/400) live in their route's test, not in a test each.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from vibe_quant.api.app import create_app
from vibe_quant.api.deps import get_state_manager
from vibe_quant.api.routers import research as research_router
from vibe_quant.db.state_manager import StateManager
from vibe_quant.research import extraction_log
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
def _isolate_log_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[Path]:
    """Redirect on-disk extraction logs into tmp so router tests don't leak
    writes into the repo's data/research/logs/ tree."""
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


def _seed_titled(sm: StateManager, external_id: str, title: str) -> int:
    return sm.create_research_item(
        source="reddit",
        external_id=external_id,
        url=f"https://reddit.com/{external_id}",
        title=title,
        body="b",
        author="u/x",
        posted_at=datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
        score=1,
        extras={"comments": []},
    )


def _seed_extraction(
    sm: StateManager,
    item_id: int,
    *,
    status: str = "parsed",
    parsed_dsl_json: str | None = None,
    evidence_level: str | None = None,
    completeness: float | None = None,
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
        evidence_level=evidence_level,
        completeness=completeness,
    )


# --------------------------------------------------------------- /sources


def test_get_sources_lists_registered(client: TestClient) -> None:
    resp = client.get("/api/research/sources")
    assert resp.status_code == 200
    assert "reddit" in resp.json()["sources"]


# ----------------------------------------------------- /credentials/status


def test_credentials_status_reports_the_user_agent(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REDDIT_USER_AGENT", "vibe-quant-research:0.1 (by /u/me)")
    resp = client.get("/api/research/credentials/status")
    assert resp.status_code == 200
    assert resp.json() == {
        "source": "reddit",
        "user_agent_value": "vibe-quant-research:0.1 (by /u/me)",
        "using_default": False,
    }

    monkeypatch.delenv("REDDIT_USER_AGENT", raising=False)
    body = client.get("/api/research/credentials/status").json()
    assert body["source"] == "reddit"
    assert body["using_default"] is True
    assert "vibe-quant-research" in body["user_agent_value"]


def test_credentials_status_unknown_source_422(client: TestClient) -> None:
    resp = client.get("/api/research/credentials/status", params={"source": "twitter"})
    assert resp.status_code == 422


# ------------------------------------------------ /settings/subreddits


def test_get_subreddits_reads_env_until_saved(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("REDDIT_SUBREDDITS", raising=False)
    body = client.get("/api/research/settings/subreddits").json()
    assert (body["using_default"], body["subreddits"]) == (True, ["algotrading"])

    monkeypatch.setenv("REDDIT_SUBREDDITS", "algotrading,quant")
    body = client.get("/api/research/settings/subreddits").json()
    assert (body["using_default"], body["subreddits"]) == (True, ["algotrading", "quant"])


def test_put_subreddits_persists_dedupes_and_validates(client: TestClient) -> None:
    resp = client.put(
        "/api/research/settings/subreddits",
        json={"subreddits": ["algotrading", "quant", "wallstreetbets"]},
    )
    assert resp.status_code == 200
    assert resp.json()["using_default"] is False
    assert resp.json()["subreddits"] == ["algotrading", "quant", "wallstreetbets"]

    got = client.get("/api/research/settings/subreddits").json()
    assert (got["using_default"], got["subreddits"]) == (
        False,
        ["algotrading", "quant", "wallstreetbets"],
    ), "the saved row wins over the env default"

    deduped = client.put(
        "/api/research/settings/subreddits",
        json={"subreddits": ["algotrading", "quant", "algotrading"]},
    ).json()
    assert deduped["subreddits"] == ["algotrading", "quant"]

    for bad in ([], ["r/foo"], ["Foo"], ["foo bar"], ["ab"], ["x" * 22], [""]):
        resp = client.put("/api/research/settings/subreddits", json={"subreddits": bad})
        assert resp.status_code == 422, f"expected 422 for {bad!r}"


def test_delete_subreddits_resets_to_default(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("REDDIT_SUBREDDITS", raising=False)
    client.put("/api/research/settings/subreddits", json={"subreddits": ["custom"]})
    resp = client.delete("/api/research/settings/subreddits")
    assert resp.status_code == 200
    assert resp.json()["using_default"] is True
    assert resp.json()["subreddits"] == ["algotrading"]
    assert client.get("/api/research/settings/subreddits").json()["using_default"] is True


# ------------------------------------------------------------- /scrape


def test_post_scrape_creates_run_and_spawns_subprocess(
    client: TestClient, sm: StateManager
) -> None:
    with patch("vibe_quant.api.routers.research.subprocess.Popen") as popen:
        resp = client.post("/api/research/scrape", json={"source": "reddit", "limit": 10})

    assert resp.status_code == 201
    body = resp.json()
    assert body["source"] == "reddit"
    assert body["status"] == "running"
    row = sm.get_scrape_run(body["id"])
    assert row is not None and row["status"] == "running"
    popen.assert_called_once()
    cmd = popen.call_args[0][0]
    assert "vibe_quant.research" in cmd
    assert "--scrape-run-id" in cmd
    assert str(body["id"]) in cmd


def test_post_scrape_refusals(client: TestClient, sm: StateManager) -> None:
    assert (
        client.post("/api/research/scrape", json={"source": "nonexistent", "limit": 10}).status_code
        == 422
    ), "an unregistered source is refused"
    assert (
        client.post("/api/research/scrape", json={"source": "Bad-Name!", "limit": 10}).status_code
        == 422
    ), "a malformed source name is refused by the schema"

    sm.create_scrape_run(source="reddit", pid=99999, config={"x": 1})
    with patch("vibe_quant.api.routers.research.subprocess.Popen") as popen:
        resp = client.post("/api/research/scrape", json={"source": "reddit", "limit": 10})
    assert resp.status_code == 409
    popen.assert_not_called()
    assert len(sm.list_active_scrape_runs(source="reddit")) == 1, "no second run row"


def test_get_scrape_returns_the_row_or_404(client: TestClient, sm: StateManager) -> None:
    rid = sm.create_scrape_run(source="reddit", pid=None, config={"x": 1})
    resp = client.get(f"/api/research/scrape/{rid}")
    assert resp.status_code == 200
    assert resp.json()["id"] == rid
    assert client.get("/api/research/scrape/9999999").status_code == 404


def test_get_scrape_latest_returns_most_recent_or_null(
    client: TestClient, sm: StateManager
) -> None:
    assert client.get("/api/research/scrape/latest", params={"source": "reddit"}).json() is None

    sm.create_scrape_run(source="reddit", pid=None)
    rid2 = sm.create_scrape_run(source="reddit", pid=None)
    resp = client.get("/api/research/scrape/latest", params={"source": "reddit"})
    assert resp.status_code == 200
    assert resp.json()["id"] == rid2


def test_delete_scrape_kills_or_refuses(client: TestClient, sm: StateManager) -> None:
    running = sm.create_scrape_run(source="reddit", pid=12345)
    with patch("vibe_quant.api.routers.research.os.kill") as kill:
        resp = client.delete(f"/api/research/scrape/{running}")
    assert resp.status_code == 200
    kill.assert_called_once()

    done = sm.create_scrape_run(source="reddit", pid=None)
    sm.complete_scrape_run(done, status="completed")
    assert client.delete(f"/api/research/scrape/{done}").status_code == 400

    nopid = sm.create_scrape_run(source="reddit", pid=None)
    resp = client.delete(f"/api/research/scrape/{nopid}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "killed"

    dead = sm.create_scrape_run(source="reddit", pid=999999)
    with patch(
        "vibe_quant.api.routers.research.os.kill", side_effect=ProcessLookupError
    ):
        resp = client.delete(f"/api/research/scrape/{dead}")
    assert resp.status_code == 200
    row = sm.get_scrape_run(dead)
    assert row is not None and row["status"] == "killed"


# --------------------------------------------------------------- /items


def test_list_items_filters_and_paginates(client: TestClient, sm: StateManager) -> None:
    for i in range(5):
        _seed_item(sm, external_id=f"e{i}")
    resp = client.get("/api/research/items", params={"source": "reddit", "limit": 2, "offset": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 5
    assert len(body["items"]) == 2
    assert body["limit"] == 2
    assert body["offset"] == 1


def test_list_items_q_filters_on_title(client: TestClient, sm: StateManager) -> None:
    """?q=<word> is a case-insensitive title substring (bd-dwtx)."""
    _seed_titled(sm, "a", "RSI mean reversion on BTC")
    _seed_titled(sm, "b", "Momentum breakout strategy")
    _seed_titled(sm, "c", "MACD + rsi combo")

    body = client.get("/api/research/items", params={"q": "rsi"}).json()
    assert body["total"] == 2
    assert {it["title"] for it in body["items"]} == {
        "RSI mean reversion on BTC",
        "MACD + rsi combo",
    }

    whitespace = client.get("/api/research/items", params={"q": "   "}).json()
    assert whitespace["total"] == 3, "a whitespace-only q is no filter"

    none = client.get("/api/research/items", params={"q": "XYZNOMATCH"}).json()
    assert (none["total"], none["items"]) == (0, [])


def test_list_items_hide_low_trade(client: TestClient, sm: StateManager) -> None:
    low = _seed_item(sm, external_id="low")
    low_ex = _seed_extraction(sm, low)
    sm.update_extraction_screen_results(
        low_ex,
        screen_sharpe=2.0,
        screen_status="done",
        screen_run_id=None,
        screen_trades=10,
        screen_completed_at="2026-05-20T00:00:00+00:00",
    )
    high = _seed_item(sm, external_id="high")
    high_ex = _seed_extraction(sm, high)
    sm.update_extraction_screen_results(
        high_ex,
        screen_sharpe=1.5,
        screen_status="done",
        screen_run_id=None,
        screen_trades=200,
        screen_completed_at="2026-05-20T00:00:00+00:00",
    )

    resp = client.get("/api/research/items", params={"hide_low_trade": "true"})
    assert resp.status_code == 200
    ids = {it["id"] for it in resp.json()["items"]}
    assert high in ids
    assert low not in ids


def test_list_items_sort_screen_sharpe(client: TestClient, sm: StateManager) -> None:
    for ext, sharpe in [("a", 0.3), ("b", 2.1), ("c", 1.0)]:
        iid = _seed_item(sm, external_id=ext)
        eid = _seed_extraction(sm, iid)
        sm.update_extraction_screen_results(
            eid,
            screen_sharpe=sharpe,
            screen_status="done",
            screen_run_id=None,
            screen_trades=100,
            screen_completed_at="2026-05-20T00:00:00+00:00",
        )

    resp = client.get("/api/research/items", params={"sort": "screen_sharpe"})
    assert resp.status_code == 200
    assert [it["external_id"] for it in resp.json()["items"]][:3] == ["b", "c", "a"]


def test_list_items_sort_credibility(client: TestClient, sm: StateManager) -> None:
    cases = [
        ("idea", "idea_only", 0.99),
        ("backtest-low", "backtested", 0.25),
        ("backtest-high", "backtested", 0.8),
        ("live", "live_traded", 0.1),
        ("unknown", None, None),
    ]
    for external_id, evidence_level, completeness in cases:
        iid = _seed_item(sm, external_id=external_id)
        _seed_extraction(
            sm, iid, evidence_level=evidence_level, completeness=completeness
        )

    resp = client.get("/api/research/items", params={"sort": "credibility"})

    assert resp.status_code == 200
    assert [item["external_id"] for item in resp.json()["items"]] == [
        "live",
        "backtest-high",
        "backtest-low",
        "idea",
        "unknown",
    ]


def test_list_items_invalid_sort_422(client: TestClient) -> None:
    assert client.get("/api/research/items", params={"sort": "bogus"}).status_code == 422


def test_get_item_returns_item_extractions_and_latest_job(
    client: TestClient, sm: StateManager
) -> None:
    iid = _seed_item(sm)
    body = client.get(f"/api/research/items/{iid}").json()
    assert body["id"] == iid
    assert body["extractions"] == []
    assert body["latest_job"] is None

    eid = _seed_extraction(sm, iid, evidence_level="live_traded", completeness=0.93)
    body = client.get(f"/api/research/items/{iid}").json()
    assert [e["id"] for e in body["extractions"]] == [eid]
    assert body["extractions"][0]["evidence_level"] == "live_traded"
    assert body["extractions"][0]["completeness"] == 0.93

    assert client.get("/api/research/items/9999999").status_code == 404


def test_get_item_surfaces_the_failed_job(client: TestClient, sm: StateManager) -> None:
    """bd-j68g.2: item detail carries last_error + attempts of the newest job."""
    iid = _seed_item(sm)
    job_id = sm.enqueue_extraction_job(iid, max_attempts=2)
    sm.claim_next_extraction_job()
    sm.fail_extraction_job(job_id, "RuntimeError: kaboom")  # retry -> queued
    sm.claim_next_extraction_job()
    sm.fail_extraction_job(job_id, "RuntimeError: kaboom-final")  # final fail

    body = client.get(f"/api/research/items/{iid}").json()
    latest = body["latest_job"]
    assert latest is not None
    assert latest["id"] == job_id
    assert latest["status"] == "failed"
    assert (latest["attempts"], latest["max_attempts"]) == (2, 2)
    assert "kaboom-final" in (latest["last_error"] or "")


# ------------------------------------------- /items/{id}/extract


def test_extract_item_enqueues_and_the_worker_drains(
    client: TestClient, sm: StateManager
) -> None:
    iid = _seed_item(sm)
    _seed_extraction(sm, iid)  # existing one — must not be overwritten

    from vibe_quant.research.schema import ExtractionBatch, ExtractionResult
    from vibe_quant.research.worker import process_one_job

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
    fake_batch = ExtractionBatch(prompt="P", raw_response="{}", results=[fake_result])

    class _FakeExt:
        def extract_all(self, _item: Any) -> ExtractionBatch:
            return fake_batch

        def extract(self, _item: Any) -> ExtractionResult:
            return fake_result

    resp = client.post(f"/api/research/items/{iid}/extract")
    assert resp.status_code == 202
    body = resp.json()
    assert (body["item_id"], body["status"]) == (iid, "queued")
    job_id = body["job_id"]
    assert isinstance(job_id, int)

    queued = sm.get_research_item(iid)
    assert queued is not None and queued["extraction_status"] == "queued"
    assert len(sm.list_extractions_for_item(iid)) == 1

    second = client.post(f"/api/research/items/{iid}/extract")
    assert second.status_code == 409, "a queued item is refused"
    assert client.post("/api/research/items/9999999/extract").status_code == 404

    with patch("vibe_quant.research.extractor.get_default_extractor", return_value=_FakeExt()):
        result = process_one_job(sm)
    assert result is not None and result["id"] == job_id

    assert len(sm.list_extractions_for_item(iid)) == 2  # history preserved
    after = sm.get_research_item(iid)
    assert after is not None and after["extraction_status"] == "skipped"
    job = sm.get_extraction_job(job_id)
    assert job is not None and job["status"] == "done"


# ------------------------------------------------- /extraction-queue


def test_extraction_queue_lists_active_jobs_with_item_info(
    client: TestClient, sm: StateManager
) -> None:
    """bd-j68g.3: queued + running jobs, joined with the item title/url."""
    iid_a = _seed_titled(sm, "a", "alpha post")
    iid_b = _seed_titled(sm, "b", "beta post")
    job_a = sm.enqueue_extraction_job(iid_a)
    job_b = sm.enqueue_extraction_job(iid_b)
    sm.claim_next_extraction_job()  # one running, one queued

    resp = client.get("/api/research/extraction-queue")
    assert resp.status_code == 200
    body = resp.json()
    assert body["active_count"] == 2
    assert {j["id"] for j in body["jobs"]} == {job_a, job_b}
    assert {j["status"] for j in body["jobs"]} == {"queued", "running"}
    assert {j["item_title"] for j in body["jobs"]} == {"alpha post", "beta post"}


def test_extraction_queue_status_counts(client: TestClient, sm: StateManager) -> None:
    iid = _seed_titled(sm, "a", "alpha")
    sm.enqueue_extraction_job(iid)
    resp = client.get("/api/research/extraction-queue/status")
    assert resp.status_code == 200
    body = resp.json()
    assert (body["queued_count"], body["running_count"], body["active_count"]) == (1, 0, 1)


# ------------------------------------ /extraction-jobs/{id}/cancel


def test_cancel_queued_job_restores_the_item(client: TestClient, sm: StateManager) -> None:
    """bd-j68g.3: a cancelled queued job pops the item back to 'pending'."""
    assert client.post("/api/research/extraction-jobs/9999999/cancel").status_code == 404

    iid = _seed_titled(sm, "a", "alpha")
    job_id = sm.enqueue_extraction_job(iid)
    assert sm.get_research_item(iid)["extraction_status"] == "queued"

    resp = client.post(f"/api/research/extraction-jobs/{job_id}/cancel")
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "cancelled"
    after = sm.get_research_item(iid)
    assert after is not None and after["extraction_status"] == "pending"

    again = client.post(f"/api/research/extraction-jobs/{job_id}/cancel")
    assert (again.status_code, again.json()["status"]) == (200, "cancelled"), "idempotent"

    redo = client.post(f"/api/research/items/{iid}/extract")
    assert (redo.status_code, redo.json()["status"]) == (202, "queued"), "re-extract works"


@pytest.mark.parametrize("final_status", ["done", "failed"])
def test_cancel_terminal_job_returns_409(
    client: TestClient, sm: StateManager, final_status: str
) -> None:
    """A job already done/failed at request time is refused: 409 naming the status, row unchanged."""
    iid = _seed_titled(sm, "a", "alpha")
    job_id = sm.enqueue_extraction_job(iid)
    sm.claim_next_extraction_job()
    sm.complete_extraction_job(job_id, status=final_status)

    resp = client.post(f"/api/research/extraction-jobs/{job_id}/cancel")
    assert resp.status_code == 409, resp.text
    assert final_status in resp.json()["detail"]
    row = sm.get_extraction_job(job_id)
    assert row is not None and row["status"] == final_status


def test_cancel_running_job_waits_for_the_worker(client: TestClient, sm: StateManager) -> None:
    iid = _seed_titled(sm, "a", "alpha")
    job_id = sm.enqueue_extraction_job(iid)
    sm.claim_next_extraction_job()

    def _confirm() -> None:
        # Ordering handshake: confirm after the endpoint stamped
        # cancel_requested_at (the signal the worker's watcher polls in prod).
        deadline = time.monotonic() + 10.0
        while not sm.extraction_job_cancel_requested(job_id):
            if time.monotonic() > deadline:
                return
            time.sleep(0.01)
        sm.finish_cancelled_extraction_job(job_id, error_message="cancelled by user")

    th = threading.Thread(target=_confirm)
    th.start()
    try:
        resp = client.post(f"/api/research/extraction-jobs/{job_id}/cancel")
    finally:
        th.join(timeout=15.0)

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "cancelled"
    row = sm.get_extraction_job(job_id)
    assert row is not None
    assert row["status"] == "cancelled"
    assert row["cancel_requested_at"] is not None
    after = sm.get_research_item(iid)
    assert after is not None and after["extraction_status"] == "pending"


@pytest.mark.parametrize("final_status", ["done", "failed"])
def test_cancel_running_job_finished_before_confirm_is_409(
    client: TestClient,
    sm: StateManager,
    monkeypatch: pytest.MonkeyPatch,
    final_status: str,
) -> None:
    """A job that reaches a REAL terminal outcome before the worker confirms 409s."""
    monkeypatch.setattr(research_router, "_CANCEL_WAIT_SECONDS", 2.0)
    monkeypatch.setattr(research_router, "_CANCEL_POLL_INTERVAL_S", 0.05)
    iid = _seed_titled(sm, "a", "alpha")
    job_id = sm.enqueue_extraction_job(iid)
    sm.claim_next_extraction_job()

    def _finish() -> None:
        time.sleep(0.15)
        sm.complete_extraction_job(job_id, status=final_status)

    th = threading.Thread(target=_finish)
    th.start()
    try:
        resp = client.post(f"/api/research/extraction-jobs/{job_id}/cancel")
    finally:
        th.join(timeout=5.0)

    assert resp.status_code == 409, resp.text
    assert final_status in resp.json()["detail"]
    row = sm.get_extraction_job(job_id)
    assert row is not None and row["status"] == final_status


def test_cancel_running_job_without_a_confirm_reports_pending(
    client: TestClient, sm: StateManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No worker confirms in the wait window → 202 + response-only 'cancel_pending'."""
    monkeypatch.setattr(research_router, "_CANCEL_WAIT_SECONDS", 0.3)
    monkeypatch.setattr(research_router, "_CANCEL_POLL_INTERVAL_S", 0.05)
    iid = _seed_titled(sm, "a", "alpha")
    job_id = sm.enqueue_extraction_job(iid)
    sm.claim_next_extraction_job()

    resp = client.post(f"/api/research/extraction-jobs/{job_id}/cancel")
    assert resp.status_code == 202, resp.text
    assert resp.json()["status"] == "cancel_pending"
    row = sm.get_extraction_job(job_id)
    assert row is not None
    assert row["status"] == "running", "'cancel_pending' is never stored"
    assert row["cancel_requested_at"] is not None


def test_cancel_running_job_requeued_by_worker_reports_cancelled(
    client: TestClient, sm: StateManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F3: a job the worker re-queued while we waited is still cancellable."""
    monkeypatch.setattr(research_router, "_CANCEL_WAIT_SECONDS", 2.0)
    monkeypatch.setattr(research_router, "_CANCEL_POLL_INTERVAL_S", 0.05)
    iid = _seed_titled(sm, "a", "alpha")
    job_id = sm.enqueue_extraction_job(iid)
    sm.claim_next_extraction_job()

    requeue_ran: list[bool] = []

    def _requeue() -> None:
        # Ordering handshake: requeue only after the endpoint stamped
        # cancel_requested_at (a fixed sleep let the requeue land first).
        deadline = time.monotonic() + 10.0
        while not sm.extraction_job_cancel_requested(job_id):
            if time.monotonic() > deadline:
                return
            time.sleep(0.01)
        sm.fail_extraction_job(job_id, "boom")  # attempts 1 < max 3 -> queued
        requeue_ran.append(True)

    th = threading.Thread(target=_requeue)
    th.start()
    try:
        resp = client.post(f"/api/research/extraction-jobs/{job_id}/cancel")
    finally:
        th.join(timeout=15.0)

    assert requeue_ran == [True], "worker never requeued the job after the cancel stamp"
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "cancelled"
    row = sm.get_extraction_job(job_id)
    assert row is not None
    assert row["status"] == "cancelled"
    assert row["attempts"] == 1  # the requeue actually incremented attempts


# ------------------------------------------------ /extractions/{id}/promote


def test_promote_extraction_branches(client: TestClient, sm: StateManager) -> None:
    iid = _seed_item(sm)
    eid = _seed_extraction(sm, iid)
    resp = client.post(f"/api/research/extractions/{eid}/promote")
    assert resp.status_code == 200
    strategy_id = resp.json()["strategy_id"]
    assert isinstance(strategy_id, int)

    after = sm.get_extraction(eid)
    assert after is not None
    assert after["status"] == "promoted"
    assert after["strategy_id"] == strategy_id
    assert sm.get_strategy(strategy_id) is not None

    again = client.post(f"/api/research/extractions/{eid}/promote")
    assert again.status_code == 400
    assert str(strategy_id) in again.json()["detail"]

    failed = _seed_extraction(sm, iid, status="failed", parsed_dsl_json=None)
    resp = client.post(f"/api/research/extractions/{failed}/promote")
    assert resp.status_code == 400
    assert "unparsed" in resp.json()["detail"].lower()

    rejected = _seed_extraction(sm, iid)
    sm.update_extraction_status(rejected, status="rejected")
    assert client.post(f"/api/research/extractions/{rejected}/promote").status_code == 400

    assert client.post("/api/research/extractions/9999999/promote").status_code == 404


# ------------------------------------------------- /extractions/{id}/reject


def test_reject_extraction_branches(client: TestClient, sm: StateManager) -> None:
    iid = _seed_item(sm)
    eid = _seed_extraction(sm, iid)
    resp = client.post(f"/api/research/extractions/{eid}/reject")
    assert resp.status_code == 200
    assert resp.json()["status"] == "rejected"
    assert sm.get_extraction(eid)["status"] == "rejected"  # type: ignore[index]

    promoted = _seed_extraction(sm, iid)
    assert client.post(f"/api/research/extractions/{promoted}/promote").status_code == 200
    assert client.post(f"/api/research/extractions/{promoted}/reject").status_code == 400


# ----------------------------------------------- /extractions/{id}/rescreen


def test_rescreen_branches(
    client: TestClient, sm: StateManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vibe_quant.research import auto_screen
    from vibe_quant.screening.types import BacktestMetrics

    assert client.post("/api/research/extractions/99999/rescreen").status_code == 404

    iid = _seed_item(sm)
    unparsed = _seed_extraction(sm, iid, status="failed", parsed_dsl_json=None)
    assert client.post(f"/api/research/extractions/{unparsed}/rescreen").status_code == 400

    eid = _seed_extraction(sm, iid)
    monkeypatch.setattr(auto_screen, "_normalize_dsl", lambda s: {"timeframe": "1h"})
    monkeypatch.setattr(
        auto_screen,
        "_run_single_metrics",
        lambda *a, **k: BacktestMetrics(
            parameters={},
            sharpe_ratio=0.5,
            profit_factor=1.1,
            max_drawdown=0.2,
            total_return=0.05,
            total_trades=60,
        ),
    )
    auto_screen.auto_screen_extraction(sm, eid, "{}")
    first = sm.get_extraction(eid)
    assert first is not None
    first_run_id = first["screen_run_id"]
    assert first["screen_sharpe"] == 0.5

    monkeypatch.setattr(
        auto_screen,
        "_run_single_metrics",
        lambda *a, **k: BacktestMetrics(
            parameters={},
            sharpe_ratio=2.7,
            profit_factor=1.9,
            max_drawdown=0.04,
            total_return=0.48,
            total_trades=150,
        ),
    )
    resp = client.post(f"/api/research/extractions/{eid}/rescreen")
    assert resp.status_code == 200
    body = resp.json()
    assert body["screen_sharpe"] == 2.7
    assert body["screen_trades"] == 150
    assert body["screen_run_id"] != first_run_id, "a new backtest_runs row"
    assert sm.get_backtest_run(first_run_id) is not None, "the prior run is preserved"
