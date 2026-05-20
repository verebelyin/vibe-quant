"""Pipeline orchestrator tests with fake sources (no network)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.research import auto_screen, extraction_log
from vibe_quant.research.pipeline import run_scrape
from vibe_quant.research.schema import ExtractionBatch, ExtractionResult, RawItem
from vibe_quant.research.sources import _reset_for_tests, register_source

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolate_registry() -> Generator[None]:
    _reset_for_tests()
    yield
    _reset_for_tests()


@pytest.fixture(autouse=True)
def _isolate_log_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[Path]:
    """Redirect on-disk extraction logs into tmp so tests don't leak into
    the repo's data/ tree."""
    log_root = tmp_path / "research-logs"
    monkeypatch.setattr(extraction_log, "DEFAULT_LOG_ROOT", log_root)
    yield log_root


_REAL_AUTO_SCREEN = auto_screen.auto_screen_extraction


@pytest.fixture(autouse=True)
def _stub_auto_screen(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Skip real NT screening in unit tests — pipeline hook would otherwise
    try to compile and backtest a DSL against the on-disk catalog."""
    monkeypatch.setattr(
        auto_screen, "auto_screen_extraction", lambda sm, eid, dsl: None
    )
    yield


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

    def fake_extract(item: RawItem, item_id: int) -> ExtractionBatch:  # noqa: ARG001
        calls.append(item_id)
        return ExtractionBatch(
            prompt="P",
            raw_response="{}",
            results=[
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
            ],
        )

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


def test_scrape_writes_extraction_log_per_item(
    sm: StateManager, _isolate_log_root: Path
) -> None:
    """Per-extraction prompt + raw response must land under
    data/research/logs/<scrape_run_id>/<item_id>.json so future
    prompt-engineering analysis can replay verbatim."""
    import json

    _register_fake_source([_item(i) for i in range(2)])

    def fake_extract(item: RawItem, item_id: int) -> ExtractionBatch:  # noqa: ARG001
        return ExtractionBatch(
            prompt=f"PROMPT for {item_id}",
            raw_response=f'{{"out": {item_id}}}',
            results=[
                ExtractionResult(
                    status="parsed",
                    confidence=1.0,
                    rationale="r",
                    raw_response="",
                    dsl_yaml="name: x\n",
                    parsed_dsl_json='{"name":"x"}',
                    parse_error=None,
                    llm_model="t",
                )
            ],
        )

    summary = run_scrape(sm=sm, source_name="fake", limit=10, extract_fn=fake_extract)
    assert summary.items_extracted == 2

    run_dir = _isolate_log_root / str(summary.scrape_run_id)
    files = sorted(run_dir.glob("*.json"))
    assert len(files) == 2
    payloads = [json.loads(f.read_text()) for f in files]
    prompts = {p["prompt"] for p in payloads}
    assert prompts == {f"PROMPT for {p['item_id']}" for p in payloads}
    for p in payloads:
        assert p["scrape_run_id"] == summary.scrape_run_id
        assert p["raw_response"] == f'{{"out": {p["item_id"]}}}'
        assert p["extractor_version"].startswith("claude-p:")
        assert len(p["findings"]) == 1


def test_extractor_failure_does_not_abort_run(sm: StateManager) -> None:
    _register_fake_source([_item(i) for i in range(5)])

    def flaky_extract(item: RawItem, item_id: int) -> ExtractionBatch:  # noqa: ARG001
        if "e2" in item.external_id:
            raise RuntimeError("boom")
        return ExtractionBatch(
            prompt="P",
            raw_response="",
            results=[
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
            ],
        )

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

    def skip_extract(item: RawItem, item_id: int) -> ExtractionBatch:  # noqa: ARG001
        return ExtractionBatch(
            prompt="P",
            raw_response="{}",
            results=[
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
            ],
        )

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


def test_auto_screen_runs_once_per_parsed_extraction(
    sm: StateManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each parsed extraction with non-empty parsed_dsl_json must trigger
    auto_screen exactly once; skipped/failed/null-DSL must not."""
    _register_fake_source([_item(i) for i in range(3)])

    calls: list[tuple[int, str]] = []

    def fake_screen(sm_arg, extraction_id: int, parsed_dsl_json: str) -> None:  # noqa: ANN001
        calls.append((extraction_id, parsed_dsl_json))
        run_id = sm_arg.create_backtest_run(
            strategy_id=None,
            run_mode="screening",
            symbols=["BTCUSDT-PERP.BINANCE"],
            timeframe="1h",
            start_date="2026-01-01",
            end_date="2026-02-01",
            parameters={"auto_screen_source": {"extraction_id": extraction_id}},
        )
        sm_arg.update_extraction_screen_results(
            extraction_id,
            screen_sharpe=1.23,
            screen_status="done",
            screen_run_id=run_id,
        )

    monkeypatch.setattr(auto_screen, "auto_screen_extraction", fake_screen)

    def mixed_extract(item: RawItem, item_id: int) -> ExtractionBatch:  # noqa: ARG001
        # 3 findings per item: parsed-with-dsl, parsed-without-dsl, skipped
        return ExtractionBatch(
            prompt="P",
            raw_response="{}",
            results=[
                ExtractionResult(
                    status="parsed",
                    confidence=0.9,
                    rationale=None,
                    raw_response="",
                    dsl_yaml="name: s\n",
                    parsed_dsl_json='{"name":"s"}',
                    parse_error=None,
                    llm_model="t",
                ),
                ExtractionResult(
                    status="parsed",
                    confidence=0.5,
                    rationale=None,
                    raw_response="",
                    dsl_yaml=None,
                    parsed_dsl_json=None,
                    parse_error=None,
                    llm_model="t",
                ),
                ExtractionResult(
                    status="skipped",
                    confidence=0.0,
                    rationale=None,
                    raw_response="",
                    dsl_yaml=None,
                    parsed_dsl_json=None,
                    parse_error=None,
                    llm_model="t",
                ),
            ],
        )

    summary = run_scrape(sm=sm, source_name="fake", limit=10, extract_fn=mixed_extract)
    assert summary.items_new == 3
    # one parsed-with-DSL invocation per item
    assert len(calls) == 3
    for _eid, payload in calls:
        assert payload == '{"name":"s"}'

    # screen columns surfaced on the parsed-with-DSL rows
    for it in sm.list_research_items(source="fake"):
        exs = sm.list_extractions_for_item(it["id"])
        with_dsl = [e for e in exs if e["parsed_dsl_json"]]
        without_dsl = [e for e in exs if not e["parsed_dsl_json"]]
        assert len(with_dsl) == 1
        assert with_dsl[0]["screen_status"] == "done"
        assert with_dsl[0]["screen_sharpe"] == pytest.approx(1.23)
        assert with_dsl[0]["screen_run_id"] is not None
        for e in without_dsl:
            assert e["screen_status"] is None
            assert e["screen_sharpe"] is None


def test_auto_screen_runner_failure_recorded_as_failed(
    sm: StateManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NTScreeningRunner exceptions must be swallowed by auto_screen and
    recorded as screen_status='failed' — scrape itself completes cleanly."""
    _register_fake_source([_item(i) for i in range(2)])

    def _blow_up(*_a: object, **_k: object) -> None:
        raise RuntimeError("nt fail")

    monkeypatch.setattr(auto_screen, "_normalize_dsl", lambda s: {"timeframe": "1h"})
    monkeypatch.setattr(auto_screen, "_run_single_sharpe", _blow_up)
    # Restore the real auto_screen function (the autouse stub replaced it).
    monkeypatch.setattr(auto_screen, "auto_screen_extraction", _REAL_AUTO_SCREEN)

    def fake_extract(item: RawItem, item_id: int) -> ExtractionBatch:  # noqa: ARG001
        return ExtractionBatch(
            prompt="P",
            raw_response="{}",
            results=[
                ExtractionResult(
                    status="parsed",
                    confidence=1.0,
                    rationale=None,
                    raw_response="",
                    dsl_yaml="x",
                    parsed_dsl_json='{"name":"x"}',
                    parse_error=None,
                    llm_model="t",
                )
            ],
        )

    summary = run_scrape(sm=sm, source_name="fake", limit=10, extract_fn=fake_extract)
    assert summary.status == "completed"
    # extraction rows persisted with screen_status='failed'
    for it in sm.list_research_items(source="fake"):
        exs = sm.list_extractions_for_item(it["id"])
        assert len(exs) == 1
        assert exs[0]["screen_status"] == "failed"
        assert exs[0]["screen_sharpe"] is None
        # a backtest_run row was created for traceability before the failure
        assert exs[0]["screen_run_id"] is not None


def test_auto_screen_invalid_dsl_records_failed_without_run(
    sm: StateManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Garbage parsed_dsl_json must be recorded as failed with no run row."""
    _register_fake_source([_item(0)])

    monkeypatch.setattr(auto_screen, "auto_screen_extraction", _REAL_AUTO_SCREEN)

    def fake_extract(item: RawItem, item_id: int) -> ExtractionBatch:  # noqa: ARG001
        return ExtractionBatch(
            prompt="P",
            raw_response="{}",
            results=[
                ExtractionResult(
                    status="parsed",
                    confidence=1.0,
                    rationale=None,
                    raw_response="",
                    dsl_yaml="x",
                    parsed_dsl_json='{"this": "is not a valid dsl"}',
                    parse_error=None,
                    llm_model="t",
                )
            ],
        )

    summary = run_scrape(sm=sm, source_name="fake", limit=10, extract_fn=fake_extract)
    assert summary.status == "completed"
    items = sm.list_research_items(source="fake")
    exs = sm.list_extractions_for_item(items[0]["id"])
    assert len(exs) == 1
    assert exs[0]["screen_status"] == "failed"
    assert exs[0]["screen_sharpe"] is None
    # Invalid DSL caught before we ever create a backtest_runs row
    assert exs[0]["screen_run_id"] is None


def test_auto_screen_backtest_run_carries_traceability(
    sm: StateManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The created backtest_runs row must point back at the extraction."""
    _register_fake_source([_item(0)])

    monkeypatch.setattr(auto_screen, "_normalize_dsl", lambda s: {"timeframe": "1h"})
    monkeypatch.setattr(auto_screen, "_run_single_sharpe", lambda *a, **k: 2.5)
    monkeypatch.setattr(auto_screen, "auto_screen_extraction", _REAL_AUTO_SCREEN)

    def fake_extract(item: RawItem, item_id: int) -> ExtractionBatch:  # noqa: ARG001
        return ExtractionBatch(
            prompt="P",
            raw_response="{}",
            results=[
                ExtractionResult(
                    status="parsed",
                    confidence=1.0,
                    rationale=None,
                    raw_response="",
                    dsl_yaml="x",
                    parsed_dsl_json='{"name":"trace_test"}',
                    parse_error=None,
                    llm_model="t",
                )
            ],
        )

    summary = run_scrape(sm=sm, source_name="fake", limit=10, extract_fn=fake_extract)
    assert summary.status == "completed"
    items = sm.list_research_items(source="fake")
    exs = sm.list_extractions_for_item(items[0]["id"])
    assert exs[0]["screen_status"] == "done"
    assert exs[0]["screen_sharpe"] == pytest.approx(2.5)
    run_id = exs[0]["screen_run_id"]
    assert run_id is not None

    run = sm.get_backtest_run(int(run_id))
    assert run is not None
    assert run["run_mode"] == "screening"
    assert run["strategy_id"] is None
    src = (run["parameters"] or {}).get("auto_screen_source")
    assert src == {"extraction_id": exs[0]["id"]}
