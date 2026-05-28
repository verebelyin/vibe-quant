"""Pipeline orchestrator tests with fake sources (no network)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.research import auto_screen, extraction_log
from vibe_quant.research.pipeline import persist_extractions, run_scrape
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


def test_persist_extractions_stores_prompt_on_db_row(sm: StateManager) -> None:
    """Per-extraction prompt must round-trip through research_extractions
    so the UI can show what was actually sent to the LLM (bd-gf7w)."""
    _register_fake_source([_item(0)])

    def fake_extract(item: RawItem, item_id: int) -> ExtractionBatch:  # noqa: ARG001
        return ExtractionBatch(
            prompt="PROMPT-UNDER-TEST",
            raw_response='{"out": 0}',
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

    run_scrape(sm=sm, source_name="fake", limit=1, extract_fn=fake_extract)
    rows = sm.conn.execute(
        "SELECT prompt FROM research_extractions ORDER BY id"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["prompt"] == "PROMPT-UNDER-TEST"


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
    monkeypatch.setattr(auto_screen, "_run_single_metrics", _blow_up)
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

    from vibe_quant.screening.types import BacktestMetrics

    monkeypatch.setattr(auto_screen, "_normalize_dsl", lambda s: {"timeframe": "1h"})
    monkeypatch.setattr(
        auto_screen,
        "_run_single_metrics",
        lambda *a, **k: BacktestMetrics(
            parameters={},
            sharpe_ratio=2.5,
            profit_factor=1.8,
            max_drawdown=0.05,
            total_return=0.42,
            total_trades=128,
        ),
    )
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
    assert exs[0]["screen_pf"] == pytest.approx(1.8)
    assert exs[0]["screen_max_dd"] == pytest.approx(0.05)
    assert exs[0]["screen_return"] == pytest.approx(0.42)
    assert exs[0]["screen_trades"] == 128
    assert exs[0]["screen_error"] is None
    assert exs[0]["screen_completed_at"] is not None
    run_id = exs[0]["screen_run_id"]
    assert run_id is not None

    run = sm.get_backtest_run(int(run_id))
    assert run is not None
    assert run["run_mode"] == "screening"
    assert run["strategy_id"] is None
    src = (run["parameters"] or {}).get("auto_screen_source")
    assert src == {"extraction_id": exs[0]["id"]}


def test_auto_screen_records_compile_error(
    sm: StateManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DSL compile failure populates screen_error + screen_completed_at and
    leaves screen_run_id None (caught before backtest_runs row created)."""
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
                    parsed_dsl_json='not-valid-json',
                    parse_error=None,
                    llm_model="t",
                )
            ],
        )

    summary = run_scrape(sm=sm, source_name="fake", limit=10, extract_fn=fake_extract)
    assert summary.status == "completed"
    exs = sm.list_extractions_for_item(
        sm.list_research_items(source="fake")[0]["id"]
    )
    assert exs[0]["screen_status"] == "failed"
    assert exs[0]["screen_run_id"] is None
    assert exs[0]["screen_error"] is not None
    assert "DSL invalid" in exs[0]["screen_error"]
    assert exs[0]["screen_completed_at"] is not None


def test_auto_screen_low_trades_still_done(
    sm: StateManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trades<50 must still be screen_status='done' (not 'failed') — UI
    handles the not-a-winner styling."""
    _register_fake_source([_item(0)])

    from vibe_quant.screening.types import BacktestMetrics

    monkeypatch.setattr(auto_screen, "_normalize_dsl", lambda s: {"timeframe": "1h"})
    monkeypatch.setattr(
        auto_screen,
        "_run_single_metrics",
        lambda *a, **k: BacktestMetrics(
            parameters={},
            sharpe_ratio=2.0,
            total_trades=12,
        ),
    )
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
                    parsed_dsl_json='{"name":"low"}',
                    parse_error=None,
                    llm_model="t",
                )
            ],
        )

    summary = run_scrape(sm=sm, source_name="fake", limit=10, extract_fn=fake_extract)
    assert summary.status == "completed"
    exs = sm.list_extractions_for_item(
        sm.list_research_items(source="fake")[0]["id"]
    )
    assert exs[0]["screen_status"] == "done"
    assert exs[0]["screen_trades"] == 12
    assert exs[0]["screen_sharpe"] == pytest.approx(2.0)


def test_auto_screen_timeout_marks_failed_with_timeout_error(
    sm: StateManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A screening run that doesn't return within the wall-clock budget must
    be killed and recorded as 'timeout'."""
    _register_fake_source([_item(0)])

    from vibe_quant.research.auto_screen import _ScreenTimeout

    def _hang(*_a: object, **_k: object) -> None:
        raise _ScreenTimeout("screening exceeded 300s")

    monkeypatch.setattr(auto_screen, "_normalize_dsl", lambda s: {"timeframe": "1h"})
    monkeypatch.setattr(auto_screen, "_run_single_metrics", _hang)
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
                    parsed_dsl_json='{"name":"slow"}',
                    parse_error=None,
                    llm_model="t",
                )
            ],
        )

    summary = run_scrape(sm=sm, source_name="fake", limit=10, extract_fn=fake_extract)
    assert summary.status == "completed"
    exs = sm.list_extractions_for_item(
        sm.list_research_items(source="fake")[0]["id"]
    )
    assert exs[0]["screen_status"] == "failed"
    assert exs[0]["screen_error"] == "timeout"
    # backtest_run row was created before the timeout — useful for triage
    assert exs[0]["screen_run_id"] is not None
    assert exs[0]["screen_completed_at"] is not None


def test_auto_screen_failure_in_one_extraction_does_not_block_others(
    sm: StateManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DSL-compile failure on finding #1 must NOT prevent finding #2 from
    being screened — both rows persist with correct statuses."""
    _register_fake_source([_item(0)])

    from vibe_quant.screening.types import BacktestMetrics

    monkeypatch.setattr(
        auto_screen,
        "_run_single_metrics",
        lambda *a, **k: BacktestMetrics(
            parameters={}, sharpe_ratio=1.5, total_trades=80
        ),
    )
    # _normalize_dsl: raise for the bad payload, pass-through for the good one
    real_normalize = auto_screen._normalize_dsl

    def _selective_normalize(payload: str) -> dict[str, object]:
        if "good" in payload:
            return {"timeframe": "1h"}
        return real_normalize(payload)

    monkeypatch.setattr(auto_screen, "_normalize_dsl", _selective_normalize)
    monkeypatch.setattr(auto_screen, "auto_screen_extraction", _REAL_AUTO_SCREEN)

    def fake_extract(item: RawItem, item_id: int) -> ExtractionBatch:  # noqa: ARG001
        return ExtractionBatch(
            prompt="P",
            raw_response="{}",
            results=[
                ExtractionResult(
                    status="parsed",
                    confidence=0.9,
                    rationale=None,
                    raw_response="",
                    dsl_yaml="x",
                    parsed_dsl_json='garbage-not-json',
                    parse_error=None,
                    llm_model="t",
                ),
                ExtractionResult(
                    status="parsed",
                    confidence=0.9,
                    rationale=None,
                    raw_response="",
                    dsl_yaml="y",
                    parsed_dsl_json='{"name":"good"}',
                    parse_error=None,
                    llm_model="t",
                ),
            ],
        )

    summary = run_scrape(sm=sm, source_name="fake", limit=10, extract_fn=fake_extract)
    assert summary.status == "completed"
    exs = sm.list_extractions_for_item(
        sm.list_research_items(source="fake")[0]["id"]
    )
    # Two extraction rows, one done, one failed
    statuses = sorted(e["screen_status"] for e in exs)
    assert statuses == ["done", "failed"]
    done = next(e for e in exs if e["screen_status"] == "done")
    failed = next(e for e in exs if e["screen_status"] == "failed")
    assert done["screen_sharpe"] == pytest.approx(1.5)
    assert failed["screen_run_id"] is None
    assert failed["screen_error"] is not None


# --- Re-extract / re-scrape idempotency (vibe-quant-w5az0) ---------------------
# l685's parent AC "kill scrape mid-screening -> next scrape leaves pending rows
# alone (idempotent)" assumed a screen_status='pending' queue that was never
# built — auto_screen runs synchronously inline in persist_extractions. These
# tests are the documented retirement of that obsolete AC: they prove the
# synchronous design has no equivalent hazard (no duplicate/orphaned screen rows,
# no wedged extraction) on the two paths that could re-touch a screened item.


def _screen_runs_by_extraction(sm: StateManager) -> dict[int, list[int]]:
    """Map extraction_id -> screening backtest_run ids tagged with it via
    ``parameters.auto_screen_source``. A correct synchronous auto_screen leaves
    exactly one run id per parsed extraction; >1 in a bucket is a double-screen,
    an unexpected bucket is a mislabeled/orphaned run."""
    buckets: dict[int, list[int]] = {}
    for run in sm.list_backtest_runs():
        if run["run_mode"] != "screening":
            continue
        src = (run["parameters"] or {}).get("auto_screen_source")
        if not isinstance(src, dict) or src.get("extraction_id") is None:
            continue
        buckets.setdefault(int(src["extraction_id"]), []).append(int(run["id"]))
    return buckets


def _parsed_with_dsl() -> ExtractionResult:
    return ExtractionResult(
        status="parsed",
        confidence=1.0,
        rationale=None,
        raw_response="",
        dsl_yaml="name: x\n",
        parsed_dsl_json='{"name":"x"}',
        parse_error=None,
        llm_model="t",
    )


def _restore_hermetic_screen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore the real auto_screen (the autouse fixture no-ops it) but keep it
    hermetic: skip DSL compile + NT backtest, return fixed 'done' metrics. A
    real backtest_runs row is still created + back-tagged — that's what we
    count."""
    from vibe_quant.screening.types import BacktestMetrics

    monkeypatch.setattr(auto_screen, "_normalize_dsl", lambda s: {"timeframe": "1h"})
    monkeypatch.setattr(
        auto_screen,
        "_run_single_metrics",
        lambda *a, **k: BacktestMetrics(
            parameters={},
            sharpe_ratio=2.0,
            profit_factor=1.5,
            max_drawdown=0.04,
            total_return=0.3,
            total_trades=99,
        ),
    )
    monkeypatch.setattr(auto_screen, "auto_screen_extraction", _REAL_AUTO_SCREEN)


def test_reextract_creates_fresh_screen_run_and_preserves_prior(
    sm: StateManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-extracting an already-screened item INSERTs a new extraction row with
    its own fresh screen run; the prior extraction's backtest_runs row is
    preserved (l685.3) and there is exactly one screen run per extraction — no
    duplicate/orphaned rows, no wedge (vibe-quant-w5az0 AC#1/#3)."""
    _restore_hermetic_screen(monkeypatch)
    item_id = sm.create_research_item(
        source="fake",
        external_id="e0",
        url="u",
        title="t",
        body="b",
        author=None,
        posted_at=None,
        score=0,
    )

    # First extraction (E1 + run R1) — the "already-screened" baseline.
    persist_extractions(sm, item_id, [_parsed_with_dsl()])
    after_first = sm.list_extractions_for_item(item_id)
    assert len(after_first) == 1
    e1 = after_first[0]
    r1 = e1["screen_run_id"]
    assert e1["screen_status"] == "done"
    assert r1 is not None

    # Re-extract the same item (E2 + run R2).
    persist_extractions(sm, item_id, [_parsed_with_dsl()])
    after_second = sm.list_extractions_for_item(item_id)
    assert len(after_second) == 2, "re-extract must INSERT a new extraction row"
    e2 = next(e for e in after_second if e["id"] != e1["id"])
    r2 = e2["screen_run_id"]
    assert r2 is not None
    assert r2 != r1, "re-extract must create a fresh screen run, not reuse R1"

    # Prior run preserved (l685.3) — re-extract never deletes/overwrites it.
    assert sm.get_backtest_run(int(r1)) is not None
    assert sm.get_backtest_run(int(r2)) is not None

    # Exactly one screening run per extraction, each correctly back-tagged:
    # no double-screen (no bucket of len>1), no orphan (no extra bucket).
    assert _screen_runs_by_extraction(sm) == {e1["id"]: [int(r1)], e2["id"]: [int(r2)]}


def test_rescrape_over_extracted_items_does_not_double_screen(
    sm: StateManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running a scrape over items that already have extractions does NOT
    re-extract or re-screen them — items dedup on (source, external_id), so the
    screen-run set is unchanged by the second scrape (vibe-quant-w5az0 AC#2:
    intended behavior, asserted explicitly)."""
    _restore_hermetic_screen(monkeypatch)
    _register_fake_source([_item(i) for i in range(2)])

    def fake_extract(item: RawItem, item_id: int) -> ExtractionBatch:  # noqa: ARG001
        return ExtractionBatch(prompt="P", raw_response="{}", results=[_parsed_with_dsl()])

    first = run_scrape(sm=sm, source_name="fake", limit=10, extract_fn=fake_extract)
    assert first.items_new == 2
    assert first.items_extracted == 2
    runs_after_first = _screen_runs_by_extraction(sm)
    assert len(runs_after_first) == 2  # one screen run per item's extraction

    # Second scrape, same source + items → dedup, no re-extract, no re-screen.
    second = run_scrape(sm=sm, source_name="fake", limit=10, extract_fn=fake_extract)
    assert second.items_new == 0
    assert second.items_extracted == 0

    for it in sm.list_research_items(source="fake"):
        assert len(sm.list_extractions_for_item(it["id"])) == 1
    assert _screen_runs_by_extraction(sm) == runs_after_first, (
        "2nd scrape must not create new/duplicate screen runs"
    )
