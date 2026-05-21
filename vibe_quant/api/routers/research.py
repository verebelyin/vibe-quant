"""Research scrape + extraction + promote endpoints."""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from vibe_quant.api.deps import get_state_manager
from vibe_quant.api.schemas.research import (
    CredentialsStatusResponse,
    ExtractEnqueueResponse,
    ExtractionJobResponse,
    ExtractionQueueJobResponse,
    ExtractionQueueResponse,
    ExtractionQueueStatusResponse,
    ExtractionResponse,
    IndicatorScaffoldResponse,
    PromoteResponse,
    ResearchItemDetailResponse,
    ResearchItemListResponse,
    ResearchItemResponse,
    ScrapeRequest,
    ScrapeRunResponse,
    SourceListResponse,
    SubredditsResponse,
    SubredditsUpdateRequest,
)
from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl.indicators import indicator_registry
from vibe_quant.research.config import RedditConfig, subreddits_from_env
from vibe_quant.research.indicator_scaffold import (
    CodegenError,
    InvalidProposalError,
    ScaffoldError,
    proposed_to_spec_args,
    scaffold_full,
    suggest_alt_name,
)
from vibe_quant.research.sources import list_sources, load_builtin_sources

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/research", tags=["research"])

StateMgr = Annotated[StateManager, Depends(get_state_manager)]


def _run_to_response(row: dict[str, Any]) -> ScrapeRunResponse:
    return ScrapeRunResponse(
        id=int(row["id"]),
        source=str(row["source"]),
        status=str(row["status"]),
        started_at=row.get("started_at"),
        completed_at=row.get("completed_at"),
        items_fetched=int(row.get("items_fetched") or 0),
        items_new=int(row.get("items_new") or 0),
        items_extracted=int(row.get("items_extracted") or 0),
        items_failed=int(row.get("items_failed") or 0),
        error_message=row.get("error_message"),
        pid=row.get("pid"),
        heartbeat_at=row.get("heartbeat_at"),
    )


def _item_to_response(row: dict[str, Any]) -> ResearchItemResponse:
    extras_str = row.get("extras_json")
    extras: dict[str, Any] | None = (
        json.loads(extras_str) if isinstance(extras_str, str) else None
    )
    if "num_comments" in row and row["num_comments"] is not None:
        if extras is None:
            extras = {}
        extras.setdefault("num_comments", row["num_comments"])
    if "latest_confidence" in row and row["latest_confidence"] is not None:
        if extras is None:
            extras = {}
        extras["latest_confidence"] = row["latest_confidence"]
    return ResearchItemResponse(
        id=int(row["id"]),
        source=str(row["source"]),
        external_id=str(row["external_id"]),
        url=str(row["url"]),
        title=row.get("title"),
        body=row.get("body"),
        author=row.get("author"),
        posted_at=row.get("posted_at"),
        score=row.get("score"),
        extras=extras,
        fetched_at=row.get("fetched_at"),
        extraction_status=str(row.get("extraction_status") or "pending"),
    )


def _extraction_to_response(row: dict[str, Any]) -> ExtractionResponse:
    return ExtractionResponse(
        id=int(row["id"]),
        research_item_id=int(row["research_item_id"]),
        extracted_at=row.get("extracted_at"),
        llm_model=row.get("llm_model"),
        confidence=row.get("confidence"),
        rationale=row.get("rationale"),
        raw_response=row.get("raw_response"),
        prompt=row.get("prompt"),
        dsl_yaml=row.get("dsl_yaml"),
        parsed_dsl_json=row.get("parsed_dsl_json"),
        parse_error=row.get("parse_error"),
        proposed_indicators_json=row.get("proposed_indicators_json"),
        strategy_id=row.get("strategy_id"),
        status=str(row.get("status") or "parsed"),
        screen_sharpe=row.get("screen_sharpe"),
        screen_status=row.get("screen_status"),
        screen_run_id=row.get("screen_run_id"),
        screen_pf=row.get("screen_pf"),
        screen_max_dd=row.get("screen_max_dd"),
        screen_return=row.get("screen_return"),
        screen_trades=row.get("screen_trades"),
        screen_error=row.get("screen_error"),
        screen_completed_at=row.get("screen_completed_at"),
    )


@router.get("/sources", response_model=SourceListResponse)
def get_sources() -> SourceListResponse:
    load_builtin_sources()
    return SourceListResponse(sources=list_sources())


@router.get("/credentials/status", response_model=CredentialsStatusResponse)
def get_credentials_status(source: str = "reddit") -> CredentialsStatusResponse:
    """Report the User-Agent the Reddit source will send (never raw secrets)."""
    if source != "reddit":
        raise HTTPException(status_code=422, detail=f"unknown source '{source}'")
    cfg = RedditConfig.from_env()
    return CredentialsStatusResponse(
        source=source,
        user_agent_value=cfg.user_agent,
        using_default=cfg.using_default,
    )


@router.get("/settings/subreddits", response_model=SubredditsResponse)
def get_subreddits(sm: StateMgr) -> SubredditsResponse:
    """Return the configured subreddit list for the reddit source."""
    saved = sm.get_research_subreddits("reddit")
    if saved:
        return SubredditsResponse(source="reddit", subreddits=saved, using_default=False)
    return SubredditsResponse(
        source="reddit", subreddits=subreddits_from_env(), using_default=True
    )


@router.put("/settings/subreddits", response_model=SubredditsResponse)
def set_subreddits(body: SubredditsUpdateRequest, sm: StateMgr) -> SubredditsResponse:
    """Persist the subreddit list. Validation enforced by the request schema."""
    # Dedupe while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for s in body.subreddits:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    sm.set_research_subreddits("reddit", deduped)
    return SubredditsResponse(source="reddit", subreddits=deduped, using_default=False)


@router.delete("/settings/subreddits", response_model=SubredditsResponse)
def reset_subreddits(sm: StateMgr) -> SubredditsResponse:
    """Drop the saved row so the source falls back to env defaults."""
    sm.clear_research_subreddits("reddit")
    return SubredditsResponse(
        source="reddit", subreddits=subreddits_from_env(), using_default=True
    )


@router.post("/scrape", response_model=ScrapeRunResponse, status_code=201)
def start_scrape(body: ScrapeRequest, sm: StateMgr) -> ScrapeRunResponse:
    load_builtin_sources()
    if body.source not in list_sources():
        raise HTTPException(
            status_code=422,
            detail=f"unknown source '{body.source}'. Available: {sorted(list_sources())}",
        )

    # 409 if a scrape for this source is already running
    active = sm.list_active_scrape_runs(source=body.source)
    if active:
        raise HTTPException(
            status_code=409,
            detail=f"a scrape for '{body.source}' is already running (id={active[0]['id']})",
        )

    # Pre-create the run row so we can return its id and the subprocess
    # can adopt it. pid is None until the subprocess sets it via adopt.
    config = {"source": body.source, "limit": body.limit, "extract": body.extract}
    run_id = sm.create_scrape_run(source=body.source, pid=None, config=config)

    cmd = [
        sys.executable,
        "-m",
        "vibe_quant.research",
        "scrape",
        "--source",
        body.source,
        "--limit",
        str(body.limit),
        "--scrape-run-id",
        str(run_id),
    ]
    if not body.extract:
        cmd.append("--no-extract")

    try:
        subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=os.environ.copy(),
        )
    except OSError as e:
        sm.complete_scrape_run(run_id, status="failed", error_message=f"spawn failed: {e}")
        raise HTTPException(status_code=500, detail=f"failed to spawn scraper: {e}") from e

    row = sm.get_scrape_run(run_id)
    assert row is not None
    return _run_to_response(row)


@router.get("/scrape/latest", response_model=ScrapeRunResponse | None)
def get_latest_scrape(source: str, sm: StateMgr) -> ScrapeRunResponse | None:
    row = sm.latest_scrape_run(source)
    return _run_to_response(row) if row else None


@router.get("/scrape/{run_id}", response_model=ScrapeRunResponse)
def get_scrape(run_id: int, sm: StateMgr) -> ScrapeRunResponse:
    row = sm.get_scrape_run(run_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"scrape_run {run_id} not found")
    return _run_to_response(row)


@router.delete("/scrape/{run_id}", response_model=ScrapeRunResponse)
def kill_scrape(run_id: int, sm: StateMgr) -> ScrapeRunResponse:
    row = sm.get_scrape_run(run_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"scrape_run {run_id} not found")
    if row["status"] != "running":
        raise HTTPException(
            status_code=400,
            detail=f"scrape_run {run_id} is not running (status={row['status']})",
        )
    pid = row.get("pid")
    if isinstance(pid, int):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            # Process already gone — finalize the row defensively
            logger.warning("scrape_run %s pid %s already gone; finalizing", run_id, pid)
            sm.complete_scrape_run(run_id, status="killed")
        except PermissionError as e:
            raise HTTPException(status_code=500, detail=f"cannot signal pid {pid}: {e}") from e
    else:
        # No pid recorded — finalize directly
        sm.complete_scrape_run(run_id, status="killed")

    after = sm.get_scrape_run(run_id)
    assert after is not None
    return _run_to_response(after)


_VALID_SORTS = {
    "newest_scraped",
    "newest_posted",
    "highest_score",
    "highest_confidence",
    "screen_sharpe",
}


@router.get("/items", response_model=ResearchItemListResponse)
def list_items(
    sm: StateMgr,
    source: Annotated[str | None, Query()] = None,
    status: Annotated[str | None, Query()] = None,
    sort: Annotated[str, Query()] = "newest_scraped",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    hide_low_trade: Annotated[bool, Query()] = False,
    q: Annotated[str | None, Query(max_length=200)] = None,
) -> ResearchItemListResponse:
    if sort not in _VALID_SORTS:
        raise HTTPException(status_code=422, detail=f"invalid sort: {sort}")
    q_normalized = q.strip() if q else None
    if q_normalized == "":
        q_normalized = None
    rows = sm.list_research_items(
        source=source,
        status=status,
        sort=sort,
        limit=limit,
        offset=offset,
        hide_low_trade=hide_low_trade,
        q=q_normalized,
    )
    total = sm.count_research_items(
        source=source,
        status=status,
        hide_low_trade=hide_low_trade,
        q=q_normalized,
    )
    return ResearchItemListResponse(
        items=[_item_to_response(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/items/{item_id}", response_model=ResearchItemDetailResponse)
def get_item(item_id: int, sm: StateMgr) -> ResearchItemDetailResponse:
    row = sm.get_research_item(item_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"research_item {item_id} not found")
    extractions = sm.list_extractions_for_item(item_id)
    base = _item_to_response(row)
    latest = _latest_job_for_item(sm, item_id)
    return ResearchItemDetailResponse(
        **base.model_dump(),
        extractions=[_extraction_to_response(e) for e in extractions],
        latest_job=latest,
    )


def _latest_job_for_item(sm: StateMgr, item_id: int) -> ExtractionJobResponse | None:
    """Pull the most recent extraction job row for an item, if any."""
    rows = sm.conn.execute(
        """SELECT * FROM research_extraction_jobs
           WHERE research_item_id = ?
           ORDER BY id DESC LIMIT 1""",
        (item_id,),
    ).fetchone()
    if not rows:
        return None
    r = dict(rows)
    return ExtractionJobResponse(
        id=int(r["id"]),
        research_item_id=int(r["research_item_id"]),
        status=str(r["status"]),
        queued_at=r.get("queued_at"),
        started_at=r.get("started_at"),
        completed_at=r.get("completed_at"),
        attempts=int(r.get("attempts") or 0),
        max_attempts=int(r.get("max_attempts") or 3),
        last_error=r.get("last_error"),
        error_message=r.get("error_message"),
        heartbeat_at=r.get("heartbeat_at"),
    )


@router.post(
    "/items/{item_id}/extract",
    response_model=ExtractEnqueueResponse,
    status_code=202,
)
def extract_item(item_id: int, sm: StateMgr) -> ExtractEnqueueResponse:
    """Enqueue extraction onto the persistent queue. The
    `vibe-quant extraction-worker` process drains the queue.

    Returns immediately with the new job id; clients poll
    GET /items/{id} or GET /extraction-jobs/{id} for progress.
    """
    item_row = sm.get_research_item(item_id)
    if not item_row:
        raise HTTPException(status_code=404, detail=f"research_item {item_id} not found")
    current = item_row.get("extraction_status")
    if current in ("queued", "running"):
        raise HTTPException(
            status_code=409,
            detail=f"extraction already {current} for item {item_id}",
        )

    job_id = sm.enqueue_extraction_job(item_id)
    return ExtractEnqueueResponse(job_id=job_id, item_id=item_id, status="queued")


_QUEUE_STATUSES = {"queued", "running", "done", "failed", "cancelled"}


def _queue_job_to_response(row: dict[str, Any]) -> ExtractionQueueJobResponse:
    return ExtractionQueueJobResponse(
        id=int(row["id"]),
        research_item_id=int(row["research_item_id"]),
        status=str(row["status"]),
        queued_at=row.get("queued_at"),
        started_at=row.get("started_at"),
        completed_at=row.get("completed_at"),
        attempts=int(row.get("attempts") or 0),
        max_attempts=int(row.get("max_attempts") or 0),
        last_error=row.get("last_error"),
        error_message=row.get("error_message"),
        heartbeat_at=row.get("heartbeat_at"),
        item_title=row.get("item_title"),
        item_url=str(row.get("item_url") or ""),
        item_source=str(row.get("item_source") or ""),
    )


@router.get("/extraction-queue", response_model=ExtractionQueueResponse)
def list_extraction_queue(
    sm: StateMgr,
    status: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> ExtractionQueueResponse:
    """List active (and optionally completed) extraction jobs joined with
    parent item title + url for display on the queue page.

    `status` is a comma-separated list (e.g. ``queued,running``). Defaults
    to active jobs only (``queued,running``).
    """
    if status:
        parts = [s.strip() for s in status.split(",") if s.strip()]
        bad = [s for s in parts if s not in _QUEUE_STATUSES]
        if bad:
            raise HTTPException(status_code=422, detail=f"invalid status: {bad}")
        statuses = tuple(parts)
    else:
        statuses = ("queued", "running")
    rows = sm.list_extraction_queue(statuses=statuses, limit=limit)
    return ExtractionQueueResponse(
        jobs=[_queue_job_to_response(r) for r in rows],
        active_count=sm.count_active_extraction_jobs(),
    )


@router.get("/extraction-queue/status", response_model=ExtractionQueueStatusResponse)
def extraction_queue_status(sm: StateMgr) -> ExtractionQueueStatusResponse:
    """Compact counts for the header badge — kept cheap so the UI can poll it."""
    queued = sm.list_extraction_queue(statuses=("queued",), limit=10_000)
    running = sm.list_extraction_queue(statuses=("running",), limit=10_000)
    return ExtractionQueueStatusResponse(
        queued_count=len(queued),
        running_count=len(running),
        active_count=len(queued) + len(running),
    )


@router.post(
    "/extraction-jobs/{job_id}/cancel",
    response_model=ExtractionJobResponse,
)
def cancel_extraction_job(job_id: int, sm: StateMgr) -> ExtractionJobResponse:
    """Cancel a *queued* extraction job. Cancelling a running job is not
    supported yet (bd-ma1j) — those return 409 until that bead ships."""
    try:
        job = sm.cancel_queued_extraction_job(job_id)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    if job is None:
        raise HTTPException(status_code=404, detail=f"extraction job {job_id} not found")
    return ExtractionJobResponse(
        id=int(job["id"]),
        research_item_id=int(job["research_item_id"]),
        status=str(job["status"]),
        queued_at=job.get("queued_at"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        attempts=int(job.get("attempts") or 0),
        max_attempts=int(job.get("max_attempts") or 0),
        last_error=job.get("last_error"),
        error_message=job.get("error_message"),
        heartbeat_at=job.get("heartbeat_at"),
    )


@router.post("/extractions/{extraction_id}/promote", response_model=PromoteResponse)
def promote_extraction(extraction_id: int, sm: StateMgr) -> PromoteResponse:
    ex = sm.get_extraction(extraction_id)
    if not ex:
        raise HTTPException(status_code=404, detail=f"extraction {extraction_id} not found")

    status = ex.get("status")
    if status == "promoted":
        existing = ex.get("strategy_id")
        raise HTTPException(
            status_code=400,
            detail=f"already promoted to strategy_id={existing}",
        )
    if status == "rejected":
        raise HTTPException(
            status_code=400,
            detail="cannot promote a rejected extraction; re-extract first",
        )
    if status != "parsed":
        raise HTTPException(
            status_code=400,
            detail=f"cannot promote unparsed extraction (status={status})",
        )

    parsed_json = ex.get("parsed_dsl_json")
    if not isinstance(parsed_json, str) or not parsed_json:
        raise HTTPException(status_code=400, detail="extraction has no parsed_dsl_json")

    try:
        dsl_dict = json.loads(parsed_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"corrupt parsed_dsl_json: {e}") from e

    item = sm.get_research_item(int(ex["research_item_id"]))
    base_name = (
        f"{item['source']}_{item['external_id']}"
        if item
        else f"research_extraction_{extraction_id}"
    )
    name = base_name.lower().replace("-", "_")[:100]

    strategy_id = sm.create_strategy(name=name, dsl_config=dsl_dict)
    sm.update_extraction_status(extraction_id, status="promoted", strategy_id=strategy_id)
    return PromoteResponse(strategy_id=strategy_id, extraction_id=extraction_id)


@router.post("/extractions/{extraction_id}/reject", response_model=ExtractionResponse)
def reject_extraction(extraction_id: int, sm: StateMgr) -> ExtractionResponse:
    ex = sm.get_extraction(extraction_id)
    if not ex:
        raise HTTPException(status_code=404, detail=f"extraction {extraction_id} not found")
    if ex.get("status") == "promoted":
        raise HTTPException(
            status_code=400,
            detail=f"cannot reject already-promoted extraction (strategy_id={ex.get('strategy_id')})",
        )
    sm.update_extraction_status(extraction_id, status="rejected")
    after = sm.get_extraction(extraction_id)
    assert after is not None
    return _extraction_to_response(after)


@router.post("/extractions/{extraction_id}/rescreen", response_model=ExtractionResponse)
def rescreen_extraction(extraction_id: int, sm: StateMgr) -> ExtractionResponse:
    """Re-run the auto-screen for an extraction synchronously.

    Creates a new `backtest_runs` row (the prior one is preserved) and
    overwrites the `screen_*` columns on the extraction. Only valid for
    extractions whose DSL parsed successfully.
    """
    ex = sm.get_extraction(extraction_id)
    if not ex:
        raise HTTPException(status_code=404, detail=f"extraction {extraction_id} not found")
    parsed_json = ex.get("parsed_dsl_json")
    if not isinstance(parsed_json, str) or not parsed_json:
        raise HTTPException(
            status_code=400, detail="extraction has no parsed_dsl_json; cannot re-screen"
        )

    from vibe_quant.research.auto_screen import auto_screen_extraction

    auto_screen_extraction(sm, extraction_id, parsed_json)
    after = sm.get_extraction(extraction_id)
    assert after is not None
    return _extraction_to_response(after)


def _load_proposed_indicator(
    sm: StateManager, extraction_id: int, idx: int
) -> dict[str, Any]:
    """Resolve a single proposal from an extraction or raise 404.

    Returns the raw dict so callers can hand it straight to the mapper.
    Raises HTTPException(404) when the extraction is missing, the
    extraction has no proposed_indicators_json, or idx is out of range.
    """
    ex = sm.get_extraction(extraction_id)
    if not ex:
        raise HTTPException(status_code=404, detail=f"extraction {extraction_id} not found")
    raw = ex.get("proposed_indicators_json")
    if not isinstance(raw, str) or not raw:
        raise HTTPException(
            status_code=404,
            detail=f"extraction {extraction_id} has no proposed_indicators",
        )
    try:
        proposals = json.loads(raw)
    except json.JSONDecodeError as e:
        # Corrupt JSON on disk is a server bug, not a client error.
        raise HTTPException(
            status_code=500,
            detail=f"corrupt proposed_indicators_json on extraction {extraction_id}: {e}",
        ) from e
    if not isinstance(proposals, list) or idx < 0 or idx >= len(proposals):
        raise HTTPException(
            status_code=404,
            detail=(
                f"proposal idx {idx} out of range for extraction {extraction_id} "
                f"(have {len(proposals) if isinstance(proposals, list) else 0})"
            ),
        )
    entry = proposals[idx]
    if not isinstance(entry, dict):
        raise HTTPException(
            status_code=500,
            detail=f"proposal idx {idx} is not an object",
        )
    return entry


@router.post(
    "/extractions/{extraction_id}/indicators/{idx}/scaffold",
    response_model=IndicatorScaffoldResponse,
)
def scaffold_proposed_indicator(
    extraction_id: int,
    idx: int,
    sm: StateMgr,
    force: Annotated[bool, Query()] = False,
) -> IndicatorScaffoldResponse:
    """Scaffold a proposed indicator from an extraction into a plugin file.

    Slice 1 of bd-3p1k.1 — only the validation/cache layer is wired up.
    The success path returns ``not_implemented`` until slices 2 + 3 land
    LLM codegen, AST safety, and auto-commit. The frontend can already
    surface ``invalid_input``, ``name_collision``, and
    ``already_scaffolded`` against this endpoint.
    """
    proposal = _load_proposed_indicator(sm, extraction_id, idx)

    try:
        spec_args = proposed_to_spec_args(proposal)
    except InvalidProposalError as e:
        return IndicatorScaffoldResponse(
            status="invalid_input",
            extraction_id=extraction_id,
            idx=idx,
            error=str(e),
        )

    registered = set(indicator_registry.list_indicators())
    if spec_args.name in registered:
        return IndicatorScaffoldResponse(
            status="name_collision",
            extraction_id=extraction_id,
            idx=idx,
            name=spec_args.name,
            suggested_name=suggest_alt_name(spec_args.name, registered),
            error=f"indicator name {spec_args.name!r} is already registered",
        )

    cached = sm.get_indicator_scaffold(extraction_id, idx)
    if cached is not None and not force and cached.get("status") == "ok":
        return IndicatorScaffoldResponse(
            status="already_scaffolded",
            extraction_id=extraction_id,
            idx=idx,
            name=spec_args.name,
            plugin_path=cached.get("plugin_path"),
            test_path=cached.get("test_path"),
            commit_sha=cached.get("commit_sha"),
        )

    if force and cached is not None:
        sm.delete_indicator_scaffold(extraction_id, idx)

    formula = str(proposal.get("formula") or "").strip()
    source_quote = proposal.get("source_quote")
    if not isinstance(source_quote, str):
        source_quote = None

    try:
        result = scaffold_full(
            spec_args,
            formula=formula,
            extraction_id=extraction_id,
            source_quote=source_quote,
        )
    except CodegenError as e:
        sm.upsert_indicator_scaffold(
            extraction_id=extraction_id,
            idx=idx,
            status="codegen_failed",
            error=f"{e.code}:{e.detail}" if e.detail else e.code,
            test_output=None,
        )
        return IndicatorScaffoldResponse(
            status="codegen_failed",
            extraction_id=extraction_id,
            idx=idx,
            name=spec_args.name,
            error=f"{e.code}:{e.detail}" if e.detail else e.code,
        )
    except ScaffoldError as e:
        # Test failure and commit failure both surface as ``test_failed``
        # so the UI only has one bucket to render; the actual reason
        # lives in test_output (pytest stdout or git stderr).
        sm.upsert_indicator_scaffold(
            extraction_id=extraction_id,
            idx=idx,
            status="test_failed",
            error=e.code,
            test_output=e.output,
        )
        return IndicatorScaffoldResponse(
            status="test_failed",
            extraction_id=extraction_id,
            idx=idx,
            name=spec_args.name,
            error=e.code,
            test_output=e.output,
        )

    # Refresh the plugin loader so the freshly-written file is picked up
    # without a process restart. Done after the commit lands so a broken
    # plugin can never enter the registry.
    from vibe_quant.dsl.plugin_loader import reload_plugins

    reload_plugins()

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(Path.cwd()))
        except ValueError:
            return str(p)

    rel_plugin = _rel(result.plugin_path)
    rel_test = _rel(result.test_path)
    row = sm.upsert_indicator_scaffold(
        extraction_id=extraction_id,
        idx=idx,
        status="ok",
        plugin_path=rel_plugin,
        test_path=rel_test,
        commit_sha=result.commit_sha,
    )
    return IndicatorScaffoldResponse(
        status="ok",
        extraction_id=extraction_id,
        idx=idx,
        name=spec_args.name,
        plugin_path=row.get("plugin_path"),
        test_path=row.get("test_path"),
        commit_sha=row.get("commit_sha"),
    )
