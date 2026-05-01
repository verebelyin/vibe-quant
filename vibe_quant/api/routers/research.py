"""Research scrape + extraction + promote endpoints."""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from vibe_quant.api.deps import get_state_manager
from vibe_quant.api.schemas.research import (
    ExtractionResponse,
    PromoteResponse,
    ResearchItemDetailResponse,
    ResearchItemListResponse,
    ResearchItemResponse,
    ScrapeRequest,
    ScrapeRunResponse,
    SourceListResponse,
)
from vibe_quant.db.state_manager import StateManager
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
        dsl_yaml=row.get("dsl_yaml"),
        parsed_dsl_json=row.get("parsed_dsl_json"),
        parse_error=row.get("parse_error"),
        strategy_id=row.get("strategy_id"),
        status=str(row.get("status") or "parsed"),
    )


@router.get("/sources", response_model=SourceListResponse)
def get_sources() -> SourceListResponse:
    load_builtin_sources()
    return SourceListResponse(sources=list_sources())


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


_VALID_SORTS = {"newest_scraped", "newest_posted", "highest_score", "highest_confidence"}


@router.get("/items", response_model=ResearchItemListResponse)
def list_items(
    sm: StateMgr,
    source: Annotated[str | None, Query()] = None,
    status: Annotated[str | None, Query()] = None,
    sort: Annotated[str, Query()] = "newest_scraped",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ResearchItemListResponse:
    if sort not in _VALID_SORTS:
        raise HTTPException(status_code=422, detail=f"invalid sort: {sort}")
    rows = sm.list_research_items(
        source=source, status=status, sort=sort, limit=limit, offset=offset
    )
    total = sm.count_research_items(source=source, status=status)
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
    return ResearchItemDetailResponse(
        **base.model_dump(),
        extractions=[_extraction_to_response(e) for e in extractions],
    )


@router.post("/items/{item_id}/extract", response_model=ExtractionResponse, status_code=201)
def extract_item(item_id: int, sm: StateMgr) -> ExtractionResponse:
    """Re-run extraction for a single item (preserves history — adds new row)."""
    item_row = sm.get_research_item(item_id)
    if not item_row:
        raise HTTPException(status_code=404, detail=f"research_item {item_id} not found")

    try:
        from vibe_quant.research.extractor import get_default_extractor
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"extractor module unavailable: {e}") from e

    try:
        extractor = get_default_extractor()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    from vibe_quant.research.archive import _row_to_raw_item

    raw = _row_to_raw_item(item_row)
    result = extractor.extract(raw)

    extraction_id = sm.create_extraction(
        research_item_id=item_id,
        status=result.status,
        llm_model=result.llm_model,
        confidence=result.confidence,
        rationale=result.rationale,
        raw_response=result.raw_response,
        dsl_yaml=result.dsl_yaml,
        parsed_dsl_json=result.parsed_dsl_json,
        parse_error=result.parse_error,
    )
    item_status_map = {"parsed": "extracted", "failed": "failed", "skipped": "skipped"}
    sm.update_research_item_status(item_id, item_status_map.get(result.status, "failed"))

    row = sm.get_extraction(extraction_id)
    assert row is not None
    return _extraction_to_response(row)


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
