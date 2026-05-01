"""Pydantic schemas for the research router."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field


class ScrapeRequest(BaseModel):
    source: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")]
    limit: Annotated[int, Field(ge=1, le=500)] = 50
    extract: bool = True
    subreddits: list[str] | None = None


class ScrapeRunResponse(BaseModel):
    id: int
    source: str
    status: str
    started_at: str | None
    completed_at: str | None
    items_fetched: int
    items_new: int
    items_extracted: int
    items_failed: int
    error_message: str | None
    pid: int | None
    heartbeat_at: str | None


class ExtractionResponse(BaseModel):
    id: int
    research_item_id: int
    extracted_at: str | None
    llm_model: str | None
    confidence: float | None
    rationale: str | None
    raw_response: str | None
    dsl_yaml: str | None
    parsed_dsl_json: str | None
    parse_error: str | None
    strategy_id: int | None
    status: str


class ResearchItemResponse(BaseModel):
    id: int
    source: str
    external_id: str
    url: str
    title: str | None
    body: str | None
    author: str | None
    posted_at: str | None
    score: int | None
    extras: dict | None = None
    fetched_at: str | None
    extraction_status: str


class ResearchItemDetailResponse(ResearchItemResponse):
    extractions: list[ExtractionResponse]


class ResearchItemListResponse(BaseModel):
    items: list[ResearchItemResponse]
    total: int
    limit: int
    offset: int


class PromoteResponse(BaseModel):
    strategy_id: int
    extraction_id: int


class SourceListResponse(BaseModel):
    sources: list[str]


class CredentialsStatusResponse(BaseModel):
    """Reports presence of Reddit env-var credentials. Never includes raw values."""

    source: str
    configured: bool
    missing: list[str]
    set_vars: list[str]
