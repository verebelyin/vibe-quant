"""Pydantic schemas for the research router."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field


class ScrapeRequest(BaseModel):
    source: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")]
    limit: Annotated[int, Field(ge=1, le=500)] = 50
    extract: bool = True


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
    proposed_indicators_json: str | None = None
    strategy_id: int | None
    status: str
    screen_sharpe: float | None = None
    screen_status: str | None = None
    screen_run_id: int | None = None
    screen_pf: float | None = None
    screen_max_dd: float | None = None
    screen_return: float | None = None
    screen_trades: int | None = None
    screen_error: str | None = None
    screen_completed_at: str | None = None


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


SUBREDDIT_NAME_PATTERN = r"^[a-z0-9_]{3,21}$"


class SubredditsResponse(BaseModel):
    """Configured subreddit list for the reddit source.

    `using_default` is True when no custom row exists in the DB and the list
    is the env-derived fallback (REDDIT_SUBREDDITS or hard-coded default).
    """

    source: str
    subreddits: list[str]
    using_default: bool


class SubredditsUpdateRequest(BaseModel):
    subreddits: Annotated[
        list[
            Annotated[
                str,
                Field(min_length=3, max_length=21, pattern=SUBREDDIT_NAME_PATTERN),
            ]
        ],
        Field(min_length=1, max_length=50),
    ]


class CredentialsStatusResponse(BaseModel):
    """Reports the User-Agent that the Reddit source will use.

    The Reddit `.json` endpoint needs no auth; only `REDDIT_USER_AGENT` matters.
    `user_agent_value` always returns the actual UA the source will send so
    the UI can show what's really on the wire.
    """

    source: str
    user_agent_value: str
    using_default: bool
