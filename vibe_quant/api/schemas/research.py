"""Pydantic schemas for the research router."""

from __future__ import annotations

from typing import Annotated, Literal

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


class IndicatorScaffoldRow(BaseModel):
    """One ``research_indicator_scaffolds`` row, keyed by ``(extraction_id, idx)``.

    Surfaced inside ``ExtractionResponse.scaffolds`` so the UI has the
    full per-proposal status on first paint — no extra round-trip.
    """

    idx: int
    status: str
    plugin_path: str | None = None
    test_path: str | None = None
    commit_sha: str | None = None
    error: str | None = None
    test_output: str | None = None
    updated_at: str | None = None


class ExtractionResponse(BaseModel):
    id: int
    research_item_id: int
    extracted_at: str | None
    llm_model: str | None
    confidence: float | None
    evidence_level: Literal["live_traded", "backtested", "idea_only"] | None
    completeness: float | None
    rationale: str | None
    raw_response: str | None
    prompt: str | None = None
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
    scaffolds: list[IndicatorScaffoldRow] = Field(default_factory=list)


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
    extras: dict[str, object] | None = None
    fetched_at: str | None
    extraction_status: str


class ExtractionJobResponse(BaseModel):
    """A row from `research_extraction_jobs` — surfaced for failure diagnostics."""

    id: int
    research_item_id: int
    status: str
    queued_at: str | None
    started_at: str | None
    completed_at: str | None
    attempts: int
    max_attempts: int
    last_error: str | None
    error_message: str | None
    heartbeat_at: str | None


class ExtractionQueueJobResponse(ExtractionJobResponse):
    """Queue-page view: includes parent item title + url for display."""

    item_title: str | None
    item_url: str
    item_source: str


class ExtractionQueueResponse(BaseModel):
    jobs: list[ExtractionQueueJobResponse]
    active_count: int


class ExtractionQueueStatusResponse(BaseModel):
    """Compact response for the header badge: just the live counts."""

    active_count: int
    queued_count: int
    running_count: int


class ResearchItemDetailResponse(ResearchItemResponse):
    extractions: list[ExtractionResponse]
    latest_job: ExtractionJobResponse | None = None


class ResearchItemListResponse(BaseModel):
    items: list[ResearchItemResponse]
    total: int
    limit: int
    offset: int


class PromoteResponse(BaseModel):
    strategy_id: int
    extraction_id: int


class ExtractEnqueueResponse(BaseModel):
    """Returned by POST /items/{id}/extract once the job is on the queue."""

    job_id: int
    item_id: int
    status: str


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


class IndicatorScaffoldResponse(BaseModel):
    """Response from POST /api/research/extractions/{id}/indicators/{idx}/scaffold.

    Success bodies only: ``status`` is ``ok`` or ``already_scaffolded``
    (idempotent re-scaffold), both HTTP 200. Failures map to real codes
    — invalid_input → 400, name_collision → 409, codegen_failed /
    test_failed → 422 — and carry the same fields (``suggested_name``,
    ``error``, ``test_output``) inside the standard ``{"detail": {...}}``
    error body so the UI loses nothing. Test- AND commit-stage failures
    both surface as ``test_failed``; the reason lives in ``test_output``
    (pytest stdout or git stderr).
    """

    status: str
    extraction_id: int
    idx: int
    name: str | None = None
    suggested_name: str | None = None
    plugin_path: str | None = None
    test_path: str | None = None
    commit_sha: str | None = None
    error: str | None = None
    test_output: str | None = None


class PromoteIndicatorResponse(BaseModel):
    """Response from POST /api/research/indicators/{name}/promote.

    Only the success path returns this body (``status`` is always
    ``ok``). Failures now map to real HTTP codes — invalid name → 400,
    no proposed file → 404, collision → 409, write/commit failure → 500
    — surfaced via the standard ``{"detail": ...}`` error body, not this
    model. ``bd remember`` failures stay non-fatal: the promote still
    succeeds (200) and the outcome is reported in ``bd_remember_ok``.
    """

    status: str
    name: str
    plugin_path: str | None = None
    commit_sha: str | None = None
    bd_remember_ok: bool = False
    bd_remember_output: str | None = None
    error: str | None = None


class CredentialsStatusResponse(BaseModel):
    """Reports the User-Agent that the Reddit source will use.

    The Reddit `.json` endpoint needs no auth; only `REDDIT_USER_AGENT` matters.
    `user_agent_value` always returns the actual UA the source will send so
    the UI can show what's really on the wire.
    """

    source: str
    user_agent_value: str
    using_default: bool
