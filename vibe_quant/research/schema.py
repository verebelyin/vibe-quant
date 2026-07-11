"""Domain types for the research pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from datetime import datetime

EvidenceLevel = Literal["live_traded", "backtested", "idea_only"]


@dataclass(frozen=True)
class RawItem:
    """A single source-agnostic post/comment/article fetched from an external source.

    Source-specific fields (subreddit, comments, flair, arxiv categories, ...) live
    in `extras` so that a generic archiver can persist any source without knowing
    its shape.
    """

    source: str
    external_id: str
    url: str
    title: str
    body: str
    author: str | None
    posted_at: datetime | None
    score: int | None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionResult:
    """Outcome of running the LLM extractor over a `RawItem`."""

    status: str  # parsed | failed | skipped
    confidence: float | None
    rationale: str | None
    raw_response: str
    dsl_yaml: str | None
    parsed_dsl_json: str | None
    parse_error: str | None
    llm_model: str | None
    proposed_indicators_json: str | None = None
    evidence_level: EvidenceLevel | None = None
    completeness: float | None = None


@dataclass(frozen=True)
class ExtractionBatch:
    """Bundle of one LLM call's IO + the per-finding results it produced.

    Returned by ``ClaudePExtractor.extract_all`` so callers can persist the
    full prompt + raw response to a log file alongside the parsed findings.
    """

    prompt: str
    raw_response: str
    results: list[ExtractionResult] = field(default_factory=list)
