"""Hacker News source: Algolia HN Search API for stories + their top comments.

Keyless public API — no auth, no account. We query
``https://hn.algolia.com/api/v1/search_by_date`` once per configured query,
then pull each story's comment tree from ``items/<id>``. Algolia documents
~10k requests/hour per IP for the HN index, so we stay polite with a 1s
inter-request floor and bounded backoff on 429/5xx/timeouts.
"""

from __future__ import annotations

import html
import logging
import os
import re
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import httpx

from vibe_quant.research.schema import RawItem
from vibe_quant.research.sources import register_source

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

logger = logging.getLogger(__name__)

SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"
ITEMS_URL = "https://hn.algolia.com/api/v1/items/{item_id}"
HN_ITEM_URL = "https://news.ycombinator.com/item?id={item_id}"

ENV_HN_QUERIES = "HN_QUERIES"
DEFAULT_QUERIES = (
    "algorithmic trading",
    "trading strategy",
    "backtest",
    "quant trading",
    "crypto trading",
)

MIN_REQUEST_INTERVAL_S = 1.0
RETRY_BACKOFF_S = (1.0, 2.0, 4.0)
HTTP_TIMEOUT_S = 15.0
MAX_HITS_PER_PAGE = 100
MAX_COMMENTS = 10
TAGS_STORY = "story"
USER_AGENT = "vibe-quant-research/0.1 (HN Algolia API)"

_TAG_RE = re.compile(r"<[^>]+>")
# Block-level tags become newlines before the generic strip: HN writes
# paragraphs as a bare `<p>` and line breaks as `<br>`, and dropping them
# would fuse two paragraphs into one run-on line.
_BLOCK_TAG_RE = re.compile(
    r"<\s*/?\s*(?:br|p|div|li|tr|h[1-6]|blockquote|pre)\b[^>]*>", re.IGNORECASE
)
_HORIZONTAL_SPACE_RE = re.compile(r"[^\S\n]+")


class _HttpGetter(Protocol):
    """The subset of ``httpx.Client`` this source relies on."""

    def get(self, url: str, *, params: Mapping[str, str] | None = None) -> httpx.Response: ...

    def close(self) -> None: ...


def queries_from_env() -> tuple[str, ...]:
    """Parse ``HN_QUERIES`` (comma-separated); fall back to DEFAULT_QUERIES."""
    raw = os.getenv(ENV_HN_QUERIES)
    if raw is None:
        return DEFAULT_QUERIES
    queries = tuple(q.strip() for q in raw.split(",") if q.strip())
    return queries or DEFAULT_QUERIES


@register_source("hackernews")
class HackerNewsSource:
    """Read-only Hacker News source over the keyless Algolia search API."""

    name: ClassVar[str] = "hackernews"

    def __init__(
        self,
        queries: list[str] | None = None,
        client: _HttpGetter | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._queries = tuple(queries) if queries is not None else queries_from_env()
        self._client: _HttpGetter = client or httpx.Client(
            timeout=HTTP_TIMEOUT_S, headers={"User-Agent": USER_AGENT}
        )
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._last_request_at: float | None = None

    def close(self) -> None:
        self._client.close()

    def fetch(self, since: datetime | None, limit: int) -> Iterable[RawItem]:
        """Yield stories matching each configured query (deduped across queries).

        Args:
            since: If set, skip stories published before this instant.
            limit: Per-query cap on hits requested (clamped to the API's 100).
        """
        since_ts = since.timestamp() if since is not None else None
        failures: list[tuple[str, httpx.HTTPError]] = []
        seen: set[str] = set()
        yielded = False
        for query in self._queries:
            try:
                for item in self._fetch_query(query, since_ts, limit, seen):
                    yielded = True
                    yield item
            except httpx.HTTPError as e:
                logger.warning("HN query %r failed: %s", query, e)
                failures.append((query, e))
                continue
        # Every query failed and nothing was yielded: raise so the scrape run
        # records status=failed instead of a silent completed/0-items (same
        # rule as reddit.py — HN/Algolia can hard-throttle the whole IP).
        if failures and not yielded and len(failures) == len(self._queries):
            names = ", ".join(name for name, _ in failures)
            raise RuntimeError(
                f"all {len(failures)} HN query/queries failed ({names}): {failures[-1][1]}"
            ) from failures[-1][1]

    def _fetch_query(
        self, query: str, since_ts: float | None, limit: int, seen: set[str]
    ) -> Iterable[RawItem]:
        for hit in self._search(query, limit):
            object_id = str(hit.get("objectID") or "")
            if not object_id or object_id in seen:
                continue
            if since_ts is not None and float(hit.get("created_at_i") or 0.0) < since_ts:
                continue
            seen.add(object_id)
            yield self._to_raw_item(hit, query)

    def _search(self, query: str, limit: int) -> list[dict[str, Any]]:
        params = {
            "query": query,
            "tags": TAGS_STORY,
            "hitsPerPage": str(min(limit, MAX_HITS_PER_PAGE)),
        }
        body = self._request(SEARCH_URL, params)
        return list(body.get("hits") or [])

    def _fetch_comments(self, object_id: str) -> list[dict[str, Any]]:
        # Best-effort: a story is still useful when its comment thread fails,
        # so a failed items call is logged and treated as "no comments".
        try:
            body = self._request(ITEMS_URL.format(item_id=object_id), None)
        except httpx.HTTPError as e:
            logger.warning("HN comments for item %s failed: %s", object_id, e)
            return []
        comments: list[dict[str, Any]] = []
        for child in body.get("children") or []:
            text = _strip_html(child.get("text"))
            if not text:
                continue
            comments.append({"author": child.get("author") or None, "body": text, "score": None})
            if len(comments) >= MAX_COMMENTS:
                break
        return comments

    def _to_raw_item(self, hit: dict[str, Any], query: str) -> RawItem:
        object_id = str(hit.get("objectID") or "")
        hn_url = HN_ITEM_URL.format(item_id=object_id)
        num_comments = int(hit.get("num_comments") or 0)
        comments = self._fetch_comments(object_id) if num_comments > 0 else []
        return RawItem(
            source="hackernews",
            external_id=object_id,
            url=str(hit.get("url") or "") or hn_url,
            title=str(hit.get("title") or ""),
            body=_strip_html(hit.get("story_text")),
            author=hit.get("author") or None,
            posted_at=datetime.fromtimestamp(float(hit.get("created_at_i") or 0.0), tz=UTC),
            score=int(hit.get("points") or 0),
            extras={
                "query": query,
                "num_comments": num_comments,
                "comments": comments,
                "hn_url": hn_url,
            },
        )

    def _request(self, url: str, params: Mapping[str, str] | None) -> Any:
        last_exc: httpx.HTTPError | None = None
        for attempt in range(len(RETRY_BACKOFF_S) + 1):
            self._respect_rate_limit()
            try:
                response = self._client.get(url, params=params)
                response.raise_for_status()
                return response.json()
            except ValueError as e:
                # A 200 with a non-JSON body (proxy page, corrupt cache)
                # raises ValueError, which the per-query handler does not
                # catch. Wrap it as an httpx error so it fails that one query.
                raise httpx.DecodingError(f"non-JSON response from {url}: {e}") from e
            except httpx.HTTPError as e:
                last_exc = e
                if not _is_retryable(e):
                    raise
            finally:
                # Count every attempt, success or failure, so the 1s floor
                # cannot be bypassed by an endpoint that keeps erroring.
                self._last_request_at = self._clock()
            if attempt < len(RETRY_BACKOFF_S):
                backoff = RETRY_BACKOFF_S[attempt]
                logger.warning(
                    "retryable error on %s (attempt %d/%d); sleeping %.1fs",
                    url,
                    attempt + 1,
                    len(RETRY_BACKOFF_S) + 1,
                    backoff,
                )
                self._sleep(backoff)
        assert last_exc is not None
        raise last_exc

    def _respect_rate_limit(self) -> None:
        if self._last_request_at is None:
            return
        elapsed = self._clock() - self._last_request_at
        if elapsed < MIN_REQUEST_INTERVAL_S:
            self._sleep(MIN_REQUEST_INTERVAL_S - elapsed)


def _is_retryable(exc: httpx.HTTPError) -> bool:
    """429 / 5xx / transport (timeout, connection) errors are worth retrying."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return isinstance(exc, httpx.TransportError)


def _strip_html(value: object) -> str:
    """Strip HTML to text, keeping block boundaries as newlines.

    Block tags (``<p>``, ``<br>``, ...) become newlines, remaining tags are
    dropped, entities are unescaped, then each line is collapsed to single
    spaces and blank lines are removed.
    """
    if not value:
        return ""
    text = _BLOCK_TAG_RE.sub("\n", str(value))
    text = html.unescape(_TAG_RE.sub("", text))
    lines = (_HORIZONTAL_SPACE_RE.sub(" ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)
