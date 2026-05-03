"""Reddit source: scrape r/<subreddit>/new + top comments via the public .json endpoint.

No auth, no app, no account — just polite use of the long-standing
`https://www.reddit.com/r/<sub>/new.json` and `https://www.reddit.com<permalink>.json`
URLs that Reddit has served unauthenticated for 15+ years.

Rate limit: ~10 req/min for unauthenticated traffic per IP. We enforce a 6s
inter-request floor and respect Retry-After on 429.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

import httpx

from vibe_quant.research.config import RedditConfig, subreddits_from_env
from vibe_quant.research.schema import RawItem
from vibe_quant.research.sources import register_source

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

BASE_URL = "https://www.reddit.com"
MIN_REQUEST_INTERVAL_S = 6.0
MAX_RETRIES = 3
RETRY_BACKOFF_S = (6.0, 12.0, 24.0)
RETRY_AFTER_BUFFER_S = 0.5
HTTP_TIMEOUT_S = 15.0
TOP_COMMENT_COUNT = 10
DELETED_AUTHOR_PLACEHOLDERS = frozenset({"[deleted]", "[removed]"})
KIND_POST = "t3"
KIND_COMMENT = "t1"


@register_source("reddit")
class RedditSource:
    """Read-only Reddit source over the public unauthenticated `.json` endpoint."""

    name: ClassVar[str] = "reddit"

    def __init__(
        self,
        config: RedditConfig | None = None,
        subreddits: list[str] | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config or RedditConfig.from_env()
        self._subreddits = subreddits if subreddits is not None else subreddits_from_env()
        self._client = client or httpx.Client(
            timeout=HTTP_TIMEOUT_S,
            headers={
                "User-Agent": self._config.user_agent,
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate",
            },
        )
        self._last_request_at: float = 0.0

    def fetch(self, since: datetime | None, limit: int) -> Iterable[RawItem]:
        """Yield RawItems from each configured subreddit's `/new` listing.

        Args:
            since: If set, stop yielding once a post older than `since` is
                seen (Reddit's `/new` is reverse-chronological so this is a
                cheap early exit).
            limit: Per-subreddit cap on submissions to inspect.
        """
        since_ts = since.timestamp() if since is not None else None
        for subreddit in self._subreddits:
            try:
                yield from self._fetch_subreddit(subreddit, since_ts, limit)
            except httpx.HTTPError as e:
                logger.warning("subreddit %r failed: %s", subreddit, e)
                continue

    def _fetch_subreddit(
        self, subreddit: str, since_ts: float | None, limit: int
    ) -> Iterable[RawItem]:
        listing = self._fetch_listing(subreddit, limit)
        if listing is None:
            return
        for child in listing:
            if child.get("kind") != KIND_POST:
                continue
            data = child.get("data") or {}
            try:
                if since_ts is not None and float(data.get("created_utc", 0.0)) < since_ts:
                    return  # /new is reverse-chronological — stop early
                yield self._to_raw_item(data, subreddit)
            except httpx.HTTPError as e:
                logger.warning(
                    "submission %s in r/%s skipped: %s",
                    data.get("id", "?"),
                    subreddit,
                    e,
                )
                continue

    def _fetch_listing(self, subreddit: str, limit: int) -> list[dict[str, Any]] | None:
        url = f"{BASE_URL}/r/{subreddit}/new.json"
        params = {"limit": str(limit), "raw_json": "1"}
        body = self._request(url, params)
        if body is None:
            return None
        children = (body.get("data") or {}).get("children") or []
        return list(children)

    def _fetch_comments(self, permalink: str) -> list[dict[str, Any]]:
        """Return up to `TOP_COMMENT_COUNT` top-level comments by score.

        Reddit's `<permalink>.json` returns `[post_listing, comment_listing]`.
        We keep top-level only — nested replies (`data.replies`) are ignored
        to match the prior praw behavior.
        """
        url = f"{BASE_URL}{permalink.rstrip('/')}.json"
        params = {"limit": str(TOP_COMMENT_COUNT), "sort": "top", "raw_json": "1"}
        body = self._request(url, params)
        if body is None or not isinstance(body, list) or len(body) < 2:
            return []
        comment_listing = body[1] or {}
        children = (comment_listing.get("data") or {}).get("children") or []
        comments: list[dict[str, Any]] = []
        for c in children:
            if c.get("kind") != KIND_COMMENT:
                continue  # skip "more" placeholders
            data = c.get("data") or {}
            comments.append(
                {
                    "author": _normalize_author(data.get("author")),
                    "body": str(data.get("body") or ""),
                    "score": int(data.get("score") or 0),
                }
            )
        comments.sort(key=lambda c: c["score"], reverse=True)
        return comments[:TOP_COMMENT_COUNT]

    def _to_raw_item(self, data: dict[str, Any], subreddit: str) -> RawItem:
        permalink = str(data.get("permalink") or "")
        comments = self._fetch_comments(permalink) if permalink else []
        return RawItem(
            source="reddit",
            external_id=str(data.get("id") or ""),
            url=f"{BASE_URL}{permalink}",
            title=str(data.get("title") or ""),
            body=str(data.get("selftext") or ""),
            author=_normalize_author(data.get("author")),
            posted_at=datetime.fromtimestamp(float(data.get("created_utc") or 0.0), tz=UTC),
            score=int(data.get("score") or 0),
            extras={
                "subreddit": subreddit,
                "flair": data.get("link_flair_text"),
                "num_comments": int(data.get("num_comments") or 0),
                "comments": comments,
            },
        )

    def _request(self, url: str, params: dict[str, str]) -> Any:
        """GET with rate-limit floor + 429/Retry-After + bounded retries.

        Returns the parsed JSON body on success, or None when all retries are
        exhausted (caller treats None as "skip this fetch and continue").
        """
        for attempt in range(MAX_RETRIES):
            self._respect_rate_limit()
            response = self._client.get(url, params=params)
            self._last_request_at = time.monotonic()
            if response.status_code == 429:
                sleep_s = _parse_retry_after(response) or RETRY_BACKOFF_S[attempt]
                logger.warning(
                    "429 on %s (attempt %d/%d); sleeping %.1fs",
                    url,
                    attempt + 1,
                    MAX_RETRIES,
                    sleep_s,
                )
                time.sleep(sleep_s + RETRY_AFTER_BUFFER_S)
                continue
            response.raise_for_status()
            return response.json()
        logger.warning("exhausted %d retries on %s — skipping", MAX_RETRIES, url)
        return None

    def _respect_rate_limit(self) -> None:
        if self._last_request_at == 0.0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < MIN_REQUEST_INTERVAL_S:
            time.sleep(MIN_REQUEST_INTERVAL_S - elapsed)


def _normalize_author(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value)
    if s in DELETED_AUTHOR_PLACEHOLDERS or not s:
        return None
    return s


def _parse_retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
