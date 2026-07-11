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
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import httpx

from vibe_quant.research.config import (
    RedditConfig,
    subreddits_from_env,
    use_challenge_from_env,
)
from vibe_quant.research.schema import RawItem
from vibe_quant.research.sources import register_source
from vibe_quant.research.sources._reddit_challenge import ChallengeClient

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


class _HttpGetter(Protocol):
    """The subset of ``httpx.Client`` this source relies on."""

    def get(self, url: str, *, params: Mapping[str, str] | None = None) -> httpx.Response: ...

    def close(self) -> None: ...

logger = logging.getLogger(__name__)

BASE_URL = "https://www.reddit.com"
MIN_REQUEST_INTERVAL_S = 6.0
RETRY_BACKOFF_S = (6.0, 12.0, 24.0)
MAX_RETRIES = len(RETRY_BACKOFF_S)
RETRY_AFTER_BUFFER_S = 0.5
HTTP_TIMEOUT_S = 15.0
TOP_COMMENT_COUNT = 10
DELETED_AUTHOR_PLACEHOLDERS = frozenset({"[deleted]", "[removed]"})
KIND_POST = "t3"
KIND_COMMENT = "t1"

VALID_LISTINGS = frozenset({"new", "top", "hot", "rising"})
VALID_TIME_FILTERS = frozenset({"hour", "day", "week", "month", "year", "all"})


@register_source("reddit")
class RedditSource:
    """Read-only Reddit source over the public unauthenticated `.json` endpoint."""

    name: ClassVar[str] = "reddit"

    def __init__(
        self,
        config: RedditConfig | None = None,
        subreddits: list[str] | None = None,
        client: _HttpGetter | None = None,
        listing: str = "new",
        time_filter: str | None = None,
        use_challenge: bool | None = None,
    ) -> None:
        if listing not in VALID_LISTINGS:
            raise ValueError(f"listing must be one of {sorted(VALID_LISTINGS)}, got {listing!r}")
        if time_filter is not None and time_filter not in VALID_TIME_FILTERS:
            raise ValueError(
                f"time_filter must be one of {sorted(VALID_TIME_FILTERS)} or None, got {time_filter!r}"
            )
        if time_filter is not None and listing != "top":
            raise ValueError(f"time_filter only applies to listing='top' (got listing={listing!r})")
        self._listing = listing
        self._time_filter = time_filter
        self._config = config or RedditConfig.from_env()
        self._subreddits = subreddits if subreddits is not None else subreddits_from_env()
        use_challenge = use_challenge_from_env() if use_challenge is None else use_challenge
        if client is not None:
            self._client: _HttpGetter = client
        elif use_challenge:
            # ToS-grey stopgap until OAuth is approved — see _reddit_challenge.
            # Uses a browser User-Agent (its default) to match the spoofed TLS
            # fingerprint; the API-style config UA would defeat the impersonation.
            self._client = ChallengeClient()
        else:
            self._client = httpx.Client(
                timeout=HTTP_TIMEOUT_S,
                headers={
                    "User-Agent": self._config.user_agent,
                    "Accept": "application/json",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate",
                },
            )
        self._last_request_at: float = 0.0

    def close(self) -> None:
        self._client.close()

    def fetch(self, since: datetime | None, limit: int) -> Iterable[RawItem]:
        """Yield RawItems from each configured subreddit's `/new` listing.

        Args:
            since: If set, stop yielding once a post older than `since` is
                seen (Reddit's `/new` is reverse-chronological so this is a
                cheap early exit).
            limit: Per-subreddit cap on submissions to inspect.
        """
        since_ts = since.timestamp() if since is not None else None
        failures: list[tuple[str, httpx.HTTPError]] = []
        yielded = False
        for subreddit in self._subreddits:
            try:
                for item in self._fetch_subreddit(subreddit, since_ts, limit):
                    yielded = True
                    yield item
            except httpx.HTTPError as e:
                logger.warning("subreddit %r failed: %s", subreddit, e)
                failures.append((subreddit, e))
                continue
        # Every subreddit failed and nothing was yielded: raise so the scrape
        # run records status=failed instead of a silent completed/0-items
        # (e.g. Reddit hard-403s the whole IP — vibe-quant-ux7t0).
        if failures and not yielded and len(failures) == len(self._subreddits):
            names = ", ".join(name for name, _ in failures)
            raise RuntimeError(
                f"all {len(failures)} subreddit(s) failed ({names}): {failures[-1][1]}"
            ) from failures[-1][1]

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
            if since_ts is not None:
                created_utc = float(data.get("created_utc", 0.0))
                if created_utc < since_ts:
                    # /new is reverse-chronological → stop. /top is score-ordered
                    # so a stale post doesn't imply the rest are stale → skip one.
                    if self._listing == "new":
                        return
                    continue
            yield self._to_raw_item(data, subreddit)

    def _fetch_listing(self, subreddit: str, limit: int) -> list[dict[str, Any]] | None:
        url = f"{BASE_URL}/r/{subreddit}/{self._listing}.json"
        params = {"limit": str(limit), "raw_json": "1"}
        if self._listing == "top" and self._time_filter is not None:
            params["t"] = self._time_filter
        body = self._request(url, params)
        if body is None:
            return None
        children = (body.get("data") or {}).get("children") or []
        return list(children)

    def _fetch_comments(self, permalink: str) -> list[dict[str, Any]]:
        # Top-level only — nested replies (`data.replies`) are ignored.
        # The URL passes `sort=top&limit=N`; the local sort+slice are a defensive
        # fallback for API drift and exercised by tests.
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
        num_comments = int(data.get("num_comments") or 0)
        comments = self._fetch_comments(permalink) if permalink and num_comments > 0 else []
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
                "num_comments": num_comments,
                "comments": comments,
            },
        )

    def _request(self, url: str, params: dict[str, str]) -> Any:
        # Returns parsed JSON, or None on retry exhaustion (caller skips).
        for attempt in range(MAX_RETRIES):
            self._respect_rate_limit()
            response = self._client.get(url, params=params)
            if response.status_code == 429:
                sleep_s = _parse_retry_after(response) or RETRY_BACKOFF_S[attempt]
                logger.warning(
                    "429 on %s (attempt %d/%d); sleeping %.1fs",
                    url,
                    attempt + 1,
                    MAX_RETRIES,
                    sleep_s,
                )
                # Don't stamp _last_request_at — the Retry-After sleep covers the gap.
                time.sleep(sleep_s + RETRY_AFTER_BUFFER_S)
                continue
            response.raise_for_status()
            self._last_request_at = time.monotonic()
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
