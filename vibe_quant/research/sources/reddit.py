"""Reddit research source backed by the free Arctic Shift API.

Reddit 403s unauthenticated ``www.reddit.com/.../*.json`` access (2026-05), so
this source reads posts + top comments from the community-run Arctic Shift
archive at ``https://arctic-shift.photon-reddit.com`` instead. No account, no
API key, no browser spoofing.

Arctic Shift is best-effort with "no uptime or performance guarantees": a
couple of requests per second is fine, the limits are dynamic, and a 429 reply
carries ``X-RateLimit-Reset`` (seconds until reset). Overloaded queries answer
with a non-payload body carrying a timeout error (200 ``Query timed out``, or
422 ``Timeout. Maybe slow down a bit``) — those are retried with backoff, as is
a 2xx body that is not valid JSON. Any valid JSON body is a real payload and is
never text-scanned for those markers: an object is judged only by its ``error``
field (a ``data`` list always wins), and a bare list/number/string/null is
accepted (RULING A11b).

Freshness: a new post's ``score``/``num_comments`` stay 0/1 for roughly 36 h
(README), so one pass only asks for posts older than ``SETTLE_DELAY_S`` (36 h)
via the ``before`` parameter — that single pass then sees settled scores, the
comment count and the comments themselves.
"""

from __future__ import annotations

import html
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar, Protocol
from urllib.parse import urlparse

import httpx

from vibe_quant.research.config import RedditConfig, subreddits_from_env
from vibe_quant.research.schema import RawItem
from vibe_quant.research.sources import register_source

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping


class _HttpGetter(Protocol):
    """The subset of ``httpx.Client`` this source relies on."""

    def get(self, url: str, *, params: Mapping[str, str] | None = None) -> httpx.Response: ...

    def close(self) -> None: ...


class ArcticShiftError(httpx.HTTPError):
    """A retryable Arctic Shift failure (5xx, transport error, timeout body)."""


logger = logging.getLogger(__name__)

API_BASE_URL = "https://arctic-shift.photon-reddit.com"
API_HOST = "arctic-shift.photon-reddit.com"
POSTS_PATH = "/api/posts/search"
COMMENTS_PATH = "/api/comments/search"
#: Permalinks in ``RawItem.url`` always point at the canonical Reddit post.
REDDIT_URL_BASE = "https://www.reddit.com"

#: Only posts older than this (seconds) are requested: Arctic Shift scores and
#: comment counts settle ~36 h after the post lands.
SETTLE_DELAY_S = 129_600
#: Politeness floor between our own requests (README: "a couple requests/sec").
MIN_REQUEST_INTERVAL_S = 1.5
RETRY_BACKOFF_S = (1.0, 2.0, 4.0)
MAX_RETRIES = len(RETRY_BACKOFF_S)
#: 429 gets its own, smaller budget: 3 attempts total (RULING A11b).
MAX_RATE_LIMIT_ATTEMPTS = 3
RATE_LIMIT_BUFFER_S = 0.5
HTTP_TIMEOUT_S = 15.0
TOP_COMMENT_COUNT = 10
COMMENTS_LIMIT = 100
#: ``/api/posts/search`` rejects a ``limit`` above 100 (the scrape API allows 500).
MAX_POSTS_LIMIT = 100
#: How much of a non-JSON body is scanned for the transient-overload markers.
MAX_ERROR_BODY_CHARS = 500
DELETED_AUTHOR_PLACEHOLDERS = frozenset({"[deleted]", "[removed]"})
#: Substrings of the transient-overload error bodies Arctic Shift returns.
TIMEOUT_MARKERS = ("timed out", "timeout")
VIA = "arctic_shift"

# Reddit strategy posts often carry the real content in screenshots (rule
# tables, TradingView setups, equity curves). Capture their URLs so the
# scraper can archive them for the vision-capable extractor. Capped small so a
# gallery post can't balloon the archive or the extractor's image budget.
MAX_IMAGES_PER_POST = 4
IMAGE_URL_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
REDDIT_IMAGE_HOSTS = frozenset({"i.redd.it"})


@register_source("reddit")
class RedditSource:
    """Read-only Reddit source over the free Arctic Shift search API."""

    name: ClassVar[str] = "reddit"

    def __init__(
        self,
        config: RedditConfig | None = None,
        subreddits: list[str] | None = None,
        client: _HttpGetter | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._config = config or RedditConfig.from_env()
        self._subreddits = subreddits if subreddits is not None else subreddits_from_env()
        self._clock = clock or time.time
        if client is not None:
            self._client: _HttpGetter = client
        else:
            self._client = httpx.Client(
                timeout=HTTP_TIMEOUT_S,
                headers={
                    "User-Agent": self._config.user_agent,
                    "Accept": "application/json",
                },
            )
        self._last_request_at: float = 0.0

    def close(self) -> None:
        self._client.close()

    def fetch(self, since: datetime | None, limit: int) -> Iterable[RawItem]:
        """Yield RawItems from each configured subreddit's settled posts.

        Args:
            since: If set, stop yielding once a post older than `since` is
                seen (`sort=desc` makes that a cheap early exit).
            limit: Per-subreddit cap on posts to inspect — a per-source-unit
                cap, not a run total (`fetch` applies it once per subreddit).
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
        # (vibe-quant-ux7t0).
        if failures and not yielded and len(failures) == len(self._subreddits):
            names = ", ".join(name for name, _ in failures)
            raise RuntimeError(
                f"all {len(failures)} subreddit(s) failed ({names}): {failures[-1][1]}"
            ) from failures[-1][1]

    def _fetch_subreddit(
        self, subreddit: str, since_ts: float | None, limit: int
    ) -> Iterable[RawItem]:
        for data in self._fetch_posts(subreddit, limit):
            if since_ts is not None:
                created_utc = float(data.get("created_utc") or 0.0)
                if created_utc < since_ts:
                    # sort=desc → the first stale post means the rest are stale.
                    return
            yield self._to_raw_item(data, subreddit)

    def _fetch_posts(self, subreddit: str, limit: int) -> list[dict[str, Any]]:
        params = {
            "subreddit": subreddit,
            "before": str(int(self._clock() - SETTLE_DELAY_S)),
            "limit": str(min(limit, MAX_POSTS_LIMIT)),
            "sort": "desc",
        }
        body = self._request(f"{API_BASE_URL}{POSTS_PATH}", params)
        return _data_rows(body)

    def _fetch_comments(self, post_id: str) -> list[dict[str, Any]]:
        # Top-level comments only (`link_id` scopes them to the post); the
        # response is score-ordered but we re-sort and slice locally so a
        # server-side ordering drift cannot change what we archive.
        params = {"link_id": f"t3_{post_id}", "limit": str(COMMENTS_LIMIT)}
        body = self._request(f"{API_BASE_URL}{COMMENTS_PATH}", params)
        selected = sorted(_data_rows(body), key=_comment_order_key)[:TOP_COMMENT_COUNT]
        return [
            {
                "author": _normalize_author(row.get("author")),
                "body": str(row.get("body") or ""),
                "score": int(row.get("score") or 0),
            }
            for row in selected
        ]

    def _to_raw_item(self, data: dict[str, Any], subreddit: str) -> RawItem:
        post_id = str(data.get("id") or "")
        num_comments = int(data.get("num_comments") or 0)
        comments = self._fetch_comments(post_id) if post_id and num_comments > 0 else []
        permalink = str(data.get("permalink") or "")
        url = f"{REDDIT_URL_BASE}{permalink}" if permalink else f"{REDDIT_URL_BASE}/comments/{post_id}"
        extras: dict[str, Any] = {
            "subreddit": subreddit,
            "num_comments": num_comments,
            "comments": comments,
            "via": VIA,
        }
        image_urls = _extract_image_urls(data)
        if image_urls:
            extras["image_urls"] = image_urls
        return RawItem(
            source="reddit",
            external_id=post_id,
            url=url,
            title=str(data.get("title") or ""),
            body=str(data.get("selftext") or ""),
            author=_normalize_author(data.get("author")),
            posted_at=datetime.fromtimestamp(float(data.get("created_utc") or 0.0), tz=UTC),
            score=int(data.get("score") or 0),
            extras=extras,
        )

    def _request(self, url: str, params: dict[str, str]) -> Any:
        last_exc: httpx.HTTPError | None = None
        rate_limit_attempts = 0
        for attempt in range(MAX_RETRIES + 1):
            self._respect_rate_limit()
            response: httpx.Response | None = None
            try:
                response = self._client.get(url, params=params)
            except httpx.HTTPError as e:
                last_exc = e
            finally:
                # Count every attempt, success or failure, so an endpoint that
                # keeps erroring cannot bypass the politeness floor (A11c).
                self._last_request_at = time.monotonic()
            if response is not None:
                if response.status_code == 429:
                    rate_limit_attempts += 1
                    last_exc = ArcticShiftError(f"429 Too Many Requests from {url}")
                    if rate_limit_attempts >= MAX_RATE_LIMIT_ATTEMPTS:
                        # Budget spent — don't sleep after the final attempt.
                        break
                    wait_s = max(_parse_reset_after(response) or 0.0, 1.0)
                    logger.warning(
                        "429 on %s (attempt %d/%d); sleeping %.1fs",
                        url,
                        rate_limit_attempts,
                        MAX_RATE_LIMIT_ATTEMPTS,
                        wait_s,
                    )
                    time.sleep(wait_s + RATE_LIMIT_BUFFER_S)
                    continue
                if response.status_code >= 500 or _is_timeout_body(response):
                    last_exc = ArcticShiftError(
                        f"retryable status {response.status_code} from {url}"
                    )
                elif response.status_code >= 400:
                    # A real 4xx (no timeout marker): fail this subreddit now.
                    response.raise_for_status()
                else:
                    try:
                        payload = response.json()
                    except ValueError as e:
                        # A 2xx body that is not JSON is a failed attempt, not a
                        # run-aborting ValueError (httpx.DecodingError is an
                        # httpx.HTTPError, so `fetch` skips this subreddit).
                        last_exc = httpx.DecodingError(f"invalid JSON from {url}: {e}")
                    else:
                        return payload
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_S[attempt])
        logger.warning("exhausted retries on %s — skipping", url)
        assert last_exc is not None
        raise last_exc

    def _respect_rate_limit(self) -> None:
        if self._last_request_at == 0.0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < MIN_REQUEST_INTERVAL_S:
            time.sleep(MIN_REQUEST_INTERVAL_S - elapsed)


def _data_rows(body: Any) -> list[dict[str, Any]]:
    if not isinstance(body, dict):
        return []
    rows = body.get("data")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _comment_order_key(row: Mapping[str, Any]) -> tuple[int, float]:
    """Top comments: score desc, ties broken by created_utc asc (oldest first)."""
    return (-int(row.get("score") or 0), float(row.get("created_utc") or 0.0))


#: Distinct from a JSON ``null`` body (which is a valid payload, not an error).
_INVALID_JSON = object()


def _is_timeout_body(response: httpx.Response) -> bool:
    """True when the body reports a transient Arctic Shift overload.

    Only a body that is not valid JSON is scanned as text. Any body that parses
    as JSON is a real payload and is never text-scanned — a post whose text
    merely contains "timeout" must not be retried (rev-gr-arctic-1). A JSON
    object is judged by its ``error`` field, with a ``data`` list always winning;
    a JSON list/number/string/null carries no error field and is accepted. A
    non-JSON body falls back to the first ``MAX_ERROR_BODY_CHARS`` of its text,
    whatever the status (RULING A11b).
    """
    payload = _json_or_invalid(response)
    if payload is not _INVALID_JSON:
        if not isinstance(payload, dict):
            return False
        if isinstance(payload.get("data"), list):
            return False
        error = str(payload.get("error") or "").lower()
        return any(marker in error for marker in TIMEOUT_MARKERS)
    text = _response_text(response)[:MAX_ERROR_BODY_CHARS].lower()
    return any(marker in text for marker in TIMEOUT_MARKERS)


def _json_or_invalid(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return _INVALID_JSON


def _response_text(response: httpx.Response) -> str:
    try:
        return response.text
    except (httpx.HTTPError, UnicodeDecodeError):
        return ""


def _is_image_link(url: str) -> bool:
    low = url.lower()
    if low.endswith(IMAGE_URL_SUFFIXES):
        return True
    host = (urlparse(url).hostname or "").lower()
    return host in REDDIT_IMAGE_HOSTS


def _extract_image_urls(data: dict[str, Any]) -> list[str]:
    """Pull image URLs off a Reddit-shaped post, top-level post only.

    Covers three shapes: a direct image `data.url` (i.redd.it or an image
    suffix), the `data.preview.images[].source.url` thumbnails, and gallery
    posts via `data.media_metadata`. Reddit HTML-escapes `&` to `&amp;` in
    preview/gallery URLs, so every candidate is unescaped. Order-preserving,
    de-duplicated, and capped at ``MAX_IMAGES_PER_POST``.
    """
    urls: list[str] = []

    def _add(candidate: Any) -> None:
        if not isinstance(candidate, str) or not candidate:
            return
        unescaped = html.unescape(candidate)
        if unescaped and unescaped not in urls:
            urls.append(unescaped)

    direct = data.get("url")
    if isinstance(direct, str) and direct and _is_image_link(direct):
        _add(direct)

    preview = data.get("preview")
    if isinstance(preview, dict):
        for img in preview.get("images") or []:
            if isinstance(img, dict) and isinstance(img.get("source"), dict):
                _add(img["source"].get("url"))

    media = data.get("media_metadata")
    if isinstance(media, dict):
        for meta in media.values():
            if isinstance(meta, dict) and isinstance(meta.get("s"), dict):
                s = meta["s"]
                _add(s.get("u") or s.get("gif"))

    return urls[:MAX_IMAGES_PER_POST]


def _normalize_author(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value)
    if s in DELETED_AUTHOR_PLACEHOLDERS or not s:
        return None
    return s


def _parse_reset_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("X-RateLimit-Reset")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
