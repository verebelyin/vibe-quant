"""RedditSource unit tests — Arctic Shift API, fully mocked via httpx.MockTransport.

No network: `tests/unit/conftest.py` blocks outbound sockets, and every request
is served by an in-process `httpx.MockTransport` handler.
"""

from __future__ import annotations

import json
import logging
from contextlib import ExitStack
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import httpx
import pytest

from vibe_quant.research.config import (
    DEFAULT_USER_AGENT,
    ENV_REDDIT_USER_AGENT,
    RedditConfig,
)
from vibe_quant.research.sources.reddit import (
    API_HOST,
    MAX_RETRIES,
    MIN_REQUEST_INTERVAL_S,
    RATE_LIMIT_BUFFER_S,
    REDDIT_URL_BASE,
    RETRY_BACKOFF_S,
    SETTLE_DELAY_S,
    TOP_COMMENT_COUNT,
    RedditSource,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


UA = "vibe-quant-test:0.1 (by /u/tester)"
#: Fixed wall clock the injectable clock returns; every post is older than
#: ``FIXED_NOW - SETTLE_DELAY_S`` unless a test says otherwise.
FIXED_NOW = 1_800_000_000.0
#: A settled default post time (two settle windows back).
SETTLED_UTC = FIXED_NOW - 2 * SETTLE_DELAY_S


# ---------- Fixtures ----------


@pytest.fixture(autouse=True)
def _reset_warning_state() -> Generator[None]:
    """Clear warning dedup state so each test starts fresh."""
    from vibe_quant.research import config as cfg

    cfg._default_ua_warned = False
    cfg._deprecated_creds_warned.clear()
    yield
    cfg._default_ua_warned = False
    cfg._deprecated_creds_warned.clear()


@pytest.fixture
def ua_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_REDDIT_USER_AGENT, UA)


# ---------- Helpers ----------


def _post(
    *,
    sid: str,
    title: str = "t",
    selftext: str = "b",
    score: int = 1,
    created_utc: float = SETTLED_UTC,
    author: str | None = "u/x",
    num_comments: int = 1,
    permalink: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": sid,
        "title": title,
        "selftext": selftext,
        "score": score,
        "created_utc": created_utc,
        "num_comments": num_comments,
        "author": author,
        "permalink": permalink if permalink is not None else f"/r/algotrading/comments/{sid}/",
    }
    data.update(extra)
    return data


def _comment(
    *, author: str | None, body: str, score: int, created_utc: float = 0.0
) -> dict[str, Any]:
    return {
        "id": f"c-{author}-{body}",
        "link_id": "t3_x",
        "author": author,
        "body": body,
        "score": score,
        "created_utc": created_utc,
    }


def _api(data: list[dict[str, Any]]) -> dict[str, Any]:
    return {"data": data}


def _make_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"User-Agent": UA, "Accept": "application/json"},
    )


def _no_sleep(_: float) -> None:
    return None


def _fixed_clock() -> float:
    return FIXED_NOW


def _source(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    subreddits: tuple[str, ...] = ("algotrading",),
    clock: Callable[[], float] = _fixed_clock,
) -> RedditSource:
    return RedditSource(
        config=RedditConfig(user_agent=UA, using_default=False),
        subreddits=list(subreddits),
        client=_make_client(handler),
        clock=clock,
    )


class _FakeClock:
    """Monotonic clock advanced by the fake sleep (models wall-clock time).

    Any test asserting on ``sleep`` calls must patch ``time.monotonic`` too: a
    fake sleep that does not advance the clock makes the 1.5 s politeness floor
    look unelapsed and adds phantom waits.
    """

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def install(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(patch("time.monotonic", self.monotonic))
        stack.enter_context(patch("time.sleep", self.sleep))
        return stack


# ---------- Request shape ----------


def test_posts_request_uses_settle_delay_and_desc_sort(ua_env: None) -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.url.path == "/api/posts/search":
            return httpx.Response(200, json=_api([_post(sid="a", num_comments=0)]))
        return httpx.Response(200, json=_api([]))

    with patch("time.sleep", _no_sleep):
        items = list(_source(handler).fetch(since=None, limit=7))

    assert [i.external_id for i in items] == ["a"]
    posts_req = next(r for r in seen if r.url.path == "/api/posts/search")
    assert posts_req.url.host == API_HOST == "arctic-shift.photon-reddit.com"
    assert posts_req.url.params["subreddit"] == "algotrading"
    assert posts_req.url.params["before"] == str(int(FIXED_NOW - SETTLE_DELAY_S))
    assert posts_req.url.params["sort"] == "desc"
    assert posts_req.url.params["limit"] == "7"
    assert SETTLE_DELAY_S == 129_600


def test_posts_limit_clamped_to_100(ua_env: None) -> None:
    """The API allows `limit` up to 500; posts/search only accepts 100."""
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.url.path == "/api/posts/search":
            return httpx.Response(200, json=_api([_post(sid="a", num_comments=0)]))
        return httpx.Response(200, json=_api([]))

    with patch("time.sleep", _no_sleep):
        list(_source(handler).fetch(since=None, limit=200))

    posts_req = next(r for r in seen if r.url.path == "/api/posts/search")
    assert posts_req.url.params["limit"] == "100"


def test_never_requests_www_reddit(ua_env: None) -> None:
    hosts: set[str] = set()

    def handler(req: httpx.Request) -> httpx.Response:
        hosts.add(req.url.host or "")
        if req.url.path == "/api/posts/search":
            return httpx.Response(200, json=_api([_post(sid="a", num_comments=0)]))
        return httpx.Response(200, json=_api([]))

    with patch("time.sleep", _no_sleep):
        list(_source(handler).fetch(since=None, limit=1))

    assert hosts == {API_HOST}
    assert all(h != "www.reddit.com" for h in hosts)


def test_comments_request_uses_link_id_and_limit_100(ua_env: None) -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/posts/search":
            return httpx.Response(200, json=_api([_post(sid="abc", num_comments=3)]))
        seen[req.url.path] = req
        return httpx.Response(200, json=_api([]))

    with patch("time.sleep", _no_sleep):
        items = list(_source(handler).fetch(since=None, limit=1))

    assert len(items) == 1
    creq = seen["/api/comments/search"]
    assert creq.url.host == API_HOST
    assert creq.url.params["link_id"] == "t3_abc"
    assert creq.url.params["limit"] == "100"


def test_zero_comment_post_skips_comment_fetch(ua_env: None) -> None:
    paths: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        paths.append(req.url.path)
        if req.url.path == "/api/posts/search":
            return httpx.Response(200, json=_api([_post(sid="zc", num_comments=0)]))
        return httpx.Response(200, json=_api([]))

    with patch("time.sleep", _no_sleep):
        items = list(_source(handler).fetch(since=None, limit=1))

    assert len(items) == 1
    assert items[0].extras["comments"] == []
    assert "/api/comments/search" not in paths


# ---------- Translation tests ----------


def test_raw_item_fields(ua_env: None) -> None:
    post = _post(
        sid="raw1",
        title="Arctic post",
        selftext="body text",
        score=88,
        author="u/star",
        num_comments=2,
        permalink="/r/algotrading/comments/raw1/slug/",
    )
    comments = [
        _comment(author="u/top", body="hi", score=9, created_utc=2.0),
        _comment(author="[deleted]", body="x", score=1, created_utc=1.0),
    ]

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/posts/search":
            return httpx.Response(200, json=_api([post]))
        if req.url.path == "/api/comments/search":
            return httpx.Response(200, json=_api(comments))
        return httpx.Response(404)

    with patch("time.sleep", _no_sleep):
        items = list(_source(handler).fetch(since=None, limit=1))

    assert len(items) == 1
    it = items[0]
    assert it.source == "reddit"
    assert it.external_id == "raw1"
    assert it.url == f"{REDDIT_URL_BASE}/r/algotrading/comments/raw1/slug/"
    assert it.title == "Arctic post"
    assert it.body == "body text"
    assert it.author == "u/star"
    assert it.score == 88
    assert it.posted_at == datetime.fromtimestamp(post["created_utc"], tz=UTC)
    assert it.posted_at is not None and it.posted_at.tzinfo is UTC
    assert it.extras["subreddit"] == "algotrading"
    assert it.extras["num_comments"] == 2
    assert it.extras["via"] == "arctic_shift"
    assert it.extras["comments"] == [
        {"author": "u/top", "body": "hi", "score": 9},
        {"author": None, "body": "x", "score": 1},
    ]
    assert "flair" not in it.extras


def test_link_post_has_empty_body_and_permalink_fallback(ua_env: None) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/posts/search":
            return httpx.Response(
                200,
                json=_api(
                    [_post(sid="nop", selftext="", author=None, num_comments=0, permalink="")]
                ),
            )
        return httpx.Response(200, json=_api([]))

    with patch("time.sleep", _no_sleep):
        items = list(_source(handler).fetch(since=None, limit=1))

    it = items[0]
    assert it.body == ""
    assert it.author is None
    assert it.url == "https://www.reddit.com/comments/nop"
    assert it.extras["comments"] == []


def test_top_comments_capped_and_sorted_by_score_then_created(ua_env: None) -> None:
    comments = [
        _comment(author=f"u/{i}", body=f"b{i}", score=i, created_utc=float(i)) for i in range(14)
    ]
    comments += [
        _comment(author="A", body="A", score=14, created_utc=100.0),
        _comment(author="B", body="B", score=14, created_utc=50.0),
    ]

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/posts/search":
            return httpx.Response(200, json=_api([_post(sid="cap", num_comments=5)]))
        return httpx.Response(200, json=_api(comments))

    with patch("time.sleep", _no_sleep):
        items = list(_source(handler).fetch(since=None, limit=1))

    cs = items[0].extras["comments"]
    assert len(cs) == TOP_COMMENT_COUNT == 10
    scores = [c["score"] for c in cs]
    assert scores == sorted(scores, reverse=True)
    # The two score-14 ties come first, ordered by created_utc ascending (B=50 < A=100).
    assert [c["author"] for c in cs[:2]] == ["B", "A"]
    assert scores[0] == 14


def test_deleted_and_removed_authors_normalized_to_none(ua_env: None) -> None:
    posts = [
        _post(sid="p1", author=None, num_comments=1),
        _post(sid="p2", author="[deleted]", num_comments=1),
        _post(sid="p3", author="[removed]", num_comments=1),
    ]
    comments = [
        _comment(author=None, body="x", score=1),
        _comment(author="[deleted]", body="y", score=2),
        _comment(author="[removed]", body="z", score=3),
    ]

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/posts/search":
            return httpx.Response(200, json=_api(posts))
        return httpx.Response(200, json=_api(comments))

    with patch("time.sleep", _no_sleep):
        items = list(_source(handler).fetch(since=None, limit=10))

    assert all(it.author is None for it in items)
    for it in items:
        assert all(c["author"] is None for c in it.extras["comments"])


def test_snapshot_parity(ua_env: None) -> None:
    """Frozen baseline: a known Arctic Shift input produces this exact RawItem dict."""
    post = _post(
        sid="snap1",
        title="Snapshot test",
        selftext="body text",
        score=99,
        created_utc=1700000000.0,
        author="u/snapshotter",
        num_comments=2,
    )
    comments = [
        _comment(author="u/topvoter", body="great post", score=42),
        _comment(author="[deleted]", body="[deleted]", score=1),
    ]

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/posts/search":
            return httpx.Response(200, json=_api([post]))
        return httpx.Response(200, json=_api(comments))

    with patch("time.sleep", _no_sleep):
        items = list(_source(handler).fetch(since=None, limit=1))

    assert len(items) == 1
    item = items[0]
    expected = {
        "source": "reddit",
        "external_id": "snap1",
        "url": f"{REDDIT_URL_BASE}/r/algotrading/comments/snap1/",
        "title": "Snapshot test",
        "body": "body text",
        "author": "u/snapshotter",
        "posted_at": "2023-11-14T22:13:20+00:00",
        "score": 99,
        "extras": {
            "subreddit": "algotrading",
            "num_comments": 2,
            "comments": [
                {"author": "u/topvoter", "body": "great post", "score": 42},
                {"author": None, "body": "[deleted]", "score": 1},
            ],
            "via": "arctic_shift",
        },
    }
    actual = {
        "source": item.source,
        "external_id": item.external_id,
        "url": item.url,
        "title": item.title,
        "body": item.body,
        "author": item.author,
        "posted_at": item.posted_at.isoformat() if item.posted_at else None,
        "score": item.score,
        "extras": item.extras,
    }
    assert actual == expected, json.dumps({"expected": expected, "actual": actual}, indent=2)


# ---------- Behaviour tests ----------


def test_since_stops_at_first_older_post(ua_env: None) -> None:
    cutoff = datetime(2025, 6, 1, tzinfo=UTC).timestamp()
    posts = [
        _post(sid="new", created_utc=cutoff + 3600, num_comments=0),
        _post(sid="old", created_utc=cutoff - 3600, num_comments=0),
        _post(sid="older", created_utc=cutoff - 7200, num_comments=0),
    ]

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_api(posts))

    with patch("time.sleep", _no_sleep):
        items = list(_source(handler).fetch(since=datetime.fromtimestamp(cutoff, tz=UTC), limit=10))

    assert [i.external_id for i in items] == ["new"]


def test_empty_subreddit_returns_no_items(ua_env: None) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_api([]))

    with patch("time.sleep", _no_sleep):
        items = list(_source(handler).fetch(since=None, limit=10))

    assert items == []


# ---------- Error handling ----------


def test_429_reset_header_sleeps_then_succeeds(ua_env: None) -> None:
    calls = {"n": 0}
    clock = _FakeClock()

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if req.url.path == "/api/posts/search" and calls["n"] == 1:
            return httpx.Response(429, headers={"X-RateLimit-Reset": "50"}, json={})
        if req.url.path == "/api/posts/search":
            return httpx.Response(200, json=_api([_post(sid="r", num_comments=0)]))
        return httpx.Response(200, json=_api([]))

    with clock.install():
        items = list(_source(handler).fetch(since=None, limit=1))

    assert [i.external_id for i in items] == ["r"]
    # max(X-RateLimit-Reset=50, 1) + 0.5 buffer. The reset wait already covers
    # the 1.5 s politeness floor, so no extra floor sleep follows it.
    assert clock.sleeps == [50.5]


@pytest.mark.parametrize("reset_header", [None, "0", "0.2"])
def test_429_sub_second_reset_sleeps_the_floor_plus_buffer(
    ua_env: None, reset_header: str | None
) -> None:
    """A 429 whose X-RateLimit-Reset is missing, zero or sub-second still waits
    the full 1.0 s floor (not just the 0.5 s buffer): a single sleep, and no
    phantom politeness top-up on the next attempt."""
    calls = {"n": 0}
    clock = _FakeClock()

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if req.url.path == "/api/posts/search" and calls["n"] == 1:
            headers = {} if reset_header is None else {"X-RateLimit-Reset": reset_header}
            return httpx.Response(429, headers=headers, json={})
        if req.url.path == "/api/posts/search":
            return httpx.Response(200, json=_api([_post(sid="f", num_comments=0)]))
        return httpx.Response(200, json=_api([]))

    with clock.install():
        items = list(_source(handler).fetch(since=None, limit=1))

    assert [i.external_id for i in items] == ["f"]
    # max(reset=0, floor=1.0) + 0.5 buffer; the 1.5 s wait also covers the
    # politeness floor, so exactly one sleep is recorded.
    assert clock.sleeps == [1.0 + RATE_LIMIT_BUFFER_S]


def test_5xx_backoff_retries_then_subreddit_skipped(
    ua_env: None, caplog: pytest.LogCaptureFixture
) -> None:
    clock = _FakeClock()

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.params.get("subreddit") == "broken":
            return httpx.Response(503, text="service unavailable")
        if req.url.path == "/api/posts/search":
            return httpx.Response(200, json=_api([_post(sid="ok", num_comments=0)]))
        return httpx.Response(200, json=_api([]))

    with clock.install(), caplog.at_level(logging.WARNING):
        items = list(_source(handler, subreddits=("broken", "algotrading")).fetch(None, 5))

    assert [i.external_id for i in items] == ["ok"]
    # Backoff (1, 2, 4); the 1.0 s wait is topped up to the 1.5 s politeness
    # floor, and the surviving subreddit's first request waits out the floor
    # left by "broken"'s final attempt (A11c).
    assert clock.sleeps == [1.0, MIN_REQUEST_INTERVAL_S - 1.0, 2.0, 4.0, MIN_REQUEST_INTERVAL_S]
    assert RETRY_BACKOFF_S == (1.0, 2.0, 4.0)
    assert MAX_RETRIES == 3
    assert any("broken" in rec.message for rec in caplog.records)


def test_timeout_body_is_retried(ua_env: None) -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/posts/search":
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(200, text='{"error": "Query timed out"}')
            return httpx.Response(200, json=_api([_post(sid="t", num_comments=0)]))
        return httpx.Response(200, json=_api([]))

    with patch("time.sleep", _no_sleep):
        items = list(_source(handler).fetch(since=None, limit=1))

    assert [i.external_id for i in items] == ["t"]
    assert calls["n"] == 2


def test_transient_timeout_422_is_retried(ua_env: None) -> None:
    """Arctic Shift reports overload as 422 + a timeout error body (RULING A11b)."""
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/posts/search":
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(
                    422, json={"data": None, "error": "Timeout. Maybe slow down a bit"}
                )
            return httpx.Response(200, json=_api([_post(sid="r422", num_comments=0)]))
        return httpx.Response(200, json=_api([]))

    with patch("time.sleep", _no_sleep):
        items = list(_source(handler).fetch(since=None, limit=1))

    assert [i.external_id for i in items] == ["r422"]
    assert calls["n"] == 2


def test_422_without_timeout_text_is_not_retried(ua_env: None) -> None:
    """A genuine 4xx (no timeout marker) must fail fast, not be retried."""
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(422, json={"data": None, "error": "invalid 'sort' value"})

    with patch("time.sleep", _no_sleep):
        src = _source(handler)
        with pytest.raises(RuntimeError, match="all 1 subreddit"):
            list(src.fetch(since=None, limit=1))

    assert calls["n"] == 1


def test_200_with_timeout_word_in_post_text_is_accepted(ua_env: None) -> None:
    """Regression (rev-gr-arctic-1 blocking): a real 2xx `data` payload must
    never be scanned for timeout markers. A live post whose selftext read
    "my order hit a timeout on binance" used to be retried 4x and dropped."""
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/posts/search":
            calls["n"] += 1
            return httpx.Response(
                200,
                json=_api(
                    [
                        _post(
                            sid="tp",
                            selftext="my order hit a timeout on binance",
                            num_comments=0,
                        )
                    ]
                ),
            )
        return httpx.Response(200, json=_api([]))

    with patch("time.sleep", _no_sleep):
        items = list(_source(handler).fetch(since=None, limit=1))

    assert [i.external_id for i in items] == ["tp"]
    assert items[0].body == "my order hit a timeout on binance"
    assert calls["n"] == 1


def test_200_data_payload_with_error_field_is_accepted(ua_env: None) -> None:
    """A 2xx body with a `data` list wins over a trailing `error` field."""
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/posts/search":
            calls["n"] += 1
            return httpx.Response(
                200,
                json={"data": [_post(sid="de", num_comments=0)], "error": "timeout"},
            )
        return httpx.Response(200, json=_api([]))

    with patch("time.sleep", _no_sleep):
        items = list(_source(handler).fetch(since=None, limit=1))

    assert [i.external_id for i in items] == ["de"]
    assert calls["n"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        [{"title": "timeout strategy"}],
        "timeout strategy",
        42,
        None,
    ],
)
def test_200_non_dict_json_body_is_not_scanned_or_retried(
    ua_env: None, payload: Any
) -> None:
    """A body that parses as JSON is a payload, never a text error message. JSON
    that is not a dict (a list, string, number or ``null``) has no ``error``
    field to check and must be accepted — even when its raw text mentions
    "timeout" (the raw-text fallback caught the list and the string and retried
    them 4x before dropping the subreddit)."""
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            200,
            content=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )

    with patch("time.sleep", _no_sleep):
        items = list(_source(handler).fetch(since=None, limit=1))

    assert items == []
    assert calls["n"] == 1


def test_non_json_200_body_with_timeout_text_is_retried(ua_env: None) -> None:
    """Non-JSON bodies fall back to the first 500 chars of the text."""
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/posts/search":
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(200, text="Query timed out, please retry")
            return httpx.Response(200, json=_api([_post(sid="nj", num_comments=0)]))
        return httpx.Response(200, json=_api([]))

    with patch("time.sleep", _no_sleep):
        items = list(_source(handler).fetch(since=None, limit=1))

    assert [i.external_id for i in items] == ["nj"]
    assert calls["n"] == 2


def test_invalid_json_200_then_valid_succeeds(ua_env: None) -> None:
    """A 2xx with invalid JSON is a failed attempt: back off, then retry."""
    calls = {"n": 0}
    clock = _FakeClock()

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/posts/search":
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(200, text="<html>upstream proxy error</html>")
            return httpx.Response(200, json=_api([_post(sid="ok", num_comments=0)]))
        return httpx.Response(200, json=_api([]))

    with clock.install():
        items = list(_source(handler).fetch(since=None, limit=1))

    assert [i.external_id for i in items] == ["ok"]
    assert calls["n"] == 2
    # 1.0 s backoff, then a top-up to the 1.5 s floor before the retry.
    assert clock.sleeps == [RETRY_BACKOFF_S[0], MIN_REQUEST_INTERVAL_S - RETRY_BACKOFF_S[0]]


def test_persistent_invalid_json_skips_subreddit_not_whole_run(ua_env: None) -> None:
    """Invalid JSON must never abort the run (an uncaught ValueError did)."""

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.params.get("subreddit") == "broken":
            return httpx.Response(200, text="not json at all")
        if req.url.path == "/api/posts/search":
            return httpx.Response(200, json=_api([_post(sid="ok", num_comments=0)]))
        return httpx.Response(200, json=_api([]))

    with patch("time.sleep", _no_sleep):
        items = list(_source(handler, subreddits=("broken", "algotrading")).fetch(None, 5))

    assert [i.external_id for i in items] == ["ok"]


def test_429_exhausts_three_attempts_without_trailing_sleep(ua_env: None) -> None:
    """At most 3 attempts total for 429, and no sleep after the final one."""
    calls = {"n": 0}
    clock = _FakeClock()

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, headers={"X-RateLimit-Reset": "2"}, json={})

    with clock.install(), pytest.raises(RuntimeError, match="all 1 subreddit"):
        list(_source(handler).fetch(since=None, limit=1))

    assert calls["n"] == 3
    # Two waits, both before their next attempt — never after the third.
    assert clock.sleeps == [2.5, 2.5]


def test_failing_subreddit_still_starts_the_politeness_floor(ua_env: None) -> None:
    """A 4xx fails fast with no backoff, but the next subreddit's request must
    still wait out the 1.5 s floor (A11c — a failing endpoint cannot let the
    next request race ahead)."""
    clock = _FakeClock()
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.params.get("subreddit") or "")
        if req.url.params.get("subreddit") == "broken":
            return httpx.Response(422, json={"data": None, "error": "invalid 'sort' value"})
        if req.url.path == "/api/posts/search":
            return httpx.Response(200, json=_api([_post(sid="ok", num_comments=0)]))
        return httpx.Response(200, json=_api([]))

    with clock.install():
        items = list(_source(handler, subreddits=("broken", "algotrading")).fetch(None, 5))

    assert [i.external_id for i in items] == ["ok"]
    assert seen == ["broken", "algotrading"]
    assert clock.sleeps == [MIN_REQUEST_INTERVAL_S]


def test_transport_error_still_starts_the_politeness_floor(ua_env: None) -> None:
    """A transport error (no HTTP response at all) must still stamp the request
    clock, so the next subreddit's first request waits out the 1.5 s floor. The
    failing call is the LAST attempt of subreddit A, so no retry backoff sleeps
    after it — the floor is the only thing between A's failure and B's request."""
    clock = _FakeClock()
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        subreddit = req.url.params.get("subreddit") or ""
        seen.append(subreddit)
        if subreddit == "broken":
            raise httpx.ConnectError("connection reset by peer")
        if req.url.path == "/api/posts/search":
            return httpx.Response(200, json=_api([_post(sid="ok", num_comments=0)]))
        return httpx.Response(200, json=_api([]))

    with clock.install():
        items = list(_source(handler, subreddits=("broken", "algotrading")).fetch(None, 5))

    assert [i.external_id for i in items] == ["ok"]
    assert seen == ["broken"] * 4 + ["algotrading"]
    # Backoffs after A's first three attempts, the 1.0 s one topped up to the
    # 1.5 s floor; then A's 4th (final) attempt stamps the clock and B's first
    # request waits the full floor. No sleep follows A's last attempt.
    assert clock.sleeps == [
        1.0,
        MIN_REQUEST_INTERVAL_S - 1.0,
        2.0,
        4.0,
        MIN_REQUEST_INTERVAL_S,
    ]


def test_all_subreddits_failing_raises(ua_env: None) -> None:
    """Hard failure on every subreddit must raise so the scrape run records
    failed instead of a silent completed/0-items (vibe-quant-ux7t0)."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    with patch("time.sleep", _no_sleep):
        src = _source(handler, subreddits=("algotrading", "quant"))
        with pytest.raises(RuntimeError, match="all 2 subreddit"):
            list(src.fetch(since=None, limit=5))


def test_partial_subreddit_failure_stays_lenient(ua_env: None) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.params.get("subreddit") == "quant":
            return httpx.Response(503, text="down")
        if req.url.path == "/api/posts/search":
            return httpx.Response(200, json=_api([_post(sid="ok1", num_comments=0)]))
        return httpx.Response(200, json=_api([]))

    with patch("time.sleep", _no_sleep):
        items = list(_source(handler, subreddits=("algotrading", "quant")).fetch(None, 5))

    assert [i.external_id for i in items] == ["ok1"]


# ---------- HTTP plumbing tests ----------


def test_source_client_carries_configured_user_agent(ua_env: None) -> None:
    src = RedditSource(
        config=RedditConfig(user_agent="vq-cfg:1 (by /u/x)", using_default=False),
        subreddits=["algotrading"],
    )
    try:
        assert src._client.headers["user-agent"] == "vq-cfg:1 (by /u/x)"
        assert src._client.headers["accept"] == "application/json"
    finally:
        src.close()


def test_rate_limit_floor_enforced_between_requests(ua_env: None) -> None:
    fake_now = [1000.0]
    sleeps: list[float] = []

    def fake_monotonic() -> float:
        return fake_now[0]

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        fake_now[0] += seconds

    def handler(req: httpx.Request) -> httpx.Response:
        fake_now[0] += 0.01  # request "duration"
        if req.url.path == "/api/posts/search":
            return httpx.Response(200, json=_api([_post(sid="rl1", num_comments=1)]))
        return httpx.Response(200, json=_api([]))

    with patch("time.monotonic", fake_monotonic), patch("time.sleep", fake_sleep):
        list(_source(handler).fetch(since=None, limit=1))

    rate_limit_sleeps = [s for s in sleeps if s >= MIN_REQUEST_INTERVAL_S - 0.1]
    assert len(rate_limit_sleeps) >= 1
    assert MIN_REQUEST_INTERVAL_S == 1.5


def test_default_user_agent_used_when_env_unset(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv(ENV_REDDIT_USER_AGENT, raising=False)
    with caplog.at_level(logging.WARNING):
        cfg = RedditConfig.from_env()
    assert cfg.user_agent == DEFAULT_USER_AGENT
    assert cfg.using_default is True
    assert any(ENV_REDDIT_USER_AGENT in rec.message for rec in caplog.records)


def test_deprecated_creds_warning_emitted_once_per_var(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("REDDIT_CLIENT_ID", "stale-id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "stale-secret")
    monkeypatch.setenv(ENV_REDDIT_USER_AGENT, "ua")
    with caplog.at_level(logging.WARNING):
        RedditConfig.from_env()
        RedditConfig.from_env()  # second call: must not re-emit

    msgs = [rec.getMessage() for rec in caplog.records]
    id_warnings = [m for m in msgs if "REDDIT_CLIENT_ID" in m]
    secret_warnings = [m for m in msgs if "REDDIT_CLIENT_SECRET" in m]
    assert len(id_warnings) == 1
    assert len(secret_warnings) == 1
    for m in (*id_warnings, *secret_warnings):
        assert "no longer used" in m
        assert "Safe to remove" in m


# ---------- Image URL extraction tests ----------


def _fetch_one(post: dict[str, Any], ua_env: None) -> Any:  # noqa: ARG001
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/posts/search":
            return httpx.Response(200, json=_api([post]))
        return httpx.Response(200, json=_api([]))

    with patch("time.sleep", _no_sleep):
        items = list(_source(handler).fetch(since=None, limit=1))
    assert len(items) == 1
    return items[0]


def test_direct_image_link_captured(ua_env: None) -> None:
    post = _post(sid="d1", num_comments=0)
    post["url"] = "https://i.redd.it/abc123.png"
    item = _fetch_one(post, ua_env)
    assert item.extras["image_urls"] == ["https://i.redd.it/abc123.png"]


def test_direct_non_image_link_ignored(ua_env: None) -> None:
    post = _post(sid="d2", num_comments=0)
    post["url"] = "https://example.com/article.html"
    item = _fetch_one(post, ua_env)
    assert "image_urls" not in item.extras


def test_preview_image_url_html_unescaped(ua_env: None) -> None:
    post = _post(sid="p1", num_comments=0)
    post["preview"] = {
        "images": [
            {
                "source": {
                    "url": "https://preview.redd.it/x.png?width=640&amp;crop=smart&amp;s=deadbeef"
                }
            }
        ]
    }
    item = _fetch_one(post, ua_env)
    assert item.extras["image_urls"] == [
        "https://preview.redd.it/x.png?width=640&crop=smart&s=deadbeef"
    ]


def test_gallery_media_metadata_captured(ua_env: None) -> None:
    post = _post(sid="g1", num_comments=0)
    post["media_metadata"] = {
        "m1": {"s": {"u": "https://preview.redd.it/one.jpg?s=aaa&amp;w=1"}},
        "m2": {"s": {"gif": "https://preview.redd.it/two.gif?s=bbb"}},
    }
    item = _fetch_one(post, ua_env)
    assert item.extras["image_urls"] == [
        "https://preview.redd.it/one.jpg?s=aaa&w=1",
        "https://preview.redd.it/two.gif?s=bbb",
    ]


def test_image_urls_capped_at_four(ua_env: None) -> None:
    post = _post(sid="cap", num_comments=0)
    post["media_metadata"] = {
        f"m{i}": {"s": {"u": f"https://preview.redd.it/{i}.png"}} for i in range(10)
    }
    item = _fetch_one(post, ua_env)
    assert len(item.extras["image_urls"]) == 4


def test_image_urls_deduplicated(ua_env: None) -> None:
    post = _post(sid="dup", num_comments=0)
    post["url"] = "https://i.redd.it/same.png"
    post["preview"] = {"images": [{"source": {"url": "https://i.redd.it/same.png"}}]}
    item = _fetch_one(post, ua_env)
    assert item.extras["image_urls"] == ["https://i.redd.it/same.png"]


def test_post_without_images_has_no_image_urls_key(ua_env: None) -> None:
    item = _fetch_one(_post(sid="none", num_comments=0), ua_env)
    assert "image_urls" not in item.extras


# ---------- Regression guards ----------


def test_no_praw_imports_remain() -> None:
    """Regression guard: praw must not creep back into the module."""
    import inspect

    from vibe_quant.research.sources import reddit as _mod

    contents = inspect.getsource(_mod)
    assert "import praw" not in contents
    assert "import prawcore" not in contents
