"""RedditSource unit tests — fully mocked via httpx.MockTransport, no network."""

from __future__ import annotations

import json
import logging
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
    BASE_URL,
    MIN_REQUEST_INTERVAL_S,
    RedditSource,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


# ---------- Fixtures ----------


@pytest.fixture(autouse=True)
def _reset_warning_state() -> Generator[None]:
    """Clear default-UA warning dedup so each test starts fresh."""
    from vibe_quant.research import config as cfg

    cfg._default_ua_warned = False
    yield
    cfg._default_ua_warned = False


@pytest.fixture
def ua_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_REDDIT_USER_AGENT, "vibe-quant-test:0.1 (by /u/tester)")


def _post(
    *,
    sid: str,
    title: str = "t",
    selftext: str = "b",
    score: int = 1,
    created_utc: float = 1735689600.0,
    author: str | None = "u/x",
    flair: str | None = None,
    num_comments: int = 1,
) -> dict[str, Any]:
    return {
        "kind": "t3",
        "data": {
            "id": sid,
            "title": title,
            "selftext": selftext,
            "score": score,
            "created_utc": created_utc,
            "permalink": f"/r/algotrading/comments/{sid}/",
            "link_flair_text": flair,
            "num_comments": num_comments,
            "author": author,
        },
    }


def _comment(*, author: str | None, body: str, score: int) -> dict[str, Any]:
    return {
        "kind": "t1",
        "data": {"author": author, "body": body, "score": score},
    }


def _listing(children: list[dict[str, Any]]) -> dict[str, Any]:
    return {"data": {"children": children, "after": None, "before": None}}


def _comment_thread(comments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reddit's <permalink>.json shape: [post_listing, comment_listing]."""
    return [_listing([]), _listing(comments)]


def _make_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={
            "User-Agent": "vibe-quant-test:0.1 (by /u/tester)",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
        },
    )


def _no_sleep(_: float) -> None:
    return None


# ---------- Translation tests ----------


def test_listing_translation_yields_raw_items(ua_env: None) -> None:
    posts = [
        _post(
            sid="aaa",
            title="Mean reversion idea",
            selftext="RSI < 30 entry",
            score=42,
            author="u/quantnerd",
            flair="Strategy",
            num_comments=3,
        ),
        _post(sid="bbb", title="Link only", selftext="", author=None, num_comments=0),
    ]

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/new.json"):
            return httpx.Response(200, json=_listing(posts))
        # comment threads — return empty for both posts
        return httpx.Response(200, json=_comment_thread([]))

    with patch("time.sleep", _no_sleep):
        src = RedditSource(
            config=RedditConfig.from_env(),
            subreddits=["algotrading"],
            client=_make_client(handler),
        )
        items = list(src.fetch(since=None, limit=10))

    assert len(items) == 2
    a = items[0]
    assert a.source == "reddit"
    assert a.external_id == "aaa"
    assert a.title == "Mean reversion idea"
    assert a.body == "RSI < 30 entry"
    assert a.score == 42
    assert a.author == "u/quantnerd"
    assert a.posted_at == datetime.fromtimestamp(1735689600.0, tz=UTC)
    assert a.url == f"{BASE_URL}/r/algotrading/comments/aaa/"
    assert a.extras["subreddit"] == "algotrading"
    assert a.extras["flair"] == "Strategy"
    assert a.extras["num_comments"] == 3
    assert a.extras["comments"] == []

    b = items[1]
    assert b.body == ""
    assert b.author is None  # null author normalized


def test_snapshot_parity_with_prior_praw_output(ua_env: None) -> None:
    """Frozen baseline: a known input produces this exact RawItem dict.

    Captures the praw-era contract so any future regression of field names,
    types, or extras shape is caught.
    """
    post = _post(
        sid="snap1",
        title="Snapshot test",
        selftext="body text",
        score=99,
        created_utc=1700000000.0,
        author="u/snapshotter",
        flair="Discussion",
        num_comments=2,
    )
    comments = [
        _comment(author="u/topvoter", body="great post", score=42),
        _comment(author="[deleted]", body="[deleted]", score=1),
    ]

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/new.json"):
            return httpx.Response(200, json=_listing([post]))
        return httpx.Response(200, json=_comment_thread(comments))

    with patch("time.sleep", _no_sleep):
        src = RedditSource(
            config=RedditConfig.from_env(),
            subreddits=["algotrading"],
            client=_make_client(handler),
        )
        items = list(src.fetch(since=None, limit=1))

    assert len(items) == 1
    item = items[0]
    expected = {
        "source": "reddit",
        "external_id": "snap1",
        "url": f"{BASE_URL}/r/algotrading/comments/snap1/",
        "title": "Snapshot test",
        "body": "body text",
        "author": "u/snapshotter",
        "posted_at": "2023-11-14T22:13:20+00:00",
        "score": 99,
        "extras": {
            "subreddit": "algotrading",
            "flair": "Discussion",
            "num_comments": 2,
            "comments": [
                {"author": "u/topvoter", "body": "great post", "score": 42},
                {"author": None, "body": "[deleted]", "score": 1},
            ],
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


def test_comments_top_level_only_no_nested_replies(ua_env: None) -> None:
    """Nested `data.replies` trees must not pollute extras.comments."""
    nested = {
        "kind": "t1",
        "data": {
            "author": "u/parent",
            "body": "top level",
            "score": 5,
            "replies": _listing([_comment(author="u/child", body="nested", score=99)]),
        },
    }

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/new.json"):
            return httpx.Response(200, json=_listing([_post(sid="n", num_comments=1)]))
        return httpx.Response(200, json=_comment_thread([nested]))

    with patch("time.sleep", _no_sleep):
        src = RedditSource(
            config=RedditConfig.from_env(),
            subreddits=["algotrading"],
            client=_make_client(handler),
        )
        items = list(src.fetch(since=None, limit=1))

    cs = items[0].extras["comments"]
    assert len(cs) == 1
    assert cs[0]["author"] == "u/parent"
    assert cs[0]["body"] == "top level"
    # nested child comment NOT included
    assert all("nested" not in c["body"] for c in cs)


def test_more_kind_children_skipped(ua_env: None) -> None:
    """`{kind: "more"}` placeholders must not appear in extras.comments."""
    children = [
        _comment(author="u/real", body="real", score=10),
        {"kind": "more", "data": {"count": 50, "children": ["abc", "def"]}},
        _comment(author="u/real2", body="real2", score=5),
    ]

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/new.json"):
            return httpx.Response(200, json=_listing([_post(sid="m")]))
        return httpx.Response(200, json=_comment_thread(children))

    with patch("time.sleep", _no_sleep):
        src = RedditSource(
            config=RedditConfig.from_env(),
            subreddits=["algotrading"],
            client=_make_client(handler),
        )
        items = list(src.fetch(since=None, limit=1))

    cs = items[0].extras["comments"]
    assert len(cs) == 2
    assert {c["author"] for c in cs} == {"u/real", "u/real2"}


def test_deleted_and_removed_authors_normalized_to_none(ua_env: None) -> None:
    posts = [
        _post(sid="p1", author=None),
        _post(sid="p2", author="[deleted]"),
        _post(sid="p3", author="[removed]"),
    ]
    comments = [
        _comment(author=None, body="x", score=1),
        _comment(author="[deleted]", body="y", score=2),
        _comment(author="[removed]", body="z", score=3),
    ]

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/new.json"):
            return httpx.Response(200, json=_listing(posts))
        return httpx.Response(200, json=_comment_thread(comments))

    with patch("time.sleep", _no_sleep):
        src = RedditSource(
            config=RedditConfig.from_env(),
            subreddits=["algotrading"],
            client=_make_client(handler),
        )
        items = list(src.fetch(since=None, limit=10))

    assert all(it.author is None for it in items)
    for it in items:
        assert all(c["author"] is None for c in it.extras["comments"])


def test_top_comments_capped_at_10_and_sorted(ua_env: None) -> None:
    comments = [_comment(author=f"u/{i}", body="x", score=i) for i in range(15)]

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/new.json"):
            return httpx.Response(200, json=_listing([_post(sid="cap")]))
        return httpx.Response(200, json=_comment_thread(comments))

    with patch("time.sleep", _no_sleep):
        src = RedditSource(
            config=RedditConfig.from_env(),
            subreddits=["algotrading"],
            client=_make_client(handler),
        )
        items = list(src.fetch(since=None, limit=1))

    cs = items[0].extras["comments"]
    assert len(cs) == 10
    assert [c["score"] for c in cs] == sorted([c["score"] for c in cs], reverse=True)
    assert cs[0]["score"] == 14


# ---------- Behaviour tests ----------


def test_since_cutoff_short_circuits_on_old_post(ua_env: None) -> None:
    cutoff = datetime(2025, 6, 1, tzinfo=UTC).timestamp()
    posts = [
        _post(sid="new", created_utc=cutoff + 3600),
        _post(sid="old", created_utc=cutoff - 3600),
    ]

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/new.json"):
            return httpx.Response(200, json=_listing(posts))
        return httpx.Response(200, json=_comment_thread([]))

    with patch("time.sleep", _no_sleep):
        src = RedditSource(
            config=RedditConfig.from_env(),
            subreddits=["algotrading"],
            client=_make_client(handler),
        )
        items = list(src.fetch(since=datetime.fromtimestamp(cutoff, tz=UTC), limit=10))

    assert len(items) == 1
    assert items[0].external_id == "new"


def test_subreddit_404_logged_and_others_continue(
    ua_env: None, caplog: pytest.LogCaptureFixture
) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if "/r/broken/" in req.url.path:
            return httpx.Response(404, json={"error": 404, "message": "Not Found"})
        if req.url.path.endswith("/new.json"):
            return httpx.Response(200, json=_listing([_post(sid="g1")]))
        return httpx.Response(200, json=_comment_thread([]))

    with patch("time.sleep", _no_sleep), caplog.at_level(logging.WARNING):
        src = RedditSource(
            config=RedditConfig.from_env(),
            subreddits=["broken", "algotrading"],
            client=_make_client(handler),
        )
        items = list(src.fetch(since=None, limit=5))

    assert len(items) == 1
    assert items[0].external_id == "g1"
    assert any("broken" in rec.message for rec in caplog.records)


def test_zero_comment_post_skips_comment_fetch(ua_env: None) -> None:
    """Posts with `num_comments=0` must not trigger a comments request."""
    seen_paths: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen_paths.append(req.url.path)
        if req.url.path.endswith("/new.json"):
            return httpx.Response(200, json=_listing([_post(sid="zc", num_comments=0)]))
        return httpx.Response(200, json=_comment_thread([]))

    with patch("time.sleep", _no_sleep):
        src = RedditSource(
            config=RedditConfig.from_env(),
            subreddits=["algotrading"],
            client=_make_client(handler),
        )
        items = list(src.fetch(since=None, limit=1))

    assert len(items) == 1
    assert items[0].extras["comments"] == []
    assert all("/comments/" not in p for p in seen_paths)


def test_empty_subreddit_returns_no_items(ua_env: None) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/new.json"):
            return httpx.Response(200, json=_listing([]))
        return httpx.Response(200, json=_comment_thread([]))

    with patch("time.sleep", _no_sleep):
        src = RedditSource(
            config=RedditConfig.from_env(),
            subreddits=["algotrading"],
            client=_make_client(handler),
        )
        items = list(src.fetch(since=None, limit=10))

    assert items == []


# ---------- HTTP plumbing tests ----------


def test_outgoing_requests_carry_required_headers_and_raw_json(ua_env: None) -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.url.path.endswith("/new.json"):
            return httpx.Response(200, json=_listing([_post(sid="h")]))
        return httpx.Response(200, json=_comment_thread([]))

    # Build the client with the source's normal __init__ path so headers come
    # from the implementation, not the test helper.
    real_client = httpx.Client(
        transport=httpx.MockTransport(handler),
        timeout=15.0,
        headers={
            "User-Agent": "vibe-quant-test:0.1 (by /u/tester)",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
        },
    )
    with patch("time.sleep", _no_sleep):
        src = RedditSource(
            config=RedditConfig.from_env(),
            subreddits=["algotrading"],
            client=real_client,
        )
        list(src.fetch(since=None, limit=1))

    assert len(seen) >= 2  # listing + comments
    for req in seen:
        assert "vibe-quant-test" in req.headers["user-agent"]
        assert req.headers["accept"] == "application/json"
        assert "en-US" in req.headers["accept-language"]
        assert "gzip" in req.headers["accept-encoding"]
        assert req.url.params.get("raw_json") == "1"


def test_default_user_agent_used_when_env_unset(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv(ENV_REDDIT_USER_AGENT, raising=False)
    with caplog.at_level(logging.WARNING):
        cfg = RedditConfig.from_env()
    assert cfg.user_agent == DEFAULT_USER_AGENT
    assert cfg.using_default is True
    assert any(ENV_REDDIT_USER_AGENT in rec.message for rec in caplog.records)


def test_rate_limit_floor_enforced_between_requests(ua_env: None) -> None:
    fake_now = [1000.0]

    def fake_monotonic() -> float:
        return fake_now[0]

    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        fake_now[0] += seconds

    def handler(req: httpx.Request) -> httpx.Response:
        # Each handled request advances the clock by 0.01s (request "duration")
        fake_now[0] += 0.01
        if req.url.path.endswith("/new.json"):
            return httpx.Response(200, json=_listing([_post(sid="rl1"), _post(sid="rl2")]))
        return httpx.Response(200, json=_comment_thread([]))

    with patch("time.monotonic", fake_monotonic), patch("time.sleep", fake_sleep):
        src = RedditSource(
            config=RedditConfig.from_env(),
            subreddits=["algotrading"],
            client=_make_client(handler),
        )
        list(src.fetch(since=None, limit=2))

    # First request: no sleep (last_request_at == 0). Subsequent requests must
    # sleep approximately MIN_REQUEST_INTERVAL_S each.
    rate_limit_sleeps = [s for s in sleeps if s >= MIN_REQUEST_INTERVAL_S - 0.1]
    assert len(rate_limit_sleeps) >= 2  # at least 2 inter-request sleeps for 3+ calls


def test_429_with_retry_after_header_sleeps_then_succeeds(ua_env: None) -> None:
    call_count = {"n": 0}
    sleeps: list[float] = []

    def handler(req: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if req.url.path.endswith("/new.json") and call_count["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "5"}, json={})
        if req.url.path.endswith("/new.json"):
            return httpx.Response(200, json=_listing([_post(sid="r")]))
        return httpx.Response(200, json=_comment_thread([]))

    def fake_sleep(s: float) -> None:
        sleeps.append(s)

    with patch("time.sleep", fake_sleep):
        src = RedditSource(
            config=RedditConfig.from_env(),
            subreddits=["algotrading"],
            client=_make_client(handler),
        )
        items = list(src.fetch(since=None, limit=1))

    assert len(items) == 1
    assert any(4.5 < s < 6.0 for s in sleeps), f"expected ~5.5s retry sleep, saw {sleeps}"


def test_429_without_retry_after_exhausts_retries_and_skips(
    ua_env: None, caplog: pytest.LogCaptureFixture
) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={})

    with patch("time.sleep", _no_sleep), caplog.at_level(logging.WARNING):
        src = RedditSource(
            config=RedditConfig.from_env(),
            subreddits=["algotrading"],
            client=_make_client(handler),
        )
        items = list(src.fetch(since=None, limit=1))

    assert items == []
    assert any("exhausted" in rec.message for rec in caplog.records)


def test_no_praw_imports_remain() -> None:
    """Regression guard: praw must not creep back into the module."""
    import inspect

    from vibe_quant.research.sources import reddit as _mod

    contents = inspect.getsource(_mod)
    assert "import praw" not in contents
    assert "import prawcore" not in contents
