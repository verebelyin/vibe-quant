"""HackerNewsSource unit tests — a fake client, no network access."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from vibe_quant.research.schema import RawItem
from vibe_quant.research.sources.hackernews import (
    DEFAULT_QUERIES,
    ENV_HN_QUERIES,
    SEARCH_URL,
    HackerNewsSource,
    _strip_html,
    queries_from_env,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

_NOW = 1_700_000_000  # 2023-11-14T22:13:20Z


# ---------- fakes ----------


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: Any = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.test")
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=request,
                response=httpx.Response(self.status_code, request=request),
            )


class _NonJsonResponse:
    """A 200 whose body is not JSON — ``json()`` raises ``ValueError`` like httpx."""

    status_code = 200
    headers: Mapping[str, str] = {}

    def json(self) -> Any:
        raise ValueError("Expecting value: line 1 column 1 (char 0)")

    def raise_for_status(self) -> None:
        return None


class _FakeClient:
    def __init__(self, handler: Callable[[str, Mapping[str, str] | None], Any]) -> None:
        self.calls: list[tuple[str, Mapping[str, str] | None]] = []
        self.closed = False
        self._handler = handler

    def get(self, url: str, *, params: Mapping[str, str] | None = None) -> Any:
        self.calls.append((url, params))
        return self._handler(url, params)

    def close(self) -> None:
        self.closed = True


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


def _story(object_id: str = "1", **overrides: Any) -> dict[str, Any]:
    hit: dict[str, Any] = {
        "objectID": object_id,
        "title": "Story",
        "url": "https://example.com/a",
        "story_text": None,
        "author": "pg",
        "created_at_i": _NOW,
        "points": 7,
        "num_comments": 0,
    }
    hit.update(overrides)
    return hit


def _item(children: list[dict[str, Any]]) -> dict[str, Any]:
    return {"id": 1, "children": children}


def _comment(comment_id: int, author: str | None, text: str | None) -> dict[str, Any]:
    return {"id": comment_id, "author": author, "text": text, "children": []}


def _router(
    search_hits: dict[str, list[dict[str, Any]]],
    items: dict[str, dict[str, Any]] | None = None,
) -> Callable[[str, Mapping[str, str] | None], Any]:
    item_map = items or {}

    def handler(url: str, params: Mapping[str, str] | None) -> Any:
        if url == SEARCH_URL:
            query = (params or {}).get("query")
            return _FakeResponse(200, {"hits": search_hits.get(query or "", [])})
        if url.startswith("https://hn.algolia.com/api/v1/items/"):
            object_id = url.rsplit("/", 1)[-1]
            return _FakeResponse(200, item_map.get(object_id, {"id": object_id, "children": []}))
        raise AssertionError(f"unexpected url {url}")

    return handler


def _source(
    handler: Callable[[str, Mapping[str, str] | None], Any],
    queries: list[str] | tuple[str, ...] = ("q",),
) -> tuple[HackerNewsSource, _FakeClient, _Clock]:
    clock = _Clock()
    client = _FakeClient(handler)
    return (
        HackerNewsSource(queries=list(queries), client=client, clock=clock.now, sleep=clock.sleep),
        client,
        clock,
    )


# ---------- request shape ----------


def test_search_request_uses_search_by_date_story_tag_and_caps_page_at_100() -> None:
    src, client, _ = _source(_router({"quant trading": []}), queries=["quant trading"])
    assert list(src.fetch(None, 250)) == []
    assert client.calls == [
        (SEARCH_URL, {"query": "quant trading", "tags": "story", "hitsPerPage": "100"})
    ]


def test_search_request_uses_limit_below_the_cap() -> None:
    src, client, _ = _source(_router({"q": []}), queries=["q"])
    list(src.fetch(None, 3))
    assert client.calls[0][1] == {"query": "q", "tags": "story", "hitsPerPage": "3"}


# ---------- translation ----------


def test_hit_translates_to_raw_item_with_comments() -> None:
    hits = {
        "q": [
            _story(
                object_id="42",
                title="Mean reversion",
                url="https://ex.com/p",
                story_text="<p>RSI &amp; MA</p>",
                author="pg",
                created_at_i=_NOW,
                points=11,
                num_comments=2,
            )
        ]
    }
    items = {"42": _item([_comment(1, "alice", "<p>Try RSI &lt; 30</p>")])}
    src, _, _ = _source(_router(hits, items))

    out = list(src.fetch(None, 10))
    assert len(out) == 1
    item = out[0]
    assert isinstance(item, RawItem)
    assert item.source == "hackernews"
    assert item.external_id == "42"
    assert item.url == "https://ex.com/p"
    assert item.title == "Mean reversion"
    assert item.body == "RSI & MA"
    assert item.author == "pg"
    assert item.posted_at == datetime.fromtimestamp(_NOW, tz=UTC)
    assert item.posted_at is not None and item.posted_at.tzinfo is not None
    assert item.score == 11
    assert item.extras == {
        "query": "q",
        "num_comments": 2,
        "comments": [{"author": "alice", "body": "Try RSI < 30", "score": None}],
        "hn_url": "https://news.ycombinator.com/item?id=42",
    }


def test_missing_author_becomes_none_and_empty_story_text_becomes_empty_string() -> None:
    hits = {"q": [_story("9", author=None, story_text=None)]}
    src, _, _ = _source(_router(hits))
    item = next(iter(src.fetch(None, 10)))
    assert item.author is None
    assert item.body == ""


def test_stories_deduped_across_queries() -> None:
    hits = {"a": [_story("1"), _story("2")], "b": [_story("2"), _story("3")]}
    src, _, _ = _source(_router(hits), queries=["a", "b"])
    out = list(src.fetch(None, 10))
    assert [i.external_id for i in out] == ["1", "2", "3"]
    assert [i.extras["query"] for i in out] == ["a", "a", "b"]


def test_since_skips_older_stories() -> None:
    hits = {"q": [_story("n", created_at_i=_NOW), _story("o", created_at_i=_NOW - 10)]}
    src, _, _ = _source(_router(hits))
    since = datetime.fromtimestamp(_NOW - 5, tz=UTC)
    assert [i.external_id for i in src.fetch(since, 10)] == ["n"]


def test_url_falls_back_to_hn_permalink_when_empty() -> None:
    src, _, _ = _source(_router({"q": [_story("7", url="")]}))
    item = next(iter(src.fetch(None, 10)))
    assert item.url == "https://news.ycombinator.com/item?id=7"
    assert item.extras["hn_url"] == item.url


# ---------- comments ----------


def test_paragraph_breaks_survive_stripping() -> None:
    # HN separates paragraphs with a bare <p> and lines with <br>; those must
    # become newlines, not vanish, or two paragraphs fuse into one run-on line.
    assert _strip_html('<p>first para.<p>second &gt; x<br>y <a href="u">link</a>') == (
        "first para.\nsecond > x\ny link"
    )
    assert _strip_html("<p>a</p><p>b</p><br><p></p><p>c</p>") == "a\nb\nc"


def test_comments_are_top_level_stripped_capped_and_in_api_order() -> None:
    children = [_comment(i, f"u{i}", f"<p>body {i}</p>") for i in range(12)]
    children[0]["children"] = [_comment(999, "nested", "ignore me")]
    children.insert(1, _comment(50, "ghost", ""))
    children.insert(2, _comment(51, None, None))
    items = {"1": _item(children)}
    src, _, _ = _source(_router({"q": [_story("1", num_comments=len(children))]}, items))

    item = next(iter(src.fetch(None, 10)))
    comments = item.extras["comments"]
    assert len(comments) == 10
    assert [c["author"] for c in comments] == [f"u{i}" for i in range(10)]
    assert comments[0] == {"author": "u0", "body": "body 0", "score": None}
    assert all(c["score"] is None for c in comments)
    assert all("ignore me" not in c["body"] for c in comments)


def test_comments_not_fetched_when_num_comments_is_zero() -> None:
    src, client, _ = _source(_router({"q": [_story("1", num_comments=0)]}))
    list(src.fetch(None, 10))
    assert [url for url, _ in client.calls] == [SEARCH_URL]


# ---------- failure handling ----------


def test_failed_query_is_logged_and_skipped_when_others_succeed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(url: str, params: Mapping[str, str] | None) -> Any:
        if (params or {}).get("query") == "bad":
            raise httpx.ConnectError("boom")
        return _FakeResponse(200, {"hits": [_story("1")]})

    src, _, _ = _source(handler, queries=["good", "bad"])
    with caplog.at_level(logging.WARNING):
        out = list(src.fetch(None, 10))
    assert [i.external_id for i in out] == ["1"]
    assert any("bad" in record.message for record in caplog.records)


def test_non_json_response_skips_that_query_and_keeps_going(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(url: str, params: Mapping[str, str] | None) -> Any:
        if (params or {}).get("query") == "bad":
            return _NonJsonResponse()
        return _FakeResponse(200, {"hits": [_story("1")]})

    src, _, _ = _source(handler, queries=["bad", "good"])
    with caplog.at_level(logging.WARNING):
        out = list(src.fetch(None, 10))
    assert [i.external_id for i in out] == ["1"]
    assert any("bad" in record.message for record in caplog.records)


def test_all_queries_failed_and_nothing_yielded_raises_runtime_error() -> None:
    def handler(url: str, params: Mapping[str, str] | None) -> Any:
        raise httpx.ConnectError("down")

    src, _, _ = _source(handler, queries=["a", "b"])
    with pytest.raises(RuntimeError) as exc:
        list(src.fetch(None, 10))
    assert "a" in str(exc.value)
    assert "b" in str(exc.value)


def test_retryable_429_backs_off_then_raises_for_that_query() -> None:
    src, client, clock = _source(lambda url, params: _FakeResponse(429))
    with pytest.raises(RuntimeError):
        list(src.fetch(None, 10))
    assert clock.sleeps == [1.0, 2.0, 4.0]
    assert len(client.calls) == 4


def test_retryable_5xx_is_retried_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(url: str, params: Mapping[str, str] | None) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse(503)
        return _FakeResponse(200, {"hits": [_story("1")]})

    src, client, clock = _source(handler)
    assert [i.external_id for i in src.fetch(None, 10)] == ["1"]
    assert clock.sleeps == [1.0]
    assert len(client.calls) == 2


def test_timeout_is_retried() -> None:
    calls = {"n": 0}

    def handler(url: str, params: Mapping[str, str] | None) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("slow")
        return _FakeResponse(200, {"hits": []})

    src, _, clock = _source(handler)
    assert list(src.fetch(None, 10)) == []
    assert clock.sleeps == [1.0]


def test_non_retryable_4xx_fails_query_without_retrying() -> None:
    src, client, clock = _source(lambda url, params: _FakeResponse(400))
    with pytest.raises(RuntimeError):
        list(src.fetch(None, 10))
    assert len(client.calls) == 1
    assert clock.sleeps == []


def test_rate_limit_enforces_one_second_floor_between_requests() -> None:
    src, _, clock = _source(_router({"a": [], "b": []}), queries=["a", "b"])
    list(src.fetch(None, 10))
    assert clock.sleeps == [1.0]


def test_rate_limit_counts_failed_requests_too() -> None:
    # A non-retryable 4xx skips its query; the next query must still wait out
    # the 1s floor, or a failing endpoint lets us burst through the limit.
    def handler(url: str, params: Mapping[str, str] | None) -> Any:
        if (params or {}).get("query") == "bad":
            return _FakeResponse(400)
        return _FakeResponse(200, {"hits": [_story("1")]})

    src, _, clock = _source(handler, queries=["bad", "good"])
    assert [i.external_id for i in src.fetch(None, 10)] == ["1"]
    assert clock.sleeps == [1.0]


# ---------- construction ----------


def test_queries_from_env_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_HN_QUERIES, raising=False)
    assert queries_from_env() == DEFAULT_QUERIES


def test_queries_from_env_parses_and_strips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_HN_QUERIES, " a , b ,, c ")
    assert queries_from_env() == ("a", "b", "c")


def test_queries_from_env_blank_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_HN_QUERIES, " , ")
    assert queries_from_env() == DEFAULT_QUERIES


def test_constructor_reads_env_when_queries_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_HN_QUERIES, "only")
    clock = _Clock()
    client = _FakeClient(_router({"only": [_story("1")]}))
    src = HackerNewsSource(client=client, clock=clock.now, sleep=clock.sleep)
    assert [i.external_id for i in src.fetch(None, 5)] == ["1"]
    assert (client.calls[0][1] or {})["query"] == "only"


def test_close_delegates_to_client() -> None:
    src, client, _ = _source(_router({}))
    src.close()
    assert client.closed is True


# ---------- registry ----------


def test_registry_lists_hackernews() -> None:
    from vibe_quant.research.sources import list_sources, load_builtin_sources

    load_builtin_sources(force=True)
    assert "hackernews" in list_sources()
