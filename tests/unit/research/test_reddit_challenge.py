"""Challenge-based Reddit access unit tests — fully mocked, no network.

Covers the glance-style stopgap: JS-challenge solve -> loid cookie -> browser-UA
`.json` requests, plus the RedditSource wiring that opts into it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import pytest

from vibe_quant.research.config import ENV_REDDIT_USE_CHALLENGE, RedditConfig
from vibe_quant.research.sources._reddit_challenge import (
    HOMEPAGE_URL,
    ChallengeClient,
    RedditChallengeError,
)
from vibe_quant.research.sources.reddit import RedditSource

if TYPE_CHECKING:
    from collections.abc import Mapping

# A challenge page that matches glance's regexes.
_CHALLENGE_PAGE = (
    '<html><body><script>await(async e => e + e)("ABC123");</script>'
    '<input type="hidden" name="token" value="tok-xyz"></body></html>'
)


class _FakeCookies:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get(self, name: str) -> str | None:
        return self._store.get(name)

    def set(self, name: str, value: str, domain: str | None = None) -> None:
        self._store[name] = value


class _FakeResponse:
    def __init__(self, status_code: int, *, text: str = "", content: bytes = b"") -> None:
        self.status_code = status_code
        self.text = text
        self.content = content or text.encode()
        self.headers: dict[str, str] = {}
        self.url = "https://www.reddit.com/"


class _FakeSession:
    """Records requests and replays a scripted list of responses."""

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = responses
        self.cookies = _FakeCookies()
        self.requests: list[tuple[str, Mapping[str, str], str]] = []
        self.closed = False

    def get(
        self,
        url: str,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        ua = (headers or {}).get("User-Agent", "")
        self.requests.append((url, dict(params or {}), ua))
        return self._responses.pop(0)

    def close(self) -> None:
        self.closed = True


def _solved_session(json_body: bytes = b'{"data": {"children": []}}') -> _FakeSession:
    """A session that solves the challenge then serves one JSON response,
    and sets the loid cookie on the challenge-submit step."""
    session = _FakeSession(
        [
            _FakeResponse(200, text=_CHALLENGE_PAGE),  # homepage/challenge
            _FakeResponse(200, text="ok"),  # challenge submit
            _FakeResponse(200, content=json_body),  # the .json request
        ]
    )

    original_get = session.get

    def get_with_cookie(
        url: str,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        resp = original_get(url, params, headers, timeout)
        # The real submit step sets loid via Set-Cookie; emulate on 2nd call.
        if len(session.requests) == 2:
            session.cookies.set("loid", "loid-abc", domain=".reddit.com")
        return resp

    session.get = get_with_cookie  # type: ignore[method-assign]
    return session


# ---------- ChallengeClient ----------


def test_challenge_flow_obtains_loid_and_fetches_json() -> None:
    session = _solved_session()
    client = ChallengeClient(session=session)

    resp = client.get("https://www.reddit.com/r/algotrading/new.json", params={"limit": "5"})

    assert resp.status_code == 200
    assert resp.json() == {"data": {"children": []}}
    # Three calls: challenge page, challenge submit, then the json request.
    assert len(session.requests) == 3
    assert session.requests[0][0] == HOMEPAGE_URL
    assert "js_challenge" in session.requests[1][1]
    assert session.requests[1][1]["solution"] == "ABC123ABC123"  # e + e
    # The json request carried a browser UA.
    assert "Firefox" in session.requests[2][2]


def test_loid_cookie_is_cached_across_requests() -> None:
    session = _solved_session()
    # Append a second json response so a 2nd get() can be served.
    session._responses.append(_FakeResponse(200, content=b'{"data": {"children": []}}'))
    client = ChallengeClient(session=session)

    client.get("https://www.reddit.com/r/a/new.json")
    client.get("https://www.reddit.com/r/b/new.json")

    # Challenge solved once (2 calls); then 2 json calls = 4 total, not 5.
    challenge_page_hits = [r for r in session.requests if r[0] == HOMEPAGE_URL]
    assert len(challenge_page_hits) == 2  # one page + one submit, not repeated


def test_missing_challenge_raises() -> None:
    session = _FakeSession([_FakeResponse(200, text="<html>no challenge here</html>")])
    client = ChallengeClient(session=session)

    with pytest.raises(RedditChallengeError, match="challenge/token not found"):
        client.get("https://www.reddit.com/r/a/new.json")


def test_no_loid_after_submit_raises() -> None:
    # Challenge present, submit OK, but cookie never set.
    session = _FakeSession(
        [_FakeResponse(200, text=_CHALLENGE_PAGE), _FakeResponse(200, text="ok")]
    )
    client = ChallengeClient(session=session)

    with pytest.raises(RedditChallengeError, match="no loid cookie"):
        client.get("https://www.reddit.com/r/a/new.json")


def test_challenge_page_non_200_raises() -> None:
    session = _FakeSession([_FakeResponse(403, text="blocked")])
    client = ChallengeClient(session=session)

    with pytest.raises(RedditChallengeError, match="status 403"):
        client.get("https://www.reddit.com/r/a/new.json")


# ---------- RedditSource wiring ----------


def test_source_uses_challenge_client_when_flag_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_REDDIT_USE_CHALLENGE, "1")
    source = RedditSource(config=RedditConfig(user_agent="x", using_default=False))
    assert isinstance(source._client, ChallengeClient)


def test_source_uses_httpx_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_REDDIT_USE_CHALLENGE, raising=False)
    source = RedditSource(config=RedditConfig(user_agent="x", using_default=False))
    assert isinstance(source._client, httpx.Client)


def test_challenge_source_yields_items_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("time.sleep", lambda _: None)
    post = {
        "kind": "t3",
        "data": {
            "id": "p1",
            "title": "backtest results",
            "selftext": "body",
            "score": 5,
            "created_utc": 1735689600.0,
            "permalink": "/r/algotrading/comments/p1/",
            "num_comments": 0,
            "author": "u/x",
        },
    }
    listing = b'{"data": {"children": [%s], "after": null}}' % _dumps(post)
    session = _solved_session(json_body=listing)
    client = ChallengeClient(session=session)
    source = RedditSource(
        config=RedditConfig(user_agent="x", using_default=False),
        subreddits=["algotrading"],
        client=client,
    )

    items = list(source.fetch(since=None, limit=5))

    assert len(items) == 1
    assert items[0].title == "backtest results"
    assert items[0].external_id == "p1"


def _dumps(obj: Any) -> bytes:
    import json

    return json.dumps(obj).encode()
