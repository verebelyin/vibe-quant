"""RedditSource unit tests — fully mocked, no network."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vibe_quant.alerts.telegram import ConfigurationError
from vibe_quant.research.config import RedditConfig


def _make_creds() -> None:
    os.environ["REDDIT_CLIENT_ID"] = "id"
    os.environ["REDDIT_CLIENT_SECRET"] = "secret"
    os.environ["REDDIT_USER_AGENT"] = "vibe-quant/test"


def _clear_creds() -> None:
    for v in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"):
        os.environ.pop(v, None)


@pytest.fixture
def reddit_creds() -> None:
    _make_creds()
    yield
    _clear_creds()


def _make_comment(author_name: str | None, body: str, score: int) -> SimpleNamespace:
    author = SimpleNamespace(name=author_name) if author_name is not None else None
    return SimpleNamespace(author=author, body=body, score=score)


def _make_submission(
    *,
    sid: str,
    title: str,
    selftext: str,
    score: int,
    created_utc: float,
    author_name: str | None,
    comments: list[SimpleNamespace],
    flair: str | None = None,
) -> MagicMock:
    sub = MagicMock()
    sub.id = sid
    sub.title = title
    sub.selftext = selftext
    sub.score = score
    sub.created_utc = created_utc
    sub.permalink = f"/r/algotrading/comments/{sid}/"
    sub.link_flair_text = flair
    sub.num_comments = len(comments)
    sub.author = SimpleNamespace(name=author_name) if author_name is not None else None
    forest = MagicMock()
    forest.__iter__.return_value = iter(comments)
    forest.replace_more = MagicMock(return_value=None)
    sub.comments = forest
    return sub


def test_missing_creds_raises_configuration_error() -> None:
    _clear_creds()
    from vibe_quant.research.sources.reddit import RedditSource

    with pytest.raises(ConfigurationError, match="REDDIT_CLIENT_ID"):
        RedditSource()


def test_fetch_translates_submissions(reddit_creds: None) -> None:
    submissions = [
        _make_submission(
            sid="aaa",
            title="Mean reversion idea",
            selftext="RSI < 30 entry, RSI > 70 exit",
            score=42,
            created_utc=1735689600.0,  # 2025-01-01 UTC
            author_name="u/quantnerd",
            comments=[
                _make_comment("u/foo", "+1", 5),
                _make_comment("u/bar", "doesn't work", 12),
                _make_comment(None, "[deleted]", 1),
            ],
            flair="Strategy",
        ),
        _make_submission(
            sid="bbb",
            title="Link only",
            selftext="",
            score=8,
            created_utc=1735693200.0,
            author_name=None,  # deleted author → submission.author is None
            comments=[],
        ),
    ]
    fake_subreddit = MagicMock()
    fake_subreddit.new.return_value = iter(submissions)
    fake_reddit = MagicMock()
    fake_reddit.subreddit.return_value = fake_subreddit

    with patch("praw.Reddit", return_value=fake_reddit):
        from vibe_quant.research.sources.reddit import RedditSource

        src = RedditSource(
            config=RedditConfig.from_env(), subreddits=["algotrading"]
        )
        items = list(src.fetch(since=None, limit=10))

    assert len(items) == 2

    a = items[0]
    assert a.source == "reddit"
    assert a.external_id == "aaa"
    assert a.title == "Mean reversion idea"
    assert a.body.startswith("RSI < 30")
    assert a.score == 42
    assert a.author == "u/quantnerd"
    assert a.posted_at == datetime.fromtimestamp(1735689600.0, tz=UTC)
    # Top comments sorted by score desc; deleted author normalized to None
    comments = a.extras["comments"]
    assert [c["score"] for c in comments] == [12, 5, 1]
    assert comments[2]["author"] is None
    assert a.extras["subreddit"] == "algotrading"
    assert a.extras["flair"] == "Strategy"

    b = items[1]
    assert b.body == ""  # link-only post
    assert b.author is None  # deleted author
    assert b.extras["comments"] == []  # 0 comments


def test_fetch_stops_at_since(reddit_creds: None) -> None:
    cutoff = datetime(2025, 6, 1, tzinfo=UTC).timestamp()
    submissions = [
        _make_submission(
            sid="new",
            title="recent",
            selftext="",
            score=1,
            created_utc=cutoff + 3600,
            author_name="u/a",
            comments=[],
        ),
        _make_submission(
            sid="old",
            title="old",
            selftext="",
            score=1,
            created_utc=cutoff - 3600,
            author_name="u/a",
            comments=[],
        ),
    ]
    fake_subreddit = MagicMock()
    fake_subreddit.new.return_value = iter(submissions)
    fake_reddit = MagicMock()
    fake_reddit.subreddit.return_value = fake_subreddit

    with patch("praw.Reddit", return_value=fake_reddit):
        from vibe_quant.research.sources.reddit import RedditSource

        src = RedditSource(
            config=RedditConfig.from_env(), subreddits=["algotrading"]
        )
        since = datetime.fromtimestamp(cutoff, tz=UTC)
        items = list(src.fetch(since=since, limit=10))

    assert len(items) == 1
    assert items[0].external_id == "new"


def test_bad_subreddit_does_not_kill_run(reddit_creds: None) -> None:
    import prawcore

    good_submissions = [
        _make_submission(
            sid="g1",
            title="ok",
            selftext="",
            score=1,
            created_utc=1735689600.0,
            author_name="u/a",
            comments=[],
        )
    ]

    fake_reddit = MagicMock()

    def subreddit_router(name: str) -> MagicMock:
        if name == "broken":
            bad = MagicMock()
            bad.new.side_effect = prawcore.exceptions.NotFound(
                response=MagicMock(status_code=404, headers={})
            )
            return bad
        good = MagicMock()
        good.new.return_value = iter(good_submissions)
        return good

    fake_reddit.subreddit.side_effect = subreddit_router

    with patch("praw.Reddit", return_value=fake_reddit):
        from vibe_quant.research.sources.reddit import RedditSource

        src = RedditSource(
            config=RedditConfig.from_env(),
            subreddits=["broken", "algotrading"],
        )
        items = list(src.fetch(since=None, limit=5))

    assert len(items) == 1
    assert items[0].external_id == "g1"


def test_top_comments_capped_at_10_and_sorted(reddit_creds: None) -> None:
    comments = [_make_comment(f"u/{i}", "x", i) for i in range(15)]
    submission = _make_submission(
        sid="cap",
        title="t",
        selftext="b",
        score=1,
        created_utc=1735689600.0,
        author_name="u/a",
        comments=comments,
    )
    fake_subreddit = MagicMock()
    fake_subreddit.new.return_value = iter([submission])
    fake_reddit = MagicMock()
    fake_reddit.subreddit.return_value = fake_subreddit

    with patch("praw.Reddit", return_value=fake_reddit):
        from vibe_quant.research.sources.reddit import RedditSource

        src = RedditSource(
            config=RedditConfig.from_env(), subreddits=["algotrading"]
        )
        items = list(src.fetch(since=None, limit=1))

    cs = items[0].extras["comments"]
    assert len(cs) == 10
    assert [c["score"] for c in cs] == sorted([c["score"] for c in cs], reverse=True)
    assert cs[0]["score"] == 14
