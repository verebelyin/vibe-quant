"""Reddit source: scrape r/<subreddit>/new + top comments via PRAW."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

import praw
import prawcore

from vibe_quant.research.config import RedditConfig, subreddits_from_env
from vibe_quant.research.schema import RawItem
from vibe_quant.research.sources import register_source

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

TOP_COMMENT_COUNT = 10
DELETED_AUTHOR_PLACEHOLDER = "[deleted]"


@register_source("reddit")
class RedditSource:
    """Read-only script-app Reddit source over r/<sub>/new."""

    name: ClassVar[str] = "reddit"

    def __init__(
        self,
        config: RedditConfig | None = None,
        subreddits: list[str] | None = None,
    ) -> None:
        self._config = config or RedditConfig.from_env()
        self._subreddits = subreddits if subreddits is not None else subreddits_from_env()
        self._reddit = praw.Reddit(
            client_id=self._config.client_id,
            client_secret=self._config.client_secret,
            user_agent=self._config.user_agent,
            check_for_async=False,
        )
        self._reddit.read_only = True

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
            except prawcore.exceptions.PrawcoreException as e:
                logger.warning("subreddit %r failed: %s", subreddit, e)
                continue
            except praw.exceptions.RedditAPIException as e:
                logger.warning("subreddit %r reddit-api error: %s", subreddit, e)
                continue

    def _fetch_subreddit(
        self, subreddit: str, since_ts: float | None, limit: int
    ) -> Iterable[RawItem]:
        sub = self._reddit.subreddit(subreddit)
        for submission in sub.new(limit=limit):
            try:
                if since_ts is not None and submission.created_utc < since_ts:
                    return  # /new is reverse-chronological, so we can stop here
                yield self._submission_to_raw_item(submission, subreddit)
            except (prawcore.exceptions.PrawcoreException, praw.exceptions.RedditAPIException) as e:
                logger.warning(
                    "submission %s in r/%s skipped: %s",
                    getattr(submission, "id", "?"),
                    subreddit,
                    e,
                )
                continue

    def _submission_to_raw_item(self, submission: Any, subreddit: str) -> RawItem:
        author = submission.author
        if author is None:
            author_str: str | None = None
        else:
            author_name = getattr(author, "name", None)
            if author_name is None or author_name == DELETED_AUTHOR_PLACEHOLDER:
                author_str = None
            else:
                author_str = author_name

        comments = self._top_comments(submission)
        flair = getattr(submission, "link_flair_text", None)

        return RawItem(
            source="reddit",
            external_id=str(submission.id),
            url=f"https://www.reddit.com{getattr(submission, 'permalink', '')}",
            title=str(getattr(submission, "title", "") or ""),
            body=str(getattr(submission, "selftext", "") or ""),
            author=author_str,
            posted_at=datetime.fromtimestamp(float(submission.created_utc), tz=UTC),
            score=int(getattr(submission, "score", 0) or 0),
            extras={
                "subreddit": subreddit,
                "flair": flair,
                "num_comments": int(getattr(submission, "num_comments", 0) or 0),
                "comments": comments,
            },
        )

    def _top_comments(self, submission: Any) -> list[dict[str, object]]:
        """Return up to `TOP_COMMENT_COUNT` top-level comments by score."""
        try:
            comment_forest = submission.comments
            comment_forest.replace_more(limit=0)
            top_level = list(comment_forest)
        except (
            prawcore.exceptions.PrawcoreException,
            praw.exceptions.RedditAPIException,
            AttributeError,
        ) as e:
            logger.warning(
                "comments fetch failed for %s: %s",
                getattr(submission, "id", "?"),
                e,
            )
            return []

        top_level.sort(key=lambda c: int(getattr(c, "score", 0) or 0), reverse=True)
        out: list[dict[str, object]] = []
        for c in top_level[:TOP_COMMENT_COUNT]:
            author_obj = getattr(c, "author", None)
            author_name = getattr(author_obj, "name", None) if author_obj is not None else None
            if author_name == DELETED_AUTHOR_PLACEHOLDER:
                author_name = None
            out.append(
                {
                    "author": author_name,
                    "body": str(getattr(c, "body", "") or ""),
                    "score": int(getattr(c, "score", 0) or 0),
                }
            )
        return out
