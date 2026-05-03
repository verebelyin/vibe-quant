"""Configuration for research sources."""

from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass

ENV_REDDIT_USER_AGENT = "REDDIT_USER_AGENT"
ENV_REDDIT_SUBREDDITS = "REDDIT_SUBREDDITS"

DEFAULT_SUBREDDITS = ("algotrading",)
DEFAULT_USER_AGENT = "vibe-quant-research:0.1 (by anonymous)"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RedditConfig:
    """Reddit unauthenticated `.json` endpoint config (User-Agent only)."""

    user_agent: str
    using_default: bool

    @classmethod
    def from_env(cls) -> RedditConfig:
        ua = os.getenv(ENV_REDDIT_USER_AGENT)
        if ua:
            return cls(user_agent=ua, using_default=False)
        _warn_default_ua_once()
        return cls(user_agent=DEFAULT_USER_AGENT, using_default=True)


@functools.lru_cache(maxsize=1)
def _warn_default_ua_once() -> None:
    logger.warning(
        "%s not set — using default %r. Reddit may throttle generic UAs; "
        "set %s to '<platform>:<app-id>:<version> (by /u/<your-username>)'.",
        ENV_REDDIT_USER_AGENT,
        DEFAULT_USER_AGENT,
        ENV_REDDIT_USER_AGENT,
    )


def subreddits_from_env() -> list[str]:
    """Parse REDDIT_SUBREDDITS=foo,bar,baz with fallback to ['algotrading']."""
    raw = os.getenv(ENV_REDDIT_SUBREDDITS, "").strip()
    if not raw:
        return list(DEFAULT_SUBREDDITS)
    return [s.strip() for s in raw.split(",") if s.strip()]
