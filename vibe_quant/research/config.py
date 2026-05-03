"""Configuration for research sources."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

ENV_REDDIT_USER_AGENT = "REDDIT_USER_AGENT"
ENV_REDDIT_SUBREDDITS = "REDDIT_SUBREDDITS"

DEFAULT_SUBREDDITS = ("algotrading",)
DEFAULT_USER_AGENT = "vibe-quant-research:0.1 (by anonymous)"

# Vars from the praw era. We no longer use them (.json endpoint is unauthenticated)
# but warn so users can scrub stale .env files.
DEPRECATED_REDDIT_VARS = ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RedditConfig:
    """Reddit unauthenticated `.json` endpoint config (User-Agent only)."""

    user_agent: str
    using_default: bool

    @classmethod
    def from_env(cls) -> RedditConfig:
        _warn_deprecated_creds_once()
        ua = os.getenv(ENV_REDDIT_USER_AGENT)
        if ua:
            return cls(user_agent=ua, using_default=False)
        _warn_default_ua_once()
        return cls(user_agent=DEFAULT_USER_AGENT, using_default=True)


_default_ua_warned: bool = False
_deprecated_creds_warned: set[str] = set()


def _warn_default_ua_once() -> None:
    global _default_ua_warned
    if _default_ua_warned:
        return
    _default_ua_warned = True
    logger.warning(
        "%s not set — using default %r. Reddit may throttle generic UAs; "
        "set %s to '<platform>:<app-id>:<version> (by /u/<your-username>)'.",
        ENV_REDDIT_USER_AGENT,
        DEFAULT_USER_AGENT,
        ENV_REDDIT_USER_AGENT,
    )


def _warn_deprecated_creds_once() -> None:
    for var in DEPRECATED_REDDIT_VARS:
        if var in _deprecated_creds_warned:
            continue
        if os.getenv(var):
            _deprecated_creds_warned.add(var)
            logger.warning(
                "%s detected — no longer used since the praw→.json swap. "
                "Safe to remove from env.",
                var,
            )


def subreddits_from_env() -> list[str]:
    """Parse REDDIT_SUBREDDITS=foo,bar,baz with fallback to ['algotrading']."""
    raw = os.getenv(ENV_REDDIT_SUBREDDITS, "").strip()
    if not raw:
        return list(DEFAULT_SUBREDDITS)
    return [s.strip() for s in raw.split(",") if s.strip()]
