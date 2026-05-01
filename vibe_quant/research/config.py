"""Configuration for research sources."""

from __future__ import annotations

import os
from dataclasses import dataclass

from vibe_quant.alerts.telegram import ConfigurationError

ENV_REDDIT_CLIENT_ID = "REDDIT_CLIENT_ID"
ENV_REDDIT_CLIENT_SECRET = "REDDIT_CLIENT_SECRET"
ENV_REDDIT_USER_AGENT = "REDDIT_USER_AGENT"
ENV_REDDIT_SUBREDDITS = "REDDIT_SUBREDDITS"

DEFAULT_SUBREDDITS = ("algotrading",)


@dataclass(frozen=True)
class RedditConfig:
    """Reddit script-app read-only credentials."""

    client_id: str
    client_secret: str
    user_agent: str

    @classmethod
    def from_env(cls) -> RedditConfig:
        client_id = os.getenv(ENV_REDDIT_CLIENT_ID)
        client_secret = os.getenv(ENV_REDDIT_CLIENT_SECRET)
        user_agent = os.getenv(ENV_REDDIT_USER_AGENT)

        if not client_id:
            raise ConfigurationError(f"Missing {ENV_REDDIT_CLIENT_ID} environment variable")
        if not client_secret:
            raise ConfigurationError(f"Missing {ENV_REDDIT_CLIENT_SECRET} environment variable")
        if not user_agent:
            raise ConfigurationError(f"Missing {ENV_REDDIT_USER_AGENT} environment variable")

        return cls(client_id=client_id, client_secret=client_secret, user_agent=user_agent)


def subreddits_from_env() -> list[str]:
    """Parse REDDIT_SUBREDDITS=foo,bar,baz with fallback to ['algotrading']."""
    raw = os.getenv(ENV_REDDIT_SUBREDDITS, "").strip()
    if not raw:
        return list(DEFAULT_SUBREDDITS)
    return [s.strip() for s in raw.split(",") if s.strip()]
