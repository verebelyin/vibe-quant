"""glance-style no-account Reddit access (ToS-grey stopgap).

Reddit shut down unauthenticated ``.json`` access in 2026-05 (see
``docs/reddit-access-research.md``); a plain request now 403s. This ports
glance's workaround (``glanceapp/glance`` ``widget-reddit.go``): impersonate a
real browser's TLS fingerprint via ``curl_cffi``, solve the trivial JS
challenge on ``reddit.com`` to obtain a ``loid`` cookie, then send ordinary
``.json`` requests carrying that cookie + a browser User-Agent.

This is a bridge until the OAuth path (vibe-quant-7go7m.2) is approved. It is
low-volume personal-research use and can break whenever Reddit changes the
challenge; on failure it raises so the scrape run records ``failed`` rather
than silently degrading. The ``loid`` cookie is cached and shared to keep the
challenge-solving request count low, mirroring glance.

The public ``get()`` returns an httpx-compatible response so the caller's
existing retry / ``raise_for_status`` / ``Retry-After`` handling is unchanged.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

HOMEPAGE_URL = "https://www.reddit.com/"
# Firefox because glance uses uTLS HelloFirefox_Auto and curl_cffi's firefox
# impersonation targets the same fingerprint family.
DEFAULT_IMPERSONATE = "firefox"
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0"
)
# The page's inline JS computes the answer as `e + e` (string concat) over the
# challenge literal; these mirror glance's regexes.
_CHALLENGE_RE = re.compile(r'await\(async \w+\s*=>\s*\w+\s*\+\s*\w+\)\("([^"]+)"\)')
_TOKEN_RE = re.compile(r'name="token"\s+value="([^"]+)"')
_LOID_TTL_S = 6 * 60 * 60
_DEFAULT_TIMEOUT_S = 15.0


class RedditChallengeError(RuntimeError):
    """The browser-challenge flow could not produce a usable ``loid`` cookie."""


class ChallengeClient:
    """Duck-compatible drop-in for the ``httpx.Client`` the source calls.

    Only the surface ``RedditSource._request`` needs is implemented:
    ``get(url, params=...) -> response`` (with ``.status_code``, ``.headers``,
    ``.json()``, ``.raise_for_status()``) and ``close()``.
    """

    def __init__(
        self,
        *,
        user_agent: str = BROWSER_USER_AGENT,
        impersonate: str = DEFAULT_IMPERSONATE,
        timeout: float = _DEFAULT_TIMEOUT_S,
        session: Any | None = None,
    ) -> None:
        self._ua = user_agent
        self._timeout = timeout
        self._session = session if session is not None else _new_curl_session(impersonate)
        self._loid: str | None = None
        self._loid_at: float = 0.0
        self._lock = threading.Lock()

    def get(self, url: str, params: Mapping[str, str] | None = None) -> httpx.Response:
        loid = self._ensure_loid()
        self._session.cookies.set("loid", loid, domain=".reddit.com")
        raw = self._session.get(
            url, params=dict(params or {}), headers={"User-Agent": self._ua}, timeout=self._timeout
        )
        return _adapt_response(raw)

    def close(self) -> None:
        self._session.close()

    def _ensure_loid(self, *, force: bool = False) -> str:
        with self._lock:
            fresh = self._loid is not None and (time.monotonic() - self._loid_at) < _LOID_TTL_S
            if not force and fresh and self._loid is not None:
                return self._loid
            self._loid = self._solve_challenge()
            self._loid_at = time.monotonic()
            return self._loid

    def _solve_challenge(self) -> str:
        page = self._session.get(
            HOMEPAGE_URL, headers={"User-Agent": self._ua}, timeout=self._timeout
        )
        if page.status_code != 200:
            raise RedditChallengeError(f"challenge page returned status {page.status_code}")
        challenge = _CHALLENGE_RE.search(page.text)
        token = _TOKEN_RE.search(page.text)
        if challenge is None or token is None:
            raise RedditChallengeError(
                "challenge/token not found on reddit.com — the flow likely changed"
            )
        solution = challenge.group(1) + challenge.group(1)
        submit = self._session.get(
            HOMEPAGE_URL,
            params={"solution": solution, "js_challenge": "1", "token": token.group(1)},
            headers={"User-Agent": self._ua},
            timeout=self._timeout,
        )
        if submit.status_code != 200:
            raise RedditChallengeError(f"challenge submit returned status {submit.status_code}")
        loid = self._session.cookies.get("loid")
        if not loid:
            raise RedditChallengeError("no loid cookie after solving the challenge")
        logger.info("obtained reddit loid cookie via browser challenge")
        return str(loid)


def _new_curl_session(impersonate: str) -> Any:
    try:
        from curl_cffi import requests  # noqa: PLC0415 — optional heavy dep, import on use
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RedditChallengeError(
            "curl_cffi is required for challenge-based Reddit access "
            "(pip install curl_cffi); or disable REDDIT_USE_CHALLENGE"
        ) from exc
    # curl_cffi's stub types impersonate as a Literal of browser names; our
    # value ("firefox") is valid but arrives as a plain str.
    return requests.Session(impersonate=impersonate)  # type: ignore[arg-type]


def _adapt_response(raw: Any) -> httpx.Response:
    """Wrap a curl_cffi response as an ``httpx.Response`` so the caller's
    httpx-based error handling (``raise_for_status``, ``Retry-After``) works
    uniformly across the OAuth/httpx path and this one.

    curl_cffi has already decompressed the body, so the content-encoding /
    content-length headers must be dropped or httpx would try to decode the
    plaintext again.
    """
    headers = {
        k: v
        for k, v in dict(raw.headers).items()
        if k.lower() not in {"content-encoding", "content-length"}
    }
    return httpx.Response(
        status_code=int(raw.status_code),
        headers=headers,
        content=raw.content,
        request=httpx.Request("GET", str(raw.url)),
    )
