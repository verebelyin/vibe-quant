"""Download + archive research-post images so the extractor can read them.

Reddit strategy posts often put the real content in screenshots. The scraper
captures their URLs (see ``sources/reddit.py``); this module streams them to
``data/archive/research_images/<source>/<external_id>/<idx>.<ext>`` so the
vision-capable ``claude -p`` extractor can ``Read`` them off disk.

Design constraints (mirror the data-archive conventions — the archive is
rebuildable runtime data, never committed):

* per-file size cap, enforced while streaming (abort + drop a too-big file
  before it lands),
* total count cap so a gallery can't balloon the archive,
* per-file timeout,
* skip-on-failure — one bad URL must never fail the whole item,
* a browser User-Agent retry when a redd.it CDN 403s a plain request (mirrors
  the challenge client's UA).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx

from vibe_quant.research.sources._reddit_challenge import BROWSER_USER_AGENT

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vibe_quant.research.schema import RawItem

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_ROOT = Path("data/archive/research_images")
MAX_IMAGES = 4
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB per file
DOWNLOAD_TIMEOUT_S = 20.0
_CHUNK_SIZE = 65536

_ALLOWED_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif")
_EXT_BY_CONTENT_TYPE = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
_REDDIT_CDN_HOSTS = ("redd.it",)


def archive_item_images(
    item: RawItem,
    *,
    root: Path | None = None,
    client: httpx.Client | None = None,
) -> list[str]:
    """Download an item's ``extras['image_urls']`` and record local paths.

    Mutates ``item.extras['image_paths']`` in place (the frozen ``RawItem``
    keeps a mutable ``extras`` dict) so the archiver persists the local paths
    to ``extras_json``. Returns the list of saved absolute paths (empty when
    the item has no image URLs or every download failed).
    """
    extras = item.extras
    if not isinstance(extras, dict):
        return []
    raw_urls = extras.get("image_urls")
    if not isinstance(raw_urls, list) or not raw_urls:
        return []
    urls = [u for u in raw_urls if isinstance(u, str) and u]
    paths = download_images(
        source=item.source,
        external_id=item.external_id,
        image_urls=urls,
        root=root,
        client=client,
    )
    if paths:
        extras["image_paths"] = paths
    return paths


def download_images(
    *,
    source: str,
    external_id: str,
    image_urls: Sequence[str],
    root: Path | None = None,
    client: httpx.Client | None = None,
    max_images: int = MAX_IMAGES,
    max_bytes: int = MAX_IMAGE_BYTES,
) -> list[str]:
    """Stream up to ``max_images`` URLs to the per-item archive directory.

    Returns the absolute paths of the files that landed. Failures (HTTP
    errors, oversize bodies, timeouts) are logged and skipped so a single bad
    URL never sinks the item.
    """
    if not image_urls:
        return []
    # source/external_id come from untrusted upstream JSON and flow into a
    # filesystem write path — sanitise so a crafted id can't traverse out of
    # the archive root.
    dest_dir = (root or DEFAULT_IMAGE_ROOT) / _safe_segment(source) / _safe_segment(external_id)
    owns_client = client is None
    if client is None:
        client = httpx.Client(timeout=DOWNLOAD_TIMEOUT_S, follow_redirects=True)
    saved: list[str] = []
    try:
        for url in list(image_urls)[:max_images]:
            path = _download_one(client, url, dest_dir, len(saved), max_bytes)
            if path is not None:
                saved.append(str(path))
    finally:
        if owns_client:
            client.close()
    return saved


def _safe_segment(value: str) -> str:
    """Reduce an untrusted id to a single safe path segment.

    Keeps alnum, dash, underscore, and dot; collapses everything else to '_'.
    Guards against empty / dot-only results (which would traverse or collide).
    """
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in value)
    cleaned = cleaned.strip(".")
    return cleaned or "unknown"


def _download_one(
    client: httpx.Client, url: str, dest_dir: Path, idx: int, max_bytes: int
) -> Path | None:
    data_ext = _fetch(client, url, max_bytes)
    if data_ext is None:
        return None
    data, ext = data_ext
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = (dest_dir / f"{idx}{ext}").resolve()
    dest.write_bytes(data)
    return dest


def _fetch(client: httpx.Client, url: str, max_bytes: int) -> tuple[bytes, str] | None:
    """Fetch one URL, retrying redd.it 403s with a browser User-Agent.

    Returns ``(bytes, extension)`` or ``None`` on any failure / oversize body.
    """
    header_attempts: list[dict[str, str]] = [{}]
    if _is_reddit_cdn(url):
        header_attempts.append({"User-Agent": BROWSER_USER_AGENT})
    for i, headers in enumerate(header_attempts):
        try:
            with client.stream("GET", url, headers=headers) as resp:
                if resp.status_code == 403 and i + 1 < len(header_attempts):
                    continue  # retry with the next (browser-UA) attempt
                resp.raise_for_status()
                body = _read_capped(resp, max_bytes)
                if body is None:
                    logger.warning("image %s exceeds %d bytes — skipping", url, max_bytes)
                    return None
                return body, _resolve_ext(url, resp.headers.get("content-type"))
        except httpx.HTTPError as e:
            logger.warning("image download failed for %s: %s", url, e)
            return None
    return None


def _read_capped(resp: httpx.Response, max_bytes: int) -> bytes | None:
    """Accumulate the streamed body, aborting the moment it exceeds the cap."""
    data = bytearray()
    for chunk in resp.iter_bytes(_CHUNK_SIZE):
        data.extend(chunk)
        if len(data) > max_bytes:
            return None
    return bytes(data)


def _is_reddit_cdn(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == h or host.endswith("." + h) for h in _REDDIT_CDN_HOSTS)


def _resolve_ext(url: str, content_type: str | None) -> str:
    path = urlparse(url).path.lower()
    for ext in _ALLOWED_EXTS:
        if path.endswith(ext):
            return ext
    if content_type:
        ct = content_type.split(";", 1)[0].strip().lower()
        if ct in _EXT_BY_CONTENT_TYPE:
            return _EXT_BY_CONTENT_TYPE[ct]
    return ".jpg"
