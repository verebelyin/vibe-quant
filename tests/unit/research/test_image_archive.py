"""Image archiver unit tests — mocked httpx transport, no network."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx

from vibe_quant.research.image_archive import (
    BROWSER_USER_AGENT,
    archive_item_images,
    download_images,
)
from vibe_quant.research.schema import RawItem

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def _png(nbytes: int = 32) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * nbytes


def test_downloads_images_to_per_item_dir(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_png(), headers={"content-type": "image/png"})

    paths = download_images(
        source="reddit",
        external_id="abc123",
        image_urls=["https://i.redd.it/a.png", "https://i.redd.it/b.png"],
        root=tmp_path,
        client=_client(handler),
    )

    assert len(paths) == 2
    expected_dir = tmp_path / "reddit" / "abc123"
    assert {p for p in paths} == {str(expected_dir / "0.png"), str(expected_dir / "1.png")}
    for p in paths:
        from pathlib import Path as _P

        assert _P(p).exists()
        assert _P(p).is_absolute()


def test_extension_falls_back_to_content_type(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        # URL has no image suffix; content-type drives the extension.
        return httpx.Response(200, content=_png(), headers={"content-type": "image/webp"})

    paths = download_images(
        source="reddit",
        external_id="ct",
        image_urls=["https://i.redd.it/noext?query=1"],
        root=tmp_path,
        client=_client(handler),
    )
    assert len(paths) == 1
    assert paths[0].endswith("0.webp")


def test_size_cap_aborts_and_drops_oversize_file(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_png(10_000), headers={"content-type": "image/png"})

    paths = download_images(
        source="reddit",
        external_id="big",
        image_urls=["https://i.redd.it/huge.png"],
        root=tmp_path,
        client=_client(handler),
        max_bytes=1024,
    )
    assert paths == []
    # No partial file left behind.
    assert not (tmp_path / "reddit" / "big" / "0.png").exists()


def test_count_cap_limits_downloads(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_png(), headers={"content-type": "image/png"})

    urls = [f"https://i.redd.it/{i}.png" for i in range(6)]
    paths = download_images(
        source="reddit",
        external_id="many",
        image_urls=urls,
        root=tmp_path,
        client=_client(handler),
        max_images=3,
    )
    assert len(paths) == 3


def test_failing_url_skipped_others_saved(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if "bad" in req.url.path:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, content=_png(), headers={"content-type": "image/png"})

    paths = download_images(
        source="reddit",
        external_id="mixed",
        image_urls=[
            "https://i.redd.it/good1.png",
            "https://i.redd.it/bad.png",
            "https://i.redd.it/good2.png",
        ],
        root=tmp_path,
        client=_client(handler),
    )
    # Bad one skipped; the two good ones land with contiguous indices.
    assert len(paths) == 2
    assert paths[0].endswith("0.png")
    assert paths[1].endswith("1.png")


def test_reddit_403_retries_with_browser_user_agent(tmp_path: Path) -> None:
    seen_uas: list[str | None] = []

    def handler(req: httpx.Request) -> httpx.Response:
        ua = req.headers.get("user-agent")
        seen_uas.append(ua)
        # First attempt (no/other UA) → 403; browser-UA retry → 200.
        if ua == BROWSER_USER_AGENT:
            return httpx.Response(200, content=_png(), headers={"content-type": "image/png"})
        return httpx.Response(403, text="blocked")

    paths = download_images(
        source="reddit",
        external_id="challenge",
        image_urls=["https://i.redd.it/gated.png"],
        root=tmp_path,
        client=_client(handler),
    )
    assert len(paths) == 1
    assert BROWSER_USER_AGENT in seen_uas
    assert len(seen_uas) == 2  # one 403, one browser-UA retry


def test_non_reddit_403_not_retried(tmp_path: Path) -> None:
    attempts = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(403, text="blocked")

    paths = download_images(
        source="reddit",
        external_id="ext403",
        image_urls=["https://example.com/x.png"],
        root=tmp_path,
        client=_client(handler),
    )
    assert paths == []
    assert attempts["n"] == 1  # no browser-UA retry for a non-redd.it host


def test_archive_item_images_mutates_extras(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_png(), headers={"content-type": "image/png"})

    item = RawItem(
        source="reddit",
        external_id="itm",
        url="https://reddit.com/x",
        title="t",
        body="b",
        author=None,
        posted_at=datetime(2026, 5, 1, tzinfo=UTC),
        score=1,
        extras={"image_urls": ["https://i.redd.it/a.png"]},
    )
    paths = archive_item_images(item, root=tmp_path, client=_client(handler))
    assert len(paths) == 1
    assert item.extras["image_paths"] == paths


def test_archive_item_images_no_urls_is_noop(tmp_path: Path) -> None:
    item = RawItem(
        source="reddit",
        external_id="noimg",
        url="https://reddit.com/x",
        title="t",
        body="b",
        author=None,
        posted_at=None,
        score=None,
        extras={"comments": []},
    )
    paths = archive_item_images(item, root=tmp_path)
    assert paths == []
    assert "image_paths" not in item.extras


def test_untrusted_external_id_cannot_escape_archive_root(tmp_path: Path) -> None:
    """A crafted source/external_id must not write outside the archive root."""

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_png(), headers={"Content-Type": "image/png"})

    saved = download_images(
        source="reddit",
        external_id="../../escape",
        image_urls=["https://i.redd.it/a.png"],
        root=tmp_path,
        client=_client(handler),
    )

    assert len(saved) == 1
    resolved = saved[0]
    # The written file stays inside the archive root — no traversal.
    assert resolved.startswith(str(tmp_path.resolve()) + "/")
