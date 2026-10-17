# Research source plugins

Drop a `.py` file in this directory. Define a class implementing the
`ResearchSource` protocol and decorate it with `@register_source("name")`.
The source is auto-discovered at startup via `load_builtin_sources()` and
exposed under its registered name through `get_source(name)` /
`list_sources()`.

## Protocol

```python
class ResearchSource(Protocol):
    name: ClassVar[str]                                     # registry key
    def fetch(
        self,
        since: datetime | None,
        limit: int,
    ) -> Iterable[RawItem]: ...
```

`limit` is a **per source unit** cap, not a run total: `hackernews` applies it
per configured query (5 queries × limit 3 can yield up to 15 stories, deduped),
`reddit` per configured subreddit. A run can therefore fetch several times
`limit` items across its queries/subreddits.

`RawItem` (from `vibe_quant.research.schema`) is the canonical wire format:

| Field         | Required | What goes here                                        |
|---------------|----------|-------------------------------------------------------|
| `source`      | yes      | The same string you passed to `@register_source`.     |
| `external_id` | yes      | Source-side primary key (Reddit submission id, ...).  |
| `url`         | yes      | Permalink the user clicks to open the original.       |
| `title`       | yes      | One-line headline. Empty string if the source has none.|
| `body`        | yes      | Markdown text body. Empty string for link-only posts. |
| `author`      | no       | `None` if deleted/anonymous (NOT the string "[deleted]"). |
| `posted_at`   | no       | Aware UTC `datetime` of original publication.         |
| `score`       | no       | Signed integer (upvotes/likes). `None` if N/A.        |
| `extras`      | no       | Source-specific JSON (comments, flair, categories).   |

`extras` is the right home for anything that doesn't fit one of the columns —
the archive serialises it to JSON. Common keys we use today:

- `comments: list[dict]` — Reddit/HN: `{author, body, score}` per comment.
- `image_urls: list[str]` — Reddit: image URLs on the post (top level only).
- `via: str` — which backend served the item (e.g. `arctic_shift`).

## Reddit source (Arctic Shift)

`reddit.py` reads a subreddit's settled posts and their top comments from the
free [Arctic Shift](https://arctic-shift.photon-reddit.com) API
(`/api/posts/search`, `/api/comments/search`) — no key and no browser spoofing.
Politeness floor: one request every 1.5 s (`MIN_REQUEST_INTERVAL_S`); a 429 is
honoured via `X-RateLimit-Reset`. Freshness: a post's score and comment count
settle ~36 h after it lands, so the source asks only for posts older than
`SETTLE_DELAY_S` (36 h, via `before=`) and gets settled numbers in one pass.

## Minimal example

```python
# vibe_quant/research/sources/arxiv.py
from __future__ import annotations
from collections.abc import Iterable
from datetime import datetime

from vibe_quant.research.schema import RawItem
from vibe_quant.research.sources import register_source


@register_source("arxiv")
class ArxivSource:
    name = "arxiv"

    def __init__(self, categories: list[str]) -> None:
        self._categories = categories

    def fetch(self, since: datetime | None, limit: int) -> Iterable[RawItem]:
        # ... query the arxiv API, yield RawItems
        yield RawItem(
            source="arxiv",
            external_id="2401.12345",
            url="https://arxiv.org/abs/2401.12345",
            title="Some paper",
            body="Abstract...",
            author="Doe, J.",
            posted_at=None,
            score=None,
            extras={"categories": ["q-fin.ST"]},
        )
```

## Discovery rules

- Files starting with `_` are skipped (treat them as private helpers).
- A failing import is logged + recorded in `get_load_errors()` — other
  sources still load. Set `VQ_PLUGINS_STRICT=1` if you want CI to re-raise.
- Duplicate `register_source(name=...)` raises `ValueError` at import time.
- A class without a `fetch` method raises `TypeError` at registration time.

## Local test loop

```bash
python -c "from vibe_quant.research.sources import load_builtin_sources, list_sources; \
           load_builtin_sources(force=True); print(list_sources())"
```

## Built-in: Hacker News (`hackernews`)

`hackernews.py` searches the keyless Algolia HN API
(`https://hn.algolia.com/api/v1/search_by_date`) once per configured query with
`tags=story`, dedupes stories by `objectID`, and fetches up to 10 top-level
comments per story from `items/<id>` (HTML stripped to text). Queries come from
the comma-separated `HN_QUERIES` env var (default: `algorithmic trading`,
`trading strategy`, `backtest`, `quant trading`, `crypto trading`), or the
constructor's `queries=` argument. Requests are spaced 1s apart and retried with
backoff on 429/5xx/timeouts; a query that still fails is logged and skipped, and
all-query failure raises `RuntimeError` (so the scrape run records `failed`).
Story URLs fall back to the HN item permalink when a story has no external URL;
`extras` carries `{query, num_comments, comments, hn_url}`.
