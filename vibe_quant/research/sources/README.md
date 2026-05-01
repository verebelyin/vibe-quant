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
