"""External research pipeline.

Scrapes ideas from public sources (Reddit, arxiv, X, ...), runs LLM extraction
into the strategy DSL, and surfaces them as triageable candidates that a human
can promote into the existing strategies table.

See `sources/README.md` for how to add a new source.
"""

from __future__ import annotations
