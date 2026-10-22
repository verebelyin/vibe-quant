"""The ONE codec for the ``backtest_results.notes`` column (bead vibe-quant-rabmo.8).

Before this module every reader parsed the column by hand with its own legacy
rules (``json.loads`` plus a local ``try/except``). ``parse_notes`` is
``json.loads`` itself -- the SAME exceptions -- because each reader keeps its own
``try/except`` and every legacy failure mode must stay byte for byte.
``notes_mapping`` is the "``{}`` on anything" rule the discovery payload readers
share, and ``dump_notes`` keeps ``json.dumps``'s default separators so the
stored note bytes do not move.

Stdlib only: nothing here imports ``vibe_quant`` (this module sits under the
state layer and is imported by it).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


def parse_notes(text: str | bytes | bytearray) -> object:
    """Parse stored notes text; ``json.loads`` with its own exceptions.

    Readers keep their own ``try/except``: a bad note must fail exactly where it
    failed before this module existed.
    """
    return json.loads(text)


def notes_mapping(value: object) -> dict[str, object]:
    """The notes payload as a mapping, ``{}`` on anything else.

    Only a non-empty ``str`` that parses to a JSON object yields a payload; an
    absent / null / non-string / unparsable / non-object value is an empty run.
    """
    if not value or not isinstance(value, str):
        return {}
    try:
        data = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def dump_notes(payload: Mapping[str, object]) -> str:
    """Serialize notes exactly as ``json.dumps`` does (default separators, key order)."""
    return json.dumps(payload)
